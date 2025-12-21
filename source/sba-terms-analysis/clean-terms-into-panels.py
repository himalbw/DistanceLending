import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# Paths
# ============================================================
BASE_DIR = Path("source/sba_analysis/data")
RAW_DIR = BASE_DIR / "raw"
DERIVED_DIR = BASE_DIR / "derived"

IN_PATH = DERIVED_DIR / "sba_7a_loan_to_branch_with_terms.csv"
SOD_WITH_FLAGS_PATH = DERIVED_DIR / "sod_with_all_exog_flags.csv"
COUNTY_SHP = RAW_DIR / "cb_2024_us_county_500k.shp"

OUT_BCY_PATH = DERIVED_DIR / "sba_bcy_terms.csv"
OUT_CY_PATH  = DERIVED_DIR / "sba_cy_terms.csv"

TERMS_ANALYSIS_DIR = Path("source/sba-terms-analysis/data")
OUT_BCY_COPY = TERMS_ANALYSIS_DIR / "sba_bcy_terms.csv"
OUT_CY_COPY  = TERMS_ANALYSIS_DIR / "sba_cy_terms.csv"


# ============================================================
# Column names
# ============================================================
BANK_COL = "loan_cert"
BRANCH_COL = "branch_id"
YEAR_COL = "year"
COUNTY_COL = "borrower_county"

LAT_COL = "borrower_lat"
LON_COL = "borrower_lon"

# SoD fields
BINARY_FLAG_COLS = ["closure", "exog_closure", "true_exog_closure"]
CATEG_FLAG_COLS  = ["transformation_flag", "trans_role"]  # keep as strings

ALL_FLAG_COLS = CATEG_FLAG_COLS + BINARY_FLAG_COLS


# ============================================================
# Main
# ============================================================
def main() -> None:
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    TERMS_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading loans: {IN_PATH}")
    loans = pd.read_csv(IN_PATH, low_memory=False)

    print(f"Loading SoD: {SOD_WITH_FLAGS_PATH}")
    sod = pd.read_csv(SOD_WITH_FLAGS_PATH, low_memory=False)

    sod_branch_geo = build_sod_branch_geo(sod)
    sod_flags = build_sod_flags(sod)

    print("Attaching borrower county via shapefile (with SoD fallback)...")
    loans = attach_borrower_county_spatial(loans, COUNTY_SHP, sod_branch_geo)

    print("Attaching SoD flags and ASSET by (branch_id, year)...")
    loans = loans.merge(sod_flags, on=[BRANCH_COL, YEAR_COL], how="left")

    # Safe flag coercion
    for c in BINARY_FLAG_COLS:
        if c in loans.columns:
            loans[c] = pd.to_numeric(loans[c], errors="coerce").fillna(0).astype("int64")
        else:
            loans[c] = 0

    for c in CATEG_FLAG_COLS:
        if c in loans.columns:
            loans[c] = loans[c].astype("string")
        else:
            loans[c] = pd.Series([pd.NA] * len(loans), dtype="string")

    # County must be present now
    if COUNTY_COL not in loans.columns or loans[COUNTY_COL].isna().all():
        raise ValueError("borrower_county is still missing after spatial join + SoD fallback.")

    # Normalize county FIPS to 5-digit strings
    loans[COUNTY_COL] = loans[COUNTY_COL].astype("string").str.zfill(5)

    # Collapse
    print("Collapsing to BCY...")
    bcy = collapse_to_bcy_terms(loans)

    print("Collapsing BCY -> CY...")
    cy = collapse_bcy_to_cy_terms(bcy)

    # Save
    print(f"Saving BCY -> {OUT_BCY_PATH}")
    bcy.to_csv(OUT_BCY_PATH, index=False)

    print(f"Saving CY  -> {OUT_CY_PATH}")
    cy.to_csv(OUT_CY_PATH, index=False)

    print(f"Copy BCY -> {OUT_BCY_COPY}")
    bcy.to_csv(OUT_BCY_COPY, index=False)

    print(f"Copy CY  -> {OUT_CY_COPY}")
    cy.to_csv(OUT_CY_COPY, index=False)

    print("Done.")


# ============================================================
# SoD helpers
# ============================================================
def build_sod_branch_geo(sod: pd.DataFrame) -> pd.DataFrame:
    """
    (branch_id, year) -> STCNTYBR (5-digit county FIPS)
    """
    required = ["UNINUMBR", "YEAR", "STCNTYBR"]
    missing = [c for c in required if c not in sod.columns]
    if missing:
        raise ValueError(f"SoD missing required columns for branch geo: {missing}")

    out = (
        sod[required]
        .drop_duplicates()
        .rename(columns={"UNINUMBR": BRANCH_COL, "YEAR": YEAR_COL})
    )

    # STCNTYBR can be numeric; force 5-digit string
    out["STCNTYBR"] = out["STCNTYBR"].apply(
        lambda x: str(int(x)).zfill(5) if pd.notna(x) and str(x).strip() != "" else pd.NA
    )

    return out


def build_sod_flags(sod: pd.DataFrame) -> pd.DataFrame:
    """
    (branch_id, year) -> flags + ASSET (if available)
    """
    base_cols = ["UNINUMBR", "YEAR"]
    missing = [c for c in base_cols if c not in sod.columns]
    if missing:
        raise ValueError(f"SoD missing required keys: {missing}")

    keep = base_cols + [c for c in ALL_FLAG_COLS if c in sod.columns]
    if "ASSET" in sod.columns:
        keep.append("ASSET")

    out = (
        sod[keep]
        .drop_duplicates(subset=base_cols)
        .rename(columns={"UNINUMBR": BRANCH_COL, "YEAR": YEAR_COL})
    )
    return out


# ============================================================
# County attach (spatial join + fallback)
# ============================================================
def attach_borrower_county_spatial(
    loans: pd.DataFrame,
    county_shp_path: Path,
    sod_branch_geo: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach borrower_county by spatially joining borrower lat/lon to county polygons.
    Fallback: if borrower county still missing, impute from SoD STCNTYBR by (branch_id, year).
    """
    try:
        import geopandas as gpd
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "geopandas is required to attach borrower counties from the shapefile.\n\n"
            "Recommended install:\n"
            "  conda install -c conda-forge geopandas\n"
            "or pip:\n"
            "  pip install geopandas pyogrio shapely\n"
        ) from e

    df = loans.copy()

    if LAT_COL not in df.columns or LON_COL not in df.columns:
        raise ValueError(f"Loans missing lat/lon columns: {LAT_COL}, {LON_COL}")

    # Keep only rows with valid coordinates for the spatial join
    coord_ok = df[LAT_COL].notna() & df[LON_COL].notna()

    print(f"  Spatial join on {coord_ok.sum():,} rows with non-missing coords (out of {len(df):,}).")

    counties = gpd.read_file(county_shp_path).to_crs("EPSG:4326")
    if "GEOID" not in counties.columns:
        raise ValueError("County shapefile does not have GEOID column; check schema.")

    counties = counties[["GEOID", "geometry"]]

    pts = gpd.GeoDataFrame(
        df.loc[coord_ok].copy(),
        geometry=gpd.points_from_xy(df.loc[coord_ok, LON_COL], df.loc[coord_ok, LAT_COL]),
        crs="EPSG:4326",
    )

    joined = gpd.sjoin(pts, counties, how="left", predicate="within")
    df.loc[coord_ok, COUNTY_COL] = joined["GEOID"].astype("string").values

    # Fallback: impute missing borrower county from branch geo
    df = df.merge(
        sod_branch_geo[[BRANCH_COL, YEAR_COL, "STCNTYBR"]],
        on=[BRANCH_COL, YEAR_COL],
        how="left",
    )

    missing_cty = df[COUNTY_COL].isna()
    df.loc[missing_cty, COUNTY_COL] = df.loc[missing_cty, "STCNTYBR"]

    df = df.drop(columns=["STCNTYBR"], errors="ignore")

    # Final normalize to 5-digit string where possible
    df[COUNTY_COL] = df[COUNTY_COL].astype("string").str.zfill(5)

    print(f"  borrower_county missing after join+fallback: {df[COUNTY_COL].isna().sum():,}")

    return df


# ============================================================
# Collapse: loan -> BCY
# ============================================================
def collapse_to_bcy_terms(loans: pd.DataFrame) -> pd.DataFrame:
    df = loans.copy()

    # --- Normalize status; comparisons can yield <NA> so use fillna(False) ---
    status = (
        df.get("LoanStatus")
        .astype("string")
        .str.upper()
        .str.strip()
    )

    df["is_chgoff"] = np.where(status.eq("CHGOFF").fillna(False), 1, 0).astype("int64")
    df["is_pif"]    = np.where(status.eq("PIF").fillna(False),    1, 0).astype("int64")
    df["is_cancld"] = np.where(status.eq("CANCLD").fillna(False), 1, 0).astype("int64")

    # --- Terms ---
    df["interest_rate"] = pd.to_numeric(df.get("InitialInterestRate"), errors="coerce")
    df["term_months"]   = pd.to_numeric(df.get("TerminMonths"), errors="coerce")

    rate_type = df.get("FixedorVariableInterestRate").astype("string").str.upper().str.strip()
    df["is_fixed"]    = np.where(rate_type.eq("F").fillna(False), 1, 0).astype("int64")
    df["is_variable"] = np.where(rate_type.eq("V").fillna(False), 1, 0).astype("int64")

    coll = df.get("CollateralInd").astype("string").str.upper().str.strip()
    df["is_collateral"] = np.where(coll.eq("Y").fillna(False), 1, 0).astype("int64")

    df["is_revolver"] = pd.to_numeric(df.get("RevolverStatus"), errors="coerce").fillna(0)
    df["is_revolver"] = np.where(df["is_revolver"] == 1, 1, 0).astype("int64")

    # --- Ensure numeric for sums ---
    df["GrossApproval"] = pd.to_numeric(df.get("GrossApproval"), errors="coerce")
    df["SBAGuaranteedApproval"] = pd.to_numeric(df.get("SBAGuaranteedApproval"), errors="coerce")
    df["distance_miles"] = pd.to_numeric(df.get("distance_miles"), errors="coerce")

    keys = [BANK_COL, COUNTY_COL, YEAR_COL]

    bcy = (
        df.groupby(keys, as_index=False)
        .agg(
            n_loans=("LoanStatus", "size"),
            total_gross_approval=("GrossApproval", "sum"),
            total_sba_guaranteed=("SBAGuaranteedApproval", "sum"),
            avg_distance_miles=("distance_miles", "mean"),
            med_distance_miles=("distance_miles", "median"),

            n_chgoff=("is_chgoff", "sum"),
            n_pif=("is_pif", "sum"),
            n_cancld=("is_cancld", "sum"),

            interest_rate_mean=("interest_rate", "mean"),
            interest_rate_p50=("interest_rate", "median"),
            term_months_mean=("term_months", "mean"),
            term_months_p50=("term_months", "median"),

            fixed_share=("is_fixed", "mean"),
            variable_share=("is_variable", "mean"),
            collateral_share=("is_collateral", "mean"),
            revolver_share=("is_revolver", "mean"),
        )
    )

    bcy["chgoff_rate_all"] = np.where(
        bcy["n_loans"] > 0, bcy["n_chgoff"] / bcy["n_loans"], np.nan
    )
    bcy["n_resolved"] = bcy["n_chgoff"] + bcy["n_pif"]
    bcy["chgoff_rate_resolved"] = np.where(
        bcy["n_resolved"] > 0, bcy["n_chgoff"] / bcy["n_resolved"], np.nan
    )

    # --- Flags: max within BCY (guarantee existence) ---
    for c in BINARY_FLAG_COLS:
        if c in df.columns:
            tmp = df.groupby(keys, as_index=False)[c].max()
            bcy = bcy.merge(tmp, on=keys, how="left")
        else:
            bcy[c] = 0

    # --- Keep ASSET mean if present (thousands of dollars, consistent with your SoD) ---
    if "ASSET" in df.columns:
        asset = (
            df.groupby(keys, as_index=False)["ASSET"]
              .mean()
              .rename(columns={"ASSET": "ASSET_mean"})
        )
        bcy = bcy.merge(asset, on=keys, how="left")

    return bcy


# ============================================================
# Collapse: BCY -> CY (safe + clean)
# ============================================================
def collapse_bcy_to_cy_terms(bcy: pd.DataFrame) -> pd.DataFrame:
    keys_cy = [COUNTY_COL, YEAR_COL]
    df = bcy.copy()

    # Force numeric where needed
    num_cols = [
        "n_loans", "total_gross_approval", "total_sba_guaranteed",
        "n_chgoff", "n_pif", "n_cancld",
        "avg_distance_miles", "interest_rate_mean", "term_months_mean",
        "fixed_share", "variable_share", "collateral_share", "revolver_share",
        "med_distance_miles", "interest_rate_p50", "term_months_p50",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Ensure flags exist and are 0/1 ints
    for c in BINARY_FLAG_COLS:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int64")

    # Base CY aggregation
    cy = (
        df.groupby(keys_cy, as_index=False)
        .agg(
            n_banks=(BANK_COL, "nunique"),
            n_loans=("n_loans", "sum"),
            total_gross_approval=("total_gross_approval", "sum"),
            total_sba_guaranteed=("total_sba_guaranteed", "sum"),
            n_chgoff=("n_chgoff", "sum"),
            n_pif=("n_pif", "sum"),
            n_cancld=("n_cancld", "sum"),
            closure=("closure", "max"),
            exog_closure=("exog_closure", "max"),
            true_exog_closure=("true_exog_closure", "max"),
        )
    )

    cy["chgoff_rate_all"] = np.where(
        cy["n_loans"] > 0, cy["n_chgoff"] / cy["n_loans"], np.nan
    )
    cy["n_resolved"] = cy["n_chgoff"] + cy["n_pif"]
    cy["chgoff_rate_resolved"] = np.where(
        cy["n_resolved"] > 0, cy["n_chgoff"] / cy["n_resolved"], np.nan
    )

    # Loan-weighted means from BCY means
    tmp = df.copy()
    w = tmp["n_loans"].to_numpy(dtype=float, copy=False)

    # Weighted sums (handle NaNs safely)
    def wmean(series: pd.Series) -> pd.Series:
        x = series.to_numpy(dtype=float, copy=False)
        num = np.nansum(x * w)
        den = np.nansum(w * np.isfinite(x))
        return num / den if den > 0 else np.nan

    wrows = []
    for (cty, yr), g in df.groupby(keys_cy):
        ww = g["n_loans"].to_numpy(dtype=float, copy=False)

        def wavg(col: str) -> float:
            x = g[col].to_numpy(dtype=float, copy=False)
            m = np.isfinite(x) & np.isfinite(ww) & (ww > 0)
            if not np.any(m):
                return np.nan
            return float(np.sum(x[m] * ww[m]) / np.sum(ww[m]))

        wrows.append({
            COUNTY_COL: cty,
            YEAR_COL: yr,
            "avg_distance_miles": wavg("avg_distance_miles"),
            "interest_rate_mean_w": wavg("interest_rate_mean"),
            "term_months_mean_w": wavg("term_months_mean"),
            "fixed_share_w": wavg("fixed_share"),
            "variable_share_w": wavg("variable_share"),
            "collateral_share_w": wavg("collateral_share"),
            "revolver_share_w": wavg("revolver_share"),
            # optional: unweighted medians from BCY medians
            "med_distance_miles": float(np.nanmedian(g["med_distance_miles"])) if "med_distance_miles" in g else np.nan,
            "interest_rate_p50": float(np.nanmedian(g["interest_rate_p50"])) if "interest_rate_p50" in g else np.nan,
            "term_months_p50": float(np.nanmedian(g["term_months_p50"])) if "term_months_p50" in g else np.nan,
        })

    wmeans = pd.DataFrame(wrows)

    cy = cy.merge(wmeans, on=keys_cy, how="left")

    return cy


if __name__ == "__main__":
    main()
