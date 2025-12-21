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
BANK_COL   = "loan_cert"
BRANCH_COL = "branch_id"
YEAR_COL   = "year"
COUNTY_COL = "borrower_county"
LAT_COL    = "borrower_lat"
LON_COL    = "borrower_lon"

DEPOSIT_COL = "DEPSUMBR"   # SoD deposits
BINARY_FLAG_COLS = ["closure", "exog_closure", "true_exog_closure"]
CATEG_FLAG_COLS  = ["transformation_flag", "trans_role"]

def resolve_deposit_col(df: pd.DataFrame, preferred: str = "DEPSUMBR") -> str:
    """
    Return the actual deposit column name in df.
    Handles common merge suffixes and stray whitespace.
    """
    # normalize whitespace in column names
    df.columns = [c.strip() for c in df.columns]

    # exact match
    if preferred in df.columns:
        return preferred

    # common suffixes after merges
    candidates = [f"{preferred}_x", f"{preferred}_y", preferred.lower(), preferred.upper()]
    for c in candidates:
        if c in df.columns:
            return c

    # substring search
    hits = [c for c in df.columns if preferred in c]
    if len(hits) == 1:
        return hits[0]

    raise KeyError(
        f"Could not find a deposits column like '{preferred}' in df columns.\n"
        f"Closest hits: {hits[:10]}"
    )

def main() -> None:
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    TERMS_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading loans: {IN_PATH}")
    loans = pd.read_csv(IN_PATH, low_memory=False)

    print(f"Loading SoD: {SOD_WITH_FLAGS_PATH}")
    sod = pd.read_csv(SOD_WITH_FLAGS_PATH, low_memory=False)

    # (branch_id, year) -> STCNTYBR for fallback
    sod_branch_geo = build_sod_branch_geo(sod)

    # attach borrower_county correctly (spatial join; fallback to branch county if needed)
    loans = attach_borrower_county_spatial(loans, COUNTY_SHP, sod_branch_geo)

    # attach SoD flags + ASSET + DEPSUMBR
    loans = attach_sod_extras(loans, sod)

    # compute market share at BCY (loan share + deposit share)
    mkt_bcy = compute_bank_market_share_bcy(
        loans,
        loan_amount_col="GrossApproval",
        deposit_col=DEPOSIT_COL,
    )

    # collapse to BCY terms
    bcy = collapse_to_bcy_terms(loans)

    # merge market share into BCY
    keys = [BANK_COL, COUNTY_COL, YEAR_COL]
    bcy = bcy.merge(mkt_bcy, on=keys, how="left")
    for c in ["bank_loan_mkt_share", "bank_dep_mkt_share"]:
        if c in bcy.columns:
            bcy[c] = pd.to_numeric(bcy[c], errors="coerce").fillna(0.0)

    # collapse to CY (and include sums of bank shares as sanity stats)
    cy = collapse_bcy_to_cy_terms(bcy)

    # save
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
    req = ["UNINUMBR", "YEAR", "STCNTYBR"]
    miss = [c for c in req if c not in sod.columns]
    if miss:
        raise ValueError(f"SoD missing required columns for branch geo: {miss}")

    out = (
        sod[req]
        .drop_duplicates()
        .rename(columns={"UNINUMBR": BRANCH_COL, "YEAR": YEAR_COL})
    )

    out["STCNTYBR"] = out["STCNTYBR"].apply(
        lambda x: str(int(x)).zfill(5) if pd.notna(x) and str(x).strip() != "" else pd.NA
    )

    return out

def attach_sod_extras(loans: pd.DataFrame, sod: pd.DataFrame) -> pd.DataFrame:
    keep = ["UNINUMBR", "YEAR"]

    for c in BINARY_FLAG_COLS + CATEG_FLAG_COLS:
        if c in sod.columns:
            keep.append(c)

    if "ASSET" in sod.columns:
        keep.append("ASSET")
    if DEPOSIT_COL in sod.columns:
        keep.append(DEPOSIT_COL)

    sod_sub = (
        sod[keep]
        .drop_duplicates(subset=["UNINUMBR", "YEAR"])
        .rename(columns={"UNINUMBR": BRANCH_COL, "YEAR": YEAR_COL})
    )

    out = loans.merge(sod_sub, on=[BRANCH_COL, YEAR_COL], how="left")

    # if DEPSUMBR got suffixed, unify it to DEPSUMBR
    out.columns = [c.strip() for c in out.columns]
    if DEPOSIT_COL not in out.columns:
        # look for suffix versions and rename
        for c in [f"{DEPOSIT_COL}_x", f"{DEPOSIT_COL}_y"]:
            if c in out.columns:
                out = out.rename(columns={c: DEPOSIT_COL})
                break

    # safe casts
    for c in BINARY_FLAG_COLS:
        out[c] = pd.to_numeric(out.get(c), errors="coerce").fillna(0).astype("int64")

    for c in CATEG_FLAG_COLS:
        out[c] = out.get(c, pd.Series([pd.NA] * len(out))).astype("string")

    return out

# ============================================================
# County attach (spatial join + branch-county fallback)
# ============================================================
def attach_borrower_county_spatial(
    loans: pd.DataFrame,
    county_shp_path: Path,
    sod_branch_geo: pd.DataFrame,
) -> pd.DataFrame:
    try:
        import geopandas as gpd
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "geopandas is required to attach borrower counties from the shapefile.\n"
            "Try: conda install -c conda-forge geopandas\n"
            "or:  pip install geopandas pyogrio shapely\n"
        ) from e

    df = loans.copy()

    if LAT_COL not in df.columns or LON_COL not in df.columns:
        raise ValueError(f"Loans missing lat/lon columns: {LAT_COL}, {LON_COL}")

    coord_ok = df[LAT_COL].notna() & df[LON_COL].notna()
    print(f"Spatial join: {coord_ok.sum():,}/{len(df):,} loans have borrower coords.")

    counties = gpd.read_file(county_shp_path).to_crs("EPSG:4326")
    if "GEOID" not in counties.columns:
        raise ValueError("County shapefile does not have GEOID column.")

    counties = counties[["GEOID", "geometry"]]

    pts = gpd.GeoDataFrame(
        df.loc[coord_ok].copy(),
        geometry=gpd.points_from_xy(df.loc[coord_ok, LON_COL], df.loc[coord_ok, LAT_COL]),
        crs="EPSG:4326",
    )

    joined = gpd.sjoin(pts, counties, how="left", predicate="within")
    df.loc[coord_ok, COUNTY_COL] = joined["GEOID"].astype("string").values

    # fallback: fill missing borrower county with branch county from SoD
    df = df.merge(
        sod_branch_geo[[BRANCH_COL, YEAR_COL, "STCNTYBR"]],
        on=[BRANCH_COL, YEAR_COL],
        how="left",
    )
    missing_cty = df[COUNTY_COL].isna()
    df.loc[missing_cty, COUNTY_COL] = df.loc[missing_cty, "STCNTYBR"]
    df.drop(columns=["STCNTYBR"], inplace=True)

    df[COUNTY_COL] = df[COUNTY_COL].astype("string").str.zfill(5)

    miss_final = df[COUNTY_COL].isna().sum()
    print(f"borrower_county missing after join+fallback: {miss_final:,}")

    if miss_final == len(df):
        raise ValueError("borrower_county is fully missing after join+fallback.")

    return df


# ============================================================
# Market share
# ============================================================
def compute_bank_market_share_bcy(
    loans: pd.DataFrame,
    loan_amount_col: str = "GrossApproval",
    deposit_col: str = "DEPSUMBR",
) -> pd.DataFrame:
    df = loans.copy()

    for c in [loan_amount_col, deposit_col]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    keys_bcy = [BANK_COL, COUNTY_COL, YEAR_COL]
    keys_cy = [COUNTY_COL, YEAR_COL]

    # Loan share within county-year
    bank_loan = (
        df.groupby(keys_bcy, as_index=False)[loan_amount_col]
          .sum()
          .rename(columns={loan_amount_col: "bank_cty_loan_amt"})
    )
    cty_loan = (
        bank_loan.groupby(keys_cy, as_index=False)["bank_cty_loan_amt"]
                .sum()
                .rename(columns={"bank_cty_loan_amt": "cty_total_loan_amt"})
    )
    bank_loan = bank_loan.merge(cty_loan, on=keys_cy, how="left")
    bank_loan["bank_loan_mkt_share"] = bank_loan["bank_cty_loan_amt"] / bank_loan["cty_total_loan_amt"]

    # Deposit share within county-year (dedupe branches so deposits aren't multiplied by loan rows)
    branch_unique = (
        df[[BANK_COL, COUNTY_COL, YEAR_COL, BRANCH_COL, deposit_col]]
        .drop_duplicates(subset=[BANK_COL, COUNTY_COL, YEAR_COL, BRANCH_COL])
    )
    bank_dep = (
        branch_unique.groupby(keys_bcy, as_index=False)[deposit_col]
                     .sum()
                     .rename(columns={deposit_col: "bank_cty_deposits"})
    )
    cty_dep = (
        bank_dep.groupby(keys_cy, as_index=False)["bank_cty_deposits"]
               .sum()
               .rename(columns={"bank_cty_deposits": "cty_total_deposits"})
    )
    bank_dep = bank_dep.merge(cty_dep, on=keys_cy, how="left")
    bank_dep["bank_dep_mkt_share"] = bank_dep["bank_cty_deposits"] / bank_dep["cty_total_deposits"]

    out = bank_loan[keys_bcy + ["bank_loan_mkt_share"]].merge(
        bank_dep[keys_bcy + ["bank_dep_mkt_share"]],
        on=keys_bcy,
        how="outer",
    )

    # clean infs
    for c in ["bank_loan_mkt_share", "bank_dep_mkt_share"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out.loc[~np.isfinite(out[c]), c] = np.nan

    return out


# ============================================================
# Collapse to BCY / CY (same as your working version)
# ============================================================
def collapse_to_bcy_terms(loans: pd.DataFrame) -> pd.DataFrame:
    df = loans.copy()

    status = df.get("LoanStatus").astype("string").str.upper().str.strip()
    df["is_chgoff"] = np.where(status.eq("CHGOFF").fillna(False), 1, 0).astype("int64")
    df["is_pif"]    = np.where(status.eq("PIF").fillna(False),    1, 0).astype("int64")
    df["is_cancld"] = np.where(status.eq("CANCLD").fillna(False), 1, 0).astype("int64")

    df["interest_rate"] = pd.to_numeric(df.get("InitialInterestRate"), errors="coerce")
    df["term_months"]   = pd.to_numeric(df.get("TerminMonths"), errors="coerce")

    rt = df.get("FixedorVariableInterestRate").astype("string").str.upper().str.strip()
    df["is_fixed"]    = np.where(rt.eq("F").fillna(False), 1, 0).astype("int64")
    df["is_variable"] = np.where(rt.eq("V").fillna(False), 1, 0).astype("int64")

    coll = df.get("CollateralInd").astype("string").str.upper().str.strip()
    df["is_collateral"] = np.where(coll.eq("Y").fillna(False), 1, 0).astype("int64")

    df["is_revolver"] = pd.to_numeric(df.get("RevolverStatus"), errors="coerce").fillna(0)
    df["is_revolver"] = np.where(df["is_revolver"] == 1, 1, 0).astype("int64")

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
            closure=("closure", "max"),
            exog_closure=("exog_closure", "max"),
            true_exog_closure=("true_exog_closure", "max"),
            ASSET_mean=("ASSET", "mean") if "ASSET" in df.columns else ("distance_miles", "size"),
        )
    )

    # if ASSET wasn't present, drop bogus placeholder
    if "ASSET" not in df.columns and "ASSET_mean" in bcy.columns:
        bcy = bcy.drop(columns=["ASSET_mean"])

    bcy["chgoff_rate_all"] = np.where(bcy["n_loans"] > 0, bcy["n_chgoff"] / bcy["n_loans"], np.nan)
    bcy["n_resolved"] = bcy["n_chgoff"] + bcy["n_pif"]
    bcy["chgoff_rate_resolved"] = np.where(bcy["n_resolved"] > 0, bcy["n_chgoff"] / bcy["n_resolved"], np.nan)

    return bcy


def collapse_bcy_to_cy_terms(bcy: pd.DataFrame) -> pd.DataFrame:
    keys_cy = [COUNTY_COL, YEAR_COL]
    df = bcy.copy()

    # ensure flags exist
    for c in BINARY_FLAG_COLS:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int64")

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
            # market share sums as sanity checks
            sum_bank_loan_mkt_share=("bank_loan_mkt_share", "sum"),
            sum_bank_dep_mkt_share=("bank_dep_mkt_share", "sum"),
        )
    )

    cy["chgoff_rate_all"] = np.where(cy["n_loans"] > 0, cy["n_chgoff"] / cy["n_loans"], np.nan)
    cy["n_resolved"] = cy["n_chgoff"] + cy["n_pif"]
    cy["chgoff_rate_resolved"] = np.where(cy["n_resolved"] > 0, cy["n_chgoff"] / cy["n_resolved"], np.nan)

    return cy


if __name__ == "__main__":
    main()
