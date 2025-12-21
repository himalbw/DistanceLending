import pandas as pd
import numpy as np
from pathlib import Path


# -------------------------
# Paths
# -------------------------
IN_PATH = Path("source/sba_analysis/data/derived/sba_7a_loan_to_branch_with_terms.csv")

# fallback to recover borrower_county if it didn't survive your merge
LOAN_TO_BRANCH_PATH = Path("source/sba_analysis/data/derived/sba_7a_loan_to_branch.csv")

# SoD with flags (closure/exog/etc) at branch-year
SOD_WITH_FLAGS_PATH = Path("source/sba_analysis/data/derived/sod_with_all_exog_flags.csv")

# outputs
DERIVED_DIR = Path("source/sba_analysis/data/derived")
OUT_BCY_PATH = DERIVED_DIR / "sba_bcy_terms.csv"
OUT_CY_PATH  = DERIVED_DIR / "sba_cy_terms.csv"

# also copy for your terms analysis folder
TERMS_ANALYSIS_DIR = Path("source/sba-terms-analysis/data")
OUT_BCY_COPY = TERMS_ANALYSIS_DIR / "sba_bcy_terms.csv"
OUT_CY_COPY  = TERMS_ANALYSIS_DIR / "sba_cy_terms.csv"


# -------------------------
# Column names
# -------------------------
BANK_COL   = "loan_cert"         # bank identifier in your SBA loan data
BRANCH_COL = "branch_id"
YEAR_COL   = "year"
COUNTY_COL = "borrower_county"   # what we want to exist

FLAG_COLS = [
    "transformation_flag",
    "trans_role",
    "closure",
    "exog_closure",
    "true_exog_closure",
]

# loan terms columns (from your merged file)
TERMS_COLS = {
    "InitialInterestRate": "interest_rate",
    "FixedorVariableInterestRate": "rate_type",
    "TerminMonths": "term_months",
    "CollateralInd": "collateral_ind",
    "RevolverStatus": "revolver_status",
}


def main() -> None:
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    TERMS_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading loan-level data: {IN_PATH}")
    loans = pd.read_csv(IN_PATH, low_memory=False)

    loans = ensure_borrower_county(loans)
    loans = attach_sod_flags(loans)

    bcy = collapse_to_bcy_terms(loans)
    cy  = collapse_bcy_to_cy_terms(bcy)

    print(f"Saving BCY -> {OUT_BCY_PATH}")
    bcy.to_csv(OUT_BCY_PATH, index=False)

    print(f"Saving CY  -> {OUT_CY_PATH}")
    cy.to_csv(OUT_CY_PATH, index=False)

    # copy to terms analysis folder (as you requested)
    print(f"Copy BCY -> {OUT_BCY_COPY}")
    bcy.to_csv(OUT_BCY_COPY, index=False)

    print(f"Copy CY  -> {OUT_CY_COPY}")
    cy.to_csv(OUT_CY_COPY, index=False)

    print("Done.")


# -------------------------
# Helpers
# -------------------------
def ensure_borrower_county(loans: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure borrower_county exists.
    If missing, merge it in from sba_7a_loan_to_branch.csv using robust keys.
    """
    if COUNTY_COL in loans.columns and loans[COUNTY_COL].notna().any():
        # normalize to 5-digit string if it looks like numeric FIPS
        loans = loans.copy()
        loans[COUNTY_COL] = loans[COUNTY_COL].apply(
            lambda x: str(int(x)).zfill(5) if pd.notna(x) and str(x).strip() != "" and str(x).replace(".", "", 1).isdigit() else x
        )
        return loans

    print(
        f"WARNING: '{COUNTY_COL}' missing/empty in {IN_PATH}. "
        f"Attempting to recover from {LOAN_TO_BRANCH_PATH}..."
    )

    base = pd.read_csv(LOAN_TO_BRANCH_PATH, low_memory=False)

    if COUNTY_COL not in base.columns:
        raise ValueError(
            f"Cannot recover '{COUNTY_COL}': not found in {LOAN_TO_BRANCH_PATH} either."
        )

    # Keys: prefer identifiers that should match exactly across your merge products
    # We'll try a conservative set first, then fall back to a looser set.
    loans = loans.copy()

    # normalize county FIPS
    base = base.copy()
    base[COUNTY_COL] = base[COUNTY_COL].apply(
        lambda x: str(x).zfill(5) if pd.notna(x) else x
    )

    # Attempt 1: (loan_cert, year, ApprovalDate, GrossApproval, borrower_lat/lon, branch_id)
    keys1 = [BANK_COL, YEAR_COL, "ApprovalDate", "GrossApproval", "borrower_lat", "borrower_lon", BRANCH_COL]
    keys1 = [k for k in keys1 if k in loans.columns and k in base.columns]

    if len(keys1) >= 3:
        tmp = base[keys1 + [COUNTY_COL]].drop_duplicates(subset=keys1)
        loans = loans.merge(tmp, on=keys1, how="left", suffixes=("", "_from_base"))
        if COUNTY_COL in loans.columns and loans[COUNTY_COL].notna().any():
            return loans

    # Attempt 2: looser (loan_cert, year, branch_id, borrower_lat/lon)
    keys2 = [BANK_COL, YEAR_COL, BRANCH_COL, "borrower_lat", "borrower_lon"]
    keys2 = [k for k in keys2 if k in loans.columns and k in base.columns]

    if len(keys2) >= 3:
        tmp = base[keys2 + [COUNTY_COL]].drop_duplicates(subset=keys2)
        loans = loans.merge(tmp, on=keys2, how="left", suffixes=("", "_from_base"))

    if loans[COUNTY_COL].isna().all():
        raise ValueError(
            f"Failed to recover '{COUNTY_COL}'. "
            "Check merge keys or ensure borrower_county is produced upstream."
        )

    return loans


def attach_sod_flags(loans: pd.DataFrame) -> pd.DataFrame:
    """
    Attach closure/exog flags (and ASSET if present) from SoD by (branch_id, year).
    """
    sod = pd.read_csv(SOD_WITH_FLAGS_PATH, low_memory=False)

    needed = ["UNINUMBR", "YEAR"] + [c for c in FLAG_COLS if c in sod.columns]
    # also keep ASSET if present, because you use it later in R bank-size splits
    if "ASSET" in sod.columns:
        needed.append("ASSET")

    missing_core = [c for c in ["UNINUMBR", "YEAR"] if c not in sod.columns]
    if missing_core:
        raise ValueError(f"SoD file missing required keys: {missing_core}")

    sod_sub = (
        sod[needed]
        .rename(columns={"UNINUMBR": BRANCH_COL, "YEAR": YEAR_COL})
        .drop_duplicates(subset=[BRANCH_COL, YEAR_COL])
    )

    out = loans.merge(sod_sub, on=[BRANCH_COL, YEAR_COL], how="left")

    # fill flags to 0/1 int if present
    for c in FLAG_COLS:
        if c in out.columns:
            out[c] = out[c].fillna(0).astype(int)

    return out


def collapse_to_bcy_terms(loans: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse loan-level terms + outcomes to bank–county–year.
    Includes:
      - volumes (n_loans, totals)
      - default/chargeoff rate based on LoanStatus == CHGOFF
      - distance moments
      - term moments (rate, maturity, fixed/variable, collateral, revolver)
      - closure flags aggregated within bank-county-year
    """
    df = loans.copy()

    # ---- status-based default (your coding)
    df["LoanStatus"] = df["LoanStatus"].astype("string").str.upper().str.strip()
    df["is_chgoff"] = (df["LoanStatus"] == "CHGOFF").astype(int)
    df["is_pif"]    = (df["LoanStatus"] == "PIF").astype(int)
    df["is_cancld"] = (df["LoanStatus"] == "CANCLD").astype(int)

    # ---- term variables
    # numeric
    if "InitialInterestRate" in df.columns:
        df["interest_rate"] = pd.to_numeric(df["InitialInterestRate"], errors="coerce")
    else:
        df["interest_rate"] = np.nan

    if "TerminMonths" in df.columns:
        df["term_months"] = pd.to_numeric(df["TerminMonths"], errors="coerce")
    else:
        df["term_months"] = np.nan

    # binary contract features (your codes: F/V, Y/N, Revolver 0/1)
    df["is_fixed"] = (
        df.get("FixedorVariableInterestRate", pd.Series(index=df.index, dtype="object"))
          .astype("string").str.upper().str.strip().eq("F")
    ).astype(int)

    df["is_variable"] = (
        df.get("FixedorVariableInterestRate", pd.Series(index=df.index, dtype="object"))
          .astype("string").str.upper().str.strip().eq("V")
    ).astype(int)

    df["is_collateral"] = (
        df.get("CollateralInd", pd.Series(index=df.index, dtype="object"))
          .astype("string").str.upper().str.strip().eq("Y")
    ).astype(int)

    df["is_revolver"] = pd.to_numeric(df.get("RevolverStatus"), errors="coerce").fillna(0).astype(int)

    keys = [BANK_COL, COUNTY_COL, YEAR_COL]

    # ---- loan-level aggregation to BCY
    agg_dict = {
        "LoanStatus": ("LoanStatus", "size"),
        "GrossApproval": ("GrossApproval", "sum"),
        "SBAGuaranteedApproval": ("SBAGuaranteedApproval", "sum"),
        "distance_miles": ("distance_miles", "mean"),
        "is_chgoff": ("is_chgoff", "sum"),
        "is_pif": ("is_pif", "sum"),
        "is_cancld": ("is_cancld", "sum"),

        # terms
        "interest_rate_mean": ("interest_rate", "mean"),
        "interest_rate_p50": ("interest_rate", "median"),
        "term_months_mean": ("term_months", "mean"),
        "term_months_p50": ("term_months", "median"),
        "fixed_share": ("is_fixed", "mean"),
        "variable_share": ("is_variable", "mean"),
        "collateral_share": ("is_collateral", "mean"),
        "revolver_share": ("is_revolver", "mean"),
    }

    # build aggregation cleanly
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

    bcy["chgoff_rate_all"] = bcy["n_chgoff"] / bcy["n_loans"]

    # resolved-only denominator (PIF or CHGOFF)
    bcy["n_resolved"] = bcy["n_chgoff"] + bcy["n_pif"]
    bcy["chgoff_rate_resolved"] = np.where(
        bcy["n_resolved"] > 0,
        bcy["n_chgoff"] / bcy["n_resolved"],
        np.nan
    )

    # ---- closure flags: aggregate by max within BCY
    for c in FLAG_COLS:
        if c in df.columns:
            flags = (
                df.groupby(keys, as_index=False)[c]
                  .max()
                  .rename(columns={c: c})
            )
            bcy = bcy.merge(flags, on=keys, how="left")
        else:
            bcy[c] = 0

    # ---- ASSET (keep mean ASSET for this bank-county-year; you can later do bank-level medians in R)
    if "ASSET" in df.columns:
        asset = (
            df.groupby(keys, as_index=False)["ASSET"]
              .mean()
              .rename(columns={"ASSET": "ASSET_mean"})
        )
        bcy = bcy.merge(asset, on=keys, how="left")

    return bcy


def collapse_bcy_to_cy_terms(bcy: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse bank–county–year to county–year.
    Uses loan-weighting where sensible.
    """
    keys_cy = [COUNTY_COL, YEAR_COL]
    df = bcy.copy()

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

              # flags (county-year has any bank closure)
              closure=("closure", "max"),
              exog_closure=("exog_closure", "max"),
              true_exog_closure=("true_exog_closure", "max"),
          )
    )

    # rates at county-year
    cy["chgoff_rate_all"] = cy["n_chgoff"] / cy["n_loans"]
    cy["n_resolved"] = cy["n_chgoff"] + cy["n_pif"]
    cy["chgoff_rate_resolved"] = np.where(
        cy["n_resolved"] > 0,
        cy["n_chgoff"] / cy["n_resolved"],
        np.nan
    )

    # loan-weighted distance and terms
    tmp = df.copy()
    tmp["w_dist"] = tmp["avg_distance_miles"] * tmp["n_loans"]
    tmp["w_rate"] = tmp["interest_rate_mean"] * tmp["n_loans"]
    tmp["w_term"] = tmp["term_months_mean"] * tmp["n_loans"]

    tmp["w_fixed"] = tmp["fixed_share"] * tmp["n_loans"]
    tmp["w_var"] = tmp["variable_share"] * tmp["n_loans"]
    tmp["w_coll"] = tmp["collateral_share"] * tmp["n_loans"]
    tmp["w_rev"] = tmp["revolver_share"] * tmp["n_loans"]

    w = (
        tmp.groupby(keys_cy, as_index=False)
           .agg(
               sum_w_dist=("w_dist", "sum"),
               sum_w_rate=("w_rate", "sum"),
               sum_w_term=("w_term", "sum"),
               sum_w_fixed=("w_fixed", "sum"),
               sum_w_var=("w_var", "sum"),
               sum_w_coll=("w_coll", "sum"),
               sum_w_rev=("w_rev", "sum"),
               sum_n_loans=("n_loans", "sum"),
               med_distance_miles=("med_distance_miles", "median"),
               interest_rate_p50=("interest_rate_p50", "median"),
               term_months_p50=("term_months_p50", "median"),
           )
    )

    w["avg_distance_miles"] = w["sum_w_dist"] / w["sum_n_loans"]
    w["interest_rate_mean_w"] = w["sum_w_rate"] / w["sum_n_loans"]
    w["term_months_mean_w"] = w["sum_w_term"] / w["sum_n_loans"]

    w["fixed_share_w"] = w["sum_w_fixed"] / w["sum_n_loans"]
    w["variable_share_w"] = w["sum_w_var"] / w["sum_n_loans"]
    w["collateral_share_w"] = w["sum_w_coll"] / w["sum_n_loans"]
    w["revolver_share_w"] = w["sum_w_rev"] / w["sum_n_loans"]

    cy = cy.merge(
        w[
            keys_cy
            + [
                "avg_distance_miles",
                "med_distance_miles",
                "interest_rate_mean_w",
                "interest_rate_p50",
                "term_months_mean_w",
                "term_months_p50",
                "fixed_share_w",
                "variable_share_w",
                "collateral_share_w",
                "revolver_share_w",
            ]
        ],
        on=keys_cy,
        how="left",
    )

    return cy


if __name__ == "__main__":
    main()
