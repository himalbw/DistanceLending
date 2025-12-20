import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR      = Path("source/sba_analysis/data")
DERIVED_DIR   = BASE_DIR / "derived"
RAW_DIR   = BASE_DIR / "raw"
OUT_DESCR_DIR = Path("output/descriptives")
SBA_TERMS_PATH = DERIVED_DIR / "sba_7a_1991_2025_raw.csv"  # from your concat script

CENSUS_COUNTY_SHP  = RAW_DIR / "cb_2024_us_county_500k.shp"
SBA_LOAN_TO_BRANCH = DERIVED_DIR / "sba_7a_loan_to_branch.csv"
SOD_WITH_FLAGS     = DERIVED_DIR / "sod_with_all_exog_flags.csv"  # or sod_with_exog_flags.csv if that's what you have

LOOKBACK      = 5
LAT_COL       = "borrower_lat"
LON_COL       = "borrower_lon"
BANK_COL      = "loan_cert"
COUNTY_COL    = "borrower_county"
YEAR_COL      = "year"
BRANCH_COL    = "branch_id"
BANK_ASSET_COL = "ASSET"

FLAG_COLS = [
    "transformation_flag",
    "trans_role",
    "closure",
    "exog_closure",
    "true_exog_closure",
]

FLAG_COLS_FOR_SUMMARY = ["closure", "exog_closure", "true_exog_closure"]


def main():
    loans, sod, sod_branch_geo = load_data()
    loans_with_cty = attach_borrower_county(loans, sod_branch_geo)

    mkt = compute_branch_market_share(
        loans_with_cty,
        sod,
        loan_amount_col="GrossApproval",   # or your actual name
        deposit_col="DEPSUMBR",
    )

    bcy = collapse_sba_to_bcy(loans_with_cty, mkt, high_share_threshold=0.2)
    cy  = collapse_to_cy(bcy)

    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    bcy.to_csv(DERIVED_DIR / "sba_bcy_market_share_flags.csv", index=False)
    cy.to_csv(DERIVED_DIR / "sba_cy_market_share_flags.csv", index=False)



def load_data(
    loan_path: Path = SBA_LOAN_TO_BRANCH,
    sod_path: Path = SOD_WITH_FLAGS,
):
    """
    Load SBA loan-to-branch data and SoD data, and build a branch-year
    geography table from SoD (UNINUMBR, YEAR -> STCNTYBR).
    """
    print("Loading SBA loans and SoD data...")
    loans = pd.read_csv(loan_path)
    sod   = pd.read_csv(sod_path)

    # branch-year -> county FIPS from SoD STCNTYBR
    sod_branch_geo = (
        sod[["UNINUMBR", "YEAR", "STCNTYBR"]]
        .drop_duplicates()
        .rename(columns={"UNINUMBR": BRANCH_COL})
    )

    sod_branch_geo["STCNTYBR"] = sod_branch_geo["STCNTYBR"].apply(
        lambda x: str(int(x)).zfill(5)
    )

    return loans, sod, sod_branch_geo

def attach_borrower_county(loans: pd.DataFrame, sod_branch_geo: pd.DataFrame) -> pd.DataFrame:
    """
    Attach borrower county FIPS to SBA loans using a spatial join to county
    shapes, and then impute any missing counties from SoD branch geography.
    """
    print("Performing spatial join to counties...")

    # create GeoDataFrame of borrower points
    points = gpd.GeoDataFrame(
        loans,
        geometry=gpd.points_from_xy(loans[LON_COL], loans[LAT_COL]),
        crs="EPSG:4326",
    )

    # read and prep counties
    counties = gpd.read_file(CENSUS_COUNTY_SHP)
    counties = counties.to_crs("EPSG:4326")[
        ["STATEFP", "COUNTYFP", "GEOID", "NAME", "STUSPS", "geometry"]
    ]

    # spatial join borrowers -> counties
    joined = gpd.sjoin(points, counties, how="left", predicate="within")

    out = loans.join(
        joined[["STATEFP", "COUNTYFP", "GEOID", "NAME", "STUSPS"]]
    )

    out = out.rename(
        columns={
            "GEOID": "county_fips",
            "STATEFP": "state_fips",
            "COUNTYFP": "county_code_3",
            "STUSPS": "state_abbr",
            "NAME": "county_name",
        }
    )

    # county_fips -> borrower_county (string)
    out["county_fips"] = out["county_fips"].astype(str)
    out = out.rename(columns={"county_fips": COUNTY_COL})

    print("Imputing missing borrower counties from SoD branch geography...")

    # merge SoD county by (branch_id, year)
    out = out.merge(
        sod_branch_geo,
        left_on=[BRANCH_COL, YEAR_COL],
        right_on=[BRANCH_COL, "YEAR"],
        how="left",
        suffixes=("", "_sod"),
    )

    # fill missing borrower_county from SoD STCNTYBR
    mask_missing = out[COUNTY_COL].isna()
    out.loc[mask_missing, COUNTY_COL] = out.loc[mask_missing, "STCNTYBR"]

    # ensure borrower_county is 5-digit string
    out[COUNTY_COL] = out[COUNTY_COL].apply(
        lambda x: str(x).zfill(5) if pd.notna(x) else x
    )

    # clean up helper cols if you want:
    # out = out.drop(columns=["STCNTYBR", "YEAR_sod"], errors="ignore")

    return out


def attach_sod_flags(panel: pd.DataFrame, sod: pd.DataFrame) -> pd.DataFrame:
    """
    Attach SoD bank size (ASSET) and closure / transformation flags
    to any panel that has branch_id (UNINUMBR) and year.
    """
    cols_needed = ["UNINUMBR", "YEAR", BANK_ASSET_COL] + FLAG_COLS
    missing = [c for c in cols_needed if c not in sod.columns]
    if missing:
        raise ValueError(f"SoD is missing columns needed for flags/ASSET: {missing}")

    sod_flags = (
        sod[cols_needed]
        .rename(columns={"UNINUMBR": BRANCH_COL, "YEAR": YEAR_COL})
        .drop_duplicates(subset=[BRANCH_COL, YEAR_COL])
    )

    out = panel.merge(sod_flags, on=[BRANCH_COL, YEAR_COL], how="left")
    return out

def compute_branch_market_share(
    loans: pd.DataFrame,
    sod: pd.DataFrame,
    loan_amount_col: str = "GrossApproval",
    deposit_col: str = "DEPSUMBR",
) -> pd.DataFrame:
    """
    Build a branch–borrower-county–year panel with:
      - branch_cty_loan_amt
      - cty_total_loan_amt
      - loan_mkt_share
      - branch_deposits
      - cty_total_deposits
      - deposit_mkt_share
      - ASSET and SoD flags merged in.
    `loans` should already have borrower_county attached.
    """

    # 1) Aggregate loans to branch–county–year
    loan_bcy = (
        loans
        .groupby([BRANCH_COL, COUNTY_COL, YEAR_COL], as_index=False)[loan_amount_col]
        .sum()
        .rename(columns={loan_amount_col: "branch_cty_loan_amt"})
    )

    cty_totals = (
        loan_bcy
        .groupby([COUNTY_COL, YEAR_COL], as_index=False)["branch_cty_loan_amt"]
        .sum()
        .rename(columns={"branch_cty_loan_amt": "cty_total_loan_amt"})
    )

    loan_bcy = loan_bcy.merge(cty_totals, on=[COUNTY_COL, YEAR_COL], how="left")
    loan_bcy["loan_mkt_share"] = (
        loan_bcy["branch_cty_loan_amt"] / loan_bcy["cty_total_loan_amt"]
    )

    # 2) Attach deposits from SoD (branch-year)
    if deposit_col not in sod.columns:
        raise ValueError(f"deposit_col='{deposit_col}' not found in SoD columns.")

    sod_branch = (
        sod[["UNINUMBR", "YEAR", deposit_col]]
        .rename(
            columns={
                "UNINUMBR": BRANCH_COL,
                "YEAR": YEAR_COL,
                deposit_col: "branch_deposits",
            }
        )
        .drop_duplicates(subset=[BRANCH_COL, YEAR_COL])
    )

    mkt = loan_bcy.merge(sod_branch, on=[BRANCH_COL, YEAR_COL], how="left")

    cty_dep_totals = (
        mkt
        .groupby([COUNTY_COL, YEAR_COL], as_index=False)["branch_deposits"]
        .sum()
        .rename(columns={"branch_deposits": "cty_total_deposits"})
    )

    mkt = mkt.merge(cty_dep_totals, on=[COUNTY_COL, YEAR_COL], how="left")
    mkt["deposit_mkt_share"] = (
        mkt["branch_deposits"] / mkt["cty_total_deposits"]
    )

    # 3) Attach ASSET + closure / exog flags from SoD
    mkt = attach_sod_flags(mkt, sod)

    return mkt



def _describe_to_long(df: pd.DataFrame,
                      group_type: str,
                      group_label: str) -> pd.DataFrame:
    """
    Take a dataframe with numeric cols and return long-form descriptives:
    one row per (variable, stat) for a given group.
    """
    desc = df.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).T
    desc["variable"] = desc.index

    long = desc.melt(
        id_vars="variable",
        var_name="stat",
        value_name="value"
    )
    long["group_type"] = group_type   # e.g. "overall", "by_closure"
    long["group"] = group_label       # e.g. "all", "0", "1"

    return long


def summarize_market_share(mkt: pd.DataFrame) -> None:
    OUT_DESCR_DIR.mkdir(parents=True, exist_ok=True)

    numeric = mkt[["loan_mkt_share", "deposit_mkt_share"]]

    frames = []

    # overall
    frames.append(
        _describe_to_long(numeric, group_type="overall", group_label="all")
    )

    # by each flag (if present)
    for flag in FLAG_COLS_FOR_SUMMARY:
        if flag not in mkt.columns:
            continue

        for val, sub in mkt.groupby(flag):
            frames.append(
                _describe_to_long(
                    sub[["loan_mkt_share", "deposit_mkt_share"]],
                    group_type=f"by_{flag}",
                    group_label=str(val),
                )
            )

    descriptives = pd.concat(frames, ignore_index=True)

    csv_path = OUT_DESCR_DIR / "branch_market_share_descriptives.csv"
    descriptives.to_csv(csv_path, index=False)
    print(f"Saved descriptives to {csv_path}")

    # 2. 2×2 histogram figure as a sanity check
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # (1,1) loan share overall
    mkt["loan_mkt_share"].dropna().hist(bins=50, ax=axes[0, 0])
    axes[0, 0].set_title("Loan market share – overall")
    axes[0, 0].set_xlabel("loan_mkt_share")

    # (1,2) deposit share overall
    mkt["deposit_mkt_share"].dropna().hist(bins=50, ax=axes[0, 1])
    axes[0, 1].set_title("Deposit market share – overall")
    axes[0, 1].set_xlabel("deposit_mkt_share")

    # (2,1) & (2,2): use closure == 1 if present; otherwise reuse overall
    if "closure" in mkt.columns:
        closed = mkt[mkt["closure"] == 1]
        title_suffix = " (closure = 1)"
    else:
        closed = mkt
        title_suffix = " (all obs)"

    closed["loan_mkt_share"].dropna().hist(bins=50, ax=axes[1, 0])
    axes[1, 0].set_title(f"Loan market share{title_suffix}")
    axes[1, 0].set_xlabel("loan_mkt_share")

    closed["deposit_mkt_share"].dropna().hist(bins=50, ax=axes[1, 1])
    axes[1, 1].set_title(f"Deposit market share{title_suffix}")
    axes[1, 1].set_xlabel("deposit_mkt_share")

    fig.tight_layout()

    fig_path = OUT_DESCR_DIR / "branch_market_share_histograms.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    print(f"Saved histogram figure to {fig_path}")

def collapse_sba_to_bcy(
    loans: pd.DataFrame,
    mkt: pd.DataFrame,
    high_share_threshold: float = 0.2,
) -> pd.DataFrame:
    """
    Collapse SBA + SoD to bank–borrower-county–year (bcy).

    Parameters
    ----------
    loans : loan-level DataFrame
        Must contain:
          - BANK_COL (e.g. 'loan_cert')
          - COUNTY_COL (e.g. 'borrower_county')
          - YEAR_COL (e.g. 'year')
          - 'LoanStatus', 'GrossApproval', 'SBAGuaranteedApproval'
          - 'distance_miles'
    mkt : branch–county–year DataFrame
        Output from compute_branch_market_share(); must contain:
          - BRANCH_COL, COUNTY_COL, YEAR_COL
          - 'loan_mkt_share'
          - 'closure', 'exog_closure', 'true_exog_closure'

    Returns
    -------
    bcy : bank–county–year DataFrame with:
      - n_branches, n_closure, n_exog_closure, n_true_exog
      - any_closure, any_exog_closure, any_true_exog
      - combined_market_share_of_closure, high_market_share_closure
      - n_loans, total_gross_approval, total_sba_guaranteed
      - avg_distance_miles, med_distance_miles
      - n_chgoff, n_pif, n_cancld
      - default_rate, pct_chgoff, pct_pif, pct_cancld
    """
    loans = loans.copy()
    mkt = mkt.copy()

    # --------------------------------------------------------------
    # 0. Make sure mkt has BANK_COL (loan_cert) by merging from loans
    # --------------------------------------------------------------
    if BANK_COL not in mkt.columns:
        bank_ids = (
            loans[[BRANCH_COL, COUNTY_COL, YEAR_COL, BANK_COL]]
            .drop_duplicates()
        )
        mkt = mkt.merge(
            bank_ids,
            on=[BRANCH_COL, COUNTY_COL, YEAR_COL],
            how="left",
        )

    keys = [BANK_COL, COUNTY_COL, YEAR_COL]

    # --------------------------------------------------------------
    # 1. Loan-level prep for status indicators
    # --------------------------------------------------------------
    loans["LoanStatus"] = loans["LoanStatus"].astype(str).str.upper()

    loans["is_chgoff"] = (loans["LoanStatus"] == "CHGOFF").astype(int)
    loans["is_pif"]    = (loans["LoanStatus"] == "PIF").astype(int)
    loans["is_cancld"] = (loans["LoanStatus"] == "CANCLD").astype(int)

    loan_stats = (
        loans
        .groupby(keys, as_index=False)
        .agg(
            n_loans=("LoanStatus", "size"),
            total_gross_approval=("GrossApproval", "sum"),
            total_sba_guaranteed=("SBAGuaranteedApproval", "sum"),
            avg_distance_miles=("distance_miles", "mean"),
            med_distance_miles=("distance_miles", "median"),
            n_chgoff=("is_chgoff", "sum"),
            n_pif=("is_pif", "sum"),
            n_cancld=("is_cancld", "sum"),
        )
    )

    loan_stats["default_rate"] = loan_stats["n_chgoff"] / loan_stats["n_loans"]
    loan_stats["pct_chgoff"]   = loan_stats["n_chgoff"] / loan_stats["n_loans"]
    loan_stats["pct_pif"]      = loan_stats["n_pif"] / loan_stats["n_loans"]
    loan_stats["pct_cancld"]   = loan_stats["n_cancld"] / loan_stats["n_loans"]

    # --------------------------------------------------------------
    # 2. Branch presence / closure stats from mkt
    # --------------------------------------------------------------
    for col in ["closure", "exog_closure", "true_exog_closure"]:
        if col in mkt.columns:
            mkt[col] = mkt[col].fillna(0).astype(int)

    branch_presence = (
        mkt[[BANK_COL, BRANCH_COL, COUNTY_COL, YEAR_COL,
             "closure", "exog_closure", "true_exog_closure"]]
        .drop_duplicates()
    )

    bcy_core = (
        branch_presence
        .groupby(keys, as_index=False)
        .agg(
            n_branches=(BRANCH_COL, "nunique"),
            n_closure=("closure", "sum"),
            n_exog_closure=("exog_closure", "sum"),
            n_true_exog=("true_exog_closure", "sum"),
        )
    )

    bcy_core["any_closure"] = (bcy_core["n_closure"] > 0).astype(int)
    bcy_core["any_exog_closure"] = (bcy_core["n_exog_closure"] > 0).astype(int)
    bcy_core["any_true_exog"] = (bcy_core["n_true_exog"] > 0).astype(int)

    # --------------------------------------------------------------
    # 3. Closure market share from mkt
    # --------------------------------------------------------------
    closure_mkt = mkt[mkt["closure"] == 1][
        [BANK_COL, COUNTY_COL, YEAR_COL, "loan_mkt_share"]
    ]

    ms_closure = (
        closure_mkt
        .groupby(keys, as_index=False)["loan_mkt_share"]
        .sum()
        .rename(columns={"loan_mkt_share": "combined_market_share_of_closure"})
    )

    max_ms_closure = (
        closure_mkt
        .groupby(keys, as_index=False)["loan_mkt_share"]
        .max()
        .rename(columns={"loan_mkt_share": "max_closure_mkt_share"})
    )

    # --------------------------------------------------------------
    # 4. Combine closure stuff + loan stats into BCY
    # --------------------------------------------------------------
    bcy = (
        bcy_core
        .merge(ms_closure, on=keys, how="left")
        .merge(max_ms_closure, on=keys, how="left")
        .merge(loan_stats, on=keys, how="left")
    )

    bcy["combined_market_share_of_closure"] = (
        bcy["combined_market_share_of_closure"].fillna(0.0)
    )
    bcy["max_closure_mkt_share"] = bcy["max_closure_mkt_share"].fillna(0.0)

    bcy["high_market_share_closure"] = (
        (bcy["max_closure_mkt_share"] >= high_share_threshold).astype(int)
    )

    bcy = bcy.drop(columns=["max_closure_mkt_share"])

    return bcy

def collapse_to_cy(bcy: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse bank–county–year panel to county–year.

    Expects `bcy` from collapse_sba_to_bcy() with:
      - BANK_COL, COUNTY_COL, YEAR_COL
      - n_branches, n_closure, n_exog_closure, n_true_exog
      - any_closure, any_exog_closure, any_true_exog
      - combined_market_share_of_closure, high_market_share_closure
      - n_loans, total_gross_approval, total_sba_guaranteed
      - avg_distance_miles, med_distance_miles
      - n_chgoff, n_pif, n_cancld
    """
    bcy = bcy.copy()
    keys_cy = [COUNTY_COL, YEAR_COL]

    # ------------------------------------------------------------------
    # 1. Base county-year aggregations
    # ------------------------------------------------------------------
    cy = (
        bcy
        .groupby(keys_cy, as_index=False)
        .agg(
            n_banks=(BANK_COL, "nunique"),
            total_branches=("n_branches", "sum"),

            n_closure=("n_closure", "sum"),
            n_exog_closure=("n_exog_closure", "sum"),
            n_true_exog=("n_true_exog", "sum"),

            any_closure=("any_closure", "max"),
            any_exog_closure=("any_exog_closure", "max"),
            any_true_exog=("any_true_exog", "max"),

            combined_market_share_of_closure=(
                "combined_market_share_of_closure", "sum"
            ),
            high_market_share_closure=("high_market_share_closure", "max"),

            n_loans=("n_loans", "sum"),
            total_gross_approval=("total_gross_approval", "sum"),
            total_sba_guaranteed=("total_sba_guaranteed", "sum"),

            n_chgoff=("n_chgoff", "sum"),
            n_pif=("n_pif", "sum"),
            n_cancld=("n_cancld", "sum"),
        )
    )

    # recompute county-level rates
    cy["default_rate"] = cy["n_chgoff"] / cy["n_loans"]
    cy["pct_chgoff"]   = cy["n_chgoff"] / cy["n_loans"]
    cy["pct_pif"]      = cy["n_pif"] / cy["n_loans"]
    cy["pct_cancld"]   = cy["n_cancld"] / cy["n_loans"]

    # ------------------------------------------------------------------
    # 2. Distance: loan-weighted avg, plus median of bank-level medians
    # ------------------------------------------------------------------
    dist_tmp = (
        bcy
        .assign(weighted_dist=lambda d: d["avg_distance_miles"] * d["n_loans"])
        .groupby(keys_cy, as_index=False)
        .agg(
            sum_weighted_dist=("weighted_dist", "sum"),
            sum_n_loans=("n_loans", "sum"),
            med_distance_miles=("med_distance_miles", "median"),
        )
    )
    dist_tmp["avg_distance_miles"] = (
        dist_tmp["sum_weighted_dist"] / dist_tmp["sum_n_loans"]
    )

    cy = cy.merge(
        dist_tmp[[COUNTY_COL, YEAR_COL, "avg_distance_miles", "med_distance_miles"]],
        on=keys_cy,
        how="left",
    )

    # ------------------------------------------------------------------
    # 3. Clean indicator columns to 0/1 ints
    # ------------------------------------------------------------------
    for col in [
        "any_closure",
        "any_exog_closure",
        "any_true_exog",
        "high_market_share_closure",
    ]:
        cy[col] = cy[col].fillna(0).astype(int)

    return cy


def _norm_name(s: pd.Series) -> pd.Series:
    """Normalize names for fuzzy-ish exact matching (no heavy fuzzywuzzy dependency)."""
    s = s.astype("string").fillna("")
    s = s.str.upper().str.strip()
    # replace & with AND, remove punctuation, collapse whitespace
    s = s.str.replace("&", " AND ", regex=False)
    s = s.str.replace(r"[^A-Z0-9\s]", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    # light suffix cleanup
    s = s.str.replace(r"\b(LLC|INC|CORP|CO|COMPANY|LTD|LP|LLP|PLC)\b", "", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s


def _as_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def attach_loan_terms(
    loans: pd.DataFrame,
    terms_path: Path = SBA_TERMS_PATH,
    out_path: Path | None = DERIVED_DIR / "sba_7a_loan_to_branch_with_terms.csv",
) -> pd.DataFrame:
    """
    Attach FOIA 'loan terms' columns (rate, term months, NAICS, etc.) onto an SBA loan-to-branch file.

    Strategy:
      - Stage 1: strict-ish match on normalized borrower/bank names + approval date + gross approval (+ optional state/zip)
      - Stage 2: fallback match on normalized names + approval FY + gross approval (+ optional state/county)
    """
    if not terms_path.exists():
        raise FileNotFoundError(f"Terms file not found: {terms_path}")

    terms = pd.read_csv(terms_path)

    # ---- Harmonize column names we’ll use ----
    # loans side: you may need to adjust these if your loan-to-branch file uses different names
    # We try a few common alternatives.
    loans = loans.copy()

    def _coalesce_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    loan_borr = _coalesce_col(loans, ["BorrName", "borrower_name", "BorrowerName", "borr_name"])
    loan_bank = _coalesce_col(loans, ["BankName", "bank_name", "LenderName", "lender_name"])
    loan_amt  = _coalesce_col(loans, ["GrossApproval", "gross_approval", "loan_amount", "ApprovalAmount"])
    loan_dt   = _coalesce_col(loans, ["ApprovalDate", "approval_date", "Approval_Date"])
    loan_fy   = _coalesce_col(loans, ["ApprovalFY", "approval_fy", "fy", "fiscal_year"])
    loan_state = _coalesce_col(loans, ["ProjectState", "borrower_state", "state", "BorrState"])
    loan_zip   = _coalesce_col(loans, ["BorrZip", "borrower_zip", "zip"])

    # terms file columns (from your printout)
    term_borr = "BorrName"
    term_bank = "BankName"
    term_amt  = "GrossApproval"
    term_dt   = "ApprovalDate"
    term_fy   = "ApprovalFY"
    term_state = "ProjectState"
    term_county = "ProjectCounty"  # not always reliable, but useful in fallback
    term_zip   = "BorrZip"

    # ---- Create merge keys ----
    # Dates + FY
    if loan_dt is not None:
        loans["_approval_date"] = _as_date(loans[loan_dt])
    else:
        loans["_approval_date"] = pd.NaT

    if loan_fy is not None:
        loans["_approval_fy"] = pd.to_numeric(loans[loan_fy], errors="coerce").astype("Int64")
    else:
        # infer FY if date exists (federal FY starts Oct 1; you might prefer calendar year)
        loans["_approval_fy"] = loans["_approval_date"].dt.year.astype("Int64")

    terms["_approval_date"] = _as_date(terms[term_dt])
    terms["_approval_fy"]   = pd.to_numeric(terms[term_fy], errors="coerce").astype("Int64")

    # Amount (rounded for robustness)
    if loan_amt is None:
        raise ValueError("Could not find a loan amount column in loans. Add it or map loan_amt candidates.")
    loans["_gross"] = pd.to_numeric(loans[loan_amt], errors="coerce").round(0)
    terms["_gross"] = pd.to_numeric(terms[term_amt], errors="coerce").round(0)

    # Names
    if loan_borr is None or loan_bank is None:
        raise ValueError("Could not find borrower/bank name columns in loans. Map loan_borr/loan_bank.")
    loans["_borr_norm"] = _norm_name(loans[loan_borr])
    loans["_bank_norm"] = _norm_name(loans[loan_bank])
    terms["_borr_norm"] = _norm_name(terms[term_borr])
    terms["_bank_norm"] = _norm_name(terms[term_bank])

    # Optional geography helpers
    if loan_state is not None:
        loans["_state"] = loans[loan_state].astype("string").str.upper().str.strip()
    else:
        loans["_state"] = pd.NA
    terms["_state"] = terms[term_state].astype("string").str.upper().str.strip()

    if loan_zip is not None:
        loans["_zip5"] = loans[loan_zip].astype("string").str.replace(r"\D", "", regex=True).str.zfill(5)
    else:
        loans["_zip5"] = pd.NA
    terms["_zip5"] = terms[term_zip].astype("string").str.replace(r"\D", "", regex=True).str.zfill(5)

    terms["_county"] = terms[term_county].astype("string").str.upper().str.strip()

    # ---- What “extra information” do we want to bring over? ----
    # Keep these plus whatever else you care about.
    keep_cols = [
        "Program",
        "Subprogram",
        "ProcessingMethod",
        "InitialInterestRate",
        "FixedorVariableInterestRate",
        "TerminMonths",
        "NAICSCode",
        "NAICSDescription",
        "BusinessType",
        "BusinessAge",
        "LoanStatus",
        "JobsSupported",
        "CollateralInd",
        "SoldSecondMarketInd",
        "SBAGuaranteedApproval",
        "FirstDisbursementDate",
        "PaidinFullDate",
        "ChargeoffDate",
        "GrossChargeoffAmount",
        "RevolverStatus",
        "ProjectCounty",
        "ProjectState",
        "SBADistrictOffice",
        "CongressionalDistrict",
        "LocationID",
        "BankFDICNumber",
        "BankNCUANumber",
    ]
    keep_cols = [c for c in keep_cols if c in terms.columns]

    # Deduplicate terms on the keys we’ll use, so merge is stable.
    # If duplicates remain, we keep the first; you can change the rule.
    def _dedup(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
        return df.sort_values(keys).drop_duplicates(subset=keys, keep="first")

    # ---- Stage 1 merge: names + approval date + gross (+ state/zip if available) ----
    stage1_keys = ["_borr_norm", "_bank_norm", "_approval_date", "_gross"]
    if loans["_state"].notna().any():
        stage1_keys.append("_state")
    if loans["_zip5"].notna().any():
        stage1_keys.append("_zip5")

    terms_s1 = _dedup(terms[stage1_keys + keep_cols].copy(), stage1_keys)

    merged = loans.merge(
        terms_s1,
        on=stage1_keys,
        how="left",
        suffixes=("", "_terms"),
        validate="m:1",
    )
    merged["_terms_match_stage"] = np.where(merged[keep_cols[0]].notna() if keep_cols else False, 1, 0)

    # ---- Stage 2 merge (fallback): names + FY + gross (+ state) ----
    # Only for rows not matched in stage 1
    need = merged["_terms_match_stage"].eq(0)
    if need.any():
        stage2_keys = ["_borr_norm", "_bank_norm", "_approval_fy", "_gross"]
        if loans["_state"].notna().any():
            stage2_keys.append("_state")

        terms_s2 = _dedup(terms[stage2_keys + keep_cols].copy(), stage2_keys)

        add = merged.loc[need, stage2_keys].merge(
            terms_s2,
            on=stage2_keys,
            how="left",
            validate="m:1",
        )

        # write the newly found columns back in (only where currently missing)
        for c in keep_cols:
            if c in merged.columns and c in add.columns:
                merged.loc[need, c] = merged.loc[need, c].combine_first(add[c].values)

        merged.loc[need & merged[keep_cols[0]].notna(), "_terms_match_stage"] = 2

    # ---- Cleanup + logging ----
    n = len(merged)
    m1 = int((merged["_terms_match_stage"] == 1).sum())
    m2 = int((merged["_terms_match_stage"] == 2).sum())
    print(f"[attach_loan_terms] total loans: {n:,}")
    print(f"[attach_loan_terms] matched stage 1 (date): {m1:,} ({m1/n:.1%})")
    print(f"[attach_loan_terms] matched stage 2 (FY):   {m2:,} ({m2/n:.1%})")
    print(f"[attach_loan_terms] unmatched:             {n-m1-m2:,} ({(n-m1-m2)/n:.1%})")

    # Drop helper cols unless you want to keep them
    helper_cols = [c for c in merged.columns if c.startswith("_")]
    merged = merged.drop(columns=helper_cols)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out_path, index=False)
        print(f"[attach_loan_terms] wrote: {out_path}")

    return merged



if __name__ == "__main__":
    main()
