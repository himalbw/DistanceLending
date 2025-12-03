import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR      = Path("source/sba_analysis/data")
DERIVED_DIR   = BASE_DIR / "derived"
RAW_DIR   = BASE_DIR / "raw"
OUT_DESCR_DIR = Path("output/descriptives")

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

    # mkt: branch–county–year with market share + flags
    mkt = compute_branch_market_share(
        loans_with_cty,
        sod,
        loan_amount_col="GrossApproval",
        deposit_col="DEPSUMBR",
    )

    # make sure bank id is present in mkt (if not, merge from loans)
    if BANK_COL not in mkt.columns:
        bank_ids = (
            loans_with_cty[[BRANCH_COL, COUNTY_COL, YEAR_COL, BANK_COL]]
            .drop_duplicates()
        )
        mkt = mkt.merge(
            bank_ids,
            on=[BRANCH_COL, COUNTY_COL, YEAR_COL],
            how="left",
        )
    bcy = collapse_sba_to_bcy(mkt, high_share_threshold=0.2)
    cy = collapse_to_cy(bcy)

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
    df: pd.DataFrame,
    high_share_threshold: float = 0.2,
) -> pd.DataFrame:
    """
    Collapse branch-level SBA+SoD panel to bank–borrower-county–year (bcy).

    Expects columns:
      - BANK_COL (e.g. 'loan_cert')
      - BRANCH_COL (e.g. 'branch_id')
      - COUNTY_COL (e.g. 'borrower_county')
      - YEAR_COL (e.g. 'year')
      - 'loan_mkt_share'
      - 'closure', 'exog_closure', 'true_exog_closure'

    Creates:
      - n_branches: number of branches active (distinct branch_id)
      - any_closure, any_exog_closure, any_true_exog: 0/1 indicators
      - n_closure, n_exog_closure, n_true_exog: counts of closing branches
      - combined_market_share_of_closure: sum loan_mkt_share over closing branches
      - high_market_share_closure: 1 if any closing branch has
                                   loan_mkt_share >= high_share_threshold
    """
    keys = [BANK_COL, COUNTY_COL, YEAR_COL]

    # make sure closure flags are treated as 0/1
    for col in ["closure", "exog_closure", "true_exog_closure"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    # ------------------------------------------------------------------
    # 1. Core aggregations at bank–county–year
    # ------------------------------------------------------------------
    bcy_core = (
        df
        .groupby(keys, as_index=False)
        .agg(
            n_branches=(BRANCH_COL, "nunique"),
            n_closure=("closure", "sum"),
            n_exog_closure=("exog_closure", "sum"),
            n_true_exog=("true_exog_closure", "sum"),
        )
    )

    # indicator versions
    bcy_core["any_closure"] = (bcy_core["n_closure"] > 0).astype(int)
    bcy_core["any_exog_closure"] = (bcy_core["n_exog_closure"] > 0).astype(int)
    bcy_core["any_true_exog"] = (bcy_core["n_true_exog"] > 0).astype(int)

    # ------------------------------------------------------------------
    # 2. Combined market share of closure branches
    # ------------------------------------------------------------------
    closure_df = df[df["closure"] == 1]

    ms_closure = (
        closure_df
        .groupby(keys, as_index=False)["loan_mkt_share"]
        .sum()
        .rename(columns={"loan_mkt_share": "combined_market_share_of_closure"})
    )

    # ------------------------------------------------------------------
    # 3. High-market-share closure indicator
    # ------------------------------------------------------------------
    max_ms_closure = (
        closure_df
        .groupby(keys, as_index=False)["loan_mkt_share"]
        .max()
        .rename(columns={"loan_mkt_share": "max_closure_mkt_share"})
    )

    bcy = (
        bcy_core
        .merge(ms_closure, on=keys, how="left")
        .merge(max_ms_closure, on=keys, how="left")
    )

    # fill NaNs for groups with no closure
    bcy["combined_market_share_of_closure"] = (
        bcy["combined_market_share_of_closure"].fillna(0.0)
    )
    bcy["max_closure_mkt_share"] = bcy["max_closure_mkt_share"].fillna(0.0)

    bcy["high_market_share_closure"] = (
        (bcy["max_closure_mkt_share"] >= high_share_threshold).astype(int)
    )

    # optional: drop the helper max column
    bcy = bcy.drop(columns=["max_closure_mkt_share"])

    return bcy


def collapse_to_cy(bcy: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse bank–county–year panel to county–year.

    Expects bcy from `collapse_sba_to_bcy()` with columns:
      - BANK_COL
      - COUNTY_COL
      - YEAR_COL
      - n_branches, n_closure, n_exog_closure, n_true_exog
      - any_closure, any_exog_closure, any_true_exog
      - combined_market_share_of_closure
      - high_market_share_closure

    Produces county-year level:
      - n_banks: number of banks active in county-year
      - total_branches: total branches active in county-year
      - any_*: 1 if any bank has that flag
      - n_*: sums across banks
      - combined_market_share_of_closure: sum over banks
      - high_market_share_closure: 1 if any bank has high_market_share_closure
    """
    grp = bcy.groupby([COUNTY_COL, YEAR_COL], as_index=False)

    cy = grp.agg(
        n_banks=(BANK_COL, "nunique"),
        total_branches=("n_branches", "sum"),
        n_closure=("n_closure", "sum"),
        n_exog_closure=("n_exog_closure", "sum"),
        n_true_exog=("n_true_exog", "sum"),
        any_closure=("any_closure", "max"),
        any_exog_closure=("any_exog_closure", "max"),
        any_true_exog=("any_true_exog", "max"),
        combined_market_share_of_closure=("combined_market_share_of_closure", "sum"),
        high_market_share_closure=("high_market_share_closure", "max"),
    )

    # force indicators to be 0/1 ints
    for col in [
        "any_closure",
        "any_exog_closure",
        "any_true_exog",
        "high_market_share_closure",
    ]:
        cy[col] = cy[col].fillna(0).astype(int)

    return cy


if __name__ == "__main__":
    main()
