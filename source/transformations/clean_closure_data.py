import pandas as pd
from pathlib import Path
import ast
import numpy as np

from typing import Tuple

INDIR_SBA = Path("source/transformations/data")

def main() -> None:
    sba_bcy, sba_cy, sod_bcy, sod_cy = load_data()
    stats_bcy, stats_cy = get_summary_stats(sba_bcy, sba_cy)
    stats_bcy_cl, stats_cy_cl = get_closure_comparison_stats(sba_bcy, sba_cy)
    stats_bcy_ex, stats_cy_ex = get_exog_closure_comparison_stats(sba_bcy, sba_cy)
    stats_bcy_tex, stats_cy_tex = get_true_exog_closure_comparison_stats(sba_bcy, sba_cy)
    pretty_bcy, pretty_cy = make_pretty_summary_tables(
        stats_bcy, stats_cy,
        stats_bcy_cl, stats_cy_cl,
        stats_bcy_ex, stats_cy_ex,
        stats_bcy_tex, stats_cy_tex,
    )
    print("\nBank–county–year pretty summary:")
    print(pretty_bcy)
    print("\nCounty–year pretty summary:")
    print(pretty_cy)

def load_data():
    sba_bcy = load_sba_closure_data()
    sba_bcy = add_branch_numbers(sba_bcy)
    sba_bcy = add_loan_growth_and_windows(sba_bcy)
    sba_cy = collapse_sba_to_county_year(sba_bcy)
    sod_bcy, sod_cy = get_bank_county_covariates()
    sba_bcy.to_csv(INDIR_SBA / "sba_bcy.csv", index=False)
    sba_cy.to_csv(INDIR_SBA / "sba_cy.csv", index=False)
    sod_bcy.to_csv(INDIR_SBA / "sod_bcy.csv", index=False)
    sod_cy.to_csv(INDIR_SBA / "sod_cy.csv", index=False)
    print("Saved SBA and SOD panels to source/transformations/data")
    return sba_bcy, sba_cy, sod_bcy, sod_cy


def load_sba_closure_data(filename="sba_loan_county_year_closure_flags_11_10.csv"):
    path = INDIR_SBA / filename
    df = pd.read_csv(path)
    cols = ["loan_cert", "borrower_county", "year", "n_loans", "n_loans_past5", "dist_mean", "dist_median", "dist_mean_prev5",
        "dist_median_prev5", "dist_mean_next5", "dist_median_next5", "pif_rate", "default_rate", "bank_size",
        "branches_prev5", "n_branches_prev5", "branches_5yr", "n_branches_5yr",
        "any_closure", "any_exog_closure", "any_true_exog", "closure_branch_ids", "exog_closure_branch_ids","true_exog_closure_branch_ids"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    list_cols = [
        "branches_prev5",
        "branches_5yr",
        "closure_branch_ids",
        "exog_closure_branch_ids",
        "true_exog_closure_branch_ids",
    ]
    for col in list_cols:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else []
            )

    return df

def get_bank_county_covariates(filename="sod_bank_county_year_transformations_closures.csv"):
    path = INDIR_SBA / filename
    sod_bcy = pd.read_csv(path)
    cols = ["final_CERT", "STCNTYBR", "YEAR",
        "n_branches_total", "n_branches_flagged",
        "n_pred_branches", "n_succ_branches", "n_both_branches",
        "n_closure_branches", "n_exog_closure_branches",
        "n_true_exog_closure_branches",
        "has_transformation", "has_pred", "has_succ", "has_both",
        "has_closure", "has_exog_closure", "has_true_exog_closure",
        "SBA_YEAR"]
    sod_bcy = sod_bcy[cols]
    id_vars = ["STCNTYBR", "YEAR"]
    value_vars = [c for c in sod_bcy.columns if c not in id_vars + ["final_CERT"]]
    sod_cy = (sod_bcy .groupby(id_vars)[value_vars].sum().reset_index())
    return sod_bcy, sod_cy

def collapse_sba_to_county_year(df):
    """
    Collapse bank–county–year SBA data to county–year level.

    Uses:
      - mean for continuous variables (distance, rates, growth)
      - sum for n_loans
      - max for closure flags (was closure ANYWHERE in county this year?)
    """
    df = df.copy()

    if "borrower_county" in df.columns and "county" not in df.columns:
        df = df.rename(columns={"borrower_county": "county"})

    id_vars = ["county", "year"]

    # columns to average
    mean_vars = [
        "dist_mean", "dist_median",
        "default_rate", "pif_rate",
        "loan_growth",
        "dist_mean_prev5", "dist_mean_next5",
        "dist_median_prev5", "dist_median_next5",
        "default_rate_prev5", "default_rate_next5",
        "pif_rate_prev5", "pif_rate_next5",
        "loan_growth_prev5", "loan_growth_next5",
        # add more if needed
    ]
    mean_vars = [c for c in mean_vars if c in df.columns]

    # columns to sum
    sum_vars = ["n_loans"]
    sum_vars = [c for c in sum_vars if c in df.columns]

    # closure flags: max() = "did any bank have a closure in this county-year?"
    max_vars = ["any_closure", "any_exog_closure", "any_true_exog"]
    max_vars = [c for c in max_vars if c in df.columns]

    agg_dict = {c: "mean" for c in mean_vars}
    agg_dict.update({c: "sum" for c in sum_vars})
    agg_dict.update({c: "max" for c in max_vars})

    df_cy = df.groupby(id_vars).agg(agg_dict).reset_index()
    return df_cy

def add_branch_numbers(df):
    df = df.copy()
    df = df.sort_values(["loan_cert", "borrower_county", "year"]).reset_index(drop=True)
    def to_list(x):
        if isinstance(x, list):
            return x
        if pd.isna(x):
            return []
        if isinstance(x, str):
            x = x.strip()
            if x.startswith("["):
                try:
                    return ast.literal_eval(x)
                except Exception:
                    return []
            return [] if x == "" else [x]
        return [x]
    df["closure_branch_ids"] = df["closure_branch_ids"].apply(to_list)
    past5_list = []
    next5_list = []
    for (_, _), g in df.groupby(["loan_cert", "borrower_county"], sort=False):
        years = g["year"].tolist()
        branch_lists = g["closure_branch_ids"].tolist()
        gp_past5 = []
        gp_next5 = []
        for i, yr in enumerate(years):
            # Past 5 years: [yr-5 .. yr-1]
            past_ids = set()
            for j, y2 in enumerate(years):
                if yr - 5 <= y2 < yr:
                    past_ids.update(branch_lists[j])
            gp_past5.append(sorted(past_ids))
            # Next 5 years: [yr .. yr+4]
            next_ids = set()
            for j, y2 in enumerate(years):
                if yr <= y2 <= yr + 4:
                    next_ids.update(branch_lists[j])
            gp_next5.append(sorted(next_ids))
        past5_list.extend(gp_past5)
        next5_list.extend(gp_next5)
    df["branches_past5"] = past5_list
    df["n_branches_past5"] = df["branches_past5"].apply(len)
    df["branches_next5"] = next5_list
    df["n_branches_next5"] = df["branches_next5"].apply(len)
    return df

def get_summary_stats(sba_bcy, sba_cy):
    bcy = sba_bcy.copy()
    cy = sba_cy.copy()
    if {"loan_cert", "borrower_county", "year", "n_loans"}.issubset(bcy.columns):
        bcy = bcy.sort_values(["loan_cert", "borrower_county", "year"])
        bcy["loan_growth"] = (
            bcy.groupby(["loan_cert", "borrower_county"])["n_loans"]
               .pct_change()
        )
    id_like = {
        "loan_cert", "borrower_county",
        "final_CERT", "STCNTYBR",
        "year", "YEAR", "SBA_YEAR",
    }
    num_cols_bcy = [c for c in bcy.columns
        if c not in id_like and pd.api.types.is_numeric_dtype(bcy[c])]
    num_cols_cy = [c for c in cy.columns
        if c not in id_like and pd.api.types.is_numeric_dtype(cy[c])]
    stats_bcy = bcy[num_cols_bcy].describe().T
    stats_cy = cy[num_cols_cy].describe().T
    stats_bcy.to_csv(INDIR_SBA / "stats_sba_bcy.csv")
    stats_cy.to_csv(INDIR_SBA / "stats_sba_cy.csv")
    print("Saved Summary Stats to source/transformations/data")
    return stats_bcy, stats_cy

def _summarize_by_flag(df, flag_col, id_like=None):
    if id_like is None:
        id_like = []
    df = df.copy()
    num_cols = [
        c for c in df.columns
        if c not in id_like + [flag_col]
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    summary = df.groupby(flag_col)[num_cols].agg(["mean", "std", "count"]).T
    return summary

def get_closure_comparison_stats(sba_bcy, sba_cy):
    bcy = sba_bcy.copy()
    cy = sba_cy.copy()
    id_like_bcy = ["loan_cert", "borrower_county", "year"]
    id_like_cy = ["borrower_county", "year"]
    stats_bcy = _summarize_by_flag(bcy, flag_col="any_closure", id_like=id_like_bcy)
    stats_cy = _summarize_by_flag(cy, flag_col="any_closure", id_like=id_like_cy)
    stats_bcy.to_csv(INDIR_SBA / "stats_sba_cl_bcy.csv")
    stats_cy.to_csv(INDIR_SBA / "stats_sba_cl_cy.csv")
    print("Saved bank-county-year and county-year closure summary stats to source/transformations/data")
    return stats_bcy, stats_cy

def get_exog_closure_comparison_stats(sba_bcy, sba_cy):
    bcy = sba_bcy.copy()
    cy = sba_cy.copy()

    id_like_bcy = ["loan_cert", "borrower_county", "year"]
    id_like_cy = ["borrower_county", "year"]

    stats_bcy = _summarize_by_flag(bcy, flag_col="any_exog_closure", id_like=id_like_bcy)
    stats_cy = _summarize_by_flag(cy, flag_col="any_exog_closure", id_like=id_like_cy)

    stats_bcy.to_csv(INDIR_SBA / "stats_sba_exog_cl_bcy.csv")
    stats_cy.to_csv(INDIR_SBA / "stats_sba_exog_cl_cy.csv")
    print("Saved exogenous-closure summary stats to source/transformations/data")

    return stats_bcy, stats_cy


def get_true_exog_closure_comparison_stats(sba_bcy, sba_cy):
    bcy = sba_bcy.copy()
    cy = sba_cy.copy()

    id_like_bcy = ["loan_cert", "borrower_county", "year"]
    id_like_cy = ["borrower_county", "year"]

    stats_bcy = _summarize_by_flag(bcy, flag_col="any_true_exog", id_like=id_like_bcy)
    stats_cy = _summarize_by_flag(cy, flag_col="any_true_exog", id_like=id_like_cy)

    stats_bcy.to_csv(INDIR_SBA / "stats_sba_true_exog_cl_bcy.csv")
    stats_cy.to_csv(INDIR_SBA / "stats_sba_true_exog_cl_cy.csv")
    print("Saved true-exogenous-closure summary stats to source/transformations/data")

    return stats_bcy, stats_cy

def _build_pretty_table(stats_all, stats_cl, stats_ex, stats_tex, key_vars):
    """
    Build a compact comparison table for a single panel
    using:
      - stats_all: overall describe().T  (from get_summary_stats)
      - stats_cl:  closure splits        (from _summarize_by_flag on any_closure)
      - stats_ex:  exog closure splits   (any_exog_closure)
      - stats_tex: true exog splits      (any_true_exog)
    """
    idx = pd.IndexSlice
    rows = {}

    for var in key_vars:
        row = {}

        # overall
        if var in stats_all.index:
            row["All_mean"] = stats_all.loc[var, "mean"]
            row["All_count"] = stats_all.loc[var, "count"]
        else:
            row["All_mean"] = pd.NA
            row["All_count"] = pd.NA

        # closure: flag 0 vs 1
        if (var, "mean") in stats_cl.index:
            # handle case where only 0 or only 1 exists
            for flag in [0, 1]:
                col_suffix = f"Cl{flag}"
                if flag in stats_cl.columns:
                    row[f"{col_suffix}_mean"] = stats_cl.loc[(var, "mean"), flag]
                    row[f"{col_suffix}_count"] = stats_cl.loc[(var, "count"), flag]
                else:
                    row[f"{col_suffix}_mean"] = pd.NA
                    row[f"{col_suffix}_count"] = pd.NA

        # exogenous closure: flag 1
        if (var, "mean") in stats_ex.index and 1 in stats_ex.columns:
            row["Exog1_mean"] = stats_ex.loc[(var, "mean"), 1]
            row["Exog1_count"] = stats_ex.loc[(var, "count"), 1]
        else:
            row["Exog1_mean"] = pd.NA
            row["Exog1_count"] = pd.NA

        # true exogenous closure: flag 1
        if (var, "mean") in stats_tex.index and 1 in stats_tex.columns:
            row["TrueExog1_mean"] = stats_tex.loc[(var, "mean"), 1]
            row["TrueExog1_count"] = stats_tex.loc[(var, "count"), 1]
        else:
            row["TrueExog1_mean"] = pd.NA
            row["TrueExog1_count"] = pd.NA

        rows[var] = row

    pretty = pd.DataFrame.from_dict(rows, orient="index")

    # nice ordering of columns
    col_order = [
        "All_mean", "All_count",
        "Cl0_mean", "Cl0_count",
        "Cl1_mean", "Cl1_count",
        "Exog1_mean", "Exog1_count",
        "TrueExog1_mean", "TrueExog1_count",
    ]
    pretty = pretty[[c for c in col_order if c in pretty.columns]]

    return pretty

def make_pretty_summary_tables(
    stats_bcy, stats_cy,
    stats_bcy_cl, stats_cy_cl,
    stats_bcy_ex, stats_cy_ex,
    stats_bcy_tex, stats_cy_tex,
):
    """
    Build compact, readable summary tables for:
      - bank–county–year (sba_bcy)
      - county–year      (sba_cy)

    Each table has:
      - up to 5 key variables (you can customize below)
      - columns for overall, closure splits, exog, true exog
    """

    # choose up to 5 key vars that actually exist
    candidate_bcy = [
        "n_loans",
        "n_loans_past5",
        "bank_size",
        "n_branches_past5",
        "n_branches_next5",
    ]
    key_vars_bcy = [v for v in candidate_bcy if v in stats_bcy.index][:5]

    candidate_cy = [
        "n_loans",
        "n_loans_past5",
        "bank_size",
        "n_closures",
        "n_banks",
    ]
    key_vars_cy = [v for v in candidate_cy if v in stats_cy.index][:5]

    pretty_bcy = _build_pretty_table(
        stats_all=stats_bcy,
        stats_cl=stats_bcy_cl,
        stats_ex=stats_bcy_ex,
        stats_tex=stats_bcy_tex,
        key_vars=key_vars_bcy,
    ).round(2)

    pretty_cy = _build_pretty_table(
        stats_all=stats_cy,
        stats_cl=stats_cy_cl,
        stats_ex=stats_cy_ex,
        stats_tex=stats_cy_tex,
        key_vars=key_vars_cy,
    ).round(2)

    pretty_bcy.to_csv(INDIR_SBA / "pretty_sba_bcy_summary.csv")
    pretty_cy.to_csv(INDIR_SBA / "pretty_sba_cy_summary.csv")

    print("Saved Pretty Summary Table")
    return pretty_bcy, pretty_cy


def add_prev_next_windows(df):
    """
    Add *_prev5 and *_next5 for selected variables in the SBA bank–county–year panel.

    For each bank×county (loan_cert, borrower_county) and year t:
      base_prev5 = mean(base_s) over years s in [t-5, t-1]
      base_next5 = mean(base_s) over years s in [t,   t+4]

    Also (re)computes loan_growth = pct change of n_loans within bank×county.
    """
    df = df.copy()

    # normalize county label
    if "borrower_county" in df.columns and "county" not in df.columns:
        df = df.rename(columns={"borrower_county": "county"})

    # sort for stable group operations
    df = df.sort_values(["loan_cert", "county", "year"]).reset_index(drop=True)

    # 1) loan_growth: Δ log-ish n_loans (actually pct change)
    if "n_loans" in df.columns:
        df["loan_growth"] = (
            df.groupby(["loan_cert", "county"])["n_loans"]
            .pct_change()
        )

    # 2) choose base variables we want to window
    #    (only if they exist AND we don't already have prev/next versions)
    candidate_bases = [
        "default_rate",
        "pif_rate",
        "n_loans",
        "loan_growth",
        # add more here if you like
    ]

    base_vars = []
    for base in candidate_bases:
        if base not in df.columns:
            continue
        prev_name = f"{base}_prev5"
        next_name = f"{base}_next5"
        # if you already computed these elsewhere, don't overwrite
        if prev_name in df.columns or next_name in df.columns:
            continue
        base_vars.append(base)

    if not base_vars:
        return df  # nothing to do

    # prepare containers for new columns
    new_cols = {}
    for base in base_vars:
        new_cols[f"{base}_prev5"] = []
        new_cols[f"{base}_next5"] = []

    # 3) loop over bank×county groups and compute window means
    for (_, _), g in df.groupby(["loan_cert", "county"], sort=False):
        years = g["year"].to_numpy()

        for base in base_vars:
            vals = g[base].to_numpy()
            prev_list = []
            next_list = []

            for i, yr in enumerate(years):
                # previous 5 years: [yr-5, yr-1]
                prev_mask = (years >= yr - 5) & (years < yr)
                # next 5 years: [yr, yr+4]
                next_mask = (years >= yr) & (years <= yr + 4)

                prev_mean = np.nanmean(vals[prev_mask]) if prev_mask.any() else np.nan
                next_mean = np.nanmean(vals[next_mask]) if next_mask.any() else np.nan

                prev_list.append(prev_mean)
                next_list.append(next_mean)

            new_cols[f"{base}_prev5"].extend(prev_list)
            new_cols[f"{base}_next5"].extend(next_list)

    # 4) attach new columns
    for col, values in new_cols.items():
        df[col] = values

    return df

def add_loan_growth_and_windows(df):
    df = df.copy()
    if "borrower_county" in df.columns and "county" not in df.columns:
        df = df.rename(columns={"borrower_county": "county"})
    df = df.sort_values(["loan_cert", "county", "year"]).reset_index(drop=True)
    if "n_loans" in df.columns:
        df["loan_growth"] = (
            df.groupby(["loan_cert", "county"])["n_loans"]
              .pct_change()
        )
    base_vars = [
        "dist_mean",
        "dist_median",
        "default_rate",
        "pif_rate",
        "n_loans",
        "loan_growth",
    ]
    for base in base_vars:
        prev_name = f"{base}_prev5"
        next_name = f"{base}_next5"

        if prev_name in df.columns or next_name in df.columns:
            continue

        df[prev_name] = (
            df.groupby(["loan_cert", "county"])[base]
              .apply(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
              .reset_index(level=[0,1], drop=True)
        )

        df[next_name] = (
            df.groupby(["loan_cert", "county"])[base]
              .apply(lambda x: x.rolling(5, min_periods=1).mean())
              .reset_index(level=[0,1], drop=True)
        )

    return df



if __name__ == "__main__":
    main()
