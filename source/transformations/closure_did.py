import pandas as pd
from pathlib import Path
import statsmodels.api as sm
from clean_closure_data import load_data, add_prev_next_windows
from linearmodels.iv import AbsorbingLS
import numpy as np
import datetime


INDIR_SBA = Path("source/transformations/data")
OUTDIR_RESULTS = Path("output/transformations/did_results")
LOG_PATH = OUTDIR_RESULTS.joinpath("did_results.log")

def log_write(text):
    """Append text to a master regression log file."""
    with open(LOG_PATH, "a") as f:
        f.write(text + "\n")


def did_bank_county_year(df, input_var, outcome_var):
    """
    DID at bank×county×year level using AbsorbingLS.

    FE structure:
      - Bank FE:   loan_cert
      - County FE: county
      - Year FE:   year

    Spec (if prev var exists):
        outcome_next5 ~ prev5 + input_var + bank FE + county FE + year FE
    """

    print("\n----------------------------------------")
    print(f"[BANK×COUNTY×YEAR] Outcome: {outcome_var}, Treatment: {input_var}")
    print("----------------------------------------")

    df_work = df.copy()

    # standardize county name
    if "borrower_county" in df_work.columns and "county" not in df_work.columns:
        df_work = df_work.rename(columns={"borrower_county": "county"})

    # prev5 control if available
    prev_var = None
    if outcome_var.endswith("next5"):
        candidate_prev = outcome_var.replace("next5", "prev5")
        if candidate_prev in df_work.columns:
            prev_var = candidate_prev

    needed = ["loan_cert", "county", "year", input_var, outcome_var]
    if prev_var is not None:
        needed.append(prev_var)

    df_sub = df_work[needed].dropna().copy()

    # IDs as categories / numeric labels
    df_sub["loan_cert"] = df_sub["loan_cert"].astype(str)
    df_sub["county"] = df_sub["county"].astype(str)

    # dependent variable
    y = df_sub[outcome_var].astype(float)

    # exogenous vars: const + treatment (+ prev if present)
    exog_cols = [input_var]
    if prev_var is not None:
        exog_cols.append(prev_var)

    X = sm.add_constant(df_sub[exog_cols].astype(float))

    # absorbing FEs: bank, county, year
    absorb = df_sub[["loan_cert", "county", "year"]].astype("category")

    # cluster by bank×county pair (or change to loan_cert or county if you prefer)
    clusters = (
        df_sub["loan_cert"].astype(str) + "|" + df_sub["county"].astype(str)
    ).astype("category").cat.codes

    mod = AbsorbingLS(y, X, absorb=absorb)
    res = mod.fit(cov_type="clustered", clusters=clusters)

    print(res.summary)

    # log full table
    header = (
        "\n\n============================================\n"
        f"[BANK×COUNTY×YEAR]\n"
        f"Outcome: {outcome_var}\n"
        f"Treatment: {input_var}\n"
        f"Timestamp: {datetime.datetime.now()}\n"
        "============================================\n"
    )
    log_write(header)
    log_write(str(res.summary))

    # compact summary row (no interaction)
    def safe(series, name):
        return series[name] if name in series.index else np.nan

    row = {
        "level": "bank_county_year",
        "outcome": outcome_var,
        "treatment": input_var,
        "prev_var": prev_var,
        "coef_treat": safe(res.params, input_var),
        "se_treat": safe(res.std_errors, input_var),
        "pval_treat": safe(res.pvalues, input_var),
        "coef_prev": safe(res.params, prev_var) if prev_var else np.nan,
        "se_prev": safe(res.std_errors, prev_var) if prev_var else np.nan,
        "pval_prev": safe(res.pvalues, prev_var) if prev_var else np.nan,
        "nobs": res.nobs,
        "rsq": res.rsquared,
    }

    return res, row


def did_county_year(df, input_var, outcome_var):
    """
    DID at county×year level using AbsorbingLS.

    FE structure:
      - County FE: county
      - Year FE:   year

    Spec (if prev var exists):
        outcome_next5 ~ prev5 + input_var + county FE + year FE
    """

    print("\n----------------------------------------")
    print(f"[COUNTY×YEAR] Outcome: {outcome_var}, Treatment: {input_var}")
    print("----------------------------------------")

    df_work = df.copy()

    if "borrower_county" in df_work.columns and "county" not in df_work.columns:
        df_work = df_work.rename(columns={"borrower_county": "county"})

    prev_var = None
    if outcome_var.endswith("next5"):
        candidate_prev = outcome_var.replace("next5", "prev5")
        if candidate_prev in df_work.columns:
            prev_var = candidate_prev

    needed = ["county", "year", input_var, outcome_var]
    if prev_var is not None:
        needed.append(prev_var)

    df_sub = df_work[needed].dropna().copy()

    df_sub["county"] = df_sub["county"].astype(str)

    y = df_sub[outcome_var].astype(float)

    exog_cols = [input_var]
    if prev_var is not None:
        exog_cols.append(prev_var)
    X = sm.add_constant(df_sub[exog_cols].astype(float))

    absorb = df_sub[["county", "year"]].astype("category")
    clusters = df_sub["county"].astype("category").cat.codes

    mod = AbsorbingLS(y, X, absorb=absorb)
    res = mod.fit(cov_type="clustered", clusters=clusters)

    print(res.summary)

    header = (
        "\n\n============================================\n"
        f"[COUNTY×YEAR]\n"
        f"Outcome: {outcome_var}\n"
        f"Treatment: {input_var}\n"
        f"Timestamp: {datetime.datetime.now()}\n"
        "============================================\n"
    )
    log_write(header)
    log_write(str(res.summary))

    def safe(series, name):
        return series[name] if name in series.index else np.nan

    row = {
        "level": "county_year",
        "outcome": outcome_var,
        "treatment": input_var,
        "prev_var": prev_var,
        "coef_treat": safe(res.params, input_var),
        "se_treat": safe(res.std_errors, input_var),
        "pval_treat": safe(res.pvalues, input_var),
        "coef_prev": safe(res.params, prev_var) if prev_var else np.nan,
        "se_prev": safe(res.std_errors, prev_var) if prev_var else np.nan,
        "pval_prev": safe(res.pvalues, prev_var) if prev_var else np.nan,
        "nobs": res.nobs,
        "rsq": res.rsquared,
    }

    return res, row


def run_bcy_dids(sba_bcy):
    """
    Run DID specs on the bank–county–year SBA panel and save results.
    """
    print("\n==================== BANK × COUNTY × YEAR (SBA) ====================")

    # choose bases; we'll look for base_next5 & base_prev5
    base_vars = [
        "dist_mean",
        "dist_median",
        "default_rate",
        "pif_rate",
        "n_loans",
    ]

    treatments = [
        "any_closure",
        "any_exog_closure",
        "any_true_exog",
    ]
    treatments = [t for t in treatments if t in sba_bcy.columns]

    results_rows = []

    for base in base_vars:
        outcome = f"{base}_next5"
        prev = f"{base}_prev5"

        # only run if both next5 and prev5 exist
        if outcome not in sba_bcy.columns or prev not in sba_bcy.columns:
            continue

        for tr in treatments:
            res, row = did_bank_county_year(sba_bcy, input_var=tr, outcome_var=outcome)
            row["base_var"] = base
            results_rows.append(row)

    if results_rows:
        results_df = pd.DataFrame(results_rows)
        out_path = OUTDIR_RESULTS / "sba_did_results_bcy.csv"
        results_df.to_csv(out_path, index=False)
        print(f"\nSaved bank–county–year DID results to {out_path}")
    else:
        print("\nNo bank–county–year DIDs were run (no matching _next5/_prev5 pairs).")


def run_cy_dids(sba_cy):
    """
    Run DID specs on the county–year SBA panel and save results.
    """
    print("\n========================= COUNTY × YEAR (SBA) =======================")

    base_vars = [
        "dist_mean",
        "dist_median",
        "default_rate",
        "pif_rate",
        "n_loans",
    ]

    treatments = [
        "any_closure",
        "any_exog_closure",
        "any_true_exog",
    ]
    treatments = [t for t in treatments if t in sba_cy.columns]

    results_rows = []

    for base in base_vars:
        outcome = f"{base}_next5"
        prev = f"{base}_prev5"

        if outcome not in sba_cy.columns or prev not in sba_cy.columns:
            continue

        for tr in treatments:
            res, row = did_county_year(sba_cy, input_var=tr, outcome_var=outcome)
            row["base_var"] = base
            results_rows.append(row)

    if results_rows:
        results_df = pd.DataFrame(results_rows)
        out_path = OUTDIR_RESULTS / "sba_did_results_cy.csv"
        results_df.to_csv(out_path, index=False)
        print(f"\nSaved county–year DID results to {out_path}")
    else:
        print("\nNo county–year DIDs were run (no matching _next5/_prev5 pairs).")

def main():
    sba_bcy, sba_cy, sod_bcy, sod_cy = load_data()
    run_bcy_dids(sba_bcy)
    run_cy_dids(sba_cy)


if __name__ == "__main__":
    open(LOG_PATH, "w").close()
    main()
