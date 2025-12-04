import pandas as pd
from pathlib import Path
import statsmodels.api as sm
from linearmodels.iv import AbsorbingLS
import numpy as np
import datetime

INDIR_SBA = Path("source/sba_analysis/data/derived")
DID_DATA = Path("source/transformations/data")
DID_DATA_SB = Path("output/transformations/did_result")

OUTDIR_RESULTS = Path("output/transformations/did_results")
LOG_PATH = OUTDIR_RESULTS.joinpath("did_results_with_market_share.log")
OUTDIR_RESULTS.mkdir(parents=True, exist_ok=True)

BANK_COL = "loan_cert"
COUNTY_COL = "borrower_county"   # will be renamed to 'county' inside DID fns
YEAR_COL = "year"

def log_write(text):
    """Append text to a master regression log file."""
    with open(LOG_PATH, "a") as f:
        f.write(text + "\n")


# ============================================================
# Helpers to build prev/next 5-year windows
# ============================================================
def add_lead_lag_windows(df, unit_cols, base_vars, window=5):
    """
    For each base_var, add:
      var_prev{window}: mean over previous `window` years (t-1 .. t-window)
      var_next{window}: mean over next `window` years (t+1 .. t+window)
    """
    print(f"[add_lead_lag_windows] Start | units={unit_cols}, window={window}")
    print(f"[add_lead_lag_windows] Incoming shape: {df.shape}")

    df = df.copy()
    if YEAR_COL not in df.columns:
        raise ValueError(f"Year column '{YEAR_COL}' not in DataFrame.")

    df[YEAR_COL] = df[YEAR_COL].astype(int)
    df = df.sort_values(unit_cols + [YEAR_COL])

    for var in base_vars:
        if var not in df.columns:
            print(f"[add_lead_lag_windows] Skipping {var} (not in columns)")
            continue

        print(f"[add_lead_lag_windows] Computing prev/next {window} for {var}...")
        g = df.groupby(unit_cols)[var]

        prev = g.apply(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
        prev = prev.reset_index(level=unit_cols, drop=True)
        df[f"{var}_prev{window}"] = prev

        next_ = g.apply(
            lambda s: s[::-1].shift(1).rolling(window, min_periods=1).mean()[::-1]
        )
        next_ = next_.reset_index(level=unit_cols, drop=True)
        df[f"{var}_next{window}"] = next_

    print(f"[add_lead_lag_windows] Done | outgoing shape: {df.shape}")
    return df


def prepare_bcy_for_did(sba_bcy):
    """
    Make sure bcy panel has the base vars needed and add prev/next5 windows.
    """
    print("\n[prepare_bcy_for_did] Starting BCY prep...")
    print(f"[prepare_bcy_for_did] Incoming shape: {sba_bcy.shape}")

    df = sba_bcy.copy()

    # unify county col name used here
    if COUNTY_COL not in df.columns and "county" in df.columns:
        print("[prepare_bcy_for_did] Renaming 'county' -> borrower_county")
        df = df.rename(columns={"county": COUNTY_COL})

    # distance variables
    if "avg_distance_miles" in df.columns and "dist_mean" not in df.columns:
        print("[prepare_bcy_for_did] Creating dist_mean from avg_distance_miles")
        df["dist_mean"] = df["avg_distance_miles"]

    if "med_distance_miles" in df.columns and "dist_median" not in df.columns:
        print("[prepare_bcy_for_did] Creating dist_median from med_distance_miles")
        df["dist_median"] = df["med_distance_miles"]

    # pif_rate
    if "pif_rate" not in df.columns and "pct_pif" in df.columns:
        print("[prepare_bcy_for_did] Creating pif_rate from pct_pif")
        df["pif_rate"] = df["pct_pif"]

    # loan_growth at BCY: growth in n_loans within bank×county
    if "loan_growth" not in df.columns and "n_loans" in df.columns:
        print("[prepare_bcy_for_did] Computing loan_growth (pct_change of n_loans)")
        df = df.sort_values([BANK_COL, COUNTY_COL, YEAR_COL])
        df["loan_growth"] = (
            df.groupby([BANK_COL, COUNTY_COL])["n_loans"]
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
    print(f"[prepare_bcy_for_did] Base vars for windows: {base_vars}")

    df = add_lead_lag_windows(
        df,
        unit_cols=[BANK_COL, COUNTY_COL],
        base_vars=base_vars,
        window=5,
    )

    print(f"[prepare_bcy_for_did] Done | outgoing shape: {df.shape}")
    return df


def prepare_cy_for_did(sba_cy):
    """
    Make sure cy panel has the base vars needed and add prev/next5 windows.
    """
    print("\n[prepare_cy_for_did] Starting CY prep...")
    print(f"[prepare_cy_for_did] Incoming shape: {sba_cy.shape}")

    df = sba_cy.copy()

    if COUNTY_COL not in df.columns and "county" in df.columns:
        print("[prepare_cy_for_did] Renaming 'county' -> borrower_county")
        df = df.rename(columns={"county": COUNTY_COL})

    if "avg_distance_miles" in df.columns and "dist_mean" not in df.columns:
        print("[prepare_cy_for_did] Creating dist_mean from avg_distance_miles")
        df["dist_mean"] = df["avg_distance_miles"]

    if "med_distance_miles" in df.columns and "dist_median" not in df.columns:
        print("[prepare_cy_for_did] Creating dist_median from med_distance_miles")
        df["dist_median"] = df["med_distance_miles"]

    if "pif_rate" not in df.columns and "pct_pif" in df.columns:
        print("[prepare_cy_for_did] Creating pif_rate from pct_pif")
        df["pif_rate"] = df["pct_pif"]

    if "loan_growth" not in df.columns and "n_loans" in df.columns:
        print("[prepare_cy_for_did] Computing loan_growth (pct_change of n_loans)")
        df = df.sort_values([COUNTY_COL, YEAR_COL])
        df["loan_growth"] = (
            df.groupby(COUNTY_COL)["n_loans"]
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
    print(f"[prepare_cy_for_did] Base vars for windows: {base_vars}")

    df = add_lead_lag_windows(
        df,
        unit_cols=[COUNTY_COL],
        base_vars=base_vars,
        window=5,
    )

    print(f"[prepare_cy_for_did] Done | outgoing shape: {df.shape}")
    return df


# ============================================================
# DID: bank×county×year
# ============================================================
def did_bank_county_year(df, input_var, outcome_var):
    """
    DID at bank×county×year level using AbsorbingLS.
    """
    print("\n----------------------------------------")
    print(f"[BANK×COUNTY×YEAR] Outcome: {outcome_var}, Treatment: {input_var}")
    print("----------------------------------------")

    df_work = df.copy()

    if "borrower_county" in df_work.columns and "county" not in df_work.columns:
        df_work = df_work.rename(columns={"borrower_county": "county"})

    prev_var = None
    if outcome_var.endswith("next5"):
        candidate_prev = outcome_var.replace("next5", "prev5")
        if candidate_prev in df_work.columns:
            prev_var = candidate_prev

    needed = ["loan_cert", "county", "year", input_var, outcome_var]
    if prev_var is not None:
        needed.append(prev_var)

    print(f"[did_bank_county_year] Needed cols: {needed}")
    df_sub = df_work[needed].dropna().copy()
    print(f"[did_bank_county_year] Sample size after dropna: {df_sub.shape[0]}")

    df_sub["loan_cert"] = df_sub["loan_cert"].astype(str)
    df_sub["county"] = df_sub["county"].astype(str)

    y = df_sub[outcome_var].astype(float)

    exog_cols = [input_var]
    if prev_var is not None:
        exog_cols.append(prev_var)
    print(f"[did_bank_county_year] Exogenous vars: {exog_cols}")

    X = sm.add_constant(df_sub[exog_cols].astype(float))

    absorb = df_sub[["loan_cert", "county", "year"]].astype("category")
    clusters = (
        df_sub["loan_cert"].astype(str) + "|" + df_sub["county"].astype(str)
    ).astype("category").cat.codes

    print("[did_bank_county_year] Fitting AbsorbingLS...")
    mod = AbsorbingLS(y, X, absorb=absorb)
    res = mod.fit(cov_type="clustered", clusters=clusters)
    print("[did_bank_county_year] Fit complete.")

    print(res.summary)

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


# ============================================================
# DID: county×year
# ============================================================
def did_county_year(df, input_var, outcome_var):
    """
    DID at county×year level using AbsorbingLS.
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

    print(f"[did_county_year] Needed cols: {needed}")
    df_sub = df_work[needed].dropna().copy()
    print(f"[did_county_year] Sample size after dropna: {df_sub.shape[0]}")

    df_sub["county"] = df_sub["county"].astype(str)

    y = df_sub[outcome_var].astype(float)

    exog_cols = [input_var]
    if prev_var is not None:
        exog_cols.append(prev_var)
    print(f"[did_county_year] Exogenous vars: {exog_cols}")

    X = sm.add_constant(df_sub[exog_cols].astype(float))

    absorb = df_sub[["county", "year"]].astype("category")
    clusters = df_sub["county"].astype("category").cat.codes

    print("[did_county_year] Fitting AbsorbingLS...")
    mod = AbsorbingLS(y, X, absorb=absorb)
    res = mod.fit(cov_type="clustered", clusters=clusters)
    print("[did_county_year] Fit complete.")

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


# ============================================================
# Load SBA panels & prepare for DID
# ============================================================
def load_sba_panels():
    print("[load_sba_panels] Reading BCY and CY panels...")
    print(f"[load_sba_panels] INDIR_SBA = {INDIR_SBA}")

    sba_bcy = pd.read_csv(INDIR_SBA / "sba_bcy_market_share_flags.csv")
    sba_cy = pd.read_csv(INDIR_SBA / "sba_cy_market_share_flags.csv")

    print(f"[load_sba_panels] Raw BCY shape: {sba_bcy.shape}")
    print(f"[load_sba_panels] Raw CY shape: {sba_cy.shape}")

    sba_bcy = prepare_bcy_for_did(sba_bcy)
    sba_cy = prepare_cy_for_did(sba_cy)

    print(f"[load_sba_panels] Prepared BCY shape: {sba_bcy.shape}")
    print(f"[load_sba_panels] Prepared CY shape: {sba_cy.shape}")

    return sba_bcy, sba_cy


# ============================================================
# Run DID specs
# ============================================================
def run_bcy_dids(sba_bcy):
    """
    Run DID specs on the bank–county–year SBA panel and save results.
    """
    print("\n==================== BANK × COUNTY × YEAR (SBA) ====================")
    print(f"[run_bcy_dids] Panel shape: {sba_bcy.shape}")

    base_vars = [
        "dist_mean",
        "dist_median",
        "default_rate",
        "pif_rate",
        "n_loans",
        "loan_growth",
    ]

    treatments = [
        "any_closure",
        "any_exog_closure",
        "any_true_exog",
    ]
    treatments = [t for t in treatments if t in sba_bcy.columns]
    print(f"[run_bcy_dids] Treatments present: {treatments}")

    results_rows = []
    n_runs = 0

    for base in base_vars:
        outcome = f"{base}_next5"
        prev = f"{base}_prev5"

        if outcome not in sba_bcy.columns or prev not in sba_bcy.columns:
            print(f"[run_bcy_dids] Skipping base={base} (no {outcome} or {prev})")
            continue

        for tr in treatments:
            print(f"[run_bcy_dids] Running DID for base={base}, treat={tr}")
            res, row = did_bank_county_year(sba_bcy, input_var=tr, outcome_var=outcome)
            row["base_var"] = base
            results_rows.append(row)
            n_runs += 1

    print(f"[run_bcy_dids] Total DID regressions run (BCY): {n_runs}")

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
    print(f"[run_cy_dids] Panel shape: {sba_cy.shape}")

    base_vars = [
        "dist_mean",
        "dist_median",
        "default_rate",
        "pif_rate",
        "n_loans",
        "loan_growth",
    ]
    treatments = [
        "any_closure",
        "any_exog_closure",
        "any_true_exog",
    ]
    treatments = [t for t in treatments if t in sba_cy.columns]
    print(f"[run_cy_dids] Treatments present: {treatments}")

    results_rows = []
    n_runs = 0

    for base in base_vars:
        outcome = f"{base}_next5"
        prev = f"{base}_prev5"

        if outcome not in sba_cy.columns or prev not in sba_cy.columns:
            print(f"[run_cy_dids] Skipping base={base} (no {outcome} or {prev})")
            continue

        for tr in treatments:
            print(f"[run_cy_dids] Running DID for base={base}, treat={tr}")
            res, row = did_county_year(sba_cy, input_var=tr, outcome_var=outcome)
            row["base_var"] = base
            results_rows.append(row)
            n_runs += 1

    print(f"[run_cy_dids] Total DID regressions run (CY): {n_runs}")

    if results_rows:
        results_df = pd.DataFrame(results_rows)
        out_path = OUTDIR_RESULTS / "sba_did_results_cy.csv"
        results_df.to_csv(out_path, index=False)
        print(f"\nSaved county–year DID results to {out_path}")
    else:
        print("\nNo county–year DIDs were run (no matching _next5/_prev5 pairs).")

def did_bank_county_year_mshare(
    df,
    input_var,
    outcome_var,
    mshare_var="combined_market_share_of_closure",
    restrict_high=False,
    high_indicator="high_market_share_closure",
):
    """
    DID at bank×county×year level with treatment × market share interaction.

    Spec:
        outcome_next5 ~ prev5 + treat + mshare + treat*mshare
                        + bank FE + county FE + year FE

    If restrict_high=True, first restrict to rows where `high_indicator == 1`.
    """
    print("\n----------------------------------------")
    print(f"[BCY mshare] Outcome: {outcome_var}, Treatment: {input_var}, "
          f"mshare: {mshare_var}, restrict_high={restrict_high}")
    print("----------------------------------------")

    df_work = df.copy()

    # standardize county name
    if "borrower_county" in df_work.columns and "county" not in df_work.columns:
        df_work = df_work.rename(columns={"borrower_county": "county"})

    # optionally restrict to high-market-share closure cases
    if restrict_high:
        if high_indicator not in df_work.columns:
            raise ValueError(f"high_indicator '{high_indicator}' not in DataFrame.")
        before = len(df_work)
        df_work = df_work[df_work[high_indicator] == 1].copy()
        print(f"[BCY mshare] Restricting to {high_indicator}==1: {before} -> {len(df_work)} rows")

    # prev5 control if available
    prev_var = None
    if outcome_var.endswith("next5"):
        candidate_prev = outcome_var.replace("next5", "prev5")
        if candidate_prev in df_work.columns:
            prev_var = candidate_prev

    # make sure mshare_var exists
    if mshare_var not in df_work.columns:
        raise ValueError(f"mshare_var '{mshare_var}' not found in DataFrame.")

    inter_name = f"{input_var}_x_{mshare_var}"

    needed = ["loan_cert", "county", "year", input_var, mshare_var, outcome_var]
    if prev_var is not None:
        needed.append(prev_var)

    print(f"[BCY mshare] Needed cols: {needed}")
    df_sub = df_work[needed].dropna().copy()
    print(f"[BCY mshare] Sample size after dropna: {df_sub.shape[0]}")

    # build interaction term
    df_sub[inter_name] = df_sub[input_var] * df_sub[mshare_var]

    df_sub["loan_cert"] = df_sub["loan_cert"].astype(str)
    df_sub["county"] = df_sub["county"].astype(str)

    y = df_sub[outcome_var].astype(float)

    exog_cols = [input_var, mshare_var, inter_name]
    if prev_var is not None:
        exog_cols.append(prev_var)
    print(f"[BCY mshare] Exogenous vars: {exog_cols}")

    X = sm.add_constant(df_sub[exog_cols].astype(float))

    absorb = df_sub[["loan_cert", "county", "year"]].astype("category")
    clusters = (
        df_sub["loan_cert"].astype(str) + "|" + df_sub["county"].astype(str)
    ).astype("category").cat.codes

    print("[BCY mshare] Fitting AbsorbingLS...")
    mod = AbsorbingLS(y, X, absorb=absorb)
    res = mod.fit(cov_type="clustered", clusters=clusters)
    print("[BCY mshare] Fit complete.")

    print(res.summary)

    header = (
        "\n\n============================================\n"
        f"[BANK×COUNTY×YEAR w/ mshare]\n"
        f"Outcome: {outcome_var}\n"
        f"Treatment: {input_var}\n"
        f"Market share var: {mshare_var}\n"
        f"Restrict high: {restrict_high}\n"
        f"Timestamp: {datetime.datetime.now()}\n"
        "============================================\n"
    )
    log_write(header)
    log_write(str(res.summary))

    def safe(series, name):
        return series[name] if name in series.index else np.nan

    row = {
        "level": "bank_county_year_mshare",
        "outcome": outcome_var,
        "treatment": input_var,
        "mshare_var": mshare_var,
        "interaction": inter_name,
        "restrict_high": restrict_high,
        "prev_var": prev_var,

        "coef_treat": safe(res.params, input_var),
        "se_treat": safe(res.std_errors, input_var),
        "pval_treat": safe(res.pvalues, input_var),

        "coef_mshare": safe(res.params, mshare_var),
        "se_mshare": safe(res.std_errors, mshare_var),
        "pval_mshare": safe(res.pvalues, mshare_var),

        "coef_inter": safe(res.params, inter_name),
        "se_inter": safe(res.std_errors, inter_name),
        "pval_inter": safe(res.pvalues, inter_name),

        "coef_prev": safe(res.params, prev_var) if prev_var else np.nan,
        "se_prev": safe(res.std_errors, prev_var) if prev_var else np.nan,
        "pval_prev": safe(res.pvalues, prev_var) if prev_var else np.nan,

        "nobs": res.nobs,
        "rsq": res.rsquared,
    }

    return res, row

def run_bcy_dids_mshare(sba_bcy):
    """
    Run DID specs on the bank–county–year SBA panel with
    treatment × market share interactions, both full sample
    and restricted to high-market-share closures.
    """
    print("\n============ BANK × COUNTY × YEAR (SBA) – market share DIDs ============")
    print(f"[run_bcy_dids_mshare] Panel shape: {sba_bcy.shape}")

    base_vars = [
        "dist_mean",
        "dist_median",
        "default_rate",
        "pif_rate",
        "n_loans",
        "loan_growth",
    ]
    treatments = [
        "any_closure",
        "any_exog_closure",
        "any_true_exog",
    ]
    treatments = [t for t in treatments if t in sba_bcy.columns]
    print(f"[run_bcy_dids_mshare] Treatments present: {treatments}")

    mshare_var = "combined_market_share_of_closure"
    if mshare_var not in sba_bcy.columns:
        print(f"[run_bcy_dids_mshare] {mshare_var} not found, skipping mshare DIDs.")
        return

    results_rows = []
    n_runs = 0

    for base in base_vars:
        outcome = f"{base}_next5"
        prev = f"{base}_prev5"

        if outcome not in sba_bcy.columns or prev not in sba_bcy.columns:
            print(f"[run_bcy_dids_mshare] Skipping base={base} (no {outcome} or {prev})")
            continue

        for tr in treatments:
            # 1) full sample interaction
            print(f"[run_bcy_dids_mshare] Running (full) base={base}, treat={tr}")
            _, row_full = did_bank_county_year_mshare(
                sba_bcy,
                input_var=tr,
                outcome_var=outcome,
                mshare_var=mshare_var,
                restrict_high=False,
            )
            row_full["base_var"] = base
            row_full["sample_type"] = "full"
            results_rows.append(row_full)
            n_runs += 1

            # 2) restricted to high-market-share closures
            if "high_market_share_closure" in sba_bcy.columns:
                print(f"[run_bcy_dids_mshare] Running (high share) base={base}, treat={tr}")
                _, row_high = did_bank_county_year_mshare(
                    sba_bcy,
                    input_var=tr,
                    outcome_var=outcome,
                    mshare_var=mshare_var,
                    restrict_high=True,
                    high_indicator="high_market_share_closure",
                )
                row_high["base_var"] = base
                row_high["sample_type"] = "high_share_only"
                results_rows.append(row_high)
                n_runs += 1

    print(f"[run_bcy_dids_mshare] Total DID regressions run (BCY mshare): {n_runs}")

    if results_rows:
        results_df = pd.DataFrame(results_rows)
        out_path = OUTDIR_RESULTS / "sba_did_results_bcy_mshare.csv"
        results_df.to_csv(out_path, index=False)
        print(f"\nSaved bank–county–year mshare DID results to {out_path}")
    else:
        print("\nNo bank–county–year mshare DIDs were run.")

def did_county_year_mshare(
    df,
    input_var,
    outcome_var,
    mshare_var="combined_market_share_of_closure",
    restrict_high=False,
    high_indicator="high_market_share_closure",
):
    """
    DID at county×year level with treatment × market share interaction.

    Spec:
        outcome_next5 ~ prev5 + treat + mshare + treat*mshare
                        + county FE + year FE

    If restrict_high=True, first restrict to rows where `high_indicator == 1`.
    """
    print("\n----------------------------------------")
    print(f"[CY mshare] Outcome: {outcome_var}, Treatment: {input_var}, "
          f"mshare: {mshare_var}, restrict_high={restrict_high}")
    print("----------------------------------------")

    df_work = df.copy()

    if "borrower_county" in df_work.columns and "county" not in df_work.columns:
        df_work = df_work.rename(columns={"borrower_county": "county"})

    # optionally restrict to high-market-share closure cases
    if restrict_high:
        if high_indicator not in df_work.columns:
            raise ValueError(f"high_indicator '{high_indicator}' not in DataFrame.")
        before = len(df_work)
        df_work = df_work[df_work[high_indicator] == 1].copy()
        print(f"[CY mshare] Restricting to {high_indicator}==1: {before} -> {len(df_work)} rows")

    # prev5 control if available
    prev_var = None
    if outcome_var.endswith("next5"):
        candidate_prev = outcome_var.replace("next5", "prev5")
        if candidate_prev in df_work.columns:
            prev_var = candidate_prev

    # make sure mshare_var exists
    if mshare_var not in df_work.columns:
        raise ValueError(f"mshare_var '{mshare_var}' not found in DataFrame.")

    inter_name = f"{input_var}_x_{mshare_var}"

    needed = ["county", "year", input_var, mshare_var, outcome_var]
    if prev_var is not None:
        needed.append(prev_var)

    print(f"[CY mshare] Needed cols: {needed}")
    df_sub = df_work[needed].dropna().copy()
    print(f"[CY mshare] Sample size after dropna: {df_sub.shape[0]}")

    # build interaction term
    df_sub[inter_name] = df_sub[input_var] * df_sub[mshare_var]

    df_sub["county"] = df_sub["county"].astype(str)

    y = df_sub[outcome_var].astype(float)

    exog_cols = [input_var, mshare_var, inter_name]
    if prev_var is not None:
        exog_cols.append(prev_var)
    print(f"[CY mshare] Exogenous vars: {exog_cols}")

    X = sm.add_constant(df_sub[exog_cols].astype(float))

    absorb = df_sub[["county", "year"]].astype("category")
    clusters = df_sub["county"].astype("category").cat.codes

    print("[CY mshare] Fitting AbsorbingLS...")
    mod = AbsorbingLS(y, X, absorb=absorb)
    res = mod.fit(cov_type="clustered", clusters=clusters)
    print("[CY mshare] Fit complete.")

    print(res.summary)

    header = (
        "\n\n============================================\n"
        f"[COUNTY×YEAR w/ mshare]\n"
        f"Outcome: {outcome_var}\n"
        f"Treatment: {input_var}\n"
        f"Market share var: {mshare_var}\n"
        f"Restrict high: {restrict_high}\n"
        f"Timestamp: {datetime.datetime.now()}\n"
        "============================================\n"
    )
    log_write(header)
    log_write(str(res.summary))

    def safe(series, name):
        return series[name] if name in series.index else np.nan

    row = {
        "level": "county_year_mshare",
        "outcome": outcome_var,
        "treatment": input_var,
        "mshare_var": mshare_var,
        "interaction": inter_name,
        "restrict_high": restrict_high,
        "prev_var": prev_var,

        "coef_treat": safe(res.params, input_var),
        "se_treat": safe(res.std_errors, input_var),
        "pval_treat": safe(res.pvalues, input_var),

        "coef_mshare": safe(res.params, mshare_var),
        "se_mshare": safe(res.std_errors, mshare_var),
        "pval_mshare": safe(res.pvalues, mshare_var),

        "coef_inter": safe(res.params, inter_name),
        "se_inter": safe(res.std_errors, inter_name),
        "pval_inter": safe(res.pvalues, inter_name),

        "coef_prev": safe(res.params, prev_var) if prev_var else np.nan,
        "se_prev": safe(res.std_errors, prev_var) if prev_var else np.nan,
        "pval_prev": safe(res.pvalues, prev_var) if prev_var else np.nan,

        "nobs": res.nobs,
        "rsq": res.rsquared,
    }

    return res, row

def run_cy_dids_mshare(sba_cy):
    """
    Run DID specs on the county–year SBA panel with
    treatment × market share interactions, both full sample
    and restricted to high-market-share closures.
    """
    print("\n========= COUNTY × YEAR (SBA) – market share DIDs =========")
    print(f"[run_cy_dids_mshare] Panel shape: {sba_cy.shape}")

    base_vars = [
        "dist_mean",
        "dist_median",
        "default_rate",
        "pif_rate",
        "n_loans",
        "loan_growth",
    ]
    treatments = [
        "any_closure",
        "any_exog_closure",
        "any_true_exog",
    ]
    treatments = [t for t in treatments if t in sba_cy.columns]
    print(f"[run_cy_dids_mshare] Treatments present: {treatments}")

    mshare_var = "combined_market_share_of_closure"
    if mshare_var not in sba_cy.columns:
        print(f"[run_cy_dids_mshare] {mshare_var} not found, skipping mshare DIDs.")
        return

    results_rows = []
    n_runs = 0

    for base in base_vars:
        outcome = f"{base}_next5"
        prev = f"{base}_prev5"

        if outcome not in sba_cy.columns or prev not in sba_cy.columns:
            print(f"[run_cy_dids_mshare] Skipping base={base} (no {outcome} or {prev})")
            continue

        for tr in treatments:
            # 1) full sample
            print(f"[run_cy_dids_mshare] Running (full) base={base}, treat={tr}")
            _, row_full = did_county_year_mshare(
                sba_cy,
                input_var=tr,
                outcome_var=outcome,
                mshare_var=mshare_var,
                restrict_high=False,
            )
            row_full["base_var"] = base
            row_full["sample_type"] = "full"
            results_rows.append(row_full)
            n_runs += 1

            # 2) restricted to high-market-share closures
            if "high_market_share_closure" in sba_cy.columns:
                print(f"[run_cy_dids_mshare] Running (high share) base={base}, treat={tr}")
                _, row_high = did_county_year_mshare(
                    sba_cy,
                    input_var=tr,
                    outcome_var=outcome,
                    mshare_var=mshare_var,
                    restrict_high=True,
                    high_indicator="high_market_share_closure",
                )
                row_high["base_var"] = base
                row_high["sample_type"] = "high_share_only"
                results_rows.append(row_high)
                n_runs += 1

    print(f"[run_cy_dids_mshare] Total DID regressions run (CY mshare): {n_runs}")

    if results_rows:
        results_df = pd.DataFrame(results_rows)
        out_path = OUTDIR_RESULTS / "sba_did_results_cy_mshare.csv"
        results_df.to_csv(out_path, index=False)
        print(f"\nSaved county–year mshare DID results to {out_path}")
    else:
        print("\nNo county–year mshare DIDs were run.")

def did_bcy_marketshare_only(df, outcome_var, mshare_var="combined_market_share_of_closure"):
    """
    Regress outcome_next5 on outcome_prev5 + market_share
    with bank FE + county FE + year FE.
    """
    print("\n----------------------------------------")
    print(f"[BCY mshare-only] Outcome: {outcome_var}, mshare={mshare_var}")
    print("----------------------------------------")

    df_work = df.copy()

    # rename if needed
    if "borrower_county" in df_work.columns and "county" not in df_work.columns:
        df_work = df_work.rename(columns={"borrower_county": "county"})

    # prev5 detection
    prev_var = None
    if outcome_var.endswith("next5"):
        cand = outcome_var.replace("next5", "prev5")
        if cand in df_work.columns:
            prev_var = cand

    if mshare_var not in df_work.columns:
        raise ValueError(f"mshare_var {mshare_var} missing")

    needed = ["loan_cert", "county", "year", mshare_var, outcome_var]
    if prev_var is not None:
        needed.append(prev_var)

    df_sub = df_work[needed].dropna().copy()
    print(f"[BCY mshare-only] Sample after dropna: {df_sub.shape}")

    df_sub["loan_cert"] = df_sub["loan_cert"].astype(str)
    df_sub["county"]     = df_sub["county"].astype(str)

    y = df_sub[outcome_var].astype(float)

    exog_cols = [mshare_var]
    if prev_var:
        exog_cols.append(prev_var)

    print(f"[BCY mshare-only] Exogenous vars: {exog_cols}")

    X = sm.add_constant(df_sub[exog_cols].astype(float))

    absorb = df_sub[["loan_cert", "county", "year"]].astype("category")
    clusters = (
        df_sub["loan_cert"].astype(str) + "|" + df_sub["county"].astype(str)
    ).astype("category").cat.codes

    mod = AbsorbingLS(y, X, absorb=absorb)
    res = mod.fit(cov_type="clustered", clusters=clusters)

    print(res.summary)

    return res

def did_cy_marketshare_only(df, outcome_var, mshare_var="combined_market_share_of_closure"):
    """
    Regress outcome_next5 on outcome_prev5 + market_share
    with county FE + year FE.
    """
    print("\n----------------------------------------")
    print(f"[CY mshare-only] Outcome: {outcome_var}, mshare={mshare_var}")
    print("----------------------------------------")

    df_work = df.copy()

    # rename if needed
    if "borrower_county" in df_work.columns and "county" not in df_work.columns:
        df_work = df_work.rename(columns={"borrower_county": "county"})

    # prev detection
    prev_var = None
    if outcome_var.endswith("next5"):
        cand = outcome_var.replace("next5", "prev5")
        if cand in df_work.columns:
            prev_var = cand

    if mshare_var not in df_work.columns:
        raise ValueError(f"mshare_var {mshare_var} missing")

    needed = ["county", "year", mshare_var, outcome_var]
    if prev_var is not None:
        needed.append(prev_var)

    df_sub = df_work[needed].dropna().copy()
    print(f"[CY mshare-only] Sample after dropna: {df_sub.shape}")

    df_sub["county"] = df_sub["county"].astype(str)

    y = df_sub[outcome_var].astype(float)

    exog_cols = [mshare_var]
    if prev_var:
        exog_cols.append(prev_var)

    print(f"[CY mshare-only] Exogenous vars: {exog_cols}")

    X = sm.add_constant(df_sub[exog_cols].astype(float))

    absorb = df_sub[["county", "year"]].astype("category")
    clusters = df_sub["county"].astype("category").cat.codes

    mod = AbsorbingLS(y, X, absorb=absorb)
    res = mod.fit(cov_type="clustered", clusters=clusters)

    print(res.summary)

    return res

def run_bcy_dids_mshare_only(sba_bcy):
    """
    Run 'market-share-only' regressions at bank×county×year level:
        outcome_next5 ~ prev5 + combined_market_share_of_closure
        + bank FE + county FE + year FE

    Uses:
      - did_bcy_marketshare_only()
      - base vars: dist_mean, dist_median, default_rate, pif_rate, n_loans, loan_growth
    """
    print("\n====== BANK × COUNTY × YEAR (SBA) – market share ONLY DIDs ======")
    print(f"[run_bcy_dids_mshare_only] Panel shape: {sba_bcy.shape}")

    base_vars = [
        "dist_mean",
        "dist_median",
        "default_rate",
        "pif_rate",
        "n_loans",
        "loan_growth",
    ]
    mshare_var = "combined_market_share_of_closure"

    if mshare_var not in sba_bcy.columns:
        print(f"[run_bcy_dids_mshare_only] {mshare_var} not found, skipping.")
        return

    results_rows = []
    n_runs = 0

    def safe(series, name):
        return series[name] if name in series.index else np.nan

    for base in base_vars:
        outcome = f"{base}_next5"
        prev = f"{base}_prev5"

        # need both next5 and prev5 for this spec
        if outcome not in sba_bcy.columns or prev not in sba_bcy.columns:
            print(f"[run_bcy_dids_mshare_only] Skipping base={base} (no {outcome} or {prev})")
            continue

        print(f"[run_bcy_dids_mshare_only] Running base={base}, outcome={outcome}")
        res = did_bcy_marketshare_only(
            sba_bcy,
            outcome_var=outcome,
            mshare_var=mshare_var,
        )
        n_runs += 1

        row = {
            "level": "bank_county_year_mshare_only",
            "base_var": base,
            "outcome": outcome,
            "mshare_var": mshare_var,
            "prev_var": prev,

            "coef_mshare": safe(res.params, mshare_var),
            "se_mshare": safe(res.std_errors, mshare_var),
            "pval_mshare": safe(res.pvalues, mshare_var),

            "coef_prev": safe(res.params, prev),
            "se_prev": safe(res.std_errors, prev),
            "pval_prev": safe(res.pvalues, prev),

            "nobs": res.nobs,
            "rsq": res.rsquared,
        }
        results_rows.append(row)

    print(f"[run_bcy_dids_mshare_only] Total mshare-only regressions run (BCY): {n_runs}")

    if results_rows:
        results_df = pd.DataFrame(results_rows)
        out_path = OUTDIR_RESULTS / "sba_did_results_bcy_mshare_only.csv"
        results_df.to_csv(out_path, index=False)
        print(f"\nSaved bank–county–year mshare-only DID results to {out_path}")
    else:
        print("\nNo bank–county–year mshare-only DIDs were run.")

def run_cy_dids_mshare_only(sba_cy):
    """
    Run 'market-share-only' regressions at county×year level:
        outcome_next5 ~ prev5 + combined_market_share_of_closure
        + county FE + year FE

    Uses:
      - did_cy_marketshare_only()
      - base vars: dist_mean, dist_median, default_rate, pif_rate, n_loans, loan_growth
    """
    print("\n========== COUNTY × YEAR (SBA) – market share ONLY DIDs ==========")
    print(f"[run_cy_dids_mshare_only] Panel shape: {sba_cy.shape}")

    base_vars = [
        "dist_mean",
        "dist_median",
        "default_rate",
        "pif_rate",
        "n_loans",
        "loan_growth",
    ]
    mshare_var = "combined_market_share_of_closure"

    if mshare_var not in sba_cy.columns:
        print(f"[run_cy_dids_mshare_only] {mshare_var} not found, skipping.")
        return

    results_rows = []
    n_runs = 0

    def safe(series, name):
        return series[name] if name in series.index else np.nan

    for base in base_vars:
        outcome = f"{base}_next5"
        prev = f"{base}_prev5"

        if outcome not in sba_cy.columns or prev not in sba_cy.columns:
            print(f"[run_cy_dids_mshare_only] Skipping base={base} (no {outcome} or {prev})")
            continue

        print(f"[run_cy_dids_mshare_only] Running base={base}, outcome={outcome}")
        res = did_cy_marketshare_only(
            sba_cy,
            outcome_var=outcome,
            mshare_var=mshare_var,
        )
        n_runs += 1

        row = {
            "level": "county_year_mshare_only",
            "base_var": base,
            "outcome": outcome,
            "mshare_var": mshare_var,
            "prev_var": prev,

            "coef_mshare": safe(res.params, mshare_var),
            "se_mshare": safe(res.std_errors, mshare_var),
            "pval_mshare": safe(res.pvalues, mshare_var),

            "coef_prev": safe(res.params, prev),
            "se_prev": safe(res.std_errors, prev),
            "pval_prev": safe(res.pvalues, prev),

            "nobs": res.nobs,
            "rsq": res.rsquared,
        }
        results_rows.append(row)

    print(f"[run_cy_dids_mshare_only] Total mshare-only regressions run (CY): {n_runs}")

    if results_rows:
        results_df = pd.DataFrame(results_rows)
        out_path = OUTDIR_RESULTS / "sba_did_results_cy_mshare_only.csv"
        results_df.to_csv(out_path, index=False)
        print(f"\nSaved county–year mshare-only DID results to {out_path}")
    else:
        print("\nNo county–year mshare-only DIDs were run.")


"""def main():
    print("[main] Loading panels and preparing for DID...")
    sba_bcy, sba_cy = load_sba_panels()
    print("[main] Clearing log file and starting DID runs...")
    sba_bcy.to_csv(DID_DATA / "sba_bcy_with_market_share_and_did_quantities.csv")
    sba_cy.to_csv(DID_DATA / "sba_cy_with_market_share_and_did_quantities.csv")
    open(LOG_PATH, "w").close()
    run_bcy_dids(sba_bcy)
    run_bcy_dids_mshare(sba_bcy)   # BCY w/ mshare
    run_cy_dids(sba_cy)
    run_cy_dids_mshare(sba_cy)     # CY w/ mshare
    print("[main] All DIDs completed.")"""

def main():
    print("[main] Loading pre-saved DID-ready panels...")

    bcy_path = DID_DATA / "sba_bcy_with_market_share_and_did_quantities.csv"
    cy_path  = DID_DATA / "sba_cy_with_market_share_and_did_quantities.csv"

    sba_bcy = pd.read_csv(bcy_path)
    sba_cy  = pd.read_csv(cy_path)

    print(f"[main] Loaded BCY panel: {sba_bcy.shape}")
    print(f"[main] Loaded CY panel:  {sba_cy.shape}")

    # Clear log
    print("[main] Clearing log file...")
    open(LOG_PATH, "w").close()

    print("[main] Running BCY standard DIDs...")
    run_bcy_dids(sba_bcy)

    print("[main] Running BCY market-share-only regressions...")
    run_bcy_dids_mshare_only(sba_bcy)   # <-- NEW LINE

    print("[main] Running CY market-share-only regressions...")
    run_cy_dids_mshare_only(sba_cy)     # <-- NEW LINE

    print("[main] All DIDs completed.")



if __name__ == "__main__":
    open(LOG_PATH, "w").close()
    main()
