import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

INDIR_SBA = Path("source/transformations/data")
FIGDIR = Path("source/transformations/figures")

def load_panels():
    sba_bcy = pd.read_csv(INDIR_SBA / "sba_bcy.csv")
    sba_cy  = pd.read_csv(INDIR_SBA / "sba_cy.csv")
    sod_bcy = pd.read_csv(INDIR_SBA / "sod_bcy.csv")
    sod_cy  = pd.read_csv(INDIR_SBA / "sod_cy.csv")
    return sba_bcy, sba_cy, sod_bcy, sod_cy

def plot_sba_distance_over_time(sba_bcy: pd.DataFrame, sba_cy: pd.DataFrame) -> None:
    """
    Quick sanity plot of lending distance over time:
      - bank–county–year: mean dist_mean by year
      - county–year:      mean dist_mean by year
    Saves to FIGDIR / 'sba_distance_over_time.png'.
    """
    # guard if distance column missing
    if "dist_mean" not in sba_bcy.columns or "dist_mean" not in sba_cy.columns:
        print("dist_mean not found in one of the panels; skipping distance plot.")
        return

    bcy_dist = (
        sba_bcy
        .groupby("year", as_index=False)["dist_mean"]
        .mean()
        .rename(columns={"dist_mean": "dist_mean_bcy"})
    )

    cy_dist = (
        sba_cy
        .groupby("year", as_index=False)["dist_mean"]
        .mean()
        .rename(columns={"dist_mean": "dist_mean_cy"})
    )

    merged = pd.merge(bcy_dist, cy_dist, on="year", how="inner")

    plt.figure(figsize=(8, 5))
    plt.plot(merged["year"], merged["dist_mean_bcy"], label="Bank–county–year mean distance")
    plt.plot(merged["year"], merged["dist_mean_cy"],  label="County–year mean distance", linestyle="--")
    plt.xlabel("Year")
    plt.ylabel("Mean lending distance")
    plt.title("SBA Lending Distance Over Time")
    plt.legend()
    plt.tight_layout()

    outpath = FIGDIR / "sba_distance_over_time.png"
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"Saved lending distance plot to {outpath}")

def plot_loan_growth_over_time(sba_bcy: pd.DataFrame, sba_cy: pd.DataFrame) -> None:
    """
    Plot average loan_growth_next5 over time:
      - mean loan_growth_next5 at bank–county–year level
      - mean loan_growth_next5 at county–year level
    """
    var = "loan_growth_next5"

    if var not in sba_bcy.columns and var not in sba_cy.columns:
        print(f"{var} not found in panels; skipping loan-growth plot.")
        return

    plt.figure(figsize=(8, 5))

    if var in sba_bcy.columns:
        bcy_growth = (
            sba_bcy
            .groupby("year", as_index=False)[var]
            .mean(numeric_only=True)
        )
        plt.plot(bcy_growth["year"], bcy_growth[var],
                 label="BCY mean loan_growth_next5")

    if var in sba_cy.columns:
        cy_growth = (
            sba_cy
            .groupby("year", as_index=False)[var]
            .mean(numeric_only=True)
        )
        plt.plot(cy_growth["year"], cy_growth[var],
                 label="CY mean loan_growth_next5", linestyle="--")

    plt.xlabel("Year")
    plt.ylabel("Average loan_growth_next5")
    plt.title("Loan Growth (5-year forward window) Over Time")
    plt.legend()
    plt.tight_layout()

    outpath = FIGDIR / "sba_loan_growth_next5_over_time.png"
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"Saved loan growth (next5) plot to {outpath}")

def plot_branches_over_time_sba(sba_bcy: pd.DataFrame, sba_cy: pd.DataFrame) -> None:
    """
    Plot number of branches over time using SBA panels:

      - Unweighted mean branches at bank–county–year level
      - Unweighted mean branches at county–year level (if available)
      - Loan-weighted mean branches (weights = n_loans), BCY + CY

    Uses n_branches_next5 if available, otherwise n_branches_past5.
    """
    # pick branch variable
    branch_var_candidates = ["n_branches_next5", "n_branches_past5"]
    branch_var = None
    for cand in branch_var_candidates:
        if cand in sba_bcy.columns:
            branch_var = cand
            break

    if branch_var is None:
        print("No branch count variable (n_branches_next5/past5) in sba_bcy; skipping branches plot.")
        return

    print(f"Using branch variable '{branch_var}' for branch plots.")

    plt.figure(figsize=(8, 5))

    # --- unweighted BCY mean ---
    bcy = sba_bcy.copy()
    bcy_mean = (
        bcy.groupby("year", as_index=False)[branch_var]
           .mean(numeric_only=True)
           .rename(columns={branch_var: "branches_mean_bcy"})
    )
    plt.plot(bcy_mean["year"], bcy_mean["branches_mean_bcy"],
             label="BCY mean branches")

    # --- unweighted CY mean, if branch_var present in sba_cy ---
    if branch_var in sba_cy.columns:
        cy = sba_cy.copy()
        cy_mean = (
            cy.groupby("year", as_index=False)[branch_var]
              .mean(numeric_only=True)
              .rename(columns={branch_var: "branches_mean_cy"})
        )
        plt.plot(cy_mean["year"], cy_mean["branches_mean_cy"],
                 label="CY mean branches", linestyle="--")

    plt.xlabel("Year")
    plt.ylabel("Mean branches (unweighted)")
    plt.title("Average Branch Counts Over Time (SBA Panel)")
    plt.legend()
    plt.tight_layout()

    outpath = FIGDIR / "sba_branches_over_time_unweighted.png"
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"Saved unweighted branches-over-time plot to {outpath}")

    # --- loan-weighted mean branches ---

    if "n_loans" not in sba_bcy.columns and "n_loans" not in sba_cy.columns:
        print("n_loans not in SBA panels; skipping weighted branch plot.")
        return

    plt.figure(figsize=(8, 5))

    # BCY weighted
    if "n_loans" in sba_bcy.columns:
        b = sba_bcy.copy()
        b = b.dropna(subset=[branch_var, "n_loans"])
        # weighted mean per year: sum(branches * loans) / sum(loans)
        b_weighted = (
            b.assign(weighted_branch=b[branch_var] * b["n_loans"])
             .groupby("year", as_index=False)
             .agg(total_weighted=("weighted_branch", "sum"),
                  total_loans=("n_loans", "sum"))
        )
        b_weighted["branches_w_mean_bcy"] = (
            b_weighted["total_weighted"] / b_weighted["total_loans"]
        )
        plt.plot(b_weighted["year"], b_weighted["branches_w_mean_bcy"],
                 label="BCY loan-weighted branches")

    # CY weighted
    if "n_loans" in sba_cy.columns and branch_var in sba_cy.columns:
        c = sba_cy.copy()
        c = c.dropna(subset=[branch_var, "n_loans"])
        c_weighted = (
            c.assign(weighted_branch=c[branch_var] * c["n_loans"])
             .groupby("year", as_index=False)
             .agg(total_weighted=("weighted_branch", "sum"),
                  total_loans=("n_loans", "sum"))
        )
        c_weighted["branches_w_mean_cy"] = (
            c_weighted["total_weighted"] / c_weighted["total_loans"]
        )
        plt.plot(c_weighted["year"], c_weighted["branches_w_mean_cy"],
                 label="CY loan-weighted branches", linestyle="--")

    plt.xlabel("Year")
    plt.ylabel("Loan-weighted mean branches")
    plt.title("Loan-Weighted Branch Counts Over Time (SBA Panel)")
    plt.legend()
    plt.tight_layout()

    outpath = FIGDIR / "sba_branches_over_time_weighted.png"
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"Saved loan-weighted branches-over-time plot to {outpath}")


def ensure_loan_growth_bcy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure loan_growth exists on the bank–county–year panel.
    If it's already there (from clean_closure_data), we leave it alone.
    """
    df = df.copy()
    if "loan_growth" not in df.columns and {"loan_cert", "borrower_county", "year", "n_loans"}.issubset(df.columns):
        df = df.sort_values(["loan_cert", "borrower_county", "year"])
        df["loan_growth"] = (
            df.groupby(["loan_cert", "borrower_county"])["n_loans"]
              .pct_change()
        )
    return df

def analyze_high_distance_branches(sba_bcy: pd.DataFrame, sba_cy: pd.DataFrame, sod_cy: pd.DataFrame) -> None:
    """
    Get a sense of how many branches are in the borrower county for
    observations with really high lending distance.

    Strategy:
      - harmonize county + year between SBA and SOD
      - merge SBA (dist_mean) with SOD (n_branches_total) at county-year
      - define "high distance" as top 10% of dist_mean (at county-year level)
      - compare branch counts for high vs other
      - save a bar plot and print summary stats
    """
    if "dist_mean" not in sba_cy.columns:
        print("dist_mean not in sba_cy; skipping high-distance branch analysis.")
        return
    if "YEAR" not in sod_cy.columns or "STCNTYBR" not in sod_cy.columns or "n_branches_total" not in sod_cy.columns:
        print("SOD county-year missing YEAR/STCNTYBR/n_branches_total; skipping high-distance branch analysis.")
        return

    # Harmonize county + year
    sba = sba_cy.copy()
    if "borrower_county" in sba.columns and "county" not in sba.columns:
        sba = sba.rename(columns={"borrower_county": "county"})
    sba = sba[["county", "year", "dist_mean"]].dropna()

    sod = sod_cy.copy()
    sod = sod.rename(columns={"STCNTYBR": "county", "YEAR": "year"})
    sod = sod[["county", "year", "n_branches_total"]].dropna()

    # Make sure types line up
    sba["county"] = sba["county"].astype(int)
    sod["county"] = sod["county"].astype(int)

    merged = pd.merge(sba, sod, on=["county", "year"], how="inner")

    if merged.empty:
        print("No matches between SBA and SOD at county-year; skipping high-distance analysis.")
        return

    # Define high-distance as top 10% of dist_mean
    threshold = merged["dist_mean"].quantile(0.9)
    merged["high_dist"] = (merged["dist_mean"] >= threshold).astype(int)

    # Summary stats
    summary = (
        merged.groupby("high_dist")["n_branches_total"]
              .agg(["mean", "median", "count"])
    )
    print("\nBranches in borrower county by high-distance indicator (county–year):")
    print(summary)

    # Simple bar plot: mean branches for high vs others
    means = summary["mean"]
    labels = ["Not high distance", "High distance"]
    x = [0, 1]

    plt.figure(figsize=(6, 4))
    plt.bar(x, [means.get(0, np.nan), means.get(1, np.nan)])
    plt.xticks(x, labels, rotation=15)
    plt.ylabel("Mean number of branches in county")
    plt.title("Branches in Borrower County: High vs Other Lending Distance")
    plt.tight_layout()

    outpath = FIGDIR / "branches_vs_high_distance.png"
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"Saved high-distance vs branches plot to {outpath}")


def plot_closure_share_over_time(sba_cy: pd.DataFrame) -> None:
    """
    Plot the share of county–years with any_closure over time.
    """
    if "any_closure" not in sba_cy.columns:
        print("any_closure not found in sba_cy; skipping closure-share plot.")
        return

    # ensure binary
    sba_cy = sba_cy.copy()
    sba_cy["any_closure"] = (sba_cy["any_closure"] > 0).astype(int)

    share_by_year = (
        sba_cy
        .groupby("year", as_index=False)["any_closure"]
        .mean()
        .rename(columns={"any_closure": "closure_share"})
    )

    plt.figure(figsize=(8, 5))
    plt.plot(share_by_year["year"], share_by_year["closure_share"])
    plt.xlabel("Year")
    plt.ylabel("Share of county–years with any closure")
    plt.title("Share of County–Years with Any Bank Closure (SBA Counties)")
    plt.tight_layout()

    outpath = FIGDIR / "sba_closure_share_over_time.png"
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"Saved closure-share plot to {outpath}")

def plot_default_highdist_low_vs_high_branches(sba_cy: pd.DataFrame, sod_cy: pd.DataFrame):
    """
    Among high-distance county-years, split by branch density (low vs high)
    and compare default_rate.
    Saves a bar plot.
    """

    # Check required columns
    if "dist_mean" not in sba_cy.columns or "default_rate" not in sba_cy.columns:
        print("Missing dist_mean or default_rate in sba_cy; skipping.")
        return
    if "STCNTYBR" not in sod_cy.columns or "YEAR" not in sod_cy.columns or "n_branches_total" not in sod_cy.columns:
        print("Missing SOD county-year branch counts; skipping.")
        return

    # Harmonize columns
    sba = sba_cy.copy()
    if "borrower_county" in sba.columns and "county" not in sba.columns:
        sba = sba.rename(columns={"borrower_county": "county"})

    sod = sod_cy.rename(columns={"STCNTYBR": "county", "YEAR": "year"})

    # Make sure keys match
    sba["county"] = sba["county"].astype(int)
    sod["county"] = sod["county"].astype(int)

    merged = pd.merge(
        sba[["county", "year", "dist_mean", "default_rate"]],
        sod[["county", "year", "n_branches_total"]],
        on=["county", "year"],
        how="inner"
    )

    if merged.empty:
        print("No overlap SBA–SOD; skipping.")
        return

    # Define high-distance
    dist_thresh = merged["dist_mean"].quantile(0.9)
    merged["high_dist"] = (merged["dist_mean"] >= dist_thresh).astype(int)

    # Only keep high-distance rows
    hd = merged[merged["high_dist"] == 1].copy()

    # Split into low vs high branch counties
    branch_thresh = hd["n_branches_total"].median()
    hd["low_branch_area"] = (hd["n_branches_total"] < branch_thresh).astype(int)

    summary = (
        hd.groupby("low_branch_area")["default_rate"]
          .agg(["mean", "median", "count"])
    )
    print("\nDefault rate among HIGH-DISTANCE counties:")
    print(summary)

    # Plot
    means = summary["mean"]
    labels = ["High branches", "Low branches"]
    x = [0, 1]

    plt.figure(figsize=(6, 4))
    plt.bar(x, [means.get(0, np.nan), means.get(1, np.nan)])
    plt.xticks(x, labels)
    plt.ylabel("Default rate")
    plt.title("Default Rate in High-Distance Counties: Low vs High Branch Density")
    plt.tight_layout()

    outpath = FIGDIR / "default_highdistance_low_vs_high_branches.png"
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"Saved: {outpath}")

def plot_default_rate_by_distance_bin(sba_cy: pd.DataFrame, n_bins: int = 10):
    """
    Plot mean default rate by distance decile (or n_bins).
    """

    if "dist_mean" not in sba_cy.columns or "default_rate" not in sba_cy.columns:
        print("Missing dist_mean/default_rate in sba_cy; skipping default-by-bin plot.")
        return

    df = sba_cy[["dist_mean", "default_rate"]].dropna().copy()
    df["dist_bin"] = pd.qcut(df["dist_mean"], n_bins, labels=False, duplicates="drop")

    grouped = (
        df.groupby("dist_bin")["default_rate"]
          .mean()
          .reset_index()
          .rename(columns={"default_rate": "default_rate_mean"})
    )

    plt.figure(figsize=(8, 5))
    plt.plot(grouped["dist_bin"], grouped["default_rate_mean"], marker="o")
    plt.xlabel(f"Distance decile (0 = lowest)")
    plt.ylabel("Mean default rate")
    plt.title("Default Rate vs Distance (Decile Bins)")
    plt.tight_layout()

    outpath = FIGDIR / "default_rate_by_distance_bin.png"
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"Saved: {outpath}")

def plot_default_by_distance_and_branch_terciles(
    sba_cy: pd.DataFrame,
    sod_cy: pd.DataFrame,
    n_dist_bins: int = 10,
) -> None:
    """
    Plot mean default rate by distance bins, with separate lines for
    counties in low / mid / high branch terciles.

    Idea: look for adverse selection — borrower counties with *many*
    branches but still lending at high distance might have higher default.
    """

    # Check required columns
    if "dist_mean" not in sba_cy.columns or "default_rate" not in sba_cy.columns:
        print("Missing dist_mean or default_rate in sba_cy; skipping tercile plot.")
        return
    required_sod = {"STCNTYBR", "YEAR", "n_branches_total"}
    if not required_sod.issubset(sod_cy.columns):
        print("Missing SOD columns for branch counts; skipping tercile plot.")
        return

    # Harmonize keys
    sba = sba_cy.copy()
    if "borrower_county" in sba.columns and "county" not in sba.columns:
        sba = sba.rename(columns={"borrower_county": "county"})

    sba = sba[["county", "year", "dist_mean", "default_rate"]].dropna()
    sod = sod_cy.rename(columns={"STCNTYBR": "county", "YEAR": "year"})
    sod = sod[["county", "year", "n_branches_total"]].dropna()

    # Ensure same types for merge keys
    sba["county"] = sba["county"].astype(int)
    sod["county"] = sod["county"].astype(int)

    merged = pd.merge(sba, sod, on=["county", "year"], how="inner")

    if merged.empty:
        print("No overlap SBA–SOD at county-year; skipping tercile plot.")
        return

    # Branch terciles (low / mid / high)
    # Use qcut; duplicates="drop" handles ties if needed.
    merged["branch_tercile"] = pd.qcut(
        merged["n_branches_total"],
        q=3,
        labels=["Low branches", "Mid branches", "High branches"],
        duplicates="drop",
    )

    # Distance bins (e.g., deciles)
    merged = merged.dropna(subset=["dist_mean", "default_rate"])
    merged["dist_bin"] = pd.qcut(
        merged["dist_mean"],
        q=n_dist_bins,
        labels=False,
        duplicates="drop",
    )

    # Compute mean default per (distance bin × branch tercile)
    grouped = (
        merged
        .groupby(["branch_tercile", "dist_bin"], as_index=False)["default_rate"]
        .mean()
        .rename(columns={"default_rate": "default_rate_mean"})
    )

    if grouped.empty:
        print("No data after binning; skipping tercile plot.")
        return

    # Plot
    plt.figure(figsize=(8, 5))

    for tercile in grouped["branch_tercile"].dropna().unique():
        g = grouped[grouped["branch_tercile"] == tercile].sort_values("dist_bin")
        plt.plot(
            g["dist_bin"],
            g["default_rate_mean"],
            marker="o",
            label=str(tercile),
        )

    plt.xlabel("Distance bin (0 = lowest distance)")
    plt.ylabel("Mean default rate")
    plt.title("Default Rate vs Distance, by Branch Terciles (County–Year)")
    plt.legend(title="Branch tercile")
    plt.tight_layout()

    outpath = FIGDIR / "default_rate_by_distance_and_branch_terciles.png"
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"Saved default-by-distance-and-branch-terciles plot to {outpath}")


def main():
    sba_bcy, sba_cy, sod_bcy, sod_cy = load_panels()
    plot_sba_distance_over_time(sba_bcy, sba_cy)
    plot_loan_growth_over_time(sba_bcy, sba_cy)
    plot_branches_over_time_sba(sba_bcy, sba_cy)
    analyze_high_distance_branches(sba_bcy, sba_cy, sod_cy)
    plot_default_highdist_low_vs_high_branches(sba_cy, sod_cy)
    plot_default_rate_by_distance_bin(sba_cy)
    plot_default_by_distance_and_branch_terciles(sba_cy, sod_cy)
    print("describe_closure_data: all figures done.")

if __name__ == "__main__":
    main()

