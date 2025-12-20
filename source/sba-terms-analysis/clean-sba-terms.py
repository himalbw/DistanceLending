import pandas as pd
import numpy as np
import re
from pathlib import Path

# -------------------------
# Paths / config
# -------------------------
TERMS_PATH = Path("source/sba-terms-analysis/sba_7a_1991_2025_raw_with_terms.csv")
GEO_PATH   = Path("source/sba-terms-analysis/sba_7a_geocoded_final.csv")
DIST_PATH  = Path("source/sba_analysis/data/derived/sba_7a_loan_to_branch.csv")
OUT_PATH   = Path("source/sba_analysis/data/derived/sba_7a_loan_to_branch_with_terms.csv")

# rounding for lat/lon match (adjust to 4 or 3 if match rate is low)
COORD_ROUND = 5


# -------------------------
# Helpers
# -------------------------
def norm_name(s: pd.Series) -> pd.Series:
    """Normalize lender/bank names for join stability."""
    s = s.astype("string").fillna("")
    s = s.str.upper().str.strip()
    s = s.str.replace("&", " AND ", regex=False)
    s = s.str.replace(r"[^A-Z0-9\s]", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    s = s.str.replace(r"\b(LLC|INC|CORP|CO|COMPANY|LTD|LP|LLP|PLC)\b", "", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s


def dedup_columns(df: pd.DataFrame) -> pd.DataFrame:
    """If a dataframe has duplicate column names, keep the first occurrence."""
    return df.loc[:, ~df.columns.duplicated()]


def read_csv_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = dedup_columns(df)
    # fail fast if duplicates remain
    assert df.columns.is_unique, f"Duplicate columns remain after dedup in {path}"
    return df


def ensure_datetime(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")


def ensure_numeric_round(df: pd.DataFrame, col: str, out_col: str, round0: bool = True) -> None:
    if col in df.columns:
        x = pd.to_numeric(df[col], errors="coerce")
        df[out_col] = x.round(0) if round0 else x


# -------------------------
# Core merge logic
# -------------------------
def build_merge_keys(
    geo: pd.DataFrame,
    dist: pd.DataFrame,
    coord_round: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Adds canonical key columns to geo + dist and returns merge_keys list.
    """
    geo = geo.copy()
    dist = dist.copy()

    # canonical approval date (use this everywhere)
    ensure_datetime(geo, "ApprovalDate")
    ensure_datetime(dist, "ApprovalDate")
    geo["approval_date"]  = geo["ApprovalDate"]
    dist["approval_date"] = dist["ApprovalDate"]

    # other dates (not used in keys now, but may be helpful downstream)
    ensure_datetime(geo, "FirstDisbursementDate")
    ensure_datetime(dist, "FirstDisbursementDate")

    # amounts
    ensure_numeric_round(geo,  "GrossApproval", "gross_approval", round0=True)
    ensure_numeric_round(dist, "GrossApproval", "gross_approval", round0=True)

    ensure_numeric_round(geo,  "SBAGuaranteedApproval", "sba_guaranteed", round0=True)
    ensure_numeric_round(dist, "SBAGuaranteedApproval", "sba_guaranteed", round0=True)

    # coords: geo uses x=lon, y=lat; dist uses borrower_lon/borrower_lat
    geo["lat_r"]  = pd.to_numeric(geo["y"], errors="coerce").round(coord_round)
    geo["lon_r"]  = pd.to_numeric(geo["x"], errors="coerce").round(coord_round)
    dist["lat_r"] = pd.to_numeric(dist["borrower_lat"], errors="coerce").round(coord_round)
    dist["lon_r"] = pd.to_numeric(dist["borrower_lon"], errors="coerce").round(coord_round)

    # bank names: bank_name (dist) vs BankName (geo)
    geo["bank_norm"]  = norm_name(geo["BankName"])
    dist["bank_norm"] = norm_name(dist["bank_name"])

    # optional disambiguators (keep as keys only if present on BOTH)
    if "LoanStatus" in geo.columns and "LoanStatus" in dist.columns:
        geo["status_norm"]  = geo["LoanStatus"].astype("string").str.upper().str.strip()
        dist["status_norm"] = dist["LoanStatus"].astype("string").str.upper().str.strip()
    else:
        geo["status_norm"] = pd.NA
        dist["status_norm"] = pd.NA

    if "CollateralInd" in geo.columns and "CollateralInd" in dist.columns:
        geo["coll_norm"]  = geo["CollateralInd"].astype("string").str.upper().str.strip()
        dist["coll_norm"] = dist["CollateralInd"].astype("string").str.upper().str.strip()
    else:
        geo["coll_norm"] = pd.NA
        dist["coll_norm"] = pd.NA

    merge_keys = [
        "approval_date",
        "gross_approval",
        "sba_guaranteed",
        "lat_r",
        "lon_r",
        "bank_norm",
        "status_norm",
        "coll_norm",
    ]

    return geo, dist, merge_keys


def dedup_geo_on_keys(
    geo: pd.DataFrame,
    merge_keys: list[str],
    score_col: str = "Score",
) -> pd.DataFrame:
    """
    Deduplicate geo to ensure m:1 merge.
    Keeps the highest-Score record when multiple rows share merge_keys.
    """
    # Only add non-key columns to avoid duplicate labels in geo_dedup selection
    geo_nonkey_cols = [c for c in geo.columns if c not in merge_keys]

    geo_dedup = (
        geo[merge_keys + geo_nonkey_cols]
        .dropna(subset=["approval_date", "gross_approval", "lat_r", "lon_r"])
    )

    if score_col in geo_dedup.columns:
        geo_dedup = geo_dedup.sort_values(by=[score_col], ascending=False)

    geo_dedup = geo_dedup.drop_duplicates(subset=merge_keys, keep="first")
    return geo_dedup


def merge_terms_onto_dist(
    dist: pd.DataFrame,
    geo_dedup: pd.DataFrame,
    merge_keys: list[str],
) -> pd.DataFrame:
    merged = dist.merge(
        geo_dedup,
        on=merge_keys,
        how="left",
        validate="m:1",
        suffixes=("", "_geo"),
    )
    return merged


def report_match_rate(merged: pd.DataFrame, proxy_col: str = "BorrName") -> None:
    if proxy_col not in merged.columns:
        print(f"[merge] proxy column '{proxy_col}' not present; skipping match-rate report.")
        return
    n = len(merged)
    m = int(merged[proxy_col].notna().sum())
    print(f"[merge] matched rows: {m:,}/{n:,} ({m/n:.1%})")


# -------------------------
# Main
# -------------------------
def main() -> None:
    print("Loading input files...")
    terms = read_csv_clean(TERMS_PATH)   # not strictly needed for the final merge; kept for completeness
    geo   = read_csv_clean(GEO_PATH)
    dist  = read_csv_clean(DIST_PATH)

    print("Building merge keys...")
    geo_k, dist_k, merge_keys = build_merge_keys(geo, dist, coord_round=COORD_ROUND)

    print("Deduplicating GEO on merge keys...")
    geo_dedup = dedup_geo_on_keys(geo_k, merge_keys=merge_keys, score_col="Score")

    print("Merging terms+geo onto distance file...")
    merged = merge_terms_onto_dist(dist_k, geo_dedup, merge_keys=merge_keys)

    report_match_rate(merged, proxy_col="BorrName")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
