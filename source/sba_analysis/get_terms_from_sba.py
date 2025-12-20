import pandas as pd
from pathlib import Path

INDIR_SBA_RAW = Path("source/sba_analysis/data/raw/sba_7a_data")
OUTPATH = Path("source/sba_analysis/data/derived")
BASE_DIR      = Path("source/sba_analysis/data")

OUTPATH.mkdir(parents=True, exist_ok=True)

file_list = [
    "foia-7a-fy1991-fy1999-asof-250930.csv",
    "foia-7a-fy2000-fy2009-asof-250930.csv",
    "foia-7a-fy2010-fy2019-asof-250930.csv",
    "foia-7a-fy2020-present-asof-250930.csv",
]

dfs = {}

for fname in file_list:
    key = fname.split("-")[2]  # e.g. fy1991-fy1999
    dfs[key] = pd.read_csv(INDIR_SBA_RAW / fname)

# example access
df_91 = dfs["fy1991"]
df_00 = dfs["fy2000"]
df_10 = dfs["fy2010"]
df_20 = dfs["fy2020"]

print(df_91.columns)
print(df_00.columns)
print(df_10.columns)
print(df_20.columns)

"""
Index(['AsOfDate', 'Program', 'BorrName', 'BorrStreet', 'BorrCity',
       'BorrState', 'BorrZip', 'LocationID', 'BankName', 'BankFDICNumber',
       'BankNCUANumber', 'BankStreet', 'BankCity', 'BankState', 'BankZip',
       'GrossApproval', 'SBAGuaranteedApproval', 'ApprovalDate', 'ApprovalFY',
       'FirstDisbursementDate', 'ProcessingMethod', 'Subprogram',
       'InitialInterestRate', 'FixedorVariableInterestRate', 'TerminMonths',
       'NAICSCode', 'NAICSDescription', 'FranchiseCode', 'FranchiseName',
       'ProjectCounty', 'ProjectState', 'SBADistrictOffice',
       'CongressionalDistrict', 'BusinessType', 'BusinessAge', 'LoanStatus',
       'PaidinFullDate', 'ChargeoffDate', 'GrossChargeoffAmount',
       'RevolverStatus', 'JobsSupported', 'CollateralInd',
       'SoldSecondMarketInd'],
      dtype='object')
"""

df_all = pd.concat(
    [df_91, df_00, df_10, df_20],
    ignore_index=True
)

date_cols = [
    "AsOfDate",
    "ApprovalDate",
    "FirstDisbursementDate",
    "PaidinFullDate",
    "ChargeoffDate",
]

for c in date_cols:
    df_all[c] = pd.to_datetime(df_all[c], errors="coerce")

df_all["BankFDICNumber"] = df_all["BankFDICNumber"].astype("string")
df_all["BorrZip"] = df_all["BorrZip"].astype("string").str.zfill(5)
df_all["BankZip"] = df_all["BankZip"].astype("string").str.zfill(5)

df_all.to_csv(OUTPATH / "sba_7a_1991_2025_raw.csv", index=False)



