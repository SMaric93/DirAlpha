import pandas as pd
import numpy as np
from . import config

def load_data():
    """
    Load raw data. Tries local parquet first, then WRDS.
    """
    # Try loading from local parquet
    try:
        compustat = pd.read_parquet(config.RAW_COMPUSTAT_PATH)
        crsp = pd.read_parquet(config.RAW_CRSP_PATH)
        ccm = pd.read_parquet(config.RAW_CCM_PATH)
        print("Loaded data from local parquet files.")
        return compustat, crsp, ccm
    except FileNotFoundError:
        print("Local files not found. Attempting to load from WRDS...")

    # Connect to WRDS
    db = config.get_wrds_connection()
    if db is None:
        print("WRDS connection failed. Returning empty DataFrames for structure.")
        # Return empty DFs with expected columns for code validation
        compustat = pd.DataFrame(columns=['gvkey', 'datadate', 'fyear', 'fic', 'at', 'oibdp', 'prcc_f', 'csho', 'ceq', 'dltt', 'dlc', 'xrd', 'capx', 'sich', 'naics'])
        crsp = pd.DataFrame(columns=['permno', 'date', 'shrcd', 'siccd', 'prc', 'ret'])
        ccm = pd.DataFrame(columns=['gvkey', 'lpermno', 'linkdt', 'linkenddt', 'linktype', 'linkprim'])
        return compustat, crsp, ccm

    # Query WRDS
    print("Fetching Compustat data...")
    # Select relevant columns to reduce size
    comp_query = f"""
        SELECT gvkey, datadate, fyear, fic, at, oibdp, prcc_f, csho, ceq, dltt, dlc, xrd, capx, sich, naicsh as naics
        FROM {config.WRDS_COMP_FUNDA}
        WHERE indfmt='INDL' AND datafmt='STD' AND popsrc='D' AND consol='C'
        AND fyear >= 2000
    """
    compustat = db.raw_sql(comp_query)

    print("Fetching CRSP data...")
    # Fetch monthly stock file for price and returns
    # Join with msenames to get shrcd and siccd
    crsp_query = f"""
        SELECT a.permno, a.date, b.shrcd, b.siccd, a.prc, a.ret
        FROM {config.WRDS_CRSP_MSF} a
        JOIN {config.WRDS_CRSP_MSENAMES} b
        ON a.permno = b.permno
        AND a.date >= b.namedt AND a.date <= b.nameendt
        WHERE a.date >= '2000-01-01'
    """
    crsp = db.raw_sql(crsp_query)

    print("Fetching CCM Link Table...")
    ccm_query = f"""
        SELECT gvkey, lpermno, linkdt, linkenddt, linktype, linkprim
        FROM {config.WRDS_CCM_LINK}
        WHERE linktype IN ('LU', 'LC')
    """
    ccm = db.raw_sql(ccm_query)

    # Save to local parquet for caching
    print("Saving data to local parquet files...")
    compustat.to_parquet(config.RAW_COMPUSTAT_PATH)
    crsp.to_parquet(config.RAW_CRSP_PATH)
    ccm.to_parquet(config.RAW_CCM_PATH)

    return compustat, crsp, ccm

def run_phase0():
    print("Starting Phase 0: Universe Definition...")
    compustat, crsp, ccm = load_data()

    # 1. Merge Compustat and CRSP using CCM
    # Filter CCM link table
    ccm = ccm[ccm['linktype'].isin(['LU', 'LC'])]
    ccm = ccm[ccm['linkprim'].isin(['P', 'C'])]
    
    # Merge CCM to Compustat
    # Ensure datatypes match
    compustat['gvkey'] = compustat['gvkey'].astype(str)
    ccm['gvkey'] = ccm['gvkey'].astype(str)
    
    # Simple merge on gvkey, then filter by date range
    merged = pd.merge(compustat, ccm, on='gvkey', how='inner')
    
    # Date filtering for link validity
    merged['datadate'] = pd.to_datetime(merged['datadate'])
    merged['linkdt'] = pd.to_datetime(merged['linkdt'])
    merged['linkenddt'] = pd.to_datetime(merged['linkenddt']).fillna(pd.Timestamp('today'))
    
    merged = merged[(merged['datadate'] >= merged['linkdt']) & (merged['datadate'] <= merged['linkenddt'])]
    
    # Rename lpermno to permno for CRSP merge
    merged = merged.rename(columns={'lpermno': 'permno'})
    
    # 2. Apply Standard Filters
    
    # US Incorporated (fic = 'USA')
    merged = merged[merged['fic'] == 'USA']
    
    # Exclude Financials (6000-6999) and Utilities (4900-4949)
    # Use SICH from Compustat, if missing use SICCD from CRSP (need to merge CRSP first to get SICCD properly, 
    # but usually SICH is sufficient for Compustat universe. Let's use SICH first)
    # Note: SICH is often string or float. Handle carefully.
    merged['sich'] = pd.to_numeric(merged['sich'], errors='coerce')
    
    condition_fin = (merged['sich'] >= 6000) & (merged['sich'] <= 6999)
    condition_util = (merged['sich'] >= 4900) & (merged['sich'] <= 4949)
    
    merged = merged[~(condition_fin | condition_util)]
    
    # 3. Merge with CRSP to check Share Code (SHRCD)
    # We need CRSP info at the fiscal year end.
    # CRSP is usually monthly or daily. We need to find the closest CRSP record to datadate.
    # For efficiency in this simplified version, we assume we have a CRSP annual or we merge on year/month.
    # Let's assume 'crsp' passed here is a monthly file or similar.
    # A common approach is to extract year/month and merge.
    
    # Simplification: Just checking SHRCD if available in a firm-year CRSP summary
    # If CRSP is daily/monthly, we need to aggregate or pick the month of fiscal year end.
    
    # Let's assume we can get SHRCD from the CRSP data. 
    # For this implementation, we will assume CRSP has 'permno' and 'year' or we do a precise merge.
    # To be robust:
    crsp['date'] = pd.to_datetime(crsp['date'])
    crsp['year'] = crsp['date'].dt.year
    crsp['month'] = crsp['date'].dt.month
    
    merged['year'] = merged['datadate'].dt.year
    merged['month'] = merged['datadate'].dt.month
    
    # Merge CRSP on permno and year/month (approximate for fiscal year end)
    # Note: This can be complex due to fiscal year ends. 
    # We will do a left join on permno and year to check the SHRCD condition for that year.
    # Taking the mode or last value of SHRCD for the year is a reasonable proxy if exact date match fails.
    
    crsp_annual = crsp.groupby(['permno', 'year'])['shrcd'].first().reset_index()
    
    final_df = pd.merge(merged, crsp_annual, on=['permno', 'year'], how='inner')
    
    # Require common stock (SHRCD 10 or 11)
    final_df = final_df[final_df['shrcd'].isin([10, 11])]
    
    # 4. Output
    columns_to_keep = ['gvkey', 'permno', 'fyear', 'datadate', 'sich', 'naics', 'at', 'oibdp', 'prcc_f', 'csho', 'ceq', 'dltt', 'dlc', 'xrd', 'capx']
    # Ensure columns exist
    available_cols = [c for c in columns_to_keep if c in final_df.columns]
    firm_year_base = final_df[available_cols].copy()
    
    print(f"Phase 0 Complete. Generated {len(firm_year_base)} firm-year observations.")
    
    # Save
    firm_year_base.to_parquet(config.FIRM_YEAR_BASE_PATH)
    return firm_year_base

if __name__ == "__main__":
    run_phase0()
