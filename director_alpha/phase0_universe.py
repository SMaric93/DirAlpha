import pandas as pd
import numpy as np
from . import config, utils

def run_phase0():
    utils.logger.info("Starting Phase 0: Universe Definition...")
    
    # 1. Load Data
    compustat = utils.load_or_fetch(
        config.RAW_COMPUSTAT_PATH, 
        utils.fetch_compustat_funda, 
        start_year=2000
    )
    
    crsp = utils.load_or_fetch(
        config.RAW_CRSP_PATH,
        utils.fetch_crsp_msf,
        start_date='2000-01-01'
    )
    
    ccm = utils.load_or_fetch(
        config.RAW_CCM_PATH,
        utils.fetch_ccm_link
    )

    if compustat.empty or crsp.empty or ccm.empty:
        utils.logger.error("One or more input datasets are empty. Aborting Phase 0.")
        return

    # 2. Merge Compustat and CRSP using CCM
    utils.logger.info("Merging Compustat, CCM, and CRSP...")
    
    # Filter CCM link table (redundant if fetcher does it, but safe to keep)
    ccm = ccm[ccm['linktype'].isin(['LU', 'LC'])]
    # pyrefly: ignore [missing-attribute]
    ccm = ccm[ccm['linkprim'].isin(['P', 'C'])]
    
    # Normalize keys
    # pyrefly: ignore [bad-argument-type]
    compustat['gvkey'] = utils.normalize_gvkey(compustat['gvkey'])
    # pyrefly: ignore [bad-argument-type]
    ccm['gvkey'] = utils.normalize_gvkey(ccm['gvkey'])
    
    # Simple merge on gvkey
    # pyrefly: ignore [bad-argument-type]
    merged = pd.merge(compustat, ccm, on='gvkey', how='inner')
    
    # Date filtering for link validity
    merged['datadate'] = pd.to_datetime(merged['datadate'])
    merged['linkdt'] = pd.to_datetime(merged['linkdt'])
    merged['linkenddt'] = pd.to_datetime(merged['linkenddt']).fillna(pd.Timestamp('today'))
    
    merged = merged[(merged['datadate'] >= merged['linkdt']) & (merged['datadate'] <= merged['linkenddt'])]
    
    # Rename lpermno to permno for CRSP merge
    # pyrefly: ignore [no-matching-overload]
    merged = merged.rename(columns={'lpermno': 'permno'})
    
    # 3. Apply Standard Filters
    utils.logger.info("Applying universe filters (US Inc, Non-Fin/Util)...")
    
    # US Incorporated (fic = 'USA')
    if 'fic' in merged.columns:
        merged = merged[merged['fic'] == 'USA']
    
    # Exclude Financials (6000-6999) and Utilities (4900-4949)
    merged['sich'] = pd.to_numeric(merged['sich'], errors='coerce')
    
    condition_fin = (merged['sich'] >= 6000) & (merged['sich'] <= 6999)
    condition_util = (merged['sich'] >= 4900) & (merged['sich'] <= 4949)
    
    # pyrefly: ignore [deprecated]
    merged = merged[~(condition_fin | condition_util)]
    
    # 4. Merge with CRSP to check Share Code (SHRCD)
    utils.logger.info("Checking Share Codes (10, 11)...")
    
    crsp['date'] = pd.to_datetime(crsp['date'])
    crsp['year'] = crsp['date'].dt.year
    
    # pyrefly: ignore [missing-attribute]
    merged['year'] = merged['datadate'].dt.year
    
    # Get annual share code snapshot (first of year)
    crsp_annual = crsp.groupby(['permno', 'year'])['shrcd'].first().reset_index()
    
    # pyrefly: ignore [bad-argument-type]
    final_df = pd.merge(merged, crsp_annual, on=['permno', 'year'], how='inner')
    
    # Require common stock (SHRCD 10 or 11)
    final_df = final_df[final_df['shrcd'].isin([10, 11])]
    
    # 5. Output
    columns_to_keep = ['gvkey', 'permno', 'fyear', 'datadate', 'sich', 'naics', 'at', 'oibdp', 'prcc_f', 'csho', 'ceq', 'dltt', 'dlc', 'xrd', 'capx']
    available_cols = [c for c in columns_to_keep if c in final_df.columns]
    firm_year_base = final_df[available_cols].copy()
    
    utils.logger.info(f"Phase 0 Complete. Generated {len(firm_year_base)} firm-year observations.")
    
    # pyrefly: ignore [missing-attribute]
    firm_year_base.to_parquet(config.FIRM_YEAR_BASE_PATH)
    return firm_year_base

if __name__ == "__main__":
    run_phase0()
