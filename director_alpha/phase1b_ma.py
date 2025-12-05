import pandas as pd
import numpy as np
import subprocess
from pathlib import Path
from . import config, io, db, log, transform

logger = log.logger

MAPPING_URL = "https://raw.githubusercontent.com/michaelewens/SDC-to-Compustat-Mapping/master/dealnum_to_gvkey.csv"

def download_mapping_file():
    """
    Download the SDC to Compustat mapping file if it doesn't exist.
    """
    if config.SDC_MAPPING_CSV_PATH.exists():
        logger.info("SDC Mapping file already exists.")
        return

    logger.info(f"Downloading SDC Mapping file from {MAPPING_URL}...")
    
    try:
        # Use curl to download
        cmd = ["curl", "-o", str(config.SDC_MAPPING_CSV_PATH), MAPPING_URL]
        subprocess.run(cmd, check=True)
        logger.info("Download successful.")
    except Exception as e:
        logger.error(f"Failed to download mapping file: {e}")

def calculate_ma_variables(deals: pd.DataFrame, firm_year: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate deal data to firm-year level and calculate M&A metrics.
    """
    logger.info("Calculating Firm-Level M&A Variables...")
    
    df = deals.copy()
    
    # Ensure numeric types
    df['deal_value'] = pd.to_numeric(df['deal_value'], errors='coerce').fillna(0)
    df['pct_cash'] = pd.to_numeric(df['pct_cash'], errors='coerce').fillna(0)
    df['pct_stock'] = pd.to_numeric(df['pct_stock'], errors='coerce').fillna(0)
    
    # Year of announcement
    df['year'] = df['date_announced'].dt.year
    
    # Filter: Only deals with a valid Acquirer GVKEY
    df = df[df['acquirer_gvkey'].notna()].copy()
    
    # --- Deal Types ---
    
    # 1. Diversifying (SIC mismatch)
    # Handle mixed types or missing
    df['asic'] = pd.to_numeric(df['asic'], errors='coerce')
    df['tsic'] = pd.to_numeric(df['tsic'], errors='coerce')
    
    # 2-digit SIC match
    # Safe division / floor
    df['asic_2'] = np.floor(df['asic'] / 100)
    df['tsic_2'] = np.floor(df['tsic'] / 100)
    
    df['is_diversifying'] = (df['asic_2'] != df['tsic_2']) & df['asic_2'].notna() & df['tsic_2'].notna()
    
    # 2. Cross-Border
    # Assuming 'cross_border' column is 'Y' or similar, or distinct nation codes
    # If 'cross_border' is boolean/string
    # Let's check the content. Usually 'Yes'/'No' or 'Y'/'N'.
    # We can also use nations if available.
    # db.py fetches 'cross_border'.
    df['is_cross_border'] = df['cross_border'].astype(str).str.upper().str.startswith('Y')
    
    # 3. Public Target
    # 'status' column. Usually 'Public', 'Private', 'Subsidiary'.
    df['is_public_target'] = df['status'].astype(str).str.lower().str.contains('public')
    
    # 4. Full Cash / Full Stock
    df['is_all_cash'] = df['pct_cash'] >= 99.0
    df['is_all_stock'] = df['pct_stock'] >= 99.0
    
    # --- Aggregation ---
    
    # Group by Acquirer GVKEY and Year
    # Note: 'acquirer_gvkey' is normalized to string zfill(6) in run_phase1b
    
    agg_funcs = {
        'deal_no': 'count',           # Number of deals
        'deal_value': 'sum',          # Total Deal Value
        'is_diversifying': 'sum',     # Count of diversifying
        'is_cross_border': 'sum',     # Count of cross-border
        'is_public_target': 'sum',    # Count of public targets
        'is_all_cash': 'sum',         # Count of all-cash deals
        'is_all_stock': 'sum'         # Count of all-stock deals
    }
    
    ma_firm_year = df.groupby(['acquirer_gvkey', 'year']).agg(agg_funcs).reset_index()
    
    ma_firm_year = ma_firm_year.rename(columns={
        'acquirer_gvkey': 'gvkey',
        'year': 'fyear',
        'deal_no': 'n_deals',
        'deal_value': 'ma_volume',
        'is_diversifying': 'n_diversifying',
        'is_cross_border': 'n_cross_border',
        'is_public_target': 'n_public_targets',
        'is_all_cash': 'n_all_cash',
        'is_all_stock': 'n_all_stock'
    })
    
    # --- Normalization (Intensity) ---
    
    # Merge with Firm Year Base (Assets) to calculate intensity
    # We generally use Lagged Assets (Size at start of year)
    
    if firm_year.empty:
        logger.warning("Firm year data empty. Skipping intensity ratios.")
        return ma_firm_year
        
    # Prepare Firm Year
    # Need gvkey, fyear, at_lag (or calculate it)
    fy = firm_year[['gvkey', 'fyear', 'at']].copy()
    fy['gvkey'] = transform.normalize_gvkey(fy['gvkey'])
    
    # Calculate lag if not present (it is calculated in phase 1 but let's be safe or re-calc)
    fy = fy.sort_values(['gvkey', 'fyear'])
    fy['at_lag'] = fy.groupby('gvkey')['at'].shift(1)
    
    # Merge
    # Left merge onto M&A panel? Or onto Firm Year?
    # Usually we want the M&A variables available for the whole universe (0 if no deal).
    # So we merge M&A stats onto the full firm-year panel.
    
    final_panel = pd.merge(fy[['gvkey', 'fyear', 'at_lag']], ma_firm_year, on=['gvkey', 'fyear'], how='left')
    
    # Fill NaNs with 0 for M&A columns (implies no deals that year)
    ma_cols = ['n_deals', 'ma_volume', 'n_diversifying', 'n_cross_border', 
               'n_public_targets', 'n_all_cash', 'n_all_stock']
    
    for c in ma_cols:
        final_panel[c] = final_panel[c].fillna(0)
        
    # Calculate Ratios
    # Acquisition Ratio (Volume / Lagged Assets)
    final_panel['ma_intensity'] = final_panel['ma_volume'] / final_panel['at_lag']
    
    # Handle division by zero / missing assets
    final_panel['ma_intensity'] = final_panel['ma_intensity'].replace([np.inf, -np.inf], np.nan)
    
    # Select final columns
    return final_panel

def run_phase1b():
    logger.info("Starting Phase 1b: SDC M&A Data Integration...")

    # 1. Download Mapping
    download_mapping_file()
    
    if not config.SDC_MAPPING_CSV_PATH.exists():
        logger.error("Mapping file missing. Aborting Phase 1b.")
        return

    # 2. Load SDC Data
    logger.info("Fetching/Loading SDC M&A Data...")
    try:
        sdc_df = io.load_or_fetch(config.RAW_SDC_MA_PATH, db.fetch_sdc_ma)
    except Exception as e:
        logger.error(f"Failed to fetch SDC data: {e}")
        return
        
    if sdc_df.empty:
        logger.warning("SDC data is empty.")
        return

    # 3. Load Mapping File
    logger.info("Loading Mapping File...")
    mapping = pd.read_csv(config.SDC_MAPPING_CSV_PATH)
    
    # 4. Merge SDC with Mapping
    mapping['DealNumber'] = pd.to_numeric(mapping['DealNumber'], errors='coerce')
    sdc_df['deal_no'] = pd.to_numeric(sdc_df['deal_no'], errors='coerce')
    
    logger.info(f"Merging SDC ({len(sdc_df)} rows) with Mapping ({len(mapping)} rows)...")
    
    merged = pd.merge(sdc_df, mapping, left_on='deal_no', right_on='DealNumber', how='left')
    
    # 5. Clean and Format
    merged = merged.rename(columns={
        'agvkey': 'acquirer_gvkey',
        'tgvkey': 'target_gvkey',
        'da': 'date_announced',
        'de': 'date_effective',
        'val': 'deal_value'
    })
    
    merged['acquirer_gvkey'] = merged['acquirer_gvkey'].fillna(-1).astype(int).astype(str).str.zfill(6).replace('-00001', pd.NA)
    merged['target_gvkey'] = merged['target_gvkey'].fillna(-1).astype(int).astype(str).str.zfill(6).replace('-00001', pd.NA)
    
    merged['date_announced'] = pd.to_datetime(merged['date_announced'])
    merged['date_effective'] = pd.to_datetime(merged['date_effective'])
    
    # Save Detailed List
    merged.to_parquet(config.MA_DEALS_PATH)
    logger.info(f"Saved {len(merged)} enriched deals to {config.MA_DEALS_PATH}")
    
    # 6. Calculate Firm-Level Variables
    # Load Firm Year Performance (Phase 1) for Assets
    if config.FIRM_YEAR_PERFORMANCE_PATH.exists():
        firm_year = pd.read_parquet(config.FIRM_YEAR_PERFORMANCE_PATH)
    else:
        logger.warning("Phase 1 output missing. Calculating M&A vars without intensity normalization.")
        firm_year = pd.DataFrame()
        
    ma_vars = calculate_ma_variables(merged, firm_year)
    
    # Save Firm-Level Panel
    ma_vars.to_parquet(config.FIRM_YEAR_MA_PATH)
    logger.info(f"Phase 1b Complete. Saved firm-level M&A variables to {config.FIRM_YEAR_MA_PATH}")

if __name__ == "__main__":
    run_phase1b()