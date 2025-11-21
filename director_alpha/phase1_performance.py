import pandas as pd
import numpy as np
from . import config, utils

def calculate_financials(df):
    """
    Calculate ROA, Tobin's Q, and Controls.
    """
    df = df.copy()
    # Sort for lagging
    df = df.sort_values(['gvkey', 'fyear'])
    
    # Lagged Assets
    df['at_lag'] = df.groupby('gvkey')['at'].shift(1)
    
    # ROA = OIBDP / Lagged AT
    df['roa'] = df['oibdp'] / df['at_lag']
    
    # Tobin's Q
    market_val_equity = df['prcc_f'] * df['csho']
    df['tobins_q'] = (df['at'] + market_val_equity - df['ceq']) / df['at']
    
    # Controls
    df['size'] = np.log(df['at'])
    # Leverage
    df['leverage'] = (df['dltt'].fillna(0) + df['dlc'].fillna(0)) / df['at']
    # R&D
    df['xrd'] = df['xrd'].fillna(0)
    df['rd_intensity'] = df['xrd'] / df['at']
    # CAPEX
    df['capex_intensity'] = df['capx'] / df['at']
    
    # Firm Age (Placeholder)
    df['firm_age'] = np.nan 
    
    return df

def run_phase1():
    utils.logger.info("Starting Phase 1: Firm Performance Panel...")
    
    # Load Phase 0 output
    if not config.FIRM_YEAR_BASE_PATH.exists():
        utils.logger.error("Phase 0 output not found. Please run Phase 0 first.")
        return

    df = pd.read_parquet(config.FIRM_YEAR_BASE_PATH)

    # 1. Calculate Financials
    df = calculate_financials(df)
    
    # 2. Industry Adjustment
    utils.logger.info("Performing Industry Adjustments...")
    df = utils.industry_adjust(df, cols=['roa', 'tobins_q'])
    
    # 3. Winsorization
    utils.logger.info("Winsorizing variables (1% annual)...")
    cols_to_winsorize = ['roa', 'tobins_q', 'size', 'leverage', 'rd_intensity', 'capex_intensity', 'roa_adj', 'tobins_q_adj']
    df = utils.apply_winsorization(df, cols_to_winsorize, group_col='fyear')

    # 4. Stock Performance (CRSP)
    utils.logger.info("Calculating Stock Performance...")
    
    # Load CRSP (reusing the fetcher from Phase 0 is safe as it contains 'ret')
    crsp = utils.load_or_fetch(
        config.RAW_CRSP_PATH,
        utils.fetch_crsp_msf
    )

    if not crsp.empty:
        # Calculate Annual Returns
        crsp['date'] = pd.to_datetime(crsp['date'])
        crsp['year'] = crsp['date'].dt.year
        
        crsp = crsp.dropna(subset=['ret'])
        
        # Compound annual returns: (1+r1)*(1+r2)... - 1
        annual_ret = crsp.groupby(['permno', 'year'])['ret'].apply(lambda x: np.prod(1 + x) - 1).reset_index()
        annual_ret = annual_ret.rename(columns={'ret': 'ann_ret'})
        
        # Merge to df
        df = pd.merge(df, annual_ret, left_on=['permno', 'fyear'], right_on=['permno', 'year'], how='left')
        df = df.drop(columns=['year'])
        
        utils.logger.info("Calculated annual stock returns.")
    else:
        utils.logger.warning("Skipping stock returns calculation (CRSP data missing).")
    
    # Save
    output_path = config.INTERMEDIATE_DIR / "firm_year_performance.parquet"
    df.to_parquet(output_path)
    utils.logger.info(f"Phase 1 Complete. Saved to {output_path}")
    return df

if __name__ == "__main__":
    run_phase1()