import pandas as pd
import numpy as np
from . import config, io, db, transform, log

logger = log.logger

def calculate_financials(df: pd.DataFrame) -> pd.DataFrame:
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
    
    # Firm Age (Placeholder - requires founding date or first appearance)
    df['firm_age'] = np.nan 
    
    return df

def calculate_stock_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate annual stock returns from CRSP monthly data and merge.
    """
    logger.info("Calculating Stock Performance...")
    
    # Load CRSP
    crsp = io.load_or_fetch(
        config.RAW_CRSP_PATH,
        db.fetch_crsp_msf
    )

    if crsp.empty:
        logger.warning("Skipping stock returns calculation (CRSP data missing).")
        return df

    # Prepare CRSP
    crsp['date'] = pd.to_datetime(crsp['date'])
    crsp['year'] = crsp['date'].dt.year
    crsp = crsp.dropna(subset=['ret'])
    
    # Compound annual returns: (1+r1)*(1+r2)... - 1
    annual_ret = crsp.groupby(['permno', 'year'])['ret'].apply(lambda x: np.prod(1 + x) - 1).reset_index()
    annual_ret = annual_ret.rename(columns={'ret': 'ann_ret'})
    
    # Merge to df
    # Ensure types match
    df['permno'] = df['permno'].astype(int)
    annual_ret['permno'] = annual_ret['permno'].astype(int)
    
    df = pd.merge(df, annual_ret, left_on=['permno', 'fyear'], right_on=['permno', 'year'], how='left')
    df = df.drop(columns=['year'])
    
    logger.info("Calculated annual stock returns.")
    return df

def run_phase1():
    """
    Phase 1: Firm Performance Panel
    """
    logger.info("Starting Phase 1: Firm Performance Panel...")
    
    # 1. Load Phase 0 output
    if not config.FIRM_YEAR_BASE_PATH.exists():
        logger.error("Phase 0 output not found. Please run Phase 0 first.")
        return

    df = pd.read_parquet(config.FIRM_YEAR_BASE_PATH)

    # 2. Calculate Financials
    df = calculate_financials(df)
    
    # 3. Industry Adjustment
    logger.info("Performing Industry Adjustments...")
    df = transform.industry_adjust(df, cols=['roa', 'tobins_q'])
    
    # 4. Winsorization
    logger.info("Winsorizing variables...")
    df = transform.apply_winsorization(
        df, 
        cols=config.PERFORMANCE_COLS_TO_WINSORIZE, 
        group_col='fyear', 
        limits=config.WINSORIZE_LIMITS
    )

    # 5. Stock Performance
    df = calculate_stock_performance(df)
    
    # Save
    df.to_parquet(config.FIRM_YEAR_PERFORMANCE_PATH)
    logger.info(f"Phase 1 Complete. Saved to {config.FIRM_YEAR_PERFORMANCE_PATH}")

if __name__ == "__main__":
    run_phase1()
