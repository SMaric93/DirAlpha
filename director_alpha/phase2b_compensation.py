import pandas as pd
import numpy as np
from scipy.stats import norm
from . import config, io, db, log

logger = log.logger

def black_scholes_delta(S, K, T, r, sigma, q):
    """
    Calculate Black-Scholes Delta for a call option.
    Returns the partial derivative with respect to Stock Price (S).
    """
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return 0.0
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    delta = np.exp(-q * T) * norm.cdf(d1)
    return delta

def black_scholes_vega(S, K, T, r, sigma, q):
    """
    Calculate Black-Scholes Vega for a call option.
    Returns the change in option value for a 1% change in volatility (0.01).
    """
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return 0.0

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
    return vega * 0.01 

def calculate_volatility(crsp_msf: pd.DataFrame, min_obs=24) -> pd.DataFrame:
    """
    Calculate rolling 60-month standard deviation of returns.
    Returns a DataFrame with keys: permno, year, volatility.
    """
    logger.info("Calculating rolling volatility (60-month window)...")
    df = crsp_msf.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['permno', 'date'])
    
    # Ensure 'ret' is numeric
    df['ret'] = pd.to_numeric(df['ret'], errors='coerce')
    
    # Calculate rolling std
    # Note: This calculates std dev of monthly returns.
    df['vol_monthly'] = df.groupby('permno')['ret'].rolling(60, min_periods=min_obs).std().reset_index(0, drop=True)
    
    # Annualize: Monthly Std * sqrt(12)
    df['volatility'] = df['vol_monthly'] * np.sqrt(12)
    
    # Extract year to match with Execucomp
    df['year'] = df['date'].dt.year
    
    # Take the volatility at the end of the fiscal year (or calendar year as proxy)
    # We simply take the last available observation for each permno-year
    vol_annual = df.groupby(['permno', 'year'])['volatility'].last().reset_index()
    
    return vol_annual

def run_phase2b():
    logger.info("Starting Phase 2b: CEO Compensation (Delta/Vega)...")
    
    # -------------------------------------------------------------------------
    # 1. Load Inputs
    # -------------------------------------------------------------------------
    
    # A. Firm Performance (Base Universe + Financials)
    if not config.FIRM_YEAR_PERFORMANCE_PATH.exists():
        logger.error(f"Phase 1 output ({config.FIRM_YEAR_PERFORMANCE_PATH}) not found. Run Phase 1 first.")
        return

    firm_year = pd.read_parquet(config.FIRM_YEAR_PERFORMANCE_PATH)
    logger.info(f"Loaded Firm-Year Performance: {len(firm_year)} rows")
    
    # B. Execucomp
    logger.info("Fetching/Loading Execucomp...")
    execucomp = io.load_or_fetch(config.RAW_EXECUCOMP_PATH, db.fetch_execucomp)
    
    # C. Treasury Yields
    logger.info("Fetching/Loading Treasury Yields...")
    treasury = io.load_or_fetch(config.RAW_TREASURY_YIELDS_PATH, db.fetch_treasury_yields)
    
    # D. CRSP Monthly Stock File (for Volatility)
    logger.info("Fetching/Loading CRSP MSF...")
    crsp_msf = io.load_or_fetch(config.RAW_CRSP_PATH, db.fetch_crsp_msf)
    
    # E. Compustat (for Dividends check)
    # Ideally 'dv' is in firm_year, but if Phase 0 wasn't re-run, we fetch it.
    if 'dv' not in firm_year.columns:
        logger.info("'dv' missing in firm_year. Fetching Compustat to supplement...")
        compustat = io.load_or_fetch(config.RAW_COMPUSTAT_PATH, db.fetch_compustat_funda)
        comp_dv = compustat[['gvkey', 'fyear', 'dv']].drop_duplicates(subset=['gvkey', 'fyear'])
        # Merge
        firm_year = pd.merge(firm_year, comp_dv, on=['gvkey', 'fyear'], how='left')

    # -------------------------------------------------------------------------
    # 2. Prepare Data Variables
    # -------------------------------------------------------------------------
    
    # Dividend Yield
    # d = dv / (prcc_f * csho) (or mkvalt)
    firm_year['mkt_val'] = firm_year['prcc_f'] * firm_year['csho']
    firm_year['div_yield'] = firm_year['dv'] / firm_year['mkt_val']
    firm_year['div_yield'] = firm_year['div_yield'].fillna(0) # Assume 0 if missing
    
    # Volatility
    volatility_df = calculate_volatility(crsp_msf)
    
    # Risk-Free Rate (Annualized from Treasury)
    treasury['date'] = pd.to_datetime(treasury['date'])
    treasury['year'] = treasury['date'].dt.year
    treasury['i10'] = pd.to_numeric(treasury['i10'], errors='coerce')
    # Convert percentage to decimal (e.g., 5.0 -> 0.05)
    treasury['rf'] = treasury['i10'] / 100.0
    rf_df = treasury.groupby('year')['rf'].last().reset_index()
    
    # -------------------------------------------------------------------------
    # 3. Merge to Execucomp
    # -------------------------------------------------------------------------
    
    # Base: Execucomp
    # Join Firm Year info (Price, Dividend Yield, Permno)
    # Note: Execucomp has 'gvkey', 'year'. FirmYear has 'gvkey', 'fyear'.
    df = pd.merge(execucomp, firm_year[['gvkey', 'fyear', 'prcc_f', 'div_yield', 'permno']], 
                  left_on=['gvkey', 'year'], right_on=['gvkey', 'fyear'], how='inner')
    
    # Join Volatility (via permno, year)
    df = pd.merge(df, volatility_df, on=['permno', 'year'], how='left')
    
    # Join Risk Free Rate (via year)
    df = pd.merge(df, rf_df, on=['year'], how='left')
    
    # -------------------------------------------------------------------------
    # 4. Calculate Core & Guay (2002) Incentives
    # -------------------------------------------------------------------------
    
    logger.info("Calculating Delta and Vega (Core & Guay 2002)...")
    
    # Clean inputs
    df['prcc_f'] = pd.to_numeric(df['prcc_f'], errors='coerce')
    df['volatility'] = pd.to_numeric(df['volatility'], errors='coerce')
    df['rf'] = pd.to_numeric(df['rf'], errors='coerce')
    df['div_yield'] = pd.to_numeric(df['div_yield'], errors='coerce')
    
    # Filter valid rows for calculation
    valid_mask = df[['prcc_f', 'volatility', 'rf']].notna().all(axis=1)
    df_calc = df[valid_mask].copy()
    
    # Define Calculation Logic
    def calculate_row(row):
        S = row['prcc_f']
        r = row['rf']
        sigma = row['volatility']
        q = row['div_yield']
        
        d_grant, v_grant = 0.0, 0.0
        d_exer, v_exer = 0.0, 0.0
        d_unexer, v_unexer = 0.0, 0.0
        
        # A. New Grants
        if pd.notna(row.get('option_awards_num')) and row['option_awards_num'] > 0:
            num = row['option_awards_num']
            K = S # At-the-money assumption
            T = 10
            d_grant = num * black_scholes_delta(S, K, T, r, sigma, q)
            v_grant = num * black_scholes_vega(S, K, T, r, sigma, q)
            
        # B. Exercisable Options
        if pd.notna(row.get('opt_unex_exer_num')) and row['opt_unex_exer_num'] > 0:
            num = row['opt_unex_exer_num']
            val = row.get('opt_unex_exer_est_val', 0)
            if pd.isna(val) or val < 0: val = 0
            
            # Estimate Strike K
            est_intrinsic = val / num
            if est_intrinsic >= S:
                K = 0.1 # Deep ITM proxy
            else:
                K = S - est_intrinsic
            
            T = 4
            d_exer = num * black_scholes_delta(S, K, T, r, sigma, q)
            v_exer = num * black_scholes_vega(S, K, T, r, sigma, q)
            
        # C. Unexercisable Options
        if pd.notna(row.get('opt_unex_unexer_num')) and row['opt_unex_unexer_num'] > 0:
            num = row['opt_unex_unexer_num']
            val = row.get('opt_unex_unexer_est_val', 0)
            if pd.isna(val) or val < 0: val = 0
            
            est_intrinsic = val / num
            if est_intrinsic >= S:
                K = 0.1
            else:
                K = S - est_intrinsic
                
            T = 7
            d_unexer = num * black_scholes_delta(S, K, T, r, sigma, q)
            v_unexer = num * black_scholes_vega(S, K, T, r, sigma, q)
            
        return pd.Series([d_grant, v_grant, d_exer, v_exer, d_unexer, v_unexer])

    # Apply
    results = df_calc.apply(calculate_row, axis=1)
    results.columns = ['d_grant', 'v_grant', 'd_exer', 'v_exer', 'd_unexer', 'v_unexer']
    
    df_calc = pd.concat([df_calc, results], axis=1)
    
    # Aggregation
    df_calc['total_option_delta'] = df_calc['d_grant'] + df_calc['d_exer'] + df_calc['d_unexer']
    df_calc['total_option_vega'] = df_calc['v_grant'] + df_calc['v_exer'] + df_calc['v_unexer']
    
    # Share Ownership Delta
    # Delta of a share is 1.
    df_calc['shrown_excl_opts'] = pd.to_numeric(df_calc['shrown_excl_opts'], errors='coerce').fillna(0)
    df_calc['total_delta_shares'] = df_calc['total_option_delta'] + df_calc['shrown_excl_opts']
    
    # Dollar Delta: $ Change in wealth for 1% change in price
    # = Total Shares Equivalent * S * 0.01
    df_calc['delta'] = df_calc['total_delta_shares'] * df_calc['prcc_f'] * 0.01
    
    # Dollar Vega: $ Change in wealth for 1% change in volatility
    # Our BS_Vega returns change for 1% change.
    df_calc['vega'] = df_calc['total_option_vega']
    
    # -------------------------------------------------------------------------
    # 5. Save Output
    # -------------------------------------------------------------------------
    
    output_cols = [
        'gvkey', 'year', 'execid', 'delta', 'vega', 
        'total_delta_shares', 'total_option_vega'
    ]
    
    final_df = df_calc[output_cols].copy()
    logger.info(f"Calculated compensation for {len(final_df)} CEO-years.")
    
    final_df.to_parquet(config.FIRM_YEAR_COMPENSATION_PATH)
    logger.info(f"Saved to {config.FIRM_YEAR_COMPENSATION_PATH}")

if __name__ == "__main__":
    run_phase2b()
