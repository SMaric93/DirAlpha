import pandas as pd
import numpy as np
from . import config

def calculate_financials(df):
    """
    Calculate ROA, Tobin's Q, and Controls.
    """
    # Sort for lagging
    df = df.sort_values(['gvkey', 'fyear'])
    
    # Lagged Assets
    df['at_lag'] = df.groupby('gvkey')['at'].shift(1)
    
    # ROA = OIBDP / Lagged AT
    df['roa'] = df['oibdp'] / df['at_lag']
    
    # Tobin's Q
    # Q = (AT + (PRCC_F * CSHO) - CEQ) / AT
    # Note: Using current AT as denominator per formula in prompt, 
    # though sometimes lagged is used. Sticking to prompt: "AT" usually implies current unless specified.
    # Prompt says: Q = (AT + (PRCC_F * CSHO) - CEQ) / AT.
    market_val_equity = df['prcc_f'] * df['csho']
    df['tobins_q'] = (df['at'] + market_val_equity - df['ceq']) / df['at']
    
    # Controls
    # Size = Log(AT)
    df['size'] = np.log(df['at'])
    
    # Leverage = (DLTT + DLC) / AT
    df['leverage'] = (df['dltt'].fillna(0) + df['dlc'].fillna(0)) / df['at']
    
    # R&D Intensity = XRD / AT (missing XRD -> 0)
    df['xrd'] = df['xrd'].fillna(0)
    df['rd_intensity'] = df['xrd'] / df['at']
    
    # CAPEX = CAPX / AT
    df['capex_intensity'] = df['capx'] / df['at']
    
    # Firm Age (Placeholder - requires CRSP listing date which we might not have fully merged yet)
    # For now, we will skip or use a proxy if available. 
    # If we had 'linkdt' from CCM, we could use that as a proxy for public listing start.
    # Let's assume we can compute it later or if 'linkdt' is preserved.
    # We'll add a placeholder column.
    df['firm_age'] = np.nan 
    
    return df

def industry_adjust(df, cols=['roa', 'tobins_q']):
    """
    Subtract industry-year median.
    Using 2-digit SIC as proxy for FF48 if FF48 not available.
    """
    # Create 2-digit SIC
    df['sic2'] = df['sich'].fillna(0).astype(int) // 100
    
    for col in cols:
        # Calculate median by year and industry
        median = df.groupby(['fyear', 'sic2'])[col].transform('median')
        df[f'{col}_adj'] = df[col] - median
        
    return df

def winsorize(df, cols, limits=(0.01, 0.01)):
    """
    Winsorize variables at 1% and 99%.
    """
    from scipy.stats.mstats import winsorize as scipy_winsorize
    
    for col in cols:
        # Handle NaNs before winsorizing or use a method that handles them
        # Scipy winsorize doesn't handle NaNs well in place.
        # We'll use quantile clipping for robustness with NaNs.
        lower = df[col].quantile(limits[0])
        upper = df[col].quantile(1 - limits[1])
        df[col] = df[col].clip(lower=lower, upper=upper)
        
    return df

def run_phase1():
    print("Starting Phase 1: Firm Performance Panel...")
    
    # Load Phase 0 output
    try:
        df = pd.read_parquet(config.FIRM_YEAR_BASE_PATH)
    except FileNotFoundError:
        print("Phase 0 output not found. Please run Phase 0 first.")
        return

    # 1. Calculate Financials
    df = calculate_financials(df)
    
    # 2. Industry Adjustment
    df = industry_adjust(df, cols=['roa', 'tobins_q'])
    
    # 3. Winsorization
    # Winsorize continuous financial variables
    cols_to_winsorize = ['roa', 'tobins_q', 'size', 'leverage', 'rd_intensity', 'capex_intensity', 'roa_adj', 'tobins_q_adj']
    # Filter for columns that actually exist (in case of all-NaNs or errors)
    cols_to_winsorize = [c for c in cols_to_winsorize if c in df.columns]
    
    # Winsorize annually? Prompt says "Winsorize ... annually".
    # So we should group by year and apply.
    # However, standard practice is often over the whole sample or annually. 
    # "Winsorize ... annually" implies calculating percentiles per year.
    
    for year, group in df.groupby('fyear'):
        # We need to update the original dataframe. 
        # Iterating groups doesn't update in place easily.
        # Better to use transform or apply.
        pass # Logic below
        
    # Vectorized annual winsorization
    def winsorize_series(x):
        lower = x.quantile(0.01)
        upper = x.quantile(0.99)
        return x.clip(lower=lower, upper=upper)

    for col in cols_to_winsorize:
        df[col] = df.groupby('fyear')[col].transform(winsorize_series)

    # 4. Stock Performance (CRSP)
    # Load CRSP data
    try:
        crsp = pd.read_parquet(config.RAW_CRSP_PATH)
        print("Loaded CRSP data from local parquet.")
    except FileNotFoundError:
        print("CRSP local file not found. Attempting to load from WRDS...")
        db = config.get_wrds_connection()
        if db:
            print("Fetching CRSP data from WRDS...")
            crsp_query = f"""
                SELECT permno, date, prc, ret
                FROM {config.WRDS_CRSP_MSF}
                WHERE date >= '2000-01-01'
            """
            crsp = db.raw_sql(crsp_query)
            crsp.to_parquet(config.RAW_CRSP_PATH)
        else:
            print("Could not load CRSP data.")
            crsp = pd.DataFrame()

    if not crsp.empty:
        # Calculate Annual Returns
        # Simple approximation: Compound monthly returns by permno-year
        crsp['date'] = pd.to_datetime(crsp['date'])
        crsp['year'] = crsp['date'].dt.year
        
        # Fill missing returns with 0 for compounding? Or skip?
        # Better to drop missing.
        crsp = crsp.dropna(subset=['ret'])
        
        # Group by permno, year and compound
        # (1+r1)*(1+r2)... - 1
        annual_ret = crsp.groupby(['permno', 'year'])['ret'].apply(lambda x: np.prod(1 + x) - 1).reset_index()
        annual_ret = annual_ret.rename(columns={'ret': 'ann_ret'})
        
        # Merge to df
        # df has 'permno' and 'fyear'. We'll match 'year' to 'fyear'.
        # Note: Fiscal year vs Calendar year. This is an approximation.
        
        df = pd.merge(df, annual_ret, left_on=['permno', 'fyear'], right_on=['permno', 'year'], how='left')
        df = df.drop(columns=['year'])
        
        print("Calculated annual stock returns.")
    else:
        print("Skipping stock returns calculation.")
    
    # Save (overwriting or new file? Prompt implies building up. We can overwrite or save as new step)
    # Let's save as firm_year_performance.parquet to be safe.
    output_path = config.INTERMEDIATE_DIR / "firm_year_performance.parquet"
    df.to_parquet(output_path)
    return df

if __name__ == "__main__":
    run_phase1()
