"""
Event Study Analysis: CEO Appointments (Revised and Enhanced)

Goal:
- Calculate Cumulative Abnormal Returns (CAR) and Buy-and-Hold Abnormal Returns (BHAR).
- Models: CAPM, Fama-French 3-Factor (FF3), Fama-French 5-Factor (FF5).
- Methodology: Standard event study with precise trading day indexing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
from typing import Dict, Optional, Tuple
import logging

# ---------------------------------------------------------------------
# Configuration and Dependencies
# ---------------------------------------------------------------------
# This implementation assumes the existence of 'config' and 'utils' modules.
# 'utils' must provide data fetching (CRSP, FF5 Factors, CCM), logging, normalization, and winsorization.
# CRITICAL: utils.fetch_crsp_dsf must incorporate delisting returns (DLRET) to avoid bias, crucial for BHAR.

try:
    from . import config, utils
except ImportError:
    # Fallback/Mock definitions if imports fail (for analysis purposes).
    # In a production environment, these must be implemented correctly.
    print("[Warning] 'config' or 'utils' modules not found. Using Mocks for structural analysis.")

    class MockConfig:
        # Define necessary paths
        CEO_SPELLS_PATH = Path("data/ceo_spells.parquet")
        RAW_CCM_PATH = Path("data/raw/ccm_link.parquet")
        RAW_CRSP_DSF_PATH = Path("data/raw/crsp_dsf.parquet")
        RAW_FF5_FACTORS_DAILY_PATH = Path("data/raw/ff5_factors_daily.parquet")
        EVENT_STUDY_RESULTS_PATH = Path("results/event_study_results.parquet")
        EVENT_STUDY_RESULTS_CSV_PATH = Path("results/event_study_results.csv")

    class MockUtils:
        logger = logging.getLogger(__name__)
        def load_or_fetch(self, *args, **kwargs): return pd.DataFrame()
        def normalize_gvkey(self, s): return s.astype(str).str.zfill(6)
        def get_db(self): return None
        def apply_winsorization(self, df, cols, limits=(0.01, 0.99), **kwargs):
            # Mock implementation of winsorization
            df_out = df.copy()
            for col in cols:
                if col in df_out.columns and pd.api.types.is_numeric_dtype(df_out[col]):
                    q = df_out[col].quantile([limits[0], limits[1]])
                    df_out[col] = df_out[col].clip(lower=q.iloc[0], upper=q.iloc[1])
            return df_out
        # Mock fetch functions
        def fetch_ccm_link(self): pass
        def fetch_crsp_dsf(self, **kwargs): pass
        def fetch_fama_french_5_daily(self, **kwargs): pass

    config = MockConfig()
    utils = MockUtils()

# ---------------------------------------------------------------------
# Constants (Defined strictly in Trading Days relative to T=0)
# ---------------------------------------------------------------------

# Estimation Window: [T-282, T-31] 
# Length 252 days, ending 30 days before the event (T=0) to prevent contamination.
ESTIMATION_WINDOW_LENGTH = 252
ESTIMATION_WINDOW_END = -31
ESTIMATION_WINDOW_START = ESTIMATION_WINDOW_END - ESTIMATION_WINDOW_LENGTH + 1 # T-282

# Minimum observations required within the estimation window
MIN_OBS_ESTIMATION = 126 

# Event Windows (Inclusive)
CAR_WINDOWS = {
    'car_1_1': (-1, 1),
    'car_3_3': (-3, 3),
    'car_5_5': (-5, 5),
}

# BHAR Windows (Inclusive, starting at T=0)
BHAR_WINDOWS = {
    'bhar_1y': (0, 251),   # T=0 to T+251 (252 days total)
    'bhar_3y': (0, 755),   # T=0 to T+755 (756 days total)
}

# Factor definitions
MODEL_SPECS = {
    'capm': ['mktrf'],
    'ff3': ['mktrf', 'smb', 'hml'],
    'ff5': ['mktrf', 'smb', 'hml', 'rmw', 'cma']
}

# ---------------------------------------------------------------------
# Core Event Study Functions
# ---------------------------------------------------------------------

def align_event_data(returns_series: pd.Series, factors_df: pd.DataFrame, event_date: pd.Timestamp) -> Tuple[Optional[pd.DataFrame], Optional[int]]:
    """
    Aligns firm-specific returns with market factors and identifies the event day index (T=0).

    Args:
        returns_series: Daily returns for the specific PERMNO (indexed by date).
        factors_df: Daily Fama-French factors (indexed by date).
        event_date: The reported date of the event.

    Returns:
        Aligned data (DataFrame) and the index location (int) of T=0, or (None, None).
    """
    # Join returns and factors. Inner join ensures we only keep days where both exist.
    data = pd.concat([returns_series, factors_df], axis=1, join='inner')
    
    # Ensure essential data ('ret' and 'rf') are present. 
    # Specific factor availability (e.g., RMW/CMA) will be handled during model fitting.
    data = data.dropna(subset=['ret', 'rf'])
    data = data.sort_index()

    if data.empty:
        return None, None
        
    # Identify T=0. If the event date is a non-trading day (e.g., weekend), 
    # T=0 is the next trading day ('bfill' method).
    try:
        # get_indexer returns the integer location (iloc) for the given date label.
        t0_loc = data.index.get_indexer([event_date], method='bfill')[0]
    except KeyError:
        return None, None
        
    # If the event_date is beyond the end of the available data, get_indexer returns -1.
    if t0_loc == -1 or t0_loc >= len(data):
        return None, None
        
    return data, t0_loc

def fit_models(data_slice: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Fit CAPM, FF3, and FF5 models using OLS on the estimation window data.
    """
    results = {}

    if len(data_slice) < MIN_OBS_ESTIMATION:
        return {}
        
    # Dependent variable: Excess returns (R - Rf)
    y = (data_slice['ret'] - data_slice['rf']).values
    N = len(data_slice)
    
    for model_name, factors in MODEL_SPECS.items():
        # Prepare the independent variables (X matrix)
        X_list = [np.ones(N)] # Constant (for alpha)
        
        skip_model = False
        for factor in factors:
            if factor in data_slice.columns:
                # Ensure the factor data itself isn't missing within the estimation slice
                if data_slice[factor].isnull().any():
                    skip_model = True
                    break
                X_list.append(data_slice[factor].values)
            else:
                # If a required factor is missing in the source data (e.g. RMW/CMA), skip the model
                skip_model = True
                break
        
        if skip_model:
            continue

        X = np.vstack(X_list).T
        
        try:
            # OLS Regression using numpy's efficient least squares solver
            betas, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            
            # Store parameters
            model_params = {'const': betas[0]}
            for i, factor in enumerate(factors):
                model_params[factor] = betas[i+1]
            
            results[model_name] = model_params
            
        except np.linalg.LinAlgError:
            # Handles numerical instability or severe multicollinearity
            utils.logger.warning(f"OLS failed for model {model_name} due to LinAlgError.")
            continue
        
    return results

def calculate_expected_returns(chunk: pd.DataFrame, params: Dict[str, float]) -> pd.Series:
    """Helper to calculate E[R] based on the model parameters."""
    # E[R] = Rf + alpha + beta_i * Factor_i + ...
    # Alpha and betas were estimated on excess returns.
    
    er = chunk['rf'] + params['const']
    
    # Add factor contributions
    for factor, beta in params.items():
        if factor != 'const':
            # Check if the factor exists in the event window chunk
            if factor in chunk.columns:
                 er += beta * chunk[factor]
            else:
                # This should ideally not happen if data alignment was strict, 
                # but provides robustness against unexpected data gaps.
                raise ValueError(f"Factor {factor} missing in event window calculation.")
            
    return er

def process_event(row: pd.Series, dsf_grouped: pd.core.groupby.DataFrameGroupBy, ff_factors: pd.DataFrame) -> Dict:
    """
    Main orchestrator for a single event. Handles data alignment, estimation, and calculation.
    """
    permno = row.permno
    appt_date = row.appointment_date
    results = {}

    # A. Get PERMNO specific returns
    try:
        # Get returns, set index to date, select 'ret' column, and sort
        p_rets = dsf_grouped.get_group(permno).set_index('date')['ret'].sort_index()
    except KeyError:
        return {}
            
    # B. Align Data and find T=0
    # This step aligns the trading calendar and ensures data availability.
    event_data, t0_loc = align_event_data(p_rets, ff_factors, appt_date)
    
    if event_data is None:
        return {}
        
    # C. Define Estimation Window using precise Trading Day Indices
    # This is the critical methodological step: using indices relative to T=0.
    est_start_idx = t0_loc + ESTIMATION_WINDOW_START
    est_end_idx = t0_loc + ESTIMATION_WINDOW_END + 1 # Python slicing (iloc) is exclusive at the end
    
    if est_start_idx < 0:
        # Not enough historical data available for this event
        return {}
        
    estimation_slice = event_data.iloc[est_start_idx:est_end_idx]
    
    # D. Fit Models (CAPM, FF3, FF5)
    model_params = fit_models(estimation_slice)
    
    # Check if at least one model was successfully fitted
    if not model_params:
        return {}

    # E. Calculate Abnormal Returns (CARs and BHARs)

    # Helper to safely slice the data relative to T=0 using indices
    def get_slice(start_offset, end_offset):
        start_idx = t0_loc + start_offset
        end_idx = t0_loc + end_offset + 1 
        if start_idx < 0 or end_idx > len(event_data):
            return None
        return event_data.iloc[start_idx:end_idx]

    for model_type, params in model_params.items():

        # --- CAR Calculation ---
        # CAR = Sum(AR) = Sum(R - E[R])
        for name, (w_start, w_end) in CAR_WINDOWS.items():
            chunk = get_slice(w_start, w_end)
            if chunk is None:
                continue
            
            # Robustness: Ensure the window is complete for short-term CARs
            required_len = w_end - w_start + 1
            if len(chunk) != required_len:
                continue

            try:
                er = calculate_expected_returns(chunk, params)
                ar = chunk['ret'] - er
                results[f'{name}_{model_type}'] = ar.sum()
            except ValueError:
                continue # Skip if calculation fails due to missing factors in event window

        # --- BHAR Calculation ---
        # BHAR = Prod(1 + R) - Prod(1 + E[R])
        for name, (w_start, w_end) in BHAR_WINDOWS.items():
            chunk = get_slice(w_start, w_end)
            
            # Robustness check for long-horizon windows: require sufficient data points (e.g. 80%)
            required_len = w_end - w_start + 1
            if chunk is None or len(chunk) < required_len * 0.8:
                continue
                
            try:
                # Compounded actual return
                gross_ret = (1 + chunk['ret']).prod()
                
                # Compounded expected return (benchmark)
                er = calculate_expected_returns(chunk, params)
                gross_er = (1 + er).prod()
                
                results[f'{name}_{model_type}'] = gross_ret - gross_er
            except ValueError:
                continue

    return results

# ---------------------------------------------------------------------
# Data Preparation Helpers
# ---------------------------------------------------------------------

def link_spells_to_permno(spells: pd.DataFrame) -> pd.DataFrame:
    """Handles the time-varying link between Compustat (GVKEY) and CRSP (PERMNO) using CCM."""
    utils.logger.info("Starting CCM linking process...")

    # Load CCM Link table
    ccm = utils.load_or_fetch(config.RAW_CCM_PATH, utils.fetch_ccm_link)
    if ccm.empty:
        utils.logger.error("CCM link table is empty or failed to load.")
        return pd.DataFrame()

    # Prepare CCM
    if 'lpermno' in ccm.columns and 'permno' not in ccm.columns:
        ccm = ccm.rename(columns={'lpermno': 'permno'})
        
    ccm['linkdt'] = pd.to_datetime(ccm['linkdt'])
    # Fill missing end dates with today to ensure ongoing links are included
    ccm['linkenddt'] = pd.to_datetime(ccm['linkenddt']).fillna(pd.Timestamp.today())
    
    # Normalize GVKEYs
    spells['gvkey'] = utils.normalize_gvkey(spells['gvkey'])
    ccm['gvkey'] = utils.normalize_gvkey(ccm['gvkey'])
    
    # Merge and perform the inequality join (date between link start and end)
    merged = pd.merge(spells, ccm, on='gvkey', how='inner')
    merged = merged[
        (merged['appointment_date'] >= merged['linkdt']) & 
        (merged['appointment_date'] <= merged['linkenddt'])
    ]
    
    # Deduplicate: Prioritize the primary link ('P') if multiple links exist at the same time
    if 'linkprim' in merged.columns:
        # Create a priority score (P=1, others=0) and sort descending
        merged['link_priority'] = (merged['linkprim'] == 'P').astype(int)
        merged = merged.sort_values('link_priority', ascending=False)
        merged = merged.drop_duplicates(subset=['spell_id'])
        
    spells_linked = merged[['spell_id', 'appointment_date', 'permno', 'gvkey']].copy()
    
    # Ensure PERMNO is integer type and handle potential NaNs
    spells_linked['permno'] = pd.to_numeric(spells_linked['permno'], errors='coerce')
    spells_linked = spells_linked.dropna(subset=['permno'])
    spells_linked['permno'] = spells_linked['permno'].astype(int)

    utils.logger.info(f"Linked {len(spells_linked)} out of {len(spells)} spells to PERMNOs.")
    return spells_linked

def load_market_data(spells_linked: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads CRSP returns and FF5 factors for the required date range."""
    
    # Determine required calendar date range. We use a 1.5x buffer to convert trading days to calendar days.
    # Calculate maximum lookback/lookahead needed
    MAX_LOOKBACK = abs(ESTIMATION_WINDOW_START)
    MAX_LOOKAHEAD = max(w[1] for w in BHAR_WINDOWS.values())

    buffer_back = int(MAX_LOOKBACK * 1.5) + 30
    buffer_fwd = int(MAX_LOOKAHEAD * 1.5) + 30
    
    start_date = spells_linked['appointment_date'].min() - timedelta(days=buffer_back)
    end_date = spells_linked['appointment_date'].max() + timedelta(days=buffer_fwd)
    
    permnos = spells_linked['permno'].unique().tolist()
    db = utils.get_db()
    
    # Fetch Returns (CRSP DSF)
    utils.logger.info(f"Fetching CRSP data from {start_date.date()} to {end_date.date()}.")
    dsf = utils.load_or_fetch(
        config.RAW_CRSP_DSF_PATH, 
        utils.fetch_crsp_dsf,
        db_connection=db,
        permno_list=permnos,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    
    if dsf.empty:
        utils.logger.error("CRSP DSF data could not be loaded.")
        return pd.DataFrame(), pd.DataFrame()

    dsf['date'] = pd.to_datetime(dsf['date'])
    dsf['ret'] = pd.to_numeric(dsf['ret'], errors='coerce')

    # Fetch Factors (Fama-French 5 Daily)
    utils.logger.info("Fetching Fama-French 5-Factor daily data.")
    ff = utils.load_or_fetch(
        config.RAW_FF5_FACTORS_DAILY_PATH,
        utils.fetch_fama_french_5_daily, 
        db_connection=db,
        start_date=start_date.strftime('%Y-%m-%d')
    )

    if ff.empty:
        utils.logger.error("Fama-French 5 Factor data could not be loaded.")
        return pd.DataFrame(), pd.DataFrame()

    ff['date'] = pd.to_datetime(ff['date'])
    
    # Standardize factor column names (assuming standard Ken French library naming)
    ff = ff.rename(columns={
        'Mkt-RF': 'mktrf', 'SMB': 'smb', 'HML': 'hml', 'RMW': 'rmw', 'CMA': 'cma', 'RF': 'rf'
    })
    
    # Data Quality Check: Ensure factors are correctly scaled (decimals, not percentages)
    required_factors = list(set().union(*MODEL_SPECS.values())) + ['rf']
    
    for col in required_factors:
        if col in ff.columns:
            ff[col] = pd.to_numeric(ff[col], errors='coerce')
            # Heuristic check for scaling: if the mean absolute value is large, assume percentages.
            # A typical daily factor movement is much less than 1.
            if ff[col].abs().mean() > 0.5: 
                 utils.logger.warning(f"Factor {col} appears unscaled (likely percentages). Dividing by 100.")
                 ff[col] = ff[col] / 100.0
        # We do not exit if FF5 factors (RMW/CMA) are missing, but models requiring them will be skipped later.
        elif col not in ['rmw', 'cma']:
             utils.logger.error(f"Missing essential factor: {col}. Aborting.")
             return pd.DataFrame(), pd.DataFrame()

    # Optimize: Index by date for fast alignment
    ff = ff.set_index('date').sort_index()
    
    return dsf, ff

# ---------------------------------------------------------------------
# Main Execution Logic
# ---------------------------------------------------------------------

def run_event_study():
    utils.logger.info("Starting Event Study Analysis (Revised with FF5)...")
    
    # 1. Load Spells
    if not config.CEO_SPELLS_PATH.exists():
        utils.logger.error(f"CEO spells file not found at {config.CEO_SPELLS_PATH}.")
        spells = pd.DataFrame()
    else:
        try:
            spells = pd.read_parquet(config.CEO_SPELLS_PATH)
            spells['appointment_date'] = pd.to_datetime(spells['appointment_date'])
        except Exception as e:
            utils.logger.error(f"Failed to load CEO spells: {e}")
            return

    if spells.empty:
         utils.logger.info("No spells loaded. Exiting.")
         return

    # 2. Link to PERMNO
    spells_linked = link_spells_to_permno(spells)
    if spells_linked.empty:
        utils.logger.info("Event study halted due to linking failure.")
        return

    # 3. Load Market Data
    dsf, ff_factors = load_market_data(spells_linked)
    if dsf.empty or ff_factors.empty:
        utils.logger.info("Event study halted due to market data failure.")
        return
    
    # 4. Execute Event Study
    
    # Optimization: Group returns by PERMNO for efficient access during the loop
    dsf_grouped = dsf.groupby('permno')
    
    results_list = []
    total = len(spells_linked)
    utils.logger.info(f"Starting event processing for {total} events...")

    # Use itertuples for efficient iteration
    for i, row in enumerate(spells_linked.itertuples(index=False)):
        if (i + 1) % 100 == 0 or (i + 1) == total:
            utils.logger.info(f"Processing event {i+1}/{total}...")
            
        # Core processing
        ars = process_event(row, dsf_grouped, ff_factors)
        
        if ars:
            ars['spell_id'] = row.spell_id
            results_list.append(ars)
            
    # 5. Finalize and Save Results
    if not results_list:
        utils.logger.warning("No event study results generated.")
        return
        
    results_df = pd.DataFrame(results_list)
    utils.logger.info(f"Successfully analyzed {len(results_df)} events.")
    
    # Winsorize results at the 0.5% level (0.5% and 99.5%) to mitigate outliers
    win_cols = [c for c in results_df.columns if c != 'spell_id']
    utils.logger.info(f"Winsorizing {len(win_cols)} return metrics at 0.5% level...")
    results_df = utils.apply_winsorization(
        results_df, 
        cols=win_cols, 
        group_col=None,
        limits=(0.005, 0.995)
    )
    
    # Merge results back with event details
    final_df = pd.merge(spells_linked, results_df, on='spell_id', how='inner')
    
    utils.logger.info(f"Saving results for {len(final_df)} events.")
    
    # Ensure output directory exists
    try:
        config.EVENT_STUDY_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_parquet(config.EVENT_STUDY_RESULTS_PATH, index=False)
        final_df.to_csv(config.EVENT_STUDY_RESULTS_CSV_PATH, index=False)
    except Exception as e:
        utils.logger.error(f"Failed to save results: {e}")

    utils.logger.info(f"Event Study Complete. Saved to {config.EVENT_STUDY_RESULTS_PATH}")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Note: Running this requires the environment (config, utils, input data) to be set up.
    run_event_study()