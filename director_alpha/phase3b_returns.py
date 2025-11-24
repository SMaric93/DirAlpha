"""
Event Study Analysis: CEO Appointments (Revised and Enhanced)

Goal:
- Calculate Cumulative Abnormal Returns (CAR) and Buy-and-Hold Abnormal Returns (BHAR).
- Models: CAPM, Fama-French 3-Factor (FF3), Fama-French 5-Factor (FF5).
- Methodology: Standard event study with precise trading day indexing.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Dict, Optional, Tuple, List

from . import config, db, io, transform, log

logger = log.logger

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

    if len(data_slice) < config.MIN_OBS_ESTIMATION:
        return {}
        
    # Dependent variable: Excess returns (R - Rf)
    y = (data_slice['ret'] - data_slice['rf']).values
    N = len(data_slice)
    
    for model_name, factors in config.MODEL_SPECS.items():
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
            logger.warning(f"OLS failed for model {model_name} due to LinAlgError.")
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
    event_data, t0_loc = align_event_data(p_rets, ff_factors, appt_date)
    
    if event_data is None:
        return {}
        
    # C. Define Estimation Window using precise Trading Day Indices
    est_start_idx = t0_loc + config.ESTIMATION_WINDOW_START
    est_end_idx = t0_loc + config.ESTIMATION_WINDOW_END + 1 
    
    if est_start_idx < 0:
        return {}
        
    estimation_slice = event_data.iloc[est_start_idx:est_end_idx]
    
    # D. Fit Models (CAPM, FF3, FF5)
    model_params = fit_models(estimation_slice)
    
    if not model_params:
        return {}

    # E. Calculate Abnormal Returns (CARs and BHARs)

    def get_slice(start_offset, end_offset):
        start_idx = t0_loc + start_offset
        end_idx = t0_loc + end_offset + 1 
        if start_idx < 0 or end_idx > len(event_data):
            return None
        return event_data.iloc[start_idx:end_idx]

    for model_type, params in model_params.items():

        # --- CAR Calculation ---
        for name, (w_start, w_end) in config.CAR_WINDOWS.items():
            chunk = get_slice(w_start, w_end)
            if chunk is None:
                continue
            
            required_len = w_end - w_start + 1
            if len(chunk) != required_len:
                continue

            try:
                er = calculate_expected_returns(chunk, params)
                ar = chunk['ret'] - er
                results[f'{name}_{model_type}'] = ar.sum()
            except ValueError:
                continue

        # --- BHAR Calculation ---
        for name, (w_start, w_end) in config.BHAR_WINDOWS.items():
            chunk = get_slice(w_start, w_end)
            
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
    logger.info("Starting CCM linking process...")

    ccm = io.load_or_fetch(config.RAW_CCM_PATH, db.fetch_ccm_link)
    if ccm.empty:
        logger.error("CCM link table is empty or failed to load.")
        return pd.DataFrame()

    if 'lpermno' in ccm.columns and 'permno' not in ccm.columns:
        ccm = ccm.rename(columns={'lpermno': 'permno'})
        
    ccm['linkdt'] = pd.to_datetime(ccm['linkdt'])
    ccm['linkenddt'] = pd.to_datetime(ccm['linkenddt']).fillna(pd.Timestamp.today())
    
    spells['gvkey'] = transform.normalize_gvkey(spells['gvkey'])
    ccm['gvkey'] = transform.normalize_gvkey(ccm['gvkey'])
    
    merged = pd.merge(spells, ccm, on='gvkey', how='inner')
    merged = merged[
        (merged['appointment_date'] >= merged['linkdt']) & 
        (merged['appointment_date'] <= merged['linkenddt'])
    ]
    
    if 'linkprim' in merged.columns:
        merged['link_priority'] = (merged['linkprim'] == 'P').astype(int)
        merged = merged.sort_values('link_priority', ascending=False)
        merged = merged.drop_duplicates(subset=['spell_id'])
        
    spells_linked = merged[['spell_id', 'appointment_date', 'permno', 'gvkey']].copy()
    
    spells_linked['permno'] = pd.to_numeric(spells_linked['permno'], errors='coerce')
    spells_linked = spells_linked.dropna(subset=['permno'])
    spells_linked['permno'] = spells_linked['permno'].astype(int)

    logger.info(f"Linked {len(spells_linked)} out of {len(spells)} spells to PERMNOs.")
    return spells_linked

def load_market_data(spells_linked: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads CRSP returns and FF5 factors for the required date range."""
    
    MAX_LOOKBACK = abs(config.ESTIMATION_WINDOW_START)
    MAX_LOOKAHEAD = max(w[1] for w in config.BHAR_WINDOWS.values())

    buffer_back = int(MAX_LOOKBACK * 1.5) + 30
    buffer_fwd = int(MAX_LOOKAHEAD * 1.5) + 30
    
    start_date = spells_linked['appointment_date'].min() - timedelta(days=buffer_back)
    end_date = spells_linked['appointment_date'].max() + timedelta(days=buffer_fwd)
    
    permnos = spells_linked['permno'].unique().tolist()
    
    logger.info(f"Fetching CRSP data from {start_date.date()} to {end_date.date()}.")
    dsf = io.load_or_fetch(
        config.RAW_CRSP_DSF_PATH, 
        db.fetch_crsp_dsf,
        permno_list=permnos,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    
    if dsf.empty:
        logger.error("CRSP DSF data could not be loaded.")
        return pd.DataFrame(), pd.DataFrame()

    dsf['date'] = pd.to_datetime(dsf['date'])
    dsf['ret'] = pd.to_numeric(dsf['ret'], errors='coerce')

    logger.info("Fetching Fama-French 5-Factor daily data.")
    ff = io.load_or_fetch(
        config.RAW_FF5_FACTORS_DAILY_PATH,
        db.fetch_fama_french_5_daily, 
        start_date=start_date.strftime('%Y-%m-%d')
    )

    if ff.empty:
        logger.error("Fama-French 5 Factor data could not be loaded.")
        return pd.DataFrame(), pd.DataFrame()

    ff['date'] = pd.to_datetime(ff['date'])
    
    ff = ff.rename(columns={
        'Mkt-RF': 'mktrf', 'SMB': 'smb', 'HML': 'hml', 'RMW': 'rmw', 'CMA': 'cma', 'RF': 'rf'
    })
    
    required_factors = list(set().union(*config.MODEL_SPECS.values())) + ['rf']
    
    for col in required_factors:
        if col in ff.columns:
            ff[col] = pd.to_numeric(ff[col], errors='coerce')
            if ff[col].abs().mean() > 0.5: 
                 logger.warning(f"Factor {col} appears unscaled (likely percentages). Dividing by 100.")
                 ff[col] = ff[col] / 100.0
        elif col not in ['rmw', 'cma']:
             logger.error(f"Missing essential factor: {col}. Aborting.")
             return pd.DataFrame(), pd.DataFrame()

    ff = ff.set_index('date').sort_index()
    
    return dsf, ff

# ---------------------------------------------------------------------
# Main Execution Logic
# ---------------------------------------------------------------------

def run_event_study():
    logger.info("Starting Event Study Analysis (Revised with FF5)...")
    
    # 1. Load Spells
    if not config.CEO_SPELLS_PATH.exists():
        logger.error(f"CEO spells file not found at {config.CEO_SPELLS_PATH}.")
        spells = pd.DataFrame()
    else:
        try:
            spells = pd.read_parquet(config.CEO_SPELLS_PATH)
            spells['appointment_date'] = pd.to_datetime(spells['appointment_date'])
        except Exception as e:
            logger.error(f"Failed to load CEO spells: {e}")
            return

    if spells.empty:
         logger.info("No spells loaded. Exiting.")
         return

    # 2. Link to PERMNO
    spells_linked = link_spells_to_permno(spells)
    if spells_linked.empty:
        logger.info("Event study halted due to linking failure.")
        return

    # 3. Load Market Data
    dsf, ff_factors = load_market_data(spells_linked)
    if dsf.empty or ff_factors.empty:
        logger.info("Event study halted due to market data failure.")
        return
    
    # 4. Execute Event Study
    dsf_grouped = dsf.groupby('permno')
    
    results_list = []
    total = len(spells_linked)
    logger.info(f"Starting event processing for {total} events...")

    for i, row in enumerate(spells_linked.itertuples(index=False)):
        if (i + 1) % 100 == 0 or (i + 1) == total:
            logger.info(f"Processing event {i+1}/{total}...")
            
        ars = process_event(row, dsf_grouped, ff_factors)
        
        if ars:
            ars['spell_id'] = row.spell_id
            results_list.append(ars)
            
    # 5. Finalize and Save Results
    if not results_list:
        logger.warning("No event study results generated.")
        return
        
    results_df = pd.DataFrame(results_list)
    logger.info(f"Successfully analyzed {len(results_df)} events.")
    
    win_cols = [c for c in results_df.columns if c != 'spell_id']
    logger.info(f"Winsorizing {len(win_cols)} return metrics at 0.5% level...")
    results_df = transform.apply_winsorization(
        results_df, 
        cols=win_cols, 
        group_col=None,
        limits=(0.005, 0.995)
    )
    
    final_df = pd.merge(spells_linked, results_df, on='spell_id', how='inner')
    
    logger.info(f"Saving results for {len(final_df)} events.")
    
    try:
        config.EVENT_STUDY_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_parquet(config.EVENT_STUDY_RESULTS_PATH, index=False)
        final_df.to_csv(config.EVENT_STUDY_RESULTS_CSV_PATH, index=False)
    except Exception as e:
        logger.error(f"Failed to save results: {e}")

    logger.info(f"Event Study Complete. Saved to {config.EVENT_STUDY_RESULTS_PATH}")

if __name__ == "__main__":
    run_event_study()
