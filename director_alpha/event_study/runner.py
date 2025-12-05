"""
Event Study Runner: Main Orchestration.

Provides the main entry point for running event studies on CEO appointments.
"""
import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Dict, Optional
import logging

from .constants import (
    ESTIMATION_WINDOW_START, 
    ESTIMATION_WINDOW_END,
    CAR_WINDOWS,
    BHAR_WINDOWS,
)
from .models import fit_models, calculate_expected_returns
from .alignment import align_event_data


logger = logging.getLogger(__name__)


def process_event(
    row, 
    dsf_grouped, 
    ff_factors: pd.DataFrame
) -> Dict:
    """
    Process a single event: align data, fit models, calculate returns.
    
    Args:
        row: Named tuple with permno, event_date, spell_id
        dsf_grouped: CRSP daily data grouped by permno
        ff_factors: Fama-French factors indexed by date
    
    Returns:
        Dictionary of computed abnormal returns
    """
    permno = row.permno
    appt_date = row.event_date
    results = {}

    # A. Get PERMNO specific returns
    try:
        p_rets = dsf_grouped.get_group(permno).set_index('date')['ret']
    except KeyError:
        return {}
            
    # B. Align Data and find T=0
    event_data, t0_loc = align_event_data(p_rets, ff_factors, appt_date)
    
    if event_data is None:
        return {}
        
    # C. Define Estimation Window using precise Trading Day Indices
    est_start_idx = t0_loc + ESTIMATION_WINDOW_START
    est_end_idx = t0_loc + ESTIMATION_WINDOW_END + 1  # Python slicing exclusive
    
    if est_start_idx < 0:
        # Not enough historical data
        return {}
        
    estimation_slice = event_data.iloc[est_start_idx:est_end_idx]
    
    # D. Fit Models (CAPM, FF3, FF5)
    model_params = fit_models(estimation_slice)
    
    if not model_params:
        return {}

    # E. Calculate Abnormal Returns
    def get_slice(start_offset, end_offset):
        start_idx = t0_loc + start_offset
        end_idx = t0_loc + end_offset + 1 
        if start_idx < 0 or end_idx > len(event_data):
            return None
        return event_data.iloc[start_idx:end_idx]

    for model_type, params in model_params.items():

        # --- CAR Calculation ---
        for name, (w_start, w_end) in CAR_WINDOWS.items():
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
        for name, (w_start, w_end) in BHAR_WINDOWS.items():
            chunk = get_slice(w_start, w_end)
            
            required_len = w_end - w_start + 1
            if chunk is None or len(chunk) < required_len * 0.8:
                continue
                
            try:
                gross_ret = (1 + chunk['ret']).prod()
                er = calculate_expected_returns(chunk, params)
                gross_er = (1 + er).prod()
                
                bhar = gross_ret - gross_er
                results[f'{name}_{model_type}'] = bhar
                
                if 1 + bhar > 0:
                    results[f'{name}_{model_type}_log'] = np.log(1 + bhar)
                else:
                    results[f'{name}_{model_type}_log'] = np.nan
                    
            except ValueError:
                continue

    return results


def run_event_study(
    spells: pd.DataFrame,
    dsf: pd.DataFrame,
    ff_factors: pd.DataFrame,
    winsorize_func=None,
    winsorize_limits=(0.01, 0.99)
) -> Optional[pd.DataFrame]:
    """
    Run event study on CEO spells.
    
    Args:
        spells: DataFrame with 'spell_id', 'event_date', 'permno'
        dsf: CRSP daily stock file with 'permno', 'date', 'ret'
        ff_factors: Fama-French factors indexed by date
        winsorize_func: Optional function for winsorizing results
        winsorize_limits: Winsorization percentiles
    
    Returns:
        DataFrame with spell_id and all computed return metrics
    """
    logger.info("Starting Event Study Analysis...")
    
    # Optimize: Group returns by PERMNO
    dsf_grouped = dsf.groupby('permno')
    
    results_list = []
    total = len(spells)
    logger.info(f"Processing {total} events...")

    # Use itertuples for efficient iteration
    for i, row in enumerate(spells.itertuples(index=False)):
        if (i + 1) % 100 == 0 or (i + 1) == total:
            logger.info(f"Processing event {i+1}/{total}...")
            
        ars = process_event(row, dsf_grouped, ff_factors)
        
        if ars:
            ars['spell_id'] = row.spell_id
            results_list.append(ars)
            
    if not results_list:
        logger.warning("No event study results generated.")
        return None
        
    results_df = pd.DataFrame(results_list)
    logger.info(f"Successfully analyzed {len(results_df)} events.")
    
    # Winsorize results
    if winsorize_func is not None:
        win_cols = [c for c in results_df.columns if c != 'spell_id']
        logger.info(f"Winsorizing {len(win_cols)} return metrics...")
        results_df = winsorize_func(
            results_df, 
            cols=win_cols, 
            limits=winsorize_limits,
            group_col=None
        )
    
    # Merge with spell info
    final_df = pd.merge(spells, results_df, on='spell_id', how='inner')
    
    logger.info(f"Event Study Complete. Analyzed {len(final_df)} events.")
    return final_df
