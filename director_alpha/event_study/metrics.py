"""
Abnormal Return Metrics: CAR and BHAR.

Provides functions to calculate Cumulative Abnormal Returns and 
Buy-and-Hold Abnormal Returns.
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional

from .models import calculate_expected_returns


def calculate_car(
    chunk: pd.DataFrame, 
    params: Dict[str, float],
    window_name: str,
    model_type: str
) -> Optional[float]:
    """
    Calculate Cumulative Abnormal Returns (CAR) for a given window.
    
    CAR = Sum(AR) = Sum(R - E[R])
    
    Args:
        chunk: DataFrame slice for the event window
        params: Model parameters from fit_models
        window_name: Name of the window (for result key)
        model_type: Name of the model (capm, ff3, ff5)
    
    Returns:
        CAR value, or None if calculation fails
    """
    try:
        er = calculate_expected_returns(chunk, params)
        ar = chunk['ret'] - er
        return ar.sum()
    except ValueError:
        return None


def calculate_bhar(
    chunk: pd.DataFrame,
    params: Dict[str, float],
    window_name: str,
    model_type: str
) -> Dict[str, Optional[float]]:
    """
    Calculate Buy-and-Hold Abnormal Returns (BHAR) for a given window.
    
    BHAR = Prod(1 + R) - Prod(1 + E[R])
    
    Also calculates log-transformed BHAR: log(1 + BHAR)
    
    Args:
        chunk: DataFrame slice for the event window
        params: Model parameters from fit_models
        window_name: Name of the window (for result key)
        model_type: Name of the model (capm, ff3, ff5)
    
    Returns:
        Dictionary with BHAR and log-BHAR values
    """
    results = {}
    
    try:
        # Compounded actual return
        gross_ret = (1 + chunk['ret']).prod()
        
        # Compounded expected return (benchmark)
        er = calculate_expected_returns(chunk, params)
        gross_er = (1 + er).prod()
        
        bhar = gross_ret - gross_er
        results[f'{window_name}_{model_type}'] = bhar
        
        # Log-transformed BHAR: log(1 + BHAR)
        # This transformation helps normalize the distribution of long-horizon returns
        if 1 + bhar > 0:
            results[f'{window_name}_{model_type}_log'] = np.log(1 + bhar)
        else:
            # For extreme negative returns (BHAR < -100%), store NaN
            results[f'{window_name}_{model_type}_log'] = np.nan
            
    except ValueError:
        pass
    
    return results
