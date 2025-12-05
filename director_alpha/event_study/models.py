"""
OLS Model Fitting for Event Studies.

Provides functions to fit CAPM, FF3, and FF5 models on estimation window data.
"""
import numpy as np
import pandas as pd
from typing import Dict

from .constants import MODEL_SPECS, MIN_OBS_ESTIMATION


def fit_models(data_slice: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Fit CAPM, FF3, and FF5 models using OLS on the estimation window data.
    
    Args:
        data_slice: DataFrame with columns: ret, rf, mktrf, smb, hml, rmw, cma
                   (not all factor columns required - availability determines which models run)
    
    Returns:
        Dictionary mapping model name to fitted parameters.
        Example: {'capm': {'const': 0.001, 'mktrf': 1.05}, ...}
    """
    results = {}

    if len(data_slice) < MIN_OBS_ESTIMATION:
        return {}
        
    # Dependent variable: Excess returns (R - Rf)
    y = (data_slice['ret'] - data_slice['rf']).values
    N = len(data_slice)
    
    for model_name, factors in MODEL_SPECS.items():
        # Prepare the independent variables (X matrix)
        X_list = [np.ones(N)]  # Constant (for alpha)
        
        skip_model = False
        for factor in factors:
            if factor in data_slice.columns:
                # Ensure the factor data isn't missing within the estimation slice
                if data_slice[factor].isnull().any():
                    skip_model = True
                    break
                X_list.append(data_slice[factor].values)
            else:
                # If a required factor is missing, skip the model
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
                model_params[factor] = betas[i + 1]
            
            results[model_name] = model_params
            
        except np.linalg.LinAlgError:
            # Handles numerical instability or severe multicollinearity
            continue
        
    return results


def calculate_expected_returns(chunk: pd.DataFrame, params: Dict[str, float]) -> pd.Series:
    """
    Calculate expected returns based on model parameters.
    
    E[R] = Rf + alpha + sum(beta_i * Factor_i)
    
    Args:
        chunk: DataFrame with factor columns (rf, mktrf, etc.)
        params: Dictionary of model parameters from fit_models
    
    Returns:
        Series of expected returns
    
    Raises:
        ValueError: If a required factor is missing
    """
    er = chunk['rf'] + params['const']
    
    for factor, beta in params.items():
        if factor != 'const':
            if factor in chunk.columns:
                er = er + beta * chunk[factor]
            else:
                raise ValueError(f"Factor {factor} missing in event window calculation.")
            
    return er
