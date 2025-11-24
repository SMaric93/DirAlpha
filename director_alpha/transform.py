import pandas as pd
import numpy as np
from typing import List, Optional
from . import log

logger = log.logger

def clean_id(series: pd.Series) -> pd.Series:
    """
    Normalize ID columns (e.g. boardid, directorid) by ensuring string type
    and removing trailing '.0' which often appears from float conversion.
    """
    return series.astype(str).str.replace(r"\.0$", "", regex=True)

def normalize_gvkey(series: pd.Series) -> pd.Series:
    """Ensure GVKEY is a 6-digit zero-padded string."""
    return series.astype(str).str.zfill(6)

def normalize_ticker(series: pd.Series) -> pd.Series:
    """Ensure ticker is uppercase stripped string."""
    return series.astype(str).str.upper().str.strip()

def industry_adjust(df: pd.DataFrame, cols: List[str], group_cols: List[str] = ['fyear', 'sic2']) -> pd.DataFrame:
    """
    Subtract group-level median from specified columns.
    """
    df = df.copy()
    # Ensure group columns exist
    for c in group_cols:
        if c not in df.columns:
            if c == 'sic2' and 'sich' in df.columns:
                 df['sic2'] = df['sich'].fillna(0).astype(int) // 100
            else:
                logger.warning(f"Grouping column {c} missing for industry adjustment.")
                return df

    for col in cols:
        if col in df.columns:
            median = df.groupby(group_cols)[col].transform('median')
            df[f'{col}_adj'] = df[col] - median
    return df

def winsorize_series(x: pd.Series, limits: tuple = (0.01, 0.99)) -> pd.Series:
    """
    Clip series between quantiles.
    """
    lower = x.quantile(limits[0])
    upper = x.quantile(limits[1])
    return x.clip(lower=lower, upper=upper)

def apply_winsorization(df: pd.DataFrame, cols: List[str], group_col: Optional[str] = 'fyear', limits: tuple = (0.01, 0.99)) -> pd.DataFrame:
    """
    Apply winsorization to specified columns, optionally grouped by year.
    """
    df = df.copy()
    valid_cols = [c for c in cols if c in df.columns]
    
    if group_col and group_col in df.columns:
        for col in valid_cols:
            df[col] = df.groupby(group_col)[col].transform(lambda x: winsorize_series(x, limits=limits))
    else:
        for col in valid_cols:
            # pyrefly: ignore [bad-argument-type]
            df[col] = winsorize_series(df[col], limits=limits)
            
    return df
