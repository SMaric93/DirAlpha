"""
Data Alignment and Linking for Event Studies.

Handles GVKEY-PERMNO linking and event day alignment for trading calendars.
"""
import pandas as pd
from typing import Optional, Tuple


def align_event_data(
    returns_series: pd.Series, 
    factors_df: pd.DataFrame, 
    event_date: pd.Timestamp
) -> Tuple[Optional[pd.DataFrame], Optional[int]]:
    """
    Align firm-specific returns with market factors and identify T=0.
    
    Args:
        returns_series: Daily returns for a specific PERMNO (indexed by date)
        factors_df: Daily Fama-French factors (indexed by date)
        event_date: The reported date of the event
    
    Returns:
        Tuple of (aligned_data, t0_index) where:
        - aligned_data: DataFrame with returns and factors
        - t0_index: Integer index location of T=0, or None if not found
    """
    # Join returns and factors (inner join keeps only common dates)
    data = pd.concat([returns_series, factors_df], axis=1, join='inner')
    
    # Ensure essential data ('ret' and 'rf') are present
    data = data.dropna(subset=['ret', 'rf'])
    
    # Sort the index for monotonic ordering (required for get_indexer)
    data = data.sort_index()

    if data.empty:
        return None, None
        
    # Identify T=0. If event date is non-trading day, use next trading day ('bfill')
    try:
        t0_loc = data.index.get_indexer([event_date], method='bfill')[0]
    except KeyError:
        return None, None
        
    # If event_date is beyond available data, get_indexer returns -1
    if t0_loc == -1 or t0_loc >= len(data):
        return None, None
        
    return data, t0_loc


def link_spells_to_permno(
    spells: pd.DataFrame, 
    ccm: pd.DataFrame, 
    normalize_gvkey_func=None
) -> pd.DataFrame:
    """
    Link CEO spells to CRSP PERMNOs using CCM link table.
    
    Handles time-varying links by checking that event_date falls within
    the valid link period.
    
    Args:
        spells: DataFrame with 'spell_id', 'gvkey', 'event_date'
        ccm: CCM link table with 'gvkey', 'permno', 'linkdt', 'linkenddt'
        normalize_gvkey_func: Optional function to normalize GVKEYs
    
    Returns:
        DataFrame with 'spell_id', 'event_date', 'permno', 'gvkey'
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Prepare CCM
    if 'lpermno' in ccm.columns and 'permno' not in ccm.columns:
        ccm = ccm.rename(columns={'lpermno': 'permno'})
        
    ccm['linkdt'] = pd.to_datetime(ccm['linkdt'])
    # Fill missing end dates with today to include ongoing links
    ccm['linkenddt'] = pd.to_datetime(ccm['linkenddt']).fillna(pd.Timestamp.today())
    
    # Normalize GVKEYs if function provided
    if normalize_gvkey_func is not None:
        spells = spells.copy()
        ccm = ccm.copy()
        spells['gvkey'] = normalize_gvkey_func(spells['gvkey'])
        ccm['gvkey'] = normalize_gvkey_func(ccm['gvkey'])
    
    # Merge and perform inequality join (date between link start and end)
    merged = pd.merge(spells, ccm, on='gvkey', how='inner')
    merged = merged[
        (merged['event_date'] >= merged['linkdt']) & 
        (merged['event_date'] <= merged['linkenddt'])
    ]
    
    # Deduplicate: Prioritize primary link ('P') if multiple links exist
    if 'linkprim' in merged.columns:
        merged['link_priority'] = (merged['linkprim'] == 'P').astype(int)
        merged = merged.sort_values('link_priority', ascending=False)
        merged = merged.drop_duplicates(subset=['spell_id'])
        
    spells_linked = merged[['spell_id', 'event_date', 'permno', 'gvkey']].copy()
    
    # Ensure PERMNO is integer
    spells_linked['permno'] = pd.to_numeric(spells_linked['permno'], errors='coerce')
    spells_linked = spells_linked.dropna(subset=['permno'])
    spells_linked['permno'] = spells_linked['permno'].astype(int)

    logger.info(f"Linked {len(spells_linked)} out of {len(spells)} spells to PERMNOs.")
    return spells_linked
