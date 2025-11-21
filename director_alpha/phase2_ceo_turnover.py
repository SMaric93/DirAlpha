import pandas as pd
import numpy as np
from . import config, utils

def identify_ceos(execucomp):
    """
    Identify the primary CEO for each firm-year.
    """
    df = execucomp.copy()
    # Normalize CEO flags
    # PCEO is often 'CEO' or similar string, CEOANN is the legacy.
    if 'pceo' not in df.columns: df['pceo'] = None
    if 'ceoann' not in df.columns: df['ceoann'] = None
    
    is_ceo = (df['pceo'] == 'CEO') | (df['ceoann'] == 'CEO')
    ceos = df[is_ceo].copy()
    
    # Handling Co-CEOs: Tie-breaking rule
    def resolve_ties(group):
        if len(group) == 1:
            return group
        
        # Prioritize Chairman
        if 'title' in group.columns:
            is_chair = group['title'].str.contains('Chair', case=False, na=False)
            if is_chair.sum() == 1:
                return group[is_chair]
        
        # Prioritize highest TDC1
        if 'tdc1' in group.columns:
            return group.sort_values('tdc1', ascending=False).head(1)
            
        return group.head(1)

    ceos = ceos.groupby(['gvkey', 'year']).apply(resolve_ties).reset_index(drop=True)
    return ceos

def identify_turnover(ceos):
    """
    Identify turnover events.
    """
    ceos = ceos.sort_values(['gvkey', 'year'])
    
    ceos['prev_execid'] = ceos.groupby('gvkey')['execid'].shift(1)
    ceos['prev_year'] = ceos.groupby('gvkey')['year'].shift(1)
    
    turnover = (ceos['execid'] != ceos['prev_execid']) & \
               (ceos['prev_execid'].notna()) & \
               (ceos['year'] == ceos['prev_year'] + 1)
               
    ceos['is_turnover'] = turnover
    return ceos

def determine_dates(ceos):
    """
    Determine appointment dates.
    """
    # Use BECAMECEO
    ceos['appointment_date'] = pd.to_datetime(ceos['becameceo'], errors='coerce')
    ceos['leftofc'] = pd.to_datetime(ceos['leftofc'], errors='coerce')
    ceos['joined_co'] = pd.to_datetime(ceos['joined_co'], errors='coerce')
    return ceos

def filter_interim(spells):
    """
    Exclude interim CEOs and short spells.
    """
    if 'title' in spells.columns:
        is_interim = spells['title'].str.contains('Interim', case=False, na=False)
        spells = spells[~is_interim]
    
    # Calculate spell end date
    spells = spells.sort_values(['gvkey', 'appointment_date'])
    spells['next_appointment'] = spells.groupby('gvkey')['appointment_date'].shift(-1)
    
    spells['spell_end'] = spells['leftofc']
    spells['spell_end'] = spells['spell_end'].fillna(spells['next_appointment'])
    
    duration = (spells['spell_end'] - spells['appointment_date']).dt.days
    
    # Filter < 365 days, but keep if NaT (censored)
    short_spell = (duration < 365) & (duration.notna())
    
    spells = spells[~short_spell]
    return spells

def classify_hires(spells):
    """
    Classify Internal vs External.
    """
    days_since_join = (spells['appointment_date'] - spells['joined_co']).dt.days
    
    # External if joined within 365 days of becoming CEO
    spells['is_external'] = (days_since_join <= 365)
    return spells

def run_phase2():
    utils.logger.info("Starting Phase 2: CEO Tenures and Turnover...")
    
    execucomp = utils.load_or_fetch(
        config.RAW_EXECUCOMP_PATH,
        utils.fetch_execucomp,
        start_year=2000
    )

    if execucomp.empty:
        utils.logger.error("ExecuComp data missing. Aborting Phase 2.")
        return

    # Normalize GVKEY
    execucomp['gvkey'] = utils.normalize_gvkey(execucomp['gvkey'])

    # 1. Identify CEOs
    ceos = identify_ceos(execucomp)
    
    # 2. Identify Turnover
    ceos = identify_turnover(ceos)
    
    # Parse dates
    ceos = determine_dates(ceos)
    
    # Drop duplicates to get unique spells (start of tenure)
    spells = ceos.sort_values('year').groupby(['gvkey', 'execid', 'appointment_date']).first().reset_index()
    
    # 3. Filter Interim & Short Spells
    spells = filter_interim(spells)
    
    # 4. Classify Hires
    spells = classify_hires(spells)
    
    # Generate Spell ID
    spells['spell_id'] = spells.index + 1
    
    # Select columns
    cols = ['spell_id', 'gvkey', 'ticker', 'execid', 'appointment_date', 'spell_end', 'is_external', 'age', 'gender']
    cols = [c for c in cols if c in spells.columns]
    
    final_spells = spells[cols].copy()
    
    utils.logger.info(f"Phase 2 Complete. Identified {len(final_spells)} CEO spells.")
    
    final_spells.to_parquet(config.CEO_SPELLS_PATH)
    return final_spells

if __name__ == "__main__":
    run_phase2()