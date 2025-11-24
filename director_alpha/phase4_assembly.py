import pandas as pd
import numpy as np
from . import config, log

logger = log.logger

def calculate_tenure_performance(spells: pd.DataFrame, firm_year: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate average performance (ROA, Q) for T+1 to T+3.
    """
    # Spells has 'gvkey', 'appointment_date'.
    spells = spells.copy()
    firm_year = firm_year.copy()

    spells['appointment_date'] = pd.to_datetime(spells['appointment_date'])
    spells['fyear_appt'] = spells['appointment_date'].dt.year 
    
    # Ensure types match
    spells['gvkey'] = spells['gvkey'].astype(str).str.zfill(6)
    firm_year['gvkey'] = firm_year['gvkey'].astype(str).str.zfill(6)
    
    # Merge spells with firm-year data
    merged = pd.merge(spells[['spell_id', 'gvkey', 'fyear_appt']], 
                      firm_year[['gvkey', 'fyear', 'roa_adj', 'tobins_q_adj']], 
                      on='gvkey', how='inner')
                      
    # Filter for T+1 to T+3
    merged['rel_year'] = merged['fyear'] - merged['fyear_appt']
    merged = merged[(merged['rel_year'] >= 1) & (merged['rel_year'] <= 3)]
    
    # Calculate average per spell
    perf_agg = merged.groupby('spell_id')[['roa_adj', 'tobins_q_adj']].mean().reset_index()
    perf_agg = perf_agg.rename(columns={'roa_adj': 'avg_roa_adj_3yr', 'tobins_q_adj': 'avg_tobins_q_adj_3yr'})
    
    return perf_agg

def get_controls(spells: pd.DataFrame, firm_year: pd.DataFrame) -> pd.DataFrame:
    """
    Get controls at T-1.
    """
    spells = spells.copy()
    firm_year = firm_year.copy()
    
    spells['appointment_date'] = pd.to_datetime(spells['appointment_date'])
    spells['fyear_appt'] = spells['appointment_date'].dt.year 
    spells['target_year'] = spells['fyear_appt'] - 1
    
    # Ensure types match
    spells['gvkey'] = spells['gvkey'].astype(str).str.zfill(6)
    firm_year['gvkey'] = firm_year['gvkey'].astype(str).str.zfill(6)

    merged = pd.merge(spells[['spell_id', 'gvkey', 'target_year']], 
                      firm_year, 
                      left_on=['gvkey', 'target_year'], 
                      right_on=['gvkey', 'fyear'], 
                      how='left')
                      
    # Keep controls
    cols = ['spell_id', 'size', 'leverage', 'rd_intensity', 'capex_intensity', 'firm_age']
    cols = [c for c in cols if c in merged.columns]
    
    return merged[cols]

def run_phase4():
    logger.info("Starting Phase 4: Final Assembly...")
    
    try:
        if not config.CEO_SPELLS_PATH.exists():
             logger.error(f"CEO Spells file missing: {config.CEO_SPELLS_PATH}")
             return
        
        if not config.DIRECTOR_LINKAGE_PATH.exists():
             logger.error(f"Director Linkage file missing: {config.DIRECTOR_LINKAGE_PATH}")
             return

        spells = pd.read_parquet(config.CEO_SPELLS_PATH)
        linkage = pd.read_parquet(config.DIRECTOR_LINKAGE_PATH)
        
        if not config.FIRM_YEAR_PERFORMANCE_PATH.exists():
             logger.error(f"Firm Performance file missing: {config.FIRM_YEAR_PERFORMANCE_PATH}")
             return

        firm_year = pd.read_parquet(config.FIRM_YEAR_PERFORMANCE_PATH)

    except Exception as e:
        logger.error(f"Error loading intermediate files: {e}")
        return

    # 1. Calculate Tenure Performance
    perf = calculate_tenure_performance(spells, firm_year)
    
    # 2. Get Controls (T-1)
    controls = get_controls(spells, firm_year)

    # 3. Load Event Study Results (Phase 3b)
    if config.EVENT_STUDY_RESULTS_PATH.exists():
        try:
            event_study = pd.read_parquet(config.EVENT_STUDY_RESULTS_PATH)
            # Keep only spell_id and return metrics
            es_cols = [c for c in event_study.columns if c not in ['gvkey', 'appointment_date', 'permno']]
            event_study = event_study[es_cols]
        except Exception as e:
             logger.warning(f"Failed to load event study results: {e}")
             event_study = pd.DataFrame(columns=['spell_id'])
    else:
        logger.warning("Event study results not found. Proceeding without returns.")
        event_study = pd.DataFrame(columns=['spell_id'])
    
    # 4. Merge everything onto Linkage
    logger.info("Merging datasets...")
    
    # First merge spell attributes to spells
    spells_full = pd.merge(spells, perf, on='spell_id', how='left')
    spells_full = pd.merge(spells_full, controls, on='spell_id', how='left')
    spells_full = pd.merge(spells_full, event_study, on='spell_id', how='left')
    
    # Now merge to Linkage
    # Linkage: spell_id, directorid, company_id...
    # Ensure keys match
    spells_full['gvkey'] = spells_full['gvkey'].astype(str).str.zfill(6)
    linkage['gvkey'] = linkage['gvkey'].astype(str).str.zfill(6)
    
    analysis = pd.merge(linkage, spells_full, on=['spell_id', 'gvkey', 'appointment_date'], how='inner')
    
    # Add Cohort Year
    analysis['cohort_year'] = pd.to_datetime(analysis['appointment_date']).dt.year
    
    analysis.to_parquet(config.ANALYSIS_HDFE_PATH)
    analysis.to_csv(config.ANALYSIS_HDFE_CSV_PATH, index=False)
    
    logger.info(f"Phase 4 Complete. Assembled {len(analysis)} observations for analysis.")
    logger.info(f"Saved to {config.ANALYSIS_HDFE_PATH}")
    
    return analysis

if __name__ == "__main__":
    run_phase4()