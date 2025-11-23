import pandas as pd
import numpy as np
from . import config

def calculate_tenure_performance(spells, firm_year):
    """
    Calculate average performance (ROA, Q) for T+1 to T+3.
    """
    # Spells has 'gvkey', 'appointment_date'.
    # We need fiscal year of appointment.
    spells['appointment_date'] = pd.to_datetime(spells['appointment_date'])
    spells['fyear_appt'] = spells['appointment_date'].dt.year # Approximation, should use fiscal year map if possible
    
    # We need to link spells to firm_year to get fiscal year more accurately?
    # Or just use calendar year of appointment as base T.
    # Prompt says: "Use the first three full fiscal years following the appointment (T+1 to T+3)."
    
    # Let's assume fyear_appt is the fiscal year containing the appointment.
    # Then we want fyear_appt + 1, + 2, + 3.
    
    # Create a long format of target years
    spells_perf = []
    
    # Optimize: Instead of iterating, merge?
    # Create a list of (gvkey, fyear) needed.
    
    # We can do a range merge or just merge on gvkey and filter.
    # Merging on gvkey is safe if universe isn't huge.
    
    # Ensure types match
    spells['gvkey'] = spells['gvkey'].astype(str)
    firm_year['gvkey'] = firm_year['gvkey'].astype(str)
    
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

def get_controls(spells, firm_year):
    """
    Get controls at T-1.
    """
    # Similar logic, merge on T-1
    spells['target_year'] = spells['fyear_appt'] - 1
    
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
    print("Starting Phase 4: Final Assembly...")
    
    try:
        spells = pd.read_parquet(config.CEO_SPELLS_PATH)
        linkage = pd.read_parquet(config.DIRECTOR_LINKAGE_PATH)
        # Load Phase 1 output (Firm Performance)
        # Note: We saved it as firm_year_performance.parquet in Phase 1
        firm_year = pd.read_parquet(config.INTERMEDIATE_DIR / "firm_year_performance.parquet")
    except FileNotFoundError as e:
        print(f"Missing intermediate files: {e}")
        return

    # 1. Calculate Tenure Performance
    perf = calculate_tenure_performance(spells, firm_year)
    
    # 2. Get Controls (T-1)
    controls = get_controls(spells, firm_year)

    # 3. Load Event Study Results (Phase 3b)
    try:
        event_study = pd.read_parquet(config.EVENT_STUDY_RESULTS_PATH)
        # Keep only spell_id and return metrics
        es_cols = [c for c in event_study.columns if c not in ['gvkey', 'appointment_date', 'permno']]
        event_study = event_study[es_cols]
    except FileNotFoundError:
        print("Event study results not found (Phase 3b skipped?). Proceeding without returns.")
        # pyrefly: ignore [bad-argument-type]
        event_study = pd.DataFrame(columns=['spell_id'])
    
    # 4. Merge everything onto Linkage
    # Linkage is Director-Spell level.
    # Merge Spell info (Performance, Controls, CEO Characteristics, Returns) onto Linkage.
    
    # First merge spell attributes to spells
    spells_full = pd.merge(spells, perf, on='spell_id', how='left')
    spells_full = pd.merge(spells_full, controls, on='spell_id', how='left')
    spells_full = pd.merge(spells_full, event_study, on='spell_id', how='left')
    
    # Now merge to Linkage
    analysis = pd.merge(linkage, spells_full, on=['spell_id', 'gvkey', 'appointment_date'], how='inner')
    
    # Add Cohort Year
    analysis['cohort_year'] = pd.to_datetime(analysis['appointment_date']).dt.year
    
    analysis.to_parquet(config.ANALYSIS_HDFE_PATH)
    
    # Also save as CSV
    csv_path = config.ANALYSIS_HDFE_CSV_PATH
    analysis.to_csv(csv_path, index=False)
    
    print(f"Phase 4 Complete. Assembled {len(analysis)} observations for analysis.")
    print(f"Saved to {config.ANALYSIS_HDFE_PATH} and {csv_path}")
    
    return analysis

if __name__ == "__main__":
    run_phase4()
