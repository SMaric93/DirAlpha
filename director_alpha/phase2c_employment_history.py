import pandas as pd
from . import config, io, db, log, history_utils

logger = log.logger

def run_phase2c():
    logger.info("Starting Phase 2c: CEO Employment History (Optimized)...")
    
    # 1. Load Target CEOs
    if not config.CEO_SPELLS_PATH.exists():
        logger.error("CEO Spells file missing. Run Phase 2 first.")
        return
    spells = pd.read_parquet(config.CEO_SPELLS_PATH)
    ceo_execids = spells['execid'].dropna().unique().tolist()
    logger.info(f"Targeting {len(ceo_execids)} unique CEOs.")

    # 2. Load Link Table (WRDS People Link)
    logger.info("Loading Link Data...")
    try:
        link_df = io.load_or_fetch(config.RAW_WRDS_PEOPLE_LINK_PATH, db.fetch_wrds_people_link)
    except Exception as e:
        logger.warning(f"Link table fetch failed: {e}. Skipping phase.")
        return

    if link_df.empty:
        logger.warning("Link table is empty.")
        return

    # Filter Link Table
    link_df['execid'] = pd.to_numeric(link_df['execid'], errors='coerce')
    link_df = link_df[link_df['execid'].isin(ceo_execids)].copy()
    
    # Quality Filter
    if 'score' in link_df.columns:
        link_df['score'] = pd.to_numeric(link_df['score'], errors='coerce')
        link_df = link_df[link_df['score'] <= 3]
        
    logger.info(f"Relevant links found: {len(link_df)} (High Quality).")
    
    # Extract IDs
    relevant_director_ids = link_df['directorid'].dropna().astype(str).unique().tolist()
    relevant_person_ids = link_df['personid'].dropna().unique().tolist() # CIQ Person IDs

    history_frames = []

    # 3. Fetch BoardEx Data (Optimized)
    if relevant_director_ids:
        logger.info(f"Fetching BoardEx history for {len(relevant_director_ids)} directors...")
        
        # A. Fetch Roles
        bdx_roles = db.fetch_boardex_directors(director_ids=relevant_director_ids)
        
        if not bdx_roles.empty:
            # B. Fetch Company Names
            unique_comp_ids = bdx_roles['company_id'].dropna().unique().tolist()
            logger.info(f"Fetching names for {len(unique_comp_ids)} BoardEx companies...")
            bdx_names = db.fetch_boardex_company_names(company_ids=unique_comp_ids)
            
            # Merge Names
            bdx_full = pd.merge(bdx_roles, bdx_names, on='company_id', how='left')
            
            # Normalize
            bdx_norm = history_utils.normalize_history(bdx_full, 'BoardEx')
            
            # Map back to ExecID
            bdx_map = link_df[['directorid', 'execid']].drop_duplicates()
            bdx_map['directorid'] = bdx_map['directorid'].astype(str)
            
            bdx_merged = pd.merge(bdx_norm, bdx_map, left_on='person_id', right_on='directorid', how='inner')
            bdx_merged = bdx_merged.drop(columns=['directorid', 'person_id']) 
            
            history_frames.append(bdx_merged)
            logger.info(f"Processed {len(bdx_merged)} BoardEx records.")
        else:
            logger.warning("No BoardEx roles returned for these IDs.")

    # 4. Fetch CapitalIQ Data (Optimized)
    if relevant_person_ids:
        logger.info(f"Fetching CapitalIQ history for {len(relevant_person_ids)} people...")
        
        ciq_career = db.fetch_ciq_career(person_ids=relevant_person_ids)
        
        if not ciq_career.empty:
            # Normalize
            ciq_norm = history_utils.normalize_history(ciq_career, 'CapitalIQ')
            
            # Map back to ExecID
            ciq_map = link_df[['personid', 'execid']].drop_duplicates()
            ciq_map['personid'] = ciq_map['personid'].astype(str).str.replace(r'\.0$', '', regex=True)
            
            ciq_merged = pd.merge(ciq_norm, ciq_map, left_on='person_id', right_on='personid', how='inner')
            ciq_merged = ciq_merged.drop(columns=['personid', 'person_id'])
            
            history_frames.append(ciq_merged)
            logger.info(f"Processed {len(ciq_merged)} CapitalIQ records.")
        else:
            logger.warning("No CapitalIQ career records returned.")

    # 5. Combine & Consolidate
    if history_frames:
        full_raw = pd.concat(history_frames, ignore_index=True)
        logger.info(f"Consolidating {len(full_raw)} total records...")
        
        final_df = history_utils.consolidate_history(full_raw, id_col='execid')
    else:
        logger.warning("No history found from any source.")
        final_df = pd.DataFrame(columns=['execid', 'company_name', 'role', 'start_date', 'end_date', 'source'])

    # 6. Add CEO Names & Save
    try:
        execucomp = io.load_or_fetch(config.RAW_EXECUCOMP_PATH, db.fetch_execucomp)
        names = execucomp[['execid', 'exec_lname', 'exec_fname']].drop_duplicates('execid')
        names['execid'] = pd.to_numeric(names['execid'], errors='coerce')
        
        final_df = pd.merge(final_df, names, on='execid', how='left')
    except Exception as e:
        logger.warning(f"Could not fetch CEO names: {e}")

    # Sort
    final_cols = ['execid', 'exec_fname', 'exec_lname', 'company_name', 'role', 'start_date', 'end_date', 'source']
    for c in final_cols:
        if c not in final_df.columns:
            final_df[c] = None
            
    final_df = final_df[final_cols].sort_values(['execid', 'start_date']).reset_index(drop=True)
    
    final_df.to_csv(config.CEO_EMPLOYMENT_HISTORY_PATH, index=False)
    logger.info(f"Phase 2c Complete. Saved to {config.CEO_EMPLOYMENT_HISTORY_PATH}")

if __name__ == "__main__":
    run_phase2c()