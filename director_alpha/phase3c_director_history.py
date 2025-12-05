import pandas as pd
from . import config, io, db, log, history_utils

logger = log.logger

def run_phase3c():
    logger.info("Starting Phase 3c: Director Employment History...")
    
    # 1. Load Director Linkage (from Phase 3)
    # This file contains firm-year-director links
    if not config.DIRECTOR_LINKAGE_PATH.exists():
        logger.error(f"Director Linkage file missing: {config.DIRECTOR_LINKAGE_PATH}. Run Phase 3 first.")
        return
        
    linkage = pd.read_parquet(config.DIRECTOR_LINKAGE_PATH)
    
    # Check for directorid column
    if 'directorid' not in linkage.columns:
        logger.error("directorid column missing in linkage file.")
        return

    # Extract unique Director IDs (BoardEx IDs)
    target_director_ids = linkage['directorid'].dropna().unique().tolist()
    # Convert to str for consistency
    target_director_ids = [str(x) for x in target_director_ids]
    
    logger.info(f"Targeting {len(target_director_ids)} unique directors from the baseline sample.")

    history_frames = []

    # -------------------------------------------------------------------------
    # 2. BoardEx History
    # -------------------------------------------------------------------------
    if target_director_ids:
        logger.info(f"Fetching BoardEx history for {len(target_director_ids)} directors...")
        
        # A. Fetch Roles
        bdx_roles = db.fetch_boardex_directors(director_ids=target_director_ids)
        
        if not bdx_roles.empty:
            # B. Fetch Company Names
            unique_comp_ids = bdx_roles['company_id'].dropna().unique().tolist()
            logger.info(f"Fetching names for {len(unique_comp_ids)} BoardEx companies...")
            bdx_names = db.fetch_boardex_company_names(company_ids=unique_comp_ids)
            
            # Merge Names
            bdx_full = pd.merge(bdx_roles, bdx_names, on='company_id', how='left')
            
            # Normalize
            bdx_norm = history_utils.normalize_history(bdx_full, 'BoardEx')
            
            # For Directors, the ID is the person_id (directorid).
            # We rename person_id to directorid to be consistent with our key
            bdx_norm = bdx_norm.rename(columns={'person_id': 'directorid'})
            
            history_frames.append(bdx_norm)
            logger.info(f"Processed {len(bdx_norm)} BoardEx records.")
        else:
            logger.warning("No BoardEx roles returned for these directors.")

    # -------------------------------------------------------------------------
    # 3. Link to CapitalIQ (via WRDS People Link)
    # -------------------------------------------------------------------------
    logger.info("Loading Link Data to find CapitalIQ matches...")
    
    try:
        # We fetch the link table. 
        # OPTIMIZATION: If we could filter server-side by directorid, that would be better.
        # But fetch_wrds_people_link selects all.
        # Given we refactored db.py for history, let's use the full link table load for now 
        # (assuming it fits in memory, approx 500MB-1GB).
        link_df = io.load_or_fetch(config.RAW_WRDS_PEOPLE_LINK_PATH, db.fetch_wrds_people_link)
    except Exception as e:
        logger.warning(f"Link table fetch failed: {e}. Skipping CapitalIQ for directors.")
        link_df = pd.DataFrame()

    target_person_ids = []
    
    if not link_df.empty:
        # Filter for our target directors
        # Ensure types match
        link_df['directorid'] = link_df['directorid'].astype(str).str.replace(r'\.0$', '', regex=True)
        
        # Filter
        relevant_links = link_df[link_df['directorid'].isin(target_director_ids)].copy()
        
        # Quality Filter (Score <= 3)
        if 'score' in relevant_links.columns:
            relevant_links['score'] = pd.to_numeric(relevant_links['score'], errors='coerce')
            relevant_links = relevant_links[relevant_links['score'] <= 3]
            
        target_person_ids = relevant_links['personid'].dropna().unique().tolist()
        logger.info(f"Found {len(target_person_ids)} linked CapitalIQ Person IDs.")
    else:
        logger.warning("Link table empty or missing.")
        relevant_links = pd.DataFrame(columns=['directorid', 'personid'])

    # -------------------------------------------------------------------------
    # 4. CapitalIQ History
    # -------------------------------------------------------------------------
    if target_person_ids:
        logger.info(f"Fetching CapitalIQ history for {len(target_person_ids)} people...")
        
        ciq_career = db.fetch_ciq_career(person_ids=target_person_ids)
        
        if not ciq_career.empty:
            # Normalize
            ciq_norm = history_utils.normalize_history(ciq_career, 'CapitalIQ')
            
            # Map back to directorid
            # We need: person_id (CIQ personid) -> directorid
            ciq_map = relevant_links[['personid', 'directorid']].drop_duplicates()
            ciq_map['personid'] = ciq_map['personid'].astype(str).str.replace(r'\.0$', '', regex=True)
            
            # Merge
            ciq_merged = pd.merge(ciq_norm, ciq_map, left_on='person_id', right_on='personid', how='inner')
            ciq_merged = ciq_merged.drop(columns=['personid', 'person_id']) # Drop CIQ ID
            
            # Rename to directorid
            # ciq_merged now has 'directorid' column from the map
            
            history_frames.append(ciq_merged)
            logger.info(f"Processed {len(ciq_merged)} CapitalIQ records.")
        else:
            logger.warning("No CapitalIQ career records returned.")

    # -------------------------------------------------------------------------
    # 5. Combine & Consolidate
    # -------------------------------------------------------------------------
    if history_frames:
        full_raw = pd.concat(history_frames, ignore_index=True)
        logger.info(f"Consolidating {len(full_raw)} total records...")
        
        # Use directorid as the grouping key
        final_df = history_utils.consolidate_history(full_raw, id_col='directorid')
    else:
        logger.warning("No history found from any source.")
        final_df = pd.DataFrame(columns=['directorid', 'company_name', 'role', 'start_date', 'end_date', 'source'])

    # 6. Add Director Names
    # We can get names from BoardEx Directors (Phase 3 raw file) or People Link.
    # People Link has 'directorname'.
    if not link_df.empty:
        names = link_df[['directorid', 'directorname']].drop_duplicates('directorid')
        # We might have directors in our list that aren't in Link Table (if they are BoardEx only and have no link).
        # We can also fetch names from BoardEx directly.
        # Ideally, we use what we have.
        final_df = pd.merge(final_df, names, on='directorid', how='left')
    
    # Sort
    final_cols = ['directorid', 'directorname', 'company_name', 'role', 'start_date', 'end_date', 'source']
    for c in final_cols:
        if c not in final_df.columns:
            final_df[c] = None
            
    final_df = final_df[final_cols].sort_values(['directorid', 'start_date']).reset_index(drop=True)
    
    final_df.to_csv(config.DIRECTOR_EMPLOYMENT_HISTORY_PATH, index=False)
    logger.info(f"Phase 3c Complete. Saved to {config.DIRECTOR_EMPLOYMENT_HISTORY_PATH}")

if __name__ == "__main__":
    run_phase3c()
