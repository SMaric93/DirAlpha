import pandas as pd
import numpy as np
from difflib import SequenceMatcher
from director_alpha import config, db, io
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def similarity(a, b):
    if pd.isna(a) or pd.isna(b):
        return 0.0
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()

def run_verification():
    logger.info("Starting End-to-End Name Verification...")
    
    # 1. Load Enriched Spells (The ones used in Event Study)
    if not config.CEO_SPELLS_BOARDEX_PATH.exists():
        logger.error("ceo_spells_boardex.parquet not found. Run phase3_directors.py first.")
        return
        
    spells = pd.read_parquet(config.CEO_SPELLS_BOARDEX_PATH)
    logger.info(f"Loaded {len(spells):,} spells.")
    
    # Filter for those with BoardEx announcement dates (the ones we care about)
    spells_with_ann = spells[spells['announcement_date'].notna()].copy()
    logger.info(f"Spells with BoardEx announcement dates: {len(spells_with_ann):,}")
    
    if spells_with_ann.empty:
        logger.warning("No spells with announcement dates found.")
        return

    # 2. Get Execucomp Names
    # Spells should have exec_name or we fetch from execucomp
    if 'exec_fullname' not in spells_with_ann.columns:
        logger.info("Fetching Execucomp names...")
        execucomp = pd.read_parquet(config.RAW_EXECUCOMP_PATH)
        print(f"Execucomp columns: {execucomp.columns.tolist()}")
        
        # Construct full name if needed
        if 'exec_fullname' not in execucomp.columns:
             execucomp['exec_fullname'] = execucomp['exec_fname'].fillna('') + ' ' + execucomp['exec_lname'].fillna('')
             execucomp['exec_fullname'] = execucomp['exec_fullname'].str.strip()
             # Replace empty strings with NaN
             execucomp.loc[execucomp['exec_fullname'] == '', 'exec_fullname'] = np.nan
             
        # Execucomp might have multiple rows per execid, just get one name
        # Drop rows with missing names first to prioritize valid ones
        exec_names = execucomp[['execid', 'exec_fullname']].dropna().drop_duplicates('execid')
        
        print(f"Execucomp unique IDs with names: {len(exec_names):,}")
        print(f"Sample Exec Names: {exec_names.head().to_dict('records')}")
        
        # Ensure execid types match
        spells_with_ann['execid'] = spells_with_ann['execid'].astype(str)
        exec_names['execid'] = exec_names['execid'].astype(str)
        
        # Check coverage
        missing_ids = set(spells_with_ann['execid']) - set(exec_names['execid'])
        print(f"ExecIDs in spells but missing names: {len(missing_ids):,}")
        if missing_ids:
            print(f"Sample missing IDs: {list(missing_ids)[:5]}")
        
        spells_with_ann = pd.merge(spells_with_ann, exec_names, on='execid', how='left')
    
    # 3. Get BoardEx Names
    # We need to link director_id to director_name
    logger.info("Fetching BoardEx names...")
    link_table = io.load_or_fetch(
        config.RAW_EXEC_BOARDEX_LINK_PATH,
        db.fetch_exec_boardex_link
    )
    # link_table has director_id and director_name
    bdx_names = link_table[['director_id', 'director_name']].drop_duplicates('director_id')
    
    # Ensure director_id types match
    # Remove .0 if present (common float->str artifact)
    spells_with_ann['director_id'] = spells_with_ann['director_id'].astype(str).str.replace(r'\.0$', '', regex=True)
    bdx_names['director_id'] = bdx_names['director_id'].astype(str).str.replace(r'\.0$', '', regex=True)
    
    print(f"Sample Spells Director IDs: {spells_with_ann['director_id'].head().tolist()}")
    print(f"Sample BoardEx Director IDs: {bdx_names['director_id'].head().tolist()}")
    
    # Merge BoardEx names
    merged = pd.merge(spells_with_ann, bdx_names, on='director_id', how='left')
    
    # 4. Compare
    logger.info("Calculating similarity for linked CEOs...")
    
    # Filter for rows where both names are present
    valid_comparison = merged.dropna(subset=['exec_fullname', 'director_name']).copy()
    missing_names_count = len(merged) - len(valid_comparison)
    
    if missing_names_count > 0:
        logger.warning(f"Skipping {missing_names_count:,} rows due to missing names in Execucomp or BoardEx.")
    
    valid_comparison['similarity'] = valid_comparison.apply(
        lambda row: similarity(row['exec_fullname'], row['director_name']), axis=1
    )
    
    # Stats
    print("\nSimilarity Stats for BoardEx-Identified CEOs (Valid Names Only):")
    print(valid_comparison['similarity'].describe())
    
    # Low similarity check
    low_sim = valid_comparison[valid_comparison['similarity'] < 0.8].sort_values('similarity')
    logger.info(f"Found {len(low_sim):,} used links with similarity < 0.8")
    
    if not low_sim.empty:
        print("\nSample of low similarity matches (Lowest 20):")
        cols = ['exec_fullname', 'director_name', 'similarity', 'announcement_date']
        print(low_sim[cols].head(20))
        
    # Perfect matches
    perfect = valid_comparison[valid_comparison['similarity'] == 1.0]
    logger.info(f"Found {len(perfect):,} perfect matches ({len(perfect)/len(valid_comparison):.1%})")


if __name__ == "__main__":
    run_verification()
