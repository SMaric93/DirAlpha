import pandas as pd
import numpy as np
from difflib import SequenceMatcher

# -----------------------------------------------------------------------------
# Helper: Data Normalization
# -----------------------------------------------------------------------------

def normalize_history(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """
    Standardize history columns to a common schema.
    Target Schema:
        - person_id (str): Source-specific ID (DirectorID or ProfessionalID)
        - company_identifier (str): Source-specific Company ID
        - company_name (str): Human-readable company name
        - role (str): Job title / Role name
        - start_date (datetime): Role start
        - end_date (datetime): Role end (NaT = Current)
        - source (str): 'BoardEx' or 'CapitalIQ'
    """
    if df.empty:
        return pd.DataFrame(columns=['person_id', 'company_identifier', 'company_name', 'role', 'start_date', 'end_date', 'source'])

    df = df.copy()
    
    if source == 'BoardEx':
        # Input: director_id, company_id, company_name, rolename, date_start, date_end
        cols_map = {
            'director_id': 'person_id',
            'rolename': 'role',
            'date_start': 'start_date',
            'date_end': 'end_date',
            'company_id': 'company_identifier'
        }
        df = df.rename(columns=cols_map)
        df['source'] = 'BoardEx'
        
        # Default company name if missing
        if 'company_name' not in df.columns:
             df['company_name'] = "BoardEx_ID_" + df['company_identifier'].astype(str)
        else:
             df['company_name'] = df['company_name'].fillna("BoardEx_ID_" + df['company_identifier'].astype(str))
        
    elif source == 'CapitalIQ':
        # Input: professionalid, companyname, jobtitle, startdate, enddate, companyid
        cols_map = {
            'professionalid': 'person_id',
            'jobtitle': 'role',
            'startdate': 'start_date',
            'enddate': 'end_date',
            'companyname': 'company_name',
            'companyid': 'company_identifier'
        }
        df = df.rename(columns=cols_map)
        df['source'] = 'CapitalIQ'
    
    # Type Enforcement
    df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
    df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')
    df['person_id'] = df['person_id'].astype(str)
    df['company_identifier'] = df['company_identifier'].astype(str)
    
    # Select and Order
    final_cols = ['person_id', 'company_identifier', 'company_name', 'role', 'start_date', 'end_date', 'source']
    # Ensure all columns exist
    for c in final_cols:
        if c not in df.columns:
            df[c] = None
            
    return df[final_cols]

# -----------------------------------------------------------------------------
# Helper: Consolidation Logic
# -----------------------------------------------------------------------------

def are_similar(name1: str, name2: str, threshold: float = 0.85) -> bool:
    """Check if two company names are fuzzily similar."""
    if not isinstance(name1, str) or not isinstance(name2, str):
        return False
    # Simple optimization: direct match
    if name1.lower() == name2.lower():
        return True
    return SequenceMatcher(None, name1.lower(), name2.lower()).ratio() > threshold

def merge_records(r1: dict, r2: dict, id_col: str) -> dict:
    """
    Merge two overlapping employment records into a single combined record.
    """
    # Start Date: Earliest known
    start = min(r1['start_date'], r2['start_date'])
    
    # End Date: Logic -> If either is NaT (Current), result is NaT (Current). Else Max.
    if pd.isna(r1['end_date']) or pd.isna(r2['end_date']):
        end = pd.NaT
    else:
        end = max(r1['end_date'], r2['end_date'])
        
    # Text fields: Prefer longer strings (heuristic for descriptiveness)
    role = r1['role'] if len(str(r1.get('role',''))) >= len(str(r2.get('role',''))) else r2['role']
    comp = r1['company_name'] if len(str(r1.get('company_name',''))) >= len(str(r2.get('company_name',''))) else r2['company_name']
    
    # Source Lineage
    src1 = str(r1.get('source', ''))
    src2 = str(r2.get('source', ''))
    combined_src = f"{src1}|{src2}"
    # Deduplicate tokens
    src_tokens = sorted(list(set(combined_src.split('|'))))
    final_src = "|".join([t for t in src_tokens if t])
    
    return {
        id_col: r1[id_col],
        'company_name': comp,
        'role': role,
        'start_date': start,
        'end_date': end,
        'source': final_src
    }

def consolidate_history(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """
    Iteratively merge overlapping/duplicate records for each executive.
    """
    if df.empty:
        return df
        
    # Sort: Group by ID, then Start Date (NaT last)
    df = df.sort_values([id_col, 'start_date'], na_position='last')
    
    merged_rows = []
    
    # Group by Entity (Director/CEO)
    for entity_id, group in df.groupby(id_col):
        records = group.to_dict('records')
        if not records:
            continue
            
        # Stack of Consolidated Records
        stack = [records[0]]
        
        for i in range(1, len(records)):
            current_rec = records[i]
            merged = False
            
            # Compare with recent items in stack (lookback)
            for j in range(len(stack) - 1, -1, -1):
                candidate = stack[j]
                
                # Criteria 1: Name Similarity
                name_match = are_similar(current_rec.get('company_name'), candidate.get('company_name'))
                
                # Criteria 2: Temporal Proximity
                try:
                    days_diff = abs((current_rec['start_date'] - candidate['start_date']).days)
                    time_match = days_diff < 365
                except:
                    time_match = False 
                
                if name_match and time_match:
                    # Merge and update stack
                    new_merged = merge_records(candidate, current_rec, id_col)
                    stack[j] = new_merged
                    merged = True
                    break
            
            if not merged:
                stack.append(current_rec)
        
        merged_rows.extend(stack)
        
    result = pd.DataFrame(merged_rows)
    cols = [id_col, 'company_name', 'role', 'start_date', 'end_date', 'source']
    return result[cols]