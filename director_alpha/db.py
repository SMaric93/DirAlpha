import os
import pandas as pd
from typing import List
from . import config, log

logger = log.logger

# ---------------------------------------------------------------------
# Database Connection
# ---------------------------------------------------------------------

def get_db():
    """
    Get a WRDS database connection using configuration credentials.
    """
    try:
        # Try importing wrds
        import wrds
    except ImportError:
        logger.error("The 'wrds' library is not installed. Please install it to fetch data.")
        raise RuntimeError("'wrds' library missing.")

    # Check environment variables via config (or directly)
    username = getattr(config, "WRDS_USERNAME", os.getenv("WRDS_USERNAME"))
    
    if not username:
        logger.warning("WRDS_USERNAME not found in environment or config. Connection might fail if no .pgpass.")

    try:
        logger.info(f"Connecting to WRDS (user: {username})...")
        db = wrds.Connection(wrds_username=username)
        return db
    except Exception as e:
        logger.error(f"WRDS connection failed: {e}")
        raise RuntimeError(f"WRDS connection failed: {e}")

# ---------------------------------------------------------------------
# WRDS Fetchers (Optimized and Revised)
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Compustat
# ---------------------------------------------------------------------

def fetch_compustat_funda(db, start_year: int = config.UNIVERSE_START_YEAR) -> pd.DataFrame:
    """
    Fetch Compustat Fundamentals Annual.
    Uses historical NAICS (naicsh) for point-in-time accuracy.
    Includes historical (sich) SIC codes, aligning with config.py.
    """
    query = f"""
        SELECT 
            gvkey, datadate, fyear, fic, 
            at, oibdp, prcc_f, csho, ceq, dltt, dlc, xrd, capx, 
            sich, naicsh as naics
        FROM {config.WRDS_COMP_FUNDA}
        WHERE indfmt='INDL' AND datafmt='STD' AND popsrc='D' AND consol='C'
        AND fyear >= {start_year}
    """
    return db.raw_sql(query)

# ---------------------------------------------------------------------
# CRSP
# ---------------------------------------------------------------------

def fetch_crsp_msf(db, start_date: str = config.UNIVERSE_START_DATE) -> pd.DataFrame:
    """
    Fetch CRSP Monthly Stock File.
    Optimized: Filters for common stock (e.g., SHRCD 10, 11) server-side.
    """
    # Optimization: Push SHRCD filter to SQL
    shrcd_list = getattr(config, "SHARE_CODES", [10, 11])
    shrcd_str = ",".join(map(str, shrcd_list))

    # Join with msenames to get shrcd and siccd
    query = f"""
        SELECT a.permno, a.date, b.shrcd, b.siccd, a.prc, a.ret
        FROM {config.WRDS_CRSP_MSF} a
        JOIN {config.WRDS_CRSP_MSENAMES} b
        ON a.permno = b.permno
        AND a.date >= b.namedt AND a.date <= b.nameendt
        WHERE a.date >= '{start_date}'
        AND b.shrcd IN ({shrcd_str}) 
    """
    return db.raw_sql(query)

def fetch_crsp_dsf(db, permno_list: List[int], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch CRSP Daily Stock File for specific PERMNOs and date range.
    Optimized with chunking.
    Crucial: Incorporates delisting returns (DLRET) for accurate total returns (required for BHAR).
    """
    if not permno_list:
        return pd.DataFrame()

    # Chunking permnos
    chunk_size = 500
    dfs = []
    
    unique_permnos = list(set(permno_list))
    
    logger.info(f"Fetching daily returns for {len(unique_permnos)} permnos from {start_date} to {end_date}...")

    for i in range(0, len(unique_permnos), chunk_size):
        chunk = unique_permnos[i:i + chunk_size]
        permno_str = ",".join(map(str, chunk))
        
        # Join with Delisting Returns (Left Join)
        query = f"""
            SELECT a.permno, a.date, a.ret, b.dlret
            FROM {config.WRDS_CRSP_DSF} a
            LEFT JOIN {config.WRDS_CRSP_DSEDELIST} b
            ON a.permno = b.permno AND a.date = b.dlstdt
            WHERE a.permno IN ({permno_str})
            AND a.date >= '{start_date}' AND a.date <= '{end_date}'
        """
        try:
            df_chunk = db.raw_sql(query)
            dfs.append(df_chunk)
        except Exception as e:
            logger.warning(f"Failed to fetch DSF chunk {i}: {e}")

    if not dfs:
        return pd.DataFrame()
        
    df = pd.concat(dfs, ignore_index=True)
    
    # Calculate Total Return (incorporating delisting)
    # ret_adj = (1 + ret) * (1 + dlret) - 1
    df['ret'] = pd.to_numeric(df['ret'], errors='coerce').fillna(0)
    df['dlret'] = pd.to_numeric(df['dlret'], errors='coerce').fillna(0)
    
    # Vectorized calculation
    df['ret_adj'] = (1 + df['ret']) * (1 + df['dlret']) - 1
    
    df['ret'] = df['ret_adj']
    df = df.drop(columns=['dlret', 'ret_adj'])
    
    return df

# ---------------------------------------------------------------------
# Fama-French Factors
# ---------------------------------------------------------------------

def fetch_fama_french(db, start_date: str) -> pd.DataFrame:
    """Fetch Fama-French 3 Factors (Daily)."""
    query = f"""
        SELECT date, mktrf, smb, hml, rf
        FROM {config.WRDS_FF_FACTORS_DAILY}
        WHERE date >= '{start_date}'
    """
    return db.raw_sql(query)

def fetch_fama_french_5_daily(db, start_date: str) -> pd.DataFrame:
    """Fetch Fama-French 5 Factors (Daily)."""
    query = f"""
        SELECT date, mktrf, smb, hml, rmw, cma, rf
        FROM {config.WRDS_FF5_FACTORS_DAILY}
        WHERE date >= '{start_date}'
    """
    return db.raw_sql(query)

# ---------------------------------------------------------------------
# Linking Tables
# ---------------------------------------------------------------------

def fetch_ccm_link(db) -> pd.DataFrame:
    """
    Fetch CCM Link Table.
    Optimized: Filters for relevant Link Types (LU, LC) and Primary Links (P, C) server-side.
    """
    # Optimization: Push filters defined in config to SQL
    link_types = getattr(config, "LINK_TYPES", ["LU", "LC"])
    link_prim = getattr(config, "LINK_PRIM", ["P", "C"])
    
    link_types_str = ",".join([f"'{lt}'" for lt in link_types])
    link_prim_str = ",".join([f"'{lp}'" for lp in link_prim])

    query = f"""
        SELECT gvkey, lpermno, linkdt, linkenddt, linktype, linkprim
        FROM {config.WRDS_CCM_LINK}
        WHERE linktype IN ({link_types_str})
        AND linkprim IN ({link_prim_str})
    """
    return db.raw_sql(query)

def fetch_boardex_link(db) -> pd.DataFrame:
    """
    Fetch BoardEx-CRSP link.
    Strategy:
    1. Try dedicated WRDS table (wrdsapps.boardex_ccm_link).
    2. Fallback to manual Ticker matching (BoardEx Profile -> CRSP Stocknames -> CCM).
    """
    # -------------------------------------------------------------------------
    # Strategy 1: Dedicated WRDS Table
    # -------------------------------------------------------------------------
    DEDICATED_LINK_TABLE = getattr(config, "WRDS_BOARDEX_CCM_LINK", "wrdsapps.boardex_ccm_link")
    logger.info(f"Attempting to fetch dedicated WRDS BoardEx-CCM link table ({DEDICATED_LINK_TABLE})...")
    
    query = f"""
        SELECT 
            companyid, 
            gvkey, 
            permco, 
            score, 
            duplicate, 
            preferred
        FROM {DEDICATED_LINK_TABLE}
    """
    try:
        df = db.raw_sql(query)
        
        if 'companyid' in df.columns:
             df = df.rename(columns={'companyid': 'company_id'})
             df['company_id'] = df['company_id'].astype(str)
        
        if 'gvkey' in df.columns:
             df['gvkey'] = df['gvkey'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(6)

        logger.info(f"Fetched dedicated link table with {len(df)} rows.")
        return df
    except Exception as e:
        logger.warning(f"Dedicated BoardEx link table failed ({e}). Falling back to manual linking...")

    # -------------------------------------------------------------------------
    # Strategy 2: Manual Fallback (Ticker Matching)
    # -------------------------------------------------------------------------
    logger.info("Starting manual BoardEx-CRSP linking (Fallback)...")
    
    # 1. BoardEx Profile (Get Tickers)
    prof_query = f"SELECT boardid, ticker, isin FROM {config.WRDS_BOARDEX_PROFILE} WHERE ticker IS NOT NULL"
    try:
        prof = db.raw_sql(prof_query)
    except Exception as e:
        logger.error(f"Failed to fetch BoardEx profile: {e}")
        return pd.DataFrame()

    prof["boardid"] = prof["boardid"].astype(str)
    prof["ticker"] = prof["ticker"].astype(str).str.upper().str.strip()
    prof = prof[prof["ticker"] != ""]
    
    unique_tickers = prof["ticker"].unique().tolist()
    logger.info(f"Filtering CRSP stocknames for {len(unique_tickers)} unique BoardEx tickers...")
    
    # 2. CRSP Stocknames (Get PERMNOs)
    sn_table = getattr(config, "WRDS_CRSP_STOCKNAMES", "crsp.stocknames")
    chunk_size = 1000
    sn_dfs = []
    
    for i in range(0, len(unique_tickers), chunk_size):
        chunk = unique_tickers[i:i + chunk_size]
        chunk_str = ",".join(["'" + t.replace("'", "''") + "'" for t in chunk])
        sn_query = f"SELECT permno, ticker FROM {sn_table} WHERE ticker IN ({chunk_str})"
        try:
            sn_dfs.append(db.raw_sql(sn_query))
        except Exception:
            pass
            
    if not sn_dfs:
        logger.warning("No matching CRSP stocknames found.")
        return pd.DataFrame()
        
    sn = pd.concat(sn_dfs, ignore_index=True)
    sn["ticker"] = sn["ticker"].astype(str).str.upper().str.strip()
    sn = sn.drop_duplicates(subset=["permno", "ticker"])
    
    # 3. CCM Link (Get GVKEYs)
    ccm_table = getattr(config, "WRDS_CCM_LINKTABLE", "crsp.ccmxpf_linktable")
    ccm_query = f"SELECT gvkey, lpermno AS permno FROM {ccm_table} WHERE linktype IN ('LC','LU') AND linkprim IN ('P','C')"
    try:
        ccm = db.raw_sql(ccm_query)
    except Exception as e:
        logger.error(f"Failed to fetch CCM link table: {e}")
        return pd.DataFrame()
        
    # 4. Merge
    m1 = pd.merge(prof, sn, on="ticker", how="inner")
    m2 = pd.merge(m1, ccm, on="permno", how="left")
    
    link = m2.rename(columns={"boardid": "company_id"})
    if "gvkey" in link.columns:
        link["gvkey"] = link["gvkey"].fillna("").astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(6)
        
    # Select relevant columns
    cols = ["company_id", "gvkey", "permno", "ticker", "isin"]
    link = link[[c for c in cols if c in link.columns]].drop_duplicates()
    
    logger.info(f"Manual fallback successful. Built link table with {len(link)} rows.")
    return link

# ---------------------------------------------------------------------
# ExecuComp
# ---------------------------------------------------------------------

def fetch_execucomp(db, start_year: int = config.UNIVERSE_START_YEAR) -> pd.DataFrame:
    """Fetch ExecuComp Annual Compensation."""
    query = f"""
        SELECT gvkey, year, execid, pceo, ceoann, title, tdc1, becameceo, leftofc, joined_co, age, gender, ticker
        FROM {config.WRDS_EXECUCOMP_ANNCOMP}
        WHERE year >= {start_year}
    """
    return db.raw_sql(query)

# ---------------------------------------------------------------------
# BoardEx Composition
# ---------------------------------------------------------------------

def fetch_boardex_directors(db) -> pd.DataFrame:
    """Fetch BoardEx Directors (Composition)."""
    query = f"""
        SELECT
            companyid     AS company_id,
            directorid    AS director_id,
            datestartrole AS date_start,
            dateendrole   AS date_end,
            rolename      AS role_name
        FROM {config.WRDS_BOARDEX_DIRECTORS}
    """
    df = db.raw_sql(query)
    # Clean types immediately to avoid parquet type issues later
    df["company_id"] = df["company_id"].astype(str)
    df["director_id"] = df["director_id"].astype(str)
    return df

def fetch_boardex_committees(db) -> pd.DataFrame:
    """Fetch BoardEx Committees."""
    # Note: Using boardid as company_id based on standard BoardEx structure for this table
    query = f"""
        SELECT
            boardid       AS company_id,
            directorid    AS director_id,
            committeename AS committee_name,
            datestartrole AS c_date_start,
            dateendrole   AS c_date_end
        FROM {config.WRDS_BOARDEX_COMMITTEES}
    """
    df = db.raw_sql(query)
    df["company_id"] = df["company_id"].astype(str)
    df["director_id"] = df["director_id"].astype(str)
    return df