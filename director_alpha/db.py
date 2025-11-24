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
# WRDS Fetchers
# ---------------------------------------------------------------------

def fetch_compustat_funda(db, start_year: int = 2000) -> pd.DataFrame:
    """Fetch Compustat Fundamentals Annual."""
    query = f"""
        SELECT gvkey, datadate, fyear, fic, at, oibdp, prcc_f, csho, ceq, dltt, dlc, xrd, capx, sich, naicsh as naics
        FROM {config.WRDS_COMP_FUNDA}
        WHERE indfmt='INDL' AND datafmt='STD' AND popsrc='D' AND consol='C'
        AND fyear >= {start_year}
    """
    return db.raw_sql(query)

def fetch_crsp_msf(db, start_date: str = '2000-01-01') -> pd.DataFrame:
    """Fetch CRSP Monthly Stock File with Share Codes."""
    # Join with msenames to get shrcd and siccd
    query = f"""
        SELECT a.permno, a.date, b.shrcd, b.siccd, a.prc, a.ret
        FROM {config.WRDS_CRSP_MSF} a
        JOIN {config.WRDS_CRSP_MSENAMES} b
        ON a.permno = b.permno
        AND a.date >= b.namedt AND a.date <= b.nameendt
        WHERE a.date >= '{start_date}'
    """
    return db.raw_sql(query)

def fetch_crsp_dsf(db, permno_list: List[int], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch CRSP Daily Stock File for specific PERMNOs and date range.
    Optimized with chunking for large PERMNO lists.
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
        # Handle cases where stock delists
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

def fetch_ccm_link(db) -> pd.DataFrame:
    """Fetch CCM Link Table."""
    query = f"""
        SELECT gvkey, lpermno, linkdt, linkenddt, linktype, linkprim
        FROM {config.WRDS_CCM_LINK}
        WHERE linktype IN ('LU', 'LC')
    """
    return db.raw_sql(query)

def fetch_execucomp(db, start_year: int = 2000) -> pd.DataFrame:
    """Fetch ExecuComp Annual Compensation."""
    # Standard ExecuComp: gvkey, year, execid, pceo, ceoann, title, tdc1, becameceo, leftofc, joined_co, age, gender
    query = f"""
        SELECT gvkey, year, execid, pceo, ceoann, title, tdc1, becameceo, leftofc, joined_co, age, gender, ticker
        FROM {config.WRDS_EXECUCOMP_ANNCOMP}
        WHERE year >= {start_year}
    """
    return db.raw_sql(query)

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

def fetch_boardex_link(db) -> pd.DataFrame:
    """
    Build BoardEx–CRSP–CCM link table.
    Optimized to avoid OOM by filtering CRSP stocknames server-side using an IN clause.
    """
    # 1. BoardEx Profile
    logger.info("Fetching BoardEx profile...")
    prof_query = f"SELECT boardid, ticker, isin FROM {config.WRDS_BOARDEX_PROFILE} WHERE ticker IS NOT NULL"
    prof = db.raw_sql(prof_query)
    prof["boardid"] = prof["boardid"].astype(str)
    if "ticker" in prof.columns:
        prof["ticker"] = prof["ticker"].astype(str).str.upper().str.strip()
    
    prof = prof.dropna(subset=["ticker"])
    prof = prof[prof["ticker"] != ""]
    
    if prof.empty:
        logger.warning("BoardEx profile has no valid tickers.")
        # pyrefly: ignore [bad-argument-type]
        return pd.DataFrame(columns=["company_id", "ticker", "isin", "gvkey"])

    # 2. CRSP Stocknames (Optimized)
    # Use IN clause to filter CRSP server-side
    unique_tickers = prof["ticker"].unique().tolist()
    logger.info(f"Filtering CRSP stocknames for {len(unique_tickers)} unique BoardEx tickers...")
    
    sn_table = getattr(config, "WRDS_CRSP_STOCKNAMES", "crsp.stocknames")
    
    # Chunking to be safe (Postgres handles large queries, but let's be robust)
    chunk_size = 1000
    sn_dfs = []
    
    for i in range(0, len(unique_tickers), chunk_size):
        chunk = unique_tickers[i:i + chunk_size]
        # Escape single quotes just in case
        chunk_str = ",".join(["'" + t.replace("'", "''") + "'" for t in chunk])
        
        sn_query = f"""
            SELECT permno, ticker 
            FROM {sn_table} 
            WHERE ticker IN ({chunk_str})
        """
        try:
            df_chunk = db.raw_sql(sn_query)
            sn_dfs.append(df_chunk)
        except Exception as e:
            logger.warning(f"Failed to fetch chunk {i}-{i+chunk_size}: {e}")

    if not sn_dfs:
        logger.warning("No matching CRSP stocknames found.")
        # pyrefly: ignore [bad-argument-type]
        return pd.DataFrame(columns=["company_id", "ticker", "isin", "gvkey"])

    sn = pd.concat(sn_dfs, ignore_index=True)
    sn["ticker"] = sn["ticker"].astype(str).str.upper().str.strip()
    sn = sn.drop_duplicates(subset=["permno", "ticker"])

    # 3. CCM Link
    logger.info("Fetching CCM link table...")
    ccm_table = getattr(config, "WRDS_CCM_LINKTABLE", "crsp.ccmxpf_linktable")
    ccm_query = f"""
        SELECT gvkey, lpermno AS permno, linktype, linkprim
        FROM {ccm_table}
        WHERE linktype IN ('LC','LU') AND linkprim IN ('P','C')
    """
    ccm = db.raw_sql(ccm_query)
    ccm = ccm.dropna(subset=["permno", "gvkey"])

    # Merge
    logger.info("Merging tables...")
    m1 = pd.merge(prof, sn, on="ticker", how="inner", validate="m:m") # Inner join to keep only linked
    m2 = pd.merge(m1, ccm, on="permno", how="left", validate="m:m")

    cols = ["boardid", "ticker", "isin"]
    if "gvkey" in m2.columns:
        cols.append("gvkey")
        m2["gvkey"] = m2["gvkey"].astype(str).str.zfill(6)

    # pyrefly: ignore [no-matching-overload]
    link = m2[cols].drop_duplicates().rename(columns={"boardid": "company_id"})
    link["company_id"] = link["company_id"].astype(str)
    
    logger.info(f"Built link table with {len(link)} rows.")
    return link
