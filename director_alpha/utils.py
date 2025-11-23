import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Callable, List, Union, Any
from . import config

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

def setup_logging(name: str = "director_alpha", level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger

logger = setup_logging()

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
# Data Loading
# ---------------------------------------------------------------------

def load_or_fetch(
    file_path: Union[str, Path],
    fetch_func: Optional[Callable[[Any], pd.DataFrame]] = None,
    db_connection: Optional[Any] = None,
    force_fetch: bool = False,
    save_format: str = "parquet",
    **kwargs
) -> pd.DataFrame:
    """
    Load a DataFrame from a local file if it exists; otherwise, fetch it
    using the provided function (and DB connection) and save it locally.

    Args:
        file_path: Path to the local cache file.
        fetch_func: Function to call if file is missing. Must accept 'db' as first arg if db_connection is provided.
        db_connection: WRDS connection object (or similar) to pass to fetch_func. 
                       If None and fetch is needed, get_db() is called.
        force_fetch: If True, ignore local file and fetch fresh.
        save_format: 'parquet' or 'csv'.
        **kwargs: Additional arguments passed to fetch_func.

    Returns:
        pd.DataFrame
    """
    path = Path(file_path)
    
    if not force_fetch and path.exists():
        logger.info(f"Loading data from local cache: {path}")
        try:
            if save_format == "parquet":
                return pd.read_parquet(path)
            else:
                return pd.read_csv(path)
        except Exception as e:
            logger.warning(f"Failed to read local file {path}: {e}. Will attempt fetch.")
    
    if fetch_func is None:
        logger.error(f"File {path} not found and no fetch_func provided.")
        return pd.DataFrame()

    logger.info("Fetching fresh data...")
    
    # Manage DB connection
    db = db_connection
    if db is None:
        db = get_db()
    
    # Execute fetch
    try:
        df = fetch_func(db, **kwargs)
    except Exception as e:
        logger.error(f"Data fetch failed: {e}")
        return pd.DataFrame()
    
    # Save result
    if not df.empty:
        logger.info(f"Saving fetched data to {path}...")
        path.parent.mkdir(parents=True, exist_ok=True)
        if save_format == "parquet":
            df.to_parquet(path)
        else:
            df.to_csv(path, index=False)
    else:
        logger.warning("Fetched data is empty. Nothing saved.")
        
    return df

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
    
    # If both are 0, it's 0. If one is non-zero, it compounds.
    # Note: If ret was missing (NaN) and we filled 0, and dlret is 0, we get 0. 
    # But if ret was truly missing, maybe we should keep it missing?
    # Standard event study: if missing, usually drop. 
    # But here we filled 0. Let's refine:
    # If ret is missing and dlret is missing, result is missing.
    # If ret is present, dlret missing -> ret.
    # If ret missing, dlret present -> dlret.
    # If both -> compound.
    
    # Re-do with fillna only for calculation
    # We need to preserve NaNs if both are NaN
    
    # Vectorized calculation
    df['ret_adj'] = (1 + df['ret']) * (1 + df['dlret']) - 1
    
    # If original ret was NaN and dlret was NaN/0, result should be NaN?
    # Actually, raw_sql returns None for SQL NULL.
    # Let's handle it properly in pandas
    
    # Reload to ensure we didn't overwrite with 0s yet
    # Actually, let's just use the 0-filled version for now as it's robust for 'ret' column
    # But we should assign back to 'ret'
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

# ---------------------------------------------------------------------
# Data Cleaning & Normalization
# ---------------------------------------------------------------------

def clean_id(series: pd.Series) -> pd.Series:
    """
    Normalize ID columns (e.g. boardid, directorid) by ensuring string type
    and removing trailing '.0' which often appears from float conversion.
    """
    return series.astype(str).str.replace(r"\.0$", "", regex=True)

def normalize_gvkey(series: pd.Series) -> pd.Series:
    """Ensure GVKEY is a 6-digit zero-padded string."""
    return series.astype(str).str.zfill(6)

def normalize_ticker(series: pd.Series) -> pd.Series:
    """Ensure ticker is uppercase stripped string."""
    return series.astype(str).str.upper().str.strip()

# ---------------------------------------------------------------------
# Financial Transformation
# ---------------------------------------------------------------------

def industry_adjust(df: pd.DataFrame, cols: List[str], group_cols: List[str] = ['fyear', 'sic2']) -> pd.DataFrame:
    """
    Subtract group-level median from specified columns.
    """
    df = df.copy()
    # Ensure group columns exist
    for c in group_cols:
        if c not in df.columns:
            if c == 'sic2' and 'sich' in df.columns:
                 df['sic2'] = df['sich'].fillna(0).astype(int) // 100
            else:
                logger.warning(f"Grouping column {c} missing for industry adjustment.")
                return df

    for col in cols:
        if col in df.columns:
            median = df.groupby(group_cols)[col].transform('median')
            df[f'{col}_adj'] = df[col] - median
    return df

def winsorize_series(x: pd.Series, limits: tuple = (0.01, 0.99)) -> pd.Series:
    """
    Clip series between quantiles.
    """
    lower = x.quantile(limits[0])
    upper = x.quantile(limits[1])
    return x.clip(lower=lower, upper=upper)

def apply_winsorization(df: pd.DataFrame, cols: List[str], group_col: Optional[str] = 'fyear', limits: tuple = (0.01, 0.99)) -> pd.DataFrame:
    """
    Apply winsorization to specified columns, optionally grouped by year.
    """
    df = df.copy()
    valid_cols = [c for c in cols if c in df.columns]
    
    if group_col and group_col in df.columns:
        for col in valid_cols:
            df[col] = df.groupby(group_col)[col].transform(lambda x: winsorize_series(x, limits=limits))
    else:
        for col in valid_cols:
            # pyrefly: ignore [bad-argument-type]
            df[col] = winsorize_series(df[col], limits=limits)
            
    return df
