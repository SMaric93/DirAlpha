import pandas as pd
from pathlib import Path
from typing import Union, Optional, Callable, Any
from . import log, db

logger = log.logger

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
    _db = db_connection
    if _db is None:
        _db = db.get_db()
    
    # Execute fetch
    try:
        df = fetch_func(_db, **kwargs)
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
