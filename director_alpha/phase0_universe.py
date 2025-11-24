import pandas as pd
import numpy as np
from typing import Tuple, Optional
from . import config, db, io, transform, log

logger = log.logger

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load raw Compustat, CRSP, and CCM data.
    """
    compustat = io.load_or_fetch(
        config.RAW_COMPUSTAT_PATH,
        db.fetch_compustat_funda,
        start_year=config.UNIVERSE_START_YEAR
    )

    crsp = io.load_or_fetch(
        config.RAW_CRSP_PATH,
        db.fetch_crsp_msf,
        start_date=config.UNIVERSE_START_DATE
    )

    ccm = io.load_or_fetch(
        config.RAW_CCM_PATH,
        db.fetch_ccm_link
    )
    
    return compustat, crsp, ccm

def link_compustat_crsp(compustat: pd.DataFrame, ccm: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Merge Compustat and CRSP using the CCM link table.
    """
    logger.info("Merging Compustat and CCM...")

    # Filter CCM
    if "linktype" in ccm.columns:
        ccm = ccm[ccm["linktype"].isin(config.LINK_TYPES)]
    
    if "linkprim" in ccm.columns:
        ccm = ccm[ccm["linkprim"].isin(config.LINK_PRIM)]

    # Normalize GVKEY
    if "gvkey" not in compustat.columns or "gvkey" not in ccm.columns:
        logger.error("Missing 'gvkey' in Compustat or CCM.")
        return None

    compustat = compustat.copy()
    ccm = ccm.copy()
    
    compustat["gvkey"] = transform.normalize_gvkey(compustat["gvkey"])
    ccm["gvkey"] = transform.normalize_gvkey(ccm["gvkey"])

    # Merge
    merged = pd.merge(compustat, ccm, on="gvkey", how="inner")

    # Date Handling
    if "datadate" not in merged.columns:
        logger.error("Compustat dataframe missing 'datadate'.")
        return None

    merged["datadate"] = pd.to_datetime(merged["datadate"], errors="coerce")
    merged = merged[merged["datadate"].notna()]

    # CCM Dates
    merged["linkdt"] = pd.to_datetime(merged["linkdt"], errors="coerce")
    merged["linkenddt"] = pd.to_datetime(merged["linkenddt"], errors="coerce").fillna(pd.Timestamp("2099-12-31"))
    
    merged = merged[merged["linkdt"].notna()]

    # Valid Link Check
    link_valid_mask = (merged["datadate"] >= merged["linkdt"]) & (merged["datadate"] <= merged["linkenddt"])
    merged = merged[link_valid_mask]

    if merged.empty:
        logger.error("No valid Compustat–CCM link records found.")
        return None

    # Rename and Clean PERMNO
    merged = merged.rename(columns={"lpermno": "permno"})
    merged["permno"] = pd.to_numeric(merged["permno"], errors="coerce")
    merged = merged.dropna(subset=["permno"])
    merged["permno"] = merged["permno"].astype(int)

    return merged

def apply_universe_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply filters: US Incorporation, Non-Financial, Non-Utility.
    """
    logger.info("Applying universe filters...")
    
    # US Incorporated
    if "fic" in df.columns:
        before = len(df)
        df = df[df["fic"] == "USA"]
        logger.info(f"FIC filter (USA): kept {len(df)} of {before} rows.")
    
    # SIC Filters
    sic_col = "sich" if "sich" in df.columns else ("sic" if "sic" in df.columns else None)
    
    if sic_col:
        df = df.copy()
        df[sic_col] = pd.to_numeric(df[sic_col], errors="coerce")
        sic = df[sic_col]
        
        is_fin = (sic >= config.SIC_FIN_START) & (sic <= config.SIC_FIN_END)
        is_util = (sic >= config.SIC_UTIL_START) & (sic <= config.SIC_UTIL_END)
        
        before = len(df)
        df = df[~(is_fin | is_util)]
        logger.info(f"Financial/Utility SIC filter: removed {before - len(df)} rows.")
    
    return df

def filter_common_stock(merged_df: pd.DataFrame, crsp: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Restrict to common stock (Share Codes 10, 11) using CRSP.
    """
    logger.info("Checking Share Codes...")
    
    crsp = crsp.copy()
    crsp["permno"] = pd.to_numeric(crsp["permno"], errors="coerce")
    crsp = crsp.dropna(subset=["permno", "date", "shrcd"])
    crsp["permno"] = crsp["permno"].astype(int)
    crsp["date"] = pd.to_datetime(crsp["date"])
    crsp["year"] = crsp["date"].dt.year
    
    # Identify common stock
    crsp["is_common"] = crsp["shrcd"].isin(config.SHARE_CODES)
    
    # Permno-Year is valid if it was common stock at any point in that year
    valid_permno_years = crsp.groupby(["permno", "year"])["is_common"].any().reset_index()
    valid_permno_years = valid_permno_years[valid_permno_years["is_common"]].drop(columns=["is_common"])
    
    # Merge
    merged_df["year"] = merged_df["datadate"].dt.year
    
    final_df = pd.merge(merged_df, valid_permno_years, on=["permno", "year"], how="inner")
    
    return final_df

def run_phase0():
    """
    Phase 0: Universe Definition (Compustat–CRSP firm-year base)
    """
    logger.info("Starting Phase 0: Universe Definition...")

    # 1. Load Data
    compustat, crsp, ccm = load_data()
    if compustat.empty or crsp.empty or ccm.empty:
        logger.error("One or more input datasets are empty. Aborting Phase 0.")
        return

    # 2. Link
    merged = link_compustat_crsp(compustat, ccm)
    if merged is None:
        return

    # 3. Filter Universe
    merged = apply_universe_filters(merged)

    # 4. Filter Common Stock
    final_df = filter_common_stock(merged, crsp)
    if final_df is None or final_df.empty:
        logger.error("No records after merging with CRSP share-code universe.")
        return

    # 5. Select Columns and Deduplicate
    available_cols = [c for c in config.COMPUSTAT_COLS_TO_KEEP if c in final_df.columns]
    firm_year_base = final_df[available_cols].copy()

    if {"gvkey", "fyear", "datadate"}.issubset(firm_year_base.columns):
        firm_year_base = (
            firm_year_base
            .sort_values(["gvkey", "fyear", "datadate"])
            .groupby(["gvkey", "fyear"], as_index=False)
            .tail(1)
        )

    logger.info(f"Phase 0 Complete. Generated {len(firm_year_base)} firm-year observations.")
    
    firm_year_base.to_parquet(config.FIRM_YEAR_BASE_PATH, index=False)

if __name__ == "__main__":
    run_phase0()
