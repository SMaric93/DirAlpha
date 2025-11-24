import pandas as pd
import numpy as np
import re
from . import config, db, io, transform, log

logger = log.logger

def identify_ceos_no_coceo(execucomp: pd.DataFrame, firm_col: str = "gvkey", year_col: str = "year") -> pd.DataFrame:
    """
    Identify the primary CEO for each firm-year in ExecuComp-style data,
    dropping any firm-years with multiple CEOs (co-CEOs).

    Keeps only firm-years where exactly one executive has CEOANN == 'CEO'.
    """
    df = execucomp.copy()

    # ---- Sanity checks ----
    required = {firm_col, year_col, "ceoann"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # ---- CEO flag based on CEOANN only (historical) ----
    ceo_flag = (
        df["ceoann"]
        .astype(str)
        .str.strip()
        .str.upper()
        .eq("CEO")
    )
    ceos = df[ceo_flag].copy()

    # ---- Drop firm-years with more than one CEO (co-CEOs) ----
    counts = (
        ceos.groupby([firm_col, year_col])
            .size()
            .rename("n_ceos")
            .reset_index()
    )

    # Keep only firm-years with exactly one CEO
    single_ceo_keys = counts[counts["n_ceos"] == 1][[firm_col, year_col]]
    ceos = ceos.merge(single_ceo_keys, on=[firm_col, year_col], how="inner")

    return ceos

def identify_turnover(ceos: pd.DataFrame) -> pd.DataFrame:
    """
    Identify turnover events.
    """
    ceos = ceos.sort_values(['gvkey', 'year'])
    
    ceos['prev_execid'] = ceos.groupby('gvkey')['execid'].shift(1)
    ceos['prev_year'] = ceos.groupby('gvkey')['year'].shift(1)
    
    turnover = (ceos['execid'] != ceos['prev_execid']) & \
               (ceos['prev_execid'].notna()) & \
               (ceos['year'] == ceos['prev_year'] + 1)
               
    ceos['is_turnover'] = turnover
    return ceos

def determine_dates(ceos: pd.DataFrame) -> pd.DataFrame:
    """
    Determine appointment dates.
    """
    # Use BECAMECEO
    ceos['appointment_date'] = pd.to_datetime(ceos['becameceo'], errors='coerce')
    ceos['leftofc'] = pd.to_datetime(ceos['leftofc'], errors='coerce')
    ceos['joined_co'] = pd.to_datetime(ceos['joined_co'], errors='coerce')
    return ceos

# Match "interim/acting" + "CEO/chief executive" in either order, with separators
INTERIM_CEO_PATTERN = re.compile(
    r"(?:(?:interim|acting)[ ,();-]*(?:ceo|chief executive))|"
    r"(?:(?:ceo|chief executive)[ ,();-]*(?:interim|acting))",
    flags=re.IGNORECASE,
)

def filter_interim(spells: pd.DataFrame) -> pd.DataFrame:
    """
    Exclude interim/acting CEOs (based on title) and short non-censored spells.
    """
    spells = spells.copy()

    # ---- Required columns ----
    required = {"gvkey", "appointment_date", "leftofc"}
    missing = required - set(spells.columns)
    if missing:
        raise KeyError(f"Missing required columns in spells: {missing}")

    # ---- Ensure datetime types ----
    spells["appointment_date"] = pd.to_datetime(spells["appointment_date"])
    spells["leftofc"] = pd.to_datetime(spells["leftofc"])

    # ---- Drop interim / acting CEOs based on title ----
    if "title" in spells.columns:
        is_interim = (
            spells["title"]
            .astype(str)
            .str.contains(INTERIM_CEO_PATTERN, na=False)
        )
        spells = spells[~is_interim]

    # ---- Compute spell end: leftofc or next appointment within gvkey ----
    spells = spells.sort_values(["gvkey", "appointment_date"])
    spells["next_appointment"] = spells.groupby("gvkey")["appointment_date"].shift(-1)

    # Use leftofc if known, otherwise next appointment
    spells["spell_end"] = spells["leftofc"].where(
        spells["leftofc"].notna(), spells["next_appointment"]
    )

    # ---- Duration and short-spell filter ----
    duration = (spells["spell_end"] - spells["appointment_date"]).dt.days

    # Drop spells < 365 days when duration is observed; keep censored spells
    short_spell = duration.notna() & (duration < 365)
    spells = spells[~short_spell]

    return spells

def classify_hires(spells: pd.DataFrame) -> pd.DataFrame:
    """
    Classify Internal vs External.
    """
    days_since_join = (spells['appointment_date'] - spells['joined_co']).dt.days
    
    # External if joined within 365 days of becoming CEO
    spells['is_external'] = (days_since_join <= 365)
    return spells

def run_phase2():
    logger.info("Starting Phase 2: CEO Tenures and Turnover...")
    
    execucomp = io.load_or_fetch(
        config.RAW_EXECUCOMP_PATH,
        db.fetch_execucomp,
        start_year=2000
    )

    if execucomp.empty:
        logger.error("ExecuComp data missing. Aborting Phase 2.")
        return

    # Normalize GVKEY
    execucomp['gvkey'] = transform.normalize_gvkey(execucomp['gvkey'])

    # 1. Identify CEOs
    ceos = identify_ceos_no_coceo(execucomp)
    
    # 2. Identify Turnover
    ceos = identify_turnover(ceos)
    
    # Parse dates
    ceos = determine_dates(ceos)
    
    # Drop duplicates to get unique spells (start of tenure)
    spells = ceos.sort_values('year').groupby(['gvkey', 'execid', 'appointment_date']).first().reset_index()
    
    # 3. Filter Interim & Short Spells
    spells = filter_interim(spells)
    
    # 4. Classify Hires
    spells = classify_hires(spells)
    
    # Generate Spell ID
    spells['spell_id'] = spells.index + 1
    
    # Select columns
    cols = ['spell_id', 'gvkey', 'ticker', 'execid', 'appointment_date', 'spell_end', 'is_external', 'age', 'gender']
    cols = [c for c in cols if c in spells.columns]
    
    final_spells = spells[cols].copy()
    
    logger.info(f"Phase 2 Complete. Identified {len(final_spells)} CEO spells.")
    
    final_spells.to_parquet(config.CEO_SPELLS_PATH)
    return final_spells

if __name__ == "__main__":
    run_phase2()