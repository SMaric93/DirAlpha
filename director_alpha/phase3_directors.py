"""
Phase 3: Director Selection (BoardEx)

Pipeline:
1. Load CEO spells (with gvkey, appointment_date, spell_id, optional ticker).
2. Load or build BoardEx data:
   - Directors (board composition)
   - Committees
   - BoardEx–CRSP–CCM link (boardid <-> gvkey)
3. Link spells to BoardEx boards via gvkey (fallback: ticker).
4. Build director roster active at appointment_date.
5. Flag search committee membership (Nomination/Gov committees).
6. Compute simple director characteristics.
7. Save linkage to parquet.
"""

import pandas as pd
import numpy as np
from . import config


# ---------------------------------------------------------------------
# WRDS helpers
# ---------------------------------------------------------------------

def get_db():
    """Get a WRDS database connection from config, with a clearer error."""
    db = getattr(config, "get_wrds_connection", None)
    if db is None:
        raise RuntimeError("config.get_wrds_connection() is not defined.")
    conn = db()
    if conn is None:
        raise RuntimeError("WRDS connection failed (config.get_wrds_connection() returned None).")
    return conn


# ---------------------------------------------------------------------
# BoardEx: fetch & cache
# ---------------------------------------------------------------------

def fetch_boardex_directors(db: object) -> pd.DataFrame:
    """
    Fetch BoardEx director composition for North America.

    We use BOARDID as the board/company key everywhere and rename it to company_id.
    """
    query = f"""
        SELECT
            boardid       AS company_id,
            directorid    AS director_id,
            datestartrole AS date_start,
            dateendrole   AS date_end,
            rolename      AS role_name
        FROM {config.WRDS_BOARDEX_DIRECTORS}
    """
    df = db.raw_sql(query)
    # Clean types
    df["company_id"] = df["company_id"].astype(str)
    df["director_id"] = df["director_id"].astype(str)
    return df


def fetch_boardex_committees(db: object) -> pd.DataFrame:
    """
    Fetch BoardEx committee membership.

    Use BOARDID as company_id; date columns are prefixed with c_ to avoid clashes.
    """
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


def fetch_boardex_profile(db: object) -> pd.DataFrame:
    """
    Fetch BoardEx company profile (we only need boardid, ticker, isin).
    """
    query = f"""
        SELECT
            boardid,
            ticker,
            isin
        FROM {config.WRDS_BOARDEX_PROFILE}
    """
    df = db.raw_sql(query)
    df["boardid"] = df["boardid"].astype(str)

    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    if "isin" in df.columns:
        df["isin"] = df["isin"].astype(str).str.strip()

    return df

def build_boardex_ccm_link(db: object, profile: pd.DataFrame) -> pd.DataFrame:
    """
    Build BoardEx–CRSP–CCM link table:

    BoardEx boardid + ticker -> CRSP stocknames (permno) -> CCM linktable (gvkey).

    Output columns: company_id (boardid), ticker, isin, gvkey
    """
    if profile.empty:
        return pd.DataFrame(columns=["company_id", "ticker", "isin", "gvkey"])

    # Keep only rows with a ticker for CRSP linking
    prof = profile.copy()
    prof = prof.dropna(subset=["ticker"])
    if prof.empty:
        # No tickers: can only return BoardEx IDs without gvkey
        return prof.rename(columns={"boardid": "company_id"})[["company_id", "ticker", "isin"]]

    # CRSP stocknames
    stocknames_table = getattr(config, "WRDS_CRSP_STOCKNAMES", "crsp.stocknames")
    sn_query = f"""
        SELECT
            permno,
            ticker
        FROM {stocknames_table}
    """
    sn = db.raw_sql(sn_query)
    sn["ticker"] = sn["ticker"].astype(str).str.upper().str.strip()
    sn = sn.drop_duplicates(subset=["permno", "ticker"])

    # CCM linktable
    ccm_table = getattr(config, "WRDS_CCM_LINKTABLE", "crsp.ccmxpf_linktable")
    ccm_query = f"""
        SELECT
            gvkey,
            lpermno AS permno,
            linktype,
            linkprim
        FROM {ccm_table}
        WHERE linktype IN ('LC','LU')
          AND linkprim IN ('P','C')
    """
    ccm = db.raw_sql(ccm_query)
    ccm = ccm.dropna(subset=["permno", "gvkey"])

    # Merge BoardEx profile -> CRSP stocknames
    m1 = pd.merge(
        prof,
        sn[["permno", "ticker"]],
        on="ticker",
        how="left",
        validate="m:m",
    )

    # Merge -> CCM
    m2 = pd.merge(
        m1,
        ccm[["permno", "gvkey"]],
        on="permno",
        how="left",
        validate="m:m",
    )

    # Normalize gvkey
    if "gvkey" in m2.columns:
        m2["gvkey"] = m2["gvkey"].astype(str).str.zfill(6)

    # Build final link table
    cols = ["boardid", "ticker", "isin"]
    if "gvkey" in m2.columns:
        cols.append("gvkey")

    link = m2[cols].drop_duplicates().rename(columns={"boardid": "company_id"})
    link["company_id"] = link["company_id"].astype(str)

    if "ticker" in link.columns:
        link["ticker"] = link["ticker"].astype(str).str.upper().str.strip()

    return link

def load_or_build_boardex_data():
    """
    Load BoardEx directors, committees, and gvkey link from local parquet if available.
    If not, fetch from WRDS and cache.
    """
    try:
        directors = pd.read_parquet(config.RAW_BOARDEX_DIRECTORS_PATH)
        committees = pd.read_parquet(config.RAW_BOARDEX_COMMITTEES_PATH)
        link = pd.read_parquet(config.RAW_BOARDEX_LINK_PATH)
        print("Loaded BoardEx data from local parquet.")
        return directors, committees, link
    except FileNotFoundError:
        print("BoardEx local files not found. Fetching from WRDS...")

    db = get_db()

    print("Fetching BoardEx directors...")
    directors = fetch_boardex_directors(db)
    directors.to_parquet(config.RAW_BOARDEX_DIRECTORS_PATH)

    print("Fetching BoardEx committees...")
    committees = fetch_boardex_committees(db)
    committees.to_parquet(config.RAW_BOARDEX_COMMITTEES_PATH)

    print("Fetching BoardEx company profile...")
    profile = fetch_boardex_profile(db)

    print("Building BoardEx–CRSP–CCM link...")
    link = build_boardex_ccm_link(db, profile)
    link.to_parquet(config.RAW_BOARDEX_LINK_PATH)

    print("BoardEx data fetch complete.")
    return directors, committees, link


# ---------------------------------------------------------------------
# Linking spells to BoardEx boards
# ---------------------------------------------------------------------

def prepare_spells(spells: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize CEO spells dataframe:
    - Standardize gvkey column to 'gvkey' (6-char string).
    - Normalize ticker if present.
    """
    spells = spells.copy()

    # Standardize gvkey column
    if "gvkey" not in spells.columns and "GVKEY" in spells.columns:
        spells = spells.rename(columns={"GVKEY": "gvkey"})
    if "gvkey" in spells.columns:
        spells["gvkey"] = spells["gvkey"].astype(str).str.zfill(6)

    # Standardize ticker if available (tic or ticker)
    if "ticker" not in spells.columns and "tic" in spells.columns:
        spells = spells.rename(columns={"tic": "ticker"})
    if "ticker" in spells.columns:
        spells["ticker"] = spells["ticker"].astype(str).str.upper().str.strip()

    # Ensure appointment_date is datetime
    if "appointment_date" in spells.columns:
        spells["appointment_date"] = pd.to_datetime(spells["appointment_date"])

    return spells

def link_firms_to_boardex(spells: pd.DataFrame, link: pd.DataFrame) -> pd.DataFrame:
    """
    Attach BoardEx board/company_id to each CEO spell.

    Preferred linkage:
      spells.gvkey  <->  link.gvkey  ->  link.company_id

    Fallback:
      spells.ticker <->  link.ticker ->  link.company_id
    """
    if spells.empty or link.empty:
        print("Spells or link table is empty in link_firms_to_boardex.")
        return spells

    spells = prepare_spells(spells)
    link = link.copy()

    # 1. Clean link table IDs
    if "company_id" in link.columns:
        link["company_id"] = link["company_id"].astype(str)
        # Drop NA/nan/empty
        link = link[~link["company_id"].isin(["nan", "<NA>", "None", ""])]
        # Clean .0 suffix
        link["company_id"] = link["company_id"].str.replace(r"\.0$", "", regex=True)
    else:
        return spells

    # 2. Clean and filter gvkey
    if "gvkey" in link.columns:
        link["gvkey"] = link["gvkey"].astype(str).str.zfill(6)
        # Drop invalid gvkeys
        link = link[~link["gvkey"].str.contains("nan|NA|None", case=False)]

        # Identify and drop high-cardinality gvkeys (likely data errors)
        # We count unique company_ids per gvkey
        counts = link.groupby("gvkey")["company_id"].nunique()
        bad_gvkeys = counts[counts > 50].index
        if len(bad_gvkeys) > 0:
            print(f"Dropping {len(bad_gvkeys)} gvkeys with >50 linked company_ids to prevent explosion.")
            link = link[~link["gvkey"].isin(bad_gvkeys)]

    # 3. Clean ticker
    if "ticker" in link.columns:
        link["ticker"] = link["ticker"].astype(str).str.upper().str.strip()

    # --- Merge Strategy ---
    
    merged = pd.DataFrame()

    # Try gvkey-based linkage first
    if "gvkey" in spells.columns and "gvkey" in link.columns:
        gvkey_link = link[["company_id", "gvkey"]].drop_duplicates()
        merged = pd.merge(
            spells,
            gvkey_link,
            on="gvkey",
            how="left",
            validate="m:m",
        )
        
    # Fallback: ticker-based linkage for those not matched
    # (Or if we want to try ticker for everything, but usually gvkey is better)
    
    # If merged is empty (no gvkey col), initialize it from spells
    if merged.empty:
        merged = spells.copy()

    # If we still have missing company_id, try ticker
    # But wait, merged has all spells rows (left join).
    # Rows with no match have NaN company_id.
    
    mask_missing = merged["company_id"].isna()
    if mask_missing.any() and "ticker" in spells.columns and "ticker" in link.columns:
        # Extract unmatched spells
        unmatched = merged.loc[mask_missing, spells.columns].copy()
        
        ticker_link = link[["company_id", "ticker"]].drop_duplicates()
        # Drop bad tickers? Maybe later.
        
        matched_ticker = pd.merge(
            unmatched,
            ticker_link,
            on="ticker",
            how="left",
            validate="m:m",
        )
        
        # Update merged with new matches
        # We concatenate the matched parts (dropping the original unmatched rows from merged)
        # Actually, simpler to just combine results.
        
        # Remove unmatched rows from 'merged'
        merged = merged[~mask_missing]
        # Add 'matched_ticker' (which includes still-unmatched ones)
        merged = pd.concat([merged, matched_ticker], ignore_index=True)

    # Deduplicate: Ensure 1 row per spell_id
    if "spell_id" in merged.columns:
        # If duplicates exist, we pick one. 
        # Ideally we'd pick the "best" one, but for now, pick first.
        before_dedup = len(merged)
        merged = merged.drop_duplicates(subset=["spell_id"])
        after_dedup = len(merged)
        if before_dedup > after_dedup:
            print(f"Deduplicated linked spells: {before_dedup} -> {after_dedup}")

    return merged


# ---------------------------------------------------------------------
# Build director roster & committees
# ---------------------------------------------------------------------

def build_board_roster(spells_linked: pd.DataFrame, directors: pd.DataFrame) -> pd.DataFrame:
    """
    Build roster of directors active at the CEO appointment date for each spell.

    - Merge spells (with company_id) to BoardEx directors.
    - Keep directors whose role interval covers appointment_date.
    """
    if "company_id" not in spells_linked.columns:
        print("build_board_roster: spells_linked missing company_id.")
        return pd.DataFrame()
    if "appointment_date" not in spells_linked.columns:
        print("build_board_roster: spells_linked missing appointment_date.")
        return pd.DataFrame()

    spells = spells_linked.copy()
    dirs = directors.copy()

    # Harmonize company_id type
    def clean_id(series):
        # Convert to string, remove trailing .0 if present
        return series.astype(str).str.replace(r"\.0$", "", regex=True)

    spells["company_id"] = clean_id(spells["company_id"])
    dirs["company_id"] = clean_id(dirs["company_id"])

    # Ensure dates
    # Convert to datetime, coercing errors to NaT.
    dirs["date_start"] = pd.to_datetime(dirs["date_start"], errors="coerce")
    dirs["date_end"] = pd.to_datetime(dirs["date_end"], errors="coerce")
    
    # Treat open-ended roles as active through "today"
    dirs["date_end"] = dirs["date_end"].fillna(pd.Timestamp("today").normalize())

    merged = pd.merge(
        spells,
        dirs,
        on="company_id",
        how="inner",
        validate="m:m",
    )

    mask = (
        (merged["appointment_date"] >= merged["date_start"]) &
        (merged["appointment_date"] <= merged["date_end"])
    )
    roster = merged.loc[mask].copy()

    # Keep director_id as string for consistency
    roster["director_id"] = roster["director_id"].astype(str)

    return roster

def flag_search_committee(roster: pd.DataFrame, committees: pd.DataFrame) -> pd.DataFrame:
    """
    Flag whether each director in the roster sits on a nomination/governance committee
    at the appointment date.

    Output column: is_search_committee (bool)
    """
    if roster.empty:
        return roster

    if committees.empty:
        roster = roster.copy()
        roster["is_search_committee"] = False
        return roster

    roster = roster.copy()
    coms = committees.copy()

    # Harmonize keys
    roster["company_id"] = roster["company_id"].astype(str)
    roster["director_id"] = roster["director_id"].astype(str)
    coms["company_id"] = coms["company_id"].astype(str)
    coms["director_id"] = coms["director_id"].astype(str)

    # Ensure datetime
    roster["appointment_date"] = pd.to_datetime(roster["appointment_date"])
    coms["c_date_start"] = pd.to_datetime(coms["c_date_start"], errors="coerce")
    coms["c_date_end"] = pd.to_datetime(coms["c_date_end"], errors="coerce")
    coms["c_date_end"] = coms["c_date_end"].fillna(pd.Timestamp("today").normalize())

    # Filter relevant committees
    patterns = ["Nomination", "Governance", "Nominating", "Nom & Gov"]
    pattern = "|".join(patterns)

    rel_coms = coms[
        coms["committee_name"].str.contains(pattern, case=False, na=False)
    ].copy()

    if rel_coms.empty:
        roster["is_search_committee"] = False
        return roster

    merged = pd.merge(
        roster,
        rel_coms[
            ["company_id", "director_id", "committee_name", "c_date_start", "c_date_end"]
        ],
        on=["company_id", "director_id"],
        how="left",
        validate="m:m",
    )

    # Overlap condition
    is_member = (
        (merged["appointment_date"] >= merged["c_date_start"]) &
        (merged["appointment_date"] <= merged["c_date_end"])
    )
    merged["is_search_committee"] = is_member

    # Collapse to one row per (spell_id, director_id)
    if "spell_id" not in merged.columns:
        raise KeyError("flag_search_committee: roster must contain 'spell_id'.")

    flags = (
        merged.groupby(["spell_id", "director_id"])["is_search_committee"]
        .any()
        .reset_index()
    )

    roster = pd.merge(
        roster,
        flags,
        on=["spell_id", "director_id"],
        how="left",
        validate="m:1",
    )
    roster["is_search_committee"] = roster["is_search_committee"].fillna(False)

    return roster

def compute_director_characteristics(roster: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple director characteristics at appointment date:

    - tenure_years: (appointment_date - date_start on this board)
    - n_boards: placeholder = 1 (concurrent board network can be added later)
    """
    if roster.empty:
        return roster

    roster = roster.copy()
    roster["appointment_date"] = pd.to_datetime(roster["appointment_date"])
    roster["date_start"] = pd.to_datetime(roster["date_start"])

    roster["tenure_days"] = (roster["appointment_date"] - roster["date_start"]).dt.days
    roster["tenure_years"] = roster["tenure_days"] / 365.25

    # Placeholder: actual network computation would count concurrent board seats
    roster["n_boards"] = 1

    return roster


# ---------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------

def run_phase3():
    print("Starting Phase 3: Director Selection (BoardEx)...")

    # 1. Load CEO spells
    try:
        spells = pd.read_parquet(config.CEO_SPELLS_PATH)
    except FileNotFoundError:
        print(f"CEO spells file not found at {config.CEO_SPELLS_PATH}.")
        return

    # 2. Load BoardEx data (cached or WRDS)
    directors, committees, link = load_or_build_boardex_data()
    if directors.empty:
        print("BoardEx directors table is empty. Aborting Phase 3.")
        return

    # 3. Link spells to BoardEx boards
    spells_linked = link_firms_to_boardex(spells, link)
    
    if "company_id" not in spells_linked.columns:
        print("Failed to attach BoardEx company_id to spells. Aborting Phase 3.")
        return

    # 4. Build board roster at appointment date
    roster = build_board_roster(spells_linked, directors)
    if roster.empty:
        print("No matching director rosters found (date or linkage mismatch).")
        return

    # 5. Flag search committee members
    roster = flag_search_committee(roster, committees)

    # 6. Compute characteristics
    roster = compute_director_characteristics(roster)

    # 7. Prepare final linkage output
    roster = roster.rename(columns={"director_id": "directorid"})
    cols = [
        "spell_id",
        "directorid",
        "gvkey",
        "company_id",
        "appointment_date",
        "is_search_committee",
        "tenure_years",
        "n_boards",
    ]
    cols = [c for c in cols if c in roster.columns]
    linkage = roster[cols].copy()

    print(f"Phase 3 complete. Linked {len(linkage):,} director–spell observations.")
    linkage.to_parquet(config.DIRECTOR_LINKAGE_PATH)
    return linkage


if __name__ == "__main__":
    run_phase3()
