"""
Phase 3: Director Selection (BoardEx)
"""

import pandas as pd
import numpy as np
from . import config, utils

# ---------------------------------------------------------------------
# Linking spells to BoardEx boards
# ---------------------------------------------------------------------

def prepare_spells(spells: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize CEO spells dataframe.
    """
    spells = spells.copy()

    # Standardize keys
    if "gvkey" in spells.columns:
        spells["gvkey"] = utils.normalize_gvkey(spells["gvkey"])

    if "ticker" in spells.columns:
        spells["ticker"] = utils.normalize_ticker(spells["ticker"])

    if "appointment_date" in spells.columns:
        spells["appointment_date"] = pd.to_datetime(spells["appointment_date"])

    return spells

def link_firms_to_boardex(spells: pd.DataFrame, link: pd.DataFrame) -> pd.DataFrame:
    """
    Attach BoardEx board/company_id to each CEO spell.
    """
    if spells.empty or link.empty:
        utils.logger.warning("Spells or link table is empty.")
        return spells

    spells = prepare_spells(spells)
    link = link.copy()

    # 1. Clean link table IDs
    if "company_id" in link.columns:
        link["company_id"] = utils.clean_id(link["company_id"])
        # Drop NA/nan/empty
        link = link[~link["company_id"].isin(["nan", "<NA>", "None", ""])]
    else:
        return spells

    # 2. Clean and filter gvkey
    if "gvkey" in link.columns:
        link["gvkey"] = utils.normalize_gvkey(link["gvkey"])
        # Drop invalid gvkeys
        link = link[~link["gvkey"].str.contains("nan|NA|None", case=False)]

        # OOM Protection: Drop high-cardinality gvkeys
        counts = link.groupby("gvkey")["company_id"].nunique()
        bad_gvkeys = counts[counts > 50].index
        if len(bad_gvkeys) > 0:
            utils.logger.warning(f"Dropping {len(bad_gvkeys)} gvkeys with >50 linked company_ids.")
            link = link[~link["gvkey"].isin(bad_gvkeys)]

    # 3. Clean ticker
    if "ticker" in link.columns:
        link["ticker"] = utils.normalize_ticker(link["ticker"])

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
        
    # Fallback: ticker-based linkage
    if merged.empty:
        merged = spells.copy()

    mask_missing = merged["company_id"].isna()
    if mask_missing.any() and "ticker" in spells.columns and "ticker" in link.columns:
        unmatched = merged.loc[mask_missing, spells.columns].copy()
        ticker_link = link[["company_id", "ticker"]].drop_duplicates()
        matched_ticker = pd.merge(
            unmatched,
            ticker_link,
            on="ticker",
            how="left",
            validate="m:m",
        )
        merged = merged[~mask_missing]
        merged = pd.concat([merged, matched_ticker], ignore_index=True)

    # Deduplicate: Ensure 1 row per spell_id (pick first)
    if "spell_id" in merged.columns:
        before_dedup = len(merged)
        merged = merged.drop_duplicates(subset=["spell_id"])
        after_dedup = len(merged)
        if before_dedup > after_dedup:
            utils.logger.info(f"Deduplicated linked spells: {before_dedup} -> {after_dedup}")

    return merged

# ---------------------------------------------------------------------
# Build director roster & committees
# ---------------------------------------------------------------------

def build_board_roster(spells_linked: pd.DataFrame, directors: pd.DataFrame) -> pd.DataFrame:
    """
    Build roster of directors active at the CEO appointment date.
    """
    if "company_id" not in spells_linked.columns or "appointment_date" not in spells_linked.columns:
        return pd.DataFrame()

    spells = spells_linked.copy()
    dirs = directors.copy()

    # Harmonize company_id
    spells["company_id"] = utils.clean_id(spells["company_id"])
    dirs["company_id"] = utils.clean_id(dirs["company_id"])

    # Ensure dates
    dirs["date_start"] = pd.to_datetime(dirs["date_start"], errors="coerce")
    dirs["date_end"] = pd.to_datetime(dirs["date_end"], errors="coerce")
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
    roster["director_id"] = utils.clean_id(roster["director_id"])

    return roster

def flag_search_committee(roster: pd.DataFrame, committees: pd.DataFrame) -> pd.DataFrame:
    """
    Flag search committee membership.
    """
    if roster.empty or committees.empty:
        roster = roster.copy()
        roster["is_search_committee"] = False
        return roster

    roster = roster.copy()
    coms = committees.copy()

    # Harmonize keys
    roster["company_id"] = utils.clean_id(roster["company_id"])
    roster["director_id"] = utils.clean_id(roster["director_id"])
    coms["company_id"] = utils.clean_id(coms["company_id"])
    coms["director_id"] = utils.clean_id(coms["director_id"])

    # Ensure dates
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
        rel_coms[["company_id", "director_id", "c_date_start", "c_date_end"]],
        on=["company_id", "director_id"],
        how="left",
        validate="m:m",
    )

    is_member = (
        (merged["appointment_date"] >= merged["c_date_start"]) &
        (merged["appointment_date"] <= merged["c_date_end"])
    )
    merged["is_search_committee"] = is_member

    if "spell_id" not in merged.columns:
        return roster

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
    Compute tenure and n_boards.
    """
    if roster.empty:
        return roster

    roster = roster.copy()
    roster["appointment_date"] = pd.to_datetime(roster["appointment_date"])
    roster["date_start"] = pd.to_datetime(roster["date_start"])

    roster["tenure_days"] = (roster["appointment_date"] - roster["date_start"]).dt.days
    roster["tenure_years"] = roster["tenure_days"] / 365.25
    roster["n_boards"] = 1

    return roster

def run_phase3():
    utils.logger.info("Starting Phase 3: Director Selection (BoardEx)...")

    # 1. Load CEO spells
    if not config.CEO_SPELLS_PATH.exists():
        utils.logger.error("CEO spells file not found.")
        return
    spells = pd.read_parquet(config.CEO_SPELLS_PATH)

    # 2. Load BoardEx data
    directors = utils.load_or_fetch(
        config.RAW_BOARDEX_DIRECTORS_PATH,
        utils.fetch_boardex_directors
    )
    committees = utils.load_or_fetch(
        config.RAW_BOARDEX_COMMITTEES_PATH,
        utils.fetch_boardex_committees
    )
    link = utils.load_or_fetch(
        config.RAW_BOARDEX_LINK_PATH,
        utils.fetch_boardex_link
    )

    if directors.empty:
        utils.logger.error("BoardEx directors table is empty. Aborting Phase 3.")
        return

    # 3. Link spells to BoardEx boards
    spells_linked = link_firms_to_boardex(spells, link)
    
    if "company_id" not in spells_linked.columns:
        utils.logger.error("Failed to attach BoardEx company_id to spells.")
        return

    # 4. Build board roster
    roster = build_board_roster(spells_linked, directors)
    if roster.empty:
        utils.logger.warning("No matching director rosters found.")
        return

    # 5. Flag search committee
    roster = flag_search_committee(roster, committees)

    # 6. Compute characteristics
    roster = compute_director_characteristics(roster)

    # 7. Save
    roster = roster.rename(columns={"director_id": "directorid"})
    cols = [
        "spell_id", "directorid", "gvkey", "company_id", 
        "appointment_date", "is_search_committee", "tenure_years", "n_boards"
    ]
    cols = [c for c in cols if c in roster.columns]
    linkage = roster[cols].copy()

    utils.logger.info(f"Phase 3 complete. Linked {len(linkage):,} director–spell observations.")
    linkage.to_parquet(config.DIRECTOR_LINKAGE_PATH)
    return linkage

if __name__ == "__main__":
    run_phase3()