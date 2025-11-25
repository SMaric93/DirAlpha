"""
Phase 3: Director Selection (BoardEx)

Goal:
- Link CEO spells (Compustat/CRSP) to BoardEx boards using the WRDS
  BoardEx–CRSP–Compustat company link file.
- Construct a director–spell roster with:
    * Board members at CEO appointment date
    * Search / nomination committee flags
    * Basic director characteristics (e.g., tenure)
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path

from . import config, db, io, transform, log

logger = log.logger

# ---------------------------------------------------------------------
# Helpers: CEO spells
# ---------------------------------------------------------------------

def prepare_spells(spells: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize CEO spells dataframe: standardize keys and dates.
    Expected fields (if present):
        - spell_id
        - gvkey
        - ticker
        - permco (optional)
        - appointment_date
    """
    spells = spells.copy()

    if "gvkey" in spells.columns:
        spells["gvkey"] = transform.normalize_gvkey(spells["gvkey"])

    if "ticker" in spells.columns:
        spells["ticker"] = transform.normalize_ticker(spells["ticker"])

    if "permco" in spells.columns:
        # Ensure numeric permco, but keep as object to allow merge
        spells["permco"] = pd.to_numeric(spells["permco"], errors="coerce")

    if "appointment_date" in spells.columns:
        spells["appointment_date"] = pd.to_datetime(
            spells["appointment_date"],
            errors="coerce",
        )

    return spells


# ---------------------------------------------------------------------
# Helpers: WRDS BoardEx–CRSP–Compustat link
# ---------------------------------------------------------------------


def _normalize_boardex_link_columns(link: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names from WRDS BoardEx–CRSP–Compustat link file.
    """
    link = link.copy()
    # Map lower-case name -> actual name
    lower_to_orig = {c.lower(): c for c in link.columns}

    # CompanyID -> company_id
    if "companyid" in lower_to_orig and "company_id" not in link.columns:
        link = link.rename(columns={lower_to_orig["companyid"]: "company_id"})

    # GVKAY -> gvkey
    if "gvkay" in lower_to_orig and "gvkey" not in link.columns:
        link = link.rename(columns={lower_to_orig["gvkay"]: "gvkey"})
    elif "gvkey" in lower_to_orig and "gvkey" not in link.columns:
        link = link.rename(columns={lower_to_orig["gvkey"]: "gvkey"})

    # PERMCO -> permco
    if "permco" in lower_to_orig and "permco" not in link.columns:
        link = link.rename(columns={lower_to_orig["permco"]: "permco"})

    # SCORE -> score
    if "score" in lower_to_orig and "score" not in link.columns:
        link = link.rename(columns={lower_to_orig["score"]: "score"})

    # Preferred / prefered -> preferred
    for cand in ["preferred", "prefered"]:
        if cand in lower_to_orig and "preferred" not in link.columns:
            link = link.rename(columns={lower_to_orig[cand]: "preferred"})

    # Duplicate / duplicated -> duplicate
    for cand in ["duplicate", "duplicated"]:
        if cand in lower_to_orig and "duplicate" not in link.columns:
            link = link.rename(columns={lower_to_orig[cand]: "duplicate"})

    return link


def _clean_link_table(link: pd.DataFrame) -> pd.DataFrame:
    """
    Clean BoardEx link table IDs and gvkeys using WRDS semantics.
    """
    link = _normalize_boardex_link_columns(link)

    if "company_id" not in link.columns:
        logger.error(
            "BoardEx link table missing 'company_id' column after normalization."
        )
        return pd.DataFrame()

    # Clean company_id
    link["company_id"] = transform.clean_id(link["company_id"])
    bad_company_ids = {"", "nan", "na", "NA", "None", "<NA>"}

    mask_valid_company = (
        link["company_id"].notna()
        & (~link["company_id"].astype(str).str.strip().isin(bad_company_ids))
    )
    before = len(link)
    link = link.loc[mask_valid_company].copy()
    logger.info(
        f"Cleaned company_id in BoardEx link: {before:,} -> {len(link):,} rows."
    )

    # Normalize gvkey (from GVKAY)
    if "gvkey" in link.columns:
        link["gvkey"] = transform.normalize_gvkey(link["gvkey"])
        bad_gvkeys = {"", "nan", "na", "NA", "None", "<NA>"}
        mask_valid_gvkey = (
            link["gvkey"].notna()
            & (~link["gvkey"].astype(str).str.strip().isin(bad_gvkeys))
        )
        before_gvkey = len(link)
        link = link.loc[mask_valid_gvkey].copy()
        logger.info(
            f"Cleaned gvkey (GVKAY) in BoardEx link: {before_gvkey:,} -> {len(link):,} rows."
        )

    # Normalize permco if present
    if "permco" in link.columns:
        link["permco"] = pd.to_numeric(link["permco"], errors="coerce")

    # Apply WRDS "preferred" flag if available
    if "preferred" in link.columns:
        before_pref = len(link)
        link = link.loc[link["preferred"] == 1].copy()
        logger.info(
            f"Applied 'preferred == 1' filter: {before_pref:,} -> {len(link):,} rows."
        )
    else:
        logger.warning(
            "No 'preferred' column in BoardEx link; using all rows (may include non-preferred links)."
        )

    return link


def _collapse_link(link: pd.DataFrame, key: str) -> pd.DataFrame:
    """
    Collapse link table to unique mapping: key -> company_id.
    """
    if key not in link.columns:
        return pd.DataFrame(columns=[key, "company_id"])

    df = link.dropna(subset=[key, "company_id"]).copy()
    if df.empty:
        return pd.DataFrame(columns=[key, "company_id"])

    if "score" in df.columns:
        # Choose row with minimum score per key
        idx = df.groupby(key)["score"].idxmin()
        collapsed = df.loc[idx, [key, "company_id", "score"]].reset_index(drop=True)
    else:
        # Fallback: modal company_id per key
        collapsed = (
            df.groupby(key)["company_id"]
            .agg(lambda s: s.value_counts().index[0])
            .reset_index()
        )

    return collapsed


def link_firms_to_boardex(spells: pd.DataFrame, link: pd.DataFrame) -> pd.DataFrame:
    """
    Attach BoardEx company_id to each CEO spell using the WRDS BoardEx–CRSP–Compustat link.
    """
    if spells.empty or link.empty:
        logger.warning(
            "Spells or BoardEx link table is empty in link_firms_to_boardex."
        )
        return spells

    spells = prepare_spells(spells)
    link = _clean_link_table(link)

    if link.empty:
        logger.error(
            "Cleaned BoardEx link table is empty. Cannot link spells to company_id."
        )
        return spells

    merged = spells.copy()

    # --- gvkey-based linkage (preferred WRDS approach) ---
    if "gvkey" in spells.columns and "gvkey" in link.columns:
        gvkey_map = _collapse_link(link, "gvkey")
        logger.info(
            f"GVKEY (GVKAY) link table: {len(gvkey_map):,} unique gvkeys mapped to company_id."
        )

        merged = pd.merge(
            spells,
            gvkey_map[["gvkey", "company_id"]],
            on="gvkey",
            how="left",
            validate="m:1",
        )
    else:
        if "gvkey" not in spells.columns:
            logger.warning(
                "Spells missing 'gvkey' column; cannot use GVKAY-based linkage."
            )
        if "gvkey" not in link.columns:
            logger.warning(
                "BoardEx link missing 'gvkey' (GVKAY) column after normalization."
            )
        merged["company_id"] = np.nan

    # --- permco-based fallback ---
    if merged["company_id"].isna().any() and "permco" in spells.columns and "permco" in link.columns:
        permco_map = _collapse_link(link, "permco")
        logger.info(
            f"PERMCO link table: {len(permco_map):,} unique permcos mapped to company_id."
        )

        mask_missing = merged["company_id"].isna()
        unmatched = merged.loc[mask_missing, ["permco"]].copy()

        unmatched = pd.merge(
            unmatched,
            permco_map[["permco", "company_id"]],
            on="permco",
            how="left",
            validate="m:1",
        )

        merged.loc[mask_missing, "company_id"] = unmatched["company_id"].values

    # --- ticker-based fallback (only if link actually has tickers) ---
    if (
        merged["company_id"].isna().any()
        and "ticker" in spells.columns
        and "ticker" in link.columns
    ):
        ticker_map = _collapse_link(link, "ticker")
        logger.info(
            f"Ticker link table: {len(ticker_map):,} unique tickers mapped to company_id."
        )

        mask_missing = merged["company_id"].isna()
        unmatched = merged.loc[mask_missing, ["ticker"]].copy()

        unmatched = pd.merge(
            unmatched,
            ticker_map[["ticker", "company_id"]],
            on="ticker",
            how="left",
            validate="m:1",
        )

        merged.loc[mask_missing, "company_id"] = unmatched["company_id"].values

    # Deduplicate: ensure 1 row per spell_id (keep earliest appointment_date)
    if "spell_id" in merged.columns:
        before_dedup = len(merged)
        merged = (
            merged
            .sort_values(["spell_id", "appointment_date"])
            .drop_duplicates(subset=["spell_id"], keep="first")
        )
        after_dedup = len(merged)
        if before_dedup > after_dedup:
            logger.info(
                f"Deduplicated linked spells: {before_dedup:,} -> {after_dedup:,}"
            )

    missing_company = merged["company_id"].isna().sum()
    logger.info(
        f"After BoardEx linkage: {len(merged):,} spells, "
        f"{missing_company:,} without company_id."
    )

    return merged


# ---------------------------------------------------------------------
# Board roster & committees
# ---------------------------------------------------------------------


def build_board_roster(
    spells_linked: pd.DataFrame,
    directors: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build roster of directors active at the CEO appointment date.
    """
    required_spell_cols = {"company_id", "appointment_date"}
    if not required_spell_cols.issubset(spells_linked.columns):
        missing = required_spell_cols - set(spells_linked.columns)
        logger.error(
            f"spells_linked missing required columns {missing} in build_board_roster."
        )
        return pd.DataFrame()

    if directors.empty:
        logger.error("Directors table is empty in build_board_roster.")
        return pd.DataFrame()

    spells = spells_linked.copy()
    dirs = directors.copy()

    # Harmonize company_id
    spells["company_id"] = transform.clean_id(spells["company_id"])
    dirs["company_id"] = transform.clean_id(dirs["company_id"])

    # Ensure director_id
    if "director_id" not in dirs.columns:
        logger.error(
            "Directors table missing 'director_id' in build_board_roster."
        )
        return pd.DataFrame()
    dirs["director_id"] = transform.clean_id(dirs["director_id"])

    # Ensure dates
    dirs["date_start"] = pd.to_datetime(dirs.get("date_start"), errors="coerce")
    dirs["date_end"] = pd.to_datetime(dirs.get("date_end"), errors="coerce")
    dirs["date_end"] = dirs["date_end"].fillna(pd.Timestamp("today").normalize())

    before_dir_filter = len(dirs)
    dirs = dirs.dropna(subset=["director_id", "company_id", "date_start"])
    logger.info(
        f"Directors cleaned: {before_dir_filter:,} -> {len(dirs):,} "
        "after dropping missing director_id/company_id/date_start."
    )

    spells["appointment_date"] = pd.to_datetime(
        spells["appointment_date"],
        errors="coerce",
    )

    merged = pd.merge(
        spells,
        dirs,
        on="company_id",
        how="inner",
        validate="m:m",
    )
    logger.info(
        f"After merging spells with directors on company_id: {len(merged):,} rows."
    )

    # Director must be on the board on appointment_date
    mask = (
        (merged["appointment_date"].notna())
        & (merged["appointment_date"] >= merged["date_start"])
        & (merged["appointment_date"] <= merged["date_end"])
    )
    roster = merged.loc[mask].copy()

    logger.info(
        f"Board roster: {len(roster):,} director–spell rows after date overlap "
        f"filter (from {len(merged):,})."
    )

    if roster.empty:
        logger.warning("Board roster is empty after date filtering.")
        return roster

    roster["director_id"] = transform.clean_id(roster["director_id"])
    return roster


def flag_search_committee(
    roster: pd.DataFrame,
    committees: pd.DataFrame,
) -> pd.DataFrame:
    """
    Flag search / nomination / governance committee membership.
    """
    if roster.empty:
        roster = roster.copy()
        roster["is_search_committee"] = False
        return roster

    roster = roster.copy()

    if committees.empty:
        roster["is_search_committee"] = False
        return roster

    coms = committees.copy()

    # Harmonize keys
    roster["company_id"] = transform.clean_id(roster["company_id"])
    roster["director_id"] = transform.clean_id(roster["director_id"])
    coms["company_id"] = transform.clean_id(coms.get("company_id"))
    coms["director_id"] = transform.clean_id(coms.get("director_id"))

    # Ensure dates
    roster["appointment_date"] = pd.to_datetime(
        roster["appointment_date"],
        errors="coerce",
    )
    coms["c_date_start"] = pd.to_datetime(
        coms.get("c_date_start"),
        errors="coerce",
    )
    coms["c_date_end"] = pd.to_datetime(
        coms.get("c_date_end"),
        errors="coerce",
    )
    # Treat missing start as very early and missing end as "still serving"
    coms["c_date_start"] = coms["c_date_start"].fillna(pd.Timestamp("1900-01-01"))
    coms["c_date_end"] = coms["c_date_end"].fillna(pd.Timestamp("today").normalize())

    # Filter relevant committees
    if "committee_name" not in coms.columns:
        logger.warning(
            "Committees table missing 'committee_name'; cannot flag search committees."
        )
        roster["is_search_committee"] = False
        return roster

    patterns = [
        "Nomination",
        "Nominating",
        "Nom & Gov",
        "Governance",
        "Nom/Gov",
    ]
    pattern = "|".join(patterns)

    rel_coms = coms[
        coms["committee_name"].str.contains(pattern, case=False, na=False)
    ].copy()

    logger.info(
        f"Relevant nomination/governance committees: {len(rel_coms):,} rows."
    )

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
        (merged["appointment_date"] >= merged["c_date_start"])
        & (merged["appointment_date"] <= merged["c_date_end"])
    )
    merged["is_search_committee"] = is_member.fillna(False)

    if "spell_id" not in merged.columns:
        logger.warning(
            "'spell_id' not found when flagging search committee; "
            "returning row-level flags."
        )
        return merged

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


def compute_director_characteristics(
    roster: pd.DataFrame,
    directors: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute simple director characteristics relative to the CEO appointment:
        - tenure_years
        - n_boards (count of concurrent board seats)
    """
    if roster.empty:
        return roster

    roster = roster.copy()
    roster["appointment_date"] = pd.to_datetime(
        roster["appointment_date"],
        errors="coerce",
    )
    roster["date_start"] = pd.to_datetime(
        roster.get("date_start"),
        errors="coerce",
    )

    # Tenure
    roster["tenure_days"] = (
        roster["appointment_date"] - roster["date_start"]
    ).dt.days
    roster["tenure_years"] = roster["tenure_days"].div(365.25).clip(lower=0)

    # n_boards calculation
    unique_pairs = roster[["director_id", "appointment_date"]].drop_duplicates()
    
    if directors.empty:
        roster["n_boards"] = 1
        return roster

    directors = directors.copy()
    directors["director_id"] = transform.clean_id(directors["director_id"])
    directors["company_id"] = transform.clean_id(directors["company_id"])

    relevant_ids = unique_pairs["director_id"].unique()
    rel_dirs = directors[directors["director_id"].isin(relevant_ids)].copy()
    
    rel_dirs["date_start"] = pd.to_datetime(rel_dirs.get("date_start"), errors="coerce")
    rel_dirs["date_end"] = pd.to_datetime(rel_dirs.get("date_end"), errors="coerce")
    rel_dirs["date_end"] = rel_dirs["date_end"].fillna(pd.Timestamp("today").normalize())
    
    merged = pd.merge(
        unique_pairs,
        rel_dirs[["director_id", "company_id", "date_start", "date_end"]],
        on="director_id",
        how="inner"
    )
    
    active_roles = merged[
        (merged["appointment_date"] >= merged["date_start"]) &
        (merged["appointment_date"] <= merged["date_end"])
    ]
    
    counts = (
        active_roles.groupby(["director_id", "appointment_date"])["company_id"]
        .nunique()
        .reset_index(name="n_boards")
    )
    
    roster = pd.merge(
        roster,
        counts,
        on=["director_id", "appointment_date"],
        how="left"
    )
    roster["n_boards"] = roster["n_boards"].fillna(1).astype(int)

    return roster


# ---------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------


def run_phase3() -> pd.DataFrame | None:
    """
    Top-level entry point for Phase 3.
    """
    logger.info("Starting Phase 3: Director Selection (BoardEx)...")

    # 1. Load CEO spells
    if not config.CEO_SPELLS_PATH.exists():
        logger.error(f"CEO spells file not found at {config.CEO_SPELLS_PATH}.")
        return None

    spells = pd.read_parquet(config.CEO_SPELLS_PATH)
    logger.info(f"Loaded CEO spells: {len(spells):,} rows.")

    # 2. Load BoardEx data
    directors = io.load_or_fetch(
        config.RAW_BOARDEX_DIRECTORS_PATH,
        db.fetch_boardex_directors,
    )
    committees = io.load_or_fetch(
        config.RAW_BOARDEX_COMMITTEES_PATH,
        db.fetch_boardex_committees,
    )
    link = io.load_or_fetch(
        config.RAW_BOARDEX_LINK_PATH,
        db.fetch_boardex_link,
    )

    logger.info(
        f"BoardEx directors: {len(directors):,} rows; "
        f"committees: {len(committees):,} rows; "
        f"link table: {len(link):,} rows."
    )

    if directors.empty:
        logger.error("BoardEx directors table is empty. Aborting Phase 3.")
        return None

    # 3. Link spells to BoardEx boards
    spells_linked = link_firms_to_boardex(spells, link)

    if "company_id" not in spells_linked.columns:
        logger.error("Failed to attach BoardEx company_id to spells.")
        return None

    # 4. Build board roster
    roster = build_board_roster(spells_linked, directors)
    if roster.empty:
        logger.warning("No matching director rosters found in Phase 3.")
        return None

    # 5. Flag search committee
    roster = flag_search_committee(roster, committees)

    # 6. Compute characteristics
    roster = compute_director_characteristics(roster, directors)

    # 7. Save linkage
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

    out_path: Path = config.DIRECTOR_LINKAGE_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    linkage.to_parquet(out_path)

    csv_path = config.DIRECTOR_LINKAGE_CSV_PATH
    linkage.to_csv(csv_path, index=False)

    logger.info(
        f"Phase 3 complete. Linked {len(linkage):,} director–spell observations "
        f"to {out_path} and {csv_path}."
    )

    # ---------------------------------------------------------------------
    # Part B: Enrich Spells with Announcement Dates (for Event Study)
    # ---------------------------------------------------------------------
    enrich_spells_with_announcements(spells, spells_linked)

    return linkage

def enrich_spells_with_announcements(spells: pd.DataFrame, spells_linked: pd.DataFrame):
    """
    Link CEO spells to BoardEx Director IDs and fetch announcement dates.
    Saves to config.CEO_SPELLS_BOARDEX_PATH.
    """
    logger.info("Starting enrichment of CEO spells with BoardEx announcement dates...")

    # 1. Fetch Link Table (Execucomp -> BoardEx)
    exec_bdx_link = io.load_or_fetch(
        config.RAW_EXEC_BOARDEX_LINK_PATH,
        db.fetch_exec_boardex_link,
    )
    
    if exec_bdx_link.empty:
        logger.error("Execucomp-BoardEx link table is empty. Cannot enrich spells.")
        return

    # 2. Fetch Announcements
    announcements = io.load_or_fetch(
        config.RAW_BOARDEX_ANNOUNCEMENTS_PATH,
        db.fetch_boardex_announcements,
    )
    
    if announcements.empty:
        logger.error("BoardEx announcements table is empty. Cannot enrich spells.")
        return

    # 3. Link Spells to Director ID
    # spells has 'execid', exec_bdx_link maps 'execid' -> 'director_id'
    
    # Ensure types
    spells["execid"] = pd.to_numeric(spells["execid"], errors="coerce")
    exec_bdx_link["execid"] = pd.to_numeric(exec_bdx_link["execid"], errors="coerce")
    exec_bdx_link["director_id"] = transform.clean_id(exec_bdx_link["director_id"])
    
    # Filter for high quality matches if possible (score is often available)
    # For now, we take the best match (highest score) if duplicates exist
    if "score" in exec_bdx_link.columns:
        # Lower score is better (1=High Confidence, 12=Low)
        exec_bdx_link = exec_bdx_link.sort_values("score", ascending=True).drop_duplicates("execid")
    else:
        exec_bdx_link = exec_bdx_link.drop_duplicates("execid")

    spells_w_dir = pd.merge(
        spells,
        exec_bdx_link[["execid", "director_id"]],
        on="execid",
        how="left",
        validate="m:1"
    )
    
    linked_count = spells_w_dir["director_id"].notna().sum()
    logger.info(f"Linked {linked_count:,} / {len(spells):,} spells to BoardEx Director IDs.")

    # 4. Merge with Announcements
    # We need to match on director_id AND company_id to ensure it's the right role
    # However, spells_linked has the BoardEx company_id.
    
    # Join spells_w_dir with spells_linked to get company_id
    # spells_linked has ['spell_id', 'company_id'] (and others)
    
    spells_full = pd.merge(
        spells_w_dir,
        spells_linked[["spell_id", "company_id"]],
        on="spell_id",
        how="left"
    )
    
    # Prepare announcements
    announcements["director_id"] = transform.clean_id(announcements["director_id"])
    announcements["company_id"] = transform.clean_id(announcements["company_id"])
    announcements["announcement_date"] = pd.to_datetime(announcements["announcement_date"], errors="coerce")
    
    # Merge
    merged = pd.merge(
        spells_full,
        announcements,
        on=["director_id", "company_id"],
        how="left"
    )
    
    # 5. Select Best Announcement Date
    # Logic: 
    # - announcement_date should be close to appointment_date (becameceo)
    # - ideally before or slightly after
    # - If multiple, pick the one closest to appointment_date
    
    merged["appointment_date"] = pd.to_datetime(merged["appointment_date"])
    merged["diff_days"] = (merged["appointment_date"] - merged["announcement_date"]).dt.days
    
    # Filter: Announcement shouldn't be too far off (e.g., +/- 2 years) to avoid matching wrong roles
    # CEO roles are filtered in SQL, but double check
    mask_valid = merged["diff_days"].abs() < 730 # 2 years
    
    valid_announcements = merged[mask_valid].copy()
    
    # Sort by absolute difference to find closest
    valid_announcements["abs_diff"] = valid_announcements["diff_days"].abs()
    valid_announcements = valid_announcements.sort_values(["spell_id", "abs_diff"])
    
    best_dates = valid_announcements.drop_duplicates("spell_id", keep="first")
    
    # 6. Finalize
    # Merge back to original spells to keep all rows
    # We also want director_id from the linkage step
    spells_with_did = spells_w_dir[['spell_id', 'director_id']]
    
    final_spells = pd.merge(
        spells,
        spells_with_did,
        on='spell_id',
        how='left'
    )

    final_spells = pd.merge(
        final_spells,
        best_dates[["spell_id", "announcement_date"]],
        on="spell_id",
        how="left"
    )
    
    # Stats
    found_ann = final_spells["announcement_date"].notna().sum()
    logger.info(f"Found BoardEx announcement dates for {found_ann:,} / {len(final_spells):,} spells.")
    
    # Save
    out_path = config.CEO_SPELLS_BOARDEX_PATH
    final_spells.to_parquet(out_path)
    logger.info(f"Saved enriched spells to {out_path}")


if __name__ == "__main__":
    run_phase3()