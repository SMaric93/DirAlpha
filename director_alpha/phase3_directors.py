"""
Phase 3: Director Selection (BoardEx)

Goal:
- Link CEO spells (Compustat/CRSP) to BoardEx boards using the WRDS
  BoardEx–CRSP–Compustat company link file.
- Construct a director–spell roster with:
    * Board members at CEO appointment date
    * Search / nomination committee flags
    * Basic director characteristics (e.g., tenure)

Assumes:
- `config` provides paths:
    * CEO_SPELLS_PATH
    * RAW_BOARDEX_DIRECTORS_PATH
    * RAW_BOARDEX_COMMITTEES_PATH
    * RAW_BOARDEX_LINK_PATH
    * DIRECTOR_LINKAGE_PATH
- `utils` provides:
    * logger
    * normalize_gvkey(series)
    * normalize_ticker(series)
    * clean_id(series)
    * load_or_fetch(path, fetch_fn)
    * fetch_boardex_directors()
    * fetch_boardex_committees()
    * fetch_boardex_link()
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path

from . import config, utils


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
        # pyrefly: ignore [bad-argument-type]
        spells["gvkey"] = utils.normalize_gvkey(spells["gvkey"])

    if "ticker" in spells.columns:
        # pyrefly: ignore [bad-argument-type]
        spells["ticker"] = utils.normalize_ticker(spells["ticker"])

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
    Normalize column names from WRDS BoardEx–CRSP–Compustat link file
    to a canonical schema:
        - gvkey      (from GVKAY)
        - company_id (from COMPANYID)
        - permco
        - score
        - preferred
        - duplicate

    The WRDS manual for BoardEx–CRSP–Compustat link uses e.g.:
        PERMCO, GVKAY, COMPANYID, SCORE, Preferred, Duplicate
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

    Steps:
        1. Normalize column names.
        2. Clean company_id (drop empty / NA / placeholder).
        3. Normalize gvkey (from GVKAY) using utils.normalize_gvkey.
        4. If 'preferred' is present, keep only preferred == 1 matches.
    """
    link = _normalize_boardex_link_columns(link)

    if "company_id" not in link.columns:
        utils.logger.error(
            "BoardEx link table missing 'company_id' column after normalization."
        )
        return pd.DataFrame()

    # Clean company_id
    # pyrefly: ignore [bad-argument-type]
    link["company_id"] = utils.clean_id(link["company_id"])
    bad_company_ids = {"", "nan", "na", "NA", "None", "<NA>"}

    mask_valid_company = (
        link["company_id"].notna()
        & (~link["company_id"].astype(str).str.strip().isin(bad_company_ids))
    )
    before = len(link)
    link = link.loc[mask_valid_company].copy()
    utils.logger.info(
        f"Cleaned company_id in BoardEx link: {before:,} -> {len(link):,} rows."
    )

    # Normalize gvkey (from GVKAY)
    if "gvkey" in link.columns:
        link["gvkey"] = utils.normalize_gvkey(link["gvkey"])
        bad_gvkeys = {"", "nan", "na", "NA", "None", "<NA>"}
        mask_valid_gvkey = (
            link["gvkey"].notna()
            & (~link["gvkey"].astype(str).str.strip().isin(bad_gvkeys))
        )
        before_gvkey = len(link)
        link = link.loc[mask_valid_gvkey].copy()
        utils.logger.info(
            f"Cleaned gvkey (GVKAY) in BoardEx link: {before_gvkey:,} -> {len(link):,} rows."
        )

    # Normalize permco if present
    if "permco" in link.columns:
        link["permco"] = pd.to_numeric(link["permco"], errors="coerce")

    # Apply WRDS "preferred" flag if available
    if "preferred" in link.columns:
        before_pref = len(link)
        link = link.loc[link["preferred"] == 1].copy()
        utils.logger.info(
            f"Applied 'preferred == 1' filter: {before_pref:,} -> {len(link):,} rows."
        )
    else:
        utils.logger.warning(
            "No 'preferred' column in BoardEx link; using all rows (may include non-preferred links)."
        )

    return link


def _collapse_link(link: pd.DataFrame, key: str) -> pd.DataFrame:
    """
    Collapse link table to unique mapping: key -> company_id.

    If 'score' is available, choose the lowest-score match per key (best WRDS match).
    Otherwise, use the modal company_id per key.

    Returns:
        DataFrame with columns [key, company_id] (and 'score' if used).
    """
    if key not in link.columns:
        # pyrefly: ignore [bad-argument-type]
        return pd.DataFrame(columns=[key, "company_id"])

    df = link.dropna(subset=[key, "company_id"]).copy()
    if df.empty:
        # pyrefly: ignore [bad-argument-type]
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

    Primary linkage:
        spells.gvkey  <->  link.gvkey (from GVKAY)

    Secondary (optional) linkage:
        spells.permco <->  link.permco

    Legacy fallback (if available):
        spells.ticker <-> link.ticker

    Returns:
        spells DataFrame with an added 'company_id' column where matched.
    """
    if spells.empty or link.empty:
        utils.logger.warning(
            "Spells or BoardEx link table is empty in link_firms_to_boardex."
        )
        return spells

    spells = prepare_spells(spells)
    link = _clean_link_table(link)

    if link.empty:
        utils.logger.error(
            "Cleaned BoardEx link table is empty. Cannot link spells to company_id."
        )
        return spells

    merged = spells.copy()

    # --- gvkey-based linkage (preferred WRDS approach) ---
    if "gvkey" in spells.columns and "gvkey" in link.columns:
        gvkey_map = _collapse_link(link, "gvkey")
        utils.logger.info(
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
            utils.logger.warning(
                "Spells missing 'gvkey' column; cannot use GVKAY-based linkage."
            )
        if "gvkey" not in link.columns:
            utils.logger.warning(
                "BoardEx link missing 'gvkey' (GVKAY) column after normalization."
            )
        merged["company_id"] = np.nan

    # --- permco-based fallback ---
    if merged["company_id"].isna().any() and "permco" in spells.columns and "permco" in link.columns:
        permco_map = _collapse_link(link, "permco")
        utils.logger.info(
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
        utils.logger.info(
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
            utils.logger.info(
                f"Deduplicated linked spells: {before_dedup:,} -> {after_dedup:,}"
            )

    missing_company = merged["company_id"].isna().sum()
    utils.logger.info(
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

    Required fields:
        spells_linked:
            - company_id
            - appointment_date
            - spell_id (for downstream merges)
        directors:
            - company_id
            - director_id
            - date_start
            - date_end (can be missing for currently serving)
    """
    required_spell_cols = {"company_id", "appointment_date"}
    if not required_spell_cols.issubset(spells_linked.columns):
        missing = required_spell_cols - set(spells_linked.columns)
        utils.logger.error(
            f"spells_linked missing required columns {missing} in build_board_roster."
        )
        return pd.DataFrame()

    if directors.empty:
        utils.logger.error("Directors table is empty in build_board_roster.")
        return pd.DataFrame()

    spells = spells_linked.copy()
    dirs = directors.copy()

    # Harmonize company_id
    # pyrefly: ignore [bad-argument-type]
    spells["company_id"] = utils.clean_id(spells["company_id"])
    # pyrefly: ignore [bad-argument-type]
    dirs["company_id"] = utils.clean_id(dirs["company_id"])

    # Ensure director_id
    if "director_id" not in dirs.columns:
        utils.logger.error(
            "Directors table missing 'director_id' in build_board_roster."
        )
        return pd.DataFrame()
    # pyrefly: ignore [bad-argument-type]
    dirs["director_id"] = utils.clean_id(dirs["director_id"])

    # Ensure dates
    # pyrefly: ignore [no-matching-overload]
    dirs["date_start"] = pd.to_datetime(dirs.get("date_start"), errors="coerce")
    # pyrefly: ignore [no-matching-overload]
    dirs["date_end"] = pd.to_datetime(dirs.get("date_end"), errors="coerce")
    # pyrefly: ignore [missing-attribute]
    dirs["date_end"] = dirs["date_end"].fillna(pd.Timestamp("today").normalize())

    before_dir_filter = len(dirs)
    dirs = dirs.dropna(subset=["director_id", "company_id", "date_start"])
    utils.logger.info(
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
    utils.logger.info(
        f"After merging spells with directors on company_id: {len(merged):,} rows."
    )

    # Director must be on the board on appointment_date
    mask = (
        (merged["appointment_date"].notna())
        & (merged["appointment_date"] >= merged["date_start"])
        & (merged["appointment_date"] <= merged["date_end"])
    )
    roster = merged.loc[mask].copy()

    utils.logger.info(
        f"Board roster: {len(roster):,} director–spell rows after date overlap "
        f"filter (from {len(merged):,})."
    )

    if roster.empty:
        utils.logger.warning("Board roster is empty after date filtering.")
        return roster

    roster["director_id"] = utils.clean_id(roster["director_id"])
    return roster


def flag_search_committee(
    roster: pd.DataFrame,
    committees: pd.DataFrame,
) -> pd.DataFrame:
    """
    Flag search / nomination / governance committee membership.

    Committees table is expected to contain:
        - company_id
        - director_id
        - committee_name
        - c_date_start
        - c_date_end
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
    # pyrefly: ignore [bad-argument-type]
    roster["company_id"] = utils.clean_id(roster["company_id"])
    # pyrefly: ignore [bad-argument-type]
    roster["director_id"] = utils.clean_id(roster["director_id"])
    # pyrefly: ignore [bad-argument-type]
    coms["company_id"] = utils.clean_id(coms.get("company_id"))
    # pyrefly: ignore [bad-argument-type]
    coms["director_id"] = utils.clean_id(coms.get("director_id"))

    # Ensure dates
    roster["appointment_date"] = pd.to_datetime(
        roster["appointment_date"],
        errors="coerce",
    )
    # pyrefly: ignore [no-matching-overload]
    coms["c_date_start"] = pd.to_datetime(
        coms.get("c_date_start"),
        errors="coerce",
    )
    # pyrefly: ignore [no-matching-overload]
    coms["c_date_end"] = pd.to_datetime(
        coms.get("c_date_end"),
        errors="coerce",
    )
    # Treat missing start as very early and missing end as "still serving"
    coms["c_date_start"] = coms["c_date_start"].fillna(pd.Timestamp("1900-01-01"))
    # pyrefly: ignore [missing-attribute]
    coms["c_date_end"] = coms["c_date_end"].fillna(pd.Timestamp("today").normalize())

    # Filter relevant committees
    if "committee_name" not in coms.columns:
        utils.logger.warning(
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

    utils.logger.info(
        f"Relevant nomination/governance committees: {len(rel_coms):,} rows."
    )

    if rel_coms.empty:
        roster["is_search_committee"] = False
        return roster

    merged = pd.merge(
        roster,
        # pyrefly: ignore [bad-argument-type]
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
        utils.logger.warning(
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
    # pyrefly: ignore [no-matching-overload]
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
    # Count how many active board seats each director held at the appointment date
    
    # 1. Get unique director-date pairs from roster to minimize work
    unique_pairs = roster[["director_id", "appointment_date"]].drop_duplicates()
    
    if directors.empty:
        roster["n_boards"] = 1
        return roster

    # 2. Filter directors table to relevant directors only
    # (Optimization: avoid merging full 2M+ rows if we only need a subset)
    # Ensure IDs are clean in the reference table
    directors = directors.copy()
    # pyrefly: ignore [bad-argument-type]
    directors["director_id"] = utils.clean_id(directors["director_id"])
    # pyrefly: ignore [bad-argument-type]
    directors["company_id"] = utils.clean_id(directors["company_id"])

    # pyrefly: ignore [missing-attribute]
    relevant_ids = unique_pairs["director_id"].unique()
    rel_dirs = directors[directors["director_id"].isin(relevant_ids)].copy()
    
    # Ensure dates in reference table
    # pyrefly: ignore [no-matching-overload]
    rel_dirs["date_start"] = pd.to_datetime(rel_dirs.get("date_start"), errors="coerce")
    # pyrefly: ignore [no-matching-overload]
    rel_dirs["date_end"] = pd.to_datetime(rel_dirs.get("date_end"), errors="coerce")
    # pyrefly: ignore [missing-attribute]
    rel_dirs["date_end"] = rel_dirs["date_end"].fillna(pd.Timestamp("today").normalize())
    
    # 3. Merge unique pairs with director history
    # This gives all roles for each director at the time of interest
    merged = pd.merge(
        unique_pairs,
        # pyrefly: ignore [bad-argument-type]
        rel_dirs[["director_id", "company_id", "date_start", "date_end"]],
        on="director_id",
        how="inner"
    )
    
    # 4. Filter for active roles
    active_roles = merged[
        (merged["appointment_date"] >= merged["date_start"]) &
        (merged["appointment_date"] <= merged["date_end"])
    ]
    
    # 5. Count unique companies per director-date
    counts = (
        active_roles.groupby(["director_id", "appointment_date"])["company_id"]
        .nunique()
        # pyrefly: ignore [no-matching-overload]
        .reset_index(name="n_boards")
    )
    
    # 6. Merge back to roster
    roster = pd.merge(
        roster,
        counts,
        on=["director_id", "appointment_date"],
        how="left"
    )
    roster["n_boards"] = roster["n_boards"].fillna(1).astype(int) # Default to 1 if missing (at least the current one)

    return roster


# ---------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------


def run_phase3() -> pd.DataFrame | None:
    """
    Top-level entry point for Phase 3.

    Steps:
        1. Load CEO spells.
        2. Load / fetch BoardEx directors, committees, and link tables.
        3. Link spells to BoardEx company_id.
        4. Build director roster at appointment dates.
        5. Flag search committee membership.
        6. Compute director characteristics.
        7. Save director–spell linkage parquet.
    """
    utils.logger.info("Starting Phase 3: Director Selection (BoardEx)...")

    # 1. Load CEO spells
    if not config.CEO_SPELLS_PATH.exists():
        utils.logger.error(f"CEO spells file not found at {config.CEO_SPELLS_PATH}.")
        return None

    spells = pd.read_parquet(config.CEO_SPELLS_PATH)
    utils.logger.info(f"Loaded CEO spells: {len(spells):,} rows.")

    # 2. Load BoardEx data (local parquet or WRDS via utils.load_or_fetch)
    directors = utils.load_or_fetch(
        config.RAW_BOARDEX_DIRECTORS_PATH,
        utils.fetch_boardex_directors,
    )
    committees = utils.load_or_fetch(
        config.RAW_BOARDEX_COMMITTEES_PATH,
        utils.fetch_boardex_committees,
    )
    link = utils.load_or_fetch(
        config.RAW_BOARDEX_LINK_PATH,
        utils.fetch_boardex_link,
    )

    utils.logger.info(
        f"BoardEx directors: {len(directors):,} rows; "
        f"committees: {len(committees):,} rows; "
        f"link table: {len(link):,} rows."
    )

    if directors.empty:
        utils.logger.error("BoardEx directors table is empty. Aborting Phase 3.")
        return None

    # 3. Link spells to BoardEx boards
    spells_linked = link_firms_to_boardex(spells, link)

    if "company_id" not in spells_linked.columns:
        utils.logger.error("Failed to attach BoardEx company_id to spells.")
        return None

    # 4. Build board roster
    roster = build_board_roster(spells_linked, directors)
    if roster.empty:
        utils.logger.warning("No matching director rosters found in Phase 3.")
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

    # Also save as CSV
    csv_path = config.DIRECTOR_LINKAGE_CSV_PATH
    linkage.to_csv(csv_path, index=False)

    utils.logger.info(
        f"Phase 3 complete. Linked {len(linkage):,} director–spell observations "
        f"to {out_path} and {csv_path}."
    )

    # pyrefly: ignore [bad-return]
    return linkage


if __name__ == "__main__":
    run_phase3()
