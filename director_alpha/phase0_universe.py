import pandas as pd
import numpy as np
from . import config, utils


def run_phase0():
    """
    Phase 0: Universe Definition (Compustat–CRSP firm-year base)

    Steps:
    1. Load Compustat, CRSP, and CCM.
    2. Link Compustat to CRSP via CCM with date-valid link.
    3. Apply universe filters: US-incorporated, exclude financials & utilities.
    4. Restrict to common stock (CRSP share codes 10/11) using an annual snapshot.
    5. Output a firm-year base file.
    """
    utils.logger.info("Starting Phase 0: Universe Definition...")

    # ------------------------------------------------------------------
    # 1. Load Data
    # ------------------------------------------------------------------
    compustat = utils.load_or_fetch(
        config.RAW_COMPUSTAT_PATH,
        utils.fetch_compustat_funda,
        start_year=2000
    )

    crsp = utils.load_or_fetch(
        config.RAW_CRSP_PATH,
        utils.fetch_crsp_msf,
        start_date="2000-01-01"
    )

    ccm = utils.load_or_fetch(
        config.RAW_CCM_PATH,
        utils.fetch_ccm_link
    )

    if compustat.empty or crsp.empty or ccm.empty:
        utils.logger.error("One or more input datasets are empty. Aborting Phase 0.")
        return None

    # ------------------------------------------------------------------
    # 2. CCM: Filter and link Compustat ↔ CRSP
    # ------------------------------------------------------------------
    utils.logger.info("Merging Compustat, CCM, and CRSP...")

    # Keep standard link types / primaries
    if "linktype" in ccm.columns:
        ccm = ccm[ccm["linktype"].isin(["LU", "LC"])]
    else:
        utils.logger.warning("CCM 'linktype' column missing; link filtering may be too broad.")

    if "linkprim" in ccm.columns:
        ccm = ccm[ccm["linkprim"].isin(["P", "C"])]
    else:
        utils.logger.warning("CCM 'linkprim' column missing; link filtering may be too broad.")

    # Normalize gvkey
    if "gvkey" not in compustat.columns or "gvkey" not in ccm.columns:
        utils.logger.error("Missing 'gvkey' in Compustat or CCM. Aborting Phase 0.")
        return None

    compustat = compustat.copy()
    ccm = ccm.copy()

    compustat["gvkey"] = utils.normalize_gvkey(compustat["gvkey"])
    ccm["gvkey"] = utils.normalize_gvkey(ccm["gvkey"])

    # Inner join on gvkey
    merged = pd.merge(compustat, ccm, on="gvkey", how="inner")

    # Date handling for link validity
    # datadate: Compustat accounting date
    if "datadate" not in merged.columns:
        utils.logger.error("Compustat dataframe missing 'datadate'. Aborting Phase 0.")
        return None

    merged["datadate"] = pd.to_datetime(merged["datadate"], errors="coerce")
    merged = merged[merged["datadate"].notna()]

    # CCM link dates
    for col in ["linkdt", "linkenddt"]:
        if col not in merged.columns:
            utils.logger.error(f"CCM dataframe missing '{col}'. Aborting Phase 0.")
            return None

    merged["linkdt"] = pd.to_datetime(merged["linkdt"], errors="coerce")
    merged["linkenddt"] = pd.to_datetime(merged["linkenddt"], errors="coerce")

    # Treat open-ended links as far-future
    merged["linkenddt"] = merged["linkenddt"].fillna(pd.Timestamp("2099-12-31"))

    # Drop rows with missing linkdt (cannot check validity)
    merged = merged[merged["linkdt"].notna()]

    # Keep only rows where datadate falls within the link window
    link_valid_mask = (merged["datadate"] >= merged["linkdt"]) & (merged["datadate"] <= merged["linkenddt"])
    merged = merged[link_valid_mask]

    if merged.empty:
        utils.logger.error("No valid Compustat–CCM link records after date filtering. Aborting Phase 0.")
        return None

    # Rename lpermno → permno and normalize type
    if "lpermno" not in merged.columns:
        utils.logger.error("CCM dataframe missing 'lpermno'. Aborting Phase 0.")
        return None

    merged = merged.rename(columns={"lpermno": "permno"})
    merged["permno"] = pd.to_numeric(merged["permno"], errors="coerce")
    merged = merged[merged["permno"].notna()]
    merged["permno"] = merged["permno"].astype(int)

    # ------------------------------------------------------------------
    # 3. Universe Filters: US-incorporated, Non-Financial, Non-Utility
    # ------------------------------------------------------------------
    utils.logger.info("Applying universe filters (US Inc, Non-Fin/Util)...")

    # US incorporated (fic = 'USA'), if available
    if "fic" in merged.columns:
        before = len(merged)
        merged = merged[merged["fic"] == "USA"]
        utils.logger.info(f"FIC filter (USA): kept {len(merged)} of {before} rows.")
    else:
        utils.logger.warning("Column 'fic' not found; skipping US-incorporation filter.")

    # SIC / SICH filters for Financials and Utilities
    sic_col = None
    if "sich" in merged.columns:
        sic_col = "sich"
    elif "sic" in merged.columns:
        sic_col = "sic"

    if sic_col is not None:
        merged[sic_col] = pd.to_numeric(merged[sic_col], errors="coerce")
        sic = merged[sic_col]

        condition_fin = (sic >= 6000) & (sic <= 6999)
        condition_util = (sic >= 4900) & (sic <= 4949)

        before = len(merged)
        merged = merged[~(condition_fin | condition_util)]
        utils.logger.info(
            f"Financial/Utility SIC filter: removed {before - len(merged)} rows; "
            f"remaining {len(merged)}."
        )
    else:
        utils.logger.warning("No 'sich' or 'sic' column found; skipping Financial/Utility filter.")

    # ------------------------------------------------------------------
    # 4. Merge with CRSP and Require Common Stock (SHRCD 10/11)
    # ------------------------------------------------------------------
    utils.logger.info("Checking Share Codes (10, 11)...")

    if "permno" not in crsp.columns:
        utils.logger.error("CRSP data missing 'permno'. Aborting Phase 0.")
        return None

    crsp = crsp.copy()
    crsp["permno"] = pd.to_numeric(crsp["permno"], errors="coerce")
    crsp = crsp[crsp["permno"].notna()]
    crsp["permno"] = crsp["permno"].astype(int)

    if "date" not in crsp.columns:
        utils.logger.error("CRSP data missing 'date'. Aborting Phase 0.")
        return None

    crsp["date"] = pd.to_datetime(crsp["date"], errors="coerce")
    crsp = crsp[crsp["date"].notna()]
    crsp["year"] = crsp["date"].dt.year

    if "shrcd" not in crsp.columns:
        utils.logger.error("CRSP data missing 'shrcd'. Aborting Phase 0.")
        return None

    # Mark common stock months
    crsp["is_common"] = crsp["shrcd"].isin([10, 11])

    # For each (permno, year), check if it was EVER common during the year
    crsp_annual = (
        crsp.groupby(["permno", "year"])["is_common"]
        .any()
        .reset_index()
    )

    # Keep only permno-years that are common stock at some point in the year
    crsp_annual = crsp_annual[crsp_annual["is_common"]].drop(columns=["is_common"])

    # Create fiscal-year / calendar-year merge key for Compustat side
    # Here we use calendar year of datadate; if you want fyear instead, change this.
    merged["year"] = merged["datadate"].dt.year

    # Merge Compustat+CCM with CRSP annual share-code snapshot
    final_df = pd.merge(
        merged,
        crsp_annual,
        on=["permno", "year"],
        how="inner"
    )

    if final_df.empty:
        utils.logger.error("No records after merging with CRSP share-code universe. Aborting Phase 0.")
        return None

    # ------------------------------------------------------------------
    # 5. Build firm-year base and export
    # ------------------------------------------------------------------
    # Choose core variables to keep (only keep if available)
    columns_to_keep = [
        "gvkey",
        "permno",
        "fyear",
        "datadate",
        "sich",
        "sic",
        "naics",
        "at",
        "oibdp",
        "prcc_f",
        "csho",
        "ceq",
        "dltt",
        "dlc",
        "xrd",
        "capx",
    ]

    available_cols = [col for col in columns_to_keep if col in final_df.columns]
    firm_year_base = final_df[available_cols].copy()

    # Optionally de-duplicate to one record per (gvkey, fyear) using last datadate
    if {"gvkey", "fyear", "datadate"}.issubset(firm_year_base.columns):
        firm_year_base = (
            firm_year_base
            .sort_values(["gvkey", "fyear", "datadate"])
            .groupby(["gvkey", "fyear"], as_index=False)
            .tail(1)
        )

    utils.logger.info(
        f"Phase 0 Complete. Generated {len(firm_year_base)} firm-year observations."
    )

    # Save to parquet
    firm_year_base.to_parquet(config.FIRM_YEAR_BASE_PATH, index=False)

    return firm_year_base


if __name__ == "__main__":
    run_phase0()