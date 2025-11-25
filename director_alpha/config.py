import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ---------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INTERMEDIATE_DIR = DATA_DIR / "intermediate"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
INTERMEDIATE_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------
# WRDS Configuration
# ---------------------------------------------------------------------
WRDS_USERNAME = os.getenv("WRDS_USERNAME")

# Table Names
WRDS_COMP_FUNDA = "comp.funda"
WRDS_CRSP_MSF = "crsp.msf"
WRDS_CRSP_MSENAMES = "crsp.msenames"
WRDS_CRSP_DSF = "crsp.dsf"
WRDS_CRSP_DSEDELIST = "crsp.dsedelist"
WRDS_CRSP_STOCKNAMES = "crsp.stocknames"
WRDS_CCM_LINK = "crsp.ccmxpf_linktable"
WRDS_CCM_LINKTABLE = "crsp.ccmxpf_linktable"
WRDS_EXECUCOMP_ANNCOMP = "execcomp.anncomp"
WRDS_BOARDEX_DIRECTORS = "boardex.na_wrds_org_composition"
WRDS_BOARDEX_COMMITTEES = "boardex.na_board_dir_committees"
WRDS_BOARDEX_CCM_LINK = "wrdsapps.bdxcrspcomplink"
WRDS_EXEC_BOARDEX_LINK = "wrdsapps.exec_boardex_link"
WRDS_BOARDEX_PROFILE = "boardex.na_wrds_company_profile"
WRDS_BOARDEX_ANNOUNCEMENTS = "boardex.na_board_dir_announcements"
WRDS_FF_FACTORS_DAILY = "ff.factors_daily"
WRDS_FF5_FACTORS_DAILY = "ff.fivefactors_daily"

# ---------------------------------------------------------------------
# File Paths
# ---------------------------------------------------------------------
# Raw Data
RAW_COMPUSTAT_PATH = DATA_DIR / "compustat.parquet"
RAW_CRSP_PATH = DATA_DIR / "crsp.parquet"
RAW_CCM_PATH = DATA_DIR / "ccm.parquet"
RAW_EXECUCOMP_PATH = DATA_DIR / "execucomp.parquet"
RAW_BOARDEX_DIRECTORS_PATH = DATA_DIR / "boardex_directors.parquet"
RAW_BOARDEX_COMMITTEES_PATH = DATA_DIR / "boardex_committees.parquet"
RAW_BOARDEX_LINK_PATH = DATA_DIR / "boardex_link.parquet"
RAW_EXEC_BOARDEX_LINK_PATH = DATA_DIR / "exec_boardex_link.parquet"
RAW_BOARDEX_ANNOUNCEMENTS_PATH = DATA_DIR / "boardex_announcements.parquet"
RAW_CRSP_DSF_PATH = DATA_DIR / "crsp_dsf.parquet"
RAW_FF_FACTORS_PATH = DATA_DIR / "ff_factors.parquet"
RAW_FF5_FACTORS_DAILY_PATH = DATA_DIR / "ff5_factors_daily.parquet"

# Intermediate Data
FIRM_YEAR_BASE_PATH = INTERMEDIATE_DIR / "firm_year_base.parquet"
FIRM_YEAR_PERFORMANCE_PATH = INTERMEDIATE_DIR / "firm_year_performance.parquet"
CEO_SPELLS_PATH = INTERMEDIATE_DIR / "ceo_spells.parquet"
CEO_SPELLS_BOARDEX_PATH = INTERMEDIATE_DIR / "ceo_spells_boardex.parquet"
DIRECTOR_LINKAGE_PATH = INTERMEDIATE_DIR / "director_linkage.parquet"
DIRECTOR_LINKAGE_CSV_PATH = INTERMEDIATE_DIR / "director_linkage.csv"
EVENT_STUDY_RESULTS_PATH = INTERMEDIATE_DIR / "event_study_results.parquet"
EVENT_STUDY_RESULTS_CSV_PATH = INTERMEDIATE_DIR / "event_study_results.csv"

# Final Output
ANALYSIS_HDFE_PATH = DATA_DIR / "analysis_hdfe.parquet"
ANALYSIS_HDFE_CSV_PATH = DATA_DIR / "analysis_hdfe.csv"

# ---------------------------------------------------------------------
# Business Logic Parameters
# ---------------------------------------------------------------------

# Phase 0: Universe Definition
UNIVERSE_START_YEAR = 2000
UNIVERSE_START_DATE = "2000-01-01"

# SIC Codes for Exclusions
SIC_FIN_START = 6000
SIC_FIN_END = 6999
SIC_UTIL_START = 4900
SIC_UTIL_END = 4949
SIC_OTHER_START = 9000  # Non-operating establishments

# CCM Link Parameters
LINK_TYPES = ["LU", "LC"]
LINK_PRIM = ["P", "C"]

# CRSP Share Codes (Common Stock)
SHARE_CODES = [10, 11]

# Compustat Variables to Keep
COMPUSTAT_COLS_TO_KEEP = [
    "gvkey", "permno", "fyear", "datadate", "sich", "sic", "naics",
    "at", "oibdp", "prcc_f", "csho", "ceq", "dltt", "dlc", "xrd", "capx"
]

# Phase 1: Performance
WINSORIZE_LIMITS = (0.01, 0.99)
PERFORMANCE_COLS_TO_WINSORIZE = [
    'roa', 'tobins_q', 'size', 'leverage', 
    'rd_intensity', 'capex_intensity', 
    'roa_adj', 'tobins_q_adj'
]

# Phase 3b: Event Study
ESTIMATION_WINDOW_LENGTH = 252
ESTIMATION_WINDOW_END = -31
ESTIMATION_WINDOW_START = ESTIMATION_WINDOW_END - ESTIMATION_WINDOW_LENGTH + 1 # T-282
MIN_OBS_ESTIMATION = 126

# Event Windows (Inclusive)
CAR_WINDOWS = {
    'car_1_1': (-1, 1),
    'car_3_3': (-3, 3),
    'car_5_5': (-5, 5),
}

BHAR_WINDOWS = {
    'bhar_1y': (0, 251),   # T=0 to T+251 (252 days total)
    'bhar_3y': (0, 755),   # T=0 to T+755 (756 days total)
}

MODEL_SPECS = {
    'capm': ['mktrf'],
    'ff3': ['mktrf', 'smb', 'hml'],
    'ff5': ['mktrf', 'smb', 'hml', 'rmw', 'cma']
}