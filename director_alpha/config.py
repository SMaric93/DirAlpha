import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Ensure data directory exists
DATA_DIR.mkdir(exist_ok=True)

# WRDS Connection
WRDS_USERNAME = os.getenv("WRDS_USERNAME")

# WRDS Table Names
WRDS_COMP_FUNDA = "comp.funda"
WRDS_CRSP_MSF = "crsp.msf" # Monthly Stock File
WRDS_CRSP_MSENAMES = "crsp.msenames" # Monthly Stock Event - Names
WRDS_CRSP_DSF = "crsp.dsf" # Daily Stock File (if needed)
WRDS_CCM_LINK = "crsp.ccmxpf_linktable"
WRDS_EXECUCOMP_ANNCOMP = "execcomp.anncomp"
WRDS_BOARDEX_DIRECTORS = "boardex.na_wrds_org_composition"
WRDS_BOARDEX_COMMITTEES = "boardex.na_board_dir_committees"
WRDS_BOARDEX_PROFILE = "boardex.na_wrds_company_profile" # For linking


# File Paths for Raw Data (if using CSV/Parquet)
# These are placeholders. The user will need to populate the data directory.
RAW_COMPUSTAT_PATH = DATA_DIR / "compustat.parquet"
RAW_CRSP_PATH = DATA_DIR / "crsp.parquet"
RAW_CCM_PATH = DATA_DIR / "ccm.parquet"
RAW_EXECUCOMP_PATH = DATA_DIR / "execucomp.parquet"
RAW_ISS_DIRECTORS_PATH = DATA_DIR / "iss_directors.parquet" # Keeping for legacy reference if needed
RAW_BOARDEX_DIRECTORS_PATH = DATA_DIR / "boardex_directors.parquet"
RAW_BOARDEX_COMMITTEES_PATH = DATA_DIR / "boardex_committees.parquet"
RAW_BOARDEX_LINK_PATH = DATA_DIR / "boardex_link.parquet" # Linking table (Ticker/CUSIP to CompanyID)

# Output Paths
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
INTERMEDIATE_DIR.mkdir(exist_ok=True)

FIRM_YEAR_BASE_PATH = INTERMEDIATE_DIR / "firm_year_base.parquet"
CEO_SPELLS_PATH = INTERMEDIATE_DIR / "ceo_spells.parquet"
DIRECTOR_LINKAGE_PATH = INTERMEDIATE_DIR / "director_linkage.parquet"
ANALYSIS_HDFE_PATH = DATA_DIR / "analysis_hdfe.parquet"
