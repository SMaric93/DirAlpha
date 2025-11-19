import os
from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Ensure data directory exists
DATA_DIR.mkdir(exist_ok=True)

# Database Connection (Placeholder - assuming local files or SQL connection string)
# In a real WRDS environment, this might use wrds.Connection()
WRDS_USERNAME = os.getenv("WRDS_USERNAME")

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
