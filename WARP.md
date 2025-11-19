# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

DirAlpha is a Python-based data workflow system for academic finance research analyzing director selection and CEO turnover. The project processes financial data from multiple sources (Compustat, CRSP, ExecuComp, BoardEx) to create a connected network analysis dataset.

## Environment Setup

Python version: 3.13 (virtual environment managed in `.venv/`)

**Activate the virtual environment:**
```bash
source .venv/bin/activate
```

**Required dependencies:** pandas, numpy, scipy, networkx (no requirements.txt file exists yet)

## Common Commands

### Running the Full Pipeline

```bash
# Activate environment first
source .venv/bin/activate

# Run the complete workflow
python main.py
```

### Generating Test Data

```bash
# Generate dummy data for testing (creates Parquet files in data/ directory)
python tests/generate_dummy_data.py
```

### Running Individual Phases

Each phase can be run independently if intermediate outputs exist:

```bash
python -m director_alpha.phase0_universe
python -m director_alpha.phase1_performance
python -m director_alpha.phase2_ceo_turnover
python -m director_alpha.phase3_directors
python -m director_alpha.phase4_assembly
python -m director_alpha.phase5_connectivity
```

## Project Architecture

### Sequential Pipeline Design

The codebase follows a strict sequential data processing pipeline. Each phase depends on the output of previous phases:

**Phase 0 (Universe):** `phase0_universe.py`
- Merges Compustat, CRSP, and CCM datasets
- Applies filters: US firms, excludes financials/utilities, common stock only
- Output: `data/intermediate/firm_year_base.parquet`

**Phase 1 (Performance):** `phase1_performance.py`
- Calculates ROA, Tobin's Q, and control variables
- Performs industry adjustment (2-digit SIC)
- Applies annual winsorization at 1st/99th percentiles
- Output: `data/intermediate/firm_year_performance.parquet`

**Phase 2 (CEO Turnover):** `phase2_ceo_turnover.py`
- Identifies CEO spells from ExecuComp
- Handles co-CEOs (tie-breaking by Chairman title or highest TDC1)
- Filters interim CEOs and spells < 12 months
- Classifies internal vs external hires (based on JOINED_CO timing)
- Output: `data/intermediate/ceo_spells.parquet`

**Phase 3 (Directors):** `phase3_directors.py`
- Links firms to BoardEx via company_id
- Identifies active directors at CEO appointment dates
- Flags Nomination & Governance committee members
- Calculates director tenure and network metrics
- Output: `data/intermediate/director_linkage.parquet`

**Phase 4 (Assembly):** `phase4_assembly.py`
- Merges CEO spells, director linkage, and firm performance
- Creates tenure-specific performance measures (pre-appointment, during-tenure)
- Output: `data/analysis_hdfe.parquet`

**Phase 5 (Connectivity):** `phase5_connectivity.py`
- Constructs bipartite network graph (directors-firms)
- Identifies largest connected component using NetworkX
- Restricts sample to connected observations
- Output: `data/director_alpha_final.parquet`

### Data Flow

```
Raw Data (data/*.parquet)
  ├─> Phase 0 → firm_year_base.parquet
  ├─> Phase 1 → firm_year_performance.parquet
  ├─> Phase 2 → ceo_spells.parquet
  └─> Phase 3 → director_linkage.parquet
       ↓
  Phase 4 → analysis_hdfe.parquet
       ↓
  Phase 5 → director_alpha_final.parquet
```

### Configuration

All paths are centralized in `director_alpha/config.py`:
- `PROJECT_ROOT`: Project base directory
- `DATA_DIR`: Raw and output data location
- `INTERMEDIATE_DIR`: Intermediate processing outputs
- Expected raw data files: `compustat.parquet`, `crsp.parquet`, `ccm.parquet`, `execucomp.parquet`, `boardex_directors.parquet`, `boardex_committees.parquet`, `boardex_link.parquet`

## Key Implementation Details

### Date Handling
- All date columns are converted to pandas datetime using `pd.to_datetime()`
- Link validity checked with date ranges: `datadate >= linkdt AND datadate <= linkenddt`
- BoardEx director activity aligned to CEO appointment dates

### Merging Strategy
- Compustat-CRSP linkage via CCM with link type filtering (`LU`, `LC`) and primary/secondary flags (`P`, `C`)
- BoardEx linkage via `gvkey → company_id` mapping (requires link table)
- All merges preserve data types (explicit string conversion for `gvkey`)

### Industry Classification
- Uses 2-digit SIC codes (`sic2 = sich // 100`) as proxy for Fama-French 48
- Industry adjustment subtracts industry-year median

### Network Analysis
- Bipartite graph construction: Directors and Firms as distinct node types
- Uses `networkx.connected_components()` to find connected sets
- Node naming convention: `D_{director_id}` and `F_{gvkey}`

### Missing Data Handling
- R&D expenses (`xrd`) filled with 0 when missing
- Link end dates (`linkenddt`) filled with today's date if missing
- CEO spell end dates use `leftofc` or next CEO's `becameceo`

## Development Notes

- **No test suite:** Testing is done via dummy data generation script
- **No linting configured:** Code follows standard Python conventions
- **Data files not tracked:** All `.parquet` files should be in `data/` directory but not committed to git
- **Module imports:** Use relative imports within package (`from . import config`)
- **Error handling:** Missing files print warnings and return empty DataFrames with expected columns

## Data Requirements

The pipeline expects raw Parquet files in `data/` directory:
- `compustat.parquet`: Financial statements (columns: gvkey, datadate, fyear, fic, at, oibdp, prcc_f, csho, ceq, dltt, dlc, xrd, capx, sich, naics)
- `crsp.parquet`: Stock returns (columns: permno, date, shrcd, siccd, prc, ret, ncusip)
- `ccm.parquet`: Compustat-CRSP link (columns: gvkey, lpermno, linkdt, linkenddt, linktype, linkprim)
- `execucomp.parquet`: Executive compensation (columns: gvkey, year, execid, pceo, ceoann, becameceo, leftofc, joined_co, title, tdc1, age, gender)
- `boardex_directors.parquet`: Board membership (columns: company_id, director_id, date_start, date_end, role_name)
- `boardex_committees.parquet`: Committee membership (columns: company_id, director_id, committee_name, c_date_start, c_date_end)
- `boardex_link.parquet`: Firm identifier linkage (columns: gvkey, company_id)
