# Director Alpha

This project analyzes the relationship between director characteristics and firm performance/CEO turnover using data from WRDS (Compustat, CRSP, ExecuComp, BoardEx).

## Prerequisites

- Python 3.8+
- A valid WRDS account
- `wrds` python library installed (`pip install wrds`)
- A `.pgpass` file configured for WRDS authentication (optional but recommended) or WRDS credentials.

## Setup

We provide a setup script to automate the environment creation and configuration.

1. Run the setup script:
   ```bash
   ./setup.sh
   ```
   This script will:
   - Create a virtual environment (`.venv`)
   - Install all dependencies
   - Prompt for your WRDS username and password
   - Create a `.env` file with your credentials

2. Activate the environment:
   ```bash
   source .venv/bin/activate
   ```

## Configuration

The project uses `director_alpha/config.py` which loads credentials from the `.env` file created by the setup script.
- **WRDS Connection**: Credentials are read from `WRDS_USERNAME` and `WRDS_PASSWORD` in `.env`.
- **Data Directory**: Data is saved to `data/` in the project root.

## Running the Pipeline

The pipeline consists of sequential phases. Run them in order:

### 1. Phase 0: Universe Definition
Loads Compustat/CRSP data and defines the firm universe.
```bash
python -m director_alpha.phase0_universe
```

### 2. Phase 1: Firm Performance
Calculates financial ratios (ROA, Tobin's Q) and stock returns.
```bash
python -m director_alpha.phase1_performance
```

### 3. Phase 2: CEO Turnover
Identifies CEO spells and turnover events from ExecuComp.
```bash
python -m director_alpha.phase2_ceo_turnover
```

### 4. Phase 3: Directors
Links firms to BoardEx and identifies director characteristics (Nomination Committee, etc.).
```bash
python -m director_alpha.phase3_directors
```

### 5. Phase 3b: Event Study Returns
Calculates Cumulative Abnormal Returns (CAR) and Buy-and-Hold Abnormal Returns (BHAR) around CEO appointment dates using CAPM and Fama-French 3-Factor models.
```bash
python -m director_alpha.phase3b_returns
```

### 6. Phase 4: Assembly
Merges all data into a final analysis dataset (`data/analysis_hdfe.parquet`).
```bash
python -m director_alpha.phase4_assembly
```

### 7. Phase 5: Connectivity Analysis
Performs connectivity analysis on the director-firm network to identify the largest connected component.
```bash
python -m director_alpha.phase5_connectivity
```

## Output Files

| Phase | File | Description |
|-------|------|-------------|
| 0-1 | `data/intermediate/firm_year_base.parquet` | Base universe with financials and performance metrics. |
| 2 | `data/intermediate/ceo_spells.parquet` | CEO tenure spells. |
| 3 | `data/intermediate/director_linkage.parquet` | Director-Spell level dataset (BoardEx linked). |
| 3b | `data/intermediate/event_study_results.parquet` | CAR and BHAR metrics for CEO appointments. |
| 4 | `data/analysis_hdfe.parquet` | Final assembled dataset for analysis (pre-connectivity filter). |
| 5 | `data/director_alpha_final.parquet` | Final connected dataset ready for HDFE estimation. |

## Running Tests

To verify the WRDS loading logic (using mocks):
```bash
python -m pytest tests/test_wrds_loading.py
```
