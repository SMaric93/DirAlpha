# WRDS Data Loading Implementation

I have refactored the codebase to support loading data directly from WRDS when local parquet files are missing. This ensures the pipeline can run in an environment with WRDS access without requiring pre-downloaded files.

## Changes

### Configuration
- **`config.py`**: Added `get_wrds_connection()` function and defined WRDS table names (`comp.funda`, `crsp.msf`, `comp.anncomp`, `boardex.na_wrds_org_composition`, etc.).

### Phase 0: Universe Definition
- **`phase0_universe.py`**: Updated `load_data()` to:
    1. Attempt to load from local parquet files.
    2. If missing, connect to WRDS and fetch Compustat, CRSP, and CCM data.
    3. Save fetched data to local parquet files for caching.

### Phase 1: Performance
- **`phase1_performance.py`**: Updated `run_phase1()` to fetch CRSP monthly stock data (`crsp.msf`) from WRDS if the local file is missing, enabling stock return calculations.

### Phase 2: CEO Turnover
- **`phase2_ceo_turnover.py`**: Updated `run_phase2()` to fetch ExecuComp data (`comp.anncomp`) from WRDS if the local file is missing.

### Phase 3: Directors
- **`phase3_directors.py`**: Updated `load_boardex_data()` to fetch BoardEx directors, committees, and company profile data from WRDS if local files are missing.

## Verification

I created a test file `tests/test_wrds_loading.py` that mocks the `wrds` library and verifies the data loading logic for all phases.

### Test Results
```
tests/test_wrds_loading.py ....                                          [100%]
============================== 4 passed in 0.99s ===============================
```
- `test_phase0_load_data_wrds`: Verified Compustat, CRSP, and CCM loading.
- `test_phase1_load_crsp_wrds`: Verified CRSP loading for performance.
- `test_phase2_load_execucomp_wrds`: Verified ExecuComp loading.
- `test_phase3_load_boardex_wrds`: Verified BoardEx loading.
