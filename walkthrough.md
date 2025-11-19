# Director Alpha Data Workflow - Walkthrough

I have implemented the full data workflow for the "Director Alpha" project. The code is organized into a modular Python package `director_alpha`.

## Implementation Details

### Project Structure
- `director_alpha/config.py`: Configuration for file paths.
- `director_alpha/phase0_universe.py`: Universe definition and filtering.
- `director_alpha/phase1_performance.py`: Calculation of ROA, Tobin's Q, and controls.
- `director_alpha/phase2_ceo_turnover.py`: Identification of CEO spells and turnover.
- `director_alpha/phase3_directors.py`: Linkage of BoardEx directors to selection events, including Committee identification.
- `director_alpha/phase4_assembly.py`: Assembly of the final analysis dataset.
- `director_alpha/phase5_connectivity.py`: Network analysis and restriction to the connected set.
- `main.py`: Orchestration script.

## Verification Results

I verified the implementation by generating dummy data representing 50 firms over 10 years, with simulated CEO turnovers and director rosters.

### Execution Log
The workflow executed successfully with the following results:

1.  **Phase 0 (Universe):** Generated 500 firm-year observations.
2.  **Phase 1 (Performance):** Calculated financial metrics for all 500 observations.
3.  **Phase 2 (CEO Spells):** Identified 100 CEO spells (2 per firm).
4.  **Phase 3 (Directors):** Linked 501 directors to selection events using BoardEx dummy data.
5.  **Phase 4 (Assembly):** Assembled 501 observations with tenure-specific performance.
6.  **Phase 5 (Connectivity):** Identified a connected component and restricted the sample to 21 observations.

### Key Logic Verified
- **Merging:** Compustat/CRSP/CCM merging logic handles dates and linking codes.
- **Calculations:** ROA, Tobin's Q, and industry adjustments are implemented.
- **Turnover:** CEO turnover identification correctly handles spells.
- **Director Linkage:** BoardEx directors are linked via CompanyID. Nomination & Governance committee members are correctly identified.
- **Connectivity:** Network analysis correctly identifies connected components.

## How to Run
1.  Ensure your raw data (Parquet files) is in the `data/` directory as specified in `config.py`.
2.  Run the main script:
    ```bash
    python main.py
    ```
