import sys
import logging
from director_alpha import (
    phase0_universe,
    phase1_performance,
    phase1b_ma,
    phase2_ceo_turnover,
    phase2b_compensation,
    phase2c_employment_history,
    phase3_directors,
    phase3c_director_history,
    phase3b_returns,
    phase4_assembly,
    phase5_connectivity,
    log
)

def main():
    logger = log.logger
    logger.info("Running Director Alpha Data Workflow...")
    
    try:
        # Phase 0
        phase0_universe.run_phase0()
        
        # Phase 1
        phase1_performance.run_phase1()
        
        # Phase 1b (M&A)
        phase1b_ma.run_phase1b()
        
        # Phase 2
        phase2_ceo_turnover.run_phase2()
        
        # Phase 2b (Compensation)
        phase2b_compensation.run_phase2b()
        
        # Phase 2c (Employment History)
        phase2c_employment_history.run_phase2c()
        
        # Phase 3
        phase3_directors.run_phase3()
        
        # Phase 3c (Director History)
        phase3c_director_history.run_phase3c()
        
        # Phase 3b (Event Study)
        phase3b_returns.run_event_study()
        
        # Phase 4
        phase4_assembly.run_phase4()
        
        # Phase 5
        phase5_connectivity.run_phase5()
        
        logger.info("Workflow Complete.")
        
    except Exception as e:
        logger.critical(f"Workflow failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()