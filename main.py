import sys
import logging
from director_alpha import (
    phase0_universe,
    phase1_performance,
    phase2_ceo_turnover,
    phase3_directors,
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
        
        # Phase 2
        phase2_ceo_turnover.run_phase2()
        
        # Phase 3
        phase3_directors.run_phase3()
        
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