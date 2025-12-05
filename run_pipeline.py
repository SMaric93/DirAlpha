import logging
import sys
from director_alpha import (
    phase0_universe, 
    phase1_performance, 
    phase2_ceo_turnover, 
    phase2b_compensation,
    phase3_directors,
    phase4_assembly
)

# Configure root logger to print to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def run_pipeline():
    print("="*80)
    print("Running Phase 0: Universe")
    phase0_universe.run_phase0()
    
    print("="*80)
    print("Running Phase 1: Performance")
    phase1_performance.run_phase1()
    
    print("="*80)
    print("Running Phase 2: CEO Turnover")
    phase2_ceo_turnover.run_phase2()
    
    print("="*80)
    print("Running Phase 2b: Compensation")
    phase2b_compensation.run_phase2b()
    
    print("="*80)
    print("Running Phase 3: Directors")
    phase3_directors.run_phase3()
    
    print("="*80)
    print("Running Phase 4: Assembly")
    phase4_assembly.run_phase4()

if __name__ == "__main__":
    run_pipeline()
