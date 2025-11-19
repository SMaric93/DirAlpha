import sys
from director_alpha import phase0_universe
from director_alpha import phase1_performance
from director_alpha import phase2_ceo_turnover
from director_alpha import phase3_directors
from director_alpha import phase4_assembly
from director_alpha import phase5_connectivity

def main():
    print("Running Director Alpha Data Workflow...")
    
    # Phase 0
    phase0_universe.run_phase0()
    
    # Phase 1
    phase1_performance.run_phase1()
    
    # Phase 2
    phase2_ceo_turnover.run_phase2()
    
    # Phase 3
    phase3_directors.run_phase3()
    
    # Phase 4
    phase4_assembly.run_phase4()
    
    # Phase 5
    phase5_connectivity.run_phase5()
    
    print("Workflow Complete.")

if __name__ == "__main__":
    main()
