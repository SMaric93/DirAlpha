"""
Event Study Constants and Configuration.

Defines estimation windows, event windows, and factor model specifications.
"""
from typing import Dict, List

# =============================================================================
# Estimation Window Constants
# =============================================================================

# Estimation Window: [T-282, T-31]
# Length 252 days, ending 30 days before event (T=0) to prevent contamination.
ESTIMATION_WINDOW_LENGTH: int = 252
ESTIMATION_WINDOW_END: int = -31
ESTIMATION_WINDOW_START: int = ESTIMATION_WINDOW_END - ESTIMATION_WINDOW_LENGTH + 1  # T-282

# Minimum observations required within the estimation window
MIN_OBS_ESTIMATION: int = 126

# =============================================================================
# Event Windows (Inclusive)
# =============================================================================

CAR_WINDOWS: Dict[str, tuple] = {
    'car_1_1': (-1, 1),
    'car_3_3': (-3, 3),
    'car_5_5': (-5, 5),
}

# BHAR Windows (Inclusive, starting at T=0)
BHAR_WINDOWS: Dict[str, tuple] = {
    'bhar_1y': (0, 251),   # T=0 to T+251 (252 days total)
    'bhar_3y': (0, 755),   # T=0 to T+755 (756 days total)
}

# =============================================================================
# Factor Model Specifications
# =============================================================================

MODEL_SPECS: Dict[str, List[str]] = {
    'capm': ['mktrf'],
    'ff3': ['mktrf', 'smb', 'hml'],
    'ff5': ['mktrf', 'smb', 'hml', 'rmw', 'cma']
}
