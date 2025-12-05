"""
Event Study Analysis Package.

Provides CAR and BHAR calculations using CAPM, FF3, and FF5 models.
"""
from .constants import (
    MODEL_SPECS,
    CAR_WINDOWS,
    BHAR_WINDOWS,
    ESTIMATION_WINDOW_START,
    ESTIMATION_WINDOW_END,
    ESTIMATION_WINDOW_LENGTH,
    MIN_OBS_ESTIMATION,
)
from .models import fit_models, calculate_expected_returns
from .metrics import calculate_car, calculate_bhar
from .alignment import align_event_data, link_spells_to_permno
from .runner import run_event_study, process_event

__all__ = [
    # Constants
    'MODEL_SPECS',
    'CAR_WINDOWS',
    'BHAR_WINDOWS',
    'ESTIMATION_WINDOW_START',
    'ESTIMATION_WINDOW_END',
    'ESTIMATION_WINDOW_LENGTH',
    'MIN_OBS_ESTIMATION',
    # Models
    'fit_models',
    'calculate_expected_returns',
    # Metrics
    'calculate_car',
    'calculate_bhar',
    # Alignment
    'align_event_data',
    'link_spells_to_permno',
    # Runner
    'run_event_study',
    'process_event',
]
