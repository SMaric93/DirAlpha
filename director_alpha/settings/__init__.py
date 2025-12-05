"""
Settings Package.

Provides type-safe configuration via dataclasses alongside legacy config module.
"""
from .settings import (
    Config,
    Paths,
    WRDSConfig,
    UniverseParams,
    PerformanceParams,
    EventStudyParams,
    load_config,
)

__all__ = [
    'Config',
    'Paths',
    'WRDSConfig',
    'UniverseParams',
    'PerformanceParams',
    'EventStudyParams',
    'load_config',
]
