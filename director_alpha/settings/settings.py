"""
Type-Safe Configuration with Dataclasses.

Provides structured configuration for Director Alpha pipeline.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple
import os


@dataclass
class Paths:
    """Project path configuration."""
    
    project_root: Path
    
    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"
    
    @property
    def intermediate_dir(self) -> Path:
        return self.data_dir / "intermediate"
    
    # Raw data paths
    @property
    def raw_compustat(self) -> Path:
        return self.data_dir / "compustat.parquet"
    
    @property
    def raw_crsp(self) -> Path:
        return self.data_dir / "crsp.parquet"
    
    @property
    def raw_crsp_dsf(self) -> Path:
        return self.data_dir / "crsp_dsf.parquet"
    
    @property
    def raw_ccm(self) -> Path:
        return self.data_dir / "ccm.parquet"
    
    @property
    def raw_execucomp(self) -> Path:
        return self.data_dir / "execucomp.parquet"
    
    @property
    def raw_boardex_directors(self) -> Path:
        return self.data_dir / "boardex_directors.parquet"
    
    @property
    def raw_boardex_committees(self) -> Path:
        return self.data_dir / "boardex_committees.parquet"
    
    @property
    def raw_boardex_link(self) -> Path:
        return self.data_dir / "boardex_link.parquet"
    
    @property
    def raw_ff5_factors(self) -> Path:
        return self.data_dir / "ff5_factors_daily.parquet"
    
    # Intermediate paths
    @property
    def firm_year_base(self) -> Path:
        return self.intermediate_dir / "firm_year_base.parquet"
    
    @property
    def firm_year_performance(self) -> Path:
        return self.intermediate_dir / "firm_year_performance.parquet"
    
    @property
    def ceo_spells(self) -> Path:
        return self.intermediate_dir / "ceo_spells.parquet"
    
    @property
    def ceo_spells_boardex(self) -> Path:
        return self.intermediate_dir / "ceo_spells_boardex.parquet"
    
    @property
    def director_linkage(self) -> Path:
        return self.intermediate_dir / "director_linkage.parquet"
    
    @property
    def event_study_results(self) -> Path:
        return self.intermediate_dir / "event_study_results.parquet"
    
    # Final output
    @property
    def analysis_hdfe(self) -> Path:
        return self.data_dir / "analysis_hdfe.parquet"
    
    def ensure_dirs(self) -> None:
        """Create data directories if they don't exist."""
        self.data_dir.mkdir(exist_ok=True)
        self.intermediate_dir.mkdir(exist_ok=True)


@dataclass
class WRDSConfig:
    """WRDS database configuration."""
    
    username: str = field(default_factory=lambda: os.getenv("WRDS_USERNAME", ""))
    
    # Table names
    comp_funda: str = "comp.funda"
    crsp_msf: str = "crsp.msf"
    crsp_msenames: str = "crsp.msenames"
    crsp_dsf: str = "crsp.dsf"
    crsp_dsedelist: str = "crsp.dsedelist"
    ccm_link: str = "crsp.ccmxpf_linktable"
    execucomp_anncomp: str = "execcomp.anncomp"
    boardex_directors: str = "boardex.na_wrds_org_composition"
    boardex_committees: str = "boardex.na_board_dir_committees"
    boardex_ccm_link: str = "wrdsapps.bdxcrspcomplink"
    ff_factors_daily: str = "ff.factors_daily"
    ff5_factors_daily: str = "ff.fivefactors_daily"


@dataclass
class UniverseParams:
    """Parameters for universe definition (Phase 0)."""
    
    start_year: int = 2000
    start_date: str = "2000-01-01"
    
    # SIC code ranges for exclusions
    sic_fin_range: Tuple[int, int] = (6000, 6999)
    sic_util_range: Tuple[int, int] = (4900, 4949)
    sic_other_start: int = 9000  # Non-operating
    
    # CCM link parameters
    link_types: List[str] = field(default_factory=lambda: ["LU", "LC"])
    link_prim: List[str] = field(default_factory=lambda: ["P", "C"])
    
    # CRSP share codes (common stock)
    share_codes: List[int] = field(default_factory=lambda: [10, 11])
    
    # Compustat columns
    compustat_cols: List[str] = field(default_factory=lambda: [
        "gvkey", "permno", "fyear", "datadate", "sich", "sic", "naics",
        "at", "oibdp", "prcc_f", "csho", "ceq", "dltt", "dlc", "xrd", "capx", "dv"
    ])


@dataclass
class PerformanceParams:
    """Parameters for performance calculations (Phase 1)."""
    
    winsorize_limits: Tuple[float, float] = (0.01, 0.99)
    
    cols_to_winsorize: List[str] = field(default_factory=lambda: [
        'roa', 'tobins_q', 'size', 'leverage', 
        'rd_intensity', 'capex_intensity', 
        'roa_adj', 'tobins_q_adj'
    ])


@dataclass
class EventStudyParams:
    """Parameters for event study analysis (Phase 3b)."""
    
    estimation_window_length: int = 252
    estimation_window_end: int = -31
    min_obs_estimation: int = 126
    
    car_windows: Dict[str, Tuple[int, int]] = field(default_factory=lambda: {
        'car_1_1': (-1, 1),
        'car_3_3': (-3, 3),
        'car_5_5': (-5, 5),
    })
    
    bhar_windows: Dict[str, Tuple[int, int]] = field(default_factory=lambda: {
        'bhar_1y': (0, 251),
        'bhar_3y': (0, 755),
    })
    
    model_specs: Dict[str, List[str]] = field(default_factory=lambda: {
        'capm': ['mktrf'],
        'ff3': ['mktrf', 'smb', 'hml'],
        'ff5': ['mktrf', 'smb', 'hml', 'rmw', 'cma']
    })
    
    @property
    def estimation_window_start(self) -> int:
        return self.estimation_window_end - self.estimation_window_length + 1


@dataclass
class Config:
    """Main configuration container."""
    
    paths: Paths
    wrds: WRDSConfig = field(default_factory=WRDSConfig)
    universe: UniverseParams = field(default_factory=UniverseParams)
    performance: PerformanceParams = field(default_factory=PerformanceParams)
    event_study: EventStudyParams = field(default_factory=EventStudyParams)
    
    @classmethod
    def create(cls, project_root: Path = None) -> 'Config':
        """Create configuration with default settings."""
        if project_root is None:
            # Default to parent of director_alpha package
            project_root = Path(__file__).resolve().parent.parent.parent
        
        config = cls(paths=Paths(project_root=project_root))
        config.paths.ensure_dirs()
        return config


def load_config() -> Config:
    """Load default configuration."""
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent.parent
    return Config.create(project_root)
