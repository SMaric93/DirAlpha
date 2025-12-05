"""
Base Phase Class for Director Alpha Pipeline.

Provides a standardized interface for all pipeline phases, reducing boilerplate
and ensuring consistent logging, error handling, and file management.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
from . import log

logger = log.logger


class BasePhase(ABC):
    """
    Abstract base class for pipeline phases.
    
    Subclasses must implement:
        - name: Human-readable phase name
        - input_paths: List of required input file paths
        - output_path: Path for the output file
        - process(): Core processing logic
    
    Example usage:
        class Phase1Performance(BasePhase):
            @property
            def name(self) -> str:
                return "Phase 1: Firm Performance Panel"
            
            @property
            def input_paths(self) -> List[Path]:
                return [config.FIRM_YEAR_BASE_PATH]
            
            @property
            def output_path(self) -> Path:
                return config.FIRM_YEAR_PERFORMANCE_PATH
            
            def process(self, inputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
                df = inputs['firm_year_base']
                # ... processing logic ...
                return df
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for the phase (e.g., 'Phase 1: Firm Performance')."""
        pass
    
    @property
    @abstractmethod
    def input_paths(self) -> List[Path]:
        """
        List of required input file paths.
        
        Each path should point to a parquet file. The file's stem (name without extension)
        will be used as the key in the inputs dict passed to process().
        """
        pass
    
    @property
    @abstractmethod
    def output_path(self) -> Path:
        """Path where the output parquet file will be saved."""
        pass
    
    @property
    def save_csv(self) -> bool:
        """Whether to also save a CSV copy of the output. Default: False."""
        return False
    
    @abstractmethod
    def process(self, inputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Core processing logic for the phase.
        
        Args:
            inputs: Dictionary mapping input file stems to their DataFrames.
                   e.g., {'firm_year_base': pd.DataFrame(...)}
        
        Returns:
            Processed DataFrame to be saved.
        """
        pass
    
    def validate_inputs(self) -> bool:
        """Check that all required input files exist."""
        missing = []
        for path in self.input_paths:
            if not path.exists():
                missing.append(path)
        
        if missing:
            for path in missing:
                logger.error(f"Missing input file: {path}")
            return False
        return True
    
    def load_inputs(self) -> Dict[str, pd.DataFrame]:
        """Load all input files into a dictionary."""
        inputs = {}
        for path in self.input_paths:
            key = path.stem
            logger.info(f"Loading {key} from {path}...")
            inputs[key] = pd.read_parquet(path)
        return inputs
    
    def save_output(self, df: pd.DataFrame) -> None:
        """Save the output DataFrame to parquet (and optionally CSV)."""
        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_parquet(self.output_path, index=False)
        logger.info(f"Saved output to {self.output_path}")
        
        if self.save_csv:
            csv_path = self.output_path.with_suffix('.csv')
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved CSV to {csv_path}")
    
    def run(self) -> Optional[pd.DataFrame]:
        """
        Execute the phase with standard logging and error handling.
        
        Returns:
            The processed DataFrame, or None if the phase failed.
        """
        logger.info(f"Starting {self.name}...")
        
        # Validate inputs
        if not self.validate_inputs():
            logger.error(f"{self.name} aborted: missing inputs.")
            return None
        
        try:
            # Load inputs
            inputs = self.load_inputs()
            
            # Check for empty inputs
            for key, df in inputs.items():
                if df.empty:
                    logger.warning(f"Input '{key}' is empty.")
            
            # Process
            result = self.process(inputs)
            
            if result is None or result.empty:
                logger.warning(f"{self.name} produced no output.")
                return None
            
            # Save
            self.save_output(result)
            
            logger.info(f"{self.name} complete. Generated {len(result)} records.")
            return result
            
        except Exception as e:
            logger.error(f"{self.name} failed: {e}", exc_info=True)
            return None


class FetchPhase(BasePhase):
    """
    Base class for phases that fetch data from external sources (e.g., WRDS).
    
    Extends BasePhase with support for data fetching via io.load_or_fetch().
    """
    
    @property
    def fetch_configs(self) -> List[Dict]:
        """
        Configuration for data fetching.
        
        Each config dict should have:
            - 'path': Output path for the fetched data
            - 'fetch_func': Function to fetch the data
            - 'kwargs': Optional kwargs for the fetch function
        """
        return []
    
    def fetch_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch all required datasets."""
        from . import io, db
        
        data = {}
        for cfg in self.fetch_configs:
            path = cfg['path']
            key = path.stem
            
            kwargs = cfg.get('kwargs', {})
            df = io.load_or_fetch(path, cfg['fetch_func'], **kwargs)
            data[key] = df
            
        return data
