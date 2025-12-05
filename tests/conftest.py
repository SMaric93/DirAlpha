"""
Pytest fixtures for Director Alpha tests.

Provides sample data generators and mock objects for both unit and integration testing.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys

# Mock wrds module before any director_alpha imports
sys.modules["wrds"] = MagicMock()


# =============================================================================
# Sample Data Generators
# =============================================================================

@pytest.fixture
def sample_compustat():
    """
    Sample Compustat Fundamentals Annual data.
    Returns a firm-year panel with 3 firms over 3 years.
    """
    np.random.seed(42)
    firms = ['001234', '002345', '003456']
    years = [2020, 2021, 2022]
    
    rows = []
    for gvkey in firms:
        for fyear in years:
            rows.append({
                'gvkey': gvkey,
                'fyear': fyear,
                'datadate': f'{fyear}-12-31',
                'fic': 'USA',
                'sich': 3570,  # Manufacturing
                'naics': '334111',
                'at': np.random.uniform(100, 1000),  # Total Assets
                'oibdp': np.random.uniform(10, 100),  # Operating Income
                'prcc_f': np.random.uniform(10, 100),  # Stock Price
                'csho': np.random.uniform(10, 50),  # Shares Outstanding
                'ceq': np.random.uniform(50, 500),  # Common Equity
                'dltt': np.random.uniform(0, 100),  # Long-term Debt
                'dlc': np.random.uniform(0, 50),  # Current Debt
                'xrd': np.random.uniform(0, 20),  # R&D
                'capx': np.random.uniform(5, 50),  # CAPEX
                'dv': np.random.uniform(0, 10),  # Dividends
            })
    
    df = pd.DataFrame(rows)
    df['datadate'] = pd.to_datetime(df['datadate'])
    return df


@pytest.fixture
def sample_compustat_financial():
    """Sample Compustat data for financial firms (should be excluded)."""
    return pd.DataFrame({
        'gvkey': ['004567'],
        'fyear': [2021],
        'datadate': ['2021-12-31'],
        'fic': 'USA',
        'sich': 6021,  # Financial - Commercial Banks
        'at': [500],
    })


@pytest.fixture
def sample_crsp_msf():
    """
    Sample CRSP Monthly Stock File data.
    Matches permnos to sample_compustat gvkeys via CCM.
    """
    np.random.seed(42)
    permnos = [10001, 10002, 10003]
    
    rows = []
    for permno in permnos:
        for year in [2020, 2021, 2022]:
            for month in range(1, 13):
                date = datetime(year, month, 15)
                rows.append({
                    'permno': permno,
                    'date': date,
                    'shrcd': 10,  # Common stock
                    'siccd': 3570,
                    'prc': np.random.uniform(10, 100),
                    'ret': np.random.uniform(-0.1, 0.1),
                })
    
    return pd.DataFrame(rows)


@pytest.fixture
def sample_crsp_dsf():
    """
    Sample CRSP Daily Stock File data for event studies.
    Provides ~2 years of daily returns for 2 firms.
    """
    np.random.seed(42)
    permnos = [10001, 10002]
    
    rows = []
    start_date = datetime(2020, 1, 2)
    
    for permno in permnos:
        for day in range(504):  # ~2 years of trading days
            date = start_date + timedelta(days=int(day * 1.4))  # Approximate trading days
            if date.weekday() < 5:  # Skip weekends
                rows.append({
                    'permno': permno,
                    'date': date,
                    'ret': np.random.normal(0.0005, 0.02),  # ~12% annual, 32% vol
                })
    
    return pd.DataFrame(rows)


@pytest.fixture
def sample_ccm_link():
    """
    Sample CRSP-Compustat Merged (CCM) link table.
    Links sample gvkeys to sample permnos.
    """
    return pd.DataFrame({
        'gvkey': ['001234', '002345', '003456'],
        'lpermno': [10001, 10002, 10003],
        'linkdt': pd.to_datetime(['2000-01-01'] * 3),
        'linkenddt': pd.to_datetime(['2099-12-31'] * 3),
        'linktype': ['LC'] * 3,
        'linkprim': ['P'] * 3,
    })


@pytest.fixture
def sample_execucomp():
    """
    Sample ExecuComp CEO data.
    Provides CEO tenure information for sample firms.
    """
    return pd.DataFrame({
        'gvkey': ['001234', '001234', '002345'],
        'year': [2020, 2021, 2021],
        'execid': [100, 101, 200],
        'exec_lname': ['Smith', 'Jones', 'Brown'],
        'exec_fname': ['John', 'Jane', 'Robert'],
        'pceo': ['CEO'] * 3,
        'ceoann': ['CEO'] * 3,
        'title': ['CEO', 'CEO', 'CEO'],
        'becameceo': pd.to_datetime(['2018-06-01', '2021-01-15', '2019-03-01']),
        'leftofc': pd.to_datetime([pd.NaT, pd.NaT, pd.NaT]),
        'joined_co': pd.to_datetime(['2015-01-01', '2020-01-01', '2010-01-01']),
        'age': [55, 48, 62],
        'gender': ['M', 'F', 'M'],
        'ticker': ['ABC', 'ABC', 'DEF'],
        'tdc1': [5000000, 6000000, 4500000],
    })


@pytest.fixture
def sample_ceo_spells():
    """Sample CEO spells (output of Phase 2)."""
    return pd.DataFrame({
        'spell_id': [1, 2, 3],
        'gvkey': ['001234', '001234', '002345'],
        'execid': [100, 101, 200],
        'appointment_date': pd.to_datetime(['2018-06-01', '2021-01-15', '2019-03-01']),
        'spell_end': pd.to_datetime([pd.NaT, pd.NaT, pd.NaT]),
        'is_external': [False, True, False],
        'ticker': ['ABC', 'ABC', 'DEF'],
    })


@pytest.fixture
def sample_boardex_directors():
    """Sample BoardEx director composition data."""
    return pd.DataFrame({
        'company_id': ['BX001', 'BX001', 'BX002', 'BX002'],
        'director_id': ['D100', 'D101', 'D200', 'D201'],
        'date_start': pd.to_datetime(['2015-01-01', '2017-06-01', '2016-01-01', '2018-06-01']),
        'date_end': pd.to_datetime([pd.NaT, pd.NaT, pd.NaT, pd.NaT]),
        'role_name': ['Director', 'Director', 'Director', 'Director'],
    })


@pytest.fixture
def sample_boardex_committees():
    """Sample BoardEx committee membership data."""
    return pd.DataFrame({
        'company_id': ['BX001', 'BX001', 'BX002'],
        'director_id': ['D100', 'D101', 'D200'],
        'committee_name': ['Nominating Committee', 'Audit Committee', 'Governance Committee'],
        'c_date_start': pd.to_datetime(['2015-01-01', '2017-06-01', '2016-01-01']),
        'c_date_end': pd.to_datetime([pd.NaT, pd.NaT, pd.NaT]),
    })


@pytest.fixture
def sample_ff5_factors():
    """Sample Fama-French 5-factor daily data."""
    np.random.seed(42)
    
    rows = []
    start_date = datetime(2020, 1, 2)
    
    for day in range(504):  # ~2 years
        date = start_date + timedelta(days=int(day * 1.4))
        if date.weekday() < 5:
            rows.append({
                'date': date,
                'mktrf': np.random.normal(0.0004, 0.01),
                'smb': np.random.normal(0.0001, 0.005),
                'hml': np.random.normal(0.0001, 0.005),
                'rmw': np.random.normal(0.0001, 0.004),
                'cma': np.random.normal(0.0001, 0.004),
                'rf': 0.0001,  # ~2.5% annual
            })
    
    return pd.DataFrame(rows)


@pytest.fixture
def sample_firm_year_base():
    """Sample Phase 0 output (firm-year base panel)."""
    np.random.seed(42)
    
    return pd.DataFrame({
        'gvkey': ['001234', '001234', '001234', '002345', '002345', '002345'],
        'fyear': [2020, 2021, 2022] * 2,
        'permno': [10001] * 3 + [10002] * 3,
        'datadate': pd.to_datetime(['2020-12-31', '2021-12-31', '2022-12-31'] * 2),
        'sich': [3570] * 6,
        'at': [100, 120, 150, 200, 220, 250],
        'oibdp': [10, 12, 15, 20, 22, 25],
        'prcc_f': [50, 55, 60, 40, 45, 50],
        'csho': [10, 10, 10, 20, 20, 20],
        'ceq': [50, 60, 70, 100, 110, 120],
        'dltt': [20, 25, 30, 40, 45, 50],
        'dlc': [5, 6, 7, 10, 11, 12],
        'xrd': [2, 3, 4, 5, 6, 7],
        'capx': [8, 9, 10, 15, 16, 17],
    })


# =============================================================================
# Mock Fixtures (for unit testing)
# =============================================================================

@pytest.fixture
def mock_wrds_connection():
    """Mock WRDS database connection."""
    with patch("director_alpha.db.get_db") as mock_get_db:
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db
        yield mock_db


@pytest.fixture
def mock_io_load_or_fetch():
    """Mock io.load_or_fetch to return provided data."""
    with patch("director_alpha.io.load_or_fetch") as mock_load:
        yield mock_load


# =============================================================================
# Temporary Directory Fixtures
# =============================================================================

@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data directory structure."""
    data_dir = tmp_path / "data"
    intermediate_dir = data_dir / "intermediate"
    
    data_dir.mkdir()
    intermediate_dir.mkdir()
    
    return {
        'data': data_dir,
        'intermediate': intermediate_dir,
    }


@pytest.fixture
def mock_config(temp_data_dir):
    """Mock config with temporary paths."""
    with patch.multiple(
        "director_alpha.config",
        DATA_DIR=temp_data_dir['data'],
        INTERMEDIATE_DIR=temp_data_dir['intermediate'],
        RAW_COMPUSTAT_PATH=temp_data_dir['data'] / "compustat.parquet",
        RAW_CRSP_PATH=temp_data_dir['data'] / "crsp.parquet",
        RAW_CCM_PATH=temp_data_dir['data'] / "ccm.parquet",
        FIRM_YEAR_BASE_PATH=temp_data_dir['intermediate'] / "firm_year_base.parquet",
        FIRM_YEAR_PERFORMANCE_PATH=temp_data_dir['intermediate'] / "firm_year_performance.parquet",
        CEO_SPELLS_PATH=temp_data_dir['intermediate'] / "ceo_spells.parquet",
    ):
        yield


# =============================================================================
# Integration Test Data Setup
# =============================================================================

@pytest.fixture
def setup_integration_data(temp_data_dir, sample_compustat, sample_crsp_msf, sample_ccm_link):
    """
    Write sample data to temporary files for integration testing.
    """
    # Write raw data files
    sample_compustat.to_parquet(temp_data_dir['data'] / "compustat.parquet")
    sample_crsp_msf.to_parquet(temp_data_dir['data'] / "crsp.parquet")
    sample_ccm_link.to_parquet(temp_data_dir['data'] / "ccm.parquet")
    
    return temp_data_dir
