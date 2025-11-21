import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
import sys
import os

# Mock wrds module
sys.modules["wrds"] = MagicMock()

from director_alpha import config
from director_alpha import phase0_universe
from director_alpha import phase1_performance
from director_alpha import phase2_ceo_turnover
from director_alpha import phase3_directors

@pytest.fixture
def mock_wrds_connection():
    with patch("director_alpha.config.get_wrds_connection") as mock_get_conn:
        mock_db = MagicMock()
        mock_get_conn.return_value = mock_db
        yield mock_db

@pytest.fixture
def mock_read_parquet():
    with patch("pandas.read_parquet") as mock_read:
        mock_read.side_effect = FileNotFoundError
        yield mock_read

@pytest.fixture
def mock_to_parquet():
    with patch("pandas.DataFrame.to_parquet") as mock_to:
        yield mock_to

def test_phase0_load_data_wrds(mock_wrds_connection, mock_read_parquet, mock_to_parquet):
    # Setup mock return values
    mock_wrds_connection.raw_sql.return_value = pd.DataFrame({'col': [1, 2]})
    
    # Run
    comp, crsp, ccm = phase0_universe.load_data()
    
    # Verify
    assert mock_wrds_connection.raw_sql.call_count == 3
    assert not comp.empty
    assert not crsp.empty
    assert not ccm.empty
    # Verify to_parquet was called but didn't write to disk
    assert mock_to_parquet.called

def test_phase1_load_crsp_wrds(mock_wrds_connection, mock_read_parquet, mock_to_parquet):
    # Setup mock return values
    mock_wrds_connection.raw_sql.return_value = pd.DataFrame({'permno': [1], 'date': ['2020-01-01'], 'ret': [0.1]})
    
    # Mock phase0 output loading (firm_year_base)
    # We need to patch read_parquet specifically for firm_year_base, but fail for raw_crsp
    # This is tricky with a single mock.
    # Let's patch pd.read_parquet to return a dummy DF for firm_year_base and raise Error for others.
    
    def side_effect(path):
        if "firm_year_base" in str(path):
            return pd.DataFrame({
                'gvkey': ['1'], 'fyear': [2020], 'permno': [1],
                'at': [100], 'oibdp': [10], 'prcc_f': [10], 'csho': [10], 'ceq': [50],
                'dltt': [20], 'dlc': [10], 'xrd': [0], 'capx': [5], 'sich': [1000]
            })
        raise FileNotFoundError
        
    mock_read_parquet.side_effect = side_effect
    
    # Run
    df = phase1_performance.run_phase1()
    
    # Verify
    # Should call raw_sql for CRSP
    assert mock_wrds_connection.raw_sql.called
    assert "SELECT permno" in mock_wrds_connection.raw_sql.call_args[0][0]

def test_phase2_load_execucomp_wrds(mock_wrds_connection, mock_read_parquet, mock_to_parquet):
    # Setup mock
    mock_wrds_connection.raw_sql.return_value = pd.DataFrame({
        'gvkey': ['1'], 'year': [2020], 'execid': ['1'], 'pceo': ['CEO'], 'ceoann': ['CEO'],
        'becameceo': ['2020-01-01'], 'joined_co': ['2019-01-01'], 'leftofc': [None],
        'title': ['CEO'], 'gender': ['M'], 'age': [50]
    })
    
    # Run
    df = phase2_ceo_turnover.run_phase2()
    
    # Verify
    assert mock_wrds_connection.raw_sql.called
    assert "SELECT gvkey" in mock_wrds_connection.raw_sql.call_args[0][0]
    assert not df.empty

def test_phase3_load_boardex_wrds(mock_wrds_connection, mock_read_parquet, mock_to_parquet):
    # Setup mock
    mock_wrds_connection.raw_sql.return_value = pd.DataFrame({'col': [1]})
    
    # Mock CEO spells loading
    def side_effect(path):
        if "ceo_spells" in str(path):
            return pd.DataFrame({'spell_id': [1], 'gvkey': ['1'], 'appointment_date': ['2020-01-01']})
        raise FileNotFoundError
    
    mock_read_parquet.side_effect = side_effect
    
    # Run
    df = phase3_directors.run_phase3()
    
    # Verify
    # Should call raw_sql 3 times (Directors, Committees, Link)
    assert mock_wrds_connection.raw_sql.call_count == 3
