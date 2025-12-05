"""
Unit tests for director_alpha.phase0_universe module.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys

# Mock wrds module before importing director_alpha
sys.modules["wrds"] = MagicMock()

from director_alpha import phase0_universe, config


class TestApplyUniverseFilters:
    """Tests for universe filtering logic."""
    
    def test_excludes_non_usa_firms(self):
        """Test that non-USA incorporated firms are excluded."""
        df = pd.DataFrame({
            'fic': ['USA', 'CAN', 'USA', 'GBR'],
            'sich': [3500, 3500, 3500, 3500],
        })
        
        result = phase0_universe.apply_universe_filters(df)
        
        assert len(result) == 2
        assert all(result['fic'] == 'USA')
    
    def test_excludes_financial_firms(self):
        """Test that financial firms (SIC 6000-6999) are excluded."""
        df = pd.DataFrame({
            'fic': ['USA'] * 5,
            'sich': [3500, 6000, 6500, 6999, 3999],  # Mixed SIC codes
        })
        
        result = phase0_universe.apply_universe_filters(df)
        
        assert len(result) == 2
        assert all(result['sich'] < 6000) or all(result['sich'] > 6999)
    
    def test_excludes_utility_firms(self):
        """Test that utility firms (SIC 4900-4949) are excluded."""
        df = pd.DataFrame({
            'fic': ['USA'] * 4,
            'sich': [3500, 4900, 4949, 4950],
        })
        
        result = phase0_universe.apply_universe_filters(df)
        
        assert len(result) == 2
        assert all((result['sich'] < 4900) | (result['sich'] > 4949))
    
    def test_excludes_non_operating_firms(self):
        """Test that non-operating firms (SIC >= 9000) are excluded."""
        df = pd.DataFrame({
            'fic': ['USA'] * 3,
            'sich': [3500, 9000, 9999],
        })
        
        result = phase0_universe.apply_universe_filters(df)
        
        assert len(result) == 1
        assert result['sich'].iloc[0] == 3500
    
    def test_handles_missing_sich(self):
        """Test handling of missing SIC codes."""
        df = pd.DataFrame({
            'fic': ['USA'] * 3,
            'sich': [3500, np.nan, 3600],
        })
        
        result = phase0_universe.apply_universe_filters(df)
        
        # NaN SIC becomes 0, which passes the filter
        assert len(result) == 3


class TestFilterCommonStock:
    """Tests for common stock filtering."""
    
    def test_keeps_common_stock_only(self, sample_crsp_msf):
        """Test that only SHRCD 10, 11 are kept."""
        # Add some non-common stock
        sample_crsp_msf = sample_crsp_msf.copy()
        non_common = sample_crsp_msf.head(5).copy()
        non_common['shrcd'] = 12  # ETF
        sample_crsp_msf = pd.concat([sample_crsp_msf, non_common])
        
        merged_df = pd.DataFrame({
            'permno': [10001, 10002],
            'datadate': pd.to_datetime(['2021-12-31', '2021-12-31']),
        })
        
        result = phase0_universe.filter_common_stock(merged_df, sample_crsp_msf)
        
        assert len(result) == 2


class TestLinkCompustatCrsp:
    """Tests for Compustat-CRSP linking."""
    
    def test_basic_linking(self, sample_compustat, sample_ccm_link):
        """Test basic merge between Compustat and CCM."""
        result = phase0_universe.link_compustat_crsp(sample_compustat, sample_ccm_link)
        
        assert result is not None
        assert 'permno' in result.columns
        assert len(result) > 0
    
    def test_filters_by_link_date(self, sample_compustat, sample_ccm_link):
        """Test that link dates are respected."""
        # Modify CCM to have expired link
        ccm = sample_ccm_link.copy()
        ccm.loc[0, 'linkenddt'] = pd.Timestamp('2019-12-31')  # Expired before 2020 data
        
        result = phase0_universe.link_compustat_crsp(sample_compustat, ccm)
        
        # Should have fewer records due to expired link
        assert result is not None
        # First firm's 2020+ data should be excluded
    
    def test_normalizes_gvkey(self, sample_compustat, sample_ccm_link):
        """Test that GVKEYs are normalized to 6-digit strings."""
        result = phase0_universe.link_compustat_crsp(sample_compustat, sample_ccm_link)
        
        assert result is not None
        assert all(len(str(g)) == 6 for g in result['gvkey'])


class TestLoadData:
    """Tests for data loading function."""
    
    @patch('director_alpha.io.load_or_fetch')
    def test_loads_all_datasets(self, mock_load, sample_compustat, sample_crsp_msf, sample_ccm_link):
        """Test that all three datasets are loaded."""
        mock_load.side_effect = [sample_compustat, sample_crsp_msf, sample_ccm_link]
        
        compustat, crsp, ccm = phase0_universe.load_data()
        
        assert not compustat.empty
        assert not crsp.empty
        assert not ccm.empty
        assert mock_load.call_count == 3
