"""
Unit tests for director_alpha.transform module.
"""
import pytest
import pandas as pd
import numpy as np
from director_alpha import transform


class TestNormalizeGvkey:
    """Tests for normalize_gvkey function."""
    
    def test_basic_normalization(self):
        """Test basic zero-padding to 6 digits."""
        series = pd.Series(['123', '1234', '12345', '123456'])
        result = transform.normalize_gvkey(series)
        
        expected = pd.Series(['000123', '001234', '012345', '123456'])
        pd.testing.assert_series_equal(result, expected)
    
    def test_handles_integer_input(self):
        """Test conversion from integer types."""
        series = pd.Series([123, 1234, 123456])
        result = transform.normalize_gvkey(series)
        
        expected = pd.Series(['000123', '001234', '123456'])
        pd.testing.assert_series_equal(result, expected)
    
    def test_handles_float_artifacts(self):
        """Test removal of trailing .0 from float conversion."""
        series = pd.Series(['123.0', '1234.0', '123456.0'])
        result = transform.normalize_gvkey(series)
        
        expected = pd.Series(['000123', '001234', '123456'])
        pd.testing.assert_series_equal(result, expected)
    
    def test_fill_na_parameter(self):
        """Test NA handling with fill_na parameter."""
        series = pd.Series(['123', None, '456'])
        result = transform.normalize_gvkey(series, fill_na='000000')
        
        expected = pd.Series(['000123', '000000', '000456'])
        pd.testing.assert_series_equal(result, expected)
    
    def test_na_without_fill(self):
        """Test behavior with NAs when fill_na is None."""
        series = pd.Series(['123', None, '456'])
        result = transform.normalize_gvkey(series)
        
        # NaN/None becomes 'None' string without fill_na
        assert result.iloc[0] == '000123'
        assert 'none' in result.iloc[1].lower()
        assert result.iloc[2] == '000456'


class TestNormalizeGvkeyColumns:
    """Tests for normalize_gvkey_columns helper."""
    
    def test_normalizes_specified_columns(self):
        """Test normalization of multiple columns."""
        df = pd.DataFrame({
            'gvkey': ['123', '456'],
            'target_gvkey': ['789', '012'],
            'other_col': ['abc', 'def'],
        })
        
        result = transform.normalize_gvkey_columns(df, ['gvkey', 'target_gvkey'])
        
        assert result['gvkey'].tolist() == ['000123', '000456']
        assert result['target_gvkey'].tolist() == ['000789', '000012']
        assert result['other_col'].tolist() == ['abc', 'def']  # Unchanged
    
    def test_handles_missing_columns(self):
        """Test graceful handling of missing columns."""
        df = pd.DataFrame({'gvkey': ['123', '456']})
        
        # Should not raise error for non-existent column
        result = transform.normalize_gvkey_columns(df, ['gvkey', 'missing_col'])
        
        assert 'gvkey' in result.columns
        assert result['gvkey'].tolist() == ['000123', '000456']


class TestCleanId:
    """Tests for clean_id function."""
    
    def test_removes_trailing_decimal(self):
        """Test removal of .0 suffix from IDs."""
        series = pd.Series(['12345.0', '67890.0', '11111'])
        result = transform.clean_id(series)
        
        expected = pd.Series(['12345', '67890', '11111'])
        pd.testing.assert_series_equal(result, expected)
    
    def test_preserves_non_decimal_strings(self):
        """Test that non-decimal strings are preserved."""
        series = pd.Series(['ABC123', 'DEF456'])
        result = transform.clean_id(series)
        
        expected = pd.Series(['ABC123', 'DEF456'])
        pd.testing.assert_series_equal(result, expected)


class TestNormalizeTicker:
    """Tests for normalize_ticker function."""
    
    def test_uppercases_and_strips(self):
        """Test uppercase conversion and whitespace stripping."""
        series = pd.Series(['  aapl  ', 'msft', '  GOOG'])
        result = transform.normalize_ticker(series)
        
        expected = pd.Series(['AAPL', 'MSFT', 'GOOG'])
        pd.testing.assert_series_equal(result, expected)


class TestIndustryAdjust:
    """Tests for industry_adjust function."""
    
    def test_basic_adjustment(self):
        """Test median subtraction by industry-year."""
        df = pd.DataFrame({
            'fyear': [2020, 2020, 2020, 2021, 2021, 2021],
            'sich': [3500, 3500, 3500, 3500, 3500, 3500],
            'roa': [0.05, 0.10, 0.15, 0.08, 0.12, 0.16],
        })
        
        result = transform.industry_adjust(df, cols=['roa'])
        
        # Median of [0.05, 0.10, 0.15] = 0.10
        # Adjusted: [-0.05, 0, 0.05]
        assert 'roa_adj' in result.columns
        assert result.loc[result['fyear'] == 2020, 'roa_adj'].iloc[1] == pytest.approx(0.0, abs=1e-10)
    
    def test_creates_sic2_from_sich(self):
        """Test automatic SIC2 creation from SICH."""
        df = pd.DataFrame({
            'fyear': [2020, 2020],
            'sich': [3570, 3580],
            'roa': [0.10, 0.12],
        })
        
        result = transform.industry_adjust(df, cols=['roa'])
        
        assert 'sic2' in result.columns
        assert result['sic2'].iloc[0] == 35


class TestWinsorization:
    """Tests for winsorization functions."""
    
    def test_winsorize_series(self):
        """Test basic series winsorization."""
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])  # 100 is outlier
        result = transform.winsorize_series(series, limits=(0.1, 0.9))
        
        # 100 should be clipped to 90th percentile
        assert result.max() < 100
        assert result.min() >= 1
    
    def test_apply_winsorization_grouped(self):
        """Test grouped winsorization by year."""
        df = pd.DataFrame({
            'fyear': [2020] * 10 + [2021] * 10,
            'roa': list(np.linspace(-0.5, 0.5, 10)) + list(np.linspace(-0.3, 0.7, 10)),
        })
        
        result = transform.apply_winsorization(df, cols=['roa'], group_col='fyear')
        
        # Values should be clipped within each year
        assert result['roa'].max() <= df['roa'].max()
        assert result['roa'].min() >= df['roa'].min()
    
    def test_apply_winsorization_ungrouped(self):
        """Test ungrouped (global) winsorization."""
        df = pd.DataFrame({
            'roa': [1, 2, 3, 4, 5, 6, 7, 8, 9, 100],
        })
        
        result = transform.apply_winsorization(df, cols=['roa'], group_col=None)
        
        assert result['roa'].max() < 100
