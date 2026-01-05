"""
Unit Tests for Data Pipeline Module.

Tests for data download, cleaning, alignment, and validation functions.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline import DataPipeline
from src.utils import validate_dataframe


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    close = 1.3000 + np.cumsum(np.random.randn(100) * 0.001)
    high = close + np.abs(np.random.randn(100) * 0.002)
    low = close - np.abs(np.random.randn(100) * 0.002)
    open_price = close + np.random.randn(100) * 0.001
    volume = np.random.randint(1000, 10000, 100)
    
    df = pd.DataFrame({
        'Open': open_price,
        'High': np.maximum.reduce([open_price, close, high]),
        'Low': np.minimum.reduce([open_price, close, low]),
        'Close': close,
        'Volume': volume
    }, index=dates)
    
    df.index.name = 'Date'
    return df


@pytest.fixture
def sample_combined_data(sample_ohlcv_data):
    """Create sample combined data with multiple tickers."""
    df = pd.DataFrame(index=sample_ohlcv_data.index)
    
    for ticker in ['GBPUSD', 'EURUSD', 'XAUUSD']:
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if ticker == 'XAUUSD' and col in ['Open', 'High', 'Low', 'Close']:
                # Gold has different price scale
                df[f'{ticker}_{col}'] = sample_ohlcv_data[col] * 1000
            else:
                df[f'{ticker}_{col}'] = sample_ohlcv_data[col]
    
    return df


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    return {
        'data': {
            'tickers': {
                'main': ['GBPUSD=X', 'EURUSD=X', 'XAUUSD=X'],
                'intermarket': ['DX-Y.NYB', '^VIX']
            },
            'period': '10y',
            'interval': '1d',
            'paths': {
                'raw': 'data/raw/',
                'processed': 'data/processed/',
                'combined': 'data/processed/combined_data.parquet',
                'features': 'data/processed/features.parquet'
            }
        },
        'features': {
            'technical': {},
            'intermarket': {'correlation_window': 20, 'lag_periods': [1, 5]},
            'multitimeframe': {'weekly_sma': [50, 200]}
        },
        'risk': {},
        'model': {},
        'signals': {},
        'logging': {'level': 'INFO', 'file': 'logs/test.log', 'format': '%(message)s'}
    }


# ============================================================================
# DATA VALIDATION TESTS
# ============================================================================

class TestDataValidation:
    """Tests for data validation functions."""
    
    def test_validate_dataframe_basic(self, sample_ohlcv_data):
        """Test basic DataFrame validation."""
        result = validate_dataframe(sample_ohlcv_data)
        
        assert result['is_valid'] is True
        assert result['row_count'] == 100
        assert result['column_count'] == 5
    
    def test_validate_dataframe_missing_columns(self, sample_ohlcv_data):
        """Test validation with missing required columns."""
        result = validate_dataframe(
            sample_ohlcv_data,
            required_columns=['Open', 'High', 'Low', 'Close', 'MissingColumn']
        )
        
        assert result['is_valid'] is False
        assert 'MissingColumn' in str(result['issues'])
    
    def test_validate_dataframe_ohlc_check(self, sample_ohlcv_data):
        """Test OHLC relationship validation."""
        result = validate_dataframe(sample_ohlcv_data, check_ohlc=True)
        
        assert result['is_valid'] is True
    
    def test_validate_dataframe_invalid_ohlc(self, sample_ohlcv_data):
        """Test validation catches invalid OHLC relationships."""
        # Corrupt data: set High below Close
        sample_ohlcv_data.loc[sample_ohlcv_data.index[0], 'High'] = (
            sample_ohlcv_data.loc[sample_ohlcv_data.index[0], 'Close'] - 0.01
        )
        
        result = validate_dataframe(sample_ohlcv_data, check_ohlc=True)
        
        assert result['is_valid'] is False
        assert any('High' in issue for issue in result['issues'])
    
    def test_validate_dataframe_missing_data(self):
        """Test validation catches high missing percentage."""
        df = pd.DataFrame({
            'A': [1, 2, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 9, 10],
            'B': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })
        
        result = validate_dataframe(df, max_missing_pct=5.0)
        
        assert result['is_valid'] is False


# ============================================================================
# DATA PIPELINE TESTS
# ============================================================================

class TestDataPipeline:
    """Tests for DataPipeline class."""
    
    @patch('src.data_pipeline.load_config')
    def test_pipeline_initialization(self, mock_load_config, mock_config):
        """Test DataPipeline initialization."""
        mock_load_config.return_value = mock_config
        
        pipeline = DataPipeline('config/config.yaml')
        
        assert len(pipeline.main_tickers) == 3
        assert 'GBPUSD=X' in pipeline.main_tickers
    
    @patch('src.data_pipeline.yf.Ticker')
    @patch('src.data_pipeline.load_config')
    def test_download_single_ticker(self, mock_load_config, mock_ticker, mock_config, sample_ohlcv_data):
        """Test single ticker download."""
        mock_load_config.return_value = mock_config
        
        # Mock yfinance Ticker
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = sample_ohlcv_data
        mock_ticker.return_value = mock_ticker_instance
        
        pipeline = DataPipeline('config/config.yaml')
        result = pipeline._download_single_ticker('GBPUSD=X', '10y', '1d')
        
        assert result is not None
        assert len(result) == 100
        assert 'Close' in result.columns
    
    def test_normalize_timezone(self, sample_ohlcv_data, mock_config):
        """Test timezone normalization."""
        with patch('src.data_pipeline.load_config', return_value=mock_config):
            pipeline = DataPipeline('config/config.yaml')
            
            # Add timezone to test data
            sample_ohlcv_data.index = sample_ohlcv_data.index.tz_localize('US/Eastern')
            
            result = pipeline._normalize_timezone(sample_ohlcv_data)
            
            assert result.index.tz is None
    
    def test_validate_ohlc(self, sample_ohlcv_data, mock_config):
        """Test OHLC validation."""
        with patch('src.data_pipeline.load_config', return_value=mock_config):
            pipeline = DataPipeline('config/config.yaml')
            
            issues = pipeline._validate_ohlc(sample_ohlcv_data, 'TEST')
            
            assert len(issues) == 0
    
    def test_validate_ohlc_catches_errors(self, sample_ohlcv_data, mock_config):
        """Test OHLC validation catches invalid relationships."""
        with patch('src.data_pipeline.load_config', return_value=mock_config):
            pipeline = DataPipeline('config/config.yaml')
            
            # Corrupt data
            sample_ohlcv_data.loc[sample_ohlcv_data.index[0], 'Low'] = (
                sample_ohlcv_data.loc[sample_ohlcv_data.index[0], 'High'] + 0.01
            )
            
            issues = pipeline._validate_ohlc(sample_ohlcv_data, 'TEST')
            
            assert len(issues) > 0
    
    def test_detect_outliers(self, sample_ohlcv_data, mock_config):
        """Test outlier detection."""
        with patch('src.data_pipeline.load_config', return_value=mock_config):
            pipeline = DataPipeline('config/config.yaml')
            
            # Add an extreme outlier
            sample_ohlcv_data.loc[sample_ohlcv_data.index[50], 'Close'] = (
                sample_ohlcv_data['Close'].mean() * 2
            )
            
            outliers = pipeline._detect_outliers(sample_ohlcv_data, 'TEST')
            
            # Should detect the outlier
            assert 'Close' in outliers
    
    def test_find_date_gaps(self, sample_ohlcv_data, mock_config):
        """Test date gap detection."""
        with patch('src.data_pipeline.load_config', return_value=mock_config):
            pipeline = DataPipeline('config/config.yaml')
            
            # Create a gap by dropping some rows
            df_with_gap = sample_ohlcv_data.drop(sample_ohlcv_data.index[20:30])
            
            gaps = pipeline._find_date_gaps(df_with_gap, max_gap_days=5)
            
            assert len(gaps) > 0
            assert gaps[0]['days'] == 10


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests (require network access)."""
    
    @pytest.mark.skip(reason="Requires network access")
    def test_download_real_data(self, mock_config):
        """Test downloading real data from yfinance."""
        with patch('src.data_pipeline.load_config', return_value=mock_config):
            pipeline = DataPipeline('config/config.yaml')
            
            df = pipeline._download_single_ticker('EURUSD=X', '1mo', '1d')
            
            assert df is not None
            assert len(df) > 0
            assert 'Close' in df.columns


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
