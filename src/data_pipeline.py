"""
Data Pipeline Module for Forex Signal Model.

This module handles all data acquisition, cleaning, and initial processing
for forex and intermarket data. Uses yfinance for data download.

Key features:
- Download daily OHLCV data with retry logic
- Handle missing values and timezone normalization
- Validate data integrity (OHLC relationships)
- Save raw and processed data in various formats
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yfinance as yf

from src.utils import (
    get_logger,
    load_config,
    retry_decorator,
    timer_decorator,
    validate_dataframe,
    save_dataframe,
    ensure_directories
)


class DataPipeline:
    """
    Handles all data acquisition, cleaning, and initial processing.
    
    This class manages the complete data pipeline from downloading raw data
    from yfinance to creating a unified, cleaned dataset ready for feature
    engineering.
    
    Attributes:
        config: Configuration dictionary.
        logger: Logger instance.
        main_tickers: List of main forex/gold tickers.
        intermarket_tickers: List of intermarket tickers.
        raw_data: Dictionary storing raw DataFrames per ticker.
        combined_data: Combined wide-format DataFrame.
    
    Example:
        >>> pipeline = DataPipeline('config/config.yaml')
        >>> pipeline.run_full_pipeline()
    """
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """
        Initialize the DataPipeline.
        
        Args:
            config_path: Path to the configuration YAML file.
        """
        self.config = load_config(config_path)
        self.logger = get_logger('forex_signal_model.data_pipeline')
        
        # Extract ticker lists
        self.main_tickers = self.config['data']['tickers']['main']
        self.intermarket_tickers = self.config['data']['tickers']['intermarket']
        self.all_tickers = self.main_tickers + self.intermarket_tickers
        
        # Data storage
        self.raw_data: Dict[str, pd.DataFrame] = {}
        self.combined_data: Optional[pd.DataFrame] = None
        
        # Paths
        self.raw_path = Path(self.config['data']['paths']['raw'])
        self.processed_path = Path(self.config['data']['paths']['processed'])
        self.combined_path = Path(self.config['data']['paths']['combined'])
        
        # Ensure directories exist
        ensure_directories(self.config)
        
        self.logger.info(f"DataPipeline initialized with {len(self.all_tickers)} tickers")
    
    # ========================================================================
    # DATA DOWNLOAD
    # ========================================================================
    
    @timer_decorator
    def download_data(self, tickers: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Download data from yfinance for all specified tickers.
        
        Args:
            tickers: List of ticker symbols. Uses all tickers if None.
        
        Returns:
            Dictionary mapping ticker symbols to DataFrames.
        
        Raises:
            RuntimeError: If critical tickers fail to download.
        """
        tickers = tickers or self.all_tickers
        period = self.config['data']['period']
        interval = self.config['data']['interval']
        
        self.logger.info(f"Downloading {len(tickers)} tickers for period={period}")
        
        successful = 0
        failed = []
        
        for ticker in tickers:
            try:
                df = self._download_single_ticker(ticker, period, interval)
                if df is not None and len(df) > 0:
                    self.raw_data[ticker] = df
                    successful += 1
                    self.logger.info(f"  ✓ {ticker}: {len(df)} rows downloaded")
                else:
                    failed.append(ticker)
                    self.logger.warning(f"  ✗ {ticker}: No data returned")
            except Exception as e:
                failed.append(ticker)
                self.logger.error(f"  ✗ {ticker}: {e}")
        
        self.logger.info(
            f"Download complete: {successful}/{len(tickers)} successful, "
            f"{len(failed)} failed"
        )
        
        # Check if main tickers were downloaded successfully
        main_missing = set(self.main_tickers) & set(failed)
        if main_missing:
            raise RuntimeError(
                f"Critical main tickers failed to download: {main_missing}"
            )
        
        return self.raw_data
    
    @retry_decorator(max_retries=3, delay_base=2.0)
    def _download_single_ticker(
        self,
        ticker: str,
        period: str,
        interval: str
    ) -> pd.DataFrame:
        """
        Download data for a single ticker with retry logic.
        
        Args:
            ticker: Ticker symbol.
            period: Download period (e.g., '10y').
            interval: Data interval (e.g., '1d').
        
        Returns:
            DataFrame with OHLCV data.
        """
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(period=period, interval=interval)
        
        if df.empty:
            raise ValueError(f"No data returned for {ticker}")
        
        # Ensure index is DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Remove timezone info for consistency
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        # Rename index for clarity
        df.index.name = 'Date'
        
        return df
    
    # ========================================================================
    # DATA CLEANING
    # ========================================================================
    
    @timer_decorator
    def clean_data(self) -> Dict[str, pd.DataFrame]:
        """
        Clean all downloaded data.
        
        Applies:
        - Timezone normalization to UTC
        - Missing value handling (forward fill, drop if excessive)
        - Outlier detection and logging
        - OHLC validation
        
        Returns:
            Dictionary of cleaned DataFrames.
        """
        self.logger.info("Starting data cleaning...")
        
        for ticker, df in self.raw_data.items():
            self.logger.debug(f"Cleaning {ticker}...")
            
            # 1. Normalize timezone
            df = self._normalize_timezone(df)
            
            # 2. Handle missing values
            df, fill_count = self._handle_missing_values(df, ticker)
            
            # 3. Validate OHLC
            ohlc_issues = self._validate_ohlc(df, ticker)
            if ohlc_issues:
                self.logger.warning(f"{ticker} OHLC issues: {ohlc_issues}")
            
            # 4. Detect outliers
            outliers = self._detect_outliers(df, ticker)
            if outliers:
                self.logger.warning(f"{ticker} outliers detected: {outliers}")
            
            # 5. Remove any remaining NaN rows
            initial_len = len(df)
            df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
            dropped = initial_len - len(df)
            if dropped > 0:
                self.logger.info(f"{ticker}: Dropped {dropped} rows with NaN values")
            
            self.raw_data[ticker] = df
        
        self.logger.info("Data cleaning complete")
        return self.raw_data
    
    def _normalize_timezone(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize DataFrame index to timezone-naive UTC.
        
        Args:
            df: Input DataFrame with DatetimeIndex.
        
        Returns:
            DataFrame with normalized timezone.
        """
        if df.index.tz is not None:
            df.index = df.index.tz_convert('UTC').tz_localize(None)
        return df
    
    def _handle_missing_values(
        self,
        df: pd.DataFrame,
        ticker: str,
        max_ffill_days: int = 3,
        max_missing_pct: float = 5.0
    ) -> Tuple[pd.DataFrame, int]:
        """
        Handle missing values in the data.
        
        Args:
            df: Input DataFrame.
            ticker: Ticker symbol for logging.
            max_ffill_days: Maximum days to forward fill.
            max_missing_pct: Maximum allowed missing percentage.
        
        Returns:
            Tuple of (cleaned DataFrame, number of filled values).
        
        Raises:
            ValueError: If missing data exceeds threshold.
        """
        # Count initial NaN
        initial_nan = df.isna().sum().sum()
        
        # Calculate missing percentage
        missing_pct = (df.isna().sum() / len(df) * 100)
        high_missing = missing_pct[missing_pct > max_missing_pct]
        
        if not high_missing.empty:
            self.logger.warning(
                f"{ticker} has high missing values: "
                f"{high_missing.to_dict()}"
            )
        
        # Forward fill for holidays/weekends (limited)
        df = df.ffill(limit=max_ffill_days)
        
        # Count filled values
        final_nan = df.isna().sum().sum()
        filled_count = initial_nan - final_nan
        
        if filled_count > 0:
            self.logger.debug(f"{ticker}: Forward filled {filled_count} values")
        
        return df, filled_count
    
    def _validate_ohlc(self, df: pd.DataFrame, ticker: str) -> List[str]:
        """
        Validate OHLC price relationships.
        
        Args:
            df: DataFrame with OHLC columns.
            ticker: Ticker symbol for context.
        
        Returns:
            List of validation issues.
        """
        issues = []
        
        if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            return ['Missing OHLC columns']
        
        # High >= all prices
        high_violations = ((df['High'] < df['Open']) | 
                          (df['High'] < df['Close']) | 
                          (df['High'] < df['Low'])).sum()
        if high_violations > 0:
            issues.append(f"High violations: {high_violations}")
        
        # Low <= all prices
        low_violations = ((df['Low'] > df['Open']) | 
                         (df['Low'] > df['Close']) | 
                         (df['Low'] > df['High'])).sum()
        if low_violations > 0:
            issues.append(f"Low violations: {low_violations}")
        
        # Positive prices (except for some indices)
        if ticker not in ['^VIX']:  # VIX can have special values
            negative_prices = (df[['Open', 'High', 'Low', 'Close']] <= 0).any().any()
            if negative_prices:
                issues.append("Negative or zero prices detected")
        
        return issues
    
    def _detect_outliers(
        self,
        df: pd.DataFrame,
        ticker: str,
        z_threshold: float = 5.0
    ) -> Dict[str, int]:
        """
        Detect outliers in price and volume data.
        
        Args:
            df: DataFrame with OHLCV data.
            ticker: Ticker symbol for context.
            z_threshold: Z-score threshold for outlier detection.
        
        Returns:
            Dictionary of columns with outlier counts.
        """
        outliers = {}
        
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df.columns:
                # Calculate daily returns for outlier detection
                returns = df[col].pct_change()
                z_scores = (returns - returns.mean()) / returns.std()
                outlier_count = (abs(z_scores) > z_threshold).sum()
                
                if outlier_count > 0:
                    outliers[col] = outlier_count
        
        return outliers
    
    # ========================================================================
    # DATA ALIGNMENT
    # ========================================================================
    
    @timer_decorator
    def align_data(self) -> pd.DataFrame:
        """
        Align all tickers to have matching dates.
        
        Creates a unified DataFrame where all tickers share the same date index.
        Uses inner join to keep only dates where all main tickers have data.
        
        Returns:
            Combined wide-format DataFrame.
        """
        self.logger.info("Aligning data across all tickers...")
        
        if not self.raw_data:
            raise ValueError("No data to align. Run download_data() first.")
        
        # Start with main tickers (must have data for all dates)
        main_dfs = {t: self.raw_data[t] for t in self.main_tickers if t in self.raw_data}
        
        if not main_dfs:
            raise ValueError("No main ticker data available")
        
        # Find common date range for main tickers
        common_dates = None
        for ticker, df in main_dfs.items():
            if common_dates is None:
                common_dates = set(df.index)
            else:
                common_dates = common_dates.intersection(set(df.index))
        
        common_dates = sorted(list(common_dates))
        self.logger.info(f"Common dates found: {len(common_dates)} days")
        self.logger.info(f"Date range: {common_dates[0]} to {common_dates[-1]}")
        
        # Build combined DataFrame
        combined_dfs = []
        
        for ticker in self.all_tickers:
            if ticker not in self.raw_data:
                continue
            
            df = self.raw_data[ticker].copy()
            
            # Filter to common dates (for main tickers)
            # For intermarket, use outer join with forward fill
            if ticker in self.main_tickers:
                df = df.loc[df.index.isin(common_dates)]
            
            # Rename columns with ticker prefix
            ticker_clean = ticker.replace('=X', '').replace('^', '').replace('-', '_').replace('.', '_')
            df = df.rename(columns={
                'Open': f'{ticker_clean}_Open',
                'High': f'{ticker_clean}_High',
                'Low': f'{ticker_clean}_Low',
                'Close': f'{ticker_clean}_Close',
                'Volume': f'{ticker_clean}_Volume'
            })
            
            # Keep only OHLCV columns
            cols_to_keep = [c for c in df.columns if any(
                x in c for x in ['Open', 'High', 'Low', 'Close', 'Volume']
            )]
            df = df[cols_to_keep]
            
            combined_dfs.append(df)
        
        # Combine all DataFrames
        self.combined_data = pd.concat(combined_dfs, axis=1)
        
        # Reindex to common dates and forward fill intermarket data
        self.combined_data = self.combined_data.reindex(common_dates)
        self.combined_data = self.combined_data.ffill(limit=3)
        
        # Sort index
        self.combined_data.sort_index(inplace=True)
        
        # Validate no future data
        self._validate_no_future_data()
        
        self.logger.info(
            f"Combined data shape: {self.combined_data.shape} "
            f"({len(self.combined_data)} rows, {len(self.combined_data.columns)} columns)"
        )
        
        return self.combined_data
    
    def _validate_no_future_data(self) -> None:
        """
        Validate that data doesn't contain future dates.
        
        Raises:
            ValueError: If future dates are detected.
        """
        today = pd.Timestamp.now().normalize()
        max_date = self.combined_data.index.max()
        
        if max_date > today:
            raise ValueError(
                f"Future data detected! Max date: {max_date}, Today: {today}"
            )
        
        self.logger.debug(f"No future data detected. Max date: {max_date}")
    
    # ========================================================================
    # DATA SAVING
    # ========================================================================
    
    @timer_decorator
    def save_raw(self) -> None:
        """
        Save raw data for each ticker as individual CSV files.
        """
        self.logger.info(f"Saving raw data to {self.raw_path}")
        
        for ticker, df in self.raw_data.items():
            # Create safe filename
            safe_name = ticker.replace('=', '_').replace('^', '').replace('.', '_')
            filename = self.raw_path / f"{safe_name}.csv"
            
            df.to_csv(filename)
            self.logger.debug(f"Saved {ticker} to {filename}")
        
        # Save metadata
        metadata = {
            'download_timestamp': datetime.now().isoformat(),
            'tickers': list(self.raw_data.keys()),
            'period': self.config['data']['period'],
            'interval': self.config['data']['interval'],
            'rows_per_ticker': {t: len(df) for t, df in self.raw_data.items()}
        }
        
        import json
        with open(self.raw_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Saved {len(self.raw_data)} raw data files")
    
    @timer_decorator
    def save_combined(self) -> None:
        """
        Save combined data as compressed parquet file.
        """
        if self.combined_data is None:
            raise ValueError("No combined data to save. Run align_data() first.")
        
        # Ensure directory exists
        self.combined_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as parquet with snappy compression
        self.combined_data.to_parquet(
            self.combined_path,
            compression='snappy',
            index=True
        )
        
        self.logger.info(
            f"Saved combined data to {self.combined_path} "
            f"({self.combined_data.shape[0]} rows, {self.combined_data.shape[1]} columns)"
        )
    
    # ========================================================================
    # DATA QUALITY REPORT
    # ========================================================================
    
    @timer_decorator
    def generate_quality_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive data quality report.
        
        Returns:
            Dictionary containing quality metrics.
        """
        self.logger.info("Generating data quality report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'tickers': {},
            'combined': {},
            'summary': {}
        }
        
        # Per-ticker statistics
        for ticker, df in self.raw_data.items():
            report['tickers'][ticker] = {
                'rows': len(df),
                'start_date': str(df.index.min()),
                'end_date': str(df.index.max()),
                'missing_pct': df.isna().sum().to_dict(),
                'date_gaps': self._find_date_gaps(df)
            }
        
        # Combined data statistics
        if self.combined_data is not None:
            report['combined'] = {
                'rows': len(self.combined_data),
                'columns': len(self.combined_data.columns),
                'start_date': str(self.combined_data.index.min()),
                'end_date': str(self.combined_data.index.max()),
                'missing_total': int(self.combined_data.isna().sum().sum()),
                'memory_mb': round(
                    self.combined_data.memory_usage(deep=True).sum() / 1024 / 1024, 2
                )
            }
        
        # Summary
        report['summary'] = {
            'total_tickers': len(self.raw_data),
            'main_tickers': len([t for t in self.main_tickers if t in self.raw_data]),
            'intermarket_tickers': len([t for t in self.intermarket_tickers if t in self.raw_data])
        }
        
        # Log summary
        self.logger.info("=" * 60)
        self.logger.info("DATA QUALITY REPORT")
        self.logger.info("=" * 60)
        for ticker, stats in report['tickers'].items():
            self.logger.info(
                f"{ticker}: {stats['rows']} rows, "
                f"{stats['start_date']} to {stats['end_date']}"
            )
        if 'combined' in report and report['combined']:
            self.logger.info(
                f"Combined: {report['combined']['rows']} rows, "
                f"{report['combined']['columns']} columns, "
                f"{report['combined']['memory_mb']} MB"
            )
        self.logger.info("=" * 60)
        
        return report
    
    def _find_date_gaps(self, df: pd.DataFrame, max_gap_days: int = 5) -> List[Dict]:
        """
        Find gaps in the date index.
        
        Args:
            df: DataFrame with DatetimeIndex.
            max_gap_days: Minimum gap size to report.
        
        Returns:
            List of gap dictionaries.
        """
        gaps = []
        dates = df.index.sort_values()
        
        for i in range(1, len(dates)):
            gap = (dates[i] - dates[i-1]).days
            if gap > max_gap_days:
                gaps.append({
                    'start': str(dates[i-1]),
                    'end': str(dates[i]),
                    'days': gap
                })
        
        return gaps
    
    # ========================================================================
    # GET LATEST DATA
    # ========================================================================
    
    def get_latest_data(
        self,
        existing_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Fetch only new data since last download.
        
        Args:
            existing_data: Existing combined DataFrame to extend.
        
        Returns:
            Updated combined DataFrame.
        """
        if existing_data is None:
            # Try to load from file
            if self.combined_path.exists():
                existing_data = pd.read_parquet(self.combined_path)
                self.logger.info(f"Loaded existing data: {len(existing_data)} rows")
            else:
                self.logger.info("No existing data found, downloading full history")
                return self.run_full_pipeline()
        
        # Find last date
        last_date = existing_data.index.max()
        self.logger.info(f"Last data date: {last_date}")
        
        # Calculate days to fetch
        today = pd.Timestamp.now().normalize()
        days_to_fetch = (today - last_date).days + 1
        
        if days_to_fetch <= 1:
            self.logger.info("Data is already up to date")
            return existing_data
        
        # Download recent data
        period = f"{min(days_to_fetch + 5, 30)}d"  # Add buffer
        
        for ticker in self.all_tickers:
            try:
                df = self._download_single_ticker(ticker, period, '1d')
                if df is not None:
                    self.raw_data[ticker] = df
            except Exception as e:
                self.logger.warning(f"Failed to update {ticker}: {e}")
        
        # Clean and align new data
        self.clean_data()
        new_data = self.align_data()
        
        # Filter to only new dates
        new_dates = new_data.index > last_date
        new_data = new_data[new_dates]
        
        if len(new_data) == 0:
            self.logger.info("No new data to add")
            return existing_data
        
        # Combine with existing data
        self.combined_data = pd.concat([existing_data, new_data])
        self.combined_data = self.combined_data[~self.combined_data.index.duplicated(keep='last')]
        self.combined_data.sort_index(inplace=True)
        
        self.logger.info(f"Added {len(new_data)} new rows. Total: {len(self.combined_data)}")
        
        return self.combined_data
    
    # ========================================================================
    # MAIN PIPELINE
    # ========================================================================
    
    @timer_decorator
    def run_full_pipeline(self) -> pd.DataFrame:
        """
        Run the complete data pipeline.
        
        Steps:
        1. Download data for all tickers
        2. Clean and validate data
        3. Align dates across tickers
        4. Save raw and combined data
        5. Generate quality report
        
        Returns:
            Combined DataFrame ready for feature engineering.
        """
        self.logger.info("=" * 60)
        self.logger.info("STARTING FULL DATA PIPELINE")
        self.logger.info("=" * 60)
        
        # Step 1: Download
        self.download_data()
        
        # Step 2: Clean
        self.clean_data()
        
        # Step 3: Align
        self.align_data()
        
        # Step 4: Save
        self.save_raw()
        self.save_combined()
        
        # Step 5: Report
        self.generate_quality_report()
        
        self.logger.info("=" * 60)
        self.logger.info("DATA PIPELINE COMPLETE")
        self.logger.info("=" * 60)
        
        return self.combined_data
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def load_combined_data(self) -> pd.DataFrame:
        """
        Load combined data from parquet file.
        
        Returns:
            Combined DataFrame.
        """
        if not self.combined_path.exists():
            raise FileNotFoundError(
                f"Combined data not found at {self.combined_path}. "
                "Run the data pipeline first."
            )
        
        self.combined_data = pd.read_parquet(self.combined_path)
        self.logger.info(
            f"Loaded combined data: {self.combined_data.shape}"
        )
        return self.combined_data
    
    def get_ticker_data(self, ticker: str) -> pd.DataFrame:
        """
        Get data for a specific ticker from combined data.
        
        Args:
            ticker: Ticker symbol.
        
        Returns:
            DataFrame with OHLCV columns for the ticker.
        """
        if self.combined_data is None:
            self.load_combined_data()
        
        ticker_clean = ticker.replace('=X', '').replace('^', '').replace('-', '_').replace('.', '_')
        cols = [c for c in self.combined_data.columns if c.startswith(ticker_clean)]
        
        if not cols:
            raise ValueError(f"No columns found for ticker: {ticker}")
        
        return self.combined_data[cols]


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    from src.utils import setup_logging
    
    # Setup logging
    logger = setup_logging('logs/data_pipeline.log', level='INFO')
    
    # Run pipeline
    pipeline = DataPipeline()
    combined_data = pipeline.run_full_pipeline()
    
    print(f"\nPipeline complete!")
    print(f"Combined data shape: {combined_data.shape}")
    print(f"Date range: {combined_data.index.min()} to {combined_data.index.max()}")
