"""
Utility functions for the Forex Signal Model.

This module provides common utilities including:
- Logging setup with rotation
- Configuration loading and validation
- Date/time utilities for market hours
- Data validation functions
- Error handling decorators
"""

import logging
import os
import sys
import time
import functools
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from logging.handlers import RotatingFileHandler

import yaml
import pandas as pd
import numpy as np


# ============================================================================
# LOGGING UTILITIES
# ============================================================================

def setup_logging(
    log_file: Optional[str] = None,
    level: str = 'INFO',
    log_format: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up logging with console and rotating file handlers.
    
    Args:
        log_file: Path to log file. If None, logs to console only.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_format: Custom log format string.
        max_bytes: Maximum size of each log file before rotation.
        backup_count: Number of backup files to keep.
    
    Returns:
        Configured logger instance.
    
    Example:
        >>> logger = setup_logging('logs/forex_model.log', level='INFO')
        >>> logger.info("Pipeline started")
    """
    # Default format if not provided
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Get the root logger for the package
    logger = logging.getLogger('forex_signal_model')
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = 'forex_signal_model') -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Name of the logger (usually __name__).
    
    Returns:
        Logger instance.
    """
    return logging.getLogger(name)


# ============================================================================
# CONFIGURATION UTILITIES
# ============================================================================

def load_config(config_path: str = 'config/config.yaml') -> Dict[str, Any]:
    """
    Load and validate configuration from YAML file.
    
    Args:
        config_path: Path to the configuration YAML file.
    
    Returns:
        Dictionary containing configuration parameters.
    
    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is invalid YAML.
        ValueError: If required config keys are missing.
    
    Example:
        >>> config = load_config('config/config.yaml')
        >>> tickers = config['data']['tickers']['main']
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Validate required keys
    _validate_config(config)
    
    return config


def _validate_config(config: Dict[str, Any]) -> None:
    """
    Validate that all required configuration keys are present.
    
    Args:
        config: Configuration dictionary.
    
    Raises:
        ValueError: If required keys are missing.
    """
    required_sections = ['data', 'features', 'risk', 'model', 'signals', 'logging']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: '{section}'")
    
    # Validate data section
    if 'tickers' not in config['data']:
        raise ValueError("Missing 'tickers' in data configuration")
    
    if 'main' not in config['data']['tickers']:
        raise ValueError("Missing 'main' tickers in data configuration")


# ============================================================================
# DIRECTORY UTILITIES
# ============================================================================

def ensure_directories(config: Dict[str, Any], base_path: Optional[Path] = None) -> None:
    """
    Create all required directories from configuration.
    
    Args:
        config: Configuration dictionary.
        base_path: Base path for relative directories. Uses cwd if None.
    
    Example:
        >>> config = load_config()
        >>> ensure_directories(config)
    """
    if base_path is None:
        base_path = Path.cwd()
    else:
        base_path = Path(base_path)
    
    # Directories to create
    directories = [
        config['data']['paths']['raw'],
        config['data']['paths']['processed'],
        'logs',
        'results',
        'notebooks',
        'tests'
    ]
    
    for dir_path in directories:
        full_path = base_path / dir_path
        full_path.mkdir(parents=True, exist_ok=True)


# ============================================================================
# DATE/TIME UTILITIES
# ============================================================================

def get_market_hours(timezone: str = 'US/Eastern') -> Dict[str, Any]:
    """
    Get forex market trading hours information.
    
    Args:
        timezone: Timezone for market hours.
    
    Returns:
        Dictionary with market session information.
    
    Note:
        Forex market is open 24/5 (Sunday 5 PM to Friday 5 PM EST).
    """
    return {
        'timezone': timezone,
        'sessions': {
            'sydney': {'open': '17:00', 'close': '02:00'},
            'tokyo': {'open': '19:00', 'close': '04:00'},
            'london': {'open': '03:00', 'close': '12:00'},
            'new_york': {'open': '08:00', 'close': '17:00'}
        },
        'weekly_close': 'Friday 17:00',
        'weekly_open': 'Sunday 17:00'
    }


def is_market_open(timestamp: Optional[datetime] = None, timezone: str = 'US/Eastern') -> bool:
    """
    Check if forex market is currently open.
    
    Args:
        timestamp: Timestamp to check. Uses current time if None.
        timezone: Timezone for market hours.
    
    Returns:
        True if market is open, False otherwise.
    """
    try:
        import pytz
        tz = pytz.timezone(timezone)
    except ImportError:
        # Fallback without pytz
        return True  # Assume market is open
    
    if timestamp is None:
        timestamp = datetime.now(tz)
    elif timestamp.tzinfo is None:
        timestamp = tz.localize(timestamp)
    
    # Forex market: Sunday 5 PM to Friday 5 PM EST
    weekday = timestamp.weekday()
    hour = timestamp.hour
    
    # Saturday: closed
    if weekday == 5:
        return False
    
    # Sunday before 5 PM: closed
    if weekday == 6 and hour < 17:
        return False
    
    # Friday after 5 PM: closed
    if weekday == 4 and hour >= 17:
        return False
    
    return True


def get_trading_days(
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    exclude_weekends: bool = True
) -> List[datetime]:
    """
    Get list of trading days between two dates.
    
    Args:
        start_date: Start date.
        end_date: End date.
        exclude_weekends: Whether to exclude Saturday and Sunday.
    
    Returns:
        List of trading day datetime objects.
    """
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    if exclude_weekends:
        # Exclude Saturday (5) and Sunday (6)
        date_range = date_range[date_range.dayofweek < 5]
    
    return date_range.tolist()


# ============================================================================
# DATA VALIDATION UTILITIES
# ============================================================================

def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    check_ohlc: bool = False,
    max_missing_pct: float = 5.0
) -> Dict[str, Any]:
    """
    Validate a DataFrame for data quality issues.
    
    Args:
        df: DataFrame to validate.
        required_columns: List of columns that must be present.
        check_ohlc: Whether to validate OHLC relationships.
        max_missing_pct: Maximum allowed missing percentage per column.
    
    Returns:
        Dictionary with validation results.
    
    Example:
        >>> result = validate_dataframe(df, check_ohlc=True)
        >>> if not result['is_valid']:
        >>>     print(result['issues'])
    """
    result = {
        'is_valid': True,
        'issues': [],
        'missing_data': {},
        'row_count': len(df),
        'column_count': len(df.columns)
    }
    
    # Check required columns
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            result['is_valid'] = False
            result['issues'].append(f"Missing required columns: {missing_cols}")
    
    # Check missing values
    for col in df.columns:
        missing_pct = (df[col].isna().sum() / len(df)) * 100
        result['missing_data'][col] = missing_pct
        
        if missing_pct > max_missing_pct:
            result['is_valid'] = False
            result['issues'].append(
                f"Column '{col}' has {missing_pct:.2f}% missing values "
                f"(max allowed: {max_missing_pct}%)"
            )
    
    # Validate OHLC relationships
    if check_ohlc and all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
        ohlc_issues = _validate_ohlc_relationships(df)
        if ohlc_issues:
            result['is_valid'] = False
            result['issues'].extend(ohlc_issues)
    
    return result


def _validate_ohlc_relationships(df: pd.DataFrame) -> List[str]:
    """
    Validate OHLC price relationships.
    
    Args:
        df: DataFrame with OHLC columns.
    
    Returns:
        List of issues found.
    """
    issues = []
    
    # High should be >= all other prices
    high_violations = (
        (df['High'] < df['Open']) |
        (df['High'] < df['Close']) |
        (df['High'] < df['Low'])
    ).sum()
    if high_violations > 0:
        issues.append(f"High price violations: {high_violations} rows")
    
    # Low should be <= all other prices
    low_violations = (
        (df['Low'] > df['Open']) |
        (df['Low'] > df['Close']) |
        (df['Low'] > df['High'])
    ).sum()
    if low_violations > 0:
        issues.append(f"Low price violations: {low_violations} rows")
    
    # Check for negative prices
    negative_prices = (df[['Open', 'High', 'Low', 'Close']] < 0).any().any()
    if negative_prices:
        issues.append("Negative prices detected")
    
    return issues


def check_lookahead_bias(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: List[str]
) -> Dict[str, Any]:
    """
    Check for potential look-ahead bias in features.
    
    Args:
        df: DataFrame with features and target.
        target_column: Name of target column.
        feature_columns: List of feature column names.
    
    Returns:
        Dictionary with check results.
    """
    result = {
        'has_lookahead': False,
        'suspicious_features': [],
        'perfect_correlations': []
    }
    
    if target_column not in df.columns:
        return result
    
    target = df[target_column]
    
    for col in feature_columns:
        if col not in df.columns:
            continue
        
        # Check for perfect correlation (suspicious)
        try:
            corr = df[col].corr(target)
            if abs(corr) > 0.99:
                result['has_lookahead'] = True
                result['perfect_correlations'].append({
                    'feature': col,
                    'correlation': corr
                })
        except Exception:
            pass
    
    return result


# ============================================================================
# DECORATOR UTILITIES
# ============================================================================

def timer_decorator(func: Callable) -> Callable:
    """
    Decorator to measure and log execution time of functions.
    
    Args:
        func: Function to wrap.
    
    Returns:
        Wrapped function with timing.
    
    Example:
        >>> @timer_decorator
        >>> def process_data():
        >>>     ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger()
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"{func.__name__} completed in {elapsed:.2f} seconds")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"{func.__name__} failed after {elapsed:.2f} seconds: {e}")
            raise
    
    return wrapper


def retry_decorator(
    max_retries: int = 3,
    delay_base: float = 1.0,
    exceptions: tuple = (Exception,)
) -> Callable:
    """
    Decorator for retrying failed function calls with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts.
        delay_base: Base delay in seconds (exponential backoff).
        exceptions: Tuple of exceptions to catch and retry.
    
    Returns:
        Decorator function.
    
    Example:
        >>> @retry_decorator(max_retries=3)
        >>> def fetch_data():
        >>>     ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger()
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = delay_base * (2 ** attempt)
                        logger.warning(
                            f"{func.__name__} attempt {attempt + 1}/{max_retries} "
                            f"failed: {e}. Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_retries} attempts: {e}"
                        )
            
            raise last_exception
        
        return wrapper
    return decorator


def error_handler(
    default_return: Any = None,
    reraise: bool = False
) -> Callable:
    """
    Decorator for handling errors gracefully with logging.
    
    Args:
        default_return: Value to return on error (if not reraising).
        reraise: Whether to re-raise the exception after logging.
    
    Returns:
        Decorator function.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger()
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
                if reraise:
                    raise
                return default_return
        return wrapper
    return decorator


# ============================================================================
# NUMERIC UTILITIES
# ============================================================================

def safe_divide(
    numerator: Union[float, np.ndarray, pd.Series],
    denominator: Union[float, np.ndarray, pd.Series],
    fill_value: float = 0.0
) -> Union[float, np.ndarray, pd.Series]:
    """
    Safely divide two numbers, handling division by zero.
    
    Args:
        numerator: Numerator value(s).
        denominator: Denominator value(s).
        fill_value: Value to use when division by zero.
    
    Returns:
        Result of division with fill_value where division by zero occurs.
    """
    if isinstance(numerator, pd.Series) or isinstance(denominator, pd.Series):
        result = numerator / denominator
        result = result.replace([np.inf, -np.inf], fill_value)
        result = result.fillna(fill_value)
        return result
    elif isinstance(numerator, np.ndarray) or isinstance(denominator, np.ndarray):
        with np.errstate(divide='ignore', invalid='ignore'):
            result = numerator / denominator
            result = np.where(np.isfinite(result), result, fill_value)
        return result
    else:
        if denominator == 0:
            return fill_value
        return numerator / denominator


def normalize_series(
    series: pd.Series,
    method: str = 'zscore'
) -> pd.Series:
    """
    Normalize a pandas Series.
    
    Args:
        series: Series to normalize.
        method: Normalization method ('zscore', 'minmax', 'robust').
    
    Returns:
        Normalized series.
    """
    if method == 'zscore':
        mean = series.mean()
        std = series.std()
        if std == 0:
            return pd.Series(0, index=series.index)
        return (series - mean) / std
    
    elif method == 'minmax':
        min_val = series.min()
        max_val = series.max()
        if max_val == min_val:
            return pd.Series(0.5, index=series.index)
        return (series - min_val) / (max_val - min_val)
    
    elif method == 'robust':
        median = series.median()
        q75, q25 = series.quantile([0.75, 0.25])
        iqr = q75 - q25
        if iqr == 0:
            return pd.Series(0, index=series.index)
        return (series - median) / iqr
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


# ============================================================================
# FILE UTILITIES
# ============================================================================

def save_dataframe(
    df: pd.DataFrame,
    path: str,
    format: str = 'parquet',
    compression: str = 'snappy'
) -> None:
    """
    Save DataFrame to file with specified format.
    
    Args:
        df: DataFrame to save.
        path: Output file path.
        format: File format ('parquet', 'csv', 'pickle').
        compression: Compression for parquet ('snappy', 'gzip', None).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'parquet':
        df.to_parquet(path, compression=compression)
    elif format == 'csv':
        df.to_csv(path)
    elif format == 'pickle':
        df.to_pickle(path)
    else:
        raise ValueError(f"Unknown format: {format}")
    
    logger = get_logger()
    logger.info(f"Saved DataFrame ({len(df)} rows) to {path}")


def load_dataframe(path: str) -> pd.DataFrame:
    """
    Load DataFrame from file, auto-detecting format.
    
    Args:
        path: File path to load.
    
    Returns:
        Loaded DataFrame.
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    suffix = path.suffix.lower()
    
    if suffix == '.parquet':
        return pd.read_parquet(path)
    elif suffix == '.csv':
        return pd.read_csv(path, index_col=0, parse_dates=True)
    elif suffix in ['.pkl', '.pickle']:
        return pd.read_pickle(path)
    else:
        raise ValueError(f"Unknown file format: {suffix}")


# ============================================================================
# VERSION LOGGING
# ============================================================================

def log_versions() -> Dict[str, str]:
    """
    Log versions of key packages for reproducibility.
    
    Returns:
        Dictionary of package versions.
    """
    logger = get_logger()
    versions = {}
    
    packages = [
        'pandas', 'numpy', 'yfinance', 'pandas_ta',
        'sklearn', 'xgboost', 'lightgbm'
    ]
    
    for pkg in packages:
        try:
            module = __import__(pkg)
            version = getattr(module, '__version__', 'unknown')
            versions[pkg] = version
        except ImportError:
            versions[pkg] = 'not installed'
    
    logger.info(f"Package versions: {versions}")
    return versions
