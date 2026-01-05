"""
Main Execution Script for Forex Signal Model.

This script orchestrates the complete pipeline for downloading data,
engineering features, and preparing data for model training.

Usage:
    python main.py --mode download    # Download fresh data
    python main.py --mode features    # Generate features
    python main.py --mode all         # Full pipeline
    python main.py --mode update      # Update with latest data only

Example:
    python main.py --mode all --config config/config.yaml
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import setup_logging, load_config, timer_decorator, log_versions
from src.data_pipeline import DataPipeline
from src.feature_engineering import FeatureEngine


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description='Forex Signal Model - Data Pipeline and Feature Engineering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run complete pipeline (download + features)
    python main.py --mode all

    # Download data only
    python main.py --mode download

    # Generate features from existing data
    python main.py --mode features

    # Update with latest data
    python main.py --mode update

    # Use custom config file
    python main.py --mode all --config path/to/config.yaml
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['download', 'features', 'all', 'update'],
        default='all',
        help='Pipeline mode to run (default: all)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose (DEBUG) logging'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without executing'
    )
    
    return parser.parse_args()


@timer_decorator
def run_download_pipeline(config_path: str) -> None:
    """
    Run the data download and cleaning pipeline.
    
    Args:
        config_path: Path to configuration file.
    """
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("STARTING DATA DOWNLOAD PIPELINE")
    logger.info("=" * 60)
    
    pipeline = DataPipeline(config_path)
    combined_data = pipeline.run_full_pipeline()
    
    logger.info(f"Download complete: {combined_data.shape}")
    logger.info(f"Date range: {combined_data.index.min()} to {combined_data.index.max()}")


@timer_decorator
def run_feature_pipeline(config_path: str) -> None:
    """
    Run the feature engineering pipeline.
    
    Args:
        config_path: Path to configuration file.
    """
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("STARTING FEATURE ENGINEERING PIPELINE")
    logger.info("=" * 60)
    
    # Load combined data
    pipeline = DataPipeline(config_path)
    
    try:
        combined_data = pipeline.load_combined_data()
    except FileNotFoundError:
        logger.error("Combined data not found. Run download pipeline first.")
        logger.info("Run: python main.py --mode download")
        raise
    
    # Run feature engineering
    engine = FeatureEngine(config_path)
    features = engine.run_full_pipeline(combined_data)
    
    logger.info(f"Features generated: {features.shape}")
    logger.info(f"Date range: {features.index.min()} to {features.index.max()}")
    
    # Print feature statistics
    _print_feature_summary(features, engine)


@timer_decorator
def run_update_pipeline(config_path: str) -> None:
    """
    Update data with latest available data.
    
    Args:
        config_path: Path to configuration file.
    """
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("STARTING DATA UPDATE PIPELINE")
    logger.info("=" * 60)
    
    pipeline = DataPipeline(config_path)
    updated_data = pipeline.get_latest_data()
    
    if updated_data is not None:
        pipeline.save_combined()
        
        # Re-run feature engineering
        engine = FeatureEngine(config_path)
        features = engine.run_full_pipeline(updated_data)
        
        logger.info(f"Update complete: {features.shape}")
    else:
        logger.info("Data is already up to date")


@timer_decorator
def run_full_pipeline(config_path: str) -> None:
    """
    Run the complete pipeline (download + features).
    
    Args:
        config_path: Path to configuration file.
    """
    logger = setup_logging()
    start_time = datetime.now()
    
    logger.info("=" * 60)
    logger.info("STARTING FULL PIPELINE")
    logger.info(f"Start time: {start_time.isoformat()}")
    logger.info("=" * 60)
    
    # Log package versions for reproducibility
    log_versions()
    
    # Step 1: Download and process data
    logger.info("\n" + "=" * 40)
    logger.info("STEP 1: DATA DOWNLOAD AND PROCESSING")
    logger.info("=" * 40)
    
    pipeline = DataPipeline(config_path)
    combined_data = pipeline.run_full_pipeline()
    
    # Step 2: Feature engineering
    logger.info("\n" + "=" * 40)
    logger.info("STEP 2: FEATURE ENGINEERING")
    logger.info("=" * 40)
    
    engine = FeatureEngine(config_path)
    features = engine.run_full_pipeline(combined_data)
    
    # Final summary
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    
    logger.info("\n" + "=" * 60)
    logger.info("FULL PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    logger.info(f"Combined data: {combined_data.shape}")
    logger.info(f"Feature matrix: {features.shape}")
    logger.info(f"Date range: {features.index.min()} to {features.index.max()}")
    
    # Print feature summary
    _print_feature_summary(features, engine)
    
    # Generate summary report
    _generate_summary_report(config_path, combined_data, features, elapsed)


def _print_feature_summary(features, engine: FeatureEngine) -> None:
    """
    Print summary of generated features.
    
    Args:
        features: Feature DataFrame.
        engine: FeatureEngine instance.
    """
    logger = setup_logging()
    
    # Count features by category
    categories = {}
    for name, meta in engine.feature_metadata.items():
        cat = meta.get('category', 'unknown')
        categories[cat] = categories.get(cat, 0) + 1
    
    logger.info("\nFeature categories:")
    for cat, count in sorted(categories.items()):
        logger.info(f"  {cat}: {count}")
    
    # Target distribution
    target_cols = [c for c in features.columns if 'Target_Direction' in c]
    for col in target_cols:
        ticker = col.replace('_Target_Direction', '')
        up_pct = features[col].mean() * 100
        logger.info(f"\n{ticker} target distribution:")
        logger.info(f"  Up days: {up_pct:.1f}%")
        logger.info(f"  Down days: {100 - up_pct:.1f}%")


def _generate_summary_report(
    config_path: str,
    combined_data,
    features,
    elapsed: float
) -> None:
    """
    Generate and save a summary report.
    
    Args:
        config_path: Config file path.
        combined_data: Combined price data.
        features: Feature DataFrame.
        elapsed: Execution time in seconds.
    """
    report_path = Path('results/pipeline_report.txt')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("FOREX SIGNAL MODEL - PIPELINE REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Execution time: {elapsed:.1f} seconds\n\n")
        
        f.write("DATA SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Combined data shape: {combined_data.shape}\n")
        f.write(f"Feature matrix shape: {features.shape}\n")
        f.write(f"Date range: {features.index.min()} to {features.index.max()}\n")
        f.write(f"Total trading days: {len(features)}\n\n")
        
        f.write("FEATURE COLUMNS\n")
        f.write("-" * 40 + "\n")
        for i, col in enumerate(features.columns, 1):
            f.write(f"  {i:3d}. {col}\n")
        
        f.write("\n")
        f.write("=" * 60 + "\n")
        f.write("END OF REPORT\n")
    
    logger = setup_logging()
    logger.info(f"Summary report saved to: {report_path}")


def main() -> int:
    """
    Main entry point.
    
    Returns:
        Exit code (0 for success, 1 for failure).
    """
    args = parse_arguments()
    
    # Setup logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    config = load_config(args.config)
    log_file = config['logging']['file']
    
    logger = setup_logging(log_file, level=log_level)
    
    logger.info("=" * 60)
    logger.info("FOREX SIGNAL MODEL")
    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Log level: {log_level}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    
    if args.dry_run:
        logger.info("\n[DRY RUN] Would execute the following:")
        logger.info(f"  - Mode: {args.mode}")
        logger.info(f"  - Config: {args.config}")
        return 0
    
    try:
        if args.mode == 'download':
            run_download_pipeline(args.config)
        elif args.mode == 'features':
            run_feature_pipeline(args.config)
        elif args.mode == 'update':
            run_update_pipeline(args.config)
        elif args.mode == 'all':
            run_full_pipeline(args.config)
        
        logger.info("\nPipeline completed successfully!")
        return 0
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    
    except ValueError as e:
        logger.error(f"Value error: {e}")
        return 1
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
