#!/usr/bin/env python3
"""
Data Collection Script for ACS-ML Trading System

This script monitors MT4 CSV exports from the Dynamic Forex28 Navigator indicator
and processes them into a structured format for ML training.

Key Data Points to Extract:
- Currency strength values (8 currencies: USD, EUR, GBP, JPY, CHF, CAD, AUD, NZD)
- BUY/SELL Momentum arrows with trigger values
- Dual Momentum signals for currency pairs
- Dynamic Market Fibonacci levels (MFib 23, 100, 161, 261)
- Outer Range warnings (±100 levels)
- HOOK alerts (reversal signals)
- X-STOP signals (market exhaustion)
- Market activity status (5 possible states)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_collector.log'),
        logging.StreamHandler()
    ]
)

class MT4DataHandler(FileSystemEventHandler):
    """Handles file system events from MT4 CSV exports"""
    
    def __init__(self, processor):
        self.processor = processor
        
    def on_modified(self, event):
        if event.is_directory:
            return
            
        if event.src_path.endswith('.csv'):
            logger.info(f"Detected change in {event.src_path}")
            self.processor.process_file(event.src_path)


class DataCollector:
    """Main data collection class for processing MT4 indicator exports"""
    
    def __init__(self, config_path: str = "config/data_config.yaml"):
        """Initialize the data collector with configuration"""
        self.config = self._load_config(config_path)
        self.data_dir = Path(self.config['paths']['raw_data'])
        self.processed_dir = Path(self.config['paths']['processed_data'])
        self.mt4_export_dir = Path(self.config['paths']['mt4_exports'])
        
        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Currency mapping for Dynamic Forex28 Navigator
        self.currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']
        self.currency_pairs = self._generate_currency_pairs()
        
        logger.info(f"DataCollector initialized. Monitoring: {self.mt4_export_dir}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """Default configuration if config file not found"""
        return {
            'paths': {
                'raw_data': 'data/raw',
                'processed_data': 'data/processed',
                'mt4_exports': 'data/mt4_exports'  # Where MT4 writes CSV files
            },
            'collection': {
                'batch_size': 1000,
                'max_memory_mb': 512,
                'cleanup_days': 30
            },
            'indicators': {
                'momentum_trigger_range': [18, 26],
                'outer_range_levels': [100, 161, 261],
                'mfib_levels': [23, 100, 161, 261]
            }
        }
    
    def _generate_currency_pairs(self) -> List[str]:
        """Generate all 28 currency pairs needed for Dynamic Forex28 Navigator"""
        pairs = []
        for i, base in enumerate(self.currencies):
            for quote in self.currencies[i+1:]:
                pairs.extend([f"{base}{quote}", f"{quote}{base}"])
        return sorted(pairs)
    
    def process_file(self, file_path: str) -> None:
        """Process a single CSV file from MT4 export"""
        try:
            df = pd.read_csv(file_path, parse_dates=['timestamp'], index_col=0)
            
            # Determine file type based on columns
            if self._is_indicator_data(df):
                self._process_indicator_data(df, file_path)
            elif self._is_ohlcv_data(df):
                self._process_ohlcv_data(df, file_path)
            else:
                logger.warning(f"Unknown file format: {file_path}")
                
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
    
    def _is_indicator_data(self, df: pd.DataFrame) -> bool:
        """Check if dataframe contains indicator data"""
        indicator_columns = ['currency_strength', 'momentum_value', 'mfib_level']
        return any(col in df.columns for col in indicator_columns)
    
    def _is_ohlcv_data(self, df: pd.DataFrame) -> bool:
        """Check if dataframe contains OHLCV data"""
        ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
        return all(col in df.columns for col in ohlcv_columns)
    
    def start_monitoring(self) -> None:
        """Start monitoring MT4 export directory for new files"""
        if not self.mt4_export_dir.exists():
            logger.warning(f"MT4 export directory {self.mt4_export_dir} does not exist. Creating...")
            self.mt4_export_dir.mkdir(parents=True, exist_ok=True)
        
        event_handler = MT4DataHandler(self)
        observer = Observer()
        observer.schedule(event_handler, str(self.mt4_export_dir), recursive=True)
        observer.start()
        
        logger.info(f"Started monitoring {self.mt4_export_dir} for MT4 exports")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
            logger.info("Stopping data collection...")
        
        observer.join()


def main():
    """Main function to run the data collector"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ACS-ML Data Collector')
    parser.add_argument('--mode', choices=['monitor', 'historical'], 
                       default='monitor', help='Operation mode')
    parser.add_argument('--directory', type=str, 
                       help='Directory to process (for historical mode)')
    parser.add_argument('--config', type=str, default='config/data_config.yaml',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    collector = DataCollector(args.config)
    
    if args.mode == 'monitor':
        logger.info("Starting real-time monitoring mode...")
        collector.start_monitoring()
    elif args.mode == 'historical':
        logger.info("Processing historical data...")
        # Add historical processing method call here


if __name__ == "__main__":
    main()
    # Example: process exported files from MT4
    # These files should be exported by your MT4 EA
    signal_file = "forex28_signals.csv"  # From indicator
    price_file = "eurusd_h1_prices.csv"   # OHLCV data
    output_file = "eurusd_h1_with_signals.csv"
    
    # Check if files exist
    if not (collector.raw_data_path / signal_file).exists():
        logger.warning(f"Signal file not found: {signal_file}")
        logger.info("Please export indicator signals from MT4 first")
        return
    
    if not (collector.raw_data_path / price_file).exists():
        logger.warning(f"Price file not found: {price_file}")
        logger.info("Please export price data from MT4 first")
        return
    
    # Run collection
    collector.collect_and_process(signal_file, price_file, output_file)

if __name__ == "__main__":
    main()