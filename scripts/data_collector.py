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
logger = logging.getLogger(__name__)


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
            df = pd.read_csv(file_path)
            
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
    
    def _process_indicator_data(self, df: pd.DataFrame, source_file: str) -> None:
        """Process Dynamic Forex28 Navigator indicator data"""
        logger.info(f"Processing indicator data from {source_file}")
        
        # Structure the indicator data
        processed_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'source_file': source_file,
            'data_type': 'indicator',
            'signals': {}
        }
        
        # Process currency strength values
        for currency in self.currencies:
            currency_col = f"{currency.lower()}_strength"
            if currency_col in df.columns:
                processed_data['signals'][f'{currency}_strength'] = df[currency_col].iloc[-1]
        
        # Save processed data
        self._save_processed_data(processed_data, 'indicator')
    
    def _process_ohlcv_data(self, df: pd.DataFrame, source_file: str) -> None:
        """Process OHLCV market data"""
        logger.info(f"Processing OHLCV data from {source_file}")
        
        # Extract symbol from filename
        symbol = Path(source_file).stem.split('_')[0] if '_' in Path(source_file).stem else 'UNKNOWN'
        
        # Add symbol column
        df['symbol'] = symbol
        
        # Save to processed directory
        output_file = self.processed_dir / f"ohlcv_{symbol}_{datetime.now().strftime('%Y%m%d')}.csv"
        df.to_csv(output_file, index=False)
        
        logger.info(f"Saved OHLCV data to {output_file}")
    
    def _save_processed_data(self, data: Dict, data_type: str) -> None:
        """Save processed data to JSON file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = self.processed_dir / f"{data_type}_data_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {data_type} data to {output_file}")


def main():
    """Main function to run the data collector"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ACS-ML Data Collector')
    parser.add_argument('--config', type=str, default='config/data_config.yaml',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    collector = DataCollector(args.config)
    logger.info("Data collector initialized. Ready to process MT4 exports.")
    
    # Example usage - process existing files
    mt4_export_dir = Path("data/mt4_exports")
    if mt4_export_dir.exists():
        csv_files = list(mt4_export_dir.glob("*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files to process")
        for csv_file in csv_files:
            collector.process_file(str(csv_file))
    else:
        logger.info(f"MT4 export directory {mt4_export_dir} not found. Create it and export CSV files from MT4.")


if __name__ == "__main__":
    main()