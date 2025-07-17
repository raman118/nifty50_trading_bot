import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import os
import pickle
from pathlib import Path

from ..api.angel_one_client import AngelOneClient
from ..utils.logger import TradingLogger
from ..utils.config_manager import ConfigManager


class DataCollector:
    """Data collection and storage manager"""

    def __init__(self):
        self.logger = TradingLogger().get_logger()
        self.config = ConfigManager()
        self.data_config = self.config.get_data_config()
        self.trading_config = self.config.get_trading_config()

        # Initialize API client
        self.api_client = AngelOneClient()

        # Data storage paths
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)

        # Cache settings
        self.cache_enabled = self.data_config.get('cache_data', True)
        self.cache_duration = self.data_config.get('cache_duration_hours', 24)

    def collect_historical_data(self, symbol: str, days_back: int = None) -> pd.DataFrame:
        """Collect historical data for training"""
        try:
            if days_back is None:
                days_back = self.data_config['lookback_days']

            self.logger.info(f"Collecting historical data for {symbol} ({days_back} days)")

            # Check cache first
            if self.cache_enabled:
                cached_data = self.load_cached_data(symbol, days_back)
                if cached_data is not None:
                    self.logger.info("Using cached historical data")
                    return cached_data

            # Collect fresh data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)

            # Get data from API
            raw_data = self.api_client.get_historical_data(
                symbol,
                self.trading_config['timeframe'],
                start_date,
                end_date
            )

            # Process and clean data
            processed_data = self.process_raw_data(raw_data)

            # Cache the data
            if self.cache_enabled:
                self.cache_data(processed_data, symbol, days_back)

            self.logger.info(f"Collected {len(processed_data)} historical records")
            return processed_data

        except Exception as e:
            self.logger.error(f"Error collecting historical data: {str(e)}")
            raise

    def process_raw_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Process and clean raw market data"""
        try:
            self.logger.info("Processing raw market data...")

            # Make a copy
            df = raw_data.copy()

            # Remove any rows with NaN values
            df = df.dropna()

            # Ensure proper data types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Remove outliers (prices that are too far from median)
            for col in ['open', 'high', 'low', 'close']:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

            # Ensure OHLC relationships are valid
            df = df[
                (df['high'] >= df['open']) &
                (df['high'] >= df['close']) &
                (df['low'] <= df['open']) &
                (df['low'] <= df['close'])
                ]

            # Sort by timestamp
            df = df.sort_index()

            # Remove duplicates
            df = df[~df.index.duplicated(keep='first')]

            self.logger.info(f"Processed data: {len(df)} records after cleaning")
            return df

        except Exception as e:
            self.logger.error(f"Error processing raw data: {str(e)}")
            raise

    def load_cached_data(self, symbol: str, days_back: int) -> Optional[pd.DataFrame]:
        """Load cached data if available and fresh"""
        try:
            cache_file = self.data_dir / f"{symbol}_{days_back}_days.pkl"

            if not cache_file.exists():
                return None

            # Check if cache is still fresh
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if cache_age.total_seconds() > (self.cache_duration * 3600):
                self.logger.info("Cache expired, will fetch fresh data")
                return None

            # Load cached data
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)

            self.logger.info(f"Loaded {len(data)} records from cache")
            return data

        except Exception as e:
            self.logger.error(f"Error loading cached data: {str(e)}")
            return None

    def cache_data(self, data: pd.DataFrame, symbol: str, days_back: int):
        """Cache processed data"""
        try:
            cache_file = self.data_dir / f"{symbol}_{days_back}_days.pkl"

            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)

            self.logger.info(f"Cached {len(data)} records to {cache_file}")

        except Exception as e:
            self.logger.error(f"Error caching data: {str(e)}")

    def get_real_time_data(self, symbol: str) -> Dict[str, Any]:
        """Get real-time market data"""
        try:
            return self.api_client.get_market_data(symbol)
        except Exception as e:
            self.logger.error(f"Error getting real-time data: {str(e)}")
            raise

    def start_real_time_feed(self, callback):
        """Start real-time data feed"""
        try:
            self.logger.info("Starting real-time data feed...")
            self.api_client.start_websocket_feed(callback)
        except Exception as e:
            self.logger.error(f"Error starting real-time feed: {str(e)}")
            raise

    def stop_real_time_feed(self):
        """Stop real-time data feed"""
        try:
            self.api_client.stop_websocket_feed()
            self.logger.info("Real-time data feed stopped")
        except Exception as e:
            self.logger.error(f"Error stopping real-time feed: {str(e)}")

    def export_data(self, data: pd.DataFrame, filename: str, format: str = 'csv'):
        """Export data to file"""
        try:
            export_path = self.data_dir / filename

            if format.lower() == 'csv':
                data.to_csv(export_path)
            elif format.lower() == 'json':
                data.to_json(export_path)
            elif format.lower() == 'excel':
                data.to_excel(export_path)
            else:
                raise ValueError(f"Unsupported format: {format}")

            self.logger.info(f"Data exported to {export_path}")

        except Exception as e:
            self.logger.error(f"Error exporting data: {str(e)}")
            raise

    def get_data_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get summary statistics of the data"""
        try:
            summary = {
                'total_records': len(data),
                'date_range': {
                    'start': data.index.min(),
                    'end': data.index.max()
                },
                'price_stats': {
                    'min': data['close'].min(),
                    'max': data['close'].max(),
                    'mean': data['close'].mean(),
                    'std': data['close'].std()
                },
                'volume_stats': {
                    'min': data['volume'].min(),
                    'max': data['volume'].max(),
                    'mean': data['volume'].mean()
                },
                'missing_values': data.isnull().sum().to_dict()
            }

            return summary

        except Exception as e:
            self.logger.error(f"Error generating data summary: {str(e)}")
            raise
