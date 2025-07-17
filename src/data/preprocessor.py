import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Dict, List, Optional, Tuple, Any
import joblib
from pathlib import Path

from ..features.technical_indicators import TechnicalIndicators
from ..utils.logger import TradingLogger
from ..utils.config_manager import ConfigManager


class DataPreprocessor:
    """Data preprocessing for machine learning"""

    def __init__(self):
        self.logger = TradingLogger().get_logger()
        self.config = ConfigManager()
        self.model_config = self.config.get_model_config()

        # Technical indicators calculator
        self.indicators = TechnicalIndicators()

        # Scalers
        self.feature_scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()

        # Model paths
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)

        # Feature columns
        self.feature_columns = None
        self.target_column = 'close'

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for machine learning"""
        try:
            self.logger.info("Preparing features for machine learning...")

            # Calculate technical indicators
            df_with_indicators = self.indicators.calculate_all_indicators(data)

            # Add additional features
            df_features = self.add_time_features(df_with_indicators)
            df_features = self.add_lag_features(df_features)
            df_features = self.add_rolling_features(df_features)

            # Remove rows with NaN values (due to indicators)
            df_features = df_features.dropna()

            self.logger.info(f"Prepared {len(df_features.columns)} features")
            return df_features

        except Exception as e:
            self.logger.error(f"Error preparing features: {str(e)}")
            raise

    def add_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        try:
            df = data.copy()

            # Extract time features
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month
            df['quarter'] = df.index.quarter

            # Market session features
            df['market_open'] = ((df.index.hour >= 9) & (df.index.hour < 15.5)).astype(int)
            df['pre_market'] = ((df.index.hour >= 9) & (df.index.hour < 9.25)).astype(int)
            df['post_market'] = (df.index.hour >= 15.5).astype(int)

            return df

        except Exception as e:
            self.logger.error(f"Error adding time features: {str(e)}")
            raise

    def add_lag_features(self, data: pd.DataFrame, lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """Add lagged price features"""
        try:
            df = data.copy()

            # Price lags
            for lag in lags:
                df[f'close_lag_{lag}'] = df['close'].shift(lag)
                df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
                df[f'price_change_lag_{lag}'] = df['price_change'].shift(lag)

            return df

        except Exception as e:
            self.logger.error(f"Error adding lag features: {str(e)}")
            raise

    def add_rolling_features(self, data: pd.DataFrame, windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """Add rolling window features"""
        try:
            df = data.copy()

            for window in windows:
                # Rolling statistics
                df[f'close_rolling_mean_{window}'] = df['close'].rolling(window=window).mean()
                df[f'close_rolling_std_{window}'] = df['close'].rolling(window=window).std()
                df[f'volume_rolling_mean_{window}'] = df['volume'].rolling(window=window).mean()

                # Rolling min/max
                df[f'close_rolling_min_{window}'] = df['close'].rolling(window=window).min()
                df[f'close_rolling_max_{window}'] = df['close'].rolling(window=window).max()

                # Position relative to rolling window
                df[f'close_vs_rolling_mean_{window}'] = df['close'] / df[f'close_rolling_mean_{window}']

            return df

        except Exception as e:
            self.logger.error(f"Error adding rolling features: {str(e)}")
            raise

    def create_sequences(self, data: pd.DataFrame, sequence_length: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        try:
            if sequence_length is None:
                sequence_length = self.model_config['sequence_length']

            self.logger.info(f"Creating sequences with length {sequence_length}")

            # Select feature columns (exclude target)
            feature_cols = [col for col in data.columns if col != self.target_column]
            self.feature_columns = feature_cols

            # Get feature and target data
            features = data[feature_cols].values
            targets = data[self.target_column].values

            # Create sequences
            X, y = [], []
            for i in range(sequence_length, len(features)):
                X.append(features[i - sequence_length:i])
                y.append(targets[i])

            X = np.array(X)
            y = np.array(y)

            self.logger.info(f"Created {len(X)} sequences with shape {X.shape}")
            return X, y

        except Exception as e:
            self.logger.error(f"Error creating sequences: {str(e)}")
            raise

    def scale_features(self, X_train: np.ndarray, X_val: np.ndarray = None, X_test: np.ndarray = None) -> Tuple[
        np.ndarray, ...]:
        """Scale features for training"""
        try:
            self.logger.info("Scaling features...")

            # Reshape for scaling
            original_shape = X_train.shape
            X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])

            # Fit and transform training data
            X_train_scaled = self.feature_scaler.fit_transform(X_train_reshaped)
            X_train_scaled = X_train_scaled.reshape(original_shape)

            results = [X_train_scaled]

            # Transform validation data if provided
            if X_val is not None:
                X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])
                X_val_scaled = self.feature_scaler.transform(X_val_reshaped)
                X_val_scaled = X_val_scaled.reshape(X_val.shape)
                results.append(X_val_scaled)

            # Transform test data if provided
            if X_test is not None:
                X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
                X_test_scaled = self.feature_scaler.transform(X_test_reshaped)
                X_test_scaled = X_test_scaled.reshape(X_test.shape)
                results.append(X_test_scaled)

            return tuple(results)

        except Exception as e:
            self.logger.error(f"Error scaling features: {str(e)}")
            raise

    def scale_targets(self, y_train: np.ndarray, y_val: np.ndarray = None, y_test: np.ndarray = None) -> Tuple[
        np.ndarray, ...]:
        """Scale target values"""
        try:
            self.logger.info("Scaling targets...")

            # Reshape for scaling
            y_train_reshaped = y_train.reshape(-1, 1)
            y_train_scaled = self.target_scaler.fit_transform(y_train_reshaped)
            y_train_scaled = y_train_scaled.flatten()

            results = [y_train_scaled]

            # Transform validation targets if provided
            if y_val is not None:
                y_val_reshaped = y_val.reshape(-1, 1)
                y_val_scaled = self.target_scaler.transform(y_val_reshaped)
                y_val_scaled = y_val_scaled.flatten()
                results.append(y_val_scaled)

            # Transform test targets if provided
            if y_test is not None:
                y_test_reshaped = y_test.reshape(-1, 1)
                y_test_scaled = self.target_scaler.transform(y_test_reshaped)
                y_test_scaled = y_test_scaled.flatten()
                results.append(y_test_scaled)

            return tuple(results)

        except Exception as e:
            self.logger.error(f"Error scaling targets: {str(e)}")
            raise

    def inverse_transform_targets(self, y_scaled: np.ndarray) -> np.ndarray:
        """Convert scaled targets back to original scale"""
        try:
            y_reshaped = y_scaled.reshape(-1, 1)
            y_original = self.target_scaler.inverse_transform(y_reshaped)
            return y_original.flatten()
        except Exception as e:
            self.logger.error(f"Error inverse transforming targets: {str(e)}")
            raise

    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train/validation/test sets"""
        try:
            self.logger.info("Splitting data into train/validation/test sets...")

            # Get split ratios
            train_ratio = self.config.get_data_config()['train_split']
            val_ratio = self.config.get_data_config()['validation_split']

            # Calculate split indices
            total_samples = len(X)
            train_size = int(total_samples * train_ratio)
            val_size = int(total_samples * val_ratio)

            # Split data
            X_train = X[:train_size]
            y_train = y[:train_size]

            X_val = X[train_size:train_size + val_size]
            y_val = y[train_size:train_size + val_size]

            X_test = X[train_size + val_size:]
            y_test = y[train_size + val_size:]

            self.logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

            return X_train, X_val, X_test, y_train, y_val, y_test

        except Exception as e:
            self.logger.error(f"Error splitting data: {str(e)}")
            raise

    def save_scalers(self):
        """Save fitted scalers"""
        try:
            feature_scaler_path = self.models_dir / "feature_scaler.pkl"
            target_scaler_path = self.models_dir / "target_scaler.pkl"

            joblib.dump(self.feature_scaler, feature_scaler_path)
            joblib.dump(self.target_scaler, target_scaler_path)

            self.logger.info("Scalers saved successfully")

        except Exception as e:
            self.logger.error(f"Error saving scalers: {str(e)}")
            raise

    def load_scalers(self):
        """Load fitted scalers"""
        try:
            feature_scaler_path = self.models_dir / "feature_scaler.pkl"
            target_scaler_path = self.models_dir / "target_scaler.pkl"

            if feature_scaler_path.exists() and target_scaler_path.exists():
                self.feature_scaler = joblib.load(feature_scaler_path)
                self.target_scaler = joblib.load(target_scaler_path)
                self.logger.info("Scalers loaded successfully")
                return True
            else:
                self.logger.warning("Scaler files not found")
                return False

        except Exception as e:
            self.logger.error(f"Error loading scalers: {str(e)}")
            return False

    def process_for_training(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Complete preprocessing pipeline for training"""
        try:
            self.logger.info("Starting complete preprocessing pipeline...")

            # Prepare features
            features_df = self.prepare_features(data)

            # Create sequences
            X, y = self.create_sequences(features_df)

            # Split data
            X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)

            # Scale features
            X_train_scaled, X_val_scaled, X_test_scaled = self.scale_features(X_train, X_val, X_test)

            # Scale targets
            y_train_scaled, y_val_scaled, y_test_scaled = self.scale_targets(y_train, y_val, y_test)

            # Save scalers
            self.save_scalers()

            result = {
                'X_train': X_train_scaled,
                'X_val': X_val_scaled,
                'X_test': X_test_scaled,
                'y_train': y_train_scaled,
                'y_val': y_val_scaled,
                'y_test': y_test_scaled,
                'feature_columns': self.feature_columns,
                'original_data': features_df
            }

            self.logger.info("Preprocessing pipeline completed successfully")
            return result

        except Exception as e:
            self.logger.error(f"Error in preprocessing pipeline: {str(e)}")
            raise
