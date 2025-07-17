import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError
from typing import Dict, List, Optional, Tuple, Any
import joblib
from pathlib import Path
import json

from ..utils.logger import TradingLogger
from ..utils.config_manager import ConfigManager


class LSTMModel:
    """LSTM-based neural network for stock price prediction"""

    def __init__(self):
        self.logger = TradingLogger().get_logger()
        self.config = ConfigManager()
        self.model_config = self.config.get_model_config()

        # Model components
        self.model = None
        self.history = None

        # Model paths
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)

        # Model parameters
        self.sequence_length = self.model_config['sequence_length']
        self.batch_size = self.model_config['batch_size']
        self.epochs = self.model_config['epochs']
        self.learning_rate = self.model_config['learning_rate']

        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)

    def build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Build LSTM model architecture"""
        try:
            self.logger.info(f"Building LSTM model with input shape: {input_shape}")

            model = Sequential([
                # First LSTM layer
                LSTM(
                    units=self.model_config['lstm_units'],
                    return_sequences=True,
                    input_shape=input_shape,
                    dropout=self.model_config['dropout_rate'],
                    recurrent_dropout=self.model_config['dropout_rate'],
                    kernel_regularizer=l1_l2(l1=0.001, l2=0.001),
                    name='lstm_1'
                ),
                BatchNormalization(),

                # Second LSTM layer
                LSTM(
                    units=self.model_config['lstm_units'],
                    return_sequences=True,
                    dropout=self.model_config['dropout_rate'],
                    recurrent_dropout=self.model_config['dropout_rate'],
                    kernel_regularizer=l1_l2(l1=0.001, l2=0.001),
                    name='lstm_2'
                ),
                BatchNormalization(),

                # Third LSTM layer
                LSTM(
                    units=self.model_config['lstm_units'] // 2,
                    return_sequences=False,
                    dropout=self.model_config['dropout_rate'],
                    recurrent_dropout=self.model_config['dropout_rate'],
                    kernel_regularizer=l1_l2(l1=0.001, l2=0.001),
                    name='lstm_3'
                ),
                BatchNormalization(),

                # Dense layers
                Dense(
                    units=self.model_config['dense_units'],
                    activation='relu',
                    kernel_regularizer=l1_l2(l1=0.001, l2=0.001),
                    name='dense_1'
                ),
                Dropout(self.model_config['dropout_rate']),

                Dense(
                    units=self.model_config['dense_units'] // 2,
                    activation='relu',
                    kernel_regularizer=l1_l2(l1=0.001, l2=0.001),
                    name='dense_2'
                ),
                Dropout(self.model_config['dropout_rate']),

                # Output layer
                Dense(1, activation='linear', name='output')
            ])

            # Compile model with explicit metric imports
            model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='mse',
                metrics=[MeanAbsoluteError(), MeanAbsolutePercentageError()]
            )

            self.model = model
            self.logger.info("LSTM model built successfully")

            # Print model summary
            self.logger.info("Model Architecture:")
            model.summary(print_fn=self.logger.info)

            return model

        except Exception as e:
            self.logger.error(f"Error building LSTM model: {str(e)}")
            raise

    def build_advanced_model(self, input_shape: Tuple[int, int]) -> Model:
        """Build advanced LSTM model with attention mechanism"""
        try:
            self.logger.info(f"Building advanced LSTM model with input shape: {input_shape}")

            # Input layer
            inputs = Input(shape=input_shape)

            # LSTM layers with residual connections
            lstm_1 = LSTM(
                units=self.model_config['lstm_units'],
                return_sequences=True,
                dropout=self.model_config['dropout_rate'],
                recurrent_dropout=self.model_config['dropout_rate'],
                kernel_regularizer=l1_l2(l1=0.001, l2=0.001)
            )(inputs)
            lstm_1 = BatchNormalization()(lstm_1)

            lstm_2 = LSTM(
                units=self.model_config['lstm_units'],
                return_sequences=True,
                dropout=self.model_config['dropout_rate'],
                recurrent_dropout=self.model_config['dropout_rate'],
                kernel_regularizer=l1_l2(l1=0.001, l2=0.001)
            )(lstm_1)
            lstm_2 = BatchNormalization()(lstm_2)

            # Attention mechanism (simplified)
            attention_weights = Dense(1, activation='tanh')(lstm_2)
            attention_weights = tf.nn.softmax(attention_weights, axis=1)
            attention_output = tf.reduce_sum(lstm_2 * attention_weights, axis=1)

            # Final LSTM layer
            lstm_3 = LSTM(
                units=self.model_config['lstm_units'] // 2,
                return_sequences=False,
                dropout=self.model_config['dropout_rate'],
                recurrent_dropout=self.model_config['dropout_rate'],
                kernel_regularizer=l1_l2(l1=0.001, l2=0.001)
            )(lstm_2)
            lstm_3 = BatchNormalization()(lstm_3)

            # Combine attention output with LSTM output
            combined = tf.concat([attention_output, lstm_3], axis=1)

            # Dense layers
            dense_1 = Dense(
                units=self.model_config['dense_units'],
                activation='relu',
                kernel_regularizer=l1_l2(l1=0.001, l2=0.001)
            )(combined)
            dense_1 = Dropout(self.model_config['dropout_rate'])(dense_1)

            dense_2 = Dense(
                units=self.model_config['dense_units'] // 2,
                activation='relu',
                kernel_regularizer=l1_l2(l1=0.001, l2=0.001)
            )(dense_1)
            dense_2 = Dropout(self.model_config['dropout_rate'])(dense_2)

            # Output layer
            outputs = Dense(1, activation='linear')(dense_2)

            # Create model
            model = Model(inputs=inputs, outputs=outputs)

            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='mse',
                metrics=[MeanAbsoluteError(), MeanAbsolutePercentageError()]
            )

            self.model = model
            self.logger.info("Advanced LSTM model built successfully")

            return model

        except Exception as e:
            self.logger.error(f"Error building advanced LSTM model: {str(e)}")
            raise

    def build_gru_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Build GRU-based model as alternative to LSTM"""
        try:
            self.logger.info(f"Building GRU model with input shape: {input_shape}")

            model = Sequential([
                # First GRU layer
                GRU(
                    units=self.model_config['lstm_units'],
                    return_sequences=True,
                    input_shape=input_shape,
                    dropout=self.model_config['dropout_rate'],
                    recurrent_dropout=self.model_config['dropout_rate'],
                    kernel_regularizer=l1_l2(l1=0.001, l2=0.001)
                ),
                BatchNormalization(),

                # Second GRU layer
                GRU(
                    units=self.model_config['lstm_units'],
                    return_sequences=True,
                    dropout=self.model_config['dropout_rate'],
                    recurrent_dropout=self.model_config['dropout_rate'],
                    kernel_regularizer=l1_l2(l1=0.001, l2=0.001)
                ),
                BatchNormalization(),

                # Third GRU layer
                GRU(
                    units=self.model_config['lstm_units'] // 2,
                    return_sequences=False,
                    dropout=self.model_config['dropout_rate'],
                    recurrent_dropout=self.model_config['dropout_rate'],
                    kernel_regularizer=l1_l2(l1=0.001, l2=0.001)
                ),
                BatchNormalization(),

                # Dense layers
                Dense(
                    units=self.model_config['dense_units'],
                    activation='relu',
                    kernel_regularizer=l1_l2(l1=0.001, l2=0.001)
                ),
                Dropout(self.model_config['dropout_rate']),

                Dense(
                    units=self.model_config['dense_units'] // 2,
                    activation='relu',
                    kernel_regularizer=l1_l2(l1=0.001, l2=0.001)
                ),
                Dropout(self.model_config['dropout_rate']),

                # Output layer
                Dense(1, activation='linear')
            ])

            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='mse',
                metrics=[MeanAbsoluteError(), MeanAbsolutePercentageError()]
            )

            self.model = model
            self.logger.info("GRU model built successfully")

            return model

        except Exception as e:
            self.logger.error(f"Error building GRU model: {str(e)}")
            raise

    def get_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """Get training callbacks"""
        try:
            callbacks = [
                # Early stopping
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.model_config['early_stopping_patience'],
                    restore_best_weights=True,
                    verbose=1
                ),

                # Learning rate reduction
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7,
                    verbose=1
                ),

                # Model checkpoint
                ModelCheckpoint(
                    filepath=str(self.models_dir / 'best_model.h5'),
                    monitor='val_loss',
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=1
                )
            ]

            return callbacks

        except Exception as e:
            self.logger.error(f"Error creating callbacks: {str(e)}")
            raise

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              model_type: str = 'lstm') -> Dict[str, Any]:
        """Train the model"""
        try:
            self.logger.info(f"Starting model training with {model_type} architecture...")

            # Build model based on type
            input_shape = (X_train.shape[1], X_train.shape[2])

            if model_type == 'lstm':
                self.build_model(input_shape)
            elif model_type == 'advanced_lstm':
                self.build_advanced_model(input_shape)
            elif model_type == 'gru':
                self.build_gru_model(input_shape)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            # Get callbacks
            callbacks = self.get_callbacks()

            # Train model
            self.history = self.model.fit(
                X_train, y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1,
                shuffle=True
            )

            self.logger.info("Model training completed successfully")

            # Return training history
            return {
                'history': self.history.history,
                'model': self.model,
                'best_val_loss': min(self.history.history['val_loss']),
                'best_epoch': np.argmin(self.history.history['val_loss']) + 1
            }

        except Exception as e:
            self.logger.error(f"Error during model training: {str(e)}")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        try:
            if self.model is None:
                raise ValueError("Model not trained or loaded")

            predictions = self.model.predict(X, batch_size=self.batch_size)
            return predictions.flatten()

        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            raise

    def predict_single(self, sequence: np.ndarray) -> float:
        """Make single prediction"""
        try:
            if len(sequence.shape) == 2:
                sequence = sequence.reshape(1, sequence.shape[0], sequence.shape[1])

            prediction = self.model.predict(sequence, verbose=0)
            return float(prediction[0][0])

        except Exception as e:
            self.logger.error(f"Error making single prediction: {str(e)}")
            raise

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        try:
            if self.model is None:
                raise ValueError("Model not trained or loaded")

            # Get predictions
            predictions = self.predict(X_test)

            # Calculate metrics
            mse = np.mean((y_test - predictions) ** 2)
            mae = np.mean(np.abs(y_test - predictions))
            rmse = np.sqrt(mse)

            # Calculate percentage errors
            mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

            # Calculate R-squared
            ss_res = np.sum((y_test - predictions) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2 = 1 - (ss_res / ss_tot)

            # Direction accuracy
            actual_direction = np.sign(np.diff(y_test))
            predicted_direction = np.sign(np.diff(predictions))
            direction_accuracy = np.mean(actual_direction == predicted_direction) * 100

            metrics = {
                'mse': float(mse),
                'mae': float(mae),
                'rmse': float(rmse),
                'mape': float(mape),
                'r2': float(r2),
                'direction_accuracy': float(direction_accuracy)
            }

            self.logger.info(f"Model evaluation metrics: {metrics}")
            return metrics

        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
            raise

    def save_model(self, filename: str = 'nifty_lstm_model.h5'):
        """Save trained model"""
        try:
            if self.model is None:
                raise ValueError("No model to save")

            model_path = self.models_dir / filename
            self.model.save(str(model_path))

            # Save training history
            history_path = self.models_dir / 'training_history.json'
            if self.history:
                # Convert numpy types to Python types for JSON serialization
                history_data = {}
                for key, value in self.history.history.items():
                    if isinstance(value, list):
                        history_data[key] = [float(v) for v in value]
                    else:
                        history_data[key] = float(value)

                with open(history_path, 'w') as f:
                    json.dump(history_data, f, indent=2)

            # Save model config
            config_path = self.models_dir / 'model_config.json'

            # Convert config to JSON-serializable format
            config_clean = {}
            for key, value in self.model_config.items():
                if isinstance(value, (int, float, str, bool)):
                    config_clean[key] = value
                elif isinstance(value, np.integer):
                    config_clean[key] = int(value)
                elif isinstance(value, np.floating):
                    config_clean[key] = float(value)
                else:
                    config_clean[key] = str(value)

            with open(config_path, 'w') as f:
                json.dump(config_clean, f, indent=2)

            self.logger.info(f"Model saved to {model_path}")

        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, filename: str = 'nifty_lstm_model.h5'):
        """Load trained model with compatibility fix"""
        try:
            model_path = self.models_dir / filename

            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            # Define custom objects for loading
            custom_objects = {
                'mse': tf.keras.losses.MeanSquaredError(),
                'mae': tf.keras.metrics.MeanAbsoluteError(),
                'mape': tf.keras.metrics.MeanAbsolutePercentageError(),
                'MeanSquaredError': tf.keras.losses.MeanSquaredError,
                'MeanAbsoluteError': tf.keras.metrics.MeanAbsoluteError,
                'MeanAbsolutePercentageError': tf.keras.metrics.MeanAbsolutePercentageError
            }

            # Load model with custom objects
            self.model = tf.keras.models.load_model(str(model_path), custom_objects=custom_objects)

            # Load training history if available
            history_path = self.models_dir / 'training_history.json'
            if history_path.exists():
                with open(history_path, 'r') as f:
                    history_data = json.load(f)
                    self.history = type('History', (), {'history': history_data})()

            self.logger.info(f"Model loaded from {model_path}")

        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary information"""
        try:
            if self.model is None:
                return {'status': 'No model loaded'}

            # Count parameters
            total_params = self.model.count_params()
            trainable_params = sum([tf.size(w).numpy() for w in self.model.trainable_weights])

            summary = {
                'model_type': type(self.model).__name__,
                'total_parameters': int(total_params),
                'trainable_parameters': int(trainable_params),
                'layers': len(self.model.layers),
                'input_shape': self.model.input_shape,
                'output_shape': self.model.output_shape,
                'optimizer': self.model.optimizer.__class__.__name__,
                'loss_function': self.model.loss,
                'metrics': self.model.metrics_names
            }

            if self.history:
                summary['training_epochs'] = len(self.history.history['loss'])
                summary['best_val_loss'] = min(self.history.history['val_loss'])
                summary['final_train_loss'] = self.history.history['loss'][-1]
                summary['final_val_loss'] = self.history.history['val_loss'][-1]

            return summary

        except Exception as e:
            self.logger.error(f"Error getting model summary: {str(e)}")
            raise
