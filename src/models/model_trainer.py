import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import json
from pathlib import Path

from .lstm_model import LSTMModel
from ..data.data_collector import DataCollector
from ..data.preprocessor import DataPreprocessor
from ..utils.logger import TradingLogger
from ..utils.config_manager import ConfigManager


class ModelTrainer:
    """Complete model training pipeline"""

    def __init__(self):
        self.logger = TradingLogger().get_logger()
        self.config = ConfigManager()

        # Initialize components
        self.data_collector = DataCollector()
        self.preprocessor = DataPreprocessor()
        self.model = LSTMModel()

        # Results storage
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)

        # Training results
        self.training_results = None
        self.evaluation_results = None

    def prepare_data(self, symbol: str = "NIFTY 50", days_back: int = 365) -> Dict[str, Any]:
        """Prepare data for training"""
        try:
            self.logger.info(f"Preparing data for {symbol}...")

            # Collect historical data
            historical_data = self.data_collector.collect_historical_data(symbol, days_back)

            # Complete preprocessing pipeline
            processed_data = self.preprocessor.process_for_training(historical_data)

            self.logger.info("Data preparation completed")
            return processed_data

        except Exception as e:
            self.logger.error(f"Error preparing data: {str(e)}")
            raise

    def train_model(self, processed_data: Dict[str, Any], model_type: str = 'lstm') -> Dict[str, Any]:
        """Train the model"""
        try:
            self.logger.info(f"Starting model training with {model_type}...")

            # Extract training data
            X_train = processed_data['X_train']
            y_train = processed_data['y_train']
            X_val = processed_data['X_val']
            y_val = processed_data['y_val']

            # Train model
            training_results = self.model.train(X_train, y_train, X_val, y_val, model_type)

            # Save training results
            self.training_results = training_results

            self.logger.info("Model training completed successfully")
            return training_results

        except Exception as e:
            self.logger.error(f"Error during model training: {str(e)}")
            raise

    def evaluate_model(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate trained model"""
        try:
            self.logger.info("Evaluating trained model...")

            # Test data
            X_test = processed_data['X_test']
            y_test = processed_data['y_test']

            # Evaluate model
            evaluation_results = self.model.evaluate(X_test, y_test)

            # Convert scaled predictions back to original scale
            predictions_scaled = self.model.predict(X_test)
            predictions = self.preprocessor.inverse_transform_targets(predictions_scaled)
            actual = self.preprocessor.inverse_transform_targets(y_test)

            # Calculate additional metrics on original scale
            price_mae = np.mean(np.abs(actual - predictions))
            price_rmse = np.sqrt(np.mean((actual - predictions) ** 2))
            price_mape = np.mean(np.abs((actual - predictions) / actual)) * 100

            evaluation_results.update({
                'price_mae': float(price_mae),
                'price_rmse': float(price_rmse),
                'price_mape': float(price_mape),
                'predictions': predictions.tolist(),
                'actual': actual.tolist()
            })

            self.evaluation_results = evaluation_results

            self.logger.info("Model evaluation completed")
            return evaluation_results

        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
            raise

    def plot_training_history(self, save_plot: bool = True):
        """Plot training history"""
        try:
            if self.training_results is None:
                raise ValueError("No training results to plot")

            history = self.training_results['history']

            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # Loss plot
            axes[0, 0].plot(history['loss'], label='Training Loss')
            axes[0, 0].plot(history['val_loss'], label='Validation Loss')
            axes[0, 0].set_title('Model Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)

            # MAE plot
            axes[0, 1].plot(history['mae'], label='Training MAE')
            axes[0, 1].plot(history['val_mae'], label='Validation MAE')
            axes[0, 1].set_title('Mean Absolute Error')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('MAE')
            axes[0, 1].legend()
            axes[0, 1].grid(True)

            # MAPE plot
            axes[1, 0].plot(history['mape'], label='Training MAPE')
            axes[1, 0].plot(history['val_mape'], label='Validation MAPE')
            axes[1, 0].set_title('Mean Absolute Percentage Error')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('MAPE')
            axes[1, 0].legend()
            axes[1, 0].grid(True)

            # Learning rate plot (if available)
            if 'lr' in history:
                axes[1, 1].plot(history['lr'], label='Learning Rate')
                axes[1, 1].set_title('Learning Rate')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Learning Rate')
                axes[1, 1].legend()
                axes[1, 1].grid(True)
            else:
                axes[1, 1].text(0.5, 0.5, 'Learning Rate\nNot Available',
                                ha='center', va='center', transform=axes[1, 1].transAxes)

            plt.tight_layout()

            if save_plot:
                plot_path = self.results_dir / 'training_history.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Training history plot saved to {plot_path}")

            plt.show()

        except Exception as e:
            self.logger.error(f"Error plotting training history: {str(e)}")
            raise

    def plot_predictions(self, processed_data: Dict[str, Any], save_plot: bool = True):
        """Plot predictions vs actual values"""
        try:
            if self.evaluation_results is None:
                raise ValueError("No evaluation results to plot")

            predictions = np.array(self.evaluation_results['predictions'])
            actual = np.array(self.evaluation_results['actual'])

            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # Time series plot
            axes[0, 0].plot(actual, label='Actual', alpha=0.7)
            axes[0, 0].plot(predictions, label='Predicted', alpha=0.7)
            axes[0, 0].set_title('Actual vs Predicted Prices')
            axes[0, 0].set_xlabel('Time')
            axes[0, 0].set_ylabel('Price')
            axes[0, 0].legend()
            axes[0, 0].grid(True)

            # Scatter plot
            axes[0, 1].scatter(actual, predictions, alpha=0.5)
            axes[0, 1].plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--')
            axes[0, 1].set_title('Predicted vs Actual Scatter')
            axes[0, 1].set_xlabel('Actual Price')
            axes[0, 1].set_ylabel('Predicted Price')
            axes[0, 1].grid(True)

            # Error distribution
            errors = predictions - actual
            axes[1, 0].hist(errors, bins=50, alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('Prediction Error Distribution')
            axes[1, 0].set_xlabel('Prediction Error')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True)

            # Residuals plot
            axes[1, 1].scatter(range(len(errors)), errors, alpha=0.5)
            axes[1, 1].axhline(y=0, color='r', linestyle='--')
            axes[1, 1].set_title('Residuals Plot')
            axes[1, 1].set_xlabel('Index')
            axes[1, 1].set_ylabel('Residuals')
            axes[1, 1].grid(True)

            plt.tight_layout()

            if save_plot:
                plot_path = self.results_dir / 'predictions_analysis.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Predictions plot saved to {plot_path}")

            plt.show()

        except Exception as e:
            self.logger.error(f"Error plotting predictions: {str(e)}")
            raise

    def run_complete_training(self, symbol: str = "NIFTY 50", days_back: int = 365,
                              model_type: str = 'lstm') -> Dict[str, Any]:
        """Run complete training pipeline"""
        try:
            self.logger.info("🚀 Starting complete training pipeline...")

            # Step 1: Prepare data
            self.logger.info("Step 1: Preparing data...")
            processed_data = self.prepare_data(symbol, days_back)

            # Step 2: Train model
            self.logger.info("Step 2: Training model...")
            training_results = self.train_model(processed_data, model_type)

            # Step 3: Evaluate model
            self.logger.info("Step 3: Evaluating model...")
            evaluation_results = self.evaluate_model(processed_data)

            # Step 4: Save model
            self.logger.info("Step 4: Saving model...")
            self.model.save_model()

            # Step 5: Generate plots
            self.logger.info("Step 5: Generating plots...")
            self.plot_training_history()
            self.plot_predictions(processed_data)

            # Step 6: Save results
            self.logger.info("Step 6: Saving results...")
            self.save_results(processed_data, training_results, evaluation_results)

            # Complete results
            complete_results = {
                'data_info': {
                    'symbol': symbol,
                    'days_back': days_back,
                    'total_samples': len(processed_data['X_train']) + len(processed_data['X_val']) + len(
                        processed_data['X_test']),
                    'training_samples': len(processed_data['X_train']),
                    'validation_samples': len(processed_data['X_val']),
                    'test_samples': len(processed_data['X_test']),
                    'features': len(processed_data['feature_columns'])
                },
                'training_info': {
                    'model_type': model_type,
                    'best_epoch': training_results['best_epoch'],
                    'best_val_loss': training_results['best_val_loss'],
                    'total_epochs': len(training_results['history']['loss'])
                },
                'evaluation_metrics': evaluation_results,
                'model_summary': self.model.get_model_summary()
            }

            self.logger.info("✅ Complete training pipeline finished successfully!")
            return complete_results

        except Exception as e:
            self.logger.error(f"Error in complete training pipeline: {str(e)}")
            raise

    def save_results(self, processed_data: Dict[str, Any], training_results: Dict[str, Any],
                     evaluation_results: Dict[str, Any]):
        """Save training results to files"""
        try:
            # Save results summary
            results_summary = {
                'timestamp': datetime.now().isoformat(),
                'data_info': {
                    'total_samples': len(processed_data['X_train']) + len(processed_data['X_val']) + len(
                        processed_data['X_test']),
                    'features': len(processed_data['feature_columns']),
                    'sequence_length': processed_data['X_train'].shape[1]
                },
                'training_results': {
                    'best_epoch': training_results['best_epoch'],
                    'best_val_loss': training_results['best_val_loss'],
                    'final_train_loss': training_results['history']['loss'][-1],
                    'final_val_loss': training_results['history']['val_loss'][-1]
                },
                'evaluation_metrics': evaluation_results,
                'model_summary': self.model.get_model_summary()
            }

            # Save to JSON
            results_path = self.results_dir / 'training_results.json'
            with open(results_path, 'w') as f:
                json.dump(results_summary, f, indent=2)

            # Save feature importance
            feature_importance_path = self.results_dir / 'feature_columns.json'
            with open(feature_importance_path, 'w') as f:
                json.dump(processed_data['feature_columns'], f, indent=2)

            self.logger.info(f"Results saved to {results_path}")

        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            raise

    def hyperparameter_tuning(self, processed_data: Dict[str, Any],
                              param_grid: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Perform hyperparameter tuning"""
        try:
            self.logger.info("Starting hyperparameter tuning...")

            best_score = float('inf')
            best_params = None
            results = []

            # Generate parameter combinations
            from itertools import product
            param_combinations = list(product(*param_grid.values()))
            param_names = list(param_grid.keys())

            for i, params in enumerate(param_combinations):
                try:
                    self.logger.info(
                        f"Testing combination {i + 1}/{len(param_combinations)}: {dict(zip(param_names, params))}")

                    # Update model config
                    for param_name, param_value in zip(param_names, params):
                        self.model.model_config[param_name] = param_value

                    # Train model
                    training_results = self.train_model(processed_data, 'lstm')

                    # Evaluate
                    evaluation_results = self.evaluate_model(processed_data)

                    # Track best results
                    score = evaluation_results['mse']
                    if score < best_score:
                        best_score = score
                        best_params = dict(zip(param_names, params))

                    results.append({
                        'params': dict(zip(param_names, params)),
                        'score': score,
                        'metrics': evaluation_results
                    })

                except Exception as e:
                    self.logger.error(f"Error in parameter combination {i + 1}: {str(e)}")
                    continue

            tuning_results = {
                'best_params': best_params,
                'best_score': best_score,
                'all_results': results
            }

            # Save tuning results
            tuning_path = self.results_dir / 'hyperparameter_tuning.json'
            with open(tuning_path, 'w') as f:
                json.dump(tuning_results, f, indent=2)

            self.logger.info(f"Hyperparameter tuning completed. Best params: {best_params}")
            return tuning_results

        except Exception as e:
            self.logger.error(f"Error in hyperparameter tuning: {str(e)}")
            raise
