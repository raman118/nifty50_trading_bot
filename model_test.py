from src.models.model_trainer import ModelTrainer
from src.utils.logger import TradingLogger
import numpy as np


def test_part4():
    """Test Part 4: Machine Learning Model"""

    logger = TradingLogger().get_logger()

    try:
        logger.info(" Testing model: Machine Learning Model")

        # Initialize trainer
        trainer = ModelTrainer()

        # Test 1: Quick training run (small dataset)
        logger.info("1. Testing quick training run...")

        # Run complete training pipeline
        results = trainer.run_complete_training(
            symbol="NIFTY 50",
            days_back=30,  # Small dataset for quick test
            model_type='lstm'
        )

        logger.info(" Training completed successfully!")

        # Display results
        logger.info(f" Training Results Summary:")
        logger.info(f"   - Total samples: {results['data_info']['total_samples']}")
        logger.info(f"   - Features: {results['data_info']['features']}")
        logger.info(f"   - Best epoch: {results['training_info']['best_epoch']}")
        logger.info(f"   - Best validation loss: {results['training_info']['best_val_loss']:.6f}")

        logger.info(f" Evaluation Metrics:")
        metrics = results['evaluation_metrics']
        logger.info(f"   - MSE: {metrics['mse']:.6f}")
        logger.info(f"   - MAE: {metrics['mae']:.6f}")
        logger.info(f"   - RMSE: {metrics['rmse']:.6f}")
        logger.info(f"   - MAPE: {metrics['mape']:.2f}%")
        logger.info(f"   - R²: {metrics['r2']:.4f}")
        logger.info(f"   - Direction Accuracy: {metrics['direction_accuracy']:.2f}%")

        # Test 2: Model predictions
        logger.info("2. Testing model predictions...")

        # Test single prediction
        processed_data = trainer.prepare_data("NIFTY 50", days_back=30)
        X_test = processed_data['X_test']

        if len(X_test) > 0:
            sample_sequence = X_test[0]
            prediction = trainer.model.predict_single(sample_sequence)
            logger.info(f" Single prediction test: {prediction:.2f}")

        # Test 3: Model summary
        logger.info("3. Testing model summary...")
        model_summary = trainer.model.get_model_summary()
        logger.info(f" Model Summary:")
        logger.info(f"   - Model type: {model_summary['model_type']}")
        logger.info(f"   - Total parameters: {model_summary['total_parameters']:,}")
        logger.info(f"   - Training epochs: {model_summary.get('training_epochs', 'N/A')}")

        # Test 4: Different model types
        logger.info("4. Testing different model architectures...")

        # Test GRU model
        try:
            gru_results = trainer.train_model(processed_data, model_type='gru')
            logger.info(f" GRU model trained successfully")
        except Exception as e:
            logger.warning(f"GRU model test failed: {str(e)}")

        # Test advanced LSTM
        try:
            advanced_results = trainer.train_model(processed_data, model_type='advanced_lstm')
            logger.info(f" Advanced LSTM model trained successfully")
        except Exception as e:
            logger.warning(f"Advanced LSTM model test failed: {str(e)}")

        # Test 5: Model save/load
        logger.info("5. Testing model save/load...")

        # Save model
        trainer.model.save_model('test_model.h5')
        logger.info(" Model saved successfully")

        # Load model
        new_trainer = ModelTrainer()
        new_trainer.model.load_model('test_model.h5')
        logger.info(" Model loaded successfully")

        # Test prediction with loaded model
        if len(X_test) > 0:
            loaded_prediction = new_trainer.model.predict_single(X_test[0])
            logger.info(f" Loaded model prediction: {loaded_prediction:.2f}")

        logger.info(" Part 4 testing completed successfully!")

    except Exception as e:
        logger.error(f" model test failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_part4()
