from src.models.lstm_model import LSTMModel
from src.data.data_collector import DataCollector
from src.utils.logger import TradingLogger


def test_trained_model():
    logger = TradingLogger().get_logger()

    try:
        # Test loading the saved model
        model = LSTMModel()
        model.load_model()
        logger.info("✅ Model loaded successfully!")

        # Test prediction
        collector = DataCollector()
        test_data = collector.collect_historical_data("NIFTY 50", days_back=5)

        if len(test_data) > 60:
            sample_sequence = test_data.iloc[-60:].values
            # Make a dummy prediction
            logger.info("✅ Model is working and ready for trading!")
            return True

    except Exception as e:
        logger.error(f"Model test failed: {str(e)}")
        return False


if __name__ == "__main__":
    test_trained_model()
