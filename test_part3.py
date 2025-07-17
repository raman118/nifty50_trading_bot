from datetime import datetime, timedelta
from src.data.data_collector import DataCollector
from src.data.preprocessor import DataPreprocessor
from src.features.technical_indicators import TechnicalIndicators
from src.utils.logger import TradingLogger


def test_part3():
    """Test Part 3: Data Collection & Preprocessing"""

    logger = TradingLogger().get_logger()

    try:
        logger.info("🚀 Testing Part 3: Data Collection & Preprocessing")

        # Test 1: Data Collection
        logger.info("1. Testing Data Collection...")
        collector = DataCollector()

        # Collect historical data
        historical_data = collector.collect_historical_data("NIFTY 50", days_back=30)
        logger.info(f"✅ Collected {len(historical_data)} historical records")

        # Data summary
        summary = collector.get_data_summary(historical_data)
        logger.info(f"📊 Data Summary: {summary}")

        # Test 2: Technical Indicators
        logger.info("2. Testing Technical Indicators...")
        indicators = TechnicalIndicators()

        # Calculate indicators
        data_with_indicators = indicators.calculate_all_indicators(historical_data)
        logger.info(f"✅ Added technical indicators: {len(data_with_indicators.columns)} total columns")

        # Show some indicators
        indicator_columns = [col for col in data_with_indicators.columns if col not in historical_data.columns]
        logger.info(f"📈 New indicators: {indicator_columns[:10]}...")

        # Test 3: Data Preprocessing
        logger.info("3. Testing Data Preprocessing...")
        preprocessor = DataPreprocessor()

        # Complete preprocessing pipeline
        processed_data = preprocessor.process_for_training(historical_data)

        logger.info(f"✅ Preprocessing completed:")
        logger.info(f"   - Training samples: {len(processed_data['X_train'])}")
        logger.info(f"   - Validation samples: {len(processed_data['X_val'])}")
        logger.info(f"   - Test samples: {len(processed_data['X_test'])}")
        logger.info(f"   - Features: {len(processed_data['feature_columns'])}")
        logger.info(f"   - Sequence shape: {processed_data['X_train'].shape}")

        # Test 4: Feature Analysis
        logger.info("4. Feature Analysis...")
        feature_df = processed_data['original_data']

        # Show feature importance (correlation with target)
        correlations = feature_df.corr()['close'].abs().sort_values(ascending=False)
        logger.info(f"📊 Top 10 features by correlation with price:")
        for i, (feature, corr) in enumerate(correlations.head(10).items()):
            logger.info(f"   {i + 1}. {feature}: {corr:.3f}")

        # Test 5: Data Export
        logger.info("5. Testing Data Export...")
        collector.export_data(data_with_indicators, "processed_data.csv")
        logger.info("✅ Data exported successfully")

        logger.info("🎉 Part 3 testing completed successfully!")

    except Exception as e:
        logger.error(f"❌ Part 3 test failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_part3()
