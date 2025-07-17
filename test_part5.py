import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.trading.signal_generator import SignalGenerator
from src.trading.risk_manager import RiskManager
from src.data.data_collector import DataCollector
from src.utils.logger import TradingLogger


def test_part5():
    """Test Part 5: Trading Logic & Risk Management"""

    logger = TradingLogger().get_logger()

    try:
        logger.info("🚀 Testing Part 5: Trading Logic & Risk Management")

        # Test 1: Signal Generator
        logger.info("1. Testing Signal Generator...")
        signal_generator = SignalGenerator()

        # Get some sample data
        collector = DataCollector()
        sample_data = collector.collect_historical_data("NIFTY 50", days_back=30)

        # Generate a signal
        current_price = 18000.0
        signal_result = signal_generator.generate_signal(sample_data, current_price)

        logger.info(f"✅ Signal Generated:")
        logger.info(f"   - Signal: {signal_result['signal'].value}")
        logger.info(f"   - Strength: {signal_result['strength']:.2f}")
        logger.info(f"   - Confidence: {signal_result['confidence']:.2f}")
        logger.info(f"   - ML Prediction: ₹{signal_result['ml_prediction']:.2f}")
        logger.info(f"   - Current Price: ₹{signal_result['current_price']:.2f}")

        # Test 2: Risk Manager
        logger.info("2. Testing Risk Manager...")
        risk_manager = RiskManager()

        # Generate trade recommendation
        recommendation = risk_manager.generate_trade_recommendation(signal_result)

        if recommendation:
            logger.info(f"✅ Trade Recommendation Generated:")
            logger.info(f"   - Action: {recommendation.action}")
            logger.info(f"   - Quantity: {recommendation.quantity}")
            logger.info(f"   - Price: ₹{recommendation.price:.2f}")
            logger.info(f"   - Stop Loss: ₹{recommendation.stop_loss:.2f}")
            logger.info(f"   - Take Profit: ₹{recommendation.take_profit:.2f}")
            logger.info(f"   - Risk-Reward Ratio: {recommendation.risk_reward_ratio:.2f}")
            logger.info(f"   - Position Size: {recommendation.position_size_pct:.2%}")

            # Test 3: Execute Trade
            logger.info("3. Testing Trade Execution...")
            trade_result = risk_manager.execute_trade(recommendation)

            if trade_result['status'] == 'EXECUTED':
                logger.info(f"✅ Trade Executed Successfully:")
                logger.info(f"   - Trade ID: {trade_result['trade_id']}")
                logger.info(f"   - Action: {trade_result['action']}")
                logger.info(f"   - Quantity: {trade_result['quantity']}")
                logger.info(f"   - Price: ₹{trade_result['price']:.2f}")
        else:
            logger.info("⚠️ No trade recommendation generated (may be due to risk limits)")

        # Test 4: Portfolio Summary
        logger.info("4. Testing Portfolio Summary...")
        portfolio_summary = risk_manager.get_portfolio_summary()

        logger.info(f"✅ Portfolio Summary:")
        logger.info(f"   - Portfolio Value: ₹{portfolio_summary['portfolio_value']:,.2f}")
        logger.info(f"   - Available Capital: ₹{portfolio_summary['available_capital']:,.2f}")
        logger.info(f"   - Total Return: {portfolio_summary['total_return']:.2f}%")
        logger.info(f"   - Active Positions: {portfolio_summary['active_positions']}")
        logger.info(f"   - Total Trades: {portfolio_summary['total_trades']}")
        logger.info(f"   - Win Rate: {portfolio_summary['win_rate']:.1f}%")

        # Test 5: Risk Limits
        logger.info("5. Testing Risk Limits...")
        risk_check = risk_manager.check_risk_limits()

        logger.info(f"✅ Risk Check:")
        logger.info(f"   - Can Trade: {risk_check['can_trade']}")
        logger.info(f"   - Violations: {len(risk_check['violations'])}")

        for violation in risk_check['violations']:
            logger.info(f"   - {violation['type']}: {violation['message']}")

        # Test 6: Multiple Signals
        logger.info("6. Testing Multiple Signals...")

        # Generate multiple signals with different prices
        test_prices = [17900, 18100, 18200, 17800, 18050]

        for i, price in enumerate(test_prices):
            logger.info(f"   Signal {i + 1} at ₹{price}:")
            signal = signal_generator.generate_signal(sample_data, price)
            logger.info(f"      - {signal['signal'].value} (Strength: {signal['strength']:.2f})")

        # Test 7: Signal Statistics
        logger.info("7. Testing Signal Statistics...")
        signal_stats = signal_generator.get_signal_statistics()

        logger.info(f"✅ Signal Statistics:")
        logger.info(f"   - Total Signals: {signal_stats['total_signals']}")
        logger.info(f"   - Buy Signals: {signal_stats.get('buy_signals', 0)}")
        logger.info(f"   - Sell Signals: {signal_stats.get('sell_signals', 0)}")
        logger.info(f"   - Hold Signals: {signal_stats.get('hold_signals', 0)}")
        logger.info(f"   - Average Strength: {signal_stats.get('average_strength', 0):.2f}")
        logger.info(f"   - Average Confidence: {signal_stats.get('average_confidence', 0):.2f}")

        # Test 8: Position Management
        logger.info("8. Testing Position Management...")

        # Simulate price changes and position updates
        if risk_manager.positions:
            symbol = list(risk_manager.positions.keys())[0]

            # Test price updates
            for price_change in [50, -30, 100, -70]:
                new_price = current_price + price_change
                risk_manager.update_position_pnl(symbol, new_price)

                # Check for position management signals
                updated_signal = signal_generator.generate_signal(sample_data, new_price)
                position_recommendation = risk_manager.handle_existing_position(symbol, updated_signal)

                if position_recommendation:
                    logger.info(f"   Position Management at ₹{new_price}: {position_recommendation.action}")

        logger.info("🎉 Part 5 testing completed successfully!")

    except Exception as e:
        logger.error(f"❌ Part 5 test failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_part5()
