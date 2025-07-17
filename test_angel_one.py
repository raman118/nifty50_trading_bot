import time
import numpy as np  # Add this import
from datetime import datetime, timedelta
from src.api.angel_one_client import AngelOneClient
from src.utils.logger import TradingLogger


def test_angel_one_mock():
    """Test Angel One API client with mock data"""

    logger = TradingLogger().get_logger()

    try:
        # Initialize client
        logger.info("🚀 Testing Angel One API Client (Mock Mode)...")
        client = AngelOneClient()

        # Test 1: Connection status
        status = client.get_connection_status()
        logger.info(f"📊 Connection Status: {status}")

        # Test 2: Current price
        price = client.get_current_price("NIFTY 50")
        logger.info(f"💰 Current Nifty 50 Price: ₹{price}")

        # Test 3: Market data
        market_data = client.get_market_data("NIFTY 50")
        logger.info(f"📈 Market Data: {market_data}")

        # Test 4: Historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=2)

        hist_data = client.get_historical_data("NIFTY 50", "ONE_MINUTE", start_date, end_date)
        logger.info(f"📊 Historical data shape: {hist_data.shape}")
        logger.info(f"📊 Historical data sample:\n{hist_data.head()}")

        # Test 5: WebSocket feed
        def on_tick_data(tick):
            try:
                logger.info(f"📡 Live Tick: Price=₹{tick['ltp']}, Change={tick.get('change', 0):.2f}")
            except Exception as e:
                logger.error(f"Error in tick callback: {e}")

        client.start_websocket_feed(on_tick_data)
        logger.info("🔴 Mock WebSocket feed started - receiving live data...")

        # Let it run for 10 seconds
        time.sleep(10)

        # Test 6: Get recent tick data
        recent_ticks = client.get_tick_data(5)
        if recent_ticks:
            tick_prices = [f"₹{tick['ltp']}" for tick in recent_ticks]
            logger.info(f"📊 Recent 5 ticks: {tick_prices}")
        else:
            logger.info("📊 No recent ticks available")

        # Test 7: Price history
        price_history = client.get_price_history(10)
        if price_history:
            logger.info(f"📈 Price history (last 10): {price_history}")
        else:
            logger.info("📈 No price history available")

        # Test 8: Market status
        market_open = client.is_market_open()
        logger.info(f"🏢 Market Open: {market_open}")

        # Stop WebSocket
        client.stop_websocket_feed()

        logger.info("✅ Angel One API Client (Mock Mode) test completed successfully!")

    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_angel_one_mock()
