from src.api.nse_scraper import NSEDataScraper
from src.utils.logger import TradingLogger
import time


def test_nse_scraper():
    """Test NSE India data scraper"""
    logger = TradingLogger().get_logger()

    try:
        logger.info("🌐 Testing NSE India Data Scraper...")

        # Initialize scraper
        nse = NSEDataScraper()

        # Test 1: Get current Nifty data
        logger.info("1. Testing current Nifty 50 data...")
        data = nse.get_nifty_data()

        if data:
            logger.info(f"✅ Current Nifty 50: ₹{data['ltp']:.2f}")
            logger.info(f"   Change: {data['change']:+.2f} ({data['pChange']:+.2f}%)")
            logger.info(f"   High: ₹{data['high']:.2f}, Low: ₹{data['low']:.2f}")
        else:
            logger.error("❌ Failed to get Nifty data")

        # Test 2: Market status
        logger.info("2. Testing market status...")
        status = nse.get_market_status()
        logger.info(f"✅ Market Status: {status['status']} (Open: {status['market_open']})")

        # Test 3: Live feed
        logger.info("3. Testing live data feed...")

        def on_data(tick):
            logger.info(f"📊 Live Update: ₹{tick['ltp']:.2f} at {tick['timestamp'].strftime('%H:%M:%S')}")

        nse.start_live_feed(on_data)

        # Let it run for 2 minutes
        logger.info("🔄 Running live feed for 2 minutes...")
        time.sleep(120)

        nse.stop_live_feed()
        logger.info("✅ NSE scraper test completed!")

    except Exception as e:
        logger.error(f"❌ NSE scraper test failed: {str(e)}")


if __name__ == "__main__":
    test_nse_scraper()
