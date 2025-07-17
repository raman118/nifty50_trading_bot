from src.api.tradingview_scraper import TradingViewScraper
from src.utils.logger import TradingLogger


def test_tradingview():
    logger = TradingLogger().get_logger()

    # Test TradingView scraper
    tv = TradingViewScraper(logger)

    # Test current data
    data = tv.get_nifty_data()
    if data:
        print(f"✅ TradingView Nifty 50: ₹{data['ltp']:.2f}")
        print(f"   Change: {data['change']:+.2f} ({data['pChange']:+.2f}%)")

    # Test chart data
    chart_data = tv.get_chart_data()
    if chart_data is not None:
        print(f"✅ Chart data: {len(chart_data)} records")
        print(chart_data.tail())


if __name__ == "__main__":
    test_tradingview()
