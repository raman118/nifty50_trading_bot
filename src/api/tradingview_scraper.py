import requests
import json
import websocket
import threading
import time
from datetime import datetime
import pandas as pd
from typing import Dict, Any, Optional, Callable


class TradingViewScraper:
    """Real-time data scraper for TradingView"""

    def __init__(self, logger):
        self.logger = logger
        self.base_url = "https://scanner.tradingview.com"
        self.chart_url = "https://chartdata-feed.tradingview.com"

        # Session setup
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Origin': 'https://www.tradingview.com',
            'Referer': 'https://www.tradingview.com/'
        })

        # Data storage
        self.current_price = 0.0
        self.is_running = False
        self.data_callback = None

    def get_nifty_data(self) -> Optional[Dict[str, Any]]:
        """Get real-time Nifty 50 data from TradingView"""
        try:
            # TradingView scanner API for Nifty 50
            payload = {
                "filter": [{"left": "name", "operation": "match", "right": "NIFTY"}],
                "options": {"lang": "en"},
                "symbols": {"query": {"types": []}, "tickers": ["NSE:NIFTY"]},
                "columns": ["name", "close", "change", "change_abs", "high", "low", "volume", "market_cap_basic"],
                "sort": {"sortBy": "name", "sortOrder": "asc"},
                "range": [0, 50]
            }

            response = self.session.post(
                f"{self.base_url}/india/scan",
                json=payload,
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()

                if data.get('data') and len(data['data']) > 0:
                    nifty_data = data['data'][0]['d']

                    # Parse TradingView response
                    current_price = float(nifty_data[1])  # close price
                    change = float(nifty_data[2])  # change %
                    change_abs = float(nifty_data[3])  # absolute change
                    high = float(nifty_data[4])  # high
                    low = float(nifty_data[5])  # low
                    volume = float(nifty_data[6]) if nifty_data[6] else 1000000

                    self.current_price = current_price

                    processed_data = {
                        'symbol': 'NIFTY 50',
                        'ltp': current_price,
                        'high': high,
                        'low': low,
                        'change': change_abs,
                        'pChange': change,
                        'volume': int(volume),
                        'open': current_price - change_abs,  # Estimated
                        'close': current_price - change_abs,  # Previous close
                        'timestamp': datetime.now(),
                        'source': 'TRADINGVIEW'
                    }

                    self.logger.info(f"📊 TradingView: Nifty 50 = ₹{current_price:.2f} ({change:+.2f}%)")
                    return processed_data

            return None

        except Exception as e:
            self.logger.error(f"TradingView data fetch error: {str(e)}")
            return None

    def get_chart_data(self, symbol: str = "NSE:NIFTY", timeframe: str = "1") -> Optional[pd.DataFrame]:
        """Get historical chart data from TradingView"""
        try:
            # TradingView chart API
            url = f"{self.chart_url}/api/v1/history"
            params = {
                'symbol': symbol,
                'resolution': timeframe,
                'from': int(time.time()) - (24 * 60 * 60),  # Last 24 hours
                'to': int(time.time())
            }

            response = self.session.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()

                if data.get('s') == 'ok':
                    df_data = []

                    for i in range(len(data['t'])):
                        df_data.append({
                            'timestamp': pd.to_datetime(data['t'][i], unit='s'),
                            'open': data['o'][i],
                            'high': data['h'][i],
                            'low': data['l'][i],
                            'close': data['c'][i],
                            'volume': data['v'][i] if 'v' in data else 1000
                        })

                    df = pd.DataFrame(df_data)
                    df = df.set_index('timestamp')

                    self.logger.info(f"Retrieved {len(df)} TradingView chart records")
                    return df

            return None

        except Exception as e:
            self.logger.error(f"TradingView chart data error: {str(e)}")
            return None

    def start_live_feed(self, callback: Callable):
        """Start live TradingView data feed"""
        self.data_callback = callback
        self.is_running = True

        def tv_data_updater():
            """Background thread for TradingView updates"""
            consecutive_failures = 0
            max_failures = 3

            while self.is_running:
                try:
                    tv_data = self.get_nifty_data()

                    if tv_data:
                        consecutive_failures = 0

                        # Convert to standard tick format
                        tick_data = {
                            'symbol': tv_data['symbol'],
                            'ltp': tv_data['ltp'],
                            'timestamp': tv_data['timestamp'],
                            'volume': tv_data['volume'],
                            'change': tv_data['change'],
                            'high': tv_data['high'],
                            'low': tv_data['low'],
                            'source': 'TRADINGVIEW'
                        }

                        if self.data_callback:
                            self.data_callback(tick_data)

                        # Update every 15-30 seconds
                        sleep_time = 20

                    else:
                        consecutive_failures += 1
                        self.logger.warning(f"TradingView fetch failed ({consecutive_failures}/{max_failures})")
                        sleep_time = 60

                    time.sleep(sleep_time)

                except Exception as e:
                    self.logger.error(f"TradingView updater error: {str(e)}")
                    time.sleep(60)

            self.logger.info("TradingView feed stopped")

        # Start background thread
        self.feed_thread = threading.Thread(target=tv_data_updater, daemon=True)
        self.feed_thread.start()

        self.logger.info("🌐 TradingView live feed started")

    def stop_live_feed(self):
        """Stop TradingView live feed"""
        self.is_running = False
