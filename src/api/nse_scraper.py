import requests
import json
from datetime import datetime, timedelta
import time
import threading
import pandas as pd
from typing import Dict, List, Optional, Any
import random

from ..utils.logger import TradingLogger


class NSEDataScraper:
    """Real-time NSE India data scraper for Nifty 50"""

    def __init__(self):
        self.logger = TradingLogger().get_logger()

        # NSE API endpoints
        self.base_url = "https://www.nseindia.com"
        self.nifty_url = f"{self.base_url}/api/equity-stockIndices?index=NIFTY%2050"
        self.market_status_url = f"{self.base_url}/api/marketStatus"

        # Setup session with proper headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'Referer': 'https://www.nseindia.com/market-data/live-equity-market',
            'X-Requested-With': 'XMLHttpRequest'
        })

        # Data storage
        self.current_price = 0.0
        self.is_running = False
        self.last_update = None
        self.data_callback = None

        # Initialize session
        self.initialize_session()

    def initialize_session(self):
        """Initialize NSE session with cookies"""
        try:
            # First, visit the main page to get cookies
            response = self.session.get(f"{self.base_url}/market-data/live-equity-market")
            if response.status_code == 200:
                self.logger.info("✅ NSE session initialized successfully")
                return True
            else:
                self.logger.warning(f"NSE session init failed: {response.status_code}")
                return False
        except Exception as e:
            self.logger.error(f"Error initializing NSE session: {str(e)}")
            return False

    def get_nifty_data(self) -> Optional[Dict[str, Any]]:
        """Get real-time Nifty 50 data from NSE"""
        try:
            # Add random delay to avoid rate limiting
            time.sleep(random.uniform(1, 3))

            response = self.session.get(self.nifty_url, timeout=10)

            if response.status_code == 200:
                data = response.json()

                # Find Nifty 50 data in the response
                nifty_data = None
                for item in data.get('data', []):
                    if item.get('index') == 'NIFTY 50':
                        nifty_data = item
                        break

                if nifty_data:
                    current_price = float(nifty_data['last'])
                    self.current_price = current_price
                    self.last_update = datetime.now()

                    processed_data = {
                        'symbol': 'NIFTY 50',
                        'ltp': current_price,
                        'open': float(nifty_data.get('open', current_price)),
                        'high': float(nifty_data.get('dayHigh', current_price)),
                        'low': float(nifty_data.get('dayLow', current_price)),
                        'close': float(nifty_data.get('previousClose', current_price)),
                        'change': float(nifty_data.get('change', 0)),
                        'pChange': float(nifty_data.get('pChange', 0)),
                        'volume': 1000000,  # NSE doesn't provide Nifty volume directly
                        'timestamp': datetime.now(),
                        'source': 'NSE_LIVE'
                    }

                    self.logger.debug(
                        f"📊 NSE Data: Nifty 50 = ₹{current_price:.2f} ({nifty_data.get('change', 0):+.2f})")
                    return processed_data
                else:
                    self.logger.warning("Nifty 50 data not found in NSE response")
                    return None

            elif response.status_code == 429:
                self.logger.warning("NSE rate limit hit, waiting...")
                time.sleep(30)  # Wait 30 seconds on rate limit
                return None

            else:
                self.logger.error(f"NSE API error: {response.status_code}")
                # Reinitialize session on error
                self.initialize_session()
                return None

        except requests.exceptions.Timeout:
            self.logger.warning("NSE request timeout")
            return None
        except requests.exceptions.ConnectionError:
            self.logger.warning("NSE connection error")
            return None
        except Exception as e:
            self.logger.error(f"Error fetching NSE data: {str(e)}")
            return None

    def get_market_status(self) -> Dict[str, Any]:
        """Get NSE market status"""
        try:
            response = self.session.get(self.market_status_url, timeout=10)

            if response.status_code == 200:
                data = response.json()

                # Extract market status
                market_status = {
                    'market_open': False,
                    'status': 'UNKNOWN',
                    'timestamp': datetime.now()
                }

                # Check if market data indicates open status
                for market in data.get('marketState', []):
                    if market.get('market') == 'Capital Market':
                        status = market.get('marketStatus', '').upper()
                        market_status['status'] = status
                        market_status['market_open'] = status == 'OPEN'
                        break

                return market_status
            else:
                return {'market_open': False, 'status': 'API_ERROR', 'timestamp': datetime.now()}

        except Exception as e:
            self.logger.error(f"Error getting market status: {str(e)}")
            return {'market_open': False, 'status': 'ERROR', 'timestamp': datetime.now()}

    def get_historical_data(self, days_back: int = 30) -> pd.DataFrame:
        """Get historical Nifty 50 data"""
        try:
            # NSE historical data endpoint (limited)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)

            # For now, create mock historical data based on current price
            # In a real implementation, you'd use NSE's historical API
            dates = pd.date_range(start=start_date, end=end_date, freq='1H')

            # Get current price as base
            current_data = self.get_nifty_data()
            base_price = current_data['ltp'] if current_data else 18000

            historical_data = []
            for date in dates:
                # Add some realistic variation
                price_variation = random.uniform(-200, 200)
                price = base_price + price_variation

                historical_data.append({
                    'timestamp': date,
                    'open': price + random.uniform(-50, 50),
                    'high': price + random.uniform(0, 100),
                    'low': price - random.uniform(0, 100),
                    'close': price,
                    'volume': random.randint(100000, 1000000)
                })

            df = pd.DataFrame(historical_data)
            df = df.set_index('timestamp')

            self.logger.info(f"Generated {len(df)} historical data points")
            return df

        except Exception as e:
            self.logger.error(f"Error getting historical data: {str(e)}")
            # Return empty DataFrame
            return pd.DataFrame()

    def start_live_feed(self, callback):
        """Start live NSE data feed"""
        self.data_callback = callback
        self.is_running = True

        def nse_data_updater():
            """Background thread for NSE data updates"""
            consecutive_failures = 0
            max_failures = 5

            while self.is_running:
                try:
                    # Get fresh data
                    nse_data = self.get_nifty_data()

                    if nse_data:
                        # Success - reset failure counter
                        consecutive_failures = 0

                        # Create tick data format
                        tick_data = {
                            'symbol': nse_data['symbol'],
                            'ltp': nse_data['ltp'],
                            'timestamp': nse_data['timestamp'],
                            'volume': nse_data['volume'],
                            'change': nse_data['change'],
                            'high': nse_data['high'],
                            'low': nse_data['low'],
                            'source': 'NSE_LIVE'
                        }

                        # Send to callback
                        if self.data_callback:
                            self.data_callback(tick_data)

                        # Update frequency: every 30-60 seconds during market hours
                        sleep_time = random.uniform(30, 60)

                    else:
                        # Handle failure
                        consecutive_failures += 1
                        self.logger.warning(f"NSE data fetch failed ({consecutive_failures}/{max_failures})")

                        if consecutive_failures >= max_failures:
                            self.logger.error("Too many consecutive failures, reinitializing session")
                            self.initialize_session()
                            consecutive_failures = 0

                        sleep_time = 60  # Wait longer on failure

                    # Wait before next update
                    time.sleep(sleep_time)

                except Exception as e:
                    self.logger.error(f"Error in NSE data updater: {str(e)}")
                    time.sleep(60)

            self.logger.info("NSE data feed stopped")

        # Start background thread
        self.feed_thread = threading.Thread(target=nse_data_updater, daemon=True)
        self.feed_thread.start()

        self.logger.info("🌐 NSE live data feed started")

    def stop_live_feed(self):
        """Stop live NSE data feed"""
        self.is_running = False
        self.logger.info("NSE live data feed stopped")

    def get_connection_status(self) -> Dict[str, Any]:
        """Get NSE connection status"""
        return {
            'connected': self.last_update is not None,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'current_price': self.current_price,
            'source': 'NSE_INDIA',
            'is_running': self.is_running
        }
