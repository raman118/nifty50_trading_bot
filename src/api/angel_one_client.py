import asyncio
import websockets
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
import threading
import time
import requests
import pyotp
import hashlib
import hmac
import base64
import numpy as np


# Mock SmartConnect - No external dependencies needed
class SmartConnect:
    def __init__(self, api_key):
        self.api_key = api_key
        print(f"Mock SmartConnect initialized")

    def generateSession(self, client_code, password, totp):
        return {'status': False, 'message': 'Mock session - using placeholder credentials'}

    def setSessionExpiryHook(self, callback):
        pass

    def getProfile(self):
        return {'status': True, 'data': {'name': 'Mock User'}}

    def renewAccessToken(self, refresh_token, api_key):
        return {'status': False, 'message': 'Mock refresh'}

    def ltpData(self, exchange, tradingsymbol, symboltoken):
        import random
        return {'status': True, 'data': {'ltp': round(18000 + random.uniform(-500, 500), 2)}}

    def getCandleData(self, params):
        return {'status': False, 'message': 'Mock historical data'}

    def getMarketData(self, mode, exchangeTokens):
        return {'status': False, 'message': 'Mock market data'}


from ..utils.logger import TradingLogger
from ..utils.config_manager import ConfigManager


class AngelOneClient:
    """Comprehensive Angel One API client with Enhanced Mock Data (NSE-Free Version)"""

    def __init__(self):
        self.logger = TradingLogger().get_logger()
        self.config_manager = ConfigManager()
        self.api_config = self.config_manager.get_api_config('angel_one')
        self.websocket_config = self.config_manager.get_api_config('websocket')
        self.trading_config = self.config_manager.get_trading_config()

        # API client
        self.smart_api = None
        self.auth_token = None
        self.refresh_token = None
        self.feed_token = None

        # WebSocket
        self.websocket_client = None
        self.websocket_thread = None
        self.is_websocket_connected = False
        self.data_callback = None
        self.reconnect_attempts = 0

        # Session management
        self.session_active = False
        self.last_heartbeat = None

        # Data storage
        self.current_price = None
        self.price_history = []
        self.tick_data = []

        # Enhanced realistic market simulator
        self.base_price = 18200.0  # Starting Nifty 50 price
        self.price_trend = 1
        self.volatility_factor = 1.0
        self.session_open = self.base_price
        self.session_high = self.base_price
        self.session_low = self.base_price

        self.initialize_api()

    def initialize_api(self):
        """Initialize Angel One Smart API connection (Enhanced Mock Version)"""
        try:
            self.logger.info("Initializing Angel One API connection (Mock Mode)...")

            # Initialize Mock SmartConnect
            self.smart_api = SmartConnect(api_key=self.api_config['api_key'])

            # Mock session - always fails for demo
            self.session_active = False
            self.logger.info("Mock Angel One API initialized - using enhanced realistic mock data")

            # Start mock heartbeat monitor
            self.start_heartbeat_monitor()

        except Exception as e:
            self.logger.error(f"Failed to initialize Angel One API: {str(e)}")
            self.session_active = False

    def generate_totp(self) -> str:
        """Generate TOTP token for authentication (Mock Version)"""
        try:
            return "123456"
        except Exception as e:
            self.logger.error(f"Failed to generate TOTP: {str(e)}")
            return "123456"

    def session_expired_callback(self):
        """Handle session expiry (Mock Version)"""
        self.logger.warning("Mock session expired")
        self.session_active = False

    def refresh_session(self):
        """Refresh authentication session (Mock Version)"""
        try:
            self.logger.info("Mock session refresh")
            self.session_active = False
        except Exception as e:
            self.logger.error(f"Failed to refresh session: {str(e)}")

    def start_heartbeat_monitor(self):
        """Start heartbeat monitoring thread (Enhanced Version)"""

        def heartbeat_worker():
            while True:
                try:
                    self.last_heartbeat = datetime.now()
                    self.logger.debug("Enhanced mock heartbeat successful")
                except Exception as e:
                    self.logger.error(f"Mock heartbeat error: {str(e)}")

                time.sleep(self.websocket_config['heartbeat_interval'])

        heartbeat_thread = threading.Thread(target=heartbeat_worker, daemon=True)
        heartbeat_thread.start()

    def is_market_hours(self) -> bool:
        """Check if current time is within market hours"""
        now = datetime.now()

        # Market hours: 9:15 AM to 3:30 PM IST, Monday to Friday
        if now.weekday() > 4:  # Saturday = 5, Sunday = 6
            return False

        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)

        return market_open <= now <= market_close

    def get_realistic_price_movement(self) -> float:
        """Generate realistic price movement based on market behavior"""
        now = datetime.now()

        # Different volatility based on time of day
        if 9 <= now.hour <= 10:  # Opening hour - high volatility
            self.volatility_factor = 2.0
        elif 14 <= now.hour <= 15:  # Closing hour - high volatility
            self.volatility_factor = 1.8
        elif 11 <= now.hour <= 14:  # Mid-day - normal volatility
            self.volatility_factor = 1.0
        else:  # After hours - low volatility
            self.volatility_factor = 0.3

        # Base price movement
        trend_movement = np.random.normal(0, 15) * self.volatility_factor

        # Add some momentum (trending behavior)
        momentum = self.price_trend * np.random.uniform(0, 10)

        # Occasionally reverse trend (10% chance)
        if np.random.random() < 0.1:
            self.price_trend *= -1

        # Combine movements
        total_movement = trend_movement + momentum

        # Limit extreme movements
        total_movement = max(-100, min(100, total_movement))

        return total_movement

    def get_historical_data(self, symbol: str, timeframe: str,
                            from_date: datetime, to_date: datetime) -> pd.DataFrame:
        """Fetch enhanced mock historical data"""
        try:
            self.logger.info(f"Generating enhanced realistic historical data for {symbol}")
            return self.get_enhanced_mock_historical_data(from_date, to_date)
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {str(e)}")
            return self.get_enhanced_mock_historical_data(from_date, to_date)

    def get_enhanced_mock_historical_data(self, from_date: datetime, to_date: datetime) -> pd.DataFrame:
        """Generate enhanced realistic mock historical data"""

        # Generate date range (every minute)
        date_range = pd.date_range(start=from_date, end=to_date, freq='1min')

        # More realistic base prices around current market levels
        base_prices = [17850, 17920, 18100, 18050, 17980, 18200, 18150]
        current_base = base_prices[datetime.now().day % len(base_prices)]

        data = []
        current_price = current_base

        for i, timestamp in enumerate(date_range):
            # More sophisticated price movement
            time_factor = i / 100.0

            # Multiple trend components
            daily_trend = np.sin(time_factor / 10) * 100  # Daily trend
            hourly_volatility = np.sin(time_factor) * 50  # Hourly volatility
            random_noise = np.random.normal(0, 25)  # Random noise

            # Market hours effect (higher volatility during market hours)
            hour = timestamp.hour
            if 9 <= hour <= 15:  # Market hours
                volatility_multiplier = 1.5
            else:
                volatility_multiplier = 0.3

            price_change = (daily_trend + hourly_volatility + random_noise) * volatility_multiplier
            current_price += price_change * 0.01  # Smaller incremental changes

            # Keep price within realistic bounds
            current_price = max(16000, min(20000, current_price))

            # Generate OHLC with realistic relationships
            open_price = current_price + np.random.normal(0, 5)
            high_price = max(open_price, current_price) + abs(np.random.normal(0, 15))
            low_price = min(open_price, current_price) - abs(np.random.normal(0, 15))
            close_price = current_price + np.random.normal(0, 3)
            volume = np.random.randint(50000, 500000)

            data.append({
                'timestamp': timestamp,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': volume
            })

        df = pd.DataFrame(data)
        df = df.set_index('timestamp')

        self.logger.info(f"Generated {len(df)} enhanced mock historical records")
        return df

    def get_current_price(self, symbol: str) -> float:
        """Get current market price (Enhanced Realistic Mock Version)"""
        try:
            # Apply realistic price movement
            price_movement = self.get_realistic_price_movement()
            self.base_price += price_movement * 0.001  # Small incremental changes

            # Keep within realistic bounds
            self.base_price = max(17000, min(19000, self.base_price))

            # Update session tracking
            self.session_high = max(self.session_high, self.base_price)
            self.session_low = min(self.session_low, self.base_price)

            self.current_price = round(self.base_price, 2)
            self.logger.debug(f"Enhanced realistic price for {symbol}: ₹{self.current_price}")
            return self.current_price

        except Exception as e:
            self.logger.error(f"Error getting current price: {str(e)}")
            import random
            mock_price = 18000 + random.uniform(-500, 500)
            self.current_price = round(mock_price, 2)
            return self.current_price

    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive market data (Enhanced Realistic Mock Version)"""
        try:
            current_price = self.get_current_price(symbol)

            # Generate realistic market data based on current session
            open_price = self.session_open
            high_price = self.session_high
            low_price = self.session_low
            previous_close = self.session_open + np.random.normal(0, 20)

            volume = np.random.randint(1000000, 10000000)
            change = current_price - previous_close
            change_percent = (change / previous_close) * 100

            market_data = {
                'symbol': symbol,
                'ltp': current_price,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(previous_close, 2),
                'volume': volume,
                'change': round(change, 2),
                'change_percent': round(change_percent, 2),
                'timestamp': datetime.now(),
                'source': 'ENHANCED_MOCK'
            }

            self.logger.debug(f"Enhanced market data for {symbol}: {market_data}")
            return market_data
        except Exception as e:
            self.logger.error(f"Error getting market data: {str(e)}")
            return self.get_fallback_market_data(symbol)

    def get_fallback_market_data(self, symbol: str) -> Dict[str, Any]:
        """Fallback market data"""
        import random
        current_price = 18000 + random.uniform(-500, 500)

        return {
            'symbol': symbol,
            'ltp': round(current_price, 2),
            'open': round(current_price + random.uniform(-100, 100), 2),
            'high': round(current_price + random.uniform(0, 200), 2),
            'low': round(current_price - random.uniform(0, 200), 2),
            'close': round(current_price + random.uniform(-50, 50), 2),
            'volume': random.randint(1000000, 5000000),
            'change': round(random.uniform(-200, 200), 2),
            'change_percent': round(random.uniform(-2, 2), 2),
            'timestamp': datetime.now(),
            'source': 'FALLBACK_MOCK'
        }

    def start_websocket_feed(self, callback: Callable[[Dict], None]):
        """Start WebSocket feed for real-time data (Enhanced Mock Version)"""
        self.data_callback = callback
        self.start_enhanced_mock_websocket_feed()

    def start_enhanced_mock_websocket_feed(self):
        """Start enhanced realistic mock WebSocket feed"""

        def enhanced_mock_feed_worker():
            self.logger.info("🚀 Starting enhanced realistic market data feed...")

            while self.is_websocket_connected:
                try:
                    # Apply realistic price movement
                    price_movement = self.get_realistic_price_movement()
                    self.base_price += price_movement * 0.001

                    # Keep price within realistic bounds
                    self.base_price = max(17000, min(19000, self.base_price))

                    # Update session tracking
                    self.session_high = max(self.session_high, self.base_price)
                    self.session_low = min(self.session_low, self.base_price)

                    tick_data = {
                        'symbol': self.trading_config['symbol'],
                        'ltp': round(self.base_price, 2),
                        'timestamp': datetime.now(),
                        'volume': np.random.randint(1000, 10000),
                        'change': round(price_movement, 2),
                        'high': round(self.session_high, 2),
                        'low': round(self.session_low, 2),
                        'source': 'ENHANCED_MOCK'
                    }

                    # Update current price
                    self.current_price = tick_data['ltp']

                    # Store tick data
                    self.tick_data.append(tick_data)

                    # Keep only last 1000 ticks
                    if len(self.tick_data) > 1000:
                        self.tick_data = self.tick_data[-1000:]

                    # Call callback if provided
                    if self.data_callback:
                        self.data_callback(tick_data)

                    # Update frequency: faster during market hours
                    if self.is_market_hours():
                        time.sleep(1)  # Every second during market hours
                    else:
                        time.sleep(5)  # Every 5 seconds after hours

                except Exception as e:
                    self.logger.error(f"Enhanced mock feed error: {str(e)}")
                    time.sleep(5)

        self.websocket_thread = threading.Thread(target=enhanced_mock_feed_worker, daemon=True)
        self.websocket_thread.start()
        self.is_websocket_connected = True

        self.logger.info("Enhanced realistic mock WebSocket feed started")

    def get_tick_data(self, last_n: int = 100) -> List[Dict[str, Any]]:
        """Get recent tick data"""
        return self.tick_data[-last_n:] if self.tick_data else []

    def get_price_history(self, last_n: int = 100) -> List[float]:
        """Get price history"""
        return [tick['ltp'] for tick in self.tick_data[-last_n:]]

    def is_market_open(self) -> bool:
        """Check if market is currently open (Enhanced Version)"""
        return self.is_market_hours()

    def stop_websocket_feed(self):
        """Stop WebSocket feed"""
        self.is_websocket_connected = False
        if self.websocket_thread:
            self.websocket_thread.join(timeout=2)

        self.logger.info("Enhanced mock WebSocket feed stopped")

    def get_connection_status(self) -> Dict[str, Any]:
        """Get connection status information"""
        return {
            'session_active': self.session_active,
            'websocket_connected': self.is_websocket_connected,
            'last_heartbeat': self.last_heartbeat,
            'current_price': self.current_price,
            'reconnect_attempts': self.reconnect_attempts,
            'market_open': self.is_market_open(),
            'mode': 'ENHANCED_MOCK',
            'data_source': 'REALISTIC_SIMULATION',
            'session_high': self.session_high,
            'session_low': self.session_low,
            'price_trend': 'UP' if self.price_trend > 0 else 'DOWN'
        }

    def __del__(self):
        """Cleanup on object destruction"""
        try:
            self.stop_websocket_feed()
        except:
            pass
