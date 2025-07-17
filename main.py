import asyncio
import threading
import time
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
from pathlib import Path

# Import all components
from src.api.angel_one_client import AngelOneClient
from src.data.data_collector import DataCollector
from src.data.preprocessor import DataPreprocessor
from src.models.model_trainer import ModelTrainer
from src.models.lstm_model import LSTMModel
from src.trading.signal_generator import SignalGenerator
from src.trading.risk_manager import RiskManager
from src.features.technical_indicators import TechnicalIndicators
from src.utils.logger import TradingLogger
from src.utils.config_manager import ConfigManager

# Import web dashboard
from src.web.dashboard import TradingDashboard


class NiftyTradingBot:
    """Complete Nifty 50 Trading Bot with Machine Learning"""

    def __init__(self):
        self.logger = TradingLogger().get_logger()
        self.config = ConfigManager()

        # Initialize components
        self.api_client = AngelOneClient()
        self.data_collector = DataCollector()
        self.preprocessor = DataPreprocessor()
        self.model_trainer = ModelTrainer()
        self.signal_generator = None  # Will be initialized after model training
        self.risk_manager = RiskManager()

        # Runtime state
        self.is_running = False
        self.trading_active = False
        self.market_data_thread = None
        self.trading_thread = None

        # Data storage
        self.live_data_buffer = []
        self.current_price = 0.0
        self.last_signal_time = None

        # Performance tracking
        self.daily_stats = {
            'start_time': datetime.now(),
            'signals_generated': 0,
            'trades_executed': 0,
            'total_pnl': 0.0
        }

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        self.logger.info("🤖 Nifty 50 Trading Bot Initialized")

    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"📶 Received signal {signum}, shutting down gracefully...")
        self.shutdown()
        sys.exit(0)

    def check_and_train_model(self, retrain: bool = False) -> bool:
        """Check if model exists, train if needed"""
        try:
            model_path = Path("models/nifty_lstm_model.h5")

            if model_path.exists() and not retrain:
                self.logger.info("✅ Trained model found, skipping training")
                return True

            self.logger.info("🔄 Training model...")

            # Train model with reasonable dataset
            results = self.model_trainer.run_complete_training(
                symbol="NIFTY 50",
                days_back=60,  # 60 days of data
                model_type='lstm'
            )

            self.logger.info("✅ Model training completed successfully")
            self.logger.info(f"📊 Model Performance:")
            self.logger.info(f"   - RMSE: {results['evaluation_metrics']['rmse']:.4f}")
            self.logger.info(f"   - MAE: {results['evaluation_metrics']['mae']:.4f}")
            self.logger.info(f"   - Direction Accuracy: {results['evaluation_metrics']['direction_accuracy']:.2f}%")

            return True

        except Exception as e:
            self.logger.error(f"❌ Error in model training: {str(e)}")
            return False

    def initialize_trading_components(self):
        """Initialize trading components after model is ready"""
        try:
            # Initialize signal generator (requires trained model)
            self.signal_generator = SignalGenerator()
            self.logger.info("✅ Signal generator initialized")

            # Initialize risk manager
            self.risk_manager = RiskManager()
            self.logger.info("✅ Risk manager initialized")

            return True

        except Exception as e:
            self.logger.error(f"❌ Error initializing trading components: {str(e)}")
            return False

    def start_market_data_feed(self):
        """Start real-time market data feed"""
        try:
            def on_market_data(tick_data):
                """Handle incoming market data"""
                try:
                    self.current_price = tick_data['ltp']
                    self.live_data_buffer.append(tick_data)

                    # Keep only last 1000 ticks
                    if len(self.live_data_buffer) > 1000:
                        self.live_data_buffer = self.live_data_buffer[-1000:]

                    # Log every 10th tick to avoid spam
                    if len(self.live_data_buffer) % 10 == 0:
                        self.logger.debug(f"📊 Market Data: ₹{self.current_price:.2f}")

                except Exception as e:
                    self.logger.error(f"Error processing market data: {str(e)}")

            # Start WebSocket feed
            self.api_client.start_websocket_feed(on_market_data)
            self.logger.info("📡 Market data feed started")

        except Exception as e:
            self.logger.error(f"❌ Error starting market data feed: {str(e)}")
            raise

    def get_recent_data_for_analysis(self) -> Optional[Dict[str, Any]]:
        """Get recent data formatted for analysis"""
        try:
            if len(self.live_data_buffer) < 60:  # Need at least 60 data points
                self.logger.debug("Not enough data for analysis")
                return None

            # Convert tick data to OHLC format
            recent_ticks = self.live_data_buffer[-60:]  # Last 60 ticks

            # Create DataFrame from tick data
            df_data = []
            for tick in recent_ticks:
                df_data.append({
                    'timestamp': tick['timestamp'],
                    'open': tick['ltp'],
                    'high': tick.get('high', tick['ltp']),
                    'low': tick.get('low', tick['ltp']),
                    'close': tick['ltp'],
                    'volume': tick.get('volume', 1000)
                })

            import pandas as pd
            df = pd.DataFrame(df_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')

            return df

        except Exception as e:
            self.logger.error(f"Error preparing recent data: {str(e)}")
            return None

    def trading_loop(self):
        """Main trading loop"""
        self.logger.info("🔄 Starting trading loop...")

        while self.is_running and self.trading_active:
            try:
                # Check if market is open
                if not self.api_client.is_market_open():
                    self.logger.debug("Market is closed, waiting...")
                    time.sleep(60)  # Wait 1 minute
                    continue

                # Get recent data for analysis
                recent_data = self.get_recent_data_for_analysis()
                if recent_data is None:
                    time.sleep(10)  # Wait 10 seconds
                    continue

                # Generate trading signal
                signal_result = self.signal_generator.generate_signal(recent_data, self.current_price)

                if signal_result:
                    self.daily_stats['signals_generated'] += 1
                    self.last_signal_time = datetime.now()

                    # Generate trade recommendation
                    recommendation = self.risk_manager.generate_trade_recommendation(signal_result)

                    if recommendation:
                        self.logger.info(
                            f"💡 TRADE RECOMMENDATION: {recommendation.action} {recommendation.quantity} shares")

                        # Execute trade (in simulation mode)
                        trade_result = self.risk_manager.execute_trade(recommendation)

                        if trade_result['status'] == 'EXECUTED':
                            self.daily_stats['trades_executed'] += 1
                            self.logger.info(
                                f"✅ Trade executed: {trade_result['action']} {trade_result['quantity']} at ₹{trade_result['price']:.2f}")

                        # Update portfolio stats
                        portfolio_summary = self.risk_manager.get_portfolio_summary()
                        self.daily_stats['total_pnl'] = portfolio_summary.get('daily_pnl', 0.0)

                # Wait before next iteration
                time.sleep(30)  # Check every 30 seconds

            except Exception as e:
                self.logger.error(f"❌ Error in trading loop: {str(e)}")
                time.sleep(60)  # Wait 1 minute before retrying

        self.logger.info("🔄 Trading loop stopped")

    def monitor_performance(self):
        """Monitor and log performance metrics"""
        self.logger.info("📊 Starting performance monitoring...")

        while self.is_running:
            try:
                # Get current portfolio summary
                portfolio_summary = self.risk_manager.get_portfolio_summary()

                # Get signal statistics
                signal_stats = self.signal_generator.get_signal_statistics() if self.signal_generator else {}

                # Log performance every 5 minutes
                performance_data = {
                    'timestamp': datetime.now().isoformat(),
                    'current_price': self.current_price,
                    'portfolio_value': portfolio_summary.get('portfolio_value', 0),
                    'daily_pnl': portfolio_summary.get('daily_pnl', 0),
                    'total_return': portfolio_summary.get('total_return', 0),
                    'active_positions': portfolio_summary.get('active_positions', 0),
                    'signals_generated': self.daily_stats['signals_generated'],
                    'trades_executed': self.daily_stats['trades_executed'],
                    'win_rate': portfolio_summary.get('win_rate', 0),
                    'total_signals': signal_stats.get('total_signals', 0)
                }

                # Log to performance logger
                self.logger.info(f"📈 PERFORMANCE: Portfolio: ₹{performance_data['portfolio_value']:,.2f} | "
                                 f"Daily P&L: ₹{performance_data['daily_pnl']:,.2f} | "
                                 f"Positions: {performance_data['active_positions']} | "
                                 f"Signals: {performance_data['signals_generated']} | "
                                 f"Trades: {performance_data['trades_executed']}")

                # Save performance data
                self.save_performance_data(performance_data)

                # Wait 5 minutes
                time.sleep(300)

            except Exception as e:
                self.logger.error(f"❌ Error in performance monitoring: {str(e)}")
                time.sleep(300)

    def save_performance_data(self, performance_data: Dict[str, Any]):
        """Save performance data to file"""
        try:
            performance_file = Path("logs/performance_data.json")

            # Load existing data
            if performance_file.exists():
                with open(performance_file, 'r') as f:
                    existing_data = json.load(f)
            else:
                existing_data = []

            # Add new data
            existing_data.append(performance_data)

            # Keep only last 1000 records
            if len(existing_data) > 1000:
                existing_data = existing_data[-1000:]

            # Save back to file
            with open(performance_file, 'w') as f:
                json.dump(existing_data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Error saving performance data: {str(e)}")

    def start_web_dashboard(self, host='127.0.0.1', port=5000):
        """Start the beautiful purple web dashboard"""
        try:
            self.dashboard = TradingDashboard(self)

            # Start dashboard in a separate thread
            dashboard_thread = threading.Thread(
                target=self.dashboard.run,
                kwargs={'host': host, 'port': port, 'debug': False},
                daemon=True
            )
            dashboard_thread.start()

            self.logger.info(f"🎨 Purple Trading Dashboard started at http://{host}:{port}")
            print(f"🌐 Web Dashboard available at: http://{host}:{port}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start web dashboard: {str(e)}")
            return False

    def start_trading(self, start_dashboard=True):
        """Start the complete trading system with web interface"""
        try:
            self.logger.info("🚀 Starting Nifty 50 Trading Bot...")

            # Step 1: Check and train model if needed
            if not self.check_and_train_model():
                self.logger.error("❌ Model training failed, cannot start trading")
                return False

            # Step 2: Initialize trading components
            if not self.initialize_trading_components():
                self.logger.error("❌ Failed to initialize trading components")
                return False

            # Step 3: Start market data feed
            self.start_market_data_feed()

            # Start web dashboard
            if start_dashboard:
                self.start_web_dashboard()

            # Step 4: Set runtime flags
            self.is_running = True
            self.trading_active = True

            # Step 5: Start trading thread
            self.trading_thread = threading.Thread(target=self.trading_loop, daemon=True)
            self.trading_thread.start()

            # Step 6: Start performance monitoring
            performance_thread = threading.Thread(target=self.monitor_performance, daemon=True)
            performance_thread.start()

            self.logger.info("✅ Trading bot started successfully!")
            self.logger.info("💡 Press Ctrl+C to stop the bot")

            return True

        except Exception as e:
            self.logger.error(f"❌ Error starting trading bot: {str(e)}")
            return False

    def stop_trading(self):
        """Stop trading but keep monitoring"""
        self.trading_active = False
        self.logger.info("⏸️ Trading stopped (monitoring continues)")

    def resume_trading(self):
        """Resume trading"""
        if self.is_running and not self.trading_active:
            self.trading_active = True
            self.trading_thread = threading.Thread(target=self.trading_loop, daemon=True)
            self.trading_thread.start()
            self.logger.info("▶️ Trading resumed")

    def get_status(self) -> Dict[str, Any]:
        """Get current bot status"""
        try:
            portfolio_summary = self.risk_manager.get_portfolio_summary()

            status = {
                'is_running': self.is_running,
                'trading_active': self.trading_active,
                'market_open': self.api_client.is_market_open(),
                'current_price': self.current_price,
                'data_buffer_size': len(self.live_data_buffer),
                'last_signal_time': self.last_signal_time.isoformat() if self.last_signal_time else None,
                'portfolio_value': portfolio_summary.get('portfolio_value', 0),
                'daily_pnl': portfolio_summary.get('daily_pnl', 0),
                'active_positions': portfolio_summary.get('active_positions', 0),
                'daily_stats': self.daily_stats
            }

            return status

        except Exception as e:
            self.logger.error(f"Error getting status: {str(e)}")
            return {'error': str(e)}

    def shutdown(self):
        """Graceful shutdown"""
        try:
            self.logger.info("🔄 Shutting down trading bot...")

            # Stop trading
            self.is_running = False
            self.trading_active = False

            # Stop market data feed
            if self.api_client:
                self.api_client.stop_websocket_feed()

            # Wait for threads to finish
            if self.trading_thread and self.trading_thread.is_alive():
                self.trading_thread.join(timeout=5)

            # Final performance summary
            final_summary = self.risk_manager.get_portfolio_summary()
            self.logger.info("📊 FINAL SUMMARY:")
            self.logger.info(f"   - Portfolio Value: ₹{final_summary.get('portfolio_value', 0):,.2f}")
            self.logger.info(f"   - Daily P&L: ₹{final_summary.get('daily_pnl', 0):,.2f}")
            self.logger.info(f"   - Total Return: {final_summary.get('total_return', 0):.2f}%")
            self.logger.info(f"   - Total Trades: {final_summary.get('total_trades', 0)}")
            self.logger.info(f"   - Win Rate: {final_summary.get('win_rate', 0):.1f}%")

            self.logger.info("✅ Trading bot shutdown complete")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")


def main():
    """Main function to run the trading bot"""
    try:
        # Create trading bot
        bot = NiftyTradingBot()

        # Start trading
        if bot.start_trading():
            # Keep the main thread alive
            while bot.is_running:
                try:
                    # Print status every 2 minutes
                    time.sleep(120)
                    status = bot.get_status()

                    print(f"\n{'=' * 50}")
                    print(f"🤖 NIFTY 50 TRADING BOT STATUS")
                    print(f"{'=' * 50}")
                    print(f"📊 Current Price: ₹{status['current_price']:.2f}")
                    print(f"💰 Portfolio Value: ₹{status['portfolio_value']:,.2f}")
                    print(f"📈 Daily P&L: ₹{status['daily_pnl']:,.2f}")
                    print(f"🎯 Active Positions: {status['active_positions']}")
                    print(f"📡 Signals Generated: {status['daily_stats']['signals_generated']}")
                    print(f"💼 Trades Executed: {status['daily_stats']['trades_executed']}")
                    print(f"🏢 Market Open: {status['market_open']}")
                    print(f"⚡ Bot Running: {status['is_running']}")
                    print(f"🔄 Trading Active: {status['trading_active']}")
                    print(f"{'=' * 50}\n")

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Error in main loop: {str(e)}")
                    time.sleep(60)

        # Shutdown
        bot.shutdown()

    except Exception as e:
        print(f"Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
