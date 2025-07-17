import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import threading
import time

from ..models.lstm_model import LSTMModel
from ..data.preprocessor import DataPreprocessor
from ..features.technical_indicators import TechnicalIndicators
from ..utils.logger import TradingLogger
from ..utils.config_manager import ConfigManager


class Signal(Enum):
    """Trading signal types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    STRONG_BUY = "STRONG_BUY"
    STRONG_SELL = "STRONG_SELL"


class SignalGenerator:
    """Advanced trading signal generator using ML predictions and technical analysis"""

    def __init__(self):
        self.logger = TradingLogger().get_logger()
        self.config = ConfigManager()
        self.trading_config = self.config.get_trading_config()

        # Initialize components
        self.model = LSTMModel()
        self.preprocessor = DataPreprocessor()
        self.indicators = TechnicalIndicators()

        # Load trained model and scalers
        self.load_model_components()

        # Signal generation parameters
        self.min_confidence = self.trading_config.get('min_confidence_threshold', 0.7)
        self.signal_history = []
        self.current_signal = Signal.HOLD
        self.signal_strength = 0.0

        # Price tracking
        self.price_history = []
        self.last_signal_time = None
        self.signal_cooldown = 60  # seconds between signals

    def load_model_components(self):
        """Load trained model and preprocessing components"""
        try:
            # Load trained model
            self.model.load_model()
            self.logger.info("✅ Trained model loaded successfully")

            # Load scalers
            scalers_loaded = self.preprocessor.load_scalers()
            if scalers_loaded:
                self.logger.info("✅ Scalers loaded successfully")
            else:
                self.logger.warning("⚠️ Scalers not found - using default scaling")

        except Exception as e:
            self.logger.error(f"❌ Error loading model components: {str(e)}")
            raise

    def prepare_real_time_data(self, recent_data: pd.DataFrame) -> np.ndarray:
        """Prepare real-time data for model prediction"""
        try:
            # Calculate technical indicators
            data_with_indicators = self.indicators.calculate_all_indicators(recent_data)

            # Prepare features (same as training)
            features_df = self.preprocessor.prepare_features(data_with_indicators)

            # Get the most recent sequence
            sequence_length = self.config.get_model_config()['sequence_length']

            if len(features_df) < sequence_length:
                raise ValueError(f"Not enough data: need {sequence_length}, got {len(features_df)}")

            # Select feature columns (exclude target)
            if hasattr(self.preprocessor, 'feature_columns') and self.preprocessor.feature_columns:
                feature_cols = self.preprocessor.feature_columns
            else:
                feature_cols = [col for col in features_df.columns if col != 'close']

            # Get the latest sequence
            latest_sequence = features_df[feature_cols].iloc[-sequence_length:].values

            # Scale the features
            if hasattr(self.preprocessor, 'feature_scaler'):
                latest_sequence_scaled = self.preprocessor.feature_scaler.transform(latest_sequence)
            else:
                latest_sequence_scaled = latest_sequence

            # Reshape for model input
            model_input = latest_sequence_scaled.reshape(1, sequence_length, -1)

            return model_input

        except Exception as e:
            self.logger.error(f"Error preparing real-time data: {str(e)}")
            raise

    def get_ml_prediction(self, model_input: np.ndarray) -> Tuple[float, float]:
        """Get ML model prediction and confidence"""
        try:
            # Get prediction
            prediction_scaled = self.model.predict_single(model_input)

            # Convert back to original scale
            if hasattr(self.preprocessor, 'target_scaler'):
                prediction = self.preprocessor.target_scaler.inverse_transform([[prediction_scaled]])[0][0]
            else:
                prediction = prediction_scaled

            # Calculate confidence based on model's certainty
            # This is a simplified confidence calculation
            confidence = min(0.95, max(0.5, 1.0 - abs(prediction_scaled - 0.5) * 2))

            return float(prediction), float(confidence)

        except Exception as e:
            self.logger.error(f"Error getting ML prediction: {str(e)}")
            raise

    def calculate_technical_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical analysis signals"""
        try:
            latest_data = data.iloc[-1]
            signals = {}

            # RSI signals
            rsi = latest_data.get('rsi', 50)
            if rsi > 70:
                signals['rsi'] = {'signal': 'SELL', 'strength': min(1.0, (rsi - 70) / 30)}
            elif rsi < 30:
                signals['rsi'] = {'signal': 'BUY', 'strength': min(1.0, (30 - rsi) / 30)}
            else:
                signals['rsi'] = {'signal': 'HOLD', 'strength': 0.0}

            # MACD signals
            macd = latest_data.get('macd', 0)
            macd_signal = latest_data.get('macd_signal', 0)
            macd_histogram = latest_data.get('macd_histogram', 0)

            if macd > macd_signal and macd_histogram > 0:
                signals['macd'] = {'signal': 'BUY', 'strength': min(1.0, abs(macd_histogram) / 100)}
            elif macd < macd_signal and macd_histogram < 0:
                signals['macd'] = {'signal': 'SELL', 'strength': min(1.0, abs(macd_histogram) / 100)}
            else:
                signals['macd'] = {'signal': 'HOLD', 'strength': 0.0}

            # Bollinger Bands signals
            close_price = latest_data.get('close', 0)
            bb_upper = latest_data.get('bb_upper', close_price)
            bb_lower = latest_data.get('bb_lower', close_price)
            bb_middle = latest_data.get('bb_middle', close_price)

            if close_price > bb_upper:
                signals['bollinger'] = {'signal': 'SELL', 'strength': 0.7}
            elif close_price < bb_lower:
                signals['bollinger'] = {'signal': 'BUY', 'strength': 0.7}
            else:
                signals['bollinger'] = {'signal': 'HOLD', 'strength': 0.0}

            # Moving Average signals
            sma_20 = latest_data.get('sma_20', close_price)
            sma_50 = latest_data.get('sma_50', close_price)

            if close_price > sma_20 > sma_50:
                signals['sma'] = {'signal': 'BUY', 'strength': 0.6}
            elif close_price < sma_20 < sma_50:
                signals['sma'] = {'signal': 'SELL', 'strength': 0.6}
            else:
                signals['sma'] = {'signal': 'HOLD', 'strength': 0.0}

            # Stochastic signals
            stoch_k = latest_data.get('stoch_k', 50)
            stoch_d = latest_data.get('stoch_d', 50)

            if stoch_k > 80 and stoch_d > 80:
                signals['stochastic'] = {'signal': 'SELL', 'strength': 0.5}
            elif stoch_k < 20 and stoch_d < 20:
                signals['stochastic'] = {'signal': 'BUY', 'strength': 0.5}
            else:
                signals['stochastic'] = {'signal': 'HOLD', 'strength': 0.0}

            return signals

        except Exception as e:
            self.logger.error(f"Error calculating technical signals: {str(e)}")
            return {}

    def combine_signals(self, ml_prediction: float, current_price: float,
                        ml_confidence: float, technical_signals: Dict[str, Any]) -> Dict[str, Any]:
        """Combine ML prediction with technical analysis"""
        try:
            # Calculate price change prediction
            price_change_pct = ((ml_prediction - current_price) / current_price) * 100

            # ML signal based on predicted price change
            if price_change_pct > 0.5:
                ml_signal = 'BUY'
                ml_strength = min(1.0, abs(price_change_pct) / 2.0)
            elif price_change_pct < -0.5:
                ml_signal = 'SELL'
                ml_strength = min(1.0, abs(price_change_pct) / 2.0)
            else:
                ml_signal = 'HOLD'
                ml_strength = 0.0

            # Weight ML signal by confidence
            ml_strength *= ml_confidence

            # Combine technical signals
            tech_buy_strength = 0.0
            tech_sell_strength = 0.0
            tech_hold_strength = 0.0

            for indicator, signal_info in technical_signals.items():
                signal_type = signal_info['signal']
                strength = signal_info['strength']

                if signal_type == 'BUY':
                    tech_buy_strength += strength
                elif signal_type == 'SELL':
                    tech_sell_strength += strength
                else:
                    tech_hold_strength += strength

            # Normalize technical strengths
            total_tech_signals = len(technical_signals)
            if total_tech_signals > 0:
                tech_buy_strength /= total_tech_signals
                tech_sell_strength /= total_tech_signals
                tech_hold_strength /= total_tech_signals

            # Combine ML and technical signals (60% ML, 40% technical)
            ml_weight = 0.6
            tech_weight = 0.4

            combined_buy_strength = (ml_weight * (ml_strength if ml_signal == 'BUY' else 0) +
                                     tech_weight * tech_buy_strength)
            combined_sell_strength = (ml_weight * (ml_strength if ml_signal == 'SELL' else 0) +
                                      tech_weight * tech_sell_strength)
            combined_hold_strength = (ml_weight * (ml_strength if ml_signal == 'HOLD' else 0) +
                                      tech_weight * tech_hold_strength)

            # Determine final signal
            max_strength = max(combined_buy_strength, combined_sell_strength, combined_hold_strength)

            if max_strength < self.min_confidence:
                final_signal = Signal.HOLD
                final_strength = combined_hold_strength
            elif combined_buy_strength == max_strength:
                if combined_buy_strength > 0.8:
                    final_signal = Signal.STRONG_BUY
                else:
                    final_signal = Signal.BUY
                final_strength = combined_buy_strength
            elif combined_sell_strength == max_strength:
                if combined_sell_strength > 0.8:
                    final_signal = Signal.STRONG_SELL
                else:
                    final_signal = Signal.SELL
                final_strength = combined_sell_strength
            else:
                final_signal = Signal.HOLD
                final_strength = combined_hold_strength

            return {
                'signal': final_signal,
                'strength': final_strength,
                'confidence': ml_confidence,
                'ml_prediction': ml_prediction,
                'current_price': current_price,
                'price_change_pct': price_change_pct,
                'ml_signal': ml_signal,
                'technical_signals': technical_signals,
                'combined_strengths': {
                    'buy': combined_buy_strength,
                    'sell': combined_sell_strength,
                    'hold': combined_hold_strength
                }
            }

        except Exception as e:
            self.logger.error(f"Error combining signals: {str(e)}")
            raise

    def generate_signal(self, recent_data: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """Generate trading signal from recent market data"""
        try:
            # Check cooldown period
            current_time = datetime.now()
            if (self.last_signal_time and
                    (current_time - self.last_signal_time).total_seconds() < self.signal_cooldown):
                return {
                    'signal': self.current_signal,
                    'strength': self.signal_strength,
                    'message': 'Signal in cooldown period',
                    'timestamp': current_time
                }

            # Prepare data for ML model
            model_input = self.prepare_real_time_data(recent_data)

            # Get ML prediction
            ml_prediction, ml_confidence = self.get_ml_prediction(model_input)

            # Calculate technical signals
            technical_signals = self.calculate_technical_signals(recent_data)

            # Combine all signals
            combined_result = self.combine_signals(
                ml_prediction, current_price, ml_confidence, technical_signals
            )

            # Update current signal
            self.current_signal = combined_result['signal']
            self.signal_strength = combined_result['strength']
            self.last_signal_time = current_time

            # Add timestamp and additional info
            signal_result = {
                **combined_result,
                'timestamp': current_time,
                'signal_id': len(self.signal_history) + 1,
                'model_confidence': ml_confidence,
                'technical_agreement': self.calculate_technical_agreement(technical_signals),
                'market_sentiment': self.analyze_market_sentiment(recent_data)
            }

            # Store in history
            self.signal_history.append(signal_result)

            # Keep only last 100 signals
            if len(self.signal_history) > 100:
                self.signal_history = self.signal_history[-100:]

            # Log the signal
            self.logger.info(
                f"🎯 SIGNAL GENERATED: {signal_result['signal'].value} | "
                f"Strength: {signal_result['strength']:.2f} | "
                f"Confidence: {signal_result['confidence']:.2f} | "
                f"Price: ₹{current_price:.2f} | "
                f"Predicted: ₹{ml_prediction:.2f}"
            )

            return signal_result

        except Exception as e:
            self.logger.error(f"Error generating signal: {str(e)}")
            return {
                'signal': Signal.HOLD,
                'strength': 0.0,
                'error': str(e),
                'timestamp': datetime.now()
            }

    def calculate_technical_agreement(self, technical_signals: Dict[str, Any]) -> float:
        """Calculate agreement between technical indicators"""
        try:
            if not technical_signals:
                return 0.0

            signal_types = [signal['signal'] for signal in technical_signals.values()]

            # Count occurrences of each signal type
            buy_count = signal_types.count('BUY')
            sell_count = signal_types.count('SELL')
            hold_count = signal_types.count('HOLD')

            total_signals = len(signal_types)

            # Calculate agreement as the percentage of indicators agreeing
            max_agreement = max(buy_count, sell_count, hold_count)
            agreement_percentage = max_agreement / total_signals if total_signals > 0 else 0.0

            return agreement_percentage

        except Exception as e:
            self.logger.error(f"Error calculating technical agreement: {str(e)}")
            return 0.0

    def analyze_market_sentiment(self, recent_data: pd.DataFrame) -> str:
        """Analyze overall market sentiment"""
        try:
            if len(recent_data) < 10:
                return "NEUTRAL"

            # Calculate recent price movement
            recent_closes = recent_data['close'].tail(10)
            price_change = (recent_closes.iloc[-1] - recent_closes.iloc[0]) / recent_closes.iloc[0]

            # Calculate volume trend
            recent_volumes = recent_data['volume'].tail(10)
            volume_trend = recent_volumes.iloc[-1] / recent_volumes.mean()

            # Determine sentiment
            if price_change > 0.02 and volume_trend > 1.2:
                return "BULLISH"
            elif price_change < -0.02 and volume_trend > 1.2:
                return "BEARISH"
            elif abs(price_change) < 0.01:
                return "NEUTRAL"
            else:
                return "MIXED"

        except Exception as e:
            self.logger.error(f"Error analyzing market sentiment: {str(e)}")
            return "UNKNOWN"

    def get_signal_history(self, last_n: int = 10) -> List[Dict[str, Any]]:
        """Get recent signal history"""
        return self.signal_history[-last_n:] if self.signal_history else []

    def get_signal_statistics(self) -> Dict[str, Any]:
        """Get statistics about generated signals"""
        try:
            if not self.signal_history:
                return {'total_signals': 0}

            signal_types = [signal['signal'].value for signal in self.signal_history]

            stats = {
                'total_signals': len(self.signal_history),
                'buy_signals': signal_types.count('BUY'),
                'sell_signals': signal_types.count('SELL'),
                'hold_signals': signal_types.count('HOLD'),
                'strong_buy_signals': signal_types.count('STRONG_BUY'),
                'strong_sell_signals': signal_types.count('STRONG_SELL'),
                'average_strength': np.mean([signal['strength'] for signal in self.signal_history]),
                'average_confidence': np.mean([signal['confidence'] for signal in self.signal_history]),
                'last_signal': self.signal_history[-1]['signal'].value if self.signal_history else None,
                'last_signal_time': self.signal_history[-1]['timestamp'] if self.signal_history else None
            }

            return stats

        except Exception as e:
            self.logger.error(f"Error calculating signal statistics: {str(e)}")
            return {'error': str(e)}
