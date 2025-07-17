import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from ..utils.logger import TradingLogger
from ..utils.config_manager import ConfigManager


class TechnicalIndicators:
    """Technical indicators calculator for stock market analysis"""

    def __init__(self):
        self.logger = TradingLogger().get_logger()
        self.config = ConfigManager()
        self.indicators_config = self.config.get_config('indicators')

    def calculate_sma(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return data.rolling(window=period).mean()

    def calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return data.ewm(span=period).mean()

    def calculate_rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        exp1 = data.ewm(span=fast).mean()
        exp2 = data.ewm(span=slow).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line

        return {
            'macd': macd,
            'signal': signal_line,
            'histogram': histogram
        }

    def calculate_bollinger_bands(self, data: pd.Series, period: int = 20, std: float = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = self.calculate_sma(data, period)
        std_dev = data.rolling(window=period).std()

        return {
            'upper': sma + (std_dev * std),
            'middle': sma,
            'lower': sma - (std_dev * std)
        }

    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series,
                             k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()

        return {
            'k': k_percent,
            'd': d_percent
        }

    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index"""
        plus_dm = high.diff()
        minus_dm = low.diff()

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = minus_dm.abs()

        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - close.shift(1)))
        tr3 = pd.DataFrame(abs(low - close.shift(1)))
        frames = [tr1, tr2, tr3]
        tr = pd.concat(frames, axis=1, join='inner').max(axis=1)

        atr = tr.rolling(window=period).mean()

        plus_di = 100 * (plus_dm.ewm(alpha=1 / period).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1 / period).mean() / atr)

        dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
        adx = dx.ewm(alpha=1 / period).mean()

        return adx

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators for a DataFrame"""
        try:
            self.logger.info("Calculating all technical indicators...")

            # Make a copy to avoid modifying original
            result_df = df.copy()

            # Simple Moving Averages
            for period in self.indicators_config['sma_periods']:
                result_df[f'sma_{period}'] = self.calculate_sma(df['close'], period)

            # Exponential Moving Averages
            for period in self.indicators_config['ema_periods']:
                result_df[f'ema_{period}'] = self.calculate_ema(df['close'], period)

            # RSI
            result_df['rsi'] = self.calculate_rsi(df['close'], self.indicators_config['rsi_period'])

            # MACD
            macd = self.calculate_macd(
                df['close'],
                self.indicators_config['macd_fast'],
                self.indicators_config['macd_slow'],
                self.indicators_config['macd_signal']
            )
            result_df['macd'] = macd['macd']
            result_df['macd_signal'] = macd['signal']
            result_df['macd_histogram'] = macd['histogram']

            # Bollinger Bands
            bb = self.calculate_bollinger_bands(
                df['close'],
                self.indicators_config['bb_period'],
                self.indicators_config['bb_std']
            )
            result_df['bb_upper'] = bb['upper']
            result_df['bb_middle'] = bb['middle']
            result_df['bb_lower'] = bb['lower']

            # Stochastic
            stoch = self.calculate_stochastic(
                df['high'], df['low'], df['close'],
                self.indicators_config['stoch_k'],
                self.indicators_config['stoch_d']
            )
            result_df['stoch_k'] = stoch['k']
            result_df['stoch_d'] = stoch['d']

            # ADX
            result_df['adx'] = self.calculate_adx(
                df['high'], df['low'], df['close'],
                self.indicators_config['adx_period']
            )

            # Price-based indicators
            result_df['price_change'] = df['close'].pct_change()
            result_df['price_change_abs'] = df['close'].diff()
            result_df['volume_sma'] = self.calculate_sma(df['volume'], 20)
            result_df['volume_ratio'] = df['volume'] / result_df['volume_sma']

            # High-Low indicators
            result_df['hl_pct'] = (df['high'] - df['low']) / df['close']
            result_df['oc_pct'] = (df['open'] - df['close']) / df['close']

            self.logger.info(f"Calculated {len(result_df.columns) - len(df.columns)} technical indicators")
            return result_df

        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {str(e)}")
            raise
