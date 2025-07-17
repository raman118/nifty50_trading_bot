import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from .signal_generator import Signal
from ..utils.logger import TradingLogger
from ..utils.config_manager import ConfigManager


class PositionType(Enum):
    """Position types"""
    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"


@dataclass
class Position:
    """Trading position data class"""
    symbol: str
    position_type: PositionType
    entry_price: float
    quantity: int
    entry_time: datetime
    current_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class TradeRecommendation:
    """Trade recommendation data class"""
    action: str  # BUY, SELL, HOLD, CLOSE
    symbol: str
    quantity: int
    price: float
    stop_loss: float
    take_profit: float
    confidence: float
    reason: str
    risk_reward_ratio: float
    position_size_pct: float


class RiskManager:
    """Advanced risk management system"""

    def __init__(self):
        self.logger = TradingLogger().get_logger()
        self.config = ConfigManager()
        self.trading_config = self.config.get_trading_config()

        # Risk parameters
        self.max_position_size = self.trading_config.get('max_position_size', 100000)
        self.risk_per_trade = self.trading_config.get('risk_per_trade', 0.02)
        self.stop_loss_pct = self.trading_config.get('stop_loss_pct', 0.015)
        self.take_profit_pct = self.trading_config.get('take_profit_pct', 0.025)

        # Portfolio tracking
        self.portfolio_value = 1000000  # Initial portfolio value
        self.available_capital = self.portfolio_value
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Dict[str, Any]] = []

        # Risk metrics
        self.max_drawdown = 0.0
        self.peak_portfolio_value = self.portfolio_value
        self.daily_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0

        # Risk limits
        self.max_daily_loss = self.portfolio_value * 0.05  # 5% daily loss limit
        self.max_positions = 3  # Maximum concurrent positions
        self.correlation_limit = 0.8  # Maximum correlation between positions

    def calculate_position_size(self, signal_strength: float, current_price: float,
                                stop_loss_price: float) -> int:
        """Calculate optimal position size based on risk management"""
        try:
            # Risk amount per trade
            risk_amount = self.available_capital * self.risk_per_trade

            # Risk per share
            risk_per_share = abs(current_price - stop_loss_price)

            if risk_per_share == 0:
                return 0

            # Base position size
            base_quantity = int(risk_amount / risk_per_share)

            # Adjust based on signal strength
            strength_multiplier = min(1.0, signal_strength * 1.5)
            adjusted_quantity = int(base_quantity * strength_multiplier)

            # Apply maximum position size limit
            max_quantity = int(self.max_position_size / current_price)
            final_quantity = min(adjusted_quantity, max_quantity)

            # Ensure we don't exceed available capital
            required_capital = final_quantity * current_price
            if required_capital > self.available_capital:
                final_quantity = int(self.available_capital / current_price)

            return max(0, final_quantity)

        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return 0

    def calculate_stop_loss(self, entry_price: float, signal_type: Signal) -> float:
        """Calculate stop loss price"""
        try:
            if signal_type in [Signal.BUY, Signal.STRONG_BUY]:
                # For long positions, stop loss is below entry price
                stop_loss = entry_price * (1 - self.stop_loss_pct)
            elif signal_type in [Signal.SELL, Signal.STRONG_SELL]:
                # For short positions, stop loss is above entry price
                stop_loss = entry_price * (1 + self.stop_loss_pct)
            else:
                stop_loss = entry_price

            return round(stop_loss, 2)

        except Exception as e:
            self.logger.error(f"Error calculating stop loss: {str(e)}")
            return entry_price

    def calculate_take_profit(self, entry_price: float, signal_type: Signal) -> float:
        """Calculate take profit price"""
        try:
            if signal_type in [Signal.BUY, Signal.STRONG_BUY]:
                # For long positions, take profit is above entry price
                take_profit = entry_price * (1 + self.take_profit_pct)
            elif signal_type in [Signal.SELL, Signal.STRONG_SELL]:
                # For short positions, take profit is below entry price
                take_profit = entry_price * (1 - self.take_profit_pct)
            else:
                take_profit = entry_price

            return round(take_profit, 2)

        except Exception as e:
            self.logger.error(f"Error calculating take profit: {str(e)}")
            return entry_price

    def check_risk_limits(self) -> Dict[str, Any]:
        """Check if current positions violate risk limits"""
        try:
            violations = []

            # Check daily loss limit
            if self.daily_pnl < -self.max_daily_loss:
                violations.append({
                    'type': 'DAILY_LOSS_LIMIT',
                    'message': f'Daily loss limit exceeded: ${self.daily_pnl:.2f}',
                    'severity': 'HIGH'
                })

            # Check maximum positions
            if len(self.positions) >= self.max_positions:
                violations.append({
                    'type': 'MAX_POSITIONS',
                    'message': f'Maximum positions reached: {len(self.positions)}/{self.max_positions}',
                    'severity': 'MEDIUM'
                })

            # Check portfolio drawdown
            current_drawdown = (self.peak_portfolio_value - self.get_portfolio_value()) / self.peak_portfolio_value
            if current_drawdown > 0.15:  # 15% drawdown limit
                violations.append({
                    'type': 'DRAWDOWN_LIMIT',
                    'message': f'Portfolio drawdown: {current_drawdown:.2%}',
                    'severity': 'HIGH'
                })

            # Check available capital
            if self.available_capital < self.portfolio_value * 0.1:  # 10% minimum cash
                violations.append({
                    'type': 'LOW_CAPITAL',
                    'message': f'Low available capital: ${self.available_capital:.2f}',
                    'severity': 'MEDIUM'
                })

            return {
                'violations': violations,
                'can_trade': len([v for v in violations if v['severity'] == 'HIGH']) == 0
            }

        except Exception as e:
            self.logger.error(f"Error checking risk limits: {str(e)}")
            return {'violations': [], 'can_trade': False}

    def generate_trade_recommendation(self, signal_data: Dict[str, Any]) -> Optional[TradeRecommendation]:
        """Generate trade recommendation based on signal and risk assessment"""
        try:
            signal = signal_data['signal']
            symbol = "NIFTY 50"
            current_price = signal_data['current_price']
            confidence = signal_data['confidence']
            strength = signal_data['strength']

            # Check risk limits
            risk_check = self.check_risk_limits()
            if not risk_check['can_trade']:
                self.logger.warning("Trade blocked due to risk limit violations")
                return None

            # Skip if confidence is too low
            if confidence < self.trading_config.get('min_confidence_threshold', 0.7):
                return None

            # Handle existing positions
            if symbol in self.positions:
                return self.handle_existing_position(symbol, signal_data)

            # Generate new position recommendation
            if signal in [Signal.BUY, Signal.STRONG_BUY]:
                action = "BUY"
                position_type = PositionType.LONG
            elif signal in [Signal.SELL, Signal.STRONG_SELL]:
                action = "SELL"
                position_type = PositionType.SHORT
            else:
                return None  # No action for HOLD

            # Calculate trade parameters
            stop_loss = self.calculate_stop_loss(current_price, signal)
            take_profit = self.calculate_take_profit(current_price, signal)
            quantity = self.calculate_position_size(strength, current_price, stop_loss)

            if quantity == 0:
                return None

            # Calculate risk-reward ratio
            risk = abs(current_price - stop_loss)
            reward = abs(take_profit - current_price)
            risk_reward_ratio = reward / risk if risk > 0 else 0

            # Position size as percentage of portfolio
            position_value = quantity * current_price
            position_size_pct = position_value / self.get_portfolio_value()

            # Generate recommendation
            recommendation = TradeRecommendation(
                action=action,
                symbol=symbol,
                quantity=quantity,
                price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=confidence,
                reason=f"{signal.value} signal with {strength:.2f} strength",
                risk_reward_ratio=risk_reward_ratio,
                position_size_pct=position_size_pct
            )

            self.logger.info(
                f"💡 TRADE RECOMMENDATION: {action} {quantity} shares of {symbol} "
                f"at ₹{current_price:.2f} | SL: ₹{stop_loss:.2f} | TP: ₹{take_profit:.2f} | "
                f"R:R = {risk_reward_ratio:.2f}"
            )

            return recommendation

        except Exception as e:
            self.logger.error(f"Error generating trade recommendation: {str(e)}")
            return None

    def handle_existing_position(self, symbol: str, signal_data: Dict[str, Any]) -> Optional[TradeRecommendation]:
        """Handle existing position based on new signal"""
        try:
            position = self.positions[symbol]
            current_price = signal_data['current_price']
            signal = signal_data['signal']

            # Update position's current price and PnL
            self.update_position_pnl(symbol, current_price)

            # Check if we should close the position
            should_close = False
            close_reason = ""

            # Stop loss check
            if ((position.position_type == PositionType.LONG and current_price <= position.stop_loss) or
                    (position.position_type == PositionType.SHORT and current_price >= position.stop_loss)):
                should_close = True
                close_reason = "Stop loss triggered"

            # Take profit check
            elif ((position.position_type == PositionType.LONG and current_price >= position.take_profit) or
                  (position.position_type == PositionType.SHORT and current_price <= position.take_profit)):
                should_close = True
                close_reason = "Take profit triggered"

            # Signal reversal check
            elif ((position.position_type == PositionType.LONG and signal in [Signal.SELL, Signal.STRONG_SELL]) or
                  (position.position_type == PositionType.SHORT and signal in [Signal.BUY, Signal.STRONG_BUY])):
                should_close = True
                close_reason = "Signal reversal"

            if should_close:
                action = "SELL" if position.position_type == PositionType.LONG else "BUY"
                recommendation = TradeRecommendation(
                    action=action,
                    symbol=symbol,
                    quantity=position.quantity,
                    price=current_price,
                    stop_loss=0.0,
                    take_profit=0.0,
                    confidence=signal_data['confidence'],
                    reason=close_reason,
                    risk_reward_ratio=0.0,
                    position_size_pct=0.0
                )

                self.logger.info(f"🔄 POSITION CLOSE: {close_reason} for {symbol}")
                return recommendation

            return None

        except Exception as e:
            self.logger.error(f"Error handling existing position: {str(e)}")
            return None

    def execute_trade(self, recommendation: TradeRecommendation) -> Dict[str, Any]:
        """Execute trade recommendation (simulation)"""
        try:
            symbol = recommendation.symbol
            action = recommendation.action
            quantity = recommendation.quantity
            price = recommendation.price

            # Simulate trade execution
            trade_result = {
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'price': price,
                'timestamp': datetime.now(),
                'status': 'EXECUTED',
                'trade_id': len(self.trade_history) + 1
            }

            if action == "BUY":
                if symbol in self.positions:
                    # Close short position or increase long position
                    existing_position = self.positions[symbol]
                    if existing_position.position_type == PositionType.SHORT:
                        # Close short position
                        pnl = (existing_position.entry_price - price) * existing_position.quantity
                        self.realize_pnl(symbol, pnl)
                        del self.positions[symbol]
                    else:
                        # Add to long position (average down/up)
                        self.add_to_position(symbol, quantity, price)
                else:
                    # Open new long position
                    self.open_position(symbol, PositionType.LONG, quantity, price,
                                       recommendation.stop_loss, recommendation.take_profit)

                self.available_capital -= quantity * price

            elif action == "SELL":
                if symbol in self.positions:
                    # Close long position or increase short position
                    existing_position = self.positions[symbol]
                    if existing_position.position_type == PositionType.LONG:
                        # Close long position
                        pnl = (price - existing_position.entry_price) * existing_position.quantity
                        self.realize_pnl(symbol, pnl)
                        del self.positions[symbol]
                    else:
                        # Add to short position
                        self.add_to_position(symbol, quantity, price)
                else:
                    # Open new short position
                    self.open_position(symbol, PositionType.SHORT, quantity, price,
                                       recommendation.stop_loss, recommendation.take_profit)

                self.available_capital += quantity * price

            # Update trade statistics
            self.total_trades += 1
            self.trade_history.append(trade_result)

            self.logger.info(f"✅ TRADE EXECUTED: {action} {quantity} {symbol} at ₹{price:.2f}")

            return trade_result

        except Exception as e:
            self.logger.error(f"Error executing trade: {str(e)}")
            return {'status': 'FAILED', 'error': str(e)}

    def open_position(self, symbol: str, position_type: PositionType, quantity: int,
                      entry_price: float, stop_loss: float, take_profit: float):
        """Open new position"""
        position = Position(
            symbol=symbol,
            position_type=position_type,
            entry_price=entry_price,
            quantity=quantity,
            entry_time=datetime.now(),
            current_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

        self.positions[symbol] = position
        self.logger.info(f"📈 NEW POSITION: {position_type.value} {quantity} {symbol} at ₹{entry_price:.2f}")

    def add_to_position(self, symbol: str, additional_quantity: int, price: float):
        """Add to existing position"""
        if symbol in self.positions:
            position = self.positions[symbol]

            # Calculate new average entry price
            total_quantity = position.quantity + additional_quantity
            total_value = (position.entry_price * position.quantity) + (price * additional_quantity)
            new_avg_price = total_value / total_quantity

            # Update position
            position.quantity = total_quantity
            position.entry_price = new_avg_price

            self.logger.info(f"➕ ADDED TO POSITION: {additional_quantity} {symbol} at ₹{price:.2f}")

    def update_position_pnl(self, symbol: str, current_price: float):
        """Update position's unrealized PnL"""
        if symbol in self.positions:
            position = self.positions[symbol]
            position.current_price = current_price

            if position.position_type == PositionType.LONG:
                position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
            else:
                position.unrealized_pnl = (position.entry_price - current_price) * position.quantity

    def realize_pnl(self, symbol: str, pnl: float):
        """Realize PnL from closed position"""
        self.daily_pnl += pnl

        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        self.logger.info(f"💰 REALIZED PnL: ₹{pnl:.2f} from {symbol}")

    def get_portfolio_value(self) -> float:
        """Calculate current portfolio value"""
        try:
            portfolio_value = self.available_capital

            for position in self.positions.values():
                if position.position_type == PositionType.LONG:
                    portfolio_value += position.quantity * position.current_price
                else:
                    # For short positions, we have cash from the sale
                    portfolio_value += position.quantity * (2 * position.entry_price - position.current_price)

            return portfolio_value

        except Exception as e:
            self.logger.error(f"Error calculating portfolio value: {str(e)}")
            return self.available_capital

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        try:
            current_value = self.get_portfolio_value()
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())

            # Update peak value and drawdown
            if current_value > self.peak_portfolio_value:
                self.peak_portfolio_value = current_value

            current_drawdown = (self.peak_portfolio_value - current_value) / self.peak_portfolio_value
            self.max_drawdown = max(self.max_drawdown, current_drawdown)

            # Calculate win rate
            win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0

            summary = {
                'portfolio_value': current_value,
                'available_capital': self.available_capital,
                'total_unrealized_pnl': total_unrealized_pnl,
                'daily_pnl': self.daily_pnl,
                'total_return': (current_value - self.portfolio_value) / self.portfolio_value * 100,
                'max_drawdown': self.max_drawdown * 100,
                'current_drawdown': current_drawdown * 100,
                'active_positions': len(self.positions),
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'win_rate': win_rate,
                'positions': [
                    {
                        'symbol': pos.symbol,
                        'type': pos.position_type.value,
                        'quantity': pos.quantity,
                        'entry_price': pos.entry_price,
                        'current_price': pos.current_price,
                        'unrealized_pnl': pos.unrealized_pnl,
                        'entry_time': pos.entry_time
                    } for pos in self.positions.values()
                ]
            }

            return summary

        except Exception as e:
            self.logger.error(f"Error generating portfolio summary: {str(e)}")
            return {}
