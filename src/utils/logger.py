import logging
import os
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler
import json
from typing import Dict, Any, Optional


class TradingLogger:
    """Advanced logging system for the trading bot"""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.setup_logger()
            self._initialized = True

    def setup_logger(self):
        """Setup comprehensive logging configuration"""
        from .config_manager import ConfigManager

        # Get logging configuration
        config_manager = ConfigManager()
        log_config = config_manager.get_logging_config()

        # Create logs directory
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)

        # Main logger
        self.logger = logging.getLogger('TradingBot')
        self.logger.setLevel(getattr(logging, log_config['level']))

        # Clear existing handlers
        self.logger.handlers.clear()

        # File handler with rotation
        file_handler = RotatingFileHandler(
            filename=log_config['file'],
            maxBytes=log_config.get('max_file_size', 10485760),  # 10MB
            backupCount=log_config.get('backup_count', 5),
            encoding='utf-8'
        )

        file_formatter = logging.Formatter(log_config['format'])
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Console handler
        if log_config.get('console_output', True):
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

        # Specialized loggers
        self.setup_specialized_loggers()

        self.logger.info("Trading Bot Logger initialized successfully")

    def setup_specialized_loggers(self):
        """Setup specialized loggers for different components"""
        # Trading signals logger
        self.trade_logger = logging.getLogger('TradingBot.Trades')
        trade_handler = logging.FileHandler('logs/trades.log', encoding='utf-8')
        trade_formatter = logging.Formatter('%(asctime)s - %(message)s')
        trade_handler.setFormatter(trade_formatter)
        self.trade_logger.addHandler(trade_handler)

        # Performance logger
        self.performance_logger = logging.getLogger('TradingBot.Performance')
        perf_handler = logging.FileHandler('logs/performance.log', encoding='utf-8')
        perf_formatter = logging.Formatter('%(asctime)s - %(message)s')
        perf_handler.setFormatter(perf_formatter)
        self.performance_logger.addHandler(perf_handler)

        # Error logger
        self.error_logger = logging.getLogger('TradingBot.Errors')
        error_handler = logging.FileHandler('logs/errors.log', encoding='utf-8')
        error_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        error_handler.setFormatter(error_formatter)
        self.error_logger.addHandler(error_handler)

    def get_logger(self) -> logging.Logger:
        """Get main logger instance"""
        return self.logger

    def log_trade_signal(self, signal: str, price: float, confidence: float,
                         indicators: Dict[str, Any], timestamp: Optional[datetime] = None):
        """Log trading signals with detailed information"""
        if timestamp is None:
            timestamp = datetime.now()

        trade_data = {
            'timestamp': timestamp.isoformat(),
            'signal': signal,
            'price': price,
            'confidence': confidence,
            'indicators': indicators
        }

        self.trade_logger.info(json.dumps(trade_data, indent=2))
        self.logger.info(f"TRADE_SIGNAL: {signal} | Price: ₹{price:.2f} | Confidence: {confidence:.2%}")

    def log_model_performance(self, metrics: Dict[str, Any]):
        """Log model performance metrics"""
        performance_data = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }

        self.performance_logger.info(json.dumps(performance_data, indent=2))
        self.logger.info(f"MODEL_PERFORMANCE: {metrics}")

    def log_error(self, error: Exception, context: str = "", additional_data: Dict[str, Any] = None):
        """Log errors with comprehensive context"""
        error_data = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'additional_data': additional_data or {}
        }

        self.error_logger.error(json.dumps(error_data, indent=2))
        self.logger.error(f"ERROR in {context}: {str(error)}")

    def log_api_call(self, endpoint: str, status: str, response_time: float):
        """Log API calls for monitoring"""
        self.logger.debug(f"API_CALL: {endpoint} | Status: {status} | Time: {response_time:.2f}ms")

    def log_system_status(self, component: str, status: str, details: str = ""):
        """Log system component status"""
        self.logger.info(f"SYSTEM_STATUS: {component} - {status} {details}")

    def log_data_update(self, data_type: str, records_count: int, source: str):
        """Log data updates"""
        self.logger.info(f"DATA_UPDATE: {data_type} | Records: {records_count} | Source: {source}")

    def log_prediction(self, predicted_price: float, actual_price: float, accuracy: float):
        """Log model predictions vs actual"""
        self.logger.info(
            f"PREDICTION: Predicted: ₹{predicted_price:.2f} | Actual: ₹{actual_price:.2f} | Accuracy: {accuracy:.2%}")
