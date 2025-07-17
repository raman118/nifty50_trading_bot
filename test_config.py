# Test in PyCharm console
from src.utils.config_manager import ConfigManager
from src.utils.logger import TradingLogger

config = ConfigManager()
logger = TradingLogger()

print("Configuration loaded successfully!")
print(f"Trading symbol: {config.get_trading_config()['symbol']}")

