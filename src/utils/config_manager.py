import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path
import logging


class ConfigManager:
    """Singleton configuration manager for the trading bot"""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.config_dir = Path("config")
            self.config = {}
            self.api_config = {}
            self.load_all_configs()
            self._initialized = True

    def load_all_configs(self):
        """Load all configuration files"""
        try:
            # Load main configuration
            self.load_main_config()

            # Load API configuration
            self.load_api_config()

            # Validate configurations
            self.validate_configs()

        except Exception as e:
            raise RuntimeError(f"Failed to load configurations: {str(e)}")

    def load_main_config(self):
        """Load main configuration from config.yaml"""
        config_path = self.config_dir / "config.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Main config file not found: {config_path}")

        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                self.config = yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing main config YAML: {str(e)}")

    def load_api_config(self):
        """Load API configuration from api_config.yaml"""
        api_config_path = self.config_dir / "api_config.yaml"

        if not api_config_path.exists():
            raise FileNotFoundError(f"API config file not found: {api_config_path}")

        try:
            with open(api_config_path, 'r', encoding='utf-8') as file:
                self.api_config = yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing API config YAML: {str(e)}")

    def validate_configs(self):
        """Validate essential configuration parameters"""
        required_sections = ['trading', 'model', 'data', 'logging']

        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")

        # Validate API config
        if 'angel_one' not in self.api_config:
            raise ValueError("Missing Angel One API configuration")

        # Check for placeholder values
        angel_config = self.api_config['angel_one']
        placeholders = ['YOUR_API_KEY_HERE', 'YOUR_CLIENT_CODE_HERE',
                        'YOUR_PASSWORD_HERE', 'YOUR_TOTP_SECRET_HERE']

        for key, value in angel_config.items():
            if value in placeholders:
                print(f"WARNING: Placeholder value found for {key}. Please update with actual credentials.")

    def get_config(self, section: Optional[str] = None, key: Optional[str] = None) -> Any:
        """Get configuration value(s)"""
        if section is None:
            return self.config

        if section not in self.config:
            raise KeyError(f"Configuration section '{section}' not found")

        if key is None:
            return self.config[section]

        if key not in self.config[section]:
            raise KeyError(f"Configuration key '{key}' not found in section '{section}'")

        return self.config[section][key]

    def get_api_config(self, section: Optional[str] = None, key: Optional[str] = None) -> Any:
        """Get API configuration value(s)"""
        if section is None:
            return self.api_config

        if section not in self.api_config:
            raise KeyError(f"API configuration section '{section}' not found")

        if key is None:
            return self.api_config[section]

        if key not in self.api_config[section]:
            raise KeyError(f"API configuration key '{key}' not found in section '{section}'")

        return self.api_config[section][key]

    def update_config(self, section: str, key: str, value: Any):
        """Update configuration value"""
        if section not in self.config:
            self.config[section] = {}

        self.config[section][key] = value

    def save_config(self, backup: bool = True):
        """Save configuration to file with optional backup"""
        config_path = self.config_dir / "config.yaml"

        if backup and config_path.exists():
            backup_path = config_path.with_suffix('.yaml.bak')
            config_path.rename(backup_path)

        try:
            with open(config_path, 'w', encoding='utf-8') as file:
                yaml.dump(self.config, file, default_flow_style=False, indent=2)
        except Exception as e:
            raise RuntimeError(f"Failed to save configuration: {str(e)}")

    def reload_configs(self):
        """Reload all configurations from files"""
        self.load_all_configs()

    def get_trading_config(self) -> Dict[str, Any]:
        """Get trading-specific configuration"""
        return self.get_config('trading')

    def get_model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration"""
        return self.get_config('model')

    def get_data_config(self) -> Dict[str, Any]:
        """Get data-specific configuration"""
        return self.get_config('data')

    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging-specific configuration"""
        return self.get_config('logging')

    def get_angel_one_config(self) -> Dict[str, Any]:
        """Get Angel One API configuration"""
        return self.get_api_config('angel_one')
