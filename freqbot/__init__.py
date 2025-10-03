"""
FreqBot - 量化交易机器人框架

基于 FreqTrade 的 Docker 化交易策略管理平台
仅提供配置管理和策略注册功能，所有交易执行通过 Docker 完成
"""

__version__ = "2.0.0"
__author__ = "FreqBot Team"

from .config.manager import ConfigManager
from .strategies.registry import StrategyRegistry

__all__ = ["ConfigManager", "StrategyRegistry"]