"""
FreqBot - 量化交易机器人框架

基于 FreqTrade 的统一交易策略管理和执行平台
"""

__version__ = "1.0.0"
__author__ = "FreqBot Team"

from .core.engine import TradingEngine
from .config.manager import ConfigManager
from .strategies.registry import StrategyRegistry

__all__ = ["TradingEngine", "ConfigManager", "StrategyRegistry"]