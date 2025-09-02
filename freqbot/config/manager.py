"""配置管理器 - 统一管理所有配置文件"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class ExchangeConfig:
    """交易所配置"""
    name: str
    api_key: str = ""
    api_secret: str = ""
    sandbox: bool = True
    ccxt_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.ccxt_config is None:
            self.ccxt_config = {
                "enableRateLimit": True,
                "rateLimit": 200
            }

@dataclass
class TradingConfig:
    """交易配置"""
    dry_run: bool = True
    dry_run_wallet: int = 10000
    max_open_trades: int = 10
    stake_currency: str = "USDT"
    stake_amount: str = "unlimited"
    tradable_balance_ratio: float = 0.8
    timeframe: str = "1m"
    minimum_trade_amount: float = 10.0

@dataclass
class EnvironmentConfig:
    """环境配置"""
    name: str
    trading: TradingConfig
    exchange: ExchangeConfig
    api_server: Dict[str, Any]
    database_url: str
    log_level: str = "INFO"

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.config_dir = self.project_root / "configs"
        self.templates_dir = self.config_dir / "templates"
        self.environments_dir = self.config_dir / "environments"
        self.strategies_dir = self.config_dir / "strategies"
        
        # 确保目录存在
        for dir_path in [self.config_dir, self.templates_dir, 
                        self.environments_dir, self.strategies_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def get_environment_config(self, env_name: str) -> Dict[str, Any]:
        """获取环境配置"""
        config_file = self.environments_dir / f"{env_name}.json"
        
        if not config_file.exists():
            raise FileNotFoundError(f"环境配置文件不存在: {config_file}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_strategy_config(self, strategy_name: str) -> Dict[str, Any]:
        """获取策略配置"""
        config_file = self.strategies_dir / f"{strategy_name}.json"
        
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        return {}
    
    def create_freqtrade_config(self, strategy_name: str, env_name: str = "demo") -> Dict[str, Any]:
        """生成 FreqTrade 配置"""
        env_config = self.get_environment_config(env_name)
        strategy_config = self.get_strategy_config(strategy_name)
        
        # 合并配置
        config = {**env_config}
        
        # 添加策略相关配置
        config.update({
            "strategy": strategy_name,
            "strategy_path": "strategies/",
            **strategy_config
        })
        
        return config
    
    def save_environment_config(self, env_name: str, config: Dict[str, Any]):
        """保存环境配置"""
        config_file = self.environments_dir / f"{env_name}.json"
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        logger.info(f"环境配置已保存: {config_file}")
    
    def save_strategy_config(self, strategy_name: str, config: Dict[str, Any]):
        """保存策略配置"""
        config_file = self.strategies_dir / f"{strategy_name}.json"
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        logger.info(f"策略配置已保存: {config_file}")
    
    def list_environments(self) -> list[str]:
        """列出所有环境"""
        return [f.stem for f in self.environments_dir.glob("*.json")]
    
    def list_strategies(self) -> list[str]:
        """列出所有策略配置"""
        return [f.stem for f in self.strategies_dir.glob("*.json")]
    
    def migrate_legacy_config(self, legacy_config_path: str, env_name: str):
        """迁移旧的配置文件"""
        legacy_path = Path(legacy_config_path)
        
        if not legacy_path.exists():
            raise FileNotFoundError(f"旧配置文件不存在: {legacy_path}")
        
        with open(legacy_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 提取策略名称
        strategy_name = config.get("strategy", "unknown")
        
        # 分离环境配置和策略配置
        env_config = {k: v for k, v in config.items() 
                     if k not in ["strategy", "strategy_path"]}
        
        strategy_config = {}
        if "strategy_path" in config:
            strategy_config["strategy_path"] = config["strategy_path"]
        
        # 保存配置
        self.save_environment_config(env_name, env_config)
        if strategy_config:
            self.save_strategy_config(strategy_name, strategy_config)
        
        logger.info(f"已迁移配置文件 {legacy_path} -> {env_name} 环境")
        
        return strategy_name, env_name