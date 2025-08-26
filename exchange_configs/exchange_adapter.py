#!/usr/bin/env python3
"""
交易所适配器 - 统一管理不同交易所的接入
"""

import ccxt
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class ExchangeInfo:
    """交易所信息数据类"""
    name: str
    ccxt_id: str
    description: str
    features: List[str]
    sandbox_available: bool
    api_docs: str
    default_config: Dict[str, Any]
    trading_fees: Dict[str, float]


class ExchangeAdapter:
    """交易所适配器"""
    
    def __init__(self, config_file: str = None):
        """
        初始化交易所适配器
        
        Args:
            config_file: 交易所配置文件路径
        """
        self.logger = logging.getLogger(__name__)
        
        # 加载交易所配置
        if config_file is None:
            config_file = Path(__file__).parent / "exchanges.json"
        
        self.exchange_configs = self._load_exchange_configs(config_file)
        
        # 验证ccxt支持
        self._validate_ccxt_support()
    
    def _load_exchange_configs(self, config_file: str) -> Dict[str, ExchangeInfo]:
        """加载交易所配置"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            exchanges = {}
            for ex_id, config in data['supported_exchanges'].items():
                exchanges[ex_id] = ExchangeInfo(
                    name=config['name'],
                    ccxt_id=config['ccxt_id'],
                    description=config['description'],
                    features=config['features'],
                    sandbox_available=config['sandbox_available'],
                    api_docs=config['api_docs'],
                    default_config=config['default_config'],
                    trading_fees=config['trading_fees']
                )
            
            self.logger.info(f"已加载 {len(exchanges)} 个交易所配置")
            return exchanges
            
        except Exception as e:
            self.logger.error(f"加载交易所配置失败: {e}")
            return {}
    
    def _validate_ccxt_support(self):
        """验证ccxt库对交易所的支持"""
        supported_count = 0
        unsupported = []
        
        for ex_id, info in self.exchange_configs.items():
            if info.ccxt_id in ccxt.exchanges:
                supported_count += 1
            else:
                unsupported.append(ex_id)
        
        if unsupported:
            self.logger.warning(f"以下交易所不受ccxt支持: {unsupported}")
        
        self.logger.info(f"ccxt支持的交易所: {supported_count}/{len(self.exchange_configs)}")
    
    def list_supported_exchanges(self) -> List[str]:
        """列出支持的交易所"""
        return list(self.exchange_configs.keys())
    
    def get_exchange_info(self, exchange_id: str) -> Optional[ExchangeInfo]:
        """获取交易所信息"""
        return self.exchange_configs.get(exchange_id)
    
    def create_exchange(self, exchange_id: str, api_key: str = "", secret: str = "", 
                       password: str = "", sandbox: bool = True, **kwargs) -> Optional[ccxt.Exchange]:
        """
        创建交易所实例
        
        Args:
            exchange_id: 交易所ID
            api_key: API密钥
            secret: API密钥
            password: API密码 (部分交易所需要)
            sandbox: 是否使用沙盒环境
            **kwargs: 其他配置参数
        
        Returns:
            交易所实例或None
        """
        if exchange_id not in self.exchange_configs:
            self.logger.error(f"不支持的交易所: {exchange_id}")
            return None
        
        exchange_info = self.exchange_configs[exchange_id]
        
        # 检查沙盒支持
        if sandbox and not exchange_info.sandbox_available:
            self.logger.warning(f"{exchange_info.name} 不支持沙盒环境，将使用实盘环境")
            sandbox = False
        
        try:
            # 获取ccxt交易所类
            exchange_class = getattr(ccxt, exchange_info.ccxt_id)
            
            # 构建配置
            config = {
                'apiKey': api_key,
                'secret': secret,
                'sandbox': sandbox,
                **exchange_info.default_config,
                **kwargs
            }
            
            # 部分交易所需要password
            if password:
                config['password'] = password
            
            # 创建交易所实例
            exchange = exchange_class(config)
            
            # 加载市场信息
            exchange.load_markets()
            
            self.logger.info(f"成功创建 {exchange_info.name} 交易所实例 "
                           f"({'沙盒' if sandbox else '实盘'})")
            
            return exchange
            
        except Exception as e:
            self.logger.error(f"创建 {exchange_info.name} 交易所实例失败: {e}")
            return None
    
    def get_exchange_trading_fees(self, exchange_id: str) -> Optional[Dict[str, float]]:
        """获取交易所手续费"""
        if exchange_id in self.exchange_configs:
            return self.exchange_configs[exchange_id].trading_fees
        return None
    
    def search_exchanges_by_feature(self, feature: str) -> List[str]:
        """根据功能搜索交易所"""
        result = []
        for ex_id, info in self.exchange_configs.items():
            if feature in info.features:
                result.append(ex_id)
        return result
    
    def print_exchange_summary(self):
        """打印交易所摘要信息"""
        print("=" * 80)
        print("FreqBot 支持的交易所")
        print("=" * 80)
        
        for ex_id, info in self.exchange_configs.items():
            print(f"\n📈 {info.name} ({ex_id})")
            print(f"   描述: {info.description}")
            print(f"   功能: {', '.join(info.features)}")
            print(f"   沙盒: {'✅' if info.sandbox_available else '❌'}")
            print(f"   手续费: Maker {info.trading_fees['maker']*100:.2f}% / "
                  f"Taker {info.trading_fees['taker']*100:.2f}%")
            print(f"   文档: {info.api_docs}")
        
        print(f"\n总计: {len(self.exchange_configs)} 个交易所")
        print("=" * 80)


def create_exchange_config_template(exchange_id: str, output_file: str = None):
    """创建交易所配置模板"""
    adapter = ExchangeAdapter()
    
    if exchange_id not in adapter.exchange_configs:
        print(f"错误: 不支持的交易所 {exchange_id}")
        return
    
    info = adapter.exchange_configs[exchange_id]
    
    template = {
        "symbol": "BTC/USDT",
        "exchange": {
            "name": info.ccxt_id,
            "apiKey": "your_api_key_here",
            "secret": "your_api_secret_here",
            "sandbox": True,
            **info.default_config
        },
        "strategy_params": {
            "min_order_size": 0.001,
            "spread_ratio": 0.002,
            "price_update_threshold": 0.0005,
            "check_interval": 1.0
        },
        "position_balance": {
            "position_imbalance_threshold": 0.1,
            "max_position_value_ratio": 0.8,
            "rebalance_urgency_multiplier": 2.0,
            "retreat_distance": 0.001
        },
        "risk_management": {
            "max_daily_loss": 100.0,
            "position_limit": 1000.0
        },
        "exchange_info": {
            "name": info.name,
            "description": info.description,
            "features": info.features,
            "trading_fees": info.trading_fees,
            "api_docs": info.api_docs
        }
    }
    
    if output_file is None:
        output_file = f"{exchange_id}_config.json"
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(template, f, indent=4, ensure_ascii=False)
        
        print(f"✅ 已创建 {info.name} 配置模板: {output_file}")
        print(f"请编辑配置文件并填入您的API密钥")
        
        if info.sandbox_available:
            print(f"💡 建议先在沙盒环境测试")
        else:
            print(f"⚠️  {info.name} 不支持沙盒，请谨慎使用实盘")
        
        print(f"📚 API文档: {info.api_docs}")
        
    except Exception as e:
        print(f"创建配置模板失败: {e}")


def main():
    """主函数 - 演示用法"""
    import argparse
    
    parser = argparse.ArgumentParser(description='交易所适配器工具')
    parser.add_argument('--list', action='store_true', help='列出支持的交易所')
    parser.add_argument('--info', type=str, help='显示交易所详细信息')
    parser.add_argument('--create-config', type=str, help='创建交易所配置模板')
    parser.add_argument('--output', type=str, help='配置文件输出路径')
    parser.add_argument('--feature', type=str, help='按功能搜索交易所 (spot/futures/options)')
    
    args = parser.parse_args()
    
    adapter = ExchangeAdapter()
    
    if args.list:
        adapter.print_exchange_summary()
    
    elif args.info:
        info = adapter.get_exchange_info(args.info)
        if info:
            print(f"\n交易所: {info.name}")
            print(f"ID: {args.info}")
            print(f"描述: {info.description}")
            print(f"功能: {', '.join(info.features)}")
            print(f"沙盒: {'支持' if info.sandbox_available else '不支持'}")
            print(f"文档: {info.api_docs}")
        else:
            print(f"未找到交易所: {args.info}")
    
    elif args.create_config:
        create_exchange_config_template(args.create_config, args.output)
    
    elif args.feature:
        exchanges = adapter.search_exchanges_by_feature(args.feature)
        if exchanges:
            print(f"支持 {args.feature} 功能的交易所:")
            for ex in exchanges:
                info = adapter.get_exchange_info(ex)
                print(f"- {info.name} ({ex})")
        else:
            print(f"没有找到支持 {args.feature} 功能的交易所")
    
    else:
        print("使用 --help 查看可用选项")


if __name__ == "__main__":
    main()