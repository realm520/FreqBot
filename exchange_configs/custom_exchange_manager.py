#!/usr/bin/env python3
"""
自定义交易所管理器
统一管理ccxt支持的交易所和自定义交易所
"""

import ccxt
import importlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from custom_exchange_base import CustomExchangeBase


class ExchangeManager:
    """
    交易所管理器
    统一管理ccxt交易所和自定义交易所
    """
    
    def __init__(self):
        """初始化交易所管理器"""
        self.logger = logging.getLogger(__name__)
        
        # 已注册的自定义交易所
        self.custom_exchanges: Dict[str, type] = {}
        
        # 已创建的交易所实例
        self.exchange_instances: Dict[str, Union[ccxt.Exchange, CustomExchangeBase]] = {}
        
        # 自动发现并注册自定义交易所
        self._discover_custom_exchanges()
    
    def _discover_custom_exchanges(self):
        """自动发现自定义交易所"""
        try:
            # 扫描当前目录下的自定义交易所文件
            exchange_dir = Path(__file__).parent
            
            for file_path in exchange_dir.glob("*_exchange.py"):
                if file_path.name.startswith("custom_exchange_base"):
                    continue
                
                try:
                    # 动态导入模块
                    module_name = file_path.stem
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # 查找继承自CustomExchangeBase的类
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (isinstance(attr, type) and 
                            issubclass(attr, CustomExchangeBase) and 
                            attr != CustomExchangeBase):
                            
                            exchange_id = attr_name.lower().replace('exchange', '')
                            self.custom_exchanges[exchange_id] = attr
                            self.logger.info(f"发现自定义交易所: {exchange_id}")
                
                except Exception as e:
                    self.logger.warning(f"加载自定义交易所失败 {file_path}: {e}")
        
        except Exception as e:
            self.logger.error(f"自动发现自定义交易所失败: {e}")
    
    def register_custom_exchange(self, exchange_id: str, exchange_class: type):
        """
        手动注册自定义交易所
        
        Args:
            exchange_id: 交易所ID
            exchange_class: 交易所类
        """
        if not issubclass(exchange_class, CustomExchangeBase):
            raise ValueError("交易所类必须继承自CustomExchangeBase")
        
        self.custom_exchanges[exchange_id] = exchange_class
        self.logger.info(f"注册自定义交易所: {exchange_id}")
    
    def list_supported_exchanges(self) -> Dict[str, List[str]]:
        """
        列出支持的交易所
        
        Returns:
            {'ccxt': [...], 'custom': [...]}
        """
        return {
            'ccxt': list(ccxt.exchanges),
            'custom': list(self.custom_exchanges.keys())
        }
    
    def is_custom_exchange(self, exchange_id: str) -> bool:
        """判断是否为自定义交易所"""
        return exchange_id in self.custom_exchanges
    
    def is_ccxt_exchange(self, exchange_id: str) -> bool:
        """判断是否为ccxt交易所"""
        return exchange_id in ccxt.exchanges
    
    def create_exchange(self, exchange_id: str, config: Dict[str, Any]) -> Union[ccxt.Exchange, CustomExchangeBase]:
        """
        创建交易所实例
        
        Args:
            exchange_id: 交易所ID
            config: 配置参数
            
        Returns:
            交易所实例
        """
        instance_key = f"{exchange_id}_{hash(str(sorted(config.items())))}"
        
        # 检查是否已存在实例
        if instance_key in self.exchange_instances:
            return self.exchange_instances[instance_key]
        
        try:
            # 创建自定义交易所
            if self.is_custom_exchange(exchange_id):
                exchange_class = self.custom_exchanges[exchange_id]
                instance = exchange_class(config)
                self.logger.info(f"创建自定义交易所实例: {exchange_id}")
            
            # 创建ccxt交易所
            elif self.is_ccxt_exchange(exchange_id):
                exchange_class = getattr(ccxt, exchange_id)
                instance = exchange_class(config)
                instance.load_markets()
                self.logger.info(f"创建ccxt交易所实例: {exchange_id}")
            
            else:
                raise ValueError(f"不支持的交易所: {exchange_id}")
            
            # 缓存实例
            self.exchange_instances[instance_key] = instance
            return instance
            
        except Exception as e:
            self.logger.error(f"创建交易所实例失败 {exchange_id}: {e}")
            raise
    
    def get_exchange_type(self, exchange_id: str) -> str:
        """
        获取交易所类型
        
        Args:
            exchange_id: 交易所ID
            
        Returns:
            'ccxt' 或 'custom' 或 'unknown'
        """
        if self.is_custom_exchange(exchange_id):
            return 'custom'
        elif self.is_ccxt_exchange(exchange_id):
            return 'ccxt'
        else:
            return 'unknown'
    
    def create_unified_config(self, exchange_id: str, **kwargs) -> Dict[str, Any]:
        """
        创建统一的配置格式
        
        Args:
            exchange_id: 交易所ID
            **kwargs: 配置参数
            
        Returns:
            统一的配置字典
        """
        # 基础配置
        config = {
            'timeout': kwargs.get('timeout', 30000),
            'enableRateLimit': kwargs.get('enableRateLimit', True),
            'sandbox': kwargs.get('sandbox', True)
        }
        
        # API认证
        if 'api_key' in kwargs:
            config['apiKey'] = kwargs['api_key']
            config['api_key'] = kwargs['api_key']
        
        if 'secret_key' in kwargs:
            config['secret'] = kwargs['secret_key']
            config['secret_key'] = kwargs['secret_key']
        
        if 'passphrase' in kwargs:
            config['password'] = kwargs['passphrase']
            config['passphrase'] = kwargs['passphrase']
        
        # 自定义交易所特定配置
        if self.is_custom_exchange(exchange_id):
            config.update({
                'base_url': kwargs.get('base_url', ''),
            })
        
        # 其他配置
        config.update({k: v for k, v in kwargs.items() 
                      if k not in config})
        
        return config
    
    def validate_exchange_config(self, exchange_id: str, config: Dict[str, Any]) -> bool:
        """
        验证交易所配置
        
        Args:
            exchange_id: 交易所ID
            config: 配置参数
            
        Returns:
            验证结果
        """
        try:
            # 基本验证
            if not exchange_id:
                self.logger.error("交易所ID不能为空")
                return False
            
            if not self.is_custom_exchange(exchange_id) and not self.is_ccxt_exchange(exchange_id):
                self.logger.error(f"不支持的交易所: {exchange_id}")
                return False
            
            # API密钥验证
            if self.is_custom_exchange(exchange_id):
                required_keys = ['api_key', 'secret_key']
                if not config.get('base_url'):
                    self.logger.warning("自定义交易所缺少base_url配置")
            else:
                required_keys = ['apiKey', 'secret']
            
            for key in required_keys:
                if not config.get(key):
                    self.logger.warning(f"缺少必要配置: {key}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"验证交易所配置失败: {e}")
            return False
    
    def print_exchange_info(self):
        """打印交易所信息"""
        supported = self.list_supported_exchanges()
        
        print("=" * 60)
        print("FreqBot 交易所管理器")
        print("=" * 60)
        
        print(f"\n🏦 CCXT支持的交易所 ({len(supported['ccxt'])}个):")
        for i, exchange in enumerate(sorted(supported['ccxt'])[:10], 1):
            print(f"  {i:2d}. {exchange}")
        if len(supported['ccxt']) > 10:
            print(f"     ... 还有{len(supported['ccxt']) - 10}个")
        
        print(f"\n🔧 自定义交易所 ({len(supported['custom'])}个):")
        if supported['custom']:
            for i, exchange in enumerate(supported['custom'], 1):
                print(f"  {i:2d}. {exchange} (自定义实现)")
        else:
            print("  暂无自定义交易所")
        
        print("\n📝 如何添加新的交易所:")
        print("  1. 继承CustomExchangeBase类")
        print("  2. 实现所有抽象方法")
        print("  3. 保存为 {exchange_name}_exchange.py")
        print("  4. 重启程序自动发现")
        
        print("=" * 60)


# 全局交易所管理器实例
exchange_manager = ExchangeManager()


def create_exchange_instance(exchange_config: Dict[str, Any]) -> Union[ccxt.Exchange, CustomExchangeBase]:
    """
    便捷函数：创建交易所实例
    
    Args:
        exchange_config: 交易所配置，必须包含'name'字段
        
    Returns:
        交易所实例
    """
    if 'name' not in exchange_config:
        raise ValueError("配置中缺少'name'字段")
    
    exchange_id = exchange_config['name']
    config = {k: v for k, v in exchange_config.items() if k != 'name'}
    
    # 统一配置格式
    unified_config = exchange_manager.create_unified_config(exchange_id, **config)
    
    # 验证配置
    if not exchange_manager.validate_exchange_config(exchange_id, unified_config):
        raise ValueError(f"无效的交易所配置: {exchange_id}")
    
    # 创建实例
    return exchange_manager.create_exchange(exchange_id, unified_config)


if __name__ == "__main__":
    # 演示用法
    manager = ExchangeManager()
    
    # 显示支持的交易所
    manager.print_exchange_info()
    
    # 创建ccxt交易所示例
    try:
        ccxt_config = {
            'apiKey': 'test_key',
            'secret': 'test_secret',
            'sandbox': True
        }
        binance = manager.create_exchange('binance', ccxt_config)
        print(f"\n✅ 成功创建Binance实例: {type(binance).__name__}")
    except Exception as e:
        print(f"\n❌ 创建Binance实例失败: {e}")
    
    # 创建自定义交易所示例
    if manager.custom_exchanges:
        try:
            custom_id = list(manager.custom_exchanges.keys())[0]
            custom_config = {
                'api_key': 'test_key',
                'secret_key': 'test_secret',
                'base_url': 'https://api.test.com',
                'sandbox': True
            }
            custom_exchange = manager.create_exchange(custom_id, custom_config)
            print(f"\n✅ 成功创建自定义交易所实例: {custom_id}")
        except Exception as e:
            print(f"\n❌ 创建自定义交易所实例失败: {e}")