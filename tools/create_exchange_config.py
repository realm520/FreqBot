#!/usr/bin/env python3
"""
交易所配置创建工具
快速为不同交易所创建配置文件
"""

import sys
import json
from pathlib import Path

# 交易所配置模板
EXCHANGE_TEMPLATES = {
    'binance': {
        "symbol": "BTC/USDT",
        "exchange": {
            "name": "binance",
            "apiKey": "your_binance_api_key_here",
            "secret": "your_binance_secret_here",
            "sandbox": True,
            "enableRateLimit": True,
            "timeout": 30000,
            "options": {
                "defaultType": "spot"
            }
        }
    },
    
    'okx': {
        "symbol": "BTC/USDT",
        "exchange": {
            "name": "okx", 
            "apiKey": "your_okx_api_key_here",
            "secret": "your_okx_secret_here",
            "password": "your_okx_passphrase_here",
            "sandbox": True,
            "enableRateLimit": True,
            "timeout": 30000,
            "rateLimit": 100
        }
    },
    
    'bybit': {
        "symbol": "BTC/USDT",
        "exchange": {
            "name": "bybit",
            "apiKey": "your_bybit_api_key_here", 
            "secret": "your_bybit_secret_here",
            "sandbox": True,
            "enableRateLimit": True,
            "timeout": 30000,
            "options": {
                "recv_window": 5000
            }
        }
    },
    
    'coinbase': {
        "symbol": "BTC/USD",
        "exchange": {
            "name": "coinbase",
            "apiKey": "your_coinbase_api_key_here",
            "secret": "your_coinbase_secret_here",
            "password": "your_coinbase_passphrase_here",
            "sandbox": True,
            "enableRateLimit": True,
            "timeout": 30000
        }
    },
    
    'kraken': {
        "symbol": "BTC/USD",
        "exchange": {
            "name": "kraken",
            "apiKey": "your_kraken_api_key_here",
            "secret": "your_kraken_secret_here",
            "sandbox": False,  # Kraken 无沙盒
            "enableRateLimit": True,
            "timeout": 60000   # Kraken 较慢
        }
    },
    
    'kucoin': {
        "symbol": "BTC/USDT",
        "exchange": {
            "name": "kucoin",
            "apiKey": "your_kucoin_api_key_here",
            "secret": "your_kucoin_secret_here", 
            "password": "your_kucoin_passphrase_here",
            "sandbox": True,
            "enableRateLimit": True,
            "timeout": 30000
        }
    },
    
    'huobi': {
        "symbol": "BTC/USDT",
        "exchange": {
            "name": "huobi",
            "apiKey": "your_huobi_api_key_here",
            "secret": "your_huobi_secret_here",
            "sandbox": False,  # Huobi 无沙盒
            "enableRateLimit": True,
            "timeout": 30000
        }
    },
    
    'bitget': {
        "symbol": "BTC/USDT", 
        "exchange": {
            "name": "bitget",
            "apiKey": "your_bitget_api_key_here",
            "secret": "your_bitget_secret_here",
            "password": "your_bitget_passphrase_here",
            "sandbox": True,
            "enableRateLimit": True,
            "timeout": 30000
        }
    },
    
    'gate': {
        "symbol": "BTC/USDT",
        "exchange": {
            "name": "gate",
            "apiKey": "your_gate_api_key_here",
            "secret": "your_gate_secret_here",
            "sandbox": True,
            "enableRateLimit": True,
            "timeout": 30000
        }
    },
    
    'mexc': {
        "symbol": "BTC/USDT",
        "exchange": {
            "name": "mexc",
            "apiKey": "your_mexc_api_key_here",
            "secret": "your_mexc_secret_here",
            "sandbox": False,  # MEXC 无沙盒
            "enableRateLimit": True,
            "timeout": 30000
        }
    }
}

# 通用配置部分
COMMON_CONFIG = {
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
    "logging": {
        "level": "INFO",
        "file": "market_maker.log"
    }
}

def create_config(exchange: str, output_file: str = None):
    """创建交易所配置文件"""
    
    if exchange not in EXCHANGE_TEMPLATES:
        print(f"❌ 不支持的交易所: {exchange}")
        print(f"支持的交易所: {', '.join(EXCHANGE_TEMPLATES.keys())}")
        return False
    
    # 构建完整配置
    config = {**EXCHANGE_TEMPLATES[exchange], **COMMON_CONFIG}
    
    # 确定输出文件名
    if output_file is None:
        output_file = f"{exchange}_config.json"
    
    try:
        # 写入配置文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        
        print(f"✅ 成功创建 {exchange.upper()} 配置文件: {output_file}")
        
        # 提供使用提示
        sandbox_status = "支持沙盒" if config['exchange'].get('sandbox', False) else "⚠️  无沙盒，直接实盘"
        print(f"📊 交易所: {exchange.upper()} ({sandbox_status})")
        print(f"💰 默认交易对: {config['symbol']}")
        print(f"⚙️  请编辑配置文件并填入您的API密钥")
        
        # 运行命令
        print(f"\n🚀 运行命令:")
        print(f"python run_position_balanced_mm.py --config {output_file}")
        
        # 特殊提醒
        if not config['exchange'].get('sandbox', False):
            print(f"\n⚠️  警告: {exchange.upper()} 不支持沙盒环境")
            print(f"建议先用小资金测试!")
            
        if 'password' in config['exchange']:
            print(f"\n🔑 注意: {exchange.upper()} 需要 passphrase/password 参数")
        
        return True
        
    except Exception as e:
        print(f"❌ 创建配置文件失败: {e}")
        return False


def list_supported_exchanges():
    """列出支持的交易所"""
    print("🏢 FreqBot 支持的交易所:")
    print("=" * 50)
    
    for exchange in sorted(EXCHANGE_TEMPLATES.keys()):
        config = EXCHANGE_TEMPLATES[exchange]
        sandbox = "沙盒✅" if config['exchange'].get('sandbox', False) else "实盘⚠️"
        symbol = config['symbol']
        needs_password = "🔑" if 'password' in config['exchange'] else "  "
        
        print(f"{needs_password} {exchange:<10} {symbol:<10} {sandbox}")
    
    print("\n🔑 = 需要 passphrase/password")
    print("=" * 50)


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("📖 使用方法:")
        print("  python create_exchange_config.py <交易所名称> [输出文件]")
        print("  python create_exchange_config.py --list")
        print()
        print("示例:")
        print("  python create_exchange_config.py binance")
        print("  python create_exchange_config.py okx my_okx_config.json")
        print("  python create_exchange_config.py --list")
        return
    
    if sys.argv[1] == '--list':
        list_supported_exchanges()
        return
    
    exchange = sys.argv[1].lower()
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = create_config(exchange, output_file)
    
    if not success:
        print("\n💡 提示:")
        print("使用 --list 查看支持的交易所")
        sys.exit(1)


if __name__ == "__main__":
    main()