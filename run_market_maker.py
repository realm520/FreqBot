#!/usr/bin/env python3
"""
做市商策略启动脚本
运行前请确保:
1. 已配置正确的API密钥
2. 已设置合适的风险参数
3. 建议先在测试环境运行
"""

import sys
import os
import json
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "user_data" / "strategies"))

from AdvancedMarketMaker import AdvancedMarketMaker


def load_config(config_path: str = "market_maker_config.json"):
    """加载配置文件"""
    config_file = project_root / config_path
    
    if not config_file.exists():
        print(f"配置文件不存在: {config_file}")
        print("请先创建配置文件并填入API密钥")
        return None
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        return None


def validate_config(config: dict) -> bool:
    """验证配置参数"""
    required_keys = ['symbol', 'exchange']
    
    for key in required_keys:
        if key not in config:
            print(f"配置文件缺少必要参数: {key}")
            return False
    
    # 检查API密钥
    exchange_config = config['exchange']
    if not exchange_config.get('apiKey') or exchange_config['apiKey'] == 'your_binance_api_key_here':
        print("警告: 未配置有效的API密钥，将使用沙盒模式")
        exchange_config['sandbox'] = True
    
    return True


def print_welcome():
    """打印欢迎信息"""
    print("=" * 60)
    print("        FreqBot 高级做市商策略")
    print("=" * 60)
    print("功能特点:")
    print("• 永远在买一卖一挂最小数量订单")
    print("• 智能盘口监控和订单管理")
    print("• 动态价格调整和风险控制")
    print("• 实时统计和异常处理")
    print("=" * 60)
    print()


def print_config_info(config: dict):
    """打印配置信息"""
    print("当前配置:")
    print(f"• 交易对: {config['symbol']}")
    print(f"• 交易所: {config['exchange']['name']}")
    print(f"• 沙盒模式: {'是' if config['exchange'].get('sandbox', True) else '否'}")
    
    if 'strategy_params' in config:
        params = config['strategy_params']
        print(f"• 最小订单: {params.get('min_order_size', 0.001)}")
        print(f"• 价差比例: {params.get('spread_ratio', 0.002) * 100:.2f}%")
        print(f"• 检查间隔: {params.get('check_interval', 1.0)}秒")
    
    if 'risk_management' in config:
        risk = config['risk_management']
        print(f"• 最大日亏损: {risk.get('max_daily_loss', 100.0)}")
        print(f"• 持仓限制: {risk.get('position_limit', 1000.0)}")
    
    print("-" * 60)


def main():
    """主函数"""
    print_welcome()
    
    # 加载配置
    config = load_config()
    if not config:
        return 1
    
    # 验证配置
    if not validate_config(config):
        return 1
    
    # 显示配置信息
    print_config_info(config)
    
    # 安全确认
    if not config['exchange'].get('sandbox', True):
        print("⚠️  警告: 您正在使用实盘模式!")
        print("请确保:")
        print("1. 已充分测试策略")
        print("2. 设置了合理的风险参数") 
        print("3. 监控资金安全")
        
        confirm = input("\n确认继续吗? (输入 'YES' 继续): ")
        if confirm != 'YES':
            print("已取消运行")
            return 0
    
    # 创建策略实例
    try:
        print("正在初始化做市商策略...")
        
        # 构建策略参数
        strategy_kwargs = {}
        if 'strategy_params' in config:
            strategy_kwargs.update(config['strategy_params'])
        if 'risk_management' in config:
            strategy_kwargs.update(config['risk_management'])
        
        strategy = AdvancedMarketMaker(
            symbol=config['symbol'],
            exchange=config['exchange'],
            **strategy_kwargs
        )
        
        print("策略初始化成功!")
        print("按 Ctrl+C 停止策略")
        print("=" * 60)
        
        # 运行策略
        strategy.run()
        
    except KeyboardInterrupt:
        print("\n\n收到停止信号，正在安全退出...")
    except Exception as e:
        print(f"\n策略运行出错: {e}")
        return 1
    
    print("策略已停止")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)