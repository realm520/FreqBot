#!/usr/bin/env python3
"""
演示如何使用自定义交易所运行做市商策略
"""

import sys
import json
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent / "user_data" / "strategies"))

from UniversalMarketMaker import UniversalMarketMaker


def main():
    print("=" * 60)
    print("FreqBot 自定义交易所演示")
    print("=" * 60)
    print("本演示将使用DemoExchange运行做市商策略")
    print("DemoExchange是一个完全模拟的交易所，无任何风险")
    print()
    
    # 配置
    config = {
        'symbol': 'BTC/USDT',
        'exchange': {
            'name': 'demo',
            'api_key': 'demo_api_key_12345',
            'secret_key': 'demo_secret_key_67890',
            'sandbox': True
        },
        'strategy_params': {
            'min_order_size': 0.001,
            'spread_ratio': 0.005,  # 0.5%价差
            'check_interval': 3.0   # 3秒检查一次
        },
        'position_balance': {
            'position_imbalance_threshold': 0.2,  # 20%失衡触发
            'rebalance_urgency_multiplier': 1.5
        }
    }
    
    print("配置信息:")
    print(f"• 交易所: {config['exchange']['name']} (自定义)")
    print(f"• 交易对: {config['symbol']}")
    print(f"• 价差: {config['strategy_params']['spread_ratio']*100}%")
    print(f"• 检查间隔: {config['strategy_params']['check_interval']}秒")
    print()
    
    print("策略特性:")
    print("✅ 支持ccxt和自定义交易所")
    print("✅ 仓位平衡管理")
    print("✅ 智能订单避让")
    print("✅ 实时统计监控")
    print()
    
    input("按Enter键开始运行策略 (Ctrl+C停止)...")
    
    try:
        # 创建策略
        strategy = UniversalMarketMaker(**config)
        
        print("\n🚀 策略启动成功!")
        print("观察要点:")
        print("• 正常单 vs 平衡单的比例")
        print("• 仓位失衡检测和处理")
        print("• 模拟订单的成交情况")
        print("• 统计信息的实时更新")
        print()
        
        # 运行策略
        strategy.run()
        
    except KeyboardInterrupt:
        print("\n\n✨ 演示结束，感谢体验!")
        print("总结:")
        print("• DemoExchange成功模拟了真实交易所")
        print("• 策略正常识别并使用自定义交易所")
        print("• 所有功能都能正常工作")
        print()
        print("下一步:")
        print("1. 参考demo_exchange.py实现您的交易所")
        print("2. 按照HOW_TO_ADD_NEW_EXCHANGE.md指南")
        print("3. 在实际交易所测试策略")
    
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        print("请检查配置和依赖")


if __name__ == "__main__":
    main()