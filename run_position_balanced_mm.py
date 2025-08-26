#!/usr/bin/env python3
"""
仓位平衡做市商策略启动脚本

新功能：
1. 自动监控仓位失衡
2. 将失衡仓位挂到反向买卖一
3. 同方向订单智能避让
4. 动态调整订单价格和数量

使用前请确保：
1. 准备充足的双边资金（基础货币+计价货币）
2. 设置合适的失衡阈值
3. 先在测试环境验证策略逻辑
"""

import sys
import os
import json
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "user_data" / "strategies"))

from PositionBalancedMarketMaker import PositionBalancedMarketMaker


def load_config(config_path: str = "position_balanced_config.json"):
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
    print("      FreqBot 仓位平衡做市商策略")
    print("=" * 60)
    print("核心功能:")
    print("• 永远在买一卖一挂最小数量订单")
    print("• 实时监控仓位失衡情况")
    print("• 自动将失衡仓位挂到反向买卖一")
    print("• 同方向订单智能避让平仓订单")
    print("• 动态调整订单价格和数量")
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
    
    if 'position_balance' in config:
        balance = config['position_balance']
        print(f"• 失衡阈值: {balance.get('position_imbalance_threshold', 0.1) * 100:.1f}%")
        print(f"• 平衡倍数: {balance.get('rebalance_urgency_multiplier', 2.0)}x")
        print(f"• 避让距离: {balance.get('retreat_distance', 0.001) * 100:.1f}%")
    
    if 'risk_management' in config:
        risk = config['risk_management']
        print(f"• 最大日亏损: {risk.get('max_daily_loss', 100.0)}")
        print(f"• 持仓限制: {risk.get('position_limit', 1000.0)}")
    
    print("-" * 60)


def print_strategy_explanation():
    """打印策略说明"""
    print("策略工作原理:")
    print()
    print("1. 正常情况:")
    print("   - 在买一价挂买单，卖一价挂卖单")
    print("   - 维持最小数量的双边流动性")
    print()
    print("2. 仓位失衡时:")
    print("   - 监控基础货币与计价货币的价值比例")
    print("   - 当失衡超过阈值时触发平衡机制")
    print()
    print("3. 平衡机制:")
    print("   - 基础货币过多: 在买一价挂大量卖单平仓")
    print("   - 计价货币过多: 在卖一价挂大量买单建仓")
    print("   - 同方向的正常做市单会后退避让")
    print()
    print("4. 动态调整:")
    print("   - 平衡订单数量根据失衡程度动态调整")
    print("   - 价格根据市场情况实时更新")
    print()
    print("-" * 60)


def main():
    """主函数"""
    print_welcome()
    print_strategy_explanation()
    
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
        print("仓位平衡策略会主动管理您的仓位，请确保:")
        print("1. 充分理解策略逻辑")
        print("2. 准备了充足的双边资金")
        print("3. 设置了合理的平衡参数")
        print("4. 在测试环境充分验证")
        
        confirm = input("\n确认继续吗? (输入 'YES' 继续): ")
        if confirm != 'YES':
            print("已取消运行")
            return 0
    
    # 创建策略实例
    try:
        print("正在初始化仓位平衡做市商策略...")
        
        # 构建策略参数
        strategy_kwargs = {}
        if 'strategy_params' in config:
            strategy_kwargs.update(config['strategy_params'])
        if 'position_balance' in config:
            strategy_kwargs.update(config['position_balance'])
        if 'risk_management' in config:
            strategy_kwargs.update(config['risk_management'])
        
        strategy = PositionBalancedMarketMaker(
            symbol=config['symbol'],
            exchange=config['exchange'],
            **strategy_kwargs
        )
        
        print("策略初始化成功!")
        print()
        print("运行提示:")
        print("• 策略会自动监控和平衡仓位")
        print("• 统计信息每10个周期打印一次")
        print("• 关注'仓位失衡'状态和平衡订单")
        print("• 按 Ctrl+C 安全停止策略")
        print("=" * 60)
        
        # 运行策略
        strategy.run()
        
    except KeyboardInterrupt:
        print("\n\n收到停止信号，正在安全退出...")
    except Exception as e:
        print(f"\n策略运行出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("策略已停止")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)