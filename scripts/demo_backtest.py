#!/usr/bin/env python3
"""
演示回测脚本
运行一个简短的回测以验证系统功能
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

def run_demo_backtest():
    """运行演示回测"""
    print("=" * 60)
    print("Enhanced Grid Strategy 回测演示")
    print("=" * 60)
    
    # 检查策略文件是否存在
    strategy_file = PROJECT_ROOT / "user_data/strategies/EnhancedGridStrategy.py"
    if not strategy_file.exists():
        print("❌ 策略文件不存在，请先运行完整的策略部署")
        return False
    
    # 检查配置文件
    config_file = PROJECT_ROOT / "user_data/config.json"
    if not config_file.exists():
        print("❌ 配置文件不存在")
        return False
        
    # 检查数据文件
    data_files = list((PROJECT_ROOT / "user_data/data").glob("**/*.json"))
    if not data_files:
        print("❌ 没有找到历史数据文件")
        print("请先运行数据下载脚本: ./docker_download_data.sh")
        return False
        
    print(f"✓ 找到 {len(data_files)} 个数据文件")
    
    # 运行简短回测
    print("\n🚀 开始运行演示回测...")
    print("时间范围: 2024-10-01 到 2024-11-01 (1个月)")
    print("交易对: BTC/USDT, ETH/USDT")
    
    cmd = [
        "freqtrade", "backtesting",
        "--config", str(config_file),
        "--strategy", "EnhancedGridStrategy", 
        "--timeframe", "5m",
        "--timerange", "20241001-20241101",
        "--pairs", "BTC/USDT", "ETH/USDT",
        "--export", "trades,signals",
        "--export-filename", "demo_backtest"
    ]
    
    try:
        print(f"\n执行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("\n✅ 回测完成!")
            print("\n📊 回测结果:")
            print("-" * 40)
            
            # 解析输出中的关键信息
            lines = result.stdout.split('\n')
            for line in lines:
                line = line.strip()
                if any(keyword in line for keyword in [
                    'Total trades', 'Total Profit', 'Win Rate', 'Max Drawdown', 
                    'Avg. Duration', 'Sharpe Ratio'
                ]):
                    print(line)
                    
            # 检查是否生成了导出文件
            export_dir = PROJECT_ROOT / "user_data/backtest_results"
            if export_dir.exists():
                export_files = list(export_dir.glob("*demo_backtest*"))
                if export_files:
                    print(f"\n📁 导出文件: {len(export_files)} 个")
                    for f in export_files[:3]:  # 显示前3个文件
                        print(f"   - {f.name}")
                        
        else:
            print("\n❌ 回测失败:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("\n⏰ 回测超时")
        return False
    except Exception as e:
        print(f"\n❌ 回测异常: {e}")
        return False
        
    print("\n" + "=" * 60)
    print("演示回测完成！")
    print("\n下一步:")
    print("1. 运行完整回测: python scripts/backtest_enhanced_grid.py")
    print("2. 运行场景测试: python scripts/backtest_scenarios.py")
    print("3. 生成分析报告: python scripts/analyze_backtest_results.py")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = run_demo_backtest()
    sys.exit(0 if success else 1)