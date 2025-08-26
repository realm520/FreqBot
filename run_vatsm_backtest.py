#!/usr/bin/env python3
"""
VATSM策略回测脚本

使用方法:
python run_vatsm_backtest.py --symbol BTC/USDT --start-date 2020-01-01 --end-date 2024-08-26
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

def run_vatsm_backtest(symbol: str = "BTC/USDT", start_date: str = "2020-01-01", 
                      end_date: str = "2024-08-26", timeframe: str = "1d", 
                      initial_balance: float = 10000.0):
    """
    运行VATSM策略回测
    
    Args:
        symbol: 交易对，如 "BTC/USDT"
        start_date: 开始日期，格式 "YYYY-MM-DD"
        end_date: 结束日期，格式 "YYYY-MM-DD"  
        timeframe: 时间周期，如 "1d", "4h", "1h"
        initial_balance: 初始资金
    """
    
    # 更新配置文件
    config_path = "vatsm_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 更新配置参数
    config['timeframe'] = timeframe
    config['stake_amount'] = initial_balance * 0.1  # 每次交易使用10%资金
    config['exchange']['pair_whitelist'] = [symbol]
    
    # 保存更新后的配置
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"开始VATSM策略回测:")
    print(f"交易对: {symbol}")
    print(f"时间周期: {timeframe}")
    print(f"开始日期: {start_date}")
    print(f"结束日期: {end_date}")
    print(f"初始资金: {initial_balance} USDT")
    print("-" * 50)
    
    # 首先下载数据
    print("正在下载历史数据...")
    download_cmd = f"""
freqtrade download-data \\
    --config {config_path} \\
    --timeframe {timeframe} \\
    --timerange {start_date}-{end_date} \\
    --exchange binance
"""
    
    print(f"执行命令: {download_cmd}")
    os.system(download_cmd.replace('\\\n', ''))
    
    # 运行回测
    print("\n正在运行回测...")
    backtest_cmd = f"""
freqtrade backtesting \\
    --config {config_path} \\
    --strategy VATSMStrategy \\
    --timeframe {timeframe} \\
    --timerange {start_date}-{end_date} \\
    --breakdown day \\
    --cache none
"""
    
    print(f"执行命令: {backtest_cmd}")
    os.system(backtest_cmd.replace('\\\n', ''))
    
    # 生成分析报告
    print("\n正在生成分析报告...")
    plot_cmd = f"""
freqtrade plot-dataframe \\
    --config {config_path} \\
    --strategy VATSMStrategy \\
    --timeframe {timeframe} \\
    --timerange {start_date}-{end_date} \\
    --indicators1 momentum,vatsm_signal \\
    --indicators2 vol_short,vol_long,vol_ratio \\
    --indicators3 target_exposure,desired_exposure
"""
    
    print(f"执行命令: {plot_cmd}")
    os.system(plot_cmd.replace('\\\n', ''))

def main():
    parser = argparse.ArgumentParser(description='VATSM策略回测工具')
    parser.add_argument('--symbol', type=str, default='BTC/USDT', 
                       help='交易对 (默认: BTC/USDT)')
    parser.add_argument('--start-date', type=str, default='2020-01-01',
                       help='开始日期 YYYY-MM-DD (默认: 2020-01-01)')
    parser.add_argument('--end-date', type=str, default='2024-08-26',
                       help='结束日期 YYYY-MM-DD (默认: 2024-08-26)')
    parser.add_argument('--timeframe', type=str, default='1d',
                       help='时间周期 (默认: 1d)')
    parser.add_argument('--balance', type=float, default=10000.0,
                       help='初始资金 (默认: 10000.0)')
    
    args = parser.parse_args()
    
    run_vatsm_backtest(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        timeframe=args.timeframe,
        initial_balance=args.balance
    )

if __name__ == "__main__":
    main()