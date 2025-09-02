#!/usr/bin/env python3
"""
简化版运行脚本 - 不包含 webserver
专注于策略运行和命令行监控
"""

import subprocess
import time
import sys
import threading
import sqlite3
from datetime import datetime
from pathlib import Path

def check_dependencies():
    """检查依赖"""
    try:
        import freqtrade
        print(f"✅ FreqTrade 版本: {freqtrade.__version__}")
    except ImportError:
        print("❌ FreqTrade 未安装")
        return False
    
    config_file = Path("config_demo.json")
    if not config_file.exists():
        print("❌ 配置文件不存在")
        return False
    
    strategy_file = Path("user_data/strategies/AdvancedMarketMakerV2.py")
    if not strategy_file.exists():
        print("❌ 策略文件不存在")
        return False
        
    print("✅ 所有依赖检查通过")
    return True

def create_directories():
    """创建必要的目录"""
    directories = [
        "user_data/data",
        "logs",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def download_data():
    """下载历史数据"""
    print("📥 下载 BTC/USDT 历史数据...")
    cmd = [
        "uv", "run", "freqtrade", "download-data",
        "--config", "config_demo.json",
        "--timeframe", "1m",
        "--timerange", "20240901-",
        "--exchange", "binance"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            print("✅ 数据下载成功")
        else:
            print(f"⚠️ 数据下载可能失败: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("⚠️ 数据下载超时，将使用现有数据")
    except Exception as e:
        print(f"⚠️ 数据下载异常: {e}")

def monitor_trades_simple():
    """简化的交易监控"""
    db_path = "demo_trades.sqlite"
    
    print(f"\n📊 监控交易数据库: {db_path}")
    print("=" * 60)
    
    last_trade_count = 0
    
    while True:
        try:
            if Path(db_path).exists():
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # 检查交易表
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='trades'
                """)
                
                if cursor.fetchone():
                    # 获取交易统计
                    cursor.execute("""
                        SELECT 
                            COUNT(*) as total_trades,
                            COUNT(CASE WHEN is_open = 1 THEN 1 END) as open_trades,
                            COUNT(CASE WHEN is_open = 0 THEN 1 END) as closed_trades,
                            SUM(CASE WHEN is_open = 0 AND profit_ratio > 0 THEN 1 ELSE 0 END) as winning_trades,
                            SUM(CASE WHEN is_open = 0 THEN profit_ratio ELSE 0 END) as total_profit
                        FROM trades
                    """)
                    
                    stats = cursor.fetchone()
                    if stats and stats[0] > 0:
                        total, open_count, closed, winning, total_profit = stats
                        
                        # 只在交易数量变化时打印
                        if total != last_trade_count:
                            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 📊 交易更新:")
                            print(f"  总交易: {total} | 进行中: {open_count} | 已完成: {closed}")
                            
                            if closed > 0:
                                win_rate = (winning / closed * 100)
                                print(f"  胜率: {win_rate:.1f}% | 累计盈亏: {total_profit:.4f}")
                            
                            # 显示当前持仓
                            cursor.execute("""
                                SELECT 
                                    pair,
                                    SUM(CASE WHEN is_open = 1 AND is_short = 0 THEN amount ELSE 0 END) as long_amount,
                                    SUM(CASE WHEN is_open = 1 AND is_short = 1 THEN amount ELSE 0 END) as short_amount
                                FROM trades 
                                WHERE is_open = 1
                                GROUP BY pair
                            """)
                            
                            inventory = cursor.fetchall()
                            if inventory:
                                print("  📦 当前库存:")
                                for pair, long_amt, short_amt in inventory:
                                    net_position = long_amt - short_amt
                                    balance_ratio = abs(net_position) / (long_amt + short_amt) if (long_amt + short_amt) > 0 else 0
                                    status = "⚖️ 平衡" if balance_ratio < 0.2 else ("📈 偏多" if net_position > 0 else "📉 偏空")
                                    print(f"    {pair}: 多头={long_amt:.6f} 空头={short_amt:.6f} 净持仓={net_position:.6f} {status}")
                            
                            last_trade_count = total
                
                conn.close()
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ⏳ 等待交易数据生成...")
                
        except Exception as e:
            print(f"❌ 监控错误: {e}")
        
        time.sleep(15)  # 每15秒检查一次

def run_strategy():
    """运行策略"""
    print("🤖 启动做市商策略（干跑模式）...")
    
    cmd = [
        "uv", "run", "freqtrade", "trade",
        "--config", "config_demo.json",
        "--dry-run"
    ]
    
    try:
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        print("📋 策略运行日志:")
        print("=" * 60)
        
        # 启动监控线程
        monitor_thread = threading.Thread(target=monitor_trades_simple, daemon=True)
        monitor_thread.start()
        
        # 实时输出关键日志
        for line in iter(process.stdout.readline, ''):
            if line:
                line = line.rstrip()
                
                # 过滤和格式化日志输出
                if any(keyword in line for keyword in ['BUY', 'SELL', 'Entry', 'Exit']):
                    print(f"🔔 {line}")
                elif 'ERROR' in line:
                    print(f"❌ {line}")
                elif 'WARNING' in line:
                    print(f"⚠️ {line}")
                elif 'AdvancedMarketMakerV2' in line:
                    print(f"📊 {line}")
                elif any(keyword in line for keyword in ['Starting', 'Stopping', 'Bot']):
                    print(f"ℹ️ {line}")
        
        process.stdout.close()
        return_code = process.wait()
        
        if return_code != 0:
            print(f"❌ 策略退出，返回代码: {return_code}")
        else:
            print("✅ 策略正常退出")
            
    except KeyboardInterrupt:
        print("\n🛑 用户中断策略运行")
        if 'process' in locals():
            process.terminate()
    except Exception as e:
        print(f"❌ 运行策略异常: {e}")

def main():
    """主函数"""
    print("🚀 AdvancedMarketMakerV2 策略运行")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        return
    
    # 创建目录
    create_directories()
    
    # 询问是否下载数据
    try:
        download_choice = input("\n是否下载最新数据？(y/n，默认 n): ").strip().lower()
        if download_choice == 'y':
            download_data()
    except KeyboardInterrupt:
        print("\n👋 退出")
        return
    
    print("\n💡 运行说明:")
    print("- 策略运行在干跑模式，不会真实交易")
    print("- 做市商策略会同时开多空单来赚取价差")
    print("- 按 Ctrl+C 停止运行")
    print("- 监控信息会自动显示在控制台")
    print("")
    
    try:
        input("按回车键开始运行策略...")
    except KeyboardInterrupt:
        print("\n👋 取消运行")
        return
    
    # 运行策略
    run_strategy()

if __name__ == "__main__":
    main()