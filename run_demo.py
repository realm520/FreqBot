#!/usr/bin/env python3
"""
AdvancedMarketMakerV2 策略演示运行脚本
"""

import subprocess
import time
import sys
import threading
import sqlite3
import json
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
        print(f"✅ 目录创建: {directory}")

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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("✅ 数据下载成功")
        else:
            print(f"⚠️ 数据下载可能失败: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("⚠️ 数据下载超时，将使用现有数据")
    except Exception as e:
        print(f"⚠️ 数据下载异常: {e}")

def start_webserver():
    """启动 Web 服务器"""
    print("🌐 启动 FreqTrade Web 服务器...")
    cmd = [
        "uv", "run", "freqtrade", "webserver",
        "--config", "config_demo.json"
    ]
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(3)  # 等待启动
        
        if process.poll() is None:
            print("✅ Web 服务器启动成功")
            print("🔗 访问地址: http://localhost:8080")
            print("🔐 用户名: demo, 密码: demo123")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Web 服务器启动失败: {stderr.decode()}")
            return None
    except Exception as e:
        print(f"❌ 启动 Web 服务器异常: {e}")
        return None

def monitor_database():
    """监控数据库中的交易"""
    db_path = "demo_trades.sqlite"
    
    print(f"\n📊 监控交易数据库: {db_path}")
    print("=" * 60)
    
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
                    # 获取所有交易
                    cursor.execute("""
                        SELECT id, pair, is_open, amount, open_rate, close_rate, 
                               profit_ratio, open_date, close_date, trade_duration,
                               exit_reason, is_short
                        FROM trades 
                        ORDER BY open_date DESC 
                        LIMIT 10
                    """)
                    
                    trades = cursor.fetchall()
                    
                    if trades:
                        print(f"\n🔄 最新交易记录 ({datetime.now().strftime('%H:%M:%S')}):")
                        print("-" * 120)
                        print(f"{'ID':<4} {'交易对':<10} {'方向':<4} {'状态':<6} {'数量':<12} {'开仓价':<10} {'平仓价':<10} {'盈亏率':<8} {'持续时间':<12}")
                        print("-" * 120)
                        
                        for trade in trades:
                            trade_id, pair, is_open, amount, open_rate, close_rate, profit_ratio, open_date, close_date, duration, exit_reason, is_short = trade
                            
                            direction = "空头" if is_short else "多头"
                            status = "进行中" if is_open else "已平仓"
                            close_rate_str = f"{close_rate:.4f}" if close_rate else "N/A"
                            profit_str = f"{profit_ratio:.2%}" if profit_ratio else "N/A"
                            duration_str = str(duration) if duration else "N/A"
                            
                            print(f"{trade_id:<4} {pair:<10} {direction:<4} {status:<6} {amount:<12.6f} {open_rate:<10.4f} {close_rate_str:<10} {profit_str:<8} {duration_str:<12}")
                    
                    # 获取统计信息
                    cursor.execute("""
                        SELECT 
                            COUNT(*) as total_trades,
                            COUNT(CASE WHEN is_open = 1 THEN 1 END) as open_trades,
                            COUNT(CASE WHEN is_open = 0 THEN 1 END) as closed_trades,
                            SUM(CASE WHEN is_open = 0 AND profit_ratio > 0 THEN 1 ELSE 0 END) as winning_trades,
                            AVG(CASE WHEN is_open = 0 THEN profit_ratio END) as avg_profit_ratio
                        FROM trades
                    """)
                    
                    stats = cursor.fetchone()
                    if stats and stats[0] > 0:
                        total, open_count, closed, winning, avg_profit = stats
                        win_rate = (winning / closed * 100) if closed > 0 else 0
                        
                        print(f"\n📈 统计信息:")
                        print(f"   总交易数: {total}, 进行中: {open_count}, 已完成: {closed}")
                        print(f"   胜率: {win_rate:.1f}%, 平均盈亏率: {avg_profit:.2%}" if avg_profit else "   胜率: 0%, 平均盈亏率: N/A")
                
                conn.close()
            else:
                print("⏳ 等待数据库创建...")
                
        except Exception as e:
            print(f"❌ 数据库监控错误: {e}")
        
        time.sleep(10)  # 每10秒更新一次

def run_strategy_dry():
    """运行策略（干跑模式）"""
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
        
        # 实时输出日志
        print("📋 策略运行日志:")
        print("=" * 60)
        
        for line in iter(process.stdout.readline, ''):
            if line:
                print(f"[FreqTrade] {line.rstrip()}")
                
                # 检查关键信息
                if "BUY" in line or "SELL" in line:
                    print(f"🔔 交易信号: {line.rstrip()}")
                elif "ERROR" in line:
                    print(f"⚠️ 错误: {line.rstrip()}")
        
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
    print("🚀 AdvancedMarketMakerV2 策略演示")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        return
    
    # 创建目录
    create_directories()
    
    # 下载数据
    download_data()
    
    print("\n选择运行模式:")
    print("1. 仅启动 Web 监控界面")
    print("2. 启动策略并实时监控")
    print("3. 仅运行策略（无 Web 界面）")
    
    try:
        choice = input("\n请输入选择 (1-3): ").strip()
    except KeyboardInterrupt:
        print("\n👋 退出")
        return
    
    if choice == "1":
        # 仅启动 Web 服务器
        webserver_process = start_webserver()
        if webserver_process:
            try:
                print("\n按 Ctrl+C 停止服务器...")
                webserver_process.wait()
            except KeyboardInterrupt:
                print("\n🛑 停止 Web 服务器")
                webserver_process.terminate()
    
    elif choice == "2":
        # 启动 Web 服务器和策略
        webserver_process = start_webserver()
        
        if webserver_process:
            # 在新线程中启动数据库监控
            monitor_thread = threading.Thread(target=monitor_database, daemon=True)
            monitor_thread.start()
            
            print("\n⏳ 等待 5 秒后启动策略...")
            time.sleep(5)
            
            try:
                run_strategy_dry()
            except KeyboardInterrupt:
                print("\n🛑 停止所有服务")
            finally:
                if webserver_process.poll() is None:
                    webserver_process.terminate()
    
    elif choice == "3":
        # 仅运行策略
        run_strategy_dry()
    
    else:
        print("❌ 无效选择")

if __name__ == "__main__":
    main()