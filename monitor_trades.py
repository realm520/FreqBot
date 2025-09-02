#!/usr/bin/env python3
"""
实时监控交易和库存状态
"""

import sqlite3
import json
import time
import sys
from pathlib import Path
from datetime import datetime
import requests

def get_balance_info():
    """获取余额信息（通过 API）"""
    try:
        # FreqTrade API 端点
        api_url = "http://localhost:8080"
        auth = ("demo", "demo123")
        
        # 获取余额
        response = requests.get(f"{api_url}/api/v1/balance", auth=auth, timeout=5)
        if response.status_code == 200:
            balance_data = response.json()
            return balance_data
    except:
        pass
    return None

def get_open_trades_api():
    """通过 API 获取开放交易"""
    try:
        api_url = "http://localhost:8080"
        auth = ("demo", "demo123")
        
        response = requests.get(f"{api_url}/api/v1/trades", auth=auth, timeout=5)
        if response.status_code == 200:
            trades_data = response.json()
            return trades_data
    except:
        pass
    return None

def monitor_database_detailed():
    """详细监控数据库"""
    db_path = "demo_trades.sqlite"
    
    while True:
        print("\033[2J\033[H")  # 清屏
        print("🔄 AdvancedMarketMakerV2 实时监控")
        print(f"⏰ 更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # API 信息
        balance_info = get_balance_info()
        trades_api = get_open_trades_api()
        
        if balance_info:
            print("💰 账户余额 (API):")
            for currency, data in balance_info.get('currencies', {}).items():
                if data.get('balance', 0) > 0:
                    print(f"   {currency}: 可用={data.get('free', 0):.4f}, 锁定={data.get('used', 0):.4f}, 总计={data.get('balance', 0):.4f}")
            print()
        
        if trades_api:
            open_trades = [t for t in trades_api if t.get('is_open', False)]
            if open_trades:
                print(f"📊 当前持仓 ({len(open_trades)} 个):")
                print("-" * 80)
                print(f"{'交易对':<12} {'方向':<4} {'数量':<12} {'开仓价':<10} {'当前价':<10} {'盈亏率':<8} {'持续时间':<10}")
                print("-" * 80)
                
                for trade in open_trades:
                    pair = trade.get('pair', 'N/A')
                    direction = "空头" if trade.get('is_short', False) else "多头"
                    amount = trade.get('amount', 0)
                    open_rate = trade.get('open_rate', 0)
                    current_rate = trade.get('current_rate', 0)
                    profit_ratio = trade.get('profit_ratio', 0)
                    duration = trade.get('trade_duration_s', 0)
                    
                    duration_str = f"{duration//60}m" if duration < 3600 else f"{duration//3600}h{(duration%3600)//60}m"
                    
                    print(f"{pair:<12} {direction:<4} {amount:<12.6f} {open_rate:<10.4f} {current_rate:<10.4f} {profit_ratio:<8.2%} {duration_str:<10}")
                print()
        
        # 数据库信息
        if Path(db_path).exists():
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # 检查表是否存在
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='trades'
                """)
                
                if cursor.fetchone():
                    # 最新交易
                    cursor.execute("""
                        SELECT id, pair, is_open, amount, open_rate, close_rate, 
                               profit_ratio, open_date, close_date, exit_reason, is_short
                        FROM trades 
                        ORDER BY open_date DESC 
                        LIMIT 5
                    """)
                    
                    recent_trades = cursor.fetchall()
                    
                    if recent_trades:
                        print("📈 最新交易记录:")
                        print("-" * 80)
                        print(f"{'ID':<4} {'交易对':<12} {'方向':<4} {'状态':<6} {'数量':<12} {'开仓价':<10} {'盈亏率':<8}")
                        print("-" * 80)
                        
                        for trade in recent_trades:
                            trade_id, pair, is_open, amount, open_rate, close_rate, profit_ratio, open_date, close_date, exit_reason, is_short = trade
                            
                            direction = "空头" if is_short else "多头"  
                            status = "进行中" if is_open else "已完成"
                            profit_str = f"{profit_ratio:.2%}" if profit_ratio else "0.00%"
                            
                            print(f"{trade_id:<4} {pair:<12} {direction:<4} {status:<6} {amount:<12.6f} {open_rate:<10.4f} {profit_str:<8}")
                        print()
                    
                    # 统计信息
                    cursor.execute("""
                        SELECT 
                            COUNT(*) as total,
                            COUNT(CASE WHEN is_open = 1 THEN 1 END) as open_count,
                            COUNT(CASE WHEN is_open = 0 THEN 1 END) as closed,
                            SUM(CASE WHEN is_open = 0 AND profit_ratio > 0 THEN 1 ELSE 0 END) as winning,
                            SUM(CASE WHEN is_open = 0 THEN profit_ratio ELSE 0 END) as total_profit,
                            AVG(CASE WHEN is_open = 0 THEN profit_ratio END) as avg_profit
                        FROM trades
                    """)
                    
                    stats = cursor.fetchone()
                    if stats and stats[0] > 0:
                        total, open_count, closed, winning, total_profit, avg_profit = stats
                        win_rate = (winning / closed * 100) if closed > 0 else 0
                        
                        print("📊 交易统计:")
                        print(f"   总交易: {total} | 进行中: {open_count} | 已完成: {closed}")
                        print(f"   胜率: {win_rate:.1f}% | 平均盈亏: {avg_profit:.2%}" if avg_profit else "   胜率: 0% | 平均盈亏: 0%")
                        print(f"   累计盈亏: {total_profit:.2%}" if total_profit else "   累计盈亏: 0%")
                        print()
                    
                    # 库存分析
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
                        print("📦 库存状态:")
                        print("-" * 50)
                        print(f"{'交易对':<12} {'多头':<12} {'空头':<12} {'净持仓':<12}")
                        print("-" * 50)
                        
                        for pair, long_amt, short_amt in inventory:
                            net_position = long_amt - short_amt
                            print(f"{pair:<12} {long_amt:<12.6f} {short_amt:<12.6f} {net_position:<12.6f}")
                        print()
                
                conn.close()
                
            except Exception as e:
                print(f"❌ 数据库访问错误: {e}")
        else:
            print("⏳ 等待交易数据生成...")
        
        print("💡 提示:")
        print("   - 访问 http://localhost:8080 查看 Web 界面 (用户名: demo, 密码: demo123)")
        print("   - 按 Ctrl+C 退出监控")
        print("   - 做市商策略会同时开多空单来赚取价差")
        
        try:
            time.sleep(10)  # 每10秒更新一次
        except KeyboardInterrupt:
            print("\n👋 退出监控")
            break

if __name__ == "__main__":
    try:
        monitor_database_detailed()
    except KeyboardInterrupt:
        print("\n👋 监控已停止")
        sys.exit(0)