#!/usr/bin/env python3
"""
纯命令行监控脚本 - 无 webserver
实时监控交易和库存状态
"""

import sqlite3
import time
import sys
from pathlib import Path
from datetime import datetime

def print_header():
    """打印监控头部信息"""
    print("\033[2J\033[H")  # 清屏
    print("🔄 AdvancedMarketMakerV2 实时监控")
    print(f"⏰ 更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

def monitor_database_console():
    """控制台数据库监控"""
    db_path = "demo_trades.sqlite"
    
    while True:
        print_header()
        
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
                    # 获取当前持仓
                    cursor.execute("""
                        SELECT id, pair, is_short, amount, open_rate, 
                               (julianday('now') - julianday(open_date)) * 24 * 60 as duration_minutes,
                               profit_ratio
                        FROM trades 
                        WHERE is_open = 1
                        ORDER BY open_date DESC
                    """)
                    
                    open_trades = cursor.fetchall()
                    
                    if open_trades:
                        print(f"📊 当前持仓 ({len(open_trades)} 个):")
                        print("-" * 80)
                        print(f"{'ID':<4} {'交易对':<12} {'方向':<4} {'数量':<12} {'开仓价':<10} {'持续时间':<10} {'盈亏率':<8}")
                        print("-" * 80)
                        
                        for trade in open_trades:
                            trade_id, pair, is_short, amount, open_rate, duration_minutes, profit_ratio = trade
                            
                            direction = "空头" if is_short else "多头"
                            duration_str = f"{int(duration_minutes)}m" if duration_minutes < 60 else f"{int(duration_minutes//60)}h{int(duration_minutes%60)}m"
                            profit_str = f"{profit_ratio:.2%}" if profit_ratio else "0.00%"
                            
                            print(f"{trade_id:<4} {pair:<12} {direction:<4} {amount:<12.6f} {open_rate:<10.4f} {duration_str:<10} {profit_str:<8}")
                        print()
                    else:
                        print("📊 当前持仓: 无\n")
                    
                    # 最近完成的交易
                    cursor.execute("""
                        SELECT id, pair, is_short, amount, open_rate, close_rate, 
                               profit_ratio, exit_reason,
                               (julianday(close_date) - julianday(open_date)) * 24 * 60 as duration_minutes
                        FROM trades 
                        WHERE is_open = 0
                        ORDER BY close_date DESC 
                        LIMIT 5
                    """)
                    
                    recent_closed = cursor.fetchall()
                    
                    if recent_closed:
                        print("📈 最近完成的交易:")
                        print("-" * 80)
                        print(f"{'ID':<4} {'交易对':<12} {'方向':<4} {'数量':<12} {'开仓价':<10} {'平仓价':<10} {'盈亏率':<8} {'原因':<10}")
                        print("-" * 80)
                        
                        for trade in recent_closed:
                            trade_id, pair, is_short, amount, open_rate, close_rate, profit_ratio, exit_reason, duration_minutes = trade
                            
                            direction = "空头" if is_short else "多头"
                            profit_str = f"{profit_ratio:.2%}" if profit_ratio else "0.00%"
                            exit_reason = exit_reason or "N/A"
                            
                            print(f"{trade_id:<4} {pair:<12} {direction:<4} {amount:<12.6f} {open_rate:<10.4f} {close_rate:<10.4f} {profit_str:<8} {exit_reason:<10}")
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
                        if closed > 0:
                            print(f"   胜率: {win_rate:.1f}% | 平均盈亏: {avg_profit:.2%} | 累计盈亏: {total_profit:.4f}")
                        else:
                            print(f"   胜率: 0% | 平均盈亏: 0% | 累计盈亏: 0%")
                        print()
                    
                    # 库存分析
                    cursor.execute("""
                        SELECT 
                            pair,
                            SUM(CASE WHEN is_open = 1 AND is_short = 0 THEN amount ELSE 0 END) as long_amount,
                            SUM(CASE WHEN is_open = 1 AND is_short = 1 THEN amount ELSE 0 END) as short_amount,
                            COUNT(CASE WHEN is_open = 1 AND is_short = 0 THEN 1 END) as long_count,
                            COUNT(CASE WHEN is_open = 1 AND is_short = 1 THEN 1 END) as short_count
                        FROM trades 
                        WHERE is_open = 1
                        GROUP BY pair
                    """)
                    
                    inventory = cursor.fetchall()
                    if inventory:
                        print("📦 库存状态:")
                        print("-" * 70)
                        print(f"{'交易对':<12} {'多头数量':<12} {'空头数量':<12} {'净持仓':<12} {'平衡状态':<10}")
                        print("-" * 70)
                        
                        for pair, long_amt, short_amt, long_count, short_count in inventory:
                            net_position = long_amt - short_amt
                            total_amount = long_amt + short_amt
                            
                            if total_amount > 0:
                                imbalance_ratio = abs(net_position) / total_amount
                                if imbalance_ratio < 0.1:
                                    status = "✅ 平衡"
                                elif imbalance_ratio < 0.3:
                                    status = "⚠️ 轻微失衡"
                                else:
                                    status = "❌ 严重失衡"
                            else:
                                status = "➖ 无持仓"
                            
                            print(f"{pair:<12} {long_amt:<12.6f} {short_amt:<12.6f} {net_position:<12.6f} {status:<10}")
                        print()
                    else:
                        print("📦 库存状态: 无持仓\n")
                
                conn.close()
                
            except Exception as e:
                print(f"❌ 数据库访问错误: {e}\n")
        else:
            print("⏳ 等待交易数据生成...\n")
        
        print("💡 提示:")
        print("   - 策略使用干跑模式，不会真实交易")
        print("   - 做市商策略同时开多空单来赚取价差")
        print("   - 理想状态是多空持仓保持平衡")
        print("   - 按 Ctrl+C 退出监控")
        print()
        print("📊 监控间隔: 10秒")
        
        try:
            time.sleep(10)  # 每10秒更新一次
        except KeyboardInterrupt:
            print("\n👋 退出监控")
            break

def show_summary():
    """显示最终总结"""
    db_path = "demo_trades.sqlite"
    
    if Path(db_path).exists():
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='trades'
            """)
            
            if cursor.fetchone():
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total,
                        COUNT(CASE WHEN is_open = 0 THEN 1 END) as closed,
                        SUM(CASE WHEN is_open = 0 AND profit_ratio > 0 THEN 1 ELSE 0 END) as winning,
                        SUM(CASE WHEN is_open = 0 THEN profit_ratio ELSE 0 END) as total_profit
                    FROM trades
                """)
                
                stats = cursor.fetchone()
                if stats and stats[0] > 0:
                    total, closed, winning, total_profit = stats
                    
                    print("\n" + "=" * 50)
                    print("📊 会话总结")
                    print("=" * 50)
                    print(f"总交易数: {total}")
                    print(f"已完成交易: {closed}")
                    if closed > 0:
                        win_rate = (winning / closed * 100)
                        print(f"胜率: {win_rate:.1f}%")
                        print(f"累计盈亏: {total_profit:.4f}")
                    print("=" * 50)
            
            conn.close()
        except Exception as e:
            print(f"获取总结数据失败: {e}")

if __name__ == "__main__":
    try:
        print("🚀 启动命令行监控")
        print("等待策略数据...")
        time.sleep(2)
        monitor_database_console()
    except KeyboardInterrupt:
        pass
    finally:
        show_summary()
        print("\n👋 监控已停止")
        sys.exit(0)