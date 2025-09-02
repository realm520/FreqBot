"""交易监控 - 统一的交易状态监控"""

import sqlite3
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class TradeMonitor:
    """交易监控器"""
    
    def __init__(self, db_path: str = "trades.sqlite"):
        self.db_path = db_path
        self.monitoring = False
        self.callbacks: List[Callable[[Dict], None]] = []
        self.last_trade_count = 0
        self.update_interval = 15  # 秒
    
    def add_callback(self, callback: Callable[[Dict], None]):
        """添加状态更新回调"""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[Dict], None]):
        """移除状态更新回调"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def get_trade_stats(self) -> Optional[Dict[str, Any]]:
        """获取交易统计"""
        if not Path(self.db_path).exists():
            return None
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 检查表是否存在
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='trades'
            """)
            
            if not cursor.fetchone():
                conn.close()
                return None
            
            # 获取基础统计
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_trades,
                    COUNT(CASE WHEN is_open = 1 THEN 1 END) as open_trades,
                    COUNT(CASE WHEN is_open = 0 THEN 1 END) as closed_trades,
                    SUM(CASE WHEN is_open = 0 AND close_profit > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(CASE WHEN is_open = 0 THEN close_profit ELSE 0 END) as total_profit_ratio,
                    SUM(CASE WHEN is_open = 0 THEN close_profit_abs ELSE 0 END) as total_profit_abs,
                    AVG(CASE WHEN is_open = 0 THEN close_profit ELSE NULL END) as avg_profit_ratio,
                    MIN(CASE WHEN is_open = 0 THEN close_profit ELSE NULL END) as min_profit_ratio,
                    MAX(CASE WHEN is_open = 0 THEN close_profit ELSE NULL END) as max_profit_ratio
                FROM trades
            """)
            
            stats = cursor.fetchone()
            
            if not stats or stats[0] == 0:
                conn.close()
                return {
                    "total_trades": 0,
                    "open_trades": 0,
                    "closed_trades": 0,
                    "winning_trades": 0,
                    "win_rate": 0.0,
                    "total_profit_ratio": 0.0,
                    "total_profit_abs": 0.0,
                    "avg_profit_ratio": 0.0,
                    "positions": [],
                    "recent_trades": []
                }
            
            total, open_count, closed, winning = stats[:4]
            total_profit_ratio, total_profit_abs, avg_profit_ratio = stats[4:7]
            min_profit_ratio, max_profit_ratio = stats[7:9]
            
            win_rate = (winning / closed * 100) if closed > 0 else 0.0
            
            # 获取当前持仓
            cursor.execute("""
                SELECT 
                    pair,
                    CASE WHEN is_short = 1 THEN 'short' ELSE 'long' END as side,
                    SUM(amount) as total_amount,
                    AVG(open_rate) as avg_open_rate,
                    COUNT(*) as position_count
                FROM trades 
                WHERE is_open = 1
                GROUP BY pair, is_short
                ORDER BY pair, is_short
            """)
            
            positions = []
            for row in cursor.fetchall():
                pair, side, amount, avg_rate, count = row
                positions.append({
                    "pair": pair,
                    "side": side,
                    "amount": amount,
                    "avg_open_rate": avg_rate,
                    "position_count": count
                })
            
            # 获取最近交易
            cursor.execute("""
                SELECT 
                    pair, 
                    CASE WHEN is_short = 1 THEN 'short' ELSE 'long' END as side, 
                    amount, open_rate, close_rate,
                    close_profit, close_profit_abs, open_date, close_date
                FROM trades 
                WHERE is_open = 0
                ORDER BY close_date DESC
                LIMIT 10
            """)
            
            recent_trades = []
            for row in cursor.fetchall():
                pair, side, amount, open_rate, close_rate = row[:5]
                close_profit, close_profit_abs, open_date, close_date = row[5:9]
                
                recent_trades.append({
                    "pair": pair,
                    "side": side,
                    "amount": amount,
                    "open_rate": open_rate,
                    "close_rate": close_rate,
                    "profit_ratio": close_profit,
                    "profit_abs": close_profit_abs,
                    "open_date": open_date,
                    "close_date": close_date
                })
            
            conn.close()
            
            return {
                "total_trades": total,
                "open_trades": open_count,
                "closed_trades": closed,
                "winning_trades": winning,
                "win_rate": win_rate,
                "total_profit_ratio": total_profit_ratio or 0.0,
                "total_profit_abs": total_profit_abs or 0.0,
                "avg_profit_ratio": avg_profit_ratio or 0.0,
                "min_profit_ratio": min_profit_ratio or 0.0,
                "max_profit_ratio": max_profit_ratio or 0.0,
                "positions": positions,
                "recent_trades": recent_trades,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"获取交易统计失败: {e}")
            return None
    
    def get_inventory_balance(self) -> Dict[str, Any]:
        """获取库存平衡情况"""
        stats = self.get_trade_stats()
        if not stats or not stats["positions"]:
            return {"balanced": True, "pairs": {}}
        
        pairs_balance = {}
        overall_balanced = True
        
        # 按交易对分组计算平衡
        for pos in stats["positions"]:
            pair = pos["pair"]
            if pair not in pairs_balance:
                pairs_balance[pair] = {"long": 0, "short": 0}
            
            if pos["side"] == "long":
                pairs_balance[pair]["long"] += pos["amount"]
            else:
                pairs_balance[pair]["short"] += pos["amount"]
        
        # 计算平衡比例
        for pair, balance in pairs_balance.items():
            long_amount = balance["long"]
            short_amount = balance["short"]
            total_amount = long_amount + short_amount
            
            if total_amount > 0:
                net_position = long_amount - short_amount
                balance_ratio = abs(net_position) / total_amount
                
                balance["net_position"] = net_position
                balance["balance_ratio"] = balance_ratio
                balance["balanced"] = balance_ratio < 0.2  # 20% 以内认为平衡
                
                if not balance["balanced"]:
                    overall_balanced = False
            else:
                balance["net_position"] = 0
                balance["balance_ratio"] = 0
                balance["balanced"] = True
        
        return {
            "balanced": overall_balanced,
            "pairs": pairs_balance
        }
    
    def start_monitoring(self, callback: Optional[Callable[[Dict], None]] = None):
        """开始监控"""
        if callback:
            self.add_callback(callback)
        
        self.monitoring = True
        logger.info("开始交易监控")
        
        while self.monitoring:
            try:
                stats = self.get_trade_stats()
                if stats:
                    # 检查是否有变化
                    if stats["total_trades"] != self.last_trade_count:
                        self.last_trade_count = stats["total_trades"]
                        
                        # 通知所有回调
                        for callback in self.callbacks:
                            try:
                                callback(stats)
                            except Exception as e:
                                logger.error(f"回调执行失败: {e}")
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"监控异常: {e}")
                time.sleep(self.update_interval)
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        logger.info("停止交易监控")
    
    def print_stats(self, stats: Optional[Dict] = None):
        """打印统计信息"""
        if stats is None:
            stats = self.get_trade_stats()
        
        if not stats:
            print("📊 暂无交易数据")
            return
        
        print(f"\n📊 交易统计 [{stats.get('timestamp', '')}]")
        print("=" * 60)
        print(f"总交易: {stats['total_trades']} | 进行中: {stats['open_trades']} | 已完成: {stats['closed_trades']}")
        
        if stats['closed_trades'] > 0:
            print(f"胜率: {stats['win_rate']:.1f}% | 累计收益率: {stats['total_profit_ratio']:.4f}")
            print(f"平均收益率: {stats['avg_profit_ratio']:.4f} | 累计收益: {stats['total_profit_abs']:.4f}")
        
        # 显示持仓
        if stats['positions']:
            print("\n📦 当前持仓:")
            balance_info = self.get_inventory_balance()
            
            for pair, balance in balance_info['pairs'].items():
                long_amt = balance['long']
                short_amt = balance['short']
                net_pos = balance['net_position']
                status = "⚖️ 平衡" if balance['balanced'] else ("📈 偏多" if net_pos > 0 else "📉 偏空")
                
                print(f"  {pair}: 多={long_amt:.6f} 空={short_amt:.6f} 净={net_pos:.6f} {status}")
        
        # 显示最近交易
        if stats['recent_trades']:
            print(f"\n📋 最近交易 (显示最新{min(5, len(stats['recent_trades']))}笔):")
            for trade in stats['recent_trades'][:5]:
                profit_sign = "📈" if trade['profit_ratio'] > 0 else "📉"
                print(f"  {profit_sign} {trade['pair']} {trade['side']} "
                      f"收益: {trade['profit_ratio']:.4f} ({trade['profit_abs']:.4f})")
    
    def export_stats_json(self, file_path: str):
        """导出统计为JSON"""
        import json
        
        stats = self.get_trade_stats()
        if stats:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            logger.info(f"统计数据已导出: {file_path}")
        else:
            logger.warning("无统计数据可导出")