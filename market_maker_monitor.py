#!/usr/bin/env python3
"""
FreqBot做市商策略监控和管理工具
提供实时监控、风控管理、数据查询等功能
"""

import sys
import time
import json
import sqlite3
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import argparse
import logging


@dataclass 
class MarketMakerMetrics:
    """做市商关键指标"""
    total_trades: int = 0
    win_rate: float = 0.0
    avg_profit: float = 0.0
    total_pnl: float = 0.0
    daily_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    inventory_balance: float = 0.0
    active_orders: int = 0
    spread_capture: float = 0.0


class FreqTradeAPIClient:
    """FreqTrade API客户端"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8080", username: str = "market_maker", password: str = "mm123456"):
        self.base_url = base_url.rstrip('/')
        self.username = username
        self.password = password
        self.token = None
        self.session = requests.Session()
        
    def authenticate(self) -> bool:
        """API认证"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/token/login",
                json={"username": self.username, "password": self.password}
            )
            if response.status_code == 200:
                data = response.json()
                self.token = data.get("access_token")
                self.session.headers.update({"Authorization": f"Bearer {self.token}"})
                return True
            else:
                print(f"认证失败: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"认证错误: {e}")
            return False
    
    def get_status(self) -> Optional[Dict]:
        """获取机器人状态"""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/status")
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            print(f"获取状态失败: {e}")
            return None
    
    def get_trades(self) -> Optional[List[Dict]]:
        """获取交易记录"""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/trades")
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            print(f"获取交易记录失败: {e}")
            return None
    
    def get_open_trades(self) -> Optional[List[Dict]]:
        """获取开放交易"""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/trades/open")
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            print(f"获取开放交易失败: {e}")
            return None
    
    def get_profit(self) -> Optional[Dict]:
        """获取盈亏统计"""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/profit")
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            print(f"获取盈亏统计失败: {e}")
            return None
    
    def force_entry(self, pair: str, side: str) -> Optional[Dict]:
        """强制入场"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/forceentry",
                json={"pair": pair, "side": side}
            )
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            print(f"强制入场失败: {e}")
            return None
    
    def force_exit(self, trade_id: int) -> Optional[Dict]:
        """强制出场"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/forceexit",
                json={"tradeid": trade_id}
            )
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            print(f"强制出场失败: {e}")
            return None
    
    def get_balance(self) -> Optional[Dict]:
        """获取余额"""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/balance")
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            print(f"获取余额失败: {e}")
            return None


class MarketMakerMonitor:
    """做市商策略监控器"""
    
    def __init__(self, db_path: str = "user_data/market_maker_trades.db"):
        self.db_path = db_path
        self.api_client = FreqTradeAPIClient()
        self.logger = self.setup_logger()
        
    def setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger("MarketMakerMonitor")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def connect_api(self) -> bool:
        """连接API"""
        return self.api_client.authenticate()
    
    def calculate_metrics(self) -> MarketMakerMetrics:
        """计算做市商关键指标"""
        metrics = MarketMakerMetrics()
        
        # 从API获取数据
        trades = self.api_client.get_trades()
        open_trades = self.api_client.get_open_trades()
        profit_data = self.api_client.get_profit()
        
        if trades:
            # 基础交易统计
            metrics.total_trades = len(trades)
            
            if trades:
                closed_trades = [t for t in trades if t.get('is_open', True) == False]
                if closed_trades:
                    profits = [float(t.get('profit_abs', 0)) for t in closed_trades]
                    metrics.avg_profit = sum(profits) / len(profits)
                    metrics.win_rate = len([p for p in profits if p > 0]) / len(profits)
        
        if profit_data:
            metrics.total_pnl = float(profit_data.get('profit_all_fiat', 0))
            metrics.daily_pnl = float(profit_data.get('profit_today_fiat', 0))
        
        if open_trades:
            metrics.active_orders = len(open_trades)
            # 计算库存平衡
            long_amount = sum(float(t.get('amount', 0)) for t in open_trades if not t.get('is_short', False))
            short_amount = sum(float(t.get('amount', 0)) for t in open_trades if t.get('is_short', False))
            total_amount = long_amount + short_amount
            
            if total_amount > 0:
                metrics.inventory_balance = (long_amount - short_amount) / total_amount
        
        return metrics
    
    def get_risk_assessment(self, metrics: MarketMakerMetrics) -> Dict[str, Any]:
        """风险评估"""
        risk_assessment = {
            "risk_level": "LOW",
            "warnings": [],
            "recommendations": []
        }
        
        # PnL风险检查
        if metrics.daily_pnl < -100:
            risk_assessment["risk_level"] = "HIGH"
            risk_assessment["warnings"].append(f"日亏损过大: {metrics.daily_pnl:.2f}")
            risk_assessment["recommendations"].append("考虑停止交易或降低仓位")
        
        # 库存失衡检查
        if abs(metrics.inventory_balance) > 0.3:
            risk_level = "MEDIUM" if risk_assessment["risk_level"] == "LOW" else "HIGH"
            risk_assessment["risk_level"] = risk_level
            risk_assessment["warnings"].append(f"库存失衡严重: {metrics.inventory_balance:.3f}")
            risk_assessment["recommendations"].append("需要平衡库存")
        
        # 胜率检查
        if metrics.win_rate < 0.4 and metrics.total_trades > 10:
            risk_level = "MEDIUM" if risk_assessment["risk_level"] == "LOW" else "HIGH"  
            risk_assessment["risk_level"] = risk_level
            risk_assessment["warnings"].append(f"胜率过低: {metrics.win_rate:.2%}")
            risk_assessment["recommendations"].append("检查策略参数或市场条件")
        
        return risk_assessment
    
    def print_dashboard(self):
        """打印监控仪表盘"""
        print("\n" + "="*80)
        print("           FreqBot 做市商策略监控仪表盘")
        print("="*80)
        
        # API状态检查
        if not self.connect_api():
            print("❌ API连接失败，请检查FreqTrade是否运行")
            return
        
        # 获取状态
        status = self.api_client.get_status()
        if status:
            state = status.get('state', 'UNKNOWN')
            print(f"🤖 机器人状态: {state}")
            if status.get('strategy'):
                print(f"📊 当前策略: {status['strategy']}")
        
        # 计算指标
        metrics = self.calculate_metrics()
        
        # 显示关键指标
        print("\n📈 关键指标:")
        print(f"   总交易数: {metrics.total_trades}")
        print(f"   活跃订单: {metrics.active_orders}")
        print(f"   胜率: {metrics.win_rate:.2%}")
        print(f"   平均盈利: {metrics.avg_profit:.4f}")
        print(f"   总PnL: {metrics.total_pnl:.2f}")
        print(f"   今日PnL: {metrics.daily_pnl:.2f}")
        print(f"   库存平衡: {metrics.inventory_balance:.3f}")
        
        # 风险评估
        risk = self.get_risk_assessment(metrics)
        risk_color = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
        print(f"\n⚡ 风险等级: {risk_color.get(risk['risk_level'], '⚪')} {risk['risk_level']}")
        
        if risk["warnings"]:
            print("⚠️  警告:")
            for warning in risk["warnings"]:
                print(f"   • {warning}")
        
        if risk["recommendations"]:
            print("💡 建议:")
            for rec in risk["recommendations"]:
                print(f"   • {rec}")
        
        # 余额信息
        balance = self.api_client.get_balance()
        if balance:
            print("\n💰 余额信息:")
            for currency, data in balance.get('currencies', {}).items():
                free = data.get('free', 0)
                used = data.get('used', 0)
                total = data.get('total', 0)
                if total > 0:
                    print(f"   {currency}: {total:.6f} (可用: {free:.6f}, 占用: {used:.6f})")
        
        print("\n" + "="*80)
        print(f"📅 更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
    
    def export_data(self, output_file: str = "market_maker_report.json"):
        """导出数据报告"""
        if not self.connect_api():
            print("API连接失败")
            return
        
        # 收集所有数据
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "status": self.api_client.get_status(),
            "trades": self.api_client.get_trades(),
            "open_trades": self.api_client.get_open_trades(),
            "profit": self.api_client.get_profit(),
            "balance": self.api_client.get_balance(),
            "metrics": self.calculate_metrics().__dict__,
            "risk_assessment": self.get_risk_assessment(self.calculate_metrics())
        }
        
        # 保存到文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✅ 数据已导出到: {output_file}")
    
    def continuous_monitor(self, interval: int = 30):
        """持续监控模式"""
        print(f"🔄 开始持续监控，刷新间隔: {interval}秒")
        print("按 Ctrl+C 停止监控")
        
        try:
            while True:
                self.print_dashboard()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n👋 监控已停止")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='FreqBot做市商策略监控工具')
    parser.add_argument('--mode', choices=['dashboard', 'monitor', 'export'], 
                       default='dashboard', help='运行模式')
    parser.add_argument('--interval', type=int, default=30, 
                       help='持续监控的刷新间隔（秒）')
    parser.add_argument('--output', default='market_maker_report.json',
                       help='导出文件名')
    parser.add_argument('--db', default='user_data/market_maker_trades.db',
                       help='数据库路径')
    
    args = parser.parse_args()
    
    monitor = MarketMakerMonitor(db_path=args.db)
    
    if args.mode == 'dashboard':
        monitor.print_dashboard()
    elif args.mode == 'monitor':
        monitor.continuous_monitor(interval=args.interval)
    elif args.mode == 'export':
        monitor.export_data(output_file=args.output)


if __name__ == "__main__":
    main()