#!/usr/bin/env python3
"""
通用做市商策略
支持ccxt交易所和自定义交易所
"""

import sys
import os
import time
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# 添加exchange_configs路径
exchange_configs_path = Path(__file__).parent.parent.parent / "exchange_configs"
sys.path.append(str(exchange_configs_path))

try:
    import ccxt
    from custom_exchange_manager import create_exchange_instance, exchange_manager
    from custom_exchange_base import CustomExchangeBase
except ImportError as e:
    print(f"导入依赖失败: {e}")
    print("请确保已安装ccxt库并正确配置exchange_configs路径")
    sys.exit(1)


class OrderStatus(Enum):
    """订单状态枚举"""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    FAILED = "failed"


class OrderType(Enum):
    """订单类型枚举"""
    NORMAL = "normal"  # 正常做市订单
    REBALANCE = "rebalance"  # 仓位平衡订单


@dataclass
class OrderInfo:
    """订单信息数据类"""
    order_id: str
    side: str  # 'buy' or 'sell'
    price: float
    amount: float
    timestamp: datetime
    order_type: OrderType = OrderType.NORMAL
    status: OrderStatus = OrderStatus.PENDING


@dataclass
class Position:
    """持仓信息"""
    base_balance: float = 0.0  # 基础货币余额
    quote_balance: float = 0.0  # 计价货币余额
    base_locked: float = 0.0   # 基础货币锁定
    quote_locked: float = 0.0  # 计价货币锁定
    
    @property
    def base_available(self) -> float:
        return self.base_balance - self.base_locked
    
    @property
    def quote_available(self) -> float:
        return self.quote_balance - self.quote_locked


@dataclass
class PositionImbalance:
    """仓位失衡信息"""
    is_imbalanced: bool = False
    excess_side: str = ""  # 'base' or 'quote'
    excess_amount: float = 0.0
    target_price: float = 0.0  # 目标平衡价格


class UniversalMarketMaker:
    """
    通用做市商策略
    
    支持特性：
    1. ccxt交易所和自定义交易所统一接口
    2. 仓位平衡管理
    3. 智能订单避让
    4. 风险控制
    """
    
    def __init__(self, config_file: str = None, **kwargs):
        """初始化通用做市商策略"""
        # 加载配置
        self.config = self._load_config(config_file, **kwargs)
        
        # 基础设置
        self.symbol = self.config['symbol']
        
        # 创建交易所实例 (支持ccxt和自定义交易所)
        self.exchange = self._init_exchange(self.config['exchange'])
        self.exchange_type = exchange_manager.get_exchange_type(self.config['exchange']['name'])
        
        # 订单管理
        self.active_orders: Dict[str, OrderInfo] = {}
        self.order_history: List[OrderInfo] = []
        self.position = Position()
        
        # 策略参数
        self.min_order_size = self.config.get('min_order_size', 0.001)
        self.spread_ratio = self.config.get('spread_ratio', 0.002)
        self.price_update_threshold = self.config.get('price_update_threshold', 0.0005)
        self.check_interval = self.config.get('check_interval', 1.0)
        
        # 仓位平衡参数
        self.position_imbalance_threshold = self.config.get('position_imbalance_threshold', 0.1)
        self.rebalance_urgency_multiplier = self.config.get('rebalance_urgency_multiplier', 2.0)
        self.retreat_distance = self.config.get('retreat_distance', 0.001)
        
        # 风险控制
        self.max_daily_loss = self.config.get('max_daily_loss', 100.0)
        self.position_limit = self.config.get('position_limit', 1000.0)
        
        # 运行状态
        self.is_running = False
        self.last_orderbook_update = None
        self.daily_pnl = 0.0
        self.start_time = datetime.now()
        self.current_mid_price = 0.0
        
        # 日志和统计
        self.logger = self._setup_logger()
        self.stats = {
            'orders_placed': 0,
            'orders_filled': 0,
            'total_volume': 0.0,
            'rebalance_orders': 0,
            'normal_orders': 0
        }
        
        # 初始化市场信息
        self._load_market_info()
    
    def _load_config(self, config_file: str, **kwargs) -> Dict:
        """加载配置文件"""
        config = {}
        
        if config_file:
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            except Exception as e:
                logging.warning(f"无法加载配置文件 {config_file}: {e}")
        
        # 默认配置
        default_config = {
            'symbol': 'BTC/USDT',
            'exchange': {
                'name': 'binance',
                'apiKey': '',
                'secret': '',
                'sandbox': True,
                'enableRateLimit': True,
            },
            'min_order_size': 0.001,
            'spread_ratio': 0.002,
            'price_update_threshold': 0.0005,
            'check_interval': 1.0,
            'position_imbalance_threshold': 0.1,
            'rebalance_urgency_multiplier': 2.0,
            'retreat_distance': 0.001,
            'max_daily_loss': 100.0,
            'position_limit': 1000.0,
        }
        
        # 合并配置
        for key, value in default_config.items():
            config.setdefault(key, value)
        
        config.update(kwargs)
        return config
    
    def _init_exchange(self, exchange_config: Dict) -> Union['ccxt.Exchange', CustomExchangeBase]:
        """初始化交易所实例"""
        try:
            exchange = create_exchange_instance(exchange_config)
            exchange_name = exchange_config.get('name', 'Unknown')
            exchange_type = exchange_manager.get_exchange_type(exchange_name)
            
            self.logger.info(f"成功创建{exchange_type}交易所实例: {exchange_name}")
            return exchange
            
        except Exception as e:
            self.logger.error(f"初始化交易所失败: {e}")
            raise RuntimeError(f"初始化交易所失败: {e}")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志系统"""
        exchange_name = self.config['exchange'].get('name', 'unknown')
        logger = logging.getLogger(f'UniversalMM_{exchange_name}_{self.symbol.replace("/", "_")}')
        logger.setLevel(logging.INFO)
        
        # 文件处理器
        log_file = f'universal_mm_{exchange_name}_{self.symbol.replace("/", "_")}_{datetime.now().strftime("%Y%m%d")}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 格式器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _load_market_info(self):
        """加载市场信息"""
        try:
            # 自定义交易所
            if self.exchange_type == 'custom':
                markets = self.exchange.load_markets()
            # ccxt交易所
            else:
                markets = self.exchange.load_markets()
            
            if self.symbol not in markets:
                available_symbols = list(markets.keys())[:10]
                raise ValueError(f"不支持的交易对: {self.symbol}。可用交易对: {available_symbols}...")
            
            market = markets[self.symbol]
            
            # 统一处理市场信息格式
            if self.exchange_type == 'custom':
                self.price_precision = market.get('precision', {}).get('price', 8)
                self.amount_precision = market.get('precision', {}).get('amount', 8)
                self.min_amount = market.get('limits', {}).get('amount', {}).get('min', 0.001)
                self.min_cost = market.get('limits', {}).get('cost', {}).get('min', 10.0)
            else:
                self.price_precision = market['precision']['price']
                self.amount_precision = market['precision']['amount']
                self.min_amount = market['limits']['amount']['min']
                self.min_cost = market['limits']['cost']['min'] if market['limits']['cost']['min'] else 10.0
            
            # 调整最小订单数量
            if self.min_order_size < self.min_amount:
                self.min_order_size = self.min_amount
            
            self.logger.info(f"市场信息加载完成 - {self.symbol} ({self.exchange_type})")
            self.logger.info(f"价格精度: {self.price_precision}, 数量精度: {self.amount_precision}")
            self.logger.info(f"最小数量: {self.min_amount}, 最小成本: {self.min_cost}")
            
        except Exception as e:
            self.logger.error(f"加载市场信息失败: {e}")
            raise
    
    def _round_price(self, price: float) -> float:
        """调整价格精度"""
        if self.price_precision is not None:
            return round(price, self.price_precision)
        return price
    
    def _round_amount(self, amount: float) -> float:
        """调整数量精度"""
        if self.amount_precision is not None:
            return round(amount, self.amount_precision)
        return amount
    
    def get_balance(self) -> Position:
        """获取账户余额 - 统一接口"""
        try:
            balance = self.exchange.fetch_balance()
            base_currency, quote_currency = self.symbol.split('/')
            
            # 统一处理余额格式
            if self.exchange_type == 'custom':
                base_info = balance.get(base_currency, {'free': 0.0, 'used': 0.0})
                quote_info = balance.get(quote_currency, {'free': 0.0, 'used': 0.0})
            else:
                base_info = balance.get(base_currency, {'free': 0.0, 'used': 0.0})
                quote_info = balance.get(quote_currency, {'free': 0.0, 'used': 0.0})
            
            position = Position(
                base_balance=base_info.get('free', 0.0),
                quote_balance=quote_info.get('free', 0.0),
                base_locked=base_info.get('used', 0.0),
                quote_locked=quote_info.get('used', 0.0),
            )
            
            return position
            
        except Exception as e:
            self.logger.error(f"获取余额失败: {e}")
            return self.position
    
    def get_orderbook(self) -> Optional[Dict]:
        """获取订单簿 - 统一接口"""
        try:
            orderbook = self.exchange.fetch_order_book(self.symbol, limit=10)
            self.last_orderbook_update = datetime.now()
            return orderbook
        except Exception as e:
            self.logger.error(f"获取订单簿失败: {e}")
            return None
    
    def calculate_position_imbalance(self, mid_price: float) -> PositionImbalance:
        """计算仓位失衡情况"""
        self.position = self.get_balance()
        
        # 计算各货币的价值
        base_value = self.position.base_balance * mid_price
        quote_value = self.position.quote_balance
        total_value = base_value + quote_value
        
        if total_value <= 0:
            return PositionImbalance()
        
        # 计算失衡程度
        target_base_value = total_value * 0.5
        base_deviation = abs(base_value - target_base_value) / total_value
        
        # 判断是否需要平衡
        if base_deviation > self.position_imbalance_threshold:
            if base_value > target_base_value:
                # 基础货币过多
                excess_amount = (base_value - target_base_value) / mid_price
                return PositionImbalance(
                    is_imbalanced=True,
                    excess_side='base',
                    excess_amount=excess_amount,
                    target_price=mid_price
                )
            else:
                # 计价货币过多
                excess_amount = (quote_value - target_base_value) / mid_price
                return PositionImbalance(
                    is_imbalanced=True,
                    excess_side='quote',
                    excess_amount=excess_amount,
                    target_price=mid_price
                )
        
        return PositionImbalance()
    
    def calculate_optimal_prices(self, orderbook: Dict, imbalance: PositionImbalance) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """计算最优挂单价格"""
        if not orderbook or not orderbook['bids'] or not orderbook['asks']:
            return None, None, None, None
        
        best_bid = orderbook['bids'][0][0]
        best_ask = orderbook['asks'][0][0]
        mid_price = (best_bid + best_ask) / 2
        self.current_mid_price = mid_price
        
        # 基础价差
        spread = mid_price * self.spread_ratio
        
        # 正常做市价格
        normal_bid = self._round_price(mid_price - spread / 2)
        normal_ask = self._round_price(mid_price + spread / 2)
        
        # 平衡订单价格
        rebalance_bid = None
        rebalance_ask = None
        
        if imbalance.is_imbalanced:
            if imbalance.excess_side == 'base':
                # 基础货币过多，优先卖出
                rebalance_ask = self._round_price(best_bid)
                retreat_distance = mid_price * self.retreat_distance
                normal_bid = self._round_price(normal_bid - retreat_distance)
            elif imbalance.excess_side == 'quote':
                # 计价货币过多，优先买入
                rebalance_bid = self._round_price(best_ask)
                retreat_distance = mid_price * self.retreat_distance
                normal_ask = self._round_price(normal_ask + retreat_distance)
        
        # 确保价格合理性
        if normal_bid >= best_ask:
            normal_bid = self._round_price(best_bid * 0.999)
        if normal_ask <= best_bid:
            normal_ask = self._round_price(best_ask * 1.001)
        
        return normal_bid, normal_ask, rebalance_bid, rebalance_ask
    
    def place_order(self, side: str, price: float, amount: float, order_type: OrderType = OrderType.NORMAL) -> Optional[OrderInfo]:
        """下单 - 统一接口"""
        try:
            price = self._round_price(price)
            amount = self._round_amount(amount)
            
            # 检查最小订单要求
            if amount < self.min_amount or price * amount < self.min_cost:
                self.logger.warning(f"订单不满足最小要求: {side} {amount} @ {price}")
                return None
            
            # 下单 (统一接口)
            if side == 'buy':
                if hasattr(self.exchange, 'create_limit_buy_order'):
                    order = self.exchange.create_limit_buy_order(self.symbol, amount, price)
                else:
                    order = self.exchange.create_order(self.symbol, 'limit', 'buy', amount, price)
            else:
                if hasattr(self.exchange, 'create_limit_sell_order'):
                    order = self.exchange.create_limit_sell_order(self.symbol, amount, price)
                else:
                    order = self.exchange.create_order(self.symbol, 'limit', 'sell', amount, price)
            
            # 创建订单信息
            order_info = OrderInfo(
                order_id=order['id'],
                side=side,
                price=price,
                amount=amount,
                timestamp=datetime.now(),
                order_type=order_type,
                status=OrderStatus.PENDING
            )
            
            # 记录订单
            self.active_orders[order['id']] = order_info
            self.stats['orders_placed'] += 1
            
            if order_type == OrderType.REBALANCE:
                self.stats['rebalance_orders'] += 1
                self.logger.info(f"下平衡单: ID={order['id']}, {side} {amount} @ {price}")
            else:
                self.stats['normal_orders'] += 1
                self.logger.info(f"下正常单: ID={order['id']}, {side} {amount} @ {price}")
            
            return order_info
            
        except Exception as e:
            self.logger.error(f"下单失败 {side} {amount} @ {price}: {e}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """撤销订单 - 统一接口"""
        try:
            self.exchange.cancel_order(order_id, self.symbol)
            
            if order_id in self.active_orders:
                order = self.active_orders[order_id]
                order.status = OrderStatus.CANCELLED
                order_type_desc = "平衡" if order.order_type == OrderType.REBALANCE else "正常"
                self.logger.info(f"撤销{order_type_desc}订单: {order_id}")
                del self.active_orders[order_id]
            
            return True
            
        except Exception as e:
            self.logger.error(f"撤销订单失败 {order_id}: {e}")
            return False
    
    def update_order_status(self):
        """更新订单状态 - 统一接口"""
        try:
            if not self.active_orders:
                return
            
            open_orders = self.exchange.fetch_open_orders(self.symbol)
            open_order_ids = {order['id'] for order in open_orders}
            
            for order_id, order_info in list(self.active_orders.items()):
                if order_id not in open_order_ids:
                    try:
                        order_detail = self.exchange.fetch_order(order_id, self.symbol)
                        if order_detail['status'] == 'closed' and order_detail.get('filled', 0) > 0:
                            order_info.status = OrderStatus.FILLED
                            self.stats['orders_filled'] += 1
                            self.stats['total_volume'] += order_detail.get('filled', 0)
                            
                            order_type_desc = "平衡" if order_info.order_type == OrderType.REBALANCE else "正常"
                            self.logger.info(f"{order_type_desc}订单成交: {order_id} {order_info.side} {order_detail.get('filled', 0)}")
                        else:
                            order_info.status = OrderStatus.CANCELLED
                    except:
                        order_info.status = OrderStatus.CANCELLED
                    
                    self.order_history.append(order_info)
                    del self.active_orders[order_id]
                    
        except Exception as e:
            self.logger.error(f"更新订单状态失败: {e}")
    
    def run_strategy_cycle(self):
        """运行一个策略周期"""
        try:
            # 更新订单状态
            self.update_order_status()
            
            # 获取订单簿
            orderbook = self.get_orderbook()
            if not orderbook:
                return
            
            # 计算中间价
            best_bid = orderbook['bids'][0][0] if orderbook['bids'] else None
            best_ask = orderbook['asks'][0][0] if orderbook['asks'] else None
            if not best_bid or not best_ask:
                return
            
            mid_price = (best_bid + best_ask) / 2
            
            # 检查仓位失衡
            imbalance = self.calculate_position_imbalance(mid_price)
            
            # 计算最优价格
            prices = self.calculate_optimal_prices(orderbook, imbalance)
            normal_bid, normal_ask, rebalance_bid, rebalance_ask = prices
            
            # 检查是否需要更新订单
            current_orders = len(self.active_orders)
            expected_orders = 2 + (1 if imbalance.is_imbalanced else 0)
            
            if current_orders != expected_orders or self._should_update_prices(prices):
                # 撤销所有旧订单
                for order_id in list(self.active_orders.keys()):
                    self.cancel_order(order_id)
                
                time.sleep(0.5)  # 等待撤单完成
                
                # 下新订单
                if normal_bid:
                    self.place_order('buy', normal_bid, self.min_order_size, OrderType.NORMAL)
                if normal_ask:
                    self.place_order('sell', normal_ask, self.min_order_size, OrderType.NORMAL)
                
                # 下平衡订单
                if imbalance.is_imbalanced:
                    rebalance_amount = min(
                        imbalance.excess_amount * 0.1,
                        self.min_order_size * self.rebalance_urgency_multiplier
                    )
                    rebalance_amount = max(rebalance_amount, self.min_order_size)
                    
                    if rebalance_bid:
                        self.place_order('buy', rebalance_bid, rebalance_amount, OrderType.REBALANCE)
                    if rebalance_ask:
                        self.place_order('sell', rebalance_ask, rebalance_amount, OrderType.REBALANCE)
            
        except Exception as e:
            self.logger.error(f"策略周期执行失败: {e}")
    
    def _should_update_prices(self, prices: Tuple) -> bool:
        """判断是否需要更新价格"""
        normal_bid, normal_ask, rebalance_bid, rebalance_ask = prices
        
        for order in self.active_orders.values():
            if order.order_type == OrderType.NORMAL:
                if order.side == 'buy' and normal_bid:
                    if abs(order.price - normal_bid) / order.price > self.price_update_threshold:
                        return True
                if order.side == 'sell' and normal_ask:
                    if abs(order.price - normal_ask) / order.price > self.price_update_threshold:
                        return True
        
        return False
    
    def print_stats(self):
        """打印统计信息"""
        runtime = datetime.now() - self.start_time
        
        # 计算仓位信息
        mid_price = self.current_mid_price
        base_value = self.position.base_balance * mid_price if mid_price > 0 else 0
        total_value = base_value + self.position.quote_balance
        base_ratio = (base_value / total_value * 100) if total_value > 0 else 0
        
        # 计算失衡情况
        imbalance = self.calculate_position_imbalance(mid_price) if mid_price > 0 else PositionImbalance()
        
        exchange_name = self.config['exchange'].get('name', 'Unknown')
        
        stats_info = (
            f"\n=== 通用做市商策略状态 ===\n"
            f"交易所: {exchange_name} ({self.exchange_type})\n"
            f"交易对: {self.symbol}\n"
            f"运行时间: {runtime}\n"
            f"中间价: {mid_price:.6f}\n"
            f"\n--- 订单统计 ---\n"
            f"总下单: {self.stats['orders_placed']}\n"
            f"总成交: {self.stats['orders_filled']}\n"
            f"正常单: {self.stats['normal_orders']}\n"
            f"平衡单: {self.stats['rebalance_orders']}\n"
            f"成交率: {self.stats['orders_filled']/max(1, self.stats['orders_placed'])*100:.1f}%\n"
            f"活跃订单: {len(self.active_orders)}\n"
            f"\n--- 仓位状态 ---\n"
            f"基础余额: {self.position.base_balance:.6f}\n"
            f"计价余额: {self.position.quote_balance:.2f}\n"
            f"基础价值占比: {base_ratio:.1f}%\n"
            f"仓位失衡: {'是' if imbalance.is_imbalanced else '否'}\n"
        )
        
        if imbalance.is_imbalanced:
            stats_info += (
                f"失衡方向: {'基础货币过多' if imbalance.excess_side == 'base' else '计价货币过多'}\n"
                f"失衡数量: {imbalance.excess_amount:.6f}\n"
            )
        
        stats_info += "==============================\n"
        
        print(stats_info)
        self.logger.info("统计信息已更新")
    
    def run(self):
        """运行通用做市商策略"""
        self.is_running = True
        self.logger.info(f"启动通用做市商策略 - {self.config['exchange'].get('name')} {self.symbol}")
        
        try:
            cycle_count = 0
            while self.is_running:
                self.run_strategy_cycle()
                
                cycle_count += 1
                if cycle_count % 10 == 0:
                    self.print_stats()
                
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            self.logger.info("收到停止信号")
        except Exception as e:
            self.logger.error(f"策略运行异常: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """停止策略"""
        self.is_running = False
        
        try:
            # 撤销所有活跃订单
            for order_id in list(self.active_orders.keys()):
                self.cancel_order(order_id)
            
            # 打印最终统计
            self.print_stats()
            
            self.logger.info("策略已安全停止")
            
        except Exception as e:
            self.logger.error(f"停止策略时出错: {e}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='通用做市商策略')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--exchange', type=str, help='交易所名称')
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='交易对')
    parser.add_argument('--list-exchanges', action='store_true', help='列出支持的交易所')
    
    args = parser.parse_args()
    
    # 列出支持的交易所
    if args.list_exchanges:
        exchange_manager.print_exchange_info()
        return
    
    # 创建策略实例
    try:
        kwargs = {}
        if args.exchange:
            kwargs['exchange'] = {'name': args.exchange}
        if args.symbol:
            kwargs['symbol'] = args.symbol
        
        strategy = UniversalMarketMaker(
            config_file=args.config,
            **kwargs
        )
        
        strategy.run()
        
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()