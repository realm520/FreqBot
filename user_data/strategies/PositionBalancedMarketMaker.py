#!/usr/bin/env python3
"""
仓位平衡做市商策略
增加功能：
1. 仓位失衡时，将多余仓位挂到反向买卖一
2. 同方向订单往后退避让反向平仓订单
3. 动态调整挂单价格和数量
"""

import ccxt
import time
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal, ROUND_DOWN
import threading
from dataclasses import dataclass
from enum import Enum


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
    base_balance: float = 0.0  # 基础货币余额 (如BTC)
    quote_balance: float = 0.0  # 计价货币余额 (如USDT)
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


class PositionBalancedMarketMaker:
    """
    仓位平衡做市商策略
    
    核心功能：
    1. 正常做市：在买一卖一挂最小数量订单
    2. 仓位监控：实时监控基础货币和计价货币的平衡
    3. 失衡处理：仓位失衡时优先平衡仓位
    4. 智能避让：同方向订单为反向平仓订单让路
    """
    
    def __init__(self, config_file: str = None, **kwargs):
        # 加载配置
        self.config = self._load_config(config_file, **kwargs)
        
        # 基础设置
        self.symbol = self.config['symbol']
        self.exchange = self._init_exchange(self.config['exchange'])
        
        # 订单管理
        self.active_orders: Dict[str, OrderInfo] = {}
        self.order_history: List[OrderInfo] = []
        self.position = Position()
        
        # 基础策略参数
        self.min_order_size = self.config.get('min_order_size', 0.001)
        self.spread_ratio = self.config.get('spread_ratio', 0.002)
        self.price_update_threshold = self.config.get('price_update_threshold', 0.0005)
        self.check_interval = self.config.get('check_interval', 1.0)
        
        # 仓位平衡参数
        self.position_imbalance_threshold = self.config.get('position_imbalance_threshold', 0.1)  # 10%失衡触发
        self.max_position_value_ratio = self.config.get('max_position_value_ratio', 0.8)  # 最大持仓价值比例
        self.rebalance_urgency_multiplier = self.config.get('rebalance_urgency_multiplier', 2.0)  # 平衡订单数量倍数
        self.retreat_distance = self.config.get('retreat_distance', 0.001)  # 退避距离 (0.1%)
        
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
        
        # 初始化
        self._load_market_info()
        self.monitor_thread = None
    
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
            # 仓位平衡参数
            'position_imbalance_threshold': 0.1,
            'max_position_value_ratio': 0.8,
            'rebalance_urgency_multiplier': 2.0,
            'retreat_distance': 0.001,
            # 风险控制
            'max_daily_loss': 100.0,
            'position_limit': 1000.0,
        }
        
        # 合并配置
        for key, value in default_config.items():
            config.setdefault(key, value)
        
        config.update(kwargs)
        return config
    
    def _init_exchange(self, exchange_config: Dict) -> ccxt.Exchange:
        """初始化交易所连接"""
        try:
            exchange_class = getattr(ccxt, exchange_config['name'])
            exchange = exchange_class({
                'apiKey': exchange_config.get('apiKey', ''),
                'secret': exchange_config.get('secret', ''),
                'sandbox': exchange_config.get('sandbox', True),
                'enableRateLimit': exchange_config.get('enableRateLimit', True),
                'timeout': 30000,
            })
            
            exchange.load_markets()
            return exchange
            
        except Exception as e:
            raise RuntimeError(f"初始化交易所失败: {e}")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志系统"""
        logger = logging.getLogger(f'PositionBalancedMM_{self.symbol.replace("/", "_")}')
        logger.setLevel(logging.INFO)
        
        # 文件处理器
        file_handler = logging.FileHandler(
            f'position_balanced_mm_{self.symbol.replace("/", "_")}_{datetime.now().strftime("%Y%m%d")}.log'
        )
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
            markets = self.exchange.load_markets()
            market = markets[self.symbol]
            
            self.price_precision = market['precision']['price']
            self.amount_precision = market['precision']['amount']
            self.min_amount = market['limits']['amount']['min']
            self.min_cost = market['limits']['cost']['min'] if market['limits']['cost']['min'] else 10.0
            
            if self.min_order_size < self.min_amount:
                self.min_order_size = self.min_amount
            
            self.logger.info(f"市场信息加载完成 - {self.symbol}")
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
        """获取账户余额"""
        try:
            balance = self.exchange.fetch_balance()
            base_currency, quote_currency = self.symbol.split('/')
            
            position = Position(
                base_balance=balance.get(base_currency, {}).get('free', 0.0),
                quote_balance=balance.get(quote_currency, {}).get('free', 0.0),
                base_locked=balance.get(base_currency, {}).get('used', 0.0),
                quote_locked=balance.get(quote_currency, {}).get('used', 0.0),
            )
            
            return position
            
        except Exception as e:
            self.logger.error(f"获取余额失败: {e}")
            return self.position
    
    def get_orderbook(self) -> Optional[Dict]:
        """获取订单簿"""
        try:
            orderbook = self.exchange.fetch_order_book(self.symbol, limit=10)
            self.last_orderbook_update = datetime.now()
            return orderbook
        except Exception as e:
            self.logger.error(f"获取订单簿失败: {e}")
            return None
    
    def calculate_position_imbalance(self, mid_price: float) -> PositionImbalance:
        """计算仓位失衡情况"""
        # 更新持仓
        self.position = self.get_balance()
        
        # 计算各货币的价值 (以计价货币为准)
        base_value = self.position.base_balance * mid_price
        quote_value = self.position.quote_balance
        total_value = base_value + quote_value
        
        if total_value <= 0:
            return PositionImbalance()
        
        # 计算理想平衡点 (50:50)
        target_base_value = total_value * 0.5
        target_quote_value = total_value * 0.5
        
        # 计算失衡程度
        base_deviation = abs(base_value - target_base_value) / total_value
        quote_deviation = abs(quote_value - target_quote_value) / total_value
        
        max_deviation = max(base_deviation, quote_deviation)
        
        # 判断是否需要平衡
        if max_deviation > self.position_imbalance_threshold:
            if base_value > target_base_value:
                # 基础货币过多，需要卖出
                excess_amount = (base_value - target_base_value) / mid_price
                return PositionImbalance(
                    is_imbalanced=True,
                    excess_side='base',
                    excess_amount=excess_amount,
                    target_price=mid_price  # 在买一价格卖出
                )
            else:
                # 计价货币过多，需要买入
                excess_amount = (quote_value - target_quote_value) / mid_price
                return PositionImbalance(
                    is_imbalanced=True,
                    excess_side='quote',
                    excess_amount=excess_amount,
                    target_price=mid_price  # 在卖一价格买入
                )
        
        return PositionImbalance()
    
    def calculate_optimal_prices(self, orderbook: Dict, imbalance: PositionImbalance) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """
        计算最优挂单价格
        
        Returns:
            normal_bid, normal_ask, rebalance_bid, rebalance_ask
        """
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
                # 基础货币过多，优先在买一价格卖出
                rebalance_ask = self._round_price(best_bid)
                # 同方向的正常买单后退
                retreat_distance = mid_price * self.retreat_distance
                normal_bid = self._round_price(normal_bid - retreat_distance)
                
            elif imbalance.excess_side == 'quote':
                # 计价货币过多，优先在卖一价格买入
                rebalance_bid = self._round_price(best_ask)
                # 同方向的正常卖单后退
                retreat_distance = mid_price * self.retreat_distance
                normal_ask = self._round_price(normal_ask + retreat_distance)
        
        # 确保价格合理性
        if normal_bid >= best_ask:
            normal_bid = self._round_price(best_bid * 0.999)
        if normal_ask <= best_bid:
            normal_ask = self._round_price(best_ask * 1.001)
        
        return normal_bid, normal_ask, rebalance_bid, rebalance_ask
    
    def calculate_order_amounts(self, imbalance: PositionImbalance) -> Tuple[float, float]:
        """
        计算订单数量
        
        Returns:
            normal_amount, rebalance_amount
        """
        normal_amount = self.min_order_size
        
        if imbalance.is_imbalanced:
            # 平衡订单数量 = 失衡数量的一部分
            rebalance_amount = min(
                imbalance.excess_amount * 0.1,  # 每次平衡10%
                self.min_order_size * self.rebalance_urgency_multiplier
            )
            rebalance_amount = max(rebalance_amount, self.min_order_size)
            rebalance_amount = self._round_amount(rebalance_amount)
        else:
            rebalance_amount = 0.0
        
        return normal_amount, rebalance_amount
    
    def should_update_orders(self, new_prices: Tuple, current_imbalance: PositionImbalance) -> bool:
        """判断是否需要更新订单"""
        normal_bid, normal_ask, rebalance_bid, rebalance_ask = new_prices
        
        # 获取当前订单
        current_orders = [order for order in self.active_orders.values() 
                         if order.status == OrderStatus.PENDING]
        
        # 如果订单数量不足，需要更新
        expected_orders = 2  # 正常买卖各一单
        if current_imbalance.is_imbalanced:
            expected_orders += 1  # 加上平衡订单
        
        if len(current_orders) < expected_orders:
            return True
        
        # 检查价格变化
        for order in current_orders:
            if order.order_type == OrderType.NORMAL:
                if order.side == 'buy' and normal_bid:
                    if abs(order.price - normal_bid) / order.price > self.price_update_threshold:
                        return True
                if order.side == 'sell' and normal_ask:
                    if abs(order.price - normal_ask) / order.price > self.price_update_threshold:
                        return True
            elif order.order_type == OrderType.REBALANCE:
                # 平衡订单需要更频繁更新
                target_price = rebalance_bid if order.side == 'buy' else rebalance_ask
                if target_price and abs(order.price - target_price) / order.price > 0.0001:
                    return True
        
        return False
    
    def place_order(self, side: str, price: float, amount: float, order_type: OrderType = OrderType.NORMAL) -> Optional[OrderInfo]:
        """下单"""
        try:
            price = self._round_price(price)
            amount = self._round_amount(amount)
            
            # 检查最小订单要求
            if amount < self.min_amount or price * amount < self.min_cost:
                self.logger.warning(f"订单不满足最小要求: {side} {amount} @ {price}")
                return None
            
            # 下单
            if side == 'buy':
                order = self.exchange.create_limit_buy_order(self.symbol, amount, price)
            else:
                order = self.exchange.create_limit_sell_order(self.symbol, amount, price)
            
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
                self.logger.info(f"下平衡单: ID={order['id']}, {side} {amount} @ {price} [仓位平衡]")
            else:
                self.stats['normal_orders'] += 1
                self.logger.info(f"下正常单: ID={order['id']}, {side} {amount} @ {price}")
            
            return order_info
            
        except Exception as e:
            self.logger.error(f"下单失败 {side} {amount} @ {price}: {e}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """撤销订单"""
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
        """更新订单状态"""
        try:
            if not self.active_orders:
                return
            
            open_orders = self.exchange.fetch_open_orders(self.symbol)
            open_order_ids = {order['id'] for order in open_orders}
            
            for order_id, order_info in list(self.active_orders.items()):
                if order_id not in open_order_ids:
                    try:
                        order_detail = self.exchange.fetch_order(order_id, self.symbol)
                        if order_detail['status'] == 'closed' and order_detail['filled'] > 0:
                            order_info.status = OrderStatus.FILLED
                            self.stats['orders_filled'] += 1
                            self.stats['total_volume'] += order_detail['filled']
                            
                            order_type_desc = "平衡" if order_info.order_type == OrderType.REBALANCE else "正常"
                            self.logger.info(f"{order_type_desc}订单成交: {order_id} {order_info.side} {order_detail['filled']}")
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
            if self.should_update_orders(prices, imbalance):
                # 撤销所有旧订单
                for order_id in list(self.active_orders.keys()):
                    self.cancel_order(order_id)
                
                time.sleep(0.5)  # 等待撤单完成
                
                # 计算订单数量
                normal_amount, rebalance_amount = self.calculate_order_amounts(imbalance)
                
                # 下新订单
                if normal_bid:
                    self.place_order('buy', normal_bid, normal_amount, OrderType.NORMAL)
                if normal_ask:
                    self.place_order('sell', normal_ask, normal_amount, OrderType.NORMAL)
                
                # 下平衡订单
                if imbalance.is_imbalanced:
                    if rebalance_bid:
                        self.place_order('buy', rebalance_bid, rebalance_amount, OrderType.REBALANCE)
                    if rebalance_ask:
                        self.place_order('sell', rebalance_ask, rebalance_amount, OrderType.REBALANCE)
            
        except Exception as e:
            self.logger.error(f"策略周期执行失败: {e}")
    
    def print_stats(self):
        """打印统计信息"""
        runtime = datetime.now() - self.start_time
        
        # 计算仓位信息
        mid_price = self.current_mid_price
        base_value = self.position.base_balance * mid_price if mid_price > 0 else 0
        total_value = base_value + self.position.quote_balance
        base_ratio = (base_value / total_value * 100) if total_value > 0 else 0
        quote_ratio = 100 - base_ratio
        
        # 计算失衡情况
        imbalance = self.calculate_position_imbalance(mid_price) if mid_price > 0 else PositionImbalance()
        
        stats_info = (
            f"\n=== 仓位平衡做市商策略状态 ===\n"
            f"交易对: {self.symbol}\n"
            f"运行时间: {runtime}\n"
            f"中间价: {mid_price:.6f}\n"
            f"\n--- 订单统计 ---\n"
            f"总下单: {self.stats['orders_placed']}\n"
            f"总成交: {self.stats['orders_filled']}\n"
            f"正常单: {self.stats['normal_orders']}\n"
            f"平衡单: {self.stats['rebalance_orders']}\n"
            f"成交率: {self.stats['orders_filled']/max(1, self.stats['orders_placed'])*100:.1f}%\n"
            f"交易量: {self.stats['total_volume']:.6f}\n"
            f"活跃订单: {len(self.active_orders)}\n"
            f"\n--- 仓位状态 ---\n"
            f"基础余额: {self.position.base_balance:.6f}\n"
            f"计价余额: {self.position.quote_balance:.2f}\n"
            f"基础价值占比: {base_ratio:.1f}%\n"
            f"计价价值占比: {quote_ratio:.1f}%\n"
            f"仓位失衡: {'是' if imbalance.is_imbalanced else '否'}\n"
        )
        
        if imbalance.is_imbalanced:
            stats_info += (
                f"失衡方向: {'基础货币过多' if imbalance.excess_side == 'base' else '计价货币过多'}\n"
                f"失衡数量: {imbalance.excess_amount:.6f}\n"
            )
        
        stats_info += f"===============================\n"
        
        print(stats_info)
        self.logger.info("统计信息已更新")
    
    def run(self):
        """运行仓位平衡做市商策略"""
        self.is_running = True
        self.logger.info(f"启动仓位平衡做市商策略 - {self.symbol}")
        
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
    
    parser = argparse.ArgumentParser(description='仓位平衡做市商策略')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='交易对')
    parser.add_argument('--sandbox', action='store_true', help='使用沙盒环境')
    
    args = parser.parse_args()
    
    # 创建策略实例
    strategy = PositionBalancedMarketMaker(
        config_file=args.config,
        symbol=args.symbol,
        exchange={'sandbox': args.sandbox} if args.sandbox else {}
    )
    
    try:
        strategy.run()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    finally:
        strategy.stop()


if __name__ == "__main__":
    main()