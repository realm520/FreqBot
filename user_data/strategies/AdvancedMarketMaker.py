# Advanced Market Maker Strategy - 高级做市商策略
# 改进版本：增加风险控制、持仓管理和动态调整

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


@dataclass
class OrderInfo:
    """订单信息数据类"""
    order_id: str
    side: str  # 'buy' or 'sell'
    price: float
    amount: float
    timestamp: datetime
    status: OrderStatus = OrderStatus.PENDING


@dataclass
class Position:
    """持仓信息"""
    base_balance: float = 0.0
    quote_balance: float = 0.0
    base_locked: float = 0.0
    quote_locked: float = 0.0


class AdvancedMarketMaker:
    """
    高级做市商策略
    
    功能特点：
    1. 智能订单管理 - 避免频繁撤单重挂
    2. 动态价格调整 - 根据市场情况调整报价
    3. 风险控制 - 持仓限制和止损机制
    4. 盈亏统计 - 实时统计交易盈亏
    5. 异常处理 - 网络异常和API错误处理
    """
    
    def __init__(self, config_file: str = None, **kwargs):
        """
        初始化高级做市商策略
        
        Args:
            config_file: 配置文件路径
            **kwargs: 其他配置参数
        """
        # 加载配置
        self.config = self._load_config(config_file, **kwargs)
        
        # 基础设置
        self.symbol = self.config['symbol']
        self.exchange = self._init_exchange(self.config['exchange'])
        
        # 订单管理
        self.active_orders: Dict[str, OrderInfo] = {}
        self.order_history: List[OrderInfo] = []
        self.position = Position()
        
        # 策略参数
        self.min_order_size = self.config.get('min_order_size', 0.001)
        self.spread_ratio = self.config.get('spread_ratio', 0.002)  # 价差比例
        self.price_update_threshold = self.config.get('price_update_threshold', 0.0005)
        self.check_interval = self.config.get('check_interval', 1.0)
        self.max_position_ratio = self.config.get('max_position_ratio', 0.3)
        
        # 风险控制
        self.max_daily_loss = self.config.get('max_daily_loss', 100.0)  # 最大日亏损
        self.position_limit = self.config.get('position_limit', 1000.0)  # 持仓限制
        
        # 运行状态
        self.is_running = False
        self.last_orderbook_update = None
        self.daily_pnl = 0.0
        self.start_time = datetime.now()
        
        # 日志和监控
        self.logger = self._setup_logger()
        self.stats = {'orders_placed': 0, 'orders_filled': 0, 'total_volume': 0.0}
        
        # 初始化市场信息
        self._load_market_info()
        
        # 启动监控线程
        self.monitor_thread = None
    
    def _load_config(self, config_file: str, **kwargs) -> Dict:
        """加载配置文件"""
        config = {}
        
        # 从配置文件加载
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
            'max_position_ratio': 0.3,
            'max_daily_loss': 100.0,
            'position_limit': 1000.0,
        }
        
        # 合并配置
        for key, value in default_config.items():
            config.setdefault(key, value)
        
        # 命令行参数覆盖
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
                'timeout': 30000,  # 30秒超时
            })
            
            # 测试连接
            exchange.load_markets()
            return exchange
            
        except Exception as e:
            raise RuntimeError(f"初始化交易所失败: {e}")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志系统"""
        logger = logging.getLogger(f'AdvancedMarketMaker_{self.symbol.replace("/", "_")}')
        logger.setLevel(logging.INFO)
        
        # 文件处理器
        file_handler = logging.FileHandler(
            f'market_maker_{self.symbol.replace("/", "_")}_{datetime.now().strftime("%Y%m%d")}.log'
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
            
            # 交易规则
            self.price_precision = market['precision']['price']
            self.amount_precision = market['precision']['amount']
            self.min_amount = market['limits']['amount']['min']
            self.min_cost = market['limits']['cost']['min'] if market['limits']['cost']['min'] else 10.0
            
            # 调整最小订单数量
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
    
    def calculate_quote_prices(self, orderbook: Dict) -> Tuple[float, float]:
        """计算报价价格"""
        if not orderbook or not orderbook['bids'] or not orderbook['asks']:
            return None, None
        
        best_bid = orderbook['bids'][0][0]
        best_ask = orderbook['asks'][0][0]
        mid_price = (best_bid + best_ask) / 2
        
        # 计算价差
        spread = mid_price * self.spread_ratio
        
        # 我们的报价
        our_bid = self._round_price(mid_price - spread / 2)
        our_ask = self._round_price(mid_price + spread / 2)
        
        # 确保价格合理
        if our_bid >= best_ask:
            our_bid = self._round_price(best_bid * 0.999)
        if our_ask <= best_bid:
            our_ask = self._round_price(best_ask * 1.001)
        
        return our_bid, our_ask
    
    def should_update_orders(self, new_bid: float, new_ask: float) -> bool:
        """判断是否需要更新订单"""
        current_orders = [order for order in self.active_orders.values() 
                         if order.status == OrderStatus.PENDING]
        
        if len(current_orders) < 2:  # 没有足够的挂单
            return True
        
        # 检查价格变化
        for order in current_orders:
            if order.side == 'buy' and abs(order.price - new_bid) / order.price > self.price_update_threshold:
                return True
            if order.side == 'sell' and abs(order.price - new_ask) / order.price > self.price_update_threshold:
                return True
        
        return False
    
    def place_order(self, side: str, price: float, amount: float) -> Optional[OrderInfo]:
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
                status=OrderStatus.PENDING
            )
            
            # 记录订单
            self.active_orders[order['id']] = order_info
            self.stats['orders_placed'] += 1
            
            self.logger.info(f"成功下{side}单: ID={order['id']}, 价格={price}, 数量={amount}")
            return order_info
            
        except Exception as e:
            self.logger.error(f"下单失败 {side} {amount} @ {price}: {e}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """撤销订单"""
        try:
            self.exchange.cancel_order(order_id, self.symbol)
            
            if order_id in self.active_orders:
                self.active_orders[order_id].status = OrderStatus.CANCELLED
                del self.active_orders[order_id]
            
            self.logger.info(f"成功撤销订单: {order_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"撤销订单失败 {order_id}: {e}")
            return False
    
    def update_order_status(self):
        """更新订单状态"""
        try:
            if not self.active_orders:
                return
            
            # 获取所有挂单状态
            open_orders = self.exchange.fetch_open_orders(self.symbol)
            open_order_ids = {order['id'] for order in open_orders}
            
            # 检查已完成的订单
            for order_id, order_info in list(self.active_orders.items()):
                if order_id not in open_order_ids:
                    # 订单已完成或被撤销
                    try:
                        order_detail = self.exchange.fetch_order(order_id, self.symbol)
                        if order_detail['status'] == 'closed' and order_detail['filled'] > 0:
                            order_info.status = OrderStatus.FILLED
                            self.stats['orders_filled'] += 1
                            self.stats['total_volume'] += order_detail['filled']
                            self.logger.info(f"订单已成交: {order_id} {order_info.side} {order_detail['filled']}")
                        else:
                            order_info.status = OrderStatus.CANCELLED
                    except:
                        order_info.status = OrderStatus.CANCELLED
                    
                    # 移动到历史记录
                    self.order_history.append(order_info)
                    del self.active_orders[order_id]
                    
        except Exception as e:
            self.logger.error(f"更新订单状态失败: {e}")
    
    def check_risk_limits(self) -> bool:
        """检查风险限制"""
        # 更新持仓
        self.position = self.get_balance()
        
        # 检查日亏损限制
        if self.daily_pnl < -self.max_daily_loss:
            self.logger.warning(f"达到日亏损限制: {self.daily_pnl}")
            return False
        
        # 检查持仓限制
        base_value = self.position.base_balance + self.position.base_locked
        if base_value > self.position_limit:
            self.logger.warning(f"达到持仓限制: {base_value}")
            return False
        
        return True
    
    def run_strategy_cycle(self):
        """运行一个策略周期"""
        try:
            # 检查风险限制
            if not self.check_risk_limits():
                self.logger.warning("风险检查失败，暂停交易")
                return
            
            # 更新订单状态
            self.update_order_status()
            
            # 获取订单簿
            orderbook = self.get_orderbook()
            if not orderbook:
                return
            
            # 计算报价
            bid_price, ask_price = self.calculate_quote_prices(orderbook)
            if not bid_price or not ask_price:
                return
            
            # 检查是否需要更新订单
            if self.should_update_orders(bid_price, ask_price):
                # 撤销旧订单
                for order_id in list(self.active_orders.keys()):
                    self.cancel_order(order_id)
                
                time.sleep(0.5)  # 等待撤单完成
                
                # 下新订单
                self.place_order('buy', bid_price, self.min_order_size)
                self.place_order('sell', ask_price, self.min_order_size)
            
        except Exception as e:
            self.logger.error(f"策略周期执行失败: {e}")
    
    def print_stats(self):
        """打印统计信息"""
        runtime = datetime.now() - self.start_time
        
        stats_info = (
            f"\n=== 做市商策略运行状态 ===\n"
            f"交易对: {self.symbol}\n"
            f"运行时间: {runtime}\n"
            f"下单数量: {self.stats['orders_placed']}\n"
            f"成交数量: {self.stats['orders_filled']}\n"
            f"成交率: {self.stats['orders_filled']/max(1, self.stats['orders_placed'])*100:.1f}%\n"
            f"交易量: {self.stats['total_volume']:.6f}\n"
            f"活跃订单: {len(self.active_orders)}\n"
            f"当日盈亏: {self.daily_pnl:.4f}\n"
            f"基础余额: {self.position.base_balance:.6f}\n"
            f"计价余额: {self.position.quote_balance:.2f}\n"
            f"=========================="
        )
        
        print(stats_info)
        self.logger.info("统计信息已更新")
    
    def run(self):
        """运行做市商策略"""
        self.is_running = True
        self.logger.info(f"启动高级做市商策略 - {self.symbol}")
        
        # 启动监控线程
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        try:
            cycle_count = 0
            while self.is_running:
                self.run_strategy_cycle()
                
                # 每10个周期打印一次统计
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
    
    def _monitor_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                # 检查连接状态
                if self.last_orderbook_update:
                    time_since_update = datetime.now() - self.last_orderbook_update
                    if time_since_update > timedelta(minutes=2):
                        self.logger.warning("订单簿更新超时，可能存在连接问题")
                
                time.sleep(30)  # 每30秒检查一次
            except Exception as e:
                self.logger.error(f"监控循环异常: {e}")
    
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


# 启动脚本
def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='高级做市商策略')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='交易对')
    parser.add_argument('--sandbox', action='store_true', help='使用沙盒环境')
    
    args = parser.parse_args()
    
    # 创建策略实例
    strategy = AdvancedMarketMaker(
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