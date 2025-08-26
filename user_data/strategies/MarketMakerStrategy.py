# MarketMaker Strategy - 做市商策略
# 永远在买一卖一挂最小数量订单

import ccxt
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from decimal import Decimal, ROUND_DOWN
import asyncio


class MarketMakerStrategy:
    """
    做市商策略实现
    
    策略逻辑：
    1. 永远在买一卖一挂最小数量的限价单
    2. 当盘口价格变化时，撤销旧订单并重新挂单
    3. 如果订单被成交，立即挂新的订单
    4. 维持买卖双边的流动性提供
    """
    
    def __init__(self, exchange_config: dict, symbol: str, min_order_size: float = None):
        """
        初始化做市商策略
        
        Args:
            exchange_config: 交易所配置信息
            symbol: 交易对符号 (如 'BTC/USDT')
            min_order_size: 最小订单数量，如果为None则自动获取交易所限制
        """
        self.symbol = symbol
        self.exchange = self._init_exchange(exchange_config)
        self.min_order_size = min_order_size
        
        # 订单管理
        self.buy_order_id = None
        self.sell_order_id = None
        self.last_bid = None
        self.last_ask = None
        
        # 策略参数
        self.spread_buffer = 0.0001  # 价差缓冲，防止频繁撤单重挂
        self.check_interval = 0.5    # 检查间隔（秒）
        self.max_position = 0.1      # 最大持仓比例
        
        # 运行状态
        self.is_running = False
        self.logger = self._setup_logger()
        
        # 获取交易所规则
        self._load_market_info()
    
    def _init_exchange(self, config: dict) -> ccxt.Exchange:
        """初始化交易所连接"""
        exchange_class = getattr(ccxt, config['name'])
        return exchange_class({
            'apiKey': config.get('key', ''),
            'secret': config.get('secret', ''),
            'sandbox': config.get('sandbox', True),  # 默认使用测试环境
            'enableRateLimit': True,
        })
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger(f'MarketMaker_{self.symbol}')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def _load_market_info(self):
        """加载市场信息和交易规则"""
        try:
            markets = self.exchange.load_markets()
            market = markets[self.symbol]
            
            # 获取最小订单数量
            if self.min_order_size is None:
                self.min_order_size = market['limits']['amount']['min']
            
            # 价格和数量精度
            self.price_precision = market['precision']['price']
            self.amount_precision = market['precision']['amount']
            
            self.logger.info(f"已加载市场信息 - 最小订单: {self.min_order_size}, "
                           f"价格精度: {self.price_precision}, 数量精度: {self.amount_precision}")
            
        except Exception as e:
            self.logger.error(f"加载市场信息失败: {e}")
            raise
    
    def _round_price(self, price: float) -> float:
        """根据交易所精度要求调整价格"""
        if self.price_precision is not None:
            multiplier = 10 ** self.price_precision
            return float(int(price * multiplier) / multiplier)
        return price
    
    def _round_amount(self, amount: float) -> float:
        """根据交易所精度要求调整数量"""
        if self.amount_precision is not None:
            multiplier = 10 ** self.amount_precision
            return float(int(amount * multiplier) / multiplier)
        return amount
    
    def get_orderbook(self) -> Optional[Dict]:
        """获取订单簿"""
        try:
            orderbook = self.exchange.fetch_order_book(self.symbol, limit=5)
            return orderbook
        except Exception as e:
            self.logger.error(f"获取订单簿失败: {e}")
            return None
    
    def get_current_orders(self) -> Tuple[Optional[str], Optional[str]]:
        """获取当前挂单状态"""
        try:
            open_orders = self.exchange.fetch_open_orders(self.symbol)
            buy_order = None
            sell_order = None
            
            for order in open_orders:
                if order['side'] == 'buy':
                    buy_order = order['id']
                elif order['side'] == 'sell':
                    sell_order = order['id']
            
            return buy_order, sell_order
        
        except Exception as e:
            self.logger.error(f"获取挂单状态失败: {e}")
            return None, None
    
    def cancel_order(self, order_id: str) -> bool:
        """撤销订单"""
        try:
            self.exchange.cancel_order(order_id, self.symbol)
            self.logger.info(f"成功撤销订单: {order_id}")
            return True
        except Exception as e:
            self.logger.error(f"撤销订单失败 {order_id}: {e}")
            return False
    
    def place_buy_order(self, price: float, amount: float) -> Optional[str]:
        """下买单"""
        try:
            price = self._round_price(price)
            amount = self._round_amount(amount)
            
            order = self.exchange.create_limit_buy_order(self.symbol, amount, price)
            order_id = order['id']
            
            self.logger.info(f"成功下买单: ID={order_id}, 价格={price}, 数量={amount}")
            return order_id
        
        except Exception as e:
            self.logger.error(f"下买单失败: 价格={price}, 数量={amount}, 错误={e}")
            return None
    
    def place_sell_order(self, price: float, amount: float) -> Optional[str]:
        """下卖单"""
        try:
            price = self._round_price(price)
            amount = self._round_amount(amount)
            
            order = self.exchange.create_limit_sell_order(self.symbol, amount, price)
            order_id = order['id']
            
            self.logger.info(f"成功下卖单: ID={order_id}, 价格={price}, 数量={amount}")
            return order_id
        
        except Exception as e:
            self.logger.error(f"下卖单失败: 价格={price}, 数量={amount}, 错误={e}")
            return None
    
    def should_update_orders(self, current_bid: float, current_ask: float) -> bool:
        """判断是否需要更新订单"""
        if self.last_bid is None or self.last_ask is None:
            return True
        
        # 如果价格变化超过缓冲范围，则需要更新
        bid_change = abs(current_bid - self.last_bid) / self.last_bid
        ask_change = abs(current_ask - self.last_ask) / self.last_ask
        
        return bid_change > self.spread_buffer or ask_change > self.spread_buffer
    
    def update_orders(self, orderbook: Dict):
        """更新订单逻辑"""
        try:
            current_bid = orderbook['bids'][0][0] if orderbook['bids'] else None
            current_ask = orderbook['asks'][0][0] if orderbook['asks'] else None
            
            if not current_bid or not current_ask:
                self.logger.warning("无法获取有效的买一卖一价格")
                return
            
            # 检查是否需要更新订单
            if not self.should_update_orders(current_bid, current_ask):
                return
            
            # 获取当前挂单状态
            current_buy_order, current_sell_order = self.get_current_orders()
            
            # 撤销旧的买单
            if current_buy_order and current_buy_order != self.buy_order_id:
                self.cancel_order(current_buy_order)
                current_buy_order = None
            
            # 撤销旧的卖单  
            if current_sell_order and current_sell_order != self.sell_order_id:
                self.cancel_order(current_sell_order)
                current_sell_order = None
            
            # 挂新的买单（在当前买一价格）
            if not current_buy_order:
                self.buy_order_id = self.place_buy_order(current_bid, self.min_order_size)
            
            # 挂新的卖单（在当前卖一价格）
            if not current_sell_order:
                self.sell_order_id = self.place_sell_order(current_ask, self.min_order_size)
            
            # 更新记录的价格
            self.last_bid = current_bid
            self.last_ask = current_ask
            
        except Exception as e:
            self.logger.error(f"更新订单失败: {e}")
    
    def run_strategy(self):
        """运行做市商策略"""
        self.is_running = True
        self.logger.info(f"开始运行做市商策略 - 交易对: {self.symbol}")
        
        try:
            while self.is_running:
                # 获取订单簿
                orderbook = self.get_orderbook()
                if orderbook:
                    self.update_orders(orderbook)
                
                # 等待下一次检查
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            self.logger.info("收到停止信号，正在安全退出...")
            self.stop_strategy()
        except Exception as e:
            self.logger.error(f"策略运行出错: {e}")
            self.stop_strategy()
    
    def stop_strategy(self):
        """停止策略并清理资源"""
        self.is_running = False
        
        try:
            # 撤销所有挂单
            if self.buy_order_id:
                self.cancel_order(self.buy_order_id)
            if self.sell_order_id:
                self.cancel_order(self.sell_order_id)
            
            self.logger.info("策略已安全停止")
            
        except Exception as e:
            self.logger.error(f"停止策略时出错: {e}")


# 使用示例
def main():
    """主函数示例"""
    # 交易所配置（请填入真实的API密钥）
    exchange_config = {
        'name': 'binance',
        'key': 'your_api_key',
        'secret': 'your_api_secret',
        'sandbox': True,  # 使用测试环境
    }
    
    # 创建做市商策略实例
    strategy = MarketMakerStrategy(
        exchange_config=exchange_config,
        symbol='BTC/USDT',
        min_order_size=0.001  # 最小订单数量
    )
    
    # 运行策略
    try:
        strategy.run_strategy()
    except KeyboardInterrupt:
        print("程序被用户中断")
    finally:
        strategy.stop_strategy()


if __name__ == "__main__":
    main()