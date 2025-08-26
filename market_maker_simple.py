#!/usr/bin/env python3
"""
简单做市商策略 - 适合快速测试和理解
专注核心功能: 在买一卖一挂最小数量订单
"""

import ccxt
import time
import logging
from datetime import datetime
from typing import Optional, Tuple


class SimpleMarketMaker:
    """
    简单做市商策略
    
    核心逻辑:
    1. 获取买一卖一价格
    2. 在买一卖一各挂一个最小数量的订单
    3. 如果价格变化，撤销重新挂单
    4. 如果成交了，立即挂新单
    """
    
    def __init__(self, symbol: str = "BTC/USDT"):
        self.symbol = symbol
        self.exchange = None
        self.buy_order_id = None
        self.sell_order_id = None
        self.last_bid = None
        self.last_ask = None
        self.min_order_size = 0.001
        self.running = False
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_exchange(self, api_key: str = "", secret: str = "", sandbox: bool = True):
        """设置交易所连接"""
        try:
            self.exchange = ccxt.binance({
                'apiKey': api_key,
                'secret': secret,
                'sandbox': sandbox,
                'enableRateLimit': True,
            })
            
            # 加载市场信息
            self.exchange.load_markets()
            market = self.exchange.markets[self.symbol]
            self.min_order_size = max(market['limits']['amount']['min'], 0.001)
            
            self.logger.info(f"交易所设置完成 - {self.symbol}, 最小订单: {self.min_order_size}")
            return True
            
        except Exception as e:
            self.logger.error(f"设置交易所失败: {e}")
            return False
    
    def get_best_prices(self) -> Tuple[Optional[float], Optional[float]]:
        """获取买一卖一价格"""
        try:
            orderbook = self.exchange.fetch_order_book(self.symbol, limit=1)
            
            if orderbook['bids'] and orderbook['asks']:
                best_bid = orderbook['bids'][0][0]
                best_ask = orderbook['asks'][0][0]
                return best_bid, best_ask
            else:
                return None, None
                
        except Exception as e:
            self.logger.error(f"获取价格失败: {e}")
            return None, None
    
    def get_open_orders(self):
        """获取当前挂单"""
        try:
            orders = self.exchange.fetch_open_orders(self.symbol)
            buy_order = None
            sell_order = None
            
            for order in orders:
                if order['side'] == 'buy':
                    buy_order = order['id']
                elif order['side'] == 'sell':
                    sell_order = order['id']
            
            return buy_order, sell_order
            
        except Exception as e:
            self.logger.error(f"获取挂单失败: {e}")
            return None, None
    
    def cancel_order_safe(self, order_id: str):
        """安全撤单"""
        if not order_id:
            return
        
        try:
            self.exchange.cancel_order(order_id, self.symbol)
            self.logger.info(f"撤单成功: {order_id}")
        except Exception as e:
            self.logger.warning(f"撤单失败 {order_id}: {e}")
    
    def place_buy_order(self, price: float) -> Optional[str]:
        """下买单"""
        try:
            order = self.exchange.create_limit_buy_order(
                self.symbol, 
                self.min_order_size, 
                price
            )
            order_id = order['id']
            self.logger.info(f"买单成功: {order_id} @ {price}")
            return order_id
            
        except Exception as e:
            self.logger.error(f"下买单失败 @ {price}: {e}")
            return None
    
    def place_sell_order(self, price: float) -> Optional[str]:
        """下卖单"""
        try:
            order = self.exchange.create_limit_sell_order(
                self.symbol,
                self.min_order_size,
                price
            )
            order_id = order['id']
            self.logger.info(f"卖单成功: {order_id} @ {price}")
            return order_id
            
        except Exception as e:
            self.logger.error(f"下卖单失败 @ {price}: {e}")
            return None
    
    def should_update(self, current_bid: float, current_ask: float) -> bool:
        """判断是否需要更新订单"""
        if self.last_bid is None or self.last_ask is None:
            return True
        
        # 价格变化超过0.01%就更新
        bid_change = abs(current_bid - self.last_bid) / self.last_bid
        ask_change = abs(current_ask - self.last_ask) / self.last_ask
        
        return bid_change > 0.0001 or ask_change > 0.0001
    
    def run_cycle(self):
        """运行一个周期"""
        try:
            # 1. 获取当前买一卖一价格
            current_bid, current_ask = self.get_best_prices()
            if not current_bid or not current_ask:
                self.logger.warning("无法获取价格")
                return
            
            # 2. 检查是否需要更新订单
            if not self.should_update(current_bid, current_ask):
                return
            
            # 3. 获取当前挂单状态
            buy_order, sell_order = self.get_open_orders()
            
            # 4. 撤销旧订单
            if buy_order:
                self.cancel_order_safe(buy_order)
            if sell_order:
                self.cancel_order_safe(sell_order)
            
            # 等待撤单完成
            time.sleep(0.2)
            
            # 5. 挂新订单
            new_buy_order = self.place_buy_order(current_bid)
            new_sell_order = self.place_sell_order(current_ask)
            
            # 6. 更新记录
            if new_buy_order:
                self.buy_order_id = new_buy_order
            if new_sell_order:
                self.sell_order_id = new_sell_order
            
            self.last_bid = current_bid
            self.last_ask = current_ask
            
        except Exception as e:
            self.logger.error(f"运行周期出错: {e}")
    
    def run(self, check_interval: float = 1.0):
        """运行做市商策略"""
        if not self.exchange:
            self.logger.error("请先设置交易所连接")
            return
        
        self.running = True
        self.logger.info(f"开始运行简单做市商策略 - {self.symbol}")
        
        try:
            while self.running:
                self.run_cycle()
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            self.logger.info("收到停止信号")
        finally:
            self.stop()
    
    def stop(self):
        """停止策略"""
        self.running = False
        
        try:
            # 撤销所有挂单
            if self.buy_order_id:
                self.cancel_order_safe(self.buy_order_id)
            if self.sell_order_id:
                self.cancel_order_safe(self.sell_order_id)
            
            self.logger.info("策略已停止")
            
        except Exception as e:
            self.logger.error(f"停止策略出错: {e}")


def main():
    """示例使用"""
    print("=== 简单做市商策略 ===")
    print("这是一个简化版本，用于理解核心逻辑")
    print("实际使用请填入真实的API密钥")
    print("建议先在沙盒环境测试")
    print("按 Ctrl+C 停止")
    print("=" * 40)
    
    # 创建策略实例
    maker = SimpleMarketMaker("BTC/USDT")
    
    # 设置交易所(沙盒模式)
    success = maker.setup_exchange(
        api_key="your_api_key_here",     # 请填入真实API密钥
        secret="your_secret_here",       # 请填入真实API密钥
        sandbox=True                     # 使用沙盒模式
    )
    
    if not success:
        print("交易所设置失败，请检查API密钥")
        return
    
    try:
        # 运行策略
        maker.run(check_interval=1.0)  # 每秒检查一次
    except KeyboardInterrupt:
        print("\\n程序被用户中断")


if __name__ == "__main__":
    main()