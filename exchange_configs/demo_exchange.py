#!/usr/bin/env python3
"""
演示交易所实现
展示如何实现一个自定义交易所
"""

from custom_exchange_base import CustomExchangeBase, OrderSide, OrderType, OrderStatus
from typing import Dict, List, Optional, Any
import hmac
import hashlib
import time
import random
import json
from datetime import datetime


class DemoExchange(CustomExchangeBase):
    """
    演示交易所 - 模拟交易所实现
    
    用于测试和演示自定义交易所的完整实现
    包含模拟的订单簿、余额、交易等功能
    """
    
    def __init__(self, config: Dict[str, Any]):
        # 设置默认配置
        default_config = {
            'base_url': 'https://api.demo-exchange.com',
            'sandbox': True,
            'timeout': 30
        }
        default_config.update(config)
        
        super().__init__(default_config)
        
        # 模拟数据
        self._mock_balances = {
            'BTC': {'free': 1.0, 'used': 0.0},
            'USDT': {'free': 50000.0, 'used': 0.0},
            'ETH': {'free': 10.0, 'used': 0.0}
        }
        
        # 模拟订单簿数据
        self._mock_orderbook = {
            'BTC/USDT': {
                'bids': [[65000.0, 0.1], [64950.0, 0.2], [64900.0, 0.3]],
                'asks': [[65050.0, 0.1], [65100.0, 0.2], [65150.0, 0.3]],
                'timestamp': int(time.time() * 1000)
            },
            'ETH/USDT': {
                'bids': [[2500.0, 1.0], [2495.0, 2.0], [2490.0, 3.0]],
                'asks': [[2505.0, 1.0], [2510.0, 2.0], [2515.0, 3.0]],
                'timestamp': int(time.time() * 1000)
            }
        }
        
        # 模拟未成交订单
        self._mock_open_orders = {}
        self._order_counter = 1000
        
        self.logger.info("DemoExchange 初始化完成 - 这是一个模拟交易所")
    
    def _generate_signature(self, method: str, path: str, params: Dict = None, body: str = "") -> str:
        """生成模拟签名"""
        timestamp = self._get_timestamp()
        
        # 构建签名字符串
        if params:
            query_string = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
            message = f"{timestamp}{method.upper()}{path}?{query_string}{body}"
        else:
            message = f"{timestamp}{method.upper()}{path}{body}"
        
        # 生成HMAC-SHA256签名
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def _get_timestamp(self) -> str:
        """获取毫秒时间戳"""
        return str(int(time.time() * 1000))
    
    def _get_auth_headers(self, signature: str) -> Dict[str, str]:
        """获取认证头"""
        timestamp = self._get_timestamp()
        return {
            'X-DEMO-API-KEY': self.api_key,
            'X-DEMO-TIMESTAMP': timestamp,
            'X-DEMO-SIGNATURE': signature,
            'Content-Type': 'application/json'
        }
    
    def _mock_api_call(self, method: str, path: str, params: Dict = None, body: Dict = None):
        """模拟API调用 - 不实际发起网络请求"""
        # 模拟网络延迟
        time.sleep(0.01 + random.uniform(0, 0.05))
        
        # 根据路径返回模拟数据
        if 'symbols' in path:
            return self._get_mock_symbols()
        elif 'ticker' in path:
            symbol = params.get('symbol', 'BTCUSDT') if params else 'BTCUSDT'
            return self._get_mock_ticker(symbol)
        elif 'depth' in path:
            symbol = params.get('symbol', 'BTCUSDT') if params else 'BTCUSDT'
            return self._get_mock_depth(symbol)
        elif 'account' in path:
            return self._get_mock_account()
        elif 'orders' in path and method == 'POST':
            return self._mock_create_order(body)
        elif 'orders' in path and method == 'DELETE':
            order_id = path.split('/')[-1]
            return self._mock_cancel_order(order_id)
        elif 'orders' in path and 'open' in path:
            return self._get_mock_open_orders()
        elif 'orders' in path and method == 'GET':
            order_id = path.split('/')[-1]
            return self._get_mock_order(order_id)
        else:
            return {'success': True, 'message': f'Mock response for {method} {path}'}
    
    def _get_mock_symbols(self):
        """获取模拟交易对信息"""
        return {
            'code': 200,
            'data': [
                {
                    'symbol': 'BTCUSDT',
                    'base_currency': 'BTC',
                    'quote_currency': 'USDT',
                    'price_precision': 2,
                    'amount_precision': 6,
                    'min_amount': 0.001,
                    'min_cost': 10.0,
                    'status': 'trading'
                },
                {
                    'symbol': 'ETHUSDT',
                    'base_currency': 'ETH',
                    'quote_currency': 'USDT',
                    'price_precision': 2,
                    'amount_precision': 6,
                    'min_amount': 0.01,
                    'min_cost': 10.0,
                    'status': 'trading'
                }
            ]
        }
    
    def _get_mock_ticker(self, symbol: str):
        """获取模拟行情数据"""
        if symbol == 'BTCUSDT':
            base_price = 65000.0
        elif symbol == 'ETHUSDT':
            base_price = 2500.0
        else:
            base_price = 100.0
        
        # 添加随机波动
        price_change = random.uniform(-0.02, 0.02)
        current_price = base_price * (1 + price_change)
        
        return {
            'code': 200,
            'data': {
                'symbol': symbol,
                'price': round(current_price, 2),
                'bid': round(current_price * 0.999, 2),
                'ask': round(current_price * 1.001, 2),
                'high': round(current_price * 1.05, 2),
                'low': round(current_price * 0.95, 2),
                'volume': round(random.uniform(100, 1000), 2),
                'timestamp': int(time.time() * 1000)
            }
        }
    
    def _get_mock_depth(self, symbol: str):
        """获取模拟订单簿"""
        formatted_symbol = symbol.replace('USDT', '/USDT').replace('BTC', 'BTC').replace('ETH', 'ETH')
        
        if formatted_symbol in self._mock_orderbook:
            data = self._mock_orderbook[formatted_symbol].copy()
            
            # 添加随机波动
            for i, (price, amount) in enumerate(data['bids']):
                data['bids'][i][0] = round(price * (1 + random.uniform(-0.001, 0.001)), 2)
            
            for i, (price, amount) in enumerate(data['asks']):
                data['asks'][i][0] = round(price * (1 + random.uniform(-0.001, 0.001)), 2)
            
            data['timestamp'] = int(time.time() * 1000)
            
            return {'code': 200, 'data': data}
        else:
            # 生成默认订书簿
            mid_price = 1000.0
            return {
                'code': 200,
                'data': {
                    'bids': [[mid_price * 0.999, 1.0], [mid_price * 0.998, 2.0]],
                    'asks': [[mid_price * 1.001, 1.0], [mid_price * 1.002, 2.0]],
                    'timestamp': int(time.time() * 1000)
                }
            }
    
    def _get_mock_account(self):
        """获取模拟账户信息"""
        balances = []
        for currency, balance in self._mock_balances.items():
            balances.append({
                'currency': currency,
                'available': balance['free'],
                'frozen': balance['used'],
                'total': balance['free'] + balance['used']
            })
        
        return {
            'code': 200,
            'data': {
                'balances': balances
            }
        }
    
    def _mock_create_order(self, body: Dict):
        """模拟创建订单"""
        order_id = f"demo_{self._order_counter}"
        self._order_counter += 1
        
        # 保存到模拟未成交订单
        self._mock_open_orders[order_id] = {
            'order_id': order_id,
            'symbol': body.get('symbol', ''),
            'side': body.get('side', ''),
            'type': body.get('type', ''),
            'amount': body.get('amount', ''),
            'price': body.get('price', ''),
            'filled_amount': '0',
            'status': 'open',
            'created_at': int(time.time() * 1000)
        }
        
        return {
            'code': 200,
            'data': {
                'order_id': order_id,
                'status': 'open',
                'created_at': int(time.time() * 1000)
            }
        }
    
    def _mock_cancel_order(self, order_id: str):
        """模拟取消订单"""
        if order_id in self._mock_open_orders:
            self._mock_open_orders[order_id]['status'] = 'canceled'
            del self._mock_open_orders[order_id]
        
        return {
            'code': 200,
            'data': {
                'order_id': order_id,
                'status': 'canceled'
            }
        }
    
    def _get_mock_open_orders(self):
        """获取模拟未成交订单"""
        orders = list(self._mock_open_orders.values())
        return {
            'code': 200,
            'data': {
                'orders': orders
            }
        }
    
    def _get_mock_order(self, order_id: str):
        """获取模拟订单详情"""
        if order_id in self._mock_open_orders:
            order_data = self._mock_open_orders[order_id]
        else:
            # 模拟已完成订单
            order_data = {
                'order_id': order_id,
                'symbol': 'BTCUSDT',
                'side': 'buy',
                'type': 'limit',
                'amount': '0.001',
                'price': '65000',
                'filled_amount': '0.001',
                'status': 'closed',
                'created_at': int(time.time() * 1000)
            }
        
        return {
            'code': 200,
            'data': order_data
        }
    
    # 实现抽象方法
    def load_markets(self) -> Dict:
        """加载市场信息"""
        try:
            self.logger.info("加载DemoExchange市场信息...")
            
            # 模拟API调用
            data = self._mock_api_call('GET', '/api/v1/symbols')
            
            # 解析市场信息
            markets = {}
            for item in data['data']:
                symbol_id = item['symbol']
                symbol = f"{item['base_currency']}/{item['quote_currency']}"
                
                markets[symbol] = {
                    'id': symbol_id,
                    'symbol': symbol,
                    'base': item['base_currency'],
                    'quote': item['quote_currency'],
                    'precision': {
                        'price': item['price_precision'],
                        'amount': item['amount_precision']
                    },
                    'limits': {
                        'amount': {
                            'min': item['min_amount']
                        },
                        'cost': {
                            'min': item['min_cost']
                        }
                    },
                    'active': item['status'] == 'trading'
                }
            
            self.markets = markets
            self.logger.info(f"DemoExchange加载了 {len(markets)} 个交易对")
            return markets
            
        except Exception as e:
            self.logger.error(f"加载市场信息失败: {e}")
            raise
    
    def fetch_ticker(self, symbol: str) -> Dict:
        """获取行情数据"""
        try:
            symbol_id = symbol.replace('/', '')
            data = self._mock_api_call('GET', '/api/v1/ticker', {'symbol': symbol_id})
            
            ticker_data = data['data']
            return {
                'symbol': symbol,
                'last': float(ticker_data['price']),
                'bid': float(ticker_data['bid']),
                'ask': float(ticker_data['ask']),
                'high': float(ticker_data['high']),
                'low': float(ticker_data['low']),
                'volume': float(ticker_data['volume']),
                'timestamp': ticker_data['timestamp']
            }
            
        except Exception as e:
            self.logger.error(f"获取行情失败: {e}")
            raise
    
    def fetch_order_book(self, symbol: str, limit: int = 10) -> Dict:
        """获取订单簿"""
        try:
            symbol_id = symbol.replace('/', '')
            data = self._mock_api_call('GET', '/api/v1/depth', {'symbol': symbol_id, 'limit': limit})
            
            depth_data = data['data']
            return {
                'symbol': symbol,
                'bids': depth_data['bids'][:limit],
                'asks': depth_data['asks'][:limit],
                'timestamp': depth_data['timestamp']
            }
            
        except Exception as e:
            self.logger.error(f"获取订单簿失败: {e}")
            raise
    
    def fetch_balance(self) -> Dict:
        """获取账户余额"""
        try:
            data = self._mock_api_call('GET', '/api/v1/account')
            
            balances = {}
            for item in data['data']['balances']:
                currency = item['currency']
                balances[currency] = {
                    'free': float(item['available']),
                    'used': float(item['frozen']),
                    'total': float(item['total'])
                }
            
            return balances
            
        except Exception as e:
            self.logger.error(f"获取余额失败: {e}")
            raise
    
    def create_order(self, symbol: str, type: str, side: str, amount: float, price: float = None) -> Dict:
        """创建订单"""
        try:
            body = {
                'symbol': symbol.replace('/', ''),
                'side': side,
                'type': type,
                'amount': str(amount)
            }
            
            if type == OrderType.LIMIT and price is not None:
                body['price'] = str(price)
            
            data = self._mock_api_call('POST', '/api/v1/orders', body=body)
            
            order_data = data['data']
            return {
                'id': order_data['order_id'],
                'symbol': symbol,
                'side': side,
                'type': type,
                'amount': amount,
                'price': price,
                'status': order_data['status'],
                'timestamp': order_data['created_at']
            }
            
        except Exception as e:
            self.logger.error(f"创建订单失败: {e}")
            raise
    
    def cancel_order(self, order_id: str, symbol: str) -> Dict:
        """取消订单"""
        try:
            data = self._mock_api_call('DELETE', f'/api/v1/orders/{order_id}')
            
            order_data = data['data']
            return {
                'id': order_data['order_id'],
                'status': order_data['status']
            }
            
        except Exception as e:
            self.logger.error(f"取消订单失败: {e}")
            raise
    
    def fetch_order(self, order_id: str, symbol: str) -> Dict:
        """查询订单"""
        try:
            data = self._mock_api_call('GET', f'/api/v1/orders/{order_id}')
            
            order_data = data['data']
            return {
                'id': order_data['order_id'],
                'symbol': symbol,
                'side': order_data['side'],
                'type': order_data['type'],
                'amount': float(order_data['amount']),
                'price': float(order_data['price']) if order_data.get('price') else None,
                'filled': float(order_data['filled_amount']),
                'status': order_data['status'],
                'timestamp': order_data['created_at']
            }
            
        except Exception as e:
            self.logger.error(f"查询订单失败: {e}")
            raise
    
    def fetch_open_orders(self, symbol: str = None) -> List[Dict]:
        """获取未成交订单"""
        try:
            params = {}
            if symbol:
                params['symbol'] = symbol.replace('/', '')
            
            data = self._mock_api_call('GET', '/api/v1/orders/open', params)
            
            orders = []
            for item in data['data']['orders']:
                orders.append({
                    'id': item['order_id'],
                    'symbol': f"{item['symbol'][:3]}/{item['symbol'][3:]}",
                    'side': item['side'],
                    'type': item['type'],
                    'amount': float(item['amount']),
                    'price': float(item['price']) if item.get('price') else None,
                    'filled': float(item['filled_amount']),
                    'status': item['status'],
                    'timestamp': item['created_at']
                })
            
            return orders
            
        except Exception as e:
            self.logger.error(f"获取未成交订单失败: {e}")
            raise


# 工厂函数
def create_demo_exchange(config: Dict[str, Any]) -> DemoExchange:
    """创建演示交易所实例"""
    return DemoExchange(config)


if __name__ == "__main__":
    # 测试演示交易所
    print("=== DemoExchange 测试 ===")
    
    config = {
        'api_key': 'demo_api_key',
        'secret_key': 'demo_secret_key',
        'sandbox': True
    }
    
    try:
        # 创建交易所实例
        exchange = DemoExchange(config)
        print("✅ DemoExchange创建成功")
        
        # 测试加载市场
        markets = exchange.load_markets()
        print(f"✅ 加载市场: {len(markets)}个交易对")
        
        # 测试获取订单簿
        orderbook = exchange.fetch_order_book('BTC/USDT')
        print(f"✅ 获取订单簿: 买一 {orderbook['bids'][0]} 卖一 {orderbook['asks'][0]}")
        
        # 测试获取余额
        balance = exchange.fetch_balance()
        print(f"✅ 获取余额: BTC {balance.get('BTC', {}).get('free', 0)} USDT {balance.get('USDT', {}).get('free', 0)}")
        
        # 测试下单
        order = exchange.create_order('BTC/USDT', 'limit', 'buy', 0.001, 60000.0)
        print(f"✅ 创建订单: {order['id']}")
        
        # 测试查询订单
        order_detail = exchange.fetch_order(order['id'], 'BTC/USDT')
        print(f"✅ 查询订单: {order_detail['status']}")
        
        # 测试取消订单
        cancel_result = exchange.cancel_order(order['id'], 'BTC/USDT')
        print(f"✅ 取消订单: {cancel_result['status']}")
        
        print("\n🎉 DemoExchange测试完成！")
        print("这个模拟交易所可以用于:")
        print("- 测试做市商策略逻辑")
        print("- 学习自定义交易所实现")
        print("- 无风险的策略调试")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()