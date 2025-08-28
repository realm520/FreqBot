# 如何新增自定义交易所

FreqBot支持两种类型的交易所：
1. **CCXT支持的交易所** - 已有105个主流交易所
2. **自定义交易所** - 您可以自己实现的交易所

## 🎯 新增自定义交易所的步骤

### 1. 继承基础类

创建新文件 `{exchange_name}_exchange.py`，继承 `CustomExchangeBase`：

```python
from custom_exchange_base import CustomExchangeBase
from typing import Dict, List, Optional, Any

class YourExchange(CustomExchangeBase):
    def __init__(self, config: Dict[str, Any]):
        # 设置默认配置
        default_config = {
            'base_url': 'https://api.yourexchange.com',
            'sandbox': True,
            'timeout': 30
        }
        default_config.update(config)
        super().__init__(default_config)
    
    # 实现所有抽象方法...
```

### 2. 实现必需方法

您需要实现以下11个抽象方法：

#### 🔐 认证相关
```python
def _generate_signature(self, method: str, path: str, params: Dict = None, body: str = "") -> str:
    """生成API签名 - 根据交易所文档实现"""
    # 实现您的交易所的签名算法
    pass

def _get_timestamp(self) -> str:
    """获取时间戳格式"""
    return str(int(time.time() * 1000))  # 毫秒时间戳

def _get_auth_headers(self, signature: str) -> Dict[str, str]:
    """设置认证头"""
    return {
        'X-API-KEY': self.api_key,
        'X-SIGNATURE': signature,
        'X-TIMESTAMP': self._get_timestamp()
    }
```

#### 📊 市场数据
```python
def load_markets(self) -> Dict:
    """加载交易对信息"""
    data = self._make_request('GET', '/api/symbols')
    markets = {}
    for item in data:
        symbol = f"{item['base']}/{item['quote']}"
        markets[symbol] = {
            'id': item['symbol'],
            'symbol': symbol,
            'base': item['base'],
            'quote': item['quote'],
            'precision': {
                'price': item['price_precision'],
                'amount': item['amount_precision']
            },
            'limits': {
                'amount': {'min': item['min_amount']},
                'cost': {'min': item['min_cost']}
            }
        }
    return markets

def fetch_ticker(self, symbol: str) -> Dict:
    """获取行情数据"""
    pass

def fetch_order_book(self, symbol: str, limit: int = 10) -> Dict:
    """获取订单簿"""
    pass
```

#### 💰 账户管理
```python
def fetch_balance(self) -> Dict:
    """获取账户余额"""
    data = self._make_request('GET', '/api/account')
    balances = {}
    for item in data['balances']:
        balances[item['currency']] = {
            'free': float(item['available']),
            'used': float(item['frozen']),
            'total': float(item['total'])
        }
    return balances
```

#### 📝 订单管理
```python
def create_order(self, symbol: str, type: str, side: str, amount: float, price: float = None) -> Dict:
    """创建订单"""
    pass

def cancel_order(self, order_id: str, symbol: str) -> Dict:
    """取消订单"""
    pass

def fetch_order(self, order_id: str, symbol: str) -> Dict:
    """查询订单"""
    pass

def fetch_open_orders(self, symbol: str = None) -> List[Dict]:
    """获取未成交订单"""
    pass
```

### 3. 研究交易所API文档

在实现之前，您需要仔细阅读目标交易所的API文档：

#### 🔍 重点关注的内容：
1. **认证方式** - 签名算法、请求头格式
2. **API端点** - 基础URL、具体路径
3. **请求格式** - 参数格式、时间戳要求
4. **响应格式** - 数据结构、错误码
5. **限频规则** - 请求频率限制
6. **测试环境** - 沙盒API地址

#### 📝 常见签名算法：
```python
# HMAC-SHA256 (最常见)
signature = hmac.new(
    secret_key.encode(),
    message.encode(),
    hashlib.sha256
).hexdigest()

# HMAC-SHA512
signature = hmac.new(
    secret_key.encode(), 
    message.encode(),
    hashlib.sha512
).hexdigest()

# RSA签名 (少见)
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
```

### 4. 创建配置文件

创建 `{exchange_name}_config.json`：

```json
{
    "symbol": "BTC/USDT",
    "exchange": {
        "name": "yourexchange",
        "api_key": "your_api_key",
        "secret_key": "your_secret_key",
        "base_url": "https://api.yourexchange.com",
        "sandbox": true,
        "timeout": 30000
    },
    "strategy_params": {
        "min_order_size": 0.001,
        "spread_ratio": 0.002,
        "check_interval": 1.0
    }
}
```

### 5. 测试实现

```python
if __name__ == "__main__":
    config = {
        'api_key': 'test_key',
        'secret_key': 'test_secret',
        'sandbox': True
    }
    
    exchange = YourExchange(config)
    
    # 测试基础功能
    markets = exchange.load_markets()
    balance = exchange.fetch_balance()
    orderbook = exchange.fetch_order_book('BTC/USDT')
    
    print("测试完成！")
```

## 🚀 使用自定义交易所

### 1. 自动发现
将您的交易所文件命名为 `{name}_exchange.py` 并放在 `exchange_configs/` 目录下，系统会自动发现。

### 2. 手动注册
```python
from exchange_configs.custom_exchange_manager import exchange_manager
from your_exchange import YourExchange

exchange_manager.register_custom_exchange('yourexchange', YourExchange)
```

### 3. 在做市商中使用
```python
from UniversalMarketMaker import UniversalMarketMaker

config = {
    'symbol': 'BTC/USDT',
    'exchange': {
        'name': 'yourexchange',  # 您的交易所名称
        'api_key': 'your_api_key',
        'secret_key': 'your_secret_key'
    }
}

strategy = UniversalMarketMaker(config)
strategy.run()
```

## 📋 完整示例

查看 `demo_exchange.py` 获取完整的实现示例，它演示了：
- ✅ 完整的API接口实现
- ✅ 模拟数据生成
- ✅ 错误处理
- ✅ 日志记录
- ✅ 单元测试

## 🛠️ 实用工具

### 查看支持的交易所
```bash
python exchange_configs/exchange_adapter.py --list
```

### 创建配置模板
```bash
python exchange_configs/exchange_adapter.py --create-config yourexchange
```

### 测试自定义交易所
```bash
python exchange_configs/demo_exchange.py
```

### 运行通用做市商
```bash
python user_data/strategies/UniversalMarketMaker.py --exchange yourexchange
```

## ⚠️ 开发注意事项

### 1. 安全性
- 🔐 妥善保管API密钥
- 🧪 先在沙盒环境测试
- 🔍 验证所有输入参数
- 🚫 不要在日志中记录敏感信息

### 2. 稳定性  
- ⏱️ 实现适当的重试机制
- 🚦 遵守API限频规则
- 💾 缓存市场数据
- 🔄 处理网络异常

### 3. 兼容性
- 📊 统一数据格式
- 🎯 精确处理价格和数量精度
- ⚡ 优化响应时间
- 📝 详细的错误信息

## 🎉 完成！

按照以上步骤，您就可以成功将任何交易所接入FreqBot了！

如果遇到问题，可以：
1. 参考 `demo_exchange.py` 示例
2. 查看交易所官方API文档
3. 在项目Issues中提问

祝您接入顺利！ 🚀