# 如何新增交易所

FreqBot 基于 ccxt 库，支持100+主流交易所。以下是详细的新增交易所指南。

## 🏢 支持的交易所

### 主要交易所（已配置）
- **Binance** - 全球最大交易所，功能最全
- **OKX** - 全球领先数字资产平台
- **Bybit** - 专业衍生品交易平台
- **Coinbase** - 美国最大交易所
- **Kraken** - 老牌欧美交易所
- **KuCoin** - 全球知名交易所
- **Huobi** - 全球化金融服务商
- **Bitget** - 数字资产衍生品平台
- **MEXC** - 全球数字资产平台
- **Gate.io** - 知名数字资产平台

### 其他支持的交易所
ccxt 支持 105+ 交易所，包括但不限于：
- Bitfinex, BitMEX, Bitmex
- FTX (已停服), Deribit
- Alpaca, Interactive Brokers
- 以及更多...

## 🚀 快速添加新交易所

### 方法一：使用交易所适配器（推荐）

```bash
# 1. 查看支持的交易所
python exchange_configs/exchange_adapter.py --list

# 2. 搜索特定功能的交易所
python exchange_configs/exchange_adapter.py --feature spot

# 3. 查看交易所详细信息
python exchange_configs/exchange_adapter.py --info okx

# 4. 创建配置模板
python exchange_configs/exchange_adapter.py --create-config okx
```

### 方法二：手动创建配置

#### 步骤1：创建交易所配置文件

以 OKX 为例，创建 `okx_config.json`：

```json
{
    "symbol": "BTC/USDT",
    "exchange": {
        "name": "okx",
        "apiKey": "your_okx_api_key",
        "secret": "your_okx_secret",
        "password": "your_okx_passphrase",
        "sandbox": true,
        "enableRateLimit": true,
        "timeout": 30000
    },
    "strategy_params": {
        "min_order_size": 0.001,
        "spread_ratio": 0.002,
        "price_update_threshold": 0.0005,
        "check_interval": 1.0
    },
    "position_balance": {
        "position_imbalance_threshold": 0.1,
        "rebalance_urgency_multiplier": 2.0,
        "retreat_distance": 0.001
    },
    "risk_management": {
        "max_daily_loss": 100.0,
        "position_limit": 1000.0
    }
}
```

#### 步骤2：运行策略

```bash
# 使用配置文件运行
python run_position_balanced_mm.py --config okx_config.json
```

## 📋 交易所特殊配置

### Binance 配置示例
```json
{
    "exchange": {
        "name": "binance",
        "apiKey": "your_key",
        "secret": "your_secret",
        "sandbox": true,
        "enableRateLimit": true,
        "options": {
            "defaultType": "spot"  // spot, margin, future
        }
    }
}
```

### OKX 配置示例
```json
{
    "exchange": {
        "name": "okx",
        "apiKey": "your_key",
        "secret": "your_secret", 
        "password": "your_passphrase",  // OKX 需要
        "sandbox": true,
        "enableRateLimit": true
    }
}
```

### Bybit 配置示例
```json
{
    "exchange": {
        "name": "bybit",
        "apiKey": "your_key",
        "secret": "your_secret",
        "sandbox": true,
        "enableRateLimit": true,
        "options": {
            "recv_window": 5000
        }
    }
}
```

### Kraken 配置示例
```json
{
    "exchange": {
        "name": "kraken",
        "apiKey": "your_key",
        "secret": "your_secret",
        "sandbox": false,  // Kraken 无沙盒
        "enableRateLimit": true,
        "timeout": 60000   // Kraken 较慢
    }
}
```

## 🔧 添加全新交易所

如果要添加 ccxt 支持但我们未配置的交易所：

### 1. 检查 ccxt 支持
```python
import ccxt
print('your_exchange' in ccxt.exchanges)
```

### 2. 添加到交易所配置
编辑 `exchange_configs/exchanges.json`：

```json
{
    "supported_exchanges": {
        "new_exchange": {
            "name": "New Exchange",
            "ccxt_id": "newexchange",
            "description": "新交易所描述",
            "features": ["spot", "futures"],
            "sandbox_available": true,
            "api_docs": "https://api-docs-url.com",
            "default_config": {
                "enableRateLimit": true,
                "timeout": 30000,
                "rateLimit": 1000
            },
            "trading_fees": {
                "maker": 0.001,
                "taker": 0.001
            }
        }
    }
}
```

### 3. 创建配置模板
```bash
python exchange_configs/exchange_adapter.py --create-config new_exchange
```

## ⚠️ 重要注意事项

### API 权限要求
确保您的 API 密钥具有以下权限：
- ✅ 现货交易权限
- ✅ 查询账户余额权限
- ✅ 查询订单信息权限
- ❌ 不需要提币权限

### 安全建议
1. **使用子账户**: 为做市交易创建专门的子账户
2. **限制权限**: 仅开启必要的API权限
3. **IP白名单**: 设置API的IP访问白名单
4. **资金限制**: 不要在主账户放置过多资金

### 测试流程
1. **沙盒测试**: 优先使用沙盒环境测试
2. **小资金**: 实盘先用小资金测试
3. **监控运行**: 初期密切监控策略运行状态

## 📚 交易所 API 文档

| 交易所 | API 文档 | 沙盒支持 |
|--------|----------|----------|
| Binance | [文档](https://binance-docs.github.io/apidocs/) | ✅ |
| OKX | [文档](https://www.okx.com/docs-v5/) | ✅ |
| Bybit | [文档](https://bybit-exchange.github.io/docs/) | ✅ |
| Coinbase | [文档](https://docs.cloud.coinbase.com/) | ✅ |
| Kraken | [文档](https://docs.kraken.com/rest/) | ❌ |
| KuCoin | [文档](https://docs.kucoin.com/) | ✅ |

## 🛠️ 故障排除

### 常见问题

**1. API 连接失败**
```
解决方案：
- 检查 API 密钥是否正确
- 确认 API 权限设置
- 检查网络连接
- 验证时间同步
```

**2. 订单下单失败** 
```
解决方案：
- 检查最小订单数量限制
- 确认账户余额充足
- 验证交易对是否正确
- 检查价格精度设置
```

**3. 沙盒环境问题**
```
解决方案：
- 确认交易所支持沙盒
- 检查沙盒 API 端点
- 验证沙盒账户设置
```

### 调试技巧

**开启详细日志**：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**测试连接**：
```python
import ccxt

# 创建交易所实例
exchange = ccxt.binance({
    'apiKey': 'your_key',
    'secret': 'your_secret',
    'sandbox': True,
})

# 测试连接
try:
    balance = exchange.fetch_balance()
    print("连接成功!")
    print(f"余额: {balance}")
except Exception as e:
    print(f"连接失败: {e}")
```

## 💡 最佳实践

1. **选择合适的交易所**
   - 考虑手续费率
   - 评估API稳定性
   - 查看流动性深度

2. **参数调优**
   - 根据交易所特点调整检查间隔
   - 适配不同的价格精度
   - 考虑交易所的限频规则

3. **风险管理**
   - 设置合理的仓位限制
   - 配置日亏损上限
   - 定期检查策略表现

通过以上指南，您可以轻松地将做市商策略部署到不同的交易所！