# AdvancedMarketMakerV2 策略使用指南

## 策略概述

AdvancedMarketMakerV2 是基于 FreqTrade 框架的高级做市商策略，相比原始的 AdvancedMarketMaker.py，它具有以下优势：

### 主要改进

1. **框架化订单管理**: 使用 FreqTrade 自动管理订单状态，无需手动调用 `update_order_status()`
2. **完整的回测支持**: 支持 FreqTrade 的回测功能，可以验证策略历史表现
3. **UI 集成**: 完全支持 FreqUI 实时监控和管理
4. **参数优化**: 支持超参数优化功能
5. **更好的稳定性**: 依赖成熟的 FreqTrade 框架，减少自制代码的 bug 风险

## 核心功能

### 1. 智能做市
- **动态价差**: 根据市场波动率、ATR、布林带宽度动态调整价差
- **库存平衡**: 自动监控多空仓位平衡，偏向调整买卖比例
- **价格更新控制**: 避免频繁撤单重挂，提高资金利用效率

### 2. 风险控制
- **日亏损限制**: 防止单日亏损超过设定比例
- **持仓限制**: 控制总持仓不超过资金的一定比例
- **波动率控制**: 在高波动期暂停交易
- **流动性检查**: 确保市场有足够的流动性

### 3. 参数优化
所有关键参数都支持优化：
- `spread_ratio`: 基础价差比例 (0.0005-0.01)
- `base_order_amount_ratio`: 单笔订单资金比例 (0.001-0.02)
- `inventory_rebalance_threshold`: 库存失衡阈值 (0.1-0.3)
- `volatility_threshold`: 波动率阈值 (0.02-0.1)

## 使用方法

### 1. 基础配置

复制配置文件模板：
```bash
cp config_marketmaker_v2.json my_config.json
```

编辑配置文件，修改以下关键项：
- `exchange.key` 和 `exchange.secret`: 你的交易所 API 密钥
- `exchange.pair_whitelist`: 要做市的交易对
- `dry_run`: 设为 false 开始实盘交易
- `stake_amount`: 每笔交易的资金量

### 2. 启动策略

```bash
# 回测
freqtrade backtesting --config my_config.json --strategy AdvancedMarketMakerV2 --timerange 20240101-20240201

# 干跑模式（模拟交易）
freqtrade trade --config my_config.json --dry-run

# 实盘交易
freqtrade trade --config my_config.json
```

### 3. 监控和管理

启动 FreqUI：
```bash
freqtrade webserver --config my_config.json
```

然后访问 http://localhost:8080 查看策略运行状态。

### 4. 参数优化

```bash
freqtrade hyperopt --config my_config.json --strategy AdvancedMarketMakerV2 --hyperopt-loss SharpeHyperOptLoss --epochs 100
```

## 策略参数说明

### 做市核心参数
- `spread_ratio`: 基础价差比例，决定盈利空间
- `min_spread_ratio`: 最小价差比例，防止价差过小
- `max_spread_ratio`: 最大价差比例，防止价差过大

### 订单管理参数
- `base_order_amount_ratio`: 基础订单资金比例
- `price_update_threshold`: 价格更新阈值，控制重新挂单频率

### 库存管理参数
- `inventory_target_ratio`: 库存目标比例（0.5表示多空平衡）
- `inventory_rebalance_threshold`: 库存失衡触发阈值

### 风险控制参数
- `max_position_ratio`: 最大持仓比例
- `max_daily_loss_ratio`: 最大日亏损比例
- `volatility_threshold`: 波动率阈值
- `min_volume_ratio`: 最小成交量比例

## 与原版本对比

| 特性 | AdvancedMarketMaker.py | AdvancedMarketMakerV2.py |
|------|------------------------|---------------------------|
| 框架依赖 | 独立 CCXT 实现 | FreqTrade IStrategy |
| 订单管理 | 手动状态更新 | 框架自动管理 |
| 回测支持 | 无 | 完整支持 |
| UI 监控 | 基础日志 | FreqUI 集成 |
| 参数优化 | 无 | Hyperopt 支持 |
| 数据持久化 | 自制实现 | 框架提供 |
| 稳定性 | 依赖自制代码 | 依赖成熟框架 |

## 监控指标

策略运行时会跟踪以下关键指标：

1. **交易统计**
   - 总订单数 / 成交订单数
   - 成交率和成交量
   - 已实现/未实现盈亏

2. **风险指标**
   - 当前库存失衡度
   - 持仓集中度
   - 日累计盈亏

3. **市场指标**
   - 平均价差百分比
   - 市场波动率
   - 流动性得分

## 注意事项

1. **API 限制**: 确保交易所 API 有足够的请求限制配额
2. **资金管理**: 建议先用小资金测试，验证策略有效性
3. **市场选择**: 选择流动性好、波动率适中的交易对
4. **参数调优**: 根据具体交易对的特性调整参数
5. **风险控制**: 始终设置合理的风险限制，防止大幅亏损

## 故障排除

### 常见问题

1. **订单创建失败**: 检查 API 权限和余额
2. **价差过小**: 调整 `min_spread_ratio` 参数
3. **成交率低**: 降低 `spread_ratio` 或检查市场流动性
4. **频繁撤单**: 增加 `price_update_threshold` 值

### 日志分析

关键日志位置：
- FreqTrade 主日志: `logs/freqtrade.log`
- 策略专用日志: 在主日志中搜索 "AdvancedMarketMakerV2"

## 进阶使用

### 多交易对做市
在配置文件中添加多个交易对：
```json
"pair_whitelist": [
    "BTC/USDT",
    "ETH/USDT", 
    "BNB/USDT",
    "ADA/USDT"
]
```

### 动态参数调整
可以通过 FreqUI 或 API 动态调整某些参数，无需重启策略。

### 集成监控
配置 Telegram 通知获得重要事件提醒：
```json
"telegram": {
    "enabled": true,
    "token": "your_bot_token",
    "chat_id": "your_chat_id"
}
```

这个策略为量化交易者提供了一个强大、稳定、易于监控的做市商解决方案。通过 FreqTrade 框架的集成，大大简化了策略开发和运维的复杂性。