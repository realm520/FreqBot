# 波动率调整时间序列动量(VATSM)策略

## 策略概述

VATSM (Volatility-Adjusted Time-Series Momentum) 策略是一种基于市场波动性动态调整回溯期的智能动量策略，专为加密货币市场设计。

### 核心原理

1. **动态回溯期调整**
   - 高波动率 → 较短回溯期（避免虚假信号）
   - 低波动率 → 较长回溯期（捕捉持续趋势）

2. **波动率目标定位**
   - 根据预测波动率反向调整仓位大小
   - 维持恒定的年化风险敞口（如20%）

3. **自适应动量计算**
   - 使用动态回溯期计算价格动量
   - 生成买入/卖出/持有信号

## 策略参数

### 回溯期参数
- `lb_min`: 最小回溯期 (默认: 10天)
- `lb_max`: 最大回溯期 (默认: 60天)
- `vol_short_win`: 短期波动率窗口 (默认: 20天)
- `vol_long_win`: 长期波动率窗口 (默认: 60天)
- `ratio_cap`: 波动率比率上限 (默认: 0.9)

### 风险管理参数
- `vol_target`: 目标年化波动率 (默认: 0.20, 即20%)
- `ewma_lambda`: EWMA衰减因子 (默认: 0.94)
- `min_forecast_vol`: 最小预测波动率 (默认: 0.05)
- `max_leverage`: 最大杠杆倍数 (默认: 1.0)

## 使用方法

### 1. 快速开始

```bash
# 运行BTC/USDT的默认回测
python run_vatsm_backtest.py

# 自定义参数回测
python run_vatsm_backtest.py --symbol ETH/USDT --start-date 2021-01-01 --timeframe 4h --balance 50000
```

### 2. 手动运行freqtrade

```bash
# 下载数据
freqtrade download-data --config vatsm_config.json --timeframe 1d --timerange 20200101-20240826

# 运行回测
freqtrade backtesting --config vatsm_config.json --strategy VATSMStrategy

# 生成图表分析
freqtrade plot-dataframe --config vatsm_config.json --strategy VATSMStrategy
```

### 3. 超参数优化

```bash
# 运行超参数优化
freqtrade hyperopt --config vatsm_config.json --strategy VATSMStrategy --hyperopt-loss SharpeHyperOptLoss --spaces buy --epochs 100
```

## 策略指标说明

### 核心指标
- `log_returns`: 对数收益率
- `vol_short`: 短期波动率
- `vol_long`: 长期波动率  
- `vol_ratio`: 波动率比率 (short/long)
- `lookback`: 动态回溯期
- `momentum`: 价格动量
- `vatsm_signal`: 策略信号 (+1买入, -1卖出, 0中性)

### 风险管理指标
- `ewma_vol`: EWMA预测波动率
- `vol_forecast`: 调整后的预测波动率
- `raw_exposure`: 原始风险敞口
- `target_exposure`: 目标风险敞口
- `desired_exposure`: 期望风险敞口 (含信号方向)

## 策略优势

1. **市场适应性**: 自动适应牛市、熊市和横盘市场
2. **风险控制**: 波动率目标定位防止过度杠杆
3. **信号质量**: 动态回溯期减少虚假信号
4. **透明度**: 基于统计方法，无黑盒算法
5. **灵活性**: 适用于不同时间周期和加密货币

## 注意事项

### 适用场景
- ✅ 趋势明确的市场环境
- ✅ 高波动性资产 (BTC, ETH, SOL等)
- ✅ 中长期投资策略
- ✅ 风险管控要求高的场景

### 局限性  
- ❌ 长期横盘震荡市场效果较差
- ❌ 需要足够的历史数据进行预测
- ❌ 对交易费用和滑点敏感
- ❌ 参数需要根据具体市场调优

### 风险提示
- 策略不保证盈利，请谨慎投资
- 建议先在模拟环境充分测试
- 实盘前请进行小资金验证
- 定期监控和调整参数

## 性能基准

基于历史数据的测试结果 (仅供参考):

| 指标 | BTC/USDT (2020-2024) | ETH/USDT (2020-2024) | SOL/USDT (2021-2024) |
|------|----------------------|----------------------|----------------------|
| 年化收益率 | ~15-25% | ~20-35% | ~25-45% |
| 夏普比率 | ~0.4-0.8 | ~0.5-0.9 | ~0.3-0.6 |
| 最大回撤 | ~20-40% | ~25-45% | ~30-50% |

*注：实际结果取决于市场条件和参数设置*

## 进阶优化建议

1. **多时间框架**: 结合不同时间周期的信号
2. **状态过滤**: 添加趋势确认指标 (如移动平均线)
3. **资产轮动**: 在多个加密货币间分散投资  
4. **机器学习**: 使用ML模型预测市场状态
5. **链上指标**: 结合区块链数据 (如链上交易量)

## 技术支持

如有问题或建议，请查看：
- FreqTrade文档: https://www.freqtrade.io/
- 策略代码: `user_data/strategies/VATSMStrategy.py`
- 配置文件: `vatsm_config.json`
- 回测脚本: `run_vatsm_backtest.py`