# VATSM策略开发指南

## 概述

VATSM (Volatility-Adjusted Time Series Momentum) 策略是一个基于波动率调整的时间序列动量策略，专门设计用于捕捉市场动量并适应不同波动率环境。

## 核心原理

### 1. 动态回溯期计算
```python
# 根据波动率比率动态调整回溯期
lookback = lb_min + (lb_max - lb_min) * (1.0 - vol_ratio)
```

### 2. 波动率目标定位
- 使用EWMA计算预测波动率
- 根据目标波动率调整仓位大小
- 维持恒定的风险敞口

### 3. 多重确认机制
- EMA趋势确认
- RSI过滤
- 成交量确认
- 价格突破确认

## 参数说明

### 核心参数
- `vol_target`: 目标波动率 (默认: 0.20)
- `max_leverage`: 最大杠杆 (默认: 1.0)
- `min_momentum_threshold`: 最小动量阈值 (默认: 0.005)

### 技术指标参数
- `ema_fast`: 快速EMA周期 (默认: 12)
- `ema_slow`: 慢速EMA周期 (默认: 26)
- `adx_threshold`: ADX趋势强度阈值 (默认: 25)
- `rsi_period`: RSI计算周期 (默认: 14)

### 风险管理参数
- `max_daily_loss`: 单日最大损失 (默认: 5%)
- `max_weekly_loss`: 单周最大损失 (默认: 10%)
- `drawdown_pause_threshold`: 回撤暂停阈值 (默认: 15%)

## 使用方法

### 1. 配置文件
```json
{
    "strategy": "VATSMStrategy",
    "timeframe": "15m",
    "stake_amount": 1000,
    "max_open_trades": 3
}
```

### 2. 运行回测
```bash
uv run freqtrade backtesting --config vatsm_btc_config.json --strategy VATSMStrategy --timerange 20240701-20240827
```

### 3. 超参数优化
```bash
uv run freqtrade hyperopt --config vatsm_btc_config.json --strategy VATSMStrategy --hyperopt-loss ShortTradeDurHyperOptLoss --spaces buy sell --epochs 50
```

## 市场适用性

### 适用环境
- ✅ **趋势市场**: 强势趋势中表现优异
- ✅ **中等波动**: 适中波动率环境下稳定
- ❌ **震荡市场**: 在横盘震荡中容易产生虚假信号

### 最佳实践
1. 使用多时间框架确认
2. 结合市场环境识别
3. 定期重新优化参数
4. 监控策略表现并及时调整

## 版本历史

- **Phase 1**: 基础VATSM实现
- **Phase 2A**: 添加多重确认机制
- **Phase 2B**: 集成多时间框架分析
- **Profit版本**: 专注盈利优化

## 相关文档

- [VATSM Phase 2B 报告](../reports/VATSM_PHASE_2B_REPORT.md)
- [超参数优化报告](../reports/VATSM_Hyperparameter_Optimization_Report.md)
- [最终优化总结](../reports/VATSM_Final_Optimization_Summary.md)