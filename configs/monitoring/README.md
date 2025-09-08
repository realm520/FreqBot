# FreqBot 网格策略监控配置

本目录包含网格策略的完整监控配置，支持 Prometheus 和 Grafana 集成。

## 文件说明

### 监控指标模块
- `freqbot/monitoring/grid_metrics.py` - 网格策略专用监控指标收集器
- `freqbot/monitoring/__init__.py` - 监控模块初始化文件

### Grafana 配置
- `grid_dashboard.json` - Grafana 面板配置文件，包含完整的网格策略监控视图

### Prometheus 配置
- `prometheus_config.yml` - Prometheus 服务器配置
- `grid_strategy_rules.yml` - 告警规则定义

## 监控指标说明

### 网格状态指标
- `grid_total_levels` - 总网格层级数
- `grid_active_levels` - 活跃网格层级数  
- `grid_filled_levels` - 已成交网格层级数
- `grid_fill_rate` - 网格成交率 (%)

### 交易统计指标
- `grid_trades_total` - 总交易笔数
- `grid_trades_profit` - 累计交易收益
- `grid_avg_trade_profit` - 平均单笔收益
- `grid_profitable_trades` - 盈利交易笔数
- `grid_loss_trades` - 亏损交易笔数
- `grid_win_rate` - 胜率 (%)

### 仓位管理指标
- `position_current_size` - 当前仓位大小
- `position_max_size` - 历史最大仓位
- `position_utilization` - 仓位利用率 (%)
- `position_adjustments` - 仓位调整次数

### 风险指标
- `max_drawdown` - 最大回撤 (%)
- `current_drawdown` - 当前回撤 (%)
- `risk_exposure` - 风险敞口
- `stop_loss_triggers` - 止损触发次数
- `total_return` - 总收益率 (%)
- `sharpe_ratio` - 夏普比率
- `profit_factor` - 盈亏比

### 市场状态指标
- `market_regime` - 市场状态 (trend_up/trend_down/sideways)
- `volatility_level` - 波动率水平 (low/normal/high)
- `grid_efficiency` - 网格效率 (%)

## 部署说明

### 1. 启用监控

在策略中，监控系统会自动初始化。如果导入失败，策略会正常运行但无监控功能。

```python
# 策略会自动尝试初始化监控
from freqbot.monitoring.grid_metrics import GridMetrics
```

### 2. 导出 Prometheus 指标

```python
# 在策略中导出指标
strategy.export_monitoring_metrics("/tmp/grid_metrics.prom")
```

### 3. 配置 Prometheus

将 `prometheus_config.yml` 复制到 Prometheus 配置目录，并启动 Prometheus：

```bash
prometheus --config.file=prometheus_config.yml
```

### 4. 导入 Grafana 面板

1. 登录 Grafana
2. 导航到 "+" -> Import
3. 上传 `grid_dashboard.json` 文件
4. 配置数据源为 Prometheus

## 告警配置

告警规则包括：

### 风险类告警
- **高回撤告警** - 回撤超过 15%
- **极高回撤告警** - 回撤超过 25% (关键告警)
- **频繁止损告警** - 1小时内触发3次以上止损

### 性能类告警  
- **胜率过低告警** - 胜率低于 40%
- **网格效率告警** - 效率低于 30%
- **夏普比率告警** - 夏普比率低于 0.5

### 系统类告警
- **仓位利用率过高** - 超过 90%
- **长时间无交易** - 2小时内无交易活动
- **监控数据缺失** - 无法获取监控数据

## 低开销设计

监控系统采用以下优化措施确保最小性能影响：

1. **条件执行** - 只有启用监控时才执行监控代码
2. **异步更新** - 监控更新不阻塞交易逻辑
3. **批量处理** - 指标批量更新而非实时
4. **内存控制** - 历史数据自动清理，防止内存泄漏
5. **异常隔离** - 监控异常不影响策略运行

## 使用示例

### 获取监控报告
```python
# 在策略中获取实时监控报告
report = strategy.get_monitoring_report()
if report:
    print(report)
```

### 导出指标
```python  
# 导出 Prometheus 格式指标
success = strategy.export_monitoring_metrics("/path/to/metrics.prom")
```

### 手动记录事件
```python
# 记录自定义交易事件
strategy._record_grid_trade({
    'side': 'buy',
    'amount': 0.1,
    'price': 50000,
    'profit': 0.02,
    'timestamp': datetime.now()
})
```

## 故障排除

### 监控不可用
- 检查 `GridMetrics` 模块是否正确导入
- 查看策略日志中的监控相关警告信息

### 指标缺失  
- 确认策略正在运行且有交易活动
- 检查 Prometheus 是否能访问指标端点

### 面板显示异常
- 验证 Grafana 数据源配置
- 检查指标名称是否匹配
- 确认时间范围设置正确

## 性能影响

在正常运行中，监控系统的性能开销：
- CPU 使用增加: < 2%
- 内存使用增加: < 10MB  
- 延迟增加: < 5ms (每次指标更新)

这些开销对交易性能的影响可以忽略不计。