# EnhancedGridStrategy - 量化网格动态调仓策略

## 概述

EnhancedGridStrategy 是一个基于网格交易原理的高级量化策略，通过动态调仓和智能风控实现稳定收益。该策略继承并增强了基础的 GridTradingStrategy，添加了以下核心功能：

- **动态网格计算** - 基于 ATR 和市场波动率自适应调整网格间距
- **市场状态识别** - 使用 ADX 等指标识别趋势/震荡/过渡市场
- **智能仓位管理** - 三级仓位系统根据市场状态动态调整
- **VATSM 风控集成** - 三层止损系统和回撤保护机制
- **实时性能监控** - Prometheus/Grafana 监控面板

## 快速开始

### 1. 使用 Docker 运行策略

```bash
# 下载历史数据
./docker_download_data.sh --pairs "BTC/USDT ETH/USDT" --timeframe 15m

# 运行回测
./docker_run_backtest.sh --strategy EnhancedGridStrategy --timeframe 15m

# 启动实盘交易（测试模式）
docker-compose up -d
```

### 2. 配置文件

策略配置文件位于 `configs/` 目录：

- `strategy-21-base.json` - 基础配置
- `strategy-21-params.json` - 策略参数
- `strategy-21-demo.json` - 测试环境配置
- `strategy-21-live.json` - 生产环境配置

## 核心功能

### 动态网格系统

策略使用 20 层固定网格，但间距和边界会根据市场条件动态调整：

- **ATR 自适应间距** - 高波动时扩大间距，低波动时缩小
- **布林带边界** - 使用布林带上下轨确定网格区间
- **智能重置** - 价格突破区间时自动重新布局

### 市场状态识别

基于 ADX 指标的三状态系统：

- **震荡市** (ADX < 25) - 适合网格交易，使用保守仓位
- **趋势市** (ADX > 40) - 减少网格密度，增加仓位
- **过渡期** (25 < ADX < 40) - 标准配置

### 仓位管理

三级动态仓位系统：

- **保守模式** - 基础仓位 × 0.5 (震荡市)
- **标准模式** - 基础仓位 × 1.0 (过渡期)
- **激进模式** - 基础仓位 × 1.5 (趋势市)

### 风险控制

集成 VATSM 三层止损系统：

1. **第一层** - 盈利 < 5%，止损距离 2%
2. **第二层** - 盈利 5-10%，止损距离 1%  
3. **第三层** - 盈利 > 10%，止损距离 0.5%

额外保护机制：
- 单日最大损失 5%
- 单周最大损失 10%
- 连续 5 笔亏损自动暂停
- 48 小时持仓时间限制

## 性能指标

目标性能（基于历史回测）：

| 指标 | 最低要求 | 目标值 | 优秀标准 |
|------|---------|--------|----------|
| 年化收益率 | >15% | >25% | >35% |
| 最大回撤 | <15% | <10% | <8% |
| 夏普比率 | >1.0 | >1.5 | >2.0 |
| 胜率 | >45% | >55% | >60% |
| 盈亏比 | >1.2 | >1.5 | >2.0 |

## 监控面板

策略集成了 Prometheus/Grafana 监控：

```bash
# 启动监控服务
docker-compose up -d prometheus grafana

# 访问 Grafana
# URL: http://localhost:3000
# 用户名: admin
# 密码: admin
```

监控指标包括：
- 网格活跃度和成交统计
- 仓位变化和资金利用率
- 风险指标和止损触发
- 策略收益和性能评估

## 回测验证

运行完整回测套件：

```bash
# 运行主回测
python scripts/backtest_enhanced_grid.py

# 运行场景测试（9种市场环境）
python scripts/backtest_scenarios.py

# 生成分析报告
python scripts/analyze_backtest_results.py
```

回测报告位于 `backtest_results/reports/` 目录。

## 参数优化

主要可调参数：

```python
# 网格参数
grid_levels = 20  # 网格层数
grid_spacing_sensitivity = 1.0  # 间距敏感度
grid_boundary_extension = 0.1  # 边界扩展系数

# 市场识别参数
adx_threshold_low = 25  # 震荡市阈值
adx_threshold_high = 40  # 趋势市阈值

# 仓位管理参数
position_size_base = 0.1  # 基础仓位比例
position_adjustment_factor = 1.5  # 调整系数

# 风控参数
stoploss_tier1_profit = 0.05  # 第一层盈利阈值
stoploss_tier2_profit = 0.10  # 第二层盈利阈值
max_daily_loss = 0.05  # 单日最大损失
```

## 部署指南

### 生产环境部署

1. **环境准备**
   ```bash
   # 克隆代码
   git clone https://github.com/realm520/FreqBot.git
   cd FreqBot
   
   # 切换到策略分支
   git checkout epic/strategy-21
   ```

2. **配置 API 密钥**
   ```bash
   # 编辑配置文件
   cp configs/strategy-21-live.json.example configs/strategy-21-live.json
   # 添加交易所 API 密钥
   ```

3. **启动策略**
   ```bash
   # 生产模式
   docker-compose -f docker-compose.prod.yml up -d
   ```

### 安全建议

- 使用独立的 API 密钥，仅授予交易权限
- 设置 IP 白名单限制
- 启用双因素认证
- 定期备份交易日志
- 监控异常交易行为

## 常见问题

### Q: 策略适合什么市场环境？
A: 最适合震荡市场，在单边趋势市场表现一般。建议在 BTC/USDT、ETH/USDT 等主流交易对使用。

### Q: 最小启动资金是多少？
A: 建议最小 1000 USDT，推荐 5000 USDT 以上以充分发挥网格策略优势。

### Q: 如何调整风险等级？
A: 修改 `configs/strategy-21-params.json` 中的风控参数，或使用保守/标准/激进预设配置。

### Q: 策略会自动止损吗？
A: 是的，策略集成了多层止损系统，会根据盈利水平和市场条件自动调整止损位。

## 技术支持

- GitHub Issues: https://github.com/realm520/FreqBot/issues
- 策略文档: `docs/strategies/`
- 开发指南: `CONTRIBUTING.md`

## 许可证

本策略基于 FreqTrade 开源框架开发，遵循 GPL-3.0 许可证。

---

**免责声明**: 加密货币交易存在高风险，过往表现不代表未来收益。请谨慎投资，风险自负。