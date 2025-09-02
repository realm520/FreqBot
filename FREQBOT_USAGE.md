# FreqBot 统一平台使用指南

## 简介

FreqBot 是一个基于 FreqTrade 的统一量化交易平台，提供了策略管理、配置管理、执行引擎和监控功能的完整解决方案。

## 快速开始

### 1. 查看可用功能

```bash
# 查看所有命令
uv run python main.py --help

# 列出所有可用策略
uv run python main.py list-strategies

# 列出所有环境
uv run python main.py list-envs
```

### 2. 运行策略（模拟交易）

```bash
# 运行做市商策略（模拟模式）
uv run python main.py run --strategy AdvancedMarketMakerV2 --env demo --monitor

# 运行VATSM策略
uv run python main.py run --strategy VATSMStrategy --env vatsm --monitor

# 运行网格策略
uv run python main.py run --strategy GridTradingStrategy --env demo --monitor
```

### 3. 回测策略

```bash
# 先下载数据
uv run python main.py download-data --env demo

# 运行回测
uv run python main.py backtest --strategy AdvancedMarketMakerV2 --env demo --timerange 20240901-20240902
```

### 4. 监控交易

```bash
# 监控当前交易（15秒刷新）
uv run python main.py monitor --db demo_trades.sqlite

# 自定义刷新间隔（5秒）
uv run python main.py monitor --db demo_trades.sqlite --interval 5
```

## 命令详解

### list-strategies - 列出策略
```bash
# 列出所有策略
uv run python main.py list-strategies

# 按分类筛选
uv run python main.py list-strategies --category market_maker
```

### run - 运行策略
```bash
# 基本运行
uv run python main.py run --strategy STRATEGY_NAME --env ENV_NAME

# 强制模拟模式
uv run python main.py run --strategy STRATEGY_NAME --env ENV_NAME --dry-run

# 强制实盘模式（谨慎使用）
uv run python main.py run --strategy STRATEGY_NAME --env ENV_NAME --live

# 启用监控
uv run python main.py run --strategy STRATEGY_NAME --env ENV_NAME --monitor
```

### backtest - 回测策略
```bash
# 基本回测
uv run python main.py backtest --strategy STRATEGY_NAME --env ENV_NAME

# 指定时间范围
uv run python main.py backtest --strategy STRATEGY_NAME --env ENV_NAME --timerange 20240101-20240201
```

### download-data - 下载数据
```bash
# 下载环境配置中的默认数据
uv run python main.py download-data --env demo

# 指定交易对和时间框架
uv run python main.py download-data --env demo --pairs BTC/USDT ETH/USDT --timeframe 5m --timerange 20240901-
```

### migrate-config - 迁移旧配置
```bash
# 迁移现有配置文件
uv run python main.py migrate-config --file old_config.json --env new_env_name
```

## 项目结构

```
FreqBot/
├── freqbot/                    # 核心框架
│   ├── config/                 # 配置管理
│   ├── strategies/             # 策略管理
│   └── core/                   # 执行引擎和监控
├── configs/                    # 配置文件
│   ├── environments/           # 环境配置（demo.json, production.json等）
│   └── strategies/             # 策略特定配置
├── strategies/                 # 策略实现
│   ├── market_maker/           # 做市商策略
│   ├── vatsm/                  # VATSM策略
│   ├── grid_trading/           # 网格交易策略
│   └── ...
└── user_data/                  # FreqTrade数据目录
```

## 可用策略

### 1. AdvancedMarketMakerV2 (market_maker)
- **描述**: 智能做市商策略，提供流动性并从买卖价差中获利
- **适用场景**: 高频交易，震荡市场
- **风险等级**: 中等

### 2. VATSMStrategy (vatsm)
- **描述**: 波动率适应性策略，多重确认机制和多时间框架融合
- **适用场景**: 趋势跟踪，波动率交易
- **风险等级**: 中高

### 3. GridTradingStrategy (grid_trading)
- **描述**: 网格交易策略，适合震荡行情的分批买卖
- **适用场景**: 震荡市场，区间交易
- **风险等级**: 低中

## 环境配置

### demo 环境
- 模拟交易模式
- 初始资金: 10,000 USDT
- 交易对: BTC/USDT
- API服务器启用

### 生产环境配置
1. 复制 `configs/templates/production_environment.json`
2. 修改交易所API配置
3. 设置实际资金和风险参数
4. 确保充分回测验证

## 安全提示

⚠️ **重要安全提示**:

1. **API密钥安全**: 
   - 生产环境API密钥通过环境变量设置
   - 不要在配置文件中硬编码敏感信息

2. **实盘交易前准备**:
   - 充分的策略回测验证
   - 小额资金测试
   - 监控和风控机制

3. **风险管理**:
   - 设置合适的止损和仓位限制
   - 定期监控和评估策略表现
   - 保持资金管理纪律

## 开发和扩展

### 添加新策略
1. 在 `strategies/` 目录下创建策略文件
2. 继承 FreqTrade 的 `IStrategy` 类
3. 添加策略元数据：
   ```python
   STRATEGY_NAME = "MyStrategy"
   STRATEGY_VERSION = "1.0.0"
   STRATEGY_AUTHOR = "Your Name"
   STRATEGY_CATEGORY = "category_name"
   STRATEGY_DESCRIPTION = "策略描述"
   ```

### 自定义配置
- 环境配置: `configs/environments/`
- 策略配置: `configs/strategies/`
- 配置模板: `configs/templates/`

## 故障排除

### 常见问题

1. **策略未找到**
   - 检查策略文件路径和名称
   - 确保策略类继承了 IStrategy
   - 查看日志中的具体错误信息

2. **配置文件错误**
   - 验证JSON格式
   - 检查必需字段
   - 使用 `migrate-config` 从旧配置迁移

3. **数据下载失败**
   - 检查网络连接
   - 验证交易所配置
   - 确认交易对有效性

### 日志查看
- FreqTrade日志: `logs/` 目录
- FreqBot日志: 控制台输出

## 技术支持

如遇问题，请检查：
1. 日志文件中的详细错误信息
2. 配置文件的正确性
3. FreqTrade依赖是否正确安装

---

**免责声明**: 量化交易存在风险，请谨慎操作，确保充分理解策略逻辑和风险控制措施。