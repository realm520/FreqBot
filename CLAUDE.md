# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

FreqBot 是基于 FreqTrade 的统一量化交易平台，提供策略管理、配置管理、执行引擎和监控功能的完整解决方案。项目已重构为统一架构，所有功能通过 `main.py` 作为单一入口点访问。

## 核心技术栈

- **Python**: 3.12+（使用 uv 进行包管理）
- **FreqTrade**: 2025.7+（交易执行框架）
- **SQLite**: 交易数据存储
- **依赖管理**: uv（现代 Python 包管理工具）

## 统一 CLI 命令

所有功能通过 `uv run python main.py` 访问，主要命令包括：

```bash
# 策略管理
uv run python main.py list-strategies [--category CATEGORY]
uv run python main.py init-config --strategy NAME --env ENV

# 环境管理
uv run python main.py list-envs
uv run python main.py migrate-config --file OLD_CONFIG --env NEW_ENV

# 交易执行
uv run python main.py run --strategy NAME --env ENV [--monitor] [--dry-run|--live]
uv run python main.py backtest --strategy NAME --env ENV [--timerange RANGE]

# 数据管理
uv run python main.py download-data --env ENV [--pairs PAIRS] [--timeframe TF] [--timerange RANGE]

# 监控
uv run python main.py monitor [--db DB_FILE] [--interval SECONDS]

# 依赖管理
uv add <package>              # 添加依赖
uv add --dev <package>        # 添加开发依赖
uv sync                       # 同步依赖
uv lock                       # 锁定版本
```

## 项目架构

### 目录结构
```
freqbot/               # 核心框架
├── config/            # ConfigManager - 统一配置管理
├── strategies/        # StrategyRegistry, StrategyLoader - 策略发现和加载
├── core/              # TradingEngine, TradeMonitor - 执行和监控
└── cli.py             # FreqBotCLI - 命令行接口

strategies/            # 策略实现（按类别组织）
├── market_maker/      # 做市商策略
├── vatsm/             # 波动率适应性策略
└── grid_trading/      # 网格交易策略

configs/               # 配置文件（环境隔离）
├── environments/      # 环境配置（demo.json, production.json等）
├── strategies/        # 策略特定配置
└── templates/         # 配置模板

user_data/             # FreqTrade 数据目录
```

### 架构关键点

1. **配置管理** (`freqbot/config/manager.py`)
   - `ConfigManager`: 统一配置加载和管理
   - `EnvironmentConfig`, `TradingConfig`, `ExchangeConfig`: 配置数据类
   - 环境隔离：demo, production, custom

2. **策略系统** (`freqbot/strategies/`)
   - `StrategyRegistry`: 自动发现和注册策略
   - `StrategyLoader`: 动态加载策略类
   - 策略元数据：版本、作者、描述、分类

3. **执行引擎** (`freqbot/core/engine.py`)
   - `TradingEngine`: FreqTrade 集成和执行
   - 支持干跑（dry-run）和实盘模式

4. **监控系统** (`freqbot/core/monitor.py`)
   - `TradeMonitor`: 实时交易统计和风险监控
   - SQLite 数据库集成

### 策略开发模式

所有策略必须：
1. 继承 `freqtrade.strategy.IStrategy`
2. 实现 `populate_indicators()`, `populate_entry_trend()`, `populate_exit_trend()`
3. 放置在 `strategies/` 目录下（可按类别分组）
4. 包含元数据字典（可选但推荐）

示例结构：
```python
class MyStrategy(IStrategy):
    STRATEGY_METADATA = {
        "name": "MyStrategy",
        "version": "1.0.0",
        "author": "Your Name",
        "category": "trend_following"
    }
    # 策略实现...
```

## 配置系统

### 环境配置优先级
1. 命令行参数（`--dry-run`, `--live`）
2. 环境配置文件（`configs/environments/{env}.json`）
3. 策略配置文件（`configs/strategies/{strategy}.json`）
4. 默认模板

### 配置迁移
使用 `migrate-config` 命令将旧的 FreqTrade 配置迁移到新架构：
```bash
uv run python main.py migrate-config --file old_config.json --env production
```

## 测试

```bash
# 运行所有测试
uv run pytest tests/

# 进入虚拟环境
uv shell

# 同步依赖
uv sync

# 锁定依赖版本
uv lock
```

## 架构设计原则

### 模块结构建议
- **策略模块** (`strategies/`): 包含各种交易策略
- **数据模块** (`data/`): 市场数据获取和处理
- **执行模块** (`execution/`): 订单执行和风险管理
- **回测模块** (`backtest/`): 策略回测和性能分析
- **配置模块** (`config/`): 交易参数和API配置
- **工具模块** (`utils/`): 通用工具函数

### 核心组件
1. **策略引擎**: 实现交易信号生成逻辑
2. **风险管理**: 仓位控制和止损机制
3. **数据管理**: 实时和历史数据处理
4. **订单管理**: 交易执行和状态跟踪

## 安全注意事项

- API密钥和私钥必须通过环境变量或配置文件管理，绝不可硬编码
- 实盘交易前务必进行充分的回测和模拟交易
- 实现适当的风险控制机制，包括最大损失限制和仓位限制

## 开发流程

1. **策略开发**: 在`strategies/`目录下实现新策略
2. **回测验证**: 使用历史数据验证策略有效性
3. **模拟交易**: 在模拟环境中测试策略
4. **实盘部署**: 小资金实盘验证后逐步扩大规模
- 使用docker_download_data.sh下载数据