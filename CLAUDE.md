# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

FreqBot 是基于 FreqTrade 的 Docker 化量化交易平台。所有交易执行、回测和监控均通过 Docker 容器完成，本地代码仅提供策略管理和配置工具。

## 核心技术栈

- **Python**: 3.12+（使用 uv 进行包管理）
- **FreqTrade**: 2025.7+（Docker 镜像）
- **Docker**: 容器化部署和执行环境
- **依赖管理**: uv（现代 Python 包管理工具）

## Docker 工作流（主要使用方式）

### 回测策略
```bash
./docker_run_backtest.sh --strategy VATSMStrategy --timeframe 15m --timerange 20240101-20240826
```

### 下载历史数据
```bash
./docker_download_data.sh --pairs 'ETH/USDT BTC/USDT' --timeframe 15m --timerange 20240101-20240826
```

### 启动 Web UI 监控
```bash
docker-compose up -d
# 访问 http://localhost:8080
# 用户名: freqtrade
# 密码: freqtrade123
```

### 停止服务
```bash
docker-compose down
```

## 项目架构

### 目录结构
```
freqbot/               # 轻量级 Python 框架（仅用于本地工具）
├── config/            # ConfigManager - 配置文件管理
└── strategies/        # StrategyRegistry, StrategyLoader - 策略发现和加载

strategies/            # 策略实现（按类别组织）
├── market_maker/      # 做市商策略
├── vatsm/             # 波动率适应性策略
└── grid_trading/      # 网格交易策略

configs/               # 配置文件（环境隔离）
├── environments/      # 环境配置（demo.json, production.json等）
├── strategies/        # 策略特定配置
└── templates/         # 配置模板

user_data/             # FreqTrade 数据目录（映射到 Docker 容器）
├── data/              # 历史数据
├── backtest_results/  # 回测结果
└── config_docker.json # Docker 容器使用的配置

docker-compose.yml     # Docker 服务编排
docker_run_backtest.sh # 回测脚本
docker_download_data.sh # 数据下载脚本
```

### 架构关键点

1. **配置管理** (`freqbot/config/manager.py`)
   - `ConfigManager`: 统一配置加载和管理
   - 生成 Docker 容器使用的配置文件
   - 环境隔离：demo, production, custom

2. **策略系统** (`freqbot/strategies/`)
   - `StrategyRegistry`: 自动发现和注册策略
   - `StrategyLoader`: 动态加载策略类
   - 策略元数据：版本、作者、描述、分类

3. **Docker 执行**
   - 所有交易执行通过 `freqtradeorg/freqtrade:stable` 镜像
   - 容器挂载 `user_data/` 目录访问策略和配置
   - Web UI 提供实时监控和回测结果查看

### 策略开发模式

所有策略必须：
1. 继承 `freqtrade.strategy.IStrategy`
2. 实现 `populate_indicators()`, `populate_entry_trend()`, `populate_exit_trend()`
3. 放置在 `strategies/` 或 `user_data/strategies/` 目录下（可按类别分组）
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

### Docker 配置文件
主配置文件位于 `user_data/config_docker.json`，Docker 容器启动时自动加载。

### 环境配置
- `configs/environments/demo.json` - 模拟交易环境
- `configs/environments/production.json` - 实盘交易环境
- 使用 `ConfigManager` 生成和管理配置

## 本地开发工具

### Python 依赖管理
```bash
uv add <package>              # 添加依赖
uv add --dev <package>        # 添加开发依赖
uv sync                       # 同步依赖
uv lock                       # 锁定版本
```

### 测试
```bash
# 运行所有测试
uv run pytest tests/

# 进入虚拟环境
uv shell
```

## 安全注意事项

- API 密钥和私钥必须通过环境变量或配置文件管理，绝不可硬编码
- 实盘交易前务必进行充分的回测和模拟交易
- 配置文件中的敏感信息已通过 `.gitignore` 排除版本控制

## 开发流程

1. **策略开发**: 在 `strategies/` 目录下实现新策略
2. **回测验证**: 使用 `docker_run_backtest.sh` 验证策略有效性
3. **模拟交易**: 通过 Docker Compose 在模拟环境中测试策略
4. **实盘部署**: 小资金实盘验证后逐步扩大规模

## 监控和调试

- **Web UI**: `http://localhost:8080` 提供完整的监控界面
- **日志**: Docker 容器日志通过 `docker-compose logs -f` 查看
- **回测结果**: 保存在 `user_data/backtest_results/` 目录
- **数据检查**: 历史数据存储在 `user_data/data/` 目录
