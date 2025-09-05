---
created: 2025-09-05T03:22:30Z
last_updated: 2025-09-05T03:22:30Z
version: 1.0
author: Claude Code PM System
---

# Technology Context

## Language & Runtime
- **Primary Language**: Python
- **Python Version**: 3.12+ (specified in .python-version)
- **Package Manager**: uv (modern Python package and project manager)
- **Virtual Environment**: .venv (managed by uv)

## Core Dependencies

### Trading Framework
- **freqtrade**: >=2025.7
  - 核心交易执行框架
  - 提供回测、实盘交易、策略管理功能
  - 支持多交易所接口

### Optimization Libraries
- **optuna**: >=4.5.0
  - 超参数优化框架
  - 用于策略参数自动调优
  
- **scikit-optimize**: >=0.10.2
  - 贝叶斯优化工具
  - 策略参数优化辅助

### Utility Libraries
- **filelock**: >=3.19.1
  - 文件锁定机制
  - 确保并发操作安全

## Technical Analysis
- **TA-Lib**: 0.6.4 (ta-lib_0.6.4_amd64.deb)
  - 技术指标计算库
  - 提供200+技术分析指标

## Data Storage
- **SQLite**: 交易数据存储
  - demo_trades.sqlite - 演示交易数据
  - 轻量级嵌入式数据库
  - 支持WAL模式提升并发性能

## Data Processing (Implied)
- **pandas**: 数据分析（FreqTrade依赖）
- **numpy**: 数值计算（FreqTrade依赖）
- **ccxt**: 加密货币交易所API（FreqTrade依赖）

## Development Tools

### Version Control
- **Git**: 版本管理
- **GitHub**: 代码托管 (realm520/FreqBot)

### Containerization
- **Docker**: 容器化部署
- **docker-compose**: 多容器编排
- Docker脚本：
  - docker_download_data.sh - 数据下载
  - docker_run_backtest.sh - 回测执行

### Build & Deploy
- **uv**: 统一的包管理和项目管理
  - 依赖管理：uv.lock
  - 虚拟环境：uv shell
  - 脚本运行：uv run

## Supported Exchanges
基于FreqTrade支持的交易所：
- **Binance**: 主要数据源（有示例数据）
- 其他FreqTrade支持的100+交易所

## Configuration Formats
- **JSON**: 策略和交易配置
- **YAML**: 文档元数据（frontmatter）
- **TOML**: 项目配置（pyproject.toml）

## Architecture Stack
```
应用层
├── CLI接口 (Click-based)
├── 策略引擎 (Plugin Architecture)
└── 监控系统 (Real-time)

框架层
├── FreqTrade Core
├── 策略加载器 (Dynamic Import)
└── 配置管理器 (Multi-env)

数据层
├── SQLite (Trading Data)
├── JSON (Market Data)
└── File System (Configs)
```

## Development Environment
- **IDE Support**: 支持任何Python IDE
- **CLI Tools**: 完整的命令行接口
- **热重载**: 策略开发支持动态加载

## Performance Considerations
- **异步处理**: FreqTrade内置异步支持
- **数据缓存**: 市场数据本地缓存
- **并行回测**: 支持多策略并行测试

## Security Features
- **API密钥管理**: 环境变量隔离
- **配置加密**: 支持敏感数据加密
- **审计日志**: 交易操作记录

## Testing Infrastructure
- **单元测试**: tests/目录
- **回测框架**: FreqTrade内置
- **模拟交易**: Demo环境支持

## Monitoring & Logging
- **实时监控**: 内置monitor模块
- **日志压缩**: .jsonl.gz格式
- **性能指标**: SQLite数据分析