---
created: 2025-09-05T03:22:30Z
last_updated: 2025-09-05T03:22:30Z
version: 1.0
author: Claude Code PM System
---

# Project Structure

## Root Directory Organization

```
FreqBot/
├── main.py                    # 统一入口点
├── pyproject.toml             # Python项目配置（uv管理）
├── uv.lock                    # 依赖锁定文件
├── README.md                  # 项目主文档
├── CLAUDE.md                  # Claude代码助手指南
├── FREQBOT_USAGE.md          # 详细使用文档
├── SIMPLE_USAGE.md           # 简单使用指南
└── .python-version           # Python版本配置
```

## Core Framework
```
freqbot/                       # 核心框架目录
├── __init__.py               # 框架初始化
├── cli.py                    # 命令行接口实现
├── config/                   # 配置管理
│   ├── __init__.py
│   └── manager.py           # 统一配置管理器
├── strategies/               # 策略管理
│   ├── __init__.py
│   ├── loader.py           # 动态策略加载器
│   └── registry.py         # 策略注册表
└── core/                     # 核心功能
    ├── __init__.py
    ├── engine.py            # 交易执行引擎
    └── monitor.py           # 实时监控器
```

## Configurations
```
configs/                       # 统一配置目录
├── environments/             # 环境配置（demo, production等）
├── strategies/              # 策略配置文件
└── templates/               # 配置模板
```

## Strategies Implementation
```
strategies/                    # 策略实现目录
├── market_maker/             # 做市商策略
│   ├── AdvancedMarketMakerV2.py
│   └── FreqTradeMarketMaker.py
├── vatsm/                    # VATSM波动率策略
│   └── VATSMStrategy.py
├── grid_trading/             # 网格交易策略
│   └── GridTradingStrategy.py
└── MeanReversionProfitStrategy.py  # 均值回归策略
```

## User Data (FreqTrade Compatible)
```
user_data/                    # FreqTrade兼容用户数据
├── strategies/              # 用户自定义策略
│   ├── FreqTradeMarketMaker.py
│   ├── TrendBreakoutStrategy.py
│   ├── TrendBreakoutStrategyV2.py
│   ├── TrendFollowProfitStrategy.py
│   └── VolatilityIndicatorsStrategy.py
├── config_docker.json       # Docker配置文件
└── backtest_results/        # 回测结果（.gitignore）
```

## FreqTrade Configuration
```
freqbot_config/               # FreqTrade原始配置
├── data/                    # 市场数据
│   └── binance/            # 币安交易所数据
│       ├── BTC_USDT-15m.json
│       └── ETH_USDT-15m.json
├── strategies/              # FreqTrade策略示例
│   ├── sample_strategy.py
│   └── GridTradingStrategy.py
└── hyperopts/              # 超参数优化
    └── sample_hyperopt_loss.py
```

## Documentation
```
docs/                         # 文档目录
├── strategies/              # 策略文档
│   └── VolatilityIndicatorsStrategy.md
├── guides/                  # 使用指南
└── DOCKER_WEBUI_GUIDE.md  # Docker WebUI指南
```

## Docker Support
```
Docker相关文件：
├── docker-compose.yml       # Docker编排配置
├── docker_download_data.sh  # 数据下载脚本
├── docker_run_backtest.sh   # 回测运行脚本
└── ta-lib_0.6.4_amd64.deb  # TA-Lib依赖包
```

## Testing
```
tests/                        # 测试目录
└── (测试文件)
```

## Tools & Scripts
```
tools/                        # 工具脚本目录
└── (辅助工具)
```

## Exchange Configurations
```
exchange_configs/             # 交易所配置
└── (交易所特定配置)
```

## Claude Assistant
```
.claude/                      # Claude助手配置
├── context/                 # 项目上下文
│   └── README.md
└── (其他配置)
```

## Version Control
```
.git/                        # Git版本控制
.gitignore                   # Git忽略文件配置
```

## Virtual Environment
```
.venv/                       # Python虚拟环境（uv管理）
```

## Database & Logs
```
临时文件（运行时生成）：
├── demo_trades.sqlite       # 交易数据库
├── demo_trades.sqlite-shm   # 共享内存文件
├── demo_trades.sqlite-wal   # 预写式日志
└── *.jsonl.gz              # 压缩的日志文件
```

## File Naming Patterns
- 策略文件：`*Strategy.py` 或 `*Maker.py`
- 配置文件：`config_*.json` 或 `*_config.json`
- 文档文件：大写 `*.md` 文件
- 脚本文件：`docker_*.sh` 或 `*.sh`

## Module Organization
- **分层架构**：核心框架 → 策略实现 → 用户配置
- **模块化设计**：每个功能独立模块，低耦合
- **插件式策略**：策略通过注册表动态加载
- **环境分离**：开发、测试、生产环境独立配置