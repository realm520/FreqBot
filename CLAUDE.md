# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

FreqBot 是一个量化交易机器人项目，用于开发简单的交易策略和自动化交易功能。

## 开发环境设置

项目使用 uv 进行依赖管理和虚拟环境管理。

### Python技术栈
- **主要框架**: ccxt (加密货币交易), pandas (数据处理), numpy (数值计算)
- **回测框架**: backtrader, zipline, 或 freqtrade
- **API客户端**: 各大交易所API封装

### 常用开发命令
```bash
# 安装依赖
uv add <package-name>

# 安装开发依赖
uv add --dev <package-name>

# 运行主程序
uv run main.py

# 运行回测
uv run backtest.py

# 运行测试
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