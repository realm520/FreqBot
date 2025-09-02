# FreqBot - 统一量化交易平台

FreqBot 是一个基于 FreqTrade 框架的统一量化交易平台，提供策略管理、配置管理、执行引擎和监控功能的完整解决方案。

## ✨ 核心特性

### 🎯 统一入口
- **单一CLI命令**: 所有功能通过 `uv run python main.py` 访问
- **命令行界面**: 直观的命令和参数，无需记忆复杂脚本
- **交互式操作**: 友好的用户体验和实时反馈

### 📊 策略管理
- **自动发现**: 智能识别和注册所有策略文件
- **分类管理**: 按类别组织策略（做市商、网格交易、VATSM等）
- **元数据支持**: 策略版本、作者、描述等完整信息
- **验证机制**: 自动验证策略有效性

### ⚙️ 配置管理
- **环境分离**: 开发、测试、生产环境独立配置
- **模板化**: 基于模板快速生成配置文件
- **配置迁移**: 一键迁移现有配置到新架构
- **参数验证**: 自动验证配置文件正确性

### 🚀 执行引擎
- **FreqTrade集成**: 无缝集成FreqTrade执行引擎
- **实时监控**: 策略运行过程的实时日志和状态监控
- **回测支持**: 完整的策略回测和性能分析
- **数据管理**: 自动下载和管理历史数据

## 🚀 快速开始

### 基本命令

```bash
# 查看所有可用命令
uv run python main.py --help

# 列出所有策略
uv run python main.py list-strategies

# 列出所有环境
uv run python main.py list-envs

# 运行策略（模拟交易 + 监控）
uv run python main.py run --strategy AdvancedMarketMakerV2 --env demo --monitor

# 回测策略
uv run python main.py backtest --strategy VATSMStrategy --env demo --timerange 20240901-20240902

# 监控交易
uv run python main.py monitor --db demo_trades.sqlite
```

### 从旧版本迁移

```bash
# 迁移现有配置文件
uv run python main.py migrate-config --file old_config.json --env production

# 查看迁移后的环境
uv run python main.py list-envs
```

## 📁 项目架构

```
FreqBot/
├── freqbot/                    # 核心框架
│   ├── config/                 # 配置管理器
│   │   ├── manager.py         # 统一配置管理
│   │   └── templates/         # 配置模板
│   ├── strategies/             # 策略管理器
│   │   ├── registry.py        # 策略注册表
│   │   └── loader.py          # 动态加载器
│   ├── core/                   # 核心功能
│   │   ├── engine.py          # 交易执行引擎
│   │   └── monitor.py         # 实时监控器
│   └── cli.py                  # 命令行接口
├── configs/                    # 统一配置目录
│   ├── environments/           # 环境配置
│   ├── strategies/            # 策略配置
│   └── templates/             # 配置模板
├── strategies/                 # 策略实现
│   ├── market_maker/          # 做市商策略
│   ├── vatsm/                 # VATSM策略
│   ├── grid_trading/          # 网格策略
│   └── ...
├── user_data/                 # FreqTrade数据
└── main.py                    # 统一入口点
```

## 📊 可用策略

### 🤖 做市商策略
- **AdvancedMarketMakerV2**: 智能做市商，提供流动性并从价差获利
- **FreqTradeMarketMaker**: 基础做市商策略

### 📈 趋势策略  
- **VATSMStrategy**: 波动率适应性策略，多时间框架融合
- **MeanReversionProfitStrategy**: 均值回归策略

### 🔲 网格策略
- **GridTradingStrategy**: 网格交易，适合震荡市场

## 🔧 配置管理

### 环境配置
- **demo**: 模拟交易环境，用于测试和学习
- **production**: 生产环境模板，需要配置真实API
- **custom**: 自定义环境配置

### 策略配置
- 每个策略可以有独立的参数配置
- 支持策略参数优化和版本管理
- 配置热更新，无需重启

## 📊 监控和分析

### 实时监控
- 交易统计和盈亏分析
- 持仓状态和库存平衡
- 风险指标和性能评估
- 实时日志和告警

### 数据导出
- JSON格式统计数据导出
- 兼容现有分析工具
- 自定义报告生成

## 🔒 安全特性

### 风险管理
- 多层风险控制机制
- 仓位和止损限制
- 实时风险监控和告警

### 数据安全
- API密钥通过环境变量管理
- 配置文件加密支持
- 审计日志记录

## 📚 详细文档

- **[使用指南](FREQBOT_USAGE.md)**: 详细的命令和配置说明
- **[策略文档](docs/strategies/)**: 各策略详细说明和分析
- **[开发指南](docs/guides/)**: 自定义策略和扩展开发

## 🛠 技术栈

- **Python 3.12+**: 现代Python特性支持
- **FreqTrade 2025.7+**: 强大的交易执行框架
- **SQLite**: 轻量级数据存储
- **uv**: 现代Python包管理工具

## 🤝 贡献指南

我们欢迎所有形式的贡献：

1. **Bug报告**: 发现问题请创建Issue
2. **功能建议**: 提出新功能想法
3. **代码贡献**: 提交Pull Request
4. **文档改进**: 完善文档和示例

## ⚠️ 风险提示

⚠️ **重要安全提示**:

1. **学习目的**: 本项目主要用于量化交易学习和研究
2. **充分测试**: 实盘交易前务必进行充分回测验证
3. **风险控制**: 量化交易存在风险，请设置合适的止损和仓位控制
4. **渐进式部署**: 建议从小资金开始，逐步验证策略有效性

## 📄 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE) 文件。

---

**让量化交易更简单、更安全、更高效！** 🚀