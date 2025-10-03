# FreqBot - Docker 化量化交易平台

FreqBot 是基于 FreqTrade 的 Docker 化量化交易平台。所有交易执行、回测和监控均通过 Docker 容器完成，专注于策略开发和管理。

## ✨ 核心特性

### 🐳 Docker-First 架构
- **容器化执行**: 所有交易通过 FreqTrade Docker 镜像运行
- **环境隔离**: 开发、测试、生产环境完全隔离
- **一键部署**: 简化的部署和管理流程
- **Web UI 监控**: 实时监控和回测结果可视化

### 📊 丰富的策略库
- **资金费率套利**: U本位永续合约资金费率增强策略
- **做市商策略**: 智能做市和流动性提供
- **趋势策略**: VATSM、均值回归、趋势突破
- **网格策略**: 震荡市场网格交易
- **波动率策略**: 基于波动率的交易策略

### ⚙️ 灵活的配置管理
- **配置模板**: 快速生成和定制配置
- **环境配置**: 独立的环境参数管理
- **策略配置**: 策略特定的参数优化
- **热更新**: 配置变更无需重启

### 🚀 强大的执行引擎
- **FreqTrade 集成**: 基于成熟的交易框架
- **实时监控**: 完整的监控和告警系统
- **回测支持**: 历史数据回测和性能分析
- **数据管理**: 自动化的数据下载和管理

## 🚀 快速开始

### 前置要求

- Docker 和 Docker Compose
- Python 3.12+ (可选，用于本地工具)
- uv (推荐用于 Python 包管理)

### 基本使用

#### 1. 回测策略

```bash
# 回测资金费率套利策略
./docker_run_backtest.sh --strategy FundingRateEnhancedStrategy \
  --timeframe 15m \
  --timerange 20240101-20241001

# 回测 VATSM 策略
./docker_run_backtest.sh --strategy VATSMStrategy \
  --timeframe 15m \
  --timerange 20240901-20240902
```

#### 2. 下载历史数据

```bash
# 下载永续合约数据
./docker_download_data.sh \
  --pairs 'BTC/USDT:USDT ETH/USDT:USDT' \
  --timeframe 15m \
  --timerange 20240101-20241001

# 下载现货数据
./docker_download_data.sh \
  --pairs 'BTC/USDT ETH/USDT' \
  --timeframe 15m \
  --timerange 20240101-20241001
```

#### 3. 启动 Web UI 监控

```bash
# 启动服务
docker-compose up -d

# 访问 Web UI
# URL: http://localhost:8080
# 用户名: freqtrade
# 密码: freqtrade123

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

## 📁 项目架构

```
FreqBot/
├── user_data/                    # FreqTrade 数据目录 (映射到 Docker)
│   ├── strategies/               # 策略实现
│   │   ├── FundingRateEnhancedStrategy.py    # 资金费率套利
│   │   ├── VATSMStrategy.py                  # VATSM 策略
│   │   ├── AdvancedMarketMakerV2.py         # 做市商 V2
│   │   ├── GridTradingStrategy.py           # 网格交易
│   │   └── ...                               # 其他策略
│   ├── data/                     # 历史数据
│   ├── backtest_results/         # 回测结果
│   └── config_docker.json        # Docker 容器配置
│
├── configs/                      # 配置管理
│   ├── environments/             # 环境配置
│   │   ├── demo.json            # 模拟交易
│   │   └── production.json      # 实盘交易
│   ├── strategies/              # 策略配置
│   └── templates/               # 配置模板
│
├── freqbot/                     # 本地工具框架
│   ├── config/                  # 配置管理器
│   └── strategies/              # 策略注册和加载
│
├── docker-compose.yml           # Docker 服务编排
├── docker_run_backtest.sh      # 回测脚本
├── docker_download_data.sh     # 数据下载脚本
└── CLAUDE.md                   # 项目开发指南
```

## 📊 策略详解

### 🪙 资金费率套利策略 (FundingRateEnhancedStrategy)

**最新策略** - U本位永续合约资金费率增强策略

**策略特点**:
- 利用永续合约资金费率作为主要 Alpha 来源
- 费率增强的方向性交易（非纯套利）
- 多重风控机制（硬止损、费率反转、时间止损、盈亏比、追踪止盈）
- 动态仓位管理（基于费率极端度和波动率）

**核心逻辑**:
- 高正费率时做空收取费率（年化 >30%）
- 负费率时做多收取费率
- 技术面多重过滤（RSI、ADX、ATR）
- 严格的成本-收益控制

**预期表现**:
- 年化收益: 15-35%
- 最大回撤: <12%
- 胜率: 50-60%
- 夏普比率: >1.2

**详细文档**: [user_data/strategies/FundingRateEnhancedStrategy_README.md](user_data/strategies/FundingRateEnhancedStrategy_README.md)

### 📈 VATSM 策略 (VATSMStrategy)

**波动率适应性趋势动量策略**

**策略特点**:
- 动态调整回溯期（基于波动率）
- 波动率目标定位（恒定风险敞口）
- 多时间框架融合（1h趋势 + 5m入场）
- 三层追踪止损优化

**核心逻辑**:
- 高波动时缩短回溯期，低波动时延长
- 根据预测波动率调整仓位大小
- 多重确认机制（RSI、成交量、市场结构）

**详细文档**: [user_data/strategies/VATSMStrategy.json](user_data/strategies/VATSMStrategy.json)

### 🤖 做市商策略

#### AdvancedMarketMakerV2
- 智能做市，提供流动性
- 动态价差调整
- 库存风险管理
- 适合高频交易

#### FreqTradeMarketMaker
- 基础做市商实现
- 简单价差策略
- 适合学习和测试

### 🔲 网格交易策略

#### GridTradingStrategy
- 震荡市场网格交易
- 动态网格间距
- 波动率自适应
- 适合横盘行情

#### EnhancedGridStrategy
- 增强版网格策略
- 趋势识别优化
- 动态仓位管理

### 📊 其他策略

- **MeanReversionProfitStrategy**: 均值回归策略
- **TrendBreakoutStrategy**: 趋势突破策略
- **TrendFollowProfitStrategy**: 趋势跟踪策略
- **VolatilityIndicatorsStrategy**: 波动率指标策略

## 🔧 配置说明

### Docker 配置文件

主配置文件: `user_data/config_docker.json`

**现货交易配置**:
```json
{
  "trading_mode": "spot",
  "margin_mode": "",
  "pair_whitelist": ["BTC/USDT", "ETH/USDT"]
}
```

**永续合约配置** (资金费率套利):
```json
{
  "trading_mode": "futures",
  "margin_mode": "isolated",
  "pair_whitelist": ["BTC/USDT:USDT", "ETH/USDT:USDT"]
}
```

### 环境配置

- **demo**: 模拟交易环境（dry_run: true）
- **production**: 实盘交易环境（需配置真实 API）
- **custom**: 自定义环境配置

### 策略特定配置

每个策略可以在 `configs/strategies/` 下有独立配置文件，用于参数优化和版本管理。

## 📊 监控和分析

### Web UI 功能

访问 `http://localhost:8080` 可以:

- 查看实时交易状态
- 分析历史回测结果
- 监控持仓和盈亏
- 查看交易日志
- 管理策略配置

### 数据导出

- 回测结果: `user_data/backtest_results/`
- 交易数据: `*.sqlite` 数据库
- 日志文件: `user_data/logs/`

## 🔒 安全和风险管理

### 风险控制

1. **仓位管理**: 严格的仓位限制和动态调整
2. **止损机制**: 多层止损保护（硬止损、追踪止损、时间止损）
3. **风险监控**: 实时风险指标监控和告警
4. **回测验证**: 充分的历史数据回测

### 数据安全

1. **API 密钥**: 通过环境变量或加密配置管理
2. **敏感信息**: `.gitignore` 排除敏感文件
3. **权限控制**: API 权限最小化原则
4. **审计日志**: 完整的操作日志记录

### 交易安全

⚠️ **重要风险提示**:

1. **充分测试**: 实盘前务必进行充分回测和模拟交易
2. **小资金起步**: 建议从小资金($5K-$10K)开始验证
3. **严格风控**: 遵守止损纪律，不手动干预
4. **渐进式部署**: 根据实际表现逐步扩大规模
5. **监管合规**: 遵守当地法律法规

## 🛠 开发和扩展

### 策略开发

所有策略必须:

1. 继承 `freqtrade.strategy.IStrategy`
2. 实现必要方法: `populate_indicators()`, `populate_entry_trend()`, `populate_exit_trend()`
3. 放置在 `user_data/strategies/` 目录
4. 包含策略元数据（推荐）

**策略模板**:
```python
from freqtrade.strategy import IStrategy

class MyStrategy(IStrategy):
    """
    策略描述
    """
    INTERFACE_VERSION = 3

    # 策略元数据（可选）
    STRATEGY_METADATA = {
        "name": "MyStrategy",
        "version": "1.0.0",
        "author": "Your Name",
        "category": "trend_following",
        "description": "策略简要说明"
    }

    # 策略参数
    minimal_roi = {"0": 0.1}
    stoploss = -0.05
    timeframe = '15m'

    def populate_indicators(self, dataframe, metadata):
        # 计算技术指标
        return dataframe

    def populate_entry_trend(self, dataframe, metadata):
        # 生成入场信号
        return dataframe

    def populate_exit_trend(self, dataframe, metadata):
        # 生成出场信号
        return dataframe
```

### 本地开发工具

```bash
# Python 依赖管理
uv add <package>              # 添加依赖
uv add --dev <package>        # 添加开发依赖
uv sync                       # 同步依赖
uv lock                       # 锁定版本

# 进入虚拟环境
uv shell
```

### 回测和验证

```bash
# 1. 下载历史数据
./docker_download_data.sh --pairs 'BTC/USDT:USDT' --timeframe 15m --timerange 20240101-20241001

# 2. 运行回测
./docker_run_backtest.sh --strategy MyStrategy --timeframe 15m --timerange 20240101-20240901

# 3. 查看结果
ls -lh user_data/backtest_results/

# 4. 模拟交易测试
docker-compose up -d
# 访问 Web UI 监控
```

## 📚 文档资源

### 项目文档

- **[CLAUDE.md](CLAUDE.md)**: 项目架构和开发指南
- **[资金费率策略文档](user_data/strategies/FundingRateEnhancedStrategy_README.md)**: 详细的策略说明和使用指南

### FreqTrade 官方文档

- [FreqTrade 文档](https://www.freqtrade.io/en/stable/)
- [策略开发指南](https://www.freqtrade.io/en/stable/strategy-customization/)
- [永续合约配置](https://www.freqtrade.io/en/stable/leverage/)
- [回测指南](https://www.freqtrade.io/en/stable/backtesting/)

### 学习资源

- [Binance 资金费率说明](https://www.binance.com/en/support/faq/360033525031)
- [量化交易策略研究](https://arxiv.org/abs/2106.00168)
- [技术指标详解](https://www.investopedia.com/technical-analysis-4689657)

## 🛠 技术栈

- **Python**: 3.12+ (现代 Python 特性)
- **FreqTrade**: 2025.7+ (Docker 镜像)
- **Docker**: 容器化部署
- **SQLite**: 轻量级数据存储
- **uv**: 现代 Python 包管理
- **TA-Lib**: 技术分析库
- **ccxt**: 交易所连接库

## 📈 性能和优化

### 系统性能

- **回测速度**: 支持多线程回测加速
- **内存优化**: 流式数据处理，内存占用低
- **并发支持**: 多策略并行运行
- **缓存机制**: 智能数据缓存

### 策略优化

- **参数优化**: Hyperopt 自动参数寻优
- **性能分析**: 详细的回测指标分析
- **风险评估**: 多维度风险指标
- **持续改进**: 基于实盘数据优化

## 🤝 贡献指南

我们欢迎各种形式的贡献:

### 贡献方式

1. **Bug 报告**: 发现问题请创建 Issue，描述复现步骤
2. **功能建议**: 提出新策略想法或功能改进
3. **代码贡献**: 提交 Pull Request，遵循代码规范
4. **文档改进**: 完善文档、添加示例和教程
5. **策略分享**: 分享你的策略和优化经验

### 贡献流程

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-strategy`)
3. 提交变更 (`git commit -m 'feat: 添加新策略'`)
4. 推送到分支 (`git push origin feature/amazing-strategy`)
5. 创建 Pull Request

### 代码规范

- 遵循 PEP 8 Python 代码规范
- 使用 Conventional Commits 提交规范
- 添加必要的注释和文档字符串
- 确保所有测试通过

## 🔄 版本历史

### 最新更新 (2025-10-03)

- ✨ **新增**: U本位永续合约资金费率增强策略
- 📚 **文档**: 完整的策略使用文档和风险说明
- 🔧 **配置**: 支持永续合约交易模式
- 🎯 **优化**: Docker-first 架构优化

### 历史版本

- **v2.0**: Docker 化架构重构
- **v1.5**: VATSM 策略优化
- **v1.0**: 初始版本发布

## ⚠️ 免责声明

**重要风险提示**:

1. 本项目**仅供学习和研究使用**
2. 加密货币交易存在**极高风险**
3. **可能损失全部本金**
4. 不构成任何**投资建议**
5. 使用本项目产生的**盈亏由使用者自行承担**
6. 请充分理解策略逻辑和风险后再使用
7. 建议从**小资金开始测试**
8. 遵守当地**法律法规和监管要求**

**交易风险**:
- 市场波动风险
- 流动性风险
- 技术故障风险
- 交易所风险
- 监管政策风险

**建议**:
- 只用闲钱投资
- 充分回测验证
- 严格风险控制
- 保持理性决策
- 及时止损退出

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 🚀 快速链接

- 📖 [项目架构说明](CLAUDE.md)
- 🪙 [资金费率策略文档](user_data/strategies/FundingRateEnhancedStrategy_README.md)
- 🐳 [FreqTrade 官方文档](https://www.freqtrade.io/en/stable/)
- 💬 [问题反馈](https://github.com/your-repo/issues)

---

**让量化交易更简单、更安全、更高效！** 🚀

**记住: 风控第一，收益第二！** 🛡️
