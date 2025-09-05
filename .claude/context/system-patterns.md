---
created: 2025-09-05T03:22:30Z
last_updated: 2025-09-05T03:22:30Z
version: 1.0
author: Claude Code PM System
---

# System Patterns

## Architectural Style
**插件式微服务架构**
- 核心框架提供基础设施
- 策略作为独立插件动态加载
- 松耦合的模块化设计

## Design Patterns

### 1. Strategy Pattern（策略模式）
**位置**: `strategies/` 目录
- 每个交易策略继承基础策略接口
- 允许运行时切换不同策略
- 示例：VATSMStrategy, GridTradingStrategy

### 2. Registry Pattern（注册表模式）
**位置**: `freqbot/strategies/registry.py`
- 集中管理所有可用策略
- 动态注册和发现策略
- 提供策略元数据管理

### 3. Factory Pattern（工厂模式）
**位置**: `freqbot/strategies/loader.py`
- 动态创建策略实例
- 根据配置加载不同策略
- 处理策略初始化逻辑

### 4. Singleton Pattern（单例模式）
**位置**: `freqbot/config/manager.py`
- 配置管理器全局唯一实例
- 确保配置一致性
- 避免重复加载配置

### 5. Command Pattern（命令模式）
**位置**: `freqbot/cli.py`
- CLI命令封装为独立对象
- 支持命令撤销和重做
- 命令参数验证和处理

### 6. Observer Pattern（观察者模式）
**位置**: `freqbot/core/monitor.py`
- 监控系统订阅交易事件
- 实时推送状态更新
- 解耦监控和交易逻辑

## Data Flow Patterns

### 数据流向
```
市场数据 → 数据采集 → 策略分析 → 信号生成 → 订单执行 → 监控反馈
   ↑                                                        ↓
   └────────────────── 持续循环 ──────────────────────┘
```

### 数据处理管道
1. **数据获取层**
   - 交易所API调用
   - 数据标准化处理
   - 缓存机制

2. **策略计算层**
   - 技术指标计算
   - 信号生成逻辑
   - 风险评估

3. **执行层**
   - 订单管理
   - 仓位控制
   - 止损止盈

4. **监控层**
   - 性能跟踪
   - 风险监控
   - 日志记录

## Configuration Management

### 环境隔离模式
```
configs/
├── environments/
│   ├── demo/        # 演示环境
│   ├── test/        # 测试环境
│   └── production/  # 生产环境
```

### 配置继承链
```
默认配置 → 环境配置 → 策略配置 → 运行时参数
```

## Module Communication

### 同步通信
- 直接函数调用
- 返回值传递
- 异常处理链

### 异步通信
- FreqTrade事件系统
- 回调函数注册
- 消息队列（未来扩展）

## Error Handling Patterns

### 分层错误处理
1. **策略层**: 捕获计算错误，返回安全默认值
2. **执行层**: 处理交易异常，记录失败订单
3. **框架层**: 全局异常捕获，优雅降级

### 错误恢复策略
- 自动重试机制
- 熔断器模式（防止级联失败）
- 降级服务（只读模式）

## State Management

### 状态存储
- **内存状态**: 运行时变量
- **持久状态**: SQLite数据库
- **配置状态**: JSON文件

### 状态同步
- 事务性更新
- 版本控制
- 状态快照

## Security Patterns

### 密钥管理
- 环境变量隔离
- 配置文件加密
- 运行时解密

### 访问控制
- API限流
- 权限分级
- 审计日志

## Performance Optimization

### 缓存策略
- 市场数据缓存
- 计算结果缓存
- 配置缓存

### 并发处理
- 多策略并行执行
- 异步IO操作
- 线程池管理

## Extension Points

### 策略扩展点
- 继承BaseStrategy
- 实现required方法
- 注册到策略表

### 数据源扩展点
- 自定义数据提供器
- 新交易所接入
- 另类数据集成

### 监控扩展点
- 自定义指标
- 告警规则
- 报告生成

## Anti-Patterns to Avoid

### 避免的模式
- ❌ 硬编码配置值
- ❌ 紧耦合的模块
- ❌ 全局变量滥用
- ❌ 同步阻塞调用
- ❌ 忽略错误处理

### 最佳实践
- ✅ 配置外部化
- ✅ 依赖注入
- ✅ 明确的接口定义
- ✅ 异步非阻塞
- ✅ 完善的错误处理