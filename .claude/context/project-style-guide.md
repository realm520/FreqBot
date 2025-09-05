---
created: 2025-09-05T03:22:30Z
last_updated: 2025-09-05T03:22:30Z
version: 1.0
author: Claude Code PM System
---

# Project Style Guide

## Coding Standards

### Python Style

#### PEP 8 Compliance
- **缩进**: 4个空格
- **行长度**: 最大120字符
- **空行**: 类之间2行，方法之间1行
- **导入**: 按标准库、第三方库、本地模块分组

#### 命名规范
```python
# 类名：PascalCase
class TradingStrategy:
    pass

# 函数和方法：snake_case
def calculate_position_size():
    pass

# 常量：UPPER_SNAKE_CASE
MAX_POSITION_SIZE = 100000

# 私有方法：前置单下划线
def _internal_method():
    pass

# 模块级变量：snake_case
default_config = {}
```

#### 类型提示
```python
from typing import List, Dict, Optional, Union

def execute_trade(
    symbol: str,
    quantity: float,
    price: Optional[float] = None
) -> Dict[str, Union[str, float]]:
    """执行交易订单"""
    pass
```

### 文档规范

#### Docstring格式
```python
def complex_function(param1: str, param2: int) -> bool:
    """
    函数简要描述。

    详细描述（如果需要）。

    Args:
        param1: 参数1的描述
        param2: 参数2的描述

    Returns:
        返回值的描述

    Raises:
        ValueError: 当参数无效时

    Example:
        >>> complex_function("test", 42)
        True
    """
    pass
```

#### 注释规范
```python
# 单行注释：解释为什么，而不是做什么
risk_factor = 0.02  # 2%风险因子，基于Kelly公式

# 多行注释：用于复杂逻辑说明
# 这里使用动态时间规整(DTW)算法来比较两个时间序列
# 因为传统的欧几里得距离在处理相位差异时效果不佳
# DTW可以找到最佳的对齐方式

# TODO注释：标记待办事项
# TODO: 优化这个O(n^2)算法
# FIXME: 修复内存泄漏问题
# NOTE: 这里假设数据已经标准化
```

## File Structure Patterns

### 项目文件组织
```
module/
├── __init__.py      # 模块初始化和导出
├── core.py          # 核心功能实现
├── models.py        # 数据模型定义
├── utils.py         # 辅助功能
├── constants.py     # 常量定义
└── exceptions.py    # 自定义异常
```

### 策略文件模板
```python
"""
策略名称: MyStrategy
作者: 开发者名
版本: 1.0.0
描述: 策略的详细描述
"""

from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta

class MyStrategy(IStrategy):
    """策略实现类"""
    
    # 策略参数
    INTERFACE_VERSION = 3
    
    # 最小ROI设置
    minimal_roi = {
        "0": 0.10,
        "10": 0.05,
        "20": 0.01
    }
    
    # 止损设置
    stoploss = -0.10
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """添加技术指标"""
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """定义入场条件"""
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """定义出场条件"""
        return dataframe
```

## Comment Style

### 中英文注释规范
- **代码内注释**: 使用中文，便于本地开发者理解
- **API文档**: 使用英文，保持国际化兼容
- **提交信息**: 中文描述+英文类型标记

### 注释示例
```python
class OrderManager:
    """订单管理器（Order Manager）
    
    负责处理所有交易订单的生命周期管理。
    Manages the lifecycle of all trading orders.
    """
    
    def place_order(self, symbol: str, quantity: float):
        """下单
        
        Args:
            symbol: 交易对符号 (Trading pair symbol)
            quantity: 下单数量 (Order quantity)
        """
        # 验证订单参数
        self._validate_order(symbol, quantity)
        
        # 检查账户余额是否充足
        if not self._check_balance(quantity):
            raise InsufficientFundsError("余额不足")
```

## Version Control

### Git提交规范

#### 提交信息格式
```
<type>: <subject>

<body>

<footer>
```

#### 类型标记
- **feat**: 新功能
- **fix**: 修复bug
- **docs**: 文档更新
- **style**: 代码格式调整
- **refactor**: 重构代码
- **test**: 测试相关
- **chore**: 构建或辅助工具变动

#### 提交示例
```
feat: 添加网格交易策略

- 实现基础网格逻辑
- 支持动态网格调整
- 添加回测测试用例

Closes #123
```

### 分支策略
```
main          # 主分支，稳定版本
├── develop   # 开发分支
├── feature/* # 功能分支
├── fix/*     # 修复分支
└── release/* # 发布分支
```

## Error Handling

### 异常处理规范
```python
# 具体异常捕获
try:
    result = risky_operation()
except ValueError as e:
    logger.error(f"参数错误: {e}")
    # 处理特定错误
except Exception as e:
    logger.exception("未预期的错误")
    # 记录完整堆栈
    raise  # 重新抛出

# 自定义异常
class TradingError(Exception):
    """交易异常基类"""
    pass

class InsufficientFundsError(TradingError):
    """余额不足异常"""
    pass
```

### 日志规范
```python
import logging

logger = logging.getLogger(__name__)

# 日志级别使用
logger.debug("调试信息：变量值 = %s", value)
logger.info("重要操作：开始执行交易")
logger.warning("警告：接近风险限制")
logger.error("错误：订单执行失败")
logger.critical("严重：系统异常，需要立即处理")
```

## Testing Standards

### 测试文件组织
```
tests/
├── unit/           # 单元测试
├── integration/    # 集成测试
├── fixtures/       # 测试数据
└── conftest.py    # pytest配置
```

### 测试命名规范
```python
def test_should_calculate_correct_position_size():
    """测试应该正确计算仓位大小"""
    pass

def test_raises_error_when_invalid_input():
    """测试当输入无效时应抛出错误"""
    pass
```

## Configuration Files

### JSON配置格式
```json
{
    "strategy": "GridTradingStrategy",
    "stake_currency": "USDT",
    "stake_amount": 100,
    "dry_run": true,
    "exchange": {
        "name": "binance",
        "key": "",
        "secret": ""
    }
}
```

### YAML配置格式
```yaml
strategy:
  name: GridTradingStrategy
  params:
    grid_levels: 10
    grid_spacing: 0.01
    
risk_management:
  max_position_size: 10000
  stop_loss: -0.05
```

## Import Order

### 导入顺序规范
```python
# 1. 标准库
import os
import sys
from datetime import datetime

# 2. 第三方库
import pandas as pd
import numpy as np
from freqtrade.strategy import IStrategy

# 3. 本地模块
from freqbot.core import Engine
from freqbot.strategies import BaseStrategy
from .utils import calculate_risk
```

## Best Practices

### DO's ✅
- 使用描述性变量名
- 保持函数单一职责
- 编写单元测试
- 处理边界情况
- 记录重要决策
- 使用类型提示
- 遵循DRY原则

### DON'Ts ❌
- 避免魔法数字
- 不要忽略异常
- 避免过深嵌套
- 不要过度优化
- 避免全局变量
- 不要硬编码密钥
- 避免过长函数

## Code Review Checklist

### 审查要点
- [ ] 代码风格符合规范
- [ ] 有充分的测试覆盖
- [ ] 文档和注释完整
- [ ] 没有明显的性能问题
- [ ] 错误处理适当
- [ ] 没有安全漏洞
- [ ] 遵循设计模式