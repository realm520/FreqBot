"""网格策略专用监控指标

提供网格交易策略的关键性能指标收集和计算功能，包括：
- 网格活跃度和成交统计
- 仓位管理和调整频率
- 风险指标和止损触发
- 市场状态和策略性能
"""

import time
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from collections import deque, defaultdict
from dataclasses import dataclass, field
import threading

logger = logging.getLogger(__name__)


@dataclass
class GridLevel:
    """网格层级数据"""
    price: float
    side: str  # 'buy' or 'sell'
    filled: bool = False
    fill_time: Optional[datetime] = None
    amount: float = 0.0
    
    
@dataclass
class GridMetricsData:
    """网格指标数据容器"""
    # 网格状态指标
    total_grid_levels: int = 0
    active_grid_levels: int = 0
    filled_grid_levels: int = 0
    grid_fill_rate: float = 0.0
    
    # 交易统计
    grid_trades_count: int = 0
    grid_trades_profit: float = 0.0
    avg_trade_profit: float = 0.0
    profitable_trades_count: int = 0
    loss_trades_count: int = 0
    
    # 仓位管理指标
    current_position_size: float = 0.0
    max_position_size: float = 0.0
    position_utilization: float = 0.0
    position_adjustments_count: int = 0
    
    # 风险指标
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    risk_exposure: float = 0.0
    stop_loss_triggers: int = 0
    
    # 市场状态指标
    market_regime: str = "unknown"  # trend_up, trend_down, sideways
    volatility_level: str = "normal"  # low, normal, high
    grid_efficiency: float = 0.0
    
    # 性能指标
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    
    # 时间戳
    timestamp: datetime = field(default_factory=datetime.now)


class GridMetrics:
    """网格策略监控指标收集器
    
    提供低开销的实时监控功能，支持Prometheus/Grafana集成
    """
    
    def __init__(self, strategy_name: str = "grid_strategy", max_history: int = 1000):
        """初始化监控指标收集器
        
        Args:
            strategy_name: 策略名称
            max_history: 最大历史记录数量
        """
        self.strategy_name = strategy_name
        self.max_history = max_history
        
        # 指标数据
        self.current_metrics = GridMetricsData()
        self.metrics_history: deque = deque(maxlen=max_history)
        
        # 网格层级追踪
        self.grid_levels: Dict[float, GridLevel] = {}
        
        # 交易历史
        self.trade_history: deque = deque(maxlen=max_history)
        
        # 性能计算相关
        self.peak_value = 0.0
        self.initial_balance = 0.0
        self.returns_series: deque = deque(maxlen=max_history)
        
        # 统计计数器
        self.counters = defaultdict(int)
        
        # 线程锁
        self._lock = threading.Lock()
        
        # 启动时间
        self.start_time = datetime.now()
        
        logger.info(f"网格监控指标已初始化: {strategy_name}")
    
    def update_grid_levels(self, levels: List[Dict[str, Any]]):
        """更新网格层级信息
        
        Args:
            levels: 网格层级列表，每个包含 {price, side, filled, amount}
        """
        with self._lock:
            self.grid_levels.clear()
            
            total_levels = len(levels)
            filled_levels = 0
            active_levels = 0
            
            for level_data in levels:
                price = level_data['price']
                level = GridLevel(
                    price=price,
                    side=level_data['side'],
                    filled=level_data.get('filled', False),
                    amount=level_data.get('amount', 0.0)
                )
                
                if level.filled:
                    filled_levels += 1
                    if level.fill_time is None:
                        level.fill_time = datetime.now()
                else:
                    active_levels += 1
                    
                self.grid_levels[price] = level
            
            # 更新网格指标
            self.current_metrics.total_grid_levels = total_levels
            self.current_metrics.active_grid_levels = active_levels  
            self.current_metrics.filled_grid_levels = filled_levels
            self.current_metrics.grid_fill_rate = (filled_levels / total_levels * 100) if total_levels > 0 else 0.0
    
    def record_trade(self, trade_data: Dict[str, Any]):
        """记录网格交易
        
        Args:
            trade_data: 交易数据，包含 {side, amount, price, profit, timestamp}
        """
        with self._lock:
            trade_data['timestamp'] = trade_data.get('timestamp', datetime.now())
            self.trade_history.append(trade_data)
            
            # 更新交易统计
            self.counters['total_trades'] += 1
            self.current_metrics.grid_trades_count = self.counters['total_trades']
            
            profit = trade_data.get('profit', 0.0)
            self.current_metrics.grid_trades_profit += profit
            
            if profit > 0:
                self.counters['profitable_trades'] += 1
                self.current_metrics.profitable_trades_count = self.counters['profitable_trades']
            elif profit < 0:
                self.counters['loss_trades'] += 1
                self.current_metrics.loss_trades_count = self.counters['loss_trades']
            
            # 更新平均交易收益
            if self.current_metrics.grid_trades_count > 0:
                self.current_metrics.avg_trade_profit = (
                    self.current_metrics.grid_trades_profit / self.current_metrics.grid_trades_count
                )
            
            # 更新胜率
            if self.current_metrics.grid_trades_count > 0:
                self.current_metrics.win_rate = (
                    self.current_metrics.profitable_trades_count / self.current_metrics.grid_trades_count * 100
                )
    
    def update_position(self, current_size: float, max_allowed: float):
        """更新仓位信息
        
        Args:
            current_size: 当前仓位大小
            max_allowed: 最大允许仓位
        """
        with self._lock:
            self.current_metrics.current_position_size = current_size
            
            if current_size > self.current_metrics.max_position_size:
                self.current_metrics.max_position_size = current_size
            
            # 计算仓位利用率
            if max_allowed > 0:
                self.current_metrics.position_utilization = (current_size / max_allowed * 100)
    
    def record_position_adjustment(self):
        """记录仓位调整事件"""
        with self._lock:
            self.counters['position_adjustments'] += 1
            self.current_metrics.position_adjustments_count = self.counters['position_adjustments']
    
    def update_risk_metrics(self, current_balance: float, max_risk_exposure: float):
        """更新风险指标
        
        Args:
            current_balance: 当前余额
            max_risk_exposure: 最大风险敞口
        """
        with self._lock:
            if self.initial_balance == 0:
                self.initial_balance = current_balance
                self.peak_value = current_balance
            
            # 更新峰值
            if current_balance > self.peak_value:
                self.peak_value = current_balance
            
            # 计算回撤
            if self.peak_value > 0:
                current_drawdown = (self.peak_value - current_balance) / self.peak_value * 100
                self.current_metrics.current_drawdown = current_drawdown
                
                if current_drawdown > self.current_metrics.max_drawdown:
                    self.current_metrics.max_drawdown = current_drawdown
            
            # 计算风险敞口
            self.current_metrics.risk_exposure = max_risk_exposure
            
            # 计算总收益率
            if self.initial_balance > 0:
                self.current_metrics.total_return = (
                    (current_balance - self.initial_balance) / self.initial_balance * 100
                )
            
            # 记录收益率序列用于夏普比率计算
            if len(self.returns_series) > 0:
                last_balance = self.returns_series[-1] if self.returns_series else self.initial_balance
                if last_balance > 0:
                    period_return = (current_balance - last_balance) / last_balance
                    self.returns_series.append(period_return)
            else:
                self.returns_series.append(0.0)
    
    def record_stop_loss_trigger(self):
        """记录止损触发事件"""
        with self._lock:
            self.counters['stop_loss_triggers'] += 1
            self.current_metrics.stop_loss_triggers = self.counters['stop_loss_triggers']
    
    def update_market_regime(self, regime: str, volatility: str, efficiency: float = 0.0):
        """更新市场状态指标
        
        Args:
            regime: 市场状态 (trend_up, trend_down, sideways)
            volatility: 波动率水平 (low, normal, high)  
            efficiency: 网格效率 (0-100)
        """
        with self._lock:
            self.current_metrics.market_regime = regime
            self.current_metrics.volatility_level = volatility
            self.current_metrics.grid_efficiency = efficiency
    
    def calculate_performance_metrics(self):
        """计算性能指标"""
        with self._lock:
            # 计算夏普比率
            if len(self.returns_series) > 1:
                try:
                    import numpy as np
                    returns = list(self.returns_series)
                    if len(returns) > 0:
                        mean_return = np.mean(returns)
                        std_return = np.std(returns)
                        
                        if std_return > 0:
                            # 假设无风险利率为0，年化为252个交易日
                            self.current_metrics.sharpe_ratio = (mean_return / std_return) * np.sqrt(252)
                except ImportError:
                    # 如果numpy不可用，跳过夏普比率计算
                    pass
            
            # 计算盈亏比
            if self.current_metrics.loss_trades_count > 0 and self.current_metrics.profitable_trades_count > 0:
                avg_win = (self.current_metrics.grid_trades_profit / 
                          self.current_metrics.profitable_trades_count if self.current_metrics.profitable_trades_count > 0 else 0)
                avg_loss = abs(self.current_metrics.grid_trades_profit / 
                              self.current_metrics.loss_trades_count if self.current_metrics.loss_trades_count > 0 else 1)
                
                if avg_loss > 0:
                    self.current_metrics.profit_factor = avg_win / avg_loss
    
    def get_current_metrics(self) -> GridMetricsData:
        """获取当前指标快照"""
        with self._lock:
            # 计算性能指标
            self.calculate_performance_metrics()
            
            # 更新时间戳
            self.current_metrics.timestamp = datetime.now()
            
            return self.current_metrics
    
    def get_metrics_dict(self) -> Dict[str, Any]:
        """获取指标字典格式（用于Prometheus导出）"""
        metrics = self.get_current_metrics()
        
        return {
            # 网格指标
            f"grid_total_levels_{self.strategy_name}": metrics.total_grid_levels,
            f"grid_active_levels_{self.strategy_name}": metrics.active_grid_levels,
            f"grid_filled_levels_{self.strategy_name}": metrics.filled_grid_levels,
            f"grid_fill_rate_{self.strategy_name}": metrics.grid_fill_rate,
            
            # 交易指标  
            f"grid_trades_total_{self.strategy_name}": metrics.grid_trades_count,
            f"grid_trades_profit_{self.strategy_name}": metrics.grid_trades_profit,
            f"grid_avg_trade_profit_{self.strategy_name}": metrics.avg_trade_profit,
            f"grid_profitable_trades_{self.strategy_name}": metrics.profitable_trades_count,
            f"grid_loss_trades_{self.strategy_name}": metrics.loss_trades_count,
            f"grid_win_rate_{self.strategy_name}": metrics.win_rate,
            
            # 仓位指标
            f"position_current_size_{self.strategy_name}": metrics.current_position_size,
            f"position_max_size_{self.strategy_name}": metrics.max_position_size,
            f"position_utilization_{self.strategy_name}": metrics.position_utilization,
            f"position_adjustments_{self.strategy_name}": metrics.position_adjustments_count,
            
            # 风险指标
            f"max_drawdown_{self.strategy_name}": metrics.max_drawdown,
            f"current_drawdown_{self.strategy_name}": metrics.current_drawdown,
            f"risk_exposure_{self.strategy_name}": metrics.risk_exposure,
            f"stop_loss_triggers_{self.strategy_name}": metrics.stop_loss_triggers,
            
            # 性能指标
            f"total_return_{self.strategy_name}": metrics.total_return,
            f"sharpe_ratio_{self.strategy_name}": metrics.sharpe_ratio,
            f"profit_factor_{self.strategy_name}": metrics.profit_factor,
            f"grid_efficiency_{self.strategy_name}": metrics.grid_efficiency,
            
            # 状态标签
            f"market_regime_{self.strategy_name}": metrics.market_regime,
            f"volatility_level_{self.strategy_name}": metrics.volatility_level,
        }
    
    def get_summary_report(self) -> str:
        """获取监控摘要报告"""
        metrics = self.get_current_metrics()
        runtime = datetime.now() - self.start_time
        
        report = f"""
=== 网格策略监控报告 ({self.strategy_name}) ===
运行时长: {runtime}
更新时间: {metrics.timestamp}

📊 网格状态:
  - 总网格层级: {metrics.total_grid_levels}
  - 活跃层级: {metrics.active_grid_levels} 
  - 已成交层级: {metrics.filled_grid_levels}
  - 成交率: {metrics.grid_fill_rate:.2f}%

💰 交易统计:
  - 网格交易数: {metrics.grid_trades_count}
  - 总收益: {metrics.grid_trades_profit:.6f}
  - 平均收益: {metrics.avg_trade_profit:.6f}
  - 胜率: {metrics.win_rate:.2f}%
  - 盈亏比: {metrics.profit_factor:.2f}

📈 仓位管理:
  - 当前仓位: {metrics.current_position_size:.6f}
  - 最大仓位: {metrics.max_position_size:.6f}
  - 仓位利用率: {metrics.position_utilization:.2f}%
  - 调整次数: {metrics.position_adjustments_count}

⚠️  风险指标:
  - 总收益率: {metrics.total_return:.2f}%
  - 最大回撤: {metrics.max_drawdown:.2f}%
  - 当前回撤: {metrics.current_drawdown:.2f}%
  - 夏普比率: {metrics.sharpe_ratio:.2f}
  - 止损触发: {metrics.stop_loss_triggers}次

🏪 市场状态:
  - 市场状态: {metrics.market_regime}
  - 波动率: {metrics.volatility_level}
  - 网格效率: {metrics.grid_efficiency:.2f}%
"""
        return report
    
    def save_history_snapshot(self):
        """保存当前指标到历史记录"""
        with self._lock:
            self.metrics_history.append(self.get_current_metrics())
    
    def reset_metrics(self):
        """重置所有指标（用于新的交易周期）"""
        with self._lock:
            self.current_metrics = GridMetricsData()
            self.grid_levels.clear()
            self.trade_history.clear()
            self.returns_series.clear()
            self.counters.clear()
            self.peak_value = 0.0
            self.initial_balance = 0.0
            self.start_time = datetime.now()
            
            logger.info(f"网格监控指标已重置: {self.strategy_name}")
    
    def export_prometheus_metrics(self, output_file: str):
        """导出Prometheus格式的指标"""
        metrics_dict = self.get_metrics_dict()
        
        with open(output_file, 'w') as f:
            f.write("# HELP grid_strategy_metrics Grid trading strategy metrics\n")
            f.write("# TYPE grid_strategy_metrics gauge\n")
            
            for metric_name, value in metrics_dict.items():
                if isinstance(value, (int, float)):
                    f.write(f"{metric_name} {value}\n")
                else:
                    # 对于字符串标签，转为数值编码
                    f.write(f"{metric_name} 1\n")
        
        logger.info(f"Prometheus指标已导出到: {output_file}")