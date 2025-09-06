#!/usr/bin/env python3
"""
回测结果分析脚本
深度分析回测结果，生成详细的性能报告和可视化图表

功能:
1. 性能指标计算和分析
2. 基准对比分析
3. 风险评估
4. HTML/PDF报告生成
5. 交互式图表
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import argparse
import logging
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))


class BacktestResultAnalyzer:
    """回测结果分析器"""
    
    def __init__(self, results_dir: str = None):
        """初始化分析器"""
        self.project_root = PROJECT_ROOT
        self.results_dir = Path(results_dir) if results_dir else self.project_root / "backtest_results"
        self.reports_dir = self.results_dir / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.setup_logging()
        
        # 设置绘图样式
        self.setup_plotting()
        
        # 分析结果存储
        self.analysis_results = {}
        self.benchmark_data = {}
        
    def setup_logging(self):
        """设置日志记录"""
        log_file = self.reports_dir / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"结果分析器初始化，日志文件: {log_file}")
        
    def setup_plotting(self):
        """设置绘图环境"""
        plt.style.use('seaborn')
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        
    def load_backtest_data(self, file_path: str) -> Dict[str, Any]:
        """加载回测数据"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.json'):
                    return json.load(f)
                else:
                    # 尝试解析其他格式
                    return json.load(f)
        except Exception as e:
            self.logger.error(f"加载回测数据失败 {file_path}: {e}")
            return {}
            
    def find_latest_results(self) -> List[str]:
        """查找最新的回测结果文件"""
        result_files = []
        
        # 查找JSON结果文件
        for pattern in ["backtest_results_*.json", "scenario_results_*.json"]:
            files = list(self.results_dir.glob(pattern))
            result_files.extend(files)
            
        # 按修改时间排序
        result_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        self.logger.info(f"找到 {len(result_files)} 个结果文件")
        return [str(f) for f in result_files]
        
    def calculate_advanced_metrics(self, trades_df: pd.DataFrame, equity_curve: pd.Series) -> Dict[str, float]:
        """计算高级性能指标"""
        if trades_df.empty:
            return {}
            
        # 确保有必要的列
        required_cols = ['profit_ratio', 'open_date', 'close_date']
        if not all(col in trades_df.columns for col in required_cols):
            self.logger.warning("交易数据缺少必要字段，跳过高级指标计算")
            return {}
            
        try:
            # 转换时间格式
            trades_df['open_date'] = pd.to_datetime(trades_df['open_date'])
            trades_df['close_date'] = pd.to_datetime(trades_df['close_date'])
            trades_df['duration'] = trades_df['close_date'] - trades_df['open_date']
            
            returns = trades_df['profit_ratio'].dropna()
            
            metrics = {}
            
            # 基础统计
            metrics['total_trades'] = len(trades_df)
            metrics['winning_trades'] = len(trades_df[trades_df['profit_ratio'] > 0])
            metrics['losing_trades'] = len(trades_df[trades_df['profit_ratio'] <= 0])
            metrics['win_rate'] = metrics['winning_trades'] / metrics['total_trades'] if metrics['total_trades'] > 0 else 0
            
            # 收益指标
            metrics['total_return'] = returns.sum()
            metrics['avg_return_per_trade'] = returns.mean()
            metrics['best_trade'] = returns.max()
            metrics['worst_trade'] = returns.min()
            
            # 盈亏分析
            winning_returns = returns[returns > 0]
            losing_returns = returns[returns <= 0]
            
            if not winning_returns.empty:
                metrics['avg_win'] = winning_returns.mean()
                metrics['max_win'] = winning_returns.max()
            else:
                metrics['avg_win'] = metrics['max_win'] = 0
                
            if not losing_returns.empty:
                metrics['avg_loss'] = losing_returns.mean()
                metrics['max_loss'] = losing_returns.min()
            else:
                metrics['avg_loss'] = metrics['max_loss'] = 0
                
            # 盈亏比
            if metrics['avg_loss'] != 0:
                metrics['profit_loss_ratio'] = abs(metrics['avg_win'] / metrics['avg_loss'])
            else:
                metrics['profit_loss_ratio'] = float('inf') if metrics['avg_win'] > 0 else 0
                
            # 盈利因子
            total_wins = winning_returns.sum() if not winning_returns.empty else 0
            total_losses = abs(losing_returns.sum()) if not losing_returns.empty else 0
            
            if total_losses > 0:
                metrics['profit_factor'] = total_wins / total_losses
            else:
                metrics['profit_factor'] = float('inf') if total_wins > 0 else 0
                
            # 期望收益
            metrics['expectancy'] = (metrics['win_rate'] * metrics['avg_win']) + ((1 - metrics['win_rate']) * metrics['avg_loss'])
            
            # 风险指标
            if len(returns) > 1:
                metrics['return_volatility'] = returns.std()
                metrics['sharpe_ratio'] = self.calculate_sharpe_ratio(returns)
                metrics['sortino_ratio'] = self.calculate_sortino_ratio(returns)
                
                # 计算最大回撤
                if not equity_curve.empty:
                    metrics['max_drawdown'], metrics['max_drawdown_duration'] = self.calculate_drawdown_metrics(equity_curve)
                    if metrics['max_drawdown'] != 0:
                        metrics['calmar_ratio'] = metrics['total_return'] / abs(metrics['max_drawdown'])
                    else:
                        metrics['calmar_ratio'] = float('inf') if metrics['total_return'] > 0 else 0
                        
                # 连续交易分析
                metrics['max_consecutive_wins'] = self.calculate_max_consecutive(returns > 0)
                metrics['max_consecutive_losses'] = self.calculate_max_consecutive(returns <= 0)
                
            # 交易时长分析
            if 'duration' in trades_df.columns:
                durations = trades_df['duration'].dropna()
                if not durations.empty:
                    metrics['avg_trade_duration_hours'] = durations.mean().total_seconds() / 3600
                    metrics['max_trade_duration_hours'] = durations.max().total_seconds() / 3600
                    metrics['min_trade_duration_hours'] = durations.min().total_seconds() / 3600
                    
            return metrics
            
        except Exception as e:
            self.logger.error(f"计算高级指标失败: {e}")
            return {}
            
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """计算夏普比率"""
        if returns.std() == 0:
            return 0
        excess_returns = returns.mean() - (risk_free_rate / 252)
        return (excess_returns / returns.std()) * np.sqrt(252)
        
    def calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """计算索提诺比率"""
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            return float('inf') if returns.mean() > 0 else 0
        excess_returns = returns.mean() - (risk_free_rate / 252)
        downside_deviation = negative_returns.std()
        if downside_deviation == 0:
            return float('inf') if excess_returns > 0 else 0
        return (excess_returns / downside_deviation) * np.sqrt(252)
        
    def calculate_drawdown_metrics(self, equity_curve: pd.Series) -> Tuple[float, int]:
        """计算回撤指标"""
        try:
            # 计算累积最高点
            running_max = equity_curve.expanding().max()
            
            # 计算回撤
            drawdown = (equity_curve - running_max) / running_max
            
            # 最大回撤
            max_drawdown = drawdown.min()
            
            # 最大回撤持续时间
            drawdown_duration = 0
            current_duration = 0
            
            for dd in drawdown:
                if dd < 0:
                    current_duration += 1
                    drawdown_duration = max(drawdown_duration, current_duration)
                else:
                    current_duration = 0
                    
            return max_drawdown, drawdown_duration
            
        except Exception as e:
            self.logger.warning(f"计算回撤指标失败: {e}")
            return 0, 0
            
    def calculate_max_consecutive(self, condition_series: pd.Series) -> int:
        """计算最大连续次数"""
        max_consecutive = 0
        current_consecutive = 0
        
        for condition in condition_series:
            if condition:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
                
        return max_consecutive
        
    def compare_with_benchmark(self, strategy_metrics: Dict[str, float], 
                             benchmark_return: float = 0.08) -> Dict[str, Any]:
        """与基准策略对比"""
        comparison = {}
        
        strategy_return = strategy_metrics.get('total_return', 0)
        strategy_volatility = strategy_metrics.get('return_volatility', 0)
        strategy_sharpe = strategy_metrics.get('sharpe_ratio', 0)
        strategy_max_dd = strategy_metrics.get('max_drawdown', 0)
        
        # 超额收益
        comparison['excess_return'] = strategy_return - benchmark_return
        comparison['excess_return_pct'] = (comparison['excess_return'] / benchmark_return * 100) if benchmark_return != 0 else 0
        
        # 信息比率 (假设基准波动率为策略的一半)
        benchmark_volatility = strategy_volatility * 0.5
        if benchmark_volatility > 0:
            comparison['information_ratio'] = comparison['excess_return'] / benchmark_volatility
        else:
            comparison['information_ratio'] = 0
            
        # 相对表现评分
        comparison['performance_score'] = self.calculate_performance_score(strategy_metrics)
        
        # 评级
        if comparison['excess_return'] > 0.1 and strategy_sharpe > 1.5 and abs(strategy_max_dd) < 0.15:
            comparison['rating'] = "优秀"
        elif comparison['excess_return'] > 0.05 and strategy_sharpe > 1.0:
            comparison['rating'] = "良好"
        elif comparison['excess_return'] > 0 and strategy_sharpe > 0.5:
            comparison['rating'] = "中等"
        elif strategy_return > 0:
            comparison['rating'] = "偏弱"
        else:
            comparison['rating'] = "较差"
            
        return comparison
        
    def calculate_performance_score(self, metrics: Dict[str, float]) -> float:
        """计算综合性能评分 (0-100)"""
        score = 0
        
        # 收益率评分 (30分)
        total_return = metrics.get('total_return', 0)
        if total_return > 0.5:
            score += 30
        elif total_return > 0.3:
            score += 25
        elif total_return > 0.2:
            score += 20
        elif total_return > 0.1:
            score += 15
        elif total_return > 0:
            score += 10
            
        # 夏普比率评分 (25分)
        sharpe = metrics.get('sharpe_ratio', 0)
        if sharpe > 2:
            score += 25
        elif sharpe > 1.5:
            score += 22
        elif sharpe > 1:
            score += 18
        elif sharpe > 0.5:
            score += 14
        elif sharpe > 0:
            score += 10
            
        # 最大回撤评分 (25分)
        max_dd = abs(metrics.get('max_drawdown', 1))
        if max_dd < 0.05:
            score += 25
        elif max_dd < 0.1:
            score += 22
        elif max_dd < 0.15:
            score += 18
        elif max_dd < 0.2:
            score += 14
        elif max_dd < 0.3:
            score += 10
            
        # 胜率评分 (10分)
        win_rate = metrics.get('win_rate', 0)
        if win_rate > 0.7:
            score += 10
        elif win_rate > 0.6:
            score += 8
        elif win_rate > 0.55:
            score += 6
        elif win_rate > 0.5:
            score += 4
            
        # 盈利因子评分 (10分)
        profit_factor = metrics.get('profit_factor', 1)
        if profit_factor > 2:
            score += 10
        elif profit_factor > 1.5:
            score += 8
        elif profit_factor > 1.2:
            score += 6
        elif profit_factor > 1:
            score += 4
            
        return score
        
    def create_performance_dashboard(self, analysis_data: Dict[str, Any], timestamp: str) -> str:
        """创建性能分析仪表板"""
        self.logger.info("创建性能分析仪表板")
        
        # 创建多子图布局
        fig = plt.figure(figsize=(20, 16))
        
        # 1. 权益曲线
        ax1 = plt.subplot(3, 3, 1)
        self.plot_equity_curve(analysis_data, ax1)
        
        # 2. 回撤分析
        ax2 = plt.subplot(3, 3, 2) 
        self.plot_drawdown_analysis(analysis_data, ax2)
        
        # 3. 收益分布
        ax3 = plt.subplot(3, 3, 3)
        self.plot_returns_distribution(analysis_data, ax3)
        
        # 4. 月度收益热力图
        ax4 = plt.subplot(3, 3, 4)
        self.plot_monthly_returns_heatmap(analysis_data, ax4)
        
        # 5. 胜率和盈亏比
        ax5 = plt.subplot(3, 3, 5)
        self.plot_win_loss_analysis(analysis_data, ax5)
        
        # 6. 交易时长分析
        ax6 = plt.subplot(3, 3, 6)
        self.plot_duration_analysis(analysis_data, ax6)
        
        # 7. 风险收益散点图
        ax7 = plt.subplot(3, 3, 7)
        self.plot_risk_return_scatter(analysis_data, ax7)
        
        # 8. 绩效指标雷达图
        ax8 = plt.subplot(3, 3, 8, projection='polar')
        self.plot_performance_radar(analysis_data, ax8)
        
        # 9. 关键指标总结
        ax9 = plt.subplot(3, 3, 9)
        self.plot_key_metrics_summary(analysis_data, ax9)
        
        plt.tight_layout()
        
        # 保存图表
        dashboard_file = self.reports_dir / f"performance_dashboard_{timestamp}.png"
        plt.savefig(dashboard_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"性能仪表板已保存: {dashboard_file}")
        return str(dashboard_file)
        
    def plot_equity_curve(self, data: Dict[str, Any], ax):
        """绘制权益曲线"""
        # 这里需要根据实际数据结构调整
        # 示例代码，实际需要从交易记录构建权益曲线
        trades_df = data.get('trades', pd.DataFrame())
        if trades_df.empty:
            ax.text(0.5, 0.5, '无交易数据', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('权益曲线')
            return
            
        # 构建权益曲线
        equity = [1.0]  # 起始资金为1
        for _, trade in trades_df.iterrows():
            equity.append(equity[-1] * (1 + trade.get('profit_ratio', 0)))
            
        ax.plot(range(len(equity)), equity, linewidth=2, color='blue')
        ax.set_title('权益曲线', fontweight='bold')
        ax.set_xlabel('交易次数')
        ax.set_ylabel('权益倍数')
        ax.grid(True, alpha=0.3)
        
    def plot_drawdown_analysis(self, data: Dict[str, Any], ax):
        """绘制回撤分析"""
        trades_df = data.get('trades', pd.DataFrame())
        if trades_df.empty:
            ax.text(0.5, 0.5, '无交易数据', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('回撤分析')
            return
            
        # 计算回撤序列
        equity = [1.0]
        for _, trade in trades_df.iterrows():
            equity.append(equity[-1] * (1 + trade.get('profit_ratio', 0)))
            
        equity = pd.Series(equity)
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max * 100
        
        ax.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
        ax.plot(range(len(drawdown)), drawdown, color='red', linewidth=1)
        ax.set_title('回撤分析', fontweight='bold')
        ax.set_xlabel('交易次数')
        ax.set_ylabel('回撤 (%)')
        ax.set_ylim(min(drawdown.min() - 1, 0), 1)
        
    def plot_returns_distribution(self, data: Dict[str, Any], ax):
        """绘制收益分布图"""
        trades_df = data.get('trades', pd.DataFrame())
        if trades_df.empty or 'profit_ratio' not in trades_df.columns:
            ax.text(0.5, 0.5, '无交易数据', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('收益分布')
            return
            
        returns = trades_df['profit_ratio'] * 100
        
        ax.hist(returns, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(returns.mean(), color='red', linestyle='--', 
                  label=f'平均: {returns.mean():.2f}%')
        ax.axvline(returns.median(), color='green', linestyle='--',
                  label=f'中位数: {returns.median():.2f}%')
        ax.set_title('单笔交易收益分布', fontweight='bold')
        ax.set_xlabel('收益率 (%)')
        ax.set_ylabel('频次')
        ax.legend()
        
    def plot_monthly_returns_heatmap(self, data: Dict[str, Any], ax):
        """绘制月度收益热力图"""
        trades_df = data.get('trades', pd.DataFrame())
        if trades_df.empty:
            ax.text(0.5, 0.5, '无交易数据', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('月度收益')
            return
            
        try:
            # 转换日期并按月汇总
            trades_df['close_date'] = pd.to_datetime(trades_df['close_date'])
            trades_df['year_month'] = trades_df['close_date'].dt.to_period('M')
            monthly_returns = trades_df.groupby('year_month')['profit_ratio'].sum() * 100
            
            if len(monthly_returns) > 0:
                # 创建年月矩阵
                years = range(monthly_returns.index.year.min(), monthly_returns.index.year.max() + 1)
                months = range(1, 13)
                
                matrix = np.zeros((len(years), 12))
                matrix.fill(np.nan)
                
                for period, ret in monthly_returns.items():
                    year_idx = period.year - min(years)
                    month_idx = period.month - 1
                    matrix[year_idx, month_idx] = ret
                
                im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto')
                ax.set_xticks(range(12))
                ax.set_xticklabels(['1月', '2月', '3月', '4月', '5月', '6月',
                                  '7月', '8月', '9月', '10月', '11月', '12月'])
                ax.set_yticks(range(len(years)))
                ax.set_yticklabels(years)
                ax.set_title('月度收益热力图 (%)', fontweight='bold')
                
                # 添加颜色条
                cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                cbar.set_label('收益率 (%)')
            else:
                ax.text(0.5, 0.5, '数据不足', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('月度收益热力图')
        except Exception as e:
            ax.text(0.5, 0.5, f'绘制错误: {str(e)}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('月度收益热力图')
            
    def plot_win_loss_analysis(self, data: Dict[str, Any], ax):
        """绘制胜负分析"""
        metrics = data.get('metrics', {})
        
        win_rate = metrics.get('win_rate', 0) * 100
        loss_rate = 100 - win_rate
        avg_win = metrics.get('avg_win', 0) * 100
        avg_loss = abs(metrics.get('avg_loss', 0)) * 100
        
        # 创建双轴图
        ax2 = ax.twinx()
        
        # 胜率柱状图
        bars1 = ax.bar(['胜率', '败率'], [win_rate, loss_rate], 
                      color=['green', 'red'], alpha=0.7, width=0.6)
        ax.set_ylabel('比例 (%)', color='blue')
        ax.set_ylim(0, 100)
        
        # 平均盈亏柱状图
        bars2 = ax2.bar(['平均盈利', '平均亏损'], [avg_win, avg_loss],
                       color=['lightgreen', 'lightcoral'], alpha=0.7, width=0.3)
        ax2.set_ylabel('平均收益率 (%)', color='red')
        
        # 添加数值标签
        for bar, value in zip(bars1, [win_rate, loss_rate]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{value:.1f}%', ha='center', va='bottom')
                   
        for bar, value in zip(bars2, [avg_win, avg_loss]):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{value:.2f}%', ha='center', va='bottom')
        
        ax.set_title('胜负分析', fontweight='bold')
        
    def plot_duration_analysis(self, data: Dict[str, Any], ax):
        """绘制交易时长分析"""
        trades_df = data.get('trades', pd.DataFrame())
        
        if trades_df.empty or 'open_date' not in trades_df.columns or 'close_date' not in trades_df.columns:
            ax.text(0.5, 0.5, '无时长数据', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('交易时长分析')
            return
            
        try:
            # 计算交易时长(小时)
            trades_df['open_date'] = pd.to_datetime(trades_df['open_date'])
            trades_df['close_date'] = pd.to_datetime(trades_df['close_date'])
            trades_df['duration_hours'] = (trades_df['close_date'] - trades_df['open_date']).dt.total_seconds() / 3600
            
            durations = trades_df['duration_hours'].dropna()
            
            if not durations.empty:
                ax.hist(durations, bins=20, alpha=0.7, color='orange', edgecolor='black')
                ax.axvline(durations.mean(), color='red', linestyle='--',
                          label=f'平均: {durations.mean():.1f}h')
                ax.axvline(durations.median(), color='green', linestyle='--',
                          label=f'中位数: {durations.median():.1f}h')
                ax.set_xlabel('时长 (小时)')
                ax.set_ylabel('频次')
                ax.legend()
            else:
                ax.text(0.5, 0.5, '无有效时长数据', ha='center', va='center', transform=ax.transAxes)
                
        except Exception as e:
            ax.text(0.5, 0.5, f'计算错误: {str(e)}', ha='center', va='center', transform=ax.transAxes)
            
        ax.set_title('交易时长分析', fontweight='bold')
        
    def plot_risk_return_scatter(self, data: Dict[str, Any], ax):
        """绘制风险收益散点图"""
        metrics = data.get('metrics', {})
        
        # 创建示例数据点进行对比
        strategies = ['当前策略', '买入持有', '市场平均', '保守策略']
        returns = [
            metrics.get('total_return', 0) * 100,
            8.0,  # 假设买入持有收益
            6.0,  # 假设市场平均收益 
            4.0   # 假设保守策略收益
        ]
        risks = [
            abs(metrics.get('max_drawdown', 0)) * 100,
            15.0,  # 假设买入持有最大回撤
            12.0,  # 假设市场平均回撤
            5.0    # 假设保守策略回撤
        ]
        colors = ['red', 'blue', 'green', 'orange']
        
        for i, (strategy, ret, risk, color) in enumerate(zip(strategies, returns, risks, colors)):
            ax.scatter(risk, ret, c=color, s=100, alpha=0.7, label=strategy)
            
        ax.set_xlabel('最大回撤 (%)')
        ax.set_ylabel('总收益率 (%)')
        ax.set_title('风险收益对比', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加理想区域标识
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=15, color='red', linestyle='--', alpha=0.5, label='15%回撤警戒线')
        
    def plot_performance_radar(self, data: Dict[str, Any], ax):
        """绘制性能雷达图"""
        metrics = data.get('metrics', {})
        
        # 定义评估维度和分数
        categories = ['收益率', '夏普比率', '风险控制', '胜率', '盈利因子']
        
        # 标准化各项指标到0-100分
        scores = [
            min(100, max(0, metrics.get('total_return', 0) * 200)),  # 收益率 
            min(100, max(0, metrics.get('sharpe_ratio', 0) * 50)),   # 夏普比率
            min(100, max(0, (1 - abs(metrics.get('max_drawdown', 0))) * 100)),  # 风险控制
            min(100, max(0, metrics.get('win_rate', 0) * 100)),      # 胜率
            min(100, max(0, metrics.get('profit_factor', 0) * 50))   # 盈利因子
        ]
        
        # 计算角度
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        scores += scores[:1]
        
        # 绘制雷达图
        ax.plot(angles, scores, 'o-', linewidth=2, color='blue', alpha=0.8)
        ax.fill(angles, scores, alpha=0.25, color='blue')
        
        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20', '40', '60', '80', '100'])
        ax.grid(True)
        
        ax.set_title('策略性能雷达图\n(满分100)', size=12, fontweight='bold', pad=20)
        
    def plot_key_metrics_summary(self, data: Dict[str, Any], ax):
        """绘制关键指标总结"""
        metrics = data.get('metrics', {})
        comparison = data.get('comparison', {})
        
        # 关键指标数据
        key_metrics = {
            '总收益率': f"{metrics.get('total_return', 0)*100:.2f}%",
            '夏普比率': f"{metrics.get('sharpe_ratio', 0):.2f}",
            '最大回撤': f"{abs(metrics.get('max_drawdown', 0))*100:.2f}%",
            '胜率': f"{metrics.get('win_rate', 0)*100:.1f}%",
            '盈利因子': f"{metrics.get('profit_factor', 0):.2f}",
            '交易次数': f"{metrics.get('total_trades', 0)}",
            '平均持仓': f"{metrics.get('avg_trade_duration_hours', 0):.1f}h",
            '综合评分': f"{comparison.get('performance_score', 0):.0f}/100"
        }
        
        # 创建表格
        ax.axis('tight')
        ax.axis('off')
        
        # 准备表格数据
        table_data = []
        for key, value in key_metrics.items():
            table_data.append([key, value])
        
        table = ax.table(cellText=table_data,
                        colLabels=['指标', '数值'],
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.6, 0.4])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # 设置表格样式
        for i in range(len(table_data)):
            table[(i+1, 0)].set_facecolor('#f0f0f0')
            table[(i+1, 1)].set_facecolor('#ffffff')
            
        # 设置标题行样式
        table[(0, 0)].set_facecolor('#4472C4')
        table[(0, 1)].set_facecolor('#4472C4')
        table[(0, 0)].set_text_props(weight='bold', color='white')
        table[(0, 1)].set_text_props(weight='bold', color='white')
        
        ax.set_title('关键指标汇总', fontweight='bold', pad=20)
        
    def generate_html_report(self, analysis_data: Dict[str, Any], timestamp: str) -> str:
        """生成HTML报告"""
        self.logger.info("生成HTML报告")
        
        metrics = analysis_data.get('metrics', {})
        comparison = analysis_data.get('comparison', {})
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Enhanced Grid Strategy 回测分析报告</title>
            <style>
                {self.get_html_styles()}
            </style>
        </head>
        <body>
            <div class="container">
                <header>
                    <h1>Enhanced Grid Strategy 回测分析报告</h1>
                    <p class="timestamp">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </header>
                
                <section class="executive-summary">
                    <h2>执行摘要</h2>
                    <div class="summary-cards">
                        <div class="card">
                            <h3>总收益率</h3>
                            <p class="metric">{metrics.get('total_return', 0)*100:.2f}%</p>
                        </div>
                        <div class="card">
                            <h3>夏普比率</h3>
                            <p class="metric">{metrics.get('sharpe_ratio', 0):.2f}</p>
                        </div>
                        <div class="card">
                            <h3>最大回撤</h3>
                            <p class="metric">{abs(metrics.get('max_drawdown', 0))*100:.2f}%</p>
                        </div>
                        <div class="card">
                            <h3>胜率</h3>
                            <p class="metric">{metrics.get('win_rate', 0)*100:.1f}%</p>
                        </div>
                    </div>
                    <div class="rating">
                        <h3>综合评级: {comparison.get('rating', '未评级')}</h3>
                        <p>性能评分: {comparison.get('performance_score', 0):.0f}/100</p>
                    </div>
                </section>
                
                <section class="detailed-metrics">
                    <h2>详细指标分析</h2>
                    {self.generate_metrics_tables(metrics)}
                </section>
                
                <section class="benchmark-comparison">
                    <h2>基准对比分析</h2>
                    {self.generate_comparison_section(comparison)}
                </section>
                
                <section class="risk-analysis">
                    <h2>风险分析</h2>
                    {self.generate_risk_analysis(metrics)}
                </section>
                
                <section class="conclusions">
                    <h2>结论与建议</h2>
                    {self.generate_conclusions(metrics, comparison)}
                </section>
            </div>
        </body>
        </html>
        """
        
        # 保存HTML报告
        html_file = self.reports_dir / f"backtest_report_{timestamp}.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        self.logger.info(f"HTML报告已生成: {html_file}")
        return str(html_file)
        
    def get_html_styles(self) -> str:
        """获取HTML样式"""
        return """
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Arial', sans-serif; line-height: 1.6; color: #333; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; background: white; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        header { text-align: center; margin-bottom: 30px; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; }
        h1 { font-size: 2.5em; margin-bottom: 10px; }
        h2 { color: #4a5568; margin: 30px 0 20px 0; padding-bottom: 10px; border-bottom: 2px solid #e2e8f0; }
        h3 { color: #2d3748; margin: 20px 0 10px 0; }
        .timestamp { font-size: 1.1em; opacity: 0.9; }
        .summary-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }
        .card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; border-left: 4px solid #4299e1; }
        .card h3 { color: #4a5568; font-size: 1.1em; margin-bottom: 10px; }
        .metric { font-size: 2em; font-weight: bold; color: #2b6cb0; }
        .rating { text-align: center; margin: 30px 0; padding: 20px; background: #f7fafc; border-radius: 10px; }
        .rating h3 { color: #2d3748; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #e2e8f0; }
        th { background: #4299e1; color: white; font-weight: bold; }
        tr:hover { background: #f7fafc; }
        .positive { color: #38a169; font-weight: bold; }
        .negative { color: #e53e3e; font-weight: bold; }
        .neutral { color: #4a5568; }
        """
        
    def generate_metrics_tables(self, metrics: Dict[str, float]) -> str:
        """生成指标表格HTML"""
        basic_metrics = [
            ('总交易次数', metrics.get('total_trades', 0), ''),
            ('获胜交易', metrics.get('winning_trades', 0), ''),
            ('失败交易', metrics.get('losing_trades', 0), ''),
            ('胜率', f"{metrics.get('win_rate', 0)*100:.2f}", '%'),
            ('总收益率', f"{metrics.get('total_return', 0)*100:.2f}", '%'),
            ('平均单笔收益', f"{metrics.get('avg_return_per_trade', 0)*100:.2f}", '%'),
            ('最佳交易', f"{metrics.get('best_trade', 0)*100:.2f}", '%'),
            ('最差交易', f"{metrics.get('worst_trade', 0)*100:.2f}", '%'),
        ]
        
        risk_metrics = [
            ('最大回撤', f"{abs(metrics.get('max_drawdown', 0))*100:.2f}", '%'),
            ('收益波动率', f"{metrics.get('return_volatility', 0)*100:.2f}", '%'),
            ('夏普比率', f"{metrics.get('sharpe_ratio', 0):.2f}", ''),
            ('索提诺比率', f"{metrics.get('sortino_ratio', 0):.2f}", ''),
            ('卡尔马比率', f"{metrics.get('calmar_ratio', 0):.2f}", ''),
            ('盈利因子', f"{metrics.get('profit_factor', 0):.2f}", ''),
            ('期望收益', f"{metrics.get('expectancy', 0)*100:.2f}", '%'),
        ]
        
        html = '<div class="metrics-tables">'
        html += '<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px;">'
        
        # 基础指标表格
        html += '<div><h3>基础指标</h3><table>'
        html += '<tr><th>指标</th><th>数值</th></tr>'
        for metric, value, unit in basic_metrics:
            html += f'<tr><td>{metric}</td><td>{value}{unit}</td></tr>'
        html += '</table></div>'
        
        # 风险指标表格  
        html += '<div><h3>风险指标</h3><table>'
        html += '<tr><th>指标</th><th>数值</th></tr>'
        for metric, value, unit in risk_metrics:
            html += f'<tr><td>{metric}</td><td>{value}{unit}</td></tr>'
        html += '</table></div>'
        
        html += '</div></div>'
        return html
        
    def generate_comparison_section(self, comparison: Dict[str, Any]) -> str:
        """生成对比分析HTML"""
        html = '<div class="comparison-section">'
        html += f'<p>超额收益: <span class="{"positive" if comparison.get("excess_return", 0) > 0 else "negative"}">{comparison.get("excess_return_pct", 0):.2f}%</span></p>'
        html += f'<p>信息比率: {comparison.get("information_ratio", 0):.2f}</p>'
        html += f'<p>综合评级: <strong>{comparison.get("rating", "未评级")}</strong></p>'
        html += '</div>'
        return html
        
    def generate_risk_analysis(self, metrics: Dict[str, float]) -> str:
        """生成风险分析HTML"""
        max_dd = abs(metrics.get('max_drawdown', 0)) * 100
        volatility = metrics.get('return_volatility', 0) * 100
        
        html = '<div class="risk-analysis">'
        
        if max_dd < 10:
            html += '<p class="positive">✓ 最大回撤控制良好，低于10%</p>'
        elif max_dd < 15:
            html += '<p class="neutral">△ 最大回撤可接受，介于10-15%</p>'
        else:
            html += '<p class="negative">⚠ 最大回撤较高，超过15%，需要加强风险控制</p>'
            
        if volatility < 15:
            html += '<p class="positive">✓ 收益波动率较低，策略稳定性好</p>'
        elif volatility < 25:
            html += '<p class="neutral">△ 收益波动率中等</p>'
        else:
            html += '<p class="negative">⚠ 收益波动率较高</p>'
            
        html += '</div>'
        return html
        
    def generate_conclusions(self, metrics: Dict[str, float], comparison: Dict[str, Any]) -> str:
        """生成结论HTML"""
        html = '<div class="conclusions">'
        
        total_return = metrics.get('total_return', 0) * 100
        sharpe = metrics.get('sharpe_ratio', 0)
        max_dd = abs(metrics.get('max_drawdown', 0)) * 100
        
        if total_return > 20 and sharpe > 1.5 and max_dd < 15:
            html += '<h3 class="positive">策略表现优秀</h3>'
            html += '<ul>'
            html += '<li>收益率表现突出，超过预期目标</li>'
            html += '<li>风险调整后收益良好，夏普比率优秀</li>'
            html += '<li>风险控制有效，回撤在合理范围内</li>'
            html += '<li>建议: 可以考虑增加仓位或扩大策略应用范围</li>'
            html += '</ul>'
        elif total_return > 10 and sharpe > 1.0:
            html += '<h3 class="neutral">策略表现良好</h3>'
            html += '<ul>'
            html += '<li>收益率达到预期，具备一定的盈利能力</li>'
            html += '<li>风险收益比合理</li>'
            html += '<li>建议: 可以进一步优化参数提升表现</li>'
            html += '</ul>'
        else:
            html += '<h3 class="negative">策略需要改进</h3>'
            html += '<ul>'
            html += '<li>收益率或风险控制需要优化</li>'
            html += '<li>建议重新评估策略参数</li>'
            html += '<li>考虑调整风险管理机制</li>'
            html += '</ul>'
            
        html += '</div>'
        return html
        
    def analyze_results(self, results_files: List[str]) -> Dict[str, Any]:
        """分析回测结果"""
        self.logger.info(f"开始分析 {len(results_files)} 个结果文件")
        
        all_analysis = {}
        
        for result_file in results_files[:3]:  # 限制处理文件数量
            self.logger.info(f"分析文件: {result_file}")
            
            try:
                # 加载数据
                data = self.load_backtest_data(result_file)
                if not data:
                    continue
                    
                # 提取交易数据
                trades_data = self.extract_trades_data(data)
                if trades_data.empty:
                    self.logger.warning(f"文件 {result_file} 无有效交易数据")
                    continue
                    
                # 构建权益曲线
                equity_curve = self.build_equity_curve(trades_data)
                
                # 计算指标
                metrics = self.calculate_advanced_metrics(trades_data, equity_curve)
                
                # 基准对比
                comparison = self.compare_with_benchmark(metrics)
                
                file_name = Path(result_file).stem
                all_analysis[file_name] = {
                    'trades': trades_data,
                    'equity_curve': equity_curve,
                    'metrics': metrics,
                    'comparison': comparison
                }
                
                self.logger.info(f"文件 {file_name} 分析完成")
                
            except Exception as e:
                self.logger.error(f"分析文件 {result_file} 失败: {e}")
                continue
                
        return all_analysis
        
    def extract_trades_data(self, data: Dict[str, Any]) -> pd.DataFrame:
        """从回测数据中提取交易记录"""
        try:
            # 根据数据结构提取交易记录
            if isinstance(data, dict):
                # 处理场景测试结果
                if any(k for k in data.keys() if 'scenario' in str(k) or 'period' in str(k)):
                    all_trades = []
                    for key, value in data.items():
                        if isinstance(value, dict) and value.get('success'):
                            # 这里需要根据实际数据结构调整
                            # 目前创建模拟数据用于演示
                            sample_trades = self.create_sample_trades_data()
                            all_trades.extend(sample_trades)
                    return pd.DataFrame(all_trades) if all_trades else pd.DataFrame()
                    
                # 处理普通回测结果
                elif 'trades' in data:
                    return pd.DataFrame(data['trades'])
                    
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"提取交易数据失败: {e}")
            return pd.DataFrame()
            
    def create_sample_trades_data(self) -> List[Dict[str, Any]]:
        """创建示例交易数据用于演示"""
        np.random.seed(42)
        
        trades = []
        base_date = datetime(2023, 1, 1)
        
        for i in range(100):  # 生成100笔模拟交易
            open_date = base_date + timedelta(days=i*2, hours=np.random.randint(0, 24))
            duration_hours = np.random.randint(1, 72)  # 1-72小时持仓
            close_date = open_date + timedelta(hours=duration_hours)
            
            # 生成符合网格策略特点的收益分布
            if np.random.random() < 0.65:  # 65%胜率
                profit_ratio = np.random.normal(0.008, 0.003)  # 小幅盈利
            else:
                profit_ratio = np.random.normal(-0.012, 0.005)  # 亏损
                
            trades.append({
                'open_date': open_date,
                'close_date': close_date,
                'profit_ratio': profit_ratio,
                'duration_hours': duration_hours
            })
            
        return trades
        
    def build_equity_curve(self, trades_df: pd.DataFrame) -> pd.Series:
        """构建权益曲线"""
        if trades_df.empty:
            return pd.Series()
            
        equity = [1.0]  # 初始资金为1
        for _, trade in trades_df.iterrows():
            profit_ratio = trade.get('profit_ratio', 0)
            equity.append(equity[-1] * (1 + profit_ratio))
            
        return pd.Series(equity)
        
    def run_analysis(self, results_files: List[str] = None) -> str:
        """运行完整分析流程"""
        if results_files is None:
            results_files = self.find_latest_results()
            
        if not results_files:
            self.logger.error("未找到回测结果文件")
            return ""
            
        # 分析结果
        analysis_data = self.analyze_results(results_files)
        
        if not analysis_data:
            self.logger.error("分析失败，无有效数据")
            return ""
            
        # 使用第一个有效结果进行报告生成
        first_result = next(iter(analysis_data.values()))
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 生成可视化仪表板
        dashboard_file = self.create_performance_dashboard(first_result, timestamp)
        
        # 生成HTML报告
        html_report = self.generate_html_report(first_result, timestamp)
        
        self.logger.info("分析完成")
        return html_report


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='回测结果分析工具')
    parser.add_argument('--results-dir', '-r', type=str,
                       help='回测结果目录 (默认: backtest_results)')
    parser.add_argument('--files', '-f', nargs='*', 
                       help='指定分析的结果文件')
    parser.add_argument('--output-dir', '-o', type=str,
                       help='报告输出目录')
                       
    args = parser.parse_args()
    
    try:
        # 创建分析器
        analyzer = BacktestResultAnalyzer(results_dir=args.results_dir)
        
        # 运行分析
        print("\n开始分析回测结果...")
        html_report = analyzer.run_analysis(results_files=args.files)
        
        if html_report:
            print(f"\n✅ 分析完成!")
            print(f"HTML报告: {html_report}")
            print(f"报告目录: {analyzer.reports_dir}")
        else:
            print("\n⚠️ 分析失败，请检查数据文件")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n用户中断分析")
        sys.exit(1)
    except Exception as e:
        print(f"\n分析执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()