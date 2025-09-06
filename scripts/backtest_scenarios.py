#!/usr/bin/env python3
"""
多场景回测脚本
针对不同市场环境和极端情况的策略测试

测试场景:
1. 牛市场景 - 持续上涨趋势
2. 熊市场景 - 持续下跌趋势  
3. 震荡市场 - 横盘震荡
4. 高波动率 - 剧烈波动
5. 黑天鹅事件 - 极端市场条件
"""

import os
import sys
import json
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))


class ScenarioBacktester:
    """多场景回测器"""
    
    def __init__(self, config_path: str = None):
        """初始化场景回测器"""
        self.project_root = PROJECT_ROOT
        self.config_path = config_path or str(self.project_root / "configs" / "strategy-21-base.json")
        self.results_dir = self.project_root / "backtest_results" / "scenarios"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.setup_logging()
        
        # 定义测试场景
        self.scenarios = self.define_test_scenarios()
        
        # 结果存储
        self.scenario_results = {}
        
        # 设置绘图
        self.setup_plotting()
        
    def setup_logging(self):
        """设置日志记录"""
        log_file = self.results_dir / f"scenario_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"场景测试器初始化，日志文件: {log_file}")
        
    def setup_plotting(self):
        """设置绘图环境"""
        plt.style.use('seaborn')
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.figsize'] = (12, 8)
        
    def define_test_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """定义测试场景"""
        scenarios = {
            # 牛市场景
            "bull_market_2023q1": {
                "name": "2023年Q1牛市",
                "start_date": "20230101",
                "end_date": "20230331", 
                "description": "典型牛市上涨行情",
                "market_type": "bull",
                "expected_performance": "high_return"
            },
            
            "bull_market_2023q2": {
                "name": "2023年Q2持续牛市",
                "start_date": "20230401", 
                "end_date": "20230630",
                "description": "牛市延续阶段",
                "market_type": "bull",
                "expected_performance": "high_return"
            },
            
            # 熊市场景  
            "bear_market_2023": {
                "name": "2023年下半年熊市",
                "start_date": "20230701",
                "end_date": "20231031",
                "description": "典型熊市下跌行情",
                "market_type": "bear", 
                "expected_performance": "defensive"
            },
            
            # 震荡市场
            "sideways_2023q4": {
                "name": "2023年Q4震荡",
                "start_date": "20231101",
                "end_date": "20231231", 
                "description": "横盘震荡市场",
                "market_type": "sideways",
                "expected_performance": "moderate_profit"
            },
            
            "sideways_2024": {
                "name": "2024年震荡市",
                "start_date": "20240101",
                "end_date": "20240630",
                "description": "长期横盘震荡",
                "market_type": "sideways", 
                "expected_performance": "moderate_profit"
            },
            
            # 高波动率场景
            "high_volatility_march2023": {
                "name": "2023年3月银行危机",
                "start_date": "20230308", 
                "end_date": "20230325",
                "description": "银行业危机引发高波动",
                "market_type": "high_volatility",
                "expected_performance": "stress_test"
            },
            
            "high_volatility_summer2023": {
                "name": "2023年夏季波动",
                "start_date": "20230801",
                "end_date": "20230831", 
                "description": "夏季市场调整期",
                "market_type": "high_volatility",
                "expected_performance": "stress_test"
            },
            
            # 趋势转换
            "trend_reversal_1": {
                "name": "牛转熊趋势转换",
                "start_date": "20230615",
                "end_date": "20230715", 
                "description": "牛市向熊市转换期",
                "market_type": "trend_reversal",
                "expected_performance": "adaptive"
            },
            
            # 近期表现
            "recent_performance": {
                "name": "2024年近期表现",
                "start_date": "20240701",
                "end_date": "20241130",
                "description": "最新市场环境测试", 
                "market_type": "recent",
                "expected_performance": "current_adaptive"
            }
        }
        
        return scenarios
        
    def execute_scenario_backtest(self, scenario_name: str, scenario_config: Dict[str, Any]) -> Dict[str, Any]:
        """执行单个场景回测"""
        self.logger.info(f"开始场景回测: {scenario_name}")
        
        # 构建回测命令
        cmd = [
            "freqtrade", "backtesting",
            "--config", self.config_path,
            "--strategy", "EnhancedGridStrategy",
            "--timeframe", "5m",
            "--timerange", f"{scenario_config['start_date']}-{scenario_config['end_date']}",
            "--export", "trades,signals",
            "--cache", "none",
            "--export-filename", f"scenario_{scenario_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ]
        
        try:
            # 执行回测
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True, 
                text=True,
                timeout=1800  # 30分钟超时
            )
            
            if result.returncode != 0:
                self.logger.error(f"场景 {scenario_name} 回测失败: {result.stderr}")
                return {
                    "success": False,
                    "scenario": scenario_name,
                    "error": result.stderr,
                    "config": scenario_config
                }
                
            # 解析结果
            performance_data = self.parse_backtest_output(result.stdout)
            
            # 分析场景表现
            scenario_analysis = self.analyze_scenario_performance(
                performance_data, 
                scenario_config
            )
            
            self.logger.info(f"场景 {scenario_name} 回测完成")
            
            return {
                "success": True,
                "scenario": scenario_name,
                "config": scenario_config,
                "performance": performance_data,
                "analysis": scenario_analysis,
                "raw_output": result.stdout
            }
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"场景 {scenario_name} 回测超时")
            return {
                "success": False,
                "scenario": scenario_name,
                "error": "回测超时",
                "config": scenario_config
            }
        except Exception as e:
            self.logger.error(f"场景 {scenario_name} 回测异常: {str(e)}")
            return {
                "success": False,
                "scenario": scenario_name, 
                "error": str(e),
                "config": scenario_config
            }
            
    def parse_backtest_output(self, output: str) -> Dict[str, Any]:
        """解析回测输出"""
        try:
            lines = output.split('\n')
            data = {}
            
            for line in lines:
                line = line.strip()
                
                if 'Total trades' in line:
                    data['total_trades'] = self.extract_number(line, int)
                elif 'Total Profit' in line and '%' in line:
                    data['total_profit_pct'] = self.extract_percentage(line)
                elif 'Win Rate' in line:
                    data['win_rate_pct'] = self.extract_percentage(line)
                elif 'Max Drawdown' in line and '%' in line:
                    data['max_drawdown_pct'] = abs(self.extract_percentage(line))
                elif 'Avg. Duration' in line:
                    data['avg_duration'] = line.split(':')[-1].strip()
                elif 'Sharpe Ratio' in line:
                    data['sharpe_ratio'] = self.extract_number(line, float, default=0.0)
                elif 'Profit Factor' in line:
                    data['profit_factor'] = self.extract_number(line, float, default=1.0)
                elif 'Expectancy' in line:
                    data['expectancy'] = self.extract_number(line, float, default=0.0)
                    
            return data
            
        except Exception as e:
            self.logger.warning(f"解析输出失败: {e}")
            return {}
            
    def extract_number(self, line: str, num_type=float, default=None):
        """从行中提取数字"""
        try:
            value_str = line.split(':')[-1].strip()
            # 移除百分号和其他符号
            value_str = value_str.replace('%', '').replace(',', '')
            return num_type(value_str)
        except:
            return default if default is not None else (0 if num_type == int else 0.0)
            
    def extract_percentage(self, line: str) -> float:
        """从行中提取百分比数值"""
        return self.extract_number(line, float, default=0.0)
        
    def analyze_scenario_performance(self, performance: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """分析场景表现"""
        market_type = config.get("market_type", "unknown")
        expected = config.get("expected_performance", "moderate")
        
        analysis = {
            "market_adaptation": self.evaluate_market_adaptation(performance, market_type),
            "risk_management": self.evaluate_risk_management(performance, market_type),
            "profit_efficiency": self.evaluate_profit_efficiency(performance, expected),
            "trade_quality": self.evaluate_trade_quality(performance),
            "overall_score": 0  # 将在下面计算
        }
        
        # 计算综合评分 (0-100)
        scores = [v for v in analysis.values() if isinstance(v, (int, float))]
        analysis["overall_score"] = np.mean(scores) if scores else 0
        
        # 生成评估结论
        analysis["conclusion"] = self.generate_scenario_conclusion(analysis, config)
        
        return analysis
        
    def evaluate_market_adaptation(self, performance: Dict[str, Any], market_type: str) -> float:
        """评估市场适应性 (0-100分)"""
        total_return = performance.get("total_profit_pct", 0)
        win_rate = performance.get("win_rate_pct", 0)
        
        if market_type == "bull":
            # 牛市应该有较高收益
            if total_return > 30:
                return 90
            elif total_return > 20:
                return 80
            elif total_return > 10:
                return 70
            elif total_return > 0:
                return 60
            else:
                return 30
                
        elif market_type == "bear":
            # 熊市重点是控制损失
            if total_return > 10:
                return 95  # 熊市还能盈利很优秀
            elif total_return > 0:
                return 85
            elif total_return > -5:
                return 75  # 小幅亏损可接受
            elif total_return > -10:
                return 60
            else:
                return 30
                
        elif market_type == "sideways":
            # 震荡市场适合网格策略
            if total_return > 15:
                return 90
            elif total_return > 10:
                return 85
            elif total_return > 5:
                return 75
            elif total_return > 0:
                return 65
            else:
                return 40
                
        elif market_type == "high_volatility":
            # 高波动环境下的风险控制
            max_dd = performance.get("max_drawdown_pct", 100)
            if total_return > 0 and max_dd < 10:
                return 90
            elif total_return > -5 and max_dd < 15:
                return 75
            elif max_dd < 20:
                return 60
            else:
                return 30
                
        else:
            # 默认评估
            if total_return > 10:
                return 80
            elif total_return > 0:
                return 70
            else:
                return 40
                
    def evaluate_risk_management(self, performance: Dict[str, Any], market_type: str) -> float:
        """评估风险管理能力"""
        max_dd = performance.get("max_drawdown_pct", 100)
        sharpe = performance.get("sharpe_ratio", 0)
        
        # 回撤控制评分
        if max_dd < 5:
            dd_score = 100
        elif max_dd < 10:
            dd_score = 90
        elif max_dd < 15:
            dd_score = 80
        elif max_dd < 20:
            dd_score = 70
        elif max_dd < 30:
            dd_score = 60
        else:
            dd_score = 40
            
        # 夏普比率评分
        if sharpe > 2:
            sharpe_score = 100
        elif sharpe > 1.5:
            sharpe_score = 90
        elif sharpe > 1:
            sharpe_score = 80
        elif sharpe > 0.5:
            sharpe_score = 70
        elif sharpe > 0:
            sharpe_score = 60
        else:
            sharpe_score = 40
            
        return (dd_score + sharpe_score) / 2
        
    def evaluate_profit_efficiency(self, performance: Dict[str, Any], expected: str) -> float:
        """评估盈利效率"""
        total_return = performance.get("total_profit_pct", 0)
        total_trades = performance.get("total_trades", 0)
        
        # 根据预期表现调整评估标准
        if expected == "high_return":
            target_return = 20
        elif expected == "moderate_profit":
            target_return = 10
        elif expected == "defensive":
            target_return = 0
        else:
            target_return = 5
            
        # 收益评分
        if total_return >= target_return * 1.5:
            return_score = 100
        elif total_return >= target_return:
            return_score = 85
        elif total_return >= target_return * 0.5:
            return_score = 70
        elif total_return > 0:
            return_score = 60
        else:
            return_score = 40
            
        # 考虑交易效率 (收益/交易次数)
        if total_trades > 0:
            return_per_trade = total_return / total_trades
            if return_per_trade > 2:
                efficiency_bonus = 10
            elif return_per_trade > 1:
                efficiency_bonus = 5
            else:
                efficiency_bonus = 0
        else:
            efficiency_bonus = 0
            
        return min(100, return_score + efficiency_bonus)
        
    def evaluate_trade_quality(self, performance: Dict[str, Any]) -> float:
        """评估交易质量"""
        win_rate = performance.get("win_rate_pct", 0) 
        profit_factor = performance.get("profit_factor", 1.0)
        
        # 胜率评分
        if win_rate > 70:
            wr_score = 100
        elif win_rate > 60:
            wr_score = 90
        elif win_rate > 55:
            wr_score = 80
        elif win_rate > 50:
            wr_score = 70
        elif win_rate > 45:
            wr_score = 60
        else:
            wr_score = 40
            
        # 盈亏比评分
        if profit_factor > 2:
            pf_score = 100
        elif profit_factor > 1.5:
            pf_score = 90
        elif profit_factor > 1.2:
            pf_score = 80
        elif profit_factor > 1:
            pf_score = 70
        else:
            pf_score = 40
            
        return (wr_score + pf_score) / 2
        
    def generate_scenario_conclusion(self, analysis: Dict[str, Any], config: Dict[str, Any]) -> str:
        """生成场景分析结论"""
        score = analysis["overall_score"]
        market_type = config.get("market_type", "unknown")
        scenario_name = config.get("name", "Unknown")
        
        if score >= 85:
            performance_level = "优秀"
        elif score >= 75:
            performance_level = "良好" 
        elif score >= 65:
            performance_level = "中等"
        elif score >= 55:
            performance_level = "偏弱"
        else:
            performance_level = "较差"
            
        conclusion = f"在{scenario_name}场景中，策略表现{performance_level}（评分: {score:.1f}分）。"
        
        # 添加具体建议
        if market_type == "bull" and score < 80:
            conclusion += " 在牛市中表现不够理想，建议优化趋势跟踪能力。"
        elif market_type == "bear" and analysis["risk_management"] < 70:
            conclusion += " 在熊市中风险控制需要加强，建议调整止损策略。"
        elif market_type == "sideways" and score > 80:
            conclusion += " 在震荡市场中表现出色，网格策略优势明显。"
        elif market_type == "high_volatility" and analysis["risk_management"] > 75:
            conclusion += " 在高波动环境下风险控制良好，策略稳健性较强。"
            
        return conclusion
        
    def run_all_scenarios(self) -> Dict[str, Any]:
        """运行所有场景测试"""
        self.logger.info(f"开始运行 {len(self.scenarios)} 个场景测试")
        
        results = {}
        
        # 并行执行场景测试
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}
            
            for scenario_name, scenario_config in self.scenarios.items():
                future = executor.submit(
                    self.execute_scenario_backtest,
                    scenario_name, 
                    scenario_config
                )
                futures[scenario_name] = future
                
            # 收集结果
            for scenario_name, future in futures.items():
                try:
                    result = future.result(timeout=2400)  # 40分钟超时
                    results[scenario_name] = result
                    
                    if result["success"]:
                        self.logger.info(f"✓ {scenario_name} 测试完成")
                    else:
                        self.logger.error(f"✗ {scenario_name} 测试失败: {result.get('error', '未知错误')}")
                        
                except Exception as e:
                    self.logger.error(f"✗ {scenario_name} 测试异常: {str(e)}")
                    results[scenario_name] = {
                        "success": False,
                        "scenario": scenario_name,
                        "error": str(e)
                    }
                    
        self.scenario_results = results
        return results
        
    def generate_scenario_analysis_report(self) -> str:
        """生成场景分析报告"""
        if not self.scenario_results:
            return "无场景测试结果可供分析"
            
        self.logger.info("生成场景分析报告")
        
        report = []
        report.append("=" * 90)
        report.append("Enhanced Grid Strategy 多场景回测分析报告")
        report.append("=" * 90)
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"测试场景数量: {len(self.scenario_results)}")
        report.append("")
        
        # 成功统计
        successful = [r for r in self.scenario_results.values() if r.get("success")]
        failed = [r for r in self.scenario_results.values() if not r.get("success")]
        
        report.append(f"测试结果: {len(successful)}/{len(self.scenario_results)} 个场景成功")
        if failed:
            report.append("失败场景:")
            for result in failed:
                report.append(f"  - {result.get('scenario', 'Unknown')}: {result.get('error', 'Unknown error')}")
        report.append("")
        
        # 成功场景汇总表
        if successful:
            report.append("场景测试结果汇总:")
            report.append("-" * 90)
            report.append(f"{'场景名称':<20} {'市场类型':<15} {'收益率(%)':<10} {'最大回撤(%)':<12} {'胜率(%)':<8} {'评分':<8}")
            report.append("-" * 90)
            
            for result in successful:
                scenario_name = result["scenario"]
                config = result["config"]
                performance = result["performance"]
                analysis = result["analysis"]
                
                report.append(
                    f"{config.get('name', scenario_name)[:19]:<20} "
                    f"{config.get('market_type', 'unknown'):<15} "
                    f"{performance.get('total_profit_pct', 0):<10.2f} "
                    f"{performance.get('max_drawdown_pct', 0):<12.2f} "
                    f"{performance.get('win_rate_pct', 0):<8.2f} "
                    f"{analysis.get('overall_score', 0):<8.1f}"
                )
                
            report.append("-" * 90)
            report.append("")
            
            # 按市场类型分析
            market_types = {}
            for result in successful:
                market_type = result["config"].get("market_type", "unknown")
                if market_type not in market_types:
                    market_types[market_type] = []
                market_types[market_type].append(result)
                
            report.append("按市场类型分析:")
            report.append("-" * 50)
            
            for market_type, results_list in market_types.items():
                if not results_list:
                    continue
                    
                avg_return = np.mean([r["performance"].get("total_profit_pct", 0) for r in results_list])
                avg_drawdown = np.mean([r["performance"].get("max_drawdown_pct", 0) for r in results_list])
                avg_score = np.mean([r["analysis"].get("overall_score", 0) for r in results_list])
                
                report.append(f"\n{market_type.upper()} 市场 ({len(results_list)}个场景):")
                report.append(f"  平均收益率: {avg_return:.2f}%")
                report.append(f"  平均最大回撤: {avg_drawdown:.2f}%")
                report.append(f"  平均评分: {avg_score:.1f}")
                
                # 表现评估
                if market_type == "bull" and avg_return > 20:
                    report.append(f"  评估: ✓ 牛市表现优秀")
                elif market_type == "bear" and avg_return > -5:
                    report.append(f"  评估: ✓ 熊市防守良好")
                elif market_type == "sideways" and avg_return > 8:
                    report.append(f"  评估: ✓ 震荡市场优势明显")
                elif market_type == "high_volatility" and avg_drawdown < 15:
                    report.append(f"  评估: ✓ 高波动环境风控有效")
                else:
                    report.append(f"  评估: ⚠ 该市场环境表现需要改进")
                    
            # 整体评估和建议
            report.append("\n整体评估和建议:")
            report.append("-" * 50)
            
            overall_avg_return = np.mean([r["performance"].get("total_profit_pct", 0) for r in successful])
            overall_avg_score = np.mean([r["analysis"].get("overall_score", 0) for r in successful])
            
            report.append(f"策略平均收益率: {overall_avg_return:.2f}%")
            report.append(f"策略平均评分: {overall_avg_score:.1f}")
            
            if overall_avg_score >= 80:
                report.append("✓ 策略在多种市场环境下表现优秀，具备较强的适应性")
            elif overall_avg_score >= 70:
                report.append("✓ 策略整体表现良好，在大多数场景下能获得正收益")
            elif overall_avg_score >= 60:
                report.append("⚠ 策略表现中等，建议针对低分场景进行优化")
            else:
                report.append("⚠ 策略在多个场景下表现不佳，需要重新评估和调优")
                
            # 具体建议
            weak_scenarios = [r for r in successful if r["analysis"].get("overall_score", 0) < 60]
            if weak_scenarios:
                report.append(f"\n需要关注的场景 ({len(weak_scenarios)}个):")
                for result in weak_scenarios:
                    scenario_name = result["config"].get("name", result["scenario"])
                    score = result["analysis"].get("overall_score", 0)
                    report.append(f"  - {scenario_name}: {score:.1f}分")
                    
        report.append("\n" + "=" * 90)
        
        return "\n".join(report)
        
    def save_scenario_results(self) -> Tuple[str, str]:
        """保存场景测试结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存详细结果
        results_file = self.results_dir / f"scenario_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.scenario_results, f, ensure_ascii=False, indent=2)
            
        # 保存分析报告
        report = self.generate_scenario_analysis_report()
        report_file = self.results_dir / f"scenario_analysis_{timestamp}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
            
        self.logger.info(f"场景测试结果已保存: {results_file}, {report_file}")
        return str(results_file), str(report_file)
        
    def create_performance_charts(self) -> str:
        """创建性能可视化图表"""
        if not self.scenario_results:
            return ""
            
        successful_results = [r for r in self.scenario_results.values() if r.get("success")]
        if not successful_results:
            return ""
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        charts_dir = self.results_dir / "charts"
        charts_dir.mkdir(exist_ok=True)
        
        # 1. 收益率对比图
        self.plot_returns_comparison(successful_results, charts_dir, timestamp)
        
        # 2. 风险收益散点图
        self.plot_risk_return_scatter(successful_results, charts_dir, timestamp)
        
        # 3. 市场类型表现雷达图
        self.plot_market_type_radar(successful_results, charts_dir, timestamp)
        
        self.logger.info(f"性能图表已生成: {charts_dir}")
        return str(charts_dir)
        
    def plot_returns_comparison(self, results: List[Dict], charts_dir: Path, timestamp: str):
        """绘制收益率对比图"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        scenario_names = [r["config"].get("name", r["scenario"])[:15] for r in results]
        returns = [r["performance"].get("total_profit_pct", 0) for r in results]
        colors = ['green' if r > 0 else 'red' for r in returns]
        
        bars = ax.bar(range(len(scenario_names)), returns, color=colors, alpha=0.7)
        
        ax.set_title('各场景收益率对比', fontsize=16, fontweight='bold')
        ax.set_xlabel('测试场景', fontsize=12)
        ax.set_ylabel('收益率 (%)', fontsize=12)
        ax.set_xticks(range(len(scenario_names)))
        ax.set_xticklabels(scenario_names, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars, returns):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height >= 0 else -1.5),
                   f'{value:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
        
        plt.tight_layout()
        plt.savefig(charts_dir / f"returns_comparison_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_risk_return_scatter(self, results: List[Dict], charts_dir: Path, timestamp: str):
        """绘制风险收益散点图"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        returns = [r["performance"].get("total_profit_pct", 0) for r in results]
        drawdowns = [r["performance"].get("max_drawdown_pct", 0) for r in results]
        market_types = [r["config"].get("market_type", "unknown") for r in results]
        
        # 不同市场类型使用不同颜色
        type_colors = {
            "bull": "green", 
            "bear": "red",
            "sideways": "blue",
            "high_volatility": "orange", 
            "trend_reversal": "purple",
            "recent": "brown"
        }
        
        for market_type in set(market_types):
            mask = [mt == market_type for mt in market_types]
            x_data = [drawdowns[i] for i, m in enumerate(mask) if m]
            y_data = [returns[i] for i, m in enumerate(mask) if m]
            
            ax.scatter(x_data, y_data, 
                      c=type_colors.get(market_type, "gray"),
                      label=market_type,
                      alpha=0.7, s=60)
        
        ax.set_title('风险收益关系图', fontsize=16, fontweight='bold')
        ax.set_xlabel('最大回撤 (%)', fontsize=12)
        ax.set_ylabel('总收益率 (%)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 添加理想区域标识
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=15, color='red', linestyle='--', alpha=0.5, label='15% 回撤警戒线')
        
        plt.tight_layout()
        plt.savefig(charts_dir / f"risk_return_scatter_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_market_type_radar(self, results: List[Dict], charts_dir: Path, timestamp: str):
        """绘制市场类型表现雷达图"""
        # 按市场类型汇总数据
        market_performance = {}
        for result in results:
            market_type = result["config"].get("market_type", "unknown")
            if market_type not in market_performance:
                market_performance[market_type] = []
            market_performance[market_type].append(result["analysis"].get("overall_score", 0))
        
        # 计算平均分数
        market_avg_scores = {k: np.mean(v) for k, v in market_performance.items()}
        
        if len(market_avg_scores) < 3:
            return  # 需要至少3个维度才能画雷达图
            
        # 创建雷达图
        categories = list(market_avg_scores.keys())
        values = list(market_avg_scores.values())
        
        # 计算角度
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # 完成圆形
        values += values[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        ax.plot(angles, values, 'o-', linewidth=2, color='blue', alpha=0.8)
        ax.fill(angles, values, alpha=0.25, color='blue')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20', '40', '60', '80', '100'])
        ax.grid(True)
        
        ax.set_title('各市场类型适应性评分\n(满分100分)', size=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(charts_dir / f"market_type_radar_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='多场景回测工具')
    parser.add_argument('--config', '-c', type=str,
                       help='配置文件路径')
    parser.add_argument('--scenarios', '-s', nargs='*',
                       help='指定测试场景 (默认测试所有场景)')
    parser.add_argument('--output-dir', '-o', type=str,
                       help='输出目录')
    
    args = parser.parse_args()
    
    try:
        # 创建场景测试器
        tester = ScenarioBacktester(config_path=args.config)
        
        # 执行场景测试
        print("\n开始多场景回测...")
        results = tester.run_all_scenarios()
        
        # 生成报告和图表
        print("\n生成分析报告...")
        results_file, report_file = tester.save_scenario_results()
        
        print("\n生成性能图表...")
        charts_dir = tester.create_performance_charts()
        
        # 显示简要结果
        print("\n" + tester.generate_scenario_analysis_report())
        
        successful_count = sum(1 for r in results.values() if r.get("success"))
        total_count = len(results)
        
        print(f"\n场景测试完成: {successful_count}/{total_count} 个场景成功")
        print(f"详细报告: {report_file}")
        if charts_dir:
            print(f"性能图表: {charts_dir}")
        
        if successful_count == 0:
            print("⚠️  所有场景测试均失败，请检查配置")
            sys.exit(1)
        elif successful_count < total_count:
            print("⚠️  部分场景测试失败，请查看日志")
            sys.exit(1) 
        else:
            print("✅ 所有场景测试成功完成")
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n用户中断测试")
        sys.exit(1)
    except Exception as e:
        print(f"\n场景测试失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()