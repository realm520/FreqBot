#!/usr/bin/env python3
"""
Enhanced Grid Strategy 回测脚本
基于 FreqTrade 框架的自动化回测系统

功能:
1. 自动化回测执行
2. 性能指标计算
3. 多时期对比分析
4. 结果报告生成
"""

import os
import sys
import json
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))


class EnhancedGridBacktester:
    """增强网格策略回测器"""
    
    def __init__(self, config_path: str = None):
        """初始化回测器"""
        self.project_root = PROJECT_ROOT
        self.config_path = config_path or str(self.project_root / "configs" / "strategy-21-base.json")
        self.results_dir = self.project_root / "backtest_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # 设置日志
        self.setup_logging()
        
        # 加载配置
        self.config = self.load_config()
        
        # 回测结果存储
        self.backtest_results = {}
        
        # 设置中文字体支持
        self.setup_matplotlib()
        
    def setup_logging(self):
        """设置日志记录"""
        log_file = self.results_dir / f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"回测器初始化，日志文件: {log_file}")
        
    def setup_matplotlib(self):
        """设置matplotlib中文字体支持"""
        # 尝试设置中文字体
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
        except:
            self.logger.warning("无法设置中文字体，将使用默认字体")
            
    def load_config(self) -> Dict[str, Any]:
        """加载策略配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.logger.info(f"已加载配置文件: {self.config_path}")
            return config
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {e}")
            raise
            
    def build_backtest_command(self, 
                             start_date: str, 
                             end_date: str,
                             timeframe: str = "5m",
                             strategy: str = "EnhancedGridStrategy",
                             export_filename: str = None) -> List[str]:
        """构建 FreqTrade 回测命令"""
        
        cmd = [
            "freqtrade", "backtesting",
            "--config", self.config_path,
            "--strategy", strategy,
            "--timeframe", timeframe,
            "--timerange", f"{start_date}-{end_date}",
            "--export", "trades,signals",
            "--cache", "none",  # 禁用缓存确保每次都是全新计算
        ]
        
        if export_filename:
            cmd.extend(["--export-filename", export_filename])
            
        self.logger.info(f"构建回测命令: {' '.join(cmd)}")
        return cmd
        
    def execute_backtest(self, 
                        start_date: str, 
                        end_date: str,
                        name: str = None,
                        timeframe: str = "5m") -> Dict[str, Any]:
        """执行单次回测"""
        
        name = name or f"backtest_{start_date}_to_{end_date}"
        export_filename = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"开始回测: {name} ({start_date} 到 {end_date})")
        
        # 构建命令
        cmd = self.build_backtest_command(
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
            export_filename=export_filename
        )
        
        try:
            # 执行回测命令
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=3600  # 1小时超时
            )
            
            if result.returncode != 0:
                self.logger.error(f"回测失败: {result.stderr}")
                return {
                    "success": False,
                    "error": result.stderr,
                    "name": name,
                    "start_date": start_date,
                    "end_date": end_date
                }
            
            self.logger.info(f"回测完成: {name}")
            
            # 解析回测结果
            backtest_data = self.parse_backtest_output(result.stdout)
            
            return {
                "success": True,
                "name": name,
                "start_date": start_date,
                "end_date": end_date,
                "timeframe": timeframe,
                "raw_output": result.stdout,
                "data": backtest_data,
                "export_filename": export_filename
            }
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"回测超时: {name}")
            return {
                "success": False,
                "error": "回测执行超时",
                "name": name,
                "start_date": start_date,
                "end_date": end_date
            }
        except Exception as e:
            self.logger.error(f"回测异常: {name} - {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "name": name,
                "start_date": start_date,
                "end_date": end_date
            }
            
    def parse_backtest_output(self, output: str) -> Dict[str, Any]:
        """解析回测输出结果"""
        try:
            # 从输出中提取关键信息
            lines = output.split('\n')
            result_data = {}
            
            # 解析基础统计信息
            for line in lines:
                line = line.strip()
                if 'Total trades' in line:
                    result_data['total_trades'] = int(line.split(':')[-1].strip())
                elif 'Total Profit' in line and '%' in line:
                    profit_str = line.split(':')[-1].strip().replace('%', '')
                    result_data['total_profit_pct'] = float(profit_str)
                elif 'Avg. Duration' in line:
                    result_data['avg_duration'] = line.split(':')[-1].strip()
                elif 'Win Rate' in line:
                    win_rate_str = line.split(':')[-1].strip().replace('%', '')
                    result_data['win_rate_pct'] = float(win_rate_str)
                elif 'Max Drawdown' in line and '%' in line:
                    dd_str = line.split(':')[-1].strip().replace('%', '')
                    result_data['max_drawdown_pct'] = float(dd_str)
                elif 'Sharpe Ratio' in line:
                    try:
                        sharpe_str = line.split(':')[-1].strip()
                        result_data['sharpe_ratio'] = float(sharpe_str)
                    except:
                        result_data['sharpe_ratio'] = 0.0
                        
            return result_data
            
        except Exception as e:
            self.logger.warning(f"解析回测输出失败: {e}")
            return {}
            
    def run_comprehensive_backtest(self) -> Dict[str, Any]:
        """运行综合回测分析"""
        self.logger.info("开始综合回测分析")
        
        # 定义测试时期
        test_periods = {
            "full_period": {
                "start": "20230101",
                "end": "20241201", 
                "description": "完整时期测试"
            },
            "bull_market": {
                "start": "20230101",
                "end": "20230630",
                "description": "牛市测试"
            },
            "consolidation": {
                "start": "20230701",
                "end": "20231231", 
                "description": "震荡期测试"
            },
            "recent_period": {
                "start": "20240101",
                "end": "20241201",
                "description": "近期表现测试"
            }
        }
        
        results = {}
        
        # 并行执行回测
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}
            
            for period_name, period_config in test_periods.items():
                future = executor.submit(
                    self.execute_backtest,
                    start_date=period_config["start"],
                    end_date=period_config["end"], 
                    name=period_name
                )
                futures[period_name] = future
                
            # 收集结果
            for period_name, future in futures.items():
                try:
                    result = future.result(timeout=1800)  # 30分钟超时
                    results[period_name] = result
                    
                    if result["success"]:
                        self.logger.info(f"{period_name} 回测成功")
                    else:
                        self.logger.error(f"{period_name} 回测失败: {result.get('error', '未知错误')}")
                        
                except Exception as e:
                    self.logger.error(f"{period_name} 回测异常: {str(e)}")
                    results[period_name] = {
                        "success": False,
                        "error": str(e),
                        "name": period_name
                    }
                    
        self.backtest_results = results
        return results
        
    def calculate_performance_metrics(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """计算详细的性能指标"""
        if not result.get("success") or not result.get("data"):
            return {"error": "无有效回测数据"}
            
        data = result["data"]
        
        metrics = {
            "基础指标": {
                "总交易次数": data.get("total_trades", 0),
                "总收益率(%)": data.get("total_profit_pct", 0),
                "胜率(%)": data.get("win_rate_pct", 0),
                "平均持仓时间": data.get("avg_duration", "N/A"),
                "最大回撤(%)": data.get("max_drawdown_pct", 0)
            },
            "风险指标": {
                "夏普比率": data.get("sharpe_ratio", 0),
                "风险调整收益": data.get("total_profit_pct", 0) / max(data.get("max_drawdown_pct", 1), 1),
                "交易频率": data.get("total_trades", 0) / max(self.calculate_days_between(result["start_date"], result["end_date"]), 1)
            }
        }
        
        # 评估等级
        metrics["评估等级"] = self.evaluate_performance(metrics)
        
        return metrics
        
    def calculate_days_between(self, start_date: str, end_date: str) -> int:
        """计算两个日期之间的天数"""
        try:
            start = datetime.strptime(start_date, "%Y%m%d")
            end = datetime.strptime(end_date, "%Y%m%d")
            return (end - start).days
        except:
            return 365  # 默认一年
            
    def evaluate_performance(self, metrics: Dict[str, Any]) -> str:
        """评估策略表现等级"""
        basic_metrics = metrics.get("基础指标", {})
        risk_metrics = metrics.get("风险指标", {})
        
        total_return = basic_metrics.get("总收益率(%)", 0)
        max_dd = basic_metrics.get("最大回撤(%)", 100)
        sharpe = risk_metrics.get("夏普比率", 0)
        win_rate = basic_metrics.get("胜率(%)", 0)
        
        score = 0
        
        # 收益率评分 (40%)
        if total_return > 50:
            score += 40
        elif total_return > 30:
            score += 32
        elif total_return > 20:
            score += 24
        elif total_return > 10:
            score += 16
        elif total_return > 0:
            score += 8
            
        # 风险评分 (30%)
        if abs(max_dd) < 5:
            score += 30
        elif abs(max_dd) < 10:
            score += 24
        elif abs(max_dd) < 15:
            score += 18
        elif abs(max_dd) < 20:
            score += 12
        elif abs(max_dd) < 30:
            score += 6
            
        # 夏普比率评分 (20%)
        if sharpe > 2.0:
            score += 20
        elif sharpe > 1.5:
            score += 16
        elif sharpe > 1.0:
            score += 12
        elif sharpe > 0.5:
            score += 8
        elif sharpe > 0:
            score += 4
            
        # 胜率评分 (10%)
        if win_rate > 70:
            score += 10
        elif win_rate > 60:
            score += 8
        elif win_rate > 55:
            score += 6
        elif win_rate > 50:
            score += 4
        elif win_rate > 45:
            score += 2
            
        # 评级
        if score >= 85:
            return "A+ 优秀"
        elif score >= 75:
            return "A 良好"
        elif score >= 65:
            return "B+ 中等偏上"
        elif score >= 55:
            return "B 中等"
        elif score >= 45:
            return "C+ 中等偏下"
        elif score >= 35:
            return "C 较差"
        else:
            return "D 不合格"
            
    def generate_comparison_report(self) -> str:
        """生成对比分析报告"""
        if not self.backtest_results:
            return "无回测结果可供分析"
            
        self.logger.info("生成对比分析报告")
        
        report = []
        report.append("=" * 80)
        report.append("Enhanced Grid Strategy 回测分析报告")
        report.append("=" * 80)
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 汇总表格
        report.append("回测结果汇总:")
        report.append("-" * 80)
        report.append(f"{'时期':<15} {'收益率(%)':<12} {'最大回撤(%)':<12} {'胜率(%)':<10} {'夏普比率':<10} {'评级':<15}")
        report.append("-" * 80)
        
        for period_name, result in self.backtest_results.items():
            if result.get("success"):
                metrics = self.calculate_performance_metrics(result)
                basic = metrics.get("基础指标", {})
                risk = metrics.get("风险指标", {})
                
                report.append(
                    f"{period_name:<15} "
                    f"{basic.get('总收益率(%)', 0):<12.2f} "
                    f"{basic.get('最大回撤(%)', 0):<12.2f} "
                    f"{basic.get('胜率(%)', 0):<10.2f} "
                    f"{risk.get('夏普比率', 0):<10.2f} "
                    f"{metrics.get('评估等级', 'N/A'):<15}"
                )
            else:
                report.append(f"{period_name:<15} {'失败':<60}")
                
        report.append("-" * 80)
        report.append("")
        
        # 详细分析
        for period_name, result in self.backtest_results.items():
            if result.get("success"):
                report.append(f"【{period_name}】 详细分析:")
                report.append(f"时间范围: {result['start_date']} 至 {result['end_date']}")
                
                metrics = self.calculate_performance_metrics(result)
                for category, values in metrics.items():
                    if isinstance(values, dict):
                        report.append(f"\n{category}:")
                        for key, value in values.items():
                            if isinstance(value, (int, float)):
                                report.append(f"  {key}: {value:.2f}")
                            else:
                                report.append(f"  {key}: {value}")
                    else:
                        report.append(f"{category}: {values}")
                        
                report.append("")
                
        # 总结和建议
        report.append("总结和建议:")
        report.append("-" * 40)
        
        # 计算整体表现
        successful_results = [r for r in self.backtest_results.values() if r.get("success")]
        if successful_results:
            avg_return = np.mean([r["data"].get("total_profit_pct", 0) for r in successful_results])
            avg_drawdown = np.mean([abs(r["data"].get("max_drawdown_pct", 0)) for r in successful_results])
            
            if avg_return > 20 and avg_drawdown < 15:
                report.append("✓ 策略整体表现良好，建议继续优化参数")
            elif avg_return > 10:
                report.append("✓ 策略有一定盈利能力，建议关注风险控制")
            else:
                report.append("⚠ 策略表现需要改进，建议重新评估参数设置")
                
            if avg_drawdown > 20:
                report.append("⚠ 最大回撤较高，建议加强风险管理")
                
        else:
            report.append("⚠ 所有回测均失败，请检查策略配置和数据质量")
            
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
        
    def save_results(self) -> str:
        """保存回测结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存详细结果到JSON
        results_file = self.results_dir / f"backtest_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.backtest_results, f, ensure_ascii=False, indent=2)
            
        # 保存分析报告
        report = self.generate_comparison_report()
        report_file = self.results_dir / f"backtest_report_{timestamp}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
            
        self.logger.info(f"结果已保存: {results_file}, {report_file}")
        return str(report_file)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Enhanced Grid Strategy 回测工具')
    parser.add_argument('--config', '-c', type=str, 
                       help='配置文件路径 (默认: configs/strategy-21-base.json)')
    parser.add_argument('--period', '-p', type=str,
                       choices=['full', 'bull', 'bear', 'recent', 'all'],
                       default='all',
                       help='回测时期选择')
    parser.add_argument('--output-dir', '-o', type=str,
                       help='输出目录 (默认: backtest_results)')
                       
    args = parser.parse_args()
    
    try:
        # 创建回测器
        backtester = EnhancedGridBacktester(config_path=args.config)
        
        # 执行回测
        print("\n开始回测分析...")
        results = backtester.run_comprehensive_backtest()
        
        # 生成报告
        print("\n生成分析报告...")
        report_file = backtester.save_results()
        
        # 显示简要结果
        print("\n" + backtester.generate_comparison_report())
        print(f"\n详细报告已保存至: {report_file}")
        
        # 检查是否所有回测都成功
        successful_count = sum(1 for r in results.values() if r.get("success"))
        total_count = len(results)
        
        print(f"\n回测完成: {successful_count}/{total_count} 个时期成功")
        
        if successful_count == 0:
            print("⚠️  所有回测均失败，请检查配置和数据")
            sys.exit(1)
        elif successful_count < total_count:
            print("⚠️  部分回测失败，请查看日志了解详情")
            sys.exit(1)
        else:
            print("✅ 所有回测成功完成")
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n用户中断回测")
        sys.exit(1)
    except Exception as e:
        print(f"\n回测执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()