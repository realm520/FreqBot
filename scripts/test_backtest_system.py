#!/usr/bin/env python3
"""
测试回测系统
在没有FreqTrade的情况下测试回测脚本的完整性
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

def test_backtest_system():
    """测试回测系统"""
    print("=" * 60)
    print("Enhanced Grid Strategy 回测系统测试")
    print("=" * 60)
    
    success_count = 0
    total_tests = 0
    
    # 测试1: 导入回测脚本
    total_tests += 1
    try:
        from scripts.backtest_enhanced_grid import EnhancedGridBacktester
        print("✅ 测试1: 导入主回测脚本 - 成功")
        success_count += 1
    except Exception as e:
        print(f"❌ 测试1: 导入主回测脚本 - 失败: {e}")
    
    # 测试2: 导入场景测试脚本
    total_tests += 1
    try:
        from scripts.backtest_scenarios import ScenarioBacktester
        print("✅ 测试2: 导入场景测试脚本 - 成功")
        success_count += 1
    except Exception as e:
        print(f"❌ 测试2: 导入场景测试脚本 - 失败: {e}")
    
    # 测试3: 导入分析脚本
    total_tests += 1
    try:
        from scripts.analyze_backtest_results import BacktestResultAnalyzer
        print("✅ 测试3: 导入分析脚本 - 成功")
        success_count += 1
    except Exception as e:
        print(f"❌ 测试3: 导入分析脚本 - 失败: {e}")
    
    # 测试4: 初始化回测器
    total_tests += 1
    try:
        backtester = EnhancedGridBacktester()
        print("✅ 测试4: 初始化回测器 - 成功")
        print(f"   配置文件: {backtester.config_path}")
        print(f"   结果目录: {backtester.results_dir}")
        success_count += 1
    except Exception as e:
        print(f"❌ 测试4: 初始化回测器 - 失败: {e}")
    
    # 测试5: 场景定义
    total_tests += 1
    try:
        scenario_tester = ScenarioBacktester()
        scenarios = scenario_tester.scenarios
        print(f"✅ 测试5: 场景定义 - 成功 (共{len(scenarios)}个场景)")
        for i, (name, config) in enumerate(list(scenarios.items())[:3]):
            print(f"   {i+1}. {config.get('name', name)}: {config.get('description', '无描述')}")
        if len(scenarios) > 3:
            print(f"   ... 还有{len(scenarios)-3}个场景")
        success_count += 1
    except Exception as e:
        print(f"❌ 测试5: 场景定义 - 失败: {e}")
    
    # 测试6: 配置文件加载
    total_tests += 1
    try:
        config_file = PROJECT_ROOT / "configs/backtest/enhanced_grid_backtest.json"
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print("✅ 测试6: 配置文件加载 - 成功")
            print(f"   策略: {config.get('strategy', {}).get('strategy_class', 'Unknown')}")
            print(f"   回测时期: {len(config.get('backtest_periods', {}))}个")
            success_count += 1
        else:
            print("❌ 测试6: 配置文件加载 - 失败: 文件不存在")
    except Exception as e:
        print(f"❌ 测试6: 配置文件加载 - 失败: {e}")
    
    # 测试7: 结果分析器
    total_tests += 1
    try:
        analyzer = BacktestResultAnalyzer()
        print("✅ 测试7: 结果分析器 - 成功")
        print(f"   报告目录: {analyzer.reports_dir}")
        success_count += 1
    except Exception as e:
        print(f"❌ 测试7: 结果分析器 - 失败: {e}")
    
    # 测试8: 模拟数据处理
    total_tests += 1
    try:
        analyzer = BacktestResultAnalyzer()
        sample_data = analyzer.create_sample_trades_data()
        if sample_data:
            import pandas as pd
            df = pd.DataFrame(sample_data)
            print(f"✅ 测试8: 模拟数据处理 - 成功 (生成{len(sample_data)}笔交易)")
            success_count += 1
        else:
            print("❌ 测试8: 模拟数据处理 - 失败: 无数据生成")
    except Exception as e:
        print(f"❌ 测试8: 模拟数据处理 - 失败: {e}")
    
    # 测试9: 性能指标计算
    total_tests += 1
    try:
        analyzer = BacktestResultAnalyzer()
        sample_data = analyzer.create_sample_trades_data()
        import pandas as pd
        trades_df = pd.DataFrame(sample_data)
        equity_curve = analyzer.build_equity_curve(trades_df)
        metrics = analyzer.calculate_advanced_metrics(trades_df, equity_curve)
        
        if metrics:
            print("✅ 测试9: 性能指标计算 - 成功")
            print(f"   总交易: {metrics.get('total_trades', 0)}")
            print(f"   胜率: {metrics.get('win_rate', 0):.1%}")
            print(f"   总收益: {metrics.get('total_return', 0):.2%}")
            print(f"   夏普比率: {metrics.get('sharpe_ratio', 0):.2f}")
            success_count += 1
        else:
            print("❌ 测试9: 性能指标计算 - 失败: 无指标生成")
    except Exception as e:
        print(f"❌ 测试9: 性能指标计算 - 失败: {e}")
    
    # 测试10: 文件系统权限
    total_tests += 1
    try:
        results_dir = PROJECT_ROOT / "backtest_results"
        results_dir.mkdir(exist_ok=True)
        test_file = results_dir / "test_file.txt"
        test_file.write_text("测试文件")
        test_file.unlink()
        print("✅ 测试10: 文件系统权限 - 成功")
        success_count += 1
    except Exception as e:
        print(f"❌ 测试10: 文件系统权限 - 失败: {e}")
    
    # 总结
    print("\n" + "=" * 60)
    print(f"测试总结: {success_count}/{total_tests} 通过")
    
    if success_count == total_tests:
        print("🎉 所有测试通过! 回测系统就绪")
        status = "完全可用"
        color = "✅"
    elif success_count >= total_tests * 0.8:
        print("⚠️  大部分测试通过，系统基本可用")
        status = "基本可用"
        color = "⚠️"
    else:
        print("❌ 多个测试失败，系统可能有问题")
        status = "需要修复"
        color = "❌"
    
    print(f"\n{color} 系统状态: {status}")
    
    # 使用说明
    print("\n📋 使用说明:")
    print("1. 安装FreqTrade后可运行完整回测:")
    print("   python scripts/backtest_enhanced_grid.py")
    print("2. 运行场景测试:")
    print("   python scripts/backtest_scenarios.py")
    print("3. 分析回测结果:")
    print("   python scripts/analyze_backtest_results.py")
    print("4. 查看配置文件:")
    print("   configs/backtest/enhanced_grid_backtest.json")
    
    if success_count < total_tests:
        print("\n⚠️  如需完整功能，请:")
        print("1. 安装ta-lib库: sudo apt-get install libta-lib-dev")
        print("2. 安装FreqTrade: uv add freqtrade")
        print("3. 下载历史数据: ./docker_download_data.sh")
    
    print("=" * 60)
    
    return success_count / total_tests

def create_sample_report():
    """创建示例分析报告"""
    try:
        from scripts.analyze_backtest_results import BacktestResultAnalyzer
        analyzer = BacktestResultAnalyzer()
        
        # 创建示例数据
        sample_data = analyzer.create_sample_trades_data()
        import pandas as pd
        trades_df = pd.DataFrame(sample_data)
        equity_curve = analyzer.build_equity_curve(trades_df)
        metrics = analyzer.calculate_advanced_metrics(trades_df, equity_curve)
        comparison = analyzer.compare_with_benchmark(metrics)
        
        analysis_data = {
            'trades': trades_df,
            'equity_curve': equity_curve, 
            'metrics': metrics,
            'comparison': comparison
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 生成HTML报告
        html_report = analyzer.generate_html_report(analysis_data, timestamp)
        print(f"\n📊 示例报告已生成: {html_report}")
        
        return html_report
        
    except Exception as e:
        print(f"\n❌ 示例报告生成失败: {e}")
        return None

if __name__ == "__main__":
    # 运行系统测试
    score = test_backtest_system()
    
    # 如果测试通过率超过80%，生成示例报告
    if score >= 0.8:
        print("\n生成示例分析报告...")
        sample_report = create_sample_report()
    
    sys.exit(0 if score >= 0.8 else 1)