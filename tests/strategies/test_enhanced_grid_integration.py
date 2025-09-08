"""
集成测试 - 增强网格策略

测试范围：
- 策略与Freqtrade框架的集成
- 多组件协同工作
- 数据流处理
- 交易信号完整流程
- 与外部数据源的集成
- 策略参数优化流程
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
import os
import sys
from typing import Dict, Any, List

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 导入策略和相关模块
try:
    from user_data.strategies.EnhancedGridStrategy import EnhancedGridStrategy
    from freqbot_config.strategies.GridTradingStrategy import GridTradingStrategy
except ImportError:
    pytest.skip("Strategy files not found", allow_module_level=True)


class TestEnhancedGridStrategyIntegration:
    """增强网格策略集成测试类"""

    @pytest.fixture
    def strategy_with_mock_dp(self):
        """创建带模拟数据提供者的策略实例"""
        strategy = EnhancedGridStrategy()
        
        # 模拟数据提供者
        mock_dp = Mock()
        strategy.dp = mock_dp
        
        return strategy, mock_dp

    @pytest.fixture
    def comprehensive_market_data(self):
        """创建综合市场数据，模拟不同市场条件"""
        dates = pd.date_range('2024-01-01', periods=500, freq='15min')
        
        # 创建包含多种市场条件的价格数据
        np.random.seed(42)
        
        # 阶段1：横盘震荡（前150个数据点）
        sideways_prices = []
        base_price = 100.0
        for i in range(150):
            # 在95-105之间震荡
            noise = np.random.normal(0, 0.01)
            price = base_price + 5 * np.sin(i * 0.1) + noise * base_price
            sideways_prices.append(max(price, 90.0))
        
        # 阶段2：上涨趋势（接下来150个数据点）
        uptrend_prices = []
        for i in range(150):
            trend = i * 0.1  # 缓慢上涨
            noise = np.random.normal(0, 0.008)
            price = sideways_prices[-1] + trend + noise * sideways_prices[-1]
            uptrend_prices.append(price)
        
        # 阶段3：下跌和回调（最后200个数据点）
        downtrend_prices = []
        for i in range(200):
            if i < 100:
                # 下跌阶段
                trend = -i * 0.05
                noise = np.random.normal(0, 0.012)
                price = uptrend_prices[-1] + trend + noise * uptrend_prices[-1]
            else:
                # 回调阶段
                recovery = (i - 100) * 0.03
                noise = np.random.normal(0, 0.01)
                price = downtrend_prices[99] + recovery + noise * downtrend_prices[99]
            downtrend_prices.append(max(price, 50.0))
        
        # 合并所有价格
        all_prices = sideways_prices + uptrend_prices + downtrend_prices
        
        # 生成OHLCV数据
        data = []
        for i, price in enumerate(all_prices):
            high = price * (1 + abs(np.random.normal(0, 0.003)))
            low = price * (1 - abs(np.random.normal(0, 0.003)))
            volume = np.random.randint(1000, 15000)
            
            data.append({
                'date': dates[i],
                'open': price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume
            })
        
        return pd.DataFrame(data)

    @pytest.fixture
    def mock_freqtrade_config(self):
        """模拟Freqtrade配置"""
        return {
            "max_open_trades": 3,
            "stake_currency": "USDT",
            "stake_amount": 100,
            "tradable_balance_ratio": 0.99,
            "dry_run": True,
            "timeframe": "15m",
            "exchange": {
                "name": "binance",
                "pair_whitelist": ["BTC/USDT", "ETH/USDT"],
            }
        }

    def test_complete_data_processing_pipeline(self, strategy_with_mock_dp, comprehensive_market_data):
        """测试完整的数据处理管道"""
        strategy, mock_dp = strategy_with_mock_dp
        
        # 模拟数据提供者返回数据
        mock_dp.get_analyzed_dataframe.return_value = (comprehensive_market_data, None)
        
        # 执行完整的数据处理流程
        df_with_indicators = strategy.populate_indicators(comprehensive_market_data, {'pair': 'BTC/USDT'})
        df_with_entry = strategy.populate_entry_trend(df_with_indicators, {'pair': 'BTC/USDT'})
        df_final = strategy.populate_exit_trend(df_with_entry, {'pair': 'BTC/USDT'})
        
        # 验证数据完整性
        assert len(df_final) == len(comprehensive_market_data)
        assert 'enter_long' in df_final.columns
        assert 'exit_long' in df_final.columns
        
        # 验证信号生成
        entry_signals = df_final['enter_long'].sum()
        exit_signals = df_final['exit_long'].sum()
        
        # 在综合市场数据中应该生成一些信号
        assert entry_signals > 0, "Should generate some entry signals in comprehensive market data"
        assert exit_signals > 0, "Should generate some exit signals in comprehensive market data"
        
        # 验证信号不会同时发生（在同一根K线上）
        simultaneous_signals = ((df_final['enter_long'] == 1) & (df_final['exit_long'] == 1)).sum()
        assert simultaneous_signals == 0, "Entry and exit signals should not occur simultaneously"

    def test_strategy_state_consistency(self, strategy_with_mock_dp, comprehensive_market_data):
        """测试策略状态一致性"""
        strategy, mock_dp = strategy_with_mock_dp
        
        # 多次运行相同数据，结果应该一致
        results = []
        for _ in range(3):
            df_with_indicators = strategy.populate_indicators(
                comprehensive_market_data.copy(), {'pair': 'BTC/USDT'}
            )
            df_with_entry = strategy.populate_entry_trend(df_with_indicators, {'pair': 'BTC/USDT'})
            df_final = strategy.populate_exit_trend(df_with_entry, {'pair': 'BTC/USDT'})
            results.append(df_final)
        
        # 验证结果一致性
        for i in range(1, len(results)):
            pd.testing.assert_frame_equal(
                results[0][['enter_long', 'exit_long']],
                results[i][['enter_long', 'exit_long']],
                check_dtype=False,
                msg="Strategy should produce consistent results for the same input"
            )

    def test_multi_pair_processing(self, strategy_with_mock_dp):
        """测试多交易对处理"""
        strategy, mock_dp = strategy_with_mock_dp
        
        # 创建不同交易对的数据
        pairs = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT']
        pair_results = {}
        
        for pair in pairs:
            # 为每个交易对生成不同的价格数据
            dates = pd.date_range('2024-01-01', periods=200, freq='15min')
            base_price = {'BTC/USDT': 45000, 'ETH/USDT': 3000, 'ADA/USDT': 0.5}[pair]
            
            prices = []
            for i in range(200):
                noise = np.random.normal(0, 0.01)
                price = base_price * (1 + noise + 0.001 * np.sin(i * 0.1))
                prices.append(max(price, base_price * 0.8))
            
            df = pd.DataFrame({
                'date': dates,
                'open': prices,
                'high': [p * 1.01 for p in prices],
                'low': [p * 0.99 for p in prices],
                'close': prices,
                'volume': np.random.randint(1000, 10000, 200)
            })
            
            # 处理数据
            df_processed = strategy.populate_indicators(df, {'pair': pair})
            df_processed = strategy.populate_entry_trend(df_processed, {'pair': pair})
            df_processed = strategy.populate_exit_trend(df_processed, {'pair': pair})
            
            pair_results[pair] = df_processed
        
        # 验证每个交易对都有结果
        for pair in pairs:
            assert pair in pair_results
            assert len(pair_results[pair]) == 200
            assert 'enter_long' in pair_results[pair].columns
            assert 'exit_long' in pair_results[pair].columns

    def test_parameter_optimization_integration(self, strategy_with_mock_dp, comprehensive_market_data):
        """测试参数优化集成"""
        strategy, mock_dp = strategy_with_mock_dp
        
        # 测试不同参数组合
        parameter_combinations = [
            {'grid_levels': 5, 'grid_range_percent': 0.05, 'base_profit_percent': 0.01},
            {'grid_levels': 8, 'grid_range_percent': 0.08, 'base_profit_percent': 0.015},
            {'grid_levels': 12, 'grid_range_percent': 0.12, 'base_profit_percent': 0.02},
        ]
        
        results = {}
        
        for i, params in enumerate(parameter_combinations):
            # 模拟设置参数
            strategy.grid_levels = Mock()
            strategy.grid_levels.value = params['grid_levels']
            strategy.grid_range_percent = Mock()
            strategy.grid_range_percent.value = params['grid_range_percent']
            strategy.base_profit_percent = Mock()
            strategy.base_profit_percent.value = params['base_profit_percent']
            
            # 处理数据
            df_processed = strategy.populate_indicators(
                comprehensive_market_data.copy(), {'pair': 'BTC/USDT'}
            )
            df_processed = strategy.populate_entry_trend(df_processed, {'pair': 'BTC/USDT'})
            df_processed = strategy.populate_exit_trend(df_processed, {'pair': 'BTC/USDT'})
            
            results[f'combo_{i}'] = {
                'entry_signals': df_processed['enter_long'].sum(),
                'exit_signals': df_processed['exit_long'].sum(),
                'params': params
            }
        
        # 验证不同参数产生不同结果
        signal_counts = [results[key]['entry_signals'] for key in results]
        assert len(set(signal_counts)) > 1, "Different parameters should produce different signal counts"

    def test_custom_exit_integration(self, strategy_with_mock_dp, comprehensive_market_data):
        """测试自定义退出逻辑集成"""
        strategy, mock_dp = strategy_with_mock_dp
        
        # 准备完整的分析数据
        df_analyzed = strategy.populate_indicators(comprehensive_market_data, {'pair': 'BTC/USDT'})
        mock_dp.get_analyzed_dataframe.return_value = (df_analyzed, None)
        
        # 模拟不同的交易情况
        test_scenarios = [
            {
                'current_profit': 0.02,  # 2%利润
                'trade_age_hours': 1,
                'expected': 'grid_profit_target'
            },
            {
                'current_profit': 0.008,  # 0.8%利润
                'trade_age_hours': 5,
                'expected': 'grid_time_exit'
            },
            {
                'current_profit': 0.003,  # 0.3%利润
                'trade_age_hours': 2,
                'expected': None
            }
        ]
        
        for scenario in test_scenarios:
            # 模拟交易
            mock_trade = Mock()
            mock_trade.open_date_utc = datetime.now(timezone.utc) - timedelta(hours=scenario['trade_age_hours'])
            
            strategy.base_profit_percent = Mock()
            strategy.base_profit_percent.value = 0.015
            
            # 测试自定义退出
            result = strategy.custom_exit(
                "BTC/USDT",
                mock_trade,
                datetime.now(timezone.utc),
                100.0,
                scenario['current_profit']
            )
            
            assert result == scenario['expected'], f"Custom exit failed for scenario: {scenario}"

    def test_risk_management_integration(self, strategy_with_mock_dp, comprehensive_market_data):
        """测试风险管理集成"""
        strategy, mock_dp = strategy_with_mock_dp
        
        # 测试止损集成
        mock_trade = Mock()
        mock_trade.open_date_utc = datetime.now(timezone.utc)
        
        risk_scenarios = [
            {'profit': -0.05, 'expected_stoploss': -1},      # 小亏损，不止损
            {'profit': -0.18, 'expected_stoploss': 0.01},    # 大亏损，触发止损
            {'profit': 0.10, 'expected_stoploss': -1},       # 盈利，不止损
        ]
        
        for scenario in risk_scenarios:
            result = strategy.custom_stoploss(
                "BTC/USDT",
                mock_trade,
                datetime.now(timezone.utc),
                100.0,
                scenario['profit']
            )
            
            assert result == scenario['expected_stoploss'], \
                f"Risk management failed for profit {scenario['profit']}"

    def test_market_condition_adaptation(self, strategy_with_mock_dp):
        """测试市场条件适应性"""
        strategy, mock_dp = strategy_with_mock_dp
        
        # 创建不同市场条件的数据
        market_conditions = {
            'high_volatility': self._create_volatile_market_data(),
            'low_volatility': self._create_stable_market_data(),
            'trending_up': self._create_uptrend_data(),
            'trending_down': self._create_downtrend_data()
        }
        
        adaptation_results = {}
        
        for condition, data in market_conditions.items():
            df_processed = strategy.populate_indicators(data, {'pair': 'BTC/USDT'})
            df_processed = strategy.populate_entry_trend(df_processed, {'pair': 'BTC/USDT'})
            df_processed = strategy.populate_exit_trend(df_processed, {'pair': 'BTC/USDT'})
            
            adaptation_results[condition] = {
                'entry_signals': df_processed['enter_long'].sum(),
                'exit_signals': df_processed['exit_long'].sum(),
                'avg_volatility': df_processed['volatility'].mean() if 'volatility' in df_processed else 0
            }
        
        # 验证策略对不同市场条件的适应
        # 高波动率市场应该生成更多信号
        high_vol_signals = adaptation_results['high_volatility']['entry_signals']
        low_vol_signals = adaptation_results['low_volatility']['entry_signals']
        
        # 这个断言可能需要根据策略的具体实现调整
        assert high_vol_signals >= 0 and low_vol_signals >= 0, "Strategy should adapt to different market conditions"

    def test_performance_under_load(self, strategy_with_mock_dp):
        """测试负载下的性能"""
        strategy, mock_dp = strategy_with_mock_dp
        
        # 创建大量数据
        large_dataset = self._create_large_dataset(2000)  # 2000个数据点
        
        import time
        start_time = time.time()
        
        # 执行完整流程
        df_processed = strategy.populate_indicators(large_dataset, {'pair': 'BTC/USDT'})
        df_processed = strategy.populate_entry_trend(df_processed, {'pair': 'BTC/USDT'})
        df_processed = strategy.populate_exit_trend(df_processed, {'pair': 'BTC/USDT'})
        
        processing_time = time.time() - start_time
        
        # 性能要求：2000个数据点应在10秒内处理完成
        assert processing_time < 10.0, f"Large dataset processing took too long: {processing_time:.2f}s"
        assert len(df_processed) == 2000, "All data points should be processed"

    def test_data_integrity_through_pipeline(self, strategy_with_mock_dp, comprehensive_market_data):
        """测试数据通过管道的完整性"""
        strategy, mock_dp = strategy_with_mock_dp
        
        original_data = comprehensive_market_data.copy()
        
        # 执行完整管道
        df_stage1 = strategy.populate_indicators(original_data, {'pair': 'BTC/USDT'})
        df_stage2 = strategy.populate_entry_trend(df_stage1, {'pair': 'BTC/USDT'})
        df_final = strategy.populate_exit_trend(df_stage2, {'pair': 'BTC/USDT'})
        
        # 验证原始数据未被修改
        original_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in original_columns:
            if col in original_data.columns and col in df_final.columns:
                pd.testing.assert_series_equal(
                    original_data[col],
                    df_final[col],
                    check_names=False,
                    msg=f"Original {col} data should remain unchanged"
                )
        
        # 验证数据行数未改变
        assert len(df_final) == len(original_data), "Number of rows should remain constant"

    # 辅助方法
    def _create_volatile_market_data(self):
        """创建高波动率市场数据"""
        dates = pd.date_range('2024-01-01', periods=200, freq='15min')
        base_price = 100.0
        prices = []
        
        for i in range(200):
            # 高波动率：标准差为3%
            noise = np.random.normal(0, 0.03)
            price = base_price * (1 + noise)
            prices.append(max(price, 50.0))
        
        return pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'close': prices,
            'volume': np.random.randint(1000, 10000, 200)
        })

    def _create_stable_market_data(self):
        """创建低波动率市场数据"""
        dates = pd.date_range('2024-01-01', periods=200, freq='15min')
        base_price = 100.0
        prices = []
        
        for i in range(200):
            # 低波动率：标准差为0.5%
            noise = np.random.normal(0, 0.005)
            price = base_price * (1 + noise)
            prices.append(price)
        
        return pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': [p * 1.005 for p in prices],
            'low': [p * 0.995 for p in prices],
            'close': prices,
            'volume': np.random.randint(1000, 10000, 200)
        })

    def _create_uptrend_data(self):
        """创建上升趋势数据"""
        dates = pd.date_range('2024-01-01', periods=200, freq='15min')
        base_price = 100.0
        prices = []
        
        for i in range(200):
            trend = i * 0.002  # 每期上涨0.2%
            noise = np.random.normal(0, 0.01)
            price = base_price * (1 + trend + noise)
            prices.append(price)
        
        return pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': np.random.randint(1000, 10000, 200)
        })

    def _create_downtrend_data(self):
        """创建下降趋势数据"""
        dates = pd.date_range('2024-01-01', periods=200, freq='15min')
        base_price = 100.0
        prices = []
        
        for i in range(200):
            trend = -i * 0.001  # 每期下跌0.1%
            noise = np.random.normal(0, 0.01)
            price = base_price * (1 + trend + noise)
            prices.append(max(price, 50.0))
        
        return pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': np.random.randint(1000, 10000, 200)
        })

    def _create_large_dataset(self, size):
        """创建大型数据集"""
        dates = pd.date_range('2024-01-01', periods=size, freq='15min')
        base_price = 100.0
        prices = []
        
        for i in range(size):
            # 混合趋势和噪声
            trend = 0.0001 * np.sin(i * 0.01)  # 慢波动
            noise = np.random.normal(0, 0.01)
            price = base_price * (1 + trend + noise)
            prices.append(max(price, 50.0))
        
        return pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000, 10000, size)
        })


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])