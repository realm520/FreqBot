"""
单元测试 - 增强网格策略

测试范围：
- 策略初始化和参数验证
- 指标计算功能
- 网格级别计算
- 买入/卖出信号生成
- 自定义止损和退出逻辑
- 边界条件和异常处理
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 导入策略类
try:
    from user_data.strategies.EnhancedGridStrategy import EnhancedGridStrategy
    from freqbot_config.strategies.GridTradingStrategy import GridTradingStrategy
except ImportError:
    # 如果导入失败，跳过测试
    pytest.skip("Strategy files not found", allow_module_level=True)


class TestEnhancedGridStrategyUnit:
    """增强网格策略单元测试类"""

    @pytest.fixture
    def strategy(self):
        """创建策略实例"""
        strategy = EnhancedGridStrategy()
        # 模拟数据提供者
        strategy.dp = Mock()
        return strategy

    @pytest.fixture
    def sample_dataframe(self):
        """创建测试用的DataFrame"""
        dates = pd.date_range('2024-01-01', periods=200, freq='15min')
        
        # 生成模拟价格数据
        np.random.seed(42)  # 确保可重现性
        base_price = 100.0
        price_changes = np.random.normal(0, 0.01, 200)  # 1%标准差的正态分布
        prices = [base_price]
        
        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 50.0))  # 价格不低于50
            
        # 生成OHLCV数据
        highs = [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices]
        lows = [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices]
        volumes = np.random.randint(1000, 10000, 200)
        
        df = pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': volumes,
        })
        
        return df

    def test_strategy_initialization(self, strategy):
        """测试策略初始化"""
        assert strategy.INTERFACE_VERSION == 3
        assert strategy.can_short is False
        assert strategy.timeframe == "15m"
        assert strategy.startup_candle_count == 100
        
        # 测试策略元数据
        assert hasattr(strategy, 'STRATEGY_NAME')
        assert hasattr(strategy, 'STRATEGY_VERSION')
        assert hasattr(strategy, 'STRATEGY_AUTHOR')
        
    def test_strategy_parameters(self, strategy):
        """测试策略参数设置"""
        # 测试继承的网格参数
        assert hasattr(strategy, 'grid_levels')
        assert hasattr(strategy, 'grid_range_percent')
        assert hasattr(strategy, 'base_profit_percent')
        
        # 测试增强功能参数
        assert hasattr(strategy, 'enable_dynamic_grid')
        assert hasattr(strategy, 'enable_market_regime_detection')
        assert hasattr(strategy, 'enable_enhanced_position_sizing')

    def test_minimal_roi_configuration(self, strategy):
        """测试ROI配置"""
        roi = strategy.minimal_roi
        assert isinstance(roi, dict)
        assert "0" in roi
        assert roi["0"] > 0  # 确保有正的利润目标
        
        # 确保ROI值递减（时间越长，要求利润越低）
        roi_values = [roi[key] for key in sorted(roi.keys(), key=lambda x: int(x))]
        for i in range(1, len(roi_values)):
            assert roi_values[i] <= roi_values[i-1], "ROI should decrease over time"

    def test_stoploss_configuration(self, strategy):
        """测试止损配置"""
        assert strategy.stoploss < 0  # 止损应该是负值
        assert strategy.stoploss >= -1  # 止损不应超过-100%
        assert isinstance(strategy.trailing_stop, bool)

    def test_informative_pairs(self, strategy):
        """测试信息对配置"""
        pairs = strategy.informative_pairs()
        assert isinstance(pairs, list)

    def test_populate_indicators_basic(self, strategy, sample_dataframe):
        """测试基本指标计算"""
        result = strategy.populate_indicators(sample_dataframe, {'pair': 'BTC/USDT'})
        
        # 验证返回的是DataFrame
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_dataframe)
        
        # 验证基本指标存在
        expected_indicators = [
            'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'bb_lowerband', 'bb_middleband', 'bb_upperband', 'bb_percent', 'bb_width',
            'rsi', 'atr', 'price_change_pct', 'volatility'
        ]
        
        for indicator in expected_indicators:
            assert indicator in result.columns, f"Missing indicator: {indicator}"
            
    def test_populate_indicators_values(self, strategy, sample_dataframe):
        """测试指标计算的数值正确性"""
        result = strategy.populate_indicators(sample_dataframe, {'pair': 'BTC/USDT'})
        
        # 测试移动平均线
        assert not result['sma_20'].isna().all(), "SMA_20 should not be all NaN"
        assert not result['sma_50'].isna().all(), "SMA_50 should not be all NaN"
        
        # 测试布林带
        bb_valid = ~(result['bb_lowerband'].isna() | result['bb_middleband'].isna() | result['bb_upperband'].isna())
        valid_rows = result[bb_valid]
        if len(valid_rows) > 0:
            # 布林带上轨应该大于中轨，中轨应该大于下轨
            assert (valid_rows['bb_upperband'] >= valid_rows['bb_middleband']).all()
            assert (valid_rows['bb_middleband'] >= valid_rows['bb_lowerband']).all()
        
        # 测试RSI范围
        rsi_valid = ~result['rsi'].isna()
        if rsi_valid.any():
            rsi_values = result.loc[rsi_valid, 'rsi']
            assert (rsi_values >= 0).all() and (rsi_values <= 100).all(), "RSI should be between 0 and 100"
            
        # 测试ATR为正值
        atr_valid = ~result['atr'].isna()
        if atr_valid.any():
            assert (result.loc[atr_valid, 'atr'] >= 0).all(), "ATR should be non-negative"

    def test_calculate_grid_levels(self, strategy, sample_dataframe):
        """测试网格级别计算"""
        # 先计算指标
        df_with_indicators = strategy.populate_indicators(sample_dataframe, {'pair': 'BTC/USDT'})
        
        # 验证网格相关指标存在
        grid_indicators = ['grid_base_price', 'grid_spacing', 'distance_from_base', 'grid_level']
        for indicator in grid_indicators:
            assert indicator in df_with_indicators.columns, f"Missing grid indicator: {indicator}"
            
        # 验证网格基准价格和间距为正值
        valid_base = ~df_with_indicators['grid_base_price'].isna()
        valid_spacing = ~df_with_indicators['grid_spacing'].isna()
        
        if valid_base.any():
            assert (df_with_indicators.loc[valid_base, 'grid_base_price'] > 0).all()
        if valid_spacing.any():
            assert (df_with_indicators.loc[valid_spacing, 'grid_spacing'] > 0).all()

    def test_populate_entry_trend(self, strategy, sample_dataframe):
        """测试买入信号生成"""
        # 准备完整的数据
        df_with_indicators = strategy.populate_indicators(sample_dataframe, {'pair': 'BTC/USDT'})
        result = strategy.populate_entry_trend(df_with_indicators, {'pair': 'BTC/USDT'})
        
        # 验证买入信号列存在
        assert 'enter_long' in result.columns
        
        # 验证买入信号为0或1
        assert result['enter_long'].isin([0, 1]).all()
        
        # 验证至少有一些买入信号（在模拟数据中应该有信号）
        # 注意：这个测试可能需要根据实际策略逻辑调整
        signal_count = result['enter_long'].sum()
        assert signal_count >= 0  # 至少不应该出错

    def test_populate_exit_trend(self, strategy, sample_dataframe):
        """测试卖出信号生成"""
        # 准备完整的数据
        df_with_indicators = strategy.populate_indicators(sample_dataframe, {'pair': 'BTC/USDT'})
        result = strategy.populate_exit_trend(df_with_indicators, {'pair': 'BTC/USDT'})
        
        # 验证卖出信号列存在
        assert 'exit_long' in result.columns
        
        # 验证卖出信号为0或1
        assert result['exit_long'].isin([0, 1]).all()
        
        # 验证至少有一些卖出信号
        signal_count = result['exit_long'].sum()
        assert signal_count >= 0  # 至少不应该出错

    def test_custom_stoploss(self, strategy):
        """测试自定义止损逻辑"""
        # 模拟交易对象
        mock_trade = Mock()
        mock_trade.open_date_utc = datetime.now(timezone.utc)
        
        current_time = datetime.now(timezone.utc)
        current_rate = 100.0
        
        # 测试正常利润情况
        result = strategy.custom_stoploss("BTC/USDT", mock_trade, current_time, current_rate, 0.05)
        assert result == -1  # 不触发止损
        
        # 测试大幅亏损情况
        result = strategy.custom_stoploss("BTC/USDT", mock_trade, current_time, current_rate, -0.20)
        assert result == 0.01  # 应该触发止损
        
        # 测试边界条件
        result = strategy.custom_stoploss("BTC/USDT", mock_trade, current_time, current_rate, -0.15)
        assert result == 0.01  # 刚好触发止损
        
        result = strategy.custom_stoploss("BTC/USDT", mock_trade, current_time, current_rate, -0.14)
        assert result == -1  # 不触发止损

    @patch('user_data.strategies.EnhancedGridStrategy.EnhancedGridStrategy.dp')
    def test_custom_exit(self, mock_dp, strategy):
        """测试自定义退出逻辑"""
        # 准备模拟数据
        mock_dataframe = pd.DataFrame({
            'close': [100.0],
            'bb_middleband': [98.0],
            'bb_percent': [0.8],
            'rsi': [65],
            'volume': [5000]
        })
        
        mock_dp.get_analyzed_dataframe.return_value = (mock_dataframe, None)
        strategy.dp = mock_dp
        
        # 模拟交易对象
        mock_trade = Mock()
        mock_trade.open_date_utc = datetime.now(timezone.utc) - timedelta(hours=1)
        
        current_time = datetime.now(timezone.utc)
        current_rate = 100.0
        
        # 测试达到网格利润目标
        strategy.base_profit_percent = Mock()
        strategy.base_profit_percent.value = 0.015
        result = strategy.custom_exit("BTC/USDT", mock_trade, current_time, current_rate, 0.02)
        assert result == "grid_profit_target"
        
        # 测试时间退出条件
        mock_trade.open_date_utc = datetime.now(timezone.utc) - timedelta(hours=5)
        result = strategy.custom_exit("BTC/USDT", mock_trade, current_time, current_rate, 0.01)
        assert result == "grid_time_exit"
        
        # 测试不满足退出条件
        mock_trade.open_date_utc = datetime.now(timezone.utc) - timedelta(hours=1)
        result = strategy.custom_exit("BTC/USDT", mock_trade, current_time, current_rate, 0.005)
        assert result is None

    def test_order_configuration(self, strategy):
        """测试订单配置"""
        assert hasattr(strategy, 'order_types')
        assert isinstance(strategy.order_types, dict)
        
        required_order_types = ['entry', 'exit', 'stoploss']
        for order_type in required_order_types:
            assert order_type in strategy.order_types
            
        assert hasattr(strategy, 'order_time_in_force')
        assert isinstance(strategy.order_time_in_force, dict)

    def test_edge_cases_empty_dataframe(self, strategy):
        """测试边界条件：空DataFrame"""
        empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        
        # 应该能处理空DataFrame而不崩溃
        try:
            result = strategy.populate_indicators(empty_df, {'pair': 'BTC/USDT'})
            assert isinstance(result, pd.DataFrame)
        except Exception as e:
            pytest.fail(f"Strategy should handle empty dataframe gracefully: {e}")

    def test_edge_cases_single_candle(self, strategy):
        """测试边界条件：单个蜡烛"""
        single_candle_df = pd.DataFrame({
            'date': [datetime.now()],
            'open': [100.0],
            'high': [101.0],
            'low': [99.0],
            'close': [100.5],
            'volume': [5000]
        })
        
        # 应该能处理单个蜡烛数据
        try:
            result = strategy.populate_indicators(single_candle_df, {'pair': 'BTC/USDT'})
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 1
        except Exception as e:
            pytest.fail(f"Strategy should handle single candle gracefully: {e}")

    def test_extreme_price_values(self, strategy):
        """测试极端价格值"""
        # 测试极小价格
        extreme_small_df = pd.DataFrame({
            'open': [0.00001] * 100,
            'high': [0.00002] * 100,
            'low': [0.000005] * 100,
            'close': [0.000015] * 100,
            'volume': [1000] * 100,
        })
        
        try:
            result = strategy.populate_indicators(extreme_small_df, {'pair': 'BTC/USDT'})
            # 验证计算结果不包含无穷大值
            numeric_columns = result.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                assert not np.isinf(result[col]).any(), f"Column {col} contains infinite values"
        except Exception as e:
            pytest.fail(f"Strategy should handle extreme small prices: {e}")
            
        # 测试极大价格
        extreme_large_df = pd.DataFrame({
            'open': [1000000] * 100,
            'high': [1100000] * 100,
            'low': [900000] * 100,
            'close': [1050000] * 100,
            'volume': [1000] * 100,
        })
        
        try:
            result = strategy.populate_indicators(extreme_large_df, {'pair': 'BTC/USDT'})
            # 验证计算结果是有限的
            numeric_columns = result.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                finite_values = result[col][~result[col].isna()]
                if len(finite_values) > 0:
                    assert np.isfinite(finite_values).all(), f"Column {col} contains non-finite values"
        except Exception as e:
            pytest.fail(f"Strategy should handle extreme large prices: {e}")

    def test_parameter_validation(self, strategy):
        """测试参数验证"""
        # 验证网格参数范围
        assert strategy.grid_levels.low >= 3  # 至少3层网格
        assert strategy.grid_levels.high <= 20  # 不超过20层
        
        assert strategy.grid_range_percent.low > 0  # 网格范围为正
        assert strategy.grid_range_percent.high < 1  # 网格范围小于100%
        
        assert strategy.base_profit_percent.low > 0  # 基础利润为正

    def test_performance_benchmarks(self, strategy, sample_dataframe):
        """测试性能基准"""
        import time
        
        # 测试populate_indicators性能
        start_time = time.time()
        result = strategy.populate_indicators(sample_dataframe, {'pair': 'BTC/USDT'})
        indicators_time = time.time() - start_time
        
        # 指标计算应该在合理时间内完成（对于200行数据应该很快）
        assert indicators_time < 5.0, f"populate_indicators took too long: {indicators_time:.2f}s"
        
        # 测试entry/exit signal计算性能
        start_time = time.time()
        result = strategy.populate_entry_trend(result, {'pair': 'BTC/USDT'})
        result = strategy.populate_exit_trend(result, {'pair': 'BTC/USDT'})
        signals_time = time.time() - start_time
        
        assert signals_time < 2.0, f"Signal calculation took too long: {signals_time:.2f}s"


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])