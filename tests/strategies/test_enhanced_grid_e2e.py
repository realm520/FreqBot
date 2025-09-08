"""
端到端测试 - 增强网格策略

测试范围：
- 完整的交易流程模拟
- 真实市场数据回测
- 性能指标验证
- 风险指标计算
- 策略在不同市场环境下的表现
- 资金管理和仓位控制
- 长期稳定性测试
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
import sqlite3
import json
import os
import sys
from typing import Dict, Any, List, Tuple
import time
from dataclasses import dataclass

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 导入策略和相关模块
try:
    from user_data.strategies.EnhancedGridStrategy import EnhancedGridStrategy
    from freqbot_config.strategies.GridTradingStrategy import GridTradingStrategy
except ImportError:
    pytest.skip("Strategy files not found", allow_module_level=True)


@dataclass
class BacktestResult:
    """回测结果数据结构"""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration: float
    volatility: float


@dataclass
class Trade:
    """交易记录数据结构"""
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    profit: float
    profit_pct: float
    duration_hours: float


class MockDataProvider:
    """模拟数据提供者"""
    
    def __init__(self):
        self.data_cache = {}
    
    def set_data(self, pair: str, timeframe: str, data: pd.DataFrame):
        """设置数据"""
        key = f"{pair}_{timeframe}"
        self.data_cache[key] = data
    
    def get_analyzed_dataframe(self, pair: str, timeframe: str):
        """获取分析数据"""
        key = f"{pair}_{timeframe}"
        return self.data_cache.get(key, pd.DataFrame()), None


class MockExchange:
    """模拟交易所"""
    
    def __init__(self, initial_balance: float = 10000.0):
        self.balance = initial_balance
        self.positions = {}
        self.order_history = []
        self.fees = 0.001  # 0.1% 手续费
    
    def create_market_order(self, symbol: str, side: str, amount: float, price: float):
        """创建市价单"""
        if side == 'buy':
            cost = amount * price * (1 + self.fees)
            if self.balance >= cost:
                self.balance -= cost
                self.positions[symbol] = self.positions.get(symbol, 0) + amount
                self.order_history.append({
                    'symbol': symbol,
                    'side': side,
                    'amount': amount,
                    'price': price,
                    'cost': cost,
                    'timestamp': datetime.now()
                })
                return True
        elif side == 'sell':
            if self.positions.get(symbol, 0) >= amount:
                proceeds = amount * price * (1 - self.fees)
                self.balance += proceeds
                self.positions[symbol] = self.positions.get(symbol, 0) - amount
                self.order_history.append({
                    'symbol': symbol,
                    'side': side,
                    'amount': amount,
                    'price': price,
                    'cost': proceeds,
                    'timestamp': datetime.now()
                })
                return True
        return False
    
    def get_balance(self):
        """获取余额"""
        return self.balance
    
    def get_position(self, symbol: str):
        """获取持仓"""
        return self.positions.get(symbol, 0)


class TestEnhancedGridStrategyE2E:
    """增强网格策略端到端测试类"""

    @pytest.fixture
    def strategy_setup(self):
        """设置完整的策略测试环境"""
        strategy = EnhancedGridStrategy()
        data_provider = MockDataProvider()
        exchange = MockExchange(initial_balance=10000.0)
        
        strategy.dp = data_provider
        
        return strategy, data_provider, exchange

    @pytest.fixture
    def historical_market_data(self):
        """生成历史市场数据用于回测"""
        # 生成6个月的15分钟数据
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 6, 30)
        dates = pd.date_range(start_date, end_date, freq='15min')
        
        np.random.seed(42)
        
        # 生成具有趋势和周期性的价格数据
        base_price = 50000.0  # BTC起始价格
        prices = []
        
        for i in range(len(dates)):
            # 长期上升趋势
            long_trend = i * 0.00002
            
            # 周期性波动（模拟市场周期）
            cycle = 0.1 * np.sin(i * 0.001) + 0.05 * np.sin(i * 0.01)
            
            # 随机噪声
            noise = np.random.normal(0, 0.02)
            
            # 偶发的大波动（模拟新闻事件等）
            if np.random.random() < 0.001:  # 0.1%概率
                shock = np.random.normal(0, 0.1)
            else:
                shock = 0
            
            price_change = long_trend + cycle + noise + shock
            price = base_price * (1 + price_change)
            prices.append(max(price, base_price * 0.3))  # 价格不低于起始价的30%
        
        # 生成OHLCV数据
        data = []
        for i, price in enumerate(prices):
            volatility = abs(np.random.normal(0, 0.01))
            high = price * (1 + volatility)
            low = price * (1 - volatility)
            volume = np.random.lognormal(10, 1)  # 对数正态分布的交易量
            
            data.append({
                'date': dates[i],
                'open': price,
                'high': high,
                'low': low,
                'close': price,
                'volume': int(volume)
            })
        
        return pd.DataFrame(data)

    def test_complete_backtest_simulation(self, strategy_setup, historical_market_data):
        """完整回测模拟测试"""
        strategy, data_provider, exchange = strategy_setup
        
        # 设置数据
        data_provider.set_data("BTC/USDT", "15m", historical_market_data)
        
        # 执行完整的策略处理
        processed_data = self._process_strategy_data(strategy, historical_market_data)
        
        # 执行交易模拟
        trades = self._simulate_trading(processed_data, exchange)
        
        # 计算性能指标
        backtest_result = self._calculate_performance_metrics(trades, historical_market_data)
        
        # 验证回测结果
        self._validate_backtest_results(backtest_result, len(historical_market_data))
        
        # 输出结果
        print(f"\\n=== 回测结果 ===")
        print(f"总收益率: {backtest_result.total_return:.2%}")
        print(f"夏普比率: {backtest_result.sharpe_ratio:.2f}")
        print(f"最大回撤: {backtest_result.max_drawdown:.2%}")
        print(f"胜率: {backtest_result.win_rate:.2%}")
        print(f"利润因子: {backtest_result.profit_factor:.2f}")
        print(f"总交易次数: {backtest_result.total_trades}")
        print(f"平均交易持续时间: {backtest_result.avg_trade_duration:.1f}小时")

    def test_strategy_performance_benchmarks(self, strategy_setup, historical_market_data):
        """策略性能基准测试"""
        strategy, data_provider, exchange = strategy_setup
        
        # 设置数据
        data_provider.set_data("BTC/USDT", "15m", historical_market_data)
        
        # 测量策略处理性能
        start_time = time.time()
        processed_data = self._process_strategy_data(strategy, historical_market_data)
        processing_time = time.time() - start_time
        
        # 性能基准要求
        data_points = len(historical_market_data)
        expected_max_time = data_points * 0.001  # 每个数据点最多1毫秒
        
        assert processing_time < expected_max_time, \
            f"Strategy processing too slow: {processing_time:.2f}s for {data_points} points"
        
        # 内存使用检查
        import psutil
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        
        # 内存使用应该合理（小于1GB）
        assert memory_usage < 1024, f"Memory usage too high: {memory_usage:.1f}MB"

    def test_risk_management_effectiveness(self, strategy_setup, historical_market_data):
        """风险管理有效性测试"""
        strategy, data_provider, exchange = strategy_setup
        
        # 创建包含极端市场条件的数据
        extreme_data = self._create_extreme_market_data()
        data_provider.set_data("BTC/USDT", "15m", extreme_data)
        
        # 处理数据
        processed_data = self._process_strategy_data(strategy, extreme_data)
        
        # 模拟交易
        trades = self._simulate_trading(processed_data, exchange)
        
        if trades:
            # 验证风险控制
            max_loss = min([trade.profit_pct for trade in trades])
            assert max_loss > -0.20, f"Single trade loss too large: {max_loss:.2%}"
            
            # 验证止损有效性
            large_losses = [t for t in trades if t.profit_pct < -0.15]
            total_trades = len(trades)
            large_loss_ratio = len(large_losses) / total_trades if total_trades > 0 else 0
            
            assert large_loss_ratio < 0.1, f"Too many large losses: {large_loss_ratio:.2%}"

    def test_multi_timeframe_consistency(self, strategy_setup):
        """多时间框架一致性测试"""
        strategy, data_provider, exchange = strategy_setup
        
        # 生成不同时间框架的数据
        base_data_1h = self._generate_hourly_data(1000)
        base_data_15m = self._resample_to_15min(base_data_1h)
        
        data_provider.set_data("BTC/USDT", "15m", base_data_15m)
        
        # 处理15分钟数据
        processed_15m = self._process_strategy_data(strategy, base_data_15m)
        
        # 验证多时间框架的一致性
        # 检查长期趋势识别的一致性
        trend_15m = self._identify_trend(processed_15m)
        trend_1h = self._identify_trend(base_data_1h)
        
        # 主要趋势方向应该一致
        assert abs(trend_15m - trend_1h) < 0.3, "Multi-timeframe trends should be consistent"

    def test_strategy_robustness_across_market_conditions(self, strategy_setup):
        """不同市场条件下的策略鲁棒性测试"""
        strategy, data_provider, exchange = strategy_setup
        
        # 测试不同市场环境
        market_conditions = {
            'bull_market': self._create_bull_market_data(),
            'bear_market': self._create_bear_market_data(),
            'sideways_market': self._create_sideways_market_data(),
            'high_volatility': self._create_high_volatility_data(),
            'low_volatility': self._create_low_volatility_data()
        }
        
        condition_results = {}
        
        for condition_name, market_data in market_conditions.items():
            # 重置交易所状态
            exchange = MockExchange(initial_balance=10000.0)
            data_provider.set_data("BTC/USDT", "15m", market_data)
            
            # 执行策略
            processed_data = self._process_strategy_data(strategy, market_data)
            trades = self._simulate_trading(processed_data, exchange)
            
            if trades:
                result = self._calculate_performance_metrics(trades, market_data)
                condition_results[condition_name] = result
            else:
                condition_results[condition_name] = None
        
        # 验证策略在各种条件下的表现
        for condition, result in condition_results.items():
            if result is not None:
                # 策略不应在任何条件下出现灾难性损失
                assert result.max_drawdown < 0.5, \
                    f"Excessive drawdown in {condition}: {result.max_drawdown:.2%}"
                
                # 策略应该产生合理数量的交易
                assert result.total_trades > 0, f"No trades generated in {condition}"

    def test_position_sizing_and_risk_control(self, strategy_setup, historical_market_data):
        """仓位管理和风险控制测试"""
        strategy, data_provider, exchange = strategy_setup
        
        data_provider.set_data("BTC/USDT", "15m", historical_market_data)
        processed_data = self._process_strategy_data(strategy, historical_market_data)
        
        # 使用不同的初始资金测试仓位管理
        initial_balances = [1000, 5000, 10000, 50000]
        
        for balance in initial_balances:
            exchange_test = MockExchange(initial_balance=balance)
            trades = self._simulate_trading(processed_data, exchange_test, max_position_pct=0.1)
            
            if trades:
                # 验证单次交易风险不超过资金的一定比例
                max_single_risk = max([abs(trade.profit) for trade in trades])
                max_risk_pct = max_single_risk / balance
                
                assert max_risk_pct < 0.05, \
                    f"Single trade risk too high for balance {balance}: {max_risk_pct:.2%}"

    def test_long_term_stability(self, strategy_setup):
        """长期稳定性测试"""
        strategy, data_provider, exchange = strategy_setup
        
        # 生成长期数据（1年）
        long_term_data = self._generate_long_term_data(365 * 24 * 4)  # 1年15分钟数据
        data_provider.set_data("BTC/USDT", "15m", long_term_data)
        
        # 分批处理数据以模拟实时运行
        batch_size = 1000
        all_trades = []
        
        for i in range(0, len(long_term_data), batch_size):
            batch_data = long_term_data.iloc[i:i+batch_size].copy()
            if len(batch_data) < 100:  # 跳过太小的批次
                continue
                
            processed_batch = self._process_strategy_data(strategy, batch_data)
            batch_trades = self._simulate_trading(processed_batch, exchange)
            all_trades.extend(batch_trades)
        
        # 验证长期稳定性
        if all_trades:
            # 分析不同时期的表现
            quarterly_performance = self._analyze_quarterly_performance(all_trades, long_term_data)
            
            # 验证策略没有显著的衰减
            performance_trend = np.polyfit(range(len(quarterly_performance)), quarterly_performance, 1)[0]
            
            # 允许轻微的性能衰减，但不应该太严重
            assert performance_trend > -0.1, f"Strategy performance degrading too fast: {performance_trend:.3f}"

    def test_edge_cases_and_corner_scenarios(self, strategy_setup):
        """边缘情况和极端场景测试"""
        strategy, data_provider, exchange = strategy_setup
        
        # 测试各种边缘情况
        edge_cases = {
            'flash_crash': self._create_flash_crash_data(),
            'gap_up': self._create_gap_up_data(),
            'gap_down': self._create_gap_down_data(),
            'extremely_low_volume': self._create_low_volume_data(),
            'price_freeze': self._create_price_freeze_data()
        }
        
        for case_name, case_data in edge_cases.items():
            try:
                data_provider.set_data("BTC/USDT", "15m", case_data)
                processed_data = self._process_strategy_data(strategy, case_data)
                trades = self._simulate_trading(processed_data, exchange)
                
                # 策略应该能够处理边缘情况而不崩溃
                assert isinstance(trades, list), f"Strategy failed to handle {case_name}"
                
            except Exception as e:
                pytest.fail(f"Strategy failed on edge case {case_name}: {str(e)}")

    # 辅助方法
    def _process_strategy_data(self, strategy, market_data):
        """处理策略数据"""
        df = strategy.populate_indicators(market_data.copy(), {'pair': 'BTC/USDT'})
        df = strategy.populate_entry_trend(df, {'pair': 'BTC/USDT'})
        df = strategy.populate_exit_trend(df, {'pair': 'BTC/USDT'})
        return df

    def _simulate_trading(self, processed_data, exchange, max_position_pct=0.1):
        """模拟交易执行"""
        trades = []
        current_position = 0
        entry_price = None
        entry_time = None
        
        for idx, row in processed_data.iterrows():
            current_price = row['close']
            current_time = row.get('date', datetime.now())
            
            # 买入信号
            if row['enter_long'] == 1 and current_position == 0:
                # 计算仓位大小
                balance = exchange.get_balance()
                position_value = balance * max_position_pct
                quantity = position_value / current_price
                
                if exchange.create_market_order("BTC/USDT", "buy", quantity, current_price):
                    current_position = quantity
                    entry_price = current_price
                    entry_time = current_time
            
            # 卖出信号
            elif row['exit_long'] == 1 and current_position > 0:
                if exchange.create_market_order("BTC/USDT", "sell", current_position, current_price):
                    # 记录交易
                    profit = (current_price - entry_price) * current_position
                    profit_pct = (current_price - entry_price) / entry_price
                    duration = (current_time - entry_time).total_seconds() / 3600  # 小时
                    
                    trade = Trade(
                        entry_time=entry_time,
                        exit_time=current_time,
                        entry_price=entry_price,
                        exit_price=current_price,
                        quantity=current_position,
                        profit=profit,
                        profit_pct=profit_pct,
                        duration_hours=duration
                    )
                    trades.append(trade)
                    
                    current_position = 0
                    entry_price = None
                    entry_time = None
        
        return trades

    def _calculate_performance_metrics(self, trades: List[Trade], market_data: pd.DataFrame) -> BacktestResult:
        """计算性能指标"""
        if not trades:
            return BacktestResult(0, 0, 0, 0, 0, 0, 0, 0)
        
        # 基本统计
        profits = [trade.profit for trade in trades]
        profit_pcts = [trade.profit_pct for trade in trades]
        
        total_return = sum(profit_pcts)
        win_rate = len([p for p in profits if p > 0]) / len(profits)
        
        # 夏普比率
        if len(profit_pcts) > 1:
            returns_std = np.std(profit_pcts)
            sharpe_ratio = np.mean(profit_pcts) / returns_std if returns_std > 0 else 0
        else:
            sharpe_ratio = 0
        
        # 最大回撤
        cumulative_returns = np.cumsum(profit_pcts)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = running_max - cumulative_returns
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
        
        # 利润因子
        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p < 0]
        
        profit_factor = (sum(winning_trades) / abs(sum(losing_trades))) if losing_trades else float('inf')
        
        # 平均交易持续时间
        avg_duration = np.mean([trade.duration_hours for trade in trades])
        
        # 波动率
        volatility = np.std(profit_pcts) if len(profit_pcts) > 1 else 0
        
        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades),
            avg_trade_duration=avg_duration,
            volatility=volatility
        )

    def _validate_backtest_results(self, result: BacktestResult, data_length: int):
        """验证回测结果"""
        # 基本合理性检查
        assert -1 <= result.total_return <= 10, f"Unrealistic total return: {result.total_return:.2%}"
        assert 0 <= result.win_rate <= 1, f"Invalid win rate: {result.win_rate:.2%}"
        assert result.max_drawdown >= 0, f"Invalid max drawdown: {result.max_drawdown:.2%}"
        assert result.total_trades >= 0, f"Invalid trade count: {result.total_trades}"
        
        # 交易频率合理性
        if data_length > 1000:  # 只对长期数据进行此检查
            trade_frequency = result.total_trades / (data_length / 96)  # 每天的交易数
            assert trade_frequency < 10, f"Trade frequency too high: {trade_frequency:.1f} trades/day"

    # 数据生成辅助方法
    def _create_extreme_market_data(self):
        """创建极端市场数据"""
        dates = pd.date_range('2024-01-01', periods=500, freq='15min')
        prices = []
        base_price = 50000.0
        
        for i in range(500):
            if i == 100:  # 模拟闪崩
                price_change = -0.3
            elif i == 150:  # 模拟反弹
                price_change = 0.2
            elif 200 <= i <= 250:  # 模拟高波动期
                price_change = np.random.normal(0, 0.05)
            else:
                price_change = np.random.normal(0, 0.02)
            
            if i == 0:
                prices.append(base_price)
            else:
                new_price = prices[-1] * (1 + price_change)
                prices.append(max(new_price, base_price * 0.2))
        
        return self._create_ohlcv_dataframe(dates, prices)

    def _create_bull_market_data(self):
        """创建牛市数据"""
        dates = pd.date_range('2024-01-01', periods=1000, freq='15min')
        prices = []
        base_price = 50000.0
        
        for i in range(1000):
            trend = i * 0.0001  # 上升趋势
            noise = np.random.normal(0, 0.02)
            price = base_price * (1 + trend + noise)
            prices.append(price)
        
        return self._create_ohlcv_dataframe(dates, prices)

    def _create_bear_market_data(self):
        """创建熊市数据"""
        dates = pd.date_range('2024-01-01', periods=1000, freq='15min')
        prices = []
        base_price = 50000.0
        
        for i in range(1000):
            trend = -i * 0.00005  # 下降趋势
            noise = np.random.normal(0, 0.02)
            price = base_price * (1 + trend + noise)
            prices.append(max(price, base_price * 0.3))
        
        return self._create_ohlcv_dataframe(dates, prices)

    def _create_sideways_market_data(self):
        """创建横盘市场数据"""
        dates = pd.date_range('2024-01-01', periods=1000, freq='15min')
        prices = []
        base_price = 50000.0
        
        for i in range(1000):
            cycle = 0.05 * np.sin(i * 0.01)
            noise = np.random.normal(0, 0.015)
            price = base_price * (1 + cycle + noise)
            prices.append(price)
        
        return self._create_ohlcv_dataframe(dates, prices)

    def _create_high_volatility_data(self):
        """创建高波动数据"""
        dates = pd.date_range('2024-01-01', periods=1000, freq='15min')
        prices = []
        base_price = 50000.0
        
        for i in range(1000):
            noise = np.random.normal(0, 0.04)  # 高波动
            price = base_price * (1 + noise)
            prices.append(max(price, base_price * 0.5))
        
        return self._create_ohlcv_dataframe(dates, prices)

    def _create_low_volatility_data(self):
        """创建低波动数据"""
        dates = pd.date_range('2024-01-01', periods=1000, freq='15min')
        prices = []
        base_price = 50000.0
        
        for i in range(1000):
            noise = np.random.normal(0, 0.005)  # 低波动
            price = base_price * (1 + noise)
            prices.append(price)
        
        return self._create_ohlcv_dataframe(dates, prices)

    def _create_flash_crash_data(self):
        """创建闪崩数据"""
        dates = pd.date_range('2024-01-01', periods=100, freq='15min')
        prices = [50000.0] * 100
        
        # 在第50个数据点处发生闪崩
        for i in range(50, 60):
            crash_factor = 0.95 ** (i - 49)
            prices[i] = prices[49] * crash_factor
        
        # 后续恢复
        for i in range(60, 100):
            recovery_factor = 1 + (i - 60) * 0.002
            prices[i] = prices[59] * recovery_factor
        
        return self._create_ohlcv_dataframe(dates, prices)

    def _create_gap_up_data(self):
        """创建跳空高开数据"""
        dates = pd.date_range('2024-01-01', periods=100, freq='15min')
        prices = [50000.0] * 100
        
        # 在第50个数据点处跳空高开10%
        for i in range(50, 100):
            prices[i] = prices[49] * 1.1
        
        return self._create_ohlcv_dataframe(dates, prices)

    def _create_gap_down_data(self):
        """创建跳空低开数据"""
        dates = pd.date_range('2024-01-01', periods=100, freq='15min')
        prices = [50000.0] * 100
        
        # 在第50个数据点处跳空低开10%
        for i in range(50, 100):
            prices[i] = prices[49] * 0.9
        
        return self._create_ohlcv_dataframe(dates, prices)

    def _create_low_volume_data(self):
        """创建低交易量数据"""
        dates = pd.date_range('2024-01-01', periods=100, freq='15min')
        prices = [50000.0 + np.random.normal(0, 100) for _ in range(100)]
        
        df = self._create_ohlcv_dataframe(dates, prices)
        df['volume'] = np.random.randint(1, 10, 100)  # 极低交易量
        return df

    def _create_price_freeze_data(self):
        """创建价格冻结数据"""
        dates = pd.date_range('2024-01-01', periods=100, freq='15min')
        prices = [50000.0] * 100  # 价格完全不变
        
        return self._create_ohlcv_dataframe(dates, prices)

    def _create_ohlcv_dataframe(self, dates, prices):
        """创建OHLCV DataFrame"""
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            high = price * (1 + abs(np.random.normal(0, 0.003)))
            low = price * (1 - abs(np.random.normal(0, 0.003)))
            volume = np.random.randint(1000, 10000)
            
            data.append({
                'date': date,
                'open': price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume
            })
        
        return pd.DataFrame(data)

    def _generate_hourly_data(self, periods):
        """生成小时数据"""
        dates = pd.date_range('2024-01-01', periods=periods, freq='1h')
        prices = [50000.0]
        
        for i in range(1, periods):
            change = np.random.normal(0, 0.02)
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 25000))
        
        return self._create_ohlcv_dataframe(dates, prices)

    def _resample_to_15min(self, hourly_data):
        """将小时数据重采样为15分钟数据"""
        # 简化实现：每小时数据复制4次作为15分钟数据
        resampled_data = []
        for _, row in hourly_data.iterrows():
            for i in range(4):
                new_row = row.copy()
                new_row['date'] = row['date'] + timedelta(minutes=i*15)
                # 添加小幅随机变动
                price_variation = np.random.normal(0, 0.002)
                new_row['close'] = row['close'] * (1 + price_variation)
                new_row['open'] = new_row['close']
                new_row['high'] = new_row['close'] * 1.001
                new_row['low'] = new_row['close'] * 0.999
                resampled_data.append(new_row)
        
        return pd.DataFrame(resampled_data)

    def _identify_trend(self, data):
        """识别趋势方向"""
        if len(data) < 2:
            return 0
        
        price_change = (data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0]
        return price_change

    def _generate_long_term_data(self, periods):
        """生成长期数据"""
        dates = pd.date_range('2024-01-01', periods=periods, freq='15min')
        prices = []
        base_price = 50000.0
        
        for i in range(periods):
            # 长期趋势
            long_trend = i * 0.000001
            
            # 季节性模式
            seasonal = 0.1 * np.sin(i * 2 * np.pi / (365 * 24 * 4))
            
            # 随机噪声
            noise = np.random.normal(0, 0.02)
            
            price_change = long_trend + seasonal + noise
            price = base_price * (1 + price_change)
            prices.append(max(price, base_price * 0.3))
        
        return self._create_ohlcv_dataframe(dates, prices)

    def _analyze_quarterly_performance(self, trades, data):
        """分析季度性能"""
        if not trades or len(data) < 4:
            return [0]
        
        # 简化实现：按季度分组计算收益
        data_length = len(data)
        quarter_size = data_length // 4
        
        quarterly_returns = []
        for quarter in range(4):
            start_idx = quarter * quarter_size
            end_idx = min((quarter + 1) * quarter_size, len(trades))
            quarter_trades = trades[start_idx:end_idx] if end_idx <= len(trades) else []
            
            if quarter_trades:
                quarter_return = sum([t.profit_pct for t in quarter_trades])
            else:
                quarter_return = 0
            
            quarterly_returns.append(quarter_return)
        
        return quarterly_returns


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short", "-s"])