# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Optional, Union, Dict
from collections import deque

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)

import talib.abstract as ta
from technical import qtpylib


class FundingRateEnhancedStrategy(IStrategy):
    """
    资金费率增强策略 - U本位永续合约
    
    策略定位：费率增强的方向性交易（非纯套利）
    
    核心逻辑：
    1. 资金费率作为主要Alpha来源
    2. 技术面多重过滤降低方向性风险
    3. 严格风控体系替代完全对冲
    4. 动态仓位管理基于费率极端度
    
    参考：标准资金费率套利的简化实现
    - 标准做法：永续合约 + 现货对冲（Delta中性）
    - 本策略：单边永续合约 + 严格风控（有限风险暴露）
    
    风险声明：
    - ⚠️ 这不是无风险套利！
    - ⚠️ 存在价格波动风险
    - ⚠️ 需要严格执行风控参数
    - ⚠️ 建议小资金测试后再扩大规模
    """
    
    INTERFACE_VERSION = 3
    can_short: bool = True  # 永续合约支持做空
    
    # ==================== 基础配置 ====================
    
    # 最小ROI（主要依赖策略信号，不依赖固定ROI）
    minimal_roi = {"0": 10}
    
    # 基础止损（会被custom_stoploss覆盖）
    stoploss = -0.02  # -2%硬止损
    
    # 追踪止损
    trailing_stop = False  # 不启用追踪止损，使用custom_stoploss实现
    
    # 时间框架
    timeframe = '15m'  # 15分钟周期，平衡信号频率和稳定性
    
    # 使用自定义止损和仓位管理
    use_custom_stoploss = True
    use_exit_signal = True
    exit_profit_only = False
    
    # ==================== 费率阈值参数（基于风险-收益矩阵）====================
    
    # 做空阈值（收取正费率）
    funding_rate_short_threshold = DecimalParameter(
        0.0002, 0.0005, default=0.0003, space="buy", optimize=True, load=True
    )  # 默认0.03%/8h ≈ 年化32%
    
    # 做多阈值（收取负费率）  
    funding_rate_long_threshold = DecimalParameter(
        -0.0008, -0.0003, default=0.0005, space="buy", optimize=True, load=True
    )  # 默认-0.05%/8h
    
    # 费率持续性要求（连续周期数）
    funding_persistence_periods = IntParameter(
        2, 5, default=3, space="buy", optimize=True, load=True
    )  # 默认连续3个8小时周期
    
    # ==================== 仓位管理参数（反直觉设计）====================
    
    # 基础仓位比例
    base_position_size = DecimalParameter(
        0.2, 0.5, default=0.3, space="buy", optimize=True, load=True
    )
    
    # 费率极端度调整系数（费率越高，系数越小）
    extreme_funding_factor = DecimalParameter(
        0.5, 0.9, default=0.7, space="buy", optimize=True, load=True
    )  # 极端费率时仓位缩减到70%
    
    # 波动率调整系数
    volatility_position_factor = DecimalParameter(
        0.5, 1.5, default=1.0, space="buy", optimize=True, load=True
    )
    
    # 最大杠杆
    max_leverage = IntParameter(
        1, 3, default=2, space="buy", optimize=True, load=True
    )
    
    # ==================== 技术面过滤参数 ====================
    
    # RSI参数
    rsi_period = IntParameter(10, 20, default=14, space="buy", optimize=True, load=True)
    rsi_overbought = IntParameter(65, 75, default=70, space="buy", optimize=True, load=True)
    rsi_oversold = IntParameter(25, 35, default=30, space="buy", optimize=True, load=True)
    
    # ADX趋势强度限制（避免强趋势中逆势交易）
    adx_max_threshold = IntParameter(30, 50, default=40, space="buy", optimize=True, load=True)
    
    # ATR波动率限制
    atr_max_threshold = DecimalParameter(
        0.03, 0.08, default=0.05, space="buy", optimize=True, load=True
    )  # 最大允许5%日波动率
    
    # ==================== 成本控制参数 ====================
    
    # 最大交易成本（手续费+滑点）
    max_trading_cost = DecimalParameter(
        0.0001, 0.0003, default=0.0002, space="buy", optimize=True, load=True
    )  # 默认0.02%
    
    # 最小预期收益倍数（预期费率收益 / 交易成本）
    min_profit_ratio = DecimalParameter(
        1.5, 3.0, default=2.0, space="buy", optimize=True, load=True
    )  # 收益必须>成本2倍
    
    # ==================== 风控参数 ====================
    
    # 硬止损比例
    hard_stoploss = DecimalParameter(
        -0.025, -0.01, default=-0.015, space="sell", optimize=True, load=True
    )  # 默认-1.5%
    
    # 费率反转阈值（费率回到此范围时平仓）
    funding_reversal_threshold = DecimalParameter(
        -0.0001, 0.0001, default=0.00005, space="sell", optimize=True, load=True
    )  # 默认±0.005%
    
    # 最大持仓时间（小时）
    max_holding_hours = IntParameter(
        24, 72, default=48, space="sell", optimize=True, load=True
    )  # 默认48小时（6个费率周期）
    
    # 盈亏比止损倍数
    loss_profit_ratio_max = DecimalParameter(
        2.0, 4.0, default=3.0, space="sell", optimize=True, load=True
    )  # 亏损超过预期收益3倍时止损
    
    # ==================== 出场参数 ====================
    
    # 目标累计费率收益
    target_funding_profit = DecimalParameter(
        0.002, 0.005, default=0.003, space="sell", optimize=True, load=True
    )  # 默认0.3%累计收益
    
    # 追踪止盈启动阈值
    trailing_profit_start = DecimalParameter(
        0.003, 0.008, default=0.005, space="sell", optimize=True, load=True
    )  # 盈利>0.5%启动追踪
    
    # 追踪止盈距离
    trailing_profit_distance = DecimalParameter(
        0.001, 0.003, default=0.002, space="sell", optimize=True, load=True
    )  # 追踪距离0.2%
    
    # ==================== 启动K线数量 ====================
    
    startup_candle_count: int = 200  # 需要足够历史数据计算指标
    
    def __init__(self, config: dict):
        super().__init__(config)
        # 初始化费率数据缓冲区
        self.funding_rate_history = deque(maxlen=24)  # 存储最近24个8小时周期
        self.trade_info = {}  # 存储每个交易的详细信息
        
    # ==================== 辅助方法 ====================
    
    def get_funding_rate(self, pair: str) -> float:
        """
        获取当前资金费率
        
        通过ccxt库从交易所API获取实时资金费率
        """
        try:
            funding_rate_data = self.dp.get_pair_dataframe(pair, self.timeframe)
            if funding_rate_data is None or len(funding_rate_data) == 0:
                return 0.0
            
            # 尝试通过exchange API获取资金费率
            try:
                funding_info = self.dp._exchange.fetch_funding_rate(pair)
                if funding_info and 'fundingRate' in funding_info:
                    return float(funding_info['fundingRate'])
            except Exception as e:
                self.log(f"无法获取资金费率 {pair}: {e}")
                
            return 0.0
        except Exception as e:
            self.log(f"获取资金费率异常 {pair}: {e}")
            return 0.0
    
    def get_funding_rate_history(self, pair: str, limit: int = 24) -> list:
        """
        获取历史资金费率数据
        
        用于计算费率移动平均和持续性
        """
        try:
            history = self.dp._exchange.fetch_funding_rate_history(
                pair,
                limit=limit
            )
            if history:
                return [float(h['fundingRate']) for h in history]
            return []
        except Exception as e:
            self.log(f"获取历史资金费率异常 {pair}: {e}")
            return []
    
    def calculate_funding_persistence(self, funding_history: list, direction: str) -> int:
        """
        计算费率持续性（连续同向周期数）
        
        Args:
            funding_history: 历史费率列表
            direction: 'positive' 或 'negative'
            
        Returns:
            连续同向周期数
        """
        if not funding_history or len(funding_history) == 0:
            return 0
        
        persistence = 0
        for rate in reversed(funding_history):
            if direction == 'positive' and rate > 0:
                persistence += 1
            elif direction == 'negative' and rate < 0:
                persistence += 1
            else:
                break
                
        return persistence
    
    def calculate_expected_funding_return(
        self, 
        current_rate: float, 
        periods: int = 3
    ) -> float:
        """
        计算预期资金费率收益
        
        Args:
            current_rate: 当前费率
            periods: 预期持有周期数（默认3个8小时）
            
        Returns:
            预期累计收益率
        """
        return abs(current_rate) * periods
    
    def calculate_position_size_multiplier(
        self,
        funding_rate: float,
        volatility: float
    ) -> float:
        """
        根据费率极端度和波动率计算仓位乘数
        
        反直觉设计：费率越极端，仓位越小（降低爆仓风险）
        
        Args:
            funding_rate: 当前资金费率
            volatility: 当前波动率（ATR%）
            
        Returns:
            仓位乘数 (0.0-1.0)
        """
        # 费率极端度调整
        abs_rate = abs(funding_rate)
        
        if abs_rate > 0.001:  # >0.1%，极端费率
            rate_multiplier = self.extreme_funding_factor.value
        elif abs_rate > 0.0005:  # 0.05%-0.1%，高费率
            rate_multiplier = 0.85
        elif abs_rate > 0.0003:  # 0.03%-0.05%，中等费率
            rate_multiplier = 1.0
        else:  # <0.03%，低费率
            rate_multiplier = 0.5  # 大幅降低
        
        # 波动率调整
        if volatility > 0.05:  # 高波动
            vol_multiplier = 0.5
        elif volatility > 0.03:  # 中等波动
            vol_multiplier = 0.8
        else:  # 低波动
            vol_multiplier = 1.2
        
        # 综合乘数
        return rate_multiplier * vol_multiplier * self.volatility_position_factor.value
    
    def log(self, message: str):
        """日志输出"""
        print(f"[{datetime.now()}] {message}")
    
    # ==================== 指标计算 ====================
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        计算所有技术指标和资金费率相关指标
        """
        pair = metadata['pair']
        
        # ========== 资金费率指标 ==========
        
        # 获取当前资金费率
        current_funding = self.get_funding_rate(pair)
        dataframe['funding_rate'] = current_funding
        
        # 获取历史费率并计算MA
        funding_history = self.get_funding_rate_history(pair, limit=24)
        if funding_history and len(funding_history) > 0:
            dataframe['funding_rate_ma'] = np.mean(funding_history)
            dataframe['funding_rate_std'] = np.std(funding_history)
            
            # 计算费率持续性
            dataframe['funding_positive_persistence'] = self.calculate_funding_persistence(
                funding_history, 'positive'
            )
            dataframe['funding_negative_persistence'] = self.calculate_funding_persistence(
                funding_history, 'negative'
            )
        else:
            dataframe['funding_rate_ma'] = 0.0
            dataframe['funding_rate_std'] = 0.0
            dataframe['funding_positive_persistence'] = 0
            dataframe['funding_negative_persistence'] = 0
        
        # 预期收益计算
        dataframe['expected_funding_return'] = dataframe['funding_rate'].apply(
            lambda x: self.calculate_expected_funding_return(x, periods=3)
        )
        
        # ========== 技术指标 ==========
        
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.rsi_period.value)
        
        # ADX（趋势强度）
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['plus_di'] = ta.PLUS_DI(dataframe, timeperiod=14)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe, timeperiod=14)
        
        # ATR（波动率）
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['atr_percent'] = dataframe['atr'] / dataframe['close']
        
        # EMA（趋势方向）
        dataframe['ema_12'] = ta.EMA(dataframe, timeperiod=12)
        dataframe['ema_26'] = ta.EMA(dataframe, timeperiod=26)
        
        # 布林带（价格位置）
        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=20, stds=2)
        dataframe['bb_upper'] = bollinger['upper']
        dataframe['bb_middle'] = bollinger['mid']
        dataframe['bb_lower'] = bollinger['lower']
        dataframe['bb_percent'] = (
            (dataframe['close'] - dataframe['bb_lower']) /
            (dataframe['bb_upper'] - dataframe['bb_lower'])
        )
        
        # 成交量指标
        dataframe['volume_ma'] = dataframe['volume'].rolling(20).mean()
        
        # ========== 综合信号强度 ==========
        
        # 费率信号强度（归一化到0-1）
        dataframe['funding_signal_strength'] = np.abs(dataframe['funding_rate']) / 0.001
        dataframe['funding_signal_strength'] = np.clip(dataframe['funding_signal_strength'], 0, 1)
        
        return dataframe
    
    # ==================== 入场信号 ====================
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        生成入场信号
        
        做空条件（收取正费率）：
        1. 资金费率 > 阈值
        2. 费率持续性达标
        3. 技术面不在极端超卖区
        4. 波动率在合理范围
        5. 预期收益 > 成本 × 倍数
        
        做多条件（收取负费率）：
        1. 资金费率 < 负阈值
        2. 费率持续性达标
        3. 技术面不在极端超买区  
        4. 波动率在合理范围
        5. 预期收益 > 成本 × 倍数
        """
        
        # ========== 做空入场（收取正费率）==========
        dataframe.loc[
            (
                # 核心费率条件
                (dataframe['funding_rate'] > self.funding_rate_short_threshold.value) &
                (dataframe['funding_positive_persistence'] >= self.funding_persistence_periods.value) &
                (dataframe['funding_rate'] > dataframe['funding_rate_ma']) &  # 费率上升趋势
                
                # 技术面过滤
                (dataframe['rsi'] > self.rsi_oversold.value) &  # 避免超卖区
                (dataframe['rsi'] < self.rsi_overbought.value) &  # 不在极端超买
                (dataframe['adx'] < self.adx_max_threshold.value) &  # 避免强趋势
                
                # 波动率控制
                (dataframe['atr_percent'] < self.atr_max_threshold.value) &  # 波动率不过高
                
                # 成本-收益检查
                (dataframe['expected_funding_return'] > self.max_trading_cost.value * self.min_profit_ratio.value) &
                
                # 基本市场条件
                (dataframe['volume'] > 0) &
                (dataframe['volume'] > dataframe['volume_ma'] * 0.5)  # 成交量足够
            ),
            'enter_short'
        ] = 1
        
        # ========== 做多入场（收取负费率）==========
        dataframe.loc[
            (
                # 核心费率条件
                (dataframe['funding_rate'] < self.funding_rate_long_threshold.value) &
                (dataframe['funding_negative_persistence'] >= self.funding_persistence_periods.value) &
                (dataframe['funding_rate'] < dataframe['funding_rate_ma']) &  # 费率下降趋势
                
                # 技术面过滤
                (dataframe['rsi'] < self.rsi_overbought.value) &  # 避免超买区
                (dataframe['rsi'] > self.rsi_oversold.value) &  # 不在极端超卖
                (dataframe['adx'] < self.adx_max_threshold.value) &  # 避免强趋势
                
                # 波动率控制
                (dataframe['atr_percent'] < self.atr_max_threshold.value) &
                
                # 成本-收益检查
                (dataframe['expected_funding_return'] > self.max_trading_cost.value * self.min_profit_ratio.value) &
                
                # 基本市场条件
                (dataframe['volume'] > 0) &
                (dataframe['volume'] > dataframe['volume_ma'] * 0.5)
            ),
            'enter_long'
        ] = 1
        
        return dataframe
    
    # ==================== 出场信号 ====================
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        生成出场信号
        
        出场条件：
        1. 费率回归中性区间
        2. 技术面反转信号
        3. 达到目标收益
        """
        
        # ========== 多头出场 ==========
        dataframe.loc[
            (
                # 费率反转（负费率回到正值或接近0）
                (
                    (dataframe['funding_rate'] > self.funding_reversal_threshold.value) |
                    (dataframe['funding_rate'] > dataframe['funding_rate_ma'] * 0.5)
                ) |
                
                # 技术面反转
                (
                    (dataframe['rsi'] > self.rsi_overbought.value) &
                    (dataframe['adx'] > 30) &
                    (dataframe['minus_di'] > dataframe['plus_di'])
                )
            ),
            'exit_long'
        ] = 1
        
        # ========== 空头出场 ==========
        dataframe.loc[
            (
                # 费率反转（正费率回到负值或接近0）
                (
                    (dataframe['funding_rate'] < self.funding_reversal_threshold.value) |
                    (dataframe['funding_rate'] < dataframe['funding_rate_ma'] * 0.5)
                ) |
                
                # 技术面反转
                (
                    (dataframe['rsi'] < self.rsi_oversold.value) &
                    (dataframe['adx'] > 30) &
                    (dataframe['plus_di'] > dataframe['minus_di'])
                )
            ),
            'exit_short'
        ] = 1
        
        return dataframe
    
    # ==================== 自定义仓位管理 ====================
    
    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: Optional[float],
        max_stake: float,
        leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs
    ) -> float:
        """
        动态仓位管理
        
        根据费率极端度和波动率动态调整仓位大小
        """
        # 获取当前数据
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or len(dataframe) == 0:
            return proposed_stake
        
        latest = dataframe.iloc[-1]
        
        # 获取资金费率和波动率
        funding_rate = latest['funding_rate']
        volatility = latest['atr_percent']
        
        # 计算仓位乘数
        position_multiplier = self.calculate_position_size_multiplier(
            funding_rate,
            volatility
        )
        
        # 基础仓位
        base_stake = proposed_stake * self.base_position_size.value
        
        # 应用乘数
        final_stake = base_stake * position_multiplier
        
        # 确保在允许范围内
        if min_stake:
            final_stake = max(final_stake, min_stake)
        final_stake = min(final_stake, max_stake)
        
        return final_stake
    
    # ==================== 自定义止损 ====================
    
    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs
    ) -> float:
        """
        多层动态止损机制
        
        1. 硬止损：-1.5%
        2. 费率反转止损
        3. 时间止损
        4. 盈亏比止损
        5. 追踪止盈
        """
        # 初始化交易信息
        trade_id = f"{pair}_{trade.open_date}"
        if trade_id not in self.trade_info:
            # 获取开仓时的费率数据
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe is not None and len(dataframe) > 0:
                entry_data = dataframe.iloc[-1]
                expected_return = entry_data.get('expected_funding_return', 0.003)
            else:
                expected_return = 0.003  # 默认0.3%
            
            self.trade_info[trade_id] = {
                'max_profit': current_profit,
                'entry_time': current_time,
                'expected_return': expected_return,
                'trailing_activated': False
            }
        
        trade_data = self.trade_info[trade_id]
        
        # 更新最大盈利
        if current_profit > trade_data['max_profit']:
            trade_data['max_profit'] = current_profit
        
        # 1. 硬止损
        if current_profit <= self.hard_stoploss.value:
            return self.hard_stoploss.value
        
        # 2. 盈亏比止损
        expected_return = trade_data['expected_return']
        if current_profit < -expected_return * self.loss_profit_ratio_max.value:
            return current_profit - 0.002  # 立即止损，留0.2%缓冲
        
        # 3. 时间止损
        holding_hours = (current_time - trade_data['entry_time']).total_seconds() / 3600
        if holding_hours > self.max_holding_hours.value:
            # 超过最大持仓时间，根据盈利情况决定
            if current_profit < 0:
                return current_profit - 0.001  # 亏损时立即止损
            elif current_profit < expected_return * 0.5:
                return current_profit - 0.001  # 收益不足预期一半时止损
        
        # 4. 追踪止盈
        if current_profit > self.trailing_profit_start.value:
            if not trade_data['trailing_activated']:
                trade_data['trailing_activated'] = True
            
            # 追踪止盈位置
            trailing_stop = trade_data['max_profit'] - self.trailing_profit_distance.value
            return max(self.hard_stoploss.value, trailing_stop)
        
        # 默认返回硬止损
        return self.hard_stoploss.value
    
    # ==================== 自定义退出 ====================
    
    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs
    ) -> Optional[Union[str, bool]]:
        """
        自定义退出逻辑
        
        检查费率反转和目标收益
        """
        # 获取当前数据
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or len(dataframe) == 0:
            return None
        
        latest = dataframe.iloc[-1]
        current_funding = latest['funding_rate']
        
        # 费率反转检查
        trade_id = f"{pair}_{trade.open_date}"
        if trade_id in self.trade_info:
            expected_return = self.trade_info[trade_id]['expected_return']
            
            # 达到目标收益
            if current_profit >= self.target_funding_profit.value:
                return "funding_target_reached"
            
            # 费率反转（多头时费率转正，空头时费率转负）
            if trade.is_short and current_funding < -self.funding_reversal_threshold.value:
                return "funding_rate_reversal"
            elif not trade.is_short and current_funding > self.funding_reversal_threshold.value:
                return "funding_rate_reversal"
        
        return None
