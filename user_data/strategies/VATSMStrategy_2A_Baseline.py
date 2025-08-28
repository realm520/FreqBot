# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
import math
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Optional, Union
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


class VATSMStrategy_2A_Baseline(IStrategy):
    """
    VATSM策略 Phase 2A 基准版本 - 用于性能对比
    
    禁用Phase 2B新特性：
    - 多重确认机制 (min_confirmations = 1)
    - 多时间框架融合 (enable_multi_timeframe = False)
    - 分批获利了结 (enable_partial_exit = False)
    - 复杂信号过滤 (使用简化版本)
    
    保留Phase 2A核心特性：
    - 三层追踪止损
    - EMA趋势过滤
    - ADX强度验证
    - 基础动量计算
    """
    
    INTERFACE_VERSION = 3
    can_short: bool = False  # 现货交易不支持做空
    
    # 基础参数
    minimal_roi = {"0": 10}  # 不使用固定ROI，依赖策略信号
    stoploss = -0.5  # 宽松止损，主要依赖策略信号
    trailing_stop = True  # 启用追踪止损
    trailing_stop_positive = 0.005  # 盈利0.5%后启动追踪
    trailing_stop_positive_offset = 0.015  # 追踪止损距离 1.5%
    timeframe = '15m'
    use_custom_stoploss = True
    
    # 策略参数 (针对15分钟周期调整)
    lb_min = IntParameter(240, 960, default=480, space="buy", optimize=False)  # 最小回溯期
    lb_max = IntParameter(1440, 4320, default=2880, space="buy", optimize=False)  # 最大回溯期
    vol_short_win = IntParameter(480, 1440, default=960, space="buy", optimize=False)  # 短期波动率窗口
    vol_long_win = IntParameter(1920, 3840, default=2880, space="buy", optimize=False)  # 长期波动率窗口
    ratio_cap = DecimalParameter(0.5, 0.95, default=0.9, space="buy", optimize=False)  # 波动率比率上限
    
    # 波动率目标参数
    vol_target = DecimalParameter(0.10, 0.30, default=0.20, space="buy", optimize=False)  # 目标波动率
    ewma_lambda = DecimalParameter(0.90, 0.98, default=0.94, space="buy", optimize=False)  # EWMA衰减因子
    min_forecast_vol = DecimalParameter(0.02, 0.08, default=0.05, space="buy", optimize=False)  # 最小预测波动率
    max_leverage = DecimalParameter(0.5, 2.0, default=1.0, space="buy", optimize=False)  # 最大杠杆
    
    # Phase 2A 基础参数
    min_momentum_threshold = DecimalParameter(0.001, 0.008, default=0.003, space="buy", optimize=False)  # 最小动量阈值
    ema_fast = IntParameter(10, 25, default=15, space="buy", optimize=False)  # 快速EMA周期
    ema_slow = IntParameter(25, 60, default=35, space="buy", optimize=False)  # 慢速EMA周期
    adx_threshold = IntParameter(15, 30, default=20, space="buy", optimize=False)  # ADX趋势强度阈值
    
    # Phase 2A 三层追踪止损参数
    tier1_profit = DecimalParameter(0.005, 0.015, default=0.008, space="sell", optimize=False)
    tier1_distance = DecimalParameter(0.003, 0.010, default=0.005, space="sell", optimize=False)
    tier2_profit = DecimalParameter(0.015, 0.030, default=0.020, space="sell", optimize=False)
    tier2_distance = DecimalParameter(0.008, 0.025, default=0.015, space="sell", optimize=False)
    tier3_profit = DecimalParameter(0.035, 0.070, default=0.050, space="sell", optimize=False)
    tier3_distance = DecimalParameter(0.020, 0.050, default=0.030, space="sell", optimize=False)
    
    # 时间止损参数
    max_hold_hours = IntParameter(12, 72, default=48, space="sell", optimize=False)
    time_exit_profit_threshold = DecimalParameter(-0.10, 0.05, default=0.0, space="sell", optimize=False)
    
    startup_candle_count: int = 3000  # Phase 2A标准启动周期
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.short_vol_buffer = deque(maxlen=self.vol_short_win.value)
        self.long_vol_buffer = deque(maxlen=self.vol_long_win.value)
        self.ewma_var = None
        self.last_close = None
        self.custom_info = {}
        
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Phase 2A 基础指标计算"""
        # 计算对数收益率
        dataframe['log_returns'] = np.log(dataframe['close'] / dataframe['close'].shift(1))
        dataframe['log_returns'] = dataframe['log_returns'].fillna(0)
        
        # 计算滚动波动率
        dataframe['vol_short'] = dataframe['log_returns'].rolling(self.vol_short_win.value).std()
        dataframe['vol_long'] = dataframe['log_returns'].rolling(self.vol_long_win.value).std()
        
        # 计算波动率比率
        dataframe['vol_ratio'] = (dataframe['vol_short'] / dataframe['vol_long']).fillna(0)
        dataframe['vol_ratio'] = np.minimum(dataframe['vol_ratio'], self.ratio_cap.value)
        
        # 动态回溯期计算
        dataframe['lookback'] = (
            self.lb_min.value + 
            (self.lb_max.value - self.lb_min.value) * (1.0 - dataframe['vol_ratio'])
        ).round().astype(int)
        
        # 添加EMA趋势指标
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=self.ema_fast.value)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=self.ema_slow.value)
        dataframe['trend_up'] = dataframe['ema_fast'] > dataframe['ema_slow']
        
        # 添加ADX趋势强度指标
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['adx_strong'] = dataframe['adx'] > self.adx_threshold.value
        
        # 动量计算 - Phase 2A简化版本
        dataframe['momentum'] = 0.0
        dataframe['vatsm_signal'] = 0.0
        
        for i in range(len(dataframe)):
            if i < self.lb_max.value:
                continue
                
            lookback = int(dataframe['lookback'].iloc[i])
            if i >= lookback:
                current_price = dataframe['close'].iloc[i]
                past_price = dataframe['close'].iloc[i - lookback]
                if past_price > 0:
                    momentum = (current_price - past_price) / past_price
                    dataframe.loc[dataframe.index[i], 'momentum'] = momentum
                    
                    # Phase 2A 简化信号生成 (单一确认)
                    if (abs(momentum) > self.min_momentum_threshold.value and 
                        dataframe['adx_strong'].iloc[i]):
                        
                        if (momentum > 0 and 
                            dataframe['trend_up'].iloc[i]):  # 多头信号
                            dataframe.loc[dataframe.index[i], 'vatsm_signal'] = 1
        
        # 计算EWMA波动率用于仓位管理
        dataframe['ewma_vol'] = 0.0
        ewma_var = None
        
        for i in range(1, len(dataframe)):
            log_ret = dataframe['log_returns'].iloc[i]
            if ewma_var is None:
                ewma_var = log_ret ** 2
            else:
                ewma_var = self.ewma_lambda.value * ewma_var + (1.0 - self.ewma_lambda.value) * (log_ret ** 2)
            
            periods_per_year = 365 * 24 * 4  # 15分钟周期
            annual_vol = math.sqrt(max(ewma_var, 0.0)) * math.sqrt(periods_per_year)
            dataframe.loc[dataframe.index[i], 'ewma_vol'] = annual_vol
        
        # 计算风险敞口 - Phase 2A简化版本
        dataframe['vol_forecast'] = np.maximum(dataframe['ewma_vol'], self.min_forecast_vol.value)
        dataframe['raw_exposure'] = self.vol_target.value / dataframe['vol_forecast']
        dataframe['target_exposure'] = np.minimum(dataframe['raw_exposure'], self.max_leverage.value)
        dataframe['signal_strength'] = np.where(
            np.abs(dataframe['momentum']) > 2 * self.min_momentum_threshold.value, 1.0, 0.7
        )
        dataframe['adjusted_exposure'] = dataframe['target_exposure'] * dataframe['signal_strength']
        dataframe['desired_exposure'] = dataframe['vatsm_signal'] * dataframe['adjusted_exposure']
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Phase 2A 基础入场逻辑"""
        dataframe.loc[
            (
                (dataframe['vatsm_signal'] > 0) &  # VATSM信号为正
                (dataframe['trend_up'] == True) &  # EMA趋势向上
                (dataframe['adx_strong'] == True) &  # 趋势强度足够
                (dataframe['momentum'] > self.min_momentum_threshold.value * 1.5) &
                (dataframe['adjusted_exposure'] > 0.2) &
                (dataframe['volume'] > 0) &
                (dataframe['close'] > dataframe['ema_fast'])
            ),
            'enter_long'
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Phase 2A 纯追踪止损模式"""
        return dataframe
    
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                          proposed_stake: float, min_stake: Optional[float], max_stake: float,
                          leverage: float, entry_tag: Optional[str], side: str,
                          **kwargs) -> float:
        """Phase 2A 基础仓位管理"""
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or len(dataframe) == 0:
            return proposed_stake
            
        latest = dataframe.iloc[-1]
        desired_exposure = latest['desired_exposure']
        
        if abs(desired_exposure) < 0.01:
            return min_stake or 0
            
        account_balance = self.wallets.get_total_stake_amount()
        target_stake = abs(desired_exposure) * account_balance
        
        if min_stake:
            target_stake = max(target_stake, min_stake)
        target_stake = min(target_stake, max_stake)
        
        return target_stake
    
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """Phase 2A 三层追踪止损"""
        trade_id = f"{pair}_{trade.open_date}"
        
        if trade_id not in self.custom_info:
            self.custom_info[trade_id] = {
                'max_profit': current_profit,
                'trailing_activated': False,
                'tier_activated': 0,
                'entry_time': current_time
            }
        
        trade_info = self.custom_info[trade_id]
        
        if current_profit > trade_info['max_profit']:
            trade_info['max_profit'] = current_profit
        
        trailing_distance = 0.0
        
        # 三层追踪止损机制
        if current_profit > self.tier3_profit.value:
            trailing_distance = self.tier3_distance.value
            if trade_info['tier_activated'] < 3:
                trade_info['tier_activated'] = 3
                trade_info['trailing_activated'] = True
        elif current_profit > self.tier2_profit.value:
            trailing_distance = self.tier2_distance.value
            if trade_info['tier_activated'] < 2:
                trade_info['tier_activated'] = 2
                trade_info['trailing_activated'] = True
        elif current_profit > self.tier1_profit.value:
            trailing_distance = self.tier1_distance.value
            if trade_info['tier_activated'] < 1:
                trade_info['tier_activated'] = 1
                trade_info['trailing_activated'] = True
        
        # 时间止损检查
        if 'entry_time' in trade_info:
            hold_duration = current_time - trade_info['entry_time']
            hold_hours = hold_duration.total_seconds() / 3600
            
            if (hold_hours > self.max_hold_hours.value and 
                current_profit > self.time_exit_profit_threshold.value):
                return current_profit - 0.005
        
        # 追踪止损逻辑
        if trade_info['trailing_activated'] and trailing_distance > 0:
            trailing_stop = trade_info['max_profit'] - trailing_distance
            return max(self.stoploss, trailing_stop)
            
        return self.stoploss