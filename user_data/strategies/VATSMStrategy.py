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


class VATSMStrategy(IStrategy):
    """
    波动率调整时间序列动量(VATSM)策略
    
    核心原理：
    1. 动态调整回溯期：高波动率时缩短回溯期，低波动率时延长回溯期
    2. 波动率目标定位：根据预测波动率调整仓位大小，维持恒定风险敞口
    3. 自适应动量计算：使用动态回溯期计算动量信号
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
    lb_min = IntParameter(240, 960, default=480, space="buy", optimize=True)  # 最小回溯期 (5天*96个15m)
    lb_max = IntParameter(1440, 4320, default=2880, space="buy", optimize=True)  # 最大回溯期 (30天*96个15m)
    vol_short_win = IntParameter(480, 1440, default=960, space="buy", optimize=True)  # 短期波动率窗口 (10天*96个15m)
    vol_long_win = IntParameter(1920, 3840, default=2880, space="buy", optimize=True)  # 长期波动率窗口 (30天*96个15m)
    ratio_cap = DecimalParameter(0.5, 0.95, default=0.9, space="buy", optimize=True)  # 波动率比率上限
    
    # 波动率目标参数
    vol_target = DecimalParameter(0.10, 0.30, default=0.20, space="buy", optimize=True)  # 目标波动率
    ewma_lambda = DecimalParameter(0.90, 0.98, default=0.94, space="buy", optimize=True)  # EWMA衰减因子
    min_forecast_vol = DecimalParameter(0.02, 0.08, default=0.05, space="buy", optimize=True)  # 最小预测波动率
    max_leverage = DecimalParameter(0.5, 2.0, default=1.0, space="buy", optimize=True)  # 最大杠杆
    
    # 胜率优化参数 (Phase 2A 精细化调优)
    min_momentum_threshold = DecimalParameter(0.001, 0.008, default=0.003, space="buy", optimize=True)  # 最小动量阈值 (0.3%)
    ema_fast = IntParameter(10, 25, default=15, space="buy", optimize=True)  # 快速EMA周期 (更敏感)
    ema_slow = IntParameter(25, 60, default=35, space="buy", optimize=True)  # 慢速EMA周期 (降低滞后性)
    adx_threshold = IntParameter(15, 30, default=20, space="buy", optimize=True)  # ADX趋势强度阈值 (放宽)
    
    # Strategy A: 三层追踪止损优化参数
    tier1_profit = DecimalParameter(0.005, 0.015, default=0.008, space="sell", optimize=True)  # 第一层启动盈利 (0.8%)
    tier1_distance = DecimalParameter(0.003, 0.010, default=0.005, space="sell", optimize=True)  # 第一层追踪距离 (0.5%)
    tier2_profit = DecimalParameter(0.015, 0.030, default=0.020, space="sell", optimize=True)  # 第二层启动盈利 (2%)
    tier2_distance = DecimalParameter(0.008, 0.025, default=0.015, space="sell", optimize=True)  # 第二层追踪距离 (1.5%)
    tier3_profit = DecimalParameter(0.035, 0.070, default=0.050, space="sell", optimize=True)  # 第三层启动盈利 (5%)
    tier3_distance = DecimalParameter(0.020, 0.050, default=0.030, space="sell", optimize=True)  # 第三层追踪距离 (3%)
    
    # 时间止损参数
    max_hold_hours = IntParameter(12, 72, default=48, space="sell", optimize=True)  # 最大持仓时间（小时）
    time_exit_profit_threshold = DecimalParameter(-0.10, 0.05, default=0.0, space="sell", optimize=True)  # 时间止损盈利阈值
    
    startup_candle_count: int = 3000  # 约31天*96个15分钟周期，适合15分钟回测
    
    def __init__(self, config: dict):
        super().__init__(config)
        # 初始化波动率缓冲区
        self.short_vol_buffer = deque(maxlen=self.vol_short_win.value)
        self.long_vol_buffer = deque(maxlen=self.vol_long_win.value)
        self.ewma_var = None
        self.last_close = None
        # 初始化追踪止盈变量
        self.custom_info = {}
        
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        添加VATSM策略所需的指标
        """
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
        
        # 动量计算 - 使用动态回溯期
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
                    
                    # 优化后的信号生成逻辑
                    # 要求: 1)动量超过阈值 2)趋势方向一致 3)ADX显示强趋势
                    if (abs(momentum) > self.min_momentum_threshold.value and 
                        dataframe['adx_strong'].iloc[i]):
                        
                        if (momentum > 0 and 
                            dataframe['trend_up'].iloc[i]):  # 多头信号
                            dataframe.loc[dataframe.index[i], 'vatsm_signal'] = 1
                        elif (momentum < 0 and 
                              not dataframe['trend_up'].iloc[i]):  # 空头信号(现货不做空)
                            dataframe.loc[dataframe.index[i], 'vatsm_signal'] = -1
        
        # 计算EWMA波动率用于仓位管理
        dataframe['ewma_vol'] = 0.0
        ewma_var = None
        
        for i in range(1, len(dataframe)):
            log_ret = dataframe['log_returns'].iloc[i]
            if ewma_var is None:
                ewma_var = log_ret ** 2
            else:
                ewma_var = self.ewma_lambda.value * ewma_var + (1.0 - self.ewma_lambda.value) * (log_ret ** 2)
            
            # 年化波动率 (假设365天交易日)
            if self.timeframe == '1d':
                periods_per_year = 365
            elif self.timeframe == '1h':
                periods_per_year = 365 * 24
            elif self.timeframe == '4h':
                periods_per_year = 365 * 6
            elif self.timeframe == '15m':
                periods_per_year = 365 * 24 * 4  # 15分钟一天有96个周期
            else:
                periods_per_year = 365  # 默认日线
                
            annual_vol = math.sqrt(max(ewma_var, 0.0)) * math.sqrt(periods_per_year)
            dataframe.loc[dataframe.index[i], 'ewma_vol'] = annual_vol
        
        # 计算风险敞口
        dataframe['vol_forecast'] = np.maximum(dataframe['ewma_vol'], self.min_forecast_vol.value)
        dataframe['raw_exposure'] = self.vol_target.value / dataframe['vol_forecast']
        dataframe['target_exposure'] = np.minimum(dataframe['raw_exposure'], self.max_leverage.value)
        
        # 根据信号强度调整风险敞口
        dataframe['signal_strength'] = np.where(
            np.abs(dataframe['momentum']) > 2 * self.min_momentum_threshold.value, 1.0, 0.7
        )
        dataframe['adjusted_exposure'] = dataframe['target_exposure'] * dataframe['signal_strength']
        dataframe['desired_exposure'] = dataframe['vatsm_signal'] * dataframe['adjusted_exposure']
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        基于优化后VATSM信号生成入场信号 - Phase 2A信号质量提升
        """
        # 优化后的多头入场条件 - 更严格的信号质量控制
        dataframe.loc[
            (
                (dataframe['vatsm_signal'] > 0) &  # VATSM信号为正
                (dataframe['trend_up'] == True) &  # EMA趋势向上
                (dataframe['adx_strong'] == True) &  # 趋势强度足够
                (dataframe['momentum'] > self.min_momentum_threshold.value * 1.5) &  # 更高动量要求
                (dataframe['adjusted_exposure'] > 0.2) &  # 提高最小风险敞口要求
                (dataframe['volume'] > 0) &  # 成交量检查
                (dataframe['close'] > dataframe['ema_fast']) &  # 价格在快速EMA之上
                (dataframe['close'] > dataframe['ema_fast'].shift(1))  # 价格相对快速EMA向上
            ),
            'enter_long'
        ] = 1

        # 空头入场（现货不做空）
        # dataframe.loc[
        #     (
        #         (dataframe['vatsm_signal'] < 0) &
        #         (dataframe['trend_up'] == False) &
        #         (dataframe['adx_strong'] == True) &
        #         (dataframe['momentum'] < -self.min_momentum_threshold.value) &
        #         (dataframe['adjusted_exposure'] > 0.1) &
        #         (dataframe['volume'] > 0)
        #     ),
        #     'enter_short'
        # ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Strategy A: 纯追踪止损模式 - 移除所有主动出场信号
        
        不设置任何出场信号，让交易完全依靠：
        1. 三层追踪止损机制 (custom_stoploss)
        2. 时间止损机制 (可选)
        3. 最大持仓时间限制
        
        这样可以避免过早出场，让盈利头寸有更多时间增长。
        """
        # Strategy A: 完全移除所有出场信号逻辑
        # 让交易只通过追踪止损自动结束
        # 
        # 注意：populate_exit_trend必须返回dataframe，但不设置任何exit条件
        # FreqTrade会依靠custom_stoploss中的追踪止损机制来管理出场
        
        return dataframe
    
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                          proposed_stake: float, min_stake: Optional[float], max_stake: float,
                          leverage: float, entry_tag: Optional[str], side: str,
                          **kwargs) -> float:
        """
        基于波动率目标调整仓位大小
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or len(dataframe) == 0:
            return proposed_stake
            
        latest = dataframe.iloc[-1]
        desired_exposure = latest['desired_exposure']
        
        if abs(desired_exposure) < 0.01:  # 避免过小的仓位
            return min_stake or 0
            
        # 计算基于风险敞口的仓位大小
        account_balance = self.wallets.get_total_stake_amount()
        target_stake = abs(desired_exposure) * account_balance
        
        # 确保在允许范围内
        if min_stake:
            target_stake = max(target_stake, min_stake)
        target_stake = min(target_stake, max_stake)
        
        return target_stake
    
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Strategy A: 优化的三层追踪止损机制
        
        使用可优化参数实现精细的追踪止损控制：
        - 第一层：小盈利保护，较小追踪距离，早期启动
        - 第二层：中盈利保护，适中追踪距离，平衡保护
        - 第三层：大盈利保护，较大追踪距离，最大化收益
        """
        # 获取交易信息
        trade_id = f"{pair}_{trade.open_date}"
        
        # 初始化追踪信息
        if trade_id not in self.custom_info:
            self.custom_info[trade_id] = {
                'max_profit': current_profit,
                'trailing_activated': False,
                'tier_activated': 0,  # 记录已启动的分级
                'entry_time': current_time  # 记录入场时间
            }
        
        trade_info = self.custom_info[trade_id]
        
        # 更新最大盈利
        if current_profit > trade_info['max_profit']:
            trade_info['max_profit'] = current_profit
        
        # 优化的分级追踪止损机制
        trailing_distance = 0.0
        
        # 第三层：大盈利保护 - 使用优化参数
        if current_profit > self.tier3_profit.value:
            trailing_distance = self.tier3_distance.value
            if trade_info['tier_activated'] < 3:
                trade_info['tier_activated'] = 3
                trade_info['trailing_activated'] = True
        
        # 第二层：中等盈利保护 - 使用优化参数  
        elif current_profit > self.tier2_profit.value:
            trailing_distance = self.tier2_distance.value
            if trade_info['tier_activated'] < 2:
                trade_info['tier_activated'] = 2
                trade_info['trailing_activated'] = True
        
        # 第一层：小盈利保护 - 使用优化参数
        elif current_profit > self.tier1_profit.value:
            trailing_distance = self.tier1_distance.value
            if trade_info['tier_activated'] < 1:
                trade_info['tier_activated'] = 1
                trade_info['trailing_activated'] = True
        
        # 时间止损检查
        if 'entry_time' in trade_info:
            hold_duration = current_time - trade_info['entry_time']
            hold_hours = hold_duration.total_seconds() / 3600
            
            # 如果持仓时间超过最大限制且盈利超过阈值，触发时间止损
            if (hold_hours > self.max_hold_hours.value and 
                current_profit > self.time_exit_profit_threshold.value):
                # 时间止损：返回当前盈利减去小幅缓冲
                return current_profit - 0.005  # 保留0.5%缓冲避免滑点损失
        
        # 如果追踪止损已启动
        if trade_info['trailing_activated'] and trailing_distance > 0:
            # 追踪止损位置 = 最大盈利 - 对应追踪距离
            trailing_stop = trade_info['max_profit'] - trailing_distance
            # 返回追踪止损值，确保不超过最大损失限制
            return max(self.stoploss, trailing_stop)
            
        # 未启动追踪止损时，使用默认止损
        return self.stoploss