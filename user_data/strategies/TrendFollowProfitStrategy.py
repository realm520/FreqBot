# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pandas import DataFrame
from typing import Optional, Union
from collections import deque

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    DecimalParameter,
    IntParameter,
    BooleanParameter,
)

import talib.abstract as ta
from technical import qtpylib


class TrendFollowProfitStrategy(IStrategy):
    """
    趋势跟踪盈利策略 - 专门在趋势市场中获利
    
    核心思想：
    1. 识别趋势市场环境
    2. 在趋势确认后进场
    3. 金字塔加仓，长期持有
    4. 追踪止盈，保护利润
    """
    
    INTERFACE_VERSION = 3
    can_short: bool = False
    
    # 趋势跟踪设置
    minimal_roi = {"0": 10}  # 不使用固定ROI
    stoploss = -0.08  # 8%止损
    trailing_stop = True
    trailing_stop_positive = 0.02  # 2%启动追踪
    trailing_stop_positive_offset = 0.03  # 3%追踪距离
    timeframe = '15m'
    use_custom_stoploss = True
    
    # 趋势识别参数
    ema_fast = IntParameter(12, 21, default=15, space="buy", optimize=True)
    ema_slow = IntParameter(26, 55, default=35, space="buy", optimize=True)
    adx_period = IntParameter(10, 21, default=14, space="buy", optimize=True)
    adx_threshold = IntParameter(25, 40, default=30, space="buy", optimize=True)
    
    # 趋势确认参数
    trend_confirmation_period = IntParameter(3, 8, default=5, space="buy", optimize=True)
    momentum_threshold = DecimalParameter(0.003, 0.015, default=0.008, space="buy", optimize=True)
    
    # 加仓参数
    enable_pyramid = BooleanParameter(default=True, space="buy", optimize=True)
    pyramid_distance = DecimalParameter(0.02, 0.05, default=0.03, space="buy", optimize=True)
    max_pyramid_levels = IntParameter(2, 4, default=3, space="buy", optimize=True)
    
    # 利润目标
    profit_target_1 = DecimalParameter(0.05, 0.10, default=0.08, space="sell", optimize=True)
    profit_target_2 = DecimalParameter(0.15, 0.25, default=0.20, space="sell", optimize=True)
    profit_target_3 = DecimalParameter(0.30, 0.50, default=0.40, space="sell", optimize=True)
    
    startup_candle_count: int = 200
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.pyramid_levels = {}
        
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        趋势跟踪指标
        """
        # EMA系统
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=self.ema_fast.value)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=self.ema_slow.value)
        dataframe['ema_trend'] = dataframe['ema_fast'] > dataframe['ema_slow']
        
        # ADX趋势强度
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=self.adx_period.value)
        dataframe['strong_trend'] = dataframe['adx'] > self.adx_threshold.value
        
        # MACD动量确认
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macd_positive'] = dataframe['macd'] > dataframe['macdsignal']
        
        # 动量计算
        dataframe['momentum'] = dataframe['close'].pct_change(self.trend_confirmation_period.value)
        dataframe['strong_momentum'] = dataframe['momentum'] > self.momentum_threshold.value
        
        # 成交量确认
        dataframe['volume_sma'] = dataframe['volume'].rolling(20).mean()
        dataframe['volume_surge'] = dataframe['volume'] > dataframe['volume_sma'] * 1.2
        
        # 综合趋势信号
        dataframe['trend_score'] = (
            dataframe['ema_trend'].astype(int) +
            dataframe['strong_trend'].astype(int) +
            dataframe['macd_positive'].astype(int) +
            dataframe['strong_momentum'].astype(int) +
            dataframe['volume_surge'].astype(int)
        )
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        趋势跟踪入场
        """
        dataframe['enter_long'] = (
            # 强趋势信号：至少4个确认
            (dataframe['trend_score'] >= 4) &
            
            # EMA排列确认
            (dataframe['ema_fast'] > dataframe['ema_slow']) &
            
            # 价格在EMA之上
            (dataframe['close'] > dataframe['ema_fast']) &
            
            # ADX确认强趋势
            (dataframe['strong_trend'] == True) &
            
            # 动量确认
            (dataframe['strong_momentum'] == True)
        )
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        趋势跟踪出场
        """
        dataframe['exit_long'] = (
            # 趋势转弱
            (dataframe['ema_fast'] < dataframe['ema_slow']) |
            
            # ADX下降
            (dataframe['adx'] < self.adx_threshold.value * 0.8) |
            
            # MACD转负
            (dataframe['macd'] < dataframe['macdsignal']) |
            
            # 动量转负
            (dataframe['momentum'] < -self.momentum_threshold.value)
        )
        
        return dataframe
    
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        动态追踪止损
        """
        # 分阶段追踪止损
        if current_profit >= self.profit_target_3.value:
            return current_profit - 0.15  # 保护85%利润
        elif current_profit >= self.profit_target_2.value:
            return current_profit - 0.08  # 保护92%利润
        elif current_profit >= self.profit_target_1.value:
            return current_profit - 0.04  # 保护96%利润
        elif current_profit >= 0.03:
            return current_profit - 0.02  # 保护部分利润
        else:
            return self.stoploss
    
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                          time_in_force: str, current_time: datetime, entry_tag: Optional[str] = None, side: str = 'long', **kwargs) -> bool:
        """
        趋势确认入场
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return False
        
        latest = dataframe.iloc[-1]
        
        # 确认趋势信号强度
        if latest['trend_score'] < 3:
            return False
        
        return True