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


class MeanReversionProfitStrategy(IStrategy):
    """
    均值回归盈利策略 - 专门在震荡市场中获利
    
    核心思想：
    1. 识别震荡市场环境
    2. 在价格偏离均值时进场
    3. 快速止盈，严格止损
    4. 高频交易，积少成多
    """
    
    INTERFACE_VERSION = 3
    can_short: bool = False
    
    # 激进盈利设置
    minimal_roi = {
        "0": 0.03,   # 3%止盈
        "30": 0.02,  # 30分钟后2%止盈  
        "60": 0.01,  # 1小时后1%止盈
        "120": 0    # 2小时后平仓
    }
    stoploss = -0.025  # 2.5%止损
    trailing_stop = False
    timeframe = '15m'
    use_custom_stoploss = False  # 使用固定止损
    
    # 策略参数
    bb_period = IntParameter(15, 25, default=20, space="buy", optimize=True)
    bb_std = DecimalParameter(1.8, 2.5, default=2.0, space="buy", optimize=True)
    rsi_period = IntParameter(10, 21, default=14, space="buy", optimize=True)
    rsi_oversold = IntParameter(25, 35, default=30, space="buy", optimize=True)
    rsi_overbought = IntParameter(65, 80, default=70, space="buy", optimize=True)
    
    # 均值回归参数
    bb_entry_threshold = DecimalParameter(0.05, 0.25, default=0.1, space="buy", optimize=True)  # 进入布林带下轨的程度
    volume_threshold = DecimalParameter(0.8, 1.5, default=1.0, space="buy", optimize=True)  # 成交量阈值
    
    # 震荡市场识别参数
    volatility_period = IntParameter(20, 50, default=30, space="buy", optimize=True)
    trend_strength_threshold = DecimalParameter(15, 30, default=20, space="buy", optimize=True)  # ADX阈值
    
    # 快速出场参数
    quick_profit_target = DecimalParameter(0.008, 0.025, default=0.015, space="sell", optimize=True)  # 1.5%快速止盈
    volume_exit_threshold = DecimalParameter(0.3, 0.8, default=0.5, space="sell", optimize=True)  # 成交量萎缩出场
    
    startup_candle_count: int = 200
    
    def __init__(self, config: dict):
        super().__init__(config)
        
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        技术指标计算
        """
        # 布林带 - 均值回归核心指标
        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=self.bb_period.value, stds=self.bb_std.value)
        dataframe['bb_lower'] = bollinger['lower']
        dataframe['bb_middle'] = bollinger['mid']
        dataframe['bb_upper'] = bollinger['upper']
        dataframe['bb_percent'] = (dataframe['close'] - dataframe['bb_lower']) / (dataframe['bb_upper'] - dataframe['bb_lower'])
        dataframe['bb_width'] = (dataframe['bb_upper'] - dataframe['bb_lower']) / dataframe['bb_middle']
        
        # RSI - 超买超卖确认
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.rsi_period.value)
        
        # ADX - 趋势强度 (震荡市场识别)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['is_ranging'] = dataframe['adx'] < self.trend_strength_threshold.value
        
        # EMA - 趋势过滤
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        
        # 成交量指标
        dataframe['volume_sma'] = dataframe['volume'].rolling(20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_sma']
        
        # 波动率计算
        dataframe['price_change'] = dataframe['close'].pct_change()
        dataframe['volatility'] = dataframe['price_change'].rolling(self.volatility_period.value).std()
        
        # 均值回归信号强度
        dataframe['mean_reversion_strength'] = (
            (dataframe['bb_percent'] < self.bb_entry_threshold.value).astype(int) +
            (dataframe['rsi'] < self.rsi_oversold.value).astype(int) +
            (dataframe['close'] < dataframe['ema_20'] * 0.99).astype(int) +
            (dataframe['volume_ratio'] > self.volume_threshold.value).astype(int) +
            (dataframe['is_ranging']).astype(int)
        )
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        入场条件 - 均值回归信号
        """
        # 强均值回归信号：多个条件同时满足
        dataframe['enter_long'] = (
            # 核心条件：价格远离布林带下轨
            (dataframe['bb_percent'] < self.bb_entry_threshold.value) &
            
            # RSI超卖确认
            (dataframe['rsi'] < self.rsi_oversold.value) &
            
            # 震荡市场确认 (非强趋势)
            (dataframe['is_ranging'] == True) &
            
            # 成交量确认
            (dataframe['volume_ratio'] > self.volume_threshold.value) &
            
            # 价格低于短期均线 (进一步确认超卖)
            (dataframe['close'] < dataframe['ema_20']) &
            
            # 布林带宽度适中 (避免极端波动)
            (dataframe['bb_width'] > 0.02) & (dataframe['bb_width'] < 0.08)
        )
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        出场条件 - 快速止盈
        """
        dataframe['exit_long'] = (
            # 快速反转信号
            (dataframe['rsi'] > self.rsi_overbought.value) |
            
            # 回到布林带中轨以上
            (dataframe['close'] > dataframe['bb_middle'] * 1.01) |
            
            # 成交量萎缩
            (dataframe['volume_ratio'] < self.volume_exit_threshold.value) |
            
            # 趋势转强 (不再是震荡市场)
            (dataframe['adx'] > self.trend_strength_threshold.value * 1.5)
        )
        
        return dataframe


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