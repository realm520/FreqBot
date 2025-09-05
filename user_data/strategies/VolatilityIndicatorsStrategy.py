# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from pandas import DataFrame
from datetime import datetime
from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,  # @informative decorator
    # Hyperopt Parameters
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    # timeframe helpers
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    # Strategy helper functions
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)
from freqtrade.persistence import Trade
from freqtrade.exchange import timeframe_to_minutes
import talib.abstract as ta
from technical.indicators import ichimoku
import freqtrade.vendor.qtpylib.indicators as qtpylib


class VolatilityIndicatorsStrategy(IStrategy):
    """
    波动率指标策略
    使用五个波动率指标捕捉交易信号：
    1. Chaikin Volatility (CHV) - 查金波动率指标
    2. Donchian Channels - 唐奇安通道
    3. Keltner Channels - 凯尔特纳通道
    4. Relative Volatility Index (RVI) - 相对波动率指数
    5. Standard Deviation - 标准差
    """
    
    # 策略参数
    INTERFACE_VERSION = 3
    can_short: bool = False  # Set to False for spot trading
    
    # 买入/卖出超参数
    buy_chv_increase = DecimalParameter(0.0, 0.5, default=0.1, space="buy", optimize=True)
    buy_rvi_threshold = IntParameter(40, 70, default=50, space="buy", optimize=True)
    sell_rvi_threshold = IntParameter(30, 60, default=50, space="sell", optimize=True)
    
    # Donchian Channel 周期
    donchian_period = IntParameter(10, 30, default=20, space="buy", optimize=True)
    
    # Keltner Channel 参数
    keltner_period = IntParameter(10, 30, default=20, space="buy", optimize=True)
    keltner_multiplier = DecimalParameter(1.0, 3.0, default=2.0, space="buy", optimize=True)
    
    # 标准差阈值
    std_threshold = DecimalParameter(0.01, 0.1, default=0.03, space="buy", optimize=True)
    
    # ROI表 - 根据持仓时间设定目标收益
    minimal_roi = {
        "0": 0.10,    # 10%
        "30": 0.05,   # 5% after 30 minutes
        "60": 0.03,   # 3% after 1 hour
        "120": 0.01,  # 1% after 2 hours
    }
    
    # 止损
    stoploss = -0.05  # -5%
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True
    
    # 时间框架
    timeframe = '15m'
    
    # 订单类型
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': True
    }
    
    # 启动蜡烛数
    startup_candle_count: int = 100
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        计算五个波动率指标
        """
        
        # 1. Chaikin Volatility (CHV)
        # CHV = ((EMA(High-Low, 10) - EMA(High-Low, 10)[10 bars ago]) / EMA(High-Low, 10)[10 bars ago]) * 100
        hl_diff = dataframe['high'] - dataframe['low']
        ema_hl = pd.Series(ta.EMA(hl_diff, timeperiod=10), index=dataframe.index)
        dataframe['chv'] = ((ema_hl - ema_hl.shift(10)) / ema_hl.shift(10)) * 100
        
        # 2. Donchian Channels
        period = self.donchian_period.value
        dataframe['donchian_upper'] = dataframe['high'].rolling(window=period).max()
        dataframe['donchian_lower'] = dataframe['low'].rolling(window=period).min()
        dataframe['donchian_middle'] = (dataframe['donchian_upper'] + dataframe['donchian_lower']) / 2
        
        # 3. Keltner Channels
        kc_period = self.keltner_period.value
        multiplier = self.keltner_multiplier.value
        
        # 计算典型价格和EMA
        typical_price = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
        dataframe['keltner_middle'] = pd.Series(ta.EMA(typical_price, timeperiod=kc_period), index=dataframe.index)
        
        # 计算ATR
        dataframe['atr'] = ta.ATR(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=kc_period)
        
        # 计算上下轨
        dataframe['keltner_upper'] = dataframe['keltner_middle'] + (multiplier * dataframe['atr'])
        dataframe['keltner_lower'] = dataframe['keltner_middle'] - (multiplier * dataframe['atr'])
        
        # 4. Relative Volatility Index (RVI)
        # 计算上涨和下跌的标准差
        period_rvi = 14
        
        # 价格变化
        price_change = dataframe['close'].diff()
        
        # 分离上涨和下跌
        up_moves = price_change.copy()
        up_moves[up_moves < 0] = 0
        down_moves = -price_change.copy()
        down_moves[down_moves < 0] = 0
        
        # 计算上涨和下跌的标准差
        up_std = up_moves.rolling(window=period_rvi).std()
        down_std = down_moves.rolling(window=period_rvi).std()
        
        # 计算RVI
        rs = up_std / down_std
        dataframe['rvi'] = 100 - (100 / (1 + rs))
        
        # 5. Standard Deviation
        dataframe['std'] = dataframe['close'].rolling(window=20).std()
        dataframe['std_normalized'] = dataframe['std'] / dataframe['close']
        
        # 添加量能指标辅助判断
        dataframe['volume_ma'] = dataframe['volume'].rolling(window=20).mean()
        
        # 价格位置指标
        dataframe['price_position_donchian'] = (dataframe['close'] - dataframe['donchian_lower']) / (dataframe['donchian_upper'] - dataframe['donchian_lower'])
        dataframe['price_position_keltner'] = (dataframe['close'] - dataframe['keltner_lower']) / (dataframe['keltner_upper'] - dataframe['keltner_lower'])
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        基于波动率指标生成买入信号
        """
        conditions = []
        
        # 买入条件1：价格突破Donchian上轨
        conditions.append(
            (dataframe['close'] > dataframe['donchian_upper'].shift(1)) &
            (dataframe['volume'] > dataframe['volume_ma'])
        )
        
        # 买入条件2：价格突破Keltner上轨且RVI上升
        conditions.append(
            (dataframe['close'] > dataframe['keltner_upper']) &
            (dataframe['rvi'] > self.buy_rvi_threshold.value) &
            (dataframe['rvi'] > dataframe['rvi'].shift(1))
        )
        
        # 买入条件3：CHV增加且标准差适中（波动率增加但不过度）
        conditions.append(
            (dataframe['chv'] > dataframe['chv'].shift(1) * (1 + self.buy_chv_increase.value)) &
            (dataframe['std_normalized'] < self.std_threshold.value * 2) &
            (dataframe['std_normalized'] > self.std_threshold.value * 0.5)
        )
        
        # 买入条件4：价格在通道下轨反弹
        conditions.append(
            (dataframe['close'] > dataframe['donchian_lower']) &
            (dataframe['close'].shift(1) <= dataframe['donchian_lower'].shift(1)) &
            (dataframe['rvi'] > 40) &
            (dataframe['volume'] > dataframe['volume_ma'] * 1.5)
        )
        
        # 合并所有买入条件
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'enter_long'] = 1
        
        # 做空条件
        short_conditions = []
        
        # 做空条件1：价格跌破Donchian下轨
        short_conditions.append(
            (dataframe['close'] < dataframe['donchian_lower'].shift(1)) &
            (dataframe['volume'] > dataframe['volume_ma'])
        )
        
        # 做空条件2：价格跌破Keltner下轨且RVI下降
        short_conditions.append(
            (dataframe['close'] < dataframe['keltner_lower']) &
            (dataframe['rvi'] < self.sell_rvi_threshold.value) &
            (dataframe['rvi'] < dataframe['rvi'].shift(1))
        )
        
        if short_conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, short_conditions),
                'enter_short'] = 1
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        基于波动率指标生成卖出信号
        """
        exit_long_conditions = []
        
        # 退出多头条件1：价格跌破Donchian中轨
        exit_long_conditions.append(
            dataframe['close'] < dataframe['donchian_middle']
        )
        
        # 退出多头条件2：RVI下降到阈值以下
        exit_long_conditions.append(
            (dataframe['rvi'] < self.sell_rvi_threshold.value) &
            (dataframe['rvi'] < dataframe['rvi'].shift(1))
        )
        
        # 退出多头条件3：价格跌破Keltner中轨且波动率增大
        exit_long_conditions.append(
            (dataframe['close'] < dataframe['keltner_middle']) &
            (dataframe['std_normalized'] > self.std_threshold.value * 2)
        )
        
        # 退出多头条件4：CHV急剧下降（波动率萎缩）
        exit_long_conditions.append(
            dataframe['chv'] < dataframe['chv'].shift(1) * 0.7
        )
        
        if exit_long_conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, exit_long_conditions),
                'exit_long'] = 1
        
        # 退出做空条件
        exit_short_conditions = []
        
        # 退出做空条件1：价格突破Donchian中轨
        exit_short_conditions.append(
            dataframe['close'] > dataframe['donchian_middle']
        )
        
        # 退出做空条件2：RVI上升到阈值以上
        exit_short_conditions.append(
            (dataframe['rvi'] > self.buy_rvi_threshold.value) &
            (dataframe['rvi'] > dataframe['rvi'].shift(1))
        )
        
        if exit_short_conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, exit_short_conditions),
                'exit_short'] = 1
        
        return dataframe
    
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs) -> Optional[str]:
        """
        自定义退出逻辑
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()
        
        # 如果是多头仓位
        if trade.is_short is False:
            # 如果CHV急剧上升，表明市场进入高波动期，考虑获利了结
            if current_candle['chv'] > current_candle['chv'] * 2:
                if current_profit > 0.02:
                    return 'high_volatility_take_profit'
            
            # 如果价格接近Donchian上轨且RVI开始下降，考虑退出
            if current_candle['price_position_donchian'] > 0.9 and current_candle['rvi'] < 60:
                if current_profit > 0:
                    return 'donchian_upper_resistance'
        
        # 如果是空头仓位
        else:
            # 如果CHV急剧上升且价格反弹，考虑止损
            if current_candle['chv'] > current_candle['chv'] * 2:
                if current_profit < -0.01:
                    return 'high_volatility_stop_loss'
            
            # 如果价格接近Donchian下轨且RVI开始上升，考虑退出
            if current_candle['price_position_donchian'] < 0.1 and current_candle['rvi'] > 40:
                if current_profit > 0:
                    return 'donchian_lower_support'
        
        return None
    
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:
        """
        根据波动率调整仓位大小
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()
        
        # 基础仓位
        stake = proposed_stake
        
        # 根据标准差调整仓位（波动率越大，仓位越小）
        if current_candle['std_normalized'] > self.std_threshold.value * 2:
            stake = stake * 0.5  # 高波动时减半仓位
        elif current_candle['std_normalized'] < self.std_threshold.value:
            stake = stake * 1.2  # 低波动时增加20%仓位
        
        # 根据RVI调整仓位（趋势越强，仓位越大）
        if side == "long":
            if current_candle['rvi'] > 70:
                stake = stake * 1.1  # 强势上涨趋势增加10%
            elif current_candle['rvi'] < 40:
                stake = stake * 0.8  # 弱势时减少20%
        
        # 确保仓位在允许范围内
        return min(max(stake, min_stake or 0), max_stake)


# 导入reduce函数
from functools import reduce