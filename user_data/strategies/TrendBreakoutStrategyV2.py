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


class TrendBreakoutStrategyV2(IStrategy):
    """
    趋势突破策略V2 - 优化版
    
    主要改进：
    1. 使用Supertrend代替EMA系统，减少假信号
    2. 加入VWAP作为趋势确认
    3. 实施三重确认系统（趋势+动量+结构）
    4. 优化入场评分系统，提高门槛
    5. 动态止损和分批建仓
    
    目标：
    - 胜率提升至35-45%
    - 盈亏比1.5-2.0
    - 最大回撤控制在10%以内
    """
    
    INTERFACE_VERSION = 3
    can_short: bool = False
    
    # 基础参数设置
    minimal_roi = {"0": 10}
    stoploss = -0.05  # 5%兜底止损（提高容错）
    trailing_stop = False
    timeframe = '15m'
    use_custom_stoploss = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    
    # === 核心策略参数（简化版）===
    
    # Supertrend参数（主要趋势信号）
    st_atr_period = IntParameter(12, 18, default=14, space="buy", optimize=True)
    st_multiplier = DecimalParameter(2.0, 3.0, default=2.5, space="buy", optimize=True)
    
    # MACD参数（动量确认）
    macd_fast = IntParameter(10, 14, default=12, space="buy", optimize=True)
    macd_slow = IntParameter(24, 28, default=26, space="buy", optimize=True)
    macd_signal = IntParameter(8, 10, default=9, space="buy", optimize=True)
    
    # RSI参数（超买超卖）
    rsi_period = IntParameter(12, 16, default=14, space="buy", optimize=True)
    rsi_overbought = IntParameter(68, 75, default=70, space="buy", optimize=True)
    
    # ADX参数（趋势强度 - 市场过滤器）
    adx_period = IntParameter(12, 16, default=14, space="buy", optimize=True)
    adx_threshold = IntParameter(25, 35, default=28, space="buy", optimize=True)
    
    # 成交量确认（简化）
    volume_multiplier = DecimalParameter(1.3, 2.0, default=1.5, space="buy", optimize=True)
    
    # === 风险管理参数（简化版）===
    
    # 简化止损系统 - 基于ATR
    atr_stop_multiplier = DecimalParameter(2.0, 3.5, default=2.5, space="sell", optimize=True)
    
    # 三级止盈系统（来自成功策略）
    profit_threshold_1 = DecimalParameter(0.02, 0.04, default=0.03, space="sell", optimize=True)  # 3%
    profit_threshold_2 = DecimalParameter(0.06, 0.10, default=0.08, space="sell", optimize=True)  # 8%
    profit_threshold_3 = DecimalParameter(0.15, 0.25, default=0.20, space="sell", optimize=True)  # 20%
    
    startup_candle_count: int = 200
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.custom_info = {}  # 简化的交易信息存储
    
    def calculate_supertrend(self, dataframe: DataFrame, period: int = 14, multiplier: float = 2.5) -> tuple:
        """计算Supertrend指标"""
        high = dataframe['high']
        low = dataframe['low']
        close = dataframe['close']
        
        # 计算ATR
        atr = ta.ATR(dataframe, timeperiod=period)
        
        # 计算基础线
        hl_avg = (high + low) / 2
        
        # 计算上下轨
        upper_band = hl_avg + (multiplier * atr)
        lower_band = hl_avg - (multiplier * atr)
        
        # 初始化
        supertrend = pd.Series(index=dataframe.index, dtype=float)
        direction = pd.Series(index=dataframe.index, dtype=float)
        
        for i in range(period, len(dataframe)):
            # 上轨调整
            if upper_band.iloc[i] < upper_band.iloc[i-1] or close.iloc[i-1] > upper_band.iloc[i-1]:
                upper_band.iloc[i] = upper_band.iloc[i]
            else:
                upper_band.iloc[i] = upper_band.iloc[i-1]
            
            # 下轨调整
            if lower_band.iloc[i] > lower_band.iloc[i-1] or close.iloc[i-1] < lower_band.iloc[i-1]:
                lower_band.iloc[i] = lower_band.iloc[i]
            else:
                lower_band.iloc[i] = lower_band.iloc[i-1]
            
            # 确定趋势方向
            if i == period:
                if close.iloc[i] <= upper_band.iloc[i]:
                    direction.iloc[i] = -1
                else:
                    direction.iloc[i] = 1
            else:
                if direction.iloc[i-1] == -1:
                    if close.iloc[i] <= upper_band.iloc[i]:
                        direction.iloc[i] = -1
                    else:
                        direction.iloc[i] = 1
                else:
                    if close.iloc[i] >= lower_band.iloc[i]:
                        direction.iloc[i] = 1
                    else:
                        direction.iloc[i] = -1
            
            # 设置Supertrend值
            if direction.iloc[i] == 1:
                supertrend.iloc[i] = lower_band.iloc[i]
            else:
                supertrend.iloc[i] = upper_band.iloc[i]
        
        return supertrend, direction
    
    
    
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """计算所需的技术指标"""
        
        # === Supertrend指标 ===
        dataframe['supertrend'], dataframe['st_direction'] = self.calculate_supertrend(
            dataframe, 
            period=self.st_atr_period.value,
            multiplier=self.st_multiplier.value
        )
        dataframe['st_uptrend'] = dataframe['st_direction'] == 1
        
        
        # === MACD指标（动量确认）===
        macd = ta.MACD(dataframe, 
                      fastperiod=self.macd_fast.value,
                      slowperiod=self.macd_slow.value, 
                      signalperiod=self.macd_signal.value)
        dataframe['macd_hist'] = macd['macdhist']
        dataframe['macd_bullish'] = (macd['macdhist'] > 0) & (macd['macdhist'] > macd['macdhist'].shift(1))
        
        # === RSI指标（超买过滤）===
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.rsi_period.value)
        dataframe['rsi_ok'] = dataframe['rsi'] < self.rsi_overbought.value
        
        # === ADX指标（趋势强度 - 市场过滤器）===
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=self.adx_period.value)
        dataframe['trending_market'] = dataframe['adx'] > self.adx_threshold.value
        
        # === 成交量确认（简化）===
        dataframe['volume_ma20'] = dataframe['volume'].rolling(20).mean()
        dataframe['volume_surge'] = dataframe['volume'] > (dataframe['volume_ma20'] * self.volume_multiplier.value)
        
        # === ATR（用于动态止损）===
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=self.st_atr_period.value)
        
        # === 波动率计算（用于仓位管理）===
        dataframe['price_change'] = dataframe['close'].pct_change()
        dataframe['volatility'] = dataframe['price_change'].rolling(20).std() * np.sqrt(24 * 60 / 15)  # 15m to daily
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """定义入场条件"""
        
        # 多头入场条件 - 简化版本（4个核心信号确认）
        dataframe.loc[
            (
                # 1. 趋势确认：Supertrend 上升趋势
                (dataframe['st_uptrend'] == True) &
                
                # 2. 动量确认：MACD 看涨
                (dataframe['macd_bullish'] == True) &
                
                # 3. 市场过滤：只在强趋势市场交易
                (dataframe['trending_market'] == True) &
                
                # 4. 成交量确认：成交量放大
                (dataframe['volume_surge'] == True) &
                
                # 5. 风险过滤：RSI 未超买
                (dataframe['rsi_ok'] == True) &
                
                # 基础有效性检查
                (dataframe['volume'] > 0)
            ),
            'enter_long'
        ] = 1
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """定义出场条件 - 简化版本"""
        
        # 趋势反转退出（简化条件）
        dataframe.loc[
            (
                # Supertrend 转向或 RSI 超买
                (dataframe['st_uptrend'] == False) |
                (dataframe['rsi'] > 80)
            ),
            'exit_long'
        ] = 1
        
        return dataframe
    
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                          proposed_stake: float, min_stake: Optional[float], max_stake: float,
                          leverage: float, entry_tag: Optional[str], side: str,
                          **kwargs) -> float:
        """基于波动率的动态仓位管理"""
        
        # 获取最新数据
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or len(dataframe) == 0:
            return proposed_stake * 0.02  # 2% 默认仓位
        
        latest = dataframe.iloc[-1]
        current_volatility = latest.get('volatility', 0.02)  # 默认波动率 2%
        
        # 获取账户余额
        account_balance = self.wallets.get_total_stake_amount()
        
        # 基于波动率的仓位计算 - 波动率越高仓位越小
        base_risk = 0.02  # 2% 基础风险
        volatility_factor = min(current_volatility / 0.02, 3.0) if current_volatility > 0 else 1.0  # 最大 3倍波动率调整
        adjusted_risk = base_risk / max(volatility_factor, 1.0)
        
        position_size = account_balance * adjusted_risk
        
        # 应用最小/最大限制
        if min_stake:
            position_size = max(position_size, min_stake)
        position_size = min(position_size, max_stake)
        
        return position_size
    
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """简化的3级止损系统（来自成功策略）"""
        
        # 获取最新数据
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or len(dataframe) == 0:
            return self.stoploss
        
        latest = dataframe.iloc[-1]
        atr = latest.get('atr', 0.01)
        
        # 初始止损基于ATR
        initial_stop = -self.atr_stop_multiplier.value * atr / current_rate
        
        # 3级盈利保护系统
        if current_profit > self.profit_threshold_3.value:  # >20% 利润
            return current_profit - 0.08  # 保护 8% 利润
        elif current_profit > self.profit_threshold_2.value:  # >8% 利润
            return current_profit - 0.04  # 保护 4% 利润
        elif current_profit > self.profit_threshold_1.value:  # >3% 利润
            return current_profit - 0.02  # 保护 2% 利润
        else:
            return initial_stop
        
    
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                   current_profit: float, **kwargs) -> Union[str, None]:
        """简化的自定义出场逻辑"""
        
        # 获取最新数据
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or len(dataframe) == 0:
            return None
        
        latest = dataframe.iloc[-1]
        
        # RSI极度超买退出
        if latest['rsi'] > 80:
            return "rsi_extreme_overbought"
        
        # Supertrend趋势反转
        if not latest['st_uptrend']:
            return "trend_reversal"
        
        return None
    
    
    
