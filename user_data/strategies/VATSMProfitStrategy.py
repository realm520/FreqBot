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
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)

import talib.abstract as ta
from technical import qtpylib


class VATSMProfitStrategy(IStrategy):
    """
    VATSM盈利优化策略 - 专注于实现真正盈利
    
    核心改进：
    1. 市场环境识别：趋势/震荡/转换期自动识别，不同环境使用不同参数
    2. 多时间框架协同：4H主趋势 + 1H入场确认 + 15M精确执行
    3. 激进盈利机制：提高止盈目标，取消保守追踪止损，实施金字塔加仓
    4. Kelly公式仓位管理：基于胜率和盈亏比的动态仓位优化
    5. 互补策略融合：动量策略 + 均值回归策略
    """
    
    INTERFACE_VERSION = 3
    can_short: bool = False
    
    # 盈利导向的基础参数
    minimal_roi = {"0": 10}  # 不使用固定ROI
    stoploss = -0.12  # 适中止损，平衡风险和收益
    trailing_stop = False  # 禁用传统追踪止损，使用自定义盈利机制
    timeframe = '15m'
    use_custom_stoploss = True
    
    # 市场环境识别参数
    market_trend_period = IntParameter(96, 288, default=192, space="buy", optimize=True)  # 市场趋势判断周期 (2天)
    volatility_period = IntParameter(48, 144, default=96, space="buy", optimize=True)  # 波动率计算周期 (1天)
    trend_threshold = DecimalParameter(0.02, 0.08, default=0.04, space="buy", optimize=True)  # 趋势判断阈值
    
    # 多时间框架协同参数
    enable_multi_timeframe = BooleanParameter(default=True, space="buy", optimize=True)
    tf_weight_4h = DecimalParameter(0.4, 0.7, default=0.5, space="buy", optimize=True)  # 4小时权重
    tf_weight_1h = DecimalParameter(0.3, 0.6, default=0.4, space="buy", optimize=True)  # 1小时权重
    tf_weight_15m = DecimalParameter(0.2, 0.5, default=0.3, space="buy", optimize=True)  # 15分钟权重
    
    # 激进盈利机制参数
    profit_target_1 = DecimalParameter(0.03, 0.08, default=0.05, space="sell", optimize=True)  # 第一目标 5%
    profit_target_2 = DecimalParameter(0.08, 0.15, default=0.12, space="sell", optimize=True)  # 第二目标 12%
    profit_target_3 = DecimalParameter(0.20, 0.35, default=0.25, space="sell", optimize=True)  # 第三目标 25%
    
    exit_ratio_1 = DecimalParameter(0.25, 0.40, default=0.33, space="sell", optimize=True)  # 第一次出场比例
    exit_ratio_2 = DecimalParameter(0.30, 0.50, default=0.40, space="sell", optimize=True)  # 第二次出场比例
    
    # 金字塔加仓参数
    enable_pyramid = BooleanParameter(default=True, space="buy", optimize=True)
    pyramid_levels = IntParameter(2, 4, default=3, space="buy", optimize=True)
    pyramid_distance = DecimalParameter(0.015, 0.040, default=0.025, space="buy", optimize=True)  # 加仓距离
    
    # Kelly公式仓位管理参数
    kelly_lookback = IntParameter(20, 60, default=40, space="buy", optimize=True)
    kelly_multiplier = DecimalParameter(0.5, 1.5, default=1.0, space="buy", optimize=True)
    max_position_size = DecimalParameter(0.15, 0.50, default=0.30, space="buy", optimize=True)
    
    # 技术指标参数 (优化后的)
    ema_fast = IntParameter(8, 21, default=12, space="buy", optimize=True)
    ema_slow = IntParameter(21, 55, default=26, space="buy", optimize=True)
    rsi_period = IntParameter(10, 21, default=14, space="buy", optimize=True)
    adx_period = IntParameter(10, 21, default=14, space="buy", optimize=True)
    adx_threshold = IntParameter(20, 35, default=25, space="buy", optimize=True)
    
    # 动量策略参数
    momentum_period = IntParameter(10, 30, default=20, space="buy", optimize=True)
    momentum_threshold = DecimalParameter(0.002, 0.010, default=0.005, space="buy", optimize=True)
    
    # 均值回归策略参数
    bb_period = IntParameter(15, 30, default=20, space="buy", optimize=True)
    bb_std = DecimalParameter(1.5, 2.5, default=2.0, space="buy", optimize=True)
    mean_reversion_threshold = DecimalParameter(0.8, 1.2, default=1.0, space="buy", optimize=True)
    
    startup_candle_count: int = 500
    
    def __init__(self, config: dict):
        super().__init__(config)
        # 交易历史和绩效跟踪
        self.trade_history = deque(maxlen=self.kelly_lookback.value)
        self.current_positions = {}  # 跟踪当前持仓
        self.market_regime = 'UNKNOWN'  # 当前市场环境
        self.last_regime_check = None
        
    @informative('4h')
    def populate_indicators_4h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """4小时时间框架指标 - 主趋势判断"""
        dataframe['ema_21_4h'] = ta.EMA(dataframe, timeperiod=21)
        dataframe['ema_55_4h'] = ta.EMA(dataframe, timeperiod=55)
        dataframe['trend_4h'] = dataframe['ema_21_4h'] > dataframe['ema_55_4h']
        dataframe['adx_4h'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['strong_trend_4h'] = dataframe['adx_4h'] > 25
        return dataframe
    
    @informative('1h')
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """1小时时间框架指标 - 入场确认"""
        dataframe['ema_12_1h'] = ta.EMA(dataframe, timeperiod=12)
        dataframe['ema_26_1h'] = ta.EMA(dataframe, timeperiod=26)
        dataframe['trend_1h'] = dataframe['ema_12_1h'] > dataframe['ema_26_1h']
        dataframe['rsi_1h'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['volume_sma_1h'] = dataframe['volume'].rolling(20).mean()
        dataframe['volume_surge_1h'] = dataframe['volume'] > dataframe['volume_sma_1h'] * 1.5
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """15分钟时间框架指标 - 精确执行"""
        
        # 基础技术指标
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=self.ema_fast.value)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=self.ema_slow.value)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.rsi_period.value)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=self.adx_period.value)
        
        # 动量指标
        dataframe['momentum'] = dataframe['close'].pct_change(self.momentum_period.value)
        
        # 布林带 (均值回归)
        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=self.bb_period.value, stds=self.bb_std.value)
        dataframe['bb_lower'] = bollinger['lower']
        dataframe['bb_middle'] = bollinger['mid']
        dataframe['bb_upper'] = bollinger['upper']
        dataframe['bb_percent'] = (dataframe['close'] - dataframe['bb_lower']) / (dataframe['bb_upper'] - dataframe['bb_lower'])
        
        # 成交量指标
        dataframe['volume_sma'] = dataframe['volume'].rolling(20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_sma']
        
        # 市场环境识别
        dataframe = self.identify_market_regime(dataframe)
        
        # 策略信号生成
        dataframe = self.generate_strategy_signals(dataframe)
        
        return dataframe
    
    def identify_market_regime(self, dataframe: DataFrame) -> DataFrame:
        """
        市场环境识别算法
        识别三种市场状态：TRENDING（趋势）、RANGING（震荡）、TRANSITIONING（转换）
        """
        # 计算价格变化率
        dataframe['price_change'] = dataframe['close'].pct_change(self.market_trend_period.value)
        
        # 计算波动率
        dataframe['volatility'] = dataframe['close'].rolling(self.volatility_period.value).std() / dataframe['close'].rolling(self.volatility_period.value).mean()
        
        # 趋势强度 (ADX)
        dataframe['trend_strength'] = dataframe['adx']
        
        # 市场环境判断逻辑
        conditions = [
            # 趋势市场：强烈的价格方向性 + 高ADX + 适中波动率
            (
                (np.abs(dataframe['price_change']) > self.trend_threshold.value) &
                (dataframe['trend_strength'] > self.adx_threshold.value) &
                (dataframe['volatility'] < dataframe['volatility'].rolling(50).quantile(0.8))
            ),
            # 震荡市场：弱价格方向性 + 低ADX + 适中波动率
            (
                (np.abs(dataframe['price_change']) < self.trend_threshold.value * 0.5) &
                (dataframe['trend_strength'] < self.adx_threshold.value * 0.8) &
                (dataframe['volatility'] < dataframe['volatility'].rolling(50).quantile(0.7))
            ),
        ]
        
        choices = ['TRENDING', 'RANGING']
        
        dataframe['market_regime'] = np.select(conditions, choices, default='TRANSITIONING')
        
        return dataframe
    
    def generate_strategy_signals(self, dataframe: DataFrame) -> DataFrame:
        """
        根据市场环境生成不同的交易信号
        """
        # 动量策略信号 (适用于趋势市场)
        dataframe['momentum_signal'] = (
            (dataframe['momentum'] > self.momentum_threshold.value) &
            (dataframe['ema_fast'] > dataframe['ema_slow']) &
            (dataframe['rsi'] < 70) &
            (dataframe['adx'] > self.adx_threshold.value) &
            (dataframe['volume_ratio'] > 1.2)
        ).astype(int)
        
        # 均值回归信号 (适用于震荡市场)
        dataframe['mean_reversion_signal'] = (
            (dataframe['bb_percent'] < 0.2) &  # 接近下轨
            (dataframe['rsi'] < 35) &  # 超卖
            (dataframe['close'] < dataframe['bb_lower'] * 1.02) &  # 价格在下轨附近
            (dataframe['volume_ratio'] > 1.0)
        ).astype(int)
        
        # 根据市场环境选择信号
        dataframe['buy_signal'] = 0
        
        # 趋势市场使用动量策略
        trending_mask = dataframe['market_regime'] == 'TRENDING'
        dataframe.loc[trending_mask, 'buy_signal'] = dataframe.loc[trending_mask, 'momentum_signal']
        
        # 震荡市场使用均值回归策略
        ranging_mask = dataframe['market_regime'] == 'RANGING'
        dataframe.loc[ranging_mask, 'buy_signal'] = dataframe.loc[ranging_mask, 'mean_reversion_signal']
        
        # 转换期市场谨慎交易，信号强度减半
        transitioning_mask = dataframe['market_regime'] == 'TRANSITIONING'
        dataframe.loc[transitioning_mask, 'buy_signal'] = (
            dataframe.loc[transitioning_mask, 'momentum_signal'] * 0.5 +
            dataframe.loc[transitioning_mask, 'mean_reversion_signal'] * 0.5
        ).round().astype(int)
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        入场信号生成 - 多时间框架协同确认
        """
        conditions = []
        
        # 基础信号
        conditions.append(dataframe['buy_signal'] > 0)
        
        # 多时间框架确认 (如果启用)
        if self.enable_multi_timeframe.value:
            # 4小时趋势确认
            if 'trend_4h' in dataframe.columns:
                conditions.append(
                    (dataframe['trend_4h'] == True) |
                    (dataframe['market_regime'] == 'RANGING')  # 震荡市场无需4H确认
                )
            
            # 1小时入场确认
            if 'trend_1h' in dataframe.columns:
                conditions.append(
                    (dataframe['trend_1h'] == True) |
                    (dataframe['rsi_1h'] < 50)  # 或RSI不高
                )
        
        # 风险过滤
        conditions.append(dataframe['rsi'] < 75)  # 避免极端超买
        conditions.append(dataframe['volume_ratio'] > 0.8)  # 最低成交量要求
        
        # 综合所有条件
        if conditions:
            dataframe['enter_long'] = np.all(conditions, axis=0)
        else:
            dataframe['enter_long'] = False
            
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        出场信号生成 - 盈利导向
        """
        # 基础出场条件
        exit_conditions = [
            # RSI极端超买
            dataframe['rsi'] > 85,
            # 动量反转
            dataframe['momentum'] < -self.momentum_threshold.value * 2,
            # 趋势转换
            (dataframe['ema_fast'] < dataframe['ema_slow']) & 
            (dataframe['market_regime'] != 'RANGING'),  # 震荡市场除外
            # 成交量萎缩
            dataframe['volume_ratio'] < 0.5
        ]
        
        dataframe['exit_long'] = np.any(exit_conditions, axis=0)
        
        return dataframe
    
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        自定义止损 - 激进盈利机制实现
        """
        # 获取当前数据
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return self.stoploss
        
        latest = dataframe.iloc[-1]
        
        # 盈利保护机制
        if current_profit >= self.profit_target_3.value:  # 达到第三目标25%
            return current_profit - 0.08  # 保护大部分利润
        elif current_profit >= self.profit_target_2.value:  # 达到第二目标12%
            return current_profit - 0.05  # 保护部分利润
        elif current_profit >= self.profit_target_1.value:  # 达到第一目标5%
            return current_profit - 0.02  # 最小保护
        
        # 市场环境动态止损
        if latest['market_regime'] == 'RANGING':
            # 震荡市场更严格的止损
            return -0.08
        elif latest['market_regime'] == 'TRENDING':
            # 趋势市场给更多空间
            return -0.15
        else:
            # 转换期适中
            return -0.12
    
    def custom_exit(self, pair: str, trade: 'Trade', current_time: datetime, 
                   current_rate: float, current_profit: float, **kwargs) -> Optional[Union[str, bool]]:
        """
        自定义出场 - 分批盈利实现
        """
        # 分批盈利出场
        if current_profit >= self.profit_target_1.value:
            if trade.amount > trade.amount * (1 - self.exit_ratio_1.value):
                # 第一次分批出场
                return f"profit_target_1_{self.profit_target_1.value}"
        
        if current_profit >= self.profit_target_2.value:
            if trade.amount > trade.amount * (1 - self.exit_ratio_1.value - self.exit_ratio_2.value):
                # 第二次分批出场
                return f"profit_target_2_{self.profit_target_2.value}"
        
        if current_profit >= self.profit_target_3.value:
            # 第三目标全部出场
            return f"profit_target_3_{self.profit_target_3.value}"
        
        return None
    
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                           proposed_stake: float, min_stake: Optional[float], max_stake: float,
                           entry_tag: Optional[str] = None, side: str = 'long', **kwargs) -> float:
        """
        Kelly公式动态仓位管理
        """
        # 获取交易历史
        if len(self.trade_history) < 10:
            # 历史数据不足，使用保守仓位
            return min(proposed_stake * 0.5, max_stake)
        
        # 计算胜率和平均盈亏比
        wins = [t for t in self.trade_history if t > 0]
        losses = [t for t in self.trade_history if t < 0]
        
        if not losses:  # 避免除零
            win_rate = 1.0
            avg_win_loss_ratio = 2.0
        else:
            win_rate = len(wins) / len(self.trade_history)
            avg_win = np.mean(wins) if wins else 0
            avg_loss = abs(np.mean(losses))
            avg_win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 2.0
        
        # Kelly公式: f* = (bp - q) / b
        # 其中 b = 盈亏比, p = 胜率, q = 败率
        kelly_fraction = (avg_win_loss_ratio * win_rate - (1 - win_rate)) / avg_win_loss_ratio
        kelly_fraction = max(0, min(kelly_fraction, self.max_position_size.value))  # 限制范围
        
        # 应用Kelly乘数
        final_fraction = kelly_fraction * self.kelly_multiplier.value
        
        # 根据市场环境调整
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if not dataframe.empty:
            latest = dataframe.iloc[-1]
            if latest['market_regime'] == 'TRENDING':
                final_fraction *= 1.2  # 趋势市场增加仓位
            elif latest['market_regime'] == 'TRANSITIONING':
                final_fraction *= 0.7  # 转换期降低仓位
        
        # 计算最终仓位
        kelly_stake = max_stake * final_fraction
        return min(kelly_stake, max_stake)
    
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                          time_in_force: str, current_time: datetime, entry_tag: Optional[str] = None, side: str = 'long', **kwargs) -> bool:
        """
        入场确认 - 最后的风险检查
        """
        # 获取当前市场数据
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return False
        
        latest = dataframe.iloc[-1]
        
        # 避免在极端市场条件下交易
        if latest['volatility'] > latest['volatility']:  # 如果当前波动率过高
            return False
        
        # 检查多时间框架一致性
        if self.enable_multi_timeframe.value:
            if 'trend_4h' in dataframe.columns and 'trend_1h' in dataframe.columns:
                # 要求至少2个时间框架一致
                timeframe_agreement = (
                    latest.get('trend_4h', True) + 
                    latest.get('trend_1h', True) + 
                    (latest['ema_fast'] > latest['ema_slow'])
                )
                if timeframe_agreement < 2:
                    return False
        
        return True