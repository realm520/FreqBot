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


class TrendBreakoutStrategy(IStrategy):
    """
    趋势突破策略 - 日内短线永续合约交易
    
    策略核心思路：
    1. 交易周期：1min ~ 15min K线，偏日内短线，不隔夜持仓
    2. 交易逻辑：趋势跟随 + 波动率突破为主，辅以资金管理约束
    3. 目标：日收益 0.5%~2%，最大回撤不超过 2%~3%
    
    入场逻辑：
    - 趋势过滤：EMA(20, 60)均线系统确定方向
    - 突破触发：布林带突破 + ATR突破双重确认
    - 成交量放大：Volume > 均值 1.5 倍
    
    出场逻辑：
    - 止损：单笔风险 0.3%~0.5%，技术止损参考布林轨
    - 止盈：1.5~2倍止损距离，移动止盈跟随5EMA
    - 风控：日累计亏损2%强制清仓，日盈利2%停止开仓
    """
    
    INTERFACE_VERSION = 3
    can_short: bool = False  # 现货交易不支持做空
    
    # 基础参数设置
    minimal_roi = {"0": 10}  # 不使用固定ROI，依赖策略信号
    stoploss = -0.03  # 3%兜底止损
    trailing_stop = False  # 使用自定义追踪止损
    timeframe = '5m'  # 默认5分钟，可调整到1m-15m
    use_custom_stoploss = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    
    # === 核心策略参数 ===
    
    # 趋势判断参数
    ema_fast = IntParameter(15, 25, default=20, space="buy", optimize=True)
    ema_slow = IntParameter(50, 70, default=60, space="buy", optimize=True)
    
    # 布林带参数
    bb_period = IntParameter(15, 25, default=20, space="buy", optimize=True)
    bb_std = DecimalParameter(1.8, 2.5, default=2.0, space="buy", optimize=True)
    
    # ATR突破参数
    atr_period = IntParameter(10, 20, default=14, space="buy", optimize=True)
    atr_multiplier = DecimalParameter(0.8, 1.5, default=1.0, space="buy", optimize=True)
    
    # 成交量确认参数
    volume_period = IntParameter(15, 25, default=20, space="buy", optimize=True)
    volume_multiplier = DecimalParameter(1.1, 1.8, default=1.3, space="buy", optimize=True)  # 放宽成交量要求
    
    # 辅助过滤参数
    rsi_period = IntParameter(10, 20, default=14, space="buy", optimize=True)
    rsi_oversold = IntParameter(25, 35, default=30, space="buy", optimize=True)
    rsi_overbought = IntParameter(65, 75, default=70, space="buy", optimize=True)
    
    adx_period = IntParameter(10, 20, default=14, space="buy", optimize=True)
    adx_threshold = IntParameter(15, 25, default=20, space="buy", optimize=True)  # 放宽ADX要求
    
    # === 风险管理参数 ===
    
    # 单笔风险控制
    risk_per_trade = DecimalParameter(0.003, 0.008, default=0.005, space="buy", optimize=True)  # 0.5%
    
    # 止盈止损比例
    profit_loss_ratio = DecimalParameter(1.5, 3.0, default=2.0, space="sell", optimize=True)
    
    # 移动止盈参数
    trailing_ema_period = IntParameter(3, 8, default=5, space="sell", optimize=True)
    
    # 日度风险限制
    max_daily_loss = DecimalParameter(0.015, 0.025, default=0.02, space="sell", optimize=True)  # 2%
    max_daily_profit = DecimalParameter(0.015, 0.025, default=0.02, space="sell", optimize=True)  # 2%
    max_daily_trades = IntParameter(5, 10, default=8, space="buy", optimize=True)
    
    # 杠杆控制
    max_leverage = DecimalParameter(2.0, 5.0, default=3.0, space="buy", optimize=True)
    margin_usage_limit = DecimalParameter(0.20, 0.35, default=0.30, space="buy", optimize=True)  # 30%
    
    # 避开重要时间段参数
    avoid_news_minutes = IntParameter(10, 20, default=15, space="buy", optimize=True)
    
    startup_candle_count: int = 200
    
    def __init__(self, config: dict):
        super().__init__(config)
        # 风险管理状态跟踪
        self.daily_pnl = 0.0
        self.daily_trade_count = 0
        self.last_day_check = None
        self.trading_paused = False
        self.trade_history = deque(maxlen=50)  # 记录交易历史
        
        # 追踪止盈状态
        self.custom_info = {}
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """计算所需的技术指标"""
        
        # === 趋势判断指标 ===
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=self.ema_fast.value)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=self.ema_slow.value)
        dataframe['trend_up'] = dataframe['ema_fast'] > dataframe['ema_slow']
        dataframe['trend_strength'] = (dataframe['ema_fast'] - dataframe['ema_slow']) / dataframe['ema_slow']
        
        # === 布林带指标 ===
        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=self.bb_period.value, stds=self.bb_std.value)
        dataframe['bb_lower'] = bollinger['lower']
        dataframe['bb_middle'] = bollinger['mid']
        dataframe['bb_upper'] = bollinger['upper']
        dataframe['bb_width'] = (dataframe['bb_upper'] - dataframe['bb_lower']) / dataframe['bb_middle']
        
        # 布林带突破信号
        dataframe['bb_break_up'] = dataframe['close'] > dataframe['bb_upper']
        dataframe['bb_break_down'] = dataframe['close'] < dataframe['bb_lower']
        
        # === ATR波动率指标 ===
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=self.atr_period.value)
        dataframe['atr_threshold'] = dataframe['atr'] * self.atr_multiplier.value
        
        # ATR突破信号
        dataframe['price_change'] = dataframe['close'] - dataframe['close'].shift(1)
        dataframe['atr_break_up'] = dataframe['price_change'] > dataframe['atr_threshold']
        dataframe['atr_break_down'] = dataframe['price_change'] < -dataframe['atr_threshold']
        
        # === 成交量确认 ===
        dataframe['volume_ma'] = dataframe['volume'].rolling(self.volume_period.value).mean()
        dataframe['volume_surge'] = dataframe['volume'] > (dataframe['volume_ma'] * self.volume_multiplier.value)
        
        # === 辅助过滤指标 ===
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.rsi_period.value)
        dataframe['rsi_oversold'] = dataframe['rsi'] < self.rsi_oversold.value
        dataframe['rsi_overbought'] = dataframe['rsi'] > self.rsi_overbought.value
        dataframe['rsi_neutral'] = (~dataframe['rsi_oversold']) & (~dataframe['rsi_overbought'])
        
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=self.adx_period.value)
        dataframe['adx_strong'] = dataframe['adx'] > self.adx_threshold.value
        
        # === 组合突破信号 ===
        # 向上突破确认：布林带突破 OR ATR突破
        dataframe['breakout_up'] = (dataframe['bb_break_up'] | dataframe['atr_break_up'])
        # 向下突破确认：布林带突破 OR ATR突破  
        dataframe['breakout_down'] = (dataframe['bb_break_down'] | dataframe['atr_break_down'])
        
        # === 移动止盈EMA ===
        dataframe['trailing_ema'] = ta.EMA(dataframe, timeperiod=self.trailing_ema_period.value)
        
        # === 信号强度评分 ===
        dataframe['signal_score'] = 0
        
        # 趋势强度评分 (0-2分)
        dataframe['signal_score'] += np.where(np.abs(dataframe['trend_strength']) > 0.02, 2, 1)
        
        # ADX强度评分 (0-1分)
        dataframe['signal_score'] += np.where(dataframe['adx_strong'], 1, 0)
        
        # 成交量确认评分 (0-1分)
        dataframe['signal_score'] += np.where(dataframe['volume_surge'], 1, 0)
        
        # 布林带宽度评分 (0-1分，波动率适中时给分)
        dataframe['signal_score'] += np.where(
            (dataframe['bb_width'] > 0.02) & (dataframe['bb_width'] < 0.08), 1, 0
        )
        
        # === 时间过滤 ===
        # 从date列获取时间信息
        if 'date' in dataframe.columns:
            dataframe['hour'] = pd.to_datetime(dataframe['date']).dt.hour
            dataframe['minute'] = pd.to_datetime(dataframe['date']).dt.minute
            # 避开重要新闻时段（可根据需要调整）
            dataframe['avoid_news_time'] = (
                # 美国开盘前后
                ((dataframe['hour'] == 21) & (dataframe['minute'] <= self.avoid_news_minutes.value)) |
                ((dataframe['hour'] == 22) & (dataframe['minute'] <= self.avoid_news_minutes.value)) |
                # 欧洲开盘前后  
                ((dataframe['hour'] == 15) & (dataframe['minute'] <= self.avoid_news_minutes.value)) |
                ((dataframe['hour'] == 16) & (dataframe['minute'] <= self.avoid_news_minutes.value))
            )
        else:
            # 如果没有date列，则不应用时间过滤
            dataframe['avoid_news_time'] = False
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """定义入场条件"""
        
        # 多头入场条件
        dataframe.loc[
            (
                # 基础趋势过滤
                (dataframe['trend_up'] == True) &  # EMA趋势向上
                
                # 突破触发条件
                (dataframe['breakout_up'] == True) &  # 布林带或ATR向上突破
                
                # 成交量确认
                (dataframe['volume_surge'] == True) &  # 成交量放大
                
                # 辅助过滤条件
                (dataframe['adx_strong'] == True) &  # 趋势足够强
                (~dataframe['rsi_overbought']) &  # 避免超买入场
                (dataframe['signal_score'] >= 2) &  # 降低信号强度要求
                
                # 时间过滤
                (~dataframe['avoid_news_time']) &  # 避开新闻时段
                
                # 基础数据有效性
                (dataframe['volume'] > 0) &
                (dataframe['close'] > dataframe['ema_fast'])  # 价格在快速均线上方
            ),
            'enter_long'
        ] = 1
        
        # 空头入场条件 - 现货交易不支持做空，注释掉
        # dataframe.loc[
        #     (
        #         # 基础趋势过滤
        #         (dataframe['trend_up'] == False) &  # EMA趋势向下
        #         
        #         # 突破触发条件
        #         (dataframe['breakout_down'] == True) &  # 布林带或ATR向下突破
        #         
        #         # 成交量确认
        #         (dataframe['volume_surge'] == True) &  # 成交量放大
        #         
        #         # 辅助过滤条件
        #         (dataframe['adx_strong'] == True) &  # 趋势足够强
        #         (~dataframe['rsi_oversold']) &  # 避免超卖入场
        #         (dataframe['signal_score'] >= 3) &  # 信号强度足够
        #         
        #         # 时间过滤
        #         (~dataframe['avoid_news_time']) &  # 避开新闻时段
        #         
        #         # 基础数据有效性
        #         (dataframe['volume'] > 0) &
        #         (dataframe['close'] < dataframe['ema_fast'])  # 价格在快速均线下方
        #     ),
        #     'enter_short'
        # ] = 1
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """定义出场条件"""
        
        # 多头出场 - 趋势反转信号
        dataframe.loc[
            (
                (dataframe['trend_up'] == False) &  # 趋势转为向下
                (dataframe['breakout_down'] == True) &  # 向下突破
                (dataframe['adx_strong'] == True) &  # 趋势强度足够
                (dataframe['volume_surge'] == True)  # 成交量确认
            ),
            'exit_long'
        ] = 1
        
        # 空头出场 - 现货交易不支持做空，注释掉
        # dataframe.loc[
        #     (
        #         (dataframe['trend_up'] == True) &  # 趋势转为向上
        #         (dataframe['breakout_up'] == True) &  # 向上突破
        #         (dataframe['adx_strong'] == True) &  # 趋势强度足够
        #         (dataframe['volume_surge'] == True)  # 成交量确认
        #     ),
        #     'exit_short'
        # ] = 1
        
        return dataframe
    
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                          proposed_stake: float, min_stake: Optional[float], max_stake: float,
                          leverage: float, entry_tag: Optional[str], side: str,
                          **kwargs) -> float:
        """动态仓位管理"""
        
        # 检查日度风险限制
        self._check_daily_limits(current_time)
        
        if self.trading_paused:
            return 0  # 暂停交易
        
        # 获取当前市场数据
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or len(dataframe) == 0:
            return min_stake or 0
        
        latest = dataframe.iloc[-1]
        
        # 计算基于ATR的仓位大小
        atr = latest['atr']
        if atr <= 0:
            return min_stake or 0
        
        # 获取账户余额
        account_balance = self.wallets.get_total_stake_amount()
        
        # 计算单笔风险金额
        risk_amount = account_balance * self.risk_per_trade.value
        
        # 基于ATR计算止损距离，放宽止损以适应5分钟时间框架
        stop_distance = atr * 2.5  # 改为2.5倍ATR作为止损距离
        stop_distance_percent = stop_distance / current_rate
        
        # 计算仓位大小：风险金额 / 止损距离
        if stop_distance_percent > 0:
            position_size = risk_amount / stop_distance_percent
        else:
            position_size = min_stake or 0
        
        # 应用杠杆限制
        max_position_with_leverage = account_balance * self.margin_usage_limit.value * self.max_leverage.value
        position_size = min(position_size, max_position_with_leverage)
        
        # 应用最小/最大限制
        if min_stake:
            position_size = max(position_size, min_stake)
        position_size = min(position_size, max_stake)
        
        return position_size
    
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """自定义止损逻辑"""
        
        # 获取交易数据
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or len(dataframe) == 0:
            return self.stoploss
        
        latest = dataframe.iloc[-1]
        trade_id = f"{pair}_{trade.open_date}"
        
        # 初始化交易追踪信息
        if trade_id not in self.custom_info:
            self.custom_info[trade_id] = {
                'initial_stop': None,
                'trailing_active': False,
                'max_profit': current_profit,
                'entry_rate': trade.open_rate
            }
        
        trade_info = self.custom_info[trade_id]
        
        # 更新最大盈利
        if current_profit > trade_info['max_profit']:
            trade_info['max_profit'] = current_profit
        
        # 计算初始止损（基于ATR） - 现货只做多头
        if trade_info['initial_stop'] is None:
            atr = latest['atr']
            # 多头：入场价 - 2.5*ATR，放宽止损距离
            stop_price = trade.open_rate - (atr * 2.5)
            trade_info['initial_stop'] = (stop_price - trade.open_rate) / trade.open_rate
        
        initial_stop = trade_info['initial_stop']
        
        # 检查是否应该启动移动止盈
        profit_threshold = abs(initial_stop) * self.profit_loss_ratio.value
        
        if current_profit > profit_threshold:
            trade_info['trailing_active'] = True
        
        # 移动止盈逻辑 - 现货只做多头
        if trade_info['trailing_active']:
            trailing_ema = latest['trailing_ema']
            
            # 多头移动止盈：基于EMA下方
            trailing_stop_price = trailing_ema * 0.995  # EMA下方0.5%
            trailing_stop = (trailing_stop_price - trade.open_rate) / trade.open_rate
            # 确保止盈不会低于初始止损
            return max(initial_stop, trailing_stop)
        
        # 如果未启动移动止盈，使用初始止损
        return initial_stop
    
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                   current_profit: float, **kwargs) -> Union[str, None]:
        """自定义出场逻辑"""
        
        # 检查日度限制
        self._check_daily_limits(current_time)
        
        # 如果达到日度盈利目标，强制平仓
        if self.daily_pnl >= self.max_daily_profit.value:
            return "daily_profit_target"
        
        # 如果达到日度亏损限制，强制平仓
        if self.daily_pnl <= -self.max_daily_loss.value:
            return "daily_loss_limit"
        
        return None
    
    def _check_daily_limits(self, current_time: datetime):
        """检查日度限制"""
        current_date = current_time.date()
        
        # 如果是新的一天，重置计数器
        if self.last_day_check != current_date:
            self.last_day_check = current_date
            self.daily_pnl = 0.0
            self.daily_trade_count = 0
            self.trading_paused = False
        
        # 检查是否需要暂停交易
        if (self.daily_pnl <= -self.max_daily_loss.value or 
            self.daily_pnl >= self.max_daily_profit.value or
            self.daily_trade_count >= self.max_daily_trades.value):
            self.trading_paused = True
        
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                          time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                          side: str, **kwargs) -> bool:
        """确认交易入场"""
        
        # 检查日度限制
        self._check_daily_limits(current_time)
        
        if self.trading_paused:
            return False
        
        # 更新交易计数
        self.daily_trade_count += 1
        
        return True
    
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                          rate: float, time_in_force: str, exit_reason: str,
                          current_time: datetime, **kwargs) -> bool:
        """确认交易出场"""
        
        # 记录交易结果到历史
        if trade.is_open:
            profit = trade.calc_profit_ratio(rate)
            self.trade_history.append(profit)
            self.daily_pnl += profit
        
        return True