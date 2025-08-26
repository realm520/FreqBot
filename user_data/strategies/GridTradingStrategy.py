# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Optional, Union

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

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
from technical import qtpylib


class GridTradingStrategy(IStrategy):
    """
    网格交易策略 - 基于Freqtrade框架的类网格交易实现
    
    策略原理：
    1. 在价格区间内设置多个网格层级
    2. 价格下跌时分批买入，上涨时分批卖出
    3. 通过动态ROI实现网格效果
    4. 适合震荡行情，不适合单边趋势
    """

    # Strategy interface version
    INTERFACE_VERSION = 3

    # Can this strategy go short?
    can_short: bool = False

    # 网格交易参数 - 优化后的参数
    grid_levels = IntParameter(low=5, high=15, default=8, space="buy", optimize=True, load=True)
    grid_range_percent = DecimalParameter(low=0.03, high=0.12, default=0.08, space="buy", optimize=True, load=True)
    base_profit_percent = DecimalParameter(low=0.015, high=0.035, default=0.025, space="sell", optimize=True, load=True)
    
    # 动态仓位管理参数
    position_size_factor = DecimalParameter(low=0.7, high=1.3, default=1.0, space="buy", optimize=True, load=True)

    # 加快获利了结的ROI配置
    minimal_roi = {
        "0": 0.020,   # 初始2%目标
        "15": 0.015,  # 15分钟后1.5%
        "45": 0.012,  # 45分钟后1.2%
        "90": 0.009,  # 90分钟后0.9%
        "180": 0.006, # 180分钟后0.6%
        "360": 0.003, # 360分钟后0.3%
    }

    # 动态止损设置 - 将在custom_stoploss中实现
    stoploss = -0.12  # 最大止损12%，但会使用动态止损

    # Trailing stoploss - 不启用以保持网格特性
    trailing_stop = False

    # 时间框架
    timeframe = "15m"

    # Run "populate_indicators()" only for new candle
    process_only_new_candles = True

    # 网格特定设置
    use_exit_signal = True
    exit_profit_only = False  # 允许亏损时也可以退出
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 100

    # Optional order type mapping
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    # Optional order time in force
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        添加网格交易所需的技术指标
        """
        # 移动平均线 - 用于确定趋势方向
        dataframe['sma_20'] = ta.SMA(dataframe, timeperiod=20)
        dataframe['sma_50'] = ta.SMA(dataframe, timeperiod=50)
        dataframe['ema_12'] = ta.EMA(dataframe, timeperiod=12)
        dataframe['ema_26'] = ta.EMA(dataframe, timeperiod=26)

        # 布林带 - 用于确定价格区间
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_percent'] = (dataframe['close'] - dataframe['bb_lowerband']) / (
            dataframe['bb_upperband'] - dataframe['bb_lowerband']
        )
        dataframe['bb_width'] = (dataframe['bb_upperband'] - dataframe['bb_lowerband']) / dataframe['bb_middleband']

        # RSI - 用于判断超买超卖
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # ATR - 用于计算网格间距和动态止损
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['atr_percent'] = dataframe['atr'] / dataframe['close']

        # MACD - 趋势方向判断
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        
        # ADX - 趋势强度判断
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['plus_di'] = ta.PLUS_DI(dataframe, timeperiod=14)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe, timeperiod=14)

        # 价格波动率
        dataframe['price_change_pct'] = dataframe['close'].pct_change()
        dataframe['volatility'] = dataframe['price_change_pct'].rolling(window=20).std()

        # 计算网格价位
        dataframe = self.calculate_grid_levels(dataframe)

        return dataframe

    def calculate_grid_levels(self, dataframe: DataFrame) -> DataFrame:
        """
        计算网格价位
        """
        # 基于布林带中轴和ATR计算网格间距
        dataframe['grid_base_price'] = dataframe['bb_middleband']
        dataframe['grid_spacing'] = dataframe['atr'] * 0.5  # ATR的一半作为网格间距

        # 计算买入信号强度（价格距离网格基准价的偏离程度）
        dataframe['distance_from_base'] = (dataframe['close'] - dataframe['grid_base_price']) / dataframe['grid_base_price']
        
        # 网格层级指标
        dataframe['grid_level'] = (dataframe['distance_from_base'] / (self.grid_range_percent.value / self.grid_levels.value)) * -1
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        网格交易买入信号
        """
        # 优化后的网格买入条件：
        # 1. 更灵活的价格触发条件
        # 2. 改进的趋势判断
        # 3. 更宽松的市场环境条件
        # 最终优化的进场条件 - 平衡严格性和交易频率
        dataframe.loc[
            (
                # 核心价格条件
                (dataframe['close'] < dataframe['bb_middleband'])
                &
                (dataframe['bb_percent'] < 0.45)  # 放宽到45%
                &
                # RSI超卖但不过度
                (dataframe['rsi'] < 50)  # 放宽到50
                &
                (dataframe['rsi'] > 20)  # 避免极度超卖
                &
                # 简化的趋势过滤
                (
                    (dataframe['ema_12'] > dataframe['ema_26'])  # EMA多头排列
                    |
                    (dataframe['macd'] > dataframe['macdsignal'])  # 或MACD金叉
                )
                &
                # 成交量有效性
                (dataframe['volume'] > 0)
                &
                # 波动率基本要求
                (dataframe['bb_width'] > 0.005)  # 最小波动
                &
                (dataframe['bb_width'] < 0.4)    # 最大波动
            ),
            'enter_long',
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        网格交易卖出信号
        """
        # 优化后的网格卖出条件：
        # 1. 更灵活的获利了结条件
        # 2. 多重退出信号
        # 智能化退出条件 - 多级别获利策略
        dataframe.loc[
            (
                # 快速获利 - 小幅度快速获利
                (dataframe['bb_percent'] > 0.65)
                &
                (dataframe['rsi'] > 55)
                &
                (dataframe['macd'] < dataframe['macdsignal'])  # MACD显示顶部
                &
                (dataframe['volume'] > dataframe['volume'].rolling(5).mean())
            )
            |
            (
                # 中等获利 - 超买区域获利
                (dataframe['rsi'] > 65)
                &
                (dataframe['close'] > dataframe['bb_middleband'])
                &
                (dataframe['volume'] > 0)
            )
            |
            (
                # 强势获利 - 明显超买
                (dataframe['rsi'] > 72)
                &
                (dataframe['close'] > dataframe['bb_upperband'] * 0.98)  # 接近上轨
            )
            |
            (
                # 趋势反转退出 - 市场环境恶化
                (dataframe['adx'] > 25)
                &
                (dataframe['minus_di'] > dataframe['plus_di'] * 1.15)  # 空头强势
                &
                (dataframe['macd'] < dataframe['macdsignal'])
                &
                (dataframe['rsi'] > 45)  # 不在极度超卖时退出
            )
            |
            (
                # 波动率突增退出 - 市场不稳定
                (dataframe['bb_width'] > 0.15)
                &
                (dataframe['atr_percent'] > 0.06)
                &
                (dataframe['rsi'] > 50)
                &
                (dataframe['close'] > dataframe['bb_middleband'])  # 价格不在底部
            ),
            'exit_long',
        ] = 1

        return dataframe

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        动态止损 - 基于ATR和市场条件的止损
        """
        # 获取当前数据
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) < 1:
            return -1
        
        last_candle = dataframe.iloc[-1].squeeze()
        
        # 更精细的动态止损算法
        holding_hours = (current_time - trade.open_date_utc).total_seconds() / 3600
        
        # 根据交易开启时的市场状态调整基础止损
        base_atr_multiplier = 2.0
        if last_candle['atr_percent'] > 0.05:  # 高波动开始
            base_atr_multiplier = 2.5  # 放宽止损
        elif last_candle['atr_percent'] < 0.02:  # 低波动开始
            base_atr_multiplier = 1.5  # 紧缩止损
        
        atr_stop_distance = last_candle['atr_percent'] * base_atr_multiplier
        
        # 时间衰减因子 - 非线性衰减
        if holding_hours <= 2:
            time_factor = 1.0  # 早期保持宽松
        elif holding_hours <= 6:
            time_factor = 0.9  # 略微紧缩
        elif holding_hours <= 12:
            time_factor = 0.8  # 中等紧缩
        else:
            time_factor = 0.7  # 長期持仓紧缩止损
        
        # 趋势与动量因子
        trend_factor = 1.0
        volume_factor = 1.0
        
        # 趋势分析
        if last_candle['adx'] > 30:  # 强趋势
            if last_candle['minus_di'] > last_candle['plus_di'] * 1.2:  # 强空头
                trend_factor = 0.7  # 严格止损
            elif last_candle['plus_di'] > last_candle['minus_di'] * 1.2:  # 强多头
                trend_factor = 1.2  # 放宽止损
        
        # 成交量分析
        recent_volume_avg = dataframe['volume'].tail(10).mean()
        if len(dataframe) > 10 and last_candle['volume'] > recent_volume_avg * 2:
            volume_factor = 0.85  # 大量放大时紧缩止损
        
        # MACD背离检测
        macd_factor = 1.0
        if (current_profit > 0.01 and  # 在盈利状态下
            last_candle['macd'] < last_candle['macdsignal'] and
            current_rate > trade.open_rate * 1.015):  # MACD顶背离
            macd_factor = 0.8  # 紧缩止损保护利润
        
        # 综合计算动态止损
        dynamic_stop = -(atr_stop_distance * trend_factor * time_factor * volume_factor * macd_factor)
        
        # 确保止损在合理范围内
        dynamic_stop = max(dynamic_stop, -0.10)  # 最大止损10%
        dynamic_stop = min(dynamic_stop, -0.025)  # 最小止损2.5%
        
        # 紧急止损情况
        emergency_conditions = (
            (last_candle['bb_width'] > 0.25 and last_candle['atr_percent'] > 0.08) or  # 极高波动
            (last_candle['adx'] > 45 and last_candle['minus_di'] > last_candle['plus_di'] * 1.5) or  # 极强空头
            (last_candle['rsi'] < 20 and current_profit < -0.03)  # RSI极低且亏损
        )
        
        if emergency_conditions:
            return max(dynamic_stop * 0.6, -0.06)  # 紧急止损最多6%
        
        return dynamic_stop

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                           proposed_stake: float, min_stake: Optional[float], max_stake: Optional[float],
                           leverage: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        """
        动态仓位管理 - 根据市场条件调整仓位大小
        """
        # 获取当前数据
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) < 1:
            return proposed_stake
        
        last_candle = dataframe.iloc[-1].squeeze()
        
        # 基础仓位因子
        position_factor = self.position_size_factor.value
        
        # 波动率调整 - 高波动低仓位
        volatility_factor = 1.0
        if last_candle['atr_percent'] > 0.05:  # 高波动
            volatility_factor = 0.7  # 减少仓位
        elif last_candle['atr_percent'] < 0.02:  # 低波动
            volatility_factor = 1.2  # 增加仓位
        
        # 趋势强度调整
        trend_factor = 1.0
        if last_candle['adx'] > 35:  # 强趋势
            if last_candle['plus_di'] > last_candle['minus_di']:  # 上升趋势
                trend_factor = 1.1  # 略微增加
            else:  # 下降趋势
                trend_factor = 0.8  # 减少仓位
        
        # RSI调整 - 超卖区域增加仓位
        rsi_factor = 1.0
        if last_candle['rsi'] < 30:  # 深度超卖
            rsi_factor = 1.15
        elif last_candle['rsi'] < 35:
            rsi_factor = 1.05
        
        # 计算最终仓位
        final_stake = proposed_stake * position_factor * volatility_factor * trend_factor * rsi_factor
        
        # 确保在合理范围内
        if max_stake:
            final_stake = min(final_stake, max_stake)
        if min_stake:
            final_stake = max(final_stake, min_stake)
        
        # 防止仓位过小
        final_stake = max(final_stake, proposed_stake * 0.5)
        
        return final_stake
    
    def custom_exit(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs) -> Optional[Union[str, bool]]:
        """
        自定义退出逻辑 - 实现网格特有的退出策略
        """
        # 获取当前数据
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        
        # 网格利润目标检查 - 动态目标
        base_target = self.base_profit_percent.value
        
        # 根据市场条件动态调整目标利润
        volatility_factor = 1.0
        trend_factor = 1.0
        
        # 波动率调整
        if last_candle['atr_percent'] > 0.04:  # 高波动
            volatility_factor = 1.4  # 更高目标
        elif last_candle['atr_percent'] < 0.02:  # 低波动
            volatility_factor = 0.75  # 降低目标
        
        # 趋势强度调整
        if last_candle['adx'] > 35:  # 强趋势
            if last_candle['plus_di'] > last_candle['minus_di']:  # 上升趋势
                trend_factor = 0.9  # 降低目标，快速获利
            else:  # 下降趋势
                trend_factor = 1.3  # 提高目标，等待反弹
        
        target_profit = base_target * volatility_factor * trend_factor
        
        # 如果达到动态目标利润
        if current_profit >= target_profit:
            return "grid_profit_target"
        
        # 持仓时间过长的处理 - 更严格的时间管理
        holding_hours = (current_time - trade.open_date_utc).total_seconds() / 3600
        
        # 智能时间管理 - 根据市场状态动态调整
        # 获取市场波动率和趋势状态
        market_volatility = last_candle['atr_percent']
        trend_strength = last_candle['adx']
        
        # 高波动率环境下更短的持仓时间
        if market_volatility > 0.04:  # 高波动
            if holding_hours > 4 and current_profit > -0.015:  # 4小时后亏损<1.5%就退出
                return "grid_time_exit_high_vol"
            elif holding_hours > 6:  # 6小时后强制退出
                return "grid_force_exit_high_vol"
        
        # 中等波动率环境下的时间管理
        elif market_volatility > 0.02:
            if holding_hours > 6 and current_profit > -0.02:
                return "grid_time_exit_med_vol"
            elif holding_hours > 10:
                return "grid_force_exit_med_vol"
        
        # 低波动率环境下允许更长持仓
        else:
            if holding_hours > 10 and current_profit > -0.025:
                return "grid_time_exit_low_vol"
            elif holding_hours > 16:
                return "grid_force_exit_low_vol"
        
        # 强空头趋势下快速退出
        if (trend_strength > 30 and 
            last_candle['minus_di'] > last_candle['plus_di'] and
            holding_hours > 2 and current_profit < 0):
            return "grid_trend_exit"
        
        return None