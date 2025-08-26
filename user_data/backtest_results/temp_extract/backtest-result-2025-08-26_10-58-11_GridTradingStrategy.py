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
    base_profit_percent = DecimalParameter(low=0.008, high=0.025, default=0.015, space="sell", optimize=True, load=True)

    # 动态ROI配置 - 更加激进的网格卖出
    minimal_roi = {
        "0": 0.015,   # 1.5%利润时退出
        "15": 0.012,  # 15分钟后1.2%利润退出
        "30": 0.01,   # 30分钟后1%利润退出
        "60": 0.008,  # 60分钟后0.8%利润退出
        "120": 0.005, # 120分钟后0.5%利润退出
    }

    # 止损设置 - 更紧的止损控制风险
    stoploss = -0.08  # 8%止损保护

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

        # ATR - 用于计算网格间距
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)

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
        dataframe.loc[
            (
                # 主要条件：价格相对位置 - 更灵活的触发
                (
                    (dataframe['close'] < dataframe['bb_middleband'])  # 价格低于布林中轴
                    |
                    (dataframe['close'] < dataframe['sma_20'] * 0.998)  # 或价格略低于20日均线
                )
                &
                # RSI超卖条件 - 放宽条件增加交易频率
                (dataframe['rsi'] < 45)  # RSI小于45
                &
                # 趋势过滤 - 简化趋势判断
                (dataframe['ema_12'] > dataframe['ema_26'])  # EMA多头排列
                &
                # 布林带位置 - 在布林带下半部分
                (dataframe['bb_percent'] < 0.6)
                &
                # 波动率控制 - 放宽条件
                (dataframe['bb_width'] > 0.015)  # 布林带宽度大于1.5%
                &
                (dataframe['bb_width'] < 0.25)   # 布林带宽度小于25%
                &
                # ATR条件 - 确保有足够的波动
                (dataframe['atr'] > dataframe['close'] * 0.005)  # ATR大于价格的0.5%
                &
                # 确保有交易量
                (dataframe['volume'] > dataframe['volume'].rolling(10).mean() * 0.5)
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
        dataframe.loc[
            (
                # 主要获利条件 - 价格上涨到布林带上半部分
                (dataframe['close'] > dataframe['bb_middleband'])
                &
                (dataframe['bb_percent'] > 0.7)  # 在布林带70%以上位置
                &
                # RSI显示超买倾向
                (dataframe['rsi'] > 60)
                &
                # 确保有交易量支持
                (dataframe['volume'] > dataframe['volume'].rolling(10).mean() * 0.8)
            )
            |
            (
                # 快速获利条件 - RSI明显超买
                (dataframe['rsi'] > 70)
                &
                (dataframe['close'] > dataframe['sma_20'])  # 价格高于20日均线
                &
                (dataframe['volume'] > 0)
            )
            |
            (
                # 风险控制退出 - 市场环境恶化
                (dataframe['bb_width'] > 0.20)  # 波动率过高
                &
                (dataframe['rsi'] > 55)  # 中等偏高RSI
                &
                (dataframe['ema_12'] < dataframe['ema_26'])  # EMA转为空头排列
            ),
            'exit_long',
        ] = 1

        return dataframe

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        自定义止损 - 网格策略的动态止损
        """
        # 网格策略通常不使用严格止损，但可以设置安全止损
        if current_profit < -0.15:  # 如果亏损超过15%
            return 0.01  # 触发止损
        
        return -1  # 不触发止损

    def custom_exit(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs) -> Optional[Union[str, bool]]:
        """
        自定义退出逻辑 - 实现网格特有的退出策略
        """
        # 获取当前数据
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        
        # 网格利润目标检查
        target_profit = self.base_profit_percent.value
        
        # 如果达到网格目标利润
        if current_profit >= target_profit:
            return "grid_profit_target"
        
        # 持仓时间过长的处理（防止资金占用过久）
        if (current_time - trade.open_date_utc).seconds > 4 * 3600:  # 4小时
            if current_profit > 0.005:  # 有微小盈利就退出
                return "grid_time_exit"
        
        return None