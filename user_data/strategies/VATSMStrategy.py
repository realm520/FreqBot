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
    timeframe = '1h'
    
    # 策略参数 (针对1小时周期调整)
    lb_min = IntParameter(120, 480, default=240, space="buy", optimize=True)  # 最小回溯期 (10天*24小时)
    lb_max = IntParameter(720, 2160, default=1440, space="buy", optimize=True)  # 最大回溯期 (60天*24小时)
    vol_short_win = IntParameter(240, 720, default=480, space="buy", optimize=True)  # 短期波动率窗口 (20天*24小时)
    vol_long_win = IntParameter(960, 1920, default=1440, space="buy", optimize=True)  # 长期波动率窗口 (60天*24小时)
    ratio_cap = DecimalParameter(0.5, 0.95, default=0.9, space="buy", optimize=True)  # 波动率比率上限
    
    # 波动率目标参数
    vol_target = DecimalParameter(0.10, 0.30, default=0.20, space="buy", optimize=True)  # 目标波动率
    ewma_lambda = DecimalParameter(0.90, 0.98, default=0.94, space="buy", optimize=True)  # EWMA衰减因子
    min_forecast_vol = DecimalParameter(0.02, 0.08, default=0.05, space="buy", optimize=True)  # 最小预测波动率
    max_leverage = DecimalParameter(0.5, 2.0, default=1.0, space="buy", optimize=True)  # 最大杠杆
    
    startup_candle_count: int = 1500  # 约60天*24小时，适合3个月回测数据
    
    def __init__(self, config: dict):
        super().__init__(config)
        # 初始化波动率缓冲区
        self.short_vol_buffer = deque(maxlen=self.vol_short_win.value)
        self.long_vol_buffer = deque(maxlen=self.vol_long_win.value)
        self.ewma_var = None
        self.last_close = None
        
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
                    
                    # 生成信号
                    if momentum > 0:
                        dataframe.loc[dataframe.index[i], 'vatsm_signal'] = 1
                    elif momentum < 0:
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
            else:
                periods_per_year = 365  # 默认日线
                
            annual_vol = math.sqrt(max(ewma_var, 0.0)) * math.sqrt(periods_per_year)
            dataframe.loc[dataframe.index[i], 'ewma_vol'] = annual_vol
        
        # 计算风险敞口
        dataframe['vol_forecast'] = np.maximum(dataframe['ewma_vol'], self.min_forecast_vol.value)
        dataframe['raw_exposure'] = self.vol_target.value / dataframe['vol_forecast']
        dataframe['target_exposure'] = np.minimum(dataframe['raw_exposure'], self.max_leverage.value)
        dataframe['desired_exposure'] = dataframe['vatsm_signal'] * dataframe['target_exposure']
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        基于VATSM信号生成入场信号
        """
        # 多头入场：动量为正且风险敞口合理
        dataframe.loc[
            (
                (dataframe['vatsm_signal'] > 0) &
                (dataframe['target_exposure'] > 0.1) &  # 最小风险敞口要求
                (dataframe['volume'] > 0)
            ),
            'enter_long'
        ] = 1

        # 空头入场：动量为负且风险敞口合理
        dataframe.loc[
            (
                (dataframe['vatsm_signal'] < 0) &
                (dataframe['target_exposure'] > 0.1) &  # 最小风险敞口要求
                (dataframe['volume'] > 0)
            ),
            'enter_short'
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        基于VATSM信号生成出场信号
        """
        # 多头出场：信号转为负或中性
        dataframe.loc[
            (
                (dataframe['vatsm_signal'] <= 0) &
                (dataframe['volume'] > 0)
            ),
            'exit_long'
        ] = 1

        # 空头出场：信号转为正或中性
        dataframe.loc[
            (
                (dataframe['vatsm_signal'] >= 0) &
                (dataframe['volume'] > 0)
            ),
            'exit_short'
        ] = 1

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