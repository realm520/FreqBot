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
    VATSM策略 Phase 2B - 多重确认机制和多时间框架融合版本
    
    核心原理：
    1. 动态调整回溯期：高波动率时缩短回溯期，低波动率时延长回溯期
    2. 波动率目标定位：根据预测波动率调整仓位大小，维持恒定风险敞口
    3. 自适应动量计算：使用动态回溯期计算动量信号
    
    Phase 2B 新增特性：
    - 多重确认机制：RSI、成交量、市场结构等多维度信号过滤
    - 多时间框架融合：1小时趋势确认 + 5分钟精确入场
    - 升级风险管理：动态仓位调整、Kelly公式优化、回撤保护
    - 混合出场策略：追踪止损 + 信号出场 + 分批获利
    - 智能信号强度：基于确认数量和多时间框架一致性的动态加权
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
    
    # Phase 2B: 信号质量优化参数
    rsi_period = IntParameter(10, 20, default=14, space="buy", optimize=True)  # RSI周期
    rsi_oversold = IntParameter(20, 35, default=30, space="buy", optimize=True)  # RSI超卖阈值
    rsi_overbought = IntParameter(65, 80, default=70, space="buy", optimize=True)  # RSI超买阈值
    
    # 成交量确认参数
    volume_lookback = IntParameter(10, 30, default=20, space="buy", optimize=True)  # 成交量回溯期
    volume_threshold = DecimalParameter(1.2, 2.5, default=1.5, space="buy", optimize=True)  # 成交量倍数阈值
    
    # 市场结构识别参数
    swing_lookback = IntParameter(5, 15, default=10, space="buy", optimize=True)  # 摆动点回溯期
    breakout_threshold = DecimalParameter(0.005, 0.020, default=0.01, space="buy", optimize=True)  # 突破阈值
    
    # 多重确认机制参数
    min_confirmations = IntParameter(2, 4, default=3, space="buy", optimize=True)  # 最少确认信号数量
    
    # Phase 2B: 混合出场策略参数
    enable_signal_exit = BooleanParameter(default=True, space="sell", optimize=True)  # 启用信号出场
    enable_partial_exit = BooleanParameter(default=True, space="sell", optimize=True)  # 启用分批出场
    
    # Phase 2B: 多时间框架融合参数
    enable_multi_timeframe = BooleanParameter(default=True, space="buy", optimize=True)  # 启用多时间框架
    higher_timeframe = '1h'  # 高级时间框架 (趋势确认)
    lower_timeframe = '5m'   # 低级时间框架 (精确入场)
    
    # 多时间框架权重参数
    higher_tf_weight = DecimalParameter(0.3, 0.7, default=0.5, space="buy", optimize=True)  # 高级时间框架权重
    current_tf_weight = DecimalParameter(0.4, 0.8, default=0.6, space="buy", optimize=True)  # 当前时间框架权重
    lower_tf_weight = DecimalParameter(0.2, 0.5, default=0.3, space="buy", optimize=True)  # 低级时间框架权重
    
    # 分批获利参数
    partial_exit_1_profit = DecimalParameter(0.02, 0.05, default=0.03, space="sell", optimize=True)  # 第一次分批出场盈利
    partial_exit_1_ratio = DecimalParameter(0.2, 0.4, default=0.3, space="sell", optimize=True)  # 第一次出场比例
    partial_exit_2_profit = DecimalParameter(0.06, 0.12, default=0.08, space="sell", optimize=True)  # 第二次分批出场盈利
    partial_exit_2_ratio = DecimalParameter(0.3, 0.5, default=0.4, space="sell", optimize=True)  # 第二次出场比例
    
    # 反转信号出场参数
    reversal_rsi_threshold = IntParameter(75, 85, default=80, space="sell", optimize=True)  # 反转RSI阈值
    reversal_momentum_threshold = DecimalParameter(-0.015, -0.005, default=-0.01, space="sell", optimize=True)  # 反转动量阈值
    
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
    
    # Phase 2B: 风险管理系统升级参数
    max_daily_loss = DecimalParameter(0.02, 0.10, default=0.05, space="sell", optimize=True)  # 单日最大损失 (5%)
    max_weekly_loss = DecimalParameter(0.05, 0.20, default=0.10, space="sell", optimize=True)  # 单周最大损失 (10%)
    drawdown_pause_threshold = DecimalParameter(0.10, 0.25, default=0.15, space="sell", optimize=True)  # 回撤暂停阈值 (15%)
    
    # 动态仓位调整参数  
    volatility_position_factor = DecimalParameter(0.5, 2.0, default=1.0, space="buy", optimize=True)  # 波动率仓位因子
    performance_position_factor = DecimalParameter(0.3, 1.5, default=1.0, space="buy", optimize=True)  # 绩效仓位因子
    
    # Kelly公式仓位管理参数
    kelly_lookback = IntParameter(20, 100, default=50, space="buy", optimize=True)  # Kelly计算回溯期
    kelly_cap = DecimalParameter(0.1, 0.5, default=0.25, space="buy", optimize=True)  # Kelly比例上限
    
    startup_candle_count: int = 4000  # Phase 2B: 增加到4000支持多时间框架和更复杂指标
    
    def __init__(self, config: dict):
        super().__init__(config)
        # 初始化波动率缓冲区
        self.short_vol_buffer = deque(maxlen=self.vol_short_win.value)
        self.long_vol_buffer = deque(maxlen=self.vol_long_win.value)
        self.ewma_var = None
        self.last_close = None
        # 初始化追踪止盈变量
        self.custom_info = {}
        
        # Phase 2B: 风险管理系统初始化
        self.daily_pnl = deque(maxlen=7)  # 记录每日损盈
        self.trade_history = deque(maxlen=self.kelly_lookback.value)  # 交易历史
        self.current_drawdown = 0.0  # 当前回撤
        self.max_equity = 0.0  # 历史最大权益
        self.trading_paused = False  # 交易暂停状态
        self.last_daily_check = None  # 上次日度检查时间
        
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
        
        # Phase 2B: 多时间框架数据融合
        if self.enable_multi_timeframe.value:
            # 获取高级时间框架数据 (趋势确认)
            dataframe_higher = self.dp.get_pair_dataframe(metadata['pair'], self.higher_timeframe)
            if not dataframe_higher.empty:
                dataframe_higher['ema_21_higher'] = ta.EMA(dataframe_higher, timeperiod=21)
                dataframe_higher['ema_50_higher'] = ta.EMA(dataframe_higher, timeperiod=50)
                dataframe_higher['trend_higher'] = dataframe_higher['ema_21_higher'] > dataframe_higher['ema_50_higher']
                dataframe_higher['adx_higher'] = ta.ADX(dataframe_higher, timeperiod=14)
                
                # 合并高级时间框架数据
                dataframe = merge_informative_pair(dataframe, dataframe_higher, self.timeframe, 
                                                 self.higher_timeframe, ffill=True)
            
            # 获取低级时间框架数据 (精确入场)
            try:
                dataframe_lower = self.dp.get_pair_dataframe(metadata['pair'], self.lower_timeframe)
                if not dataframe_lower.empty:
                    dataframe_lower['rsi_lower'] = ta.RSI(dataframe_lower, timeperiod=7)
                    dataframe_lower['volume_sma_lower'] = dataframe_lower['volume'].rolling(10).mean()
                    dataframe_lower['volume_surge_lower'] = dataframe_lower['volume'] > dataframe_lower['volume_sma_lower'] * 1.5
                    
                    # 合并低级时间框架数据
                    dataframe = merge_informative_pair(dataframe, dataframe_lower, self.timeframe,
                                                     self.lower_timeframe, ffill=True)
            except:
                pass  # 如果无法获取低级数据，则继续
        
        # Phase 2B: 信号质量优化指标
        # 1. RSI超买超卖过滤
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.rsi_period.value)
        dataframe['rsi_oversold'] = dataframe['rsi'] < self.rsi_oversold.value
        dataframe['rsi_overbought'] = dataframe['rsi'] > self.rsi_overbought.value
        dataframe['rsi_neutral'] = (~dataframe['rsi_oversold']) & (~dataframe['rsi_overbought'])
        
        # 2. 成交量确认机制
        dataframe['volume_ma'] = dataframe['volume'].rolling(self.volume_lookback.value).mean()
        dataframe['volume_above_avg'] = dataframe['volume'] > (dataframe['volume_ma'] * self.volume_threshold.value)
        
        # 3. 市场结构识别 - 摆动高低点
        dataframe['swing_high'] = dataframe['high'].rolling(self.swing_lookback.value * 2 + 1, center=True).max() == dataframe['high']
        dataframe['swing_low'] = dataframe['low'].rolling(self.swing_lookback.value * 2 + 1, center=True).min() == dataframe['low']
        
        # 4. 价格突破确认
        dataframe['price_change'] = dataframe['close'].pct_change()
        dataframe['significant_move'] = np.abs(dataframe['price_change']) > self.breakout_threshold.value
        
        # 5. 趋势一致性检查
        dataframe['price_above_ema_fast'] = dataframe['close'] > dataframe['ema_fast']
        dataframe['price_above_ema_slow'] = dataframe['close'] > dataframe['ema_slow']
        dataframe['emas_aligned'] = dataframe['ema_fast'] > dataframe['ema_slow']
        
        # 动量计算 - 使用动态回溯期
        dataframe['momentum'] = 0.0
        dataframe['vatsm_signal'] = 0.0
        dataframe['signal_confirmations'] = 0  # Phase 2B: 信号确认计数
        
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
                    
                    # Phase 2B: 多重确认机制和多时间框架信号生成
                    confirmations = 0
                    
                    # 基础确认: 动量和ADX
                    if abs(momentum) > self.min_momentum_threshold.value and dataframe['adx_strong'].iloc[i]:
                        confirmations += 1
                    
                    # EMA趋势确认
                    if momentum > 0 and dataframe['trend_up'].iloc[i] and dataframe['price_above_ema_fast'].iloc[i]:
                        confirmations += 1
                    elif momentum < 0 and not dataframe['trend_up'].iloc[i] and not dataframe['price_above_ema_fast'].iloc[i]:
                        confirmations += 1
                    
                    # RSI过滤确认 (多头时避免超买，空头时避免超卖)
                    if momentum > 0 and not dataframe['rsi_overbought'].iloc[i]:
                        confirmations += 1
                    elif momentum < 0 and not dataframe['rsi_oversold'].iloc[i]:
                        confirmations += 1
                    
                    # 成交量确认
                    if dataframe['volume_above_avg'].iloc[i]:
                        confirmations += 1
                    
                    # 价格动作确认
                    if dataframe['significant_move'].iloc[i]:
                        confirmations += 1
                    
                    # Phase 2B: 多时间框架确认
                    if self.enable_multi_timeframe.value:
                        # 高级时间框架趋势确认
                        try:
                            if (momentum > 0 and 'trend_higher_1h' in dataframe.columns and 
                                dataframe['trend_higher_1h'].iloc[i] and 
                                dataframe['adx_higher_1h'].iloc[i] > 25):
                                confirmations += 1
                            elif (momentum < 0 and 'trend_higher_1h' in dataframe.columns and 
                                  not dataframe['trend_higher_1h'].iloc[i] and 
                                  dataframe['adx_higher_1h'].iloc[i] > 25):
                                confirmations += 1
                        except:
                            pass
                            
                        # 低级时间框架动量确认
                        try:
                            if ('rsi_lower_5m' in dataframe.columns and 
                                'volume_surge_lower_5m' in dataframe.columns):
                                if (momentum > 0 and dataframe['rsi_lower_5m'].iloc[i] > 50 and 
                                    dataframe['volume_surge_lower_5m'].iloc[i]):
                                    confirmations += 1
                                elif (momentum < 0 and dataframe['rsi_lower_5m'].iloc[i] < 50 and 
                                      dataframe['volume_surge_lower_5m'].iloc[i]):
                                    confirmations += 1
                        except:
                            pass
                    
                    # 记录确认数量
                    dataframe.loc[dataframe.index[i], 'signal_confirmations'] = confirmations
                    
                    # 多时间框架加权信号强度计算
                    base_strength = 1.0
                    if self.enable_multi_timeframe.value and confirmations >= self.min_confirmations.value:
                        try:
                            # 高级时间框架加权
                            if 'trend_higher_1h' in dataframe.columns:
                                if momentum > 0 and dataframe['trend_higher_1h'].iloc[i]:
                                    base_strength *= (1 + self.higher_tf_weight.value)
                                elif momentum < 0 and not dataframe['trend_higher_1h'].iloc[i]:
                                    base_strength *= (1 + self.higher_tf_weight.value)
                            
                            # 低级时间框架加权
                            if 'volume_surge_lower_5m' in dataframe.columns and dataframe['volume_surge_lower_5m'].iloc[i]:
                                base_strength *= (1 + self.lower_tf_weight.value)
                        except:
                            pass
                    
                    # 只有达到最小确认数量才生成信号
                    if confirmations >= self.min_confirmations.value:
                        if momentum > 0:  # 多头信号
                            dataframe.loc[dataframe.index[i], 'vatsm_signal'] = base_strength
                        elif momentum < 0:  # 空头信号(现货不做空)
                            dataframe.loc[dataframe.index[i], 'vatsm_signal'] = -base_strength
        
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
        
        # Phase 2B: 动量强度分级 (在momentum计算后进行)
        dataframe['momentum_weak'] = (np.abs(dataframe['momentum']) > self.min_momentum_threshold.value) & (np.abs(dataframe['momentum']) <= 2 * self.min_momentum_threshold.value)
        dataframe['momentum_medium'] = (np.abs(dataframe['momentum']) > 2 * self.min_momentum_threshold.value) & (np.abs(dataframe['momentum']) <= 4 * self.min_momentum_threshold.value)
        dataframe['momentum_strong'] = np.abs(dataframe['momentum']) > 4 * self.min_momentum_threshold.value
        
        # Phase 2B: 根据多重因素调整信号强度
        # 基于动量强度和确认数量的综合评分
        dataframe['momentum_score'] = np.where(
            dataframe['momentum_strong'], 1.0,
            np.where(dataframe['momentum_medium'], 0.8, 0.6)
        )
        dataframe['confirmation_score'] = np.minimum(dataframe['signal_confirmations'] / self.min_confirmations.value, 1.5)
        dataframe['signal_strength'] = dataframe['momentum_score'] * dataframe['confirmation_score']
        
        dataframe['adjusted_exposure'] = dataframe['target_exposure'] * dataframe['signal_strength']
        dataframe['desired_exposure'] = dataframe['vatsm_signal'] * dataframe['adjusted_exposure']
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        基于Phase 2B多重确认机制的优化入场信号生成
        """
        # Phase 2B: 多重确认机制的多头入场条件
        dataframe.loc[
            (
                (dataframe['vatsm_signal'] > 0) &  # VATSM信号为正
                (dataframe['signal_confirmations'] >= self.min_confirmations.value) &  # 达到最小确认数
                (dataframe['trend_up'] == True) &  # EMA趋势向上
                (dataframe['adx_strong'] == True) &  # 趋势强度足够
                (dataframe['momentum'] > self.min_momentum_threshold.value * 1.5) &  # 更高动量要求
                (dataframe['adjusted_exposure'] > 0.15) &  # 最小风险敞口要求
                (dataframe['volume'] > 0) &  # 基础成交量检查
                (dataframe['volume_above_avg'] == True) &  # 成交量高于平均值
                (~dataframe['rsi_overbought']) &  # 避免超买入场
                (dataframe['price_above_ema_fast'] == True) &  # 价格在快速EMA之上
                (dataframe['emas_aligned'] == True) &  # EMA序列正确
                # 动量强度要求：至少中等动量
                (dataframe['momentum_medium'] | dataframe['momentum_strong'])
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
        Phase 2B: 混合出场策略 - 结合信号出场和追踪止损
        
        实现多元化出场机制：
        1. 反转信号出场 - 及时捉捕趋势反转
        2. 技术指标出场 - RSI过度超买等
        3. 动量衰竭出场 - 动量信号变弱
        4. 保留追踪止损作为最后保障
        """
        # Phase 2B: 有条件的信号出场逻辑
        if self.enable_signal_exit.value:
            # 1. 反转信号出场：强势反转信号
            dataframe.loc[
                (
                    # 多头位反转条件
                    (dataframe['vatsm_signal'] < 0) &  # VATSM信号转空
                    (dataframe['trend_up'] == False) &  # EMA趋势向下
                    (dataframe['rsi'] > self.reversal_rsi_threshold.value) &  # RSI过度超买
                    (dataframe['momentum'] < self.reversal_momentum_threshold.value) &  # 负动量出现
                    (dataframe['adx_strong'] == True) &  # 趋势强度足够
                    (dataframe['volume_above_avg'] == True)  # 成交量确认
                ),
                'exit_long'
            ] = 1
        
        return dataframe
    
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                   current_profit: float, **kwargs) -> Union[str, None]:
        """
        Phase 2B: 分批获利机制
        
        在不同盈利水平下分批平仓，平衡风险与收益
        """
        if not self.enable_partial_exit.value:
            return None
            
        trade_id = f"{pair}_{trade.open_date}"
        
        # 初始化分批出场记录
        if trade_id not in self.custom_info:
            self.custom_info[trade_id] = {
                'max_profit': current_profit,
                'trailing_activated': False,
                'tier_activated': 0,
                'entry_time': current_time,
                'partial_exit_1_done': False,
                'partial_exit_2_done': False
            }
        
        trade_info = self.custom_info[trade_id]
        
        # 第一次分批出场：达到第一目标盈利
        if (not trade_info['partial_exit_1_done'] and 
            current_profit > self.partial_exit_1_profit.value):
            trade_info['partial_exit_1_done'] = True
            return f"partial_exit_1_{self.partial_exit_1_ratio.value}"
            
        # 第二次分批出场：达到第二目标盈利
        if (trade_info['partial_exit_1_done'] and 
            not trade_info['partial_exit_2_done'] and 
            current_profit > self.partial_exit_2_profit.value):
            trade_info['partial_exit_2_done'] = True
            return f"partial_exit_2_{self.partial_exit_2_ratio.value}"
        
        return None
    
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                          proposed_stake: float, min_stake: Optional[float], max_stake: float,
                          leverage: float, entry_tag: Optional[str], side: str,
                          **kwargs) -> float:
        """
        Phase 2B: 升级版动态仓位管理系统
        结合波动率目标、风险管理和Kelly公式
        """
        # Phase 2B: 风险管理检查
        if self.trading_paused:
            return 0  # 交易暂停时不开仓
        
        # 每日风险检查
        if not self._check_daily_risk_limits(current_time):
            return 0  # 超过日损失限制
            
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or len(dataframe) == 0:
            return proposed_stake
            
        latest = dataframe.iloc[-1]
        desired_exposure = latest['desired_exposure']
        
        if abs(desired_exposure) < 0.01:  # 避免过小的仓位
            return min_stake or 0
            
        # 计算基础仓位
        account_balance = self.wallets.get_total_stake_amount()
        base_stake = abs(desired_exposure) * account_balance
        
        # Phase 2B: 动态仓位调整因子
        volatility_factor = self._calculate_volatility_factor(latest)
        performance_factor = self._calculate_performance_factor()
        kelly_factor = self._calculate_kelly_factor()
        
        # 综合调整因子
        total_factor = volatility_factor * performance_factor * kelly_factor
        adjusted_stake = base_stake * total_factor
        
        # 确保在允许范围内
        if min_stake:
            adjusted_stake = max(adjusted_stake, min_stake)
        adjusted_stake = min(adjusted_stake, max_stake)
        
        return adjusted_stake
    
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
        
        # 初始化追踪信息 (统一初始化所有字段)
        if trade_id not in self.custom_info:
            self.custom_info[trade_id] = {
                'max_profit': current_profit,
                'trailing_activated': False,
                'tier_activated': 0,
                'entry_time': current_time,
                'partial_exit_1_done': False,
                'partial_exit_2_done': False
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
    
    # Phase 2B: 风险管理辅助方法
    def _check_daily_risk_limits(self, current_time: datetime) -> bool:
        """检查日度和周度损失限制"""
        try:
            if self.last_daily_check is None or current_time.date() != self.last_daily_check:
                self.last_daily_check = current_time.date()
                # 更新日度PnL记录
                today_pnl = self._calculate_daily_pnl()
                self.daily_pnl.append(today_pnl)
                
            # 检查单日损失
            if len(self.daily_pnl) > 0 and self.daily_pnl[-1] < -self.max_daily_loss.value:
                return False
                
            # 检查周损失
            weekly_pnl = sum(list(self.daily_pnl)[-7:])  # 过去7天
            if weekly_pnl < -self.max_weekly_loss.value:
                return False
                
            return True
        except:
            return True  # 出错时允许交易
    
    def _calculate_daily_pnl(self) -> float:
        """计算当日损盈"""
        try:
            # 这里需要实际的交易记录，简化处理
            return 0.0  # 实际应用中需要连接交易数据
        except:
            return 0.0
    
    def _calculate_volatility_factor(self, latest_data) -> float:
        """根据市场波动性调整仓位"""
        try:
            current_vol = latest_data['ewma_vol']
            # 高波动性时减少仓位，低波动性时增加仓位
            if current_vol > 0.3:  # 高波动
                return 0.7 * self.volatility_position_factor.value
            elif current_vol < 0.1:  # 低波动
                return 1.3 * self.volatility_position_factor.value
            else:
                return 1.0 * self.volatility_position_factor.value
        except:
            return 1.0
    
    def _calculate_performance_factor(self) -> float:
        """根据近期绩效调整仓位"""
        try:
            if len(self.trade_history) < 10:
                return 1.0
                
            # 计算近期胜率
            recent_wins = sum(1 for trade in list(self.trade_history)[-20:] if trade > 0)
            win_rate = recent_wins / min(20, len(self.trade_history))
            
            if win_rate > 0.6:
                return 1.2 * self.performance_position_factor.value
            elif win_rate < 0.4:
                return 0.8 * self.performance_position_factor.value
            else:
                return 1.0 * self.performance_position_factor.value
        except:
            return 1.0
    
    def _calculate_kelly_factor(self) -> float:
        """计算Kelly公式仓位比例"""
        try:
            if len(self.trade_history) < self.kelly_lookback.value:
                return 1.0
                
            trades = list(self.trade_history)
            wins = [t for t in trades if t > 0]
            losses = [t for t in trades if t < 0]
            
            if len(wins) == 0 or len(losses) == 0:
                return 1.0
                
            win_rate = len(wins) / len(trades)
            avg_win = sum(wins) / len(wins)
            avg_loss = abs(sum(losses) / len(losses))
            
            if avg_loss == 0:
                return 1.0
                
            kelly_ratio = win_rate - (1 - win_rate) * (avg_win / avg_loss)
            kelly_ratio = max(0, min(kelly_ratio, self.kelly_cap.value))
            
            return 1.0 + kelly_ratio  # Kelly作为加权因子
        except:
            return 1.0