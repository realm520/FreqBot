# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Optional, Union, Dict, Any

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

# Import the base strategy
from .GridTradingStrategy import GridTradingStrategy


class EnhancedGridStrategy(GridTradingStrategy):
    """
    增强网格交易策略 - 基于GridTradingStrategy扩展的动态网格策略
    
    策略增强：
    1. 动态网格调整 - 根据市场波动率和趋势强度调整网格间距
    2. 市场状态识别 - 自动识别震荡、趋势、横盘等市场状态
    3. 智能仓位管理 - 根据市场条件和风险评估动态调整仓位大小
    4. 多时间框架分析 - 结合多个时间维度进行决策
    5. 风险控制增强 - 更精准的止损和风险管理机制
    
    适用场景：
    - 中长期震荡行情
    - 具有一定波动率的市场
    - 需要精细化仓位管理的交易环境
    """
    
    # 策略元数据
    STRATEGY_NAME = "EnhancedGridStrategy"
    STRATEGY_VERSION = "1.0.0"
    STRATEGY_AUTHOR = "FreqBot Team"
    STRATEGY_CATEGORY = "enhanced_grid_trading"
    STRATEGY_DESCRIPTION = "增强网格交易策略 - 具备动态调整和智能识别能力的网格策略"

    # Strategy interface version - 继承父类版本
    INTERFACE_VERSION = 3

    # Can this strategy go short? - 保持与父类一致
    can_short: bool = False

    # ===========================================
    # 增强功能控制参数
    # ===========================================
    
    # 动态网格功能开关
    enable_dynamic_grid = BooleanParameter(
        default=True, 
        space="buy", 
        optimize=False, 
        load=True,
        description="启用动态网格间距调整"
    )
    
    # 市场状态识别功能开关
    enable_market_regime_detection = BooleanParameter(
        default=True, 
        space="buy", 
        optimize=False, 
        load=True,
        description="启用市场状态自动识别"
    )
    
    # 增强仓位管理功能开关
    enable_enhanced_position_sizing = BooleanParameter(
        default=True, 
        space="buy", 
        optimize=False, 
        load=True,
        description="启用智能仓位管理"
    )
    
    # 多时间框架分析开关
    enable_multi_timeframe_analysis = BooleanParameter(
        default=False, 
        space="buy", 
        optimize=False, 
        load=True,
        description="启用多时间框架分析（实验功能）"
    )

    # ===========================================
    # 动态网格参数
    # ===========================================
    
    # 动态网格调整敏感度
    grid_adjustment_sensitivity = DecimalParameter(
        low=0.3, high=1.5, default=0.8, 
        space="buy", optimize=True, load=True,
        description="网格动态调整的敏感度，值越大调整越频繁"
    )
    
    # 最小网格间距百分比
    min_grid_spacing_pct = DecimalParameter(
        low=0.005, high=0.03, default=0.01, 
        space="buy", optimize=True, load=True,
        description="动态网格的最小间距百分比"
    )
    
    # 最大网格间距百分比  
    max_grid_spacing_pct = DecimalParameter(
        low=0.04, high=0.15, default=0.08, 
        space="buy", optimize=True, load=True,
        description="动态网格的最大间距百分比"
    )

    # ===========================================
    # 市场状态识别参数
    # ===========================================
    
    # ADX趋势识别阈值
    adx_consolidation_threshold = DecimalParameter(
        low=20, high=30, default=25, 
        space="buy", optimize=True, load=True,
        description="ADX震荡市识别阈值 (<25为震荡)"
    )
    
    adx_trend_threshold = DecimalParameter(
        low=35, high=50, default=40, 
        space="buy", optimize=True, load=True,
        description="ADX趋势市识别阈值 (>40为强趋势)"
    )
    
    # 状态平滑参数
    regime_smoothing_period = IntParameter(
        low=3, high=10, default=5, 
        space="buy", optimize=False, load=True,
        description="市场状态平滑周期，避免频繁切换"
    )
    
    # 趋势方向确认阈值
    di_ratio_threshold = DecimalParameter(
        low=1.1, high=1.5, default=1.2, 
        space="buy", optimize=True, load=True,
        description="DI指标比例阈值，确认趋势方向"
    )
    
    # 震荡市场识别参数
    consolidation_detection_period = IntParameter(
        low=10, high=30, default=20, 
        space="buy", optimize=True, load=True,
        description="震荡市场识别的周期参数"
    )

    # ===========================================
    # 风险控制增强参数
    # ===========================================
    
    # 最大同时开仓数量限制
    max_concurrent_positions = IntParameter(
        low=1, high=5, default=3, 
        space="buy", optimize=False, load=True,
        description="最大同时持有仓位数量"
    )
    
    # 三层止损系统参数
    tier1_profit = DecimalParameter(
        low=0.005, high=0.015, default=0.008, 
        space="sell", optimize=True, load=True,
        description="第一层启动盈利阈值 (0.8%)"
    )
    
    tier1_distance = DecimalParameter(
        low=0.003, high=0.010, default=0.005, 
        space="sell", optimize=True, load=True,
        description="第一层追踪距离 (0.5%)"
    )
    
    tier2_profit = DecimalParameter(
        low=0.015, high=0.030, default=0.020, 
        space="sell", optimize=True, load=True,
        description="第二层启动盈利阈值 (2%)"
    )
    
    tier2_distance = DecimalParameter(
        low=0.008, high=0.025, default=0.015, 
        space="sell", optimize=True, load=True,
        description="第二层追踪距离 (1.5%)"
    )
    
    tier3_profit = DecimalParameter(
        low=0.035, high=0.070, default=0.050, 
        space="sell", optimize=True, load=True,
        description="第三层启动盈利阈值 (5%)"
    )
    
    tier3_distance = DecimalParameter(
        low=0.020, high=0.050, default=0.030, 
        space="sell", optimize=True, load=True,
        description="第三层追踪距离 (3%)"
    )
    
    # 时间止损参数
    max_hold_hours = IntParameter(
        low=12, high=72, default=48, 
        space="sell", optimize=True, load=True,
        description="最大持仓时间（小时）"
    )
    
    time_exit_profit_threshold = DecimalParameter(
        low=-0.10, high=0.05, default=0.0, 
        space="sell", optimize=True, load=True,
        description="时间止损盈利阈值"
    )
    
    # 风险管理系统参数
    max_daily_loss = DecimalParameter(
        low=0.02, high=0.10, default=0.05, 
        space="sell", optimize=True, load=True,
        description="单日最大损失限制 (5%)"
    )
    
    max_weekly_loss = DecimalParameter(
        low=0.05, high=0.20, default=0.10, 
        space="sell", optimize=True, load=True,
        description="单周最大损失限制 (10%)"
    )
    
    drawdown_pause_threshold = DecimalParameter(
        low=0.10, high=0.25, default=0.15, 
        space="sell", optimize=True, load=True,
        description="回撤暂停交易阈值 (15%)"
    )
    
    consecutive_loss_limit = IntParameter(
        low=3, high=10, default=5, 
        space="sell", optimize=False, load=True,
        description="连续亏损熔断限制"
    )
    
    # 动态止损增强系数（保留原有参数）
    enhanced_stoploss_factor = DecimalParameter(
        low=0.8, high=1.5, default=1.1, 
        space="sell", optimize=True, load=True,
        description="增强止损系数，影响止损敏感度"
    )

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        初始化增强网格策略
        
        Args:
            config: FreqTrade配置字典
        """
        # 调用父类初始化
        super().__init__(config)
        
        # 初始化增强功能状态
        self._market_regime: str = "unknown"
        self._grid_adjustment_factor: float = 1.0
        self._enhanced_initialized: bool = False
        
        # 缓存变量
        self._indicator_cache: Dict[str, Any] = {}
        self._last_update_time: Optional[datetime] = None
        
        # 市场状态历史记录
        self._regime_history: list = []  # 存储历史状态用于分析
        self._regime_change_count: int = 0  # 状态切换计数
        
        # 风险控制系统初始化
        from collections import deque
        self.custom_trade_info: Dict[str, Any] = {}  # 交易追踪信息
        self.daily_pnl = deque(maxlen=7)  # 每日损盈记录
        self.trade_history = deque(maxlen=100)  # 交易历史
        self.consecutive_losses = 0  # 连续亏损计数
        self.current_drawdown = 0.0  # 当前回撤
        self.max_equity = 0.0  # 历史最大权益
        self.trading_paused = False  # 交易暂停状态
        self.last_daily_check: Optional[datetime] = None  # 上次日度检查时间
        self.emergency_exit_triggered = False  # 紧急退出标志
        
        # 日志记录
        self.logger.info(f"增强网格策略 {self.STRATEGY_VERSION} 初始化完成")
        self.logger.info(f"动态网格: {self.enable_dynamic_grid.value}")
        self.logger.info(f"市场状态识别: {self.enable_market_regime_detection.value}")
        self.logger.info(f"增强仓位管理: {self.enable_enhanced_position_sizing.value}")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        增强版技术指标计算
        
        在父类指标基础上添加增强功能所需的额外指标
        
        Args:
            dataframe: K线数据
            metadata: 交易对元数据
            
        Returns:
            DataFrame: 包含所有指标的数据框
        """
        # 首先调用父类的指标计算
        dataframe = super().populate_indicators(dataframe, metadata)
        
        # 添加增强功能所需的额外指标
        dataframe = self._populate_enhanced_indicators(dataframe, metadata)
        
        # 如果启用动态网格，计算动态网格指标
        if self.enable_dynamic_grid.value:
            dataframe = self._calculate_dynamic_grid_indicators(dataframe)
            # 计算动态网格层级
            dataframe = self.calculate_dynamic_grid_levels(dataframe)
        
        # 如果启用市场状态识别，添加状态识别指标
        if self.enable_market_regime_detection.value:
            dataframe = self._calculate_market_regime_indicators(dataframe)
            # 执行市场状态识别和平滑处理
            dataframe = self._apply_market_regime_detection(dataframe)
        
        return dataframe

    def _populate_enhanced_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        添加增强功能所需的技术指标
        
        Args:
            dataframe: K线数据
            metadata: 交易对元数据
            
        Returns:
            DataFrame: 包含增强指标的数据框
        """
        # 高级波动率指标
        dataframe['realized_volatility'] = dataframe['close'].pct_change().rolling(window=20).std() * np.sqrt(20)
        dataframe['volatility_ratio'] = dataframe['realized_volatility'] / dataframe['realized_volatility'].rolling(window=50).mean()
        
        # 价格动量指标
        dataframe['momentum_5'] = ta.MOM(dataframe, timeperiod=5)
        dataframe['momentum_10'] = ta.MOM(dataframe, timeperiod=10)
        dataframe['momentum_20'] = ta.MOM(dataframe, timeperiod=20)
        
        # 成交量相关指标
        dataframe['volume_sma'] = ta.SMA(dataframe['volume'], timeperiod=20)
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_sma']
        dataframe['price_volume_trend'] = ta.PVT(dataframe)
        
        # Stochastic指标
        stoch = ta.STOCH(dataframe)
        dataframe['stoch_k'] = stoch['slowk']
        dataframe['stoch_d'] = stoch['slowd']
        
        # CCI指标
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=20)
        
        # Williams %R
        dataframe['williams_r'] = ta.WILLR(dataframe, timeperiod=14)
        
        # 支撑阻力位指标
        dataframe = self._calculate_support_resistance(dataframe)
        
        return dataframe

    def _calculate_dynamic_grid_indicators(self, dataframe: DataFrame) -> DataFrame:
        """
        计算动态网格相关指标
        
        Args:
            dataframe: K线数据
            
        Returns:
            DataFrame: 包含动态网格指标的数据框
        """
        # 动态网格间距计算
        base_spacing = dataframe['atr_percent']
        volatility_adj = dataframe['volatility_ratio'].clip(0.5, 2.0)  # 限制调整范围
        
        dataframe['dynamic_grid_spacing'] = (
            base_spacing * volatility_adj * self.grid_adjustment_sensitivity.value
        ).clip(
            self.min_grid_spacing_pct.value, 
            self.max_grid_spacing_pct.value
        )
        
        # 动态网格级数调整
        dataframe['dynamic_grid_levels'] = np.where(
            dataframe['volatility_ratio'] > 1.2,
            (self.grid_levels.value * 0.8).astype(int),  # 高波动时减少级数
            np.where(
                dataframe['volatility_ratio'] < 0.8,
                (self.grid_levels.value * 1.2).astype(int),  # 低波动时增加级数
                self.grid_levels.value
            )
        )
        
        return dataframe

    def _calculate_market_regime_indicators(self, dataframe: DataFrame) -> DataFrame:
        """
        计算市场状态识别相关指标
        
        Args:
            dataframe: K线数据
            
        Returns:
            DataFrame: 包含市场状态指标的数据框
        """
        # 趋势强度评分
        dataframe['trend_strength'] = np.where(
            dataframe['adx'] > self.trend_detection_threshold.value,
            np.where(
                dataframe['plus_di'] > dataframe['minus_di'],
                1,  # 上升趋势
                -1  # 下降趋势
            ),
            0  # 震荡
        )
        
        # 震荡市场识别
        price_range = dataframe['high'].rolling(window=self.consolidation_detection_period.value).max() - \
                     dataframe['low'].rolling(window=self.consolidation_detection_period.value).min()
        avg_true_range = dataframe['atr'].rolling(window=self.consolidation_detection_period.value).mean()
        
        dataframe['consolidation_score'] = 1 - (price_range / (avg_true_range * self.consolidation_detection_period.value))
        dataframe['is_consolidating'] = dataframe['consolidation_score'] > 0.7
        
        # 市场状态综合评分
        dataframe['market_regime'] = np.select(
            [
                (dataframe['trend_strength'] == 1) & (dataframe['adx'] > self.trend_detection_threshold.value),
                (dataframe['trend_strength'] == -1) & (dataframe['adx'] > self.trend_detection_threshold.value),
                dataframe['is_consolidating'],
                True
            ],
            ['uptrend', 'downtrend', 'consolidation', 'neutral'],
            default='neutral'
        )
        
        return dataframe

    def _calculate_support_resistance(self, dataframe: DataFrame) -> DataFrame:
        """
        计算支撑阻力位
        
        Args:
            dataframe: K线数据
            
        Returns:
            DataFrame: 包含支撑阻力位的数据框
        """
        # 简单的支撑阻力位计算
        window = 20
        dataframe['resistance'] = dataframe['high'].rolling(window=window).max()
        dataframe['support'] = dataframe['low'].rolling(window=window).min()
        
        # 价格相对于支撑阻力位的位置
        dataframe['resistance_distance'] = (dataframe['resistance'] - dataframe['close']) / dataframe['close']
        dataframe['support_distance'] = (dataframe['close'] - dataframe['support']) / dataframe['close']
        
        return dataframe

    # ===========================================
    # 动态网格核心算法实现
    # ===========================================
    
    def _calculate_adaptive_grid_spacing(self, dataframe: DataFrame) -> float:
        """
        基于ATR计算自适应网格间距
        
        算法：
        1. 获取当前ATR值和ATR百分比
        2. 基于波动率调整间距系数
        3. 应用最小/最大间距限制
        4. 考虑网格调整敏感度
        
        Args:
            dataframe: K线数据
            
        Returns:
            float: 自适应网格间距（价格百分比）
        """
        try:
            # 获取最新ATR相关指标
            current_atr_pct = dataframe['atr_percent'].iloc[-1]
            current_price = dataframe['close'].iloc[-1]
            
            # 基础间距 = ATR百分比 * 调整系数
            base_spacing_pct = current_atr_pct * self.grid_adjustment_sensitivity.value
            
            # 波动率自适应调整
            # 使用realized_volatility进行二次调整
            if 'realized_volatility' in dataframe.columns:
                current_vol = dataframe['realized_volatility'].iloc[-1]
                vol_avg = dataframe['realized_volatility'].rolling(window=50).mean().iloc[-1]
                
                if not np.isnan(vol_avg) and vol_avg > 0:
                    vol_factor = min(2.0, max(0.5, current_vol / vol_avg))
                    base_spacing_pct *= vol_factor
            
            # 应用间距边界限制
            adaptive_spacing = max(
                self.min_grid_spacing_pct.value,
                min(base_spacing_pct, self.max_grid_spacing_pct.value)
            )
            
            self.logger.debug(f"ATR自适应间距: {adaptive_spacing:.4f} (ATR%: {current_atr_pct:.4f})")
            return adaptive_spacing
            
        except Exception as e:
            self.logger.error(f"ATR间距计算错误: {e}")
            # 返回默认间距
            return (self.min_grid_spacing_pct.value + self.max_grid_spacing_pct.value) / 2
    
    def _calculate_dynamic_boundaries(self, dataframe: DataFrame) -> tuple:
        """
        基于布林带计算动态网格边界
        
        算法：
        1. 获取布林带上下轨
        2. 应用边界扩展系数
        3. 确保边界合理性
        4. 防止边界过度扩张
        
        Args:
            dataframe: K线数据
            
        Returns:
            tuple: (下边界价格, 上边界价格)
        """
        try:
            # 获取最新布林带数据
            bb_lower = dataframe['bb_lowerband'].iloc[-1]
            bb_upper = dataframe['bb_upperband'].iloc[-1] 
            bb_middle = dataframe['bb_middleband'].iloc[-1]
            current_price = dataframe['close'].iloc[-1]
            
            # 计算布林带宽度
            bb_width_pct = (bb_upper - bb_lower) / bb_middle if bb_middle > 0 else 0.1
            
            # 动态边界扩展系数 - 基于布林带宽度调整
            base_expansion = 0.02  # 基础2%扩展
            if bb_width_pct > 0.15:  # 高波动
                expansion_factor = 0.01  # 减少扩展
            elif bb_width_pct < 0.05:  # 低波动
                expansion_factor = 0.03  # 增加扩展
            else:  # 中等波动
                expansion_factor = base_expansion
            
            # 计算动态边界
            lower_bound = bb_lower * (1 - expansion_factor)
            upper_bound = bb_upper * (1 + expansion_factor)
            
            # 边界合理性检查
            if lower_bound >= upper_bound or lower_bound <= 0:
                self.logger.warning("边界计算异常，使用默认边界")
                price_range_pct = 0.10  # 默认10%价格区间
                lower_bound = current_price * (1 - price_range_pct / 2)
                upper_bound = current_price * (1 + price_range_pct / 2)
            
            self.logger.debug(f"动态边界: [{lower_bound:.4f}, {upper_bound:.4f}], 宽度: {(upper_bound-lower_bound)/current_price:.4f}")
            return lower_bound, upper_bound
            
        except Exception as e:
            self.logger.error(f"边界计算错误: {e}")
            # 返回基于当前价格的默认边界
            current_price = dataframe['close'].iloc[-1] if len(dataframe) > 0 else 100.0
            return current_price * 0.95, current_price * 1.05
    
    def _generate_grid_levels(self, dataframe: DataFrame, lower_bound: float, 
                            upper_bound: float, spacing: float) -> dict:
        """
        生成20层固定网格结构
        
        算法：
        1. 均匀分布20个网格层级
        2. 计算每层的目标价位
        3. 分配买入/卖出网格
        4. 记录网格元数据
        
        Args:
            dataframe: K线数据
            lower_bound: 网格下边界
            upper_bound: 网格上边界  
            spacing: 网格间距参考值
            
        Returns:
            dict: 网格层级数据
        """
        try:
            current_price = dataframe['close'].iloc[-1]
            grid_levels = 20  # 固定20层网格
            
            # 计算网格间距
            total_range = upper_bound - lower_bound
            level_spacing = total_range / (grid_levels - 1)
            
            # 生成网格层级
            buy_levels = []
            sell_levels = []
            
            for i in range(grid_levels):
                level_price = lower_bound + (i * level_spacing)
                level_info = {
                    'level': i,
                    'price': level_price,
                    'distance_pct': (level_price - current_price) / current_price,
                    'spacing': spacing,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                
                # 分配买入/卖出网格 - 基于相对于当前价格的位置
                if level_price < current_price:
                    # 价格下方为买入网格
                    level_info['type'] = 'buy'
                    level_info['side'] = 'long_entry'
                    buy_levels.append(level_info)
                elif level_price > current_price:
                    # 价格上方为卖出网格
                    level_info['type'] = 'sell'  
                    level_info['side'] = 'long_exit'
                    sell_levels.append(level_info)
                else:
                    # 接近当前价格的中性网格
                    level_info['type'] = 'neutral'
                    level_info['side'] = 'hold'
            
            grid_data = {
                'total_levels': grid_levels,
                'buy_levels': buy_levels,
                'sell_levels': sell_levels,
                'price_range': (lower_bound, upper_bound),
                'level_spacing': level_spacing,
                'adaptive_spacing': spacing,
                'current_price': current_price,
                'created_at': datetime.now(timezone.utc).isoformat()
            }
            
            self.logger.info(f"生成动态网格: {len(buy_levels)}买入层 + {len(sell_levels)}卖出层, 间距: {level_spacing:.4f}")
            return grid_data
            
        except Exception as e:
            self.logger.error(f"网格层级生成错误: {e}")
            return {}
    
    def _add_grid_indicators(self, dataframe: DataFrame, grid_data: dict) -> DataFrame:
        """
        添加网格相关指标到数据框
        
        Args:
            dataframe: K线数据
            grid_data: 网格层级数据
            
        Returns:
            DataFrame: 包含网格指标的数据框
        """
        try:
            if not grid_data:
                return dataframe
                
            # 添加网格边界指标
            lower_bound, upper_bound = grid_data['price_range']
            dataframe['grid_lower_bound'] = lower_bound
            dataframe['grid_upper_bound'] = upper_bound
            dataframe['grid_level_spacing'] = grid_data['level_spacing']
            dataframe['grid_adaptive_spacing'] = grid_data['adaptive_spacing']
            
            # 计算当前价格在网格中的位置
            current_price = grid_data['current_price']
            grid_position = (current_price - lower_bound) / (upper_bound - lower_bound)
            dataframe['grid_position'] = grid_position
            
            # 计算距离最近网格层的距离
            buy_levels = grid_data.get('buy_levels', [])
            sell_levels = grid_data.get('sell_levels', [])
            
            if buy_levels:
                nearest_buy_distance = min([
                    abs(level['price'] - current_price) / current_price 
                    for level in buy_levels
                ])
                dataframe['nearest_buy_grid_distance'] = nearest_buy_distance
            
            if sell_levels:
                nearest_sell_distance = min([
                    abs(level['price'] - current_price) / current_price 
                    for level in sell_levels
                ])
                dataframe['nearest_sell_grid_distance'] = nearest_sell_distance
            
            # 网格状态指标
            dataframe['grid_total_levels'] = grid_data['total_levels']
            dataframe['grid_buy_levels_count'] = len(buy_levels)
            dataframe['grid_sell_levels_count'] = len(sell_levels)
            
            return dataframe
            
        except Exception as e:
            self.logger.error(f"网格指标添加错误: {e}")
            return dataframe
    
    def _should_reset_grid(self, dataframe: DataFrame) -> bool:
        """
        判断是否需要重置网格
        
        重置条件：
        1. 价格突破网格边界
        2. ATR波动率显著变化
        3. 布林带宽度剧烈变化
        4. 网格计算时间过长
        
        Args:
            dataframe: K线数据
            
        Returns:
            bool: 是否需要重置网格
        """
        try:
            if len(dataframe) < 20:
                return False
                
            current_price = dataframe['close'].iloc[-1]
            
            # 检查价格是否突破网格边界
            if 'grid_lower_bound' in dataframe.columns and 'grid_upper_bound' in dataframe.columns:
                grid_lower = dataframe['grid_lower_bound'].iloc[-1]
                grid_upper = dataframe['grid_upper_bound'].iloc[-1]
                
                # 价格突破边界的阈值
                boundary_threshold = 0.02  # 2%
                if current_price < grid_lower * (1 - boundary_threshold):
                    self.logger.info(f"价格突破下边界 {current_price} < {grid_lower * (1 - boundary_threshold)}")
                    return True
                if current_price > grid_upper * (1 + boundary_threshold):
                    self.logger.info(f"价格突破上边界 {current_price} > {grid_upper * (1 + boundary_threshold)}")
                    return True
            
            # 检查ATR波动率变化
            if 'atr_percent' in dataframe.columns:
                current_atr = dataframe['atr_percent'].iloc[-1]
                avg_atr = dataframe['atr_percent'].rolling(window=20).mean().iloc[-1]
                
                if not np.isnan(avg_atr) and avg_atr > 0:
                    atr_change_ratio = abs(current_atr - avg_atr) / avg_atr
                    if atr_change_ratio > 0.5:  # ATR变化超过50%
                        self.logger.info(f"ATR波动率显著变化: {atr_change_ratio:.3f}")
                        return True
            
            # 检查布林带宽度变化
            if 'bb_width' in dataframe.columns:
                current_bb_width = dataframe['bb_width'].iloc[-1]
                avg_bb_width = dataframe['bb_width'].rolling(window=20).mean().iloc[-1]
                
                if not np.isnan(avg_bb_width) and avg_bb_width > 0:
                    bb_width_change_ratio = abs(current_bb_width - avg_bb_width) / avg_bb_width
                    if bb_width_change_ratio > 0.4:  # 布林带宽度变化超过40%
                        self.logger.info(f"布林带宽度显著变化: {bb_width_change_ratio:.3f}")
                        return True
            
            # 时间相关重置 - 如果网格创建时间过长
            # 这里暂时不实现，后续可以基于网格创建时间戳来判断
            
            return False
            
        except Exception as e:
            self.logger.error(f"网格重置条件判断错误: {e}")
            return False
    
    def _reset_grid_cache(self, dataframe: DataFrame) -> DataFrame:
        """
        重置网格缓存和状态
        
        Args:
            dataframe: K线数据
            
        Returns:
            DataFrame: 重置后的数据框
        """
        try:
            # 清理网格相关指标
            grid_columns = [
                'grid_lower_bound', 'grid_upper_bound', 'grid_level_spacing',
                'grid_adaptive_spacing', 'grid_position', 'nearest_buy_grid_distance',
                'nearest_sell_grid_distance', 'grid_total_levels', 
                'grid_buy_levels_count', 'grid_sell_levels_count'
            ]
            
            for col in grid_columns:
                if col in dataframe.columns:
                    dataframe[col] = np.nan
            
            # 清理缓存状态
            self._indicator_cache.clear()
            self._last_update_time = datetime.now(timezone.utc)
            
            # 注意：这里不能直接调用calculate_dynamic_grid_levels，避免递归
            # 重置完成，返回清理后的数据框
            return dataframe
            
        except Exception as e:
            self.logger.error(f"网格缓存重置错误: {e}")
            return dataframe

    # ===========================================
    # 预留接口方法（为后续任务准备）
    # ===========================================

    def calculate_dynamic_grid_levels(self, dataframe: DataFrame) -> DataFrame:
        """
        动态网格层级计算 - ATR自适应网格间距和布林带区间确定
        
        核心功能：
        1. 基于ATR计算自适应网格间距
        2. 使用布林带上下轨确定价格区间
        3. 固定20层网格结构，智能分布价位
        4. 实现网格重置机制
        
        Args:
            dataframe: K线数据
            
        Returns:
            DataFrame: 包含动态网格层级的数据框
        """
        if len(dataframe) == 0:
            self.logger.warning("数据框为空，返回空网格")
            return dataframe
            
        # 确保必需指标存在
        required_indicators = ['atr', 'bb_lowerband', 'bb_upperband', 'bb_middleband', 'close']
        for indicator in required_indicators:
            if indicator not in dataframe.columns:
                self.logger.error(f"缺少必需指标: {indicator}")
                return super().calculate_grid_levels(dataframe)
        
        try:
            # 计算ATR自适应网格间距
            adaptive_spacing = self._calculate_adaptive_grid_spacing(dataframe)
            
            # 获取布林带动态边界
            lower_bound, upper_bound = self._calculate_dynamic_boundaries(dataframe)
            
            # 生成20层固定网格
            grid_levels_data = self._generate_grid_levels(
                dataframe, lower_bound, upper_bound, adaptive_spacing
            )
            
            # 添加网格相关指标到数据框
            dataframe = self._add_grid_indicators(dataframe, grid_levels_data)
            
            # 检查是否需要重置网格
            if self._should_reset_grid(dataframe):
                self.logger.info("触发网格重置条件，清理网格缓存")
                # 只是清理缓存，不再递归调用
                self._indicator_cache.clear()
                self._last_update_time = datetime.now(timezone.utc)
            
            return dataframe
            
        except Exception as e:
            self.logger.error(f"动态网格计算出错: {e}")
            # 降级到父类基础实现
            return super().calculate_grid_levels(dataframe)


    def detect_market_regime(self, dataframe: DataFrame) -> str:
        """
        增强市场状态识别 - 使用ADX+DI指标识别趋势强度和方向
        
        识别规则：
        1. 震荡市：ADX < 25
        2. 趋势市：ADX > 40
        3. 过渡期：25 <= ADX <= 40
        4. 趋势方向：基于DI指标和MA/EMA交叉确认
        5. 状态平滑：避免频繁切换
        
        Args:
            dataframe: K线数据
            
        Returns:
            str: 市场状态（'uptrend', 'downtrend', 'consolidation', 'transition'）
        """
        if len(dataframe) == 0:
            return "consolidation"
        
        try:
            # 获取最新数据
            last_candle = dataframe.iloc[-1]
            
            # 检查必需指标
            required_indicators = ['adx', 'plus_di', 'minus_di', 'ema_12', 'ema_26']
            for indicator in required_indicators:
                if indicator not in dataframe.columns:
                    self.logger.warning(f"缺少必需指标: {indicator}")
                    return "consolidation"
                    
            current_adx = last_candle['adx']
            current_plus_di = last_candle['plus_di']
            current_minus_di = last_candle['minus_di']
            current_ema_12 = last_candle['ema_12']
            current_ema_26 = last_candle['ema_26']
            
            # 检查数据有效性
            if any(np.isnan([current_adx, current_plus_di, current_minus_di, current_ema_12, current_ema_26])):
                return "consolidation"
            
            # 1. 基于ADX判断市场状态类型
            if current_adx < self.adx_consolidation_threshold.value:
                # 震荡市场 - ADX较低，趋势不明显
                regime_type = "consolidation"
            elif current_adx > self.adx_trend_threshold.value:
                # 强趋势市场 - 需要进一步确认方向
                regime_type = self._determine_trend_direction(dataframe, last_candle)
            else:
                # 过渡期 - ADX在中等范围
                regime_type = "transition"
            
            # 2. 平滑处理 - 避免频繁切换
            regime_type = self._apply_regime_smoothing(regime_type, dataframe)
            
            # 3. 记录历史状态
            self._record_regime_history(regime_type)
            
            # 4. 更新内部状态
            self._market_regime = regime_type
            
            self.logger.debug(
                f"市场状态识别: {regime_type} "
                f"(ADX:{current_adx:.2f}, +DI:{current_plus_di:.2f}, -DI:{current_minus_di:.2f})"
            )
            
            return regime_type
            
        except Exception as e:
            self.logger.error(f"市场状态识别错误: {e}")
            return "consolidation"  # 错误时返回保守状态

    def _determine_trend_direction(self, dataframe: DataFrame, last_candle) -> str:
        """
        确定趋势方向 - 结合DI指标和MA/EMA交叉分析
        
        Args:
            dataframe: K线数据
            last_candle: 最新K线数据
            
        Returns:
            str: 趋势方向 ('uptrend', 'downtrend', 'transition')
        """
        try:
            current_plus_di = last_candle['plus_di']
            current_minus_di = last_candle['minus_di']
            current_ema_12 = last_candle['ema_12']
            current_ema_26 = last_candle['ema_26']
            
            # DI指标比例分析
            di_ratio_threshold = self.di_ratio_threshold.value
            
            # EMA交叉分析
            ema_bullish = current_ema_12 > current_ema_26
            ema_bearish = current_ema_12 < current_ema_26
            
            # 综合判断趋势方向
            if (current_plus_di > current_minus_di * di_ratio_threshold and ema_bullish):
                # 多头信号：+DI明显大于-DI且EMA多头排列
                return "uptrend"
            elif (current_minus_di > current_plus_di * di_ratio_threshold and ema_bearish):
                # 空头信号：-DI明显大于+DI且EMA空头排列
                return "downtrend"
            else:
                # 信号不一致或强度不够，属于过渡状态
                return "transition"
                
        except Exception as e:
            self.logger.error(f"趋势方向判断错误: {e}")
            return "transition"
    
    def _apply_regime_smoothing(self, current_regime: str, dataframe: DataFrame) -> str:
        """
        市场状态平滑处理 - 避免频繁切换
        
        策略：
        1. 检查近期状态历史
        2. 只有在新状态持续一定时间后才确认切换
        3. 特殊情况：从趋势切换到震荡时直接确认
        
        Args:
            current_regime: 当前识别到的状态
            dataframe: K线数据
            
        Returns:
            str: 平滑后的市场状态
        """
        try:
            # 获取历史状态列（如果存在）
            if 'market_regime_raw' in dataframe.columns and len(dataframe) >= self.regime_smoothing_period.value:
                recent_regimes = dataframe['market_regime_raw'].tail(self.regime_smoothing_period.value).tolist()
            else:
                recent_regimes = []
            
            # 如果没有足够的历史数据，直接返回当前状态
            if len(recent_regimes) < 2:
                return current_regime
                
            # 获取上一个确认的状态
            last_confirmed_regime = recent_regimes[-1] if recent_regimes else current_regime
            
            # 如果状态未发生变化，直接返回
            if current_regime == last_confirmed_regime:
                return current_regime
            
            # 特殊规则：趋势切换到震荡/过渡时快速确认
            if (last_confirmed_regime in ['uptrend', 'downtrend'] and 
                current_regime in ['consolidation', 'transition']):
                self.logger.info(f"趋势结束信号：{last_confirmed_regime} -> {current_regime}")
                return current_regime
            
            # 计算新状态的一致性
            regime_consistency_count = sum(1 for regime in recent_regimes[-self.regime_smoothing_period.value:] 
                                         if regime == current_regime)
            
            # 只有在新状态达到一定一致性时才确认切换
            required_consistency = max(2, self.regime_smoothing_period.value // 2)
            
            if regime_consistency_count >= required_consistency:
                self.logger.info(f"状态切换确认：{last_confirmed_regime} -> {current_regime}")
                self._regime_change_count += 1
                return current_regime
            else:
                # 不足以确认切换，继续保持原状态
                self.logger.debug(f"状态平滑：保持 {last_confirmed_regime} (新状态 {current_regime} 一致性: {regime_consistency_count}/{required_consistency})")
                return last_confirmed_regime
                
        except Exception as e:
            self.logger.error(f"状态平滑处理错误: {e}")
            return current_regime
    
    def _record_regime_history(self, regime: str) -> None:
        """
        记录市场状态历史
        
        Args:
            regime: 市场状态
        """
        try:
            current_time = datetime.now(timezone.utc).isoformat()
            
            # 添加到历史记录
            regime_record = {
                'timestamp': current_time,
                'regime': regime,
                'previous_regime': self._market_regime if hasattr(self, '_market_regime') else 'unknown'
            }
            
            self._regime_history.append(regime_record)
            
            # 限制历史记录数量（保持最近100条记录）
            if len(self._regime_history) > 100:
                self._regime_history = self._regime_history[-100:]
                
        except Exception as e:
            self.logger.error(f"历史状态记录错误: {e}")
    
    def _get_current_dataframe(self, pair: str) -> Optional[DataFrame]:
        """
        获取当前交易对的数据框
        
        Args:
            pair: 交易对
            
        Returns:
            DataFrame: 当前数据框，如果无法获取则返回None
        """
        try:
            # 优先从缓存获取
            if hasattr(self, '_indicator_cache') and pair in self._indicator_cache:
                cached_data = self._indicator_cache[pair]
                # 检查缓存时效性（5分钟）
                if (datetime.now(timezone.utc) - cached_data.get('timestamp', datetime.min.replace(tzinfo=timezone.utc))).seconds < 300:
                    return cached_data.get('dataframe')
            
            # 缓存无效或不存在，返回None让调用方处理
            return None
            
        except Exception as e:
            self.logger.error(f"获取数据框错误 {pair}: {e}")
            return None
    
    def _apply_market_regime_detection(self, dataframe: DataFrame) -> DataFrame:
        """
        应用市场状态识别并添加相关指标
        
        Args:
            dataframe: K线数据
            
        Returns:
            DataFrame: 包含状态识别指标的数据框
        """
        try:
            if len(dataframe) == 0:
                return dataframe
                
            # 先计算原始状态（未平滑）
            raw_regimes = []
            for i in range(len(dataframe)):
                temp_df = dataframe.iloc[:i+1]
                if len(temp_df) < 20:  # 需要足够的数据计算指标
                    raw_regimes.append('consolidation')
                    continue
                    
                last_candle = temp_df.iloc[-1]
                
                # 检查必需指标
                if any(np.isnan([last_candle.get('adx', np.nan), 
                               last_candle.get('plus_di', np.nan),
                               last_candle.get('minus_di', np.nan)])):
                    raw_regimes.append('consolidation')
                    continue
                    
                current_adx = last_candle['adx']
                current_plus_di = last_candle['plus_di']
                current_minus_di = last_candle['minus_di']
                current_ema_12 = last_candle.get('ema_12', np.nan)
                current_ema_26 = last_candle.get('ema_26', np.nan)
                
                # 基本状态判断
                if current_adx < self.adx_consolidation_threshold.value:
                    regime = "consolidation"
                elif current_adx > self.adx_trend_threshold.value:
                    # 趋势方向判断
                    di_ratio = self.di_ratio_threshold.value
                    ema_bullish = not np.isnan(current_ema_12) and not np.isnan(current_ema_26) and current_ema_12 > current_ema_26
                    ema_bearish = not np.isnan(current_ema_12) and not np.isnan(current_ema_26) and current_ema_12 < current_ema_26
                    
                    if current_plus_di > current_minus_di * di_ratio and ema_bullish:
                        regime = "uptrend"
                    elif current_minus_di > current_plus_di * di_ratio and ema_bearish:
                        regime = "downtrend"
                    else:
                        regime = "transition"
                else:
                    regime = "transition"
                    
                raw_regimes.append(regime)
            
            # 添加原始状态列
            dataframe['market_regime_raw'] = raw_regimes
            
            # 平滑处理
            smoothed_regimes = []
            for i in range(len(dataframe)):
                if i < self.regime_smoothing_period.value:
                    smoothed_regimes.append(raw_regimes[i])
                else:
                    # 获取最近N个状态
                    recent_regimes = raw_regimes[max(0, i-self.regime_smoothing_period.value+1):i+1]
                    current_regime = raw_regimes[i]
                    last_smoothed = smoothed_regimes[-1]
                    
                    # 平滑逻辑
                    if current_regime == last_smoothed:
                        smoothed_regimes.append(current_regime)
                    else:
                        # 特殊情况：趋势结束
                        if (last_smoothed in ['uptrend', 'downtrend'] and 
                            current_regime in ['consolidation', 'transition']):
                            smoothed_regimes.append(current_regime)
                        else:
                            # 检查一致性
                            consistency = sum(1 for r in recent_regimes if r == current_regime)
                            required = max(2, self.regime_smoothing_period.value // 2)
                            
                            if consistency >= required:
                                smoothed_regimes.append(current_regime)
                            else:
                                smoothed_regimes.append(last_smoothed)
                                
            dataframe['market_regime'] = smoothed_regimes
            
            # 添加额外的分析指标
            dataframe['regime_adx_score'] = np.where(
                dataframe['adx'] < self.adx_consolidation_threshold.value, 0,
                np.where(dataframe['adx'] > self.adx_trend_threshold.value, 2, 1)
            )
            
            dataframe['regime_di_ratio'] = np.where(
                dataframe['minus_di'] != 0,
                dataframe['plus_di'] / dataframe['minus_di'],
                1.0
            )
            
            # 状态持续时间计数
            regime_duration = []
            current_duration = 1
            for i in range(len(smoothed_regimes)):
                if i > 0 and smoothed_regimes[i] == smoothed_regimes[i-1]:
                    current_duration += 1
                else:
                    current_duration = 1
                regime_duration.append(current_duration)
                
            dataframe['regime_duration'] = regime_duration
            
            return dataframe
            
        except Exception as e:
            self.logger.error(f"市场状态识别应用错误: {e}")
            return dataframe

    def calculate_enhanced_position_size(self, pair: str, current_time: datetime, 
                                       current_rate: float, proposed_stake: float,
                                       **kwargs) -> float:
        """
        增强仓位计算 - 基于市场状态的三级仓位管理系统
        
        仓位模式：
        - 保守模式：基础仓位 × 0.5 (震荡市)
        - 标准模式：基础仓位 × 1.0 (过渡期)
        - 激进模式：基础仓位 × 1.5 (趋势市)
        
        调整因子：
        - 市场状态调整：基于detect_market_regime()结果
        - 波动率调整：基于ATR和realized_volatility
        - 风险控制：最大仓位限制和平滑过渡
        
        Args:
            pair: 交易对
            current_time: 当前时间
            current_rate: 当前价格
            proposed_stake: 建议仓位
            **kwargs: 其他参数
            
        Returns:
            float: 调整后的仓位大小
        """
        try:
            # 1. 获取基础仓位
            base_stake = super().custom_stake_amount(
                pair=pair,
                current_time=current_time,
                current_rate=current_rate,
                proposed_stake=proposed_stake,
                min_stake=kwargs.get('min_stake'),
                max_stake=kwargs.get('max_stake'),
                leverage=kwargs.get('leverage', 1.0),
                entry_tag=kwargs.get('entry_tag'),
                side=kwargs.get('side', 'long'),
                **kwargs
            )
            
            # 2. 获取当前市场数据框（通过缓存或重新计算）
            dataframe = self._get_current_dataframe(pair)
            if dataframe is None or len(dataframe) == 0:
                self.logger.warning(f"无法获取 {pair} 的市场数据，使用基础仓位")
                return base_stake
                
            # 3. 市场状态调整因子
            market_adjustment_factor = self._calculate_market_regime_adjustment_factor(dataframe)
            
            # 4. 波动率调整因子
            volatility_adjustment_factor = self._calculate_volatility_adjustment_factor(dataframe)
            
            # 5. 综合调整因子计算
            combined_adjustment_factor = market_adjustment_factor * volatility_adjustment_factor
            
            # 6. 应用调整因子
            enhanced_stake = base_stake * combined_adjustment_factor
            
            # 7. 风险控制和边界检查
            enhanced_stake = self._apply_position_risk_controls(
                enhanced_stake, base_stake, pair, kwargs
            )
            
            self.logger.info(
                f"增强仓位计算 {pair}: 基础={base_stake:.4f}, "
                f"市场调整={market_adjustment_factor:.3f}, "
                f"波动率调整={volatility_adjustment_factor:.3f}, "
                f"最终={enhanced_stake:.4f}"
            )
            
            return enhanced_stake
            
        except Exception as e:
            self.logger.error(f"增强仓位计算错误 {pair}: {e}")
            # 错误时返回基础仓位，确保策略稳定运行
            return super().custom_stake_amount(
                pair=pair,
                current_time=current_time,
                current_rate=current_rate,
                proposed_stake=proposed_stake,
                min_stake=kwargs.get('min_stake'),
                max_stake=kwargs.get('max_stake'),
                leverage=kwargs.get('leverage', 1.0),
                entry_tag=kwargs.get('entry_tag'),
                side=kwargs.get('side', 'long'),
                **kwargs
            )

    def calculate_enhanced_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                                  current_rate: float, current_profit: float,
                                  **kwargs) -> float:
        """
        集成VATSM的三层追踪止损系统
        
        核心功能：
        1. 三层追踪止损：不同盈利水平采用不同止损距离
        2. 时间止损：持仓时间过长强制退出
        3. 动态风险调整：基于市场波动率调整止损敏感度
        4. 回撤保护：大回撤时收紧止损
        
        Args:
            pair: 交易对
            trade: 交易对象
            current_time: 当前时间
            current_rate: 当前价格
            current_profit: 当前利润
            **kwargs: 其他参数
            
        Returns:
            float: 动态止损比例
        """
        try:
            # 1. 获取或初始化交易追踪信息
            trade_id = f"{pair}_{trade.open_date_utc}"
            
            if trade_id not in self.custom_trade_info:
                self.custom_trade_info[trade_id] = {
                    'max_profit': current_profit,
                    'trailing_activated': False,
                    'tier_activated': 0,
                    'entry_time': current_time,
                    'entry_rate': trade.open_rate,
                    'stop_loss_history': [],
                    'risk_level': 'normal'  # normal, elevated, high
                }
            
            trade_info = self.custom_trade_info[trade_id]
            
            # 2. 更新最大盈利记录
            if current_profit > trade_info['max_profit']:
                trade_info['max_profit'] = current_profit
            
            # 3. 计算基础止损（来自父类）
            base_stoploss = super().custom_stoploss(
                pair=pair,
                trade=trade,
                current_time=current_time,
                current_rate=current_rate,
                current_profit=current_profit,
                **kwargs
            )
            
            # 4. 三层追踪止损计算
            trailing_stoploss = self._calculate_tiered_trailing_stop(
                trade_info, current_profit, base_stoploss
            )
            
            # 5. 时间止损检查
            time_stoploss = self._calculate_time_based_stop(
                trade_info, current_time, current_profit
            )
            
            # 6. 动态风险调整
            risk_adjusted_stoploss = self._apply_risk_adjustments(
                pair, trailing_stoploss, time_stoploss, trade_info
            )
            
            # 7. 记录止损历史用于分析
            trade_info['stop_loss_history'].append({
                'timestamp': current_time,
                'profit': current_profit,
                'stoploss': risk_adjusted_stoploss,
                'tier': trade_info['tier_activated']
            })
            
            # 8. 限制历史记录大小
            if len(trade_info['stop_loss_history']) > 50:
                trade_info['stop_loss_history'] = trade_info['stop_loss_history'][-50:]
            
            self.logger.debug(
                f"增强止损计算 {pair}: 盈利={current_profit:.4f}, "
                f"层级={trade_info['tier_activated']}, 止损={risk_adjusted_stoploss:.4f}"
            )
            
            return risk_adjusted_stoploss
            
        except Exception as e:
            self.logger.error(f"增强止损计算错误 {pair}: {e}")
            # 错误时返回保守的基础止损
            return max(self.stoploss, -0.15)

    def _calculate_tiered_trailing_stop(self, trade_info: dict, current_profit: float, base_stoploss: float) -> float:
        """
        计算三层追踪止损
        
        三层止损逻辑：
        - 第一层：小盈利保护，较小追踪距离，早期启动
        - 第二层：中盈利保护，适中追踪距离，平衡保护
        - 第三层：大盈利保护，较大追踪距离，最大化收益
        
        Args:
            trade_info: 交易追踪信息
            current_profit: 当前盈利
            base_stoploss: 基础止损
            
        Returns:
            float: 三层追踪止损值
        """
        try:
            trailing_distance = 0.0
            max_profit = trade_info['max_profit']
            
            # 第三层：大盈利保护（5%以上）
            if current_profit > self.tier3_profit.value:
                trailing_distance = self.tier3_distance.value
                if trade_info['tier_activated'] < 3:
                    trade_info['tier_activated'] = 3
                    trade_info['trailing_activated'] = True
                    self.logger.info(f"启动第三层止损保护: 盈利={current_profit:.4f}, 距离={trailing_distance:.4f}")
            
            # 第二层：中等盈利保护（2%以上）
            elif current_profit > self.tier2_profit.value:
                trailing_distance = self.tier2_distance.value
                if trade_info['tier_activated'] < 2:
                    trade_info['tier_activated'] = 2
                    trade_info['trailing_activated'] = True
                    self.logger.info(f"启动第二层止损保护: 盈利={current_profit:.4f}, 距离={trailing_distance:.4f}")
            
            # 第一层：小盈利保护（0.8%以上）
            elif current_profit > self.tier1_profit.value:
                trailing_distance = self.tier1_distance.value
                if trade_info['tier_activated'] < 1:
                    trade_info['tier_activated'] = 1
                    trade_info['trailing_activated'] = True
                    self.logger.info(f"启动第一层止损保护: 盈利={current_profit:.4f}, 距离={trailing_distance:.4f}")
            
            # 如果追踪止损已启动且有追踪距离
            if trade_info['trailing_activated'] and trailing_distance > 0:
                # 追踪止损位置 = 最大盈利 - 对应追踪距离
                trailing_stop = max_profit - trailing_distance
                # 确保追踪止损不会低于基础止损
                return max(base_stoploss, trailing_stop)
            
            # 未启动追踪止损时，使用基础止损
            return base_stoploss
            
        except Exception as e:
            self.logger.error(f"三层追踪止损计算错误: {e}")
            return base_stoploss
    
    def _calculate_time_based_stop(self, trade_info: dict, current_time: datetime, current_profit: float) -> Optional[float]:
        """
        计算时间止损
        
        时间止损规则：
        - 持仓时间超过设定最大时间且盈利满足条件时触发
        - 避免长期占用资金的无效持仓
        
        Args:
            trade_info: 交易追踪信息
            current_time: 当前时间
            current_profit: 当前盈利
            
        Returns:
            Optional[float]: 时间止损值，None表示未触发
        """
        try:
            if 'entry_time' not in trade_info:
                return None
                
            entry_time = trade_info['entry_time']
            hold_duration = current_time - entry_time
            hold_hours = hold_duration.total_seconds() / 3600
            
            # 检查是否超过最大持仓时间
            if hold_hours > self.max_hold_hours.value:
                # 检查盈利是否满足时间止损条件
                if current_profit > self.time_exit_profit_threshold.value:
                    # 时间止损：当前盈利减去小幅缓冲避免滑点
                    time_stop = current_profit - 0.005  # 0.5%缓冲
                    self.logger.info(
                        f"触发时间止损: 持仓{hold_hours:.1f}小时, 盈利={current_profit:.4f}, "
                        f"止损={time_stop:.4f}"
                    )
                    return time_stop
                else:
                    # 盈利不足但时间过长，使用更激进的止损
                    aggressive_stop = current_profit - 0.01  # 1%缓冲
                    self.logger.warning(
                        f"长期持仓盈利不足: 持仓{hold_hours:.1f}小时, 盈利={current_profit:.4f}, "
                        f"激进止损={aggressive_stop:.4f}"
                    )
                    return aggressive_stop
            
            return None
            
        except Exception as e:
            self.logger.error(f"时间止损计算错误: {e}")
            return None
    
    def _apply_risk_adjustments(self, pair: str, trailing_stoploss: float, 
                              time_stoploss: Optional[float], trade_info: dict) -> float:
        """
        应用动态风险调整
        
        风险调整因子：
        1. 市场波动率调整
        2. 当前回撤水平调整
        3. 连续亏损调整
        4. 紧急风控调整
        
        Args:
            pair: 交易对
            trailing_stoploss: 追踪止损值
            time_stoploss: 时间止损值
            trade_info: 交易信息
            
        Returns:
            float: 风险调整后的最终止损值
        """
        try:
            # 1. 选择基础止损（时间止损 vs 追踪止损）
            if time_stoploss is not None:
                # 时间止损存在时，选择更严格的
                base_stop = max(trailing_stoploss, time_stoploss)
                self.logger.debug(f"采用时间止损: {base_stop:.4f}")
            else:
                base_stop = trailing_stoploss
            
            # 2. 获取当前市场数据进行波动率调整
            try:
                dataframe = self._get_current_dataframe(pair)
                if dataframe is not None and len(dataframe) > 0:
                    volatility_adjustment = self._calculate_volatility_stop_adjustment(dataframe)
                    base_stop *= volatility_adjustment
                    
                    self.logger.debug(f"波动率调整: {volatility_adjustment:.3f}")
            except Exception as e:
                self.logger.warning(f"波动率调整失败: {e}")
            
            # 3. 回撤水平调整
            if self.current_drawdown > self.drawdown_pause_threshold.value * 0.5:
                # 回撤较大时收紧止损
                drawdown_adjustment = 1.2  # 收紧20%
                base_stop *= drawdown_adjustment
                trade_info['risk_level'] = 'elevated'
                self.logger.info(f"回撤调整止损: 回撤={self.current_drawdown:.4f}, 调整={drawdown_adjustment:.3f}")
            
            # 4. 连续亏损调整
            if self.consecutive_losses >= 3:
                # 连续亏损时更加保守
                loss_adjustment = 1.1 + (self.consecutive_losses - 3) * 0.05
                base_stop *= min(loss_adjustment, 1.5)  # 最大调整50%
                trade_info['risk_level'] = 'high'
                self.logger.info(f"连续亏损调整: 连续={self.consecutive_losses}, 调整={loss_adjustment:.3f}")
            
            # 5. 应用增强因子
            final_stop = base_stop * self.enhanced_stoploss_factor.value
            
            # 6. 边界检查：确保止损在合理范围
            final_stop = max(final_stop, -0.20)  # 最大止损20%
            final_stop = min(final_stop, 0.05)   # 最小止损（允许小幅盈利止损）
            
            return final_stop
            
        except Exception as e:
            self.logger.error(f"风险调整计算错误: {e}")
            return trailing_stoploss
    
    def _calculate_volatility_stop_adjustment(self, dataframe: DataFrame) -> float:
        """
        基于市场波动率的止损调整
        
        Args:
            dataframe: 市场数据
            
        Returns:
            float: 波动率调整因子
        """
        try:
            if len(dataframe) == 0:
                return 1.0
            
            last_candle = dataframe.iloc[-1]
            
            # 基于ATR的波动率调整
            if 'atr_percent' in dataframe.columns:
                current_atr = last_candle['atr_percent']
                avg_atr = dataframe['atr_percent'].rolling(window=20).mean().iloc[-1]
                
                if not pd.isna(current_atr) and not pd.isna(avg_atr) and avg_atr > 0:
                    atr_ratio = current_atr / avg_atr
                    
                    # 高波动时收紧止损，低波动时放松止损
                    if atr_ratio > 1.5:  # 高波动
                        return 1.15  # 收紧15%
                    elif atr_ratio < 0.7:  # 低波动
                        return 0.9   # 放松10%
                    else:
                        # 线性调整
                        return 0.9 + (atr_ratio - 0.7) * 0.31  # 0.9 到 1.15
            
            return 1.0
            
        except Exception as e:
            self.logger.error(f"波动率止损调整计算错误: {e}")
            return 1.0

    # ===========================================
    # 重写父类方法以集成增强功能
    # ===========================================

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                           proposed_stake: float, min_stake: Optional[float], max_stake: Optional[float],
                           leverage: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        """
        重写仓位计算方法以集成增强功能
        """
        if self.enable_enhanced_position_sizing.value:
            return self.calculate_enhanced_position_size(
                pair=pair,
                current_time=current_time,
                current_rate=current_rate,
                proposed_stake=proposed_stake,
                min_stake=min_stake,
                max_stake=max_stake,
                leverage=leverage,
                entry_tag=entry_tag,
                side=side,
                **kwargs
            )
        else:
            # 使用父类实现
            return super().custom_stake_amount(
                pair=pair,
                current_time=current_time,
                current_rate=current_rate,
                proposed_stake=proposed_stake,
                min_stake=min_stake,
                max_stake=max_stake,
                leverage=leverage,
                entry_tag=entry_tag,
                side=side,
                **kwargs
            )

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        重写止损方法以集成增强功能
        
        集成功能：
        1. 三层追踪止损系统
        2. 风险控制和紧急平仓
        3. 交易历史记录和分析
        4. 连续亏损熔断检查
        """
        try:
            # 1. 检查交易暂停状态
            if self.trading_paused:
                self.logger.warning(f"交易已暂停，触发紧急平仓: {pair}")
                return current_profit - 0.001  # 立即平仓
            
            # 2. 检查紧急平仓条件
            if self.emergency_exit_triggered:
                self.logger.error(f"紧急平仓触发: {pair}")
                return current_profit - 0.001  # 立即平仓
            
            # 3. 调用增强止损计算
            enhanced_stoploss = self.calculate_enhanced_stoploss(
                pair=pair,
                trade=trade,
                current_time=current_time,
                current_rate=current_rate,
                current_profit=current_profit,
                **kwargs
            )
            
            # 4. 记录交易数据用于风险分析
            self._update_trade_statistics(pair, trade, current_profit, current_time)
            
            # 5. 执行风险控制检查
            risk_adjusted_stoploss = self._apply_global_risk_controls(
                pair, enhanced_stoploss, current_profit, current_time
            )
            
            return risk_adjusted_stoploss
            
        except Exception as e:
            self.logger.error(f"自定义止损计算错误 {pair}: {e}")
            # 错误时返回保守止损
            return max(self.stoploss, -0.12)
    
    # ===========================================
    # 增强仓位管理支持方法
    # ===========================================
    
    def _calculate_market_regime_adjustment_factor(self, dataframe: DataFrame) -> float:
        """
        计算基于市场状态的仓位调整因子
        
        调整规则：
        - 保守模式 (震荡市): 0.5
        - 标准模式 (过渡期): 1.0  
        - 激进模式 (趋势市): 1.5
        
        Args:
            dataframe: K线数据
            
        Returns:
            float: 市场状态调整因子
        """
        try:
            # 获取当前市场状态
            current_market_regime = self.detect_market_regime(dataframe)
            
            # 基于市场状态确定调整因子
            if current_market_regime == 'consolidation':
                # 震荡市 - 保守模式
                adjustment_factor = 0.5
                self.logger.debug("仓位模式：保守 (震荡市)")
                
            elif current_market_regime in ['uptrend', 'downtrend']:
                # 趋势市 - 激进模式
                adjustment_factor = 1.5
                self.logger.debug(f"仓位模式：激进 ({current_market_regime})")
                
            elif current_market_regime == 'transition':
                # 过渡期 - 标准模式
                adjustment_factor = 1.0
                self.logger.debug("仓位模式：标准 (过渡期)")
                
            else:
                # 未知状态 - 默认标准模式
                adjustment_factor = 1.0
                self.logger.debug(f"仓位模式：标准 (未知状态: {current_market_regime})")
            
            # 应用趋势强度微调
            if 'adx' in dataframe.columns and len(dataframe) > 0:
                current_adx = dataframe['adx'].iloc[-1]
                if not np.isnan(current_adx):
                    # ADX强度微调：ADX越高，调整幅度越大
                    if current_market_regime in ['uptrend', 'downtrend']:
                        # 趋势市：ADX高时进一步增强
                        strength_multiplier = min(1.2, 1.0 + (current_adx - 40) / 100)
                        adjustment_factor *= strength_multiplier
                    elif current_market_regime == 'consolidation':
                        # 震荡市：ADX低时进一步保守
                        strength_multiplier = max(0.8, 1.0 - (25 - current_adx) / 100)
                        adjustment_factor *= strength_multiplier
            
            # 确保调整因子在合理范围内
            adjustment_factor = max(0.3, min(2.0, adjustment_factor))
            
            return adjustment_factor
            
        except Exception as e:
            self.logger.error(f"市场状态调整因子计算错误: {e}")
            return 1.0  # 错误时返回中性因子
    
    def _calculate_volatility_adjustment_factor(self, dataframe: DataFrame) -> float:
        """
        计算基于波动率的仓位调整因子
        
        调整逻辑：
        - 低波动率：适度增加仓位 (×1.1-1.2)
        - 正常波动率：保持基准 (×1.0)
        - 高波动率：减少仓位 (×0.8-0.9)
        
        Args:
            dataframe: K线数据
            
        Returns:
            float: 波动率调整因子
        """
        try:
            if len(dataframe) == 0:
                return 1.0
                
            # 获取最新波动率指标
            last_candle = dataframe.iloc[-1]
            
            # 基于ATR百分比的调整
            atr_adjustment = 1.0
            if 'atr_percent' in dataframe.columns:
                current_atr_pct = last_candle['atr_percent']
                if not np.isnan(current_atr_pct):
                    # ATR调整：低波动增仓，高波动减仓
                    if current_atr_pct < 0.01:  # 1%以下为低波动
                        atr_adjustment = 1.15
                    elif current_atr_pct > 0.05:  # 5%以上为高波动
                        atr_adjustment = 0.85
                    else:
                        # 正常波动，线性调整
                        normalized_atr = (current_atr_pct - 0.01) / (0.05 - 0.01)
                        atr_adjustment = 1.15 - (normalized_atr * 0.30)  # 1.15 -> 0.85
            
            # 基于realized volatility的微调
            realized_vol_adjustment = 1.0
            if 'realized_volatility' in dataframe.columns:
                current_vol = last_candle['realized_volatility']
                avg_vol = dataframe['realized_volatility'].rolling(window=20).mean().iloc[-1]
                
                if not np.isnan(current_vol) and not np.isnan(avg_vol) and avg_vol > 0:
                    vol_ratio = current_vol / avg_vol
                    
                    # 波动率相对变化调整
                    if vol_ratio < 0.7:  # 当前波动率显著低于平均
                        realized_vol_adjustment = 1.1
                    elif vol_ratio > 1.5:  # 当前波动率显著高于平均
                        realized_vol_adjustment = 0.9
                    else:
                        # 正常范围内，小幅调整
                        if vol_ratio < 1.0:
                            realized_vol_adjustment = 1.0 + (1.0 - vol_ratio) * 0.1
                        else:
                            realized_vol_adjustment = 1.0 - (vol_ratio - 1.0) * 0.1
            
            # 综合波动率调整因子
            volatility_adjustment_factor = atr_adjustment * realized_vol_adjustment
            
            # 确保调整因子在合理范围内
            volatility_adjustment_factor = max(0.6, min(1.4, volatility_adjustment_factor))
            
            self.logger.debug(
                f"波动率调整: ATR调整={atr_adjustment:.3f}, "
                f"RealizedVol调整={realized_vol_adjustment:.3f}, "
                f"综合={volatility_adjustment_factor:.3f}"
            )
            
            return volatility_adjustment_factor
            
        except Exception as e:
            self.logger.error(f"波动率调整因子计算错误: {e}")
            return 1.0  # 错误时返回中性因子
    
    def _apply_position_risk_controls(self, enhanced_stake: float, base_stake: float, 
                                    pair: str, kwargs: dict) -> float:
        """
        应用仓位风险控制
        
        控制规则：
        1. 最大仓位限制
        2. 最小仓位保障
        3. 相对变化限制（避免剧烈变动）
        4. 账户风险控制
        
        Args:
            enhanced_stake: 增强后的仓位
            base_stake: 基础仓位
            pair: 交易对
            kwargs: 其他参数
            
        Returns:
            float: 风控后的最终仓位
        """
        try:
            # 1. 获取仓位边界
            min_stake = kwargs.get('min_stake', base_stake * 0.1)
            max_stake = kwargs.get('max_stake', base_stake * 3.0)
            
            # 2. 基础边界检查
            controlled_stake = max(min_stake, min(enhanced_stake, max_stake))
            
            # 3. 相对变化限制：避免单次调整过大
            max_change_ratio = 2.0  # 最大允许200%变化
            if enhanced_stake > base_stake * max_change_ratio:
                controlled_stake = base_stake * max_change_ratio
                self.logger.warning(f"仓位增幅限制: {enhanced_stake:.4f} -> {controlled_stake:.4f}")
            elif enhanced_stake < base_stake / max_change_ratio:
                controlled_stake = base_stake / max_change_ratio  
                self.logger.warning(f"仓位减幅限制: {enhanced_stake:.4f} -> {controlled_stake:.4f}")
            
            # 4. 最大同时开仓限制检查（如果有交易信息）
            # 这里简化处理，实际实现可以检查当前开仓数量
            
            # 5. 最终合理性检查
            if controlled_stake <= 0:
                controlled_stake = min_stake
                self.logger.warning(f"仓位修正为最小值: {controlled_stake:.4f}")
            
            # 记录调整信息
            if abs(controlled_stake - enhanced_stake) > 0.0001:
                self.logger.info(
                    f"仓位风控调整 {pair}: {enhanced_stake:.4f} -> {controlled_stake:.4f}"
                )
            
            return controlled_stake
            
        except Exception as e:
            self.logger.error(f"仓位风控处理错误: {e}")
            return base_stake  # 错误时返回基础仓位
    
    # ===========================================
    # 风险控制系统支持方法
    # ===========================================
    
    def _update_trade_statistics(self, pair: str, trade: 'Trade', current_profit: float, current_time: datetime) -> None:
        """
        更新交易统计信息用于风险分析
        
        Args:
            pair: 交易对
            trade: 交易对象
            current_profit: 当前盈利
            current_time: 当前时间
        """
        try:
            # 更新历史最大权益
            current_equity = self.wallets.get_total_stake_amount() if hasattr(self, 'wallets') else 10000.0
            if current_equity > self.max_equity:
                self.max_equity = current_equity
                self.current_drawdown = 0.0  # 重置回撤
            else:
                # 计算当前回撤
                if self.max_equity > 0:
                    self.current_drawdown = (self.max_equity - current_equity) / self.max_equity
            
            # 记录交易结果（当交易结束时）
            if hasattr(trade, 'is_open') and not trade.is_open:
                # 交易已关闭，记录结果
                profit_loss = trade.calc_profit_ratio(trade.close_rate or current_profit)
                self.trade_history.append(profit_loss)
                
                # 更新连续亏损计数
                if profit_loss < 0:
                    self.consecutive_losses += 1
                    self.logger.info(f"连续亏损计数更新: {self.consecutive_losses}")
                else:
                    self.consecutive_losses = 0  # 重置连续亏损计数
                    
            # 每日风险检查
            self._check_daily_risk_limits(current_time)
            
        except Exception as e:
            self.logger.error(f"交易统计更新错误: {e}")
    
    def _apply_global_risk_controls(self, pair: str, stoploss: float, current_profit: float, current_time: datetime) -> float:
        """
        应用全局风险控制
        
        Args:
            pair: 交易对
            stoploss: 基础止损
            current_profit: 当前盈利
            current_time: 当前时间
            
        Returns:
            float: 风险调整后的止损
        """
        try:
            # 1. 连续亏损熔断检查
            if self.consecutive_losses >= self.consecutive_loss_limit.value:
                self.logger.warning(f"触发连续亏损熔断: {self.consecutive_losses}次连续亏损")
                self.trading_paused = True
                return current_profit - 0.001  # 立即平仓
            
            # 2. 回撤保护检查
            if self.current_drawdown > self.drawdown_pause_threshold.value:
                self.logger.error(f"触发回撤保护: 当前回撤={self.current_drawdown:.4f}")
                self.emergency_exit_triggered = True
                return current_profit - 0.001  # 立即平仓
            
            # 3. 单日损失限制检查
            if not self._check_daily_risk_limits(current_time):
                self.logger.warning(f"触发单日损失限制")
                return min(stoploss, current_profit - 0.002)  # 收紧止损
            
            # 4. 大回撤时收紧止损
            if self.current_drawdown > self.drawdown_pause_threshold.value * 0.6:
                tightening_factor = 1.3  # 收紧30%
                adjusted_stoploss = stoploss * tightening_factor
                self.logger.info(f"大回撤收紧止损: {stoploss:.4f} -> {adjusted_stoploss:.4f}")
                return adjusted_stoploss
            
            return stoploss
            
        except Exception as e:
            self.logger.error(f"全局风险控制错误: {e}")
            return stoploss
    
    def _check_daily_risk_limits(self, current_time: datetime) -> bool:
        """
        检查日度和周度损失限制
        
        Args:
            current_time: 当前时间
            
        Returns:
            bool: True表示通过风险检查，False表示超限
        """
        try:
            # 检查是否需要更新日度记录
            if self.last_daily_check is None or current_time.date() != self.last_daily_check:
                self.last_daily_check = current_time.date()
                # 计算昨日损益并记录
                daily_pnl = self._calculate_daily_pnl(current_time)
                self.daily_pnl.append(daily_pnl)
                
            # 检查单日损失
            if len(self.daily_pnl) > 0:
                today_pnl = self.daily_pnl[-1]
                if today_pnl < -self.max_daily_loss.value:
                    self.logger.error(f"超过单日最大损失: {today_pnl:.4f} < {-self.max_daily_loss.value:.4f}")
                    self.trading_paused = True
                    return False
                    
            # 检查周度损失
            if len(self.daily_pnl) >= 2:  # 至少有2天数据
                weekly_pnl = sum(list(self.daily_pnl)[-7:])  # 过去7天
                if weekly_pnl < -self.max_weekly_loss.value:
                    self.logger.error(f"超过单周最大损失: {weekly_pnl:.4f} < {-self.max_weekly_loss.value:.4f}")
                    self.trading_paused = True
                    return False
                    
            return True
            
        except Exception as e:
            self.logger.error(f"日度风险检查错误: {e}")
            return True  # 错误时允许交易继续
    
    def _calculate_daily_pnl(self, current_time: datetime) -> float:
        """
        计算每日损益
        
        这里是简化实现，实际应用中需要连接真实的账户数据
        
        Args:
            current_time: 当前时间
            
        Returns:
            float: 日度损益比例
        """
        try:
            # 基于交易历史估算日度损益
            if len(self.trade_history) == 0:
                return 0.0
                
            # 简化计算：使用最近几笔交易的平均值作为日度损益估计
            recent_trades = list(self.trade_history)[-10:]  # 最近10笔交易
            if recent_trades:
                daily_estimate = sum(recent_trades) / len(recent_trades)
                return daily_estimate
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"日度损益计算错误: {e}")
            return 0.0
    
    def reset_risk_controls(self) -> None:
        """
        重置风险控制状态（用于测试或手动干预）
        """
        self.trading_paused = False
        self.emergency_exit_triggered = False
        self.consecutive_losses = 0
        self.current_drawdown = 0.0
        self.logger.info("风险控制状态已重置")
    
    def get_risk_status(self) -> Dict[str, Any]:
        """
        获取当前风险状态报告
        
        Returns:
            dict: 风险状态信息
        """
        return {
            'trading_paused': self.trading_paused,
            'emergency_exit_triggered': self.emergency_exit_triggered,
            'consecutive_losses': self.consecutive_losses,
            'consecutive_loss_limit': self.consecutive_loss_limit.value,
            'current_drawdown': self.current_drawdown,
            'drawdown_threshold': self.drawdown_pause_threshold.value,
            'max_equity': self.max_equity,
            'daily_pnl_history': list(self.daily_pnl),
            'trade_history_count': len(self.trade_history),
            'risk_controls_active': self.trading_paused or self.emergency_exit_triggered
        }
    
    # ===========================================
    # 重写入场/出场逻辑集成风险控制
    # ===========================================
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        增强网格交易入场信号 - 集成风险控制
        
        在父类基础上增加风险控制检查：
        1. 交易暂停检查
        2. 连续亏损熔断
        3. 回撤保护
        4. 单日损失限制
        """
        try:
            # 首先调用父类方法获取基础入场信号
            dataframe = super().populate_entry_trend(dataframe, metadata)
            
            # 如果交易被暂停，清除所有入场信号
            if self.trading_paused:
                self.logger.warning("交易已暂停，禁止新入场信号")
                dataframe['enter_long'] = 0
                return dataframe
            
            # 如果紧急退出被触发，清除所有入场信号
            if self.emergency_exit_triggered:
                self.logger.error("紧急退出已触发，禁止新入场信号")
                dataframe['enter_long'] = 0
                return dataframe
            
            # 检查连续亏损熔断
            if self.consecutive_losses >= self.consecutive_loss_limit.value:
                self.logger.warning(f"连续亏损达到熔断条件: {self.consecutive_losses}次，禁止新入场")
                dataframe['enter_long'] = 0
                return dataframe
            
            # 检查回撤水平
            if self.current_drawdown > self.drawdown_pause_threshold.value * 0.8:
                self.logger.warning(f"回撤水平过高: {self.current_drawdown:.4f}，减少入场信号")
                # 在高回撤时，只保留最强的入场信号
                if 'enter_long' in dataframe.columns:
                    # 增加额外的过滤条件
                    enhanced_filter = (
                        (dataframe['rsi'] < 35) &  # 更严格的RSI条件
                        (dataframe['bb_percent'] < 0.3) &  # 更严格的布林带位置
                        (dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 1.2)  # 更强的成交量确认
                    )
                    dataframe.loc[~enhanced_filter, 'enter_long'] = 0
            
            # 增强版入场条件（当启用相关功能时）
            if self.enable_enhanced_position_sizing.value:
                # 基于市场状态的入场过滤
                if self.enable_market_regime_detection.value:
                    current_regime = self.detect_market_regime(dataframe)
                    if current_regime == 'downtrend':
                        # 下跌趋势中更加谨慎
                        self.logger.debug("下跌趋势检测，收紧入场条件")
                        if 'enter_long' in dataframe.columns:
                            conservative_filter = (
                                (dataframe['rsi'] < 30) &  # 极度超卖
                                (dataframe['bb_percent'] < 0.2)  # 更接近下轨
                            )
                            dataframe.loc[~conservative_filter, 'enter_long'] = 0
            
            # 记录入场信号统计
            entry_signals = dataframe['enter_long'].sum() if 'enter_long' in dataframe.columns else 0
            if entry_signals > 0:
                self.logger.info(f"生成{entry_signals}个增强入场信号")
            
            return dataframe
            
        except Exception as e:
            self.logger.error(f"增强入场信号计算错误: {e}")
            # 错误时返回安全的基础信号
            try:
                return super().populate_entry_trend(dataframe, metadata)
            except:
                # 完全失败时返回无信号的数据框
                if 'enter_long' in dataframe.columns:
                    dataframe['enter_long'] = 0
                return dataframe