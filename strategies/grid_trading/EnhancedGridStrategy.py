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
    
    # 趋势识别阈值
    trend_detection_threshold = DecimalParameter(
        low=20, high=40, default=28, 
        space="buy", optimize=True, load=True,
        description="ADX趋势强度识别阈值"
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
    
    # 动态止损增强系数
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
        市场状态识别（预留接口）
        
        此方法将在后续任务中实现具体的市场状态识别逻辑
        
        Args:
            dataframe: K线数据
            
        Returns:
            str: 市场状态（'uptrend', 'downtrend', 'consolidation', 'neutral'）
        """
        if len(dataframe) == 0:
            return "neutral"
        
        # 获取最新的市场状态
        if 'market_regime' in dataframe.columns:
            return dataframe['market_regime'].iloc[-1]
        
        return "neutral"

    def calculate_enhanced_position_size(self, pair: str, current_time: datetime, 
                                       current_rate: float, proposed_stake: float,
                                       **kwargs) -> float:
        """
        增强仓位计算（预留接口）
        
        此方法将在后续任务中实现具体的智能仓位管理逻辑
        
        Args:
            pair: 交易对
            current_time: 当前时间
            current_rate: 当前价格
            proposed_stake: 建议仓位
            **kwargs: 其他参数
            
        Returns:
            float: 调整后的仓位大小
        """
        # 当前实现：调用父类逻辑，后续将增强
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
        增强止损计算（预留接口）
        
        此方法将在后续任务中实现具体的增强止损逻辑
        
        Args:
            pair: 交易对
            trade: 交易对象
            current_time: 当前时间
            current_rate: 当前价格
            current_profit: 当前利润
            **kwargs: 其他参数
            
        Returns:
            float: 止损比例
        """
        # 当前实现：调用父类逻辑并应用增强系数，后续将进一步增强
        base_stoploss = super().custom_stoploss(
            pair=pair,
            trade=trade,
            current_time=current_time,
            current_rate=current_rate,
            current_profit=current_profit,
            **kwargs
        )
        
        # 应用增强系数
        enhanced_stoploss = base_stoploss * self.enhanced_stoploss_factor.value
        
        # 确保在合理范围内
        return max(enhanced_stoploss, -0.12)  # 最大止损12%

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
        """
        return self.calculate_enhanced_stoploss(
            pair=pair,
            trade=trade,
            current_time=current_time,
            current_rate=current_rate,
            current_profit=current_profit,
            **kwargs
        )