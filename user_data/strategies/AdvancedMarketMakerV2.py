# Advanced Market Maker Strategy V2 - FreqTrade框架版本
# 基于 freqtrade 框架的高级做市商策略
# 功能：智能订单管理、风险控制、持仓平衡、自动化订单状态管理

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, asdict

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    DecimalParameter,
    IntParameter,
    BooleanParameter,
    informative,
)

import talib.abstract as ta
from technical import qtpylib


@dataclass
class MarketMakerStats:
    """做市商统计数据"""
    total_orders: int = 0
    filled_orders: int = 0
    cancelled_orders: int = 0
    total_volume: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    inventory_imbalance: float = 0.0
    spread_captured: float = 0.0
    risk_score: float = 0.0
    uptime_hours: float = 0.0
    avg_spread_percentage: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AdvancedMarketMakerV2(IStrategy):
    """
    FreqTrade高级做市商策略V2
    
    核心功能：
    1. 智能订单管理 - 利用freqtrade自动订单状态管理
    2. 动态价格调整 - 根据市场波动和库存调整价差
    3. 风险控制系统 - 持仓限制、日亏损限制、波动率控制
    4. 库存平衡机制 - 自动调整买卖比例保持库存平衡
    5. 实时监控统计 - 集成freqUI的完整监控体验
    """
    
    INTERFACE_VERSION = 3

    # 基础设置
    timeframe = '1m'
    process_only_new_candles = True
    startup_candle_count: int = 30
    
    # 止损设置 - 做市商策略使用动态风险控制而非固定止损
    stoploss = -0.99
    
    # ROI设置 - 做市商策略不使用传统ROI
    minimal_roi = {
        "0": 0.99
    }
    
    # 信号配置
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True
    
    # ========== 策略参数 ==========
    
    # 做市核心参数
    spread_ratio = DecimalParameter(
        0.0005, 0.01, default=0.002, space="buy", optimize=True,
        load=True, decimals=4
    )
    min_spread_ratio = DecimalParameter(
        0.0001, 0.005, default=0.001, space="buy", optimize=True,
        load=True, decimals=4
    )
    max_spread_ratio = DecimalParameter(
        0.005, 0.02, default=0.01, space="buy", optimize=True,
        load=True, decimals=4
    )
    
    # 订单大小参数
    base_order_amount_ratio = DecimalParameter(
        0.001, 0.02, default=0.005, space="buy", optimize=True,
        load=True, decimals=3
    )
    
    # 库存管理参数
    inventory_target_ratio = DecimalParameter(
        0.4, 0.6, default=0.5, space="buy", optimize=False,
        load=True, decimals=2
    )
    inventory_rebalance_threshold = DecimalParameter(
        0.1, 0.3, default=0.2, space="buy", optimize=True,
        load=True, decimals=2
    )
    
    # 风险控制参数
    max_position_ratio = DecimalParameter(
        0.1, 0.5, default=0.3, space="buy", optimize=False,
        load=True, decimals=2
    )
    max_daily_loss_ratio = DecimalParameter(
        0.01, 0.1, default=0.05, space="buy", optimize=False,
        load=True, decimals=3
    )
    volatility_threshold = DecimalParameter(
        0.02, 0.1, default=0.05, space="buy", optimize=True,
        load=True, decimals=3
    )
    
    # 价格更新控制
    price_update_threshold = DecimalParameter(
        0.0001, 0.002, default=0.0005, space="buy", optimize=True,
        load=True, decimals=4
    )
    
    # 流动性控制
    min_volume_ratio = DecimalParameter(
        0.3, 1.5, default=0.7, space="buy", optimize=True,
        load=True, decimals=2
    )
    
    # ========== 初始化 ==========
    
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        
        # 统计数据
        self.market_maker_stats = MarketMakerStats()
        self.strategy_start_time = datetime.now(timezone.utc)
        
        # 价格记录
        self.last_bid_prices = {}
        self.last_ask_prices = {}
        self.last_mid_prices = {}
        
        # 风险状态
        self.risk_limit_reached = False
        self.daily_start_balance = None
        self.daily_loss_count = 0
        
        # 数据持久化键名
        self.MM_STATS_KEY = "mm_stats_v2"
        self.MM_RISK_KEY = "mm_risk_v2"
        
        # 日志设置
        self.logger = logging.getLogger(__name__)
        
        # 加载历史统计数据
        self._load_persistent_data()
    
    def _load_persistent_data(self):
        """加载持久化数据"""
        try:
            # 检查是否有数据提供者
            if hasattr(self, 'dp') and hasattr(self.dp, 'storage'):
                # 加载统计数据
                stats_data = self.dp.storage.get(self.MM_STATS_KEY, {})
                if stats_data:
                    self.market_maker_stats = MarketMakerStats(**stats_data)
                
                # 加载风险状态
                risk_data = self.dp.storage.get(self.MM_RISK_KEY, {})
                self.risk_limit_reached = risk_data.get('risk_limit_reached', False)
                self.daily_loss_count = risk_data.get('daily_loss_count', 0)
            else:
                self.logger.debug("数据提供者不可用，跳过持久化数据加载")
            
        except Exception as e:
            self.logger.warning(f"加载持久化数据失败: {e}")
    
    def _save_persistent_data(self):
        """保存持久化数据"""
        try:
            # 检查是否有数据提供者
            if hasattr(self, 'dp') and hasattr(self.dp, 'storage'):
                # 保存统计数据
                self.dp.storage.set(self.MM_STATS_KEY, self.market_maker_stats.to_dict())
                
                # 保存风险状态
                risk_data = {
                    'risk_limit_reached': self.risk_limit_reached,
                    'daily_loss_count': self.daily_loss_count,
                    'last_update': datetime.now(timezone.utc).isoformat()
                }
                self.dp.storage.set(self.MM_RISK_KEY, risk_data)
            else:
                self.logger.debug("数据提供者不可用，跳过持久化数据保存")
            
        except Exception as e:
            self.logger.error(f"保存持久化数据失败: {e}")
    
    # ========== 技术指标 ==========
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        填充做市商相关的技术指标
        """
        # 基础价格指标
        dataframe['hl2'] = (dataframe['high'] + dataframe['low']) / 2
        dataframe['hlc3'] = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
        dataframe['ohlc4'] = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4
        
        # 波动率指标
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['volatility'] = dataframe['close'].rolling(window=20).std()
        dataframe['volatility_ratio'] = dataframe['volatility'] / dataframe['volatility'].rolling(window=50).mean()
        
        # 成交量指标
        dataframe['volume_sma'] = dataframe['volume'].rolling(window=20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_sma']
        dataframe['volume_weighted_price'] = (dataframe['volume'] * dataframe['close']).rolling(window=10).sum() / dataframe['volume'].rolling(window=10).sum()
        
        # 价格变化指标
        dataframe['price_change'] = dataframe['close'].pct_change()
        dataframe['price_change_abs'] = dataframe['price_change'].abs()
        dataframe['price_momentum'] = dataframe['close'] / dataframe['close'].shift(10) - 1
        
        # 流动性和价差估计
        dataframe['bid_ask_spread_estimate'] = (dataframe['high'] - dataframe['low']) / dataframe['close']
        dataframe['liquidity_score'] = dataframe['volume'] / (dataframe['high'] - dataframe['low'] + 1e-8)
        
        # 市场状态指标 - 布林带
        bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)
        dataframe['bb_upperband'] = bollinger['upperband'] 
        dataframe['bb_middleband'] = bollinger['middleband']
        dataframe['bb_lowerband'] = bollinger['lowerband']
        dataframe['bb_width'] = (dataframe['bb_upperband'] - dataframe['bb_lowerband']) / dataframe['bb_middleband']
        
        # RSI 用于判断超买超卖
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        
        # 趋势强度
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        
        return dataframe
    
    # ========== 入场信号 ==========
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        做市商入场信号逻辑
        """
        pair = metadata['pair']
        
        # 基础市场条件
        good_market_conditions = (
            (dataframe['volume_ratio'] >= self.min_volume_ratio.value) &  # 充足流动性
            (dataframe['volatility_ratio'] < 2.0) &  # 避免极端波动
            (dataframe['price_change_abs'] < self.volatility_threshold.value) &  # 价格变化可控
            (dataframe['adx'] < 30)  # 避免强趋势市场
        )
        
        # 检查库存状态
        inventory_imbalance = self.get_inventory_imbalance(pair)
        
        # 创建库存平衡条件（布尔数组）
        need_buy_rebalance = pd.Series([inventory_imbalance < -self.inventory_rebalance_threshold.value] * len(dataframe), index=dataframe.index)
        need_sell_rebalance = pd.Series([inventory_imbalance > self.inventory_rebalance_threshold.value] * len(dataframe), index=dataframe.index)
        
        # 正常做市条件
        normal_market_making = (
            good_market_conditions &
            (abs(inventory_imbalance) <= self.inventory_rebalance_threshold.value)
        )
        
        # 风险限制条件
        risk_ok = pd.Series([not self.risk_limit_reached] * len(dataframe), index=dataframe.index)
        
        # 买入信号：正常做市 或 需要增加库存
        dataframe.loc[
            (normal_market_making | need_buy_rebalance) & risk_ok,
            'enter_long'
        ] = 1
        
        # 买入标签
        dataframe.loc[normal_market_making & risk_ok, 'enter_tag'] = 'mm_buy'
        dataframe.loc[need_buy_rebalance & risk_ok, 'enter_tag'] = 'rebalance_buy'
        
        # 卖出信号：正常做市 或 需要减少库存
        dataframe.loc[
            (normal_market_making | need_sell_rebalance) & risk_ok,
            'enter_short'  
        ] = 1
        
        return dataframe
    
    # ========== 出场信号 ==========
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        做市商出场信号逻辑
        """
        # 风险出场条件
        risk_exit_conditions = (
            (dataframe['volatility_ratio'] > 3.0) |  # 极端波动
            (dataframe['volume_ratio'] < 0.3) |  # 流动性不足
            (dataframe['price_change_abs'] > self.volatility_threshold.value * 2) |  # 剧烈价格变化
            (dataframe['adx'] > 50) |  # 强趋势市场
            self.risk_limit_reached  # 达到风险限制
        )
        
        dataframe.loc[risk_exit_conditions, 'exit_long'] = 1
        dataframe.loc[risk_exit_conditions, 'exit_short'] = 1
        dataframe.loc[risk_exit_conditions, 'exit_tag'] = 'risk_exit'
        
        return dataframe
    
    # ========== 自定义价格逻辑 ==========
    
    def custom_entry_price(self, pair: str, trade: Optional[Trade], current_time: datetime, 
                          proposed_rate: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        """
        自定义入场价格 - 做市商的核心定价逻辑
        """
        try:
            dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
            
            if dataframe.empty:
                return proposed_rate
                
            latest = dataframe.iloc[-1]
            current_price = latest['close']
            atr = latest['atr']
            volatility = latest['volatility']
            bb_width = latest['bb_width']
            
            # 动态价差计算
            base_spread = current_price * self.spread_ratio.value
            volatility_spread = max(atr * 0.5, volatility * 2)
            bb_spread = current_price * bb_width * 0.1
            
            # 综合价差
            dynamic_spread = max(
                base_spread,
                min(volatility_spread, current_price * self.max_spread_ratio.value),
                current_price * self.min_spread_ratio.value
            ) + bb_spread
            
            # 库存调整
            inventory_imbalance = self.get_inventory_imbalance(pair)
            if side == 'long' and inventory_imbalance < -self.inventory_rebalance_threshold.value:
                # 需要增加库存，提高买价
                dynamic_spread *= 0.8
            elif side == 'short' and inventory_imbalance > self.inventory_rebalance_threshold.value:
                # 需要减少库存，降低卖价
                dynamic_spread *= 0.8
            
            # 计算最终价格
            if side == 'long':
                entry_price = current_price - dynamic_spread / 2
                self.last_bid_prices[pair] = entry_price
            else:
                entry_price = current_price + dynamic_spread / 2
                self.last_ask_prices[pair] = entry_price
            
            # 记录中间价
            self.last_mid_prices[pair] = current_price
            
            # 更新统计
            self.market_maker_stats.avg_spread_percentage = dynamic_spread / current_price * 100
            
            return entry_price
            
        except Exception as e:
            self.logger.error(f"自定义入场价格计算失败 {pair}: {e}")
            return proposed_rate
    
    def custom_exit_price(self, pair: str, trade: Trade, current_time: datetime, 
                         proposed_rate: float, current_profit: float, exit_tag: Optional[str], **kwargs) -> float:
        """
        自定义出场价格 - 快速平仓使用市价
        """
        if exit_tag == 'risk_exit':
            # 风险出场使用市价快速成交
            return proposed_rate
        else:
            # 正常出场可以使用略好的价格
            return proposed_rate
    
    # ========== 仓位大小控制 ==========
    
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                           proposed_stake: float, min_stake: Optional[float], max_stake: float,
                           entry_tag: Optional[str], side: str, **kwargs) -> float:
        """
        自定义仓位大小 - 基于风险控制和库存平衡
        """
        # 风险检查
        if self.risk_limit_reached:
            return 0
        
        # 获取可用资金
        total_stake = self.wallets.get_total_stake_amount()
        if total_stake <= 0:
            return 0
        
        # 基础订单大小
        base_stake = total_stake * self.base_order_amount_ratio.value
        
        # 根据库存状态调整
        inventory_imbalance = self.get_inventory_imbalance(pair)
        
        if entry_tag == 'rebalance_buy' and inventory_imbalance < -self.inventory_rebalance_threshold.value:
            # 库存不足，增加买入量
            adjusted_stake = base_stake * 1.5
        elif entry_tag == 'rebalance_sell' and inventory_imbalance > self.inventory_rebalance_threshold.value:
            # 库存过多，增加卖出量
            adjusted_stake = base_stake * 1.5
        else:
            adjusted_stake = base_stake
        
        # 应用最大仓位限制
        max_position_stake = total_stake * self.max_position_ratio.value
        current_positions_value = sum(trade.stake_amount for trade in Trade.get_trades_proxy(is_open=True))
        available_position_stake = max_position_stake - current_positions_value
        
        final_stake = min(adjusted_stake, available_position_stake, max_stake)
        
        return max(final_stake, min_stake) if min_stake else final_stake
    
    # ========== 辅助方法 ==========
    
    def get_inventory_imbalance(self, pair: str) -> float:
        """
        计算库存失衡度
        返回值: -1到1之间，负数表示库存不足(需要买入)，正数表示库存过多(需要卖出)
        """
        try:
            trades = Trade.get_trades_proxy(pair=pair, is_open=True)
            
            if not trades:
                return 0.0
            
            long_amount = sum(trade.amount for trade in trades if not trade.is_short)
            short_amount = sum(trade.amount for trade in trades if trade.is_short)
            total_amount = long_amount + short_amount
            
            if total_amount == 0:
                return 0.0
            
            # 计算相对于目标比例的失衡
            current_long_ratio = long_amount / total_amount
            target_ratio = self.inventory_target_ratio.value
            
            # 失衡度：正值表示多头过多，负值表示多头不足
            imbalance = (current_long_ratio - target_ratio) / target_ratio
            
            return max(-1.0, min(1.0, imbalance))
        except Exception as e:
            # 在测试或初始化环境中，返回默认值
            self.logger.debug(f"无法获取交易数据，使用默认库存失衡值: {e}")
            return 0.0
    
    def check_risk_limits(self, pair: str) -> bool:
        """检查风险限制"""
        try:
            # 检查是否有wallets对象
            if not hasattr(self, 'wallets') or self.wallets is None:
                return True  # 测试环境默认允许交易
            
            # 检查日亏损限制
            if self.daily_start_balance is None:
                self.daily_start_balance = self.wallets.get_total_stake_amount()
            
            current_balance = self.wallets.get_total_stake_amount()
            daily_loss_ratio = (self.daily_start_balance - current_balance) / self.daily_start_balance
            
            if daily_loss_ratio > self.max_daily_loss_ratio.value:
                self.logger.warning(f"达到日亏损限制: {daily_loss_ratio:.2%}")
                self.risk_limit_reached = True
                self.daily_loss_count += 1
                return False
            
            # 检查持仓集中度
            total_stake = self.wallets.get_total_stake_amount()
            try:
                current_positions_value = sum(trade.stake_amount for trade in Trade.get_trades_proxy(is_open=True))
                position_ratio = current_positions_value / total_stake if total_stake > 0 else 0
                
                if position_ratio > self.max_position_ratio.value * 1.2:  # 允许20%的缓冲
                    self.logger.warning(f"持仓比例过高: {position_ratio:.2%}")
                    return False
            except:
                # 无法获取交易数据时跳过持仓检查
                pass
            
            return True
            
        except Exception as e:
            self.logger.debug(f"风险检查失败: {e}")
            return True  # 默认允许交易
    
    def update_statistics(self):
        """更新策略统计信息"""
        try:
            # 更新运行时间
            runtime = datetime.now(timezone.utc) - self.strategy_start_time
            self.market_maker_stats.uptime_hours = runtime.total_seconds() / 3600
            
            # 统计交易信息
            all_trades = Trade.get_trades_proxy()
            if all_trades:
                filled_trades = [t for t in all_trades if t.is_open == False]
                self.market_maker_stats.filled_orders = len(filled_trades)
                self.market_maker_stats.total_volume = sum(t.amount for t in filled_trades)
                
                if filled_trades:
                    self.market_maker_stats.realized_pnl = sum(
                        t.close_profit_abs for t in filled_trades if t.close_profit_abs
                    )
            
            # 计算未实现盈亏
            open_trades = Trade.get_trades_proxy(is_open=True)
            if open_trades:
                self.market_maker_stats.unrealized_pnl = sum(
                    t.unrealized_profit for t in open_trades
                )
            
            # 保存数据
            self._save_persistent_data()
            
        except Exception as e:
            self.logger.error(f"更新统计信息失败: {e}")
    
    # ========== 生命周期回调 ==========
    
    def bot_start(self, **kwargs) -> None:
        """策略启动时的初始化"""
        self.logger.info(f"高级做市商策略V2启动 - {datetime.now()}")
        self.strategy_start_time = datetime.now(timezone.utc)
        self.daily_start_balance = self.wallets.get_total_stake_amount()
    
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                           time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                           side: str, **kwargs) -> bool:
        """确认交易入场"""
        # 最后的风险检查
        if not self.check_risk_limits(pair):
            self.logger.warning(f"风险限制阻止入场: {pair} {side} {entry_tag}")
            return False
        
        self.market_maker_stats.total_orders += 1
        return True
    
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                          rate: float, time_in_force: str, exit_reason: str, current_time: datetime,
                          **kwargs) -> bool:
        """确认交易出场"""
        return True
    
    def on_trade_close(self, trade: Trade, **kwargs) -> None:
        """交易关闭时的处理"""
        self.logger.info(f"交易关闭: {trade.pair} {trade.trade_direction} "
                        f"盈亏: {trade.close_profit_abs:.4f} 持续时间: {trade.close_date - trade.open_date}")
        
        # 更新统计
        self.update_statistics()
        
        # 计算价差捕获
        if hasattr(trade, 'open_rate') and hasattr(trade, 'close_rate'):
            if trade.pair in self.last_mid_prices:
                mid_price = self.last_mid_prices[trade.pair]
                if trade.is_short:
                    spread_captured = trade.open_rate - mid_price
                else:
                    spread_captured = mid_price - trade.open_rate
                self.market_maker_stats.spread_captured += spread_captured * trade.amount