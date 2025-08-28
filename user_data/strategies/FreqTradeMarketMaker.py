# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, asdict
from enum import Enum

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


class MarketMakerOrderType(Enum):
    """做市订单类型"""
    BUY_ORDER = "buy"
    SELL_ORDER = "sell"
    REBALANCE_BUY = "rebalance_buy"
    REBALANCE_SELL = "rebalance_sell"


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
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FreqTradeMarketMaker(IStrategy):
    """
    FreqTrade做市商策略
    
    核心功能：
    1. 智能订单管理和价格调整
    2. 风险控制和持仓管理
    3. 实时PnL计算和风险评估
    4. 完整的数据持久化和监控
    5. 与FreqUI的完美集成
    """
    
    INTERFACE_VERSION = 3

    # 策略参数
    timeframe = '1m'
    
    # 风险控制参数
    max_position_ratio = DecimalParameter(0.1, 0.5, default=0.3, space="buy", optimize=False)
    max_daily_loss = DecimalParameter(50.0, 500.0, default=100.0, space="buy", optimize=False)
    inventory_target_ratio = DecimalParameter(0.4, 0.6, default=0.5, space="buy", optimize=False)
    
    # 做市参数
    spread_ratio = DecimalParameter(0.001, 0.01, default=0.002, space="buy", optimize=True)
    order_amount_ratio = DecimalParameter(0.001, 0.01, default=0.005, space="buy", optimize=True)
    price_update_threshold = DecimalParameter(0.0001, 0.001, default=0.0005, space="buy", optimize=True)
    
    # 止损设置
    stoploss = -0.99  # 做市商策略不使用传统止损
    
    # 买卖信号配置
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True
    
    # ROI表 - 做市商策略不使用传统ROI
    minimal_roi = {
        "0": 0.99
    }

    # 启动设置和日志
    startup_candle_count: int = 30
    
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        
        # 做市商状态管理
        self.market_maker_stats = MarketMakerStats()
        self.last_bid_price = {}
        self.last_ask_price = {}
        self.position_start_time = {}
        self.daily_pnl_start = {}
        
        # 风险管理
        self.risk_limit_reached = False
        self.inventory_imbalance_threshold = 0.2
        
        # 数据持久化键名
        self.MM_STATS_KEY = "market_maker_stats"
        self.MM_POSITION_KEY = "market_maker_position"
        self.MM_RISK_KEY = "market_maker_risk"
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
    
    def informative_pairs(self):
        """定义信息对 - 做市商通常只需要主要交易对"""
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        填充技术指标
        做市商策略主要关注价格波动和流动性指标
        """
        # 基础价格指标
        dataframe['hl2'] = (dataframe['high'] + dataframe['low']) / 2
        dataframe['hlc3'] = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
        
        # 波动率指标
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['volatility'] = dataframe['close'].rolling(20).std()
        
        # 成交量指标
        dataframe['volume_sma'] = dataframe['volume'].rolling(20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_sma']
        
        # 价格差值
        dataframe['price_change'] = dataframe['close'].pct_change()
        dataframe['price_change_abs'] = dataframe['price_change'].abs()
        
        # 流动性评估
        dataframe['spread_estimate'] = (dataframe['high'] - dataframe['low']) / dataframe['close']
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        入场信号 - 做市商策略的入场逻辑
        """
        pair = metadata['pair']
        
        # 做市商入场条件：
        # 1. 市场波动率适中（不太高不太低）
        # 2. 成交量足够
        # 3. 没有达到风险限制
        # 4. 需要平衡库存时
        
        entry_conditions = (
            (dataframe['volatility'] > dataframe['volatility'].rolling(50).quantile(0.2)) &
            (dataframe['volatility'] < dataframe['volatility'].rolling(50).quantile(0.8)) &
            (dataframe['volume_ratio'] > 0.5) &
            (dataframe['price_change_abs'] < 0.05)  # 避免极端价格变动
        )
        
        # 检查是否需要库存平衡
        inventory_imbalance = self.get_inventory_imbalance(pair)
        need_buy_rebalance = inventory_imbalance < -self.inventory_imbalance_threshold
        need_sell_rebalance = inventory_imbalance > self.inventory_imbalance_threshold
        
        # 买入信号：正常做市或需要增加库存
        dataframe.loc[
            entry_conditions | need_buy_rebalance,
            'enter_long'
        ] = 1
        
        # 卖出信号：正常做市或需要减少库存  
        dataframe.loc[
            entry_conditions | need_sell_rebalance,
            'enter_short'
        ] = 1
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        出场信号 - 做市商策略的出场逻辑
        """
        pair = metadata['pair']
        
        # 做市商出场条件：
        # 1. 风险限制达到
        # 2. 市场流动性不足
        # 3. 极端波动
        
        exit_conditions = (
            (dataframe['volume_ratio'] < 0.3) |  # 流动性不足
            (dataframe['price_change_abs'] > 0.1) |  # 极端波动
            self.risk_limit_reached  # 风险限制
        )
        
        dataframe.loc[exit_conditions, 'exit_long'] = 1
        dataframe.loc[exit_conditions, 'exit_short'] = 1
        
        return dataframe

    def custom_entry_price(self, pair: str, trade: Optional[Trade], current_time: datetime, proposed_rate: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        """
        自定义入场价格 - 做市商的核心定价逻辑
        """
        dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        
        if dataframe.empty:
            return proposed_rate
            
        latest = dataframe.iloc[-1]
        current_price = latest['close']
        atr = latest['atr']
        
        # 计算做市价差
        spread = max(current_price * self.spread_ratio.value, atr * 0.5)
        
        if side == 'long':
            # 买单：在当前价格下方挂单
            entry_price = current_price - spread / 2
        else:
            # 卖单：在当前价格上方挂单
            entry_price = current_price + spread / 2
        
        # 记录价格以便后续比较
        if side == 'long':
            self.last_bid_price[pair] = entry_price
        else:
            self.last_ask_price[pair] = entry_price
        
        return entry_price

    def custom_exit_price(self, pair: str, trade: Trade, current_time: datetime, proposed_rate: float, current_profit: float, exit_tag: Optional[str], **kwargs) -> float:
        """
        自定义出场价格
        """
        # 做市商策略通常使用市价出场以快速平仓
        return proposed_rate

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float, proposed_stake: float, min_stake: Optional[float], max_stake: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        """
        自定义仓位大小 - 根据库存状态和风险控制调整
        """
        # 检查风险限制
        if self.risk_limit_reached:
            return 0
        
        # 获取当前库存状态
        inventory_imbalance = self.get_inventory_imbalance(pair)
        
        # 基础仓位大小
        base_stake = self.wallets.get_total_stake_amount() * self.order_amount_ratio.value
        
        # 根据库存失衡调整仓位
        if side == 'long' and inventory_imbalance < -self.inventory_imbalance_threshold:
            # 库存过少，增加买入量
            adjusted_stake = base_stake * 1.5
        elif side == 'short' and inventory_imbalance > self.inventory_imbalance_threshold:
            # 库存过多，增加卖出量
            adjusted_stake = base_stake * 1.5
        else:
            adjusted_stake = base_stake
        
        # 确保在最大仓位限制内
        max_position_stake = self.wallets.get_total_stake_amount() * self.max_position_ratio.value
        
        return min(adjusted_stake, max_position_stake, max_stake)

    def get_inventory_imbalance(self, pair: str) -> float:
        """
        计算库存失衡度
        返回值: -1到1之间，负数表示库存不足，正数表示库存过多
        """
        trades = Trade.get_trades_proxy(pair=pair, is_open=True)
        
        if not trades:
            return 0.0
        
        long_amount = sum(trade.amount for trade in trades if trade.is_short is False)
        short_amount = sum(trade.amount for trade in trades if trade.is_short is True)
        total_amount = long_amount + short_amount
        
        if total_amount == 0:
            return 0.0
        
        # 计算失衡度：(long - short) / total
        return (long_amount - short_amount) / total_amount

    def update_market_maker_stats(self, pair: str):
        """更新做市商统计数据"""
        trades = Trade.get_trades_proxy(pair=pair)
        
        # 基础统计
        self.market_maker_stats.total_orders = len(trades)
        self.market_maker_stats.filled_orders = len([t for t in trades if not t.is_open])
        
        # PnL计算
        closed_trades = [t for t in trades if not t.is_open]
        if closed_trades:
            self.market_maker_stats.realized_pnl = sum(t.close_profit_abs or 0 for t in closed_trades)
            self.market_maker_stats.total_volume = sum(t.amount * t.open_rate for t in closed_trades)
        
        # 未实现PnL
        open_trades = [t for t in trades if t.is_open]
        if open_trades:
            self.market_maker_stats.unrealized_pnl = sum(t.unrealized_profit or 0 for t in open_trades)
        
        # 库存失衡
        self.market_maker_stats.inventory_imbalance = self.get_inventory_imbalance(pair)
        
        # 风险评分（简化版）
        daily_pnl = self.market_maker_stats.realized_pnl + self.market_maker_stats.unrealized_pnl
        max_loss = self.max_daily_loss.value
        self.market_maker_stats.risk_score = min(abs(daily_pnl) / max_loss, 1.0) if max_loss > 0 else 0

    def check_risk_limits(self, pair: str) -> bool:
        """检查风险限制"""
        # 更新统计数据
        self.update_market_maker_stats(pair)
        
        # 检查日损失限制
        daily_pnl = self.market_maker_stats.realized_pnl + self.market_maker_stats.unrealized_pnl
        if daily_pnl < -self.max_daily_loss.value:
            self.logger.warning(f"达到日亏损限制: {daily_pnl:.2f}")
            self.risk_limit_reached = True
            return False
        
        # 检查库存失衡
        if abs(self.market_maker_stats.inventory_imbalance) > 0.8:
            self.logger.warning(f"库存严重失衡: {self.market_maker_stats.inventory_imbalance:.3f}")
            # 不直接停止，而是调整策略
        
        return True

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, current_time: datetime, entry_tag: Optional[str], side: str, **kwargs) -> bool:
        """
        确认交易入场 - 最后的风险检查
        """
        # 执行风险检查
        if not self.check_risk_limits(pair):
            return False
        
        # 检查价格是否仍然有效（避免过时价格）
        dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if not dataframe.empty:
            current_price = dataframe.iloc[-1]['close']
            price_diff = abs(rate - current_price) / current_price
            
            # 如果价格偏差过大，拒绝交易
            if price_diff > self.price_update_threshold.value * 3:
                self.logger.info(f"价格偏差过大，拒绝交易: {price_diff:.4f}")
                return False
        
        return True

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float, rate: float, time_in_force: str, exit_reason: str, current_time: datetime, **kwargs) -> bool:
        """
        确认交易出场
        """
        # 做市商策略通常快速出场
        return True

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        """
        杠杆设置 - 做市商策略通常使用低杠杆或不使用杠杆
        """
        return 1.0

    def custom_trade_timeout(self, pair: str, trade: Trade, current_time: datetime, **kwargs) -> bool:
        """
        自定义交易超时 - 做市商订单需要及时更新
        """
        # 如果订单挂单时间超过5分钟且没有部分成交，考虑取消重挂
        if trade.open_order_id:
            time_since_open = current_time.timestamp() - trade.open_date.timestamp()
            if time_since_open > 300:  # 5分钟
                return True
        
        return False

    def on_trade_closed(self, trade: Trade, **kwargs):
        """
        交易关闭时的回调 - 更新统计和持久化数据
        """
        pair = trade.pair
        
        # 更新统计数据
        self.update_market_maker_stats(pair)
        
        # 保存统计数据到数据库
        trade.set_custom_data(self.MM_STATS_KEY, self.market_maker_stats.to_dict())
        
        # 记录日志
        self.logger.info(f"交易关闭: {pair}, PnL: {trade.close_profit_abs:.2f}, "
                        f"总实现PnL: {self.market_maker_stats.realized_pnl:.2f}")

    def on_order_fill(self, trade: Trade, order: Order, **kwargs):
        """
        订单成交回调 - 更新做市商状态
        """
        pair = trade.pair
        
        # 更新统计
        self.market_maker_stats.filled_orders += 1
        
        # 记录成交信息
        self.logger.info(f"订单成交: {pair}, 方向: {order.side}, "
                        f"数量: {order.filled}, 价格: {order.average}")
        
        # 如果是部分成交，记录相关信息
        if order.status == 'open':
            remaining = order.amount - order.filled
            self.logger.info(f"部分成交，剩余数量: {remaining}")

    def bot_start(self):
        """
        策略启动时的初始化
        """
        self.logger.info("FreqTrade做市商策略启动")
        
        # 重置风险状态
        self.risk_limit_reached = False
        
        # 初始化统计数据
        for pair in self.dp.current_whitelist():
            self.update_market_maker_stats(pair)
            self.daily_pnl_start[pair] = self.market_maker_stats.realized_pnl

    def bot_loop_start(self):
        """
        每次策略循环开始时的检查
        """
        current_time = datetime.now(timezone.utc)
        
        # 每小时重置风险限制检查（如果不是因为严重问题）
        if hasattr(self, '_last_risk_reset'):
            if (current_time - self._last_risk_reset).total_seconds() > 3600:
                if not self.is_severe_risk_event():
                    self.risk_limit_reached = False
                    self._last_risk_reset = current_time
        else:
            self._last_risk_reset = current_time
        
        # 定期更新统计数据
        for pair in self.dp.current_whitelist():
            self.update_market_maker_stats(pair)

    def is_severe_risk_event(self) -> bool:
        """
        判断是否发生严重风险事件
        """
        # 如果亏损超过最大限制的150%，认为是严重事件
        daily_pnl = self.market_maker_stats.realized_pnl + self.market_maker_stats.unrealized_pnl
        return daily_pnl < -self.max_daily_loss.value * 1.5

    def version(self) -> str:
        """
        策略版本
        """
        return "1.0.0"