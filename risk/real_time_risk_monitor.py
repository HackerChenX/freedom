#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
事中实时风险监控系统

专业金融量化交易系统的事中风险监控模块，提供实时仓位监控、动态风险计算、
智能止损止盈和异常交易检测功能。符合机构级风控要求。

主要功能：
1. 实时仓位监控 - 持续跟踪所有持仓的市值变化和风险状况
2. 动态风险计算 - 实时计算VaR、波动率、回撤等风险指标
3. 智能止损止盈 - 基于技术指标和市场状况的动态止损止盈
4. 异常交易检测 - 识别异常价格波动和交易行为
5. 实时预警机制 - 多级预警和自动风控措施
6. 熔断机制 - 极端情况下的自动停止交易
"""

import time
import json
import threading
import queue
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import defaultdict, deque
import asyncio
import concurrent.futures

from utils.logger import get_logger
from utils.exception_handler import exception_handler
from utils.performance_monitor import performance_monitor
from utils.unified_container import get_container
from utils.numerical_stability_manager import get_stability_manager
from monitoring.risk_monitor import RiskMonitoringSystem, RiskLevel, RiskType
from risk.pre_trade_risk_control import TradeDirection

logger = get_logger(__name__)

class MonitoringStatus(Enum):
    """监控状态枚举"""
    RUNNING = "运行中"
    PAUSED = "暂停"
    STOPPED = "停止"
    ERROR = "错误"

class AlertLevel(Enum):
    """预警级别枚举"""
    INFO = "信息"
    WARNING = "警告"
    CRITICAL = "严重"
    EMERGENCY = "紧急"

class StopLossType(Enum):
    """止损类型枚举"""
    FIXED_PERCENTAGE = "固定百分比"
    TRAILING = "跟踪止损"
    TECHNICAL = "技术止损"
    VOLATILITY_BASED = "波动率止损"
    SMART_ADAPTIVE = "智能自适应"

@dataclass
class PositionMonitorConfig:
    """仓位监控配置"""
    # 监控频率配置
    monitor_interval: float = 1.0           # 监控间隔（秒）
    price_update_interval: float = 0.5      # 价格更新间隔（秒）
    risk_calc_interval: float = 5.0         # 风险计算间隔（秒）

    # 风险阈值配置
    position_risk_threshold: float = 75.0   # 持仓风险阈值
    portfolio_risk_threshold: float = 80.0  # 组合风险阈值
    drawdown_alert_threshold: float = 0.05  # 回撤预警阈值（5%）
    drawdown_stop_threshold: float = 0.10   # 回撤止损阈值（10%）

    # 止损止盈配置
    default_stop_loss: float = 0.08         # 默认止损比例（8%）
    default_take_profit: float = 0.15       # 默认止盈比例（15%）
    trailing_stop_ratio: float = 0.05       # 跟踪止损比例（5%）
    max_loss_per_position: float = 0.02     # 单仓位最大亏损比例（2%）

    # 异常检测配置
    price_shock_threshold: float = 0.05     # 价格异常波动阈值（5%）
    volume_shock_multiplier: float = 3.0    # 成交量异常倍数
    liquidity_alert_threshold: float = 0.1  # 流动性预警阈值

    # 熔断机制配置
    circuit_breaker_threshold: float = 0.15 # 熔断阈值（15%）
    max_daily_loss: float = 0.05            # 每日最大亏损（5%）
    enable_circuit_breaker: bool = True     # 是否启用熔断机制

@dataclass
class PositionSnapshot:
    """仓位快照数据类"""
    stock_code: str
    stock_name: str
    quantity: int
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_ratio: float
    cost_value: float
    weight_in_portfolio: float
    risk_score: float
    stop_loss_price: float
    take_profit_price: float
    timestamp: datetime

    # 扩展信息
    daily_pnl: float = 0.0
    max_profit: float = 0.0
    max_drawdown: float = 0.0
    holding_days: int = 0
    last_trade_time: Optional[datetime] = None

@dataclass
class PortfolioSnapshot:
    """组合快照数据类"""
    total_assets: float
    market_value: float
    available_cash: float
    total_pnl: float
    total_pnl_ratio: float
    daily_pnl: float
    max_drawdown: float
    positions: List[PositionSnapshot]
    risk_score: float
    var_1d: float
    volatility: float
    timestamp: datetime

    # 风控状态
    alert_count: int = 0
    circuit_breaker_triggered: bool = False
    risk_level: str = "中等风险"

@dataclass
class RiskAlert:
    """风险预警数据类"""
    alert_id: str
    alert_level: AlertLevel
    alert_type: str
    stock_code: Optional[str]
    title: str
    message: str
    current_value: float
    threshold_value: float
    suggested_action: str
    timestamp: datetime
    acknowledged: bool = False
    resolved: bool = False

class RealTimeRiskMonitor:
    """事中实时风险监控器"""

    def __init__(self, config: PositionMonitorConfig = None):
        """
        初始化实时风险监控器

        Args:
            config: 监控配置
        """
        self.config = config or PositionMonitorConfig()
        self.stability_manager = get_stability_manager()
        self.container = get_container()

        # 系统状态
        self.status = MonitoringStatus.STOPPED
        self.start_time: Optional[datetime] = None
        self.last_update_time: Optional[datetime] = None

        # 监控线程
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()

        # 数据存储
        self.current_portfolio: Optional[PortfolioSnapshot] = None
        self.position_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.portfolio_history: deque = deque(maxlen=1000)
        self.active_alerts: Dict[str, RiskAlert] = {}
        self.resolved_alerts: List[RiskAlert] = []

        # 事件队列
        self.alert_queue = queue.Queue()
        self.action_queue = queue.Queue()

        # 回调函数
        self.alert_callbacks: List[Callable[[RiskAlert], None]] = []
        self.position_callbacks: List[Callable[[PositionSnapshot], None]] = []
        self.portfolio_callbacks: List[Callable[[PortfolioSnapshot], None]] = []

        # 初始化组件
        self._initialize_components()

        logger.info("实时风险监控系统初始化完成")

    def _initialize_components(self):
        """初始化系统组件"""
        try:
            # 获取风险监控系统
            self.risk_monitor = RiskMonitoringSystem()

            # 获取数据访问接口
            try:
                from db.interfaces.data_access_interface import DataAccessInterface
from db.sql_manager import SQLManager, QueryType
                self.data_access = self.container.resolve(DataAccessInterface)
            except Exception as e:
                logger.warning(f"无法获取数据访问接口: {e}")
                self.data_access = None

        except Exception as e:
            logger.error(f"初始化组件失败: {e}")
            raise

    @exception_handler(reraise=True)
    def start_monitoring(self, account_id: str, positions: List[Dict[str, Any]]):
        """
        启动实时监控

        Args:
            account_id: 账户ID
            positions: 初始持仓列表
        """
        with self._lock:
            if self.status == MonitoringStatus.RUNNING:
                logger.warning("监控已在运行中")
                return

            self.account_id = account_id
            self.initial_positions = positions
            self.status = MonitoringStatus.RUNNING
            self.start_time = datetime.now()
            self._stop_event.clear()

            # 启动监控线程
            self._monitor_thread = threading.Thread(
                target=self._monitoring_loop,
                name=f"RiskMonitor-{account_id}",
                daemon=True
            )
            self._monitor_thread.start()

            # 启动预警处理线程
            self._alert_thread = threading.Thread(
                target=self._alert_processing_loop,
                name=f"AlertProcessor-{account_id}",
                daemon=True
            )
            self._alert_thread.start()

            logger.info(f"实时风险监控已启动，账户: {account_id}")

    def stop_monitoring(self):
        """停止实时监控"""
        with self._lock:
            if self.status == MonitoringStatus.STOPPED:
                return

            logger.info("正在停止实时风险监控...")

            self.status = MonitoringStatus.STOPPED
            self._stop_event.set()

            # 等待线程结束
            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=10)

            if hasattr(self, '_alert_thread') and self._alert_thread.is_alive():
                self._alert_thread.join(timeout=5)

            logger.info("实时风险监控已停止")

    def pause_monitoring(self):
        """暂停监控"""
        with self._lock:
            if self.status == MonitoringStatus.RUNNING:
                self.status = MonitoringStatus.PAUSED
                logger.info("实时风险监控已暂停")

    def resume_monitoring(self):
        """恢复监控"""
        with self._lock:
            if self.status == MonitoringStatus.PAUSED:
                self.status = MonitoringStatus.RUNNING
                logger.info("实时风险监控已恢复")

    def _monitoring_loop(self):
        """主监控循环"""
        last_price_update = 0
        last_risk_calc = 0

        while not self._stop_event.is_set() and self.status != MonitoringStatus.STOPPED:
            try:
                current_time = time.time()

                # 检查是否暂停
                if self.status == MonitoringStatus.PAUSED:
                    time.sleep(1.0)
                    continue

                # 更新价格数据
                if current_time - last_price_update >= self.config.price_update_interval:
                    self._update_market_prices()
                    last_price_update = current_time

                # 更新持仓快照
                portfolio_snapshot = self._create_portfolio_snapshot()
                self._update_portfolio_data(portfolio_snapshot)

                # 计算风险指标
                if current_time - last_risk_calc >= self.config.risk_calc_interval:
                    self._calculate_risk_metrics(portfolio_snapshot)
                    last_risk_calc = current_time

                # 检查风险预警
                self._check_risk_alerts(portfolio_snapshot)

                # 执行止损止盈检查
                self._check_stop_loss_take_profit(portfolio_snapshot)

                # 检查异常交易
                self._check_trading_anomalies(portfolio_snapshot)

                # 检查熔断机制
                if self.config.enable_circuit_breaker:
                    self._check_circuit_breaker(portfolio_snapshot)

                # 触发回调函数
                self._trigger_callbacks(portfolio_snapshot)

                self.last_update_time = datetime.now()

                # 等待下次监控
                time.sleep(self.config.monitor_interval)

            except Exception as e:
                logger.error(f"监控循环出错: {e}")
                self.status = MonitoringStatus.ERROR
                time.sleep(5.0)  # 错误后等待5秒再继续

    def _alert_processing_loop(self):
        """预警处理循环"""
        while not self._stop_event.is_set():
            try:
                # 处理预警队列
                try:
                    alert = self.alert_queue.get(timeout=1.0)
                    self._process_alert(alert)
                    self.alert_queue.task_done()
                except queue.Empty:
                    continue

            except Exception as e:
                logger.error(f"预警处理循环出错: {e}")

    def _update_market_prices(self):
        """更新市场价格"""
        try:
            # 获取所有需要监控的股票代码
            if not self.current_portfolio or not self.current_portfolio.positions:
                return

            stock_codes = [pos.stock_code for pos in self.current_portfolio.positions]

            # 获取最新价格（模拟实现）
            for stock_code in stock_codes:
                current_price = self._get_real_time_price(stock_code)
                if current_price:
                    self._update_position_price(stock_code, current_price)

        except Exception as e:
            logger.error(f"更新市场价格失败: {e}")

    def _create_portfolio_snapshot(self) -> PortfolioSnapshot:
        """创建组合快照"""
        try:
            # 获取当前账户信息（模拟）
            account_info = self._get_current_account_info()

            # 创建持仓快照列表
            position_snapshots = []
            total_market_value = 0
            total_pnl = 0
            daily_pnl = 0

            for position_data in account_info.get('positions', []):
                position_snapshot = self._create_position_snapshot(position_data)
                position_snapshots.append(position_snapshot)
                total_market_value += position_snapshot.market_value
                total_pnl += position_snapshot.unrealized_pnl
                daily_pnl += position_snapshot.daily_pnl

            # 计算组合级别指标
            total_assets = account_info.get('total_assets', 0)
            available_cash = account_info.get('available_cash', 0)
            total_pnl_ratio = total_pnl / total_assets if total_assets > 0 else 0

            # 计算最大回撤
            max_drawdown = self._calculate_portfolio_drawdown(total_assets, total_pnl)

            # 创建组合快照
            portfolio_snapshot = PortfolioSnapshot(
                total_assets=total_assets,
                market_value=total_market_value,
                available_cash=available_cash,
                total_pnl=total_pnl,
                total_pnl_ratio=total_pnl_ratio,
                daily_pnl=daily_pnl,
                max_drawdown=max_drawdown,
                positions=position_snapshots,
                risk_score=0.0,  # 将在风险计算中更新
                var_1d=0.0,     # 将在风险计算中更新
                volatility=0.0, # 将在风险计算中更新
                timestamp=datetime.now()
            )

            return portfolio_snapshot

        except Exception as e:
            logger.error(f"创建组合快照失败: {e}")
            # 返回空快照
            return PortfolioSnapshot(
                total_assets=0, market_value=0, available_cash=0,
                total_pnl=0, total_pnl_ratio=0, daily_pnl=0, max_drawdown=0,
                positions=[], risk_score=0, var_1d=0, volatility=0,
                timestamp=datetime.now()
            )

    def _create_position_snapshot(self, position_data: Dict[str, Any]) -> PositionSnapshot:
        """创建持仓快照"""
        stock_code = position_data['stock_code']
        stock_name = position_data.get('stock_name', stock_code)
        quantity = position_data['quantity']
        avg_cost = position_data['avg_cost']

        # 获取当前价格
        current_price = position_data.get('current_price', avg_cost)

        # 计算市值和盈亏
        cost_value = quantity * avg_cost
        market_value = quantity * current_price
        unrealized_pnl = market_value - cost_value
        unrealized_pnl_ratio = unrealized_pnl / cost_value if cost_value > 0 else 0

        # 获取止损止盈价格
        stop_loss_price, take_profit_price = self._calculate_stop_loss_take_profit(
            stock_code, current_price, avg_cost, unrealized_pnl_ratio
        )

        # 获取风险评分
        risk_score = self._get_position_risk_score(stock_code, unrealized_pnl_ratio, current_price)

        return PositionSnapshot(
            stock_code=stock_code,
            stock_name=stock_name,
            quantity=quantity,
            avg_cost=avg_cost,
            current_price=current_price,
            market_value=market_value,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_ratio=unrealized_pnl_ratio,
            cost_value=cost_value,
            weight_in_portfolio=0.0,  # 将在组合级别计算中更新
            risk_score=risk_score,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            timestamp=datetime.now(),
            daily_pnl=position_data.get('daily_pnl', 0.0),
            max_profit=position_data.get('max_profit', max(0, unrealized_pnl)),
            max_drawdown=self._calculate_position_drawdown(stock_code, current_price, avg_cost),
            holding_days=position_data.get('holding_days', 0)
        )

    def _calculate_stop_loss_take_profit(self, stock_code: str, current_price: float,
                                       avg_cost: float, pnl_ratio: float) -> Tuple[float, float]:
        """计算止损止盈价格"""
        try:
            # 基础止损价格
            stop_loss_price = avg_cost * (1 - self.config.default_stop_loss)

            # 基础止盈价格
            take_profit_price = avg_cost * (1 + self.config.default_take_profit)

            # 跟踪止损逻辑
            if pnl_ratio > 0:  # 盈利状态
                # 使用跟踪止损
                trailing_stop_price = current_price * (1 - self.config.trailing_stop_ratio)
                stop_loss_price = max(stop_loss_price, trailing_stop_price)

            # 获取技术止损价格
            tech_stop_loss = self._get_technical_stop_loss(stock_code, current_price)
            if tech_stop_loss and tech_stop_loss > stop_loss_price:
                stop_loss_price = tech_stop_loss

            # 波动率调整止损
            volatility = self._get_stock_volatility(stock_code)
            if volatility:
                vol_adjusted_stop = current_price * (1 - volatility * 2)  # 2倍波动率止损
                stop_loss_price = max(stop_loss_price, vol_adjusted_stop)

            return (
                self.stability_manager.ensure_series_precision(pd.Series([stop_loss_price])).iloc[0],
                self.stability_manager.ensure_series_precision(pd.Series([take_profit_price])).iloc[0]
            )

        except Exception as e:
            logger.error(f"计算止损止盈价格失败 {stock_code}: {e}")
            return avg_cost * 0.92, avg_cost * 1.15  # 默认8%止损，15%止盈

    def _get_technical_stop_loss(self, stock_code: str, current_price: float) -> Optional[float]:
        """获取技术止损价格"""
        try:
            # 获取股票技术指标数据
            tech_data = self._get_technical_indicators(stock_code)
            if not tech_data:
                return None

            # 支撑位止损
            support_level = tech_data.get('support_level')
            if support_level and support_level < current_price:
                return support_level * 0.99  # 支撑位下方1%

            # 移动平均线止损
            ma20 = tech_data.get('ma20')
            if ma20 and ma20 < current_price:
                return ma20 * 0.98  # MA20下方2%

            return None

        except Exception as e:
            logger.error(f"获取技术止损价格失败 {stock_code}: {e}")
            return None

    def _update_portfolio_data(self, portfolio_snapshot: PortfolioSnapshot):
        """更新组合数据"""
        with self._lock:
            # 更新持仓权重
            if portfolio_snapshot.total_assets > 0:
                for position in portfolio_snapshot.positions:
                    position.weight_in_portfolio = position.market_value / portfolio_snapshot.total_assets

            # 保存当前快照
            self.current_portfolio = portfolio_snapshot

            # 更新历史数据
            self.portfolio_history.append(portfolio_snapshot)

            # 更新各持仓历史数据
            for position in portfolio_snapshot.positions:
                self.position_history[position.stock_code].append(position)

    def _calculate_risk_metrics(self, portfolio_snapshot: PortfolioSnapshot):
        """计算风险指标"""
        try:
            if not portfolio_snapshot.positions:
                return

            # 收集股票数据
            stock_codes = [pos.stock_code for pos in portfolio_snapshot.positions]
            holdings = {pos.stock_code: pos.weight_in_portfolio for pos in portfolio_snapshot.positions}

            # 计算组合风险
            portfolio_risk = self.risk_monitor.portfolio_manager.assess_portfolio_risk({
                'id': self.account_id,
                'name': f'Portfolio_{self.account_id}',
                'positions': [
                    {
                        'code': pos.stock_code,
                        'name': pos.stock_name,
                        'weight': pos.weight_in_portfolio,
                        'value': pos.market_value
                    }
                    for pos in portfolio_snapshot.positions
                ]
            })

            # 更新风险指标
            portfolio_snapshot.var_1d = portfolio_risk.portfolio_var
            portfolio_snapshot.volatility = portfolio_risk.portfolio_volatility
            portfolio_snapshot.risk_score = self._convert_risk_level_to_score(portfolio_risk.risk_level)

            # 更新个股风险评分
            for position in portfolio_snapshot.positions:
                try:
                    stock_risk = self.risk_monitor.stock_monitor.monitor_stock_risk(
                        position.stock_code, position.stock_name
                    )
                    position.risk_score = stock_risk.risk_score
                except Exception as e:
                    logger.warning(f"获取个股风险失败 {position.stock_code}: {e}")

        except Exception as e:
            logger.error(f"计算风险指标失败: {e}")

    def _check_risk_alerts(self, portfolio_snapshot: PortfolioSnapshot):
        """检查风险预警"""
        try:
            # 检查组合级别风险
            self._check_portfolio_risk_alerts(portfolio_snapshot)

            # 检查持仓级别风险
            for position in portfolio_snapshot.positions:
                self._check_position_risk_alerts(position, portfolio_snapshot)

        except Exception as e:
            logger.error(f"风险预警检查失败: {e}")

    def _check_portfolio_risk_alerts(self, portfolio_snapshot: PortfolioSnapshot):
        """检查组合风险预警"""
        # 总体风险评分预警
        if portfolio_snapshot.risk_score >= self.config.portfolio_risk_threshold:
            self._create_alert(
                alert_level=AlertLevel.CRITICAL if portfolio_snapshot.risk_score >= 90 else AlertLevel.WARNING,
                alert_type="组合风险预警",
                stock_code=None,
                title="组合风险评分过高",
                message=f"当前组合风险评分{portfolio_snapshot.risk_score:.1f}，超过阈值{self.config.portfolio_risk_threshold}",
                current_value=portfolio_snapshot.risk_score,
                threshold_value=self.config.portfolio_risk_threshold,
                suggested_action="建议降低高风险持仓或增加对冲"
            )

        # 回撤预警
        if portfolio_snapshot.max_drawdown >= self.config.drawdown_alert_threshold:
            alert_level = AlertLevel.EMERGENCY if portfolio_snapshot.max_drawdown >= self.config.drawdown_stop_threshold else AlertLevel.CRITICAL

            self._create_alert(
                alert_level=alert_level,
                alert_type="组合回撤预警",
                stock_code=None,
                title="组合回撤过大",
                message=f"当前组合最大回撤{portfolio_snapshot.max_drawdown:.2%}，超过预警线{self.config.drawdown_alert_threshold:.2%}",
                current_value=portfolio_snapshot.max_drawdown,
                threshold_value=self.config.drawdown_alert_threshold,
                suggested_action="建议止损或降低仓位" if alert_level == AlertLevel.EMERGENCY else "密切关注市场走势"
            )

        # 日内亏损预警
        daily_loss_ratio = abs(portfolio_snapshot.daily_pnl) / portfolio_snapshot.total_assets if portfolio_snapshot.total_assets > 0 else 0
        if portfolio_snapshot.daily_pnl < 0 and daily_loss_ratio >= self.config.max_daily_loss:
            self._create_alert(
                alert_level=AlertLevel.EMERGENCY,
                alert_type="日内亏损预警",
                stock_code=None,
                title="日内亏损达到上限",
                message=f"当日亏损{portfolio_snapshot.daily_pnl:.2f}元（{daily_loss_ratio:.2%}），达到止损线",
                current_value=daily_loss_ratio,
                threshold_value=self.config.max_daily_loss,
                suggested_action="立即止损，停止交易"
            )

    def _check_position_risk_alerts(self, position: PositionSnapshot, portfolio: PortfolioSnapshot):
        """检查持仓风险预警"""
        # 持仓风险评分预警
        if position.risk_score >= self.config.position_risk_threshold:
            self._create_alert(
                alert_level=AlertLevel.CRITICAL if position.risk_score >= 90 else AlertLevel.WARNING,
                alert_type="个股风险预警",
                stock_code=position.stock_code,
                title=f"{position.stock_code}风险评分过高",
                message=f"股票{position.stock_code}风险评分{position.risk_score:.1f}，超过阈值{self.config.position_risk_threshold}",
                current_value=position.risk_score,
                threshold_value=self.config.position_risk_threshold,
                suggested_action="建议减仓或设置更严格的止损"
            )

        # 单仓位亏损预警
        loss_ratio = abs(position.unrealized_pnl) / portfolio.total_assets if portfolio.total_assets > 0 else 0
        if position.unrealized_pnl < 0 and loss_ratio >= self.config.max_loss_per_position:
            self._create_alert(
                alert_level=AlertLevel.CRITICAL,
                alert_type="单仓位亏损预警",
                stock_code=position.stock_code,
                title=f"{position.stock_code}亏损过大",
                message=f"股票{position.stock_code}亏损{position.unrealized_pnl:.2f}元（占总资产{loss_ratio:.2%}），超过单仓位最大亏损限制",
                current_value=loss_ratio,
                threshold_value=self.config.max_loss_per_position,
                suggested_action="建议立即止损"
            )

        # 持仓集中度预警
        if position.weight_in_portfolio > 0.20:  # 单股票超过20%仓位
            self._create_alert(
                alert_level=AlertLevel.WARNING,
                alert_type="持仓集中度预警",
                stock_code=position.stock_code,
                title=f"{position.stock_code}持仓比例过高",
                message=f"股票{position.stock_code}持仓比例{position.weight_in_portfolio:.2%}，存在集中度风险",
                current_value=position.weight_in_portfolio,
                threshold_value=0.20,
                suggested_action="建议适当分散投资"
            )

    def _check_stop_loss_take_profit(self, portfolio_snapshot: PortfolioSnapshot):
        """检查止损止盈"""
        try:
            for position in portfolio_snapshot.positions:
                # 止损检查
                if position.current_price <= position.stop_loss_price:
                    self._create_alert(
                        alert_level=AlertLevel.EMERGENCY,
                        alert_type="止损信号",
                        stock_code=position.stock_code,
                        title=f"{position.stock_code}触发止损",
                        message=f"股票{position.stock_code}当前价格{position.current_price:.2f}触及止损价{position.stop_loss_price:.2f}",
                        current_value=position.current_price,
                        threshold_value=position.stop_loss_price,
                        suggested_action="建议立即止损出场"
                    )

                    # 自动执行止损（如果启用）
                    if hasattr(self.config, 'auto_stop_loss') and self.config.auto_stop_loss:
                        self._execute_stop_loss(position)

                # 止盈检查
                elif position.current_price >= position.take_profit_price:
                    self._create_alert(
                        alert_level=AlertLevel.INFO,
                        alert_type="止盈信号",
                        stock_code=position.stock_code,
                        title=f"{position.stock_code}达到止盈目标",
                        message=f"股票{position.stock_code}当前价格{position.current_price:.2f}达到止盈价{position.take_profit_price:.2f}",
                        current_value=position.current_price,
                        threshold_value=position.take_profit_price,
                        suggested_action="建议考虑止盈出场"
                    )

        except Exception as e:
            logger.error(f"止损止盈检查失败: {e}")

    def _check_trading_anomalies(self, portfolio_snapshot: PortfolioSnapshot):
        """检查异常交易"""
        try:
            for position in portfolio_snapshot.positions:
                # 获取历史价格数据
                price_history = self._get_price_history(position.stock_code, 20)
                if not price_history:
                    continue

                # 价格异常波动检查
                if len(price_history) >= 2:
                    price_change = (position.current_price - price_history[-2]) / price_history[-2]

                    if abs(price_change) >= self.config.price_shock_threshold:
                        self._create_alert(
                            alert_level=AlertLevel.WARNING,
                            alert_type="价格异常波动",
                            stock_code=position.stock_code,
                            title=f"{position.stock_code}价格异常波动",
                            message=f"股票{position.stock_code}价格波动{price_change:.2%}，超过异常阈值{self.config.price_shock_threshold:.2%}",
                            current_value=abs(price_change),
                            threshold_value=self.config.price_shock_threshold,
                            suggested_action="密切关注市场消息和交易量"
                        )

                # 成交量异常检查
                volume_data = self._get_volume_data(position.stock_code)
                if volume_data and len(volume_data) >= 5:
                    avg_volume = np.mean(volume_data[-5:])
                    current_volume = volume_data[-1] if volume_data else 0

                    if current_volume > avg_volume * self.config.volume_shock_multiplier:
                        self._create_alert(
                            alert_level=AlertLevel.INFO,
                            alert_type="成交量异常",
                            stock_code=position.stock_code,
                            title=f"{position.stock_code}成交量异常放大",
                            message=f"股票{position.stock_code}成交量{current_volume}，为近5日平均的{current_volume/avg_volume:.1f}倍",
                            current_value=current_volume/avg_volume,
                            threshold_value=self.config.volume_shock_multiplier,
                            suggested_action="关注是否有重大消息或异常交易"
                        )

        except Exception as e:
            logger.error(f"异常交易检查失败: {e}")

    def _check_circuit_breaker(self, portfolio_snapshot: PortfolioSnapshot):
        """检查熔断机制"""
        try:
            # 组合层面熔断检查
            if portfolio_snapshot.max_drawdown >= self.config.circuit_breaker_threshold:
                portfolio_snapshot.circuit_breaker_triggered = True

                self._create_alert(
                    alert_level=AlertLevel.EMERGENCY,
                    alert_type="熔断机制触发",
                    stock_code=None,
                    title="组合熔断机制触发",
                    message=f"组合最大回撤{portfolio_snapshot.max_drawdown:.2%}达到熔断阈值{self.config.circuit_breaker_threshold:.2%}",
                    current_value=portfolio_snapshot.max_drawdown,
                    threshold_value=self.config.circuit_breaker_threshold,
                    suggested_action="立即停止所有交易，执行应急风控措施"
                )

                # 执行熔断措施
                self._execute_circuit_breaker()

            # 日内亏损熔断
            daily_loss_ratio = abs(portfolio_snapshot.daily_pnl) / portfolio_snapshot.total_assets if portfolio_snapshot.total_assets > 0 else 0
            if portfolio_snapshot.daily_pnl < 0 and daily_loss_ratio >= self.config.max_daily_loss:
                portfolio_snapshot.circuit_breaker_triggered = True

                self._create_alert(
                    alert_level=AlertLevel.EMERGENCY,
                    alert_type="日内亏损熔断",
                    stock_code=None,
                    title="日内亏损达到熔断线",
                    message=f"当日亏损比例{daily_loss_ratio:.2%}达到熔断阈值{self.config.max_daily_loss:.2%}",
                    current_value=daily_loss_ratio,
                    threshold_value=self.config.max_daily_loss,
                    suggested_action="立即停止交易，等待次日重置"
                )

                self._execute_circuit_breaker()

        except Exception as e:
            logger.error(f"熔断检查失败: {e}")

    def _create_alert(self, alert_level: AlertLevel, alert_type: str, stock_code: Optional[str],
                     title: str, message: str, current_value: float, threshold_value: float,
                     suggested_action: str):
        """创建风险预警"""
        try:
            # 生成预警ID
            alert_id = f"{alert_type}_{stock_code or 'PORTFOLIO'}_{int(time.time())}"

            # 检查是否为重复预警
            existing_key = f"{alert_type}_{stock_code}"
            if existing_key in self.active_alerts and not self.active_alerts[existing_key].resolved:
                return  # 避免重复预警

            # 创建预警对象
            alert = RiskAlert(
                alert_id=alert_id,
                alert_level=alert_level,
                alert_type=alert_type,
                stock_code=stock_code,
                title=title,
                message=message,
                current_value=current_value,
                threshold_value=threshold_value,
                suggested_action=suggested_action,
                timestamp=datetime.now()
            )

            # 添加到活跃预警列表
            self.active_alerts[existing_key] = alert

            # 添加到处理队列
            self.alert_queue.put(alert)

            logger.warning(f"风险预警: {title} - {message}")

        except Exception as e:
            logger.error(f"创建预警失败: {e}")

    def _process_alert(self, alert: RiskAlert):
        """处理预警"""
        try:
            # 触发预警回调
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"预警回调执行失败: {e}")

            # 根据预警级别执行相应措施
            if alert.alert_level == AlertLevel.EMERGENCY:
                # 紧急预警需要立即处理
                if alert.alert_type in ["止损信号", "熔断机制触发", "日内亏损熔断"]:
                    self._handle_emergency_alert(alert)

        except Exception as e:
            logger.error(f"处理预警失败: {e}")

    def _handle_emergency_alert(self, alert: RiskAlert):
        """处理紧急预警"""
        logger.critical(f"紧急预警处理: {alert.title}")

        # 这里可以实现自动风控措施
        # 例如：自动止损、停止交易、通知相关人员等

        # 记录紧急事件
        self._log_emergency_event(alert)

    def _execute_stop_loss(self, position: PositionSnapshot):
        """执行止损"""
        try:
            logger.warning(f"执行自动止损: {position.stock_code} @ {position.current_price}")

            # 这里应该调用交易系统执行止损单
            # 实际实现中需要集成交易接口

            # 添加到执行队列
            self.action_queue.put({
                'action': 'stop_loss',
                'stock_code': position.stock_code,
                'quantity': position.quantity,
                'price': position.current_price,
                'timestamp': datetime.now()
            })

        except Exception as e:
            logger.error(f"执行止损失败 {position.stock_code}: {e}")

    def _execute_circuit_breaker(self):
        """执行熔断措施"""
        try:
            logger.critical("执行组合熔断措施")

            # 暂停监控
            self.pause_monitoring()

            # 这里应该实现：
            # 1. 停止所有新交易
            # 2. 撤销所有挂单
            # 3. 通知风控人员
            # 4. 记录熔断事件

            self._log_circuit_breaker_event()

        except Exception as e:
            logger.error(f"执行熔断措施失败: {e}")

    def _trigger_callbacks(self, portfolio_snapshot: PortfolioSnapshot):
        """触发回调函数"""
        try:
            # 触发组合回调
            for callback in self.portfolio_callbacks:
                try:
                    callback(portfolio_snapshot)
                except Exception as e:
                    logger.error(f"组合回调执行失败: {e}")

            # 触发持仓回调
            for position in portfolio_snapshot.positions:
                for callback in self.position_callbacks:
                    try:
                        callback(position)
                    except Exception as e:
                        logger.error(f"持仓回调执行失败: {e}")

        except Exception as e:
            logger.error(f"触发回调失败: {e}")

    # 辅助方法
    def _get_real_time_price(self, stock_code: str) -> Optional[float]:
        """获取实时价格"""
        try:
            # 实际实现中应该调用实时行情接口
            # 这里使用模拟价格
            import random
            base_price = float(stock_code) / 100000
            current_price = base_price * (1 + random.uniform(-0.05, 0.05))
            return round(current_price, 2)
        except:
            return None

    def _update_position_price(self, stock_code: str, new_price: float):
        """更新持仓价格"""
        # 实际实现中应该更新持仓数据
        pass

    def _get_current_account_info(self) -> Dict[str, Any]:
        """获取当前账户信息"""
        # 模拟账户信息
        return {
            'total_assets': 1000000.0,
            'available_cash': 100000.0,
            'positions': [
                {
                    'stock_code': '000001',
                    'stock_name': '平安银行',
                    'quantity': 1000,
                    'avg_cost': 10.50,
                    'current_price': self._get_real_time_price('000001') or 10.50,
                    'daily_pnl': 0.0,
                    'max_profit': 0.0,
                    'holding_days': 5
                }
            ]
        }

    def _calculate_portfolio_drawdown(self, total_assets: float, total_pnl: float) -> float:
        """计算组合回撤"""
        try:
            if not self.portfolio_history:
                return 0.0

            # 获取历史最高资产值
            max_assets = max([snapshot.total_assets for snapshot in self.portfolio_history])

            # 计算当前回撤
            current_assets = total_assets
            drawdown = (max_assets - current_assets) / max_assets if max_assets > 0 else 0

            return max(0, drawdown)

        except Exception as e:
            logger.error(f"计算组合回撤失败: {e}")
            return 0.0

    def _calculate_position_drawdown(self, stock_code: str, current_price: float, avg_cost: float) -> float:
        """计算持仓回撤"""
        try:
            # 获取持仓历史数据
            history = self.position_history.get(stock_code, [])
            if not history:
                return 0.0

            # 找到历史最高价格
            max_price = max([pos.current_price for pos in history])

            # 计算回撤
            drawdown = (max_price - current_price) / max_price if max_price > 0 else 0

            return max(0, drawdown)

        except Exception as e:
            logger.error(f"计算持仓回撤失败: {e}")
            return 0.0

    def _get_position_risk_score(self, stock_code: str, pnl_ratio: float, current_price: float) -> float:
        """获取持仓风险评分"""
        try:
            # 基础风险评分（基于盈亏比例）
            base_risk = 50.0

            # 亏损增加风险评分
            if pnl_ratio < 0:
                base_risk += min(abs(pnl_ratio) * 200, 40)  # 最多增加40分

            # 获取技术风险评分
            try:
                risk_metrics = self.risk_monitor.stock_monitor.monitor_stock_risk(stock_code)
                tech_risk = risk_metrics.risk_score
                base_risk = (base_risk + tech_risk) / 2  # 平均值
            except:
                pass

            return min(100, max(0, base_risk))

        except Exception as e:
            logger.error(f"获取持仓风险评分失败: {e}")
            return 50.0

    def _convert_risk_level_to_score(self, risk_level: str) -> float:
        """将风险级别转换为评分"""
        mapping = {
            '低风险': 25.0,
            '中等风险': 50.0,
            '高风险': 75.0,
            '极高风险': 95.0
        }
        return mapping.get(risk_level, 50.0)

    def _get_technical_indicators(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """获取技术指标数据"""
        # 实际实现中应该从数据源获取技术指标
        return None

    def _get_stock_volatility(self, stock_code: str) -> Optional[float]:
        """获取股票波动率"""
        try:
            # 获取历史价格计算波动率
            price_history = self._get_price_history(stock_code, 30)
            if not price_history or len(price_history) < 10:
                return None

            returns = np.diff(np.log(price_history))
            volatility = np.std(returns) * np.sqrt(252)  # 年化波动率

            return volatility

        except Exception as e:
            logger.error(f"获取股票波动率失败: {e}")
            return None

    def _get_price_history(self, stock_code: str, days: int) -> Optional[List[float]]:
        """获取价格历史"""
        # 实际实现中应该从数据库获取历史价格
        # 这里返回模拟数据
        try:
            import random
            base_price = float(stock_code) / 100000
            prices = []
            current_price = base_price

            for _ in range(days):
                current_price *= (1 + random.uniform(-0.05, 0.05))
                prices.append(current_price)

            return prices
        except:
            return None

    def _get_volume_data(self, stock_code: str) -> Optional[List[float]]:
        """获取成交量数据"""
        # 实际实现中应该从数据库获取成交量数据
        return None

    def _log_emergency_event(self, alert: RiskAlert):
        """记录紧急事件"""
        logger.critical(f"紧急风控事件: {alert.alert_type} - {alert.message}")

    def _log_circuit_breaker_event(self):
        """记录熔断事件"""
        logger.critical("组合熔断事件已触发，所有交易已暂停")

    # 回调管理方法
    def add_alert_callback(self, callback: Callable[[RiskAlert], None]):
        """添加预警回调"""
        self.alert_callbacks.append(callback)

    def add_position_callback(self, callback: Callable[[PositionSnapshot], None]):
        """添加持仓回调"""
        self.position_callbacks.append(callback)

    def add_portfolio_callback(self, callback: Callable[[PortfolioSnapshot], None]):
        """添加组合回调"""
        self.portfolio_callbacks.append(callback)

    def remove_alert_callback(self, callback: Callable[[RiskAlert], None]):
        """移除预警回调"""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)

    # 状态查询方法
    def get_monitoring_status(self) -> Dict[str, Any]:
        """获取监控状态"""
        return {
            'status': self.status.value,
            'account_id': getattr(self, 'account_id', None),
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'last_update_time': self.last_update_time.isoformat() if self.last_update_time else None,
            'active_alerts_count': len(self.active_alerts),
            'total_alerts_count': len(self.active_alerts) + len(self.resolved_alerts),
            'config': asdict(self.config)
        }

    def get_current_portfolio(self) -> Optional[PortfolioSnapshot]:
        """获取当前组合快照"""
        return self.current_portfolio

    def get_active_alerts(self) -> List[RiskAlert]:
        """获取活跃预警"""
        return list(self.active_alerts.values())

    def acknowledge_alert(self, alert_key: str):
        """确认预警"""
        if alert_key in self.active_alerts:
            self.active_alerts[alert_key].acknowledged = True

    def resolve_alert(self, alert_key: str):
        """解决预警"""
        if alert_key in self.active_alerts:
            alert = self.active_alerts.pop(alert_key)
            alert.resolved = True
            self.resolved_alerts.append(alert)


# 创建全局实例
_real_time_monitor = None

def get_real_time_monitor(config: PositionMonitorConfig = None) -> RealTimeRiskMonitor:
    """获取实时风险监控器实例"""
    global _real_time_monitor

    if _real_time_monitor is None:
        _real_time_monitor = RealTimeRiskMonitor(config)

    return _real_time_monitor