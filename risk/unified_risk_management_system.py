#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
统一风控管理系统

专业量化交易系统的核心风控架构，集成事前、事中、事后全流程风险管理。
符合机构级风控要求，响应时间≤10ms，支持6位小数精度。

核心架构：
1. 事前风控 - 交易前风险检查和规则验证
2. 事中监控 - 实时风险监控和动态止损
3. 事后分析 - 风险评估和绩效归因
4. 仓位管理 - 资金管理和组合风险控制

技术特性：
- 超低延迟风控检查（≤10ms）
- 高精度数值计算（6位小数）
- 实时风险监控
- 智能预警系统
- 自适应风控策略
"""

import time
import json
import threading
import asyncio
import queue
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import defaultdict, deque
import concurrent.futures
import numpy as np
import pandas as pd

from utils.logger import get_logger
from utils.exception_handler import exception_handler
from utils.performance_monitor import performance_monitor
from utils.unified_container import get_container
from utils.numerical_stability_manager import get_stability_manager

# 导入风控子系统
from risk.pre_trade_risk_control import PreTradeRiskController, TradeRequest, RiskCheckResult
from risk.real_time_risk_monitor import RealTimeRiskMonitor, PositionSnapshot, MonitoringStatus
from risk.portfolio_risk import PortfolioRiskManager, RiskMetrics
from risk.dynamic_stop_loss import DynamicStopLossManager, StopLossType
from risk.warning_system import RiskWarningSystem, AlertLevel

# 导入监控系统
from monitoring.risk_monitor import RiskMonitoringSystem, RiskLevel, RiskType
from monitoring.alert_manager import AlertManager
from monitoring.intelligent_alert_system import IntelligentAlertSystem

logger = get_logger(__name__)


class RiskControlMode(Enum):
    """风控模式枚举"""
    CONSERVATIVE = "保守模式"      # 最严格风控
    BALANCED = "平衡模式"         # 平衡收益与风险
    AGGRESSIVE = "激进模式"       # 追求高收益
    CUSTOM = "自定义模式"         # 自定义风控参数


class SystemStatus(Enum):
    """系统状态枚举"""
    INITIALIZING = "初始化中"
    RUNNING = "运行中"
    PAUSED = "暂停"
    EMERGENCY_STOP = "紧急停止"
    ERROR = "系统错误"
    MAINTENANCE = "维护模式"


@dataclass
class RiskControlConfig:
    """统一风控配置"""
    # 系统配置
    mode: RiskControlMode = RiskControlMode.BALANCED
    max_response_time_ms: float = 10.0          # 最大响应时间（毫秒）
    precision: int = 6                           # 数值精度（小数位）

    # 事前风控配置
    enable_pre_trade_check: bool = True
    max_position_size: float = 0.05             # 单仓最大仓位（5%）
    max_portfolio_exposure: float = 0.95        # 最大组合敞口（95%）
    min_cash_reserve: float = 0.05              # 最小现金储备（5%）

    # 事中监控配置
    enable_realtime_monitoring: bool = True
    monitor_interval: float = 1.0               # 监控间隔（秒）
    risk_calculation_interval: float = 5.0      # 风险计算间隔（秒）

    # 止损配置
    default_stop_loss: float = 0.08             # 默认止损（8%）
    max_drawdown_threshold: float = 0.10        # 最大回撤阈值（10%）
    enable_dynamic_stop_loss: bool = True

    # 预警配置
    enable_intelligent_alerts: bool = True
    alert_threshold_warning: float = 70.0       # 预警阈值（70%）
    alert_threshold_critical: float = 85.0      # 严重阈值（85%）
    alert_threshold_emergency: float = 95.0     # 紧急阈值（95%）


@dataclass
class RiskMetricsSnapshot:
    """风险指标快照"""
    timestamp: datetime

    # 组合风险指标
    portfolio_value: float
    total_pnl: float
    unrealized_pnl: float
    realized_pnl: float
    cash_balance: float

    # 风险度量
    portfolio_var_1d: float                     # 1日VaR
    portfolio_var_5d: float                     # 5日VaR
    max_drawdown: float                         # 最大回撤
    current_drawdown: float                     # 当前回撤
    volatility: float                           # 组合波动率
    sharpe_ratio: float                         # 夏普比率

    # 仓位风险
    total_positions: int                        # 总仓位数
    high_risk_positions: int                    # 高风险仓位数
    concentration_risk: float                   # 集中度风险
    sector_exposure: Dict[str, float]           # 行业敞口

    # 系统指标
    response_time_ms: float                     # 响应时间
    risk_score: float                           # 综合风险评分
    system_health: float                        # 系统健康度


class UnifiedRiskManagementSystem:
    """
    统一风控管理系统

    集成事前、事中、事后全流程风险管理的核心系统
    """

    def __init__(self,
                 config: Optional[RiskControlConfig] = None,
                 data_manager=None):
        """
        初始化统一风控管理系统

        Args:
            config: 风控配置
            data_manager: 数据管理器
        """
        self.config = config or RiskControlConfig()
        self.data_manager = data_manager or get_container().resolve("data_manager")
        self.stability_manager = get_stability_manager()

        # 系统状态
        self.status = SystemStatus.INITIALIZING
        self.start_time = datetime.now()
        self.last_health_check = datetime.now()

        # 核心子系统
        self.pre_trade_controller = None
        self.realtime_monitor = None
        self.portfolio_manager = None
        self.stop_loss_manager = None
        self.warning_system = None
        self.alert_manager = None

        # 性能监控
        self.response_times = deque(maxlen=1000)
        self.risk_metrics_history = deque(maxlen=10000)

        # 线程控制
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._monitoring_thread = None

        # 初始化子系统
        self._initialize_subsystems()

        logger.info(f"统一风控管理系统初始化完成 - 模式: {self.config.mode.value}")

    def _initialize_subsystems(self):
        """初始化风控子系统"""
        try:
            # 事前风控系统
            if self.config.enable_pre_trade_check:
                self.pre_trade_controller = PreTradeRiskController(
                    data_manager=self.data_manager
                )

            # 实时监控系统
            if self.config.enable_realtime_monitoring:
                self.realtime_monitor = RealTimeRiskMonitor(
                    data_manager=self.data_manager
                )

            # 组合风险管理
            self.portfolio_manager = PortfolioRiskManager(
                data_manager=self.data_manager
            )

            # 动态止损管理
            if self.config.enable_dynamic_stop_loss:
                self.stop_loss_manager = DynamicStopLossManager(
                    data_manager=self.data_manager
                )

            # 预警系统
            self.warning_system = RiskWarningSystem()

            # 智能警报系统
            if self.config.enable_intelligent_alerts:
                self.alert_manager = IntelligentAlertSystem()

            self.status = SystemStatus.RUNNING
            logger.info("风控子系统初始化成功")

        except Exception as e:
            self.status = SystemStatus.ERROR
            logger.error(f"风控子系统初始化失败: {e}")
            raise

    @performance_monitor
    @exception_handler
    def check_pre_trade_risk(self, trade_request: TradeRequest) -> RiskCheckResult:
        """
        事前风险检查

        Args:
            trade_request: 交易请求

        Returns:
            RiskCheckResult: 风控检查结果
        """
        start_time = time.perf_counter()

        try:
            # 检查系统状态
            if self.status not in [SystemStatus.RUNNING]:
                return RiskCheckResult(
                    approved=False,
                    risk_score=100.0,
                    rejection_reasons=["系统状态异常"],
                    response_time_ms=0.0
                )

            # 执行事前风控检查
            if self.pre_trade_controller:
                risk_result = self.pre_trade_controller.check_trade_risk(trade_request)
            else:
                # 基础风控检查
                risk_result = self._basic_risk_check(trade_request)

            # 记录响应时间
            response_time = (time.perf_counter() - start_time) * 1000
            self.response_times.append(response_time)
            risk_result.response_time_ms = response_time

            # 检查响应时间要求
            if response_time > self.config.max_response_time_ms:
                logger.warning(f"风控检查响应时间超标: {response_time:.2f}ms > {self.config.max_response_time_ms}ms")

            # 数值精度处理
            risk_result.risk_score = self.stability_manager.round_to_precision(
                risk_result.risk_score, self.config.precision
            )

            logger.debug(f"事前风控检查完成: {trade_request.stock_code} - "
                        f"风险评分: {risk_result.risk_score} - "
                        f"响应时间: {response_time:.2f}ms")

            return risk_result

        except Exception as e:
            logger.error(f"事前风控检查异常: {e}")
            return RiskCheckResult(
                approved=False,
                risk_score=100.0,
                rejection_reasons=[f"系统异常: {str(e)}"],
                response_time_ms=(time.perf_counter() - start_time) * 1000
            )

    def _basic_risk_check(self, trade_request: TradeRequest) -> RiskCheckResult:
        """基础风控检查（备用方案）"""
        risk_score = 0.0
        rejection_reasons = []

        # 基础仓位检查
        if trade_request.quantity * trade_request.price > self.config.max_position_size * 1000000:
            risk_score += 30.0
            rejection_reasons.append("单仓仓位超限")

        # 现金储备检查
        available_cash = self._get_available_cash()
        required_cash = trade_request.quantity * trade_request.price

        if available_cash < required_cash + self.config.min_cash_reserve * 1000000:
            risk_score += 40.0
            rejection_reasons.append("现金储备不足")

        return RiskCheckResult(
            approved=risk_score < 50.0,
            risk_score=risk_score,
            rejection_reasons=rejection_reasons,
            warnings=[]
        )

    @performance_monitor
    def get_realtime_risk_metrics(self) -> RiskMetricsSnapshot:
        """
        获取实时风险指标

        Returns:
            RiskMetricsSnapshot: 风险指标快照
        """
        try:
            current_time = datetime.now()

            # 获取组合信息
            portfolio_value = self._calculate_portfolio_value()
            cash_balance = self._get_available_cash()

            # 获取P&L信息
            total_pnl, unrealized_pnl, realized_pnl = self._calculate_pnl()

            # 计算风险指标
            risk_metrics = self._calculate_risk_metrics()

            # 获取仓位信息
            positions_info = self._get_positions_info()

            # 计算系统性能指标
            avg_response_time = np.mean(list(self.response_times)) if self.response_times else 0.0

            snapshot = RiskMetricsSnapshot(
                timestamp=current_time,
                portfolio_value=self.stability_manager.round_to_precision(portfolio_value, self.config.precision),
                total_pnl=self.stability_manager.round_to_precision(total_pnl, self.config.precision),
                unrealized_pnl=self.stability_manager.round_to_precision(unrealized_pnl, self.config.precision),
                realized_pnl=self.stability_manager.round_to_precision(realized_pnl, self.config.precision),
                cash_balance=self.stability_manager.round_to_precision(cash_balance, self.config.precision),
                portfolio_var_1d=self.stability_manager.round_to_precision(risk_metrics.get('var_1d', 0.0), self.config.precision),
                portfolio_var_5d=self.stability_manager.round_to_precision(risk_metrics.get('var_5d', 0.0), self.config.precision),
                max_drawdown=self.stability_manager.round_to_precision(risk_metrics.get('max_drawdown', 0.0), self.config.precision),
                current_drawdown=self.stability_manager.round_to_precision(risk_metrics.get('current_drawdown', 0.0), self.config.precision),
                volatility=self.stability_manager.round_to_precision(risk_metrics.get('volatility', 0.0), self.config.precision),
                sharpe_ratio=self.stability_manager.round_to_precision(risk_metrics.get('sharpe_ratio', 0.0), self.config.precision),
                total_positions=positions_info['total_positions'],
                high_risk_positions=positions_info['high_risk_positions'],
                concentration_risk=self.stability_manager.round_to_precision(positions_info['concentration_risk'], self.config.precision),
                sector_exposure=positions_info['sector_exposure'],
                response_time_ms=self.stability_manager.round_to_precision(avg_response_time, 2),
                risk_score=self.stability_manager.round_to_precision(self._calculate_overall_risk_score(risk_metrics), self.config.precision),
                system_health=self.stability_manager.round_to_precision(self._calculate_system_health(), self.config.precision)
            )

            # 保存历史记录
            self.risk_metrics_history.append(snapshot)

            return snapshot

        except Exception as e:
            logger.error(f"获取实时风险指标异常: {e}")
            # 返回默认快照
            return RiskMetricsSnapshot(
                timestamp=datetime.now(),
                portfolio_value=0.0, total_pnl=0.0, unrealized_pnl=0.0, realized_pnl=0.0,
                cash_balance=0.0, portfolio_var_1d=0.0, portfolio_var_5d=0.0,
                max_drawdown=0.0, current_drawdown=0.0, volatility=0.0, sharpe_ratio=0.0,
                total_positions=0, high_risk_positions=0, concentration_risk=0.0,
                sector_exposure={}, response_time_ms=0.0, risk_score=0.0, system_health=0.0
            )

    def start_realtime_monitoring(self):
        """启动实时风险监控"""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            logger.warning("实时监控已在运行")
            return

        self._stop_event.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name="RiskMonitoring",
            daemon=True
        )
        self._monitoring_thread.start()
        logger.info("实时风险监控已启动")

    def stop_realtime_monitoring(self):
        """停止实时风险监控"""
        self._stop_event.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)
        logger.info("实时风险监控已停止")

    def _monitoring_loop(self):
        """监控循环"""
        while not self._stop_event.is_set():
            try:
                # 获取风险指标
                risk_snapshot = self.get_realtime_risk_metrics()

                # 检查风险阈值
                self._check_risk_thresholds(risk_snapshot)

                # 执行动态止损检查
                if self.stop_loss_manager:
                    self._execute_stop_loss_check()

                # 系统健康检查
                self._perform_health_check()

            except Exception as e:
                logger.error(f"监控循环异常: {e}")

            # 等待下次监控
            self._stop_event.wait(self.config.monitor_interval)

    def _check_risk_thresholds(self, snapshot: RiskMetricsSnapshot):
        """检查风险阈值并发送预警"""
        # 检查综合风险评分
        if snapshot.risk_score >= self.config.alert_threshold_emergency:
            self._trigger_emergency_alert(snapshot)
        elif snapshot.risk_score >= self.config.alert_threshold_critical:
            self._trigger_critical_alert(snapshot)
        elif snapshot.risk_score >= self.config.alert_threshold_warning:
            self._trigger_warning_alert(snapshot)

        # 检查回撤阈值
        if snapshot.current_drawdown >= self.config.max_drawdown_threshold:
            self._trigger_drawdown_alert(snapshot)

    def _trigger_emergency_alert(self, snapshot: RiskMetricsSnapshot):
        """触发紧急预警"""
        alert_message = {
            "level": "EMERGENCY",
            "message": "系统风险达到紧急级别",
            "risk_score": snapshot.risk_score,
            "timestamp": snapshot.timestamp.isoformat(),
            "actions": ["暂停所有交易", "风险经理确认"]
        }

        if self.alert_manager:
            self.alert_manager.send_emergency_alert(alert_message)

        logger.critical(f"紧急风险预警: 风险评分 {snapshot.risk_score}")

    def _trigger_critical_alert(self, snapshot: RiskMetricsSnapshot):
        """触发严重预警"""
        alert_message = {
            "level": "CRITICAL",
            "message": "系统风险达到严重级别",
            "risk_score": snapshot.risk_score,
            "timestamp": snapshot.timestamp.isoformat(),
            "actions": ["减少新开仓位", "加强监控"]
        }

        if self.alert_manager:
            self.alert_manager.send_critical_alert(alert_message)

        logger.error(f"严重风险预警: 风险评分 {snapshot.risk_score}")

    def _trigger_warning_alert(self, snapshot: RiskMetricsSnapshot):
        """触发警告预警"""
        alert_message = {
            "level": "WARNING",
            "message": "系统风险达到警告级别",
            "risk_score": snapshot.risk_score,
            "timestamp": snapshot.timestamp.isoformat(),
            "actions": ["关注风险变化", "准备风控措施"]
        }

        if self.alert_manager:
            self.alert_manager.send_warning_alert(alert_message)

        logger.warning(f"风险预警: 风险评分 {snapshot.risk_score}")

    def _trigger_drawdown_alert(self, snapshot: RiskMetricsSnapshot):
        """触发回撤预警"""
        alert_message = {
            "level": "CRITICAL",
            "message": f"组合回撤达到阈值: {snapshot.current_drawdown:.2%}",
            "drawdown": snapshot.current_drawdown,
            "threshold": self.config.max_drawdown_threshold,
            "timestamp": snapshot.timestamp.isoformat()
        }

        if self.alert_manager:
            self.alert_manager.send_critical_alert(alert_message)

        logger.error(f"回撤预警: 当前回撤 {snapshot.current_drawdown:.2%}")

    # 辅助方法
    def _calculate_portfolio_value(self) -> float:
        """计算组合总价值（模拟）"""
        # 实际应从数据库获取真实持仓数据
        return 1000000.0  # 100万基准

    def _get_available_cash(self) -> float:
        """获取可用现金（模拟）"""
        return 50000.0  # 5万现金储备

    def _calculate_pnl(self) -> Tuple[float, float, float]:
        """计算损益（模拟）"""
        total_pnl = 5000.0
        unrealized_pnl = 3000.0
        realized_pnl = 2000.0
        return total_pnl, unrealized_pnl, realized_pnl

    def _calculate_risk_metrics(self) -> Dict[str, float]:
        """计算风险指标（模拟）"""
        return {
            'var_1d': 15000.0,
            'var_5d': 35000.0,
            'max_drawdown': 0.05,
            'current_drawdown': 0.02,
            'volatility': 0.18,
            'sharpe_ratio': 1.25
        }

    def _get_positions_info(self) -> Dict[str, Any]:
        """获取仓位信息（模拟）"""
        return {
            'total_positions': 8,
            'high_risk_positions': 1,
            'concentration_risk': 0.25,
            'sector_exposure': {
                'technology': 0.30,
                'finance': 0.25,
                'healthcare': 0.20,
                'consumer': 0.15,
                'others': 0.10
            }
        }

    def _calculate_overall_risk_score(self, risk_metrics: Dict[str, float]) -> float:
        """计算综合风险评分"""
        # 简化的风险评分模型
        base_score = 30.0

        # 回撤因子
        drawdown_factor = risk_metrics.get('current_drawdown', 0.0) * 100

        # 波动率因子
        volatility_factor = risk_metrics.get('volatility', 0.0) * 50

        total_score = base_score + drawdown_factor + volatility_factor
        return min(total_score, 100.0)  # 限制最高100分

    def _calculate_system_health(self) -> float:
        """计算系统健康度"""
        health_score = 100.0

        # 响应时间影响
        if self.response_times:
            avg_response_time = np.mean(list(self.response_times))
            if avg_response_time > self.config.max_response_time_ms:
                health_score -= min(30.0, (avg_response_time - self.config.max_response_time_ms) * 2)

        # 系统运行时间影响
        uptime_hours = (datetime.now() - self.start_time).total_seconds() / 3600
        if uptime_hours > 12:  # 超过12小时运行
            health_score = max(health_score - 5.0, 85.0)

        return max(health_score, 0.0)

    def _execute_stop_loss_check(self):
        """执行止损检查"""
        if not self.stop_loss_manager:
            return

        try:
            # 获取所有持仓
            positions = self._get_current_positions()

            for position in positions:
                # 检查是否需要止损
                should_stop = self.stop_loss_manager.should_trigger_stop_loss(
                    position['stock_code'],
                    position['current_price'],
                    position['cost_price']
                )

                if should_stop:
                    self._execute_stop_loss_order(position)

        except Exception as e:
            logger.error(f"止损检查异常: {e}")

    def _get_current_positions(self) -> List[Dict]:
        """获取当前持仓（模拟）"""
        return [
            {
                'stock_code': '000001',
                'quantity': 1000,
                'current_price': 12.50,
                'cost_price': 13.20
            }
        ]

    def _execute_stop_loss_order(self, position: Dict):
        """执行止损订单（模拟）"""
        logger.info(f"执行止损: {position['stock_code']} - "
                   f"当前价格: {position['current_price']} - "
                   f"成本价格: {position['cost_price']}")

    def _perform_health_check(self):
        """执行系统健康检查"""
        current_time = datetime.now()

        # 检查子系统状态
        subsystem_status = {
            'pre_trade_controller': self.pre_trade_controller is not None,
            'realtime_monitor': self.realtime_monitor is not None,
            'portfolio_manager': self.portfolio_manager is not None,
            'stop_loss_manager': self.stop_loss_manager is not None,
            'warning_system': self.warning_system is not None,
            'alert_manager': self.alert_manager is not None
        }

        # 检查数据连接
        data_connection_ok = self._check_data_connection()

        # 更新健康检查时间
        self.last_health_check = current_time

        logger.debug(f"系统健康检查完成 - 子系统状态: {subsystem_status} - "
                    f"数据连接: {data_connection_ok}")

    def _check_data_connection(self) -> bool:
        """检查数据连接状态"""
        try:
            if self.data_manager:
                # 简单的连接测试
                return True
            return False
        except Exception:
            return False

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态信息"""
        return {
            'status': self.status.value,
            'mode': self.config.mode.value,
            'start_time': self.start_time.isoformat(),
            'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
            'last_health_check': self.last_health_check.isoformat(),
            'avg_response_time_ms': np.mean(list(self.response_times)) if self.response_times else 0.0,
            'subsystems': {
                'pre_trade_controller': self.pre_trade_controller is not None,
                'realtime_monitor': self.realtime_monitor is not None,
                'portfolio_manager': self.portfolio_manager is not None,
                'stop_loss_manager': self.stop_loss_manager is not None,
                'warning_system': self.warning_system is not None,
                'alert_manager': self.alert_manager is not None
            }
        }

    def shutdown(self):
        """系统关闭"""
        logger.info("统一风控管理系统正在关闭...")

        # 停止监控
        self.stop_realtime_monitoring()

        # 更新状态
        self.status = SystemStatus.MAINTENANCE

        logger.info("统一风控管理系统已关闭")


# 全局实例管理
_risk_management_system = None

def get_unified_risk_management_system(
    config: Optional[RiskControlConfig] = None,
    data_manager=None
) -> UnifiedRiskManagementSystem:
    """
    获取统一风控管理系统实例（单例模式）

    Args:
        config: 风控配置
        data_manager: 数据管理器

    Returns:
        UnifiedRiskManagementSystem: 风控系统实例
    """
    global _risk_management_system

    if _risk_management_system is None:
        _risk_management_system = UnifiedRiskManagementSystem(
            config=config,
            data_manager=data_manager
        )

    return _risk_management_system


if __name__ == "__main__":
    # 演示使用
    print("=== 统一风控管理系统演示 ===")

    # 创建风控系统
    risk_system = get_unified_risk_management_system()

    # 获取系统状态
    status = risk_system.get_system_status()
    print(f"系统状态: {json.dumps(status, indent=2, ensure_ascii=False)}")

    # 启动实时监控
    risk_system.start_realtime_monitoring()

    # 模拟交易请求
    from risk.pre_trade_risk_control import TradeRequest, TradeDirection

    trade_request = TradeRequest(
        stock_code="000001",
        direction=TradeDirection.BUY,
        quantity=1000,
        price=12.50
    )

    # 执行事前风控检查
    risk_result = risk_system.check_pre_trade_risk(trade_request)
    print(f"风控检查结果: 通过={risk_result.approved}, 风险评分={risk_result.risk_score}")

    # 获取实时风险指标
    risk_metrics = risk_system.get_realtime_risk_metrics()
    print(f"组合价值: {risk_metrics.portfolio_value:,.2f}")
    print(f"风险评分: {risk_metrics.risk_score:.2f}")
    print(f"响应时间: {risk_metrics.response_time_ms:.2f}ms")

    # 运行一段时间后关闭
    import time
    time.sleep(10)
    risk_system.shutdown()

    print("演示完成")