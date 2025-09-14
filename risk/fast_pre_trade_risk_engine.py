#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
高性能事前风控检查引擎

专业金融量化交易系统的事前风控核心模块，实现超低延迟（≤10ms）的
交易前风险检查，确保每笔交易都符合风控要求。

核心特性：
1. 超低延迟检查 - 目标响应时间≤10ms
2. 多层风控规则 - 资金、仓位、集中度、流动性检查
3. 高精度计算 - 6位小数精度保障
4. 缓存优化 - 热点数据缓存，提升检查性能
5. 异步检查 - 支持并发风控检查
6. 实时规则更新 - 动态风控参数调整

技术架构：
- 基于规则引擎的风控检查
- 内存缓存加速重复检查
- 异步并发处理提升吞吐量
- 数值稳定性保障
"""

import time
import asyncio
import threading
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import defaultdict, deque
import concurrent.futures
import numpy as np
import pandas as pd
from functools import lru_cache
import json

from utils.logger import get_logger
from utils.exception_handler import exception_handler
from utils.performance_monitor import performance_monitor
from utils.unified_container import get_container
from utils.numerical_stability_manager import get_stability_manager

logger = get_logger(__name__)


class RiskCheckStatus(Enum):
    """风控检查状态"""
    APPROVED = "通过"
    REJECTED = "拒绝"
    CONDITIONAL = "条件通过"
    ERROR = "检查异常"


class RiskRuleType(Enum):
    """风控规则类型"""
    POSITION_SIZE = "仓位规模"
    CASH_REQUIREMENT = "资金要求"
    CONCENTRATION = "集中度"
    LIQUIDITY = "流动性"
    SECTOR_EXPOSURE = "行业敞口"
    CORRELATION = "相关性"
    VOLATILITY = "波动性"
    DRAWDOWN = "回撤"
    CUSTOM = "自定义"


@dataclass
class FastRiskCheckConfig:
    """快速风控检查配置"""
    # 性能配置
    max_response_time_ms: float = 10.0          # 最大响应时间（毫秒）
    enable_cache: bool = True                    # 启用缓存
    cache_ttl_seconds: int = 300                # 缓存TTL（秒）
    enable_async: bool = True                   # 启用异步处理

    # 基础风控参数
    max_single_position_ratio: float = 0.05     # 单仓最大占比（5%）
    max_sector_exposure: float = 0.30           # 单行业最大敞口（30%）
    min_cash_reserve_ratio: float = 0.05        # 最小现金储备比例（5%）
    max_correlation_threshold: float = 0.70     # 最大相关性阈值

    # 流动性要求
    min_daily_volume: int = 1000000             # 最小日交易量
    max_position_volume_ratio: float = 0.10     # 仓位/成交量最大比例（10%）

    # 风险评分阈值
    approval_threshold: float = 70.0            # 通过阈值
    rejection_threshold: float = 90.0           # 拒绝阈值

    # 数值精度
    precision: int = 6                          # 计算精度


@dataclass
class RiskCheckInput:
    """风控检查输入"""
    # 基本交易信息
    stock_code: str
    trade_direction: str                        # "BUY" or "SELL"
    quantity: int
    price: float

    # 可选信息
    order_type: str = "LIMIT"                   # 订单类型
    account_id: Optional[str] = None            # 账户ID
    strategy_id: Optional[str] = None           # 策略ID

    # 时间戳
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RiskCheckOutput:
    """风控检查输出"""
    # 检查结果
    status: RiskCheckStatus
    approved: bool
    risk_score: float

    # 详细信息
    rule_results: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    rejection_reasons: List[str] = field(default_factory=list)

    # 性能指标
    response_time_ms: float = 0.0
    cache_hit: bool = False

    # 时间戳
    check_time: datetime = field(default_factory=datetime.now)


@dataclass
class RiskRule:
    """风控规则定义"""
    rule_id: str
    rule_type: RiskRuleType
    rule_name: str
    enabled: bool = True
    weight: float = 1.0                         # 规则权重
    threshold: float = 0.0                      # 阈值
    check_function: Optional[Callable] = None   # 检查函数

    def to_dict(self) -> Dict[str, Any]:
        return {
            'rule_id': self.rule_id,
            'rule_type': self.rule_type.value,
            'rule_name': self.rule_name,
            'enabled': self.enabled,
            'weight': self.weight,
            'threshold': self.threshold
        }


class FastPreTradeRiskEngine:
    """
    高性能事前风控检查引擎

    实现超低延迟的交易前风险检查，确保系统稳定性和合规性
    """

    def __init__(self,
                 config: Optional[FastRiskCheckConfig] = None,
                 data_manager=None):
        """
        初始化高性能风控引擎

        Args:
            config: 风控配置
            data_manager: 数据管理器
        """
        self.config = config or FastRiskCheckConfig()
        self.data_manager = data_manager or get_container().resolve("data_manager")
        self.stability_manager = get_stability_manager()

        # 性能监控
        self.check_times = deque(maxlen=1000)
        self.cache_hits = 0
        self.total_checks = 0

        # 缓存系统
        self._cache = {} if self.config.enable_cache else None
        self._cache_timestamps = {} if self.config.enable_cache else None

        # 风控规则引擎
        self.risk_rules = self._initialize_risk_rules()

        # 线程池（用于异步处理）
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="RiskCheck"
        ) if self.config.enable_async else None

        # 实时数据缓存
        self._market_data_cache = {}
        self._position_data_cache = {}
        self._account_data_cache = {}

        logger.info(f"高性能事前风控引擎初始化完成 - 目标响应时间: {self.config.max_response_time_ms}ms")

    def _initialize_risk_rules(self) -> List[RiskRule]:
        """初始化风控规则"""
        rules = [
            # 仓位规模检查
            RiskRule(
                rule_id="R001",
                rule_type=RiskRuleType.POSITION_SIZE,
                rule_name="单仓位规模检查",
                weight=25.0,
                threshold=self.config.max_single_position_ratio,
                check_function=self._check_position_size
            ),

            # 资金检查
            RiskRule(
                rule_id="R002",
                rule_type=RiskRuleType.CASH_REQUIREMENT,
                rule_name="资金充足性检查",
                weight=30.0,
                threshold=self.config.min_cash_reserve_ratio,
                check_function=self._check_cash_requirement
            ),

            # 集中度检查
            RiskRule(
                rule_id="R003",
                rule_type=RiskRuleType.CONCENTRATION,
                rule_name="投资组合集中度检查",
                weight=20.0,
                threshold=self.config.max_sector_exposure,
                check_function=self._check_concentration_risk
            ),

            # 流动性检查
            RiskRule(
                rule_id="R004",
                rule_type=RiskRuleType.LIQUIDITY,
                rule_name="股票流动性检查",
                weight=15.0,
                threshold=self.config.min_daily_volume,
                check_function=self._check_liquidity_risk
            ),

            # 相关性检查
            RiskRule(
                rule_id="R005",
                rule_type=RiskRuleType.CORRELATION,
                rule_name="持仓相关性检查",
                weight=10.0,
                threshold=self.config.max_correlation_threshold,
                check_function=self._check_correlation_risk
            )
        ]

        logger.info(f"已加载 {len(rules)} 条风控规则")
        return rules

    @performance_monitor
    @exception_handler
    async def check_risk_async(self, input_data: RiskCheckInput) -> RiskCheckOutput:
        """
        异步风控检查（主要接口）

        Args:
            input_data: 风控检查输入

        Returns:
            RiskCheckOutput: 检查结果
        """
        start_time = time.perf_counter()

        try:
            # 检查缓存
            cache_key = self._generate_cache_key(input_data)
            cached_result = self._get_cached_result(cache_key)

            if cached_result:
                self.cache_hits += 1
                cached_result.cache_hit = True
                cached_result.response_time_ms = (time.perf_counter() - start_time) * 1000
                return cached_result

            # 并行执行所有风控检查
            tasks = []
            for rule in self.risk_rules:
                if rule.enabled and rule.check_function:
                    task = asyncio.create_task(
                        self._execute_rule_async(rule, input_data)
                    )
                    tasks.append((rule, task))

            # 等待所有检查完成
            rule_results = {}
            total_risk_score = 0.0

            for rule, task in tasks:
                try:
                    rule_result = await task
                    rule_results[rule.rule_id] = rule_result

                    # 计算加权风险评分
                    if rule_result.get('risk_factor', 0) > 0:
                        total_risk_score += rule_result['risk_factor'] * rule.weight

                except Exception as e:
                    logger.warning(f"风控规则 {rule.rule_id} 执行异常: {e}")
                    rule_results[rule.rule_id] = {
                        'passed': False,
                        'risk_factor': 1.0,
                        'message': f"规则执行异常: {str(e)}"
                    }
                    total_risk_score += rule.weight

            # 计算最终结果
            result = self._calculate_final_result(
                input_data, rule_results, total_risk_score, start_time
            )

            # 缓存结果
            if self.config.enable_cache:
                self._cache_result(cache_key, result)

            return result

        except Exception as e:
            logger.error(f"风控检查异常: {e}")
            return RiskCheckOutput(
                status=RiskCheckStatus.ERROR,
                approved=False,
                risk_score=100.0,
                rejection_reasons=[f"系统异常: {str(e)}"],
                response_time_ms=(time.perf_counter() - start_time) * 1000
            )
        finally:
            self.total_checks += 1

    def check_risk_sync(self, input_data: RiskCheckInput) -> RiskCheckOutput:
        """
        同步风控检查接口

        Args:
            input_data: 风控检查输入

        Returns:
            RiskCheckOutput: 检查结果
        """
        if self.config.enable_async:
            # 使用事件循环运行异步检查
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self.check_risk_async(input_data))
                return result
            finally:
                loop.close()
        else:
            # 同步执行
            return asyncio.run(self.check_risk_async(input_data))

    async def _execute_rule_async(self, rule: RiskRule, input_data: RiskCheckInput) -> Dict[str, Any]:
        """异步执行单个风控规则"""
        try:
            loop = asyncio.get_event_loop()

            # 在线程池中执行规则检查（避免阻塞事件循环）
            if self.executor:
                result = await loop.run_in_executor(
                    self.executor, rule.check_function, input_data
                )
            else:
                result = rule.check_function(input_data)

            return result

        except Exception as e:
            logger.warning(f"规则 {rule.rule_id} 执行异常: {e}")
            return {
                'passed': False,
                'risk_factor': 1.0,
                'message': f"规则执行异常: {str(e)}"
            }

    def _check_position_size(self, input_data: RiskCheckInput) -> Dict[str, Any]:
        """检查仓位规模"""
        try:
            # 计算交易金额
            trade_value = input_data.quantity * input_data.price

            # 获取账户总资产（模拟数据）
            total_assets = self._get_total_assets()

            # 计算仓位比例
            position_ratio = trade_value / total_assets if total_assets > 0 else 0

            # 精度处理
            position_ratio = self.stability_manager.round_to_precision(
                position_ratio, self.config.precision
            )

            # 风险评估
            if position_ratio <= self.config.max_single_position_ratio:
                return {
                    'passed': True,
                    'risk_factor': 0.0,
                    'position_ratio': position_ratio,
                    'message': f"仓位比例正常: {position_ratio:.2%}"
                }
            else:
                risk_factor = min(1.0, (position_ratio - self.config.max_single_position_ratio) * 10)
                return {
                    'passed': False,
                    'risk_factor': risk_factor,
                    'position_ratio': position_ratio,
                    'message': f"仓位比例超限: {position_ratio:.2%} > {self.config.max_single_position_ratio:.2%}"
                }

        except Exception as e:
            logger.error(f"仓位规模检查异常: {e}")
            return {
                'passed': False,
                'risk_factor': 1.0,
                'message': f"检查异常: {str(e)}"
            }

    def _check_cash_requirement(self, input_data: RiskCheckInput) -> Dict[str, Any]:
        """检查资金充足性"""
        try:
            # 计算所需资金
            required_cash = input_data.quantity * input_data.price

            # 获取可用资金
            available_cash = self._get_available_cash()

            # 获取总资产
            total_assets = self._get_total_assets()

            # 计算交易后现金储备比例
            remaining_cash = available_cash - required_cash
            reserve_ratio = remaining_cash / total_assets if total_assets > 0 else 0

            # 精度处理
            reserve_ratio = self.stability_manager.round_to_precision(
                reserve_ratio, self.config.precision
            )

            if reserve_ratio >= self.config.min_cash_reserve_ratio and remaining_cash >= 0:
                return {
                    'passed': True,
                    'risk_factor': 0.0,
                    'available_cash': available_cash,
                    'required_cash': required_cash,
                    'reserve_ratio': reserve_ratio,
                    'message': f"资金充足: 现金储备 {reserve_ratio:.2%}"
                }
            else:
                # 资金不足的风险因子
                if remaining_cash < 0:
                    risk_factor = 1.0  # 资金不足，最高风险
                else:
                    shortage = self.config.min_cash_reserve_ratio - reserve_ratio
                    risk_factor = min(1.0, shortage * 5)  # 按缺口比例计算风险

                return {
                    'passed': False,
                    'risk_factor': risk_factor,
                    'available_cash': available_cash,
                    'required_cash': required_cash,
                    'reserve_ratio': reserve_ratio,
                    'message': f"资金不足: 现金储备 {reserve_ratio:.2%} < {self.config.min_cash_reserve_ratio:.2%}"
                }

        except Exception as e:
            logger.error(f"资金检查异常: {e}")
            return {
                'passed': False,
                'risk_factor': 1.0,
                'message': f"检查异常: {str(e)}"
            }

    def _check_concentration_risk(self, input_data: RiskCheckInput) -> Dict[str, Any]:
        """检查集中度风险"""
        try:
            # 获取股票行业信息
            sector = self._get_stock_sector(input_data.stock_code)

            # 计算交易金额
            trade_value = input_data.quantity * input_data.price

            # 获取当前行业敞口
            current_sector_exposure = self._get_sector_exposure(sector)

            # 获取总资产
            total_assets = self._get_total_assets()

            # 计算交易后行业敞口比例
            new_sector_exposure = (current_sector_exposure + trade_value) / total_assets if total_assets > 0 else 0

            # 精度处理
            new_sector_exposure = self.stability_manager.round_to_precision(
                new_sector_exposure, self.config.precision
            )

            if new_sector_exposure <= self.config.max_sector_exposure:
                return {
                    'passed': True,
                    'risk_factor': 0.0,
                    'sector': sector,
                    'current_exposure': current_sector_exposure / total_assets if total_assets > 0 else 0,
                    'new_exposure': new_sector_exposure,
                    'message': f"行业集中度正常: {sector} {new_sector_exposure:.2%}"
                }
            else:
                excess = new_sector_exposure - self.config.max_sector_exposure
                risk_factor = min(1.0, excess * 3)

                return {
                    'passed': False,
                    'risk_factor': risk_factor,
                    'sector': sector,
                    'current_exposure': current_sector_exposure / total_assets if total_assets > 0 else 0,
                    'new_exposure': new_sector_exposure,
                    'message': f"行业集中度超限: {sector} {new_sector_exposure:.2%} > {self.config.max_sector_exposure:.2%}"
                }

        except Exception as e:
            logger.error(f"集中度检查异常: {e}")
            return {
                'passed': False,
                'risk_factor': 0.5,  # 中等风险
                'message': f"检查异常: {str(e)}"
            }

    def _check_liquidity_risk(self, input_data: RiskCheckInput) -> Dict[str, Any]:
        """检查流动性风险"""
        try:
            # 获取股票流动性数据
            daily_volume = self._get_daily_volume(input_data.stock_code)
            avg_volume_20d = self._get_avg_volume_20d(input_data.stock_code)

            # 计算仓位占成交量比例
            position_volume_ratio = input_data.quantity / daily_volume if daily_volume > 0 else 1.0

            # 精度处理
            position_volume_ratio = self.stability_manager.round_to_precision(
                position_volume_ratio, self.config.precision
            )

            # 流动性风险评估
            liquidity_adequate = (
                daily_volume >= self.config.min_daily_volume and
                position_volume_ratio <= self.config.max_position_volume_ratio
            )

            if liquidity_adequate:
                return {
                    'passed': True,
                    'risk_factor': 0.0,
                    'daily_volume': daily_volume,
                    'avg_volume_20d': avg_volume_20d,
                    'position_volume_ratio': position_volume_ratio,
                    'message': f"流动性充足: 日均成交量 {daily_volume:,}"
                }
            else:
                # 计算流动性风险因子
                volume_risk = 0.0 if daily_volume >= self.config.min_daily_volume else 0.5
                ratio_risk = max(0.0, (position_volume_ratio - self.config.max_position_volume_ratio) * 2)
                risk_factor = min(1.0, volume_risk + ratio_risk)

                return {
                    'passed': False,
                    'risk_factor': risk_factor,
                    'daily_volume': daily_volume,
                    'avg_volume_20d': avg_volume_20d,
                    'position_volume_ratio': position_volume_ratio,
                    'message': f"流动性不足: 成交量 {daily_volume:,} 或占比 {position_volume_ratio:.2%} 过高"
                }

        except Exception as e:
            logger.error(f"流动性检查异常: {e}")
            return {
                'passed': False,
                'risk_factor': 0.3,  # 中等风险
                'message': f"检查异常: {str(e)}"
            }

    def _check_correlation_risk(self, input_data: RiskCheckInput) -> Dict[str, Any]:
        """检查相关性风险"""
        try:
            # 获取当前持仓
            current_positions = self._get_current_positions()

            if not current_positions:
                return {
                    'passed': True,
                    'risk_factor': 0.0,
                    'message': "无持仓，相关性风险为零"
                }

            # 计算与现有持仓的平均相关性
            correlations = []
            for pos_stock_code in current_positions:
                if pos_stock_code != input_data.stock_code:
                    corr = self._get_stock_correlation(input_data.stock_code, pos_stock_code)
                    if corr is not None:
                        correlations.append(abs(corr))

            if not correlations:
                return {
                    'passed': True,
                    'risk_factor': 0.0,
                    'message': "无法计算相关性"
                }

            # 计算平均相关性
            avg_correlation = np.mean(correlations)

            # 精度处理
            avg_correlation = self.stability_manager.round_to_precision(
                avg_correlation, self.config.precision
            )

            if avg_correlation <= self.config.max_correlation_threshold:
                return {
                    'passed': True,
                    'risk_factor': 0.0,
                    'avg_correlation': avg_correlation,
                    'max_correlation': max(correlations),
                    'message': f"相关性正常: 平均相关性 {avg_correlation:.3f}"
                }
            else:
                excess = avg_correlation - self.config.max_correlation_threshold
                risk_factor = min(1.0, excess * 2)

                return {
                    'passed': False,
                    'risk_factor': risk_factor,
                    'avg_correlation': avg_correlation,
                    'max_correlation': max(correlations),
                    'message': f"相关性过高: 平均相关性 {avg_correlation:.3f} > {self.config.max_correlation_threshold:.3f}"
                }

        except Exception as e:
            logger.error(f"相关性检查异常: {e}")
            return {
                'passed': False,
                'risk_factor': 0.2,  # 低风险
                'message': f"检查异常: {str(e)}"
            }

    def _calculate_final_result(self,
                               input_data: RiskCheckInput,
                               rule_results: Dict[str, Any],
                               total_risk_score: float,
                               start_time: float) -> RiskCheckOutput:
        """计算最终风控结果"""

        # 精度处理
        total_risk_score = self.stability_manager.round_to_precision(
            total_risk_score, self.config.precision
        )

        # 收集警告和拒绝原因
        warnings = []
        rejection_reasons = []

        for rule_id, result in rule_results.items():
            if not result.get('passed', True):
                message = result.get('message', '未知错误')
                if result.get('risk_factor', 0) >= 0.8:
                    rejection_reasons.append(f"{rule_id}: {message}")
                else:
                    warnings.append(f"{rule_id}: {message}")

        # 确定最终状态
        if total_risk_score <= self.config.approval_threshold:
            status = RiskCheckStatus.APPROVED
            approved = True
        elif total_risk_score >= self.config.rejection_threshold:
            status = RiskCheckStatus.REJECTED
            approved = False
        else:
            status = RiskCheckStatus.CONDITIONAL
            approved = len(rejection_reasons) == 0  # 有严重问题就拒绝

        # 计算响应时间
        response_time_ms = (time.perf_counter() - start_time) * 1000

        # 记录性能指标
        self.check_times.append(response_time_ms)

        # 响应时间超标警告
        if response_time_ms > self.config.max_response_time_ms:
            warnings.append(f"响应时间超标: {response_time_ms:.2f}ms > {self.config.max_response_time_ms}ms")

        return RiskCheckOutput(
            status=status,
            approved=approved,
            risk_score=total_risk_score,
            rule_results=rule_results,
            warnings=warnings,
            rejection_reasons=rejection_reasons,
            response_time_ms=self.stability_manager.round_to_precision(response_time_ms, 2)
        )

    # 缓存相关方法
    def _generate_cache_key(self, input_data: RiskCheckInput) -> str:
        """生成缓存键"""
        return f"risk_check_{input_data.stock_code}_{input_data.trade_direction}_{input_data.quantity}_{input_data.price}"

    def _get_cached_result(self, cache_key: str) -> Optional[RiskCheckOutput]:
        """获取缓存结果"""
        if not self.config.enable_cache or not self._cache:
            return None

        if cache_key in self._cache:
            cached_time = self._cache_timestamps.get(cache_key)
            if cached_time and (datetime.now() - cached_time).total_seconds() < self.config.cache_ttl_seconds:
                return self._cache[cache_key]
            else:
                # 缓存过期，清理
                del self._cache[cache_key]
                del self._cache_timestamps[cache_key]

        return None

    def _cache_result(self, cache_key: str, result: RiskCheckOutput):
        """缓存结果"""
        if not self.config.enable_cache or not self._cache:
            return

        self._cache[cache_key] = result
        self._cache_timestamps[cache_key] = datetime.now()

        # 清理过期缓存
        self._cleanup_expired_cache()

    def _cleanup_expired_cache(self):
        """清理过期缓存"""
        if not self._cache:
            return

        current_time = datetime.now()
        expired_keys = []

        for key, timestamp in self._cache_timestamps.items():
            if (current_time - timestamp).total_seconds() > self.config.cache_ttl_seconds:
                expired_keys.append(key)

        for key in expired_keys:
            self._cache.pop(key, None)
            self._cache_timestamps.pop(key, None)

    # 数据获取方法（模拟实现）
    @lru_cache(maxsize=1000)
    def _get_total_assets(self) -> float:
        """获取账户总资产（模拟）"""
        return 1000000.0  # 100万

    @lru_cache(maxsize=1000)
    def _get_available_cash(self) -> float:
        """获取可用现金（模拟）"""
        return 200000.0  # 20万

    @lru_cache(maxsize=1000)
    def _get_stock_sector(self, stock_code: str) -> str:
        """获取股票行业（模拟）"""
        # 简单的行业分类模拟
        if stock_code.startswith('00'):
            return '金融'
        elif stock_code.startswith('30'):
            return '科技'
        elif stock_code.startswith('60'):
            return '制造业'
        else:
            return '其他'

    @lru_cache(maxsize=1000)
    def _get_sector_exposure(self, sector: str) -> float:
        """获取行业敞口（模拟）"""
        # 模拟不同行业的当前敞口
        sector_exposures = {
            '金融': 150000.0,
            '科技': 200000.0,
            '制造业': 100000.0,
            '其他': 50000.0
        }
        return sector_exposures.get(sector, 0.0)

    @lru_cache(maxsize=1000)
    def _get_daily_volume(self, stock_code: str) -> int:
        """获取日成交量（模拟）"""
        # 根据股票代码模拟成交量
        if stock_code.startswith('00'):
            return 2000000  # 200万股
        elif stock_code.startswith('30'):
            return 5000000  # 500万股
        else:
            return 1000000  # 100万股

    @lru_cache(maxsize=1000)
    def _get_avg_volume_20d(self, stock_code: str) -> int:
        """获取20日平均成交量（模拟）"""
        return int(self._get_daily_volume(stock_code) * 0.8)

    @lru_cache(maxsize=100)
    def _get_current_positions(self) -> List[str]:
        """获取当前持仓股票列表（模拟）"""
        return ['000001', '000002', '300001', '600001']

    @lru_cache(maxsize=10000)
    def _get_stock_correlation(self, stock1: str, stock2: str) -> Optional[float]:
        """获取两只股票的相关系数（模拟）"""
        # 简单的相关性模拟
        hash_value = hash(f"{stock1}_{stock2}") % 1000
        return (hash_value / 1000 - 0.5) * 2  # -1 到 1 之间

    # 性能统计方法
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        check_times_list = list(self.check_times)

        return {
            'total_checks': self.total_checks,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': self.cache_hits / self.total_checks if self.total_checks > 0 else 0.0,
            'avg_response_time_ms': np.mean(check_times_list) if check_times_list else 0.0,
            'p50_response_time_ms': np.percentile(check_times_list, 50) if check_times_list else 0.0,
            'p95_response_time_ms': np.percentile(check_times_list, 95) if check_times_list else 0.0,
            'p99_response_time_ms': np.percentile(check_times_list, 99) if check_times_list else 0.0,
            'max_response_time_ms': max(check_times_list) if check_times_list else 0.0,
            'target_response_time_ms': self.config.max_response_time_ms,
            'sla_compliance_rate': sum(1 for t in check_times_list if t <= self.config.max_response_time_ms) / len(check_times_list) if check_times_list else 0.0
        }

    def shutdown(self):
        """关闭引擎"""
        if self.executor:
            self.executor.shutdown(wait=True)
        logger.info("高性能事前风控引擎已关闭")


# 全局实例管理
_fast_risk_engine = None

def get_fast_pre_trade_risk_engine(
    config: Optional[FastRiskCheckConfig] = None,
    data_manager=None
) -> FastPreTradeRiskEngine:
    """
    获取高性能事前风控引擎实例（单例模式）

    Args:
        config: 风控配置
        data_manager: 数据管理器

    Returns:
        FastPreTradeRiskEngine: 风控引擎实例
    """
    global _fast_risk_engine

    if _fast_risk_engine is None:
        _fast_risk_engine = FastPreTradeRiskEngine(
            config=config,
            data_manager=data_manager
        )

    return _fast_risk_engine


if __name__ == "__main__":
    # 演示使用
    import asyncio

    async def test_fast_risk_engine():
        print("=== 高性能事前风控引擎演示 ===")

        # 创建引擎
        engine = get_fast_pre_trade_risk_engine()

        # 模拟交易请求
        test_requests = [
            RiskCheckInput(
                stock_code="000001",
                trade_direction="BUY",
                quantity=1000,
                price=12.50
            ),
            RiskCheckInput(
                stock_code="300001",
                trade_direction="BUY",
                quantity=5000,  # 较大仓位
                price=25.80
            ),
            RiskCheckInput(
                stock_code="600001",
                trade_direction="SELL",
                quantity=2000,
                price=18.90
            )
        ]

        # 并行执行风控检查
        tasks = [engine.check_risk_async(req) for req in test_requests]
        results = await asyncio.gather(*tasks)

        # 显示结果
        for i, (req, result) in enumerate(zip(test_requests, results)):
            print(f"\n--- 检查结果 {i+1} ---")
            print(f"股票代码: {req.stock_code}")
            print(f"交易方向: {req.trade_direction}")
            print(f"数量: {req.quantity:,}")
            print(f"价格: {req.price:.2f}")
            print(f"检查状态: {result.status.value}")
            print(f"是否通过: {result.approved}")
            print(f"风险评分: {result.risk_score:.2f}")
            print(f"响应时间: {result.response_time_ms:.2f}ms")

            if result.warnings:
                print(f"警告: {', '.join(result.warnings)}")
            if result.rejection_reasons:
                print(f"拒绝原因: {', '.join(result.rejection_reasons)}")

        # 性能统计
        print(f"\n--- 性能统计 ---")
        stats = engine.get_performance_stats()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.3f}")
            else:
                print(f"{key}: {value}")

        # 关闭引擎
        engine.shutdown()

    # 运行演示
    asyncio.run(test_fast_risk_engine())