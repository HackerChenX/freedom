#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
事前风控系统

专业金融量化交易系统的事前风险控制模块，遵循监管合规性标准。
提供交易前全面风险检查，确保每笔交易符合风控要求。

主要功能：
1. 资金充足性检查 - 确保账户资金足够支撑交易
2. 持仓限制检查 - 控制单股票、单行业、总仓位限制
3. 杠杆控制检查 - 防范过度杠杆风险
4. 市场时间验证 - 确保在正确的交易时间执行
5. 交易规则验证 - 符合交易所和监管规定
6. 黑名单白名单机制 - 禁止/允许特定股票交易
7. 风险评级限制 - 基于个股风险评级的交易限制
"""

import time
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta, time as dt_time
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from decimal import Decimal, getcontext

from utils.logger import get_logger
from utils.exception_handler import exception_handler
from utils.performance_monitor import performance_monitor
from utils.unified_container import get_container
from utils.numerical_stability_manager import get_stability_manager
from monitoring.risk_monitor import RiskMonitoringSystem, RiskLevel

logger = get_logger(__name__)

# 设置高精度计算
getcontext().prec = 28

class PreTradeCheckResult(Enum):
    """事前检查结果枚举"""
    APPROVED = "交易通过"
    REJECTED = "交易拒绝"
    WARNING = "交易警告"
    CONDITIONAL = "条件通过"

class RiskControlLevel(Enum):
    """风控级别枚举"""
    STRICT = "严格模式"      # 机构投资者标准
    MODERATE = "适中模式"    # 专业投资者标准
    RELAXED = "宽松模式"     # 一般投资者标准
    CUSTOM = "自定义模式"    # 自定义风控参数

class TradeDirection(Enum):
    """交易方向枚举"""
    BUY = "买入"
    SELL = "卖出"
    SHORT = "做空"
    COVER = "平仓"

@dataclass
class TradeRequest:
    """交易请求数据类"""
    stock_code: str              # 股票代码
    stock_name: str              # 股票名称
    direction: TradeDirection    # 交易方向
    quantity: int                # 交易数量（股）
    price: float                # 交易价格
    order_type: str             # 订单类型（市价单/限价单等）
    request_time: datetime       # 请求时间
    account_id: str             # 账户ID
    strategy_id: Optional[str]   # 策略ID
    user_id: Optional[str]      # 用户ID
    notes: Optional[str]        # 备注信息

@dataclass
class PreTradeCheckConfig:
    """事前风控配置"""
    # 资金管理配置
    max_single_position_ratio: float = 0.10    # 单股票最大仓位比例（10%）
    max_total_position_ratio: float = 0.95     # 总仓位最大比例（95%）
    min_cash_reserve_ratio: float = 0.05       # 最小现金储备比例（5%）
    max_sector_ratio: float = 0.30             # 单行业最大仓位比例（30%）

    # 杠杆控制配置
    max_leverage_ratio: float = 1.0            # 最大杠杆比例（1倍）
    margin_safety_factor: float = 1.2          # 保证金安全系数

    # 交易限制配置
    max_daily_trades: int = 100                # 每日最大交易次数
    max_single_order_value: float = 1000000   # 单笔订单最大金额（100万）
    min_order_value: float = 1000             # 单笔订单最小金额（1000元）

    # 风险评级限制
    allow_high_risk_stocks: bool = False       # 是否允许高风险股票
    max_risk_stock_ratio: float = 0.05        # 高风险股票最大仓位比例

    # 市场时间配置
    trading_start_time: dt_time = dt_time(9, 30)    # 交易开始时间
    trading_end_time: dt_time = dt_time(15, 0)      # 交易结束时间
    allow_after_hours: bool = False                  # 是否允许盘后交易

    # 价格限制配置
    max_price_deviation: float = 0.10          # 最大价格偏离比例（10%）
    enable_price_check: bool = True            # 是否启用价格检查

@dataclass
class AccountInfo:
    """账户信息数据类"""
    account_id: str
    total_assets: float          # 总资产
    available_cash: float        # 可用现金
    market_value: float         # 市值
    total_liability: float      # 总负债
    margin_used: float          # 已用保证金
    margin_available: float     # 可用保证金
    positions: List[Dict]       # 持仓列表
    daily_trades_count: int     # 当日交易次数
    frozen_amount: float        # 冻结资金

@dataclass
class PreTradeCheckResult:
    """事前检查结果数据类"""
    result: PreTradeCheckResult        # 检查结果
    trade_request: TradeRequest        # 原始交易请求
    passed_checks: List[str]          # 通过的检查项
    failed_checks: List[str]          # 失败的检查项
    warning_messages: List[str]       # 警告信息
    rejection_reasons: List[str]      # 拒绝原因
    risk_score: float                 # 整体风险评分
    check_duration: float            # 检查耗时
    timestamp: datetime              # 检查时间
    detailed_results: Dict[str, Any] # 详细检查结果

class PreTradeRiskController:
    """事前风险控制器"""

    def __init__(self, config: PreTradeCheckConfig = None, risk_level: RiskControlLevel = RiskControlLevel.MODERATE):
        """
        初始化事前风险控制器

        Args:
            config: 风控配置
            risk_level: 风控级别
        """
        self.config = config or PreTradeCheckConfig()
        self.risk_level = risk_level
        self.stability_manager = get_stability_manager()
        self.container = get_container()

        # 初始化组件
        self._initialize_components()

        # 黑名单和白名单
        self.blacklist: set = set()        # 股票黑名单
        self.whitelist: set = set()        # 股票白名单
        self.sector_limits: Dict[str, float] = {}  # 行业限制

        # 缓存和性能优化
        self._cache = {}
        self._cache_ttl = 60  # 缓存有效期（秒）
        self._lock = threading.RLock()

        logger.info(f"事前风控系统初始化完成，风控级别: {risk_level.value}")

    def _initialize_components(self):
        """初始化系统组件"""
        try:
            # 获取风险监控系统
            self.risk_monitor = RiskMonitoringSystem()

            # 获取数据访问接口
            try:
                from db.interfaces.data_access_interface import DataAccessInterface
                self.data_access = self.container.resolve(DataAccessInterface)
            except Exception as e:
                logger.warning(f"无法获取数据访问接口: {e}")
                self.data_access = None

        except Exception as e:
            logger.error(f"初始化组件失败: {e}")
            raise

    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=0.01)  # 风控检查要求10ms内完成
    def check_trade_request(self, trade_request: TradeRequest, account_info: AccountInfo) -> PreTradeCheckResult:
        """
        执行事前交易风控检查

        Args:
            trade_request: 交易请求
            account_info: 账户信息

        Returns:
            PreTradeCheckResult: 检查结果
        """
        check_start_time = time.time()

        # 初始化检查结果
        passed_checks = []
        failed_checks = []
        warning_messages = []
        rejection_reasons = []

        try:
            # 1. 基础数据验证
            basic_check = self._check_basic_data(trade_request, account_info)
            if basic_check['passed']:
                passed_checks.extend(basic_check['checks'])
            else:
                failed_checks.extend(basic_check['checks'])
                rejection_reasons.extend(basic_check['reasons'])

            # 2. 市场时间检查
            time_check = self._check_trading_time(trade_request)
            if time_check['passed']:
                passed_checks.extend(time_check['checks'])
            else:
                failed_checks.extend(time_check['checks'])
                rejection_reasons.extend(time_check['reasons'])

            # 3. 黑白名单检查
            list_check = self._check_stock_lists(trade_request)
            if list_check['passed']:
                passed_checks.extend(list_check['checks'])
            else:
                failed_checks.extend(list_check['checks'])
                rejection_reasons.extend(list_check['reasons'])

            # 4. 资金充足性检查
            fund_check = self._check_fund_sufficiency(trade_request, account_info)
            if fund_check['passed']:
                passed_checks.extend(fund_check['checks'])
            else:
                failed_checks.extend(fund_check['checks'])
                rejection_reasons.extend(fund_check['reasons'])

            # 5. 持仓限制检查
            position_check = self._check_position_limits(trade_request, account_info)
            if position_check['passed']:
                passed_checks.extend(position_check['checks'])
            else:
                failed_checks.extend(position_check['checks'])
                rejection_reasons.extend(position_check['reasons'])

            # 6. 杠杆控制检查
            leverage_check = self._check_leverage_limits(trade_request, account_info)
            if leverage_check['passed']:
                passed_checks.extend(leverage_check['checks'])
            else:
                failed_checks.extend(leverage_check['checks'])
                rejection_reasons.extend(leverage_check['reasons'])

            # 7. 价格合理性检查
            price_check = self._check_price_reasonableness(trade_request)
            if price_check['passed']:
                passed_checks.extend(price_check['checks'])
            elif price_check['warning']:
                warning_messages.extend(price_check['warnings'])
                passed_checks.extend(price_check['checks'])
            else:
                failed_checks.extend(price_check['checks'])
                rejection_reasons.extend(price_check['reasons'])

            # 8. 个股风险评级检查
            risk_check = self._check_stock_risk_rating(trade_request)
            if risk_check['passed']:
                passed_checks.extend(risk_check['checks'])
            elif risk_check['warning']:
                warning_messages.extend(risk_check['warnings'])
                passed_checks.extend(risk_check['checks'])
            else:
                failed_checks.extend(risk_check['checks'])
                rejection_reasons.extend(risk_check['reasons'])

            # 9. 交易频率检查
            frequency_check = self._check_trading_frequency(trade_request, account_info)
            if frequency_check['passed']:
                passed_checks.extend(frequency_check['checks'])
            elif frequency_check['warning']:
                warning_messages.extend(frequency_check['warnings'])
                passed_checks.extend(frequency_check['checks'])
            else:
                failed_checks.extend(frequency_check['checks'])
                rejection_reasons.extend(frequency_check['reasons'])

            # 10. 订单规模检查
            size_check = self._check_order_size(trade_request, account_info)
            if size_check['passed']:
                passed_checks.extend(size_check['checks'])
            elif size_check['warning']:
                warning_messages.extend(size_check['warnings'])
                passed_checks.extend(size_check['checks'])
            else:
                failed_checks.extend(size_check['checks'])
                rejection_reasons.extend(size_check['reasons'])

            # 计算整体风险评分
            risk_score = self._calculate_overall_risk_score(
                trade_request, account_info, passed_checks, failed_checks, warning_messages
            )

            # 确定最终结果
            if failed_checks:
                final_result = PreTradeCheckResult.REJECTED
            elif warning_messages:
                final_result = PreTradeCheckResult.WARNING
            else:
                final_result = PreTradeCheckResult.APPROVED

            # 计算检查耗时
            check_duration = time.time() - check_start_time

            # 构建详细结果
            detailed_results = {
                'basic_check': basic_check,
                'time_check': time_check,
                'list_check': list_check,
                'fund_check': fund_check,
                'position_check': position_check,
                'leverage_check': leverage_check,
                'price_check': price_check,
                'risk_check': risk_check,
                'frequency_check': frequency_check,
                'size_check': size_check,
                'account_summary': self._generate_account_summary(account_info),
                'trade_impact': self._calculate_trade_impact(trade_request, account_info)
            }

            result = PreTradeCheckResult(
                result=final_result,
                trade_request=trade_request,
                passed_checks=passed_checks,
                failed_checks=failed_checks,
                warning_messages=warning_messages,
                rejection_reasons=rejection_reasons,
                risk_score=risk_score,
                check_duration=check_duration,
                timestamp=datetime.now(),
                detailed_results=detailed_results
            )

            # 记录检查结果
            self._log_check_result(result)

            return result

        except Exception as e:
            logger.error(f"事前风控检查失败: {e}")
            # 安全起见，检查失败时拒绝交易
            return PreTradeCheckResult(
                result=PreTradeCheckResult.REJECTED,
                trade_request=trade_request,
                passed_checks=[],
                failed_checks=['系统检查失败'],
                warning_messages=[],
                rejection_reasons=[f"系统错误: {str(e)}"],
                risk_score=100.0,
                check_duration=time.time() - check_start_time,
                timestamp=datetime.now(),
                detailed_results={}
            )

    def _check_basic_data(self, trade_request: TradeRequest, account_info: AccountInfo) -> Dict[str, Any]:
        """基础数据验证"""
        checks = []
        reasons = []

        # 验证股票代码
        if not trade_request.stock_code or len(trade_request.stock_code) != 6:
            reasons.append("无效的股票代码")
        else:
            checks.append("股票代码格式正确")

        # 验证交易数量
        if trade_request.quantity <= 0:
            reasons.append("交易数量必须大于0")
        elif trade_request.quantity % 100 != 0:
            reasons.append("交易数量必须是100的整数倍")
        else:
            checks.append("交易数量有效")

        # 验证交易价格
        if trade_request.price <= 0:
            reasons.append("交易价格必须大于0")
        else:
            checks.append("交易价格有效")

        # 验证账户信息
        if not account_info.account_id:
            reasons.append("无效的账户ID")
        else:
            checks.append("账户ID有效")

        # 验证账户资金状态
        if account_info.total_assets <= 0:
            reasons.append("账户总资产必须大于0")
        else:
            checks.append("账户总资产正常")

        return {
            'passed': len(reasons) == 0,
            'checks': checks,
            'reasons': reasons
        }

    def _check_trading_time(self, trade_request: TradeRequest) -> Dict[str, Any]:
        """市场时间检查"""
        checks = []
        reasons = []

        current_time = datetime.now()

        # 检查是否在交易日（简化版本，实际需要考虑节假日）
        if current_time.weekday() >= 5:  # 周六周日
            reasons.append("非交易日，禁止交易")
        else:
            checks.append("交易日检查通过")

        # 检查是否在交易时间
        current_time_only = current_time.time()

        # 上午交易时间: 9:30-11:30
        morning_start = dt_time(9, 30)
        morning_end = dt_time(11, 30)

        # 下午交易时间: 13:00-15:00
        afternoon_start = dt_time(13, 0)
        afternoon_end = dt_time(15, 0)

        in_morning_session = morning_start <= current_time_only <= morning_end
        in_afternoon_session = afternoon_start <= current_time_only <= afternoon_end

        if not (in_morning_session or in_afternoon_session):
            if not self.config.allow_after_hours:
                reasons.append("非交易时间，禁止交易")
            else:
                checks.append("盘后交易时间检查通过")
        else:
            checks.append("交易时间检查通过")

        return {
            'passed': len(reasons) == 0,
            'checks': checks,
            'reasons': reasons
        }

    def _check_stock_lists(self, trade_request: TradeRequest) -> Dict[str, Any]:
        """黑白名单检查"""
        checks = []
        reasons = []

        stock_code = trade_request.stock_code

        # 黑名单检查
        if stock_code in self.blacklist:
            reasons.append(f"股票{stock_code}在黑名单中，禁止交易")
        else:
            checks.append("黑名单检查通过")

        # 白名单检查（如果启用）
        if self.whitelist:  # 如果设置了白名单
            if stock_code not in self.whitelist:
                reasons.append(f"股票{stock_code}不在白名单中，禁止交易")
            else:
                checks.append("白名单检查通过")
        else:
            checks.append("白名单检查跳过（未启用）")

        return {
            'passed': len(reasons) == 0,
            'checks': checks,
            'reasons': reasons
        }

    def _check_fund_sufficiency(self, trade_request: TradeRequest, account_info: AccountInfo) -> Dict[str, Any]:
        """资金充足性检查"""
        checks = []
        reasons = []

        trade_value = trade_request.quantity * trade_request.price

        if trade_request.direction == TradeDirection.BUY:
            # 买入交易，检查可用现金
            required_cash = trade_value * 1.001  # 加上手续费

            if account_info.available_cash < required_cash:
                reasons.append(f"可用现金不足，需要{required_cash:.2f}元，可用{account_info.available_cash:.2f}元")
            else:
                checks.append("可用现金充足")

            # 检查现金储备比例
            remaining_cash = account_info.available_cash - required_cash
            cash_reserve_ratio = remaining_cash / account_info.total_assets if account_info.total_assets > 0 else 0

            if cash_reserve_ratio < self.config.min_cash_reserve_ratio:
                reasons.append(f"现金储备比例过低，当前{cash_reserve_ratio:.2%}，最低要求{self.config.min_cash_reserve_ratio:.2%}")
            else:
                checks.append("现金储备比例充足")

        else:  # 卖出交易
            # 检查是否有足够的持仓
            current_position = self._get_position_quantity(account_info, trade_request.stock_code)

            if current_position < trade_request.quantity:
                reasons.append(f"持仓数量不足，需要{trade_request.quantity}股，持有{current_position}股")
            else:
                checks.append("持仓数量充足")

        return {
            'passed': len(reasons) == 0,
            'checks': checks,
            'reasons': reasons
        }

    def _check_position_limits(self, trade_request: TradeRequest, account_info: AccountInfo) -> Dict[str, Any]:
        """持仓限制检查"""
        checks = []
        reasons = []

        if trade_request.direction == TradeDirection.BUY:
            trade_value = trade_request.quantity * trade_request.price

            # 单股票持仓比例检查
            current_position_value = self._get_position_value(account_info, trade_request.stock_code)
            new_position_value = current_position_value + trade_value
            single_position_ratio = new_position_value / account_info.total_assets if account_info.total_assets > 0 else 0

            if single_position_ratio > self.config.max_single_position_ratio:
                reasons.append(f"单股票持仓比例过高，交易后将达到{single_position_ratio:.2%}，最大允许{self.config.max_single_position_ratio:.2%}")
            else:
                checks.append("单股票持仓比例检查通过")

            # 总仓位比例检查
            new_total_market_value = account_info.market_value + trade_value
            total_position_ratio = new_total_market_value / account_info.total_assets if account_info.total_assets > 0 else 0

            if total_position_ratio > self.config.max_total_position_ratio:
                reasons.append(f"总仓位比例过高，交易后将达到{total_position_ratio:.2%}，最大允许{self.config.max_total_position_ratio:.2%}")
            else:
                checks.append("总仓位比例检查通过")

            # 行业集中度检查（如果有行业信息）
            sector = self._get_stock_sector(trade_request.stock_code)
            if sector:
                current_sector_value = self._get_sector_value(account_info, sector)
                new_sector_value = current_sector_value + trade_value
                sector_ratio = new_sector_value / account_info.total_assets if account_info.total_assets > 0 else 0

                max_sector_ratio = self.sector_limits.get(sector, self.config.max_sector_ratio)

                if sector_ratio > max_sector_ratio:
                    reasons.append(f"行业{sector}持仓比例过高，交易后将达到{sector_ratio:.2%}，最大允许{max_sector_ratio:.2%}")
                else:
                    checks.append("行业持仓比例检查通过")
            else:
                checks.append("行业持仓比例检查跳过（无行业信息）")
        else:
            checks.append("卖出交易跳过持仓限制检查")

        return {
            'passed': len(reasons) == 0,
            'checks': checks,
            'reasons': reasons
        }

    def _check_leverage_limits(self, trade_request: TradeRequest, account_info: AccountInfo) -> Dict[str, Any]:
        """杠杆控制检查"""
        checks = []
        reasons = []

        # 计算当前杠杆比例
        current_leverage = account_info.total_liability / (account_info.total_assets - account_info.total_liability) if (account_info.total_assets - account_info.total_liability) > 0 else 0

        if current_leverage > self.config.max_leverage_ratio:
            reasons.append(f"当前杠杆比例过高{current_leverage:.2f}，最大允许{self.config.max_leverage_ratio:.2f}")
        else:
            checks.append("当前杠杆比例检查通过")

        # 保证金充足性检查
        if account_info.margin_available < 0:
            reasons.append("保证金不足，无法进行交易")
        else:
            checks.append("保证金充足性检查通过")

        # 保证金安全系数检查
        if trade_request.direction == TradeDirection.BUY:
            required_margin = trade_request.quantity * trade_request.price * 0.5  # 假设50%保证金要求
            safety_margin = required_margin * self.config.margin_safety_factor

            if account_info.margin_available < safety_margin:
                reasons.append(f"保证金安全系数不足，需要{safety_margin:.2f}元，可用{account_info.margin_available:.2f}元")
            else:
                checks.append("保证金安全系数检查通过")
        else:
            checks.append("卖出交易跳过保证金检查")

        return {
            'passed': len(reasons) == 0,
            'checks': checks,
            'reasons': reasons
        }

    def _check_price_reasonableness(self, trade_request: TradeRequest) -> Dict[str, Any]:
        """价格合理性检查"""
        checks = []
        reasons = []
        warnings = []

        if not self.config.enable_price_check:
            checks.append("价格检查跳过（未启用）")
            return {'passed': True, 'checks': checks, 'reasons': reasons, 'warnings': warnings, 'warning': False}

        # 获取当前市场价格（模拟）
        current_price = self._get_current_market_price(trade_request.stock_code)

        if current_price is None:
            warnings.append("无法获取当前市场价格，跳过价格合理性检查")
            checks.append("价格检查跳过（无市场数据）")
            return {'passed': True, 'checks': checks, 'reasons': reasons, 'warnings': warnings, 'warning': True}

        # 计算价格偏离
        price_deviation = abs(trade_request.price - current_price) / current_price if current_price > 0 else 0

        if price_deviation > self.config.max_price_deviation:
            if self.risk_level == RiskControlLevel.STRICT:
                reasons.append(f"价格偏离过大{price_deviation:.2%}，最大允许{self.config.max_price_deviation:.2%}")
            else:
                warnings.append(f"价格偏离较大{price_deviation:.2%}，请注意风险")
                checks.append("价格合理性检查通过（警告）")
        else:
            checks.append("价格合理性检查通过")

        # 价格有效性检查
        if trade_request.price <= 0:
            reasons.append("交易价格必须大于0")
        elif trade_request.price > current_price * 2:  # 价格不能超过当前价格2倍
            reasons.append(f"交易价格过高，当前价格{current_price:.2f}，交易价格{trade_request.price:.2f}")
        else:
            checks.append("价格有效性检查通过")

        is_warning_only = len(warnings) > 0 and len(reasons) == 0

        return {
            'passed': len(reasons) == 0,
            'warning': is_warning_only,
            'checks': checks,
            'reasons': reasons,
            'warnings': warnings
        }

    def _check_stock_risk_rating(self, trade_request: TradeRequest) -> Dict[str, Any]:
        """个股风险评级检查"""
        checks = []
        reasons = []
        warnings = []

        try:
            # 获取个股风险评级
            risk_metrics = self.risk_monitor.stock_monitor.monitor_stock_risk(trade_request.stock_code, trade_request.stock_name)

            risk_level = risk_metrics.risk_level
            risk_score = risk_metrics.risk_score

            # 高风险股票检查
            if risk_level in ['高风险', '极高风险']:
                if not self.config.allow_high_risk_stocks:
                    reasons.append(f"股票{trade_request.stock_code}为{risk_level}等级，系统禁止交易高风险股票")
                else:
                    warnings.append(f"股票{trade_request.stock_code}为{risk_level}等级，请谨慎交易")
                    checks.append("高风险股票检查通过（警告）")
            else:
                checks.append("股票风险等级检查通过")

            # 风险评分检查
            if risk_score >= 80:
                if self.risk_level == RiskControlLevel.STRICT:
                    reasons.append(f"股票风险评分过高{risk_score:.1f}，严格模式下禁止交易")
                else:
                    warnings.append(f"股票风险评分较高{risk_score:.1f}，建议谨慎操作")
                    checks.append("风险评分检查通过（警告）")
            else:
                checks.append("风险评分检查通过")

        except Exception as e:
            logger.warning(f"获取股票风险评级失败: {e}")
            warnings.append("无法获取股票风险评级，跳过相关检查")
            checks.append("风险评级检查跳过（系统异常）")

        is_warning_only = len(warnings) > 0 and len(reasons) == 0

        return {
            'passed': len(reasons) == 0,
            'warning': is_warning_only,
            'checks': checks,
            'reasons': reasons,
            'warnings': warnings
        }

    def _check_trading_frequency(self, trade_request: TradeRequest, account_info: AccountInfo) -> Dict[str, Any]:
        """交易频率检查"""
        checks = []
        reasons = []
        warnings = []

        # 检查当日交易次数
        if account_info.daily_trades_count >= self.config.max_daily_trades:
            reasons.append(f"当日交易次数已达上限{self.config.max_daily_trades}")
        elif account_info.daily_trades_count >= self.config.max_daily_trades * 0.8:
            warnings.append(f"当日交易次数接近上限，已交易{account_info.daily_trades_count}次")
            checks.append("交易频率检查通过（警告）")
        else:
            checks.append("交易频率检查通过")

        is_warning_only = len(warnings) > 0 and len(reasons) == 0

        return {
            'passed': len(reasons) == 0,
            'warning': is_warning_only,
            'checks': checks,
            'reasons': reasons,
            'warnings': warnings
        }

    def _check_order_size(self, trade_request: TradeRequest, account_info: AccountInfo) -> Dict[str, Any]:
        """订单规模检查"""
        checks = []
        reasons = []
        warnings = []

        order_value = trade_request.quantity * trade_request.price

        # 最小订单金额检查
        if order_value < self.config.min_order_value:
            reasons.append(f"订单金额过小{order_value:.2f}元，最小要求{self.config.min_order_value:.2f}元")
        else:
            checks.append("最小订单金额检查通过")

        # 最大订单金额检查
        if order_value > self.config.max_single_order_value:
            if self.risk_level == RiskControlLevel.STRICT:
                reasons.append(f"订单金额过大{order_value:.2f}元，最大允许{self.config.max_single_order_value:.2f}元")
            else:
                warnings.append(f"订单金额较大{order_value:.2f}元，请确认交易意图")
                checks.append("最大订单金额检查通过（警告）")
        else:
            checks.append("最大订单金额检查通过")

        # 订单占总资产比例检查
        order_ratio = order_value / account_info.total_assets if account_info.total_assets > 0 else 0
        if order_ratio > 0.5:  # 单笔订单超过总资产50%
            warnings.append(f"单笔订单占总资产比例过高{order_ratio:.2%}")
            checks.append("订单规模比例检查通过（警告）")
        else:
            checks.append("订单规模比例检查通过")

        is_warning_only = len(warnings) > 0 and len(reasons) == 0

        return {
            'passed': len(reasons) == 0,
            'warning': is_warning_only,
            'checks': checks,
            'reasons': reasons,
            'warnings': warnings
        }

    def _calculate_overall_risk_score(self, trade_request: TradeRequest, account_info: AccountInfo,
                                    passed_checks: List[str], failed_checks: List[str],
                                    warning_messages: List[str]) -> float:
        """计算整体风险评分"""
        try:
            # 基础风险评分（基于失败检查的数量）
            base_risk = len(failed_checks) * 20  # 每个失败检查增加20分风险

            # 警告风险评分
            warning_risk = len(warning_messages) * 5  # 每个警告增加5分风险

            # 订单规模风险
            order_value = trade_request.quantity * trade_request.price
            order_ratio = order_value / account_info.total_assets if account_info.total_assets > 0 else 0
            size_risk = min(order_ratio * 50, 30)  # 订单规模风险最多30分

            # 流动性风险（简化）
            liquidity_risk = 5 if trade_request.quantity > 10000 else 0  # 大额交易增加流动性风险

            # 市场时间风险
            current_time = datetime.now().time()
            time_risk = 10 if current_time < dt_time(10, 0) or current_time > dt_time(14, 30) else 0

            # 综合风险评分
            total_risk = base_risk + warning_risk + size_risk + liquidity_risk + time_risk

            # 确保风险评分在0-100范围内
            total_risk = max(0, min(100, total_risk))

            return self.stability_manager.ensure_series_precision(pd.Series([total_risk])).iloc[0]

        except Exception as e:
            logger.error(f"计算风险评分失败: {e}")
            return 50.0  # 默认中等风险

    def _get_position_quantity(self, account_info: AccountInfo, stock_code: str) -> int:
        """获取持仓数量"""
        for position in account_info.positions:
            if position.get('stock_code') == stock_code:
                return position.get('quantity', 0)
        return 0

    def _get_position_value(self, account_info: AccountInfo, stock_code: str) -> float:
        """获取持仓市值"""
        for position in account_info.positions:
            if position.get('stock_code') == stock_code:
                return position.get('market_value', 0.0)
        return 0.0

    def _get_stock_sector(self, stock_code: str) -> Optional[str]:
        """获取股票行业（简化实现）"""
        # 实际实现中应该查询数据库获取行业信息
        sector_map = {
            '000001': '银行业',
            '000002': '房地产',
            '600036': '银行业',
            '600519': '食品饮料'
        }
        return sector_map.get(stock_code)

    def _get_sector_value(self, account_info: AccountInfo, sector: str) -> float:
        """获取行业持仓总市值"""
        total_value = 0.0
        for position in account_info.positions:
            stock_code = position.get('stock_code')
            if self._get_stock_sector(stock_code) == sector:
                total_value += position.get('market_value', 0.0)
        return total_value

    def _get_current_market_price(self, stock_code: str) -> Optional[float]:
        """获取当前市场价格（模拟实现）"""
        try:
            # 实际实现中应该调用实时行情接口
            # 这里使用模拟价格
            import random
            base_price = float(stock_code) / 100000  # 基于股票代码生成基础价格
            current_price = base_price * (1 + random.uniform(-0.1, 0.1))  # 加入10%随机波动
            return round(current_price, 2)
        except:
            return None

    def _generate_account_summary(self, account_info: AccountInfo) -> Dict[str, Any]:
        """生成账户摘要"""
        return {
            'total_assets': account_info.total_assets,
            'available_cash': account_info.available_cash,
            'market_value': account_info.market_value,
            'cash_ratio': account_info.available_cash / account_info.total_assets if account_info.total_assets > 0 else 0,
            'position_ratio': account_info.market_value / account_info.total_assets if account_info.total_assets > 0 else 0,
            'leverage_ratio': account_info.total_liability / (account_info.total_assets - account_info.total_liability) if (account_info.total_assets - account_info.total_liability) > 0 else 0,
            'position_count': len(account_info.positions)
        }

    def _calculate_trade_impact(self, trade_request: TradeRequest, account_info: AccountInfo) -> Dict[str, Any]:
        """计算交易影响"""
        order_value = trade_request.quantity * trade_request.price

        if trade_request.direction == TradeDirection.BUY:
            new_total_assets = account_info.total_assets
            new_cash = account_info.available_cash - order_value
            new_market_value = account_info.market_value + order_value
        else:  # 卖出
            new_total_assets = account_info.total_assets
            new_cash = account_info.available_cash + order_value
            new_market_value = account_info.market_value - order_value

        return {
            'order_value': order_value,
            'impact_on_cash': new_cash - account_info.available_cash,
            'impact_on_market_value': new_market_value - account_info.market_value,
            'new_cash_ratio': new_cash / new_total_assets if new_total_assets > 0 else 0,
            'new_position_ratio': new_market_value / new_total_assets if new_total_assets > 0 else 0
        }

    def _log_check_result(self, result: PreTradeCheckResult):
        """记录检查结果"""
        log_level = 'INFO'
        if result.result == PreTradeCheckResult.REJECTED:
            log_level = 'WARNING'
        elif result.result == PreTradeCheckResult.WARNING:
            log_level = 'INFO'

        logger.log(
            getattr(logging, log_level),
            f"事前风控检查完成 - 股票: {result.trade_request.stock_code}, "
            f"结果: {result.result.value}, 风险评分: {result.risk_score}, "
            f"耗时: {result.check_duration:.3f}s"
        )

    # 配置管理方法
    def update_config(self, new_config: PreTradeCheckConfig):
        """更新风控配置"""
        with self._lock:
            self.config = new_config
            logger.info("事前风控配置已更新")

    def add_to_blacklist(self, stock_codes: Union[str, List[str]]):
        """添加股票到黑名单"""
        with self._lock:
            if isinstance(stock_codes, str):
                stock_codes = [stock_codes]

            for stock_code in stock_codes:
                self.blacklist.add(stock_code)

            logger.info(f"已添加{len(stock_codes)}只股票到黑名单")

    def remove_from_blacklist(self, stock_codes: Union[str, List[str]]):
        """从黑名单移除股票"""
        with self._lock:
            if isinstance(stock_codes, str):
                stock_codes = [stock_codes]

            for stock_code in stock_codes:
                self.blacklist.discard(stock_code)

            logger.info(f"已从黑名单移除{len(stock_codes)}只股票")

    def set_whitelist(self, stock_codes: List[str]):
        """设置白名单"""
        with self._lock:
            self.whitelist = set(stock_codes)
            logger.info(f"已设置白名单，包含{len(stock_codes)}只股票")

    def clear_whitelist(self):
        """清除白名单"""
        with self._lock:
            self.whitelist.clear()
            logger.info("已清除白名单")

    def set_sector_limit(self, sector: str, max_ratio: float):
        """设置行业持仓限制"""
        with self._lock:
            self.sector_limits[sector] = max_ratio
            logger.info(f"已设置行业{sector}最大持仓比例为{max_ratio:.2%}")

    def get_risk_control_status(self) -> Dict[str, Any]:
        """获取风控系统状态"""
        return {
            'risk_level': self.risk_level.value,
            'blacklist_count': len(self.blacklist),
            'whitelist_count': len(self.whitelist),
            'sector_limits_count': len(self.sector_limits),
            'config': asdict(self.config),
            'system_status': 'running',
            'last_update': datetime.now().isoformat()
        }


# 创建全局实例
_pre_trade_controller = None

def get_pre_trade_controller(config: PreTradeCheckConfig = None,
                           risk_level: RiskControlLevel = RiskControlLevel.MODERATE) -> PreTradeRiskController:
    """获取事前风控控制器实例"""
    global _pre_trade_controller

    if _pre_trade_controller is None:
        _pre_trade_controller = PreTradeRiskController(config, risk_level)

    return _pre_trade_controller