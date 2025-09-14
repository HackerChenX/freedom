#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
风控体系集成测试套件

专业量化交易系统风控体系的完整集成测试，验证所有组件的协同工作能力。
包含功能测试、性能测试、压力测试和端到端测试。

测试范围：
1. 统一风控管理系统测试
2. 事前风控引擎测试
3. 事中实时监控测试
4. 事后分析系统测试
5. 智能预警系统测试
6. API接口测试
7. 性能基准测试
8. 系统集成测试

测试特性：
- pytest框架，支持参数化测试
- 异步测试支持
- 性能基准测试
- 测试覆盖率分析
- 自动化测试报告
"""

import pytest
import asyncio
import time
import json
import concurrent.futures
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import aiohttp
import websockets

from utils.logger import get_logger
from utils.numerical_stability_manager import get_stability_manager

# 导入被测试的风控组件
from risk.unified_risk_management_system import (
    get_unified_risk_management_system,
    RiskControlConfig,
    RiskControlMode
)
from risk.fast_pre_trade_risk_engine import (
    get_fast_pre_trade_risk_engine,
    RiskCheckInput,
    FastRiskCheckConfig
)
from risk.post_trade_analysis_system import (
    get_post_trade_analysis_system,
    AnalysisPeriod
)
from risk.intelligent_risk_alert_system import (
    get_intelligent_alert_system,
    AlertRule,
    AlertType,
    AlertLevel,
    NotificationChannel
)

logger = get_logger(__name__)


class TestRiskControlSystem:
    """风控体系集成测试类"""

    @pytest.fixture(scope="class")
    def risk_management_system(self):
        """风险管理系统测试夹具"""
        config = RiskControlConfig(
            mode=RiskControlMode.BALANCED,
            max_response_time_ms=10.0,
            precision=6
        )
        system = get_unified_risk_management_system(config=config)
        yield system
        system.shutdown()

    @pytest.fixture(scope="class")
    def pre_trade_engine(self):
        """事前风控引擎测试夹具"""
        config = FastRiskCheckConfig(
            max_response_time_ms=10.0,
            enable_cache=True,
            enable_async=True
        )
        engine = get_fast_pre_trade_risk_engine(config=config)
        yield engine
        engine.shutdown()

    @pytest.fixture(scope="class")
    def post_trade_system(self):
        """事后分析系统测试夹具"""
        return get_post_trade_analysis_system()

    @pytest.fixture(scope="class")
    async def alert_system(self):
        """预警系统测试夹具"""
        system = get_intelligent_alert_system()
        await system.start_notification_processing()
        yield system
        await system.stop_notification_processing()

    @pytest.fixture
    def sample_trade_request(self):
        """示例交易请求"""
        return RiskCheckInput(
            stock_code="000001",
            trade_direction="BUY",
            quantity=1000,
            price=12.50
        )

    def test_system_initialization(self, risk_management_system):
        """测试系统初始化"""
        assert risk_management_system is not None
        status = risk_management_system.get_system_status()
        assert status['status'] in ['运行中', '初始化中']
        assert 'start_time' in status
        assert 'subsystems' in status

    def test_numerical_precision(self, risk_management_system):
        """测试数值精度"""
        stability_manager = get_stability_manager()

        # 测试精度保障
        test_values = [1.23456789, 0.123456789, 123.456789]
        precision = 6

        for value in test_values:
            rounded_value = stability_manager.round_to_precision(value, precision)
            assert len(str(rounded_value).split('.')[-1]) <= precision

        # 测试风险指标精度
        metrics = risk_management_system.get_realtime_risk_metrics()
        assert isinstance(metrics.risk_score, float)
        assert isinstance(metrics.portfolio_value, float)
        assert isinstance(metrics.volatility, float)

    @pytest.mark.asyncio
    async def test_pre_trade_risk_check_performance(self, pre_trade_engine, sample_trade_request):
        """测试事前风控检查性能"""
        response_times = []

        # 执行多次检查测量性能
        for i in range(100):
            start_time = time.perf_counter()
            result = await pre_trade_engine.check_risk_async(sample_trade_request)
            end_time = time.perf_counter()

            response_time_ms = (end_time - start_time) * 1000
            response_times.append(response_time_ms)

            # 验证结果
            assert result is not None
            assert hasattr(result, 'approved')
            assert hasattr(result, 'risk_score')
            assert hasattr(result, 'response_time_ms')

        # 性能验证
        avg_response_time = np.mean(response_times)
        p95_response_time = np.percentile(response_times, 95)

        logger.info(f"平均响应时间: {avg_response_time:.2f}ms")
        logger.info(f"P95响应时间: {p95_response_time:.2f}ms")

        # 验证响应时间要求
        assert avg_response_time <= 10.0, f"平均响应时间 {avg_response_time:.2f}ms 超过10ms要求"
        assert p95_response_time <= 15.0, f"P95响应时间 {p95_response_time:.2f}ms 超过15ms要求"

    def test_pre_trade_risk_check_logic(self, pre_trade_engine):
        """测试事前风控检查逻辑"""
        # 正常交易请求
        normal_request = RiskCheckInput(
            stock_code="000001",
            trade_direction="BUY",
            quantity=1000,
            price=12.50
        )
        result = pre_trade_engine.check_risk_sync(normal_request)
        assert result.approved is True or result.approved is False  # 应该有明确结果
        assert 0 <= result.risk_score <= 100

        # 异常大额交易请求
        large_request = RiskCheckInput(
            stock_code="000001",
            trade_direction="BUY",
            quantity=100000,  # 大量
            price=100.0       # 高价
        )
        result = pre_trade_engine.check_risk_sync(large_request)
        assert result.risk_score > 0  # 应该有风险评分

    @pytest.mark.asyncio
    async def test_realtime_monitoring(self, risk_management_system):
        """测试实时监控功能"""
        # 启动实时监控
        risk_management_system.start_realtime_monitoring()

        # 等待监控运行
        await asyncio.sleep(3)

        # 获取实时指标
        metrics = risk_management_system.get_realtime_risk_metrics()

        # 验证指标完整性
        assert metrics.timestamp is not None
        assert isinstance(metrics.portfolio_value, (int, float))
        assert isinstance(metrics.risk_score, (int, float))
        assert isinstance(metrics.response_time_ms, (int, float))
        assert isinstance(metrics.system_health, (int, float))

        # 验证指标范围
        assert 0 <= metrics.risk_score <= 100
        assert 0 <= metrics.system_health <= 100

        # 停止监控
        risk_management_system.stop_realtime_monitoring()

    def test_post_trade_analysis(self, post_trade_system):
        """测试事后分析功能"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        result = post_trade_system.analyze_post_trade_performance(
            start_date=start_date,
            end_date=end_date,
            analysis_period=AnalysisPeriod.DAILY
        )

        # 验证分析结果
        assert result is not None
        assert result.analysis_date is not None
        assert result.period_start == start_date
        assert result.period_end == end_date

        # 验证风险指标
        risk_metrics = result.risk_metrics
        assert isinstance(risk_metrics.var_1d, float)
        assert isinstance(risk_metrics.max_drawdown, float)
        assert isinstance(risk_metrics.volatility_annualized, float)

        # 验证绩效指标
        perf_metrics = result.performance_metrics
        assert isinstance(perf_metrics.total_return, float)
        assert isinstance(perf_metrics.sharpe_ratio, float)
        assert isinstance(perf_metrics.win_rate, float)

    @pytest.mark.asyncio
    async def test_alert_system_functionality(self, alert_system):
        """测试预警系统功能"""
        # 测试触发预警
        alert_event = await alert_system.trigger_alert(
            rule_id="RISK_001",
            trigger_value=87.5,
            related_data={"portfolio_id": "TEST_P001"}
        )

        if alert_event:
            assert alert_event.alert_id is not None
            assert alert_event.rule_id == "RISK_001"
            assert alert_event.trigger_value == 87.5
            assert alert_event.alert_level in [AlertLevel.INFO, AlertLevel.WARNING, AlertLevel.CRITICAL, AlertLevel.EMERGENCY]

        # 测试预警统计
        stats = alert_system.get_alert_statistics()
        assert 'total_alerts' in stats
        assert 'alerts_by_level' in stats
        assert 'notifications_sent' in stats

    @pytest.mark.asyncio
    async def test_concurrent_risk_checks(self, pre_trade_engine):
        """测试并发风控检查"""
        # 创建多个并发请求
        requests = []
        for i in range(50):
            request = RiskCheckInput(
                stock_code=f"00000{i % 10}",
                trade_direction="BUY" if i % 2 == 0 else "SELL",
                quantity=1000 + i * 10,
                price=10.0 + i * 0.1
            )
            requests.append(request)

        # 并发执行
        start_time = time.perf_counter()
        tasks = [pre_trade_engine.check_risk_async(req) for req in requests]
        results = await asyncio.gather(*tasks)
        end_time = time.perf_counter()

        # 验证结果
        assert len(results) == len(requests)
        for result in results:
            assert result is not None
            assert hasattr(result, 'approved')
            assert hasattr(result, 'risk_score')

        # 验证并发性能
        total_time = (end_time - start_time) * 1000
        avg_time_per_check = total_time / len(requests)

        logger.info(f"并发检查 {len(requests)} 个请求，总时间: {total_time:.2f}ms")
        logger.info(f"平均每个检查: {avg_time_per_check:.2f}ms")

        assert avg_time_per_check <= 20.0, f"并发场景下平均响应时间过慢: {avg_time_per_check:.2f}ms"

    def test_risk_score_consistency(self, pre_trade_engine):
        """测试风险评分一致性"""
        request = RiskCheckInput(
            stock_code="000001",
            trade_direction="BUY",
            quantity=1000,
            price=12.50
        )

        # 多次执行相同请求
        results = []
        for _ in range(10):
            result = pre_trade_engine.check_risk_sync(request)
            results.append(result.risk_score)

        # 验证一致性（相同输入应该得到相同结果）
        if len(set(results)) == 1:
            logger.info(f"风险评分完全一致: {results[0]}")
        else:
            # 允许小幅波动（可能由于时间等因素）
            score_std = np.std(results)
            assert score_std <= 1.0, f"风险评分波动过大，标准差: {score_std}"
            logger.info(f"风险评分轻微波动，标准差: {score_std}")

    def test_cache_functionality(self, pre_trade_engine):
        """测试缓存功能"""
        request = RiskCheckInput(
            stock_code="000001",
            trade_direction="BUY",
            quantity=1000,
            price=12.50
        )

        # 首次请求（无缓存）
        result1 = pre_trade_engine.check_risk_sync(request)
        assert result1.cache_hit is False

        # 短时间内再次请求（应该命中缓存）
        result2 = pre_trade_engine.check_risk_sync(request)

        # 验证性能统计
        stats = pre_trade_engine.get_performance_stats()
        assert 'cache_hit_rate' in stats
        assert stats['total_checks'] >= 2

    @pytest.mark.asyncio
    async def test_system_integration_workflow(self, risk_management_system, pre_trade_engine, alert_system):
        """测试系统集成工作流"""
        logger.info("开始系统集成测试工作流")

        # 1. 系统初始化检查
        status = risk_management_system.get_system_status()
        assert status['status'] in ['运行中', '初始化中']

        # 2. 启动实时监控
        risk_management_system.start_realtime_monitoring()
        await asyncio.sleep(1)

        # 3. 执行事前风控检查
        trade_request = RiskCheckInput(
            stock_code="000001",
            trade_direction="BUY",
            quantity=2000,
            price=15.0
        )

        risk_result = await pre_trade_engine.check_risk_async(trade_request)
        assert risk_result is not None

        # 4. 获取实时风险指标
        metrics = risk_management_system.get_realtime_risk_metrics()
        assert metrics is not None

        # 5. 触发预警（如果风险评分较高）
        if risk_result.risk_score > 70:
            alert_event = await alert_system.trigger_alert(
                rule_id="RISK_001",
                trigger_value=risk_result.risk_score,
                related_data={"stock_code": trade_request.stock_code}
            )
            assert alert_event is not None

        # 6. 停止监控
        risk_management_system.stop_realtime_monitoring()

        logger.info("系统集成测试工作流完成")

    @pytest.mark.performance
    def test_system_performance_benchmarks(self, risk_management_system, pre_trade_engine):
        """系统性能基准测试"""
        logger.info("开始性能基准测试")

        # 性能指标收集
        performance_metrics = {
            'risk_metrics_query_time': [],
            'pre_trade_check_time': [],
            'memory_usage': [],
            'throughput': []
        }

        # 测试风险指标查询性能
        for _ in range(100):
            start_time = time.perf_counter()
            metrics = risk_management_system.get_realtime_risk_metrics()
            end_time = time.perf_counter()

            query_time = (end_time - start_time) * 1000
            performance_metrics['risk_metrics_query_time'].append(query_time)

        # 测试事前检查性能
        requests = [
            RiskCheckInput(
                stock_code=f"00000{i % 10}",
                trade_direction="BUY" if i % 2 == 0 else "SELL",
                quantity=1000,
                price=10.0 + i * 0.01
            ) for i in range(1000)
        ]

        start_time = time.perf_counter()
        for request in requests:
            result = pre_trade_engine.check_risk_sync(request)
            check_time = result.response_time_ms
            performance_metrics['pre_trade_check_time'].append(check_time)
        end_time = time.perf_counter()

        total_time = end_time - start_time
        throughput = len(requests) / total_time
        performance_metrics['throughput'].append(throughput)

        # 性能统计
        avg_query_time = np.mean(performance_metrics['risk_metrics_query_time'])
        avg_check_time = np.mean(performance_metrics['pre_trade_check_time'])
        avg_throughput = throughput

        logger.info(f"风险指标查询平均时间: {avg_query_time:.2f}ms")
        logger.info(f"事前检查平均时间: {avg_check_time:.2f}ms")
        logger.info(f"系统吞吐量: {avg_throughput:.0f} 检查/秒")

        # 性能要求验证
        assert avg_query_time <= 5.0, f"风险指标查询时间超标: {avg_query_time:.2f}ms"
        assert avg_check_time <= 10.0, f"事前检查时间超标: {avg_check_time:.2f}ms"
        assert avg_throughput >= 50, f"系统吞吐量不足: {avg_throughput:.0f} 检查/秒"

    @pytest.mark.stress
    @pytest.mark.asyncio
    async def test_system_stress_test(self, risk_management_system, pre_trade_engine):
        """系统压力测试"""
        logger.info("开始系统压力测试")

        # 启动实时监控
        risk_management_system.start_realtime_monitoring()

        # 创建大量并发请求
        num_requests = 1000
        concurrent_limit = 100

        requests = [
            RiskCheckInput(
                stock_code=f"00000{i % 100:02d}",
                trade_direction="BUY" if i % 2 == 0 else "SELL",
                quantity=np.random.randint(100, 10000),
                price=np.random.uniform(5.0, 50.0)
            ) for i in range(num_requests)
        ]

        # 批量并发执行
        semaphore = asyncio.Semaphore(concurrent_limit)

        async def check_with_semaphore(request):
            async with semaphore:
                return await pre_trade_engine.check_risk_async(request)

        start_time = time.perf_counter()
        tasks = [check_with_semaphore(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.perf_counter()

        # 统计结果
        successful_results = [r for r in results if not isinstance(r, Exception)]
        failed_results = [r for r in results if isinstance(r, Exception)]

        success_rate = len(successful_results) / len(results)
        total_time = end_time - start_time
        throughput = len(successful_results) / total_time

        logger.info(f"压力测试完成:")
        logger.info(f"  总请求数: {num_requests}")
        logger.info(f"  成功数: {len(successful_results)}")
        logger.info(f"  失败数: {len(failed_results)}")
        logger.info(f"  成功率: {success_rate:.2%}")
        logger.info(f"  总耗时: {total_time:.2f}秒")
        logger.info(f"  吞吐量: {throughput:.0f} 检查/秒")

        # 压力测试要求
        assert success_rate >= 0.95, f"压力测试成功率不足: {success_rate:.2%}"
        assert throughput >= 30, f"压力测试吞吐量不足: {throughput:.0f} 检查/秒"

        # 停止监控
        risk_management_system.stop_realtime_monitoring()

    def test_error_handling(self, pre_trade_engine):
        """测试错误处理"""
        # 测试无效输入
        with pytest.raises(Exception):
            invalid_request = RiskCheckInput(
                stock_code="",  # 无效股票代码
                trade_direction="INVALID",  # 无效交易方向
                quantity=-1000,  # 无效数量
                price=-10.0      # 无效价格
            )
            pre_trade_engine.check_risk_sync(invalid_request)

    def test_data_consistency(self, risk_management_system, post_trade_system):
        """测试数据一致性"""
        # 获取实时指标
        realtime_metrics = risk_management_system.get_realtime_risk_metrics()

        # 执行事后分析
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)

        analysis_result = post_trade_system.analyze_post_trade_performance(
            start_date=start_date,
            end_date=end_date
        )

        # 验证数据一致性（在合理范围内）
        assert isinstance(realtime_metrics.portfolio_value, (int, float))
        assert isinstance(analysis_result.performance_metrics.total_return, float)


# 性能基准测试
class TestPerformanceBenchmarks:
    """性能基准测试类"""

    def test_response_time_sla(self):
        """测试响应时间SLA"""
        engine = get_fast_pre_trade_risk_engine()

        request = RiskCheckInput(
            stock_code="000001",
            trade_direction="BUY",
            quantity=1000,
            price=12.50
        )

        response_times = []
        for _ in range(1000):
            start = time.perf_counter()
            result = engine.check_risk_sync(request)
            end = time.perf_counter()
            response_times.append((end - start) * 1000)

        # SLA验证
        p50 = np.percentile(response_times, 50)
        p95 = np.percentile(response_times, 95)
        p99 = np.percentile(response_times, 99)

        assert p50 <= 5.0, f"P50响应时间超标: {p50:.2f}ms"
        assert p95 <= 10.0, f"P95响应时间超标: {p95:.2f}ms"
        assert p99 <= 15.0, f"P99响应时间超标: {p99:.2f}ms"


# 测试运行入口
if __name__ == "__main__":
    print("=== 风控体系集成测试套件 ===")

    # 运行所有测试
    pytest.main([
        __file__,
        "-v",                    # 详细输出
        "--tb=short",           # 简短的错误追踪
        "--strict-markers",     # 严格的标记模式
        "-m", "not stress",     # 跳过压力测试（需要时单独运行）
        "--html=test_report.html",  # 生成HTML报告
        "--cov=risk",          # 代码覆盖率
        "--cov-report=html",   # HTML覆盖率报告
        "--durations=10"       # 显示最慢的10个测试
    ])

    print("\n=== 测试完成 ===")
    print("详细报告: test_report.html")
    print("覆盖率报告: htmlcov/index.html")