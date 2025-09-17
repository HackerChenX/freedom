#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合选股测试系统 - 主控制器

测试所有技术指标的所有形态，通过闭环验证确保选股准确性
要求：5分钟内完成4000+股票的全面测试
"""

import asyncio
import time
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import gc

from utils.logger import getLogger
from utils.dependency_injection import get_service
from db.interfaces.data_access_interface import DataAccessInterface
from analysis.buypoints.analyze_buypoints import BuyPointAnalyzer
from indicators.pattern_registry import get_pattern_registry, PatternRegistry
from .indicator_discovery import IndicatorDiscovery
from .pattern_registry_manager import PatternRegistryManager
from .stock_selection_engine import StockSelectionEngine, StockSelectionCriteria
from .buypoint_verification_engine import BuypointVerificationEngine, BatchVerificationResult
from .batch_verification_processor import BatchVerificationProcessor, BatchProcessingConfig
from .test_result_models import (
    TestResults, IndicatorTestResult, PatternTestResult, 
    StockSelection, VerificationResult, TestResultSummary
)
from .enhanced_report_generator import EnhancedReportGenerator
from .performance_monitor import PerformanceMonitor, PerformanceOptimizationService, PerformanceConfig
from .stock_selection_reporter import StockSelectionReporter
from .cache_manager import get_cache_manager
from .query_optimizer import ClickHouseOptimizer
from .config_manager import get_config_manager, TestConfig
from .error_handler import TestErrorHandler, ErrorCategory, with_error_handling, get_error_handler
from .system_integration import get_system_integrator
from db.sql_manager import SQLManager, QueryType

logger = getLogger(__name__)


@dataclass
class DateRange:
    """日期范围"""
    start_date: str
    end_date: str


@dataclass
class TestConfig:
    """测试配置参数"""
    date_range: DateRange
    stock_universe: List[str] = field(default_factory=list)
    indicators_to_test: Optional[List[str]] = None
    patterns_to_test: Optional[List[str]] = None
    verification_threshold: float = 0.7
    batch_size: int = 1000
    parallel_workers: int = 20
    output_format: str = "json"
    report_level: str = "comprehensive"
    timeout_seconds: int = 300  # 5分钟
    performance_threshold: float = 0.8


# 使用新的性能监控系统替代旧的TestMonitor和PerformanceManager


class ComprehensiveStockSelectionTester:
    """综合选股测试系统主控制器"""
    
    def __init__(self, config_path: str = "tests/comprehensive/test_config.yaml"):
        """初始化测试系统"""
        logger.info("初始化综合选股测试系统...")
        
        # 加载配置
        self.config_manager = get_config_manager(config_path)
        self.config = self.config_manager.get_config()
        
        # 验证配置
        config_errors = self.config_manager.validate_config()
        if config_errors:
            logger.warning("配置验证发现以下问题:")
            for error in config_errors:
                logger.warning(f"  - {error}")
        
        # 初始化组件
        self.data_access = get_service(DataAccessInterface)
        self.pattern_registry = get_pattern_registry()
        self.buypoint_analyzer = BuyPointAnalyzer(self.data_access)
        
        # 初始化新组件
        self.indicator_discovery = IndicatorDiscovery()
        self.pattern_manager = PatternRegistryManager(self.indicator_discovery)

        # 验证重构后的系统组件
        self._validate_refactored_components()
        self.selection_engine = StockSelectionEngine(
            self.data_access, self.indicator_discovery, self.pattern_manager
        )
        self.verification_engine = BuypointVerificationEngine(self.buypoint_analyzer)
        
        # 初始化批量验证处理器
        verification_config = BatchProcessingConfig(
            max_workers=getattr(self.config, 'parallel_workers', 4),
            batch_size=getattr(self.config, 'batch_size', 100),
            timeout_seconds=getattr(self.config, 'timeout_seconds', 300),
            retry_count=2,
            enable_diagnostics=True
        )
        self.verification_processor = BatchVerificationProcessor(
            self.verification_engine, verification_config
        )
        
        # 初始化报告生成器
        self.report_generator = EnhancedReportGenerator()
        
        # 初始化性能监控系统
        performance_config = PerformanceConfig(
            timeout_seconds=getattr(self.config, 'timeout_seconds', 300),
            performance_threshold=getattr(self.config, 'performance_threshold', 0.8),
            memory_threshold_gb=6.0,
            cpu_threshold_percent=80.0,
            enable_early_stopping=True,
            enable_performance_optimization=True
        )
        self.performance_monitor = PerformanceMonitor(performance_config)
        self.performance_optimizer = PerformanceOptimizationService(self.performance_monitor)
        
        # 注册早停回调
        self.performance_monitor.register_early_stop_callback(self._handle_early_stop)
        self.performance_monitor.register_optimization_callback(self._handle_performance_optimization)
        
        # 测试结果
        self.test_results = None
        
        # 测试结果
        self.test_results = None
        
        logger.info("综合选股测试系统初始化完成")

    def _validate_refactored_components(self):
        """验证重构后的系统组件是否正常工作"""
        logger.info("验证重构后的系统组件...")

        validation_results = {}

        try:
            # 验证买点分析器
            test_result = self.buypoint_analyzer.analyze_stock("000001", "20240101", "测试")
            validation_results['buypoint_analyzer'] = test_result is not None

            # 验证形态注册表
            patterns = self.pattern_registry.get_all_patterns()
            validation_results['pattern_registry'] = len(patterns) > 0
            logger.info(f"发现 {len(patterns)} 个已注册形态")

            # 验证数据访问
            validation_results['data_access'] = self.data_access is not None

            # 验证指标发现
            indicators = self.indicator_discovery.discover_all_indicators()
            validation_results['indicator_discovery'] = len(indicators) > 0
            logger.info(f"发现 {len(indicators)} 个可用指标")

            # 记录验证结果
            for component, status in validation_results.items():
                status_text = "✓" if status else "✗"
                logger.info(f"  {component}: {status_text}")

            if not all(validation_results.values()):
                logger.warning("部分系统组件验证失败，可能影响测试结果")

        except Exception as e:
            logger.error(f"系统组件验证失败: {e}")

        return validation_results
        
    # 已删除_load_config方法，使用config_manager代替
    
    async def run_comprehensive_test(self) -> TestResults:
        """执行综合测试"""
        test_id = f"comprehensive_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()
        
        logger.info(f"开始执行综合选股测试 - ID: {test_id}")
        logger.info(f"测试配置: 超时{self.config.timeout_seconds}秒, 并行度{self.config.parallel_workers}, 批大小{self.config.batch_size}")
        
        # 初始化测试结果
        self.test_results = TestResults(
            test_id=test_id,
            start_time=start_time,
            end_time=start_time,  # 临时设置，完成时更新
            total_indicators_tested=0,
            total_patterns_tested=0,
            total_stocks_selected=0,
            total_verifications_performed=0,
            overall_success_rate=0.0
        )
        
        try:
            # 第一阶段：发现和初始化指标
            logger.info("第一阶段：发现和初始化所有指标...")
            indicators = await self._discover_indicators()
            
            if not indicators:
                logger.error("未发现任何指标，测试终止")
                return self.test_results
            
            # 估算总任务数
            total_tasks = sum(len(self._get_indicator_patterns(ind)) for ind in indicators)
            self.performance_monitor.start_monitoring(total_tasks)
            
            logger.info(f"发现 {len(indicators)} 个指标，总计 {total_tasks} 个形态需要测试")
            
            # 第二阶段：并行测试所有指标
            logger.info("第二阶段：开始并行测试所有指标...")
            indicator_results = await self._test_all_indicators_parallel(indicators)
            
            # 检查是否需要早停
            if self.performance_monitor.check_timeout():
                logger.warning("检测到超时，触发早停机制")
                self.performance_monitor.trigger_early_stop()
                self.test_results.end_time = datetime.now()
                self.test_results.summary_statistics['early_stop'] = True
                self.test_results.summary_statistics['stop_reason'] = '超过5分钟执行时间限制'
                self.test_results.summary_statistics['early_stop'] = True
                self.test_results.summary_statistics['stop_reason'] = '超过5分钟执行时间限制'
                return self.test_results
            
            # 第三阶段：汇总结果
            logger.info("第三阶段：汇总测试结果...")
            self._aggregate_results(indicator_results)
            
            # 完成测试
            self.test_results.end_time = datetime.now()
            execution_time = (self.test_results.end_time - self.test_results.start_time).total_seconds()
            
            logger.info(f"综合测试完成！")
            logger.info(f"执行时间: {execution_time:.1f}秒")
            logger.info(f"测试指标: {self.test_results.total_indicators_tested}")
            logger.info(f"测试形态: {self.test_results.total_patterns_tested}")
            logger.info(f"选出股票: {self.test_results.total_stocks_selected}")
            logger.info(f"总体成功率: {self.test_results.overall_success_rate:.2%}")
            
            return self.test_results
            
        except Exception as e:
            logger.error(f"综合测试执行失败: {e}")
            self.test_results.end_time = datetime.now()
            self.test_results.summary_statistics['error'] = str(e)
            return self.test_results
    
    async def _discover_indicators(self) -> List[str]:
        """发现所有可用指标"""
        logger.info("开始发现系统中的所有指标...")
        
        # 使用指标发现系统
        indicator_infos = self.indicator_discovery.discover_all_indicators()
        
        # 过滤出可加载的指标
        loadable_indicators = [
            info.name for info in indicator_infos 
            if info.is_loadable and info.pattern_count > 0
        ]
        
        logger.info(f"发现 {len(indicator_infos)} 个指标，其中 {len(loadable_indicators)} 个可用于测试")
        
        # 如果没有发现指标，使用备用列表
        if not loadable_indicators:
            logger.warning("未发现可用指标，使用备用指标列表")
            loadable_indicators = [
                'MA', 'MACD', 'KDJ', 'RSI', 'BOLL', 'WR', 'CCI', 'ROC', 
                'MTM', 'BIAS', 'PSY', 'VR', 'SAR', 'EMV', 'WVAD'
            ]
        
        return loadable_indicators
    
    def _get_indicator_patterns(self, indicator_name: str) -> List[str]:
        """获取指标的所有形态"""
        # 确保指标形态已注册
        self.pattern_manager.ensure_patterns_registered(indicator_name)
        
        # 从形态注册表获取指标的形态
        patterns = self.pattern_registry.get_patterns_by_indicator(indicator_name)
        
        if not patterns:
            # 如果仍然没有形态，创建默认形态
            logger.warning(f"指标 {indicator_name} 没有注册形态，创建默认形态")
            patterns = [f"{indicator_name}_BULLISH", f"{indicator_name}_BEARISH", f"{indicator_name}_NEUTRAL"]
        
        return patterns
    
    async def _test_all_indicators_parallel(self, indicators: List[str]) -> Dict[str, IndicatorTestResult]:
        """并行测试所有指标"""
        results = {}
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            # 提交所有指标测试任务
            future_to_indicator = {
                executor.submit(self._test_single_indicator, indicator): indicator 
                for indicator in indicators
            }
            
            # 收集结果
            for future in as_completed(future_to_indicator):
                indicator = future_to_indicator[future]
                
                # 检查超时
                if self.performance_monitor.check_timeout():
                    logger.warning(f"超时检测：取消剩余指标测试")
                    break
                
                try:
                    result = future.result(timeout=30)  # 单个指标30秒超时
                    if result:
                        results[indicator] = result
                        self.performance_monitor.update_progress(1)
                        logger.info(f"指标 {indicator} 测试完成")
                    else:
                        logger.warning(f"指标 {indicator} 测试失败")
                        
                except Exception as e:
                    logger.error(f"指标 {indicator} 测试异常: {e}")
                    continue
        
        return results
    
    @with_error_handling(get_error_handler(), ErrorCategory.INDICATOR)
    def _test_single_indicator(self, indicator_name: str) -> Optional[IndicatorTestResult]:
        """测试单个指标"""
        logger.info(f"开始测试指标: {indicator_name}")
        
        # 获取指标的所有形态
        patterns = self._get_indicator_patterns(indicator_name)
        
        if not patterns:
            logger.warning(f"指标 {indicator_name} 没有可测试的形态")
            return None
        
        # 初始化指标测试结果
        indicator_result = IndicatorTestResult(
            indicator_name=indicator_name,
            total_patterns=len(patterns),
            patterns_tested=0,
            patterns_with_selections=0,
            total_stocks_selected=0,
            verification_success_rate=0.0
        )
        
        # 测试每个形态
        for pattern_id in patterns:
            if self.performance_monitor.check_timeout():
                logger.warning(f"超时检测：停止测试指标 {indicator_name} 的剩余形态")
                break
            
            try:
                pattern_result = self._test_single_pattern(pattern_id, indicator_name)
                if pattern_result:
                    indicator_result.pattern_results[pattern_id] = pattern_result
                    indicator_result.patterns_tested += 1
                    
                    if pattern_result.stocks_selected > 0:
                        indicator_result.patterns_with_selections += 1
                        indicator_result.total_stocks_selected += pattern_result.stocks_selected
            except Exception as e:
                # 使用错误处理器处理形态错误
                self.error_handler.handle_pattern_error(pattern_id, e)
                logger.error(f"测试形态 {pattern_id} 失败: {e}")
                continue
        
        # 计算成功率
        if indicator_result.patterns_tested > 0:
            indicator_result.verification_success_rate = (
                indicator_result.patterns_with_selections / indicator_result.patterns_tested
            )
        
        logger.info(f"指标 {indicator_name} 测试完成: {indicator_result.patterns_with_selections}/{indicator_result.patterns_tested} 形态成功")
        return indicator_result
    
    @with_error_handling(get_error_handler(), ErrorCategory.PATTERN)
    def _test_single_pattern(self, pattern_id: str, indicator_name: str) -> Optional[PatternTestResult]:
        """测试单个形态"""
        logger.debug(f"开始测试形态: {pattern_id} ({indicator_name})")
        
        # 使用选股引擎进行真实选股
        date_range = DateRange(
            start_date=self.config.date_range.start_date,
            end_date=self.config.date_range.end_date
        )
        
        # 创建选股条件
        criteria = StockSelectionCriteria(
            min_volume=self.config.stock_selection.min_volume,
            min_price=self.config.stock_selection.min_price,
            max_price=self.config.stock_selection.max_price,
            exclude_st=self.config.stock_selection.exclude_st,
            exclude_suspended=self.config.stock_selection.exclude_suspended
        )
        
        # 执行选股
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            selection_result = loop.run_until_complete(
                self.selection_engine.select_stocks_for_pattern(
                    pattern_id, indicator_name, date_range, criteria
                )
            )
        except Exception as e:
            # 使用错误处理器处理选股错误
            self.error_handler.handle_pattern_error(pattern_id, e)
            logger.error(f"形态 {pattern_id} 选股执行失败: {e}")
            raise
        finally:
            loop.close()
        
        if not selection_result.success:
            logger.warning(f"形态 {pattern_id} 选股失败: {selection_result.error_message}")
            return PatternTestResult(
                pattern_id=pattern_id,
                pattern_name=f"{indicator_name}_{pattern_id}",
                stocks_selected=0,
                verifications_attempted=0,
                verifications_successful=0,
                success_rate=0.0
            )
        
        # 转换选股结果格式
        selected_stocks = []
        for stock_sel in selection_result.selected_stocks:
            selected_stocks.append(StockSelection(
                stock_code=stock_sel.stock_code,
                stock_name=stock_sel.stock_name,
                date=stock_sel.date,
                pattern_id=stock_sel.pattern_id,
                confidence_score=stock_sel.confidence_score
            ))
        
        if not selected_stocks:
            logger.warning(f"形态 {pattern_id} 未选出任何股票")
            return PatternTestResult(
                pattern_id=pattern_id,
                pattern_name=f"{indicator_name}_{pattern_id}",
                stocks_selected=0,
                verifications_attempted=0,
                verifications_successful=0,
                success_rate=0.0
            )
        
        try:
            # 执行买点验证
            verification_results = self._perform_buypoint_verification(selected_stocks, pattern_id)
            
            successful_verifications = sum(1 for v in verification_results if v.pattern_match)
            success_rate = successful_verifications / len(verification_results) if verification_results else 0.0
            
            logger.debug(f"形态 {pattern_id} 测试完成: 选出{len(selected_stocks)}只股票, 验证成功率{success_rate:.2%}")
            
            return PatternTestResult(
                pattern_id=pattern_id,
                pattern_name=f"{indicator_name}_{pattern_id}",
                stocks_selected=len(selected_stocks),
                verifications_attempted=len(verification_results),
                verifications_successful=successful_verifications,
                success_rate=success_rate,
                selected_stocks=selected_stocks,
                verification_results=verification_results
            )
        except Exception as e:
            # 使用错误处理器处理验证错误
            if selected_stocks:
                self.error_handler.handle_verification_error(
                    selected_stocks[0].stock_code, selected_stocks[0].date, e
                )
            logger.error(f"形态 {pattern_id} 验证失败: {e}")
            raise
    
    def _simulate_stock_selection(self, pattern_id: str, indicator_name: str) -> List[StockSelection]:
        """模拟股票选择（占位符实现）"""
        # 这是临时的模拟实现，确保每个形态至少选出一只股票
        import random
        
        sample_stocks = [
            ("000001", "平安银行"), ("000002", "万科A"), ("000858", "五粮液"),
            ("600036", "招商银行"), ("600519", "贵州茅台"), ("000725", "京东方A")
        ]
        
        # 随机选择1-3只股票
        num_stocks = random.randint(1, 3)
        selected = random.sample(sample_stocks, min(num_stocks, len(sample_stocks)))
        
        results = []
        for code, name in selected:
            # 随机生成一个测试日期
            test_date = "20241201"  # 固定测试日期
            
            results.append(StockSelection(
                stock_code=code,
                stock_name=name,
                date=test_date,
                pattern_id=pattern_id,
                confidence_score=random.uniform(0.7, 0.95)
            ))
        
        return results
    
    def _simulate_verification(self, selected_stocks: List[StockSelection], pattern_id: str) -> List[VerificationResult]:
        """模拟验证过程（占位符实现）"""
        import random
        
        results = []
        for stock in selected_stocks:
            # 模拟买点分析检测到的形态
            detected_patterns = [pattern_id]  # 简化：假设检测到相同态
            
            # 随机决定是否匹配成功
            pattern_match = random.random() > 0.2  # 80%成功率
            
            results.append(VerificationResult(
                stock_code=stock.stock_code,
                date=stock.date,
                expected_pattern=pattern_id,
                detected_patterns=detected_patterns,
                pattern_match=pattern_match,
                confidence_score=random.uniform(0.6, 0.9)
            ))
        
        return results
    
    def _aggregate_results(self, indicator_results: Dict[str, IndicatorTestResult]):
        """汇总测试结果"""
        self.test_results.total_indicators_tested = len(indicator_results)
        self.test_results.indicator_results = indicator_results
        
        # 统计总数
        total_patterns = 0
        total_stocks = 0
        total_verifications = 0
        successful_verifications = 0
        
        for result in indicator_results.values():
            total_patterns += result.patterns_tested
            total_stocks += result.total_stocks_selected
            
            for pattern_result in result.pattern_results.values():
                total_verifications += pattern_result.verifications_attempted
                successful_verifications += pattern_result.verifications_successful
        
        self.test_results.total_patterns_tested = total_patterns
        self.test_results.total_stocks_selected = total_stocks
        self.test_results.total_verifications_performed = total_verifications
        
        # 计算总体成功率
        if total_verifications > 0:
            self.test_results.overall_success_rate = successful_verifications / total_verifications
        
        # 添加汇总统计
        self.test_results.summary_statistics = {
            'indicators_with_successful_patterns': sum(
                1 for r in indicator_results.values() if r.patterns_with_selections > 0
            ),
            'patterns_with_selections': sum(
                r.patterns_with_selections for r in indicator_results.values()
            ),
            'average_stocks_per_pattern': (
                total_stocks / total_patterns if total_patterns > 0 else 0
            ),
            'execution_time_seconds': (
                (self.test_results.end_time - self.test_results.start_time).total_seconds()
                if self.test_results.end_time > self.test_results.start_time else 0
            )
        }
    
    def _perform_buypoint_verification(self, selected_stocks: List[StockSelection], pattern_id: str) -> List[VerificationResult]:
        """执行买点验证 - 使用批量验证处理器"""
        try:
            # 使用批量验证处理器进行验证
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # 设置形态过滤器，只验证当前形态
                pattern_filter = {pattern_id}
                
                batch_result = loop.run_until_complete(
                    self.verification_processor.process_verification_batch(
                        selected_stocks, pattern_filter
                    )
                )
                verification_results = batch_result.verification_results
                
                logger.debug(f"批量验证完成: {batch_result.successful_verifications}/{batch_result.total_verifications} 成功 ({batch_result.success_rate:.1%})")
                
                # 获取验证统计信息
                stats = self.verification_processor.get_verification_statistics()
                logger.debug(f"验证统计: {stats}")
                
            finally:
                loop.close()
            
            return verification_results
            
        except Exception as e:
            logger.error(f"批量验证失败: {e}")
            # 返回空结果
            return [] 
    
    def _extract_patterns_from_buypoint_result(self, buypoint_result: Dict[str, Any], expected_pattern: str) -> List[str]:
        """从买点分析结果中提取形态"""
        detected_patterns = []
        
        try:
            # 根据买点分析结果判断检测到的形态
            pattern_upper = expected_pattern.upper()
            
            # 检查看涨形态
            if 'BULLISH' in pattern_upper or 'GOLDEN' in pattern_upper:
                if (buypoint_result.get('macd_gold', False) or 
                    buypoint_result.get('ma_up', False) or
                    buypoint_result.get('price_stable', False)):
                    detected_patterns.append(expected_pattern)
            
            # 检查均线相关形态
            elif 'MA' in pattern_upper:
                if buypoint_result.get('touch_ma', False) or buypoint_result.get('ma_up', False):
                    detected_patterns.append(expected_pattern)
            
            # 检查MACD相关形态
            elif 'MACD' in pattern_upper:
                if buypoint_result.get('macd_gold', False):
                    detected_patterns.append(expected_pattern)
            
            # 检查RSI相关形态
            elif 'RSI' in pattern_upper:
                if buypoint_result.get('rsi_oversold', False):
                    detected_patterns.append(expected_pattern)
            
            # 检查成交量相关形态
            elif 'VOL' in pattern_upper or 'VOLUME' in pattern_upper:
                if buypoint_result.get('money_in', False) or not buypoint_result.get('vol_shrink', True):
                    detected_patterns.append(expected_pattern)
            
            # 默认检查：如果有任何正面信号，认为检测到形态
            else:
                positive_signals = [
                    buypoint_result.get('touch_ma', False),
                    buypoint_result.get('price_stable', False),
                    buypoint_result.get('ma_up', False),
                    buypoint_result.get('money_in', False),
                    buypoint_result.get('macd_gold', False)
                ]
                
                if any(positive_signals):
                    detected_patterns.append(expected_pattern)
            
        except Exception as e:
            logger.debug(f"提取形态失败: {e}")
        
        return detected_patterns
    
    def _check_pattern_match(self, expected_pattern: str, detected_patterns: List[str], buypoint_result: Dict[str, Any]) -> bool:
        """检查形态是否匹配"""
        # 直接匹配
        if expected_pattern in detected_patterns:
            return True
        
        # 基于买点分析评分的匹配
        score = buypoint_result.get('score', 0)
        if score >= 50:  # 评分超过50分认为匹配
            return True
        
        return False
    
    def _calculate_verification_confidence(self, buypoint_result: Dict[str, Any], pattern_match: bool) -> float:
        """计算验证置信度"""
        if not pattern_match:
            return 0.0
        
        # 基于买点分析评分计算置信度
        score = buypoint_result.get('score', 0)
        confidence = min(score / 100.0, 1.0)  # 将评分转换为0-1的置信度
        
        return max(confidence, 0.1)  # 最低置信度0.1
        
    def generate_test_report(self, output_dir: str = "test_reports") -> Dict[str, str]:
        """
        生成测试报告
        
        Args:
            output_dir: 输出目录
            
        Returns:
            Dict[str, str]: 报告文件路径
        """
        if not self.test_results:
            logger.error("没有测试结果可供生成报告")
            return {}
        
        logger.info("开始生成测试报告...")
        
        try:
            # 使用增强版报告生成器
            formats = ["json", "csv", "html", "md"]
            report_files = self.report_generator.generate_reports(
                self.test_results, output_dir, formats
            )
            
            logger.info(f"测试报告生成完成，保存在: {output_dir}")
            return report_files
            
        except Exception as e:
            logger.error(f"生成测试报告失败: {e}")
            return {}
    
    def _perform_buypoint_verification(self, selected_stocks: List[StockSelection], pattern_id: str) -> List[VerificationResult]:
        """执行买点验证 - 使用BuypointVerificationEngine"""
        try:
            # 使用验证引擎进行批量验证
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                batch_result = loop.run_until_complete(
                    self.verification_engine.batch_verify_selections(selected_stocks)
                )
                verification_results = batch_result.verification_results
                
                logger.debug(f"批量验证完成: {batch_result.successful_verifications}/{batch_result.total_verifications} 成功 ({batch_result.success_rate:.1%})")
                
            finally:
                loop.close()
            
            return verification_results
            
        except Exception as e:
            logger.error(f"批量验证失败: {e}")
            # 返回空结果
            return []
    
    def _handle_early_stop(self) -> None:
        """处理早停回调"""
        logger.warning("执行早停处理：提前结束测试并返回部分结果")
        
        if self.test_results:
            self.test_results.end_time = datetime.now()
            self.test_results.summary_statistics['early_stop'] = True
            self.test_results.summary_statistics['stop_reason'] = '超过5分钟执行时间限制'
            self.test_results.summary_statistics['completion_percentage'] = (
                self.performance_monitor.completed_tasks / self.performance_monitor.total_tasks * 100
                if self.performance_monitor.total_tasks > 0 else 0
            )
    
    def _handle_performance_optimization(self) -> None:
        """处理性能优化回调"""
        logger.warning("执行性能优化：调整批处理大小和并行度")
        
        # 减少批处理大小
        self.config.batch_size = max(100, self.config.batch_size // 2)
        
        # 调整并行度
        if self.config.parallel_workers > 5:
            self.config.parallel_workers = max(5, self.config.parallel_workers // 2)
        
        # 更新验证处理器配置
        verification_config = BatchProcessingConfig(
            max_workers=getattr(self.config, 'parallel_workers', 4),
            batch_size=getattr(self.config, 'batch_size', 100),
            timeout_seconds=getattr(self.config, 'timeout_seconds', 300),
            retry_count=1,  # 减少重试次数
            enable_diagnostics=False  # 关闭诊断以提高性能
        )
        self.verification_processor = BatchVerificationProcessor(
            self.verification_engine, verification_config
        )
        
        # 强制垃圾回收
        gc.collect()
        
        logger.info(f"性能优化完成：批处理大小={self.config.batch_size}, 并行度={self.config.parallel_workers}")