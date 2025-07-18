#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量验证处理器

提供高性能的批量验证处理，支持大规模股票验证
包含详细的验证失败分析和诊断功能
"""

import asyncio
import time
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

from utils.logger import getLogger
from .buypoint_verification_engine import BuypointVerificationEngine, VerificationResult, BatchVerificationResult
from .stock_selection_engine import StockSelection

logger = getLogger(__name__)


@dataclass
class VerificationDiagnostics:
    """验证诊断信息"""
    pattern_id: str
    total_verifications: int
    successful_verifications: int
    failed_verifications: int
    success_rate: float
    common_failure_reasons: Dict[str, int] = field(default_factory=dict)
    pattern_match_statistics: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0


@dataclass
class BatchProcessingConfig:
    """批处理配置"""
    max_workers: int = 10
    batch_size: int = 100
    timeout_seconds: int = 300
    retry_count: int = 2
    use_process_pool: bool = False
    enable_diagnostics: bool = True


class BatchVerificationProcessor:
    """批量验证处理器 - 提供高性能的批量验证处理"""
    
    def __init__(self, 
                 verification_engine: BuypointVerificationEngine,
                 config: Optional[BatchProcessingConfig] = None):
        """
        初始化批量验证处理器
        
        Args:
            verification_engine: 验证引擎
            config: 批处理配置
        """
        self.verification_engine = verification_engine
        self.config = config or BatchProcessingConfig()
        
        # 诊断信息
        self.diagnostics_by_pattern = {}
        self.failed_verifications = []
        
        logger.info("批量验证处理器初始化完成")
    
    async def process_verification_batch(self, 
                                       selections: List[StockSelection],
                                       pattern_filter: Optional[Set[str]] = None) -> BatchVerificationResult:
        """
        处理验证批次
        
        Args:
            selections: 选股结果列表
            pattern_filter: 形态过滤器，只验证指定形态
            
        Returns:
            BatchVerificationResult: 批量验证结果
        """
        start_time = time.time()
        
        # 过滤选股结果
        if pattern_filter:
            filtered_selections = [
                s for s in selections 
                if s.pattern_id in pattern_filter
            ]
        else:
            filtered_selections = selections
        
        logger.info(f"开始批量验证 {len(filtered_selections)} 个选股结果...")
        
        # 按形态分组
        selections_by_pattern = {}
        for selection in filtered_selections:
            if selection.pattern_id not in selections_by_pattern:
                selections_by_pattern[selection.pattern_id] = []
            selections_by_pattern[selection.pattern_id].append(selection)
        
        # 初始化诊断信息
        self.diagnostics_by_pattern = {
            pattern_id: VerificationDiagnostics(
                pattern_id=pattern_id,
                total_verifications=len(pattern_selections),
                successful_verifications=0,
                failed_verifications=0,
                success_rate=0.0
            )
            for pattern_id, pattern_selections in selections_by_pattern.items()
        }
        
        # 分批处理
        all_verification_results = []
        
        # 使用线程池或进程池
        executor_class = ProcessPoolExecutor if self.config.use_process_pool else ThreadPoolExecutor
        
        with executor_class(max_workers=self.config.max_workers) as executor:
            # 提交所有验证任务
            futures = []
            
            for pattern_id, pattern_selections in selections_by_pattern.items():
                # 分批处理每个形态的选股结果
                for i in range(0, len(pattern_selections), self.config.batch_size):
                    batch = pattern_selections[i:i + self.config.batch_size]
                    
                    # 提交批次任务
                    future = executor.submit(
                        self._process_batch_with_retry,
                        batch,
                        pattern_id
                    )
                    futures.append((future, pattern_id))
            
            # 收集结果
            for future, pattern_id in futures:
                try:
                    batch_result = future.result(timeout=self.config.timeout_seconds)
                    all_verification_results.extend(batch_result.verification_results)
                    
                    # 更新诊断信息
                    self._update_diagnostics(pattern_id, batch_result)
                    
                except Exception as e:
                    logger.error(f"处理形态 {pattern_id} 的批次失败: {e}")
        
        # 计算总体结果
        successful_verifications = sum(1 for r in all_verification_results if r.pattern_match)
        failed_verifications = len(all_verification_results) - successful_verifications
        success_rate = successful_verifications / len(all_verification_results) if all_verification_results else 0.0
        execution_time = time.time() - start_time
        
        # 更新诊断信息的成功率
        for pattern_id, diagnostics in self.diagnostics_by_pattern.items():
            if diagnostics.total_verifications > 0:
                diagnostics.success_rate = diagnostics.successful_verifications / diagnostics.total_verifications
            diagnostics.execution_time = execution_time
        
        result = BatchVerificationResult(
            total_verifications=len(all_verification_results),
            successful_verifications=successful_verifications,
            failed_verifications=failed_verifications,
            success_rate=success_rate,
            execution_time=execution_time,
            verification_results=all_verification_results
        )
        
        logger.info(f"批量验证完成: {successful_verifications}/{len(all_verification_results)} 成功 ({success_rate:.1%})")
        
        # 生成诊断报告
        if self.config.enable_diagnostics:
            self._generate_diagnostics_report()
        
        return result
    
    def _process_batch_with_retry(self, 
                                batch: List[StockSelection],
                                pattern_id: str) -> BatchVerificationResult:
        """
        处理批次并支持重试
        
        Args:
            batch: 选股结果批次
            pattern_id: 形态ID
            
        Returns:
            BatchVerificationResult: 批量验证结果
        """
        retry_count = 0
        last_error = None
        
        while retry_count <= self.config.retry_count:
            try:
                # 创建事件循环
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    # 执行批量验证
                    batch_result = loop.run_until_complete(
                        self.verification_engine.batch_verify_selections(batch)
                    )
                    return batch_result
                    
                finally:
                    loop.close()
                    
            except Exception as e:
                last_error = e
                retry_count += 1
                logger.warning(f"验证批次失败 (重试 {retry_count}/{self.config.retry_count}): {e}")
                time.sleep(1)  # 重试前等待1秒
        
        # 所有重试都失败
        logger.error(f"验证批次在 {self.config.retry_count} 次重试后仍然失败: {last_error}")
        
        # 返回空结果
        return BatchVerificationResult(
            total_verifications=len(batch),
            successful_verifications=0,
            failed_verifications=len(batch),
            success_rate=0.0,
            execution_time=0.0,
            verification_results=[]
        )
    
    def _update_diagnostics(self, pattern_id: str, batch_result: BatchVerificationResult) -> None:
        """
        更新诊断信息
        
        Args:
            pattern_id: 形态ID
            batch_result: 批量验证结果
        """
        if pattern_id not in self.diagnostics_by_pattern:
            return
        
        diagnostics = self.diagnostics_by_pattern[pattern_id]
        
        # 更新成功/失败计数
        diagnostics.successful_verifications += batch_result.successful_verifications
        diagnostics.failed_verifications += batch_result.failed_verifications
        
        # 分析失败原因
        for result in batch_result.verification_results:
            if not result.pattern_match:
                # 记录失败的验证
                self.failed_verifications.append(result)
                
                # 分析失败原因
                failure_reason = self._analyze_failure_reason(result)
                
                if failure_reason not in diagnostics.common_failure_reasons:
                    diagnostics.common_failure_reasons[failure_reason] = 0
                diagnostics.common_failure_reasons[failure_reason] += 1
        
        # 分析形态匹配统计
        pattern_matches = {}
        for result in batch_result.verification_results:
            for detected_pattern in result.detected_patterns:
                if detected_pattern not in pattern_matches:
                    pattern_matches[detected_pattern] = 0
                pattern_matches[detected_pattern] += 1
        
        diagnostics.pattern_match_statistics = pattern_matches
    
    def _analyze_failure_reason(self, result: VerificationResult) -> str:
        """
        分析验证失败原因
        
        Args:
            result: 验证结果
            
        Returns:
            str: 失败原因
        """
        # 检查是否有买点分析结果
        if not result.verification_details:
            return "买点分析失败"
        
        # 检查是否检测到任何形态
        if not result.detected_patterns:
            return "未检测到形态"
        
        # 检查买点分析评分
        score = result.verification_details.get('score', 0)
        if score < 50:
            return f"买点评分过低 ({score})"
        
        # 检查是否有特定的技术指标问题
        if not result.verification_details.get('macd_gold', False):
            return "MACD未金叉"
        
        if not result.verification_details.get('price_stable', False):
            return "价格不稳定"
        
        # 默认原因
        return "形态不匹配"
    
    def _generate_diagnostics_report(self) -> Dict[str, Any]:
        """
        生成诊断报告
        
        Returns:
            Dict[str, Any]: 诊断报告
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_patterns': len(self.diagnostics_by_pattern),
            'pattern_diagnostics': {
                pattern_id: {
                    'total_verifications': diag.total_verifications,
                    'successful_verifications': diag.successful_verifications,
                    'failed_verifications': diag.failed_verifications,
                    'success_rate': diag.success_rate,
                    'common_failure_reasons': diag.common_failure_reasons,
                    'pattern_match_statistics': diag.pattern_match_statistics,
                    'execution_time': diag.execution_time
                }
                for pattern_id, diag in self.diagnostics_by_pattern.items()
            },
            'failed_verifications_count': len(self.failed_verifications),
            'failed_verification_samples': [
                {
                    'stock_code': result.stock_code,
                    'date': result.date,
                    'expected_pattern': result.expected_pattern,
                    'detected_patterns': result.detected_patterns
                }
                for result in self.failed_verifications[:10]  # 只包含前10个样本
            ]
        }
        
        # 计算总体统计
        total_verifications = sum(diag.total_verifications for diag in self.diagnostics_by_pattern.values())
        successful_verifications = sum(diag.successful_verifications for diag in self.diagnostics_by_pattern.values())
        
        report['overall_statistics'] = {
            'total_verifications': total_verifications,
            'successful_verifications': successful_verifications,
            'failed_verifications': total_verifications - successful_verifications,
            'overall_success_rate': successful_verifications / total_verifications if total_verifications > 0 else 0.0
        }
        
        # 输出诊断报告
        logger.info("验证诊断报告:")
        logger.info(f"总形态数: {report['total_patterns']}")
        logger.info(f"总验证数: {total_verifications}")
        logger.info(f"成功验证: {successful_verifications}")
        logger.info(f"总体成功率: {report['overall_statistics']['overall_success_rate']:.1%}")
        
        # 输出前5个形态的诊断信息
        top_patterns = sorted(
            self.diagnostics_by_pattern.items(),
            key=lambda x: x[1].total_verifications,
            reverse=True
        )[:5]
        
        logger.info("前5个形态的诊断信息:")
        for pattern_id, diag in top_patterns:
            logger.info(f"  {pattern_id}: {diag.successful_verifications}/{diag.total_verifications} 成功 ({diag.success_rate:.1%})")
            
            # 输出常见失败原因
            if diag.common_failure_reasons:
                top_reasons = sorted(
                    diag.common_failure_reasons.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
                
                logger.info(f"    常见失败原因: {', '.join(f'{reason}({count})' for reason, count in top_reasons)}")
        
        return report
    
    def export_diagnostics_report(self, output_dir: str = "verification_reports") -> str:
        """
        导出诊断报告
        
        Args:
            output_dir: 输出目录
            
        Returns:
            str: 报告文件路径
        """
        # 生成诊断报告
        report = self._generate_diagnostics_report()
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"verification_diagnostics_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        # 导出报告
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"诊断报告已导出到: {filepath}")
        
        return filepath
    
    def get_verification_statistics(self) -> Dict[str, Any]:
        """
        获取验证统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        # 计算总体统计
        total_verifications = sum(diag.total_verifications for diag in self.diagnostics_by_pattern.values())
        successful_verifications = sum(diag.successful_verifications for diag in self.diagnostics_by_pattern.values())
        
        return {
            'total_patterns': len(self.diagnostics_by_pattern),
            'total_verifications': total_verifications,
            'successful_verifications': successful_verifications,
            'failed_verifications': total_verifications - successful_verifications,
            'overall_success_rate': successful_verifications / total_verifications if total_verifications > 0 else 0.0,
            'pattern_success_rates': {
                pattern_id: diag.success_rate
                for pattern_id, diag in self.diagnostics_by_pattern.items()
            }
        }


async def main():
    """测试批量验证处理器"""
    from analysis.buypoints.analyze_buypoints import BuyPointAnalyzer
    from utils.dependency_injection import get_service
    from db.interfaces.data_access_interface import DataAccessInterface
    
    # 创建验证引擎
    data_access = get_service(DataAccessInterface)
    buypoint_analyzer = BuyPointAnalyzer(data_access)
    verification_engine = BuypointVerificationEngine(buypoint_analyzer)
    
    # 创建批量验证处理器
    config = BatchProcessingConfig(
        max_workers=10,
        batch_size=100,
        timeout_seconds=300,
        retry_count=2,
        enable_diagnostics=True
    )
    processor = BatchVerificationProcessor(verification_engine, config)
    
    # 创建测试数据
    test_selections = []
    for i in range(100):
        test_selections.append(StockSelection(
            stock_code=f"60000{i % 10}",
            stock_name=f"测试股票{i}",
            date="20240101",
            pattern_id=f"TEST_PATTERN_{i % 5}",
            confidence_score=0.8
        ))
    
    # 执行批量验证
    result = await processor.process_verification_batch(test_selections)
    
    # 输出结果
    print(f"验证结果: {result.successful_verifications}/{result.total_verifications} 成功 ({result.success_rate:.1%})")
    
    # 导出诊断报告
    processor.export_diagnostics_report()


if __name__ == "__main__":
    asyncio.run(main())