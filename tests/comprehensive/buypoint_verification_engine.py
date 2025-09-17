#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
买点验证引擎

复用现有的BuyPointAnalyzer进行闭环验证
整合选股结果与买点分析的形态匹配逻辑
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from utils.logger import getLogger
from analysis.buypoints.analyze_buypoints import BuyPointAnalyzer
from .stock_selection_engine import StockSelection
from db.sql_manager import SQLManager, QueryType

logger = getLogger(__name__)


@dataclass
class VerificationResult:
    """验证结果"""
    stock_code: str
    date: str
    expected_pattern: str
    detected_patterns: List[str]
    pattern_match: bool
    confidence_score: float
    verification_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchVerificationResult:
    """批量验证结果"""
    total_verifications: int
    successful_verifications: int
    failed_verifications: int
    success_rate: float
    execution_time: float
    verification_results: List[VerificationResult] = field(default_factory=list)


class BuypointVerificationEngine:
    """买点验证引擎 - 复用现有BuyPointAnalyzer"""
    
    def __init__(self, buypoint_analyzer: BuyPointAnalyzer):
        """
        初始化买点验证引擎
        
        Args:
            buypoint_analyzer: 现有的买点分析器实例
        """
        self.buypoint_analyzer = buypoint_analyzer
        self.max_workers = 10  # 并行验证线程数
        
        logger.info("买点验证引擎初始化完成")
    
    async def verify_stock_selection(self, 
                                   stock_code: str,
                                   date: str,
                                   stock_name: str,
                                   expected_pattern: str) -> VerificationResult:
        """
        验证单个股票选择
        
        Args:
            stock_code: 股票代码
            date: 日期
            stock_name: 股票名称
            expected_pattern: 预期形态
            
        Returns:
            VerificationResult: 验证结果
        """
        try:
            # 使用现有的买点分析器
            buypoint_result = self.buypoint_analyzer.analyze_stock(
                stock_code=stock_code,
                buy_date=date,
                stock_name=stock_name
            )
            
            if buypoint_result:
                # 提取检测到的形态
                detected_patterns = self._extract_patterns_from_buypoint_result(
                    buypoint_result, expected_pattern
                )
                
                # 判断形态匹配
                pattern_match = self._check_pattern_match(
                    expected_pattern, detected_patterns, buypoint_result
                )
                
                # 计算置信度
                confidence_score = self._calculate_verification_confidence(
                    buypoint_result, pattern_match
                )
                
                return VerificationResult(
                    stock_code=stock_code,
                    date=date,
                    expected_pattern=expected_pattern,
                    detected_patterns=detected_patterns,
                    pattern_match=pattern_match,
                    confidence_score=confidence_score,
                    verification_details=buypoint_result
                )
            else:
                # 买点分析失败
                return VerificationResult(
                    stock_code=stock_code,
                    date=date,
                    expected_pattern=expected_pattern,
                    detected_patterns=[],
                    pattern_match=False,
                    confidence_score=0.0
                )
                
        except Exception as e:
            logger.debug(f"验证股票 {stock_code} 失败: {e}")
            return VerificationResult(
                stock_code=stock_code,
                date=date,
                expected_pattern=expected_pattern,
                detected_patterns=[],
                pattern_match=False,
                confidence_score=0.0
            )
    
    async def batch_verify_selections(self, 
                                    selections: List[StockSelection]) -> BatchVerificationResult:
        """
        批量验证股票选择
        
        Args:
            selections: 股票选择列表
            
        Returns:
            BatchVerificationResult: 批量验证结果
        """
        start_time = time.time()
        
        logger.info(f"开始批量验证 {len(selections)} 个选股结果...")
        
        verification_results = []
        
        # 使用线程池并行验证
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有验证任务
            future_to_selection = {
                executor.submit(
                    self._verify_single_selection, selection
                ): selection
                for selection in selections
            }
            
            # 收集结果
            for future in as_completed(future_to_selection):
                selection = future_to_selection[future]
                
                try:
                    result = future.result(timeout=30)  # 30秒超时
                    verification_results.append(result)
                    
                except Exception as e:
                    logger.error(f"验证选股 {selection.stock_code} 失败: {e}")
                    # 添加失败结果
                    verification_results.append(VerificationResult(
                        stock_code=selection.stock_code,
                        date=selection.date,
                        expected_pattern=selection.pattern_id,
                        detected_patterns=[],
                        pattern_match=False,
                        confidence_score=0.0
                    ))
        
        # 统计结果
        successful_verifications = sum(1 for r in verification_results if r.pattern_match)
        failed_verifications = len(verification_results) - successful_verifications
        success_rate = successful_verifications / len(verification_results) if verification_results else 0.0
        execution_time = time.time() - start_time
        
        result = BatchVerificationResult(
            total_verifications=len(verification_results),
            successful_verifications=successful_verifications,
            failed_verifications=failed_verifications,
            success_rate=success_rate,
            execution_time=execution_time,
            verification_results=verification_results
        )
        
        logger.info(f"批量验证完成: {successful_verifications}/{len(verification_results)} 成功 ({success_rate:.1%})")
        
        return result
    
    def _verify_single_selection(self, selection: StockSelection) -> VerificationResult:
        """
        验证单个选股结果（同步方法，用于线程池）
        
        Args:
            selection: 股票选择
            
        Returns:
            VerificationResult: 验证结果
        """
        try:
            # 使用现有的买点分析器
            buypoint_result = self.buypoint_analyzer.analyze_stock(
                stock_code=selection.stock_code,
                buy_date=selection.date,
                stock_name=selection.stock_name
            )
            
            if buypoint_result:
                # 提取检测到的形态
                detected_patterns = self._extract_patterns_from_buypoint_result(
                    buypoint_result, selection.pattern_id
                )
                
                # 判断形态匹配
                pattern_match = self._check_pattern_match(
                    selection.pattern_id, detected_patterns, buypoint_result
                )
                
                # 计算置信度
                confidence_score = self._calculate_verification_confidence(
                    buypoint_result, pattern_match
                )
                
                return VerificationResult(
                    stock_code=selection.stock_code,
                    date=selection.date,
                    expected_pattern=selection.pattern_id,
                    detected_patterns=detected_patterns,
                    pattern_match=pattern_match,
                    confidence_score=confidence_score,
                    verification_details=buypoint_result
                )
            else:
                # 买点分析失败
                return VerificationResult(
                    stock_code=selection.stock_code,
                    date=selection.date,
                    expected_pattern=selection.pattern_id,
                    detected_patterns=[],
                    pattern_match=False,
                    confidence_score=0.0
                )
                
        except Exception as e:
            logger.debug(f"验证选股 {selection.stock_code} 失败: {e}")
            return VerificationResult(
                stock_code=selection.stock_code,
                date=selection.date,
                expected_pattern=selection.pattern_id,
                detected_patterns=[],
                pattern_match=False,
                confidence_score=0.0
            )
    
    def compare_patterns(self, 
                        expected_pattern: str,
                        detected_patterns: List[str]) -> Dict[str, Any]:
        """
        比较预期形态与检测形态
        
        Args:
            expected_pattern: 预期形态
            detected_patterns: 检测到的形态列表
            
        Returns:
            Dict[str, Any]: 比较结果
        """
        return {
            'expected_pattern': expected_pattern,
            'detected_patterns': detected_patterns,
            'exact_match': expected_pattern in detected_patterns,
            'pattern_count': len(detected_patterns),
            'similarity_score': self._calculate_pattern_similarity(expected_pattern, detected_patterns)
        }
    
    def _extract_patterns_from_buypoint_result(self, 
                                             buypoint_result: Dict[str, Any], 
                                             expected_pattern: str) -> List[str]:
        """
        从买点分析结果中提取形态（复用现有逻辑）
        
        Args:
            buypoint_result: 买点分析结果
            expected_pattern: 预期形态
            
        Returns:
            List[str]: 检测到的形态列表
        """
        detected_patterns = []
        
        try:
            pattern_upper = expected_pattern.upper()
            
            # 看涨形态检测
            if 'BULLISH' in pattern_upper or 'GOLDEN' in pattern_upper:
                if (buypoint_result.get('macd_gold', False) or 
                    buypoint_result.get('ma_up', False) or
                    buypoint_result.get('price_stable', False)):
                    detected_patterns.append(expected_pattern)
            
            # 均线相关形态
            elif 'MA' in pattern_upper:
                if buypoint_result.get('touch_ma', False) or buypoint_result.get('ma_up', False):
                    detected_patterns.append(expected_pattern)
            
            # MACD相关形态
            elif 'MACD' in pattern_upper:
                if buypoint_result.get('macd_gold', False):
                    detected_patterns.append(expected_pattern)
            
            # RSI相关形态
            elif 'RSI' in pattern_upper:
                if buypoint_result.get('rsi_oversold', False):
                    detected_patterns.append(expected_pattern)
            
            # 成交量相关形态
            elif 'VOL' in pattern_upper or 'VOLUME' in pattern_upper:
                if buypoint_result.get('money_in', False) or not buypoint_result.get('vol_shrink', True):
                    detected_patterns.append(expected_pattern)
            
            # 默认检查：基于综合评分
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
    
    def _check_pattern_match(self, 
                           expected_pattern: str, 
                           detected_patterns: List[str], 
                           buypoint_result: Dict[str, Any]) -> bool:
        """
        检查形态是否匹配（复用现有逻辑）
        
        Args:
            expected_pattern: 预期形态
            detected_patterns: 检测到的形态
            buypoint_result: 买点分析结果
            
        Returns:
            bool: 是否匹配
        """
        # 直接匹配
        if expected_pattern in detected_patterns:
            return True
        
        # 基于买点分析评分的匹配
        score = buypoint_result.get('score', 0)
        if score >= 50:  # 评分超过50分认为匹配
            return True
        
        return False
    
    def _calculate_verification_confidence(self, 
                                         buypoint_result: Dict[str, Any], 
                                         pattern_match: bool) -> float:
        """
        计算验证置信度（复用现有逻辑）
        
        Args:
            buypoint_result: 买点分析结果
            pattern_match: 是否匹配
            
        Returns:
            float: 置信度
        """
        if not pattern_match:
            return 0.0
        
        # 基于买点分析评分计算置信度
        score = buypoint_result.get('score', 0)
        confidence = min(score / 100.0, 1.0)  # 将评分转换为0-1的置信度
        
        return max(confidence, 0.1)  # 最低置信度0.1
    
    def _calculate_pattern_similarity(self, 
                                    expected_pattern: str, 
                                    detected_patterns: List[str]) -> float:
        """
        计算形态相似度
        
        Args:
            expected_pattern: 预期形态
            detected_patterns: 检测到的形态列表
            
        Returns:
            float: 相似度分数
        """
        if not detected_patterns:
            return 0.0
        
        # 精确匹配
        if expected_pattern in detected_patterns:
            return 1.0
        
        # 基于关键词的相似度匹配
        expected_keywords = set(expected_pattern.upper().split('_'))
        
        max_similarity = 0.0
        for pattern in detected_patterns:
            pattern_keywords = set(pattern.upper().split('_'))
            
            # 计算交集比例
            intersection = expected_keywords & pattern_keywords
            union = expected_keywords | pattern_keywords
            
            if union:
                similarity = len(intersection) / len(union)
                max_similarity = max(max_similarity, similarity)
        
        return max_similarity
    
    def get_verification_statistics(self, 
                                  verification_results: List[VerificationResult]) -> Dict[str, Any]:
        """
        获取验证统计信息
        
        Args:
            verification_results: 验证结果列表
            
        Returns:
            Dict[str, Any]: 统计信息
        """
        if not verification_results:
            return {}
        
        successful = sum(1 for r in verification_results if r.pattern_match)
        total = len(verification_results)
        
        # 按形态统计
        pattern_stats = {}
        for result in verification_results:
            pattern = result.expected_pattern
            if pattern not in pattern_stats:
                pattern_stats[pattern] = {'total': 0, 'successful': 0}
            
            pattern_stats[pattern]['total'] += 1
            if result.pattern_match:
                pattern_stats[pattern]['successful'] += 1
        
        # 计算各形态成功率
        for pattern, stats in pattern_stats.items():
            stats['success_rate'] = stats['successful'] / stats['total'] if stats['total'] > 0 else 0.0
        
        return {
            'total_verifications': total,
            'successful_verifications': successful,
            'overall_success_rate': successful / total,
            'average_confidence': sum(r.confidence_score for r in verification_results) / total,
            'pattern_statistics': pattern_stats
        }