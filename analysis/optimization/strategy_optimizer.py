"""
策略优化器

基于验证结果自动优化策略条件，提升策略有效性
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json

from utils.logger import get_logger
from analysis.validation.buypoint_validator import BuyPointValidator

logger = get_logger(__name__)


class StrategyOptimizer:
    """策略优化器"""
    
    def __init__(self):
        """初始化优化器"""
        self.validator = BuyPointValidator()
        
    def optimize_strategy(self, 
                         original_strategy: Dict[str, Any],
                         original_buypoints: pd.DataFrame,
                         validation_date: str,
                         max_iterations: int = 5) -> Dict[str, Any]:
        """
        优化策略以提升匹配率
        
        Args:
            original_strategy: 原始策略
            original_buypoints: 原始买点数据
            validation_date: 验证日期
            max_iterations: 最大优化迭代次数
            
        Returns:
            Dict: 优化结果
        """
        optimization_results = {
            'original_strategy': original_strategy.copy(),
            'optimized_strategy': None,
            'optimization_history': [],
            'final_validation': {},
            'improvement_summary': {}
        }
        
        current_strategy = original_strategy.copy()
        best_strategy = current_strategy.copy()
        best_match_rate = 0.0
        
        try:
            # 初始验证
            initial_validation = self.validator.validate_strategy_roundtrip(
                original_buypoints, current_strategy, validation_date)
            initial_match_rate = initial_validation['match_analysis'].get('match_rate', 0)
            best_match_rate = initial_match_rate
            
            logger.info(f"初始匹配率: {initial_match_rate:.2%}")
            
            optimization_results['optimization_history'].append({
                'iteration': 0,
                'strategy': current_strategy.copy(),
                'validation_result': initial_validation,
                'match_rate': initial_match_rate,
                'optimization_type': 'initial'
            })
            
            # 迭代优化
            for iteration in range(1, max_iterations + 1):
                logger.info(f"开始第{iteration}轮优化")
                
                # 获取优化建议
                recommendations = initial_validation.get('recommendations', [])
                if not recommendations:
                    logger.info("没有优化建议，停止优化")
                    break
                
                # 应用优化
                optimized_strategy = self._apply_optimizations(
                    current_strategy, recommendations, original_buypoints)
                
                # 验证优化效果
                validation_result = self.validator.validate_strategy_roundtrip(
                    original_buypoints, optimized_strategy, validation_date)
                
                match_rate = validation_result['match_analysis'].get('match_rate', 0)
                
                # 记录优化历史
                optimization_results['optimization_history'].append({
                    'iteration': iteration,
                    'strategy': optimized_strategy.copy(),
                    'validation_result': validation_result,
                    'match_rate': match_rate,
                    'optimization_type': 'iterative',
                    'applied_recommendations': recommendations
                })
                
                # 检查是否有改进
                if match_rate > best_match_rate:
                    best_strategy = optimized_strategy.copy()
                    best_match_rate = match_rate
                    current_strategy = optimized_strategy
                    initial_validation = validation_result
                    logger.info(f"第{iteration}轮优化成功，匹配率提升至: {match_rate:.2%}")
                else:
                    logger.info(f"第{iteration}轮优化无改进，匹配率: {match_rate:.2%}")
                    # 如果连续没有改进，可以考虑停止
                    if match_rate < best_match_rate * 0.95:  # 如果下降超过5%，停止
                        break
            
            # 设置最终结果
            optimization_results['optimized_strategy'] = best_strategy
            optimization_results['final_validation'] = self.validator.validate_strategy_roundtrip(
                original_buypoints, best_strategy, validation_date)
            
            # 生成改进总结
            optimization_results['improvement_summary'] = self._generate_improvement_summary(
                initial_match_rate, best_match_rate, optimization_results['optimization_history'])
            
            logger.info(f"策略优化完成，最终匹配率: {best_match_rate:.2%}")
            
        except Exception as e:
            logger.error(f"策略优化过程出错: {e}")
            optimization_results['error'] = str(e)
            optimization_results['optimized_strategy'] = original_strategy
        
        return optimization_results
    
    def _apply_optimizations(self, 
                           strategy: Dict[str, Any], 
                           recommendations: List[Dict],
                           original_buypoints: pd.DataFrame) -> Dict[str, Any]:
        """应用优化建议"""
        optimized_strategy = strategy.copy()
        
        for recommendation in recommendations:
            action = recommendation.get('action', '')
            priority = recommendation.get('priority', 'LOW')
            
            # 只处理高优先级和中优先级的建议
            if priority not in ['HIGH', 'MEDIUM']:
                continue
            
            if action == 'adjust_thresholds':
                optimized_strategy = self._adjust_thresholds(optimized_strategy)
            elif action == 'simplify_conditions':
                optimized_strategy = self._simplify_conditions(optimized_strategy)
            elif action == 'add_filters':
                optimized_strategy = self._add_filters(optimized_strategy)
            elif action == 'analyze_missed_stocks':
                # 这个需要更复杂的分析，暂时跳过
                pass
        
        return optimized_strategy
    
    def _adjust_thresholds(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """调整评分阈值"""
        optimized_strategy = strategy.copy()
        conditions = optimized_strategy.get('conditions', [])
        
        for condition in conditions:
            if condition.get('type') == 'indicator' and 'score_threshold' in condition:
                # 降低阈值10%
                current_threshold = condition.get('score_threshold', 60)
                if isinstance(current_threshold, str) and current_threshold.startswith('${'):
                    # 处理变量引用，暂时跳过
                    continue
                
                new_threshold = max(0, current_threshold * 0.9)
                condition['score_threshold'] = new_threshold
        
        logger.info("已调整评分阈值")
        return optimized_strategy
    
    def _simplify_conditions(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """简化策略条件"""
        optimized_strategy = strategy.copy()
        conditions = optimized_strategy.get('conditions', [])
        
        if len(conditions) > 100:
            # 保留前50%的条件（假设它们按重要性排序）
            simplified_conditions = conditions[:len(conditions)//2]
            optimized_strategy['conditions'] = simplified_conditions
            logger.info(f"简化策略条件从{len(conditions)}个减少到{len(simplified_conditions)}个")
        
        return optimized_strategy
    
    def _add_filters(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """添加额外的过滤条件"""
        optimized_strategy = strategy.copy()
        
        # 添加结果过滤器
        if 'result_filters' not in optimized_strategy:
            optimized_strategy['result_filters'] = {}
        
        # 添加最小评分要求
        if 'min_score' not in optimized_strategy['result_filters']:
            optimized_strategy['result_filters']['min_score'] = 70
        
        # 添加最大结果数量限制
        if 'max_results' not in optimized_strategy['result_filters']:
            optimized_strategy['result_filters']['max_results'] = 100
        
        logger.info("已添加额外的过滤条件")
        return optimized_strategy
    
    def _generate_improvement_summary(self, 
                                    initial_rate: float, 
                                    final_rate: float,
                                    history: List[Dict]) -> Dict[str, Any]:
        """生成改进总结"""
        improvement = final_rate - initial_rate
        improvement_percentage = (improvement / initial_rate * 100) if initial_rate > 0 else 0
        
        return {
            'initial_match_rate': initial_rate,
            'final_match_rate': final_rate,
            'absolute_improvement': improvement,
            'percentage_improvement': improvement_percentage,
            'total_iterations': len(history) - 1,  # 减去初始状态
            'optimization_successful': improvement > 0,
            'best_iteration': max(history, key=lambda x: x.get('match_rate', 0))['iteration']
        }
    
    def evaluate_condition_importance(self, 
                                    conditions: List[Dict],
                                    historical_performance: Dict) -> Dict[str, float]:
        """评估条件重要性"""
        importance_scores = {}
        
        for i, condition in enumerate(conditions):
            condition_id = f"condition_{i}"
            
            # 基于历史表现计算重要性
            # 这里使用简化的评估逻辑
            base_score = 0.5
            
            # 根据条件类型调整分数
            if condition.get('type') == 'indicator':
                indicator = condition.get('indicator', '')
                
                # 核心指标权重更高
                if indicator in ['MACD', 'RSI', 'KDJ', 'MA']:
                    base_score += 0.3
                elif indicator in ['BOLL', 'CCI', 'WR']:
                    base_score += 0.2
                else:
                    base_score += 0.1
                
                # 根据周期调整权重
                period = condition.get('period', '')
                if period == 'daily':
                    base_score += 0.2
                elif period in ['weekly', 'monthly']:
                    base_score += 0.1
            
            importance_scores[condition_id] = min(1.0, base_score)
        
        return importance_scores
    
    def auto_optimize_conditions(self, 
                               strategy: Dict[str, Any], 
                               max_conditions: int = 50) -> Dict[str, Any]:
        """自动优化策略条件数量"""
        conditions = strategy.get('conditions', [])
        
        if len(conditions) <= max_conditions:
            return strategy
        
        # 评估条件重要性
        importance_scores = self.evaluate_condition_importance(conditions, {})
        
        # 按重要性排序
        condition_importance = [
            (i, importance_scores.get(f"condition_{i}", 0))
            for i in range(len(conditions))
        ]
        condition_importance.sort(key=lambda x: x[1], reverse=True)
        
        # 保留最重要的条件
        selected_indices = [idx for idx, _ in condition_importance[:max_conditions]]
        selected_indices.sort()  # 保持原始顺序
        
        optimized_conditions = [conditions[i] for i in selected_indices]
        
        optimized_strategy = strategy.copy()
        optimized_strategy['conditions'] = optimized_conditions
        
        logger.info(f"自动优化条件数量从{len(conditions)}减少到{len(optimized_conditions)}")
        
        return optimized_strategy
