"""
指标评分管理器模块

提供指标形态评分的统一管理
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from utils.logger import get_logger
from indicators.pattern_registry import PatternRegistry

logger = get_logger(__name__)


class IndicatorScoreManager:
    """指标评分管理器类"""
    
    def __init__(self):
        """初始化评分管理器"""
        self.pattern_registry = PatternRegistry()
        self.default_score = 0
        self.max_score = 100
        self.min_score = 0
        
    def score_pattern(self, pattern_id: str, strength: float = 1.0) -> float:
        """
        为指定形态计算评分
        
        Args:
            pattern_id: 形态ID
            strength: 形态强度系数，默认为1.0
            
        Returns:
            float: 形态评分
        """
        try:
            # 获取形态信息
            pattern_info = self.pattern_registry.get_pattern(pattern_id)
            
            if pattern_info is None:
                logger.warning(f"未找到形态 {pattern_id} 的信息，使用默认评分")
                return self.default_score
                
            # 根据形态类型和强度计算基础分数
            base_score = self._calculate_base_score(pattern_info)
            
            # 应用强度系数
            final_score = base_score * strength
            
            # 限制评分范围
            return max(min(final_score, self.max_score), self.min_score)
        except Exception as e:
            logger.error(f"评分形态 {pattern_id} 时出错: {e}")
            return self.default_score
            
    def _calculate_base_score(self, pattern_info: Dict[str, Any]) -> float:
        """
        计算形态的基础评分
        
        Args:
            pattern_info: 形态信息
            
        Returns:
            float: 基础评分
        """
        # 获取形态强度
        strength = pattern_info.get('default_strength', 'medium')
        
        # 根据强度设置基础分数
        if strength == 'strong':
            base_score = 80
        elif strength == 'medium':
            base_score = 50
        elif strength == 'weak':
            base_score = 30
        else:
            base_score = 0
            
        return base_score
        
    def score_multiple_patterns(self, patterns: List[str], 
                               strength_factors: Optional[Dict[str, float]] = None) -> float:
        """
        为多个形态计算综合评分
        
        Args:
            patterns: 形态ID列表
            strength_factors: 形态强度因子字典，可选
            
        Returns:
            float: 综合评分
        """
        if not patterns:
            return self.default_score
            
        if strength_factors is None:
            strength_factors = {pattern: 1.0 for pattern in patterns}
            
        # 计算每个形态的评分
        scores = []
        for pattern in patterns:
            strength = strength_factors.get(pattern, 1.0)
            score = self.score_pattern(pattern, strength)
            scores.append(score)
            
        # 计算综合评分 - 使用加权平均
        total_score = sum(scores) / len(scores)
        
        return total_score 