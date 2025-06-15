"""
张新民指标基类

提供ZXM系列指标的共同基类和通用方法
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger

logger = get_logger(__name__)


class BaseZXMIndicator(BaseIndicator, ABC):
    """张新民指标基类"""
    
    def __init__(self, name: str, **kwargs):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """
        初始化ZXM指标基类
        
        Args:
            name: 指标名称
            **kwargs: 其他参数
        """
        super().__init__(name, **kwargs)
        self._score_range = (0, 100)  # ZXM指标的得分范围，默认0-100
        
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> float:
        """
        计算原始评分
        
        这是一个默认实现，子类应该覆盖此方法以提供具体的评分逻辑
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            float: 原始评分值
        """
        logger.warning(f"指标 {self.name} 使用了基类的默认calculate_raw_score方法，返回默认分数0")
        return 0.0
        
    def normalize_score(self, raw_score: float) -> float:
        """
        标准化得分到指定范围
        
        Args:
            raw_score: 原始得分
            
        Returns:
            float: 标准化后的得分
        """
        min_score, max_score = self._score_range
        # 将原始分数标准化到范围内
        normalized = max(min(raw_score, max_score), min_score)
        return normalized
        
    def calculate_indicator_score(self, data: pd.DataFrame, **kwargs) -> float:
        """
        计算指标得分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            float: 指标得分
        """
        try:
            # 计算原始得分
            raw_score = self.calculate_raw_score(data, **kwargs)
            
            # 标准化得分
            normalized_score = self.normalize_score(raw_score)
            
            return normalized_score
        except Exception as e:
            logger.error(f"计算指标 {self.name} 得分时出错: {e}")
            return 0.0  # 错误情况下返回0分 

    def get_pattern_info(self, pattern_id: str) -> dict:
        """
        获取形态信息
        
        Args:
            pattern_id: 形态ID
            
        Returns:
            dict: 形态信息字典
        """
        # 默认形态信息映射
        pattern_info_map = {
            # 基础形态
            'bullish': {'name': '看涨形态', 'description': '指标显示看涨信号', 'type': 'BULLISH'},
            'bearish': {'name': '看跌形态', 'description': '指标显示看跌信号', 'type': 'BEARISH'},
            'neutral': {'name': '中性形态', 'description': '指标显示中性信号', 'type': 'NEUTRAL'},
            
            # 通用形态
            'strong_signal': {'name': '强信号', 'description': '强烈的技术信号', 'type': 'STRONG'},
            'weak_signal': {'name': '弱信号', 'description': '较弱的技术信号', 'type': 'WEAK'},
            'trend_up': {'name': '上升趋势', 'description': '价格呈上升趋势', 'type': 'BULLISH'},
            'trend_down': {'name': '下降趋势', 'description': '价格呈下降趋势', 'type': 'BEARISH'},
        }
        
        # 默认形态信息
        default_pattern = {
            'name': pattern_id.replace('_', ' ').title(),
            'description': f'{pattern_id}形态',
            'type': 'UNKNOWN'
        }
        
        return pattern_info_map.get(pattern_id, default_pattern)
