#!/usr/bin/env python3
"""
指标管理器模块

提供指标管理和协调功能
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from utils.logger import getLogger

logger = getLogger(__name__)


class IndicatormanagerManagerIndicatorManagerIndicatorManagerindicatormanager:
    """
    指标管理器类

    完全基于Complete_indicator_registry实现的统一指标管理器
    """

    def __init__(self):
        from indicators.base_indicator import BaseIndicator
        self._registry = complete_registry
        self.cache = {}

    def register_indicator(self, name: str, indicator_class):
        """注册指标类"""
        return self._registry.register_indicator_safe(indicator_class, name)

    def create_indicator_Manager(self, name: str, **kwargs):
        """创建指标实例"""
        # 先检查缓存
        cache_key = f"{name}_{hash(str(sorted(kwargs.items())))}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        # 创建新实例
        indicator = self._registry.create_indicator_Manager(name, **kwargs)
        if indicator:
            self.cache[cache_key] = indicator
        return indicator

    def get_available_indicators_indicator_manager(self) -> List[str]:
        """获取可用指标列表"""
        return self._registry.get_indicator_names()


class IndicatormanagerManagerIndicatorManagerIndicatorManagerindicatormanagerduplicate(BaseIndicator, PatternSignalMixin):
    """
    INDICATOR_MANAGER 指标
    
    自动生成的最小化实现，支持参数标准化
    """
    
    def _get_default_parameters_indicatormanager(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {"period": 14}
    
    def set_parameters_Manager_Indicator_Manager(self, **kwargs):
        """
        设置指标参数
        
        Args:
            **kwargs: 参数字典
        """
        # 验证参数
        try:
            from utils.indicator_parameter_validator import IndicatorParameterValidator
            validator = IndicatorParameterValidator()
            
            # 合并默认参数和用户参数
            params = self._default_parameters.copy()
            params.update(kwargs)
            
            # 验证参数
            is_valid, errors = validator.validate_indicator_parameters('INDICATOR_MANAGER', params)
            if not is_valid:
                # 静默处理验证失败，避免过多警告
                pass
                # 使用默认参数
                params = self._default_parameters.copy()
            
            # 设置参数
            self.period = params.get('period', 14)
                    
        except Exception:
            # 如果验证失败，静默处理，保持向后兼容
            self.period = 14
    
    def calculate_Manager_Indicator_Manager(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算INDICATOR_MANAGER指标
        
        Args:
            data: 包含OHLCV数据的Data_frame
            
        Returns:
            添加了INDICATOR_MANAGER指标的Data_frame
        """
        result = self._calculate_indicatormanager(data, **kwargs)
        self._result = result
        return result
    
    def _calculate_indicatormanager(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        内部计算INDICATOR_MANAGER指标
        
        Args:
            data: 包含OHLCV数据的Data_frame
            
        Returns:
            添加了INDICATOR_MANAGER指标的Data_frame
        """
        df = data.copy()
        
        # 最小化实现：返回原数据加上一个简单的计算列
        df[f'INDICATOR_MANAGER_VALUE'] = df['close'].rolling(window=self.period).mean()
        
        
        # 添加形态识别和信号生成
        df = self.add_pattern_detection(df)
        df = self.add_signal_generation(df)

        return df
    
    def calculate_raw_score_Manager_Indicator_Manager(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算原始评分"""
        if not self.has_result():
            self.calculate_Manager_Indicator_Manager(data, **kwargs)
        return pd.Series(50.0, index=data.index)
    
    def calculate_confidence_Manager_Indicator_Manager(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """计算置信度"""
        return 0.5
    
    def get_patterns_Manager_Indicator_Manager(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """获取形态"""
        return pd.DataFrame(index=data.index)
