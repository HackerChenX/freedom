#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
指标适配器系统

用于集成不同的指标系统，提供统一的接口
"""

import pandas as pd
import numpy as np
from typing import Union, List, Dict, Optional, Tuple, Any, Callable
import warnings

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger

# 静默警告
warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = get_logger(__name__)


class IndicatorAdapter:
    """
    指标适配器
    
    将不同的指标系统适配到统一的接口，支持不同的数据格式和计算方法
    """
    
    def __init__(self, indicator: BaseIndicator):
        """
        初始化指标适配器
        
        Args:
            indicator: 要适配的指标对象
        """
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        self.indicator = indicator
        self.name = indicator.name
        self.description = indicator.description
    
    def adapt(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        适配指标计算方法，统一输入输出格式
        
        Args:
            data: 输入的数据框
            **kwargs: 其他参数
            
        Returns:
            pd.DataFrame: 包含计算结果的数据框
        """
        # 输入数据预处理
        adapted_data = self._preprocess_data(data)
        
        # 调用原始指标的计算方法
        try:
            result = self.indicator.calculate(adapted_data, **kwargs)
            
            # 输出数据后处理
            final_result = self._postprocess_result(result, data)
            return final_result
        except Exception as e:
            logger.error(f"指标 {self.name} 计算失败: {str(e)}")
            raise
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        预处理输入数据，确保符合指标的输入要求
        
        Args:
            data: 原始输入数据
            
        Returns:
            pd.DataFrame: 预处理后的数据
        """
        # 确保列名标准化
        renamed_data = data.copy()
        
        # 标准列名映射
        column_map = {
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
            'amount': 'amount',
            'turnover': 'turnover',
            'OPEN': 'open',
            'HIGH': 'high',
            'LOW': 'low',
            'CLOSE': 'close',
            'VOLUME': 'volume',
            'AMOUNT': 'amount',
            'TURNOVER': 'turnover',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Amount': 'amount',
            'Turnover': 'turnover',
        }
        
        # 重命名已存在的列
        for old_name, new_name in column_map.items():
            if old_name in renamed_data.columns and old_name != new_name:
                renamed_data.rename(columns={old_name: new_name}, inplace=True)
        
        return renamed_data
    
    def _postprocess_result(self, result: pd.DataFrame, original_data: pd.DataFrame) -> pd.DataFrame:
        """
        后处理结果数据，确保输出格式统一
        
        Args:
            result: 指标计算结果
            original_data: 原始输入数据
            
        Returns:
            pd.DataFrame: 后处理后的结果
        """
        # 如果结果是Series，转换为DataFrame
        if isinstance(result, pd.Series):
            result = result.to_frame()
        
        # 确保结果数据的索引与原始数据一致
        if not isinstance(result, pd.DataFrame):
            logger.warning(f"指标 {self.name} 返回的结果不是DataFrame，尝试转换")
            try:
                result = pd.DataFrame(result)
            except:
                raise ValueError(f"无法将指标 {self.name} 的结果转换为DataFrame")
        
        # 如果结果没有索引或索引与原始数据不同，使用原始数据索引
        if len(result) == len(original_data):
            result.index = original_data.index
        
        return result


class IndicatorRegistry:
    """
    指标注册表
    
    管理和维护所有已注册的指标适配器
    """
    
    def __init__(self):
        """初始化指标注册表"""
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        self.adapters = {}
    
    def register(self, indicator: BaseIndicator, name: Optional[str] = None) -> None:
        """
        注册一个指标到注册表
        
        Args:
            indicator: 要注册的指标对象
            name: 指标的注册名称，如果为None则使用指标自身的name属性
        """
        adapter_name = name or indicator.name
        adapter = IndicatorAdapter(indicator)
        self.adapters[adapter_name] = adapter
        logger.info(f"已注册指标适配器: {adapter_name}")
    
    def get(self, name: str) -> Optional[IndicatorAdapter]:
        """
        获取指定名称的指标适配器
        
        Args:
            name: 指标适配器的名称
            
        Returns:
            IndicatorAdapter: 找到的指标适配器，如果未找到则返回None
        """
        if name in self.adapters:
            return self.adapters[name]
        
        # 尝试不区分大小写的查找
        name_lower = name.lower()
        for adapter_name, adapter in self.adapters.items():
            if adapter_name.lower() == name_lower:
                return adapter
        
        logger.warning(f"未找到名为 {name} 的指标适配器")
        return None
    
    def list_all(self) -> List[str]:
        """
        列出所有已注册的指标适配器名称
        
        Returns:
            List[str]: 指标适配器名称列表
        """
        return list(self.adapters.keys())
    
    def calculate(self, name: str, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        使用指定名称的指标适配器计算指标
        
        Args:
            name: 指标适配器的名称
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.DataFrame: 计算结果
            
        Raises:
            ValueError: 如果找不到指定名称的指标适配器
        """
        adapter = self.get(name)
        if adapter is None:
            raise ValueError(f"未找到名为 {name} 的指标适配器")
        
        return adapter.adapt(data, **kwargs)


# 创建全局指标注册表实例
indicator_registry = IndicatorRegistry()


def register_indicator(indicator: BaseIndicator, name: Optional[str] = None) -> None:
    """
    向全局注册表注册指标
    
    Args:
        indicator: 要注册的指标对象
        name: 指标的注册名称，如果为None则使用指标自身的name属性
    """
    indicator_registry.register(indicator, name)


def get_indicator(name: str) -> Optional[IndicatorAdapter]:
    """
    从全局注册表获取指标适配器
    
    Args:
        name: 指标适配器的名称
        
    Returns:
        IndicatorAdapter: 找到的指标适配器，如果未找到则返回None
    """
    return indicator_registry.get(name)


def calculate_indicator(name: str, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    使用全局注册表中的指标适配器计算指标
    
    Args:
        name: 指标适配器的名称
        data: 输入数据
        **kwargs: 其他参数
        
    Returns:
        pd.DataFrame: 计算结果
        
    Raises:
        ValueError: 如果找不到指定名称的指标适配器
    """
    return indicator_registry.calculate(name, data, **kwargs)


def list_all_indicators() -> List[str]:
    """
    列出全局注册表中所有已注册的指标适配器名称
    
    Returns:
        List[str]: 指标适配器名称列表
    """
    return indicator_registry.list_all()


class CompositeIndicator(BaseIndicator):
    """
    复合指标
    
    组合多个指标的结果，生成一个综合性指标
    """
    def __init__(self, name: str, indicators: List[Union[str, BaseIndicator]], 
                 combination_func: Callable[[List[pd.DataFrame], pd.DataFrame], pd.DataFrame],
                 description: str = ""):
        """
        初始化复合指标
        
        Args:
            name: 复合指标的名称
            indicators: 指标列表，可以是指标名称或指标对象
            combination_func: 组合函数，用于将多个指标结果合并
            description: 指标描述
        """
        super().__init__(name=name, description=description)
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        
        # 验证并初始化指标列表
        self.indicators = []
        
        # 解析指标引用，获取适配器
        for indicator_ref in indicators:
            if isinstance(indicator_ref, str):
                # 从注册表获取指标适配器
                adapter = get_indicator(indicator_ref)
                if adapter is None:
                    logger.warning(f"未找到名为 {indicator_ref} 的指标适配器，将在计算时尝试再次获取")
                self.indicators.append(adapter)
            else:
                # 为指标对象创建新的适配器
                self.indicators.append(IndicatorAdapter(indicator_ref))
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算组合指标
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.DataFrame: 组合指标计算结果
        """
        # 计算所有子指标
        results = []
        
        for i, indicator_ref in enumerate(self.indicators):
            # 如果适配器为None，尝试再次获取
            if indicator_ref is None and isinstance(indicator_ref, str):
                self.indicators[i] = get_indicator(indicator_ref)
                if self.indicators[i] is None:
                    raise ValueError(f"未找到名为 {indicator_ref} 的指标适配器")
            
            # 计算子指标
            try:
                result = self.indicators[i].adapt(data, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"计算子指标 {i} 时出错: {str(e)}")
                raise
        
        # 应用组合函数
        try:
            combined_result = self.combination_func(results, data)
            self._result = combined_result
            return combined_result
        except Exception as e:
            logger.error(f"应用组合函数时出错: {str(e)}")
            raise
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算组合指标原始评分
        
        Args:
            data: 输入数据
            **kwargs: 额外参数
            
        Returns:
            pd.Series: 评分序列，取值范围0-100
        """
        if not self.has_result():
            self.calculate(data, **kwargs)
            
        # 检查是否有名为'score'或'composite_score'的列
        if self._result is not None:
            if 'score' in self._result.columns:
                return self._result['score']
            elif 'composite_score' in self._result.columns:
                return self._result['composite_score']
                
        # 如果没有特定的评分列，尝试从子指标获取评分并平均
        scores = []
        
        for i, adapter in enumerate(self.indicators):
            if adapter is not None and hasattr(adapter.indicator, 'calculate_raw_score'):
                try:
                    score = adapter.indicator.calculate_raw_score(data, **kwargs)
                    scores.append(score)
                except Exception as e:
                    logger.warning(f"获取子指标 {i} 的评分时出错: {str(e)}")
        
        # 如果有可用的子指标评分，计算平均值
        if scores:
            # 将所有评分堆叠成一个DataFrame
            all_scores = pd.concat(scores, axis=1)
            # 计算每行的平均值
            mean_score = all_scores.mean(axis=1)
            return mean_score
        
        # 如果无法获取评分，返回默认的中性评分
        return pd.Series(50.0, index=data.index) 