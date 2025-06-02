#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
指标管理器

管理技术指标的实例化和获取
"""

import pandas as pd
from typing import Dict, Any, Optional, Type, List
import logging
from collections import defaultdict

from indicators.factory import IndicatorFactory
from utils.decorators import singleton
from utils.logger import get_logger

logger = get_logger(__name__)

@singleton
class IndicatorManager:
    """
    指标管理器
    
    管理技术指标实例，提供缓存和检索功能
    """
    
    def __init__(self):
        """初始化指标管理器"""
        self._indicator_cache = {}
        self._factory = IndicatorFactory()
        self._indicator_params = defaultdict(dict)
        logger.info("指标管理器初始化完成")
    
    def get_indicator(self, indicator_type: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        获取指标实例
        
        Args:
            indicator_type: 指标类型
            params: 指标参数
            
        Returns:
            指标实例
        """
        # 如果没有提供参数，使用空字典
        if params is None:
            params = {}
        
        # 创建缓存键
        cache_key = self._make_cache_key(indicator_type, params)
        
        # 检查缓存
        if cache_key in self._indicator_cache:
            return self._indicator_cache[cache_key]
        
        # 创建新的指标实例
        try:
            indicator = self._factory.create_indicator(indicator_type, params)
            
            # 缓存指标实例
            self._indicator_cache[cache_key] = indicator
            
            return indicator
        except Exception as e:
            logger.error(f"创建指标 {indicator_type} 失败: {str(e)}")
            return None
    
    def calculate_indicator(self, indicator_type: str, data: pd.DataFrame, 
                          params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        计算指标
        
        Args:
            indicator_type: 指标类型
            data: 输入数据
            params: 指标参数
            
        Returns:
            pd.DataFrame: 计算结果
        """
        indicator = self.get_indicator(indicator_type, params)
        
        if indicator is None:
            logger.error(f"无法获取指标 {indicator_type}")
            return pd.DataFrame()
            
        try:
            result = indicator.calculate(data)
            return result
        except Exception as e:
            logger.error(f"计算指标 {indicator_type} 失败: {str(e)}")
            return pd.DataFrame()
    
    def get_indicator_value(self, indicator_type: str, data: pd.DataFrame, 
                          field: str, params: Optional[Dict[str, Any]] = None) -> pd.Series:
        """
        获取指标特定字段的值
        
        Args:
            indicator_type: 指标类型
            data: 输入数据
            field: 字段名
            params: 指标参数
            
        Returns:
            pd.Series: 指标字段值
        """
        indicator = self.get_indicator(indicator_type, params)
        
        if indicator is None:
            logger.error(f"无法获取指标 {indicator_type}")
            return pd.Series()
            
        try:
            result = indicator.calculate(data)
            
            if field in result:
                return result[field]
            else:
                logger.error(f"指标 {indicator_type} 没有字段 {field}")
                return pd.Series()
        except Exception as e:
            logger.error(f"获取指标 {indicator_type} 的字段 {field} 失败: {str(e)}")
            return pd.Series()
    
    def set_default_params(self, indicator_type: str, params: Dict[str, Any]) -> None:
        """
        设置指标默认参数
        
        Args:
            indicator_type: 指标类型
            params: 默认参数
        """
        self._indicator_params[indicator_type] = params
        logger.info(f"设置指标 {indicator_type} 的默认参数: {params}")
    
    def get_default_params(self, indicator_type: str) -> Dict[str, Any]:
        """
        获取指标默认参数
        
        Args:
            indicator_type: 指标类型
            
        Returns:
            Dict[str, Any]: 默认参数
        """
        return self._indicator_params.get(indicator_type, {})
    
    def clear_cache(self) -> None:
        """清除指标缓存"""
        self._indicator_cache.clear()
        logger.info("指标缓存已清除")
    
    def _make_cache_key(self, indicator_type: str, params: Dict[str, Any]) -> str:
        """
        创建缓存键
        
        Args:
            indicator_type: 指标类型
            params: 指标参数
            
        Returns:
            str: 缓存键
        """
        # 将参数字典转为排序后的字符串
        params_str = str(sorted(params.items()))
        return f"{indicator_type}_{params_str}"
    
    def get_available_indicators(self) -> List[str]:
        """
        获取所有可用的指标列表
        
        Returns:
            List[str]: 指标类型列表
        """
        return self._factory.get_available_indicators() 