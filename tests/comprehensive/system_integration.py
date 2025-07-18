#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统集成适配器

连接现有的指标和形态系统，确保兼容性和一致性
"""

import importlib
import inspect
from typing import Dict, List, Any, Optional, Type, Tuple, Set
import pandas as pd
import numpy as np

from utils.logger import getLogger
from indicators.pattern_registry import get_pattern_registry, PatternRegistry
from .error_handler import get_error_handler, ErrorCategory, with_error_handling

logger = getLogger(__name__)


class IndicatorSystemAdapter:
    """指标系统适配器"""
    
    def __init__(self):
        """初始化指标系统适配器"""
        self.pattern_registry = get_pattern_registry()
        self.error_handler = get_error_handler()
        self.indicator_modules = {}
        self.indicator_classes = {}
        
        # 加载核心指标模块
        self._load_core_indicator_modules()
        
        logger.info("指标系统适配器初始化完成")
    
    def _load_core_indicator_modules(self):
        """加载核心指标模块"""
        core_modules = [
            'indicators.ma',
            'indicators.macd',
            'indicators.kdj',
            'indicators.rsi',
            'indicators.boll',
            'indicators.sar',
            'indicators.cci',
            'indicators.common'
        ]
        
        for module_name in core_modules:
            try:
                module = importlib.import_module(module_name)
                self.indicator_modules[module_name] = module
                logger.debug(f"加载指标模块: {module_name}")
                
                # 查找模块中的指标类
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if self._is_indicator_class(obj):
                        self.indicator_classes[name] = obj
                        logger.debug(f"发现指标类: {name}")
                
            except ImportError as e:
                logger.warning(f"无法加载指标模块 {module_name}: {e}")
    
    def _is_indicator_class(self, cls: Type) -> bool:
        """
        判断是否为指标类
        
        Args:
            cls: 类对象
            
        Returns:
            bool: 是否为指标类
        """
        # 检查是否有指标相关的方法
        indicator_methods = ['calculate', 'get_patterns', 'register_patterns']
        has_indicator_methods = any(hasattr(cls, method) for method in indicator_methods)
        
        # 检查类名是否包含指标相关关键词
        indicator_keywords = ['Indicator', 'MA', 'MACD', 'KDJ', 'RSI', 'BOLL', 'SAR', 'CCI']
        has_indicator_name = any(keyword in cls.__name__ for keyword in indicator_keywords)
        
        return has_indicator_methods or has_indicator_name
    
    @with_error_handling(get_error_handler(), ErrorCategory.INDICATOR)
    def create_indicator_instance(self, indicator_name: str) -> Optional[Any]:
        """
        创建指标实例
        
        Args:
            indicator_name: 指标名称
            
        Returns:
            Optional[Any]: 指标实例
        """
        # 尝试直接从已加载的类创建实例
        if indicator_name in self.indicator_classes:
            try:
                return self.indicator_classes[indicator_name]()
            except Exception as e:
                logger.debug(f"无法直接实例化 {indicator_name}: {e}")
        
        # 尝试从模块导入
        try:
            # 常见的指标模块路径
            module_paths = [
                f"indicators.{indicator_name.lower()}",
                f"indicators.trend.{indicator_name.lower()}",
                f"indicators.oscillator.{indicator_name.lower()}",
                f"indicators.volume.{indicator_name.lower()}"
            ]
            
            for module_path in module_paths:
                try:
                    module = importlib.import_module(module_path)
                    
                    # 查找模块中的指标类
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if indicator_name.lower() in name.lower() and self._is_indicator_class(obj):
                            try:
                                return obj()
                            except Exception as e:
                                logger.debug(f"无法实例化 {name}: {e}")
                                continue
                    
                except ImportError:
                    continue
            
            logger.warning(f"无法找到指标 {indicator_name}")
            return None
            
        except Exception as e:
            logger.error(f"创建指标实例 {indicator_name} 失败: {e}")
            return None
    
    @with_error_handling(get_error_handler(), ErrorCategory.INDICATOR)
    def calculate_indicator(self, indicator_name: str, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        计算指标值
        
        Args:
            indicator_name: 指标名称
            data: 股票数据
            
        Returns:
            Optional[pd.DataFrame]: 指标计算结果
        """
        # 创建指标实例
        indicator = self.create_indicator_instance(indicator_name)
        if not indicator:
            return None
        
        # 计算指标值
        try:
            if hasattr(indicator, 'calculate'):
                result = indicator.calculate(data)
                if isinstance(result, pd.DataFrame):
                    return result
                else:
                    logger.warning(f"指标 {indicator_name} 计算结果不是DataFrame")
                    return None
            else:
                logger.warning(f"指标 {indicator_name} 没有calculate方法")
                return None
                
        except Exception as e:
            logger.error(f"计算指标 {indicator_name} 失败: {e}")
            return None
    
    @with_error_handling(get_error_handler(), ErrorCategory.PATTERN)
    def get_indicator_patterns(self, indicator_name: str, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        获取指标形态
        
        Args:
            indicator_name: 指标名称
            data: 股票数据
            
        Returns:
            Optional[pd.DataFrame]: 形态检测结果
        """
        # 创建指标实例
        indicator = self.create_indicator_instance(indicator_name)
        if not indicator:
            return None
        
        # 获取形态
        try:
            if hasattr(indicator, 'get_patterns'):
                patterns = indicator.get_patterns(data)
                if isinstance(patterns, pd.DataFrame):
                    return patterns
                else:
                    logger.warning(f"指标 {indicator_name} 形态结果不是DataFrame")
                    return None
            else:
                logger.warning(f"指标 {indicator_name} 没有get_patterns方法")
                return None
                
        except Exception as e:
            logger.error(f"获取指标 {indicator_name} 形态失败: {e}")
            return None
    
    @with_error_handling(get_error_handler(), ErrorCategory.PATTERN)
    def register_indicator_patterns(self, indicator_name: str) -> bool:
        """
        注册指标形态
        
        Args:
            indicator_name: 指标名称
            
        Returns:
            bool: 是否成功注册
        """
        # 创建指标实例
        indicator = self.create_indicator_instance(indicator_name)
        if not indicator:
            return False
        
        # 注册形态
        try:
            if hasattr(indicator, 'register_patterns'):
                indicator.register_patterns()
                return True
            elif hasattr(indicator, '_register_patterns'):
                indicator._register_patterns()
                return True
            else:
                logger.warning(f"指标 {indicator_name} 没有register_patterns方法")
                return False
                
        except Exception as e:
            logger.error(f"注册指标 {indicator_name} 形态失败: {e}")
            return False
    
    def get_registered_patterns(self, indicator_name: str) -> List[str]:
        """
        获取已注册的形态
        
        Args:
            indicator_name: 指标名称
            
        Returns:
            List[str]: 形态ID列表
        """
        return self.pattern_registry.get_patterns_by_indicator(indicator_name)
    
    def get_pattern_info(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """
        获取形态信息
        
        Args:
            pattern_id: 形态ID
            
        Returns:
            Optional[Dict[str, Any]]: 形态信息
        """
        return self.pattern_registry.get_pattern(pattern_id)


class PatternSystemAdapter:
    """形态系统适配器"""
    
    def __init__(self):
        """初始化形态系统适配器"""
        self.pattern_registry = get_pattern_registry()
        self.error_handler = get_error_handler()
        
        logger.info("形态系统适配器初始化完成")
    
    @with_error_handling(get_error_handler(), ErrorCategory.PATTERN)
    def register_pattern(self, 
                       pattern_id: str, 
                       display_name: str, 
                       indicator_id: str,
                       pattern_type: str = "NEUTRAL",
                       description: str = "",
                       score_impact: float = 0.0) -> bool:
        """
        注册形态
        
        Args:
            pattern_id: 形态ID
            display_name: 显示名称
            indicator_id: 指标ID
            pattern_type: 形态类型
            description: 形态描述
            score_impact: 评分影响
            
        Returns:
            bool: 是否成功注册
        """
        try:
            # 导入形态类型枚举
            from indicators.pattern_registry import PatternTypePatternRegistry
            
            # 转换形态类型
            pattern_type_enum = None
            for enum_item in PatternTypePatternRegistry:
                if enum_item.name == pattern_type or enum_item.value == pattern_type:
                    pattern_type_enum = enum_item
                    break
            
            if pattern_type_enum is None:
                pattern_type_enum = PatternTypePatternRegistry.NEUTRAL
            
            # 注册形态
            self.pattern_registry.register_pattern_registry(
                pattern_id=pattern_id,
                display_name=display_name,
                indicator_id=indicator_id,
                pattern_type=pattern_type_enum,
                description=description,
                score_impact=score_impact,
                allow_override=True
            )
            
            return True
            
        except Exception as e:
            logger.error(f"注册形态 {pattern_id} 失败: {e}")
            return False
    
    def get_all_patterns(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有形态
        
        Returns:
            Dict[str, Dict[str, Any]]: 形态ID到形态信息的映射
        """
        return self.pattern_registry.get_all_patterns()
    
    def get_patterns_by_indicator(self, indicator_id: str) -> List[str]:
        """
        获取指标的所有形态
        
        Args:
            indicator_id: 指标ID
            
        Returns:
            List[str]: 形态ID列表
        """
        return self.pattern_registry.get_patterns_by_indicator(indicator_id)
    
    def get_pattern_info(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """
        获取形态信息
        
        Args:
            pattern_id: 形态ID
            
        Returns:
            Optional[Dict[str, Any]]: 形态信息
        """
        return self.pattern_registry.get_pattern(pattern_id)
    
    def get_pattern_display_name(self, pattern_id: str) -> str:
        """
        获取形态显示名称
        
        Args:
            pattern_id: 形态ID
            
        Returns:
            str: 形态显示名称
        """
        pattern = self.get_pattern_info(pattern_id)
        return pattern.get('display_name', pattern_id) if pattern else pattern_id
    
    def get_pattern_score_impact(self, pattern_id: str) -> float:
        """
        获取形态评分影响
        
        Args:
            pattern_id: 形态ID
            
        Returns:
            float: 形态评分影响
        """
        pattern = self.get_pattern_info(pattern_id)
        return pattern.get('score_impact', 0.0) if pattern else 0.0

class SystemIntegrator:
    """系统集成器"""
    
    def __init__(self):
        """初始化系统集成器"""
        self.indicator_adapter = IndicatorSystemAdapter()
        self.pattern_adapter = PatternSystemAdapter()
        self.error_handler = get_error_handler()
        
        logger.info("系统集成器初始化完成")
    
    def initialize_system(self) -> bool:
        """
        初始化系统
        
        Returns:
            bool: 是否成功初始化
        """
        try:
            # 初始化核心指标
            self._initialize_core_indicators()
            
            # 验证形态注册
            self._validate_pattern_registration()
            
            return True
            
        except Exception as e:
            logger.error(f"初始化系统失败: {e}")
            return False
    
    def _initialize_core_indicators(self) -> None:
        """初始化核心指标"""
        core_indicators = [
            'MA', 'MACD', 'KDJ', 'RSI', 'BOLL', 'SAR', 'CCI'
        ]
        
        for indicator_name in core_indicators:
            try:
                # 创建指标实例
                indicator = self.indicator_adapter.create_indicator_instance(indicator_name)
                if indicator:
                    # 注册形态
                    self.indicator_adapter.register_indicator_patterns(indicator_name)
                    
                    # 获取已注册的形态
                    patterns = self.indicator_adapter.get_registered_patterns(indicator_name)
                    logger.info(f"指标 {indicator_name} 已注册 {len(patterns)} 个形态")
                    
            except Exception as e:
                logger.warning(f"初始化指标 {indicator_name} 失败: {e}")
    
    def _validate_pattern_registration(self) -> None:
        """验证形态注册"""
        # 获取所有形态
        all_patterns = self.pattern_adapter.get_all_patterns()
        
        # 检查是否有形态
        if not all_patterns:
            logger.warning("未找到任何已注册的形态")
            
            # 注册一些默认形态
            self._register_default_patterns()
        else:
            logger.info(f"已注册 {len(all_patterns)} 个形态")
    
    def _register_default_patterns(self) -> None:
        """注册默认形态"""
        default_patterns = [
            # MA形态
            {
                'pattern_id': 'MA_GOLDEN_CROSS',
                'display_name': 'MA金叉',
                'indicator_id': 'MA',
                'pattern_type': 'BULLISH',
                'description': '短期均线上穿长期均线',
                'score_impact': 15.0
            },
            {
                'pattern_id': 'MA_DEATH_CROSS',
                'display_name': 'MA死叉',
                'indicator_id': 'MA',
                'pattern_type': 'BEARISH',
                'description': '短期均线下穿长期均线',
                'score_impact': -15.0
            },
            
            # MACD形态
            {
                'pattern_id': 'MACD_GOLDEN_CROSS',
                'display_name': 'MACD金叉',
                'indicator_id': 'MACD',
                'pattern_type': 'BULLISH',
                'description': 'MACD DIF上穿DEA',
                'score_impact': 15.0
            },
            {
                'pattern_id': 'MACD_DEATH_CROSS',
                'display_name': 'MACD死叉',
                'indicator_id': 'MACD',
                'pattern_type': 'BEARISH',
                'description': 'MACD DIF下穿DEA',
                'score_impact': -15.0
            },
            
            # KDJ形态
            {
                'pattern_id': 'KDJ_OVERSOLD',
                'display_name': 'KDJ超卖',
                'indicator_id': 'KDJ',
                'pattern_type': 'BULLISH',
                'description': 'KDJ指标进入超卖区域',
                'score_impact': 12.0
            },
            {
                'pattern_id': 'KDJ_OVERBOUGHT',
                'display_name': 'KDJ超买',
                'indicator_id': 'KDJ',
                'pattern_type': 'BEARISH',
                'description': 'KDJ指标进入超买区域',
                'score_impact': -12.0
            }
        ]
        
        for pattern in default_patterns:
            self.pattern_adapter.register_pattern(**pattern)
            
        logger.info(f"注册了 {len(default_patterns)} 个默认形态")
    
    def get_indicator_instance(self, indicator_name: str) -> Optional[Any]:
        """
        获取指标实例
        
        Args:
            indicator_name: 指标名称
            
        Returns:
            Optional[Any]: 指标实例
        """
        return self.indicator_adapter.create_indicator_instance(indicator_name)
    
    def calculate_indicator(self, indicator_name: str, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        计算指标值
        
        Args:
            indicator_name: 指标名称
            data: 股票数据
            
        Returns:
            Optional[pd.DataFrame]: 指标计算结果
        """
        return self.indicator_adapter.calculate_indicator(indicator_name, data)
    
    def get_indicator_patterns(self, indicator_name: str, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        获取指标形态
        
        Args:
            indicator_name: 指标名称
            data: 股票数据
            
        Returns:
            Optional[pd.DataFrame]: 形态检测结果
        """
        return self.indicator_adapter.get_indicator_patterns(indicator_name, data)
    
    def get_pattern_info(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """
        获取形态信息
        
        Args:
            pattern_id: 形态ID
            
        Returns:
            Optional[Dict[str, Any]]: 形态信息
        """
        return self.pattern_adapter.get_pattern_info(pattern_id)


# 全局系统集成器实例
_system_integrator = None


def get_system_integrator() -> SystemIntegrator:
    """
    获取全局系统集成器实例
    
    Returns:
        SystemIntegrator: 系统集成器实例
    """
    global _system_integrator
    if _system_integrator is None:
        _system_integrator = SystemIntegrator()
    return _system_integrator


def main():
    """测试系统集成"""
    print("测试系统集成...")
    
    # 创建系统集成器
    integrator = get_system_integrator()
    
    # 初始化系统
    print("\n初始化系统...")
    success = integrator.initialize_system()
    print(f"初始化结果: {'成功' if success else '失败'}")
    
    # 测试指标实例化
    print("\n测试指标实例化...")
    for indicator_name in ['MA', 'MACD', 'KDJ']:
        indicator = integrator.get_indicator_instance(indicator_name)
        print(f"指标 {indicator_name}: {'✓' if indicator else '✗'}")
    
    # 测试形态获取
    print("\n测试形态获取...")
    for indicator_name in ['MA', 'MACD', 'KDJ']:
        patterns = integrator.indicator_adapter.get_registered_patterns(indicator_name)
        print(f"指标 {indicator_name} 形态: {patterns}")


if __name__ == "__main__":
    main()