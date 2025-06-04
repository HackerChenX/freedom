"""
指标工厂模块

提供统一的指标创建接口
"""

from typing import Dict, Type, Any, Optional, List, Union
import pandas as pd
import importlib
import inspect
import os
import sys
import pkgutil

from enums.indicator_types import IndicatorType
from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger

logger = get_logger(__name__)


class IndicatorFactory:
    """
    指标工厂类
    
    提供统一的指标创建方法
    """
    
    # 指标类型映射表
    _indicators: Dict[str, callable] = {}
    _has_auto_registered = False  # 标记是否已经执行过自动注册
    
    @classmethod
    def create(cls, indicator_type: str, **params) -> Optional[BaseIndicator]:
        """
        创建指标实例
        
        Args:
            indicator_type: 指标类型
            **params: 指标参数
            
        Returns:
            BaseIndicator: 指标实例
        """
        # 确保已经自动注册所有指标
        cls._ensure_auto_registered()
            
        # 获取指标类
        indicator_factory = cls._indicators.get(indicator_type)
        
        if indicator_factory is None:
            logger.error(f"未知的指标类型: {indicator_type}")
            return None
        
        # 创建指标实例
        try:
            return indicator_factory(**params)
        except Exception as e:
            logger.error(f"创建指标实例失败: {e}")
            return None
    
    @classmethod
    def register_indicator(cls, indicator_type: str, indicator_class: Type[BaseIndicator]) -> None:
        """
        注册指标类型
        
        Args:
            indicator_type: 指标类型
            indicator_class: 指标类
        """
        # 使用lambda创建工厂函数
        if indicator_type not in cls._indicators:
            cls._indicators[indicator_type] = lambda **params: indicator_class(**params)
            logger.info(f"已注册指标: {indicator_type}")
        else:
            logger.debug(f"指标 {indicator_type} 已存在，跳过重复注册")
    
    @classmethod
    def get_indicator_types(cls) -> Dict[str, callable]:
        """
        获取所有指标类型
        
        Returns:
            Dict[str, callable]: 指标类型映射表
        """
        # 确保已经自动注册所有指标
        cls._ensure_auto_registered()
            
        return cls._indicators.copy()

    @classmethod
    def create_indicator(cls, indicator_type: str, **kwargs) -> Optional[BaseIndicator]:
        """
        创建指标实例
        
        Args:
            indicator_type: 指标类型
            **kwargs: 传递给指标构造函数的参数
            
        Returns:
            Optional[BaseIndicator]: 指标实例，如果创建失败则返回None
        """
        try:
            # 获取指标类
            indicator_class = cls._indicators.get(indicator_type)
            if indicator_class is None:
                logger.error(f"未找到指标类型: {indicator_type}")
                return None
            
            # 检查是否是抽象类且有未实现的抽象方法
            if hasattr(indicator_class, "__abstractmethods__"):
                abstract_methods = getattr(indicator_class, "__abstractmethods__")
                if abstract_methods:
                    logger.warning(f"指标类 {indicator_type} 有未实现的抽象方法: {abstract_methods}，跳过创建")
                    return None
            
            # 创建指标实例
            indicator = indicator_class(**kwargs)
            
            # 初始化注册的形态
            if hasattr(indicator, "register_patterns") and callable(getattr(indicator, "register_patterns")):
                indicator.register_patterns()
            
            return indicator
        except Exception as e:
            logger.error(f"创建指标实例失败: {e}")
            return None
        
    @classmethod
    def get_supported_indicators(cls) -> list:
        """
        获取所有支持的指标类型名称
        
        Returns:
            list: 指标类型名称列表
        """
        # 确保已经自动注册所有指标
        cls._ensure_auto_registered()
            
        return list(cls._indicators.keys())
    
    @classmethod
    def _ensure_auto_registered(cls) -> None:
        """确保已经执行过自动注册"""
        if not cls._has_auto_registered:
            cls.auto_register_all_indicators()
            cls._has_auto_registered = True
    
    @classmethod
    def auto_register_all_indicators(cls) -> None:
        """
        自动扫描并注册所有继承自BaseIndicator的指标类
        
        会扫描indicators包及其子包中的所有模块，找出所有继承自BaseIndicator的类并注册
        """
        logger.info("开始自动注册所有指标...")
        
        # 获取indicators包的路径
        indicators_pkg = sys.modules.get('indicators')
        if not indicators_pkg:
            logger.error("无法找到indicators包")
            return
            
        indicators_path = os.path.dirname(indicators_pkg.__file__)
        
        # 记录已注册的指标数量
        registered_count = 0
        
        # 遍历indicators包及其子包
        for _, module_name, is_pkg in pkgutil.iter_modules([indicators_path]):
            if module_name in ['__pycache__', 'base_indicator', 'factory', 'indicator_registry', 'common']:
                continue
                
            try:
                # 导入模块
                module = importlib.import_module(f'indicators.{module_name}')
                
                # 从模块中找出所有继承自BaseIndicator的类
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, BaseIndicator) and 
                        obj != BaseIndicator):
                        
                        # 使用类名的大写作为指标ID
                        indicator_id = name.upper()
                        
                        # 注册指标
                        if indicator_id not in cls._indicators:
                            cls.register_indicator(indicator_id, obj)
                            registered_count += 1
                            
            except Exception as e:
                logger.error(f"注册模块 {module_name} 中的指标时出错: {e}")
        
        # 处理子包
        for _, pkg_name, is_pkg in pkgutil.iter_modules([indicators_path]):
            if not is_pkg or pkg_name == '__pycache__':
                continue
                
            try:
                # 导入子包
                pkg = importlib.import_module(f'indicators.{pkg_name}')
                pkg_path = os.path.dirname(pkg.__file__)
                
                # 遍历子包中的模块
                for _, module_name, _ in pkgutil.iter_modules([pkg_path]):
                    if module_name in ['__pycache__', '__init__']:
                        continue
                        
                    try:
                        # 导入模块
                        module = importlib.import_module(f'indicators.{pkg_name}.{module_name}')
                        
                        # 从模块中找出所有继承自BaseIndicator的类
                        for name, obj in inspect.getmembers(module):
                            if (inspect.isclass(obj) and 
                                issubclass(obj, BaseIndicator) and 
                                obj != BaseIndicator):
                                
                                # 使用类名的大写作为指标ID
                                indicator_id = name.upper()
                                
                                # 注册指标
                                if indicator_id not in cls._indicators:
                                    cls.register_indicator(indicator_id, obj)
                                    registered_count += 1
                                    
                    except Exception as e:
                        logger.error(f"注册模块 indicators.{pkg_name}.{module_name} 中的指标时出错: {e}")
                        
            except Exception as e:
                logger.error(f"处理子包 {pkg_name} 时出错: {e}")
        
        logger.info(f"自动注册完成，共注册了 {registered_count} 个指标")
        
        # 设置标记，避免重复注册
        cls._has_auto_registered = True
    
    @classmethod
    def get_all_registered_indicators(cls) -> List[str]:
        """
        获取所有已注册的指标ID
        
        Returns:
            List[str]: 指标ID列表
        """
        # 确保已经自动注册所有指标
        cls._ensure_auto_registered()
            
        return list(cls._indicators.keys())