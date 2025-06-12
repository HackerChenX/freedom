"""
指标工厂模块

提供统一的指标创建接口
"""

import importlib
import inspect
import os
import sys
from typing import Dict, Type, Any, Optional, List

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger
from utils.exceptions import IndicatorNotFoundError, IndicatorError

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
        # 使用直接赋值而非lambda，简化创建过程
        if indicator_type not in cls._indicators:
            cls._indicators[indicator_type] = indicator_class
            logger.info(f"已注册指标: {indicator_type} -> {indicator_class.__name__}")
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
        # 确保已经自动注册所有指标
        cls._ensure_auto_registered()
        indicator_class = None  # 确保在try块外可见
        try:
            # 获取指标类
            indicator_class = cls._indicators.get(indicator_type)
            if indicator_class is None:
                raise IndicatorNotFoundError(f"未找到指标类型: {indicator_type}")
            
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
        except IndicatorNotFoundError:
            # 直接重新引发，以便上层可以专门捕获
            raise
        except Exception as e:
            error_msg = f"创建指标 '{indicator_type}' 实例失败"
            if indicator_class:
                error_msg += f" (类: {indicator_class.__name__})"
            logger.error(f"{error_msg}: {e}", exc_info=True)
            raise IndicatorError(error_msg) from e
    
    @classmethod
    def create_indicator_from_config(cls, config: Dict[str, Any]) -> Optional[BaseIndicator]:
        """
        从配置创建指标实例
        
        Args:
            config: 配置字典，至少包含'name'键
            
        Returns:
            Optional[BaseIndicator]: 指标实例，如果创建失败则返回None
        """
        # 提取指标类型名称
        indicator_type = config.get('name')
        if not indicator_type:
            logger.error("配置中缺少'name'字段")
            return None
        
        # 复制配置，移除名称
        params = config.copy()
        params.pop('name', None)
        
        # 创建指标
        return cls.create_indicator(indicator_type, **params)
        
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
    def get_registered_indicators(cls) -> List[str]:
        """
        获取所有已注册的指标类型名称
        
        Returns:
            List[str]: 指标类型名称列表
        """
        # 确保已经自动注册所有指标
        cls._ensure_auto_registered()
            
        return list(cls._indicators.keys())
    
    @classmethod
    def is_registered(cls, indicator_type: str) -> bool:
        """
        检查指标类型是否已注册
        
        Args:
            indicator_type: 指标类型名称
            
        Returns:
            bool: 是否已注册
        """
        return indicator_type in cls._indicators
    
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
        package_name = indicators_pkg.__name__

        # 记录已注册的指标数量
        registered_count = 0
        
        # 遍历indicators包及其子包
        for root, _, files in os.walk(indicators_path):
            for file in files:
                if file.endswith('.py') and file != '__init__.py':
                    module_name = file[:-3]
                    
                    # 构建模块的完整导入路径
                    relative_path = os.path.relpath(root, indicators_path)
                    if relative_path == '.':
                        full_module_name = f"{package_name}.{module_name}"
                    else:
                        sub_package = relative_path.replace(os.sep, '.')
                        full_module_name = f"{package_name}.{sub_package}.{module_name}"

                    logger.info(f"尝试导入模块: {full_module_name}")
                    try:
                        # 导入模块
                        module = importlib.import_module(full_module_name)
                        logger.info(f"正在扫描模块: {full_module_name}")
                        
                        # 从模块中找出所有继承自BaseIndicator的类
                        for name, obj in inspect.getmembers(module):
                            if (inspect.isclass(obj) and 
                                issubclass(obj, BaseIndicator) and 
                                obj is not BaseIndicator and 
                                not inspect.isabstract(obj)):
                                
                                try:
                                    # 实例化指标以检查其可用性
                                    indicator_instance = obj()
                                    if indicator_instance.is_available:
                                        # 使用类名作为默认的指标类型名称
                                        indicator_type = obj.__name__.upper()
                                        cls.register_indicator(indicator_type, obj)
                                        registered_count += 1
                                    else:
                                        logger.debug(f"指标 {obj.__name__} 未标记为可用，跳过注册。")
                                except Exception as e:
                                    logger.warning(f"无法实例化指标 {obj.__name__} 来检查可用性，跳过注册。错误: {e}")
                    except Exception as e:
                        logger.error(f"导入或注册模块 {full_module_name} 时出错: {e}", exc_info=True)
        
        logger.info(f"自动注册完成，共注册了 {registered_count} 个可用指标。")
        
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