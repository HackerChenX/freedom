"""
L4核心服务层 - 指标管理器统一接口

基于L1L2L3成功修复经验设计的生产级指标管理接口
确保单一入口原则、严格依赖注入、完整向后兼容性
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Type, Optional, Union
from indicators.base_indicator import BaseIndicator


class DependencyInjectionError(Exception):
    """依赖注入错误 - 严格模式异常"""
    pass


class IndicatorRegistrationError(Exception):
    """指标注册错误"""
    pass


class IIndicatorManager(ABC):
    """
    指标管理器统一接口
    
    L4层唯一的指标管理接口，所有指标管理功能必须通过此接口实现
    基于L3层成功经验的严格接口设计
    """
    
    @abstractmethod
    def register_indicator(self, name: str, indicator_class: Type[BaseIndicator], 
                          aliases: Optional[List[str]] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        注册指标类
        
        Args:
            name: 指标名称
            indicator_class: 指标类，必须继承BaseIndicator
            aliases: 指标别名列表
            metadata: 指标元数据
            
        Returns:
            bool: 注册成功返回True
            
        Raises:
            IndicatorRegistrationError: 注册失败时抛出
        """
        pass
    
    @abstractmethod
    def get_indicator(self, name: str) -> Optional[Type[BaseIndicator]]:
        """
        获取指标类
        
        Args:
            name: 指标名称或别名
            
        Returns:
            Optional[Type[BaseIndicator]]: 指标类，未找到返回None
        """
        pass
    
    @abstractmethod
    def create_indicator(self, name: str, **kwargs) -> BaseIndicator:
        """
        创建指标实例
        
        Args:
            name: 指标名称
            **kwargs: 指标参数
            
        Returns:
            BaseIndicator: 指标实例
            
        Raises:
            IndicatorRegistrationError: 指标未注册时抛出
        """
        pass
    
    @abstractmethod
    def list_indicators(self) -> List[str]:
        """
        列出所有已注册指标
        
        Returns:
            List[str]: 指标名称列表
        """
        pass
    
    @abstractmethod
    def get_indicator_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """
        获取指标元数据
        
        Args:
            name: 指标名称
            
        Returns:
            Optional[Dict[str, Any]]: 指标元数据，未找到返回None
        """
        pass
    
    @abstractmethod
    def is_registered(self, name: str) -> bool:
        """
        检查指标是否已注册
        
        Args:
            name: 指标名称
            
        Returns:
            bool: 已注册返回True
        """
        pass
    
    @abstractmethod
    def get_registration_stats(self) -> Dict[str, Any]:
        """
        获取注册统计信息
        
        Returns:
            Dict[str, Any]: 包含注册成功数、失败数、成功率等统计信息
        """
        pass
    
    @abstractmethod
    def validate_indicator_class(self, indicator_class: Type[BaseIndicator]) -> bool:
        """
        验证指标类是否符合规范
        
        Args:
            indicator_class: 指标类
            
        Returns:
            bool: 验证通过返回True
        """
        pass


class IIndicatorFactory(ABC):
    """
    指标工厂接口
    
    提供指标实例创建的统一接口
    """
    
    @abstractmethod
    def create_indicator_instance(self, name: str, **kwargs) -> BaseIndicator:
        """
        创建指标实例
        
        Args:
            name: 指标名称
            **kwargs: 指标参数
            
        Returns:
            BaseIndicator: 指标实例
        """
        pass
    
    @abstractmethod
    def batch_create_indicators(self, indicator_configs: List[Dict[str, Any]]) -> List[BaseIndicator]:
        """
        批量创建指标实例
        
        Args:
            indicator_configs: 指标配置列表，每个配置包含name和参数
            
        Returns:
            List[BaseIndicator]: 指标实例列表
        """
        pass


class IIndicatorRegistry(ABC):
    """
    指标注册表接口
    
    提供指标注册和查询的底层接口
    """
    
    @abstractmethod
    def register_indicator_safe(self, indicator_class: Type[BaseIndicator], name: str) -> bool:
        """
        安全注册指标
        
        Args:
            indicator_class: 指标类
            name: 指标名称
            
        Returns:
            bool: 注册成功返回True
        """
        pass
    
    @abstractmethod
    def get_registered_indicators(self) -> Dict[str, str]:
        """
        获取所有已注册指标
        
        Returns:
            Dict[str, str]: 指标名称到类路径的映射
        """
        pass
    
    @abstractmethod
    def get_failed_indicators(self) -> List[str]:
        """
        获取注册失败的指标
        
        Returns:
            List[str]: 失败指标列表
        """
        pass


# 导出接口
__all__ = [
    'IIndicatorManager',
    'IIndicatorFactory', 
    'IIndicatorRegistry',
    'DependencyInjectionError',
    'IndicatorRegistrationError'
]
