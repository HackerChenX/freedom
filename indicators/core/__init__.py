"""
L4核心服务层 - 统一入口模块

基于L1L2L3成功修复经验的单一入口实现
确保所有指标管理功能通过统一接口访问
"""

from indicators.core.unified_indicator_manager import (
    UnifiedIndicatorManager,
    unified_indicator_manager,
    indicator_manager,
    complete_registry,
    CompatibilityWrapper
)

from indicators.core.indicator_manager_interface import (
    IIndicatorManager,
    IIndicatorFactory,
    IIndicatorRegistry,
    DependencyInjectionError,
    IndicatorRegistrationError
)

# L4层唯一指标管理入口
get_indicator_manager = lambda: unified_indicator_manager

# 向后兼容性函数
def get_complete_registry():
    """获取完整指标注册表 - 向后兼容"""
    return complete_registry

def get_indicator_factory():
    """获取指标工厂 - 向后兼容"""
    return unified_indicator_manager

def create_indicator(name: str, **kwargs):
    """创建指标实例 - 便捷函数"""
    return unified_indicator_manager.create_indicator(name, **kwargs)

def register_indicator(name: str, indicator_class, **kwargs):
    """注册指标 - 便捷函数"""
    return unified_indicator_manager.register_indicator(name, indicator_class, **kwargs)

def list_all_indicators():
    """列出所有指标 - 便捷函数"""
    return unified_indicator_manager.list_indicators()

def get_registration_statistics():
    """获取注册统计 - 便捷函数"""
    return unified_indicator_manager.get_registration_stats()

# 导出所有公共接口
__all__ = [
    # 核心类
    'UnifiedIndicatorManager',
    'unified_indicator_manager',
    
    # 向后兼容别名
    'indicator_manager',
    'complete_registry',
    
    # 接口定义
    'IIndicatorManager',
    'IIndicatorFactory', 
    'IIndicatorRegistry',
    
    # 异常类
    'DependencyInjectionError',
    'IndicatorRegistrationError',
    
    # 便捷函数
    'get_indicator_manager',
    'get_complete_registry',
    'get_indicator_factory',
    'create_indicator',
    'register_indicator',
    'list_all_indicators',
    'get_registration_statistics',
    
    # 兼容性
    'CompatibilityWrapper'
]
