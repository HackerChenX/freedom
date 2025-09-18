"""
技术形态分析系统 - 指标模块

基于L1L2L3成功修复经验的L4层统一入口
提供各种技术指标的计算和分析功能
"""

# 导入核心类
from indicators.base_indicator import BaseIndicator

# 导入统一管理器 - L4层唯一入口
from indicators.core import (
    unified_indicator_manager,
    indicator_manager,
    complete_registry,
    get_indicator_manager,
    create_indicator,
    register_indicator,
    list_all_indicators
)

# 向后兼容性 - 保持原有接口
IndicatorFactory = unified_indicator_manager
CompleteIndicatorRegistry = complete_registry

__all__ = [
    "BaseIndicator",

    # 统一管理器
    "unified_indicator_manager",
    "indicator_manager",
    "complete_registry",

    # 便捷函数
    "get_indicator_manager",
    "create_indicator",
    "register_indicator",
    "list_all_indicators",

    # 向后兼容
    "IndicatorFactory",
    "CompleteIndicatorRegistry",
]
