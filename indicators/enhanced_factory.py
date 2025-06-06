"""
增强型指标工厂模块

提供创建增强型技术指标的工厂类
"""

from typing import Dict, Any, Optional, Type

from indicators.base_indicator import BaseIndicator
from indicators.enhanced_macd import EnhancedMACD
from indicators.enhanced_rsi import EnhancedRSI
from indicators.trend.enhanced_macd import EnhancedMACD as TrendEnhancedMACD
from indicators.trend.enhanced_cci import EnhancedCCI
from indicators.trend.enhanced_trix import EnhancedTRIX
from indicators.psy import PSY


class EnhancedIndicatorFactory:
    """
    增强型指标工厂类
    
    用于创建各种增强型技术指标的实例
    """
    
    _indicator_classes: Dict[str, Type[BaseIndicator]] = {
        "MACD": EnhancedMACD,
        "TREND_MACD": TrendEnhancedMACD,
        "RSI": EnhancedRSI,
        "CCI": EnhancedCCI,
        "TRIX": EnhancedTRIX,
        "PSY": PSY,  # 使用合并后的PSY类
    }
    
    @classmethod
    def create(cls, name: str, **kwargs) -> Optional[BaseIndicator]:
        """
        创建增强型指标实例
        
        Args:
            name: 指标名称
            **kwargs: 指标参数
            
        Returns:
            BaseIndicator: 创建的指标实例，如果指标不存在则返回None
        """
        indicator_class = cls._indicator_classes.get(name.upper())
        if indicator_class:
            # 对于PSY指标，自动添加enhanced=True参数
            if name.upper() == "PSY":
                kwargs["enhanced"] = True
            return indicator_class(**kwargs)
        return None
    
    @classmethod
    def register(cls, name: str, indicator_class: Type[BaseIndicator]) -> None:
        """
        注册新的增强型指标类
        
        Args:
            name: 指标名称
            indicator_class: 指标类
        """
        cls._indicator_classes[name.upper()] = indicator_class
    
    @classmethod
    def get_available_indicators(cls) -> Dict[str, Type[BaseIndicator]]:
        """
        获取所有可用的增强型指标
        
        Returns:
            Dict[str, Type[BaseIndicator]]: 指标名称到指标类的映射
        """
        return cls._indicator_classes.copy()
    
    @classmethod
    def has_indicator(cls, name: str) -> bool:
        """
        检查指定名称的增强型指标是否存在
        
        Args:
            name: 指标名称
            
        Returns:
            bool: 如果指标存在则返回True，否则返回False
        """
        return name.upper() in cls._indicator_classes 