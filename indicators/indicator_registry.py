#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
指标注册表模块

用于统一管理所有指标的唯一标识和创建函数
"""

from enum import Enum
import importlib
import inspect
from typing import Dict, Type, Callable, Any, Optional, List

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger
from indicators.macd_score import MACDScore
from indicators.kdj_score import KDJScore
from indicators.rsi_score import RSIScore
from indicators.boll_score import BOLLScore
from indicators.scoring_framework import IndicatorScoreBase, IndicatorScoreManager
from indicators.volume_score import VolumeScore

logger = get_logger(__name__)

class IndicatorRegistry:
    """指标注册表，管理所有指标的唯一标识和创建函数"""
    
    _registry: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register(cls, indicator_id: str, indicator_class: Type[BaseIndicator], 
                display_name: str = None, description: str = None,
                parameter_mapping: Dict[str, str] = None):
        """
        注册指标
        
        Args:
            indicator_id: 指标唯一标识
            indicator_class: 指标类
            display_name: 显示名称
            description: 描述
            parameter_mapping: 参数映射，用于将外部参数名映射到指标类初始化参数名
        """
        if indicator_id in cls._registry:
            logger.warning(f"指标 {indicator_id} 已存在，将被覆盖")
            
        if not display_name:
            display_name = indicator_class.__name__
            
        cls._registry[indicator_id] = {
            'class': indicator_class,
            'display_name': display_name,
            'description': description,
            'parameter_mapping': parameter_mapping or {}
        }
        
        logger.debug(f"注册指标: {indicator_id} -> {indicator_class.__name__}")
        
    @classmethod
    def get_indicator_class(cls, indicator_id: str) -> Optional[Type[BaseIndicator]]:
        """获取指标类"""
        if indicator_id not in cls._registry:
            return None
        return cls._registry[indicator_id]['class']
    
    @classmethod
    def get_display_name(cls, indicator_id: str) -> str:
        """获取指标显示名称"""
        if indicator_id not in cls._registry:
            return indicator_id
        return cls._registry[indicator_id]['display_name']
    
    @classmethod
    def get_description(cls, indicator_id: str) -> Optional[str]:
        """获取指标描述"""
        if indicator_id not in cls._registry:
            return None
        return cls._registry[indicator_id]['description']
    
    @classmethod
    def get_parameter_mapping(cls, indicator_id: str) -> Dict[str, str]:
        """获取参数映射"""
        if indicator_id not in cls._registry:
            return {}
        return cls._registry[indicator_id]['parameter_mapping']
    
    @classmethod
    def create_indicator(cls, indicator_id: str, **params) -> Optional[BaseIndicator]:
        """
        创建指标实例
        
        Args:
            indicator_id: 指标唯一标识
            **params: 指标参数
            
        Returns:
            BaseIndicator: 指标实例
        """
        if indicator_id not in cls._registry:
            logger.error(f"指标 {indicator_id} 未注册")
            return None
            
        indicator_class = cls._registry[indicator_id]['class']
        parameter_mapping = cls._registry[indicator_id]['parameter_mapping']
        
        # 应用参数映射
        mapped_params = {}
        for param_name, param_value in params.items():
            # 如果参数名在映射中，使用映射后的名称
            if param_name in parameter_mapping:
                mapped_params[parameter_mapping[param_name]] = param_value
            else:
                mapped_params[param_name] = param_value
                
        try:
            # 创建指标实例
            indicator = indicator_class(**mapped_params)
            return indicator
        except Exception as e:
            logger.error(f"创建指标 {indicator_id} 实例失败: {e}")
            logger.error(f"参数: {mapped_params}")
            return None
    
    @classmethod
    def get_all_indicator_ids(cls) -> List[str]:
        """获取所有指标ID"""
        return list(cls._registry.keys())
    
    @classmethod
    def get_all_indicators(cls) -> Dict[str, Dict[str, Any]]:
        """获取所有指标信息"""
        return cls._registry
    
    @classmethod
    def auto_register_from_module(cls, module_name: str):
        """
        从模块自动注册指标
        
        Args:
            module_name: 模块名，如 'indicators.technical'
        """
        try:
            module = importlib.import_module(module_name)
            
            # 遍历模块中的所有类
            for name, obj in inspect.getmembers(module):
                # 如果是类，并且是BaseIndicator的子类，但不是BaseIndicator本身
                if (inspect.isclass(obj) and 
                    issubclass(obj, BaseIndicator) and 
                    obj != BaseIndicator):
                    
                    # 获取类的 indicator_id 属性，如果没有则使用类名
                    indicator_id = getattr(obj, 'indicator_id', name.upper())
                    
                    # 获取类的 display_name 属性，如果没有则使用类名
                    display_name = getattr(obj, 'display_name', name)
                    
                    # 获取类的 description 属性
                    description = getattr(obj, 'description', obj.__doc__)
                    
                    # 获取类的 parameter_mapping 属性
                    parameter_mapping = getattr(obj, 'parameter_mapping', {})
                    
                    # 注册指标
                    cls.register(
                        indicator_id=indicator_id,
                        indicator_class=obj,
                        display_name=display_name,
                        description=description,
                        parameter_mapping=parameter_mapping
                    )
            
            logger.info(f"从模块 {module_name} 自动注册了 {len(cls._registry)} 个指标")
            
        except Exception as e:
            logger.error(f"从模块 {module_name} 自动注册指标失败: {e}")

    def register_scoring_indicator(self, name: str, indicator_class: type):
        """
        注册评分指标类
        
        Args:
            name: 指标名称
            indicator_class: 指标类
        """
        if not issubclass(indicator_class, IndicatorScoreBase):
            raise ValueError(f"指标类 {indicator_class.__name__} 必须继承自 IndicatorScoreBase")
        
        self._registry[name] = {
            'class': indicator_class,
            'display_name': name,
            'description': indicator_class.__doc__,
            'parameter_mapping': {}
        }
        logger.info(f"注册评分指标: {name} -> {indicator_class.__name__}")

    def create_scoring_indicator(self, name: str, **kwargs) -> IndicatorScoreBase:
        """
        创建评分指标实例
        
        Args:
            name: 指标名称
            **kwargs: 指标参数
            
        Returns:
            IndicatorScoreBase: 指标实例
            
        Raises:
            ValueError: 如果指标未注册
        """
        if name not in self._registry:
            available = list(self._registry.keys())
            raise ValueError(f"未注册的评分指标: {name}. 可用指标: {available}")
        
        indicator_class = self._registry[name]['class']
        return indicator_class(**kwargs)

    def get_available_scoring_indicators(self) -> List[str]:
        """
        获取所有可用的评分指标名称
        
        Returns:
            List[str]: 评分指标名称列表
        """
        return list(self._registry.keys())

    def create_score_manager(self, indicator_configs: List[Dict[str, Any]]) -> IndicatorScoreManager:
        """
        创建指标评分管理器
        
        Args:
            indicator_configs: 指标配置列表，每个配置包含name、weight和其他参数
            
        Returns:
            IndicatorScoreManager: 评分管理器实例
            
        Example:
            configs = [
                {'name': 'macd_score', 'weight': 1.5, 'fast_period': 12},
                {'name': 'kdj_score', 'weight': 1.2, 'n': 9},
                {'name': 'rsi_score', 'weight': 1.0, 'period': 14}
            ]
            manager = registry.create_score_manager(configs)
        """
        manager = IndicatorScoreManager()
        
        for config in indicator_configs:
            name = config.pop('name')
            weight = config.pop('weight', 1.0)
            
            # 创建指标实例
            indicator = self.create_scoring_indicator(name, weight=weight, **config)
            
            # 注册到管理器
            manager.register_indicator(indicator, weight)
            
            logger.info(f"添加评分指标到管理器: {name}, 权重: {weight}")
        
        return manager

# 导出 IndicatorRegistry 单例
indicator_registry = IndicatorRegistry()

# 自动注册评分指标
def _register_scoring_indicators():
    """自动注册所有评分指标"""
    scoring_indicators = {
        'macd_score': MACDScore,
        'kdj_score': KDJScore,
        'rsi_score': RSIScore,
        'boll_score': BOLLScore,
        'volume_score': VolumeScore,
    }
    
    for name, indicator_class in scoring_indicators.items():
        try:
            indicator_registry.register_scoring_indicator(name, indicator_class)
        except Exception as e:
            logger.error(f"注册评分指标 {name} 失败: {e}")

# 执行自动注册
_register_scoring_indicators()

# 定义指标常量枚举
class IndicatorEnum(str, Enum):
    """指标常量枚举"""
    
    # ZXM系列指标
    ZXM_ABSORB = "ZXM_ABSORB"                 # ZXM吸筹
    ZXM_TURNOVER = "ZXM_TURNOVER"             # ZXM换手买点
    ZXM_DAILY_MACD = "ZXM_DAILY_MACD"         # ZXM日MACD买点
    ZXM_MA_CALLBACK = "ZXM_MA_CALLBACK"       # ZXM回踩均线买点
    ZXM_RISE_ELASTICITY = "ZXM_RISE_ELASTICITY"  # ZXM涨幅弹性
    ZXM_AMPLITUDE_ELASTICITY = "ZXM_AMPLITUDE_ELASTICITY"  # ZXM振幅弹性
    ZXM_ELASTICITY_SCORE = "ZXM_ELASTICITY_SCORE"  # ZXM弹性满足
    ZXM_BUYPOINT_SCORE = "ZXM_BUYPOINT_SCORE"  # ZXM买点满足
    ZXM_DAILY_TREND_UP = "ZXM_DAILY_TREND_UP"  # ZXM趋势满足
    
    # 经典技术指标
    MACD = "MACD"                             # MACD指标
    KDJ = "KDJ"                               # KDJ指标
    MA = "MA"                                 # 移动平均线
    BOLL = "BOLL"                             # 布林带
    VR = "VR"                                 # 成交量比率
    OBV = "OBV"                               # 能量潮
    RSI = "RSI"                               # 相对强弱指数
    DIVERGENCE = "DIVERGENCE"                 # 背离
    
    # 新增增强指标
    UNIFIED_MA = "UNIFIED_MA"                 # 统一移动平均线
    ENHANCED_MACD = "ENHANCED_MACD"           # 增强版MACD
    ENHANCED_RSI = "ENHANCED_RSI"             # 增强版RSI

# 定义标准指标参数映射
STANDARD_PARAMETER_MAPPING = {
    IndicatorEnum.ZXM_TURNOVER: {
        "turnover_threshold": "threshold"
    },
    IndicatorEnum.ZXM_DAILY_MACD: {
        "macd_threshold": "threshold"
    },
    IndicatorEnum.ZXM_MA_CALLBACK: {
        "ma_periods": "periods"
    },
    IndicatorEnum.ZXM_RISE_ELASTICITY: {
        "threshold": "rise_threshold"
    },
    IndicatorEnum.ZXM_AMPLITUDE_ELASTICITY: {
        "threshold": "amplitude_threshold"
    },
    IndicatorEnum.ZXM_ABSORB: {
        "threshold": "absorb_threshold"
    },
    IndicatorEnum.ZXM_ELASTICITY_SCORE: {
        "threshold": "threshold"
    },
    IndicatorEnum.ZXM_BUYPOINT_SCORE: {
        "threshold": "threshold"
    },
    # 新增增强指标参数映射
    IndicatorEnum.UNIFIED_MA: {
        "periods": "periods",
        "ma_type": "ma_type",
        "price_column": "price_col"
    },
    IndicatorEnum.ENHANCED_MACD: {
        "fast_period": "fast_period",
        "slow_period": "slow_period",
        "signal_period": "signal_period",
        "price_column": "price_col",
        "use_dual_macd": "use_secondary_macd"
    },
    IndicatorEnum.ENHANCED_RSI: {
        "periods": "periods",
        "price_column": "price_col",
        "use_multi_period": "use_multi_period"
    }
} 