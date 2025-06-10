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
from indicators.trend_strength import TrendStrength
from indicators.unified_ma import UnifiedMA
from indicators.vol import VOL
from indicators.vosc import VOSC
from indicators.volume_ratio import VR
from indicators.vol import VOL as Volume
from indicators.vortex import Vortex
from indicators.wma import WMA
from indicators.wr import WR

logger = get_logger(__name__)

class IndicatorRegistry:
    """指标注册表，管理所有可用的技术指标"""
    
    _indicators = {}
    _instance = None
    
    def __new__(cls):
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super(IndicatorRegistry, cls).__new__(cls)
            cls._instance._initialize_default_indicators()
        return cls._instance
    
    def _initialize_default_indicators(self):
        """初始化默认指标"""
        # 这里可以初始化系统默认提供的指标
        # 将在后面使用register_indicator注册
        pass
    
    def register_indicator(self, indicator_class, name=None, description=None, overwrite=False):
        """
        注册一个技术指标
        
        Args:
            indicator_class: 指标类
            name: 指标名称，如果为None则使用类名
            description: 指标描述
            overwrite: 是否覆盖已存在的指标
        """
        if name is None:
            name = indicator_class.__name__
            
        if name in self._indicators and not overwrite:
            logger.warning(f"指标 {name} 已存在，将不会覆盖。如需覆盖请设置overwrite=True")
            return False
            
        self._indicators[name] = {
            'class': indicator_class,
            'description': description or getattr(indicator_class, 'description', ''),
            'name': name
        }
        
        logger.info(f"注册指标: {name}")
        return True
    
    def get_indicator_class(self, name):
        """获取指标类"""
        if name not in self._indicators:
            logger.error(f"指标 {name} 未注册")
            return None
        return self._indicators[name]['class']
    
    def create_indicator(self, name, **kwargs):
        """
        创建指标实例
        
        Args:
            name: 指标名称
            **kwargs: 传递给指标构造函数的参数
            
        Returns:
            指标实例或None
        """
        indicator_class = self.get_indicator_class(name)
        if indicator_class is None:
            return None
            
        try:
            return indicator_class(**kwargs)
        except Exception as e:
            logger.error(f"创建指标 {name} 实例失败: {e}")
            return None
    
    def get_all_indicators(self):
        """获取所有注册的指标信息"""
        return self._indicators.copy()
    
    def get_indicator_names(self):
        """获取所有注册的指标名称"""
        return list(self._indicators.keys())

    def register_standard_indicators(self):
        """注册标准指标集"""
        # 导入所有指标类
        from indicators.macd import MACD
        from indicators.kdj import KDJ
        from indicators.rsi import RSI
        from indicators.boll import BOLL
        from indicators.ma import MA
        from indicators.volume import Volume
        
        # 注册指标
        self.register_indicator(MACD, name="MACD", description="移动平均线收敛散度指标")
        self.register_indicator(KDJ, name="KDJ", description="随机指标")
        self.register_indicator(RSI, name="RSI", description="相对强弱指数")
        self.register_indicator(BOLL, name="BOLL", description="布林带")
        self.register_indicator(MA, name="MA", description="移动平均线")
        self.register_indicator(Volume, name="Volume", description="成交量指标")

    def initialize_pattern_registry(self):
        """初始化形态注册表，将所有指标的形态导入到形态注册表中"""
        from indicators.pattern_registry import PatternRegistry
        
        # 创建所有指标的实例
        indicators = []
        for name, info in self._indicators.items():
            try:
                indicator = self.create_indicator(name)
                if indicator:
                    indicators.append(indicator)
            except Exception as e:
                logger.error(f"创建指标 {name} 实例失败: {e}")
                
        # 将所有指标的形态注册到PatternRegistry
        PatternRegistry.auto_register_from_indicators(indicators)
        
        # 记录已注册的形态
        pattern_count = len(PatternRegistry.get_all_pattern_ids())
        logger.info(f"从指标中注册了 {pattern_count} 个形态")

# 初始化并注册标准指标
indicator_registry = IndicatorRegistry()
indicator_registry.register_standard_indicators()
indicator_registry.initialize_pattern_registry()

def get_indicator(name, **kwargs):
    """
    获取指标实例的便捷函数
    
    Args:
        name: 指标名称
        **kwargs: 传递给指标构造函数的参数
        
    Returns:
        指标实例或None
    """
    return indicator_registry.create_indicator(name, **kwargs)

def register_indicator(indicator_class, name=None, description=None, overwrite=False):
    """
    注册指标的便捷函数
    
    Args:
        indicator_class: 指标类
        name: 指标名称
        description: 指标描述
        overwrite: 是否覆盖已存在的指标
        
    Returns:
        bool: 是否成功注册
    """
    return indicator_registry.register_indicator(indicator_class, name, description, overwrite)

def get_all_indicator_names():
    """获取所有注册的指标名称"""
    return indicator_registry.get_indicator_names()

def create_indicator_score_manager(indicators=None, weights=None, market_environment=None):
    """
    创建指标评分管理器
    
    Args:
        indicators: 指标名称列表，如果为None则使用所有注册的指标
        weights: 指标权重字典，键为指标名称，值为权重
        market_environment: 市场环境
        
    Returns:
        IndicatorScoreManager: 指标评分管理器
    """
    from indicators.scoring_framework import IndicatorScoreManager, MarketEnvironment
    
    # 创建管理器
    manager = IndicatorScoreManager()
    
    # 设置市场环境
    if market_environment:
        manager.set_market_environment(market_environment)
        
    # 如果没有指定指标，使用所有注册的指标
    if indicators is None:
        indicators = get_all_indicator_names()
        
    # 如果没有指定权重，使用默认权重1.0
    if weights is None:
        weights = {name: 1.0 for name in indicators}
        
    # 添加指标
    for name in indicators:
        indicator = get_indicator(name)
        if indicator:
            weight = weights.get(name, 1.0)
            manager.add_indicator(indicator, weight)
            
    return manager

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