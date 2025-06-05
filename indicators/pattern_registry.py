"""
形态注册表模块

为技术指标提供形态注册和管理机制
"""

from typing import Dict, List, Any, Optional, Callable, Tuple
from enum import Enum
import pandas as pd
import numpy as np
from utils.logger import get_logger
import os
import json

# 获取日志记录器
logger = get_logger(__name__)


class PatternType(Enum):
    """形态类型枚举"""
    BULLISH = "看涨形态"  # 看涨形态
    BEARISH = "看跌形态"  # 看跌形态
    NEUTRAL = "中性形态"  # 中性形态
    REVERSAL = "反转形态"  # 反转形态
    CONTINUATION = "持续形态"  # 持续形态
    VOLATILITY = "波动形态"  # 波动形态
    OTHER = "其他形态"  # 其他形态
    # 新增类型，兼容现有代码中的字符串类型
    TREND = "趋势形态"  # 趋势形态
    MOMENTUM = "动量形态"  # 动量形态
    EXHAUSTION = "耗尽形态"  # 耗尽形态
    SUPPORT = "支撑形态"  # 支撑形态
    RESISTANCE = "阻力形态"  # 阻力形态
    CONSOLIDATION = "整理形态"  # 整理形态
    # SAR指标特有类型
    STABILITY = "稳定性形态"  # 稳定性形态
    SUPPORT_RESISTANCE = "支撑阻力形态"  # 支撑阻力形态
    WARNING = "警告形态"  # 警告形态
    # BOLL指标特有类型
    BREAKOUT = "突破形态"  # 突破形态


class PatternStrength(Enum):
    """形态强度枚举"""
    VERY_STRONG = 5  # 非常强
    STRONG = 4  # 强
    MEDIUM = 3  # 中等
    WEAK = 2  # 弱
    VERY_WEAK = 1  # 非常弱


class PatternInfo:
    """形态信息类"""
    
    def __init__(self, 
                pattern_id: str,
                display_name: str,
                indicator_id: str,
                pattern_type: PatternType,
                description: str = "",
                default_strength: PatternStrength = PatternStrength.MEDIUM,
                score_impact: int = 0,
                detection_function: Optional[Callable] = None):
        """
        初始化形态信息
        
        Args:
            pattern_id: 形态唯一标识符
            display_name: 形态显示名称
            indicator_id: 关联的指标ID
            pattern_type: 形态类型
            description: 形态描述
            default_strength: 默认形态强度
            score_impact: 形态对评分的影响 (-100 到 100)
            detection_function: 形态检测函数
        """
        self.pattern_id = pattern_id
        self.display_name = display_name
        self.indicator_id = indicator_id
        self.pattern_type = pattern_type
        self.description = description
        self.default_strength = default_strength
        self.score_impact = score_impact
        self.detection_function = detection_function
        
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典表示
        
        Returns:
            Dict[str, Any]: 形态信息的字典表示
        """
        return {
            'pattern_id': self.pattern_id,
            'display_name': self.display_name,
            'indicator_id': self.indicator_id,
            'pattern_type': self.pattern_type.value,
            'description': self.description,
            'default_strength': self.default_strength.value,
            'score_impact': self.score_impact
        }


class PatternRegistry:
    """
    形态注册表，管理所有技术形态的唯一标识和相关信息
    """
    
    _instance = None
    _allow_override = False  # 默认不允许覆盖
    _registered_patterns = set()  # 用于跟踪已注册的形态ID
    
    def __new__(cls):
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super(PatternRegistry, cls).__new__(cls)
            cls._instance._patterns = {}
            cls._instance._patterns_by_indicator = {}  # 按指标名称组织的形态
        return cls._instance
    
    @classmethod
    def _normalize_pattern_id(cls, pattern_id: str, indicator_id: str) -> str:
        """
        规范化形态ID
        
        Args:
            pattern_id: 原始形态ID
            indicator_id: 指标ID
            
        Returns:
            str: 规范化后的形态ID
        """
        # 如果形态ID已经包含指标前缀，则直接使用
        if pattern_id.startswith(f"{indicator_id}_"):
            return pattern_id.upper()
        # 否则添加指标前缀
        return f"{indicator_id}_{pattern_id}".upper()
    
    def register(self, 
                pattern_id: str, 
                display_name: str, 
                indicator_id: str, 
                pattern_type: PatternType = PatternType.NEUTRAL, 
                default_strength: PatternStrength = PatternStrength.MEDIUM,
                description: str = '',
                score_impact: float = 0.0,
                detection_function = None,
                _allow_override: bool = None) -> None:
        """
        注册形态（统一入口）
        
        Args:
            pattern_id: 形态ID
            display_name: 显示名称
            indicator_id: 对应的指标ID
            pattern_type: 形态类型
            default_strength: 默认强度
            description: 形态描述
            score_impact: 对评分的影响值
            detection_function: 形态检测函数
            _allow_override: 是否允许覆盖已注册的形态，默认使用类属性
        """
        # 规范化形态ID
        normalized_pattern_id = self._normalize_pattern_id(pattern_id, indicator_id)
        
        # 如果未指定是否允许覆盖，使用类属性
        if _allow_override is None:
            _allow_override = self.__class__._allow_override
            
        # 检查形态是否已存在
        if normalized_pattern_id in self._registered_patterns and not _allow_override:
            logger.warning(f"形态 {normalized_pattern_id} 已存在，将被跳过")
            return
        elif normalized_pattern_id in self._registered_patterns:
            logger.warning(f"形态 {normalized_pattern_id} 已存在，将被覆盖")
            
        # 创建形态信息
        pattern_info = {
            'pattern_id': normalized_pattern_id,
            'display_name': display_name,
            'indicator_id': indicator_id,
            'pattern_type': pattern_type,
            'description': description,
            'default_strength': default_strength,
            'score_impact': score_impact,
            'detection_function': detection_function
        }
        
        # 存储形态信息
        self._patterns[normalized_pattern_id] = pattern_info
        self._registered_patterns.add(normalized_pattern_id)
        
        # 按指标组织形态
        if indicator_id not in self._patterns_by_indicator:
            self._patterns_by_indicator[indicator_id] = set()
        self._patterns_by_indicator[indicator_id].add(normalized_pattern_id)
        
        logger.debug(f"注册形态: {normalized_pattern_id} ({display_name})")
    
    @classmethod
    def register_indicator_pattern(cls, indicator_type: str, pattern_id: str, 
                                 display_name: str, description: str = None,
                                 score_impact: float = 0.0, signal_type: str = None) -> str:
        """
        注册指标特定的形态（兼容旧接口）
        
        Args:
            indicator_type: 指标类型
            pattern_id: 形态唯一标识
            display_name: 显示名称
            description: 描述
            score_impact: 对评分的影响
            signal_type: 信号类型
            
        Returns:
            str: 规范化后的形态ID
        """
        instance = cls()
        
        # 判断形态类型
        pattern_type = PatternType.NEUTRAL
        if score_impact > 0:
            pattern_type = PatternType.BULLISH
        elif score_impact < 0:
            pattern_type = PatternType.BEARISH
            
        # 判断形态强度
        default_strength = PatternStrength.MEDIUM
        abs_impact = abs(score_impact)
        if abs_impact > 15:
            default_strength = PatternStrength.STRONG
        elif abs_impact < 5:
            default_strength = PatternStrength.WEAK
            
        # 注册形态
        normalized_pattern_id = cls._normalize_pattern_id(pattern_id, indicator_type)
        instance.register(
            pattern_id=normalized_pattern_id,
            display_name=display_name,
            indicator_id=indicator_type,
            pattern_type=pattern_type,
            default_strength=default_strength,
            description=description,
            score_impact=score_impact,
            _allow_override=True
        )
        
        return normalized_pattern_id
    
    @classmethod
    def register_patterns_batch(cls, patterns: List[PatternInfo], _allow_override: bool = False) -> None:
        """
        批量注册形态（兼容旧接口）
        
        Args:
            patterns: 形态信息对象列表
            _allow_override: 是否允许覆盖已注册的形态
        """
        instance = cls()
        for pattern in patterns:
            instance.register(
                pattern_id=pattern.pattern_id,
                display_name=pattern.display_name,
                indicator_id=pattern.indicator_id,
                pattern_type=pattern.pattern_type,
                default_strength=pattern.default_strength,
                description=pattern.description,
                score_impact=pattern.score_impact,
                detection_function=pattern.detection_function,
                _allow_override=_allow_override
            )
    
    @classmethod
    def auto_register_from_indicators(cls, indicators: List) -> None:
        """
        从指标列表中自动注册所有形态
        
        Args:
            indicators: 指标实例列表
        """
        for indicator in indicators:
            if hasattr(indicator, '_register_patterns'):
                indicator._register_patterns()
            elif hasattr(indicator, 'register_patterns_to_registry'):
                indicator.register_patterns_to_registry()
    
    @classmethod
    def set_allow_override(cls, allow: bool) -> None:
        """
        设置是否允许覆盖已存在的形态
        
        Args:
            allow: 是否允许覆盖
        """
        cls._allow_override = allow
    
    @classmethod
    def clear_registry(cls) -> None:
        """
        清空注册表（用于测试）
        """
        instance = cls()
        instance._patterns.clear()
        instance._patterns_by_indicator.clear()
        cls._registered_patterns.clear()
    
    def get_pattern(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """
        获取形态信息
        
        Args:
            pattern_id: 形态ID
            
        Returns:
            Optional[Dict[str, Any]]: 形态信息，如果不存在则返回None
        """
        return self._patterns.get(pattern_id)
        
    @classmethod
    def get_patterns_by_indicator(cls, indicator_id: str) -> List[str]:
        """获取适用于指定指标类型的所有形态ID"""
        instance = cls()
        
        # 将指标ID转为大写以确保匹配
        indicator_id_upper = indicator_id.upper()
        
        # 检查是否在_patterns_by_indicator中存在
        if indicator_id_upper in instance._patterns_by_indicator:
            return list(instance._patterns_by_indicator[indicator_id_upper])
        
        # 兼容旧代码：查找所有以该指标ID开头的形态
        matched_patterns = []
        for pattern_id in instance._patterns.keys():
            if pattern_id.startswith(f"{indicator_id_upper}_"):
                matched_patterns.append(pattern_id)
        
        return matched_patterns
    
    @classmethod
    def get_pattern_info(cls, pattern_id: str) -> Optional[Dict[str, Any]]:
        """获取形态信息"""
        instance = cls()
        if pattern_id not in instance._patterns:
            return None
        return instance._patterns[pattern_id].copy()
    
    @classmethod
    def get_display_name(cls, pattern_id: str) -> str:
        """获取形态显示名称"""
        instance = cls()
        if pattern_id not in instance._patterns:
            return pattern_id
        return instance._patterns[pattern_id]['display_name']
    
    @classmethod
    def get_description(cls, pattern_id: str) -> Optional[str]:
        """获取形态描述"""
        instance = cls()
        if pattern_id not in instance._patterns:
            return None
        return instance._patterns[pattern_id]['description']
    
    @classmethod
    def get_score_impact(cls, pattern_id: str) -> float:
        """获取形态对评分的影响"""
        instance = cls()
        if pattern_id not in instance._patterns:
            return 0.0
        return instance._patterns[pattern_id]['score_impact']
    
    @classmethod
    def get_signal_type(cls, pattern_id: str) -> Optional[str]:
        """获取形态信号类型"""
        instance = cls()
        if pattern_id not in instance._patterns:
            return None
        return instance._patterns[pattern_id].get('signal_type')
    
    @classmethod
    def get_pattern_by_signal_type(cls, signal_type: str) -> List[str]:
        """获取指定信号类型的所有形态ID"""
        instance = cls()
        return [pid for pid, info in instance._patterns.items() 
                if info.get('signal_type') == signal_type]
    
    @classmethod
    def get_all_pattern_ids(cls) -> List[str]:
        """获取所有形态ID"""
        instance = cls()
        return list(instance._patterns.keys())
    
    @classmethod
    def get_all_patterns(cls) -> Dict[str, Dict[str, Any]]:
        """获取所有形态信息"""
        instance = cls()
        return instance._patterns.copy()
    
    @classmethod
    def calculate_combined_score_impact(cls, patterns: List[str]) -> float:
        """
        计算多个形态组合的评分影响
        
        Args:
            patterns: 形态ID列表
            
        Returns:
            float: 组合评分影响
        """
        instance = cls()
        total_impact = 0.0
        weights = {
            'bullish': 1.0,
            'bearish': 1.0,
            'neutral': 0.5
        }
        
        bullish_impact = 0.0
        bearish_impact = 0.0
        neutral_impact = 0.0
        
        for pattern_id in patterns:
            if pattern_id not in instance._patterns:
                logger.warning(f"未找到形态: {pattern_id}")
                continue
                
            pattern_info = instance._patterns[pattern_id]
            impact = pattern_info.get('score_impact', 0.0)
            
            # 根据形态类型分类评分影响
            if isinstance(pattern_info['pattern_type'], PatternType):
                pattern_type = pattern_info['pattern_type']
                if pattern_type == PatternType.BULLISH:
                    bullish_impact += impact
                elif pattern_type == PatternType.BEARISH:
                    bearish_impact += impact
                else:
                    neutral_impact += impact
            else:
                # 兼容直接存储字符串值的情况
                pattern_type_str = str(pattern_info['pattern_type'])
                if '看涨' in pattern_type_str:
                    bullish_impact += impact
                elif '看跌' in pattern_type_str:
                    bearish_impact += impact
                else:
                    neutral_impact += impact
                
        # 应用权重
        total_impact = (
            bullish_impact * weights['bullish'] +
            bearish_impact * weights['bearish'] +
            neutral_impact * weights['neutral']
        )
        
        logger.debug(f"计算评分影响 - 多头: {bullish_impact}, 空头: {bearish_impact}, 中性: {neutral_impact}, 总计: {total_impact}")
        
        # 限制总影响范围
        return np.clip(total_impact, -25.0, 25.0)

    @classmethod
    def import_patterns_from_indicator(cls, indicator):
        """
        从指标实例导入形态（已弃用，保留此方法仅用于兼容性）

        Args:
            indicator: 指标实例
        """
        logger.warning(f"import_patterns_from_indicator 方法已弃用，指标 {indicator.name} 的形态现在直接注册到PatternRegistry")
        return
    
    @classmethod
    def register_patterns_from_config(cls, config_file: str) -> None:
        """
        从配置文件注册形态
        
        Args:
            config_file: 配置文件路径
        """
        if not os.path.exists(config_file):
            logger.warning(f"形态配置文件不存在: {config_file}")
            return
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            registry = cls()
            
            for pattern_config in config.get('patterns', []):
                pattern_id = pattern_config.get('id')
                if not pattern_id:
                    logger.warning(f"形态配置缺少ID，跳过注册: {pattern_config}")
                    continue
                
                indicator_id = pattern_config.get('indicator', '')
                display_name = pattern_config.get('name', pattern_id)
                
                # 解析形态类型
                pattern_type_str = pattern_config.get('type', 'neutral')
                pattern_type = PatternType.NEUTRAL
                if pattern_type_str.lower() == 'bullish':
                    pattern_type = PatternType.BULLISH
                elif pattern_type_str.lower() == 'bearish':
                    pattern_type = PatternType.BEARISH
                
                # 解析形态强度
                strength_str = pattern_config.get('strength', 'medium')
                default_strength = PatternStrength.MEDIUM
                if strength_str.lower() == 'strong':
                    default_strength = PatternStrength.STRONG
                elif strength_str.lower() == 'weak':
                    default_strength = PatternStrength.WEAK
                
                # 注册形态
                registry.register(
                    pattern_id=pattern_id,
                    display_name=display_name,
                    indicator_id=indicator_id,
                    pattern_type=pattern_type,
                    default_strength=default_strength,
                    _allow_override=True
                )
            
            logger.info(f"从配置文件 {config_file} 注册了 {len(config.get('patterns', []))} 个形态")
        except Exception as e:
            logger.error(f"从配置文件注册形态时出错: {e}") 

    @classmethod
    def register_indicator_patterns(cls, indicator_type: str, patterns: List[Dict[str, Any]]) -> None:
        """
        批量注册指标形态
        
        Args:
            indicator_type: 指标类型
            patterns: 形态列表，每个形态为一个字典，包含id、name等信息
        """
        registry = cls()
        
        for pattern_info in patterns:
            pattern_id = pattern_info.get('id')
            if not pattern_id:
                logger.warning(f"形态信息缺少ID，跳过注册: {pattern_info}")
                continue
            
            # 构建完整的形态ID
            full_pattern_id = f"{indicator_type}_{pattern_id}".upper()
            
            # 提取形态信息
            display_name = pattern_info.get('name', pattern_id)
            pattern_type_str = pattern_info.get('type', 'neutral')
            
            # 转换形态类型
            pattern_type = PatternType.NEUTRAL
            if pattern_type_str.lower() == 'bullish':
                pattern_type = PatternType.BULLISH
            elif pattern_type_str.lower() == 'bearish':
                pattern_type = PatternType.BEARISH
            
            # 转换形态强度
            strength_str = pattern_info.get('strength', 'medium')
            default_strength = PatternStrength.MEDIUM
            if strength_str.lower() == 'strong':
                default_strength = PatternStrength.STRONG
            elif strength_str.lower() == 'weak':
                default_strength = PatternStrength.WEAK
            
            # 注册形态
            registry.register(
                pattern_id=full_pattern_id,
                display_name=display_name,
                indicator_id=indicator_type,
                pattern_type=pattern_type,
                default_strength=default_strength,
                _allow_override=True  # 允许覆盖
            ) 

def register_adx_patterns():
    """注册ADX指标相关形态"""
    registry = PatternRegistry()
    
    # 强趋势
    registry.register(
        pattern_id="ADX_STRONG_RISING",
        display_name="ADX强度上升",
        indicator_id="ADX",
        pattern_type=PatternType.NEUTRAL,
        default_strength=PatternStrength.MEDIUM
    )
    
    registry.register(
        pattern_id="ADX_STRONG_FALLING",
        display_name="ADX强度下降",
        indicator_id="ADX",
        pattern_type=PatternType.NEUTRAL,
        default_strength=PatternStrength.MEDIUM
    )
    
    registry.register(
        pattern_id="ADX_WEAK_TREND",
        display_name="ADX弱趋势",
        indicator_id="ADX",
        pattern_type=PatternType.NEUTRAL,
        default_strength=PatternStrength.WEAK
    )
    
    # 趋势方向
    registry.register(
        pattern_id="ADX_BULLISH_CROSS",
        display_name="ADX多头交叉",
        indicator_id="ADX",
        pattern_type=PatternType.BULLISH,
        default_strength=PatternStrength.MEDIUM
    )
    
    registry.register(
        pattern_id="ADX_BEARISH_CROSS",
        display_name="ADX空头交叉",
        indicator_id="ADX",
        pattern_type=PatternType.BEARISH,
        default_strength=PatternStrength.MEDIUM
    )
    
    registry.register(
        pattern_id="ADX_UPTREND",
        display_name="ADX上升趋势",
        indicator_id="ADX",
        pattern_type=PatternType.BULLISH,
        default_strength=PatternStrength.MEDIUM
    )
    
    registry.register(
        pattern_id="ADX_DOWNTREND",
        display_name="ADX下降趋势",
        indicator_id="ADX",
        pattern_type=PatternType.BEARISH,
        default_strength=PatternStrength.MEDIUM
    )
    
    # 趋势变化
    registry.register(
        pattern_id="ADX_TREND_STRENGTHENING",
        display_name="ADX趋势增强",
        indicator_id="ADX",
        pattern_type=PatternType.NEUTRAL,
        default_strength=PatternStrength.STRONG
    )
    
    registry.register(
        pattern_id="ADX_TREND_WEAKENING",
        display_name="ADX趋势减弱",
        indicator_id="ADX",
        pattern_type=PatternType.NEUTRAL,
        default_strength=PatternStrength.WEAK
    )
    
    # 极端趋势
    registry.register(
        pattern_id="ADX_EXTREME_UPTREND",
        display_name="ADX极端上升趋势",
        indicator_id="ADX",
        pattern_type=PatternType.BULLISH,
        default_strength=PatternStrength.STRONG
    )
    
    registry.register(
        pattern_id="ADX_EXTREME_DOWNTREND",
        display_name="ADX极端下降趋势",
        indicator_id="ADX",
        pattern_type=PatternType.BEARISH,
        default_strength=PatternStrength.STRONG
    ) 

def register_atr_patterns():
    """注册ATR指标相关形态"""
    registry = PatternRegistry()
    
    # 波动性分类
    registry.register(
        pattern_id="ATR_HIGH_VOLATILITY",
        display_name="ATR高波动性",
        indicator_id="ATR",
        pattern_type=PatternType.NEUTRAL,
        default_strength=PatternStrength.STRONG
    )
    
    registry.register(
        pattern_id="ATR_LOW_VOLATILITY",
        display_name="ATR低波动性",
        indicator_id="ATR",
        pattern_type=PatternType.NEUTRAL,
        default_strength=PatternStrength.WEAK
    )
    
    registry.register(
        pattern_id="ATR_NORMAL_VOLATILITY",
        display_name="ATR正常波动性",
        indicator_id="ATR",
        pattern_type=PatternType.NEUTRAL,
        default_strength=PatternStrength.MEDIUM
    )
    
    # 波动性变化
    registry.register(
        pattern_id="ATR_RISING_STRONG",
        display_name="ATR强烈上升",
        indicator_id="ATR",
        pattern_type=PatternType.NEUTRAL,
        default_strength=PatternStrength.STRONG
    )
    
    registry.register(
        pattern_id="ATR_RISING",
        display_name="ATR上升",
        indicator_id="ATR",
        pattern_type=PatternType.NEUTRAL,
        default_strength=PatternStrength.MEDIUM
    )
    
    registry.register(
        pattern_id="ATR_FALLING_STRONG",
        display_name="ATR强烈下降",
        indicator_id="ATR",
        pattern_type=PatternType.NEUTRAL,
        default_strength=PatternStrength.STRONG
    )
    
    registry.register(
        pattern_id="ATR_FALLING",
        display_name="ATR下降",
        indicator_id="ATR",
        pattern_type=PatternType.NEUTRAL,
        default_strength=PatternStrength.MEDIUM
    )
    
    registry.register(
        pattern_id="ATR_FLAT",
        display_name="ATR平稳",
        indicator_id="ATR",
        pattern_type=PatternType.NEUTRAL,
        default_strength=PatternStrength.WEAK
    )
    
    # 特殊形态
    registry.register(
        pattern_id="ATR_BREAKOUT",
        display_name="ATR突破",
        indicator_id="ATR",
        pattern_type=PatternType.NEUTRAL,
        default_strength=PatternStrength.STRONG
    )
    
    registry.register(
        pattern_id="ATR_PRICE_VOLATILE",
        display_name="ATR价格波动",
        indicator_id="ATR",
        pattern_type=PatternType.NEUTRAL,
        default_strength=PatternStrength.STRONG
    )
    
    registry.register(
        pattern_id="ATR_PRICE_STABLE",
        display_name="ATR价格稳定",
        indicator_id="ATR",
        pattern_type=PatternType.NEUTRAL,
        default_strength=PatternStrength.WEAK
    )
    
    # 趋势分析
    registry.register(
        pattern_id="ATR_CONVERGENCE",
        display_name="ATR收敛",
        indicator_id="ATR",
        pattern_type=PatternType.NEUTRAL,
        default_strength=PatternStrength.MEDIUM
    )
    
    registry.register(
        pattern_id="ATR_DIVERGENCE",
        display_name="ATR发散",
        indicator_id="ATR",
        pattern_type=PatternType.NEUTRAL,
        default_strength=PatternStrength.MEDIUM
    )
    
    # 极端波动
    registry.register(
        pattern_id="ATR_VOLATILITY_EXPLOSION",
        display_name="ATR波动性爆发",
        indicator_id="ATR",
        pattern_type=PatternType.NEUTRAL,
        default_strength=PatternStrength.STRONG
    )
    
    registry.register(
        pattern_id="ATR_VOLATILITY_COLLAPSE",
        display_name="ATR波动性崩塌",
        indicator_id="ATR",
        pattern_type=PatternType.NEUTRAL,
        default_strength=PatternStrength.STRONG
    )
    
    # 市场状态
    registry.register(
        pattern_id="ATR_MARKET_VOLATILE",
        display_name="ATR市场波动",
        indicator_id="ATR",
        pattern_type=PatternType.NEUTRAL,
        default_strength=PatternStrength.STRONG
    )
    
    registry.register(
        pattern_id="ATR_MARKET_QUIET",
        display_name="ATR市场平静",
        indicator_id="ATR",
        pattern_type=PatternType.NEUTRAL,
        default_strength=PatternStrength.WEAK
    ) 