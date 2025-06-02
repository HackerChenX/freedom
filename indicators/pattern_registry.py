"""
形态注册表模块

为技术指标提供形态注册和管理机制
"""

from typing import Dict, List, Any, Optional, Callable, Tuple
from enum import Enum
import pandas as pd
import numpy as np
from utils.logger import get_logger

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
    
    _registry: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register(cls, pattern_id: str, display_name: str, description: str = None,
                 indicator_types: List[str] = None, score_impact: float = 0.0,
                 detection_method: str = None, pattern_type: str = None,
                 signal_type: str = None, min_duration: int = 1, 
                 pattern_strength_range: Tuple[float, float] = (1.0, 10.0)):
        """
        注册技术形态
        
        Args:
            pattern_id: 形态唯一标识
            display_name: 显示名称
            description: 描述
            indicator_types: 适用的指标类型列表
            score_impact: 对评分的影响，正数增加评分，负数减少评分
            detection_method: 检测方法，如'rule_based', 'ml', 'hybrid'等
            pattern_type: 形态类型，如'reversal', 'continuation', 'volatility'等
            signal_type: 信号类型，如'bullish', 'bearish', 'neutral'
            min_duration: 最小持续周期
            pattern_strength_range: 形态强度范围(最小值, 最大值)
        """
        if pattern_id in cls._registry:
            logger.warning(f"形态 {pattern_id} 已存在，将被覆盖")
            
        cls._registry[pattern_id] = {
            'display_name': display_name,
            'description': description,
            'indicator_types': indicator_types or [],
            'score_impact': score_impact,
            'detection_method': detection_method,
            'pattern_type': pattern_type,
            'signal_type': signal_type,
            'min_duration': min_duration,
            'pattern_strength_range': pattern_strength_range
        }
        
        logger.debug(f"注册形态: {pattern_id} -> {display_name}")
        
    @classmethod
    def get_pattern_info(cls, pattern_id: str) -> Optional[Dict[str, Any]]:
        """获取形态信息"""
        if pattern_id not in cls._registry:
            return None
        return cls._registry[pattern_id].copy()
    
    @classmethod
    def get_display_name(cls, pattern_id: str) -> str:
        """获取形态显示名称"""
        if pattern_id not in cls._registry:
            return pattern_id
        return cls._registry[pattern_id]['display_name']
    
    @classmethod
    def get_description(cls, pattern_id: str) -> Optional[str]:
        """获取形态描述"""
        if pattern_id not in cls._registry:
            return None
        return cls._registry[pattern_id]['description']
    
    @classmethod
    def get_score_impact(cls, pattern_id: str) -> float:
        """获取形态对评分的影响"""
        if pattern_id not in cls._registry:
            return 0.0
        return cls._registry[pattern_id]['score_impact']
    
    @classmethod
    def get_signal_type(cls, pattern_id: str) -> Optional[str]:
        """获取形态信号类型"""
        if pattern_id not in cls._registry:
            return None
        return cls._registry[pattern_id]['signal_type']
    
    @classmethod
    def get_pattern_by_signal_type(cls, signal_type: str) -> List[str]:
        """获取指定信号类型的所有形态ID"""
        return [pid for pid, info in cls._registry.items() 
                if info['signal_type'] == signal_type]
    
    @classmethod
    def get_patterns_by_indicator(cls, indicator_type: str) -> List[str]:
        """获取适用于指定指标类型的所有形态ID"""
        return [pid for pid, info in cls._registry.items() 
                if indicator_type in info['indicator_types']]
    
    @classmethod
    def get_all_pattern_ids(cls) -> List[str]:
        """获取所有形态ID"""
        return list(cls._registry.keys())
    
    @classmethod
    def get_all_patterns(cls) -> Dict[str, Dict[str, Any]]:
        """获取所有形态信息"""
        return cls._registry.copy()
    
    @classmethod
    def register_indicator_pattern(cls, indicator_type: str, pattern_id: str, 
                                  display_name: str, description: str = None,
                                  score_impact: float = 0.0, signal_type: str = None):
        """
        注册指标特定的形态
        
        Args:
            indicator_type: 指标类型
            pattern_id: 形态唯一标识
            display_name: 显示名称
            description: 描述
            score_impact: 对评分的影响
            signal_type: 信号类型
        """
        full_pattern_id = f"{indicator_type}_{pattern_id}"
        cls.register(
            pattern_id=full_pattern_id,
            display_name=display_name,
            description=description,
            indicator_types=[indicator_type],
            score_impact=score_impact,
            signal_type=signal_type
        )
        return full_pattern_id
    
    @classmethod
    def import_patterns_from_indicator(cls, indicator):
        """
        从指标实例导入形态

        Args:
            indicator: 指标实例，必须有_registered_patterns属性
        """
        if not hasattr(indicator, '_registered_patterns'):
            logger.warning(f"指标 {indicator.name} 没有_registered_patterns属性")
            return
            
        indicator_type = indicator.name.upper()
        for pattern_id, pattern_info in indicator._registered_patterns.items():
            # 如果形态ID已包含指标类型前缀，则直接使用，否则添加前缀
            if not pattern_id.startswith(f"{indicator_type}_"):
                full_pattern_id = f"{indicator_type}_{pattern_id}"
            else:
                full_pattern_id = pattern_id
                
            cls.register(
                pattern_id=full_pattern_id,
                display_name=pattern_info.get('display_name', pattern_id),
                description=pattern_info.get('description', None),
                indicator_types=[indicator_type],
                score_impact=pattern_info.get('score_impact', 0.0),
                signal_type='bullish' if pattern_info.get('score_impact', 0) > 0 else 
                           ('bearish' if pattern_info.get('score_impact', 0) < 0 else 'neutral')
            )
    
    @classmethod
    def auto_register_from_indicators(cls, indicators: List):
        """
        从指标列表中自动注册所有形态
        
        Args:
            indicators: 指标实例列表
        """
        for indicator in indicators:
            cls.import_patterns_from_indicator(indicator)
            
    @classmethod
    def calculate_combined_score_impact(cls, patterns: List[str]) -> float:
        """
        计算多个形态组合的评分影响
        
        Args:
            patterns: 形态ID列表
            
        Returns:
            float: 组合评分影响
        """
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
            if pattern_id not in cls._registry:
                continue
                
            impact = cls._registry[pattern_id]['score_impact']
            signal_type = cls._registry[pattern_id]['signal_type']
            
            if signal_type == 'bullish':
                bullish_impact += impact
            elif signal_type == 'bearish':
                bearish_impact += impact
            else:
                neutral_impact += impact
                
        # 应用权重
        total_impact = (
            bullish_impact * weights['bullish'] +
            bearish_impact * weights['bearish'] +
            neutral_impact * weights['neutral']
        )
        
        # 限制总影响范围
        return np.clip(total_impact, -25.0, 25.0)


# 注册常见的指标形态
def register_common_patterns():
    """注册常见的技术形态"""
    
    # MACD指标形态
    PatternRegistry.register(
        pattern_id="MACD_GOLDEN_CROSS",
        display_name="MACD金叉",
        description="MACD快线从下方穿过慢线",
        indicator_types=["MACD"],
        score_impact=15.0,
        pattern_type="signal",
        signal_type="bullish"
    )
    
    PatternRegistry.register(
        pattern_id="MACD_DEATH_CROSS",
        display_name="MACD死叉",
        description="MACD快线从上方穿过慢线",
        indicator_types=["MACD"],
        score_impact=-15.0,
        pattern_type="signal",
        signal_type="bearish"
    )
    
    PatternRegistry.register(
        pattern_id="MACD_BULLISH_DIVERGENCE",
        display_name="MACD底背离",
        description="价格创新低而MACD未创新低",
        indicator_types=["MACD"],
        score_impact=20.0,
        pattern_type="reversal",
        signal_type="bullish"
    )
    
    PatternRegistry.register(
        pattern_id="MACD_BEARISH_DIVERGENCE",
        display_name="MACD顶背离",
        description="价格创新高而MACD未创新高",
        indicator_types=["MACD"],
        score_impact=-20.0,
        pattern_type="reversal",
        signal_type="bearish"
    )
    
    # KDJ指标形态
    PatternRegistry.register(
        pattern_id="KDJ_GOLDEN_CROSS",
        display_name="KDJ金叉",
        description="KDJ指标K线从下方穿过D线",
        indicator_types=["KDJ"],
        score_impact=15.0,
        pattern_type="signal",
        signal_type="bullish"
    )
    
    PatternRegistry.register(
        pattern_id="KDJ_DEATH_CROSS",
        display_name="KDJ死叉",
        description="KDJ指标K线从上方穿过D线",
        indicator_types=["KDJ"],
        score_impact=-15.0,
        pattern_type="signal",
        signal_type="bearish"
    )
    
    PatternRegistry.register(
        pattern_id="KDJ_OVERSOLD",
        display_name="KDJ超卖",
        description="KDJ指标K值和D值都低于20",
        indicator_types=["KDJ"],
        score_impact=10.0,
        pattern_type="signal",
        signal_type="bullish"
    )
    
    PatternRegistry.register(
        pattern_id="KDJ_OVERBOUGHT",
        display_name="KDJ超买",
        description="KDJ指标K值和D值都高于80",
        indicator_types=["KDJ"],
        score_impact=-10.0,
        pattern_type="signal",
        signal_type="bearish"
    )
    
    # RSI指标形态
    PatternRegistry.register(
        pattern_id="RSI_OVERSOLD",
        display_name="RSI超卖",
        description="RSI指标值低于30",
        indicator_types=["RSI"],
        score_impact=15.0,
        pattern_type="signal",
        signal_type="bullish"
    )
    
    PatternRegistry.register(
        pattern_id="RSI_OVERBOUGHT",
        display_name="RSI超买",
        description="RSI指标值高于70",
        indicator_types=["RSI"],
        score_impact=-15.0,
        pattern_type="signal",
        signal_type="bearish"
    )
    
    PatternRegistry.register(
        pattern_id="RSI_BULLISH_DIVERGENCE",
        display_name="RSI底背离",
        description="价格创新低而RSI未创新低",
        indicator_types=["RSI"],
        score_impact=20.0,
        pattern_type="reversal",
        signal_type="bullish"
    )
    
    PatternRegistry.register(
        pattern_id="RSI_BEARISH_DIVERGENCE",
        display_name="RSI顶背离",
        description="价格创新高而RSI未创新高",
        indicator_types=["RSI"],
        score_impact=-20.0,
        pattern_type="reversal",
        signal_type="bearish"
    )
    
    # 布林带指标形态
    PatternRegistry.register(
        pattern_id="BOLL_PRICE_BREAK_UPPER",
        display_name="布林带价格突破上轨",
        description="价格突破布林带上轨",
        indicator_types=["BOLL"],
        score_impact=-15.0,
        pattern_type="signal",
        signal_type="bearish"
    )
    
    PatternRegistry.register(
        pattern_id="BOLL_PRICE_BREAK_LOWER",
        display_name="布林带价格突破下轨",
        description="价格突破布林带下轨",
        indicator_types=["BOLL"],
        score_impact=15.0,
        pattern_type="signal",
        signal_type="bullish"
    )
    
    PatternRegistry.register(
        pattern_id="BOLL_BANDWIDTH_EXPANDING",
        display_name="布林带带宽扩大",
        description="布林带带宽明显扩大",
        indicator_types=["BOLL"],
        score_impact=5.0,
        pattern_type="volatility",
        signal_type="neutral"
    )
    
    PatternRegistry.register(
        pattern_id="BOLL_BANDWIDTH_CONTRACTING",
        display_name="布林带带宽收缩",
        description="布林带带宽明显收缩",
        indicator_types=["BOLL"],
        score_impact=-5.0,
        pattern_type="volatility",
        signal_type="neutral"
    )
    
    PatternRegistry.register(
        pattern_id="BOLL_W_BOTTOM",
        display_name="布林带W底",
        description="价格在布林带下轨附近形成W形态",
        indicator_types=["BOLL"],
        score_impact=20.0,
        pattern_type="reversal",
        signal_type="bullish"
    )
    
    PatternRegistry.register(
        pattern_id="BOLL_M_TOP",
        display_name="布林带M顶",
        description="价格在布林带上轨附近形成M形态",
        indicator_types=["BOLL"],
        score_impact=-20.0,
        pattern_type="reversal",
        signal_type="bearish"
    )

# 初始化时注册常见形态
register_common_patterns()

# 单例模式访问
def get_pattern_registry() -> PatternRegistry:
    """
    获取形态注册表实例
    
    Returns:
        PatternRegistry: 形态注册表实例
    """
    return PatternRegistry()


# 注册默认形态的函数
def register_kdj_patterns():
    """注册KDJ指标的形态"""
    registry = get_pattern_registry()
    
    patterns = [
        PatternInfo(
            pattern_id="KDJ_GOLDEN_CROSS",
            display_name="KDJ金叉",
            indicator_id="KDJ",
            pattern_type=PatternType.BULLISH,
            description="K线从下方穿越D线",
            default_strength=PatternStrength.STRONG,
            score_impact=30
        ),
        PatternInfo(
            pattern_id="KDJ_DEATH_CROSS",
            display_name="KDJ死叉",
            indicator_id="KDJ",
            pattern_type=PatternType.BEARISH,
            description="K线从上方穿越D线",
            default_strength=PatternStrength.STRONG,
            score_impact=-30
        ),
        PatternInfo(
            pattern_id="KDJ_OVERSOLD",
            display_name="KDJ超卖",
            indicator_id="KDJ",
            pattern_type=PatternType.BULLISH,
            description="K和D均低于20",
            default_strength=PatternStrength.MEDIUM,
            score_impact=20
        ),
        PatternInfo(
            pattern_id="KDJ_OVERBOUGHT",
            display_name="KDJ超买",
            indicator_id="KDJ",
            pattern_type=PatternType.BEARISH,
            description="K和D均高于80",
            default_strength=PatternStrength.MEDIUM,
            score_impact=-20
        )
    ]
    
    registry.register_patterns_batch(patterns)


def register_rsi_patterns():
    """注册RSI指标的形态"""
    registry = get_pattern_registry()
    
    patterns = [
        PatternInfo(
            pattern_id="RSI_OVERSOLD",
            display_name="RSI超卖",
            indicator_id="RSI",
            pattern_type=PatternType.BULLISH,
            description="RSI低于30",
            default_strength=PatternStrength.MEDIUM,
            score_impact=20
        ),
        PatternInfo(
            pattern_id="RSI_OVERBOUGHT",
            display_name="RSI超买",
            indicator_id="RSI",
            pattern_type=PatternType.BEARISH,
            description="RSI高于70",
            default_strength=PatternStrength.MEDIUM,
            score_impact=-20
        ),
        PatternInfo(
            pattern_id="RSI_BULLISH_DIVERGENCE",
            display_name="RSI底背离",
            indicator_id="RSI",
            pattern_type=PatternType.BULLISH,
            description="价格创新低但RSI未创新低",
            default_strength=PatternStrength.VERY_STRONG,
            score_impact=40
        ),
        PatternInfo(
            pattern_id="RSI_BEARISH_DIVERGENCE",
            display_name="RSI顶背离",
            indicator_id="RSI",
            pattern_type=PatternType.BEARISH,
            description="价格创新高但RSI未创新高",
            default_strength=PatternStrength.VERY_STRONG,
            score_impact=-40
        )
    ]
    
    registry.register_patterns_batch(patterns)


def register_boll_patterns():
    """注册BOLL指标的形态"""
    registry = get_pattern_registry()
    
    patterns = [
        PatternInfo(
            pattern_id="BOLL_SQUEEZE",
            display_name="布林带挤压",
            indicator_id="BOLL",
            pattern_type=PatternType.VOLATILITY,
            description="布林带宽度缩小，预示爆发行情",
            default_strength=PatternStrength.MEDIUM,
            score_impact=0  # 中性信号，需要配合其他信号判断方向
        ),
        PatternInfo(
            pattern_id="BOLL_EXPANSION",
            display_name="布林带扩张",
            indicator_id="BOLL",
            pattern_type=PatternType.VOLATILITY,
            description="布林带宽度扩大，表明波动率增加",
            default_strength=PatternStrength.MEDIUM,
            score_impact=0  # 中性信号
        ),
        PatternInfo(
            pattern_id="BOLL_UPPER_BREAKOUT",
            display_name="突破上轨",
            indicator_id="BOLL",
            pattern_type=PatternType.BULLISH,
            description="价格突破布林带上轨",
            default_strength=PatternStrength.STRONG,
            score_impact=25
        ),
        PatternInfo(
            pattern_id="BOLL_LOWER_BREAKOUT",
            display_name="突破下轨",
            indicator_id="BOLL",
            pattern_type=PatternType.BEARISH,
            description="价格突破布林带下轨",
            default_strength=PatternStrength.STRONG,
            score_impact=-25
        ),
        PatternInfo(
            pattern_id="BOLL_MIDDLE_BOUNCE",
            display_name="中轨支撑反弹",
            indicator_id="BOLL",
            pattern_type=PatternType.BULLISH,
            description="价格在中轨附近获得支撑并反弹",
            default_strength=PatternStrength.MEDIUM,
            score_impact=15
        )
    ]
    
    registry.register_patterns_batch(patterns)


# 注册所有默认形态
def register_all_default_patterns():
    """注册所有默认形态"""
    register_kdj_patterns()
    register_rsi_patterns()
    register_boll_patterns()
    # 后续添加更多指标的形态 