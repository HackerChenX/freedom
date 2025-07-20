"""
形态注册表模块

为技术指标提供形态注册和管理机制
"""

from typing import Dict, List, Any, Optional, Callable, Tuple
from enum import Enum
import pandas as pd
import numpy as np
from utils.dependency_injection import get_logger
from utils.dependency_injection import get_container
import os
import json

# 获取日志记录器
logger = get_logger(__name__)


class PatternTypePatternRegistry(Enum):
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


class PatternPolarity(Enum):
    """模式极性枚举 - 用于买点分析过滤"""
    POSITIVE = "POSITIVE"    # 正面/看涨信号，适合买点分析
    NEGATIVE = "NEGATIVE"    # 负面/看跌信号，不适合买点分析
    NEUTRAL = "NEUTRAL"      # 中性信号，信息性质


class PatternStrengthPatternRegistry(Enum):
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
                pattern_type: PatternTypePatternRegistry,
                description: str = "",
                default_strength: PatternStrengthPatternRegistry = PatternStrengthPatternRegistry.MEDIUM,
                score_impact: int = 0,
                detection_function: Optional[Callable] = None,
                polarity: PatternPolarity = PatternPolarity.NEUTRAL):
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
            polarity: 模式极性，用于买点分析过滤
        """
        self.pattern_id = pattern_id
        self.display_name = display_name
        self.indicator_id = indicator_id
        self.pattern_type = pattern_type
        self.description = description
        self.default_strength = default_strength
        self.score_impact = score_impact
        self.detection_function = detection_function
        self.polarity = polarity
        
    def to_dict_registry(self) -> Dict[str, Any]:
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
            'score_impact': self.score_impact,
            'polarity': self.polarity.value
        }


class PatternRegistry:
    """
    形态注册表，管理所有技术形态的唯一标识和相关信息
    重构为普通类，支持依赖注入
    """
    
    def __init__(self):
        """初始化形态注册表"""
        self._patterns = {}
        self._patterns_by_indicator = {}  # 按指标名称组织的形态
        self._allow_override = False  # 默认不允许覆盖
        self._registered_patterns = set()  # 用于跟踪已注册的形态ID
        logger.info("形态注册表初始化完成")
    
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
    
    def register_pattern_registry(self,
                pattern_id: str,
                display_name: str,
                indicator_id: str,
                pattern_type: PatternTypePatternRegistry = PatternTypePatternRegistry.NEUTRAL,
                default_strength: PatternStrengthPatternRegistry = PatternStrengthPatternRegistry.MEDIUM,
                description: str = '',
                score_impact: float = 0.0,
                detection_function = None,
                polarity: PatternPolarity = None,
                allow_override: bool = None) -> None:
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
            polarity: 模式极性，用于买点分析过滤
            allow_override: 是否允许覆盖已注册的形态，默认使用类属性
        """
        # 规范化形态ID
        normalized_pattern_id = self._normalize_pattern_id(pattern_id, indicator_id)

        # 如果未指定是否允许覆盖，使用实例属性
        if allow_override is None:
            allow_override = self._allow_override

        # 检查形态是否已存在
        if normalized_pattern_id in self._registered_patterns and not allow_override:
            return
        elif normalized_pattern_id in self._registered_patterns:
            logger.debug(f"形态 {normalized_pattern_id} 已存在，将被覆盖")

        # 自动推断极性（如果未指定）
        if polarity is None:
            polarity = self._infer_polarity(pattern_type, score_impact, display_name)
            
        # 创建形态信息
        pattern_info = {
            'pattern_id': normalized_pattern_id,
            'display_name': display_name,
            'indicator_id': indicator_id,
            'pattern_type': pattern_type,
            'description': description,
            'default_strength': default_strength,
            'score_impact': score_impact,
            'detection_function': detection_function,
            'polarity': polarity
        }
        
        # 注册形态
        self._patterns[normalized_pattern_id] = pattern_info
        self._registered_patterns.add(normalized_pattern_id)
        
        # 按指标组织
        if indicator_id not in self._patterns_by_indicator:
            self._patterns_by_indicator[indicator_id] = []
        
        if normalized_pattern_id not in self._patterns_by_indicator[indicator_id]:
            self._patterns_by_indicator[indicator_id].append(normalized_pattern_id)
        
        logger.debug(f"注册形态: {normalized_pattern_id} ({display_name})")

    def _infer_polarity(self, pattern_type: PatternTypePatternRegistry, score_impact: float, display_name: str) -> PatternPolarity:
        """
        自动推断形态极性
        
        Args:
            pattern_type: 形态类型
            score_impact: 评分影响
            display_name: 显示名称
            
        Returns:
            PatternPolarity: 推断的极性
        """
        # 基于形态类型推断
        if pattern_type in [PatternTypePatternRegistry.BULLISH, PatternTypePatternRegistry.SUPPORT]:
            return PatternPolarity.POSITIVE
        elif pattern_type in [PatternTypePatternRegistry.BEARISH, PatternTypePatternRegistry.RESISTANCE]:
            return PatternPolarity.NEGATIVE
        
        # 基于评分影响推断
        if score_impact > 5:
            return PatternPolarity.POSITIVE
        elif score_impact < -5:
            return PatternPolarity.NEGATIVE
        
        # 基于显示名称推断
        positive_keywords = ['看涨', '买入', '支撑', '突破', '上涨', '强势']
        negative_keywords = ['看跌', '卖出', '阻力', '下跌', '弱势', '空头']
        
        display_lower = display_name.lower()
        if any(keyword in display_lower for keyword in positive_keywords):
            return PatternPolarity.POSITIVE
        elif any(keyword in display_lower for keyword in negative_keywords):
            return PatternPolarity.NEGATIVE
        
        return PatternPolarity.NEUTRAL

    def register_all_patterns(self) -> None:
        """注册所有全局形态"""
        self._register_global_patterns()
    
    def _register_global_patterns(self) -> None:
        """注册全局通用形态"""
        global_patterns = [
            ("BULLISH_SIGNAL", "看涨信号", "GLOBAL", PatternTypePatternRegistry.BULLISH),
            ("BEARISH_SIGNAL", "看跌信号", "GLOBAL", PatternTypePatternRegistry.BEARISH),
            ("NEUTRAL_SIGNAL", "中性信号", "GLOBAL", PatternTypePatternRegistry.NEUTRAL),
            ("TREND_REVERSAL", "趋势反转", "GLOBAL", PatternTypePatternRegistry.REVERSAL),
            ("TREND_CONTINUATION", "趋势持续", "GLOBAL", PatternTypePatternRegistry.CONTINUATION),
        ]
        
        for pattern_id, display_name, indicator_id, pattern_type in global_patterns:
            self.register(
                pattern_id=pattern_id,
                display_name=display_name,
                indicator_id=indicator_id,
                pattern_type=pattern_type,
                allow_override=True
            )

    def register_indicator_pattern(self, indicator_type: str, pattern_id: str, 
                                 display_name: str, description: str = None,
                                 score_impact: float = 0.0, signal_type: str = None) -> str:
        """
        注册指标形态（兼容性方法）
        
        Args:
            indicator_type: 指标类型
            pattern_id: 形态ID
            display_name: 显示名称
            description: 形态描述
            score_impact: 评分影响
            signal_type: 信号类型
            
        Returns:
            str: 完整的形态ID
        """
        # 推断形态类型
        pattern_type = PatternTypePatternRegistry.NEUTRAL
        if signal_type:
            if 'bullish' in signal_type.lower() or '看涨' in signal_type:
                pattern_type = PatternTypePatternRegistry.BULLISH
            elif 'bearish' in signal_type.lower() or '看跌' in signal_type:
                pattern_type = PatternTypePatternRegistry.BEARISH
        
        # 注册形态
        self.register(
            pattern_id=pattern_id,
            display_name=display_name,
            indicator_id=indicator_type,
            pattern_type=pattern_type,
            description=description or "",
            score_impact=score_impact,
            allow_override=True
        )
        
        return self._normalize_pattern_id(pattern_id, indicator_type)

    def register_patterns_batch(self, patterns: List[PatternInfo], allow_override: bool = False) -> None:
        """
        批量注册形态
        
        Args:
            patterns: 形态信息列表
            allow_override: 是否允许覆盖已存在的形态
        """
        for pattern_info in patterns:
            self.register(
                pattern_id=pattern_info.pattern_id,
                display_name=pattern_info.display_name,
                indicator_id=pattern_info.indicator_id,
                pattern_type=pattern_info.pattern_type,
                default_strength=pattern_info.default_strength,
                description=pattern_info.description,
                score_impact=pattern_info.score_impact,
                detection_function=pattern_info.detection_function,
                polarity=pattern_info.polarity,
                allow_override=allow_override
            )

    def auto_register_from_indicators(self, indicators: List) -> None:
        """
        从指标列表自动注册形态
        
        Args:
            indicators: 指标列表
        """
        for indicator in indicators:
            if hasattr(indicator, 'get_patterns'):
                patterns = indicator.get_patterns()
                for pattern_id, pattern_info in patterns.items():
                    self.register(**pattern_info, allow_override=True)

    def set_allow_override(self, allow: bool) -> None:
        """
        设置是否允许覆盖已注册的形态
        
        Args:
            allow: 是否允许覆盖
        """
        self._allow_override = allow

    def clear_registry(self) -> None:
        """清空注册表"""
        self._patterns.clear()
        self._patterns_by_indicator.clear()
        self._registered_patterns.clear()
        logger.info("形态注册表已清空")

    def get_pattern(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """
        获取形态信息
        
        Args:
            pattern_id: 形态ID
            
        Returns:
            Optional[Dict[str, Any]]: 形态信息字典，如果不存在返回None
        """
        return self._patterns.get(pattern_id.upper())

    def get_patterns_by_indicator(self, indicator_id: str) -> List[str]:
        """
        获取指定指标的所有形态ID
        
        Args:
            indicator_id: 指标ID
            
        Returns:
            List[str]: 形态ID列表
        """
        return self._patterns_by_indicator.get(indicator_id, [])

    def get_pattern_infos_by_indicator(self, indicator_id: str) -> List[Dict[str, Any]]:
        """
        获取指定指标的所有形态信息
        
        Args:
            indicator_id: 指标ID
            
        Returns:
            List[Dict[str, Any]]: 形态信息列表
        """
        pattern_ids = self.get_patterns_by_indicator(indicator_id)
        return [self._patterns[pattern_id] for pattern_id in pattern_ids if pattern_id in self._patterns]

    def get_pattern_info_registry(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """获取形态信息（兼容性方法）"""
        return self.get_pattern(pattern_id)

    def get_display_name(self, pattern_id: str) -> str:
        """获取形态显示名称"""
        pattern = self.get_pattern(pattern_id)
        return pattern.get('display_name', pattern_id) if pattern else pattern_id

    def get_description(self, pattern_id: str) -> Optional[str]:
        """获取形态描述"""
        pattern = self.get_pattern(pattern_id)
        return pattern.get('description') if pattern else None

    def get_score_impact(self, pattern_id: str) -> float:
        """获取形态评分影响"""
        pattern = self.get_pattern(pattern_id)
        return pattern.get('score_impact', 0.0) if pattern else 0.0

    def get_signal_type(self, pattern_id: str) -> Optional[str]:
        """获取信号类型（兼容性方法）"""
        pattern = self.get_pattern(pattern_id)
        if pattern:
            pattern_type = pattern.get('pattern_type')
            if hasattr(pattern_type, 'value'):
                return pattern_type.value
            return str(pattern_type)
        return None

    def get_pattern_by_signal_type(self, signal_type: str) -> List[str]:
        """根据信号类型获取形态"""
        matching_patterns = []
        for pattern_id, pattern_info in self._patterns.items():
            if signal_type.lower() in str(pattern_info.get('pattern_type', '')).lower():
                matching_patterns.append(pattern_id)
        return matching_patterns

    def get_all_pattern_ids(self) -> List[str]:
        """获取所有形态ID"""
        return list(self._patterns.keys())

    def get_all_patterns(self) -> Dict[str, Dict[str, Any]]:
        """获取所有形态信息"""
        return self._patterns.copy()

    def get_patterns_by_polarity(self, polarity: PatternPolarity) -> List[str]:
        """
        根据极性获取形态
        
        Args:
            polarity: 形态极性
            
        Returns:
            List[str]: 匹配的形态ID列表
        """
        matching_patterns = []
        for pattern_id, pattern_info in self._patterns.items():
            if pattern_info.get('polarity') == polarity:
                matching_patterns.append(pattern_id)
        return matching_patterns

    def get_positive_patterns(self) -> List[str]:
        """获取正面形态列表"""
        return self.get_patterns_by_polarity(PatternPolarity.POSITIVE)

    def get_negative_patterns(self) -> List[str]:
        """获取负面形态列表"""
        return self.get_patterns_by_polarity(PatternPolarity.NEGATIVE)

    def get_neutral_patterns(self) -> List[str]:
        """获取中性形态列表"""
        return self.get_patterns_by_polarity(PatternPolarity.NEUTRAL)

    def calculate_combined_score_impact(self, patterns: List[str]) -> float:
        """
        计算多个形态的综合评分影响
        
        Args:
            patterns: 形态ID列表
            
        Returns:
            float: 综合评分影响值
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
            if pattern_id not in self._patterns:
                logger.warning(f"未找到形态: {pattern_id}")
                continue
                
            pattern_info = self._patterns[pattern_id]
            impact = pattern_info.get('score_impact', 0.0)
            
            # 根据形态类型分类评分影响
            if isinstance(pattern_info['pattern_type'], PatternTypePatternRegistry):
                pattern_type = pattern_info['pattern_type']
                if pattern_type == PatternTypePatternRegistry.BULLISH:
                    bullish_impact += impact
                elif pattern_type == PatternTypePatternRegistry.BEARISH:
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

    def import_patterns_from_indicator(self, indicator):
        """
        从指标实例导入形态（已弃用，保留此方法仅用于兼容性）

        Args:
            indicator: 指标实例
        """
        logger.warning(f"import_patterns_from_indicator 方法已弃用，指标 {indicator.name} 的形态现在直接注册到PatternRegistry")
        return
    
    def register_patterns_from_config(self, config_file: str) -> None:
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
            
            for pattern_config in config.get('patterns', []):
                pattern_id = pattern_config.get('id')
                if not pattern_id:
                    logger.warning(f"形态配置缺少ID，跳过注册: {pattern_config}")
                    continue
                
                indicator_id = pattern_config.get('indicator', '')
                display_name = pattern_config.get('name', pattern_id)
                
                # 解析形态类型
                pattern_type_str = pattern_config.get('type', 'neutral')
                pattern_type = PatternTypePatternRegistry.NEUTRAL
                if pattern_type_str.lower() == 'bullish':
                    pattern_type = PatternTypePatternRegistry.BULLISH
                elif pattern_type_str.lower() == 'bearish':
                    pattern_type = PatternTypePatternRegistry.BEARISH
                
                # 解析形态强度
                strength_str = pattern_config.get('strength', 'medium')
                default_strength = PatternStrengthPatternRegistry.MEDIUM
                if strength_str.lower() == 'strong':
                    default_strength = PatternStrengthPatternRegistry.STRONG
                elif strength_str.lower() == 'weak':
                    default_strength = PatternStrengthPatternRegistry.WEAK
                
                # 注册形态
                self.register(
                    pattern_id=pattern_id,
                    display_name=display_name,
                    indicator_id=indicator_id,
                    pattern_type=pattern_type,
                    default_strength=default_strength,
                    allow_override=True
                )
            
            logger.info(f"从配置文件 {config_file} 注册了 {len(config.get('patterns', []))} 个形态")
        except Exception as e:
            logger.error(f"从配置文件注册形态时出错: {e}") 

    def register_indicator_patterns(self, indicator_type: str, patterns: List[Dict[str, Any]]) -> None:
        """
        批量注册指标形态
        
        Args:
            indicator_type: 指标类型
            patterns: 形态列表，每个形态为一个字典，包含id、name等信息
        """
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
            pattern_type = PatternTypePatternRegistry.NEUTRAL
            if pattern_type_str.lower() == 'bullish':
                pattern_type = PatternTypePatternRegistry.BULLISH
            elif pattern_type_str.lower() == 'bearish':
                pattern_type = PatternTypePatternRegistry.BEARISH
            
            # 转换形态强度
            strength_str = pattern_info.get('strength', 'medium')
            default_strength = PatternStrengthPatternRegistry.MEDIUM
            if strength_str.lower() == 'strong':
                default_strength = PatternStrengthPatternRegistry.STRONG
            elif strength_str.lower() == 'weak':
                default_strength = PatternStrengthPatternRegistry.WEAK
            
            # 注册形态
            self.register(
                pattern_id=full_pattern_id,
                display_name=display_name,
                indicator_id=indicator_type,
                pattern_type=pattern_type,
                default_strength=default_strength,
                allow_override=True  # 允许覆盖
            ) 


# ===== 依赖注入和兼容性接口 =====

def get_pattern_registry() -> PatternRegistry:
    """
    获取形态注册表实例（依赖注入方式）
    
    Returns:
        PatternRegistry: 形态注册表实例
    """
    try:
        container = get_container()
        return container.resolve(PatternRegistry())
    except Exception as e:
        logger.warning(f"从依赖注入容器获取PatternRegistry失败，创建新实例: {e}")
        return PatternRegistry()


def get_global_pattern_registry() -> PatternRegistry:
    """
    获取全局形态注册表实例（向后兼容）
    
    Returns:
        PatternRegistry: 全局形态注册表实例
    """
    return get_pattern_registry()


# 注册到依赖注入容器
try:
    container = get_container()
    if not container.is_registered(PatternRegistry):
        container.register_singleton(PatternRegistry, PatternRegistry)
        logger.info("PatternRegistry已注册到依赖注入容器")
except Exception as e:
    logger.warning(f"注册PatternRegistry到依赖注入容器失败: {e}")


# ===== 兼容性别名 =====
# 为了向后兼容，提供下划线命名的别名
PATTERN_REGISTRY = PatternRegistry
PATTERN_TYPE = PatternTypePatternRegistry
PATTERN_STRENGTH = PatternStrengthPatternRegistry
PATTERN_POLARITY = PatternPolarity
PATTERN_INFO = PatternInfo

if __name__ == "__main__":
    print("Pattern Registry Utility")
    print("使用方法: 在代码中导入并使用 PatternRegistry 类来管理技术形态")
    print("现在支持依赖注入：使用 get_pattern_registry() 获取实例")
