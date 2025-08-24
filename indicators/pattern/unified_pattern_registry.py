#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一形态注册表 - 解决形态命名不一致问题

这个模块建立了统一的形态管理规范，确保：
1. 每个形态只在一个地方定义
2. 所有组件使用统一的形态名称
3. 形态名称映射关系清晰
4. 避免重复定义和不一致问题
"""

from typing import Dict, List, Set, Optional, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class PatternType(Enum):
    """形态类型枚举"""
    BULLISH = "BULLISH"          # 看涨形态
    BEARISH = "BEARISH"          # 看跌形态
    NEUTRAL = "NEUTRAL"          # 中性形态
    REVERSAL = "REVERSAL"        # 反转形态
    CONTINUATION = "CONTINUATION" # 持续形态
    DIVERGENCE = "DIVERGENCE"    # 背离形态

class PatternCategory(Enum):
    """形态分类枚举"""
    CROSS = "CROSS"              # 交叉类形态
    BREAKOUT = "BREAKOUT"        # 突破类形态
    OSCILLATOR = "OSCILLATOR"    # 振荡器形态
    TREND = "TREND"              # 趋势形态
    VOLUME = "VOLUME"            # 成交量形态
    CANDLESTICK = "CANDLESTICK"  # K线形态
    DIVERGENCE = "DIVERGENCE"    # 背离形态

class UnifiedPatternRegistry:
    """统一形态注册表"""
    
    def __init__(self):
        """初始化统一形态注册表"""
        self._patterns: Dict[str, Dict[str, Any]] = {}
        self._indicator_patterns: Dict[str, Set[str]] = {}
        self._pattern_aliases: Dict[str, str] = {}
        self._initialized = False
        
    def initialize(self):
        """初始化所有标准形态"""
        if self._initialized:
            return
            
        logger.info("🚀 初始化统一形态注册表")
        
        # 注册标准形态
        self._register_standard_patterns()
        
        # 注册指标特定形态
        self._register_indicator_specific_patterns()
        
        # 注册形态别名映射
        self._register_pattern_aliases()
        
        self._initialized = True
        logger.info(f"✅ 统一形态注册表初始化完成，共注册 {len(self._patterns)} 个形态")
    
    def _register_standard_patterns(self):
        """注册标准通用形态"""
        
        # 交叉类形态
        self._register_pattern(
            pattern_id="GOLDEN_CROSS",
            display_name="金叉",
            description="快线上穿慢线，看涨信号",
            pattern_type=PatternType.BULLISH,
            category=PatternCategory.CROSS,
            indicators=["MACD", "MA", "EMA", "WMA", "DMA", "KDJ", "RSI", "STOCHRSI", "DMI", "CCI", "BIAS"]
        )
        
        self._register_pattern(
            pattern_id="DEATH_CROSS",
            display_name="死叉",
            description="快线下穿慢线，看跌信号",
            pattern_type=PatternType.BEARISH,
            category=PatternCategory.CROSS,
            indicators=["MACD", "MA", "EMA", "WMA", "DMA", "KDJ", "RSI", "STOCHRSI", "DMI", "CCI", "BIAS"]
        )
        
        # 背离类形态
        self._register_pattern(
            pattern_id="BULLISH_DIVERGENCE",
            display_name="牛市背离",
            description="价格创新低，指标不创新低",
            pattern_type=PatternType.DIVERGENCE,
            category=PatternCategory.OSCILLATOR,
            indicators=["MACD", "RSI", "KDJ", "CCI", "MFI", "STOCHRSI"]
        )
        
        self._register_pattern(
            pattern_id="BEARISH_DIVERGENCE",
            display_name="熊市背离",
            description="价格创新高，指标不创新高",
            pattern_type=PatternType.DIVERGENCE,
            category=PatternCategory.OSCILLATOR,
            indicators=["MACD", "RSI", "KDJ", "CCI", "MFI", "STOCHRSI"]
        )
        
        # 突破类形态
        self._register_pattern(
            pattern_id="UPPER_BREAKOUT",
            display_name="上轨突破",
            description="价格突破上轨阻力",
            pattern_type=PatternType.BULLISH,
            category=PatternCategory.BREAKOUT,
            indicators=["BOLL", "KC", "DONCHIAN"]
        )
        
        self._register_pattern(
            pattern_id="LOWER_BREAKOUT",
            display_name="下轨突破",
            description="价格跌破下轨支撑",
            pattern_type=PatternType.BEARISH,
            category=PatternCategory.BREAKOUT,
            indicators=["BOLL", "KC", "DONCHIAN"]
        )
        
        # 振荡器形态
        self._register_pattern(
            pattern_id="OVERBOUGHT",
            display_name="超买",
            description="指标进入超买区域",
            pattern_type=PatternType.BEARISH,
            category=PatternCategory.OSCILLATOR,
            indicators=["RSI", "KDJ", "CCI", "WR", "STOCHRSI"]
        )
        
        self._register_pattern(
            pattern_id="OVERSOLD",
            display_name="超卖",
            description="指标进入超卖区域",
            pattern_type=PatternType.BULLISH,
            category=PatternCategory.OSCILLATOR,
            indicators=["RSI", "KDJ", "CCI", "WR", "STOCHRSI"]
        )
    
    def _register_indicator_specific_patterns(self):
        """注册指标特定形态"""
        
        # MACD特定形态 (基于实际MACD指标支持的形态)
        self._register_pattern(
            pattern_id="MACD_ABOVE_ZERO_GOLDEN",
            display_name="MACD零轴上方金叉",
            description="MACD线在零轴上方形成金叉",
            pattern_type=PatternType.BULLISH,
            category=PatternCategory.CROSS,
            indicators=["MACD"]
        )
        
        # BOLL特定形态
        self._register_pattern(
            pattern_id="BOLL_SQUEEZE",
            display_name="布林带收缩",
            description="布林带上下轨距离缩小，波动性降低",
            pattern_type=PatternType.NEUTRAL,
            category=PatternCategory.TREND,
            indicators=["BOLL"]
        )
        
        # KDJ特定形态
        self._register_pattern(
            pattern_id="KDJ_TOP_DIVERGENCE",
            display_name="KDJ顶背离",
            description="价格创新高，KDJ不创新高",
            pattern_type=PatternType.BEARISH,
            category=PatternCategory.DIVERGENCE,
            indicators=["KDJ"]
        )
        
        # 成交量形态
        self._register_pattern(
            pattern_id="VOLUME_BREAKOUT",
            display_name="放量突破",
            description="成交量放大伴随价格突破",
            pattern_type=PatternType.BULLISH,
            category=PatternCategory.VOLUME,
            indicators=["OBV", "MFI", "VR", "VOSC"]
        )
    
    def _register_pattern_aliases(self):
        """注册形态别名映射"""
        
        # 处理历史遗留的形态名称
        aliases = {
            # MACD相关别名
            "MACD_GOLDEN_CROSS": "GOLDEN_CROSS",
            "MACD_DEATH_CROSS": "DEATH_CROSS",
            "MACD_HISTOGRAM_DIVERGENCE": "BEARISH_DIVERGENCE",
            "DIVERGENCE": "BEARISH_DIVERGENCE",
            
            # 通用别名
            "BULLISH_SIGNAL": "GOLDEN_CROSS",
            "BEARISH_SIGNAL": "DEATH_CROSS",
            "TREND_REVERSAL": "BULLISH_DIVERGENCE",
            
            # 布林带别名
            "UPPER_BAND_BREAKOUT": "UPPER_BREAKOUT",
            "LOWER_BAND_BREAKOUT": "LOWER_BREAKOUT",
            
            # 振荡器别名
            "RSI_OVERBOUGHT": "OVERBOUGHT",
            "RSI_OVERSOLD": "OVERSOLD",
            "KDJ_OVERBOUGHT": "OVERBOUGHT",
            "KDJ_OVERSOLD": "OVERSOLD",
        }
        
        for alias, canonical in aliases.items():
            self._pattern_aliases[alias] = canonical
    
    def _register_pattern(self, pattern_id: str, display_name: str, description: str,
                         pattern_type: PatternType, category: PatternCategory,
                         indicators: List[str], **kwargs):
        """注册单个形态"""
        
        pattern_info = {
            'pattern_id': pattern_id,
            'display_name': display_name,
            'description': description,
            'pattern_type': pattern_type.value,
            'category': category.value,
            'indicators': indicators,
            **kwargs
        }
        
        self._patterns[pattern_id] = pattern_info
        
        # 更新指标-形态映射
        for indicator in indicators:
            if indicator not in self._indicator_patterns:
                self._indicator_patterns[indicator] = set()
            self._indicator_patterns[indicator].add(pattern_id)
    
    def get_canonical_pattern_name(self, pattern_name: str) -> str:
        """获取规范的形态名称"""
        if not self._initialized:
            self.initialize()
            
        # 如果是别名，返回规范名称
        if pattern_name in self._pattern_aliases:
            return self._pattern_aliases[pattern_name]
        
        # 如果是规范名称，直接返回
        if pattern_name in self._patterns:
            return pattern_name
        
        # 尝试模糊匹配
        for canonical_name in self._patterns.keys():
            if pattern_name.upper() in canonical_name.upper() or canonical_name.upper() in pattern_name.upper():
                return canonical_name
        
        # 如果找不到，记录警告并返回原名称
        logger.warning(f"⚠️ 未找到形态 '{pattern_name}' 的规范名称")
        return pattern_name
    
    def get_pattern_info(self, pattern_name: str) -> Optional[Dict[str, Any]]:
        """获取形态信息"""
        if not self._initialized:
            self.initialize()
            
        canonical_name = self.get_canonical_pattern_name(pattern_name)
        return self._patterns.get(canonical_name)
    
    def get_indicator_patterns(self, indicator_name: str) -> List[str]:
        """获取指标支持的所有形态"""
        if not self._initialized:
            self.initialize()

        patterns = list(self._indicator_patterns.get(indicator_name, set()))

        # 特殊处理：MACD指标只返回实际支持的形态
        if indicator_name == "MACD":
            # 基于MACD指标实际实现，只返回确实支持的形态
            actual_macd_patterns = [
                "GOLDEN_CROSS",
                "DEATH_CROSS",
                "MACD_ABOVE_ZERO_GOLDEN",
                "BEARISH_DIVERGENCE"
            ]
            # 只返回在注册表中且实际支持的形态
            patterns = [p for p in patterns if p in actual_macd_patterns]

        return patterns
    
    def get_all_patterns(self) -> Dict[str, Dict[str, Any]]:
        """获取所有形态"""
        if not self._initialized:
            self.initialize()
            
        return self._patterns.copy()
    
    def validate_pattern_for_indicator(self, indicator_name: str, pattern_name: str) -> bool:
        """验证形态是否适用于指标"""
        if not self._initialized:
            self.initialize()
            
        canonical_name = self.get_canonical_pattern_name(pattern_name)
        pattern_info = self._patterns.get(canonical_name)
        
        if not pattern_info:
            return False
            
        return indicator_name in pattern_info.get('indicators', [])
    
    def get_pattern_mapping_for_data_generator(self) -> Dict[str, str]:
        """获取数据生成器使用的形态映射"""
        if not self._initialized:
            self.initialize()
            
        # 返回从指标特定形态名称到数据生成器期望名称的映射
        mapping = {}
        
        for pattern_id, pattern_info in self._patterns.items():
            # 数据生成器使用的简化名称
            if pattern_info['category'] == PatternCategory.CROSS.value:
                if pattern_info['pattern_type'] == PatternType.BULLISH.value:
                    mapping[pattern_id] = "GOLDEN_CROSS"
                elif pattern_info['pattern_type'] == PatternType.BEARISH.value:
                    mapping[pattern_id] = "DEATH_CROSS"
            elif pattern_info['category'] == PatternCategory.DIVERGENCE.value:
                if pattern_info['pattern_type'] == PatternType.BEARISH.value:
                    mapping[pattern_id] = "MACD_BEARISH_DIVERGENCE"
                elif pattern_info['pattern_type'] == PatternType.BULLISH.value:
                    mapping[pattern_id] = "MACD_BULLISH_DIVERGENCE"
                else:
                    mapping[pattern_id] = "DIVERGENCE"
            elif pattern_info['category'] == PatternCategory.BREAKOUT.value:
                if "UPPER" in pattern_id:
                    mapping[pattern_id] = "UPPER_BREAKOUT"
                elif "LOWER" in pattern_id:
                    mapping[pattern_id] = "LOWER_BREAKOUT"
            elif pattern_info['category'] == PatternCategory.OSCILLATOR.value:
                if "OVERBOUGHT" in pattern_id:
                    mapping[pattern_id] = "OVERBOUGHT"
                elif "OVERSOLD" in pattern_id:
                    mapping[pattern_id] = "OVERSOLD"
                elif "BEARISH_DIVERGENCE" in pattern_id:
                    mapping[pattern_id] = "DIVERGENCE"  # 使用通用背离形态
                elif "BULLISH_DIVERGENCE" in pattern_id:
                    mapping[pattern_id] = "DIVERGENCE"  # 使用通用背离形态
            else:
                # 默认使用形态ID
                mapping[pattern_id] = pattern_id
        
        return mapping

# 全局单例实例
_unified_pattern_registry = None

def get_unified_pattern_registry() -> UnifiedPatternRegistry:
    """获取统一形态注册表单例"""
    global _unified_pattern_registry
    if _unified_pattern_registry is None:
        _unified_pattern_registry = UnifiedPatternRegistry()
        _unified_pattern_registry.initialize()
    return _unified_pattern_registry

def get_canonical_pattern_name(pattern_name: str) -> str:
    """获取规范的形态名称（便捷函数）"""
    return get_unified_pattern_registry().get_canonical_pattern_name(pattern_name)

def get_pattern_info(pattern_name: str) -> Optional[Dict[str, Any]]:
    """获取形态信息（便捷函数）"""
    return get_unified_pattern_registry().get_pattern_info(pattern_name)

def get_indicator_patterns(indicator_name: str) -> List[str]:
    """获取指标支持的所有形态（便捷函数）"""
    return get_unified_pattern_registry().get_indicator_patterns(indicator_name)

def validate_pattern_for_indicator(indicator_name: str, pattern_name: str) -> bool:
    """验证形态是否适用于指标（便捷函数）"""
    return get_unified_pattern_registry().validate_pattern_for_indicator(indicator_name, pattern_name)
