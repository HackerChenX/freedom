#!/usr/bin/env python3
"""
模式极性分类工具

用于自动分析和分类技术指标模式的极性，支持买点分析系统的过滤需求。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from enum import Enum

from indicators.complete_indicator_registry import complete_registry
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PolarityClassification:
    """极性分类结果"""
    pattern_id: str
    display_name: str
    current_polarity: PatternPolarity
    suggested_polarity: PatternPolarity
    confidence: float
    reasoning: str
    needs_review: bool = False


class PolarityClassifier:
    """模式极性分类器"""
    
    def __init__(self):
        self.registry = PatternRegistry()
        
        # 负面关键词（看跌/不适合买点）
        self.negative_keywords = {
            # 中文关键词
            '空头', '下行', '死叉', '下降', '负值', '超卖', '弱', '低于', '看跌',
            '下跌', '回调', '深度', '短期下降', '无', '极低', '严重', '虚弱',
            '耗尽', '阻力', '压制', '破位', '跌破', '失守', '恶化', '疲软',
            '衰竭', '反转向下', '顶部', '高位', '过热', '泡沫', '风险',
            
            # 英文关键词
            'falling', 'bearish', 'below', 'negative', 'weak', 'down',
            'decline', 'drop', 'sell', 'short', 'resistance', 'break_down',
            'oversold', 'exhaustion', 'reversal_down', 'top', 'high', 'risk'
        }
        
        # 正面关键词（看涨/适合买点）
        self.positive_keywords = {
            # 中文关键词
            '多头', '上行', '金叉', '上升', '正值', '超买', '强', '高于', '看涨',
            '上涨', '突破', '买点', '信号', '满足', '极高', '强烈', '支撑',
            '反弹', '回升', '企稳', '站上', '向上', '加速', '放量', '活跃',
            '底部', '低位', '机会', '确认', '建仓', '增持',
            
            # 英文关键词
            'rising', 'bullish', 'above', 'positive', 'strong', 'up',
            'increase', 'buy', 'long', 'support', 'breakout', 'golden_cross',
            'oversold_bounce', 'bottom', 'low', 'opportunity', 'signal'
        }
        
        # 中性关键词（信息性质）
        self.neutral_keywords = {
            '正常', '中等', '平稳', '稳定', '震荡', '横盘', '整理', '等待',
            '观望', '中位', '均衡', '一般', '常规', '标准', '基础',
            'normal', 'medium', 'stable', 'sideways', 'consolidation',
            'neutral', 'average', 'standard', 'baseline'
        }
    
    def classify_pattern(self, pattern_id: str) -> PolarityClassification:
        """
        分类单个模式的极性
        
        Args:
            pattern_id: 模式ID
            
        Returns:
            PolarityClassification: 分类结果
        """
        pattern_info = self.registry.get_pattern(pattern_id)
        if not pattern_info:
            return PolarityClassification(
                pattern_id=pattern_id,
                display_name="未知模式",
                current_polarity=PatternPolarity.NEUTRAL,
                suggested_polarity=PatternPolarity.NEUTRAL,
                confidence=0.0,
                reasoning="模式不存在",
                needs_review=True
            )
        
        display_name = pattern_info.get('display_name', '')
        pattern_type = pattern_info.get('pattern_type')
        score_impact = pattern_info.get('score_impact', 0.0)
        current_polarity = pattern_info.get('polarity', PatternPolarity.NEUTRAL)
        
        # 分析建议极性
        suggested_polarity, confidence, reasoning = self._analyze_polarity(
            display_name, pattern_type, score_impact
        )
        
        # 判断是否需要人工审查
        needs_review = (
            confidence < 0.7 or  # 置信度低
            current_polarity != suggested_polarity or  # 当前分类与建议不符
            self._contains_conflicting_keywords(display_name)  # 包含冲突关键词
        )
        
        return PolarityClassification(
            pattern_id=pattern_id,
            display_name=display_name,
            current_polarity=current_polarity,
            suggested_polarity=suggested_polarity,
            confidence=confidence,
            reasoning=reasoning,
            needs_review=needs_review
        )
    
    def _analyze_polarity(self, display_name: str, pattern_type: PatternType, 
                         score_impact: float) -> Tuple[PatternPolarity, float, str]:
        """
        分析模式极性
        
        Returns:
            Tuple[PatternPolarity, float, str]: (建议极性, 置信度, 推理过程)
        """
        reasoning_parts = []
        confidence_scores = []
        
        # 1. 基于关键词分析
        keyword_polarity, keyword_confidence, keyword_reasoning = self._analyze_keywords(display_name)
        if keyword_confidence > 0:
            reasoning_parts.append(keyword_reasoning)
            confidence_scores.append(keyword_confidence)
        
        # 2. 基于形态类型分析
        type_polarity, type_confidence, type_reasoning = self._analyze_pattern_type(pattern_type)
        if type_confidence > 0:
            reasoning_parts.append(type_reasoning)
            confidence_scores.append(type_confidence * 0.8)  # 类型权重稍低
        
        # 3. 基于评分影响分析
        score_polarity, score_confidence, score_reasoning = self._analyze_score_impact(score_impact)
        if score_confidence > 0:
            reasoning_parts.append(score_reasoning)
            confidence_scores.append(score_confidence * 0.6)  # 评分权重最低
        
        # 综合判断
        if not confidence_scores:
            return PatternPolarity.NEUTRAL, 0.0, "无法确定极性"
        
        # 选择最高置信度的判断
        max_confidence_idx = confidence_scores.index(max(confidence_scores))
        polarities = [keyword_polarity, type_polarity, score_polarity]
        
        final_polarity = polarities[max_confidence_idx]
        final_confidence = max(confidence_scores)
        final_reasoning = " | ".join(reasoning_parts)
        
        return final_polarity, final_confidence, final_reasoning
    
    def _analyze_keywords(self, display_name: str) -> Tuple[PatternPolarity, float, str]:
        """基于关键词分析极性"""
        if not display_name:
            return PatternPolarity.NEUTRAL, 0.0, ""
        
        name_lower = display_name.lower()
        
        # 统计各类关键词出现次数
        negative_count = sum(1 for keyword in self.negative_keywords if keyword in name_lower)
        positive_count = sum(1 for keyword in self.positive_keywords if keyword in name_lower)
        neutral_count = sum(1 for keyword in self.neutral_keywords if keyword in name_lower)
        
        total_keywords = negative_count + positive_count + neutral_count
        
        if total_keywords == 0:
            return PatternPolarity.NEUTRAL, 0.0, ""
        
        # 计算置信度和极性
        if negative_count > positive_count and negative_count > neutral_count:
            confidence = min(0.9, 0.5 + negative_count * 0.2)
            return PatternPolarity.NEGATIVE, confidence, f"包含{negative_count}个负面关键词"
        elif positive_count > negative_count and positive_count > neutral_count:
            confidence = min(0.9, 0.5 + positive_count * 0.2)
            return PatternPolarity.POSITIVE, confidence, f"包含{positive_count}个正面关键词"
        else:
            confidence = min(0.7, 0.3 + neutral_count * 0.2)
            return PatternPolarity.NEUTRAL, confidence, f"包含{neutral_count}个中性关键词"
    
    def _analyze_pattern_type(self, pattern_type: PatternType) -> Tuple[PatternPolarity, float, str]:
        """基于形态类型分析极性"""
        if not pattern_type:
            return PatternPolarity.NEUTRAL, 0.0, ""
        
        if pattern_type == PatternType.BEARISH:
            return PatternPolarity.NEGATIVE, 0.8, "形态类型为看跌"
        elif pattern_type == PatternType.BULLISH:
            return PatternPolarity.POSITIVE, 0.8, "形态类型为看涨"
        else:
            return PatternPolarity.NEUTRAL, 0.5, f"形态类型为{pattern_type.value}"
    
    def _analyze_score_impact(self, score_impact: float) -> Tuple[PatternPolarity, float, str]:
        """基于评分影响分析极性"""
        if score_impact < -10:
            return PatternPolarity.NEGATIVE, 0.7, f"评分影响为{score_impact}(强负面)"
        elif score_impact < -5:
            return PatternPolarity.NEGATIVE, 0.6, f"评分影响为{score_impact}(负面)"
        elif score_impact > 10:
            return PatternPolarity.POSITIVE, 0.7, f"评分影响为{score_impact}(强正面)"
        elif score_impact > 5:
            return PatternPolarity.POSITIVE, 0.6, f"评分影响为{score_impact}(正面)"
        else:
            return PatternPolarity.NEUTRAL, 0.4, f"评分影响为{score_impact}(中性)"
    
    def _contains_conflicting_keywords(self, display_name: str) -> bool:
        """检查是否包含冲突的关键词"""
        if not display_name:
            return False
        
        name_lower = display_name.lower()
        
        has_negative = any(keyword in name_lower for keyword in self.negative_keywords)
        has_positive = any(keyword in name_lower for keyword in self.positive_keywords)
        
        return has_negative and has_positive
    
    def classify_all_patterns(self) -> List[PolarityClassification]:
        """分类所有模式"""
        all_patterns = self.registry.get_all_pattern_ids()
        return [self.classify_pattern(pattern_id) for pattern_id in all_patterns]
    
    def get_problematic_patterns(self) -> List[PolarityClassification]:
        """获取有问题的模式（需要审查的）"""
        all_classifications = self.classify_all_patterns()
        return [c for c in all_classifications if c.needs_review]


def main():
    """主函数"""
    print("🔍 模式极性分类工具")
    print("=" * 50)
    
    classifier = PolarityClassifier()
    
    # 分类所有模式
    print("正在分析所有模式...")
    all_classifications = classifier.classify_all_patterns()
    
    # 统计结果
    total = len(all_classifications)
    needs_review = len([c for c in all_classifications if c.needs_review])
    positive = len([c for c in all_classifications if c.suggested_polarity == PatternPolarity.POSITIVE])
    negative = len([c for c in all_classifications if c.suggested_polarity == PatternPolarity.NEGATIVE])
    neutral = len([c for c in all_classifications if c.suggested_polarity == PatternPolarity.NEUTRAL])
    
    print(f"\n📊 分析结果统计:")
    print(f"总模式数量: {total}")
    print(f"建议正面极性: {positive}")
    print(f"建议负面极性: {negative}")
    print(f"建议中性极性: {neutral}")
    print(f"需要人工审查: {needs_review}")
    
    # 显示需要审查的模式
    if needs_review > 0:
        print(f"\n⚠️  需要审查的模式 (前10个):")
        problematic = classifier.get_problematic_patterns()[:10]
        for i, classification in enumerate(problematic, 1):
            print(f"{i:2d}. {classification.pattern_id}")
            print(f"    显示名称: {classification.display_name}")
            print(f"    当前极性: {classification.current_polarity.value}")
            print(f"    建议极性: {classification.suggested_polarity.value}")
            print(f"    置信度: {classification.confidence:.2f}")
            print(f"    推理: {classification.reasoning}")
            print()


if __name__ == "__main__":
    main()
