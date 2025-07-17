#!/usr/bin/env python3
"""
分析买点分析报告中的模式极性

直接从报告文件中提取模式并分析其极性，识别不适合买点分析的负面模式。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
from typing import List, Dict, Tuple
from dataclasses import dataclass

from indicators.complete_indicator_registry import complete_registry
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Report_pattern:
    """报告中的模式"""
    indicator_name: str
    pattern_name: str
    hit_ratio: float
    hit_count: int
    avg_score: float
    period: str
    line_number: int


class Report_pattern_analyzer:
    """报告模式分析器"""
    
    def __init__(self, report_path: str):
        self.report_path = report_path
        
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
    
    def parse_report(self) -> List[Report_pattern]:
        """解析报告文件，提取所有模式"""
        patterns = []
        current_period = ""
        
        try:
            with open(self.report_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except File_not_found_error:
            logger.error(f"报告文件不存在: {self.report_path}")
            return []
        
        # 正则表达式匹配表格行
        table_pattern = re.compile(r'\|\s*indicator\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([\d.]+)%\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|')
        period_pattern = re.compile(r'##\s*📈\s*(\w+)\s*周期共性指标')
        
        for line_num, line in enumerate(lines, 1):
            # 检测周期
            period_match = period_pattern.search(line)
            if period_match:
                current_period = period_match.group(1)
                continue
            
            # 匹配表格行
            match = table_pattern.search(line)
            if match:
                indicator_name = match.group(1).strip()
                pattern_name = match.group(2).strip()
                hit_ratio = float(match.group(3))
                hit_count = int(match.group(4))
                avg_score = float(match.group(5))
                
                patterns.append(Report_pattern(
                    indicator_name=indicator_name,
                    pattern_name=pattern_name,
                    hit_ratio=hit_ratio,
                    hit_count=hit_count,
                    avg_score=avg_score,
                    period=current_period,
                    line_number=line_num
                ))
        
        return patterns
    
    def classify_pattern_polarity(self, pattern: Report_pattern) -> Tuple[Pattern_polarity, float, str]:
        """
        分类模式极性
        
        Returns:
            Tuple[Pattern_polarity, float, str]: (极性, 置信度, 推理)
        """
        text = f"{pattern.indicator_name} {pattern.pattern_name}".lower()
        
        # 统计关键词
        negative_count = sum(1 for keyword in self.negative_keywords if keyword in text)
        positive_count = sum(1 for keyword in self.positive_keywords if keyword in text)
        
        reasoning_parts = []
        
        # 基于关键词判断
        if negative_count > positive_count:
            confidence = min(0.9, 0.6 + negative_count * 0.15)
            reasoning_parts.append(f"包含{negative_count}个负面关键词")
            polarity = Pattern_polarity.NEGATIVE
        elif positive_count > negative_count:
            confidence = min(0.9, 0.6 + positive_count * 0.15)
            reasoning_parts.append(f"包含{positive_count}个正面关键词")
            polarity = Pattern_polarity.POSITIVE
        else:
            confidence = 0.3
            reasoning_parts.append("无明显极性关键词")
            polarity = Pattern_polarity.NEUTRAL
        
        # 特殊规则调整
        if '无' in pattern.pattern_name and '信号' in pattern.pattern_name:
            polarity = Pattern_polarity.NEGATIVE
            confidence = max(confidence, 0.8)
            reasoning_parts.append("'无...信号'模式为负面")
        
        if '买点' in pattern.pattern_name:
            polarity = Pattern_polarity.POSITIVE
            confidence = max(confidence, 0.8)
            reasoning_parts.append("包含'买点'为正面")
        
        reasoning = " | ".join(reasoning_parts)
        return polarity, confidence, reasoning
    
    def analyze_patterns(self) -> Dict[str, List[Report_pattern]]:
        """分析所有模式并按极性分组"""
        patterns = self.parse_report()
        
        result = {
            'negative': [],
            'positive': [],
            'neutral': [],
            'high_confidence_negative': []
        }
        
        for pattern in patterns:
            polarity, confidence, reasoning = self.classify_pattern_polarity(pattern)
            
            # 添加分析结果到模式对象
            pattern.polarity = polarity
            pattern.confidence = confidence
            pattern.reasoning = reasoning
            
            # 分组
            if polarity == Pattern_polarity.NEGATIVE:
                result['negative'].append(pattern)
                if confidence >= 0.7:
                    result['high_confidence_negative'].append(pattern)
            elif polarity == Pattern_polarity.POSITIVE:
                result['positive'].append(pattern)
            else:
                result['neutral'].append(pattern)
        
        return result
    
    def generate_report(self) -> str:
        """生成分析报告"""
        analysis = self.analyze_patterns()
        
        report = []
        report.append("# 买点分析报告模式极性分析")
        report.append("=" * 50)
        report.append("")
        
        # 统计信息
        total = sum(len(patterns) for patterns in analysis.values() if isinstance(patterns, list))
        negative_count = len(analysis['negative'])
        positive_count = len(analysis['positive'])
        neutral_count = len(analysis['neutral'])
        high_conf_negative = len(analysis['high_confidence_negative'])
        
        report.append(f"## 📊 统计信息")
        report.append(f"- 总模式数量: {total}")
        report.append(f"- 负面模式: {negative_count} ({negative_count/total*100:.1f}%)")
        report.append(f"- 正面模式: {positive_count} ({positive_count/total*100:.1f}%)")
        report.append(f"- 中性模式: {neutral_count} ({neutral_count/total*100:.1f}%)")
        report.append(f"- 高置信度负面模式: {high_conf_negative}")
        report.append("")
        
        # 高置信度负面模式（需要从买点分析中排除）
        if analysis['high_confidence_negative']:
            report.append("## ⚠️ 高置信度负面模式（应从买点分析中排除）")
            report.append("")
            
            # 按命中率排序
            sorted_negative = sorted(analysis['high_confidence_negative'], 
                                   key=lambda x: x.hit_ratio, reverse=True)
            
            for i, pattern in enumerate(sorted_negative[:20], 1):  # 显示前20个
                report.append(f"### {i}. {pattern.indicator_name} - {pattern.pattern_name}")
                report.append(f"- **命中率**: {pattern.hit_ratio}%")
                report.append(f"- **命中数量**: {pattern.hit_count}")
                report.append(f"- **周期**: {pattern.period}")
                report.append(f"- **置信度**: {pattern.confidence:.2f}")
                report.append(f"- **推理**: {pattern.reasoning}")
                report.append(f"- **行号**: {pattern.line_number}")
                report.append("")
        
        # 所有负面模式列表
        if analysis['negative']:
            report.append("## 📋 所有负面模式列表")
            report.append("")
            report.append("| 指标名称 | 模式名称 | 命中率 | 置信度 | 周期 | 推理 |")
            report.append("|---------|----------|--------|--------|------|------|")
            
            sorted_all_negative = sorted(analysis['negative'], 
                                       key=lambda x: (x.hit_ratio, x.confidence), reverse=True)
            
            for pattern in sorted_all_negative:
                report.append(f"| {pattern.indicator_name} | {pattern.pattern_name} | "
                            f"{pattern.hit_ratio}% | {pattern.confidence:.2f} | "
                            f"{pattern.period} | {pattern.reasoning} |")
        
        return "\n".join(report)


def main_analyzereportpatterns():
    """主函数"""
    report_path = "results/analysis/common_indicators_report.md"
    
    if not os.path.exists(report_path):
        print(f"❌ 报告文件不存在: {report_path}")
        return
    
    print("🔍 分析买点分析报告中的模式极性...")
    
    analyzer = Report_pattern_analyzer(report_path)
    analysis_report = analyzer.generate_report()
    
    # 保存分析报告
    output_path = "results/analysis/pattern_polarity_analysis.md"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(analysis_report)
    
    print(f"✅ 分析完成，报告已保存到: {output_path}")
    
    # 显示摘要
    analyzer_obj = Report_pattern_analyzer(report_path)
    analysis = analyzer_obj.analyze_patterns()
    
    total = sum(len(patterns) for patterns in analysis.values() if isinstance(patterns, list))
    negative_count = len(analysis['negative'])
    high_conf_negative = len(analysis['high_confidence_negative'])
    
    print(f"\n📊 分析摘要:")
    print(f"总模式数量: {total}")
    print(f"负面模式数量: {negative_count}")
    print(f"高置信度负面模式: {high_conf_negative}")
    
    if high_conf_negative > 0:
        print(f"\n⚠️  发现 {high_conf_negative} 个高置信度负面模式需要从买点分析中排除！")


if __name__ == "__main__":
    main_analyzereportpatterns()
