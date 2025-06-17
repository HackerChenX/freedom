#!/usr/bin/env python3
"""
极性标注验证工具

验证技术指标模式的极性标注是否正确和一致，确保买点分析系统的可靠性。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

from indicators.pattern_registry import PatternRegistry, PatternPolarity
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """验证结果"""
    pattern_id: str
    indicator_name: str
    display_name: str
    polarity: PatternPolarity
    has_polarity: bool
    is_consistent: bool
    issues: List[str]


class PolarityValidator:
    """极性标注验证器"""
    
    def __init__(self):
        self.registry = PatternRegistry()
        
        # 初始化所有指标以注册模式
        self._initialize_indicators()
    
    def _initialize_indicators(self):
        """初始化所有指标以确保模式被注册"""
        indicators = []

        # 定义要初始化的指标列表 - 扩大范围包含更多指标
        indicator_configs = [
            # 基础技术指标
            ('indicators.macd', 'MACD'),
            ('indicators.ma', 'MA'),
            ('indicators.rsi', 'RSI'),
            ('indicators.kdj', 'KDJ'),
            ('indicators.boll', 'BOLL'),
            ('indicators.vol', 'VOL'),
            ('indicators.chaikin', 'Chaikin'),
            ('indicators.dmi', 'DMI'),
            ('indicators.emv', 'EMV'),
            ('indicators.obv', 'OBV'),
            ('indicators.cci', 'CCI'),
            ('indicators.wr', 'WR'),
            ('indicators.ichimoku', 'Ichimoku'),
            ('indicators.bias', 'BIAS'),
            ('indicators.sar', 'SAR'),
            ('indicators.mfi', 'MFI'),
            ('indicators.atr', 'ATR'),
            ('indicators.adx', 'ADX'),
            ('indicators.aroon', 'AROON'),
            ('indicators.momentum', 'Momentum'),
            ('indicators.mtm', 'MTM'),
            ('indicators.psy', 'PSY'),
            ('indicators.pvt', 'PVT'),
            ('indicators.roc', 'ROC'),
            ('indicators.trix', 'TRIX'),
            ('indicators.vix', 'VIX'),
            ('indicators.volume_ratio', 'VolumeRatio'),
            ('indicators.vosc', 'VOSC'),
            ('indicators.vr', 'VR'),
            ('indicators.vortex', 'VORTEX'),
            ('indicators.ad', 'AD'),
            ('indicators.kc', 'KC'),
            ('indicators.ema', 'EMA'),
            ('indicators.wma', 'WMA'),
            ('indicators.dma', 'DMA'),
            ('indicators.cmo', 'CMO'),

            # 增强指标
            ('indicators.trend.enhanced_macd', 'EnhancedMACD'),
            ('indicators.volume.enhanced_mfi', 'EnhancedMFI'),
            ('indicators.volume.enhanced_obv', 'EnhancedOBV'),
            ('indicators.trend.enhanced_trix', 'EnhancedTRIX'),
            ('indicators.trend.enhanced_dmi', 'EnhancedDMI'),
            ('indicators.trend.enhanced_cci', 'EnhancedCCI'),

            # K线形态指标
            ('indicators.pattern.candlestick_patterns', 'CandlestickPatterns'),
            ('indicators.pattern.advanced_candlestick_patterns', 'AdvancedCandlestickPatterns'),

            # 专业指标
            ('indicators.fibonacci_tools', 'FibonacciTools'),
            ('indicators.elliott_wave', 'ElliottWave'),
            ('indicators.gann_tools', 'GannTools'),
            ('indicators.chip_distribution', 'ChipDistribution'),

            # 复合指标
            ('indicators.composite.composite_indicator', 'CompositeIndicator'),
            ('indicators.composite.unified_ma', 'UnifiedMA'),
            ('indicators.composite.chip_distribution', 'ChipDistribution'),
            ('indicators.composite.institutional_behavior', 'InstitutionalBehavior'),
            ('indicators.composite.stock_vix', 'StockVIX'),

            # ZXM体系指标
            ('indicators.zxm_absorb', 'ZXMAbsorb'),
            ('indicators.zxm_washplate', 'ZXMWashplate'),
            ('indicators.pattern.zxm_patterns', 'ZXMPatterns'),
        ]

        # 尝试导入和初始化每个指标
        for module_name, class_name in indicator_configs:
            try:
                module = __import__(module_name, fromlist=[class_name])
                indicator_class = getattr(module, class_name)

                # 特殊处理某些指标的初始化参数
                if class_name == 'MA':
                    indicator = indicator_class(periods=[5, 10, 20])
                elif class_name == 'UnifiedMA':
                    indicator = indicator_class(periods=[5, 10, 20, 60])
                elif class_name == 'CompositeIndicator':
                    # 复合指标需要子指标列表
                    indicator = indicator_class(sub_indicators=['MACD', 'RSI', 'KDJ'])
                else:
                    indicator = indicator_class()

                indicators.append(indicator)
                logger.debug(f"成功初始化指标: {class_name}")
            except Exception as e:
                logger.warning(f"无法初始化指标 {class_name}: {e}")
                continue

        logger.info(f"已初始化 {len(indicators)} 个指标")
    
    def validate_all_patterns(self) -> List[ValidationResult]:
        """验证所有模式的极性标注"""
        results = []
        all_patterns = self.registry.get_all_patterns()
        
        logger.info(f"开始验证 {len(all_patterns)} 个模式的极性标注")
        
        for pattern_id, pattern_info in all_patterns.items():
            result = self._validate_single_pattern(pattern_id, pattern_info)
            results.append(result)
        
        return results
    
    def _validate_single_pattern(self, pattern_id: str, pattern_info: Dict) -> ValidationResult:
        """验证单个模式的极性标注"""
        issues = []
        
        # 提取基本信息
        indicator_name = pattern_info.get('indicator_id', 'Unknown')
        display_name = pattern_info.get('display_name', 'Unknown')
        polarity = pattern_info.get('polarity')
        pattern_type = pattern_info.get('pattern_type')
        score_impact = pattern_info.get('score_impact', 0.0)
        
        # 检查是否有极性标注
        has_polarity = polarity is not None
        if not has_polarity:
            issues.append("缺少极性标注")
        
        # 检查极性与其他属性的一致性
        is_consistent = True
        
        if has_polarity:
            # 检查极性与模式类型的一致性
            if polarity == PatternPolarity.POSITIVE:
                if pattern_type and 'BEARISH' in str(pattern_type):
                    issues.append(f"极性为POSITIVE但模式类型为{pattern_type}")
                    is_consistent = False
                if score_impact < -5:
                    issues.append(f"极性为POSITIVE但评分影响为{score_impact}")
                    is_consistent = False
            
            elif polarity == PatternPolarity.NEGATIVE:
                if pattern_type and 'BULLISH' in str(pattern_type):
                    issues.append(f"极性为NEGATIVE但模式类型为{pattern_type}")
                    is_consistent = False
                if score_impact > 5:
                    issues.append(f"极性为NEGATIVE但评分影响为{score_impact}")
                    is_consistent = False
            
            # 检查显示名称与极性的一致性
            display_lower = display_name.lower()
            
            if polarity == PatternPolarity.POSITIVE:
                negative_words = ['死叉', '下行', '空头', '看跌', '下跌', 'bearish', 'death', 'falling']
                for word in negative_words:
                    if word in display_lower:
                        issues.append(f"极性为POSITIVE但显示名称包含负面词汇: {word}")
                        is_consistent = False
                        break
            
            elif polarity == PatternPolarity.NEGATIVE:
                positive_words = ['金叉', '上行', '多头', '看涨', '上涨', 'bullish', 'golden', 'rising']
                for word in positive_words:
                    if word in display_lower:
                        issues.append(f"极性为NEGATIVE但显示名称包含正面词汇: {word}")
                        is_consistent = False
                        break
        
        return ValidationResult(
            pattern_id=pattern_id,
            indicator_name=indicator_name,
            display_name=display_name,
            polarity=polarity,
            has_polarity=has_polarity,
            is_consistent=is_consistent,
            issues=issues
        )
    
    def generate_validation_report(self, results: List[ValidationResult]) -> str:
        """生成验证报告"""
        report = []
        report.append("# 模式极性标注验证报告")
        report.append("=" * 50)
        report.append("")
        
        # 统计信息
        total = len(results)
        has_polarity = len([r for r in results if r.has_polarity])
        consistent = len([r for r in results if r.is_consistent])
        has_issues = len([r for r in results if r.issues])
        
        report.append("## 📊 验证统计")
        report.append(f"- 总模式数量: {total}")
        has_polarity_pct = has_polarity/total*100 if total > 0 else 0.0
        consistent_pct = consistent/total*100 if total > 0 else 0.0
        has_issues_pct = has_issues/total*100 if total > 0 else 0.0

        report.append(f"- 已标注极性: {has_polarity} ({has_polarity_pct:.1f}%)")
        report.append(f"- 标注一致: {consistent} ({consistent_pct:.1f}%)")
        report.append(f"- 存在问题: {has_issues} ({has_issues_pct:.1f}%)")
        report.append("")
        
        # 按极性分组统计
        polarity_stats = defaultdict(int)
        for result in results:
            if result.has_polarity:
                polarity_stats[result.polarity.value] += 1
            else:
                polarity_stats['未标注'] += 1
        
        report.append("## 🏷️ 极性分布")
        for polarity, count in polarity_stats.items():
            report.append(f"- {polarity}: {count} ({count/total*100:.1f}%)")
        report.append("")
        
        # 问题模式列表
        problematic = [r for r in results if r.issues]
        if problematic:
            report.append("## ⚠️ 存在问题的模式")
            report.append("")
            
            # 按问题类型分组
            missing_polarity = [r for r in problematic if not r.has_polarity]
            inconsistent = [r for r in problematic if r.has_polarity and not r.is_consistent]
            
            if missing_polarity:
                report.append("### 缺少极性标注的模式")
                report.append("")
                for result in missing_polarity[:20]:  # 显示前20个
                    report.append(f"- **{result.indicator_name}** - {result.display_name}")
                    report.append(f"  - 模式ID: {result.pattern_id}")
                    report.append("")
            
            if inconsistent:
                report.append("### 极性标注不一致的模式")
                report.append("")
                for result in inconsistent[:20]:  # 显示前20个
                    report.append(f"- **{result.indicator_name}** - {result.display_name}")
                    report.append(f"  - 模式ID: {result.pattern_id}")
                    report.append(f"  - 当前极性: {result.polarity.value}")
                    for issue in result.issues:
                        report.append(f"  - ⚠️ {issue}")
                    report.append("")
        
        # 正确标注的示例
        correct_examples = [r for r in results if r.has_polarity and r.is_consistent]
        if correct_examples:
            report.append("## ✅ 正确标注示例")
            report.append("")
            
            # 按极性分组显示示例
            for polarity in [PatternPolarity.POSITIVE, PatternPolarity.NEGATIVE, PatternPolarity.NEUTRAL]:
                examples = [r for r in correct_examples if r.polarity == polarity]
                if examples:
                    report.append(f"### {polarity.value} 极性示例")
                    for result in examples[:5]:  # 每种极性显示5个示例
                        report.append(f"- **{result.indicator_name}** - {result.display_name}")
                    report.append("")
        
        return "\n".join(report)
    
    def get_missing_polarity_patterns(self) -> List[str]:
        """获取缺少极性标注的模式ID列表"""
        results = self.validate_all_patterns()
        return [r.pattern_id for r in results if not r.has_polarity]
    
    def get_inconsistent_patterns(self) -> List[ValidationResult]:
        """获取极性标注不一致的模式"""
        results = self.validate_all_patterns()
        return [r for r in results if r.has_polarity and not r.is_consistent]


def main():
    """主函数"""
    print("🔍 开始验证模式极性标注...")
    
    validator = PolarityValidator()
    results = validator.validate_all_patterns()
    
    # 生成报告
    report = validator.generate_validation_report(results)
    
    # 保存报告
    output_path = "results/analysis/polarity_validation_report.md"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ 验证完成，报告已保存到: {output_path}")
    
    # 显示摘要
    total = len(results)
    has_polarity = len([r for r in results if r.has_polarity])
    has_issues = len([r for r in results if r.issues])
    
    print(f"\n📊 验证摘要:")
    print(f"总模式数量: {total}")
    print(f"已标注极性: {has_polarity}")
    print(f"存在问题: {has_issues}")
    
    if has_issues > 0:
        print(f"\n⚠️  发现 {has_issues} 个模式存在问题，请查看详细报告！")
    else:
        print(f"\n✅ 所有模式的极性标注都正确！")


if __name__ == "__main__":
    main()
