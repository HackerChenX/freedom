#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
全面的极性标注完整性检查
"""

import pandas as pd
from indicators.pattern_registry import PatternRegistry
from utils.logger import get_logger

logger = get_logger(__name__)

def comprehensive_polarity_check():
    """执行全面的极性标注完整性检查"""
    
    print("🔍 开始全面的极性标注完整性检查...")
    
    # 初始化所有指标以触发模式注册
    indicators_to_test = [
        # 基础指标
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
        ('indicators.chip_distribution', 'ChipDistribution'),
        ('indicators.fibonacci_tools', 'FibonacciTools'),
        ('indicators.elliott_wave', 'ElliottWave'),
    ]
    
    initialized_indicators = []
    failed_indicators = []
    
    # 初始化指标
    for module_name, class_name in indicators_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            indicator_class = getattr(module, class_name)
            
            # 特殊处理某些指标的初始化参数
            if class_name == 'MA':
                indicator = indicator_class(periods=[5, 10, 20])
            else:
                indicator = indicator_class()
                
            # 注册模式
            if hasattr(indicator, 'register_patterns'):
                indicator.register_patterns()
                
            initialized_indicators.append((class_name, indicator))
            print(f"✅ 成功初始化: {class_name}")
            
        except Exception as e:
            failed_indicators.append((class_name, str(e)))
            print(f"❌ 初始化失败: {class_name} - {e}")
    
    print(f"\n📊 初始化统计:")
    print(f"- 成功初始化: {len(initialized_indicators)} 个指标")
    print(f"- 初始化失败: {len(failed_indicators)} 个指标")
    
    # 获取模式注册表
    registry = PatternRegistry()
    all_patterns = registry.get_all_patterns()
    
    print(f"\n📋 模式统计:")
    print(f"- 总注册模式数量: {len(all_patterns)}")
    
    # 检查极性标注
    patterns_with_polarity = 0
    patterns_without_polarity = 0
    polarity_issues = []
    
    positive_patterns = 0
    negative_patterns = 0
    neutral_patterns = 0
    
    for pattern_id, pattern_info in all_patterns.items():
        if 'polarity' in pattern_info and pattern_info['polarity'] is not None:
            patterns_with_polarity += 1
            
            # 统计极性分布
            polarity = pattern_info['polarity']
            if polarity.name == 'POSITIVE':
                positive_patterns += 1
            elif polarity.name == 'NEGATIVE':
                negative_patterns += 1
            elif polarity.name == 'NEUTRAL':
                neutral_patterns += 1
                
            # 检查极性与模式类型的一致性
            pattern_type = pattern_info.get('pattern_type')
            score_impact = pattern_info.get('score_impact', 0)
            
            consistency_issues = []
            
            # 检查极性与pattern_type的一致性
            if polarity.name == 'POSITIVE' and pattern_type.name == 'BEARISH':
                consistency_issues.append("极性为POSITIVE但模式类型为BEARISH")
            elif polarity.name == 'NEGATIVE' and pattern_type.name == 'BULLISH':
                consistency_issues.append("极性为NEGATIVE但模式类型为BULLISH")
                
            # 检查极性与score_impact的一致性
            if polarity.name == 'POSITIVE' and score_impact < 0:
                consistency_issues.append("极性为POSITIVE但评分影响为负值")
            elif polarity.name == 'NEGATIVE' and score_impact > 0:
                consistency_issues.append("极性为NEGATIVE但评分影响为正值")
                
            if consistency_issues:
                polarity_issues.append({
                    'pattern_id': pattern_id,
                    'issues': consistency_issues,
                    'polarity': polarity.name,
                    'pattern_type': pattern_type.name,
                    'score_impact': score_impact
                })
        else:
            patterns_without_polarity += 1
            print(f"⚠️  缺少极性标注: {pattern_id}")
    
    print(f"\n🏷️  极性标注统计:")
    print(f"- 已标注极性: {patterns_with_polarity} ({patterns_with_polarity/len(all_patterns)*100:.1f}%)")
    print(f"- 缺少极性: {patterns_without_polarity} ({patterns_without_polarity/len(all_patterns)*100:.1f}%)")
    print(f"- POSITIVE: {positive_patterns} ({positive_patterns/len(all_patterns)*100:.1f}%)")
    print(f"- NEGATIVE: {negative_patterns} ({negative_patterns/len(all_patterns)*100:.1f}%)")
    print(f"- NEUTRAL: {neutral_patterns} ({neutral_patterns/len(all_patterns)*100:.1f}%)")
    
    # 报告一致性问题
    if polarity_issues:
        print(f"\n❌ 发现 {len(polarity_issues)} 个极性一致性问题:")
        for issue in polarity_issues[:10]:  # 只显示前10个
            print(f"  - {issue['pattern_id']}: {', '.join(issue['issues'])}")
    else:
        print(f"\n✅ 所有模式的极性标注都与模式类型和评分影响保持一致!")
    
    # 最终结果
    if patterns_without_polarity == 0 and len(polarity_issues) == 0:
        print(f"\n🎉 极性标注完整性检查通过!")
        print(f"   - 所有 {len(all_patterns)} 个模式都有正确的极性标注")
        print(f"   - 极性标注与模式类型、评分影响完全一致")
        return True
    else:
        print(f"\n⚠️  极性标注完整性检查发现问题:")
        if patterns_without_polarity > 0:
            print(f"   - {patterns_without_polarity} 个模式缺少极性标注")
        if polarity_issues:
            print(f"   - {len(polarity_issues)} 个模式存在一致性问题")
        return False

if __name__ == "__main__":
    comprehensive_polarity_check()
