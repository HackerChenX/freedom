#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Validate Critical Polarity Issues
Focus on obviously incorrect polarity annotations that violate basic technical analysis principles
"""

import importlib
from indicators.pattern_registry import PatternRegistry
from utils.logger import get_logger

logger = get_logger(__name__)

def validate_critical_issues():
    """Validate only the most critical polarity issues"""
    
    # Initialize all indicators
    indicators_to_init = [
        ('indicators.vix', 'VIX'),
        ('indicators.bias', 'BIAS'),
        ('indicators.pvt', 'PVT'),
        ('indicators.zxm_absorb', 'ZXMAbsorb'),
        ('indicators.sar', 'SAR'),
        ('indicators.wma', 'WMA'),
        ('indicators.vortex', 'Vortex'),
        ('indicators.ma', 'MA'),
        ('indicators.fibonacci_tools', 'FibonacciTools'),
        ('indicators.kdj', 'KDJ'),
        ('indicators.emv', 'EMV'),
        ('indicators.zxm_washplate', 'ZXMWashPlate'),
        ('indicators.composite_indicator', 'CompositeIndicator'),
        ('indicators.vosc', 'VOSC'),
        ('indicators.macd', 'MACD'),
        ('indicators.stochrsi', 'STOCHRSI'),
        ('indicators.ema', 'EMA'),
        ('indicators.cmo', 'CMO'),
        ('indicators.atr', 'ATR'),
        ('indicators.boll', 'BOLL'),
        ('indicators.roc', 'ROC'),
        ('indicators.dma', 'DMA'),
        ('indicators.gann_tools', 'GannTools'),
        ('indicators.cci', 'CCI'),
        ('indicators.institutional_behavior', 'InstitutionalBehavior'),
        ('indicators.psy', 'PSY'),
        ('indicators.chip_distribution', 'ChipDistribution'),
        ('indicators.mfi', 'MFI'),
        ('indicators.ichimoku', 'Ichimoku'),
        ('indicators.obv', 'OBV'),
        ('indicators.vol', 'VOL'),
        ('indicators.chaikin', 'Chaikin'),
        ('indicators.unified_ma', 'UnifiedMA'),
        ('indicators.volume_ratio', 'VolumeRatio'),
        ('indicators.wr', 'WR'),
        ('indicators.mtm', 'MTM'),
        ('indicators.vr', 'VR'),
        ('indicators.stock_vix', 'StockVIX'),
        ('indicators.kc', 'KC'),
        ('indicators.elliott_wave', 'ElliottWave'),
        ('indicators.rsi', 'RSI'),
        ('indicators.trix', 'TRIX'),
        ('indicators.aroon', 'Aroon'),
        ('indicators.momentum', 'Momentum'),
        ('indicators.dmi', 'DMI'),
        ('indicators.trend.enhanced_dmi', 'EnhancedDMI'),
        ('indicators.trend.enhanced_macd', 'EnhancedMACD'),
        ('indicators.trend.enhanced_cci', 'EnhancedCCI'),
        ('indicators.trend.enhanced_trix', 'EnhancedTRIX'),
        ('indicators.oscillator.enhanced_kdj', 'EnhancedKDJ'),
        ('indicators.volume.enhanced_mfi', 'EnhancedMFI'),
        ('indicators.volume.enhanced_obv', 'EnhancedOBV'),
        ('indicators.pattern.candlestick_patterns', 'CandlestickPatterns'),
        ('indicators.pattern.zxm_patterns', 'ZXMPatternIndicator'),
        ('indicators.pattern.advanced_candlestick_patterns', 'AdvancedCandlestickPatterns'),
    ]
    
    for module_name, class_name in indicators_to_init:
        try:
            module = importlib.import_module(module_name)
            indicator_class = getattr(module, class_name)
            
            indicator = None
            try:
                indicator = indicator_class()
            except:
                try:
                    indicator = indicator_class(periods=[5, 10, 20])
                except:
                    try:
                        indicator = indicator_class(period=14)
                    except:
                        pass
            
            if indicator and hasattr(indicator, 'register_patterns'):
                indicator.register_patterns()
                
        except Exception as e:
            print(f"Failed to initialize {class_name}: {e}")
    
    # Get all patterns
    registry = PatternRegistry()
    all_patterns = registry.get_all_patterns()
    
    print("🔍 CRITICAL POLARITY VALIDATION")
    print("=" * 60)
    print("重点检查明显违反技术分析原理的极性标注")
    print()
    
    critical_issues = []
    
    for pattern_id, pattern_info in all_patterns.items():
        display_name = pattern_info.get('display_name', '')
        description = pattern_info.get('description', '')
        pattern_type = pattern_info.get('pattern_type', '').name if hasattr(pattern_info.get('pattern_type', ''), 'name') else str(pattern_info.get('pattern_type', ''))
        score_impact = pattern_info.get('score_impact', 0)
        polarity = pattern_info.get('polarity', '').name if hasattr(pattern_info.get('polarity', ''), 'name') else str(pattern_info.get('polarity', ''))
        
        # 检查明显的极性-类型不匹配
        type_polarity_mismatch = False
        if pattern_type == 'BULLISH' and polarity == 'NEGATIVE':
            type_polarity_mismatch = True
        elif pattern_type == 'BEARISH' and polarity == 'POSITIVE':
            type_polarity_mismatch = True
        
        # 检查明显的极性-评分不匹配
        score_polarity_mismatch = False
        if polarity == 'POSITIVE' and score_impact < -5:  # 容忍小的不匹配
            score_polarity_mismatch = True
        elif polarity == 'NEGATIVE' and score_impact > 5:  # 容忍小的不匹配
            score_polarity_mismatch = True
        
        # 检查明显的技术逻辑错误
        technical_error = False
        pattern_lower = pattern_id.lower()
        
        # 金叉死叉逻辑
        if 'golden_cross' in pattern_lower and polarity == 'NEGATIVE':
            technical_error = True
        elif 'death_cross' in pattern_lower and polarity == 'POSITIVE':
            technical_error = True
        
        # 超买超卖逻辑
        elif 'oversold' in pattern_lower and polarity == 'NEGATIVE':
            technical_error = True
        elif 'overbought' in pattern_lower and polarity == 'POSITIVE':
            technical_error = True
        
        # 突破方向逻辑
        elif 'breakout' in pattern_lower and 'up' in pattern_lower and polarity == 'NEGATIVE':
            technical_error = True
        elif 'breakout' in pattern_lower and 'down' in pattern_lower and polarity == 'POSITIVE':
            technical_error = True
        
        # 多空排列逻辑
        elif 'bullish_arrangement' in pattern_lower and polarity == 'NEGATIVE':
            technical_error = True
        elif 'bearish_arrangement' in pattern_lower and polarity == 'POSITIVE':
            technical_error = True
        
        if type_polarity_mismatch or score_polarity_mismatch or technical_error:
            critical_issues.append({
                'pattern_id': pattern_id,
                'display_name': display_name,
                'description': description,
                'pattern_type': pattern_type,
                'score_impact': score_impact,
                'polarity': polarity,
                'issues': {
                    'type_mismatch': type_polarity_mismatch,
                    'score_mismatch': score_polarity_mismatch,
                    'technical_error': technical_error
                }
            })
    
    print(f"📊 验证结果:")
    print(f"- 总模式数: {len(all_patterns)}")
    print(f"- 关键问题: {len(critical_issues)}")
    print(f"- 关键问题率: {len(critical_issues)/len(all_patterns)*100:.1f}%")
    
    if critical_issues:
        print(f"\n❌ 发现 {len(critical_issues)} 个关键极性问题:")
        print("=" * 60)
        
        for issue in critical_issues:
            print(f"\n🚨 {issue['pattern_id']}")
            print(f"   名称: {issue['display_name']}")
            print(f"   类型: {issue['pattern_type']} | 评分: {issue['score_impact']} | 极性: {issue['polarity']}")
            
            problems = []
            if issue['issues']['type_mismatch']:
                problems.append("类型与极性不匹配")
            if issue['issues']['score_mismatch']:
                problems.append("评分与极性不匹配")
            if issue['issues']['technical_error']:
                problems.append("违反技术分析原理")
            
            print(f"   问题: {', '.join(problems)}")
            
            # 建议修正
            if issue['pattern_type'] == 'BULLISH':
                suggested = 'POSITIVE'
            elif issue['pattern_type'] == 'BEARISH':
                suggested = 'NEGATIVE'
            elif issue['score_impact'] > 0:
                suggested = 'POSITIVE'
            elif issue['score_impact'] < 0:
                suggested = 'NEGATIVE'
            else:
                suggested = 'NEUTRAL'
            
            print(f"   建议极性: {suggested}")
    else:
        print(f"\n✅ 没有发现关键极性问题！")
        print("所有模式的极性标注都符合基本技术分析原理")
    
    return critical_issues

if __name__ == "__main__":
    issues = validate_critical_issues()

    if issues:
        print(f"\n💡 建议:")
        print(f"请优先修复上述 {len(issues)} 个关键问题，这些是明显违反技术分析原理的")
        print(f"其他'信号不一致'的情况可能是正常的复杂技术含义")
    else:
        print(f"\n🎉 极性标注质量良好！")
        print(f"没有发现明显违反技术分析原理的极性标注")
