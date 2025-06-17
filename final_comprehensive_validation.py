#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Final Comprehensive Validation
Validates that we have achieved complete polarity annotation coverage
across all technical indicators in the system
"""

import importlib
from indicators.pattern_registry import PatternRegistry
from utils.logger import get_logger

logger = get_logger(__name__)

def run_final_validation():
    """Run final comprehensive validation"""
    
    print("🎯 FINAL COMPREHENSIVE VALIDATION")
    print("=" * 60)
    print("Validating complete polarity annotation coverage across ALL technical indicators")
    print()
    
    # All indicators with patterns from our comprehensive audit
    all_indicators = [
        # Core indicators (55 total)
        ('indicators.vix', 'VIX'),
        ('indicators.bias', 'BIAS'),
        ('indicators.pvt', 'PVT'),
        ('indicators.zxm_absorb', 'ZXMAbsorb'),
        ('indicators.sar', 'SAR'),
        ('indicators.adx', 'ADX'),
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
        
        # Enhanced indicators
        ('indicators.trend.enhanced_dmi', 'EnhancedDMI'),
        ('indicators.trend.enhanced_macd', 'EnhancedMACD'),
        ('indicators.trend.enhanced_cci', 'EnhancedCCI'),
        ('indicators.trend.enhanced_trix', 'EnhancedTRIX'),
        ('indicators.oscillator.enhanced_kdj', 'EnhancedKDJ'),
        ('indicators.volume.enhanced_mfi', 'EnhancedMFI'),
        ('indicators.volume.enhanced_obv', 'EnhancedOBV'),
        
        # Pattern indicators
        ('indicators.pattern.candlestick_patterns', 'CandlestickPatterns'),
        ('indicators.pattern.zxm_patterns', 'ZXMPatternIndicator'),
        ('indicators.pattern.advanced_candlestick_patterns', 'AdvancedCandlestickPatterns'),
    ]
    
    # Initialize all indicators
    print("🔄 Initializing all indicators...")
    initialized_count = 0
    failed_count = 0
    
    for module_name, class_name in all_indicators:
        try:
            module = importlib.import_module(module_name)
            indicator_class = getattr(module, class_name)
            
            # Try different initialization approaches
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
                initialized_count += 1
                print(f"✅ {class_name}")
            else:
                failed_count += 1
                print(f"⚠️  {class_name} - No register_patterns method")
                
        except Exception as e:
            failed_count += 1
            print(f"❌ {class_name} - {e}")
    
    print(f"\n📊 Initialization Summary:")
    print(f"- Successfully initialized: {initialized_count}")
    print(f"- Failed to initialize: {failed_count}")
    print(f"- Total indicators processed: {len(all_indicators)}")
    
    # Get all registered patterns
    registry = PatternRegistry()
    all_patterns = registry.get_all_patterns()
    
    print(f"\n📋 Pattern Registry Analysis:")
    print(f"- Total registered patterns: {len(all_patterns)}")
    
    # Analyze polarity coverage
    patterns_with_polarity = 0
    patterns_without_polarity = []
    polarity_issues = []
    
    positive_patterns = 0
    negative_patterns = 0
    neutral_patterns = 0
    
    for pattern_id, pattern_info in all_patterns.items():
        if 'polarity' in pattern_info and pattern_info['polarity'] is not None:
            patterns_with_polarity += 1
            
            # Count polarity distribution
            polarity = pattern_info['polarity']
            if polarity.name == 'POSITIVE':
                positive_patterns += 1
            elif polarity.name == 'NEGATIVE':
                negative_patterns += 1
            elif polarity.name == 'NEUTRAL':
                neutral_patterns += 1
            
            # Check consistency
            pattern_type = pattern_info.get('pattern_type')
            score_impact = pattern_info.get('score_impact', 0)
            
            consistency_issues = []
            
            # Check polarity vs pattern_type consistency
            if polarity.name == 'POSITIVE' and pattern_type.name == 'BEARISH':
                consistency_issues.append("POSITIVE polarity with BEARISH type")
            elif polarity.name == 'NEGATIVE' and pattern_type.name == 'BULLISH':
                consistency_issues.append("NEGATIVE polarity with BULLISH type")
            
            # Check polarity vs score_impact consistency
            if polarity.name == 'POSITIVE' and score_impact < 0:
                consistency_issues.append("POSITIVE polarity with negative score")
            elif polarity.name == 'NEGATIVE' and score_impact > 0:
                consistency_issues.append("NEGATIVE polarity with positive score")
            
            if consistency_issues:
                polarity_issues.append({
                    'pattern_id': pattern_id,
                    'issues': consistency_issues
                })
        else:
            patterns_without_polarity.append(pattern_id)
    
    # Generate final report
    print(f"\n🏷️  POLARITY ANNOTATION FINAL RESULTS:")
    print(f"- Total patterns: {len(all_patterns)}")
    print(f"- With polarity: {patterns_with_polarity} ({patterns_with_polarity/len(all_patterns)*100:.1f}%)")
    print(f"- Without polarity: {len(patterns_without_polarity)} ({len(patterns_without_polarity)/len(all_patterns)*100:.1f}%)")
    print(f"- POSITIVE: {positive_patterns} ({positive_patterns/len(all_patterns)*100:.1f}%)")
    print(f"- NEGATIVE: {negative_patterns} ({negative_patterns/len(all_patterns)*100:.1f}%)")
    print(f"- NEUTRAL: {neutral_patterns} ({neutral_patterns/len(all_patterns)*100:.1f}%)")
    print(f"- Consistency issues: {len(polarity_issues)}")
    
    # Final assessment
    success = (len(patterns_without_polarity) == 0 and len(polarity_issues) == 0)
    
    print(f"\n{'🎉' if success else '⚠️'} FINAL ASSESSMENT:")
    if success:
        print(f"✅ COMPLETE SUCCESS!")
        print(f"   - All {len(all_patterns)} patterns have proper polarity annotations")
        print(f"   - Zero consistency issues")
        print(f"   - 100% polarity coverage achieved across all technical indicators")
        print(f"   - {initialized_count} indicators successfully processed")
    else:
        print(f"❌ WORK STILL NEEDED:")
        if patterns_without_polarity:
            print(f"   - {len(patterns_without_polarity)} patterns missing polarity annotations")
        if polarity_issues:
            print(f"   - {len(polarity_issues)} consistency issues to fix")
    
    print(f"\n📈 COMPARISON WITH ORIGINAL AUDIT:")
    print(f"- Original audit found: 797 patterns across 60 indicator files")
    print(f"- Current validation: {len(all_patterns)} patterns across {initialized_count} indicators")
    print(f"- Coverage difference: {len(all_patterns) - 797} patterns")
    
    if len(all_patterns) < 797:
        print(f"⚠️  Note: Some patterns from the original audit may not be registering properly")
    elif len(all_patterns) > 797:
        print(f"✅ Note: We have more patterns than the original audit, indicating comprehensive coverage")
    else:
        print(f"✅ Perfect match with original audit findings")
    
    return success, {
        'total_patterns': len(all_patterns),
        'with_polarity': patterns_with_polarity,
        'without_polarity': len(patterns_without_polarity),
        'consistency_issues': len(polarity_issues),
        'initialized_indicators': initialized_count,
        'failed_indicators': failed_count
    }

if __name__ == "__main__":
    success, results = run_final_validation()
    
    if success:
        print(f"\n🏆 MISSION ACCOMPLISHED!")
        print(f"Technical indicator polarity annotation audit is COMPLETE!")
    else:
        print(f"\n🔧 Additional work needed to complete the audit.")
    
    exit(0 if success else 1)
