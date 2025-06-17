#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Find Missing Polarity Annotations
Systematically identifies all indicators that still lack polarity annotations
"""

import importlib
from indicators.pattern_registry import PatternRegistry
from utils.logger import get_logger

logger = get_logger(__name__)

def find_missing_polarity():
    """Find all patterns missing polarity annotations"""
    
    # All indicators from comprehensive audit
    all_indicators = [
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
    
    print("🔍 Scanning for missing polarity annotations...")
    
    # Initialize all indicators
    for module_name, class_name in all_indicators:
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
    
    # Get all patterns and check for missing polarity
    registry = PatternRegistry()
    all_patterns = registry.get_all_patterns()
    
    patterns_without_polarity = []
    patterns_by_indicator = {}
    
    for pattern_id, pattern_info in all_patterns.items():
        if 'polarity' not in pattern_info or pattern_info['polarity'] is None:
            patterns_without_polarity.append(pattern_id)
            
            # Group by indicator
            indicator_name = pattern_id.split('_')[0]
            if indicator_name not in patterns_by_indicator:
                patterns_by_indicator[indicator_name] = []
            patterns_by_indicator[indicator_name].append(pattern_id)
    
    print(f"\n📊 Missing Polarity Summary:")
    print(f"Total patterns: {len(all_patterns)}")
    print(f"Missing polarity: {len(patterns_without_polarity)}")
    print(f"Coverage: {(len(all_patterns) - len(patterns_without_polarity))/len(all_patterns)*100:.1f}%")
    
    if patterns_without_polarity:
        print(f"\n❌ Indicators with missing polarity annotations:")
        for indicator, patterns in sorted(patterns_by_indicator.items()):
            print(f"\n{indicator.upper()} ({len(patterns)} patterns):")
            for pattern in patterns:
                print(f"  - {pattern}")
    else:
        print(f"\n🎉 All patterns have polarity annotations!")
    
    return patterns_without_polarity, patterns_by_indicator

if __name__ == "__main__":
    missing_patterns, by_indicator = find_missing_polarity()
    
    if missing_patterns:
        print(f"\n⚠️  {len(missing_patterns)} patterns still need polarity annotations")
        print(f"📋 {len(by_indicator)} indicators need work")
    else:
        print(f"\n✅ 100% polarity coverage achieved!")
