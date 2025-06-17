#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fix Polarity Consistency Issues
Identifies and reports all polarity consistency issues for manual fixing
"""

import importlib
from indicators.pattern_registry import PatternRegistry
from utils.logger import get_logger

logger = get_logger(__name__)

def get_all_polarity_issues():
    """Get detailed information about all polarity consistency issues"""
    
    # Initialize all indicators to register patterns
    indicators_to_init = [
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
    
    # Initialize indicators
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
    
    # Get all patterns and check for issues
    registry = PatternRegistry()
    all_patterns = registry.get_all_patterns()
    
    polarity_issues = []
    
    for pattern_id, pattern_info in all_patterns.items():
        if 'polarity' in pattern_info and pattern_info['polarity'] is not None:
            polarity = pattern_info['polarity']
            pattern_type = pattern_info.get('pattern_type')
            score_impact = pattern_info.get('score_impact', 0)
            
            issues = []
            
            # Check polarity vs pattern_type consistency
            if polarity.name == 'POSITIVE' and pattern_type.name == 'BEARISH':
                issues.append("POSITIVE polarity with BEARISH type")
            elif polarity.name == 'NEGATIVE' and pattern_type.name == 'BULLISH':
                issues.append("NEGATIVE polarity with BULLISH type")
            
            # Check polarity vs score_impact consistency
            if polarity.name == 'POSITIVE' and score_impact < 0:
                issues.append("POSITIVE polarity with negative score")
            elif polarity.name == 'NEGATIVE' and score_impact > 0:
                issues.append("NEGATIVE polarity with positive score")
            
            if issues:
                polarity_issues.append({
                    'pattern_id': pattern_id,
                    'issues': issues,
                    'current_polarity': polarity.name,
                    'pattern_type': pattern_type.name,
                    'score_impact': score_impact,
                    'suggested_polarity': suggest_correct_polarity(pattern_id, pattern_type.name, score_impact)
                })
    
    return polarity_issues

def suggest_correct_polarity(pattern_id, pattern_type, score_impact):
    """Suggest the correct polarity based on pattern analysis"""
    
    # Analyze pattern ID for semantic clues
    pattern_lower = pattern_id.lower()
    
    # Strong bullish indicators
    bullish_keywords = [
        'oversold', 'support', 'bounce', 'golden', 'cross_up', 'breakout', 'bullish',
        'rising', 'uptrend', 'above', 'strong_up', 'buy', 'accumulation', 'bottom'
    ]
    
    # Strong bearish indicators  
    bearish_keywords = [
        'overbought', 'resistance', 'breakdown', 'death', 'cross_down', 'bearish',
        'falling', 'downtrend', 'below', 'strong_down', 'sell', 'distribution', 'top'
    ]
    
    # Check for semantic clues in pattern ID
    for keyword in bullish_keywords:
        if keyword in pattern_lower:
            return 'POSITIVE'
    
    for keyword in bearish_keywords:
        if keyword in pattern_lower:
            return 'NEGATIVE'
    
    # Fall back to pattern type and score
    if pattern_type == 'BULLISH' or score_impact > 0:
        return 'POSITIVE'
    elif pattern_type == 'BEARISH' or score_impact < 0:
        return 'NEGATIVE'
    else:
        return 'NEUTRAL'

def generate_fix_report(issues):
    """Generate a detailed report of issues and suggested fixes"""
    
    print("🔧 POLARITY CONSISTENCY ISSUES REPORT")
    print("=" * 60)
    print(f"Total issues found: {len(issues)}")
    print()
    
    # Group by indicator
    indicator_groups = {}
    for issue in issues:
        pattern_id = issue['pattern_id']
        indicator = pattern_id.split('_')[0]
        if indicator not in indicator_groups:
            indicator_groups[indicator] = []
        indicator_groups[indicator].append(issue)
    
    for indicator, indicator_issues in indicator_groups.items():
        print(f"📊 {indicator.upper()} ({len(indicator_issues)} issues)")
        print("-" * 40)
        
        for issue in indicator_issues:
            print(f"Pattern: {issue['pattern_id']}")
            print(f"  Current: {issue['current_polarity']} | Type: {issue['pattern_type']} | Score: {issue['score_impact']}")
            print(f"  Issues: {', '.join(issue['issues'])}")
            print(f"  Suggested: {issue['suggested_polarity']}")
            print()
        print()

if __name__ == "__main__":
    print("🔍 Analyzing polarity consistency issues...")
    issues = get_all_polarity_issues()
    generate_fix_report(issues)
    
    print(f"📋 Summary: {len(issues)} patterns need polarity fixes")
    print("💡 Use the suggested polarities to manually fix each indicator file")
