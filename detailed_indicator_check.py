#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Detailed Indicator Check
Systematically checks each indicator file for missing polarity annotations
"""

import os
import re
import importlib
from indicators.pattern_registry import PatternRegistry
from utils.logger import get_logger

logger = get_logger(__name__)

def scan_file_for_patterns(file_path):
    """Scan a file for pattern registrations and check polarity"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find all register_pattern_to_registry calls
        pattern_regex = r'register_pattern_to_registry\(\s*pattern_id=["\']([^"\']+)["\'][^)]*\)'
        matches = re.findall(pattern_regex, content, re.DOTALL)
        
        # Check which ones have polarity
        patterns_with_polarity = []
        patterns_without_polarity = []
        
        for pattern_id in matches:
            # Check if this pattern registration has polarity
            pattern_block_regex = rf'register_pattern_to_registry\(\s*pattern_id=["\']' + re.escape(pattern_id) + r'["\'][^)]*polarity[^)]*\)'
            if re.search(pattern_block_regex, content, re.DOTALL):
                patterns_with_polarity.append(pattern_id)
            else:
                patterns_without_polarity.append(pattern_id)
        
        return {
            'total_patterns': len(matches),
            'with_polarity': patterns_with_polarity,
            'without_polarity': patterns_without_polarity
        }
        
    except Exception as e:
        return {'error': str(e)}

def detailed_indicator_check():
    """Perform detailed check of all indicator files"""
    
    print("🔍 DETAILED INDICATOR POLARITY CHECK")
    print("=" * 60)
    
    # Get all indicator files from the comprehensive audit
    indicator_files = [
        'indicators/vix.py',
        'indicators/bias.py', 
        'indicators/pvt.py',
        'indicators/zxm_absorb.py',
        'indicators/sar.py',
        'indicators/adx.py',
        'indicators/wma.py',
        'indicators/vortex.py',
        'indicators/ma.py',
        'indicators/fibonacci_tools.py',
        'indicators/kdj.py',
        'indicators/emv.py',
        'indicators/zxm_washplate.py',
        'indicators/composite_indicator.py',
        'indicators/vosc.py',
        'indicators/macd.py',
        'indicators/stochrsi.py',
        'indicators/ema.py',
        'indicators/cmo.py',
        'indicators/atr.py',
        'indicators/boll.py',
        'indicators/roc.py',
        'indicators/dma.py',
        'indicators/gann_tools.py',
        'indicators/cci.py',
        'indicators/institutional_behavior.py',
        'indicators/psy.py',
        'indicators/chip_distribution.py',
        'indicators/mfi.py',
        'indicators/ichimoku.py',
        'indicators/obv.py',
        'indicators/vol.py',
        'indicators/chaikin.py',
        'indicators/unified_ma.py',
        'indicators/volume_ratio.py',
        'indicators/wr.py',
        'indicators/mtm.py',
        'indicators/vr.py',
        'indicators/stock_vix.py',
        'indicators/kc.py',
        'indicators/elliott_wave.py',
        'indicators/rsi.py',
        'indicators/trix.py',
        'indicators/aroon.py',
        'indicators/momentum.py',
        'indicators/dmi.py',
        'indicators/trend/enhanced_dmi.py',
        'indicators/trend/enhanced_macd.py',
        'indicators/trend/enhanced_cci.py',
        'indicators/trend/enhanced_trix.py',
        'indicators/oscillator/enhanced_kdj.py',
        'indicators/volume/enhanced_mfi.py',
        'indicators/volume/enhanced_obv.py',
        'indicators/pattern/candlestick_patterns.py',
        'indicators/pattern/zxm_patterns.py',
        'indicators/pattern/advanced_candlestick_patterns.py',
    ]
    
    total_files_checked = 0
    total_patterns_found = 0
    total_missing_polarity = 0
    indicators_with_issues = []
    
    for file_path in indicator_files:
        if os.path.exists(file_path):
            print(f"\n📁 Checking: {file_path}")
            result = scan_file_for_patterns(file_path)
            
            if 'error' in result:
                print(f"  ❌ Error: {result['error']}")
                continue
                
            total_files_checked += 1
            total_patterns_found += result['total_patterns']
            
            if result['without_polarity']:
                total_missing_polarity += len(result['without_polarity'])
                indicators_with_issues.append({
                    'file': file_path,
                    'missing_patterns': result['without_polarity']
                })
                print(f"  ⚠️  {len(result['without_polarity'])} patterns missing polarity:")
                for pattern in result['without_polarity']:
                    print(f"    - {pattern}")
            else:
                print(f"  ✅ All {result['total_patterns']} patterns have polarity")
        else:
            print(f"  ❌ File not found: {file_path}")
    
    print(f"\n📊 DETAILED CHECK SUMMARY:")
    print(f"- Files checked: {total_files_checked}")
    print(f"- Total patterns found: {total_patterns_found}")
    print(f"- Patterns missing polarity: {total_missing_polarity}")
    print(f"- Coverage: {(total_patterns_found - total_missing_polarity)/total_patterns_found*100:.1f}%" if total_patterns_found > 0 else "- Coverage: N/A")
    
    if indicators_with_issues:
        print(f"\n❌ INDICATORS NEEDING ATTENTION ({len(indicators_with_issues)}):")
        for issue in indicators_with_issues:
            print(f"\n{issue['file']} ({len(issue['missing_patterns'])} patterns):")
            for pattern in issue['missing_patterns']:
                print(f"  - {pattern}")
    else:
        print(f"\n🎉 ALL INDICATORS HAVE COMPLETE POLARITY COVERAGE!")
    
    return indicators_with_issues

def cross_check_with_registry():
    """Cross-check file scan results with pattern registry"""
    print(f"\n🔄 Cross-checking with pattern registry...")
    
    # Initialize all indicators first
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
    
    # Check registry
    registry = PatternRegistry()
    all_patterns = registry.get_all_patterns()
    
    registry_missing = []
    for pattern_id, pattern_info in all_patterns.items():
        if 'polarity' not in pattern_info or pattern_info['polarity'] is None:
            registry_missing.append(pattern_id)
    
    print(f"📋 Registry check:")
    print(f"- Total patterns in registry: {len(all_patterns)}")
    print(f"- Missing polarity in registry: {len(registry_missing)}")
    
    if registry_missing:
        print(f"❌ Registry patterns missing polarity:")
        for pattern in registry_missing:
            print(f"  - {pattern}")
    else:
        print(f"✅ All registry patterns have polarity!")
    
    return registry_missing

if __name__ == "__main__":
    # Run detailed file check
    file_issues = detailed_indicator_check()
    
    # Cross-check with registry
    registry_issues = cross_check_with_registry()
    
    print(f"\n🎯 FINAL ASSESSMENT:")
    if not file_issues and not registry_issues:
        print(f"🎉 PERFECT! All indicators have complete polarity coverage!")
    else:
        print(f"⚠️  Issues found:")
        if file_issues:
            print(f"  - {len(file_issues)} indicator files have missing polarity")
        if registry_issues:
            print(f"  - {len(registry_issues)} registry patterns missing polarity")
