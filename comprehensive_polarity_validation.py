#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comprehensive Polarity Validation for ALL Technical Indicators
Validates polarity annotations across all 797 patterns found in the audit
"""

import pandas as pd
import importlib
from indicators.pattern_registry import PatternRegistry
from utils.logger import get_logger

logger = get_logger(__name__)

class ComprehensivePolarityValidator:
    def __init__(self):
        self.indicators_with_patterns = [
            # Core indicators from audit
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
        
        self.initialized_indicators = []
        self.failed_indicators = []
        
    def initialize_all_indicators(self):
        """Initialize all indicators to trigger pattern registration"""
        print("🔄 Initializing all indicators...")
        
        for module_name, class_name in self.indicators_with_patterns:
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
                    self.initialized_indicators.append((module_name, class_name, indicator))
                    print(f"✅ {class_name}")
                else:
                    self.failed_indicators.append((module_name, class_name, "No register_patterns method"))
                    print(f"⚠️  {class_name} - No register_patterns method")
                    
            except Exception as e:
                self.failed_indicators.append((module_name, class_name, str(e)))
                print(f"❌ {class_name} - {e}")
        
        print(f"\n📊 Initialization complete:")
        print(f"- Successfully initialized: {len(self.initialized_indicators)}")
        print(f"- Failed to initialize: {len(self.failed_indicators)}")
    
    def validate_all_patterns(self):
        """Validate polarity annotations for all patterns"""
        print("\n🔍 Validating polarity annotations...")
        
        # Get all registered patterns
        registry = PatternRegistry()
        all_patterns = registry.get_all_patterns()
        
        print(f"📋 Total registered patterns: {len(all_patterns)}")
        
        # Check polarity annotations
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
                        'issues': consistency_issues,
                        'polarity': polarity.name,
                        'pattern_type': pattern_type.name,
                        'score_impact': score_impact
                    })
            else:
                patterns_without_polarity.append(pattern_id)
        
        # Generate report
        print(f"\n🏷️  Polarity Annotation Results:")
        print(f"- Total patterns: {len(all_patterns)}")
        print(f"- With polarity: {patterns_with_polarity} ({patterns_with_polarity/len(all_patterns)*100:.1f}%)")
        print(f"- Without polarity: {len(patterns_without_polarity)} ({len(patterns_without_polarity)/len(all_patterns)*100:.1f}%)")
        print(f"- POSITIVE: {positive_patterns} ({positive_patterns/len(all_patterns)*100:.1f}%)")
        print(f"- NEGATIVE: {negative_patterns} ({negative_patterns/len(all_patterns)*100:.1f}%)")
        print(f"- NEUTRAL: {neutral_patterns} ({neutral_patterns/len(all_patterns)*100:.1f}%)")
        
        if patterns_without_polarity:
            print(f"\n❌ Patterns missing polarity annotations:")
            for pattern_id in patterns_without_polarity[:20]:  # Show first 20
                print(f"  - {pattern_id}")
            if len(patterns_without_polarity) > 20:
                print(f"  ... and {len(patterns_without_polarity) - 20} more")
        
        if polarity_issues:
            print(f"\n⚠️  Polarity consistency issues:")
            for issue in polarity_issues[:10]:  # Show first 10
                print(f"  - {issue['pattern_id']}: {', '.join(issue['issues'])}")
            if len(polarity_issues) > 10:
                print(f"  ... and {len(polarity_issues) - 10} more")
        
        return {
            'total_patterns': len(all_patterns),
            'with_polarity': patterns_with_polarity,
            'without_polarity': patterns_without_polarity,
            'polarity_issues': polarity_issues,
            'positive_patterns': positive_patterns,
            'negative_patterns': negative_patterns,
            'neutral_patterns': neutral_patterns
        }
    
    def generate_missing_polarity_task_list(self, validation_results):
        """Generate prioritized task list for missing polarity annotations"""
        patterns_without_polarity = validation_results['without_polarity']
        
        if not patterns_without_polarity:
            print("\n🎉 All patterns have polarity annotations!")
            return []
        
        # Group patterns by indicator
        indicator_groups = {}
        for pattern_id in patterns_without_polarity:
            # Extract indicator name from pattern ID
            parts = pattern_id.split('_')
            if len(parts) >= 2:
                indicator_name = parts[0]
                if indicator_name not in indicator_groups:
                    indicator_groups[indicator_name] = []
                indicator_groups[indicator_name].append(pattern_id)
        
        # Create prioritized task list
        task_list = []
        for indicator_name, patterns in indicator_groups.items():
            task_list.append({
                'indicator': indicator_name,
                'pattern_count': len(patterns),
                'patterns': patterns,
                'priority': 'HIGH' if len(patterns) > 10 else 'MEDIUM' if len(patterns) > 5 else 'LOW'
            })
        
        # Sort by pattern count (descending)
        task_list.sort(key=lambda x: x['pattern_count'], reverse=True)
        
        print(f"\n📋 Task List for Missing Polarity Annotations:")
        print(f"Total indicators needing work: {len(task_list)}")
        print(f"Total patterns needing polarity: {len(patterns_without_polarity)}")
        
        for i, task in enumerate(task_list, 1):
            print(f"{i}. {task['indicator']} ({task['priority']}): {task['pattern_count']} patterns")
        
        return task_list

if __name__ == "__main__":
    validator = ComprehensivePolarityValidator()
    
    # Initialize all indicators
    validator.initialize_all_indicators()
    
    # Validate polarity annotations
    results = validator.validate_all_patterns()
    
    # Generate task list for missing annotations
    task_list = validator.generate_missing_polarity_task_list(results)
    
    # Final summary
    if results['without_polarity']:
        print(f"\n⚠️  WORK NEEDED: {len(results['without_polarity'])} patterns missing polarity annotations")
        print(f"📊 Coverage: {results['with_polarity']}/{results['total_patterns']} ({results['with_polarity']/results['total_patterns']*100:.1f}%)")
    else:
        print(f"\n🎉 SUCCESS: All {results['total_patterns']} patterns have polarity annotations!")
        print(f"✅ 100% polarity coverage achieved!")
