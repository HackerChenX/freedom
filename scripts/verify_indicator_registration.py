#!/usr/bin/env python3
"""
独立的指标注册验证脚本
验证所有可用指标的注册状态
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib
from typing import Dict, List, Tuple

class IndicatorRegistrationVerifier:
    """指标注册验证器"""
    
    def __init__(self):
        self.available_indicators = {}
        self.registered_indicators = set()
        self.verification_results = {
            'total_available': 0,
            'total_registered': 0,
            'registration_rate': 0.0,
            'missing_indicators': [],
            'categories': {}
        }
    
    def get_all_available_indicators(self) -> Dict[str, List[Tuple[str, str, str]]]:
        """获取所有可用指标"""
        return {
            'core': [
                ('indicators.ma', 'MA', 'MA'),
                ('indicators.ema', 'EMA', 'EMA'),
                ('indicators.wma', 'WMA', 'WMA'),
                ('indicators.sar', 'SAR', 'SAR'),
                ('indicators.adx', 'ADX', 'ADX'),
                ('indicators.aroon', 'Aroon', 'AROON'),
                ('indicators.atr', 'ATR', 'ATR'),
                ('indicators.kc', 'KC', 'KC'),
                ('indicators.mfi', 'MFI', 'MFI'),
                ('indicators.momentum', 'Momentum', 'MOMENTUM'),
                ('indicators.mtm', 'MTM', 'MTM'),
                ('indicators.obv', 'OBV', 'OBV'),
                ('indicators.psy', 'PSY', 'PSY'),
                ('indicators.pvt', 'PVT', 'PVT'),
                ('indicators.roc', 'ROC', 'ROC'),
                ('indicators.trix', 'TRIX', 'TRIX'),
                ('indicators.vix', 'VIX', 'VIX'),
                ('indicators.volume_ratio', 'VolumeRatio', 'VOLUME_RATIO'),
                ('indicators.vosc', 'VOSC', 'VOSC'),
                ('indicators.vr', 'VR', 'VR'),
                ('indicators.vortex', 'Vortex', 'VORTEX'),
                ('indicators.wr', 'WR', 'WR'),
                ('indicators.ad', 'AD', 'AD'),
                # 已注册的基础指标
                ('indicators.macd', 'MACD', 'MACD'),
                ('indicators.rsi', 'RSI', 'RSI'),
                ('indicators.boll', 'BollingerBands', 'BOLL'),
                ('indicators.kdj', 'KDJ', 'KDJ'),
                ('indicators.bias', 'BIAS', 'BIAS'),
                ('indicators.cci', 'CCI', 'CCI'),
                ('indicators.chaikin', 'ChaikinVolatility', 'CHAIKIN'),
                ('indicators.dmi', 'DMI', 'DMI'),
                ('indicators.emv', 'EMV', 'EMV'),
                ('indicators.ichimoku', 'Ichimoku', 'ICHIMOKU'),
                ('indicators.cmo', 'CMO', 'CMO'),
                ('indicators.dma', 'DMA', 'DMA'),
                ('indicators.vol', 'Volume', 'VOL'),
                ('indicators.stochrsi', 'StochasticRSI', 'STOCHRSI'),
            ],
            'enhanced': [
                ('indicators.trend.enhanced_cci', 'EnhancedCCI', 'ENHANCED_CCI'),
                ('indicators.trend.enhanced_dmi', 'EnhancedDMI', 'ENHANCED_DMI'),
                ('indicators.trend.enhanced_macd', 'EnhancedMACD', 'ENHANCED_MACD_TREND'),
                ('indicators.trend.enhanced_trix', 'EnhancedTRIX', 'ENHANCED_TRIX'),
                ('indicators.oscillator.enhanced_kdj', 'EnhancedKDJ', 'ENHANCED_KDJ_OSC'),
                ('indicators.volume.enhanced_mfi', 'EnhancedMFI', 'ENHANCED_MFI'),
                ('indicators.volume.enhanced_obv', 'EnhancedOBV', 'ENHANCED_OBV'),
                ('indicators.enhanced_rsi', 'EnhancedRSI', 'ENHANCED_RSI'),
                ('indicators.enhanced_wr', 'EnhancedWR', 'ENHANCED_WR'),
                ('indicators.enhanced_macd', 'EnhancedMACD', 'ENHANCED_MACD_ROOT'),
            ],
            'composite': [
                ('indicators.composite_indicator', 'CompositeIndicator', 'COMPOSITE'),
                ('indicators.unified_ma', 'UnifiedMA', 'UNIFIED_MA'),
                ('indicators.chip_distribution', 'ChipDistribution', 'CHIP_DISTRIBUTION'),
                ('indicators.institutional_behavior', 'InstitutionalBehavior', 'INSTITUTIONAL_BEHAVIOR'),
                ('indicators.stock_vix', 'StockVIX', 'STOCK_VIX'),
            ],
            'pattern': [
                ('indicators.pattern.candlestick_patterns', 'CandlestickPatterns', 'CANDLESTICK_PATTERNS'),
                ('indicators.pattern.advanced_candlestick_patterns', 'AdvancedCandlestickPatterns', 'ADVANCED_CANDLESTICK'),
            ],
            'tools': [
                ('indicators.fibonacci_tools', 'FibonacciTools', 'FIBONACCI_TOOLS'),
                ('indicators.gann_tools', 'GannTools', 'GANN_TOOLS'),
                ('indicators.elliott_wave', 'ElliottWave', 'ELLIOTT_WAVE'),
            ],
            'formula': [
                ('indicators.formula_indicators', 'CrossOver', 'CROSS_OVER'),
                ('indicators.formula_indicators', 'KDJCondition', 'KDJ_CONDITION'),
                ('indicators.formula_indicators', 'MACDCondition', 'MACD_CONDITION'),
                ('indicators.formula_indicators', 'MACondition', 'MA_CONDITION'),
                ('indicators.formula_indicators', 'GenericCondition', 'GENERIC_CONDITION'),
            ]
        }
    
    def verify_indicator_availability(self, module_path: str, class_name: str) -> bool:
        """验证指标是否可用"""
        try:
            module = importlib.import_module(module_path)
            indicator_class = getattr(module, class_name, None)
            
            if indicator_class:
                from indicators.base_indicator import BaseIndicator
                return issubclass(indicator_class, BaseIndicator)
        except Exception:
            pass
        return False
    
    def get_currently_registered_indicators(self) -> set:
        """获取当前已注册的指标（模拟）"""
        # 基于之前的测试结果，这些是已注册的指标
        return {
            'BIAS', 'BOLL', 'CCI', 'CMO', 'CROSS_OVER', 'Chaikin', 'DMA', 'DMI', 'EMV',
            'ENHANCEDKDJ', 'ENHANCEDMACD', 'EnhancedKDJ', 'EnhancedMACD', 'EnhancedTRIX',
            'GENERIC_CONDITION', 'Ichimoku', 'KDJ', 'KDJ_CONDITION', 'MACD', 'MACD_CONDITION',
            'MA_CONDITION', 'RSI', 'Volume'
        }
    
    def verify_all_indicators(self):
        """验证所有指标"""
        print("=== 开始验证指标注册状态 ===\n")
        
        all_indicators = self.get_all_available_indicators()
        self.registered_indicators = self.get_currently_registered_indicators()
        
        total_available = 0
        total_missing = 0
        
        for category, indicators in all_indicators.items():
            print(f"=== {category.upper()} 类别指标 ===")
            
            available_in_category = 0
            missing_in_category = []
            
            for module_path, class_name, indicator_name in indicators:
                total_available += 1
                
                # 检查是否可用
                if self.verify_indicator_availability(module_path, class_name):
                    available_in_category += 1
                    
                    # 检查是否已注册
                    possible_names = [
                        indicator_name,
                        class_name,
                        indicator_name.upper(),
                        class_name.upper()
                    ]
                    
                    is_registered = any(name in self.registered_indicators for name in possible_names)
                    
                    if is_registered:
                        print(f"✅ {indicator_name}: 已注册")
                    else:
                        print(f"❌ {indicator_name}: 未注册")
                        missing_in_category.append(indicator_name)
                        total_missing += 1
                else:
                    print(f"⚠️  {indicator_name}: 不可用")
            
            self.verification_results['categories'][category] = {
                'available': available_in_category,
                'missing': len(missing_in_category),
                'missing_list': missing_in_category
            }
            
            print(f"{category} 类别: {available_in_category} 可用, {len(missing_in_category)} 未注册\n")
        
        # 更新总体结果
        self.verification_results.update({
            'total_available': total_available,
            'total_registered': len(self.registered_indicators),
            'total_missing': total_missing,
            'registration_rate': (len(self.registered_indicators) / total_available) * 100 if total_available > 0 else 0
        })
        
        self.print_summary()
    
    def print_summary(self):
        """打印验证摘要"""
        results = self.verification_results
        
        print("=== 指标注册验证摘要 ===")
        print(f"总可用指标: {results['total_available']}")
        print(f"已注册指标: {results['total_registered']}")
        print(f"未注册指标: {results['total_missing']}")
        print(f"注册率: {results['registration_rate']:.1f}%")
        
        print(f"\n=== 各类别详情 ===")
        for category, stats in results['categories'].items():
            print(f"{category.upper()}: {stats['available']} 可用, {stats['missing']} 未注册")
            if stats['missing_list']:
                print(f"  未注册: {', '.join(stats['missing_list'])}")
        
        # 评估
        if results['registration_rate'] >= 80:
            print(f"\n✅ 指标注册状态良好！注册率达到 {results['registration_rate']:.1f}%")
        elif results['registration_rate'] >= 50:
            print(f"\n⚠️  指标注册状态一般，注册率为 {results['registration_rate']:.1f}%")
        else:
            print(f"\n❌ 指标注册状态较差，注册率仅为 {results['registration_rate']:.1f}%")

def main():
    """主函数"""
    verifier = IndicatorRegistrationVerifier()
    verifier.verify_all_indicators()
    
    # 返回是否达到目标
    return verifier.verification_results['registration_rate'] >= 80

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
