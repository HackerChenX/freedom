#!/usr/bin/env python3
"""
检查未注册的指标脚本
分析系统中所有可用指标与已注册指标的差异
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib
from typing import List, Dict, Set, Tuple

def get_currently_registered_indicators() -> Set[str]:
    """获取当前已注册的指标"""
    try:
        from indicators.indicator_registry import get_registry
        registry = get_registry()
        return set(registry.get_indicator_names())
    except Exception as e:
        print(f"❌ 获取已注册指标失败: {e}")
        return set()

def get_all_available_indicators() -> Dict[str, List[Tuple[str, str]]]:
    """获取所有可用的指标"""
    indicators = {
        'core': [
            # 基础技术指标 (37个)
            ('indicators.ma', 'MA'),
            ('indicators.ema', 'EMA'),
            ('indicators.wma', 'WMA'),
            ('indicators.sar', 'SAR'),
            ('indicators.adx', 'ADX'),
            ('indicators.aroon', 'Aroon'),
            ('indicators.atr', 'ATR'),
            ('indicators.bias', 'BIAS'),
            ('indicators.cci', 'CCI'),
            ('indicators.chaikin', 'ChaikinVolatility'),
            ('indicators.cmo', 'CMO'),
            ('indicators.dma', 'DMA'),
            ('indicators.dmi', 'DMI'),
            ('indicators.emv', 'EMV'),
            ('indicators.ichimoku', 'Ichimoku'),
            ('indicators.kc', 'KC'),
            ('indicators.mfi', 'MFI'),
            ('indicators.momentum', 'Momentum'),
            ('indicators.mtm', 'MTM'),
            ('indicators.obv', 'OBV'),
            ('indicators.psy', 'PSY'),
            ('indicators.pvt', 'PVT'),
            ('indicators.roc', 'ROC'),
            ('indicators.stochrsi', 'StochasticRSI'),
            ('indicators.trix', 'TRIX'),
            ('indicators.vix', 'VIX'),
            ('indicators.vol', 'Volume'),
            ('indicators.volume_ratio', 'VolumeRatio'),
            ('indicators.vosc', 'VOSC'),
            ('indicators.vr', 'VR'),
            ('indicators.vortex', 'Vortex'),
            ('indicators.wr', 'WR'),
            ('indicators.ad', 'AD'),
            # 已在注册表中的基础指标
            ('indicators.macd', 'MACD'),
            ('indicators.rsi', 'RSI'),
            ('indicators.boll', 'BollingerBands'),
            ('indicators.kdj', 'KDJ'),
        ],
        'enhanced': [
            # 增强型指标 (11个)
            ('indicators.trend.enhanced_cci', 'EnhancedCCI'),
            ('indicators.trend.enhanced_dmi', 'EnhancedDMI'),
            ('indicators.trend.enhanced_macd', 'EnhancedMACD'),
            ('indicators.trend.enhanced_trix', 'EnhancedTRIX'),
            ('indicators.oscillator.enhanced_kdj', 'EnhancedKDJ'),
            ('indicators.volume.enhanced_mfi', 'EnhancedMFI'),
            ('indicators.volume.enhanced_obv', 'EnhancedOBV'),
            ('indicators.enhanced_rsi', 'EnhancedRSI'),
            ('indicators.enhanced_stochrsi', 'EnhancedStochasticRSI'),
            ('indicators.enhanced_wr', 'EnhancedWR'),
            ('indicators.enhanced_macd', 'EnhancedMACD'),  # 根目录版本
        ],
        'composite': [
            # 复合指标 (5个)
            ('indicators.composite_indicator', 'CompositeIndicator'),
            ('indicators.unified_ma', 'UnifiedMA'),
            ('indicators.chip_distribution', 'ChipDistribution'),
            ('indicators.institutional_behavior', 'InstitutionalBehavior'),
            ('indicators.stock_vix', 'StockVIX'),
        ],
        'pattern': [
            # 形态指标 (3个)
            ('indicators.pattern.candlestick_patterns', 'CandlestickPatterns'),
            ('indicators.pattern.advanced_candlestick_patterns', 'AdvancedCandlestickPatterns'),
            ('indicators.pattern.zxm_patterns', 'ZXMPatterns'),
        ],
        'tools': [
            # 特色工具 (3个)
            ('indicators.fibonacci_tools', 'FibonacciTools'),
            ('indicators.gann_tools', 'GannTools'),
            ('indicators.elliott_wave', 'ElliottWave'),
        ],
        'zxm': [
            # ZXM体系指标 (25个)
            # ZXM Trend (9个)
            ('indicators.zxm.trend_indicators', 'ZXMDailyTrendUp'),
            ('indicators.zxm.trend_indicators', 'ZXMWeeklyTrendUp'),
            ('indicators.zxm.trend_indicators', 'ZXMMonthlyKDJTrendUp'),
            ('indicators.zxm.trend_indicators', 'ZXMWeeklyKDJDOrDEATrendUp'),
            ('indicators.zxm.trend_indicators', 'ZXMWeeklyKDJDTrendUp'),
            ('indicators.zxm.trend_indicators', 'ZXMMonthlyMACD'),
            ('indicators.zxm.trend_indicators', 'TrendDetector'),
            ('indicators.zxm.trend_indicators', 'TrendDuration'),
            ('indicators.zxm.trend_indicators', 'ZXMWeeklyMACD'),
            # ZXM Buy Points (5个)
            ('indicators.zxm.buy_point_indicators', 'ZXMDailyMACD'),
            ('indicators.zxm.buy_point_indicators', 'ZXMTurnover'),
            ('indicators.zxm.buy_point_indicators', 'ZXMVolumeShrink'),
            ('indicators.zxm.buy_point_indicators', 'ZXMMACallback'),
            ('indicators.zxm.buy_point_indicators', 'ZXMBSAbsorb'),
            # ZXM Elasticity (4个)
            ('indicators.zxm.elasticity_indicators', 'AmplitudeElasticity'),
            ('indicators.zxm.elasticity_indicators', 'ZXMRiseElasticity'),
            ('indicators.zxm.elasticity_indicators', 'Elasticity'),
            ('indicators.zxm.elasticity_indicators', 'BounceDetector'),
            # ZXM Score (3个)
            ('indicators.zxm.score_indicators', 'ZXMElasticityScore'),
            ('indicators.zxm.score_indicators', 'ZXMBuyPointScore'),
            ('indicators.zxm.score_indicators', 'StockScoreCalculator'),
            # ZXM其他 (4个)
            ('indicators.zxm.market_breadth', 'ZXMMarketBreadth'),
            ('indicators.zxm.selection_model', 'SelectionModel'),
            ('indicators.zxm.diagnostics', 'ZXMDiagnostics'),
            ('indicators.zxm.buy_point_indicators', 'BuyPointDetector'),  # 假设存在
        ],
        'formula': [
            # 公式指标 (已在注册表中)
            ('indicators.formula_indicators', 'CrossOver'),
            ('indicators.formula_indicators', 'KDJCondition'),
            ('indicators.formula_indicators', 'MACDCondition'),
            ('indicators.formula_indicators', 'MACondition'),
            ('indicators.formula_indicators', 'GenericCondition'),
        ]
    }
    
    return indicators

def test_indicator_availability(module_path: str, class_name: str) -> bool:
    """测试指标是否可用"""
    try:
        module = importlib.import_module(module_path)
        indicator_class = getattr(module, class_name, None)
        
        if indicator_class:
            from indicators.base_indicator import BaseIndicator
            if issubclass(indicator_class, BaseIndicator):
                return True
    except Exception:
        pass
    return False

def analyze_unregistered_indicators():
    """分析未注册的指标"""
    print("=== 检查未注册指标 ===\n")
    
    # 获取已注册指标
    registered = get_currently_registered_indicators()
    print(f"当前已注册指标数量: {len(registered)}")
    print(f"已注册指标: {sorted(registered)}\n")
    
    # 获取所有可用指标
    all_indicators = get_all_available_indicators()
    
    unregistered_by_category = {}
    total_available = 0
    total_unregistered = 0
    
    for category, indicators in all_indicators.items():
        print(f"=== {category.upper()} 类别指标 ===")
        available_indicators = []
        unregistered_indicators = []
        
        for module_path, class_name in indicators:
            total_available += 1
            
            # 生成可能的注册名称
            possible_names = [
                class_name,
                class_name.upper(),
                class_name.replace('Enhanced', 'ENHANCED_'),
                class_name.replace('ZXM', 'ZXM_'),
            ]
            
            # 检查是否已注册
            is_registered = any(name in registered for name in possible_names)
            
            if test_indicator_availability(module_path, class_name):
                available_indicators.append((module_path, class_name))
                if not is_registered:
                    unregistered_indicators.append((module_path, class_name))
                    total_unregistered += 1
                    print(f"❌ 未注册: {class_name} ({module_path})")
                else:
                    print(f"✅ 已注册: {class_name}")
            else:
                print(f"⚠️  不可用: {class_name} ({module_path})")
        
        unregistered_by_category[category] = unregistered_indicators
        print(f"{category} 类别: {len(available_indicators)} 可用, {len(unregistered_indicators)} 未注册\n")
    
    # 总结
    print("=== 总结 ===")
    print(f"总可用指标: {total_available}")
    print(f"已注册指标: {len(registered)}")
    print(f"未注册指标: {total_unregistered}")
    print(f"注册率: {(len(registered)/total_available)*100:.1f}%")
    
    if total_unregistered > 0:
        print(f"\n=== 未注册指标详细列表 ===")
        for category, indicators in unregistered_by_category.items():
            if indicators:
                print(f"\n{category.upper()} 类别未注册指标:")
                for i, (module_path, class_name) in enumerate(indicators, 1):
                    print(f"  {i}. {class_name} - {module_path}")
    
    return unregistered_by_category

if __name__ == "__main__":
    analyze_unregistered_indicators()
