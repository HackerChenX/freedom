#!/usr/bin/env python3
"""
独立的指标注册脚本
用于测试和验证指标注册功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib
from typing import List, Tuple

def test_indicator_import(module_path: str, class_name: str) -> bool:
    """测试指标导入"""
    try:
        module = importlib.import_module(module_path)
        indicator_class = getattr(module, class_name, None)
        
        if indicator_class:
            # 检查是否为BaseIndicator子类
            from indicators.base_indicator import BaseIndicator
            if issubclass(indicator_class, BaseIndicator):
                print(f"✅ {module_path}.{class_name} - 导入成功")
                return True
            else:
                print(f"❌ {module_path}.{class_name} - 不是BaseIndicator子类")
        else:
            print(f"❌ {module_path}.{class_name} - 类不存在")
    except ImportError as e:
        print(f"❌ {module_path}.{class_name} - 导入失败: {e}")
    except Exception as e:
        print(f"❌ {module_path}.{class_name} - 其他错误: {e}")
    
    return False

def test_core_indicators():
    """测试核心指标"""
    print("=== 测试核心指标 ===")
    
    core_indicators = [
        ('indicators.ma', 'MA'),
        ('indicators.ema', 'EMA'),
        ('indicators.wma', 'WMA'),
        ('indicators.sar', 'SAR'),
        ('indicators.adx', 'ADX'),
        ('indicators.aroon', 'Aroon'),
        ('indicators.atr', 'ATR'),
        ('indicators.bias', 'BIAS'),
        ('indicators.cci', 'CCI'),
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
    ]
    
    success_count = 0
    for module_path, class_name in core_indicators:
        if test_indicator_import(module_path, class_name):
            success_count += 1
    
    print(f"\n核心指标测试结果: {success_count}/{len(core_indicators)} 成功")
    return success_count

def test_enhanced_indicators():
    """测试增强指标"""
    print("\n=== 测试增强指标 ===")
    
    enhanced_indicators = [
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
    ]
    
    success_count = 0
    for module_path, class_name in enhanced_indicators:
        if test_indicator_import(module_path, class_name):
            success_count += 1
    
    print(f"\n增强指标测试结果: {success_count}/{len(enhanced_indicators)} 成功")
    return success_count

def test_composite_indicators():
    """测试复合指标"""
    print("\n=== 测试复合指标 ===")
    
    composite_indicators = [
        ('indicators.composite_indicator', 'CompositeIndicator'),
        ('indicators.unified_ma', 'UnifiedMA'),
        ('indicators.chip_distribution', 'ChipDistribution'),
        ('indicators.institutional_behavior', 'InstitutionalBehavior'),
        ('indicators.stock_vix', 'StockVIX'),
    ]
    
    success_count = 0
    for module_path, class_name in composite_indicators:
        if test_indicator_import(module_path, class_name):
            success_count += 1
    
    print(f"\n复合指标测试结果: {success_count}/{len(composite_indicators)} 成功")
    return success_count

def test_pattern_indicators():
    """测试形态指标"""
    print("\n=== 测试形态指标 ===")
    
    pattern_indicators = [
        ('indicators.pattern.candlestick_patterns', 'CandlestickPatterns'),
        ('indicators.pattern.advanced_candlestick_patterns', 'AdvancedCandlestickPatterns'),
        ('indicators.pattern.zxm_patterns', 'ZXMPatterns'),
    ]
    
    success_count = 0
    for module_path, class_name in pattern_indicators:
        if test_indicator_import(module_path, class_name):
            success_count += 1
    
    print(f"\n形态指标测试结果: {success_count}/{len(pattern_indicators)} 成功")
    return success_count

def test_tool_indicators():
    """测试工具指标"""
    print("\n=== 测试工具指标 ===")
    
    tool_indicators = [
        ('indicators.fibonacci_tools', 'FibonacciTools'),
        ('indicators.gann_tools', 'GannTools'),
        ('indicators.elliott_wave', 'ElliottWave'),
    ]
    
    success_count = 0
    for module_path, class_name in tool_indicators:
        if test_indicator_import(module_path, class_name):
            success_count += 1
    
    print(f"\n工具指标测试结果: {success_count}/{len(tool_indicators)} 成功")
    return success_count

def main():
    """主函数"""
    print("开始测试指标导入...")
    
    # 测试各类指标
    core_success = test_core_indicators()
    enhanced_success = test_enhanced_indicators()
    composite_success = test_composite_indicators()
    pattern_success = test_pattern_indicators()
    tool_success = test_tool_indicators()
    
    # 总结
    total_success = core_success + enhanced_success + composite_success + pattern_success + tool_success
    total_indicators = 33 + 10 + 5 + 3 + 3  # 各类指标总数
    
    print(f"\n=== 总结 ===")
    print(f"总指标数: {total_indicators}")
    print(f"成功导入: {total_success}")
    print(f"成功率: {total_success/total_indicators*100:.1f}%")
    
    if total_success >= 40:  # 目标是至少40个指标能成功导入
        print("✅ 指标导入测试通过！")
        return True
    else:
        print("❌ 指标导入测试未达到预期目标")
        return False

if __name__ == "__main__":
    main()
