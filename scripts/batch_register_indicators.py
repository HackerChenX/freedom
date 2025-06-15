#!/usr/bin/env python3
"""
批量注册指标脚本
将所有可用但未注册的指标批量注册到系统中
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib
from typing import List, Dict, Tuple

def batch_register_core_indicators():
    """批量注册核心指标"""
    print("=== 批量注册核心指标 ===")
    
    from indicators.indicator_registry import get_registry
    registry = get_registry()
    
    core_indicators = [
        ('indicators.ma', 'MA', 'MA', '移动平均线'),
        ('indicators.ema', 'EMA', 'EMA', '指数移动平均线'),
        ('indicators.wma', 'WMA', 'WMA', '加权移动平均线'),
        ('indicators.sar', 'SAR', 'SAR', '抛物线转向指标'),
        ('indicators.adx', 'ADX', 'ADX', '平均趋向指标'),
        ('indicators.aroon', 'Aroon', 'AROON', 'Aroon指标'),
        ('indicators.atr', 'ATR', 'ATR', '平均真实波幅'),
        ('indicators.kc', 'KC', 'KC', '肯特纳通道'),
        ('indicators.mfi', 'MFI', 'MFI', '资金流量指标'),
        ('indicators.momentum', 'Momentum', 'MOMENTUM', '动量指标'),
        ('indicators.mtm', 'MTM', 'MTM', '动量指标'),
        ('indicators.obv', 'OBV', 'OBV', '能量潮指标'),
        ('indicators.psy', 'PSY', 'PSY', '心理线指标'),
        ('indicators.pvt', 'PVT', 'PVT', '价量趋势指标'),
        ('indicators.roc', 'ROC', 'ROC', '变动率指标'),
        ('indicators.trix', 'TRIX', 'TRIX', 'TRIX指标'),
        ('indicators.vix', 'VIX', 'VIX', '恐慌指数'),
        ('indicators.volume_ratio', 'VolumeRatio', 'VOLUME_RATIO', '量比指标'),
        ('indicators.vosc', 'VOSC', 'VOSC', '成交量震荡器'),
        ('indicators.vr', 'VR', 'VR', '成交量比率'),
        ('indicators.vortex', 'Vortex', 'VORTEX', '涡流指标'),
        ('indicators.wr', 'WR', 'WR', '威廉指标'),
        ('indicators.ad', 'AD', 'AD', '累积/派发线'),
    ]
    
    success_count = 0
    for module_path, class_name, indicator_name, description in core_indicators:
        try:
            module = importlib.import_module(module_path)
            indicator_class = getattr(module, class_name, None)
            
            if indicator_class:
                from indicators.base_indicator import BaseIndicator
                if issubclass(indicator_class, BaseIndicator):
                    registry.register_indicator(indicator_class, name=indicator_name, description=description)
                    success_count += 1
                    print(f"✅ 成功注册: {indicator_name}")
                else:
                    print(f"❌ 跳过非BaseIndicator: {class_name}")
            else:
                print(f"❌ 未找到类: {class_name}")
                
        except ImportError as e:
            print(f"❌ 导入失败: {module_path} - {e}")
        except Exception as e:
            print(f"❌ 注册失败: {indicator_name} - {e}")
    
    print(f"核心指标注册完成: {success_count}/{len(core_indicators)}")
    return success_count

def batch_register_enhanced_indicators():
    """批量注册增强指标"""
    print("\n=== 批量注册增强指标 ===")
    
    from indicators.indicator_registry import get_registry
    registry = get_registry()
    
    enhanced_indicators = [
        ('indicators.trend.enhanced_cci', 'EnhancedCCI', 'ENHANCED_CCI', '增强版CCI'),
        ('indicators.trend.enhanced_dmi', 'EnhancedDMI', 'ENHANCED_DMI', '增强版DMI'),
        ('indicators.volume.enhanced_mfi', 'EnhancedMFI', 'ENHANCED_MFI', '增强版MFI'),
        ('indicators.volume.enhanced_obv', 'EnhancedOBV', 'ENHANCED_OBV', '增强版OBV'),
        ('indicators.enhanced_rsi', 'EnhancedRSI', 'ENHANCED_RSI', '增强版RSI'),
        ('indicators.enhanced_wr', 'EnhancedWR', 'ENHANCED_WR', '增强版威廉指标'),
    ]
    
    success_count = 0
    for module_path, class_name, indicator_name, description in enhanced_indicators:
        try:
            module = importlib.import_module(module_path)
            indicator_class = getattr(module, class_name, None)
            
            if indicator_class:
                from indicators.base_indicator import BaseIndicator
                if issubclass(indicator_class, BaseIndicator):
                    registry.register_indicator(indicator_class, name=indicator_name, description=description)
                    success_count += 1
                    print(f"✅ 成功注册: {indicator_name}")
                else:
                    print(f"❌ 跳过非BaseIndicator: {class_name}")
            else:
                print(f"❌ 未找到类: {class_name}")
                
        except ImportError as e:
            print(f"❌ 导入失败: {module_path} - {e}")
        except Exception as e:
            print(f"❌ 注册失败: {indicator_name} - {e}")
    
    print(f"增强指标注册完成: {success_count}/{len(enhanced_indicators)}")
    return success_count

def batch_register_composite_indicators():
    """批量注册复合指标"""
    print("\n=== 批量注册复合指标 ===")
    
    from indicators.indicator_registry import get_registry
    registry = get_registry()
    
    composite_indicators = [
        ('indicators.composite_indicator', 'CompositeIndicator', 'COMPOSITE', '复合指标'),
        ('indicators.unified_ma', 'UnifiedMA', 'UNIFIED_MA', '统一移动平均线'),
        ('indicators.chip_distribution', 'ChipDistribution', 'CHIP_DISTRIBUTION', '筹码分布'),
        ('indicators.institutional_behavior', 'InstitutionalBehavior', 'INSTITUTIONAL_BEHAVIOR', '机构行为'),
        ('indicators.stock_vix', 'StockVIX', 'STOCK_VIX', '个股恐慌指数'),
    ]
    
    success_count = 0
    for module_path, class_name, indicator_name, description in composite_indicators:
        try:
            module = importlib.import_module(module_path)
            indicator_class = getattr(module, class_name, None)
            
            if indicator_class:
                from indicators.base_indicator import BaseIndicator
                if issubclass(indicator_class, BaseIndicator):
                    registry.register_indicator(indicator_class, name=indicator_name, description=description)
                    success_count += 1
                    print(f"✅ 成功注册: {indicator_name}")
                else:
                    print(f"❌ 跳过非BaseIndicator: {class_name}")
            else:
                print(f"❌ 未找到类: {class_name}")
                
        except ImportError as e:
            print(f"❌ 导入失败: {module_path} - {e}")
        except Exception as e:
            print(f"❌ 注册失败: {indicator_name} - {e}")
    
    print(f"复合指标注册完成: {success_count}/{len(composite_indicators)}")
    return success_count

def batch_register_pattern_indicators():
    """批量注册形态指标"""
    print("\n=== 批量注册形态指标 ===")
    
    from indicators.indicator_registry import get_registry
    registry = get_registry()
    
    pattern_indicators = [
        ('indicators.pattern.candlestick_patterns', 'CandlestickPatterns', 'CANDLESTICK_PATTERNS', 'K线形态'),
        ('indicators.pattern.advanced_candlestick_patterns', 'AdvancedCandlestickPatterns', 'ADVANCED_CANDLESTICK', '高级K线形态'),
    ]
    
    success_count = 0
    for module_path, class_name, indicator_name, description in pattern_indicators:
        try:
            module = importlib.import_module(module_path)
            indicator_class = getattr(module, class_name, None)
            
            if indicator_class:
                from indicators.base_indicator import BaseIndicator
                if issubclass(indicator_class, BaseIndicator):
                    registry.register_indicator(indicator_class, name=indicator_name, description=description)
                    success_count += 1
                    print(f"✅ 成功注册: {indicator_name}")
                else:
                    print(f"❌ 跳过非BaseIndicator: {class_name}")
            else:
                print(f"❌ 未找到类: {class_name}")
                
        except ImportError as e:
            print(f"❌ 导入失败: {module_path} - {e}")
        except Exception as e:
            print(f"❌ 注册失败: {indicator_name} - {e}")
    
    print(f"形态指标注册完成: {success_count}/{len(pattern_indicators)}")
    return success_count

def batch_register_tool_indicators():
    """批量注册工具指标"""
    print("\n=== 批量注册工具指标 ===")
    
    from indicators.indicator_registry import get_registry
    registry = get_registry()
    
    tool_indicators = [
        ('indicators.fibonacci_tools', 'FibonacciTools', 'FIBONACCI_TOOLS', '斐波那契工具'),
        ('indicators.gann_tools', 'GannTools', 'GANN_TOOLS', '江恩工具'),
        ('indicators.elliott_wave', 'ElliottWave', 'ELLIOTT_WAVE', '艾略特波浪'),
    ]
    
    success_count = 0
    for module_path, class_name, indicator_name, description in tool_indicators:
        try:
            module = importlib.import_module(module_path)
            indicator_class = getattr(module, class_name, None)
            
            if indicator_class:
                from indicators.base_indicator import BaseIndicator
                if issubclass(indicator_class, BaseIndicator):
                    registry.register_indicator(indicator_class, name=indicator_name, description=description)
                    success_count += 1
                    print(f"✅ 成功注册: {indicator_name}")
                else:
                    print(f"❌ 跳过非BaseIndicator: {class_name}")
            else:
                print(f"❌ 未找到类: {class_name}")
                
        except ImportError as e:
            print(f"❌ 导入失败: {module_path} - {e}")
        except Exception as e:
            print(f"❌ 注册失败: {indicator_name} - {e}")
    
    print(f"工具指标注册完成: {success_count}/{len(tool_indicators)}")
    return success_count

def main():
    """主函数"""
    print("开始批量注册未注册指标...")
    
    # 获取注册前的指标数量
    from indicators.indicator_registry import get_registry
    registry = get_registry()
    before_count = len(registry.get_indicator_names())
    print(f"注册前指标数量: {before_count}")
    
    # 批量注册各类指标
    core_success = batch_register_core_indicators()
    enhanced_success = batch_register_enhanced_indicators()
    composite_success = batch_register_composite_indicators()
    pattern_success = batch_register_pattern_indicators()
    tool_success = batch_register_tool_indicators()
    
    # 获取注册后的指标数量
    after_count = len(registry.get_indicator_names())
    new_registered = after_count - before_count
    
    # 总结
    total_success = core_success + enhanced_success + composite_success + pattern_success + tool_success
    print(f"\n=== 批量注册总结 ===")
    print(f"注册前指标数量: {before_count}")
    print(f"注册后指标数量: {after_count}")
    print(f"新增注册指标: {new_registered}")
    print(f"尝试注册指标: {23 + 6 + 5 + 2 + 3}")  # 各类指标总数
    print(f"成功注册指标: {total_success}")
    print(f"成功率: {total_success/(23 + 6 + 5 + 2 + 3)*100:.1f}%")
    
    if after_count >= 60:  # 目标是至少60个指标
        print("✅ 批量注册成功！指标数量达到预期目标")
        return True
    else:
        print("⚠️  批量注册部分成功，但未达到预期目标")
        return False

if __name__ == "__main__":
    main()
