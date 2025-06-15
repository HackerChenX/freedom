#!/usr/bin/env python3
"""
简化的指标测试脚本
直接测试指标导入和基本功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_basic_imports():
    """测试基础导入"""
    print("=== 测试基础导入 ===")
    
    try:
        print("1. 测试pandas...")
        import pandas as pd
        print("✅ pandas导入成功")
        
        print("2. 测试numpy...")
        import numpy as np
        print("✅ numpy导入成功")
        
        print("3. 测试BaseIndicator...")
        from indicators.base_indicator import BaseIndicator
        print("✅ BaseIndicator导入成功")
        
        print("4. 测试单个指标...")
        from indicators.macd import MACD
        print("✅ MACD导入成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 基础导入失败: {e}")
        return False

def test_core_indicators():
    """测试核心指标导入"""
    print("\n=== 测试核心指标导入 ===")
    
    core_indicators = [
        ('indicators.ma', 'MA'),
        ('indicators.ema', 'EMA'),
        ('indicators.wma', 'WMA'),
        ('indicators.sar', 'SAR'),
        ('indicators.adx', 'ADX'),
        ('indicators.aroon', 'Aroon'),
        ('indicators.atr', 'ATR'),
        ('indicators.kc', 'KC'),
        ('indicators.mfi', 'MFI'),
        ('indicators.momentum', 'Momentum'),
        ('indicators.mtm', 'MTM'),
        ('indicators.obv', 'OBV'),
        ('indicators.psy', 'PSY'),
        ('indicators.pvt', 'PVT'),
        ('indicators.roc', 'ROC'),
        ('indicators.trix', 'TRIX'),
        ('indicators.vix', 'VIX'),
        ('indicators.volume_ratio', 'VolumeRatio'),
        ('indicators.vosc', 'VOSC'),
        ('indicators.vr', 'VR'),
        ('indicators.vortex', 'Vortex'),
        ('indicators.wr', 'WR'),
        ('indicators.ad', 'AD'),
    ]
    
    successful = 0
    failed = 0
    
    for module_path, class_name in core_indicators:
        try:
            import importlib
            module = importlib.import_module(module_path)
            indicator_class = getattr(module, class_name, None)
            
            if indicator_class:
                from indicators.base_indicator import BaseIndicator
                if issubclass(indicator_class, BaseIndicator):
                    print(f"✅ {class_name}: 导入成功")
                    successful += 1
                else:
                    print(f"❌ {class_name}: 不是BaseIndicator子类")
                    failed += 1
            else:
                print(f"❌ {class_name}: 类不存在")
                failed += 1
                
        except ImportError as e:
            print(f"❌ {class_name}: 导入失败 - {e}")
            failed += 1
        except Exception as e:
            print(f"❌ {class_name}: 其他错误 - {e}")
            failed += 1
    
    print(f"\n核心指标测试结果: {successful} 成功, {failed} 失败")
    return successful, failed

def test_enhanced_indicators():
    """测试增强指标导入"""
    print("\n=== 测试增强指标导入 ===")
    
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
    
    successful = 0
    failed = 0
    
    for module_path, class_name in enhanced_indicators:
        try:
            import importlib
            module = importlib.import_module(module_path)
            indicator_class = getattr(module, class_name, None)
            
            if indicator_class:
                from indicators.base_indicator import BaseIndicator
                if issubclass(indicator_class, BaseIndicator):
                    print(f"✅ {class_name}: 导入成功")
                    successful += 1
                else:
                    print(f"❌ {class_name}: 不是BaseIndicator子类")
                    failed += 1
            else:
                print(f"❌ {class_name}: 类不存在")
                failed += 1
                
        except ImportError as e:
            print(f"❌ {class_name}: 导入失败 - {e}")
            failed += 1
        except Exception as e:
            print(f"❌ {class_name}: 其他错误 - {e}")
            failed += 1
    
    print(f"\n增强指标测试结果: {successful} 成功, {failed} 失败")
    return successful, failed

def test_composite_indicators():
    """测试复合指标导入"""
    print("\n=== 测试复合指标导入 ===")
    
    composite_indicators = [
        ('indicators.composite_indicator', 'CompositeIndicator'),
        ('indicators.unified_ma', 'UnifiedMA'),
        ('indicators.chip_distribution', 'ChipDistribution'),
        ('indicators.institutional_behavior', 'InstitutionalBehavior'),
        ('indicators.stock_vix', 'StockVIX'),
    ]
    
    successful = 0
    failed = 0
    
    for module_path, class_name in composite_indicators:
        try:
            import importlib
            module = importlib.import_module(module_path)
            indicator_class = getattr(module, class_name, None)
            
            if indicator_class:
                from indicators.base_indicator import BaseIndicator
                if issubclass(indicator_class, BaseIndicator):
                    print(f"✅ {class_name}: 导入成功")
                    successful += 1
                else:
                    print(f"❌ {class_name}: 不是BaseIndicator子类")
                    failed += 1
            else:
                print(f"❌ {class_name}: 类不存在")
                failed += 1
                
        except ImportError as e:
            print(f"❌ {class_name}: 导入失败 - {e}")
            failed += 1
        except Exception as e:
            print(f"❌ {class_name}: 其他错误 - {e}")
            failed += 1
    
    print(f"\n复合指标测试结果: {successful} 成功, {failed} 失败")
    return successful, failed

def main():
    """主函数"""
    print("开始简化指标测试...\n")
    
    # 测试基础导入
    if not test_basic_imports():
        print("❌ 基础导入失败，无法继续测试")
        return False
    
    # 测试各类指标
    core_success, core_failed = test_core_indicators()
    enhanced_success, enhanced_failed = test_enhanced_indicators()
    composite_success, composite_failed = test_composite_indicators()
    
    # 总结
    total_success = core_success + enhanced_success + composite_success
    total_failed = core_failed + enhanced_failed + composite_failed
    total_tested = total_success + total_failed
    
    print(f"\n=== 总结 ===")
    print(f"总测试指标: {total_tested}")
    print(f"成功导入: {total_success}")
    print(f"导入失败: {total_failed}")
    print(f"成功率: {(total_success/total_tested*100):.1f}%" if total_tested > 0 else "0%")
    
    if total_success >= 30:  # 目标是至少30个指标能成功导入
        print("✅ 指标导入测试通过！")
        return True
    else:
        print("❌ 指标导入测试未达到预期")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
