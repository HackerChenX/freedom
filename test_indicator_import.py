#!/usr/bin/env python3
"""
测试指标导入问题
"""
import sys
import traceback

def test_indicator_import():
    """测试指标导入"""
    print("=== 测试指标导入问题 ===")
    
    # 测试失败的指标
    failed_indicators = ['ENHANCED_TRIX', 'TRIX', 'SAR', 'ROC_OSCILLATOR']
    
    try:
        print("1. 导入指标注册表...")
        from indicators.complete_indicator_registry import get_indicator_registry
        
        print("2. 获取注册表实例...")
        registry = get_indicator_registry()
        
        print("3. 测试指标获取...")
        for indicator_name in failed_indicators:
            try:
                print(f"   测试 {indicator_name}...")
                indicator = registry.get_indicator(indicator_name)
                if indicator:
                    print(f"   ✅ {indicator_name} 获取成功: {type(indicator)}")
                    
                    # 测试基本计算
                    try:
                        import pandas as pd
                        import numpy as np
                        
                        # 创建测试数据
                        test_data = pd.DataFrame({
                            'open': np.random.uniform(10, 20, 100),
                            'high': np.random.uniform(15, 25, 100),
                            'low': np.random.uniform(5, 15, 100),
                            'close': np.random.uniform(10, 20, 100),
                            'volume': np.random.uniform(1000, 10000, 100)
                        })
                        
                        print(f"   测试 {indicator_name} 计算...")
                        result = indicator.calculate(test_data)
                        print(f"   ✅ {indicator_name} 计算成功: {type(result)}, shape: {result.shape if hasattr(result, 'shape') else 'N/A'}")
                        
                    except Exception as calc_e:
                        print(f"   ❌ {indicator_name} 计算失败: {calc_e}")
                        traceback.print_exc()
                        
                else:
                    print(f"   ❌ {indicator_name} 获取失败: 返回None")
                    
            except Exception as e:
                print(f"   ❌ {indicator_name} 异常: {e}")
                traceback.print_exc()
                print("   ---")
                
    except Exception as e:
        print(f"注册表导入失败: {e}")
        traceback.print_exc()

def test_direct_import():
    """测试直接导入指标类"""
    print("\n=== 测试直接导入指标类 ===")
    
    indicator_imports = [
        ('ENHANCED_TRIX', 'indicators.trend.enhanced_trix', 'EnhancedTRIX'),
        ('TRIX', 'indicators.trix', 'TRIX'),
        ('SAR', 'indicators.sar', 'SAR'),
        ('ROC_OSCILLATOR', 'indicators.roc', 'ROC'),  # ROC_OSCILLATOR是ROC的别名
    ]
    
    for indicator_name, module_path, class_name in indicator_imports:
        try:
            print(f"测试直接导入 {indicator_name} ({module_path}.{class_name})...")
            
            # 动态导入模块
            module = __import__(module_path, fromlist=[class_name])
            indicator_class = getattr(module, class_name)
            
            # 创建实例
            indicator = indicator_class()
            print(f"✅ {indicator_name} 直接导入成功: {type(indicator)}")
            
        except Exception as e:
            print(f"❌ {indicator_name} 直接导入失败: {e}")
            traceback.print_exc()
            print("---")

if __name__ == "__main__":
    test_indicator_import()
    test_direct_import()
