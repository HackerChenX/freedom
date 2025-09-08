#!/usr/bin/env python3
"""
追踪"real"模块导入错误的来源
"""
import sys
import traceback
import importlib

def test_step_by_step_import():
    """逐步测试导入过程"""
    print("=== 逐步测试导入过程 ===")
    
    # 测试指标信息
    indicators = [
        {
            'name': 'ENHANCED_TRIX',
            'module': 'indicators.trend.enhanced_trix',
            'class': 'EnhancedTrix'
        },
        {
            'name': 'TRIX', 
            'module': 'indicators.trix',
            'class': 'TripleExponentialAverage'
        },
        {
            'name': 'SAR',
            'module': 'indicators.sar', 
            'class': 'Sar'
        },
        {
            'name': 'ROC_OSCILLATOR',
            'module': 'indicators.roc',
            'class': 'RateOfChange'
        }
    ]
    
    for indicator_info in indicators:
        print(f"\n--- 测试 {indicator_info['name']} ---")
        
        try:
            # 步骤1: 导入模块
            print(f"1. 导入模块: {indicator_info['module']}")
            module = importlib.import_module(indicator_info['module'])
            print(f"   ✅ 模块导入成功")
            
            # 步骤2: 获取类
            print(f"2. 获取类: {indicator_info['class']}")
            if hasattr(module, indicator_info['class']):
                indicator_class = getattr(module, indicator_info['class'])
                print(f"   ✅ 类获取成功: {indicator_class}")
                
                # 步骤3: 创建实例
                print(f"3. 创建实例")
                try:
                    indicator = indicator_class()
                    print(f"   ✅ 实例创建成功: {type(indicator)}")
                    
                    # 步骤4: 测试基本方法
                    print(f"4. 测试基本方法")
                    if hasattr(indicator, 'calculate'):
                        print(f"   ✅ 有calculate方法")
                    else:
                        print(f"   ❌ 没有calculate方法")
                        
                except Exception as e:
                    print(f"   ❌ 实例创建失败: {e}")
                    print(f"   错误类型: {type(e)}")
                    traceback.print_exc()
                    
            else:
                print(f"   ❌ 类不存在，可用的类: {[attr for attr in dir(module) if not attr.startswith('_')]}")
                
        except Exception as e:
            print(f"   ❌ 模块导入失败: {e}")
            traceback.print_exc()

def test_import_hook():
    """使用导入钩子追踪"real"模块的导入"""
    print("\n=== 使用导入钩子追踪 ===")
    
    original_import = __builtins__.__import__
    
    def trace_import(name, globals=None, locals=None, fromlist=(), level=0):
        if 'real' in name:
            print(f"🔍 发现real模块导入: {name}")
            print(f"   调用栈:")
            for line in traceback.format_stack():
                print(f"     {line.strip()}")
        return original_import(name, globals, locals, fromlist, level)
    
    __builtins__.__import__ = trace_import
    
    try:
        print("测试TRIX指标导入...")
        from indicators.trix import TripleExponentialAverage
        indicator = TripleExponentialAverage()
        print("✅ TRIX指标创建成功")
    except Exception as e:
        print(f"❌ TRIX指标创建失败: {e}")
    finally:
        __builtins__.__import__ = original_import

if __name__ == "__main__":
    test_step_by_step_import()
    test_import_hook()
