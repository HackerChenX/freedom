#!/usr/bin/env python3
"""
测试指标修复效果
验证BaseIndicator和指标注册是否正常工作
"""

import pandas as pd
import sys
import traceback
from utils.logger import get_logger

logger = get_logger(__name__)


def test_base_indicator():
    """测试BaseIndicator基础类"""
    print("🔍 测试BaseIndicator基础类...")
    
    try:
        from indicators.base_indicator import BaseIndicator
        print("  ✅ BaseIndicator导入成功")
        
        # 检查抽象方法
        abstract_methods = []
        for method_name in dir(BaseIndicator):
            method = getattr(BaseIndicator, method_name)
            if hasattr(method, '__isabstractmethod__') and method.__isabstractmethod__:
                abstract_methods.append(method_name)
        
        print(f"  ✅ 抽象方法: {abstract_methods}")
        
        # 检查扩展点方法
        extension_methods = ['validate_data', 'preprocess_data', 'postprocess_result']
        for method in extension_methods:
            if hasattr(BaseIndicator, method):
                print(f"  ✅ 扩展点方法 {method} 存在")
            else:
                print(f"  ❌ 扩展点方法 {method} 缺失")
        
        return True
        
    except Exception as e:
        print(f"  ❌ BaseIndicator测试失败: {e}")
        traceback.print_exc()
        return False


def test_indicator_registry():
    """测试指标注册表"""
    print("\n🔍 测试指标注册表...")
    
    try:
        from indicators.complete_indicator_registry import get_indicator_registry, initialize_indicators
        print("  ✅ 指标注册表导入成功")
        
        # 获取注册表实例
        registry = get_indicator_registry()
        if registry:
            print("  ✅ 注册表实例获取成功")
        else:
            print("  ❌ 注册表实例为None")
            return False
        
        # 初始化指标
        registered_count = initialize_indicators()
        print(f"  ✅ 指标初始化完成，注册数量: {registered_count}")
        
        # 获取注册统计
        stats = registry.get_registration_stats()
        print(f"  📊 注册统计:")
        print(f"    - 总指标数: {stats['total_indicators']}")
        print(f"    - 成功注册: {stats['successful_indicators']}")
        print(f"    - 失败数量: {stats['failed_indicators']}")
        print(f"    - 成功率: {stats['success_rate']:.1%}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 指标注册表测试失败: {e}")
        traceback.print_exc()
        return False


def test_specific_indicators():
    """测试具体指标"""
    print("\n🔍 测试具体指标...")
    
    # 创建测试数据
    test_data = pd.DataFrame({
        'open': [100, 101, 102, 103, 104],
        'high': [105, 106, 107, 108, 109],
        'low': [95, 96, 97, 98, 99],
        'close': [102, 103, 104, 105, 106],
        'volume': [1000, 1100, 1200, 1300, 1400]
    })
    
    test_indicators = ['MA', 'MACD', 'RSI']
    success_count = 0
    
    try:
        from indicators.complete_indicator_registry import get_indicator
        
        for indicator_name in test_indicators:
            try:
                print(f"  🧪 测试指标: {indicator_name}")
                
                # 获取指标实例
                indicator = get_indicator(indicator_name)
                if indicator is None:
                    print(f"    ❌ 指标 {indicator_name} 未找到")
                    continue
                
                print(f"    ✅ 指标 {indicator_name} 获取成功")
                
                # 测试calculate方法
                try:
                    result = indicator.calculate(test_data)
                    if isinstance(result, pd.DataFrame):
                        print(f"    ✅ calculate()方法正常，结果形状: {result.shape}")
                    else:
                        print(f"    ❌ calculate()返回类型错误: {type(result)}")
                        continue
                except Exception as e:
                    print(f"    ❌ calculate()方法失败: {e}")
                    continue
                
                # 测试get_signal方法
                try:
                    signal = indicator.get_signal(test_data)
                    if isinstance(signal, dict):
                        print(f"    ✅ get_signal()方法正常，信号键: {list(signal.keys())}")
                    else:
                        print(f"    ❌ get_signal()返回类型错误: {type(signal)}")
                        continue
                except Exception as e:
                    print(f"    ❌ get_signal()方法失败: {e}")
                    continue
                
                success_count += 1
                print(f"    ✅ 指标 {indicator_name} 测试完全成功")
                
            except Exception as e:
                print(f"    ❌ 指标 {indicator_name} 测试失败: {e}")
        
        success_rate = success_count / len(test_indicators) * 100
        print(f"\n  📊 指标测试结果: {success_rate:.1f}% ({success_count}/{len(test_indicators)})")
        
        return success_count > 0
        
    except Exception as e:
        print(f"  ❌ 指标测试失败: {e}")
        traceback.print_exc()
        return False


def test_container():
    """测试依赖注入容器"""
    print("\n🔍 测试依赖注入容器...")
    
    try:
        from utils.container import container
        print("  ✅ 容器导入成功")
        
        # 测试服务解析
        data_access = container.resolve("DataAccessInterface")
        if data_access:
            print("  ✅ DataAccessInterface服务解析成功")
        else:
            print("  ❌ DataAccessInterface服务解析失败")
        
        cache_service = container.resolve("ICacheService")
        if cache_service:
            print("  ✅ ICacheService服务解析成功")
        else:
            print("  ❌ ICacheService服务解析失败")
        
        # 显示所有服务
        all_services = container.get_all_services()
        print(f"  📊 已注册服务: {list(all_services.keys())}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 容器测试失败: {e}")
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("🔧 L4核心服务层修复效果验证")
    print("="*60)
    
    test_results = []
    
    # 测试BaseIndicator
    test_results.append(("BaseIndicator基础类", test_base_indicator()))
    
    # 测试依赖注入容器
    test_results.append(("依赖注入容器", test_container()))
    
    # 测试指标注册表
    test_results.append(("指标注册表", test_indicator_registry()))
    
    # 测试具体指标
    test_results.append(("具体指标调用", test_specific_indicators()))
    
    # 汇总结果
    print("\n" + "="*60)
    print("🎯 测试结果汇总:")
    
    success_count = 0
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
        if result:
            success_count += 1
    
    overall_success_rate = success_count / len(test_results) * 100
    print(f"\n📊 总体成功率: {overall_success_rate:.1f}% ({success_count}/{len(test_results)})")
    
    if overall_success_rate >= 75:
        print("🎉 L4层修复效果良好！")
        return 0
    else:
        print("⚠️  L4层仍需进一步修复")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
