#!/usr/bin/env python3
"""
简化架构修复验证脚本
"""

import sys
import os

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

def test_basic_imports():
    """测试基本导入"""
    print("🔍 测试基本导入...")
    
    try:
        from utils.dependency_injection import get_service
        from strategy.unified_base_strategy import UnifiedBaseStrategy
        print("✅ 基本导入成功")
        return True
    except ImportError as e:
        print(f"❌ 基本导入失败: {e}")
        return False

def test_service_initialization():
    """测试服务初始化"""
    print("🔍 测试服务初始化...")
    
    try:
        from config.service_initializer import initialize_all_services
        container = initialize_all_services()
        print("✅ 服务初始化成功")
        return True
    except Exception as e:
        print(f"❌ 服务初始化失败: {e}")
        return False

def test_unified_base_strategy():
    """测试统一基类"""
    print("🔍 测试统一基类...")
    
    try:
        from strategy.unified_base_strategy import UnifiedBaseStrategy
        
        class TestStrategy(UnifiedBaseStrategy):
            def select_stocks(self, universe, start_date, end_date, **kwargs):
                import pandas as pd
                return pd.DataFrame({'code': universe[:3], 'score': [100, 90, 80]})
        
        strategy = TestStrategy("测试策略")
        info = strategy.get_info()
        
        assert info['name'] == "测试策略"
        print("✅ 统一基类测试成功")
        return True
        
    except Exception as e:
        print(f"❌ 统一基类测试失败: {e}")
        return False

def main_simple_verification():
    """主函数"""
    print("🚀 开始简化架构验证...\n")
    
    tests = [
        ("基本导入测试", test_basic_imports),
        ("服务初始化测试", test_service_initialization),
        ("统一基类测试", test_unified_base_strategy)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)
        
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} 执行失败: {e}")
    
    print(f"\n📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 架构验证通过！")
        return 0
    else:
        print("⚠️  部分测试失败")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
