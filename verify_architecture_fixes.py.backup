#!/usr/bin/env python3
"""
架构修复验证脚本

验证所有主要架构修复是否有效
"""

import sys
import os

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

def test_imports():
    """测试关键导入是否正常"""
    print("🔍 测试关键导入...")
    
    try:
        # 测试统一基类
        from strategy.unified_base_strategy import UnifiedBaseStrategy
        from strategy.base_strategy import BaseStrategy
        from strategy.enhanced_base_strategy import EnhancedBaseStrategy
        print("✅ 策略基类导入成功")
        
        # 测试依赖注入
        from utils.dependency_injection import get_service
        from db.interfaces.data_access_interface import DataAccessInterface
        print("✅ 依赖注入系统导入成功")
        
        # 测试修复后的工厂
        from indicators.factory import IndicatorFactory
        print("✅ 指标工厂导入成功")
        
        # 测试策略管理器
        from strategy.strategy_manager import StrategyManager
        print("✅ 策略管理器导入成功")
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_unified_base_strategy():
    """测试统一基类功能"""
    print("🔍 测试统一基类功能...")
    
    try:
        from strategy.unified_base_strategy import UnifiedBaseStrategy
        
        # 创建测试策略
        class TestStrategy(UnifiedBaseStrategy):
            def select_stocks(self, universe, start_date, end_date, **kwargs):
                import pandas as pd
                return pd.DataFrame({'code': universe[:5], 'score': [100, 90, 80, 70, 60]})
        
        strategy = TestStrategy("测试策略", "这是一个测试策略")
        info = strategy.get_info()
        
        assert info['name'] == "测试策略"
        assert info['default_period'] == '1d'
        
        print("✅ 统一基类功能正常")
        return True
        
    except Exception as e:
        print(f"❌ 统一基类测试失败: {e}")
        return False

def test_dependency_injection():
    """测试依赖注入是否工作"""
    print("🔍 测试依赖注入...")
    
    try:
        from config.service_initializer import initialize_all_services
        from utils.dependency_injection import get_service
        from db.interfaces.data_access_interface import DataAccessInterface
        
        # 初始化服务
        container = initialize_all_services()
        
        # 测试获取服务
        data_access = get_service(DataAccessInterface)
        print("✅ 依赖注入系统正常工作")
        return True
        
    except Exception as e:
        print(f"❌ 依赖注入测试失败: {e}")
        return False

def main_verify_architecture_fixes():
    """主函数"""
    print("🚀 开始架构修复验证...\n")
    
    tests = [
        ("导入测试", test_imports),
        ("统一基类测试", test_unified_base_strategy), 
        ("依赖注入测试", test_dependency_injection)
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
        print("🎉 所有架构修复验证通过！")
        return 0
    else:
        print("⚠️  部分测试失败，需要进一步检查")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)