#!/usr/bin/env python3
"""
L1基础设施层容器修复验证测试
验证容器统一修复是否成功
"""

import sys
import os
import time

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_container_uniqueness():
    """测试容器唯一性"""
    print("🔍 测试容器唯一性...")
    
    # 检查废弃容器文件是否已删除
    deprecated_files = [
        'db/container.py',
        'utils/optimized_dependency_injection.py'
    ]
    
    for file_path in deprecated_files:
        if os.path.exists(file_path):
            print(f"❌ 废弃容器文件仍存在: {file_path}")
            return False
    
    # 检查标准容器文件是否存在
    if not os.path.exists('utils/unified_container.py'):
        print("❌ 标准容器文件不存在: utils/unified_container.py")
        return False
    
    print("✅ 容器唯一性验证通过")
    return True

def test_container_functionality():
    """测试容器功能"""
    print("🔍 测试容器功能...")
    
    try:
        from utils.unified_container import UnifiedServiceContainer, ServiceLifecycle
        
        # 创建容器实例
        container = UnifiedServiceContainer()
        
        # 测试服务注册
        class MockService:
            def test_method(self):
                return "test_success"
        
        # 注册测试服务
        container.register(MockService, MockService, lifecycle=ServiceLifecycle.SINGLETON)
        
        # 解析测试服务
        service = container.resolve(MockService)
        assert service is not None
        assert service.test_method() == "test_success"
        
        print("✅ 容器功能验证通过")
        return True
        
    except Exception as e:
        print(f"❌ 容器功能测试失败: {e}")
        return False

def test_sql_manager_fix():
    """测试SQL管理器修复"""
    print("🔍 测试SQL管理器修复...")
    
    try:
        from db.sql_manager import QueryType
        
        # 检查是否还有重复的STOCK_LIST
        query_types = list(QueryType)
        stock_list_count = sum(1 for qt in query_types if qt.value == "stock_list")
        
        if stock_list_count > 1:
            print(f"❌ 仍存在重复的STOCK_LIST定义: {stock_list_count}个")
            return False
        
        # 检查STOCK_LIST是否存在
        if QueryType.STOCK_LIST not in query_types:
            print("❌ STOCK_LIST枚举值不存在")
            return False
        
        print("✅ SQL管理器修复验证通过")
        return True
        
    except Exception as e:
        print(f"❌ SQL管理器测试失败: {e}")
        return False

def test_no_import_errors():
    """测试无导入错误"""
    print("🔍 测试导入错误...")
    
    try:
        # 测试主要模块导入
        from utils.unified_container import UnifiedServiceContainer
        from db.sql_manager import SQLManager, QueryType
        
        print("✅ 导入测试通过")
        return True
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 其他错误: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始L1基础设施层容器修复验证...")
    print("=" * 50)
    
    tests = [
        test_container_uniqueness,
        test_container_functionality,
        test_sql_manager_fix,
        test_no_import_errors
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ 测试失败: {test_func.__name__}")
        except Exception as e:
            print(f"❌ 测试异常: {test_func.__name__} - {e}")
        
        print("-" * 30)
    
    print(f"\n📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 L1.1任务 - 依赖注入容器统一修复成功！")
        return True
    else:
        print("🚫 L1.1任务修复失败，需要进一步处理")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
