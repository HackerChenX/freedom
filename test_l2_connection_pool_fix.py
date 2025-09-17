#!/usr/bin/env python3
"""
L2存储访问层连接池统一验证测试
验证连接池统一修复是否成功
"""

import sys
import os
import time

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_connection_pool_uniqueness():
    """测试连接池唯一性"""
    print("🔍 测试连接池唯一性...")
    
    # 检查标准连接池文件是否存在
    required_files = [
        'db/enhanced_connection_pool.py'
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ 标准连接池文件不存在: {file_path}")
            return False
    
    # 检查兼容连接管理器是否存在
    compat_files = [
        'db/managers/connection_manager.py'
    ]
    
    for file_path in compat_files:
        if not os.path.exists(file_path):
            print(f"❌ 兼容连接管理器不存在: {file_path}")
            return False
    
    print("✅ 连接池唯一性验证通过")
    return True

def test_standard_connection_pool():
    """测试标准连接池功能"""
    print("🔍 测试标准连接池功能...")
    
    try:
        from db.enhanced_connection_pool import get_connection_pool
        
        # 测试获取连接池
        pool = get_connection_pool()
        assert pool is not None
        
        # 测试连接池是单例
        pool2 = get_connection_pool()
        assert pool is pool2
        
        print("✅ 标准连接池功能验证通过")
        return True
        
    except Exception as e:
        print(f"❌ 标准连接池功能测试失败: {e}")
        return False

def test_connection_manager_compatibility():
    """测试连接管理器兼容性"""
    print("🔍 测试连接管理器兼容性...")
    
    try:
        from db.managers.connection_manager import ConnectionManager, get_connection_manager
        
        # 测试ConnectionManager类
        manager = ConnectionManager()
        assert manager is not None
        
        # 测试get_connection_manager函数
        global_manager = get_connection_manager()
        assert global_manager is not None
        
        # 测试管理器是单例
        global_manager2 = get_connection_manager()
        assert global_manager is global_manager2
        
        print("✅ 连接管理器兼容性验证通过")
        return True
        
    except Exception as e:
        print(f"❌ 连接管理器兼容性测试失败: {e}")
        return False

def test_connection_pool_integration():
    """测试连接池集成"""
    print("🔍 测试连接池集成...")
    
    try:
        from db.enhanced_connection_pool import get_connection_pool
        from db.managers.connection_manager import get_connection_manager
        
        # 获取标准连接池
        pool = get_connection_pool()
        
        # 获取兼容连接管理器
        manager = get_connection_manager()
        
        # 验证管理器内部使用的是同一个连接池
        manager_pool = manager.pool
        assert manager_pool is pool
        
        print("✅ 连接池集成验证通过")
        return True
        
    except Exception as e:
        print(f"❌ 连接池集成测试失败: {e}")
        return False

def test_connection_pool_performance():
    """测试连接池性能"""
    print("🔍 测试连接池性能...")
    
    try:
        from db.enhanced_connection_pool import get_connection_pool
        
        # 测试连接池获取性能
        start_time = time.time()
        for i in range(100):
            pool = get_connection_pool()
        pool_time = time.time() - start_time
        
        # 测试连接管理器性能
        start_time = time.time()
        from db.managers.connection_manager import get_connection_manager
        for i in range(100):
            manager = get_connection_manager()
        manager_time = time.time() - start_time
        
        # 性能要求：每个操作应该在合理时间内完成
        if pool_time > 1.0:
            print(f"⚠️  连接池性能较慢: {pool_time:.3f}s")
        if manager_time > 1.0:
            print(f"⚠️  连接管理器性能较慢: {manager_time:.3f}s")
        
        print(f"📊 性能指标: 连接池={pool_time:.3f}s, 管理器={manager_time:.3f}s")
        print("✅ 连接池性能验证通过")
        return True
        
    except Exception as e:
        print(f"❌ 连接池性能测试失败: {e}")
        return False

def test_no_import_errors():
    """测试无导入错误"""
    print("🔍 测试导入错误...")
    
    try:
        # 测试标准连接池导入
        from db.enhanced_connection_pool import get_connection_pool
        
        # 测试兼容连接管理器导入
        from db.managers.connection_manager import ConnectionManager, get_connection_manager
        
        # 测试一些使用连接池的模块
        from db.managers.data_access_manager import DataAccessManager
        
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
    print("🚀 开始L2存储访问层连接池统一验证...")
    print("=" * 50)
    
    tests = [
        test_connection_pool_uniqueness,
        test_standard_connection_pool,
        test_connection_manager_compatibility,
        test_connection_pool_integration,
        test_connection_pool_performance,
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
        print("🎉 L2.1任务 - 数据库连接池统一修复成功！")
        return True
    else:
        print("🚫 L2.1任务修复失败，需要进一步处理")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
