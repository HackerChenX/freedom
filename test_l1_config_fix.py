#!/usr/bin/env python3
"""
L1基础设施层配置管理统一验证测试
验证配置管理系统统一是否成功
"""

import sys
import os
import time

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_config_file_uniqueness():
    """测试配置文件唯一性"""
    print("🔍 测试配置文件唯一性...")
    
    # 检查废弃配置文件是否已删除
    deprecated_files = [
        'backup/deprecated_entries/20250916/config_legacy.py.bak',
        'backup/deprecated_entries/20250916/database_config_manager_legacy.py.bak'
    ]
    
    for file_path in deprecated_files:
        if not os.path.exists(file_path):
            print(f"⚠️  废弃配置文件备份不存在: {file_path}")
    
    # 检查标准配置文件是否存在
    required_files = [
        'config/config.py',
        'config/unified_config_manager.py',
        'config/database_config_manager.py'
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ 必需配置文件不存在: {file_path}")
            return False
    
    print("✅ 配置文件唯一性验证通过")
    return True

def test_config_api_compatibility():
    """测试配置API兼容性"""
    print("🔍 测试配置API兼容性...")
    
    try:
        # 测试主配置API
        from config.unified_config_manager import get_config, get_database_config
        
        # 测试数据库配置API
        from config.database_config_manager import DatabaseConfigManager
        
        # 测试统一配置管理器
        from config.unified_config_manager import UnifiedConfigManager
        
        print("✅ 配置API兼容性验证通过")
        return True
        
    except ImportError as e:
        print(f"❌ 配置API导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 配置API测试失败: {e}")
        return False

def test_config_functionality():
    """测试配置功能"""
    print("🔍 测试配置功能...")
    
    try:
        from config.unified_config_manager import get_config, get_database_config
        
        # 测试获取数据库配置
        db_config = get_database_config()
        assert isinstance(db_config, dict)
        assert 'host' in db_config
        assert 'port' in db_config
        assert 'database' in db_config
        
        # 测试获取通用配置
        host = get_config('database.host', 'localhost')
        assert host is not None
        
        print("✅ 配置功能验证通过")
        return True
        
    except Exception as e:
        print(f"❌ 配置功能测试失败: {e}")
        return False

def test_database_config_manager():
    """测试数据库配置管理器"""
    print("🔍 测试数据库配置管理器...")
    
    try:
        from config.database_config_manager import DatabaseConfigManager, database_config_manager
        
        # 测试类实例化
        manager = DatabaseConfigManager()
        config = manager.get_config()
        assert isinstance(config, dict)
        
        # 测试全局实例
        global_config = database_config_manager.get_config()
        assert isinstance(global_config, dict)
        
        # 测试静态方法
        static_config = DatabaseConfigManager.get_database_config()
        assert isinstance(static_config, dict)
        
        print("✅ 数据库配置管理器验证通过")
        return True
        
    except Exception as e:
        print(f"❌ 数据库配置管理器测试失败: {e}")
        return False

def test_unified_config_manager():
    """测试统一配置管理器"""
    print("🔍 测试统一配置管理器...")
    
    try:
        from config.unified_config_manager import UnifiedConfigManager
        
        # 创建管理器实例
        manager = UnifiedConfigManager()
        
        # 测试基本功能
        manager.set('test.key', 'test_value')
        value = manager.get('test.key')
        assert value == 'test_value'
        
        # 测试默认值
        default_value = manager.get('non.existent.key', 'default')
        assert default_value == 'default'
        
        print("✅ 统一配置管理器验证通过")
        return True
        
    except Exception as e:
        print(f"❌ 统一配置管理器测试失败: {e}")
        return False

def test_no_import_errors():
    """测试无导入错误"""
    print("🔍 测试导入错误...")
    
    try:
        # 测试主要配置模块导入
        from config.unified_config_manager import get_config, get_database_config
        from config.database_config_manager import DatabaseConfigManager
        from config.unified_config_manager import UnifiedConfigManager
        
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
    print("🚀 开始L1基础设施层配置管理统一验证...")
    print("=" * 50)
    
    tests = [
        test_config_file_uniqueness,
        test_config_api_compatibility,
        test_config_functionality,
        test_database_config_manager,
        test_unified_config_manager,
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
        print("🎉 L1.2任务 - 配置管理系统统一修复成功！")
        return True
    else:
        print("🚫 L1.2任务修复失败，需要进一步处理")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
