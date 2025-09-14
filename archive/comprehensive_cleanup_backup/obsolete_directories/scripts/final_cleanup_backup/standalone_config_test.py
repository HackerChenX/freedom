#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
独立的数据库配置管理器测试
避免与现有config模块的冲突
"""

import sys
import os
import importlib.util
from pathlib import Path

def load_config_manager():
    """直接加载配置管理器模块"""
    project_root = Path(__file__).parent.parent
    config_manager_path = project_root / 'config' / 'database_config_manager.py'
    
    spec = importlib.util.spec_from_file_location("database_config_manager", config_manager_path)
    config_manager_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_manager_module)
    
    return config_manager_module

def test_basic_functionality():
    """测试基础功能"""
    print("🔍 测试基础功能")
    print("=" * 40)
    
    try:
        # 加载配置管理器
        config_module = load_config_manager()
        database_config_manager = config_module.Database_config_manager
        
        # 创建实例
        manager = Database_config_manager()
        
        # 测试获取配置
        config = manager.get_config()
        print(f"✅ 配置加载成功")
        print(f"   主机: {config['host']}")
        print(f"   端口: {config['port']}")
        print(f"   数据库: {config['database']}")
        print(f"   用户: {config['user']}")
        print(f"   密码: {'已设置' if config.get('password') else '未设置'}")
        
        # 测试连接配置
        conn_config = manager.get_connection_config()
        print(f"✅ 连接配置获取成功")
        
        # 测试配置验证
        if manager.validate_config():
            print("✅ 配置验证通过")
        else:
            print("❌ 配置验证失败")
        
        return True
        
    except Exception as e:
        print(f"❌ 基础功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_environment_variables():
    """测试环境变量"""
    print("\n🌍 测试环境变量")
    print("=" * 40)
    
    try:
        # 设置测试环境变量
        os.environ['CLICKHOUSE_HOST'] = 'localhost'
        os.environ['CLICKHOUSE_PORT'] = '9001'
        os.environ['CLICKHOUSE_PASSWORD'] = '123456'
        
        # 加载配置管理器
        config_module = load_config_manager()
        database_config_manager = config_module.Database_config_manager
        
        # 创建新实例
        manager = Database_config_manager()
        config = manager.get_config()
        
        # 验证环境变量
        success = True
        if config['host'] == 'env-test-host':
            print("✅ CLICKHOUSE_HOST 环境变量生效")
        else:
            print(f"❌ CLICKHOUSE_HOST 未生效: {config['host']}")
            success = False
        
        if config['port'] == 9001:
            print("✅ CLICKHOUSE_PORT 环境变量生效")
        else:
            print(f"❌ CLICKHOUSE_PORT 未生效: {config['port']}")
            success = False
        
        if config['password'] == 'env-test-password':
            print("✅ CLICKHOUSE_PASSWORD 环境变量生效")
        else:
            print(f"❌ CLICKHOUSE_PASSWORD 未生效")
            success = False
        
        return success
        
    except Exception as e:
        print(f"❌ 环境变量测试失败: {e}")
        return False
    finally:
        # 清理环境变量
        for key in ['CLICKHOUSE_HOST', 'CLICKHOUSE_PORT', 'CLICKHOUSE_PASSWORD']:
            if key in os.environ:
                del os.environ[key]

def test_password_management():
    """测试密码管理"""
    print("\n🔒 测试密码管理")
    print("=" * 40)
    
    try:
        # 加载配置管理器
        config_module = load_config_manager()
        database_config_manager = config_module.Database_config_manager
        
        # 创建实例
        manager = Database_config_manager()
        
        # 测试设置密码（不保存到文件）
        test_password = os.getenv('TEST_PASSWORD', "test_password_123")
        manager.set_password(test_password, encrypt=True, save_to_file=False)
        
        # 验证密码
        config = manager.get_config()
        if config.get('password') == test_password:
            print("✅ 密码设置成功")
        else:
            print("❌ 密码设置失败")
            return False
        
        # 测试密码加密
        encrypted = manager._encrypt_password(test_password)
        if encrypted.startswith("ENC:"):
            print("✅ 密码加密成功")
        else:
            print("❌ 密码加密失败")
            return False
        
        # 测试密码解密
        decrypted = manager._decrypt_password(encrypted)
        if decrypted == test_password:
            print("✅ 密码解密成功")
        else:
            print("❌ 密码解密失败")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 密码管理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_file_operations():
    """测试文件操作"""
    print("\n📁 测试文件操作")
    print("=" * 40)
    
    try:
        # 加载配置管理器
        config_module = load_config_manager()
        database_config_manager = config_module.Database_config_manager
        
        # 创建实例
        manager = Database_config_manager()
        
        # 测试配置文件加载
        config = manager.get_config()
        print("✅ 配置文件加载成功")
        
        # 测试密钥文件生成
        key = manager._get_encryption_key()
        if key and len(key) == 32:  # Fernet密钥长度
            print("✅ 加密密钥生成成功")
        else:
            print("❌ 加密密钥生成失败")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 文件操作测试失败: {e}")
        return False

def main_standaloneconfigtest():
    """主函数"""
    print("🧪 独立数据库配置管理器测试")
    print("=" * 60)
    
    tests = [
        ("基础功能", test_basic_functionality),
        ("环境变量", test_environment_variables),
        ("密码管理", test_password_management),
        ("文件操作", test_file_operations)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        if test_func():
            passed += 1
            print(f"✅ {test_name} - 通过")
        else:
            print(f"❌ {test_name} - 失败")
    
    print("\n" + "=" * 60)
    print(f"📊 测试结果: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 所有测试通过！配置管理器工作正常")
        print("\n💡 使用建议:")
        print("1. 设置环境变量 CLICKHOUSE_PASSWORD")
        print("2. 或使用配置工具: python scripts/manage_db_config.py set-password")
        print("3. 在代码中使用: from config.database_config_manager import get_clickhouse_connection_config")
    else:
        print("⚠️  部分测试失败，需要进一步检查")
    
    return passed == total

if __name__ == '__main__':
    success = main_standaloneconfigtest()
    sys.exit(0 if success else 1)
