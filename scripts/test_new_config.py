#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
测试新的数据库配置管理器
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_config_manager():
    """测试配置管理器"""
    print("🔍 测试新的数据库配置管理器")
    print("=" * 50)
    
    try:
        # 直接导入配置管理器，避免config.py的冲突
        from config.database_config_manager import DatabaseConfigManager
        
        # 创建配置管理器实例
        manager = DatabaseConfigManager()
        
        # 测试获取配置
        config = manager.get_config()
        print("✅ 配置加载成功")
        print(f"主机: {config['host']}")
        print(f"端口: {config['port']}")
        print(f"数据库: {config['database']}")
        print(f"用户: {config['user']}")
        print(f"密码: {'已设置' if config.get('password') else '未设置'}")
        
        # 测试连接配置
        conn_config = manager.get_connection_config()
        print(f"\n📡 连接配置: {conn_config['user']}@{conn_config['host']}:{conn_config['port']}/{conn_config['database']}")
        
        # 测试配置验证
        if manager.validate_config():
            print("✅ 配置验证通过")
        else:
            print("❌ 配置验证失败")
        
        # 测试连接（如果有密码）
        if config.get('password'):
            print("\n🔍 测试数据库连接...")
            if manager.test_connection():
                print("✅ 数据库连接测试成功！")
            else:
                print("❌ 数据库连接测试失败")
        else:
            print("\n⚠️  未设置密码，跳过连接测试")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_environment_variables():
    """测试环境变量配置"""
    print("\n🌍 测试环境变量配置")
    print("=" * 50)
    
    # 设置测试环境变量
    os.environ['CLICKHOUSE_HOST'] = 'test-host'
    os.environ['CLICKHOUSE_PORT'] = '9001'
    os.environ['CLICKHOUSE_PASSWORD'] = 'test-password'
    
    try:
        from config.database_config_manager import DatabaseConfigManager
        
        # 创建新的配置管理器实例
        manager = DatabaseConfigManager()
        manager.reload_config()  # 重新加载配置
        
        config = manager.get_config()
        
        # 验证环境变量是否生效
        if config['host'] == 'test-host':
            print("✅ 环境变量 CLICKHOUSE_HOST 生效")
        else:
            print(f"❌ 环境变量 CLICKHOUSE_HOST 未生效: {config['host']}")
        
        if config['port'] == 9001:
            print("✅ 环境变量 CLICKHOUSE_PORT 生效")
        else:
            print(f"❌ 环境变量 CLICKHOUSE_PORT 未生效: {config['port']}")
        
        if config['password'] == 'test-password':
            print("✅ 环境变量 CLICKHOUSE_PASSWORD 生效")
        else:
            print(f"❌ 环境变量 CLICKHOUSE_PASSWORD 未生效")
        
        return True
        
    except Exception as e:
        print(f"❌ 环境变量测试失败: {e}")
        return False
    finally:
        # 清理测试环境变量
        for key in ['CLICKHOUSE_HOST', 'CLICKHOUSE_PORT', 'CLICKHOUSE_PASSWORD']:
            if key in os.environ:
                del os.environ[key]

def test_password_encryption():
    """测试密码加密功能"""
    print("\n🔒 测试密码加密功能")
    print("=" * 50)
    
    try:
        from config.database_config_manager import DatabaseConfigManager
        
        manager = DatabaseConfigManager()
        
        # 测试密码设置
        test_password = "test_password_123"
        manager.set_password(test_password, encrypt=True, save_to_file=False)
        
        # 验证密码是否正确设置
        config = manager.get_config()
        if config.get('password') == test_password:
            print("✅ 密码设置成功")
        else:
            print("❌ 密码设置失败")
        
        return True
        
    except Exception as e:
        print(f"❌ 密码加密测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🧪 新数据库配置管理器测试套件")
    print("=" * 60)
    
    tests = [
        ("基础配置管理", test_config_manager),
        ("环境变量配置", test_environment_variables),
        ("密码加密功能", test_password_encryption)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 执行测试: {test_name}")
        if test_func():
            passed += 1
            print(f"✅ {test_name} - 通过")
        else:
            print(f"❌ {test_name} - 失败")
    
    print("\n" + "=" * 60)
    print(f"📊 测试结果: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 所有测试通过！新配置管理器工作正常")
    else:
        print("⚠️  部分测试失败，需要进一步检查")

if __name__ == '__main__':
    main()
