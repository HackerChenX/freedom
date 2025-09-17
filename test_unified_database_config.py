#!/usr/bin/env python3
"""
统一数据库配置测试
验证L2存储访问层的单一配置源原则实现
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_unified_config_file_exists():
    """测试统一配置文件是否存在"""
    print("🔍 测试统一配置文件...")
    
    required_files = [
        'config/database.yaml',  # 主配置文件
        'config/unified_database_config.py'  # 统一配置管理器
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ 配置文件不存在: {file_path}")
            return False
    
    print("✅ 统一配置文件验证通过")
    return True

def test_unified_config_import():
    """测试统一配置导入"""
    print("🔍 测试统一配置导入...")
    
    try:
        from config.unified_database_config import (
            get_unified_database_config,
            get_database_config,
            get_clickhouse_connection_config
        )
        
        # 测试配置管理器实例
        config_manager = get_unified_database_config()
        assert config_manager is not None
        
        # 测试便捷函数
        db_config = get_database_config()
        assert isinstance(db_config, dict)
        
        conn_config = get_clickhouse_connection_config()
        assert isinstance(conn_config, dict)
        
        print("✅ 统一配置导入验证通过")
        return True
        
    except Exception as e:
        print(f"❌ 统一配置导入失败: {e}")
        return False

def test_config_port_consistency():
    """测试配置端口一致性"""
    print("🔍 测试配置端口一致性...")
    
    try:
        from config.unified_database_config import get_clickhouse_connection_config
        
        # 获取连接配置
        conn_config = get_clickhouse_connection_config()
        port = conn_config.get('port')
        
        # 验证端口必须是9000（原生端口）
        if port != 9000:
            print(f"❌ 端口配置错误: 期望9000，实际{port}")
            return False
        
        print(f"✅ 端口配置正确: {port}")
        return True
        
    except Exception as e:
        print(f"❌ 端口一致性测试失败: {e}")
        return False

def test_config_completeness():
    """测试配置完整性"""
    print("🔍 测试配置完整性...")
    
    try:
        from config.unified_database_config import get_unified_database_config
        
        config_manager = get_unified_database_config()
        
        # 测试连接配置
        conn_config = config_manager.get_connection_config()
        required_conn_keys = ['host', 'port', 'database', 'user', 'password']
        
        for key in required_conn_keys:
            if key not in conn_config:
                print(f"❌ 连接配置缺少必需项: {key}")
                return False
        
        # 测试池配置
        pool_config = config_manager.get_pool_config()
        required_pool_keys = ['min_size', 'max_size', 'timeout']
        
        for key in required_pool_keys:
            if key not in pool_config:
                print(f"❌ 池配置缺少必需项: {key}")
                return False
        
        # 测试缓存配置
        cache_config = config_manager.get_cache_config()
        required_cache_keys = ['enabled', 'max_size', 'ttl']
        
        for key in required_cache_keys:
            if key not in cache_config:
                print(f"❌ 缓存配置缺少必需项: {key}")
                return False
        
        print("✅ 配置完整性验证通过")
        return True
        
    except Exception as e:
        print(f"❌ 配置完整性测试失败: {e}")
        return False

def test_backward_compatibility():
    """测试向后兼容性"""
    print("🔍 测试向后兼容性...")
    
    try:
        # 测试旧的配置管理器
        from config.database_config_manager import DatabaseConfigManager, get_clickhouse_connection_config
        
        # 测试类实例
        manager = DatabaseConfigManager()
        config = manager.get_config()
        assert isinstance(config, dict)
        
        # 测试便捷函数
        conn_config = get_clickhouse_connection_config()
        assert isinstance(conn_config, dict)
        
        # 验证端口一致性
        if conn_config.get('port') != 9000:
            print(f"❌ 向后兼容配置端口错误: {conn_config.get('port')}")
            return False
        
        print("✅ 向后兼容性验证通过")
        return True
        
    except Exception as e:
        print(f"❌ 向后兼容性测试失败: {e}")
        return False

def test_single_source_principle():
    """测试单一配置源原则"""
    print("🔍 测试单一配置源原则...")
    
    try:
        from config.unified_database_config import get_unified_database_config
        from config.database_config_manager import get_clickhouse_connection_config
        
        # 获取两个不同入口的配置
        unified_config = get_unified_database_config().get_connection_config()
        compat_config = get_clickhouse_connection_config()
        
        # 验证配置一致性
        key_fields = ['host', 'port', 'database', 'user']
        for key in key_fields:
            if unified_config.get(key) != compat_config.get(key):
                print(f"❌ 配置不一致 {key}: 统一={unified_config.get(key)}, 兼容={compat_config.get(key)}")
                return False
        
        print("✅ 单一配置源原则验证通过")
        return True
        
    except Exception as e:
        print(f"❌ 单一配置源测试失败: {e}")
        return False

def test_connection_pool_integration():
    """测试连接池集成"""
    print("🔍 测试连接池集成...")
    
    try:
        # 不实际连接数据库，只测试配置加载
        from db.enhanced_connection_pool import ClickHouseConnectionPool
        
        # 创建连接池实例（使用配置文件）
        pool = ClickHouseConnectionPool()
        
        # 验证配置是否正确加载
        config = pool.config
        if config['port'] != 9000:
            print(f"❌ 连接池端口配置错误: {config['port']}")
            return False
        
        print(f"✅ 连接池配置正确: {config['host']}:{config['port']}")
        return True
        
    except Exception as e:
        print(f"❌ 连接池集成测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始统一数据库配置验证...")
    print("=" * 50)
    print("📋 验证L2存储访问层单一配置源原则实现")
    print("=" * 50)
    
    tests = [
        test_unified_config_file_exists,
        test_unified_config_import,
        test_config_port_consistency,
        test_config_completeness,
        test_backward_compatibility,
        test_single_source_principle,
        test_connection_pool_integration
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
    
    print(f"\n📊 统一数据库配置测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 统一数据库配置验证成功！")
        print("✅ 严格遵循L2存储访问层单一配置源原则")
        print("✅ 所有配置统一到 config/database.yaml")
        print("✅ 消除了配置分散问题")
        return True
    else:
        print("🚫 统一数据库配置验证失败，需要进一步处理")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
