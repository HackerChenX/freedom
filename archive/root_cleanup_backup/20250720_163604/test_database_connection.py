#!/usr/bin/env python3
from config import get_config
"""
数据库连接测试脚本
用于诊断ClickHouse连接问题
"""

import os
import sys
import traceback
from pathlib import Path

# 添加项目根目录到路径
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))

def test_basic_connection():
    """测试基础连接"""
    print("🔍 测试基础ClickHouse连接...")
    
    try:
        from clickhouse_driver import Client
        
        # 基础配置
        config = {
            get_config('database.host'),
            get_config('database.port'),
            get_config('database.user'),
            get_config('database.password'),
            get_config('database.name')
        }
        
        print(f"连接配置: {config}")
        
        client = Client(**config)
        result = client.execute("SELECT 1")
        print(f"✅ 基础连接成功: {result}")
        return True
        
    except Exception as e:
        print(f"❌ 基础连接失败: {e}")
        print(f"错误详情: {traceback.format_exc()}")
        return False

def test_config_manager():
    """测试配置管理器"""
    print("\n🔍 测试配置管理器...")
    
    try:
        from config.database_config_manager import get_clickhouse_connection_config
        config = get_clickhouse_connection_config()
        print(f"✅ 配置管理器加载成功")
        print(f"配置内容: {config}")
        return config
        
    except Exception as e:
        print(f"❌ 配置管理器失败: {e}")
        print(f"错误详情: {traceback.format_exc()}")
        return None

def test_config_connection(config):
    """测试配置文件连接"""
    print("\n🔍 测试配置文件连接...")
    
    try:
        from clickhouse_driver import Client
        
        client = Client(**config)
        result = client.execute("SELECT 1")
        print(f"✅ 配置文件连接成功: {result}")
        return True
        
    except Exception as e:
        print(f"❌ 配置文件连接失败: {e}")
        print(f"错误详情: {traceback.format_exc()}")
        return False

def test_database_exists(config):
    """测试数据库是否存在"""
    print("\n🔍 测试数据库是否存在...")
    
    try:
        from clickhouse_driver import Client
        
        # 连接到默认数据库
        test_config = config.copy()
        test_config['database'] = 'default'
        
        client = Client(**test_config)
        
        # 检查数据库是否存在
        result = client.execute("SHOW DATABASES")
        databases = [row[0] for row in result]
        print(f"现有数据库: {databases}")
        
        if config['database'] in databases:
            print(f"✅ 数据库 '{config['database']}' 存在")
            return True
        else:
            print(f"❌ 数据库 '{config['database']}' 不存在")
            return False
            
    except Exception as e:
        print(f"❌ 检查数据库失败: {e}")
        print(f"错误详情: {traceback.format_exc()}")
        return False

def test_clickhouse_service():
    """测试ClickHouse服务状态"""
    print("\n🔍 测试ClickHouse服务状态...")
    
    try:
        import socket
        
        # 测试端口连通性
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex(('localhost', 9000))
        sock.close()
        
        if result == 0:
            print("✅ ClickHouse服务端口9000可访问")
            return True
        else:
            print("❌ ClickHouse服务端口9000不可访问")
            return False
            
    except Exception as e:
        print(f"❌ 服务状态检查失败: {e}")
        return False

def test_connection_pool():
    """测试连接池"""
    print("\n🔍 测试连接池...")
    
    try:
        from db.enhanced_connection_pool import get_connection_pool
        
        pool = get_connection_pool()
        print(f"✅ 连接池创建成功: {pool}")
        
        with pool.get_connection() as conn:
            result = conn.execute("SELECT 1")
            print(f"✅ 连接池查询成功: {result}")
            
        return True
        
    except Exception as e:
        print(f"❌ 连接池测试失败: {e}")
        print(f"错误详情: {traceback.format_exc()}")
        return False

def main_test_database_connection():
    """主测试函数"""
    print("🚀 开始数据库连接诊断...")
    print("=" * 50)
    
    # 1. 测试ClickHouse服务状态
    service_ok = test_clickhouse_service()
    
    # 2. 测试基础连接
    basic_ok = test_basic_connection()
    
    # 3. 测试配置管理器
    config = test_config_manager()
    
    # 4. 测试配置文件连接
    config_ok = False
    if config:
        config_ok = test_config_connection(config)
        
        # 5. 测试数据库是否存在
        if config_ok:
            test_database_exists(config)
    
    # 6. 测试连接池
    pool_ok = test_connection_pool()
    
    print("\n" + "=" * 50)
    print("📊 诊断结果总结:")
    print(f"ClickHouse服务: {'✅' if service_ok else '❌'}")
    print(f"基础连接: {'✅' if basic_ok else '❌'}")
    print(f"配置管理器: {'✅' if config else '❌'}")
    print(f"配置文件连接: {'✅' if config_ok else '❌'}")
    print(f"连接池: {'✅' if pool_ok else '❌'}")
    
    if all([service_ok, basic_ok, config, config_ok, pool_ok]):
        print("\n🎉 所有测试通过！数据库连接正常")
        return True
    else:
        print("\n⚠️ 存在连接问题，需要修复")
        return False

if __name__ == "__main__":
    main_test_database_connection()
