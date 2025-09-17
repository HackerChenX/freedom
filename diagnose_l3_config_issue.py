#!/usr/bin/env python3
"""
L3数据服务层配置问题诊断脚本
精确定位导致"expected str, bytes or os.PathLike object, not NoneType"错误的根本原因
"""

import sys
import traceback
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def diagnose_config_chain():
    """诊断配置调用链"""
    print("🔍 开始诊断L3数据服务层配置问题...")
    
    # 1. 测试基础配置管理器
    print("\n1. 测试统一配置管理器...")
    try:
        from config.unified_config_manager import get_config_manager, get_config
        config_manager = get_config_manager()
        print(f"✅ 统一配置管理器初始化成功")
        print(f"   配置目录: {config_manager.config_dir}")
        
        # 测试基本配置获取
        db_host = get_config('database.host', 'localhost')
        print(f"   数据库主机: {db_host}")
        
    except Exception as e:
        print(f"❌ 统一配置管理器失败: {e}")
        traceback.print_exc()
        return False
    
    # 2. 测试数据库配置管理器
    print("\n2. 测试数据库配置管理器...")
    try:
        from config.database_config_manager import DatabaseConfigManager
        db_config_manager = DatabaseConfigManager()
        print(f"✅ 数据库配置管理器初始化成功")
        
        # 测试配置获取
        config = db_config_manager.get_database_config()
        print(f"   配置内容: {config}")
        
    except Exception as e:
        print(f"❌ 数据库配置管理器失败: {e}")
        traceback.print_exc()
        return False
    
    # 3. 测试统一数据库配置
    print("\n3. 测试统一数据库配置...")
    try:
        from config.unified_database_config import get_unified_database_config
        unified_db_config = get_unified_database_config()
        print(f"✅ 统一数据库配置初始化成功")
        
        # 测试配置获取
        clickhouse_config = unified_db_config.get_clickhouse_config()
        print(f"   ClickHouse配置: {clickhouse_config}")
        
    except Exception as e:
        print(f"❌ 统一数据库配置失败: {e}")
        traceback.print_exc()
        return False
    
    # 4. 测试连接池获取
    print("\n4. 测试连接池获取...")
    try:
        from db.enhanced_connection_pool import get_connection_pool
        connection_pool = get_connection_pool()
        print(f"✅ 连接池获取成功: {type(connection_pool)}")
        
    except Exception as e:
        print(f"❌ 连接池获取失败: {e}")
        traceback.print_exc()
        return False
    
    # 5. 测试数据访问管理器
    print("\n5. 测试数据访问管理器...")
    try:
        from db.managers.data_access_manager import DataAccessManager
        data_access_manager = DataAccessManager()
        print(f"✅ 数据访问管理器初始化成功")
        
    except Exception as e:
        print(f"❌ 数据访问管理器失败: {e}")
        traceback.print_exc()
        return False
    
    # 6. 测试缓存服务
    print("\n6. 测试缓存服务...")
    try:
        from db.services.cache_service import CacheService
        cache_service = CacheService()
        print(f"✅ 缓存服务初始化成功")
        
    except Exception as e:
        print(f"❌ 缓存服务失败: {e}")
        traceback.print_exc()
        return False
    
    # 7. 测试服务注册
    print("\n7. 测试服务注册...")
    try:
        from db.service_registry import ServiceRegistry
        service_registry = ServiceRegistry()
        print(f"✅ 服务注册初始化成功")
        
    except Exception as e:
        print(f"❌ 服务注册失败: {e}")
        traceback.print_exc()
        return False
    
    print("\n🎉 所有L3层组件诊断完成!")
    return True

def diagnose_path_issues():
    """诊断路径相关问题"""
    print("\n🔍 诊断路径相关问题...")
    
    # 检查当前工作目录
    import os
    print(f"当前工作目录: {os.getcwd()}")
    
    # 检查项目根目录
    print(f"项目根目录: {project_root}")
    
    # 检查配置文件
    config_files = [
        "config/database.yaml",
        "config/unified_config.json",
        "config/base.yaml",
        "config/config.yaml"
    ]
    
    for config_file in config_files:
        config_path = project_root / config_file
        exists = config_path.exists()
        print(f"配置文件 {config_file}: {'✅ 存在' if exists else '❌ 不存在'}")
        if exists:
            print(f"   绝对路径: {config_path.absolute()}")

def diagnose_import_issues():
    """诊断导入相关问题"""
    print("\n🔍 诊断导入相关问题...")
    
    # 测试关键模块导入
    modules_to_test = [
        "config.unified_config_manager",
        "config.database_config_manager", 
        "config.unified_database_config",
        "db.enhanced_connection_pool",
        "db.managers.data_access_manager",
        "db.services.cache_service",
        "db.service_registry",
        "utils.logger"
    ]
    
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"✅ {module_name}: 导入成功")
        except Exception as e:
            print(f"❌ {module_name}: 导入失败 - {e}")

def main():
    """主函数"""
    print("🚀 L3数据服务层配置问题诊断")
    print("=" * 60)
    
    # 诊断路径问题
    diagnose_path_issues()
    
    # 诊断导入问题
    diagnose_import_issues()
    
    # 诊断配置调用链
    success = diagnose_config_chain()
    
    if success:
        print("\n🎉 诊断完成: 所有组件正常工作!")
        print("✅ L3数据服务层配置问题已解决")
    else:
        print("\n❌ 诊断发现问题: 需要进一步修复")
        print("🔧 请根据上述错误信息进行修复")

if __name__ == "__main__":
    main()
