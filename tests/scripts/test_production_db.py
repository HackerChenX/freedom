from db.query_executor import get_query_executor
from db.sql_manager import QueryType
#!/usr/bin/env python3
"""
生产环境数据库连接测试工具
"""

import sys
import os
import time
from datetime import datetime

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

from config.database_config_manager import get_database_config_manager
from db.unified_data_manager import Unified_data_manager
from utils.logger import get_logger

logger = get_logger(__name__)


def test_database_connection():
    """测试数据库连接"""
    print("=" * 60)
    print("🚀 生产环境数据库连接测试")
    print("=" * 60)
    
    try:
        # 获取配置
        config_manager = get_database_config_manager()
        config = config_manager.get_connection_config()
        
        print(f"连接配置:")
        print(f"  主机: {config['host']}")
        print(f"  端口: {config['port']}")
        print(f"  数据库: {config['database']}")
        print(f"  用户: {config['user']}")
        print(f"  密码: {'***' if config['password'] else '(空)'}")
        
        # 创建数据管理器
        print("\n🔍 创建数据管理器...")
        data_manager = Unified_data_manager()
        
        # 测试连接
        print("🔗 测试数据库连接...")
        start_time = time.time()
        
        connection_success = data_manager.test_connection()
        duration = time.time() - start_time
        
        if connection_success:
            print(f"✅ 数据库连接成功！耗时: {duration:.2f}秒")
            
            # 测试简单查询
            print("📊 测试数据查询...")
            try:
                query = "SELECT COUNT(*) as count FROM system.databases"
                result = data_manager.execute_query(query)
                if result is not None and not result.empty:
                    db_count = result.iloc[0]['count']
                    print(f"✅ 查询成功，数据库数量: {db_count}")
                else:
                    print("❌ 查询返回空结果")
            except Exception as e:
                print(f"❌ 查询失败: {e}")
            
            return True
        else:
            print(f"❌ 数据库连接失败！耗时: {duration:.2f}秒")
            return False
            
    except Exception as e:
        print(f"❌ 测试过程中发生异常: {e}")
        logger.exception("数据库连接测试异常")
        return False


def main_testproductiondb():
    """主函数"""
    try:
        success = test_database_connection()
        
        print("\n" + "=" * 60)
        if success:
            print("🎉 生产环境数据库测试通过！")
            print("系统可以正常连接到ClickHouse数据库。")
        else:
            print("⚠️  生产环境数据库测试失败！")
            print("请检查ClickHouse服务是否运行，或配置是否正确。")
        print("=" * 60)
        
        sys.exit(0 if success else 1)
        
    except Keyboard_interrupt:
        print("\n\n⚠️  测试被用户中断")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main_testproductiondb()
