#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import sys

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from config.config import get_config
from utils.dependency_injection import get_service
from db.interfaces.data_access_interface import IData_access
from db.interfaces.connection_interface import IConnection_manager
from utils.logger import get_logger
from utils.decorators import exception_handler, performance_monitor

logger = get_logger(__name__)

@exception_handler(reraise=True)
@performance_monitor(threshold_seconds=5.0)
def init_database():
    """
    初始化Click_house数据库和表
    
    Returns:
        bool: 初始化是否成功
    """
    try:
        logger.info("开始初始化数据库")
        
        # 获取配置
        config = get_config()
        db_config = config.get('database', {})
        
        # 使用依赖注入获取连接管理器
        container = get_container()
        connection_manager = get_service(Data_access_interface)
        
        # 检查连接健康状态
        if not connection_manager.is_healthy():
            logger.error("数据库连接不健康，无法初始化")
            return False
        
        logger.info(f"连接到ClickHouse数据库: {db_config.get('host')}:{db_config.get('port')}")
        
        # 获取数据访问接口
        data_access = get_service(Data_access_interface)
        
        # 初始化数据库
        database_name = db_config.get('database', 'stock_data')
        logger.info(f"初始化数据库: {database_name}")
        
        # 创建数据库（如果不存在）
        create_db_sql = f"CREATE DATABASE IF NOT EXISTS {database_name}"
        data_access.query(create_db_sql)
        
        # 创建股票信息表（如果不存在）
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {database_name}.stock_info WHERE 1=1 (
            code String,
            name String,
            date Date,
            level String,
            open Float64,
            close Float64,
            high Float64,
            low Float64,
            volume UInt64,
            turnover_rate Float64,
            price_change Float64,
            price_range Float64,
            industry String,
            datetime Date_time,
            seq UInt64
        ) engine = Merge_tree()
        ORDER BY (code, date, level)
        """
        data_access.query(create_table_sql)
        
        logger.info("数据库初始化完成")
        return True
        
    except Exception as e:
        logger.error(f"数据库初始化失败: {e}")
        return False

if __name__ == "__main__":
    if init_database():
        logger.info("数据库和表初始化成功，现在可以开始同步股票数据了")
        print("数据库和表初始化成功，现在可以开始同步股票数据了")
    else:
        logger.error("数据库初始化失败，请检查ClickHouse服务是否正常运行")
        print("数据库初始化失败，请检查ClickHouse服务是否正常运行") 