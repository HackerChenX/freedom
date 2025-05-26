#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import sys

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from db.clickhouse_db import get_clickhouse_db, get_default_config

def init_database():
    """
    初始化ClickHouse数据库和表
    """
    # 获取数据库配置
    db_config = get_default_config()
    
    # 连接数据库
    print(f"连接到ClickHouse数据库: {db_config['host']}:{db_config['port']}")
    db = get_clickhouse_db(config=db_config)
    
    # 初始化数据库和表
    print(f"初始化数据库: {db_config['database']}")
    if db.init_database(db_config['database']):
        print(f"数据库 {db_config['database']} 初始化成功")
    else:
        print(f"数据库 {db_config['database']} 初始化失败")
        return False
    
    print("数据库初始化完成")
    return True

if __name__ == "__main__":
    if init_database():
        print("数据库和表初始化成功，现在可以开始同步股票数据了")
    else:
        print("数据库初始化失败，请检查ClickHouse服务是否正常运行") 