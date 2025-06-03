#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from db.db_manager import DBManager
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_stock_info():
    db = DBManager()
    
    # 测试用例1：无效的level参数
    print("\n测试用例1：无效的level参数")
    try:
        db.db.get_stock_info('603359', 'daily', '2025-05-20', '2025-05-23')
    except ValueError as e:
        print(f"参数验证错误: {e}")
    
    # 测试用例2：正确的参数
    print("\n测试用例2：正确的参数")
    try:
        result = db.db.get_stock_info('603359', 'day', '2025-05-20', '2025-05-23')
        print(f"查询结果:\n{result.head()}")
    except Exception as e:
        print(f"查询错误: {e}")
    
    # 测试用例3：无效的日期格式
    print("\n测试用例3：无效的日期格式")
    try:
        db.db.get_stock_info('603359', 'day', '2025/05/20', '2025-05-23')
    except ValueError as e:
        print(f"参数验证错误: {e}")
    
    # 测试用例4：日期范围错误
    print("\n测试用例4：日期范围错误")
    try:
        db.db.get_stock_info('603359', 'day', '2025-05-23', '2025-05-20')
    except ValueError as e:
        print(f"参数验证错误: {e}")
    
    # 测试用例5：不存在的股票代码
    print("\n测试用例5：不存在的股票代码")
    try:
        result = db.db.get_stock_info('999999', 'day', '2025-05-20', '2025-05-23')
        print(f"查询结果:\n{result}")
    except Exception as e:
        print(f"查询错误: {e}")

if __name__ == '__main__':
    test_stock_info() 