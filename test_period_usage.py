#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.clickhouse_db import ClickHouseDB
from scripts.backtest.data_manager import BacktestDataManager
from enums.period import Period
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_period_usage():
    """测试周期参数的使用"""
    try:
        # 初始化数据管理器
        data_manager = BacktestDataManager()
        
        # 测试用例1：使用 Period 枚举
        logger.info("\n测试用例1：使用 Period 枚举")
        try:
            data = data_manager.get_stock_data(
                stock_code='603359',
                period=Period.DAY,
                start_date='2025-05-20',
                end_date='2025-05-23'
            )
            logger.info(f"成功获取数据，行数: {len(data)}")
        except Exception as e:
            logger.error(f"测试用例1失败: {e}")
        
        # 测试用例2：使用字符串（应该自动转换为枚举）
        logger.info("\n测试用例2：使用字符串")
        try:
            data = data_manager.get_stock_data(
                stock_code='603359',
                period='day',
                start_date='2025-05-20',
                end_date='2025-05-23'
            )
            logger.info(f"成功获取数据，行数: {len(data)}")
        except Exception as e:
            logger.error(f"测试用例2失败: {e}")
        
        # 测试用例3：使用无效的周期字符串
        logger.info("\n测试用例3：使用无效的周期字符串")
        try:
            data = data_manager.get_stock_data(
                stock_code='603359',
                period='daily',  # 无效的周期
                start_date='2025-05-20',
                end_date='2025-05-23'
            )
            logger.info(f"成功获取数据，行数: {len(data)}")
        except ValueError as e:
            logger.info(f"预期的错误: {e}")
        except Exception as e:
            logger.error(f"测试用例3失败: {e}")
        
        # 测试用例4：使用无效的类型
        logger.info("\n测试用例4：使用无效的类型")
        try:
            data = data_manager.get_stock_data(
                stock_code='603359',
                period=123,  # 无效的类型
                start_date='2025-05-20',
                end_date='2025-05-23'
            )
            logger.info(f"成功获取数据，行数: {len(data)}")
        except ValueError as e:
            logger.info(f"预期的错误: {e}")
        except Exception as e:
            logger.error(f"测试用例4失败: {e}")
        
        # 测试用例5：测试所有支持的周期
        logger.info("\n测试用例5：测试所有支持的周期")
        for period in Period.get_all_period_values():
            try:
                data = data_manager.get_stock_data(
                    stock_code='603359',
                    period=period,
                    start_date='2025-05-20',
                    end_date='2025-05-23'
                )
                logger.info(f"周期 {period} 数据获取成功，行数: {len(data)}")
            except Exception as e:
                logger.error(f"周期 {period} 测试失败: {e}")
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        raise

if __name__ == '__main__':
    test_period_usage() 