#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
测试ZXM评分指标
"""

import os
import sys
import pandas as pd
from datetime import datetime

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from indicators.zxm.score_indicators import ZXMElasticityScore, ZXMBuyPointScore
from utils.logger import get_logger, init_logging
from db.clickhouse_db import get_clickhouse_db

logger = get_logger(__name__)


def test_zxm_elasticity_score():
    """测试ZXM弹性评分指标"""
    logger.info("开始测试ZXM弹性评分指标")
    
    # 初始化指标
    elasticity_score = ZXMElasticityScore(threshold=75)
    
    # 获取测试数据
    db = get_clickhouse_db()
    stock_code = "600585"  # 海螺水泥
    end_date = "2025-04-15"
    sql = f"""
    SELECT
        toDate(date) as trade_date,
        open, high, low, close,
        volume, turnover,
        amount
    FROM stock_daily_data
    WHERE code = '{stock_code}'
    AND date <= '{end_date}'
    ORDER BY date
    LIMIT 250
    """
    
    data = db.query(sql)
    data.set_index('trade_date', inplace=True)
    
    # 计算指标
    result = elasticity_score.calculate(data)
    
    # 显示结果
    logger.info(f"数据大小: {len(data)}")
    logger.info(f"结果大小: {len(result)}")
    logger.info(f"最后5行结果:\n{result.tail()}")
    
    # 统计信号生成情况
    signal_count = result['Signal'].sum()
    logger.info(f"信号生成数量: {signal_count}")
    logger.info(f"信号生成比例: {signal_count / len(result) * 100:.2f}%")
    

def test_zxm_buypoint_score():
    """测试ZXM买点评分指标"""
    logger.info("开始测试ZXM买点评分指标")
    
    # 初始化指标
    buypoint_score = ZXMBuyPointScore(threshold=75)
    
    # 获取测试数据
    db = get_clickhouse_db()
    stock_code = "600585"  # 海螺水泥
    end_date = "2025-04-15"
    sql = f"""
    SELECT
        toDate(date) as trade_date,
        open, high, low, close,
        volume, turnover,
        amount
    FROM stock_daily_data
    WHERE code = '{stock_code}'
    AND date <= '{end_date}'
    ORDER BY date
    LIMIT 250
    """
    
    data = db.query(sql)
    data.set_index('trade_date', inplace=True)
    
    # 计算指标
    result = buypoint_score.calculate(data)
    
    # 显示结果
    logger.info(f"数据大小: {len(data)}")
    logger.info(f"结果大小: {len(result)}")
    logger.info(f"最后5行结果:\n{result.tail()}")
    
    # 统计信号生成情况
    signal_count = result['Signal'].sum()
    logger.info(f"信号生成数量: {signal_count}")
    logger.info(f"信号生成比例: {signal_count / len(result) * 100:.2f}%")


if __name__ == "__main__":
    init_logging()
    
    try:
        test_zxm_elasticity_score()
        print("\n" + "-" * 50 + "\n")
        test_zxm_buypoint_score()
    except Exception as e:
        logger.error(f"测试出错: {e}", exc_info=True)
    
    logger.info("测试完成") 