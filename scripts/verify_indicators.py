#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
技术指标验证脚本

用于验证回测系统中所有技术指标的计算是否正常
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from indicators.factory import IndicatorFactory
from utils.logger import get_logger

logger = get_logger(__name__)


def create_test_data(n_rows=100):
    """
    创建测试数据
    
    Args:
        n_rows: 数据行数
    
    Returns:
        包含OHLCV数据的DataFrame
    """
    # 创建日期索引
    today = datetime.now()
    dates = [today - timedelta(days=i) for i in range(n_rows)]
    dates.reverse()
    
    # 创建模拟价格数据
    close = np.random.normal(100, 10, n_rows).cumsum() + 1000
    high = close + np.random.normal(2, 1, n_rows).cumsum()
    low = close - np.random.normal(2, 1, n_rows).cumsum()
    open_price = (high + low) / 2
    volume = np.random.normal(1000000, 500000, n_rows).astype(int)
    volume = np.abs(volume)
    
    # 创建DataFrame
    df = pd.DataFrame({
        'date': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    df.set_index('date', inplace=True)
    return df


def test_all_indicators():
    """
    测试所有技术指标
    
    Returns:
        包含测试结果的字典
    """
    factory = IndicatorFactory()
    supported_indicators = factory.get_supported_indicators()
    
    logger.info(f"开始测试 {len(supported_indicators)} 个指标...")
    
    # 创建测试数据
    df = create_test_data(200)
    
    # 测试结果
    results = {}
    
    # 测试每个指标
    for indicator_type in supported_indicators:
        try:
            logger.info(f"测试指标: {indicator_type}")
            
            # 创建指标实例
            indicator = factory.create_indicator(indicator_type)
            
            if indicator is None:
                logger.warning(f"无法创建指标: {indicator_type}")
                results[indicator_type] = {
                    "status": "error",
                    "message": "无法创建指标实例"
                }
                continue
            
            # 计算指标
            start_time = datetime.now()
            result_df = indicator.compute(df)
            elapsed_time = (datetime.now() - start_time).total_seconds()
            
            # 检查结果
            if result_df is None or result_df.empty:
                logger.warning(f"指标 {indicator_type} 计算结果为空")
                results[indicator_type] = {
                    "status": "error",
                    "message": "计算结果为空"
                }
                continue
            
            # 统计结果中的指标列数量
            indicator_columns = [col for col in result_df.columns if col not in df.columns]
            
            results[indicator_type] = {
                "status": "success",
                "columns": indicator_columns,
                "rows": len(result_df),
                "elapsed_time": elapsed_time
            }
            
            logger.info(f"指标 {indicator_type} 测试成功，生成列: {indicator_columns}")
            
        except Exception as e:
            logger.error(f"测试指标 {indicator_type} 失败: {str(e)}")
            results[indicator_type] = {
                "status": "error",
                "message": str(e)
            }
    
    # 汇总测试结果
    success_count = sum(1 for r in results.values() if r["status"] == "success")
    error_count = sum(1 for r in results.values() if r["status"] == "error")
    
    logger.info(f"测试完成: 成功 {success_count}, 失败 {error_count}")
    
    return results


def save_results(results, output_file):
    """
    保存测试结果到文件
    
    Args:
        results: 测试结果
        output_file: 输出文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"测试结果已保存到 {output_file}")


if __name__ == "__main__":
    # 执行测试
    results = test_all_indicators()
    
    # 保存结果
    output_dir = os.path.join(root_dir, 'data', 'result')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'indicator_verification_result.json')
    save_results(results, output_file) 