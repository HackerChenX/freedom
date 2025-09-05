#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
诊断ZXM_CORRELATION_MATRIX指标失败原因
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.dependency_injection import get_logger

logger = get_logger(__name__)


def create_test_data(length: int = 100) -> pd.DataFrame:
    """创建测试数据"""
    np.random.seed(42)
    
    # 生成模拟股价数据
    dates = pd.date_range('2024-01-01', periods=length, freq='D')
    
    # 生成价格序列
    initial_price = 100.0
    returns = np.random.normal(0.001, 0.02, length)  # 日收益率
    prices = [initial_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # 生成OHLC数据
    close_prices = np.array(prices[1:])  # 去掉初始价格
    data_length = len(close_prices)
    high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.01, data_length)))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.01, data_length)))
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = initial_price
    
    # 生成成交量数据
    volumes = np.random.lognormal(10, 0.5, data_length)
    
    data = pd.DataFrame({
        'date': dates[:data_length],  # 确保日期长度匹配
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    })
    
    return data


def diagnose_zxm_correlation_matrix():
    """诊断ZXM_CORRELATION_MATRIX指标问题"""
    logger.info("🔍 开始诊断ZXM_CORRELATION_MATRIX指标...")
    
    try:
        # 尝试导入指标类
        logger.info("📦 尝试导入ZXM_CORRELATION_MATRIX指标...")
        
        from indicators.zxm.zxm_correlation_matrix import ZXMCorrelationMatrix
        
        indicator = ZXMCorrelationMatrix()
        logger.info(f"✅ 成功导入: ZXMCorrelationMatrix")
        
        # 检查指标属性
        logger.info("🔍 检查指标属性...")
        logger.info(f"  - 指标名称: {getattr(indicator, 'name', 'N/A')}")
        logger.info(f"  - 指标描述: {getattr(indicator, 'description', 'N/A')}")
        logger.info(f"  - 指标类型: {getattr(indicator, 'indicator_type', 'N/A')}")
        logger.info(f"  - 最小周期: {getattr(indicator, 'minimum_periods', 'N/A')}")
        
        # 创建测试数据
        test_data = create_test_data(100)
        logger.info(f"✅ 测试数据创建成功: {len(test_data)} 行")
        
        # 测试calculate方法
        logger.info("🧪 测试calculate方法...")
        try:
            result = indicator.calculate(test_data)
            
            if result is None:
                logger.error("❌ calculate方法返回None")
                return False
            elif isinstance(result, dict):
                logger.info(f"✅ calculate返回字典，键: {list(result.keys())}")
                if not result:
                    logger.warning("⚠️ calculate返回空字典")
                else:
                    # 显示计算结果
                    for key, value in result.items():
                        logger.info(f"  - {key}: {value}")
            elif isinstance(result, pd.DataFrame):
                logger.info(f"✅ calculate返回DataFrame，形状: {result.shape}")
                logger.info(f"  - 列名: {list(result.columns)}")
                if result.empty:
                    logger.warning("⚠️ calculate返回空DataFrame")
            else:
                logger.warning(f"⚠️ calculate返回未知类型: {type(result)}")
                
        except Exception as e:
            logger.error(f"❌ calculate方法执行失败: {e}")
            import traceback
            logger.error(f"错误详情: {traceback.format_exc()}")
            return False
        
        # 测试get_patterns方法
        logger.info("🧪 测试get_patterns方法...")
        try:
            patterns = indicator.get_patterns()
            
            if patterns is None:
                logger.error("❌ get_patterns方法返回None")
            elif isinstance(patterns, dict):
                logger.info(f"✅ get_patterns返回字典，键: {list(patterns.keys())}")
                if not patterns:
                    logger.warning("⚠️ get_patterns返回空字典")
            else:
                logger.warning(f"⚠️ get_patterns返回未知类型: {type(patterns)}")
                
        except Exception as e:
            logger.error(f"❌ get_patterns方法执行失败: {e}")
            import traceback
            logger.error(f"错误详情: {traceback.format_exc()}")
        
        # 检查BaseIndicator抽象方法
        logger.info("🔍 检查BaseIndicator抽象方法...")
        
        required_methods = [
            '_calculate_baseindicator',
            'calculate_raw_score_Indicator_Base_Indicator',
            'get_patterns_Indicator_Base_Indicator',
            'calculate_confidence_Indicator_Base_Indicator',
            'set_parameters_Indicator_Base_Indicator'
        ]
        
        missing_methods = []
        for method_name in required_methods:
            if not hasattr(indicator, method_name):
                missing_methods.append(method_name)
        
        if missing_methods:
            logger.error(f"❌ 缺少BaseIndicator抽象方法: {missing_methods}")
        else:
            logger.info("✅ 所有BaseIndicator抽象方法都存在")
        
        # 测试边界情况
        logger.info("🧪 测试边界情况...")
        
        # 测试空数据
        try:
            empty_result = indicator.calculate(pd.DataFrame())
            logger.info(f"✅ 空数据处理: {type(empty_result)}")
        except Exception as e:
            logger.error(f"❌ 空数据处理失败: {e}")
        
        # 测试少量数据
        try:
            small_data = create_test_data(10)
            small_result = indicator.calculate(small_data)
            logger.info(f"✅ 少量数据处理: {type(small_result)}")
        except Exception as e:
            logger.error(f"❌ 少量数据处理失败: {e}")
        
        logger.info("🎯 诊断完成")
        return True
        
    except Exception as e:
        logger.error(f"❌ 诊断过程中发生异常: {e}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
        return False


def main():
    """主函数"""
    success = diagnose_zxm_correlation_matrix()
    
    if success:
        logger.info("✅ ZXM_CORRELATION_MATRIX诊断完成")
    else:
        logger.error("❌ ZXM_CORRELATION_MATRIX诊断失败")
    
    return success


if __name__ == "__main__":
    main()
