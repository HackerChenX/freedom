#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
诊断ADX指标失败原因
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


def diagnose_adx_indicator():
    """诊断ADX指标问题"""
    logger.info("🔍 开始诊断ADX指标...")
    
    try:
        # 尝试导入ADX指标类
        logger.info("📦 尝试导入ADX指标...")
        
        from indicators.adx import AverageDirectionalIndex
        
        indicator = AverageDirectionalIndex()
        logger.info(f"✅ 成功导入: AverageDirectionalIndex")
        
        # 检查指标属性
        logger.info("🔍 检查指标属性...")
        logger.info(f"  - 指标名称: {getattr(indicator, 'name', 'N/A')}")
        logger.info(f"  - 指标描述: {getattr(indicator, 'description', 'N/A')}")
        logger.info(f"  - 必需列: {getattr(indicator, 'REQUIRED_COLUMNS', 'N/A')}")
        
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
            elif isinstance(result, pd.DataFrame):
                logger.info(f"✅ calculate返回DataFrame，形状: {result.shape}")
                logger.info(f"  - 列名: {list(result.columns)}")
                
                # 检查ADX相关列
                adx_columns = [col for col in result.columns if 'ADX' in col or 'PDI' in col or 'MDI' in col]
                logger.info(f"  - ADX相关列: {adx_columns}")
                
                if result.empty:
                    logger.warning("⚠️ calculate返回空DataFrame")
            else:
                logger.warning(f"⚠️ calculate返回未知类型: {type(result)}")
                
        except Exception as e:
            logger.error(f"❌ calculate方法执行失败: {e}")
            import traceback
            logger.error(f"错误详情: {traceback.format_exc()}")
            return False
        
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
        
        # 测试形态识别功能
        logger.info("🧪 测试形态识别功能...")
        try:
            # 检查是否有get_patterns方法
            if hasattr(indicator, 'get_patterns'):
                patterns = indicator.get_patterns(test_data)
                logger.info(f"✅ get_patterns方法存在，返回类型: {type(patterns)}")
                
                if isinstance(patterns, pd.DataFrame):
                    logger.info(f"  - 形态DataFrame形状: {patterns.shape}")
                    logger.info(f"  - 形态列名: {list(patterns.columns)}")
                    
                    # 检查形态数量
                    pattern_count = 0
                    for col in patterns.columns:
                        if patterns[col].dtype == bool:
                            pattern_count += patterns[col].sum()
                    logger.info(f"  - 检测到的形态数量: {pattern_count}")
                    
                elif isinstance(patterns, list):
                    logger.info(f"  - 形态列表长度: {len(patterns)}")
                else:
                    logger.warning(f"⚠️ get_patterns返回未知类型: {type(patterns)}")
            else:
                logger.warning("⚠️ 缺少get_patterns方法")
                
        except Exception as e:
            logger.error(f"❌ 形态识别测试失败: {e}")
        
        # 测试生产就绪性
        logger.info("🧪 测试生产就绪性...")
        
        # 检查性能
        import time
        start_time = time.time()
        for _ in range(10):
            indicator.calculate(test_data)
        end_time = time.time()
        avg_time = (end_time - start_time) / 10
        
        logger.info(f"  - 平均计算时间: {avg_time:.4f}秒")
        
        if avg_time > 0.1:
            logger.warning(f"⚠️ 计算时间过长: {avg_time:.4f}秒 > 0.1秒")
        else:
            logger.info("✅ 计算性能良好")
        
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
        
        logger.info("🎯 ADX指标诊断完成")
        return True
        
    except Exception as e:
        logger.error(f"❌ 诊断过程中发生异常: {e}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
        return False


def main():
    """主函数"""
    success = diagnose_adx_indicator()
    
    if success:
        logger.info("✅ ADX指标诊断完成")
    else:
        logger.error("❌ ADX指标诊断失败")
    
    return success


if __name__ == "__main__":
    main()
