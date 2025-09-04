#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试修复后的ZXM_CORRELATION_MATRIX指标
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
from indicators.zxm.zxm_correlation_matrix import ZXMCorrelationMatrix

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


def test_fixed_correlation_matrix():
    """测试修复后的ZXM_CORRELATION_MATRIX指标"""
    logger.info("🧪 开始测试修复后的ZXM_CORRELATION_MATRIX指标...")
    
    try:
        # 创建指标实例
        indicator = ZXMCorrelationMatrix()
        logger.info(f"✅ 指标实例创建成功: {indicator.name}")
        
        # 创建测试数据
        test_data = create_test_data(100)
        logger.info(f"✅ 测试数据创建成功: {len(test_data)} 行")
        
        # 测试calculate方法
        logger.info("🧪 测试calculate方法...")
        result = indicator.calculate(test_data)
        
        if result is None:
            logger.error("❌ calculate方法返回None")
            return False
        elif not result:
            logger.error("❌ calculate方法返回空结果")
            return False
        else:
            logger.info(f"✅ calculate方法成功，返回类型: {type(result)}")
            
            if isinstance(result, dict):
                logger.info(f"  - 返回键: {list(result.keys())}")
                
                # 检查相关性指标
                expected_keys = ['autocorr_lag1', 'autocorr_lag5', 'market_correlation', 
                               'rolling_corr_mean', 'rolling_corr_std', 'correlation_stability',
                               'correlation_strength', 'correlation_direction', 'diversification_benefit',
                               'diversification_level', 'systematic_risk', 'systematic_risk_level']
                for key in expected_keys:
                    if key in result:
                        logger.info(f"  - {key}: {result[key]}")
                    else:
                        logger.warning(f"  - 缺少键: {key}")
        
        # 测试get_patterns方法
        logger.info("🧪 测试get_patterns方法...")
        patterns = indicator.get_patterns()
        
        if patterns is None:
            logger.error("❌ get_patterns方法返回None")
            return False
        elif not patterns:
            logger.error("❌ get_patterns方法返回空字典")
            return False
        else:
            logger.info(f"✅ get_patterns方法成功，返回键: {list(patterns.keys())}")
            
            # 检查关键信息
            if 'indicator_type' in patterns:
                logger.info(f"  - 指标类型: {patterns['indicator_type']}")
            if 'metrics' in patterns:
                logger.info(f"  - 指标度量: {patterns['metrics']}")
            if 'correlation_strengths' in patterns:
                logger.info(f"  - 相关性强度: {patterns['correlation_strengths']}")
        
        # 测试BaseIndicator方法
        logger.info("🧪 测试BaseIndicator方法...")
        
        # 测试_calculate_baseindicator
        try:
            base_result = indicator._calculate_baseindicator(test_data)
            if isinstance(base_result, pd.DataFrame):
                logger.info(f"✅ _calculate_baseindicator成功: {base_result.shape}")
            else:
                logger.error("❌ _calculate_baseindicator返回非DataFrame")
                return False
        except Exception as e:
            logger.error(f"❌ _calculate_baseindicator失败: {e}")
            return False
        
        # 测试calculate_raw_score_Indicator_Base_Indicator
        try:
            raw_score = indicator.calculate_raw_score_Indicator_Base_Indicator(test_data)
            if isinstance(raw_score, pd.Series):
                logger.info(f"✅ calculate_raw_score成功: {raw_score.iloc[0]:.2f}")
            else:
                logger.error("❌ calculate_raw_score返回非Series")
                return False
        except Exception as e:
            logger.error(f"❌ calculate_raw_score失败: {e}")
            return False
        
        # 测试边界情况
        logger.info("🧪 测试边界情况...")
        
        # 测试空数据
        empty_result = indicator.calculate(pd.DataFrame())
        if isinstance(empty_result, dict):
            logger.info(f"✅ 空数据处理成功: {empty_result}")
        else:
            logger.error("❌ 空数据处理失败")
            return False
        
        # 测试少量数据
        small_data = create_test_data(10)
        small_result = indicator.calculate(small_data)
        if isinstance(small_result, dict):
            logger.info(f"✅ 少量数据处理成功: {small_result}")
        else:
            logger.error("❌ 少量数据处理失败")
            return False
        
        logger.info("🎉 ZXM_CORRELATION_MATRIX指标修复测试通过！")
        return True
        
    except Exception as e:
        logger.error(f"❌ 测试过程中发生异常: {e}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
        return False


def main():
    """主函数"""
    success = test_fixed_correlation_matrix()
    
    if success:
        logger.info("✅ ZXM_CORRELATION_MATRIX指标修复测试成功！")
    else:
        logger.error("❌ ZXM_CORRELATION_MATRIX指标修复测试失败！")
    
    return success


if __name__ == "__main__":
    main()
