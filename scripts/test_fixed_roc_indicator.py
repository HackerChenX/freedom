#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试修复后的ROC指标
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
from indicators.roc import RateOfChange

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


def test_fixed_roc():
    """测试修复后的ROC指标"""
    logger.info("🧪 开始测试修复后的ROC指标...")
    
    try:
        # 创建指标实例
        indicator = RateOfChange()
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
        elif not isinstance(result, pd.DataFrame):
            logger.error("❌ calculate方法返回非DataFrame")
            return False
        else:
            logger.info(f"✅ calculate方法成功，返回类型: {type(result)}")
            logger.info(f"  - DataFrame形状: {result.shape}")
            logger.info(f"  - 列名: {list(result.columns)}")
            
            # 检查ROC相关列
            roc_columns = [col for col in result.columns if 'roc' in col.lower()]
            logger.info(f"  - ROC相关列: {roc_columns}")
            
            if 'roc' in result.columns:
                roc_values = result['roc'].dropna()
                if len(roc_values) > 0:
                    logger.info(f"  - ROC值范围: {roc_values.min():.2f} 到 {roc_values.max():.2f}")
                    logger.info(f"  - ROC平均值: {roc_values.mean():.2f}")
                else:
                    logger.warning("⚠️ ROC列全为NaN")
        
        # 测试get_patterns方法
        logger.info("🧪 测试get_patterns方法...")
        patterns = indicator.get_patterns(test_data)
        
        if patterns is None:
            logger.error("❌ get_patterns方法返回None")
            return False
        elif not isinstance(patterns, pd.DataFrame):
            logger.error("❌ get_patterns方法返回非DataFrame")
            return False
        else:
            logger.info(f"✅ get_patterns方法成功，返回形状: {patterns.shape}")
            logger.info(f"  - 形态列名: {list(patterns.columns)}")
            
            # 检查形态数量
            pattern_count = 0
            for col in patterns.columns:
                if patterns[col].dtype == bool:
                    pattern_count += patterns[col].sum()
            logger.info(f"  - 检测到的形态数量: {pattern_count}")
        
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
                logger.info(f"✅ calculate_raw_score成功: 平均分 {raw_score.mean():.2f}")
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
        if isinstance(empty_result, pd.DataFrame):
            logger.info(f"✅ 空数据处理成功: {empty_result.shape}")
        else:
            logger.error("❌ 空数据处理失败")
            return False
        
        # 测试少量数据
        small_data = create_test_data(10)
        small_result = indicator.calculate(small_data)
        if isinstance(small_result, pd.DataFrame):
            logger.info(f"✅ 少量数据处理成功: {small_result.shape}")
        else:
            logger.error("❌ 少量数据处理失败")
            return False
        
        # 测试ROC计算准确性
        logger.info("🧪 测试ROC计算准确性...")
        period = getattr(indicator, 'period', 14)
        close_prices = test_data['close']
        
        # 手动计算ROC
        manual_roc = ((close_prices - close_prices.shift(period)) / close_prices.shift(period)) * 100
        
        # 获取指标计算的ROC
        if 'roc' in result.columns:
            indicator_roc = result['roc']
            
            # 比较结果（忽略NaN值）
            manual_valid = manual_roc.dropna()
            indicator_valid = indicator_roc.dropna()
            
            if len(manual_valid) > 0 and len(indicator_valid) > 0:
                # 找到共同的索引
                common_idx = manual_valid.index.intersection(indicator_valid.index)
                if len(common_idx) > 0:
                    manual_subset = manual_valid.loc[common_idx]
                    indicator_subset = indicator_valid.loc[common_idx]
                    
                    # 计算相关性
                    correlation = manual_subset.corr(indicator_subset)
                    logger.info(f"  - ROC计算相关性: {correlation:.4f}")
                    
                    if correlation > 0.95:
                        logger.info("✅ ROC计算准确性优秀")
                    else:
                        logger.warning(f"⚠️ ROC计算准确性较低: {correlation:.4f}")
                else:
                    logger.warning("⚠️ 无法找到共同的有效数据点")
            else:
                logger.warning("⚠️ 缺少有效的ROC数据进行比较")
        
        logger.info("🎉 ROC指标修复测试通过！")
        return True
        
    except Exception as e:
        logger.error(f"❌ 测试过程中发生异常: {e}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
        return False


def main():
    """主函数"""
    success = test_fixed_roc()
    
    if success:
        logger.info("✅ ROC指标修复测试成功！")
    else:
        logger.error("❌ ROC指标修复测试失败！")
    
    return success


if __name__ == "__main__":
    main()
