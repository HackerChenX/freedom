#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
简单测试工厂模式指标验证系统
确保基础功能正常工作
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.dependency_injection import get_logger

logger = get_logger(__name__)


def generate_simple_test_data() -> pd.DataFrame:
    """生成简单的测试数据"""
    logger.info("📊 生成简单测试数据...")
    
    # 生成50天的简单测试数据
    dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
    
    # 生成简单的价格序列
    base_price = 100.0
    prices = [base_price]
    
    np.random.seed(42)
    for i in range(1, 50):
        change = np.random.normal(0, 1)
        new_price = max(prices[-1] + change, 1.0)
        prices.append(new_price)
    
    # 生成OHLC数据
    data = []
    for i, price in enumerate(prices):
        high = price * (1 + np.random.uniform(0, 0.02))
        low = price * (1 - np.random.uniform(0, 0.02))
        open_price = prices[i-1] if i > 0 else price
        close = price
        volume = np.random.randint(1000000, 3000000)
        
        # 确保OHLC关系正确
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        data.append({
            'date': dates[i],
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    logger.info(f"✅ 生成简单测试数据: {len(df)}行")
    return df


def test_indicator_registry():
    """测试指标注册表功能"""
    logger.info("🔍 测试指标注册表功能...")
    
    try:
        from indicators.complete_indicator_registry import get_indicator_registry
        
        # 获取指标注册表实例
        registry = get_indicator_registry()
        logger.info("✅ 成功获取指标注册表")
        
        # 测试创建一个简单的指标
        test_indicators = ['ZXM_DAILY_MACD', 'DOJI', 'HAMMER']
        
        for indicator_name in test_indicators:
            try:
                indicator = registry.create_indicator(indicator_name)
                if indicator is not None:
                    logger.info(f"✅ 成功创建指标: {indicator_name}")
                    
                    # 测试基本方法
                    if hasattr(indicator, 'calculate'):
                        logger.info(f"✅ {indicator_name} 具有calculate方法")
                    else:
                        logger.warning(f"⚠️ {indicator_name} 缺少calculate方法")
                        
                else:
                    logger.warning(f"⚠️ 无法创建指标: {indicator_name}")
                    
            except Exception as e:
                logger.error(f"❌ 创建指标{indicator_name}失败: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 指标注册表测试失败: {e}")
        return False


def test_single_indicator_calculation(indicator_name: str, test_data: pd.DataFrame):
    """测试单个指标的计算功能"""
    logger.info(f"🔍 测试指标计算: {indicator_name}")
    
    try:
        from indicators.complete_indicator_registry import get_indicator_registry
        
        # 获取指标注册表实例
        registry = get_indicator_registry()
        
        # 创建指标实例
        indicator = registry.create_indicator(indicator_name)
        if indicator is None:
            logger.error(f"❌ 无法创建指标: {indicator_name}")
            return False
        
        # 测试计算功能
        result = indicator.calculate(test_data)
        
        if result is not None:
            logger.info(f"✅ {indicator_name} 计算成功")
            logger.info(f"📊 {indicator_name} 返回类型: {type(result)}")
            
            if isinstance(result, dict):
                logger.info(f"📊 {indicator_name} 返回dict，键数量: {len(result)}")
                if len(result) > 0:
                    sample_key = list(result.keys())[0]
                    sample_value = result[sample_key]
                    logger.info(f"📊 {indicator_name} 示例数据: {sample_key} = {type(sample_value)}")
            elif isinstance(result, pd.DataFrame):
                logger.info(f"📊 {indicator_name} 返回DataFrame，形状: {result.shape}")
                if not result.empty:
                    logger.info(f"📊 {indicator_name} 列名: {list(result.columns)}")
            
            return True
        else:
            logger.warning(f"⚠️ {indicator_name} 计算返回None")
            return False
            
    except Exception as e:
        logger.error(f"❌ {indicator_name} 计算测试失败: {e}")
        return False


def main():
    """主函数"""
    logger.info("🚀 开始简单测试工厂模式指标验证系统...")
    
    # 1. 生成测试数据
    test_data = generate_simple_test_data()
    
    # 2. 测试指标注册表
    registry_ok = test_indicator_registry()
    if not registry_ok:
        logger.error("❌ 指标注册表测试失败，停止测试")
        return False
    
    # 3. 测试几个具体指标的计算
    test_indicators = [
        'ZXM_DAILY_MACD',  # ZXM体系指标
        'DOJI',            # 形态识别指标
        'HAMMER'           # 形态识别指标
    ]
    
    success_count = 0
    total_count = len(test_indicators)
    
    for indicator_name in test_indicators:
        if test_single_indicator_calculation(indicator_name, test_data):
            success_count += 1
    
    # 4. 输出测试结果
    success_rate = (success_count / total_count) * 100
    logger.info(f"📊 测试结果: {success_count}/{total_count} 成功")
    logger.info(f"📊 成功率: {success_rate:.1f}%")
    
    if success_rate >= 66.7:  # 至少2/3成功
        logger.info("🎉 简单测试通过，系统基本功能正常")
        return True
    else:
        logger.warning("⚠️ 简单测试未完全通过，需要进一步调试")
        return False


if __name__ == "__main__":
    main()
