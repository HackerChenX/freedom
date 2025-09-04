#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
快速验证高优先级指标是否已经工作
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
    
    dates = pd.date_range('2024-01-01', periods=length, freq='D')
    
    # 生成价格序列
    initial_price = 100.0
    returns = np.random.normal(0.001, 0.02, length)
    prices = [initial_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    close_prices = np.array(prices[1:])
    data_length = len(close_prices)
    
    high_prices = []
    low_prices = []
    open_prices = []
    
    for i in range(data_length):
        close = close_prices[i]
        open_price = close + np.random.normal(0, 0.5)
        high_price = max(open_price, close) + abs(np.random.normal(0, 0.3))
        low_price = min(open_price, close) - abs(np.random.normal(0, 0.3))
        
        open_prices.append(open_price)
        high_prices.append(high_price)
        low_prices.append(low_price)
    
    volumes = np.random.lognormal(10, 0.3, data_length)
    
    data = pd.DataFrame({
        'date': dates[:data_length],
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    })
    
    return data


def quick_validate_indicators():
    """快速验证高优先级指标"""
    logger.info("🚀 快速验证高优先级指标...")
    
    # 高优先级指标列表
    high_priority_indicators = [
        'BIAS',
        'DMA', 
        'WMA',
        'MOMENTUM',
        'PSY',
        'WR',
        'VORTEX',
        'CHAIKIN'
    ]
    
    test_data = create_test_data(100)
    logger.info(f"✅ 测试数据创建成功: {len(test_data)} 行")
    
    working_indicators = []
    failed_indicators = []
    
    try:
        from indicators.complete_indicator_registry import get_indicator_registry
        registry = get_indicator_registry()
        
        for indicator_name in high_priority_indicators:
            logger.info(f"🔍 测试指标: {indicator_name}")
            
            try:
                # 尝试创建指标实例
                indicator = registry.create_indicator(indicator_name)
                
                if indicator:
                    # 尝试计算
                    result = indicator.calculate(test_data)
                    
                    if result is not None and not result.empty:
                        logger.info(f"  ✅ {indicator_name}: 工作正常，结果形状 {result.shape}")
                        working_indicators.append(indicator_name)
                    else:
                        logger.warning(f"  ⚠️ {indicator_name}: 计算结果为空")
                        failed_indicators.append(indicator_name)
                else:
                    logger.warning(f"  ⚠️ {indicator_name}: 无法创建实例")
                    failed_indicators.append(indicator_name)
                    
            except Exception as e:
                logger.error(f"  ❌ {indicator_name}: 测试失败 - {e}")
                failed_indicators.append(indicator_name)
        
        # 生成报告
        logger.info("=" * 60)
        logger.info("📊 高优先级指标验证报告")
        logger.info("=" * 60)
        logger.info(f"总测试指标数: {len(high_priority_indicators)}")
        logger.info(f"工作正常指标: {len(working_indicators)}")
        logger.info(f"需要修复指标: {len(failed_indicators)}")
        logger.info(f"工作正常率: {len(working_indicators)/len(high_priority_indicators)*100:.1f}%")
        
        if working_indicators:
            logger.info(f"✅ 工作正常的指标: {working_indicators}")
        
        if failed_indicators:
            logger.info(f"❌ 需要修复的指标: {failed_indicators}")
        
        logger.info("=" * 60)
        
        return {
            'total': len(high_priority_indicators),
            'working': len(working_indicators),
            'failed': len(failed_indicators),
            'working_list': working_indicators,
            'failed_list': failed_indicators
        }
        
    except Exception as e:
        logger.error(f"❌ 验证过程中发生异常: {e}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
        return None


def main():
    """主函数"""
    try:
        result = quick_validate_indicators()
        
        if result:
            logger.info("✅ 高优先级指标验证完成")
            logger.info(f"📊 工作正常率: {result['working']}/{result['total']} ({result['working']/result['total']*100:.1f}%)")
            
            if result['working'] > 0:
                logger.info("🎉 发现更多已工作的指标，项目完成率将进一步提升！")
        else:
            logger.error("❌ 高优先级指标验证失败")
        
        return result is not None
        
    except Exception as e:
        logger.error(f"❌ 主函数执行失败: {e}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    main()
