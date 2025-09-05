#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
调试ROC指标实现
"""

import sys
import os
import pandas as pd
import numpy as np

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.dependency_injection import get_logger

logger = get_logger(__name__)


def debug_roc():
    """调试ROC指标"""
    logger.info("🔍 调试ROC指标...")
    
    # 生成简单测试数据
    dates = pd.date_range(start='2024-01-01', periods=20, freq='D')
    prices = [100, 102, 104, 103, 105, 107, 106, 108, 110, 109, 
              111, 113, 112, 114, 116, 115, 117, 119, 118, 120]
    
    data = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': [p + 1 for p in prices],
        'low': [p - 1 for p in prices],
        'close': prices,
        'volume': [1000000] * 20
    })
    
    logger.info(f"测试数据:\n{data[['date', 'close']].head(10)}")
    
    # 测试ROC指标
    try:
        from indicators.roc import RateOfChange
        
        roc_indicator = RateOfChange(period=5)
        result = roc_indicator.calculate(data)
        
        logger.info(f"ROC计算结果列: {result.columns.tolist()}")
        logger.info(f"ROC结果形状: {result.shape}")
        
        # 查看ROC相关列
        roc_cols = [col for col in result.columns if 'roc' in col.lower()]
        logger.info(f"ROC相关列: {roc_cols}")
        
        if roc_cols:
            for col in roc_cols:
                values = result[col].dropna()
                logger.info(f"{col} 值: {values.tail(10).tolist()}")
        
        # 手动计算ROC验证
        period = 5
        manual_roc = []
        for i in range(period, len(data)):
            current_price = data.iloc[i]['close']
            past_price = data.iloc[i-period]['close']
            if past_price != 0:
                roc_val = ((current_price - past_price) / past_price) * 100
                manual_roc.append(roc_val)
            else:
                manual_roc.append(0)
        
        logger.info(f"手动计算ROC: {manual_roc}")
        
        # 比较结果
        if 'roc' in result.columns:
            calc_roc = result['roc'].dropna().tolist()
            logger.info(f"指标计算ROC: {calc_roc}")
            
            if len(calc_roc) >= len(manual_roc):
                # 比较最后几个值
                calc_subset = calc_roc[-len(manual_roc):]
                differences = [abs(a - b) for a, b in zip(calc_subset, manual_roc)]
                max_diff = max(differences) if differences else 0
                logger.info(f"最大差异: {max_diff}")
                
                if max_diff < 0.01:
                    logger.info("✅ ROC计算正确")
                else:
                    logger.warning("⚠️ ROC计算可能有误")
        
        # 测试参数
        logger.info(f"ROC指标参数: period={getattr(roc_indicator, 'period', 'N/A')}")
        
        # 测试方法
        methods = ['get_signals', 'get_patterns', 'calculate_raw_score']
        for method in methods:
            if hasattr(roc_indicator, method):
                try:
                    result_method = getattr(roc_indicator, method)(data)
                    logger.info(f"✅ {method} 方法可用: {type(result_method)}")
                except Exception as e:
                    logger.warning(f"⚠️ {method} 方法失败: {e}")
            else:
                logger.warning(f"❌ {method} 方法不存在")
        
    except Exception as e:
        logger.error(f"❌ ROC调试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    debug_roc()
