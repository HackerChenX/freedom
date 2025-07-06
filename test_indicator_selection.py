#!/usr/bin/env python3
"""
测试所有指标的选股功能
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

from indicators.complete_indicator_registry import Complete_indicator_registry
from utils.logger import get_logger

logger = get_logger(__name__)

def generate_test_data():
    """生成测试数据"""
    np.random.seed(42)
    n_days = 100
    dates = pd.date_range('2024-01-01', periods=n_days, freq='D')
    
    # 模拟价格数据
    base_price = 100
    price_changes = np.random.normal(0, 0.02, n_days)
    closes = [base_price]
    for change in price_changes[1:]:
        closes.append(closes[-1] * (1 + change))
    
    # 创建OHLCV数据
    data = pd.DataFrame({
        'trade_date': dates,
        'open': [c * (1 + np.random.normal(0, 0.005)) for c in closes],
        'high': [c * (1 + abs(np.random.normal(0, 0.01))) for c in closes],
        'low': [c * (1 - abs(np.random.normal(0, 0.01))) for c in closes],
        'close': closes,
        'volume': np.random.randint(1000000, 10000000, n_days)
    })
    
    # 确保OHLC逻辑正确
    for i in range(len(data)):
        data.loc[i, 'high'] = max(data.loc[i, 'open'], data.loc[i, 'high'], data.loc[i, 'close'])
        data.loc[i, 'low'] = min(data.loc[i, 'open'], data.loc[i, 'low'], data.loc[i, 'close'])
    
    return data

def test_all_indicators():
    """测试所有指标的选股功能"""
    print("=== 开始测试所有指标的选股功能 ===")
    
    # 初始化指标注册表
    registry = Complete_indicator_registry()
    
    # 生成测试数据
    test_data = generate_test_data()
    print(f"测试数据生成完成，共{len(test_data)}行")
    
    # 获取所有已注册的指标
    all_indicators = registry.get_indicator_names()
    print(f"共有{len(all_indicators)}个指标待测试")
    
    # 测试结果统计
    success_count = 0
    fail_count = 0
    failed_indicators = []
    
    for indicator_name in all_indicators:
        try:
            print(f"\n--- 测试指标: {indicator_name} ---")
            
            # 直接创建指标实例
            indicator = registry.create_indicator(indicator_name)
            
            # 计算指标
            result = indicator.calculate(test_data)
            
            # 计算评分
            score = indicator.calculate_raw_score(test_data)
            
            # 检查结果
            if score is not None and len(score) > 0:
                # 获取最后一个有效评分
                last_score = score.iloc[-1]
                if pd.isna(last_score):
                    print(f"❌ {indicator_name}: 评分为NaN")
                    fail_count += 1
                    failed_indicators.append(f"{indicator_name} (评分NaN)")
                else:
                    print(f"✅ {indicator_name}: 评分={last_score:.2f}")
                    success_count += 1
            else:
                print(f"❌ {indicator_name}: 无评分结果")
                fail_count += 1
                failed_indicators.append(f"{indicator_name} (无结果)")
                
        except Exception as e:
            print(f"❌ {indicator_name}: 异常 - {str(e)}")
            fail_count += 1
            failed_indicators.append(f"{indicator_name} (异常: {str(e)})")
    
    # 输出测试总结
    print(f"\n=== 测试总结 ===")
    print(f"总指标数: {len(all_indicators)}")
    print(f"成功指标: {success_count}")
    print(f"失败指标: {fail_count}")
    print(f"成功率: {success_count/len(all_indicators)*100:.1f}%")
    
    if failed_indicators:
        print(f"\n失败指标列表:")
        for indicator in failed_indicators:
            print(f"  - {indicator}")
    
    return success_count, fail_count

if __name__ == "__main__":
    test_all_indicators() 