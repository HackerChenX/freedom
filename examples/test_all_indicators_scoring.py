#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试所有指标评分功能

验证所有已实现评分功能的指标是否正常工作
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from indicators.complete_indicator_registry import complete_registry
from indicators.complete_indicator_registry import complete_registry
from indicators.complete_indicator_registry import complete_registry
from indicators.complete_indicator_registry import complete_registry
from indicators.complete_indicator_registry import complete_registry
from indicators.kdj import KDJ
from indicators.rsi import RSI
from indicators.boll import BOLL
from indicators.obv import OBV
from indicators.wr import WR
from indicators.cci import CCI
from indicators.atr import ATR
from indicators.dmi import DMI
from utils.logger import get_logger

logger = get_logger(__name__)


def generate_test_data(length: int = 100) -> pd.DataFrame:
    """
    生成测试数据
    
    Args:
        length: 数据长度
        
    Returns:
        pd.DataFrame: 测试数据
    """
    np.random.seed(42)
    
    # 生成日期索引
    dates = pd.date_range(start='2023-01-01', periods=length, freq='D')
    
    # 生成价格数据（模拟上升趋势）
    base_price = 100
    price_changes = np.random.normal(0.5, 2, length)  # 轻微上升趋势
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change / 100)
        prices.append(max(new_price, prices[-1] * 0.95))  # 防止价格过度下跌
    
    # 生成OHLC数据
    close_prices = np.array(prices)
    high_prices = close_prices * (1 + np.random.uniform(0, 0.03, length))
    low_prices = close_prices * (1 - np.random.uniform(0, 0.03, length))
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]
    
    # 生成成交量
    volumes = np.random.uniform(1000000, 5000000, length)
    
    data = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }, index=dates)
    
    return data


def test_indicator_scoring(indicator_name: str, indicator_instance, data: pd.DataFrame) -> bool:
    """
    测试单个指标的评分功能
    
    Args:
        indicator_name: 指标名称
        indicator_instance: 指标实例
        data: 测试数据
        
    Returns:
        bool: 测试是否成功
    """
    try:
        # 计算评分
        score_result = indicator_instance.calculate_score(data)
        
        print(f"✅ {indicator_name}评分计算成功")
        print(f"  评分范围: {score_result['final_score'].min():.2f} - {score_result['final_score'].max():.2f}")
        print(f"  平均评分: {score_result['final_score'].mean():.2f}")
        print(f"  标准差: {score_result['final_score'].std():.2f}")
        print(f"  市场环境: {score_result['market_environment']}")
        print(f"  置信度: {score_result['confidence']:.2f}")
        print(f"  识别形态数量: {len(score_result['patterns'])}")
        
        # 验证评分范围
        assert 0 <= score_result['final_score'].min() <= 100, f"{indicator_name}评分超出范围"
        assert 0 <= score_result['final_score'].max() <= 100, f"{indicator_name}评分超出范围"
        
        return True
        
    except Exception as e:
        print(f"❌ {indicator_name}评分测试失败: {str(e)}")
        return False


def main():
    """主函数"""
    print("开始测试所有指标评分功能")
    print("=" * 80)
    
    # 生成测试数据
    data = generate_test_data(100)
    
    # 定义所有已实现评分功能的指标
    indicators = [
        ("MACD", MACD()),
        ("KDJ", KDJ()),
        ("RSI", RSI()),
        ("BOLL", BOLL()),
        ("OBV", OBV()),
        ("WR", WR()),
        ("CCI", CCI()),
        ("ATR", ATR()),
        ("DMI", DMI()),
        ("MA", MA(periods=[5, 10, 20])),
        ("EMA", EMA(periods=[5, 10, 20])),
        ("SAR", SAR()),
        ("TRIX", TRIX()),
    ]
    
    test_results = []
    
    # 测试每个指标
    for indicator_name, indicator_instance in indicators:
        print(f"\n{'='*50}")
        print(f"测试{indicator_name}指标评分功能")
        print(f"{'='*50}")
        
        result = test_indicator_scoring(indicator_name, indicator_instance, data)
        test_results.append((indicator_name, result))
    
    # 汇总测试结果
    print("\n" + "=" * 80)
    print("测试结果汇总")
    print("=" * 80)
    
    passed = 0
    total = len(test_results)
    
    for indicator_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{indicator_name}: {status}")
        if result:
            passed += 1
    
    print("=" * 80)
    print(f"测试完成: {passed}/{total} 个指标测试通过")
    
    if passed == total:
        print("🎉 所有指标评分功能正常工作！")
    else:
        print("⚠️ 部分指标测试失败，需要检查和修复")
    
    # 测试评分一致性
    print("\n" + "=" * 80)
    print("测试评分一致性")
    print("=" * 80)
    
    market_environments = []
    
    for indicator_name, indicator_instance in indicators[:5]:  # 测试前5个指标
        try:
            score_result = indicator_instance.calculate_score(data)
            market_environments.append(score_result['market_environment'])
            print(f"{indicator_name}: {score_result['market_environment']}")
        except Exception as e:
            print(f"{indicator_name}: 评分失败 - {str(e)}")
    
    # 检查市场环境检测一致性
    if len(set(market_environments)) <= 2:  # 允许少量差异
        print("✅ 市场环境检测基本一致")
    else:
        print("⚠️ 市场环境检测存在较大差异")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 