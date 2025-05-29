#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试MA、EMA、SAR指标评分功能

验证新增指标的统一评分系统是否正常工作
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from indicators.ma import MA
from indicators.ema import EMA
from indicators.sar import SAR
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


def test_ma_scoring():
    """测试MA指标评分功能"""
    print("=" * 50)
    print("测试MA指标评分功能")
    print("=" * 50)
    
    # 生成测试数据
    data = generate_test_data(100)
    
    # 创建MA指标
    ma_indicator = MA(periods=[5, 10, 20, 30])
    
    try:
        # 计算评分
        score_result = ma_indicator.calculate_score(data)
        
        print(f"✅ MA评分计算成功")
        print(f"评分范围: {score_result['final_score'].min():.2f} - {score_result['final_score'].max():.2f}")
        print(f"平均评分: {score_result['final_score'].mean():.2f}")
        print(f"市场环境: {score_result['market_environment']}")
        print(f"置信度: {score_result['confidence']:.2f}")
        print(f"识别形态数量: {len(score_result['patterns'])}")
        print(f"识别的形态: {score_result['patterns'][:5]}")  # 显示前5个形态
        
        # 验证评分范围
        assert 0 <= score_result['final_score'].min() <= 100, "评分超出范围"
        assert 0 <= score_result['final_score'].max() <= 100, "评分超出范围"
        
        return True
        
    except Exception as e:
        print(f"❌ MA评分测试失败: {str(e)}")
        return False


def test_ema_scoring():
    """测试EMA指标评分功能"""
    print("=" * 50)
    print("测试EMA指标评分功能")
    print("=" * 50)
    
    # 生成测试数据
    data = generate_test_data(100)
    
    # 创建EMA指标
    ema_indicator = EMA(periods=[5, 10, 20, 30])
    
    try:
        # 计算评分
        score_result = ema_indicator.calculate_score(data)
        
        print(f"✅ EMA评分计算成功")
        print(f"评分范围: {score_result['final_score'].min():.2f} - {score_result['final_score'].max():.2f}")
        print(f"平均评分: {score_result['final_score'].mean():.2f}")
        print(f"市场环境: {score_result['market_environment']}")
        print(f"置信度: {score_result['confidence']:.2f}")
        print(f"识别形态数量: {len(score_result['patterns'])}")
        print(f"识别的形态: {score_result['patterns'][:5]}")  # 显示前5个形态
        
        # 验证评分范围
        assert 0 <= score_result['final_score'].min() <= 100, "评分超出范围"
        assert 0 <= score_result['final_score'].max() <= 100, "评分超出范围"
        
        return True
        
    except Exception as e:
        print(f"❌ EMA评分测试失败: {str(e)}")
        return False


def test_sar_scoring():
    """测试SAR指标评分功能"""
    print("=" * 50)
    print("测试SAR指标评分功能")
    print("=" * 50)
    
    # 生成测试数据
    data = generate_test_data(100)
    
    # 创建SAR指标
    sar_indicator = SAR(acceleration=0.02, maximum=0.2)
    
    try:
        # 计算评分
        score_result = sar_indicator.calculate_score(data)
        
        print(f"✅ SAR评分计算成功")
        print(f"评分范围: {score_result['final_score'].min():.2f} - {score_result['final_score'].max():.2f}")
        print(f"平均评分: {score_result['final_score'].mean():.2f}")
        print(f"市场环境: {score_result['market_environment']}")
        print(f"置信度: {score_result['confidence']:.2f}")
        print(f"识别形态数量: {len(score_result['patterns'])}")
        print(f"识别的形态: {score_result['patterns'][:5]}")  # 显示前5个形态
        
        # 验证评分范围
        assert 0 <= score_result['final_score'].min() <= 100, "评分超出范围"
        assert 0 <= score_result['final_score'].max() <= 100, "评分超出范围"
        
        return True
        
    except Exception as e:
        print(f"❌ SAR评分测试失败: {str(e)}")
        return False


def test_scoring_consistency():
    """测试评分一致性"""
    print("=" * 50)
    print("测试评分一致性")
    print("=" * 50)
    
    # 生成测试数据
    data = generate_test_data(100)
    
    indicators = [
        ("MA", MA(periods=[5, 10, 20])),
        ("EMA", EMA(periods=[5, 10, 20])),
        ("SAR", SAR())
    ]
    
    results = {}
    
    for name, indicator in indicators:
        try:
            score_result = indicator.calculate_score(data)
            results[name] = {
                'mean_score': score_result['final_score'].mean(),
                'std_score': score_result['final_score'].std(),
                'market_env': score_result['market_environment'],
                'confidence': score_result['confidence'],
                'pattern_count': len(score_result['patterns'])
            }
            print(f"✅ {name}: 平均{results[name]['mean_score']:.2f}分, "
                  f"标准差{results[name]['std_score']:.2f}, "
                  f"置信度{results[name]['confidence']:.2f}")
        except Exception as e:
            print(f"❌ {name}评分失败: {str(e)}")
            return False
    
    # 检查市场环境检测一致性
    market_envs = [result['market_env'] for result in results.values()]
    if len(set(market_envs)) == 1:
        print(f"✅ 市场环境检测一致: {market_envs[0]}")
    else:
        print(f"⚠️ 市场环境检测不一致: {market_envs}")
    
    return True


def test_edge_cases():
    """测试边界情况"""
    print("=" * 50)
    print("测试边界情况")
    print("=" * 50)
    
    # 测试空数据
    try:
        empty_data = pd.DataFrame()
        ma_indicator = MA(periods=[5, 10])
        score_result = ma_indicator.calculate_score(empty_data)
        print("❌ 空数据应该抛出异常")
        return False
    except Exception:
        print("✅ 空数据正确处理")
    
    # 测试数据不足
    try:
        small_data = generate_test_data(3)
        ma_indicator = MA(periods=[5, 10])
        score_result = ma_indicator.calculate_score(small_data)
        print("✅ 小数据集正确处理")
    except Exception as e:
        print(f"⚠️ 小数据集处理异常: {str(e)}")
    
    # 测试包含NaN的数据
    try:
        nan_data = generate_test_data(50)
        nan_data.loc[nan_data.index[10:15], 'close'] = np.nan
        ma_indicator = MA(periods=[5, 10])
        score_result = ma_indicator.calculate_score(nan_data)
        print("✅ NaN数据正确处理")
    except Exception as e:
        print(f"⚠️ NaN数据处理异常: {str(e)}")
    
    return True


def main():
    """主函数"""
    print("开始测试MA、EMA、SAR指标评分功能")
    print("=" * 80)
    
    test_results = []
    
    # 运行各项测试
    test_results.append(("MA评分功能", test_ma_scoring()))
    test_results.append(("EMA评分功能", test_ema_scoring()))
    test_results.append(("SAR评分功能", test_sar_scoring()))
    test_results.append(("评分一致性", test_scoring_consistency()))
    test_results.append(("边界情况", test_edge_cases()))
    
    # 汇总测试结果
    print("=" * 80)
    print("测试结果汇总")
    print("=" * 80)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print("=" * 80)
    print(f"测试完成: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！MA、EMA、SAR指标评分功能正常工作")
    else:
        print("⚠️ 部分测试失败，需要检查和修复")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 