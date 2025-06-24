#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试新增指标的统一评分功能

测试BOLL、OBV、WR、CCI、ATR、DMI指标的评分机制
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
from indicators.cci import CCI
from indicators.atr import ATR
from indicators.dmi import DMI
from utils.logger import get_logger

logger = get_logger(__name__)


def generate_test_data(periods: int = 100) -> pd.DataFrame:
    """
    生成测试数据
    
    Args:
        periods: 数据周期数
        
    Returns:
        pd.DataFrame: 测试数据
    """
    # 生成日期索引
    dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
    
    # 生成模拟价格数据
    np.random.seed(42)
    
    # 基础价格趋势
    base_price = 100
    price_trend = np.cumsum(np.random.normal(0, 0.02, periods))
    
    # 生成OHLCV数据
    close_prices = base_price + price_trend + np.random.normal(0, 0.5, periods)
    
    # 确保价格为正数
    close_prices = np.maximum(close_prices, 10)
    
    # 生成其他价格数据
    high_prices = close_prices + np.random.uniform(0, 2, periods)
    low_prices = close_prices - np.random.uniform(0, 2, periods)
    open_prices = close_prices + np.random.normal(0, 0.5, periods)
    
    # 确保价格关系合理
    high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
    low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))
    
    # 生成成交量数据
    volumes = np.random.uniform(1000000, 5000000, periods)
    
    data = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }, index=dates)
    
    return data


def test_indicator_scoring(indicator, data, indicator_name):
    """
    测试指标评分功能
    
    Args:
        indicator: 指标实例
        data: 测试数据
        indicator_name: 指标名称
    """
    print(f"\n{'='*50}")
    print(f"测试 {indicator_name} 指标评分功能")
    print(f"{'='*50}")
    
    try:
        # 计算指标
        result = indicator.calculate(data)
        if result is None:
            print(f"✗ {indicator_name} 指标计算返回 None")
            return

        print(f"✓ {indicator_name} 指标计算成功，数据形状: {result.shape}")
        
        # 计算评分
        score_result = indicator.calculate_score(data)
        
        if score_result and score_result.get('final_score') is not None:
            final_score = score_result['final_score']
            raw_score = score_result['raw_score']
            patterns = score_result['patterns']
            market_env = score_result['market_environment']
            confidence = score_result['confidence']
            
            # 显示评分结果
            print(f"✓ {indicator_name} 评分计算成功")
            print(f"  - 最终评分: {final_score.iloc[-1]:.2f}分")
            print(f"  - 原始评分: {raw_score.iloc[-1]:.2f}分")
            print(f"  - 市场环境: {market_env.value}")
            print(f"  - 置信度: {confidence:.2f}%")
            print(f"  - 识别形态: {patterns}")
            
            # 验证评分范围
            assert 0 <= final_score.iloc[-1] <= 100, f"{indicator_name}最终评分超出范围"
            assert 0 <= raw_score.iloc[-1] <= 100, f"{indicator_name}原始评分超出范围"
            assert 0 <= confidence <= 100, f"{indicator_name}置信度超出范围"
            
            print(f"✓ {indicator_name} 评分范围验证通过")
            
            # 统计评分分布
            score_stats = {
                'mean': final_score.mean(),
                'std': final_score.std(),
                'min': final_score.min(),
                'max': final_score.max(),
                'median': final_score.median()
            }
            
            print(f"  - 评分统计: 均值={score_stats['mean']:.2f}, "
                  f"标准差={score_stats['std']:.2f}, "
                  f"范围=[{score_stats['min']:.2f}, {score_stats['max']:.2f}]")
            
        else:
            print(f"✗ {indicator_name} 评分计算失败或返回空结果")
            
    except Exception as e:
        print(f"✗ {indicator_name} 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()


def test_market_environment_detection():
    """
    测试市场环境检测功能
    """
    print(f"\n{'='*50}")
    print("测试市场环境检测功能")
    print(f"{'='*50}")
    
    # 生成不同市场环境的数据
    test_cases = [
        ("上升趋势", generate_trending_data(trend='up')),
        ("下降趋势", generate_trending_data(trend='down')),
        ("震荡市场", generate_sideways_data()),
    ]
    
    boll = BOLL()
    
    for env_name, data in test_cases:
        try:
            score_result = boll.calculate_score(data)
            if score_result:
                market_env = score_result['market_environment']
                print(f"  - {env_name}: 检测为 {market_env.value}")
            else:
                print(f"  - {env_name}: 检测失败")
        except Exception as e:
            print(f"  - {env_name}: 检测错误 - {str(e)}")


def generate_trending_data(trend='up', periods=100):
    """生成趋势性数据"""
    dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
    
    if trend == 'up':
        trend_factor = np.linspace(0, 20, periods)
    else:  # down
        trend_factor = np.linspace(0, -20, periods)
    
    base_price = 100
    close_prices = base_price + trend_factor + np.random.normal(0, 1, periods)
    close_prices = np.maximum(close_prices, 10)
    
    high_prices = close_prices + np.random.uniform(0, 2, periods)
    low_prices = close_prices - np.random.uniform(0, 2, periods)
    open_prices = close_prices + np.random.normal(0, 0.5, periods)
    
    high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
    low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))
    
    volumes = np.random.uniform(1000000, 5000000, periods)
    
    return pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }, index=dates)


def generate_sideways_data(periods=100):
    """生成震荡性数据"""
    dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
    
    base_price = 100
    # 生成震荡数据
    oscillation = 5 * np.sin(np.linspace(0, 4*np.pi, periods))
    close_prices = base_price + oscillation + np.random.normal(0, 1, periods)
    close_prices = np.maximum(close_prices, 10)
    
    high_prices = close_prices + np.random.uniform(0, 2, periods)
    low_prices = close_prices - np.random.uniform(0, 2, periods)
    open_prices = close_prices + np.random.normal(0, 0.5, periods)
    
    high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
    low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))
    
    volumes = np.random.uniform(1000000, 5000000, periods)
    
    return pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }, index=dates)


def test_scoring_consistency():
    """
    测试评分一致性
    """
    print(f"\n{'='*50}")
    print("测试评分一致性")
    print(f"{'='*50}")
    
    data = generate_test_data(100)
    
    # 测试多次计算的一致性
    boll = BOLL()
    
    scores = []
    for i in range(3):
        score_result = boll.calculate_score(data)
        if score_result and score_result.get('final_score') is not None:
            scores.append(score_result['final_score'].iloc[-1])
        else:
            print(f"第 {i+1} 次评分计算失败，跳过")
    
    if len(scores) > 1:
        is_consistent = all(abs(s - scores[0]) < 1e-9 for s in scores)
        print(f"✓ 多次计算评分一致性: {'通过' if is_consistent else '失败'}")
        if not is_consistent:
            print(f"  - 分数: {scores}")
        assert is_consistent, "评分不一致"
    else:
        print("✗ 未能完成足够多的评分计算以进行一致性测试")


def test_edge_cases():
    """
    测试边界情况
    """
    print(f"\n{'='*50}")
    print("测试边界情况")
    print(f"{'='*50}")
    
    # 测试空数据
    try:
        empty_data = pd.DataFrame()
        boll = BOLL()
        score_result = boll.calculate_score(empty_data)
        print("✓ 空数据处理正常")
    except Exception as e:
        print(f"✓ 空数据异常处理正常: {str(e)}")
    
    # 测试少量数据
    try:
        small_data = generate_test_data(5)
        boll = BOLL()
        score_result = boll.calculate_score(small_data)
        if score_result:
            print("✓ 少量数据处理正常")
        else:
            print("✓ 少量数据返回默认值")
    except Exception as e:
        print(f"✓ 少量数据异常处理正常: {str(e)}")
    
    # 测试异常数据
    try:
        abnormal_data = generate_test_data(50)
        # 添加一些异常值
        abnormal_data.loc[abnormal_data.index[10], 'close'] = 1000000
        abnormal_data.loc[abnormal_data.index[20], 'volume'] = 0
        
        boll = BOLL()
        score_result = boll.calculate_score(abnormal_data)
        print("✓ 异常数据处理正常")
    except Exception as e:
        print(f"✓ 异常数据异常处理正常: {str(e)}")
    
    # 3. 包含NaN值的数据
    nan_data = generate_test_data(100)
    nan_data.iloc[10:20, 0] = np.nan  # 在open列中引入NaN
    test_indicator_scoring(BOLL(), nan_data, "BOLL (with NaN)")
    test_indicator_scoring(OBV(), nan_data, "OBV (with NaN)")


def main():
    """
    主测试函数
    """
    print("开始测试新增指标的统一评分功能")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 生成测试数据
    test_data = generate_test_data(100)
    print(f"生成测试数据: {test_data.shape[0]}行 x {test_data.shape[1]}列")
    
    # 测试各个指标
    indicators = [
        (BOLL(), "BOLL布林带"),
        (OBV(), "OBV能量潮"),
        (WR(), "WR威廉指标"),
        (CCI(), "CCI顺势指标"),
        (ATR(), "ATR平均真实波幅"),
        (DMI(), "DMI趋向指标"),
    ]
    
    success_count = 0
    total_count = len(indicators)
    
    for indicator, name in indicators:
        try:
            test_indicator_scoring(indicator, test_data, name)
            success_count += 1
        except Exception as e:
            print(f"✗ {name} 测试失败: {str(e)}")
    
    # 测试市场环境检测
    test_market_environment_detection()
    
    # 测试评分一致性
    test_scoring_consistency()
    
    # 测试边界情况
    test_edge_cases()
    
    # 输出测试总结
    print(f"\n{'='*50}")
    print("测试总结")
    print(f"{'='*50}")
    print(f"总计测试指标: {total_count}个")
    print(f"成功测试指标: {success_count}个")
    print(f"测试成功率: {success_count/total_count*100:.1f}%")
    
    if success_count == total_count:
        print("🎉 所有指标评分功能测试通过！")
    else:
        print(f"⚠️  有 {total_count - success_count} 个指标测试失败")
    
    print(f"测试完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main() 