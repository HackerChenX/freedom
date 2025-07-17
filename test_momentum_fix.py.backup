#!/usr/bin/env python3
"""
MOMENTUM指标修复验证测试

验证MOMENTUM指标修复后的功能：
1. 计算功能是否正常
2. 评分是否动态变化
3. 评分分布是否合理
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

from indicators.momentum import MOMENTUM

def create_test_data_Fix_Test_Momentum_Fix():
    """创建测试数据"""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    # 创建有不同动量特征的价格数据
    np.random.seed(42)
    base_price = 100
    prices = []
    
    for i in range(100):
        if i < 15:
            # 前15天：加速上涨
            price = base_price + i * 2.0 + (i * 0.1) ** 2 + np.random.normal(0, 0.3)
        elif i < 30:
            # 15-30天：减速上涨
            price = base_price + 30 + (i - 15) * 0.8 - (i - 15) * 0.02 + np.random.normal(0, 0.5)
        elif i < 50:
            # 30-50天：横盘震荡
            price = base_price + 42 + np.sin((i - 30) * 0.3) * 2 + np.random.normal(0, 1)
        elif i < 70:
            # 50-70天：加速下跌
            price = base_price + 42 - (i - 50) * 1.5 - (i - 50) * 0.05 + np.random.normal(0, 0.5)
        else:
            # 70-100天：减速下跌然后反弹
            if i < 85:
                price = base_price + 12 - (i - 70) * 0.3 + np.random.normal(0, 0.5)
            else:
                price = base_price + 7.5 + (i - 85) * 0.8 + np.random.normal(0, 0.3)
        
        prices.append(max(price, 10))  # 确保价格不为负
    
    # 创建OHLCV数据
    data = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': [p + abs(np.random.normal(0, 0.5)) for p in prices],
        'low': [p - abs(np.random.normal(0, 0.5)) for p in prices],
        'close': prices,
        'volume': [1000000 + np.random.randint(-200000, 200000) for _ in range(100)]
    })
    
    # 确保high >= low
    data['high'] = np.maximum(data['high'], data['close'])
    data['low'] = np.minimum(data['low'], data['close'])
    
    return data

def test_momentum_calculation():
    """测试MOMENTUM计算功能"""
    print("=== MOMENTUM计算功能测试 ===")
    
    # 创建测试数据
    data = create_test_data_Fix_Test_Momentum_Fix()
    
    # 创建MOMENTUM指标实例
    momentum = MOMENTUM(period=14)
    
    # 计算指标
    result = momentum.calculate(data)
    
    # 检查计算结果
    print(f"数据行数: {len(result)}")
    print(f"包含的列: {list(result.columns)}")
    
    # 检查必要的列是否存在
    required_columns = ['momentum', 'momentum_ma', 'momentum_std', 'momentum_ratio']
    for col in required_columns:
        if col in result.columns:
            print(f"✓ {col} 列存在")
        else:
            print(f"✗ {col} 列不存在")
    
    # 检查数值范围
    momentum_values = result['momentum'].dropna()
    momentum_ma_values = result['momentum_ma'].dropna()
    momentum_std_values = result['momentum_std'].dropna()
    momentum_ratio_values = result['momentum_ratio'].dropna()
    
    print(f"\nMOMENTUM 统计:")
    print(f"  范围: {momentum_values.min():.2f} - {momentum_values.max():.2f}")
    print(f"  平均值: {momentum_values.mean():.2f}")
    print(f"  标准差: {momentum_values.std():.2f}")
    print(f"  有效值数量: {len(momentum_values)}")
    
    print(f"\nMOMENTUM 移动平均统计:")
    print(f"  范围: {momentum_ma_values.min():.2f} - {momentum_ma_values.max():.2f}")
    print(f"  平均值: {momentum_ma_values.mean():.2f}")
    print(f"  有效值数量: {len(momentum_ma_values)}")
    
    print(f"\nMOMENTUM 标准差统计:")
    print(f"  范围: {momentum_std_values.min():.2f} - {momentum_std_values.max():.2f}")
    print(f"  平均值: {momentum_std_values.mean():.2f}")
    print(f"  有效值数量: {len(momentum_std_values)}")
    
    print(f"\nMOMENTUM 比率统计:")
    print(f"  范围: {momentum_ratio_values.min():.2f}% - {momentum_ratio_values.max():.2f}%")
    print(f"  平均值: {momentum_ratio_values.mean():.2f}%")
    print(f"  有效值数量: {len(momentum_ratio_values)}")
    
    # 检查MOMENTUM计算的合理性
    if len(momentum_values) > 0:
        # MOMENTUM应该反映价格变化
        positive_momentum = len(momentum_values[momentum_values > 0])
        negative_momentum = len(momentum_values[momentum_values < 0])
        neutral_momentum = len(momentum_values[abs(momentum_values) < 1])
        
        print(f"\nMOMENTUM分布:")
        print(f"  正值: {positive_momentum} ({positive_momentum/len(momentum_values)*100:.1f}%)")
        print(f"  负值: {negative_momentum} ({negative_momentum/len(momentum_values)*100:.1f}%)")
        print(f"  中性(-1到1): {neutral_momentum} ({neutral_momentum/len(momentum_values)*100:.1f}%)")
    
    return result

def test_momentum_scoring():
    """测试MOMENTUM评分功能"""
    print("\n=== MOMENTUM评分功能测试 ===")
    
    # 创建测试数据
    data = create_test_data_Fix_Test_Momentum_Fix()
    
    # 创建MOMENTUM指标实例
    momentum = MOMENTUM(period=14)
    
    # 计算评分
    scores = momentum.calculate_raw_score(data)
    
    # 检查评分结果
    valid_scores = scores.dropna()
    print(f"有效评分数量: {len(valid_scores)}")
    
    if len(valid_scores) > 0:
        print(f"评分范围: {valid_scores.min():.2f} - {valid_scores.max():.2f}")
        print(f"平均评分: {valid_scores.mean():.2f}")
        print(f"评分标准差: {valid_scores.std():.2f}")
        print(f"评分唯一值数量: {len(valid_scores.unique())}")
        
        # 检查是否还是固定50分
        if len(valid_scores.unique()) == 1 and valid_scores.iloc[0] == 50.0:
            print("✗ 评分仍然是固定50分")
        else:
            print("✓ 评分已实现动态变化")
            
        # 检查评分分布
        score_ranges = {
            '0-20': len(valid_scores[(valid_scores >= 0) & (valid_scores < 20)]),
            '20-40': len(valid_scores[(valid_scores >= 20) & (valid_scores < 40)]),
            '40-60': len(valid_scores[(valid_scores >= 40) & (valid_scores < 60)]),
            '60-80': len(valid_scores[(valid_scores >= 60) & (valid_scores < 80)]),
            '80-100': len(valid_scores[(valid_scores >= 80) & (valid_scores <= 100)])
        }
        
        print(f"\n评分分布:")
        for range_name, count in score_ranges.items():
            percentage = count / len(valid_scores) * 100
            print(f"  {range_name}: {count} ({percentage:.1f}%)")
    
    return scores

def test_momentum_patterns():
    """测试MOMENTUM形态识别"""
    print("\n=== MOMENTUM形态识别测试 ===")
    
    # 创建测试数据
    data = create_test_data_Fix_Test_Momentum_Fix()
    
    # 创建MOMENTUM指标实例
    momentum = MOMENTUM(period=14)
    
    # 获取形态
    patterns = momentum.get_patterns(data)
    
    print(f"形态数量: {len(patterns.columns)}")
    print(f"形态列: {list(patterns.columns)}")
    
    # 统计各种形态的出现次数
    for col in patterns.columns:
        if col in patterns.columns:
            count = patterns[col].sum()
            percentage = count / len(patterns) * 100
            print(f"  {col}: {count} 次 ({percentage:.1f}%)")
    
    return patterns

def test_momentum_confidence():
    """测试MOMENTUM置信度计算"""
    print("\n=== MOMENTUM置信度测试 ===")
    
    # 创建测试数据
    data = create_test_data_Fix_Test_Momentum_Fix()
    
    # 创建MOMENTUM指标实例
    momentum = MOMENTUM(period=14)
    
    # 计算指标和评分
    result = momentum.calculate(data)
    scores = momentum.calculate_raw_score(data)
    patterns = momentum.get_patterns(data)
    
    # 计算置信度
    confidence = momentum.calculate_confidence(scores, patterns, {})
    
    print(f"置信度: {confidence:.3f}")
    
    # 检查置信度合理性
    if 0.2 <= confidence <= 0.9:
        print("✓ 置信度在合理范围内")
    else:
        print("✗ 置信度超出合理范围")
    
    return confidence

def demonstrate_momentum_behavior():
    """演示MOMENTUM在不同市场条件下的行为"""
    print("\n=== MOMENTUM行为演示 ===")
    
    # 创建测试数据
    data = create_test_data_Fix_Test_Momentum_Fix()
    
    # 创建MOMENTUM指标实例
    momentum = MOMENTUM(period=14)
    
    # 计算指标和评分
    result = momentum.calculate(data)
    scores = momentum.calculate_raw_score(data)
    
    # 显示不同阶段的MOMENTUM表现
    print("不同市场阶段的MOMENTUM表现:")
    
    stages = [
        ("加速上涨期", 5, 15),
        ("减速上涨期", 20, 30), 
        ("横盘震荡期", 35, 45),
        ("加速下跌期", 55, 65),
        ("减速下跌期", 75, 85),
        ("反弹期", 90, 100)
    ]
    
    for stage_name, start_idx, end_idx in stages:
        stage_data = result.iloc[start_idx:end_idx]
        stage_scores = scores.iloc[start_idx:end_idx]
        
        if len(stage_data) > 0:
            avg_momentum = stage_data['momentum'].mean()
            avg_score = stage_scores.mean()
            momentum_range = f"{stage_data['momentum'].min():.2f} - {stage_data['momentum'].max():.2f}"
            
            print(f"  {stage_name}:")
            print(f"    平均MOMENTUM: {avg_momentum:.2f}")
            print(f"    MOMENTUM范围: {momentum_range}")
            print(f"    平均评分: {avg_score:.2f}")

def main_testmomentumfix():
    """主函数"""
    print("MOMENTUM指标修复验证测试")
    print("=" * 50)
    
    # 测试计算功能
    result = test_momentum_calculation()
    
    # 测试评分功能
    scores = test_momentum_scoring()
    
    # 测试形态识别
    patterns = test_momentum_patterns()
    
    # 测试置信度
    confidence = test_momentum_confidence()
    
    # 演示MOMENTUM行为
    demonstrate_momentum_behavior()
    
    print("\n" + "=" * 50)
    print("测试完成")

if __name__ == "__main__":
    main_testmomentumfix() 