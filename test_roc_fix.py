#!/usr/bin/env python3
"""
ROC指标修复验证测试

验证ROC指标修复后的功能：
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

from indicators.roc import ROC

def create_test_data():
    """创建测试数据"""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    # 创建有不同动量特征的价格数据
    np.random.seed(42)
    base_price = 100
    prices = []
    
    for i in range(100):
        if i < 20:
            # 前20天：快速上涨
            price = base_price + i * 1.5 + np.random.normal(0, 0.3)
        elif i < 40:
            # 20-40天：慢速上涨
            price = base_price + 30 + (i - 20) * 0.3 + np.random.normal(0, 0.5)
        elif i < 60:
            # 40-60天：横盘整理
            price = base_price + 36 + np.random.normal(0, 1)
        elif i < 80:
            # 60-80天：慢速下跌
            price = base_price + 36 - (i - 60) * 0.5 + np.random.normal(0, 0.5)
        else:
            # 80-100天：快速下跌
            price = base_price + 26 - (i - 80) * 1.2 + np.random.normal(0, 0.3)
        
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

def test_roc_calculation():
    """测试ROC计算功能"""
    print("=== ROC计算功能测试 ===")
    
    # 创建测试数据
    data = create_test_data()
    
    # 创建ROC指标实例
    roc = ROC(period=14)
    
    # 计算指标
    result = roc.calculate(data)
    
    # 检查计算结果
    print(f"数据行数: {len(result)}")
    print(f"包含的列: {list(result.columns)}")
    
    # 检查必要的列是否存在
    required_columns = ['roc', 'roc_ma']
    for col in required_columns:
        if col in result.columns:
            print(f"✓ {col} 列存在")
        else:
            print(f"✗ {col} 列不存在")
    
    # 检查数值范围
    roc_values = result['roc'].dropna()
    roc_ma_values = result['roc_ma'].dropna()
    
    print(f"\nROC 统计:")
    print(f"  范围: {roc_values.min():.2f} - {roc_values.max():.2f}")
    print(f"  平均值: {roc_values.mean():.2f}")
    print(f"  标准差: {roc_values.std():.2f}")
    print(f"  有效值数量: {len(roc_values)}")
    
    print(f"\nROC 移动平均统计:")
    print(f"  范围: {roc_ma_values.min():.2f} - {roc_ma_values.max():.2f}")
    print(f"  平均值: {roc_ma_values.mean():.2f}")
    print(f"  标准差: {roc_ma_values.std():.2f}")
    print(f"  有效值数量: {len(roc_ma_values)}")
    
    # 检查ROC计算的合理性
    if len(roc_values) > 0:
        # ROC应该反映价格变化
        positive_roc = len(roc_values[roc_values > 0])
        negative_roc = len(roc_values[roc_values < 0])
        neutral_roc = len(roc_values[abs(roc_values) < 1])
        
        print(f"\nROC分布:")
        print(f"  正值: {positive_roc} ({positive_roc/len(roc_values)*100:.1f}%)")
        print(f"  负值: {negative_roc} ({negative_roc/len(roc_values)*100:.1f}%)")
        print(f"  中性(-1到1): {neutral_roc} ({neutral_roc/len(roc_values)*100:.1f}%)")
    
    return result

def test_roc_scoring():
    """测试ROC评分功能"""
    print("\n=== ROC评分功能测试 ===")
    
    # 创建测试数据
    data = create_test_data()
    
    # 创建ROC指标实例
    roc = ROC(period=14)
    
    # 计算评分
    scores = roc.calculate_raw_score(data)
    
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

def test_roc_patterns():
    """测试ROC形态识别"""
    print("\n=== ROC形态识别测试 ===")
    
    # 创建测试数据
    data = create_test_data()
    
    # 创建ROC指标实例
    roc = ROC(period=14)
    
    # 获取形态
    patterns = roc.get_patterns(data)
    
    print(f"形态数量: {len(patterns.columns)}")
    print(f"形态列: {list(patterns.columns)}")
    
    # 统计各种形态的出现次数
    for col in patterns.columns:
        if col in patterns.columns:
            count = patterns[col].sum()
            percentage = count / len(patterns) * 100
            print(f"  {col}: {count} 次 ({percentage:.1f}%)")
    
    return patterns

def test_roc_confidence():
    """测试ROC置信度计算"""
    print("\n=== ROC置信度测试 ===")
    
    # 创建测试数据
    data = create_test_data()
    
    # 创建ROC指标实例
    roc = ROC(period=14)
    
    # 计算指标和评分
    result = roc.calculate(data)
    scores = roc.calculate_raw_score(data)
    patterns = roc.get_patterns(data)
    
    # 计算置信度
    confidence = roc.calculate_confidence(scores, patterns, {})
    
    print(f"置信度: {confidence:.3f}")
    
    # 检查置信度合理性
    if 0.2 <= confidence <= 0.9:
        print("✓ 置信度在合理范围内")
    else:
        print("✗ 置信度超出合理范围")
    
    return confidence

def demonstrate_roc_behavior():
    """演示ROC在不同市场条件下的行为"""
    print("\n=== ROC行为演示 ===")
    
    # 创建测试数据
    data = create_test_data()
    
    # 创建ROC指标实例
    roc = ROC(period=14)
    
    # 计算指标和评分
    result = roc.calculate(data)
    scores = roc.calculate_raw_score(data)
    
    # 显示不同阶段的ROC表现
    print("不同市场阶段的ROC表现:")
    
    stages = [
        ("快速上涨期", 10, 20),
        ("慢速上涨期", 25, 35), 
        ("横盘整理期", 45, 55),
        ("慢速下跌期", 65, 75),
        ("快速下跌期", 85, 95)
    ]
    
    for stage_name, start_idx, end_idx in stages:
        stage_data = result.iloc[start_idx:end_idx]
        stage_scores = scores.iloc[start_idx:end_idx]
        
        if len(stage_data) > 0:
            avg_roc = stage_data['roc'].mean()
            avg_score = stage_scores.mean()
            roc_range = f"{stage_data['roc'].min():.2f} - {stage_data['roc'].max():.2f}"
            
            print(f"  {stage_name}:")
            print(f"    平均ROC: {avg_roc:.2f}")
            print(f"    ROC范围: {roc_range}")
            print(f"    平均评分: {avg_score:.2f}")

def main():
    """主函数"""
    print("ROC指标修复验证测试")
    print("=" * 50)
    
    # 测试计算功能
    result = test_roc_calculation()
    
    # 测试评分功能
    scores = test_roc_scoring()
    
    # 测试形态识别
    patterns = test_roc_patterns()
    
    # 测试置信度
    confidence = test_roc_confidence()
    
    # 演示ROC行为
    demonstrate_roc_behavior()
    
    print("\n" + "=" * 50)
    print("测试完成")

if __name__ == "__main__":
    main() 