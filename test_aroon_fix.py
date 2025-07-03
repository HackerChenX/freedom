#!/usr/bin/env python3
"""
AROON指标修复验证测试

验证AROON指标修复后的功能：
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

from indicators.aroon import AROON
from db.clickhouse_db import get_clickhouse_db

def create_test_data():
    """创建测试数据"""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    # 创建有趋势的价格数据
    np.random.seed(42)
    base_price = 100
    prices = []
    
    for i in range(100):
        if i < 30:
            # 前30天：上升趋势
            price = base_price + i * 0.5 + np.random.normal(0, 0.5)
        elif i < 60:
            # 中间30天：横盘整理
            price = base_price + 15 + np.random.normal(0, 1)
        else:
            # 后40天：下降趋势
            price = base_price + 15 - (i - 60) * 0.3 + np.random.normal(0, 0.5)
        
        prices.append(price)
    
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

def test_aroon_calculation():
    """测试AROON计算功能"""
    print("=== AROON计算功能测试 ===")
    
    # 创建测试数据
    data = create_test_data()
    
    # 创建AROON指标实例
    aroon = AROON(period=14)
    
    # 计算指标
    result = aroon.calculate(data)
    
    # 检查计算结果
    print(f"数据行数: {len(result)}")
    print(f"包含的列: {list(result.columns)}")
    
    # 检查必要的列是否存在
    required_columns = ['aroon_up', 'aroon_down', 'aroon_oscillator']
    for col in required_columns:
        if col in result.columns:
            print(f"✓ {col} 列存在")
        else:
            print(f"✗ {col} 列不存在")
    
    # 检查数值范围
    aroon_up = result['aroon_up'].dropna()
    aroon_down = result['aroon_down'].dropna()
    aroon_osc = result['aroon_oscillator'].dropna()
    
    print(f"\nAROON UP 统计:")
    print(f"  范围: {aroon_up.min():.2f} - {aroon_up.max():.2f}")
    print(f"  平均值: {aroon_up.mean():.2f}")
    print(f"  标准差: {aroon_up.std():.2f}")
    print(f"  有效值数量: {len(aroon_up)}")
    
    print(f"\nAROON DOWN 统计:")
    print(f"  范围: {aroon_down.min():.2f} - {aroon_down.max():.2f}")
    print(f"  平均值: {aroon_down.mean():.2f}")
    print(f"  标准差: {aroon_down.std():.2f}")
    print(f"  有效值数量: {len(aroon_down)}")
    
    print(f"\nAROON 震荡器统计:")
    print(f"  范围: {aroon_osc.min():.2f} - {aroon_osc.max():.2f}")
    print(f"  平均值: {aroon_osc.mean():.2f}")
    print(f"  标准差: {aroon_osc.std():.2f}")
    print(f"  有效值数量: {len(aroon_osc)}")
    
    # 检查AROON UP和DOWN是否在0-100范围内
    if len(aroon_up) > 0:
        up_in_range = all(0 <= val <= 100 for val in aroon_up)
        print(f"\nAROON UP 值在0-100范围内: {'✓' if up_in_range else '✗'}")
    
    if len(aroon_down) > 0:
        down_in_range = all(0 <= val <= 100 for val in aroon_down)
        print(f"AROON DOWN 值在0-100范围内: {'✓' if down_in_range else '✗'}")
    
    return result

def test_aroon_scoring():
    """测试AROON评分功能"""
    print("\n=== AROON评分功能测试 ===")
    
    # 创建测试数据
    data = create_test_data()
    
    # 创建AROON指标实例
    aroon = AROON(period=14)
    
    # 计算评分
    scores = aroon.calculate_raw_score(data)
    
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

def test_aroon_patterns():
    """测试AROON形态识别"""
    print("\n=== AROON形态识别测试 ===")
    
    # 创建测试数据
    data = create_test_data()
    
    # 创建AROON指标实例
    aroon = AROON(period=14)
    
    # 获取形态
    patterns = aroon.get_patterns(data)
    
    print(f"形态数量: {len(patterns.columns)}")
    print(f"形态列: {list(patterns.columns)}")
    
    # 统计各种形态的出现次数
    for col in patterns.columns:
        if col in patterns.columns:
            count = patterns[col].sum()
            percentage = count / len(patterns) * 100
            print(f"  {col}: {count} 次 ({percentage:.1f}%)")
    
    return patterns

def test_with_real_data():
    """使用真实数据测试"""
    print("\n=== 真实数据测试 ===")
    
    try:
        # 获取数据库连接
        db = get_clickhouse_db()
        
        # 查询真实数据
        query = """
        SELECT date, open, high, low, close, volume
        FROM stock_data 
        WHERE code = '000001' 
        ORDER BY date DESC 
        LIMIT 100
        """
        
        real_data = db.query_dataframe(query)
        
        if len(real_data) > 0:
            print(f"获取到真实数据: {len(real_data)} 条")
            
            # 创建AROON指标实例
            aroon = AROON(period=14)
            
            # 计算指标
            result = aroon.calculate(real_data)
            
            # 计算评分
            scores = aroon.calculate_raw_score(real_data)
            
            # 显示结果
            aroon_up = result['aroon_up'].dropna()
            aroon_down = result['aroon_down'].dropna()
            valid_scores = scores.dropna()
            
            print(f"\n真实数据AROON结果:")
            print(f"AROON UP 范围: {aroon_up.min():.2f} - {aroon_up.max():.2f}")
            print(f"AROON DOWN 范围: {aroon_down.min():.2f} - {aroon_down.max():.2f}")
            print(f"评分范围: {valid_scores.min():.2f} - {valid_scores.max():.2f}")
            print(f"评分平均值: {valid_scores.mean():.2f}")
            print(f"评分标准差: {valid_scores.std():.2f}")
            print(f"评分唯一值: {len(valid_scores.unique())}")
            
            # 显示最近几天的数据
            print(f"\n最近5天的AROON数据:")
            recent_data = result[['date', 'aroon_up', 'aroon_down', 'aroon_oscillator']].tail(5)
            recent_scores = scores.tail(5)
            
            for i, (idx, row) in enumerate(recent_data.iterrows()):
                score = recent_scores.iloc[i] if i < len(recent_scores) else 0
                print(f"  {row['date']}: UP={row['aroon_up']:.2f}, DOWN={row['aroon_down']:.2f}, OSC={row['aroon_oscillator']:.2f}, Score={score:.2f}")
            
        else:
            print("无法获取真实数据")
            
    except Exception as e:
        print(f"真实数据测试失败: {e}")

def main():
    """主函数"""
    print("AROON指标修复验证测试")
    print("=" * 50)
    
    # 测试计算功能
    result = test_aroon_calculation()
    
    # 测试评分功能
    scores = test_aroon_scoring()
    
    # 测试形态识别
    patterns = test_aroon_patterns()
    
    # 测试真实数据
    test_with_real_data()
    
    print("\n" + "=" * 50)
    print("测试完成")

if __name__ == "__main__":
    main() 