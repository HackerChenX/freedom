#!/usr/bin/env python3
"""
测试CCI、STOCHRSI、ATR三个指标的修复效果
"""
import sys
import os
import numpy as np
import pandas as pd

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

from indicators.cci import CCI
from indicators.stochrsi import STOCHRSI
from indicators.atr import ATR

def create_test_data_Fix_Test_Three_Indicators_Fix(length=100):
    """创建测试数据"""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=length, freq='D')
    
    # 创建有趋势的价格数据
    base_price = 100
    price_changes = np.random.normal(0, 2, length)
    trend = np.linspace(0, 20, length)  # 上升趋势
    
    closes = [base_price]
    for i in range(1, length):
        new_price = closes[-1] + price_changes[i] + trend[i] - trend[i-1]
        closes.append(max(new_price, 10))  # 确保价格不为负
    
    # 生成OHLC数据
    data = []
    for i, close in enumerate(closes):
        high = close + np.random.uniform(0, 3)
        low = close - np.random.uniform(0, 3)
        open_price = low + np.random.uniform(0, high - low)
        volume = np.random.uniform(1000, 10000)
        
        data.append({
            'date': dates[i],
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    return pd.DataFrame(data).set_index('date')

def test_indicator_calculation(indicator_class, indicator_name, test_data):
    """测试指标计算"""
    print(f"\n=== 测试 {indicator_name} 指标 ===")
    
    try:
        # 创建指标实例
        indicator = indicator_class()
        
        # 计算指标
        result = indicator.calculate(test_data)
        
        # 检查结果
        print(f"结果DataFrame形状: {result.shape}")
        print(f"结果列名: {list(result.columns)}")
        
        # 检查指标列
        indicator_cols = [col for col in result.columns if indicator_name in col or 
                         (indicator_name == 'STOCHRSI' and ('STOCHRSI_K' in col or 'STOCHRSI_D' in col))]
        
        if indicator_cols:
            print(f"指标列: {indicator_cols}")
            for col in indicator_cols:
                values = result[col].dropna()
                if len(values) > 0:
                    print(f"  {col}: 有效值数量={len(values)}, 平均值={values.mean():.4f}, 标准差={values.std():.4f}")
                    print(f"    范围: {values.min():.4f} - {values.max():.4f}")
                else:
                    print(f"  {col}: 无有效值")
        else:
            print(f"❌ 未找到 {indicator_name} 指标列")
            return False
        
        # 测试评分功能
        print(f"\n--- 测试 {indicator_name} 评分功能 ---")
        
        # 检查是否有结果
        has_result = indicator.has_result()
        print(f"has_result(): {has_result}")
        
        if has_result:
            # 计算评分
            score = indicator.calculate_raw_score(test_data)
            print(f"评分结果类型: {type(score)}")
            print(f"评分长度: {len(score)}")
            
            # 分析评分
            if len(score) > 0:
                unique_scores = score.nunique()
                print(f"唯一评分数量: {unique_scores}")
                print(f"评分统计: 平均={score.mean():.4f}, 标准差={score.std():.4f}")
                print(f"评分范围: {score.min():.4f} - {score.max():.4f}")
                
                if unique_scores == 1 and score.iloc[0] == 50.0:
                    print(f"❌ {indicator_name} 评分仍然固定为50分")
                    return False
                else:
                    print(f"✅ {indicator_name} 评分正常，有动态变化")
                    return True
            else:
                print(f"❌ {indicator_name} 评分为空")
                return False
        else:
            print(f"❌ {indicator_name} 没有计算结果")
            return False
            
    except Exception as e:
        print(f"❌ 测试 {indicator_name} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def mainTestthreeindicatorsfix():
    """主函数"""
    print("=== CCI、STOCHRSI、ATR 指标修复效果测试 ===")
    
    # 创建测试数据
    test_data = create_test_data_Fix_Test_Three_Indicators_Fix(100)
    print(f"测试数据形状: {test_data.shape}")
    print(f"测试数据列: {list(test_data.columns)}")
    
    # 测试指标列表
    indicators_to_test = [
        (CCI, 'CCI'),
        (STOCHRSI, 'STOCHRSI'),
        (ATR, 'ATR')
    ]
    
    # 测试结果
    results = {}
    
    for indicator_class, indicator_name in indicators_to_test:
        success = test_indicator_calculation(indicator_class, indicator_name, test_data)
        results[indicator_name] = success
    
    # 总结结果
    print("\n" + "="*50)
    print("测试结果总结:")
    print("="*50)
    
    for indicator_name, success in results.items():
        status = "✅ 修复成功" if success else "❌ 仍需修复"
        print(f"{indicator_name:12}: {status}")
    
    successful_count = sum(results.values())
    total_count = len(results)
    
    print(f"\n修复成功: {successful_count}/{total_count} 个指标")
    
    if successful_count == total_count:
        print("🎉 所有指标都已成功修复！")
    else:
        print("⚠️  仍有指标需要进一步修复")

if __name__ == "__main__":
    mainTestthreeindicatorsfix() 