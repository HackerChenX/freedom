#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试WMA、BIAS、MTM指标评分功能

验证新实现的三个指标的评分和形态识别功能
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from indicators.complete_indicator_registry import complete_registry
from indicators.complete_indicator_registry import complete_registry
from indicators.complete_indicator_registry import complete_registry
from utils.logger import get_logger

logger = get_logger(__name__)


def create_test_data_Scoring(length=100):
    """
    创建测试数据
    
    Args:
        length: 数据长度
        
    Returns:
        pd.DataFrame: 测试数据
    """
    # 创建日期索引
    dates = pd.date_range(start='2023-01-01', periods=length, freq='D')
    
    # 生成模拟价格数据
    np.random.seed(42)
    
    # 基础价格趋势
    base_price = 100
    trend = np.linspace(0, 20, length)  # 上升趋势
    noise = np.random.normal(0, 2, length)  # 随机噪音
    
    close_prices = base_price + trend + noise
    
    # 生成OHLC数据
    high_prices = close_prices + np.random.uniform(0.5, 2, length)
    low_prices = close_prices - np.random.uniform(0.5, 2, length)
    open_prices = close_prices + np.random.uniform(-1, 1, length)
    
    # 生成成交量数据
    volumes = np.random.uniform(1000000, 5000000, length)
    
    data = pd.DataFrame({
        'date': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    })
    
    data.set_index('date', inplace=True)
    return data


def test_wma_scoring():
    """测试WMA指标评分功能"""
    print("\n" + "="*50)
    print("测试WMA（加权移动平均线）指标评分功能")
    print("="*50)
    
    # 创建测试数据
    data = create_test_data_Scoring(100)
    
    # 初始化WMA指标
    wma = WMA(periods=[5, 10, 20])
    
    try:
        # 计算指标
        result_data = wma.calculate(data)
        print(f"✅ WMA指标计算成功")
        
        # 计算评分
        scores = wma.calculate_score(data)
        print(f"✅ WMA评分计算成功")
        
        # 识别形态
        patterns = wma.identify_patterns(data)
        print(f"✅ WMA形态识别成功")
        
        # 输出结果统计
        print(f"\n📊 WMA评分统计:")
        if isinstance(scores, dict):
            final_scores = scores.get('final_score', pd.Series())
            confidence = scores.get('confidence', 50.0)
            
            if isinstance(final_scores, pd.Series) and len(final_scores) > 0:
                print(f"   平均分: {final_scores.mean():.2f}")
                print(f"   最高分: {final_scores.max():.2f}")
                print(f"   最低分: {final_scores.min():.2f}")
                print(f"   标准差: {final_scores.std():.2f}")
            else:
                print(f"   评分数据为空或格式错误")
            
            if isinstance(confidence, (int, float)):
                print(f"   置信度: {confidence:.2f}")
            else:
                print(f"   置信度: 未知")
        else:
            print(f"   评分数据格式错误: {type(scores)}")
        
        print(f"\n🔍 识别的形态:")
        for pattern in patterns[:5]:  # 显示前5个形态
            print(f"   - {pattern}")
        
        print(f"\n📈 最新评分详情:")
        if isinstance(scores, dict):
            final_scores = scores.get('final_score', pd.Series())
            raw_scores = scores.get('raw_score', pd.Series())
            market_env = scores.get('market_environment', '未知')
            confidence = scores.get('confidence', 50.0)
            
            if isinstance(final_scores, pd.Series) and len(final_scores) > 0:
                print(f"   原始评分: {raw_scores.iloc[-1] if isinstance(raw_scores, pd.Series) and len(raw_scores) > 0 else '未知':.2f}")
                print(f"   最终评分: {final_scores.iloc[-1]:.2f}")
                print(f"   市场环境: {market_env}")
                print(f"   置信度: {confidence:.2f}")
            else:
                print(f"   评分数据为空")
        else:
            print(f"   评分数据格式错误")
        
        return True
        
    except Exception as e:
        print(f"❌ WMA测试失败: {str(e)}")
        return False


def test_bias_scoring():
    """测试BIAS指标评分功能"""
    print("\n" + "="*50)
    print("测试BIAS（乖离率）指标评分功能")
    print("="*50)
    
    # 创建测试数据
    data = create_test_data_Scoring(100)
    
    # 初始化BIAS指标
    bias = BIAS(periods=[6, 12, 24])
    
    try:
        # 计算指标
        result_data = bias.calculate(data)
        print(f"✅ BIAS指标计算成功")
        
        # 计算评分
        scores = bias.calculate_score(data)
        print(f"✅ BIAS评分计算成功")
        
        # 识别形态
        patterns = bias.identify_patterns(data)
        print(f"✅ BIAS形态识别成功")
        
        # 输出结果统计
        print(f"\n📊 BIAS评分统计:")
        if isinstance(scores, dict):
            final_scores = scores.get('final_score', pd.Series())
            confidence = scores.get('confidence', 50.0)
            
            if isinstance(final_scores, pd.Series) and len(final_scores) > 0:
                print(f"   平均分: {final_scores.mean():.2f}")
                print(f"   最高分: {final_scores.max():.2f}")
                print(f"   最低分: {final_scores.min():.2f}")
                print(f"   标准差: {final_scores.std():.2f}")
            else:
                print(f"   评分数据为空或格式错误")
            
            if isinstance(confidence, (int, float)):
                print(f"   置信度: {confidence:.2f}")
            else:
                print(f"   置信度: 未知")
        else:
            print(f"   评分数据格式错误: {type(scores)}")
        
        print(f"\n🔍 识别的形态:")
        for pattern in patterns[:5]:  # 显示前5个形态
            print(f"   - {pattern}")
        
        print(f"\n📈 最新评分详情:")
        if isinstance(scores, dict):
            final_scores = scores.get('final_score', pd.Series())
            raw_scores = scores.get('raw_score', pd.Series())
            market_env = scores.get('market_environment', '未知')
            confidence = scores.get('confidence', 50.0)
            
            if isinstance(final_scores, pd.Series) and len(final_scores) > 0:
                print(f"   原始评分: {raw_scores.iloc[-1] if isinstance(raw_scores, pd.Series) and len(raw_scores) > 0 else '未知':.2f}")
                print(f"   最终评分: {final_scores.iloc[-1]:.2f}")
                print(f"   市场环境: {market_env}")
                print(f"   置信度: {confidence:.2f}")
            else:
                print(f"   评分数据为空")
        else:
            print(f"   评分数据格式错误")
        
        return True
        
    except Exception as e:
        print(f"❌ BIAS测试失败: {str(e)}")
        return False


def test_mtm_scoring():
    """测试MTM指标评分功能"""
    print("\n" + "="*50)
    print("测试MTM（动量指标）指标评分功能")
    print("="*50)
    
    # 创建测试数据
    data = create_test_data_Scoring(100)
    
    # 初始化MTM指标
    mtm = MTM(period=14, signal_period=6)
    
    try:
        # 计算指标
        result_data = mtm.calculate(data)
        print(f"✅ MTM指标计算成功")
        
        # 计算评分
        scores = mtm.calculate_score(data)
        print(f"✅ MTM评分计算成功")
        
        # 识别形态
        patterns = mtm.identify_patterns(data)
        print(f"✅ MTM形态识别成功")
        
        # 输出结果统计
        print(f"\n�� MTM评分统计:")
        if isinstance(scores, dict):
            final_scores = scores.get('final_score', pd.Series())
            confidence = scores.get('confidence', 50.0)
            
            if isinstance(final_scores, pd.Series) and len(final_scores) > 0:
                print(f"   平均分: {final_scores.mean():.2f}")
                print(f"   最高分: {final_scores.max():.2f}")
                print(f"   最低分: {final_scores.min():.2f}")
                print(f"   标准差: {final_scores.std():.2f}")
            else:
                print(f"   评分数据为空或格式错误")
            
            if isinstance(confidence, (int, float)):
                print(f"   置信度: {confidence:.2f}")
            else:
                print(f"   置信度: 未知")
        else:
            print(f"   评分数据格式错误: {type(scores)}")
        
        print(f"\n🔍 识别的形态:")
        for pattern in patterns[:5]:  # 显示前5个形态
            print(f"   - {pattern}")
        
        print(f"\n📈 最新评分详情:")
        if isinstance(scores, dict):
            final_scores = scores.get('final_score', pd.Series())
            raw_scores = scores.get('raw_score', pd.Series())
            market_env = scores.get('market_environment', '未知')
            confidence = scores.get('confidence', 50.0)
            
            if isinstance(final_scores, pd.Series) and len(final_scores) > 0:
                print(f"   原始评分: {raw_scores.iloc[-1] if isinstance(raw_scores, pd.Series) and len(raw_scores) > 0 else '未知':.2f}")
                print(f"   最终评分: {final_scores.iloc[-1]:.2f}")
                print(f"   市场环境: {market_env}")
                print(f"   置信度: {confidence:.2f}")
            else:
                print(f"   评分数据为空")
        else:
            print(f"   评分数据格式错误")
        
        return True
        
    except Exception as e:
        print(f"❌ MTM测试失败: {str(e)}")
        return False


def main_testwmabiasmtmscoring():
    """主函数"""
    print("🚀 开始测试WMA、BIAS、MTM指标评分功能")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 测试结果统计
    test_results = []
    
    # 测试WMA指标
    wma_result = test_wma_scoring()
    test_results.append(('WMA', wma_result))
    
    # 测试BIAS指标
    bias_result = test_bias_scoring()
    test_results.append(('BIAS', bias_result))
    
    # 测试MTM指标
    mtm_result = test_mtm_scoring()
    test_results.append(('MTM', mtm_result))
    
    # 输出测试总结
    print("\n" + "="*60)
    print("📋 测试结果总结")
    print("="*60)
    
    passed_count = 0
    total_count = len(test_results)
    
    for indicator_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{indicator_name:15} {status}")
        if result:
            passed_count += 1
    
    print(f"\n📊 总体统计:")
    print(f"   测试总数: {total_count}")
    print(f"   通过数量: {passed_count}")
    print(f"   失败数量: {total_count - passed_count}")
    print(f"   通过率: {passed_count/total_count*100:.1f}%")
    
    if passed_count == total_count:
        print(f"\n🎉 所有指标评分功能测试通过！")
    else:
        print(f"\n⚠️  有 {total_count - passed_count} 个指标测试失败，请检查实现。")
    
    return passed_count == total_count


if __name__ == "__main__":
    success = main_testwmabiasmtmscoring()
    sys.exit(0 if success else 1) 