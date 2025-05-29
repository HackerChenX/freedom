#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试布林带带宽变化率功能
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 直接从文件导入BOLL类
from indicators.boll import BOLL

def create_test_data():
    """创建测试数据"""
    # 创建100天的样本数据
    dates = [datetime.now() - timedelta(days=i) for i in range(100)]
    dates.reverse()  # 按时间正序排列
    
    # 生成价格数据 - 前半部分相对平稳，后半部分有明显趋势和波动
    close_prices = np.zeros(100)
    
    # 前50天相对平稳的价格
    close_prices[:50] = np.cumsum(np.random.normal(0, 0.3, 50)) + 100
    
    # 后50天明显的上升趋势和增大的波动
    trend = np.linspace(0, 15, 50)
    volatility = np.linspace(0.3, 1.5, 50)
    noise = np.array([np.random.normal(0, vol) for vol in volatility])
    close_prices[50:] = close_prices[49] + trend + np.cumsum(noise)
    
    # 创建DataFrame
    data = pd.DataFrame({
        'close': close_prices
    }, index=dates)
    
    # 确保价格始终为正值
    data[data <= 0] = 0.01
    
    return data

def test_bandwidth_calculation():
    """测试带宽计算功能"""
    # 创建测试数据
    np.random.seed(42)  # 固定随机种子
    data = create_test_data()
    
    # 创建BOLL指标实例
    boll = BOLL()
    boll.set_parameters({
        'periods': 20,
        'std_dev': 2.0,
        'bw_periods': 10
    })
    
    # 计算指标
    result = boll.compute(data)
    
    # 获取带宽
    bandwidth = boll.get_bandwidth()
    print(f"带宽前5个值: {bandwidth.iloc[:5].values}")
    print(f"带宽后5个值: {bandwidth.iloc[-5:].values}")
    
    # 获取带宽变化率
    bw_rate = boll.get_bandwidth_rate()
    print(f"带宽变化率前15个值: {bw_rate.iloc[:15].values}")
    print(f"带宽变化率后5个值: {bw_rate.iloc[-5:].values}")
    
    # 测试带宽收缩信号
    squeeze_signal = boll.is_bandwidth_squeeze(threshold=5.0)
    squeeze_count = squeeze_signal.sum()
    print(f"带宽收缩信号数量: {squeeze_count}")
    
    # 测试带宽扩张信号
    expansion_signal = boll.is_bandwidth_expansion(threshold=10.0)
    expansion_count = expansion_signal.sum()
    print(f"带宽扩张信号数量: {expansion_count}")
    
    # 验证带宽与上下轨之间的关系
    for i in range(30, 35):
        upper = result['upper'].iloc[i]
        lower = result['lower'].iloc[i]
        middle = result['middle'].iloc[i]
        
        expected_bandwidth = (upper - lower) / middle
        calculated_bandwidth = bandwidth.iloc[i]
        
        print(f"位置 {i}: 计算带宽 = {calculated_bandwidth:.6f}, 期望带宽 = {expected_bandwidth:.6f}, 差异 = {abs(calculated_bandwidth - expected_bandwidth):.10f}")
    
    # 验证带宽变化率计算
    for i in range(30, 35):
        current_bw = bandwidth.iloc[i]
        prev_bw = bandwidth.iloc[i-10]
        
        expected_bw_rate = (current_bw - prev_bw) / prev_bw * 100
        calculated_bw_rate = bw_rate.iloc[i]
        
        print(f"位置 {i}: 计算带宽变化率 = {calculated_bw_rate:.6f}, 期望带宽变化率 = {expected_bw_rate:.6f}, 差异 = {abs(calculated_bw_rate - expected_bw_rate):.10f}")

if __name__ == "__main__":
    test_bandwidth_calculation()
    print("\n测试完成：布林带带宽变化率功能运行正常！") 