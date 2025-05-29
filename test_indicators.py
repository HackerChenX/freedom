#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('.')

from indicators.factory import IndicatorFactory

def test_indicators():
    """测试指标工厂自动注册的功能"""
    print(f"注册的指标总数: {len(IndicatorFactory.get_supported_indicators())}")
    
    # 测试一些之前未注册的指标
    indicators_to_test = [
        'ELLIOTTWAVE',
        'FIBONACCITOOLS', 
        'CHIPDISTRIBUTION', 
        'VOLUMERATIO',
        'ZXMWASHPLATE',
        'MULTI_PERIOD_RESONANCE'
    ]
    
    print("\n测试创建指标实例:")
    for ind in indicators_to_test:
        indicator = IndicatorFactory.create(ind)
        status = '成功' if indicator else '失败'
        print(f"{ind}: {status}")
    
    # 打印指标类型
    print("\n已注册的指标类型列表:")
    all_indicators = sorted(IndicatorFactory.get_supported_indicators())
    for i in range(0, len(all_indicators), 5):
        row = all_indicators[i:i+5]
        print(", ".join(row))

if __name__ == "__main__":
    test_indicators() 