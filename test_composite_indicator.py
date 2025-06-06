#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试CompositeIndicator实现
"""

import pandas as pd
import numpy as np
from typing import List

# 导入两个CompositeIndicator实现
from indicators.composite_indicator import CompositeIndicator as CompositeIndicator1
from indicators.adapter import CompositeIndicator as CompositeIndicator2

# 导入其他指标用于测试
from indicators.macd import MACD
from indicators.rsi import RSI

def test_composite_indicator_1():
    """测试indicators.composite_indicator.CompositeIndicator"""
    print("测试 composite_indicator.py 中的 CompositeIndicator...")
    
    # 创建一个CompositeIndicator实例
    ci = CompositeIndicator1(
        name="TestComposite",
        description="测试组合指标",
        indicators=[MACD(), RSI()]
    )
    
    print(f"成功创建 {ci.name} 实例")
    return True

def test_composite_indicator_2():
    """测试indicators.adapter.CompositeIndicator"""
    print("测试 adapter.py 中的 CompositeIndicator...")
    
    # 定义组合函数
    def combine_func(results: List[pd.DataFrame], data: pd.DataFrame) -> pd.DataFrame:
        return pd.concat(results, axis=1)
    
    # 创建一个CompositeIndicator实例
    ci = CompositeIndicator2(
        name="TestAdapterComposite",
        indicators=[MACD(), RSI()],
        combination_func=combine_func
    )
    
    print(f"成功创建 {ci.name} 实例")
    return True

def main():
    """主函数"""
    
    success1 = test_composite_indicator_1()
    success2 = test_composite_indicator_2()
    
    if success1 and success2:
        print("所有测试通过!")
    else:
        print("测试失败!")

if __name__ == "__main__":
    main() 