#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试CompositeIndicator实现
"""

import pandas as pd
from typing import List
import pytest

# 导入两个CompositeIndicator实现
from indicators.composite_indicator import CompositeIndicator as CompositeIndicator1
from indicators.adapter import CompositeIndicator as CompositeIndicator2

# 导入其他指标用于测试
from indicators.macd import MACD
from indicators.rsi import RSI
from indicators.factory import IndicatorFactory

@pytest.fixture(autouse=True)
def setup_indicators():
    IndicatorFactory.auto_register_all_indicators()

def test_composite_indicator_1():
    """测试indicators.composite_indicator.CompositeIndicator"""
    print("测试 composite_indicator.py 中的 CompositeIndicator...")
    
    # 创建一个CompositeIndicator实例
    ci = CompositeIndicator1(
        name="TestComposite",
        description="测试组合指标",
        indicators=[MACD(), RSI()]
    )
    
    assert ci is not None
    print(f"成功创建 {ci.name} 实例")

def test_composite_indicator_2():
    """测试indicators.adapter.CompositeIndicator"""
    print("测试 adapter.py 中的 CompositeIndicator...")
    
    # 定义组合函数
    def combine_func(results: List[pd.DataFrame], data: pd.DataFrame) -> pd.DataFrame:
        # 在实践中，这里可能需要更复杂的合并逻辑
        # 为简化测试，我们只连接列
        if not results:
            return data
        
        # 提取每个结果中独有的列
        merged_df = data.copy()
        for df in results:
            for col in df.columns:
                if col not in merged_df.columns:
                    merged_df[col] = df[col]
        return merged_df

    # 创建一个CompositeIndicator实例
    ci = CompositeIndicator2(
        name="TestAdapterComposite",
        indicators=["MACD", "RSI"],  # 使用指标名称
        combination_func=combine_func
    )
    
    assert ci is not None
    print(f"成功创建 {ci.name} 实例") 