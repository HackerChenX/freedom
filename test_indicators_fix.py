#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试指标修复效果的脚本
"""

import pandas as pd
import numpy as np
from indicators.chaikin import Chaikin
from indicators.cci import CCI
from indicators.bias import BIAS
from indicators.macd import MACD
from indicators.adx import ADX
from indicators.trix import TRIX
from indicators.rsi import RSI
from indicators.boll import BOLL
from indicators.atr import ATR
from indicators.pvt import PVT
from indicators.obv import OBV
from indicators.cmo import CMO
from indicators.mfi import MFI
from indicators.wr import WR

def create_test_data():
    """创建测试数据"""
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    test_data = pd.DataFrame({
        'code': '000001.SZ',
        'name': '平安银行',
        'date': dates,
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 105,
        'low': np.random.randn(100).cumsum() + 95,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    })
    test_data.set_index('date', inplace=True)
    return test_data

def test_indicator_patterns():
    """测试指标形态是否只包含布尔类型列"""
    print("=== 测试指标形态数据污染修复 ===\n")

    # 创建测试数据
    test_data = create_test_data()

    # 测试指标
    indicators = [
        ('Chaikin', Chaikin()),
        ('CCI', CCI()),
        ('BIAS', BIAS()),
        ('MACD', MACD()),
        ('ADX', ADX()),
        ('TRIX', TRIX()),
        ('RSI', RSI()),
        ('BOLL', BOLL()),
        ('ATR', ATR()),
        ('PVT', PVT()),
        ('OBV', OBV()),
        ('CMO', CMO()),
        ('MFI', MFI()),
        ('WR', WR())
    ]

    all_passed = True

    for name, indicator in indicators:
        print(f"测试 {name} 指标...")
        try:
            # 获取指标形态
            patterns = indicator.get_patterns(test_data)

            print(f"  ✅ {name} 返回了形态 DataFrame，shape: {patterns.shape}")
            print(f"  形态列: {list(patterns.columns)}")

            # 检查是否有非布尔类型的列
            non_bool_cols = []
            problematic_cols = []

            for col in patterns.columns:
                if patterns[col].dtype != bool:
                    non_bool_cols.append(f"{col}({patterns[col].dtype})")

                # 检查是否包含原始数据列名
                if col.lower() in ['code', 'name', 'date', 'open', 'high', 'low', 'close', 'volume']:
                    problematic_cols.append(col)

            if non_bool_cols:
                print(f"  ❌ {name} 有非布尔类型的列: {non_bool_cols}")
                all_passed = False
            else:
                print(f"  ✅ {name} 所有列都是布尔类型")

            if problematic_cols:
                print(f"  ❌ {name} 包含原始数据列: {problematic_cols}")
                all_passed = False
            else:
                print(f"  ✅ {name} 没有包含原始数据列")

        except Exception as e:
            print(f"  ❌ {name} 测试失败: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

        print()

    return all_passed

def main():
    """主函数"""
    print("开始测试指标脚本数据污染修复效果...\n")

    # 测试指标形态
    patterns_test_passed = test_indicator_patterns()

    print("\n=== 测试总结 ===")
    if patterns_test_passed:
        print("✅ 指标形态测试通过！数据污染问题已修复。")
    else:
        print("❌ 指标形态测试失败，仍存在数据污染问题。")

if __name__ == "__main__":
    main()
