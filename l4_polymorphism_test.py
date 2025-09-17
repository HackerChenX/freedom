#!/usr/bin/env python3
"""多态性测试脚本"""

import pandas as pd
from indicators.base_indicator import BaseIndicator
from indicators.complete_indicator_registry import get_all_indicators

def test_polymorphism():
    """测试多态性调用"""
    indicators = get_all_indicators()
    test_data = pd.DataFrame({
        'open': [100, 101, 102, 103, 104],
        'high': [105, 106, 107, 108, 109],
        'low': [95, 96, 97, 98, 99],
        'close': [102, 103, 104, 105, 106],
        'volume': [1000, 1100, 1200, 1300, 1400]
    })

    success_count = 0
    total_count = len(indicators)

    for name, indicator_class in indicators.items():
        try:
            if issubclass(indicator_class, BaseIndicator):
                indicator = indicator_class()
                result = indicator.calculate(test_data)
                signal = indicator.get_signal(test_data)
                success_count += 1
        except Exception:
            pass

    return (success_count / total_count * 100) if total_count > 0 else 0

if __name__ == "__main__":
    success_rate = test_polymorphism()
    print(f"多态性测试通过率: {success_rate:.1f}%")
