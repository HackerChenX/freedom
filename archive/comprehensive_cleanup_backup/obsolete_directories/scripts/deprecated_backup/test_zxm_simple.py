#!/usr/bin/env python3
"""
简单测试ZXM_PATTERNS指标实例化
"""

import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from indicators.pattern.zxm_patterns import ZxmpatternIndicator
    from indicators.base_indicator import BaseIndicator
    
    print("开始测试ZXM_PATTERNS指标实例化...")
    
    # 尝试实例化
    indicator = ZxmpatternIndicator()
    print(f"✅ 成功实例化ZXM_PATTERNS指标")
    print(f"指标名称: {indicator.name}")
    print(f"是否继承BaseIndicator: {isinstance(indicator, BaseIndicator)}")
    print(f"minimum_periods: {indicator.minimum_periods}")
    
    # 检查抽象方法
    abstract_methods = [
        '_calculate_baseindicator',
        'calculate_confidence_Indicator_Base_Indicator',
        'calculate_raw_score_Indicator_Base_Indicator',
        'get_patterns_Indicator_Base_Indicator',
        'set_parameters_Indicator_Base_Indicator'
    ]
    
    for method in abstract_methods:
        if hasattr(indicator, method):
            print(f"✅ 方法 {method} 存在")
        else:
            print(f"❌ 方法 {method} 缺失")
    
    print("测试完成！")
    
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
