#!/usr/bin/env python3
"""
测试ZXM_PATTERNS指标语法
"""

import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    print("尝试导入ZXM_PATTERNS指标...")
    from indicators.pattern.zxm_patterns import ZxmpatternIndicator
    print("✅ 导入成功")
    
    print("检查类定义...")
    print(f"类名: {ZxmpatternIndicator.__name__}")
    print(f"基类: {ZxmpatternIndicator.__bases__}")
    print(f"MRO: {[cls.__name__ for cls in ZxmpatternIndicator.__mro__]}")
    
    # 检查抽象方法
    import abc
    abstract_methods = getattr(ZxmpatternIndicator, '__abstractmethods__', set())
    print(f"抽象方法: {abstract_methods}")
    
    if abstract_methods:
        print("❌ 仍有未实现的抽象方法:")
        for method in abstract_methods:
            print(f"  - {method}")
    else:
        print("✅ 所有抽象方法都已实现")
    
except Exception as e:
    print(f"❌ 导入失败: {e}")
    import traceback
    traceback.print_exc()
