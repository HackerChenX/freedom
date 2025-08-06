#!/usr/bin/env python3
"""
最小化测试 - 逐步隔离阻塞源
"""
import sys

print("Step 1: 基础导入测试")
try:
    import pandas as pd
    print("✅ pandas OK")
    
    import numpy as np
    print("✅ numpy OK")
    
    # 直接测试最可疑的模块
    print("Step 2: 测试 utils.dependency_injection")
    sys.stdout.flush()
    from utils.dependency_injection import get_logger
    print("✅ dependency_injection OK")
    
    print("Step 3: 测试 indicators.pattern_registry") 
    sys.stdout.flush()
    from indicators.pattern_registry import PatternRegistry
    print("✅ pattern_registry OK")
    
    print("🎊 所有测试完成")
    
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()

print("脚本结束")