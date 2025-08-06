#!/usr/bin/env python3
"""
逐步测试导入以定位阻塞源
"""
import sys

print("开始逐步导入测试...")

try:
    print("1. 导入基础库...")
    import pandas as pd
    import numpy as np
    print("✅ 基础库正常")
    
    print("2. 测试简单的 get_logger...")
    sys.stdout.flush()
    from utils.dependency_injection import get_logger
    simple_logger = get_logger("test")
    print("✅ get_logger正常")
    
    print("3. 测试 get_container...")
    sys.stdout.flush() 
    from utils.dependency_injection import get_container
    container = get_container()
    print("✅ get_container正常")
    
    print("4. 现在开始导入 PatternRegistry (这里可能卡住)...")
    sys.stdout.flush()
    from indicators.pattern_registry import PatternRegistry
    print("✅ PatternRegistry导入成功")
    
    print("🎊 所有导入测试完成")
    
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()

print("脚本结束")