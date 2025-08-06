#!/usr/bin/env python3
"""
Ultra Think阻塞问题定位脚本
逐步隔离导致脚本不响应中断的根本原因
"""

import sys
import signal
import time

def signal_handler(signum, frame):
    print(f"\n收到信号 {signum}，正在退出...")
    sys.exit(0)

# 注册信号处理器
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

print("开始阻塞问题调试...")

try:
    print("Step 1: 测试基础Python导入")
    import pandas as pd
    import numpy as np
    print("✅ 基础库导入正常")
    
    print("Step 2: 测试简单计算")
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    result = df.sum()
    print("✅ 基础计算正常")
    
    print("Step 3: 开始逐个模块测试...")
    
    # 测试1: 直接导入enums
    print("测试 3.1: 导入enums...")
    from enums.indicator_types import Trend_type, Cross_type
    print("✅ enums导入正常")
    
    # 测试2: 导入indicators.common  
    print("测试 3.2: 导入indicators.common...")
    from indicators.common import crossover, crossunder
    print("✅ indicators.common导入正常")
    
    print("🎯 如果看到这行消息，说明基础模块都正常")
    print("现在开始测试可能有问题的模块...")
    
    # 测试3: 导入base_indicator（最可能的问题源）
    print("测试 3.3: 导入base_indicator...")
    sys.stdout.flush()  # 确保输出立即显示
    
    from indicators.base_indicator import BaseIndicator
    print("✅ BaseIndicator导入正常")
    
    print("测试 3.4: 导入pattern_signal_mixin...")  
    from indicators.base.pattern_signal_mixin import PatternSignalMixin
    print("✅ PatternSignalMixin导入正常")
    
    print("🎊 所有测试通过！问题可能在VIX/KC的具体实现中")
    
except Exception as e:
    print(f"❌ 错误发生在: {e}")
    import traceback
    traceback.print_exc()
    
except KeyboardInterrupt:
    print("\n⚠️ 手动中断")
    sys.exit(0)

print("脚本正常结束")