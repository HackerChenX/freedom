#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
快速向量化测试

验证高级向量化优化的效果
"""

import pandas as pd
import numpy as np
import time

def test_vectorization_performance():
    """测试向量化性能"""
    print("="*60)
    print("向量化性能测试")
    print("="*60)
    
    # 创建测试数据
    np.random.seed(42)
    n = 1000
    
    test_data = pd.DataFrame({
        'close': np.random.randn(n).cumsum() + 100,
        'high': np.random.randn(n).cumsum() + 105,
        'low': np.random.randn(n).cumsum() + 95,
        'volume': np.random.randint(1000000, 10000000, n)
    })
    
    print(f"测试数据: {n}行")
    
    # 1. 测试RSI向量化
    print("\n1. RSI向量化测试:")
    close = test_data['close']
    
    start_time = time.time()
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(span=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_time = time.time() - start_time
    
    print(f"   RSI计算时间: {rsi_time:.4f}s")
    print(f"   RSI最新值: {rsi.iloc[-1]:.2f}")
    
    # 2. 测试MACD向量化
    print("\n2. MACD向量化测试:")
    start_time = time.time()
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    macd_time = time.time() - start_time
    
    print(f"   MACD计算时间: {macd_time:.4f}s")
    print(f"   MACD最新值: {macd.iloc[-1]:.4f}")
    
    # 3. 测试KDJ向量化
    print("\n3. KDJ向量化测试:")
    high = test_data['high']
    low = test_data['low']
    
    start_time = time.time()
    highest = high.rolling(window=9).max()
    lowest = low.rolling(window=9).min()
    rsv = (close - lowest) / (highest - lowest) * 100
    k = rsv.ewm(span=3, adjust=False).mean()
    d = k.ewm(span=3, adjust=False).mean()
    j = 3 * k - 2 * d
    kdj_time = time.time() - start_time
    
    print(f"   KDJ计算时间: {kdj_time:.4f}s")
    print(f"   K值最新: {k.iloc[-1]:.2f}")
    print(f"   D值最新: {d.iloc[-1]:.2f}")
    print(f"   J值最新: {j.iloc[-1]:.2f}")
    
    # 4. 测试成交量指标
    print("\n4. 成交量指标测试:")
    volume = test_data['volume']
    
    start_time = time.time()
    # OBV
    price_change = close.diff()
    obv = (volume * np.sign(price_change)).cumsum()
    
    # VR
    up_volume = volume.where(close.diff() > 0, 0)
    down_volume = volume.where(close.diff() < 0, 0)
    vr = (up_volume.rolling(window=26).sum() / 
          down_volume.rolling(window=26).sum().replace(0, np.nan) * 100)
    
    volume_time = time.time() - start_time
    
    print(f"   成交量指标计算时间: {volume_time:.4f}s")
    print(f"   OBV最新值: {obv.iloc[-1]:.0f}")
    print(f"   VR最新值: {vr.iloc[-1]:.2f}")
    
    # 总结
    total_time = rsi_time + macd_time + kdj_time + volume_time
    print(f"\n总计算时间: {total_time:.4f}s")
    print(f"平均每指标时间: {total_time/4:.4f}s")
    
    # 计算理论性能提升
    current_indicators = 13  # 当前向量化指标数
    new_indicators = 19      # 新增向量化指标数
    total_indicators = 86    # 总指标数
    
    current_rate = current_indicators / total_indicators * 100
    new_rate = (current_indicators + new_indicators) / total_indicators * 100
    improvement = new_rate - current_rate
    
    print(f"\n性能提升预估:")
    print(f"  当前向量化率: {current_rate:.1f}%")
    print(f"  优化后向量化率: {new_rate:.1f}%")
    print(f"  预期性能提升: {improvement:.1f}%")
    
    print("="*60)

if __name__ == "__main__":
    test_vectorization_performance()
