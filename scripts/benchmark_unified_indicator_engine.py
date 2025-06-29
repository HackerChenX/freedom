#!/usr/bin/env python3
"""
统一指标计算引擎性能基准测试
"""

import os
import sys
import time
import pandas as pd
import numpy as np

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from analysis.engines.unified_indicator_engine import UnifiedIndicatorEngine

def generate_test_data(size: int = 1000) -> pd.DataFrame:
    np.random.seed(42)
    base_price = 10.0
    price_changes = np.random.normal(0, 0.02, size).cumsum()
    close_prices = base_price * (1 + price_changes)
    close_prices = np.maximum(close_prices, 1.0)
    
    high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.01, size)))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.01, size)))
    open_prices = close_prices + np.random.normal(0, 0.005, size)
    volumes = np.random.randint(100000, 1000000, size)
    dates = pd.date_range(start="2020-01-01", periods=size, freq="D")
    
    return pd.DataFrame({
        "date": dates,
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": volumes
    })

def main():
    print("🚀 统一指标计算引擎性能基准测试")
    print("=" * 60)
    
    engine = UnifiedIndicatorEngine(enable_cache=False)
    test_data = generate_test_data(1000)
    
    print("📊 测试MA指标...")
    start_time = time.time()
    ma5 = engine.calculate_ma(test_data, 5)
    ma_time = time.time() - start_time
    print(f"MA5计算时间: {ma_time:.4f}秒")
    
    print("📊 测试MACD指标...")
    start_time = time.time()
    macd_result = engine.calculate_macd(test_data)
    macd_time = time.time() - start_time
    print(f"MACD计算时间: {macd_time:.4f}秒")
    
    print("📊 测试KDJ指标...")
    start_time = time.time()
    kdj_result = engine.calculate_kdj(test_data)
    kdj_time = time.time() - start_time
    print(f"KDJ计算时间: {kdj_time:.4f}秒")
    
    print("📊 测试RSI指标...")
    start_time = time.time()
    rsi = engine.calculate_rsi(test_data)
    rsi_time = time.time() - start_time
    print(f"RSI计算时间: {rsi_time:.4f}秒")
    
    print("\n✅ 计算准确性验证:")
    print(f"MA5长度: {len(ma5)}")
    print(f"MACD DIF长度: {len(macd_result['DIF'])}")
    print(f"KDJ K值范围: {kdj_result['K'].min():.2f} - {kdj_result['K'].max():.2f}")
    print(f"RSI值范围: {rsi.min():.2f} - {rsi.max():.2f}")
    
    stats = engine.get_performance_stats()
    print(f"\n📊 性能统计: 总计算次数: {stats['total_calculations']}")
    print("🎉 基准测试完成！")

if __name__ == "__main__":
    main()
