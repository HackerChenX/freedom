#!/usr/bin/env python3
"""
简单的价格生成调试脚本
"""

import random
import numpy as np

def debug_price_progression():
    """调试价格递进逻辑"""
    print("🔍 调试价格递进逻辑")
    
    # 设置固定种子
    random.seed(12345)
    np.random.seed(12345)
    
    base_price = 50.0
    n = 150
    
    print(f"\n基础价格: {base_price}")
    print(f"总天数: {n}")
    
    # 测试OVERSOLD逻辑
    print(f"\n📊 OVERSOLD 逻辑测试:")
    prices = []
    
    for i in range(min(n, 10)):  # 只测试前10天
        progress = i / (n - 1) if n > 1 else 0
        print(f"Day {i}: progress = {progress:.4f}")
        
        # 总下跌幅度设为60%，分阶段递增下跌
        if progress < 0.3:
            # 前30%：小幅下跌（总计10%）
            decline_ratio = 0.10 * (progress / 0.3)
            print(f"  前30%阶段: decline_ratio = 0.10 * ({progress:.4f} / 0.3) = {decline_ratio:.4f}")
        elif progress < 0.7:
            # 中40%：加速下跌（总计再下跌30%）
            decline_ratio = 0.10 + 0.30 * ((progress - 0.3) / 0.4)
            print(f"  中40%阶段: decline_ratio = 0.10 + 0.30 * (({progress:.4f} - 0.3) / 0.4) = {decline_ratio:.4f}")
        else:
            # 后30%：持续下跌到最终（总计再下跌20%）
            decline_ratio = 0.40 + 0.20 * ((progress - 0.7) / 0.3)
            print(f"  后30%阶段: decline_ratio = 0.40 + 0.20 * (({progress:.4f} - 0.7) / 0.3) = {decline_ratio:.4f}")
        
        # 计算新价格
        new_price = base_price * (1 - decline_ratio)
        daily_noise = random.uniform(0.998, 1.002)
        final_price = new_price * daily_noise
        
        print(f"  理论价格: {base_price} * (1 - {decline_ratio:.4f}) = {new_price:.2f}")
        print(f"  随机噪音: {daily_noise:.4f}")
        print(f"  最终价格: {final_price:.2f}")
        print(f"  价格变化: {((final_price / base_price) - 1) * 100:.2f}%")
        print()
        
        prices.append(final_price)
    
    # 测试最后几天
    print(f"\n📊 OVERSOLD 最后几天测试:")
    for i in range(max(n-5, 145), n):  # 最后5天
        progress = i / (n - 1) if n > 1 else 0
        print(f"Day {i}: progress = {progress:.4f}")
        
        # 应该是后30%阶段
        decline_ratio = 0.40 + 0.20 * ((progress - 0.7) / 0.3)
        new_price = base_price * (1 - decline_ratio)
        daily_noise = random.uniform(0.998, 1.002)
        final_price = new_price * daily_noise
        
        print(f"  decline_ratio = 0.40 + 0.20 * (({progress:.4f} - 0.7) / 0.3) = {decline_ratio:.4f}")
        print(f"  最终价格: {final_price:.2f} (变化: {((final_price / base_price) - 1) * 100:.2f}%)")
        print()
    
    # 测试OVERBOUGHT逻辑
    print(f"\n📊 OVERBOUGHT 逻辑测试 (最后几天):")
    random.seed(12345)  # 重置种子
    
    for i in range(max(n-5, 145), n):  # 最后5天
        progress = i / (n - 1) if n > 1 else 0
        print(f"Day {i}: progress = {progress:.4f}")
        
        # 应该是后30%阶段
        rise_ratio = 0.55 + 0.25 * ((progress - 0.7) / 0.3)
        new_price = base_price * (1 + rise_ratio)
        daily_noise = random.uniform(0.998, 1.002)
        final_price = new_price * daily_noise
        
        print(f"  rise_ratio = 0.55 + 0.25 * (({progress:.4f} - 0.7) / 0.3) = {rise_ratio:.4f}")
        print(f"  最终价格: {final_price:.2f} (变化: {((final_price / base_price) - 1) * 100:.2f}%)")
        print()

if __name__ == "__main__":
    debug_price_progression() 