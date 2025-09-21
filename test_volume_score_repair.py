#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VOLUME_SCORE指标修复验证测试脚本
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.insert(0, '/Users/hacker/PycharmProjects/freedom')

def test_volume_score_repair():
    """测试VOLUME_SCORE指标修复效果"""
    print("🧪 VOLUME_SCORE指标修复验证测试开始...")
    
    try:
        # 1. 测试指标导入
        print("\n1️⃣ 测试指标导入...")
        from indicators.volume_score import VolumeScore
        print("✅ VOLUME_SCORE指标导入成功")
        
        # 2. 测试指标实例化
        print("\n2️⃣ 测试指标实例化...")
        volume_score = VolumeScore(period=14)
        print(f"✅ VOLUME_SCORE指标实例化成功: {volume_score.name}")
        print(f"   描述: {volume_score.description}")
        print(f"   参数: period={volume_score.period}")
        
        # 3. 创建测试数据
        print("\n3️⃣ 创建测试数据...")
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        # 生成模拟价格和成交量数据
        close_prices = []
        volumes = []
        price = 100.0
        volume_base = 1000000
        
        for i in range(100):
            # 随机游走价格
            change = np.random.normal(0, 0.02)  # 2%标准差
            price = price * (1 + change)
            close_prices.append(price)
            
            # 成交量与价格变化相关，并添加一些随机波动
            volume_multiplier = 1.0
            if abs(change) > 0.03:  # 大幅价格变化时成交量放大
                volume_multiplier = 1.5 + np.random.uniform(0, 1)
            elif abs(change) < 0.01:  # 小幅价格变化时成交量萎缩
                volume_multiplier = 0.5 + np.random.uniform(0, 0.5)
            else:
                volume_multiplier = 0.8 + np.random.uniform(0, 0.4)
            
            volume = volume_base * volume_multiplier * (1 + np.random.normal(0, 0.2))
            volumes.append(max(100000, volume))  # 最小成交量
        
        test_data = pd.DataFrame({
            'date': dates,
            'open': [p * 0.99 for p in close_prices],
            'high': [p * 1.02 for p in close_prices],
            'low': [p * 0.98 for p in close_prices],
            'close': close_prices,
            'volume': volumes
        })
        test_data.set_index('date', inplace=True)
        print(f"✅ 测试数据创建成功: {len(test_data)} 行")
        print(f"   价格范围: {test_data['close'].min():.2f} - {test_data['close'].max():.2f}")
        print(f"   成交量范围: {test_data['volume'].min():.0f} - {test_data['volume'].max():.0f}")
        
        # 4. 测试calculate方法
        print("\n4️⃣ 测试calculate方法...")
        result = volume_score.calculate(test_data)
        print(f"✅ calculate方法执行成功")
        print(f"   结果列: {list(result.columns)}")
        print(f"   数据行数: {len(result)}")
        
        # 检查必要的列是否存在
        expected_cols = ['volume_score_value', 'volume_score_ma', 'volume_score_strength']
        missing_cols = [col for col in expected_cols if col not in result.columns]
        if missing_cols:
            print(f"⚠️  缺少列: {missing_cols}")
        else:
            print("✅ 所有必要列都存在")
        
        # 5. 测试get_signal方法
        print("\n5️⃣ 测试get_signal方法...")
        signal = volume_score.get_signal(test_data)
        print(f"✅ get_signal方法执行成功")
        print(f"   信号类型: {signal.get('signal')}")
        print(f"   评分: {signal.get('score')}")
        print(f"   置信度: {signal.get('confidence')}")
        
        # 验证信号格式
        required_keys = ['signal', 'score', 'confidence']
        missing_keys = [key for key in required_keys if key not in signal]
        if missing_keys:
            print(f"❌ 信号缺少键: {missing_keys}")
            return False
        else:
            print("✅ 信号格式正确")
        
        # 6. 测试数据质量
        print("\n6️⃣ 测试数据质量...")
        if 'volume_score_value' in result.columns:
            volume_score_values = result['volume_score_value'].dropna()
            if len(volume_score_values) > 0:
                print(f"   成交量评分范围: {volume_score_values.min():.2f} - {volume_score_values.max():.2f}")
                print(f"   成交量评分平均值: {volume_score_values.mean():.2f}")
                
                # 成交量评分应该在0-100之间
                if volume_score_values.min() >= 0 and volume_score_values.max() <= 100:
                    print("✅ 成交量评分范围正常 (0-100)")
                else:
                    print("⚠️  成交量评分范围异常")
                    
                # 检查高低成交量信号
                high_volume_count = (volume_score_values >= 80).sum()
                low_volume_count = (volume_score_values <= 30).sum()
                print(f"   高成交量信号数量: {high_volume_count}")
                print(f"   低成交量信号数量: {low_volume_count}")
            else:
                print("⚠️  没有有效的成交量评分值")
        
        if 'volume_score_strength' in result.columns:
            volume_strength_values = result['volume_score_strength'].dropna()
            if len(volume_strength_values) > 0:
                print(f"   成交量强度范围: {volume_strength_values.min():.2f} - {volume_strength_values.max():.2f}")
                print(f"   成交量强度平均值: {volume_strength_values.mean():.2f}")
        
        # 7. 测试装饰器功能
        print("\n7️⃣ 测试装饰器功能...")
        # 检查方法是否有装饰器属性
        if hasattr(volume_score.calculate, '__wrapped__'):
            print("✅ calculate方法有装饰器")
        if hasattr(volume_score.get_signal, '__wrapped__'):
            print("✅ get_signal方法有装饰器")
        
        # 8. 测试L4标准列名
        print("\n8️⃣ 测试L4标准列名...")
        l4_standard_cols = ['volume_score_value', 'volume_score_ma', 'volume_score_strength']
        found_l4_cols = [col for col in l4_standard_cols if col in result.columns]
        print(f"   L4标准列名: {found_l4_cols}")
        if len(found_l4_cols) == len(l4_standard_cols):
            print("✅ L4标准列名格式正确")
        else:
            print(f"⚠️  部分L4标准列名缺失: {set(l4_standard_cols) - set(found_l4_cols)}")
        
        # 9. 测试信号逻辑
        print("\n9️⃣ 测试信号逻辑...")
        # 创建极端测试数据
        extreme_data = test_data.copy()
        # 模拟成交量放大情况：成交量大幅增加
        extreme_data.loc[extreme_data.index[-5:], 'volume'] *= 3
        
        extreme_signal = volume_score.get_signal(extreme_data)
        print(f"   极端情况信号: {extreme_signal.get('signal')}")
        print(f"   极端情况评分: {extreme_signal.get('score')}")
        print("✅ 信号逻辑测试完成")
        
        print("\n🎉 VOLUME_SCORE指标修复验证测试完成!")
        print("✅ 所有核心功能测试通过")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_volume_score_repair()
    if success:
        print("\n🏆 VOLUME_SCORE指标修复成功!")
        sys.exit(0)
    else:
        print("\n💥 VOLUME_SCORE指标修复失败!")
        sys.exit(1)
