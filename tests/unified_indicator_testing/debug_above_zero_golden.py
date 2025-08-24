#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试MACD_ABOVE_ZERO_GOLDEN形态问题
"""

import sys
import pandas as pd
import numpy as np

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

def debug_above_zero_golden():
    """调试MACD_ABOVE_ZERO_GOLDEN形态"""
    print("🔍 调试MACD_ABOVE_ZERO_GOLDEN形态")
    print("=" * 60)
    
    try:
        # 导入数据生成器
        from tests.unified_indicator_testing.components.stockinfo_compatible_data_generator import StockInfoCompatibleDataGenerator
        generator = StockInfoCompatibleDataGenerator()
        
        # 生成金叉数据
        print("📊 生成金叉数据...")
        data = generator.generate_stockinfo_compatible_data(
            indicator_name='MACD', 
            pattern_type='GOLDEN_CROSS', 
            stock_code='TEST_DEBUG', 
            history_days=60
        )
        
        print(f"✅ 数据生成成功: {len(data)}天")
        
        # 导入MACD指标
        from indicators.macd import MacdMacd
        macd = MacdMacd()
        
        # 计算MACD
        macd_result = macd.calculate(data)
        print(f"✅ MACD计算完成")
        
        # 获取形态
        patterns_result = macd.get_patterns(data)
        print(f"✅ 形态识别完成")
        
        # 分析MACD值
        print(f"\n📈 MACD数据分析:")
        print(f"  MACD线范围: {macd_result['macd_line'].min():.4f} ~ {macd_result['macd_line'].max():.4f}")
        print(f"  信号线范围: {macd_result['macd_signal'].min():.4f} ~ {macd_result['macd_signal'].max():.4f}")
        print(f"  柱状图范围: {macd_result['macd_histogram'].min():.4f} ~ {macd_result['macd_histogram'].max():.4f}")
        
        # 检查MACD线是否有在零轴上方的时候
        macd_above_zero = macd_result['macd_line'] > 0
        print(f"  MACD线在零轴上方的天数: {macd_above_zero.sum()}/{len(macd_result)}")
        
        if macd_above_zero.sum() > 0:
            print(f"  MACD线上方位置: {macd_result[macd_above_zero].index.tolist()}")
        
        # 检查各种形态的检测结果
        print(f"\n📋 形态检测结果:")
        for col in patterns_result.columns:
            count = patterns_result[col].sum()
            print(f"  {col}: {count}个")
            if count > 0:
                positions = patterns_result[patterns_result[col]].index.tolist()
                print(f"    位置: {positions}")
        
        # 特别分析MACD_ABOVE_ZERO_GOLDEN的条件
        print(f"\n🔍 MACD_ABOVE_ZERO_GOLDEN条件分析:")
        
        # 查找金叉点
        macd_line = macd_result['macd_line']
        signal_line = macd_result['macd_signal']
        
        golden_crosses = []
        for i in range(1, len(macd_line)):
            if (macd_line.iloc[i-1] <= signal_line.iloc[i-1] and 
                macd_line.iloc[i] > signal_line.iloc[i]):
                golden_crosses.append(i)
        
        print(f"  发现的金叉点: {golden_crosses}")
        
        # 检查金叉点是否在零轴上方
        above_zero_golden = []
        for cross_point in golden_crosses:
            macd_value = macd_line.iloc[cross_point]
            signal_value = signal_line.iloc[cross_point]
            print(f"  金叉点{cross_point}: MACD={macd_value:.4f}, Signal={signal_value:.4f}")
            
            if macd_value > 0 and signal_value > 0:
                above_zero_golden.append(cross_point)
                print(f"    ✅ 这是零轴上方的金叉")
            else:
                print(f"    ❌ 这不是零轴上方的金叉")
        
        print(f"  零轴上方的金叉点: {above_zero_golden}")
        
        # 检查MACD_ABOVE_ZERO_GOLDEN的实际检测逻辑
        detected_above_zero = patterns_result['MACD_ABOVE_ZERO_GOLDEN'].sum()
        print(f"  实际检测到的MACD_ABOVE_ZERO_GOLDEN: {detected_above_zero}个")
        
        if detected_above_zero == 0 and len(above_zero_golden) > 0:
            print(f"  ⚠️  问题：应该检测到{len(above_zero_golden)}个，但实际检测到0个")
            print(f"  🔧 可能的原因：MACD_ABOVE_ZERO_GOLDEN的检测逻辑有问题")
        
        return True
        
    except Exception as e:
        print(f"❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_above_zero_golden()
