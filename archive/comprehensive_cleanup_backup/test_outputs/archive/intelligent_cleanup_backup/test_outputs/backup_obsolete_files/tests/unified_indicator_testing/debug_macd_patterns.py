#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试MACD形态识别 - 查看实际返回的形态名称
"""

import sys
import pandas as pd
import numpy as np

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

def debug_macd_patterns():
    """调试MACD形态识别"""
    print("🔍 调试MACD形态识别")
    print("=" * 50)
    
    try:
        # 导入MACD指标
        from indicators.macd import MacdMacd
        macd = MacdMacd()
        print("✅ MACD指标导入成功")
        
        # 创建测试数据
        dates = pd.date_range(start='2023-01-01', periods=60, freq='D')
        prices = 100 + np.cumsum(np.random.normal(0.5, 1, 60))  # 上升趋势
        
        test_data = pd.DataFrame({
            'date': dates,
            'open': prices * 0.99,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, 60)
        })
        
        print(f"📊 测试数据: {len(test_data)}天")
        
        # 计算MACD
        macd_result = macd.calculate(test_data)
        print(f"✅ MACD计算完成")
        print(f"  计算结果类型: {type(macd_result)}")
        if hasattr(macd_result, 'columns'):
            print(f"  计算结果列名: {list(macd_result.columns)}")
        
        # 获取形态
        patterns_result = macd.get_patterns(test_data)
        print(f"✅ 形态识别完成")
        print(f"  形态结果类型: {type(patterns_result)}")
        
        if isinstance(patterns_result, pd.DataFrame):
            print(f"  形态结果形状: {patterns_result.shape}")
            print(f"  形态列名: {list(patterns_result.columns)}")
            
            # 显示每个形态的统计
            print("\n📋 形态统计:")
            for col in patterns_result.columns:
                if patterns_result[col].dtype == bool:
                    count = patterns_result[col].sum()
                    print(f"  {col}: {count}个")
                    
                    # 如果有形态，显示位置
                    if count > 0:
                        positions = patterns_result[patterns_result[col]].index.tolist()
                        print(f"    位置: {positions}")
        else:
            print(f"  形态结果内容: {patterns_result}")
        
        # 检查我们期望的形态名称
        expected_patterns = ['GOLDEN_CROSS', 'DEATH_CROSS', 'DIVERGENCE', 'HISTOGRAM_REVERSAL']
        print(f"\n🎯 期望的形态名称: {expected_patterns}")
        
        if isinstance(patterns_result, pd.DataFrame):
            print("\n🔍 形态名称匹配检查:")
            for pattern in expected_patterns:
                if pattern in patterns_result.columns:
                    print(f"  ✅ {pattern}: 存在")
                else:
                    print(f"  ❌ {pattern}: 不存在")
                    
                    # 查找相似的形态名称
                    similar = [col for col in patterns_result.columns if pattern.lower() in col.lower() or col.lower() in pattern.lower()]
                    if similar:
                        print(f"    🔍 相似形态: {similar}")
        
        return True
        
    except Exception as e:
        print(f"❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_macd_patterns()
