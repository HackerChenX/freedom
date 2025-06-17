#!/usr/bin/env python3
"""
验证错误修复效果的脚本
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

def test_aroon_indicator():
    """测试AROON指标是否正常工作"""
    print("🔍 测试AROON指标...")
    try:
        from indicators.aroon import Aroon
        
        # 创建测试数据
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        test_data = pd.DataFrame({
            'date': dates,
            'open': np.random.uniform(10, 20, 50),
            'high': np.random.uniform(15, 25, 50),
            'low': np.random.uniform(5, 15, 50),
            'close': np.random.uniform(10, 20, 50),
            'volume': np.random.uniform(1000, 10000, 50)
        })
        test_data.set_index('date', inplace=True)
        
        # 创建AROON指标实例
        aroon = Aroon()
        
        # 计算指标
        result = aroon.calculate(test_data)
        
        print("✅ AROON指标测试通过")
        return True
        
    except Exception as e:
        print(f"❌ AROON指标测试失败: {e}")
        return False

def test_adx_indicator():
    """测试ADX指标是否正常工作"""
    print("🔍 测试ADX指标...")
    try:
        from indicators.adx import ADX
        
        # 创建测试数据
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        test_data = pd.DataFrame({
            'date': dates,
            'open': np.random.uniform(10, 20, 50),
            'high': np.random.uniform(15, 25, 50),
            'low': np.random.uniform(5, 15, 50),
            'close': np.random.uniform(10, 20, 50),
            'volume': np.random.uniform(1000, 10000, 50)
        })
        test_data.set_index('date', inplace=True)
        
        # 创建ADX指标实例
        adx = ADX()
        
        # 计算指标
        result = adx.calculate(test_data)
        
        print("✅ ADX指标测试通过")
        return True
        
    except Exception as e:
        print(f"❌ ADX指标测试失败: {e}")
        return False

def test_atr_indicator():
    """测试ATR指标是否正常工作"""
    print("🔍 测试ATR指标...")
    try:
        from indicators.atr import ATR
        
        # 创建测试数据
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        test_data = pd.DataFrame({
            'date': dates,
            'open': np.random.uniform(10, 20, 50),
            'high': np.random.uniform(15, 25, 50),
            'low': np.random.uniform(5, 15, 50),
            'close': np.random.uniform(10, 20, 50),
            'volume': np.random.uniform(1000, 10000, 50)
        })
        test_data.set_index('date', inplace=True)
        
        # 创建ATR指标实例
        atr = ATR()
        
        # 计算指标
        result = atr.calculate(test_data)
        
        print("✅ ATR指标测试通过")
        return True
        
    except Exception as e:
        print(f"❌ ATR指标测试失败: {e}")
        return False

def test_buypoint_analyzer():
    """测试买点分析器是否正常工作"""
    print("🔍 测试买点分析器...")
    try:
        from analysis.buypoints.buypoint_batch_analyzer import BuyPointBatchAnalyzer
        
        # 创建分析器实例
        analyzer = BuyPointBatchAnalyzer()
        
        # 测试共性指标报告生成方法
        test_common_indicators = {
            'daily': [
                {
                    'type': 'indicator',
                    'name': 'TEST',
                    'pattern': 'test_pattern',
                    'display_name': '测试指标',
                    'hit_ratio': 0.8,
                    'hit_count': 8,
                    'avg_score': 75.0,
                    'hits': []
                }
            ]
        }
        
        # 测试报告生成（不实际写入文件）
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            temp_file = f.name
        
        analyzer._generate_indicators_report(test_common_indicators, temp_file)
        
        # 清理临时文件
        os.unlink(temp_file)
        
        print("✅ 买点分析器测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 买点分析器测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始验证错误修复效果...\n")
    
    test_results = []
    
    # 测试各个组件
    test_results.append(test_aroon_indicator())
    test_results.append(test_adx_indicator())
    test_results.append(test_atr_indicator())
    test_results.append(test_buypoint_analyzer())
    
    # 统计结果
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"\n📊 测试结果统计:")
    print(f"总测试数: {total}")
    print(f"通过数: {passed}")
    print(f"失败数: {total - passed}")
    print(f"成功率: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 所有测试通过！错误修复验证成功！")
        return True
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败，请检查相关代码。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
