#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MACD指标诊断工具 - Ultra Think方法论应用

用于深度诊断MACD指标的计算逻辑、形态识别算法和数据流向问题
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

def diagnose_macd_calculation():
    """诊断MACD基础计算逻辑"""
    print("🔍 开始诊断MACD基础计算逻辑")
    print("=" * 60)
    
    try:
        # 导入MACD指标
        from analysis.indicators.core_indicators import MACD
        
        # 创建测试数据
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        # 创建一个明显的上升趋势，便于观察MACD行为
        prices = 100 + np.cumsum(np.random.normal(0.5, 1, 100))  # 上升趋势
        
        test_data = pd.DataFrame({
            'date': dates,
            'open': prices * 0.99,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, 100)
        })
        
        print(f"📊 测试数据概览:")
        print(f"  数据长度: {len(test_data)}")
        print(f"  价格范围: {test_data['close'].min():.2f} - {test_data['close'].max():.2f}")
        print(f"  价格趋势: {'上升' if test_data['close'].iloc[-1] > test_data['close'].iloc[0] else '下降'}")
        
        # 初始化MACD指标
        macd = MACD()
        print(f"\n✅ MACD指标初始化成功")
        print(f"  默认参数: {macd.parameters}")
        
        # 计算MACD
        result = macd.calculate(test_data)
        print(f"\n📈 MACD计算结果:")
        print(f"  结果类型: {type(result)}")
        print(f"  结果形状: {result.shape if hasattr(result, 'shape') else 'N/A'}")
        
        if isinstance(result, pd.DataFrame):
            print(f"  列名: {list(result.columns)}")
            print(f"  非空值数量: {result.count().to_dict()}")
            
            # 显示最后几行数据
            print(f"\n📋 最后5行MACD数据:")
            print(result.tail().to_string())
            
            # 检查是否有明显的金叉死叉
            if 'macd' in result.columns and 'signal' in result.columns:
                macd_line = result['macd'].dropna()
                signal_line = result['signal'].dropna()
                
                if len(macd_line) > 1 and len(signal_line) > 1:
                    # 寻找金叉死叉点
                    crosses = []
                    for i in range(1, min(len(macd_line), len(signal_line))):
                        if (macd_line.iloc[i-1] <= signal_line.iloc[i-1] and 
                            macd_line.iloc[i] > signal_line.iloc[i]):
                            crosses.append(('金叉', i, macd_line.iloc[i], signal_line.iloc[i]))
                        elif (macd_line.iloc[i-1] >= signal_line.iloc[i-1] and 
                              macd_line.iloc[i] < signal_line.iloc[i]):
                            crosses.append(('死叉', i, macd_line.iloc[i], signal_line.iloc[i]))
                    
                    print(f"\n🔄 发现的交叉点: {len(crosses)}个")
                    for cross_type, idx, macd_val, signal_val in crosses[-3:]:  # 显示最后3个
                        print(f"  {cross_type}: 位置{idx}, MACD={macd_val:.4f}, Signal={signal_val:.4f}")
        
        return True, result
        
    except Exception as e:
        print(f"❌ MACD计算诊断失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def diagnose_pattern_recognition():
    """诊断MACD形态识别逻辑"""
    print("\n🔍 开始诊断MACD形态识别逻辑")
    print("=" * 60)
    
    try:
        from analysis.indicators.core_indicators import MACD
        
        # 创建包含明显金叉的数据
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        
        # 前半段下降，后半段上升，制造金叉
        prices_down = 100 - np.linspace(0, 10, 25)  # 下降
        prices_up = 90 + np.linspace(0, 15, 25)     # 上升
        prices = np.concatenate([prices_down, prices_up])
        
        test_data = pd.DataFrame({
            'date': dates,
            'open': prices * 0.99,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.randint(1000000, 2000000, 50)
        })
        
        print(f"📊 形态测试数据概览:")
        print(f"  设计模式: 前半段下降，后半段上升（制造金叉）")
        print(f"  价格变化: {test_data['close'].iloc[0]:.2f} → {test_data['close'].iloc[24]:.2f} → {test_data['close'].iloc[-1]:.2f}")
        
        # 计算MACD
        macd = MACD()
        result = macd.calculate(test_data)
        
        # 获取形态
        patterns = macd.get_patterns()
        print(f"\n📋 形态识别结果:")
        print(f"  形态类型: {type(patterns)}")
        print(f"  形态形状: {patterns.shape if hasattr(patterns, 'shape') else 'N/A'}")
        
        if isinstance(patterns, pd.DataFrame):
            print(f"  形态列名: {list(patterns.columns)}")
            
            # 检查各种形态
            for pattern_name in ['GOLDEN_CROSS', 'DEATH_CROSS', 'DIVERGENCE', 'HISTOGRAM_REVERSAL']:
                if pattern_name in patterns.columns:
                    pattern_count = patterns[pattern_name].sum()
                    print(f"  {pattern_name}: {pattern_count}个")
                    
                    if pattern_count > 0:
                        pattern_indices = patterns[patterns[pattern_name]].index.tolist()
                        print(f"    位置: {pattern_indices}")
        
        # 测试特定形态分析
        for pattern_name in ['GOLDEN_CROSS', 'DEATH_CROSS']:
            try:
                analysis = macd.analyze_pattern(pattern_name)
                print(f"\n🔍 {pattern_name} 分析结果:")
                print(f"  分析类型: {type(analysis)}")
                if isinstance(analysis, dict):
                    for key, value in analysis.items():
                        print(f"  {key}: {value}")
            except Exception as e:
                print(f"❌ {pattern_name} 分析失败: {e}")
        
        return True, patterns
        
    except Exception as e:
        print(f"❌ 形态识别诊断失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def diagnose_data_generator():
    """诊断数据生成器的输出质量"""
    print("\n🔍 开始诊断数据生成器输出质量")
    print("=" * 60)
    
    try:
        from tests.unified_indicator_testing.components.stockinfo_compatible_data_generator import StockInfoCompatibleDataGenerator
        
        generator = StockInfoCompatibleDataGenerator()
        print("✅ 数据生成器初始化成功")
        
        # 测试各种形态的数据生成
        patterns_to_test = ['GOLDEN_CROSS', 'DEATH_CROSS', 'DIVERGENCE', 'HISTOGRAM_REVERSAL']
        
        for pattern_name in patterns_to_test:
            print(f"\n📊 测试形态: {pattern_name}")
            
            try:
                # 生成数据
                generated_data = generator.generate_data_for_pattern('MACD', pattern_name)
                
                print(f"  生成数据类型: {type(generated_data)}")
                print(f"  生成数据形状: {generated_data.shape if hasattr(generated_data, 'shape') else 'N/A'}")
                
                if isinstance(generated_data, pd.DataFrame):
                    print(f"  数据列名: {list(generated_data.columns)}")
                    print(f"  数据长度: {len(generated_data)}")
                    print(f"  价格范围: {generated_data['close'].min():.2f} - {generated_data['close'].max():.2f}")
                    
                    # 用MACD指标验证生成的数据
                    from analysis.indicators.core_indicators import MACD
                    macd = MACD()
                    
                    # 计算MACD
                    macd_result = macd.calculate(generated_data)
                    patterns_result = macd.get_patterns()
                    
                    # 检查是否真的包含目标形态
                    if isinstance(patterns_result, pd.DataFrame) and pattern_name in patterns_result.columns:
                        pattern_count = patterns_result[pattern_name].sum()
                        print(f"  ✅ 验证结果: 发现{pattern_count}个{pattern_name}形态")
                        
                        if pattern_count == 0:
                            print(f"  ⚠️  警告: 生成的数据未包含目标形态！")
                    else:
                        print(f"  ❌ 验证失败: 无法检测形态")
                
            except Exception as e:
                print(f"  ❌ {pattern_name} 数据生成失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据生成器诊断失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_comprehensive_diagnosis():
    """运行综合诊断"""
    print("🚀 MACD指标Ultra Think综合诊断")
    print("=" * 80)
    print(f"📅 诊断时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    results = {}
    
    # 1. 基础计算诊断
    calc_success, calc_result = diagnose_macd_calculation()
    results['calculation'] = {'success': calc_success, 'result': calc_result}
    
    # 2. 形态识别诊断
    pattern_success, pattern_result = diagnose_pattern_recognition()
    results['pattern_recognition'] = {'success': pattern_success, 'result': pattern_result}
    
    # 3. 数据生成器诊断
    generator_success = diagnose_data_generator()
    results['data_generator'] = {'success': generator_success}
    
    # 综合评估
    print("\n" + "=" * 80)
    print("📊 综合诊断结果")
    print("=" * 80)
    
    total_tests = 3
    passed_tests = sum([calc_success, pattern_success, generator_success])
    
    print(f"✅ 通过测试: {passed_tests}/{total_tests}")
    print(f"📈 通过率: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("🎉 所有诊断测试通过！")
    else:
        print("⚠️  发现问题，需要进一步修复")
        
        if not calc_success:
            print("  🔧 需要修复: MACD基础计算逻辑")
        if not pattern_success:
            print("  🔧 需要修复: 形态识别算法")
        if not generator_success:
            print("  🔧 需要修复: 数据生成器")
    
    return results

if __name__ == "__main__":
    results = run_comprehensive_diagnosis()
