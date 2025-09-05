#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
详细调试EMA指标阶段3形态识别验证问题
"""

import sys
import os
import pandas as pd
import numpy as np

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

from tests.framework.real_data_validator import RealDataValidator
from tests.ema_standardized_validator import EMAStandardizedValidator

def debug_ema_stage3_detailed():
    """详细调试EMA指标阶段3形态识别验证"""
    print("🔍 详细调试EMA指标阶段3形态识别验证...")
    
    try:
        # 创建EMA验证器
        validator = EMAStandardizedValidator()
        
        from indicators.ema import EmaEma
        ema = EmaEma()
        
        # 获取真实数据
        real_data = validator.get_real_data(limit=2000)
        print(f"✅ 获取真实数据: {len(real_data)}条")
        
        # 创建标准测试数据
        standard_data = validator._create_standard_test_data(2000)
        print(f"✅ 创建标准数据: {len(standard_data)}条")
        
        test_results = {}
        
        # 测试1: 趋势识别（使用真实数据）
        print("\n📊 测试1: 趋势识别（使用真实数据）")
        trend_test = validator._test_trend_identification_with_real_data(ema, real_data)
        test_results['trend_identification'] = trend_test
        print(f"  - 返回结果: {trend_test}")
        print(f"  - 评分: {trend_test.get('score', 0)}")
        
        # 测试2: 信号质量（使用真实数据）
        print("\n📊 测试2: 信号质量（使用真实数据）")
        signal_test = validator._test_signal_quality_with_real_data(ema, real_data)
        test_results['signal_quality'] = signal_test
        print(f"  - 返回结果: {signal_test}")
        print(f"  - 评分: {signal_test.get('score', 0)}")
        
        # 测试3: 多周期分析（使用标准数据）
        print("\n📊 测试3: 多周期分析（使用标准数据）")
        multi_period_test = validator._test_multi_period_analysis(ema, standard_data)
        test_results['multi_period_analysis'] = multi_period_test
        print(f"  - 返回结果: {multi_period_test}")
        print(f"  - 评分: {multi_period_test.get('score', 0)}")
        
        # 测试4: 形态准确性（混合数据）
        print("\n📊 测试4: 形态准确性（混合数据）")
        pattern_test = validator._test_pattern_accuracy(ema, real_data, standard_data)
        test_results['pattern_accuracy'] = pattern_test
        print(f"  - 返回结果: {pattern_test}")
        print(f"  - 评分: {pattern_test.get('score', 0)}")
        
        # 计算总体评分
        print(f"\n📊 评分计算:")
        scores = [result.get('score', 0) for result in test_results.values()]
        print(f"  - 各项评分: {scores}")
        overall_score = sum(scores) / len(scores) if scores else 0
        print(f"  - 总体评分: {overall_score:.1f}/100")
        
        if overall_score < 99.0:
            print(f"❌ 未达到99分标准")
            
            # 分析哪些测试项目低于99分
            print(f"\n🔍 低于99分的测试项目:")
            for test_name, result in test_results.items():
                score = result.get('score', 0)
                if score < 99.0:
                    print(f"  - {test_name}: {score:.1f}分 (低于99分)")
        else:
            print(f"✅ 达到99分标准")
        
    except Exception as e:
        print(f"❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_ema_stage3_detailed()
