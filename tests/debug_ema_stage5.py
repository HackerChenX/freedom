#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试EMA指标阶段5生产就绪性验证问题
"""

import sys
import os
import pandas as pd
import numpy as np

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

from tests.ema_standardized_validator import EMAStandardizedValidator

def debug_ema_stage5():
    """调试EMA指标阶段5生产就绪性验证"""
    print("🔍 调试EMA指标阶段5生产就绪性验证...")
    
    try:
        # 创建EMA验证器
        validator = EMAStandardizedValidator()
        
        from indicators.ema import EmaEma
        ema = EmaEma()
        
        # 获取真实数据（100%）
        real_data = validator.get_real_data(limit=10000)
        print(f"✅ 获取真实数据: {len(real_data)}条")
        
        test_results = {}
        
        # 测试1: 性能测试（使用真实数据）
        print("\n📊 测试1: 性能测试（使用真实数据）")
        performance_test = validator._test_performance_with_real_data(ema, real_data)
        test_results['performance'] = performance_test
        print(f"  - 返回结果: {performance_test}")
        print(f"  - 评分: {performance_test.get('score', 0)}")
        
        # 测试2: 可靠性测试（使用真实数据）
        print("\n📊 测试2: 可靠性测试（使用真实数据）")
        reliability_test = validator._test_reliability_with_real_data(ema, real_data)
        test_results['reliability'] = reliability_test
        print(f"  - 返回结果: {reliability_test}")
        print(f"  - 评分: {reliability_test.get('score', 0)}")
        
        # 测试3: 可维护性测试
        print("\n📊 测试3: 可维护性测试")
        maintainability_test = validator._test_maintainability(ema)
        test_results['maintainability'] = maintainability_test
        print(f"  - 返回结果: {maintainability_test}")
        print(f"  - 评分: {maintainability_test.get('score', 0)}")
        
        # 测试4: 大规模数据测试（使用真实数据）
        print("\n📊 测试4: 大规模数据测试（使用真实数据）")
        scale_test = validator._test_large_scale_with_real_data(ema, real_data)
        test_results['scalability'] = scale_test
        print(f"  - 返回结果: {scale_test}")
        print(f"  - 评分: {scale_test.get('score', 0)}")
        
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
                    if 'error' in result:
                        print(f"    错误: {result['error']}")
                    if 'warning' in result:
                        print(f"    警告: {result['warning']}")
        else:
            print(f"✅ 达到99分标准")
        
    except Exception as e:
        print(f"❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_ema_stage5()
