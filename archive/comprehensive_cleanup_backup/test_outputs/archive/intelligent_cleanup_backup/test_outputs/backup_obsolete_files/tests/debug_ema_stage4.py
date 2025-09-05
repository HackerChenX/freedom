#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试EMA指标阶段4架构合规性验证问题
"""

import sys
import os
import pandas as pd
import numpy as np

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

from tests.ema_standardized_validator import EMAStandardizedValidator

def debug_ema_stage4():
    """调试EMA指标阶段4架构合规性验证"""
    print("🔍 调试EMA指标阶段4架构合规性验证...")
    
    try:
        # 创建EMA验证器
        validator = EMAStandardizedValidator()
        
        from indicators.ema import EmaEma
        ema = EmaEma()
        
        test_results = {}
        
        # 测试1: 分层架构
        print("\n📊 测试1: 分层架构")
        architecture_test = validator._test_layered_architecture(ema)
        test_results['layered_architecture'] = architecture_test
        print(f"  - 返回结果: {architecture_test}")
        print(f"  - 评分: {architecture_test.get('score', 0)}")
        
        # 测试2: 依赖注入
        print("\n📊 测试2: 依赖注入")
        di_test = validator._test_dependency_injection(ema)
        test_results['dependency_injection'] = di_test
        print(f"  - 返回结果: {di_test}")
        print(f"  - 评分: {di_test.get('score', 0)}")
        
        # 测试3: 无直接SQL
        print("\n📊 测试3: 无直接SQL")
        sql_test = validator._test_no_direct_sql(ema)
        test_results['no_direct_sql'] = sql_test
        print(f"  - 返回结果: {sql_test}")
        print(f"  - 评分: {sql_test.get('score', 0)}")
        
        # 测试4: 接口合规性
        print("\n📊 测试4: 接口合规性")
        interface_test = validator._test_interface_compliance(ema)
        test_results['interface_compliance'] = interface_test
        print(f"  - 返回结果: {interface_test}")
        print(f"  - 评分: {interface_test.get('score', 0)}")
        
        # 测试5: 关注点分离
        print("\n📊 测试5: 关注点分离")
        separation_test = validator._test_separation_of_concerns(ema)
        test_results['separation_of_concerns'] = separation_test
        print(f"  - 返回结果: {separation_test}")
        print(f"  - 评分: {separation_test.get('score', 0)}")
        
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
                    if 'architecture_checks' in result:
                        print(f"    架构检查: {result['architecture_checks']}")
                    if 'di_checks' in result:
                        print(f"    依赖注入检查: {result['di_checks']}")
                    if 'missing_methods' in result:
                        print(f"    缺失方法: {result['missing_methods']}")
        else:
            print(f"✅ 达到99分标准")
        
    except Exception as e:
        print(f"❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_ema_stage4()
