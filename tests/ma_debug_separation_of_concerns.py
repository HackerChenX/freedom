#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MA指标关注点分离调试脚本
"""

import sys
import os

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

def debug_separation_of_concerns():
    """调试关注点分离"""
    from indicators.ma import MaMa
    
    print("🔍 调试MA关注点分离...")
    
    ma = MaMa()
    
    # 获取所有方法（安全地获取）
    all_methods = []
    for method_name in dir(ma):
        if not method_name.startswith('__'):
            try:
                attr = getattr(ma, method_name)
                if callable(attr):
                    all_methods.append(method_name)
            except Exception as e:
                print(f"跳过属性 {method_name}: {e}")
                continue
    
    print(f"总方法数: {len(all_methods)}")
    print(f"所有方法: {all_methods}")
    
    # 按职责分类方法
    calculation_methods = [m for m in all_methods if 'calculate' in m.lower()]
    parameter_methods = [m for m in all_methods if 'parameter' in m.lower() or 'set_' in m.lower()]
    pattern_methods = [m for m in all_methods if 'pattern' in m.lower() or 'signal' in m.lower()]
    utility_methods = [m for m in all_methods if m not in calculation_methods + parameter_methods + pattern_methods]
    
    print(f"\n📊 方法分类:")
    print(f"计算方法 ({len(calculation_methods)}): {calculation_methods}")
    print(f"参数方法 ({len(parameter_methods)}): {parameter_methods}")
    print(f"形态方法 ({len(pattern_methods)}): {pattern_methods}")
    print(f"工具方法 ({len(utility_methods)}): {utility_methods}")
    
    # 检查职责分离情况
    has_calculation_separation = len(calculation_methods) > 0
    has_parameter_separation = len(parameter_methods) > 0
    has_pattern_separation = len(pattern_methods) > 0
    has_utility_separation = len(utility_methods) > 0
    
    print(f"\n✅ 职责分离检查:")
    print(f"有计算分离: {has_calculation_separation}")
    print(f"有参数分离: {has_parameter_separation}")
    print(f"有形态分离: {has_pattern_separation}")
    print(f"有工具分离: {has_utility_separation}")
    
    # 检查方法数量分布是否合理
    total_methods = len(all_methods)
    method_distribution_reasonable = total_methods >= 5  # 至少有5个方法
    
    print(f"方法分布合理: {method_distribution_reasonable} (总数: {total_methods})")
    
    # 评分
    score = 0
    if has_calculation_separation:
        score += 25
        print("✅ +25分: 有计算分离")
    else:
        print("❌ +0分: 无计算分离")
        
    if has_parameter_separation:
        score += 25
        print("✅ +25分: 有参数分离")
    else:
        print("❌ +0分: 无参数分离")
        
    if has_pattern_separation:
        score += 25
        print("✅ +25分: 有形态分离")
    else:
        print("❌ +0分: 无形态分离")
        
    if method_distribution_reasonable:
        score += 25
        print("✅ +25分: 方法分布合理")
    else:
        print("❌ +0分: 方法分布不合理")
    
    print(f"\n📊 最终评分: {score}/100")
    
    return score

if __name__ == "__main__":
    debug_separation_of_concerns()
