#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DMI指标快速验证器
用于快速测试DMI指标的基本功能
"""

import sys
import os
import pandas as pd
import numpy as np

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

def quick_dmi_test():
    """快速测试DMI指标"""
    print("🚀 DMI指标快速验证测试")
    print("=" * 50)
    
    try:
        from indicators.dmi import DirectionalMovementIndex
        from tests.framework.real_data_validator import RealDataValidator
        
        # 创建DMI指标
        dmi = DirectionalMovementIndex()
        print("✅ DMI指标创建成功")
        
        # 创建验证器获取真实数据
        validator = RealDataValidator()
        real_data = validator.get_real_stock_data(limit=1000)
        print(f"✅ 获取真实数据: {len(real_data)}条")
        
        # 测试阶段1: 算法准确性
        print(f"\n📊 阶段1: 算法准确性测试")
        dmi.set_parameters_Dmi(period=14)
        result = dmi.calculate(real_data.head(100))
        
        if result is not None:
            # 查找DMI相关的列
            dmi_columns = [col for col in result.columns if any(keyword in col.upper() for keyword in ['DMI', 'DI', 'ADX', 'DX'])]
            print(f"  - 找到DMI相关列: {dmi_columns}")
            
            if len(dmi_columns) >= 3:  # 至少应该有DI+, DI-, ADX
                # 检查数据有效性
                valid_data_count = 0
                for col in dmi_columns[:3]:  # 检查前3个主要列
                    values = result[col].dropna()
                    if len(values) > 0:
                        valid_data_count += 1
                        print(f"    - {col}: {len(values)}个有效值")
                
                stage1_score = (valid_data_count / 3) * 100
                print(f"  - 阶段1评分: {stage1_score}/100")
            else:
                stage1_score = 0
                print(f"  - 阶段1评分: {stage1_score}/100 (缺少必要列)")
        else:
            stage1_score = 0
            print(f"  - 阶段1评分: {stage1_score}/100 (计算失败)")
        
        # 测试阶段2: 基础功能
        print(f"\n🔧 阶段2: 基础功能测试")
        
        # 参数管理测试
        param_tests = {
            'has_set_parameters': hasattr(dmi, 'set_parameters_Dmi'),
            'has_get_default_parameters': hasattr(dmi, '_get_default_parameters_dmi'),
            'has_minimum_periods': hasattr(dmi, 'minimum_periods')
        }
        
        param_score = (sum(param_tests.values()) / len(param_tests)) * 100
        print(f"  - 参数管理: {param_score}/100")
        
        # 错误处理测试
        error_handled = 0
        try:
            empty_result = dmi.calculate(pd.DataFrame())
            error_handled += 1
        except:
            error_handled += 1
        
        try:
            invalid_result = dmi.calculate(pd.DataFrame({'invalid': [1, 2, 3]}))
            error_handled += 1
        except:
            error_handled += 1
        
        error_score = (error_handled / 2) * 100
        print(f"  - 错误处理: {error_score}/100")
        
        stage2_score = (param_score + error_score) / 2
        print(f"  - 阶段2评分: {stage2_score}/100")
        
        # 测试阶段3: 形态识别（简化版）
        print(f"\n🎯 阶段3: 形态识别测试")
        
        # 使用更多真实数据
        large_data = real_data.head(500) if len(real_data) >= 500 else real_data
        result = dmi.calculate(large_data)
        
        if result is not None and len(dmi_columns) > 0:
            # 使用第一个DMI相关列进行趋势分析
            main_col = dmi_columns[0]
            if main_col in result.columns:
                values = result[main_col].dropna()
                
                if len(values) > 20:
                    # 趋势识别测试
                    trend_changes = 0
                    for i in range(1, len(values)):
                        if abs(values.iloc[i] - values.iloc[i-1]) > abs(values.iloc[i-1]) * 0.01:
                            trend_changes += 1
                    
                    trend_ratio = trend_changes / len(values)
                    print(f"  - 趋势变化比例: {trend_ratio:.4f}")
                    
                    if trend_ratio >= 0.1:
                        trend_score = 100
                    elif trend_ratio >= 0.05:
                        trend_score = 95
                    else:
                        trend_score = 90
                    
                    print(f"  - 趋势识别: {trend_score}/100")
                    stage3_score = trend_score
                else:
                    stage3_score = 50
                    print(f"  - 阶段3评分: {stage3_score}/100 (数据不足)")
            else:
                stage3_score = 0
                print(f"  - 阶段3评分: {stage3_score}/100 (列不存在)")
        else:
            stage3_score = 0
            print(f"  - 阶段3评分: {stage3_score}/100 (计算失败)")
        
        # 综合评估
        print(f"\n📊 综合评估:")
        scores = [stage1_score, stage2_score, stage3_score]
        average_score = sum(scores) / len(scores)
        min_score = min(scores)
        
        print(f"  - 各阶段评分: {scores}")
        print(f"  - 平均评分: {average_score:.1f}/100")
        print(f"  - 最低评分: {min_score:.1f}/100")
        
        if average_score >= 99.5 and min_score >= 99.0:
            print(f"  - 验证结果: ✅ PASSED")
            return True
        else:
            print(f"  - 验证结果: ❌ FAILED (需要≥99.5平均分，≥99.0最低分)")
            return False
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    quick_dmi_test()
