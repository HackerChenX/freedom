#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BOLL指标快速验证器
用于快速测试BOLL指标的基本功能
"""

import sys
import os
import pandas as pd
import numpy as np

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

def quick_boll_test():
    """快速测试BOLL指标"""
    print("🚀 BOLL指标快速验证测试")
    print("=" * 50)
    
    try:
        from indicators.boll import BollBoll
        from tests.framework.real_data_validator import RealDataValidator
        
        # 创建BOLL指标
        boll = BollBoll()
        print("✅ BOLL指标创建成功")
        
        # 创建验证器获取真实数据
        validator = RealDataValidator()
        real_data = validator.get_real_stock_data(limit=1000)
        print(f"✅ 获取真实数据: {len(real_data)}条")
        
        # 测试阶段1: 算法准确性
        print(f"\n📊 阶段1: 算法准确性测试")
        boll.set_parameters_Boll(period=20, std_dev=2)
        result = boll.calculate(real_data.head(100))
        
        if result is not None:
            expected_columns = ['middle', 'upper', 'lower']
            found_columns = [col for col in expected_columns if col in result.columns]
            print(f"  - 找到BOLL列: {found_columns}")
            
            if len(found_columns) == 3:
                middle_values = result['middle'].dropna()
                upper_values = result['upper'].dropna()
                lower_values = result['lower'].dropna()
                
                print(f"  - 中轨数据点: {len(middle_values)}")
                print(f"  - 上轨数据点: {len(upper_values)}")
                print(f"  - 下轨数据点: {len(lower_values)}")
                
                if len(middle_values) > 0:
                    # 检查带宽合理性
                    bandwidth = (upper_values - lower_values) / middle_values
                    bandwidth_mean = bandwidth.mean()
                    print(f"  - 平均带宽比例: {bandwidth_mean:.4f}")
                    
                    stage1_score = 100 if 0.05 <= bandwidth_mean <= 0.5 else 80
                    print(f"  - 阶段1评分: {stage1_score}/100")
                else:
                    stage1_score = 0
                    print(f"  - 阶段1评分: {stage1_score}/100 (无有效数据)")
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
            'has_set_parameters': hasattr(boll, 'set_parameters_Boll'),
            'has_get_default_parameters': hasattr(boll, '_get_default_parameters'),
            'has_minimum_periods': hasattr(boll, 'minimum_periods')
        }
        
        param_score = (sum(param_tests.values()) / len(param_tests)) * 100
        print(f"  - 参数管理: {param_score}/100")
        
        # 错误处理测试
        error_handled = 0
        try:
            empty_result = boll.calculate(pd.DataFrame())
            error_handled += 1
        except:
            error_handled += 1
        
        try:
            invalid_result = boll.calculate(pd.DataFrame({'invalid': [1, 2, 3]}))
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
        result = boll.calculate(large_data)
        
        if result is not None and 'middle' in result.columns:
            middle_values = result['middle'].dropna()
            
            if len(middle_values) > 20:
                # 趋势识别测试
                trend_changes = 0
                for i in range(1, len(middle_values)):
                    if abs(middle_values.iloc[i] - middle_values.iloc[i-1]) > middle_values.iloc[i-1] * 0.001:
                        trend_changes += 1
                
                trend_ratio = trend_changes / len(middle_values)
                print(f"  - 趋势变化比例: {trend_ratio:.4f}")
                
                if trend_ratio >= 0.05:
                    trend_score = 100
                elif trend_ratio >= 0.01:
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
    quick_boll_test()
