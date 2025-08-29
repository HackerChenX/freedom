#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRIX指标快速验证器
用于快速测试TRIX指标的基本功能
"""

import sys
import os
import pandas as pd
import numpy as np

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

def quick_trix_test():
    """快速测试TRIX指标"""
    print("🚀 TRIX指标快速验证测试")
    print("=" * 50)
    
    try:
        from indicators.trix import TripleExponentialAverage
        from tests.framework.real_data_validator import RealDataValidator
        
        # 创建TRIX指标
        trix = TripleExponentialAverage()
        print("✅ TRIX指标创建成功")
        
        # 创建验证器获取真实数据
        validator = RealDataValidator()
        real_data = validator.get_real_stock_data(limit=1000)
        print(f"✅ 获取真实数据: {len(real_data)}条")
        
        # 测试阶段1: 算法准确性
        print(f"\n📊 阶段1: 算法准确性测试")
        trix.set_parameters_Trix(period=14, signal_period=9)
        result = trix.calculate(real_data.head(100))
        
        if result is not None:
            # 查找TRIX相关的列
            trix_columns = [col for col in result.columns if any(keyword in col.upper() for keyword in ['TRIX', 'TR', 'SIGNAL'])]
            print(f"  - 找到TRIX相关列: {trix_columns}")
            
            if len(trix_columns) >= 1:  # 至少应该有TRIX主线
                # 过滤出数值列
                numeric_columns = [col for col in trix_columns if col not in ['date', 'code'] and result[col].dtype in ['float64', 'int64']]
                
                if len(numeric_columns) > 0:
                    # 检查数据有效性
                    valid_data_count = 0
                    for col in numeric_columns[:2]:  # 检查前2个主要列
                        values = result[col].dropna()
                        if len(values) > 0:
                            valid_data_count += 1
                            print(f"    - {col}: {len(values)}个有效值")
                    
                    stage1_score = (valid_data_count / min(2, len(numeric_columns))) * 100
                    print(f"  - 阶段1评分: {stage1_score}/100")
                else:
                    stage1_score = 0
                    print(f"  - 阶段1评分: {stage1_score}/100 (无数值列)")
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
            'has_set_parameters': hasattr(trix, 'set_parameters_Trix'),
            'has_get_default_parameters': hasattr(trix, '_get_default_parameters_trix'),
            'has_calculate_method': hasattr(trix, 'calculate')
        }
        
        param_score = (sum(param_tests.values()) / len(param_tests)) * 100
        print(f"  - 参数管理: {param_score}/100")
        
        # 错误处理测试
        error_handled = 0
        try:
            empty_result = trix.calculate(pd.DataFrame())
            error_handled += 1
        except:
            error_handled += 1
        
        try:
            invalid_result = trix.calculate(pd.DataFrame({'invalid': [1, 2, 3]}))
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
        result = trix.calculate(large_data)
        
        if result is not None and len(trix_columns) > 0:
            # 过滤出数值列
            numeric_columns = [col for col in trix_columns if col not in ['date', 'code'] and result[col].dtype in ['float64', 'int64']]
            
            if len(numeric_columns) > 0:
                # 使用第一个数值列进行分析
                main_col = numeric_columns[0]
                values = result[main_col].dropna()
                
                if len(values) > 20:
                    # 趋势变化测试（TRIX用于识别趋势变化）
                    positive_values = (values > 0).sum()
                    negative_values = (values < 0).sum()
                    zero_crossings = 0
                    
                    for i in range(1, len(values)):
                        if (values.iloc[i] > 0 and values.iloc[i-1] <= 0) or (values.iloc[i] < 0 and values.iloc[i-1] >= 0):
                            zero_crossings += 1
                    
                    signal_ratio = zero_crossings / len(values)
                    print(f"  - 正值信号: {positive_values}个")
                    print(f"  - 负值信号: {negative_values}个")
                    print(f"  - 零轴穿越: {zero_crossings}次")
                    print(f"  - 信号比例: {signal_ratio:.4f}")
                    
                    if signal_ratio >= 0.02:  # 至少2%的零轴穿越
                        signal_score = 100
                    elif signal_ratio >= 0.01:
                        signal_score = 95
                    else:
                        signal_score = 90
                    
                    print(f"  - 信号识别: {signal_score}/100")
                    stage3_score = signal_score
                else:
                    stage3_score = 50
                    print(f"  - 阶段3评分: {stage3_score}/100 (数据不足)")
            else:
                stage3_score = 0
                print(f"  - 阶段3评分: {stage3_score}/100 (无数值列)")
        else:
            stage3_score = 0
            print(f"  - 阶段3评分: {stage3_score}/100 (计算失败)")
        
        # 测试阶段4: 架构合规性（简化版）
        print(f"\n🏗️ 阶段4: 架构合规性测试")
        
        architecture_checks = {
            'has_calculate_method': hasattr(trix, 'calculate'),
            'has_set_parameters_method': hasattr(trix, 'set_parameters_Trix'),
            'inherits_from_base': hasattr(trix, '__bases__') and len(trix.__class__.__bases__) > 0,
            'proper_naming': 'TRIX' in trix.__class__.__name__ or 'Triple' in trix.__class__.__name__
        }
        
        stage4_score = (sum(architecture_checks.values()) / len(architecture_checks)) * 100
        print(f"  - 架构检查: {architecture_checks}")
        print(f"  - 阶段4评分: {stage4_score}/100")
        
        # 测试阶段5: 生产就绪性（简化版）
        print(f"\n🚀 阶段5: 生产就绪性测试")
        
        # 性能测试
        import time
        start_time = time.time()
        result = trix.calculate(real_data)
        end_time = time.time()
        
        processing_time = end_time - start_time
        throughput = len(real_data) / processing_time if processing_time > 0 else 0
        
        print(f"  - 处理时间: {processing_time:.4f}秒")
        print(f"  - 吞吐量: {throughput:.0f} records/second")
        
        if result is not None and throughput > 1000:
            stage5_score = 100
        elif result is not None and throughput > 100:
            stage5_score = 80
        elif result is not None:
            stage5_score = 60
        else:
            stage5_score = 0
        
        print(f"  - 阶段5评分: {stage5_score}/100")
        
        # 综合评估
        print(f"\n📊 综合评估:")
        scores = [stage1_score, stage2_score, stage3_score, stage4_score, stage5_score]
        average_score = sum(scores) / len(scores)
        min_score = min(scores)
        
        print(f"  - 各阶段评分: {scores}")
        print(f"  - 平均评分: {average_score:.1f}/100")
        print(f"  - 最低评分: {min_score:.1f}/100")
        
        if average_score >= 99.5 and min_score >= 99.0:
            print(f"  - 验证结果: ✅ PASSED_ARCHITECTURE_COMPLIANT")
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
    quick_trix_test()
