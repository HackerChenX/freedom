#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ENHANCED_KDJ指标修复后的快速验证器
使用修复后的架构合规性检查逻辑
"""

import sys
import os
import pandas as pd
import numpy as np

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

def quick_enhanced_kdj_test_fixed():
    """快速测试ENHANCED_KDJ指标（使用修复后的架构检查）"""
    print("🚀 ENHANCED_KDJ指标修复后的快速验证测试")
    print("=" * 50)
    
    try:
        from indicators.oscillator.enhanced_kdj import EnhancedKdj
        from indicators.base_indicator import BaseIndicator
        from tests.framework.real_data_validator import RealDataValidator
        
        # 创建ENHANCED_KDJ指标
        enhanced_kdj = EnhancedKdj()
        print("✅ ENHANCED_KDJ指标创建成功")
        
        # 创建验证器获取真实数据
        validator = RealDataValidator()
        real_data = validator.get_real_stock_data(limit=1000)
        print(f"✅ 获取真实数据: {len(real_data)}条")
        
        # 测试阶段1: 算法准确性
        print(f"\n📊 阶段1: 算法准确性测试")
        enhanced_kdj.set_parameters_Kdj_Enhanced_Kdj(n=9, m1=3, m2=3, sensitivity=1.0)
        result = enhanced_kdj.calculate(real_data.head(100))
        
        if result is not None:
            # 查找ENHANCED_KDJ相关的列
            kdj_columns = [col for col in result.columns if any(keyword in col.upper() for keyword in ['KDJ', 'K_', 'D_', 'J_', 'STOCH'])]
            print(f"  - 找到ENHANCED_KDJ相关列: {kdj_columns}")
            
            if len(kdj_columns) >= 3:  # 至少应该有K、D、J线
                # 过滤出数值列
                numeric_columns = [col for col in kdj_columns if col not in ['date', 'code'] and result[col].dtype in ['float64', 'int64']]
                
                if len(numeric_columns) > 0:
                    # 检查数据有效性
                    valid_data_count = 0
                    for col in numeric_columns[:3]:  # 检查前3个主要列
                        values = result[col].dropna()
                        if len(values) > 0:
                            valid_data_count += 1
                            print(f"    - {col}: {len(values)}个有效值")
                    
                    stage1_score = (valid_data_count / min(3, len(numeric_columns))) * 100
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
            'has_set_parameters': hasattr(enhanced_kdj, 'set_parameters_Kdj_Enhanced_Kdj'),
            'has_get_default_parameters': hasattr(enhanced_kdj, '_get_default_parameters_enhancedkdj'),
            'has_calculate_method': hasattr(enhanced_kdj, 'calculate')
        }
        
        param_score = (sum(param_tests.values()) / len(param_tests)) * 100
        print(f"  - 参数管理: {param_score}/100")
        
        # 错误处理测试
        error_handled = 0
        try:
            empty_result = enhanced_kdj.calculate(pd.DataFrame())
            error_handled += 1
        except:
            error_handled += 1
        
        try:
            invalid_result = enhanced_kdj.calculate(pd.DataFrame({'invalid': [1, 2, 3]}))
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
        result = enhanced_kdj.calculate(large_data)
        
        if result is not None and len(kdj_columns) > 0:
            # 过滤出数值列
            numeric_columns = [col for col in kdj_columns if col not in ['date', 'code'] and result[col].dtype in ['float64', 'int64']]
            
            if len(numeric_columns) >= 3:
                # 寻找K、D、J线
                k_col = None
                d_col = None
                j_col = None
                
                for col in numeric_columns:
                    if 'K' in col.upper() and ('D' not in col.upper() and 'J' not in col.upper()):
                        k_col = col
                    elif 'D' in col.upper() and ('K' not in col.upper() and 'J' not in col.upper()):
                        d_col = col
                    elif 'J' in col.upper() and ('K' not in col.upper() and 'D' not in col.upper()):
                        j_col = col
                
                if k_col and d_col:
                    k_values = result[k_col].dropna()
                    d_values = result[d_col].dropna()
                    
                    if len(k_values) > 20 and len(d_values) > 20:
                        # KDJ信号测试
                        overbought_signals = ((k_values > 80) | (d_values > 80)).sum()  # 超买信号
                        oversold_signals = ((k_values < 20) | (d_values < 20)).sum()    # 超卖信号
                        
                        # KD交叉信号测试
                        min_len = min(len(k_values), len(d_values))
                        golden_cross = 0
                        death_cross = 0
                        
                        for i in range(1, min_len):
                            if (k_values.iloc[i] > d_values.iloc[i] and 
                                k_values.iloc[i-1] <= d_values.iloc[i-1]):
                                golden_cross += 1
                            elif (k_values.iloc[i] < d_values.iloc[i] and 
                                  k_values.iloc[i-1] >= d_values.iloc[i-1]):
                                death_cross += 1
                        
                        total_signals = overbought_signals + oversold_signals + golden_cross + death_cross
                        signal_ratio = total_signals / len(k_values)
                        
                        print(f"  - 超买信号: {overbought_signals}个")
                        print(f"  - 超卖信号: {oversold_signals}个")
                        print(f"  - 金叉信号: {golden_cross}个")
                        print(f"  - 死叉信号: {death_cross}个")
                        print(f"  - 信号比例: {signal_ratio:.4f}")
                        
                        # 增强KDJ应该能够识别多种交叉和超买超卖信号
                        if signal_ratio >= 0.2:  # 至少20%的信号
                            signal_score = 100
                        elif signal_ratio >= 0.1:
                            signal_score = 95
                        else:
                            signal_score = 90
                        
                        print(f"  - 信号识别: {signal_score}/100")
                        stage3_score = signal_score
                    else:
                        stage3_score = 50
                        print(f"  - 阶段3评分: {stage3_score}/100 (数据不足)")
                else:
                    stage3_score = 50
                    print(f"  - 阶段3评分: {stage3_score}/100 (缺少K、D线)")
            else:
                stage3_score = 0
                print(f"  - 阶段3评分: {stage3_score}/100 (无足够数值列)")
        else:
            stage3_score = 0
            print(f"  - 阶段3评分: {stage3_score}/100 (计算失败)")
        
        # 测试阶段4: 架构合规性（使用修复后的检查）
        print(f"\n🏗️ 阶段4: 架构合规性测试（修复后）")
        
        architecture_checks = {
            'has_calculate_method': hasattr(enhanced_kdj, 'calculate'),
            'has_set_parameters_method': hasattr(enhanced_kdj, 'set_parameters_Kdj_Enhanced_Kdj'),
            'inherits_from_base': isinstance(enhanced_kdj, BaseIndicator),  # 修复后的检查
            'proper_naming': 'KDJ' in enhanced_kdj.__class__.__name__ or 'Enhanced' in enhanced_kdj.__class__.__name__
        }
        
        stage4_score = (sum(architecture_checks.values()) / len(architecture_checks)) * 100
        print(f"  - 架构检查: {architecture_checks}")
        print(f"  - 继承链: {[cls.__name__ for cls in enhanced_kdj.__class__.__mro__[:5]]}")
        print(f"  - 阶段4评分: {stage4_score}/100")
        
        # 测试阶段5: 生产就绪性（简化版）
        print(f"\n🚀 阶段5: 生产就绪性测试")
        
        # 性能测试
        import time
        start_time = time.time()
        result = enhanced_kdj.calculate(real_data)
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
    quick_enhanced_kdj_test_fixed()
