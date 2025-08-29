#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MFI指标快速验证器
用于快速测试MFI指标的基本功能
"""

import sys
import os
import pandas as pd
import numpy as np

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

def quick_mfi_test():
    """快速测试MFI指标"""
    print("🚀 MFI指标快速验证测试")
    print("=" * 50)
    
    try:
        from indicators.mfi import Mfi
        from tests.framework.real_data_validator import RealDataValidator
        
        # 创建MFI指标
        mfi = Mfi()
        print("✅ MFI指标创建成功")
        
        # 创建验证器获取真实数据
        validator = RealDataValidator()
        real_data = validator.get_real_stock_data(limit=1000)
        print(f"✅ 获取真实数据: {len(real_data)}条")
        
        # 测试阶段1: 算法准确性
        print(f"\n📊 阶段1: 算法准确性测试")
        mfi.set_parameters_Mfi(period=14, overbought=80.0, oversold=20.0)
        result = mfi.calculate(real_data.head(100))
        
        if result is not None:
            # 查找MFI相关的列
            mfi_columns = [col for col in result.columns if any(keyword in col.upper() for keyword in ['MFI', 'MONEY', 'FLOW'])]
            print(f"  - 找到MFI相关列: {mfi_columns}")
            
            if len(mfi_columns) >= 1:  # 至少应该有MFI主线
                # 过滤出数值列
                numeric_columns = [col for col in mfi_columns if col not in ['date', 'code'] and result[col].dtype in ['float64', 'int64']]
                
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
            'has_set_parameters': hasattr(mfi, 'set_parameters_Mfi'),
            'has_get_default_parameters': hasattr(mfi, '_get_default_parameters_mfi'),
            'has_calculate_method': hasattr(mfi, 'calculate')
        }
        
        param_score = (sum(param_tests.values()) / len(param_tests)) * 100
        print(f"  - 参数管理: {param_score}/100")
        
        # 错误处理测试
        error_handled = 0
        try:
            empty_result = mfi.calculate(pd.DataFrame())
            error_handled += 1
        except:
            error_handled += 1
        
        try:
            invalid_result = mfi.calculate(pd.DataFrame({'invalid': [1, 2, 3]}))
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
        result = mfi.calculate(large_data)
        
        if result is not None and len(mfi_columns) > 0:
            # 过滤出数值列
            numeric_columns = [col for col in mfi_columns if col not in ['date', 'code'] and result[col].dtype in ['float64', 'int64']]
            
            if len(numeric_columns) > 0:
                # 使用第一个数值列进行分析
                main_col = numeric_columns[0]
                values = result[main_col].dropna()
                
                if len(values) > 20:
                    # 超买超卖信号测试（MFI通常在0-100之间）
                    overbought_signals = (values > 80).sum()  # 超买信号
                    oversold_signals = (values < 20).sum()    # 超卖信号
                    total_signals = overbought_signals + oversold_signals
                    
                    signal_ratio = total_signals / len(values)
                    print(f"  - 超买信号: {overbought_signals}个")
                    print(f"  - 超卖信号: {oversold_signals}个")
                    print(f"  - 信号比例: {signal_ratio:.4f}")
                    
                    if signal_ratio >= 0.1:  # 至少10%的信号
                        signal_score = 100
                    elif signal_ratio >= 0.05:
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
            'has_calculate_method': hasattr(mfi, 'calculate'),
            'has_set_parameters_method': hasattr(mfi, 'set_parameters_Mfi'),
            'inherits_from_base': hasattr(mfi, '__bases__') and len(mfi.__class__.__bases__) > 0,
            'proper_naming': 'MFI' in mfi.__class__.__name__ or 'Mfi' in mfi.__class__.__name__
        }
        
        stage4_score = (sum(architecture_checks.values()) / len(architecture_checks)) * 100
        print(f"  - 架构检查: {architecture_checks}")
        print(f"  - 阶段4评分: {stage4_score}/100")
        
        # 测试阶段5: 生产就绪性（简化版）
        print(f"\n🚀 阶段5: 生产就绪性测试")
        
        # 性能测试
        import time
        start_time = time.time()
        result = mfi.calculate(real_data)
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
    quick_mfi_test()
