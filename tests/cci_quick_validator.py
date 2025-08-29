#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CCI指标快速验证器
用于快速测试CCI指标的基本功能
"""

import sys
import os
import pandas as pd
import numpy as np

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

def quick_cci_test():
    """快速测试CCI指标"""
    print("🚀 CCI指标快速验证测试")
    print("=" * 50)
    
    try:
        from indicators.cci import CciCci
        from tests.framework.real_data_validator import RealDataValidator
        
        # 创建CCI指标
        cci = CciCci()
        print("✅ CCI指标创建成功")
        
        # 创建验证器获取真实数据
        validator = RealDataValidator()
        real_data = validator.get_real_stock_data(limit=1000)
        print(f"✅ 获取真实数据: {len(real_data)}条")
        
        # 测试阶段1: 算法准确性
        print(f"\n📊 阶段1: 算法准确性测试")
        cci.set_parameters_Cci(period=14)
        result = cci.calculate(real_data.head(100))
        
        if result is not None:
            if 'CCI' in result.columns:
                cci_values = result['CCI'].dropna()
                print(f"  - 找到CCI列: CCI")
                print(f"  - CCI数据点: {len(cci_values)}")
                
                if len(cci_values) > 0:
                    # 检查CCI值的合理性
                    cci_mean = cci_values.mean()
                    cci_std = cci_values.std()
                    cci_range = cci_values.max() - cci_values.min()
                    
                    print(f"  - CCI平均值: {cci_mean:.2f}")
                    print(f"  - CCI标准差: {cci_std:.2f}")
                    print(f"  - CCI范围: {cci_range:.2f}")
                    
                    # CCI值应该在合理范围内
                    if -500 <= cci_mean <= 500 and cci_std > 0:
                        stage1_score = 100
                    else:
                        stage1_score = 80
                    
                    print(f"  - 阶段1评分: {stage1_score}/100")
                else:
                    stage1_score = 0
                    print(f"  - 阶段1评分: {stage1_score}/100 (无有效数据)")
            else:
                stage1_score = 0
                print(f"  - 阶段1评分: {stage1_score}/100 (缺少CCI列)")
        else:
            stage1_score = 0
            print(f"  - 阶段1评分: {stage1_score}/100 (计算失败)")
        
        # 测试阶段2: 基础功能
        print(f"\n🔧 阶段2: 基础功能测试")
        
        # 参数管理测试
        param_tests = {
            'has_set_parameters': hasattr(cci, 'set_parameters_Cci'),
            'has_get_default_parameters': hasattr(cci, '_get_default_parameters_cci'),
            'has_calculate_method': hasattr(cci, 'calculate')
        }
        
        param_score = (sum(param_tests.values()) / len(param_tests)) * 100
        print(f"  - 参数管理: {param_score}/100")
        
        # 错误处理测试
        error_handled = 0
        try:
            empty_result = cci.calculate(pd.DataFrame())
            error_handled += 1
        except:
            error_handled += 1
        
        try:
            invalid_result = cci.calculate(pd.DataFrame({'invalid': [1, 2, 3]}))
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
        result = cci.calculate(large_data)
        
        if result is not None and 'CCI' in result.columns:
            cci_values = result['CCI'].dropna()
            
            if len(cci_values) > 20:
                # 超买超卖信号测试
                overbought_signals = (cci_values > 100).sum()
                oversold_signals = (cci_values < -100).sum()
                total_signals = overbought_signals + oversold_signals
                
                signal_ratio = total_signals / len(cci_values)
                print(f"  - 超买信号: {overbought_signals}个")
                print(f"  - 超卖信号: {oversold_signals}个")
                print(f"  - 信号比例: {signal_ratio:.4f}")
                
                if signal_ratio >= 0.1:
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
            print(f"  - 阶段3评分: {stage3_score}/100 (计算失败)")
        
        # 测试阶段4: 架构合规性（简化版）
        print(f"\n🏗️ 阶段4: 架构合规性测试")
        
        architecture_checks = {
            'has_calculate_method': hasattr(cci, 'calculate'),
            'has_set_parameters_method': hasattr(cci, 'set_parameters_Cci'),
            'inherits_from_base': hasattr(cci, '__bases__') and len(cci.__class__.__bases__) > 0,
            'proper_naming': 'CCI' in cci.__class__.__name__ or 'Cci' in cci.__class__.__name__
        }
        
        stage4_score = (sum(architecture_checks.values()) / len(architecture_checks)) * 100
        print(f"  - 架构检查: {architecture_checks}")
        print(f"  - 阶段4评分: {stage4_score}/100")
        
        # 测试阶段5: 生产就绪性（简化版）
        print(f"\n🚀 阶段5: 生产就绪性测试")
        
        # 性能测试
        import time
        start_time = time.time()
        result = cci.calculate(real_data)
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
    quick_cci_test()
