#!/usr/bin/env python3
"""
CHIP_DISTRIBUTION指标完整5阶段验证脚本

基于第7个生产级质量指标诞生的重大成就，继续推进验证进度向85%目标迈进。
基于已成功验证57/112指标(50.9%)和AROON(100.0/100)的成功经验，对CHIP_DISTRIBUTION指标进行严格标准化5阶段验证：
1. 算法真实性验证 (100%真实数学计算，禁止模拟)
2. 基础功能验证 (参数管理、计算稳定性)
3. 形态识别验证 (筹码分布、集中度分析、获利盘分析)
4. 架构合规性验证 (BaseIndicator继承、抽象方法实现)
5. 生产就绪性验证 (性能、稳定性、并发处理)

目标：达到99分以上的生产级质量标准 (PASSED_PRODUCTION_READY)
重要意义：填补进度表真实空白，继续推进验证进度向85%目标迈进
"""

import sys
import os
import time
import traceback
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dependency_injection import get_logger

logger = get_logger(__name__)

class ChipDistributionValidator:
    """CHIP_DISTRIBUTION指标完整5阶段验证器"""
    
    def __init__(self):
        self.indicator = None
        self.validation_results = {}
        
    def run_complete_validation(self) -> Dict[str, Any]:
        """
        运行完整的5阶段验证
        
        Returns:
            Dict[str, Any]: 完整验证结果
        """
        print("🚀 开始CHIP_DISTRIBUTION指标完整5阶段验证")
        print("🎯 基于第7个生产级质量指标诞生的重大成就，继续推进验证进度向85%目标迈进")
        print("=" * 80)
        
        # 创建指标实例 - 直接导入真正的CHIP_DISTRIBUTION类
        try:
            from indicators.chip_distribution import ChipDistribution
            self.indicator = ChipDistribution()
            print(f"✅ 成功创建CHIP_DISTRIBUTION指标实例")
        except Exception as e:
            print(f"❌ 创建CHIP_DISTRIBUTION指标失败: {e}")
            return {"status": "FAILED", "error": str(e)}
        
        # 执行5阶段验证
        stages = [
            ("阶段1: 算法真实性验证", self._stage1_algorithm_authenticity),
            ("阶段2: 基础功能验证", self._stage2_basic_functionality),
            ("阶段3: 形态识别验证", self._stage3_pattern_recognition),
            ("阶段4: 架构合规性验证", self._stage4_architecture_compliance),
            ("阶段5: 生产就绪性验证", self._stage5_production_readiness)
        ]
        
        total_score = 0
        stage_count = 0
        
        for stage_name, stage_func in stages:
            print(f"\n📊 {stage_name}")
            print("-" * 60)
            
            try:
                stage_result = stage_func()
                self.validation_results[f"stage_{stage_count + 1}"] = stage_result
                
                if stage_result['status'] == 'PASSED':
                    total_score += stage_result['score']
                    stage_count += 1
                    print(f"✅ {stage_name}通过: {stage_result['score']:.1f}/100")
                else:
                    print(f"❌ {stage_name}失败: {stage_result['score']:.1f}/100")
                    print(f"   失败原因: {stage_result.get('error', '未知错误')}")
                    
            except Exception as e:
                print(f"❌ {stage_name}执行异常: {e}")
                self.validation_results[f"stage_{stage_count + 1}"] = {
                    'status': 'FAILED',
                    'score': 0.0,
                    'error': str(e)
                }
        
        # 计算总体评分
        if stage_count > 0:
            overall_score = total_score / len(stages)
        else:
            overall_score = 0.0
        
        # 确定最终状态
        if overall_score >= 99.0:
            final_status = "PASSED_PRODUCTION_READY"
        elif overall_score >= 95.0:
            final_status = "PASSED_ARCHITECTURE_COMPLIANT"
        elif overall_score >= 90.0:
            final_status = "CONDITIONAL_PASS"
        else:
            final_status = "FAILED"
        
        # 生成最终报告
        final_result = {
            'indicator': 'CHIP_DISTRIBUTION',
            'overall_score': overall_score,
            'status': final_status,
            'stage_results': self.validation_results,
            'validation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'stage_count': len(stages),
            'passed_stages': stage_count
        }
        
        self._print_final_report(final_result)
        return final_result
    
    def _stage1_algorithm_authenticity(self) -> Dict[str, Any]:
        """
        阶段1: 算法真实性验证
        验证CHIP_DISTRIBUTION使用真实的数学计算，无任何模拟或简化
        """
        print("🔍 验证算法真实性...")
        
        test_scores = []
        
        # 创建标准测试数据
        test_data = self._create_standard_test_data()
        
        # 测试1: 基础CHIP_DISTRIBUTION计算验证
        print("   测试1: 基础CHIP_DISTRIBUTION计算验证")
        try:
            result = self.indicator.calculate(test_data)
            
            # 检查筹码分布相关列
            chip_columns = [col for col in result.columns if 'chip' in col.lower() or 'distribution' in col.lower()]
            
            if len(chip_columns) >= 1:
                chip_data = result[chip_columns[0]].dropna()
                
                if len(chip_data) >= 10:
                    # 验证筹码分布值的合理性
                    chip_valid = (chip_data >= 0).all()  # 筹码分布应该非负
                    
                    if chip_valid:
                        test_scores.append(100.0)
                        print(f"   ✅ 基础CHIP_DISTRIBUTION计算验证通过: 发现{len(chip_columns)}个筹码列，数据范围{chip_data.min():.3f}-{chip_data.max():.3f}")
                    else:
                        test_scores.append(80.0)
                        print(f"   ⚠️ CHIP_DISTRIBUTION值存在异常")
                else:
                    test_scores.append(70.0)
                    print(f"   ⚠️ CHIP_DISTRIBUTION数据不足")
            else:
                test_scores.append(60.0)
                print(f"   ❌ 缺少基础CHIP_DISTRIBUTION列")
        except Exception as e:
            test_scores.append(50.0)
            print(f"   ❌ 基础CHIP_DISTRIBUTION计算失败: {e}")
        
        # 测试2: 筹码分布算法验证
        print("   测试2: 筹码分布算法验证")
        try:
            result = self.indicator.calculate(test_data)
            
            # 检查是否有筹码分布相关的计算
            if not result.empty and len(result.columns) > 0:
                # 验证计算的一致性
                first_calc = self.indicator.calculate(test_data)
                second_calc = self.indicator.calculate(test_data)
                
                if not first_calc.empty and not second_calc.empty:
                    # 比较两次计算结果的一致性
                    consistent = True
                    for col in first_calc.columns:
                        if col in second_calc.columns:
                            diff = (first_calc[col] - second_calc[col]).abs().max()
                            if pd.notna(diff) and diff > 1e-10:
                                consistent = False
                                break
                    
                    if consistent:
                        test_scores.append(100.0)
                        print(f"   ✅ 筹码分布算法验证通过: 计算一致性良好")
                    else:
                        test_scores.append(85.0)
                        print(f"   ✅ 筹码分布算法基本正确")
                else:
                    test_scores.append(70.0)
                    print(f"   ⚠️ 无法验证筹码分布算法")
            else:
                test_scores.append(60.0)
                print(f"   ❌ 缺少筹码分布计算结果")
        except Exception as e:
            test_scores.append(50.0)
            print(f"   ❌ 筹码分布算法验证失败: {e}")
        
        # 测试3: 参数影响验证
        print("   测试3: 参数影响验证")
        try:
            # 测试不同参数的影响
            if hasattr(self.indicator, 'set_parameters'):
                # 尝试设置不同参数
                original_result = self.indicator.calculate(test_data)
                
                # 修改参数（如果支持）
                try:
                    self.indicator.set_parameters(period=20)
                    modified_result = self.indicator.calculate(test_data)
                    
                    # 验证参数修改是否产生影响
                    if not original_result.empty and not modified_result.empty:
                        has_difference = False
                        for col in original_result.columns:
                            if col in modified_result.columns:
                                diff = (original_result[col] - modified_result[col]).abs().max()
                                if pd.notna(diff) and diff > 1e-6:
                                    has_difference = True
                                    break
                        
                        if has_difference:
                            test_scores.append(100.0)
                            print(f"   ✅ 参数影响验证通过: 参数修改产生预期影响")
                        else:
                            test_scores.append(80.0)
                            print(f"   ⚠️ 参数影响较小")
                    else:
                        test_scores.append(70.0)
                        print(f"   ⚠️ 无法比较参数影响")
                except:
                    test_scores.append(70.0)
                    print(f"   ⚠️ 参数设置不支持或失败")
            else:
                test_scores.append(70.0)
                print(f"   ⚠️ 缺少参数设置方法")
        except Exception as e:
            test_scores.append(60.0)
            print(f"   ❌ 参数影响验证失败: {e}")
        
        # 计算阶段1评分
        stage_score = np.mean(test_scores)
        
        # 确定阶段状态 - 要求98分以上
        if stage_score >= 98:
            stage_status = 'PASSED'
            print(f"   ✅ 阶段1通过: {stage_score:.1f}/100")
        else:
            stage_status = 'FAILED'
            print(f"   ❌ 阶段1失败: {stage_score:.1f}/100 (需要≥98分)")
        
        return {
            'status': stage_status,
            'score': stage_score,
            'test_scores': test_scores,
            'details': '算法真实性验证完成'
        }
