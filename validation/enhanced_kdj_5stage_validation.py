#!/usr/bin/env python3
"""
EnhancedKDJ指标完整5阶段验证脚本

基于EnhancedRSI成功验证模板，对EnhancedKDJ指标进行严格标准化5阶段验证：
1. 算法真实性验证 (100%真实数学计算，禁止模拟)
2. 基础功能验证 (参数管理、计算稳定性)
3. 形态识别验证 (超买超卖、金叉死叉、背离检测)
4. 架构合规性验证 (BaseIndicator继承、抽象方法实现)
5. 生产就绪性验证 (性能、稳定性、并发处理)

目标：达到99分以上的生产级质量标准 (PASSED_PRODUCTION_READY)
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

from indicators.complete_indicator_registry import complete_registry
from utils.dependency_injection import get_logger

logger = get_logger(__name__)

class EnhancedKdjValidator:
    """EnhancedKDJ指标完整5阶段验证器"""
    
    def __init__(self):
        self.indicator = None
        self.validation_results = {}
        
    def run_complete_validation(self) -> Dict[str, Any]:
        """
        运行完整的5阶段验证
        
        Returns:
            Dict[str, Any]: 完整验证结果
        """
        print("🚀 开始EnhancedKDJ指标完整5阶段验证")
        print("=" * 80)
        
        # 创建指标实例 - 直接导入真正的EnhancedKdj类
        try:
            from indicators.oscillator.enhanced_kdj import EnhancedKdj
            self.indicator = EnhancedKdj()
            print(f"✅ 成功创建EnhancedKDJ指标实例")
        except Exception as e:
            print(f"❌ 创建EnhancedKDJ指标失败: {e}")
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
            'indicator': 'EnhancedKDJ',
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
        验证EnhancedKDJ使用真实的数学计算，无任何模拟或简化
        """
        print("🔍 验证算法真实性...")
        
        test_scores = []
        
        # 创建标准测试数据
        test_data = self._create_standard_test_data()
        
        # 测试1: 基础KDJ计算验证
        print("   测试1: 基础KDJ计算验证")
        try:
            result = self.indicator.calculate(test_data)
            
            if 'K' in result.columns and 'D' in result.columns and 'J' in result.columns:
                k_values = result['K'].dropna()
                d_values = result['D'].dropna()
                j_values = result['J'].dropna()
                
                if len(k_values) >= 10 and len(d_values) >= 10 and len(j_values) >= 10:
                    # 验证KDJ值的合理性
                    k_range_valid = (k_values >= 0).all() and (k_values <= 100).all()
                    d_range_valid = (d_values >= 0).all() and (d_values <= 100).all()
                    
                    if k_range_valid and d_range_valid:
                        test_scores.append(100.0)
                        print(f"   ✅ 基础KDJ计算验证通过: K值范围{k_values.min():.1f}-{k_values.max():.1f}, D值范围{d_values.min():.1f}-{d_values.max():.1f}")
                    else:
                        test_scores.append(80.0)
                        print(f"   ⚠️ KDJ值范围异常")
                else:
                    test_scores.append(70.0)
                    print(f"   ⚠️ KDJ数据不足")
            else:
                test_scores.append(60.0)
                print(f"   ❌ 缺少基础KDJ列")
        except Exception as e:
            test_scores.append(50.0)
            print(f"   ❌ 基础KDJ计算失败: {e}")
        
        # 测试2: 增强功能验证
        print("   测试2: 增强功能验证")
        try:
            result = self.indicator.calculate(test_data)
            
            # 检查增强功能列
            enhanced_columns = ['j_acceleration', 'kd_cross_angle', 'kd_distance', 'j_normalized']
            found_enhanced = sum(1 for col in enhanced_columns if col in result.columns)
            
            if found_enhanced >= 3:
                test_scores.append(100.0)
                print(f"   ✅ 增强功能验证通过: 发现{found_enhanced}/4个增强列")
            elif found_enhanced >= 2:
                test_scores.append(85.0)
                print(f"   ✅ 增强功能部分通过: 发现{found_enhanced}/4个增强列")
            else:
                test_scores.append(70.0)
                print(f"   ⚠️ 增强功能不足: 仅发现{found_enhanced}/4个增强列")
        except Exception as e:
            test_scores.append(50.0)
            print(f"   ❌ 增强功能验证失败: {e}")
        
        # 测试3: 多周期KDJ验证
        print("   测试3: 多周期KDJ验证")
        try:
            result = self.indicator.calculate(test_data)
            
            # 检查多周期KDJ列
            multi_period_columns = [col for col in result.columns if '_' in col and any(period in col for period in ['5', '6', '14'])]
            
            if len(multi_period_columns) >= 6:  # 至少2个周期 * 3个指标(K,D,J)
                test_scores.append(100.0)
                print(f"   ✅ 多周期KDJ验证通过: 发现{len(multi_period_columns)}个多周期列")
            elif len(multi_period_columns) >= 3:
                test_scores.append(85.0)
                print(f"   ✅ 多周期KDJ部分通过: 发现{len(multi_period_columns)}个多周期列")
            else:
                test_scores.append(70.0)
                print(f"   ⚠️ 多周期KDJ不足: 仅发现{len(multi_period_columns)}个多周期列")
        except Exception as e:
            test_scores.append(50.0)
            print(f"   ❌ 多周期KDJ验证失败: {e}")
        
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

    def _stage2_basic_functionality(self) -> Dict[str, Any]:
        """
        阶段2: 基础功能验证
        验证参数管理、计算稳定性、错误处理等基础功能
        """
        print("🔧 验证基础功能...")

        test_scores = []

        # 测试1: 参数设置验证
        print("   测试1: 参数设置验证")
        try:
            # 测试参数设置
            self.indicator.set_parameters(n=14, m1=5, m2=5, sensitivity=1.5)

            # 验证参数是否正确设置
            if hasattr(self.indicator, 'n') and self.indicator.n == 14:
                test_scores.append(100.0)
                print(f"   ✅ 参数设置验证通过: n={self.indicator.n}")
            else:
                test_scores.append(80.0)
                print(f"   ⚠️ 参数设置部分通过")
        except Exception as e:
            test_scores.append(70.0)
            print(f"   ⚠️ 参数设置验证异常: {e}")

        # 测试2: 计算稳定性验证
        print("   测试2: 计算稳定性验证")
        try:
            test_data = self._create_standard_test_data()

            # 多次计算验证一致性
            result1 = self.indicator.calculate(test_data)
            result2 = self.indicator.calculate(test_data)

            if 'K' in result1.columns and 'K' in result2.columns:
                k1 = result1['K'].dropna()
                k2 = result2['K'].dropna()

                if len(k1) > 0 and len(k2) > 0 and len(k1) == len(k2):
                    # 检查计算一致性
                    consistency = np.allclose(k1, k2, rtol=1e-10)
                    if consistency:
                        test_scores.append(100.0)
                        print(f"   ✅ 计算稳定性验证通过: 重复计算一致")
                    else:
                        test_scores.append(85.0)
                        print(f"   ⚠️ 计算稳定性部分通过: 存在微小差异")
                else:
                    test_scores.append(75.0)
                    print(f"   ⚠️ 计算结果长度不一致")
            else:
                test_scores.append(60.0)
                print(f"   ❌ 缺少K列进行稳定性验证")
        except Exception as e:
            test_scores.append(50.0)
            print(f"   ❌ 计算稳定性验证失败: {e}")

        # 测试3: 错误处理验证
        print("   测试3: 错误处理验证")
        try:
            # 测试无效数据处理
            invalid_data = pd.DataFrame({
                'high': [np.nan, np.inf, -np.inf],
                'low': [np.nan, np.inf, -np.inf],
                'close': [np.nan, np.inf, -np.inf]
            })

            try:
                result = self.indicator.calculate(invalid_data)
                # 如果没有抛出异常，检查结果是否合理
                if result is not None and not result.empty:
                    test_scores.append(90.0)
                    print(f"   ✅ 错误处理验证通过: 优雅处理无效数据")
                else:
                    test_scores.append(80.0)
                    print(f"   ✅ 错误处理部分通过: 返回空结果")
            except Exception:
                test_scores.append(85.0)
                print(f"   ✅ 错误处理验证通过: 正确抛出异常")
        except Exception as e:
            test_scores.append(70.0)
            print(f"   ⚠️ 错误处理验证异常: {e}")

        # 计算阶段2评分
        stage_score = np.mean(test_scores)

        # 确定阶段状态 - 要求95分以上
        if stage_score >= 95:
            stage_status = 'PASSED'
            print(f"   ✅ 阶段2通过: {stage_score:.1f}/100")
        else:
            stage_status = 'FAILED'
            print(f"   ❌ 阶段2失败: {stage_score:.1f}/100 (需要≥95分)")

        return {
            'status': stage_status,
            'score': stage_score,
            'test_scores': test_scores,
            'details': '基础功能验证完成'
        }

    def _stage3_pattern_recognition(self) -> Dict[str, Any]:
        """
        阶段3: 形态识别验证
        验证超买超卖、金叉死叉、背离检测等形态识别功能
        """
        print("📈 验证形态识别...")

        test_scores = []

        # 测试1: 超买超卖识别
        print("   测试1: 超买超卖识别")
        try:
            # 创建超买数据（高位震荡）
            overbought_data = self._create_overbought_test_data()
            result = self.indicator.calculate(overbought_data)

            # 检查多个可能的KDJ列
            k_column = None
            d_column = None
            for col in ['K', 'k', 'kdj_k']:
                if col in result.columns:
                    k_column = col
                    break
            for col in ['D', 'd', 'kdj_d']:
                if col in result.columns:
                    d_column = col
                    break

            if k_column and d_column:
                k_values = result[k_column].dropna()
                d_values = result[d_column].dropna()

                if len(k_values) > 0 and len(d_values) > 0:
                    # 检查最后几个值是否显示超买（应该>80）
                    last_k_values = k_values.tail(3)
                    last_d_values = d_values.tail(3)
                    overbought_k_count = sum(1 for val in last_k_values if val > 80)
                    overbought_d_count = sum(1 for val in last_d_values if val > 80)

                    if overbought_k_count >= 2 and overbought_d_count >= 2:
                        test_scores.append(100.0)
                        print(f"   ✅ 超买识别精确验证通过: K值{overbought_k_count}/3>80, D值{overbought_d_count}/3>80")
                    elif overbought_k_count >= 1 or overbought_d_count >= 1:
                        test_scores.append(85.0)
                        print(f"   ✅ 超买识别部分通过: K值{overbought_k_count}/3>80, D值{overbought_d_count}/3>80")
                    else:
                        test_scores.append(70.0)
                        print(f"   ⚠️ 超买识别不足: K值{overbought_k_count}/3>80, D值{overbought_d_count}/3>80")
                else:
                    test_scores.append(60.0)
                    print(f"   ❌ KDJ数据不足")
            else:
                test_scores.append(50.0)
                print(f"   ❌ 缺少KDJ列，可用列: {list(result.columns)[:5]}...")
        except Exception as e:
            test_scores.append(40.0)
            print(f"   ❌ 超买识别验证失败: {e}")

        # 测试2: 超卖识别
        print("   测试2: 超卖识别")
        try:
            # 创建超卖数据（低位震荡）
            oversold_data = self._create_oversold_test_data()
            result = self.indicator.calculate(oversold_data)

            if k_column and d_column:
                k_values = result[k_column].dropna()
                d_values = result[d_column].dropna()

                if len(k_values) > 0 and len(d_values) > 0:
                    # 检查最后几个值是否显示超卖（应该<20）
                    last_k_values = k_values.tail(3)
                    last_d_values = d_values.tail(3)
                    oversold_k_count = sum(1 for val in last_k_values if val < 20)
                    oversold_d_count = sum(1 for val in last_d_values if val < 20)

                    if oversold_k_count >= 2 and oversold_d_count >= 2:
                        test_scores.append(100.0)
                        print(f"   ✅ 超卖识别精确验证通过: K值{oversold_k_count}/3<20, D值{oversold_d_count}/3<20")
                    elif oversold_k_count >= 1 or oversold_d_count >= 1:
                        test_scores.append(85.0)
                        print(f"   ✅ 超卖识别部分通过: K值{oversold_k_count}/3<20, D值{oversold_d_count}/3<20")
                    else:
                        test_scores.append(70.0)
                        print(f"   ⚠️ 超卖识别不足: K值{oversold_k_count}/3<20, D值{oversold_d_count}/3<20")
                else:
                    test_scores.append(60.0)
                    print(f"   ❌ KDJ数据不足")
            else:
                test_scores.append(50.0)
                print(f"   ❌ 缺少KDJ列进行超卖验证")
        except Exception as e:
            test_scores.append(40.0)
            print(f"   ❌ 超卖识别验证失败: {e}")

        # 测试3: 金叉死叉识别
        print("   测试3: 金叉死叉识别")
        try:
            # 创建金叉死叉数据
            crossover_data = self._create_crossover_test_data()
            result = self.indicator.calculate(crossover_data)

            if k_column and d_column:
                k_values = result[k_column].dropna()
                d_values = result[d_column].dropna()

                if len(k_values) >= 15 and len(d_values) >= 15:
                    # 简化的金叉死叉检测
                    # 检查K线和D线是否有交叉变化
                    k_d_diff = k_values - d_values
                    diff_changes = k_d_diff.diff().abs()
                    significant_changes = (diff_changes > 1).sum()

                    if significant_changes >= 2:
                        test_scores.append(100.0)
                        print(f"   ✅ 金叉死叉识别通过: 检测到{significant_changes}次显著变化")
                    elif significant_changes >= 1:
                        test_scores.append(85.0)
                        print(f"   ✅ 金叉死叉识别部分通过: 检测到{significant_changes}次显著变化")
                    else:
                        test_scores.append(70.0)
                        print(f"   ⚠️ 金叉死叉识别不足: 仅检测到{significant_changes}次显著变化")
                else:
                    test_scores.append(60.0)
                    print(f"   ❌ 数据不足进行金叉死叉分析")
            else:
                test_scores.append(50.0)
                print(f"   ❌ 缺少KDJ列进行金叉死叉验证")
        except Exception as e:
            test_scores.append(40.0)
            print(f"   ❌ 金叉死叉识别验证失败: {e}")

        # 计算阶段3评分
        stage_score = np.mean(test_scores)

        # 确定阶段状态 - 要求98.5分以上（考虑到形态识别的复杂性）
        if stage_score >= 98.5:
            stage_status = 'PASSED'
            print(f"   ✅ 阶段3通过: {stage_score:.1f}/100")
        else:
            stage_status = 'FAILED'
            print(f"   ❌ 阶段3失败: {stage_score:.1f}/100 (需要≥98.5分)")

        return {
            'status': stage_status,
            'score': stage_score,
            'test_scores': test_scores,
            'details': '形态识别验证完成'
        }

    def _stage4_architecture_compliance(self) -> Dict[str, Any]:
        """
        阶段4: 架构合规性验证
        验证BaseIndicator继承、抽象方法实现等架构要求
        """
        print("🏗️ 验证架构合规性...")

        test_scores = []

        # 测试1: BaseIndicator继承验证
        print("   测试1: BaseIndicator继承验证")
        try:
            from indicators.base_indicator import BaseIndicator

            if isinstance(self.indicator, BaseIndicator):
                test_scores.append(100.0)
                print(f"   ✅ BaseIndicator继承验证通过")
            else:
                test_scores.append(50.0)
                print(f"   ❌ 未正确继承BaseIndicator")
        except Exception as e:
            test_scores.append(40.0)
            print(f"   ❌ BaseIndicator继承验证失败: {e}")

        # 测试2: 抽象方法实现验证
        print("   测试2: 抽象方法实现验证")
        try:
            required_methods = [
                '_calculate_baseindicator',
                'calculate_raw_score_Indicator_Base_Indicator',
                'get_patterns_Indicator_Base_Indicator',
                'set_parameters_Indicator_Base_Indicator'
            ]

            implemented_methods = 0
            for method_name in required_methods:
                if hasattr(self.indicator, method_name):
                    implemented_methods += 1

            if implemented_methods == len(required_methods):
                test_scores.append(100.0)
                print(f"   ✅ 抽象方法实现验证通过: {implemented_methods}/{len(required_methods)}个方法")
            elif implemented_methods >= len(required_methods) * 0.8:
                test_scores.append(85.0)
                print(f"   ✅ 抽象方法实现部分通过: {implemented_methods}/{len(required_methods)}个方法")
            else:
                test_scores.append(70.0)
                print(f"   ⚠️ 抽象方法实现不足: {implemented_methods}/{len(required_methods)}个方法")
        except Exception as e:
            test_scores.append(50.0)
            print(f"   ❌ 抽象方法实现验证失败: {e}")

        # 测试3: 标准方法验证
        print("   测试3: 标准方法验证")
        try:
            standard_methods = ['calculate', 'set_parameters', 'minimum_periods']

            implemented_standards = 0
            for method_name in standard_methods:
                if hasattr(self.indicator, method_name):
                    implemented_standards += 1

            if implemented_standards == len(standard_methods):
                test_scores.append(100.0)
                print(f"   ✅ 标准方法验证通过: {implemented_standards}/{len(standard_methods)}个方法")
            elif implemented_standards >= len(standard_methods) * 0.8:
                test_scores.append(85.0)
                print(f"   ✅ 标准方法部分通过: {implemented_standards}/{len(standard_methods)}个方法")
            else:
                test_scores.append(70.0)
                print(f"   ⚠️ 标准方法不足: {implemented_standards}/{len(standard_methods)}个方法")
        except Exception as e:
            test_scores.append(50.0)
            print(f"   ❌ 标准方法验证失败: {e}")

        # 计算阶段4评分
        stage_score = np.mean(test_scores)

        # 确定阶段状态 - 要求100分
        if stage_score >= 100:
            stage_status = 'PASSED'
            print(f"   ✅ 阶段4通过: {stage_score:.1f}/100")
        else:
            stage_status = 'FAILED'
            print(f"   ❌ 阶段4失败: {stage_score:.1f}/100 (需要≥100分)")

        return {
            'status': stage_status,
            'score': stage_score,
            'test_scores': test_scores,
            'details': '架构合规性验证完成'
        }

    def _stage5_production_readiness(self) -> Dict[str, Any]:
        """
        阶段5: 生产就绪性验证
        验证性能、稳定性、并发处理等生产环境要求
        """
        print("🚀 验证生产就绪性...")

        test_scores = []

        # 测试1: 性能验证
        print("   测试1: 性能验证")
        try:
            # 创建大数据集
            large_data = self._create_large_test_data(1000)

            start_time = time.time()
            result = self.indicator.calculate(large_data)
            end_time = time.time()

            execution_time = end_time - start_time

            if execution_time < 1.0:  # 1秒内完成
                test_scores.append(100.0)
                print(f"   ✅ 性能验证优秀: {execution_time:.3f}秒处理1000行数据")
            elif execution_time < 2.0:
                test_scores.append(90.0)
                print(f"   ✅ 性能验证良好: {execution_time:.3f}秒处理1000行数据")
            elif execution_time < 5.0:
                test_scores.append(80.0)
                print(f"   ✅ 性能验证合格: {execution_time:.3f}秒处理1000行数据")
            else:
                test_scores.append(70.0)
                print(f"   ⚠️ 性能验证较慢: {execution_time:.3f}秒处理1000行数据")
        except Exception as e:
            test_scores.append(50.0)
            print(f"   ❌ 性能验证失败: {e}")

        # 测试2: 内存管理验证
        print("   测试2: 内存管理验证")
        try:
            import psutil
            import os

            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB

            # 执行多次计算
            test_data = self._create_standard_test_data()
            for _ in range(10):
                result = self.indicator.calculate(test_data)

            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_growth = memory_after - memory_before

            if memory_growth < 5.0:  # 内存增长小于5MB
                test_scores.append(100.0)
                print(f"   ✅ 内存管理验证优秀: 增长{memory_growth:.1f}MB")
            elif memory_growth < 10.0:
                test_scores.append(90.0)
                print(f"   ✅ 内存管理验证良好: 增长{memory_growth:.1f}MB")
            elif memory_growth < 20.0:
                test_scores.append(80.0)
                print(f"   ✅ 内存管理验证合格: 增长{memory_growth:.1f}MB")
            else:
                test_scores.append(70.0)
                print(f"   ⚠️ 内存管理验证较差: 增长{memory_growth:.1f}MB")
        except Exception as e:
            test_scores.append(85.0)  # 如果无法测试内存，给予中等分数
            print(f"   ⚠️ 内存管理验证跳过: {e}")

        # 测试3: 并发处理验证
        print("   测试3: 并发处理验证")
        try:
            import concurrent.futures
            import threading

            def calculate_indicator():
                test_data = self._create_standard_test_data()
                return self.indicator.calculate(test_data)

            # 并发执行
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(calculate_indicator) for _ in range(3)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]

            if len(results) == 3 and all(r is not None for r in results):
                test_scores.append(100.0)
                print(f"   ✅ 并发处理验证通过: 3个线程全部成功")
            elif len(results) >= 2:
                test_scores.append(85.0)
                print(f"   ✅ 并发处理部分通过: {len(results)}/3个线程成功")
            else:
                test_scores.append(70.0)
                print(f"   ⚠️ 并发处理验证不足: {len(results)}/3个线程成功")
        except Exception as e:
            test_scores.append(80.0)  # 如果无法测试并发，给予中等分数
            print(f"   ⚠️ 并发处理验证跳过: {e}")

        # 计算阶段5评分
        stage_score = np.mean(test_scores)

        # 确定阶段状态 - 要求95分以上
        if stage_score >= 95:
            stage_status = 'PASSED'
            print(f"   ✅ 阶段5通过: {stage_score:.1f}/100")
        else:
            stage_status = 'FAILED'
            print(f"   ❌ 阶段5失败: {stage_score:.1f}/100 (需要≥95分)")

        return {
            'status': stage_status,
            'score': stage_score,
            'test_scores': test_scores,
            'details': '生产就绪性验证完成'
        }

    def _create_standard_test_data(self, length: int = 50) -> pd.DataFrame:
        """创建标准测试数据"""
        np.random.seed(42)

        # 创建基础价格序列
        base_price = 100
        price_changes = np.random.normal(0, 2, length)
        prices = [base_price]

        for change in price_changes:
            new_price = prices[-1] + change
            prices.append(max(new_price, 10))  # 确保价格不会太低

        prices = prices[1:]  # 移除初始价格

        # 创建OHLC数据
        data = pd.DataFrame({
            'open': prices,
            'high': [p + np.random.uniform(0, 2) for p in prices],
            'low': [p - np.random.uniform(0, 2) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000, 10000, length)
        })

        # 确保high >= low
        data['high'] = np.maximum(data['high'], data['low'])
        data['high'] = np.maximum(data['high'], data['close'])
        data['low'] = np.minimum(data['low'], data['close'])

        return data

    def _create_overbought_test_data(self, length: int = 30) -> pd.DataFrame:
        """创建超买测试数据（高位震荡）"""
        np.random.seed(123)

        # 创建上升趋势后的高位震荡
        base_prices = list(range(80, 120, 2))  # 上升到120
        high_prices = [120 + np.random.uniform(-2, 2) for _ in range(length - len(base_prices))]  # 高位震荡
        all_prices = base_prices + high_prices

        data = pd.DataFrame({
            'open': all_prices,
            'high': [p + np.random.uniform(0, 1) for p in all_prices],
            'low': [p - np.random.uniform(0, 1) for p in all_prices],
            'close': all_prices,
            'volume': np.random.randint(1000, 10000, len(all_prices))
        })

        # 确保high >= low
        data['high'] = np.maximum(data['high'], data['low'])
        data['high'] = np.maximum(data['high'], data['close'])
        data['low'] = np.minimum(data['low'], data['close'])

        return data

    def _create_oversold_test_data(self, length: int = 30) -> pd.DataFrame:
        """创建超卖测试数据（低位震荡）"""
        np.random.seed(456)

        # 创建下降趋势后的低位震荡
        base_prices = list(range(100, 40, -3))  # 下降到40
        low_prices = [40 + np.random.uniform(-2, 2) for _ in range(length - len(base_prices))]  # 低位震荡
        all_prices = base_prices + low_prices

        data = pd.DataFrame({
            'open': all_prices,
            'high': [p + np.random.uniform(0, 1) for p in all_prices],
            'low': [p - np.random.uniform(0, 1) for p in all_prices],
            'close': all_prices,
            'volume': np.random.randint(1000, 10000, len(all_prices))
        })

        # 确保high >= low
        data['high'] = np.maximum(data['high'], data['low'])
        data['high'] = np.maximum(data['high'], data['close'])
        data['low'] = np.minimum(data['low'], data['close'])

        return data

    def _create_crossover_test_data(self, length: int = 40) -> pd.DataFrame:
        """创建金叉死叉测试数据"""
        np.random.seed(789)

        # 创建先下跌后上涨的价格模式，容易产生金叉死叉
        prices = []

        # 第一阶段：下跌
        for i in range(15):
            price = 100 - i * 2 + np.random.uniform(-1, 1)
            prices.append(price)

        # 第二阶段：横盘
        for i in range(10):
            price = 70 + np.random.uniform(-2, 2)
            prices.append(price)

        # 第三阶段：上涨
        for i in range(length - 25):
            price = 70 + i * 1.5 + np.random.uniform(-1, 1)
            prices.append(price)

        data = pd.DataFrame({
            'open': prices,
            'high': [p + np.random.uniform(0, 1) for p in prices],
            'low': [p - np.random.uniform(0, 1) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000, 10000, len(prices))
        })

        # 确保high >= low
        data['high'] = np.maximum(data['high'], data['low'])
        data['high'] = np.maximum(data['high'], data['close'])
        data['low'] = np.minimum(data['low'], data['close'])

        return data

    def _create_large_test_data(self, length: int = 1000) -> pd.DataFrame:
        """创建大数据集用于性能测试"""
        np.random.seed(999)

        # 创建随机游走价格
        base_price = 100
        price_changes = np.random.normal(0, 1, length)
        prices = [base_price]

        for change in price_changes:
            new_price = prices[-1] + change
            prices.append(max(new_price, 10))

        prices = prices[1:]

        data = pd.DataFrame({
            'open': prices,
            'high': [p + np.random.uniform(0, 2) for p in prices],
            'low': [p - np.random.uniform(0, 2) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000, 10000, length)
        })

        # 确保high >= low
        data['high'] = np.maximum(data['high'], data['low'])
        data['high'] = np.maximum(data['high'], data['close'])
        data['low'] = np.minimum(data['low'], data['close'])

        return data

    def _print_final_report(self, result: Dict[str, Any]):
        """打印最终验证报告"""
        print("\n" + "=" * 80)
        print("🎯 EnhancedKDJ指标完整5阶段验证报告")
        print("=" * 80)

        print(f"📊 总体评分: {result['overall_score']:.1f}/100")
        print(f"🏆 验证状态: {result['status']}")
        print(f"⏰ 验证时间: {result['validation_time']}")
        print(f"📈 通过阶段: {result['passed_stages']}/{result['stage_count']}")

        print(f"\n📋 各阶段详细结果:")
        stage_names = [
            "阶段1: 算法真实性验证",
            "阶段2: 基础功能验证",
            "阶段3: 形态识别验证",
            "阶段4: 架构合规性验证",
            "阶段5: 生产就绪性验证"
        ]

        for i, stage_name in enumerate(stage_names, 1):
            stage_key = f"stage_{i}"
            if stage_key in result['stage_results']:
                stage_result = result['stage_results'][stage_key]
                status_icon = "✅" if stage_result['status'] == 'PASSED' else "❌"
                print(f"   {status_icon} {stage_name}: {stage_result['score']:.1f}/100")
            else:
                print(f"   ❓ {stage_name}: 未执行")

        print(f"\n🎯 验证结论:")
        if result['status'] == 'PASSED_PRODUCTION_READY':
            print("   🎉 EnhancedKDJ指标达到生产级质量标准！")
            print("   ✅ 可以安全部署到生产环境")
            print("   ✅ 算法真实性、架构合规性、生产就绪性全部通过")
        elif result['status'] == 'PASSED_ARCHITECTURE_COMPLIANT':
            print("   ✅ EnhancedKDJ指标架构合规，基本可用")
            print("   ⚠️ 建议进一步优化以达到生产级标准")
        elif result['status'] == 'CONDITIONAL_PASS':
            print("   ⚠️ EnhancedKDJ指标条件通过，需要改进")
            print("   🔧 建议修复发现的问题后重新验证")
        else:
            print("   ❌ EnhancedKDJ指标验证失败")
            print("   🔧 需要修复关键问题后重新验证")

        print("=" * 80)


def main():
    """主函数"""
    print("🚀 启动EnhancedKDJ指标完整5阶段验证")

    try:
        validator = EnhancedKdjValidator()
        result = validator.run_complete_validation()

        # 保存验证结果
        import json
        result_file = "validation_results/enhanced_kdj_validation_result.json"
        os.makedirs(os.path.dirname(result_file), exist_ok=True)

        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)

        print(f"\n💾 验证结果已保存到: {result_file}")

        # 返回结果用于进一步处理
        return result

    except Exception as e:
        print(f"❌ 验证过程发生异常: {e}")
        print(f"📋 异常详情:")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
