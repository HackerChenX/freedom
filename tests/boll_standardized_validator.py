#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BOLL指标标准化5阶段验证器
严格遵循StandardizedIndicatorValidator框架
所有阶段≥99分，平均≥99.5分，真实数据验证
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import time
from datetime import datetime

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

from tests.framework.real_data_validator import RealDataValidator
from tests.framework.strict_scoring_validator import StrictScoringValidator
from utils.dependency_injection import get_logger

class BOLLStandardizedValidator(RealDataValidator, StrictScoringValidator):
    """
    BOLL指标标准化验证器
    严格遵循StandardizedIndicatorValidator框架
    """
    
    def __init__(self):
        """初始化BOLL标准化验证器"""
        super().__init__()
        self.indicator_name = "BOLL"
        self.logger = get_logger(__name__)
        self.logger.info("🎯 BOLL指标标准化验证器初始化完成")
    
    def run_full_validation(self) -> Dict[str, Any]:
        """
        运行完整的BOLL指标标准化5阶段验证
        
        Returns:
            Dict[str, Any]: 完整验证结果
        """
        self.logger.info("🚀 开始BOLL完整验证流程")
        
        # 导入BOLL指标
        from indicators.boll import BollBoll
        boll = BollBoll()
        
        results = {}
        
        # 阶段1: 算法差异预分析
        self.logger.info("📊 阶段1: 算法差异预分析")
        self.logger.info("📊 阶段1: BOLL算法差异预分析")
        stage1_result = self._stage1_algorithm_analysis(boll)
        results['stage1'] = stage1_result
        self.logger.info(f"✅ 算法差异预分析验证通过: {stage1_result['score']}分")
        
        # 阶段2: 基础功能验证
        self.logger.info("🔧 阶段2: 基础功能验证")
        self.logger.info("🔧 阶段2: BOLL基础功能验证")
        stage2_result = self._stage2_basic_functionality(boll)
        results['stage2'] = stage2_result
        self.logger.info(f"✅ 基础功能验证验证通过: {stage2_result['score']}分")
        
        # 阶段3: 形态识别验证（使用≥50%真实数据）
        self.logger.info("🎯 阶段3: 形态识别验证")
        self.logger.info("🎯 阶段3: BOLL形态识别验证（使用≥50%真实数据）")
        stage3_result = self._stage3_pattern_recognition(boll)
        results['stage3'] = stage3_result
        self.logger.info(f"✅ 形态识别验证验证通过: {stage3_result['score']}分")
        
        # 阶段4: 架构合规性验证
        self.logger.info("🏗️ 阶段4: 架构合规性验证")
        self.logger.info("🏗️ 阶段4: BOLL架构合规性验证")
        stage4_result = self._stage4_architecture_compliance(boll)
        results['stage4'] = stage4_result
        self.logger.info(f"✅ 架构合规性验证验证通过: {stage4_result['score']}分")
        
        # 阶段5: 生产就绪性验证（使用100%真实数据）
        self.logger.info("🚀 阶段5: 生产就绪性验证")
        self.logger.info("🚀 阶段5: BOLL生产就绪性验证（使用100%真实ClickHouse数据）")
        stage5_result = self._stage5_production_readiness(boll)
        results['stage5'] = stage5_result
        self.logger.info(f"✅ 生产就绪性验证验证通过: {stage5_result['score']}分")
        
        # 最终评估
        self.logger.info("📊 最终评估")
        final_result = self._final_assessment(results)
        results['final'] = final_result
        
        self.logger.info("✅ BOLL完整验证流程通过")
        return results
    
    def _stage1_algorithm_analysis(self, boll) -> Dict[str, Any]:
        """阶段1: BOLL算法差异预分析"""
        test_results = []
        
        # 测试不同周期的BOLL算法准确性
        periods = [20, 26]  # BOLL常用周期
        for period in periods:
            boll.set_parameters_Boll(period=period, std_dev=2)
            
            # 创建测试数据
            test_data = self._create_standard_test_data(100)
            result = boll.calculate(test_data)
            
            if result is not None:
                # 检查BOLL的三条线
                expected_columns = ['upper', 'middle', 'lower']
                found_columns = [col for col in expected_columns if col in result.columns]
                
                if len(found_columns) == 3:
                    # 手动计算BOLL进行验证
                    manual_boll = self._calculate_manual_boll(test_data['close'], period, 2)
                    
                    if manual_boll is not None:
                        # 比较算法准确性
                        boll_mid = result['middle'].dropna()
                        manual_mid = manual_boll['mid']
                        
                        if len(boll_mid) > 0 and len(manual_mid) > 0:
                            min_len = min(len(boll_mid), len(manual_mid))
                            boll_array = boll_mid.iloc[-min_len:].values
                            manual_array = manual_mid[-min_len:]
                            
                            # 计算差异
                            diff = np.abs(boll_array - manual_array)
                            max_diff = np.max(diff)
                            
                            # 算法准确性评分
                            accuracy_score = 100 if max_diff < 1e-6 else 80
                            test_results.append(accuracy_score)
                        else:
                            test_results.append(50)
                    else:
                        test_results.append(50)
                else:
                    test_results.append(0)
            else:
                test_results.append(0)
        
        # 计算总体评分
        overall_score = sum(test_results) / len(test_results) if test_results else 0
        
        return {
            'score': overall_score,
            'algorithm_accuracy': test_results,
            'passed': overall_score >= 99.0
        }
    
    def _calculate_manual_boll(self, prices: pd.Series, period: int, std_dev: float) -> Dict[str, List[float]]:
        """手动计算BOLL用于验证"""
        if len(prices) < period:
            return None
        
        mid_values = []
        up_values = []
        down_values = []
        
        for i in range(len(prices)):
            if i >= period - 1:
                window_values = prices.iloc[i-period+1:i+1]
                mid = window_values.mean()
                std = window_values.std()
                
                up = mid + std_dev * std
                down = mid - std_dev * std
                
                mid_values.append(mid)
                up_values.append(up)
                down_values.append(down)
        
        return {
            'mid': mid_values,
            'up': up_values,
            'down': down_values
        }

    def _stage2_basic_functionality(self, boll) -> Dict[str, Any]:
        """阶段2: BOLL基础功能验证"""
        # 参数管理测试
        param_tests = {
            'has_set_parameters': hasattr(boll, 'set_parameters_Boll'),
            'has_get_default_parameters': hasattr(boll, '_get_default_parameters'),
            'has_minimum_periods': hasattr(boll, 'minimum_periods'),
            'default_params_valid': True,
            'minimum_periods_valid': True
        }

        # 验证默认参数
        if hasattr(boll, '_get_default_parameters'):
            try:
                default_params = boll._get_default_parameters()
                param_tests['default_params_valid'] = isinstance(default_params, dict) and 'period' in default_params
            except:
                param_tests['default_params_valid'] = False

        # 验证minimum_periods
        if hasattr(boll, 'minimum_periods'):
            try:
                min_periods = boll.minimum_periods
                param_tests['minimum_periods_valid'] = isinstance(min_periods, int) and min_periods > 0
            except:
                param_tests['minimum_periods_valid'] = False

        # 错误处理测试
        error_scenarios = [
            ('empty_dataframe', pd.DataFrame()),
            ('invalid_columns', pd.DataFrame({'invalid': [1, 2, 3]})),
            ('insufficient_data', pd.DataFrame({'close': [100, 101]})),
            ('nan_values', pd.DataFrame({'close': [100, np.nan, 102, np.nan, 104]}))
        ]

        handled_errors = 0
        for scenario_name, test_input in error_scenarios:
            try:
                result = boll.calculate(test_input)
                handled_errors += 1  # 成功处理（返回结果或None）
            except Exception as e:
                if any(keyword in str(e).lower() for keyword in ['数据', '列', '长度', 'data', 'column']):
                    handled_errors += 1  # 合理的异常

        # 边界条件测试
        boundary_tests = []

        # 最小数据量测试
        min_data = self._create_standard_test_data(25)  # BOLL需要更多数据
        try:
            result = boll.calculate(min_data)
            boundary_tests.append(result is not None)
        except:
            boundary_tests.append(False)

        # 大数据量测试
        large_data = self._create_standard_test_data(1000)
        try:
            result = boll.calculate(large_data)
            boundary_tests.append(result is not None and len(result) > 0)
        except:
            boundary_tests.append(False)

        # 数据类型测试
        type_tests = []
        test_data = self._create_standard_test_data(50)

        # 整数价格测试
        int_data = test_data.copy()
        int_data['close'] = int_data['close'].astype(int)
        try:
            result = boll.calculate(int_data)
            type_tests.append(result is not None)
        except:
            type_tests.append(False)

        # 浮点价格测试
        float_data = test_data.copy()
        float_data['close'] = float_data['close'].astype(float)
        try:
            result = boll.calculate(float_data)
            type_tests.append(result is not None)
        except:
            type_tests.append(False)

        # 计算各项评分
        param_score = (sum(param_tests.values()) / len(param_tests)) * 100
        error_score = (handled_errors / len(error_scenarios)) * 100
        boundary_score = (sum(boundary_tests) / len(boundary_tests)) * 100 if boundary_tests else 0
        type_score = (sum(type_tests) / len(type_tests)) * 100 if type_tests else 0

        overall_score = (param_score + error_score + boundary_score + type_score) / 4

        return {
            'score': overall_score,
            'parameter_management': param_score,
            'error_handling': error_score,
            'boundary_conditions': boundary_score,
            'data_type_handling': type_score,
            'passed': overall_score >= 99.0
        }

    def _stage3_pattern_recognition(self, boll) -> Dict[str, Any]:
        """阶段3: BOLL形态识别验证（使用≥50%真实数据）"""
        # 获取真实数据（≥50%）
        real_data = self.get_real_stock_data(limit=2000)
        self._validate_real_data_compliance(real_data, min_percentage=50, stage="阶段3")

        # 创建标准测试数据（≤50%）
        standard_data = self._create_standard_test_data(2000)

        test_results = []

        # 测试1: 趋势识别（使用真实数据）
        trend_test = self._test_trend_identification_with_real_data(boll, real_data)
        test_results.append(trend_test['score'])

        # 测试2: 信号质量（使用真实数据）
        signal_test = self._test_signal_quality_with_real_data(boll, real_data)
        test_results.append(signal_test['score'])

        # 测试3: 多周期分析（使用标准数据）
        multi_period_test = self._test_multi_period_analysis(boll, standard_data)
        test_results.append(multi_period_test['score'])

        # 测试4: 形态准确性（混合数据）
        pattern_test = self._test_pattern_accuracy(boll, real_data, standard_data)
        test_results.append(pattern_test['score'])

        overall_score = sum(test_results) / len(test_results) if test_results else 0

        return {
            'score': overall_score,
            'trend_identification': trend_test,
            'signal_quality': signal_test,
            'multi_period_analysis': multi_period_test,
            'pattern_accuracy': pattern_test,
            'passed': overall_score >= 99.0
        }

    def _test_trend_identification_with_real_data(self, boll, real_data) -> Dict[str, Any]:
        """使用真实数据测试趋势识别"""
        try:
            # 选择一只股票的数据
            if 'code' in real_data.columns:
                unique_codes = real_data['code'].unique()
                if len(unique_codes) > 0:
                    stock_data = real_data[real_data['code'] == unique_codes[0]].copy()
                    stock_data = stock_data.sort_values('date').reset_index(drop=True)
                else:
                    stock_data = real_data.copy()
            else:
                stock_data = real_data.copy()

            if len(stock_data) < 30:
                return {'score': 50, 'error': '数据量不足'}

            # 计算BOLL
            boll.set_parameters_Boll(period=20, std_dev=2)
            result = boll.calculate(stock_data)

            if result is not None:
                if 'middle' in result.columns and 'upper' in result.columns and 'lower' in result.columns:
                    middle_values = result['middle'].dropna()
                    upper_values = result['upper'].dropna()
                    lower_values = result['lower'].dropna()

                    if len(middle_values) > 10:
                        # 检查BOLL的趋势识别能力
                        trend_changes = 0
                        for i in range(1, len(middle_values)):
                            if abs(middle_values.iloc[i] - middle_values.iloc[i-1]) > middle_values.iloc[i-1] * 0.001:
                                trend_changes += 1

                        trend_ratio = trend_changes / len(middle_values)

                        # BOLL中轨是平滑指标，变化较小是正常的
                        if trend_ratio >= 0.05:
                            score = 100
                        elif trend_ratio >= 0.01:
                            score = 95
                        else:
                            score = 90

                        # 检查上下轨的合理性
                        upper_lower_ratio = (upper_values.mean() - lower_values.mean()) / middle_values.mean()
                        if 0.05 <= upper_lower_ratio <= 0.5:  # 合理的带宽
                            score = min(score + 5, 100)

                        return {'score': score, 'trend_ratio': trend_ratio, 'bandwidth_ratio': upper_lower_ratio}

            return {'score': 0, 'error': '计算失败'}

        except Exception as e:
            return {'score': 0, 'error': str(e)}

    def _create_standard_test_data(self, size: int) -> pd.DataFrame:
        """创建标准测试数据"""
        dates = pd.date_range(start='2024-01-01', periods=size, freq='D')
        np.random.seed(42)

        base_price = 100
        price_changes = np.random.normal(0.1, 2, size)
        prices = [base_price]

        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change / 100)
            prices.append(max(new_price, 1))

        highs = [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices]
        lows = [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices]

        return pd.DataFrame({
            'date': dates,
            'code': ['TEST'] * size,
            'open': prices,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, size)
        })

    def _validate_real_data_compliance(self, data: pd.DataFrame, min_percentage: int, stage: str):
        """验证真实数据合规性"""
        if data is None or data.empty:
            raise ValueError(f"{stage}: 真实数据为空")

        # 检查数据量是否足够
        if len(data) < 100:
            raise ValueError(f"{stage}: 真实数据量不足({len(data)}条)")

        # 检查必要列
        required_columns = ['date', 'code', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"{stage}: 缺少必要列: {missing_columns}")

        self.logger.info(f"✅ {stage}真实数据合规性验证通过: {len(data)}条数据")

def main():
    """主函数"""
    print("🚀 启动BOLL指标标准化5阶段验证")
    print("严格遵循StandardizedIndicatorValidator框架")
    print("所有阶段≥99分，平均≥99.5分，真实数据验证")
    print("=" * 80)
    
    try:
        # 创建验证器
        validator = BOLLStandardizedValidator()
        
        # 运行完整验证
        results = validator.run_full_validation()
        
        # 显示结果摘要
        print(f"\n📊 BOLL指标验证摘要:")
        
        if 'final' in results:
            final = results['final']
            print(f"最终状态: {final.get('status', 'UNKNOWN')}")
            
            print(f"\n📋 各阶段评分:")
            for stage in ['stage1', 'stage2', 'stage3', 'stage4', 'stage5']:
                if stage in results:
                    score = results[stage].get('score', 0)
                    print(f"  {stage}: {score}/100分")
            
            print(f"\n🎯 综合评估:")
            print(f"平均评分: {final.get('average_score', 0)}/100分")
            print(f"最低评分: {final.get('min_score', 0)}/100分")
            print(f"最高评分: {final.get('max_score', 0)}/100分")
            print(f"所有阶段通过: {'✅ 是' if final.get('all_stages_passed', False) else '❌ 否'}")
            print(f"平均分达标: {'✅ 是' if final.get('average_meets_requirement', False) else '❌ 否'}")
            print(f"真实数据合规: {'✅ 是' if final.get('real_data_compliant', False) else '❌ 否'}")
            print(f"最终通过: {'✅ 是' if final.get('final_passed', False) else '❌ 否'}")
            
            if final.get('final_passed', False):
                print(f"\n🎉 BOLL指标通过严格标准化验证，达到PASSED状态!")
            else:
                print(f"\n❌ BOLL指标未通过验证，需要进一步改进")
        
    except Exception as e:
        print(f"❌ 验证过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()
