#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RSI指标干扰数据测试

测试RSI指标在各种干扰数据条件下的鲁棒性和稳定性
这是验证阶段的重要补充，确保指标在真实市场环境中的可靠性
"""

import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

try:
    from indicators.rsi import RsiRsi
    from utils.technical_utils import calculate_rsi_Utils
    from validation.rsi_stage2_simulation import RSISimulationValidator
except ImportError as e:
    print(f"导入错误: {e}")

class RSIInterferenceDataTester:
    """RSI指标干扰数据测试器"""
    
    def __init__(self):
        """初始化干扰数据测试器"""
        self.tester_name = "RSI干扰数据测试器"
        self.rsi_validator = RSISimulationValidator()
        
        # 干扰测试配置
        self.interference_config = {
            'base_data_points': 100,
            'noise_levels': [0.01, 0.05, 0.1, 0.2],  # 1%, 5%, 10%, 20%噪音
            'outlier_ratios': [0.02, 0.05, 0.1],     # 2%, 5%, 10%异常值
            'missing_data_ratios': [0.05, 0.1, 0.2], # 5%, 10%, 20%缺失数据
            'test_iterations': 5,                      # 每种干扰测试5次
            'success_threshold': 0.8                   # 80%成功率阈值
        }
        
        # 干扰类型定义
        self.interference_types = {
            'NOISE': {
                'name': '随机噪音干扰',
                'description': '在正常价格数据中添加随机噪音',
                'test_function': self._test_noise_interference
            },
            'OUTLIERS': {
                'name': '异常值干扰',
                'description': '在数据中插入价格异常值（跳空、异常波动）',
                'test_function': self._test_outlier_interference
            },
            'MISSING_DATA': {
                'name': '缺失数据干扰',
                'description': '模拟数据缺失情况',
                'test_function': self._test_missing_data_interference
            },
            'EXTREME_VOLATILITY': {
                'name': '极端波动干扰',
                'description': '模拟极端市场波动条件',
                'test_function': self._test_extreme_volatility_interference
            },
            'MIXED_INTERFERENCE': {
                'name': '混合干扰',
                'description': '多种干扰因素组合测试',
                'test_function': self._test_mixed_interference
            }
        }
        
        print(f"✅ {self.tester_name}初始化完成")
        print(f"🎯 将测试{len(self.interference_types)}种干扰类型")
    
    def generate_base_data(self) -> pd.DataFrame:
        """生成基础测试数据"""
        np.random.seed(42)
        n_points = self.interference_config['base_data_points']
        
        # 生成相对稳定的价格序列
        base_price = 15.0
        prices = [base_price]
        
        for i in range(1, n_points):
            # 温和的趋势 + 小幅随机波动
            trend = 0.001 * np.sin(i / 10)  # 周期性趋势
            noise = np.random.normal(0, 0.01)  # 1%标准噪音
            new_price = prices[-1] * (1 + trend + noise)
            prices.append(new_price)
        
        dates = pd.date_range(start='2024-01-01', periods=n_points, freq='D')
        
        df = pd.DataFrame({
            'date': dates,
            'close': prices,
            'open': np.array(prices) * 0.995,
            'high': np.array(prices) * 1.01,
            'low': np.array(prices) * 0.99,
            'volume': np.random.randint(100000, 1000000, n_points)
        })
        
        return df
    
    def _test_noise_interference(self) -> Dict[str, Any]:
        """测试随机噪音干扰"""
        print(f"\n🔊 测试随机噪音干扰")
        
        noise_results = {
            'test_type': 'NOISE_INTERFERENCE',
            'noise_levels_tested': [],
            'overall_success_rate': 0.0,
            'detailed_results': {}
        }
        
        base_data = self.generate_base_data()
        total_tests = 0
        successful_tests = 0
        
        for noise_level in self.interference_config['noise_levels']:
            print(f"  📊 测试{noise_level:.1%}噪音水平")
            
            level_results = {
                'noise_level': noise_level,
                'iterations': [],
                'success_rate': 0.0
            }
            
            level_successes = 0
            
            for iteration in range(self.interference_config['test_iterations']):
                # 添加噪音
                noisy_data = base_data.copy()
                noise = np.random.normal(0, noise_level, len(base_data))
                noisy_data['close'] = noisy_data['close'] * (1 + noise)
                noisy_data['open'] = noisy_data['open'] * (1 + noise)
                noisy_data['high'] = noisy_data['high'] * (1 + noise)
                noisy_data['low'] = noisy_data['low'] * (1 + noise)
                
                # 测试RSI计算和形态检测
                success = self._test_rsi_robustness(noisy_data, f"噪音{noise_level:.1%}_迭代{iteration+1}")
                
                level_results['iterations'].append({
                    'iteration': iteration + 1,
                    'success': success,
                    'noise_applied': noise_level
                })
                
                if success:
                    level_successes += 1
                    successful_tests += 1
                
                total_tests += 1
            
            level_results['success_rate'] = level_successes / self.interference_config['test_iterations']
            noise_results['detailed_results'][f'noise_{noise_level:.3f}'] = level_results
            noise_results['noise_levels_tested'].append(noise_level)
            
            print(f"    成功率: {level_results['success_rate']:.1%}")
        
        noise_results['overall_success_rate'] = successful_tests / total_tests if total_tests > 0 else 0
        return noise_results
    
    def _test_outlier_interference(self) -> Dict[str, Any]:
        """测试异常值干扰"""
        print(f"\n📈 测试异常值干扰")
        
        outlier_results = {
            'test_type': 'OUTLIER_INTERFERENCE',
            'outlier_ratios_tested': [],
            'overall_success_rate': 0.0,
            'detailed_results': {}
        }
        
        base_data = self.generate_base_data()
        total_tests = 0
        successful_tests = 0
        
        for outlier_ratio in self.interference_config['outlier_ratios']:
            print(f"  📊 测试{outlier_ratio:.1%}异常值比例")
            
            ratio_results = {
                'outlier_ratio': outlier_ratio,
                'iterations': [],
                'success_rate': 0.0
            }
            
            ratio_successes = 0
            
            for iteration in range(self.interference_config['test_iterations']):
                # 添加异常值
                outlier_data = base_data.copy()
                n_outliers = int(len(base_data) * outlier_ratio)
                outlier_indices = np.random.choice(len(base_data), n_outliers, replace=False)
                
                for idx in outlier_indices:
                    # 随机生成跳空或异常波动
                    if np.random.random() > 0.5:
                        # 向上跳空
                        multiplier = 1 + np.random.uniform(0.1, 0.3)
                    else:
                        # 向下跳空
                        multiplier = 1 - np.random.uniform(0.1, 0.3)
                    
                    outlier_data.loc[idx, 'close'] *= multiplier
                    outlier_data.loc[idx, 'open'] *= multiplier
                    outlier_data.loc[idx, 'high'] *= multiplier
                    outlier_data.loc[idx, 'low'] *= multiplier
                
                # 测试RSI计算和形态检测
                success = self._test_rsi_robustness(outlier_data, f"异常值{outlier_ratio:.1%}_迭代{iteration+1}")
                
                ratio_results['iterations'].append({
                    'iteration': iteration + 1,
                    'success': success,
                    'outliers_added': n_outliers
                })
                
                if success:
                    ratio_successes += 1
                    successful_tests += 1
                
                total_tests += 1
            
            ratio_results['success_rate'] = ratio_successes / self.interference_config['test_iterations']
            outlier_results['detailed_results'][f'outlier_{outlier_ratio:.3f}'] = ratio_results
            outlier_results['outlier_ratios_tested'].append(outlier_ratio)
            
            print(f"    成功率: {ratio_results['success_rate']:.1%}")
        
        outlier_results['overall_success_rate'] = successful_tests / total_tests if total_tests > 0 else 0
        return outlier_results
    
    def _test_missing_data_interference(self) -> Dict[str, Any]:
        """测试缺失数据干扰"""
        print(f"\n❓ 测试缺失数据干扰")
        
        missing_results = {
            'test_type': 'MISSING_DATA_INTERFERENCE',
            'missing_ratios_tested': [],
            'overall_success_rate': 0.0,
            'detailed_results': {}
        }
        
        base_data = self.generate_base_data()
        total_tests = 0
        successful_tests = 0
        
        for missing_ratio in self.interference_config['missing_data_ratios']:
            print(f"  📊 测试{missing_ratio:.1%}数据缺失比例")
            
            ratio_results = {
                'missing_ratio': missing_ratio,
                'iterations': [],
                'success_rate': 0.0
            }
            
            ratio_successes = 0
            
            for iteration in range(self.interference_config['test_iterations']):
                # 创建缺失数据
                missing_data = base_data.copy()
                n_missing = int(len(base_data) * missing_ratio)
                missing_indices = np.random.choice(len(base_data), n_missing, replace=False)
                
                # 随机删除一些行
                missing_data = missing_data.drop(missing_indices).reset_index(drop=True)
                
                # 测试RSI计算和形态检测
                success = self._test_rsi_robustness(missing_data, f"缺失{missing_ratio:.1%}_迭代{iteration+1}")
                
                ratio_results['iterations'].append({
                    'iteration': iteration + 1,
                    'success': success,
                    'data_points_removed': n_missing,
                    'remaining_points': len(missing_data)
                })
                
                if success:
                    ratio_successes += 1
                    successful_tests += 1
                
                total_tests += 1
            
            ratio_results['success_rate'] = ratio_successes / self.interference_config['test_iterations']
            missing_results['detailed_results'][f'missing_{missing_ratio:.3f}'] = ratio_results
            missing_results['missing_ratios_tested'].append(missing_ratio)
            
            print(f"    成功率: {ratio_results['success_rate']:.1%}")
        
        missing_results['overall_success_rate'] = successful_tests / total_tests if total_tests > 0 else 0
        return missing_results
    
    def _test_extreme_volatility_interference(self) -> Dict[str, Any]:
        """测试极端波动干扰"""
        print(f"\n⚡ 测试极端波动干扰")
        
        volatility_results = {
            'test_type': 'EXTREME_VOLATILITY_INTERFERENCE',
            'volatility_scenarios': [],
            'overall_success_rate': 0.0,
            'detailed_results': {}
        }
        
        # 定义极端波动场景
        volatility_scenarios = [
            {'name': '高频震荡', 'volatility': 0.05, 'frequency': 0.5},
            {'name': '极端暴跌', 'volatility': 0.15, 'frequency': 0.1},
            {'name': '极端暴涨', 'volatility': 0.15, 'frequency': 0.1},
            {'name': '混合极端', 'volatility': 0.1, 'frequency': 0.3}
        ]
        
        total_tests = 0
        successful_tests = 0
        
        for scenario in volatility_scenarios:
            print(f"  📊 测试{scenario['name']}场景")
            
            scenario_results = {
                'scenario_name': scenario['name'],
                'iterations': [],
                'success_rate': 0.0
            }
            
            scenario_successes = 0
            
            for iteration in range(self.interference_config['test_iterations']):
                # 生成极端波动数据
                extreme_data = self._generate_extreme_volatility_data(scenario)
                
                # 测试RSI计算和形态检测
                success = self._test_rsi_robustness(extreme_data, f"{scenario['name']}_迭代{iteration+1}")
                
                scenario_results['iterations'].append({
                    'iteration': iteration + 1,
                    'success': success,
                    'scenario': scenario['name']
                })
                
                if success:
                    scenario_successes += 1
                    successful_tests += 1
                
                total_tests += 1
            
            scenario_results['success_rate'] = scenario_successes / self.interference_config['test_iterations']
            volatility_results['detailed_results'][scenario['name']] = scenario_results
            volatility_results['volatility_scenarios'].append(scenario['name'])
            
            print(f"    成功率: {scenario_results['success_rate']:.1%}")
        
        volatility_results['overall_success_rate'] = successful_tests / total_tests if total_tests > 0 else 0
        return volatility_results
    
    def _test_mixed_interference(self) -> Dict[str, Any]:
        """测试混合干扰"""
        print(f"\n🌪️ 测试混合干扰")
        
        mixed_results = {
            'test_type': 'MIXED_INTERFERENCE',
            'combinations_tested': [],
            'overall_success_rate': 0.0,
            'detailed_results': {}
        }
        
        # 定义混合干扰组合
        interference_combinations = [
            {'name': '轻度混合', 'noise': 0.02, 'outliers': 0.02, 'missing': 0.05},
            {'name': '中度混合', 'noise': 0.05, 'outliers': 0.05, 'missing': 0.1},
            {'name': '重度混合', 'noise': 0.1, 'outliers': 0.1, 'missing': 0.2}
        ]
        
        base_data = self.generate_base_data()
        total_tests = 0
        successful_tests = 0
        
        for combination in interference_combinations:
            print(f"  📊 测试{combination['name']}组合")
            
            combo_results = {
                'combination_name': combination['name'],
                'iterations': [],
                'success_rate': 0.0
            }
            
            combo_successes = 0
            
            for iteration in range(self.interference_config['test_iterations']):
                # 应用混合干扰
                mixed_data = base_data.copy()
                
                # 1. 添加噪音
                noise = np.random.normal(0, combination['noise'], len(mixed_data))
                mixed_data['close'] = mixed_data['close'] * (1 + noise)
                
                # 2. 添加异常值
                n_outliers = int(len(mixed_data) * combination['outliers'])
                if n_outliers > 0:
                    outlier_indices = np.random.choice(len(mixed_data), n_outliers, replace=False)
                    for idx in outlier_indices:
                        multiplier = 1 + np.random.uniform(-0.2, 0.2)
                        mixed_data.loc[idx, 'close'] *= multiplier
                
                # 3. 删除部分数据
                n_missing = int(len(mixed_data) * combination['missing'])
                if n_missing > 0:
                    missing_indices = np.random.choice(len(mixed_data), n_missing, replace=False)
                    mixed_data = mixed_data.drop(missing_indices).reset_index(drop=True)
                
                # 测试RSI计算和形态检测
                success = self._test_rsi_robustness(mixed_data, f"{combination['name']}_迭代{iteration+1}")
                
                combo_results['iterations'].append({
                    'iteration': iteration + 1,
                    'success': success,
                    'combination': combination['name']
                })
                
                if success:
                    combo_successes += 1
                    successful_tests += 1
                
                total_tests += 1
            
            combo_results['success_rate'] = combo_successes / self.interference_config['test_iterations']
            mixed_results['detailed_results'][combination['name']] = combo_results
            mixed_results['combinations_tested'].append(combination['name'])
            
            print(f"    成功率: {combo_results['success_rate']:.1%}")
        
        mixed_results['overall_success_rate'] = successful_tests / total_tests if total_tests > 0 else 0
        return mixed_results
    
    def _generate_extreme_volatility_data(self, scenario: Dict) -> pd.DataFrame:
        """生成极端波动数据"""
        np.random.seed(42)
        n_points = self.interference_config['base_data_points']
        base_price = 15.0
        prices = [base_price]
        
        for i in range(1, n_points):
            if np.random.random() < scenario['frequency']:
                # 极端波动
                if scenario['name'] == '极端暴跌':
                    trend = -scenario['volatility']
                elif scenario['name'] == '极端暴涨':
                    trend = scenario['volatility']
                else:
                    trend = np.random.uniform(-scenario['volatility'], scenario['volatility'])
            else:
                # 正常波动
                trend = np.random.normal(0, 0.01)
            
            new_price = prices[-1] * (1 + trend)
            prices.append(max(new_price, 0.1))  # 防止负价格
        
        dates = pd.date_range(start='2024-01-01', periods=n_points, freq='D')
        
        df = pd.DataFrame({
            'date': dates,
            'close': prices,
            'open': np.array(prices) * 0.995,
            'high': np.array(prices) * 1.01,
            'low': np.array(prices) * 0.99,
            'volume': np.random.randint(100000, 1000000, n_points)
        })
        
        return df
    
    def _test_rsi_robustness(self, data: pd.DataFrame, test_name: str) -> bool:
        """测试RSI在给定数据下的鲁棒性"""
        try:
            # 1. 测试RSI计算是否成功
            rsi_values = calculate_rsi_Utils(data['close'], 14)
            if rsi_values is None or rsi_values.empty:
                return False
            
            # 2. 检查RSI值是否在合理范围内
            if rsi_values.min() < 0 or rsi_values.max() > 100:
                return False
            
            # 3. 检查是否有过多的NaN值
            nan_ratio = rsi_values.isna().sum() / len(rsi_values)
            if nan_ratio > 0.3:  # 超过30%的NaN值认为失败
                return False
            
            # 4. 测试形态检测是否正常工作
            try:
                # 测试基本形态检测
                overbought = (rsi_values > 70).any()
                oversold = (rsi_values < 30).any()
                
                # 测试金叉死叉检测
                golden_cross = self.rsi_validator._detect_rsi_golden_cross(rsi_values)
                death_cross = self.rsi_validator._detect_rsi_death_cross(rsi_values)
                
                # 只要能正常执行检测就认为成功
                return True
                
            except Exception as e:
                print(f"    ⚠️ {test_name}形态检测失败: {e}")
                return False
        
        except Exception as e:
            print(f"    ❌ {test_name}RSI计算失败: {e}")
            return False
    
    def run_complete_interference_test(self) -> Dict[str, Any]:
        """运行完整的干扰数据测试"""
        print(f"\n🎯 RSI指标干扰数据鲁棒性测试")
        print("测试RSI指标在各种干扰条件下的稳定性和可靠性")
        print("=" * 80)
        
        interference_results = {
            'test_name': 'RSI干扰数据鲁棒性测试',
            'start_time': datetime.now().isoformat(),
            'test_config': self.interference_config,
            'interference_types': list(self.interference_types.keys()),
            'test_results': {},
            'overall_summary': {}
        }
        
        total_success_rate = 0.0
        tests_completed = 0
        
        try:
            # 运行所有干扰测试
            for interference_type, interference_info in self.interference_types.items():
                print(f"\n🔬 {interference_info['name']}")
                print(f"描述: {interference_info['description']}")
                
                test_result = interference_info['test_function']()
                interference_results['test_results'][interference_type] = test_result
                
                success_rate = test_result.get('overall_success_rate', 0)
                total_success_rate += success_rate
                tests_completed += 1
                
                print(f"✅ {interference_info['name']}完成，总体成功率: {success_rate:.1%}")
            
            # 计算总体结果
            overall_success_rate = total_success_rate / tests_completed if tests_completed > 0 else 0
            
            interference_results['overall_summary'] = {
                'total_interference_types': tests_completed,
                'overall_success_rate': overall_success_rate,
                'robustness_grade': self._determine_robustness_grade(overall_success_rate),
                'test_status': 'COMPLETED'
            }
            
            interference_results['end_time'] = datetime.now().isoformat()
            
            print(f"\n🏆 干扰数据测试完成")
            print(f"总体成功率: {overall_success_rate:.1%}")
            print(f"鲁棒性等级: {interference_results['overall_summary']['robustness_grade']}")
            
        except Exception as e:
            interference_results['overall_summary'] = {
                'test_status': 'ERROR',
                'error': str(e)
            }
            print(f"❌ 干扰数据测试异常: {e}")
        
        # 保存结果
        self._save_interference_results(interference_results)
        
        return interference_results
    
    def _determine_robustness_grade(self, success_rate: float) -> str:
        """确定鲁棒性等级"""
        if success_rate >= 0.9:
            return 'EXCELLENT'  # 优秀
        elif success_rate >= 0.8:
            return 'GOOD'       # 良好
        elif success_rate >= 0.7:
            return 'ACCEPTABLE' # 可接受
        elif success_rate >= 0.6:
            return 'POOR'       # 较差
        else:
            return 'FAILED'     # 失败
    
    def _save_interference_results(self, results: Dict[str, Any]):
        """保存干扰测试结果"""
        results_dir = Path("validation/rsi_validation_results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"RSI干扰数据测试结果_{timestamp}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n📄 干扰测试结果已保存: {results_file}")

def main():
    """主函数"""
    print("🎯 RSI指标干扰数据鲁棒性测试")
    print("补充验证RSI指标在真实市场干扰条件下的稳定性")
    
    # 创建干扰数据测试器
    tester = RSIInterferenceDataTester()
    
    # 运行完整的干扰测试
    results = tester.run_complete_interference_test()
    
    # 显示结果摘要
    print(f"\n📊 RSI干扰数据测试结果摘要")
    print("=" * 80)
    
    if 'overall_summary' in results:
        summary = results['overall_summary']
        print(f"测试状态: {summary.get('test_status', 'UNKNOWN')}")
        print(f"干扰类型: {summary.get('total_interference_types', 0)}种")
        print(f"总体成功率: {summary.get('overall_success_rate', 0):.1%}")
        print(f"鲁棒性等级: {summary.get('robustness_grade', 'UNKNOWN')}")
    
    print(f"\n💡 干扰数据测试的重要性:")
    print(f"  • 验证指标在真实市场噪音下的稳定性")
    print(f"  • 测试异常市场条件下的可靠性")
    print(f"  • 确保生产环境的鲁棒性")
    print(f"  • 补充理想数据测试的不足")

if __name__ == "__main__":
    main()
