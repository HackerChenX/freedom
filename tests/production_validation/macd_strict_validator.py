#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MACD严格验证器 - 100%生产级标准

验证新的严格噪声抗性算法是否达到100%标准：
1. 0假阳性 + 0假阴性
2. 所有4个MACD形态完美识别
3. 每个形态至少选出1支股票
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

from indicators.macd import MacdMacd
from indicators.pattern.unified_pattern_registry import get_unified_pattern_registry
from tests.unified_indicator_testing.components.stockinfo_compatible_data_generator import StockInfoCompatibleDataGenerator
from utils.logger import get_logger

logger = get_logger(__name__)

class MACDStrictValidator:
    """MACD严格验证器 - 100%生产级标准"""
    
    def __init__(self):
        """初始化严格验证器"""
        self.macd = MacdMacd()
        self.pattern_registry = get_unified_pattern_registry()
        self.data_generator = StockInfoCompatibleDataGenerator()
        
        # 100%严格标准
        self.strict_standards = {
            'target_false_positives': 0,      # 目标假阳性：0个
            'target_false_negatives': 0,      # 目标假阴性：0个
            'required_accuracy': 1.0,         # 要求准确率：100%
            'noise_test_rounds': 10,          # 噪声测试轮数：10轮
            'pattern_test_rounds': 5          # 形态测试轮数：5轮
        }
        
        logger.info("🔥 MACD严格验证器初始化完成 - 100%生产级标准")
    
    def run_strict_validation(self) -> Dict[str, Any]:
        """运行严格验证"""
        
        print("🔥 开始MACD严格验证 - 100%生产级标准")
        print("=" * 80)
        
        validation_result = {
            'indicator_name': 'MACD',
            'validation_timestamp': datetime.now().isoformat(),
            'strict_standards': self.strict_standards,
            'pattern_validation_results': {},
            'noise_resistance_results': {},
            'overall_validation': None,
            'success': False,
            'issues_found': []
        }
        
        try:
            # 步骤1: 严格形态验证
            print("\n🎯 步骤1: 严格形态验证 (100%准确率要求)")
            pattern_results = self._strict_pattern_validation()
            validation_result['pattern_validation_results'] = pattern_results
            
            if not pattern_results['success']:
                validation_result['issues_found'].extend(pattern_results['issues'])
                print(f"❌ 形态验证失败: {pattern_results['issues']}")
                return validation_result
            
            print(f"✅ 形态验证成功: {pattern_results['success_rate']:.1%}")
            
            # 步骤2: 严格噪声抗性验证
            print("\n🛡️ 步骤2: 严格噪声抗性验证 (0假阳性要求)")
            noise_results = self._strict_noise_resistance_validation()
            validation_result['noise_resistance_results'] = noise_results
            
            if not noise_results['success']:
                validation_result['issues_found'].extend(noise_results['issues'])
                print(f"❌ 噪声抗性验证失败: {noise_results['issues']}")
                return validation_result
            
            print(f"✅ 噪声抗性验证成功: {noise_results['total_false_positives']} 假阳性")
            
            # 步骤3: 综合验证
            print("\n📊 步骤3: 综合100%标准验证")
            overall_results = self._comprehensive_validation()
            validation_result['overall_validation'] = overall_results
            
            if overall_results['success']:
                validation_result['success'] = True
                print(f"🎉 MACD指标达到100%生产级标准！")
            else:
                validation_result['issues_found'].extend(overall_results['issues'])
                print(f"❌ 综合验证失败: {overall_results['issues']}")
        
        except Exception as e:
            logger.error(f"❌ MACD严格验证异常: {e}")
            validation_result['issues_found'].append(f"验证过程异常: {str(e)}")
        
        return validation_result
    
    def _strict_pattern_validation(self) -> Dict[str, Any]:
        """严格形态验证"""
        
        result = {
            'success': False,
            'success_rate': 0.0,
            'pattern_results': {},
            'issues': []
        }
        
        try:
            macd_patterns = self.pattern_registry.get_indicator_patterns('MACD')
            
            total_patterns = len(macd_patterns)
            successful_patterns = 0
            
            for pattern in macd_patterns:
                print(f"    🔍 严格验证形态: {pattern}")
                
                pattern_result = self._validate_single_pattern_strict(pattern)
                result['pattern_results'][pattern] = pattern_result
                
                if pattern_result['perfect_recognition']:
                    successful_patterns += 1
                    print(f"      ✅ {pattern}: 完美识别")
                else:
                    print(f"      ❌ {pattern}: {pattern_result['issues']}")
                    result['issues'].extend(pattern_result['issues'])
            
            result['success_rate'] = successful_patterns / total_patterns
            result['success'] = result['success_rate'] == 1.0  # 100%要求
            
        except Exception as e:
            result['issues'].append(f"形态验证异常: {str(e)}")
        
        return result
    
    def _validate_single_pattern_strict(self, pattern_name: str) -> Dict[str, Any]:
        """严格验证单个形态"""
        
        result = {
            'perfect_recognition': False,
            'recognition_rate': 0.0,
            'false_negatives': 0,
            'issues': []
        }
        
        try:
            # 多轮测试确保稳定性
            successful_recognitions = 0
            total_tests = self.strict_standards['pattern_test_rounds']
            
            for test_round in range(total_tests):
                # 生成目标形态数据
                data_mapping = self.pattern_registry.get_pattern_mapping_for_data_generator()
                generator_pattern = data_mapping.get(pattern_name, pattern_name)
                
                test_data = self.data_generator.generate_stockinfo_compatible_data(
                    indicator_name='MACD',
                    pattern_type=generator_pattern,
                    stock_code=f'STRICT_{pattern_name}_{test_round}',
                    history_days=80  # 增加数据长度提高稳定性
                )
                
                if test_data is not None:
                    patterns_result = self.macd.get_patterns(test_data)
                    
                    if (patterns_result is not None and 
                        pattern_name in patterns_result.columns and
                        patterns_result[pattern_name].sum() > 0):
                        successful_recognitions += 1
            
            result['recognition_rate'] = successful_recognitions / total_tests
            result['false_negatives'] = total_tests - successful_recognitions
            
            # 严格标准：100%识别率
            if result['recognition_rate'] == 1.0:
                result['perfect_recognition'] = True
            else:
                result['issues'].append(f"识别率{result['recognition_rate']:.1%} < 100%")
        
        except Exception as e:
            result['issues'].append(f"形态验证异常: {str(e)}")
        
        return result
    
    def _strict_noise_resistance_validation(self) -> Dict[str, Any]:
        """严格噪声抗性验证"""
        
        result = {
            'success': False,
            'total_false_positives': 0,
            'noise_test_results': {},
            'issues': []
        }
        
        try:
            total_false_positives = 0
            noise_rounds = self.strict_standards['noise_test_rounds']
            
            for noise_round in range(noise_rounds):
                print(f"      🔍 噪声测试轮次 {noise_round + 1}/{noise_rounds}")
                
                # 生成不同类型的噪声数据
                noise_data = self._generate_advanced_noise_data(noise_round)
                noise_patterns = self.macd.get_patterns(noise_data)
                
                if noise_patterns is not None:
                    round_false_positives = noise_patterns.sum().sum()
                    total_false_positives += round_false_positives
                    
                    result['noise_test_results'][f'round_{noise_round}'] = {
                        'false_positives': round_false_positives,
                        'noise_type': self._get_noise_type_description(noise_round)
                    }
                    
                    if round_false_positives > 0:
                        print(f"        ⚠️ 轮次{noise_round + 1}: {round_false_positives}个假阳性")
                    else:
                        print(f"        ✅ 轮次{noise_round + 1}: 0个假阳性")
            
            result['total_false_positives'] = total_false_positives
            
            # 严格标准：0假阳性
            if total_false_positives == 0:
                result['success'] = True
            else:
                result['issues'].append(f"总假阳性{total_false_positives}个 > 0个目标")
        
        except Exception as e:
            result['issues'].append(f"噪声抗性验证异常: {str(e)}")
        
        return result
    
    def _generate_advanced_noise_data(self, noise_type: int) -> pd.DataFrame:
        """生成高级噪声数据"""
        
        days = 80
        base_price = 50.0
        data = []
        
        # 10种不同的噪声模式
        noise_patterns = {
            0: lambda i: np.random.normal(0, 0.01),  # 纯随机
            1: lambda i: 0.005 * np.sin(i * 0.2) + np.random.normal(0, 0.008),  # 周期性
            2: lambda i: 0.001 * i + np.random.normal(0, 0.012),  # 微弱趋势
            3: lambda i: 0.02 if i % 15 == 0 else np.random.normal(0, 0.006),  # 突发性
            4: lambda i: np.random.choice([-0.02, 0.02]) if i % 10 == 0 else np.random.normal(0, 0.005),  # 跳跃性
            5: lambda i: 0.008 * np.cos(i * 0.15) + np.random.normal(0, 0.01),  # 反向周期
            6: lambda i: np.random.normal(0, 0.015) if i < 40 else np.random.normal(0, 0.005),  # 变化波动
            7: lambda i: 0.003 * (i % 20 - 10) / 10 + np.random.normal(0, 0.008),  # 锯齿波
            8: lambda i: np.random.exponential(0.005) * np.random.choice([-1, 1]),  # 指数分布
            9: lambda i: 0.01 * np.tanh((i - 40) / 10) + np.random.normal(0, 0.01)  # S型曲线
        }
        
        noise_func = noise_patterns.get(noise_type % 10, noise_patterns[0])
        
        for i in range(days):
            price_change = noise_func(i)
            base_price *= (1 + price_change)
            
            # 确保价格合理
            base_price = max(base_price, 10.0)
            base_price = min(base_price, 200.0)
            
            data.append({
                'date': pd.Timestamp.now() - pd.Timedelta(days=days-i),
                'open': base_price * (1 + np.random.normal(0, 0.005)),
                'high': base_price * (1 + abs(np.random.normal(0, 0.01))),
                'low': base_price * (1 - abs(np.random.normal(0, 0.01))),
                'close': base_price,
                'volume': np.random.randint(1000000, 10000000)
            })
        
        return pd.DataFrame(data)
    
    def _get_noise_type_description(self, noise_type: int) -> str:
        """获取噪声类型描述"""
        descriptions = {
            0: "纯随机噪声", 1: "周期性波动", 2: "微弱趋势", 3: "突发性变化", 4: "跳跃性波动",
            5: "反向周期", 6: "变化波动", 7: "锯齿波动", 8: "指数分布", 9: "S型曲线"
        }
        return descriptions.get(noise_type % 10, "未知噪声")
    
    def _comprehensive_validation(self) -> Dict[str, Any]:
        """综合验证"""
        
        result = {
            'success': False,
            'final_accuracy': 0.0,
            'final_false_positives': 0,
            'final_false_negatives': 0,
            'issues': []
        }
        
        try:
            # 最终综合测试
            macd_patterns = self.pattern_registry.get_indicator_patterns('MACD')
            
            total_pattern_tests = 0
            successful_pattern_tests = 0
            total_false_positives = 0
            total_false_negatives = 0
            
            # 每个形态进行最终测试
            for pattern in macd_patterns:
                # 正向测试
                for test_round in range(3):
                    data_mapping = self.pattern_registry.get_pattern_mapping_for_data_generator()
                    generator_pattern = data_mapping.get(pattern, pattern)
                    
                    test_data = self.data_generator.generate_stockinfo_compatible_data(
                        indicator_name='MACD',
                        pattern_type=generator_pattern,
                        stock_code=f'FINAL_{pattern}_{test_round}',
                        history_days=80
                    )
                    
                    if test_data is not None:
                        patterns_result = self.macd.get_patterns(test_data)
                        total_pattern_tests += 1
                        
                        if (patterns_result is not None and 
                            pattern_name in patterns_result.columns and
                            patterns_result[pattern_name].sum() > 0):
                            successful_pattern_tests += 1
                        else:
                            total_false_negatives += 1
            
            # 噪声测试
            for noise_round in range(5):
                noise_data = self._generate_advanced_noise_data(noise_round)
                noise_patterns = self.macd.get_patterns(noise_data)
                
                if noise_patterns is not None:
                    total_false_positives += noise_patterns.sum().sum()
            
            # 计算最终指标
            result['final_accuracy'] = successful_pattern_tests / total_pattern_tests if total_pattern_tests > 0 else 0
            result['final_false_positives'] = total_false_positives
            result['final_false_negatives'] = total_false_negatives
            
            # 100%标准判断
            if (result['final_accuracy'] == 1.0 and 
                result['final_false_positives'] == 0 and 
                result['final_false_negatives'] == 0):
                result['success'] = True
            else:
                if result['final_accuracy'] < 1.0:
                    result['issues'].append(f"准确率{result['final_accuracy']:.1%} < 100%")
                if result['final_false_positives'] > 0:
                    result['issues'].append(f"假阳性{result['final_false_positives']}个 > 0个")
                if result['final_false_negatives'] > 0:
                    result['issues'].append(f"假阴性{result['final_false_negatives']}个 > 0个")
        
        except Exception as e:
            result['issues'].append(f"综合验证异常: {str(e)}")
        
        return result

def main():
    """主函数"""
    validator = MACDStrictValidator()
    
    # 运行严格验证
    results = validator.run_strict_validation()
    
    print("\n" + "="*80)
    print("🎯 MACD严格验证结果汇总 - 100%生产级标准")
    print("="*80)
    
    print(f"🔥 整体成功: {results['success']}")
    
    if results['pattern_validation_results']:
        pattern_val = results['pattern_validation_results']
        print(f"🎯 形态验证: {pattern_val['success']} (成功率: {pattern_val['success_rate']:.1%})")
    
    if results['noise_resistance_results']:
        noise_val = results['noise_resistance_results']
        print(f"🛡️ 噪声抗性: {noise_val['success']} (假阳性: {noise_val['total_false_positives']})")
    
    if results['overall_validation']:
        overall = results['overall_validation']
        print(f"📊 综合验证: {overall['success']}")
        print(f"   准确率: {overall['final_accuracy']:.1%}")
        print(f"   假阳性: {overall['final_false_positives']}")
        print(f"   假阴性: {overall['final_false_negatives']}")
    
    if results['issues_found']:
        print(f"\n⚠️ 发现问题:")
        for issue in results['issues_found']:
            print(f"   - {issue}")
    
    if results['success']:
        print(f"\n🎉 MACD指标达到100%生产级标准！")
        print(f"✅ 可以开始验证下一个P0指标 (RSI)")
    else:
        print(f"\n🔧 MACD指标需要进一步优化")

if __name__ == "__main__":
    main()
