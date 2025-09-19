#!/usr/bin/env python3
"""
EMA指标深度测试脚本
确保EMA指标100%符合L4文档设计预期和基类设计预期
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any
from indicators.core.unified_indicator_manager import UnifiedIndicatorManager
from utils.logger import get_logger

logger = get_logger(__name__)

class EMAIndicatorDeepTester:
    """EMA指标深度测试器"""
    
    def __init__(self):
        self.indicator_manager = UnifiedIndicatorManager()
        self.test_data = self._generate_comprehensive_test_data()
        self.test_results = {}
    
    def _generate_comprehensive_test_data(self) -> pd.DataFrame:
        """生成全面的测试数据"""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        
        # 生成带有明确趋势的测试数据
        base_price = 100.0
        trend = np.linspace(0, 20, 100)  # 上升趋势
        noise = np.random.normal(0, 1, 100)  # 随机波动
        
        prices = base_price + trend + noise
        
        data = pd.DataFrame({
            'date': dates,
            'open': prices * 0.995,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 100)
        })
        
        return data
    
    def test_ema_instance_creation(self) -> Dict[str, Any]:
        """测试EMA指标实例创建"""
        print("🔍 测试EMA指标实例创建...")
        
        result = {
            'test_name': 'EMA实例创建测试',
            'passed': False,
            'details': {},
            'errors': []
        }
        
        try:
            # 测试通过指标管理器创建
            ema_indicator = self.indicator_manager.create_indicator('EMA')
            
            if ema_indicator is not None:
                result['details']['manager_creation'] = True
                result['details']['indicator_name'] = ema_indicator.name
                result['details']['has_calculate_method'] = hasattr(ema_indicator, 'calculate')
                result['details']['has_get_signal_method'] = hasattr(ema_indicator, 'get_signal')
                
                # 检查基类继承
                from indicators.base_indicator import BaseIndicator
                result['details']['inherits_base_indicator'] = isinstance(ema_indicator, BaseIndicator)
                
                result['passed'] = all([
                    result['details']['manager_creation'],
                    result['details']['has_calculate_method'],
                    result['details']['has_get_signal_method'],
                    result['details']['inherits_base_indicator']
                ])
            else:
                result['errors'].append("无法通过指标管理器创建EMA指标")
                
        except Exception as e:
            result['errors'].append(f"实例创建失败: {str(e)}")
        
        return result
    
    def test_ema_calculate_method(self) -> Dict[str, Any]:
        """测试EMA指标calculate()方法"""
        print("🔍 测试EMA指标calculate()方法...")
        
        result = {
            'test_name': 'EMA指标calculate()测试',
            'passed': False,
            'details': {},
            'errors': []
        }
        
        try:
            ema_indicator = self.indicator_manager.create_indicator('EMA')
            
            if ema_indicator is None:
                result['errors'].append("无法创建EMA指标实例")
                return result
            
            # 测试计算功能
            calc_result = ema_indicator.calculate(self.test_data)
            
            if calc_result is not None and not calc_result.empty:
                result['details']['calculation_success'] = True
                result['details']['result_type'] = type(calc_result).__name__
                result['details']['result_shape'] = calc_result.shape
                result['details']['columns'] = list(calc_result.columns)
                
                # 检查是否包含EMA相关列
                ema_columns = [col for col in calc_result.columns if 'EMA' in col.upper() or col.lower().startswith('ema')]
                result['details']['ema_columns_count'] = len(ema_columns)
                result['details']['ema_columns'] = ema_columns
                
                # 检查数值合理性
                if ema_columns:
                    first_ema_col = ema_columns[0]
                    ema_values = calc_result[first_ema_col].dropna()
                    if len(ema_values) > 0:
                        result['details']['ema_values_range'] = {
                            'min': float(ema_values.min()),
                            'max': float(ema_values.max()),
                            'mean': float(ema_values.mean())
                        }
                        result['details']['values_reasonable'] = all([
                            ema_values.min() > 0,  # EMA值应该为正数
                            ema_values.max() < 1000,  # 合理范围
                            not ema_values.isna().all()  # 不全是NaN
                        ])
                
                result['passed'] = all([
                    result['details']['calculation_success'],
                    result['details']['ema_columns_count'] > 0,
                    result['details'].get('values_reasonable', True)
                ])
            else:
                result['errors'].append("calculate()方法返回空结果")
                
        except Exception as e:
            result['errors'].append(f"calculate()测试失败: {str(e)}")
        
        return result
    
    def test_ema_get_signal_method(self) -> Dict[str, Any]:
        """测试EMA指标get_signal()方法"""
        print("🔍 测试EMA指标get_signal()方法...")
        
        result = {
            'test_name': 'EMA指标get_signal()测试',
            'passed': False,
            'details': {},
            'errors': []
        }
        
        try:
            ema_indicator = self.indicator_manager.create_indicator('EMA')
            
            if ema_indicator is None:
                result['errors'].append("无法创建EMA指标实例")
                return result
            
            # 测试信号生成
            signal = ema_indicator.get_signal(self.test_data)
            
            if isinstance(signal, dict):
                result['details']['signal_is_dict'] = True
                result['details']['signal_keys'] = list(signal.keys())
                
                # 检查标准字段
                required_fields = ['signal_type', 'strength', 'confidence', 'timestamp', 'reason', 'metadata']
                field_check = {}
                for field in required_fields:
                    field_check[field] = field in signal
                
                result['details']['required_fields_check'] = field_check
                result['details']['all_required_fields_present'] = all(field_check.values())
                
                # 检查字段值类型和范围
                if 'signal_type' in signal:
                    valid_signal_types = ['buy', 'sell', 'hold']
                    result['details']['signal_type_valid'] = signal['signal_type'] in valid_signal_types
                    result['details']['signal_type_value'] = signal['signal_type']
                
                if 'strength' in signal:
                    strength = signal['strength']
                    result['details']['strength_valid'] = isinstance(strength, (int, float)) and 0.0 <= strength <= 1.0
                    result['details']['strength_value'] = strength
                
                if 'confidence' in signal:
                    confidence = signal['confidence']
                    result['details']['confidence_valid'] = isinstance(confidence, (int, float)) and 0.0 <= confidence <= 1.0
                    result['details']['confidence_value'] = confidence
                
                if 'reason' in signal:
                    result['details']['reason_provided'] = bool(signal['reason'])
                    result['details']['reason_value'] = signal['reason']
                
                if 'metadata' in signal:
                    result['details']['metadata_is_dict'] = isinstance(signal['metadata'], dict)
                    result['details']['metadata_keys'] = list(signal['metadata'].keys()) if isinstance(signal['metadata'], dict) else None
                
                result['passed'] = all([
                    result['details'].get('all_required_fields_present', False),
                    result['details'].get('signal_type_valid', False),
                    result['details'].get('strength_valid', False),
                    result['details'].get('confidence_valid', False),
                    result['details'].get('metadata_is_dict', False)
                ])
            else:
                result['errors'].append(f"get_signal()返回类型错误: {type(signal)}")
                
        except Exception as e:
            result['errors'].append(f"get_signal()测试失败: {str(e)}")
        
        return result
    
    def test_ema_data_validation(self) -> Dict[str, Any]:
        """测试EMA指标数据验证"""
        print("🔍 测试EMA指标数据验证...")
        
        result = {
            'test_name': 'EMA指标数据验证测试',
            'passed': False,
            'details': {},
            'errors': []
        }
        
        try:
            ema_indicator = self.indicator_manager.create_indicator('EMA')
            
            if ema_indicator is None:
                result['errors'].append("无法创建EMA指标实例")
                return result
            
            # 测试空数据处理
            try:
                empty_data = pd.DataFrame()
                signal_empty = ema_indicator.get_signal(empty_data)
                result['details']['handles_empty_data'] = isinstance(signal_empty, dict) and signal_empty.get('signal_type') == 'hold'
            except Exception as e:
                result['details']['handles_empty_data'] = False
                result['details']['empty_data_error'] = str(e)
            
            # 测试缺少列的数据
            try:
                invalid_data = pd.DataFrame({
                    'open': [100, 101, 102],
                    'high': [105, 106, 107],
                    'low': [95, 96, 97]
                    # 故意缺少close列
                })
                signal_invalid = ema_indicator.get_signal(invalid_data)
                result['details']['handles_missing_columns'] = isinstance(signal_invalid, dict) and signal_invalid.get('signal_type') == 'hold'
            except Exception as e:
                result['details']['handles_missing_columns'] = False
                result['details']['missing_columns_error'] = str(e)
            
            # 测试数据量不足
            try:
                insufficient_data = self.test_data.head(5)  # 只有5个数据点
                signal_insufficient = ema_indicator.get_signal(insufficient_data)
                result['details']['handles_insufficient_data'] = isinstance(signal_insufficient, dict)
            except Exception as e:
                result['details']['handles_insufficient_data'] = False
                result['details']['insufficient_data_error'] = str(e)
            
            result['passed'] = all([
                result['details'].get('handles_empty_data', False),
                result['details'].get('handles_missing_columns', False),
                result['details'].get('handles_insufficient_data', False)
            ])
                
        except Exception as e:
            result['errors'].append(f"数据验证测试失败: {str(e)}")
        
        return result
    
    def test_ema_performance(self) -> Dict[str, Any]:
        """测试EMA指标性能"""
        print("🔍 测试EMA指标性能...")
        
        result = {
            'test_name': 'EMA指标性能测试',
            'passed': False,
            'details': {},
            'errors': []
        }
        
        try:
            ema_indicator = self.indicator_manager.create_indicator('EMA')
            
            if ema_indicator is None:
                result['errors'].append("无法创建EMA指标实例")
                return result
            
            import time
            
            # 测试calculate()性能
            start_time = time.time()
            calc_result = ema_indicator.calculate(self.test_data)
            calc_time = time.time() - start_time
            
            result['details']['calculate_time'] = calc_time
            result['details']['calculate_performance_ok'] = calc_time < 2.0  # 2秒阈值
            
            # 测试get_signal()性能
            start_time = time.time()
            signal = ema_indicator.get_signal(self.test_data)
            signal_time = time.time() - start_time
            
            result['details']['signal_time'] = signal_time
            result['details']['signal_performance_ok'] = signal_time < 1.0  # 1秒阈值
            
            result['passed'] = all([
                result['details']['calculate_performance_ok'],
                result['details']['signal_performance_ok']
            ])
                
        except Exception as e:
            result['errors'].append(f"性能测试失败: {str(e)}")
        
        return result
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """运行EMA指标的全面测试"""
        print("🎯 开始EMA指标全面测试")
        print("=" * 60)
        
        tests = [
            self.test_ema_instance_creation,
            self.test_ema_calculate_method,
            self.test_ema_get_signal_method,
            self.test_ema_data_validation,
            self.test_ema_performance
        ]
        
        all_results = []
        passed_count = 0
        
        for test_func in tests:
            test_result = test_func()
            all_results.append(test_result)
            
            if test_result['passed']:
                passed_count += 1
                print(f"✅ {test_result['test_name']}: 通过")
            else:
                print(f"❌ {test_result['test_name']}: 失败")
                if test_result['errors']:
                    for error in test_result['errors']:
                        print(f"   - {error}")
                # 打印详细信息用于调试
                print(f"   详细信息: {test_result['details']}")
        
        success_rate = (passed_count / len(tests)) * 100
        
        summary = {
            'total_tests': len(tests),
            'passed_tests': passed_count,
            'failed_tests': len(tests) - passed_count,
            'success_rate': success_rate,
            'overall_passed': success_rate == 100.0,
            'test_results': all_results
        }
        
        print("\n" + "=" * 60)
        print(f"🎯 EMA指标测试结果汇总:")
        print(f"   总测试数: {summary['total_tests']}")
        print(f"   通过测试: {summary['passed_tests']}")
        print(f"   失败测试: {summary['failed_tests']}")
        print(f"   成功率: {summary['success_rate']:.1f}%")
        
        if summary['overall_passed']:
            print("🎉 EMA指标100%通过所有测试，符合L4文档设计预期！")
        else:
            print("⚠️ EMA指标未能100%通过测试，需要进一步修复")
        
        return summary

def main():
    """主函数"""
    tester = EMAIndicatorDeepTester()
    results = tester.run_comprehensive_test()
    
    # 返回是否100%通过
    return results['overall_passed']

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
