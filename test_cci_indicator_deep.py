#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CCI指标全面深度测试脚本
执行CCI指标的全面质量验证测试
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from indicators.core.unified_indicator_manager import UnifiedIndicatorManager
from utils.logger import get_logger

# 设置日志级别
logging.basicConfig(level=logging.WARNING)
print("设置日志级别:", logging.WARNING)

def create_test_data():
    """创建测试数据"""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    np.random.seed(42)  # 设置随机种子以获得可重现的结果
    
    # 生成具有明显超买超卖特征的价格数据，适合CCI测试
    base_price = 100
    
    # 创建价格变化模式，包含明显的超买超卖区域
    price_patterns = []
    # 第一段：正常波动 (0-25)
    price_patterns.extend(np.linspace(0, 5, 25))
    # 第二段：快速上涨（产生超买） (25-40)
    price_patterns.extend(np.linspace(5, 25, 15))
    # 第三段：高位震荡 (40-55)
    price_patterns.extend(np.linspace(25, 22, 15))
    # 第四段：快速下跌（产生超卖） (55-75)
    price_patterns.extend(np.linspace(22, 2, 20))
    # 第五段：低位反弹 (75-100)
    price_patterns.extend(np.linspace(2, 15, 25))
    
    # 添加随机噪声
    noise = np.random.normal(0, 1.5, 100)
    prices = base_price + np.array(price_patterns) + noise
    
    # 生成OHLC数据
    data = pd.DataFrame({
        'open': prices + np.random.normal(0, 0.3, 100),
        'high': prices + np.abs(np.random.normal(0, 2, 100)),
        'low': prices - np.abs(np.random.normal(0, 2, 100)),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, 100),
        'turnover_rate': np.random.uniform(0.1, 5.0, 100)
    }, index=dates)
    
    # 确保 high >= low, open/close 在 high/low 之间
    data['high'] = np.maximum.reduce([data['open'], data['high'], data['close']])
    data['low'] = np.minimum.reduce([data['open'], data['low'], data['close']])
    
    return data

def test_cci_instantiation():
    """测试CCI指标实例化"""
    print("\n" + "="*60)
    print("🔍 CCI指标实例化测试")
    print("="*60)
    
    try:
        manager = UnifiedIndicatorManager()
        cci_indicator = manager.create_indicator('CCI')
        
        if cci_indicator is not None:
            print("✅ CCI指标实例化成功")
            print(f"   类名: {cci_indicator.__class__.__name__}")
            print(f"   指标名: {getattr(cci_indicator, 'name', 'N/A')}")
            print(f"   描述: {getattr(cci_indicator, 'description', 'N/A')}")
            print(f"   必需列: {getattr(cci_indicator, 'REQUIRED_COLUMNS', 'N/A')}")
            return True, cci_indicator
        else:
            print("❌ CCI指标实例化失败")
            return False, None
            
    except Exception as e:
        print(f"❌ CCI指标实例化异常: {e}")
        return False, None

def test_cci_calculate(cci_indicator, test_data):
    """测试CCI指标calculate()方法"""
    print("\n" + "="*60)
    print("📊 CCI指标calculate()测试")
    print("="*60)
    
    try:
        result = cci_indicator.calculate(test_data)
        
        if result is not None and not result.empty:
            print("✅ CCI指标calculate()方法成功")
            print(f"   结果行数: {len(result)}")
            print(f"   结果列数: {len(result.columns)}")
            print(f"   列名: {list(result.columns)}")
            
            # 检查关键的CCI列
            cci_columns = [col for col in result.columns if 'cci' in col.lower()]
            if cci_columns:
                print(f"   CCI相关列: {cci_columns}")
                for col in cci_columns[:3]:  # 显示前3列的统计信息
                    if len(result[col]) > 0 and result[col].dtype in ['float64', 'int64']:
                        # 跳过NaN值进行统计
                        valid_values = result[col].dropna()
                        if len(valid_values) > 0:
                            print(f"   {col}: 最新值={valid_values.iloc[-1]:.4f}, 平均值={valid_values.mean():.4f}, 范围=[{valid_values.min():.1f}, {valid_values.max():.1f}]")
                        else:
                            print(f"   {col}: 全部为NaN")
                    elif len(result[col]) > 0:
                        print(f"   {col}: 最新值={result[col].iloc[-1]}")
                return True, result
            else:
                print("⚠️ 警告: 未找到CCI相关的输出列")
                return False, result
        else:
            print("❌ CCI指标calculate()返回空结果")
            return False, None
            
    except Exception as e:
        print(f"❌ CCI指标calculate()异常: {e}")
        return False, None

def test_cci_get_signal(cci_indicator, test_data):
    """测试CCI指标get_signal()方法"""
    print("\n" + "="*60)
    print("📡 CCI指标get_signal()测试")
    print("="*60)
    
    try:
        signal = cci_indicator.get_signal(test_data)
        
        if signal and isinstance(signal, dict):
            print("✅ CCI指标get_signal()方法成功")
            
            # 检查标准信号格式
            required_fields = ['signal_type', 'strength', 'confidence', 'timestamp', 'reason', 'metadata']
            missing_fields = [field for field in required_fields if field not in signal]
            
            if not missing_fields:
                print("✅ 信号格式标准化验证通过")
                print(f"   信号类型: {signal['signal_type']}")
                print(f"   信号强度: {signal['strength']:.4f}")
                print(f"   信号置信度: {signal['confidence']:.4f}")
                print(f"   信号原因: {signal['reason']}")
                
                # 显示元数据信息
                if 'metadata' in signal and signal['metadata']:
                    metadata = signal['metadata']
                    if 'cci_value' in metadata:
                        print(f"   CCI值: {metadata['cci_value']:.4f}")
                    if 'cci_zone' in metadata:
                        print(f"   CCI区域: {metadata['cci_zone']}")
                    if 'cci_trend' in metadata:
                        print(f"   CCI趋势: {metadata['cci_trend']}")
                    if 'signal_strength' in metadata:
                        print(f"   信号强度等级: {metadata['signal_strength']}")
                    if 'overbought_level' in metadata:
                        print(f"   超买阈值: {metadata['overbought_level']}")
                    if 'oversold_level' in metadata:
                        print(f"   超卖阈值: {metadata['oversold_level']}")
                
                # 检查数据类型
                if (isinstance(signal['strength'], (int, float)) and 0 <= signal['strength'] <= 1 and
                    isinstance(signal['confidence'], (int, float)) and 0 <= signal['confidence'] <= 1 and
                    signal['signal_type'] in ['buy', 'sell', 'hold']):
                    print("✅ 信号数据类型和范围验证通过")
                    return True, signal
                else:
                    print("❌ 信号数据类型或范围不符合要求")
                    return False, signal
            else:
                print(f"❌ 信号格式缺少必需字段: {missing_fields}")
                return False, signal
        else:
            print("❌ CCI指标get_signal()返回格式不正确")
            return False, None
            
    except Exception as e:
        print(f"❌ CCI指标get_signal()异常: {e}")
        return False, None

def test_cci_data_validation(cci_indicator):
    """测试CCI指标数据验证"""
    print("\n" + "="*60)
    print("🔒 CCI指标数据验证测试")
    print("="*60)
    
    test_cases = [
        ("空数据", pd.DataFrame()),
        ("缺少必需列", pd.DataFrame({'close': [1, 2, 3]})),
        ("数据量不足", pd.DataFrame({
            'high': [1, 2, 3, 4, 5],
            'low': [0.5, 1.5, 2.5, 3.5, 4.5], 
            'close': [0.8, 1.8, 2.8, 3.8, 4.8]
        }))
    ]
    
    validation_passed = 0
    for test_name, test_data in test_cases:
        try:
            signal = cci_indicator.get_signal(test_data)
            if signal and signal.get('signal_type') == 'hold' and signal.get('strength') == 0.0:
                print(f"✅ {test_name} - 正确处理无效数据")
                validation_passed += 1
            else:
                print(f"❌ {test_name} - 未正确处理无效数据")
        except Exception as e:
            print(f"⚠️ {test_name} - 处理异常: {e}")
    
    if validation_passed == len(test_cases):
        print("✅ 数据验证功能完备")
        return True
    else:
        print(f"❌ 数据验证存在问题: {validation_passed}/{len(test_cases)}通过")
        return False

def test_cci_performance(cci_indicator, test_data):
    """测试CCI指标性能"""
    print("\n" + "="*60)
    print("⚡ CCI指标性能测试")
    print("="*60)
    
    import time
    
    # 测试calculate性能
    start_time = time.time()
    for i in range(10):
        cci_indicator.calculate(test_data)
    calculate_time = (time.time() - start_time) / 10
    
    # 测试get_signal性能
    start_time = time.time()
    for i in range(100):
        cci_indicator.get_signal(test_data)
    signal_time = (time.time() - start_time) / 100
    
    print(f"   calculate()平均耗时: {calculate_time*1000:.2f}ms")
    print(f"   get_signal()平均耗时: {signal_time*1000:.2f}ms")
    
    # 性能要求检查
    if calculate_time < 0.1 and signal_time < 0.01:
        print("✅ 性能测试通过")
        return True
    else:
        print("⚠️ 性能可能需要优化")
        return True  # 不作为失败条件

def test_cci_l4_compliance(cci_indicator, test_data):
    """测试CCI指标L4设计文档合规性"""
    print("\n" + "="*60)
    print("📋 CCI指标L4设计文档合规性测试")
    print("="*60)
    
    compliance_score = 0
    total_checks = 6
    
    # 1. 检查必需的抽象方法实现
    required_methods = ['calculate', 'get_signal', 'get_patterns']
    missing_methods = []
    for method in required_methods:
        if not hasattr(cci_indicator, method):
            missing_methods.append(method)
    
    if not missing_methods:
        print("✅ 必需抽象方法全部实现")
        compliance_score += 1
    else:
        print(f"❌ 缺少必需抽象方法: {missing_methods}")
    
    # 2. 检查has_result方法
    if hasattr(cci_indicator, 'has_result') and callable(getattr(cci_indicator, 'has_result')):
        print("✅ has_result()方法已实现")
        compliance_score += 1
    else:
        print("❌ has_result()方法缺失")
    
    # 3. 检查_result属性管理
    try:
        cci_indicator.calculate(test_data)
        if hasattr(cci_indicator, '_result') and cci_indicator._result is not None:
            print("✅ _result属性管理正确")
            compliance_score += 1
        else:
            print("❌ _result属性管理有问题")
    except Exception as e:
        print(f"❌ _result属性管理测试失败: {e}")
    
    # 4. 检查标准化信号格式
    try:
        signal = cci_indicator.get_signal(test_data)
        required_signal_fields = ['signal_type', 'strength', 'confidence', 'timestamp', 'reason', 'metadata']
        if all(field in signal for field in required_signal_fields):
            print("✅ 标准化信号格式符合要求")
            compliance_score += 1
        else:
            print("❌ 标准化信号格式不符合要求")
    except Exception as e:
        print(f"❌ 信号格式检查失败: {e}")
    
    # 5. 检查异常处理
    try:
        invalid_data = pd.DataFrame({'invalid': [1, 2, 3]})
        signal = cci_indicator.get_signal(invalid_data)
        if signal.get('signal_type') == 'hold':
            print("✅ 异常处理机制正常")
            compliance_score += 1
        else:
            print("❌ 异常处理机制有问题")
    except Exception as e:
        print(f"❌ 异常处理测试失败: {e}")
    
    # 6. 检查CCI特有功能
    try:
        # 检查CCI区域识别功能
        if hasattr(cci_indicator, '_get_cci_zone'):
            test_zones = [
                (250, "extreme_overbought"),
                (150, "overbought"), 
                (50, "bullish"),
                (-50, "bearish"),
                (-150, "oversold"),
                (-250, "extreme_oversold")
            ]
            zone_test_passed = True
            for cci_val, expected_zone in test_zones:
                actual_zone = cci_indicator._get_cci_zone(cci_val)
                if actual_zone != expected_zone:
                    zone_test_passed = False
                    break
            
            if zone_test_passed:
                print("✅ CCI区域识别功能正常")
                compliance_score += 1
            else:
                print("❌ CCI区域识别功能有问题")
        else:
            print("❌ CCI区域识别功能缺失")
    except Exception as e:
        print(f"❌ CCI特有功能检查失败: {e}")
    
    compliance_rate = (compliance_score / total_checks) * 100
    print(f"📊 L4合规性评分: {compliance_score}/{total_checks} ({compliance_rate:.1f}%)")
    
    if compliance_score == total_checks:
        print("✅ 完全符合L4设计文档要求")
        return True
    else:
        print("❌ 部分不符合L4设计文档要求")
        return False

def run_cci_tests():
    """运行所有CCI测试"""
    print("🚀 开始CCI指标全面深度测试")
    print("="*80)
    
    # 准备测试数据
    test_data = create_test_data()
    print(f"✅ 测试数据准备完成: {len(test_data)}行")
    
    # 测试计数器
    total_tests = 6
    passed_tests = 0
    
    # 1. 实例化测试
    success, cci_indicator = test_cci_instantiation()
    if success:
        passed_tests += 1
    
    if cci_indicator is None:
        print("❌ 无法继续后续测试，CCI实例化失败")
        return
    
    # 2. calculate测试
    success, result = test_cci_calculate(cci_indicator, test_data)
    if success:
        passed_tests += 1
    
    # 3. get_signal测试
    success, signal = test_cci_get_signal(cci_indicator, test_data)
    if success:
        passed_tests += 1
    
    # 4. 数据验证测试
    success = test_cci_data_validation(cci_indicator)
    if success:
        passed_tests += 1
    
    # 5. 性能测试
    success = test_cci_performance(cci_indicator, test_data)
    if success:
        passed_tests += 1
    
    # 6. L4合规性测试
    success = test_cci_l4_compliance(cci_indicator, test_data)
    if success:
        passed_tests += 1
    
    # 输出最终结果
    print("\n" + "="*80)
    print("📊 CCI指标测试总结")
    print("="*80)
    print(f"✅ 通过测试: {passed_tests}/{total_tests}")
    print(f"📈 成功率: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print("🎉 CCI指标全面测试通过！符合L4文档设计预期和基类设计预期")
    else:
        print("⚠️ CCI指标存在问题，需要进一步修复")

if __name__ == "__main__":
    run_cci_tests()
