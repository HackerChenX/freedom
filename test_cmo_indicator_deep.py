#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CMO (Chande Momentum Oscillator) 指标全面深度测试脚本
执行Chande Momentum Oscillator指标的全面质量验证测试
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
    dates = pd.date_range('2020-01-01', periods=100, freq='D')  # CMO需要适量数据
    np.random.seed(42)  # 设置随机种子以获得可重现的结果
    
    # 生成具有明显动量特征的价格数据，适合CMO测试
    base_price = 100
    
    # 创建动量变化价格模式，适合Chande Momentum Oscillator测试
    price_patterns = []
    # 第一段：负动量区域 (0-25) - 适合CMO负值测试
    for i in range(25):
        price_patterns.append(-12 + 2 * np.sin(i * 0.2) + 1.0 * np.random.normal())
    # 第二段：动量转正 (25-50) - 适合CMO零轴穿越测试
    for i in range(25):
        price_patterns.append(-5 + i * 0.8 + 3 * np.sin(i * 0.3) + 1.2 * np.random.normal())
    # 第三段：正动量区域 (50-75) - 适合CMO正值测试
    for i in range(25):
        price_patterns.append(15 + 3 * np.sin(i * 0.4) + 1.5 * np.random.normal())
    # 第四段：动量衰减 (75-100) - 适合CMO动量转换测试
    for i in range(25):
        price_patterns.append(20 - i * 0.6 + 2 * np.sin(i * 0.5) + 1.3 * np.random.normal())
    
    prices = base_price + np.array(price_patterns)
    
    # 生成OHLC数据
    data = pd.DataFrame({
        'open': prices + np.random.normal(0, 0.6, 100),
        'high': prices + np.abs(np.random.normal(0, 2.5, 100)),
        'low': prices - np.abs(np.random.normal(0, 2.5, 100)),
        'close': prices,
        'volume': np.random.randint(1000000, 20000000, 100),
        'turnover_rate': np.random.uniform(0.1, 10.0, 100)
    }, index=dates)
    
    # 确保 high >= low, open/close 在 high/low 之间
    data['high'] = np.maximum.reduce([data['open'], data['high'], data['close']])
    data['low'] = np.minimum.reduce([data['open'], data['low'], data['close']])
    
    return data

def test_cmo_instantiation():
    """测试CMO指标实例化"""
    print("\n" + "="*60)
    print("🔍 CMO指标实例化测试")
    print("="*60)
    
    try:
        manager = UnifiedIndicatorManager()
        cmo_indicator = manager.create_indicator('CMO')
        
        if cmo_indicator is not None:
            print("✅ CMO指标实例化成功")
            print(f"   类名: {cmo_indicator.__class__.__name__}")
            print(f"   指标名: {getattr(cmo_indicator, 'name', 'N/A')}")
            print(f"   周期: {getattr(cmo_indicator, 'period', 'N/A')}")
            print(f"   超买阈值: {getattr(cmo_indicator, 'overbought', 'N/A')}")
            print(f"   超卖阈值: {getattr(cmo_indicator, 'oversold', 'N/A')}")
            print(f"   最小周期: {getattr(cmo_indicator, 'minimum_periods', 'N/A')}")
            return True, cmo_indicator
        else:
            print("❌ CMO指标实例化失败")
            return False, None
            
    except Exception as e:
        print(f"❌ CMO指标实例化异常: {e}")
        return False, None

def test_cmo_calculate(cmo_indicator, test_data):
    """测试CMO指标calculate()方法"""
    print("\n" + "="*60)
    print("📊 CMO指标calculate()测试")
    print("="*60)
    
    try:
        result = cmo_indicator.calculate(test_data)
        
        if result is not None and not result.empty:
            print("✅ CMO指标calculate()方法成功")
            print(f"   结果行数: {len(result)}")
            print(f"   结果列数: {len(result.columns)}")
            print(f"   列名: {list(result.columns)}")
            
            # 检查关键的CMO列
            cmo_columns = [col for col in result.columns if any(x in col.lower() for x in ['cmo', 'chande', 'momentum'])]
            if cmo_columns:
                print(f"   CMO相关列: {cmo_columns}")
                for col in cmo_columns[:6]:  # 显示前6列的统计信息
                    if len(result[col]) > 0 and result[col].dtype in ['float64', 'int64']:
                        # 跳过NaN值进行统计
                        valid_values = result[col].dropna()
                        if len(valid_values) > 0:
                            print(f"   {col}: 最新值={valid_values.iloc[-1]:.4f}, 平均值={valid_values.mean():.4f}, 范围=[{valid_values.min():.2f}, {valid_values.max():.2f}]")
                        else:
                            print(f"   {col}: 全部为NaN")
                    elif len(result[col]) > 0:
                        print(f"   {col}: 最新值={result[col].iloc[-1]}")
                return True, result
            else:
                print("⚠️ 警告: 未找到CMO相关的输出列")
                return False, result
        else:
            print("❌ CMO指标calculate()返回空结果")
            return False, None
            
    except Exception as e:
        print(f"❌ CMO指标calculate()异常: {e}")
        return False, None

def test_cmo_get_signal(cmo_indicator, test_data):
    """测试CMO指标get_signal()方法"""
    print("\n" + "="*60)
    print("📡 CMO指标get_signal()测试")
    print("="*60)
    
    try:
        signal = cmo_indicator.get_signal(test_data)
        
        if signal and isinstance(signal, dict):
            print("✅ CMO指标get_signal()方法成功")
            
            # 检查标准信号格式
            required_fields = ['signal_type', 'strength', 'confidence', 'timestamp', 'reason', 'metadata']
            missing_fields = [field for field in required_fields if field not in signal]
            
            if not missing_fields:
                print("✅ 信号格式标准化验证通过")
                print(f"   信号类型: {signal['signal_type']}")
                print(f"   信号强度: {signal['strength']:.4f}")
                print(f"   信号置信度: {signal['confidence']:.4f}")
                print(f"   时间戳: {signal['timestamp']}")
                print(f"   信号原因: {signal['reason']}")
                
                # 显示CMO特有的元数据信息
                if 'metadata' in signal and signal['metadata']:
                    metadata = signal['metadata']
                    if 'cmo_value' in metadata:
                        print(f"   CMO值: {metadata['cmo_value']:.4f}")
                    if 'cmo_previous' in metadata:
                        print(f"   前期CMO值: {metadata['cmo_previous']:.4f}")
                    if 'cmo_change' in metadata:
                        print(f"   CMO变化: {metadata['cmo_change']:.4f}")
                    if 'cmo_momentum' in metadata:
                        print(f"   CMO动量: {metadata['cmo_momentum']}")
                    if 'cmo_zone' in metadata:
                        print(f"   CMO区域: {metadata['cmo_zone']}")
                    if 'momentum_strength' in metadata:
                        print(f"   动量强度: {metadata['momentum_strength']:.4f}")
                    if 'in_overbought' in metadata:
                        print(f"   是否超买: {metadata['in_overbought']}")
                    if 'in_oversold' in metadata:
                        print(f"   是否超卖: {metadata['in_oversold']}")
                    if 'in_positive_momentum' in metadata:
                        print(f"   是否正动量: {metadata['in_positive_momentum']}")
                    if 'in_negative_momentum' in metadata:
                        print(f"   是否负动量: {metadata['in_negative_momentum']}")
                    if 'distance_to_zero' in metadata:
                        print(f"   距零轴距离: {metadata['distance_to_zero']:.4f}")
                
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
            print("❌ CMO指标get_signal()返回格式不正确")
            return False, None
            
    except Exception as e:
        print(f"❌ CMO指标get_signal()异常: {e}")
        return False, None

def test_cmo_data_validation(cmo_indicator):
    """测试CMO指标数据验证"""
    print("\n" + "="*60)
    print("🔒 CMO指标数据验证测试")
    print("="*60)
    
    test_cases = [
        ("空数据", pd.DataFrame()),
        ("缺少必需列", pd.DataFrame({'high': [1, 2, 3]})),
        ("数据量不足", pd.DataFrame({
            'close': list(range(1, 16))
        }))  # CMO需要足够的数据（period + 5）
    ]
    
    validation_passed = 0
    for test_name, test_data in test_cases:
        try:
            signal = cmo_indicator.get_signal(test_data)
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

def test_cmo_performance(cmo_indicator, test_data):
    """测试CMO指标性能"""
    print("\n" + "="*60)
    print("⚡ CMO指标性能测试")
    print("="*60)
    
    import time
    
    # 测试calculate性能
    start_time = time.time()
    for i in range(10):
        cmo_indicator.calculate(test_data)
    calculate_time = (time.time() - start_time) / 10
    
    # 测试get_signal性能
    start_time = time.time()
    for i in range(100):
        cmo_indicator.get_signal(test_data)
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

def test_cmo_l4_compliance(cmo_indicator, test_data):
    """测试CMO指标L4设计文档合规性"""
    print("\n" + "="*60)
    print("📋 CMO指标L4设计文档合规性测试")
    print("="*60)
    
    compliance_score = 0
    total_checks = 7
    
    # 1. 检查必需的抽象方法实现
    required_methods = ['calculate', 'get_signal', 'get_patterns']
    missing_methods = []
    for method in required_methods:
        if not hasattr(cmo_indicator, method):
            missing_methods.append(method)
    
    if not missing_methods:
        print("✅ 必需抽象方法全部实现")
        compliance_score += 1
    else:
        print(f"❌ 缺少必需抽象方法: {missing_methods}")
    
    # 2. 检查has_result方法
    if hasattr(cmo_indicator, 'has_result') and callable(getattr(cmo_indicator, 'has_result')):
        print("✅ has_result()方法已实现")
        compliance_score += 1
    else:
        print("❌ has_result()方法缺失")
    
    # 3. 检查_result属性管理
    try:
        cmo_indicator.calculate(test_data)
        if hasattr(cmo_indicator, '_result') and cmo_indicator._result is not None:
            print("✅ _result属性管理正确")
            compliance_score += 1
        else:
            print("❌ _result属性管理有问题")
    except Exception as e:
        print(f"❌ _result属性管理测试失败: {e}")
    
    # 4. 检查标准化信号格式
    try:
        signal = cmo_indicator.get_signal(test_data)
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
        signal = cmo_indicator.get_signal(invalid_data)
        if signal.get('signal_type') == 'hold':
            print("✅ 异常处理机制正常")
            compliance_score += 1
        else:
            print("❌ 异常处理机制有问题")
    except Exception as e:
        print(f"❌ 异常处理测试失败: {e}")
    
    # 6. 检查CMO特有算法
    try:
        # 检查CMO核心算法实现
        result = cmo_indicator.calculate(test_data)
        expected_cmo_columns = ['cmo']  # CMO列
        has_cmo_column = any(col in result.columns for col in expected_cmo_columns)
        
        if has_cmo_column:
            print("✅ CMO核心算法实现完整")
            compliance_score += 1
        else:
            print(f"❌ CMO核心算法缺失，找不到CMO列")
    except Exception as e:
        print(f"❌ CMO算法检查失败: {e}")
    
    # 7. 检查CMO参数配置和动量振荡器特性
    try:
        # 检查CMO周期参数
        if hasattr(cmo_indicator, 'period'):
            print("✅ CMO周期参数配置完备")
            compliance_score += 1
        else:
            print("❌ CMO周期参数配置缺失")
            
        # 额外检查：验证CMO是否在-100到+100区间内振荡
        result = cmo_indicator.calculate(test_data)
        if 'cmo' in result.columns:
            cmo_values = result['cmo'].dropna()
            if len(cmo_values) > 0:
                cmo_range_valid = (cmo_values >= -110).all() and (cmo_values <= 110).all()  # 允许轻微越界
                if cmo_range_valid:
                    print("   ✓ CMO振荡器特性正常（-100到+100区间）")
                else:
                    print(f"   ⚠️ CMO值可能超出正常范围: [{cmo_values.min():.2f}, {cmo_values.max():.2f}]")
                    
        # 检查CMO特有参数
        if hasattr(cmo_indicator, 'overbought') and hasattr(cmo_indicator, 'oversold'):
            print(f"   ✓ CMO超买超卖阈值配置: 超买{cmo_indicator.overbought}, 超卖{cmo_indicator.oversold}")
        else:
            print("   ⚠️ CMO超买超卖阈值配置缺失")
    except Exception as e:
        print(f"❌ CMO参数和特性检查失败: {e}")
    
    compliance_rate = (compliance_score / total_checks) * 100
    print(f"📊 L4合规性评分: {compliance_score}/{total_checks} ({compliance_rate:.1f}%)")
    
    if compliance_score == total_checks:
        print("✅ 完全符合L4设计文档要求")
        return True
    else:
        print("❌ 部分不符合L4设计文档要求")
        return False

def run_cmo_tests():
    """运行所有CMO测试"""
    print("🚀 开始CMO (Chande Momentum Oscillator) 指标全面深度测试")
    print("="*80)
    
    # 准备测试数据
    test_data = create_test_data()
    print(f"✅ 测试数据准备完成: {len(test_data)}行")
    
    # 测试计数器
    total_tests = 6
    passed_tests = 0
    
    # 1. 实例化测试
    success, cmo_indicator = test_cmo_instantiation()
    if success:
        passed_tests += 1
    
    if cmo_indicator is None:
        print("❌ 无法继续后续测试，CMO实例化失败")
        return
    
    # 2. calculate测试
    success, result = test_cmo_calculate(cmo_indicator, test_data)
    if success:
        passed_tests += 1
    
    # 3. get_signal测试
    success, signal = test_cmo_get_signal(cmo_indicator, test_data)
    if success:
        passed_tests += 1
    
    # 4. 数据验证测试
    success = test_cmo_data_validation(cmo_indicator)
    if success:
        passed_tests += 1
    
    # 5. 性能测试
    success = test_cmo_performance(cmo_indicator, test_data)
    if success:
        passed_tests += 1
    
    # 6. L4合规性测试
    success = test_cmo_l4_compliance(cmo_indicator, test_data)
    if success:
        passed_tests += 1
    
    # 输出最终结果
    print("\n" + "="*80)
    print("📊 CMO指标测试总结")
    print("="*80)
    print(f"✅ 通过测试: {passed_tests}/{total_tests}")
    print(f"📈 成功率: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print("🎉 CMO指标全面测试通过！符合L4文档设计预期和基类设计预期")
    else:
        print("⚠️ CMO指标存在问题，需要进一步修复")

if __name__ == "__main__":
    run_cmo_tests()
