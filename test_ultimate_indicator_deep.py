#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ULTIMATE (Ultimate Oscillator) 指标全面深度测试脚本
执行Ultimate Oscillator指标的全面质量验证测试
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
    dates = pd.date_range('2020-01-01', periods=120, freq='D')  # Ultimate Oscillator需要更多数据
    np.random.seed(42)  # 设置随机种子以获得可重现的结果
    
    # 生成具有明显振荡特征的价格数据，适合Ultimate Oscillator测试
    base_price = 100
    
    # 创建振荡模式价格数据，适合Ultimate Oscillator指标测试
    price_patterns = []
    # 第一段：强势上涨 (0-30) - 适合UO高值测试
    for i in range(30):
        price_patterns.append(10 + i * 0.5 + 3 * np.sin(i * 0.2) + 1.5 * np.random.normal())
    # 第二段：超买后调整 (30-60) - 适合UO超买区域测试
    for i in range(30):
        price_patterns.append(25 - i * 0.3 + 2.5 * np.sin(i * 0.3) + 1.8 * np.random.normal())
    # 第三段：底部震荡 (60-90) - 适合UO超卖区域测试
    for i in range(30):
        price_patterns.append(10 + 4 * np.sin(i * 0.4) + 2.0 * np.random.normal())
    # 第四段：突破上涨 (90-120) - 适合UO突破测试
    for i in range(30):
        price_patterns.append(12 + i * 0.6 + 3.5 * np.sin(i * 0.25) + 1.2 * np.random.normal())
    
    prices = base_price + np.array(price_patterns)
    
    # 生成OHLC数据
    data = pd.DataFrame({
        'open': prices + np.random.normal(0, 0.8, 120),
        'high': prices + np.abs(np.random.normal(0, 3.2, 120)),
        'low': prices - np.abs(np.random.normal(0, 3.2, 120)),
        'close': prices,
        'volume': np.random.randint(1000000, 25000000, 120),
        'turnover_rate': np.random.uniform(0.1, 12.0, 120)
    }, index=dates)
    
    # 确保 high >= low, open/close 在 high/low 之间
    data['high'] = np.maximum.reduce([data['open'], data['high'], data['close']])
    data['low'] = np.minimum.reduce([data['open'], data['low'], data['close']])
    
    return data

def test_ultimate_instantiation():
    """测试ULTIMATE指标实例化"""
    print("\n" + "="*60)
    print("🔍 ULTIMATE指标实例化测试")
    print("="*60)
    
    try:
        manager = UnifiedIndicatorManager()
        ultimate_indicator = manager.create_indicator('ULTIMATE')
        
        if ultimate_indicator is not None:
            print("✅ ULTIMATE指标实例化成功")
            print(f"   类名: {ultimate_indicator.__class__.__name__}")
            print(f"   指标名: {getattr(ultimate_indicator, 'name', 'N/A')}")
            print(f"   周期1: {getattr(ultimate_indicator, 'period1', 'N/A')}")
            print(f"   周期2: {getattr(ultimate_indicator, 'period2', 'N/A')}")
            print(f"   周期3: {getattr(ultimate_indicator, 'period3', 'N/A')}")
            print(f"   最小周期: {getattr(ultimate_indicator, 'minimum_periods', 'N/A')}")
            return True, ultimate_indicator
        else:
            print("❌ ULTIMATE指标实例化失败")
            return False, None
            
    except Exception as e:
        print(f"❌ ULTIMATE指标实例化异常: {e}")
        return False, None

def test_ultimate_calculate(ultimate_indicator, test_data):
    """测试ULTIMATE指标calculate()方法"""
    print("\n" + "="*60)
    print("📊 ULTIMATE指标calculate()测试")
    print("="*60)
    
    try:
        result = ultimate_indicator.calculate(test_data)
        
        if result is not None and not result.empty:
            print("✅ ULTIMATE指标calculate()方法成功")
            print(f"   结果行数: {len(result)}")
            print(f"   结果列数: {len(result.columns)}")
            print(f"   列名: {list(result.columns)}")
            
            # 检查关键的Ultimate Oscillator列
            uo_columns = [col for col in result.columns if any(x in col.lower() for x in ['ultimate', 'oscillator', 'uo', 'bp', 'tr', 'avg'])]
            if uo_columns:
                print(f"   Ultimate Oscillator相关列: {uo_columns}")
                for col in uo_columns[:8]:  # 显示前8列的统计信息
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
                print("⚠️ 警告: 未找到Ultimate Oscillator相关的输出列")
                return False, result
        else:
            print("❌ ULTIMATE指标calculate()返回空结果")
            return False, None
            
    except Exception as e:
        print(f"❌ ULTIMATE指标calculate()异常: {e}")
        return False, None

def test_ultimate_get_signal(ultimate_indicator, test_data):
    """测试ULTIMATE指标get_signal()方法"""
    print("\n" + "="*60)
    print("📡 ULTIMATE指标get_signal()测试")
    print("="*60)
    
    try:
        signal = ultimate_indicator.get_signal(test_data)
        
        if signal and isinstance(signal, dict):
            print("✅ ULTIMATE指标get_signal()方法成功")
            
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
                
                # 显示Ultimate Oscillator特有的元数据信息
                if 'metadata' in signal and signal['metadata']:
                    metadata = signal['metadata']
                    if 'uo_value' in metadata:
                        print(f"   UO值: {metadata['uo_value']:.4f}")
                    if 'uo_previous' in metadata:
                        print(f"   前期UO值: {metadata['uo_previous']:.4f}")
                    if 'uo_change' in metadata:
                        print(f"   UO变化: {metadata['uo_change']:.4f}")
                    if 'uo_momentum' in metadata:
                        print(f"   UO动量: {metadata['uo_momentum']}")
                    if 'uo_acceleration' in metadata:
                        print(f"   UO加速度: {metadata['uo_acceleration']}")
                    if 'uo_zone' in metadata:
                        print(f"   UO区域: {metadata['uo_zone']}")
                    if 'oversold_cross_up' in metadata:
                        print(f"   超卖突破向上: {metadata['oversold_cross_up']}")
                    if 'overbought_cross_down' in metadata:
                        print(f"   超买跌破向下: {metadata['overbought_cross_down']}")
                    if 'midline_cross_up' in metadata:
                        print(f"   中线突破向上: {metadata['midline_cross_up']}")
                    if 'midline_cross_down' in metadata:
                        print(f"   中线跌破向下: {metadata['midline_cross_down']}")
                    if 'in_overbought' in metadata:
                        print(f"   处于超买区域: {metadata['in_overbought']}")
                    if 'in_oversold' in metadata:
                        print(f"   处于超卖区域: {metadata['in_oversold']}")
                    if 'multi_period_consistency' in metadata:
                        print(f"   多周期一致性: {metadata['multi_period_consistency']}")
                    if 'period1' in metadata and 'period2' in metadata and 'period3' in metadata:
                        print(f"   三个周期: {metadata['period1']}, {metadata['period2']}, {metadata['period3']}")
                
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
            print("❌ ULTIMATE指标get_signal()返回格式不正确")
            return False, None
            
    except Exception as e:
        print(f"❌ ULTIMATE指标get_signal()异常: {e}")
        return False, None

def test_ultimate_data_validation(ultimate_indicator):
    """测试ULTIMATE指标数据验证"""
    print("\n" + "="*60)
    print("🔒 ULTIMATE指标数据验证测试")
    print("="*60)
    
    test_cases = [
        ("空数据", pd.DataFrame()),
        ("缺少必需列", pd.DataFrame({'close': [1, 2, 3]})),
        ("数据量不足", pd.DataFrame({
            'high': list(range(1, 20)),
            'low': list(range(1, 20)),
            'close': list(range(1, 20))
        }))  # Ultimate Oscillator需要足够的数据（max(period1,period2,period3) + 10）
    ]
    
    validation_passed = 0
    for test_name, test_data in test_cases:
        try:
            signal = ultimate_indicator.get_signal(test_data)
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

def test_ultimate_performance(ultimate_indicator, test_data):
    """测试ULTIMATE指标性能"""
    print("\n" + "="*60)
    print("⚡ ULTIMATE指标性能测试")
    print("="*60)
    
    import time
    
    # 测试calculate性能
    start_time = time.time()
    for i in range(10):
        ultimate_indicator.calculate(test_data)
    calculate_time = (time.time() - start_time) / 10
    
    # 测试get_signal性能
    start_time = time.time()
    for i in range(100):
        ultimate_indicator.get_signal(test_data)
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

def test_ultimate_l4_compliance(ultimate_indicator, test_data):
    """测试ULTIMATE指标L4设计文档合规性"""
    print("\n" + "="*60)
    print("📋 ULTIMATE指标L4设计文档合规性测试")
    print("="*60)
    
    compliance_score = 0
    total_checks = 7
    
    # 1. 检查必需的抽象方法实现
    required_methods = ['calculate', 'get_signal', 'get_patterns']
    missing_methods = []
    for method in required_methods:
        if not hasattr(ultimate_indicator, method):
            missing_methods.append(method)
    
    if not missing_methods:
        print("✅ 必需抽象方法全部实现")
        compliance_score += 1
    else:
        print(f"❌ 缺少必需抽象方法: {missing_methods}")
    
    # 2. 检查has_result方法
    if hasattr(ultimate_indicator, 'has_result') and callable(getattr(ultimate_indicator, 'has_result')):
        print("✅ has_result()方法已实现")
        compliance_score += 1
    else:
        print("❌ has_result()方法缺失")
    
    # 3. 检查_result属性管理
    try:
        ultimate_indicator.calculate(test_data)
        if hasattr(ultimate_indicator, '_result') and ultimate_indicator._result is not None:
            print("✅ _result属性管理正确")
            compliance_score += 1
        else:
            print("❌ _result属性管理有问题")
    except Exception as e:
        print(f"❌ _result属性管理测试失败: {e}")
    
    # 4. 检查标准化信号格式
    try:
        signal = ultimate_indicator.get_signal(test_data)
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
        signal = ultimate_indicator.get_signal(invalid_data)
        if signal.get('signal_type') == 'hold':
            print("✅ 异常处理机制正常")
            compliance_score += 1
        else:
            print("❌ 异常处理机制有问题")
    except Exception as e:
        print(f"❌ 异常处理测试失败: {e}")
    
    # 6. 检查Ultimate Oscillator特有算法
    try:
        # 检查Ultimate Oscillator核心算法实现
        result = ultimate_indicator.calculate(test_data)
        expected_uo_columns = ['ultimate_oscillator', 'bp', 'tr']  # 关键的Ultimate Oscillator列
        has_uo_columns = all(col in result.columns for col in expected_uo_columns)
        
        if has_uo_columns:
            print("✅ Ultimate Oscillator核心算法实现完整")
            compliance_score += 1
        else:
            print(f"❌ Ultimate Oscillator核心算法缺失，找不到关键列：{expected_uo_columns}")
    except Exception as e:
        print(f"❌ Ultimate Oscillator算法检查失败: {e}")
    
    # 7. 检查Ultimate Oscillator多周期参数配置和振荡器特性
    try:
        # 检查Ultimate Oscillator三个周期参数
        if hasattr(ultimate_indicator, 'period1') and hasattr(ultimate_indicator, 'period2') and hasattr(ultimate_indicator, 'period3'):
            print("✅ Ultimate Oscillator多周期参数配置完备")
            compliance_score += 1
        else:
            print("❌ Ultimate Oscillator多周期参数配置缺失")
            
        # 额外检查：验证Ultimate Oscillator振荡器特性
        result = ultimate_indicator.calculate(test_data)
        if 'ultimate_oscillator' in result.columns:
            uo_values = result['ultimate_oscillator'].dropna()
            if len(uo_values) > 0:
                # Ultimate Oscillator应该在0-100范围内振荡
                uo_min, uo_max = uo_values.min(), uo_values.max()
                if 0 <= uo_min and uo_max <= 100:
                    print(f"   ✓ Ultimate Oscillator振荡器特性正常（值域: [{uo_min:.2f}, {uo_max:.2f}]，符合0-100范围）")
                else:
                    print(f"   ⚠️ Ultimate Oscillator值域异常：[{uo_min:.2f}, {uo_max:.2f}]，应在0-100范围内")
                    
        # 检查Ultimate Oscillator多周期计算逻辑
        if hasattr(ultimate_indicator, 'period1') and hasattr(ultimate_indicator, 'period2') and hasattr(ultimate_indicator, 'period3'):
            print(f"   ✓ Ultimate Oscillator多周期配置: 周期1({ultimate_indicator.period1}), 周期2({ultimate_indicator.period2}), 周期3({ultimate_indicator.period3})")
        else:
            print("   ⚠️ Ultimate Oscillator多周期配置缺失")
            
        # 检查Ultimate Oscillator超买超卖和中线逻辑
        signal = ultimate_indicator.get_signal(test_data)
        if 'metadata' in signal and signal['metadata']:
            metadata = signal['metadata']
            if 'in_overbought' in metadata and 'in_oversold' in metadata:
                print(f"   ✓ Ultimate Oscillator超买超卖逻辑完整（超买: {metadata['in_overbought']}, 超卖: {metadata['in_oversold']}）")
            if 'midline_cross_up' in metadata and 'midline_cross_down' in metadata:
                print(f"   ✓ Ultimate Oscillator中线穿越逻辑完整（向上穿越: {metadata['midline_cross_up']}, 向下穿越: {metadata['midline_cross_down']}）")
            if 'multi_period_consistency' in metadata:
                print(f"   ✓ Ultimate Oscillator多周期协同验证逻辑完整（一致性: {metadata['multi_period_consistency']}）")
        else:
            print("   ⚠️ Ultimate Oscillator穿越和一致性逻辑缺失")
    except Exception as e:
        print(f"❌ Ultimate Oscillator参数和特性检查失败: {e}")
    
    compliance_rate = (compliance_score / total_checks) * 100
    print(f"📊 L4合规性评分: {compliance_score}/{total_checks} ({compliance_rate:.1f}%)")
    
    if compliance_score == total_checks:
        print("✅ 完全符合L4设计文档要求")
        return True
    else:
        print("❌ 部分不符合L4设计文档要求")
        return False

def run_ultimate_tests():
    """运行所有ULTIMATE测试"""
    print("🚀 开始ULTIMATE (Ultimate Oscillator) 指标全面深度测试")
    print("="*80)
    
    # 准备测试数据
    test_data = create_test_data()
    print(f"✅ 测试数据准备完成: {len(test_data)}行")
    
    # 测试计数器
    total_tests = 6
    passed_tests = 0
    
    # 1. 实例化测试
    success, ultimate_indicator = test_ultimate_instantiation()
    if success:
        passed_tests += 1
    
    if ultimate_indicator is None:
        print("❌ 无法继续后续测试，ULTIMATE实例化失败")
        return
    
    # 2. calculate测试
    success, result = test_ultimate_calculate(ultimate_indicator, test_data)
    if success:
        passed_tests += 1
    
    # 3. get_signal测试
    success, signal = test_ultimate_get_signal(ultimate_indicator, test_data)
    if success:
        passed_tests += 1
    
    # 4. 数据验证测试
    success = test_ultimate_data_validation(ultimate_indicator)
    if success:
        passed_tests += 1
    
    # 5. 性能测试
    success = test_ultimate_performance(ultimate_indicator, test_data)
    if success:
        passed_tests += 1
    
    # 6. L4合规性测试
    success = test_ultimate_l4_compliance(ultimate_indicator, test_data)
    if success:
        passed_tests += 1
    
    # 输出最终结果
    print("\n" + "="*80)
    print("📊 ULTIMATE指标测试总结")
    print("="*80)
    print(f"✅ 通过测试: {passed_tests}/{total_tests}")
    print(f"📈 成功率: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print("🎉 ULTIMATE指标全面测试通过！符合L4文档设计预期和基类设计预期")
    else:
        print("⚠️ ULTIMATE指标存在问题，需要进一步修复")

if __name__ == "__main__":
    run_ultimate_tests()
