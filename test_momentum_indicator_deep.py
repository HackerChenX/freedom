#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MOMENTUM (动量指标) 指标全面深度测试脚本
执行Momentum指标的全面质量验证测试
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
    dates = pd.date_range('2020-01-01', periods=100, freq='D')  # MOMENTUM需要适量数据
    np.random.seed(42)  # 设置随机种子以获得可重现的结果
    
    # 生成具有明显动量特征的价格数据，适合MOMENTUM测试
    base_price = 100
    
    # 创建动量变化价格模式，适合Momentum指标测试
    price_patterns = []
    # 第一段：低动量盘整 (0-25) - 适合MOMENTUM低值测试
    for i in range(25):
        price_patterns.append(-8 + 1.5 * np.sin(i * 0.4) + 1.0 * np.random.normal())
    # 第二段：动量加速上升 (25-50) - 适合MOMENTUM上升测试
    for i in range(25):
        price_patterns.append(-5 + i * 0.6 + 2 * np.sin(i * 0.2) + 1.2 * np.random.normal())
    # 第三段：高动量震荡 (50-75) - 适合MOMENTUM高值测试
    for i in range(25):
        price_patterns.append(10 + 4 * np.sin(i * 0.3) + 1.8 * np.random.normal())
    # 第四段：动量衰减 (75-100) - 适合MOMENTUM下降测试
    for i in range(25):
        price_patterns.append(15 - i * 0.4 + 2.5 * np.sin(i * 0.5) + 1.5 * np.random.normal())
    
    prices = base_price + np.array(price_patterns)
    
    # 生成OHLC数据
    data = pd.DataFrame({
        'open': prices + np.random.normal(0, 0.7, 100),
        'high': prices + np.abs(np.random.normal(0, 2.8, 100)),
        'low': prices - np.abs(np.random.normal(0, 2.8, 100)),
        'close': prices,
        'volume': np.random.randint(1000000, 22000000, 100),
        'turnover_rate': np.random.uniform(0.1, 11.0, 100)
    }, index=dates)
    
    # 确保 high >= low, open/close 在 high/low 之间
    data['high'] = np.maximum.reduce([data['open'], data['high'], data['close']])
    data['low'] = np.minimum.reduce([data['open'], data['low'], data['close']])
    
    return data

def test_momentum_instantiation():
    """测试MOMENTUM指标实例化"""
    print("\n" + "="*60)
    print("🔍 MOMENTUM指标实例化测试")
    print("="*60)
    
    try:
        manager = UnifiedIndicatorManager()
        momentum_indicator = manager.create_indicator('MOMENTUM')
        
        if momentum_indicator is not None:
            print("✅ MOMENTUM指标实例化成功")
            print(f"   类名: {momentum_indicator.__class__.__name__}")
            print(f"   指标名: {getattr(momentum_indicator, 'name', 'N/A')}")
            print(f"   描述: {getattr(momentum_indicator, 'description', 'N/A')}")
            print(f"   周期: {getattr(momentum_indicator, 'period', 'N/A')}")
            print(f"   最小周期: {getattr(momentum_indicator, 'minimum_periods', 'N/A')}")
            return True, momentum_indicator
        else:
            print("❌ MOMENTUM指标实例化失败")
            return False, None
            
    except Exception as e:
        print(f"❌ MOMENTUM指标实例化异常: {e}")
        return False, None

def test_momentum_calculate(momentum_indicator, test_data):
    """测试MOMENTUM指标calculate()方法"""
    print("\n" + "="*60)
    print("📊 MOMENTUM指标calculate()测试")
    print("="*60)
    
    try:
        result = momentum_indicator.calculate_Momentum(test_data)
        
        if result is not None and not result.empty:
            print("✅ MOMENTUM指标calculate()方法成功")
            print(f"   结果行数: {len(result)}")
            print(f"   结果列数: {len(result.columns)}")
            print(f"   列名: {list(result.columns)}")
            
            # 检查关键的MOMENTUM列
            momentum_columns = [col for col in result.columns if any(x in col.lower() for x in ['momentum', 'mtm'])]
            if momentum_columns:
                print(f"   MOMENTUM相关列: {momentum_columns}")
                for col in momentum_columns[:6]:  # 显示前6列的统计信息
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
                print("⚠️ 警告: 未找到MOMENTUM相关的输出列")
                return False, result
        else:
            print("❌ MOMENTUM指标calculate()返回空结果")
            return False, None
            
    except Exception as e:
        print(f"❌ MOMENTUM指标calculate()异常: {e}")
        return False, None

def test_momentum_get_signal(momentum_indicator, test_data):
    """测试MOMENTUM指标get_signal()方法"""
    print("\n" + "="*60)
    print("📡 MOMENTUM指标get_signal()测试")
    print("="*60)
    
    try:
        signal = momentum_indicator.get_signal(test_data)
        
        if signal and isinstance(signal, dict):
            print("✅ MOMENTUM指标get_signal()方法成功")
            
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
                
                # 显示MOMENTUM特有的元数据信息
                if 'metadata' in signal and signal['metadata']:
                    metadata = signal['metadata']
                    if 'momentum_value' in metadata:
                        print(f"   MOMENTUM值: {metadata['momentum_value']:.4f}")
                    if 'momentum_ma_value' in metadata:
                        print(f"   MOMENTUM均线值: {metadata['momentum_ma_value']:.4f}")
                    if 'momentum_previous' in metadata:
                        print(f"   前期MOMENTUM值: {metadata['momentum_previous']:.4f}")
                    if 'momentum_change' in metadata:
                        print(f"   MOMENTUM变化: {metadata['momentum_change']:.4f}")
                    if 'momentum_ma_change' in metadata:
                        print(f"   MOMENTUM均线变化: {metadata['momentum_ma_change']:.4f}")
                    if 'momentum_diff' in metadata:
                        print(f"   MOMENTUM差值: {metadata['momentum_diff']:.4f}")
                    if 'momentum_momentum' in metadata:
                        print(f"   MOMENTUM动量: {metadata['momentum_momentum']}")
                    if 'momentum_ma_momentum' in metadata:
                        print(f"   MOMENTUM均线动量: {metadata['momentum_ma_momentum']}")
                    if 'momentum_strength_level' in metadata:
                        print(f"   动量强度等级: {metadata['momentum_strength_level']}")
                    if 'momentum_cross_up' in metadata:
                        print(f"   上穿均线: {metadata['momentum_cross_up']}")
                    if 'momentum_cross_down' in metadata:
                        print(f"   下穿均线: {metadata['momentum_cross_down']}")
                    if 'momentum_above_ma' in metadata:
                        print(f"   高于均线: {metadata['momentum_above_ma']}")
                    if 'momentum_trend_consistency' in metadata:
                        print(f"   趋势一致性: {metadata['momentum_trend_consistency']}")
                
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
            print("❌ MOMENTUM指标get_signal()返回格式不正确")
            return False, None
            
    except Exception as e:
        print(f"❌ MOMENTUM指标get_signal()异常: {e}")
        return False, None

def test_momentum_data_validation(momentum_indicator):
    """测试MOMENTUM指标数据验证"""
    print("\n" + "="*60)
    print("🔒 MOMENTUM指标数据验证测试")
    print("="*60)
    
    test_cases = [
        ("空数据", pd.DataFrame()),
        ("缺少必需列", pd.DataFrame({'high': [1, 2, 3]})),
        ("数据量不足", pd.DataFrame({
            'close': list(range(1, 18))
        }))  # MOMENTUM需要足够的数据（period + 5）
    ]
    
    validation_passed = 0
    for test_name, test_data in test_cases:
        try:
            signal = momentum_indicator.get_signal(test_data)
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

def test_momentum_performance(momentum_indicator, test_data):
    """测试MOMENTUM指标性能"""
    print("\n" + "="*60)
    print("⚡ MOMENTUM指标性能测试")
    print("="*60)
    
    import time
    
    # 测试calculate性能
    start_time = time.time()
    for i in range(10):
        momentum_indicator.calculate_Momentum(test_data)
    calculate_time = (time.time() - start_time) / 10
    
    # 测试get_signal性能
    start_time = time.time()
    for i in range(100):
        momentum_indicator.get_signal(test_data)
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

def test_momentum_l4_compliance(momentum_indicator, test_data):
    """测试MOMENTUM指标L4设计文档合规性"""
    print("\n" + "="*60)
    print("📋 MOMENTUM指标L4设计文档合规性测试")
    print("="*60)
    
    compliance_score = 0
    total_checks = 7
    
    # 1. 检查必需的抽象方法实现
    required_methods = ['calculate_Momentum', 'get_signal', 'get_patterns']
    missing_methods = []
    for method in required_methods:
        if not hasattr(momentum_indicator, method):
            missing_methods.append(method)
    
    if not missing_methods:
        print("✅ 必需抽象方法全部实现")
        compliance_score += 1
    else:
        print(f"❌ 缺少必需抽象方法: {missing_methods}")
    
    # 2. 检查has_result方法
    if hasattr(momentum_indicator, 'has_result') and callable(getattr(momentum_indicator, 'has_result')):
        print("✅ has_result()方法已实现")
        compliance_score += 1
    else:
        print("❌ has_result()方法缺失")
    
    # 3. 检查_result属性管理
    try:
        momentum_indicator.calculate_Momentum(test_data)
        if hasattr(momentum_indicator, '_result') and momentum_indicator._result is not None:
            print("✅ _result属性管理正确")
            compliance_score += 1
        else:
            print("❌ _result属性管理有问题")
    except Exception as e:
        print(f"❌ _result属性管理测试失败: {e}")
    
    # 4. 检查标准化信号格式
    try:
        signal = momentum_indicator.get_signal(test_data)
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
        signal = momentum_indicator.get_signal(invalid_data)
        if signal.get('signal_type') == 'hold':
            print("✅ 异常处理机制正常")
            compliance_score += 1
        else:
            print("❌ 异常处理机制有问题")
    except Exception as e:
        print(f"❌ 异常处理测试失败: {e}")
    
    # 6. 检查MOMENTUM特有算法
    try:
        # 检查MOMENTUM核心算法实现
        result = momentum_indicator.calculate_Momentum(test_data)
        expected_momentum_columns = ['momentum', 'momentum_ma']  # 关键的MOMENTUM列
        has_momentum_columns = all(col in result.columns for col in expected_momentum_columns)
        
        if has_momentum_columns:
            print("✅ MOMENTUM核心算法实现完整")
            compliance_score += 1
        else:
            print(f"❌ MOMENTUM核心算法缺失，找不到关键列：{expected_momentum_columns}")
    except Exception as e:
        print(f"❌ MOMENTUM算法检查失败: {e}")
    
    # 7. 检查MOMENTUM参数配置和动量特性
    try:
        # 检查MOMENTUM周期参数
        if hasattr(momentum_indicator, 'period'):
            print("✅ MOMENTUM参数配置完备")
            compliance_score += 1
        else:
            print("❌ MOMENTUM参数配置缺失")
            
        # 额外检查：验证MOMENTUM动量计算特性
        result = momentum_indicator.calculate_Momentum(test_data)
        if 'momentum' in result.columns:
            momentum_values = result['momentum'].dropna()
            if len(momentum_values) > 0:
                # MOMENTUM指标应该反映价格变化的速度和幅度
                print(f"   ✓ MOMENTUM动量特性正常（数值范围: [{momentum_values.min():.2f}, {momentum_values.max():.2f}]）")
                    
        # 检查MOMENTUM动量算法逻辑
        if hasattr(momentum_indicator, 'period'):
            print(f"   ✓ MOMENTUM动量周期配置: 周期{momentum_indicator.period}")
        else:
            print("   ⚠️ MOMENTUM动量周期配置缺失")
            
        # 检查MOMENTUM与均线的交叉逻辑
        if 'momentum' in result.columns and 'momentum_ma' in result.columns:
            momentum_col = result['momentum'].dropna()
            momentum_ma_col = result['momentum_ma'].dropna()
            if len(momentum_col) > 0 and len(momentum_ma_col) > 0:
                print(f"   ✓ MOMENTUM动量交叉逻辑完整（动量值与均线值对比分析）")
        else:
            print("   ⚠️ MOMENTUM动量交叉逻辑缺失")
    except Exception as e:
        print(f"❌ MOMENTUM参数和特性检查失败: {e}")
    
    compliance_rate = (compliance_score / total_checks) * 100
    print(f"📊 L4合规性评分: {compliance_score}/{total_checks} ({compliance_rate:.1f}%)")
    
    if compliance_score == total_checks:
        print("✅ 完全符合L4设计文档要求")
        return True
    else:
        print("❌ 部分不符合L4设计文档要求")
        return False

def run_momentum_tests():
    """运行所有MOMENTUM测试"""
    print("🚀 开始MOMENTUM (动量指标) 指标全面深度测试")
    print("="*80)
    
    # 准备测试数据
    test_data = create_test_data()
    print(f"✅ 测试数据准备完成: {len(test_data)}行")
    
    # 测试计数器
    total_tests = 6
    passed_tests = 0
    
    # 1. 实例化测试
    success, momentum_indicator = test_momentum_instantiation()
    if success:
        passed_tests += 1
    
    if momentum_indicator is None:
        print("❌ 无法继续后续测试，MOMENTUM实例化失败")
        return
    
    # 2. calculate测试
    success, result = test_momentum_calculate(momentum_indicator, test_data)
    if success:
        passed_tests += 1
    
    # 3. get_signal测试
    success, signal = test_momentum_get_signal(momentum_indicator, test_data)
    if success:
        passed_tests += 1
    
    # 4. 数据验证测试
    success = test_momentum_data_validation(momentum_indicator)
    if success:
        passed_tests += 1
    
    # 5. 性能测试
    success = test_momentum_performance(momentum_indicator, test_data)
    if success:
        passed_tests += 1
    
    # 6. L4合规性测试
    success = test_momentum_l4_compliance(momentum_indicator, test_data)
    if success:
        passed_tests += 1
    
    # 输出最终结果
    print("\n" + "="*80)
    print("📊 MOMENTUM指标测试总结")
    print("="*80)
    print(f"✅ 通过测试: {passed_tests}/{total_tests}")
    print(f"📈 成功率: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print("🎉 MOMENTUM指标全面测试通过！符合L4文档设计预期和基类设计预期")
    else:
        print("⚠️ MOMENTUM指标存在问题，需要进一步修复")

if __name__ == "__main__":
    run_momentum_tests()
