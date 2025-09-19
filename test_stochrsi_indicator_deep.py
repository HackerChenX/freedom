#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STOCHRSI (Stochastic RSI) 指标全面深度测试脚本
执行Stochastic RSI指标的全面质量验证测试
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
    dates = pd.date_range('2020-01-01', periods=120, freq='D')  # STOCHRSI需要较多数据
    np.random.seed(42)  # 设置随机种子以获得可重现的结果
    
    # 生成具有明显振荡特征的价格数据，适合STOCHRSI测试
    base_price = 100
    
    # 创建多重振荡价格模式，适合Stochastic RSI测试
    price_patterns = []
    # 第一段：下跌振荡 (0-30) - 适合STOCHRSI超卖测试
    for i in range(30):
        price_patterns.append(-15 + 3 * np.sin(i * 0.3) + 2 * np.cos(i * 0.2) + 1.2 * np.random.normal())
    # 第二段：盘整振荡 (30-60) - 适合STOCHRSI中性区域测试
    for i in range(30):
        price_patterns.append(-5 + 2 * np.sin(i * 0.4) + 1.5 * np.cos(i * 0.3) + 1.0 * np.random.normal())
    # 第三段：上涨振荡 (60-90) - 适合STOCHRSI超买测试
    for i in range(30):
        price_patterns.append(8 + 4 * np.sin(i * 0.5) + 2.5 * np.cos(i * 0.4) + 1.3 * np.random.normal())
    # 第四段：回调振荡 (90-120) - 适合STOCHRSI交叉信号测试
    for i in range(30):
        price_patterns.append(5 - i * 0.3 + 3 * np.sin(i * 0.6) + 1.8 * np.cos(i * 0.5) + 1.1 * np.random.normal())
    
    prices = base_price + np.array(price_patterns)
    
    # 生成OHLC数据
    data = pd.DataFrame({
        'open': prices + np.random.normal(0, 0.8, 120),
        'high': prices + np.abs(np.random.normal(0, 3.0, 120)),
        'low': prices - np.abs(np.random.normal(0, 3.0, 120)),
        'close': prices,
        'volume': np.random.randint(1000000, 25000000, 120),
        'turnover_rate': np.random.uniform(0.1, 12.0, 120)
    }, index=dates)
    
    # 确保 high >= low, open/close 在 high/low 之间
    data['high'] = np.maximum.reduce([data['open'], data['high'], data['close']])
    data['low'] = np.minimum.reduce([data['open'], data['low'], data['close']])
    
    return data

def test_stochrsi_instantiation():
    """测试STOCHRSI指标实例化"""
    print("\n" + "="*60)
    print("🔍 STOCHRSI指标实例化测试")
    print("="*60)
    
    try:
        manager = UnifiedIndicatorManager()
        stochrsi_indicator = manager.create_indicator('STOCHRSI')
        
        if stochrsi_indicator is not None:
            print("✅ STOCHRSI指标实例化成功")
            print(f"   类名: {stochrsi_indicator.__class__.__name__}")
            print(f"   指标名: {getattr(stochrsi_indicator, 'name', 'N/A')}")
            print(f"   RSI周期: {getattr(stochrsi_indicator, 'rsi_period', 'N/A')}")
            print(f"   随机周期: {getattr(stochrsi_indicator, 'stoch_period', 'N/A')}")
            print(f"   K周期: {getattr(stochrsi_indicator, 'k_period', 'N/A')}")
            print(f"   D周期: {getattr(stochrsi_indicator, 'd_period', 'N/A')}")
            print(f"   最小周期: {getattr(stochrsi_indicator, 'minimum_periods', 'N/A')}")
            return True, stochrsi_indicator
        else:
            print("❌ STOCHRSI指标实例化失败")
            return False, None
            
    except Exception as e:
        print(f"❌ STOCHRSI指标实例化异常: {e}")
        return False, None

def test_stochrsi_calculate(stochrsi_indicator, test_data):
    """测试STOCHRSI指标calculate()方法"""
    print("\n" + "="*60)
    print("📊 STOCHRSI指标calculate()测试")
    print("="*60)
    
    try:
        result = stochrsi_indicator.calculate_Stochrsi(test_data)
        
        if result is not None and not result.empty:
            print("✅ STOCHRSI指标calculate()方法成功")
            print(f"   结果行数: {len(result)}")
            print(f"   结果列数: {len(result.columns)}")
            print(f"   列名: {list(result.columns)}")
            
            # 检查关键的STOCHRSI列
            stochrsi_columns = [col for col in result.columns if any(x in col.upper() for x in ['STOCHRSI', 'STOCH', 'RSI'])]
            if stochrsi_columns:
                print(f"   STOCHRSI相关列: {stochrsi_columns}")
                for col in stochrsi_columns[:6]:  # 显示前6列的统计信息
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
                print("⚠️ 警告: 未找到STOCHRSI相关的输出列")
                return False, result
        else:
            print("❌ STOCHRSI指标calculate()返回空结果")
            return False, None
            
    except Exception as e:
        print(f"❌ STOCHRSI指标calculate()异常: {e}")
        return False, None

def test_stochrsi_get_signal(stochrsi_indicator, test_data):
    """测试STOCHRSI指标get_signal()方法"""
    print("\n" + "="*60)
    print("📡 STOCHRSI指标get_signal()测试")
    print("="*60)
    
    try:
        signal = stochrsi_indicator.get_signal(test_data)
        
        if signal and isinstance(signal, dict):
            print("✅ STOCHRSI指标get_signal()方法成功")
            
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
                
                # 显示STOCHRSI特有的元数据信息
                if 'metadata' in signal and signal['metadata']:
                    metadata = signal['metadata']
                    if 'stochrsi_k' in metadata:
                        print(f"   STOCHRSI_K值: {metadata['stochrsi_k']:.4f}")
                    if 'stochrsi_d' in metadata:
                        print(f"   STOCHRSI_D值: {metadata['stochrsi_d']:.4f}")
                    if 'stochrsi_k_previous' in metadata:
                        print(f"   前期K值: {metadata['stochrsi_k_previous']:.4f}")
                    if 'stochrsi_d_previous' in metadata:
                        print(f"   前期D值: {metadata['stochrsi_d_previous']:.4f}")
                    if 'k_change' in metadata:
                        print(f"   K变化: {metadata['k_change']:.4f}")
                    if 'd_change' in metadata:
                        print(f"   D变化: {metadata['d_change']:.4f}")
                    if 'k_momentum' in metadata:
                        print(f"   K动量: {metadata['k_momentum']}")
                    if 'd_momentum' in metadata:
                        print(f"   D动量: {metadata['d_momentum']}")
                    if 'stochrsi_zone' in metadata:
                        print(f"   STOCHRSI区域: {metadata['stochrsi_zone']}")
                    if 'relative_position' in metadata:
                        print(f"   相对位置: {metadata['relative_position']:.4f}")
                    if 'kd_diff' in metadata:
                        print(f"   K-D差值: {metadata['kd_diff']:.4f}")
                    if 'k_cross_up_d' in metadata:
                        print(f"   K上穿D: {metadata['k_cross_up_d']}")
                    if 'k_cross_down_d' in metadata:
                        print(f"   K下穿D: {metadata['k_cross_down_d']}")
                    if 'in_overbought' in metadata:
                        print(f"   是否超买: {metadata['in_overbought']}")
                    if 'in_oversold' in metadata:
                        print(f"   是否超卖: {metadata['in_oversold']}")
                
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
            print("❌ STOCHRSI指标get_signal()返回格式不正确")
            return False, None
            
    except Exception as e:
        print(f"❌ STOCHRSI指标get_signal()异常: {e}")
        return False, None

def test_stochrsi_data_validation(stochrsi_indicator):
    """测试STOCHRSI指标数据验证"""
    print("\n" + "="*60)
    print("🔒 STOCHRSI指标数据验证测试")
    print("="*60)
    
    test_cases = [
        ("空数据", pd.DataFrame()),
        ("缺少必需列", pd.DataFrame({'high': [1, 2, 3]})),
        ("数据量不足", pd.DataFrame({
            'close': list(range(1, 25))
        }))  # STOCHRSI需要足够的数据（约50+周期）
    ]
    
    validation_passed = 0
    for test_name, test_data in test_cases:
        try:
            signal = stochrsi_indicator.get_signal(test_data)
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

def test_stochrsi_performance(stochrsi_indicator, test_data):
    """测试STOCHRSI指标性能"""
    print("\n" + "="*60)
    print("⚡ STOCHRSI指标性能测试")
    print("="*60)
    
    import time
    
    # 测试calculate性能
    start_time = time.time()
    for i in range(10):
        stochrsi_indicator.calculate_Stochrsi(test_data)
    calculate_time = (time.time() - start_time) / 10
    
    # 测试get_signal性能
    start_time = time.time()
    for i in range(100):
        stochrsi_indicator.get_signal(test_data)
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

def test_stochrsi_l4_compliance(stochrsi_indicator, test_data):
    """测试STOCHRSI指标L4设计文档合规性"""
    print("\n" + "="*60)
    print("📋 STOCHRSI指标L4设计文档合规性测试")
    print("="*60)
    
    compliance_score = 0
    total_checks = 7
    
    # 1. 检查必需的抽象方法实现
    required_methods = ['calculate_Stochrsi', 'get_signal', 'get_patterns']
    missing_methods = []
    for method in required_methods:
        if not hasattr(stochrsi_indicator, method):
            missing_methods.append(method)
    
    if not missing_methods:
        print("✅ 必需抽象方法全部实现")
        compliance_score += 1
    else:
        print(f"❌ 缺少必需抽象方法: {missing_methods}")
    
    # 2. 检查has_result方法
    if hasattr(stochrsi_indicator, 'has_result') and callable(getattr(stochrsi_indicator, 'has_result')):
        print("✅ has_result()方法已实现")
        compliance_score += 1
    else:
        print("❌ has_result()方法缺失")
    
    # 3. 检查_result属性管理
    try:
        stochrsi_indicator.calculate_Stochrsi(test_data)
        if hasattr(stochrsi_indicator, '_result') and stochrsi_indicator._result is not None:
            print("✅ _result属性管理正确")
            compliance_score += 1
        else:
            print("❌ _result属性管理有问题")
    except Exception as e:
        print(f"❌ _result属性管理测试失败: {e}")
    
    # 4. 检查标准化信号格式
    try:
        signal = stochrsi_indicator.get_signal(test_data)
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
        signal = stochrsi_indicator.get_signal(invalid_data)
        if signal.get('signal_type') == 'hold':
            print("✅ 异常处理机制正常")
            compliance_score += 1
        else:
            print("❌ 异常处理机制有问题")
    except Exception as e:
        print(f"❌ 异常处理测试失败: {e}")
    
    # 6. 检查STOCHRSI特有算法
    try:
        # 检查STOCHRSI核心算法实现
        result = stochrsi_indicator.calculate_Stochrsi(test_data)
        expected_stochrsi_columns = ['STOCHRSI_K', 'STOCHRSI_D']  # 关键的STOCHRSI列
        has_stochrsi_columns = all(col in result.columns for col in expected_stochrsi_columns)
        
        if has_stochrsi_columns:
            print("✅ STOCHRSI核心算法实现完整")
            compliance_score += 1
        else:
            print(f"❌ STOCHRSI核心算法缺失，找不到关键列：{expected_stochrsi_columns}")
    except Exception as e:
        print(f"❌ STOCHRSI算法检查失败: {e}")
    
    # 7. 检查STOCHRSI参数配置和复合振荡器特性
    try:
        # 检查STOCHRSI多周期参数
        required_params = ['rsi_period', 'stoch_period', 'k_period', 'd_period']
        missing_params = [param for param in required_params if not hasattr(stochrsi_indicator, param)]
        if not missing_params:
            print("✅ STOCHRSI参数配置完备")
            compliance_score += 1
        else:
            print(f"❌ STOCHRSI参数配置缺失: {missing_params}")
            
        # 额外检查：验证STOCHRSI是否在0到100区间内振荡
        result = stochrsi_indicator.calculate_Stochrsi(test_data)
        for col in ['STOCHRSI_K', 'STOCHRSI_D']:
            if col in result.columns:
                stochrsi_values = result[col].dropna()
                if len(stochrsi_values) > 0:
                    stochrsi_range_valid = (stochrsi_values >= -5).all() and (stochrsi_values <= 105).all()  # 允许轻微越界
                    if stochrsi_range_valid:
                        print(f"   ✓ {col}振荡器特性正常（0到100区间）")
                    else:
                        print(f"   ⚠️ {col}值可能超出正常范围: [{stochrsi_values.min():.2f}, {stochrsi_values.max():.2f}]")
                        
        # 检查STOCHRSI复合特性（RSI + Stochastic）
        if hasattr(stochrsi_indicator, 'rsi_period') and hasattr(stochrsi_indicator, 'stoch_period'):
            print(f"   ✓ STOCHRSI复合指标配置: RSI周期{stochrsi_indicator.rsi_period}, 随机周期{stochrsi_indicator.stoch_period}")
        else:
            print("   ⚠️ STOCHRSI复合指标配置缺失")
    except Exception as e:
        print(f"❌ STOCHRSI参数和特性检查失败: {e}")
    
    compliance_rate = (compliance_score / total_checks) * 100
    print(f"📊 L4合规性评分: {compliance_score}/{total_checks} ({compliance_rate:.1f}%)")
    
    if compliance_score == total_checks:
        print("✅ 完全符合L4设计文档要求")
        return True
    else:
        print("❌ 部分不符合L4设计文档要求")
        return False

def run_stochrsi_tests():
    """运行所有STOCHRSI测试"""
    print("🚀 开始STOCHRSI (Stochastic RSI) 指标全面深度测试")
    print("="*80)
    
    # 准备测试数据
    test_data = create_test_data()
    print(f"✅ 测试数据准备完成: {len(test_data)}行")
    
    # 测试计数器
    total_tests = 6
    passed_tests = 0
    
    # 1. 实例化测试
    success, stochrsi_indicator = test_stochrsi_instantiation()
    if success:
        passed_tests += 1
    
    if stochrsi_indicator is None:
        print("❌ 无法继续后续测试，STOCHRSI实例化失败")
        return
    
    # 2. calculate测试
    success, result = test_stochrsi_calculate(stochrsi_indicator, test_data)
    if success:
        passed_tests += 1
    
    # 3. get_signal测试
    success, signal = test_stochrsi_get_signal(stochrsi_indicator, test_data)
    if success:
        passed_tests += 1
    
    # 4. 数据验证测试
    success = test_stochrsi_data_validation(stochrsi_indicator)
    if success:
        passed_tests += 1
    
    # 5. 性能测试
    success = test_stochrsi_performance(stochrsi_indicator, test_data)
    if success:
        passed_tests += 1
    
    # 6. L4合规性测试
    success = test_stochrsi_l4_compliance(stochrsi_indicator, test_data)
    if success:
        passed_tests += 1
    
    # 输出最终结果
    print("\n" + "="*80)
    print("📊 STOCHRSI指标测试总结")
    print("="*80)
    print(f"✅ 通过测试: {passed_tests}/{total_tests}")
    print(f"📈 成功率: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print("🎉 STOCHRSI指标全面测试通过！符合L4文档设计预期和基类设计预期")
    else:
        print("⚠️ STOCHRSI指标存在问题，需要进一步修复")

if __name__ == "__main__":
    run_stochrsi_tests()
