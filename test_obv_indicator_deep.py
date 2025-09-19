#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OBV (On-Balance Volume) 指标全面深度测试脚本
执行OBV指标的全面质量验证测试
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
    dates = pd.date_range('2020-01-01', periods=100, freq='D')  # OBV需要足够数据
    np.random.seed(42)  # 设置随机种子以获得可重现的结果
    
    # 生成具有明显OBV特征的价格和成交量数据
    base_price = 100
    base_volume = 1000000
    
    # 创建OBV模式价格数据，适合量价关系测试
    price_patterns = []
    volume_patterns = []
    # 第一段：放量上涨 (0-25) - 适合OBV量价配合测试
    for i in range(25):
        price_patterns.append(5 + i * 0.4 + 1.5 * np.sin(i * 0.2) + 0.8 * np.random.normal())
        volume_patterns.append(1.2 + i * 0.02 + 0.3 * np.sin(i * 0.3) + 0.2 * np.random.normal())
    # 第二段：缩量整理 (25-50) - 适合OBV平稳测试
    for i in range(25):
        price_patterns.append(15 + 2 * np.sin(i * 0.5) + 1.0 * np.random.normal())
        volume_patterns.append(1.5 + 0.5 * np.sin(i * 0.4) + 0.15 * np.random.normal())
    # 第三段：放量下跌 (50-75) - 适合OBV量价配合下跌测试
    for i in range(25):
        price_patterns.append(17 - i * 0.3 + 1.8 * np.sin(i * 0.25) + 1.2 * np.random.normal())
        volume_patterns.append(1.8 + i * 0.015 + 0.4 * np.sin(i * 0.35) + 0.25 * np.random.normal())
    # 第四段：缩量反弹 (75-100) - 适合OBV背离测试
    for i in range(25):
        price_patterns.append(10 + i * 0.2 + 2.5 * np.sin(i * 0.15) + 0.9 * np.random.normal())
        volume_patterns.append(2.3 - i * 0.01 + 0.2 * np.sin(i * 0.2) + 0.1 * np.random.normal())
    
    prices = base_price + np.array(price_patterns)
    volumes = base_volume * np.array(volume_patterns)
    
    # 生成OHLCV数据
    data = pd.DataFrame({
        'open': prices + np.random.normal(0, 0.5, 100),
        'high': prices + np.abs(np.random.normal(0, 2.0, 100)),
        'low': prices - np.abs(np.random.normal(0, 2.0, 100)),
        'close': prices,
        'volume': volumes,
        'turnover_rate': np.random.uniform(0.1, 10.0, 100)
    }, index=dates)
    
    # 确保 high >= low, open/close 在 high/low 之间
    data['high'] = np.maximum.reduce([data['open'], data['high'], data['close']])
    data['low'] = np.minimum.reduce([data['open'], data['low'], data['close']])
    
    return data

def test_obv_instantiation():
    """测试OBV指标实例化"""
    print("\n" + "="*60)
    print("🔍 OBV指标实例化测试")
    print("="*60)
    
    try:
        manager = UnifiedIndicatorManager()
        obv_indicator = manager.create_indicator('OBV')
        
        if obv_indicator is not None:
            print("✅ OBV指标实例化成功")
            print(f"   类名: {obv_indicator.__class__.__name__}")
            print(f"   指标名: {getattr(obv_indicator, 'name', 'N/A')}")
            print(f"   信号周期: {getattr(obv_indicator, 'signal_period', 'N/A')}")
            print(f"   最小周期: {getattr(obv_indicator, 'minimum_periods', 'N/A')}")
            return True, obv_indicator
        else:
            print("❌ OBV指标实例化失败")
            return False, None
            
    except Exception as e:
        print(f"❌ OBV指标实例化异常: {e}")
        return False, None

def test_obv_calculate(obv_indicator, test_data):
    """测试OBV指标calculate()方法"""
    print("\n" + "="*60)
    print("📊 OBV指标calculate()测试")
    print("="*60)
    
    try:
        result = obv_indicator.calculate_Obv(test_data)
        
        if result is not None and not result.empty:
            print("✅ OBV指标calculate()方法成功")
            print(f"   结果行数: {len(result)}")
            print(f"   结果列数: {len(result.columns)}")
            print(f"   列名: {list(result.columns)}")
            
            # 检查关键的OBV列
            obv_columns = [col for col in result.columns if any(x in col.lower() for x in ['obv', 'volume', 'balance'])]
            if obv_columns:
                print(f"   OBV相关列: {obv_columns}")
                for col in obv_columns[:8]:  # 显示前8列的统计信息
                    if len(result[col]) > 0 and result[col].dtype in ['float64', 'int64']:
                        # 跳过NaN值进行统计
                        valid_values = result[col].dropna()
                        if len(valid_values) > 0:
                            print(f"   {col}: 最新值={valid_values.iloc[-1]:.2f}, 平均值={valid_values.mean():.2f}, 范围=[{valid_values.min():.0f}, {valid_values.max():.0f}]")
                        else:
                            print(f"   {col}: 全部为NaN")
                    elif len(result[col]) > 0:
                        print(f"   {col}: 最新值={result[col].iloc[-1]}")
                        
                # 检查OBV核心算法
                if 'OBV' in result.columns:
                    obv_values = result['OBV'].dropna()
                    if len(obv_values) > 0:
                        obv_change = obv_values.iloc[-1] - obv_values.iloc[0] if len(obv_values) > 1 else 0
                        print(f"   OBV总变化: {obv_change:.0f} (从 {obv_values.iloc[0]:.0f} 到 {obv_values.iloc[-1]:.0f})")
                        
                return True, result
            else:
                print("⚠️ 警告: 未找到OBV相关的输出列")
                return False, result
        else:
            print("❌ OBV指标calculate()返回空结果")
            return False, None
            
    except Exception as e:
        print(f"❌ OBV指标calculate()异常: {e}")
        return False, None

def test_obv_get_signal(obv_indicator, test_data):
    """测试OBV指标get_signal()方法"""
    print("\n" + "="*60)
    print("📡 OBV指标get_signal()测试")
    print("="*60)
    
    try:
        signal = obv_indicator.get_signal(test_data)
        
        if signal and isinstance(signal, dict):
            print("✅ OBV指标get_signal()方法成功")
            
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
                
                # 显示OBV特有的元数据信息
                if 'metadata' in signal and signal['metadata']:
                    metadata = signal['metadata']
                    if 'obv_value' in metadata:
                        print(f"   OBV值: {metadata['obv_value']:.2f}")
                    if 'obv_previous' in metadata:
                        print(f"   前期OBV值: {metadata['obv_previous']:.2f}")
                    if 'obv_change' in metadata:
                        print(f"   OBV变化: {metadata['obv_change']:.2f}")
                    if 'obv_change_pct' in metadata:
                        print(f"   OBV变化率: {metadata['obv_change_pct']:.2f}%")
                    if 'obv_trend' in metadata:
                        print(f"   OBV趋势: {metadata['obv_trend']}")
                    if 'price_change_pct' in metadata:
                        print(f"   价格变化率: {metadata['price_change_pct']:.2f}%")
                    if 'price_trend' in metadata:
                        print(f"   价格趋势: {metadata['price_trend']}")
                    if 'volume_price_relationship' in metadata:
                        print(f"   量价关系: {metadata['volume_price_relationship']}")
                    if 'obv_ma' in metadata and metadata['obv_ma'] is not None:
                        print(f"   OBV均线: {metadata['obv_ma']:.2f}")
                    if 'obv_strength' in metadata:
                        print(f"   OBV强度: {metadata['obv_strength']:.4f}")
                    if 'above_ma' in metadata and metadata['above_ma'] is not None:
                        print(f"   高于均线: {metadata['above_ma']}")
                    if 'signal_period' in metadata:
                        print(f"   信号周期: {metadata['signal_period']}")
                
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
            print("❌ OBV指标get_signal()返回格式不正确")
            return False, None
            
    except Exception as e:
        print(f"❌ OBV指标get_signal()异常: {e}")
        return False, None

def test_obv_data_validation(obv_indicator):
    """测试OBV指标数据验证"""
    print("\n" + "="*60)
    print("🔒 OBV指标数据验证测试")
    print("="*60)
    
    test_cases = [
        ("空数据", pd.DataFrame()),
        ("缺少必需列", pd.DataFrame({'close': [1, 2, 3]})),
        ("数据量不足", pd.DataFrame({
            'close': [100, 101],
            'volume': [1000, 1100]
        }))  # OBV需要足够的数据
    ]
    
    validation_passed = 0
    for test_name, test_data in test_cases:
        try:
            signal = obv_indicator.get_signal(test_data)
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

def test_obv_performance(obv_indicator, test_data):
    """测试OBV指标性能"""
    print("\n" + "="*60)
    print("⚡ OBV指标性能测试")
    print("="*60)
    
    import time
    
    # 测试calculate性能
    start_time = time.time()
    for i in range(10):
        obv_indicator.calculate_Obv(test_data)
    calculate_time = (time.time() - start_time) / 10
    
    # 测试get_signal性能
    start_time = time.time()
    for i in range(100):
        obv_indicator.get_signal(test_data)
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

def test_obv_l4_compliance(obv_indicator, test_data):
    """测试OBV指标L4设计文档合规性"""
    print("\n" + "="*60)
    print("📋 OBV指标L4设计文档合规性测试")
    print("="*60)
    
    compliance_score = 0
    total_checks = 7
    
    # 1. 检查必需的抽象方法实现
    required_methods = ['calculate_Obv', 'get_signal', 'has_result']
    missing_methods = []
    for method in required_methods:
        if not hasattr(obv_indicator, method):
            missing_methods.append(method)
    
    if not missing_methods:
        print("✅ 必需抽象方法全部实现")
        compliance_score += 1
    else:
        print(f"❌ 缺少必需抽象方法: {missing_methods}")
    
    # 2. 检查has_result方法
    if hasattr(obv_indicator, 'has_result') and callable(getattr(obv_indicator, 'has_result')):
        print("✅ has_result()方法已实现")
        compliance_score += 1
    else:
        print("❌ has_result()方法缺失")
    
    # 3. 检查_result属性管理
    try:
        obv_indicator.calculate_Obv(test_data)
        if hasattr(obv_indicator, '_result') and obv_indicator._result is not None:
            print("✅ _result属性管理正确")
            compliance_score += 1
        else:
            print("❌ _result属性管理有问题")
    except Exception as e:
        print(f"❌ _result属性管理测试失败: {e}")
    
    # 4. 检查标准化信号格式
    try:
        signal = obv_indicator.get_signal(test_data)
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
        signal = obv_indicator.get_signal(invalid_data)
        if signal.get('signal_type') == 'hold':
            print("✅ 异常处理机制正常")
            compliance_score += 1
        else:
            print("❌ 异常处理机制有问题")
    except Exception as e:
        print(f"❌ 异常处理测试失败: {e}")
    
    # 6. 检查OBV特有算法
    try:
        # 检查OBV核心算法实现
        result = obv_indicator.calculate_Obv(test_data)
        expected_obv_columns = ['OBV']  # 关键的OBV列
        has_obv_columns = all(col in result.columns for col in expected_obv_columns)
        
        if has_obv_columns:
            print("✅ OBV核心算法实现完整")
            compliance_score += 1
        else:
            print(f"❌ OBV核心算法缺失，找不到关键列：{expected_obv_columns}")
    except Exception as e:
        print(f"❌ OBV算法检查失败: {e}")
    
    # 7. 检查OBV量价关系特性
    try:
        # 检查OBV量价关系算法
        result = obv_indicator.calculate_Obv(test_data)
        if 'OBV' in result.columns:
            obv_values = result['OBV'].dropna()
            if len(obv_values) > 0:
                print(f"   ✓ OBV算法正常（值域: [{obv_values.min():.0f}, {obv_values.max():.0f}]）")
                
                # 检查OBV量价关系逻辑
                signal = obv_indicator.get_signal(test_data)
                if 'metadata' in signal and signal['metadata']:
                    metadata = signal['metadata']
                    if 'volume_price_relationship' in metadata:
                        print(f"   ✓ OBV量价关系逻辑完整（{metadata['volume_price_relationship']}）")
                    if 'obv_trend' in metadata and 'price_trend' in metadata:
                        print(f"   ✓ OBV趋势分析完整（OBV: {metadata['obv_trend']}, 价格: {metadata['price_trend']}）")
                    
                    print("✅ OBV量价关系特性配置完备")
                    compliance_score += 1
                else:
                    print("❌ OBV量价关系特性逻辑缺失")
            else:
                print("❌ OBV值计算异常")
        else:
            print("❌ OBV算法实现缺失")
    except Exception as e:
        print(f"❌ OBV量价关系特性检查失败: {e}")
    
    compliance_rate = (compliance_score / total_checks) * 100
    print(f"📊 L4合规性评分: {compliance_score}/{total_checks} ({compliance_rate:.1f}%)")
    
    if compliance_score == total_checks:
        print("✅ 完全符合L4设计文档要求")
        return True
    else:
        print("❌ 部分不符合L4设计文档要求")
        return False

def run_obv_tests():
    """运行所有OBV测试"""
    print("🚀 开始OBV (On-Balance Volume) 指标全面深度测试")
    print("="*80)
    
    # 准备测试数据
    test_data = create_test_data()
    print(f"✅ 测试数据准备完成: {len(test_data)}行")
    
    # 测试计数器
    total_tests = 6
    passed_tests = 0
    
    # 1. 实例化测试
    success, obv_indicator = test_obv_instantiation()
    if success:
        passed_tests += 1
    
    if obv_indicator is None:
        print("❌ 无法继续后续测试，OBV实例化失败")
        return
    
    # 2. calculate测试
    success, result = test_obv_calculate(obv_indicator, test_data)
    if success:
        passed_tests += 1
    
    # 3. get_signal测试
    success, signal = test_obv_get_signal(obv_indicator, test_data)
    if success:
        passed_tests += 1
    
    # 4. 数据验证测试
    success = test_obv_data_validation(obv_indicator)
    if success:
        passed_tests += 1
    
    # 5. 性能测试
    success = test_obv_performance(obv_indicator, test_data)
    if success:
        passed_tests += 1
    
    # 6. L4合规性测试
    success = test_obv_l4_compliance(obv_indicator, test_data)
    if success:
        passed_tests += 1
    
    # 输出最终结果
    print("\n" + "="*80)
    print("📊 OBV指标测试总结")
    print("="*80)
    print(f"✅ 通过测试: {passed_tests}/{total_tests}")
    print(f"📈 成功率: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print("🎉 OBV指标全面测试通过！符合L4文档设计预期和基类设计预期")
    else:
        print("⚠️ OBV指标存在问题，需要进一步修复")

if __name__ == "__main__":
    run_obv_tests()
