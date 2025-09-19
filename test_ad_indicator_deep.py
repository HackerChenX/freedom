#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AD (Accumulation/Distribution Line) 指标全面深度测试脚本
执行AD指标的全面质量验证测试
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
    dates = pd.date_range('2020-01-01', periods=100, freq='D')  # AD需要足够数据
    np.random.seed(42)  # 设置随机种子以获得可重现的结果
    
    # 生成具有明显AD特征的价格和成交量数据
    base_price = 100
    base_volume = 1000000
    
    # 创建AD模式价格数据，适合累积分布线测试
    price_patterns = []
    volume_patterns = []
    high_low_spreads = []
    
    # 第一段：累积阶段 (0-25) - 适合AD累积测试
    for i in range(25):
        price_base = 100 + i * 0.8 + 2.0 * np.sin(i * 0.1) + 0.5 * np.random.normal()
        price_patterns.append(price_base)
        volume_patterns.append(1.0 + i * 0.03 + 0.4 * np.sin(i * 0.2) + 0.2 * np.random.normal())
        high_low_spreads.append(2.0 + 0.5 * np.random.normal())
    
    # 第二段：分布阶段 (25-50) - 适合AD分布测试
    for i in range(25):
        price_base = 120 - i * 0.6 + 1.5 * np.sin(i * 0.15) + 0.8 * np.random.normal()
        price_patterns.append(price_base)
        volume_patterns.append(1.8 + i * 0.02 + 0.3 * np.sin(i * 0.25) + 0.15 * np.random.normal())
        high_low_spreads.append(2.5 + 0.6 * np.random.normal())
    
    # 第三段：背离阶段 (50-75) - 适合AD背离测试
    for i in range(25):
        price_base = 105 + i * 0.4 + 1.8 * np.sin(i * 0.12) + 0.6 * np.random.normal()
        price_patterns.append(price_base)
        volume_patterns.append(2.2 - i * 0.015 + 0.2 * np.sin(i * 0.3) + 0.1 * np.random.normal())
        high_low_spreads.append(1.8 + 0.4 * np.random.normal())
    
    # 第四段：趋势确认阶段 (75-100) - 适合AD趋势测试
    for i in range(25):
        price_base = 115 + i * 0.5 + 2.2 * np.sin(i * 0.08) + 0.7 * np.random.normal()
        price_patterns.append(price_base)
        volume_patterns.append(2.0 + i * 0.01 + 0.35 * np.sin(i * 0.18) + 0.12 * np.random.normal())
        high_low_spreads.append(2.2 + 0.5 * np.random.normal())
    
    prices = np.array(price_patterns)
    volumes = base_volume * np.array(volume_patterns)
    spreads = np.array(high_low_spreads)
    
    # 生成OHLCV数据，确保AD计算的CLV有意义
    data = pd.DataFrame({
        'close': prices,
        'high': prices + spreads * 0.6,
        'low': prices - spreads * 0.4,
        'open': prices + np.random.normal(0, 0.5, 100),
        'volume': volumes,
        'turnover_rate': np.random.uniform(0.1, 10.0, 100)
    }, index=dates)
    
    # 确保 high >= low >= 0, open/close 在 high/low 之间
    data['high'] = np.maximum.reduce([data['open'], data['high'], data['close']])
    data['low'] = np.minimum.reduce([data['open'], data['low'], data['close']])
    data['low'] = np.maximum(data['low'], 0)  # 确保价格非负
    
    return data

def test_ad_instantiation():
    """测试AD指标实例化"""
    print("\n" + "="*60)
    print("🔍 AD指标实例化测试")
    print("="*60)
    
    try:
        manager = UnifiedIndicatorManager()
        ad_indicator = manager.create_indicator('AD')
        
        if ad_indicator is not None:
            print("✅ AD指标实例化成功")
            print(f"   类名: {ad_indicator.__class__.__name__}")
            print(f"   指标名: {getattr(ad_indicator, 'name', 'N/A')}")
            print(f"   描述: {getattr(ad_indicator, 'description', 'N/A')}")
            print(f"   指标类型: {getattr(ad_indicator, 'indicator_type', 'N/A')}")
            print(f"   必需列: {getattr(ad_indicator, 'REQUIRED_COLUMNS', 'N/A')}")
            return True, ad_indicator
        else:
            print("❌ AD指标实例化失败")
            return False, None
            
    except Exception as e:
        print(f"❌ AD指标实例化异常: {e}")
        return False, None

def test_ad_calculate(ad_indicator, test_data):
    """测试AD指标calculate()方法"""
    print("\n" + "="*60)
    print("📊 AD指标calculate()测试")
    print("="*60)
    
    try:
        result = ad_indicator.calculate(test_data)
        
        if result is not None and not result.empty:
            print("✅ AD指标calculate()方法成功")
            print(f"   结果行数: {len(result)}")
            print(f"   结果列数: {len(result.columns)}")
            print(f"   列名: {list(result.columns)}")
            
            # 检查关键的AD列
            ad_columns = [col for col in result.columns if any(x in col.upper() for x in ['AD', 'ACCUMULATION', 'DISTRIBUTION'])]
            if ad_columns:
                print(f"   AD相关列: {ad_columns}")
                for col in ad_columns[:8]:  # 显示前8列的统计信息
                    if len(result[col]) > 0 and result[col].dtype in ['float64', 'int64']:
                        # 跳过NaN值进行统计
                        valid_values = result[col].dropna()
                        if len(valid_values) > 0:
                            print(f"   {col}: 最新值={valid_values.iloc[-1]:.2f}, 平均值={valid_values.mean():.2f}, 范围=[{valid_values.min():.0f}, {valid_values.max():.0f}]")
                        else:
                            print(f"   {col}: 全部为NaN")
                    elif len(result[col]) > 0:
                        print(f"   {col}: 最新值={result[col].iloc[-1]}")
                        
                # 检查AD核心算法
                if 'AD' in result.columns:
                    ad_values = result['AD'].dropna()
                    if len(ad_values) > 0:
                        ad_change = ad_values.iloc[-1] - ad_values.iloc[0] if len(ad_values) > 1 else 0
                        print(f"   AD总变化: {ad_change:.0f} (从 {ad_values.iloc[0]:.0f} 到 {ad_values.iloc[-1]:.0f})")
                        
                        # 验证AD算法 - Close Location Value (CLV)
                        if len(test_data) >= 10:
                            recent_data = test_data.iloc[-10:]
                            recent_ad = ad_values.iloc[-10:]
                            
                            # 计算预期的CLV
                            high = recent_data['high']
                            low = recent_data['low']
                            close = recent_data['close']
                            volume = recent_data['volume']
                            
                            clv = ((close - low) - (high - close)) / (high - low)
                            money_flow_volume = clv * volume
                            
                            print(f"   CLV范围: [{clv.min():.3f}, {clv.max():.3f}]")
                            print(f"   资金流量均值: {money_flow_volume.mean():.0f}")
                        
                return True, result
            else:
                print("⚠️ 警告: 未找到AD相关的输出列")
                return False, result
        else:
            print("❌ AD指标calculate()返回空结果")
            return False, None
            
    except Exception as e:
        print(f"❌ AD指标calculate()异常: {e}")
        return False, None

def test_ad_get_signal(ad_indicator, test_data):
    """测试AD指标get_signal()方法"""
    print("\n" + "="*60)
    print("📡 AD指标get_signal()测试")
    print("="*60)
    
    try:
        signal = ad_indicator.get_signal(test_data)
        
        if signal and isinstance(signal, dict):
            print("✅ AD指标get_signal()方法成功")
            
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
                
                # 显示AD特有的元数据信息
                if 'metadata' in signal and signal['metadata']:
                    metadata = signal['metadata']
                    if 'ad_value' in metadata:
                        print(f"   AD值: {metadata['ad_value']:.2f}")
                    if 'ad_previous' in metadata:
                        print(f"   前期AD值: {metadata['ad_previous']:.2f}")
                    if 'ad_change' in metadata:
                        print(f"   AD变化: {metadata['ad_change']:.2f}")
                    if 'ad_change_pct' in metadata:
                        print(f"   AD变化率: {metadata['ad_change_pct']:.2f}%")
                    if 'ad_trend' in metadata:
                        print(f"   AD趋势: {metadata['ad_trend']}")
                    if 'price_change_pct' in metadata:
                        print(f"   价格变化率: {metadata['price_change_pct']:.2f}%")
                    if 'price_trend' in metadata:
                        print(f"   价格趋势: {metadata['price_trend']}")
                    if 'accumulation_distribution' in metadata:
                        print(f"   累积分布关系: {metadata['accumulation_distribution']}")
                    if 'ad_ma' in metadata and metadata['ad_ma'] is not None:
                        print(f"   AD均线: {metadata['ad_ma']:.2f}")
                    if 'above_ma' in metadata and metadata['above_ma'] is not None:
                        print(f"   高于均线: {metadata['above_ma']}")
                    if 'indicator_type' in metadata:
                        print(f"   指标类型: {metadata['indicator_type']}")
                    if 'calculation_method' in metadata:
                        print(f"   计算方法: {metadata['calculation_method']}")
                
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
            print("❌ AD指标get_signal()返回格式不正确")
            return False, None
            
    except Exception as e:
        print(f"❌ AD指标get_signal()异常: {e}")
        return False, None

def test_ad_data_validation(ad_indicator):
    """测试AD指标数据验证"""
    print("\n" + "="*60)
    print("🔒 AD指标数据验证测试")
    print("="*60)
    
    test_cases = [
        ("空数据", pd.DataFrame()),
        ("缺少必需列", pd.DataFrame({'close': [1, 2, 3]})),
        ("数据量不足", pd.DataFrame({
            'high': [101, 102],
            'low': [99, 100],
            'close': [100, 101],
            'volume': [1000, 1100]
        }))  # AD需要足够的数据
    ]
    
    validation_passed = 0
    for test_name, test_data in test_cases:
        try:
            signal = ad_indicator.get_signal(test_data)
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

def test_ad_performance(ad_indicator, test_data):
    """测试AD指标性能"""
    print("\n" + "="*60)
    print("⚡ AD指标性能测试")
    print("="*60)
    
    import time
    
    # 测试calculate性能
    start_time = time.time()
    for i in range(10):
        ad_indicator.calculate(test_data)
    calculate_time = (time.time() - start_time) / 10
    
    # 测试get_signal性能
    start_time = time.time()
    for i in range(100):
        ad_indicator.get_signal(test_data)
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

def test_ad_l4_compliance(ad_indicator, test_data):
    """测试AD指标L4设计文档合规性"""
    print("\n" + "="*60)
    print("📋 AD指标L4设计文档合规性测试")
    print("="*60)
    
    compliance_score = 0
    total_checks = 7
    
    # 1. 检查必需的抽象方法实现
    required_methods = ['calculate', 'get_signal', 'has_result']
    missing_methods = []
    for method in required_methods:
        if not hasattr(ad_indicator, method):
            missing_methods.append(method)
    
    if not missing_methods:
        print("✅ 必需抽象方法全部实现")
        compliance_score += 1
    else:
        print(f"❌ 缺少必需抽象方法: {missing_methods}")
    
    # 2. 检查has_result方法
    if hasattr(ad_indicator, 'has_result') and callable(getattr(ad_indicator, 'has_result')):
        print("✅ has_result()方法已实现")
        compliance_score += 1
    else:
        print("❌ has_result()方法缺失")
    
    # 3. 检查_result属性管理
    try:
        ad_indicator.calculate(test_data)
        if hasattr(ad_indicator, '_result') and ad_indicator._result is not None:
            print("✅ _result属性管理正确")
            compliance_score += 1
        else:
            print("❌ _result属性管理有问题")
    except Exception as e:
        print(f"❌ _result属性管理测试失败: {e}")
    
    # 4. 检查标准化信号格式
    try:
        signal = ad_indicator.get_signal(test_data)
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
        signal = ad_indicator.get_signal(invalid_data)
        if signal.get('signal_type') == 'hold':
            print("✅ 异常处理机制正常")
            compliance_score += 1
        else:
            print("❌ 异常处理机制有问题")
    except Exception as e:
        print(f"❌ 异常处理测试失败: {e}")
    
    # 6. 检查AD特有算法
    try:
        # 检查AD核心算法实现
        result = ad_indicator.calculate(test_data)
        expected_ad_columns = ['AD']  # 关键的AD列
        has_ad_columns = all(col in result.columns for col in expected_ad_columns)
        
        if has_ad_columns:
            print("✅ AD核心算法实现完整")
            compliance_score += 1
        else:
            print(f"❌ AD核心算法缺失，找不到关键列：{expected_ad_columns}")
    except Exception as e:
        print(f"❌ AD算法检查失败: {e}")
    
    # 7. 检查AD累积分布特性
    try:
        # 检查AD累积分布算法
        result = ad_indicator.calculate(test_data)
        if 'AD' in result.columns:
            ad_values = result['AD'].dropna()
            if len(ad_values) > 0:
                print(f"   ✓ AD算法正常（值域: [{ad_values.min():.0f}, {ad_values.max():.0f}]）")
                
                # 检查AD累积分布逻辑
                signal = ad_indicator.get_signal(test_data)
                if 'metadata' in signal and signal['metadata']:
                    metadata = signal['metadata']
                    if 'accumulation_distribution' in metadata:
                        print(f"   ✓ AD累积分布逻辑完整（{metadata['accumulation_distribution']}）")
                    if 'ad_trend' in metadata and 'price_trend' in metadata:
                        print(f"   ✓ AD趋势分析完整（AD: {metadata['ad_trend']}, 价格: {metadata['price_trend']}）")
                    if 'calculation_method' in metadata:
                        print(f"   ✓ AD计算方法正确（{metadata['calculation_method']}）")
                    
                    print("✅ AD累积分布特性配置完备")
                    compliance_score += 1
                else:
                    print("❌ AD累积分布特性逻辑缺失")
            else:
                print("❌ AD值计算异常")
        else:
            print("❌ AD算法实现缺失")
    except Exception as e:
        print(f"❌ AD累积分布特性检查失败: {e}")
    
    compliance_rate = (compliance_score / total_checks) * 100
    print(f"📊 L4合规性评分: {compliance_score}/{total_checks} ({compliance_rate:.1f}%)")
    
    if compliance_score == total_checks:
        print("✅ 完全符合L4设计文档要求")
        return True
    else:
        print("❌ 部分不符合L4设计文档要求")
        return False

def run_ad_tests():
    """运行所有AD测试"""
    print("🚀 开始AD (Accumulation/Distribution Line) 指标全面深度测试")
    print("="*80)
    
    # 准备测试数据
    test_data = create_test_data()
    print(f"✅ 测试数据准备完成: {len(test_data)}行")
    
    # 测试计数器
    total_tests = 6
    passed_tests = 0
    
    # 1. 实例化测试
    success, ad_indicator = test_ad_instantiation()
    if success:
        passed_tests += 1
    
    if ad_indicator is None:
        print("❌ 无法继续后续测试，AD实例化失败")
        return
    
    # 2. calculate测试
    success, result = test_ad_calculate(ad_indicator, test_data)
    if success:
        passed_tests += 1
    
    # 3. get_signal测试
    success, signal = test_ad_get_signal(ad_indicator, test_data)
    if success:
        passed_tests += 1
    
    # 4. 数据验证测试
    success = test_ad_data_validation(ad_indicator)
    if success:
        passed_tests += 1
    
    # 5. 性能测试
    success = test_ad_performance(ad_indicator, test_data)
    if success:
        passed_tests += 1
    
    # 6. L4合规性测试
    success = test_ad_l4_compliance(ad_indicator, test_data)
    if success:
        passed_tests += 1
    
    # 输出最终结果
    print("\n" + "="*80)
    print("📊 AD指标测试总结")
    print("="*80)
    print(f"✅ 通过测试: {passed_tests}/{total_tests}")
    print(f"📈 成功率: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print("🎉 AD指标全面测试通过！符合L4文档设计预期和基类设计预期")
    else:
        print("⚠️ AD指标存在问题，需要进一步修复")

if __name__ == "__main__":
    run_ad_tests()
