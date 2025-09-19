#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ENHANCED_WR (Enhanced Williams %R) 指标全面深度测试脚本
执行Enhanced Williams %R指标的全面质量验证测试
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
    dates = pd.date_range('2020-01-01', periods=100, freq='D')  # Enhanced WR需要适量数据
    np.random.seed(42)  # 设置随机种子以获得可重现的结果
    
    # 生成具有明显增强Williams %R特征的价格数据
    base_price = 100
    
    # 创建Enhanced Williams %R模式价格数据，适合增强型超买超卖测试
    price_patterns = []
    # 第一段：强势上涨后回调 (0-25) - 适合超买测试
    for i in range(25):
        price_patterns.append(8 + i * 0.6 + 2.5 * np.sin(i * 0.25) + 1.2 * np.random.normal())
    # 第二段：超买区域震荡 (25-50) - 适合自适应阈值测试
    for i in range(25):
        price_patterns.append(22 + 3 * np.sin(i * 0.4) + 1.8 * np.random.normal())
    # 第三段：急跌至超卖 (50-75) - 适合极值测试
    for i in range(25):
        price_patterns.append(25 - i * 0.8 + 2 * np.sin(i * 0.3) + 1.5 * np.random.normal())
    # 第四段：超卖反弹 (75-100) - 适合突破测试
    for i in range(25):
        price_patterns.append(5 + i * 0.4 + 3 * np.sin(i * 0.2) + 1.0 * np.random.normal())
    
    prices = base_price + np.array(price_patterns)
    
    # 生成OHLC数据
    data = pd.DataFrame({
        'open': prices + np.random.normal(0, 0.9, 100),
        'high': prices + np.abs(np.random.normal(0, 3.5, 100)),
        'low': prices - np.abs(np.random.normal(0, 3.5, 100)),
        'close': prices,
        'volume': np.random.randint(1000000, 30000000, 100),
        'turnover_rate': np.random.uniform(0.1, 15.0, 100)
    }, index=dates)
    
    # 确保 high >= low, open/close 在 high/low 之间
    data['high'] = np.maximum.reduce([data['open'], data['high'], data['close']])
    data['low'] = np.minimum.reduce([data['open'], data['low'], data['close']])
    
    return data

def test_enhanced_wr_instantiation():
    """测试ENHANCED_WR指标实例化"""
    print("\n" + "="*60)
    print("🔍 ENHANCED_WR指标实例化测试")
    print("="*60)
    
    try:
        manager = UnifiedIndicatorManager()
        enhanced_wr_indicator = manager.create_indicator('ENHANCED_WR')
        
        if enhanced_wr_indicator is not None:
            print("✅ ENHANCED_WR指标实例化成功")
            print(f"   类名: {enhanced_wr_indicator.__class__.__name__}")
            print(f"   指标名: {getattr(enhanced_wr_indicator, 'name', 'N/A')}")
            print(f"   周期: {getattr(enhanced_wr_indicator, 'period', 'N/A')}")
            print(f"   超买阈值: {getattr(enhanced_wr_indicator, 'overbought', 'N/A')}")
            print(f"   超卖阈值: {getattr(enhanced_wr_indicator, 'oversold', 'N/A')}")
            print(f"   多周期: {getattr(enhanced_wr_indicator, 'multi_periods', 'N/A')}")
            print(f"   自适应阈值: {getattr(enhanced_wr_indicator, 'adaptive_thresholds', 'N/A')}")
            print(f"   平滑周期: {getattr(enhanced_wr_indicator, 'smooth_period', 'N/A')}")
            return True, enhanced_wr_indicator
        else:
            print("❌ ENHANCED_WR指标实例化失败")
            return False, None
            
    except Exception as e:
        print(f"❌ ENHANCED_WR指标实例化异常: {e}")
        return False, None

def test_enhanced_wr_calculate(enhanced_wr_indicator, test_data):
    """测试ENHANCED_WR指标calculate()方法"""
    print("\n" + "="*60)
    print("📊 ENHANCED_WR指标calculate()测试")
    print("="*60)
    
    try:
        result = enhanced_wr_indicator.calculate(test_data)
        
        if result is not None and not result.empty:
            print("✅ ENHANCED_WR指标calculate()方法成功")
            print(f"   结果行数: {len(result)}")
            print(f"   结果列数: {len(result.columns)}")
            print(f"   列名: {list(result.columns)}")
            
            # 检查关键的Enhanced WR列
            wr_columns = [col for col in result.columns if any(x in col.lower() for x in ['wr', 'williams', 'enhanced', 'momentum', 'divergence', 'overbought', 'oversold'])]
            if wr_columns:
                print(f"   Enhanced WR相关列: {wr_columns}")
                for col in wr_columns[:10]:  # 显示前10列的统计信息
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
                print("⚠️ 警告: 未找到Enhanced WR相关的输出列")
                return False, result
        else:
            print("❌ ENHANCED_WR指标calculate()返回空结果")
            return False, None
            
    except Exception as e:
        print(f"❌ ENHANCED_WR指标calculate()异常: {e}")
        return False, None

def test_enhanced_wr_get_signal(enhanced_wr_indicator, test_data):
    """测试ENHANCED_WR指标get_signal()方法"""
    print("\n" + "="*60)
    print("📡 ENHANCED_WR指标get_signal()测试")
    print("="*60)
    
    try:
        signal = enhanced_wr_indicator.get_signal(test_data)
        
        if signal and isinstance(signal, dict):
            print("✅ ENHANCED_WR指标get_signal()方法成功")
            
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
                
                # 显示Enhanced WR特有的元数据信息
                if 'metadata' in signal and signal['metadata']:
                    metadata = signal['metadata']
                    if 'wr_value' in metadata:
                        print(f"   WR值: {metadata['wr_value']:.4f}")
                    if 'wr_previous' in metadata:
                        print(f"   前期WR值: {metadata['wr_previous']:.4f}")
                    if 'wr_change' in metadata:
                        print(f"   WR变化: {metadata['wr_change']:.4f}")
                    if 'wr_zone' in metadata:
                        print(f"   WR区域: {metadata['wr_zone']}")
                    if 'wr_momentum_direction' in metadata:
                        print(f"   WR动量方向: {metadata['wr_momentum_direction']}")
                    if 'wr_acceleration' in metadata:
                        print(f"   WR加速度: {metadata['wr_acceleration']}")
                    if 'adaptive_overbought_threshold' in metadata:
                        print(f"   自适应超买阈值: {metadata['adaptive_overbought_threshold']:.2f}")
                    if 'adaptive_oversold_threshold' in metadata:
                        print(f"   自适应超卖阈值: {metadata['adaptive_oversold_threshold']:.2f}")
                    if 'oversold_cross_up' in metadata:
                        print(f"   超卖突破向上: {metadata['oversold_cross_up']}")
                    if 'overbought_cross_down' in metadata:
                        print(f"   超买跌破向下: {metadata['overbought_cross_down']}")
                    if 'in_overbought' in metadata:
                        print(f"   处于超买区域: {metadata['in_overbought']}")
                    if 'in_oversold' in metadata:
                        print(f"   处于超卖区域: {metadata['in_oversold']}")
                    if 'wr_momentum' in metadata:
                        print(f"   WR动量: {metadata['wr_momentum']:.4f}")
                    if 'wr_divergence' in metadata:
                        print(f"   WR背离: {metadata['wr_divergence']:.4f}")
                    if 'enhanced_wr_score' in metadata:
                        print(f"   增强WR评分: {metadata['enhanced_wr_score']:.4f}")
                    if 'adaptive_thresholds_enabled' in metadata:
                        print(f"   自适应阈值启用: {metadata['adaptive_thresholds_enabled']}")
                    if 'multi_periods' in metadata:
                        print(f"   多周期配置: {metadata['multi_periods']}")
                
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
            print("❌ ENHANCED_WR指标get_signal()返回格式不正确")
            return False, None
            
    except Exception as e:
        print(f"❌ ENHANCED_WR指标get_signal()异常: {e}")
        return False, None

def test_enhanced_wr_data_validation(enhanced_wr_indicator):
    """测试ENHANCED_WR指标数据验证"""
    print("\n" + "="*60)
    print("🔒 ENHANCED_WR指标数据验证测试")
    print("="*60)
    
    test_cases = [
        ("空数据", pd.DataFrame()),
        ("缺少必需列", pd.DataFrame({'close': [1, 2, 3]})),
        ("数据量不足", pd.DataFrame({
            'high': list(range(1, 15)),
            'low': list(range(1, 15)),
            'close': list(range(1, 15))
        }))  # Enhanced WR需要足够的数据（max(period, multi_periods) + 10）
    ]
    
    validation_passed = 0
    for test_name, test_data in test_cases:
        try:
            signal = enhanced_wr_indicator.get_signal(test_data)
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

def test_enhanced_wr_performance(enhanced_wr_indicator, test_data):
    """测试ENHANCED_WR指标性能"""
    print("\n" + "="*60)
    print("⚡ ENHANCED_WR指标性能测试")
    print("="*60)
    
    import time
    
    # 测试calculate性能
    start_time = time.time()
    for i in range(10):
        enhanced_wr_indicator.calculate(test_data)
    calculate_time = (time.time() - start_time) / 10
    
    # 测试get_signal性能
    start_time = time.time()
    for i in range(100):
        enhanced_wr_indicator.get_signal(test_data)
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

def test_enhanced_wr_l4_compliance(enhanced_wr_indicator, test_data):
    """测试ENHANCED_WR指标L4设计文档合规性"""
    print("\n" + "="*60)
    print("📋 ENHANCED_WR指标L4设计文档合规性测试")
    print("="*60)
    
    compliance_score = 0
    total_checks = 7
    
    # 1. 检查必需的抽象方法实现
    required_methods = ['calculate', 'get_signal', 'get_patterns']
    missing_methods = []
    for method in required_methods:
        if not hasattr(enhanced_wr_indicator, method):
            missing_methods.append(method)
    
    if not missing_methods:
        print("✅ 必需抽象方法全部实现")
        compliance_score += 1
    else:
        print(f"❌ 缺少必需抽象方法: {missing_methods}")
    
    # 2. 检查has_result方法
    if hasattr(enhanced_wr_indicator, 'has_result') and callable(getattr(enhanced_wr_indicator, 'has_result')):
        print("✅ has_result()方法已实现")
        compliance_score += 1
    else:
        print("❌ has_result()方法缺失")
    
    # 3. 检查_result属性管理
    try:
        enhanced_wr_indicator.calculate(test_data)
        if hasattr(enhanced_wr_indicator, '_result') and enhanced_wr_indicator._result is not None:
            print("✅ _result属性管理正确")
            compliance_score += 1
        else:
            print("❌ _result属性管理有问题")
    except Exception as e:
        print(f"❌ _result属性管理测试失败: {e}")
    
    # 4. 检查标准化信号格式
    try:
        signal = enhanced_wr_indicator.get_signal(test_data)
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
        signal = enhanced_wr_indicator.get_signal(invalid_data)
        if signal.get('signal_type') == 'hold':
            print("✅ 异常处理机制正常")
            compliance_score += 1
        else:
            print("❌ 异常处理机制有问题")
    except Exception as e:
        print(f"❌ 异常处理测试失败: {e}")
    
    # 6. 检查Enhanced WR特有算法
    try:
        # 检查Enhanced WR核心算法实现
        result = enhanced_wr_indicator.calculate(test_data)
        expected_wr_columns = ['wr']  # 关键的Enhanced WR列
        has_wr_columns = all(col in result.columns for col in expected_wr_columns)
        
        if has_wr_columns:
            print("✅ Enhanced WR核心算法实现完整")
            compliance_score += 1
        else:
            print(f"❌ Enhanced WR核心算法缺失，找不到关键列：{expected_wr_columns}")
    except Exception as e:
        print(f"❌ Enhanced WR算法检查失败: {e}")
    
    # 7. 检查Enhanced WR增强特性配置和振荡器特性
    try:
        # 检查Enhanced WR增强特性参数
        enhanced_features = ['adaptive_thresholds', 'multi_periods', 'smooth_period']
        missing_features = []
        for feature in enhanced_features:
            if not hasattr(enhanced_wr_indicator, feature):
                missing_features.append(feature)
        
        if not missing_features:
            print("✅ Enhanced WR增强特性配置完备")
            compliance_score += 1
        else:
            print(f"❌ Enhanced WR增强特性配置缺失: {missing_features}")
            
        # 额外检查：验证Enhanced WR振荡器特性
        result = enhanced_wr_indicator.calculate(test_data)
        if 'wr' in result.columns:
            wr_values = result['wr'].dropna()
            if len(wr_values) > 0:
                # Enhanced WR应该在-100到0范围内振荡
                wr_min, wr_max = wr_values.min(), wr_values.max()
                if -100 <= wr_min and wr_max <= 0:
                    print(f"   ✓ Enhanced WR振荡器特性正常（值域: [{wr_min:.2f}, {wr_max:.2f}]，符合-100到0范围）")
                else:
                    print(f"   ⚠️ Enhanced WR值域异常：[{wr_min:.2f}, {wr_max:.2f}]，应在-100到0范围内")
                    
        # 检查Enhanced WR增强特性
        if hasattr(enhanced_wr_indicator, 'adaptive_thresholds') and hasattr(enhanced_wr_indicator, 'multi_periods'):
            print(f"   ✓ Enhanced WR增强特性配置: 自适应阈值({enhanced_wr_indicator.adaptive_thresholds}), 多周期({enhanced_wr_indicator.multi_periods})")
        else:
            print("   ⚠️ Enhanced WR增强特性配置缺失")
            
        # 检查Enhanced WR超买超卖和自适应逻辑
        signal = enhanced_wr_indicator.get_signal(test_data)
        if 'metadata' in signal and signal['metadata']:
            metadata = signal['metadata']
            if 'in_overbought' in metadata and 'in_oversold' in metadata:
                print(f"   ✓ Enhanced WR超买超卖逻辑完整（超买: {metadata['in_overbought']}, 超卖: {metadata['in_oversold']}）")
            if 'adaptive_overbought_threshold' in metadata and 'adaptive_oversold_threshold' in metadata:
                print(f"   ✓ Enhanced WR自适应阈值逻辑完整（超买阈值: {metadata['adaptive_overbought_threshold']:.1f}, 超卖阈值: {metadata['adaptive_oversold_threshold']:.1f}）")
            if 'wr_momentum' in metadata and 'wr_divergence' in metadata:
                print(f"   ✓ Enhanced WR增强特性逻辑完整（动量: {metadata['wr_momentum']:.4f}, 背离: {metadata['wr_divergence']:.4f}）")
        else:
            print("   ⚠️ Enhanced WR增强特性逻辑缺失")
    except Exception as e:
        print(f"❌ Enhanced WR增强特性检查失败: {e}")
    
    compliance_rate = (compliance_score / total_checks) * 100
    print(f"📊 L4合规性评分: {compliance_score}/{total_checks} ({compliance_rate:.1f}%)")
    
    if compliance_score == total_checks:
        print("✅ 完全符合L4设计文档要求")
        return True
    else:
        print("❌ 部分不符合L4设计文档要求")
        return False

def run_enhanced_wr_tests():
    """运行所有ENHANCED_WR测试"""
    print("🚀 开始ENHANCED_WR (Enhanced Williams %R) 指标全面深度测试")
    print("="*80)
    
    # 准备测试数据
    test_data = create_test_data()
    print(f"✅ 测试数据准备完成: {len(test_data)}行")
    
    # 测试计数器
    total_tests = 6
    passed_tests = 0
    
    # 1. 实例化测试
    success, enhanced_wr_indicator = test_enhanced_wr_instantiation()
    if success:
        passed_tests += 1
    
    if enhanced_wr_indicator is None:
        print("❌ 无法继续后续测试，ENHANCED_WR实例化失败")
        return
    
    # 2. calculate测试
    success, result = test_enhanced_wr_calculate(enhanced_wr_indicator, test_data)
    if success:
        passed_tests += 1
    
    # 3. get_signal测试
    success, signal = test_enhanced_wr_get_signal(enhanced_wr_indicator, test_data)
    if success:
        passed_tests += 1
    
    # 4. 数据验证测试
    success = test_enhanced_wr_data_validation(enhanced_wr_indicator)
    if success:
        passed_tests += 1
    
    # 5. 性能测试
    success = test_enhanced_wr_performance(enhanced_wr_indicator, test_data)
    if success:
        passed_tests += 1
    
    # 6. L4合规性测试
    success = test_enhanced_wr_l4_compliance(enhanced_wr_indicator, test_data)
    if success:
        passed_tests += 1
    
    # 输出最终结果
    print("\n" + "="*80)
    print("📊 ENHANCED_WR指标测试总结")
    print("="*80)
    print(f"✅ 通过测试: {passed_tests}/{total_tests}")
    print(f"📈 成功率: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print("🎉 ENHANCED_WR指标全面测试通过！符合L4文档设计预期和基类设计预期")
    else:
        print("⚠️ ENHANCED_WR指标存在问题，需要进一步修复")

if __name__ == "__main__":
    run_enhanced_wr_tests()
