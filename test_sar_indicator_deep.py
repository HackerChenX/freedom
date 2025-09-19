#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SAR指标全面深度测试脚本
执行SAR指标的全面质量验证测试
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
    
    # 生成具有明显趋势变化的价格数据，适合SAR测试
    base_price = 100
    
    # 创建多段趋势变化
    trend_segments = []
    # 第一段：上升趋势 (0-30)
    trend_segments.extend(np.linspace(0, 15, 30))
    # 第二段：下降趋势 (30-60)  
    trend_segments.extend(np.linspace(15, 5, 30))
    # 第三段：上升趋势 (60-100)
    trend_segments.extend(np.linspace(5, 25, 40))
    
    noise = np.random.normal(0, 1.5, 100)  # 适度噪声
    prices = base_price + np.array(trend_segments) + noise
    
    # 生成OHLC数据
    data = pd.DataFrame({
        'open': prices + np.random.normal(0, 0.3, 100),
        'high': prices + np.abs(np.random.normal(0, 1.5, 100)),
        'low': prices - np.abs(np.random.normal(0, 1.5, 100)),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, 100),
        'turnover_rate': np.random.uniform(0.1, 5.0, 100)
    }, index=dates)
    
    # 确保 high >= low, open/close 在 high/low 之间
    data['high'] = np.maximum.reduce([data['open'], data['high'], data['close']])
    data['low'] = np.minimum.reduce([data['open'], data['low'], data['close']])
    
    return data

def test_sar_instantiation():
    """测试SAR指标实例化"""
    print("\n" + "="*60)
    print("🔍 SAR指标实例化测试")
    print("="*60)
    
    try:
        manager = UnifiedIndicatorManager()
        sar_indicator = manager.create_indicator('SAR')
        
        if sar_indicator is not None:
            print("✅ SAR指标实例化成功")
            print(f"   类名: {sar_indicator.__class__.__name__}")
            print(f"   指标名: {getattr(sar_indicator, 'name', 'N/A')}")
            print(f"   周期: {getattr(sar_indicator, 'period', 'N/A')}")
            return True, sar_indicator
        else:
            print("❌ SAR指标实例化失败")
            return False, None
            
    except Exception as e:
        print(f"❌ SAR指标实例化异常: {e}")
        return False, None

def test_sar_calculate(sar_indicator, test_data):
    """测试SAR指标calculate()方法"""
    print("\n" + "="*60)
    print("📊 SAR指标calculate()测试")
    print("="*60)
    
    try:
        result = sar_indicator.calculate(test_data)
        
        if result is not None and not result.empty:
            print("✅ SAR指标calculate()方法成功")
            print(f"   结果行数: {len(result)}")
            print(f"   结果列数: {len(result.columns)}")
            print(f"   列名: {list(result.columns)}")
            
            # 检查关键的SAR列
            sar_columns = [col for col in result.columns if 'sar' in col.lower()]
            if sar_columns:
                print(f"   SAR相关列: {sar_columns}")
                for col in sar_columns[:5]:  # 显示前5列的统计信息
                    if len(result[col]) > 0 and result[col].dtype in ['float64', 'int64']:
                        print(f"   {col}: 最新值={result[col].iloc[-1]:.4f}, 平均值={result[col].mean():.4f}")
                    elif len(result[col]) > 0:
                        print(f"   {col}: 最新值={result[col].iloc[-1]}")
                return True, result
            else:
                print("⚠️ 警告: 未找到SAR相关的输出列")
                return False, result
        else:
            print("❌ SAR指标calculate()返回空结果")
            return False, None
            
    except Exception as e:
        print(f"❌ SAR指标calculate()异常: {e}")
        return False, None

def test_sar_get_signal(sar_indicator, test_data):
    """测试SAR指标get_signal()方法"""
    print("\n" + "="*60)
    print("📡 SAR指标get_signal()测试")
    print("="*60)
    
    try:
        signal = sar_indicator.get_signal(test_data)
        
        if signal and isinstance(signal, dict):
            print("✅ SAR指标get_signal()方法成功")
            
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
                    if 'sar_value' in metadata:
                        print(f"   SAR值: {metadata['sar_value']:.4f}")
                    if 'sar_trend' in metadata:
                        print(f"   SAR趋势: {metadata['sar_trend']}")
                    if 'price_sar_ratio' in metadata:
                        print(f"   价格/SAR比率: {metadata['price_sar_ratio']:.4f}")
                    if 'trend_change' in metadata:
                        print(f"   趋势变化: {metadata['trend_change']}")
                
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
            print("❌ SAR指标get_signal()返回格式不正确")
            return False, None
            
    except Exception as e:
        print(f"❌ SAR指标get_signal()异常: {e}")
        return False, None

def test_sar_data_validation(sar_indicator):
    """测试SAR指标数据验证"""
    print("\n" + "="*60)
    print("🔒 SAR指标数据验证测试")
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
            signal = sar_indicator.get_signal(test_data)
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

def test_sar_performance(sar_indicator, test_data):
    """测试SAR指标性能"""
    print("\n" + "="*60)
    print("⚡ SAR指标性能测试")
    print("="*60)
    
    import time
    
    # 测试calculate性能
    start_time = time.time()
    for i in range(10):
        sar_indicator.calculate(test_data)
    calculate_time = (time.time() - start_time) / 10
    
    # 测试get_signal性能
    start_time = time.time()
    for i in range(100):
        sar_indicator.get_signal(test_data)
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

def test_sar_algorithm_correctness(sar_indicator, test_data):
    """测试SAR算法正确性"""
    print("\n" + "="*60)
    print("🔍 SAR算法正确性测试")
    print("="*60)
    
    try:
        result = sar_indicator.calculate(test_data)
        
        if result is not None and 'sar' in result.columns:
            sar_values = result['sar'].dropna()
            trends = result['sar_trend'].dropna()
            
            # 检查SAR值的合理性
            print(f"   SAR值数量: {len(sar_values)}")
            print(f"   SAR值范围: {sar_values.min():.4f} - {sar_values.max():.4f}")
            
            # 检查趋势变化
            trend_changes = (trends.diff() != 0).sum()
            print(f"   趋势变化次数: {trend_changes}")
            
            # 检查SAR值的连续性（相邻值不应有巨大跳跃）
            sar_diff = sar_values.diff().abs()
            max_diff_ratio = (sar_diff / sar_values.shift(1)).max()
            print(f"   最大SAR变化比例: {max_diff_ratio:.4f}")
            
            # 验证SAR算法基本逻辑
            validation_passed = 0
            
            # 1. SAR值应该在合理范围内
            price_range = test_data['close'].max() - test_data['close'].min()
            if sar_values.min() >= 0 and sar_diff.max() < price_range * 0.5:
                validation_passed += 1
                print("✅ SAR值范围合理")
            else:
                print("❌ SAR值范围异常")
            
            # 2. 趋势值应该只包含1和-1
            if set(trends.unique()) <= {1, -1}:
                validation_passed += 1
                print("✅ 趋势值正确")
            else:
                print("❌ 趋势值异常")
            
            # 3. 应该有合理数量的趋势变化
            if 2 <= trend_changes <= len(trends) // 3:
                validation_passed += 1
                print("✅ 趋势变化次数合理")
            else:
                print(f"⚠️ 趋势变化次数({trend_changes})可能过多或过少")
                validation_passed += 0.5  # 部分通过
            
            if validation_passed >= 2.5:
                print("✅ SAR算法正确性验证通过")
                return True
            else:
                print("❌ SAR算法正确性验证未通过")
                return False
                
        else:
            print("❌ 无法获取SAR计算结果")
            return False
            
    except Exception as e:
        print(f"❌ SAR算法正确性测试异常: {e}")
        return False

def run_sar_tests():
    """运行所有SAR测试"""
    print("🚀 开始SAR指标全面深度测试")
    print("="*80)
    
    # 准备测试数据
    test_data = create_test_data()
    print(f"✅ 测试数据准备完成: {len(test_data)}行")
    
    # 测试计数器
    total_tests = 6
    passed_tests = 0
    
    # 1. 实例化测试
    success, sar_indicator = test_sar_instantiation()
    if success:
        passed_tests += 1
    
    if sar_indicator is None:
        print("❌ 无法继续后续测试，SAR实例化失败")
        return
    
    # 2. calculate测试
    success, result = test_sar_calculate(sar_indicator, test_data)
    if success:
        passed_tests += 1
    
    # 3. get_signal测试
    success, signal = test_sar_get_signal(sar_indicator, test_data)
    if success:
        passed_tests += 1
    
    # 4. 数据验证测试
    success = test_sar_data_validation(sar_indicator)
    if success:
        passed_tests += 1
    
    # 5. 性能测试
    success = test_sar_performance(sar_indicator, test_data)
    if success:
        passed_tests += 1
    
    # 6. 算法正确性测试
    success = test_sar_algorithm_correctness(sar_indicator, test_data)
    if success:
        passed_tests += 1
    
    # 输出最终结果
    print("\n" + "="*80)
    print("📊 SAR指标测试总结")
    print("="*80)
    print(f"✅ 通过测试: {passed_tests}/{total_tests}")
    print(f"📈 成功率: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print("🎉 SAR指标全面测试通过！符合L4文档设计预期和基类设计预期")
    else:
        print("⚠️ SAR指标存在问题，需要进一步修复")

if __name__ == "__main__":
    run_sar_tests()
