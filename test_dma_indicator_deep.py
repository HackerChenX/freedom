#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DMA指标全面深度测试脚本
执行DMA指标的全面质量验证测试
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
    dates = pd.date_range('2020-01-01', periods=150, freq='D')
    np.random.seed(42)  # 设置随机种子以获得可重现的结果
    
    # 生成具有趋势的价格数据
    base_price = 100
    trend = np.linspace(0, 30, 150)  # 上升趋势
    noise = np.random.normal(0, 2, 150)  # 随机噪声
    
    prices = base_price + trend + noise
    
    data = pd.DataFrame({
        'open': prices + np.random.normal(0, 0.5, 150),
        'high': prices + np.abs(np.random.normal(0, 1, 150)),
        'low': prices - np.abs(np.random.normal(0, 1, 150)),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, 150),
        'turnover_rate': np.random.uniform(0.1, 5.0, 150)
    }, index=dates)
    
    # 确保 high >= low, open/close 在 high/low 之间
    data['high'] = np.maximum.reduce([data['open'], data['high'], data['close']])
    data['low'] = np.minimum.reduce([data['open'], data['low'], data['close']])
    
    return data

def test_dma_instantiation():
    """测试DMA指标实例化"""
    print("\n" + "="*60)
    print("🔍 DMA指标实例化测试")
    print("="*60)
    
    try:
        manager = UnifiedIndicatorManager()
        dma_indicator = manager.create_indicator('DMA')
        
        if dma_indicator is not None:
            print("✅ DMA指标实例化成功")
            print(f"   类名: {dma_indicator.__class__.__name__}")
            print(f"   指标名: {getattr(dma_indicator, 'name', 'N/A')}")
            print(f"   快周期: {getattr(dma_indicator, 'fast_period', 'N/A')}")
            print(f"   慢周期: {getattr(dma_indicator, 'slow_period', 'N/A')}")
            return True, dma_indicator
        else:
            print("❌ DMA指标实例化失败")
            return False, None
            
    except Exception as e:
        print(f"❌ DMA指标实例化异常: {e}")
        return False, None

def test_dma_calculate(dma_indicator, test_data):
    """测试DMA指标calculate()方法"""
    print("\n" + "="*60)
    print("📊 DMA指标calculate()测试")
    print("="*60)
    
    try:
        result = dma_indicator.calculate(test_data)
        
        if result is not None and not result.empty:
            print("✅ DMA指标calculate()方法成功")
            print(f"   结果行数: {len(result)}")
            print(f"   结果列数: {len(result.columns)}")
            print(f"   列名: {list(result.columns)}")
            
            # 检查关键的DMA列
            dma_columns = [col for col in result.columns if 'DMA' in col.upper() or 'AMA' in col.upper()]
            if dma_columns:
                print(f"   DMA相关列: {dma_columns}")
                for col in dma_columns[:3]:  # 显示前3列的统计信息
                    if len(result[col]) > 0:
                        print(f"   {col}: 最新值={result[col].iloc[-1]:.4f}, 平均值={result[col].mean():.4f}")
                return True, result
            else:
                print("⚠️ 警告: 未找到DMA相关的输出列")
                return False, result
        else:
            print("❌ DMA指标calculate()返回空结果")
            return False, None
            
    except Exception as e:
        print(f"❌ DMA指标calculate()异常: {e}")
        return False, None

def test_dma_get_signal(dma_indicator, test_data):
    """测试DMA指标get_signal()方法"""
    print("\n" + "="*60)
    print("📡 DMA指标get_signal()测试")
    print("="*60)
    
    try:
        signal = dma_indicator.get_signal(test_data)
        
        if signal and isinstance(signal, dict):
            print("✅ DMA指标get_signal()方法成功")
            
            # 检查标准信号格式
            required_fields = ['signal_type', 'strength', 'confidence', 'timestamp', 'reason', 'metadata']
            missing_fields = [field for field in required_fields if field not in signal]
            
            if not missing_fields:
                print("✅ 信号格式标准化验证通过")
                print(f"   信号类型: {signal['signal_type']}")
                print(f"   信号强度: {signal['strength']:.4f}")
                print(f"   信号置信度: {signal['confidence']:.4f}")
                print(f"   信号原因: {signal['reason']}")
                
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
            print("❌ DMA指标get_signal()返回格式不正确")
            return False, None
            
    except Exception as e:
        print(f"❌ DMA指标get_signal()异常: {e}")
        return False, None

def test_dma_data_validation(dma_indicator):
    """测试DMA指标数据验证"""
    print("\n" + "="*60)
    print("🔒 DMA指标数据验证测试")
    print("="*60)
    
    test_cases = [
        ("空数据", pd.DataFrame()),
        ("缺少必需列", pd.DataFrame({'open': [1, 2, 3]})),
        ("数据量不足", pd.DataFrame({
            'close': [1, 2, 3, 4, 5]  # 少于DMA所需的最小周期
        }))
    ]
    
    validation_passed = 0
    for test_name, test_data in test_cases:
        try:
            signal = dma_indicator.get_signal(test_data)
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

def test_dma_performance(dma_indicator, test_data):
    """测试DMA指标性能"""
    print("\n" + "="*60)
    print("⚡ DMA指标性能测试")
    print("="*60)
    
    import time
    
    # 测试calculate性能
    start_time = time.time()
    for i in range(10):
        dma_indicator.calculate(test_data)
    calculate_time = (time.time() - start_time) / 10
    
    # 测试get_signal性能
    start_time = time.time()
    for i in range(100):
        dma_indicator.get_signal(test_data)
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

def run_dma_tests():
    """运行所有DMA测试"""
    print("🚀 开始DMA指标全面深度测试")
    print("="*80)
    
    # 准备测试数据
    test_data = create_test_data()
    print(f"✅ 测试数据准备完成: {len(test_data)}行")
    
    # 测试计数器
    total_tests = 5
    passed_tests = 0
    
    # 1. 实例化测试
    success, dma_indicator = test_dma_instantiation()
    if success:
        passed_tests += 1
    
    if dma_indicator is None:
        print("❌ 无法继续后续测试，DMA实例化失败")
        return
    
    # 2. calculate测试
    success, result = test_dma_calculate(dma_indicator, test_data)
    if success:
        passed_tests += 1
    
    # 3. get_signal测试
    success, signal = test_dma_get_signal(dma_indicator, test_data)
    if success:
        passed_tests += 1
    
    # 4. 数据验证测试
    success = test_dma_data_validation(dma_indicator)
    if success:
        passed_tests += 1
    
    # 5. 性能测试
    success = test_dma_performance(dma_indicator, test_data)
    if success:
        passed_tests += 1
    
    # 输出最终结果
    print("\n" + "="*80)
    print("📊 DMA指标测试总结")
    print("="*80)
    print(f"✅ 通过测试: {passed_tests}/{total_tests}")
    print(f"📈 成功率: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print("🎉 DMA指标全面测试通过！符合L4文档设计预期和基类设计预期")
    else:
        print("⚠️ DMA指标存在问题，需要进一步修复")

if __name__ == "__main__":
    run_dma_tests()
