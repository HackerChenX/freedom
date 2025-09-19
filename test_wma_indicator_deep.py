#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WMA指标全面深度测试脚本
执行WMA指标的全面质量验证测试
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
    
    # 生成具有趋势的价格数据
    base_price = 100
    trend = np.linspace(0, 20, 100)  # 上升趋势
    noise = np.random.normal(0, 2, 100)  # 随机噪声
    
    prices = base_price + trend + noise
    
    data = pd.DataFrame({
        'open': prices + np.random.normal(0, 0.5, 100),
        'high': prices + np.abs(np.random.normal(0, 1, 100)),
        'low': prices - np.abs(np.random.normal(0, 1, 100)),
        'close': prices,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    return data

def test_wma_instantiation():
    """测试WMA指标实例创建"""
    try:
        manager = UnifiedIndicatorManager()
        
        # 测试WMA指标创建
        wma_indicator = manager.create_indicator("WMA")
        
        if wma_indicator is not None:
            print("✅ WMA实例创建测试: 通过")
            return True, {"indicator_type": type(wma_indicator).__name__}
        else:
            print("❌ WMA实例创建测试: 失败")
            print("   - 无法创建WMA指标实例")
            return False, {}
            
    except Exception as e:
        print("❌ WMA实例创建测试: 失败")
        print(f"   - 实例创建失败: {e}")
        return False, {}

def test_wma_calculate():
    """测试WMA指标calculate()方法"""
    try:
        manager = UnifiedIndicatorManager()
        wma_indicator = manager.create_indicator("WMA")
        
        if wma_indicator is None:
            print("❌ WMA指标calculate()测试: 失败")
            print("   - calculate()测试失败: 未注册的指标: WMA")
            return False, {}
        
        # 创建测试数据
        test_data = create_test_data()
        
        # 执行计算
        result = wma_indicator.calculate(test_data)
        
        if result is not None and not result.empty:
            # 检查结果中是否有WMA相关列
            wma_columns = [col for col in result.columns if 'WMA' in col.upper() or 'wma' in col]
            
            if wma_columns:
                print("✅ WMA指标calculate()测试: 通过")
                return True, {
                    "result_shape": result.shape,
                    "wma_columns": wma_columns[:3],  # 只显示前3个
                    "has_signals": any(col in result.columns for col in ['buy_signal', 'sell_signal'])
                }
            else:
                print("❌ WMA指标calculate()测试: 失败")
                print(f"   - 结果中未找到WMA相关列，实际列: {list(result.columns)[:5]}")
                return False, {}
        else:
            print("❌ WMA指标calculate()测试: 失败")
            print("   - calculate()方法返回空结果")
            return False, {}
            
    except Exception as e:
        print("❌ WMA指标calculate()测试: 失败")
        print(f"   - calculate()执行失败: {e}")
        return False, {}

def test_wma_get_signal():
    """测试WMA指标get_signal()方法"""
    try:
        manager = UnifiedIndicatorManager()
        wma_indicator = manager.create_indicator("WMA")
        
        if wma_indicator is None:
            print("❌ WMA指标get_signal()测试: 失败")
            print("   - get_signal()测试失败: 未注册的指标: WMA")
            return False, {}
        
        # 创建测试数据
        test_data = create_test_data()
        
        # 执行信号生成
        signal = wma_indicator.get_signal(test_data)
        
        # 验证信号格式
        required_fields = ['signal_type', 'strength', 'confidence', 'timestamp', 'reason', 'metadata']
        
        if all(field in signal for field in required_fields):
            # 验证信号值的有效性
            signal_type_valid = signal['signal_type'] in ['buy', 'sell', 'hold']
            strength_valid = 0 <= signal['strength'] <= 1
            confidence_valid = 0 <= signal['confidence'] <= 1
            
            if signal_type_valid and strength_valid and confidence_valid:
                print("✅ WMA指标get_signal()测试: 通过")
                return True, {
                    "signal_type": signal['signal_type'],
                    "strength": round(signal['strength'], 3),
                    "confidence": round(signal['confidence'], 3),
                    "reason": signal['reason'][:50] + "..." if len(signal['reason']) > 50 else signal['reason']
                }
            else:
                print("❌ WMA指标get_signal()测试: 失败")
                print(f"   - 信号值无效: type={signal_type_valid}, strength={strength_valid}, confidence={confidence_valid}")
                return False, {}
        else:
            missing_fields = [field for field in required_fields if field not in signal]
            print("❌ WMA指标get_signal()测试: 失败")
            print(f"   - 信号格式不完整，缺少字段: {missing_fields}")
            return False, {}
            
    except Exception as e:
        print("❌ WMA指标get_signal()测试: 失败")
        print(f"   - get_signal()执行失败: {e}")
        return False, {}

def test_wma_data_validation():
    """测试WMA指标数据验证"""
    try:
        manager = UnifiedIndicatorManager()
        wma_indicator = manager.create_indicator("WMA")
        
        if wma_indicator is None:
            print("❌ WMA指标数据验证测试: 失败")
            print("   - 数据验证测试失败: 未注册的指标: WMA")
            return False, {}
        
        # 测试空数据
        empty_data = pd.DataFrame()
        signal_empty = wma_indicator.get_signal(empty_data)
        
        # 测试不足数据
        insufficient_data = create_test_data().head(5)  # 只有5行数据
        signal_insufficient = wma_indicator.get_signal(insufficient_data)
        
        # 测试缺少必要列的数据
        invalid_data = pd.DataFrame({'invalid_column': [1, 2, 3, 4, 5]})
        signal_invalid = wma_indicator.get_signal(invalid_data)
        
        # 验证所有情况都返回默认信号
        default_checks = [
            signal_empty['signal_type'] == 'hold',
            signal_insufficient['signal_type'] == 'hold',
            signal_invalid['signal_type'] == 'hold'
        ]
        
        if all(default_checks):
            print("✅ WMA指标数据验证测试: 通过")
            return True, {
                "empty_data_handled": True,
                "insufficient_data_handled": True,
                "invalid_data_handled": True
            }
        else:
            print("❌ WMA指标数据验证测试: 失败")
            print(f"   - 数据验证失败: {default_checks}")
            return False, {}
            
    except Exception as e:
        print("❌ WMA指标数据验证测试: 失败")
        print(f"   - 数据验证测试失败: {e}")
        return False, {}

def test_wma_performance():
    """测试WMA指标性能"""
    try:
        manager = UnifiedIndicatorManager()
        wma_indicator = manager.create_indicator("WMA")
        
        if wma_indicator is None:
            print("❌ WMA指标性能测试: 失败")
            print("   - 性能测试失败: 未注册的指标: WMA")
            return False, {}
        
        # 创建大量测试数据
        large_data = create_test_data()
        # 扩展到更大的数据集
        for _ in range(10):
            large_data = pd.concat([large_data, create_test_data()], ignore_index=True)
        
        # 测量计算时间
        start_time = datetime.now()
        result = wma_indicator.calculate(large_data)
        calc_time = (datetime.now() - start_time).total_seconds()
        
        # 测量信号生成时间
        start_time = datetime.now()
        signal = wma_indicator.get_signal(large_data)
        signal_time = (datetime.now() - start_time).total_seconds()
        
        # 性能基准（秒）
        calc_threshold = 2.0  # 2秒
        signal_threshold = 0.5  # 0.5秒
        
        calc_pass = calc_time < calc_threshold
        signal_pass = signal_time < signal_threshold
        
        if calc_pass and signal_pass:
            print("✅ WMA指标性能测试: 通过")
            return True, {
                "data_size": len(large_data),
                "calc_time": round(calc_time, 3),
                "signal_time": round(signal_time, 3),
                "calc_threshold": calc_threshold,
                "signal_threshold": signal_threshold
            }
        else:
            print("❌ WMA指标性能测试: 失败")
            print(f"   - 性能不达标: calc_time={calc_time:.3f}s (threshold={calc_threshold}s), signal_time={signal_time:.3f}s (threshold={signal_threshold}s)")
            return False, {}
            
    except Exception as e:
        print("❌ WMA指标性能测试: 失败")
        print(f"   - 性能测试失败: {e}")
        return False, {}

def main():
    """主测试函数"""
    print("🎯 开始WMA指标全面测试")
    print("=" * 60)
    
    # 测试项目列表
    tests = [
        ("🔍 测试WMA指标实例创建...", test_wma_instantiation),
        ("🔍 测试WMA指标calculate()方法...", test_wma_calculate),
        ("🔍 测试WMA指标get_signal()方法...", test_wma_get_signal),
        ("🔍 测试WMA指标数据验证...", test_wma_data_validation),
        ("🔍 测试WMA指标性能...", test_wma_performance),
    ]
    
    # 执行测试
    total_tests = len(tests)
    passed_tests = 0
    test_details = []
    
    for test_name, test_func in tests:
        print(test_name)
        try:
            success, details = test_func()
            if success:
                passed_tests += 1
            test_details.append((test_name, success, details))
        except Exception as e:
            print(f"❌ {test_name}: 测试执行异常: {e}")
            test_details.append((test_name, False, {"error": str(e)}))
    
    # 输出测试结果汇总
    print("\n" + "=" * 60)
    print("🎯 WMA指标测试结果汇总:")
    print(f"   总测试数: {total_tests}")
    print(f"   通过测试: {passed_tests}")
    print(f"   失败测试: {total_tests - passed_tests}")
    print(f"   成功率: {passed_tests/total_tests*100:.1f}%")
    
    # 详细结果
    if passed_tests == total_tests:
        print("🎉 WMA指标100%通过所有测试，符合L4文档设计预期！")
    else:
        print("⚠️ WMA指标未能100%通过测试，需要进一步修复")
        
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
