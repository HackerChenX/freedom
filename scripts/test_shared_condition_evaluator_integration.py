#!/usr/bin/env python3
"""
共享条件评估器集成测试脚本

测试共享条件评估器与统一指标引擎的集成，以及与买点分析数据的兼容性
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
import time

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from analysis.engines.shared_condition_evaluator import SharedConditionEvaluator
from analysis.engines.unified_indicator_engine import UnifiedIndicatorEngine
from utils.logger import get_logger

logger = get_logger(__name__)


def create_sample_stock_data(length=100):
    """
    创建示例股票数据
    
    Args:
        length: 数据长度
        
    Returns:
        Dict[str, np.ndarray]: 股票数据字典
    """
    # 生成模拟股票数据
    np.random.seed(42)  # 确保结果可重现
    
    # 基础价格数据
    base_price = 10.0
    price_changes = np.random.normal(0, 0.02, length)
    close_prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = close_prices[-1] * (1 + change)
        close_prices.append(max(new_price, 0.1))  # 确保价格为正
    
    close = np.array(close_prices)
    
    # 生成开盘价、最高价、最低价
    open_price = close * (1 + np.random.normal(0, 0.005, length))
    high = np.maximum(close, open_price) * (1 + np.abs(np.random.normal(0, 0.01, length)))
    low = np.minimum(close, open_price) * (1 - np.abs(np.random.normal(0, 0.01, length)))
    
    # 生成成交量
    volume = np.random.lognormal(10, 0.5, length).astype(int)
    
    return {
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }


def calculate_indicators_with_unified_engine(stock_data):
    """
    使用统一指标引擎计算技术指标
    
    Args:
        stock_data: 股票数据
        
    Returns:
        Dict[str, np.ndarray]: 包含指标的数据字典
    """
    engine = UnifiedIndicatorEngine()
    
    # 计算各种技术指标
    indicators = {}
    
    # 移动平均线
    indicators['ma5'] = engine.calculate_ma(stock_data['close'], 5)
    indicators['ma10'] = engine.calculate_ma(stock_data['close'], 10)
    indicators['ma20'] = engine.calculate_ma(stock_data['close'], 20)
    
    # EMA
    indicators['ema5'] = engine.calculate_ema(stock_data['close'], 5)
    indicators['ema10'] = engine.calculate_ema(stock_data['close'], 10)
    
    # MACD
    macd_result = engine.calculate_macd(stock_data['close'])
    indicators['macd'] = macd_result['macd']
    indicators['dif'] = macd_result['dif']
    indicators['dea'] = macd_result['dea']
    
    # KDJ
    kdj_result = engine.calculate_kdj(stock_data['high'], stock_data['low'], stock_data['close'])
    indicators['kdj_k'] = kdj_result['k']
    indicators['kdj_d'] = kdj_result['d']
    indicators['kdj_j'] = kdj_result['j']
    
    # RSI
    indicators['rsi'] = engine.calculate_rsi(stock_data['close'])
    
    # 成交量移动平均
    indicators['vol5'] = engine.calculate_ma(stock_data['volume'], 5)
    indicators['vol10'] = engine.calculate_ma(stock_data['volume'], 10)
    
    # 合并原始数据和指标数据
    result = {**stock_data, **indicators}
    
    return result


def simulate_buypoint_analysis_data(data, date_idx):
    """
    模拟买点分析数据
    
    Args:
        data: 股票和指标数据
        date_idx: 日期索引
        
    Returns:
        Dict[str, Any]: 买点分析数据
    """
    # 模拟买点分析的逻辑判断
    close = data['close']
    ma5 = data['ma5']
    ma10 = data['ma10']
    ma20 = data['ma20']
    volume = data['volume']
    vol5 = data['vol5']
    rsi = data['rsi']
    macd = data['macd']
    dif = data['dif']
    kdj_k = data['kdj_k']
    
    # 触及均线判断
    touch_ma10 = abs(data['low'][date_idx] / ma10[date_idx] - 1) < 0.01
    touch_ma20 = abs(data['low'][date_idx] / ma20[date_idx] - 1) < 0.01
    touch_ma = touch_ma10 or touch_ma20
    
    # 价格企稳
    if date_idx > 0:
        price_stable = (close[date_idx] > close[date_idx-1] and 
                       data['low'][date_idx] > data['low'][date_idx-1] * 0.995)
    else:
        price_stable = False
    
    # 均线上移
    if date_idx > 0:
        ma_up = (abs(ma5[date_idx] / ma10[date_idx] - 1) < 0.01 and 
                ma5[date_idx] > ma5[date_idx-1] and 
                ma10[date_idx] > ma10[date_idx-1])
    else:
        ma_up = False
    
    # 资金流入（简化版）
    money_in = volume[date_idx] > vol5[date_idx] * 1.2
    
    # K线形态（简化版）
    open_price = data['open'][date_idx]
    high_price = data['high'][date_idx]
    low_price = data['low'][date_idx]
    close_price = close[date_idx]
    
    xsmall = abs(open_price - close_price) / close_price < 0.01
    kpattern = xsmall
    
    # 成交量缩量
    vol_shrink = volume[date_idx] < vol5[date_idx] * 0.8
    
    # MACD金叉
    if date_idx > 0:
        macd_gold = (macd[date_idx-1] < 0 and macd[date_idx] > 0)
    else:
        macd_gold = False
    
    # 技术指标变化
    if date_idx > 0:
        dif_up = dif[date_idx] > dif[date_idx-1]
        k_up = kdj_k[date_idx] > kdj_k[date_idx-1]
    else:
        dif_up = False
        k_up = False
    
    # 构建买点分析数据
    buypoint_data = {
        **data,  # 包含所有原始数据和指标
        'touch_ma': touch_ma,
        'touch_ma10': touch_ma10,
        'touch_ma20': touch_ma20,
        'price_stable': price_stable,
        'ma_up': ma_up,
        'money_in': money_in,
        'kpattern': kpattern,
        'vol_shrink': vol_shrink,
        'macd_gold': macd_gold,
        'dif_up': dif_up,
        'k_up': k_up,
        'xc': money_in,  # 吸筹信号
    }
    
    return buypoint_data


def test_basic_conditions():
    """测试基础条件评估"""
    print("\n=== 测试基础条件评估 ===")
    
    evaluator = SharedConditionEvaluator()
    stock_data = create_sample_stock_data(50)
    data = calculate_indicators_with_unified_engine(stock_data)
    
    # 测试价格条件
    conditions = [
        {
            'type': 'basic',
            'field': 'close',
            'operator': '>',
            'reference_field': 'ma5'
        },
        {
            'type': 'basic',
            'field': 'volume',
            'operator': '>',
            'reference_field': 'vol5'
        }
    ]
    
    success_count = 0
    for i, condition in enumerate(conditions):
        result = evaluator.evaluate_condition(condition, data, date_idx=-1)
        print(f"条件 {i+1}: {result}")
        if isinstance(result, bool):
            success_count += 1
    
    print(f"基础条件测试: {success_count}/{len(conditions)} 通过")
    return success_count == len(conditions)


def test_indicator_conditions():
    """测试指标条件评估"""
    print("\n=== 测试指标条件评估 ===")
    
    evaluator = SharedConditionEvaluator()
    stock_data = create_sample_stock_data(50)
    data = calculate_indicators_with_unified_engine(stock_data)
    
    # 测试指标条件
    conditions = [
        {
            'type': 'indicator',
            'indicator': 'ma',
            'field': '5',
            'operator': '>',
            'reference_indicator': 'ma',
            'reference_field': '10'
        },
        {
            'type': 'indicator',
            'indicator': 'rsi',
            'field': '',
            'operator': '<',
            'value': 70
        },
        {
            'type': 'indicator',
            'indicator': 'macd',
            'field': '',
            'operator': '>',
            'value': 0
        }
    ]
    
    success_count = 0
    for i, condition in enumerate(conditions):
        result = evaluator.evaluate_condition(condition, data, date_idx=-1)
        print(f"指标条件 {i+1}: {result}")
        if isinstance(result, bool):
            success_count += 1
    
    print(f"指标条件测试: {success_count}/{len(conditions)} 通过")
    return success_count == len(conditions)


def test_buypoint_pattern_conditions():
    """测试买点形态条件评估"""
    print("\n=== 测试买点形态条件评估 ===")
    
    evaluator = SharedConditionEvaluator()
    stock_data = create_sample_stock_data(50)
    indicator_data = calculate_indicators_with_unified_engine(stock_data)
    buypoint_data = simulate_buypoint_analysis_data(indicator_data, -1)
    
    # 测试买点形态条件
    conditions = [
        {
            'type': 'pattern',
            'pattern': 'touch_ma'
        },
        {
            'type': 'pattern',
            'pattern': 'price_stable'
        },
        {
            'type': 'pattern',
            'pattern': 'money_in'
        },
        {
            'type': 'pattern',
            'pattern': 'kpattern'
        }
    ]
    
    success_count = 0
    for i, condition in enumerate(conditions):
        result = evaluator.evaluate_condition(condition, buypoint_data, date_idx=-1)
        pattern_name = condition['pattern']
        expected = buypoint_data.get(pattern_name, False)
        print(f"形态条件 {i+1} ({pattern_name}): {result} (期望: {expected})")
        if result == expected:
            success_count += 1
    
    print(f"买点形态条件测试: {success_count}/{len(conditions)} 通过")
    return success_count == len(conditions)


def test_complex_logical_conditions():
    """测试复杂逻辑条件"""
    print("\n=== 测试复杂逻辑条件 ===")
    
    evaluator = SharedConditionEvaluator()
    stock_data = create_sample_stock_data(50)
    indicator_data = calculate_indicators_with_unified_engine(stock_data)
    buypoint_data = simulate_buypoint_analysis_data(indicator_data, -1)
    
    # 测试复杂逻辑条件
    complex_condition = {
        'type': 'logical',
        'operator': 'AND',
        'conditions': [
            {
                'type': 'basic',
                'field': 'close',
                'operator': '>',
                'reference_field': 'ma5'
            },
            {
                'type': 'logical',
                'operator': 'OR',
                'conditions': [
                    {
                        'type': 'pattern',
                        'pattern': 'touch_ma'
                    },
                    {
                        'type': 'pattern',
                        'pattern': 'money_in'
                    }
                ]
            },
            {
                'type': 'indicator',
                'indicator': 'rsi',
                'field': '',
                'operator': '<',
                'value': 80
            }
        ]
    }
    
    result = evaluator.evaluate_condition(complex_condition, buypoint_data, date_idx=-1)
    print(f"复杂逻辑条件结果: {result}")
    
    # 验证逻辑正确性
    close_gt_ma5 = buypoint_data['close'][-1] > buypoint_data['ma5'][-1]
    touch_or_money = buypoint_data.get('touch_ma', False) or buypoint_data.get('money_in', False)
    rsi_lt_80 = buypoint_data['rsi'][-1] < 80
    
    expected = close_gt_ma5 and touch_or_money and rsi_lt_80
    print(f"期望结果: {expected}")
    print(f"子条件: close>ma5={close_gt_ma5}, touch_or_money={touch_or_money}, rsi<80={rsi_lt_80}")
    
    success = result == expected
    print(f"复杂逻辑条件测试: {'通过' if success else '失败'}")
    return success


def test_multiple_conditions_evaluation():
    """测试多条件组合评估"""
    print("\n=== 测试多条件组合评估 ===")
    
    evaluator = SharedConditionEvaluator()
    stock_data = create_sample_stock_data(50)
    indicator_data = calculate_indicators_with_unified_engine(stock_data)
    buypoint_data = simulate_buypoint_analysis_data(indicator_data, -1)
    
    # 定义多个条件
    conditions = [
        {
            'type': 'basic',
            'field': 'close',
            'operator': '>',
            'reference_field': 'ma10'
        },
        {
            'type': 'pattern',
            'pattern': 'price_stable'
        },
        {
            'type': 'indicator',
            'indicator': 'rsi',
            'field': '',
            'operator': '>',
            'value': 30
        }
    ]
    
    # 测试AND逻辑
    and_result = evaluator.evaluate_conditions(conditions, buypoint_data, logic="AND", date_idx=-1)
    print(f"AND逻辑结果: {and_result}")
    
    # 测试OR逻辑
    or_result = evaluator.evaluate_conditions(conditions, buypoint_data, logic="OR", date_idx=-1)
    print(f"OR逻辑结果: {or_result}")
    
    # 验证每个条件
    individual_results = []
    for i, condition in enumerate(conditions):
        result = evaluator.evaluate_condition(condition, buypoint_data, date_idx=-1)
        individual_results.append(result)
        print(f"条件 {i+1}: {result}")
    
    expected_and = all(individual_results)
    expected_or = any(individual_results)
    
    success_and = and_result == expected_and
    success_or = or_result == expected_or
    
    print(f"AND逻辑测试: {'通过' if success_and else '失败'}")
    print(f"OR逻辑测试: {'通过' if success_or else '失败'}")
    
    return success_and and success_or


def test_performance():
    """测试性能"""
    print("\n=== 性能测试 ===")
    
    evaluator = SharedConditionEvaluator()
    stock_data = create_sample_stock_data(200)
    indicator_data = calculate_indicators_with_unified_engine(stock_data)
    buypoint_data = simulate_buypoint_analysis_data(indicator_data, -1)
    
    # 定义测试条件
    test_condition = {
        'type': 'logical',
        'operator': 'AND',
        'conditions': [
            {
                'type': 'basic',
                'field': 'close',
                'operator': '>',
                'reference_field': 'ma5'
            },
            {
                'type': 'indicator',
                'indicator': 'rsi',
                'field': '',
                'operator': '<',
                'value': 70
            },
            {
                'type': 'pattern',
                'pattern': 'touch_ma'
            }
        ]
    }
    
    # 性能测试
    num_iterations = 1000
    start_time = time.time()
    
    for _ in range(num_iterations):
        evaluator.evaluate_condition(test_condition, buypoint_data, date_idx=-1)
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / num_iterations
    
    print(f"执行 {num_iterations} 次条件评估")
    print(f"总时间: {total_time:.4f} 秒")
    print(f"平均时间: {avg_time*1000:.4f} 毫秒/次")
    
    # 获取统计信息
    stats = evaluator.get_stats()
    print(f"缓存命中率: {stats['cache_hit_rate']:.2%}")
    print(f"总评估次数: {stats['evaluations']}")
    
    # 性能要求：平均时间应小于10毫秒
    performance_ok = avg_time < 0.01
    print(f"性能测试: {'通过' if performance_ok else '失败'}")
    
    return performance_ok


def run_integration_tests():
    """运行所有集成测试"""
    print("开始共享条件评估器集成测试")
    print("=" * 50)
    
    test_results = []
    
    # 执行各项测试
    test_functions = [
        ("基础条件评估", test_basic_conditions),
        ("指标条件评估", test_indicator_conditions),
        ("买点形态条件评估", test_buypoint_pattern_conditions),
        ("复杂逻辑条件", test_complex_logical_conditions),
        ("多条件组合评估", test_multiple_conditions_evaluation),
        ("性能测试", test_performance),
    ]
    
    for test_name, test_func in test_functions:
        try:
            result = test_func()
            test_results.append((test_name, result))
            print(f"\n{test_name}: {'✅ 通过' if result else '❌ 失败'}")
        except Exception as e:
            test_results.append((test_name, False))
            print(f"\n{test_name}: ❌ 错误 - {e}")
            logger.error(f"{test_name} 测试失败: {e}")
    
    # 汇总结果
    print("\n" + "=" * 50)
    print("集成测试结果汇总:")
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
    
    print(f"\n总体结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有集成测试通过！")
        return True
    else:
        print("⚠️  部分测试失败，需要检查和修复")
        return False


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1) 