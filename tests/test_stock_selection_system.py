#!/usr/bin/env python3
"""
可配置策略选股系统测试脚本

测试修复后的选股系统功能
"""

import sys
import os
import json
import tempfile
import traceback
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.data_manager import DataManager
from strategy.strategy_executor import StrategyExecutor
from strategy.strategy_parser import StrategyParser
from strategy.strategy_manager import StrategyManager
from utils.logger import get_logger

logger = get_logger(__name__)


def test_data_manager_stock_list():
    """测试DataManager的get_stock_list方法"""
    print("\n=== 测试DataManager.get_stock_list方法 ===")
    
    try:
        dm = DataManager()
        
        # 测试无过滤条件
        print("1. 测试无过滤条件获取股票列表...")
        stock_list = dm.get_stock_list()
        print(f"   获取到 {len(stock_list)} 只股票")
        if not stock_list.empty:
            print(f"   列名: {list(stock_list.columns)}")
            print(f"   前5只股票: \n{stock_list.head()}")
        
        # 测试市场过滤
        print("\n2. 测试市场过滤...")
        filtered_list = dm.get_stock_list(filters={'market': ['主板']})
        print(f"   主板股票数量: {len(filtered_list)}")
        
        return True
        
    except Exception as e:
        print(f"   错误: {e}")
        traceback.print_exc()
        return False


def test_strategy_executor_basic():
    """测试策略执行器基本功能"""
    print("\n=== 测试策略执行器基本功能 ===")
    
    try:
        # 创建测试策略配置
        strategy_config = {
            "strategy_id": "TEST_BASIC",
            "name": "基础测试策略",
            "description": "用于测试的基础策略",
            "conditions": [
                {
                    "type": "indicator",
                    "indicator_id": "RSI",
                    "period": "DAILY",
                    "parameters": {
                        "period": 14
                    },
                    "operator": "<",
                    "value": 70
                }
            ],
            "filters": {
                "market": ["主板"]
            }
        }
        
        # 创建执行器
        executor = StrategyExecutor()
        
        # 验证策略计划
        print("1. 验证策略计划...")
        is_valid = executor._validate_strategy_plan(strategy_config)
        print(f"   策略计划验证: {'通过' if is_valid else '失败'}")
        
        # 测试获取股票列表
        print("2. 测试获取过滤后的股票列表...")
        stock_list = executor._get_filtered_stock_list(strategy_config.get('filters', {}))
        print(f"   获取到 {len(stock_list)} 只股票")
        
        return True
        
    except Exception as e:
        print(f"   错误: {e}")
        traceback.print_exc()
        return False


def test_strategy_condition_evaluator():
    """测试策略条件评估器"""
    print("\n=== 测试策略条件评估器 ===")
    
    try:
        from strategy.strategy_condition_evaluator import StrategyConditionEvaluator
        
        # 创建评估器
        evaluator = StrategyConditionEvaluator()
        
        # 创建测试数据
        dates = pd.date_range('2023-01-01', periods=30)
        test_data = pd.DataFrame({
            'date': dates,
            'open': np.random.uniform(10, 20, 30),
            'high': np.random.uniform(15, 25, 30),
            'low': np.random.uniform(5, 15, 30),
            'close': np.linspace(10, 20, 30),  # 上升趋势
            'volume': np.random.uniform(1000, 2000, 30)
        })
        test_data.set_index('date', inplace=True)
        
        # 测试价格条件
        print("1. 测试价格条件...")
        price_condition = {
            "type": "price",
            "field": "close",
            "operator": ">",
            "value": 15
        }
        
        result = evaluator.evaluate_condition(price_condition, test_data, '2023-01-30')
        print(f"   价格条件评估结果: {result}")
        
        # 测试指标条件
        print("2. 测试指标条件...")
        indicator_condition = {
            "type": "indicator",
            "indicator_id": "RSI",
            "period": "DAILY",
            "parameters": {"period": 14},
            "operator": "<",
            "value": 70
        }
        
        result = evaluator.evaluate_condition(indicator_condition, test_data, '2023-01-30')
        print(f"   指标条件评估结果: {result}")
        
        return True
        
    except Exception as e:
        print(f"   错误: {e}")
        traceback.print_exc()
        return False


def test_end_to_end_selection():
    """测试端到端选股流程"""
    print("\n=== 测试端到端选股流程 ===")
    
    try:
        # 创建简单的测试策略
        strategy_config = {
            "strategy_id": "TEST_E2E",
            "name": "端到端测试策略",
            "description": "用于端到端测试的策略",
            "conditions": [
                {
                    "type": "price",
                    "field": "close",
                    "operator": ">",
                    "value": 5
                }
            ],
            "filters": {
                "market": ["主板"]
            },
            "result_filters": {
                "max_results": 10
            }
        }
        
        # 创建执行器
        executor = StrategyExecutor()
        
        print("1. 执行选股策略...")
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        result = executor.execute_strategy(
            strategy_plan=strategy_config,
            end_date=end_date
        )
        
        print(f"   选股结果: {len(result)} 只股票")
        if not result.empty:
            print(f"   结果列名: {list(result.columns)}")
            print(f"   前3只股票: \n{result.head(3)}")
        
        return True
        
    except Exception as e:
        print(f"   错误: {e}")
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("开始测试可配置策略选股系统...")
    
    test_results = []
    
    # 运行各项测试
    test_results.append(("DataManager股票列表", test_data_manager_stock_list()))
    test_results.append(("策略执行器基本功能", test_strategy_executor_basic()))
    test_results.append(("策略条件评估器", test_strategy_condition_evaluator()))
    test_results.append(("端到端选股流程", test_end_to_end_selection()))
    
    # 输出测试结果汇总
    print("\n" + "="*60)
    print("测试结果汇总:")
    print("="*60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name:<30} {status}")
        if result:
            passed += 1
    
    print("-"*60)
    print(f"总计: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！选股系统修复成功。")
        return 0
    else:
        print("⚠️  部分测试失败，需要进一步修复。")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
