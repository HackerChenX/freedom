#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试指标数据查询
直接测试指标计算和数据查询是否正常
"""

import os
import sys

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from db.unified_data_manager import Unified_data_manager
from indicators.complete_indicator_registry import Complete_indicator_registry
from utils.logger import get_logger

logger = get_logger(__name__)

def test_basic_data_query():
    """测试基本数据查询"""
    print("=== 测试基本数据查询 ===")
    
    try:
        # 初始化数据管理器
        data_manager = Unified_data_manager()
        
        # 获取股票列表
        stocks = data_manager.get_all_stock_codes()[:5]  # 只测试前5只股票
        print(f"获取到 {len(stocks)} 只股票: {stocks}")
        
        # 测试获取单只股票的基本信息
        for stock_code in stocks:
            try:
                stock_info = data_manager.get_stock_info(stock_code)
                print(f"股票 {stock_code}: {stock_info}")
                
                # 获取最新价格数据
                price_data = data_manager.get_stock_data(
                    stock_code=stock_code,
                    start_date='2024-12-01',
                    end_date='2024-12-28'
                )
                
                if price_data is not None and not price_data.empty:
                    print(f"股票 {stock_code} 价格数据: {len(price_data)} 条记录")
                    print(f"最新数据: {price_data.tail(1).to_dict('records')}")
                else:
                    print(f"股票 {stock_code} 无价格数据")
                    
                break  # 只测试第一只股票
                
            except Exception as e:
                print(f"股票 {stock_code} 查询失败: {e}")
                continue
                
    except Exception as e:
        print(f"基本数据查询测试失败: {e}")

def test_indicator_calculation_Data():
    """测试指标计算"""
    print("\n=== 测试指标计算 ===")
    
    try:
        # 初始化指标注册器
        registry = Complete_indicator_registry()
        
        # 初始化数据管理器
        data_manager = Unified_data_manager()
        
        # 获取一只股票进行测试
        stocks = data_manager.get_all_stock_codes()[:1]
        if not stocks:
            print("没有可用的股票数据")
            return
            
        stock_code = stocks[0]
        print(f"测试股票: {stock_code}")
        
        # 获取价格数据
        price_data = data_manager.get_stock_data(
            stock_code=stock_code,
            start_date='2024-10-01',
            end_date='2024-12-28'
        )
        
        if price_data is None or price_data.empty:
            print(f"股票 {stock_code} 无价格数据")
            return
            
        print(f"价格数据: {len(price_data)} 条记录")
        
        # 测试MA指标计算
        try:
            ma_indicator = registry.get_indicator('MA')
            if ma_indicator:
                ma_result = ma_indicator.calculate(price_data, period=5)
                if ma_result is not None and not ma_result.empty:
                    print(f"MA5计算成功，结果: {len(ma_result)} 条记录")
                    latest_ma = ma_result.tail(1)
                    print(f"最新MA5值: {latest_ma.to_dict('records')}")
                else:
                    print("MA5计算失败或结果为空")
            else:
                print("MA指标未找到")
        except Exception as e:
            print(f"MA指标计算失败: {e}")
            
        # 测试RSI指标计算
        try:
            rsi_indicator = registry.get_indicator('RSI')
            if rsi_indicator:
                rsi_result = rsi_indicator.calculate(price_data, period=14)
                if rsi_result is not None and not rsi_result.empty:
                    print(f"RSI14计算成功，结果: {len(rsi_result)} 条记录")
                    latest_rsi = rsi_result.tail(1)
                    print(f"最新RSI14值: {latest_rsi.to_dict('records')}")
                else:
                    print("RSI14计算失败或结果为空")
            else:
                print("RSI指标未找到")
        except Exception as e:
            print(f"RSI指标计算失败: {e}")
            
    except Exception as e:
        print(f"指标计算测试失败: {e}")

def test_strategy_condition_Data():
    """测试策略条件评估"""
    print("\n=== 测试策略条件评估 ===")
    
    try:
        from strategy.strategy_condition_evaluator import Strategy_condition_evaluator
        
        # 初始化条件评估器
        evaluator = Strategy_condition_evaluator()
        
        # 初始化数据管理器
        data_manager = Unified_data_manager()
        
        # 获取一只股票进行测试
        stocks = data_manager.get_all_stock_codes()[:1]
        if not stocks:
            print("没有可用的股票数据")
            return
            
        stock_code = stocks[0]
        print(f"测试股票: {stock_code}")
        
        # 测试简单的MA条件
        ma_condition = {
            'type': 'indicator',
            'indicator_id': 'MA',
            'period': 5,
            'operator': '>',
            'value': 0,
            'description': 'MA5大于0'
        }
        
        try:
            result = evaluator.evaluate_condition(ma_condition, stock_code, '2024-12-28')
            print(f"MA条件评估结果: {result}")
        except Exception as e:
            print(f"MA条件评估失败: {e}")
            
        # 测试简单的RSI条件
        rsi_condition = {
            'type': 'indicator',
            'indicator_id': 'RSI',
            'period': 14,
            'operator': '<',
            'value': 80,
            'description': 'RSI小于80'
        }
        
        try:
            result = evaluator.evaluate_condition(rsi_condition, stock_code, '2024-12-28')
            print(f"RSI条件评估结果: {result}")
        except Exception as e:
            print(f"RSI条件评估失败: {e}")
            
    except Exception as e:
        print(f"策略条件测试失败: {e}")

def main_testindicatordata():
    """主函数"""
    print("开始指标数据测试")
    
    test_basic_data_query()
    test_indicator_calculation_Data()
    test_strategy_condition_Data()
    
    print("\n指标数据测试完成")

if __name__ == "__main__":
    main_testindicatordata() 