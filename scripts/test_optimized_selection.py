#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
测试优化后的股票选股性能
"""

import time
import pandas as pd
import os
import sys
from datetime import datetime

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from utils.logger import get_logger

logger = get_logger(__name__)


def test_batch_data_optimizer():
    """测试批量数据优化器"""
    print("🧪 测试批量数据优化器")
    
    try:
        from strategy.batch_data_optimizer import get_batch_optimizer
        
        # 创建优化器
        optimizer = get_batch_optimizer(batch_size=50, cache_enabled=True)
        
        # 模拟股票代码
        stock_codes = [f"{i:06d}.SH" for i in range(1, 101)]  # 100只股票
        
        # 测试批量获取数据
        start_time = time.time()
        stocks_data = optimizer.get_stocks_data_batch(
            stock_codes=stock_codes,
            end_date=datetime.now().strftime("%Y-%m-%d"),
            days_back=30
        )
        end_time = time.time()
        
        print(f"✅ 批量数据获取测试完成")
        print(f"   📊 股票数量: {len(stock_codes)}")
        print(f"   ⏱️  耗时: {end_time - start_time:.2f}秒")
        print(f"   📈 成功获取: {len(stocks_data)} 只股票数据")
        
        # 测试批量指标计算
        if stocks_data:
            indicators = ['ma', 'rsi', 'macd', 'volume_ma']
            start_time = time.time()
            indicators_data = optimizer.calculate_indicators_batch(
                stocks_data=stocks_data,
                indicators=indicators
            )
            end_time = time.time()
            
            print(f"✅ 批量指标计算测试完成")
            print(f"   📊 指标数量: {len(indicators)}")
            print(f"   ⏱️  耗时: {end_time - start_time:.2f}秒")
            print(f"   📈 成功计算: {len(indicators_data)} 只股票指标")
        
        return True
        
    except Exception as e:
        logger.error(f"批量数据优化器测试失败: {e}")
        return False


def test_optimized_executor_Selection():
    """测试优化的策略执行器"""
    print("\n🧪 测试优化的策略执行器")
    
    try:
        from strategy.strategy_executor import UnifiedStrategyExecutor as Optimized_strategy_executor
        
        # 创建优化执行器
        executor = Optimized_strategy_executor(
            max_workers=16,
            cache_enabled=True,
            batch_size=50,
            enable_memory_monitoring=True
        )
        
        # 创建简单测试策略
        test_strategy = {
            "strategy_id": "TEST_OPTIMIZED",
            "name": "优化测试策略",
            "conditions": [
                {
                    "type": "price",
                    "field": "close",
                    "operator": ">",
                    "value": 5.0
                }
            ],
            "filters": {
                "price": {"min": 3.0, "max": 100.0}
            },
            "result_filters": {
                "max_results": 10
            }
        }
        
        # 执行测试
        start_time = time.time()
        
        def progress_callback_Selection_Test_Optimized_Selection(progress, message):
            print(f"   进度: {progress:.1%} - {message}")
        
        results = executor.execute_strategy_optimized(
            strategy_plan=test_strategy,
            end_date=datetime.now().strftime("%Y-%m-%d"),
            progress_callback=progress_callback,
            enable_early_stop=False,
            max_results=10
        )
        
        end_time = time.time()
        
        print(f"✅ 优化执行器测试完成")
        print(f"   ⏱️  总耗时: {end_time - start_time:.2f}秒")
        print(f"   📈 结果数量: {len(results) if results is not None else 0}")
        
        # 获取性能报告
        performance_report = executor.get_performance_report()
        print(f"   🎯 性能报告: {performance_report}")
        
        return True
        
    except Exception as e:
        logger.error(f"优化执行器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main_testoptimizedselection():
    """主函数"""
    print("🚀 开始股票选股性能优化测试")
    print("="*60)
    
    # 测试批量数据优化器
    batch_test_success = test_batch_data_optimizer()
    
    # 测试优化执行器
    executor_test_success = test_optimized_executor_Selection()
    
    # 总结
    print("\n" + "="*60)
    print("📋 测试总结:")
    print(f"   批量数据优化器: {'✅ 通过' if batch_test_success else '❌ 失败'}")
    print(f"   优化策略执行器: {'✅ 通过' if executor_test_success else '❌ 失败'}")
    
    if batch_test_success and executor_test_success:
        print("\n🎉 所有测试通过！优化方案可以投入使用。")
    else:
        print("\n⚠️  部分测试失败，需要进一步调试。")


if __name__ == "__main__":
    main_testoptimizedselection() 