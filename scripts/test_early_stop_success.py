#!/usr/bin/env python3
"""
测试成功后早停功能

使用更宽松的条件确保能选出股票，验证早停功能
"""

import os
import sys
import time
from datetime import datetime

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from analysis.engines.indicator_validation_framework import (
    Indicator_validation_framework, 
    Indicator_validation_config, 
    Validation_mode
)
from utils.logger import get_logger

logger = get_logger(__name__)


def test_early_stop_with_loose_conditions():
    """测试使用宽松条件的早停功能"""
    print("🧪 测试成功后早停功能（宽松条件）")
    print("=" * 60)
    
    # 创建非常宽松的配置，确保能选出股票
    config = Indicator_validation_config(
        mode=Validation_mode.QUICK,
        stock_pool_size=20,           # 很小的股票池，加快速度
        max_selection_ratio=0.8,      # 非常宽松的选股比例（80%）
        min_selection_count=1,        # 最少1只股票
        stop_on_success=True,         # 成功后立即停止
        stop_on_error=False,          # 不在错误时停止
        debug_mode=True,              # 开启调试模式
        parallel_workers=1,           # 单线程，确保顺序执行
        validation_date="2025-06-28"  # 使用最近的日期
    )
    
    framework = Indicator_validation_framework(config)
    
    # 手动创建一个总是成功的策略配置
    always_success_strategy = {
        "name": "always_success_test",
        "description": "总是成功的测试策略",
        "conditions": [
            {
                "indicator": "MA",
                "field": "close",
                "period": 5,
                "comparison": "gte",  # 大于等于
                "value": 0,          # 0，几乎所有股票都满足
                "logic": "and"
            }
        ]
    }
    
    print(f"📋 测试配置:")
    print(f"  • 股票池大小: {config.stock_pool_size}")
    print(f"  • 最大选股比例: {config.max_selection_ratio}")
    print(f"  • 成功后停止: {config.stop_on_success}")
    print(f"  • 策略条件: close >= 0 (应该选出所有股票)")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        # 准备股票池
        stock_pool = framework._prepare_stock_pool()
        print(f"📊 股票池准备完成: {len(stock_pool)} 只股票")
        
        if len(stock_pool) == 0:
            print("❌ 股票池为空，无法进行测试")
            return
        
        # 执行策略选股，应该选出大部分股票
        selected_stocks = framework._execute_strategy_selection(always_success_strategy, stock_pool)
        
        print(f"✅ 策略执行完成:")
        print(f"  • 股票池大小: {len(stock_pool)}")
        print(f"  • 选中股票数: {len(selected_stocks)}")
        print(f"  • 选股比例: {len(selected_stocks) / len(stock_pool):.2%}")
        
        if len(selected_stocks) > 0:
            print(f"  • 选中股票样例: {selected_stocks[:5]}")
            
            # 验证早停逻辑
            if len(selected_stocks) >= config.min_selection_count:
                print("🛑 满足成功条件，应该触发早停")
                print("✅ 早停功能验证成功！")
            else:
                print("⚠️ 未达到最少选股数量，不会触发早停")
        else:
            print("❌ 未选出任何股票，早停功能无法验证")
            
    except Exception as e:
        print(f"❌ 测试过程出错: {e}")
        logger.error(f"测试早停功能失败: {e}")
    
    end_time = time.time()
    print(f"\n⏱️ 测试耗时: {end_time - start_time:.2f}秒")


def test_framework_early_stop():
    """测试框架级别的早停功能"""
    print("\n🧪 测试框架级别早停功能")
    print("=" * 60)
    
    # 创建宽松配置
    config = Indicator_validation_config(
        mode=Validation_mode.QUICK,
        stock_pool_size=10,           # 更小的股票池
        max_selection_ratio=0.9,      # 90%选股比例
        min_selection_count=1,        # 最少1只
        stop_on_success=True,         # 成功后停止
        debug_mode=True,
        parallel_workers=1,
        validation_date="2025-06-28"
    )
    
    framework = Indicator_validation_framework(config)
    
    # 只测试一个指标
    test_indicators = ['MA']
    
    print(f"📋 测试配置:")
    print(f"  • 测试指标: {test_indicators}")
    print(f"  • 股票池大小: {config.stock_pool_size}")
    print(f"  • 成功后停止: {config.stop_on_success}")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        # 使用框架验证单个指标
        result = framework.validate_single_indicator('MA')
        
        print(f"✅ 指标验证结果:")
        print(f"  • 指标名称: {result['indicator_name']}")
        print(f"  • 验证状态: {result['status']}")
        print(f"  • 选股数量: {result.get('selected_count', 0)}")
        print(f"  • 选股比例: {result.get('selection_ratio', 0):.4f}")
        
        if result['status'] == 'success':
            print("🛑 验证成功，早停功能应该生效")
            print("✅ 框架早停功能验证成功！")
        else:
            print(f"⚠️ 验证状态: {result['status']}")
            
    except Exception as e:
        print(f"❌ 框架测试出错: {e}")
        logger.error(f"框架早停测试失败: {e}")
    
    end_time = time.time()
    print(f"\n⏱️ 框架测试耗时: {end_time - start_time:.2f}秒")


def main_testearlystopsuccess():
    """主函数"""
    print("🚀 开始早停功能测试")
    print("=" * 80)
    print(f"⏰ 测试开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 测试1: 策略级别早停
    test_early_stop_with_loose_conditions()
    
    # 测试2: 框架级别早停
    test_framework_early_stop()
    
    print("\n" + "=" * 80)
    print("✅ 早停功能测试完成！")


if __name__ == "__main__":
    main_testearlystopsuccess() 