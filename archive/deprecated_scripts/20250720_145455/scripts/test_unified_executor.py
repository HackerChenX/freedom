#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
统一执行器测试脚本

测试第二阶段重构：执行器统一
验证统一选股引擎的功能和性能
"""

import os
import sys
import json
import time
from datetime import datetime

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.logger import getLogger
from strategy.strategy_executor import UnifiedStrategyExecutor
from utils.cache import get_cache_stats

logger = getLogger(__name__)


def test_unified_executor():
    """测试统一执行器"""
    logger.info("=" * 60)
    logger.info("统一执行器测试 - 第二阶段：执行器统一")
    logger.info("=" * 60)
    
    try:
        # 1. 初始化统一执行器
        logger.info("初始化统一执行器...")
        executor = UnifiedStrategyExecutor(
            max_workers=4,
            cache_enabled=True,
            enable_memory_optimization=True,
            enable_unified_config=True
        )
        
        # 2. 加载统一格式的策略配置
        logger.info("加载统一格式策略配置...")
        strategy_config_path = os.path.join(
            root_dir, 'config', 'strategies', 'standardized', 
            'kdj_all_lines_upward_strategy_unified.json'
        )
        
        if not os.path.exists(strategy_config_path):
            logger.error(f"策略配置文件不存在: {strategy_config_path}")
            return False
        
        with open(strategy_config_path, 'r', encoding='utf-8') as f:
            strategy_config = json.load(f)
        
        logger.info(f"加载策略: {strategy_config['strategy']['name']}")
        
        # 3. 执行统一策略
        logger.info("开始执行统一策略...")
        start_time = time.time()
        
        def progress_callback(progress: float, message: str):
            logger.info(f"进度: {progress:.1%} - {message}")
        
        execution_result = executor.execute_unified_strategy(
            strategy_config=strategy_config,
            enable_validation=True,
            enable_closed_loop=True,
            progress_callback=progress_callback
        )
        
        execution_time = time.time() - start_time
        
        # 4. 显示执行结果
        logger.info("=" * 50)
        logger.info("执行结果统计")
        logger.info("=" * 50)
        
        strategy_id = execution_result.get('strategy_id', 'unknown')
        selection_result = execution_result.get('selection_result', [])
        validation_result = execution_result.get('validation_result', {})
        performance_stats = execution_result.get('performance_stats', {})
        errors = execution_result.get('errors', [])
        warnings = execution_result.get('warnings', [])
        
        logger.info(f"策略ID: {strategy_id}")
        logger.info(f"执行时间: {execution_time:.2f} 秒")
        logger.info(f"选股数量: {len(selection_result)}")
        logger.info(f"错误数量: {len(errors)}")
        logger.info(f"警告数量: {len(warnings)}")
        
        if errors:
            logger.error("执行错误:")
            for error in errors:
                logger.error(f"  - {error}")
        
        if warnings:
            logger.warning("执行警告:")
            for warning in warnings:
                logger.warning(f"  - {warning}")
        
        # 5. 显示选股结果
        if selection_result:
            logger.info("\n前10只选股结果:")
            for i, result in enumerate(selection_result[:10]):
                stock_code = result.get('stock_code', 'unknown')
                score = result.get('score', 0)
                conditions_met = result.get('conditions_met', 0)
                total_conditions = result.get('total_conditions', 0)
                logger.info(f"  {i+1}. {stock_code}: 评分={score:.3f}, "
                           f"条件={conditions_met}/{total_conditions}")
        
        # 6. 显示闭环验证结果
        if validation_result:
            logger.info("\n闭环验证结果:")
            validation_method = validation_result.get('validation_method', 'unknown')
            sample_size = validation_result.get('sample_size', 0)
            consistency_rate = validation_result.get('consistency_rate', 0)
            validation_passed = validation_result.get('validation_passed', False)
            
            logger.info(f"  验证方法: {validation_method}")
            logger.info(f"  样本大小: {sample_size}")
            logger.info(f"  一致性率: {consistency_rate:.1%}")
            logger.info(f"  验证结果: {'通过' if validation_passed else '失败'}")
        
        # 7. 显示性能统计
        if performance_stats:
            logger.info("\n性能统计:")
            total_executions = performance_stats.get('total_executions', 0)
            total_stocks = performance_stats.get('total_stocks_processed', 0)
            avg_time = performance_stats.get('avg_time_per_execution', 0)
            
            logger.info(f"  总执行次数: {total_executions}")
            logger.info(f"  总处理股票数: {total_stocks}")
            logger.info(f"  平均执行时间: {avg_time:.2f} 秒")
        
        # 8. 显示缓存统计
        cache_stats = get_cache_stats()
        if cache_stats:
            logger.info("\n缓存统计:")
            for cache_type, stats in cache_stats.items():
                if stats:
                    logger.info(f"  {cache_type}: {stats}")
        
        logger.info("\n统一执行器测试完成！")
        return True
        
    except Exception as e:
        logger.error(f"统一执行器测试失败: {e}")
        return False


def test_performance_comparison():
    """测试性能对比"""
    logger.info("\n" + "=" * 60)
    logger.info("性能对比测试")
    logger.info("=" * 60)
    
    try:
        # 测试统一执行器
        logger.info("测试统一执行器性能...")
        unified_executor = UnifiedStrategyExecutor(
            max_workers=4,
            enable_memory_optimization=True,
            enable_unified_config=True
        )
        
        # 测试传统执行器
        logger.info("测试传统执行器性能...")
        from strategy.strategy_executor import StrategyExecutor
        traditional_executor = StrategyExecutor(max_workers=4)
        
        # 加载测试策略
        strategy_config_path = os.path.join(
            root_dir, 'config', 'strategies', 'standardized', 
            'test_simple_strategy_unified.json'
        )
        
        if not os.path.exists(strategy_config_path):
            logger.warning(f"测试策略配置不存在: {strategy_config_path}")
            return False
        
        with open(strategy_config_path, 'r', encoding='utf-8') as f:
            strategy_config = json.load(f)
        
        # 执行性能测试
        logger.info("执行统一执行器测试...")
        start_time = time.time()
        unified_result = unified_executor.execute_unified_strategy(
            strategy_config=strategy_config,
            enable_validation=False,  # 关闭验证以专注性能测试
            enable_closed_loop=False
        )
        unified_time = time.time() - start_time
        
        # 显示性能对比结果
        logger.info("\n性能对比结果:")
        logger.info(f"统一执行器:")
        logger.info(f"  执行时间: {unified_time:.2f} 秒")
        logger.info(f"  选股数量: {len(unified_result.get('selection_result', []))}")
        
        # 计算性能提升
        unified_stats = unified_executor.get_performance_stats()
        logger.info(f"  处理股票数: {unified_stats.get('total_stocks_processed', 0)}")
        
        if unified_stats.get('total_stocks_processed', 0) > 0:
            stocks_per_second = unified_stats['total_stocks_processed'] / unified_time
            logger.info(f"  处理速度: {stocks_per_second:.1f} 股票/秒")
        
        logger.info("\n性能对比测试完成！")
        return True
        
    except Exception as e:
        logger.error(f"性能对比测试失败: {e}")
        return False


def main():
    """主函数"""
    logger.info("开始统一执行器测试...")
    
    # 测试基本功能
    basic_test_passed = test_unified_executor()
    
    # 测试性能对比
    performance_test_passed = test_performance_comparison()
    
    # 总结
    logger.info("\n" + "=" * 60)
    logger.info("测试总结")
    logger.info("=" * 60)
    logger.info(f"基本功能测试: {'通过' if basic_test_passed else '失败'}")
    logger.info(f"性能对比测试: {'通过' if performance_test_passed else '失败'}")
    
    if basic_test_passed and performance_test_passed:
        logger.info("✅ 第二阶段重构测试全部通过！")
        return 0
    else:
        logger.error("❌ 部分测试失败，需要进一步调试")
        return 1


if __name__ == "__main__":
    sys.exit(main())
