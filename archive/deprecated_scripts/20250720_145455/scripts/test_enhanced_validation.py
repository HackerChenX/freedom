#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强闭环验证测试脚本

测试第三阶段重构：闭环验证增强
验证入口点分析和反向验证机制
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
from analysis.enhanced_closed_loop_validator import EnhancedClosedLoopValidator
from strategy.strategy_executor import UnifiedStrategyExecutor

logger = getLogger(__name__)


def test_enhanced_validation():
    """测试增强闭环验证"""
    logger.info("=" * 60)
    logger.info("增强闭环验证测试 - 第三阶段：闭环验证增强")
    logger.info("=" * 60)
    
    try:
        # 1. 初始化增强验证器
        logger.info("初始化增强闭环验证器...")
        validator = EnhancedClosedLoopValidator(enable_entry_point_analysis=True)
        
        # 2. 创建模拟选股结果
        logger.info("创建模拟选股结果...")
        mock_selection_results = [
            {
                'stock_code': '000001.SZ',
                'score': 0.85,
                'conditions_met': 4,
                'total_conditions': 5,
                'indicators_data': {
                    'MACD': {'macd': 0.15, 'signal': 0.12, 'histogram': 0.03},
                    'KDJ': {'k': 75, 'd': 70, 'j': 85}
                }
            },
            {
                'stock_code': '000002.SZ',
                'score': 0.78,
                'conditions_met': 3,
                'total_conditions': 4,
                'indicators_data': {
                    'MACD': {'macd': 0.08, 'signal': 0.06, 'histogram': 0.02},
                    'KDJ': {'k': 68, 'd': 65, 'j': 74}
                }
            },
            {
                'stock_code': '600000.SH',
                'score': 0.92,
                'conditions_met': 5,
                'total_conditions': 5,
                'indicators_data': {
                    'MACD': {'macd': 0.22, 'signal': 0.18, 'histogram': 0.04},
                    'KDJ': {'k': 82, 'd': 78, 'j': 90}
                }
            }
        ]
        
        # 3. 创建模拟策略配置
        strategy_config = {
            'strategy': {
                'id': 'TEST_ENHANCED_VALIDATION',
                'name': '增强验证测试策略'
            },
            'technical_indicators': {
                'primary_indicators': [
                    {
                        'indicator_id': 'MACD',
                        'parameters': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
                        'conditions': [
                            {'field': 'macd', 'operator': '>', 'value': 0},
                            {'field': 'histogram', 'operator': '>', 'value': 0}
                        ]
                    },
                    {
                        'indicator_id': 'KDJ',
                        'parameters': {'k_period': 9, 'd_period': 3},
                        'conditions': [
                            {'field': 'k', 'operator': '>', 'value': 50},
                            {'field': 'd', 'operator': '>', 'value': 50}
                        ]
                    }
                ]
            },
            'time_criteria': {
                'time_frames': [{'level': 'daily', 'priority': 1}],
                'date_range': {'target_date': '2025-07-20'}
            },
            'validation': {
                'validation_method': 'entry_point_analysis',
                'validation_threshold': 0.8,
                'sample_size': 3
            }
        }
        
        # 4. 测试不同验证方法
        validation_methods = ['entry_point_analysis', 'pattern_recognition', 'indicator_consistency']
        
        for method in validation_methods:
            logger.info(f"\n测试验证方法: {method}")
            
            start_time = time.time()
            validation_result = validator.validate_strategy_selection(
                selection_results=mock_selection_results,
                strategy_config=strategy_config,
                validation_method=method
            )
            execution_time = time.time() - start_time
            
            # 显示验证结果
            logger.info(f"验证方法: {method}")
            logger.info(f"执行时间: {execution_time:.2f} 秒")
            logger.info(f"验证样本数: {validation_result.get('validation_samples', 0)}")
            logger.info(f"一致性率: {validation_result.get('consistency_rate', 0):.1%}")
            logger.info(f"验证结果: {'通过' if validation_result.get('validation_passed', False) else '失败'}")
            
            # 显示详细结果
            validation_details = validation_result.get('validation_details', [])
            if validation_details:
                logger.info("验证详情:")
                for detail in validation_details:
                    stock_code = detail.get('stock_code', 'unknown')
                    is_consistent = detail.get('is_consistent', False)
                    original_score = detail.get('original_score', 0)
                    revalidation_score = detail.get('revalidation_score', 0)
                    
                    logger.info(f"  {stock_code}: {'一致' if is_consistent else '不一致'}, "
                               f"原始评分={original_score:.3f}, 重验评分={revalidation_score:.3f}")
            
            # 显示建议
            recommendations = validation_result.get('recommendations', [])
            if recommendations:
                logger.info("改进建议:")
                for rec in recommendations:
                    logger.info(f"  - {rec}")
        
        # 5. 显示验证统计
        logger.info("\n验证统计信息:")
        stats = validator.get_validation_stats()
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        logger.info("\n增强闭环验证测试完成！")
        return True
        
    except Exception as e:
        logger.error(f"增强闭环验证测试失败: {e}")
        return False


def test_integrated_validation():
    """测试集成验证（统一执行器 + 增强验证）"""
    logger.info("\n" + "=" * 60)
    logger.info("集成验证测试")
    logger.info("=" * 60)
    
    try:
        # 1. 初始化统一执行器（启用增强验证）
        logger.info("初始化统一执行器...")
        executor = UnifiedStrategyExecutor(
            max_workers=2,
            cache_enabled=True,
            enable_memory_optimization=True,
            enable_unified_config=True
        )
        
        # 2. 加载策略配置
        strategy_config_path = os.path.join(
            root_dir, 'config', 'strategies', 'standardized',
            'kdj_all_lines_upward_strategy_unified.json'
        )
        
        if not os.path.exists(strategy_config_path):
            logger.warning(f"策略配置文件不存在: {strategy_config_path}")
            return False
        
        with open(strategy_config_path, 'r', encoding='utf-8') as f:
            strategy_config = json.load(f)
        
        # 确保启用闭环验证
        if 'validation' not in strategy_config:
            strategy_config['validation'] = {}
        
        strategy_config['validation'].update({
            'validation_method': 'entry_point_analysis',
            'validation_threshold': 0.7,
            'sample_size': 5
        })
        
        logger.info(f"加载策略: {strategy_config['strategy']['name']}")
        
        # 3. 执行带增强验证的策略
        logger.info("执行带增强验证的统一策略...")
        
        def progress_callback(progress: float, message: str):
            logger.info(f"进度: {progress:.1%} - {message}")
        
        start_time = time.time()
        execution_result = executor.execute_unified_strategy(
            strategy_config=strategy_config,
            enable_validation=True,
            enable_closed_loop=True,  # 启用增强闭环验证
            progress_callback=progress_callback
        )
        execution_time = time.time() - start_time
        
        # 4. 分析验证结果
        logger.info("\n集成验证结果分析:")
        logger.info(f"总执行时间: {execution_time:.2f} 秒")
        
        selection_result = execution_result.get('selection_result', [])
        validation_result = execution_result.get('validation_result', {})
        
        logger.info(f"选股数量: {len(selection_result)}")
        
        if validation_result:
            logger.info("增强闭环验证结果:")
            logger.info(f"  验证方法: {validation_result.get('validation_method', 'unknown')}")
            logger.info(f"  验证样本: {validation_result.get('validation_samples', 0)}")
            logger.info(f"  一致性率: {validation_result.get('consistency_rate', 0):.1%}")
            logger.info(f"  验证状态: {'通过' if validation_result.get('validation_passed', False) else '失败'}")
            
            # 显示验证总结
            summary = validation_result.get('summary', {})
            if summary:
                logger.info("验证总结:")
                for key, value in summary.items():
                    logger.info(f"  {key}: {value}")
            
            # 显示改进建议
            recommendations = validation_result.get('recommendations', [])
            if recommendations:
                logger.info("改进建议:")
                for rec in recommendations:
                    logger.info(f"  - {rec}")
        
        # 5. 性能分析
        performance_stats = execution_result.get('performance_stats', {})
        if performance_stats:
            logger.info("\n性能统计:")
            for key, value in performance_stats.items():
                logger.info(f"  {key}: {value}")
        
        logger.info("\n集成验证测试完成！")
        return True
        
    except Exception as e:
        logger.error(f"集成验证测试失败: {e}")
        return False


def main():
    """主函数"""
    logger.info("开始增强闭环验证测试...")
    
    # 测试增强验证器
    enhanced_test_passed = test_enhanced_validation()
    
    # 测试集成验证
    integrated_test_passed = test_integrated_validation()
    
    # 总结
    logger.info("\n" + "=" * 60)
    logger.info("测试总结")
    logger.info("=" * 60)
    logger.info(f"增强验证器测试: {'通过' if enhanced_test_passed else '失败'}")
    logger.info(f"集成验证测试: {'通过' if integrated_test_passed else '失败'}")
    
    if enhanced_test_passed and integrated_test_passed:
        logger.info("✅ 第三阶段重构测试全部通过！")
        return 0
    else:
        logger.error("❌ 部分测试失败，需要进一步调试")
        return 1


if __name__ == "__main__":
    sys.exit(main())
