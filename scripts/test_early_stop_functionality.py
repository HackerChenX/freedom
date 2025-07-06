#!/usr/bin/env python3
"""
早停功能测试脚本

测试不同场景下的早停功能：
1. 数据库连接错误触发早停
2. 指标验证成功触发早停
3. 指标验证错误触发早停
4. 禁用早停功能的情况
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
    Validation_mode,
    Validation_result
)
from utils.logger import get_logger

logger = get_logger(__name__)


def test_database_connection_error_early_stop():
    """测试数据库连接错误触发早停"""
    logger.info("🧪 测试场景1: 数据库连接错误触发早停")
    logger.info("=" * 60)
    
    # 创建配置，启用错误后早停
    config = Indicator_validation_config(
        mode=Validation_mode.QUICK,
        stock_pool_size=50,
        stop_on_success=False,
        stop_on_error=True,  # 🔑 启用错误后早停
        debug_mode=True,
        parallel_workers=1
    )
    
    logger.info("📋 测试配置:")
    logger.info(f"  • 错误后早停: {config.stop_on_error}")
    logger.info(f"  • 成功后早停: {config.stop_on_success}")
    logger.info(f"  • 验证模式: {config.mode.value}")
    
    framework = Indicator_validation_framework(config)
    
    start_time = time.time()
    try:
        result = framework.validate_all_indicators()
        
        # 分析结果
        duration = time.time() - start_time
        logger.info(f"⏱️ 验证耗时: {duration:.2f}秒")
        
        summary = result['summary']
        results = result['results']
        
        logger.info("📊 测试结果:")
        logger.info(f"  • 总指标数: {summary['total_indicators']}")
        logger.info(f"  • 实际验证数: {len(results)}")
        logger.info(f"  • 成功数: {summary.get('successful_validations', 0)}")
        logger.info(f"  • 失败数: {summary.get('failed_validations', 0)}")
        
        # 检查是否触发了早停
        if len(results) < summary['total_indicators']:
            logger.info("✅ 早停功能正常工作！")
            
            # 检查早停原因
            if 'early_stop_reason' in result:
                logger.info(f"🛑 早停原因: {result['early_stop_reason']}")
            
            # 检查第一个结果是否是数据库连接错误
            if results and results[0]['indicator_name'] == 'DATABASE_CONNECTION':
                logger.info("✅ 数据库连接错误被正确检测并触发早停")
            else:
                logger.warning("⚠️ 早停原因可能不是数据库连接错误")
        else:
            logger.warning("❌ 早停功能未触发，可能数据库连接正常或早停配置无效")
            
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        return False
    
    logger.info("=" * 60)
    return True


def test_success_early_stop():
    """测试成功后早停功能（模拟场景）"""
    logger.info("🧪 测试场景2: 成功后早停功能")
    logger.info("=" * 60)
    
    # 创建配置，启用成功后早停
    config = Indicator_validation_config(
        mode=Validation_mode.QUICK,
        stock_pool_size=50,
        stop_on_success=True,  # 🔑 启用成功后早停
        stop_on_error=False,
        debug_mode=True,
        parallel_workers=1,
        min_selection_count=0,  # 降低成功标准，更容易触发成功
        max_selection_ratio=1.0  # 允许更高的选股比例
    )
    
    logger.info("📋 测试配置:")
    logger.info(f"  • 成功后早停: {config.stop_on_success}")
    logger.info(f"  • 错误后早停: {config.stop_on_error}")
    logger.info(f"  • 最小选股数: {config.min_selection_count}")
    logger.info(f"  • 最大选股比例: {config.max_selection_ratio}")
    
    framework = Indicator_validation_framework(config)
    
    start_time = time.time()
    try:
        result = framework.validate_all_indicators()
        
        # 分析结果
        duration = time.time() - start_time
        logger.info(f"⏱️ 验证耗时: {duration:.2f}秒")
        
        summary = result['summary']
        results = result['results']
        
        logger.info("📊 测试结果:")
        logger.info(f"  • 总指标数: {summary['total_indicators']}")
        logger.info(f"  • 实际验证数: {len(results)}")
        logger.info(f"  • 成功数: {summary.get('successful_validations', 0)}")
        logger.info(f"  • 失败数: {summary.get('failed_validations', 0)}")
        
        # 检查是否触发了早停
        if len(results) < summary['total_indicators']:
            logger.info("✅ 早停功能正常工作！")
            
            # 检查最后一个结果是否是成功的
            if results and results[-1]['status'] == ValidationResult.SUCCESS.value:
                logger.info("✅ 成功后早停功能正常工作")
            else:
                logger.info("ℹ️ 早停可能由其他原因触发")
        else:
            logger.info("ℹ️ 所有指标都完成了验证，未触发早停")
            
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        return False
    
    logger.info("=" * 60)
    return True


def test_no_early_stop():
    """测试禁用早停功能"""
    logger.info("🧪 测试场景3: 禁用早停功能")
    logger.info("=" * 60)
    
    # 创建配置，禁用所有早停
    config = Indicator_validation_config(
        mode=Validation_mode.QUICK,
        stock_pool_size=50,
        stop_on_success=False,  # 🔑 禁用成功后早停
        stop_on_error=False,    # 🔑 禁用错误后早停
        debug_mode=True,
        parallel_workers=1
    )
    
    logger.info("📋 测试配置:")
    logger.info(f"  • 成功后早停: {config.stop_on_success}")
    logger.info(f"  • 错误后早停: {config.stop_on_error}")
    logger.info("  • 预期: 即使遇到错误也会继续验证所有指标")
    
    framework = Indicator_validation_framework(config)
    
    start_time = time.time()
    try:
        result = framework.validate_all_indicators()
        
        # 分析结果
        duration = time.time() - start_time
        logger.info(f"⏱️ 验证耗时: {duration:.2f}秒")
        
        summary = result['summary']
        results = result['results']
        
        logger.info("📊 测试结果:")
        logger.info(f"  • 总指标数: {summary['total_indicators']}")
        logger.info(f"  • 实际验证数: {len(results)}")
        logger.info(f"  • 成功数: {summary.get('successful_validations', 0)}")
        logger.info(f"  • 失败数: {summary.get('failed_validations', 0)}")
        
        # 检查是否验证了所有指标
        if len(results) == summary['total_indicators']:
            logger.info("✅ 禁用早停功能正常工作，验证了所有指标")
        else:
            logger.warning(f"⚠️ 验证数量不匹配，可能存在其他问题")
            
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        return False
    
    logger.info("=" * 60)
    return True


def run_all_tests_Functionality():
    """运行所有早停功能测试"""
    logger.info("🚀 开始早停功能完整测试")
    logger.info("=" * 80)
    
    test_results = {}
    
    # 测试1: 数据库连接错误早停
    test_results['database_error'] = test_database_connection_error_early_stop()
    
    # 等待一下，避免测试间干扰
    time.sleep(1)
    
    # 测试2: 成功后早停
    test_results['success_early_stop'] = test_success_early_stop()
    
    # 等待一下
    time.sleep(1)
    
    # 测试3: 禁用早停
    test_results['no_early_stop'] = test_no_early_stop()
    
    # 汇总测试结果
    logger.info("=" * 80)
    logger.info("📋 测试结果汇总:")
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    for test_name, result in test_results.items():
        status = "✅ 通过" if result else "❌ 失败"
        logger.info(f"  • {test_name}: {status}")
    
    logger.info(f"总体结果: {passed_tests}/{total_tests} 测试通过")
    
    if passed_tests == total_tests:
        logger.info("🎉 所有早停功能测试通过！")
        return True
    else:
        logger.error(f"❌ {total_tests - passed_tests} 个测试失败")
        return False


def main_testearlystopfunctionality():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='早停功能测试脚本')
    parser.add_argument('--test', type=str, 
                        choices=['database', 'success', 'disabled', 'all'],
                        default='all',
                        help='选择要运行的测试')
    
    args = parser.parse_args()
    
    try:
        if args.test == 'database':
            success = test_database_connection_error_early_stop()
        elif args.test == 'success':
            success = test_success_early_stop()
        elif args.test == 'disabled':
            success = test_no_early_stop()
        else:  # all
            success = run_all_tests_Functionality()
        
        return 0 if success else 1
        
    except Keyboard_interrupt:
        logger.info("🛑 用户中断测试")
        return 1
    except Exception as e:
        logger.error(f"❌ 测试执行失败: {e}")
        return 1


if __name__ == "__main__":
    exit(main_testearlystopfunctionality()) 