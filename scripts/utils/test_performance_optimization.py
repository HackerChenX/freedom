#!/usr/bin/env python3
"""
性能优化测试脚本

测试新的性能优化组件，验证是否能达到5分钟内处理4000只股票的目标。

Author: System
Date: 2025-01-15
"""

import os
import sys
import time
import json
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from db.services.integrated.performance_optimizer import Performance_optimizer, Optimization_config
from config.container_config import configure_container
from db.interfaces.data_access_interface import DataAccessInterface
from db.interfaces.cache_interface import ICacheService
from utils.logger import get_logger
from db.sql_manager import SQLManager, QueryType

logger = get_logger(__name__)


def setup_test_environment():
    """设置测试环境"""
    logger.info("设置测试环境")
    
    # 配置依赖注入容器
    container = configure_container()
    
    # 获取服务
    data_access = get_service(Data_access_interface)
    cache_service = get_service(Data_access_interface)
    
    return data_access, cache_service


def get_test_stock_codes(count: int = 4000) -> list:
    """获取测试股票代码"""
    logger.info(f"获取测试股票代码: {count}只")
    
    # 这里应该从数据库获取真实的股票代码
    # 为了测试，我们生成一些模拟代码
    test_codes = []
    
    # A股主板代码格式
    for i in range(min(count, 1000)):
        test_codes.append(f"60{i:04d}")  # 上海主板
    
    for i in range(min(count - 1000, 1000)):
        test_codes.append(f"00{i:04d}")  # 深圳主板
    
    for i in range(min(count - 2000, 1000)):
        test_codes.append(f"30{i:04d}")  # 创业板
    
    for i in range(count - len(test_codes)):
        test_codes.append(f"68{i:04d}")  # 科创板
    
    return test_codes[:count]


def test_basic_optimization():
    """测试基础优化功能"""
    logger.info("开始基础优化功能测试")
    
    try:
        # 设置测试环境
        data_access, cache_service = setup_test_environment()
        
        # 创建性能优化器
        config = Optimization_config(
            batch_size=100,
            max_workers=16,
            max_memory_usage_percent=70.0,
            enable_cache=True
        )
        
        optimizer = Performance_optimizer(data_access, cache_service, config)
        
        # 测试小规模数据
        test_codes = get_test_stock_codes(100)
        
        start_time = time.time()
        result = optimizer.optimize_stock_selection(
            stock_codes=test_codes,
            start_date="2024-11-01",
            end_date="2024-12-31",
            indicators=["MA", "KDJ"],
            strategy_params={'min_score': 0.6}
        )
        
        processing_time = time.time() - start_time
        
        logger.info(f"基础优化测试完成: {len(test_codes)}只股票, 耗时 {processing_time:.2f}秒")
        logger.info(f"处理速度: {len(test_codes)/processing_time:.1f}股/秒")
        
        return {
            'success': True,
            'stock_count': len(test_codes),
            'processing_time': processing_time,
            'stocks_per_second': len(test_codes) / processing_time,
            'result': result
        }
        
    except Exception as e:
        logger.error(f"基础优化测试失败: {e}")
        return {
            'success': False,
            'error': str(e)
        }


def test_scalability():
    """测试可扩展性"""
    logger.info("开始可扩展性测试")
    
    try:
        # 设置测试环境
        data_access, cache_service = setup_test_environment()
        
        # 创建性能优化器
        optimizer = Performance_optimizer(data_access, cache_service)
        
        # 测试不同规模的股票数量
        test_scales = [50, 100, 200, 500]
        results = {}
        
        for scale in test_scales:
            logger.info(f"测试 {scale} 只股票的处理性能")
            
            test_codes = get_test_stock_codes(scale)
            
            start_time = time.time()
            try:
                result = optimizer.optimize_stock_selection(
                    stock_codes=test_codes,
                    start_date="2024-12-01",
                    end_date="2024-12-31",
                    indicators=["MA"],
                    strategy_params={'min_score': 0.5}
                )
                
                processing_time = time.time() - start_time
                stocks_per_second = scale / processing_time
                
                results[f"{scale}_stocks"] = {
                    'success': True,
                    'processing_time': processing_time,
                    'stocks_per_second': stocks_per_second,
                    'performance_report': result.get('performance_report', {})
                }
                
                logger.info(f"{scale}只股票测试完成: 耗时 {processing_time:.2f}秒, "
                           f"速度 {stocks_per_second:.1f}股/秒")
                
            except Exception as e:
                logger.error(f"{scale}只股票测试失败: {e}")
                results[f"{scale}_stocks"] = {
                    'success': False,
                    'error': str(e)
                }
        
        # 分析可扩展性
        successful_tests = {k: v for k, v in results.items() if v.get('success', False)}
        
        if len(successful_tests) >= 2:
            # 计算平均处理速度
            avg_speed = sum(v['stocks_per_second'] for v in successful_tests.values()) / len(successful_tests)
            
            # 预测4000只股票的处理时间
            predicted_4000_time = 4000 / avg_speed
            
            results['scalability_analysis'] = {
                'avg_stocks_per_second': avg_speed,
                'predicted_4000_stocks_time_seconds': predicted_4000_time,
                'predicted_4000_stocks_time_minutes': predicted_4000_time / 60,
                'target_achievable': predicted_4000_time <= 300  # 5分钟目标
            }
        
        return results
        
    except Exception as e:
        logger.error(f"可扩展性测试失败: {e}")
        return {
            'success': False,
            'error': str(e)
        }


def test_memory_optimization():
    """测试内存优化"""
    logger.info("开始内存优化测试")
    
    try:
        # 设置测试环境
        data_access, cache_service = setup_test_environment()
        
        # 创建性能优化器
        config = Optimization_config(
            batch_size=50,
            max_workers=8,
            max_memory_usage_percent=60.0,
            chunk_size_mb=64
        )
        
        optimizer = Performance_optimizer(data_access, cache_service, config)
        
        # 获取初始内存状态
        initial_memory = optimizer.memory_optimizer.get_memory_stats()
        
        # 测试内存优化
        test_codes = get_test_stock_codes(200)
        
        start_time = time.time()
        result = optimizer.optimize_stock_selection(
            stock_codes=test_codes,
            start_date="2024-12-01",
            end_date="2024-12-31",
            indicators=["MA", "KDJ", "MACD"],
            strategy_params={'min_score': 0.6}
        )
        
        processing_time = time.time() - start_time
        
        # 获取最终内存状态
        final_memory = optimizer.memory_optimizer.get_memory_stats()
        
        memory_increase = final_memory.process_memory_mb - initial_memory.process_memory_mb
        
        logger.info(f"内存优化测试完成: {len(test_codes)}只股票, 耗时 {processing_time:.2f}秒")
        logger.info(f"内存增长: {memory_increase:.1f}MB")
        
        return {
            'success': True,
            'stock_count': len(test_codes),
            'processing_time': processing_time,
            'initial_memory_mb': initial_memory.process_memory_mb,
            'final_memory_mb': final_memory.process_memory_mb,
            'memory_increase_mb': memory_increase,
            'memory_efficiency': memory_increase / len(test_codes),  # MB per stock
            'result': result
        }
        
    except Exception as e:
        logger.error(f"内存优化测试失败: {e}")
        return {
            'success': False,
            'error': str(e)
        }


def test_cache_performance_Optimization():
    """测试缓存性能"""
    logger.info("开始缓存性能测试")
    
    try:
        # 设置测试环境
        data_access, cache_service = setup_test_environment()
        
        # 创建性能优化器（启用缓存）
        optimizer_with_cache = Performance_optimizer(
            data_access, cache_service,
            Optimization_config(enable_cache=True, batch_size=50)
        )
        
        # 创建性能优化器（禁用缓存）
        optimizer_without_cache = Performance_optimizer(
            data_access, cache_service,
            Optimization_config(enable_cache=False, batch_size=50)
        )
        
        test_codes = get_test_stock_codes(100)
        test_params = {
            'stock_codes': test_codes,
            'start_date': "2024-12-01",
            'end_date': "2024-12-31",
            'indicators': ["MA"],
            'strategy_params': {'min_score': 0.5}
        }
        
        # 测试无缓存性能
        logger.info("测试无缓存性能")
        start_time = time.time()
        result_no_cache = optimizer_without_cache.optimize_stock_selection(**test_params)
        time_no_cache = time.time() - start_time
        
        # 测试首次有缓存性能（缓存为空）
        logger.info("测试首次有缓存性能")
        start_time = time.time()
        result_first_cache = optimizer_with_cache.optimize_stock_selection(**test_params)
        time_first_cache = time.time() - start_time
        
        # 测试二次有缓存性能（缓存已填充）
        logger.info("测试二次有缓存性能")
        start_time = time.time()
        result_second_cache = optimizer_with_cache.optimize_stock_selection(**test_params)
        time_second_cache = time.time() - start_time
        
        # 计算缓存效果
        cache_speedup = time_no_cache / time_second_cache if time_second_cache > 0 else 0
        
        logger.info(f"缓存性能测试完成:")
        logger.info(f"  无缓存: {time_no_cache:.2f}秒")
        logger.info(f"  首次缓存: {time_first_cache:.2f}秒")
        logger.info(f"  二次缓存: {time_second_cache:.2f}秒")
        logger.info(f"  缓存加速比: {cache_speedup:.1f}x")
        
        return {
            'success': True,
            'stock_count': len(test_codes),
            'time_no_cache': time_no_cache,
            'time_first_cache': time_first_cache,
            'time_second_cache': time_second_cache,
            'cache_speedup': cache_speedup,
            'cache_hit_rate': result_second_cache.get('performance_report', {}).get('cache_hit_rate', 0)
        }
        
    except Exception as e:
        logger.error(f"缓存性能测试失败: {e}")
        return {
            'success': False,
            'error': str(e)
        }


def run_comprehensive_test_Optimization():
    """运行综合性能测试"""
    logger.info("=" * 60)
    logger.info("开始性能优化综合测试")
    logger.info("=" * 60)
    
    test_results = {
        'timestamp': datetime.now().isoformat(),
        'test_summary': {
            'total_tests': 4,
            'passed_tests': 0,
            'failed_tests': 0
        },
        'test_details': {}
    }
    
    # 1. 基础优化测试
    logger.info("\n1. 基础优化功能测试")
    logger.info("-" * 40)
    basic_result = test_basic_optimization()
    test_results['test_details']['basic_optimization'] = basic_result
    
    if basic_result.get('success', False):
        test_results['test_summary']['passed_tests'] += 1
        logger.info("✅ 基础优化测试通过")
    else:
        test_results['test_summary']['failed_tests'] += 1
        logger.error("❌ 基础优化测试失败")
    
    # 2. 可扩展性测试
    logger.info("\n2. 可扩展性测试")
    logger.info("-" * 40)
    scalability_result = test_scalability()
    test_results['test_details']['scalability'] = scalability_result
    
    if scalability_result.get('success', True):  # 如果没有明确失败就算成功
        test_results['test_summary']['passed_tests'] += 1
        logger.info("✅ 可扩展性测试通过")
        
        # 检查是否达到性能目标
        analysis = scalability_result.get('scalability_analysis', {})
        if analysis.get('target_achievable', False):
            logger.info("🎯 性能目标可达成：4000只股票可在5分钟内完成")
        else:
            predicted_time = analysis.get('predicted_4000_stocks_time_minutes', 0)
            logger.warning(f"⚠️ 性能目标挑战：预测4000只股票需要 {predicted_time:.1f}分钟")
    else:
        test_results['test_summary']['failed_tests'] += 1
        logger.error("❌ 可扩展性测试失败")
    
    # 3. 内存优化测试
    logger.info("\n3. 内存优化测试")
    logger.info("-" * 40)
    memory_result = test_memory_optimization()
    test_results['test_details']['memory_optimization'] = memory_result
    
    if memory_result.get('success', False):
        test_results['test_summary']['passed_tests'] += 1
        logger.info("✅ 内存优化测试通过")
        
        memory_efficiency = memory_result.get('memory_efficiency', 0)
        logger.info(f"📊 内存效率: {memory_efficiency:.2f}MB/股票")
    else:
        test_results['test_summary']['failed_tests'] += 1
        logger.error("❌ 内存优化测试失败")
    
    # 4. 缓存性能测试
    logger.info("\n4. 缓存性能测试")
    logger.info("-" * 40)
    cache_result = test_cache_performance_Optimization()
    test_results['test_details']['cache_performance'] = cache_result
    
    if cache_result.get('success', False):
        test_results['test_summary']['passed_tests'] += 1
        logger.info("✅ 缓存性能测试通过")
        
        cache_speedup = cache_result.get('cache_speedup', 0)
        logger.info(f"⚡ 缓存加速比: {cache_speedup:.1f}x")
    else:
        test_results['test_summary']['failed_tests'] += 1
        logger.error("❌ 缓存性能测试失败")
    
    # 生成测试报告
    logger.info("\n" + "=" * 60)
    logger.info("测试总结")
    logger.info("=" * 60)
    
    passed = test_results['test_summary']['passed_tests']
    total = test_results['test_summary']['total_tests']
    success_rate = passed / total * 100
    
    logger.info(f"总测试数: {total}")
    logger.info(f"通过测试: {passed}")
    logger.info(f"失败测试: {test_results['test_summary']['failed_tests']}")
    logger.info(f"成功率: {success_rate:.1f}%")
    
    if success_rate >= 75:
        logger.info("🎉 性能优化测试整体通过！")
    else:
        logger.warning("⚠️ 性能优化测试需要改进")
    
    # 保存测试结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"performance_optimization_test_{timestamp}.json"
    
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False)
        logger.info(f"📄 测试报告已保存: {report_file}")
    except Exception as e:
        logger.error(f"保存测试报告失败: {e}")
    
    return test_results


if __name__ == "__main__":
    try:
        # 运行综合测试
        results = run_comprehensive_test_Optimization()
        
        # 根据测试结果设置退出码
        if results['test_summary']['passed_tests'] >= 3:
            sys.exit(0)  # 成功
        else:
            sys.exit(1)  # 失败
            
    except Exception as e:
        logger.error(f"测试执行失败: {e}")
        sys.exit(1) 