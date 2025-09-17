#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
查询缓存性能测试 - 任务5.2验证脚本

测试智能查询缓存的性能提升效果，包括：
- 缓存命中率测试
- 查询响应时间对比
- 内存和磁盘缓存效果
- 缓存一致性验证
"""

import time
import sys
import os
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime, timedelta

# 添加项目路径
sys.path.insert(0, '/Users/hacker/PycharmProjects/freedom')

from db.enhanced_connection_pool import get_connection_pool
from utils.logger import getLogger
from db.sql_manager import SQLManager, QueryType

logger = getLogger(__name__)


class CachePerformanceTester:
    """查询缓存性能测试器"""
    
    def __init__(self):
        """初始化测试器"""
        self.pool = get_connection_pool()
        self.test_queries = self._prepare_test_queries()
        self.results = {}
        
    def _prepare_test_queries(self) -> List[str]:
        """准备测试查询"""
        return [
            # 基础股票信息查询
            "SELECT code, name, date, close FROM stock_info WHERE code = '000001' AND level = '日线' ORDER BY date DESC LIMIT 100",
            
            # 时间范围查询
            "SELECT code, name, date, open, close, high, low, volume FROM stock_info WHERE code = %(code)s AND date >= '2024-01-01' AND date <= '2024-01-31' AND level = '日线' LIMIT 1000",
            
            # 聚合查询
            "SELECT code, COUNT(*) as count, AVG(close) as avg_close FROM stock_info WHERE code = %(code)s AND level = '日线' GROUP BY code LIMIT 50",
            
            # 复杂条件查询
            "SELECT code, name, date, close, volume FROM stock_info WHERE code = %(code)s AND close > 10 AND volume > 1000000 AND level = '日线' ORDER BY date DESC LIMIT 200",
            
            # 行业统计查询
            "SELECT industry, COUNT(*) as stock_count FROM stock_info WHERE code = %(code)s AND level = '日线' GROUP BY industry LIMIT 20"
        ]
    
    def test_cache_performance(self) -> Dict[str, Any]:
        """测试缓存性能"""
        print("🚀 开始查询缓存性能测试")
        print("=" * 70)
        
        results = {
            'cache_disabled_times': [],
            'cache_enabled_times': [],
            'cache_stats': {},
            'performance_improvement': {}
        }
        
        # 1. 测试无缓存性能
        print("\n📊 1. 无缓存性能测试")
        self.pool.enable_query_cache = False
        no_cache_times = self._run_query_tests("无缓存")
        results['cache_disabled_times'] = no_cache_times
        
        # 2. 清空缓存并启用
        print("\n📊 2. 启用缓存性能测试")
        self.pool.enable_query_cache = True
        if self.pool.query_cache:
            self.pool.query_cache.clear_cache()
        
        # 3. 第一次运行（填充缓存）
        print("\n📊 3. 缓存填充测试")
        first_run_times = self._run_query_tests("缓存填充")
        
        # 4. 第二次运行（缓存命中）
        print("\n📊 4. 缓存命中测试")
        cached_times = self._run_query_tests("缓存命中")
        results['cache_enabled_times'] = cached_times
        
        # 5. 获取缓存统计
        if self.pool.query_cache:
            results['cache_stats'] = self.pool.get_cache_stats()
        
        # 6. 计算性能提升
        results['performance_improvement'] = self._calculate_improvement(
            no_cache_times, cached_times
        )
        
        return results
    
    def _run_query_tests(self, test_name: str) -> List[float]:
        """运行查询测试"""
        times = []
        
        for i, query in enumerate(self.test_queries):
            try:
                start_time = time.time()
                
                with self.pool.get_connection() as conn:
                    result = conn.query_dataframe(query)
                
                end_time = time.time()
                query_time = end_time - start_time
                times.append(query_time)
                
                print(f"  查询 {i+1}: {query_time:.4f}s ({len(result)} 行)")
                
            except Exception as e:
                logger.error(f"查询失败: {e}")
                times.append(float('inf'))
        
        avg_time = sum(t for t in times if t != float('inf')) / len([t for t in times if t != float('inf')])
        print(f"  {test_name} 平均时间: {avg_time:.4f}s")
        
        return times
    
    def _calculate_improvement(self, no_cache_times: List[float], cached_times: List[float]) -> Dict[str, Any]:
        """计算性能提升"""
        improvements = []
        
        for i, (no_cache, cached) in enumerate(zip(no_cache_times, cached_times)):
            if no_cache != float('inf') and cached != float('inf') and no_cache > 0:
                improvement = ((no_cache - cached) / no_cache) * 100
                improvements.append(improvement)
            else:
                improvements.append(0)
        
        avg_improvement = sum(improvements) / len(improvements) if improvements else 0
        max_improvement = max(improvements) if improvements else 0
        
        return {
            'individual_improvements': improvements,
            'average_improvement': avg_improvement,
            'max_improvement': max_improvement,
            'total_no_cache_time': sum(no_cache_times),
            'total_cached_time': sum(cached_times)
        }
    
    def test_cache_consistency(self) -> Dict[str, Any]:
        """测试缓存一致性"""
        print("\n🔍 缓存一致性测试")
        print("=" * 50)
        
        test_query = self.test_queries[0]  # 使用第一个查询
        
        # 清空缓存
        if self.pool.query_cache:
            self.pool.query_cache.clear_cache()
        
        # 第一次查询
        with self.pool.get_connection() as conn:
            result1 = conn.query_dataframe(test_query)
        
        # 第二次查询（应该从缓存获取）
        with self.pool.get_connection() as conn:
            result2 = conn.query_dataframe(test_query)
        
        # 比较结果
        is_consistent = result1.equals(result2) if not result1.empty and not result2.empty else True
        
        return {
            'consistent': is_consistent,
            'result1_shape': result1.shape,
            'result2_shape': result2.shape,
            'test_query': test_query
        }
    
    def test_cache_warmup(self) -> Dict[str, Any]:
        """测试缓存预热"""
        print("\n🔥 缓存预热测试")
        print("=" * 50)
        
        # 清空缓存
        if self.pool.query_cache:
            self.pool.query_cache.clear_cache()
        
        # 执行预热
        warmup_result = self.pool.warm_up_cache(self.test_queries)
        
        # 验证预热效果
        cache_stats = self.pool.get_cache_stats()
        
        return {
            'warmup_result': warmup_result,
            'cache_stats_after_warmup': cache_stats
        }


def main():
    """主测试函数"""
    print("🔧 任务5.2: 查询缓存机制增强 - 性能验证测试")
    print("=" * 70)
    
    try:
        tester = CachePerformanceTester()
        
        # 1. 缓存性能测试
        performance_results = tester.test_cache_performance()
        
        # 2. 缓存一致性测试
        consistency_results = tester.test_cache_consistency()
        
        # 3. 缓存预热测试
        warmup_results = tester.test_cache_warmup()
        
        # 4. 输出综合报告
        print("\n🎯 查询缓存性能测试报告")
        print("=" * 70)
        
        # 性能提升报告
        improvement = performance_results['performance_improvement']
        print(f"📈 性能提升:")
        print(f"  平均提升: {improvement['average_improvement']:.2f}%")
        print(f"  最大提升: {improvement['max_improvement']:.2f}%")
        print(f"  总时间对比: {improvement['total_no_cache_time']:.4f}s → {improvement['total_cached_time']:.4f}s")
        
        # 缓存统计报告
        if 'cache_stats' in performance_results:
            cache_stats = performance_results['cache_stats']
            print(f"\n📊 缓存统计:")
            print(f"  总请求数: {cache_stats.get('total_requests', 0)}")
            print(f"  缓存命中数: {cache_stats.get('cache_hits', 0)}")
            print(f"  命中率: {cache_stats.get('hit_rate', 0):.2%}")
            print(f"  内存命中率: {cache_stats.get('memory_hit_rate', 0):.2%}")
            print(f"  内存缓存大小: {cache_stats.get('memory_cache_size', 0)}")
        
        # 一致性报告
        print(f"\n🔍 缓存一致性:")
        print(f"  数据一致性: {'✅ 通过' if consistency_results['consistent'] else '❌ 失败'}")
        
        # 预热报告
        warmup = warmup_results['warmup_result']
        print(f"\n🔥 缓存预热:")
        print(f"  预热成功: {warmup['warmed_queries']}/{warmup['total_queries']}")
        print(f"  预热失败: {warmup['failed_queries']}")
        
        # 最终评估
        print(f"\n🎉 任务5.2查询缓存机制增强 - 测试结果")
        print("=" * 70)
        
        success_criteria = {
            'performance_improvement': improvement['average_improvement'] > 10,  # 平均提升>10%
            'cache_hit_rate': cache_stats.get('hit_rate', 0) > 0.5,  # 命中率>50%
            'consistency': consistency_results['consistent'],  # 数据一致性
            'warmup_success': warmup['warmed_queries'] > 0  # 预热成功
        }
        
        all_passed = all(success_criteria.values())
        
        for criterion, passed in success_criteria.items():
            status = "✅ 通过" if passed else "❌ 失败"
            print(f"  {criterion}: {status}")
        
        if all_passed:
            print("\n🎉 查询缓存机制增强 - 100%成功！")
            print("🚀 缓存性能显著提升，系统优化达到预期目标")
        else:
            print("\n⚠️ 部分测试未通过，需要进一步优化")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ 缓存性能测试失败: {e}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
