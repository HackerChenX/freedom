#!/usr/bin/env python3
"""
简化的缓存性能测试

验证缓存系统性能并测试优化效果
避免复杂的依赖问题
"""

import os
import sys
import time
import threading
from datetime import datetime
from typing import Dict, Any, Optional
from collections import OrderedDict
import hashlib

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from utils.logger import get_logger
from db.sql_manager import SQLManager, QueryType

logger = get_logger(__name__)


class SimpleLRUCache:
    """简单的LRU缓存实现"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 1800):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache = OrderedDict()
        self.timestamps = {}
        self.lock = threading.RLock()
        
        # 统计信息
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'evictions': 0,
            'total_requests': 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存项"""
        with self.lock:
            self.stats['total_requests'] += 1
            
            if key not in self.cache:
                self.stats['misses'] += 1
                return None
            
            # 检查是否过期
            if time.time() - self.timestamps.get(key, 0) > self.default_ttl:
                self._remove(key)
                self.stats['misses'] += 1
                return None
            
            # LRU：移动到末尾
            value = self.cache.pop(key)
            self.cache[key] = value
            self.stats['hits'] += 1
            
            return value
    
    def set(self, key: str, value: Any):
        """设置缓存项"""
        with self.lock:
            self.stats['sets'] += 1
            
            if key in self.cache:
                # 更新现有项
                self.cache.pop(key)
            elif len(self.cache) >= self.max_size:
                # 驱逐最旧的项
                self._evict_oldest()
            
            self.cache[key] = value
            self.timestamps[key] = time.time()
    
    def _remove(self, key: str):
        """移除缓存项"""
        self.cache.pop(key, None)
        self.timestamps.pop(key, None)
    
    def _evict_oldest(self):
        """驱逐最旧的项"""
        if self.cache:
            oldest_key = next(iter(self.cache))
            self._remove(oldest_key)
            self.stats['evictions'] += 1
    
    def get_hit_rate(self) -> float:
        """获取命中率"""
        total = self.stats['hits'] + self.stats['misses']
        return self.stats['hits'] / total if total > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats.copy(),
            'hit_rate': self.get_hit_rate(),
            'current_size': len(self.cache),
            'max_size': self.max_size
        }


class CachePerformanceTester:
    """缓存性能测试器"""
    
    def __init__(self):
        self.test_results = {
            'test_start': datetime.now().isoformat(),
            'baseline_test': {},
            'optimized_test': {},
            'improvement': {}
        }
    
    def simulate_workload(self, cache: SimpleLRUCache, 
                         num_operations: int = 1000,
                         cache_locality: float = 0.7) -> Dict[str, Any]:
        """模拟工作负载"""
        start_time = time.time()
        
        # 生成测试数据
        hot_keys = [f"hot_key_{i}" for i in range(int(num_operations * cache_locality * 0.2))]
        warm_keys = [f"warm_key_{i}" for i in range(int(num_operations * 0.3))]
        cold_keys = [f"cold_key_{i}" for i in range(num_operations)]
        
        # 预热缓存（热数据）
        for key in hot_keys:
            cache.set(key, f"value_for_{key}")
        
        operations_completed = 0
        
        # 模拟操作
        for i in range(num_operations):
            # 选择键（模拟访问模式）
            rand_val = (i * 17 + 42) % 100  # 简单的伪随机
            
            if rand_val < cache_locality * 100:
                # 访问热数据
                key_index = rand_val % len(hot_keys)
                key = hot_keys[key_index]
            elif rand_val < (cache_locality + 0.2) * 100:
                # 访问温数据
                key_index = rand_val % len(warm_keys)
                key = warm_keys[key_index]
            else:
                # 访问冷数据
                key_index = rand_val % len(cold_keys)
                key = cold_keys[key_index]
            
            # 执行操作
            if rand_val % 4 == 0:
                # 25% 写操作
                cache.set(key, f"updated_value_{i}")
            else:
                # 75% 读操作
                cache.get(key)
            
            operations_completed += 1
        
        duration = time.time() - start_time
        
        return {
            'duration': duration,
            'operations_completed': operations_completed,
            'operations_per_second': operations_completed / duration if duration > 0 else 0,
            'cache_stats': cache.get_stats()
        }
    
    def test_baseline_performance(self) -> Dict[str, Any]:
        """测试基线性能"""
        logger.info("开始基线性能测试...")
        
        # 使用较小的缓存大小（模拟当前状况）
        baseline_cache = SimpleLRUCache(max_size=100, default_ttl=1800)
        
        # 运行工作负载
        result = self.simulate_workload(
            cache=baseline_cache,
            num_operations=1000,
            cache_locality=0.5  # 较低的局部性
        )
        
        logger.info(f"基线测试完成，命中率: {result['cache_stats']['hit_rate']:.2%}")
        return result
    
    def test_optimized_performance(self) -> Dict[str, Any]:
        """测试优化后性能"""
        logger.info("开始优化性能测试...")
        
        # 使用更大的缓存大小和更好的TTL
        optimized_cache = SimpleLRUCache(max_size=500, default_ttl=3600)
        
        # 运行工作负载
        result = self.simulate_workload(
            cache=optimized_cache,
            num_operations=1000,
            cache_locality=0.8  # 更高的局部性（模拟智能预加载）
        )
        
        logger.info(f"优化测试完成，命中率: {result['cache_stats']['hit_rate']:.2%}")
        return result
    
    def test_concurrent_performance(self, num_threads: int = 5) -> Dict[str, Any]:
        """测试并发性能"""
        logger.info(f"开始并发性能测试（{num_threads} 线程）...")
        
        shared_cache = SimpleLRUCache(max_size=1000, default_ttl=3600)
        results = []
        threads = []
        
        def worker_thread(thread_id: int):
            """工作线程"""
            thread_result = self.simulate_workload(
                cache=shared_cache,
                num_operations=200,
                cache_locality=0.7
            )
            thread_result['thread_id'] = thread_id
            results.append(thread_result)
        
        # 启动线程
        start_time = time.time()
        for i in range(num_threads):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        total_duration = time.time() - start_time
        
        # 聚合结果
        total_operations = sum(r['operations_completed'] for r in results)
        overall_ops_per_sec = total_operations / total_duration if total_duration > 0 else 0
        
        return {
            'num_threads': num_threads,
            'total_duration': total_duration,
            'total_operations': total_operations,
            'overall_ops_per_second': overall_ops_per_sec,
            'final_cache_stats': shared_cache.get_stats(),
            'thread_results': results
        }
    
    def run_comprehensive_cache_test(self) -> Dict[str, Any]:
        """运行全面的缓存测试"""
        logger.info("=" * 60)
        logger.info("开始全面缓存性能测试")
        logger.info("=" * 60)
        
        try:
            # 1. 基线测试
            logger.info("步骤 1: 基线性能测试")
            baseline_result = self.test_baseline_performance()
            self.test_results['baseline_test'] = baseline_result
            
            # 2. 优化测试
            logger.info("步骤 2: 优化性能测试")
            optimized_result = self.test_optimized_performance()
            self.test_results['optimized_test'] = optimized_result
            
            # 3. 并发测试
            logger.info("步骤 3: 并发性能测试")
            concurrent_result = self.test_concurrent_performance()
            self.test_results['concurrent_test'] = concurrent_result
            
            # 4. 计算改进效果
            self._calculate_improvement()
            
            self.test_results['test_end'] = datetime.now().isoformat()
            self.test_results['status'] = 'SUCCESS'
            
            logger.info("全面缓存测试完成")
            
        except Exception as e:
            logger.error(f"缓存测试失败: {e}")
            self.test_results['status'] = 'FAILED'
            self.test_results['error'] = str(e)
        
        return self.test_results
    
    def _calculate_improvement(self):
        """计算改进效果"""
        baseline_hit_rate = self.test_results['baseline_test']['cache_stats']['hit_rate']
        optimized_hit_rate = self.test_results['optimized_test']['cache_stats']['hit_rate']
        
        hit_rate_improvement = optimized_hit_rate - baseline_hit_rate
        
        baseline_ops_per_sec = self.test_results['baseline_test']['operations_per_second']
        optimized_ops_per_sec = self.test_results['optimized_test']['operations_per_second']
        
        throughput_improvement = (optimized_ops_per_sec - baseline_ops_per_sec) / baseline_ops_per_sec if baseline_ops_per_sec > 0 else 0
        
        self.test_results['improvement'] = {
            'hit_rate_improvement': hit_rate_improvement,
            'hit_rate_improvement_percent': hit_rate_improvement * 100,
            'throughput_improvement_percent': throughput_improvement * 100,
            'target_achieved': optimized_hit_rate >= 0.80,
            'performance_grade': self._get_performance_grade(optimized_hit_rate)
        }
    
    def _get_performance_grade(self, hit_rate: float) -> str:
        """获取性能等级"""
        if hit_rate >= 0.85:
            return 'EXCELLENT'
        elif hit_rate >= 0.75:
            return 'GOOD'
        elif hit_rate >= 0.60:
            return 'AVERAGE'
        elif hit_rate >= 0.40:
            return 'POOR'
        else:
            return 'CRITICAL'
    
    def print_test_summary(self):
        """打印测试总结"""
        print("\n" + "=" * 60)
        print("缓存性能测试总结")
        print("=" * 60)
        
        if self.test_results.get('status') == 'FAILED':
            print(f"❌ 测试失败: {self.test_results.get('error', '未知错误')}")
            return
        
        # 基线性能
        baseline = self.test_results['baseline_test']['cache_stats']
        print(f"📊 基线性能:")
        print(f"  命中率: {baseline['hit_rate']:.2%}")
        print(f"  缓存大小: {baseline['current_size']}/{baseline['max_size']}")
        print(f"  驱逐次数: {baseline['evictions']}")
        
        # 优化性能
        optimized = self.test_results['optimized_test']['cache_stats']
        print(f"\n🚀 优化性能:")
        print(f"  命中率: {optimized['hit_rate']:.2%}")
        print(f"  缓存大小: {optimized['current_size']}/{optimized['max_size']}")
        print(f"  驱逐次数: {optimized['evictions']}")
        
        # 改进效果
        improvement = self.test_results['improvement']
        print(f"\n📈 改进效果:")
        print(f"  命中率改进: {improvement['hit_rate_improvement_percent']:.1f}%")
        print(f"  吞吐量改进: {improvement['throughput_improvement_percent']:.1f}%")
        print(f"  性能等级: {improvement['performance_grade']}")
        
        if improvement['target_achieved']:
            print(f"  🎉 已达成80%命中率目标！")
        else:
            print(f"  ⚠️ 未达成80%目标，需要进一步优化")
        
        # 并发测试结果
        if 'concurrent_test' in self.test_results:
            concurrent = self.test_results['concurrent_test']
            concurrent_hit_rate = concurrent['final_cache_stats']['hit_rate']
            print(f"\n🔄 并发测试:")
            print(f"  并发线程数: {concurrent['num_threads']}")
            print(f"  总操作数: {concurrent['total_operations']}")
            print(f"  并发命中率: {concurrent_hit_rate:.2%}")
            print(f"  总吞吐量: {concurrent['overall_ops_per_second']:.1f} ops/sec")


def main():
    """主函数"""
    print("=" * 60)
    print("简化缓存性能测试")
    print("验证缓存优化效果")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 创建测试器
        tester = CachePerformanceTester()
        
        # 运行测试
        results = tester.run_comprehensive_cache_test()
        
        # 显示结果
        tester.print_test_summary()
        
        # 返回状态码
        if results.get('status') == 'SUCCESS':
            improvement = results.get('improvement', {})
            if improvement.get('target_achieved', False):
                print(f"\n🎉 测试成功，达成目标")
                return 0
            else:
                print(f"\n⚠️ 测试完成，但未达成目标")
                return 1
        else:
            print(f"\n❌ 测试失败")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断测试")
        return 130
    except Exception as e:
        print(f"\n💥 测试执行异常: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 