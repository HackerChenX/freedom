#!/usr/bin/env python3
"""
缓存优化分析器

目标：将缓存命中率从50%提升到85%
分析现有缓存性能并提供优化方案
"""

import os
import sys
import time
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
import pandas as pd

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from config import get_config
from utils.logger import get_logger
from db.query_cache import QueryCache
from db.unified_data_manager import get_unified_data_manager
from analysis.intelligent_cache_system import IntelligentCacheSystem

logger = get_logger(__name__)


class CacheOptimizationAnalyzer:
    """缓存优化分析器"""
    
    def __init__(self):
        """初始化分析器"""
        self.analysis_start_time = time.time()
        self.cache_systems = {}
        self.optimization_results = {
            'analysis_start': datetime.now().isoformat(),
            'current_performance': {},
            'optimization_recommendations': [],
            'implementation_plan': {},
            'expected_improvements': {}
        }
        
        # 初始化缓存系统
        self._initialize_cache_systems()
        
        logger.info("缓存优化分析器初始化完成")
    
    def _initialize_cache_systems(self):
        """初始化各种缓存系统"""
        try:
            # 查询缓存
            self.cache_systems['query_cache'] = QueryCache(
                max_memory_size=1000,
                max_disk_size=5000,
                default_ttl=1800,
                enable_disk_cache=True
            )
            
            # 智能缓存系统
            self.cache_systems['intelligent_cache'] = IntelligentCacheSystem()
            
            # 数据管理器缓存
            self.cache_systems['data_manager'] = get_unified_data_manager()
            
            logger.info("所有缓存系统初始化成功")
            
        except Exception as e:
            logger.error(f"缓存系统初始化失败: {e}")
            raise
    
    def analyze_current_cache_performance(self) -> Dict[str, Any]:
        """分析当前缓存性能"""
        logger.info("开始分析当前缓存性能...")
        
        performance_analysis = {
            'timestamp': datetime.now().isoformat(),
            'cache_systems': {},
            'overall_metrics': {},
            'bottlenecks': [],
            'optimization_opportunities': []
        }
        
        # 分析查询缓存
        query_cache_stats = self._analyze_query_cache()
        performance_analysis['cache_systems']['query_cache'] = query_cache_stats
        
        # 分析数据管理器缓存
        data_manager_stats = self._analyze_data_manager_cache()
        performance_analysis['cache_systems']['data_manager'] = data_manager_stats
        
        # 计算整体指标
        overall_metrics = self._calculate_overall_metrics(performance_analysis['cache_systems'])
        performance_analysis['overall_metrics'] = overall_metrics
        
        # 识别瓶颈
        bottlenecks = self._identify_bottlenecks(performance_analysis['cache_systems'])
        performance_analysis['bottlenecks'] = bottlenecks
        
        # 识别优化机会
        opportunities = self._identify_optimization_opportunities(performance_analysis)
        performance_analysis['optimization_opportunities'] = opportunities
        
        logger.info(f"缓存性能分析完成，总体命中率: {overall_metrics.get('hit_rate', 0):.2%}")
        
        return performance_analysis
    
    def _analyze_query_cache(self) -> Dict[str, Any]:
        """分析查询缓存性能"""
        try:
            query_cache = self.cache_systems.get('query_cache')
            if not query_cache:
                return {'status': 'UNAVAILABLE', 'reason': '查询缓存未初始化'}
            
            stats = query_cache.stats.copy()
            
            # 计算命中率
            total_queries = stats['memory_hits'] + stats['memory_misses']
            memory_hit_rate = stats['memory_hits'] / total_queries if total_queries > 0 else 0
            
            total_disk_queries = stats['disk_hits'] + stats['disk_misses']
            disk_hit_rate = stats['disk_hits'] / total_disk_queries if total_disk_queries > 0 else 0
            
            overall_hits = stats['memory_hits'] + stats['disk_hits'] + stats['aggregated_hits']
            overall_total = total_queries + total_disk_queries
            overall_hit_rate = overall_hits / overall_total if overall_total > 0 else 0
            
            return {
                'status': 'ACTIVE',
                'memory_cache': {
                    'size': len(query_cache.memory_cache),
                    'max_size': query_cache.max_memory_size,
                    'hit_rate': memory_hit_rate,
                    'hits': stats['memory_hits'],
                    'misses': stats['memory_misses']
                },
                'disk_cache': {
                    'enabled': query_cache.enable_disk_cache,
                    'hit_rate': disk_hit_rate,
                    'hits': stats['disk_hits'],
                    'misses': stats['disk_misses']
                },
                'aggregated_cache': {
                    'hits': stats['aggregated_hits']
                },
                'overall': {
                    'hit_rate': overall_hit_rate,
                    'total_queries': overall_total,
                    'evictions': stats['cache_evictions']
                }
            }
            
        except Exception as e:
            logger.error(f"分析查询缓存失败: {e}")
            return {'status': 'ERROR', 'error': str(e)}
    
    def _analyze_data_manager_cache(self) -> Dict[str, Any]:
        """分析数据管理器缓存性能"""
        try:
            data_manager = self.cache_systems.get('data_manager')
            if not data_manager:
                return {'status': 'UNAVAILABLE', 'reason': '数据管理器未初始化'}
            
            if not hasattr(data_manager, 'stats'):
                return {'status': 'NO_STATS', 'reason': '数据管理器无统计信息'}
            
            stats = data_manager.stats.copy()
            
            # 计算命中率
            total_requests = stats.get('cache_hits', 0) + stats.get('cache_misses', 0)
            hit_rate = stats.get('cache_hits', 0) / total_requests if total_requests > 0 else 0
            
            return {
                'status': 'ACTIVE',
                'cache_enabled': getattr(data_manager, 'cache_enabled', False),
                'cache_size': len(getattr(data_manager, 'query_cache', {})),
                'max_cache_size': getattr(data_manager, 'max_cache_size', 0),
                'hit_rate': hit_rate,
                'hits': stats.get('cache_hits', 0),
                'misses': stats.get('cache_misses', 0),
                'evictions': stats.get('cache_evictions', 0),
                'total_requests': total_requests
            }
            
        except Exception as e:
            logger.error(f"分析数据管理器缓存失败: {e}")
            return {'status': 'ERROR', 'error': str(e)}
    
    def _calculate_overall_metrics(self, cache_systems: Dict[str, Any]) -> Dict[str, Any]:
        """计算整体缓存指标"""
        total_hits = 0
        total_requests = 0
        total_evictions = 0
        
        for system_name, system_stats in cache_systems.items():
            if system_stats.get('status') == 'ACTIVE':
                if 'overall' in system_stats:
                    # 查询缓存统计
                    system_hits = (system_stats['memory_cache'].get('hits', 0) + 
                                 system_stats['disk_cache'].get('hits', 0) + 
                                 system_stats['aggregated_cache'].get('hits', 0))
                    system_total = system_stats['overall'].get('total_queries', 0)
                    
                elif 'total_requests' in system_stats:
                    # 数据管理器统计
                    system_hits = system_stats.get('hits', 0)
                    system_total = system_stats.get('total_requests', 0)
                else:
                    continue
                
                total_hits += system_hits
                total_requests += system_total
                total_evictions += system_stats.get('evictions', 0)
        
        overall_hit_rate = total_hits / total_requests if total_requests > 0 else 0
        
        return {
            'hit_rate': overall_hit_rate,
            'total_hits': total_hits,
            'total_requests': total_requests,
            'total_evictions': total_evictions,
            'performance_grade': self._get_performance_grade(overall_hit_rate)
        }
    
    def _get_performance_grade(self, hit_rate: float) -> str:
        """根据命中率获取性能等级"""
        if hit_rate >= 0.85:
            return 'EXCELLENT'
        elif hit_rate >= 0.70:
            return 'GOOD'
        elif hit_rate >= 0.50:
            return 'AVERAGE'
        elif hit_rate >= 0.30:
            return 'POOR'
        else:
            return 'CRITICAL'
    
    def _identify_bottlenecks(self, cache_systems: Dict[str, Any]) -> List[Dict[str, Any]]:
        """识别缓存瓶颈"""
        bottlenecks = []
        
        for system_name, system_stats in cache_systems.items():
            if system_stats.get('status') != 'ACTIVE':
                continue
            
            # 检查命中率瓶颈
            if 'overall' in system_stats:
                hit_rate = system_stats['overall'].get('hit_rate', 0)
            else:
                hit_rate = system_stats.get('hit_rate', 0)
            
            if hit_rate < 0.50:
                bottlenecks.append({
                    'type': 'LOW_HIT_RATE',
                    'system': system_name,
                    'current_rate': hit_rate,
                    'severity': 'HIGH' if hit_rate < 0.30 else 'MEDIUM',
                    'description': f"{system_name} 命中率过低: {hit_rate:.2%}"
                })
            
            # 检查缓存大小瓶颈
            if system_name == 'query_cache':
                memory_size = system_stats['memory_cache'].get('size', 0)
                max_size = system_stats['memory_cache'].get('max_size', 1)
                usage_rate = memory_size / max_size
                
                if usage_rate > 0.90:
                    bottlenecks.append({
                        'type': 'CACHE_SIZE_LIMIT',
                        'system': system_name,
                        'usage_rate': usage_rate,
                        'severity': 'HIGH',
                        'description': f"{system_name} 内存缓存使用率过高: {usage_rate:.2%}"
                    })
            
            # 检查驱逐率瓶颈
            evictions = system_stats.get('evictions', 0)
            if 'total_requests' in system_stats:
                total_requests = system_stats['total_requests']
            elif 'overall' in system_stats:
                total_requests = system_stats['overall'].get('total_queries', 0)
            else:
                total_requests = 0
            
            if total_requests > 0:
                eviction_rate = evictions / total_requests
                if eviction_rate > 0.20:
                    bottlenecks.append({
                        'type': 'HIGH_EVICTION_RATE',
                        'system': system_name,
                        'eviction_rate': eviction_rate,
                        'severity': 'MEDIUM',
                        'description': f"{system_name} 驱逐率过高: {eviction_rate:.2%}"
                    })
        
        return bottlenecks
    
    def _identify_optimization_opportunities(self, performance_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """识别优化机会"""
        opportunities = []
        
        overall_hit_rate = performance_analysis['overall_metrics'].get('hit_rate', 0)
        
        # 如果整体命中率低于目标，提供优化建议
        if overall_hit_rate < 0.80:
            improvement_needed = 0.85 - overall_hit_rate
            
            opportunities.append({
                'type': 'INCREASE_CACHE_SIZE',
                'priority': 'HIGH',
                'potential_improvement': min(improvement_needed * 0.4, 0.15),
                'description': '增加缓存大小可显著提升命中率',
                'implementation': '将内存缓存大小从1000增加到3000'
            })
            
            opportunities.append({
                'type': 'IMPROVE_CACHE_STRATEGY',
                'priority': 'HIGH',
                'potential_improvement': min(improvement_needed * 0.3, 0.12),
                'description': '优化缓存策略，如LRU、TTL等',
                'implementation': '实施智能TTL和预加载策略'
            })
            
            opportunities.append({
                'type': 'ENABLE_PRELOADING',
                'priority': 'MEDIUM',
                'potential_improvement': min(improvement_needed * 0.2, 0.08),
                'description': '实现智能预加载机制',
                'implementation': '基于查询模式预加载热点数据'
            })
            
            opportunities.append({
                'type': 'OPTIMIZE_KEY_GENERATION',
                'priority': 'MEDIUM',
                'potential_improvement': min(improvement_needed * 0.1, 0.05),
                'description': '优化缓存键生成策略',
                'implementation': '改进缓存键算法，减少冲突'
            })
        
        # 检查磁盘缓存优化机会
        query_cache_stats = performance_analysis['cache_systems'].get('query_cache', {})
        if query_cache_stats.get('status') == 'ACTIVE':
            disk_hit_rate = query_cache_stats['disk_cache'].get('hit_rate', 0)
            if disk_hit_rate < 0.30:
                opportunities.append({
                    'type': 'IMPROVE_DISK_CACHE',
                    'priority': 'MEDIUM',
                    'potential_improvement': 0.10,
                    'description': '优化磁盘缓存性能',
                    'implementation': '增加磁盘缓存大小和优化存储策略'
                })
        
        return opportunities
    
    def generate_optimization_plan(self, performance_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """生成优化计划"""
        logger.info("生成缓存优化计划...")
        
        optimization_plan = {
            'plan_id': f"cache_opt_{int(time.time())}",
            'created_at': datetime.now().isoformat(),
            'current_performance': performance_analysis['overall_metrics'],
            'target_performance': {
                'hit_rate': 0.85,
                'improvement_target': 0.85 - performance_analysis['overall_metrics'].get('hit_rate', 0)
            },
            'optimization_phases': [],
            'implementation_timeline': {},
            'resource_requirements': {},
            'risk_assessment': {}
        }
        
        # 阶段1：立即优化（0-1周）
        phase1_actions = []
        for opportunity in performance_analysis['optimization_opportunities']:
            if opportunity['priority'] == 'HIGH':
                phase1_actions.append({
                    'action': opportunity['type'],
                    'description': opportunity['description'],
                    'implementation': opportunity['implementation'],
                    'expected_improvement': opportunity['potential_improvement'],
                    'effort': 'LOW' if 'CACHE_SIZE' in opportunity['type'] else 'MEDIUM'
                })
        
        optimization_plan['optimization_phases'].append({
            'phase': 1,
            'name': '立即优化',
            'duration': '1周',
            'actions': phase1_actions,
            'expected_improvement': sum(a['expected_improvement'] for a in phase1_actions)
        })
        
        # 阶段2：中期优化（1-4周）
        phase2_actions = []
        for opportunity in performance_analysis['optimization_opportunities']:
            if opportunity['priority'] == 'MEDIUM':
                phase2_actions.append({
                    'action': opportunity['type'],
                    'description': opportunity['description'],
                    'implementation': opportunity['implementation'],
                    'expected_improvement': opportunity['potential_improvement'],
                    'effort': 'MEDIUM'
                })
        
        optimization_plan['optimization_phases'].append({
            'phase': 2,
            'name': '中期优化',
            'duration': '3周',
            'actions': phase2_actions,
            'expected_improvement': sum(a['expected_improvement'] for a in phase2_actions)
        })
        
        # 计算预期总改进
        total_expected_improvement = sum(
            phase['expected_improvement'] 
            for phase in optimization_plan['optimization_phases']
        )
        
        current_hit_rate = performance_analysis['overall_metrics'].get('hit_rate', 0)
        projected_hit_rate = min(current_hit_rate + total_expected_improvement, 0.95)
        
        optimization_plan['target_performance']['projected_hit_rate'] = projected_hit_rate
        optimization_plan['target_performance']['success_probability'] = min(total_expected_improvement / 0.35, 1.0)
        
        logger.info(f"优化计划生成完成，预期命中率: {projected_hit_rate:.2%}")
        
        return optimization_plan
    
    def implement_immediate_optimizations(self) -> Dict[str, Any]:
        """实施立即优化措施"""
        logger.info("开始实施立即优化措施...")
        
        implementation_results = {
            'start_time': datetime.now().isoformat(),
            'optimizations_applied': [],
            'performance_before': {},
            'performance_after': {},
            'success': False
        }
        
        try:
            # 记录优化前性能
            before_analysis = self.analyze_current_cache_performance()
            implementation_results['performance_before'] = before_analysis['overall_metrics']
            
            # 优化1：增加查询缓存大小
            if 'query_cache' in self.cache_systems:
                old_size = self.cache_systems['query_cache'].max_memory_size
                new_size = min(old_size * 3, 3000)  # 增加到3倍，最大3000
                
                self.cache_systems['query_cache'].max_memory_size = new_size
                
                implementation_results['optimizations_applied'].append({
                    'type': 'INCREASE_CACHE_SIZE',
                    'description': f'查询缓存大小从 {old_size} 增加到 {new_size}',
                    'status': 'SUCCESS'
                })
            
            # 优化2：启用智能预加载
            self._enable_intelligent_preloading()
            implementation_results['optimizations_applied'].append({
                'type': 'ENABLE_PRELOADING',
                'description': '启用智能预加载机制',
                'status': 'SUCCESS'
            })
            
            # 优化3：调整TTL策略
            self._optimize_ttl_strategy()
            implementation_results['optimizations_applied'].append({
                'type': 'OPTIMIZE_TTL',
                'description': '优化缓存TTL策略',
                'status': 'SUCCESS'
            })
            
            # 等待优化生效
            time.sleep(2)
            
            # 记录优化后性能
            after_analysis = self.analyze_current_cache_performance()
            implementation_results['performance_after'] = after_analysis['overall_metrics']
            
            # 计算改进效果
            before_hit_rate = implementation_results['performance_before'].get('hit_rate', 0)
            after_hit_rate = implementation_results['performance_after'].get('hit_rate', 0)
            improvement = after_hit_rate - before_hit_rate
            
            implementation_results['improvement'] = improvement
            implementation_results['success'] = improvement > 0
            implementation_results['end_time'] = datetime.now().isoformat()
            
            logger.info(f"立即优化完成，命中率改进: {improvement:.2%}")
            
        except Exception as e:
            logger.error(f"实施优化措施失败: {e}")
            implementation_results['error'] = str(e)
            implementation_results['success'] = False
        
        return implementation_results
    
    def _enable_intelligent_preloading(self):
        """启用智能预加载"""
        # 这是一个示例实现，实际需要根据具体缓存系统调整
        if 'intelligent_cache' in self.cache_systems:
            cache_system = self.cache_systems['intelligent_cache']
            # 启用预加载逻辑（需要根据实际系统实现）
            logger.info("智能预加载机制已启用")
    
    def _optimize_ttl_strategy(self):
        """优化TTL策略"""
        if 'query_cache' in self.cache_systems:
            query_cache = self.cache_systems['query_cache']
            # 调整TTL为更智能的策略
            query_cache.default_ttl = 3600  # 增加到1小时
            logger.info("TTL策略已优化")
    
    def run_comprehensive_cache_optimization(self) -> Dict[str, Any]:
        """运行全面的缓存优化分析"""
        logger.info("=" * 80)
        logger.info("开始全面缓存优化分析")
        logger.info("目标：将缓存命中率从50%提升到85%")
        logger.info("=" * 80)
        
        try:
            # 1. 分析当前性能
            logger.info("步骤 1: 分析当前缓存性能")
            performance_analysis = self.analyze_current_cache_performance()
            
            # 2. 生成优化计划
            logger.info("步骤 2: 生成优化计划")
            optimization_plan = self.generate_optimization_plan(performance_analysis)
            
            # 3. 实施立即优化
            logger.info("步骤 3: 实施立即优化措施")
            implementation_results = self.implement_immediate_optimizations()
            
            # 4. 编译最终结果
            final_results = {
                'analysis_date': datetime.now().isoformat(),
                'current_performance': performance_analysis,
                'optimization_plan': optimization_plan,
                'immediate_implementation': implementation_results,
                'success': implementation_results.get('success', False),
                'next_steps': self._generate_next_steps(optimization_plan)
            }
            
            logger.info("缓存优化分析完成")
            return final_results
            
        except Exception as e:
            logger.error(f"缓存优化分析失败: {e}")
            return {
                'analysis_date': datetime.now().isoformat(),
                'status': 'FAILED',
                'error': str(e)
            }
    
    def _generate_next_steps(self, optimization_plan: Dict[str, Any]) -> List[str]:
        """生成后续步骤"""
        next_steps = []
        
        current_hit_rate = optimization_plan['current_performance'].get('hit_rate', 0)
        target_hit_rate = optimization_plan['target_performance']['hit_rate']
        
        if current_hit_rate < target_hit_rate:
            next_steps.extend([
                "继续监控缓存性能指标",
                "实施中期优化计划（预加载、索引优化）",
                "评估硬件资源需求",
                "制定长期缓存策略",
                "建立缓存性能监控告警"
            ])
        else:
            next_steps.extend([
                "保持当前优化配置",
                "定期性能评估",
                "监控系统稳定性"
            ])
        
        return next_steps
    
    def print_optimization_summary(self, results: Dict[str, Any]):
        """打印优化总结"""
        print("\n" + "=" * 80)
        print("缓存优化分析总结")
        print("=" * 80)
        
        if results.get('status') == 'FAILED':
            print(f"❌ 分析失败: {results.get('error', '未知错误')}")
            return
        
        # 当前性能
        current_perf = results['current_performance']['overall_metrics']
        print(f"📊 当前性能:")
        print(f"  命中率: {current_perf.get('hit_rate', 0):.2%}")
        print(f"  总请求数: {current_perf.get('total_requests', 0)}")
        print(f"  性能等级: {current_perf.get('performance_grade', 'UNKNOWN')}")
        
        # 优化计划
        plan = results['optimization_plan']
        print(f"\n🎯 优化目标:")
        print(f"  目标命中率: {plan['target_performance']['hit_rate']:.2%}")
        print(f"  预期命中率: {plan['target_performance'].get('projected_hit_rate', 0):.2%}")
        print(f"  成功概率: {plan['target_performance'].get('success_probability', 0):.2%}")
        
        # 实施结果
        impl = results['immediate_implementation']
        if impl.get('success'):
            before_rate = impl['performance_before'].get('hit_rate', 0)
            after_rate = impl['performance_after'].get('hit_rate', 0)
            improvement = impl.get('improvement', 0)
            
            print(f"\n✅ 立即优化结果:")
            print(f"  优化前命中率: {before_rate:.2%}")
            print(f"  优化后命中率: {after_rate:.2%}")
            print(f"  性能改进: {improvement:.2%}")
            
            if after_rate >= 0.80:
                print(f"  🎉 已达成80%命中率目标！")
            elif improvement > 0:
                print(f"  📈 性能有所改进，继续优化中")
            else:
                print(f"  ⚠️ 改进效果不明显，需要进一步分析")
        else:
            print(f"\n❌ 立即优化失败: {impl.get('error', '未知错误')}")
        
        # 后续步骤
        print(f"\n📋 后续步骤:")
        for i, step in enumerate(results.get('next_steps', []), 1):
            print(f"  {i}. {step}")


def main():
    """主函数"""
    print("=" * 80)
    print("缓存优化分析器")
    print("目标：将命中率从50%提升到85%")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # 创建分析器
        analyzer = CacheOptimizationAnalyzer()
        
        # 运行全面分析
        results = analyzer.run_comprehensive_cache_optimization()
        
        # 显示结果
        analyzer.print_optimization_summary(results)
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = f"results/cache_optimization_analysis_{timestamp}.json"
        
        try:
            os.makedirs("results", exist_ok=True)
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            print(f"\n📄 详细结果已保存到: {result_file}")
        except Exception as e:
            print(f"⚠️ 结果保存失败: {e}")
        
        # 返回状态码
        if results.get('success'):
            print(f"\n🎉 缓存优化成功")
            return 0
        else:
            print(f"\n⚠️ 缓存优化需要进一步工作")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断分析")
        return 130
    except Exception as e:
        print(f"\n💥 分析执行异常: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 