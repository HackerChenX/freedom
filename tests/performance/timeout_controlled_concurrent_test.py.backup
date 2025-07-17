#!/usr/bin/env python3
"""
超时控制的并发性能测试

特性：
1. 自动超时检测和停止
2. 实时性能监控
3. 分阶段测试控制
4. 智能阈值判断
"""

import os
import sys
import time
import signal
import threading
import concurrent.futures
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from config import get_config
from db.enhanced_connection_pool import initialize_connection_pool, get_connection_pool
from db.unified_data_manager import get_unified_data_manager
from monitoring.production_performance_monitor import ProductionPerformanceMonitor
from utils.logger import get_logger

logger = get_logger(__name__)


class TimeoutControlledTest:
    """超时控制的并发测试"""
    
    # 性能阈值配置
    PERFORMANCE_THRESHOLDS = {
        'max_single_query_time': 5.0,      # 单个查询最大时间（秒）
        'max_concurrent_test_time': 30.0,   # 并发测试最大时间（秒）
        'max_total_test_time': 300.0,       # 总测试时间（秒）
        'min_success_rate': 0.80,           # 最低成功率
        'max_response_time': 2.0,           # 最大响应时间（秒）
        'max_memory_usage': 80.0,           # 最大内存使用率（%）
        'max_cpu_usage': 90.0               # 最大CPU使用率（%）
    }
    
    def __init__(self):
        """初始化测试器"""
        self.test_start_time = None
        self.is_stopping = False
        self._stop_event = threading.Event()
        self.performance_monitor = ProductionPerformanceMonitor()
        
        # 初始化数据管理器
        try:
            self.data_manager = get_unified_data_manager()
            logger.info("数据管理器初始化成功")
        except Exception as e:
            logger.error(f"数据管理器初始化失败: {e}")
            self.data_manager = None
        
        # 测试结果
        self.test_results = {
            'start_time': None,
            'end_time': None,
            'tests_executed': [],
            'timeout_events': [],
            'performance_metrics': {},
            'overall_status': 'PENDING'
        }
        
        # 注册信号处理器（用于手动停止）
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """信号处理器 - 用于优雅停止"""
        logger.warning(f"接收到停止信号 {signum}，开始优雅停止测试...")
        self.stop_test("手动停止")
    
    def stop_test(self, reason: str = "超时"):
        """停止测试"""
        if not self.is_stopping:
            self.is_stopping = True
            self._stop_event.set()
            logger.warning(f"测试被停止: {reason}")
            
            # 记录停止事件
            self.test_results['timeout_events'].append({
                'timestamp': datetime.now().isoformat(),
                'reason': reason,
                'elapsed_time': time.time() - self.test_start_time if self.test_start_time else 0
            })
    
    def _check_timeout_conditions(self) -> Optional[str]:
        """检查超时条件"""
        if not self.test_start_time:
            return None
            
        elapsed_time = time.time() - self.test_start_time
        
        # 检查总测试时间
        if elapsed_time > self.PERFORMANCE_THRESHOLDS['max_total_test_time']:
            return f"总测试时间超时 ({elapsed_time:.1f}s > {self.PERFORMANCE_THRESHOLDS['max_total_test_time']}s)"
        
        # 检查系统资源
        try:
            import psutil
            memory_percent = psutil.virtual_memory().percent
            cpu_percent = psutil.cpu_percent(interval=1)
            
            if memory_percent > self.PERFORMANCE_THRESHOLDS['max_memory_usage']:
                return f"内存使用率过高 ({memory_percent:.1f}% > {self.PERFORMANCE_THRESHOLDS['max_memory_usage']}%)"
            
            if cpu_percent > self.PERFORMANCE_THRESHOLDS['max_cpu_usage']:
                return f"CPU使用率过高 ({cpu_percent:.1f}% > {self.PERFORMANCE_THRESHOLDS['max_cpu_usage']}%)"
                
        except Exception as e:
            logger.warning(f"无法检查系统资源: {e}")
        
        return None
    
    def _monitor_test_progress(self, test_name: str, expected_duration: float):
        """监控测试进度"""
        def monitor():
            start_time = time.time()
            while not self._stop_event.wait(1.0):  # 每秒检查一次
                elapsed = time.time() - start_time
                
                # 检查测试特定超时
                if elapsed > expected_duration * 1.5:  # 允许50%的缓冲时间
                    self.stop_test(f"{test_name} 执行时间超时 ({elapsed:.1f}s > {expected_duration * 1.5:.1f}s)")
                    break
                
                # 检查全局超时条件
                timeout_reason = self._check_timeout_conditions()
                if timeout_reason:
                    self.stop_test(timeout_reason)
                    break
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        return monitor_thread
    
    def test_single_connection_with_timeout(self, timeout: float = 10.0) -> Dict[str, Any]:
        """测试单连接性能（带超时）"""
        if self.is_stopping:
            return {'status': 'SKIPPED', 'reason': '测试已停止'}
        
        logger.info(f"开始单连接测试（超时: {timeout}s）")
        test_start = time.time()
        
        # 启动监控
        monitor_thread = self._monitor_test_progress("单连接测试", timeout)
        
        try:
            if not self.data_manager:
                return {
                    'status': 'FAILED',
                    'reason': '数据管理器未初始化',
                    'duration': 0
                }
            
            # 执行测试查询
            start_time = time.time()
            result = self.data_manager.get_stock_info(
                code='000001',
                start_date='2024-01-01',
                end_date='2024-01-10',
                level='daily'
            )
            duration = time.time() - start_time
            
            # 检查是否被停止
            if self.is_stopping:
                return {'status': 'STOPPED', 'reason': '测试被中断'}
            
            # 验证结果
            success = result and hasattr(result, 'data') and not result.data.empty
            
            test_result = {
                'status': 'SUCCESS' if success else 'FAILED',
                'duration': duration,
                'data_rows': len(result.data) if success else 0,
                'timeout_occurred': duration > timeout,
                'performance_ok': duration <= self.PERFORMANCE_THRESHOLDS['max_single_query_time']
            }
            
            if test_result['timeout_occurred']:
                logger.warning(f"单连接测试超时: {duration:.3f}s > {timeout}s")
            
            return test_result
            
        except Exception as e:
            logger.error(f"单连接测试失败: {e}")
            return {
                'status': 'ERROR',
                'error': str(e),
                'duration': time.time() - test_start
            }
        finally:
            self._stop_event.set()  # 通知监控线程停止
    
    def test_concurrent_performance_with_timeout(self, 
                                               concurrent_count: int = 10, 
                                               queries_per_thread: int = 3,
                                               timeout: float = 30.0) -> Dict[str, Any]:
        """测试并发性能（带超时控制）"""
        if self.is_stopping:
            return {'status': 'SKIPPED', 'reason': '测试已停止'}
        
        logger.info(f"开始并发测试（并发数: {concurrent_count}, 超时: {timeout}s）")
        test_start = time.time()
        
        # 启动监控
        monitor_thread = self._monitor_test_progress("并发测试", timeout)
        
        try:
            if not self.data_manager:
                return {
                    'status': 'FAILED',
                    'reason': '数据管理器未初始化'
                }
            
            # 准备测试数据
            test_stocks = ['000001', '000002', '600000', '600036', '300001']
            results = []
            errors = []
            
            def execute_concurrent_query(thread_id: int) -> Dict[str, Any]:
                """并发查询执行函数"""
                thread_results = {
                    'thread_id': thread_id,
                    'queries_completed': 0,
                    'total_time': 0,
                    'errors': []
                }
                
                thread_start = time.time()
                
                for i in range(queries_per_thread):
                    if self.is_stopping or self._stop_event.is_set():
                        break
                    
                    try:
                        query_start = time.time()
                        stock_code = test_stocks[i % len(test_stocks)]
                        
                        result = self.data_manager.get_stock_info(
                            code=stock_code,
                            start_date='2024-01-01',
                            end_date='2024-01-05',
                            level='daily'
                        )
                        
                        query_time = time.time() - query_start
                        thread_results['queries_completed'] += 1
                        
                        # 检查单个查询超时
                        if query_time > self.PERFORMANCE_THRESHOLDS['max_single_query_time']:
                            logger.warning(f"线程 {thread_id} 查询 {i+1} 超时: {query_time:.3f}s")
                        
                    except Exception as e:
                        thread_results['errors'].append(str(e))
                        logger.error(f"线程 {thread_id} 查询 {i+1} 失败: {e}")
                
                thread_results['total_time'] = time.time() - thread_start
                return thread_results
            
            # 执行并发测试
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_count) as executor:
                # 提交任务
                futures = [executor.submit(execute_concurrent_query, i) for i in range(concurrent_count)]
                
                # 等待完成（带超时）
                for future in concurrent.futures.as_completed(futures, timeout=timeout):
                    if self.is_stopping:
                        # 取消剩余任务
                        for f in futures:
                            f.cancel()
                        break
                    
                    try:
                        result = future.result(timeout=5.0)  # 每个任务5秒超时
                        results.append(result)
                    except concurrent.futures.TimeoutError:
                        errors.append("任务执行超时")
                        logger.warning("并发任务执行超时")
                    except Exception as e:
                        errors.append(str(e))
                        logger.error(f"并发任务执行失败: {e}")
            
            total_time = time.time() - test_start
            
            # 分析结果
            if results:
                total_queries = sum(r['queries_completed'] for r in results)
                successful_threads = len([r for r in results if r['queries_completed'] > 0])
                success_rate = successful_threads / concurrent_count if concurrent_count > 0 else 0
                avg_response_time = total_time / total_queries if total_queries > 0 else float('inf')
            else:
                total_queries = 0
                successful_threads = 0
                success_rate = 0
                avg_response_time = float('inf')
            
            test_result = {
                'status': 'SUCCESS' if success_rate >= self.PERFORMANCE_THRESHOLDS['min_success_rate'] else 'FAILED',
                'concurrent_count': concurrent_count,
                'success_rate': success_rate,
                'total_queries': total_queries,
                'total_time': total_time,
                'avg_response_time': avg_response_time,
                'timeout_occurred': total_time > timeout,
                'performance_ok': avg_response_time <= self.PERFORMANCE_THRESHOLDS['max_response_time'],
                'errors': errors,
                'thread_results': results
            }
            
            if test_result['timeout_occurred']:
                logger.warning(f"并发测试超时: {total_time:.3f}s > {timeout}s")
            
            return test_result
            
        except concurrent.futures.TimeoutError:
            logger.error("并发测试整体超时")
            return {
                'status': 'TIMEOUT',
                'duration': time.time() - test_start,
                'timeout_threshold': timeout
            }
        except Exception as e:
            logger.error(f"并发测试失败: {e}")
            return {
                'status': 'ERROR',
                'error': str(e),
                'duration': time.time() - test_start
            }
        finally:
            self._stop_event.set()  # 通知监控线程停止
    
    def run_comprehensive_timeout_controlled_test(self) -> Dict[str, Any]:
        """运行全面的超时控制测试"""
        logger.info("=" * 80)
        logger.info("开始超时控制的并发性能测试")
        logger.info("=" * 80)
        
        self.test_start_time = time.time()
        self.test_results['start_time'] = datetime.now().isoformat()
        
        try:
            # 阶段1：单连接测试
            logger.info("阶段 1: 单连接性能测试")
            single_test = self.test_single_connection_with_timeout(timeout=10.0)
            self.test_results['tests_executed'].append({
                'name': '单连接测试',
                'result': single_test
            })
            
            if single_test['status'] in ['FAILED', 'ERROR', 'STOPPED']:
                logger.warning("单连接测试失败，跳过后续测试")
                self.test_results['overall_status'] = 'EARLY_TERMINATION'
                return self.test_results
            
            # 阶段2：小规模并发测试
            if not self.is_stopping:
                logger.info("阶段 2: 小规模并发测试（5并发）")
                small_concurrent_test = self.test_concurrent_performance_with_timeout(
                    concurrent_count=5, queries_per_thread=2, timeout=20.0
                )
                self.test_results['tests_executed'].append({
                    'name': '小规模并发测试',
                    'result': small_concurrent_test
                })
                
                # 如果小规模测试失败，不进行大规模测试
                if small_concurrent_test['status'] in ['FAILED', 'ERROR', 'TIMEOUT', 'STOPPED']:
                    logger.warning("小规模并发测试失败，跳过大规模测试")
                    self.test_results['overall_status'] = 'PARTIAL_COMPLETION'
                    return self.test_results
            
            # 阶段3：中等规模并发测试
            if not self.is_stopping:
                logger.info("阶段 3: 中等规模并发测试（10并发）")
                medium_concurrent_test = self.test_concurrent_performance_with_timeout(
                    concurrent_count=10, queries_per_thread=3, timeout=30.0
                )
                self.test_results['tests_executed'].append({
                    'name': '中等规模并发测试',
                    'result': medium_concurrent_test
                })
            
            # 阶段4：性能压力测试（可选）
            if not self.is_stopping and all(
                test['result']['status'] == 'SUCCESS' 
                for test in self.test_results['tests_executed']
            ):
                logger.info("阶段 4: 大规模并发压力测试（20并发）")
                large_concurrent_test = self.test_concurrent_performance_with_timeout(
                    concurrent_count=20, queries_per_thread=2, timeout=45.0
                )
                self.test_results['tests_executed'].append({
                    'name': '大规模并发测试',
                    'result': large_concurrent_test
                })
            
            # 设置最终状态
            if self.is_stopping:
                self.test_results['overall_status'] = 'STOPPED'
            else:
                successful_tests = sum(1 for test in self.test_results['tests_executed'] 
                                     if test['result']['status'] == 'SUCCESS')
                total_tests = len(self.test_results['tests_executed'])
                
                if successful_tests == total_tests:
                    self.test_results['overall_status'] = 'COMPLETE_SUCCESS'
                elif successful_tests >= total_tests * 0.5:
                    self.test_results['overall_status'] = 'PARTIAL_SUCCESS'
                else:
                    self.test_results['overall_status'] = 'MOSTLY_FAILED'
            
        except Exception as e:
            logger.error(f"测试过程发生异常: {e}")
            self.test_results['overall_status'] = 'EXCEPTION'
            self.test_results['exception'] = str(e)
        
        finally:
            self.test_results['end_time'] = datetime.now().isoformat()
            self.test_results['total_duration'] = time.time() - self.test_start_time
        
        return self.test_results
    
    def generate_test_summary(self) -> str:
        """生成测试总结报告"""
        if not self.test_results['tests_executed']:
            return "❌ 没有执行任何测试"
        
        summary = []
        summary.append("=" * 60)
        summary.append("超时控制并发测试总结报告")
        summary.append("=" * 60)
        summary.append(f"测试开始时间: {self.test_results['start_time']}")
        summary.append(f"测试结束时间: {self.test_results['end_time']}")
        summary.append(f"总耗时: {self.test_results.get('total_duration', 0):.2f}秒")
        summary.append(f"最终状态: {self.test_results['overall_status']}")
        summary.append("")
        
        # 超时事件
        if self.test_results['timeout_events']:
            summary.append("⚠️ 超时事件:")
            for event in self.test_results['timeout_events']:
                summary.append(f"  - {event['reason']} (耗时: {event['elapsed_time']:.1f}s)")
            summary.append("")
        
        # 测试结果详情
        summary.append("📊 测试结果详情:")
        for test in self.test_results['tests_executed']:
            name = test['name']
            result = test['result']
            status = result['status']
            
            status_icon = {
                'SUCCESS': '✅',
                'FAILED': '❌',
                'ERROR': '💥',
                'TIMEOUT': '⏰',
                'STOPPED': '🛑',
                'SKIPPED': '⏭️'
            }.get(status, '❓')
            
            summary.append(f"  {status_icon} {name}: {status}")
            
            if 'duration' in result:
                summary.append(f"    耗时: {result['duration']:.3f}秒")
            
            if 'success_rate' in result:
                summary.append(f"    成功率: {result['success_rate']:.2%}")
            
            if 'avg_response_time' in result:
                summary.append(f"    平均响应时间: {result['avg_response_time']:.3f}秒")
        
        summary.append("")
        
        # 性能评估
        successful_tests = sum(1 for test in self.test_results['tests_executed'] 
                             if test['result']['status'] == 'SUCCESS')
        total_tests = len(self.test_results['tests_executed'])
        success_rate = successful_tests / total_tests if total_tests > 0 else 0
        
        summary.append("📈 性能评估:")
        summary.append(f"  测试通过率: {success_rate:.2%} ({successful_tests}/{total_tests})")
        
        if success_rate >= 0.8:
            summary.append("  🎉 系统性能表现优秀")
        elif success_rate >= 0.6:
            summary.append("  ⚠️ 系统性能需要优化")
        else:
            summary.append("  ❌ 系统性能存在严重问题")
        
        return "\n".join(summary)


def main():
    """主函数"""
    print("=" * 80)
    print("超时控制的并发性能测试")
    print("支持自动超时检测和优雅停止")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # 创建测试实例
        test_framework = TimeoutControlledTest()
        
        # 显示性能阈值
        print("📋 性能阈值配置:")
        for key, value in test_framework.PERFORMANCE_THRESHOLDS.items():
            print(f"  {key}: {value}")
        print()
        
        # 运行测试
        print("🚀 开始执行测试...")
        results = test_framework.run_comprehensive_timeout_controlled_test()
        
        # 显示测试总结
        print()
        print(test_framework.generate_test_summary())
        
        # 返回状态码
        if results['overall_status'] in ['COMPLETE_SUCCESS', 'PARTIAL_SUCCESS']:
            return 0
        else:
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断测试")
        return 130
    except Exception as e:
        print(f"\n💥 测试执行异常: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 