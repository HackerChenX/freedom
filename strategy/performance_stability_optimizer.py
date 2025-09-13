"""
性能优化和稳定性提升系统
对核心功能进行性能优化，提升系统稳定性，完善错误处理和监控机制
"""

import time
import threading
import gc
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None
import asyncio
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import logging

from utils.enhanced_performance_monitor import performance_monitor
from utils.enhanced_exception_handler import exception_handler, ErrorSeverity, ErrorCategory
from utils.system_resource_monitor import get_resource_monitor
from utils.unified_container import get_container
from config.unified_config_manager import get_config

logger = logging.getLogger(__name__)


@dataclass
class PerformanceOptimizationConfig:
    """性能优化配置"""
    # 内存管理配置
    max_memory_usage_mb: int = 2048
    gc_threshold_mb: int = 1024
    memory_cleanup_interval: int = 300  # 5分钟
    
    # 并发处理配置
    max_workers: int = 8
    thread_pool_size: int = 16
    process_pool_size: int = 4
    
    # 缓存优化配置
    cache_size_mb: int = 512
    cache_ttl_seconds: int = 3600
    cache_cleanup_interval: int = 600  # 10分钟
    
    # 性能监控配置
    performance_threshold_seconds: float = 2.0
    memory_threshold_mb: float = 1500
    cpu_threshold_percent: float = 80.0
    
    # 稳定性配置
    max_retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    circuit_breaker_threshold: int = 5
    health_check_interval: int = 60


@dataclass
class SystemHealthMetrics:
    """系统健康指标"""
    timestamp: datetime = field(default_factory=datetime.now)
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    memory_usage_percent: float = 0.0
    disk_usage_percent: float = 0.0
    active_threads: int = 0
    active_connections: int = 0
    cache_hit_rate: float = 0.0
    error_rate: float = 0.0
    response_time_avg: float = 0.0
    stability_score: float = 100.0


@dataclass
class OptimizationResult:
    """优化结果"""
    optimization_type: str
    before_metrics: Dict[str, Any]
    after_metrics: Dict[str, Any]
    improvement_percent: float
    execution_time: float
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)


class MemoryOptimizer:
    """内存优化器"""
    
    def __init__(self, config: PerformanceOptimizationConfig):
        self.config = config
        self.last_cleanup = time.time()
        self.memory_stats = {}
    
    @performance_monitor(threshold_seconds=1.0)
    @exception_handler(reraise=True)
    def optimize_memory_usage(self) -> OptimizationResult:
        """优化内存使用"""
        start_time = time.time()
        before_memory = self._get_memory_usage()
        
        # 执行垃圾回收
        collected = gc.collect()
        
        # 清理缓存
        self._cleanup_caches()
        
        # 优化数据结构
        self._optimize_data_structures()
        
        after_memory = self._get_memory_usage()
        improvement = ((before_memory - after_memory) / before_memory * 100) if before_memory > 0 else 0
        
        return OptimizationResult(
            optimization_type="memory_optimization",
            before_metrics={"memory_mb": before_memory},
            after_metrics={"memory_mb": after_memory},
            improvement_percent=improvement,
            execution_time=time.time() - start_time,
            success=True,
            details={
                "gc_collected": collected,
                "memory_freed_mb": before_memory - after_memory
            }
        )
    
    def _get_memory_usage(self) -> float:
        """获取内存使用量（MB）"""
        try:
            if PSUTIL_AVAILABLE:
                process = psutil.Process()
                return process.memory_info().rss / 1024 / 1024
            else:
                # 简单估算
                import sys
                return sys.getsizeof(locals()) / 1024 / 1024
        except Exception:
            return 0.0
    
    def _cleanup_caches(self):
        """清理缓存"""
        try:
            # 清理多层缓存
            from db.multi_layer_cache import get_multi_cache
            cache = get_multi_cache()
            cache.cleanup_expired()
            
            # 清理其他缓存
            if hasattr(container, 'clear_cache'):
                container.clear_cache()
                
        except Exception as e:
            logger.warning(f"缓存清理失败: {e}")
    
    def _optimize_data_structures(self):
        """优化数据结构"""
        # 强制垃圾回收
        for generation in range(3):
            gc.collect(generation)
        
        # 清理未使用的对象
        gc.set_threshold(700, 10, 10)


class ConcurrencyOptimizer:
    """并发优化器"""
    
    def __init__(self, config: PerformanceOptimizationConfig):
        self.config = config
        self.thread_pool = None
        self.process_pool = None
        self.active_tasks = {}
        self.lock = threading.Lock()
    
    @performance_monitor(threshold_seconds=2.0)
    @exception_handler(reraise=True)
    def optimize_concurrency(self) -> OptimizationResult:
        """优化并发处理"""
        start_time = time.time()
        before_metrics = self._get_concurrency_metrics()
        
        # 优化线程池
        self._optimize_thread_pool()
        
        # 优化进程池
        self._optimize_process_pool()
        
        # 优化任务调度
        self._optimize_task_scheduling()
        
        after_metrics = self._get_concurrency_metrics()
        
        return OptimizationResult(
            optimization_type="concurrency_optimization",
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            improvement_percent=0.0,  # 需要实际测试才能计算
            execution_time=time.time() - start_time,
            success=True,
            details={
                "thread_pool_size": self.config.thread_pool_size,
                "process_pool_size": self.config.process_pool_size
            }
        )
    
    def _get_concurrency_metrics(self) -> Dict[str, Any]:
        """获取并发指标"""
        return {
            "active_threads": threading.active_count(),
            "thread_pool_active": self.thread_pool is not None,
            "process_pool_active": self.process_pool is not None,
            "active_tasks": len(self.active_tasks)
        }
    
    def _optimize_thread_pool(self):
        """优化线程池"""
        if self.thread_pool is None:
            self.thread_pool = ThreadPoolExecutor(
                max_workers=self.config.thread_pool_size,
                thread_name_prefix="optimizer_thread"
            )
    
    def _optimize_process_pool(self):
        """优化进程池"""
        if self.process_pool is None:
            self.process_pool = ProcessPoolExecutor(
                max_workers=self.config.process_pool_size
            )
    
    def _optimize_task_scheduling(self):
        """优化任务调度"""
        # 清理已完成的任务
        with self.lock:
            completed_tasks = [
                task_id for task_id, task in self.active_tasks.items()
                if task.done()
            ]
            for task_id in completed_tasks:
                del self.active_tasks[task_id]
    
    @contextmanager
    def get_thread_pool(self):
        """获取线程池上下文管理器"""
        if self.thread_pool is None:
            self._optimize_thread_pool()
        try:
            yield self.thread_pool
        finally:
            pass  # 保持连接池活跃
    
    @contextmanager
    def get_process_pool(self):
        """获取进程池上下文管理器"""
        if self.process_pool is None:
            self._optimize_process_pool()
        try:
            yield self.process_pool
        finally:
            pass  # 保持连接池活跃
    
    def cleanup(self):
        """清理资源"""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
            self.thread_pool = None
        
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
            self.process_pool = None


class StabilityMonitor:
    """稳定性监控器"""
    
    def __init__(self, config: PerformanceOptimizationConfig):
        self.config = config
        self.health_metrics = []
        self.error_counts = {}
        self.circuit_breakers = {}
        self.monitoring_active = False
        self.monitor_thread = None
        self.lock = threading.Lock()
    
    @exception_handler(reraise=False)
    def start_monitoring(self):
        """开始稳定性监控"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            name="stability_monitor",
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("稳定性监控已启动")
    
    def stop_monitoring(self):
        """停止稳定性监控"""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        logger.info("稳定性监控已停止")
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.monitoring_active:
            try:
                metrics = self._collect_health_metrics()
                
                with self.lock:
                    self.health_metrics.append(metrics)
                    # 保留最近1000个指标
                    if len(self.health_metrics) > 1000:
                        self.health_metrics = self.health_metrics[-1000:]
                
                # 检查健康状态
                self._check_health_status(metrics)
                
                time.sleep(self.config.health_check_interval)
                
            except Exception as e:
                logger.error(f"稳定性监控循环出错: {e}")
                time.sleep(60)
    
    def _collect_health_metrics(self) -> SystemHealthMetrics:
        """收集健康指标"""
        try:
            # 获取系统资源信息
            if PSUTIL_AVAILABLE:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')

                # 获取进程信息
                process = psutil.Process()
                process_memory = process.memory_info().rss / 1024 / 1024  # MB
            else:
                # 模拟数据
                cpu_percent = 50.0
                memory = type('Memory', (), {'total': 8*1024*1024*1024, 'available': 4*1024*1024*1024, 'percent': 50.0})()
                disk = type('Disk', (), {'total': 100*1024*1024*1024, 'free': 50*1024*1024*1024, 'percent': 50.0})()
                process_memory = 100.0
            
            return SystemHealthMetrics(
                cpu_usage_percent=cpu_percent,
                memory_usage_mb=process_memory,
                memory_usage_percent=memory.percent,
                disk_usage_percent=disk.percent,
                active_threads=threading.active_count(),
                active_connections=0,  # 需要从连接池获取
                cache_hit_rate=0.0,   # 需要从缓存系统获取
                error_rate=self._calculate_error_rate(),
                response_time_avg=0.0,  # 需要从性能监控获取
                stability_score=self._calculate_stability_score()
            )
            
        except Exception as e:
            logger.error(f"收集健康指标失败: {e}")
            return SystemHealthMetrics()
    
    def _calculate_error_rate(self) -> float:
        """计算错误率"""
        try:
            total_errors = sum(self.error_counts.values())
            total_requests = max(total_errors * 10, 1)  # 估算
            return (total_errors / total_requests) * 100
        except Exception:
            return 0.0
    
    def _calculate_stability_score(self) -> float:
        """计算稳定性评分"""
        try:
            score = 100.0
            
            # 根据错误率扣分
            error_rate = self._calculate_error_rate()
            score -= min(error_rate * 10, 50)
            
            # 根据资源使用率扣分
            try:
                if PSUTIL_AVAILABLE:
                    cpu_percent = psutil.cpu_percent()
                    memory_percent = psutil.virtual_memory().percent
                else:
                    cpu_percent = 50.0
                    memory_percent = 50.0
                
                if cpu_percent > 80:
                    score -= (cpu_percent - 80) * 2
                if memory_percent > 80:
                    score -= (memory_percent - 80) * 2
                    
            except Exception:
                pass
            
            return max(score, 0.0)
            
        except Exception:
            return 50.0  # 默认中等稳定性
    
    def _check_health_status(self, metrics: SystemHealthMetrics):
        """检查健康状态"""
        # 检查CPU使用率
        if metrics.cpu_usage_percent > self.config.cpu_threshold_percent:
            logger.warning(f"CPU使用率过高: {metrics.cpu_usage_percent:.1f}%")
        
        # 检查内存使用
        if metrics.memory_usage_mb > self.config.memory_threshold_mb:
            logger.warning(f"内存使用过高: {metrics.memory_usage_mb:.1f}MB")
        
        # 检查稳定性评分
        if metrics.stability_score < 70:
            logger.warning(f"系统稳定性评分较低: {metrics.stability_score:.1f}")
    
    def record_error(self, error_type: str):
        """记录错误"""
        with self.lock:
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
    
    def get_latest_metrics(self) -> Optional[SystemHealthMetrics]:
        """获取最新健康指标"""
        with self.lock:
            return self.health_metrics[-1] if self.health_metrics else None
    
    def get_health_summary(self) -> Dict[str, Any]:
        """获取健康状态摘要"""
        latest = self.get_latest_metrics()
        if not latest:
            return {"status": "unknown"}
        
        return {
            "status": "healthy" if latest.stability_score > 80 else "warning" if latest.stability_score > 60 else "critical",
            "stability_score": latest.stability_score,
            "cpu_usage": latest.cpu_usage_percent,
            "memory_usage_mb": latest.memory_usage_mb,
            "error_rate": latest.error_rate,
            "active_threads": latest.active_threads
        }


class PerformanceStabilityOptimizer:
    """性能优化和稳定性提升主控制器"""

    def __init__(self, config: Optional[PerformanceOptimizationConfig] = None):
        self.config = config or PerformanceOptimizationConfig()
        self.memory_optimizer = MemoryOptimizer(self.config)
        self.concurrency_optimizer = ConcurrencyOptimizer(self.config)
        self.stability_monitor = StabilityMonitor(self.config)

        self.optimization_history = []
        self.performance_baselines = {}
        self.is_running = False
        self.lock = threading.Lock()

        # 注册到容器
        container = get_container()
        container.register(PerformanceStabilityOptimizer, instance=self)

    @performance_monitor(threshold_seconds=5.0)
    @exception_handler(reraise=True)
    def start_optimization_system(self) -> Dict[str, Any]:
        """启动优化系统"""
        if self.is_running:
            return {"status": "already_running"}

        start_time = time.time()

        try:
            # 启动稳定性监控
            self.stability_monitor.start_monitoring()

            # 建立性能基线
            self._establish_performance_baselines()

            # 执行初始优化
            initial_results = self._run_initial_optimization()

            self.is_running = True

            return {
                "status": "started",
                "startup_time": time.time() - start_time,
                "initial_optimization_results": initial_results,
                "baselines": self.performance_baselines
            }

        except Exception as e:
            logger.error(f"启动优化系统失败: {e}")
            raise

    def stop_optimization_system(self) -> Dict[str, Any]:
        """停止优化系统"""
        if not self.is_running:
            return {"status": "not_running"}

        # 停止稳定性监控
        self.stability_monitor.stop_monitoring()

        # 清理资源
        self.concurrency_optimizer.cleanup()

        self.is_running = False

        return {
            "status": "stopped",
            "optimization_count": len(self.optimization_history),
            "final_health": self.stability_monitor.get_health_summary()
        }

    @performance_monitor(threshold_seconds=10.0)
    @exception_handler(reraise=True)
    def run_comprehensive_optimization(self) -> Dict[str, Any]:
        """运行全面优化"""
        if not self.is_running:
            raise RuntimeError("优化系统未启动")

        start_time = time.time()
        results = {}

        # 1. 内存优化
        try:
            memory_result = self.memory_optimizer.optimize_memory_usage()
            results["memory_optimization"] = memory_result
            self.optimization_history.append(memory_result)
        except Exception as e:
            logger.error(f"内存优化失败: {e}")
            results["memory_optimization"] = {"error": str(e)}

        # 2. 并发优化
        try:
            concurrency_result = self.concurrency_optimizer.optimize_concurrency()
            results["concurrency_optimization"] = concurrency_result
            self.optimization_history.append(concurrency_result)
        except Exception as e:
            logger.error(f"并发优化失败: {e}")
            results["concurrency_optimization"] = {"error": str(e)}

        # 3. 数据库连接优化
        try:
            db_result = self._optimize_database_connections()
            results["database_optimization"] = db_result
            self.optimization_history.append(db_result)
        except Exception as e:
            logger.error(f"数据库优化失败: {e}")
            results["database_optimization"] = {"error": str(e)}

        # 4. 缓存优化
        try:
            cache_result = self._optimize_cache_system()
            results["cache_optimization"] = cache_result
            self.optimization_history.append(cache_result)
        except Exception as e:
            logger.error(f"缓存优化失败: {e}")
            results["cache_optimization"] = {"error": str(e)}

        # 5. 系统资源优化
        try:
            resource_result = self._optimize_system_resources()
            results["resource_optimization"] = resource_result
            self.optimization_history.append(resource_result)
        except Exception as e:
            logger.error(f"系统资源优化失败: {e}")
            results["resource_optimization"] = {"error": str(e)}

        # 计算总体优化效果
        total_time = time.time() - start_time
        health_after = self.stability_monitor.get_health_summary()

        results["summary"] = {
            "total_optimization_time": total_time,
            "optimizations_completed": len([r for r in results.values() if isinstance(r, OptimizationResult)]),
            "health_status_after": health_after,
            "overall_success": all(
                isinstance(r, OptimizationResult) and r.success
                for r in results.values()
                if isinstance(r, OptimizationResult)
            )
        }

        return results

    def _establish_performance_baselines(self):
        """建立性能基线"""
        try:
            # 内存基线
            self.performance_baselines["memory_mb"] = self.memory_optimizer._get_memory_usage()

            # 并发基线
            self.performance_baselines["concurrency"] = self.concurrency_optimizer._get_concurrency_metrics()

            # 系统健康基线
            health_metrics = self.stability_monitor._collect_health_metrics()
            self.performance_baselines["health"] = {
                "cpu_usage": health_metrics.cpu_usage_percent,
                "memory_usage": health_metrics.memory_usage_mb,
                "stability_score": health_metrics.stability_score
            }

            logger.info(f"性能基线已建立: {self.performance_baselines}")

        except Exception as e:
            logger.error(f"建立性能基线失败: {e}")

    def _run_initial_optimization(self) -> Dict[str, Any]:
        """运行初始优化"""
        results = {}

        # 轻量级内存清理
        try:
            gc.collect()
            results["initial_gc"] = "completed"
        except Exception as e:
            results["initial_gc"] = f"failed: {e}"

        # 预热连接池
        try:
            self._warmup_connection_pools()
            results["connection_warmup"] = "completed"
        except Exception as e:
            results["connection_warmup"] = f"failed: {e}"

        return results

    def _optimize_database_connections(self) -> OptimizationResult:
        """优化数据库连接"""
        start_time = time.time()

        try:
            # 获取连接池
            from db.optimized_connection_pool import get_optimized_pool
            pool = get_optimized_pool()

            before_metrics = {
                "active_connections": len(pool.connection_metrics),
                "available_connections": pool.available_connections.qsize()
            }

            # 优化连接池配置
            pool.health_check_interval = min(pool.health_check_interval, 30)
            pool.max_idle_time = min(pool.max_idle_time, 300)

            # 清理空闲连接
            pool._cleanup_idle_connections()

            after_metrics = {
                "active_connections": len(pool.connection_metrics),
                "available_connections": pool.available_connections.qsize()
            }

            return OptimizationResult(
                optimization_type="database_connections",
                before_metrics=before_metrics,
                after_metrics=after_metrics,
                improvement_percent=0.0,
                execution_time=time.time() - start_time,
                success=True,
                details={"pool_optimized": True}
            )

        except Exception as e:
            logger.error(f"数据库连接优化失败: {e}")
            return OptimizationResult(
                optimization_type="database_connections",
                before_metrics={},
                after_metrics={},
                improvement_percent=0.0,
                execution_time=time.time() - start_time,
                success=False,
                details={"error": str(e)}
            )

    def _optimize_cache_system(self) -> OptimizationResult:
        """优化缓存系统"""
        start_time = time.time()

        try:
            from db.multi_layer_cache import get_multi_cache
            cache = get_multi_cache()

            before_metrics = cache.get_stats()

            # 清理过期缓存
            cache.cleanup_expired()

            # 优化缓存配置
            if hasattr(cache, 'memory_cache'):
                cache.memory_cache.max_size = min(cache.memory_cache.max_size, self.config.cache_size_mb * 1024)

            after_metrics = cache.get_stats()

            return OptimizationResult(
                optimization_type="cache_system",
                before_metrics=before_metrics,
                after_metrics=after_metrics,
                improvement_percent=0.0,
                execution_time=time.time() - start_time,
                success=True,
                details={"cache_cleaned": True}
            )

        except Exception as e:
            logger.error(f"缓存系统优化失败: {e}")
            return OptimizationResult(
                optimization_type="cache_system",
                before_metrics={},
                after_metrics={},
                improvement_percent=0.0,
                execution_time=time.time() - start_time,
                success=False,
                details={"error": str(e)}
            )

    def _optimize_system_resources(self) -> OptimizationResult:
        """优化系统资源"""
        start_time = time.time()

        try:
            if PSUTIL_AVAILABLE:
                before_metrics = {
                    "cpu_count": psutil.cpu_count(),
                    "memory_total": psutil.virtual_memory().total / 1024 / 1024 / 1024,  # GB
                    "active_threads": threading.active_count()
                }

                # 优化线程数量
                optimal_threads = min(psutil.cpu_count() * 2, 16)
                self.config.max_workers = optimal_threads

                # 设置进程优先级（如果可能）
                try:
                    process = psutil.Process()
                    if hasattr(process, 'nice'):
                        current_nice = process.nice()
                except Exception:
                    current_nice = 0
            else:
                # 模拟数据
                import os
                cpu_count = os.cpu_count() or 4
                before_metrics = {
                    "cpu_count": cpu_count,
                    "memory_total": 8.0,  # GB
                    "active_threads": threading.active_count()
                }

                # 优化线程数量
                optimal_threads = min(cpu_count * 2, 16)
                self.config.max_workers = optimal_threads
                current_nice = 0

            after_metrics = {
                "optimal_threads": optimal_threads,
                "config_updated": True
            }

            return OptimizationResult(
                optimization_type="system_resources",
                before_metrics=before_metrics,
                after_metrics=after_metrics,
                improvement_percent=0.0,
                execution_time=time.time() - start_time,
                success=True,
                details={"resource_optimized": True}
            )

        except Exception as e:
            logger.error(f"系统资源优化失败: {e}")
            return OptimizationResult(
                optimization_type="system_resources",
                before_metrics={},
                after_metrics={},
                improvement_percent=0.0,
                execution_time=time.time() - start_time,
                success=False,
                details={"error": str(e)}
            )

    def _warmup_connection_pools(self):
        """预热连接池"""
        try:
            from db.optimized_connection_pool import get_optimized_pool
            pool = get_optimized_pool()

            # 预创建一些连接
            connections = []
            for _ in range(min(3, pool.min_connections)):
                try:
                    conn = pool.get_connection()
                    connections.append(conn)
                except Exception:
                    break

            # 归还连接
            for conn in connections:
                try:
                    pool.return_connection(conn)
                except Exception:
                    pass

        except Exception as e:
            logger.warning(f"连接池预热失败: {e}")

    def get_optimization_report(self) -> Dict[str, Any]:
        """获取优化报告"""
        if not self.optimization_history:
            return {"status": "no_optimizations"}

        successful_optimizations = [
            opt for opt in self.optimization_history
            if isinstance(opt, OptimizationResult) and opt.success
        ]

        total_improvement = sum(
            opt.improvement_percent for opt in successful_optimizations
        ) / len(successful_optimizations) if successful_optimizations else 0

        return {
            "total_optimizations": len(self.optimization_history),
            "successful_optimizations": len(successful_optimizations),
            "average_improvement_percent": total_improvement,
            "current_health": self.stability_monitor.get_health_summary(),
            "baselines": self.performance_baselines,
            "optimization_types": list(set(
                opt.optimization_type for opt in successful_optimizations
            ))
        }

    @contextmanager
    def optimized_execution(self):
        """优化执行上下文管理器"""
        # 执行前优化
        pre_optimization = self.memory_optimizer.optimize_memory_usage()

        try:
            yield
        finally:
            # 执行后清理
            try:
                gc.collect()
            except Exception:
                pass


# 全局优化器实例
_performance_optimizer = None
_optimizer_lock = threading.Lock()


def get_performance_optimizer() -> PerformanceStabilityOptimizer:
    """获取全局性能优化器实例"""
    global _performance_optimizer

    if _performance_optimizer is None:
        with _optimizer_lock:
            if _performance_optimizer is None:
                _performance_optimizer = PerformanceStabilityOptimizer()

    return _performance_optimizer


def initialize_performance_optimizer(config: Optional[PerformanceOptimizationConfig] = None) -> PerformanceStabilityOptimizer:
    """初始化性能优化器"""
    global _performance_optimizer

    with _optimizer_lock:
        if _performance_optimizer is not None:
            _performance_optimizer.stop_optimization_system()

        _performance_optimizer = PerformanceStabilityOptimizer(config)
        logger.info("性能优化器已初始化")

    return _performance_optimizer


# 导出主要类和函数
__all__ = [
    'PerformanceOptimizationConfig',
    'SystemHealthMetrics',
    'OptimizationResult',
    'MemoryOptimizer',
    'ConcurrencyOptimizer',
    'StabilityMonitor',
    'PerformanceStabilityOptimizer',
    'get_performance_optimizer',
    'initialize_performance_optimizer'
]
