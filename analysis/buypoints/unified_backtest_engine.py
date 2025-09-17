from utils.container import container
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
统一高性能历史回测引擎接口

整合所有优化组件，提供统一的高性能回测服务：
- 向量化计算优化
- 多进程并行处理
- 内存管理优化
- 智能缓存系统

满足PMO性能要求：
- 回测速度>10,000条/秒
- 内存使用<4GB
- 计算精度>99.99%
"""

import os
import time
import json
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

from utils.logger import get_logger
from utils.decorators import performance_monitor, exception_handler

# 导入所有优化组件
from .high_performance_backtest_engine import (
    HighPerformanceBacktestEngine, PerformanceTarget, OptimizationConfig
)
from .parallel_processing_optimizer import (
    ParallelProcessingOptimizer, ProcessorConfig
)
from .memory_optimizer import (
    MemoryOptimizationService, MemoryTarget, ChunkConfig
)
from .intelligent_cache_system import (
    IntelligentCacheSystem, CacheConfig
)

logger = get_logger(__name__)

class EngineMode(Enum):
    """引擎模式"""
    SINGLE_THREAD = "single_thread"  # 单线程向量化
    MULTI_THREAD = "multi_thread"    # 多线程并行
    MULTI_PROCESS = "multi_process"  # 多进程并行
    ADAPTIVE = "adaptive"            # 自适应模式

class OptimizationLevel(Enum):
    """优化级别"""
    BASIC = 1      # 基础优化
    STANDARD = 2   # 标准优化
    AGGRESSIVE = 3 # 激进优化
    MAXIMUM = 4    # 最大优化

@dataclass
class UnifiedConfig:
    """统一配置"""
    # 引擎配置
    engine_mode: EngineMode = EngineMode.ADAPTIVE
    optimization_level: OptimizationLevel = OptimizationLevel.STANDARD

    # 性能目标
    target_speed: int = 10000       # 目标速度(条/秒)
    max_memory_gb: float = 4.0      # 最大内存(GB)
    target_accuracy: float = 99.99  # 目标精度(%)

    # 处理配置
    max_workers: int = 8            # 最大工作进程数
    chunk_size: int = 1000          # 数据块大小
    batch_size: int = 100           # 批处理大小

    # 缓存配置
    enable_caching: bool = True     # 启用缓存
    cache_size_mb: int = 1024       # 缓存大小(MB)

    # 监控配置
    enable_monitoring: bool = True  # 启用监控
    enable_profiling: bool = False  # 启用性能分析

@dataclass
class BacktestRequest:
    """回测请求"""
    stock_codes: List[str]
    start_date: str
    end_date: str
    indicators: List[str] = None
    strategy_params: Dict[str, Any] = None
    priority: int = 1  # 优先级 1-5

@dataclass
class BacktestResponse:
    """回测响应"""
    request_id: str
    success: bool
    results: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]
    execution_time: float
    error_message: Optional[str] = None

class UnifiedHighPerformanceBacktestEngine:
    """
    统一高性能历史回测引擎

    整合所有优化组件，提供统一的高性能回测服务接口
    """

    def __init__(self, config: Optional[UnifiedConfig] = None):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        """初始化统一引擎"""
        self.config = config or UnifiedConfig()
        self.logger = logger

        # 核心组件
        self.vectorized_engine = None
        self.parallel_optimizer = None
        self.memory_optimizer = None
        self.cache_system = None

        # 性能统计
        self.performance_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_stocks_processed': 0,
            'total_execution_time': 0.0,
            'average_speed': 0.0,
            'cache_hit_rate': 0.0
        }

        # 初始化组件
        self._initialize_components()

        self.logger.info(f"统一高性能回测引擎初始化完成")
        self.logger.info(f"引擎模式: {self.config.engine_mode.value}")
        self.logger.info(f"优化级别: {self.config.optimization_level.value}")

    def _initialize_components(self):
        """初始化核心组件"""
        try:
            # 初始化内存优化器
            memory_target = MemoryTarget(
                max_memory_gb=self.config.max_memory_gb,
                warning_threshold=self.config.max_memory_gb * 0.8,
                critical_threshold=self.config.max_memory_gb * 0.95
            )
            self.memory_optimizer = MemoryOptimizationService(memory_target)

            # 初始化缓存系统
            if self.config.enable_caching:
                cache_config = CacheConfig(
                    memory_max_size_mb=self.config.cache_size_mb,
                    disk_cache_enabled=True,
                    compression_enabled=True
                )
                self.cache_system = IntelligentCacheSystem(cache_config)

            # 根据优化级别初始化引擎
            self._initialize_engine_by_level()

            # 根据引擎模式初始化并行处理器
            if self.config.engine_mode in [EngineMode.MULTI_PROCESS, EngineMode.ADAPTIVE]:
                self._initialize_parallel_processor()

            self.logger.info("所有核心组件初始化完成")

        except Exception as e:
            self.logger.error(f"组件初始化失败: {e}")
            raise

    def _initialize_engine_by_level(self):
        """根据优化级别初始化引擎"""
        # 性能目标配置
        perf_target = PerformanceTarget(
            target_speed=self.config.target_speed,
            max_memory_gb=self.config.max_memory_gb,
            target_accuracy=self.config.target_accuracy
        )

        # 根据优化级别设置参数
        if self.config.optimization_level == OptimizationLevel.BASIC:
            opt_config = OptimizationConfig(
                chunk_size=500,
                max_workers=2,
                enable_multiprocessing=False,
                enable_caching=self.config.enable_caching,
                gc_frequency=200
            )
        elif self.config.optimization_level == OptimizationLevel.STANDARD:
            opt_config = OptimizationConfig(
                chunk_size=self.config.chunk_size,
                max_workers=4,
                enable_multiprocessing=True,
                enable_caching=self.config.enable_caching,
                gc_frequency=100
            )
        elif self.config.optimization_level == OptimizationLevel.AGGRESSIVE:
            opt_config = OptimizationConfig(
                chunk_size=self.config.chunk_size * 2,
                max_workers=self.config.max_workers,
                enable_multiprocessing=True,
                enable_caching=self.config.enable_caching,
                gc_frequency=50
            )
        else:  # MAXIMUM
            opt_config = OptimizationConfig(
                chunk_size=self.config.chunk_size * 3,
                max_workers=self.config.max_workers * 2,
                enable_multiprocessing=True,
                enable_caching=self.config.enable_caching,
                gc_frequency=25
            )

        self.vectorized_engine = HighPerformanceBacktestEngine(opt_config, perf_target)

    def _initialize_parallel_processor(self):
        """初始化并行处理器"""
        processor_config = ProcessorConfig(
            max_workers=self.config.max_workers,
            batch_size=self.config.batch_size,
            enable_load_balancing=True,
            memory_threshold_gb=self.config.max_memory_gb * 0.9
        )
        self.parallel_optimizer = ParallelProcessingOptimizer(processor_config)

    @exception_handler(reraise=True)
    @performance_monitor(threshold=600.0)
    def run_backtest(self, request: BacktestRequest) -> BacktestResponse:
        """
        运行回测

        Args:
            request: 回测请求

        Returns:
            BacktestResponse: 回测响应
        """
        request_id = f"bt_{int(time.time())}_{id(request)}"
        start_time = time.time()

        self.logger.info(f"开始处理回测请求 {request_id}")
        self.logger.info(f"股票数量: {len(request.stock_codes)}")

        try:
            # 更新统计
            self.performance_stats['total_requests'] += 1

            # 检查缓存
            cached_result = self._check_cache(request)
            if cached_result:
                self.logger.info(f"回测请求 {request_id} 命中缓存")
                return self._create_response(request_id, True, cached_result, time.time() - start_time)

            # 选择执行策略
            result = self._execute_by_mode(request)

            if result:
                # 缓存结果
                self._cache_result(request, result)

                # 更新统计
                self.performance_stats['successful_requests'] += 1
                self.performance_stats['total_stocks_processed'] += len(request.stock_codes)

                execution_time = time.time() - start_time
                self.performance_stats['total_execution_time'] += execution_time

                # 更新平均速度
                if self.performance_stats['total_execution_time'] > 0:
                    self.performance_stats['average_speed'] = (
                        self.performance_stats['total_stocks_processed'] /
                        self.performance_stats['total_execution_time']
                    )

                self.logger.info(f"回测请求 {request_id} 完成，耗时: {execution_time:.2f}秒")
                return self._create_response(request_id, True, result, execution_time)

            else:
                self.performance_stats['failed_requests'] += 1
                return self._create_response(request_id, False, None, time.time() - start_time,
                                           "回测执行失败")

        except Exception as e:
            self.performance_stats['failed_requests'] += 1
            self.logger.error(f"回测请求 {request_id} 执行失败: {e}")
            return self._create_response(request_id, False, None, time.time() - start_time, str(e))

    def _execute_by_mode(self, request: BacktestRequest) -> Optional[Dict[str, Any]]:
        """根据引擎模式执行回测"""
        stock_count = len(request.stock_codes)

        # 自适应模式选择
        if self.config.engine_mode == EngineMode.ADAPTIVE:
            if stock_count < 100:
                mode = EngineMode.SINGLE_THREAD
            elif stock_count < 500:
                mode = EngineMode.MULTI_THREAD
            else:
                mode = EngineMode.MULTI_PROCESS
        else:
            mode = self.config.engine_mode

        self.logger.info(f"使用执行模式: {mode.value}")

        # 根据模式执行
        if mode == EngineMode.SINGLE_THREAD:
            return self._execute_single_thread(request)
        elif mode == EngineMode.MULTI_THREAD:
            return self._execute_multi_thread(request)
        else:  # MULTI_PROCESS
            return self._execute_multi_process(request)

    def _execute_single_thread(self, request: BacktestRequest) -> Optional[Dict[str, Any]]:
        """单线程执行"""
        if not self.vectorized_engine:
            self.logger.error("向量化引擎未初始化")
            return None

        # 使用内存优化器处理
        def processor(stock_batch):
            return self.vectorized_engine.run_high_performance_backtest(
                stock_codes=stock_batch,
                start_date=request.start_date,
                end_date=request.end_date,
                indicators=request.indicators or ['MA', 'MACD', 'RSI']
            )

        results = self.memory_optimizer.optimize_stock_batch_processing(
            stock_codes=request.stock_codes,
            processor_func=processor,
            memory_per_stock_mb=8
        )

        # 合并结果
        return self._merge_batch_results(results)

    def _execute_multi_thread(self, request: BacktestRequest) -> Optional[Dict[str, Any]]:
        """多线程执行"""
        # 配置多线程向量化引擎
        opt_config = OptimizationConfig(
            chunk_size=self.config.chunk_size,
            max_workers=4,
            enable_multiprocessing=False,  # 使用线程池而非进程池
            enable_caching=self.config.enable_caching
        )

        engine = HighPerformanceBacktestEngine(opt_config)
        return engine.run_high_performance_backtest(
            stock_codes=request.stock_codes,
            start_date=request.start_date,
            end_date=request.end_date,
            indicators=request.indicators or ['MA', 'MACD', 'RSI']
        )

    def _execute_multi_process(self, request: BacktestRequest) -> Optional[Dict[str, Any]]:
        """多进程执行"""
        if not self.parallel_optimizer:
            self.logger.error("并行处理器未初始化")
            return None

        return self.parallel_optimizer.run_optimized_parallel_backtest(
            stock_codes=request.stock_codes,
            start_date=request.start_date,
            end_date=request.end_date,
            indicators=request.indicators or ['MA', 'MACD', 'RSI']
        )

    def _merge_batch_results(self, batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """合并批次结果"""
        merged_results = []
        total_execution_time = 0.0
        total_stocks = 0

        for batch_result in batch_results:
            if isinstance(batch_result, dict) and 'results' in batch_result:
                merged_results.extend(batch_result['results'])
                if 'performance_metrics' in batch_result:
                    total_execution_time += batch_result['performance_metrics'].get('execution_time', 0)
                if 'summary' in batch_result:
                    total_stocks += batch_result['summary'].get('total_stocks_processed', 0)

        return {
            'results': merged_results,
            'summary': {
                'total_stocks_processed': total_stocks,
                'total_execution_time': total_execution_time,
                'average_speed': total_stocks / total_execution_time if total_execution_time > 0 else 0
            },
            'performance_metrics': {
                'processing_speed': total_stocks / total_execution_time if total_execution_time > 0 else 0,
                'memory_usage_gb': self.memory_optimizer.monitor.get_current_memory_gb(),
                'execution_time': total_execution_time
            }
        }

    def _check_cache(self, request: BacktestRequest) -> Optional[Dict[str, Any]]:
        """检查缓存"""
        if not self.cache_system:
            return None

        return self.cache_system.get_backtest_cache(
            stock_codes=request.stock_codes,
            start_date=request.start_date,
            end_date=request.end_date,
            strategy_params=request.strategy_params or {}
        )

    def _cache_result(self, request: BacktestRequest, result: Dict[str, Any]):
        """缓存结果"""
        if not self.cache_system:
            return

        self.cache_system.set_backtest_cache(
            stock_codes=request.stock_codes,
            start_date=request.start_date,
            end_date=request.end_date,
            strategy_params=request.strategy_params or {},
            result=result,
            ttl=3600  # 1小时缓存
        )

    def _create_response(self,
                        request_id: str,
                        success: bool,
                        result: Optional[Dict[str, Any]],
                        execution_time: float,
                        error_message: Optional[str] = None) -> BacktestResponse:
        """创建响应"""
        if success and result:
            return BacktestResponse(
                request_id=request_id,
                success=True,
                results=result.get('results', []),
                performance_metrics=result.get('performance_metrics', {}),
                execution_time=execution_time
            )
        else:
            return BacktestResponse(
                request_id=request_id,
                success=False,
                results=[],
                performance_metrics={},
                execution_time=execution_time,
                error_message=error_message
            )

    def run_batch_backtest(self, requests: List[BacktestRequest]) -> List[BacktestResponse]:
        """批量运行回测"""
        self.logger.info(f"开始批量回测，请求数量: {len(requests)}")
        responses = []

        # 按优先级排序
        sorted_requests = sorted(requests, key=lambda r: r.priority, reverse=True)

        for request in sorted_requests:
            response = self.run_backtest(request)
            responses.append(response)

        return responses

    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        report = {
            'engine_info': {
                'mode': self.config.engine_mode.value,
                'optimization_level': self.config.optimization_level.value,
                'max_workers': self.config.max_workers,
                'chunk_size': self.config.chunk_size
            },
            'performance_stats': self.performance_stats.copy(),
            'component_reports': {}
        }

        # 添加各组件报告
        if self.vectorized_engine:
            report['component_reports']['vectorized_engine'] = self.vectorized_engine.get_performance_report()

        if self.memory_optimizer:
            report['component_reports']['memory_optimizer'] = self.memory_optimizer.get_memory_report()

        if self.cache_system:
            report['component_reports']['cache_system'] = self.cache_system.get_comprehensive_stats()

        # 更新缓存命中率
        if self.cache_system:
            cache_stats = self.cache_system.get_comprehensive_stats()
            overall_stats = cache_stats.get('overall_stats', {})
            self.performance_stats['cache_hit_rate'] = overall_stats.get('hit_rate', 0.0)

        return report

    def optimize_configuration(self, target_metrics: Dict[str, Any]) -> UnifiedConfig:
        """优化配置参数"""
        # 基于目标指标动态调整配置
        new_config = UnifiedConfig()

        # 根据目标速度调整
        target_speed = target_metrics.get('target_speed', self.config.target_speed)
        if target_speed > 15000:
            new_config.optimization_level = OptimizationLevel.MAXIMUM
            new_config.engine_mode = EngineMode.MULTI_PROCESS
            new_config.max_workers = 16
        elif target_speed > 10000:
            new_config.optimization_level = OptimizationLevel.AGGRESSIVE
            new_config.engine_mode = EngineMode.MULTI_PROCESS
            new_config.max_workers = 8
        else:
            new_config.optimization_level = OptimizationLevel.STANDARD
            new_config.engine_mode = EngineMode.ADAPTIVE

        # 根据内存限制调整
        max_memory = target_metrics.get('max_memory_gb', self.config.max_memory_gb)
        if max_memory < 2.0:
            new_config.chunk_size = 200
            new_config.cache_size_mb = 256
        elif max_memory < 4.0:
            new_config.chunk_size = 500
            new_config.cache_size_mb = 512
        else:
            new_config.chunk_size = 1000
            new_config.cache_size_mb = 1024

        return new_config

    def shutdown(self):
        """关闭引擎"""
        self.logger.info("关闭统一高性能回测引擎...")

        if self.parallel_optimizer:
            self.parallel_optimizer.shutdown_workers()

        if self.memory_optimizer:
            del self.memory_optimizer

        if self.cache_system:
            del self.cache_system

        self.logger.info("引擎关闭完成")

    def __del__(self):
        """析构函数"""
        try:
            self.shutdown()
        except:
            pass

# 便捷函数
def create_engine(mode: str = "adaptive",
                 optimization: str = "standard",
                 max_memory_gb: float = 4.0,
                 target_speed: int = 10000) -> UnifiedHighPerformanceBacktestEngine:
    """创建引擎的便捷函数"""
    config = UnifiedConfig(
        engine_mode=EngineMode(mode),
        optimization_level=getattr(OptimizationLevel, optimization.upper()),
        max_memory_gb=max_memory_gb,
        target_speed=target_speed
    )
    return UnifiedHighPerformanceBacktestEngine(config)

def run_simple_backtest(stock_codes: List[str],
                       start_date: str,
                       end_date: str,
                       indicators: List[str] = None) -> Dict[str, Any]:
    """简单回测的便捷函数"""
    engine = create_engine()

    request = BacktestRequest(
        stock_codes=stock_codes,
        start_date=start_date,
        end_date=end_date,
        indicators=indicators
    )

    response = engine.run_backtest(request)
    engine.shutdown()

    if response.success:
        return {
            'results': response.results,
            'performance_metrics': response.performance_metrics,
            'execution_time': response.execution_time
        }
    else:
        raise Exception(f"回测失败: {response.error_message}")