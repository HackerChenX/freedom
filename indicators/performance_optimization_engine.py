from typing import Dict, Any
from utils.container import container
from indicators.base_indicator import BaseIndicator
"""
指标计算性能优化引擎
提供指标计算缓存,批量优化,时间预测和调度优化
"""

import time
import threading
import hashlib
import pickle
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

import pandas as pd
import numpy as np

from utils.advanced_performance_monitor import (
    indicator_performance_monitor, 
    get_performance_analyzer,
    AdvancedPerformanceMetric
)
from utils.enhanced_exception_handler import exception_handler, ErrorSeverity, ErrorCategory
from config.unified_config_manager import get_config
from indicators.complete_indicator_registry import get_indicator_registry
from db.sql_manager import SQLManager, QueryType

logger = logging.getLogger(__name__)


@dataclass
class IndicatorPerformanceProfile(BaseIndicator):
    """指标性能档案"""
    indicator_name: str
    avg_calculation_time: float
    min_calculation_time: float
    max_calculation_time: float
    memory_usage: float
    complexity_score: float
    cache_hit_rate: float
    optimization_level: str
    last_updated: datetime
    calculation_count: int = 0
    error_count: int = 0
    data_size_correlation: float = 0.0  # 数据大小与计算时间的相关性


@dataclass
class BatchCalculationTask(BaseIndicator):
    """批量计算任务"""
    task_id: str
    indicator_names: List[str]
    data: pd.DataFrame
    priority: int
    estimated_time: float
    created_at: datetime
    dependencies: List[str] = field(default_factory=list)
    callback: Optional[Callable] = None


class IndicatorCacheService(BaseIndicator):
    """指标缓存管理器"""
    
    def __init__(self, max_memory_cache: int = 1000, enable_disk_cache: bool = True):  # TODO: 将魔法数字提取到配置中
            super().__init__(name=self.__class__.__name__, **kwargs)
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        self.max_memory_cache = max_memory_cache
        self.enable_disk_cache = enable_disk_cache
        
        # 内存缓存
        self.memory_cache: Dict[str, Any] = {}
        self.cache_access_times: Dict[str, datetime] = {}
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        
        # 缓存策略配置
        self.cache_ttl = get_config('indicator_cache.ttl', 3600)  # 1小时  # TODO: 将魔法数字提取到配置中
        self.cache_size_limit = get_config('indicator_cache.size_limit', 100 * 1024 * 1024)  # 100MB  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        
        # 线程安全
        self.lock = threading.RLock()
        
        logger.info(f"指标缓存管理器初始化 - 内存缓存: {max_memory_cache}, 磁盘缓存: {enable_disk_cache}")
    
    def _generate_cache_key(self, indicator_name: str, data: pd.DataFrame, params: Dict = None) -> str:
        """生成缓存键"""
        # 使用数据哈希和参数生成唯一键
        data_hash = pd.util.hash_pandas_object(data).sum()
        params_str = str(sorted((params or {}).items()))
        
        key_content = f"{indicator_name}_{data_hash}_{params_str}"
        return hashlib.md5(key_content.encode()).hexdigest()
    
    @indicator_performance_monitor(threshold_seconds=0.1)
    def get_cached_result(self, indicator_name: str, data: pd.DataFrame, params: Dict = None) -> Optional[Any]:
        """获取缓存结果"""
        cache_key = self._generate_cache_key(indicator_name, data, params)
        
        with self.lock:
            # 检查内存缓存
            if cache_key in self.memory_cache:
                # 检查TTL
                access_time = self.cache_access_times.get(cache_key)
                if access_time and (datetime.now() - access_time).total_seconds() < self.cache_ttl:
                    self.cache_access_times[cache_key] = datetime.now()
                    self.cache_hit_count += 1
                    logger.debug(f"缓存命中: {indicator_name}")
                    return self.memory_cache[cache_key]
                else:
                    # 过期,删除缓存
                    del self.memory_cache[cache_key]
                    if cache_key in self.cache_access_times:
                        del self.cache_access_times[cache_key]
            
            self.cache_miss_count += 1
            return None
    
    @indicator_performance_monitor(threshold_seconds=0.1)
    def cache_result(self, indicator_name: str, data: pd.DataFrame, result: Any, params: Dict = None):
        """缓存计算结果"""
        cache_key = self._generate_cache_key(indicator_name, data, params)
        
        with self.lock:
            # 检查缓存大小限制
            if len(self.memory_cache) >= self.max_memory_cache:
                self._evict_oldest_cache()
            
            self.memory_cache[cache_key] = result
            self.cache_access_times[cache_key] = datetime.now()
            
            logger.debug(f"缓存结果: {indicator_name}")
    
    def _evict_oldest_cache(self):
        """驱逐最旧的缓存"""
        if not self.cache_access_times:
            return
        
        # 找到最旧的缓存项
        oldest_key = min(self.cache_access_times.keys(), 
                        key=lambda k: self.cache_access_times[k])
        
        # 删除最旧的缓存
        if oldest_key in self.memory_cache:
            del self.memory_cache[oldest_key]
        del self.cache_access_times[oldest_key]
        
        logger.debug(f"驱逐缓存: {oldest_key}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total_requests = self.cache_hit_count + self.cache_miss_count
        hit_rate = (self.cache_hit_count / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'hit_count': self.cache_hit_count,
            'miss_count': self.cache_miss_count,
            'hit_rate': hit_rate,
            'cache_size': len(self.memory_cache),
            'max_cache_size': self.max_memory_cache
        }


class IndicatorPerformanceProfiler(BaseIndicator):
    """指标性能分析器"""
    
    def __init__(self):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        self.profiles: Dict[str, IndicatorPerformanceProfile] = {}
        self.calculation_history: deque = deque(maxlen=10000)  # TODO: 将魔法数字提取到配置中
        self.lock = threading.RLock()
        
        logger.info("指标性能分析器初始化完成")
    
    @indicator_performance_monitor(threshold_seconds=0.1)
    def record_calculation(self, indicator_name: str, calculation_time: float, 
                          memory_usage: float, data_size: int, success: bool):
        """记录指标计算性能"""
        with self.lock:
            # 更新或创建性能档案
            if indicator_name not in self.profiles:
                self.profiles[indicator_name] = IndicatorPerformanceProfile(
                    indicator_name=indicator_name,
                    avg_calculation_time=calculation_time,
                    min_calculation_time=calculation_time,
                    max_calculation_time=calculation_time,
                    memory_usage=memory_usage,
                    complexity_score=self._calculate_complexity_score(calculation_time, data_size),
                    cache_hit_rate=0.0,
                    optimization_level="none",
                    last_updated=datetime.now(),
                    calculation_count=1,
                    error_count=0 if success else 1
                )
            else:
                profile = self.profiles[indicator_name]
                profile.calculation_count += 1
                if not success:
                    profile.error_count += 1
                
                # 更新统计信息
                profile.avg_calculation_time = (
                    (profile.avg_calculation_time * (profile.calculation_count - 1) + calculation_time) 
                    / profile.calculation_count
                )
                profile.min_calculation_time = min(profile.min_calculation_time, calculation_time)
                profile.max_calculation_time = max(profile.max_calculation_time, calculation_time)
                profile.memory_usage = max(profile.memory_usage, memory_usage)
                profile.complexity_score = self._calculate_complexity_score(
                    profile.avg_calculation_time, data_size
                )
                profile.last_updated = datetime.now()
            
            # 记录历史
            self.calculation_history.append({
                'indicator_name': indicator_name,
                'calculation_time': calculation_time,
                'memory_usage': memory_usage,
                'data_size': data_size,
                'success': success,
                'timestamp': datetime.now()
            })
    
    def _calculate_complexity_score(self, calculation_time: float, data_size: int) -> float:
        """计算复杂度分数"""
        # 基于计算时间和数据大小的复杂度评分
        if data_size == 0:
            return calculation_time * 10
        
        time_per_record = calculation_time / data_size
        
        if time_per_record < 0.0001:  # < 0.1ms per record
            return 1.0  # 低复杂度
        elif time_per_record < 0.001:  # < 1ms per record
            return 2.0  # 中等复杂度
        elif time_per_record < 0.01:   # < 10ms per record
            return 3.0  # 高复杂度  # TODO: 将魔法数字提取到配置中
        else:
            return 4.0  # 极高复杂度  # TODO: 将魔法数字提取到配置中
    
    def predict_calculation_time(self, indicator_name: str, data_size: int) -> float:
        """预测指标计算时间"""
        if indicator_name not in self.profiles:
            # 没有历史数据,使用默认估算
            return self._estimate_default_time(indicator_name, data_size)
        
        profile = self.profiles[indicator_name]
        
        # 基于历史数据和数据大小预测
        base_time = profile.avg_calculation_time
        
        # 考虑数据大小的影响
        if profile.data_size_correlation > 0:
            # 线性相关
            estimated_time = base_time * (data_size / 1000)  # 假设基准是1000条记录  # TODO: 将魔法数字提取到配置中
        else:
            # 固定时间
            estimated_time = base_time
        
        # 考虑复杂度
        complexity_multiplier = 1.0 + (profile.complexity_score - 1.0) * 0.2
        estimated_time *= complexity_multiplier
        
        return max(0.001, estimated_time)  # 最少1ms
    
    def _estimate_default_time(self, indicator_name: str, data_size: int) -> float:
        """估算默认计算时间"""
        # 基于指标类型的默认估算
        base_times = {
            'MA': 0.001,      # 移动平均 - 简单
            'EMA': 0.002,     # 指数移动平均 - 简单
            'MACD': 0.005,    # MACD - 中等  # TODO: 将魔法数字提取到配置中
            'RSI': 0.003,     # RSI - 中等  # TODO: 将魔法数字提取到配置中
            'BOLL': 0.004,    # 布林带 - 中等  # TODO: 将魔法数字提取到配置中
            'KDJ': 0.006,     # KDJ - 复杂  # TODO: 将魔法数字提取到配置中
            'CCI': 0.008,     # CCI - 复杂  # TODO: 将魔法数字提取到配置中
            'ATR': 0.004,     # ATR - 中等  # TODO: 将魔法数字提取到配置中
        }
        
        # 查找匹配的基础时间
        base_time = 0.005  # 默认5ms  # TODO: 将魔法数字提取到配置中
        for pattern, time_val in base_times.items():
            if pattern in indicator_name.upper():
                base_time = time_val
                break
        
        # 根据数据大小调整
        return base_time * (data_size / 1000)  # TODO: 将魔法数字提取到配置中
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        with self.lock:
            if not self.profiles:
                return {'message': '暂无性能数据'}
            
            # 计算总体统计
            total_calculations = sum(p.calculation_count for p in self.profiles.values())
            total_errors = sum(p.error_count for p in self.profiles.values())
            avg_time = np.mean([p.avg_calculation_time for p in self.profiles.values()])
            
            # 最慢的指标
            slowest_indicators = sorted(
                self.profiles.values(), 
                key=lambda p: p.avg_calculation_time, 
                reverse=True
            )[:10]
            
            # 最复杂的指标
            most_complex_indicators = sorted(
                self.profiles.values(),
                key=lambda p: p.complexity_score,
                reverse=True
            )[:10]
            
            return {
                'total_indicators': len(self.profiles),
                'total_calculations': total_calculations,
                'total_errors': total_errors,
                'error_rate': (total_errors / total_calculations * 100) if total_calculations > 0 else 0,
                'avg_calculation_time': avg_time,
                'slowest_indicators': [
                    {
                        'name': p.indicator_name,
                        'avg_time': p.avg_calculation_time,
                        'complexity_score': p.complexity_score
                    }
                    for p in slowest_indicators
                ],
                'most_complex_indicators': [
                    {
                        'name': p.indicator_name,
                        'complexity_score': p.complexity_score,
                        'avg_time': p.avg_calculation_time
                    }
                    for p in most_complex_indicators
                ]
            }


class BatchCalculationOptimizer(BaseIndicator):
    """批量计算优化器"""
    
    def __init__(self, max_workers: int = 4):  # TODO: 将魔法数字提取到配置中
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        self.max_workers = max_workers
        self.task_queue: deque = deque()
        self.active_tasks: Dict[str, BatchCalculationTask] = {}
        self.completed_tasks: Dict[str, Any] = {}
        
        self.profiler = IndicatorPerformanceProfiler()
        self.cache_manager = IndicatorCacheService()
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.lock = threading.RLock()
        
        logger.info(f"批量计算优化器初始化 - 工作线程: {max_workers}")
    
    @indicator_performance_monitor(threshold_seconds=5.0)  # TODO: 将魔法数字提取到配置中
    def submit_batch_calculation(self, indicator_names: List[str], data: pd.DataFrame,
                               priority: int = 1, dependencies: List[str] = None,
                               callback: Callable = None) -> str:
        """提交批量计算任务"""
        task_id = f"batch_{int(time.time() * 1000)}"  # TODO: 将魔法数字提取到配置中
        
        # 预测总执行时间
        estimated_time = sum(
            self.profiler.predict_calculation_time(name, len(data))
            for name in indicator_names
        )
        
        task = BatchCalculationTask(
            task_id=task_id,
            indicator_names=indicator_names,
            data=data,
            priority=priority,
            estimated_time=estimated_time,
            created_at=datetime.now(),
            dependencies=dependencies or [],
            callback=callback
        )
        
        with self.lock:
            self.task_queue.append(task)
            self.active_tasks[task_id] = task
        
        # 启动任务处理
        self._process_tasks()
        
        logger.info(f"提交批量计算任务: {task_id}, 指标数: {len(indicator_names)}, 预计时间: {estimated_time:.3f}s")
        return task_id
    
    def _process_tasks(self):
        """处理任务队列"""
        with self.lock:
            # 按优先级和预计时间排序
            sorted_tasks = sorted(
                self.task_queue,
                key=lambda t: (t.priority, t.estimated_time),
                reverse=True
            )
            
            for task in sorted_tasks:
                if self._can_execute_task(task):
                    self.task_queue.remove(task)
                    future = self.executor.submit(self._execute_batch_task, task)
                    future.add_done_callback(lambda f, t=task: self._task_completed(t, f))
    
    def _can_execute_task(self, task: BatchCalculationTask) -> bool:
        """检查任务是否可以执行"""
        # 检查依赖是否完成
        for dep in task.dependencies:
            if dep not in self.completed_tasks:
                return False
        return True
    
    @indicator_performance_monitor(threshold_seconds=10.0)
    def _execute_batch_task(self, task: BatchCalculationTask) -> Dict[str, Any]:
        """执行批量计算任务"""
        results = {}
        start_time = time.time()
        
        logger.info(f"开始执行批量任务: {task.task_id}")
        
        for indicator_name in task.indicator_names:
            try:
                # 检查缓存
                cached_result = self.cache_manager.get_cached_result(indicator_name, task.data)
                
                if cached_result is not None:
                    results[indicator_name] = cached_result
                    logger.debug(f"使用缓存结果: {indicator_name}")
                    continue
                
                # 执行计算
                calc_start = time.time()
                registry = get_indicator_registry()
                indicator = registry.get_indicator(indicator_name)
                
                if indicator:
                    result = indicator.calculate(task.data)
                    calc_time = time.time() - calc_start
                    
                    # 记录性能
                    self.profiler.record_calculation(
                        indicator_name, calc_time, 0, len(task.data), True
                    )
                    
                    # 缓存结果
                    self.cache_manager.cache_result(indicator_name, task.data, result)
                    
                    results[indicator_name] = result
                    logger.debug(f"计算完成: {indicator_name} ({calc_time:.3f}s)")
                else:
                    logger.warning(f"指标未找到: {indicator_name}")
                    results[indicator_name] = None
                    
            except Exception as e:
                logger.error(f"指标计算失败 {indicator_name}: {e}")
                results[indicator_name] = None
                
                # 记录错误
                self.profiler.record_calculation(
                    indicator_name, 0, 0, len(task.data), False
                )
        
        total_time = time.time() - start_time
        logger.info(f"批量任务完成: {task.task_id} ({total_time:.3f}s)")
        
        return results
    
    def _task_completed(self, task: BatchCalculationTask, future):
        """任务完成回调"""
        try:
            results = future.result()
            
            with self.lock:
                self.completed_tasks[task.task_id] = results
                if task.task_id in self.active_tasks:
                    del self.active_tasks[task.task_id]
            
            # 执行回调
            if task.callback:
                try:
                    task.callback(task.task_id, results)
                except Exception as e:
                    logger.error(f"任务回调执行失败 {task.task_id}: {e}")
            
            # 处理等待的任务
            self._process_tasks()
            
        except Exception as e:
            logger.error(f"任务执行失败 {task.task_id}: {e}")
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """获取任务状态"""
        with self.lock:
            if task_id in self.completed_tasks:
                return {
                    'status': 'completed',
                    'results': self.completed_tasks[task_id]
                }
            elif task_id in self.active_tasks:
                return {
                    'status': 'running',
                    'task': self.active_tasks[task_id]
                }
            else:
                return {'status': 'not_found'}
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """获取优化统计"""
        cache_stats = self.cache_manager.get_cache_stats()
        performance_summary = self.profiler.get_performance_summary()
        
        with self.lock:
            queue_size = len(self.task_queue)
            active_count = len(self.active_tasks)
            completed_count = len(self.completed_tasks)
        
        return {
            'cache_stats': cache_stats,
            'performance_summary': performance_summary,
            'task_stats': {
                'queue_size': queue_size,
                'active_tasks': active_count,
                'completed_tasks': completed_count
            }
        }


# 全局优化器实例
_batch_optimizer = None
_optimizer_lock = threading.Lock()


def get_batch_optimizer() -> BatchCalculationOptimizer:
    """获取全局批量计算优化器实例"""
    global _batch_optimizer
    
    if _batch_optimizer is None:
        with _optimizer_lock:
            if _batch_optimizer is None:
                _batch_optimizer = BatchCalculationOptimizer()
    
    return _batch_optimizer


# 导出主要类
__all__ = [
    'IndicatorCacheService',
    'IndicatorPerformanceProfiler',
    'BatchCalculationOptimizer',
    'IndicatorPerformanceProfile',
    'BatchCalculationTask',
    'get_batch_optimizer'
]

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算指标值
        
        Args:
            data: 输入数据,包含OHLCV等字段
            
        Returns:
            pd.DataFrame: 包含指标计算结果的数据框
        """
        if not self.validate_data(data):
            raise ValueError("输入数据不符合要求")
        
        # 预处理数据
        processed_data = self.preprocess_data(data)
        
        # TODO: 实现具体的指标计算逻辑
        result = processed_data.copy()
        result[f'{self.name}_value'] = processed_data['close'].rolling(window=self.period).mean()
        
        # 后处理结果
        result = self.postprocess_result(result)
        
        # 保存结果
        self._result = result
        
        return result

    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        获取交易信号
        
        Args:
            data: 包含指标计算结果的数据
            
        Returns:
            Dict[str, Any]: 交易信号信息
        """
        if data.empty:
            return {'signal': 'hold', 'strength': 0.0, 'timestamp': None}
        
        # TODO: 实现具体的信号生成逻辑
        latest_close = data['close'].iloc[-1] if 'close' in data.columns else 0
        
        return {
            'signal': 'hold',
            'strength': 0.0,
            'timestamp': data.index[-1] if not data.empty else None,
            'price': latest_close,
            'indicator': self.name
        }
