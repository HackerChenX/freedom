from utils.container import container
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
高性能智能缓存系统

实现多层次缓存机制，避免重复计算，包括：
1. 内存缓存 - LRU算法，快速访问
2. 磁盘缓存 - 持久化存储，大容量
3. 分布式缓存 - 多进程共享缓存
4. 智能预加载和预测缓存
"""

import os
import time
import pickle
import hashlib
import sqlite3
import threading
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

from utils.logger import get_logger
from db.sql_manager import SQLManager, QueryType

logger = get_logger(__name__)

@dataclass
class CacheConfig:
    """缓存配置"""
    # 内存缓存配置
    memory_cache_size: int = 10000  # 内存缓存条目数
    memory_max_size_mb: float = 1024  # 内存缓存最大大小(MB)

    # 磁盘缓存配置
    disk_cache_enabled: bool = True
    disk_cache_dir: str = "cache"
    disk_cache_size_gb: float = 2.0  # 磁盘缓存最大大小(GB)

    # 缓存策略配置
    default_ttl: int = 3600  # 默认TTL(秒)
    cleanup_interval: int = 300  # 清理间隔(秒)
    preload_enabled: bool = True  # 启用预加载

    # 压缩配置
    compression_enabled: bool = True
    compression_threshold_mb: float = 1.0  # 压缩阈值(MB)

@dataclass
class CacheStats:
    """缓存统计"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    memory_hits: int = 0
    disk_hits: int = 0
    evictions: int = 0
    compression_saves_mb: float = 0.0

    @property
    def hit_rate(self) -> float:
        """缓存命中率"""
        if self.total_requests == 0:
            return 0.0
        return (self.cache_hits / self.total_requests) * 100

    @property
    def memory_hit_rate(self) -> float:
        """内存缓存命中率"""
        if self.total_requests == 0:
            return 0.0
        return (self.memory_hits / self.total_requests) * 100

class CacheKey:
    """缓存键生成器"""

    @staticmethod
    def generate_key(prefix: str, params: Dict[str, Any]) -> str:
        """生成缓存键"""
        # 排序参数确保一致性
        sorted_params = sorted(params.items())
        params_str = str(sorted_params)

        # 生成哈希
        hash_object = hashlib.md5(params_str.encode())
        hash_hex = hash_object.hexdigest()

        return f"{prefix}:{hash_hex}"

    @staticmethod
    def generate_indicator_key(stock_code: str, indicator: str, period: str,
                             start_date: str, end_date: str, params: Dict[str, Any]) -> str:
        """生成指标缓存键"""
        key_params = {
            'stock_code': stock_code,
            'indicator': indicator,
            'period': period,
            'start_date': start_date,
            'end_date': end_date,
            'params': params
        }
        return CacheKey.generate_key("indicator", key_params)

    @staticmethod
    def generate_backtest_key(stock_codes: List[str], start_date: str,
                            end_date: str, strategy_params: Dict[str, Any]) -> str:
        """生成回测缓存键"""
        # 对股票代码排序确保一致性
        sorted_codes = sorted(stock_codes)
        key_params = {
            'stock_codes': sorted_codes,
            'start_date': start_date,
            'end_date': end_date,
            'strategy_params': strategy_params
        }
        return CacheKey.generate_key("backtest", key_params)

class MemoryCache:
    """内存缓存 - LRU算法"""

    def __init__(self, config: CacheConfig):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        self.config = config
        self.cache = OrderedDict()
        self.timestamps = {}
        self.sizes = {}  # 存储每个项目的大小
        self.current_size_mb = 0.0
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        """获取缓存项"""
        with self._lock:
            if key not in self.cache:
                return None

            # 检查TTL
            if self._is_expired(key):
                self._remove_item(key)
                return None

            # 移动到末尾(最近使用)
            value = self.cache[key]
            self.cache.move_to_end(key)
            return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存项"""
        with self._lock:
            # 估算大小
            item_size_mb = self._estimate_size(value)

            # 检查是否超过单项大小限制
            if item_size_mb > self.config.memory_max_size_mb * 0.1:
                logger.warning(f"缓存项过大，跳过内存缓存: {item_size_mb:.2f}MB")
                return False

            # 清理空间
            while (self.current_size_mb + item_size_mb > self.config.memory_max_size_mb or
                   len(self.cache) >= self.config.memory_cache_size):
                if not self._evict_lru():
                    break

            # 添加新项
            self.cache[key] = value
            self.timestamps[key] = time.time() + (ttl or self.config.default_ttl)
            self.sizes[key] = item_size_mb
            self.current_size_mb += item_size_mb

            return True

    def remove(self, key: str) -> bool:
        """移除缓存项"""
        with self._lock:
            if key in self.cache:
                self._remove_item(key)
                return True
            return False

    def clear(self):
        """清空缓存"""
        with self._lock:
            self.cache.clear()
            self.timestamps.clear()
            self.sizes.clear()
            self.current_size_mb = 0.0

    def cleanup_expired(self) -> int:
        """清理过期项"""
        with self._lock:
            expired_keys = []
            current_time = time.time()

            for key, expire_time in self.timestamps.items():
                if current_time > expire_time:
                    expired_keys.append(key)

            for key in expired_keys:
                self._remove_item(key)

            return len(expired_keys)

    def _is_expired(self, key: str) -> bool:
        """检查是否过期"""
        if key not in self.timestamps:
            return True
        return time.time() > self.timestamps[key]

    def _remove_item(self, key: str):
        """移除项目"""
        if key in self.cache:
            del self.cache[key]
        if key in self.timestamps:
            del self.timestamps[key]
        if key in self.sizes:
            self.current_size_mb -= self.sizes[key]
            del self.sizes[key]

    def _evict_lru(self) -> bool:
        """驱逐最近最少使用的项"""
        if not self.cache:
            return False

        # 获取最旧的项
        oldest_key = next(iter(self.cache))
        self._remove_item(oldest_key)
        return True

    def _estimate_size(self, value: Any) -> float:
        """估算对象大小(MB)"""
        try:
            if isinstance(value, pd.DataFrame):
                return value.memory_usage(deep=True).sum() / (1024 * 1024)
            elif isinstance(value, np.ndarray):
                return value.nbytes / (1024 * 1024)
            else:
                # 使用pickle序列化估算
                pickled = pickle.dumps(value)
                return len(pickled) / (1024 * 1024)
        except:
            return 0.1  # 默认估算

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        with self._lock:
            return {
                'items': len(self.cache),
                'size_mb': self.current_size_mb,
                'max_size_mb': self.config.memory_max_size_mb,
                'usage_percent': (self.current_size_mb / self.config.memory_max_size_mb) * 100
            }

class DiskCache:
    """磁盘缓存"""

    def __init__(self, config: CacheConfig):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        self.config = config
        self.cache_dir = config.disk_cache_dir
        self.db_path = os.path.join(self.cache_dir, "cache_metadata.db")
        self._lock = threading.RLock()

        # 确保缓存目录存在
        os.makedirs(self.cache_dir, exist_ok=True)

        # 初始化数据库
        self._init_database()

    def _init_database(self):
        """初始化数据库"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS cache_metadata (
                    key TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    created_time REAL NOT NULL,
                    expire_time REAL NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    last_access REAL NOT NULL
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_expire_time ON cache_metadata(expire_time)')

    def get(self, key: str) -> Optional[Any]:
        """获取缓存项"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        'SELECT file_path, expire_time FROM cache_metadata WHERE key = ?',
                        (key,)
                    )
                    row = cursor.fetchone()

                    if not row:
                        return None

                    file_path, expire_time = row

                    # 检查是否过期
                    if time.time() > expire_time:
                        self._remove_item(key, file_path)
                        return None

                    # 读取文件
                    full_path = os.path.join(self.cache_dir, file_path)
                    if not os.path.exists(full_path):
                        self._remove_metadata(key)
                        return None

                    with open(full_path, 'rb') as f:
                        data = f.read()

                    # 解压缩(如果需要)
                    if self.config.compression_enabled and file_path.endswith('.gz'):
                        import gzip
                        data = gzip.decompress(data)

                    # 反序列化
                    value = pickle.loads(data)

                    # 更新访问统计
                    conn.execute(
                        'UPDATE cache_metadata SET access_count = access_count + 1, last_access = ? WHERE key = ?',
                        (time.time(), key)
                    )

                    return value

            except Exception as e:
                logger.error(f"磁盘缓存读取失败 {key}: {e}")
                return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存项"""
        with self._lock:
            try:
                # 序列化
                data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
                original_size = len(data)

                # 压缩(如果需要)
                file_extension = ""
                if (self.config.compression_enabled and
                    original_size > self.config.compression_threshold_mb * 1024 * 1024):
                    import gzip
                    data = gzip.compress(data, compresslevel=6)
                    file_extension = ".gz"

                # 生成文件路径
                file_name = f"{hashlib.md5(key.encode()).hexdigest()}.cache{file_extension}"
                file_path = os.path.join(self.cache_dir, file_name)

                # 检查磁盘空间
                if not self._check_disk_space(len(data)):
                    self._cleanup_old_items()

                # 写入文件
                with open(file_path, 'wb') as f:
                    f.write(data)

                # 更新元数据
                expire_time = time.time() + (ttl or self.config.default_ttl)
                current_time = time.time()

                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        '''INSERT OR REPLACE INTO cache_metadata
                           (key, file_path, created_time, expire_time, size_bytes, last_access)
                           VALUES (?, ?, ?, ?, ?, ?)''',
                        (key, file_name, current_time, expire_time, len(data), current_time)
                    )

                return True

            except Exception as e:
                logger.error(f"磁盘缓存写入失败 {key}: {e}")
                return False

    def remove(self, key: str) -> bool:
        """移除缓存项"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute('SELECT file_path FROM cache_metadata WHERE key = ?', (key,))
                    row = cursor.fetchone()

                    if row:
                        file_path = row[0]
                        self._remove_item(key, file_path)
                        return True

                return False

            except Exception as e:
                logger.error(f"移除磁盘缓存失败 {key}: {e}")
                return False

    def cleanup_expired(self) -> int:
        """清理过期项"""
        with self._lock:
            try:
                current_time = time.time()
                with sqlite3.connect(self.db_path) as conn:
                    # 查找过期项
                    cursor = conn.execute(
                        'SELECT key, file_path FROM cache_metadata WHERE expire_time < ?',
                        (current_time,)
                    )
                    expired_items = cursor.fetchall()

                    # 删除过期项
                    for key, file_path in expired_items:
                        self._remove_item(key, file_path)

                    return len(expired_items)

            except Exception as e:
                logger.error(f"清理过期磁盘缓存失败: {e}")
                return 0

    def _check_disk_space(self, required_bytes: int) -> bool:
        """检查磁盘空间"""
        try:
            # 计算当前缓存大小
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('SELECT SUM(size_bytes) FROM cache_metadata')
                current_size = cursor.fetchone()[0] or 0

            max_size_bytes = self.config.disk_cache_size_gb * 1024 * 1024 * 1024
            return (current_size + required_bytes) <= max_size_bytes

        except:
            return True  # 假设有足够空间

    def _cleanup_old_items(self):
        """清理旧项目"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 删除最旧的20%项目
                cursor = conn.execute(
                    'SELECT key, file_path FROM cache_metadata ORDER BY last_access ASC LIMIT '
                    '(SELECT COUNT(*) * 0.2 FROM cache_metadata)'
                )
                old_items = cursor.fetchall()

                for key, file_path in old_items:
                    self._remove_item(key, file_path)

                logger.info(f"清理了 {len(old_items)} 个旧缓存项")

        except Exception as e:
            logger.error(f"清理旧缓存项失败: {e}")

    def _remove_item(self, key: str, file_path: str):
        """移除项目"""
        try:
            # 删除文件
            full_path = os.path.join(self.cache_dir, file_path)
            if os.path.exists(full_path):
                os.remove(full_path)

            # 删除元数据
            self._remove_metadata(key)

        except Exception as e:
            logger.error(f"移除缓存项失败 {key}: {e}")

    def _remove_metadata(self, key: str):
        """移除元数据"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('DELETE FROM cache_metadata WHERE key = ?', (key,))
        except Exception as e:
            logger.error(f"移除缓存元数据失败 {key}: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """获取磁盘缓存统计"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    'SELECT COUNT(*), SUM(size_bytes), AVG(access_count) FROM cache_metadata'
                )
                count, total_size, avg_access = cursor.fetchone()

                return {
                    'items': count or 0,
                    'size_bytes': total_size or 0,
                    'size_mb': (total_size or 0) / (1024 * 1024),
                    'average_access_count': avg_access or 0,
                    'max_size_gb': self.config.disk_cache_size_gb
                }

        except Exception as e:
            logger.error(f"获取磁盘缓存统计失败: {e}")
            return {'items': 0, 'size_bytes': 0, 'size_mb': 0, 'average_access_count': 0}

class IntelligentCacheSystem:
    """
    智能缓存系统

    多层次缓存架构，支持内存和磁盘缓存，智能预加载和预测缓存
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        """初始化智能缓存系统"""
        self.config = config or CacheConfig()
        self.logger = logger

        # 初始化缓存层
        self.memory_cache = MemoryCache(self.config)
        self.disk_cache = DiskCache(self.config) if self.config.disk_cache_enabled else None

        # 统计信息
        self.stats = CacheStats()

        # 清理线程
        self.cleanup_thread = None
        self.cleanup_active = False

        # 预加载器
        self.preloader = CachePreloader(self) if self.config.preload_enabled else None

        # 启动清理线程
        self.start_cleanup_thread()

        self.logger.info("智能缓存系统初始化完成")

    def get(self, key: str) -> Optional[Any]:
        """获取缓存项"""
        self.stats.total_requests += 1

        # 首先尝试内存缓存
        value = self.memory_cache.get(key)
        if value is not None:
            self.stats.cache_hits += 1
            self.stats.memory_hits += 1
            return value

        # 然后尝试磁盘缓存
        if self.disk_cache:
            value = self.disk_cache.get(key)
            if value is not None:
                self.stats.cache_hits += 1
                self.stats.disk_hits += 1

                # 提升到内存缓存
                self.memory_cache.set(key, value)
                return value

        # 缓存未命中
        self.stats.cache_misses += 1
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存项"""
        success = False

        # 设置内存缓存
        if self.memory_cache.set(key, value, ttl):
            success = True

        # 设置磁盘缓存
        if self.disk_cache and self.disk_cache.set(key, value, ttl):
            success = True

        return success

    def remove(self, key: str) -> bool:
        """移除缓存项"""
        memory_removed = self.memory_cache.remove(key)
        disk_removed = self.disk_cache.remove(key) if self.disk_cache else False

        return memory_removed or disk_removed

    def clear(self):
        """清空所有缓存"""
        self.memory_cache.clear()
        if self.disk_cache:
            # 磁盘缓存清理需要删除文件
            pass  # 实现磁盘缓存清理

    # 业务特定的缓存方法

    def get_indicator_cache(self, stock_code: str, indicator: str, period: str,
                          start_date: str, end_date: str, params: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """获取指标缓存"""
        key = CacheKey.generate_indicator_key(stock_code, indicator, period, start_date, end_date, params)
        return self.get(key)

    def set_indicator_cache(self, stock_code: str, indicator: str, period: str,
                          start_date: str, end_date: str, params: Dict[str, Any],
                          data: pd.DataFrame, ttl: Optional[int] = None) -> bool:
        """设置指标缓存"""
        key = CacheKey.generate_indicator_key(stock_code, indicator, period, start_date, end_date, params)
        return self.set(key, data, ttl)

    def get_backtest_cache(self, stock_codes: List[str], start_date: str,
                         end_date: str, strategy_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """获取回测缓存"""
        key = CacheKey.generate_backtest_key(stock_codes, start_date, end_date, strategy_params)
        return self.get(key)

    def set_backtest_cache(self, stock_codes: List[str], start_date: str,
                         end_date: str, strategy_params: Dict[str, Any],
                         result: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """设置回测缓存"""
        key = CacheKey.generate_backtest_key(stock_codes, start_date, end_date, strategy_params)
        return self.set(key, result, ttl)

    def start_cleanup_thread(self):
        """启动清理线程"""
        if self.cleanup_thread is None:
            self.cleanup_active = True
            self.cleanup_thread = threading.Thread(target=self._cleanup_loop)
            self.cleanup_thread.daemon = True
            self.cleanup_thread.start()

    def stop_cleanup_thread(self):
        """停止清理线程"""
        self.cleanup_active = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=10)

    def _cleanup_loop(self):
        """清理循环"""
        while self.cleanup_active:
            try:
                # 清理过期项
                memory_expired = self.memory_cache.cleanup_expired()
                disk_expired = self.disk_cache.cleanup_expired() if self.disk_cache else 0

                if memory_expired + disk_expired > 0:
                    self.logger.debug(f"清理过期缓存: 内存 {memory_expired}, 磁盘 {disk_expired}")

                # 等待下次清理
                threading.Event().wait(self.config.cleanup_interval)

            except Exception as e:
                self.logger.error(f"缓存清理出错: {e}")

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """获取综合缓存统计"""
        memory_stats = self.memory_cache.get_stats()
        disk_stats = self.disk_cache.get_stats() if self.disk_cache else {}

        return {
            'overall_stats': asdict(self.stats),
            'memory_cache': memory_stats,
            'disk_cache': disk_stats,
            'config': asdict(self.config)
        }

    def __del__(self):
        """析构函数"""
        self.stop_cleanup_thread()

class CachePreloader:
    """缓存预加载器"""

    def __init__(self, cache_system: IntelligentCacheSystem):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        self.cache_system = cache_system
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.preload_patterns = {}

    def preload_indicators(self, stock_codes: List[str], indicators: List[str],
                         date_range: Tuple[str, str]):
        """预加载指标数据"""
        def preload_task():
            start_date, end_date = date_range
            for stock_code in stock_codes:
                for indicator in indicators:
                    # 检查是否已缓存
                    key = CacheKey.generate_indicator_key(
                        stock_code, indicator, '日线', start_date, end_date, {}
                    )
                    if self.cache_system.get(key) is None:
                        # 这里应该调用实际的指标计算函数
                        # 然后将结果缓存
                        pass

        self.executor.submit(preload_task)

    def add_preload_pattern(self, pattern_name: str, pattern_func: Callable):
        """添加预加载模式"""
        self.preload_patterns[pattern_name] = pattern_func

    def execute_preload_pattern(self, pattern_name: str, **kwargs):
        """执行预加载模式"""
        if pattern_name in self.preload_patterns:
            pattern_func = self.preload_patterns[pattern_name]
            self.executor.submit(pattern_func, **kwargs)