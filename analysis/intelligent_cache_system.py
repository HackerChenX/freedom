#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
智能缓存系统

实现技术指标计算结果的智能缓存机制
"""

import os
import sys
import time
import hashlib
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import json
from datetime import datetime, timedelta
from collections import OrderedDict
import threading

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from utils.logger import get_logger

logger = get_logger(__name__)


class LRUCache:
    """LRU缓存实现"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.lock = threading.RLock()
        
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self.lock:
            if key in self.cache:
                # 移动到末尾（最近使用）
                value = self.cache.pop(key)
                self.cache[key] = value
                return value
            return None
    
    def set(self, key: str, value: Any) -> None:
        """设置缓存值"""
        with self.lock:
            if key in self.cache:
                # 更新现有值
                self.cache.pop(key)
            elif len(self.cache) >= self.max_size:
                # 删除最久未使用的项
                self.cache.popitem(last=False)
            
            self.cache[key] = value
    
    def clear(self) -> None:
        """清空缓存"""
        with self.lock:
            self.cache.clear()
    
    def size(self) -> int:
        """获取缓存大小"""
        return len(self.cache)
    
    def hit_rate(self) -> float:
        """获取缓存命中率（简化版本）"""
        return 0.0  # 需要额外的统计逻辑


class IntelligentCacheSystem:
    """智能缓存系统"""
    
    def __init__(self, max_memory_cache_size: int = 1000, 
                 enable_disk_cache: bool = True,
                 cache_dir: str = "data/cache"):
        self.memory_cache = LRUCache(max_memory_cache_size)
        self.enable_disk_cache = enable_disk_cache
        self.cache_dir = cache_dir
        self.hit_count = 0
        self.miss_count = 0
        
        # 创建缓存目录
        if self.enable_disk_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def _generate_cache_key(self, stock_code: str, indicator_name: str, 
                          end_date: str, params: Dict[str, Any]) -> str:
        """生成缓存键"""
        # 创建参数的哈希值
        params_str = json.dumps(params, sort_keys=True)
        key_data = f"{stock_code}_{indicator_name}_{end_date}_{params_str}"
        
        # 使用MD5生成短键
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_disk_cache_path(self, cache_key: str) -> str:
        """获取磁盘缓存文件路径"""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def _load_from_disk(self, cache_key: str) -> Optional[Any]:
        """从磁盘加载缓存"""
        if not self.enable_disk_cache:
            return None
        
        cache_path = self._get_disk_cache_path(cache_key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # 检查缓存是否过期（24小时）
                if 'timestamp' in cached_data:
                    cache_time = datetime.fromisoformat(cached_data['timestamp'])
                    if datetime.now() - cache_time < timedelta(hours=24):
                        return cached_data['data']
                    else:
                        # 删除过期缓存
                        os.remove(cache_path)
                
            except Exception as e:
                logger.warning(f"加载磁盘缓存失败: {e}")
                # 删除损坏的缓存文件
                try:
                    os.remove(cache_path)
                except:
                    pass
        
        return None
    
    def _save_to_disk(self, cache_key: str, data: Any) -> None:
        """保存到磁盘缓存"""
        if not self.enable_disk_cache:
            return
        
        cache_path = self._get_disk_cache_path(cache_key)
        try:
            cached_data = {
                'timestamp': datetime.now().isoformat(),
                'data': data
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cached_data, f)
                
        except Exception as e:
            logger.warning(f"保存磁盘缓存失败: {e}")
    
    def get_cached_indicator(self, stock_code: str, indicator_name: str, 
                           end_date: str, params: Dict[str, Any] = None) -> Optional[Any]:
        """获取缓存的指标结果"""
        if params is None:
            params = {}
        
        cache_key = self._generate_cache_key(stock_code, indicator_name, end_date, params)
        
        # 首先尝试内存缓存
        result = self.memory_cache.get(cache_key)
        if result is not None:
            self.hit_count += 1
            logger.debug(f"内存缓存命中: {indicator_name} for {stock_code}")
            return result
        
        # 然后尝试磁盘缓存
        result = self._load_from_disk(cache_key)
        if result is not None:
            # 将磁盘缓存结果加载到内存缓存
            self.memory_cache.set(cache_key, result)
            self.hit_count += 1
            logger.debug(f"磁盘缓存命中: {indicator_name} for {stock_code}")
            return result
        
        self.miss_count += 1
        return None
    
    def cache_indicator_result(self, stock_code: str, indicator_name: str, 
                             end_date: str, result: Any, 
                             params: Dict[str, Any] = None) -> None:
        """缓存指标结果"""
        if params is None:
            params = {}
        
        cache_key = self._generate_cache_key(stock_code, indicator_name, end_date, params)
        
        # 保存到内存缓存
        self.memory_cache.set(cache_key, result)
        
        # 保存到磁盘缓存
        self._save_to_disk(cache_key, result)
        
        logger.debug(f"缓存指标结果: {indicator_name} for {stock_code}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'memory_cache_size': self.memory_cache.size(),
            'disk_cache_enabled': self.enable_disk_cache
        }
    
    def clear_cache(self, clear_disk: bool = False) -> None:
        """清空缓存"""
        self.memory_cache.clear()
        
        if clear_disk and self.enable_disk_cache:
            try:
                import shutil
                shutil.rmtree(self.cache_dir)
                os.makedirs(self.cache_dir, exist_ok=True)
                logger.info("磁盘缓存已清空")
            except Exception as e:
                logger.warning(f"清空磁盘缓存失败: {e}")
        
        # 重置统计
        self.hit_count = 0
        self.miss_count = 0
        
        logger.info("缓存已清空")


class CachedIndicatorCalculator:
    """带缓存的指标计算器"""
    
    def __init__(self, cache_system: IntelligentCacheSystem):
        self.cache_system = cache_system
    
    def calculate_with_cache(self, stock_code: str, indicator_name: str, 
                           end_date: str, data: pd.DataFrame, 
                           calculation_func: callable, 
                           params: Dict[str, Any] = None) -> Any:
        """
        带缓存的指标计算
        
        Args:
            stock_code: 股票代码
            indicator_name: 指标名称
            end_date: 结束日期
            data: 股票数据
            calculation_func: 计算函数
            params: 计算参数
            
        Returns:
            Any: 计算结果
        """
        if params is None:
            params = {}
        
        # 尝试从缓存获取
        cached_result = self.cache_system.get_cached_indicator(
            stock_code, indicator_name, end_date, params
        )
        
        if cached_result is not None:
            return cached_result
        
        # 缓存未命中，执行计算
        start_time = time.time()
        result = calculation_func(data, **params)
        calculation_time = time.time() - start_time
        
        # 缓存结果
        self.cache_system.cache_indicator_result(
            stock_code, indicator_name, end_date, result, params
        )
        
        logger.debug(f"计算并缓存指标 {indicator_name}: {calculation_time:.4f}s")
        
        return result


def benchmark_cache_performance():
    """缓存性能基准测试"""
    import os
    import json
    import numpy as np
    import pandas as pd
    from datetime import datetime

    print("="*60)
    print("智能缓存系统性能测试")
    print("="*60)
    
    # 创建缓存系统
    cache_system = IntelligentCacheSystem(max_memory_cache_size=100)
    calculator = CachedIndicatorCalculator(cache_system)
    
    # 创建测试数据
    np.random.seed(42)
    test_data = pd.DataFrame({
        'close': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 1000)
    })
    
    # 定义测试指标计算函数
    def calculate_ma(data, period=20):
        time.sleep(0.01)  # 模拟计算时间
        return data['close'].rolling(window=period).mean()
    
    def calculate_rsi(data, period=14):
        time.sleep(0.02)  # 模拟计算时间
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    # 测试场景
    test_scenarios = [
        ('000001', 'MA', '20250101', calculate_ma, {'period': 20}),
        ('000001', 'RSI', '20250101', calculate_rsi, {'period': 14}),
        ('000002', 'MA', '20250101', calculate_ma, {'period': 20}),
        ('000001', 'MA', '20250102', calculate_ma, {'period': 20}),
    ]
    
    # 第一轮：无缓存计算
    print("第一轮：无缓存计算...")
    start_time = time.time()
    for stock_code, indicator, end_date, func, params in test_scenarios:
        result = calculator.calculate_with_cache(
            stock_code, indicator, end_date, test_data, func, params
        )
    first_round_time = time.time() - start_time
    
    # 第二轮：有缓存计算（重复相同请求）
    print("第二轮：有缓存计算...")
    start_time = time.time()
    for stock_code, indicator, end_date, func, params in test_scenarios:
        result = calculator.calculate_with_cache(
            stock_code, indicator, end_date, test_data, func, params
        )
    second_round_time = time.time() - start_time
    
    # 显示结果
    stats = cache_system.get_cache_stats()
    
    print(f"\n性能测试结果:")
    print(f"第一轮时间: {first_round_time:.4f}s")
    print(f"第二轮时间: {second_round_time:.4f}s")
    print(f"性能提升: {(first_round_time - second_round_time) / first_round_time * 100:.1f}%")
    
    print(f"\n缓存统计:")
    print(f"命中次数: {stats['hit_count']}")
    print(f"未命中次数: {stats['miss_count']}")
    print(f"命中率: {stats['hit_rate']:.1f}%")
    print(f"内存缓存大小: {stats['memory_cache_size']}")
    
    # 保存结果
    output_dir = "data/result/cache_performance"
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'performance': {
            'first_round_time': first_round_time,
            'second_round_time': second_round_time,
            'improvement_percentage': (first_round_time - second_round_time) / first_round_time * 100
        },
        'cache_stats': stats
    }
    
    results_path = os.path.join(output_dir, 'cache_performance_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n结果已保存到: {results_path}")
    print("="*60)


if __name__ == "__main__":
    benchmark_cache_performance()
