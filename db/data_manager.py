"""
数据管理器模块

负责从数据库获取数据，并提供高效的缓存机制
"""

import time
import threading
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
import pandas as pd
import json
import hashlib

from db.clickhouse_db import get_clickhouse_db
from utils.logger import get_logger
from utils.decorators import singleton, performance_monitor, safe_run, cache_result
from utils.exceptions import DataAccessError, DataNotFoundError, DataValidationError
from enums.period import Period
from models.stock_info import StockInfo  # 导入StockInfo类

logger = get_logger(__name__)


@singleton
class DataManager:
    """
    数据管理器，负责数据的获取和缓存
    
    使用单例模式确保系统中只有一个数据管理器实例
    """
    
    def __init__(self, cache_enabled=True, max_cache_size=1000, default_ttl=3600):
        """
        初始化数据管理器
        
        Args:
            cache_enabled: 是否启用缓存
            max_cache_size: 最大缓存条目数
            default_ttl: 默认缓存有效期（秒）
        """
        self.db = get_clickhouse_db()
        self.cache_enabled = cache_enabled
        self.default_ttl = default_ttl
        self.max_cache_size = max_cache_size
        
        # 缓存数据
        self.cache = {}
        self.cache_timestamps = {}
        self.cache_access_count = {}
        
        # 缓存锁，确保线程安全
        self.cache_lock = threading.RLock()
        
        # 缓存统计
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions = 0
        
        logger.info(f"数据管理器初始化完成，缓存{'启用' if cache_enabled else '禁用'}，"
                   f"最大缓存条目数: {max_cache_size}，默认缓存有效期: {default_ttl}秒")
    
    @performance_monitor(threshold=0.5)
    def get_stock_info(self, stock_code: Union[str, List[str]] = None, level: Union[str, Period] = None,
                       start_date: Optional[str] = None, end_date: Optional[str] = None,
                       filters: Optional[Dict[str, Any]] = None,
                       limit: Optional[int] = None, order_by: str = "date DESC",
                       group_by: Optional[str] = None, cache_ttl: Optional[int] = None) -> StockInfo:
        """
        获取股票数据并返回StockInfo对象
        
        Args:
            stock_code: 股票代码或股票代码列表，如果为None则查询所有股票
            level: K线周期，可以是 Period 枚举值或字符串 ('day', 'week', 'month', '15min', '30min', '60min')
            start_date: 开始日期，格式YYYY-MM-DD
            end_date: 结束日期，格式YYYY-MM-DD
            filters: 过滤条件字典，可包含 price, industry, market 等字段
            limit: 限制返回的记录数量
            order_by: 排序规则
            group_by: 分组字段
            cache_ttl: 缓存有效期（秒），None表示使用默认值
            
        Returns:
            StockInfo: 股票数据对象
            
        Raises:
            DataAccessError: 数据访问错误
        """
        try:
            # 构建缓存键
            cache_params = {
                "stock_code": stock_code,
                "level": level,
                "start_date": start_date,
                "end_date": end_date,
                "filters": filters,
                "limit": limit,
                "order_by": order_by,
                "group_by": group_by
            }
            cache_key = f"stock_info_{hashlib.md5(json.dumps(cache_params, sort_keys=True).encode()).hexdigest()}"
            
            # 获取缓存有效期
            ttl = cache_ttl if cache_ttl is not None else self.default_ttl
            
            # 检查缓存
            cached_data = self._get_from_cache(cache_key, ttl)
            if cached_data is not None:
                return cached_data
            
            # 使用ClickHouseDB的get_stock_info方法获取数据
            stock_info = self.db.get_stock_info(
                stock_code=stock_code,
                level=level,
                start_date=start_date,
                end_date=end_date,
                filters=filters,
                limit=limit,
                order_by=order_by,
                group_by=group_by
            )
            
            # 缓存结果
            self._set_cache(cache_key, stock_info, ttl)
            
            return stock_info
            
        except Exception as e:
            logger.error(f"获取股票数据出错: {e}")
            raise DataAccessError(f"获取股票数据失败: {e}")

    def get_kline_data(self, stock_code: str, start_date: Optional[str] = None,
                         end_date: Optional[str] = None, level: str = 'day',
                         **kwargs) -> StockInfo:
        """
        获取K线数据的兼容性接口（别名）。
        内部直接调用 get_stock_info 并返回其结果。
        """
        logger.warning("方法 get_kline_data 已被弃用，请尽快切换到 get_stock_info。")

        # 调用主方法并直接返回StockInfo对象
        return self.get_stock_info(
            stock_code=stock_code,
            start_date=start_date,
            end_date=end_date,
            level=level
        )
    
    @performance_monitor(threshold=0.5)
    def save_selection_result(self, result: pd.DataFrame, strategy_id: str, 
                             selection_date: str = None) -> bool:
        """
        保存选股结果
        
        Args:
            result: 选股结果DataFrame
            strategy_id: 策略ID
            selection_date: 选股日期，默认为当前日期
            
        Returns:
            保存成功返回True，否则返回False
        
        Raises:
            DataAccessError: 数据访问错误
            DataValidationError: 参数验证错误
        """
        try:
            # 参数验证
            if result is None or len(result) == 0:
                logger.warning(f"选股结果为空，不保存")
                return True
                
            if not strategy_id:
                raise DataValidationError("策略ID不能为空")
            
            # 使用ClickHouseDB的save_selection_result方法保存数据
            return self.db.save_selection_result(result, strategy_id, selection_date)
            
        except Exception as e:
            logger.error(f"保存选股结果出错: {e}")
            raise DataAccessError(f"保存选股结果失败: {str(e)}")
    
    @performance_monitor(threshold=0.5)
    def get_selection_history(self, strategy_id: Optional[str] = None, 
                             start_date: Optional[str] = None, 
                             end_date: Optional[str] = None,
                             limit: int = 100) -> pd.DataFrame:
        """
        获取选股历史记录
        
        Args:
            strategy_id: 策略ID，None表示所有策略
            start_date: 开始日期，None表示不限制
            end_date: 结束日期，None表示当前日期
            limit: 返回记录数限制
            
        Returns:
            选股历史记录DataFrame
        
        Raises:
            DataAccessError: 数据访问错误
        """
        try:
            # 使用ClickHouseDB的get_selection_history方法获取数据
            return self.db.get_selection_history(strategy_id, start_date, end_date, limit)
            
        except Exception as e:
            logger.error(f"获取选股历史记录出错: {e}")
            raise DataAccessError(f"获取选股历史记录失败: {str(e)}")
    
    @performance_monitor(threshold=0.5)
    def get_selection_result(self, strategy_id: str, selection_date: str) -> pd.DataFrame:
        """
        获取指定日期的选股结果
        
        Args:
            strategy_id: 策略ID
            selection_date: 选股日期
            
        Returns:
            选股结果DataFrame
        
        Raises:
            DataAccessError: 数据访问错误
            DataValidationError: 参数验证错误
        """
        try:
            # 参数验证
            if not strategy_id:
                raise DataValidationError("策略ID不能为空")
                
            if not selection_date:
                raise DataValidationError("选股日期不能为空")
            
            # 使用ClickHouseDB的get_selection_result方法获取数据
            return self.db.get_selection_result(strategy_id, selection_date)
            
        except Exception as e:
            if isinstance(e, DataValidationError):
                raise
            logger.error(f"获取选股结果出错: {e}")
            raise DataAccessError(f"获取选股结果失败: {str(e)}")
    
    def clear_cache(self, pattern: str = None):
        """
        清除缓存
        
        Args:
            pattern: 缓存键匹配模式，None表示清除所有缓存
        """
        with self.cache_lock:
            if pattern is None:
                old_size = len(self.cache)
                self.cache.clear()
                self.cache_timestamps.clear()
                self.cache_access_count.clear()
                logger.info(f"已清除所有缓存，共 {old_size} 项")
            else:
                keys_to_remove = [k for k in self.cache.keys() if pattern in k]
                for key in keys_to_remove:
                    self.cache.pop(key, None)
                    self.cache_timestamps.pop(key, None)
                    self.cache_access_count.pop(key, None)
                logger.info(f"已清除匹配 '{pattern}' 的缓存，共 {len(keys_to_remove)} 项")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            缓存统计信息字典
        """
        with self.cache_lock:
            stats = {
                'enabled': self.cache_enabled,
                'size': len(self.cache),
                'max_size': self.max_cache_size,
                'hits': self.cache_hits,
                'misses': self.cache_misses,
                'evictions': self.cache_evictions,
                'hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
                'types': {}
            }
            
            # 统计不同类型的缓存数量
            for key in self.cache.keys():
                type_name = key.split('_')[0]
                stats['types'][type_name] = stats['types'].get(type_name, 0) + 1
                
            return stats
    
    def transaction(self):
        """
        创建事务上下文
        
        Returns:
            事务上下文管理器
        """
        return self.db.transaction()
    
    def _get_from_cache(self, key: str, ttl: int) -> Optional[Any]:
        """
        从缓存中获取数据
        
        Args:
            key: 缓存键
            ttl: 缓存有效期（秒）
            
        Returns:
            缓存的数据，不存在或已过期返回None
        """
        if not self.cache_enabled:
            return None
            
        with self.cache_lock:
            # 检查缓存是否存在
            if key not in self.cache:
                self.cache_misses += 1
                return None
                
            # 检查缓存是否过期
            current_time = time.time()
            if current_time - self.cache_timestamps.get(key, 0) > ttl:
                # 缓存已过期，删除
                self.cache.pop(key, None)
                self.cache_timestamps.pop(key, None)
                self.cache_access_count.pop(key, None)
                self.cache_misses += 1
                return None
                
            # 更新访问计数
            self.cache_access_count[key] = self.cache_access_count.get(key, 0) + 1
            self.cache_hits += 1
            
            return self.cache[key]
    
    def _add_to_cache(self, key: str, value: Any):
        """
        添加数据到缓存
        
        Args:
            key: 缓存键
            value: 要缓存的数据
        """
        if not self.cache_enabled:
            return
            
        with self.cache_lock:
            # 检查缓存大小是否超限
            if len(self.cache) >= self.max_cache_size and key not in self.cache:
                # 缓存已满，删除最不常用的项
                self._evict_cache_item()
            
            # 更新缓存
            self.cache[key] = value
            self.cache_timestamps[key] = time.time()
            self.cache_access_count[key] = 1
    
    def _evict_cache_item(self):
        """驱逐最不常用的缓存项"""
        if not self.cache:
            return
            
        # 查找访问次数最少的项
        min_key = min(self.cache_access_count.items(), key=lambda x: x[1])[0]
        
        # 删除该项
        self.cache.pop(min_key, None)
        self.cache_timestamps.pop(min_key, None)
        self.cache_access_count.pop(min_key, None)
        
        self.cache_evictions += 1
        logger.debug(f"缓存驱逐: {min_key}")
    
    def _set_cache(self, key: str, value: Any, ttl: int) -> None:
        """
        设置缓存
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 缓存有效期（秒）
        """
        if self.cache_enabled:
            # 如果缓存已满，先清理
            if len(self.cache) >= self.max_cache_size:
                self._evict_cache_item()
                
            # 设置缓存
            self.cache[key] = value
            self.cache_timestamps[key] = time.time()
            self.cache_access_count[key] = 1
            logger.debug(f"缓存已设置: {key}")
    
    @performance_monitor(threshold=0.5)
    def get_stock_info_info(self, stock_codes: List[str]) -> pd.DataFrame:
        """
        获取股票基本信息 (兼容旧接口，内部调用get_stock_info)
        
        Args:
            stock_codes: 股票代码列表
            
        Returns:
            pd.DataFrame: 股票基本信息DataFrame
            
        Raises:
            DataAccessError: 数据访问错误
        """
        try:
            if not stock_codes:
                return pd.DataFrame(columns=['stock_code', 'stock_name', 'industry'])
            
            # 构建缓存键
            stock_codes_str = "_".join(sorted(stock_codes))
            cache_key = f"stock_info_basic_{hashlib.md5(stock_codes_str.encode()).hexdigest()}"
            
            # 获取缓存有效期
            ttl = self.default_ttl
            
            # 检查缓存
            cached_data = self._get_from_cache(cache_key, ttl)
            if cached_data is not None:
                return cached_data
            
            # 使用ClickHouseDB的get_stock_info方法获取数据
            result = self.db.get_stock_info(
                stock_code=stock_codes,
                level='日线',
                fields=["code as stock_code", "name as stock_name", "industry"],
                order_by="date DESC"
            )
            
            # 缓存结果
            self._set_cache(cache_key, result, ttl)
            
            return result
            
        except Exception as e:
            logger.error(f"获取股票基本信息出错: {e}")
            raise DataAccessError(f"获取股票基本信息失败: {e}")
            
    def get_industry_list(self) -> pd.DataFrame:
        """
        获取行业列表
        
        Returns:
            pd.DataFrame: 行业列表
        """
        try:
            # 使用ClickHouseDB的get_industry_list方法获取数据
            return self.db.get_industry_list()
        except Exception as e:
            logger.error(f"获取行业列表出错: {e}")
            raise DataAccessError(f"获取行业列表失败: {e}") 