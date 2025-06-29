"""
智能日期管理器模块

统一处理系统中的日期管理需求，包括：
- 自动最新日期检测
- 交易日历支持
- 时间序列数据处理
- 多种日期格式转换
- 日期范围验证
- 历史数据回溯
"""

import datetime
import pandas as pd
import numpy as np
from typing import Union, Optional, List, Dict, Any, Tuple
from enum import Enum
import re
from functools import lru_cache
import warnings

from db.clickhouse_db import get_clickhouse_db
from utils.logger import get_logger
from utils.cache import MemoryCache

logger = get_logger(__name__)


class DateFormat(Enum):
    """支持的日期格式枚举"""
    YYYYMMDD = "%Y%m%d"
    YYYY_MM_DD = "%Y-%m-%d"
    YYYY_MM_DD_HH_MM_SS = "%Y-%m-%d %H:%M:%S"
    YYYYMMDD_HHMMSS = "%Y%m%d_%H%M%S"
    ISO_FORMAT = "%Y-%m-%dT%H:%M:%S"
    TIMESTAMP = "timestamp"


class DateRange(Enum):
    """预定义的日期范围枚举"""
    LAST_1_DAY = 1
    LAST_3_DAYS = 3
    LAST_7_DAYS = 7
    LAST_15_DAYS = 15
    LAST_30_DAYS = 30
    LAST_60_DAYS = 60
    LAST_90_DAYS = 90
    LAST_180_DAYS = 180
    LAST_1_YEAR = 365


class WeekDay(Enum):
    """星期枚举"""
    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4
    SATURDAY = 5
    SUNDAY = 6


class DateManager:
    """
    智能日期管理器
    
    提供统一的日期处理接口，支持：
    - 自动最新日期检测
    - 交易日历管理
    - 日期格式转换
    - 时间序列操作
    - 日期范围计算
    - 缓存优化
    """
    
    def __init__(self, cache_size: int = 1000, cache_ttl: int = 3600):
        """
        初始化日期管理器
        
        Args:
            cache_size: 缓存大小
            cache_ttl: 缓存过期时间（秒）
        """
        self.db = get_clickhouse_db()
        # 使用LRUCache代替MemoryCache，因为MemoryCache是单例模式
        from utils.cache import LRUCache
        self.cache = LRUCache(capacity=cache_size)
        self._cache_ttl = cache_ttl
        self._trading_calendar_cache = {}
        self._latest_date_cache = None
        self._latest_date_cache_time = None
        self._cache_expire_seconds = 300  # 5分钟缓存过期
        
        logger.info("智能日期管理器初始化完成")
    
    def get_current_datetime(self) -> datetime.datetime:
        """
        获取当前日期时间
        
        Returns:
            datetime.datetime: 当前日期时间
        """
        return datetime.datetime.now()
    
    def get_current_date(self, format_type: DateFormat = DateFormat.YYYY_MM_DD) -> str:
        """
        获取当前日期字符串
        
        Args:
            format_type: 日期格式类型
            
        Returns:
            str: 当前日期字符串
        """
        now = self.get_current_datetime()
        if format_type == DateFormat.TIMESTAMP:
            return str(int(now.timestamp()))
        return now.strftime(format_type.value)
    
    def parse_date(self, date_input: Union[str, datetime.datetime, pd.Timestamp, int, float], 
                   input_format: Optional[DateFormat] = None) -> datetime.datetime:
        """
        解析日期输入为datetime对象
        
        Args:
            date_input: 日期输入，支持多种类型
            input_format: 输入格式，如果为None则自动检测
            
        Returns:
            datetime.datetime: 解析后的datetime对象
            
        Raises:
            ValueError: 无法解析的日期格式
        """
        # 检查None值
        if date_input is None:
            raise ValueError("日期输入不能为None")
            
        # 检查不可哈希类型
        try:
            hash(date_input)
        except TypeError:
            raise ValueError(f"不支持的日期输入类型: {type(date_input)}")
        
        # 尝试从缓存获取
        cache_key = (str(date_input), input_format.name if input_format else None)
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        result = None
        
        # 使用类型名称字符串检查，以处理模拟对象
        if hasattr(date_input, '__class__') and 'datetime' in str(type(date_input)):
            result = date_input
        elif hasattr(date_input, 'to_pydatetime'):  # pandas Timestamp
            result = date_input.to_pydatetime()
        elif isinstance(date_input, (int, float)):
            # 时间戳处理
            if date_input > 1e10:  # 毫秒时间戳
                result = datetime.datetime.fromtimestamp(date_input / 1000)
            else:  # 秒时间戳
                result = datetime.datetime.fromtimestamp(date_input)
        elif isinstance(date_input, str):
            # 字符串日期解析
            if input_format:
                if input_format == DateFormat.TIMESTAMP:
                    result = datetime.datetime.fromtimestamp(float(date_input))
                else:
                    result = datetime.datetime.strptime(date_input, input_format.value)
            else:
                # 自动检测格式
                result = self._auto_parse_date_string(date_input)
        else:
            raise ValueError(f"不支持的日期输入类型: {type(date_input)}")
        
        # 缓存结果
        self.cache.set(cache_key, result)
        return result
    
    def _auto_parse_date_string(self, date_str: str) -> datetime.datetime:
        """
        自动检测并解析日期字符串
        
        Args:
            date_str: 日期字符串
            
        Returns:
            datetime.datetime: 解析后的datetime对象
        """
        # 移除空白字符
        date_str = date_str.strip()
        
        # 常见格式模式
        patterns = [
            (r'^\d{8}$', DateFormat.YYYYMMDD),  # 20240101
            (r'^\d{4}-\d{2}-\d{2}$', DateFormat.YYYY_MM_DD),  # 2024-01-01
            (r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}$', DateFormat.YYYY_MM_DD_HH_MM_SS),  # 2024-01-01 12:00:00
            (r'^\d{8}_\d{6}$', DateFormat.YYYYMMDD_HHMMSS),  # 20240101_120000
            (r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', DateFormat.ISO_FORMAT),  # 2024-01-01T12:00:00
        ]
        
        for pattern, format_type in patterns:
            if re.match(pattern, date_str):
                try:
                    return datetime.datetime.strptime(date_str, format_type.value)
                except ValueError:
                    continue
        
        # 尝试pandas解析
        try:
            return pd.to_datetime(date_str).to_pydatetime()
        except:
            pass
        
        raise ValueError(f"无法解析日期字符串: {date_str}")
    
    def format_date(self, date_input: Union[str, datetime.datetime, pd.Timestamp], 
                   output_format: DateFormat = DateFormat.YYYY_MM_DD,
                   input_format: Optional[DateFormat] = None) -> str:
        """
        格式化日期
        
        Args:
            date_input: 日期输入
            output_format: 输出格式
            input_format: 输入格式（用于字符串输入）
            
        Returns:
            str: 格式化后的日期字符串
        """
        dt = self.parse_date(date_input, input_format)
        
        if output_format == DateFormat.TIMESTAMP:
            return str(int(dt.timestamp()))
        
        return dt.strftime(output_format.value)
    
    def get_latest_trading_date(self, format_type: DateFormat = DateFormat.YYYY_MM_DD) -> str:
        """
        获取最近的交易日期
        
        Args:
            format_type: 返回的日期格式
            
        Returns:
            str: 最近交易日期字符串
        """
        cache_key = f"latest_trading_date_{format_type.name}"
        
        # 检查缓存
        if self._latest_date_cache and self._latest_date_cache_time:
            if (datetime.datetime.now() - self._latest_date_cache_time).seconds < self._cache_expire_seconds:
                cached_date = self.cache.get(cache_key)
                if cached_date:
                    return cached_date
        
        try:
            # 从数据库获取最新交易日期
            latest_date = self._get_latest_date_from_db()
            if latest_date:
                # 转换格式
                formatted_date = self.format_date(latest_date, format_type)
                
                # 更新缓存
                self._latest_date_cache = latest_date
                self._latest_date_cache_time = datetime.datetime.now()
                self.cache.set(cache_key, formatted_date)
                
                logger.info(f"获取到最新交易日期: {formatted_date}")
                return formatted_date
        except Exception as e:
            logger.error(f"从数据库获取最新交易日期失败: {e}")
        
        # 如果数据库查询失败，使用智能估算
        estimated_date = self._estimate_latest_trading_date()
        formatted_date = self.format_date(estimated_date, format_type)
        
        logger.warning(f"使用估算的最新交易日期: {formatted_date}")
        return formatted_date
    
    def _get_latest_date_from_db(self) -> Optional[datetime.datetime]:
        """
        从数据库获取最新交易日期
        
        Returns:
            Optional[datetime.datetime]: 最新交易日期，如果获取失败返回None
        """
        try:
            # 尝试从股票数据表获取最新日期
            sql = """
            SELECT MAX(date) as latest_date
            FROM stock_kline_daily
            WHERE date <= today()
            """
            result = self.db.query_df(sql)
            
            if not result.empty and not pd.isna(result['latest_date'].iloc[0]):
                latest_date = pd.to_datetime(result['latest_date'].iloc[0])
                return latest_date.to_pydatetime()
            
            # 如果股票数据表查询失败，尝试交易日历表
            sql = """
            SELECT MAX(date) as latest_date
            FROM trading_calendar
            WHERE is_open = 1 AND date <= today()
            """
            result = self.db.query_df(sql)
            
            if not result.empty and not pd.isna(result['latest_date'].iloc[0]):
                latest_date = pd.to_datetime(result['latest_date'].iloc[0])
                return latest_date.to_pydatetime()
                
        except Exception as e:
            logger.error(f"数据库查询最新日期失败: {e}")
        
        return None
    
    def _estimate_latest_trading_date(self) -> datetime.datetime:
        """
        智能估算最新交易日期
        
        Returns:
            datetime.datetime: 估算的最新交易日期
        """
        now = datetime.datetime.now()
        
        # 如果是周末，回退到上周五
        if now.weekday() == WeekDay.SATURDAY.value:  # 周六
            return now - datetime.timedelta(days=1)
        elif now.weekday() == WeekDay.SUNDAY.value:  # 周日
            return now - datetime.timedelta(days=2)
        else:  # 工作日
            # 如果当前时间在15点之前，使用前一个交易日
            if now.hour < 15:
                if now.weekday() == WeekDay.MONDAY.value:  # 周一
                    return now - datetime.timedelta(days=3)  # 上周五
                else:
                    return now - datetime.timedelta(days=1)  # 前一天
            else:
                return now  # 当天
    
    def get_previous_trading_dates(self, base_date: Union[str, datetime.datetime], 
                                 count: int = 1,
                                 format_type: DateFormat = DateFormat.YYYY_MM_DD) -> List[str]:
        """
        获取指定日期之前的N个交易日
        
        Args:
            base_date: 基准日期
            count: 需要获取的交易日数量
            format_type: 返回的日期格式
            
        Returns:
            List[str]: 前N个交易日列表，按时间倒序排列
        """
        base_dt = self.parse_date(base_date)
        trading_dates = []
        current_date = base_dt
        
        while len(trading_dates) < count:
            current_date -= datetime.timedelta(days=1)
            
            # 跳过周末
            if current_date.weekday() < 5:  # 周一到周五
                formatted_date = self.format_date(current_date, format_type)
                trading_dates.append(formatted_date)
        
        return trading_dates
    
    def get_next_trading_dates(self, base_date: Union[str, datetime.datetime], 
                              count: int = 1,
                              format_type: DateFormat = DateFormat.YYYY_MM_DD) -> List[str]:
        """
        获取指定日期之后的N个交易日
        
        Args:
            base_date: 基准日期
            count: 需要获取的交易日数量
            format_type: 返回的日期格式
            
        Returns:
            List[str]: 后N个交易日列表，按时间正序排列
        """
        base_dt = self.parse_date(base_date)
        trading_dates = []
        current_date = base_dt
        
        while len(trading_dates) < count:
            current_date += datetime.timedelta(days=1)
            
            # 跳过周末
            if current_date.weekday() < 5:  # 周一到周五
                formatted_date = self.format_date(current_date, format_type)
                trading_dates.append(formatted_date)
        
        return trading_dates
    
    def get_date_range(self, start_date: Union[str, datetime.datetime], 
                      end_date: Union[str, datetime.datetime],
                      trading_days_only: bool = True,
                      format_type: DateFormat = DateFormat.YYYY_MM_DD) -> List[str]:
        """
        生成日期范围列表
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            trading_days_only: 是否只包含交易日
            format_type: 返回的日期格式
            
        Returns:
            List[str]: 日期列表
        """
        start_dt = self.parse_date(start_date)
        end_dt = self.parse_date(end_date)
        
        if start_dt > end_dt:
            raise ValueError("开始日期不能晚于结束日期")
        
        date_list = []
        current_date = start_dt
        
        while current_date <= end_dt:
            # 如果只需要交易日，跳过周末
            if not trading_days_only or current_date.weekday() < 5:
                formatted_date = self.format_date(current_date, format_type)
                date_list.append(formatted_date)
            
            current_date += datetime.timedelta(days=1)
        
        return date_list
    
    def get_predefined_date_range(self, range_type: DateRange, 
                                 base_date: Optional[Union[str, datetime.datetime]] = None,
                                 trading_days_only: bool = True,
                                 format_type: DateFormat = DateFormat.YYYY_MM_DD) -> Tuple[str, List[str]]:
        """
        获取预定义的日期范围
        
        Args:
            range_type: 日期范围类型
            base_date: 基准日期，默认为当前日期
            trading_days_only: 是否只包含交易日
            format_type: 返回的日期格式
            
        Returns:
            Tuple[str, List[str]]: (结束日期, 日期列表)
        """
        if base_date is None:
            base_dt = self.get_current_datetime()
        else:
            base_dt = self.parse_date(base_date)
        
        # 计算开始日期
        days_back = range_type.value
        start_dt = base_dt - datetime.timedelta(days=days_back)
        
        # 生成日期范围
        date_list = self.get_date_range(start_dt, base_dt, trading_days_only, format_type)
        end_date = self.format_date(base_dt, format_type)
        
        return end_date, date_list
    
    def is_trading_day(self, date: Union[str, datetime.datetime]) -> bool:
        """
        判断是否为交易日
        
        Args:
            date: 日期
            
        Returns:
            bool: 是否为交易日
        """
        dt = self.parse_date(date)
        
        # 简单判断：周一到周五为交易日
        # 实际应用中可以查询交易日历表
        return dt.weekday() < 5
    
    def get_trading_days_between(self, start_date: Union[str, datetime.datetime], 
                                end_date: Union[str, datetime.datetime]) -> int:
        """
        计算两个日期之间的交易日数量
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            int: 交易日数量
        """
        date_list = self.get_date_range(start_date, end_date, trading_days_only=True)
        return len(date_list)
    
    def add_trading_days(self, base_date: Union[str, datetime.datetime], 
                        days: int,
                        format_type: DateFormat = DateFormat.YYYY_MM_DD) -> str:
        """
        在基准日期上增加指定的交易日数
        
        Args:
            base_date: 基准日期
            days: 要增加的交易日数（可以为负数）
            format_type: 返回的日期格式
            
        Returns:
            str: 计算后的日期
        """
        if days == 0:
            return self.format_date(base_date, format_type)
        
        if days > 0:
            # 向前推进
            dates = self.get_next_trading_dates(base_date, days, format_type)
            return dates[-1] if dates else self.format_date(base_date, format_type)
        else:
            # 向后回退
            dates = self.get_previous_trading_dates(base_date, abs(days), format_type)
            return dates[-1] if dates else self.format_date(base_date, format_type)
    
    def validate_date_range(self, start_date: Union[str, datetime.datetime], 
                           end_date: Union[str, datetime.datetime],
                           max_days: Optional[int] = None) -> bool:
        """
        验证日期范围的有效性
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            max_days: 最大天数限制
            
        Returns:
            bool: 日期范围是否有效
        """
        try:
            start_dt = self.parse_date(start_date)
            end_dt = self.parse_date(end_date)
            
            # 检查日期顺序
            if start_dt > end_dt:
                return False
            
            # 检查最大天数限制
            if max_days is not None:
                days_diff = (end_dt - start_dt).days
                if days_diff > max_days:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"日期范围验证失败: {e}")
            return False
    
    def get_time_series_dates(self, start_date: Union[str, datetime.datetime], 
                             end_date: Union[str, datetime.datetime],
                             frequency: str = 'D',
                             trading_days_only: bool = True) -> pd.DatetimeIndex:
        """
        生成时间序列日期索引
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            frequency: 频率（'D'=日, 'W'=周, 'M'=月等）
            trading_days_only: 是否只包含交易日
            
        Returns:
            pd.DatetimeIndex: 时间序列日期索引
        """
        start_dt = self.parse_date(start_date)
        end_dt = self.parse_date(end_date)
        
        # 生成基础日期范围
        if trading_days_only and frequency == 'D':
            # 对于日频率且只要交易日，使用自定义逻辑
            date_list = self.get_date_range(start_dt, end_dt, trading_days_only=True)
            return pd.to_datetime(date_list)
        else:
            # 使用pandas的日期范围生成
            return pd.date_range(start=start_dt, end=end_dt, freq=frequency)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        获取性能统计信息
        
        Returns:
            Dict[str, Any]: 性能统计数据
        """
        stats = self.cache.get_stats()
        return {
            'cache_size': stats.get('size', 0),
            'cache_max_size': stats.get('capacity', 0),
            'cache_hit_rate': stats.get('hit_rate', 0.0),
            'cache_hits': stats.get('hits', 0),  # 添加缓存命中次数
            'cache_misses': stats.get('misses', 0),  # 添加缓存未命中次数
            'total_operations': stats.get('hits', 0) + stats.get('misses', 0),  # 总操作次数
            'latest_date_cached': self._latest_date_cache is not None,
            'cache_expire_seconds': self._cache_expire_seconds
        }
    
    def clear_cache(self):
        """清除所有缓存"""
        self.cache.clear()
        self._latest_date_cache = None
        self._latest_date_cache_time = None
        self._trading_calendar_cache.clear()
        logger.info("日期管理器缓存已清除")
    
    def __str__(self) -> str:
        """字符串表示"""
        stats = self.cache.get_stats()
        return f"DateManager(cache_size={stats.get('size', 0)}, latest_date={self._latest_date_cache})"
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        stats = self.cache.get_stats()
        return (f"DateManager(cache_size={stats.get('size', 0)}, "
                f"cache_hit_rate={stats.get('hit_rate', 0.0):.2%}, "
                f"latest_date={self._latest_date_cache})") 