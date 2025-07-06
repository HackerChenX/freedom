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
import time

from utils.dependency_injection import get_service
from db.interfaces.data_access_interface import IData_access
from utils.logger import getLogger
from utils.cache import Memory_cache
from utils.decorators import exception_handler, performance_monitor

logger = getLogger(__name__)


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
    monday = 0
    tuesday = 1
    wednesday = 2
    thursday = 3
    friday = 4
    saturday = 5
    sunday = 6


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
    
    def __init___114(self):
        """初始化智能日期管理器"""
        # 使用依赖注入架构
        self.container = get_container()
        self.data_access = self.get_service(Data_access_interface)
        
        # 原有的初始化代码保持不变
        self.date_cache = {}
        self.trading_calendar = None
        self.last_cache_update = None
        self.cache_ttl = 3600  # 缓存1小时
        self.statistics = {
            'cache_hits': 0,
            'cache_misses': 0,
            'db_queries': 0,
            'calendar_updates': 0,
            'error_count': 0,
            'performance_metrics': {
                'avg_query_time': 0.0,
                'max_query_time': 0.0,
                'total_query_time': 0.0
            }
        }
        
        logger.info("智能日期管理器初始化完成，使用依赖注入架构")
    
    def get_current_datetime(self) -> datetime.datetime:
        """
        获取当前日期时间
        
        Returns:
            datetime.datetime: 当前日期时间
        """
        return datetime.datetime.now()
    
    def get_current_date(self, format_type: date_format = Date_format.YYYY_MM_DD) -> str:
        """
        获取当前日期字符串
        
        Args:
            format_type: 日期格式类型
            
        Returns:
            str: 当前日期字符串
        """
        now = self.get_current_datetime()
        if format_type == Date_format.TIMESTAMP:
            return str(int(now.timestamp()))
        return now.strftime(format_type.value)
    
    def parse_date(self, date_input: Union[str, datetime.datetime, pd.Timestamp, int, float], 
                   input_format: Optional[Date_format] = None) -> datetime.datetime:
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
        cached_result = self.date_cache.get(cache_key)
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
                if input_format == Date_format.TIMESTAMP:
                    result = datetime.datetime.fromtimestamp(float(date_input))
                else:
                    result = datetime.datetime.strptime(date_input, input_format.value)
            else:
                # 自动检测格式
                result = self._auto_parse_date_string(date_input)
        else:
            raise ValueError(f"不支持的日期输入类型: {type(date_input)}")
        
        # 缓存结果
        self.date_cache[cache_key] = result
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
                   output_format: date_format = Date_format.YYYY_MM_DD,
                   input_format: Optional[Date_format] = None) -> str:
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
        
        if output_format == Date_format.TIMESTAMP:
            return str(int(dt.timestamp()))
        
        return dt.strftime(output_format.value)
    
    def get_latest_trading_date(self, format_type: date_format = Date_format.YYYY_MM_DD) -> str:
        """
        获取最近的交易日期
        
        Args:
            format_type: 返回的日期格式
            
        Returns:
            str: 最近交易日期字符串
        """
        cache_key = f"latest_trading_date_{format_type.name}"
        
        # 检查缓存
        if self.trading_calendar and self.last_cache_update:
            if (datetime.datetime.now() - self.last_cache_update).seconds < self.cache_ttl:
                cached_date = self.date_cache.get(cache_key)
                if cached_date:
                    return cached_date
        
        try:
            # 从数据库获取最新交易日期
            calendar_data = self._fetch_trading_calendar_from_db_Date_Manager_Date_Manager_1_datemanager()
            if not calendar_data.empty:
                latest_date = pd.to_datetime(calendar_data['date'].iloc[0])
                formatted_date = self.format_date(latest_date, format_type)
                
                # 更新缓存
                self.trading_calendar = calendar_data
                self.last_cache_update = datetime.datetime.now()
                self.date_cache[cache_key] = formatted_date
                
                logger.info(f"获取到最新交易日期: {formatted_date}")
                return formatted_date
        except Exception as e:
            logger.error(f"从数据库获取最新交易日期失败: {e}")
        
        # 如果数据库查询失败，使用智能估算
        estimated_date = self._estimate_latest_trading_date()
        formatted_date = self.format_date(estimated_date, format_type)
        
        logger.warning(f"使用估算的最新交易日期: {formatted_date}")
        return formatted_date
    
    def _fetch_trading_calendar_from_db_Date_Manager_Date_Manager_1_datemanager(self) -> pd.DataFrame:
        """从数据库获取交易日历"""
        try:
            start_time = time.time()
            
            # 使用依赖注入的数据访问接口
            calendar_data = self.data_access.get_trading_calendar()
            
            query_time = time.time() - start_time
            
            # 更新统计信息
            self.statistics['db_queries'] += 1
            self.statistics['calendar_updates'] += 1
            self.statistics['performance_metrics']['total_query_time'] += query_time
            self.statistics['performance_metrics']['max_query_time'] = max(
                self.statistics['performance_metrics']['max_query_time'], query_time
            )
            
            if not calendar_data.empty:
                logger.info(f"成功从数据库获取交易日历，包含 {len(calendar_data)} 条记录")
                return calendar_data
            else:
                logger.warning("数据库返回空的交易日历")
                self.statistics['error_count'] += 1
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"从数据库获取交易日历失败: {e}")
            self.statistics['error_count'] += 1
            raise
    
    def _estimate_latest_trading_date(self) -> datetime.datetime:
        """
        智能估算最新交易日期
        
        Returns:
            datetime.datetime: 估算的最新交易日期
        """
        now = datetime.datetime.now()
        
        # 如果是周末，回退到上周五
        if now.weekday() == Week_day.SATURDAY.value:  # 周六
            return now - datetime.timedelta(days=1)
        elif now.weekday() == Week_day.SUNDAY.value:  # 周日
            return now - datetime.timedelta(days=2)
        else:  # 工作日
            # 如果当前时间在15点之前，使用前一个交易日
            if now.hour < 15:
                if now.weekday() == Week_day.MONDAY.value:  # 周一
                    return now - datetime.timedelta(days=3)  # 上周五
                else:
                    return now - datetime.timedelta(days=1)  # 前一天
            else:
                return now  # 当天
    
    def get_previous_trading_dates(self, base_date: Union[str, datetime.datetime], 
                                 count: int = 1,
                                 format_type: date_format = Date_format.YYYY_MM_DD) -> List[str]:
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
                              format_type: date_format = Date_format.YYYY_MM_DD) -> List[str]:
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
                      format_type: date_format = Date_format.YYYY_MM_DD) -> List[str]:
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
    
    def get_predefined_date_range(self, range_type: Date_range, 
                                 base_date: Optional[Union[str, datetime.datetime]] = None,
                                 trading_days_only: bool = True,
                                 format_type: date_format = Date_format.YYYY_MM_DD) -> Tuple[str, List[str]]:
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
                        format_type: date_format = Date_format.YYYY_MM_DD) -> str:
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
                             trading_days_only: bool = True) -> pd.Datetime_index:
        """
        生成时间序列日期索引
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            frequency: 频率（'D'=日, 'W'=周, 'M'=月等）
            trading_days_only: 是否只包含交易日
            
        Returns:
            pd.Datetime_index: 时间序列日期索引
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
    
    def get_performance_stats_Manager(self) -> Dict[str, Any]:
        """
        获取性能统计信息
        
        Returns:
            Dict[str, Any]: 性能统计数据
        """
        stats = self.date_cache.copy()
        stats['cache_size'] = len(self.date_cache)
        stats['cache_hit_rate'] = (stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses']) * 100) if (stats['cache_hits'] + stats['cache_misses']) > 0 else 0
        stats['total_operations'] = stats['cache_hits'] + stats['cache_misses']
        stats['latest_date_cached'] = self.trading_calendar is not None
        stats['cache_expire_seconds'] = self.cache_ttl
        return stats
    
    def clear_cache_Manager(self):
        """清除所有缓存"""
        self.date_cache.clear()
        self.trading_calendar = None
        self.last_cache_update = None
        logger.info("日期管理器缓存已清除")
    
    def __str__(self) -> str:
        """字符串表示"""
        stats = self.date_cache.copy()
        stats['cache_size'] = len(self.date_cache)
        stats['cache_hit_rate'] = (stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses']) * 100) if (stats['cache_hits'] + stats['cache_misses']) > 0 else 0
        stats['total_operations'] = stats['cache_hits'] + stats['cache_misses']
        stats['latest_date_cached'] = self.trading_calendar is not None
        stats['cache_expire_seconds'] = self.cache_ttl
        return f"DateManager(cache_size={stats['cache_size']}, latest_date={self.trading_calendar['date'].iloc[0] if self.trading_calendar is not None else None})"
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        stats = self.date_cache.copy()
        stats['cache_size'] = len(self.date_cache)
        stats['cache_hit_rate'] = (stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses']) * 100) if (stats['cache_hits'] + stats['cache_misses']) > 0 else 0
        stats['total_operations'] = stats['cache_hits'] + stats['cache_misses']
        stats['latest_date_cached'] = self.trading_calendar is not None
        stats['cache_expire_seconds'] = self.cache_ttl
        return (f"DateManager(cache_size={stats['cache_size']}, "
                f"cache_hit_rate={stats['cache_hit_rate']:.2%}, "
                f"latest_date={self.trading_calendar['date'].iloc[0] if self.trading_calendar is not None else None})")
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=2.0)
    def get_detailed_statistics(self) -> Dict[str, Any]:
        """
        获取详细统计信息
        
        Returns:
            Dict[str, Any]: 详细统计数据
        """
        try:
            current_stats = self.statistics.copy()
            
            # 计算缓存命中率
            total_requests = current_stats['cache_hits'] + current_stats['cache_misses']
            cache_hit_rate = (current_stats['cache_hits'] / total_requests * 100) if total_requests > 0 else 0
            
            # 计算平均查询时间
            if current_stats['db_queries'] > 0:
                current_stats['performance_metrics']['avg_query_time'] = (
                    current_stats['performance_metrics']['total_query_time'] / current_stats['db_queries']
                )
            
            # 添加计算字段
            current_stats['cache_hit_rate'] = cache_hit_rate
            current_stats['total_requests'] = total_requests
            current_stats['cache_size'] = len(self.date_cache)
            current_stats['last_update'] = self.last_cache_update.isoformat() if self.last_cache_update else None
            
            # 系统健康状态
            if current_stats['error_count'] == 0 and cache_hit_rate > 80:
                current_stats['health_status'] = 'excellent'
            elif current_stats['error_count'] < 5 and cache_hit_rate > 60:
                current_stats['health_status'] = 'good'
            elif current_stats['error_count'] < 10 and cache_hit_rate > 40:
                current_stats['health_status'] = 'fair'
            else:
                current_stats['health_status'] = 'poor'
            
            return current_stats
            
        except Exception as e:
            logger.error(f"获取详细统计信息失败: {e}")
            return self.statistics.copy()
    
    @exception_handler(reraise=True)
    def reset_statistics(self) -> None:
        """重置统计信息"""
        try:
            self.statistics = {
                'cache_hits': 0,
                'cache_misses': 0,
                'db_queries': 0,
                'calendar_updates': 0,
                'error_count': 0,
                'performance_metrics': {
                    'avg_query_time': 0.0,
                    'max_query_time': 0.0,
                    'total_query_time': 0.0
                }
            }
            logger.info("统计信息已重置")
            
        except Exception as e:
            logger.error(f"重置统计信息失败: {e}")
    
    @exception_handler(reraise=True)
    def optimize_cache(self) -> Dict[str, Any]:
        """
        优化缓存性能
        
        Returns:
            Dict[str, Any]: 优化结果
        """
        try:
            logger.info("开始优化缓存")
            
            original_size = len(self.date_cache)
            removed_count = 0
            
            # 移除过期的缓存项
            current_time = datetime.now()
            expired_keys = []
            
            for key, (value, timestamp) in self.date_cache.items():
                if (current_time - timestamp).seconds > self.cache_ttl:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.date_cache[key]
                removed_count += 1
            
            # 如果缓存仍然很大，移除最老的项目
            max_cache_size = 1000
            if len(self.date_cache) > max_cache_size:
                # 按时间戳排序，移除最老的项目
                sorted_items = sorted(
                    self.date_cache.items(),
                    key=lambda x: x[1][1]  # 按时间戳排序
                )
                
                items_to_remove = len(self.date_cache) - max_cache_size
                for i in range(items_to_remove):
                    key = sorted_items[i][0]
                    del self.date_cache[key]
                    removed_count += 1
            
            optimization_result = {
                'original_size': original_size,
                'final_size': len(self.date_cache),
                'removed_count': removed_count,
                'optimization_time': datetime.now().isoformat()
            }
            
            logger.info(f"缓存优化完成，移除了 {removed_count} 个项目")
            return optimization_result
            
        except Exception as e:
            logger.error(f"优化缓存失败: {e}")
            return {'error': str(e)} 