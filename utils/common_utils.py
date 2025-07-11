"""
公共功能提取模块

提取系统中的公共功能，包括：
1. 通用数据处理工具
2. 通用日期时间工具
3. 通用验证工具
4. 通用格式化工具
5. 通用缓存工具
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
import re
from functools import wraps
from utils.logger import getLogger

logger = getLogger(__name__)


class DataProcessor:
    """通用数据处理工具类"""
    
    @staticmethod
    def clean_dataframe(df: pd.DataFrame, 
                       drop_na: bool = True,
                       fill_na_value: Any = None,
                       deduplicate: bool = True) -> pd.DataFrame:
        """
        清理DataFrame数据
        
        Args:
            df: 输入DataFrame
            drop_na: 是否删除空值行
            fill_na_value: 空值填充值
            deduplicate: 是否去重
            
        Returns:
            pd.DataFrame: 清理后的DataFrame
        """
        if df.empty:
            return df
        
        result = df.copy()
        
        # 处理空值
        if drop_na:
            result = result.dropna()
        elif fill_na_value is not None:
            result = result.fillna(fill_na_value)
        
        # 去重
        if deduplicate:
            result = result.drop_duplicates()
        
        return result
    
    @staticmethod
    def normalize_stock_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        标准化股票数据格式
        
        Args:
            df: 输入股票数据
            
        Returns:
            pd.DataFrame: 标准化后的数据
        """
        if df.empty:
            return df
        
        result = df.copy()
        
        # 确保必要的列存在
        required_columns = ['date', 'code', 'close']
        missing_columns = [col for col in required_columns if col not in result.columns]
        
        if missing_columns:
            logger.warning(f"缺少必要列: {missing_columns}")
        
        # 标准化日期列
        if 'date' in result.columns:
            result['date'] = pd.to_datetime(result['date'])
        
        # 标准化数值列
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in result.columns:
                result[col] = pd.to_numeric(result[col], errors='coerce')
        
        # 排序
        if 'date' in result.columns:
            result = result.sort_values('date')
        
        return result
    
    @staticmethod
    def calculate_returns(df: pd.DataFrame, 
                         price_col: str = 'close',
                         periods: int = 1) -> pd.Series:
        """
        计算收益率
        
        Args:
            df: 输入数据
            price_col: 价格列名
            periods: 计算周期
            
        Returns:
            pd.Series: 收益率序列
        """
        if df.empty or price_col not in df.columns:
            return pd.Series(dtype=float)
        
        prices = df[price_col]
        returns = prices.pct_change(periods=periods)
        
        return returns
    
    @staticmethod
    def calculate_rolling_stats(df: pd.DataFrame,
                              column: str,
                              window: int,
                              stats: List[str] = ['mean', 'std']) -> pd.DataFrame:
        """
        计算滚动统计指标
        
        Args:
            df: 输入数据
            column: 计算列
            window: 滚动窗口
            stats: 统计指标列表
            
        Returns:
            pd.DataFrame: 包含统计指标的DataFrame
        """
        if df.empty or column not in df.columns:
            return pd.DataFrame()
        
        result = df.copy()
        rolling = result[column].rolling(window=window)
        
        for stat in stats:
            if hasattr(rolling, stat):
                result[f'{column}_{stat}_{window}'] = getattr(rolling, stat)()
        
        return result


class DateTimeUtils:
    """日期时间工具类"""
    
    @staticmethod
    def get_trading_days(start_date: str, end_date: str) -> List[str]:
        """
        获取交易日列表（简化版，实际应该考虑节假日）
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            List[str]: 交易日列表
        """
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # 生成工作日（周一到周五）
        business_days = pd.bdate_range(start=start, end=end)
        
        return [d.strftime('%Y-%m-%d') for d in business_days]
    
    @staticmethod
    def get_previous_trading_day(date: str, days: int = 1) -> str:
        """
        获取前N个交易日
        
        Args:
            date: 基准日期
            days: 向前天数
            
        Returns:
            str: 前N个交易日
        """
        base_date = pd.to_datetime(date)
        previous_date = base_date - pd.Timedelta(days=days * 1.5)  # 考虑周末
        
        # 找到最近的工作日
        while previous_date.weekday() >= 5:  # 周末
            previous_date -= pd.Timedelta(days=1)
        
        return previous_date.strftime('%Y-%m-%d')
    
    @staticmethod
    def get_period_start_date(end_date: str, period: str) -> str:
        """
        根据周期获取开始日期
        
        Args:
            end_date: 结束日期
            period: 周期 ('1d', '1w', '1M', '3M', '6M', '1Y')
            
        Returns:
            str: 开始日期
        """
        end = pd.to_datetime(end_date)
        
        if period == '1d':
            start = end - pd.Timedelta(days=1)
        elif period == '1w':
            start = end - pd.Timedelta(weeks=1)
        elif period == '1M':
            start = end - pd.DateOffset(months=1)
        elif period == '3M':
            start = end - pd.DateOffset(months=3)
        elif period == '6M':
            start = end - pd.DateOffset(months=6)
        elif period == '1Y':
            start = end - pd.DateOffset(years=1)
        else:
            start = end - pd.DateOffset(months=3)  # 默认3个月
        
        return start.strftime('%Y-%m-%d')
    
    @staticmethod
    def is_trading_day(date: str) -> bool:
        """
        判断是否为交易日
        
        Args:
            date: 日期
            
        Returns:
            bool: 是否为交易日
        """
        dt = pd.to_datetime(date)
        return dt.weekday() < 5  # 简化版，实际应该考虑节假日


class ValidationUtils:
    """验证工具类"""
    
    @staticmethod
    def validate_stock_code(code: str) -> bool:
        """
        验证股票代码格式
        
        Args:
            code: 股票代码
            
        Returns:
            bool: 是否有效
        """
        if not code or not isinstance(code, str):
            return False
        
        # 简单的股票代码格式验证
        patterns = [
            r'^\d{6}$',  # 6位数字
            r'^\d{6}\.[A-Z]{2}$',  # 6位数字.交易所
        ]
        
        return any(re.match(pattern, code) for pattern in patterns)
    
    @staticmethod
    def validate_date_format(date_str: str) -> bool:
        """
        验证日期格式
        
        Args:
            date_str: 日期字符串
            
        Returns:
            bool: 是否有效
        """
        if not date_str or not isinstance(date_str, str):
            return False
        
        try:
            pd.to_datetime(date_str)
            return True
        except:
            return False
    
    @staticmethod
    def validate_numeric_range(value: float, 
                             min_val: Optional[float] = None,
                             max_val: Optional[float] = None) -> bool:
        """
        验证数值范围
        
        Args:
            value: 数值
            min_val: 最小值
            max_val: 最大值
            
        Returns:
            bool: 是否在范围内
        """
        if not isinstance(value, (int, float)) or np.isnan(value):
            return False
        
        if min_val is not None and value < min_val:
            return False
        
        if max_val is not None and value > max_val:
            return False
        
        return True
    
    @staticmethod
    def validate_dataframe_schema(df: pd.DataFrame, 
                                required_columns: List[str]) -> Tuple[bool, List[str]]:
        """
        验证DataFrame结构
        
        Args:
            df: 输入DataFrame
            required_columns: 必需列
            
        Returns:
            Tuple[bool, List[str]]: (是否有效, 缺失列列表)
        """
        if df.empty:
            return False, required_columns
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        return len(missing_columns) == 0, missing_columns


class FormatUtils:
    """格式化工具类"""
    
    @staticmethod
    def format_number(value: float, 
                     decimal_places: int = 2,
                     percentage: bool = False) -> str:
        """
        格式化数字
        
        Args:
            value: 数值
            decimal_places: 小数位数
            percentage: 是否为百分比
            
        Returns:
            str: 格式化后的字符串
        """
        if np.isnan(value):
            return "N/A"
        
        if percentage:
            return f"{value:.{decimal_places}%}"
        else:
            return f"{value:.{decimal_places}f}"
    
    @staticmethod
    def format_large_number(value: float) -> str:
        """
        格式化大数字（添加千分位分隔符）
        
        Args:
            value: 数值
            
        Returns:
            str: 格式化后的字符串
        """
        if np.isnan(value):
            return "N/A"
        
        return f"{value:,.0f}"
    
    @staticmethod
    def format_currency(value: float, currency: str = "¥") -> str:
        """
        格式化货币
        
        Args:
            value: 数值
            currency: 货币符号
            
        Returns:
            str: 格式化后的字符串
        """
        if np.isnan(value):
            return "N/A"
        
        return f"{currency}{value:,.2f}"
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 50) -> str:
        """
        截断文本
        
        Args:
            text: 文本
            max_length: 最大长度
            
        Returns:
            str: 截断后的文本
        """
        if not text:
            return ""
        
        if len(text) <= max_length:
            return text
        
        return text[:max_length-3] + "..."


class CacheUtils:
    """缓存工具类"""
    
    _cache = {}
    
    @classmethod
    def get(cls, key: str) -> Any:
        """获取缓存值"""
        return cls._cache.get(key)
    
    @classmethod
    def set(cls, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """设置缓存值"""
        cls._cache[key] = {
            'value': value,
            'timestamp': datetime.now(),
            'ttl': ttl
        }
    
    @classmethod
    def has(cls, key: str) -> bool:
        """检查缓存是否存在且有效"""
        if key not in cls._cache:
            return False
        
        cache_item = cls._cache[key]
        if cache_item['ttl'] is not None:
            elapsed = (datetime.now() - cache_item['timestamp']).total_seconds()
            if elapsed > cache_item['ttl']:
                del cls._cache[key]
                return False
        
        return True
    
    @classmethod
    def clear(cls) -> None:
        """清空缓存"""
        cls._cache.clear()
    
    @classmethod
    def cache_with_ttl(cls, ttl: int = 300):
        """
        缓存装饰器
        
        Args:
            ttl: 过期时间（秒）
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # 生成缓存键
                cache_key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
                
                # 检查缓存
                if cls.has(cache_key):
                    return cls.get(cache_key)['value']
                
                # 执行函数并缓存结果
                result = func(*args, **kwargs)
                cls.set(cache_key, result, ttl)
                
                return result
            return wrapper
        return decorator


# 便捷函数
def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """安全除法"""
    try:
        if b == 0:
            return default
        return a / b
    except:
        return default


def safe_log(value: float, default: float = 0.0) -> float:
    """安全对数"""
    try:
        if value <= 0:
            return default
        return np.log(value)
    except:
        return default


def batch_process(items: List[Any], 
                 batch_size: int = 100,
                 process_func: callable = None) -> List[Any]:
    """
    批量处理数据
    
    Args:
        items: 待处理项目
        batch_size: 批次大小
        process_func: 处理函数
        
    Returns:
        List[Any]: 处理结果
    """
    if not process_func:
        return items
    
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = process_func(batch)
        results.extend(batch_results)
    
    return results


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """合并多个字典"""
    result = {}
    for d in dicts:
        if d:
            result.update(d)
    return result


def flatten_list(nested_list: List[List[Any]]) -> List[Any]:
    """扁平化嵌套列表"""
    return [item for sublist in nested_list for item in sublist]


def remove_duplicates(items: List[Any], key_func: callable = None) -> List[Any]:
    """
    去重，支持自定义键函数
    
    Args:
        items: 项目列表
        key_func: 键函数
        
    Returns:
        List[Any]: 去重后的列表
    """
    if not key_func:
        return list(set(items))
    
    seen = set()
    result = []
    for item in items:
        key = key_func(item)
        if key not in seen:
            seen.add(key)
            result.append(item)
    
    return result