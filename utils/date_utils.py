"""
日期处理工具模块
"""

import datetime
from typing import Union, Optional

def get_today(format_str: str = "%Y%m%d") -> str:
    """
    获取今天的日期字符串
    
    Args:
        format_str: 日期格式化字符串
        
    Returns:
        str: 今天的日期字符串
    """
    return datetime.datetime.now().strftime(format_str)

def get_yesterday(format_str: str = "%Y%m%d") -> str:
    """
    获取昨天的日期字符串
    
    Args:
        format_str: 日期格式化字符串
        
    Returns:
        str: 昨天的日期字符串
    """
    return (datetime.datetime.now() - datetime.timedelta(days=1)).strftime(format_str)

def get_n_days_ago(n: int, format_str: str = "%Y%m%d") -> str:
    """
    获取n天前的日期字符串
    
    Args:
        n: 天数
        format_str: 日期格式化字符串
        
    Returns:
        str: n天前的日期字符串
    """
    return (datetime.datetime.now() - datetime.timedelta(days=n)).strftime(format_str)

def get_next_day(date_str: str, format_str: str = "%Y%m%d") -> str:
    """
    获取给定日期的下一天
    
    Args:
        date_str: 日期字符串
        format_str: 日期格式化字符串
        
    Returns:
        str: 下一天的日期字符串
    """
    date = datetime.datetime.strptime(date_str, format_str)
    next_day = date + datetime.timedelta(days=1)
    return next_day.strftime(format_str)

def parse_date(date_str: str, format_str: str = "%Y%m%d") -> datetime.datetime:
    """
    解析日期字符串为datetime对象
    
    Args:
        date_str: 日期字符串
        format_str: 日期格式化字符串
        
    Returns:
        datetime: 解析后的datetime对象
    """
    return datetime.datetime.strptime(date_str, format_str)

def format_date(date: Union[datetime.datetime, str], 
               src_format: str = "%Y%m%d", 
               target_format: str = "%Y%m%d") -> str:
    """
    格式化日期
    
    Args:
        date: 日期对象或字符串
        src_format: 源日期格式(当date为字符串时使用)
        target_format: 目标日期格式
        
    Returns:
        str: 格式化后的日期字符串
    """
    if isinstance(date, str):
        date = datetime.datetime.strptime(date, src_format)
    return date.strftime(target_format)

def is_trading_time() -> bool:
    """
    判断当前是否为交易时间(9:30-11:30, 13:00-15:00, 周一至周五)
    
    Returns:
        bool: 是否为交易时间
    """
    now = datetime.datetime.now()
    weekday = now.weekday()
    
    # 周末不是交易日
    if weekday >= 5:
        return False
    
    hour = now.hour
    minute = now.minute
    
    # 上午交易时间: 9:30-11:30
    if (hour == 9 and minute >= 30) or (hour == 10) or (hour == 11 and minute <= 30):
        return True
    
    # 下午交易时间: 13:00-15:00
    if (hour >= 13 and hour < 15):
        return True
    
    return False

def get_trading_day(include_today: bool = True) -> str:
    """
    获取最近的交易日期
    
    Args:
        include_today: 是否包含今天
        
    Returns:
        str: 交易日期字符串
    """
    now = datetime.datetime.now()
    
    # 如果当前时间晚于15点，且include_today为True，则返回今天
    if include_today and now.hour >= 15 and now.weekday() < 5:
        return now.strftime("%Y%m%d")
    
    # 否则返回前一个交易日
    days_to_subtract = 1
    if now.weekday() == 0:  # 周一
        days_to_subtract = 3  # 退回到上周五
    elif now.weekday() == 6:  # 周日
        days_to_subtract = 2  # 退回到上周五
    
    trading_day = now - datetime.timedelta(days=days_to_subtract)
    return trading_day.strftime("%Y%m%d")

def date_range(start_date: str, end_date: str, 
               format_str: str = "%Y%m%d") -> list:
    """
    生成日期范围列表
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
        format_str: 日期格式化字符串
        
    Returns:
        list: 日期字符串列表
    """
    start = datetime.datetime.strptime(start_date, format_str)
    end = datetime.datetime.strptime(end_date, format_str)
    
    date_list = []
    current = start
    
    while current <= end:
        date_list.append(current.strftime(format_str))
        current += datetime.timedelta(days=1)
    
    return date_list 