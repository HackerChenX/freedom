#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
统一周期枚举模块

提供统一的周期定义，用于K线数据查询和策略配置
合并了原有的 KlinePeriod 和 PeriodType 的功能
"""

from enum import Enum, unique
from typing import Dict, Any, Optional, Union


@unique
class Period(Enum):
    """
    统一周期枚举
    
    用于K线数据查询和策略配置中的周期定义
    内部值使用标准化格式，便于数据库查询和策略配置
    """
    # 分钟级别周期
    MIN_5 = "MIN5"         # 5分钟线
    MIN_15 = "MIN15"       # 15分钟线
    MIN_30 = "MIN30"       # 30分钟线
    MIN_60 = "MIN60"       # 60分钟线
    
    # 日级别周期
    DAILY = "DAILY"        # 日线
    WEEKLY = "WEEKLY"      # 周线
    MONTHLY = "MONTHLY"    # 月线
    
    @classmethod
    def get_description(cls, period) -> str:
        """
        获取周期类型的人类可读描述
        
        Args:
            period: 周期枚举值
            
        Returns:
            str: 周期类型的中文描述
        """
        descriptions = {
            cls.MIN_5: "5分钟线",
            cls.MIN_15: "15分钟线",
            cls.MIN_30: "30分钟线",
            cls.MIN_60: "60分钟线",
            cls.DAILY: "日线",
            cls.WEEKLY: "周线",
            cls.MONTHLY: "月线"
        }
        return descriptions.get(period, "未知周期")
    
    @classmethod
    def get_klt_code(cls, period) -> int:
        """
        获取对应周期的klt代码(用于efinance接口)
        
        Args:
            period: 周期枚举值
            
        Returns:
            int: efinance接口使用的klt代码
        """
        klt_map = {
            cls.MIN_5: 5,
            cls.MIN_15: 15,
            cls.MIN_30: 30,
            cls.MIN_60: 60,
            cls.DAILY: 101,
            cls.WEEKLY: 102,
            cls.MONTHLY: 103
        }
        return klt_map.get(period, 101)  # 默认返回日线代码
    
    @classmethod
    def get_table_name(cls, period) -> str:
        """
        获取对应周期的数据库表名
        
        Args:
            period: 周期枚举值
            
        Returns:
            str: 数据库表名
        """
        table_map = {
            cls.MIN_5: "kline_5min",
            cls.MIN_15: "kline_15min",
            cls.MIN_30: "kline_30min",
            cls.MIN_60: "kline_60min",
            cls.DAILY: "kline_daily",
            cls.WEEKLY: "kline_weekly",
            cls.MONTHLY: "kline_monthly"
        }
        return table_map.get(period, "kline_daily")  # 默认返回日线表名
    
    @classmethod
    def from_string(cls, period_str: str) -> 'Period':
        """
        从字符串创建周期枚举
        
        支持多种字符串格式，包括枚举名称、中文描述和简写形式
        
        Args:
            period_str: 周期字符串
            
        Returns:
            Period: 对应的周期枚举值
        """
        period_str = period_str.upper().strip()
        
        # 直接匹配枚举名称
        try:
            return cls[period_str]
        except (KeyError, ValueError):
            pass
        
        # 匹配值
        for period in cls:
            if period.value == period_str:
                return period
        
        # 匹配简写和描述
        mapping = {
            # 中文描述
            "5分钟": cls.MIN_5,
            "5分钟线": cls.MIN_5,
            "15分钟": cls.MIN_15,
            "15分钟线": cls.MIN_15,
            "30分钟": cls.MIN_30,
            "30分钟线": cls.MIN_30,
            "60分钟": cls.MIN_60,
            "60分钟线": cls.MIN_60,
            "1小时": cls.MIN_60,
            "日": cls.DAILY,
            "日线": cls.DAILY,
            "周": cls.WEEKLY,
            "周线": cls.WEEKLY,
            "月": cls.MONTHLY,
            "月线": cls.MONTHLY,
            
            # 简写
            "5": cls.MIN_5,
            "15": cls.MIN_15,
            "30": cls.MIN_30,
            "60": cls.MIN_60,
            "D": cls.DAILY,
            "DAY": cls.DAILY,
            "W": cls.WEEKLY,
            "WEEK": cls.WEEKLY,
            "M": cls.MONTHLY,
            "MONTH": cls.MONTHLY,
            
            # 原 KlinePeriod 兼容
            "MIN_5": cls.MIN_5,
            "MIN_15": cls.MIN_15,
            "MIN_30": cls.MIN_30,
            "MIN_60": cls.MIN_60,
            
            # 原 PeriodType 兼容
            "MIN5": cls.MIN_5,
            "MIN15": cls.MIN_15,
            "MIN30": cls.MIN_30,
            "MIN60": cls.MIN_60
        }
        
        return mapping.get(period_str, cls.DAILY)  # 默认返回日线
    
    @staticmethod
    def is_valid_period(period_str: str) -> bool:
        """
        检查字符串是否是有效的周期
        
        Args:
            period_str: 周期字符串
            
        Returns:
            bool: 是否有效
        """
        try:
            Period.from_string(period_str)
            return True
        except Exception:
            return False
    
    @staticmethod
    def get_all_periods() -> list:
        """
        获取所有支持的周期列表
        
        Returns:
            list: 所有周期枚举值列表
        """
        return list(Period)
    
    @staticmethod
    def get_all_period_names() -> list:
        """
        获取所有支持的周期名称列表
        
        Returns:
            list: 所有周期名称列表
        """
        return [p.name for p in Period]
    
    @staticmethod
    def get_all_period_values() -> list:
        """
        获取所有支持的周期值列表
        
        Returns:
            list: 所有周期值列表
        """
        return [p.value for p in Period]
    
    @staticmethod
    def get_all_period_descriptions() -> list:
        """
        获取所有支持的周期描述列表
        
        Returns:
            list: 所有周期描述列表
        """
        return [Period.get_description(p) for p in Period]
    
    def __str__(self) -> str:
        """
        返回周期的字符串表示
        
        Returns:
            str: 周期的描述
        """
        return self.get_description(self)
    
    def __repr__(self) -> str:
        """
        返回周期的程序表示
        
        Returns:
            str: 周期的程序表示
        """
        return f"Period.{self.name}"


# 为了向后兼容，提供别名
KlinePeriod = Period
PeriodType = Period 