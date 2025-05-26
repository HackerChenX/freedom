#!/usr/bin/python
# -*- coding: UTF-8 -*-

from enum import Enum, unique

@unique
class KlinePeriod(Enum):
    """
    K线周期枚举类
    用于统一管理K线周期的定义，替代原来的字符串表示方式
    """
    MIN_15 = "15分钟"
    MIN_30 = "30分钟"
    MIN_60 = "60分钟"
    DAILY = "日线"
    WEEKLY = "周线"
    MONTHLY = "月线"
    
    @classmethod
    def get_klt_code(cls, period):
        """
        获取对应周期的klt代码(用于efinance接口)
        """
        klt_map = {
            cls.MIN_15: 15,
            cls.MIN_30: 30,
            cls.MIN_60: 60,
            cls.DAILY: 101,
            cls.WEEKLY: 102,
            cls.MONTHLY: 103
        }
        return klt_map.get(period, 101)  # 默认返回日线代码
    
    def __str__(self):
        return self.value 