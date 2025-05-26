"""
周期类型枚举模块

用于定义策略中使用的时间周期类型
"""

from enum import Enum, unique


@unique
class PeriodType(Enum):
    """
    周期类型枚举
    
    用于统一管理策略中使用的时间周期定义
    """
    DAILY = "DAILY"       # 日线
    WEEKLY = "WEEKLY"     # 周线
    MONTHLY = "MONTHLY"   # 月线
    MIN60 = "MIN60"       # 60分钟线
    MIN30 = "MIN30"       # 30分钟线
    MIN15 = "MIN15"       # 15分钟线
    MIN5 = "MIN5"         # 5分钟线
    
    @classmethod
    def get_description(cls, period_type):
        """
        获取周期类型的描述
        
        Args:
            period_type: 周期类型枚举值
            
        Returns:
            str: 周期类型描述
        """
        descriptions = {
            cls.DAILY: "日线",
            cls.WEEKLY: "周线",
            cls.MONTHLY: "月线",
            cls.MIN60: "60分钟线",
            cls.MIN30: "30分钟线",
            cls.MIN15: "15分钟线",
            cls.MIN5: "5分钟线"
        }
        return descriptions.get(period_type, "未知周期")
    
    @classmethod
    def from_kline_period(cls, kline_period):
        """
        从KlinePeriod转换为PeriodType
        
        Args:
            kline_period: KlinePeriod枚举值
            
        Returns:
            PeriodType: 对应的PeriodType枚举值
        """
        from enums.kline_period import KlinePeriod
        
        mapping = {
            KlinePeriod.DAILY: cls.DAILY,
            KlinePeriod.WEEKLY: cls.WEEKLY,
            KlinePeriod.MONTHLY: cls.MONTHLY,
            KlinePeriod.MIN_60: cls.MIN60,
            KlinePeriod.MIN_30: cls.MIN30,
            KlinePeriod.MIN_15: cls.MIN15
        }
        
        return mapping.get(kline_period, cls.DAILY)  # 默认返回日线
    
    def to_kline_period(self):
        """
        转换为KlinePeriod
        
        Returns:
            KlinePeriod: 对应的KlinePeriod枚举值
        """
        from enums.kline_period import KlinePeriod
        
        mapping = {
            self.DAILY: KlinePeriod.DAILY,
            self.WEEKLY: KlinePeriod.WEEKLY,
            self.MONTHLY: KlinePeriod.MONTHLY,
            self.MIN60: KlinePeriod.MIN_60,
            self.MIN30: KlinePeriod.MIN_30,
            self.MIN15: KlinePeriod.MIN_15
        }
        
        return mapping.get(self, KlinePeriod.DAILY)  # 默认返回日线 