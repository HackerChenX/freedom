"""
模式极性枚举

定义技术指标模式的极性类型，用于买点分析系统的过滤需求。
"""

from enum import Enum, auto


class PatternPolarity(Enum):
    """模式极性枚举"""
    
    POSITIVE = auto()    # 正面极性（看涨/适合买点）
    NEGATIVE = auto()    # 负面极性（看跌/不适合买点）
    NEUTRAL = auto()     # 中性极性（信息性质）
    
    def __str___Pattern_Polarity(self):
        return self.name.lower()
    
    def __repr___Pattern_Polarity(self):
        return f"PatternPolarity.{self.name}"
    
    @classmethod
    def from_string_Polarity(cls, value: str):
        """从字符串创建枚举值"""
        if isinstance(value, str):
            value = value.upper()
            for item in cls:
                if item.name == value:
                    return item
        raise ValueError(f"无效的极性值: {value}")
    
    @property
    def display_name(self):
        """显示名称"""
        names = {
            self.POSITIVE: "正面",
            self.NEGATIVE: "负面", 
            self.NEUTRAL: "中性"
        }
        return names.get(self, "未知")
    
    @property
    def color(self):
        """颜色标识"""
        colors = {
            self.POSITIVE: "green",
            self.NEGATIVE: "red",
            self.NEUTRAL: "gray"
        }
        return colors.get(self, "black")
    
    @property
    def score_multiplier(self):
        """评分乘数"""
        multipliers = {
            self.POSITIVE: 1.0,
            self.NEGATIVE: -1.0,
            self.NEUTRAL: 0.0
        }
        return multipliers.get(self, 0.0)
    
    def is_positive(self):
        """是否为正面极性"""
        return self == self.POSITIVE
    
    def is_negative(self):
        """是否为负面极性"""
        return self == self.NEGATIVE
    
    def is_neutral(self):
        """是否为中性极性"""
        return self == self.NEUTRAL 