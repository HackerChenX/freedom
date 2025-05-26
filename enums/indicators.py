"""
技术指标枚举模块

定义各种技术指标相关的枚举类型
"""

from enum import Enum, auto


class IndicatorType(Enum):
    """指标类型枚举"""
    MA = auto()  # 移动平均线
    EMA = auto()  # 指数移动平均线
    BOLL = auto()  # 布林带
    KDJ = auto()  # KDJ指标
    MACD = auto()  # MACD指标
    RSI = auto()  # RSI指标
    VOLUME = auto()  # 成交量指标
    VWAP = auto()  # 成交量加权平均价格
    OBV = auto()  # 能量潮
    ADX = auto()  # 平均趋向指数
    ATR = auto()  # 真实波幅
    CCI = auto()  # 顺势指标
    STOCH = auto()  # 随机指标
    WILLIAMSR = auto()  # 威廉指标


class CrossType(Enum):
    """交叉类型枚举"""
    GOLDEN_CROSS = auto()  # 金叉
    DEATH_CROSS = auto()  # 死叉
    ABOVE = auto()  # 在上方
    BELOW = auto()  # 在下方
    BETWEEN = auto()  # 在两者之间


class TrendType(Enum):
    """趋势类型枚举"""
    UPTREND = auto()  # 上升趋势
    DOWNTREND = auto()  # 下降趋势
    SIDEWAYS = auto()  # 震荡趋势
    REVERSAL_UP = auto()  # 向上反转
    REVERSAL_DOWN = auto()  # 向下反转


class VolumePattern(Enum):
    """成交量模式枚举"""
    INCREASING = auto()  # 放量
    DECREASING = auto()  # 缩量
    SURGE = auto()  # 暴量
    DRYING_UP = auto()  # 量枯
    CONSISTENT = auto()  # 均衡


class PatternType(Enum):
    """形态类型枚举"""
    HEAD_AND_SHOULDERS = auto()  # 头肩顶
    INVERSE_HEAD_AND_SHOULDERS = auto()  # 头肩底
    DOUBLE_TOP = auto()  # 双顶
    DOUBLE_BOTTOM = auto()  # 双底
    TRIPLE_TOP = auto()  # 三重顶
    TRIPLE_BOTTOM = auto()  # 三重底
    RISING_WEDGE = auto()  # 上升楔形
    FALLING_WEDGE = auto()  # 下降楔形
    SYMMETRICAL_TRIANGLE = auto()  # 对称三角形
    ASCENDING_TRIANGLE = auto()  # 上升三角形
    DESCENDING_TRIANGLE = auto()  # 下降三角形
    FLAG = auto()  # 旗形
    PENNANT = auto()  # 三角旗
    RECTANGLE = auto()  # 矩形整理
    CUP_AND_HANDLE = auto()  # 杯柄形态
    ISLAND_REVERSAL = auto()  # 岛形反转 