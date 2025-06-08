"""
趋势类型枚举模块

定义各种趋势类型的枚举值
"""

from enum import Enum, auto

class TrendDirection(Enum):
    """趋势方向枚举"""
    UP = auto()        # 上升趋势
    DOWN = auto()      # 下降趋势
    SIDEWAYS = auto()  # 盘整趋势
    UNKNOWN = auto()   # 未知趋势

class TrendStrength(Enum):
    """趋势强度枚举"""
    STRONG = auto()    # 强势趋势
    MEDIUM = auto()    # 中等趋势
    WEAK = auto()      # 弱势趋势
    UNCERTAIN = auto() # 不确定趋势

class TrendPhase(Enum):
    """趋势阶段枚举"""
    BEGINNING = auto()  # 趋势初期
    MIDDLE = auto()     # 趋势中期
    ENDING = auto()     # 趋势后期
    REVERSAL = auto()   # 趋势反转
    CONTINUATION = auto() # 趋势延续

class TrendPattern(Enum):
    """趋势形态枚举"""
    BREAKOUT = auto()     # 突破
    PULLBACK = auto()     # 回调
    CONSOLIDATION = auto() # 整固
    REVERSAL = auto()      # 反转
    ACCELERATION = auto()  # 加速
    EXHAUSTION = auto()    # 衰竭 