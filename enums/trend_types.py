"""
趋势类型枚举模块

定义各种趋势类型的枚举值
"""

from enum import Enum, auto

class Trend_direction(Enum):
    """趋势方向枚举"""
    up = auto()        # 上升趋势
    down = auto()      # 下降趋势
    sideways = auto()  # 盘整趋势
    unknown = auto()   # 未知趋势

class Trendstrength_types(Enum):
    """趋势强度枚举"""
    strong = auto()    # 强势趋势
    medium = auto()    # 中等趋势
    weak = auto()      # 弱势趋势
    uncertain = auto() # 不确定趋势

class Trend_phase(Enum):
    """趋势阶段枚举"""
    beginning = auto()  # 趋势初期
    middle = auto()     # 趋势中期
    ending = auto()     # 趋势后期
    reversal = auto()   # 趋势反转
    continuation = auto() # 趋势延续

class Trend_pattern(Enum):
    """趋势形态枚举"""
    breakout = auto()     # 突破
    pullback = auto()     # 回调
    consolidation = auto() # 整固
    reversal = auto()      # 反转
    acceleration = auto()  # 加速
    exhaustion = auto()    # 衰竭 