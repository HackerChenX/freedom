"""
形态识别指标模块
"""

# 导出主要的形态识别类
from .candlestick_patterns import (
    CandlestickPatterns,
    Doji,
    Hammer,
    ShootingStar,
    Engulfing,
    Harami,
    PiercingLine,
    DarkCloudCover,
    MorningStar,
    EveningStar,
    ThreeBlackCrows,
    ThreeWhiteSoldiers,
)

from .advanced_candlestick_patterns import (
    AdvancedCandlestickPatterns,
    HeadShoulders,
    DoubleTop,
    DoubleBottom,
    Triangle,
    Wedge,
    Flag,
    Pennant,
)

try:
    from .rectangle import Rectangle
except ImportError:
    Rectangle = None

try:
    from .cup_and_handle import CupAndHandle
except ImportError:
    CupAndHandle = None

__all__ = [
    "CandlestickPatterns",
    "Doji",
    "Hammer",
    "ShootingStar",
    "Engulfing",
    "Harami",
    "PiercingLine",
    "DarkCloudCover",
    "MorningStar",
    "EveningStar",
    "ThreeBlackCrows",
    "ThreeWhiteSoldiers",
    "AdvancedCandlestickPatterns",
    "HeadShoulders",
    "DoubleTop",
    "DoubleBottom",
    "Triangle",
    "Wedge",
    "Flag",
    "Pennant",
    "Rectangle",
    "CupAndHandle",
]
