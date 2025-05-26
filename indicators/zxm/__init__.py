"""
ZXM体系指标包

包含ZXM体系的选股模型指标
"""

# 导入ZXM趋势识别指标
from .trend_indicators import (
    ZXMDailyTrendUp,
    ZXMWeeklyTrendUp,
    ZXMMonthlyKDJTrendUp,
    ZXMWeeklyKDJDOrDEATrendUp,
    ZXMWeeklyKDJDTrendUp,
    ZXMMonthlyMACD,
    ZXMWeeklyMACD
)

# 导入ZXM弹性指标
from .elasticity_indicators import (
    ZXMAmplitudeElasticity,
    ZXMRiseElasticity
)

# 导入ZXM买点指标
from .buy_point_indicators import (
    ZXMDailyMACD,
    ZXMTurnover,
    ZXMVolumeShrink,
    ZXMMACallback,
    ZXMBSAbsorb
)

# 导入ZXM通用选股模型
from .selection_model import ZXMSelectionModel 