"""
ZXM体系指标模块

包含ZXM体系的各类指标实现
"""

from indicators.zxm.trend_indicators import (
    ZXMDailyTrendUp, ZXMWeeklyTrendUp, ZXMMonthlyKDJTrendUp,
    ZXMWeeklyKDJDOrDEATrendUp, ZXMWeeklyKDJDTrendUp,
    ZXMMonthlyMACD, ZXMWeeklyMACD, TrendDetector,
    TrendStrength, TrendDuration
)

from indicators.zxm.elasticity_indicators import (
    ZXMAmplitudeElasticity, ZXMRiseElasticity,
    ElasticityIndicator, BounceDetector
)

from indicators.zxm.buy_point_indicators import (
    ZXMDailyMACD, ZXMTurnover, ZXMVolumeShrink,
    ZXMMACallback, ZXMBSAbsorb, BuyPointDetector
)

from indicators.zxm.score_indicators import (
    ZXMElasticityScore, ZXMBuyPointScore, StockScoreCalculator
)

from indicators.zxm.selection_model import SelectionModel

__all__ = [
    # 趋势指标
    'ZXMDailyTrendUp', 'ZXMWeeklyTrendUp', 'ZXMMonthlyKDJTrendUp',
    'ZXMWeeklyKDJDOrDEATrendUp', 'ZXMWeeklyKDJDTrendUp',
    'ZXMMonthlyMACD', 'ZXMWeeklyMACD', 'TrendDetector',
    'TrendStrength', 'TrendDuration',
    
    # 弹性指标
    'ZXMAmplitudeElasticity', 'ZXMRiseElasticity',
    'ElasticityIndicator', 'BounceDetector',
    
    # 买点指标
    'ZXMDailyMACD', 'ZXMTurnover', 'ZXMVolumeShrink',
    'ZXMMACallback', 'ZXMBSAbsorb', 'BuyPointDetector',
    
    # 评分指标
    'ZXMElasticityScore', 'ZXMBuyPointScore', 'StockScoreCalculator',
    
    # 选股模型
    'SelectionModel'
] 