"""
ZXM体系指标模块

包含ZXM体系的各类指标实现
"""

from indicators.zxm.trend_indicators import (
    ZXMDailyTrendUp, ZXMWeeklyTrendUp, ZXMMonthlyKDJTrendUp,
    ZXMWeeklyKDJDOrDEATrendUp, ZXMWeeklyKDJDTrendUp,
    ZXMMonthlyMACD, ZXMWeeklyMACD, TrendDetector,
    TrendDuration
)

from indicators.zxm.elasticity_indicators import (
    AmplitudeElasticity, ZXMRiseElasticity,
    Elasticity, BounceDetector
)

from indicators.zxm.buy_point_indicators import (
    ZXMDailyMACD, ZXMTurnover, ZXMVolumeShrink,
    ZXMMACallback, ZXMBSAbsorb, BuyPointDetector
)

from indicators.zxm.score_indicators import (
    ZXMElasticityScore, ZXMBuyPointScore, StockScoreCalculator
)

from indicators.zxm.selection_model import SelectionModel

from indicators.zxm.diagnostics import ZXMDiagnostics
from indicators.zxm.market_breadth import ZXMMarketBreadth

# from indicators.zxm.market_indicators import (
#     MarketSentiment, MarketVolatility, SectorRotation, MarketBreadth
# )
# from indicators.zxm.volume_price_indicators import (
#     VolumePriceBreakout, VolumeFlow, PriceVolumeTrend
# )

__all__ = [
    # 趋势指标
    'ZXMDailyTrendUp', 'ZXMWeeklyTrendUp', 'ZXMMonthlyKDJTrendUp',
    'ZXMWeeklyKDJDOrDEATrendUp', 'ZXMWeeklyKDJDTrendUp',
    'ZXMMonthlyMACD', 'ZXMWeeklyMACD', 'TrendDetector',
    'TrendDuration',
    
    # 弹性指标
    'AmplitudeElasticity', 'ZXMRiseElasticity',
    'Elasticity', 'BounceDetector',
    
    # 买点指标
    'ZXMDailyMACD', 'ZXMTurnover', 'ZXMVolumeShrink',
    'ZXMMACallback', 'ZXMBSAbsorb', 'BuyPointDetector',
    
    # 评分指标
    'ZXMElasticityScore', 'ZXMBuyPointScore', 'StockScoreCalculator',
    
    # 选股模型
    'SelectionModel',
    
    # 诊断指标
    'ZXMDiagnostics',
    
    # 市场宽度指标
    'ZXMMarketBreadth',
    
    # # 市场指标
    # 'MarketSentiment', 'MarketVolatility', 'SectorRotation', 'MarketBreadth',
    
    # # 价格指标
    # 'VolumePriceBreakout', 'VolumeFlow', 'PriceVolumeTrend'
] 