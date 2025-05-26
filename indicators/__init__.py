"""
指标模块初始化文件

导出所有技术指标类
"""

from indicators.base_indicator import BaseIndicator
from indicators.ma import MA
from indicators.ema import EMA
from indicators.wma import WMA
from indicators.macd import MACD
from indicators.bias import BIAS
from indicators.boll import BOLL
from indicators.sar import SAR
from indicators.dmi import DMI
from indicators.rsi import RSI
from indicators.kdj import KDJ
from indicators.wr import WR
from indicators.cci import CCI
from indicators.mtm import MTM
from indicators.roc import ROC
from indicators.stochrsi import STOCHRSI
from indicators.vol import VOL
from indicators.obv import OBV
from indicators.vosc import VOSC
from indicators.mfi import MFI
from indicators.vr import VR
from indicators.pvt import PVT
from indicators.volume_ratio import VolumeRatio
from indicators.platform_breakout import PlatformBreakout
from indicators.pattern.candlestick_patterns import CandlestickPatterns
from indicators.zxm_washplate import ZXMWashPlate
from indicators.chip_distribution import ChipDistribution
from indicators.fibonacci_tools import FibonacciTools
from indicators.elliott_wave import ElliottWave
from indicators.gann_tools import GannTools
from indicators.emv import EMV
from indicators.intraday_volatility import IntradayVolatility
from indicators.v_shaped_reversal import VShapedReversal
from indicators.island_reversal import IslandReversal
from indicators.time_cycle_analysis import TimeCycleAnalysis
from indicators.factory import IndicatorFactory
from indicators.momentum import Momentum
from indicators.rsima import RSIMA
# 添加已有但未导入的指标
from indicators.trix import TRIX
from indicators.vix import VIX
from indicators.divergence import DIVERGENCE
from indicators.multi_period_resonance import MULTI_PERIOD_RESONANCE
from indicators.zxm_absorb import ZXM_ABSORB

# 导入ZXM体系指标
from indicators.zxm.trend_indicators import (
    ZXMDailyTrendUp, ZXMWeeklyTrendUp, ZXMMonthlyKDJTrendUp, 
    ZXMWeeklyKDJDOrDEATrendUp, ZXMWeeklyKDJDTrendUp,
    ZXMMonthlyMACD, ZXMWeeklyMACD
)
from indicators.zxm.elasticity_indicators import (
    ZXMAmplitudeElasticity, ZXMRiseElasticity
)
from indicators.zxm.buy_point_indicators import (
    ZXMDailyMACD, ZXMTurnover, ZXMVolumeShrink,
    ZXMMACallback, ZXMBSAbsorb
)
from indicators.zxm.selection_model import ZXMSelectionModel

__all__ = [
    'BaseIndicator',
    'MA',
    'EMA',
    'WMA',
    'MACD',
    'BIAS',
    'BOLL',
    'SAR',
    'DMI',
    'RSI',
    'KDJ',
    'WR',
    'CCI',
    'MTM',
    'ROC',
    'STOCHRSI',
    'VOL',
    'OBV',
    'VOSC',
    'MFI',
    'VR',
    'PVT',
    'VolumeRatio',
    'PlatformBreakout',
    'CandlestickPatterns',
    'ZXMWashPlate',
    'ChipDistribution',
    'FibonacciTools',
    'ElliottWave',
    'GannTools',
    'EMV',
    'IntradayVolatility',
    'VShapedReversal',
    'IslandReversal',
    'TimeCycleAnalysis',
    'Momentum',
    'RSIMA',
    # 添加已有但未导入的指标
    'TRIX',
    'VIX',
    'DIVERGENCE',
    'MULTI_PERIOD_RESONANCE',
    'ZXM_ABSORB',
    # 添加ZXM体系指标
    'ZXMDailyTrendUp',
    'ZXMWeeklyTrendUp',
    'ZXMMonthlyKDJTrendUp',
    'ZXMWeeklyKDJDOrDEATrendUp',
    'ZXMWeeklyKDJDTrendUp',
    'ZXMMonthlyMACD',
    'ZXMWeeklyMACD',
    'ZXMAmplitudeElasticity',
    'ZXMRiseElasticity',
    'ZXMDailyMACD',
    'ZXMTurnover',
    'ZXMVolumeShrink',
    'ZXMMACallback',
    'ZXMBSAbsorb',
    'ZXMSelectionModel',
    'IndicatorFactory'
] 