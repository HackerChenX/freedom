"""
技术形态分析系统 - 指标模块

提供各种技术指标的计算和分析功能
"""

# 导入指标类
from .base_indicator import BaseIndicator
from .factory import IndicatorFactory
from .macd import MACD
from .kdj import KDJIndicator, KDJ
from .adapter import CompositeIndicator
from .ma import MA
from .rsi import RSI
from .boll import BOLL
from .bias import BIAS
from .cci import CCI
from .dmi import DMI
from .ema import EMA
from .wma import WMA
from .obv import OBV
from .pvt import PVT
from .vr import VR
from .mfi import MFI
from .vosc import VOSC
from .vol import VOL
from .mtm import MTM
from .roc import ROC
from .wr import WR
from .atr import ATR
from .adx import ADX
from .aroon import Aroon
from .ichimoku import Ichimoku
from .vortex import Vortex
from .stochrsi import STOCHRSI
from .intraday_volatility import IntradayVolatility
from .v_shaped_reversal import VShapedReversal
from .island_reversal import IslandReversal
from .time_cycle_analysis import TimeCycleAnalysis
from .volume_ratio import VolumeRatio
from .platform_breakout import PlatformBreakout
from .divergence import DIVERGENCE
from .emv import EMV

# 导入常用技术分析函数
from .common import (
    ma, ema, sma, wma, highest, lowest, ref, std, diff, macd as macd_func,
    kdj as kdj_func, rsi as rsi_func, boll as boll_func, atr as atr_func,
    obv as obv_func, cross, crossover, crossunder, barslast
)

# 为了兼容以前的代码，提供别名
LLV = lowest
HHV = highest
REF = ref
SMA = sma
EMA = ema

__all__ = [
    # 基础类
    'BaseIndicator', 'IndicatorFactory',
    
    # 指标类
    'MACD', 'KDJIndicator', 'KDJ', 'CompositeIndicator', 'MA', 'RSI', 'BOLL',
    'BIAS', 'CCI', 'DMI', 'EMA', 'WMA', 'OBV', 'PVT', 'VR', 'MFI',
    'VOSC', 'VOL', 'MTM', 'ROC', 'WR', 'ATR', 'ADX', 'Aroon', 'Ichimoku',
    'Vortex', 'STOCHRSI', 'IntradayVolatility', 'VShapedReversal',
    'IslandReversal', 'TimeCycleAnalysis', 'VolumeRatio', 'PlatformBreakout',
    'DIVERGENCE', 'EMV',
    
    # 技术分析函数
    'ma', 'ema', 'sma', 'wma', 'highest', 'lowest', 'ref', 'std', 'diff',
    'macd_func', 'kdj_func', 'rsi_func', 'boll_func', 'atr_func', 'obv_func',
    'cross', 'crossover', 'crossunder', 'barslast',
    
    # 别名
    'LLV', 'HHV', 'REF', 'SMA', 'EMA'
] 