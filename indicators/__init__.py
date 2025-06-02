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
from indicators.chip_distribution import ChipDistribution
from indicators.adapter import IndicatorAdapter, register_indicator, get_indicator, calculate_indicator, list_all_indicators
from indicators.composite import TechnicalComposite, technical_composite
from indicators.institutional_behavior import InstitutionalBehavior
from indicators.stock_vix import StockVIX
from indicators.fibonacci import Fibonacci
from indicators.sentiment_analysis import SentimentAnalysis
from indicators.adx import ADX
from indicators.atr import ATR
from indicators.trend_classification import TrendClassification, TrendType
from indicators.multi_period_resonance import MultiPeriodResonance

# 注册指标
register_indicator(MA())
register_indicator(EMA())
register_indicator(WMA())
register_indicator(MACD())
register_indicator(BIAS())
register_indicator(BOLL())
register_indicator(SAR())
register_indicator(DMI())
register_indicator(RSI())
register_indicator(KDJ())
register_indicator(WR())
register_indicator(CCI())
register_indicator(MTM())
register_indicator(ROC())
register_indicator(STOCHRSI())
register_indicator(VOL())
register_indicator(OBV())
register_indicator(VOSC())
register_indicator(MFI())
register_indicator(VR())
register_indicator(PVT())
register_indicator(VolumeRatio())
register_indicator(ChipDistribution())
register_indicator(InstitutionalBehavior())
register_indicator(StockVIX())
register_indicator(Fibonacci())
register_indicator(SentimentAnalysis())
register_indicator(ADX())
register_indicator(ATR())
register_indicator(TrendClassification())
register_indicator(MultiPeriodResonance())

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
    'ChipDistribution',
    'IndicatorAdapter',
    'register_indicator',
    'get_indicator',
    'calculate_indicator',
    'list_all_indicators',
    'TechnicalComposite',
    'technical_composite',
    'InstitutionalBehavior',
    'StockVIX',
    'Fibonacci',
    'SentimentAnalysis',
    'ADX',
    'ATR',
    'TrendClassification',
    'TrendType',
    'MultiPeriodResonance'
] 