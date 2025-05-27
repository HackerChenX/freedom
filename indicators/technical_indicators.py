#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
技术指标集合模块

集中导入各个独立的技术指标，方便统一调用
"""

# 导入指标
from indicators.ma import MA
from indicators.macd import MACD
from indicators.kdj import KDJ
from indicators.rsi import RSI
from indicators.boll import BOLL
from indicators.bias import BIAS
from indicators.cci import CCI
from indicators.dmi import DMI
from indicators.ema import EMA
from indicators.wma import WMA
from indicators.obv import OBV
from indicators.pvt import PVT
from indicators.vr import VR
from indicators.mfi import MFI
from indicators.vosc import VOSC
from indicators.vol import VOL
from indicators.mtm import MTM
from indicators.roc import ROC
from indicators.wr import WR

# 将指标类添加到__all__列表中，明确导出
__all__ = [
    'MA', 'MACD', 'KDJ', 'RSI', 'BOLL',
    'BIAS', 'CCI', 'DMI', 'EMA', 'WMA', 'OBV',
    'PVT', 'VR', 'MFI', 'VOSC', 'VOL', 'MTM',
    'ROC', 'WR'
] 