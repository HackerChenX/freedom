#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
信号类型枚举

定义各种交易信号类型的枚举
"""

from enum import Enum, auto

class SignalType(Enum):
    """交易信号类型"""
    
    # 基本信号类型
    BUY = "买入"
    SELL = "卖出"
    HOLD = "持有"
    NEUTRAL = "中性"
    
    # 强度信号
    STRONG_BUY = "强烈买入"
    WEAK_BUY = "弱买入"
    STRONG_SELL = "强烈卖出"
    WEAK_SELL = "弱卖出"
    
    # 技术信号
    BREAKOUT = "突破"
    BREAKDOWN = "跌破"
    PULLBACK = "回调"
    BOUNCE = "反弹"
    
    # 趋势信号
    UPTREND = "上升趋势"
    DOWNTREND = "下降趋势"
    SIDEWAYS = "横盘"
    REVERSAL = "反转"
    
    # 动量信号
    MOMENTUM_UP = "动量向上"
    MOMENTUM_DOWN = "动量向下"
    MOMENTUM_NEUTRAL = "动量中性"
    
    # 成交量信号
    VOLUME_SURGE = "放量"
    VOLUME_DRY = "缩量"
    VOLUME_NORMAL = "正常量"

class SignalStrength(Enum):
    """信号强度"""
    
    VERY_STRONG = 5
    STRONG = 4
    MODERATE = 3
    WEAK = 2
    VERY_WEAK = 1
    NEUTRAL = 0

class SignalConfidence(Enum):
    """信号置信度"""
    
    VERY_HIGH = 0.9
    HIGH = 0.8
    MEDIUM = 0.6
    LOW = 0.4
    VERY_LOW = 0.2

# 兼容性别名
Signal_Type = SignalType
Signal_Strength = SignalStrength
Signal_Confidence = SignalConfidence
