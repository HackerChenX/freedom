#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from enums.indicator_types import TrendType, CrossType
from indicators.indicator_registry import IndicatorEnum
from utils.signal_utils import crossover, crossunder
from .base_indicator import BaseIndicator

logger = logging.getLogger(__name__)

class KC(BaseIndicator):
    """
    肯特纳通道指标 (Keltner Channels)
    
    肯特纳通道是一种结合了移动平均线和波动率的技术指标，由中轨(EMA)、上轨(EMA + ATR倍数)和下轨(EMA - ATR倍数)组成。
    与布林带相似，肯特纳通道可用于识别趋势方向、判断突破信号和市场波动情况。
    
    参数:
        ema_period: EMA周期，默认为20
        atr_period: ATR周期，默认为10
        multiplier: ATR倍数，默认为2
    """
    
    def __init__(self, name: str = "KC", description: str = "肯特纳通道指标",
                 ema_period: int = 20, atr_period: int = 10, multiplier: float = 2):
        """初始化肯特纳通道指标"""
        super().__init__(name, description)
        self.indicator_type = IndicatorEnum.KC.name
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.multiplier = multiplier
        self._result = None
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算肯特纳通道指标
        
        Args:
            df: 包含high, low, close列的DataFrame
            
        Returns:
            包含KC_Upper, KC_Middle, KC_Lower列的DataFrame
        """
        if self._result is not None:
            return self._result
            
        result = df.copy()
        
        # 计算EMA作为中轨
        result['KC_Middle'] = result['close'].ewm(span=self.ema_period, adjust=False).mean()
        
        # 计算ATR
        result['tr'] = np.maximum(
            np.maximum(
                result['high'] - result['low'],
                np.abs(result['high'] - result['close'].shift())
            ),
            np.abs(result['low'] - result['close'].shift())
        )
        result['atr'] = result['tr'].rolling(window=self.atr_period).mean()
        
        # 计算上下轨
        result['KC_Upper'] = result['KC_Middle'] + self.multiplier * result['atr']
        result['KC_Lower'] = result['KC_Middle'] - self.multiplier * result['atr']
        
        # 计算带宽
        result['KC_Width'] = (result['KC_Upper'] - result['KC_Lower']) / result['KC_Middle'] * 100
        
        # 计算价格相对位置
        result['KC_Position'] = (result['close'] - result['KC_Lower']) / (result['KC_Upper'] - result['KC_Lower'])
        
        # 删除临时列
        result = result.drop(['tr', 'atr'], axis=1)
        
        self._result = result
        return result
    
    def generate_signals(self, df: pd.DataFrame) -> List[Dict]:
        """
        生成标准化的交易信号
        
        Args:
            df: 包含OHLCV数据的DataFrame
            
        Returns:
            包含交易信号的字典列表
        """
        signals = []
        result = self.calculate(df)
        
        # 确保有足够的数据
        if len(result) < self.ema_period + 5:
            return signals
            
        # 获取最新数据
        latest = result.iloc[-1]
        prev = result.iloc[-2]
        
        # 当前价格
        current_price = latest['close']
        
        # 通道状态
        kc_upper = latest['KC_Upper']
        kc_middle = latest['KC_Middle']
        kc_lower = latest['KC_Lower']
        kc_width = latest['KC_Width']
        kc_position = latest['KC_Position']
        
        # 计算通道宽度变化率
        width_change = (kc_width / result['KC_Width'].iloc[-5:].mean() - 1) * 100
        
        # 判断趋势方向
        if kc_middle > result['KC_Middle'].shift(5).iloc[-1]:
            trend = TrendType.UPTREND
            trend_strength = min(100, 50 + width_change * 0.5)
        elif kc_middle < result['KC_Middle'].shift(5).iloc[-1]:
            trend = TrendType.DOWNTREND
            trend_strength = min(100, 50 + width_change * 0.5)
        else:
            trend = TrendType.SIDEWAYS
            trend_strength = 50
        
        # 基础信号评分(0-100)
        score = 50  # 中性分值
        
        # 判断价格相对位置
        if current_price > kc_upper:  # 突破上轨
            position_score = 80
            signal_type = "突破上轨"
            signal_desc = "价格突破上轨，可能处于强势上涨"
            cross_type = CrossType.PRICE_CROSS_UPPER
        elif current_price < kc_lower:  # 突破下轨
            position_score = 20
            signal_type = "突破下轨"
            signal_desc = "价格突破下轨，可能处于强势下跌"
            cross_type = CrossType.PRICE_CROSS_LOWER
        elif kc_position > 0.8:  # 接近上轨
            position_score = 70
            signal_type = "接近上轨"
            signal_desc = "价格接近上轨，上涨动能较强"
            cross_type = CrossType.APPROACHING_UPPER
        elif kc_position < 0.2:  # 接近下轨
            position_score = 30
            signal_type = "接近下轨"
            signal_desc = "价格接近下轨，下跌动能较强"
            cross_type = CrossType.APPROACHING_LOWER
        elif crossover(result['close'], result['KC_Middle']):  # 向上穿越中轨
            position_score = 65
            signal_type = "穿越中轨"
            signal_desc = "价格向上穿越中轨，转为偏多"
            cross_type = CrossType.CROSS_OVER
        elif crossunder(result['close'], result['KC_Middle']):  # 向下穿越中轨
            position_score = 35
            signal_type = "穿越中轨"
            signal_desc = "价格向下穿越中轨，转为偏空"
            cross_type = CrossType.CROSS_UNDER
        else:  # 在通道内
            position_score = 40 + kc_position * 20
            signal_type = "通道内运行"
            signal_desc = "价格在通道内运行"
            cross_type = CrossType.NO_CROSS
        
        # 结合趋势和位置调整最终评分
        if trend == TrendType.UPTREND:
            score = position_score * 1.1
        elif trend == TrendType.DOWNTREND:
            score = position_score * 0.9
        else:
            score = position_score
            
        # 通道收缩/扩张判断
        if width_change > 20:
            width_signal = "通道扩张"
            width_desc = "通道宽度扩大，波动性增强"
            if score > 50:  # 偏多时
                score += 5
            else:  # 偏空时
                score -= 5
        elif width_change < -20:
            width_signal = "通道收缩"
            width_desc = "通道宽度收缩，波动性降低，可能蓄势待发"
            # 通道收缩时，评分向中性靠拢
            score = 50 + (score - 50) * 0.7
        else:
            width_signal = "通道正常"
            width_desc = "通道宽度正常"
        
        # 计算建议仓位(0-100%)
        if score >= 70:
            position_pct = min(100, score)
        elif score <= 30:
            position_pct = 0
        else:
            position_pct = (score - 30) * 100 / 40
            
        # 生成买卖信号
        if score >= 70:
            buy_signal = True
            sell_signal = False
        elif score <= 30:
            buy_signal = False
            sell_signal = True
        else:
            buy_signal = False
            sell_signal = False
            
        # 计算置信度(0-100%)
        if cross_type in [CrossType.PRICE_CROSS_UPPER, CrossType.PRICE_CROSS_LOWER]:
            confidence = min(90, 70 + width_change * 0.5)
        elif cross_type in [CrossType.APPROACHING_UPPER, CrossType.APPROACHING_LOWER]:
            confidence = 75
        elif cross_type in [CrossType.CROSS_OVER, CrossType.CROSS_UNDER]:
            confidence = 70
        else:
            confidence = 60
            
        # 调整置信度根据通道宽度变化
        if abs(width_change) > 30:
            confidence = max(50, confidence - 10)
            
        # 风险等级(1-5)
        risk_level = 3
        if score > 80 or score < 20:
            risk_level = 4
        elif width_change > 30:
            risk_level = 5
            
        # 止损计算
        if buy_signal:
            stop_loss = kc_lower
        elif sell_signal:
            stop_loss = kc_upper
        else:
            stop_loss = None
            
        # 创建信号字典
        signal = {
            "indicator": "KC",
            "timestamp": df.index[-1],
            "buy_signal": buy_signal,
            "sell_signal": sell_signal,
            "score": score,
            "trend": trend.value,
            "trend_strength": trend_strength,
            "signal_type": signal_type,
            "signal_desc": signal_desc,
            "cross_type": cross_type.value,
            "confidence": confidence,
            "risk_level": risk_level,
            "position_pct": position_pct,
            "stop_loss": stop_loss,
            "additional_info": {
                "kc_upper": kc_upper,
                "kc_middle": kc_middle,
                "kc_lower": kc_lower,
                "kc_width": kc_width,
                "kc_position": kc_position,
                "width_change": width_change,
                "width_signal": width_signal,
                "width_desc": width_desc
            },
            "market_environment": self.detect_market_environment(df).value if hasattr(self, 'detect_market_environment') else None,
            "volume_confirmation": self.check_volume_confirmation(df) if hasattr(self, 'check_volume_confirmation') else None
        }
        
        signals.append(signal)
        return signals
        
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算原始评分(0-100分)
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            包含每个时间点评分的Series
        """
        result = self.calculate(data)
        
        # 初始化评分Series
        scores = pd.Series(index=result.index, data=50.0)
        
        # 基于价格相对位置计算评分
        for i in range(len(result)):
            if i < self.ema_period:
                continue
                
            # 获取当前数据
            row = result.iloc[i]
            position = row['KC_Position']
            
            # 基础评分
            base_score = 50
            
            # 根据价格位置调整评分
            if row['close'] > row['KC_Upper']:
                position_score = 80
            elif row['close'] < row['KC_Lower']:
                position_score = 20
            else:
                position_score = 40 + position * 20
                
            # 考虑通道斜率
            if i >= 5:
                slope = (row['KC_Middle'] - result['KC_Middle'].iloc[i-5]) / result['KC_Middle'].iloc[i-5] * 100
                if slope > 1:
                    slope_score = min(15, slope * 3)
                elif slope < -1:
                    slope_score = max(-15, slope * 3)
                else:
                    slope_score = 0
            else:
                slope_score = 0
                
            # 考虑通道宽度变化
            if i >= 5:
                width_now = row['KC_Width']
                width_prev = result['KC_Width'].iloc[i-5]
                width_change = (width_now / width_prev - 1) * 100
                
                if width_change > 20:
                    width_score = 5
                elif width_change < -20:
                    width_score = -5
                else:
                    width_score = 0
            else:
                width_score = 0
                
            # 计算最终评分
            final_score = base_score + (position_score - 50) + slope_score + width_score
            
            # 限制评分范围
            scores.iloc[i] = np.clip(final_score, 0, 100)
            
        return scores
        
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别技术形态
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            识别出的形态列表
        """
        result = self.calculate(data)
        patterns = []
        
        # 确保有足够的数据
        if len(result) < self.ema_period + 5:
            return patterns
            
        # 获取最新数据
        latest = result.iloc[-1]
        
        # 通道宽度变化率
        width_change = (latest['KC_Width'] / result['KC_Width'].iloc[-10:-5].mean() - 1) * 100
        
        # 识别形态
        
        # 1. 通道突破
        if latest['close'] > latest['KC_Upper']:
            patterns.append("上轨突破")
        elif latest['close'] < latest['KC_Lower']:
            patterns.append("下轨突破")
            
        # 2. 通道收缩/扩张
        if width_change < -30:
            patterns.append("通道极度收缩")
        elif width_change < -20:
            patterns.append("通道收缩")
        elif width_change > 50:
            patterns.append("通道极度扩张")
        elif width_change > 30:
            patterns.append("通道扩张")
            
        # 3. 价格与中轨关系
        if crossover(result['close'], result['KC_Middle']):
            patterns.append("向上穿越中轨")
        elif crossunder(result['close'], result['KC_Middle']):
            patterns.append("向下穿越中轨")
            
        # 4. 通道方向
        slope = (latest['KC_Middle'] - result['KC_Middle'].iloc[-6]) / result['KC_Middle'].iloc[-6] * 100
        if slope > 2:
            patterns.append("通道上升")
        elif slope < -2:
            patterns.append("通道下降")
        else:
            patterns.append("通道平行")
            
        # 5. 反弹/回调到中轨
        if 0.45 < latest['KC_Position'] < 0.55:
            if result['KC_Position'].iloc[-5] > 0.7:
                patterns.append("从上轨回调至中轨")
            elif result['KC_Position'].iloc[-5] < 0.3:
                patterns.append("从下轨反弹至中轨")
                
        return patterns 