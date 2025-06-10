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
    肯特纳通道指标 (Keltner Channel)
    
    肯特纳通道是一种波动通道指标，由中轨(通常为EMA)加减一定倍数的ATR形成上下轨。
    相比于布林带使用标准差，肯特纳通道使用ATR衡量波动性，对价格突破和异常波动的反应更平滑。
    
    参数:
        period: 中轨移动平均周期，默认为20
        atr_period: ATR计算周期，默认为10
        multiplier: ATR乘数，用于计算通道宽度，默认为2.0
    """
    
    def __init__(self, period: int = 20, atr_period: int = 10, multiplier: float = 2.0,
                 name: str = "KC", description: str = "肯特纳通道指标"):
        """初始化KC指标"""
        super().__init__(name, description)
        self.indicator_type = IndicatorEnum.KC.name
        self.period = period
        self.atr_period = atr_period
        self.multiplier = multiplier
        self._result = None
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算KC指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
            
        Returns:
            包含KC_UPPER, KC_MIDDLE, KC_LOWER列的DataFrame
        """
        if self._result is not None:
            return self._result
            
        result = df.copy()
        
        # 计算中轨(EMA)
        result['KC_MIDDLE'] = result['close'].ewm(span=self.period, adjust=False).mean()
        
        # 计算真实波幅(TR)
        result['TR'] = np.maximum(
            result['high'] - result['low'],
            np.maximum(
                np.abs(result['high'] - result['close'].shift(1)),
                np.abs(result['low'] - result['close'].shift(1))
            )
        )
        
        # 填充NaN值
        result['TR'] = result['TR'].fillna(result['high'] - result['low'])
        
        # 计算ATR
        result['ATR'] = result['TR'].rolling(window=self.atr_period).mean()
        
        # 计算上下轨
        result['KC_UPPER'] = result['KC_MIDDLE'] + self.multiplier * result['ATR']
        result['KC_LOWER'] = result['KC_MIDDLE'] - self.multiplier * result['ATR']
        
        # 计算通道宽度百分比(相对于中轨价格)
        result['KC_WIDTH'] = (result['KC_UPPER'] - result['KC_LOWER']) / result['KC_MIDDLE'] * 100
        
        # 计算价格相对于通道的位置(0-100%)，0表示在下轨，100表示在上轨
        channel_range = result['KC_UPPER'] - result['KC_LOWER']
        # 避免除以零的情况
        result['KC_POSITION'] = np.where(
            channel_range > 0,
            (result['close'] - result['KC_LOWER']) / channel_range * 100,
            50  # 默认为中间位置
        )
        
        # 计算通道宽度变化率
        result['KC_WIDTH_CHG'] = result['KC_WIDTH'].pct_change(periods=5) * 100
        
        # 删除临时计算列
        result = result.drop(['TR', 'ATR'], axis=1)
        
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
        if len(result) < self.period + 5:
            return signals
            
        # 获取最新数据
        latest = result.iloc[-1]
        prev = result.iloc[-2]
        
        # 当前价格
        current_price = latest['close']
        
        # KC指标状态
        kc_middle = latest['KC_MIDDLE']
        kc_upper = latest['KC_UPPER']
        kc_lower = latest['KC_LOWER']
        kc_position = latest['KC_POSITION']
        kc_width = latest['KC_WIDTH']
        kc_width_chg = latest['KC_WIDTH_CHG']
        
        # 判断趋势方向
        if current_price > kc_middle:
            trend = TrendType.UPTREND
            trend_strength = 50 + kc_position * 0.5  # 50-100
        elif current_price < kc_middle:
            trend = TrendType.DOWNTREND
            trend_strength = 50 - (100 - kc_position) * 0.5  # 0-50
        else:
            trend = TrendType.SIDEWAYS
            trend_strength = 50
        
        # 基础信号评分(0-100)
        score = 50  # 中性分值
        
        # 判断价格与通道的关系
        if crossover(result['close'], result['KC_UPPER']):  # 价格上穿上轨
            signal_type = "上穿上轨"
            signal_desc = "价格上穿肯特纳通道上轨，显示强势突破"
            cross_type = CrossType.CROSS_OVER
            score = 80
        elif crossunder(result['close'], result['KC_LOWER']):  # 价格下穿下轨
            signal_type = "下穿下轨"
            signal_desc = "价格下穿肯特纳通道下轨，显示弱势突破"
            cross_type = CrossType.CROSS_UNDER
            score = 20
        elif current_price > kc_upper:  # 价格在上轨之上
            signal_type = "上轨之上"
            signal_desc = "价格位于肯特纳通道上轨之上，显示超买状态"
            cross_type = CrossType.NO_CROSS
            score = 70 + (current_price - kc_upper) / kc_upper * 100  # 根据超出程度增加评分
        elif current_price < kc_lower:  # 价格在下轨之下
            signal_type = "下轨之下"
            signal_desc = "价格位于肯特纳通道下轨之下，显示超卖状态"
            cross_type = CrossType.NO_CROSS
            score = 30 - (kc_lower - current_price) / kc_lower * 100  # 根据超出程度减少评分
        elif crossover(result['close'], result['KC_MIDDLE']):  # 价格上穿中轨
            signal_type = "上穿中轨"
            signal_desc = "价格上穿肯特纳通道中轨，显示由弱转强"
            cross_type = CrossType.CROSS_OVER
            score = 60
        elif crossunder(result['close'], result['KC_MIDDLE']):  # 价格下穿中轨
            signal_type = "下穿中轨"
            signal_desc = "价格下穿肯特纳通道中轨，显示由强转弱"
            cross_type = CrossType.CROSS_UNDER
            score = 40
        elif current_price > kc_middle:  # 价格在中轨和上轨之间
            signal_type = "中上区域"
            signal_desc = "价格位于肯特纳通道中轨和上轨之间，显示温和强势"
            cross_type = CrossType.NO_CROSS
            score = 55 + kc_position * 0.15  # 根据位置线性调整55-70
        elif current_price < kc_middle:  # 价格在中轨和下轨之间
            signal_type = "中下区域"
            signal_desc = "价格位于肯特纳通道中轨和下轨之间，显示温和弱势"
            cross_type = CrossType.NO_CROSS
            score = 45 - (100 - kc_position) * 0.15  # 根据位置线性调整30-45
        else:  # 价格在中轨上
            signal_type = "中轨位置"
            signal_desc = "价格位于肯特纳通道中轨，显示中性"
            cross_type = CrossType.NO_CROSS
            score = 50
            
        # 考虑通道宽度变化
        if kc_width_chg > 10:
            if current_price > kc_middle:
                score += 5
                signal_desc += f"，通道宽度扩大({kc_width_chg:.2f}%)，上升波动加剧"
            else:
                score -= 5
                signal_desc += f"，通道宽度扩大({kc_width_chg:.2f}%)，下降波动加剧"
        elif kc_width_chg < -10:
            signal_desc += f"，通道宽度收窄({kc_width_chg:.2f}%)，波动减弱，可能酝酿大行情"
            
        # 考虑通道宽度绝对水平
        if kc_width > 10:
            signal_desc += f"，当前通道宽度较大({kc_width:.2f}%)，市场波动性高"
        elif kc_width < 3:
            signal_desc += f"，当前通道宽度较小({kc_width:.2f}%)，市场波动性低，可能即将爆发"
            
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
        if cross_type in [CrossType.CROSS_OVER, CrossType.CROSS_UNDER]:
            if current_price > kc_upper or current_price < kc_lower:
                confidence = 85  # 突破外轨的交叉信号
            else:
                confidence = 75  # 内部的交叉信号
        elif current_price > kc_upper or current_price < kc_lower:
            confidence = 80  # 持续在外轨
        else:
            confidence = 60 + abs(kc_position - 50) * 0.4  # 根据位置调整
            
        # 风险等级(1-5)
        risk_level = 3
        if kc_width > 8:
            risk_level = 4  # 通道宽度大，波动性高
        elif kc_width < 3:
            risk_level = 2  # 通道宽度小，波动性低
            
        # 止损计算
        if buy_signal:
            # 止损设为通道下轨或最近5天最低价，取较高者
            stop_loss = max(kc_lower, df['low'].iloc[-5:].min())
        elif sell_signal:
            # 止损设为通道上轨或最近5天最高价，取较低者
            stop_loss = min(kc_upper, df['high'].iloc[-5:].max())
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
                "kc_middle": kc_middle,
                "kc_upper": kc_upper,
                "kc_lower": kc_lower,
                "kc_position": kc_position,
                "kc_width": kc_width,
                "kc_width_chg": kc_width_chg
            }
        }
        
        signals.append(signal)
        return signals
        
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算原始评分(0-100分)
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            包含评分的Series，范围0-100
        """
        # 确保已计算指标
        if not isinstance(data, pd.DataFrame) or 'KC_MIDDLE' not in data.columns:
            data = self.calculate(data)
        
        # 获取KC指标值
        close = data['close']
        middle = data['KC_MIDDLE']
        upper = data['KC_UPPER']
        lower = data['KC_LOWER']
        position = data['KC_POSITION']
        
        # 初始化评分
        score = pd.Series(50, index=data.index)  # 默认中性评分
        
        # 价格位置评分
        # 1. 价格在上轨之上
        above_upper_mask = close > upper
        score[above_upper_mask] = 70 + (close[above_upper_mask] - upper[above_upper_mask]) / upper[above_upper_mask] * 100
        
        # 2. 价格在下轨之下
        below_lower_mask = close < lower
        score[below_lower_mask] = 30 - (lower[below_lower_mask] - close[below_lower_mask]) / lower[below_lower_mask] * 100
        
        # 3. 价格在中轨和上轨之间
        between_mid_upper_mask = (close >= middle) & (close <= upper)
        score[between_mid_upper_mask] = 55 + position[between_mid_upper_mask] * 0.15
        
        # 4. 价格在中轨和下轨之间
        between_mid_lower_mask = (close <= middle) & (close >= lower)
        score[between_mid_lower_mask] = 45 - (100 - position[between_mid_lower_mask]) * 0.15
        
        # 考虑交叉情况
        if len(data) >= 2:
            # 价格上穿上轨
            cross_up_upper_mask = (data['close'].shift(1) <= data['KC_UPPER'].shift(1)) & (data['close'] > data['KC_UPPER'])
            score[cross_up_upper_mask] = 80
            
            # 价格下穿下轨
            cross_down_lower_mask = (data['close'].shift(1) >= data['KC_LOWER'].shift(1)) & (data['close'] < data['KC_LOWER'])
            score[cross_down_lower_mask] = 20
            
            # 价格上穿中轨
            cross_up_middle_mask = (data['close'].shift(1) <= data['KC_MIDDLE'].shift(1)) & (data['close'] > data['KC_MIDDLE'])
            score[cross_up_middle_mask] = 60
            
            # 价格下穿中轨
            cross_down_middle_mask = (data['close'].shift(1) >= data['KC_MIDDLE'].shift(1)) & (data['close'] < data['KC_MIDDLE'])
            score[cross_down_middle_mask] = 40
            
        # 考虑通道宽度变化
        width_chg = data['KC_WIDTH_CHG']
        # 通道扩大，上升波动
        up_vol_mask = (width_chg > 10) & (close > middle)
        score[up_vol_mask] += 5
        
        # 通道扩大，下降波动
        down_vol_mask = (width_chg > 10) & (close < middle)
        score[down_vol_mask] -= 5
        
        # 确保分数在0-100范围内
        score = score.clip(0, 100)
        
        return score

    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别KC指标形态
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            形态描述列表
        """
        # 确保已计算指标
        if not isinstance(data, pd.DataFrame) or 'KC_MIDDLE' not in data.columns:
            data = self.calculate(data)
            
        # 获取KC数据
        close = data['close']
        middle = data['KC_MIDDLE']
        upper = data['KC_UPPER']
        lower = data['KC_LOWER']
        width = data['KC_WIDTH']
        width_chg = data['KC_WIDTH_CHG']
        
        patterns = []
        
        # 检查价格位置
        if close.iloc[-1] > upper.iloc[-1]:
            patterns.append("KC超买区域")
        elif close.iloc[-1] < lower.iloc[-1]:
            patterns.append("KC超卖区域")
        elif close.iloc[-1] > middle.iloc[-1]:
            patterns.append("KC上行区域")
        elif close.iloc[-1] < middle.iloc[-1]:
            patterns.append("KC下行区域")
            
        # 检查交叉
        if len(data) >= 2:
            if close.iloc[-2] <= upper.iloc[-2] and close.iloc[-1] > upper.iloc[-1]:
                patterns.append("KC上穿上轨")
            elif close.iloc[-2] >= lower.iloc[-2] and close.iloc[-1] < lower.iloc[-1]:
                patterns.append("KC下穿下轨")
            elif close.iloc[-2] <= middle.iloc[-2] and close.iloc[-1] > middle.iloc[-1]:
                patterns.append("KC上穿中轨")
            elif close.iloc[-2] >= middle.iloc[-2] and close.iloc[-1] < middle.iloc[-1]:
                patterns.append("KC下穿中轨")
                
        # 检查通道宽度
        if width.iloc[-1] > 10:
            patterns.append("KC通道宽度大")
        elif width.iloc[-1] < 3:
            patterns.append("KC通道宽度小")
            
        if width_chg.iloc[-1] > 10:
            patterns.append("KC通道扩张")
        elif width_chg.iloc[-1] < -10:
            patterns.append("KC通道收缩")
            
        # 检查价格在通道内的行为模式
        # 通道内震荡
        if (close.iloc[-5:] < upper.iloc[-5:]).all() and (close.iloc[-5:] > lower.iloc[-5:]).all():
            crossing_middle = False
            for i in range(1, 5):
                if ((close.iloc[-i-1] < middle.iloc[-i-1]) and (close.iloc[-i] > middle.iloc[-i])) or \
                   ((close.iloc[-i-1] > middle.iloc[-i-1]) and (close.iloc[-i] < middle.iloc[-i])):
                    crossing_middle = True
                    break
                    
            if crossing_middle:
                patterns.append("KC通道内震荡")
                
        # 连续触及上轨但未突破
        if ((close.iloc[-3:] <= upper.iloc[-3:]) & (close.iloc[-3:] >= upper.iloc[-3:] * 0.99)).any():
            patterns.append("KC顶部测试")
            
        # 连续触及下轨但未突破
        if ((close.iloc[-3:] >= lower.iloc[-3:]) & (close.iloc[-3:] <= lower.iloc[-3:] * 1.01)).any():
            patterns.append("KC底部测试")
            
        return patterns 