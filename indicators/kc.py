#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from enums.indicator_types import TrendType, CrossType
from enums.indicator_enum import IndicatorEnum
from indicators.common import crossover, crossunder
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
        self.REQUIRED_COLUMNS = ['high', 'low', 'close']

    def set_parameters(self, period: int = None, atr_period: int = None, multiplier: float = None):
        """
        设置指标参数

        Args:
            period: 中轨移动平均周期
            atr_period: ATR计算周期
            multiplier: ATR乘数
        """
        if period is not None:
            self.period = period
        if atr_period is not None:
            self.atr_period = atr_period
        if multiplier is not None:
            self.multiplier = multiplier
        
    def _calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算KC指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
            
        Returns:
            包含kc_upper, kc_middle, kc_lower列的DataFrame
        """
        if self._result is not None:
            return self._result
            
        result = df.copy()
        
        # 计算中轨(EMA)
        result['kc_middle'] = result['close'].ewm(span=self.period, adjust=False).mean()
        
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
        result['kc_upper'] = result['kc_middle'] + self.multiplier * result['ATR']
        result['kc_lower'] = result['kc_middle'] - self.multiplier * result['ATR']
        
        # 计算通道宽度百分比(相对于中轨价格)
        result['kc_width'] = (result['kc_upper'] - result['kc_lower']) / result['kc_middle'] * 100
        
        # 计算价格相对于通道的位置(0-100%)，0表示在下轨，100表示在上轨
        channel_range = result['kc_upper'] - result['kc_lower']
        # 避免除以零的情况
        result['kc_position'] = np.where(
            channel_range > 0,
            (result['close'] - result['kc_lower']) / channel_range * 100,
            50  # 默认为中间位置
        )
        
        # 计算通道宽度变化率
        result['kc_width_chg'] = result['kc_width'].pct_change(periods=5, fill_method=None) * 100
        
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
        kc_middle = latest['kc_middle']
        kc_upper = latest['kc_upper']
        kc_lower = latest['kc_lower']
        kc_position = latest['kc_position']
        kc_width = latest['kc_width']
        kc_width_chg = latest['kc_width_chg']
        
        # 判断趋势方向
        if current_price > kc_middle:
            trend = TrendType.UP
            trend_strength = 50 + kc_position * 0.5  # 50-100
        elif current_price < kc_middle:
            trend = TrendType.DOWN
            trend_strength = 50 - (100 - kc_position) * 0.5  # 0-50
        else:
            trend = TrendType.FLAT
            trend_strength = 50
        
        # 基础信号评分(0-100)
        score = 50  # 中性分值
        
        # 判断价格与通道的关系
        if crossover(result['close'], result['kc_upper']).any():  # 价格上穿上轨
            signal_type = "上穿上轨"
            signal_desc = "价格上穿肯特纳通道上轨，显示强势突破"
            cross_type = "GOLDEN_CROSS"
            score = 80
        elif crossunder(result['close'], result['kc_lower']).any():  # 价格下穿下轨
            signal_type = "下穿下轨"
            signal_desc = "价格下穿肯特纳通道下轨，显示弱势突破"
            cross_type = "DEATH_CROSS"
            score = 20
        elif current_price > kc_upper:  # 价格在上轨之上
            signal_type = "上轨之上"
            signal_desc = "价格位于肯特纳通道上轨之上，显示超买状态"
            cross_type = CrossType.NO_CROSS
            score = 70 + (current_price - kc_upper) / kc_upper * 100  # 根据超出程度增加评分
        elif current_price < kc_lower:  # 价格在下轨之下
            signal_type = "下轨之下"
            signal_desc = "价格位于肯特纳通道下轨之下，显示超卖状态"
            cross_type = "NO_CROSS"
            score = 30 - (kc_lower - current_price) / kc_lower * 100  # 根据超出程度减少评分
        elif crossover(result['close'], result['kc_middle']).any():  # 价格上穿中轨
            signal_type = "上穿中轨"
            signal_desc = "价格上穿肯特纳通道中轨，显示由弱转强"
            cross_type = "GOLDEN_CROSS"
            score = 60
        elif crossunder(result['close'], result['kc_middle']).any():  # 价格下穿中轨
            signal_type = "下穿中轨"
            signal_desc = "价格下穿肯特纳通道中轨，显示由强转弱"
            cross_type = "DEATH_CROSS"
            score = 40
        elif current_price > kc_middle:  # 价格在中轨和上轨之间
            signal_type = "中上区域"
            signal_desc = "价格位于肯特纳通道中轨和上轨之间，显示温和强势"
            cross_type = "NO_CROSS"
            score = 55 + kc_position * 0.15  # 根据位置线性调整55-70
        elif current_price < kc_middle:  # 价格在中轨和下轨之间
            signal_type = "中下区域"
            signal_desc = "价格位于肯特纳通道中轨和下轨之间，显示温和弱势"
            cross_type = "NO_CROSS"
            score = 45 - (100 - kc_position) * 0.15  # 根据位置线性调整30-45
        else:  # 价格在中轨上
            signal_type = "中轨位置"
            signal_desc = "价格位于肯特纳通道中轨，显示中性"
            cross_type = "NO_CROSS"
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
        if cross_type in ["GOLDEN_CROSS", "DEATH_CROSS"]:
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
            "cross_type": cross_type,
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
        if not isinstance(data, pd.DataFrame) or 'kc_middle' not in data.columns:
            data = self.calculate(data)
        
        # 获取KC指标值
        close = data['close']
        middle = data['kc_middle']
        upper = data['kc_upper']
        lower = data['kc_lower']
        position = data['kc_position']
        
        # 初始化评分
        score = pd.Series(50.0, index=data.index)  # 默认中性评分
        
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
            cross_up_upper_mask = (data['close'].shift(1) <= data['kc_upper'].shift(1)) & (data['close'] > data['kc_upper'])
            score[cross_up_upper_mask] = 80
            
            # 价格下穿下轨
            cross_down_lower_mask = (data['close'].shift(1) >= data['kc_lower'].shift(1)) & (data['close'] < data['kc_lower'])
            score[cross_down_lower_mask] = 20
            
            # 价格上穿中轨
            cross_up_middle_mask = (data['close'].shift(1) <= data['kc_middle'].shift(1)) & (data['close'] > data['kc_middle'])
            score[cross_up_middle_mask] = 60
            
            # 价格下穿中轨
            cross_down_middle_mask = (data['close'].shift(1) >= data['kc_middle'].shift(1)) & (data['close'] < data['kc_middle'])
            score[cross_down_middle_mask] = 40
            
        # 考虑通道宽度变化
        width_chg = data['kc_width_chg']
        # 通道扩大，上升波动
        up_vol_mask = (width_chg > 10) & (close > middle)
        score[up_vol_mask] += 5
        
        # 通道扩大，下降波动
        down_vol_mask = (width_chg > 10) & (close < middle)
        score[down_vol_mask] -= 5
        
        # 确保分数在0-100范围内
        score = score.clip(0, 100)
        
        return score

    def calculate_confidence(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """
        计算KC指标的置信度

        Args:
            score: 得分序列
            patterns: 检测到的形态DataFrame
            signals: 生成的信号字典

        Returns:
            float: 置信度分数 (0-1)
        """
        if score.empty:
            return 0.5

        # 基础置信度
        confidence = 0.5

        # 1. 基于评分的置信度
        last_score = score.iloc[-1]

        # 极端评分置信度较高
        if last_score > 80 or last_score < 20:
            confidence += 0.25
        # 中性评分置信度中等
        elif 40 <= last_score <= 60:
            confidence += 0.1
        else:
            confidence += 0.15

        # 2. 基于形态的置信度
        if isinstance(patterns, pd.DataFrame) and not patterns.empty:
            try:
                # 统计最近几个周期的形态数量
                numeric_cols = patterns.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    recent_data = patterns[numeric_cols].iloc[-5:] if len(patterns) >= 5 else patterns[numeric_cols]
                    recent_patterns = recent_data.sum().sum()
                    if recent_patterns > 0:
                        confidence += min(recent_patterns * 0.05, 0.2)
            except:
                pass

        # 3. 基于KC通道宽度的置信度
        if hasattr(self, '_result') and self._result is not None and 'kc_width' in self._result.columns:
            try:
                width_values = self._result['kc_width'].dropna()
                if len(width_values) > 0:
                    last_width = width_values.iloc[-1]
                    # 通道宽度适中时置信度较高
                    if 3 <= last_width <= 10:
                        confidence += 0.15
                    elif last_width > 15:  # 通道过宽，波动性过高
                        confidence -= 0.1
                    elif last_width < 1:  # 通道过窄，可能即将突破
                        confidence += 0.1
            except:
                pass

        # 4. 基于评分稳定性的置信度
        if len(score) >= 5:
            recent_scores = score.iloc[-5:]
            score_stability = 1.0 - (recent_scores.std() / 50.0)
            confidence += score_stability * 0.1

        return min(confidence, 1.0)

    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别KC指标形态
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            形态描述列表
        """
        # 确保已计算指标
        if not isinstance(data, pd.DataFrame) or 'kc_middle' not in data.columns:
            data = self.calculate(data)
            
        # 获取KC数据
        close = data['close']
        middle = data['kc_middle']
        upper = data['kc_upper']
        lower = data['kc_lower']
        width = data['kc_width']
        width_chg = data['kc_width_chg']
        
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

    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取KC指标的技术形态

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含形态信息的DataFrame
        """
        # 确保已计算KC
        if not self.has_result():
            self.calculate(data, **kwargs)

        if self._result is None:
            return pd.DataFrame(index=data.index)

        close = self._result['close']
        middle = self._result['kc_middle']
        upper = self._result['kc_upper']
        lower = self._result['kc_lower']
        width = self._result['kc_width']

        patterns_df = pd.DataFrame(index=data.index)

        # 1. 价格位置形态
        patterns_df['KC_ABOVE_UPPER'] = close > upper
        patterns_df['KC_BELOW_LOWER'] = close < lower
        patterns_df['KC_ABOVE_MIDDLE'] = (close > middle) & (close <= upper)
        patterns_df['KC_BELOW_MIDDLE'] = (close < middle) & (close >= lower)
        patterns_df['KC_AT_MIDDLE'] = abs(close - middle) / middle < 0.01

        # 2. 突破形态
        patterns_df['KC_BREAK_UPPER'] = crossover(close, upper)
        patterns_df['KC_BREAK_LOWER'] = crossunder(close, lower)
        patterns_df['KC_BREAK_MIDDLE_UP'] = crossover(close, middle)
        patterns_df['KC_BREAK_MIDDLE_DOWN'] = crossunder(close, middle)

        # 3. 通道宽度形态
        if len(width) >= 20:
            width_ma = width.rolling(20).mean()
            patterns_df['KC_WIDE_CHANNEL'] = width > width_ma * 1.5
            patterns_df['KC_NARROW_CHANNEL'] = width < width_ma * 0.5
            patterns_df['KC_EXPANDING'] = width > width.shift(1)
            patterns_df['KC_CONTRACTING'] = width < width.shift(1)

        # 4. 极值形态
        patterns_df['KC_EXTREME_OVERBOUGHT'] = close > upper * 1.02
        patterns_df['KC_EXTREME_OVERSOLD'] = close < lower * 0.98

        # 5. 回归形态
        patterns_df['KC_RETURN_TO_MIDDLE'] = (
            (close.shift(1) > upper.shift(1)) & (close <= upper) |
            (close.shift(1) < lower.shift(1)) & (close >= lower)
        )

        # 6. 震荡形态
        if len(close) >= 10:
            # 检查是否在通道内震荡
            recent_close = close.iloc[-10:]
            recent_upper = upper.iloc[-10:]
            recent_lower = lower.iloc[-10:]
            recent_middle = middle.iloc[-10:]

            in_channel = (recent_close < recent_upper) & (recent_close > recent_lower)
            cross_middle = (
                ((recent_close.shift(1) < recent_middle.shift(1)) & (recent_close > recent_middle)) |
                ((recent_close.shift(1) > recent_middle.shift(1)) & (recent_close < recent_middle))
            ).any()

            patterns_df['KC_OSCILLATING'] = in_channel.all() & cross_middle

        return patterns_df

    def register_patterns(self):
        """
        注册KC指标的技术形态
        """
        # 注册价格突破形态
        self.register_pattern_to_registry(
            pattern_id="KC_BREAK_UPPER",
            display_name="KC上轨突破",
            description="价格突破肯特纳通道上轨，强势信号",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=25.0
        )
    def get_pattern_info(self, pattern_id: str) -> dict:
        """
        获取指定形态的详细信息
        
        Args:
            pattern_id: 形态ID
            
        Returns:
            dict: 形态详细信息
        """
        # 默认形态信息
        default_pattern = {
            "id": pattern_id,
            "name": pattern_id,
            "description": f"{pattern_id}形态",
            "type": "NEUTRAL",
            "strength": "MEDIUM",
            "score_impact": 0.0
        }
        
        # KC指标特定的形态信息映射
        pattern_info_map = {
            # 基础形态
            "超买区域": {
                "id": "超买区域",
                "name": "超买区域",
                "description": "指标进入超买区域，可能面临回调压力",
                "type": "BEARISH",
                "strength": "MEDIUM",
                "score_impact": -10.0
            },
            "超卖区域": {
                "id": "超卖区域", 
                "name": "超卖区域",
                "description": "指标进入超卖区域，可能出现反弹机会",
                "type": "BULLISH",
                "strength": "MEDIUM",
                "score_impact": 10.0
            },
            "中性区域": {
                "id": "中性区域",
                "name": "中性区域", 
                "description": "指标处于中性区域，趋势不明确",
                "type": "NEUTRAL",
                "strength": "WEAK",
                "score_impact": 0.0
            },
            # 趋势形态
            "上升趋势": {
                "id": "上升趋势",
                "name": "上升趋势",
                "description": "指标显示上升趋势，看涨信号",
                "type": "BULLISH", 
                "strength": "STRONG",
                "score_impact": 15.0
            },
            "下降趋势": {
                "id": "下降趋势",
                "name": "下降趋势",
                "description": "指标显示下降趋势，看跌信号",
                "type": "BEARISH",
                "strength": "STRONG", 
                "score_impact": -15.0
            },
            # 信号形态
            "买入信号": {
                "id": "买入信号",
                "name": "买入信号",
                "description": "指标产生买入信号，建议关注",
                "type": "BULLISH",
                "strength": "STRONG",
                "score_impact": 20.0
            },
            "卖出信号": {
                "id": "卖出信号", 
                "name": "卖出信号",
                "description": "指标产生卖出信号，建议谨慎",
                "type": "BEARISH",
                "strength": "STRONG",
                "score_impact": -20.0
            }
        }
        
        return pattern_info_map.get(pattern_id, default_pattern)


        self.register_pattern_to_registry(
            pattern_id="KC_BREAK_LOWER",
            display_name="KC下轨突破",
            description="价格跌破肯特纳通道下轨，弱势信号",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-25.0
        )

        # 注册中轨突破形态
        self.register_pattern_to_registry(
            pattern_id="KC_BREAK_MIDDLE_UP",
            display_name="KC中轨向上突破",
            description="价格向上突破肯特纳通道中轨，由弱转强",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=15.0
        )

        self.register_pattern_to_registry(
            pattern_id="KC_BREAK_MIDDLE_DOWN",
            display_name="KC中轨向下突破",
            description="价格向下突破肯特纳通道中轨，由强转弱",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-15.0
        )

        # 注册极值形态
        self.register_pattern_to_registry(
            pattern_id="KC_EXTREME_OVERBOUGHT",
            display_name="KC极度超买",
            description="价格远超肯特纳通道上轨，极度超买",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-20.0
        )

        self.register_pattern_to_registry(
            pattern_id="KC_EXTREME_OVERSOLD",
            display_name="KC极度超卖",
            description="价格远低于肯特纳通道下轨，极度超卖",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=20.0
        )

        # 注册通道形态
        self.register_pattern_to_registry(
            pattern_id="KC_WIDE_CHANNEL",
            display_name="KC通道扩张",
            description="肯特纳通道宽度扩张，波动性增加",
            pattern_type="NEUTRAL",
            default_strength="WEAK",
            score_impact=5.0
        )

        self.register_pattern_to_registry(
            pattern_id="KC_NARROW_CHANNEL",
            display_name="KC通道收缩",
            description="肯特纳通道宽度收缩，可能酝酿突破",
            pattern_type="NEUTRAL",
            default_strength="MEDIUM",
            score_impact=10.0
        )