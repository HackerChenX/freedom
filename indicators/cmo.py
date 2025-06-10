#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from indicators.indicator_registry import IndicatorEnum

from enums.indicator_types import TrendType, CrossType
from utils.signal_utils import crossover, crossunder
from .base_indicator import BaseIndicator

logger = logging.getLogger(__name__)

class CMO(BaseIndicator):
    """
    钱德动量摆动指标 (Chande Momentum Oscillator)
    
    CMO是一种由图莎尔·钱德(Tushar Chande)创建的动量指标，结合了动量和波动的元素。
    该指标通过比较一段时间内上涨和下跌的总和来计算，范围为-100至+100。
    
    CMO = 100 × ((Su - Sd) / (Su + Sd))
    其中：
    - Su是特定周期内价格上涨总和
    - Sd是特定周期内价格下跌的绝对值总和
    
    参数:
        period: 计算周期，默认为14
        oversold: 超卖阈值，默认为-40
        overbought: 超买阈值，默认为40
    """
    
    def __init__(self, period: int = 14, oversold: float = -40, overbought: float = 40,
                 name: str = "CMO", description: str = "钱德动量摆动指标"):
        """初始化CMO指标"""
        super().__init__(name, description)
        self.indicator_type = IndicatorEnum.CMO.name
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self._result = None
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算CMO指标
        
        Args:
            df: 包含close列的DataFrame
            
        Returns:
            包含CMO列的DataFrame
        """
        if self._result is not None:
            return self._result
            
        result = df.copy()
        
        # 计算价格变化
        result['price_change'] = result['close'].diff()
        
        # 计算上涨和下跌值
        result['up'] = result['price_change'].apply(lambda x: x if x > 0 else 0)
        result['down'] = result['price_change'].apply(lambda x: abs(x) if x < 0 else 0)
        
        # 计算上涨和下跌的移动和
        result['up_sum'] = result['up'].rolling(window=self.period).sum()
        result['down_sum'] = result['down'].rolling(window=self.period).sum()
        
        # 计算CMO，避免除以零的情况
        up_down_sum = result['up_sum'] + result['down_sum']
        # 处理可能的零除情况
        result['CMO'] = np.where(
            up_down_sum > 0,
            100 * ((result['up_sum'] - result['down_sum']) / up_down_sum),
            0  # 如果分母为零，则返回0
        )
        
        # 删除临时列
        result = result.drop(['price_change', 'up', 'down', 'up_sum', 'down_sum'], axis=1)
        
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
        
        # CMO值
        cmo = latest['CMO']
        prev_cmo = prev['CMO']
        
        # 判断趋势方向
        if cmo > 0:
            trend = TrendType.UPTREND
            trend_strength = min(100, 50 + cmo * 0.5)
        elif cmo < 0:
            trend = TrendType.DOWNTREND
            trend_strength = min(100, 50 - cmo * 0.5)
        else:
            trend = TrendType.SIDEWAYS
            trend_strength = 50
        
        # 基础信号评分(0-100)
        score = 50  # 中性分值
        
        # 判断CMO位置和交叉情况
        if cmo > self.overbought:  # 超买区
            position_score = 70
            signal_type = "超买区域"
            signal_desc = f"CMO位于超买区域({cmo:.2f})，可能出现回调"
            cross_type = CrossType.NO_CROSS
            
            # 如果刚刚进入超买区，强调这一点
            if prev_cmo <= self.overbought:
                signal_type = "进入超买区域"
                signal_desc = f"CMO刚刚进入超买区域({cmo:.2f})，上涨动能强劲但注意可能回调"
                cross_type = CrossType.CROSS_OVER
                position_score = 75
                
        elif cmo < self.oversold:  # 超卖区
            position_score = 30
            signal_type = "超卖区域"
            signal_desc = f"CMO位于超卖区域({cmo:.2f})，可能出现反弹"
            cross_type = CrossType.NO_CROSS
            
            # 如果刚刚进入超卖区，强调这一点
            if prev_cmo >= self.oversold:
                signal_type = "进入超卖区域"
                signal_desc = f"CMO刚刚进入超卖区域({cmo:.2f})，下跌动能强劲但注意可能反弹"
                cross_type = CrossType.CROSS_UNDER
                position_score = 25
                
        elif crossover(result['CMO'], 0):  # 上穿零轴
            position_score = 65
            signal_type = "上穿零轴"
            signal_desc = "CMO上穿零轴，动量由负转正，看涨信号"
            cross_type = CrossType.CROSS_OVER
            
        elif crossunder(result['CMO'], 0):  # 下穿零轴
            position_score = 35
            signal_type = "下穿零轴"
            signal_desc = "CMO下穿零轴，动量由正转负，看跌信号"
            cross_type = CrossType.CROSS_UNDER
            
        elif crossover(result['CMO'], self.oversold):  # 上穿超卖线
            position_score = 60
            signal_type = "离开超卖区域"
            signal_desc = f"CMO上穿超卖线({self.oversold})，下跌动能减弱，可能反弹"
            cross_type = CrossType.CROSS_OVER
            
        elif crossunder(result['CMO'], self.overbought):  # 下穿超买线
            position_score = 40
            signal_type = "离开超买区域"
            signal_desc = f"CMO下穿超买线({self.overbought})，上涨动能减弱，可能回调"
            cross_type = CrossType.CROSS_UNDER
            
        else:  # 中性区域
            # 根据CMO值在中性区域内的位置线性调整评分
            position_pct = (cmo - self.oversold) / (self.overbought - self.oversold)
            position_score = 40 + position_pct * 20
            
            if cmo > 0:
                signal_type = "正动量区域"
                signal_desc = f"CMO在正区域({cmo:.2f})，市场呈现正动量"
            else:
                signal_type = "负动量区域"
                signal_desc = f"CMO在负区域({cmo:.2f})，市场呈现负动量"
                
            cross_type = CrossType.NO_CROSS
            
        # 考虑CMO斜率调整评分
        if len(result) >= 5:
            cmo_slope = (cmo - result['CMO'].iloc[-5]) / 5
            
            if abs(cmo_slope) > 3:  # 快速变化
                if cmo_slope > 0:
                    score = position_score + 5
                    signal_desc += f"，CMO快速上升({cmo_slope:.2f}/天)"
                else:
                    score = position_score - 5
                    signal_desc += f"，CMO快速下降({cmo_slope:.2f}/天)"
            else:
                score = position_score
        else:
            score = position_score
            
        # 检查背离
        if len(result) >= 20:
            # 价格创新高但CMO没有创新高 - 顶背离
            price_high = df['close'].iloc[-20:].max() == current_price
            cmo_high = result['CMO'].iloc[-20:].max() == cmo
            
            if price_high and not cmo_high and cmo > 0:
                score -= 10
                signal_desc += "，出现顶背离迹象，上涨动能减弱"
                
            # 价格创新低但CMO没有创新低 - 底背离
            price_low = df['close'].iloc[-20:].min() == current_price
            cmo_low = result['CMO'].iloc[-20:].min() == cmo
            
            if price_low and not cmo_low and cmo < 0:
                score += 10
                signal_desc += "，出现底背离迹象，下跌动能减弱"
                
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
            confidence = 75
        elif abs(cmo) > 60:  # 极端值
            confidence = 80
        else:
            confidence = 60 + abs(cmo) * 0.2
            
        # 风险等级(1-5)
        risk_level = 3
            
        # 止损计算
        if buy_signal:
            # 止损设为当前价格的95%或最近5天最低价，取较高者
            stop_loss = max(current_price * 0.95, df['low'].iloc[-5:].min())
        elif sell_signal:
            # 止损设为当前价格的105%或最近5天最高价，取较低者
            stop_loss = min(current_price * 1.05, df['high'].iloc[-5:].max())
        else:
            stop_loss = None
            
        # 创建信号字典
        signal = {
            "indicator": "CMO",
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
                "cmo": cmo,
                "oversold": self.oversold,
                "overbought": self.overbought
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
        if not isinstance(data, pd.DataFrame) or 'CMO' not in data.columns:
            data = self.calculate(data)
        
        # 获取CMO值
        cmo = data['CMO']
        
        # 初始化评分
        score = pd.Series(50, index=data.index)  # 默认中性评分
        
        # CMO大于0为看涨，小于0为看跌
        bullish_mask = cmo > 0
        bearish_mask = cmo < 0
        
        # 超买区域
        overbought_mask = cmo > self.overbought
        # 超卖区域
        oversold_mask = cmo < self.oversold
        
        # 设置基础分
        # 1. 超买区域：反转思路，分数越高越看跌
        score[overbought_mask] = 100 - (cmo[overbought_mask] - self.overbought) * 0.5
        # 2. 超卖区域：反转思路，分数越低越看涨
        score[oversold_mask] = 0 + (cmo[oversold_mask] - self.oversold) * 0.5
        # 3. 中性区域看涨
        neutral_bullish_mask = ~(overbought_mask | oversold_mask) & bullish_mask
        score[neutral_bullish_mask] = 50 + cmo[neutral_bullish_mask] * 0.5
        # 4. 中性区域看跌
        neutral_bearish_mask = ~(overbought_mask | oversold_mask) & bearish_mask
        score[neutral_bearish_mask] = 50 + cmo[neutral_bearish_mask] * 0.5
        
        # 考虑CMO斜率
        if len(data) >= 5:
            # 计算5日CMO变化率
            cmo_change = data['CMO'] - data['CMO'].shift(5)
            
            # 上升动量加分
            up_momentum_mask = cmo_change > 3
            score[up_momentum_mask] += 5
            
            # 下降动量减分
            down_momentum_mask = cmo_change < -3
            score[down_momentum_mask] -= 5
            
        # 确保分数在0-100范围内
        score = score.clip(0, 100)
        
        return score
        
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别CMO指标的技术形态
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            形态描述列表
        """
        # 确保已计算指标
        if not isinstance(data, pd.DataFrame) or 'CMO' not in data.columns:
            data = self.calculate(data)
            
        # 获取CMO数据
        cmo = data['CMO']
        close = data['close']
        
        patterns = []
        
        # 检查超买/超卖条件
        if cmo.iloc[-1] > self.overbought:
            patterns.append("CMO超买")
            
        if cmo.iloc[-1] < self.oversold:
            patterns.append("CMO超卖")
            
        # 检查零线交叉
        if len(data) >= 2:
            if cmo.iloc[-2] <= 0 and cmo.iloc[-1] > 0:
                patterns.append("CMO上穿零线")
                
            if cmo.iloc[-2] >= 0 and cmo.iloc[-1] < 0:
                patterns.append("CMO下穿零线")
                
        # 检查超买/超卖区域的离开
        if len(data) >= 2:
            if cmo.iloc[-2] >= self.overbought and cmo.iloc[-1] < self.overbought:
                patterns.append("CMO离开超买区")
                
            if cmo.iloc[-2] <= self.oversold and cmo.iloc[-1] > self.oversold:
                patterns.append("CMO离开超卖区")
                
        # 检查背离
        if len(data) >= 20:
            # 找出最近20天的最高价和最低价
            last_20_high_idx = close.iloc[-20:].idxmax()
            last_20_low_idx = close.iloc[-20:].idxmin()
            
            # 检查顶背离：价格创新高，但CMO未创新高
            if last_20_high_idx == data.index[-1]:  # 最新价格是20天内最高
                cmo_at_price_high = cmo.iloc[-1]
                max_cmo_in_period = cmo.iloc[-20:].max()
                
                if cmo_at_price_high < max_cmo_in_period * 0.9:  # CMO比之前最高点低10%以上
                    patterns.append("CMO顶背离")
                    
            # 检查底背离：价格创新低，但CMO未创新低
            if last_20_low_idx == data.index[-1]:  # 最新价格是20天内最低
                cmo_at_price_low = cmo.iloc[-1]
                min_cmo_in_period = cmo.iloc[-20:].min()
                
                if cmo_at_price_low > min_cmo_in_period * 0.9:  # CMO比之前最低点高10%以上
                    patterns.append("CMO底背离")
                    
        return patterns 