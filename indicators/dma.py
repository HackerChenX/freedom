#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional
import logging
from .base_indicator import BaseIndicator
from utils.signal_utils import crossover, crossunder
from enums.trend_types import TrendType
from enums.cross_types import CrossType

logger = logging.getLogger(__name__)

class DMA(BaseIndicator):
    """
    轨道线指标 (Different of Moving Average)
    
    DMA指标由两条均线的差值组成，通过快速均线与慢速均线之差以及这个差值的移动平均线来判断中长期的买卖点。
    该指标适合中长期趋势判断，是一种典型的趋势跟踪指标。
    
    参数:
        fast_period: 短期均线周期，默认为10
        slow_period: 长期均线周期，默认为50
        ama_period: 差值平均线周期，默认为10
    """
    
    def __init__(self, fast_period: int = 10, slow_period: int = 50, ama_period: int = 10):
        """初始化DMA指标"""
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.ama_period = ama_period
        self._result = None
        
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算DMA指标
        
        Args:
            df: 包含close列的DataFrame
            
        Returns:
            包含DMA, AMA列的DataFrame
        """
        if self._result is not None:
            return self._result
            
        result = df.copy()
        
        # 计算快速均线和慢速均线
        result['FAST_MA'] = result['close'].rolling(window=self.fast_period).mean()
        result['SLOW_MA'] = result['close'].rolling(window=self.slow_period).mean()
        
        # 计算DMA值（两条均线之差）
        result['DMA'] = result['FAST_MA'] - result['SLOW_MA']
        
        # 计算DMA的移动平均线(AMA)
        result['AMA'] = result['DMA'].rolling(window=self.ama_period).mean()
        
        # 计算FASTMA与SLOWMA的百分比差值
        result['DMA_PCT'] = (result['FAST_MA'] / result['SLOW_MA'] - 1) * 100
        
        # 计算FASTMA的变化率
        result['FAST_MA_CHG'] = result['FAST_MA'].pct_change(periods=5) * 100
        
        # 删除不需要的临时列
        result = result.drop(['FAST_MA', 'SLOW_MA'], axis=1)
        
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
        if len(result) < self.slow_period + 5:
            return signals
            
        # 获取最新数据
        latest = result.iloc[-1]
        prev = result.iloc[-2]
        
        # 当前价格
        current_price = latest['close']
        
        # DMA指标状态
        dma = latest['DMA']
        ama = latest['AMA']
        dma_pct = latest['DMA_PCT']
        fast_ma_chg = latest['FAST_MA_CHG']
        
        # 判断趋势方向
        if dma > 0 and dma > ama:
            trend = TrendType.UPTREND
            trend_strength = min(100, 50 + dma_pct * 2)
        elif dma < 0 and dma < ama:
            trend = TrendType.DOWNTREND
            trend_strength = min(100, 50 - dma_pct * 2)
        else:
            trend = TrendType.SIDEWAYS
            trend_strength = 50
        
        # 基础信号评分(0-100)
        score = 50  # 中性分值
        
        # 判断DMA和AMA的关系及DMA的绝对水平
        if crossover(result['DMA'], result['AMA']):  # DMA上穿AMA
            signal_type = "DMA上穿AMA"
            signal_desc = "DMA上穿AMA，显示由空头转为多头趋势"
            cross_type = CrossType.CROSS_OVER
            score = 70
        elif crossunder(result['DMA'], result['AMA']):  # DMA下穿AMA
            signal_type = "DMA下穿AMA"
            signal_desc = "DMA下穿AMA，显示由多头转为空头趋势"
            cross_type = CrossType.CROSS_UNDER
            score = 30
        elif dma > ama and dma_pct > 0:  # DMA在AMA上方且为正
            signal_type = "多头趋势增强"
            signal_desc = f"DMA位于AMA上方，百分比差值为{dma_pct:.2f}%，多头趋势增强"
            cross_type = CrossType.NO_CROSS
            score = 60 + min(30, dma_pct * 1.5)
        elif dma < ama and dma_pct < 0:  # DMA在AMA下方且为负
            signal_type = "空头趋势增强"
            signal_desc = f"DMA位于AMA下方，百分比差值为{dma_pct:.2f}%，空头趋势增强"
            cross_type = CrossType.NO_CROSS
            score = 40 - min(30, abs(dma_pct * 1.5))
        elif dma > 0 and ama > 0:  # DMA和AMA都为正
            signal_type = "弱势多头"
            signal_desc = "DMA和AMA均为正值，处于弱势多头"
            cross_type = CrossType.NO_CROSS
            score = 55
        elif dma < 0 and ama < 0:  # DMA和AMA都为负
            signal_type = "弱势空头"
            signal_desc = "DMA和AMA均为负值，处于弱势空头"
            cross_type = CrossType.NO_CROSS
            score = 45
        else:  # 其他情况
            signal_type = "震荡整理"
            signal_desc = "DMA指标处于震荡状态，无明确方向"
            cross_type = CrossType.NO_CROSS
            score = 50
            
        # 考虑FASTMA变化率调整评分
        if fast_ma_chg > 2:
            score += 5
            if signal_type.startswith("多头"):
                signal_desc += f"，短期均线加速上涨({fast_ma_chg:.2f}%)"
        elif fast_ma_chg < -2:
            score -= 5
            if signal_type.startswith("空头"):
                signal_desc += f"，短期均线加速下跌({fast_ma_chg:.2f}%)"
                
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
            confidence = 80
        elif abs(dma_pct) > 5:
            confidence = 75
        else:
            confidence = 60
            
        # 调整置信度根据趋势一致性
        if (dma > 0 and ama > 0) or (dma < 0 and ama < 0):
            confidence += 10
            
        # 风险等级(1-5)
        risk_level = 3
        if abs(dma_pct) > 10:
            risk_level = 4
            
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
            "indicator": "DMA",
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
                "dma": dma,
                "ama": ama,
                "dma_pct": dma_pct,
                "fast_ma_chg": fast_ma_chg
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
        
        # 基于DMA和AMA关系计算评分
        for i in range(len(result)):
            if i < self.slow_period:
                continue
                
            # 获取当前数据
            row = result.iloc[i]
            dma = row['DMA']
            ama = row['AMA']
            dma_pct = row['DMA_PCT']
            
            # 基础评分
            base_score = 50
            
            # DMA和AMA关系评分
            if dma > ama:
                cross_score = 10
            elif dma < ama:
                cross_score = -10
            else:
                cross_score = 0
                
            # DMA绝对水平评分
            if dma > 0:
                level_score = min(20, dma_pct * 2)
            else:
                level_score = max(-20, dma_pct * 2)
                
            # 交叉信号评分
            if i > 0:
                prev_row = result.iloc[i-1]
                if prev_row['DMA'] <= prev_row['AMA'] and dma > ama:
                    cross_signal_score = 20  # 上穿
                elif prev_row['DMA'] >= prev_row['AMA'] and dma < ama:
                    cross_signal_score = -20  # 下穿
                else:
                    cross_signal_score = 0
            else:
                cross_signal_score = 0
                
            # 计算最终评分
            final_score = base_score + cross_score + level_score + cross_signal_score
            
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
        if len(result) < self.slow_period + 5:
            return patterns
            
        # 获取最新数据
        latest = result.iloc[-1]
        
        # 识别形态
        
        # 1. DMA和AMA交叉
        if crossover(result['DMA'], result['AMA']):
            patterns.append("DMA金叉AMA")
        elif crossunder(result['DMA'], result['AMA']):
            patterns.append("DMA死叉AMA")
            
        # 2. DMA零轴穿越
        if crossover(result['DMA'], 0):
            patterns.append("DMA上穿零轴")
        elif crossunder(result['DMA'], 0):
            patterns.append("DMA下穿零轴")
            
        # 3. DMA趋势
        if latest['DMA'] > 0 and latest['AMA'] > 0:
            patterns.append("DMA多头排列")
        elif latest['DMA'] < 0 and latest['AMA'] < 0:
            patterns.append("DMA空头排列")
            
        # 4. DMA背离
        if len(result) >= 20:
            # 检查最近20个交易日内的高点和低点
            high_idx = result['close'].iloc[-20:].idxmax()
            low_idx = result['close'].iloc[-20:].idxmin()
            
            # 价格创新高但DMA没有创新高 - 顶背离
            if high_idx == result.index[-1] and result['DMA'].iloc[-1] < result['DMA'].iloc[-20:-1].max():
                patterns.append("DMA顶背离")
                
            # 价格创新低但DMA没有创新低 - 底背离
            if low_idx == result.index[-1] and result['DMA'].iloc[-1] > result['DMA'].iloc[-20:-1].min():
                patterns.append("DMA底背离")
                
        # 5. DMA百分比水平
        if latest['DMA_PCT'] > 5:
            patterns.append("DMA强势多头区域")
        elif latest['DMA_PCT'] < -5:
            patterns.append("DMA强势空头区域")
        elif -1 < latest['DMA_PCT'] < 1:
            patterns.append("DMA中性区域")
            
        return patterns 