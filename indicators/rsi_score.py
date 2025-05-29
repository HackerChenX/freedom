"""
RSI评分指标

基于统一评分框架的RSI指标，提供评分、形态识别和信号生成功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple

from indicators.scoring_framework import IndicatorScoreBase, PatternRecognitionMixin
from indicators.common import rsi as calc_rsi
from utils.logger import get_logger

logger = get_logger(__name__)


class RSIScore(IndicatorScoreBase, PatternRecognitionMixin):
    """
    RSI评分指标
    
    基于RSI指标的评分系统，包含形态识别和信号生成
    """
    
    def __init__(self, period: int = 14, weight: float = 1.0):
        """
        初始化RSI评分指标
        
        Args:
            period: RSI计算周期
            weight: 指标权重
        """
        super().__init__(name="RSI", weight=weight)
        self.period = period
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算RSI原始评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 原始评分序列（0-100分）
        """
        # 计算RSI指标
        rsi_values = calc_rsi(data['close'].values, self.period)
        
        # 初始化评分
        score = pd.Series(50.0, index=data.index)  # 基础分50分
        
        # 1. 位置评分（30分）
        position_score = self._calculate_position_score(rsi_values)
        score += position_score
        
        # 2. 趋势评分（15分）
        trend_score = self._calculate_trend_score(rsi_values)
        score += trend_score
        
        # 3. 反转评分（15分）
        reversal_score = self._calculate_reversal_score(rsi_values)
        score += reversal_score
        
        # 4. 形态评分（10分）
        pattern_score = self._calculate_pattern_score(rsi_values, data['close'])
        score += pattern_score
        
        # 确保评分在0-100范围内
        return np.clip(score, 0, 100)
    
    def _calculate_position_score(self, rsi_values: np.ndarray) -> pd.Series:
        """
        计算位置评分
        
        Args:
            rsi_values: RSI值数组
            
        Returns:
            pd.Series: 位置评分（-30到30分）
        """
        score = pd.Series(0.0, index=range(len(rsi_values)))
        
        for i in range(len(rsi_values)):
            if np.isnan(rsi_values[i]):
                continue
            
            rsi = rsi_values[i]
            
            # 极度超卖区域（0-20）
            if rsi <= 20:
                score.iloc[i] += 25 + (20 - rsi) * 0.25  # 最多30分
            # 超卖区域（20-30）
            elif rsi <= 30:
                score.iloc[i] += 15 + (30 - rsi) * 1.0   # 15-25分
            # 偏低区域（30-40）
            elif rsi <= 40:
                score.iloc[i] += 5 + (40 - rsi) * 1.0    # 5-15分
            # 中性区域（40-60）
            elif 40 < rsi < 60:
                score.iloc[i] += 0  # 中性
            # 偏高区域（60-70）
            elif rsi >= 60 and rsi < 70:
                score.iloc[i] -= (rsi - 60) * 1.0        # -10到0分
            # 超买区域（70-80）
            elif rsi >= 70 and rsi < 80:
                score.iloc[i] -= 10 + (rsi - 70) * 1.5   # -10到-25分
            # 极度超买区域（80-100）
            else:
                score.iloc[i] -= 25 + (rsi - 80) * 0.25  # 最多-30分
        
        return score
    
    def _calculate_trend_score(self, rsi_values: np.ndarray) -> pd.Series:
        """
        计算趋势评分
        
        Args:
            rsi_values: RSI值数组
            
        Returns:
            pd.Series: 趋势评分（-15到15分）
        """
        score = pd.Series(0.0, index=range(len(rsi_values)))
        
        for i in range(3, len(rsi_values)):
            if any(np.isnan(rsi_values[i-j]) for j in range(4)):
                continue
            
            # 计算RSI短期趋势（3期）
            recent_trend = rsi_values[i] - rsi_values[i-2]
            
            # 计算RSI中期趋势（5期）
            if i >= 4:
                medium_trend = (rsi_values[i] + rsi_values[i-1]) / 2 - (rsi_values[i-3] + rsi_values[i-4]) / 2
            else:
                medium_trend = recent_trend
            
            # 趋势一致性评分
            if recent_trend > 2 and medium_trend > 1:
                score.iloc[i] += 15  # 强烈上升趋势
            elif recent_trend > 1 and medium_trend > 0.5:
                score.iloc[i] += 10  # 上升趋势
            elif recent_trend > 0.5:
                score.iloc[i] += 5   # 轻微上升
            elif recent_trend < -2 and medium_trend < -1:
                score.iloc[i] -= 15  # 强烈下降趋势
            elif recent_trend < -1 and medium_trend < -0.5:
                score.iloc[i] -= 10  # 下降趋势
            elif recent_trend < -0.5:
                score.iloc[i] -= 5   # 轻微下降
        
        return score
    
    def _calculate_reversal_score(self, rsi_values: np.ndarray) -> pd.Series:
        """
        计算反转评分
        
        Args:
            rsi_values: RSI值数组
            
        Returns:
            pd.Series: 反转评分（-15到15分）
        """
        score = pd.Series(0.0, index=range(len(rsi_values)))
        
        for i in range(2, len(rsi_values)):
            if any(np.isnan(rsi_values[i-j]) for j in range(3)):
                continue
            
            current_rsi = rsi_values[i]
            prev_rsi = rsi_values[i-1]
            prev2_rsi = rsi_values[i-2]
            
            # 超卖反转
            if current_rsi <= 30 and prev_rsi <= 30:
                if current_rsi > prev_rsi > prev2_rsi:
                    # 连续上升，反转信号
                    intensity = min(15, (30 - current_rsi) * 0.75 + 5)
                    score.iloc[i] += intensity
                elif current_rsi > prev_rsi:
                    # 开始反转
                    score.iloc[i] += 8
            
            # 超买反转
            elif current_rsi >= 70 and prev_rsi >= 70:
                if current_rsi < prev_rsi < prev2_rsi:
                    # 连续下降，反转信号
                    intensity = min(15, (current_rsi - 70) * 0.75 + 5)
                    score.iloc[i] -= intensity
                elif current_rsi < prev_rsi:
                    # 开始反转
                    score.iloc[i] -= 8
            
            # 中线穿越
            elif prev_rsi <= 50 < current_rsi:
                score.iloc[i] += 10  # 上穿中线
            elif prev_rsi >= 50 > current_rsi:
                score.iloc[i] -= 10  # 下穿中线
        
        return score
    
    def _calculate_pattern_score(self, rsi_values: np.ndarray, price_data: pd.Series) -> pd.Series:
        """
        计算形态评分
        
        Args:
            rsi_values: RSI值数组
            price_data: 价格数据
            
        Returns:
            pd.Series: 形态评分（-10到10分）
        """
        score = pd.Series(0.0, index=range(len(rsi_values)))
        
        # 检测背离
        rsi_series = pd.Series(rsi_values, index=price_data.index)
        divergence = self.detect_divergence(price_data, rsi_series, window=10)
        
        # 背离评分
        for i in range(len(score)):
            if i < len(divergence['bottom_divergence']) and divergence['bottom_divergence'].iloc[i]:
                score.iloc[i] += 10  # 底背离，看涨
            elif i < len(divergence['top_divergence']) and divergence['top_divergence'].iloc[i]:
                score.iloc[i] -= 10  # 顶背离，看跌
        
        return score
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别RSI技术形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别出的形态列表
        """
        patterns = []
        
        # 计算RSI指标
        rsi_values = calc_rsi(data['close'].values, self.period)
        rsi_series = pd.Series(rsi_values, index=data.index)
        
        # 检测超买超卖
        oversold = rsi_series <= 30
        overbought = rsi_series >= 70
        extreme_oversold = rsi_series <= 20
        extreme_overbought = rsi_series >= 80
        
        if extreme_oversold.any():
            patterns.append("RSI极度超卖")
        elif oversold.any():
            patterns.append("RSI超卖")
        
        if extreme_overbought.any():
            patterns.append("RSI极度超买")
        elif overbought.any():
            patterns.append("RSI超买")
        
        # 检测中线穿越
        centerline_cross_up = (rsi_series > 50) & (rsi_series.shift(1) <= 50)
        centerline_cross_down = (rsi_series < 50) & (rsi_series.shift(1) >= 50)
        
        if centerline_cross_up.any():
            patterns.append("RSI上穿中线")
        if centerline_cross_down.any():
            patterns.append("RSI下穿中线")
        
        # 检测背离
        divergence = self.detect_divergence(data['close'], rsi_series)
        if divergence['bottom_divergence'].any():
            patterns.append("RSI底背离")
        if divergence['top_divergence'].any():
            patterns.append("RSI顶背离")
        
        # 检测反转形态
        oversold_reversal = (oversold.shift(1) & (rsi_series > rsi_series.shift(1)) & 
                           (rsi_series.shift(1) > rsi_series.shift(2)))
        overbought_reversal = (overbought.shift(1) & (rsi_series < rsi_series.shift(1)) & 
                             (rsi_series.shift(1) < rsi_series.shift(2)))
        
        if oversold_reversal.any():
            patterns.append("RSI超卖反转")
        if overbought_reversal.any():
            patterns.append("RSI超买反转")
        
        return patterns
    
    def generate_signals(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """
        生成RSI交易信号
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            Dict[str, pd.Series]: 信号字典
        """
        # 计算RSI指标
        rsi_values = calc_rsi(data['close'].values, self.period)
        rsi_series = pd.Series(rsi_values, index=data.index)
        
        signals = {}
        
        # 基本超买超卖信号
        signals['oversold'] = rsi_series <= 30
        signals['overbought'] = rsi_series >= 70
        signals['extreme_oversold'] = rsi_series <= 20
        signals['extreme_overbought'] = rsi_series >= 80
        
        # 中线穿越信号
        signals['centerline_cross_up'] = (rsi_series > 50) & (rsi_series.shift(1) <= 50)
        signals['centerline_cross_down'] = (rsi_series < 50) & (rsi_series.shift(1) >= 50)
        
        # 反转信号
        signals['oversold_reversal'] = (signals['oversold'].shift(1) & 
                                       (rsi_series > rsi_series.shift(1)) & 
                                       (rsi_series.shift(1) > rsi_series.shift(2)))
        
        signals['overbought_reversal'] = (signals['overbought'].shift(1) & 
                                         (rsi_series < rsi_series.shift(1)) & 
                                         (rsi_series.shift(1) < rsi_series.shift(2)))
        
        # 背离信号
        divergence = self.detect_divergence(data['close'], rsi_series)
        signals['bottom_divergence'] = divergence['bottom_divergence']
        signals['top_divergence'] = divergence['top_divergence']
        
        # 强势信号组合
        signals['strong_buy'] = (signals['extreme_oversold'] | 
                                (signals['oversold_reversal'] & signals['bottom_divergence']))
        
        signals['strong_sell'] = (signals['extreme_overbought'] | 
                                 (signals['overbought_reversal'] & signals['top_divergence']))
        
        # 趋势确认信号
        rsi_uptrend = (rsi_series > rsi_series.shift(1)) & (rsi_series.shift(1) > rsi_series.shift(2))
        rsi_downtrend = (rsi_series < rsi_series.shift(1)) & (rsi_series.shift(1) < rsi_series.shift(2))
        
        signals['uptrend_confirmation'] = rsi_uptrend & (rsi_series > 50)
        signals['downtrend_confirmation'] = rsi_downtrend & (rsi_series < 50)
        
        return signals 