"""
BOLL评分指标

基于统一评分框架的布林带指标，提供评分、形态识别和信号生成功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple

from indicators.scoring_framework import IndicatorScoreBase, PatternRecognitionMixin
from indicators.common import ma, std
from utils.logger import get_logger

logger = get_logger(__name__)


class BOLLScore(IndicatorScoreBase, PatternRecognitionMixin):
    """
    BOLL评分指标
    
    基于布林带指标的评分系统，包含形态识别和信号生成
    """
    
    def __init__(self, period: int = 20, std_dev: float = 2.0, weight: float = 1.0):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """
        初始化BOLL评分指标
        
        Args:
            period: 移动平均周期
            std_dev: 标准差倍数
            weight: 指标权重
        """
        super().__init__(name="BOLL", weight=weight)
        self.period = period
        self.std_dev = std_dev
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算BOLL原始评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 原始评分序列（0-100分）
        """
        # 计算布林带指标
        close_prices = data['close'].values
        middle_band = ma(close_prices, self.period)
        std_values = std(close_prices, self.period)
        upper_band = middle_band + (std_values * self.std_dev)
        lower_band = middle_band - (std_values * self.std_dev)
        
        # 初始化评分
        score = pd.Series(50.0, index=data.index)  # 基础分50分
        
        # 1. 位置评分（25分）
        position_score = self._calculate_position_score(close_prices, upper_band, middle_band, lower_band)
        score += position_score
        
        # 2. 带宽评分（20分）
        bandwidth_score = self._calculate_bandwidth_score(upper_band, lower_band, middle_band)
        score += bandwidth_score
        
        # 3. 突破评分（20分）
        breakout_score = self._calculate_breakout_score(close_prices, upper_band, lower_band)
        score += breakout_score
        
        # 4. 回归评分（15分）
        regression_score = self._calculate_regression_score(close_prices, upper_band, middle_band, lower_band)
        score += regression_score
        
        # 确保评分在0-100范围内
        return np.clip(score, 0, 100)
    
    def _calculate_position_score(self, close_prices: np.ndarray, upper_band: np.ndarray, 
                                 middle_band: np.ndarray, lower_band: np.ndarray) -> pd.Series:
        """
        计算位置评分
        
        Args:
            close_prices: 收盘价数组
            upper_band: 上轨数组
            middle_band: 中轨数组
            lower_band: 下轨数组
            
        Returns:
            pd.Series: 位置评分（-25到25分）
        """
        score = pd.Series(0.0, index=range(len(close_prices)))
        
        for i in range(len(close_prices)):
            if (np.isnan(close_prices[i]) or np.isnan(upper_band[i]) or 
                np.isnan(middle_band[i]) or np.isnan(lower_band[i])):
                continue
            
            price = close_prices[i]
            upper = upper_band[i]
            middle = middle_band[i]
            lower = lower_band[i]
            
            # 计算价格在布林带中的相对位置
            if upper != lower:
                position_ratio = (price - lower) / (upper - lower)
            else:
                position_ratio = 0.5  # 默认中性位置
            
            # 根据位置给分
            if position_ratio <= 0.1:  # 接近下轨
                score.iloc[i] += 20 + (0.1 - position_ratio) * 50  # 最多25分
            elif position_ratio <= 0.2:  # 下轨附近
                score.iloc[i] += 15 + (0.2 - position_ratio) * 50
            elif position_ratio <= 0.3:  # 偏下位置
                score.iloc[i] += 10
            elif position_ratio <= 0.4:  # 中下位置
                score.iloc[i] += 5
            elif 0.4 < position_ratio < 0.6:  # 中性位置
                score.iloc[i] += 0
            elif position_ratio >= 0.9:  # 接近上轨
                score.iloc[i] -= 20 + (position_ratio - 0.9) * 50  # 最多-25分
            elif position_ratio >= 0.8:  # 上轨附近
                score.iloc[i] -= 15 + (position_ratio - 0.8) * 50
            elif position_ratio >= 0.7:  # 偏上位置
                score.iloc[i] -= 10
            elif position_ratio >= 0.6:  # 中上位置
                score.iloc[i] -= 5
        
        return score
    
    def _calculate_bandwidth_score(self, upper_band: np.ndarray, lower_band: np.ndarray, 
                                  middle_band: np.ndarray) -> pd.Series:
        """
        计算带宽评分
        
        Args:
            upper_band: 上轨数组
            lower_band: 下轨数组
            middle_band: 中轨数组
            
        Returns:
            pd.Series: 带宽评分（-20到20分）
        """
        score = pd.Series(0.0, index=range(len(upper_band)))
        
        # 计算带宽
        bandwidth = np.zeros(len(upper_band))
        for i in range(len(upper_band)):
            if not np.isnan(upper_band[i]) and not np.isnan(lower_band[i]) and middle_band[i] != 0:
                bandwidth[i] = (upper_band[i] - lower_band[i]) / middle_band[i]
        
        # 计算带宽的移动平均和标准差
        for i in range(20, len(bandwidth)):
            if np.isnan(bandwidth[i]):
                continue
            
            # 计算最近20期的带宽统计
            recent_bandwidth = bandwidth[i-19:i+1]
            recent_bandwidth = recent_bandwidth[~np.isnan(recent_bandwidth)]
            
            if len(recent_bandwidth) < 10:
                continue
            
            avg_bandwidth = np.mean(recent_bandwidth)
            current_bandwidth = bandwidth[i]
            
            # 带宽相对变化
            if avg_bandwidth > 0:
                bandwidth_ratio = current_bandwidth / avg_bandwidth
                
                # 带宽收窄（可能突破）
                if bandwidth_ratio < 0.7:
                    score.iloc[i] += 15  # 带宽极度收窄，可能突破
                elif bandwidth_ratio < 0.8:
                    score.iloc[i] += 10  # 带宽收窄
                elif bandwidth_ratio < 0.9:
                    score.iloc[i] += 5   # 轻微收窄
                
                # 带宽扩张（趋势加强）
                elif bandwidth_ratio > 1.3:
                    score.iloc[i] += 10  # 带宽扩张，趋势可能加强
                elif bandwidth_ratio > 1.2:
                    score.iloc[i] += 5   # 轻微扩张
        
        return score
    
    def _calculate_breakout_score(self, close_prices: np.ndarray, upper_band: np.ndarray, 
                                 lower_band: np.ndarray) -> pd.Series:
        """
        计算突破评分
        
        Args:
            close_prices: 收盘价数组
            upper_band: 上轨数组
            lower_band: 下轨数组
            
        Returns:
            pd.Series: 突破评分（-20到20分）
        """
        score = pd.Series(0.0, index=range(len(close_prices)))
        
        for i in range(1, len(close_prices)):
            if (np.isnan(close_prices[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
                np.isnan(close_prices[i-1]) or np.isnan(upper_band[i-1]) or np.isnan(lower_band[i-1])):
                continue
            
            current_price = close_prices[i]
            prev_price = close_prices[i-1]
            current_upper = upper_band[i]
            current_lower = lower_band[i]
            prev_upper = upper_band[i-1]
            prev_lower = lower_band[i-1]
            
            # 向上突破上轨
            if current_price > current_upper and prev_price <= prev_upper:
                # 计算突破强度
                if i >= 5:
                    recent_highs = close_prices[i-4:i]
                    if len(recent_highs) > 0 and current_price > np.max(recent_highs):
                        score.iloc[i] += 20  # 强势突破
                    else:
                        score.iloc[i] += 15  # 普通突破
                else:
                    score.iloc[i] += 15
            
            # 向下突破下轨
            elif current_price < current_lower and prev_price >= prev_lower:
                # 计算突破强度
                if i >= 5:
                    recent_lows = close_prices[i-4:i]
                    if len(recent_lows) > 0 and current_price < np.min(recent_lows):
                        score.iloc[i] -= 20  # 强势突破
                    else:
                        score.iloc[i] -= 15  # 普通突破
                else:
                    score.iloc[i] -= 15
            
            # 假突破回归
            elif (prev_price > prev_upper and current_price <= current_upper and 
                  current_price > (current_upper + current_lower) / 2):
                score.iloc[i] -= 10  # 假突破，减分
            elif (prev_price < prev_lower and current_price >= current_lower and 
                  current_price < (current_upper + current_lower) / 2):
                score.iloc[i] += 10  # 假跌破，加分
        
        return score
    
    def _calculate_regression_score(self, close_prices: np.ndarray, upper_band: np.ndarray, 
                                   middle_band: np.ndarray, lower_band: np.ndarray) -> pd.Series:
        """
        计算回归评分
        
        Args:
            close_prices: 收盘价数组
            upper_band: 上轨数组
            middle_band: 中轨数组
            lower_band: 下轨数组
            
        Returns:
            pd.Series: 回归评分（-15到15分）
        """
        score = pd.Series(0.0, index=range(len(close_prices)))
        
        for i in range(2, len(close_prices)):
            if (np.isnan(close_prices[i]) or np.isnan(upper_band[i]) or 
                np.isnan(middle_band[i]) or np.isnan(lower_band[i])):
                continue
            
            current_price = close_prices[i]
            prev_price = close_prices[i-1]
            prev2_price = close_prices[i-2]
            
            upper = upper_band[i]
            middle = middle_band[i]
            lower = lower_band[i]
            
            # 从上轨向中轨回归
            if (prev2_price >= upper_band[i-2] and prev_price < upper_band[i-1] and 
                current_price > middle and current_price < upper):
                # 计算回归强度
                regression_strength = (upper - current_price) / (upper - middle)
                score.iloc[i] += 10 + regression_strength * 5  # 最多15分
            
            # 从下轨向中轨回归
            elif (prev2_price <= lower_band[i-2] and prev_price > lower_band[i-1] and 
                  current_price < middle and current_price > lower):
                # 计算回归强度
                regression_strength = (current_price - lower) / (middle - lower)
                score.iloc[i] += 10 + regression_strength * 5  # 最多15分
            
            # 价格向中轨靠拢（均值回归）
            elif abs(current_price - middle) < abs(prev_price - middle):
                score.iloc[i] += 3  # 轻微加分
        
        return score
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别BOLL技术形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别出的形态列表
        """
        patterns = []
        
        # 计算布林带指标
        close_prices = data['close'].values
        middle_band = ma(close_prices, self.period)
        std_values = std(close_prices, self.period)
        upper_band = middle_band + (std_values * self.std_dev)
        lower_band = middle_band - (std_values * self.std_dev)
        
        close_series = pd.Series(close_prices, index=data.index)
        upper_series = pd.Series(upper_band, index=data.index)
        middle_series = pd.Series(middle_band, index=data.index)
        lower_series = pd.Series(lower_band, index=data.index)
        
        # 检测突破
        upper_breakout = (close_series > upper_series) & (close_series.shift(1) <= upper_series.shift(1))
        lower_breakout = (close_series < lower_series) & (close_series.shift(1) >= lower_series.shift(1))
        
        if upper_breakout.any():
            patterns.append("BOLL上轨突破")
        if lower_breakout.any():
            patterns.append("BOLL下轨突破")
        
        # 检测触及轨道
        touch_upper = (close_series >= upper_series * 0.99) & (close_series <= upper_series)
        touch_lower = (close_series <= lower_series * 1.01) & (close_series >= lower_series)
        
        if touch_upper.any():
            patterns.append("BOLL触及上轨")
        if touch_lower.any():
            patterns.append("BOLL触及下轨")
        
        # 检测带宽变化
        bandwidth = (upper_series - lower_series) / middle_series
        bandwidth_expanding = bandwidth > bandwidth.rolling(10).mean() * 1.2
        bandwidth_contracting = bandwidth < bandwidth.rolling(10).mean() * 0.8
        
        if bandwidth_expanding.any():
            patterns.append("BOLL带宽扩张")
        if bandwidth_contracting.any():
            patterns.append("BOLL带宽收缩")
        
        # 检测中轨支撑/阻力
        middle_support = (close_series.shift(1) < middle_series.shift(1)) & (close_series > middle_series)
        middle_resistance = (close_series.shift(1) > middle_series.shift(1)) & (close_series < middle_series)
        
        if middle_support.any():
            patterns.append("BOLL中轨支撑")
        if middle_resistance.any():
            patterns.append("BOLL中轨阻力")
        
        return patterns
    
    def generate_signals(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """
        生成BOLL交易信号
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            Dict[str, pd.Series]: 信号字典
        """
        # 计算布林带指标
        close_prices = data['close'].values
        middle_band = ma(close_prices, self.period)
        std_values = std(close_prices, self.period)
        upper_band = middle_band + (std_values * self.std_dev)
        lower_band = middle_band - (std_values * self.std_dev)
        
        close_series = pd.Series(close_prices, index=data.index)
        upper_series = pd.Series(upper_band, index=data.index)
        middle_series = pd.Series(middle_band, index=data.index)
        lower_series = pd.Series(lower_band, index=data.index)
        
        signals = {}
        
        # 基本位置信号
        signals['above_upper'] = close_series > upper_series
        signals['below_lower'] = close_series < lower_series
        signals['above_middle'] = close_series > middle_series
        signals['below_middle'] = close_series < middle_series
        
        # 突破信号
        signals['upper_breakout'] = (close_series > upper_series) & (close_series.shift(1) <= upper_series.shift(1))
        signals['lower_breakout'] = (close_series < lower_series) & (close_series.shift(1) >= lower_series.shift(1))
        
        # 回归信号
        signals['upper_regression'] = (close_series.shift(1) > upper_series.shift(1)) & (close_series <= upper_series)
        signals['lower_regression'] = (close_series.shift(1) < lower_series.shift(1)) & (close_series >= lower_series)
        
        # 中轨穿越信号
        signals['middle_cross_up'] = (close_series > middle_series) & (close_series.shift(1) <= middle_series.shift(1))
        signals['middle_cross_down'] = (close_series < middle_series) & (close_series.shift(1) >= middle_series.shift(1))
        
        # 带宽信号
        bandwidth = (upper_series - lower_series) / middle_series
        signals['bandwidth_expansion'] = bandwidth > bandwidth.rolling(10).mean() * 1.2
        signals['bandwidth_contraction'] = bandwidth < bandwidth.rolling(10).mean() * 0.8
        
        # 强势信号组合
        signals['strong_buy'] = (signals['lower_breakout'] | 
                                (signals['lower_regression'] & signals['bandwidth_contraction']))
        
        signals['strong_sell'] = (signals['upper_breakout'] & signals['bandwidth_expansion'])
        
        # 反转信号
        signals['oversold_bounce'] = (close_series < lower_series) & (close_series > close_series.shift(1))
        signals['overbought_decline'] = (close_series > upper_series) & (close_series < close_series.shift(1))
        
        return signals 