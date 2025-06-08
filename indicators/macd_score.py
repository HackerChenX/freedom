"""
MACD评分指标

基于统一评分框架的MACD指标，提供评分、形态识别和信号生成功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple

from indicators.scoring_framework import IndicatorScoreBase, PatternRecognitionMixin
from indicators.common import macd as calc_macd
from utils.logger import get_logger

logger = get_logger(__name__)


class MACDScore(IndicatorScoreBase, PatternRecognitionMixin):
    """
    MACD评分指标
    
    基于MACD指标的评分系统，包含形态识别和信号生成
    """
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, 
                 signal_period: int = 9, weight: float = 1.0):
        """
        初始化MACD评分指标
        
        Args:
            fast_period: 快线周期
            slow_period: 慢线周期
            signal_period: 信号线周期
            weight: 指标权重
        """
        super().__init__(name="MACD", weight=weight)
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算MACD原始评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 原始评分序列（0-100分）
        """
        # 计算MACD指标
        dif, dea, macd_hist = calc_macd(
            data['close'].values, 
            self.fast_period, 
            self.slow_period, 
            self.signal_period
        )
        
        # 初始化评分
        score = pd.Series(50.0, index=data.index)  # 基础分50分
        
        # 1. 位置评分（20分）
        position_score = self._calculate_position_score(dif, dea, macd_hist)
        score += position_score
        
        # 2. 趋势评分（15分）
        trend_score = self._calculate_trend_score(dif, dea, macd_hist)
        score += trend_score
        
        # 3. 交叉评分（15分）
        cross_score = self._calculate_cross_score(dif, dea)
        score += cross_score
        
        # 4. 能量评分（10分）
        energy_score = self._calculate_energy_score(macd_hist)
        score += energy_score
        
        # 确保评分在0-100范围内
        return np.clip(score, 0, 100)
    
    def _calculate_position_score(self, dif: np.ndarray, dea: np.ndarray, 
                                 macd_hist: np.ndarray) -> pd.Series:
        """
        计算位置评分
        
        Args:
            dif: DIF线
            dea: DEA线
            macd_hist: MACD柱状图
            
        Returns:
            pd.Series: 位置评分（-20到20分）
        """
        score = pd.Series(0.0, index=range(len(dif)))
        
        for i in range(len(dif)):
            if np.isnan(dif[i]) or np.isnan(dea[i]):
                continue
                
            # DIF和DEA都在零轴上方
            if dif[i] > 0 and dea[i] > 0:
                score.iloc[i] += 15
            # DIF在零轴上方，DEA在零轴下方
            elif dif[i] > 0 and dea[i] <= 0:
                score.iloc[i] += 10
            # DIF在零轴下方，DEA在零轴上方
            elif dif[i] <= 0 and dea[i] > 0:
                score.iloc[i] += 5
            # DIF和DEA都在零轴下方
            else:
                score.iloc[i] -= 5
            
            # MACD柱状图位置加分
            if not np.isnan(macd_hist[i]):
                if macd_hist[i] > 0:
                    score.iloc[i] += 5
                else:
                    score.iloc[i] -= 5
        
        return score
    
    def _calculate_trend_score(self, dif: np.ndarray, dea: np.ndarray, 
                              macd_hist: np.ndarray) -> pd.Series:
        """
        计算趋势评分
        
        Args:
            dif: DIF线
            dea: DEA线
            macd_hist: MACD柱状图
            
        Returns:
            pd.Series: 趋势评分（-15到15分）
        """
        score = pd.Series(0.0, index=range(len(dif)))
        
        for i in range(1, len(dif)):
            if np.isnan(dif[i]) or np.isnan(dif[i-1]):
                continue
                
            # DIF线趋势
            if dif[i] > dif[i-1]:
                score.iloc[i] += 5
            elif dif[i] < dif[i-1]:
                score.iloc[i] -= 5
            
            # DEA线趋势
            if not np.isnan(dea[i]) and not np.isnan(dea[i-1]):
                if dea[i] > dea[i-1]:
                    score.iloc[i] += 5
                elif dea[i] < dea[i-1]:
                    score.iloc[i] -= 5
            
            # MACD柱状图趋势
            if (not np.isnan(macd_hist[i]) and not np.isnan(macd_hist[i-1])):
                if macd_hist[i] > macd_hist[i-1]:
                    score.iloc[i] += 5
                elif macd_hist[i] < macd_hist[i-1]:
                    score.iloc[i] -= 5
        
        return score
    
    def _calculate_cross_score(self, dif: np.ndarray, dea: np.ndarray) -> pd.Series:
        """
        计算交叉评分
        
        Args:
            dif: DIF线
            dea: DEA线
            
        Returns:
            pd.Series: 交叉评分（-15到15分）
        """
        score = pd.Series(0.0, index=range(len(dif)))
        
        for i in range(1, len(dif)):
            if (np.isnan(dif[i]) or np.isnan(dea[i]) or 
                np.isnan(dif[i-1]) or np.isnan(dea[i-1])):
                continue
            
            # 金叉
            if dif[i] > dea[i] and dif[i-1] <= dea[i-1]:
                # 零轴下方金叉
                if dif[i] < 0:
                    score.iloc[i] += 15
                # 零轴上方金叉
                else:
                    score.iloc[i] += 10
            
            # 死叉
            elif dif[i] < dea[i] and dif[i-1] >= dea[i-1]:
                # 零轴上方死叉
                if dif[i] > 0:
                    score.iloc[i] -= 15
                # 零轴下方死叉
                else:
                    score.iloc[i] -= 10
        
        return score
    
    def _calculate_energy_score(self, macd_hist: np.ndarray) -> pd.Series:
        """
        计算能量评分
        
        Args:
            macd_hist: MACD柱状图
            
        Returns:
            pd.Series: 能量评分（-10到10分）
        """
        score = pd.Series(0.0, index=range(len(macd_hist)))
        
        # 计算MACD柱状图的强度
        for i in range(5, len(macd_hist)):
            if np.isnan(macd_hist[i]):
                continue
            
            # 计算最近5期的平均值
            recent_avg = np.nanmean(macd_hist[i-4:i+1])
            
            # 当前值与平均值比较
            if abs(macd_hist[i]) > abs(recent_avg) * 1.2:
                if macd_hist[i] > 0:
                    score.iloc[i] += 10
                else:
                    score.iloc[i] -= 10
            elif abs(macd_hist[i]) > abs(recent_avg):
                if macd_hist[i] > 0:
                    score.iloc[i] += 5
                else:
                    score.iloc[i] -= 5
        
        return score
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别MACD技术形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别出的形态列表
        """
        patterns = []
        
        # 计算MACD指标
        dif, dea, macd_hist = calc_macd(
            data['close'].values, 
            self.fast_period, 
            self.slow_period, 
            self.signal_period
        )
        
        dif_series = pd.Series(dif, index=data.index)
        dea_series = pd.Series(dea, index=data.index)
        macd_series = pd.Series(macd_hist, index=data.index)
        
        # 检测金叉死叉
        golden_cross = self.detect_golden_cross(dif_series, dea_series)
        death_cross = self.detect_death_cross(dif_series, dea_series)
        
        if golden_cross.any():
            patterns.append("MACD金叉")
        if death_cross.any():
            patterns.append("MACD死叉")
        
        # 检测背离
        divergence = self.detect_divergence(data['close'], dif_series)
        if divergence['bottom_divergence'].any():
            patterns.append("MACD底背离")
        if divergence['top_divergence'].any():
            patterns.append("MACD顶背离")
        
        # 检测零轴穿越
        zero_cross_up = (dif_series > 0) & (dif_series.shift(1) <= 0)
        zero_cross_down = (dif_series < 0) & (dif_series.shift(1) >= 0)
        
        if zero_cross_up.any():
            patterns.append("MACD零轴上穿")
        if zero_cross_down.any():
            patterns.append("MACD零轴下穿")
        
        # 检测柱状图形态
        hist_increasing = (macd_series > macd_series.shift(1)) & (macd_series > 0)
        hist_decreasing = (macd_series < macd_series.shift(1)) & (macd_series < 0)
        
        if hist_increasing.any():
            patterns.append("MACD红柱扩大")
        if hist_decreasing.any():
            patterns.append("MACD绿柱扩大")
        
        return patterns
    
    def generate_signals(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """
        生成MACD交易信号
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            Dict[str, pd.Series]: 信号字典
        """
        # 计算MACD指标
        dif, dea, macd_hist = calc_macd(
            data['close'].values, 
            self.fast_period, 
            self.slow_period, 
            self.signal_period
        )
        
        dif_series = pd.Series(dif, index=data.index)
        dea_series = pd.Series(dea, index=data.index)
        macd_series = pd.Series(macd_hist, index=data.index)
        
        signals = {}
        
        # 基本交叉信号
        signals['golden_cross'] = self.detect_golden_cross(dif_series, dea_series)
        signals['death_cross'] = self.detect_death_cross(dif_series, dea_series)
        
        # 零轴穿越信号
        signals['zero_cross_up'] = (dif_series > 0) & (dif_series.shift(1) <= 0)
        signals['zero_cross_down'] = (dif_series < 0) & (dif_series.shift(1) >= 0)
        
        # 背离信号
        divergence = self.detect_divergence(data['close'], dif_series)
        signals['bottom_divergence'] = divergence['bottom_divergence']
        signals['top_divergence'] = divergence['top_divergence']
        
        # 强势信号（零轴上方金叉）
        signals['strong_buy'] = (signals['golden_cross'] & (dif_series > 0) & (dea_series > 0))
        
        # 弱势信号（零轴下方死叉）
        signals['strong_sell'] = (signals['death_cross'] & (dif_series < 0) & (dea_series < 0))
        
        # 柱状图信号
        signals['hist_turn_positive'] = (macd_series > 0) & (macd_series.shift(1) <= 0)
        signals['hist_turn_negative'] = (macd_series < 0) & (macd_series.shift(1) >= 0)
        
        return signals 