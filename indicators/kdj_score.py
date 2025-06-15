"""
KDJ评分指标

基于统一评分框架的KDJ指标，提供评分、形态识别和信号生成功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple

from indicators.scoring_framework import IndicatorScoreBase, PatternRecognitionMixin
from indicators.common import kdj as calc_kdj
from utils.logger import get_logger

logger = get_logger(__name__)


class KDJScore(IndicatorScoreBase, PatternRecognitionMixin):
    """
    KDJ评分指标
    
    基于KDJ指标的评分系统，包含形态识别和信号生成
    """
    
    # KDJ指标只需要high、low、close列
    REQUIRED_COLUMNS = ['high', 'low', 'close']

    def __init__(self, n: int = 9, m1: int = 3, m2: int = 3, weight: float = 1.0):
        """
        初始化KDJ评分指标
        
        Args:
            n: RSV周期
            m1: K值平滑因子
            m2: D值平滑因子
            weight: 指标权重
        """
        super().__init__(name="KDJ", weight=weight)
        self.n = n
        self.m1 = m1
        self.m2 = m2
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算KDJ原始评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 原始评分序列（0-100分）
        """
        # 确保数据包含必要的列
        if not all(col in data.columns for col in ['high', 'low', 'close']):
            raise ValueError(f"输入数据缺少必要的列：high, low, close")
            
        try:
            # 计算KDJ指标
            from utils.technical_utils import calculate_kdj
            k, d, j = calculate_kdj(
                data['high'],
                data['low'],
                data['close'],
                self.n,
                self.m1,
                self.m2
            )
        except ImportError:
            # 如果导入失败，使用内部实现
            logger.warning("无法导入calculate_kdj函数，使用common.kdj")
            try:
                k, d, j = calc_kdj(
                    data['close'].values,
                    data['high'].values,
                    data['low'].values,
                    self.n,
                    self.m1,
                    self.m2
                )
            except Exception as e:
                logger.error(f"计算KDJ指标失败: {e}")
                # 返回默认评分
                return pd.Series(50.0, index=data.index)
        
        # 初始化评分
        score = pd.Series(50.0, index=data.index)  # 基础分50分
        
        # 1. 位置评分（25分）
        position_score = self._calculate_position_score(k, d, j)
        score += position_score
        
        # 2. 交叉评分（20分）
        cross_score = self._calculate_cross_score(k, d, j)
        score += cross_score
        
        # 3. 趋势评分（15分）
        trend_score = self._calculate_trend_score(k, d, j)
        score += trend_score
        
        # 4. 形态评分（10分）
        pattern_score = self._calculate_pattern_score(k, d, j)
        score += pattern_score
        
        # 确保评分在0-100范围内
        return np.clip(score, 0, 100)
    
    def _calculate_position_score(self, k: np.ndarray, d: np.ndarray, 
                                 j: np.ndarray) -> pd.Series:
        """
        计算位置评分
        
        Args:
            k: K值
            d: D值
            j: J值
            
        Returns:
            pd.Series: 位置评分（-25到25分）
        """
        score = pd.Series(0.0, index=range(len(k)))
        
        for i in range(len(k)):
            if np.isnan(k[i]) or np.isnan(d[i]) or np.isnan(j[i]):
                continue
            
            # 超卖区域（0-20）
            if k[i] <= 20 and d[i] <= 20:
                score.iloc[i] += 20  # 超卖区域，看涨
            # 低位区域（20-30）
            elif k[i] <= 30 and d[i] <= 30:
                score.iloc[i] += 15
            # 中性区域（30-70）
            elif 30 < k[i] < 70 and 30 < d[i] < 70:
                score.iloc[i] += 0  # 中性
            # 高位区域（70-80）
            elif k[i] >= 70 and d[i] >= 70:
                score.iloc[i] -= 15
            # 超买区域（80-100）
            elif k[i] >= 80 and d[i] >= 80:
                score.iloc[i] -= 20  # 超买区域，看跌
            
            # J值位置调整
            if j[i] < 0:
                score.iloc[i] += 5  # J值小于0，极度超卖
            elif j[i] > 100:
                score.iloc[i] -= 5  # J值大于100，极度超买
        
        return score
    
    def _calculate_cross_score(self, k: np.ndarray, d: np.ndarray, 
                              j: np.ndarray) -> pd.Series:
        """
        计算交叉评分
        
        Args:
            k: K值
            d: D值
            j: J值
            
        Returns:
            pd.Series: 交叉评分（-20到20分）
        """
        score = pd.Series(0.0, index=range(len(k)))
        
        for i in range(1, len(k)):
            if (np.isnan(k[i]) or np.isnan(d[i]) or np.isnan(j[i]) or
                np.isnan(k[i-1]) or np.isnan(d[i-1]) or np.isnan(j[i-1])):
                continue
            
            # K线与D线交叉
            # 金叉（K上穿D）
            if k[i] > d[i] and k[i-1] <= d[i-1]:
                if k[i] < 30:  # 低位金叉
                    score.iloc[i] += 20
                elif k[i] < 50:  # 中低位金叉
                    score.iloc[i] += 15
                else:  # 高位金叉
                    score.iloc[i] += 10
            
            # 死叉（K下穿D）
            elif k[i] < d[i] and k[i-1] >= d[i-1]:
                if k[i] > 70:  # 高位死叉
                    score.iloc[i] -= 20
                elif k[i] > 50:  # 中高位死叉
                    score.iloc[i] -= 15
                else:  # 低位死叉
                    score.iloc[i] -= 10
            
            # J线与K线交叉
            # J上穿K
            if j[i] > k[i] and j[i-1] <= k[i-1]:
                score.iloc[i] += 5
            # J下穿K
            elif j[i] < k[i] and j[i-1] >= k[i-1]:
                score.iloc[i] -= 5
        
        return score
    
    def _calculate_trend_score(self, k: np.ndarray, d: np.ndarray, 
                              j: np.ndarray) -> pd.Series:
        """
        计算趋势评分
        
        Args:
            k: K值
            d: D值
            j: J值
            
        Returns:
            pd.Series: 趋势评分（-15到15分）
        """
        score = pd.Series(0.0, index=range(len(k)))
        
        for i in range(1, len(k)):
            if (np.isnan(k[i]) or np.isnan(d[i]) or np.isnan(j[i]) or
                np.isnan(k[i-1]) or np.isnan(d[i-1]) or np.isnan(j[i-1])):
                continue
            
            # K值趋势
            if k[i] > k[i-1]:
                score.iloc[i] += 5
            elif k[i] < k[i-1]:
                score.iloc[i] -= 5
            
            # D值趋势
            if d[i] > d[i-1]:
                score.iloc[i] += 5
            elif d[i] < d[i-1]:
                score.iloc[i] -= 5
            
            # J值趋势
            if j[i] > j[i-1]:
                score.iloc[i] += 5
            elif j[i] < j[i-1]:
                score.iloc[i] -= 5
        
        return score
    
    def _calculate_pattern_score(self, k: np.ndarray, d: np.ndarray, 
                                j: np.ndarray) -> pd.Series:
        """
        计算形态评分
        
        Args:
            k: K值
            d: D值
            j: J值
            
        Returns:
            pd.Series: 形态评分（-10到10分）
        """
        score = pd.Series(0.0, index=range(len(k)))
        
        for i in range(2, len(k)):
            if (np.isnan(k[i]) or np.isnan(d[i]) or np.isnan(j[i]) or
                np.isnan(k[i-1]) or np.isnan(d[i-1]) or np.isnan(j[i-1]) or
                np.isnan(k[i-2]) or np.isnan(d[i-2]) or np.isnan(j[i-2])):
                continue
            
            # 三线同向上
            if (k[i] > k[i-1] > k[i-2] and 
                d[i] > d[i-1] > d[i-2] and 
                j[i] > j[i-1] > j[i-2]):
                score.iloc[i] += 10
            
            # 三线同向下
            elif (k[i] < k[i-1] < k[i-2] and 
                  d[i] < d[i-1] < d[i-2] and 
                  j[i] < j[i-1] < j[i-2]):
                score.iloc[i] -= 10
            
            # KDJ顺序排列（多头排列）
            if j[i] > k[i] > d[i]:
                score.iloc[i] += 5
            # KDJ逆序排列（空头排列）
            elif j[i] < k[i] < d[i]:
                score.iloc[i] -= 5
        
        return score
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别KDJ技术形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别出的形态列表
        """
        patterns = []
        
        # 计算KDJ指标
        k, d, j = calc_kdj(
            data['close'].values,
            data['high'].values,
            data['low'].values,
            self.n,
            self.m1,
            self.m2
        )
        
        k_series = pd.Series(k, index=data.index)
        d_series = pd.Series(d, index=data.index)
        j_series = pd.Series(j, index=data.index)
        
        # 检测金叉死叉
        golden_cross = self.detect_golden_cross(k_series, d_series)
        death_cross = self.detect_death_cross(k_series, d_series)
        
        if golden_cross.any():
            patterns.append("KDJ金叉")
        if death_cross.any():
            patterns.append("KDJ死叉")
        
        # 检测超买超卖
        oversold = (k_series <= 20) & (d_series <= 20)
        overbought = (k_series >= 80) & (d_series >= 80)
        
        if oversold.any():
            patterns.append("KDJ超卖")
        if overbought.any():
            patterns.append("KDJ超买")
        
        # 检测J值极值
        j_extreme_low = j_series < 0
        j_extreme_high = j_series > 100
        
        if j_extreme_low.any():
            patterns.append("J值极度超卖")
        if j_extreme_high.any():
            patterns.append("J值极度超买")
        
        # 检测三线同向
        k_up = k_series > k_series.shift(1)
        d_up = d_series > d_series.shift(1)
        j_up = j_series > j_series.shift(1)
        
        three_line_up = k_up & d_up & j_up
        three_line_down = (~k_up) & (~d_up) & (~j_up)
        
        if three_line_up.any():
            patterns.append("KDJ三线同向上")
        if three_line_down.any():
            patterns.append("KDJ三线同向下")
        
        # 检测多头排列和空头排列
        bull_arrangement = (j_series > k_series) & (k_series > d_series)
        bear_arrangement = (j_series < k_series) & (k_series < d_series)
        
        if bull_arrangement.any():
            patterns.append("KDJ多头排列")
        if bear_arrangement.any():
            patterns.append("KDJ空头排列")
        
        return patterns
    
    def generate_signals(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """
        生成KDJ交易信号
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            Dict[str, pd.Series]: 信号字典
        """
        # 计算KDJ指标
        k, d, j = calc_kdj(
            data['close'].values,
            data['high'].values,
            data['low'].values,
            self.n,
            self.m1,
            self.m2
        )
        
        k_series = pd.Series(k, index=data.index)
        d_series = pd.Series(d, index=data.index)
        j_series = pd.Series(j, index=data.index)
        
        signals = {}
        
        # 基本交叉信号
        signals['golden_cross'] = self.detect_golden_cross(k_series, d_series)
        signals['death_cross'] = self.detect_death_cross(k_series, d_series)
        
        # 超买超卖信号
        signals['oversold'] = (k_series <= 20) & (d_series <= 20)
        signals['overbought'] = (k_series >= 80) & (d_series >= 80)
        
        # 低位金叉（强买入信号）
        signals['low_golden_cross'] = signals['golden_cross'] & (k_series < 30)
        
        # 高位死叉（强卖出信号）
        signals['high_death_cross'] = signals['death_cross'] & (k_series > 70)
        
        # J值极值信号
        signals['j_extreme_oversold'] = j_series < 0
        signals['j_extreme_overbought'] = j_series > 100
        
        # 三线同向信号
        k_up = k_series > k_series.shift(1)
        d_up = d_series > d_series.shift(1)
        j_up = j_series > j_series.shift(1)
        
        signals['three_line_up'] = k_up & d_up & j_up
        signals['three_line_down'] = (~k_up) & (~d_up) & (~j_up)
        
        # 排列信号
        signals['bull_arrangement'] = (j_series > k_series) & (k_series > d_series)
        signals['bear_arrangement'] = (j_series < k_series) & (k_series < d_series)
        
        # 反转信号（从超卖区域反弹）
        signals['oversold_reversal'] = (signals['oversold'].shift(1) & 
                                       (k_series > k_series.shift(1)) & 
                                       (d_series > d_series.shift(1)))
        
        # 顶部信号（从超买区域回落）
        signals['overbought_reversal'] = (signals['overbought'].shift(1) & 
                                         (k_series < k_series.shift(1)) & 
                                         (d_series < d_series.shift(1)))
        
        return signals 