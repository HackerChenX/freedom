#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
动量指标(MTM)

当日收盘价与N日前收盘价之差
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple

from indicators.base_indicator import BaseIndicator
from indicators.common import crossover, crossunder
from utils.logger import get_logger

logger = get_logger(__name__)


class MTM(BaseIndicator):
    """
    动量指标(MTM) (MTM)
    
    分类：震荡类指标
    描述：当日收盘价与N日前收盘价之差
    """
    
    def __init__(self, period: int = 14, signal_period: int = 6):
        """
        初始化动量指标(MTM)指标
        
        Args:
            period: 计算周期，默认为14
            signal_period: 信号线周期，默认为6
        """
        super().__init__()
        self.period = period
        self.signal_period = signal_period
        self.name = "MTM"
    
    def _validate_dataframe(self, df: pd.DataFrame, required_columns: List[str]) -> None:
        """
        验证DataFrame是否包含所需的列
        
        Args:
            df: 输入数据
            required_columns: 所需的列名列表
            
        Raises:
            ValueError: 如果缺少必要的列
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"输入数据缺少必要的列: {', '.join(missing_columns)}")
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算MTM指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
                
        Returns:
            包含MTM指标的DataFrame
        """
        return self.calculate(df)
        
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算动量指标(MTM)指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
                必须包含以下列：
                - close: 收盘价
                
        Returns:
            添加了MTM指标列的DataFrame
        """
        if df.empty:
            return df
            
        # 确保数据包含必要的列
        required_columns = ['close']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 实现动量指标(MTM)计算逻辑
        # MTM = 当日收盘价 - N日前的收盘价
        df_copy['mtm'] = df_copy['close'] - df_copy['close'].shift(self.period)
        
        # 计算MTM的移动平均线作为信号线
        df_copy['signal'] = df_copy['mtm'].rolling(window=self.signal_period).mean()
        
        # 存储结果
        self._result = df_copy[['mtm', 'signal']]
        
        return df_copy
        
    def get_signals(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成动量指标(MTM)指标交易信号
        
        Args:
            df: 包含价格数据和MTM指标的DataFrame
            **kwargs: 额外参数
                
        Returns:
            添加了信号列的DataFrame:
            - mtm_buy_signal: 1=买入信号, 0=无信号
            - mtm_sell_signal: 1=卖出信号, 0=无信号
        """
        if df.empty:
            return df
            
        # 检查必要的指标列是否存在
        required_columns = ['mtm', 'signal']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 初始化信号列
        df_copy['mtm_buy_signal'] = 0
        df_copy['mtm_sell_signal'] = 0
        
        # MTM上穿信号线为买入信号
        for i in range(1, len(df_copy)):
            if df_copy['mtm'].iloc[i-1] < df_copy['signal'].iloc[i-1] and \
               df_copy['mtm'].iloc[i] > df_copy['signal'].iloc[i]:
                df_copy.iloc[i, df_copy.columns.get_loc('mtm_buy_signal')] = 1
            
            # MTM下穿信号线为卖出信号
            elif df_copy['mtm'].iloc[i-1] > df_copy['signal'].iloc[i-1] and \
                 df_copy['mtm'].iloc[i] < df_copy['signal'].iloc[i]:
                df_copy.iloc[i, df_copy.columns.get_loc('mtm_sell_signal')] = 1
                
        # MTM上穿0轴
        for i in range(1, len(df_copy)):
            if df_copy['mtm'].iloc[i-1] < 0 and df_copy['mtm'].iloc[i] > 0:
                df_copy.iloc[i, df_copy.columns.get_loc('mtm_buy_signal')] = 1
            
            # MTM下穿0轴
            elif df_copy['mtm'].iloc[i-1] > 0 and df_copy['mtm'].iloc[i] < 0:
                df_copy.iloc[i, df_copy.columns.get_loc('mtm_sell_signal')] = 1
        
        return df_copy
        
    def plot(self, df: pd.DataFrame, ax=None, **kwargs):
        """
        绘制动量指标(MTM)指标图表
        
        Args:
            df: 包含MTM指标的DataFrame
            ax: matplotlib轴对象，如果为None则创建新的
            **kwargs: 额外绘图参数
            
        Returns:
            matplotlib轴对象
        """
        import matplotlib.pyplot as plt
        
        # 检查必要的指标列是否存在
        required_columns = ['mtm', 'signal']
        self._validate_dataframe(df, required_columns)
        
        # 创建新的轴对象（如果未提供）
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
            
        # 绘制MTM指标线
        ax.plot(df.index, df['mtm'], label='MTM')
        ax.plot(df.index, df['signal'], label='信号线', linestyle='--')
        
        # 添加零轴线
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        ax.set_ylabel('动量指标(MTM)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        return ax

    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算MTM原始评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 原始评分序列（0-100分）
        """
        # 确保已计算MTM
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)
        
        score = pd.Series(50.0, index=data.index)  # 基础分50分
        
        # 1. MTM零轴穿越评分
        zero_cross_score = self._calculate_mtm_zero_cross_score()
        score += zero_cross_score
        
        # 2. MTM与信号线交叉评分
        signal_cross_score = self._calculate_mtm_signal_cross_score()
        score += signal_cross_score
        
        # 3. MTM趋势评分
        trend_score = self._calculate_mtm_trend_score()
        score += trend_score
        
        # 4. MTM强度评分
        strength_score = self._calculate_mtm_strength_score()
        score += strength_score
        
        # 5. MTM背离评分
        divergence_score = self._calculate_mtm_divergence_score(data)
        score += divergence_score
        
        return np.clip(score, 0, 100)
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别MTM技术形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别出的形态列表
        """
        patterns = []
        
        # 确保已计算MTM
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return patterns
        
        # 1. 检测MTM零轴穿越形态
        zero_cross_patterns = self._detect_mtm_zero_cross_patterns()
        patterns.extend(zero_cross_patterns)
        
        # 2. 检测MTM与信号线交叉形态
        signal_cross_patterns = self._detect_mtm_signal_cross_patterns()
        patterns.extend(signal_cross_patterns)
        
        # 3. 检测MTM趋势形态
        trend_patterns = self._detect_mtm_trend_patterns()
        patterns.extend(trend_patterns)
        
        # 4. 检测MTM强度形态
        strength_patterns = self._detect_mtm_strength_patterns()
        patterns.extend(strength_patterns)
        
        # 5. 检测MTM背离形态
        divergence_patterns = self._detect_mtm_divergence_patterns(data)
        patterns.extend(divergence_patterns)
        
        return patterns
    
    def _calculate_mtm_zero_cross_score(self) -> pd.Series:
        """
        计算MTM零轴穿越评分
        
        Returns:
            pd.Series: 零轴穿越评分
        """
        zero_cross_score = pd.Series(0.0, index=self._result.index)
        
        mtm_values = self._result['mtm']
        
        # MTM上穿零轴+25分
        mtm_cross_up = crossover(mtm_values, 0)
        zero_cross_score += mtm_cross_up * 25
        
        # MTM下穿零轴-25分
        mtm_cross_down = crossunder(mtm_values, 0)
        zero_cross_score -= mtm_cross_down * 25
        
        # MTM在零轴上方+8分
        mtm_above_zero = mtm_values > 0
        zero_cross_score += mtm_above_zero * 8
        
        # MTM在零轴下方-8分
        mtm_below_zero = mtm_values < 0
        zero_cross_score -= mtm_below_zero * 8
        
        return zero_cross_score
    
    def _calculate_mtm_signal_cross_score(self) -> pd.Series:
        """
        计算MTM与信号线交叉评分
        
        Returns:
            pd.Series: 信号线交叉评分
        """
        signal_cross_score = pd.Series(0.0, index=self._result.index)
        
        mtm_values = self._result['mtm']
        signal_values = self._result['signal']
        
        # MTM上穿信号线+20分
        mtm_cross_signal_up = crossover(mtm_values, signal_values)
        signal_cross_score += mtm_cross_signal_up * 20
        
        # MTM下穿信号线-20分
        mtm_cross_signal_down = crossunder(mtm_values, signal_values)
        signal_cross_score -= mtm_cross_signal_down * 20
        
        # MTM在信号线上方+5分
        mtm_above_signal = mtm_values > signal_values
        signal_cross_score += mtm_above_signal * 5
        
        # MTM在信号线下方-5分
        mtm_below_signal = mtm_values < signal_values
        signal_cross_score -= mtm_below_signal * 5
        
        return signal_cross_score
    
    def _calculate_mtm_trend_score(self) -> pd.Series:
        """
        计算MTM趋势评分
        
        Returns:
            pd.Series: 趋势评分
        """
        trend_score = pd.Series(0.0, index=self._result.index)
        
        mtm_values = self._result['mtm']
        signal_values = self._result['signal']
        
        # MTM上升趋势+12分
        mtm_rising = mtm_values > mtm_values.shift(1)
        trend_score += mtm_rising * 12
        
        # MTM下降趋势-12分
        mtm_falling = mtm_values < mtm_values.shift(1)
        trend_score -= mtm_falling * 12
        
        # 信号线上升趋势+8分
        signal_rising = signal_values > signal_values.shift(1)
        trend_score += signal_rising * 8
        
        # 信号线下降趋势-8分
        signal_falling = signal_values < signal_values.shift(1)
        trend_score -= signal_falling * 8
        
        # MTM加速上升+15分
        if len(mtm_values) >= 3:
            mtm_accelerating = (mtm_values.diff() > mtm_values.shift(1).diff())
            trend_score += mtm_accelerating * 15
        
        # MTM加速下降-15分
        if len(mtm_values) >= 3:
            mtm_decelerating = (mtm_values.diff() < mtm_values.shift(1).diff())
            trend_score -= mtm_decelerating * 15
        
        return trend_score
    
    def _calculate_mtm_strength_score(self) -> pd.Series:
        """
        计算MTM强度评分
        
        Returns:
            pd.Series: 强度评分
        """
        strength_score = pd.Series(0.0, index=self._result.index)
        
        mtm_values = self._result['mtm']
        
        # 计算MTM的绝对值强度
        mtm_abs = np.abs(mtm_values)
        
        # 计算MTM强度的历史分位数
        if len(mtm_abs) >= 20:
            rolling_window = min(20, len(mtm_abs))
            
            # 计算滚动分位数
            mtm_strength_percentile = mtm_abs.rolling(rolling_window).apply(
                lambda x: (x.iloc[-1] >= x).sum() / len(x) * 100
            )
            
            # MTM强度处于高位（80%分位数以上）+10分
            high_strength = mtm_strength_percentile >= 80
            strength_score += high_strength * 10
            
            # MTM强度处于极高位（95%分位数以上）额外+15分
            extreme_strength = mtm_strength_percentile >= 95
            strength_score += extreme_strength * 15
            
            # MTM强度处于低位（20%分位数以下）-5分
            low_strength = mtm_strength_percentile <= 20
            strength_score -= low_strength * 5
        
        return strength_score
    
    def _calculate_mtm_divergence_score(self, data: pd.DataFrame) -> pd.Series:
        """
        计算MTM背离评分
        
        Args:
            data: 价格数据
            
        Returns:
            pd.Series: 背离评分
        """
        divergence_score = pd.Series(0.0, index=self._result.index)
        
        if 'close' not in data.columns:
            return divergence_score
        
        close_price = data['close']
        mtm_values = self._result['mtm']
        
        # 简化的背离检测
        if len(close_price) >= 10:
            # 检查最近10个周期的价格和MTM趋势
            recent_periods = 10
            recent_price = close_price.tail(recent_periods)
            recent_mtm = mtm_values.tail(recent_periods)
            
            # 计算价格和MTM的趋势方向
            price_trend = recent_price.iloc[-1] - recent_price.iloc[0]
            mtm_trend = recent_mtm.iloc[-1] - recent_mtm.iloc[0]
            
            # 正背离：价格下跌但MTM上升
            if price_trend < 0 and mtm_trend > 0:
                divergence_score.iloc[-1] += 20
            
            # 负背离：价格上涨但MTM下降
            elif price_trend > 0 and mtm_trend < 0:
                divergence_score.iloc[-1] -= 20
        
        return divergence_score
    
    def _detect_mtm_zero_cross_patterns(self) -> List[str]:
        """
        检测MTM零轴穿越形态
        
        Returns:
            List[str]: 零轴穿越形态列表
        """
        patterns = []
        
        mtm_values = self._result['mtm']
        
        # 检查最近的零轴穿越
        recent_periods = min(5, len(mtm_values))
        recent_mtm = mtm_values.tail(recent_periods)
        
        if crossover(recent_mtm, 0).any():
            patterns.append("MTM上穿零轴")
        
        if crossunder(recent_mtm, 0).any():
            patterns.append("MTM下穿零轴")
        
        # 检查当前位置
        if len(mtm_values) > 0:
            current_mtm = mtm_values.iloc[-1]
            if not pd.isna(current_mtm):
                if current_mtm > 0:
                    patterns.append("MTM零轴上方")
                elif current_mtm < 0:
                    patterns.append("MTM零轴下方")
                else:
                    patterns.append("MTM零轴位置")
        
        return patterns
    
    def _detect_mtm_signal_cross_patterns(self) -> List[str]:
        """
        检测MTM与信号线交叉形态
        
        Returns:
            List[str]: 信号线交叉形态列表
        """
        patterns = []
        
        mtm_values = self._result['mtm']
        signal_values = self._result['signal']
        
        # 检查最近的信号线交叉
        recent_periods = min(5, len(mtm_values))
        recent_mtm = mtm_values.tail(recent_periods)
        recent_signal = signal_values.tail(recent_periods)
        
        if crossover(recent_mtm, recent_signal).any():
            patterns.append("MTM上穿信号线")
        
        if crossunder(recent_mtm, recent_signal).any():
            patterns.append("MTM下穿信号线")
        
        # 检查当前相对位置
        if len(mtm_values) > 0 and len(signal_values) > 0:
            current_mtm = mtm_values.iloc[-1]
            current_signal = signal_values.iloc[-1]
            
            if not pd.isna(current_mtm) and not pd.isna(current_signal):
                if current_mtm > current_signal:
                    patterns.append("MTM信号线上方")
                elif current_mtm < current_signal:
                    patterns.append("MTM信号线下方")
                else:
                    patterns.append("MTM信号线重合")
        
        return patterns
    
    def _detect_mtm_trend_patterns(self) -> List[str]:
        """
        检测MTM趋势形态
        
        Returns:
            List[str]: 趋势形态列表
        """
        patterns = []
        
        mtm_values = self._result['mtm']
        signal_values = self._result['signal']
        
        # 检查MTM趋势
        if len(mtm_values) >= 3:
            recent_3 = mtm_values.tail(3)
            if len(recent_3) == 3 and not recent_3.isna().any():
                if (recent_3.iloc[2] > recent_3.iloc[1] > recent_3.iloc[0]):
                    patterns.append("MTM连续上升")
                elif (recent_3.iloc[2] < recent_3.iloc[1] < recent_3.iloc[0]):
                    patterns.append("MTM连续下降")
        
        # 检查当前趋势
        if len(mtm_values) >= 2:
            current_mtm = mtm_values.iloc[-1]
            prev_mtm = mtm_values.iloc[-2]
            
            if not pd.isna(current_mtm) and not pd.isna(prev_mtm):
                if current_mtm > prev_mtm:
                    patterns.append("MTM上升")
                elif current_mtm < prev_mtm:
                    patterns.append("MTM下降")
                else:
                    patterns.append("MTM平稳")
        
        # 检查信号线趋势
        if len(signal_values) >= 2:
            current_signal = signal_values.iloc[-1]
            prev_signal = signal_values.iloc[-2]
            
            if not pd.isna(current_signal) and not pd.isna(prev_signal):
                if current_signal > prev_signal:
                    patterns.append("MTM信号线上升")
                elif current_signal < prev_signal:
                    patterns.append("MTM信号线下降")
        
        return patterns
    
    def _detect_mtm_strength_patterns(self) -> List[str]:
        """
        检测MTM强度形态
        
        Returns:
            List[str]: 强度形态列表
        """
        patterns = []
        
        mtm_values = self._result['mtm']
        
        if len(mtm_values) >= 20:
            current_mtm = mtm_values.iloc[-1]
            
            if pd.isna(current_mtm):
                return patterns
            
            # 计算强度分位数
            recent_20 = np.abs(mtm_values.tail(20))
            current_abs_mtm = abs(current_mtm)
            percentile = (current_abs_mtm >= recent_20).sum() / len(recent_20) * 100
            
            if percentile >= 95:
                patterns.append("MTM极强动量")
            elif percentile >= 80:
                patterns.append("MTM强动量")
            elif percentile <= 20:
                patterns.append("MTM弱动量")
            else:
                patterns.append("MTM中等动量")
        
        return patterns
    
    def _detect_mtm_divergence_patterns(self, data: pd.DataFrame) -> List[str]:
        """
        检测MTM背离形态
        
        Args:
            data: 价格数据
            
        Returns:
            List[str]: 背离形态列表
        """
        patterns = []
        
        if 'close' not in data.columns:
            return patterns
        
        close_price = data['close']
        mtm_values = self._result['mtm']
        
        if len(close_price) >= 10:
            # 检查最近10个周期的趋势
            recent_price = close_price.tail(10)
            recent_mtm = mtm_values.tail(10)
            
            price_trend = recent_price.iloc[-1] - recent_price.iloc[0]
            mtm_trend = recent_mtm.iloc[-1] - recent_mtm.iloc[0]
            
            # 背离检测
            if price_trend < -0.01 and mtm_trend > 0.01:  # 价格下跌但MTM上升
                patterns.append("MTM正背离")
            elif price_trend > 0.01 and mtm_trend < -0.01:  # 价格上涨但MTM下降
                patterns.append("MTM负背离")
            elif abs(price_trend) < 0.01 and abs(mtm_trend) < 0.01:
                patterns.append("MTM价格同步")
        
        return patterns

