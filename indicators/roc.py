#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
变动率(ROC)

(当日收盘价-N日前收盘价)/N日前收盘价×100
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple

from indicators.base_indicator import BaseIndicator
from indicators.common import crossover, crossunder
from utils.logger import get_logger

logger = get_logger(__name__)


class ROC(BaseIndicator):
    """
    变动率(ROC) (ROC)
    
    分类：震荡类指标
    描述：(当日收盘价-N日前收盘价)/N日前收盘价×100
    """
    
    def __init__(self, period: int = 14, signal_period: int = 6):
        """
        初始化变动率(ROC)指标
        
        Args:
            period: 计算周期，默认为14
            signal_period: 信号线周期，默认为6
        """
        super().__init__()
        self.period = period
        self.signal_period = signal_period
        self.name = "ROC"
    
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
        计算ROC指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
                
        Returns:
            包含ROC指标的DataFrame
        """
        return self.calculate(df)
        
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算变动率(ROC)指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
                必须包含以下列：
                - close: 收盘价
                
        Returns:
            添加了ROC指标列的DataFrame
        """
        if df.empty:
            return df
            
        # 确保数据包含必要的列
        required_columns = ['close']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 实现变动率(ROC)计算逻辑
        # ROC = (当日收盘价-N日前收盘价)/N日前收盘价×100
        n_days_ago = df_copy['close'].shift(self.period)
        df_copy['roc'] = (df_copy['close'] - n_days_ago) / n_days_ago * 100
        
        # 计算ROC的移动平均作为信号线
        df_copy['signal'] = df_copy['roc'].rolling(window=self.signal_period).mean()
        
        # 存储结果
        self._result = df_copy[['roc', 'signal']]
        
        return df_copy
        
    def get_signals(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成变动率(ROC)指标交易信号
        
        Args:
            df: 包含价格数据和ROC指标的DataFrame
            **kwargs: 额外参数
                overbought: 超买阈值
                oversold: 超卖阈值
                
        Returns:
            添加了信号列的DataFrame:
            - roc_buy_signal: 1=买入信号, 0=无信号
            - roc_sell_signal: 1=卖出信号, 0=无信号
        """
        if df.empty:
            return df
            
        # 检查必要的指标列是否存在
        required_columns = ['roc', 'signal']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 获取参数
        overbought = kwargs.get('overbought', 8)  # 超买阈值
        oversold = kwargs.get('oversold', -8)  # 超卖阈值
        
        # 初始化信号列
        df_copy['roc_buy_signal'] = 0
        df_copy['roc_sell_signal'] = 0
        
        # ROC上穿信号线为买入信号
        for i in range(1, len(df_copy)):
            if df_copy['roc'].iloc[i-1] < df_copy['signal'].iloc[i-1] and \
               df_copy['roc'].iloc[i] > df_copy['signal'].iloc[i]:
                df_copy.iloc[i, df_copy.columns.get_loc('roc_buy_signal')] = 1
            
            # ROC下穿信号线为卖出信号
            elif df_copy['roc'].iloc[i-1] > df_copy['signal'].iloc[i-1] and \
                 df_copy['roc'].iloc[i] < df_copy['signal'].iloc[i]:
                df_copy.iloc[i, df_copy.columns.get_loc('roc_sell_signal')] = 1
        
        # 超卖反弹
        for i in range(1, len(df_copy)):
            if df_copy['roc'].iloc[i-1] < oversold and df_copy['roc'].iloc[i] > oversold:
                df_copy.iloc[i, df_copy.columns.get_loc('roc_buy_signal')] = 1
            
            # 超买回落
            elif df_copy['roc'].iloc[i-1] > overbought and df_copy['roc'].iloc[i] < overbought:
                df_copy.iloc[i, df_copy.columns.get_loc('roc_sell_signal')] = 1
        
        # ROC上穿0轴
        for i in range(1, len(df_copy)):
            if df_copy['roc'].iloc[i-1] < 0 and df_copy['roc'].iloc[i] > 0:
                df_copy.iloc[i, df_copy.columns.get_loc('roc_buy_signal')] = 1
            
            # ROC下穿0轴
            elif df_copy['roc'].iloc[i-1] > 0 and df_copy['roc'].iloc[i] < 0:
                df_copy.iloc[i, df_copy.columns.get_loc('roc_sell_signal')] = 1
        
        return df_copy
        
    def plot(self, df: pd.DataFrame, ax=None, **kwargs):
        """
        绘制变动率(ROC)指标图表
        
        Args:
            df: 包含ROC指标的DataFrame
            ax: matplotlib轴对象，如果为None则创建新的
            **kwargs: 额外绘图参数
            
        Returns:
            matplotlib轴对象
        """
        import matplotlib.pyplot as plt
        
        # 检查必要的指标列是否存在
        required_columns = ['roc', 'signal']
        self._validate_dataframe(df, required_columns)
        
        # 创建新的轴对象（如果未提供）
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
            
        # 绘制ROC指标线
        ax.plot(df.index, df['roc'], label='ROC')
        ax.plot(df.index, df['signal'], label='信号线', linestyle='--')
        
        # 添加参考线
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.axhline(y=8, color='r', linestyle='--', alpha=0.3, label='超买区域(8)')
        ax.axhline(y=-8, color='g', linestyle='--', alpha=0.3, label='超卖区域(-8)')
        
        ax.set_ylabel('变动率(ROC)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        return ax

    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算ROC原始评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 原始评分序列（0-100分）
        """
        # 确保已计算ROC
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)
        
        score = pd.Series(50.0, index=data.index)  # 基础分50分
        
        # 1. ROC零轴穿越评分
        zero_cross_score = self._calculate_roc_zero_cross_score()
        score += zero_cross_score
        
        # 2. ROC与信号线交叉评分
        signal_cross_score = self._calculate_roc_signal_cross_score()
        score += signal_cross_score
        
        # 3. ROC超买超卖评分
        overbought_oversold_score = self._calculate_roc_overbought_oversold_score()
        score += overbought_oversold_score
        
        # 4. ROC趋势评分
        trend_score = self._calculate_roc_trend_score()
        score += trend_score
        
        # 5. ROC强度评分
        strength_score = self._calculate_roc_strength_score()
        score += strength_score
        
        # 6. ROC背离评分
        divergence_score = self._calculate_roc_divergence_score(data)
        score += divergence_score
        
        return np.clip(score, 0, 100)
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别ROC技术形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别出的形态列表
        """
        patterns = []
        
        # 确保已计算ROC
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return patterns
        
        # 1. 检测ROC零轴穿越形态
        zero_cross_patterns = self._detect_roc_zero_cross_patterns()
        patterns.extend(zero_cross_patterns)
        
        # 2. 检测ROC与信号线交叉形态
        signal_cross_patterns = self._detect_roc_signal_cross_patterns()
        patterns.extend(signal_cross_patterns)
        
        # 3. 检测ROC超买超卖形态
        overbought_oversold_patterns = self._detect_roc_overbought_oversold_patterns()
        patterns.extend(overbought_oversold_patterns)
        
        # 4. 检测ROC趋势形态
        trend_patterns = self._detect_roc_trend_patterns()
        patterns.extend(trend_patterns)
        
        # 5. 检测ROC强度形态
        strength_patterns = self._detect_roc_strength_patterns()
        patterns.extend(strength_patterns)
        
        # 6. 检测ROC背离形态
        divergence_patterns = self._detect_roc_divergence_patterns(data)
        patterns.extend(divergence_patterns)
        
        return patterns
    
    def _calculate_roc_zero_cross_score(self) -> pd.Series:
        """
        计算ROC零轴穿越评分
        
        Returns:
            pd.Series: 零轴穿越评分
        """
        zero_cross_score = pd.Series(0.0, index=self._result.index)
        
        roc_values = self._result['roc']
        
        # ROC上穿零轴+25分
        roc_cross_up = crossover(roc_values, 0)
        zero_cross_score += roc_cross_up * 25
        
        # ROC下穿零轴-25分
        roc_cross_down = crossunder(roc_values, 0)
        zero_cross_score -= roc_cross_down * 25
        
        # ROC在零轴上方+8分
        roc_above_zero = roc_values > 0
        zero_cross_score += roc_above_zero * 8
        
        # ROC在零轴下方-8分
        roc_below_zero = roc_values < 0
        zero_cross_score -= roc_below_zero * 8
        
        return zero_cross_score
    
    def _calculate_roc_signal_cross_score(self) -> pd.Series:
        """
        计算ROC与信号线交叉评分
        
        Returns:
            pd.Series: 信号线交叉评分
        """
        signal_cross_score = pd.Series(0.0, index=self._result.index)
        
        roc_values = self._result['roc']
        signal_values = self._result['signal']
        
        # ROC上穿信号线+20分
        roc_cross_signal_up = crossover(roc_values, signal_values)
        signal_cross_score += roc_cross_signal_up * 20
        
        # ROC下穿信号线-20分
        roc_cross_signal_down = crossunder(roc_values, signal_values)
        signal_cross_score -= roc_cross_signal_down * 20
        
        # ROC在信号线上方+5分
        roc_above_signal = roc_values > signal_values
        signal_cross_score += roc_above_signal * 5
        
        # ROC在信号线下方-5分
        roc_below_signal = roc_values < signal_values
        signal_cross_score -= roc_below_signal * 5
        
        return signal_cross_score
    
    def _calculate_roc_overbought_oversold_score(self) -> pd.Series:
        """
        计算ROC超买超卖评分
        
        Returns:
            pd.Series: 超买超卖评分
        """
        overbought_oversold_score = pd.Series(0.0, index=self._result.index)
        
        roc_values = self._result['roc']
        
        # 定义超买超卖阈值
        overbought_threshold = 8
        oversold_threshold = -8
        extreme_overbought = 15
        extreme_oversold = -15
        
        # 超买区域-10分
        overbought_condition = roc_values > overbought_threshold
        overbought_oversold_score -= overbought_condition * 10
        
        # 极度超买-20分
        extreme_overbought_condition = roc_values > extreme_overbought
        overbought_oversold_score -= extreme_overbought_condition * 20
        
        # 超卖区域+10分
        oversold_condition = roc_values < oversold_threshold
        overbought_oversold_score += oversold_condition * 10
        
        # 极度超卖+20分
        extreme_oversold_condition = roc_values < extreme_oversold
        overbought_oversold_score += extreme_oversold_condition * 20
        
        # 从超买区域回落+15分
        overbought_exit = crossunder(roc_values, overbought_threshold)
        overbought_oversold_score += overbought_exit * 15
        
        # 从超卖区域反弹+15分
        oversold_exit = crossover(roc_values, oversold_threshold)
        overbought_oversold_score += oversold_exit * 15
        
        return overbought_oversold_score
    
    def _calculate_roc_trend_score(self) -> pd.Series:
        """
        计算ROC趋势评分
        
        Returns:
            pd.Series: 趋势评分
        """
        trend_score = pd.Series(0.0, index=self._result.index)
        
        roc_values = self._result['roc']
        signal_values = self._result['signal']
        
        # ROC上升趋势+12分
        roc_rising = roc_values > roc_values.shift(1)
        trend_score += roc_rising * 12
        
        # ROC下降趋势-12分
        roc_falling = roc_values < roc_values.shift(1)
        trend_score -= roc_falling * 12
        
        # 信号线上升趋势+8分
        signal_rising = signal_values > signal_values.shift(1)
        trend_score += signal_rising * 8
        
        # 信号线下降趋势-8分
        signal_falling = signal_values < signal_values.shift(1)
        trend_score -= signal_falling * 8
        
        # ROC加速上升+15分
        if len(roc_values) >= 3:
            roc_accelerating = (roc_values.diff() > roc_values.shift(1).diff())
            trend_score += roc_accelerating * 15
        
        # ROC加速下降-15分
        if len(roc_values) >= 3:
            roc_decelerating = (roc_values.diff() < roc_values.shift(1).diff())
            trend_score -= roc_decelerating * 15
        
        return trend_score
    
    def _calculate_roc_strength_score(self) -> pd.Series:
        """
        计算ROC强度评分
        
        Returns:
            pd.Series: 强度评分
        """
        strength_score = pd.Series(0.0, index=self._result.index)
        
        roc_values = self._result['roc']
        
        # 计算ROC的绝对值强度
        roc_abs = np.abs(roc_values)
        
        # 计算ROC强度的历史分位数
        if len(roc_abs) >= 20:
            rolling_window = min(20, len(roc_abs))
            
            # 计算滚动分位数
            roc_strength_percentile = roc_abs.rolling(rolling_window).apply(
                lambda x: (x.iloc[-1] >= x).sum() / len(x) * 100
            )
            
            # ROC强度处于高位（80%分位数以上）+10分
            high_strength = roc_strength_percentile >= 80
            strength_score += high_strength * 10
            
            # ROC强度处于极高位（95%分位数以上）额外+15分
            extreme_strength = roc_strength_percentile >= 95
            strength_score += extreme_strength * 15
            
            # ROC强度处于低位（20%分位数以下）-5分
            low_strength = roc_strength_percentile <= 20
            strength_score -= low_strength * 5
        
        return strength_score
    
    def _calculate_roc_divergence_score(self, data: pd.DataFrame) -> pd.Series:
        """
        计算ROC背离评分
        
        Args:
            data: 价格数据
            
        Returns:
            pd.Series: 背离评分
        """
        divergence_score = pd.Series(0.0, index=self._result.index)
        
        if 'close' not in data.columns:
            return divergence_score
        
        close_price = data['close']
        roc_values = self._result['roc']
        
        # 简化的背离检测
        if len(close_price) >= 10:
            # 检查最近10个周期的价格和ROC趋势
            recent_periods = 10
            recent_price = close_price.tail(recent_periods)
            recent_roc = roc_values.tail(recent_periods)
            
            # 计算价格和ROC的趋势方向
            price_trend = recent_price.iloc[-1] - recent_price.iloc[0]
            roc_trend = recent_roc.iloc[-1] - recent_roc.iloc[0]
            
            # 正背离：价格下跌但ROC上升
            if price_trend < 0 and roc_trend > 0:
                divergence_score.iloc[-1] += 20
            
            # 负背离：价格上涨但ROC下降
            elif price_trend > 0 and roc_trend < 0:
                divergence_score.iloc[-1] -= 20
        
        return divergence_score
    
    def _detect_roc_zero_cross_patterns(self) -> List[str]:
        """
        检测ROC零轴穿越形态
        
        Returns:
            List[str]: 零轴穿越形态列表
        """
        patterns = []
        
        roc_values = self._result['roc']
        
        # 检查最近的零轴穿越
        recent_periods = min(5, len(roc_values))
        recent_roc = roc_values.tail(recent_periods)
        
        if crossover(recent_roc, 0).any():
            patterns.append("ROC上穿零轴")
        
        if crossunder(recent_roc, 0).any():
            patterns.append("ROC下穿零轴")
        
        # 检查当前位置
        if len(roc_values) > 0:
            current_roc = roc_values.iloc[-1]
            if not pd.isna(current_roc):
                if current_roc > 0:
                    patterns.append("ROC零轴上方")
                elif current_roc < 0:
                    patterns.append("ROC零轴下方")
                else:
                    patterns.append("ROC零轴位置")
        
        return patterns
    
    def _detect_roc_signal_cross_patterns(self) -> List[str]:
        """
        检测ROC与信号线交叉形态
        
        Returns:
            List[str]: 信号线交叉形态列表
        """
        patterns = []
        
        roc_values = self._result['roc']
        signal_values = self._result['signal']
        
        # 检查最近的信号线交叉
        recent_periods = min(5, len(roc_values))
        recent_roc = roc_values.tail(recent_periods)
        recent_signal = signal_values.tail(recent_periods)
        
        if crossover(recent_roc, recent_signal).any():
            patterns.append("ROC上穿信号线")
        
        if crossunder(recent_roc, recent_signal).any():
            patterns.append("ROC下穿信号线")
        
        # 检查当前相对位置
        if len(roc_values) > 0 and len(signal_values) > 0:
            current_roc = roc_values.iloc[-1]
            current_signal = signal_values.iloc[-1]
            
            if not pd.isna(current_roc) and not pd.isna(current_signal):
                if current_roc > current_signal:
                    patterns.append("ROC信号线上方")
                elif current_roc < current_signal:
                    patterns.append("ROC信号线下方")
                else:
                    patterns.append("ROC信号线重合")
        
        return patterns
    
    def _detect_roc_overbought_oversold_patterns(self) -> List[str]:
        """
        检测ROC超买超卖形态
        
        Returns:
            List[str]: 超买超卖形态列表
        """
        patterns = []
        
        roc_values = self._result['roc']
        
        if len(roc_values) > 0:
            current_roc = roc_values.iloc[-1]
            
            if pd.isna(current_roc):
                return patterns
            
            # 超买超卖区域判断
            if current_roc > 15:
                patterns.append("ROC极度超买")
            elif current_roc > 8:
                patterns.append("ROC超买")
            elif current_roc < -15:
                patterns.append("ROC极度超卖")
            elif current_roc < -8:
                patterns.append("ROC超卖")
            else:
                patterns.append("ROC中性区域")
            
            # 检查最近的超买超卖穿越
            recent_periods = min(5, len(roc_values))
            recent_roc = roc_values.tail(recent_periods)
            
            if crossunder(recent_roc, 8).any():
                patterns.append("ROC脱离超买")
            
            if crossover(recent_roc, -8).any():
                patterns.append("ROC脱离超卖")
        
        return patterns
    
    def _detect_roc_trend_patterns(self) -> List[str]:
        """
        检测ROC趋势形态
        
        Returns:
            List[str]: 趋势形态列表
        """
        patterns = []
        
        roc_values = self._result['roc']
        signal_values = self._result['signal']
        
        # 检查ROC趋势
        if len(roc_values) >= 3:
            recent_3 = roc_values.tail(3)
            if len(recent_3) == 3 and not recent_3.isna().any():
                if (recent_3.iloc[2] > recent_3.iloc[1] > recent_3.iloc[0]):
                    patterns.append("ROC连续上升")
                elif (recent_3.iloc[2] < recent_3.iloc[1] < recent_3.iloc[0]):
                    patterns.append("ROC连续下降")
        
        # 检查当前趋势
        if len(roc_values) >= 2:
            current_roc = roc_values.iloc[-1]
            prev_roc = roc_values.iloc[-2]
            
            if not pd.isna(current_roc) and not pd.isna(prev_roc):
                if current_roc > prev_roc:
                    patterns.append("ROC上升")
                elif current_roc < prev_roc:
                    patterns.append("ROC下降")
                else:
                    patterns.append("ROC平稳")
        
        # 检查信号线趋势
        if len(signal_values) >= 2:
            current_signal = signal_values.iloc[-1]
            prev_signal = signal_values.iloc[-2]
            
            if not pd.isna(current_signal) and not pd.isna(prev_signal):
                if current_signal > prev_signal:
                    patterns.append("ROC信号线上升")
                elif current_signal < prev_signal:
                    patterns.append("ROC信号线下降")
        
        return patterns
    
    def _detect_roc_strength_patterns(self) -> List[str]:
        """
        检测ROC强度形态
        
        Returns:
            List[str]: 强度形态列表
        """
        patterns = []
        
        roc_values = self._result['roc']
        
        if len(roc_values) >= 20:
            current_roc = roc_values.iloc[-1]
            
            if pd.isna(current_roc):
                return patterns
            
            # 计算强度分位数
            recent_20 = np.abs(roc_values.tail(20))
            current_abs_roc = abs(current_roc)
            percentile = (current_abs_roc >= recent_20).sum() / len(recent_20) * 100
            
            if percentile >= 95:
                patterns.append("ROC极强变动")
            elif percentile >= 80:
                patterns.append("ROC强变动")
            elif percentile <= 20:
                patterns.append("ROC弱变动")
            else:
                patterns.append("ROC中等变动")
        
        return patterns
    
    def _detect_roc_divergence_patterns(self, data: pd.DataFrame) -> List[str]:
        """
        检测ROC背离形态
        
        Args:
            data: 价格数据
            
        Returns:
            List[str]: 背离形态列表
        """
        patterns = []
        
        if 'close' not in data.columns:
            return patterns
        
        close_price = data['close']
        roc_values = self._result['roc']
        
        if len(close_price) >= 10:
            # 检查最近10个周期的趋势
            recent_price = close_price.tail(10)
            recent_roc = roc_values.tail(10)
            
            price_trend = recent_price.iloc[-1] - recent_price.iloc[0]
            roc_trend = recent_roc.iloc[-1] - recent_roc.iloc[0]
            
            # 背离检测
            if price_trend < -0.01 and roc_trend > 0.01:  # 价格下跌但ROC上升
                patterns.append("ROC正背离")
            elif price_trend > 0.01 and roc_trend < -0.01:  # 价格上涨但ROC下降
                patterns.append("ROC负背离")
            elif abs(price_trend) < 0.01 and abs(roc_trend) < 0.01:
                patterns.append("ROC价格同步")
        
        return patterns

