#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
均线多空指标(BIAS)

(收盘价-MA)/MA×100%
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple

from indicators.base_indicator import BaseIndicator
from indicators.common import crossover, crossunder
from utils.logger import get_logger

logger = get_logger(__name__)


class BIAS(BaseIndicator):
    """
    均线多空指标(BIAS) (BIAS)
    
    分类：趋势类指标
    描述：(收盘价-MA)/MA×100%
    """
    
    def __init__(self, period: int = 14, periods: List[int] = None):
        """
        初始化均线多空指标(BIAS)指标
        
        Args:
            period: 计算周期，默认为14
            periods: 多个计算周期，如果提供，将计算多个周期的BIAS
        """
        super().__init__()
        self.period = period
        self.periods = periods if periods is not None else [period]
        self.name = "BIAS"
        
    def _validate_dataframe(self, df: pd.DataFrame, required_columns: List[str]) -> None:
        """
        验证DataFrame是否包含所需的列
        
        Args:
            df: 要验证的DataFrame
            required_columns: 所需的列名列表
            
        Raises:
            ValueError: 如果缺少必要的列
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"DataFrame缺少必要的列: {', '.join(missing_columns)}")
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算均线多空指标(BIAS)指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
                必须包含以下列：
                - close: 收盘价
                - high: 最高价
                - low: 最低价
                
        Returns:
            添加了BIAS指标列的DataFrame
        """
        if df.empty:
            return df
            
        # 确保数据包含必要的列
        required_columns = ['close', 'high', 'low']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 计算不同周期的BIAS
        for p in self.periods:
            # 计算移动平均线
            ma = df_copy['close'].rolling(window=p).mean()
            
            # 计算BIAS
            df_copy[f'BIAS{p}'] = (df_copy['close'] - ma) / ma * 100
        
        # 存储结果
        self._result = df_copy[[f'BIAS{p}' for p in self.periods]]
        
        return df_copy
        
    def get_signals(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成均线多空指标(BIAS)指标交易信号
        
        Args:
            df: 包含价格数据和BIAS指标的DataFrame
            **kwargs: 额外参数
                overbought: 超买阈值
                oversold: 超卖阈值
                
        Returns:
            添加了信号列的DataFrame:
            - bias_signal: 1=买入信号, -1=卖出信号, 0=无信号
        """
        if df.empty:
            return df
        
        df_copy = df.copy()
        
        # 初始化信号列
        df_copy[f'bias_signal'] = 0
        
        # 获取参数
        overbought = kwargs.get('overbought', 6)  # 超买阈值
        oversold = kwargs.get('oversold', -6)  # 超卖阈值
        
        # 使用主要周期的BIAS生成信号
        main_period = self.periods[0]
        bias_col = f'BIAS{main_period}'
        
        # 检查必要的指标列是否存在
        if bias_col in df_copy.columns:
            # 超卖区域上穿信号线为买入信号
            df_copy.loc[crossover(df_copy[bias_col], oversold), f'bias_signal'] = 1
            
            # 超买区域下穿信号线为卖出信号
            df_copy.loc[crossunder(df_copy[bias_col], overbought), f'bias_signal'] = -1
        
        return df_copy
        
    def plot(self, df: pd.DataFrame, ax=None, **kwargs):
        """
        绘制均线多空指标(BIAS)指标图表
        
        Args:
            df: 包含BIAS指标的DataFrame
            ax: matplotlib轴对象，如果为None则创建新的
            **kwargs: 额外绘图参数
            
        Returns:
            matplotlib轴对象
        """
        import matplotlib.pyplot as plt
        
        # 创建新的轴对象（如果未提供）
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        
        # 绘制各个周期的BIAS指标线
        colors = ['blue', 'red', 'green', 'purple', 'orange']
        for i, p in enumerate(self.periods):
            bias_col = f'BIAS{p}'
            if bias_col in df.columns:
                color = colors[i % len(colors)]
                ax.plot(df.index, df[bias_col], label=f'BIAS({p})', color=color)
        
        # 添加超买超卖参考线
        overbought = kwargs.get('overbought', 6)
        oversold = kwargs.get('oversold', -6)
        ax.axhline(y=overbought, color='r', linestyle='--', alpha=0.3)
        ax.axhline(y=oversold, color='g', linestyle='--', alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
        
        ax.set_ylabel(f'均线多空指标(BIAS)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        return ax
        
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算指标并返回结果
        
        Args:
            df: 输入DataFrame
            
        Returns:
            包含计算结果的DataFrame
        """
        return self.calculate(df)

    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算BIAS原始评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 原始评分序列（0-100分）
        """
        # 确保已计算BIAS
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)
        
        score = pd.Series(50.0, index=data.index)  # 基础分50分
        
        # 1. BIAS超买超卖评分
        overbought_oversold_score = self._calculate_bias_overbought_oversold_score(data)
        score += overbought_oversold_score
        
        # 2. BIAS回归评分
        regression_score = self._calculate_bias_regression_score()
        score += regression_score
        
        # 3. BIAS极值评分
        extreme_score = self._calculate_bias_extreme_score()
        score += extreme_score
        
        # 4. BIAS趋势评分
        trend_score = self._calculate_bias_trend_score()
        score += trend_score
        
        # 5. BIAS零轴穿越评分
        zero_cross_score = self._calculate_bias_zero_cross_score()
        score += zero_cross_score
        
        # 6. 多周期协同评估（优化点）
        if len(self.periods) > 1:
            multi_period_score = self._calculate_multi_period_consistency_score()
            score += multi_period_score
        
        return np.clip(score, 0, 100)
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别BIAS技术形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别出的形态列表
        """
        patterns = []
        
        # 确保已计算BIAS
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return patterns
        
        # 1. 检测BIAS超买超卖形态
        overbought_oversold_patterns = self._detect_bias_overbought_oversold_patterns()
        patterns.extend(overbought_oversold_patterns)
        
        # 2. 检测BIAS回归形态
        regression_patterns = self._detect_bias_regression_patterns()
        patterns.extend(regression_patterns)
        
        # 3. 检测BIAS极值形态
        extreme_patterns = self._detect_bias_extreme_patterns()
        patterns.extend(extreme_patterns)
        
        # 4. 检测BIAS趋势形态
        trend_patterns = self._detect_bias_trend_patterns()
        patterns.extend(trend_patterns)
        
        # 5. 检测BIAS零轴穿越形态
        zero_cross_patterns = self._detect_bias_zero_cross_patterns()
        patterns.extend(zero_cross_patterns)
        
        return patterns
    
    def _calculate_bias_overbought_oversold_score(self, data: pd.DataFrame = None) -> pd.Series:
        """
        计算BIAS超买超卖评分
        
        Args:
            data: 输入数据，用于计算市场波动率（优化点）
            
        Returns:
            pd.Series: 超买超卖评分序列
        """
        if self._result is None:
            return pd.Series(0.0)
        
        score = pd.Series(0.0, index=self._result.index)
        
        # 获取主要周期的BIAS
        main_period = self.periods[0]
        bias_col = f'BIAS{main_period}'
        
        if bias_col not in self._result.columns:
            return score
        
        bias = self._result[bias_col]
        
        # 优化点：市场环境自适应阈值
        # 根据波动率动态调整超买超卖阈值
        if data is not None and len(data) >= 60:
            # 计算价格的20日波动率
            if 'close' in data.columns:
                price_volatility = data['close'].pct_change().rolling(window=20).std()
                
                # 计算波动率相对于近60日平均波动率的比例
                volatility_ratio = price_volatility / price_volatility.rolling(window=60).mean()
                volatility_ratio = volatility_ratio.fillna(1.0)
                
                # 根据波动率调整阈值
                overbought_threshold = 6.0 * np.clip(volatility_ratio, 0.8, 2.0)  # 高波动率时放宽阈值
                oversold_threshold = -6.0 * np.clip(volatility_ratio, 0.8, 2.0)   # 低波动率时收紧阈值
            else:
                # 无法获取价格数据时使用默认阈值
                overbought_threshold = pd.Series(6.0, index=bias.index)
                oversold_threshold = pd.Series(-6.0, index=bias.index)
        else:
            # 数据不足时使用默认阈值
            overbought_threshold = pd.Series(6.0, index=bias.index)
            oversold_threshold = pd.Series(-6.0, index=bias.index)
        
        # 超买区
        overbought = bias > overbought_threshold
        extreme_overbought = bias > (overbought_threshold * 1.5)
        
        # 超卖区
        oversold = bias < oversold_threshold
        extreme_oversold = bias < (oversold_threshold * 1.5)
        
        # 超买区评分
        score -= overbought * 15  # 超买 -15分
        score -= extreme_overbought * 10  # 极度超买 额外 -10分
        
        # 超卖区评分
        score += oversold * 15  # 超卖 +15分
        score += extreme_oversold * 10  # 极度超卖 额外 +10分
        
        return score
    
    def _calculate_bias_regression_score(self) -> pd.Series:
        """
        计算BIAS回归评分
        
        Returns:
            pd.Series: 回归评分
        """
        regression_score = pd.Series(0.0, index=self._result.index)
        
        for period in self.periods:
            bias_col = f'BIAS{period}'
            if bias_col in self._result.columns:
                bias_values = self._result[bias_col]
                
                # BIAS从超卖区域回归+18分
                bias_from_oversold = (bias_values.shift(1) < -6) & (bias_values >= -6)
                regression_score += bias_from_oversold * 18
                
                # BIAS从超买区域回归-18分
                bias_from_overbought = (bias_values.shift(1) > 6) & (bias_values <= 6)
                regression_score -= bias_from_overbought * 18
                
                # BIAS向零轴回归（绝对值减小）+8分
                bias_abs_decreasing = (np.abs(bias_values) < np.abs(bias_values.shift(1)))
                regression_score += bias_abs_decreasing * 8
        
        return regression_score / len(self.periods)  # 平均化
    
    def _calculate_bias_extreme_score(self) -> pd.Series:
        """
        计算BIAS极值评分
        
        Returns:
            pd.Series: 极值评分
        """
        extreme_score = pd.Series(0.0, index=self._result.index)
        
        for period in self.periods:
            bias_col = f'BIAS{period}'
            if bias_col in self._result.columns:
                bias_values = self._result[bias_col]
                
                # 计算BIAS的历史分位数
                if len(bias_values) >= 60:  # 至少60个数据点
                    rolling_window = min(60, len(bias_values))
                    
                    # 计算滚动分位数
                    bias_percentile = bias_values.rolling(rolling_window).apply(
                        lambda x: (x.iloc[-1] <= x).sum() / len(x) * 100
                    )
                    
                    # BIAS处于历史低位（10%分位数以下）+25分
                    low_percentile = bias_percentile <= 10
                    extreme_score += low_percentile * 25
                    
                    # BIAS处于历史高位（90%分位数以上）-25分
                    high_percentile = bias_percentile >= 90
                    extreme_score -= high_percentile * 25
                    
                    # BIAS处于极低位（5%分位数以下）额外+15分
                    extreme_low = bias_percentile <= 5
                    extreme_score += extreme_low * 15
                    
                    # BIAS处于极高位（95%分位数以上）额外-15分
                    extreme_high = bias_percentile >= 95
                    extreme_score -= extreme_high * 15
        
        return extreme_score / len(self.periods)  # 平均化
    
    def _calculate_bias_trend_score(self) -> pd.Series:
        """
        计算BIAS趋势评分
        
        Returns:
            pd.Series: 趋势评分
        """
        trend_score = pd.Series(0.0, index=self._result.index)
        
        for period in self.periods:
            bias_col = f'BIAS{period}'
            if bias_col in self._result.columns:
                bias_values = self._result[bias_col]
                
                # BIAS上升趋势+10分
                bias_rising = bias_values > bias_values.shift(1)
                trend_score += bias_rising * 10
                
                # BIAS下降趋势-10分
                bias_falling = bias_values < bias_values.shift(1)
                trend_score -= bias_falling * 10
                
                # BIAS连续上升（3个周期）+15分
                if len(bias_values) >= 3:
                    consecutive_rising = (
                        (bias_values > bias_values.shift(1)) &
                        (bias_values.shift(1) > bias_values.shift(2)) &
                        (bias_values.shift(2) > bias_values.shift(3))
                    )
                    trend_score += consecutive_rising * 15
                
                # BIAS连续下降（3个周期）-15分
                if len(bias_values) >= 3:
                    consecutive_falling = (
                        (bias_values < bias_values.shift(1)) &
                        (bias_values.shift(1) < bias_values.shift(2)) &
                        (bias_values.shift(2) < bias_values.shift(3))
                    )
                    trend_score -= consecutive_falling * 15
        
        return trend_score / len(self.periods)  # 平均化
    
    def _calculate_bias_zero_cross_score(self) -> pd.Series:
        """
        计算BIAS零轴穿越评分
        
        Returns:
            pd.Series: 零轴穿越评分
        """
        zero_cross_score = pd.Series(0.0, index=self._result.index)
        
        for period in self.periods:
            bias_col = f'BIAS{period}'
            if bias_col in self._result.columns:
                bias_values = self._result[bias_col]
                
                # BIAS上穿零轴+22分
                bias_cross_up = crossover(bias_values, 0)
                zero_cross_score += bias_cross_up * 22
                
                # BIAS下穿零轴-22分
                bias_cross_down = crossunder(bias_values, 0)
                zero_cross_score -= bias_cross_down * 22
        
        return zero_cross_score / len(self.periods)  # 平均化
    
    def _calculate_multi_period_consistency_score(self) -> pd.Series:
        """
        计算多周期一致性评分（优化点）
        
        当多个周期的BIAS指标同时给出一致信号时，信号更加可靠
        
        Returns:
            pd.Series: 多周期一致性评分
        """
        if self._result is None or len(self.periods) <= 1:
            return pd.Series(0.0)
        
        score = pd.Series(0.0, index=self._result.index)
        
        # 准备各周期的BIAS数据
        bias_cols = [f'BIAS{p}' for p in self.periods if f'BIAS{p}' in self._result.columns]
        if len(bias_cols) <= 1:
            return score
        
        # 分析各周期BIAS的趋势方向
        bias_signs = pd.DataFrame(index=self._result.index)
        for col in bias_cols:
            bias_signs[f'{col}_sign'] = np.sign(self._result[col])
        
        # 计算一致性
        row_sum = bias_signs.sum(axis=1)
        row_count = bias_signs.shape[1]
        
        # 完全一致看涨（所有BIAS为正）
        full_bullish = row_sum == row_count
        score += full_bullish * 20  # 全部看涨 +20分
        
        # 完全一致看跌（所有BIAS为负）
        full_bearish = row_sum == -row_count
        score -= full_bearish * 20  # 全部看跌 -20分
        
        # 多数看涨（>60%的BIAS为正）
        mostly_bullish = (row_sum > 0) & (abs(row_sum) >= row_count * 0.6) & ~full_bullish
        score += mostly_bullish * 10  # 多数看涨 +10分
        
        # 多数看跌（>60%的BIAS为负）
        mostly_bearish = (row_sum < 0) & (abs(row_sum) >= row_count * 0.6) & ~full_bearish
        score -= mostly_bearish * 10  # 多数看跌 -10分
        
        # 分析一致性强度（BIAS值的协方差）
        if len(bias_cols) >= 2:
            # 计算最近5个周期的BIAS协方差
            recent_bias = self._result[bias_cols].tail(5)
            
            # 计算标准差占平均值的百分比作为一致性强度指标
            if len(recent_bias) > 0:
                # 计算各周期BIAS的标准差
                bias_std = recent_bias.std(axis=1)
                # 计算各周期BIAS的绝对值平均值
                bias_abs_mean = recent_bias.abs().mean(axis=1)
                
                # 避免除零错误
                bias_abs_mean = bias_abs_mean.replace(0, np.nan)
                
                # 计算变异系数（标准差/平均值）作为一致性指标
                cov = bias_std / bias_abs_mean
                cov = cov.fillna(1.0)  # 处理可能的NaN值
                
                # 变异系数越低，一致性越高
                consistency_strength = 1 - np.clip(cov, 0, 1)
                
                # 在最近周期应用一致性强度调整
                last_idx = score.index[-1]
                if last_idx in consistency_strength.index:
                    # 一致性强度作为现有多周期分数的调节因子
                    recent_score = score.iloc[-5:]
                    # 只对非零分数应用调节因子
                    non_zero_mask = recent_score != 0
                    
                    # 一致性越高，信号越强
                    strength_factor = 1 + consistency_strength * 0.5  # 1.0-1.5的调节因子
                    score.iloc[-5:] = np.where(
                        non_zero_mask,
                        recent_score * strength_factor,
                        recent_score
                    )
        
        return score
    
    def _detect_bias_overbought_oversold_patterns(self) -> List[str]:
        """
        检测BIAS超买超卖形态
        
        Returns:
            List[str]: 超买超卖形态列表
        """
        patterns = []
        
        for period in self.periods:
            bias_col = f'BIAS{period}'
            if bias_col in self._result.columns and len(self._result) > 0:
                current_bias = self._result[bias_col].iloc[-1]
                
                if pd.isna(current_bias):
                    continue
                
                if current_bias < -10:
                    patterns.append(f"BIAS{period}极度超卖")
                elif current_bias < -6:
                    patterns.append(f"BIAS{period}超卖")
                elif current_bias > 10:
                    patterns.append(f"BIAS{period}极度超买")
                elif current_bias > 6:
                    patterns.append(f"BIAS{period}超买")
                elif -3 <= current_bias <= 3:
                    patterns.append(f"BIAS{period}中性区域")
        
        return patterns
    
    def _detect_bias_regression_patterns(self) -> List[str]:
        """
        检测BIAS回归形态
        
        Returns:
            List[str]: 回归形态列表
        """
        patterns = []
        
        for period in self.periods:
            bias_col = f'BIAS{period}'
            if bias_col in self._result.columns and len(self._result) >= 2:
                bias_values = self._result[bias_col]
                current_bias = bias_values.iloc[-1]
                prev_bias = bias_values.iloc[-2]
                
                if pd.isna(current_bias) or pd.isna(prev_bias):
                    continue
                
                # 检测从超卖区域回归
                if prev_bias < -6 and current_bias >= -6:
                    patterns.append(f"BIAS{period}从超卖区域回归")
                
                # 检测从超买区域回归
                if prev_bias > 6 and current_bias <= 6:
                    patterns.append(f"BIAS{period}从超买区域回归")
                
                # 检测向零轴回归
                if abs(current_bias) < abs(prev_bias):
                    patterns.append(f"BIAS{period}向零轴回归")
        
        return patterns
    
    def _detect_bias_extreme_patterns(self) -> List[str]:
        """
        检测BIAS极值形态
        
        Returns:
            List[str]: 极值形态列表
        """
        patterns = []
        
        for period in self.periods:
            bias_col = f'BIAS{period}'
            if bias_col in self._result.columns and len(self._result) >= 60:
                bias_values = self._result[bias_col]
                current_bias = bias_values.iloc[-1]
                
                if pd.isna(current_bias):
                    continue
                
                # 计算历史分位数
                recent_60 = bias_values.tail(60)
                percentile = (current_bias <= recent_60).sum() / len(recent_60) * 100
                
                if percentile <= 5:
                    patterns.append(f"BIAS{period}历史极低位")
                elif percentile <= 10:
                    patterns.append(f"BIAS{period}历史低位")
                elif percentile >= 95:
                    patterns.append(f"BIAS{period}历史极高位")
                elif percentile >= 90:
                    patterns.append(f"BIAS{period}历史高位")
        
        return patterns
    
    def _detect_bias_trend_patterns(self) -> List[str]:
        """
        检测BIAS趋势形态
        
        Returns:
            List[str]: 趋势形态列表
        """
        patterns = []
        
        for period in self.periods:
            bias_col = f'BIAS{period}'
            if bias_col in self._result.columns and len(self._result) >= 3:
                bias_values = self._result[bias_col]
                
                # 检查最近3个周期的趋势
                recent_3 = bias_values.tail(3)
                if len(recent_3) == 3 and not recent_3.isna().any():
                    if (recent_3.iloc[2] > recent_3.iloc[1] > recent_3.iloc[0]):
                        patterns.append(f"BIAS{period}连续上升")
                    elif (recent_3.iloc[2] < recent_3.iloc[1] < recent_3.iloc[0]):
                        patterns.append(f"BIAS{period}连续下降")
                
                # 检查当前趋势
                if len(bias_values) >= 2:
                    current_bias = bias_values.iloc[-1]
                    prev_bias = bias_values.iloc[-2]
                    
                    if not pd.isna(current_bias) and not pd.isna(prev_bias):
                        if current_bias > prev_bias:
                            patterns.append(f"BIAS{period}上升")
                        elif current_bias < prev_bias:
                            patterns.append(f"BIAS{period}下降")
                        else:
                            patterns.append(f"BIAS{period}平稳")
        
        return patterns
    
    def _detect_bias_zero_cross_patterns(self) -> List[str]:
        """
        检测BIAS零轴穿越形态
        
        Returns:
            List[str]: 零轴穿越形态列表
        """
        patterns = []
        
        for period in self.periods:
            bias_col = f'BIAS{period}'
            if bias_col in self._result.columns:
                bias_values = self._result[bias_col]
                
                # 检查最近的零轴穿越
                recent_periods = min(5, len(bias_values))
                recent_bias = bias_values.tail(recent_periods)
                
                if crossover(recent_bias, 0).any():
                    patterns.append(f"BIAS{period}上穿零轴")
                
                if crossunder(recent_bias, 0).any():
                    patterns.append(f"BIAS{period}下穿零轴")
                
                # 检查当前位置
                if len(bias_values) > 0:
                    current_bias = bias_values.iloc[-1]
                    if not pd.isna(current_bias):
                        if current_bias > 0:
                            patterns.append(f"BIAS{period}零轴上方")
                        elif current_bias < 0:
                            patterns.append(f"BIAS{period}零轴下方")
                        else:
                            patterns.append(f"BIAS{period}零轴位置")
        
        return patterns

    def generate_signals(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        生成BIAS指标标准化交易信号
        
        Args:
            data: 输入数据，包含OHLCV数据
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            pd.DataFrame: 信号结果DataFrame，包含标准化信号
        """
        # 确保已计算BIAS指标
        if not self.has_result():
            self.calculate(data)
        
        # 初始化信号DataFrame
        signals = pd.DataFrame(index=data.index)
        signals['buy_signal'] = False
        signals['sell_signal'] = False
        signals['neutral_signal'] = True  # 默认为中性信号
        signals['trend'] = 0  # 0表示中性
        signals['score'] = 50.0  # 默认评分50分
        signals['signal_type'] = None
        signals['signal_desc'] = None
        signals['confidence'] = 50.0
        signals['risk_level'] = '中'
        signals['position_size'] = 0.0
        signals['stop_loss'] = None
        signals['market_env'] = 'sideways_market'
        signals['volume_confirmation'] = False
        
        # 获取参数
        overbought = kwargs.get('overbought', 6)  # 超买阈值
        oversold = kwargs.get('oversold', -6)  # 超卖阈值
        
        # 使用主要周期的BIAS生成信号
        main_period = self.periods[0]
        bias_col = f'BIAS{main_period}'
        
        if bias_col in self._result.columns:
            bias = self._result[bias_col]
            
            # 计算评分
            score = self.calculate_raw_score(data, **kwargs)
            signals['score'] = score
            
            # 检测形态
            patterns = self.identify_patterns(data, **kwargs)
            
            # 设置买入信号
            buy_signal_idx = (
                # 超卖区域上穿信号线
                crossover(bias, oversold) | 
                # 或BIAS开始由负转正
                crossover(bias, 0)
            )
            signals.loc[buy_signal_idx, 'buy_signal'] = True
            signals.loc[buy_signal_idx, 'neutral_signal'] = False
            signals.loc[buy_signal_idx, 'trend'] = 1
            signals.loc[buy_signal_idx, 'signal_type'] = 'BIAS超卖回升'
            signals.loc[buy_signal_idx, 'signal_desc'] = 'BIAS指标从超卖区域回升或穿越0轴'
            signals.loc[buy_signal_idx, 'confidence'] = 70.0
            signals.loc[buy_signal_idx, 'position_size'] = 0.3
            
            # 设置卖出信号
            sell_signal_idx = (
                # 超买区域下穿信号线
                crossunder(bias, overbought) | 
                # 或BIAS开始由正转负
                crossunder(bias, 0)
            )
            signals.loc[sell_signal_idx, 'sell_signal'] = True
            signals.loc[sell_signal_idx, 'neutral_signal'] = False
            signals.loc[sell_signal_idx, 'trend'] = -1
            signals.loc[sell_signal_idx, 'signal_type'] = 'BIAS超买回落'
            signals.loc[sell_signal_idx, 'signal_desc'] = 'BIAS指标从超买区域回落或穿越0轴'
            signals.loc[sell_signal_idx, 'confidence'] = 70.0
            signals.loc[sell_signal_idx, 'position_size'] = 0.3
            
            # 设置强烈买入信号
            strong_buy_idx = bias < oversold * 2  # BIAS处于极度超卖区域
            signals.loc[strong_buy_idx, 'buy_signal'] = True
            signals.loc[strong_buy_idx, 'neutral_signal'] = False
            signals.loc[strong_buy_idx, 'trend'] = 1
            signals.loc[strong_buy_idx, 'signal_type'] = 'BIAS极度超卖'
            signals.loc[strong_buy_idx, 'signal_desc'] = 'BIAS指标处于极度超卖区域'
            signals.loc[strong_buy_idx, 'confidence'] = 80.0
            signals.loc[strong_buy_idx, 'position_size'] = 0.5
            
            # 设置强烈卖出信号
            strong_sell_idx = bias > overbought * 2  # BIAS处于极度超买区域
            signals.loc[strong_sell_idx, 'sell_signal'] = True
            signals.loc[strong_sell_idx, 'neutral_signal'] = False
            signals.loc[strong_sell_idx, 'trend'] = -1
            signals.loc[strong_sell_idx, 'signal_type'] = 'BIAS极度超买'
            signals.loc[strong_sell_idx, 'signal_desc'] = 'BIAS指标处于极度超买区域'
            signals.loc[strong_sell_idx, 'confidence'] = 80.0
            signals.loc[strong_sell_idx, 'position_size'] = 0.5
            
            # 根据BIAS形态设置信号
            for pattern in patterns:
                if '底背离' in pattern:
                    pattern_idx = signals.index[-5:]  # 假设形态影响最近5个周期
                    signals.loc[pattern_idx, 'buy_signal'] = True
                    signals.loc[pattern_idx, 'neutral_signal'] = False
                    signals.loc[pattern_idx, 'trend'] = 1
                    signals.loc[pattern_idx, 'signal_type'] = 'BIAS底背离'
                    signals.loc[pattern_idx, 'signal_desc'] = pattern
                    signals.loc[pattern_idx, 'confidence'] = 85.0
                    signals.loc[pattern_idx, 'position_size'] = 0.6
                elif '顶背离' in pattern:
                    pattern_idx = signals.index[-5:]  # 假设形态影响最近5个周期
                    signals.loc[pattern_idx, 'sell_signal'] = True
                    signals.loc[pattern_idx, 'neutral_signal'] = False
                    signals.loc[pattern_idx, 'trend'] = -1
                    signals.loc[pattern_idx, 'signal_type'] = 'BIAS顶背离'
                    signals.loc[pattern_idx, 'signal_desc'] = pattern
                    signals.loc[pattern_idx, 'confidence'] = 85.0
                    signals.loc[pattern_idx, 'position_size'] = 0.6
            
            # 设置止损价格
            if 'low' in data.columns and 'high' in data.columns:
                # 买入信号的止损设为最近的低点
                buy_indices = signals[signals['buy_signal']].index
                if not buy_indices.empty:
                    for idx in buy_indices:
                        if idx > data.index[10]:  # 确保有足够的历史数据
                            lookback = 5
                            signals.loc[idx, 'stop_loss'] = data.loc[idx-lookback:idx, 'low'].min() * 0.98
                
                # 卖出信号的止损设为最近的高点
                sell_indices = signals[signals['sell_signal']].index
                if not sell_indices.empty:
                    for idx in sell_indices:
                        if idx > data.index[10]:  # 确保有足够的历史数据
                            lookback = 5
                            signals.loc[idx, 'stop_loss'] = data.loc[idx-lookback:idx, 'high'].max() * 1.02
        
        return signals

