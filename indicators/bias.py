#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
均线多空指标(BIAS)

(收盘价-MA)/MA×100%
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple, Any
from scipy import stats

from indicators.base_indicator import BaseIndicator, PatternResult
from indicators.common import crossover, crossunder
from utils.logger import get_logger
from indicators.pattern_registry import PatternRegistry, PatternType, PatternStrength

logger = get_logger(__name__)


class BIAS(BaseIndicator):
    """
    均线多空指标(BIAS) (BIAS)
    
    分类：趋势类指标
    描述：(收盘价-MA)/MA×100%
    """
    
    def __init__(self, name: str = "BIAS", description: str = "均线多空指标",
                 period: int = 14, periods: List[int] = None):
        """
        初始化均线多空指标(BIAS)指标
        
        Args:
            name: 指标名称
            description: 指标描述
            period: 计算周期，默认为14
            periods: 多个计算周期，如果提供，将计算多个周期的BIAS
        """
        super().__init__(name, description)
        self.periods = periods if periods is not None else [period]
        self.indicator_type = "BIAS"
        
        # 定义必需的列
        self.REQUIRED_COLUMNS = ['close', 'high', 'low']
        
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
        
        self.data = df.copy()
        df_copy = self.data
        
        # 计算不同周期的BIAS
        for p in self.periods:
            # 计算移动平均线
            ma = df_copy['close'].rolling(window=p).mean()
            
            # 计算BIAS
            df_copy[f'BIAS{p}'] = (df_copy['close'] - ma) / ma * 100
        
        # 为了兼容性，将主周期的BIAS值存储在名为'bias'的列中
        if self.periods:
            main_period = self.periods[0]
            df_copy['bias'] = df_copy[f'BIAS{main_period}']
        
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
        
        # 使用主要周期的BIAS生成基础信号
        main_period = self.periods[0]
        bias_col = f'BIAS{main_period}'
        
        # 检查必要的指标列是否存在
        if bias_col in df_copy.columns:
            # 超卖区域上穿信号线为买入信号
            df_copy.loc[self._crossover(df_copy[bias_col], oversold), f'bias_signal'] = 1
            
            # 超买区域下穿信号线为卖出信号
            df_copy.loc[self._crossunder(df_copy[bias_col], overbought), f'bias_signal'] = -1
            
            # 使用多周期协同评估增强信号
            if len(self.periods) > 1 and 'BIAS_MP_resonance_signal' in df_copy.columns:
                # 提高共振信号的权重
                resonance_mask = (df_copy['BIAS_MP_resonance_signal'] == 1) & (df_copy[f'bias_signal'] != 0)
                
                # 设置共振信号标记
                df_copy['bias_resonance'] = False
                df_copy.loc[resonance_mask, 'bias_resonance'] = True
                
                # 增加共振信号的强度指标
                if 'BIAS_MP_resonance_strength' in df_copy.columns:
                    df_copy['bias_signal_strength'] = df_copy['bias_signal'].abs()
                    resonance_strength = df_copy['BIAS_MP_resonance_strength'] / 100  # 归一化到0-1
                    
                    # 根据共振强度增强信号强度
                    df_copy.loc[resonance_mask, 'bias_signal_strength'] = df_copy.loc[resonance_mask, 'bias_signal_strength'] * (1 + resonance_strength)
                else:
                    df_copy['bias_signal_strength'] = df_copy['bias_signal'].abs()
            
            # 使用回归速率分析优化信号
            reg_col = f'BIAS_REG_regression_efficiency_{main_period}'
            if reg_col in df_copy.columns:
                # 回归效率高的信号更可靠
                signal_mask = df_copy[f'bias_signal'] != 0
                
                if 'bias_signal_confidence' not in df_copy.columns:
                    df_copy['bias_signal_confidence'] = 0.5  # 默认中等置信度
                
                # 根据回归效率调整信号置信度
                df_copy.loc[signal_mask, 'bias_signal_confidence'] = 0.5 + df_copy.loc[signal_mask, reg_col] * 0.5
        
        return df_copy
    
    def _crossover(self, series1, series2):
        """判断series1是否上穿series2"""
        if isinstance(series2, (int, float)):
            series2 = pd.Series(series2, index=series1.index)
        
        crossover = (series1.shift(1) <= series2.shift(1)) & (series1 > series2)
        return crossover
    
    def _crossunder(self, series1, series2):
        """判断series1是否下穿series2"""
        if isinstance(series2, (int, float)):
            series2 = pd.Series(series2, index=series1.index)
        
        crossunder = (series1.shift(1) >= series2.shift(1)) & (series1 < series2)
        return crossunder
    
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
        multi_period_score = pd.Series(0.0, index=data.index)
        
        if len(self.periods) > 1:
            # 使用已经计算好的多周期协同评估结果
            if 'BIAS_MP_resonance_strength' in data.columns:
                # 根据共振强度提升评分
                resonance_strength = data['BIAS_MP_resonance_strength']
                multi_period_score = resonance_strength * 0.2  # 最多贡献20分
                
                # 当存在强共振信号时额外加分
                if 'BIAS_MP_resonance_signal' in data.columns:
                    signal_mask = data['BIAS_MP_resonance_signal'] == 1
                    multi_period_score.loc[signal_mask] += 10  # 强共振信号额外加10分
            else:
                # 如果没有预计算的共振结果，则使用内部函数计算
                multi_period_score = self._calculate_multi_period_consistency_score()
                
            score += multi_period_score
        
        # 7. 回归率分析评分
        regression_rate_score = pd.Series(0.0, index=data.index)
        
        # 使用已经计算好的回归速率分析结果
        main_period = self.periods[0]
        if f'BIAS_REG_regression_efficiency_{main_period}' in data.columns:
            # 回归效率高的BIAS可能更可靠
            regression_efficiency = data[f'BIAS_REG_regression_efficiency_{main_period}']
            # 转换为0-15分
            regression_rate_score = regression_efficiency * 0.15
            
            score += regression_rate_score
        
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
        
        # 1. 检测超买超卖形态
        overbought_oversold_patterns = self._detect_bias_overbought_oversold_patterns()
        patterns.extend(overbought_oversold_patterns)
        
        # 2. 检测回归形态（增强版）
        regression_patterns = self._detect_bias_regression_patterns()
        patterns.extend(regression_patterns)
        
        # 3. 检测极值形态
        extreme_patterns = self._detect_bias_extreme_patterns(data)
        patterns.extend(extreme_patterns)
        
        # 4. 检测趋势形态
        trend_patterns = self._detect_bias_trend_patterns()
        patterns.extend(trend_patterns)
        
        # 5. 检测零轴穿越形态
        zero_cross_patterns = self._detect_bias_zero_cross_patterns()
        patterns.extend(zero_cross_patterns)
        
        # 6. 检测多周期协同形态（新增）
        if len(self.periods) > 1:
            multi_period_patterns = self._detect_bias_multi_period_patterns()
            patterns.extend(multi_period_patterns)
        
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
        计算BIAS回归评分（增强版）
        优化实现：评估BIAS回归至均衡值的速度和强度，提高回归速率分析的精确度
        
        Returns:
            pd.Series: BIAS回归评分（-15到15分）
        """
        # 初始化评分
        regression_score = pd.Series(0.0, index=self._result.index)
        
        # 获取主要周期的BIAS
        main_period = self.periods[0]
        bias_col = f'BIAS{main_period}'
        
        if bias_col not in self._result.columns:
            return regression_score
        
        # 计算BIAS的绝对值（与零轴的距离）
        bias_abs = self._result[bias_col].abs()
        
        # 计算BIAS变动率（回归速度）
        bias_change = self._result[bias_col].diff()
        
        # 计算BIAS加速度（回归加速度）
        bias_acceleration = bias_change.diff()
        
        # 计算回归速率指标（偏离度与变化率的比值）
        regression_rate = pd.Series(index=self._result.index)
        for i in range(1, len(self._result)):
            current_bias = self._result[bias_col].iloc[i]
            current_change = bias_change.iloc[i]
            
            # 仅在回归方向时计算速率（向零轴方向移动）
            if (current_bias > 0 and current_change < 0) or (current_bias < 0 and current_change > 0):
                regression_rate.iloc[i] = abs(current_change) / (abs(current_bias) + 0.1)  # 避免除以零
            else:
                regression_rate.iloc[i] = 0
        
        # 对每个数据点评估
        for i in range(5, len(self._result)):  # 从第5个点开始，确保有足够历史数据
            # 当前BIAS值
            current_bias = self._result[bias_col].iloc[i]
            
            # BIAS向零轴回归的速度
            current_change = bias_change.iloc[i]
            
            # BIAS回归的加速度
            current_accel = bias_acceleration.iloc[i]
            
            # 回归速率
            current_rate = regression_rate.iloc[i]
            
            # 计算动态阈值 - 基于历史波动性
            lookback = min(20, i)
            bias_std = self._result[bias_col].iloc[i-lookback:i].std() if lookback > 0 else 6
            
            # 动态偏离阈值
            extreme_threshold = max(15, bias_std * 2.5)
            high_threshold = max(8, bias_std * 1.5)
            moderate_threshold = max(3, bias_std * 0.8)
            
            # 评估BIAS偏离程度与回归状态
            if abs(current_bias) > extreme_threshold:  # 极度偏离
                # 极度偏离且加速回归是强烈信号
                if (current_bias > 0 and current_change < 0) or (current_bias < 0 and current_change > 0):
                    # BIAS偏离且开始回归，这是买入/卖出信号
                    regression_score.iloc[i] += 10 * (-np.sign(current_bias))
                    
                    # 考虑加速度因素
                    if (current_bias > 0 and current_accel < 0) or (current_bias < 0 and current_accel > 0):
                        # 加速回归，信号更强
                        regression_score.iloc[i] += 5 * (-np.sign(current_bias))
                else:
                    # 极度偏离且继续偏离，这是危险信号
                    regression_score.iloc[i] -= 7 * np.sign(current_bias)
            
            elif abs(current_bias) > high_threshold:  # 高度偏离
                # 高度偏离且回归
                if (current_bias > 0 and current_change < 0) or (current_bias < 0 and current_change > 0):
                    regression_score.iloc[i] += 7 * (-np.sign(current_bias))
                    
                    # 回归加速度评分
                    if (current_bias > 0 and current_accel < 0) or (current_bias < 0 and current_accel > 0):
                        regression_score.iloc[i] += 3 * (-np.sign(current_bias))
                else:
                    # 高度偏离且继续偏离
                    regression_score.iloc[i] -= 5 * np.sign(current_bias)
            
            elif abs(current_bias) > moderate_threshold:  # 中度偏离
                # 中度偏离且回归
                if (current_bias > 0 and current_change < 0) or (current_bias < 0 and current_change > 0):
                    regression_score.iloc[i] += 5 * (-np.sign(current_bias))
                else:
                    # 中度偏离且继续偏离
                    regression_score.iloc[i] -= 3 * np.sign(current_bias)
            
            else:  # 接近零轴
                # 在零轴附近的震荡，评分轻微
                regression_score.iloc[i] += -3 * np.sign(current_change)
            
            # 考虑回归速度 - 优化点：更精确的回归速率评估
            if current_rate > 0:  # 确保是回归方向
                # 计算历史回归速率的分位数
                if i >= 20:
                    regression_rates = regression_rate.iloc[i-20:i]
                    non_zero_rates = regression_rates[regression_rates > 0]
                    
                    if len(non_zero_rates) > 0:
                        rate_percentile = np.percentile(non_zero_rates, [25, 50, 75, 90])
                        
                        # 根据回归速率分位数进行评分
                        if current_rate > rate_percentile[3]:  # 极快回归 (>90%)
                            regression_score.iloc[i] += 5 * (-np.sign(current_bias))
                        elif current_rate > rate_percentile[2]:  # 快速回归 (>75%)
                            regression_score.iloc[i] += 3 * (-np.sign(current_bias))
                        elif current_rate > rate_percentile[1]:  # 中速回归 (>50%)
                            regression_score.iloc[i] += 1.5 * (-np.sign(current_bias))
                    else:
                        # 回归速率评分（简化版，无历史数据时）
                        if current_rate > 0.3:  # 快速回归
                            regression_score.iloc[i] += 3 * (-np.sign(current_bias))
                        elif current_rate > 0.1:  # 中速回归
                            regression_score.iloc[i] += 1.5 * (-np.sign(current_bias))
                else:
                    # 回归速率评分（简化版，数据点不足时）
                    if current_rate > 0.3:  # 快速回归
                        regression_score.iloc[i] += 3 * (-np.sign(current_bias))
                    elif current_rate > 0.1:  # 中速回归
                        regression_score.iloc[i] += 1.5 * (-np.sign(current_bias))
            
            # 回归持续性评分 - 新增功能点：评估回归的持续性
            if i >= 5:
                # 检查最近5个点的回归一致性
                consistent_regression = True
                regression_direction = np.sign(current_change) * -np.sign(current_bias)
                
                for j in range(1, 5):
                    prev_bias = self._result[bias_col].iloc[i-j]
                    prev_change = bias_change.iloc[i-j]
                    prev_direction = np.sign(prev_change) * -np.sign(prev_bias)
                    
                    if prev_direction != regression_direction or prev_direction == 0:
                        consistent_regression = False
                        break
                
                # 持续回归加分
                if consistent_regression and regression_direction > 0:
                    regression_score.iloc[i] += 4
        
        # 限制评分范围
        return regression_score.clip(-15, 15)
    
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
        计算多周期BIAS一致性评分
        优化实现：增强多周期一致性评估，根据不同周期BIAS的协同性给予评分
        
        Returns:
            pd.Series: 多周期一致性评分（-15到15分）
        """
        # 初始化评分
        consistency_score = pd.Series(0.0, index=self._result.index)
        
        if len(self.periods) <= 1:
            return consistency_score
        
        # 按照从短到长排序周期
        sorted_periods = sorted(self.periods)
        
        # 计算各周期BIAS的方向一致性
        for i in range(len(self._result)):
            if i < 5:  # 前几个数据点可能不够计算
                continue
            
            # 统计上涨和下跌的BIAS数量
            up_count = 0
            down_count = 0
            
            # 统计超买和超卖的BIAS数量
            overbought_count = 0
            oversold_count = 0
            
            # 计算各周期的方向和状态
            for p in sorted_periods:
                bias_col = f'BIAS{p}'
                if bias_col not in self._result.columns:
                    continue
                
                # 计算BIAS方向
                current_bias = self._result[bias_col].iloc[i]
                prev_bias = self._result[bias_col].iloc[i-1]
                
                if current_bias > prev_bias:
                    up_count += 1
                elif current_bias < prev_bias:
                    down_count += 1
                
                # 计算超买超卖状态（使用动态阈值）
                lookback = min(20, i)
                volatility = self._result[bias_col].iloc[i-lookback:i].std() if lookback > 0 else 6
                
                overbought_threshold = max(6, volatility * 1.5)
                oversold_threshold = min(-6, -volatility * 1.5)
                
                if current_bias > overbought_threshold:
                    overbought_count += 1
                elif current_bias < oversold_threshold:
                    oversold_count += 1
            
            # 计算方向一致性得分
            total_periods = len(sorted_periods)
            if up_count == total_periods:  # 所有周期都上涨
                consistency_score.iloc[i] += 8
            elif down_count == total_periods:  # 所有周期都下跌
                consistency_score.iloc[i] -= 8
            elif up_count > down_count:  # 多数周期上涨
                consistency_score.iloc[i] += 4 * (up_count / total_periods)
            elif down_count > up_count:  # 多数周期下跌
                consistency_score.iloc[i] -= 4 * (down_count / total_periods)
            
            # 计算超买超卖一致性得分
            if overbought_count >= total_periods * 0.7:  # 70%以上周期超买
                consistency_score.iloc[i] -= 7  # 超买是卖出信号，得分降低
            elif oversold_count >= total_periods * 0.7:  # 70%以上周期超卖
                consistency_score.iloc[i] += 7  # 超卖是买入信号，得分提高
            elif overbought_count > 0 and oversold_count > 0:  # 周期分歧
                consistency_score.iloc[i] -= 3  # 信号混乱，轻微降低得分
        
        # 计算周期间的协同性（相关性）与收敛/发散性
        if len(sorted_periods) >= 2:
            # 为避免前期数据不足，从一定位置开始计算
            start_idx = max(30, sorted_periods[-1])
            
            if len(self._result) > start_idx:
                # 准备各周期的BIAS数据
                bias_data = {}
                for p in sorted_periods:
                    bias_col = f'BIAS{p}'
                    if bias_col in self._result.columns:
                        bias_data[p] = self._result[bias_col]
                
                # 计算周期间的相关性矩阵和收敛/发散性
                for i in range(start_idx, len(self._result)):
                    # 使用20个点的滚动窗口
                    window_size = min(20, i)
                    
                    # 1. 计算相关性矩阵
                    period_corrs = []
                    for j in range(len(sorted_periods)-1):
                        for k in range(j+1, len(sorted_periods)):
                            p1 = sorted_periods[j]
                            p2 = sorted_periods[k]
                            
                            if p1 in bias_data and p2 in bias_data:
                                s1 = bias_data[p1].iloc[i-window_size:i]
                                s2 = bias_data[p2].iloc[i-window_size:i]
                                
                                try:
                                    corr = s1.corr(s2)
                                    if not np.isnan(corr):
                                        period_corrs.append(corr)
                                except:
                                    pass
                    
                    # 计算平均相关性
                    if period_corrs:
                        avg_corr = np.mean(period_corrs)
                        # 高相关性加分，低相关性减分
                        consistency_score.iloc[i] += avg_corr * 7
                    
                    # 2. 计算BIAS收敛/发散性
                    current_range = []
                    prev_range = []
                    
                    for p in sorted_periods:
                        if p in bias_data:
                            current_range.append(bias_data[p].iloc[i])
                            if i > 5:
                                prev_range.append(bias_data[p].iloc[i-5])
                    
                    if current_range and prev_range:
                        current_spread = max(current_range) - min(current_range)
                        prev_spread = max(prev_range) - min(prev_range)
                        
                        # 计算收敛/发散趋势
                        if current_spread < prev_spread * 0.7:  # 收敛30%以上
                            # BIAS收敛通常表示趋势即将形成
                            consistency_score.iloc[i] += 5
                        elif current_spread > prev_spread * 1.3:  # 发散30%以上
                            # BIAS发散通常表示趋势分歧加大
                            consistency_score.iloc[i] -= 3
                        
                        # 考虑绝对收敛水平
                        if current_spread < 3:  # 极度收敛
                            consistency_score.iloc[i] += 3
                        elif current_spread > 15:  # 极度发散
                            consistency_score.iloc[i] -= 4
                    
                    # 3. 判断多周期共振信号
                    # 计算各周期的突破或回踩信号
                    breakout_count = 0
                    breakdown_count = 0
                    
                    for p in sorted_periods:
                        if p in bias_data:
                            # 零轴突破
                            if bias_data[p].iloc[i] > 0 and bias_data[p].iloc[i-1] <= 0:
                                breakout_count += 1
                            elif bias_data[p].iloc[i] < 0 and bias_data[p].iloc[i-1] >= 0:
                                breakdown_count += 1
                    
                    # 多周期共同突破/跌破是强信号
                    if breakout_count >= len(sorted_periods) * 0.5:  # 半数以上周期突破
                        consistency_score.iloc[i] += 8
                    if breakdown_count >= len(sorted_periods) * 0.5:  # 半数以上周期跌破
                        consistency_score.iloc[i] -= 8
        
        # 限制得分范围
        return consistency_score.clip(-15, 15)
    
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
        检测BIAS回归形态（增强版）
        优化实现：识别BIAS回归的速度、模式和跨周期协同关系
        
        Returns:
            List[str]: 检测到的回归形态
        """
        patterns = []
        
        # 需要至少2个周期进行多周期分析
        if len(self.periods) < 2:
            # 获取主要周期的BIAS
            main_period = self.periods[0]
            bias_col = f'BIAS{main_period}'
            
            if bias_col not in self._result.columns:
                return patterns
            
            # 需要至少5个数据点
            if len(self._result) < 5:
                return patterns
            
            # 获取最后几个点的BIAS值
            last_idx = len(self._result) - 1
            current_bias = self._result[bias_col].iloc[last_idx]
            prev_bias = self._result[bias_col].iloc[last_idx-1]
            
            # 计算BIAS变动率（回归速度）
            bias_change = self._result[bias_col].diff().iloc[last_idx]
            
            # 计算BIAS加速度
            bias_accel = self._result[bias_col].diff().diff().iloc[last_idx]
            
            # 计算最近5个点的变动方向一致性
            bias_5days = self._result[bias_col].iloc[last_idx-4:last_idx+1]
            consecutive_up = True
            consecutive_down = True
            
            for i in range(len(bias_5days) - 1):
                if bias_5days.iloc[i+1] <= bias_5days.iloc[i]:
                    consecutive_up = False
                if bias_5days.iloc[i+1] >= bias_5days.iloc[i]:
                    consecutive_down = False
            
            # 确定当前BIAS的偏离状态
            # 使用动态阈值
            lookback = min(20, last_idx)
            volatility = self._result[bias_col].iloc[last_idx-lookback:last_idx].std() if lookback > 0 else 6
            
            overbought_threshold = max(6, volatility * 1.5)
            oversold_threshold = min(-6, -volatility * 1.5)
            
            is_overbought = current_bias > overbought_threshold
            is_oversold = current_bias < oversold_threshold
            
            # 从超买区开始回归
            if is_overbought and bias_change < 0:
                patterns.append("BIAS超买开始回归")
                
                # 判断回归速度
                if abs(bias_change) > volatility:
                    patterns.append("BIAS超买快速回归")
                
                # 判断加速度
                if bias_accel < 0:
                    patterns.append("BIAS超买加速回归")
            
            # 从超卖区开始回归
            elif is_oversold and bias_change > 0:
                patterns.append("BIAS超卖开始回归")
                
                # 判断回归速度
                if abs(bias_change) > volatility:
                    patterns.append("BIAS超卖快速回归")
                
                # 判断加速度
                if bias_accel > 0:
                    patterns.append("BIAS超卖加速回归")
            
            # 连续上升或下降形态
            if consecutive_up:
                patterns.append("BIAS连续5日上升")
            elif consecutive_down:
                patterns.append("BIAS连续5日下降")
            
            # 零轴附近震荡
            if abs(current_bias) < 3 and abs(prev_bias) < 3:
                patterns.append("BIAS零轴震荡")
            
            # 回归到零轴
            if (prev_bias > 0 and current_bias <= 0) or (prev_bias < 0 and current_bias >= 0):
                patterns.append("BIAS穿越零轴")
            
            return patterns
        
        # 多周期回归分析
        # 按照从短到长排序周期
        sorted_periods = sorted(self.periods)
        
        # 获取最后几个点的数据
        last_idx = len(self._result) - 1
        if last_idx < 5:  # 确保有足够的数据点
            return patterns
        
        # 1. 计算各周期BIAS的回归状态
        bias_data = {}
        changes = {}
        accels = {}
        
        # 获取各周期的BIAS、变化率和加速度
        for period in sorted_periods:
            bias_col = f'BIAS{period}'
            if bias_col in self._result.columns:
                bias_values = self._result[bias_col]
                bias_data[period] = bias_values.iloc[last_idx]
                changes[period] = bias_values.diff().iloc[last_idx]
                accels[period] = bias_values.diff().diff().iloc[last_idx]
        
        # 2. 计算动态阈值（基于历史波动率）
        volatility = {}
        overbought_thresholds = {}
        oversold_thresholds = {}
        
        for period in sorted_periods:
            bias_col = f'BIAS{period}'
            if bias_col in self._result.columns:
                lookback = min(20, last_idx)
                vol = self._result[bias_col].iloc[last_idx-lookback:last_idx].std() if lookback > 0 else 6
                volatility[period] = vol
                overbought_thresholds[period] = max(6, vol * 1.5)
                oversold_thresholds[period] = min(-6, -vol * 1.5)
        
        # 3. 判断各周期的偏离状态
        overbought_periods = []
        oversold_periods = []
        
        for period in sorted_periods:
            if period in bias_data:
                if bias_data[period] > overbought_thresholds[period]:
                    overbought_periods.append(period)
                elif bias_data[period] < oversold_thresholds[period]:
                    oversold_periods.append(period)
        
        # 4. 判断各周期的回归状态
        regression_from_overbought = []
        regression_from_oversold = []
        
        for period in sorted_periods:
            if period in bias_data and period in changes:
                if period in overbought_periods and changes[period] < 0:
                    regression_from_overbought.append(period)
                elif period in oversold_periods and changes[period] > 0:
                    regression_from_oversold.append(period)
        
        # 5. 分析回归速率一致性
        if len(regression_from_overbought) >= 2 or len(regression_from_oversold) >= 2:
            # 计算回归速率（变化量/偏离度）
            regression_rates = {}
            
            for period in sorted_periods:
                if period in bias_data and period in changes:
                    # 计算回归速率
                    if (bias_data[period] > 0 and changes[period] < 0) or (bias_data[period] < 0 and changes[period] > 0):
                        regression_rates[period] = abs(changes[period]) / (abs(bias_data[period]) + 0.1)
            
            # 分析回归速率一致性
            if regression_rates:
                rate_values = list(regression_rates.values())
                avg_rate = np.mean(rate_values)
                max_rate = max(rate_values)
                min_rate = min(rate_values)
                
                # 判断回归速率一致性
                rate_consistency = (max_rate - min_rate) / avg_rate if avg_rate > 0 else 0
                
                if rate_consistency < 0.3:  # 速率高度一致
                    if avg_rate > 0.3:
                        patterns.append("多周期BIAS快速同步回归")
                    else:
                        patterns.append("多周期BIAS稳定同步回归")
                elif rate_consistency > 0.7:  # 速率高度不一致
                    patterns.append("多周期BIAS回归速率分化")
        
        # 6. 分析多周期回归形态
        if len(regression_from_overbought) > 0:
            if len(regression_from_overbought) == len(sorted_periods):
                patterns.append("全周期BIAS超买同步回归")
            elif len(regression_from_overbought) >= len(sorted_periods) * 0.7:
                patterns.append("多数周期BIAS超买回归")
            
            # 检查是否为领先回归（短周期先回归）
            short_regressing = sorted_periods[0] in regression_from_overbought
            long_regressing = sorted_periods[-1] in regression_from_overbought
            
            if short_regressing and not long_regressing:
                patterns.append("短周期BIAS超买先行回归")
            elif long_regressing and not short_regressing:
                patterns.append("长周期BIAS超买先行回归")
        
        if len(regression_from_oversold) > 0:
            if len(regression_from_oversold) == len(sorted_periods):
                patterns.append("全周期BIAS超卖同步回归")
            elif len(regression_from_oversold) >= len(sorted_periods) * 0.7:
                patterns.append("多数周期BIAS超卖回归")
            
            # 检查是否为领先回归（短周期先回归）
            short_regressing = sorted_periods[0] in regression_from_oversold
            long_regressing = sorted_periods[-1] in regression_from_oversold
            
            if short_regressing and not long_regressing:
                patterns.append("短周期BIAS超卖先行回归")
            elif long_regressing and not short_regressing:
                patterns.append("长周期BIAS超卖先行回归")
        
        # 7. 分析回归加速度趋势
        acceleration_up = []
        acceleration_down = []
        
        for period in sorted_periods:
            if period in accels:
                if bias_data[period] > 0 and accels[period] < 0:  # 超买加速回归
                    acceleration_up.append(period)
                elif bias_data[period] < 0 and accels[period] > 0:  # 超卖加速回归
                    acceleration_down.append(period)
        
        if len(acceleration_up) >= len(sorted_periods) * 0.7:
            patterns.append("多周期BIAS超买加速回归")
        if len(acceleration_down) >= len(sorted_periods) * 0.7:
            patterns.append("多周期BIAS超卖加速回归")
        
        # 8. 分析零轴穿越协同性
        cross_zero_up = []
        cross_zero_down = []
        
        for period in sorted_periods:
            bias_col = f'BIAS{period}'
            if bias_col in self._result.columns and last_idx > 0:
                current = self._result[bias_col].iloc[last_idx]
                prev = self._result[bias_col].iloc[last_idx-1]
                
                if current >= 0 and prev < 0:  # 上穿零轴
                    cross_zero_up.append(period)
                elif current <= 0 and prev > 0:  # 下穿零轴
                    cross_zero_down.append(period)
        
        if len(cross_zero_up) >= 2:
            patterns.append(f"{len(cross_zero_up)}周期BIAS同步上穿零轴")
        if len(cross_zero_down) >= 2:
            patterns.append(f"{len(cross_zero_down)}周期BIAS同步下穿零轴")
        
        return patterns
    
    def _detect_bias_extreme_patterns(self, data: pd.DataFrame) -> bool:
        """检测BIAS极值形态"""
        if not self.has_result():
            return False
        
        # 确保数据量足够
        if len(self._result) < 20:
            return False
        
        # 获取BIAS数据
        bias = self._result[f'bias_{self.period}'].iloc[-20:]
        
        # 计算历史极值
        bias_max = bias.max()
        bias_min = bias.min()
        current_bias = bias.iloc[-1]
        
        # 定义极值阈值（可根据具体需求调整）
        extreme_threshold = 0.8  # 达到历史极值的80%视为接近极值
        
        # 检测是否接近历史极值
        is_high_extreme = current_bias > 0 and current_bias > extreme_threshold * bias_max
        is_low_extreme = current_bias < 0 and current_bias < extreme_threshold * bias_min
        
        return is_high_extreme or is_low_extreme
    
    def _detect_bias_divergence_patterns(self, data: pd.DataFrame) -> bool:
        """检测BIAS背离形态"""
        if not self.has_result() or 'close' not in data.columns:
            return False
        
        # 确保数据量足够
        if len(self._result) < 10 or len(data) < 10:
            return False
        
        # 获取BIAS和价格数据
        bias = self._result[f'bias_{self.period}'].iloc[-10:]
        price = data['close'].iloc[-10:]
        
        # 寻找最近10个周期内的高点和低点
        bias_max_idx = bias.iloc[:-1].idxmax()
        bias_min_idx = bias.iloc[:-1].idxmin()
        price_max_idx = price.iloc[:-1].idxmax()
        price_min_idx = price.iloc[:-1].idxmin()
        
        # 检查是否有顶背离（价格创新高但BIAS未创新高）
        bearish_divergence = False
        if price_max_idx > bias_max_idx:
            if price.iloc[-1] >= price.loc[price_max_idx] and bias.iloc[-1] < bias.loc[bias_max_idx]:
                bearish_divergence = True
        
        # 检查是否有底背离（价格创新低但BIAS未创新低）
        bullish_divergence = False
        if price_min_idx > bias_min_idx:
            if price.iloc[-1] <= price.loc[price_min_idx] and bias.iloc[-1] > bias.loc[bias_min_idx]:
                bullish_divergence = True
        
        return bearish_divergence or bullish_divergence
    
    def _detect_bias_zero_cross_patterns(self, data: pd.DataFrame) -> bool:
        """检测BIAS零轴穿越形态"""
        if not self.has_result():
            return False
        
        # 确保数据量足够
        if len(self._result) < 3:
            return False
        
        # 获取BIAS数据
        bias = self._result[f'bias_{self.period}'].iloc[-3:]
        
        # 检测BIAS上穿零轴
        zero_cross_up = (bias.iloc[-3] < 0) and (bias.iloc[-1] > 0)
        
        # 检测BIAS下穿零轴
        zero_cross_down = (bias.iloc[-3] > 0) and (bias.iloc[-1] < 0)
        
        return zero_cross_up or zero_cross_down
    
    def _detect_bias_momentum_patterns(self, data: pd.DataFrame) -> bool:
        """检测BIAS动量形态"""
        if not self.has_result():
            return False
        
        # 确保数据量足够
        if len(self._result) < 5:
            return False
        
        # 获取BIAS数据
        bias = self._result[f'bias_{self.period}'].iloc[-5:]
        
        # 计算BIAS的变化率
        bias_change = bias.diff().dropna()
        
        # 连续上升或下降动量
        rising_momentum = (bias_change > 0).all()
        falling_momentum = (bias_change < 0).all()
        
        return rising_momentum or falling_momentum

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

    def evaluate_multi_period_bias(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        评估BIAS指标的多周期协同性
        
        Args:
            data: 输入数据 (当前未使用，依赖于self._result)
            
        Returns:
            pd.DataFrame: 多周期协同评估结果
        """
        if self._result is None:
            logger.warning("evaluate_multi_period_bias called before calculate, _result is None.")
            return pd.DataFrame()

        if len(self.periods) <= 1:
            return pd.DataFrame(index=self._result.index)
        
        # 按照从短到长排序周期
        sorted_periods = sorted(self.periods)
        
        # 初始化结果数据框
        result = pd.DataFrame(index=self._result.index)
        
        # 1. 计算各周期之间的相关性矩阵
        if len(self._result) >= 20:
            correlation_matrix = {}
            for i, p1 in enumerate(sorted_periods):
                bias_col1 = f'BIAS{p1}'
                if bias_col1 not in self._result.columns:
                    continue
                
                for j, p2 in enumerate(sorted_periods[i+1:], i+1):
                    bias_col2 = f'BIAS{p2}'
                    if bias_col2 not in self._result.columns:
                        continue
                    
                    # 计算20日滚动相关系数
                    rolling_corr = self._result[bias_col1].rolling(20).corr(self._result[bias_col2])
                    correlation_matrix[f'corr_{p1}_{p2}'] = rolling_corr
                    result[f'corr_{p1}_{p2}'] = rolling_corr
            
            # 计算平均相关系数
            if correlation_matrix:
                corr_cols = list(correlation_matrix.keys())
                result['avg_correlation'] = sum(result[col] for col in corr_cols) / len(corr_cols)
        
        # 2. 计算BIAS多周期协同指数
        resonance_index = pd.Series(0.0, index=self._result.index)
        
        for i in range(len(self._result)):
            if i < 5:  # 前几个数据点可能不够计算
                continue
            
            # 2.1 方向一致性
            direction_agreement = 0
            valid_periods = 0
            
            for p in sorted_periods:
                bias_col = f'BIAS{p}'
                if bias_col not in self._result.columns:
                    continue
                
                valid_periods += 1
                if i > 0:
                    # 方向：1=上升，-1=下降，0=不变
                    direction = np.sign(self._result[bias_col].iloc[i] - self._result[bias_col].iloc[i-1])
                    direction_agreement += direction
            
            if valid_periods > 0:
                # 归一化方向一致性 (-1到1)
                direction_score = abs(direction_agreement) / valid_periods
                resonance_index.iloc[i] += direction_score * 0.4  # 40%权重
            
            # 2.2 位置一致性
            position_agreement = 0
            for p in sorted_periods:
                bias_col = f'BIAS{p}'
                if bias_col not in self._result.columns:
                    continue
                
                # 位置：1=正值，-1=负值，0=零
                position = np.sign(self._result[bias_col].iloc[i])
                position_agreement += position
            
            if valid_periods > 0:
                # 归一化位置一致性 (-1到1)
                position_score = abs(position_agreement) / valid_periods
                resonance_index.iloc[i] += position_score * 0.3  # 30%权重
            
            # 2.3 极值一致性
            extreme_agreement = 0
            for p in sorted_periods:
                bias_col = f'BIAS{p}'
                if bias_col not in self._result.columns:
                    continue
                
                # 计算动态阈值
                lookback = min(20, i)
                std = self._result[bias_col].iloc[i-lookback:i].std() if lookback > 0 else 6
                
                overbought = max(6, std * 1.5)
                oversold = min(-6, -std * 1.5)
                
                current_bias = self._result[bias_col].iloc[i]
                
                # 极值：1=超买，-1=超卖，0=正常
                if current_bias > overbought:
                    extreme_agreement += 1
                elif current_bias < oversold:
                    extreme_agreement -= 1
            
            if valid_periods > 0:
                # 归一化极值一致性 (-1到1)
                extreme_score = abs(extreme_agreement) / valid_periods
                resonance_index.iloc[i] += extreme_score * 0.3  # 30%权重
        
        # 添加协同指数到结果
        result['resonance_index'] = resonance_index
        
        # 3. 计算周期扩散指数 - 衡量各周期间的分歧程度
        if len(sorted_periods) >= 2:
            dispersion_index = pd.Series(0.0, index=self._result.index)
            
            for i in range(len(self._result)):
                if i < 5:
                    continue
                
                # 获取各周期当前值
                period_values = []
                for p in sorted_periods:
                    bias_col = f'BIAS{p}'
                    if bias_col in self._result.columns:
                        period_values.append(self._result[bias_col].iloc[i])
                
                if period_values:
                    # 计算当前值的标准差作为扩散度量
                    dispersion = np.std(period_values)
                    
                    # 归一化扩散指数 (0到1)
                    max_dispersion = 15  # 经验值，可调整
                    dispersion_index.iloc[i] = min(dispersion / max_dispersion, 1.0)
            
            result['dispersion_index'] = dispersion_index
        
        # 4. 计算共振信号强度
        result['resonance_strength'] = result['resonance_index'] * (1 - result.get('dispersion_index', 0))
        
        # 5. 生成共振信号
        resonance_signal = pd.Series(0, index=self._result.index)
        
        # 强共振阈值
        strong_resonance = 0.7
        
        # 处理每个数据点
        for i in range(5, len(self._result)):
            # 检查共振强度
            if result['resonance_strength'].iloc[i] > strong_resonance:
                # 判断方向
                direction_sum = 0
                for p in sorted_periods:
                    bias_col = f'BIAS{p}'
                    if bias_col in self._result.columns:
                        direction_sum += np.sign(self._result[bias_col].iloc[i])
                
                # 多数周期为正值表示做多信号，多数为负值表示做空信号
                if direction_sum > 0:
                    resonance_signal.iloc[i] = 1  # 做多信号
                elif direction_sum < 0:
                    resonance_signal.iloc[i] = -1  # 做空信号
        
        result['resonance_signal'] = resonance_signal
        
        return result

    def analyze_regression_rate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        分析BIAS回归均值的速率，量化动量特性
        
        Args:
            data: 输入数据
            
        Returns:
            pd.DataFrame: 回归速率分析结果
        """
        # 确保已计算BIAS
        if not self.has_result():
            self.calculate(data)
        
        # 初始化结果数据框
        result = pd.DataFrame(index=self._result.index)
        
        # 对每个周期进行分析
        for period in self.periods:
            bias_col = f'BIAS{period}'
            if bias_col not in self._result.columns:
                continue
            
            # 1. 计算回归速率 - BIAS向零轴回归的速度
            bias_abs = self._result[bias_col].abs()  # 偏离度
            bias_change = self._result[bias_col].diff()  # 变动率
            
            # 回归速率：BIAS向零轴方向的变动率与偏离度的比值
            regression_rate = pd.Series(index=self._result.index)
            for i in range(1, len(self._result)):
                current_bias = self._result[bias_col].iloc[i]
                current_change = bias_change.iloc[i]
                
                # 仅在回归方向时计算速率（朝零轴方向）
                if (current_bias > 0 and current_change < 0) or (current_bias < 0 and current_change > 0):
                    regression_rate.iloc[i] = abs(current_change) / (bias_abs.iloc[i] + 0.1)  # 避免除以零
                else:
                    regression_rate.iloc[i] = 0
            
            result[f'regression_rate_{period}'] = regression_rate
            
            # 2. 计算加速度 - 回归速率的变化率
            regression_accel = regression_rate.diff()
            result[f'regression_accel_{period}'] = regression_accel
            
            # 3. 计算回归效率 - 回归速率相对于历史的百分位
            if len(self._result) >= 20:
                regression_efficiency = pd.Series(index=self._result.index)
                
                for i in range(20, len(self._result)):
                    if regression_rate.iloc[i] > 0:  # 只在回归时计算效率
                        # 获取历史回归速率（只考虑正值）
                        hist_rates = regression_rate.iloc[i-20:i]
                        positive_rates = hist_rates[hist_rates > 0]
                        
                        if len(positive_rates) > 0:
                            # 计算当前速率在历史中的百分位
                            percentile = sum(regression_rate.iloc[i] >= positive_rates) / len(positive_rates)
                            regression_efficiency.iloc[i] = percentile
                
                result[f'regression_efficiency_{period}'] = regression_efficiency
            
            # 4. 计算回归持续性 - 连续回归的时间长度
            regression_duration = pd.Series(0, index=self._result.index)
            
            current_duration = 0
            for i in range(1, len(self._result)):
                current_bias = self._result[bias_col].iloc[i]
                current_change = bias_change.iloc[i]
                
                # 检查是否处于回归状态
                if (current_bias > 0 and current_change < 0) or (current_bias < 0 and current_change > 0):
                    current_duration += 1
                else:
                    current_duration = 0
                
                regression_duration.iloc[i] = current_duration
            
            result[f'regression_duration_{period}'] = regression_duration
            
            # 5. 计算临界点 - BIAS回归的关键阶段
            regression_critical = pd.Series(0, index=self._result.index)
            
            for i in range(20, len(self._result)):
                current_bias = self._result[bias_col].iloc[i]
                
                # 动态计算临界阈值
                hist_bias = self._result[bias_col].iloc[i-20:i]
                std = hist_bias.std()
                
                # 定义临界区域
                critical_zone = std * 0.5  # 在均值附近的临界区域
                
                # 标记临界点：1=从正值接近零轴，-1=从负值接近零轴，2=刚穿过零轴转正，-2=刚穿过零轴转负
                if 0 < current_bias < critical_zone:
                    regression_critical.iloc[i] = 1
                elif -critical_zone < current_bias < 0:
                    regression_critical.iloc[i] = -1
                elif current_bias > 0 and i > 0 and self._result[bias_col].iloc[i-1] <= 0:
                    regression_critical.iloc[i] = 2
                elif current_bias < 0 and i > 0 and self._result[bias_col].iloc[i-1] >= 0:
                    regression_critical.iloc[i] = -2
            
            result[f'regression_critical_{period}'] = regression_critical
        
        # 6. 综合回归评分 - 基于回归速率、效率和持续性的综合得分
        if len(self.periods) > 0:
            regression_score = pd.Series(0.0, index=self._result.index)
            
            for period in self.periods:
                rate_col = f'regression_rate_{period}'
                eff_col = f'regression_efficiency_{period}'
                dur_col = f'regression_duration_{period}'
                
                if rate_col in result.columns:
                    # 速率得分 (0-40分)
                    regression_score += result[rate_col] * 40
                
                if eff_col in result.columns:
                    # 效率得分 (0-30分)
                    regression_score += result[eff_col] * 30
                
                if dur_col in result.columns:
                    # 持续性得分 (0-30分)，最多考虑5天的持续性
                    duration_score = result[dur_col].clip(0, 5) / 5 * 30
                    regression_score += duration_score
            
            # 平均化得分
            regression_score = regression_score / len(self.periods)
            
            # 考虑BIAS的方向，确定最终得分的正负
            for period in self.periods:
                bias_col = f'BIAS{period}'
                if bias_col in self._result.columns:
                    # 为得分添加方向：正值BIAS回归得负分，负值BIAS回归得正分
                    regression_score = regression_score * (-np.sign(self._result[bias_col]))
                    break
            
            result['regression_score'] = regression_score
        
        return result

    def _register_bias_patterns(self):
        """注册BIAS特有的形态检测方法"""
        from indicators.pattern_registry import PatternRegistry, PatternType, PatternStrength
        
        # 获取PatternRegistry实例
        registry = PatternRegistry()
        
        # 注册BIAS极值形态
        registry.register(
            pattern_id="BIAS_EXTREME",
            display_name="BIAS极值",
            description="BIAS达到历史极端值，指示可能的超买或超卖状态",
            indicator_id="BIAS",
            pattern_type=PatternType.REVERSAL,
            default_strength=PatternStrength.STRONG,
            score_impact=15.0,
            detection_function=self._detect_bias_extreme_patterns
        )
        
        # 注册BIAS背离形态
        registry.register(
            pattern_id="BIAS_DIVERGENCE",
            display_name="BIAS背离",
            description="BIAS指标与价格走势形成背离，可能指示趋势反转",
            indicator_id="BIAS",
            pattern_type=PatternType.REVERSAL,
            default_strength=PatternStrength.STRONG,
            score_impact=20.0,
            detection_function=self._detect_bias_regression_patterns
        )

    def get_patterns(self, data: pd.DataFrame) -> List[PatternResult]:
        self.ensure_columns(data, ['bias'])
        patterns = []
        
        for pattern_func in self._pattern_registry.values():
            result = pattern_func(data)
            if result:
                patterns.append(result)
        
        return patterns



    
    def generate_trading_signals(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:

    
            """

    
            生成交易信号
        

    
            Args:

    
                data: 输入数据

    
                **kwargs: 额外参数
            

    
            Returns:

    
                Dict[str, pd.Series]: 包含交易信号的字典

    
            """

    
            # 确保已计算指标

    
            if not self.has_result():

    
                self.calculate(data, **kwargs)
            

    
            # 初始化信号

    
            signals = {}

    
            signals['buy_signal'] = pd.Series(False, index=data.index)

    
            signals['sell_signal'] = pd.Series(False, index=data.index)

    
            signals['signal_strength'] = pd.Series(0, index=data.index)
        

    
            # 在这里实现指标特定的信号生成逻辑

    
            # 此处提供默认实现
        

    
            return signals
