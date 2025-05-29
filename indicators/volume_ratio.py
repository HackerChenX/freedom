#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
量比指标(VOLUME_RATIO)
量比是指当前成交量与前N个周期平均成交量的比值，用于衡量市场交易活跃度的变化。
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, List, Dict, Any

from indicators.base_indicator import BaseIndicator

class VolumeRatio(BaseIndicator):
    """
    量比指标(VOLUME_RATIO)
    
    特点:
    1. 用于衡量市场交易活跃度的变化
    2. 量比>1表示当前成交量高于参考期平均值，市场相对活跃
    3. 量比<1表示当前成交量低于参考期平均值，市场相对冷清
    4. 通常与价格趋势结合使用，判断市场热度变化
    
    计算方法:
    量比 = 当前成交量 / 前N个周期平均成交量
    
    参数:
    - reference_period: 参考周期，默认为5
    - ma_period: 量比均线周期，默认为3
    """
    
    def __init__(self, reference_period: int = 5, ma_period: int = 3):
        """
        初始化量比指标
        
        Args:
            reference_period: 参考周期，默认为5
            ma_period: 量比均线周期，默认为3
        """
        super().__init__()
        self.reference_period = reference_period
        self.ma_period = ma_period
        self.name = f"VOLUME_RATIO({reference_period},{ma_period})"
        
    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """
        验证DataFrame是否包含计算所需的列
        
        Args:
            df: 数据源DataFrame
            
        Raises:
            ValueError: 如果DataFrame缺少所需的列
        """
        if 'volume' not in df.columns:
            raise ValueError("DataFrame必须包含'volume'列")
            
        if df['volume'].isnull().all():
            raise ValueError("所有成交量数据都是缺失的")
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算量比指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
            
        Returns:
            pd.DataFrame: 包含量比指标计算结果的DataFrame
        """
        self._validate_dataframe(df)
        
        # 创建结果DataFrame
        result = pd.DataFrame(index=df.index)
        
        # 获取成交量数据
        volume = df['volume'].values
        
        # 计算量比
        volume_ratio = np.zeros_like(volume, dtype=float)
        
        # 对数据足够长的情况进行计算
        for i in range(self.reference_period, len(volume)):
            ref_avg_volume = np.mean(volume[i-self.reference_period:i])
            if ref_avg_volume > 0:  # 避免除以零
                volume_ratio[i] = volume[i] / ref_avg_volume
            else:
                volume_ratio[i] = 1.0  # 默认为1，表示与参考期相同
        
        # 计算量比的移动平均
        volume_ratio_ma = pd.Series(volume_ratio).rolling(window=self.ma_period).mean().values
        
        # 保存结果
        result['volume_ratio'] = volume_ratio
        result['volume_ratio_ma'] = volume_ratio_ma
        
        return result
    
    def generate_signals(self, df: pd.DataFrame, result: pd.DataFrame, 
                         active_threshold: float = 1.5, quiet_threshold: float = 0.7) -> pd.DataFrame:
        """
        生成量比指标的交易信号
        
        Args:
            df: 原始数据DataFrame
            result: 包含量比计算结果的DataFrame
            active_threshold: 活跃市场阈值，默认为1.5
            quiet_threshold: 冷清市场阈值，默认为0.7
            
        Returns:
            pd.DataFrame: 添加了信号列的DataFrame
        """
        # 复制结果DataFrame
        signal_df = result.copy()
        
        # 获取量比值
        volume_ratio = signal_df['volume_ratio'].values
        volume_ratio_ma = signal_df['volume_ratio_ma'].values
        
        # 初始化信号数组
        bullish_signals = np.zeros_like(volume_ratio, dtype=bool)
        bearish_signals = np.zeros_like(volume_ratio, dtype=bool)
        
        # 生成信号
        for i in range(1, len(volume_ratio)):
            # 量比突然放大，可能是主力介入
            if volume_ratio[i] > active_threshold and volume_ratio[i-1] <= active_threshold:
                bullish_signals[i] = True
                
            # 量比突然萎缩，可能是主力撤出
            if volume_ratio[i] < quiet_threshold and volume_ratio[i-1] >= quiet_threshold:
                bearish_signals[i] = True
                
            # 量比上穿均线，市场活跃度提升
            if volume_ratio[i] > volume_ratio_ma[i] and volume_ratio[i-1] <= volume_ratio_ma[i-1]:
                bullish_signals[i] = True
                
            # 量比下穿均线，市场活跃度下降
            if volume_ratio[i] < volume_ratio_ma[i] and volume_ratio[i-1] >= volume_ratio_ma[i-1]:
                bearish_signals[i] = True
        
        # 保存信号
        signal_df['bullish_signal'] = bullish_signals
        signal_df['bearish_signal'] = bearish_signals
        
        # 添加市场状态分析
        signal_df['market_state'] = np.where(volume_ratio > active_threshold, 'active',
                                   np.where(volume_ratio < quiet_threshold, 'quiet', 'normal'))
        
        return signal_df
    
    def plot(self, df: pd.DataFrame, result: pd.DataFrame, ax=None):
        """
        绘制量比指标图表
        
        Args:
            df: 原始数据DataFrame
            result: 包含量比指标计算结果的DataFrame
            ax: matplotlib轴对象，如果为None则创建新的
            
        Returns:
            matplotlib.axes.Axes: 绘制好的轴对象
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("警告: matplotlib不可用，无法绘制图表")
            return None
            
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        
        # 获取数据
        dates = df.index
        volume_ratio = result['volume_ratio'].values
        volume_ratio_ma = result['volume_ratio_ma'].values
        
        # 绘制量比线
        ax.plot(dates, volume_ratio, label=f"量比({self.reference_period}日)", color='blue', linewidth=1.5)
        ax.plot(dates, volume_ratio_ma, label=f"量比均线({self.ma_period}日)", color='red', linewidth=1.2)
        
        # 绘制参考线
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=1.5, color='green', linestyle='--', alpha=0.3)
        ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.3)
        
        # 获取信号
        if 'bullish_signal' in result.columns and 'bearish_signal' in result.columns:
            bullish_signals = result['bullish_signal'].values
            bearish_signals = result['bearish_signal'].values
            
            # 绘制买入信号
            buy_dates = [dates[i] for i in range(len(dates)) if bullish_signals[i]]
            buy_values = [volume_ratio[i] for i in range(len(volume_ratio)) if bullish_signals[i]]
            ax.scatter(buy_dates, buy_values, color='green', s=50, marker='^', label='买入信号')
            
            # 绘制卖出信号
            sell_dates = [dates[i] for i in range(len(dates)) if bearish_signals[i]]
            sell_values = [volume_ratio[i] for i in range(len(volume_ratio)) if bearish_signals[i]]
            ax.scatter(sell_dates, sell_values, color='red', s=50, marker='v', label='卖出信号')
        
        # 设置图表属性
        ax.set_title(f"量比指标 (参考期: {self.reference_period}日)")
        ax.set_ylabel('量比值')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算并生成量比指标信号
        
        Args:
            df: 包含OHLCV数据的DataFrame
            
        Returns:
            pd.DataFrame: 包含量比指标和信号的DataFrame
        """
        result = self.calculate(df)
        signal_df = self.generate_signals(df, result)
        
        return signal_df
    
    def get_market_activity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        获取市场活跃度评估
        
        Args:
            df: 包含OHLCV数据的DataFrame
            
        Returns:
            pd.DataFrame: 包含市场活跃度评估的DataFrame
        """
        # 计算量比
        result = self.compute(df)
        
        # 创建活跃度评估DataFrame
        activity_df = pd.DataFrame(index=df.index)
        
        # 量比值
        volume_ratio = result['volume_ratio'].values
        
        # 评估市场活跃度
        activity_df['market_activity'] = pd.cut(
            volume_ratio,
            bins=[-np.inf, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0, np.inf],
            labels=['极度萎缩', '严重萎缩', '轻度萎缩', '正常', '轻度活跃', '中度活跃', '高度活跃', '极度活跃']
        )
        
        # 添加数值型指标
        activity_df['activity_score'] = np.where(volume_ratio > 1, 
                                               (volume_ratio - 1) * 50 + 50, 
                                               volume_ratio * 50)
        
        # 量比变化趋势 (5日)
        if len(volume_ratio) >= 5:
            recent_ratio = volume_ratio[-5:]
            trend = np.polyfit(range(len(recent_ratio)), recent_ratio, 1)[0]
            activity_df.loc[activity_df.index[-1], 'trend'] = '上升' if trend > 0 else '下降'
            activity_df.loc[activity_df.index[-1], 'trend_strength'] = abs(trend) * 10
        
        return activity_df 