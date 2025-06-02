#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
成交量(VOL)

市场活跃度、参与度直观体现
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple, Any
from scipy import signal, stats
import warnings

from indicators.base_indicator import BaseIndicator, PatternResult
from indicators.common import crossover, crossunder
from utils.logger import get_logger

# 静默警告
warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = get_logger(__name__)


class VOL(BaseIndicator):
    """
    成交量(VOL) (VOL)
    
    分类：量能类指标
    描述：市场活跃度、参与度直观体现
    """
    
    def __init__(self, period: int = 14, enable_cycles_analysis: bool = True, enable_standardization: bool = True):
        """
        初始化成交量(VOL)指标
        
        Args:
            period: 计算周期，默认为14
            enable_cycles_analysis: 是否启用量能周期分析，默认启用
            enable_standardization: 是否启用成交量标准化，默认启用
        """
        super().__init__()
        self.period = period
        self.name = "VOL"
        self.enable_cycles_analysis = enable_cycles_analysis
        self.enable_standardization = enable_standardization
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算成交量指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
                
        Returns:
            包含VOL指标的DataFrame
        """
        return self.calculate(df)
        
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算成交量(VOL)指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
                必须包含以下列：
                - volume: 成交量
                
        Returns:
            添加了VOL指标列的DataFrame
        """
        if df.empty:
            return df
            
        # 确保数据包含必要的列
        required_columns = ['volume']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 添加原始成交量
        df_copy['vol'] = df_copy['volume']
        
        # 计算成交量移动平均
        df_copy['vol_ma5'] = df_copy['volume'].rolling(window=5).mean()
        df_copy['vol_ma10'] = df_copy['volume'].rolling(window=10).mean()
        df_copy['vol_ma20'] = df_copy['volume'].rolling(window=20).mean()
        
        # 计算相对成交量（当前成交量与N日平均成交量的比值）
        df_copy['vol_ratio'] = df_copy['volume'] / df_copy['vol_ma5']
        
        # 优化: 计算相对成交量变化率
        df_copy['vol_ratio_change'] = df_copy['vol_ratio'].pct_change()
        
        # 优化: 计算成交量波动率
        df_copy['vol_std'] = df_copy['volume'].rolling(window=20).std() / df_copy['vol_ma20']
        
        # 优化: 计算相对于60日平均的成交量比
        if len(df_copy) >= 60:
            df_copy['vol_ma60'] = df_copy['volume'].rolling(window=60).mean()
            df_copy['vol_ratio_60'] = df_copy['volume'] / df_copy['vol_ma60']
        else:
            df_copy['vol_ma60'] = df_copy['vol_ma20']  # 数据不足时使用20日均量代替
            df_copy['vol_ratio_60'] = df_copy['volume'] / df_copy['vol_ma60']
        
        # 新增: 计算成交量加速度
        df_copy['vol_acceleration'] = df_copy['volume'].pct_change().diff()
        
        # 新增: 计算短期相对长期的波动率比率
        if len(df_copy) >= 60:
            df_copy['vol_std_5'] = df_copy['volume'].rolling(window=5).std() / df_copy['vol_ma5']
            df_copy['vol_std_60'] = df_copy['volume'].rolling(window=60).std() / df_copy['vol_ma60']
            df_copy['vol_std_ratio'] = df_copy['vol_std_5'] / df_copy['vol_std_60']
        
        # 新增: 应用相对成交量标准化
        if self.enable_standardization:
            df_copy = self._calculate_standardized_relative_volume(df, df_copy)
        
        # 新增: 分析成交量周期性
        if self.enable_cycles_analysis and len(df_copy) >= 60:
            df_copy = self._analyze_volume_cycles(df_copy)
        
        # 保存结果
        self._result = df_copy
        
        return df_copy
        
    def get_signals(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成成交量(VOL)指标交易信号
        
        Args:
            df: 包含价格数据和VOL指标的DataFrame
            **kwargs: 额外参数
                vol_ratio_threshold: 相对成交量阈值，默认为1.5
                
        Returns:
            添加了信号列的DataFrame:
            - vol_signal: 1=放量信号, -1=缩量信号, 0=无信号
        """
        if df.empty:
            return df
            
        # 检查必要的指标列是否存在
        required_columns = ['vol', 'vol_ma5']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 获取参数
        vol_ratio_threshold = kwargs.get('vol_ratio_threshold', 1.5)  # 相对成交量阈值
        
        # 生成信号
        df_copy['vol_signal'] = 0
        
        # 放量信号（成交量大于N日平均的1.5倍）
        df_copy.loc[df_copy['vol_ratio'] > vol_ratio_threshold, 'vol_signal'] = 1
        
        # 缩量信号（成交量小于N日平均的0.5倍）
        df_copy.loc[df_copy['vol_ratio'] < 0.5, 'vol_signal'] = -1
        
        return df_copy
    
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
        
    def plot(self, df: pd.DataFrame, ax=None, **kwargs):
        """
        绘制成交量(VOL)指标图表
        
        Args:
            df: 包含VOL指标的DataFrame
            ax: matplotlib轴对象，如果为None则创建新的
            **kwargs: 额外绘图参数
            
        Returns:
            matplotlib轴对象
        """
        import matplotlib.pyplot as plt
        
        # 检查必要的指标列是否存在
        required_columns = ['vol', 'vol_ma5', 'vol_ma10']
        self._validate_dataframe(df, required_columns)
        
        # 创建新的轴对象（如果未提供）
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
            
        # 绘制VOL指标线
        ax.bar(df.index, df['vol'], label='成交量', alpha=0.3, color='gray')
        ax.plot(df.index, df['vol_ma5'], label='5日均量', color='red')
        ax.plot(df.index, df['vol_ma10'], label='10日均量', color='blue')
        ax.plot(df.index, df['vol_ma20'], label='20日均量', color='green')
        
        ax.set_ylabel('成交量')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        return ax

    def calculate_raw_score(self, data: pd.DataFrame) -> pd.Series:
        """
        计算成交量指标的原始评分
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            pd.Series: 包含原始评分的Series
        """
        # 计算指标值
        indicator_data = self.calculate(data)
        
        # 初始化评分
        score = pd.Series(50.0, index=data.index)  # 基础分50分
        
        # 1. 成交量水平评分
        volume_level_score = self._calculate_volume_level_score(indicator_data)
        score += volume_level_score
        
        # 2. 量价配合评分
        price_volume_score = self._calculate_price_volume_harmony(data, indicator_data)
        score += price_volume_score
        
        # 3. 成交量趋势评分
        volume_trend_score = self._calculate_volume_trend_score(indicator_data)
        score += volume_trend_score
        
        # 4. 相对量比评估 (优化)
        relative_volume_score = self._calculate_relative_volume_score(indicator_data)
        score += relative_volume_score
        
        # 5. 异常放量识别 (增强版)
        abnormal_volume_score = self._detect_abnormal_volume(data, indicator_data)
        score += abnormal_volume_score
        
        return np.clip(score, 0, 100)

    def _calculate_volume_level_score(self, indicator_data: pd.DataFrame) -> pd.Series:
        """
        计算成交量水平评分
        
        Args:
            indicator_data: 包含成交量指标的DataFrame
            
        Returns:
            pd.Series: 成交量水平评分
        """
        score = pd.Series(0.0, index=indicator_data.index)
        
        # 获取成交量数据
        vol_ratio = indicator_data['vol_ratio'].fillna(1.0)
        
        # 放量加分，缩量减分
        high_volume_mask = vol_ratio > 1.5
        score.loc[high_volume_mask] += 15
        
        very_high_volume_mask = vol_ratio > 2.0
        score.loc[very_high_volume_mask] += 25
        
        low_volume_mask = vol_ratio < 0.7
        score.loc[low_volume_mask] -= 10
        
        very_low_volume_mask = vol_ratio < 0.5
        score.loc[very_low_volume_mask] -= 15
        
        return score

    def _calculate_price_volume_harmony(self, data: pd.DataFrame, indicator_data: pd.DataFrame) -> pd.Series:
        """
        计算量价配合评分
        
        Args:
            data: 原始数据DataFrame
            indicator_data: 包含成交量指标的DataFrame
            
        Returns:
            pd.Series: 量价配合评分
        """
        score = pd.Series(0.0, index=indicator_data.index)
        
        # 获取成交量数据
        vol_ratio = indicator_data['vol_ratio'].fillna(1.0)
        
        if 'close' in data.columns:
            close_price = data['close']
            price_change = close_price.pct_change().fillna(0)
            
            # 价涨量增（理想状态）
            price_up_vol_up = (price_change > 0.02) & (vol_ratio > 1.2)
            score.loc[price_up_vol_up] += 20
            
            # 价涨量增（强势）
            strong_price_up_vol_up = (price_change > 0.05) & (vol_ratio > 1.5)
            score.loc[strong_price_up_vol_up] += 25
            
            # 价跌量增（警告信号）
            price_down_vol_up = (price_change < -0.02) & (vol_ratio > 1.2)
            score.loc[price_down_vol_up] -= 15
            
            # 价跌量增（恐慌信号）
            panic_price_down_vol_up = (price_change < -0.05) & (vol_ratio > 1.5)
            score.loc[panic_price_down_vol_up] -= 20
            
            # 价涨量缩（警告信号）
            price_up_vol_down = (price_change > 0.02) & (vol_ratio < 0.8)
            score.loc[price_up_vol_down] -= 10
            
            # 价跌量缩（可能见底）
            price_down_vol_down = (price_change < -0.02) & (vol_ratio < 0.8)
            score.loc[price_down_vol_down] += 10
        
        return score

    def _calculate_volume_trend_score(self, indicator_data: pd.DataFrame) -> pd.Series:
        """
        计算成交量趋势评分
        
        Args:
            indicator_data: 包含成交量指标的DataFrame
            
        Returns:
            pd.Series: 成交量趋势评分
        """
        score = pd.Series(0.0, index=indicator_data.index)
        
        # 获取成交量数据
        volume = indicator_data['vol']
        
        # 计算成交量趋势
        vol_trend_5 = volume.rolling(window=5).mean().pct_change().fillna(0)
        vol_trend_10 = volume.rolling(window=10).mean().pct_change().fillna(0)
        
        # 持续放量
        sustained_volume_up = (vol_trend_5 > 0.1) & (vol_trend_10 > 0.05)
        score.loc[sustained_volume_up] += 10
        
        # 急剧放量
        sharp_volume_up = vol_trend_5 > 0.3
        score.loc[sharp_volume_up] += 15
        
        # 持续缩量
        sustained_volume_down = (vol_trend_5 < -0.1) & (vol_trend_10 < -0.05)
        score.loc[sustained_volume_down] -= 10
        
        # 急剧缩量
        sharp_volume_down = vol_trend_5 < -0.3
        score.loc[sharp_volume_down] -= 15
        
        return score

    def _calculate_relative_volume_score(self, indicator_data: pd.DataFrame) -> pd.Series:
        """
        计算相对量比评分 (优化点)
        
        将当前成交量与不同周期的移动平均进行比较，以识别真实的量能变化
        
        Args:
            indicator_data: 包含成交量指标的DataFrame
            
        Returns:
            pd.Series: 相对量比评分
        """
        score = pd.Series(0.0, index=indicator_data.index)
        
        # 确保需要的列存在
        required_cols = ['vol_ratio', 'vol_ratio_60', 'vol_ratio_change']
        for col in required_cols:
            if col not in indicator_data.columns:
                return score
        
        # 1. 多周期相对量比评分
        vol_ratio = indicator_data['vol_ratio'].fillna(1.0)       # 相对5日均量
        vol_ratio_60 = indicator_data['vol_ratio_60'].fillna(1.0) # 相对60日均量
        
        # 计算权重：长期比值(60日)相对于短期比值(5日)的比例
        relative_strength = vol_ratio_60 / vol_ratio
        
        # 权重系数(1-1.5)，用于调整评分
        weight_coef = np.clip(1 + (relative_strength - 1) * 2, 0.5, 1.5)
        
        # 长短周期量比一致性得分
        # 相对60日和5日均量同时放大，信号更强
        consistent_high_volume = (vol_ratio > 1.3) & (vol_ratio_60 > 1.5)
        score.loc[consistent_high_volume] += 15 * weight_coef.loc[consistent_high_volume]
        
        # 相对60日放大更多，表明真正的量能释放
        strong_long_term_volume = (vol_ratio_60 > vol_ratio) & (vol_ratio_60 > 2.0)
        score.loc[strong_long_term_volume] += 20
        
        # 2. 相对量比变化趋势评分
        vol_ratio_change = indicator_data['vol_ratio_change'].fillna(0)
        
        # 计算相对量比的加速度（变化率的变化率）
        vol_ratio_accel = vol_ratio_change.diff().fillna(0)
        
        # 相对量比加速上升，预示主力资金加速介入
        vol_ratio_accelerating = (vol_ratio_change > 0.1) & (vol_ratio_accel > 0)
        score.loc[vol_ratio_accelerating] += 10
        
        # 相对量比加速下降，预示主力资金加速撤离
        vol_ratio_decelerating = (vol_ratio_change < -0.1) & (vol_ratio_accel < 0)
        score.loc[vol_ratio_decelerating] -= 10
        
        return score

    def _detect_abnormal_volume(self, data: pd.DataFrame, indicator_data: pd.DataFrame) -> pd.Series:
        """
        异常放量识别 (增强版)
        
        检测不符合历史模式的异常成交量，可能预示重要转折点或主力行为
        优化实现：精细化异常交易分类，提高异常交易识别准确率
        
        Args:
            data: 原始数据DataFrame
            indicator_data: 包含成交量指标的DataFrame
            
        Returns:
            pd.Series: 异常放量评分
        """
        score = pd.Series(0.0, index=indicator_data.index)
        
        # 确保需要的列存在
        if 'vol' not in indicator_data.columns or 'vol_std' not in indicator_data.columns:
            return score
        
        volume = indicator_data['vol']
        vol_ma5 = indicator_data['vol_ma5'].fillna(volume)
        vol_ma20 = indicator_data['vol_ma20'].fillna(volume)
        vol_std = indicator_data['vol_std'].fillna(0.2)  # 默认波动率
        
        # 1. 异常放量识别 - 基于标准化分数
        # 使用标准化相对量比而不是简单z-score
        if 'vol_z_score' in indicator_data.columns:
            z_score = indicator_data['vol_z_score']
        else:
            # 计算Z分数（成交量偏离均值的标准差倍数）
            z_score = (volume - vol_ma20) / (vol_ma20 * vol_std)
        
        # 超过3个标准差的异常放量
        extreme_volume = z_score > 3
        score.loc[extreme_volume] += 30
        
        # 超过2个标准差的高成交量
        high_volume = (z_score > 2) & (z_score <= 3)
        score.loc[high_volume] += 20
        
        # 新增: 各种异常成交量分类评分
        if len(volume) >= 20:
            # 2. 异常成交量类型细分
            # 2.1 持续放量型 - 连续3天以上成交量增加
            if len(volume) >= 5:
                vol_change = volume.pct_change()
                continuous_up = ((vol_change > 0).rolling(window=3).sum() >= 3) & (z_score > 1.5)
                score.loc[continuous_up] += 15
                
                # 持续放量且加速
                vol_accel = volume.pct_change().diff()
                continuous_accel = continuous_up & (vol_accel > 0)
                score.loc[continuous_accel] += 10
            
            # 2.2 脉冲放量型 - 单日巨量
            pulse_volume = (volume / volume.shift(1) > 3) & (z_score > 2.5)
            score.loc[pulse_volume] += 25
            
            # 2.3 平台突破型 - 成交量长期低迷后突然放大
            if len(volume) >= 30:
                vol_ma30 = volume.rolling(window=30).mean()
                platform_breakout = (volume > vol_ma30 * 2) & (vol_ma5.shift(5) < vol_ma30.shift(5) * 0.8)
                score.loc[platform_breakout] += 25
            
            # 2.4 筑底放量型 - 股价接近阶段低点时的放量
            if 'low' in data.columns and len(data) >= 30:
                low_price = data['low']
                low_30d = low_price.rolling(window=30).min()
                bottom_volume = (low_price < low_30d * 1.05) & (volume > vol_ma20 * 1.5)
                score.loc[bottom_volume] += 20
            
            # 2.5 顶部放量型 - 股价接近阶段高点时的放量
            if 'high' in data.columns and len(data) >= 30:
                high_price = data['high']
                high_30d = high_price.rolling(window=30).max()
                top_volume = (high_price > high_30d * 0.95) & (volume > vol_ma20 * 1.5)
                score.loc[top_volume] -= 15
        
        # 3. 价格关键位突破的放量确认
        if 'close' in data.columns and 'high' in data.columns and 'low' in data.columns:
            close_price = data['close']
            high_price = data['high']
            low_price = data['low']
            
            # 计算20日高点和低点
            high_20d = high_price.rolling(window=20).max().shift(1)
            low_20d = low_price.rolling(window=20).min().shift(1)
            
            # 向上突破20日高点且放量
            upside_breakout = (close_price > high_20d) & (z_score > 1.5)
            score.loc[upside_breakout] += 25
            
            # 向下突破20日低点且放量
            downside_breakout = (close_price < low_20d) & (z_score > 1.5)
            score.loc[downside_breakout] -= 25
            
            # 新增: 支撑位确认的放量
            support_confirm = (low_price < low_20d * 1.02) & (close_price > low_20d) & (volume > vol_ma10 * 1.3)
            score.loc[support_confirm] += 20
            
            # 新增: 阻力位确认的放量
            resistance_confirm = (high_price > high_20d * 0.98) & (close_price < high_20d) & (volume > vol_ma10 * 1.3)
            score.loc[resistance_confirm] -= 20
        
        # 4. 成交量断层识别
        vol_ratio = volume / volume.shift(1)
        vol_gap = (vol_ratio > 3) & (volume > vol_ma20 * 1.5)
        score.loc[vol_gap] += 15
        
        # 新增: 成交量节奏变化识别
        if 'vol_std_ratio' in indicator_data.columns:
            # 波动率剧烈提升 - 可能是主力活跃度提升
            vol_volatility_spike = indicator_data['vol_std_ratio'] > 2
            score.loc[vol_volatility_spike] += 10
            
            # 波动率剧烈降低 - 可能是主力活跃度降低
            vol_volatility_drop = indicator_data['vol_std_ratio'] < 0.5
            score.loc[vol_volatility_drop] -= 5
        
        # 新增: 使用标准化分数的全局视角
        if 'vol_std_score' in indicator_data.columns:
            std_score = indicator_data['vol_std_score']
            
            # 极高标准化分数 (> 90)
            extreme_high_std = std_score > 90
            score.loc[extreme_high_std] += 25
            
            # 极低标准化分数 (< 10)
            extreme_low_std = std_score < 10
            score.loc[extreme_low_std] -= 15
        
        # 新增: 历史分位数评估
        for period in [20, 60, 120]:
            percentile_col = f'vol_percentile_{period}d'
            if percentile_col in indicator_data.columns:
                # 处于历史90%分位以上的成交量
                high_percentile = indicator_data[percentile_col] > 90
                score.loc[high_percentile] += min(period/10, 10)  # 较长期的高分位更有价值
                
                # 处于历史10%分位以下的成交量
                low_percentile = indicator_data[percentile_col] < 10
                score.loc[low_percentile] -= min(period/15, 7)
        
        return score
    
    def identify_patterns(self, data: pd.DataFrame) -> List[str]:
        """
        识别成交量相关的技术形态
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            List[str]: 识别出的形态列表
        """
        patterns = []
        
        # 计算指标值
        indicator_data = self.calculate(data)
        
        if len(indicator_data) < 10:
            return patterns
        
        # 获取成交量数据
        volume = indicator_data['vol']
        vol_ma5 = indicator_data['vol_ma5']
        vol_ma10 = indicator_data['vol_ma10']
        vol_ma20 = indicator_data['vol_ma20']
        vol_ratio = indicator_data['vol_ratio']
        
        # 获取最新数据
        latest_vol = volume.iloc[-1]
        latest_vol_ratio = vol_ratio.iloc[-1]
        
        # 1. 成交量水平形态
        if pd.notna(latest_vol_ratio):
            if latest_vol_ratio > 3.0:
                patterns.append("巨量")
            elif latest_vol_ratio > 2.0:
                patterns.append("大幅放量")
            elif latest_vol_ratio > 1.5:
                patterns.append("放量")
            elif latest_vol_ratio < 0.3:
                patterns.append("极度缩量")
            elif latest_vol_ratio < 0.5:
                patterns.append("明显缩量")
            elif latest_vol_ratio < 0.7:
                patterns.append("缩量")
        
        # 2. 量价配合形态
        if 'close' in data.columns and len(data) >= 2:
            close_price = data['close']
            price_change = close_price.pct_change().iloc[-1]
            
            if pd.notna(price_change) and pd.notna(latest_vol_ratio):
                if price_change > 0.02 and latest_vol_ratio > 1.2:
                    if latest_vol_ratio > 2.0:
                        patterns.append("价涨量增强势")
                    else:
                        patterns.append("价涨量增")
                elif price_change < -0.02 and latest_vol_ratio > 1.2:
                    patterns.append("价跌量增")
                elif price_change > 0.02 and latest_vol_ratio < 0.8:
                    patterns.append("价涨量缩")
                elif price_change < -0.02 and latest_vol_ratio < 0.8:
                    patterns.append("价跌量缩")
        
        # 3. 成交量趋势形态
        if len(volume) >= 5:
            recent_vol_trend = volume.tail(5).pct_change().mean()
            
            if pd.notna(recent_vol_trend):
                if recent_vol_trend > 0.2:
                    patterns.append("成交量持续放大")
                elif recent_vol_trend > 0.1:
                    patterns.append("成交量温和放大")
                elif recent_vol_trend < -0.2:
                    patterns.append("成交量持续萎缩")
                elif recent_vol_trend < -0.1:
                    patterns.append("成交量温和萎缩")
        
        # 4. 成交量突破形态
        if len(volume) >= 20:
            vol_20_max = volume.tail(20).max()
            vol_20_avg = volume.tail(20).mean()
            
            if pd.notna(latest_vol) and pd.notna(vol_20_max) and pd.notna(vol_20_avg):
                if latest_vol >= vol_20_max:
                    patterns.append("成交量创新高")
                elif latest_vol >= vol_20_max * 0.9:
                    patterns.append("成交量接近新高")
                
                if latest_vol > vol_20_avg * 2:
                    patterns.append("突破性放量")
        
        # 5. 成交量异常形态
        if len(volume) >= 20:
            vol_std = volume.tail(20).std()
            vol_mean = volume.tail(20).mean()
            
            if pd.notna(latest_vol) and pd.notna(vol_std) and pd.notna(vol_mean):
                if latest_vol > vol_mean + 3 * vol_std:
                    patterns.append("极端异常放量")
                elif latest_vol > vol_mean + 2 * vol_std:
                    patterns.append("异常放量")
                elif latest_vol < vol_mean - 3 * vol_std:
                    patterns.append("极端异常缩量")
                elif latest_vol < vol_mean - 2 * vol_std:
                    patterns.append("异常缩量")
                    
                # 新增: 使用标准化分数识别异常
                if 'vol_std_score' in indicator_data.columns:
                    latest_std_score = indicator_data['vol_std_score'].iloc[-1]
                    if pd.notna(latest_std_score):
                        if latest_std_score > 90:
                            patterns.append("极高标准化成交量")
                        elif latest_std_score < 10:
                            patterns.append("极低标准化成交量")
        
        # 6. 成交量均线形态
        if (pd.notna(vol_ma5.iloc[-1]) and pd.notna(vol_ma10.iloc[-1]) and 
            pd.notna(vol_ma20.iloc[-1])):
            
            if vol_ma5.iloc[-1] > vol_ma10.iloc[-1] > vol_ma20.iloc[-1]:
                patterns.append("成交量多头排列")
            elif vol_ma5.iloc[-1] < vol_ma10.iloc[-1] < vol_ma20.iloc[-1]:
                patterns.append("成交量空头排列")
            
            if latest_vol > vol_ma5.iloc[-1]:
                patterns.append("成交量高于短期均线")
            elif latest_vol < vol_ma5.iloc[-1]:
                patterns.append("成交量低于短期均线")
        
        # 7. 成交量变化形态
        if len(volume) >= 3:
            # 连续放量
            recent_3_vol = volume.tail(3)
            if (recent_3_vol.iloc[-1] > recent_3_vol.iloc[-2] and 
                recent_3_vol.iloc[-2] > recent_3_vol.iloc[-3]):
                patterns.append("连续放量")
            
            # 连续缩量
            if (recent_3_vol.iloc[-1] < recent_3_vol.iloc[-2] and 
                recent_3_vol.iloc[-2] < recent_3_vol.iloc[-3]):
                patterns.append("连续缩量")
        
        # 8. 成交量极值形态
        if len(volume) >= 60:
            vol_60_max = volume.tail(60).max()
            vol_60_min = volume.tail(60).min()
            
            if pd.notna(latest_vol):
                if latest_vol >= vol_60_max:
                    patterns.append("60日成交量新高")
                elif latest_vol <= vol_60_min:
                    patterns.append("60日成交量新低")
                    
        # 9. 新增: 成交量周期性形态
        # 识别当前处于量能周期的哪个阶段
        if 'vol_cycle_phase' in indicator_data.columns:
            cycle_phase = indicator_data['vol_cycle_phase'].iloc[-1]
            if pd.notna(cycle_phase):
                if cycle_phase == 'low_volume':
                    patterns.append("量能周期低点")
                elif cycle_phase == 'rising_volume':
                    patterns.append("量能周期上升阶段")
                elif cycle_phase == 'high_volume':
                    patterns.append("量能周期高点")
                elif cycle_phase == 'falling_volume':
                    patterns.append("量能周期下降阶段")
        
        # 10. 新增: 成交量周期拐点形态
        if 'vol_acf_main_cycle' in indicator_data.columns and 'vol_acf_cycle_pos' in indicator_data.columns:
            main_cycle = indicator_data['vol_acf_main_cycle'].iloc[-1]
            cycle_pos = indicator_data['vol_acf_cycle_pos'].iloc[-1]
            
            if pd.notna(main_cycle) and pd.notna(cycle_pos):
                if cycle_pos < 5 or cycle_pos > 95:
                    patterns.append("量能周期拐点")
                elif 45 < cycle_pos < 55:
                    patterns.append("量能周期中点")
        
        # 11. 新增: 成交量分位数形态
        for period in [20, 60, 120]:
            percentile_col = f'vol_percentile_{period}d'
            if percentile_col in indicator_data.columns:
                latest_percentile = indicator_data[percentile_col].iloc[-1]
                
                if pd.notna(latest_percentile):
                    if latest_percentile > 90:
                        patterns.append(f"{period}日成交量极高分位")
                    elif latest_percentile < 10:
                        patterns.append(f"{period}日成交量极低分位")
        
        # 12. 新增: 市值标准化成交量形态
        if 'vol_mcap_ratio' in indicator_data.columns:
            mcap_ratio = indicator_data['vol_mcap_ratio'].iloc[-1]
            
            if pd.notna(mcap_ratio):
                # 需要设置合理的阈值，这里假设使用相对值
                if len(indicator_data) >= 60:
                    mcap_ratio_mean = indicator_data['vol_mcap_ratio'].tail(60).mean()
                    mcap_ratio_std = indicator_data['vol_mcap_ratio'].tail(60).std()
                    
                    if mcap_ratio > mcap_ratio_mean + 2 * mcap_ratio_std:
                        patterns.append("市值标准化成交量异常高")
                    elif mcap_ratio < mcap_ratio_mean - 2 * mcap_ratio_std:
                        patterns.append("市值标准化成交量异常低")
        
        # 13. 新增: 波动率调整后的成交量形态
        if 'vol_ratio_adj' in indicator_data.columns:
            ratio_adj = indicator_data['vol_ratio_adj'].iloc[-1]
            
            if pd.notna(ratio_adj):
                if ratio_adj > 2.0:
                    patterns.append("波动率调整后显著放量")
                elif ratio_adj < 0.5:
                    patterns.append("波动率调整后显著缩量")
        
        # 14. 新增: 成交量日内分布异常形态
        if 'intraday_vol_std' in indicator_data.columns:
            intraday_std = indicator_data['intraday_vol_std'].iloc[-1]
            
            if pd.notna(intraday_std):
                if len(indicator_data) >= 20:
                    intraday_std_mean = indicator_data['intraday_vol_std'].tail(20).mean()
                    
                    if intraday_std > intraday_std_mean * 2:
                        patterns.append("日内成交量分布异常")
        
        return patterns

    def _calculate_standardized_relative_volume(self, data: pd.DataFrame, indicator_data: pd.DataFrame) -> pd.DataFrame:
        """
        计算标准化相对成交量
        
        标准化相对成交量可以消除不同股票间的规模差异，便于跨股票比较
        实现相对成交量标准化功能，提高跨股票比较准确性
        
        Args:
            data: 原始数据DataFrame
            indicator_data: 包含成交量指标的DataFrame
            
        Returns:
            pd.DataFrame: 添加了标准化相对成交量的DataFrame
        """
        if indicator_data.empty:
            return indicator_data
            
        df_copy = indicator_data.copy()
        
        # 1. 计算成交量的市值标准化
        if 'close' in data.columns:
            # 如果有市值数据，使用市值标准化
            if 'market_cap' in data.columns:
                df_copy['vol_mcap_ratio'] = df_copy['volume'] / data['market_cap']
            # 否则使用收盘价作为近似
            else:
                df_copy['vol_price_ratio'] = df_copy['volume'] / data['close']
        
        # 2. 计算成交量的历史分位数
        lookback_periods = [20, 60, 120]
        for period in lookback_periods:
            if len(df_copy) >= period:
                # 计算N日成交量分位数
                df_copy[f'vol_percentile_{period}d'] = df_copy['volume'].rolling(window=period).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100,
                    raw=False
                )
        
        # 3. 计算成交量的行业相对值
        # 注意：这需要行业数据，这里假设没有行业数据，实际应用中可扩展
        
        # 4. 计算日内标准化成交量 (如果有分时数据)
        if 'intraday_vol' in data.columns:
            # 假设intraday_vol是日内每分钟的成交量列表
            intraday_vol = data['intraday_vol'].fillna(pd.Series([[]] * len(data)))
            
            # 计算日内成交量分布标准差
            def calc_intraday_std(vol_list):
                if not vol_list or len(vol_list) == 0:
                    return np.nan
                return np.std(vol_list) / (np.mean(vol_list) if np.mean(vol_list) > 0 else 1)
            
            df_copy['intraday_vol_std'] = intraday_vol.apply(calc_intraday_std)
        
        # 5. 计算标准化相对量比 (关键指标)
        # 使用z-score标准化当前成交量相对于历史均值的偏离程度
        if 'vol_std' in df_copy.columns and 'vol_ma20' in df_copy.columns:
            # 标准化z-score
            df_copy['vol_z_score'] = (df_copy['volume'] - df_copy['vol_ma20']) / (df_copy['vol_ma20'] * df_copy['vol_std'])
            
            # 限制极端值
            df_copy['vol_z_score'] = df_copy['vol_z_score'].clip(-4, 4)
            
            # 转换为0-100的标准化分数
            df_copy['vol_std_score'] = (df_copy['vol_z_score'] + 4) * 12.5
        
        # 6. 计算波动率调整后的相对成交量
        if 'vol_std' in df_copy.columns:
            # 波动率调整后的相对成交量
            df_copy['vol_ratio_adj'] = df_copy['vol_ratio'] / (1 + df_copy['vol_std'])
        
        return df_copy

    def _analyze_volume_cycles(self, indicator_data: pd.DataFrame, min_periods: int = 60) -> pd.DataFrame:
        """
        分析成交量的周期性特征
        
        识别成交量的周期性模式，例如季节性、周期性波动等
        
        Args:
            indicator_data: 包含成交量指标的DataFrame
            min_periods: 最小所需的数据点数，默认60
            
        Returns:
            pd.DataFrame: 添加了周期性特征的DataFrame
        """
        # 确保有足够的数据点
        if len(indicator_data) < min_periods or 'vol' not in indicator_data.columns:
            return indicator_data
            
        df_copy = indicator_data.copy()
        volume = df_copy['vol']
        
        # 1. 傅里叶变换分析成交量周期
        if len(volume) >= 120:  # 需要较长时间序列
            try:
                from scipy import signal
                
                # 去除趋势，获取成交量的变化
                volume_detrend = volume - volume.rolling(window=20).mean()
                volume_detrend = volume_detrend.fillna(0)
                
                # 应用傅里叶变换
                volume_data = volume_detrend.values
                n = len(volume_data)
                freq = np.fft.fftfreq(n)
                
                # 计算FFT并取绝对值
                fft_values = np.fft.fft(volume_data)
                fft_abs = np.abs(fft_values)
                
                # 找出前3个主要周期
                # 忽略0频率(即直流分量)，它通常是最大的
                fft_abs[0] = 0
                
                # 由于对称性，我们只看前半部分频率
                half_n = n // 2
                main_freq_indices = np.argsort(fft_abs[:half_n])[-3:]
                
                # 计算周期长度（天数）
                main_periods = np.round(1 / np.abs(freq[main_freq_indices])).astype(int)
                
                # 排除非实际的周期长度
                valid_periods = main_periods[main_periods < n // 2]
                valid_periods = valid_periods[valid_periods > 1]
                
                # 记录识别出的周期
                if len(valid_periods) > 0:
                    df_copy['vol_main_cycles'] = [valid_periods.tolist()] * len(df_copy)
                    
                    # 计算当前在周期中的位置
                    for i, period in enumerate(valid_periods[:min(2, len(valid_periods))]):
                        if period > 0:
                            # 计算当前位置占周期的百分比
                            df_copy[f'vol_cycle_{period}d_pos'] = (np.arange(len(df_copy)) % period) / period * 100
                            
                            # 计算过去一个周期的成交量变化率
                            if period < len(df_copy):
                                df_copy[f'vol_cycle_{period}d_change'] = volume / volume.shift(period) - 1
                
                # 记录最主要的周期
                if len(valid_periods) > 0:
                    df_copy['vol_primary_cycle'] = valid_periods[0] if len(valid_periods) > 0 else np.nan
            except:
                # 如果FFT分析失败，使用传统方法
                pass
        
        # 2. 自相关函数分析
        try:
            from statsmodels.tsa.stattools import acf
            
            # 计算自相关函数
            volume_normalized = (volume - volume.mean()) / volume.std()
            volume_normalized = volume_normalized.fillna(0)
            
            # 计算最多60天的自相关
            max_lag = min(60, len(volume) // 3)
            acf_values = acf(volume_normalized.dropna(), nlags=max_lag)
            
            # 找出自相关峰值（可能的周期）
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(acf_values, height=0.1, distance=3)
            
            if len(peaks) > 0:
                # 排除lag=0（它始终是峰值）
                peaks = peaks[peaks > 0]
                
                # 记录ACF分析的周期
                if len(peaks) > 0:
                    df_copy['vol_acf_cycles'] = [peaks.tolist()] * len(df_copy)
                    
                    # 主要ACF周期
                    peak_values = acf_values[peaks]
                    main_peak_idx = peaks[np.argmax(peak_values)]
                    df_copy['vol_acf_main_cycle'] = main_peak_idx
                    
                    # 计算在周期中的位置
                    if main_peak_idx > 0:
                        df_copy['vol_acf_cycle_pos'] = (np.arange(len(df_copy)) % main_peak_idx) / main_peak_idx * 100
        except:
            # 如果自相关分析失败，忽略
            pass
        
        # 3. 简单周期分析 - 周度、月度特征
        # 计算每周每月的均值和标准差
        if 'date' in df_copy.columns or df_copy.index.dtype == 'datetime64[ns]':
            try:
                # 获取日期列或索引
                if 'date' in df_copy.columns:
                    dates = pd.to_datetime(df_copy['date'])
                else:
                    dates = df_copy.index
                
                # 添加星期几和月份信息
                df_copy['day_of_week'] = dates.dayofweek
                df_copy['month'] = dates.month
                
                # 计算不同星期几的成交量分布
                weekday_stats = df_copy.groupby('day_of_week')['vol'].agg(['mean', 'std'])
                
                # 计算每周几的相对成交量水平
                for day in range(5):  # 工作日0-4
                    if day in weekday_stats.index:
                        day_mean = weekday_stats.loc[day, 'mean']
                        overall_mean = volume.mean()
                        # 为DataFrame添加每个工作日的相对成交量均值
                        df_copy[f'vol_weekday_{day}_ratio'] = day_mean / overall_mean if overall_mean > 0 else 1.0
                
                # 计算每月的相对成交量水平
                monthly_stats = df_copy.groupby('month')['vol'].agg(['mean', 'std'])
                
                # 计算每月的相对成交量水平
                for month in range(1, 13):
                    if month in monthly_stats.index:
                        month_mean = monthly_stats.loc[month, 'mean']
                        overall_mean = volume.mean()
                        # 为DataFrame添加每个月的相对成交量均值
                        df_copy[f'vol_month_{month}_ratio'] = month_mean / overall_mean if overall_mean > 0 else 1.0
            except:
                # 如果日期分析失败，忽略
                pass
        
        # 4. 成交量波动率分析
        # 计算不同时间窗口的成交量波动率
        for window in [5, 10, 20, 60]:
            if len(df_copy) >= window:
                # 计算滚动波动率
                df_copy[f'vol_volatility_{window}d'] = volume.rolling(window=window).std() / volume.rolling(window=window).mean()
                
                # 计算波动率变化趋势
                if f'vol_volatility_{window}d' in df_copy.columns:
                    df_copy[f'vol_volatility_{window}d_trend'] = df_copy[f'vol_volatility_{window}d'].pct_change(5)
        
        # 5. 识别当前周期阶段
        if 'vol_acf_main_cycle' in df_copy.columns and df_copy['vol_acf_main_cycle'].iloc[-1] > 0:
            cycle_length = int(df_copy['vol_acf_main_cycle'].iloc[-1])
            
            if cycle_length > 0 and cycle_length < len(df_copy) // 3:
                # 获取最近一个完整周期
                recent_cycle_start = max(0, len(df_copy) - 2 * cycle_length)
                recent_data = volume.iloc[recent_cycle_start:].values
                
                # 将周期分为4个阶段：低-上升-高-下降
                # 为了简化，我们基于相对位置进行划分
                cycle_pos = (len(df_copy) - 1) % cycle_length
                cycle_phase = cycle_pos / cycle_length
                
                if cycle_phase < 0.25:
                    df_copy.loc[df_copy.index[-1], 'vol_cycle_phase'] = 'low_volume'
                elif cycle_phase < 0.5:
                    df_copy.loc[df_copy.index[-1], 'vol_cycle_phase'] = 'rising_volume'
                elif cycle_phase < 0.75:
                    df_copy.loc[df_copy.index[-1], 'vol_cycle_phase'] = 'high_volume'
                else:
                    df_copy.loc[df_copy.index[-1], 'vol_cycle_phase'] = 'falling_volume'
        
        return df_copy
    
    def _register_volume_patterns(self):
        """注册成交量指标的各种形态"""
        # 放量突破形态
        self.register_pattern(
            pattern_id="VOL_BREAKOUT",
            display_name="放量突破",
            detection_func=self._detect_vol_breakout,
            score_impact=20.0
        )
        
        # 缩量回调形态
        self.register_pattern(
            pattern_id="VOL_PULLBACK",
            display_name="缩量回调",
            detection_func=self._detect_vol_pullback,
            score_impact=15.0
        )
        
        # 巨量异动形态
        self.register_pattern(
            pattern_id="VOL_ABNORMAL",
            display_name="巨量异动",
            detection_func=self._detect_vol_anomaly,
            score_impact=-25.0
        )
        
        # 量价背离形态
        self.register_pattern(
            pattern_id="VOL_PRICE_DIVERGENCE",
            display_name="量价背离",
            detection_func=self._detect_vol_price_divergence,
            score_impact=-30.0
        )
        
        # 量能蓄势形态
        self.register_pattern(
            pattern_id="VOL_ACCUMULATION",
            display_name="量能蓄势",
            detection_func=self._detect_vol_accumulation,
            score_impact=25.0
        )
        
        # 量能衰竭形态
        self.register_pattern(
            pattern_id="VOL_EXHAUSTION",
            display_name="量能衰竭",
            detection_func=self._detect_vol_exhaustion,
            score_impact=-25.0
        )
        
        # 量价同步形态
        self.register_pattern(
            pattern_id="VOL_PRICE_SYNC",
            display_name="量价同步",
            detection_func=self._detect_vol_price_sync,
            score_impact=15.0
        )
        
        # 梯量形态
        self.register_pattern(
            pattern_id="VOL_GRADUAL",
            display_name="梯量",
            detection_func=self._detect_vol_gradual_change,
            score_impact=18.0
        )
        
        # 量能平台形态
        self.register_pattern(
            pattern_id="VOL_PLATFORM",
            display_name="量能平台",
            detection_func=self._detect_vol_platform,
            score_impact=12.0
        )

    def get_patterns(self, data: pd.DataFrame, **kwargs) -> List[Dict[str, Any]]:
        """
        获取成交量指标的所有形态信息
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[Dict[str, Any]]: 包含形态信息的字典列表
        """
        if not self.has_result():
            self.calculate(data)
            
        result = []
        
        # 检测所有已注册的形态
        for pattern_id, pattern_info in self._registered_patterns.items():
            detected = pattern_info['detection_func'](data)
            if detected:
                pattern_data = {
                    'pattern_id': pattern_id,
                    'display_name': pattern_info['display_name'],
                    'indicator_id': self.name,
                    'strength': SignalStrength.MEDIUM.value,  # 默认强度
                    'duration': 1,  # 默认持续时间
                    'details': {}
                }
                
                # 添加形态细节
                if 'breakout' in pattern_id.lower():
                    pattern_data['strength'] = SignalStrength.VERY_STRONG.value
                    pattern_data['details']['current_vol'] = float(data['volume'].iloc[-1])
                    pattern_data['details']['avg_vol'] = float(data[f'vol_avg_{self.period}'].iloc[-1])
                elif 'pullback' in pattern_id.lower():
                    pattern_data['strength'] = SignalStrength.STRONG.value
                    pattern_data['details']['current_vol'] = float(data['volume'].iloc[-1])
                    pattern_data['details']['prev_vol'] = float(data['volume'].iloc[-2])
                elif 'abnormal' in pattern_id.lower():
                    pattern_data['strength'] = SignalStrength.VERY_STRONG_NEGATIVE.value
                    pattern_data['details']['current_vol'] = float(data['volume'].iloc[-1])
                    pattern_data['details']['std_dev'] = float(data['vol_std_dev'].iloc[-1])
                elif 'divergence' in pattern_id.lower():
                    pattern_data['strength'] = SignalStrength.VERY_STRONG_NEGATIVE.value
                    pattern_data['details']['price_trend'] = float(data['close'].pct_change(5).iloc[-1])
                    pattern_data['details']['vol_trend'] = float(data['volume'].pct_change(5).iloc[-1])
                elif 'accumulation' in pattern_id.lower():
                    pattern_data['strength'] = SignalStrength.VERY_STRONG.value
                    pattern_data['details']['avg_vol'] = float(data[f'vol_avg_{self.period}'].iloc[-1])
                    pattern_data['details']['vol_ratio'] = float(data[f'vol_ratio'].iloc[-1])
                elif 'exhaustion' in pattern_id.lower():
                    pattern_data['strength'] = SignalStrength.VERY_STRONG_NEGATIVE.value
                    pattern_data['details']['vol_contraction'] = float(data['volume'].pct_change().iloc[-1])
                elif 'sync' in pattern_id.lower():
                    pattern_data['strength'] = SignalStrength.STRONG.value
                    pattern_data['details']['price_change'] = float(data['close'].pct_change().iloc[-1])
                    pattern_data['details']['vol_change'] = float(data['volume'].pct_change().iloc[-1])
                elif 'gradual' in pattern_id.lower():
                    pattern_data['strength'] = SignalStrength.STRONG.value
                    pattern_data['details']['vol_change_3d'] = float(data['volume'].pct_change(3).iloc[-1])
                elif 'platform' in pattern_id.lower():
                    pattern_data['strength'] = SignalStrength.MODERATE.value
                    pattern_data['details']['vol_std'] = float(data[f'vol_std_dev'].iloc[-1])
                
                result.append(pattern_data)
        
        return result

    def calculate_score(self, data: pd.DataFrame, **kwargs) -> float:
        """
        计算成交量指标评分（0-100分制）
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            float: 综合评分（0-100）
        """
        raw_score = self.calculate_raw_score(data, **kwargs)
        
        if raw_score.empty:
            return 50.0  # 默认中性评分
        
        last_score = raw_score.iloc[-1]
        
        # 应用市场环境调整
        market_env = kwargs.get('market_env', self._market_environment)
        adjusted_score = self.apply_market_environment_adjustment(market_env, last_score)
        
        # 计算置信度
        confidence = self.calculate_confidence(adjusted_score, self.get_patterns(data), {})
        
        # 返回最终评分
        return float(np.clip(adjusted_score * confidence, 0, 100))



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
