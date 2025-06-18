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
# import talib  # 移除talib依赖

from indicators.base_indicator import BaseIndicator, PatternResult
from utils.indicator_utils import crossover, crossunder
from utils.logger import get_logger
from indicators.pattern_registry import PatternRegistry, PatternType, PatternStrength

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
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """
        初始化成交量(VOL)指标
        
        Args:
            period: 计算周期，默认为14
            enable_cycles_analysis: 是否启用量能周期分析，默认启用
            enable_standardization: 是否启用成交量标准化，默认启用
        """
        super().__init__(name="VOL", description="成交量指标，市场活跃度、参与度直观体现")
        self.period = period
        self.enable_cycles_analysis = enable_cycles_analysis
        self.enable_standardization = enable_standardization
    
    def set_parameters(self, period: int = None, enable_cycles_analysis: bool = None, enable_standardization: bool = None):
        """
        设置指标参数
        """
        if period is not None:
            self.period = period
        if enable_cycles_analysis is not None:
            self.enable_cycles_analysis = enable_cycles_analysis
        if enable_standardization is not None:
            self.enable_standardization = enable_standardization

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算VOL指标

        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 其他参数

        Returns:
            包含VOL指标的DataFrame
        """
        return self._calculate(data)

    def calculate_confidence(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """
        计算VOL指标的置信度

        Args:
            score: 得分序列
            patterns: 检测到的形态DataFrame
            signals: 生成的信号字典

        Returns:
            float: 置信度分数 (0-1)
        """
        if score.empty:
            return 0.5

        # 基础置信度
        confidence = 0.5

        # 1. 基于评分的置信度
        last_score = score.iloc[-1]

        # 极端评分置信度较高
        if last_score > 80 or last_score < 20:
            confidence += 0.25
        # 中性评分置信度中等
        elif 40 <= last_score <= 60:
            confidence += 0.1
        else:
            confidence += 0.15

        # 2. 基于形态的置信度
        if not patterns.empty:
            # 检查VOL形态
            pattern_count = patterns.sum().sum()
            if pattern_count > 0:
                confidence += min(pattern_count * 0.05, 0.2)

        # 3. 基于信号的置信度
        if signals:
            # 检查信号强度
            signal_count = sum(1 for signal in signals.values() if hasattr(signal, 'any') and signal.any())
            if signal_count > 0:
                confidence += min(signal_count * 0.1, 0.15)

        # 4. 基于评分趋势的置信度
        if len(score) >= 3:
            recent_scores = score.iloc[-3:]
            trend = recent_scores.iloc[-1] - recent_scores.iloc[0]

            # 明确的趋势增加置信度
            if abs(trend) > 10:
                confidence += 0.05

        # 确保置信度在0-1范围内
        return max(0.0, min(1.0, confidence))
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算成交量指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
                
        Returns:
            包含VOL指标的DataFrame
        """
        return self.calculate(df)
        
    def _calculate(self, df: pd.DataFrame) -> pd.DataFrame:
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
        df_copy['vol_ratio_change'] = df_copy['vol_ratio'].pct_change(fill_method=None)
        
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
        
        # 限制评分在0-100之间
        return score.clip(0, 100)
    
    def _calculate_volume_level_score(self, indicator_data: pd.DataFrame) -> pd.Series:
        """
        计算成交量水平评分
        """
        score = pd.Series(0, index=indicator_data.index)
        
        # 基于成交量与均线的关系
        vol_gt_ma5 = indicator_data['vol'] > indicator_data['vol_ma5']
        vol_gt_ma10 = indicator_data['vol'] > indicator_data['vol_ma10']
        vol_gt_ma20 = indicator_data['vol'] > indicator_data['vol_ma20']
        
        score = score.mask(vol_gt_ma5, score + 5)
        score = score.mask(vol_gt_ma10, score + 8)
        score = score.mask(vol_gt_ma20, score + 12)
        
        # 缩量情况
        vol_lt_ma5 = indicator_data['vol'] < indicator_data['vol_ma5'] * 0.6
        score = score.mask(vol_lt_ma5, score - 8)
        
        return score.clip(-15, 15)
        
    def _calculate_price_volume_harmony(self, data: pd.DataFrame, indicator_data: pd.DataFrame) -> pd.Series:
        """
        计算量价配合评分
        """
        score = pd.Series(0, index=data.index)
        
        price_up = data['close'] > data['close'].shift(1)
        price_down = data['close'] < data['close'].shift(1)
        
        vol_up = indicator_data['vol'] > indicator_data['vol'].shift(1)
        vol_down = indicator_data['vol'] < indicator_data['vol'].shift(1)
        
        # 价涨量增
        score = score.mask(price_up & vol_up, score + 15)
        
        # 价跌量缩
        score = score.mask(price_down & vol_down, score + 10)
        
        # 价涨量缩 (背离)
        score = score.mask(price_up & vol_down, score - 12)
        
        # 价跌量增 (背离)
        score = score.mask(price_down & vol_up, score - 15)
        
        return score.clip(-20, 20)

    def _calculate_volume_trend_score(self, indicator_data: pd.DataFrame) -> pd.Series:
        """
        计算成交量趋势评分
        """
        score = pd.Series(0, index=indicator_data.index)
        
        # 均量线多头排列
        ma5_gt_ma10 = indicator_data['vol_ma5'] > indicator_data['vol_ma10']
        ma10_gt_ma20 = indicator_data['vol_ma10'] > indicator_data['vol_ma20']
        
        # 均量线空头排列
        ma5_lt_ma10 = indicator_data['vol_ma5'] < indicator_data['vol_ma10']
        ma10_lt_ma20 = indicator_data['vol_ma10'] < indicator_data['vol_ma20']
        
        bullish_arrangement = ma5_gt_ma10 & ma10_gt_ma20
        bearish_arrangement = ma5_lt_ma10 & ma10_lt_ma20
        
        score = score.mask(bullish_arrangement, score + 15)
        score = score.mask(bearish_arrangement, score - 15)
        
        # 均量线金叉/死叉
        ma5_cross_ma10_up = crossover(indicator_data['vol_ma5'], indicator_data['vol_ma10'])
        ma5_cross_ma10_down = crossunder(indicator_data['vol_ma5'], indicator_data['vol_ma10'])
        
        score = score.mask(ma5_cross_ma10_up, score + 10)
        score = score.mask(ma5_cross_ma10_down, score - 10)
        
        return score.clip(-20, 20)
        
    def _calculate_relative_volume_score(self, indicator_data: pd.DataFrame) -> pd.Series:
        """
        计算相对成交量评分
        """
        score = pd.Series(0, index=indicator_data.index)
        
        # 使用60日量比
        vol_ratio_60 = indicator_data['vol_ratio_60']
        
        # 量比 > 2.5，极度放量，可能反转
        score = score.mask(vol_ratio_60 > 2.5, score - 10)
        
        # 1.5 < 量比 <= 2.5，温和放量
        score = score.mask((vol_ratio_60 > 1.5) & (vol_ratio_60 <= 2.5), score + 15)
        
        # 0.5 < 量比 <= 1.5，正常波动
        score = score.mask((vol_ratio_60 > 0.5) & (vol_ratio_60 <= 1.5), score + 5)
        
        # 量比 <= 0.5，极度缩量
        score = score.mask(vol_ratio_60 <= 0.5, score - 5)
        
        return score.clip(-15, 15)
        
    def _detect_abnormal_volume(self, data: pd.DataFrame, indicator_data: pd.DataFrame) -> pd.Series:
        """
        检测异常放量或缩量并评分
        - 使用Z-score来识别异常值
        - 结合价格波动进行评估
        """
        score = pd.Series(0.0, index=data.index)
        
        # 计算成交量的Z-score
        rolling_window = 60
        if len(indicator_data) < rolling_window:
            return score
            
        vol_mean = indicator_data['volume'].rolling(window=rolling_window).mean()
        vol_std = indicator_data['volume'].rolling(window=rolling_window).std()
        
        # 避免除以零
        vol_std.replace(0, np.nan, inplace=True)
        
        z_score = (indicator_data['volume'] - vol_mean) / vol_std
        
        # 价格波动
        price_change_pct = data['close'].pct_change().abs() * 100
        
        # 异常放量
        abnormal_high_vol = z_score > 3.0
        
        # 异常缩量
        abnormal_low_vol = z_score < -1.5
        
        # 1. 异常放量 + 价格大涨（>5%）: 强看涨信号，但有过热风险
        score = score.mask(abnormal_high_vol & (price_change_pct > 5) & (data['close'] > data['close'].shift(1)), score + 15)
        
        # 2. 异常放量 + 价格大跌（>5%）: 强看跌信号，恐慌盘
        score = score.mask(abnormal_high_vol & (price_change_pct > 5) & (data['close'] < data['close'].shift(1)), score - 20)
        
        # 3. 异常放量 + 价格窄幅波动（<1%）: 滞涨，多空分歧大
        score = score.mask(abnormal_high_vol & (price_change_pct < 1), score - 5)
        
        # 4. 异常缩量 + 价格窄幅波动: 市场冷清，方向不明
        score = score.mask(abnormal_low_vol & (price_change_pct < 1), score - 8)
        
        # 5. 异常缩量 + 价格上涨: 缩量上涨，上涨动力不足
        score = score.mask(abnormal_low_vol & (data['close'] > data['close'].shift(1)), score - 10)
        
        # 6. 异常缩量 + 价格下跌: 缩量下跌，下跌动能衰竭
        score = score.mask(abnormal_low_vol & (data['close'] < data['close'].shift(1)), score + 5)
        
        return score.clip(-20, 20)
        
    def identify_patterns(self, data: pd.DataFrame) -> List[str]:
        """
        识别成交量(VOL)的常见形态
        
        Args:
            data: 包含OHLCV和VOL指标的DataFrame
                
        Returns:
            List[str]: 识别出的形态列表
        """
        if data.empty or len(data) < 20:
            return []
            
        patterns = []
        
        # 计算指标
        df = self.calculate(data)
        
        # 检查最新数据点
        latest = df.iloc[-1]
        
        # 1. 放量上涨
        if latest['vol_ratio'] > 1.5 and latest['close'] > df['close'].iloc[-2]:
            patterns.append("放量上涨")
        
        # 2. 放量下跌
        if latest['vol_ratio'] > 1.5 and latest['close'] < df['close'].iloc[-2]:
            patterns.append("放量下跌")
        
        # 3. 缩量上涨
        if latest['vol_ratio'] < 0.7 and latest['close'] > df['close'].iloc[-2]:
            patterns.append("缩量上涨")
            
        # 4. 缩量下跌
        if latest['vol_ratio'] < 0.7 and latest['close'] < df['close'].iloc[-2]:
            patterns.append("缩量下跌")
        
        # 5. 量价背离 (最近20天)
        recent_data = df.tail(20)
        price_trend, _, _, _, _ = stats.linregress(range(len(recent_data)), recent_data['close'])
        volume_trend, _, _, _, _ = stats.linregress(range(len(recent_data)), recent_data['volume'])
        
        if price_trend > 0 and volume_trend < 0:
            patterns.append("价涨量缩背离")
        
        if price_trend < 0 and volume_trend > 0:
            patterns.append("价跌量增背离")
            
        # 6. 成交量均线多头排列
        if latest['vol_ma5'] > latest['vol_ma10'] > latest['vol_ma20']:
            patterns.append("均量线多头排列")
        
        # 7. 成交量均线空头排列
        if latest['vol_ma5'] < latest['vol_ma10'] < latest['vol_ma20']:
            patterns.append("均量线空头排列")
        
        # 8. 天量（最近半年内最大成交量）
        if len(df) >= 120:
            if latest['volume'] == df['volume'].tail(120).max():
                patterns.append("天量")
        
        # 9. 地量（最近半年内最小成交量）
        if len(df) >= 120:
            if latest['volume'] == df['volume'].tail(120).min():
                patterns.append("地量")
                
        # 10. 成交量突破 (优化)
        if self._detect_vol_breakout(df):
            patterns.append("成交量突破")
            
        # 11. 成交量回踩 (优化)
        if self._detect_vol_pullback(df):
            patterns.append("成交量回踩")
            
        # 12. 异常放量 (优化)
        if self._detect_vol_anomaly(df):
            patterns.append("异常放量")
            
        # 13. 成交量平台 (新增)
        if self._detect_vol_platform(df):
            patterns.append("成交量平台")
            
        return list(set(patterns))  # 去重

    def _calculate_standardized_relative_volume(self, data: pd.DataFrame, indicator_data: pd.DataFrame) -> pd.DataFrame:
        """
        计算标准化相对成交量 (SRV)

        Args:
            data (pd.DataFrame): 原始OHLCV数据
            indicator_data (pd.DataFrame): 包含成交量指标的DataFrame

        Returns:
            pd.DataFrame: 添加了SRV列的DataFrame
        """
        df = indicator_data.copy()

        # 确保有足够的历史数据
        rolling_window = 60
        if len(df) < rolling_window:
            df['srv'] = np.nan
            return df

        # 计算对数成交量
        df['log_vol'] = np.log1p(df['volume'])

        # 计算滚动均值和标准差
        rolling_mean = df['log_vol'].rolling(window=rolling_window).mean()
        rolling_std = df['log_vol'].rolling(window=rolling_window).std()

        # 计算SRV
        df['srv'] = (df['log_vol'] - rolling_mean) / rolling_std

        # 结合日内波动率进行调整
        intraday_volatility = (data['high'] - data['low']) / data['close']
        df['srv_adjusted'] = df['srv'] * (1 + intraday_volatility)

        # 增加短期和长期SRV的比值
        rolling_mean_short = df['log_vol'].rolling(window=20).mean()
        rolling_std_short = df['log_vol'].rolling(window=20).std()
        df['srv_short'] = (df['log_vol'] - rolling_mean_short) / rolling_std_short
        df['srv_ratio'] = df['srv_short'] / df['srv']

        # 清理中间列
        df.drop(['log_vol', 'srv_short'], axis=1, inplace=True, errors='ignore')

        return df

    def _analyze_volume_cycles(self, indicator_data: pd.DataFrame, min_periods: int = 60) -> pd.DataFrame:
        """
        使用傅里叶变换分析成交量周期性

        Args:
            indicator_data (pd.DataFrame): 包含成交量指标的DataFrame
            min_periods (int): 进行周期性分析所需的最少数据点

        Returns:
            pd.DataFrame: 添加了周期性分析结果的DataFrame
        """
        df = indicator_data.copy()

        if len(df) < min_periods:
            df['dominant_cycle'] = np.nan
            df['cycle_strength'] = np.nan
            return df

        # 提取成交量数据
        volume_series = df['volume'].dropna()
        if len(volume_series) < min_periods:
            df['dominant_cycle'] = np.nan
            df['cycle_strength'] = np.nan
            return df
            
        # 计算傅里叶变换
        fft_result = np.fft.fft(volume_series)
        fft_freq = np.fft.fftfreq(len(volume_series))
        
        # 找到主导周期
        # 忽略直流分量
        peak_idx = np.argmax(np.abs(fft_result[1:])) + 1
        dominant_freq = fft_freq[peak_idx]
        
        if dominant_freq != 0:
            dominant_cycle = 1 / dominant_freq
            cycle_strength = np.abs(fft_result[peak_idx]) / len(volume_series)
        else:
            dominant_cycle = np.nan
            cycle_strength = 0

        df['dominant_cycle'] = dominant_cycle
        df['cycle_strength'] = cycle_strength
        
        # 优化: 检测周期性共振
        # 将主导周期与其他已知周期（如5日，10日）进行比较
        known_cycles = [5, 10, 20]
        resonances = []
        for cycle in known_cycles:
            if not np.isnan(dominant_cycle) and abs(dominant_cycle - cycle) < 1.0:
                resonances.append(cycle)
        
        df['cycle_resonance'] = ','.join(map(str, resonances)) if resonances else None

        # 增加一个辅助函数来处理日内数据
        def calc_intraday_std(vol_list):
            if not isinstance(vol_list, (list, np.ndarray, pd.Series)) or len(vol_list) < 2:
                return 0
            return np.std(vol_list)

        # 如果有日内数据，可以进一步分析
        if 'intraday_volume' in df.columns:
            df['intraday_vol_std'] = df['intraday_volume'].apply(calc_intraday_std)
            # 将日内成交量波动与日间成交量波动进行比较
            df['intraday_vs_interday_vol_ratio'] = df['intraday_vol_std'] / df['vol_std']

        return df
        
    def _register_volume_patterns(self):
        """
        注册成交量形态
        """
        registry = PatternRegistry()
        
        # 注册放量上涨
        registry.register(
            pattern_id="VOL_BREAKOUT_UP",
            display_name="放量上涨",
            description="成交量显著放大，同时价格上涨，通常是趋势启动或加速的信号。",
            indicator_id="VOL",
            pattern_type=PatternType.CONTINUATION,
            score_impact=15.0,
            strength=PatternStrength.STRONG
        )
        
        # 注册放量下跌
        registry.register(
            pattern_id="VOL_BREAKOUT_DOWN",
            display_name="放量下跌",
            description="成交量显著放大，同时价格下跌，通常是恐慌性抛售或趋势反转的信号。",
            indicator_id="VOL",
            pattern_type=PatternType.REVERSAL,
            score_impact=-15.0,
            strength=PatternStrength.STRONG
        )
        
        # 注册缩量上涨
        registry.register(
            pattern_id="VOL_WEAK_UP",
            display_name="缩量上涨",
            description="价格上涨但成交量萎缩，可能表示上涨动力不足。",
            indicator_id="VOL",
            pattern_type=PatternType.DIVERGENCE,
            score_impact=-10.0,
            strength=PatternStrength.WEAK
        )
        
        # 注册缩量下跌
        registry.register(
            pattern_id="VOL_WEAK_DOWN",
            display_name="缩量下跌",
            description="价格下跌且成交量萎缩，可能表示下跌动能衰竭。",
            indicator_id="VOL",
            pattern_type=PatternType.REVERSAL,
            score_impact=10.0,
            strength=PatternStrength.MEDIUM
        )
        
        # 注册量价背离
        registry.register(
            pattern_id="VOL_PRICE_DIVERGENCE",
            display_name="量价背离",
            description="价格与成交量趋势相反，例如价格新高而成交量萎缩。",
            indicator_id="VOL",
            pattern_type=PatternType.DIVERGENCE,
            score_impact=-12.0,
            strength=PatternStrength.MEDIUM
        )
        
        # 注册天量
        registry.register(
            pattern_id="VOL_PEAK",
            display_name="天量",
            description="成交量达到近期（如半年内）的峰值，可能预示趋势即将反转。",
            indicator_id="VOL",
            pattern_type=PatternType.EXHAUSTION,
            score_impact=-8.0,
            strength=PatternStrength.STRONG
        )
        
        # 注册地量
        registry.register(
            pattern_id="VOL_TROUGH",
            display_name="地量",
            description="成交量达到近期（如半年内）的谷底，可能表示市场极度冷清或惜售。",
            indicator_id="VOL",
            pattern_type=PatternType.REVERSAL,
            score_impact=8.0,
            strength=PatternStrength.MEDIUM
        )
        
        # 注册成交量突破
        registry.register(
            pattern_id="VOL_BREAKOUT",
            display_name="成交量突破",
            description="成交量突破了前期的整理平台，通常伴随着价格的突破。",
            indicator_id="VOL",
            pattern_type=PatternType.BREAKOUT,
            score_impact=18.0,
            strength=PatternStrength.STRONG
        )
        
        # 注册成交量回踩
        registry.register(
            pattern_id="VOL_PULLBACK",
            display_name="成交量回踩",
            description="价格回调至前期支撑位，同时成交量显著萎缩，可能是买入机会。",
            indicator_id="VOL",
            pattern_type=PatternType.CONTINUATION,
            score_impact=12.0,
            strength=PatternStrength.MEDIUM
        )
        
        # 注册成交量平台
        registry.register(
            pattern_id="VOL_PLATFORM",
            display_name="成交量平台",
            description="成交量在一段时间内维持在相对稳定的水平，可能在酝酿新的趋势。",
            indicator_id="VOL",
            pattern_type=PatternType.CONSOLIDATION,
            score_impact=5.0,
            strength=PatternStrength.WEAK
        )
        
    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取VOL相关形态

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含形态信息的DataFrame
        """
        # 确保已计算指标
        if self._result is None:
            self.calculate(data)

        if self._result is None or 'vol' not in self._result.columns:
            return pd.DataFrame(index=data.index)

        # 获取VOL数据
        vol = self._result['vol']
        vol_ma5 = self._result['vol_ma5']
        vol_ma10 = self._result['vol_ma10']
        vol_ma20 = self._result['vol_ma20']
        vol_ratio = self._result['vol_ratio']

        # 创建形态DataFrame
        patterns_df = pd.DataFrame(index=data.index)

        # 1. 成交量水平形态
        patterns_df['VOL_HIGH'] = vol > vol_ma20 * 1.5
        patterns_df['VOL_VERY_HIGH'] = vol > vol_ma20 * 2.0
        patterns_df['VOL_LOW'] = vol < vol_ma20 * 0.5
        patterns_df['VOL_VERY_LOW'] = vol < vol_ma20 * 0.3

        # 2. 成交量趋势形态
        patterns_df['VOL_RISING'] = vol > vol.shift(1)
        patterns_df['VOL_FALLING'] = vol < vol.shift(1)
        patterns_df['VOL_MA_BULLISH'] = (vol_ma5 > vol_ma10) & (vol_ma10 > vol_ma20)
        patterns_df['VOL_MA_BEARISH'] = (vol_ma5 < vol_ma10) & (vol_ma10 < vol_ma20)

        # 3. 成交量突破形态
        patterns_df['VOL_BREAKOUT_UP'] = (vol_ratio > 1.5) & (data['close'] > data['close'].shift(1))
        patterns_df['VOL_BREAKOUT_DOWN'] = (vol_ratio > 1.5) & (data['close'] < data['close'].shift(1))

        # 4. 成交量背离形态
        patterns_df['VOL_WEAK_UP'] = (vol_ratio < 0.7) & (data['close'] > data['close'].shift(1))
        patterns_df['VOL_WEAK_DOWN'] = (vol_ratio < 0.7) & (data['close'] < data['close'].shift(1))

        # 5. 成交量极值形态
        if len(vol) >= 120:
            vol_120_max = vol.rolling(window=120).max()
            vol_120_min = vol.rolling(window=120).min()
            patterns_df['VOL_PEAK'] = vol >= vol_120_max
            patterns_df['VOL_TROUGH'] = vol <= vol_120_min
        else:
            patterns_df['VOL_PEAK'] = False
            patterns_df['VOL_TROUGH'] = False

        # 6. 成交量金叉死叉
        patterns_df['VOL_GOLDEN_CROSS'] = (vol_ma5 > vol_ma10) & (vol_ma5.shift(1) <= vol_ma10.shift(1))
        patterns_df['VOL_DEATH_CROSS'] = (vol_ma5 < vol_ma10) & (vol_ma5.shift(1) >= vol_ma10.shift(1))

        return patterns_df
        
    def calculate_score(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        计算最终评分

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            Dict[str, Any]: 包含评分和置信度的字典
        """
        try:
            # 1. 计算原始评分序列
            raw_scores = self.calculate_raw_score(data, **kwargs)

            # 如果数据不足，返回中性评分
            if len(raw_scores) < 3:
                return {'score': 50.0, 'confidence': 0.5}

            # 取最近的评分作为最终评分，但考虑近期趋势
            recent_scores = raw_scores.iloc[-3:]
            trend = recent_scores.iloc[-1] - recent_scores.iloc[0]

            # 最终评分 = 最新评分 + 趋势调整
            final_score = recent_scores.iloc[-1] + trend / 2

            # 确保评分在0-100范围内
            final_score = max(0, min(100, final_score))

            # 2. 获取形态和信号
            patterns = self.get_patterns(data, **kwargs)

            # 3. 计算置信度
            confidence = self.calculate_confidence(raw_scores, patterns, {})

            return {
                'score': final_score,
                'confidence': confidence
            }
        except Exception as e:
            logger.error(f"为指标 {self.name} 计算评分时出错: {e}")
            return {'score': 50.0, 'confidence': 0.0}

    def register_patterns(self):
        """
        注册VOL指标的形态到全局形态注册表
        """
        # 注册放量上涨形态
        self.register_pattern_to_registry(
            pattern_id="VOL_BREAKOUT_UP",
            display_name="放量上涨",
            description="成交量显著放大，同时价格上涨，通常是趋势启动或加速的信号",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=15.0,
            polarity="POSITIVE"
        )

        # 注册放量下跌形态
        self.register_pattern_to_registry(
            pattern_id="VOL_BREAKOUT_DOWN",
            display_name="放量下跌",
            description="成交量显著放大，同时价格下跌，通常是恐慌性抛售或趋势反转的信号",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-15.0,
            polarity="NEGATIVE"
        )

        # 注册缩量上涨形态
        self.register_pattern_to_registry(
            pattern_id="VOL_WEAK_UP",
            display_name="缩量上涨",
            description="价格上涨但成交量萎缩，需要结合位置判断意义",
            pattern_type="NEUTRAL",
            default_strength="WEAK",
            score_impact=0.0,
            polarity="NEUTRAL"
        )

        # 注册缩量下跌形态
        self.register_pattern_to_registry(
            pattern_id="VOL_WEAK_DOWN",
            display_name="缩量下跌",
            description="价格下跌且成交量萎缩，可能表示下跌动能衰竭",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=10.0,
            polarity="POSITIVE"
        )

        # 注册天量形态
        self.register_pattern_to_registry(
            pattern_id="VOL_PEAK",
            display_name="天量",
            description="成交量达到近期峰值，需要结合价格行为判断意义",
            pattern_type="NEUTRAL",
            default_strength="STRONG",
            score_impact=0.0,
            polarity="NEUTRAL"
        )

        # 注册地量形态
        self.register_pattern_to_registry(
            pattern_id="VOL_TROUGH",
            display_name="地量",
            description="成交量达到近期谷底，可能表示市场极度冷清或惜售",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=8.0,
            polarity="POSITIVE"
        )

        # 注册成交量金叉形态
        self.register_pattern_to_registry(
            pattern_id="VOL_GOLDEN_CROSS",
            display_name="成交量金叉",
            description="短期成交量均线上穿长期均线，表示成交量趋势向好",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=12.0,
            polarity="POSITIVE"
        )

        # 注册成交量死叉形态
        self.register_pattern_to_registry(
            pattern_id="VOL_DEATH_CROSS",
            display_name="成交量死叉",
            description="短期成交量均线下穿长期均线，表示成交量趋势转弱",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-12.0,
            polarity="NEGATIVE"
        )

        # 注册均量线多头排列形态
        self.register_pattern_to_registry(
            pattern_id="VOL_MA_BULLISH",
            display_name="均量线多头排列",
            description="成交量均线呈多头排列，表示成交量趋势强劲",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=15.0,
            polarity="POSITIVE"
        )

        # 注册均量线空头排列形态
        self.register_pattern_to_registry(
            pattern_id="VOL_MA_BEARISH",
            display_name="均量线空头排列",
            description="成交量均线呈空头排列，表示成交量趋势疲弱",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-15.0,
            polarity="NEGATIVE"
        )

    def generate_trading_signals(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """
        生成交易信号

        Args:
            data (pd.DataFrame): 输入数据
            **kwargs: 额外参数

        Returns:
            Dict[str, pd.Series]: 包含交易信号的字典
        """
        # 确保已计算指标
        if not self.has_result():
            self._calculate(data, **kwargs)

        # 初始化信号
        signals = {}
        signals['buy_signal'] = pd.Series(False, index=data.index)
        signals['sell_signal'] = pd.Series(False, index=data.index)
        signals['signal_strength'] = pd.Series(0, index=data.index)

        # 信号生成逻辑
        # 1. 温和放量上涨
        buy_cond1 = (self._result['vol_ratio'] > 1.5) & (self._result['vol_ratio'] <= 2.5) & \
                    (data['close'] > data['close'].shift(1))
        signals['buy_signal'] = signals['buy_signal'] | buy_cond1
        signals['signal_strength'].mask(buy_cond1, 70, inplace=True)

        # 2. 缩量下跌企稳
        sell_cond1 = (self._result['vol_ratio'] < 0.6) & \
                     (data['close'] < data['close'].shift(1)) & \
                     (data['close'].shift(1) < data['close'].shift(2)) # 连续下跌
        signals['sell_signal'] = signals['sell_signal'] | sell_cond1
        signals['signal_strength'].mask(sell_cond1, 60, inplace=True)

        # 3. 巨量下跌（恐慌盘）
        sell_cond2 = (self._result['vol_ratio'] > 3.0) & \
                     (data['close'] < data['close'].shift(1))
        signals['sell_signal'] = signals['sell_signal'] | sell_cond2
        signals['signal_strength'].mask(sell_cond2, 85, inplace=True)
        
        return signals

    def _detect_vol_breakout(self, data: pd.DataFrame) -> bool:
        """
        检测成交量突破
        - 条件：当前成交量 > 过去N天成交量均值 * M倍
        """
        if len(data) < 60:
            return False
            
        latest_vol = data['volume'].iloc[-1]
        mean_vol_60 = data['volume'].tail(60).mean()
        
        # 突破条件：当前成交量是60日均量的2倍以上
        return latest_vol > mean_vol_60 * 2.0

    def _detect_vol_pullback(self, data: pd.DataFrame) -> bool:
        """
        检测成交量回踩
        - 条件：价格回调至支撑位，且成交量显著萎缩
        """
        if len(data) < 30:
            return False
        
        # 价格回调
        is_pullback = (data['close'].iloc[-1] < data['close'].iloc[-2]) and \
                      (data['close'].iloc[-2] > data['close'].iloc[-3]) # 前一天上涨
                      
        # 成交量萎缩
        is_vol_shrink = data['volume'].iloc[-1] < data['volume'].tail(10).mean() * 0.5
        
        return is_pullback and is_vol_shrink
        
    def _detect_vol_anomaly(self, data: pd.DataFrame) -> bool:
        """
        检测异常放量
        - 使用Z-score识别统计上的异常
        """
        if len(data) < 60:
            return False
        
        rolling_mean = data['volume'].rolling(window=60).mean()
        rolling_std = data['volume'].rolling(window=60).std()
        
        # 避免除以零
        if rolling_std.iloc[-1] == 0:
            return False
            
        z_score = (data['volume'].iloc[-1] - rolling_mean.iloc[-1]) / rolling_std.iloc[-1]
        
        # Z-score > 3.0 表示异常放量
        return z_score > 3.0
        
    def _detect_vol_price_divergence(self, data: pd.DataFrame) -> bool:
        """
        检测量价背离
        - 价格新高，成交量未创新高
        - 价格新低，成交量未创新低
        """
        if len(data) < 30:
            return False
        
        recent_data = data.tail(30)
        
        # 顶背离
        if recent_data['close'].iloc[-1] == recent_data['close'].max() and \
           recent_data['volume'].iloc[-1] < recent_data['volume'].mean():
            return True
            
        # 底背离
        if recent_data['close'].iloc[-1] == recent_data['close'].min() and \
           recent_data['volume'].iloc[-1] < recent_data['volume'].mean():
            return True
            
        return False
        
    def _detect_vol_accumulation(self, data: pd.DataFrame) -> bool:
        """
        检测成交量堆积（吸筹）
        - 一段时间内，成交量温和放大，价格小幅上涨
        """
        if len(data) < 60:
            return False
            
        recent_data = data.tail(60)
        
        # 成交量温和放大
        is_vol_increasing = recent_data['volume'].iloc[-1] > recent_data['volume'].tail(30).mean()
        
        # 价格小幅上涨或横盘
        price_trend, _, _, _, _ = stats.linregress(range(len(recent_data)), recent_data['close'])
        is_price_stable = abs(price_trend) < 0.05
        
        return is_vol_increasing and is_price_stable
        
    def _detect_vol_exhaustion(self, data: pd.DataFrame) -> bool:
        """
        检测成交量耗尽
        - 长期上涨后，出现天量但价格滞涨
        """
        if len(data) < 120:
            return False
            
        # 长期上涨
        long_term_data = data.tail(120)
        price_trend, _, _, _, _ = stats.linregress(range(len(long_term_data)), long_term_data['close'])
        is_long_uptrend = price_trend > 0.1
        
        # 天量
        is_peak_vol = long_term_data['volume'].iloc[-1] == long_term_data['volume'].max()
        
        # 价格滞涨
        is_price_stagnant = abs(long_term_data['close'].pct_change().iloc[-1]) < 0.01
        
        return is_long_uptrend and is_peak_vol and is_price_stagnant
        
    def _detect_vol_price_sync(self, data: pd.DataFrame) -> bool:
        """
        检测量价齐升/齐跌
        """
        if len(data) < 2:
            return False
        
        # 量价齐升
        vol_price_up = (data['volume'].iloc[-1] > data['volume'].iloc[-2]) and \
                       (data['close'].iloc[-1] > data['close'].iloc[-2])
                       
        # 量价齐跌
        vol_price_down = (data['volume'].iloc[-1] < data['volume'].iloc[-2]) and \
                         (data['close'].iloc[-1] < data['close'].iloc[-2])
                         
        return vol_price_up or vol_price_down
        
    def _detect_vol_gradual_change(self, data: pd.DataFrame) -> bool:
        """
        检测成交量温和放大/缩小
        """
        if len(data) < 20:
            return False
            
        recent_vol = data['volume'].tail(20)
        vol_trend, _, _, _, _ = stats.linregress(range(len(recent_vol)), recent_vol)
        
        # 温和放大
        is_gradual_increase = vol_trend > 0 and abs(vol_trend) < recent_vol.mean() * 0.05
        
        # 温和缩小
        is_gradual_decrease = vol_trend < 0 and abs(vol_trend) < recent_vol.mean() * 0.05
        
        return is_gradual_increase or is_gradual_decrease
        
    def _detect_vol_platform(self, data: pd.DataFrame) -> bool:
        """
        检测成交量平台
        - 一段时间内成交量维持在相对稳定的水平
        """
        if len(data) < 30:
            return False
            
        recent_vol = data['volume'].tail(30)
        
        # 波动率小
        is_stable = recent_vol.std() / recent_vol.mean() < 0.2
        
        return is_stable

    def get_pattern_info(self, pattern_id: str) -> dict:
        """
        获取指定形态的详细信息

        Args:
            pattern_id: 形态ID

        Returns:
            dict: 形态详细信息
        """
        pattern_info_map = {
            "VOL_BREAKOUT_UP": {
                "id": "VOL_BREAKOUT_UP",
                "name": "放量上涨",
                "description": "成交量显著放大，同时价格上涨，通常是趋势启动或加速的信号",
                "type": "BULLISH",
                "strength": "STRONG",
                "score_impact": 15.0
            },
            "VOL_BREAKOUT_DOWN": {
                "id": "VOL_BREAKOUT_DOWN",
                "name": "放量下跌",
                "description": "成交量显著放大，同时价格下跌，通常是恐慌性抛售或趋势反转的信号",
                "type": "BEARISH",
                "strength": "STRONG",
                "score_impact": -15.0
            },
            "VOL_WEAK_UP": {
                "id": "VOL_WEAK_UP",
                "name": "缩量上涨",
                "description": "价格上涨但成交量萎缩，可能表示上涨动力不足",
                "type": "BEARISH",
                "strength": "WEAK",
                "score_impact": -10.0
            },
            "VOL_WEAK_DOWN": {
                "id": "VOL_WEAK_DOWN",
                "name": "缩量下跌",
                "description": "价格下跌且成交量萎缩，可能表示下跌动能衰竭",
                "type": "BULLISH",
                "strength": "MEDIUM",
                "score_impact": 10.0
            },
            "VOL_PEAK": {
                "id": "VOL_PEAK",
                "name": "天量",
                "description": "成交量达到近期峰值，可能预示趋势即将反转",
                "type": "BEARISH",
                "strength": "STRONG",
                "score_impact": -8.0
            },
            "VOL_TROUGH": {
                "id": "VOL_TROUGH",
                "name": "地量",
                "description": "成交量达到近期谷底，可能表示市场极度冷清或惜售",
                "type": "BULLISH",
                "strength": "MEDIUM",
                "score_impact": 8.0
            },
            "VOL_GOLDEN_CROSS": {
                "id": "VOL_GOLDEN_CROSS",
                "name": "成交量金叉",
                "description": "短期成交量均线上穿长期均线，表示成交量趋势向好",
                "type": "BULLISH",
                "strength": "MEDIUM",
                "score_impact": 12.0
            },
            "VOL_DEATH_CROSS": {
                "id": "VOL_DEATH_CROSS",
                "name": "成交量死叉",
                "description": "短期成交量均线下穿长期均线，表示成交量趋势转弱",
                "type": "BEARISH",
                "strength": "MEDIUM",
                "score_impact": -12.0
            },
            "VOL_MA_BULLISH": {
                "id": "VOL_MA_BULLISH",
                "name": "均量线多头排列",
                "description": "成交量均线呈多头排列，表示成交量趋势强劲",
                "type": "BULLISH",
                "strength": "STRONG",
                "score_impact": 15.0
            },
            "VOL_MA_BEARISH": {
                "id": "VOL_MA_BEARISH",
                "name": "均量线空头排列",
                "description": "成交量均线呈空头排列，表示成交量趋势疲弱",
                "type": "BEARISH",
                "strength": "STRONG",
                "score_impact": -15.0
            },
            "VOL_HIGH": {
                "id": "VOL_HIGH",
                "name": "成交量偏高",
                "description": "成交量高于平均水平，市场活跃度较高",
                "type": "NEUTRAL",
                "strength": "MEDIUM",
                "score_impact": 5.0
            },
            "VOL_VERY_HIGH": {
                "id": "VOL_VERY_HIGH",
                "name": "成交量极高",
                "description": "成交量极高，可能存在异常交易或重大消息",
                "type": "NEUTRAL",
                "strength": "STRONG",
                "score_impact": 0.0
            },
            "VOL_LOW": {
                "id": "VOL_LOW",
                "name": "成交量偏低",
                "description": "成交量低于平均水平，市场活跃度较低",
                "type": "NEUTRAL",
                "strength": "MEDIUM",
                "score_impact": -5.0
            },
            "VOL_VERY_LOW": {
                "id": "VOL_VERY_LOW",
                "name": "成交量极低",
                "description": "成交量极低，市场极度冷清",
                "type": "NEUTRAL",
                "strength": "STRONG",
                "score_impact": -10.0
            }
        }

        return pattern_info_map.get(pattern_id, {
            "id": pattern_id,
            "name": "成交量能量分析",
            "description": f"基于成交量能量变化的技术分析: {pattern_id}",
            "type": "NEUTRAL",
            "strength": "WEAK",
            "score_impact": 0.0
        })