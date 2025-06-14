#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
量比指标(VOLUME_RATIO)
量比是指当前成交量与前N个周期平均成交量的比值，用于衡量市场交易活跃度的变化。
"""

import numpy as np
import pandas as pd
# import talib  # 移除talib依赖
from typing import Optional, Union, List, Dict, Any
import logging

from indicators.base_indicator import BaseIndicator

logger = logging.getLogger(__name__)

class VR(BaseIndicator):
    """
    一个骨架实现的VR指标，用于解决导入错误。
    FIXME: 需要补充完整的计算逻辑。
    """
    def __init__(self, period: int = 26):
        """
        初始化指标
        """
        super().__init__()
        self.period = period
        self.name = f"VR({self.period})"
        self.description = f"成交量比率 (周期: {self.period}) - 骨架实现"

    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算指标。
        这是一个临时的骨架实现。
        """
        # 为了让指标能工作，我们至少返回一个占位符列
        result_df = pd.DataFrame(index=data.index)
        result_df['vr'] = 100.0  # 返回一个全为100的列作为占位符
        return result_df

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
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """
        初始化量比指标
        
        Args:
            reference_period: 参考周期，默认为5
            ma_period: 量比均线周期，默认为3
        """
        super().__init__(name=f"VOLUME_RATIO({reference_period},{ma_period})",
                         description=f"量比指标，参考周期{reference_period}，均线周期{ma_period}")
        self.reference_period = reference_period
        self.ma_period = ma_period
        
    def set_parameters(self, reference_period: int = None, ma_period: int = None):
        """
        设置指标参数
        """
        if reference_period is not None:
            self.reference_period = reference_period
        if ma_period is not None:
            self.ma_period = ma_period

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算Volume Ratio指标

        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 其他参数

        Returns:
            包含Volume Ratio指标的DataFrame
        """
        return self._calculate(data)

    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取Volume Ratio相关形态

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含形态信息的DataFrame
        """
        # 确保已计算指标
        if self._result is None:
            self.calculate(data)

        if self._result is None or 'volume_ratio' not in self._result.columns:
            return pd.DataFrame(index=data.index)

        # 获取Volume Ratio数据
        volume_ratio = self._result['volume_ratio']
        volume_ratio_ma = self._result['volume_ratio_ma']

        # 创建形态DataFrame
        patterns_df = pd.DataFrame(index=data.index)

        # 1. 量比水平形态
        patterns_df['VR_EXTREMELY_HIGH'] = volume_ratio > 3.0
        patterns_df['VR_VERY_HIGH'] = (volume_ratio > 2.0) & (volume_ratio <= 3.0)
        patterns_df['VR_HIGH'] = (volume_ratio > 1.5) & (volume_ratio <= 2.0)
        patterns_df['VR_NORMAL'] = (volume_ratio >= 0.8) & (volume_ratio <= 1.2)
        patterns_df['VR_LOW'] = (volume_ratio >= 0.5) & (volume_ratio < 0.8)
        patterns_df['VR_VERY_LOW'] = (volume_ratio >= 0.3) & (volume_ratio < 0.5)
        patterns_df['VR_EXTREMELY_LOW'] = volume_ratio < 0.3

        # 2. 量比趋势形态
        patterns_df['VR_RISING'] = volume_ratio > volume_ratio.shift(1)
        patterns_df['VR_FALLING'] = volume_ratio < volume_ratio.shift(1)
        patterns_df['VR_ACCELERATING_UP'] = (volume_ratio > volume_ratio.shift(1)) & (volume_ratio.shift(1) > volume_ratio.shift(2))
        patterns_df['VR_ACCELERATING_DOWN'] = (volume_ratio < volume_ratio.shift(1)) & (volume_ratio.shift(1) < volume_ratio.shift(2))

        # 3. 量比与均线关系
        patterns_df['VR_ABOVE_MA'] = volume_ratio > volume_ratio_ma
        patterns_df['VR_BELOW_MA'] = volume_ratio < volume_ratio_ma
        patterns_df['VR_GOLDEN_CROSS'] = (volume_ratio > volume_ratio_ma) & (volume_ratio.shift(1) <= volume_ratio_ma.shift(1))
        patterns_df['VR_DEATH_CROSS'] = (volume_ratio < volume_ratio_ma) & (volume_ratio.shift(1) >= volume_ratio_ma.shift(1))

        # 4. 量比突破形态
        patterns_df['VR_BREAKOUT_HIGH'] = (volume_ratio > 1.5) & (volume_ratio.shift(1) <= 1.5)
        patterns_df['VR_BREAKDOWN_LOW'] = (volume_ratio < 0.7) & (volume_ratio.shift(1) >= 0.7)

        # 5. 量比极值形态
        if len(volume_ratio) >= 20:
            vr_20_max = volume_ratio.rolling(window=20).max()
            vr_20_min = volume_ratio.rolling(window=20).min()
            patterns_df['VR_PEAK'] = volume_ratio >= vr_20_max
            patterns_df['VR_TROUGH'] = volume_ratio <= vr_20_min
        else:
            patterns_df['VR_PEAK'] = False
            patterns_df['VR_TROUGH'] = False

        return patterns_df

    def calculate_confidence(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """
        计算Volume Ratio指标的置信度

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
            # 检查Volume Ratio形态
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
    
    def _calculate(self, df: pd.DataFrame) -> pd.DataFrame:
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
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算Volume Ratio原始评分

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.Series: 评分序列 (0-100)
        """
        # 确保已计算指标
        if self._result is None:
            self.calculate(data)

        if self._result is None or 'volume_ratio' not in self._result.columns:
            return pd.Series(50.0, index=data.index)

        # 获取Volume Ratio数据
        volume_ratio = self._result['volume_ratio']
        volume_ratio_ma = self._result['volume_ratio_ma']

        # 初始化评分
        score = pd.Series(50.0, index=data.index)

        # 基于量比水平的评分
        # 量比>1.5为正面信号，<0.7为负面信号
        score += np.where(volume_ratio > 1.5,
                         np.minimum((volume_ratio - 1.5) * 20, 30),  # 最多加30分
                         0)

        score -= np.where(volume_ratio < 0.7,
                         np.minimum((0.7 - volume_ratio) * 30, 30),  # 最多减30分
                         0)

        # 基于量比与均线关系的评分
        score += np.where(volume_ratio > volume_ratio_ma, 5, -5)

        # 基于量比趋势的评分
        vr_change = volume_ratio.pct_change()
        score += np.where(vr_change > 0.1, 10,  # 量比快速上升
                         np.where(vr_change < -0.1, -10, 0))  # 量比快速下降

        # 基于量比稳定性的评分
        if len(volume_ratio) >= 5:
            vr_volatility = volume_ratio.rolling(window=5).std()
            # 量比过于波动降低评分
            score -= np.where(vr_volatility > 0.5, 5, 0)

        # 确保评分在0-100范围内
        return np.clip(score, 0, 100)

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
        注册Volume Ratio指标的形态到全局形态注册表
        """
        # 注册量比极高形态
        self.register_pattern_to_registry(
            pattern_id="VR_EXTREMELY_HIGH",
            display_name="量比极高",
            description="量比超过3倍，市场极度活跃，可能是重大消息或主力行为",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=25.0
        )

        # 注册量比高形态
        self.register_pattern_to_registry(
            pattern_id="VR_HIGH",
            display_name="量比偏高",
            description="量比在1.5-2倍之间，市场活跃度较高",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=15.0
        )

        # 注册量比极低形态
        self.register_pattern_to_registry(
            pattern_id="VR_EXTREMELY_LOW",
            display_name="量比极低",
            description="量比低于0.3倍，市场极度冷清",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-20.0
        )

        # 注册量比低形态
        self.register_pattern_to_registry(
            pattern_id="VR_LOW",
            display_name="量比偏低",
            description="量比在0.5-0.8倍之间，市场活跃度较低",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-10.0
        )

        # 注册量比突破形态
        self.register_pattern_to_registry(
            pattern_id="VR_BREAKOUT_HIGH",
            display_name="量比突破",
            description="量比突破1.5倍，市场活跃度显著提升",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=20.0
        )

        # 注册量比跌破形态
        self.register_pattern_to_registry(
            pattern_id="VR_BREAKDOWN_LOW",
            display_name="量比跌破",
            description="量比跌破0.7倍，市场活跃度显著下降",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-15.0
        )

        # 注册量比金叉形态
        self.register_pattern_to_registry(
            pattern_id="VR_GOLDEN_CROSS",
            display_name="量比金叉",
            description="量比上穿均线，活跃度趋势向好",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=12.0
        )

        # 注册量比死叉形态
        self.register_pattern_to_registry(
            pattern_id="VR_DEATH_CROSS",
            display_name="量比死叉",
            description="量比下穿均线，活跃度趋势转弱",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-12.0
        )

        # 注册量比加速上升形态
        self.register_pattern_to_registry(
            pattern_id="VR_ACCELERATING_UP",
            display_name="量比加速上升",
            description="量比连续上升，市场活跃度快速提升",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=18.0
        )

        # 注册量比加速下降形态
        self.register_pattern_to_registry(
            pattern_id="VR_ACCELERATING_DOWN",
            display_name="量比加速下降",
            description="量比连续下降，市场活跃度快速下降",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-18.0
        )

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