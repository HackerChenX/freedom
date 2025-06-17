#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
市场情绪分析

整合市场情绪分析指标，提供综合的市场情绪评估
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple, Any
import warnings

from indicators.base_indicator import BaseIndicator
from indicators.rsi import RSI
from indicators.cci import CCI
from indicators.vr import VR
from indicators.vosc import VOSC
from utils.logger import get_logger

# 静默警告
warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = get_logger(__name__)


class SentimentAnalysis(BaseIndicator):
    """
    市场情绪分析
    
    整合多种技术指标和市场数据，提供综合的市场情绪评估
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """
        初始化市场情绪分析
        
        Args:
            params: 参数字典，可包含：
                - rsi_period: RSI周期，默认为14
                - cci_period: CCI周期，默认为20
                - lookback_period: 回溯周期，默认为20
                - fear_threshold: 恐慌阈值，默认为30
                - greed_threshold: 贪婪阈值，默认为70
                - indicator_weights: 指标权重字典
        """
        default_params = {
            'rsi_period': 14,
            'cci_period': 20,
            'lookback_period': 20,
            'fear_threshold': 30,
            'greed_threshold': 70,
            'indicator_weights': {
                'rsi': 0.25,
                'cci': 0.15,
                'vr': 0.15,
                'price_momentum': 0.15,
                'volatility': 0.15,
                'volume_trend': 0.15
            }
        }
        
        if params:
            default_params.update(params)
        
        super().__init__(name="SentimentAnalysis", description="市场情绪分析")
        self._parameters = default_params
        
        # 初始化指标
        self.rsi = RSI({'period': self._parameters['rsi_period']})
        self.cci = CCI({'period': self._parameters['cci_period']})
        self.vr = VR()
        self.vosc = VOSC()
    
    def _calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算市场情绪指标
        
        Args:
            df: 输入数据，必须包含OHLCV数据
            
        Returns:
            pd.DataFrame: 计算结果，包含情绪指标
        """
        # 验证输入数据
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"输入数据缺少必要的列: {col}")
        
        # 初始化结果DataFrame
        result = pd.DataFrame(index=df.index)
        
        # 1. 计算RSI指标（衡量价格动量）
        rsi_result = self.rsi.calculate(df)
        if 'RSI' in rsi_result.columns:
            result['rsi'] = rsi_result['RSI']
        
        # 2. 计算CCI指标（衡量市场强弱）
        cci_result = self.cci.calculate(df)
        if 'CCI' in cci_result.columns:
            result['cci'] = cci_result['CCI']
        
        # 3. 计算VR指标（衡量成交量比率）
        vr_result = self.vr.calculate(df)
        if 'VR' in vr_result.columns:
            result['vr'] = vr_result['VR']
        
        # 4. 计算成交量震荡指标VOSC
        vosc_result = self.vosc.calculate(df)
        if 'VOSC' in vosc_result.columns:
            result['vosc'] = vosc_result['VOSC']
        
        # 5. 计算价格动量指标
        result['price_momentum'] = self._calculate_price_momentum(df)
        
        # 6. 计算波动率指标
        result['volatility'] = self._calculate_volatility(df)
        
        # 7. 计算成交量趋势
        result['volume_trend'] = self._calculate_volume_trend(df)
        
        # 8. 计算综合情绪指数
        result['sentiment_index'] = self._calculate_sentiment_index(result)
        
        # 9. 计算情绪分类
        result['sentiment_category'] = self._classify_sentiment(result['sentiment_index'])
        
        # 10. 计算情绪偏差
        result['sentiment_bias'] = self._calculate_sentiment_bias(result['sentiment_index'])
        
        # 11. 计算情绪转变信号
        result['sentiment_change'] = self._detect_sentiment_change(result['sentiment_index'])
        
        # 12. 计算过度情绪区间
        result['extreme_sentiment'] = self._detect_extreme_sentiment(result['sentiment_index'])
        
        # 13. 计算情绪反转概率
        result['reversal_probability'] = self._calculate_reversal_probability(result)
        
        return result
    
    def _calculate_price_momentum(self, df: pd.DataFrame) -> pd.Series:
        """
        计算价格动量指标
        
        Args:
            df: 输入数据
            
        Returns:
            pd.Series: 价格动量指标（0-100）
        """
        close = df['close']
        lookback = self._parameters['lookback_period']
        
        # 计算n日价格变化百分比
        price_change = close.pct_change(lookback) * 100
        
        # 将价格变化映射到0-100的区间
        # 假设±10%是正常的波动范围
        momentum = (price_change + 10) * 5
        momentum = momentum.clip(0, 100)
        
        return momentum
    
    def _calculate_volatility(self, df: pd.DataFrame) -> pd.Series:
        """
        计算波动率指标
        
        Args:
            df: 输入数据
            
        Returns:
            pd.Series: 波动率指标（0-100，越高表示波动越大）
        """
        close = df['close']
        lookback = self._parameters['lookback_period']
        
        # 计算日收益率
        returns = close.pct_change()
        
        # 计算n日波动率（标准差）
        volatility = returns.rolling(window=lookback).std() * np.sqrt(252) * 100
        
        # 将波动率映射到0-100的区间
        # 假设年化波动率40%是较高的波动水平
        normalized_volatility = volatility * 2.5
        normalized_volatility = normalized_volatility.clip(0, 100)
        
        return normalized_volatility
    
    def _calculate_volume_trend(self, df: pd.DataFrame) -> pd.Series:
        """
        计算成交量趋势指标
        
        Args:
            df: 输入数据
            
        Returns:
            pd.Series: 成交量趋势指标（0-100）
        """
        volume = df['volume']
        lookback = self._parameters['lookback_period']
        
        # 计算成交量的移动平均
        volume_ma = volume.rolling(window=lookback).mean()
        
        # 计算当前成交量相对于移动平均的比率
        volume_ratio = volume / volume_ma
        
        # 将比率映射到0-100的区间
        # 当成交量是移动平均的2倍时，认为是高成交量
        normalized_volume = (volume_ratio - 0.5) * 50
        normalized_volume = normalized_volume.clip(0, 100)
        
        return normalized_volume
    
    def _calculate_sentiment_index(self, indicators: pd.DataFrame) -> pd.Series:
        """
        计算综合情绪指数
        
        Args:
            indicators: 包含各种情绪指标的DataFrame
            
        Returns:
            pd.Series: 综合情绪指数（0-100）
        """
        # 获取各指标的权重
        weights = self._parameters['indicator_weights']
        
        # 初始化结果
        sentiment_index = pd.Series(50, index=indicators.index)  # 默认中性情绪
        
        # 检查哪些指标可用
        available_indicators = {}
        for indicator, weight in weights.items():
            if indicator in indicators.columns:
                available_indicators[indicator] = weight
        
        # 如果没有可用指标，返回默认值
        if not available_indicators:
            return sentiment_index
        
        # 归一化权重
        total_weight = sum(available_indicators.values())
        normalized_weights = {ind: w / total_weight for ind, w in available_indicators.items()}
        
        # 计算加权平均
        for indicator, weight in normalized_weights.items():
            # 对于CCI，需要将其映射到0-100的区间
            if indicator == 'cci':
                # CCI的范围很广，通常在±200之间
                cci_normalized = (indicators[indicator] + 200) / 4
                cci_normalized = cci_normalized.clip(0, 100)
                sentiment_index += weight * cci_normalized
            else:
                sentiment_index += weight * indicators[indicator]
        
        return sentiment_index
    
    def _classify_sentiment(self, sentiment_index: pd.Series) -> pd.Series:
        """
        将情绪指数分类为不同情绪类别
        
        Args:
            sentiment_index: 情绪指数
            
        Returns:
            pd.Series: 情绪类别
        """
        # 定义分类阈值
        extreme_fear_threshold = 20
        fear_threshold = self._parameters['fear_threshold']
        neutral_low_threshold = 45
        neutral_high_threshold = 55
        greed_threshold = self._parameters['greed_threshold']
        extreme_greed_threshold = 80
        
        # 初始化结果
        category = pd.Series("中性", index=sentiment_index.index)
        
        # 分类
        category[sentiment_index < extreme_fear_threshold] = "极度恐慌"
        category[(sentiment_index >= extreme_fear_threshold) & (sentiment_index < fear_threshold)] = "恐慌"
        category[(sentiment_index >= fear_threshold) & (sentiment_index < neutral_low_threshold)] = "轻微恐慌"
        category[(sentiment_index >= neutral_low_threshold) & (sentiment_index <= neutral_high_threshold)] = "中性"
        category[(sentiment_index > neutral_high_threshold) & (sentiment_index <= greed_threshold)] = "轻微贪婪"
        category[(sentiment_index > greed_threshold) & (sentiment_index <= extreme_greed_threshold)] = "贪婪"
        category[sentiment_index > extreme_greed_threshold] = "极度贪婪"
        
        return category
    
    def _calculate_sentiment_bias(self, sentiment_index: pd.Series) -> pd.Series:
        """
        计算情绪偏差
        
        Args:
            sentiment_index: 情绪指数
            
        Returns:
            pd.Series: 情绪偏差（-100到100，负值表示恐慌偏差，正值表示贪婪偏差）
        """
        # 相对于中性（50）的偏差
        bias = sentiment_index - 50
        
        return bias
    
    def _detect_sentiment_change(self, sentiment_index: pd.Series) -> pd.Series:
        """
        检测情绪转变信号
        
        Args:
            sentiment_index: 情绪指数
            
        Returns:
            pd.Series: 情绪转变信号（1=转向贪婪，-1=转向恐慌，0=无明显转变）
        """
        # 计算情绪指数的短期和长期移动平均
        short_ma = sentiment_index.rolling(window=5).mean()
        long_ma = sentiment_index.rolling(window=15).mean()
        
        # 初始化结果
        change = pd.Series(0, index=sentiment_index.index)
        
        # 短期均线上穿长期均线，表示情绪转向贪婪
        bullish_crossover = (short_ma.shift(1) <= long_ma.shift(1)) & (short_ma > long_ma)
        change[bullish_crossover] = 1
        
        # 短期均线下穿长期均线，表示情绪转向恐慌
        bearish_crossover = (short_ma.shift(1) >= long_ma.shift(1)) & (short_ma < long_ma)
        change[bearish_crossover] = -1
        
        return change
    
    def _detect_extreme_sentiment(self, sentiment_index: pd.Series) -> pd.Series:
        """
        检测过度情绪区间
        
        Args:
            sentiment_index: 情绪指数
            
        Returns:
            pd.Series: 过度情绪标记（1=过度贪婪，-1=过度恐慌，0=正常）
        """
        # 定义极端情绪阈值
        extreme_fear_threshold = 20
        extreme_greed_threshold = 80
        
        # 初始化结果
        extreme = pd.Series(0, index=sentiment_index.index)
        
        # 标记极端情绪
        extreme[sentiment_index <= extreme_fear_threshold] = -1  # 过度恐慌
        extreme[sentiment_index >= extreme_greed_threshold] = 1   # 过度贪婪
        
        return extreme
    
    def _calculate_reversal_probability(self, indicators: pd.DataFrame) -> pd.Series:
        """
        计算情绪反转概率
        
        Args:
            indicators: 包含各种情绪指标的DataFrame
            
        Returns:
            pd.Series: 情绪反转概率（0-100%）
        """
        # 初始化结果
        probability = pd.Series(50, index=indicators.index)  # 默认50%的反转概率
        
        # 计算情绪指数的距离极值的程度
        if 'sentiment_index' in indicators.columns:
            sentiment = indicators['sentiment_index']
            
            # 情绪指数距离极值越近，反转概率越高
            # 过度贪婪区域（>80）
            greed_area = sentiment > 80
            # 距离100越近，反转概率越高
            probability[greed_area] = 50 + (sentiment[greed_area] - 80) * 2.5
            
            # 过度恐慌区域（<20）
            fear_area = sentiment < 20
            # 距离0越近，反转概率越高
            probability[fear_area] = 50 + (20 - sentiment[fear_area]) * 2.5
        
        # 考虑情绪持续时间的影响
        if 'extreme_sentiment' in indicators.columns:
            extreme = indicators['extreme_sentiment']
            
            # 计算极端情绪的持续天数
            duration = pd.Series(0, index=indicators.index)
            
            for i in range(1, len(extreme)):
                if extreme.iloc[i] == extreme.iloc[i-1] and extreme.iloc[i] != 0:
                    duration.iloc[i] = duration.iloc[i-1] + 1
            
            # 持续时间越长，反转概率越高（但最多增加30%）
            probability += duration * 2
            probability = probability.clip(0, 100)
        
        return probability

    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取情绪分析指标的技术形态

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含形态信息的DataFrame
        """
        # 确保已计算情绪分析
        if not self.has_result():
            self._calculate(data)

        result = self._result.copy()
        patterns_df = pd.DataFrame(index=data.index)

        # 基于情绪分类的形态
        if 'sentiment_category' in result.columns:
            sentiment_cat = result['sentiment_category']
            patterns_df['EXTREME_FEAR'] = sentiment_cat == "极度恐慌"
            patterns_df['FEAR'] = sentiment_cat == "恐慌"
            patterns_df['SLIGHT_FEAR'] = sentiment_cat == "轻微恐慌"
            patterns_df['NEUTRAL'] = sentiment_cat == "中性"
            patterns_df['SLIGHT_GREED'] = sentiment_cat == "轻微贪婪"
            patterns_df['GREED'] = sentiment_cat == "贪婪"
            patterns_df['EXTREME_GREED'] = sentiment_cat == "极度贪婪"

        # 基于情绪转变的形态
        if 'sentiment_change' in result.columns:
            sentiment_change = result['sentiment_change']
            patterns_df['SENTIMENT_TURN_BULLISH'] = sentiment_change == 1
            patterns_df['SENTIMENT_TURN_BEARISH'] = sentiment_change == -1

        # 基于极端情绪的形态
        if 'extreme_sentiment' in result.columns:
            extreme_sentiment = result['extreme_sentiment']
            patterns_df['EXTREME_SENTIMENT_BULLISH'] = extreme_sentiment == -1  # 过度恐慌，反转看涨
            patterns_df['EXTREME_SENTIMENT_BEARISH'] = extreme_sentiment == 1   # 过度贪婪，反转看跌

        # 基于反转概率的形态
        if 'reversal_probability' in result.columns:
            reversal_prob = result['reversal_probability']
            patterns_df['HIGH_REVERSAL_PROBABILITY'] = reversal_prob > 70
            patterns_df['MEDIUM_REVERSAL_PROBABILITY'] = (reversal_prob >= 60) & (reversal_prob <= 70)

        return patterns_df

    def register_patterns(self):
        """
        注册SentimentAnalysis指标的形态到全局形态注册表
        """
        # 注册极端恐慌情绪形态
        self.register_pattern_to_registry(
            pattern_id="EXTREME_FEAR",
            display_name="极度恐慌",
            description="市场情绪极度恐慌，可能是买入机会",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=20.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="FEAR",
            display_name="恐慌情绪",
            description="市场情绪恐慌，可能接近底部",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=15.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="SLIGHT_FEAR",
            display_name="轻微恐慌",
            description="市场情绪轻微恐慌，谨慎乐观",
            pattern_type="BULLISH",
            default_strength="WEAK",
            score_impact=8.0,
            polarity="POSITIVE"
        )

        # 注册极端贪婪情绪形态
        self.register_pattern_to_registry(
            pattern_id="EXTREME_GREED",
            display_name="极度贪婪",
            description="市场情绪极度贪婪，可能是卖出机会",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-20.0,
            polarity="NEGATIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="GREED",
            display_name="贪婪情绪",
            description="市场情绪贪婪，可能接近顶部",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-15.0,
            polarity="NEGATIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="SLIGHT_GREED",
            display_name="轻微贪婪",
            description="市场情绪轻微贪婪，谨慎看跌",
            pattern_type="BEARISH",
            default_strength="WEAK",
            score_impact=-8.0,
            polarity="NEGATIVE"
        )

        # 注册中性情绪形态
        self.register_pattern_to_registry(
            pattern_id="NEUTRAL",
            display_name="中性情绪",
            description="市场情绪中性，观望为主",
            pattern_type="NEUTRAL",
            default_strength="WEAK",
            score_impact=0.0,
            polarity="NEUTRAL"
        )

        # 注册情绪转变形态
        self.register_pattern_to_registry(
            pattern_id="SENTIMENT_TURN_BULLISH",
            display_name="情绪转向看涨",
            description="市场情绪从恐慌转向贪婪，看涨信号",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=22.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="SENTIMENT_TURN_BEARISH",
            display_name="情绪转向看跌",
            description="市场情绪从贪婪转向恐慌，看跌信号",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-22.0,
            polarity="NEGATIVE"
        )

        # 注册极端情绪反转形态
        self.register_pattern_to_registry(
            pattern_id="EXTREME_SENTIMENT_BULLISH",
            display_name="极端恐慌反转",
            description="极度恐慌情绪，反转看涨机会",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=25.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="EXTREME_SENTIMENT_BEARISH",
            display_name="极端贪婪反转",
            description="极度贪婪情绪，反转看跌机会",
            pattern_type="BEARISH",
            default_strength="VERY_STRONG",
            score_impact=-25.0,
            polarity="NEGATIVE"
        )

        # 注册反转概率形态
        self.register_pattern_to_registry(
            pattern_id="HIGH_REVERSAL_PROBABILITY",
            display_name="高反转概率",
            description="情绪反转概率较高，需要关注趋势变化",
            pattern_type="NEUTRAL",
            default_strength="MEDIUM",
            score_impact=0.0,
            polarity="NEUTRAL"
        )

        self.register_pattern_to_registry(
            pattern_id="MEDIUM_REVERSAL_PROBABILITY",
            display_name="中等反转概率",
            description="情绪反转概率中等，保持观察",
            pattern_type="NEUTRAL",
            default_strength="WEAK",
            score_impact=0.0,
            polarity="NEUTRAL"
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
            self.calculate(data)
        
        # 初始化信号
        signals = {}
        signals['buy_signal'] = pd.Series(False, index=data.index)
        signals['sell_signal'] = pd.Series(False, index=data.index)
        signals['signal_strength'] = pd.Series(0, index=data.index)
    
        # 在这里实现指标特定的信号生成逻辑
        # 此处提供默认实现
    
        return signals
        
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算指标原始评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 评分(0-100)
        """
        # 确保已计算指标
        if not self.has_result():
            self.calculate(data)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)
        
        # 初始化评分
        score = pd.Series(50.0, index=data.index)
    
        # 在这里实现指标特定的评分逻辑
        # 此处提供默认实现
    
        return score 

    def get_pattern_info(self, pattern_id: str) -> dict:
        """
        获取形态信息
        
        Args:
            pattern_id: 形态ID
            
        Returns:
            dict: 形态信息字典
        """
        # 默认形态信息映射
        pattern_info_map = {
            # 基础形态
            'bullish': {'name': '看涨形态', 'description': '指标显示看涨信号', 'type': 'BULLISH'},
            'bearish': {'name': '看跌形态', 'description': '指标显示看跌信号', 'type': 'BEARISH'},
            'neutral': {'name': '中性形态', 'description': '指标显示中性信号', 'type': 'NEUTRAL'},
            
            # 通用形态
            'strong_signal': {'name': '强信号', 'description': '强烈的技术信号', 'type': 'STRONG'},
            'weak_signal': {'name': '弱信号', 'description': '较弱的技术信号', 'type': 'WEAK'},
            'trend_up': {'name': '上升趋势', 'description': '价格呈上升趋势', 'type': 'BULLISH'},
            'trend_down': {'name': '下降趋势', 'description': '价格呈下降趋势', 'type': 'BEARISH'},
        }
        
        # 默认形态信息
        default_pattern = {
            'name': pattern_id.replace('_', ' ').title(),
            'description': f'{pattern_id}形态',
            'type': 'UNKNOWN'
        }
        
        return pattern_info_map.get(pattern_id, default_pattern)
