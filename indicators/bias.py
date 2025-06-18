#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
均线多空指标(BIAS)

(收盘价-MA)/MA×100%
"""

import pandas as pd
import numpy as np
from typing import List

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger

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
        """
        super().__init__(name, description)
        self.periods = periods if periods is not None else [period]
        self.indicator_type = "BIAS"
        self.REQUIRED_COLUMNS = ['close']  # 添加必需列定义
        
    def set_parameters(self, period: int = 14, **kwargs):
        """
        设置BIAS指标的参数
        """
        self.periods = kwargs.get('periods', [period])

    def _validate_dataframe(self, df: pd.DataFrame, required_columns: List[str]) -> None:
        """
        验证DataFrame是否包含所需的列
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"DataFrame缺少必要的列: {', '.join(missing_columns)}")
    
    def _calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算均线多空指标(BIAS)指标
        """
        if data.empty:
            return data
            
        self._validate_dataframe(data, ['close'])
        
        # 创建一个临时的DataFrame来存储新计算的列
        result_df = pd.DataFrame(index=data.index)
        
        # 计算所有周期的BIAS
        for p in self.periods:
            ma = data['close'].rolling(window=p, min_periods=1).mean()
            result_df[f'BIAS{p}'] = (data['close'] - ma) / ma * 100
        
        # 为主周期创建 'BIAS' 和 'BIAS_MA' 列，以供形态识别使用
        if self.periods:
            main_period = self.periods[0]
            main_bias_col = f'BIAS{main_period}'
            if main_bias_col in result_df:
                result_df['BIAS'] = result_df[main_bias_col]
                result_df['BIAS_MA'] = result_df['BIAS'].rolling(window=main_period, min_periods=1).mean()

        # 只返回计算出的指标列，不包含原始数据列
        return result_df

    def get_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        识别所有已注册的BIAS相关形态
        """
        # 首先，调用calculate来获取所有需要的列
        calculated_data = self._calculate(data)

        # 验证必要的列是否存在
        required_cols = ['BIAS', 'BIAS_MA']
        if not all(col in calculated_data.columns for col in required_cols):
             logger.warning(f"BIAS指标在形态识别时缺少必要的计算列: {required_cols}")
             # 返回一个空的DataFrame，但保留索引
             return pd.DataFrame(index=data.index)

        # 实现BIAS形态识别逻辑
        bias_values = calculated_data['BIAS']

        # 创建形态识别结果DataFrame，只包含形态列
        patterns_df = pd.DataFrame(index=data.index)

        # 1. BIAS极值形态
        patterns_df['BIAS_EXTREME_HIGH'] = bias_values > 15.0
        patterns_df['BIAS_EXTREME_LOW'] = bias_values < -15.0

        # 2. BIAS中度偏离形态
        patterns_df['BIAS_MODERATE_HIGH'] = (bias_values > 5.0) & (bias_values <= 15.0)
        patterns_df['BIAS_MODERATE_LOW'] = (bias_values < -5.0) & (bias_values >= -15.0)

        # 3. BIAS中性形态
        patterns_df['BIAS_NEUTRAL'] = (bias_values >= -5.0) & (bias_values <= 5.0)

        # 4. BIAS背离形态（简化版本）
        if len(bias_values) >= 10:
            # 计算价格和BIAS的相关性来检测背离
            # 使用原始数据中的close列
            price_trend = data['close'].diff(5)  # 5日价格变化
            bias_trend = bias_values.diff(5)  # 5日BIAS变化

            # 背离：价格上涨但BIAS下降，或价格下跌但BIAS上升
            bullish_divergence = (price_trend < 0) & (bias_trend > 0)
            bearish_divergence = (price_trend > 0) & (bias_trend < 0)

            patterns_df['BIAS_BULLISH_DIVERGENCE'] = bullish_divergence
            patterns_df['BIAS_BEARISH_DIVERGENCE'] = bearish_divergence
            patterns_df['BIAS_DIVERGENCE'] = bullish_divergence | bearish_divergence
        else:
            patterns_df['BIAS_BULLISH_DIVERGENCE'] = False
            patterns_df['BIAS_BEARISH_DIVERGENCE'] = False
            patterns_df['BIAS_DIVERGENCE'] = False

        # 确保所有列都是布尔类型，填充NaN为False
        for col in patterns_df.columns:
            patterns_df[col] = patterns_df[col].fillna(False).astype(bool)

        return patterns_df

    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算BIAS指标的原始评分 (0-100分)

        评分逻辑：
        - BIAS值越接近0，评分越接近50（中性）
        - BIAS值为正且较大时，评分偏高（超买）
        - BIAS值为负且较大时，评分偏低（超卖）
        """
        # 首先计算指标值
        calculated_data = self._calculate(data)

        if 'BIAS' not in calculated_data.columns:
            return pd.Series(50.0, index=data.index)

        bias_values = calculated_data['BIAS']

        # 计算评分
        # BIAS在-10到+10之间为正常范围，对应40-60分
        # BIAS超过+10为超买，对应60-100分
        # BIAS低于-10为超卖，对应0-40分
        scores = pd.Series(50.0, index=data.index)

        # 处理有效值
        valid_mask = bias_values.notna()
        valid_bias = bias_values[valid_mask]

        if len(valid_bias) > 0:
            # 标准化BIAS值到评分
            # 使用sigmoid函数进行平滑转换
            normalized_bias = valid_bias / 10.0  # 将BIAS值标准化
            sigmoid_scores = 50 + 40 * (2 / (1 + pd.Series(np.exp(-normalized_bias), index=valid_bias.index)) - 1)
            scores[valid_mask] = sigmoid_scores.clip(0, 100)

        return scores

    def calculate_confidence(self, score: pd.Series, patterns: List[str], signals: dict) -> float:
        """
        计算BIAS指标的置信度

        Args:
            score: 得分序列
            patterns: 检测到的形态列表
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

        # 极端评分（超买/超卖）置信度较高
        if last_score > 70 or last_score < 30:
            confidence += 0.2
        # 中性评分置信度中等
        elif 40 <= last_score <= 60:
            confidence += 0.1
        else:
            confidence += 0.15

        # 2. 基于形态的置信度
        if isinstance(patterns, (list, pd.DataFrame)):
            if isinstance(patterns, pd.DataFrame):
                # 统计最近几个周期的形态数量
                try:
                    # 只统计数值列的形态
                    numeric_cols = patterns.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        recent_data = patterns[numeric_cols].iloc[-5:] if len(patterns) >= 5 else patterns[numeric_cols]
                        recent_patterns = recent_data.sum().sum()
                    else:
                        recent_patterns = 0
                except:
                    recent_patterns = 0
            else:
                recent_patterns = len(patterns)

            if recent_patterns > 0:
                confidence += min(recent_patterns * 0.05, 0.2)

        # 3. 基于评分稳定性的置信度
        if len(score) >= 5:
            recent_scores = score.iloc[-5:]
            score_stability = 1.0 - (recent_scores.std() / 50.0)  # 标准差越小，稳定性越高
            confidence += score_stability * 0.1

        return min(confidence, 1.0)

    def register_patterns(self):
        """
        注册BIAS指标的技术形态
        """
        # 注册BIAS极值形态
        self.register_pattern_to_registry(
            pattern_id="BIAS_EXTREME_HIGH",
            display_name="BIAS极高值",
            description="BIAS值超过+15%，表示严重超买",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-20.0,
            polarity="NEGATIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="BIAS_EXTREME_LOW",
            display_name="BIAS极低值",
            description="BIAS值低于-15%，表示严重超卖",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=20.0,
            polarity="POSITIVE"
        )

        # 注册BIAS中度偏离形态
        self.register_pattern_to_registry(
            pattern_id="BIAS_MODERATE_HIGH",
            display_name="BIAS中度偏高",
            description="BIAS值在+5%到+15%之间，表示轻度超买",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-10.0,
            polarity="NEGATIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="BIAS_MODERATE_LOW",
            display_name="BIAS中度偏低",
            description="BIAS值在-15%到-5%之间，表示轻度超卖",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=10.0,
            polarity="POSITIVE"
        )

        # 注册BIAS背离形态
        self.register_pattern_to_registry(
            pattern_id="BIAS_BULLISH_DIVERGENCE",
            display_name="BIAS底背离",
            description="价格创新低但BIAS未创新低，表明下跌动能减弱",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=15.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="BIAS_BEARISH_DIVERGENCE",
            display_name="BIAS顶背离",
            description="价格创新高但BIAS未创新高，表明上涨动能减弱",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-15.0,
            polarity="NEGATIVE"
        )

        # 注册BIAS中性形态
        self.register_pattern_to_registry(
            pattern_id="BIAS_NEUTRAL",
            display_name="BIAS中性",
            description="BIAS值在-5%到+5%之间，表示价格相对均衡",
            pattern_type="NEUTRAL",
            default_strength="WEAK",
            score_impact=0.0,
            polarity="NEUTRAL"
        )

    def get_pattern_info(self, pattern_id: str) -> dict:
        """
        获取指定形态的详细信息

        Args:
            pattern_id: 形态ID

        Returns:
            dict: 形态信息字典
        """
        pattern_info_map = {
            'BIAS_EXTREME_HIGH': {
                'name': 'BIAS极高值',
                'description': 'BIAS值超过+15%，表示严重超买',
                'strength': 'strong',
                'type': 'bearish'
            },
            'BIAS_EXTREME_LOW': {
                'name': 'BIAS极低值',
                'description': 'BIAS值低于-15%，表示严重超卖',
                'strength': 'strong',
                'type': 'bullish'
            },
            'BIAS_DIVERGENCE': {
                'name': 'BIAS背离',
                'description': '价格与BIAS指标出现背离',
                'strength': 'medium',
                'type': 'neutral'
            },
            'BIAS_MODERATE_HIGH': {
                'name': 'BIAS中度偏高',
                'description': 'BIAS值在+5%到+15%之间，表示轻度超买',
                'strength': 'medium',
                'type': 'bearish'
            },
            'BIAS_MODERATE_LOW': {
                'name': 'BIAS中度偏低',
                'description': 'BIAS值在-15%到-5%之间，表示轻度超卖',
                'strength': 'medium',
                'type': 'bullish'
            },
            'BIAS_NEUTRAL': {
                'name': 'BIAS中性',
                'description': 'BIAS值在-5%到+5%之间，表示价格相对均衡',
                'strength': 'weak',
                'type': 'neutral'
            }
        }

        return pattern_info_map.get(pattern_id, {
            'name': pattern_id,
            'description': f'BIAS形态: {pattern_id}',
            'strength': 'medium',
            'type': 'neutral'
        })
