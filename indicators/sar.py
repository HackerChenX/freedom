#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
抛物线转向系统(SAR)

判断价格趋势反转信号，提供买卖点
"""

import numpy as np
import pandas as pd
# import talib  # 暂时注释掉，使用自定义实现
from typing import Union, List, Dict, Optional, Tuple, Any

from indicators.base_indicator import BaseIndicator
from utils.indicator_utils import crossover, crossunder
from utils.logger import get_logger

logger = get_logger(__name__)


class SAR(BaseIndicator):
    """
    抛物线转向系统(SAR) (SAR)
    
    分类：趋势跟踪指标
    描述：判断价格趋势反转信号，提供买卖点
    """
    
    def __init__(self, acceleration: float = 0.02, maximum: float = 0.2):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """
        初始化抛物线转向系统(SAR)指标
        
        Args:
            acceleration: 加速因子，默认为0.02
            maximum: 加速因子最大值，默认为0.2
        """
        super().__init__(name="SAR", description="抛物线转向系统，判断价格趋势反转信号")
        self.acceleration = acceleration
        self.maximum = maximum
        
        # 注册SAR形态
        # self._register_sar_patterns()
        
    def set_parameters(self, acceleration: float = None, maximum: float = None):
        """
        设置指标参数
        """
        if acceleration is not None:
            self.acceleration = acceleration
        if maximum is not None:
            self.maximum = maximum

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算SAR指标

        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 其他参数

        Returns:
            包含SAR指标的DataFrame
        """
        return self._calculate(data)

    def calculate_confidence(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """
        计算SAR指标的置信度

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
            # 检查SAR形态
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
        
    def _validate_dataframe(self, df: pd.DataFrame, required_columns: List[str]) -> None:
        """
        验证DataFrame是否包含所需的列
        
        Args:
            df: 数据帧
            required_columns: 所需的列名列表
        
        Raises:
            ValueError: 如果DataFrame不包含所需的列
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"DataFrame缺少所需的列: {missing_columns}")
    
    def _calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算抛物线转向系统(SAR)指标
        
        Args:
            df: 包含OHLC数据的DataFrame
            
        Returns:
            包含SAR值的DataFrame
        """
        self._validate_dataframe(df, ['high', 'low', 'close'])
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        length = len(df)
        sar = np.zeros(length)
        trend = np.zeros(length)  # 1为上升趋势，-1为下降趋势
        ep = np.zeros(length)     # 极点价格
        af = np.zeros(length)     # 加速因子
        
        # 初始化
        trend[0] = 1  # 假设初始为上升趋势
        ep[0] = high[0]  # 极点价格初始为第一个最高价
        sar[0] = low[0]  # SAR初始为第一个最低价
        af[0] = self.acceleration
        
        # 计算SAR
        for i in range(1, length):
            # 上一个周期是上升趋势
            if trend[i-1] == 1:
                # 计算当前SAR值
                sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
                
                # 确保SAR不高于前两个周期的最低价
                if i >= 2:
                    sar[i] = min(sar[i], min(low[i-1], low[i-2]))
                
                # 判断趋势是否反转
                if low[i] < sar[i]:
                    # 趋势反转为下降
                    trend[i] = -1
                    sar[i] = ep[i-1]  # SAR值设为前期极点
                    ep[i] = low[i]     # 极点设为当前最低价
                    af[i] = self.acceleration  # 加速因子重置
                else:
                    # 继续上升趋势
                    trend[i] = 1
                    # 更新极点和加速因子
                    if high[i] > ep[i-1]:
                        ep[i] = high[i]
                        af[i] = min(af[i-1] + self.acceleration, self.maximum)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
            
            # 上一个周期是下降趋势
            else:
                # 计算当前SAR值
                sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
                
                # 确保SAR不低于前两个周期的最高价
                if i >= 2:
                    sar[i] = max(sar[i], max(high[i-1], high[i-2]))
                
                # 判断趋势是否反转
                if high[i] > sar[i]:
                    # 趋势反转为上升
                    trend[i] = 1
                    sar[i] = ep[i-1]  # SAR值设为前期极点
                    ep[i] = high[i]    # 极点设为当前最高价
                    af[i] = self.acceleration  # 加速因子重置
                else:
                    # 继续下降趋势
                    trend[i] = -1
                    # 更新极点和加速因子
                    if low[i] < ep[i-1]:
                        ep[i] = low[i]
                        af[i] = min(af[i-1] + self.acceleration, self.maximum)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
        
        # 创建结果DataFrame
        result = pd.DataFrame(index=df.index)
        result['sar'] = sar
        result['trend'] = trend
        result['ep'] = ep
        result['af'] = af
        
        # 存储结果
        self._result = result
        
        return result
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        根据SAR生成买卖信号
        
        Args:
            df: 包含SAR值的DataFrame
            
        Returns:
            包含买卖信号的DataFrame
        """
        signals = pd.DataFrame(index=df.index)
        trend = df['trend'].values
        
        buy_signals = np.zeros(len(df))
        sell_signals = np.zeros(len(df))
        
        # 寻找趋势反转点作为信号
        for i in range(1, len(df)):
            # 趋势由下降转为上升，买入信号
            if trend[i] == 1 and trend[i-1] == -1:
                buy_signals[i] = 1
            
            # 趋势由上升转为下降，卖出信号
            if trend[i] == -1 and trend[i-1] == 1:
                sell_signals[i] = 1
        
        signals['buy'] = buy_signals
        signals['sell'] = sell_signals
        
        return signals
    
    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取SAR相关形态

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含形态信息的DataFrame
        """
        # 确保已计算指标
        if self._result is None:
            self.calculate(data)

        if self._result is None or 'sar' not in self._result.columns:
            return pd.DataFrame(index=data.index)

        # 获取SAR和价格数据
        sar = self._result['sar']
        trend = self._result['trend']

        # 支持多种收盘价列名格式，包括中文列名
        close_columns = ['close', 'Close', 'CLOSE', 'c', 'C', '收盘', '收盘价', 'close_price']
        close_col = None

        for col in close_columns:
            if col in data.columns:
                close_col = col
                break

        if close_col is None:
            # 尝试从数值列中找到可能的收盘价列
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                # 如果有数值列，使用第一个作为收盘价（通常是价格数据）
                close_col = numeric_cols[0]
                logger.debug(f"SAR指标使用 {close_col} 列作为收盘价")
            else:
                logger.warning(f"SAR指标无法找到收盘价列，数据列名: {list(data.columns)}")
                return pd.DataFrame(index=data.index)

        close = data[close_col]

        # 创建形态DataFrame
        patterns_df = pd.DataFrame(index=data.index)

        # 1. SAR趋势反转形态
        patterns_df['SAR_BULLISH_REVERSAL'] = (trend == 1) & (trend.shift(1) == -1)
        patterns_df['SAR_BEARISH_REVERSAL'] = (trend == -1) & (trend.shift(1) == 1)

        # 2. SAR趋势持续形态
        patterns_df['SAR_UPTREND'] = trend == 1
        patterns_df['SAR_DOWNTREND'] = trend == -1

        # 3. SAR与价格位置关系
        patterns_df['SAR_BELOW_PRICE'] = sar < close
        patterns_df['SAR_ABOVE_PRICE'] = sar > close

        # 4. SAR距离形态
        price_distance = abs(close - sar) / close * 100
        patterns_df['SAR_CLOSE_TO_PRICE'] = price_distance < 1.0
        patterns_df['SAR_MODERATE_DISTANCE'] = (price_distance >= 1.0) & (price_distance < 3.0)
        patterns_df['SAR_FAR_FROM_PRICE'] = price_distance >= 3.0

        # 5. SAR加速因子形态
        if 'af' in self._result.columns:
            af = self._result['af']
            patterns_df['SAR_HIGH_ACCELERATION'] = af >= 0.15
            patterns_df['SAR_MEDIUM_ACCELERATION'] = (af >= 0.08) & (af < 0.15)
            patterns_df['SAR_LOW_ACCELERATION'] = af < 0.08
        else:
            patterns_df['SAR_HIGH_ACCELERATION'] = False
            patterns_df['SAR_MEDIUM_ACCELERATION'] = False
            patterns_df['SAR_LOW_ACCELERATION'] = False

        return patterns_df

    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算SAR原始评分

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.Series: 原始评分序列（0-100分）
        """
        # 确保已计算SAR
        if self._result is None:
            self.calculate(data)

        if self._result is None:
            return pd.Series(50.0, index=data.index)

        score = pd.Series(50.0, index=data.index)  # 基础分50分

        # 1. SAR趋势评分
        trend_score = self._calculate_sar_trend_score(data)
        score += trend_score

        # 2. SAR反转信号评分
        reversal_score = self._calculate_sar_reversal_score()
        score += reversal_score

        # 3. SAR距离评分
        distance_score = self._calculate_sar_distance_score(data)
        score += distance_score

        # 4. SAR加速因子评分
        acceleration_score = self._calculate_sar_acceleration_score()
        score += acceleration_score

        # 5. SAR稳定性评分
        stability_score = self._calculate_sar_stability_score()
        score += stability_score

        return np.clip(score, 0, 100)

    def _calculate_sar_trend_score(self, data: pd.DataFrame) -> pd.Series:
        """计算SAR趋势评分"""
        score = pd.Series(0.0, index=data.index)

        if 'trend' not in self._result.columns:
            return score

        trend = self._result['trend']

        # 上升趋势+15分，下降趋势-15分
        score += trend * 15

        return score

    def _calculate_sar_reversal_score(self) -> pd.Series:
        """计算SAR反转信号评分"""
        score = pd.Series(0.0, index=self._result.index)

        if 'trend' not in self._result.columns:
            return score

        trend = self._result['trend']

        # 趋势反转信号
        bullish_reversal = (trend == 1) & (trend.shift(1) == -1)
        bearish_reversal = (trend == -1) & (trend.shift(1) == 1)

        # 看涨反转+25分，看跌反转-25分
        score += bullish_reversal * 25
        score -= bearish_reversal * 25

        return score

    def _calculate_sar_distance_score(self, data: pd.DataFrame) -> pd.Series:
        """计算SAR距离评分"""
        score = pd.Series(0.0, index=data.index)

        if 'sar' not in self._result.columns:
            return score

        sar = self._result['sar']

        # 支持多种收盘价列名格式，包括中文列名
        close_columns = ['close', 'Close', 'CLOSE', 'c', 'C', '收盘', '收盘价', 'close_price']
        close_col = None

        for col in close_columns:
            if col in data.columns:
                close_col = col
                break

        if close_col is None:
            # 尝试从数值列中找到可能的收盘价列
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                close_col = numeric_cols[0]
                logger.debug(f"SAR指标使用 {close_col} 列作为收盘价")
            else:
                logger.warning(f"SAR指标无法找到收盘价列，数据列名: {list(data.columns)}")
                return score

        close = data[close_col]

        # 计算SAR与价格的相对距离
        distance_pct = abs(close - sar) / close * 100

        # 距离适中时评分较高
        score += np.where(distance_pct < 0.5, -5,  # 距离太近-5分
                         np.where(distance_pct < 2.0, 5,   # 距离适中+5分
                                 np.where(distance_pct < 5.0, 0, -10)))  # 距离太远-10分

        return score

    def _calculate_sar_acceleration_score(self) -> pd.Series:
        """计算SAR加速因子评分"""
        score = pd.Series(0.0, index=self._result.index)

        if 'af' not in self._result.columns:
            return score

        af = self._result['af']

        # 加速因子适中时评分较高
        score += np.where(af < 0.04, -5,  # 加速太慢-5分
                         np.where(af < 0.12, 5,   # 加速适中+5分
                                 np.where(af < 0.18, 0, -5)))  # 加速太快-5分

        return score

    def _calculate_sar_stability_score(self) -> pd.Series:
        """计算SAR稳定性评分"""
        score = pd.Series(0.0, index=self._result.index)

        if 'trend' not in self._result.columns:
            return score

        trend = self._result['trend']

        # 计算趋势持续性
        trend_changes = (trend != trend.shift(1)).astype(int)
        trend_stability = 1 - trend_changes.rolling(10, min_periods=1).mean()

        # 趋势稳定性高时+5分
        score += trend_stability * 5

        return score

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算SAR指标
        
        Args:
            df: 包含OHLC数据的DataFrame
            
        Returns:
            包含SAR值和信号的DataFrame
        """
        try:
            result = self.calculate(df)
            signals = self.generate_signals(result)
            # 合并结果
            result['buy_signal'] = signals['buy']
            result['sell_signal'] = signals['sell']
            
            return result
        except Exception as e:
            logger.error(f"计算指标 {self.name} 时出错: {str(e)}")
            raise

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
        注册SAR指标的形态到全局形态注册表
        """
        # 注册SAR看涨反转形态
        self.register_pattern_to_registry(
            pattern_id="SAR_BULLISH_REVERSAL",
            display_name="SAR看涨反转",
            description="SAR由下降趋势转为上升趋势，产生看涨信号",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=25.0,
            polarity="POSITIVE"
        )

        # 注册SAR看跌反转形态
        self.register_pattern_to_registry(
            pattern_id="SAR_BEARISH_REVERSAL",
            display_name="SAR看跌反转",
            description="SAR由上升趋势转为下降趋势，产生看跌信号",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-25.0,
            polarity="NEGATIVE"
        )

        # 注册SAR上升趋势形态
        self.register_pattern_to_registry(
            pattern_id="SAR_UPTREND",
            display_name="SAR上升趋势",
            description="SAR保持在价格下方，表示上升趋势",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=15.0,
            polarity="POSITIVE"
        )

        # 注册SAR下降趋势形态
        self.register_pattern_to_registry(
            pattern_id="SAR_DOWNTREND",
            display_name="SAR下降趋势",
            description="SAR保持在价格上方，表示下降趋势",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-15.0,
            polarity="NEGATIVE"
        )

        # 注册SAR高加速形态
        self.register_pattern_to_registry(
            pattern_id="SAR_HIGH_ACCELERATION",
            display_name="SAR高加速",
            description="SAR加速因子较高，趋势强劲但方向需结合位置判断",
            pattern_type="NEUTRAL",
            default_strength="STRONG",
            score_impact=0.0,
            polarity="NEUTRAL"
        )

        # 注册SAR距离形态
        self.register_pattern_to_registry(
            pattern_id="SAR_CLOSE_TO_PRICE",
            display_name="SAR接近价格",
            description="SAR与价格距离较近，可能即将反转",
            pattern_type="NEUTRAL",
            default_strength="WEAK",
            score_impact=0.0,
            polarity="NEUTRAL"
        )

        self.register_pattern_to_registry(
            pattern_id="SAR_FAR_FROM_PRICE",
            display_name="SAR远离价格",
            description="SAR与价格距离较远，趋势强劲但方向需结合位置判断",
            pattern_type="NEUTRAL",
            default_strength="MEDIUM",
            score_impact=0.0,
            polarity="NEUTRAL"
        )

    def _register_sar_patterns(self):
        """
        注册SAR形态
        """
        from indicators.pattern_registry import PatternRegistry, PatternType
        
        # 获取PatternRegistry实例
        registry = PatternRegistry()
        
        # 注册趋势反转形态
        registry.register(
            pattern_id="SAR_BULLISH_REVERSAL",
            display_name="SAR做多信号",
            description="SAR由下降趋势转为上升趋势，产生做多信号",
            indicator_id="SAR",
            pattern_type=PatternType.REVERSAL,
            score_impact=15.0
        )
        
        registry.register(
            pattern_id="SAR_BEARISH_REVERSAL",
            display_name="SAR做空信号",
            description="SAR由上升趋势转为下降趋势，产生做空信号",
            indicator_id="SAR",
            pattern_type=PatternType.REVERSAL,
            score_impact=-15.0
        )
        
        # 注册趋势持续形态
        registry.register(
            pattern_id="SAR_STRONG_UPTREND",
            display_name="SAR强势上升趋势",
            description="SAR长期保持在价格下方，表示强势上升趋势",
            indicator_id="SAR",
            pattern_type=PatternType.TREND,
            score_impact=10.0
        )
        
        registry.register(
            pattern_id="SAR_UPTREND",
            display_name="SAR上升趋势",
            description="SAR保持在价格下方，表示上升趋势",
            indicator_id="SAR",
            pattern_type=PatternType.TREND,
            score_impact=7.0
        )
        
        registry.register(
            pattern_id="SAR_SHORT_UPTREND",
            display_name="SAR短期上升趋势",
            description="SAR刚刚转为上升趋势",
            indicator_id="SAR",
            pattern_type=PatternType.TREND,
            score_impact=5.0
        )
        
        registry.register(
            pattern_id="SAR_STRONG_DOWNTREND",
            display_name="SAR强势下降趋势",
            description="SAR长期保持在价格上方，表示强势下降趋势",
            indicator_id="SAR",
            pattern_type=PatternType.TREND,
            score_impact=-10.0
        )
        
        registry.register(
            pattern_id="SAR_DOWNTREND",
            display_name="SAR下降趋势",
            description="SAR保持在价格上方，表示下降趋势",
            indicator_id="SAR",
            pattern_type=PatternType.TREND,
            score_impact=-7.0
        )
        
        registry.register(
            pattern_id="SAR_SHORT_DOWNTREND",
            display_name="SAR短期下降趋势",
            description="SAR刚刚转为下降趋势",
            indicator_id="SAR",
            pattern_type=PatternType.TREND,
            score_impact=-5.0
        )
        
        # 注册SAR距离形态
        registry.register(
            pattern_id="SAR_CLOSE_TO_PRICE",
            display_name="SAR接近价格",
            description="SAR与价格距离较近，可能即将反转",
            indicator_id="SAR",
            pattern_type=PatternType.WARNING,
            score_impact=0.0
        )
        
        registry.register(
            pattern_id="SAR_MODERATE_DISTANCE",
            display_name="SAR与价格中等距离",
            description="SAR与价格保持中等距离",
            indicator_id="SAR",
            pattern_type=PatternType.CONTINUATION,
            score_impact=5.0
        )
        
        registry.register(
            pattern_id="SAR_FAR_FROM_PRICE",
            display_name="SAR远离价格",
            description="SAR与价格距离较远，趋势强劲",
            indicator_id="SAR",
            pattern_type=PatternType.TREND,
            score_impact=8.0
        )
        
        # 注册加速因子形态
        registry.register(
            pattern_id="SAR_HIGH_ACCELERATION",
            display_name="SAR高加速趋势",
            description="SAR加速因子较高，趋势强劲",
            indicator_id="SAR",
            pattern_type=PatternType.MOMENTUM,
            score_impact=10.0
        )
        
        registry.register(
            pattern_id="SAR_MEDIUM_ACCELERATION",
            display_name="SAR中等加速趋势",
            description="SAR加速因子中等，趋势稳定",
            indicator_id="SAR",
            pattern_type=PatternType.MOMENTUM,
            score_impact=5.0
        )
        
        registry.register(
            pattern_id="SAR_LOW_ACCELERATION",
            display_name="SAR低加速趋势",
            description="SAR加速因子较低，趋势刚开始或较弱",
            indicator_id="SAR",
            pattern_type=PatternType.MOMENTUM,
            score_impact=2.0
        )
        
        # 注册趋势稳定性形态
        registry.register(
            pattern_id="SAR_STABLE_TREND",
            display_name="SAR稳定趋势",
            description="SAR趋势稳定，没有频繁转向",
            indicator_id="SAR",
            pattern_type=PatternType.STABILITY,
            score_impact=8.0
        )
        
        registry.register(
            pattern_id="SAR_VOLATILE_TREND",
            display_name="SAR波动趋势",
            description="SAR趋势不稳定，频繁转向",
            indicator_id="SAR",
            pattern_type=PatternType.STABILITY,
            score_impact=-5.0
        )
        
        # 注册支撑/阻力形态
        registry.register(
            pattern_id="SAR_AS_SUPPORT",
            display_name="SAR支撑",
            description="SAR作为价格支撑位",
            indicator_id="SAR",
            pattern_type=PatternType.SUPPORT_RESISTANCE,
            score_impact=12.0
        )
        
        registry.register(
            pattern_id="SAR_AS_RESISTANCE",
            display_name="SAR阻力",
            description="SAR作为价格阻力位",
            indicator_id="SAR",
            pattern_type=PatternType.SUPPORT_RESISTANCE,
            score_impact=-12.0
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

