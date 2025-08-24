#!/usr/bin/env python
from utils.dependency_injection import get_logger
# -*- coding: utf-8 -*-

"""
相对强弱指数(RSI_Rsi)

通过比较一段时期内平均收盘涨数和平均收盘跌数来分析市场买卖盘的意向和实力
"""

import numpy as np
from typing import Dict, Any
import pandas as pd
from typing import List, Dict, Any

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin
from utils.dependency_injection import get_logger

logger = get_logger(__name__)


class RsiRsi(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
    """
    相对强弱指数(RSI_Rsi)
    """

    def __init__(self, period: int = 14, ma_periods: List[int] = None, overbought: float = 70.0, oversold: float = 30.0):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        super().__init__()
        self.name = "RSI_Rsi"
        self.period = period
        self.ma_periods = ma_periods if ma_periods is not None else [5, 10]
        self.overbought = overbought
        self.oversold = oversold

        # 🔧 注册RSI形态到全局形态注册表 (关键修复)
        try:
            self.register_patterns_Rsi()
        except Exception as e:
            # 如果形态注册失败，记录警告但不影响指标初始化
            import logging
            logging.warning(f"RSI形态注册失败: {e}")

    @property
    def minimum_periods(self) -> int:
        """
        返回RSI指标计算所需的最少数据周期数

        Returns:
            int: 最少需要的数据周期数
        """
        # RSI需要period个周期计算，加上额外的缓冲期
        return self.period + max(10, self.period // 2)

    def _register_rsi_patterns(self):
        """
        注册RSI形态到全局形态注册表
        """
        # 注册RSI超买形态
        self.register_pattern_to_registry(
            pattern_id="RSI_OVERBOUGHT",
            display_name="RSI超买",
            description="RSI指标超过70，进入超买区域，存在回调压力",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-10.0,
            polarity="NEGATIVE"
        )
        
        # 注册RSI超卖形态
        self.register_pattern_to_registry(
            pattern_id="RSI_OVERSOLD",
            display_name="RSI超卖",
            description="RSI指标低于30，进入超卖区域，存在反弹机会",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=10.0,
            polarity="POSITIVE"
        )
        
        # 注册RSI底背离形态
        self.register_pattern_to_registry(
            pattern_id="RSI_BULLISH_DIVERGENCE",
            display_name="RSI底背离",
            description="价格创新低而RSI未创新低，形成底背离",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=20.0,
            polarity="POSITIVE"
        )
        
        # 注册RSI顶背离形态
        self.register_pattern_to_registry(
            pattern_id="RSI_BEARISH_DIVERGENCE",
            display_name="RSI顶背离",
            description="价格创新高而RSI未创新高，警示上涨动能不足",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-20.0,
            polarity="NEGATIVE"
        )

    def set_parameters_Rsi_Rsi_Rsi_rsi(self, period: int = 14, overbought: float = 70.0, oversold: float = 30.0, **kwargs):
        """
        设置RSI指标的参数
        """
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        if 'ma_periods' in kwargs:
            self.ma_periods = kwargs['ma_periods']

    def _calculate_rsi(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算RSI指标，并包含均线和信号
        
        Args:
            df: 包含价格数据的Data_frame
            
        Returns:
            pd.DataFrame: 添加了RSI指标的Data_frame
        """
        if data.empty:
            return data
            
        # 确保数据包含所需的列
        if 'close' not in data.columns:
            raise ValueError("输入数据必须包含'close'列")
            
        result_df = pd.DataFrame(index=data.index)
        
        # 计算价格变动
        delta = data['close'].diff()

        # 分离上涨和下跌
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)

        # 使用标准Wilder平滑方法计算RSI
        rsi_values = self._calculate_wilder_rsi(gains, losses, self.period)

        # 计算RSI
        result_df[f'rsi_{self.period}'] = rsi_values
        
        # 必须：计算RSI均线（确保形态识别正常工作）
        if self.ma_periods and len(self.ma_periods) >= 2:
            short_period = self.ma_periods[0]
            long_period = self.ma_periods[1]
        else:
            # 使用默认周期确保均线存在
            short_period = 5
            long_period = 10
        
        result_df[f'rsi_ma_{short_period}'] = result_df[f'rsi_{self.period}'].rolling(window=short_period).mean()
        result_df[f'rsi_ma_{long_period}'] = result_df[f'rsi_{self.period}'].rolling(window=long_period).mean()
        # For pattern detection - 确保这些列总是存在
        result_df['rsi_ma_short'] = result_df[f'rsi_ma_{short_period}']
        result_df['rsi_ma_long'] = result_df[f'rsi_ma_{long_period}']

        result_df['rsi_overbought'] = result_df[f'rsi_{self.period}'] > self.overbought
        result_df['rsi_oversold'] = result_df[f'rsi_{self.period}'] < self.oversold

        # 添加形态识别和信号生成
        result_df = self.add_pattern_detection(result_df)
        result_df = self.add_signal_generation(result_df)

        return result_df

    def _calculate_wilder_rsi(self, gains: pd.Series, losses: pd.Series, period: int) -> pd.Series:
        """
        使用标准Wilder平滑方法计算RSI

        Args:
            gains: 上涨序列
            losses: 下跌序列
            period: RSI周期

        Returns:
            pd.Series: RSI值序列
        """
        # 初始化结果序列
        rsi_values = pd.Series(index=gains.index, dtype=float)

        # 计算初始平均值（前period个值的简单平均）
        if len(gains) >= period:
            # 初始平均增益和损失
            initial_avg_gain = gains.iloc[1:period+1].mean()  # 跳过第一个NaN值
            initial_avg_loss = losses.iloc[1:period+1].mean()

            # 设置初始RSI值
            if initial_avg_loss == 0:
                rsi_values.iloc[period] = 100.0
            else:
                rs = initial_avg_gain / initial_avg_loss
                rsi_values.iloc[period] = 100 - (100 / (1 + rs))

            # 使用Wilder平滑方法计算后续值
            avg_gain = initial_avg_gain
            avg_loss = initial_avg_loss

            for i in range(period + 1, len(gains)):
                # Wilder平滑公式
                avg_gain = (avg_gain * (period - 1) + gains.iloc[i]) / period
                avg_loss = (avg_loss * (period - 1) + losses.iloc[i]) / period

                if avg_loss == 0:
                    rsi_values.iloc[i] = 100.0
                else:
                    rs = avg_gain / avg_loss
                    rsi_values.iloc[i] = 100 - (100 / (1 + rs))

        return rsi_values

    def get_patterns_Rsi_Rsi(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取RSI相关形态 - 生产级标准实现
        保持技术指标的准确性和专业性
        """
        calculated_data = self._calculate_rsi(data)
        patterns_df = pd.DataFrame(index=data.index)

        if f'rsi_{self.period}' not in calculated_data.columns:
            # 返回空形态但确保列存在
            patterns_df['RSI_OVERSOLD'] = False
            patterns_df['RSI_OVERBOUGHT'] = False
            patterns_df['RSI_GOLDEN_CROSS'] = False
            patterns_df['RSI_DEATH_CROSS'] = False
            return patterns_df

        rsi = calculated_data[f'rsi_{self.period}']

        # 🔧 生产级修复：使用标准RSI均线，确保技术准确性
        rsi_ma_short = calculated_data.get('rsi_ma_short', rsi.rolling(window=self.ma_periods[0], min_periods=1).mean())
        rsi_ma_long = calculated_data.get('rsi_ma_long', rsi.rolling(window=self.ma_periods[1], min_periods=1).mean())

        # 初始化形态列
        patterns_df['RSI_OVERSOLD'] = False
        patterns_df['RSI_OVERBOUGHT'] = False
        patterns_df['RSI_GOLDEN_CROSS'] = False
        patterns_df['RSI_DEATH_CROSS'] = False

        try:
            # 🔧 生产级修复1：标准RSI超买超卖条件
            # 超买：RSI持续在超买区域（不仅仅是刚突破）
            patterns_df['RSI_OVERBOUGHT'] = rsi > self.overbought
            patterns_df['RSI_OVERSOLD'] = rsi < self.oversold

            # 🔧 生产级修复2：标准RSI均线金叉死叉
            # 方法1：标准crossover检测
            try:
                from utils.indicator_utils import crossover, crossunder
                golden_cross_standard = crossover(rsi_ma_short, rsi_ma_long)
                death_cross_standard = crossunder(rsi_ma_short, rsi_ma_long)
            except ImportError:
                # 如果crossover函数不可用，使用标准逻辑
                golden_cross_standard = (rsi_ma_short > rsi_ma_long) & (rsi_ma_short.shift(1) <= rsi_ma_long.shift(1))
                death_cross_standard = (rsi_ma_short < rsi_ma_long) & (rsi_ma_short.shift(1) >= rsi_ma_long.shift(1))

            # 方法2：RSI中线穿越（50线）
            rsi_centerline_cross_up = (rsi > 50) & (rsi.shift(1) <= 50)
            rsi_centerline_cross_down = (rsi < 50) & (rsi.shift(1) >= 50)

            # 方法3：RSI背离检测（简化版）
            # 检测价格新高但RSI未新高（顶背离）
            price_high = data['close'].rolling(window=5).max()
            rsi_high = rsi.rolling(window=5).max()
            price_new_high = (data['close'] == price_high) & (data['close'] > data['close'].shift(5))
            rsi_no_new_high = (rsi < rsi_high.shift(1))
            rsi_bearish_divergence = price_new_high & rsi_no_new_high

            # 检测价格新低但RSI未新低（底背离）
            price_low = data['close'].rolling(window=5).min()
            rsi_low = rsi.rolling(window=5).min()
            price_new_low = (data['close'] == price_low) & (data['close'] < data['close'].shift(5))
            rsi_no_new_low = (rsi > rsi_low.shift(1))
            rsi_bullish_divergence = price_new_low & rsi_no_new_low

            # 🔧 生产级修复3：合并标准信号
            patterns_df['RSI_GOLDEN_CROSS'] = golden_cross_standard | rsi_centerline_cross_up | rsi_bullish_divergence
            patterns_df['RSI_DEATH_CROSS'] = death_cross_standard | rsi_centerline_cross_down | rsi_bearish_divergence

            # 添加背离形态
            patterns_df['RSI_BULLISH_DIVERGENCE'] = rsi_bullish_divergence
            patterns_df['RSI_BEARISH_DIVERGENCE'] = rsi_bearish_divergence

        except Exception as e:
            logger.warning(f"RSI形态识别失败: {e}")
            # 🔧 生产级兜底：标准RSI逻辑
            patterns_df['RSI_OVERSOLD'] = rsi < self.oversold
            patterns_df['RSI_OVERBOUGHT'] = rsi > self.overbought

            # 简单的金叉死叉逻辑
            patterns_df['RSI_GOLDEN_CROSS'] = (rsi > 50) & (rsi.shift(1) <= 50)
            patterns_df['RSI_DEATH_CROSS'] = (rsi < 50) & (rsi.shift(1) >= 50)
            patterns_df['RSI_BULLISH_DIVERGENCE'] = False
            patterns_df['RSI_BEARISH_DIVERGENCE'] = False

        # 🔧 生产级修复4：确保所有列都是布尔类型，填充NaN为False
        for col in patterns_df.columns:
            patterns_df[col] = patterns_df[col].fillna(False).astype(bool)

        return patterns_df
    def generate_signals_Rsi(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        生成RSI交易信号
        """
        calculated_data = self._calculate_rsi(data)

        patterns = self.get_patterns_Rsi_Rsi(data)

        signals = pd.DataFrame(index=data.index)
        signals['buy_signal'] = patterns['RSI_GOLDEN_CROSS'] | (patterns['RSI_OVERSOLD'])
        signals['sell_signal'] = patterns['RSI_DEATH_CROSS'] | (patterns['RSI_OVERBOUGHT'])

        return signals

    def calculate_raw_score_Rsi_Rsi(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算RSI指标的原始评分 (0-100分)
        """
        calculated_data = self._calculate_rsi(data)
        
        if calculated_data is None or f'rsi_{self.period}' not in calculated_data.columns:
            return pd.Series(50.0, index=data.index)
            
        score = pd.Series(50.0, index=data.index)
        rsi_values = calculated_data[f'rsi_{self.period}']
        
        # 基于RSI值的评分
        score += (rsi_values - 50) * 0.4 # 20-80 映射到 42-58
        
        # 超买超卖区域评分
        score[rsi_values > self.overbought] -= 15
        score[rsi_values < self.oversold] += 15
        
        # 均线交叉评分
        if 'rsi_ma_short' in calculated_data.columns and 'rsi_ma_long' in calculated_data.columns:
            short_ma = calculated_data['rsi_ma_short']
            long_ma = calculated_data['rsi_ma_long']
            
            from utils.indicator_utils import crossover, crossunder
            score[crossover(short_ma, long_ma)] += 20
            score[crossunder(short_ma, long_ma)] -= 20
            
        return score.clip(0, 100)

    def calculate_confidence_Rsi_Rsi(self, score: pd.Series, patterns: pd.DataFrame, signals: Dict[str, pd.Series]) -> float:
        """
        计算RSI指标的置信度

        Args:
            score: 得分序列
            patterns: 检测到的形态Data_frame
            signals: 生成的信号字典

        Returns:
            float: 置信度分数 (0-1)
        """
        if score.empty:
            return 0.5

        # 1. 基于得分的置信度
        last_score = score.iloc[-1]
        score_confidence = 0.5

        # 超买超卖区域置信度较高
        if last_score > 70 or last_score < 30:
            score_confidence = 0.8
        # 中性区域置信度中等
        elif 40 <= last_score <= 60:
            score_confidence = 0.6
        else:
            score_confidence = 0.7

        # 2. 基于形态的置信度
        pattern_confidence = 0.5
        if not patterns.empty:
            # 统计最近几个周期的形态数量
            recent_patterns = patterns.iloc[-5:].sum().sum() if len(patterns) >= 5 else patterns.sum().sum()

            if recent_patterns > 0:
                pattern_confidence = min(0.5 + recent_patterns * 0.1, 0.9)

        # 3. 基于信号的置信度
        signal_confidence = 0.5
        if signals:
            # 检查是否有强烈的买卖信号
            for signal_name, signal_series in signals.items():
                if isinstance(signal_series, pd.Series) and signal_series.iloc[-1]:
                    signal_confidence = 0.8
                    break

        # 综合置信度
        confidence = (score_confidence * 0.4 + pattern_confidence * 0.3 + signal_confidence * 0.3)

        return min(confidence, 1.0)

    def calculate_score_Rsi(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
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
            raw_scores = self.calculate_raw_score_Rsi_Rsi(data, **kwargs)

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
            patterns = self.get_patterns_Rsi_Rsi(data, **kwargs)
            signals = self.generate_signals_Rsi(data, **kwargs)

            # 3. 计算置信度
            confidence = self.calculate_confidence_Rsi_Rsi(raw_scores, patterns, signals.to_dict('series') if hasattr(signals, 'to_dict') else {})

            return {
                'score': final_score,
                'confidence': confidence
            }
        except Exception as e:
            logger.error(f"为指标 {self.name} 计算评分时出错: {e}")
            return {'score': 50.0, 'confidence': 0.0}

    def register_patterns_Rsi(self):
        """
        注册RSI指标的形态到全局形态注册表
        """
        # 注册RSI超买形态
        self.register_pattern_to_registry(
            pattern_id="RSI_OVERBOUGHT",
            display_name="RSI超买",
            description="RSI指标超过70，进入超买区域，存在回调压力",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-10.0,
            polarity="NEGATIVE"
        )
        
        # 注册RSI超卖形态
        self.register_pattern_to_registry(
            pattern_id="RSI_OVERSOLD",
            display_name="RSI超卖",
            description="RSI指标低于30，进入超卖区域，存在反弹机会",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=10.0,
            polarity="POSITIVE"
        )
        
        # 注册RSI底背离形态
        self.register_pattern_to_registry(
            pattern_id="RSI_BULLISH_DIVERGENCE",
            display_name="RSI底背离",
            description="价格创新低而RSI未创新低，形成底背离",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=20.0,
            polarity="POSITIVE"
        )
        
        # 注册RSI顶背离形态
        self.register_pattern_to_registry(
            pattern_id="RSI_BEARISH_DIVERGENCE",
            display_name="RSI顶背离",
            description="价格创新高而RSI未创新高，警示上涨动能不足",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-20.0,
            polarity="NEGATIVE"
        )

        # 🔧 注册RSI金叉形态 (关键修复)
        self.register_pattern_to_registry(
            pattern_id="RSI_GOLDEN_CROSS",
            display_name="RSI金叉",
            description="RSI从超卖区域回升或中线上穿，表明趋势转为看涨",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=15.0,
            polarity="POSITIVE"
        )

        # 🔧 注册RSI死叉形态 (关键修复)
        self.register_pattern_to_registry(
            pattern_id="RSI_DEATH_CROSS",
            display_name="RSI死叉",
            description="RSI从超买区域回落或中线下穿，表明趋势转为看跌",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-15.0,
            polarity="NEGATIVE"
        )

        # 🔧 注册RSI中线穿越形态
        self.register_pattern_to_registry(
            pattern_id="RSI_CENTERLINE_CROSS_UP",
            display_name="RSI中线上穿",
            description="RSI从下方穿越50中线，表明多头力量增强",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=12.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="RSI_CENTERLINE_CROSS_DOWN",
            display_name="RSI中线下穿",
            description="RSI从上方穿越50中线，表明空头力量增强",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-12.0,
            polarity="NEGATIVE"
        )

    def get_pattern_info_Rsi(self, pattern_id: str) -> dict:
        """
        获取指定形态的详细信息

        Args:
            pattern_id: 形态ID

        Returns:
            dict: 形态详细信息
        """
        pattern_info_map = {
            "RSI_OVERBOUGHT": {
                "id": "RSI_OVERBOUGHT",
                "name": "RSI超买",
                "description": "RSI指标超过70，进入超买区域，存在回调压力",
                "type": "BEARISH",
                "strength": "MEDIUM",
                "score_impact": -10.0
            },
            "RSI_OVERSOLD": {
                "id": "RSI_OVERSOLD",
                "name": "RSI超卖",
                "description": "RSI指标低于30，进入超卖区域，存在反弹机会",
                "type": "BULLISH",
                "strength": "MEDIUM",
                "score_impact": 10.0
            },
            "RSI_BULLISH_DIVERGENCE": {
                "id": "RSI_BULLISH_DIVERGENCE",
                "name": "RSI底背离",
                "description": "价格创新低而RSI未创新低，形成底背离",
                "type": "BULLISH",
                "strength": "STRONG",
                "score_impact": 20.0
            },
            "RSI_BEARISH_DIVERGENCE": {
                "id": "RSI_BEARISH_DIVERGENCE",
                "name": "RSI顶背离",
                "description": "价格创新高而RSI未创新高，警示上涨动能不足",
                "type": "BEARISH",
                "strength": "STRONG",
                "score_impact": -20.0
            },
            "RSI_GOLDEN_CROSS": {
                "id": "RSI_GOLDEN_CROSS",
                "name": "RSI金叉",
                "description": "RSI从超卖区域回升或中线上穿，表明趋势转为看涨",
                "type": "BULLISH",
                "strength": "MEDIUM",
                "score_impact": 15.0
            },
            "RSI_DEATH_CROSS": {
                "id": "RSI_DEATH_CROSS",
                "name": "RSI死叉",
                "description": "RSI从超买区域回落或中线下穿，表明趋势转为看跌",
                "type": "BEARISH",
                "strength": "MEDIUM",
                "score_impact": -15.0
            },
            "RSI_CENTERLINE_CROSS_UP": {
                "id": "RSI_CENTERLINE_CROSS_UP",
                "name": "RSI中线上穿",
                "description": "RSI从下方穿越50中线，表明多头力量增强",
                "type": "BULLISH",
                "strength": "MEDIUM",
                "score_impact": 12.0
            },
            "RSI_CENTERLINE_CROSS_DOWN": {
                "id": "RSI_CENTERLINE_CROSS_DOWN",
                "name": "RSI中线下穿",
                "description": "RSI从上方穿越50中线，表明空头力量增强",
                "type": "BEARISH",
                "strength": "MEDIUM",
                "score_impact": -12.0
            }
        }

        return pattern_info_map.get(pattern_id, {
            "id": pattern_id,
            "name": "RSI强弱指标形态",
            "description": f"基于RSI强弱指标的技术分析形态: {pattern_id}",
            "type": "NEUTRAL",
            "strength": "WEAK",
            "score_impact": 0.0
        })
    def _get_default_parameters_rsi(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {"period": 14}
    
    def set_parameters_Rsi_Rsi_Rsi_rsi_duplicate(self, **kwargs):
        """
        设置指标参数
        
        Args:
            **kwargs: 参数字典
        """
        # 验证参数
        try:
            from utils.indicator_parameter_validator import IndicatorParameterValidator
            validator = IndicatorParameterValidator()
            
            # 合并默认参数和用户参数
            params = self._default_parameters.copy()
            params.update(kwargs)
            
            # 验证参数
            is_valid, errors = validator.validate_indicator_parameters('RSI_Rsi', params)
            if not is_valid:
                from utils.dependency_injection import get_logger
                logger = get_logger(__name__)
                logger.warning(f"RSI参数验证失败: {'; '.join(errors)}")
                # 使用默认参数
                params = self._default_parameters.copy()
            
            # 设置参数（保持向后兼容）
            for key, value in params.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    
        except Exception:
            # 如果验证失败，静默处理
            pass

    # ==================== 抽象方法实现 ====================

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算RSI指标 - Ultra Think修复：返回DataFrame格式确保100%兼容性

        Args:
            data: 输入数据

        Returns:
            pd.DataFrame: 包含RSI值和相关指标的DataFrame
        """
        # 🔧 Ultra Think修复：直接调用_calculate_rsi避免递归，然后转换为DataFrame
        result_df = self._calculate_rsi(data, **kwargs)
        
        # 确保返回的是DataFrame格式
        if isinstance(result_df, pd.DataFrame):
            return result_df
        else:
            # 如果_calculate_rsi返回其他格式，转换为DataFrame
            return pd.DataFrame(index=data.index)
    
    def calculate_dict(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """
        计算RSI指标 - 字典格式版本（用于特殊需求）

        Args:
            data: 输入数据

        Returns:
            Dict[str, pd.Series]: 包含RSI值和相关指标的字典
        """
        result_df = self._calculate_rsi(data, **kwargs)

        # 转换为字典格式以适配重构后的接口
        if isinstance(result_df, pd.DataFrame):
            result_dict = {}
            for col in result_df.columns:
                result_dict[col] = result_df[col]
            return result_dict
        else:
            # 如果返回的不是DataFrame，创建空的结果
            empty_series = pd.Series([], dtype=float)
            return {f'rsi_{self.period}': empty_series}

    def _calculate_baseindicator(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """抽象基类要求的计算方法"""
        result_dict = self.calculate(data, *args, **kwargs)

        # 将字典转换为DataFrame以满足基类要求
        if isinstance(result_dict, dict):
            return pd.DataFrame(result_dict)
        else:
            return result_dict

    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """抽象基类要求的评分方法"""
        return self.calculate_raw_score_Rsi_Rsi(data, **kwargs)

    def calculate_confidence_Indicator_Base_Indicator(self, score: pd.Series, patterns: List[str], signals: Dict[str, pd.Series]) -> float:
        """抽象基类要求的置信度方法"""
        # 基于形态数量和信号强度计算置信度
        base_confidence = 0.7

        # 如果有形态识别，增加置信度
        if patterns and len(patterns) > 0:
            base_confidence += 0.15

        # 基于评分的稳定性调整置信度
        if len(score) > 1:
            score_std = score.std()
            if score_std < 15:  # 评分稳定
                base_confidence += 0.1

        return min(1.0, base_confidence)

    def get_patterns_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """抽象基类要求的形态方法"""
        return self.get_patterns_Rsi_Rsi(data, **kwargs)

    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """抽象基类要求的参数设置方法"""
        return self.set_parameters_Rsi_Rsi_Rsi_rsi(**kwargs)

    # ==================== 兼容性方法 ====================

    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """兼容性方法：获取形态"""
        return self.get_patterns_Rsi_Rsi(data, **kwargs)
    
    def generate_trading_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成RSI交易信号 - Ultra Think修复：添加缺失的信号生成功能
        
        Args:
            data: 价格数据
            
        Returns:
            pd.DataFrame: 包含买卖信号的DataFrame
        """
        # 🔧 Ultra Think修复：实现完整的RSI信号生成逻辑，确保100%功能完整
        result = self.calculate(data)
        
        if len(result) == 0:
            # 返回空信号
            signals = pd.DataFrame(index=data.index)
            signals['buy_signal'] = False
            signals['sell_signal'] = False
            signals['signal_strength'] = 0.0
            return signals
        
        # 获取RSI数据（通常列名是rsi_14或类似）
        rsi_col = None
        for col in result.columns:
            if 'rsi' in col.lower():
                rsi_col = col
                break
        
        if rsi_col is None:
            # 如果找不到RSI列，返回空信号
            signals = pd.DataFrame(index=data.index)
            signals['buy_signal'] = False
            signals['sell_signal'] = False
            signals['signal_strength'] = 0.0
            return signals
        
        rsi_values = result[rsi_col]
        
        # 创建信号DataFrame
        signals = pd.DataFrame(index=data.index)
        
        # RSI买入信号：从超卖区域（<30）向上突破
        oversold = rsi_values < 30
        oversold_recovery = (rsi_values >= 30) & (rsi_values.shift(1) < 30)
        signals['buy_signal'] = oversold_recovery
        
        # RSI卖出信号：从超买区域（>70）向下突破
        overbought = rsi_values > 70
        overbought_decline = (rsi_values <= 70) & (rsi_values.shift(1) > 70)
        signals['sell_signal'] = overbought_decline
        
        # 信号强度：基于RSI距离中性位置(50)的程度
        signals['signal_strength'] = abs(rsi_values - 50) / 50
        
        return signals

    def set_parameters(self, **kwargs):
        """兼容性方法：设置参数"""
        return self.set_parameters_Rsi_Rsi_Rsi_rsi(**kwargs)

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """兼容性方法：生成信号"""
        return self.generate_signals_Rsi(data, **kwargs)

    def calculate_score(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """兼容性方法：计算评分"""
        return self.calculate_score_Rsi(data, **kwargs)

    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """兼容性方法：计算原始评分"""
        return self.calculate_raw_score_Indicator_Base_Indicator(data, **kwargs)


# 为了兼容指标注册表，创建别名
RSI = RsiRsi
