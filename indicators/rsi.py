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
from utils.dependency_injection import get_logger

logger = get_logger(__name__)


class RsiRsi(BaseIndicator, PatternSignalMixin):
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
        
        # 计算上涨和下跌
        gain = delta.where(delta > 0, 0).ewm(span=self.period, adjust=False).mean()
        loss = -delta.where(delta < 0, 0).ewm(span=self.period, adjust=False).mean()

        # 计算相对强度
        rs = gain / loss.replace(0, 1e-9)
        
        # 计算RSI
        result_df[f'rsi_{self.period}'] = 100 - (100 / (1 + rs))
        
        # 可选：计算RSI均线
        if self.ma_periods and len(self.ma_periods) >= 2:
            result_df[f'rsi_ma_{self.ma_periods[0]}'] = result_df[f'rsi_{self.period}'].rolling(window=self.ma_periods[0]).mean()
            result_df[f'rsi_ma_{self.ma_periods[1]}'] = result_df[f'rsi_{self.period}'].rolling(window=self.ma_periods[1]).mean()
            # For pattern detection
            result_df['rsi_ma_short'] = result_df[f'rsi_ma_{self.ma_periods[0]}']
            result_df['rsi_ma_long'] = result_df[f'rsi_ma_{self.ma_periods[1]}']

        result_df['rsi_overbought'] = result_df[f'rsi_{self.period}'] > self.overbought
        result_df['rsi_oversold'] = result_df[f'rsi_{self.period}'] < self.oversold

        # 添加形态识别和信号生成
        result_df = self.add_pattern_detection(result_df)
        result_df = self.add_signal_generation(result_df)

        return result_df

    def get_patterns_Rsi_Rsi(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取RSI相关形态
        """
        calculated_data = self._calculate_rsi(data)
        patterns_df = pd.DataFrame(index=data.index)

        # 确保列存在
        if f'rsi_{self.period}' not in calculated_data.columns or 'rsi_ma_short' not in calculated_data.columns or 'rsi_ma_long' not in calculated_data.columns:
            return patterns_df

        # 金叉和死叉
        from utils.indicator_utils import crossover, crossunder
        patterns_df['RSI_GOLDEN_CROSS'] = crossover(calculated_data['rsi_ma_short'], calculated_data['rsi_ma_long'])
        patterns_df['RSI_DEATH_CROSS'] = crossunder(calculated_data['rsi_ma_short'], calculated_data['rsi_ma_long'])

        # 超买和超卖
        patterns_df['RSI_OVERBOUGHT'] = calculated_data[f'rsi_{self.period}'] > self.overbought
        patterns_df['RSI_OVERSOLD'] = calculated_data[f'rsi_{self.period}'] < self.oversold

        # 确保所有列都是布尔类型，填充NaN为False
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
        计算RSI指标 - 符合BaseIndicator标准

        Args:
            data: 输入数据

        Returns:
            pd.DataFrame: 包含RSI值和相关指标的DataFrame，符合项目级标准列名
        """
        result_df = self._calculate_rsi(data, **kwargs)

        # 确保返回DataFrame格式
        if not isinstance(result_df, pd.DataFrame):
            # 如果返回的不是DataFrame，创建空的结果
            empty_series = pd.Series([], dtype=float, index=data.index)
            result_df = pd.DataFrame({f'rsi_{self.period}': empty_series})

        # 添加项目级标准列名（符合StandardColumnNames）
        if f'rsi_{self.period}' in result_df.columns:
            # 主RSI值使用标准列名
            result_df['rsi_value'] = result_df[f'rsi_{self.period}']

        # 确保包含标准的超买超卖列
        if 'rsi_overbought' not in result_df.columns:
            result_df['rsi_overbought'] = result_df.get(f'rsi_{self.period}', 0) > self.overbought
        if 'rsi_oversold' not in result_df.columns:
            result_df['rsi_oversold'] = result_df.get(f'rsi_{self.period}', 0) < self.oversold

        return result_df

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

    def set_parameters(self, **kwargs):
        """兼容性方法：设置参数"""
        return self.set_parameters_Rsi_Rsi_Rsi_rsi(**kwargs)

    def get_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        标准方法：生成RSI交易信号DataFrame

        返回符合BaseIndicator标准的信号DataFrame，包含：
        - buy_signal: bool, 买入信号
        - sell_signal: bool, 卖出信号
        - hold_signal: bool, 持有信号
        - signal_strength: float, 信号强度(0-1)
        - signal_confidence: float, 信号置信度(0-1)
        """
        # 获取基础信号
        base_signals = self.generate_signals_Rsi(data, **kwargs)

        # 创建标准格式的信号DataFrame
        signals_df = pd.DataFrame(index=data.index)
        signals_df['buy_signal'] = base_signals.get('buy_signal', False)
        signals_df['sell_signal'] = base_signals.get('sell_signal', False)
        signals_df['hold_signal'] = ~(signals_df['buy_signal'] | signals_df['sell_signal'])

        # 计算信号强度（基于RSI值与阈值的距离）
        try:
            rsi_data = self._calculate_rsi(data)
            if rsi_data is not None and f'rsi_{self.period}' in rsi_data.columns:
                rsi_values = rsi_data[f'rsi_{self.period}']

                # 计算信号强度
                signals_df['signal_strength'] = 0.0

                # 买入信号强度：RSI越接近超卖区域，强度越高
                buy_mask = signals_df['buy_signal']
                if buy_mask.any():
                    oversold_distance = (self.oversold - rsi_values[buy_mask]).clip(0, self.oversold)
                    signals_df.loc[buy_mask, 'signal_strength'] = (oversold_distance / self.oversold).fillna(0.0)

                # 卖出信号强度：RSI越接近超买区域，强度越高
                sell_mask = signals_df['sell_signal']
                if sell_mask.any():
                    overbought_distance = (rsi_values[sell_mask] - self.overbought).clip(0, 100 - self.overbought)
                    signals_df.loc[sell_mask, 'signal_strength'] = (overbought_distance / (100 - self.overbought)).fillna(0.0)
            else:
                signals_df['signal_strength'] = 0.0
        except Exception:
            signals_df['signal_strength'] = 0.0

        # 设置信号置信度（基于RSI指标的可靠性）
        signals_df['signal_confidence'] = 0.0
        signals_df.loc[signals_df['buy_signal'] | signals_df['sell_signal'], 'signal_confidence'] = 0.8  # RSI信号置信度设为0.8

        return signals_df

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """兼容性方法：生成信号"""
        return self.generate_signals_Rsi(data, **kwargs)

    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        BaseIndicator要求的抽象方法：获取交易信号

        Args:
            data: 包含指标计算结果的数据

        Returns:
            Dict[str, Any]: 交易信号信息
        """
        try:
            # 生成信号
            signals_df = self.generate_signals_Rsi(data)

            if signals_df.empty:
                return {
                    "signal": "HOLD",
                    "strength": 0.0,
                    "confidence": 0.5,
                    "details": "无足够数据生成信号"
                }

            # 获取最新信号
            latest_signals = signals_df.iloc[-1]

            # 确定主要信号
            if latest_signals.get('buy_signal', False):
                signal_type = "BUY"
                strength = 0.8
            elif latest_signals.get('sell_signal', False):
                signal_type = "SELL"
                strength = 0.8
            else:
                signal_type = "HOLD"
                strength = 0.0

            # 计算置信度
            patterns = self.get_patterns_Rsi_Rsi(data)
            confidence = self.calculate_confidence_Rsi_Rsi(
                pd.Series([50.0]), patterns, signals_df.to_dict('series')
            )

            return {
                "signal": signal_type,
                "strength": strength,
                "confidence": confidence,
                "details": f"RSI信号基于{len(data)}个数据点"
            }

        except Exception as e:
            logger.error(f"RSI信号生成失败: {e}")
            return {
                "signal": "HOLD",
                "strength": 0.0,
                "confidence": 0.0,
                "details": f"信号生成错误: {e}"
            }

    def calculate_score(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """兼容性方法：计算评分"""
        return self.calculate_score_Rsi(data, **kwargs)

    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """兼容性方法：计算原始评分"""
        return self.calculate_raw_score_Indicator_Base_Indicator(data, **kwargs)


# 为了兼容指标注册表，创建别名
RSI = RsiRsi
