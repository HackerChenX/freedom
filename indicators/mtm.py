#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
动量指标(MTM)

反映股价波动的速度，通过计算股价与前一段时间的股价差值衡量价格动量
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any, Union, Tuple

from indicators.base_indicator import BaseIndicator, PatternResult
from utils.logger import get_logger

logger = get_logger(__name__)


class MTM(BaseIndicator):
    """
    动量指标(MTM)
    
    分类：震荡类指标
    计算方法：MTM = CLOSE - REF(CLOSE, N)
    参数：N，一般取10或12，表示计算周期
    """
    
    def __init__(self, period: int = 10, ma_period: int = 6, overbought: float = 0, oversold: float = 0):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """
        初始化MTM指标
        
        Args:
            period: 计算周期，默认为10
            ma_period: MTM平滑周期，默认为6
            overbought: 超买线，默认根据历史数据自动计算
            oversold: 超卖线，默认根据历史数据自动计算
        """
        super().__init__()
        self.name = "MTM"
        self.period = period
        self.ma_period = ma_period
        self.overbought = overbought
        self.oversold = oversold
        self._auto_threshold = (overbought == 0 and oversold == 0)
    
    def set_parameters(self, period: int = None, ma_period: int = None, overbought: float = None, oversold: float = None):
        """
        设置指标参数
        """
        if period is not None:
            self.period = period
        if ma_period is not None:
            self.ma_period = ma_period
        if overbought is not None:
            self.overbought = overbought
        if oversold is not None:
            self.oversold = oversold
        # 如果超买或超卖被手动设置，则禁用自动阈值计算
        if overbought is not None or oversold is not None:
            self._auto_threshold = False
    
    def _calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算MTM指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
                
        Returns:
            包含MTM指标的DataFrame
        """
        if df.empty:
            return df
            
        df_copy = df.copy()
        
        # 计算MTM
        df_copy['mtm'] = df_copy['close'] - df_copy['close'].shift(self.period)
        
        # 计算MTMMA
        df_copy['mtmma'] = df_copy['mtm'].rolling(window=self.ma_period).mean()
        
        # 如果需要自动计算超买超卖线
        if self._auto_threshold:
            # 使用历史数据的标准差来设置超买超卖线
            mtm_std = df_copy['mtm'].std()
            self.overbought = 2 * mtm_std
            self.oversold = -2 * mtm_std
        
        # 添加超买超卖状态
        df_copy['mtm_overbought'] = df_copy['mtm'] > self.overbought
        df_copy['mtm_oversold'] = df_copy['mtm'] < self.oversold
        
        # 存储结果
        self._result = df_copy
        
        return df_copy
    
    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取MTM相关形态

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含形态信息的DataFrame
        """
        # 确保已计算指标
        if not self.has_result():
            self.calculate(data, **kwargs)

        if self._result is None or 'mtm' not in self._result.columns:
            return pd.DataFrame(index=data.index)

        # 获取MTM和MTMMA值
        mtm = self._result['mtm']
        mtmma = self._result['mtmma']

        # 创建形态DataFrame
        patterns_df = pd.DataFrame(index=data.index)

        # 1. MTM超买超卖形态
        patterns_df['MTM_OVERBOUGHT'] = mtm > self.overbought
        patterns_df['MTM_OVERSOLD'] = mtm < self.oversold

        # 2. MTM金叉死叉形态
        # 需要导入crossover和crossunder函数
        from utils.indicator_utils import crossover, crossunder

        patterns_df['MTM_GOLDEN_CROSS'] = crossover(mtm, mtmma)
        patterns_df['MTM_DEATH_CROSS'] = crossunder(mtm, mtmma)

        # 3. MTM零轴穿越形态
        patterns_df['MTM_CROSS_UP_ZERO'] = crossover(mtm, 0)
        patterns_df['MTM_CROSS_DOWN_ZERO'] = crossunder(mtm, 0)

        # 4. MTM趋势形态
        patterns_df['MTM_ABOVE_ZERO'] = mtm > 0
        patterns_df['MTM_BELOW_ZERO'] = mtm < 0
        patterns_df['MTM_ABOVE_MA'] = mtm > mtmma
        patterns_df['MTM_BELOW_MA'] = mtm < mtmma

        # 5. MTM强势形态（基于阈值的倍数）
        if len(mtm) >= 5:
            mtm_std = mtm.rolling(window=20, min_periods=5).std()
            patterns_df['MTM_STRONG_UP'] = mtm > (mtm_std * 1.5)
            patterns_df['MTM_STRONG_DOWN'] = mtm < -(mtm_std * 1.5)
        else:
            patterns_df['MTM_STRONG_UP'] = False
            patterns_df['MTM_STRONG_DOWN'] = False

        return patterns_df
    
    def _calculate_cross_angle(self, y1_prev, y1_curr, y2_prev, y2_curr):
        """计算两条线交叉时的角度"""
        # 计算两条线的斜率
        k1 = y1_curr - y1_prev
        k2 = y2_curr - y2_prev
        
        # 计算角度（弧度）
        angle_rad = np.arctan(abs((k1 - k2) / (1 + k1 * k2))) if (1 + k1 * k2) != 0 else np.pi/2
        
        # 转换为角度
        angle_deg = angle_rad * 180 / np.pi
        
        return angle_deg
    
    def _calculate_divergence_strength(self, current_price, previous_price, current_mtm, previous_mtm):
        """计算背离强度"""
        # 计算价格变化百分比
        price_change = abs(current_price - previous_price) / previous_price
        
        # 计算MTM变化百分比，避免除以零
        mtm_denominator = abs(previous_mtm) if previous_mtm != 0 else 1e-6
        mtm_change = abs(current_mtm - previous_mtm) / mtm_denominator
        
        # 计算背离强度：价格变化和MTM变化的不一致程度
        # 背离越明显，强度越大
        return min(price_change / max(mtm_change, 1e-6), 1.0)
    
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
        signals['signal_strength'] = pd.Series(0.0, index=data.index, dtype=float)
    
        # 如果没有结果，返回空信号
        if self._result is None or 'mtm' not in self._result.columns:
            return signals
        
        # 获取MTM和MTMMA值
        mtm = self._result['mtm']
        mtmma = self._result['mtmma']
        
        # 生成金叉买入信号
        for i in range(1, len(mtm)):
            # MTM上穿MTMMA，金叉信号
            if mtm.iloc[i-1] <= mtmma.iloc[i-1] and mtm.iloc[i] > mtmma.iloc[i]:
                signals['buy_signal'].iloc[i] = True
                # 计算信号强度：基于交叉角度
                angle = self._calculate_cross_angle(mtm.iloc[i-1], mtm.iloc[i], mtmma.iloc[i-1], mtmma.iloc[i])
                signals['signal_strength'].iloc[i] = float(50 + min(angle / 90 * 40, 40))  # 50-90
        
        # 生成死叉卖出信号
        for i in range(1, len(mtm)):
            # MTM下穿MTMMA，死叉信号
            if mtm.iloc[i-1] >= mtmma.iloc[i-1] and mtm.iloc[i] < mtmma.iloc[i]:
                signals['sell_signal'].iloc[i] = True
                # 计算信号强度：基于交叉角度
                angle = self._calculate_cross_angle(mtm.iloc[i-1], mtm.iloc[i], mtmma.iloc[i-1], mtmma.iloc[i])
                signals['signal_strength'].iloc[i] = float(50 + min(angle / 90 * 40, 40))  # 50-90
        
        # 生成超卖买入信号
        for i in range(1, len(mtm)):
            # MTM从超卖区域上穿，生成买入信号
            if mtm.iloc[i-1] <= self.oversold and mtm.iloc[i] > self.oversold:
                signals['buy_signal'].iloc[i] = True
                # 信号强度基于超卖程度
                signals['signal_strength'].iloc[i] = float(60 + min((self.oversold - mtm.iloc[i-1]) / abs(self.oversold) * 30, 30))
        
        # 生成超买卖出信号
        for i in range(1, len(mtm)):
            # MTM从超买区域下穿，生成卖出信号
            if mtm.iloc[i-1] >= self.overbought and mtm.iloc[i] < self.overbought:
                signals['sell_signal'].iloc[i] = True
                # 信号强度基于超买程度
                signals['signal_strength'].iloc[i] = float(60 + min((mtm.iloc[i-1] - self.overbought) / abs(self.overbought) * 30, 30))
        
        # 基于MTM形态生成信号
        patterns_df = self.get_patterns(data, **kwargs)

        # 基于金叉形态增强买入信号
        if 'MTM_GOLDEN_CROSS' in patterns_df.columns:
            golden_cross_signals = patterns_df['MTM_GOLDEN_CROSS']
            for i, signal in enumerate(golden_cross_signals):
                if signal and i < len(signals['buy_signal']):
                    signals['buy_signal'].iloc[i] = True
                    if signals['signal_strength'].iloc[i] < 70:
                        signals['signal_strength'].iloc[i] = float(70)

        # 基于死叉形态增强卖出信号
        if 'MTM_DEATH_CROSS' in patterns_df.columns:
            death_cross_signals = patterns_df['MTM_DEATH_CROSS']
            for i, signal in enumerate(death_cross_signals):
                if signal and i < len(signals['sell_signal']):
                    signals['sell_signal'].iloc[i] = True
                    if signals['signal_strength'].iloc[i] < 70:
                        signals['signal_strength'].iloc[i] = float(70)
    
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
            self.calculate(data, **kwargs)
        
        if self._result is None or 'mtm' not in self._result.columns:
            return pd.Series(50.0, index=data.index)
        
        # 初始化评分
        score = pd.Series(50.0, index=data.index)
        
        # 获取MTM和MTMMA值
        mtm = self._result['mtm']
        mtmma = self._result['mtmma']
        
        # 根据MTM值和相对位置计算评分
        for i in range(len(mtm)):
            if mtm.iloc[i] < self.oversold:
                # MTM在超卖区，得分增加
                ratio = min((self.oversold - mtm.iloc[i]) / abs(self.oversold), 1.0) if self.oversold != 0 else 0.5
                score.iloc[i] = 50 + ratio * 30  # 50-80
            elif mtm.iloc[i] > self.overbought:
                # MTM在超买区，得分降低
                ratio = min((mtm.iloc[i] - self.overbought) / abs(self.overbought), 1.0) if self.overbought != 0 else 0.5
                score.iloc[i] = 50 - ratio * 30  # 20-50
            else:
                # MTM在中间区域，基于MTM和MTMMA的关系
                if mtmma.iloc[i] != 0:
                    ratio = min(abs(mtm.iloc[i] - mtmma.iloc[i]) / abs(mtmma.iloc[i]), 1.0)
                else:
                    ratio = min(abs(mtm.iloc[i] - mtmma.iloc[i]), 1.0)
                
                if mtm.iloc[i] > mtmma.iloc[i]:
                    # MTM大于MTMMA，动量向上，评分增加
                    score.iloc[i] = 50 + ratio * 20  # 50-70
                else:
                    # MTM小于MTMMA，动量向下，评分降低
                    score.iloc[i] = 50 - ratio * 20  # 30-50
        
        # 考虑金叉和死叉的影响
        for i in range(1, len(mtm)):
            # 金叉提高评分
            if mtm.iloc[i-1] <= mtmma.iloc[i-1] and mtm.iloc[i] > mtmma.iloc[i]:
                angle = self._calculate_cross_angle(mtm.iloc[i-1], mtm.iloc[i], mtmma.iloc[i-1], mtmma.iloc[i])
                adjust = min(angle / 90 * 15, 15)  # 最多增加15分
                score.iloc[i] = min(score.iloc[i] + adjust, 90)
            
            # 死叉降低评分
            if mtm.iloc[i-1] >= mtmma.iloc[i-1] and mtm.iloc[i] < mtmma.iloc[i]:
                angle = self._calculate_cross_angle(mtm.iloc[i-1], mtm.iloc[i], mtmma.iloc[i-1], mtmma.iloc[i])
                adjust = min(angle / 90 * 15, 15)  # 最多减少15分
                score.iloc[i] = max(score.iloc[i] - adjust, 10)
        
        # 结合形态增强评分
        patterns_df = self.get_patterns(data, **kwargs)

        # 基于强势形态调整评分
        if 'MTM_STRONG_UP' in patterns_df.columns:
            strong_up_mask = patterns_df['MTM_STRONG_UP']
            score.loc[strong_up_mask] = score.loc[strong_up_mask].apply(lambda x: min(x + 10, 90))

        if 'MTM_STRONG_DOWN' in patterns_df.columns:
            strong_down_mask = patterns_df['MTM_STRONG_DOWN']
            score.loc[strong_down_mask] = score.loc[strong_down_mask].apply(lambda x: max(x - 10, 10))
        
        return score

    def calculate_confidence(self, score: pd.Series, patterns: pd.DataFrame, signals: Dict[str, pd.Series]) -> float:
        """
        计算MTM指标的置信度

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
            # 检查强势形态
            if 'MTM_STRONG_UP' in patterns.columns and patterns['MTM_STRONG_UP'].any():
                confidence += 0.15
            if 'MTM_STRONG_DOWN' in patterns.columns and patterns['MTM_STRONG_DOWN'].any():
                confidence += 0.15

            # 检查金叉死叉形态
            if 'MTM_GOLDEN_CROSS' in patterns.columns and patterns['MTM_GOLDEN_CROSS'].any():
                confidence += 0.1
            if 'MTM_DEATH_CROSS' in patterns.columns and patterns['MTM_DEATH_CROSS'].any():
                confidence += 0.1

        # 3. 基于信号的置信度
        if signals:
            # 检查信号强度
            signal_strength = signals.get('signal_strength', pd.Series())
            if not signal_strength.empty:
                avg_strength = signal_strength.mean()
                if avg_strength > 70:
                    confidence += 0.1

        # 4. 基于评分趋势的置信度
        if len(score) >= 3:
            recent_scores = score.iloc[-3:]
            trend = recent_scores.iloc[-1] - recent_scores.iloc[0]

            # 明确的趋势增加置信度
            if abs(trend) > 10:
                confidence += 0.05

        # 确保置信度在0-1范围内
        return max(0.0, min(1.0, confidence))

    def register_patterns(self):
        """
        注册MTM指标的形态到全局形态注册表
        """
        # 注册MTM超买形态
        self.register_pattern_to_registry(
            pattern_id="MTM_OVERBOUGHT",
            display_name="MTM超买",
            description="MTM指标进入超买区域，可能暗示价格上涨过快",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-10.0
        )

        # 注册MTM超卖形态
        self.register_pattern_to_registry(
            pattern_id="MTM_OVERSOLD",
            display_name="MTM超卖",
            description="MTM指标进入超卖区域，可能暗示价格下跌过快",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=10.0
        )

        # 注册MTM金叉形态
        self.register_pattern_to_registry(
            pattern_id="MTM_GOLDEN_CROSS",
            display_name="MTM金叉",
            description="MTM上穿MTMMA，动量由负转正，可能是买入信号",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=15.0
        )

        # 注册MTM死叉形态
        self.register_pattern_to_registry(
            pattern_id="MTM_DEATH_CROSS",
            display_name="MTM死叉",
            description="MTM下穿MTMMA，动量由正转负，可能是卖出信号",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-15.0
        )

        # 注册MTM顶背离形态
        self.register_pattern_to_registry(
            pattern_id="MTM_TOP_DIVERGENCE",
            display_name="MTM顶背离",
            description="价格创新高但MTM未同步创新高，可能暗示上涨动能减弱",
            pattern_type="BEARISH",
            default_strength="VERY_STRONG",
            score_impact=-20.0
        )

        # 注册MTM底背离形态
        self.register_pattern_to_registry(
            pattern_id="MTM_BOTTOM_DIVERGENCE",
            display_name="MTM底背离",
            description="价格创新低但MTM未同步创新低，可能暗示下跌动能减弱",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=20.0
        )

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