#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
变动率指标(ROC)

通过计算当前收盘价与N个周期前收盘价的变化比率来测量价格变化速度
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any, Union, Tuple

from indicators.base_indicator import BaseIndicator, PatternResult
from utils.logger import get_logger

logger = get_logger(__name__)


class ROC(BaseIndicator):
    """
    变动率指标(ROC)
    
    分类：震荡类指标
    计算方法：ROC = (CLOSE - REF(CLOSE, N)) / REF(CLOSE, N) * 100
    参数：N，一般取12，表示计算周期
    """
    
    def __init__(self, period: int = 12, ma_period: int = 6, overbought: float = 8.0, oversold: float = -8.0):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """
        初始化ROC指标
        
        Args:
            period: 计算周期，默认为12
            ma_period: ROC平滑周期，默认为6
            overbought: 超买线，默认为8.0
            oversold: 超卖线，默认为-8.0
        """
        super().__init__()
        self.name = "ROC"
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
        计算ROC指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
                
        Returns:
            包含ROC指标的DataFrame
        """
        if df.empty:
            return df
            
        df_copy = df.copy()
        
        # 计算ROC
        df_copy['roc'] = (df_copy['close'] - df_copy['close'].shift(self.period)) / df_copy['close'].shift(self.period) * 100
        
        # 计算ROCMA
        df_copy['rocma'] = df_copy['roc'].rolling(window=self.ma_period).mean()
        
        # 如果需要自动计算超买超卖线
        if self._auto_threshold:
            # 使用历史数据的标准差来设置超买超卖线
            roc_std = df_copy['roc'].std()
            self.overbought = 2 * roc_std
            self.oversold = -2 * roc_std
        
        # 添加超买超卖状态
        df_copy['roc_overbought'] = df_copy['roc'] > self.overbought
        df_copy['roc_oversold'] = df_copy['roc'] < self.oversold
        
        # 存储结果
        self._result = df_copy
        
        return df_copy

    def calculate_confidence(self, score: pd.Series, patterns: pd.DataFrame, signals: Dict[str, pd.Series]) -> float:
        """
        计算ROC指标的置信度

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
            # 检查超买超卖形态
            if 'ROC_OVERBOUGHT' in patterns.columns and patterns['ROC_OVERBOUGHT'].any():
                confidence += 0.1
            if 'ROC_OVERSOLD' in patterns.columns and patterns['ROC_OVERSOLD'].any():
                confidence += 0.1

            # 检查金叉死叉形态
            if 'ROC_GOLDEN_CROSS' in patterns.columns and patterns['ROC_GOLDEN_CROSS'].any():
                confidence += 0.15
            if 'ROC_DEATH_CROSS' in patterns.columns and patterns['ROC_DEATH_CROSS'].any():
                confidence += 0.15

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
        注册ROC指标的形态到全局形态注册表
        """
        # 注册ROC超买形态
        self.register_pattern_to_registry(
            pattern_id="ROC_OVERBOUGHT",
            display_name="ROC超买",
            description="ROC指标进入超买区域，价格变动率过高",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-10.0,
            polarity="NEGATIVE"
        )

        # 注册ROC超卖形态
        self.register_pattern_to_registry(
            pattern_id="ROC_OVERSOLD",
            display_name="ROC超卖",
            description="ROC指标进入超卖区域，价格变动率过低",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=10.0,
            polarity="POSITIVE"
        )

        # 注册ROC金叉形态
        self.register_pattern_to_registry(
            pattern_id="ROC_GOLDEN_CROSS",
            display_name="ROC金叉",
            description="ROC上穿ROCMA，动量由负转正，可能是买入信号",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=15.0,
            polarity="POSITIVE"
        )

        # 注册ROC死叉形态
        self.register_pattern_to_registry(
            pattern_id="ROC_DEATH_CROSS",
            display_name="ROC死叉",
            description="ROC下穿ROCMA，动量由正转负，可能是卖出信号",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-15.0,
            polarity="NEGATIVE"
        )

        # 注册ROC顶背离形态
        self.register_pattern_to_registry(
            pattern_id="ROC_TOP_DIVERGENCE",
            display_name="ROC顶背离",
            description="价格创新高但ROC未同步创新高，可能暗示上涨动能减弱",
            pattern_type="BEARISH",
            default_strength="VERY_STRONG",
            score_impact=-20.0,
            polarity="NEGATIVE"
        )

        # 注册ROC底背离形态
        self.register_pattern_to_registry(
            pattern_id="ROC_BOTTOM_DIVERGENCE",
            display_name="ROC底背离",
            description="价格创新低但ROC未同步创新低，可能暗示下跌动能减弱",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=20.0,
            polarity="POSITIVE"
        )

        # 注册ROC零轴穿越形态
        self.register_pattern_to_registry(
            pattern_id="ROC_CROSS_UP_ZERO",
            display_name="ROC上穿零轴",
            description="ROC上穿零轴，价格变动率由负转正",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=8.0,
            polarity="POSITIVE"
        )

        # 注册ROC零轴穿越形态
        self.register_pattern_to_registry(
            pattern_id="ROC_CROSS_DOWN_ZERO",
            display_name="ROC下穿零轴",
            description="ROC下穿零轴，价格变动率由正转负",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-8.0,
            polarity="NEGATIVE"
        )

    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取ROC相关形态

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含形态信息的DataFrame
        """
        # 确保已计算指标
        if not self.has_result():
            self.calculate(data, **kwargs)

        if self._result is None or 'roc' not in self._result.columns:
            return pd.DataFrame(index=data.index)

        # 获取ROC和ROCMA值
        roc = self._result['roc']
        rocma = self._result['rocma']

        # 创建形态DataFrame
        patterns_df = pd.DataFrame(index=data.index)

        # 1. ROC超买超卖形态
        patterns_df['ROC_OVERBOUGHT'] = roc > self.overbought
        patterns_df['ROC_OVERSOLD'] = roc < self.oversold

        # 2. ROC金叉死叉形态
        from utils.indicator_utils import crossover, crossunder

        patterns_df['ROC_GOLDEN_CROSS'] = crossover(roc, rocma)
        patterns_df['ROC_DEATH_CROSS'] = crossunder(roc, rocma)

        # 3. ROC零轴穿越形态
        patterns_df['ROC_CROSS_UP_ZERO'] = crossover(roc, 0)
        patterns_df['ROC_CROSS_DOWN_ZERO'] = crossunder(roc, 0)

        # 4. ROC趋势形态
        patterns_df['ROC_ABOVE_ZERO'] = roc > 0
        patterns_df['ROC_BELOW_ZERO'] = roc < 0
        patterns_df['ROC_ABOVE_MA'] = roc > rocma
        patterns_df['ROC_BELOW_MA'] = roc < rocma

        # 5. ROC强势形态
        if len(roc) >= 5:
            roc_momentum = roc - roc.shift(3)
            patterns_df['ROC_STRONG_UP'] = roc_momentum > (roc.std() * 1.5)
            patterns_df['ROC_STRONG_DOWN'] = roc_momentum < -(roc.std() * 1.5)
        else:
            patterns_df['ROC_STRONG_UP'] = False
            patterns_df['ROC_STRONG_DOWN'] = False

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
    
    def _calculate_divergence_strength(self, current_price, previous_price, current_roc, previous_roc):
        """计算背离强度"""
        # 计算价格变化百分比
        price_change = abs(current_price - previous_price) / previous_price
        
        # 计算ROC变化百分比，避免除以零
        roc_denominator = abs(previous_roc) if previous_roc != 0 else 1e-6
        roc_change = abs(current_roc - previous_roc) / roc_denominator
        
        # 计算背离强度：价格变化和ROC变化的不一致程度
        # 背离越明显，强度越大
        return min(price_change / max(roc_change, 1e-6), 1.0)
    
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
        if self._result is None or 'roc' not in self._result.columns:
            return signals
        
        # 获取ROC和ROCMA值
        roc = self._result['roc']
        rocma = self._result['rocma']
        
        # 生成金叉买入信号
        for i in range(1, len(roc)):
            # ROC上穿ROCMA，金叉信号
            if roc.iloc[i-1] <= rocma.iloc[i-1] and roc.iloc[i] > rocma.iloc[i]:
                signals['buy_signal'].iloc[i] = True
                # 计算信号强度：基于交叉角度
                angle = self._calculate_cross_angle(roc.iloc[i-1], roc.iloc[i], rocma.iloc[i-1], rocma.iloc[i])
                signals['signal_strength'].iloc[i] = float(50 + min(angle / 90 * 40, 40))  # 50-90
        
        # 生成死叉卖出信号
        for i in range(1, len(roc)):
            # ROC下穿ROCMA，死叉信号
            if roc.iloc[i-1] >= rocma.iloc[i-1] and roc.iloc[i] < rocma.iloc[i]:
                signals['sell_signal'].iloc[i] = True
                # 计算信号强度：基于交叉角度
                angle = self._calculate_cross_angle(roc.iloc[i-1], roc.iloc[i], rocma.iloc[i-1], rocma.iloc[i])
                signals['signal_strength'].iloc[i] = float(50 + min(angle / 90 * 40, 40))  # 50-90
        
        # 生成超卖买入信号
        for i in range(1, len(roc)):
            # ROC从超卖区域上穿，生成买入信号
            if roc.iloc[i-1] <= self.oversold and roc.iloc[i] > self.oversold:
                signals['buy_signal'].iloc[i] = True
                # 信号强度基于超卖程度
                signals['signal_strength'].iloc[i] = float(60 + min((self.oversold - roc.iloc[i-1]) / abs(self.oversold) * 30, 30))
        
        # 生成超买卖出信号
        for i in range(1, len(roc)):
            # ROC从超买区域下穿，生成卖出信号
            if roc.iloc[i-1] >= self.overbought and roc.iloc[i] < self.overbought:
                signals['sell_signal'].iloc[i] = True
                # 信号强度基于超买程度
                signals['signal_strength'].iloc[i] = float(60 + min((roc.iloc[i-1] - self.overbought) / abs(self.overbought) * 30, 30))
        
        # 基于ROC形态生成信号
        patterns_df = self.get_patterns(data, **kwargs)

        # 基于金叉形态增强买入信号
        if 'ROC_GOLDEN_CROSS' in patterns_df.columns:
            golden_cross_signals = patterns_df['ROC_GOLDEN_CROSS']
            for i, signal in enumerate(golden_cross_signals):
                if signal and i < len(signals['buy_signal']):
                    signals['buy_signal'].iloc[i] = True
                    if signals['signal_strength'].iloc[i] < 70:
                        signals['signal_strength'].iloc[i] = float(70)

        # 基于死叉形态增强卖出信号
        if 'ROC_DEATH_CROSS' in patterns_df.columns:
            death_cross_signals = patterns_df['ROC_DEATH_CROSS']
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
        
        if self._result is None or 'roc' not in self._result.columns:
            return pd.Series(50.0, index=data.index)
        
        # 初始化评分
        score = pd.Series(50.0, index=data.index)
        
        # 获取ROC和ROCMA值
        roc = self._result['roc']
        rocma = self._result['rocma']
        
        # 根据ROC值和相对位置计算评分
        for i in range(len(roc)):
            if roc.iloc[i] < self.oversold:
                # ROC在超卖区，得分增加
                ratio = min((self.oversold - roc.iloc[i]) / abs(self.oversold), 1.0) if self.oversold != 0 else 0.5
                score.iloc[i] = 50 + ratio * 30  # 50-80
            elif roc.iloc[i] > self.overbought:
                # ROC在超买区，得分降低
                ratio = min((roc.iloc[i] - self.overbought) / abs(self.overbought), 1.0) if self.overbought != 0 else 0.5
                score.iloc[i] = 50 - ratio * 30  # 20-50
            else:
                # ROC在中间区域，基于ROC和ROCMA的关系
                if rocma.iloc[i] != 0:
                    ratio = min(abs(roc.iloc[i] - rocma.iloc[i]) / abs(rocma.iloc[i]), 1.0)
                else:
                    ratio = min(abs(roc.iloc[i] - rocma.iloc[i]), 1.0)
                
                if roc.iloc[i] > rocma.iloc[i]:
                    # ROC大于ROCMA，动量向上，评分增加
                    score.iloc[i] = 50 + ratio * 20  # 50-70
                else:
                    # ROC小于ROCMA，动量向下，评分降低
                    score.iloc[i] = 50 - ratio * 20  # 30-50
        
        # 考虑金叉和死叉的影响
        for i in range(1, len(roc)):
            # 金叉提高评分
            if roc.iloc[i-1] <= rocma.iloc[i-1] and roc.iloc[i] > rocma.iloc[i]:
                angle = self._calculate_cross_angle(roc.iloc[i-1], roc.iloc[i], rocma.iloc[i-1], rocma.iloc[i])
                adjust = min(angle / 90 * 15, 15)  # 最多增加15分
                score.iloc[i] = min(score.iloc[i] + adjust, 90)
            
            # 死叉降低评分
            if roc.iloc[i-1] >= rocma.iloc[i-1] and roc.iloc[i] < rocma.iloc[i]:
                angle = self._calculate_cross_angle(roc.iloc[i-1], roc.iloc[i], rocma.iloc[i-1], rocma.iloc[i])
                adjust = min(angle / 90 * 15, 15)  # 最多减少15分
                score.iloc[i] = max(score.iloc[i] - adjust, 10)
        
        # 结合形态增强评分
        patterns_df = self.get_patterns(data, **kwargs)

        # 基于强势形态调整评分
        if 'ROC_STRONG_UP' in patterns_df.columns:
            strong_up_mask = patterns_df['ROC_STRONG_UP']
            score.loc[strong_up_mask] = score.loc[strong_up_mask].apply(lambda x: min(x + 10, 90))

        if 'ROC_STRONG_DOWN' in patterns_df.columns:
            strong_down_mask = patterns_df['ROC_STRONG_DOWN']
            score.loc[strong_down_mask] = score.loc[strong_down_mask].apply(lambda x: max(x - 10, 10))
        
        return score
    
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

    def get_pattern_info(self, pattern_id: str) -> dict:
        """
        获取指定形态的详细信息
        
        Args:
            pattern_id: 形态ID
            
        Returns:
            dict: 形态详细信息
        """
        # 默认形态信息
        default_pattern = {
            "id": pattern_id,
            "name": pattern_id,
            "description": f"{pattern_id}形态",
            "type": "NEUTRAL",
            "strength": "MEDIUM",
            "score_impact": 0.0
        }
        
        # ROC指标特定的形态信息映射
        pattern_info_map = {
            # 基础形态
            "超买区域": {
                "id": "超买区域",
                "name": "超买区域",
                "description": "指标进入超买区域，可能面临回调压力",
                "type": "BEARISH",
                "strength": "MEDIUM",
                "score_impact": -10.0
            },
            "超卖区域": {
                "id": "超卖区域", 
                "name": "超卖区域",
                "description": "指标进入超卖区域，可能出现反弹机会",
                "type": "BULLISH",
                "strength": "MEDIUM",
                "score_impact": 10.0
            },
            "中性区域": {
                "id": "中性区域",
                "name": "中性区域", 
                "description": "指标处于中性区域，趋势不明确",
                "type": "NEUTRAL",
                "strength": "WEAK",
                "score_impact": 0.0
            },
            # 趋势形态
            "上升趋势": {
                "id": "上升趋势",
                "name": "上升趋势",
                "description": "指标显示上升趋势，看涨信号",
                "type": "BULLISH", 
                "strength": "STRONG",
                "score_impact": 15.0
            },
            "下降趋势": {
                "id": "下降趋势",
                "name": "下降趋势",
                "description": "指标显示下降趋势，看跌信号",
                "type": "BEARISH",
                "strength": "STRONG", 
                "score_impact": -15.0
            },
            # 信号形态
            "买入信号": {
                "id": "买入信号",
                "name": "买入信号",
                "description": "指标产生买入信号，建议关注",
                "type": "BULLISH",
                "strength": "STRONG",
                "score_impact": 20.0
            },
            "卖出信号": {
                "id": "卖出信号", 
                "name": "卖出信号",
                "description": "指标产生卖出信号，建议谨慎",
                "type": "BEARISH",
                "strength": "STRONG",
                "score_impact": -20.0
            }
        }
        
        return pattern_info_map.get(pattern_id, default_pattern)