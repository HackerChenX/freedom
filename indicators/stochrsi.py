#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
随机相对强弱指数(StochRSI)

结合RSI和随机指标的特点，通过计算RSI的随机值反映市场的超买超卖状况
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any, Union, Tuple

from indicators.base_indicator import BaseIndicator, PatternResult
from utils.logger import get_logger

logger = get_logger(__name__)


class STOCHRSI(BaseIndicator):
    """
    随机相对强弱指数(StochRSI)
    
    分类：震荡类指标
    计算方法：
        RSI = 相对强弱指数
        StochRSI = (RSI - 最低RSI值) / (最高RSI值 - 最低RSI值)
    参数：
        RSI_period: RSI计算周期，默认为14
        K_period: 随机值K周期，默认为3
        D_period: 随机值D周期，默认为3
    """
    
    def __init__(self, rsi_period: int = 14, k_period: int = 3, d_period: int = 3, overbought: float = 80, oversold: float = 20, **kwargs):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """
        初始化StochRSI指标
        
        Args:
            rsi_period: RSI计算周期，默认为14
            k_period: 随机值K周期，默认为3
            d_period: 随机值D周期，默认为3
            overbought: 超买线，默认为80
            oversold: 超卖线，默认为20
        """
        super().__init__(name="STOCHRSI", description="随机相对强弱指数，结合RSI和随机指标特点")
        # self.name = "STOCHRSI"  # 已在super().__init__中设置
        self.rsi_period = rsi_period
        self.k_period = k_period
        self.d_period = d_period
        self.overbought = overbought
        self.oversold = oversold

    def set_parameters(self, **kwargs):
        """
        设置指标参数

        Args:
            **kwargs: 参数字典，可包含rsi_period, k_period, d_period, overbought, oversold
        """
        if 'rsi_period' in kwargs:
            self.rsi_period = kwargs['rsi_period']
        if 'k_period' in kwargs:
            self.k_period = kwargs['k_period']
        if 'd_period' in kwargs:
            self.d_period = kwargs['d_period']
        if 'overbought' in kwargs:
            self.overbought = kwargs['overbought']
        if 'oversold' in kwargs:
            self.oversold = kwargs['oversold']

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算StochRSI指标

        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 其他参数

        Returns:
            包含StochRSI指标的DataFrame
        """
        return self._calculate(data)

    def _calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算StochRSI指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
                
        Returns:
            包含StochRSI指标的DataFrame
        """
        if df.empty:
            return df
            
        df_copy = df.copy()
        
        # 计算RSI
        delta = df_copy['close'].diff().fillna(0)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period).mean()
        
        # 处理初始值
        avg_gain.fillna(gain.iloc[:self.rsi_period].mean(), inplace=True)
        avg_loss.fillna(loss.iloc[:self.rsi_period].mean(), inplace=True)
        
        # 计算RS和RSI
        rs = avg_gain / avg_loss.replace(0, 1e-9)  # 避免除以零
        rsi = 100 - (100 / (1 + rs))
        
        # 存储RSI值供后续使用
        df_copy['rsi'] = rsi
        
        # 计算StochRSI
        # 使用滚动窗口计算最高和最低RSI值
        rsi_min = rsi.rolling(window=self.k_period).min()
        rsi_max = rsi.rolling(window=self.k_period).max()
        
        # 计算StochRSI的K值
        # 避免除以零，当max == min时，k = 50
        divisor = rsi_max - rsi_min
        k = np.where(divisor == 0, 50, 100 * (rsi - rsi_min) / divisor)

        # 确保K值在0-100范围内
        k = np.clip(k, 0, 100)

        # 转换为pandas Series以便使用rolling方法
        k = pd.Series(k, index=df_copy.index)

        # 计算StochRSI的D值 (K的简单移动平均)
        d = k.rolling(window=self.d_period).mean()
        
        # 添加结果到DataFrame
        df_copy['stochrsi_k'] = k
        df_copy['stochrsi_d'] = d
        
        # 添加超买超卖状态
        df_copy['stochrsi_overbought'] = k > self.overbought
        df_copy['stochrsi_oversold'] = k < self.oversold
        
        # 存储结果
        self._result = df_copy
        
        return df_copy
    
    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取StochRSI相关形态

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含形态信息的DataFrame
        """
        # 确保已计算指标
        if self._result is None:
            self.calculate(data)

        if self._result is None or 'stochrsi_k' not in self._result.columns:
            return pd.DataFrame(index=data.index)
        
        # 获取K线和D线值
        k = self._result['stochrsi_k']
        d = self._result['stochrsi_d']

        # 创建形态DataFrame
        patterns_df = pd.DataFrame(index=data.index)

        # 1. StochRSI超买超卖形态
        patterns_df['STOCHRSI_OVERBOUGHT'] = k > self.overbought
        patterns_df['STOCHRSI_OVERSOLD'] = k < self.oversold

        # 2. StochRSI金叉死叉形态
        from utils.indicator_utils import crossover, crossunder
        patterns_df['STOCHRSI_GOLDEN_CROSS'] = crossover(k, d)
        patterns_df['STOCHRSI_DEATH_CROSS'] = crossunder(k, d)

        # 3. StochRSI位置形态
        patterns_df['STOCHRSI_K_ABOVE_D'] = k > d
        patterns_df['STOCHRSI_K_BELOW_D'] = k < d

        # 4. StochRSI趋势形态
        patterns_df['STOCHRSI_K_RISING'] = k > k.shift(1)
        patterns_df['STOCHRSI_K_FALLING'] = k < k.shift(1)
        patterns_df['STOCHRSI_D_RISING'] = d > d.shift(1)
        patterns_df['STOCHRSI_D_FALLING'] = d < d.shift(1)

        # 5. StochRSI强势形态
        patterns_df['STOCHRSI_STRONG_BULLISH'] = (k > 80) & (k > d) & (k > k.shift(1))
        patterns_df['STOCHRSI_STRONG_BEARISH'] = (k < 20) & (k < d) & (k < k.shift(1))

        # 6. StochRSI反转形态
        patterns_df['STOCHRSI_OVERSOLD_REVERSAL'] = (k.shift(1) < self.oversold) & (k > self.oversold)
        patterns_df['STOCHRSI_OVERBOUGHT_REVERSAL'] = (k.shift(1) > self.overbought) & (k < self.overbought)

        return patterns_df

    def calculate_confidence(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """
        计算StochRSI指标的置信度

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
            # 检查StochRSI形态
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
        signals['signal_strength'] = pd.Series(0.0, index=data.index)
    
        # 如果没有结果，返回空信号
        if self._result is None or 'stochrsi_k' not in self._result.columns:
            return signals
        
        # 获取K线和D线值
        k = self._result['stochrsi_k']
        d = self._result['stochrsi_d']
        
        # 生成金叉买入信号
        for i in range(1, len(k)):
            if i < 1 or np.isnan(k.iloc[i-1]) or np.isnan(d.iloc[i-1]) or np.isnan(k.iloc[i]) or np.isnan(d.iloc[i]):
                continue
                
            # K线上穿D线，生成买入信号
            if k.iloc[i-1] <= d.iloc[i-1] and k.iloc[i] > d.iloc[i]:
                signals['buy_signal'].iloc[i] = True
                # 计算信号强度
                angle = self._calculate_cross_angle(k.iloc[i-1], k.iloc[i], d.iloc[i-1], d.iloc[i])
                # 超卖区的金叉更强
                strength_boost = 20 if k.iloc[i-1] < self.oversold else 0
                signals['signal_strength'].iloc[i] = 50 + min(angle / 90 * 30, 30) + strength_boost
        
        # 生成死叉卖出信号
        for i in range(1, len(k)):
            if i < 1 or np.isnan(k.iloc[i-1]) or np.isnan(d.iloc[i-1]) or np.isnan(k.iloc[i]) or np.isnan(d.iloc[i]):
                continue
                
            # K线下穿D线，生成卖出信号
            if k.iloc[i-1] >= d.iloc[i-1] and k.iloc[i] < d.iloc[i]:
                signals['sell_signal'].iloc[i] = True
                # 计算信号强度
                angle = self._calculate_cross_angle(k.iloc[i-1], k.iloc[i], d.iloc[i-1], d.iloc[i])
                # 超买区的死叉更强
                strength_boost = 20 if k.iloc[i-1] > self.overbought else 0
                signals['signal_strength'].iloc[i] = 50 + min(angle / 90 * 30, 30) + strength_boost
        
        # 生成超卖区上穿信号
        for i in range(1, len(k)):
            if i < 1 or np.isnan(k.iloc[i-1]) or np.isnan(k.iloc[i]):
                continue
                
            # 从超卖区上穿，生成买入信号
            if k.iloc[i-1] <= self.oversold and k.iloc[i] > self.oversold:
                signals['buy_signal'].iloc[i] = True
                # 信号强度基于与超卖线的距离
                distance = k.iloc[i] - self.oversold
                signals['signal_strength'].iloc[i] = 60 + min(distance * 0.5, 20)  # 60-80
        
        # 生成超买区下穿信号
        for i in range(1, len(k)):
            if i < 1 or np.isnan(k.iloc[i-1]) or np.isnan(k.iloc[i]):
                continue
                
            # 从超买区下穿，生成卖出信号
            if k.iloc[i-1] >= self.overbought and k.iloc[i] < self.overbought:
                signals['sell_signal'].iloc[i] = True
                # 信号强度基于与超买线的距离
                distance = self.overbought - k.iloc[i]
                signals['signal_strength'].iloc[i] = 60 + min(distance * 0.5, 20)  # 60-80
        
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
        
        if self._result is None or 'stochrsi_k' not in self._result.columns:
            return pd.Series(50.0, index=data.index)
        
        # 初始化评分
        score = pd.Series(50.0, index=data.index)
        
        # 获取K线和D线值
        k = self._result['stochrsi_k']
        d = self._result['stochrsi_d']
        
        # 根据StochRSI的值计算评分
        for i in range(len(k)):
            if np.isnan(k.iloc[i]):
                continue
                
            # 超卖区 (评分高)
            if k.iloc[i] < self.oversold:
                # 越接近0，评分越高
                ratio = (self.oversold - k.iloc[i]) / self.oversold
                score.iloc[i] = 50 + ratio * 30  # 50-80
            # 超买区 (评分低)
            elif k.iloc[i] > self.overbought:
                # 越接近100，评分越低
                ratio = (k.iloc[i] - self.overbought) / (100 - self.overbought)
                score.iloc[i] = 50 - ratio * 30  # 20-50
            # 中性区域
            else:
                # 线性插值
                ratio = (k.iloc[i] - self.oversold) / (self.overbought - self.oversold)
                score.iloc[i] = 50 + 30 - ratio * 30  # 50-80 (递减)
        
        # 考虑K线和D线的相对位置
        for i in range(1, len(k)):
            if i < 1 or np.isnan(k.iloc[i]) or np.isnan(d.iloc[i]):
                continue
                
            # K线在D线上方，略微提高评分
            if k.iloc[i] > d.iloc[i]:
                diff = (k.iloc[i] - d.iloc[i]) / 100  # 归一化差值
                score.iloc[i] = min(score.iloc[i] + diff * 10, 90)  # 最多+10分
            # K线在D线下方，略微降低评分
            else:
                diff = (d.iloc[i] - k.iloc[i]) / 100  # 归一化差值
                score.iloc[i] = max(score.iloc[i] - diff * 10, 10)  # 最多-10分
        
        # 考虑金叉和死叉的影响
        for i in range(1, len(k)):
            if i < 1 or np.isnan(k.iloc[i-1]) or np.isnan(d.iloc[i-1]) or np.isnan(k.iloc[i]) or np.isnan(d.iloc[i]):
                continue
                
            # 金叉提高评分
            if k.iloc[i-1] <= d.iloc[i-1] and k.iloc[i] > d.iloc[i]:
                angle = self._calculate_cross_angle(k.iloc[i-1], k.iloc[i], d.iloc[i-1], d.iloc[i])
                adjust = min(angle / 90 * 15, 15)  # 最多增加15分
                score.iloc[i] = min(score.iloc[i] + adjust, 90)
            
            # 死叉降低评分
            if k.iloc[i-1] >= d.iloc[i-1] and k.iloc[i] < d.iloc[i]:
                angle = self._calculate_cross_angle(k.iloc[i-1], k.iloc[i], d.iloc[i-1], d.iloc[i])
                adjust = min(angle / 90 * 15, 15)  # 最多减少15分
                score.iloc[i] = max(score.iloc[i] - adjust, 10)
        
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
        注册StochRSI指标的形态到全局形态注册表
        """
        # 注册StochRSI金叉形态
        self.register_pattern_to_registry(
            pattern_id="STOCHRSI_GOLDEN_CROSS",
            display_name="StochRSI金叉",
            description="StochRSI K线上穿D线，产生看涨信号",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=25.0,
            polarity="POSITIVE"
        )

        # 注册StochRSI死叉形态
        self.register_pattern_to_registry(
            pattern_id="STOCHRSI_DEATH_CROSS",
            display_name="StochRSI死叉",
            description="StochRSI K线下穿D线，产生看跌信号",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-25.0,
            polarity="NEGATIVE"
        )

        # 注册StochRSI超买形态
        self.register_pattern_to_registry(
            pattern_id="STOCHRSI_OVERBOUGHT",
            display_name="StochRSI超买",
            description="StochRSI进入超买区域，可能出现回调",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-15.0,
            polarity="NEGATIVE"
        )

        # 注册StochRSI超卖形态
        self.register_pattern_to_registry(
            pattern_id="STOCHRSI_OVERSOLD",
            display_name="StochRSI超卖",
            description="StochRSI进入超卖区域，可能出现反弹",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=15.0,
            polarity="POSITIVE"
        )

        # 注册StochRSI超卖反转形态
        self.register_pattern_to_registry(
            pattern_id="STOCHRSI_OVERSOLD_REVERSAL",
            display_name="StochRSI超卖反转",
            description="StochRSI从超卖区域向上突破，看涨信号",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=20.0,
            polarity="POSITIVE"
        )

        # 注册StochRSI超买反转形态
        self.register_pattern_to_registry(
            pattern_id="STOCHRSI_OVERBOUGHT_REVERSAL",
            display_name="StochRSI超买反转",
            description="StochRSI从超买区域向下突破，看跌信号",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-20.0,
            polarity="NEGATIVE"
        )

        # 注册StochRSI强势看涨形态
        self.register_pattern_to_registry(
            pattern_id="STOCHRSI_STRONG_BULLISH",
            display_name="StochRSI强势看涨",
            description="StochRSI K线在高位且上升，强势看涨",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=18.0,
            polarity="POSITIVE"
        )

        # 注册StochRSI强势看跌形态
        self.register_pattern_to_registry(
            pattern_id="STOCHRSI_STRONG_BEARISH",
            display_name="StochRSI强势看跌",
            description="StochRSI K线在低位且下降，强势看跌",
            pattern_type="BEARISH",
            default_strength="VERY_STRONG",
            score_impact=-18.0,
            polarity="NEGATIVE"
        )

    def get_pattern_info(self, pattern_id: str) -> dict:
        """
        获取指定形态的详细信息

        Args:
            pattern_id: 形态ID

        Returns:
            dict: 形态详细信息
        """
        pattern_info_map = {
            "STOCHRSI_GOLDEN_CROSS": {
                "id": "STOCHRSI_GOLDEN_CROSS",
                "name": "StochRSI金叉",
                "description": "StochRSI K线上穿D线，产生看涨信号",
                "type": "BULLISH",
                "strength": "STRONG",
                "score_impact": 25.0
            },
            "STOCHRSI_DEATH_CROSS": {
                "id": "STOCHRSI_DEATH_CROSS",
                "name": "StochRSI死叉",
                "description": "StochRSI K线下穿D线，产生看跌信号",
                "type": "BEARISH",
                "strength": "STRONG",
                "score_impact": -25.0
            },
            "STOCHRSI_OVERBOUGHT": {
                "id": "STOCHRSI_OVERBOUGHT",
                "name": "StochRSI超买",
                "description": "StochRSI进入超买区域，可能出现回调",
                "type": "BEARISH",
                "strength": "MEDIUM",
                "score_impact": -15.0
            },
            "STOCHRSI_OVERSOLD": {
                "id": "STOCHRSI_OVERSOLD",
                "name": "StochRSI超卖",
                "description": "StochRSI进入超卖区域，可能出现反弹",
                "type": "BULLISH",
                "strength": "MEDIUM",
                "score_impact": 15.0
            },
            "STOCHRSI_OVERSOLD_REVERSAL": {
                "id": "STOCHRSI_OVERSOLD_REVERSAL",
                "name": "StochRSI超卖反转",
                "description": "StochRSI从超卖区域向上突破，看涨信号",
                "type": "BULLISH",
                "strength": "STRONG",
                "score_impact": 20.0
            },
            "STOCHRSI_OVERBOUGHT_REVERSAL": {
                "id": "STOCHRSI_OVERBOUGHT_REVERSAL",
                "name": "StochRSI超买反转",
                "description": "StochRSI从超买区域向下突破，看跌信号",
                "type": "BEARISH",
                "strength": "STRONG",
                "score_impact": -20.0
            },
            "STOCHRSI_STRONG_BULLISH": {
                "id": "STOCHRSI_STRONG_BULLISH",
                "name": "StochRSI强势看涨",
                "description": "StochRSI K线在高位且上升，强势看涨",
                "type": "BULLISH",
                "strength": "VERY_STRONG",
                "score_impact": 18.0
            },
            "STOCHRSI_STRONG_BEARISH": {
                "id": "STOCHRSI_STRONG_BEARISH",
                "name": "StochRSI强势看跌",
                "description": "StochRSI K线在低位且下降，强势看跌",
                "type": "BEARISH",
                "strength": "VERY_STRONG",
                "score_impact": -18.0
            }
        }

        return pattern_info_map.get(pattern_id, {
            "id": pattern_id,
            "name": "未知形态",
            "description": "未定义的形态",
            "type": "NEUTRAL",
            "strength": "WEAK",
            "score_impact": 0.0
        })