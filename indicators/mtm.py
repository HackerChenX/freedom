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
    
    def get_patterns(self, data: pd.DataFrame, **kwargs) -> List[Dict[str, Any]]:
        """
        获取MTM相关形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[Dict[str, Any]]: 识别的形态列表
        """
        patterns = []
        
        # 确保已计算指标
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None or 'mtm' not in self._result.columns:
            return patterns
        
        # 获取MTM和MTMMA值
        mtm = self._result['mtm']
        mtmma = self._result['mtmma']
        dates = self._result.index
        
        # 1. 识别MTM超买
        for i in range(1, len(mtm)):
            if i < 1:
                continue
                
            if mtm.iloc[i] > self.overbought:
                # 计算持续时间
                duration = 1
                for j in range(i+1, len(mtm)):
                    if mtm.iloc[j] > self.overbought:
                        duration += 1
                    else:
                        break
                
                # 如果持续时间足够长，添加形态
                if duration >= 2:
                    pattern = {
                        "name": "MTM超买",
                        "start_date": dates[i],
                        "end_date": dates[min(i+duration-1, len(dates)-1)],
                        "duration": duration,
                        "strength": (mtm.iloc[i] - self.overbought) / abs(self.overbought) if self.overbought != 0 else 0.5,
                        "description": f"MTM在超买区持续{duration}天，可能暗示价格上涨过快",
                        "type": "bearish"  # 超买是看跌信号
                    }
                    patterns.append(pattern)
                
                # 跳过已经识别的区域
                i += duration - 1
        
        # 2. 识别MTM超卖
        for i in range(1, len(mtm)):
            if i < 1:
                continue
                
            if mtm.iloc[i] < self.oversold:
                # 计算持续时间
                duration = 1
                for j in range(i+1, len(mtm)):
                    if mtm.iloc[j] < self.oversold:
                        duration += 1
                    else:
                        break
                
                # 如果持续时间足够长，添加形态
                if duration >= 2:
                    pattern = {
                        "name": "MTM超卖",
                        "start_date": dates[i],
                        "end_date": dates[min(i+duration-1, len(dates)-1)],
                        "duration": duration,
                        "strength": (self.oversold - mtm.iloc[i]) / abs(self.oversold) if self.oversold != 0 else 0.5,
                        "description": f"MTM在超卖区持续{duration}天，可能暗示价格下跌过快",
                        "type": "bullish"  # 超卖是看涨信号
                    }
                    patterns.append(pattern)
                
                # 跳过已经识别的区域
                i += duration - 1
        
        # 3. 识别MTM金叉
        for i in range(1, len(mtm)):
            if i < 1:
                continue
                
            # MTM上穿MTMMA，金叉信号
            if mtm.iloc[i-1] <= mtmma.iloc[i-1] and mtm.iloc[i] > mtmma.iloc[i]:
                # 确定信号强度（基于交叉角度）
                angle = self._calculate_cross_angle(mtm.iloc[i-1], mtm.iloc[i], mtmma.iloc[i-1], mtmma.iloc[i])
                
                pattern = {
                    "name": "MTM金叉",
                    "start_date": dates[i-1],
                    "end_date": dates[i],
                    "duration": 2,
                    "strength": min(angle / 90, 1.0),  # 归一化到0-1
                    "description": "MTM上穿MTMMA，动量由负转正，可能是买入信号",
                    "type": "bullish"
                }
                patterns.append(pattern)
        
        # 4. 识别MTM死叉
        for i in range(1, len(mtm)):
            if i < 1:
                continue
                
            # MTM下穿MTMMA，死叉信号
            if mtm.iloc[i-1] >= mtmma.iloc[i-1] and mtm.iloc[i] < mtmma.iloc[i]:
                # 确定信号强度（基于交叉角度）
                angle = self._calculate_cross_angle(mtm.iloc[i-1], mtm.iloc[i], mtmma.iloc[i-1], mtmma.iloc[i])
                
                pattern = {
                    "name": "MTM死叉",
                    "start_date": dates[i-1],
                    "end_date": dates[i],
                    "duration": 2,
                    "strength": min(angle / 90, 1.0),  # 归一化到0-1
                    "description": "MTM下穿MTMMA，动量由正转负，可能是卖出信号",
                    "type": "bearish"
                }
                patterns.append(pattern)
        
        # 5. 识别MTM背离
        if 'close' in self._result.columns:
            # 价格新高但MTM没有新高 - 顶背离
            for i in range(20, len(mtm)):
                if i < 5:
                    continue
                
                # 获取近期价格和MTM
                recent_prices = self._result['close'].iloc[i-20:i+1]
                recent_mtm = mtm.iloc[i-20:i+1]
                
                # 判断价格是否创新高
                if recent_prices.iloc[-1] > recent_prices.iloc[:-1].max():
                    # 检查MTM是否没有同步创新高
                    if recent_mtm.iloc[-1] < recent_mtm.iloc[:-1].max():
                        strength = self._calculate_divergence_strength(
                            recent_prices.iloc[-1], recent_prices.iloc[:-1].max(),
                            recent_mtm.iloc[-1], recent_mtm.iloc[:-1].max()
                        )
                        
                        pattern = {
                            "name": "MTM顶背离",
                            "start_date": dates[i-5],
                            "end_date": dates[i],
                            "duration": 5,
                            "strength": strength,
                            "description": "价格创新高但MTM未同步创新高，可能暗示上涨动能减弱",
                            "type": "bearish"
                        }
                        patterns.append(pattern)
            
            # 价格新低但MTM没有新低 - 底背离
            for i in range(20, len(mtm)):
                if i < 5:
                    continue
                
                # 获取近期价格和MTM
                recent_prices = self._result['close'].iloc[i-20:i+1]
                recent_mtm = mtm.iloc[i-20:i+1]
                
                # 判断价格是否创新低
                if recent_prices.iloc[-1] < recent_prices.iloc[:-1].min():
                    # 检查MTM是否没有同步创新低
                    if recent_mtm.iloc[-1] > recent_mtm.iloc[:-1].min():
                        strength = self._calculate_divergence_strength(
                            recent_prices.iloc[-1], recent_prices.iloc[:-1].min(),
                            recent_mtm.iloc[-1], recent_mtm.iloc[:-1].min()
                        )
                        
                        pattern = {
                            "name": "MTM底背离",
                            "start_date": dates[i-5],
                            "end_date": dates[i],
                            "duration": 5,
                            "strength": strength,
                            "description": "价格创新低但MTM未同步创新低，可能暗示下跌动能减弱",
                            "type": "bullish"
                        }
                        patterns.append(pattern)
        
        return patterns
    
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
        signals['signal_strength'] = pd.Series(0, index=data.index)
    
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
                signals['signal_strength'].iloc[i] = 50 + min(angle / 90 * 40, 40)  # 50-90
        
        # 生成死叉卖出信号
        for i in range(1, len(mtm)):
            # MTM下穿MTMMA，死叉信号
            if mtm.iloc[i-1] >= mtmma.iloc[i-1] and mtm.iloc[i] < mtmma.iloc[i]:
                signals['sell_signal'].iloc[i] = True
                # 计算信号强度：基于交叉角度
                angle = self._calculate_cross_angle(mtm.iloc[i-1], mtm.iloc[i], mtmma.iloc[i-1], mtmma.iloc[i])
                signals['signal_strength'].iloc[i] = 50 + min(angle / 90 * 40, 40)  # 50-90
        
        # 生成超卖买入信号
        for i in range(1, len(mtm)):
            # MTM从超卖区域上穿，生成买入信号
            if mtm.iloc[i-1] <= self.oversold and mtm.iloc[i] > self.oversold:
                signals['buy_signal'].iloc[i] = True
                # 信号强度基于超卖程度
                signals['signal_strength'].iloc[i] = 60 + min((self.oversold - mtm.iloc[i-1]) / abs(self.oversold) * 30, 30)
        
        # 生成超买卖出信号
        for i in range(1, len(mtm)):
            # MTM从超买区域下穿，生成卖出信号
            if mtm.iloc[i-1] >= self.overbought and mtm.iloc[i] < self.overbought:
                signals['sell_signal'].iloc[i] = True
                # 信号强度基于超买程度
                signals['signal_strength'].iloc[i] = 60 + min((mtm.iloc[i-1] - self.overbought) / abs(self.overbought) * 30, 30)
        
        # 基于MTM背离形态生成信号
        patterns = self.get_patterns(data, **kwargs)
        for pattern in patterns:
            if pattern['name'] == 'MTM底背离' and pattern['type'] == 'bullish':
                end_date = pattern['end_date']
                if end_date in signals['buy_signal'].index:
                    signals['buy_signal'].loc[end_date] = True
                    signals['signal_strength'].loc[end_date] = 70 + min(pattern['strength'] * 30, 20)
            
            elif pattern['name'] == 'MTM顶背离' and pattern['type'] == 'bearish':
                end_date = pattern['end_date']
                if end_date in signals['sell_signal'].index:
                    signals['sell_signal'].loc[end_date] = True
                    signals['signal_strength'].loc[end_date] = 70 + min(pattern['strength'] * 30, 20)
    
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
        
        # 结合背离形态增强评分
        patterns = self.get_patterns(data, **kwargs)
        for pattern in patterns:
            if pattern['name'] == 'MTM底背离' and pattern['type'] == 'bullish':
                # 底背离增加评分
                end_idx = data.index.get_loc(pattern['end_date'])
                adjust_range = min(5, len(score) - end_idx - 1)
                for j in range(adjust_range):
                    idx = end_idx + j
                    strength_factor = pattern['strength'] * (1 - j/adjust_range)  # 随时间衰减
                    score.iloc[idx] = min(score.iloc[idx] + strength_factor * 20, 90)
            
            elif pattern['name'] == 'MTM顶背离' and pattern['type'] == 'bearish':
                # 顶背离降低评分
                end_idx = data.index.get_loc(pattern['end_date'])
                adjust_range = min(5, len(score) - end_idx - 1)
                for j in range(adjust_range):
                    idx = end_idx + j
                    strength_factor = pattern['strength'] * (1 - j/adjust_range)  # 随时间衰减
                    score.iloc[idx] = max(score.iloc[idx] - strength_factor * 20, 10)
        
        return score
    
    def calculate_score(self, data: pd.DataFrame, **kwargs) -> float:
        """
        计算最终评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
        
        Returns:
            float: 评分(0-100)
        """
        # 计算原始评分序列
        raw_scores = self.calculate_raw_score(data, **kwargs)
        
        # 如果数据不足，返回中性评分
        if len(raw_scores) < 3:
            return 50.0
        
        # 取最近的评分作为最终评分，但考虑近期趋势
        recent_scores = raw_scores.iloc[-3:]
        trend = recent_scores.iloc[-1] - recent_scores.iloc[0]
        
        # 最终评分 = 最新评分 + 趋势调整
        final_score = recent_scores.iloc[-1] + trend / 2
        
        # 确保评分在0-100范围内
        return max(0, min(100, final_score)) 