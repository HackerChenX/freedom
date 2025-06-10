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
        super().__init__()
        self.name = "STOCHRSI"
        self.rsi_period = rsi_period
        self.k_period = k_period
        self.d_period = d_period
        self.overbought = overbought
        self.oversold = oversold
    
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
        # 避免除以零，当max == min时，k = 0.5
        divisor = rsi_max - rsi_min
        divisor = divisor.replace(0, 1e-9)
        k = 100 * (rsi - rsi_min) / divisor
        
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
    
    def get_patterns(self, data: pd.DataFrame, **kwargs) -> List[Dict[str, Any]]:
        """
        获取StochRSI相关形态
        
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
        
        if self._result is None or 'stochrsi_k' not in self._result.columns:
            return patterns
        
        # 获取K线和D线值
        k = self._result['stochrsi_k']
        d = self._result['stochrsi_d']
        dates = self._result.index
        
        # 1. 识别StochRSI超买
        for i in range(1, len(k)):
            if i < 1:
                continue
                
            if k.iloc[i] > self.overbought:
                # 计算持续时间
                duration = 1
                for j in range(i+1, len(k)):
                    if k.iloc[j] > self.overbought:
                        duration += 1
                    else:
                        break
                
                # 如果持续时间足够长，添加形态
                if duration >= 2:
                    pattern = {
                        "name": "StochRSI超买",
                        "start_date": dates[i],
                        "end_date": dates[min(i+duration-1, len(dates)-1)],
                        "duration": duration,
                        "strength": (k.iloc[i] - self.overbought) / (100 - self.overbought),  # 归一化强度
                        "description": f"StochRSI在{self.overbought}以上持续{duration}天，表明市场可能超买",
                        "type": "bearish"  # 超买是看跌信号
                    }
                    patterns.append(pattern)
                
                # 跳过已经识别的区域
                i += duration - 1
        
        # 2. 识别StochRSI超卖
        for i in range(1, len(k)):
            if i < 1:
                continue
                
            if k.iloc[i] < self.oversold:
                # 计算持续时间
                duration = 1
                for j in range(i+1, len(k)):
                    if k.iloc[j] < self.oversold:
                        duration += 1
                    else:
                        break
                
                # 如果持续时间足够长，添加形态
                if duration >= 2:
                    pattern = {
                        "name": "StochRSI超卖",
                        "start_date": dates[i],
                        "end_date": dates[min(i+duration-1, len(dates)-1)],
                        "duration": duration,
                        "strength": (self.oversold - k.iloc[i]) / self.oversold,  # 归一化强度
                        "description": f"StochRSI在{self.oversold}以下持续{duration}天，表明市场可能超卖",
                        "type": "bullish"  # 超卖是看涨信号
                    }
                    patterns.append(pattern)
                
                # 跳过已经识别的区域
                i += duration - 1
        
        # 3. 识别StochRSI金叉
        for i in range(1, len(k)):
            if i < 1 or np.isnan(k.iloc[i-1]) or np.isnan(d.iloc[i-1]) or np.isnan(k.iloc[i]) or np.isnan(d.iloc[i]):
                continue
                
            # K线上穿D线，金叉信号
            if k.iloc[i-1] <= d.iloc[i-1] and k.iloc[i] > d.iloc[i]:
                # 计算金叉强度 (基于交叉角度)
                angle = self._calculate_cross_angle(k.iloc[i-1], k.iloc[i], d.iloc[i-1], d.iloc[i])
                
                pattern = {
                    "name": "StochRSI金叉",
                    "start_date": dates[i-1],
                    "end_date": dates[i],
                    "duration": 2,
                    "strength": min(angle / 90, 1.0),  # 归一化到0-1
                    "description": "StochRSI K线上穿D线，可能是买入信号",
                    "type": "bullish"
                }
                patterns.append(pattern)
        
        # 4. 识别StochRSI死叉
        for i in range(1, len(k)):
            if i < 1 or np.isnan(k.iloc[i-1]) or np.isnan(d.iloc[i-1]) or np.isnan(k.iloc[i]) or np.isnan(d.iloc[i]):
                continue
                
            # K线下穿D线，死叉信号
            if k.iloc[i-1] >= d.iloc[i-1] and k.iloc[i] < d.iloc[i]:
                # 计算死叉强度 (基于交叉角度)
                angle = self._calculate_cross_angle(k.iloc[i-1], k.iloc[i], d.iloc[i-1], d.iloc[i])
                
                pattern = {
                    "name": "StochRSI死叉",
                    "start_date": dates[i-1],
                    "end_date": dates[i],
                    "duration": 2,
                    "strength": min(angle / 90, 1.0),  # 归一化到0-1
                    "description": "StochRSI K线下穿D线，可能是卖出信号",
                    "type": "bearish"
                }
                patterns.append(pattern)
        
        # 5. 识别顶背离和底背离
        if 'close' in self._result.columns:
            # 价格新高但StochRSI没有新高 - 顶背离
            for i in range(20, len(k)):
                if i < 5:
                    continue
                
                # 获取近期价格和StochRSI
                recent_prices = self._result['close'].iloc[i-20:i+1]
                recent_k = k.iloc[i-20:i+1]
                
                # 判断价格是否创新高
                if recent_prices.iloc[-1] > recent_prices.iloc[:-1].max():
                    # 检查StochRSI是否没有同步创新高
                    if recent_k.iloc[-1] < recent_k.iloc[:-1].max():
                        # 背离强度
                        price_change = (recent_prices.iloc[-1] - recent_prices.iloc[:-1].max()) / recent_prices.iloc[:-1].max()
                        k_change = (recent_k.iloc[-1] - recent_k.iloc[:-1].max()) / (recent_k.iloc[:-1].max() if recent_k.iloc[:-1].max() != 0 else 1)
                        strength = min(abs(price_change - k_change) / max(abs(price_change), 0.01), 1.0)
                        
                        pattern = {
                            "name": "StochRSI顶背离",
                            "start_date": dates[i-5],  # 使用适当的起始日期
                            "end_date": dates[i],
                            "duration": 5,  # 使用固定值表示背离形态
                            "strength": strength,
                            "description": "价格创新高但StochRSI未同步创新高，可能暗示上涨动能减弱",
                            "type": "bearish"  # 顶背离是看跌信号
                        }
                        patterns.append(pattern)
            
            # 价格新低但StochRSI没有新低 - 底背离
            for i in range(20, len(k)):
                if i < 5:
                    continue
                
                # 获取近期价格和StochRSI
                recent_prices = self._result['close'].iloc[i-20:i+1]
                recent_k = k.iloc[i-20:i+1]
                
                # 判断价格是否创新低
                if recent_prices.iloc[-1] < recent_prices.iloc[:-1].min():
                    # 检查StochRSI是否没有同步创新低
                    if recent_k.iloc[-1] > recent_k.iloc[:-1].min():
                        # 背离强度
                        price_change = (recent_prices.iloc[-1] - recent_prices.iloc[:-1].min()) / recent_prices.iloc[:-1].min()
                        k_change = (recent_k.iloc[-1] - recent_k.iloc[:-1].min()) / (recent_k.iloc[:-1].min() if recent_k.iloc[:-1].min() != 0 else 1)
                        strength = min(abs(price_change - k_change) / max(abs(price_change), 0.01), 1.0)
                        
                        pattern = {
                            "name": "StochRSI底背离",
                            "start_date": dates[i-5],
                            "end_date": dates[i],
                            "duration": 5,
                            "strength": strength,
                            "description": "价格创新低但StochRSI未同步创新低，可能暗示下跌动能减弱",
                            "type": "bullish"  # 底背离是看涨信号
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
        
        # 基于背离形态生成信号
        patterns = self.get_patterns(data, **kwargs)
        for pattern in patterns:
            if pattern['name'] == 'StochRSI底背离' and pattern['type'] == 'bullish':
                end_date = pattern['end_date']
                if end_date in signals['buy_signal'].index:
                    signals['buy_signal'].loc[end_date] = True
                    signals['signal_strength'].loc[end_date] = 70 + min(pattern['strength'] * 20, 20)  # 70-90
            
            elif pattern['name'] == 'StochRSI顶背离' and pattern['type'] == 'bearish':
                end_date = pattern['end_date']
                if end_date in signals['sell_signal'].index:
                    signals['sell_signal'].loc[end_date] = True
                    signals['signal_strength'].loc[end_date] = 70 + min(pattern['strength'] * 20, 20)  # 70-90
    
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
        
        # 结合背离形态增强评分
        patterns = self.get_patterns(data, **kwargs)
        for pattern in patterns:
            if pattern['name'] == 'StochRSI底背离' and pattern['type'] == 'bullish':
                # 底背离增加评分
                end_idx = data.index.get_loc(pattern['end_date'])
                adjust_range = min(5, len(score) - end_idx - 1)
                for j in range(adjust_range):
                    idx = end_idx + j
                    strength_factor = pattern['strength'] * (1 - j/adjust_range)  # 随时间衰减
                    score.iloc[idx] = min(score.iloc[idx] + strength_factor * 20, 90)
            
            elif pattern['name'] == 'StochRSI顶背离' and pattern['type'] == 'bearish':
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