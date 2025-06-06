#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
相对强弱指数(RSI)

通过分析价格波动的幅度和速度，判断市场的超买超卖情况
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any, Union, Tuple

from indicators.base_indicator import BaseIndicator, PatternResult
from utils.logger import get_logger

logger = get_logger(__name__)


class RSI(BaseIndicator):
    """
    相对强弱指数(RSI)
    
    分类：震荡类指标
    计算方法：RSI = 100 * RS / (1 + RS)，其中RS是平均上涨幅度与平均下跌幅度的比值
    参数：N，一般取14，表示计算周期
    """
    
    def __init__(self, period: int = 14, overbought: float = 70, oversold: float = 30):
        """
        初始化RSI指标
        
        Args:
            period: 计算周期，默认为14
            overbought: 超买线，默认为70
            oversold: 超卖线，默认为30
        """
        self.name = "RSI"
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        
        # 初始化基类（会自动调用register_patterns方法）
        super().__init__()
    
    def _register_rsi_patterns(self):
        """
        注册RSI指标形态
        """
        from indicators.pattern_registry import PatternRegistry, PatternType, PatternStrength
        
        # 获取PatternRegistry实例
        registry = PatternRegistry()
        
        # 注册RSI超买/超卖形态
        registry.register(
            pattern_id="RSI_OVERBOUGHT",
            display_name="RSI超买",
            description=f"RSI值高于{self.overbought}，表明市场可能超买，存在回调风险",
            indicator_id="RSI",
            pattern_type=PatternType.BEARISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=-15.0
        )
        
        registry.register(
            pattern_id="RSI_OVERSOLD",
            display_name="RSI超卖",
            description=f"RSI值低于{self.oversold}，表明市场可能超卖，存在反弹机会",
            indicator_id="RSI",
            pattern_type=PatternType.BULLISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=15.0
        )
        
        # 注册RSI背离形态
        registry.register(
            pattern_id="RSI_BULLISH_DIVERGENCE",
            display_name="RSI底背离",
            description="价格创新低，但RSI未创新低，表明下跌动能减弱，可能即将反转向上",
            indicator_id="RSI",
            pattern_type=PatternType.BULLISH,
            default_strength=PatternStrength.STRONG,
            score_impact=25.0
        )
        
        registry.register(
            pattern_id="RSI_BEARISH_DIVERGENCE",
            display_name="RSI顶背离",
            description="价格创新高，但RSI未创新高，表明上涨动能减弱，可能即将反转向下",
            indicator_id="RSI",
            pattern_type=PatternType.BEARISH,
            default_strength=PatternStrength.STRONG,
            score_impact=-25.0
        )
        
        # 注册RSI突破形态
        registry.register(
            pattern_id="RSI_BREAKOUT_UP",
            display_name="RSI向上突破",
            description=f"RSI突破{self.overbought}水平线，表明强势上涨可能开始",
            indicator_id="RSI",
            pattern_type=PatternType.BULLISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=10.0
        )
        
        registry.register(
            pattern_id="RSI_BREAKOUT_DOWN",
            display_name="RSI向下突破",
            description=f"RSI跌破{self.oversold}水平线，表明强势下跌可能开始",
            indicator_id="RSI",
            pattern_type=PatternType.BEARISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=-10.0
        )
        
        # 注册RSI中轴穿越形态
        registry.register(
            pattern_id="RSI_CROSS_ABOVE_50",
            display_name="RSI上穿50",
            description="RSI从下方穿越50中轴线，表明多头力量增强",
            indicator_id="RSI",
            pattern_type=PatternType.BULLISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=12.0
        )
        
        registry.register(
            pattern_id="RSI_CROSS_BELOW_50",
            display_name="RSI下穿50",
            description="RSI从上方穿越50中轴线，表明空头力量增强",
            indicator_id="RSI",
            pattern_type=PatternType.BEARISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=-12.0
        )
        
        # 注册RSI钝化形态
        registry.register(
            pattern_id="RSI_BULLISH_FAILURE_SWING",
            display_name="RSI看涨钝化",
            description="RSI在超卖区触底回升后回调，未再次跌入超卖区便重新上升，形成W底",
            indicator_id="RSI",
            pattern_type=PatternType.BULLISH,
            default_strength=PatternStrength.STRONG,
            score_impact=20.0
        )
        
        registry.register(
            pattern_id="RSI_BEARISH_FAILURE_SWING",
            display_name="RSI看跌钝化",
            description="RSI在超买区触顶回落后反弹，未再次进入超买区便重新下跌，形成M顶",
            indicator_id="RSI",
            pattern_type=PatternType.BEARISH,
            default_strength=PatternStrength.STRONG,
            score_impact=-20.0
        )
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算RSI指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
                
        Returns:
            包含RSI指标的DataFrame
        """
        if df.empty:
            return df
            
        df_copy = df.copy()
        
        # 计算价格变化
        delta = df_copy['close'].diff().fillna(0)
        
        # 分离上涨和下跌
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # 计算平均上涨和下跌
        avg_gain = gain.rolling(window=self.period).mean()
        avg_loss = loss.rolling(window=self.period).mean()
        
        # 处理初始值
        avg_gain.fillna(gain.iloc[:self.period].mean(), inplace=True)
        avg_loss.fillna(loss.iloc[:self.period].mean(), inplace=True)
        
        # 计算RS和RSI
        rs = avg_gain / avg_loss.replace(0, 1e-9)  # 避免除以零
        rsi = 100 - (100 / (1 + rs))
        
        # 添加结果到DataFrame
        df_copy['rsi'] = rsi
        
        # 添加超买超卖状态
        df_copy['rsi_overbought'] = rsi > self.overbought
        df_copy['rsi_oversold'] = rsi < self.oversold
        
        # 存储结果
        self._result = df_copy
        
        return df_copy
    
    def get_patterns(self, data: pd.DataFrame, **kwargs) -> List[Dict[str, Any]]:
        """
        获取RSI相关形态
        
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
        
        if self._result is None or 'rsi' not in self._result.columns:
            return patterns
        
        # 获取RSI值
        rsi_values = self._result['rsi']
        dates = self._result.index
        
        # 1. 识别RSI超买
        for i in range(len(rsi_values)):
            if rsi_values[i] > self.overbought:
                # 计算持续时间
                duration = 1
                for j in range(i+1, len(rsi_values)):
                    if rsi_values[j] > self.overbought:
                        duration += 1
                    else:
                        break
                
                # 如果持续时间足够长，添加形态
                if duration >= 2:
                    pattern = {
                        "name": "RSI超买",
                        "start_date": dates[i],
                        "end_date": dates[min(i+duration-1, len(dates)-1)],
                        "duration": duration,
                        "strength": (rsi_values[i] - self.overbought) / (100 - self.overbought),  # 归一化强度
                        "description": f"RSI在{self.overbought}以上持续{duration}天，表明市场可能超买",
                        "type": "bearish"  # 超买是看跌信号
                    }
                    patterns.append(pattern)
                
                # 跳过已经识别的区域
                i += duration - 1
        
        # 2. 识别RSI超卖
        for i in range(len(rsi_values)):
            if rsi_values[i] < self.oversold:
                # 计算持续时间
                duration = 1
                for j in range(i+1, len(rsi_values)):
                    if rsi_values[j] < self.oversold:
                        duration += 1
                    else:
                        break
                
                # 如果持续时间足够长，添加形态
                if duration >= 2:
                    pattern = {
                        "name": "RSI超卖",
                        "start_date": dates[i],
                        "end_date": dates[min(i+duration-1, len(dates)-1)],
                        "duration": duration,
                        "strength": (self.oversold - rsi_values[i]) / self.oversold,  # 归一化强度
                        "description": f"RSI在{self.oversold}以下持续{duration}天，表明市场可能超卖",
                        "type": "bullish"  # 超卖是看涨信号
                    }
                    patterns.append(pattern)
                
                # 跳过已经识别的区域
                i += duration - 1
        
        # 3. 识别RSI背离
        # 这里需要同时分析价格和RSI
        if 'close' in self._result.columns:
            # 价格新高但RSI没有新高 - 顶背离
            for i in range(20, len(rsi_values)):
                if i < 5:
                    continue
                
                # 获取近期价格和RSI
                recent_prices = self._result['close'].iloc[i-20:i+1]
                recent_rsi = rsi_values.iloc[i-20:i+1]
                
                # 判断价格是否创新高
                if recent_prices.iloc[-1] > recent_prices.iloc[:-1].max():
                    # 检查RSI是否没有同步创新高
                    if recent_rsi.iloc[-1] < recent_rsi.iloc[:-1].max():
                        strength = self._calculate_divergence_strength(
                            recent_prices.iloc[-1], recent_prices.iloc[:-1].max(),
                            recent_rsi.iloc[-1], recent_rsi.iloc[:-1].max()
                        )
                        
                        pattern = {
                            "name": "RSI顶背离",
                            "start_date": dates[i-5],  # 使用适当的起始日期
                            "end_date": dates[i],
                            "duration": 5,  # 使用固定值表示背离形态
                            "strength": strength,
                            "description": "价格创新高但RSI未同步创新高，可能暗示上涨动能减弱",
                            "type": "bearish"  # 顶背离是看跌信号
                        }
                        patterns.append(pattern)
            
            # 价格新低但RSI没有新低 - 底背离
            for i in range(20, len(rsi_values)):
                if i < 5:
                    continue
                
                # 获取近期价格和RSI
                recent_prices = self._result['close'].iloc[i-20:i+1]
                recent_rsi = rsi_values.iloc[i-20:i+1]
                
                # 判断价格是否创新低
                if recent_prices.iloc[-1] < recent_prices.iloc[:-1].min():
                    # 检查RSI是否没有同步创新低
                    if recent_rsi.iloc[-1] > recent_rsi.iloc[:-1].min():
                        strength = self._calculate_divergence_strength(
                            recent_prices.iloc[-1], recent_prices.iloc[:-1].min(),
                            recent_rsi.iloc[-1], recent_rsi.iloc[:-1].min()
                        )
                        
                        pattern = {
                            "name": "RSI底背离",
                            "start_date": dates[i-5],  # 使用适当的起始日期
                            "end_date": dates[i],
                            "duration": 5,  # 使用固定值表示背离形态
                            "strength": strength,
                            "description": "价格创新低但RSI未同步创新低，可能暗示下跌动能减弱",
                            "type": "bullish"  # 底背离是看涨信号
                        }
                        patterns.append(pattern)
        
        return patterns
    
    def _calculate_divergence_strength(self, current_price, previous_price, current_rsi, previous_rsi):
        """
        计算背离强度
        
        Args:
            current_price: 当前价格
            previous_price: 之前价格
            current_rsi: 当前RSI
            previous_rsi: 之前RSI
            
        Returns:
            float: 背离强度，范围0-1
        """
        # 计算价格变化百分比
        price_change = abs(current_price - previous_price) / previous_price
        
        # 计算RSI变化百分比
        rsi_change = abs(current_rsi - previous_rsi) / 100
        
        # 计算背离强度：价格变化和RSI变化的不一致程度
        # 背离越明显，强度越大
        return min(price_change / max(rsi_change, 1e-6), 1.0)
    
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
        if self._result is None or 'rsi' not in self._result.columns:
            return signals
        
        # 获取RSI值
        rsi = self._result['rsi']
        
        # 生成超卖买入信号
        for i in range(1, len(rsi)):
            # RSI从超卖区域上穿，生成买入信号
            if rsi.iloc[i-1] < self.oversold and rsi.iloc[i] > self.oversold:
                signals['buy_signal'].iloc[i] = True
                # 信号强度基于RSI值
                signals['signal_strength'].iloc[i] = 50 + min((self.oversold - rsi.iloc[i-1]) * 2, 40)
        
        # 生成超买卖出信号
        for i in range(1, len(rsi)):
            # RSI从超买区域下穿，生成卖出信号
            if rsi.iloc[i-1] > self.overbought and rsi.iloc[i] < self.overbought:
                signals['sell_signal'].iloc[i] = True
                # 信号强度基于RSI值
                signals['signal_strength'].iloc[i] = 50 + min((rsi.iloc[i-1] - self.overbought) * 2, 40)
        
        # 基于RSI背离形态生成信号
        patterns = self.get_patterns(data, **kwargs)
        for pattern in patterns:
            if pattern['name'] == 'RSI底背离' and pattern['type'] == 'bullish':
                end_date = pattern['end_date']
                if end_date in signals['buy_signal'].index:
                    signals['buy_signal'].loc[end_date] = True
                    signals['signal_strength'].loc[end_date] = 60 + min(pattern['strength'] * 40, 30)
            
            elif pattern['name'] == 'RSI顶背离' and pattern['type'] == 'bearish':
                end_date = pattern['end_date']
                if end_date in signals['sell_signal'].index:
                    signals['sell_signal'].loc[end_date] = True
                    signals['signal_strength'].loc[end_date] = 60 + min(pattern['strength'] * 40, 30)
    
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
        
        if self._result is None or 'rsi' not in self._result.columns:
            return pd.Series(50.0, index=data.index)
        
        # 初始化评分
        score = pd.Series(50.0, index=data.index)
        
        # 获取RSI值
        rsi = self._result['rsi']
        
        # 根据RSI值计算评分
        # RSI在30-70之间时，得分随RSI线性变化
        # RSI低于30时，得分随RSI降低而提高
        # RSI高于70时，得分随RSI提高而降低
        for i in range(len(rsi)):
            if rsi.iloc[i] < self.oversold:
                # RSI在超卖区，得分为50-80，越低得分越高
                score.iloc[i] = 50 + (self.oversold - rsi.iloc[i]) / self.oversold * 30
            elif rsi.iloc[i] > self.overbought:
                # RSI在超买区，得分为50-20，越高得分越低
                score.iloc[i] = 50 - (rsi.iloc[i] - self.overbought) / (100 - self.overbought) * 30
            else:
                # RSI在中间区域，得分为40-60，线性变化
                normalized_rsi = (rsi.iloc[i] - self.oversold) / (self.overbought - self.oversold)
                score.iloc[i] = 40 + normalized_rsi * 20
        
        # 结合背离形态增强评分
        patterns = self.get_patterns(data, **kwargs)
        for pattern in patterns:
            if pattern['name'] == 'RSI底背离' and pattern['type'] == 'bullish':
                # 底背离增加评分
                end_idx = data.index.get_loc(pattern['end_date'])
                adjust_range = min(5, len(score) - end_idx - 1)
                for j in range(adjust_range):
                    idx = end_idx + j
                    strength_factor = pattern['strength'] * (1 - j/adjust_range)  # 随时间衰减
                    score.iloc[idx] = min(score.iloc[idx] + strength_factor * 20, 90)
            
            elif pattern['name'] == 'RSI顶背离' and pattern['type'] == 'bearish':
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