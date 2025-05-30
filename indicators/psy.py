#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
心理线指标(PSY)模块

实现心理线指标计算功能，用于判断市场情绪和超买超卖状态
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger

logger = get_logger(__name__)


class PSY(BaseIndicator):
    """
    心理线指标(Psychological Line)
    
    计算一段时间内上涨日所占百分比，反映市场人气强弱和超买超卖状态
    """
    
    def __init__(self, period: int = 12):
        """
        初始化PSY指标
        
        Args:
            period: 计算周期，默认为12日
        """
        super().__init__(name="PSY", description="心理线指标，计算一段时间内上涨日所占百分比，判断市场情绪")
        self.period = period
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算PSY指标
        
        Args:
            data: 输入数据，包含收盘价
            
        Returns:
            pd.DataFrame: 计算结果，包含PSY指标值
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close"])
        
        # 初始化结果数据框
        result = pd.DataFrame(index=data.index)
        
        # 计算价格变化
        price_change = data["close"].diff()
        
        # 统计上涨日数
        up_days = (price_change > 0).astype(int)
        
        # 计算PSY：N日内上涨天数 / N * 100
        result["psy"] = up_days.rolling(window=self.period).sum() / self.period * 100
        
        # 计算PSY的移动平均线（作为信号线）
        result["psyma"] = result["psy"].rolling(window=int(self.period/2)).mean()
        
        # 额外计算：PSY变化率
        result["psy_change"] = result["psy"].diff()
        
        # 存储结果
        self._result = result
        
        return result
    
    def get_signals(self, data: pd.DataFrame, overbought: float = 75, oversold: float = 25) -> pd.DataFrame:
        """
        生成PSY信号
        
        Args:
            data: 输入数据，包含PSY指标
            overbought: 超买阈值，默认为75
            oversold: 超卖阈值，默认为25
            
        Returns:
            pd.DataFrame: 包含PSY信号的数据框
        """
        if "psy" not in data.columns:
            data = self.calculate(data)
        
        # 初始化信号列
        data["psy_signal"] = np.nan
        
        # 生成信号
        for i in range(1, len(data)):
            if pd.notna(data["psy"].iloc[i]) and pd.notna(data["psy"].iloc[i-1]):
                # PSY下穿超买线：卖出信号
                if data["psy"].iloc[i] < overbought and data["psy"].iloc[i-1] >= overbought:
                    data.iloc[i, data.columns.get_loc("psy_signal")] = -1
                
                # PSY上穿超卖线：买入信号
                elif data["psy"].iloc[i] > oversold and data["psy"].iloc[i-1] <= oversold:
                    data.iloc[i, data.columns.get_loc("psy_signal")] = 1
                
                # PSY上穿信号线：轻微买入信号
                elif data["psy"].iloc[i] > data["psyma"].iloc[i] and data["psy"].iloc[i-1] <= data["psyma"].iloc[i-1]:
                    data.iloc[i, data.columns.get_loc("psy_signal")] = 0.5
                
                # PSY下穿信号线：轻微卖出信号
                elif data["psy"].iloc[i] < data["psyma"].iloc[i] and data["psy"].iloc[i-1] >= data["psyma"].iloc[i-1]:
                    data.iloc[i, data.columns.get_loc("psy_signal")] = -0.5
                
                # 无信号
                else:
                    data.iloc[i, data.columns.get_loc("psy_signal")] = 0
        
        # 检测PSY背离
        data["psy_divergence"] = np.nan
        window = 20  # 背离检测窗口
        
        for i in range(window, len(data)):
            # 价格新高/新低检测
            price_high = data["close"].iloc[i] >= np.max(data["close"].iloc[i-window:i])
            price_low = data["close"].iloc[i] <= np.min(data["close"].iloc[i-window:i])
            
            # PSY新高/新低检测
            psy_high = data["psy"].iloc[i] >= np.max(data["psy"].iloc[i-window:i])
            psy_low = data["psy"].iloc[i] <= np.min(data["psy"].iloc[i-window:i])
            
            # 顶背离：价格新高但PSY未创新高
            if price_high and not psy_high and data["psy"].iloc[i] < data["psy"].iloc[i-5]:
                data.iloc[i, data.columns.get_loc("psy_divergence")] = -1
            
            # 底背离：价格新低但PSY未创新低
            elif price_low and not psy_low and data["psy"].iloc[i] > data["psy"].iloc[i-5]:
                data.iloc[i, data.columns.get_loc("psy_divergence")] = 1
            
            # 无背离
            else:
                data.iloc[i, data.columns.get_loc("psy_divergence")] = 0
        
        return data
    
    def get_market_status(self, data: pd.DataFrame, overbought: float = 75, oversold: float = 25) -> pd.DataFrame:
        """
        获取市场状态
        
        Args:
            data: 输入数据，包含PSY指标
            overbought: 超买阈值，默认为75
            oversold: 超卖阈值，默认为25
            
        Returns:
            pd.DataFrame: 包含市场状态的数据框
        """
        if "psy" not in data.columns:
            data = self.calculate(data)
        
        # 初始化状态列
        data["market_status"] = np.nan
        
        # 判断市场状态
        for i in range(len(data)):
            if pd.notna(data["psy"].iloc[i]):
                # 超买区域
                if data["psy"].iloc[i] > overbought:
                    data.iloc[i, data.columns.get_loc("market_status")] = "超买"
                
                # 超卖区域
                elif data["psy"].iloc[i] < oversold:
                    data.iloc[i, data.columns.get_loc("market_status")] = "超卖"
                
                # 中性区域靠上
                elif data["psy"].iloc[i] >= 50:
                    data.iloc[i, data.columns.get_loc("market_status")] = "中性偏多"
                
                # 中性区域靠下
                else:
                    data.iloc[i, data.columns.get_loc("market_status")] = "中性偏空"
        
        return data

    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算PSY原始评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 原始评分序列（0-100分）
        """
        # 确保已计算PSY
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)
        
        score = pd.Series(50.0, index=data.index)  # 基础分50分
        
        # 1. 超买超卖评分
        score += self._calculate_psy_overbought_oversold_score()
        
        # 2. PSY与信号线交叉评分
        score += self._calculate_psy_ma_cross_score()
        
        # 3. PSY趋势评分
        score += self._calculate_psy_trend_score()
        
        # 4. PSY背离评分
        score += self._calculate_psy_divergence_score(data)
        
        return np.clip(score, 0, 100)
    
    def _calculate_psy_overbought_oversold_score(self) -> pd.Series:
        """
        计算PSY超买超卖评分
        
        Returns:
            pd.Series: 超买超卖评分
        """
        score = pd.Series(0.0, index=self._result.index)
        
        psy = self._result["psy"]
        
        # 超买区域（得分随PSY增加而降低）
        overbought_score = -1 * np.maximum(0, (psy - 75)) * 0.6
        score += overbought_score
        
        # 超卖区域（得分随PSY降低而增加）
        oversold_score = np.maximum(0, (25 - psy)) * 0.6
        score += oversold_score
        
        # 中性区域上方（小幅加分）
        neutral_high = (psy > 50) & (psy < 75)
        score.loc[neutral_high] += (psy.loc[neutral_high] - 50) * 0.2
        
        # 中性区域下方（小幅减分）
        neutral_low = (psy < 50) & (psy > 25)
        score.loc[neutral_low] -= (50 - psy.loc[neutral_low]) * 0.2
        
        return score
    
    def _calculate_psy_ma_cross_score(self) -> pd.Series:
        """
        计算PSY与信号线交叉评分
        
        Returns:
            pd.Series: 交叉评分
        """
        score = pd.Series(0.0, index=self._result.index)
        
        psy = self._result["psy"]
        psyma = self._result["psyma"]
        
        # PSY上穿信号线
        golden_cross = (psy > psyma) & (psy.shift(1) <= psyma.shift(1))
        score.loc[golden_cross] += 10
        
        # PSY下穿信号线
        death_cross = (psy < psyma) & (psy.shift(1) >= psyma.shift(1))
        score.loc[death_cross] -= 10
        
        # PSY位于信号线上方
        above_ma = psy > psyma
        score.loc[above_ma] += 5
        
        # PSY位于信号线下方
        below_ma = psy < psyma
        score.loc[below_ma] -= 5
        
        return score
    
    def _calculate_psy_trend_score(self) -> pd.Series:
        """
        计算PSY趋势评分
        
        Returns:
            pd.Series: 趋势评分
        """
        score = pd.Series(0.0, index=self._result.index)
        
        psy = self._result["psy"]
        psy_change = self._result["psy_change"]
        
        # PSY上升
        rising = psy_change > 0
        score.loc[rising] += psy_change.loc[rising] * 0.5
        
        # PSY下降
        falling = psy_change < 0
        score.loc[falling] += psy_change.loc[falling] * 0.5
        
        return score
    
    def _calculate_psy_divergence_score(self, data: pd.DataFrame) -> pd.Series:
        """
        计算PSY背离评分
        
        Args:
            data: 输入数据
            
        Returns:
            pd.Series: 背离评分
        """
        score = pd.Series(0.0, index=data.index)
        
        # 使用get_signals方法中的背离识别逻辑
        signals = self.get_signals(data)
        
        # 底背离加分
        bullish_divergence = signals["psy_divergence"] == 1
        score.loc[bullish_divergence] += 20
        
        # 顶背离减分
        bearish_divergence = signals["psy_divergence"] == -1
        score.loc[bearish_divergence] -= 20
        
        return score
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别PSY技术形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别出的形态列表
        """
        patterns = []
        
        # 确保已计算PSY
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return patterns
        
        # 获取最近的PSY值
        if len(self._result) < 5:
            return patterns
        
        recent_psy = self._result["psy"].iloc[-5:]
        current_psy = recent_psy.iloc[-1]
        
        # 1. 超买超卖形态
        if current_psy > 75:
            patterns.append("PSY超买区域(>75)")
            
            # 检测是否继续走高
            if current_psy > recent_psy.iloc[-2] and recent_psy.iloc[-2] > recent_psy.iloc[-3]:
                patterns.append("PSY超买区域继续走高")
            # 检测是否从超买区回落
            elif current_psy < recent_psy.iloc[-2]:
                patterns.append("PSY从超买区回落")
        
        elif current_psy < 25:
            patterns.append("PSY超卖区域(<25)")
            
            # 检测是否继续走低
            if current_psy < recent_psy.iloc[-2] and recent_psy.iloc[-2] < recent_psy.iloc[-3]:
                patterns.append("PSY超卖区域继续走低")
            # 检测是否从超卖区回升
            elif current_psy > recent_psy.iloc[-2]:
                patterns.append("PSY从超卖区回升")
        
        # 2. 中性区域形态
        else:
            # 中性区域偏多
            if current_psy > 50:
                patterns.append("PSY中性区域偏多")
            # 中性区域偏空
            else:
                patterns.append("PSY中性区域偏空")
            
            # 检测穿越超买超卖阈值
            if current_psy < 75 and recent_psy.iloc[-2] >= 75:
                patterns.append("PSY下穿超买阈值(75)")
            elif current_psy > 25 and recent_psy.iloc[-2] <= 25:
                patterns.append("PSY上穿超卖阈值(25)")
        
        # 3. 信号线交叉形态
        if current_psy > self._result["psyma"].iloc[-1] and recent_psy.iloc[-2] <= self._result["psyma"].iloc[-2]:
            patterns.append("PSY金叉信号线")
        elif current_psy < self._result["psyma"].iloc[-1] and recent_psy.iloc[-2] >= self._result["psyma"].iloc[-2]:
            patterns.append("PSY死叉信号线")
        
        # 4. 均值回归形态
        psy_mean = recent_psy.mean()
        if abs(current_psy - 50) < abs(recent_psy.iloc[-2] - 50) and abs(recent_psy.iloc[-2] - 50) > 15:
            patterns.append("PSY均值回归")
        
        # 5. 极值反转形态
        if current_psy > recent_psy.iloc[-2] and recent_psy.iloc[-2] < recent_psy.iloc[-3] and recent_psy.iloc[-2] < recent_psy.iloc[-4]:
            patterns.append("PSY触底反弹")
        elif current_psy < recent_psy.iloc[-2] and recent_psy.iloc[-2] > recent_psy.iloc[-3] and recent_psy.iloc[-2] > recent_psy.iloc[-4]:
            patterns.append("PSY高位回落")
        
        return patterns
    
    def generate_signals(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        生成PSY指标标准化交易信号
        
        Args:
            data: 输入数据，包含收盘价
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            pd.DataFrame: 信号结果DataFrame，包含标准化信号
        """
        # 确保已计算PSY指标
        if not self.has_result():
            self.calculate(data)
        
        # 初始化信号DataFrame
        signals = pd.DataFrame(index=data.index)
        signals['buy_signal'] = False
        signals['sell_signal'] = False
        signals['neutral_signal'] = True  # 默认为中性信号
        signals['trend'] = 0  # 0表示中性
        signals['score'] = 50.0  # 默认评分50分
        signals['signal_type'] = None
        signals['signal_desc'] = None
        signals['confidence'] = 50.0
        
        # 计算评分
        score = self.calculate_raw_score(data, **kwargs)
        signals['score'] = score
        
        # 获取PSY数据
        psy = self._result['psy']
        psyma = self._result['psyma']
        
        # 1. PSY从超卖区上穿25，买入信号
        psy_cross_oversold = (psy > 25) & (psy.shift(1) <= 25)
        signals.loc[psy_cross_oversold, 'buy_signal'] = True
        signals.loc[psy_cross_oversold, 'neutral_signal'] = False
        signals.loc[psy_cross_oversold, 'trend'] = 1
        signals.loc[psy_cross_oversold, 'signal_type'] = 'PSY超卖反弹'
        signals.loc[psy_cross_oversold, 'signal_desc'] = 'PSY从超卖区上穿25，买入信号'
        signals.loc[psy_cross_oversold, 'confidence'] = 70.0
        
        # 2. PSY从超买区下穿75，卖出信号
        psy_cross_overbought = (psy < 75) & (psy.shift(1) >= 75)
        signals.loc[psy_cross_overbought, 'sell_signal'] = True
        signals.loc[psy_cross_overbought, 'neutral_signal'] = False
        signals.loc[psy_cross_overbought, 'trend'] = -1
        signals.loc[psy_cross_overbought, 'signal_type'] = 'PSY超买回落'
        signals.loc[psy_cross_overbought, 'signal_desc'] = 'PSY从超买区下穿75，卖出信号'
        signals.loc[psy_cross_overbought, 'confidence'] = 70.0
        
        # 3. PSY上穿信号线，轻微买入信号
        psy_cross_psyma_up = (psy > psyma) & (psy.shift(1) <= psyma.shift(1))
        signals.loc[psy_cross_psyma_up, 'buy_signal'] = True
        signals.loc[psy_cross_psyma_up, 'neutral_signal'] = False
        signals.loc[psy_cross_psyma_up, 'trend'] = 0.5  # 轻微看涨
        signals.loc[psy_cross_psyma_up, 'signal_type'] = 'PSY金叉信号线'
        signals.loc[psy_cross_psyma_up, 'signal_desc'] = 'PSY上穿信号线，轻微买入信号'
        signals.loc[psy_cross_psyma_up, 'confidence'] = 60.0
        
        # 4. PSY下穿信号线，轻微卖出信号
        psy_cross_psyma_down = (psy < psyma) & (psy.shift(1) >= psyma.shift(1))
        signals.loc[psy_cross_psyma_down, 'sell_signal'] = True
        signals.loc[psy_cross_psyma_down, 'neutral_signal'] = False
        signals.loc[psy_cross_psyma_down, 'trend'] = -0.5  # 轻微看跌
        signals.loc[psy_cross_psyma_down, 'signal_type'] = 'PSY死叉信号线'
        signals.loc[psy_cross_psyma_down, 'signal_desc'] = 'PSY下穿信号线，轻微卖出信号'
        signals.loc[psy_cross_psyma_down, 'confidence'] = 60.0
        
        # 5. 根据得分产生强弱信号
        strong_buy = score > 80
        signals.loc[strong_buy, 'buy_signal'] = True
        signals.loc[strong_buy, 'neutral_signal'] = False
        signals.loc[strong_buy, 'trend'] = 1
        signals.loc[strong_buy, 'signal_type'] = 'PSY强烈买入'
        signals.loc[strong_buy, 'signal_desc'] = 'PSY综合评分超过80，强烈买入信号'
        signals.loc[strong_buy, 'confidence'] = 85.0
        
        strong_sell = score < 20
        signals.loc[strong_sell, 'sell_signal'] = True
        signals.loc[strong_sell, 'neutral_signal'] = False
        signals.loc[strong_sell, 'trend'] = -1
        signals.loc[strong_sell, 'signal_type'] = 'PSY强烈卖出'
        signals.loc[strong_sell, 'signal_desc'] = 'PSY综合评分低于20，强烈卖出信号'
        signals.loc[strong_sell, 'confidence'] = 85.0
        
        return signals 