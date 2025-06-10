#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
资金流向指标(MFI)模块

实现资金流向指标计算功能，用于识别价格反转点
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any

from indicators.base_indicator import BaseIndicator
from indicators.common import crossover, crossunder
from utils.logger import get_logger

logger = get_logger(__name__)


class MFI(BaseIndicator):
    """
    资金流向指标(Money Flow Index)
    
    计算资金流入和流出比率，结合价格与成交量判断超买超卖情况，用于识别价格反转点
    """
    
    def __init__(self, period: int = 14):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """
        初始化MFI指标
        
        Args:
            period: 计算周期，默认为14日
        """
        super().__init__(name="MFI", description="资金流向指标，计算资金流入和流出比率，判断超买超卖")
        self.period = period
    
    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算MFI指标
        
        Args:
            data: 输入数据，包含OHLCV数据
            
        Returns:
            pd.DataFrame: 计算结果，包含MFI指标值
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["high", "low", "close", "volume"])
        
        # 初始化结果数据框
        result = data.copy()
        
        # 计算典型价格: (high + low + close) / 3
        typical_price = (data["high"] + data["low"] + data["close"]) / 3
        
        # 计算资金流: 典型价格 * 成交量
        money_flow = typical_price * data["volume"]
        
        # 初始化正向资金流和负向资金流
        positive_flow = np.zeros(len(data))
        negative_flow = np.zeros(len(data))
        
        # 计算正向和负向资金流
        for i in range(1, len(data)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:  # 价格上涨
                positive_flow[i] = money_flow.iloc[i]
            elif typical_price.iloc[i] < typical_price.iloc[i-1]:  # 价格下跌
                negative_flow[i] = money_flow.iloc[i]
            else:  # 价格不变
                if money_flow.iloc[i] > money_flow.iloc[i-1]:
                    positive_flow[i] = money_flow.iloc[i]
                else:
                    negative_flow[i] = money_flow.iloc[i]
        
        # 计算周期内的正向和负向资金流总和
        positive_flow_sum = pd.Series(positive_flow).rolling(window=self.period).sum()
        negative_flow_sum = pd.Series(negative_flow).rolling(window=self.period).sum()
        
        # 计算资金比率: 正向资金流 / 负向资金流
        money_ratio = np.where(negative_flow_sum != 0, positive_flow_sum / negative_flow_sum, 100)
        
        # 计算MFI: 100 - 100 / (1 + 资金比率)
        mfi = 100 - (100 / (1 + money_ratio))
        
        # 添加到结果
        result["mfi"] = mfi
        
        # 额外计算：MFI变化率
        result["mfi_change"] = result["mfi"].diff()
        
        # 存储结果
        self._result = result
        
        return result
    
    def get_signals(self, data: pd.DataFrame, overbought: float = 80, oversold: float = 20) -> pd.DataFrame:
        """
        生成MFI信号
        
        Args:
            data: 输入数据，包含MFI指标
            overbought: 超买阈值，默认为80
            oversold: 超卖阈值，默认为20
            
        Returns:
            pd.DataFrame: 包含MFI信号的数据框
        """
        if "mfi" not in data.columns:
            data = self.calculate(data)
        
        # 初始化信号列
        data["mfi_signal"] = np.nan
        
        # 生成信号
        for i in range(1, len(data)):
            if pd.notna(data["mfi"].iloc[i]) and pd.notna(data["mfi"].iloc[i-1]):
                # MFI下穿超买线：卖出信号
                if data["mfi"].iloc[i] < overbought and data["mfi"].iloc[i-1] >= overbought:
                    data.iloc[i, data.columns.get_loc("mfi_signal")] = -1
                
                # MFI上穿超卖线：买入信号
                elif data["mfi"].iloc[i] > oversold and data["mfi"].iloc[i-1] <= oversold:
                    data.iloc[i, data.columns.get_loc("mfi_signal")] = 1
                
                # 无信号
                else:
                    data.iloc[i, data.columns.get_loc("mfi_signal")] = 0
        
        # 检测MFI背离
        data["mfi_divergence"] = np.nan
        window = 20  # 背离检测窗口
        
        for i in range(window, len(data)):
            # 价格新高/新低检测
            price_high = data["close"].iloc[i] >= np.max(data["close"].iloc[i-window:i])
            price_low = data["close"].iloc[i] <= np.min(data["close"].iloc[i-window:i])
            
            # MFI新高/新低检测
            mfi_high = data["mfi"].iloc[i] >= np.max(data["mfi"].iloc[i-window:i])
            mfi_low = data["mfi"].iloc[i] <= np.min(data["mfi"].iloc[i-window:i])
            
            # 顶背离：价格新高但MFI未创新高
            if price_high and not mfi_high and data["mfi"].iloc[i] < data["mfi"].iloc[i-5]:
                data.iloc[i, data.columns.get_loc("mfi_divergence")] = -1
            
            # 底背离：价格新低但MFI未创新低
            elif price_low and not mfi_low and data["mfi"].iloc[i] > data["mfi"].iloc[i-5]:
                data.iloc[i, data.columns.get_loc("mfi_divergence")] = 1
            
            # 无背离
            else:
                data.iloc[i, data.columns.get_loc("mfi_divergence")] = 0
        
        return data
    
    def get_market_status(self, data: pd.DataFrame, overbought: float = 80, oversold: float = 20) -> pd.DataFrame:
        """
        获取市场状态
        
        Args:
            data: 输入数据，包含MFI指标
            overbought: 超买阈值，默认为80
            oversold: 超卖阈值，默认为20
            
        Returns:
            pd.DataFrame: 包含市场状态的数据框
        """
        if "mfi" not in data.columns:
            data = self.calculate(data)
        
        # 初始化状态列
        data["market_status"] = np.nan
        
        # 判断市场状态
        for i in range(len(data)):
            if pd.notna(data["mfi"].iloc[i]):
                # 超买区域
                if data["mfi"].iloc[i] > overbought:
                    data.iloc[i, data.columns.get_loc("market_status")] = "超买"
                
                # 超卖区域
                elif data["mfi"].iloc[i] < oversold:
                    data.iloc[i, data.columns.get_loc("market_status")] = "超卖"
                
                # 中性区域靠上
                elif data["mfi"].iloc[i] >= 50:
                    data.iloc[i, data.columns.get_loc("market_status")] = "中性偏多"
                
                # 中性区域靠下
                else:
                    data.iloc[i, data.columns.get_loc("market_status")] = "中性偏空"
        
        return data

    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算MFI原始评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 原始评分序列（0-100分）
        """
        # 确保已计算MFI
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)
        
        score = pd.Series(50.0, index=data.index)  # 基础分50分
        
        # 1. MFI超买超卖评分
        overbought_oversold_score = self._calculate_mfi_overbought_oversold_score()
        score += overbought_oversold_score
        
        # 2. MFI背离评分
        divergence_score = self._calculate_mfi_divergence_score(data)
        score += divergence_score
        
        # 3. MFI中线穿越评分
        midline_cross_score = self._calculate_mfi_midline_cross_score()
        score += midline_cross_score
        
        # 4. MFI趋势评分
        trend_score = self._calculate_mfi_trend_score()
        score += trend_score
        
        # 5. MFI强度评分
        strength_score = self._calculate_mfi_strength_score()
        score += strength_score
        
        return np.clip(score, 0, 100)
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别MFI技术形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别出的形态列表
        """
        patterns = []
        
        # 确保已计算MFI
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return patterns
        
        # 1. 检测MFI超买超卖形态
        overbought_oversold_patterns = self._detect_mfi_overbought_oversold_patterns()
        patterns.extend(overbought_oversold_patterns)
        
        # 2. 检测MFI背离形态
        divergence_patterns = self._detect_mfi_divergence_patterns(data)
        patterns.extend(divergence_patterns)
        
        # 3. 检测MFI中线穿越形态
        midline_cross_patterns = self._detect_mfi_midline_cross_patterns()
        patterns.extend(midline_cross_patterns)
        
        # 4. 检测MFI趋势形态
        trend_patterns = self._detect_mfi_trend_patterns()
        patterns.extend(trend_patterns)
        
        # 5. 检测MFI强度形态
        strength_patterns = self._detect_mfi_strength_patterns()
        patterns.extend(strength_patterns)
        
        return patterns
    
    def _calculate_mfi_overbought_oversold_score(self) -> pd.Series:
        """
        计算MFI超买超卖评分
        
        Returns:
            pd.Series: 超买超卖评分
        """
        overbought_oversold_score = pd.Series(0.0, index=self._result.index)
        
        mfi_values = self._result['mfi']
        
        # MFI超卖区域（MFI < 20）+20分
        oversold_condition = mfi_values < 20
        overbought_oversold_score += oversold_condition * 20
        
        # MFI超买区域（MFI > 80）-20分
        overbought_condition = mfi_values > 80
        overbought_oversold_score -= overbought_condition * 20
        
        # MFI上穿20+25分
        mfi_cross_up_20 = crossover(mfi_values, 20)
        overbought_oversold_score += mfi_cross_up_20 * 25
        
        # MFI下穿80-25分
        mfi_cross_down_80 = crossunder(mfi_values, 80)
        overbought_oversold_score -= mfi_cross_down_80 * 25
        
        # MFI极度超卖（MFI < 10）额外+15分
        extreme_oversold = mfi_values < 10
        overbought_oversold_score += extreme_oversold * 15
        
        # MFI极度超买（MFI > 90）额外-15分
        extreme_overbought = mfi_values > 90
        overbought_oversold_score -= extreme_overbought * 15
        
        return overbought_oversold_score
    
    def _calculate_mfi_divergence_score(self, data: pd.DataFrame) -> pd.Series:
        """
        计算MFI背离评分
        
        Args:
            data: 价格数据
            
        Returns:
            pd.Series: 背离评分
        """
        divergence_score = pd.Series(0.0, index=self._result.index)
        
        if 'close' not in data.columns:
            return divergence_score
        
        close_price = data['close']
        mfi_values = self._result['mfi']
        
        # 简化的背离检测
        if len(close_price) >= 20:
            # 检查最近20个周期的价格和MFI趋势
            recent_periods = 20
            
            for i in range(recent_periods, len(close_price)):
                # 寻找最近的价格和MFI峰值/谷值
                price_window = close_price.iloc[i-recent_periods:i+1]
                mfi_window = mfi_values.iloc[i-recent_periods:i+1]
                
                # 检查是否为价格新高/新低
                current_price = close_price.iloc[i]
                current_mfi = mfi_values.iloc[i]
                
                price_is_high = current_price >= price_window.max()
                price_is_low = current_price <= price_window.min()
                mfi_is_high = current_mfi >= mfi_window.max()
                mfi_is_low = current_mfi <= mfi_window.min()
                
                # 正背离：价格创新低但MFI未创新低
                if price_is_low and not mfi_is_low:
                    divergence_score.iloc[i] += 30
                
                # 负背离：价格创新高但MFI未创新高
                elif price_is_high and not mfi_is_high:
                    divergence_score.iloc[i] -= 30
        
        return divergence_score
    
    def _calculate_mfi_midline_cross_score(self) -> pd.Series:
        """
        计算MFI中线穿越评分
        
        Returns:
            pd.Series: 中线穿越评分
        """
        midline_cross_score = pd.Series(0.0, index=self._result.index)
        
        mfi_values = self._result['mfi']
        
        # MFI上穿50+15分
        mfi_cross_up_50 = crossover(mfi_values, 50)
        midline_cross_score += mfi_cross_up_50 * 15
        
        # MFI下穿50-15分
        mfi_cross_down_50 = crossunder(mfi_values, 50)
        midline_cross_score -= mfi_cross_down_50 * 15
        
        # MFI在50上方+5分
        mfi_above_50 = mfi_values > 50
        midline_cross_score += mfi_above_50 * 5
        
        # MFI在50下方-5分
        mfi_below_50 = mfi_values < 50
        midline_cross_score -= mfi_below_50 * 5
        
        return midline_cross_score
    
    def _calculate_mfi_trend_score(self) -> pd.Series:
        """
        计算MFI趋势评分
        
        Returns:
            pd.Series: 趋势评分
        """
        trend_score = pd.Series(0.0, index=self._result.index)
        
        mfi_values = self._result['mfi']
        
        # MFI上升趋势+10分
        mfi_rising = mfi_values > mfi_values.shift(1)
        trend_score += mfi_rising * 10
        
        # MFI下降趋势-10分
        mfi_falling = mfi_values < mfi_values.shift(1)
        trend_score -= mfi_falling * 10
        
        # MFI连续上升（3个周期）+15分
        if len(mfi_values) >= 3:
            consecutive_rising = (
                (mfi_values > mfi_values.shift(1)) &
                (mfi_values.shift(1) > mfi_values.shift(2)) &
                (mfi_values.shift(2) > mfi_values.shift(3))
            )
            trend_score += consecutive_rising * 15
        
        # MFI连续下降（3个周期）-15分
        if len(mfi_values) >= 3:
            consecutive_falling = (
                (mfi_values < mfi_values.shift(1)) &
                (mfi_values.shift(1) < mfi_values.shift(2)) &
                (mfi_values.shift(2) < mfi_values.shift(3))
            )
            trend_score -= consecutive_falling * 15
        
        return trend_score
    
    def _calculate_mfi_strength_score(self) -> pd.Series:
        """
        计算MFI强度评分
        
        Returns:
            pd.Series: 强度评分
        """
        strength_score = pd.Series(0.0, index=self._result.index)
        
        mfi_values = self._result['mfi']
        
        # 计算MFI变化幅度
        mfi_change = mfi_values.diff()
        
        # MFI大幅上升（变化>10）+12分
        large_rise = mfi_change > 10
        strength_score += large_rise * 12
        
        # MFI大幅下降（变化<-10）-12分
        large_fall = mfi_change < -10
        strength_score -= large_fall * 12
        
        # MFI快速变化（绝对值>15）额外±8分
        rapid_change = np.abs(mfi_change) > 15
        rapid_change_direction = np.sign(mfi_change)
        strength_score += rapid_change * rapid_change_direction * 8
        
        return strength_score
    
    def _detect_mfi_overbought_oversold_patterns(self) -> List[str]:
        """
        检测MFI超买超卖形态
        
        Returns:
            List[str]: 超买超卖形态列表
        """
        patterns = []
        
        mfi_values = self._result['mfi']
        
        if len(mfi_values) > 0:
            current_mfi = mfi_values.iloc[-1]
            
            if pd.isna(current_mfi):
                return patterns
            
            if current_mfi < 10:
                patterns.append("MFI极度超卖")
            elif current_mfi < 20:
                patterns.append("MFI超卖")
            elif current_mfi > 90:
                patterns.append("MFI极度超买")
            elif current_mfi > 80:
                patterns.append("MFI超买")
            elif 45 <= current_mfi <= 55:
                patterns.append("MFI中性区域")
        
        # 检查最近的阈值穿越
        recent_periods = min(5, len(mfi_values))
        recent_mfi = mfi_values.tail(recent_periods)
        
        if crossover(recent_mfi, 20).any():
            patterns.append("MFI上穿超卖线")
        
        if crossunder(recent_mfi, 80).any():
            patterns.append("MFI下穿超买线")
        
        return patterns
    
    def _detect_mfi_divergence_patterns(self, data: pd.DataFrame) -> List[str]:
        """
        检测MFI背离形态
        
        Args:
            data: 价格数据
            
        Returns:
            List[str]: 背离形态列表
        """
        patterns = []
        
        if 'close' not in data.columns:
            return patterns
        
        close_price = data['close']
        mfi_values = self._result['mfi']
        
        if len(close_price) >= 20:
            # 检查最近20个周期的趋势
            recent_price = close_price.tail(20)
            recent_mfi = mfi_values.tail(20)
            
            # 简化的背离检测
            price_trend = recent_price.iloc[-1] - recent_price.iloc[0]
            mfi_trend = recent_mfi.iloc[-1] - recent_mfi.iloc[0]
            
            # 背离检测
            if price_trend < -0.02 and mfi_trend > 2:  # 价格下跌但MFI上升
                patterns.append("MFI正背离")
            elif price_trend > 0.02 and mfi_trend < -2:  # 价格上涨但MFI下降
                patterns.append("MFI负背离")
            elif abs(price_trend) < 0.01 and abs(mfi_trend) < 1:
                patterns.append("MFI价格同步")
        
        return patterns
    
    def _detect_mfi_midline_cross_patterns(self) -> List[str]:
        """
        检测MFI中线穿越形态
        
        Returns:
            List[str]: 中线穿越形态列表
        """
        patterns = []
        
        mfi_values = self._result['mfi']
        
        # 检查最近的中线穿越
        recent_periods = min(5, len(mfi_values))
        recent_mfi = mfi_values.tail(recent_periods)
        
        if crossover(recent_mfi, 50).any():
            patterns.append("MFI上穿中线")
        
        if crossunder(recent_mfi, 50).any():
            patterns.append("MFI下穿中线")
        
        # 检查当前位置
        if len(mfi_values) > 0:
            current_mfi = mfi_values.iloc[-1]
            if not pd.isna(current_mfi):
                if current_mfi > 50:
                    patterns.append("MFI中线上方")
                elif current_mfi < 50:
                    patterns.append("MFI中线下方")
                else:
                    patterns.append("MFI中线位置")
        
        return patterns
    
    def _detect_mfi_trend_patterns(self) -> List[str]:
        """
        检测MFI趋势形态
        
        Returns:
            List[str]: 趋势形态列表
        """
        patterns = []
        
        mfi_values = self._result['mfi']
        
        # 检查MFI趋势
        if len(mfi_values) >= 3:
            recent_3 = mfi_values.tail(3)
            if len(recent_3) == 3 and not recent_3.isna().any():
                if (recent_3.iloc[2] > recent_3.iloc[1] > recent_3.iloc[0]):
                    patterns.append("MFI连续上升")
                elif (recent_3.iloc[2] < recent_3.iloc[1] < recent_3.iloc[0]):
                    patterns.append("MFI连续下降")
        
        # 检查当前趋势
        if len(mfi_values) >= 2:
            current_mfi = mfi_values.iloc[-1]
            prev_mfi = mfi_values.iloc[-2]
            
            if not pd.isna(current_mfi) and not pd.isna(prev_mfi):
                if current_mfi > prev_mfi:
                    patterns.append("MFI上升")
                elif current_mfi < prev_mfi:
                    patterns.append("MFI下降")
                else:
                    patterns.append("MFI平稳")
        
        return patterns
    
    def _detect_mfi_strength_patterns(self) -> List[str]:
        """
        检测MFI强度形态
        
        Returns:
            List[str]: 强度形态列表
        """
        patterns = []
        
        mfi_values = self._result['mfi']
        
        if len(mfi_values) >= 2:
            current_mfi = mfi_values.iloc[-1]
            prev_mfi = mfi_values.iloc[-2]
            
            if not pd.isna(current_mfi) and not pd.isna(prev_mfi):
                mfi_change = current_mfi - prev_mfi
                
                if mfi_change > 15:
                    patterns.append("MFI急速上升")
                elif mfi_change > 10:
                    patterns.append("MFI大幅上升")
                elif mfi_change < -15:
                    patterns.append("MFI急速下降")
                elif mfi_change < -10:
                    patterns.append("MFI大幅下降")
                elif abs(mfi_change) <= 2:
                    patterns.append("MFI变化平缓")
        
        return patterns

    def generate_signals(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        生成MFI指标标准化交易信号
        
        Args:
            data: 输入数据，包含OHLCV数据
            *args: 位置参数
            **kwargs: 关键字参数
                overbought: 超买阈值，默认为80
                oversold: 超卖阈值，默认为20
                
        Returns:
            pd.DataFrame: 信号结果DataFrame，包含标准化信号
        """
        # 提取参数
        overbought = kwargs.get('overbought', 80)
        oversold = kwargs.get('oversold', 20)
        
        # 确保已计算MFI指标
        if not self.has_result():
            self.calculate(data)
        
        # 获取MFI值
        mfi = self._result['mfi']
        
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
        signals['risk_level'] = '中'
        signals['position_size'] = 0.0
        signals['stop_loss'] = None
        signals['market_env'] = '中性'
        signals['volume_confirmation'] = False
        
        # 计算ATR用于止损设置
        try:
            from indicators.atr import ATR
            atr_indicator = ATR()
            atr_data = atr_indicator.calculate(data)
            atr_values = atr_data['atr']
        except Exception as e:
            logger.warning(f"计算ATR失败: {e}")
            atr_values = pd.Series(0, index=data.index)
        
        # 生成信号
        for i in range(5, len(data)):
            # 超买超卖信号
            if mfi.iloc[i] < oversold:
                # 超卖区域
                signals.loc[data.index[i], 'score'] += 20
                signals.loc[data.index[i], 'signal_desc'] = f'MFI位于超卖区域: {mfi.iloc[i]:.2f}'
                
                # 如果从超卖区域上穿，生成买入信号
                if i > 0 and mfi.iloc[i-1] <= oversold and mfi.iloc[i] > oversold:
                    signals.loc[data.index[i], 'buy_signal'] = True
                    signals.loc[data.index[i], 'neutral_signal'] = False
                    signals.loc[data.index[i], 'trend'] = 1
                    signals.loc[data.index[i], 'score'] += 10  # 额外加分
                    signals.loc[data.index[i], 'signal_type'] = 'MFI超卖反弹'
                    signals.loc[data.index[i], 'signal_desc'] = f'MFI从超卖区域上穿: {mfi.iloc[i]:.2f}'
                    signals.loc[data.index[i], 'confidence'] = 70.0
                    
            elif mfi.iloc[i] > overbought:
                # 超买区域
                signals.loc[data.index[i], 'score'] -= 20
                signals.loc[data.index[i], 'signal_desc'] = f'MFI位于超买区域: {mfi.iloc[i]:.2f}'
                
                # 如果从超买区域下穿，生成卖出信号
                if i > 0 and mfi.iloc[i-1] >= overbought and mfi.iloc[i] < overbought:
                    signals.loc[data.index[i], 'sell_signal'] = True
                    signals.loc[data.index[i], 'neutral_signal'] = False
                    signals.loc[data.index[i], 'trend'] = -1
                    signals.loc[data.index[i], 'score'] -= 10  # 额外减分
                    signals.loc[data.index[i], 'signal_type'] = 'MFI超买回落'
                    signals.loc[data.index[i], 'signal_desc'] = f'MFI从超买区域下穿: {mfi.iloc[i]:.2f}'
                    signals.loc[data.index[i], 'confidence'] = 70.0
            
            # 中线穿越信号
            if i > 0 and mfi.iloc[i-1] < 50 and mfi.iloc[i] >= 50:
                signals.loc[data.index[i], 'score'] += 15
                signals.loc[data.index[i], 'trend'] = 1
                if not signals.loc[data.index[i], 'buy_signal']:
                    signals.loc[data.index[i], 'signal_desc'] = f'MFI上穿中线: {mfi.iloc[i]:.2f}'
                    signals.loc[data.index[i], 'signal_type'] = 'MFI中线上穿'
                    
            elif i > 0 and mfi.iloc[i-1] > 50 and mfi.iloc[i] <= 50:
                signals.loc[data.index[i], 'score'] -= 15
                signals.loc[data.index[i], 'trend'] = -1
                if not signals.loc[data.index[i], 'sell_signal']:
                    signals.loc[data.index[i], 'signal_desc'] = f'MFI下穿中线: {mfi.iloc[i]:.2f}'
                    signals.loc[data.index[i], 'signal_type'] = 'MFI中线下穿'
            
            # 背离信号
            # 价格创新高但MFI未创新高 - 顶背离(看跌)
            price_window = data['close'].iloc[i-20:i+1]
            mfi_window = mfi.iloc[i-20:i+1]
            
            if data['close'].iloc[i] >= price_window.max() and mfi.iloc[i] < mfi_window.max() * 0.95:
                signals.loc[data.index[i], 'sell_signal'] = True
                signals.loc[data.index[i], 'neutral_signal'] = False
                signals.loc[data.index[i], 'trend'] = -1
                signals.loc[data.index[i], 'score'] -= 25
                signals.loc[data.index[i], 'signal_type'] = 'MFI顶背离'
                signals.loc[data.index[i], 'signal_desc'] = '价格创新高但MFI未创新高，可能见顶'
                signals.loc[data.index[i], 'confidence'] = 80.0
                signals.loc[data.index[i], 'risk_level'] = '高'
                
            # 价格创新低但MFI未创新低 - 底背离(看涨)
            elif data['close'].iloc[i] <= price_window.min() and mfi.iloc[i] > mfi_window.min() * 1.05:
                signals.loc[data.index[i], 'buy_signal'] = True
                signals.loc[data.index[i], 'neutral_signal'] = False
                signals.loc[data.index[i], 'trend'] = 1
                signals.loc[data.index[i], 'score'] += 25
                signals.loc[data.index[i], 'signal_type'] = 'MFI底背离'
                signals.loc[data.index[i], 'signal_desc'] = '价格创新低但MFI未创新低，可能见底'
                signals.loc[data.index[i], 'confidence'] = 80.0
                signals.loc[data.index[i], 'risk_level'] = '高'
            
            # MFI急剧变化信号
            if i > 5:
                mfi_change = mfi.iloc[i] - mfi.iloc[i-5]
                if mfi_change > 30:  # MFI在5个周期内上涨超过30
                    signals.loc[data.index[i], 'buy_signal'] = True
                    signals.loc[data.index[i], 'neutral_signal'] = False
                    signals.loc[data.index[i], 'trend'] = 1
                    signals.loc[data.index[i], 'score'] += 15
                    signals.loc[data.index[i], 'signal_type'] = 'MFI快速上涨'
                    signals.loc[data.index[i], 'signal_desc'] = f'MFI在5个周期内上涨{mfi_change:.2f}点，资金流入加速'
                    signals.loc[data.index[i], 'confidence'] = 65.0
                
                elif mfi_change < -30:  # MFI在5个周期内下跌超过30
                    signals.loc[data.index[i], 'sell_signal'] = True
                    signals.loc[data.index[i], 'neutral_signal'] = False
                    signals.loc[data.index[i], 'trend'] = -1
                    signals.loc[data.index[i], 'score'] -= 15
                    signals.loc[data.index[i], 'signal_type'] = 'MFI快速下跌'
                    signals.loc[data.index[i], 'signal_desc'] = f'MFI在5个周期内下跌{-mfi_change:.2f}点，资金流出加速'
                    signals.loc[data.index[i], 'confidence'] = 65.0
            
            # 设置仓位大小
            if signals.loc[data.index[i], 'buy_signal']:
                signals.loc[data.index[i], 'position_size'] = min(0.3, (signals.loc[data.index[i], 'score'] - 50) / 100)
                # 止损设置为当前价格的2.5个ATR
                signals.loc[data.index[i], 'stop_loss'] = data['close'].iloc[i] - 2.5 * atr_values.iloc[i]
                # 判断成交量确认
                if i > 0 and data['volume'].iloc[i] > data['volume'].iloc[i-1] * 1.2:
                    signals.loc[data.index[i], 'volume_confirmation'] = True
                    signals.loc[data.index[i], 'confidence'] += 10.0
                    
            elif signals.loc[data.index[i], 'sell_signal']:
                signals.loc[data.index[i], 'position_size'] = min(0.3, (50 - signals.loc[data.index[i], 'score']) / 100)
                # 止损设置为当前价格的2.5个ATR
                signals.loc[data.index[i], 'stop_loss'] = data['close'].iloc[i] + 2.5 * atr_values.iloc[i]
                # 判断成交量确认
                if i > 0 and data['volume'].iloc[i] > data['volume'].iloc[i-1] * 1.2:
                    signals.loc[data.index[i], 'volume_confirmation'] = True
                    signals.loc[data.index[i], 'confidence'] += 10.0
                    
            # 根据趋势设置市场环境
            if signals.loc[data.index[i], 'score'] >= 70:
                signals.loc[data.index[i], 'market_env'] = '强势'
            elif signals.loc[data.index[i], 'score'] <= 30:
                signals.loc[data.index[i], 'market_env'] = '弱势'
            else:
                signals.loc[data.index[i], 'market_env'] = '中性'
                
            # 限制评分范围
            signals.loc[data.index[i], 'score'] = np.clip(signals.loc[data.index[i], 'score'], 0, 100)
            
            # 限制置信度范围
            signals.loc[data.index[i], 'confidence'] = np.clip(signals.loc[data.index[i], 'confidence'], 0, 100)
        
        return signals

    def _register_mfi_patterns(self):
        """
        注册MFI指标相关形态
        """
        from indicators.pattern_registry import PatternRegistry, PatternType, PatternStrength
        
        # 获取PatternRegistry实例
        registry = PatternRegistry()
        
        # 注册MFI超买超卖形态
        registry.register(
            pattern_id="MFI_OVERBOUGHT",
            display_name="MFI超买",
            description="MFI值高于80，表明市场可能超买，存在回调风险",
            indicator_id="MFI",
            pattern_type=PatternType.BEARISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=-15.0
        )
        
        registry.register(
            pattern_id="MFI_OVERSOLD",
            display_name="MFI超卖",
            description="MFI值低于20，表明市场可能超卖，存在反弹机会",
            indicator_id="MFI",
            pattern_type=PatternType.BULLISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=15.0
        )
        
        # 注册MFI零轴穿越形态
        registry.register(
            pattern_id="MFI_CROSS_ABOVE_50",
            display_name="MFI上穿50",
            description="MFI从下方穿越50，表明资金流入增加",
            indicator_id="MFI",
            pattern_type=PatternType.BULLISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=12.0
        )
        
        registry.register(
            pattern_id="MFI_CROSS_BELOW_50",
            display_name="MFI下穿50",
            description="MFI从上方穿越50，表明资金流出增加",
            indicator_id="MFI",
            pattern_type=PatternType.BEARISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=-12.0
        )
        
        # 注册MFI背离形态
        registry.register(
            pattern_id="MFI_BULLISH_DIVERGENCE",
            display_name="MFI底背离",
            description="价格创新低，但MFI未创新低，表明下跌动能减弱",
            indicator_id="MFI",
            pattern_type=PatternType.BULLISH,
            default_strength=PatternStrength.STRONG,
            score_impact=20.0
        )
        
        registry.register(
            pattern_id="MFI_BEARISH_DIVERGENCE",
            display_name="MFI顶背离",
            description="价格创新高，但MFI未创新高，表明上涨动能减弱",
            indicator_id="MFI",
            pattern_type=PatternType.BEARISH,
            default_strength=PatternStrength.STRONG,
            score_impact=-20.0
        )
        
        # 注册MFI失败摆动形态
        registry.register(
            pattern_id="MFI_FAILURE_SWING_BULLISH",
            display_name="MFI看涨失败摆动",
            description="MFI在超卖区形成双底但未创新低，看涨信号",
            indicator_id="MFI",
            pattern_type=PatternType.BULLISH,
            default_strength=PatternStrength.STRONG,
            score_impact=18.0
        )
        
        registry.register(
            pattern_id="MFI_FAILURE_SWING_BEARISH",
            display_name="MFI看跌失败摆动",
            description="MFI在超买区形成双顶但未创新高，看跌信号",
            indicator_id="MFI",
            pattern_type=PatternType.BEARISH,
            default_strength=PatternStrength.STRONG,
            score_impact=-18.0
        )
        
        # 注册MFI趋势形态
        registry.register(
            pattern_id="MFI_UPTREND",
            display_name="MFI上升趋势",
            description="MFI持续上升，表明资金流入增强",
            indicator_id="MFI",
            pattern_type=PatternType.BULLISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=10.0
        )
        
        registry.register(
            pattern_id="MFI_DOWNTREND",
            display_name="MFI下降趋势",
            description="MFI持续下降，表明资金流出增强",
            indicator_id="MFI",
            pattern_type=PatternType.BEARISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=-10.0
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
