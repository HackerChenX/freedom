#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
成交量指标(VR)模块

实现成交量指标计算功能，用于判断多空力量对比
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any

from indicators.base_indicator import BaseIndicator
from indicators.common import crossover, crossunder
from utils.logger import get_logger
from indicators.pattern_registry import PatternRegistry, PatternType, PatternStrength

logger = get_logger(__name__)


class VR(BaseIndicator):
    """
    成交量指标(Volume Ratio)
    
    计算上涨成交量与下跌成交量的比值，判断多空力量对比
    """
    
    def __init__(self, period: int = 26, ma_period: int = 6):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """
        初始化VR指标
        
        Args:
            period: VR计算周期，默认为26日
            ma_period: VR均线周期，默认为6日
        """
        super().__init__(name="VR", description="成交量指标，计算上涨成交量与下跌成交量的比值")
        self.period = period
        self.ma_period = ma_period
    
    def set_parameters(self, period: int = None, ma_period: int = None):
        """
        设置指标参数
        """
        if period is not None:
            self.period = period
        if ma_period is not None:
            self.ma_period = ma_period

    def calculate_confidence(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """
        计算VR指标的置信度。
        """
        return 0.5

    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取VR指标的技术形态
        """
        return pd.DataFrame(index=data.index)

    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算VR指标
        
        Args:
            data: 输入数据，包含OHLCV数据
            
        Returns:
            pd.DataFrame: 计算结果，包含VR指标值
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, self.REQUIRED_COLUMNS)
        
        # 初始化结果数据框
        result = data.copy()
        
        # 判断价格变动方向
        price_direction = np.zeros(len(data))
        price_direction[1:] = np.sign(data["close"].values[1:] - data["close"].values[:-1])
        
        # 初始化上涨、下跌和平盘成交量
        up_volume = np.zeros(len(data))
        down_volume = np.zeros(len(data))
        flat_volume = np.zeros(len(data))
        
        # 分类成交量
        for i in range(1, len(data)):
            if price_direction[i] > 0:  # 价格上涨
                up_volume[i] = data["volume"].iloc[i]
            elif price_direction[i] < 0:  # 价格下跌
                down_volume[i] = data["volume"].iloc[i]
            else:  # 价格不变
                flat_volume[i] = data["volume"].iloc[i]
        
        # 计算N日上涨、下跌和平盘成交量之和
        up_volume_sum = pd.Series(up_volume).rolling(window=self.period).sum()
        down_volume_sum = pd.Series(down_volume).rolling(window=self.period).sum()
        flat_volume_sum = pd.Series(flat_volume).rolling(window=self.period).sum()
        
        # 计算VR: (AVS+1/2SVS)/(BVS+1/2SVS)×100
        # 其中AVS为上涨成交量，BVS为下跌成交量，SVS为平盘成交量
        denominator = down_volume_sum + 0.5 * flat_volume_sum
        denominator[denominator == 0] = 0.000001 # 避免除以0
        vr = ((up_volume_sum + 0.5 * flat_volume_sum) / denominator) * 100
        
        # 添加到结果
        result["vr"] = vr
        
        # 计算VR均线
        result["vr_ma"] = result["vr"].rolling(window=self.ma_period).mean()
        
        # 存储结果
        self._result = result
        
        return result
    
    def get_signals(self, data: pd.DataFrame, overbought: float = 160, oversold: float = 70) -> pd.DataFrame:
        """
        生成VR信号
        
        Args:
            data: 输入数据，包含VR指标
            overbought: 超买阈值，默认为160
            oversold: 超卖阈值，默认为70
            
        Returns:
            pd.DataFrame: 包含VR信号的数据框
        """
        if "vr" not in data.columns:
            data = self.calculate(data)
        
        # 初始化信号列
        data["vr_signal"] = np.nan
        
        # 生成信号
        for i in range(1, len(data)):
            if pd.notna(data["vr"].iloc[i]) and pd.notna(data["vr"].iloc[i-1]):
                # VR下穿超买线：卖出信号
                if data["vr"].iloc[i] < overbought and data["vr"].iloc[i-1] >= overbought:
                    data.iloc[i, data.columns.get_loc("vr_signal")] = -1
                
                # VR上穿超卖线：买入信号
                elif data["vr"].iloc[i] > oversold and data["vr"].iloc[i-1] <= oversold:
                    data.iloc[i, data.columns.get_loc("vr_signal")] = 1
                
                # 无信号
                else:
                    data.iloc[i, data.columns.get_loc("vr_signal")] = 0
        
        # 检测VR与VR均线的交叉
        data["vr_ma_cross"] = np.nan
        
        for i in range(1, len(data)):
            if (pd.notna(data["vr"].iloc[i]) and pd.notna(data["vr_ma"].iloc[i]) and 
                pd.notna(data["vr"].iloc[i-1]) and pd.notna(data["vr_ma"].iloc[i-1])):
                
                # VR上穿其均线：买入信号
                if data["vr"].iloc[i] > data["vr_ma"].iloc[i] and data["vr"].iloc[i-1] <= data["vr_ma"].iloc[i-1]:
                    data.iloc[i, data.columns.get_loc("vr_ma_cross")] = 1
                
                # VR下穿其均线：卖出信号
                elif data["vr"].iloc[i] < data["vr_ma"].iloc[i] and data["vr"].iloc[i-1] >= data["vr_ma"].iloc[i-1]:
                    data.iloc[i, data.columns.get_loc("vr_ma_cross")] = -1
                
                # 无交叉
                else:
                    data.iloc[i, data.columns.get_loc("vr_ma_cross")] = 0
        
        return data
    
    def get_market_sentiment(self, data: pd.DataFrame, overbought: float = 160, 
                           oversold: float = 70, neutral_upper: float = 120, 
                           neutral_lower: float = 90) -> pd.DataFrame:
        """
        获取市场情绪
        
        Args:
            data: 输入数据，包含VR指标
            overbought: 超买阈值，默认为160
            oversold: 超卖阈值，默认为70
            neutral_upper: 中性区间上限，默认为120
            neutral_lower: 中性区间下限，默认为90
            
        Returns:
            pd.DataFrame: 包含市场情绪的数据框
        """
        if "vr" not in data.columns:
            data = self.calculate(data)
        
        # 初始化情绪列
        data["market_sentiment"] = "中性"
        
        # 判断市场情绪
        data.loc[data["vr"] > overbought, "market_sentiment"] = "极度多头"
        data.loc[data["vr"] > neutral_upper, "market_sentiment"] = "多头"
        data.loc[data["vr"] < oversold, "market_sentiment"] = "空头"
        data.loc[data["vr"] < neutral_lower, "market_sentiment"] = "中性偏空"

        return data
    
    def get_vr_change_rate(self, data: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """
        计算VR变化率
        
        Args:
            data: 输入数据，包含VR指标
            window: 计算窗口期，默认为5日
            
        Returns:
            pd.DataFrame: 包含VR变化率的数据框
        """
        if "vr" not in data.columns:
            data = self.calculate(data)
        
        # 计算VR变化率
        data["vr_change_rate"] = data["vr"].pct_change(periods=window) * 100
        
        return data

    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算VR原始评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 原始评分序列（0-100分）
        """
        # 确保已计算VR
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)
        
        score = pd.Series(50.0, index=data.index)  # 基础分50分
        
        # 1. VR超买超卖评分
        overbought_oversold_score = self._calculate_vr_overbought_oversold_score()
        score += overbought_oversold_score
        
        # 2. VR与均线关系评分
        ma_relation_score = self._calculate_vr_ma_relation_score()
        score += ma_relation_score
        
        # 3. VR趋势评分
        trend_score = self._calculate_vr_trend_score()
        score += trend_score
        
        # 4. VR背离评分
        divergence_score = self._calculate_vr_divergence_score(data)
        score += divergence_score
        
        # 5. VR强度评分
        strength_score = self._calculate_vr_strength_score()
        score += strength_score
        
        return np.clip(score, 0, 100)
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别VR技术形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别出的形态列表
        """
        patterns = []
        
        # 确保已计算VR
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return patterns
        
        # 1. 检测VR超买超卖形态
        overbought_oversold_patterns = self._detect_vr_overbought_oversold_patterns()
        patterns.extend(overbought_oversold_patterns)
        
        # 2. 检测VR与均线关系形态
        ma_relation_patterns = self._detect_vr_ma_relation_patterns()
        patterns.extend(ma_relation_patterns)
        
        # 3. 检测VR趋势形态
        trend_patterns = self._detect_vr_trend_patterns()
        patterns.extend(trend_patterns)
        
        # 4. 检测VR背离形态
        divergence_patterns = self._detect_vr_divergence_patterns(data)
        patterns.extend(divergence_patterns)
        
        # 5. 检测VR强度形态
        strength_patterns = self._detect_vr_strength_patterns()
        patterns.extend(strength_patterns)
        
        return patterns
    
    def _calculate_vr_overbought_oversold_score(self) -> pd.Series:
        """
        计算VR超买超卖评分
        
        Returns:
            pd.Series: 超买超卖评分
        """
        overbought_oversold_score = pd.Series(0.0, index=self._result.index)
        
        vr_values = self._result['vr']
        
        # VR超卖区域（VR < 70）+20分
        oversold_condition = vr_values < 70
        overbought_oversold_score += oversold_condition * 20
        
        # VR超买区域（VR > 160）-20分
        overbought_condition = vr_values > 160
        overbought_oversold_score -= overbought_condition * 20
        
        # VR上穿70+15分
        vr_cross_up_70 = crossover(vr_values, 70)
        overbought_oversold_score += vr_cross_up_70 * 15
        
        # VR下穿160-15分
        vr_cross_down_160 = crossunder(vr_values, 160)
        overbought_oversold_score -= vr_cross_down_160 * 15
        
        # VR极度超卖（VR < 50）额外+15分
        extreme_oversold = vr_values < 50
        overbought_oversold_score += extreme_oversold * 15
        
        # VR极度超买（VR > 200）额外-15分
        extreme_overbought = vr_values > 200
        overbought_oversold_score -= extreme_overbought * 15
        
        return overbought_oversold_score
    
    def _calculate_vr_ma_relation_score(self) -> pd.Series:
        """
        计算VR与均线关系评分
        
        Returns:
            pd.Series: 均线关系评分
        """
        ma_relation_score = pd.Series(0.0, index=self._result.index)
        
        vr_values = self._result['vr']
        vr_ma_values = self._result['vr_ma']
        
        # VR在均线上方+8分
        vr_above_ma = vr_values > vr_ma_values
        ma_relation_score += vr_above_ma * 8
        
        # VR在均线下方-8分
        vr_below_ma = vr_values < vr_ma_values
        ma_relation_score -= vr_below_ma * 8
        
        # VR上穿均线+20分
        vr_cross_up_ma = crossover(vr_values, vr_ma_values)
        ma_relation_score += vr_cross_up_ma * 20
        
        # VR下穿均线-20分
        vr_cross_down_ma = crossunder(vr_values, vr_ma_values)
        ma_relation_score -= vr_cross_down_ma * 20
        
        return ma_relation_score
    
    def _calculate_vr_trend_score(self) -> pd.Series:
        """
        计算VR趋势评分
        
        Returns:
            pd.Series: 趋势评分
        """
        trend_score = pd.Series(0.0, index=self._result.index)
        
        vr_values = self._result['vr']
        
        # VR上升趋势+10分
        vr_rising = vr_values > vr_values.shift(1)
        trend_score += vr_rising * 10
        
        # VR下降趋势-10分
        vr_falling = vr_values < vr_values.shift(1)
        trend_score -= vr_falling * 10
        
        # VR连续上升（3个周期）+15分
        if len(vr_values) >= 4:
            consecutive_rising = (
                (vr_values > vr_values.shift(1)) &
                (vr_values.shift(1) > vr_values.shift(2)) &
                (vr_values.shift(2) > vr_values.shift(3))
            )
            trend_score += consecutive_rising.fillna(False) * 15
        
        # VR连续下降（3个周期）-15分
        if len(vr_values) >= 4:
            consecutive_falling = (
                (vr_values < vr_values.shift(1)) &
                (vr_values.shift(1) < vr_values.shift(2)) &
                (vr_values.shift(2) < vr_values.shift(3))
            )
            trend_score -= consecutive_falling.fillna(False) * 15
        
        return trend_score
    
    def _calculate_vr_divergence_score(self, data: pd.DataFrame) -> pd.Series:
        """
        计算VR背离评分
        
        Args:
            data: 价格数据
            
        Returns:
            pd.Series: 背离评分
        """
        divergence_score = pd.Series(0.0, index=self._result.index)
        
        if 'close' not in data.columns:
            return divergence_score
        
        close_price = data['close']
        vr_values = self._result['vr']
        
        # 简化的背离检测
        if len(close_price) >= 20:
            # 检查最近20个周期的价格和VR趋势
            recent_periods = 20
            
            for i in range(recent_periods, len(close_price)):
                # 寻找最近的价格和VR峰值/谷值
                price_window = close_price.iloc[i-recent_periods:i+1]
                vr_window = vr_values.iloc[i-recent_periods:i+1]
                
                # 检查是否为价格新高/新低
                current_price = close_price.iloc[i]
                current_vr = vr_values.iloc[i]
                
                price_is_high = current_price >= price_window.max()
                price_is_low = current_price <= price_window.min()
                vr_is_high = current_vr >= vr_window.max()
                vr_is_low = current_vr <= vr_window.min()
                
                # 正背离：价格创新低但VR未创新低
                if price_is_low and not vr_is_low:
                    divergence_score.iloc[i] += 25
                
                # 负背离：价格创新高但VR未创新高
                elif price_is_high and not vr_is_high:
                    divergence_score.iloc[i] -= 25
        
        return divergence_score
    
    def _calculate_vr_strength_score(self) -> pd.Series:
        """
        计算VR强度评分
        
        Returns:
            pd.Series: 强度评分
        """
        strength_score = pd.Series(0.0, index=self._result.index)
        
        vr_values = self._result['vr']
        
        # 计算VR变化幅度
        vr_change = vr_values.diff()
        
        # VR大幅上升（变化>20）+12分
        large_rise = vr_change > 20
        strength_score += large_rise.fillna(False) * 12
        
        # VR大幅下降（变化<-20）-12分
        large_fall = vr_change < -20
        strength_score -= large_fall.fillna(False) * 12
        
        # VR快速变化（绝对值>30）额外±8分
        rapid_change = np.abs(vr_change) > 30
        rapid_change_direction = np.sign(vr_change)
        strength_score += (rapid_change * rapid_change_direction).fillna(0) * 8
        
        return strength_score
    
    def _detect_vr_overbought_oversold_patterns(self) -> List[str]:
        """
        检测VR超买超卖形态
        
        Returns:
            List[str]: 超买超卖形态列表
        """
        patterns = []
        
        vr_values = self._result['vr']
        
        if len(vr_values) > 0:
            current_vr = vr_values.iloc[-1]
            
            if pd.isna(current_vr):
                return patterns
            
            if current_vr < 50:
                patterns.append("VR极度超卖")
            elif current_vr < 70:
                patterns.append("VR超卖")
            elif current_vr > 200:
                patterns.append("VR极度超买")
            elif current_vr > 160:
                patterns.append("VR超买")
            elif 90 <= current_vr <= 120:
                patterns.append("VR中性区域")
        
        # 检查最近的阈值穿越
        recent_periods = min(5, len(vr_values))
        recent_vr = vr_values.tail(recent_periods)
        
        if crossover(recent_vr, 70).any():
            patterns.append("VR上穿超卖线")
        
        if crossunder(recent_vr, 160).any():
            patterns.append("VR下穿超买线")
        
        return patterns
    
    def _detect_vr_ma_relation_patterns(self) -> List[str]:
        """
        检测VR与均线关系形态
        
        Returns:
            List[str]: 均线关系形态列表
        """
        patterns = []
        
        vr_values = self._result['vr']
        vr_ma_values = self._result['vr_ma']
        
        # 检查最近的均线穿越
        recent_periods = min(5, len(vr_values))
        recent_vr = vr_values.tail(recent_periods)
        recent_vr_ma = vr_ma_values.tail(recent_periods)
        
        if crossover(recent_vr, recent_vr_ma).any():
            patterns.append("VR上穿均线")
        
        if crossunder(recent_vr, recent_vr_ma).any():
            patterns.append("VR下穿均线")
        
        # 检查当前位置
        if len(vr_values) > 0 and len(vr_ma_values) > 0:
            current_vr = vr_values.iloc[-1]
            current_vr_ma = vr_ma_values.iloc[-1]
            
            if not pd.isna(current_vr) and not pd.isna(current_vr_ma):
                if current_vr > current_vr_ma:
                    patterns.append("VR均线上方")
                elif current_vr < current_vr_ma:
                    patterns.append("VR均线下方")
                else:
                    patterns.append("VR均线位置")
        
        return patterns
    
    def _detect_vr_trend_patterns(self) -> List[str]:
        """
        检测VR趋势形态
        
        Returns:
            List[str]: 趋势形态列表
        """
        patterns = []
        
        vr_values = self._result['vr']
        
        # 检查VR趋势
        if len(vr_values) >= 3:
            recent_3 = vr_values.tail(3)
            if len(recent_3) == 3 and not recent_3.isna().any():
                if (recent_3.iloc[2] > recent_3.iloc[1] > recent_3.iloc[0]):
                    patterns.append("VR连续上升")
                elif (recent_3.iloc[2] < recent_3.iloc[1] < recent_3.iloc[0]):
                    patterns.append("VR连续下降")
        
        # 检查当前趋势
        if len(vr_values) >= 2:
            current_vr = vr_values.iloc[-1]
            prev_vr = vr_values.iloc[-2]
            
            if not pd.isna(current_vr) and not pd.isna(prev_vr):
                if current_vr > prev_vr:
                    patterns.append("VR上升")
                elif current_vr < prev_vr:
                    patterns.append("VR下降")
                else:
                    patterns.append("VR平稳")
        
        return patterns
    
    def _detect_vr_divergence_patterns(self, data: pd.DataFrame) -> List[str]:
        """
        检测VR背离形态
        
        Args:
            data: 价格数据
            
        Returns:
            List[str]: 背离形态列表
        """
        patterns = []
        
        if 'close' not in data.columns:
            return patterns
        
        close_price = data['close']
        vr_values = self._result['vr']
        
        if len(close_price) >= 20:
            # 检查最近20个周期的趋势
            recent_price = close_price.tail(20)
            recent_vr = vr_values.tail(20)
            
            # 简化的背离检测
            price_trend = recent_price.iloc[-1] - recent_price.iloc[0]
            vr_trend = recent_vr.iloc[-1] - recent_vr.iloc[0]
            
            # 背离检测
            if price_trend < -0.02 * recent_price.iloc[0] and vr_trend > 5:  # 价格下跌但VR上升
                patterns.append("VR正背离")
            elif price_trend > 0.02 * recent_price.iloc[0] and vr_trend < -5:  # 价格上涨但VR下降
                patterns.append("VR负背离")
            elif abs(price_trend) < 0.01 * recent_price.iloc[0] and abs(vr_trend) < 2:
                patterns.append("VR价格同步")
        
        return patterns
    
    def _detect_vr_strength_patterns(self) -> List[str]:
        """
        检测VR强度形态
        
        Returns:
            List[str]: 强度形态列表
        """
        patterns = []
        
        vr_values = self._result['vr']
        
        if len(vr_values) >= 2:
            current_vr = vr_values.iloc[-1]
            prev_vr = vr_values.iloc[-2]
            
            if not pd.isna(current_vr) and not pd.isna(prev_vr):
                vr_change = current_vr - prev_vr
                
                if vr_change > 30:
                    patterns.append("VR急速上升")
                elif vr_change > 20:
                    patterns.append("VR大幅上升")
                elif vr_change < -30:
                    patterns.append("VR急速下降")
                elif vr_change < -20:
                    patterns.append("VR大幅下降")
                elif abs(vr_change) <= 5:
                    patterns.append("VR变化平缓")
        
        return patterns

    def _register_vr_patterns(self):
        """
        注册VR指标相关形态
        """
        # 获取PatternRegistry实例
        registry = PatternRegistry()
        
        # 注册VR超买超卖形态
        registry.register(
            pattern_id="VR_OVERBOUGHT",
            display_name="VR超买",
            description="VR值高于超买阈值（通常为160-200），表明市场可能超买",
            indicator_id="VR",
            pattern_type=PatternType.BEARISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=-15.0
        )
        
        registry.register(
            pattern_id="VR_OVERSOLD",
            display_name="VR超卖",
            description="VR值低于超卖阈值（通常为40-70），表明市场可能超卖",
            indicator_id="VR",
            pattern_type=PatternType.BULLISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=15.0
        )
        
        # 注册VR趋势形态
        registry.register(
            pattern_id="VR_UPTREND",
            display_name="VR上升趋势",
            description="VR值连续上升，表明市场活跃度和买盘力量增强",
            indicator_id="VR",
            pattern_type=PatternType.BULLISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=12.0
        )
        
        registry.register(
            pattern_id="VR_DOWNTREND",
            display_name="VR下降趋势",
            description="VR值连续下降，表明市场活跃度和买盘力量减弱",
            indicator_id="VR",
            pattern_type=PatternType.BEARISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=-12.0
        )
        
        # 注册VR与均线交叉形态
        registry.register(
            pattern_id="VR_GOLDEN_CROSS",
            display_name="VR金叉",
            description="VR上穿其均线，表明买盘力量增强",
            indicator_id="VR",
            pattern_type=PatternType.BULLISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=10.0
        )
        
        registry.register(
            pattern_id="VR_DEATH_CROSS",
            display_name="VR死叉",
            description="VR下穿其均线，表明买盘力量减弱",
            indicator_id="VR",
            pattern_type=PatternType.BEARISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=-10.0
        )
        
        # 注册VR背离形态
        registry.register(
            pattern_id="VR_BULLISH_DIVERGENCE",
            display_name="VR底背离",
            description="价格创新低但VR未创新低，可能预示反弹",
            indicator_id="VR",
            pattern_type=PatternType.BULLISH,
            default_strength=PatternStrength.STRONG,
            score_impact=20.0
        )
        
        registry.register(
            pattern_id="VR_BEARISH_DIVERGENCE",
            display_name="VR顶背离",
            description="价格创新高但VR未创新高，可能预示回调",
            indicator_id="VR",
            pattern_type=PatternType.BEARISH,
            default_strength=PatternStrength.STRONG,
            score_impact=-20.0
        )

    def generate_trading_signals(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """
        生成VR指标的交易信号

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            Dict[str, pd.Series]: 包含交易信号的字典
        """
        # 计算VR指标
        vr_data = self.calculate(data, **kwargs)
        
        # 获取信号
        signals_df = self.get_signals(vr_data, **kwargs)
        
        # 提取买卖信号
        buy_signal = signals_df['vr_ma_cross'] == 1
        sell_signal = signals_df['vr_ma_cross'] == -1
        
        return {
            "buy": buy_signal,
            "sell": sell_signal
        }
