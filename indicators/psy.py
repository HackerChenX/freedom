#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
心理线指标(PSY)模块

实现心理线指标计算功能，用于判断市场情绪和超买超卖状态。
包含标准功能和增强功能，可通过参数控制使用哪些功能。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple

from indicators.base_indicator import BaseIndicator, MarketEnvironment
from utils.logger import get_logger
from indicators.pattern_registry import PatternRegistry, PatternType, PatternStrength

logger = get_logger(__name__)


class PSY(BaseIndicator):
    """
    心理线指标(Psychological Line)
    
    计算一段时间内上涨日所占百分比，反映市场人气强弱和超买超卖状态
    
    增强特性 (通过参数启用):
    1. 自适应参数设计：根据市场波动率动态调整PSY的计算周期
    2. 多周期PSY协同分析：结合不同周期的PSY指标提高信号可靠性
    3. 市场氛围评估增强：更精确地评估市场过度乐观/悲观情绪
    4. 形态识别系统：识别PSY极值反转、区间突破和均值回归等形态
    """
    
    def __init__(self, period: int = 12, 
                 secondary_period: int = 24,
                 multi_periods: List[int] = None,
                 adaptive_period: bool = False,
                 volatility_lookback: int = 20,
                 enhanced: bool = False):
        """
        初始化PSY指标
        
        Args:
            period: 计算周期，默认为12日
            secondary_period: 次要周期，默认为24日 (仅在enhanced=True时使用)
            multi_periods: 多周期分析参数，默认为[6, 12, 24, 48] (仅在enhanced=True时使用)
            adaptive_period: 是否启用自适应周期，默认为False (仅在enhanced=True时使用)
            volatility_lookback: 波动率计算回溯期，默认为20 (仅在enhanced=True时使用)
            enhanced: 是否启用增强功能，默认为False
        """
        super().__init__(name="PSY", description="心理线指标，计算一段时间内上涨日所占百分比，判断市场情绪")
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        self.period = period
        self.secondary_period = secondary_period
        self.multi_periods = multi_periods or [6, 12, 24, 48]
        self.adaptive_period = adaptive_period
        self.volatility_lookback = volatility_lookback
        self.enhanced = enhanced
        self.market_environment = "normal"
        
        # 增强版内部变量
        if enhanced:
            self.name = "EnhancedPSY"
            self.description = "增强型心理线指标，优化参数自适应性，增加多周期协同分析和市场氛围评估"
            self._secondary_psy = None
            self._multi_period_psy = {}
            self._adaptive_period = period  # 自适应后的周期
    
    def set_market_environment(self, environment: str) -> None:
        """
        设置市场环境
        
        Args:
            environment (str): 市场环境类型 ('bull_market', 'bear_market', 'sideways_market', 'volatile_market', 'normal')
        """
        valid_environments = ['bull_market', 'bear_market', 'sideways_market', 'volatile_market', 'normal']
        if environment not in valid_environments:
            raise ValueError(f"无效的市场环境类型: {environment}。有效类型: {valid_environments}")
        
        self.market_environment = environment
    
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
        result = data.copy()
        
        # 如果启用增强模式且启用自适应周期，则调整参数
        if self.enhanced and self.adaptive_period:
            self._adjust_parameters_by_volatility(data)
            current_period = self._adaptive_period
        else:
            current_period = self.period
        
        # 计算价格变化
        price_change = data["close"].diff()
        
        # 统计上涨日数
        up_days = (price_change > 0).astype(int)
        
        # 计算PSY：N日内上涨天数 / N * 100
        result["psy"] = up_days.rolling(window=current_period).sum() / current_period * 100
        
        # 计算PSY的移动平均线（作为信号线）
        result["psyma"] = result["psy"].rolling(window=int(current_period/2)).mean()
        
        # 额外计算：PSY变化率
        result["psy_change"] = result["psy"].diff()
        
        # 增强版功能
        if self.enhanced:
            # 计算次要周期PSY
            secondary_up_days = up_days.rolling(window=self.secondary_period).sum()
            result["psy_secondary"] = secondary_up_days / self.secondary_period * 100
            result["psyma_secondary"] = result["psy_secondary"].rolling(window=int(self.secondary_period/2)).mean()
            self._secondary_psy = result["psy_secondary"]
            
            # 计算多周期PSY
            self._multi_period_psy = {}
            for period in self.multi_periods:
                if period != current_period and period != self.secondary_period:
                    multi_up_days = up_days.rolling(window=period).sum()
                    result[f"psy_{period}"] = multi_up_days / period * 100
                    result[f"psyma_{period}"] = result[f"psy_{period}"].rolling(window=int(period/2)).mean()
                    self._multi_period_psy[period] = result[f"psy_{period}"]
            
            # 计算PSY动态特性
            result["psy_momentum"] = result["psy"] - result["psy"].shift(3)
            result["psy_slope"] = self._calculate_slope(result["psy"], 5)
            result["psy_accel"] = result["psy_slope"] - result["psy_slope"].shift(1)
            
            # 计算市场氛围指标
            result["market_sentiment"] = self._calculate_market_sentiment(result["psy"])
            
            # 计算均值回归特性
            result["mean_reversion"] = self._calculate_mean_reversion(result["psy"])
        
        # 存储结果
        self._result = result
        
        return result
    
    def _adjust_parameters_by_volatility(self, data: pd.DataFrame) -> None:
        """
        根据市场波动率动态调整PSY周期参数
        
        Args:
            data: 包含价格数据的DataFrame
        """
        # 计算价格波动率
        close = data['close']
        
        # 计算价格变化率
        returns = close.pct_change()
        
        # 计算波动率（标准差）
        volatility = returns.rolling(window=self.volatility_lookback).std().iloc[-1]
        
        # 如果波动率数据不足，则使用默认周期
        if pd.isna(volatility):
            self._adaptive_period = self.period
            return
        
        # 计算历史波动率
        historical_volatility = returns.rolling(window=self.volatility_lookback*5).std().iloc[-1]
        
        # 如果历史波动率数据不足，则使用默认周期
        if pd.isna(historical_volatility) or historical_volatility == 0:
            self._adaptive_period = self.period
            return
        
        # 计算相对波动率
        relative_volatility = volatility / historical_volatility if historical_volatility > 0 else 1.0
        
        # 根据相对波动率调整周期
        if relative_volatility > 1.5:  # 高波动市场
            # 增加周期以过滤噪声
            self._adaptive_period = int(self.period * 1.5)
        elif relative_volatility < 0.7:  # 低波动市场
            # 减少周期以提高敏感度
            self._adaptive_period = max(int(self.period * 0.7), 6)  # 确保最小周期为6
        else:  # 正常波动市场
            # 使用默认周期
            self._adaptive_period = self.period
        
        # 根据市场环境进一步调整
        if self.market_environment == 'bull_market':
            # 牛市中略微减少周期，更敏感地捕捉上涨趋势
            self._adaptive_period = max(int(self._adaptive_period * 0.9), 6)
        elif self.market_environment == 'bear_market':
            # 熊市中略微增加周期，过滤更多噪声
            self._adaptive_period = int(self._adaptive_period * 1.1)
        elif self.market_environment == 'volatile_market':
            # 高波动市场中增加周期，过滤更多噪声
            self._adaptive_period = int(self._adaptive_period * 1.2)
        
        logger.debug(f"调整PSY周期: 原始={self.period}, 调整后={self._adaptive_period}, "
                    f"相对波动率={relative_volatility:.2f}, 市场环境={self.market_environment}")
    
    def _calculate_slope(self, series: pd.Series, period: int = 5) -> pd.Series:
        """
        计算序列的斜率
        
        Args:
            series: 输入序列
            period: 计算周期，默认为5
            
        Returns:
            pd.Series: 斜率序列
        """
        return (series - series.shift(period)) / period
    
    def _calculate_market_sentiment(self, psy: pd.Series) -> pd.Series:
        """
        计算市场氛围指标
        
        Args:
            psy: PSY序列
            
        Returns:
            pd.Series: 市场氛围指标序列
        """
        # 将PSY从0-100的范围映射到-100至100的范围，以便于判断市场情绪
        sentiment = (psy - 50) * 2
        
        # 计算市场情绪的移动平均，以减少噪声
        sentiment_ma = sentiment.rolling(window=10).mean()
        
        # 计算情绪变化速率
        sentiment_change = sentiment - sentiment.shift(5)
        
        # 综合情绪水平和变化速率
        combined_sentiment = sentiment_ma + sentiment_change * 0.5
        
        return combined_sentiment
    
    def _calculate_mean_reversion(self, psy: pd.Series) -> pd.Series:
        """
        计算PSY均值回归特性
        
        Args:
            psy: PSY序列
            
        Returns:
            pd.Series: 均值回归特性序列
        """
        # 计算PSY与中性值(50)的距离
        distance_from_mean = psy - 50
        
        # 计算距离的变化率（向均值回归为负，远离均值为正）
        distance_change = abs(distance_from_mean) - abs(distance_from_mean.shift(1))
        
        # 向均值回归的强度（负值表示向均值回归，正值表示远离均值）
        mean_reversion = -distance_change * np.sign(distance_from_mean)
        
        return mean_reversion
    
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
        
        # 增强功能评分
        if self.enhanced:
            # 5. 市场氛围评分
            sentiment_score = self._result.get('market_sentiment', pd.Series(0.0, index=score.index)) * 0.15
            score += sentiment_score
            
            # 6. 均值回归评分
            mean_reversion_score = self._result.get('mean_reversion', pd.Series(0.0, index=score.index)) * 5
            score += mean_reversion_score
            
            # 7. 多周期协同评分
            synergy = self.analyze_multi_period_synergy()
            if not synergy.empty and 'synergy_score' in synergy.columns:
                # 协同评分影响（最大±15分）
                synergy_effect = (synergy['synergy_score'] - 50) * 0.3
                score += synergy_effect
            
            # 8. 根据市场环境调整评分
            if self.market_environment == 'bull_market':
                # 牛市中增强多头信号
                score += (score - 50).clip(0, 50) * 0.2
            elif self.market_environment == 'bear_market':
                # 熊市中增强空头信号
                score -= (50 - score).clip(0, 50) * 0.2
            elif self.market_environment == 'volatile_market':
                # 高波动市场需要更强的信号才能确认
                score = 50 + (score - 50) * 1.2
        
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
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[Dict[str, Any]]:
        """
        识别PSY指标形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[Dict[str, Any]]: 识别出的形态列表，每个形态包含类型、强度和位置信息
        """
        # 如果启用了增强功能，使用增强版模式识别
        if self.enhanced:
            return self._identify_enhanced_patterns(data, **kwargs)
        
        # 否则使用基础版模式识别
        patterns = []
        
        # 确保已计算PSY
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None or len(self._result) < 5:
            return patterns
        
        psy = self._result['psy']
        last_psy = psy.iloc[-1]
        
        # 检查PSY超买超卖状态
        if last_psy > 80:
            patterns.append({
                'type': 'overbought',
                'strength': 'strong',
                'position': len(psy) - 1
            })
        elif last_psy > 70:
            patterns.append({
                'type': 'overbought',
                'strength': 'medium',
                'position': len(psy) - 1
            })
        elif last_psy < 20:
            patterns.append({
                'type': 'oversold',
                'strength': 'strong',
                'position': len(psy) - 1
            })
        elif last_psy < 30:
            patterns.append({
                'type': 'oversold',
                'strength': 'medium',
                'position': len(psy) - 1
            })
        
        # 检查PSY与信号线交叉
        if (psy.iloc[-2] < self._result['psyma'].iloc[-2] and 
            psy.iloc[-1] >= self._result['psyma'].iloc[-1]):
            patterns.append({
                'type': 'golden_cross',
                'strength': 'medium',
                'position': len(psy) - 1
            })
        elif (psy.iloc[-2] > self._result['psyma'].iloc[-2] and 
              psy.iloc[-1] <= self._result['psyma'].iloc[-1]):
            patterns.append({
                'type': 'death_cross',
                'strength': 'medium',
                'position': len(psy) - 1
            })
        
        # 检查PSY与零轴(50)交叉
        if psy.iloc[-2] < 50 and psy.iloc[-1] >= 50:
            patterns.append({
                'type': 'zero_line_cross_up',
                'strength': 'strong',
                'position': len(psy) - 1
            })
        elif psy.iloc[-2] > 50 and psy.iloc[-1] <= 50:
            patterns.append({
                'type': 'zero_line_cross_down',
                'strength': 'strong',
                'position': len(psy) - 1
            })
        
        return patterns
    
    def _identify_enhanced_patterns(self, data: pd.DataFrame, **kwargs) -> List[Dict[str, Any]]:
        """
        增强版PSY指标形态识别
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[Dict[str, Any]]: 识别出的形态列表，每个形态包含类型、强度和位置信息
        """
        patterns = []
        
        # 确保已计算PSY
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None or len(self._result) < 10:
            return patterns
        
        psy = self._result['psy']
        last_psy = psy.iloc[-1]
        
        # 计算多周期协同信息
        synergy = self.analyze_multi_period_synergy()
        
        # 1. 检查PSY超买超卖状态
        if last_psy > 85:
            patterns.append({
                'type': 'extreme_overbought',
                'strength': 'strong',
                'description': '极度超买，市场过热',
                'position': len(psy) - 1,
                'score_impact': -25.0
            })
        elif last_psy > 75:
            patterns.append({
                'type': 'overbought',
                'strength': 'medium',
                'description': '超买，可能面临回调',
                'position': len(psy) - 1,
                'score_impact': -15.0
            })
        elif last_psy < 15:
            patterns.append({
                'type': 'extreme_oversold',
                'strength': 'strong',
                'description': '极度超卖，市场过冷',
                'position': len(psy) - 1,
                'score_impact': 25.0
            })
        elif last_psy < 25:
            patterns.append({
                'type': 'oversold',
                'strength': 'medium',
                'description': '超卖，可能即将反弹',
                'position': len(psy) - 1,
                'score_impact': 15.0
            })
        
        # 2. 检查PSY与信号线交叉
        if (psy.iloc[-2] < self._result['psyma'].iloc[-2] and 
            psy.iloc[-1] >= self._result['psyma'].iloc[-1]):
            # 计算交叉角度，评估交叉强度
            cross_angle = psy.iloc[-1] - psy.iloc[-2]
            strength = 'medium'
            impact = 10.0
            if cross_angle > 5:
                strength = 'strong'
                impact = 15.0
            
            patterns.append({
                'type': 'golden_cross',
                'strength': strength,
                'description': 'PSY上穿信号线，买入信号',
                'angle': cross_angle,
                'position': len(psy) - 1,
                'score_impact': impact
            })
        elif (psy.iloc[-2] > self._result['psyma'].iloc[-2] and 
              psy.iloc[-1] <= self._result['psyma'].iloc[-1]):
            # 计算交叉角度，评估交叉强度
            cross_angle = psy.iloc[-2] - psy.iloc[-1]
            strength = 'medium'
            impact = -10.0
            if cross_angle > 5:
                strength = 'strong'
                impact = -15.0
            
            patterns.append({
                'type': 'death_cross',
                'strength': strength,
                'description': 'PSY下穿信号线，卖出信号',
                'angle': cross_angle,
                'position': len(psy) - 1,
                'score_impact': impact
            })
        
        # 3. 检查零轴(50)交叉
        if psy.iloc[-2] < 50 and psy.iloc[-1] >= 50:
            patterns.append({
                'type': 'zero_line_cross_up',
                'strength': 'strong',
                'description': 'PSY上穿50中性线，看涨信号增强',
                'position': len(psy) - 1,
                'score_impact': 20.0
            })
        elif psy.iloc[-2] > 50 and psy.iloc[-1] <= 50:
            patterns.append({
                'type': 'zero_line_cross_down',
                'strength': 'strong',
                'description': 'PSY下穿50中性线，看跌信号增强',
                'position': len(psy) - 1,
                'score_impact': -20.0
            })
        
        # 4. 检查PSY加速和减速
        if ('psy_accel' in self._result.columns and len(self._result) > 3):
            accel = self._result['psy_accel'].iloc[-1]
            if accel > 1.0 and psy.iloc[-1] > psy.iloc[-2]:
                patterns.append({
                    'type': 'bullish_acceleration',
                    'strength': 'medium',
                    'description': 'PSY上升加速，买入动能增强',
                    'position': len(psy) - 1,
                    'score_impact': 12.0
                })
            elif accel < -1.0 and psy.iloc[-1] < psy.iloc[-2]:
                patterns.append({
                    'type': 'bearish_acceleration',
                    'strength': 'medium',
                    'description': 'PSY下降加速，卖出动能增强',
                    'position': len(psy) - 1,
                    'score_impact': -12.0
                })
        
        # 5. 检查多周期协同信号
        if not synergy.empty and 'bullish_synergy' in synergy.columns:
            if synergy['bullish_synergy'].iloc[-1]:
                patterns.append({
                    'type': 'multi_period_bullish_synergy',
                    'strength': 'strong',
                    'description': '多周期PSY协同看涨，信号更可靠',
                    'position': len(psy) - 1,
                    'score_impact': 18.0
                })
            elif synergy['bearish_synergy'].iloc[-1]:
                patterns.append({
                    'type': 'multi_period_bearish_synergy',
                    'strength': 'strong',
                    'description': '多周期PSY协同看跌，信号更可靠',
                    'position': len(psy) - 1,
                    'score_impact': -18.0
                })
        
        # 6. 检查均值回归
        if 'mean_reversion' in self._result.columns:
            mean_rev = self._result['mean_reversion'].iloc[-1]
            if mean_rev > 5 and psy.iloc[-1] < 40:
                patterns.append({
                    'type': 'mean_reversion_bullish',
                    'strength': 'medium',
                    'description': 'PSY向均值回归（向上），潜在反弹',
                    'position': len(psy) - 1,
                    'score_impact': 10.0
                })
            elif mean_rev < -5 and psy.iloc[-1] > 60:
                patterns.append({
                    'type': 'mean_reversion_bearish',
                    'strength': 'medium',
                    'description': 'PSY向均值回归（向下），潜在回调',
                    'position': len(psy) - 1,
                    'score_impact': -10.0
                })
        
        # 7. 检测PSY钝化
        stagnant_psy = True
        for i in range(-5, 0):
            if abs(psy.iloc[i] - psy.iloc[i-1]) > 2:
                stagnant_psy = False
                break
        
        if stagnant_psy:
            patterns.append({
                'type': 'psy_stagnation',
                'strength': 'weak',
                'description': 'PSY钝化，市场缺乏明确方向',
                'position': len(psy) - 1,
                'score_impact': 0.0
            })
        
        return patterns

    def analyze_multi_period_synergy(self, threshold: float = 10.0) -> pd.DataFrame:
        """
        分析多周期PSY协同性
        
        Args:
            threshold: 判断一致性的阈值，默认为10.0
            
        Returns:
            pd.DataFrame: 多周期协同分析结果
        """
        if not self.enhanced or self._result is None:
            return pd.DataFrame()
        
        # 初始化结果DataFrame
        synergy = pd.DataFrame(index=self._result.index)
        
        # 获取所有周期的PSY值
        psy_columns = [col for col in self._result.columns if col.startswith('psy_') and col != 'psy_change' 
                      and col != 'psy_momentum' and col != 'psy_slope' and col != 'psy_accel']
        psy_columns = ['psy'] + psy_columns
        
        if len(psy_columns) < 2:
            return synergy
        
        # 计算各周期PSY的平均值和标准差
        synergy['psy_mean'] = self._result[psy_columns].mean(axis=1)
        synergy['psy_std'] = self._result[psy_columns].std(axis=1)
        
        # 计算各周期PSY的一致程度
        # 一致程度由标准差的倒数表示，标准差越小表示一致性越高
        synergy['psy_consistency'] = 1 / (synergy['psy_std'] + 0.1)  # 加0.1避免除零
        
        # 计算各周期PSY的方向一致性
        # 计算所有周期PSY的方向(1=上升，-1=下降，0=不变)
        directions = {}
        for col in psy_columns:
            directions[col] = np.sign(self._result[col] - self._result[col].shift(1))
        
        # 将方向合并到一个DataFrame
        directions_df = pd.DataFrame(directions, index=self._result.index)
        
        # 计算方向一致性（1=全部一致，0=完全不一致）
        synergy['direction_consistency'] = directions_df.abs().sum(axis=1) / len(psy_columns)
        
        # 计算多周期PSY的趋势
        # 1=全部上升，-1=全部下降，0=不一致
        synergy['multi_period_trend'] = np.where(
            directions_df.sum(axis=1) == len(psy_columns),  # 全部为正
            1,
            np.where(
                directions_df.sum(axis=1) == -len(psy_columns),  # 全部为负
                -1,
                0  # 不一致
            )
        )
        
        # 计算趋势强度
        # 基于所有周期PSY的平均动量
        momentum_columns = []
        for col in psy_columns:
            momentum_col = f"{col}_momentum"
            self._result[momentum_col] = self._result[col] - self._result[col].shift(3)
            momentum_columns.append(momentum_col)
        
        synergy['trend_strength'] = self._result[momentum_columns].mean(axis=1).abs()
        
        # 计算多周期PSY信号的协同指标
        # 计算多数周期PSY的位置(1=多数大于50，-1=多数小于50，0=平衡)
        above_50_count = (self._result[psy_columns] > 50).sum(axis=1)
        below_50_count = (self._result[psy_columns] < 50).sum(axis=1)
        
        synergy['position_majority'] = np.where(
            above_50_count > below_50_count,
            1,
            np.where(
                below_50_count > above_50_count,
                -1,
                0
            )
        )
        
        # 计算看涨/看跌的协同信号
        # 看涨协同：多数周期PSY>50且同步上升
        synergy['bullish_synergy'] = (
            (synergy['position_majority'] == 1) & 
            (synergy['multi_period_trend'] == 1) & 
            (synergy['psy_std'] < threshold)
        )
        
        # 看跌协同：多数周期PSY<50且同步下降
        synergy['bearish_synergy'] = (
            (synergy['position_majority'] == -1) & 
            (synergy['multi_period_trend'] == -1) & 
            (synergy['psy_std'] < threshold)
        )
        
        # 计算协同得分（0-100）
        # 50分为中性，>50看涨，<50看跌
        synergy['synergy_score'] = 50.0
        
        # 看涨协同得分（最高+30分）
        bullish_score = synergy['bullish_synergy'].astype(int) * 30 * (1 - synergy['psy_std'] / 100)
        synergy['synergy_score'] += bullish_score
        
        # 看跌协同得分（最高-30分）
        bearish_score = synergy['bearish_synergy'].astype(int) * 30 * (1 - synergy['psy_std'] / 100)
        synergy['synergy_score'] -= bearish_score
        
        # 趋势强度得分（最高±20分）
        trend_score = synergy['trend_strength'] * 2 * np.sign(self._result['psy'] - 50)
        synergy['synergy_score'] += trend_score
        
        # 确保得分在0-100范围内
        synergy['synergy_score'] = synergy['synergy_score'].clip(0, 100)
        
        return synergy

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
        
        # 增强版特殊信号
        if self.enhanced:
            # 获取识别的模式
            patterns = self._identify_enhanced_patterns(data)
            
            # 处理增强版特殊模式的信号
            for pattern in patterns:
                pattern_type = pattern.get('type')
                score_impact = pattern.get('score_impact', 0)
                position = pattern.get('position')
                
                if position is not None and position < len(signals):
                    # 只处理对信号有重大影响的模式
                    if score_impact >= 15:  # 强烈看涨信号
                        signals.iloc[position, signals.columns.get_loc('buy_signal')] = True
                        signals.iloc[position, signals.columns.get_loc('neutral_signal')] = False
                        signals.iloc[position, signals.columns.get_loc('trend')] = 1
                        signals.iloc[position, signals.columns.get_loc('signal_type')] = f"增强PSY_{pattern_type}"
                        signals.iloc[position, signals.columns.get_loc('signal_desc')] = pattern.get('description', '')
                        signals.iloc[position, signals.columns.get_loc('confidence')] = min(85 + score_impact/5, 95)
                    elif score_impact <= -15:  # 强烈看跌信号
                        signals.iloc[position, signals.columns.get_loc('sell_signal')] = True
                        signals.iloc[position, signals.columns.get_loc('neutral_signal')] = False
                        signals.iloc[position, signals.columns.get_loc('trend')] = -1
                        signals.iloc[position, signals.columns.get_loc('signal_type')] = f"增强PSY_{pattern_type}"
                        signals.iloc[position, signals.columns.get_loc('signal_desc')] = pattern.get('description', '')
                        signals.iloc[position, signals.columns.get_loc('confidence')] = min(85 + abs(score_impact)/5, 95)
        
        return signals


# 向后兼容的代理类
class EnhancedPSY(PSY):
    """
    增强型心理线指标(Enhanced PSY)
    
    向后兼容的代理类，将请求转发到启用了增强功能的PSY类
    
    具有以下增强特性:
    1. 自适应参数设计：根据市场波动率动态调整PSY的计算周期
    2. 多周期PSY协同分析：结合不同周期的PSY指标提高信号可靠性
    3. 市场氛围评估增强：更精确地评估市场过度乐观/悲观情绪
    4. 形态识别系统：识别PSY极值反转、区间突破和均值回归等形态
    """
    
    def __init__(self, 
                 period: int = 12,
                 secondary_period: int = 24,
                 multi_periods: List[int] = None,
                 adaptive_period: bool = True,
                 volatility_lookback: int = 20):
        """
        初始化增强型PSY指标
        
        Args:
            period: 主要周期，默认为12日
            secondary_period: 次要周期，默认为24日
            multi_periods: 多周期分析参数，默认为[6, 12, 24, 48]
            adaptive_period: 是否启用自适应周期，默认为True
            volatility_lookback: 波动率计算回溯期，默认为20
        """
        import warnings
        warnings.warn(
            "EnhancedPSY类已被弃用，请使用带有enhanced=True参数的PSY类代替",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(period=period, 
                         secondary_period=secondary_period,
                         multi_periods=multi_periods,
                         adaptive_period=adaptive_period,
                         volatility_lookback=volatility_lookback,
                         enhanced=True) 