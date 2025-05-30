"""
增强型心理线指标(EnhancedPSY)模块

实现增强型PSY指标计算，提供自适应参数、多周期协同分析、市场氛围评估和形态识别功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple

from indicators.base_indicator import BaseIndicator, MarketEnvironment
from indicators.psy import PSY
from utils.logger import get_logger

logger = get_logger(__name__)


class EnhancedPSY(PSY):
    """
    增强型心理线指标(Enhanced PSY)
    
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
        super().__init__(period=period)
        self.name = "EnhancedPSY"
        self.description = "增强型心理线指标，优化参数自适应性，增加多周期协同分析和市场氛围评估"
        self.secondary_period = secondary_period
        self.multi_periods = multi_periods or [6, 12, 24, 48]
        self.adaptive_period = adaptive_period
        self.volatility_lookback = volatility_lookback
        self.market_environment = "normal"
        
        # 内部变量
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
        计算增强型PSY指标
        
        Args:
            data: 输入数据，包含OHLC数据
            
        Returns:
            pd.DataFrame: 计算结果，包含PSY及其相关指标
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close"])
        
        # 如果启用自适应周期，则调整参数
        if self.adaptive_period:
            self._adjust_parameters_by_volatility(data)
        
        # 使用调整后的周期计算主要PSY
        # 重新计算而不是调用super().calculate，以使用自适应周期
        result = pd.DataFrame(index=data.index)
        
        # 计算价格变化
        price_change = data["close"].diff()
        
        # 统计上涨日数
        up_days = (price_change > 0).astype(int)
        
        # 计算PSY：N日内上涨天数 / N * 100
        result["psy"] = up_days.rolling(window=self._adaptive_period).sum() / self._adaptive_period * 100
        
        # 计算PSY的移动平均线（作为信号线）
        result["psyma"] = result["psy"].rolling(window=int(self._adaptive_period/2)).mean()
        
        # 额外计算：PSY变化率
        result["psy_change"] = result["psy"].diff()
        
        # 计算次要周期PSY
        secondary_psy = PSY(period=self.secondary_period)
        secondary_result = secondary_psy.calculate(data)
        result["psy_secondary"] = secondary_result["psy"]
        result["psyma_secondary"] = secondary_result["psyma"]
        self._secondary_psy = result["psy_secondary"]
        
        # 计算多周期PSY
        for period in self.multi_periods:
            if period != self._adaptive_period and period != self.secondary_period:
                multi_psy = PSY(period=period)
                multi_result = multi_psy.calculate(data)
                result[f"psy_{period}"] = multi_result["psy"]
                result[f"psyma_{period}"] = multi_result["psyma"]
                self._multi_period_psy[period] = result[f"psy_{period}"]
        
        # 计算PSY动态特性
        result["psy_momentum"] = result["psy"] - result["psy"].shift(3)
        result["psy_slope"] = self._calculate_slope(result["psy"], 5)
        result["psy_accel"] = result["psy_slope"] - result["psy_slope"].shift(1)
        
        # 计算市场氛围指标
        result["market_sentiment"] = self._calculate_market_sentiment(result["psy"])
        
        # 计算均值回归特性
        result["mean_reversion"] = self._calculate_mean_reversion(result["psy"])
        
        # 保存结果
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
        reversion_strength = -1 * np.sign(distance_from_mean) * distance_change
        
        # 平滑处理
        reversion_strength = reversion_strength.rolling(window=3).mean()
        
        return reversion_strength
    
    def analyze_multi_period_synergy(self, threshold: float = 10.0) -> pd.DataFrame:
        """
        分析多周期PSY协同性
        
        Args:
            threshold: 协同判断阈值，默认为10.0
            
        Returns:
            pd.DataFrame: 协同分析结果
        """
        if self._result is None:
            return pd.DataFrame()
        
        result = pd.DataFrame(index=self._result.index)
        
        # 基本PSY值
        psy = self._result["psy"]
        
        # 次要周期PSY值
        psy_secondary = self._result["psy_secondary"]
        
        # 计算主周期和次要周期的一致性
        agreement = pd.Series(0, index=self._result.index)
        
        # 主次周期都高于50，看涨一致
        bullish_agreement = (psy > 50) & (psy_secondary > 50)
        agreement.loc[bullish_agreement] = 1
        
        # 主次周期都低于50，看跌一致
        bearish_agreement = (psy < 50) & (psy_secondary < 50)
        agreement.loc[bearish_agreement] = -1
        
        # 计算多周期一致性强度
        agreement_strength = pd.Series(0.0, index=self._result.index)
        
        # 多周期PSY同向偏离中值的程度
        for i in range(len(agreement)):
            if pd.isna(agreement.iloc[i]):
                continue
                
            if agreement.iloc[i] == 1:  # 看涨一致
                # 计算各周期PSY超过50的平均偏离度
                deviation = psy.iloc[i] - 50
                deviation += psy_secondary.iloc[i] - 50
                
                for period in self.multi_periods:
                    if f"psy_{period}" in self._result.columns:
                        period_psy = self._result[f"psy_{period}"].iloc[i]
                        if not pd.isna(period_psy) and period_psy > 50:
                            deviation += period_psy - 50
                
                # 计算平均偏离度作为一致性强度
                agreement_strength.iloc[i] = deviation / (2 + len([p for p in self.multi_periods 
                                                                if f"psy_{p}" in self._result.columns]))
                
            elif agreement.iloc[i] == -1:  # 看跌一致
                # 计算各周期PSY低于50的平均偏离度
                deviation = 50 - psy.iloc[i]
                deviation += 50 - psy_secondary.iloc[i]
                
                for period in self.multi_periods:
                    if f"psy_{period}" in self._result.columns:
                        period_psy = self._result[f"psy_{period}"].iloc[i]
                        if not pd.isna(period_psy) and period_psy < 50:
                            deviation += 50 - period_psy
                
                # 计算平均偏离度作为一致性强度（负值表示看跌一致）
                agreement_strength.iloc[i] = -deviation / (2 + len([p for p in self.multi_periods 
                                                                  if f"psy_{p}" in self._result.columns]))
        
        # 保存结果
        result["multi_period_agreement"] = agreement
        result["agreement_strength"] = agreement_strength
        
        # 多周期拐点分析
        turning_points = pd.Series(0, index=self._result.index)
        
        # 计算各周期PSY的斜率
        main_slope = self._calculate_slope(psy, 3)
        secondary_slope = self._calculate_slope(psy_secondary, 3)
        
        # 斜率同向且强度大于阈值，表示明确拐点
        clear_up_turn = (main_slope > threshold/100) & (secondary_slope > threshold/100)
        turning_points.loc[clear_up_turn] = 2  # 强上升拐点
        
        moderate_up_turn = (main_slope > 0) & (secondary_slope > 0) & ~clear_up_turn
        turning_points.loc[moderate_up_turn] = 1  # 中等上升拐点
        
        clear_down_turn = (main_slope < -threshold/100) & (secondary_slope < -threshold/100)
        turning_points.loc[clear_down_turn] = -2  # 强下降拐点
        
        moderate_down_turn = (main_slope < 0) & (secondary_slope < 0) & ~clear_down_turn
        turning_points.loc[moderate_down_turn] = -1  # 中等下降拐点
        
        # 主次周期斜率不一致，可能是反转信号
        potential_reversal = (main_slope * secondary_slope < 0)
        turning_points.loc[potential_reversal] = 0  # 不一致，可能即将反转
        
        result["turning_points"] = turning_points
        
        return result
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[Dict[str, Any]]:
        """
        识别PSY技术形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[Dict[str, Any]]: 识别出的形态列表，每个形态包含类型、位置、评分等信息
        """
        patterns = []
        
        # 确保已计算PSY
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None or len(self._result) < 10:
            return patterns
        
        psy = self._result["psy"]
        psyma = self._result["psyma"]
        
        # 最近的数据点索引
        latest_idx = psy.index[-1]
        
        # 1. 超买超卖状态形态
        if psy.iloc[-1] > 75:
            patterns.append({
                "type": "overbought",
                "subtype": "extreme" if psy.iloc[-1] > 85 else "normal",
                "position": latest_idx,
                "description": f"PSY超买状态({psy.iloc[-1]:.2f})",
                "score": -0.7 * (psy.iloc[-1] - 75),  # 超买程度越高，负面评分越大
                "confidence": min(60 + (psy.iloc[-1] - 75) * 1.5, 95)
            })
            
            # 检测超买区域回落形态
            if psy.iloc[-1] < psy.iloc[-2] and psy.iloc[-2] > 75:
                patterns.append({
                    "type": "overbought_pullback",
                    "position": latest_idx,
                    "description": "PSY从超买区域回落",
                    "score": -15,
                    "confidence": 70
                })
                
        elif psy.iloc[-1] < 25:
            patterns.append({
                "type": "oversold",
                "subtype": "extreme" if psy.iloc[-1] < 15 else "normal",
                "position": latest_idx,
                "description": f"PSY超卖状态({psy.iloc[-1]:.2f})",
                "score": 0.7 * (25 - psy.iloc[-1]),  # 超卖程度越高，正面评分越大
                "confidence": min(60 + (25 - psy.iloc[-1]) * 1.5, 95)
            })
            
            # 检测超卖区域回升形态
            if psy.iloc[-1] > psy.iloc[-2] and psy.iloc[-2] < 25:
                patterns.append({
                    "type": "oversold_bounce",
                    "position": latest_idx,
                    "description": "PSY从超卖区域回升",
                    "score": 15,
                    "confidence": 70
                })
        
        # 2. 交叉形态
        if psy.iloc[-1] > psyma.iloc[-1] and psy.iloc[-2] <= psyma.iloc[-2]:
            # 金叉
            cross_score = 10
            cross_confidence = 65
            
            # 金叉位置增强评分
            if psy.iloc[-1] < 30:  # 低位金叉，看涨信号更强
                cross_score += 10
                cross_confidence += 10
                cross_type = "golden_cross_low"
                cross_desc = "PSY低位金叉信号线"
            elif psy.iloc[-1] > 70:  # 高位金叉，看涨信号较弱
                cross_score += 5
                cross_type = "golden_cross_high"
                cross_desc = "PSY高位金叉信号线"
            else:  # 中位金叉
                cross_score += 8
                cross_confidence += 5
                cross_type = "golden_cross_mid"
                cross_desc = "PSY中位金叉信号线"
            
            patterns.append({
                "type": cross_type,
                "position": latest_idx,
                "description": cross_desc,
                "score": cross_score,
                "confidence": cross_confidence
            })
            
        elif psy.iloc[-1] < psyma.iloc[-1] and psy.iloc[-2] >= psyma.iloc[-2]:
            # 死叉
            cross_score = -10
            cross_confidence = 65
            
            # 死叉位置增强评分
            if psy.iloc[-1] > 70:  # 高位死叉，看跌信号更强
                cross_score -= 10
                cross_confidence += 10
                cross_type = "death_cross_high"
                cross_desc = "PSY高位死叉信号线"
            elif psy.iloc[-1] < 30:  # 低位死叉，看跌信号较弱
                cross_score -= 5
                cross_type = "death_cross_low"
                cross_desc = "PSY低位死叉信号线"
            else:  # 中位死叉
                cross_score -= 8
                cross_confidence += 5
                cross_type = "death_cross_mid"
                cross_desc = "PSY中位死叉信号线"
            
            patterns.append({
                "type": cross_type,
                "position": latest_idx,
                "description": cross_desc,
                "score": cross_score,
                "confidence": cross_confidence
            })
        
        # 3. 均值回归形态
        mean_reversion = self._result["mean_reversion"].iloc[-1]
        if not pd.isna(mean_reversion) and abs(mean_reversion) > 3:
            if mean_reversion > 0:  # 向50回归
                patterns.append({
                    "type": "mean_reversion_up" if psy.iloc[-1] < 50 else "mean_reversion_down",
                    "position": latest_idx,
                    "description": "PSY向均值回归" + ("(上升)" if psy.iloc[-1] < 50 else "(下降)"),
                    "score": 5 if psy.iloc[-1] < 50 else -5,
                    "confidence": 60
                })
        
        # 4. 形态变化形态
        recent_psy = psy.iloc[-5:]
        
        # V形反转（底部）
        if (recent_psy.iloc[0] > recent_psy.iloc[1] and 
            recent_psy.iloc[1] > recent_psy.iloc[2] and
            recent_psy.iloc[2] < recent_psy.iloc[3] and
            recent_psy.iloc[3] < recent_psy.iloc[4] and
            recent_psy.iloc[2] < 40):
            
            patterns.append({
                "type": "v_bottom",
                "position": latest_idx,
                "description": "PSY形成V形底部反转",
                "score": 20,
                "confidence": 75
            })
        
        # 倒V形反转（顶部）
        if (recent_psy.iloc[0] < recent_psy.iloc[1] and 
            recent_psy.iloc[1] < recent_psy.iloc[2] and
            recent_psy.iloc[2] > recent_psy.iloc[3] and
            recent_psy.iloc[3] > recent_psy.iloc[4] and
            recent_psy.iloc[2] > 60):
            
            patterns.append({
                "type": "v_top",
                "position": latest_idx,
                "description": "PSY形成倒V形顶部反转",
                "score": -20,
                "confidence": 75
            })
        
        # 5. 极端情绪形态
        sentiment = self._result["market_sentiment"].iloc[-1]
        if not pd.isna(sentiment):
            if sentiment > 60:
                patterns.append({
                    "type": "extreme_optimism",
                    "position": latest_idx,
                    "description": f"PSY显示市场极度乐观(情绪值:{sentiment:.2f})",
                    "score": -15,
                    "confidence": 70
                })
            elif sentiment < -60:
                patterns.append({
                    "type": "extreme_pessimism",
                    "position": latest_idx,
                    "description": f"PSY显示市场极度悲观(情绪值:{sentiment:.2f})",
                    "score": 15,
                    "confidence": 70
                })
        
        # 6. 多周期协同形态
        if not pd.isna(self._secondary_psy) and len(self._secondary_psy) > 0:
            # 主次周期协同上升
            if (psy.iloc[-1] > psy.iloc[-2] > psy.iloc[-3] and
                self._secondary_psy.iloc[-1] > self._secondary_psy.iloc[-2]):
                
                patterns.append({
                    "type": "multi_period_bullish",
                    "position": latest_idx,
                    "description": "PSY多周期协同上升",
                    "score": 12,
                    "confidence": 70
                })
            
            # 主次周期协同下降
            if (psy.iloc[-1] < psy.iloc[-2] < psy.iloc[-3] and
                self._secondary_psy.iloc[-1] < self._secondary_psy.iloc[-2]):
                
                patterns.append({
                    "type": "multi_period_bearish",
                    "position": latest_idx,
                    "description": "PSY多周期协同下降",
                    "score": -12,
                    "confidence": 70
                })
        
        return patterns
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算PSY原始评分，增强版
        
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
        
        score = pd.Series(50.0, index=self._result.index)  # 基础分50分
        
        # 1. 多周期协同分析评分（占比30%）
        synergy_result = self.analyze_multi_period_synergy()
        if not synergy_result.empty and "agreement_strength" in synergy_result.columns:
            # 协同强度转换为评分
            agreement_score = synergy_result["agreement_strength"] * 0.3  # 协同强度转换为加减分
            score += agreement_score
        
        # 2. 超买超卖评分（占比20%）
        psy = self._result["psy"]
        overbought_score = -0.3 * np.maximum(0, (psy - 75))  # 超买区域减分
        oversold_score = 0.3 * np.maximum(0, (25 - psy))     # 超卖区域加分
        score += overbought_score + oversold_score
        
        # 3. PSY动态评分（占比20%）
        momentum = self._result["psy_momentum"]
        slope = self._result["psy_slope"]
        accel = self._result["psy_accel"]
        
        # 动量评分
        momentum_score = momentum * 0.7
        
        # 斜率评分
        slope_score = slope * 50
        
        # 加速度评分
        accel_score = accel * 20
        
        score += momentum_score + slope_score + accel_score
        
        # 4. 市场情绪评分（占比15%）
        sentiment = self._result["market_sentiment"]
        sentiment_score = -sentiment * 0.075  # 市场情绪过高时减分，过低时加分（反向）
        score += sentiment_score
        
        # 5. 均值回归评分（占比15%）
        mean_reversion = self._result["mean_reversion"]
        mean_reversion_score = mean_reversion * 5  # 向均值回归时根据方向加减分
        score += mean_reversion_score
        
        # 6. 根据市场环境调整评分
        if self.market_environment == 'bull_market':
            # 牛市中提高低位PSY的评分
            bull_adj = np.where(psy < 40, (40 - psy) * 0.25, 0)
            score += bull_adj
        elif self.market_environment == 'bear_market':
            # 熊市中降低高位PSY的评分
            bear_adj = np.where(psy > 60, -(psy - 60) * 0.25, 0)
            score += bear_adj
        elif self.market_environment == 'volatile_market':
            # 波动市场中，减少极端值的影响
            volatile_adj = np.where(abs(psy - 50) > 20, -(abs(psy - 50) - 20) * 0.15, 0)
            score += volatile_adj
        
        # 确保评分在0-100之间
        return np.clip(score, 0, 100)
    
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
        
        # 获取PSY相关数据
        psy = self._result['psy']
        psyma = self._result['psyma']
        
        # 识别形态
        patterns = self.identify_patterns(data)
        
        # 根据形态和评分生成信号
        for i in range(len(signals)):
            # 如果数据点不足，跳过
            if i < 5 or pd.isna(score.iloc[i]):
                continue
            
            current_patterns = [p for p in patterns if p['position'] == signals.index[i]]
            
            # 如果有形态，使用形态生成信号
            if current_patterns:
                # 计算总评分和总可信度
                pattern_score = sum([p['score'] for p in current_patterns])
                weighted_confidence = sum([p['score'] * p['confidence'] for p in current_patterns if p['score'] != 0])
                total_abs_score = sum([abs(p['score']) for p in current_patterns])
                
                if total_abs_score > 0:
                    avg_confidence = weighted_confidence / total_abs_score
                else:
                    avg_confidence = 50.0
                
                # 更新信号
                if pattern_score > 15:  # 强烈买入信号
                    signals.iloc[i, signals.columns.get_loc('buy_signal')] = True
                    signals.iloc[i, signals.columns.get_loc('neutral_signal')] = False
                    signals.iloc[i, signals.columns.get_loc('trend')] = 1
                    signals.iloc[i, signals.columns.get_loc('signal_type')] = '形态买入'
                    signals.iloc[i, signals.columns.get_loc('signal_desc')] = ', '.join([p['description'] for p in current_patterns if p['score'] > 0])
                    signals.iloc[i, signals.columns.get_loc('confidence')] = avg_confidence
                
                elif pattern_score < -15:  # 强烈卖出信号
                    signals.iloc[i, signals.columns.get_loc('sell_signal')] = True
                    signals.iloc[i, signals.columns.get_loc('neutral_signal')] = False
                    signals.iloc[i, signals.columns.get_loc('trend')] = -1
                    signals.iloc[i, signals.columns.get_loc('signal_type')] = '形态卖出'
                    signals.iloc[i, signals.columns.get_loc('signal_desc')] = ', '.join([p['description'] for p in current_patterns if p['score'] < 0])
                    signals.iloc[i, signals.columns.get_loc('confidence')] = avg_confidence
                
                elif pattern_score > 5:  # 轻微买入信号
                    signals.iloc[i, signals.columns.get_loc('buy_signal')] = True
                    signals.iloc[i, signals.columns.get_loc('neutral_signal')] = False
                    signals.iloc[i, signals.columns.get_loc('trend')] = 0.5
                    signals.iloc[i, signals.columns.get_loc('signal_type')] = '轻微买入'
                    signals.iloc[i, signals.columns.get_loc('signal_desc')] = ', '.join([p['description'] for p in current_patterns if p['score'] > 0])
                    signals.iloc[i, signals.columns.get_loc('confidence')] = avg_confidence
                
                elif pattern_score < -5:  # 轻微卖出信号
                    signals.iloc[i, signals.columns.get_loc('sell_signal')] = True
                    signals.iloc[i, signals.columns.get_loc('neutral_signal')] = False
                    signals.iloc[i, signals.columns.get_loc('trend')] = -0.5
                    signals.iloc[i, signals.columns.get_loc('signal_type')] = '轻微卖出'
                    signals.iloc[i, signals.columns.get_loc('signal_desc')] = ', '.join([p['description'] for p in current_patterns if p['score'] < 0])
                    signals.iloc[i, signals.columns.get_loc('confidence')] = avg_confidence
            
            # 没有形态时，使用评分和传统信号生成逻辑
            else:
                # 1. 超买超卖信号
                if psy.iloc[i] > 75 and psy.iloc[i-1] <= 75:
                    signals.iloc[i, signals.columns.get_loc('sell_signal')] = True
                    signals.iloc[i, signals.columns.get_loc('neutral_signal')] = False
                    signals.iloc[i, signals.columns.get_loc('trend')] = -1
                    signals.iloc[i, signals.columns.get_loc('signal_type')] = 'PSY超买'
                    signals.iloc[i, signals.columns.get_loc('signal_desc')] = f'PSY进入超买区域({psy.iloc[i]:.2f})'
                    signals.iloc[i, signals.columns.get_loc('confidence')] = 70
                
                elif psy.iloc[i] < 25 and psy.iloc[i-1] >= 25:
                    signals.iloc[i, signals.columns.get_loc('buy_signal')] = True
                    signals.iloc[i, signals.columns.get_loc('neutral_signal')] = False
                    signals.iloc[i, signals.columns.get_loc('trend')] = 1
                    signals.iloc[i, signals.columns.get_loc('signal_type')] = 'PSY超卖'
                    signals.iloc[i, signals.columns.get_loc('signal_desc')] = f'PSY进入超卖区域({psy.iloc[i]:.2f})'
                    signals.iloc[i, signals.columns.get_loc('confidence')] = 70
                
                # 2. 交叉信号
                elif psy.iloc[i] > psyma.iloc[i] and psy.iloc[i-1] <= psyma.iloc[i-1]:
                    signals.iloc[i, signals.columns.get_loc('buy_signal')] = True
                    signals.iloc[i, signals.columns.get_loc('neutral_signal')] = False
                    signals.iloc[i, signals.columns.get_loc('trend')] = 0.7
                    signals.iloc[i, signals.columns.get_loc('signal_type')] = 'PSY金叉'
                    signals.iloc[i, signals.columns.get_loc('signal_desc')] = 'PSY上穿信号线'
                    signals.iloc[i, signals.columns.get_loc('confidence')] = 65
                
                elif psy.iloc[i] < psyma.iloc[i] and psy.iloc[i-1] >= psyma.iloc[i-1]:
                    signals.iloc[i, signals.columns.get_loc('sell_signal')] = True
                    signals.iloc[i, signals.columns.get_loc('neutral_signal')] = False
                    signals.iloc[i, signals.columns.get_loc('trend')] = -0.7
                    signals.iloc[i, signals.columns.get_loc('signal_type')] = 'PSY死叉'
                    signals.iloc[i, signals.columns.get_loc('signal_desc')] = 'PSY下穿信号线'
                    signals.iloc[i, signals.columns.get_loc('confidence')] = 65
                
                # 3. 基于综合评分的信号
                elif score.iloc[i] > 80:
                    signals.iloc[i, signals.columns.get_loc('buy_signal')] = True
                    signals.iloc[i, signals.columns.get_loc('neutral_signal')] = False
                    signals.iloc[i, signals.columns.get_loc('trend')] = 1
                    signals.iloc[i, signals.columns.get_loc('signal_type')] = 'PSY强烈买入'
                    signals.iloc[i, signals.columns.get_loc('signal_desc')] = f'PSY综合评分超过80({score.iloc[i]:.2f})'
                    signals.iloc[i, signals.columns.get_loc('confidence')] = 80
                
                elif score.iloc[i] < 20:
                    signals.iloc[i, signals.columns.get_loc('sell_signal')] = True
                    signals.iloc[i, signals.columns.get_loc('neutral_signal')] = False
                    signals.iloc[i, signals.columns.get_loc('trend')] = -1
                    signals.iloc[i, signals.columns.get_loc('signal_type')] = 'PSY强烈卖出'
                    signals.iloc[i, signals.columns.get_loc('signal_desc')] = f'PSY综合评分低于20({score.iloc[i]:.2f})'
                    signals.iloc[i, signals.columns.get_loc('confidence')] = 80
                
                elif score.iloc[i] > 65:
                    signals.iloc[i, signals.columns.get_loc('buy_signal')] = True
                    signals.iloc[i, signals.columns.get_loc('neutral_signal')] = False
                    signals.iloc[i, signals.columns.get_loc('trend')] = 0.5
                    signals.iloc[i, signals.columns.get_loc('signal_type')] = 'PSY轻微买入'
                    signals.iloc[i, signals.columns.get_loc('signal_desc')] = f'PSY综合评分较高({score.iloc[i]:.2f})'
                    signals.iloc[i, signals.columns.get_loc('confidence')] = 60
                
                elif score.iloc[i] < 35:
                    signals.iloc[i, signals.columns.get_loc('sell_signal')] = True
                    signals.iloc[i, signals.columns.get_loc('neutral_signal')] = False
                    signals.iloc[i, signals.columns.get_loc('trend')] = -0.5
                    signals.iloc[i, signals.columns.get_loc('signal_type')] = 'PSY轻微卖出'
                    signals.iloc[i, signals.columns.get_loc('signal_desc')] = f'PSY综合评分较低({score.iloc[i]:.2f})'
                    signals.iloc[i, signals.columns.get_loc('confidence')] = 60
        
        return signals 