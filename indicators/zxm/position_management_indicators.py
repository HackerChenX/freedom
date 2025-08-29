"""
ZXM体系仓位管理指标模块

实现ZXM体系的仓位管理分析指标
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin
from utils.dependency_injection import get_logger

logger = get_logger(__name__)


class ZXMPositionManagement(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
    """
    ZXM仓位管理指标
    
    分析最优仓位配置，提供动态仓位管理策略
    """
    
    def __init__(self, **kwargs):
        """
        初始化ZXM仓位管理指标
        
        Args:
            **kwargs: 指标参数
        """
        # 移除super().__init__调用，直接设置属性
        self.name = "ZXMPositionManagement"
        self.description = "ZXM仓位管理指标，分析最优仓位配置"
        
        # 设置默认参数
        self._default_parameters = self._get_default_parameters_zxmpositionmanagement()
        
        # 应用用户参数
        self.set_parameters_Position_Management(**kwargs)
    
    def _get_default_parameters_zxmpositionmanagement(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {
            "risk_period": 20,
            "volatility_period": 14,
            "trend_period": 30,
            "max_position": 0.8,
            "min_position": 0.1
        }
    
    def set_parameters_Position_Management(self, **kwargs):
        """
        设置指标参数
        
        Args:
            **kwargs: 参数字典
        """
        self.risk_period = kwargs.get('risk_period', 20)
        self.volatility_period = kwargs.get('volatility_period', 14)
        self.trend_period = kwargs.get('trend_period', 30)
        self.max_position = kwargs.get('max_position', 0.8)
        self.min_position = kwargs.get('min_position', 0.1)

    @property
    def minimum_periods(self) -> int:
        """
        ZXM仓位管理指标所需的最少数据周期数
        
        Returns:
            int: 最少需要的数据周期数
        """
        return max(self.risk_period, self.volatility_period, self.trend_period) + 10

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算ZXM仓位管理指标的主要入口方法
        
        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 其他参数
            
        Returns:
            包含ZXM仓位管理指标的DataFrame
        """
        result = data.copy()
        
        # 计算风险评估
        result = self._calculate_risk_assessment(result)
        
        # 计算趋势强度
        result = self._calculate_trend_strength(result)
        
        # 计算波动率调整
        result = self._calculate_volatility_adjustment(result)
        
        # 计算最优仓位大小
        result = self._calculate_optimal_position_size(result)
        
        # 生成仓位管理信号
        result = self._generate_position_signals(result)
        
        return result

    def _calculate_risk_assessment(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算风险评估"""
        result = data.copy()
        
        close = result['close']
        high = result['high']
        low = result['low']
        
        # 价格风险
        returns = close.pct_change()
        price_risk = returns.rolling(window=self.risk_period).std() * np.sqrt(252)
        
        # 回撤风险
        cumulative_max = close.expanding().max()
        drawdown = (close - cumulative_max) / cumulative_max
        max_drawdown = drawdown.rolling(window=self.risk_period).min()
        
        # 波动风险
        true_range = np.maximum(high - low, 
                               np.maximum(abs(high - close.shift(1)), 
                                         abs(low - close.shift(1))))
        atr = true_range.rolling(window=self.volatility_period).mean()
        volatility_risk = atr / close
        
        # 综合风险评分
        risk_score = (
            price_risk * 0.4 +
            (-max_drawdown) * 0.35 +
            volatility_risk * 0.25
        ) * 100
        
        result['PriceRisk'] = price_risk
        result['MaxDrawdown'] = max_drawdown
        result['VolatilityRisk'] = volatility_risk
        result['RiskScore'] = risk_score.fillna(50)
        
        return result

    def _calculate_trend_strength(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算趋势强度"""
        result = data.copy()
        
        close = result['close']
        
        # 趋势方向
        ma_short = close.rolling(window=self.volatility_period).mean()
        ma_long = close.rolling(window=self.trend_period).mean()
        trend_direction = (ma_short / ma_long - 1) * 100
        
        # 趋势一致性
        price_above_ma = (close > ma_short).rolling(window=self.volatility_period).mean()
        
        # 趋势强度
        momentum = close.pct_change(self.volatility_period)
        trend_strength = (
            abs(trend_direction) * 0.5 +
            price_above_ma * 50 * 0.3 +
            abs(momentum) * 100 * 0.2
        )
        
        result['TrendDirection'] = trend_direction
        result['TrendConsistency'] = price_above_ma
        result['TrendStrength'] = trend_strength.fillna(25)
        
        return result

    def _calculate_volatility_adjustment(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算波动率调整"""
        result = data.copy()
        
        close = result['close']
        
        # 历史波动率
        returns = close.pct_change()
        historical_vol = returns.rolling(window=self.volatility_period).std() * np.sqrt(252)
        
        # 波动率分位数
        vol_percentile = historical_vol.rolling(window=self.trend_period).rank(pct=True)
        
        # 波动率调整因子
        vol_adjustment = 1 - (vol_percentile - 0.5) * 0.5  # 高波动率降低仓位
        vol_adjustment = np.clip(vol_adjustment, 0.3, 1.5)
        
        result['HistoricalVolatility'] = historical_vol
        result['VolatilityPercentile'] = vol_percentile
        result['VolatilityAdjustment'] = vol_adjustment.fillna(1.0)
        
        return result

    def _calculate_optimal_position_size(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算最优仓位大小"""
        result = data.copy()
        
        # 基础仓位（基于趋势强度）
        trend_strength = result['TrendStrength']
        base_position = (trend_strength / 100) * self.max_position
        base_position = np.clip(base_position, self.min_position, self.max_position)
        
        # 风险调整
        risk_score = result['RiskScore']
        risk_adjustment = 1 - (risk_score / 100) * 0.5  # 高风险降低仓位
        risk_adjustment = np.clip(risk_adjustment, 0.2, 1.0)
        
        # 波动率调整
        vol_adjustment = result['VolatilityAdjustment']
        
        # 最终仓位大小
        position_size = base_position * risk_adjustment * vol_adjustment
        position_size = np.clip(position_size, self.min_position, self.max_position)
        
        # 仓位权重
        position_weight = position_size / self.max_position
        
        # 配置评分
        allocation_score = (
            trend_strength * 0.4 +
            (100 - risk_score) * 0.35 +
            (vol_adjustment - 0.5) * 100 * 0.25
        )
        allocation_score = np.clip(allocation_score, 0, 100)
        
        result['BasePosition'] = base_position
        result['RiskAdjustment'] = risk_adjustment
        result['PositionSize'] = position_size
        result['PositionWeight'] = position_weight
        result['AllocationScore'] = allocation_score.fillna(50)
        
        return result

    def _generate_position_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成仓位管理信号"""
        result = data.copy()
        
        position_size = result['PositionSize']
        risk_score = result['RiskScore']
        trend_strength = result['TrendStrength']
        allocation_score = result['AllocationScore']
        
        # 仓位调整信号
        position_change = position_size.diff()
        result['PositionAdjustSignal'] = abs(position_change) > 0.1
        
        # 风险控制信号
        result['RiskControlSignal'] = (
            (risk_score >= 80) |  # 高风险
            (position_size <= self.min_position * 1.2)  # 接近最小仓位
        )
        
        # 加仓信号
        result['IncreasePositionSignal'] = (
            (trend_strength >= 70) &
            (risk_score <= 40) &
            (position_size < self.max_position * 0.8)
        )
        
        # 减仓信号
        result['DecreasePositionSignal'] = (
            (trend_strength <= 30) |
            (risk_score >= 70) |
            (position_size > self.max_position * 0.9)
        )
        
        # 配置调整信号
        allocation_change = allocation_score.diff()
        result['AllocationAdjustSignal'] = abs(allocation_change) > 15
        
        # 综合仓位判断
        result['IsOptimalPosition'] = allocation_score >= 70
        result['IsRiskyPosition'] = allocation_score <= 30
        
        return result

    def _calculate_baseindicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        BaseIndicator要求的抽象方法实现
        
        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 其他参数
            
        Returns:
            包含ZXM仓位管理指标的DataFrame
        """
        return self.calculate(data, **kwargs)
    
    def calculate_confidence_Indicator_Base_Indicator(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """
        BaseIndicator要求的置信度计算方法
        
        Args:
            score: 得分序列
            patterns: 检测到的形态DataFrame
            signals: 生成的信号字典
            
        Returns:
            float: 置信度分数 (0-1)
        """
        return 0.88  # ZXM仓位管理指标置信度
    
    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        BaseIndicator要求的原始评分计算方法
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 原始评分序列
        """
        result = self.calculate(data, **kwargs)
        return result['AllocationScore']
    
    def get_patterns_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        BaseIndicator要求的形态获取方法
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.DataFrame: 形态DataFrame
        """
        result = self.calculate(data, **kwargs)
        patterns = pd.DataFrame(index=data.index)
        
        # 仓位管理形态
        allocation_score = result['AllocationScore']
        patterns['ZXM_OPTIMAL_ALLOCATION'] = allocation_score >= 80
        patterns['ZXM_GOOD_ALLOCATION'] = (allocation_score >= 60) & (allocation_score < 80)
        patterns['ZXM_NEUTRAL_ALLOCATION'] = (allocation_score >= 40) & (allocation_score < 60)
        patterns['ZXM_POOR_ALLOCATION'] = (allocation_score >= 20) & (allocation_score < 40)
        patterns['ZXM_RISKY_ALLOCATION'] = allocation_score < 20
        
        # 仓位管理信号形态
        patterns['ZXM_POSITION_ADJUST'] = result['PositionAdjustSignal']
        patterns['ZXM_RISK_CONTROL'] = result['RiskControlSignal']
        patterns['ZXM_INCREASE_POSITION'] = result['IncreasePositionSignal']
        patterns['ZXM_DECREASE_POSITION'] = result['DecreasePositionSignal']
        patterns['ZXM_ALLOCATION_ADJUST'] = result['AllocationAdjustSignal']
        
        return patterns
    
    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """
        BaseIndicator要求的参数设置方法
        
        Args:
            **kwargs: 参数字典
        """
        self.set_parameters_Position_Management(**kwargs)
