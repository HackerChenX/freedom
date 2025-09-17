"""
ZXM体系行业轮动指标模块

实现ZXM体系的行业轮动分析指标
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin
from utils.dependency_injection import get_logger

logger = get_logger(__name__)


class ZXMIndustryRotation(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
    """
    ZXM行业轮动指标
    
    分析行业板块轮动情况，识别强势板块和轮动机会
    """
    
    def __init__(self, **kwargs):
        """
        初始化ZXM行业轮动指标
        
        Args:
            **kwargs: 指标参数
        """
        # 移除super().__init__调用，直接设置属性
        self.name = "ZXMIndustryRotation"
        self.description = "ZXM行业轮动指标，分析行业板块轮动情况"
        
        # 设置默认参数
        self._default_parameters = self._get_default_parameters_zxmindustryrotation()
        
        # 应用用户参数
        self.set_parameters_Industry_Rotation(**kwargs)
    
    def _get_default_parameters_zxmindustryrotation(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {
            "momentum_period": 20,
            "strength_period": 10,
            "rotation_threshold": 0.02,
            "relative_strength_period": 5
        }
    
    def set_parameters_Industry_Rotation(self, **kwargs):
        """
        设置指标参数
        
        Args:
            **kwargs: 参数字典
        """
        self.momentum_period = kwargs.get('momentum_period', 20)
        self.strength_period = kwargs.get('strength_period', 10)
        self.rotation_threshold = kwargs.get('rotation_threshold', 0.02)
        self.relative_strength_period = kwargs.get('relative_strength_period', 5)

    @property
    def minimum_periods(self) -> int:
        """
        ZXM行业轮动指标所需的最少数据周期数
        
        Returns:
            int: 最少需要的数据周期数
        """
        return max(self.momentum_period, self.strength_period, self.relative_strength_period) + 10

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算ZXM行业轮动指标的主要入口方法
        
        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 其他参数
            
        Returns:
            包含ZXM行业轮动指标的DataFrame
        """
        result = data.copy()
        
        # 计算板块动量
        result = self._calculate_sector_momentum(result)
        
        # 计算相对强度
        result = self._calculate_relative_strength(result)
        
        # 计算轮动强度
        result = self._calculate_rotation_strength(result)
        
        # 计算综合轮动评分
        result = self._calculate_composite_rotation_score(result)
        
        # 生成轮动信号
        result = self._generate_rotation_signals(result)
        
        return result

    def _calculate_sector_momentum(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算板块动量"""
        result = data.copy()
        
        close = result['close']
        volume = result['volume']
        
        # 价格动量
        price_momentum = close.pct_change(self.momentum_period)
        
        # 短期价格动量
        short_momentum = close.pct_change(self.strength_period)
        
        # 成交量动量
        volume_ma = volume.rolling(window=self.momentum_period).mean()
        volume_momentum = (volume / volume_ma - 1)
        
        # 综合板块动量
        sector_momentum = (
            price_momentum * 0.5 +
            short_momentum * 0.3 +
            volume_momentum * 0.2
        )
        
        result['PriceMomentum'] = price_momentum
        result['ShortMomentum'] = short_momentum
        result['VolumeMomentum'] = volume_momentum
        result['SectorMomentum'] = sector_momentum
        
        return result

    def _calculate_relative_strength(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算相对强度"""
        result = data.copy()
        
        close = result['close']
        
        # 相对强度计算（相对于自身历史）
        ma_short = close.rolling(window=self.relative_strength_period).mean()
        ma_long = close.rolling(window=self.momentum_period).mean()
        
        # 相对强度比率
        relative_strength = ma_short / ma_long
        
        # 相对强度变化率
        rs_change = relative_strength.pct_change(self.strength_period)
        
        # 价格相对位置
        high_period = result['high'].rolling(window=self.momentum_period).max()
        low_period = result['low'].rolling(window=self.momentum_period).min()
        price_position = (close - low_period) / (high_period - low_period + 1e-10)
        
        result['RelativeStrength'] = relative_strength
        result['RSChange'] = rs_change
        result['PricePosition'] = price_position
        
        return result

    def _calculate_rotation_strength(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算轮动强度"""
        result = data.copy()
        
        sector_momentum = result['SectorMomentum']
        relative_strength = result['RelativeStrength']
        price_position = result['PricePosition']
        
        # 轮动强度评分
        rotation_strength = (
            (sector_momentum > self.rotation_threshold).astype(int) * 30 +
            (relative_strength > 1.05).astype(int) * 25 +
            (price_position > 0.7).astype(int) * 20 +
            (sector_momentum > sector_momentum.shift(self.strength_period)).astype(int) * 15 +
            (relative_strength > relative_strength.shift(self.strength_period)).astype(int) * 10
        )
        
        result['RotationStrength'] = rotation_strength
        
        return result

    def _calculate_composite_rotation_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算综合轮动评分"""
        result = data.copy()
        
        # 综合各项轮动指标
        sector_momentum = result['SectorMomentum']
        relative_strength = result['RelativeStrength']
        rotation_strength = result['RotationStrength']
        price_position = result['PricePosition']
        
        # 加权计算综合评分
        composite_score = (
            rotation_strength * 0.4 +
            (sector_momentum * 100).clip(-50, 50) * 0.3 +
            ((relative_strength - 1) * 100).clip(-50, 50) * 0.2 +
            (price_position * 100) * 0.1
        )
        
        # 标准化到0-100范围
        composite_score = np.clip(composite_score, 0, 100)
        
        result['CompositeRotationScore'] = composite_score
        
        return result

    def _generate_rotation_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成轮动信号"""
        result = data.copy()
        
        composite_score = result['CompositeRotationScore']
        sector_momentum = result['SectorMomentum']
        relative_strength = result['RelativeStrength']
        
        # 强轮动领导信号
        result['StrongRotationLeaderSignal'] = (
            (composite_score >= 80) & 
            (sector_momentum > self.rotation_threshold) & 
            (relative_strength > 1.1)
        )
        
        # 中等轮动信号
        result['ModerateRotationSignal'] = (
            (composite_score >= 60) & (composite_score < 80) &
            ((sector_momentum > self.rotation_threshold * 0.5) | 
             (relative_strength > 1.05))
        )
        
        # 轮动衰退信号
        result['RotationFadingSignal'] = (
            (composite_score < 30) & 
            (sector_momentum < 0) & 
            (relative_strength < 0.95)
        )
        
        # 综合轮动判断
        result['IsRotationLeader'] = composite_score >= 70
        
        return result

    def _calculate_baseindicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        BaseIndicator要求的抽象方法实现
        
        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 其他参数
            
        Returns:
            包含ZXM行业轮动指标的DataFrame
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
        return 0.8  # ZXM行业轮动指标置信度
    
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
        return result['CompositeRotationScore']
    
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
        
        # 轮动形态
        composite_score = result['CompositeRotationScore']
        patterns['ZXM_STRONG_ROTATION_LEADER'] = composite_score >= 80
        patterns['ZXM_MODERATE_ROTATION'] = (composite_score >= 60) & (composite_score < 80)
        patterns['ZXM_WEAK_ROTATION'] = (composite_score >= 40) & (composite_score < 60)
        patterns['ZXM_ROTATION_LAGGARD'] = composite_score < 40
        
        # 轮动信号形态
        patterns['ZXM_STRONG_ROTATION_LEADER_SIGNAL'] = result['StrongRotationLeaderSignal']
        patterns['ZXM_MODERATE_ROTATION_SIGNAL'] = result['ModerateRotationSignal']
        patterns['ZXM_ROTATION_FADING'] = result['RotationFadingSignal']
        
        return patterns
    
    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """
        BaseIndicator要求的参数设置方法
        
        Args:
            **kwargs: 参数字典
        """
        self.set_parameters_Industry_Rotation(**kwargs)
