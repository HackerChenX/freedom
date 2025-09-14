"""
ZXM体系风险控制指标模块

实现ZXM体系的风险控制分析指标
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin
from utils.dependency_injection import get_logger
from utils.numerical_stability_manager import get_stability_manager

logger = get_logger(__name__)


class ZXMRiskControl(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
    """
    ZXM风险控制指标
    
    分析股票的风险水平，提供风险控制信号和预警
    """
    
    def __init__(self, **kwargs):
        """
        初始化ZXM风险控制指标

        Args:
            **kwargs: 指标参数
        """
        # 移除super().__init__调用，直接设置属性
        self.name = "ZXMRiskControl"
        self.description = "ZXM风险控制指标，分析股票的风险水平"

        # 设置默认参数
        self._default_parameters = self._get_default_parameters_zxmriskcontrol()

        # 应用用户参数
        self.set_parameters_Risk_Control(**kwargs)

        # 初始化数值稳定性管理器
        self.stability_mgr = get_stability_manager()
    
    def _get_default_parameters_zxmriskcontrol(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {
            "volatility_period": 20,
            "drawdown_period": 60,
            "var_confidence": 0.95,
            "risk_threshold": 70
        }
    
    def set_parameters_Risk_Control(self, **kwargs):
        """
        设置指标参数
        
        Args:
            **kwargs: 参数字典
        """
        self.volatility_period = kwargs.get('volatility_period', 20)
        self.drawdown_period = kwargs.get('drawdown_period', 60)
        self.var_confidence = kwargs.get('var_confidence', 0.95)
        self.risk_threshold = kwargs.get('risk_threshold', 70)

    @property
    def minimum_periods(self) -> int:
        """
        ZXM风险控制指标所需的最少数据周期数
        
        Returns:
            int: 最少需要的数据周期数
        """
        return max(self.volatility_period, self.drawdown_period) + 10

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算ZXM风险控制指标的主要入口方法
        
        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 其他参数
            
        Returns:
            包含ZXM风险控制指标的DataFrame
        """
        result = data.copy()
        
        # 计算波动率风险
        result = self._calculate_volatility_risk(result)
        
        # 计算回撤风险
        result = self._calculate_drawdown_risk(result)
        
        # 计算VaR风险
        result = self._calculate_var_risk(result)
        
        # 计算综合风险评分
        result = self._calculate_composite_risk_score(result)
        
        # 生成风险控制信号
        result = self._generate_risk_control_signals(result)
        
        return result

    def _calculate_volatility_risk(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算波动率风险"""
        result = data.copy()

        close = result['close']

        # 计算日收益率
        returns = close.pct_change()

        # 计算历史波动率
        volatility = returns.rolling(window=self.volatility_period).std() * np.sqrt(252)

        # 计算波动率风险评分（0-100）
        # 波动率越高，风险评分越高
        volatility_percentile = volatility.rolling(window=self.drawdown_period).rank(pct=True)
        volatility_risk = volatility_percentile * 100

        # 应用数值稳定性管理器确保精度控制
        volatility_risk = self.stability_mgr.ensure_series_precision(volatility_risk)

        result['Returns'] = self.stability_mgr.ensure_series_precision(returns)
        result['Volatility'] = self.stability_mgr.ensure_series_precision(volatility)
        result['VolatilityRisk'] = self.stability_mgr.ensure_series_precision(volatility_risk.fillna(50.0))

        return result

    def _calculate_drawdown_risk(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算回撤风险"""
        result = data.copy()

        close = result['close']

        # 计算累计最高价
        cumulative_max = close.expanding().max()

        # 计算回撤
        drawdown = (close - cumulative_max) / cumulative_max

        # 计算最大回撤
        max_drawdown = drawdown.rolling(window=self.drawdown_period).min()

        # 计算回撤风险评分（0-100）
        # 回撤越大，风险评分越高
        drawdown_risk = (-max_drawdown * 100).clip(0, 100)

        # 应用精度控制
        drawdown_risk = self.stability_mgr.ensure_series_precision(drawdown_risk)

        result['CumulativeMax'] = self.stability_mgr.ensure_series_precision(cumulative_max)
        result['Drawdown'] = self.stability_mgr.ensure_series_precision(drawdown)
        result['MaxDrawdown'] = self.stability_mgr.ensure_series_precision(max_drawdown)
        result['DrawdownRisk'] = self.stability_mgr.ensure_series_precision(drawdown_risk.fillna(0.0))

        return result

    def _calculate_var_risk(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算VaR风险"""
        result = data.copy()

        returns = result['Returns']

        # 最高性能的向量化VaR计算
        # 使用pandas的rolling.quantile替代循环，大幅提升性能
        quantile_level = (1 - self.var_confidence)
        var_series = returns.rolling(window=self.volatility_period).quantile(quantile_level)

        # 计算VaR风险评分（0-100）
        # VaR损失越大，风险评分越高
        var_risk = (-var_series * 100).clip(0, 100)

        # 应用精度控制
        var_series = self.stability_mgr.ensure_series_precision(var_series)
        var_risk = self.stability_mgr.ensure_series_precision(var_risk)

        result['VaR'] = var_series
        result['VaRRisk'] = self.stability_mgr.ensure_series_precision(var_risk.fillna(50.0))

        return result

    def _calculate_composite_risk_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算综合风险评分"""
        result = data.copy()

        # 综合各项风险指标
        volatility_risk = result['VolatilityRisk']
        drawdown_risk = result['DrawdownRisk']
        var_risk = result['VaRRisk']

        # 加权计算综合风险评分
        composite_risk = (
            volatility_risk * 0.4 +
            drawdown_risk * 0.35 +
            var_risk * 0.25
        )

        # 确保在0-100范围内
        composite_risk = np.clip(composite_risk, 0, 100)

        # 应用精度控制
        composite_risk = self.stability_mgr.ensure_series_precision(composite_risk)

        result['CompositeRiskScore'] = composite_risk

        return result

    def _generate_risk_control_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成风险控制信号"""
        result = data.copy()
        
        composite_risk = result['CompositeRiskScore']
        volatility_risk = result['VolatilityRisk']
        drawdown_risk = result['DrawdownRisk']
        
        # 高风险信号
        result['HighRiskSignal'] = (
            (composite_risk >= 80) |
            (volatility_risk >= 85) |
            (drawdown_risk >= 85)
        )
        
        # 中等风险信号
        result['ModerateRiskSignal'] = (
            (composite_risk >= 60) & (composite_risk < 80) &
            (volatility_risk < 85) &
            (drawdown_risk < 85)
        )
        
        # 低风险信号
        result['LowRiskSignal'] = composite_risk <= 30
        
        # 风险预警信号
        risk_change = composite_risk.diff()
        result['RiskWarningSignal'] = (
            (risk_change > 15) |  # 风险快速上升
            (composite_risk >= self.risk_threshold)  # 超过风险阈值
        )
        
        # 风险控制信号
        result['RiskControlSignal'] = (
            (composite_risk >= 75) |  # 高风险需要控制
            (risk_change > 20)        # 风险急剧上升
        )
        
        # 综合风险判断
        result['IsHighRisk'] = composite_risk >= 75
        result['IsLowRisk'] = composite_risk <= 25
        
        return result

    def _calculate_baseindicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        BaseIndicator要求的抽象方法实现
        
        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 其他参数
            
        Returns:
            包含ZXM风险控制指标的DataFrame
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
        return 0.9  # ZXM风险控制指标置信度
    
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
        # 风险评分需要反转，风险越低评分越高
        return 100 - result['CompositeRiskScore']
    
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
        
        # 风险水平形态
        composite_risk = result['CompositeRiskScore']
        patterns['ZXM_EXTREME_HIGH_RISK'] = composite_risk >= 90
        patterns['ZXM_HIGH_RISK'] = (composite_risk >= 75) & (composite_risk < 90)
        patterns['ZXM_MODERATE_RISK'] = (composite_risk >= 50) & (composite_risk < 75)
        patterns['ZXM_LOW_RISK'] = (composite_risk >= 25) & (composite_risk < 50)
        patterns['ZXM_VERY_LOW_RISK'] = composite_risk < 25
        
        # 风险控制信号形态
        patterns['ZXM_HIGH_RISK_SIGNAL'] = result['HighRiskSignal']
        patterns['ZXM_MODERATE_RISK_SIGNAL'] = result['ModerateRiskSignal']
        patterns['ZXM_LOW_RISK_SIGNAL'] = result['LowRiskSignal']
        patterns['ZXM_RISK_WARNING'] = result['RiskWarningSignal']
        patterns['ZXM_RISK_CONTROL'] = result['RiskControlSignal']
        
        return patterns
    
    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """
        BaseIndicator要求的参数设置方法
        
        Args:
            **kwargs: 参数字典
        """
        self.set_parameters_Risk_Control(**kwargs)
