"""
ZXM体系绩效归因指标模块

实现ZXM体系的绩效归因分析指标
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin
from utils.dependency_injection import get_logger

logger = get_logger(__name__)


class ZXMPerformanceAttribution(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
    """
    ZXM绩效归因指标
    
    分析投资组合绩效来源，提供Alpha和Beta归因分析
    """
    
    def __init__(self, **kwargs):
        """
        初始化ZXM绩效归因指标
        
        Args:
            **kwargs: 指标参数
        """
        # 移除super().__init__调用，直接设置属性
        self.name = "ZXMPerformanceAttribution"
        self.description = "ZXM绩效归因指标，分析投资组合绩效来源"
        
        # 设置默认参数
        self._default_parameters = self._get_default_parameters_zxmperformanceattribution()
        
        # 应用用户参数
        self.set_parameters_Performance_Attribution(**kwargs)
    
    def _get_default_parameters_zxmperformanceattribution(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {
            "lookback_period": 60,
            "benchmark_period": 252,
            "risk_free_rate": 0.03,
            "attribution_period": 30,
            "factor_count": 3,
            "alpha_threshold": 0.02
        }
    
    def set_parameters_Performance_Attribution(self, **kwargs):
        """
        设置指标参数
        
        Args:
            **kwargs: 参数字典
        """
        self.lookback_period = kwargs.get('lookback_period', 60)
        self.benchmark_period = kwargs.get('benchmark_period', 252)
        self.risk_free_rate = kwargs.get('risk_free_rate', 0.03)
        self.attribution_period = kwargs.get('attribution_period', 30)
        self.factor_count = kwargs.get('factor_count', 3)
        self.alpha_threshold = kwargs.get('alpha_threshold', 0.02)

    @property
    def minimum_periods(self) -> int:
        """
        ZXM绩效归因指标所需的最少数据周期数
        
        Returns:
            int: 最少需要的数据周期数
        """
        return max(self.lookback_period, self.benchmark_period, self.attribution_period) + 20

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算ZXM绩效归因指标的主要入口方法
        
        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 其他参数
            
        Returns:
            包含ZXM绩效归因指标的DataFrame
        """
        result = data.copy()
        
        # 计算收益率和基准
        result = self._calculate_returns_and_benchmark(result)
        
        # 计算Alpha和Beta
        result = self._calculate_alpha_beta(result)
        
        # 计算因子贡献
        result = self._calculate_factor_contributions(result)
        
        # 计算归因评分
        result = self._calculate_attribution_score(result)
        
        # 生成归因信号
        result = self._generate_attribution_signals(result)
        
        return result

    def _calculate_returns_and_benchmark(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算收益率和基准"""
        result = data.copy()
        
        close = result['close']
        volume = result['volume']
        
        # 日收益率
        daily_returns = close.pct_change()
        
        # 基准收益率（使用市场平均收益率模拟）
        market_returns = daily_returns.rolling(window=self.lookback_period).mean()
        
        # 超额收益率
        excess_returns = daily_returns - market_returns
        
        # 累积收益率
        cumulative_returns = (1 + daily_returns).cumprod()
        cumulative_benchmark = (1 + market_returns).cumprod()
        
        # 相对收益率
        relative_returns = cumulative_returns / cumulative_benchmark - 1
        
        result['DailyReturns'] = daily_returns
        result['MarketReturns'] = market_returns
        result['ExcessReturns'] = excess_returns
        result['CumulativeReturns'] = cumulative_returns
        result['CumulativeBenchmark'] = cumulative_benchmark
        result['RelativeReturns'] = relative_returns
        
        return result

    def _calculate_alpha_beta(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算Alpha和Beta"""
        result = data.copy()
        
        daily_returns = result['DailyReturns']
        market_returns = result['MarketReturns']
        excess_returns = result['ExcessReturns']
        
        # 滚动Beta计算
        rolling_beta = pd.Series(index=data.index, dtype=float)
        rolling_alpha = pd.Series(index=data.index, dtype=float)
        
        for i in range(self.lookback_period, len(data)):
            # 获取滚动窗口数据
            window_returns = daily_returns.iloc[i-self.lookback_period:i].dropna()
            window_market = market_returns.iloc[i-self.lookback_period:i].dropna()

            # 确保两个序列长度一致且不为空
            if len(window_returns) > 5 and len(window_market) > 5:
                # 取较短的长度
                min_length = min(len(window_returns), len(window_market))
                window_returns = window_returns.iloc[-min_length:]
                window_market = window_market.iloc[-min_length:]

                # 计算协方差和方差
                try:
                    covariance = np.cov(window_returns, window_market)[0, 1]
                    market_variance = np.var(window_market)

                    # Beta计算
                    if market_variance > 0:
                        beta = covariance / market_variance
                    else:
                        beta = 1.0

                    # Alpha计算（CAPM模型）
                    portfolio_return = window_returns.mean() * 252
                    market_return = window_market.mean() * 252
                    alpha = portfolio_return - (self.risk_free_rate + beta * (market_return - self.risk_free_rate))

                    rolling_beta.iloc[i] = beta
                    rolling_alpha.iloc[i] = alpha
                except:
                    # 如果计算失败，使用默认值
                    rolling_beta.iloc[i] = 1.0
                    rolling_alpha.iloc[i] = 0.0
            else:
                # 数据不足，使用默认值
                rolling_beta.iloc[i] = 1.0
                rolling_alpha.iloc[i] = 0.0
        
        # Alpha和Beta贡献
        alpha_contribution = rolling_alpha / 252  # 日化Alpha
        beta_contribution = rolling_beta * market_returns
        
        result['RollingBeta'] = rolling_beta.fillna(1.0)
        result['RollingAlpha'] = rolling_alpha.fillna(0.0)
        result['AlphaContribution'] = alpha_contribution.fillna(0.0)
        result['BetaContribution'] = beta_contribution.fillna(0.0)
        
        return result

    def _calculate_factor_contributions(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算因子贡献"""
        result = data.copy()
        
        daily_returns = result['DailyReturns']
        volume = result['volume']
        close = result['close']
        
        # 因子1: 规模因子（基于成交量）
        volume_factor = (volume / volume.rolling(window=self.lookback_period).mean() - 1)
        size_contribution = volume_factor * daily_returns * 0.3
        
        # 因子2: 动量因子（基于价格动量）
        momentum_factor = close.pct_change(self.attribution_period)
        momentum_contribution = momentum_factor * daily_returns * 0.4
        
        # 因子3: 波动率因子（基于收益率波动）
        volatility_factor = daily_returns.rolling(window=self.lookback_period).std()
        volatility_contribution = volatility_factor * daily_returns * 0.3
        
        # 总因子贡献
        total_factor_contribution = (
            size_contribution.fillna(0) +
            momentum_contribution.fillna(0) +
            volatility_contribution.fillna(0)
        )
        
        result['SizeContribution'] = size_contribution.fillna(0)
        result['MomentumContribution'] = momentum_contribution.fillna(0)
        result['VolatilityContribution'] = volatility_contribution.fillna(0)
        result['FactorContribution'] = total_factor_contribution
        
        return result

    def _calculate_attribution_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算归因评分"""
        result = data.copy()
        
        alpha_contribution = result['AlphaContribution']
        beta_contribution = result['BetaContribution']
        factor_contribution = result['FactorContribution']
        rolling_alpha = result['RollingAlpha']
        rolling_beta = result['RollingBeta']
        
        # Alpha评分 (0-40分)
        alpha_score = np.clip(rolling_alpha * 1000 + 20, 0, 40)
        
        # Beta评分 (0-30分) - Beta接近1得高分
        beta_score = np.clip(30 - abs(rolling_beta - 1) * 30, 0, 30)
        
        # 因子贡献评分 (0-20分)
        factor_score = np.clip(abs(factor_contribution) * 1000 + 10, 0, 20)
        
        # 稳定性评分 (0-10分)
        alpha_stability = 1 / (1 + alpha_contribution.rolling(window=self.attribution_period).std().fillna(1))
        stability_score = alpha_stability * 10
        
        # 综合归因评分
        attribution_score = (
            alpha_score.fillna(20) +
            beta_score.fillna(15) +
            factor_score.fillna(10) +
            stability_score.fillna(5)
        )
        
        # 标准化到0-100范围
        attribution_score = np.clip(attribution_score, 0, 100)
        
        result['AlphaScore'] = alpha_score.fillna(20)
        result['BetaScore'] = beta_score.fillna(15)
        result['FactorScore'] = factor_score.fillna(10)
        result['StabilityScore'] = stability_score.fillna(5)
        result['AttributionScore'] = attribution_score.fillna(50)
        
        return result

    def _generate_attribution_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成归因信号"""
        result = data.copy()
        
        alpha_contribution = result['AlphaContribution']
        attribution_score = result['AttributionScore']
        rolling_alpha = result['RollingAlpha']
        rolling_beta = result['RollingBeta']
        factor_contribution = result['FactorContribution']
        
        # 归因信号
        result['AttributionSignal'] = attribution_score >= 75
        
        # Alpha信号
        result['AlphaSignal'] = (
            (rolling_alpha >= self.alpha_threshold) &  # 显著正Alpha
            (abs(alpha_contribution) >= 0.001)  # Alpha贡献显著
        )
        
        # Beta信号
        result['BetaSignal'] = abs(rolling_beta - 1) <= 0.2  # Beta接近1
        
        # 因子贡献信号
        factor_change = factor_contribution.diff()
        result['FactorSignal'] = abs(factor_change) >= 0.01
        
        # 归因改进信号
        score_improvement = attribution_score.diff()
        result['AttributionImprovement'] = score_improvement > 15
        
        # 综合归因判断
        result['IsGoodAttribution'] = attribution_score >= 70
        result['IsPoorAttribution'] = attribution_score <= 30
        
        return result

    def _calculate_baseindicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        BaseIndicator要求的抽象方法实现
        
        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 其他参数
            
        Returns:
            包含ZXM绩效归因指标的DataFrame
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
        return 0.82  # ZXM绩效归因指标置信度
    
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
        return result['AttributionScore']
    
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
        
        # 绩效归因形态
        attribution_score = result['AttributionScore']
        patterns['ZXM_EXCELLENT_ATTRIBUTION'] = attribution_score >= 85
        patterns['ZXM_GOOD_ATTRIBUTION'] = (attribution_score >= 65) & (attribution_score < 85)
        patterns['ZXM_NEUTRAL_ATTRIBUTION'] = (attribution_score >= 45) & (attribution_score < 65)
        patterns['ZXM_POOR_ATTRIBUTION'] = (attribution_score >= 25) & (attribution_score < 45)
        patterns['ZXM_BAD_ATTRIBUTION'] = attribution_score < 25
        
        # 绩效归因信号形态
        patterns['ZXM_ATTRIBUTION_SIGNAL'] = result['AttributionSignal']
        patterns['ZXM_ALPHA_SIGNAL'] = result['AlphaSignal']
        patterns['ZXM_BETA_SIGNAL'] = result['BetaSignal']
        patterns['ZXM_FACTOR_SIGNAL'] = result['FactorSignal']
        patterns['ZXM_ATTRIBUTION_IMPROVEMENT'] = result['AttributionImprovement']
        
        return patterns
    
    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """
        BaseIndicator要求的参数设置方法
        
        Args:
            **kwargs: 参数字典
        """
        self.set_parameters_Performance_Attribution(**kwargs)
