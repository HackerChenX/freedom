from utils.container import container

"""
ZXM体系Beta对冲指标模块

实现ZXM体系的Beta对冲分析指标
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin
from utils.logger import get_logger

logger = get_logger(__name__)


class ZXMBetaHedging(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
    """
    ZXM Beta对冲指标

    分析投资组合Beta风险暴露，提供动态对冲策略
    """

    def __init__(self, **kwargs):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        """
        初始化ZXM Beta对冲指标

        Args:
            **kwargs: 指标参数
        """
        # 移除super().__init__调用，直接设置属性
        self.name = "ZXMBetaHedging"
        self.description = "ZXM Beta对冲指标，分析投资组合Beta风险暴露"

        # 设置默认参数
        self._default_parameters = self._get_default_parameters_zxmbetahedging()

        # 应用用户参数
        self.set_parameters_Beta_Hedging(**kwargs)

    def _get_default_parameters_zxmbetahedging(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {
            "lookback_period": 60,  # TODO: 将魔法数字提取到配置中
            "hedge_period": 30,  # TODO: 将魔法数字提取到配置中
            "beta_threshold": 1.2,
            "hedge_threshold": 0.8,  # TODO: 将魔法数字提取到配置中
            "rebalance_frequency": 5,  # TODO: 将魔法数字提取到配置中
            "target_beta": 1.0,
        }

    def set_parameters_Beta_Hedging(self, **kwargs):
        """
        设置指标参数

        Args:
            **kwargs: 参数字典
        """
        self.lookback_period = kwargs.get("lookback_period", 60)  # TODO: 将魔法数字提取到配置中
        self.hedge_period = kwargs.get("hedge_period", 30)  # TODO: 将魔法数字提取到配置中
        self.beta_threshold = kwargs.get("beta_threshold", 1.2)
        self.hedge_threshold = kwargs.get("hedge_threshold", 0.8)  # TODO: 将魔法数字提取到配置中
        self.rebalance_frequency = kwargs.get("rebalance_frequency", 5)  # TODO: 将魔法数字提取到配置中
        self.target_beta = kwargs.get("target_beta", 1.0)

    @property
    def minimum_periods(self) -> int:
        """
        ZXM Beta对冲指标所需的最少数据周期数

        Returns:
            int: 最少需要的数据周期数
        """
        return max(self.lookback_period, self.hedge_period) + 20  # TODO: 将魔法数字提取到配置中

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算ZXM Beta对冲指标的主要入口方法

        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 其他参数

        Returns:
            包含ZXM Beta对冲指标的DataFrame
        """
        result = data.copy()

        # 计算收益率和基准
        result = self._calculate_returns_and_benchmark(result)

        # 计算Beta值
        result = self._calculate_beta_values(result)

        # 计算对冲比率
        result = self._calculate_hedge_ratios(result)

        # 计算对冲效果
        result = self._calculate_hedging_effectiveness(result)

        # 生成对冲信号
        result = self._generate_hedging_signals(result)

        return result

    def _calculate_returns_and_benchmark(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算收益率和基准"""
        result = data.copy()

        close = result["close"]
        volume = result["volume"]

        # 日收益率
        daily_returns = close.pct_change()

        # 基准收益率（使用市场平均收益率模拟）
        market_returns = daily_returns.rolling(window=self.lookback_period).mean()

        # 超额收益率
        excess_returns = daily_returns - market_returns

        result["DailyReturns"] = daily_returns
        result["MarketReturns"] = market_returns
        result["ExcessReturns"] = excess_returns

        return result

    def _calculate_beta_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算Beta值"""
        result = data.copy()

        daily_returns = result["DailyReturns"]
        market_returns = result["MarketReturns"]

        # 滚动Beta计算
        rolling_beta = pd.Series(index=data.index, dtype=float)

        for i in range(self.lookback_period, len(data)):
            # 获取滚动窗口数据
            window_returns = daily_returns.iloc[i - self.lookback_period : i].dropna()
            window_market = market_returns.iloc[i - self.lookback_period : i].dropna()

            # 确保两个序列长度一致且不为空
            if len(window_returns) > 10 and len(window_market) > 10:
                # 取较短的长度
                min_length = min(len(window_returns), len(window_market))
                window_returns = window_returns.iloc[-min_length:]
                window_market = window_market.iloc[-min_length:]

                try:
                    # 计算协方差和方差
                    covariance = np.cov(window_returns, window_market)[0, 1]
                    market_variance = np.var(window_market)

                    # Beta计算
                    if market_variance > 0:
                        beta = covariance / market_variance
                        # 限制Beta值在合理范围内
                        beta = np.clip(beta, -1.0, 3.0)  # TODO: 将魔法数字提取到配置中
                    else:
                        beta = 1.0

                    rolling_beta.iloc[i] = beta
                except:
                    # 如果计算失败，使用默认值
                    rolling_beta.iloc[i] = 1.0
            else:
                # 数据不足，使用默认值
                rolling_beta.iloc[i] = 1.0

        # Beta稳定性
        beta_volatility = rolling_beta.rolling(window=self.hedge_period).std()

        # Beta趋势
        beta_trend = rolling_beta.diff(self.hedge_period)

        result["BetaValue"] = rolling_beta.fillna(1.0)
        result["BetaVolatility"] = beta_volatility.fillna(0.1)
        result["BetaTrend"] = beta_trend.fillna(0.0)

        return result

    def _calculate_hedge_ratios(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算对冲比率"""
        result = data.copy()

        beta_value = result["BetaValue"]
        beta_volatility = result["BetaVolatility"]

        # 基础对冲比率
        base_hedge_ratio = np.maximum(0, (beta_value - self.target_beta) / beta_value)

        # 波动率调整
        volatility_adjustment = np.minimum(1.0, beta_volatility * 2)

        # 最终对冲比率
        hedge_ratio = base_hedge_ratio * (1 + volatility_adjustment)
        hedge_ratio = np.clip(hedge_ratio, 0, 1)

        # 对冲成本估算
        hedge_cost = hedge_ratio * 0.01  # 假设对冲成本为1%

        # 净对冲比率（考虑成本）
        net_hedge_ratio = np.maximum(0, hedge_ratio - hedge_cost)

        result["BaseHedgeRatio"] = base_hedge_ratio.fillna(0)
        result["VolatilityAdjustment"] = volatility_adjustment.fillna(0.1)
        result["HedgeRatio"] = hedge_ratio.fillna(0)
        result["HedgeCost"] = hedge_cost.fillna(0.01)
        result["NetHedgeRatio"] = net_hedge_ratio.fillna(0)

        return result

    def _calculate_hedging_effectiveness(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算对冲效果"""
        result = data.copy()

        beta_value = result["BetaValue"]
        hedge_ratio = result["HedgeRatio"]
        daily_returns = result["DailyReturns"]
        market_returns = result["MarketReturns"]

        # 对冲后的Beta
        hedged_beta = beta_value * (1 - hedge_ratio)

        # 对冲后的收益率
        hedged_returns = daily_returns - hedge_ratio * market_returns

        # 对冲效果评估
        original_volatility = daily_returns.rolling(window=self.hedge_period).std()
        hedged_volatility = hedged_returns.rolling(window=self.hedge_period).std()

        # 波动率降低比例
        volatility_reduction = (original_volatility - hedged_volatility) / original_volatility
        volatility_reduction = np.clip(volatility_reduction, 0, 1)

        # 对冲效率
        hedge_efficiency = volatility_reduction / (hedge_ratio + 0.01)  # 避免除零

        # 对冲评分
        hedging_score = (
            (1 - abs(hedged_beta - self.target_beta)) * 40  # Beta目标达成度  # TODO: 将魔法数字提取到配置中
            + volatility_reduction * 100 * 35  # 波动率降低效果  # TODO: 将魔法数字提取到配置中
            + hedge_efficiency * 25  # 对冲效率  # TODO: 将魔法数字提取到配置中
        )
        hedging_score = np.clip(hedging_score, 0, 100)

        result["HedgedBeta"] = hedged_beta.fillna(1.0)
        result["HedgedReturns"] = hedged_returns.fillna(0)
        result["OriginalVolatility"] = original_volatility.fillna(0.02)
        result["HedgedVolatility"] = hedged_volatility.fillna(0.02)
        result["VolatilityReduction"] = volatility_reduction.fillna(0)
        result["HedgeEfficiency"] = hedge_efficiency.fillna(0)
        result["HedgingScore"] = hedging_score.fillna(50)  # TODO: 将魔法数字提取到配置中

        return result

    def _generate_hedging_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成对冲信号"""
        result = data.copy()

        beta_value = result["BetaValue"]
        hedge_ratio = result["HedgeRatio"]
        hedging_score = result["HedgingScore"]
        beta_trend = result["BetaTrend"]

        # 对冲信号
        result["HedgingSignal"] = (beta_value >= self.beta_threshold) | (  # Beta过高
            hedging_score <= 40
        )  # 对冲效果差  # TODO: 将魔法数字提取到配置中

        # Beta调整信号
        result["BetaAdjustSignal"] = abs(beta_value - self.target_beta) >= 0.2

        # 对冲比率调整信号
        hedge_ratio_change = hedge_ratio.diff()
        result["HedgeRatioAdjustSignal"] = abs(hedge_ratio_change) >= 0.1

        # 对冲改进信号
        score_improvement = hedging_score.diff()
        result["HedgingImprovement"] = score_improvement > 10

        # 再平衡信号
        result["RebalanceSignal"] = (result.index % self.rebalance_frequency == 0) & (  # 定期再平衡
            abs(beta_trend) >= 0.1
        )  # Beta趋势变化

        # 综合对冲判断
        result["IsWellHedged"] = hedging_score >= 70  # TODO: 将魔法数字提取到配置中
        result["IsPoorlyHedged"] = hedging_score <= 30  # TODO: 将魔法数字提取到配置中

        return result

    def _calculate_baseindicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        BaseIndicator要求的抽象方法实现

        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 其他参数

        Returns:
            包含ZXM Beta对冲指标的DataFrame
        """
        return self.calculate(data, **kwargs)

    def calculate_confidence_Indicator_Base_Indicator(
        self, score: pd.Series, patterns: pd.DataFrame, signals: dict
    ) -> float:
        """
        BaseIndicator要求的置信度计算方法

        Args:
            score: 得分序列
            patterns: 检测到的形态DataFrame
            signals: 生成的信号字典

        Returns:
            float: 置信度分数 (0-1)
        """
        return 0.83  # ZXM Beta对冲指标置信度  # TODO: 将魔法数字提取到配置中

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
        return result["HedgingScore"]

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

        # Beta对冲形态
        hedging_score = result["HedgingScore"]
        patterns["ZXM_EXCELLENT_HEDGING"] = hedging_score >= 85  # TODO: 将魔法数字提取到配置中
        patterns["ZXM_GOOD_HEDGING"] = (hedging_score >= 65) & (
            hedging_score < 85
        )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        patterns["ZXM_NEUTRAL_HEDGING"] = (hedging_score >= 45) & (
            hedging_score < 65
        )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        patterns["ZXM_POOR_HEDGING"] = (hedging_score >= 25) & (
            hedging_score < 45
        )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        patterns["ZXM_BAD_HEDGING"] = hedging_score < 25  # TODO: 将魔法数字提取到配置中

        # Beta对冲信号形态
        patterns["ZXM_HEDGING_SIGNAL"] = result["HedgingSignal"]
        patterns["ZXM_BETA_ADJUST"] = result["BetaAdjustSignal"]
        patterns["ZXM_HEDGE_RATIO_ADJUST"] = result["HedgeRatioAdjustSignal"]
        patterns["ZXM_HEDGING_IMPROVEMENT"] = result["HedgingImprovement"]
        patterns["ZXM_REBALANCE"] = result["RebalanceSignal"]

        return patterns

    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """
        BaseIndicator要求的参数设置方法

        Args:
            **kwargs: 参数字典
        """
        self.set_parameters_Beta_Hedging(**kwargs)
