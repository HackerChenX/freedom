from utils.container import container

"""
ZXM体系策略组合指标模块

实现ZXM体系的策略组合分析指标
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin
from utils.logger import get_logger

logger = get_logger(__name__)


class ZXMStrategyCombination(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
    """
    ZXM策略组合指标

    分析多策略组合效果，提供动态策略权重优化
    """

    def __init__(self, **kwargs):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        """
        初始化ZXM策略组合指标

        Args:
            **kwargs: 指标参数
        """
        # 移除super().__init__调用，直接设置属性
        self.name = "ZXMStrategyCombination"
        self.description = "ZXM策略组合指标，分析多策略组合效果"

        # 设置默认参数
        self._default_parameters = self._get_default_parameters_zxmstrategycombination()

        # 应用用户参数
        self.set_parameters_Strategy_Combination(**kwargs)

    def _get_default_parameters_zxmstrategycombination(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {
            "strategy_count": 5,  # TODO: 将魔法数字提取到配置中
            "lookback_period": 30,  # TODO: 将魔法数字提取到配置中
            "rebalance_period": 10,
            "correlation_threshold": 0.7,  # TODO: 将魔法数字提取到配置中
            "performance_weight": 0.4,  # TODO: 将魔法数字提取到配置中
            "risk_weight": 0.3,  # TODO: 将魔法数字提取到配置中
            "correlation_weight": 0.3,  # TODO: 将魔法数字提取到配置中
        }

    def set_parameters_Strategy_Combination(self, **kwargs):
        """
        设置指标参数

        Args:
            **kwargs: 参数字典
        """
        self.strategy_count = kwargs.get("strategy_count", 5)  # TODO: 将魔法数字提取到配置中
        self.lookback_period = kwargs.get("lookback_period", 30)  # TODO: 将魔法数字提取到配置中
        self.rebalance_period = kwargs.get("rebalance_period", 10)
        self.correlation_threshold = kwargs.get("correlation_threshold", 0.7)  # TODO: 将魔法数字提取到配置中
        self.performance_weight = kwargs.get("performance_weight", 0.4)  # TODO: 将魔法数字提取到配置中
        self.risk_weight = kwargs.get("risk_weight", 0.3)  # TODO: 将魔法数字提取到配置中
        self.correlation_weight = kwargs.get("correlation_weight", 0.3)  # TODO: 将魔法数字提取到配置中

    @property
    def minimum_periods(self) -> int:
        """
        ZXM策略组合指标所需的最少数据周期数

        Returns:
            int: 最少需要的数据周期数
        """
        return max(self.lookback_period, self.rebalance_period) + 20  # TODO: 将魔法数字提取到配置中

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算ZXM策略组合指标的主要入口方法

        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 其他参数

        Returns:
            包含ZXM策略组合指标的DataFrame
        """
        result = data.copy()

        # 计算基础策略信号
        result = self._calculate_base_strategies(result)

        # 计算策略相关性
        result = self._calculate_strategy_correlation(result)

        # 计算策略权重
        result = self._calculate_strategy_weights(result)

        # 计算组合性能
        result = self._calculate_combination_performance(result)

        # 生成组合信号
        result = self._generate_combination_signals(result)

        return result

    def _calculate_base_strategies(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算基础策略信号"""
        result = data.copy()

        close = result["close"]
        high = result["high"]
        low = result["low"]
        volume = result["volume"]

        # 策略1: 趋势跟踪策略
        ma_short = close.rolling(window=10).mean()
        ma_long = close.rolling(window=30).mean()  # TODO: 将魔法数字提取到配置中
        trend_signal = (ma_short > ma_long).astype(float)

        # 策略2: 均值回归策略
        price_std = close.rolling(window=20).std()  # TODO: 将魔法数字提取到配置中
        price_mean = close.rolling(window=20).mean()  # TODO: 将魔法数字提取到配置中
        z_score = (close - price_mean) / price_std
        mean_reversion_signal = (z_score < -1.5).astype(float)  # TODO: 将魔法数字提取到配置中

        # 策略3: 动量策略
        momentum = close.pct_change(10)
        momentum_signal = (momentum > momentum.rolling(window=20).quantile(0.8)).astype(
            float
        )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        # 策略4: 波动率突破策略
        atr = (
            (high - low) + (high - close.shift(1)).abs() + (low - close.shift(1)).abs()
        ) / 3  # TODO: 将魔法数字提取到配置中
        atr_ma = atr.rolling(window=14).mean()  # TODO: 将魔法数字提取到配置中
        volatility_signal = (atr > atr_ma * 1.5).astype(float)  # TODO: 将魔法数字提取到配置中

        # 策略5: 成交量确认策略
        volume_ma = volume.rolling(window=20).mean()  # TODO: 将魔法数字提取到配置中
        volume_signal = (volume > volume_ma * 1.2).astype(float)

        result["Strategy1_Trend"] = trend_signal
        result["Strategy2_MeanReversion"] = mean_reversion_signal
        result["Strategy3_Momentum"] = momentum_signal
        result["Strategy4_Volatility"] = volatility_signal
        result["Strategy5_Volume"] = volume_signal

        return result

    def _calculate_strategy_correlation(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算策略相关性"""
        result = data.copy()

        strategy_columns = [
            "Strategy1_Trend",
            "Strategy2_MeanReversion",
            "Strategy3_Momentum",
            "Strategy4_Volatility",
            "Strategy5_Volume",
        ]

        # 计算滚动相关性
        correlation_matrix = pd.DataFrame(index=data.index)

        for i, strategy1 in enumerate(strategy_columns):
            for j, strategy2 in enumerate(strategy_columns):
                if i < j:  # 避免重复计算
                    corr_name = f"Corr_{strategy1}_{strategy2}"
                    correlation = result[strategy1].rolling(window=self.lookback_period).corr(result[strategy2])
                    correlation_matrix[corr_name] = correlation

        # 平均相关性
        avg_correlation = correlation_matrix.mean(axis=1)

        # 相关性评分（低相关性得高分）
        correlation_score = (1 - avg_correlation.abs()) * 100

        result["AvgCorrelation"] = avg_correlation.fillna(0)
        result["CorrelationScore"] = correlation_score.fillna(50)  # TODO: 将魔法数字提取到配置中

        return result

    def _calculate_strategy_weights(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算策略权重"""
        result = data.copy()

        close = result["close"]
        strategy_columns = [
            "Strategy1_Trend",
            "Strategy2_MeanReversion",
            "Strategy3_Momentum",
            "Strategy4_Volatility",
            "Strategy5_Volume",
        ]

        # 计算每个策略的历史表现
        strategy_returns = pd.DataFrame(index=data.index)

        for strategy in strategy_columns:
            # 模拟策略收益率
            strategy_signal = result[strategy]
            next_return = close.pct_change().shift(-1)  # 下一期收益率
            strategy_return = strategy_signal * next_return
            strategy_returns[f"{strategy}_Return"] = strategy_return

        # 计算滚动夏普比率
        strategy_sharpe = pd.DataFrame(index=data.index)
        for strategy in strategy_columns:
            return_col = f"{strategy}_Return"
            if return_col in strategy_returns.columns:
                rolling_mean = strategy_returns[return_col].rolling(window=self.lookback_period).mean()
                rolling_std = strategy_returns[return_col].rolling(window=self.lookback_period).std()
                sharpe = rolling_mean / rolling_std * np.sqrt(252)  # TODO: 将魔法数字提取到配置中
                strategy_sharpe[f"{strategy}_Sharpe"] = sharpe

        # 基于夏普比率和相关性计算权重
        correlation_score = result["CorrelationScore"]

        # 初始化权重
        weights = pd.DataFrame(index=data.index)

        for strategy in strategy_columns:
            sharpe_col = f"{strategy}_Sharpe"
            if sharpe_col in strategy_sharpe.columns:
                # 基础权重基于夏普比率
                base_weight = np.maximum(strategy_sharpe[sharpe_col], 0)

                # 相关性调整
                correlation_adjustment = correlation_score / 100

                # 最终权重
                final_weight = base_weight * correlation_adjustment
                weights[f"{strategy}_Weight"] = final_weight

        # 权重标准化
        weight_sum = weights.sum(axis=1)
        for col in weights.columns:
            weights[col] = weights[col] / weight_sum

        # 填充缺失值
        weights = weights.fillna(1.0 / len(strategy_columns))

        # 添加到结果中
        for col in weights.columns:
            result[col] = weights[col]

        # 计算综合策略权重
        result["StrategyWeight"] = weights.mean(axis=1)

        return result

    def _calculate_combination_performance(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算组合性能"""
        result = data.copy()

        close = result["close"]
        strategy_columns = [
            "Strategy1_Trend",
            "Strategy2_MeanReversion",
            "Strategy3_Momentum",
            "Strategy4_Volatility",
            "Strategy5_Volume",
        ]

        # 计算加权组合信号
        weighted_signal = pd.Series(0.0, index=data.index)

        for strategy in strategy_columns:
            weight_col = f"{strategy}_Weight"
            if weight_col in result.columns:
                weighted_signal += result[strategy] * result[weight_col]

        # 组合收益率
        next_return = close.pct_change().shift(-1)
        combination_return = weighted_signal * next_return

        # 滚动性能指标
        rolling_return = (
            combination_return.rolling(window=self.lookback_period).mean() * 252
        )  # TODO: 将魔法数字提取到配置中
        rolling_volatility = combination_return.rolling(window=self.lookback_period).std() * np.sqrt(
            252
        )  # TODO: 将魔法数字提取到配置中
        rolling_sharpe = rolling_return / rolling_volatility

        # 最大回撤
        cumulative_return = (1 + combination_return).cumprod()
        rolling_max = cumulative_return.rolling(window=self.lookback_period).max()
        drawdown = (cumulative_return - rolling_max) / rolling_max
        max_drawdown = drawdown.rolling(window=self.lookback_period).min()

        # 组合评分
        combination_score = (
            np.clip(
                rolling_sharpe * 25, 0, 50
            )  # 夏普比率评分  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            + np.clip((-max_drawdown) * 100, 0, 30)  # 回撤控制评分  # TODO: 将魔法数字提取到配置中
            + np.clip(result["CorrelationScore"] * 0.2, 0, 20)  # 分散化评分  # TODO: 将魔法数字提取到配置中
        )

        result["WeightedSignal"] = weighted_signal
        result["CombinationReturn"] = combination_return
        result["CombinationSharpe"] = rolling_sharpe
        result["CombinationDrawdown"] = max_drawdown
        result["CombinationScore"] = combination_score.fillna(50)  # TODO: 将魔法数字提取到配置中

        return result

    def _generate_combination_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成组合信号"""
        result = data.copy()

        weighted_signal = result["WeightedSignal"]
        combination_score = result["CombinationScore"]
        strategy_weight = result["StrategyWeight"]
        avg_correlation = result["AvgCorrelation"]

        # 策略组合信号
        result["StrategyCombinationSignal"] = weighted_signal > 0.6  # TODO: 将魔法数字提取到配置中

        # 策略优化信号
        result["StrategyOptimizationSignal"] = (
            combination_score >= 80
        ) | (  # 高组合评分  # TODO: 将魔法数字提取到配置中
            abs(avg_correlation) <= 0.3
        )  # 低相关性  # TODO: 将魔法数字提取到配置中

        # 权重调整信号
        weight_change = strategy_weight.diff()
        result["WeightAdjustSignal"] = abs(weight_change) > 0.1

        # 组合改进信号
        score_improvement = combination_score.diff()
        result["CombinationImprovement"] = score_improvement > 10

        # 分散化信号
        result["DiversificationSignal"] = abs(avg_correlation) <= self.correlation_threshold

        # 综合组合判断
        result["IsOptimalCombination"] = combination_score >= 75  # TODO: 将魔法数字提取到配置中
        result["IsSuboptimalCombination"] = combination_score <= 35  # TODO: 将魔法数字提取到配置中

        return result

    def _calculate_baseindicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        BaseIndicator要求的抽象方法实现

        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 其他参数

        Returns:
            包含ZXM策略组合指标的DataFrame
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
        return 0.85  # ZXM策略组合指标置信度  # TODO: 将魔法数字提取到配置中

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
        return result["CombinationScore"]

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

        # 策略组合形态
        combination_score = result["CombinationScore"]
        patterns["ZXM_OPTIMAL_COMBINATION"] = combination_score >= 85  # TODO: 将魔法数字提取到配置中
        patterns["ZXM_GOOD_COMBINATION"] = (combination_score >= 65) & (
            combination_score < 85
        )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        patterns["ZXM_NEUTRAL_COMBINATION"] = (combination_score >= 45) & (
            combination_score < 65
        )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        patterns["ZXM_POOR_COMBINATION"] = (combination_score >= 25) & (
            combination_score < 45
        )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        patterns["ZXM_SUBOPTIMAL_COMBINATION"] = combination_score < 25  # TODO: 将魔法数字提取到配置中

        # 策略组合信号形态
        patterns["ZXM_STRATEGY_COMBINATION"] = result["StrategyCombinationSignal"]
        patterns["ZXM_STRATEGY_OPTIMIZATION"] = result["StrategyOptimizationSignal"]
        patterns["ZXM_WEIGHT_ADJUST"] = result["WeightAdjustSignal"]
        patterns["ZXM_COMBINATION_IMPROVEMENT"] = result["CombinationImprovement"]
        patterns["ZXM_DIVERSIFICATION"] = result["DiversificationSignal"]

        return patterns

    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """
        BaseIndicator要求的参数设置方法

        Args:
            **kwargs: 参数字典
        """
        self.set_parameters_Strategy_Combination(**kwargs)
