from utils.container import container

"""
ZXM体系投资组合优化指标模块

实现ZXM体系的投资组合优化分析指标
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin
from utils.logger import get_logger

logger = get_logger(__name__)


class ZXMPortfolioOptimization(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
    """
    ZXM投资组合优化指标

    分析最优投资组合配置，提供动态优化策略
    """

    def __init__(self, **kwargs):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        """
        初始化ZXM投资组合优化指标

        Args:
            **kwargs: 指标参数
        """
        # 移除super().__init__调用，直接设置属性
        self.name = "ZXMPortfolioOptimization"
        self.description = "ZXM投资组合优化指标，分析最优投资组合配置"

        # 设置默认参数
        self._default_parameters = self._get_default_parameters_zxmportfoliooptimization()

        # 应用用户参数
        self.set_parameters_Portfolio_Optimization(**kwargs)

    def _get_default_parameters_zxmportfoliooptimization(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {
            "lookback_period": 60,  # TODO: 将魔法数字提取到配置中
            "rebalance_period": 20,  # TODO: 将魔法数字提取到配置中
            "risk_free_rate": 0.03,  # TODO: 将魔法数字提取到配置中
            "target_volatility": 0.15,  # TODO: 将魔法数字提取到配置中
            "max_weight": 0.4,  # TODO: 将魔法数字提取到配置中
            "min_weight": 0.05,  # TODO: 将魔法数字提取到配置中
        }

    def set_parameters_Portfolio_Optimization(self, **kwargs):
        """
        设置指标参数

        Args:
            **kwargs: 参数字典
        """
        self.lookback_period = kwargs.get("lookback_period", 60)  # TODO: 将魔法数字提取到配置中
        self.rebalance_period = kwargs.get("rebalance_period", 20)  # TODO: 将魔法数字提取到配置中
        self.risk_free_rate = kwargs.get("risk_free_rate", 0.03)  # TODO: 将魔法数字提取到配置中
        self.target_volatility = kwargs.get("target_volatility", 0.15)  # TODO: 将魔法数字提取到配置中
        self.max_weight = kwargs.get("max_weight", 0.4)  # TODO: 将魔法数字提取到配置中
        self.min_weight = kwargs.get("min_weight", 0.05)  # TODO: 将魔法数字提取到配置中

    @property
    def minimum_periods(self) -> int:
        """
        ZXM投资组合优化指标所需的最少数据周期数

        Returns:
            int: 最少需要的数据周期数
        """
        return max(self.lookback_period, self.rebalance_period) + 20  # TODO: 将魔法数字提取到配置中

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算ZXM投资组合优化指标的主要入口方法

        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 其他参数

        Returns:
            包含ZXM投资组合优化指标的DataFrame
        """
        result = data.copy()

        # 计算收益率和风险指标
        result = self._calculate_return_risk_metrics(result)

        # 计算最优权重
        result = self._calculate_optimal_weights(result)

        # 计算投资组合性能
        result = self._calculate_portfolio_performance(result)

        # 计算优化评分
        result = self._calculate_optimization_score(result)

        # 生成优化信号
        result = self._generate_optimization_signals(result)

        return result

    def _calculate_return_risk_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算收益率和风险指标"""
        result = data.copy()

        close = result["close"]

        # 日收益率
        daily_returns = close.pct_change()

        # 滚动收益率
        rolling_returns = (
            daily_returns.rolling(window=self.lookback_period).mean() * 252
        )  # TODO: 将魔法数字提取到配置中

        # 滚动波动率
        rolling_volatility = daily_returns.rolling(window=self.lookback_period).std() * np.sqrt(
            252
        )  # TODO: 将魔法数字提取到配置中

        # 滚动夏普比率
        excess_returns = rolling_returns - self.risk_free_rate
        rolling_sharpe = excess_returns / rolling_volatility

        # 最大回撤
        cumulative_returns = (1 + daily_returns).cumprod()
        rolling_max = cumulative_returns.rolling(window=self.lookback_period).max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.rolling(window=self.lookback_period).min()

        result["DailyReturns"] = daily_returns
        result["RollingReturns"] = rolling_returns
        result["RollingVolatility"] = rolling_volatility
        result["RollingSharpe"] = rolling_sharpe
        result["MaxDrawdown"] = max_drawdown

        return result

    def _calculate_optimal_weights(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算最优权重"""
        result = data.copy()

        rolling_returns = result["RollingReturns"]
        rolling_volatility = result["RollingVolatility"]
        rolling_sharpe = result["RollingSharpe"]

        # 基于风险平价的权重计算
        risk_parity_weight = 1 / rolling_volatility
        risk_parity_weight = risk_parity_weight / risk_parity_weight.rolling(window=self.rebalance_period).sum()

        # 基于夏普比率的权重调整
        sharpe_adjustment = np.maximum(rolling_sharpe, 0)  # 只考虑正夏普比率
        sharpe_weight = sharpe_adjustment / sharpe_adjustment.rolling(window=self.rebalance_period).sum()

        # 综合权重
        optimal_weight = (risk_parity_weight * 0.6 + sharpe_weight * 0.4).fillna(
            0.5
        )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        # 权重约束
        optimal_weight = np.clip(optimal_weight, self.min_weight, self.max_weight)

        # 权重标准化
        weight_sum = optimal_weight.rolling(window=self.rebalance_period).sum()
        normalized_weight = optimal_weight / weight_sum

        result["RiskParityWeight"] = risk_parity_weight
        result["SharpeWeight"] = sharpe_weight
        result["OptimalWeight"] = normalized_weight.fillna(0.5)  # TODO: 将魔法数字提取到配置中

        return result

    def _calculate_portfolio_performance(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算投资组合性能"""
        result = data.copy()

        daily_returns = result["DailyReturns"]
        optimal_weight = result["OptimalWeight"]

        # 投资组合收益率
        portfolio_returns = daily_returns * optimal_weight

        # 累积收益率
        cumulative_portfolio_returns = (1 + portfolio_returns).cumprod()

        # 投资组合波动率
        portfolio_volatility = portfolio_returns.rolling(window=self.lookback_period).std() * np.sqrt(
            252
        )  # TODO: 将魔法数字提取到配置中

        # 投资组合夏普比率
        portfolio_annual_returns = (
            portfolio_returns.rolling(window=self.lookback_period).mean() * 252
        )  # TODO: 将魔法数字提取到配置中
        portfolio_excess_returns = portfolio_annual_returns - self.risk_free_rate
        portfolio_sharpe = portfolio_excess_returns / portfolio_volatility

        # 信息比率
        benchmark_returns = daily_returns  # 假设基准为单一资产
        active_returns = portfolio_returns - benchmark_returns
        tracking_error = active_returns.rolling(window=self.lookback_period).std() * np.sqrt(
            252
        )  # TODO: 将魔法数字提取到配置中
        information_ratio = (
            active_returns.rolling(window=self.lookback_period).mean() * 252
        ) / tracking_error  # TODO: 将魔法数字提取到配置中

        result["PortfolioReturns"] = portfolio_returns
        result["CumulativePortfolioReturns"] = cumulative_portfolio_returns
        result["PortfolioVolatility"] = portfolio_volatility
        result["PortfolioSharpe"] = portfolio_sharpe
        result["InformationRatio"] = information_ratio

        return result

    def _calculate_optimization_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算优化评分"""
        result = data.copy()

        portfolio_sharpe = result["PortfolioSharpe"]
        information_ratio = result["InformationRatio"]
        portfolio_volatility = result["PortfolioVolatility"]
        max_drawdown = result["MaxDrawdown"]

        # 夏普比率评分 (0-40分)
        sharpe_score = np.clip(
            portfolio_sharpe * 20, 0, 40
        )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        # 信息比率评分 (0-30分)
        info_ratio_score = np.clip(
            information_ratio * 15, 0, 30
        )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        # 波动率控制评分 (0-20分)
        volatility_score = np.clip(
            (self.target_volatility - portfolio_volatility) / self.target_volatility * 20 + 10, 0, 20
        )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        # 回撤控制评分 (0-10分)
        drawdown_score = np.clip((-max_drawdown) * 50, 0, 10)  # TODO: 将魔法数字提取到配置中

        # 综合优化评分
        optimization_score = (
            sharpe_score.fillna(0)
            + info_ratio_score.fillna(0)
            + volatility_score.fillna(10)
            + drawdown_score.fillna(5)  # TODO: 将魔法数字提取到配置中
        )

        # 标准化到0-100范围
        optimization_score = np.clip(optimization_score, 0, 100)

        result["SharpeScore"] = sharpe_score
        result["InfoRatioScore"] = info_ratio_score
        result["VolatilityScore"] = volatility_score
        result["DrawdownScore"] = drawdown_score
        result["OptimizationScore"] = optimization_score.fillna(50)  # TODO: 将魔法数字提取到配置中

        return result

    def _generate_optimization_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成优化信号"""
        result = data.copy()

        optimal_weight = result["OptimalWeight"]
        optimization_score = result["OptimizationScore"]
        portfolio_sharpe = result["PortfolioSharpe"]
        portfolio_volatility = result["PortfolioVolatility"]

        # 再平衡信号
        weight_change = optimal_weight.diff()
        result["RebalanceSignal"] = abs(weight_change) > 0.05  # TODO: 将魔法数字提取到配置中

        # 优化信号
        result["OptimizationSignal"] = (optimization_score >= 80) | (  # 高优化评分  # TODO: 将魔法数字提取到配置中
            portfolio_sharpe >= 1.5
        )  # 高夏普比率  # TODO: 将魔法数字提取到配置中

        # 风险控制信号
        result["RiskControlSignal"] = (portfolio_volatility >= self.target_volatility * 1.2) | (  # 波动率过高
            optimization_score <= 30
        )  # 优化评分过低  # TODO: 将魔法数字提取到配置中

        # 优化改进信号
        score_improvement = optimization_score.diff()
        result["OptimizationImprovement"] = score_improvement > 10

        # 权重调整信号
        result["WeightAdjustSignal"] = (
            optimal_weight >= self.max_weight * 0.9
        ) | (  # 接近最大权重  # TODO: 将魔法数字提取到配置中
            optimal_weight <= self.min_weight * 1.1
        )  # 接近最小权重

        # 综合优化判断
        result["IsOptimalPortfolio"] = optimization_score >= 75  # TODO: 将魔法数字提取到配置中
        result["IsSuboptimalPortfolio"] = optimization_score <= 35  # TODO: 将魔法数字提取到配置中

        return result

    def _calculate_baseindicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        BaseIndicator要求的抽象方法实现

        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 其他参数

        Returns:
            包含ZXM投资组合优化指标的DataFrame
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
        return 0.90  # ZXM投资组合优化指标置信度  # TODO: 将魔法数字提取到配置中

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
        return result["OptimizationScore"]

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

        # 投资组合优化形态
        optimization_score = result["OptimizationScore"]
        patterns["ZXM_OPTIMAL_PORTFOLIO"] = optimization_score >= 85  # TODO: 将魔法数字提取到配置中
        patterns["ZXM_GOOD_PORTFOLIO"] = (optimization_score >= 65) & (
            optimization_score < 85
        )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        patterns["ZXM_NEUTRAL_PORTFOLIO"] = (optimization_score >= 45) & (
            optimization_score < 65
        )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        patterns["ZXM_POOR_PORTFOLIO"] = (optimization_score >= 25) & (
            optimization_score < 45
        )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        patterns["ZXM_SUBOPTIMAL_PORTFOLIO"] = optimization_score < 25  # TODO: 将魔法数字提取到配置中

        # 投资组合优化信号形态
        patterns["ZXM_REBALANCE"] = result["RebalanceSignal"]
        patterns["ZXM_OPTIMIZATION"] = result["OptimizationSignal"]
        patterns["ZXM_RISK_CONTROL"] = result["RiskControlSignal"]
        patterns["ZXM_OPTIMIZATION_IMPROVEMENT"] = result["OptimizationImprovement"]
        patterns["ZXM_WEIGHT_ADJUST"] = result["WeightAdjustSignal"]

        return patterns

    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """
        BaseIndicator要求的参数设置方法

        Args:
            **kwargs: 参数字典
        """
        self.set_parameters_Portfolio_Optimization(**kwargs)
