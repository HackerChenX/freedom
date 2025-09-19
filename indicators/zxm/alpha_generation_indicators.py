from utils.container import container

"""
ZXM体系Alpha生成指标模块

实现ZXM体系的Alpha生成分析指标
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin
from utils.logger import get_logger

logger = get_logger(__name__)


class ZXMAlphaGeneration(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
    """
    ZXM Alpha生成指标

    分析投资组合Alpha生成能力，提供超额收益分析
    """

    def __init__(self, **kwargs):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        """
        初始化ZXM Alpha生成指标

        Args:
            **kwargs: 指标参数
        """
        # 正确调用父类初始化
        super().__init__(name="ZXMAlphaGeneration", description="ZXM Alpha生成指标，分析投资组合Alpha生成能力")

        # 设置默认参数
        self._default_parameters = self._get_default_parameters_zxmalphagenerati()

        # 应用用户参数
        self.set_parameters_Alpha_Generation(**kwargs)

    def _get_default_parameters_zxmalphagenerati(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {
            "lookback_period": 60,  # TODO: 将魔法数字提取到配置中
            "benchmark_period": 252,  # TODO: 将魔法数字提取到配置中
            "risk_free_rate": 0.03,  # TODO: 将魔法数字提取到配置中
            "alpha_threshold": 0.02,
            "factor_count": 5,  # TODO: 将魔法数字提取到配置中
            "rolling_window": 30,  # TODO: 将魔法数字提取到配置中
        }

    def set_parameters_Alpha_Generation(self, **kwargs):
        """
        设置指标参数

        Args:
            **kwargs: 参数字典
        """
        self.lookback_period = kwargs.get("lookback_period", 60)  # TODO: 将魔法数字提取到配置中
        self.benchmark_period = kwargs.get("benchmark_period", 252)  # TODO: 将魔法数字提取到配置中
        self.risk_free_rate = kwargs.get("risk_free_rate", 0.03)  # TODO: 将魔法数字提取到配置中
        self.alpha_threshold = kwargs.get("alpha_threshold", 0.02)
        self.factor_count = kwargs.get("factor_count", 5)  # TODO: 将魔法数字提取到配置中
        self.rolling_window = kwargs.get("rolling_window", 30)  # TODO: 将魔法数字提取到配置中

    @property
    def minimum_periods(self) -> int:
        """
        ZXM Alpha生成指标所需的最少数据周期数

        Returns:
            int: 最少需要的数据周期数
        """
        return (
            max(self.lookback_period, self.benchmark_period, self.rolling_window) + 20
        )  # TODO: 将魔法数字提取到配置中

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算ZXM Alpha生成指标的主要入口方法

        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 其他参数

        Returns:
            包含ZXM Alpha生成指标的DataFrame
        """
        result = data.copy()

        # 计算收益率和基准
        result = self._calculate_returns_and_benchmark(result)

        # 计算Alpha值
        result = self._calculate_alpha_values(result)

        # 计算因子贡献
        result = self._calculate_factor_contributions(result)

        # 计算Alpha评分
        result = self._calculate_alpha_score(result)

        # 生成Alpha信号
        result = self._generate_alpha_signals(result)
        
        # 保存结果
        self._result = result

        return result

    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        获取ZXM Alpha生成交易信号（抽象方法实现）
        
        基于Alpha生成能力判断投资价值
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            Dict[str, Any]: 标准化的交易信号字典
        """
        # 1. 数据验证
        if not isinstance(data, pd.DataFrame):
            return self._get_default_signal("输入数据必须是DataFrame")
        
        if data.empty:
            return self._get_default_signal("输入数据为空")
        
        if 'close' not in data.columns:
            return self._get_default_signal("缺少必需的'close'列")
        
        if len(data) < self.minimum_periods:
            return self._get_default_signal("数据量不足")
        
        try:
            # 2. 计算ZXM Alpha生成指标
            result = self.calculate(data)
            
            if result.empty or len(result) == 0:
                return self._get_default_signal("指标计算结果为空")
                
            # 3. 获取最新数据
            latest_data = result.iloc[-1]
            
            # 4. 获取关键指标值
            alpha_score = latest_data.get('AlphaScore', 50)
            rolling_alpha = latest_data.get('RollingAlpha', 0)
            information_ratio = latest_data.get('InformationRatio', 0)
            alpha_signal = latest_data.get('AlphaSignal', False)
            positive_alpha = latest_data.get('PositiveAlphaSignal', False)
            is_good_alpha = latest_data.get('IsGoodAlpha', False)
            is_poor_alpha = latest_data.get('IsPoorAlpha', False)
            close_price = latest_data.get('close', 0)
            factor_contribution = latest_data.get('FactorContribution', 0)
            
            # 5. 信号生成逻辑
            signal_type = "hold"
            strength = 0.0
            confidence = 0.5
            reason = "ZXM Alpha生成无明显信号"
            
            # Alpha评分优秀：强力买入信号
            if alpha_score >= 85 and positive_alpha:
                signal_type = "buy"
                strength = 0.95
                confidence = 0.95
                reason = "Alpha评分优秀且正Alpha，强力买入信号"
            
            # Alpha评分良好：一般买入信号
            elif alpha_score >= 70 and is_good_alpha:
                signal_type = "buy"
                strength = 0.8
                confidence = 0.85
                reason = "Alpha评分良好，买入信号"
            
            # Alpha评分中等但有正向信号：谨慎买入
            elif alpha_score >= 60 and (alpha_signal or positive_alpha):
                signal_type = "buy"
                strength = 0.6
                confidence = 0.7
                reason = "Alpha评分中等但有正向信号，谨慎买入"
            
            # Alpha评分差：卖出信号
            elif alpha_score <= 30 or is_poor_alpha:
                signal_type = "sell"
                strength = 0.7
                confidence = 0.8
                reason = "Alpha评分差，卖出信号"
            
            # Alpha评分极差：强力卖出
            elif alpha_score <= 15:
                signal_type = "sell"
                strength = 0.9
                confidence = 0.9
                reason = "Alpha评分极差，强力卖出"
            
            else:
                # 其他情况：观望
                signal_type = "hold"
                strength = 0.0
                confidence = 0.5
                reason = "Alpha表现平平，观望"
            
            # 6. 基于信息比率的信号调整
            if information_ratio > 0.8:
                if signal_type == "buy":
                    strength = min(1.0, strength + 0.1)
                    confidence = min(1.0, confidence + 0.05)
                    reason += "，信息比率优异"
            elif information_ratio < -0.5:
                if signal_type == "buy":
                    strength = max(0.0, strength - 0.2)
                    confidence = max(0.3, confidence - 0.1)
                    reason += "，信息比率较差"
                elif signal_type == "hold":
                    signal_type = "sell"
                    strength = 0.6
                    confidence = 0.7
                    reason = "信息比率差，转为卖出信号"
            
            # 7. 基于因子贡献的信号调整
            if abs(factor_contribution) > 0.01:
                if factor_contribution > 0 and signal_type == "buy":
                    strength = min(1.0, strength + 0.05)
                    reason += "，因子贡献正向"
                elif factor_contribution < 0 and signal_type == "sell":
                    strength = min(1.0, strength + 0.05)
                    reason += "，因子贡献负向"
            
            # 8. 生成标准化信号
            signal = {
                'signal_type': signal_type,
                'strength': round(strength, 3),
                'confidence': round(confidence, 3),
                'timestamp': data.index[-1] if len(data) > 0 else None,
                'price': round(close_price, 3) if close_price > 0 else None,
                'reason': reason,
                'metadata': {
                    'indicator_type': 'zxm_alpha_generation',
                    'alpha_score': round(alpha_score, 3),
                    'rolling_alpha': round(rolling_alpha, 6),
                    'information_ratio': round(information_ratio, 3),
                    'alpha_signal': bool(alpha_signal),
                    'positive_alpha': bool(positive_alpha),
                    'is_good_alpha': bool(is_good_alpha),
                    'is_poor_alpha': bool(is_poor_alpha),
                    'factor_contribution': round(factor_contribution, 6),
                    'alpha_threshold': self.alpha_threshold,
                    'lookback_period': self.lookback_period,
                    'benchmark_period': self.benchmark_period,
                    'risk_free_rate': self.risk_free_rate
                }
            }
            
            return signal
            
        except Exception as e:
            logger.error(f"ZXM Alpha生成信号生成失败: {e}")
            return self._get_default_signal(f"计算错误: {str(e)}")

    def _get_default_signal(self, reason: str = "数据验证失败") -> Dict[str, Any]:
        """
        生成默认信号
        
        Args:
            reason: 失败原因
            
        Returns:
            Dict[str, Any]: 默认信号字典
        """
        return {
            'signal_type': 'hold',
            'strength': 0.0,
            'confidence': 0.5,
            'timestamp': None,
            'price': None,
            'reason': reason,
            'metadata': {}
        }

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

        # 累积收益率
        cumulative_returns = (1 + daily_returns).cumprod()
        cumulative_benchmark = (1 + market_returns).cumprod()

        # 相对收益率
        relative_returns = cumulative_returns / cumulative_benchmark - 1

        result["DailyReturns"] = daily_returns
        result["MarketReturns"] = market_returns
        result["ExcessReturns"] = excess_returns
        result["CumulativeReturns"] = cumulative_returns
        result["CumulativeBenchmark"] = cumulative_benchmark
        result["RelativeReturns"] = relative_returns

        return result

    def _calculate_alpha_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算Alpha值"""
        result = data.copy()

        daily_returns = result["DailyReturns"]
        market_returns = result["MarketReturns"]
        excess_returns = result["ExcessReturns"]

        # 滚动Alpha计算
        rolling_alpha = pd.Series(index=data.index, dtype=float)
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
                    else:
                        beta = 1.0

                    # Alpha计算（CAPM模型）
                    portfolio_return = window_returns.mean() * 252  # TODO: 将魔法数字提取到配置中
                    market_return = window_market.mean() * 252  # TODO: 将魔法数字提取到配置中
                    alpha = portfolio_return - (self.risk_free_rate + beta * (market_return - self.risk_free_rate))

                    rolling_alpha.iloc[i] = alpha
                    rolling_beta.iloc[i] = beta
                except:
                    # 如果计算失败，使用默认值
                    rolling_alpha.iloc[i] = 0.0
                    rolling_beta.iloc[i] = 1.0
            else:
                # 数据不足，使用默认值
                rolling_alpha.iloc[i] = 0.0
                rolling_beta.iloc[i] = 1.0

        # Alpha值标准化
        alpha_value = rolling_alpha / 252  # 日化Alpha  # TODO: 将魔法数字提取到配置中

        # 信息比率
        tracking_error = excess_returns.rolling(window=self.rolling_window).std() * np.sqrt(
            252
        )  # TODO: 将魔法数字提取到配置中
        information_ratio = (
            excess_returns.rolling(window=self.rolling_window).mean() * 252
        ) / tracking_error  # TODO: 将魔法数字提取到配置中

        result["RollingAlpha"] = rolling_alpha.fillna(0.0)
        result["RollingBeta"] = rolling_beta.fillna(1.0)
        result["AlphaValue"] = alpha_value.fillna(0.0)
        result["InformationRatio"] = information_ratio.fillna(0.0)

        return result

    def _calculate_factor_contributions(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算因子贡献"""
        result = data.copy()

        daily_returns = result["DailyReturns"]
        volume = result["volume"]
        close = result["close"]
        alpha_value = result["AlphaValue"]

        # 因子1: 规模因子
        size_factor = np.log(volume / volume.rolling(window=self.rolling_window).mean())
        size_contribution = size_factor * alpha_value * 0.2

        # 因子2: 价值因子
        value_factor = close.pct_change(self.rolling_window)
        value_contribution = value_factor * alpha_value * 0.25  # TODO: 将魔法数字提取到配置中

        # 因子3: 动量因子
        momentum_factor = daily_returns.rolling(window=self.rolling_window).mean()
        momentum_contribution = momentum_factor * alpha_value * 0.25  # TODO: 将魔法数字提取到配置中

        # 因子4: 质量因子
        quality_factor = daily_returns.rolling(window=self.rolling_window).std()
        quality_contribution = (1 / (1 + quality_factor)) * alpha_value * 0.15  # TODO: 将魔法数字提取到配置中

        # 因子5: 低波动因子
        volatility_factor = daily_returns.rolling(window=self.rolling_window).std()
        volatility_contribution = (1 / (1 + volatility_factor)) * alpha_value * 0.15  # TODO: 将魔法数字提取到配置中

        # 总因子贡献
        total_factor_contribution = (
            size_contribution.fillna(0)
            + value_contribution.fillna(0)
            + momentum_contribution.fillna(0)
            + quality_contribution.fillna(0)
            + volatility_contribution.fillna(0)
        )

        result["SizeContribution"] = size_contribution.fillna(0)
        result["ValueContribution"] = value_contribution.fillna(0)
        result["MomentumContribution"] = momentum_contribution.fillna(0)
        result["QualityContribution"] = quality_contribution.fillna(0)
        result["VolatilityContribution"] = volatility_contribution.fillna(0)
        result["FactorContribution"] = total_factor_contribution

        return result

    def _calculate_alpha_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算Alpha评分"""
        result = data.copy()

        alpha_value = result["AlphaValue"]
        information_ratio = result["InformationRatio"]
        factor_contribution = result["FactorContribution"]
        rolling_alpha = result["RollingAlpha"]

        # Alpha值评分 (0-40分)
        alpha_score = np.clip(
            rolling_alpha * 1000 + 20, 0, 40
        )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        # 信息比率评分 (0-30分)
        info_ratio_score = np.clip(
            information_ratio * 15 + 15, 0, 30
        )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        # 因子贡献评分 (0-20分)
        factor_score = np.clip(
            abs(factor_contribution) * 1000 + 10, 0, 20
        )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        # 稳定性评分 (0-10分)
        alpha_stability = 1 / (1 + alpha_value.rolling(window=self.rolling_window).std().fillna(1))
        stability_score = alpha_stability * 10

        # 综合Alpha评分
        alpha_total_score = (
            alpha_score.fillna(20)  # TODO: 将魔法数字提取到配置中
            + info_ratio_score.fillna(15)  # TODO: 将魔法数字提取到配置中
            + factor_score.fillna(10)
            + stability_score.fillna(5)  # TODO: 将魔法数字提取到配置中
        )

        # 标准化到0-100范围
        alpha_total_score = np.clip(alpha_total_score, 0, 100)

        result["AlphaValueScore"] = alpha_score.fillna(20)  # TODO: 将魔法数字提取到配置中
        result["InfoRatioScore"] = info_ratio_score.fillna(15)  # TODO: 将魔法数字提取到配置中
        result["FactorScore"] = factor_score.fillna(10)
        result["StabilityScore"] = stability_score.fillna(
            5
        )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        result["AlphaScore"] = alpha_total_score.fillna(50)  # TODO: 将魔法数字提取到配置中

        return result

    def _generate_alpha_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成Alpha信号"""
        result = data.copy()

        alpha_value = result["AlphaValue"]
        alpha_score = result["AlphaScore"]
        rolling_alpha = result["RollingAlpha"]
        information_ratio = result["InformationRatio"]
        factor_contribution = result["FactorContribution"]

        # Alpha信号
        result["AlphaSignal"] = alpha_score >= 75  # TODO: 将魔法数字提取到配置中

        # 正Alpha信号
        result["PositiveAlphaSignal"] = (rolling_alpha >= self.alpha_threshold) & (  # 显著正Alpha
            alpha_value >= 0.001
        )  # Alpha值显著

        # 信息比率信号
        result["InfoRatioSignal"] = information_ratio >= 0.5  # TODO: 将魔法数字提取到配置中

        # 因子贡献信号
        factor_change = factor_contribution.diff()
        result["FactorSignal"] = abs(factor_change) >= 0.005  # TODO: 将魔法数字提取到配置中

        # Alpha改进信号
        score_improvement = alpha_score.diff()
        result["AlphaImprovement"] = score_improvement > 10

        # 综合Alpha判断
        result["IsGoodAlpha"] = alpha_score >= 70  # TODO: 将魔法数字提取到配置中
        result["IsPoorAlpha"] = alpha_score <= 30  # TODO: 将魔法数字提取到配置中

        return result

    def _calculate_baseindicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        BaseIndicator要求的抽象方法实现

        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 其他参数

        Returns:
            包含ZXM Alpha生成指标的DataFrame
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
        return 0.87  # ZXM Alpha生成指标置信度  # TODO: 将魔法数字提取到配置中

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
        return result["AlphaScore"]

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

        # Alpha生成形态
        alpha_score = result["AlphaScore"]
        patterns["ZXM_EXCELLENT_ALPHA"] = alpha_score >= 85  # TODO: 将魔法数字提取到配置中
        patterns["ZXM_GOOD_ALPHA"] = (alpha_score >= 65) & (
            alpha_score < 85
        )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        patterns["ZXM_NEUTRAL_ALPHA"] = (alpha_score >= 45) & (
            alpha_score < 65
        )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        patterns["ZXM_POOR_ALPHA"] = (alpha_score >= 25) & (
            alpha_score < 45
        )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        patterns["ZXM_BAD_ALPHA"] = alpha_score < 25  # TODO: 将魔法数字提取到配置中

        # Alpha生成信号形态
        patterns["ZXM_ALPHA_SIGNAL"] = result["AlphaSignal"]
        patterns["ZXM_POSITIVE_ALPHA"] = result["PositiveAlphaSignal"]
        patterns["ZXM_INFO_RATIO_SIGNAL"] = result["InfoRatioSignal"]
        patterns["ZXM_FACTOR_SIGNAL"] = result["FactorSignal"]
        patterns["ZXM_ALPHA_IMPROVEMENT"] = result["AlphaImprovement"]

        return patterns

    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """
        BaseIndicator要求的参数设置方法

        Args:
            **kwargs: 参数字典
        """
        self.set_parameters_Alpha_Generation(**kwargs)
