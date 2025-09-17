from utils.container import container
import pandas as pd
import numpy as np
from typing import Dict, Any, List

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin
from utils.logger import get_logger

logger = get_logger(__name__)


class Fibonacci(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
    """
    FIBONACCI 指标

    自动生成的标准化实现
    """

    @property
    def minimum_periods(self) -> int:
        """返回计算指标所需的最小周期数"""
        return getattr(self, "period", 14) + 1  # TODO: 将魔法数字提取到配置中

    def __init__(self, **kwargs):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        """
        初始化FIBONACCI指标

        Args:
            **kwargs: 指标参数
        """
        super().__init__()
        self.name = "FIBONACCI"
        self.description = "斐波那契指标，用于识别支撑阻力位和回调目标"
        self._result = None  # 初始化结果存储

        # 设置默认参数
        self._default_parameters = self._get_default_parameters_fibonacci()

        # 应用用户参数
        self.set_parameters_Fibonacci(**kwargs)

    def _get_default_parameters_fibonacci(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {"period": 14}  # TODO: 将魔法数字提取到配置中

    def set_parameters_Fibonacci(self, **kwargs):
        """
        设置指标参数

        Args:
            **kwargs: 参数字典
        """
        # 验证参数
        try:
            from utils.indicator_parameter_validator import IndicatorParameterValidator
            from db.sql_manager import SQLManager, QueryType

            validator = IndicatorParameterValidator()

            # 合并默认参数和用户参数
            params = self._default_parameters.copy()
            params.update(kwargs)

            # 验证参数
            is_valid, errors = validator.validate_indicator_parameters("FIBONACCI", params)
            if not is_valid:
                # 静默处理验证失败，避免过多警告
                pass

        except Exception:
            # 如果验证失败，静默处理，保持向后兼容
            pass

        # 设置参数
        self.period = kwargs.get("period", 14)  # TODO: 将魔法数字提取到配置中

    def calculate_Fibonacci(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算FIBONACCI指标

        Args:
            data: 包含OHLCV数据的Data_frame

        Returns:
            添加了FIBONACCI指标的Data_frame
        """
        result = self._calculate_fibonacci(data, **kwargs)
        self._result = result
        return result

    def _calculate_fibonacci(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        内部计算FIBONACCI指标

        Args:
            data: 包含OHLCV数据的Data_frame

        Returns:
            添加了FIBONACCI指标的Data_frame
        """
        df = data.copy()

        # 计算斐波那契回调线
        if len(df) > self.period:
            # 找到最高点和最低点
            high_period = df["high"].rolling(window=self.period).max()
            low_period = df["low"].rolling(window=self.period).min()

            # 计算斐波那契水平
            fib_range = high_period - low_period

            # 斐波那契回调比例
            fib_levels = [
                0.236,
                0.382,
                0.5,
                0.618,
                0.786,
            ]  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

            for level in fib_levels:
                df[f"fib_ret_{level}"] = high_period - (fib_range * level)
        else:
            # 数据不足时返回NaN
            for level in [
                0.236,
                0.382,
                0.5,
                0.618,
                0.786,
            ]:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                df[f"fib_ret_{level}"] = np.nan

        # 基本的FIBONACCI_VALUE（向后兼容）
        df[f"FIBONACCI_VALUE"] = df["close"].rolling(window=self.period).mean()

        # 添加形态识别和信号生成
        df = self.add_pattern_detection(df)
        df = self.add_signal_generation(df)

        return df

    def calculate_raw_score_Fibonacci(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算原始评分"""
        if not self.has_result():
            self.calculate_Fibonacci(data, **kwargs)

        # 基于价格接近斐波那契水平的程度计算评分
        result = self._result
        if result is None or len(result) == 0:
            return pd.Series(50.0, index=data.index)  # TODO: 将魔法数字提取到配置中

        # 计算当前价格与各斐波那契水平的距离
        close_prices = data["close"]
        scores = []

        for i, close in enumerate(close_prices):
            if pd.isna(close):
                scores.append(50.0)  # TODO: 将魔法数字提取到配置中
                continue

            min_distance = float("inf")
            fib_levels = [
                0.236,
                0.382,
                0.5,
                0.618,
                0.786,
            ]  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

            for level in fib_levels:
                col_name = f"fib_ret_{level}"
                if col_name in result.columns and i < len(result):
                    fib_value = result.iloc[i][col_name]
                    if not pd.isna(fib_value):
                        distance = abs(close - fib_value) / close
                        min_distance = min(min_distance, distance)

            # 距离越小，评分越高
            if min_distance != float("inf"):
                score = max(0, 100 - (min_distance * 1000))  # 转换为0-100评分  # TODO: 将魔法数字提取到配置中
            else:
                score = 50.0  # TODO: 将魔法数字提取到配置中

            scores.append(min(100, max(0, score)))

        return pd.Series(scores, index=data.index)

    def calculate_confidence_Fibonacci(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """计算置信度"""
        if len(score.dropna()) == 0:
            return 0.5  # TODO: 将魔法数字提取到配置中

        # 基于评分的平均值和模式强度计算置信度
        avg_score = score.dropna().mean()
        confidence = avg_score / 100.0

        # 根据模式强度调整置信度
        if not patterns.empty and len(patterns.columns) > 0:
            pattern_strength = patterns.sum(axis=1).mean() if len(patterns) > 0 else 0
            confidence = min(1.0, confidence + pattern_strength * 0.1)

        return max(0.0, min(1.0, confidence))

    def get_patterns_Fibonacci(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """获取形态"""
        if not self.has_result():
            self.calculate_Fibonacci(data, **kwargs)

        result = self._result
        if result is None or len(result) == 0:
            return pd.DataFrame(index=data.index)

        patterns = pd.DataFrame(index=data.index)

        # 检测价格接近斐波那契水平的模式
        close_prices = data["close"]
        fib_levels = [
            0.236,
            0.382,
            0.5,
            0.618,
            0.786,
        ]  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        for level in fib_levels:
            col_name = f"fib_ret_{level}"
            if col_name in result.columns:
                fib_values = result[col_name]
                # 价格接近该斐波那契水平（误差在1%以内）
                near_fib = (abs(close_prices - fib_values) / close_prices) < 0.01
                patterns[f"FIBONACCI_NEAR_{level}"] = near_fib.fillna(False)

        return patterns

    # === 抽象方法实现 ===
    def _calculate_baseindicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """BaseIndicator抽象方法实现"""
        return self.calculate_Fibonacci(data, **kwargs)

    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """BaseIndicator抽象方法实现"""
        return self.calculate_raw_score_Fibonacci(data, **kwargs)

    def get_patterns_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """BaseIndicator抽象方法实现"""
        return self.get_patterns_Fibonacci(data, **kwargs)

    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """BaseIndicator抽象方法实现"""
        self.set_parameters_Fibonacci(**kwargs)

    def calculate_confidence_Indicator_Base_Indicator(
        self, score: pd.Series, patterns: pd.DataFrame, signals: dict
    ) -> float:
        """BaseIndicator抽象方法实现"""
        return self.calculate_confidence_Fibonacci(score, patterns, signals)

    # === 兼容性方法 ===
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """公共接口：计算指标"""
        return self.calculate_Fibonacci(data, **kwargs)

    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """公共接口：获取形态"""
        return self.get_patterns_Fibonacci(data, **kwargs)

    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """公共接口：计算原始评分"""
        return self.calculate_raw_score_Fibonacci(data, **kwargs)

    def get_signals(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """公共接口：获取信号"""
        return self.generate_signals_fibonacci(data, **kwargs)

    def calculate_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """公共接口：计算评分"""
        return self.calculate_raw_score_Fibonacci(data, **kwargs)

    def calculate_confidence(self, data: pd.DataFrame, **kwargs) -> float:
        """公共接口：计算置信度"""
        score = self.calculate_raw_score_Fibonacci(data, **kwargs)
        patterns = self.get_patterns_Fibonacci(data, **kwargs)
        signals = self.generate_signals_fibonacci(data, **kwargs)
        return self.calculate_confidence_Fibonacci(score, patterns, signals)

    def set_parameters(self, **kwargs):
        """公共接口：设置参数"""
        self.set_parameters_Fibonacci(**kwargs)

    def compute(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """公共接口：计算指标（别名）"""
        return self.calculate_Fibonacci(data, **kwargs)

    def generate_trading_signals(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """公共接口：生成交易信号"""
        return self.generate_signals_fibonacci(data, **kwargs)

    def register_patterns(self):
        """公共接口：注册形态"""
        pass  # 形态已在计算过程中处理

    def has_result(self) -> bool:
        """检查是否有计算结果"""
        return self._result is not None and not self._result.empty

    def generate_signals_fibonacci(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """生成Fibonacci信号"""
        try:
            if not self.has_result():
                self.calculate_Fibonacci(data, **kwargs)

            result = self._result
            if result is None or len(result) == 0:
                return {
                    "fibonacci_buy_signal": pd.Series([False] * len(data), index=data.index),
                    "fibonacci_sell_signal": pd.Series([False] * len(data), index=data.index),
                }

            close_prices = data["close"]
            buy_signals = pd.Series([False] * len(data), index=data.index)
            sell_signals = pd.Series([False] * len(data), index=data.index)

            # 检测支撑和阻力突破信号
            fib_levels = [
                0.236,
                0.382,
                0.5,
                0.618,
                0.786,
            ]  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

            for i in range(1, len(data)):
                current_price = close_prices.iloc[i]
                previous_price = close_prices.iloc[i - 1]

                if pd.isna(current_price) or pd.isna(previous_price):
                    continue

                for level in fib_levels:
                    col_name = f"fib_ret_{level}"
                    if col_name in result.columns and i < len(result):
                        fib_value = result.iloc[i][col_name]
                        if not pd.isna(fib_value):
                            # 向上突破斐波那契阻力
                            if previous_price <= fib_value and current_price > fib_value:
                                buy_signals.iloc[i] = True
                            # 向下跌破斐波那契支撑
                            elif previous_price >= fib_value and current_price < fib_value:
                                sell_signals.iloc[i] = True

            return {"fibonacci_buy_signal": buy_signals, "fibonacci_sell_signal": sell_signals}

        except (ValueError, KeyError, IndexError) as e:
            logger.warning(f"FIBONACCI信号生成失败: {e}, 返回空信号")
            return {
                "fibonacci_buy_signal": pd.Series([False] * len(data), index=data.index),
                "fibonacci_sell_signal": pd.Series([False] * len(data), index=data.index),
            }


# 类别名，用于向后兼容和注册
FIBONACCI = Fibonacci
Fibonacci_Indicator = Fibonacci
