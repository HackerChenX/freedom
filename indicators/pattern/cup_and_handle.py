from utils.container import container

"""
杯柄形态识别指标
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin
from utils.logger import get_logger

logger = get_logger(__name__)


class CupAndHandle(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
    """
    杯柄形态识别指标

    杯柄形态是一种看涨的持续形态，特征是：
    1. 杯子部分：U型或圆形底部
    2. 柄部分：小幅回调整理
    3. 突破时成交量放大  # TODO: 将魔法数字提取到配置中
    """

    def __init__(self, period: int = 50, handle_period: int = 10):  # TODO: 将魔法数字提取到配置中
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        """
        初始化杯柄形态识别指标

        Args:
            period: 杯子部分的计算周期
            handle_period: 柄部分的计算周期
        """
        self.name = "CUP_AND_HANDLE"
        self.period = period
        self.handle_period = handle_period
        self._parameters = {"period": period, "handle_period": handle_period, "price_col": "close"}

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算杯柄形态

        Args:
            data: 包含OHLCV数据的DataFrame

        Returns:
            包含杯柄形态信号的DataFrame
        """
        try:
            if len(data) < self.period + self.handle_period:
                return pd.DataFrame()

            result = data.copy()

            # 识别杯子形态
            result["cup"] = self._identify_cup(result)

            # 识别柄形态
            result["handle"] = self._identify_handle(result)

            # 识别完整的杯柄形态
            result["cup_and_handle"] = self._identify_cup_and_handle(result)

            # 识别突破信号
            result["breakout"] = self._identify_breakout(result)

            return result

        except Exception as e:
            logger.error(f"杯柄形态计算失败: {e}")
            return pd.DataFrame()

    def _identify_cup(self, data: pd.DataFrame) -> pd.Series:
        """识别杯子形态"""
        try:
            cup = pd.Series(False, index=data.index)

            for i in range(self.period, len(data)):
                window = data.iloc[i - self.period : i]

                # 找到最高点和最低点
                high_idx = window["high"].idxmax()
                low_idx = window["low"].idxmin()

                # 检查是否形成U型
                high_price = window.loc[high_idx, "high"]
                low_price = window.loc[low_idx, "low"]

                # 杯子深度应该在10%-50%之间  # TODO: 将魔法数字提取到配置中
                depth = (high_price - low_price) / high_price
                if 0.1 <= depth <= 0.5:  # TODO: 将魔法数字提取到配置中
                    # 检查最低点是否在中间部分
                    low_position = (low_idx - window.index[0]) / len(window)
                    if 0.3 <= low_position <= 0.7:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                        cup.iloc[i] = True

            return cup

        except Exception as e:
            logger.error(f"杯子形态识别失败: {e}")
            return pd.Series(False, index=data.index)

    def _identify_handle(self, data: pd.DataFrame) -> pd.Series:
        """识别柄形态"""
        try:
            handle = pd.Series(False, index=data.index)

            for i in range(self.handle_period, len(data)):
                if data["cup"].iloc[i - self.handle_period]:
                    window = data.iloc[i - self.handle_period : i]

                    # 柄部分应该是小幅回调
                    high_price = window["high"].max()
                    low_price = window["low"].min()
                    current_price = data["close"].iloc[i - 1]

                    # 回调幅度应该小于杯子深度的1/3  # TODO: 将魔法数字提取到配置中
                    pullback = (high_price - low_price) / high_price
                    if pullback <= 0.15:  # 回调不超过15%  # TODO: 将魔法数字提取到配置中
                        handle.iloc[i] = True

            return handle

        except Exception as e:
            logger.error(f"柄形态识别失败: {e}")
            return pd.Series(False, index=data.index)

    def _identify_cup_and_handle(self, data: pd.DataFrame) -> pd.Series:
        """识别完整的杯柄形态"""
        try:
            cup_and_handle = pd.Series(False, index=data.index)

            for i in range(len(data)):
                if data["cup"].iloc[i] and data["handle"].iloc[i]:
                    cup_and_handle.iloc[i] = True

            return cup_and_handle

        except Exception as e:
            logger.error(f"杯柄形态识别失败: {e}")
            return pd.Series(False, index=data.index)

    def _identify_breakout(self, data: pd.DataFrame) -> pd.Series:
        """识别突破信号"""
        try:
            breakout = pd.Series(False, index=data.index)

            for i in range(1, len(data)):
                if data["cup_and_handle"].iloc[i - 1]:
                    # 计算阻力位（杯子顶部）
                    cup_start = i - self.period - self.handle_period
                    if cup_start >= 0:
                        resistance = data["high"].iloc[cup_start : i - self.handle_period].max()

                        # 检查是否突破阻力位
                        if data["close"].iloc[i] > resistance * 1.02:  # 突破2%
                            breakout.iloc[i] = True

            return breakout

        except Exception as e:
            logger.error(f"突破信号识别失败: {e}")
            return pd.Series(False, index=data.index)

    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        获取杯柄形态交易信号

        Args:
            data: 计算结果数据

        Returns:
            包含交易信号的字典
        """
        try:
            if data.empty:
                return {"signal": "HOLD", "strength": 0, "message": "数据不足"}

            # 获取最新信号
            latest_cup_and_handle = data["cup_and_handle"].iloc[-1] if "cup_and_handle" in data.columns else False
            latest_breakout = data["breakout"].iloc[-1] if "breakout" in data.columns else False

            if latest_breakout:
                return {
                    "signal": "BUY",
                    "strength": 0.9,  # TODO: 将魔法数字提取到配置中
                    "message": "杯柄形态突破，强烈建议买入",
                }
            elif latest_cup_and_handle:
                return {
                    "signal": "HOLD",
                    "strength": 0.7,  # TODO: 将魔法数字提取到配置中
                    "message": "杯柄形态形成，等待突破",
                }
            else:
                return {
                    "signal": "HOLD",
                    "strength": 0.5,  # TODO: 将魔法数字提取到配置中
                    "message": "未检测到杯柄形态",
                }

        except Exception as e:
            logger.error(f"杯柄形态信号生成失败: {e}")
            return {"signal": "HOLD", "strength": 0, "message": f"信号生成失败: {e}"}

    def minimum_periods(self) -> int:
        """
        杯柄形态指标所需的最少数据周期数

        Returns:
            int: 最少需要的数据周期数
        """
        return self.period + self.handle_period + 10

    # ==================== BaseIndicator抽象方法实现 ====================

    def _calculate_baseindicator(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """抽象基类要求的计算方法"""
        return self.calculate(data, *args, **kwargs)

    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """抽象基类要求的评分方法"""
        if len(data) < self.minimum_periods():
            return pd.Series(50.0, index=data.index)  # TODO: 将魔法数字提取到配置中

        # 计算杯柄形态识别结果
        result = self.calculate(data)

        # 基于杯柄形态计算评分
        score = pd.Series(50.0, index=data.index)  # 默认中性评分  # TODO: 将魔法数字提取到配置中

        if isinstance(result, pd.DataFrame) and not result.empty:
            if "cup_and_handle" in result.columns:
                score[result["cup_and_handle"]] = 85.0  # 杯柄形态，强烈看涨  # TODO: 将魔法数字提取到配置中
            if "cup_forming" in result.columns:
                score[result["cup_forming"]] = 65.0  # 杯子形成中，轻微看涨  # TODO: 将魔法数字提取到配置中
            if "handle_forming" in result.columns:
                score[result["handle_forming"]] = 70.0  # 柄部形成中，看涨  # TODO: 将魔法数字提取到配置中

        return score

    def get_patterns_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """抽象基类要求的形态方法"""
        if len(data) < self.minimum_periods():
            return pd.DataFrame(index=data.index)

        return self.calculate(data)

    def calculate_confidence_Indicator_Base_Indicator(
        self, score: pd.Series, patterns: List[str], signals: Dict[str, pd.Series]
    ) -> float:
        """抽象基类要求的置信度方法"""
        base_confidence = 0.8  # 杯柄形态可靠性较高  # TODO: 将魔法数字提取到配置中

        # 基于形态数量调整置信度
        if patterns:
            pattern_bonus = min(
                0.15, len(patterns) * 0.05
            )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            base_confidence += pattern_bonus

        return min(1.0, base_confidence)

    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """抽象基类要求的参数设置方法"""
        for key, value in kwargs.items():
            if key in self._parameters:
                self._parameters[key] = value
                if key == "period":
                    self.period = value
                elif key == "handle_period":
                    self.handle_period = value
