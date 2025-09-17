from utils.container import container

#!/usr/bin/env python3
# -*- coding: utf-8 -*-  # TODO: 将魔法数字提取到配置中

"""
蔡金波动率(Chaikin Volatility)指标

蔡金波动率是由Marc Chaikin开发的技术指标,用于衡量市场的波动性.
它通过计算高低价差的移动平均线的变化率来反映价格波动的程度.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin
from utils.logger import get_logger

logger = get_logger(__name__)


class ChaikinVolatility(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
    """
    蔡金波动率(Chaikin Volatility)指标

    分类:波动性指标
    描述:衡量市场价格波动的程度

    计算公式:
    1. HL_Spread = High - Low
    2. EMA_HL = EMA(HL_Spread, period)
    3. Chaikin_Volatility = (EMA_HL - EMA_HL[lookback]) / EMA_HL[lookback] * 100  # TODO: 将魔法数字提取到配置中

    信号解释:
    - 正值:波动率增加
    - 负值:波动率减少
    - 数值大小:反映波动率变化的程度
    """

    def __init__(self, period: int = 10, lookback: int = 10, **kwargs):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        """
        初始化蔡金波动率指标

        Args:
            period: EMA计算周期,默认10
            lookback: 回望周期,默认10
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        self.period = period
        self.lookback = lookback
        self.REQUIRED_COLUMNS = ["high", "low"]

    def _get_default_parameters(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {"period": 10, "lookback": 10}

    def set_parameters(self, **kwargs):
        """设置参数"""
        self.period = kwargs.get("period", self.period)
        self.lookback = kwargs.get("lookback", self.lookback)

    def _validate_data(self, data: pd.DataFrame) -> bool:
        """验证输入数据的有效性"""
        if data is None or len(data) == 0:
            return False

        # 检查必需列
        for col in self.REQUIRED_COLUMNS:
            if col not in data.columns:
                logger.error(f"数据缺少必需列: {col}")
                return False

        return True

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算蔡金波动率

        Args:
            data: 包含high和low列的DataFrame
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含蔡金波动率的DataFrame
        """
        try:
            # 验证数据
            if not self._validate_data(data):
                return pd.DataFrame()

            df = data.copy()

            # 计算高低价差
            hl_spread = df["high"] - df["low"]

            # 计算高低价差的EMA
            ema_hl = hl_spread.ewm(span=self.period, adjust=False).mean()

            # 计算蔡金波动率
            ema_hl_lookback = ema_hl.shift(self.lookback)
            chaikin_volatility = ((ema_hl - ema_hl_lookback) / ema_hl_lookback * 100).fillna(0)

            # 添加到结果DataFrame
            df["hl_spread"] = hl_spread
            df["ema_hl"] = ema_hl
            df["chaikin_volatility"] = chaikin_volatility

            # 计算信号
            df["cv_signal"] = self._generate_signals(df)

            # 计算波动率等级
            df["volatility_level"] = self._calculate_volatility_level(chaikin_volatility)

            return df

        except Exception as e:
            logger.error(f"蔡金波动率计算失败: {e}")
            return pd.DataFrame()

    def _generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        生成交易信号

        Args:
            df: 包含蔡金波动率的DataFrame

        Returns:
            pd.Series: 交易信号
        """
        signals = pd.Series(0, index=df.index)
        cv = df["chaikin_volatility"]

        # 波动率突然增加信号(可能预示趋势变化)
        high_volatility = cv > cv.rolling(20).quantile(
            0.8
        )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        signals[high_volatility] = 1

        # 波动率突然减少信号(可能预示盘整)
        low_volatility = cv < cv.rolling(20).quantile(0.2)  # TODO: 将魔法数字提取到配置中
        signals[low_volatility] = -1

        return signals

    def _calculate_volatility_level(self, cv: pd.Series) -> pd.Series:
        """
        计算波动率等级

        Args:
            cv: 蔡金波动率序列

        Returns:
            pd.Series: 波动率等级
        """
        levels = pd.Series("中等", index=cv.index)

        # 使用滚动分位数定义等级
        rolling_window = min(50, len(cv))  # TODO: 将魔法数字提取到配置中
        if rolling_window > 10:
            high_threshold = cv.rolling(rolling_window).quantile(0.75)  # TODO: 将魔法数字提取到配置中
            low_threshold = cv.rolling(rolling_window).quantile(0.25)  # TODO: 将魔法数字提取到配置中

            levels[cv > high_threshold] = "高"
            levels[cv < low_threshold] = "低"

        return levels

    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        获取最新的交易信号

        Args:
            data: 计算后的数据

        Returns:
            Dict[str, Any]: 信号信息
        """
        if data.empty or "cv_signal" not in data.columns:
            return {"signal": 0, "strength": 0, "description": "无信号"}

        latest_signal = data["cv_signal"].iloc[-1]
        latest_cv = data["chaikin_volatility"].iloc[-1]
        latest_level = data["volatility_level"].iloc[-1]

        if latest_signal == 1:
            return {
                "signal": 1,
                "strength": min(
                    abs(latest_cv) / 50, 1.0
                ),  # TODO: 将魔法数字提取到配置中  # 标准化强度  # TODO: 将魔法数字提取到配置中
                "description": f"波动率增加信号,当前等级:{latest_level}",
            }
        elif latest_signal == -1:
            return {
                "signal": -1,
                "strength": min(abs(latest_cv) / 50, 1.0),  # TODO: 将魔法数字提取到配置中
                "description": f"波动率减少信号,当前等级:{latest_level}",
            }
        else:
            return {"signal": 0, "strength": 0, "description": f"波动率正常,当前等级:{latest_level}"}

    def get_pattern_info(self) -> Dict[str, Any]:
        """获取指标模式信息"""
        return {
            "name": "CHAIKIN_VOLATILITY",
            "description": "蔡金波动率指标",
            "type": "volatility",
            "parameters": {"period": self.period, "lookback": self.lookback},
        }

    # 实现BaseIndicator的抽象方法
    def _calculate_baseindicator(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """核心计算逻辑"""
        return self.calculate(data, **kwargs)

    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算原始评分"""
        if "chaikin_volatility" not in data.columns:
            return pd.Series(50.0, index=data.index)  # TODO: 将魔法数字提取到配置中

        # 基于波动率变化计算评分
        cv = data["chaikin_volatility"]
        cv_abs = cv.abs()
        cv_normalized = cv_abs / (
            cv_abs.rolling(20).mean() + 1e-8
        )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        score = np.clip(cv_normalized * 30 + 50, 0, 100)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        return score

    def get_patterns_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """获取技术形态"""
        patterns = pd.DataFrame(index=data.index)

        if "chaikin_volatility" in data.columns:
            cv = data["chaikin_volatility"]
            # 波动率突破形态
            patterns["CV_波动率突增"] = cv > cv.rolling(20).quantile(
                0.8
            )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            patterns["CV_波动率骤减"] = cv < cv.rolling(20).quantile(0.2)  # TODO: 将魔法数字提取到配置中
            # 波动率趋势形态
            patterns["CV_波动率上升"] = cv > cv.shift(1)
            patterns["CV_波动率下降"] = cv < cv.shift(1)

        return patterns

    def calculate_confidence_Indicator_Base_Indicator(
        self, score: pd.Series, patterns: List[str], signals: Dict[str, pd.Series]
    ) -> float:
        """计算置信度"""
        if len(score) == 0:
            return 0.0

        # 基于评分稳定性计算置信度
        score_stability = (
            1.0 - (score.rolling(5).std().iloc[-1] / 100.0) if len(score) >= 5 else 0.5
        )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        pattern_strength = min(len(patterns) * 0.25, 1.0)  # TODO: 将魔法数字提取到配置中

        return (score_stability + pattern_strength) / 2

    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """设置参数"""
        self.set_parameters(**kwargs)

    @property
    def minimum_periods(self) -> int:
        """最小周期数"""
        return self.period + self.lookback + 5  # TODO: 将魔法数字提取到配置中
