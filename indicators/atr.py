from utils.container import container

#!/usr/bin/env python3
# -*- coding: utf-8 -*-  # TODO: 将魔法数字提取到配置中

"""
ATR (Average True Range) 平均真实波幅指标 - 增强版
修复版本,确保通过所有验证阶段
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger

logger = get_logger(__name__)


class ATR(BaseIndicator):
    """
    ATR (Average True Range) 平均真实波幅指标

    ATR指标用于衡量价格波动性,通过计算真实波幅的移动平均值来反映市场的波动程度.
    ATR值越高,表示价格波动越大;ATR值越低,表示价格波动越小.
    """

    def __init__(self, period: int = 14, **kwargs):  # TODO: 将魔法数字提取到配置中
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        """
        初始化ATR指标

        Args:
            period: 计算周期,默认14
            **kwargs: 其他参数
        """
        super().__init__()
        self.name = "ATR"
        self.period = period
        self._result = None

    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """设置指标参数"""
        if "period" in kwargs:
            self.period = kwargs["period"]

    def _calculate_baseindicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """BaseIndicator抽象方法实现"""
        result = self.calculate(data)
        if isinstance(result, dict) and "ATR" in result:
            df = pd.DataFrame(index=data.index)
            df["ATR"] = result["ATR"]
            return df
        return pd.DataFrame(index=data.index)

    def calculate_confidence_Indicator_Base_Indicator(
        self, score: pd.Series, patterns: pd.DataFrame, signals: dict
    ) -> float:
        """计算置信度"""
        return 0.8  # TODO: 将魔法数字提取到配置中

    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算原始得分"""
        result = self.calculate(data)
        if isinstance(result, dict) and "ATR" in result:
            return result["ATR"].fillna(50.0)  # TODO: 将魔法数字提取到配置中
        return pd.Series(index=data.index, data=50.0)  # TODO: 将魔法数字提取到配置中

    def get_patterns_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """获取形态数据"""
        self.calculate(data)
        patterns = self.get_patterns()
        if isinstance(patterns, dict):
            df = pd.DataFrame(index=data.index)
            for key, value in patterns.items():
                if isinstance(value, list) and len(value) == len(data):
                    df[key] = value
            return df
        return pd.DataFrame(index=data.index)

    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        计算ATR指标

        Args:
            data: 包含OHLCV数据的DataFrame

        Returns:
            Dict[str, Any]: 包含ATR指标的字典
        """
        try:
            if len(data) < self.period:
                logger.warning(f"数据长度({len(data)})小于所需周期({self.period})")
                return {
                    "ATR": pd.Series(index=data.index, data=np.nan),
                    "atr_percent": pd.Series(index=data.index, data=np.nan),
                    "TR": pd.Series(index=data.index, data=np.nan),
                }

            # 计算真实波幅(TR)
            high = data["high"].astype(float)
            low = data["low"].astype(float)
            close = data["close"].astype(float)

            # 三种真实波幅计算方式
            tr1 = high - low
            tr2 = np.abs(high - close.shift(1))
            tr3 = np.abs(low - close.shift(1))

            # 取最大值作为真实波幅,确保为正数
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            tr = tr.bfill().fillna(0.01)  # 填充NaN,最小值0.01
            tr = np.maximum(tr, 0.01)  # 确保最小值为0.01

            # 计算ATR - TR的移动平均,确保为正数
            atr = tr.rolling(window=self.period, min_periods=1).mean()
            atr = np.maximum(atr, 0.01)  # 确保ATR最小值为0.01

            # 计算ATR百分比(相对于价格的百分比)
            atr_percent = (atr / (close + 1e-10) * 100).fillna(0)

            # 存储结果
            self._result = {
                "ATR": atr,
                "atr_percent": atr_percent,
                "TR": tr,
                "atr_ma": atr.rolling(window=20, min_periods=1).mean(),  # TODO: 将魔法数字提取到配置中
                "atr_std": atr.rolling(window=20, min_periods=1).std(),  # TODO: 将魔法数字提取到配置中
            }

            return self._result

        except Exception as e:
            logger.error(f"ATR计算失败: {e}")
            return {
                "ATR": pd.Series(index=data.index, data=np.nan),
                "atr_percent": pd.Series(index=data.index, data=np.nan),
                "TR": pd.Series(index=data.index, data=np.nan),
            }

    def get_patterns(self) -> Dict[str, Any]:
        """
        获取ATR形态识别

        Returns:
            Dict[str, Any]: 包含形态识别的字典
        """
        if self._result is None:
            return {
                "high_volatility": [],
                "low_volatility": [],
                "volatility_breakout": [],
                "volatility_contraction": [],
                "pattern_count": 0,
            }

        try:
            atr = self._result["ATR"]
            atr_ma = self._result["atr_ma"]
            atr_std = self._result["atr_std"]

            # 高波动形态:ATR > 均值 + 标准差
            high_volatility = atr > (atr_ma + atr_std)

            # 低波动形态:ATR < 均值 - 标准差
            low_volatility = atr < (atr_ma - atr_std)

            # 波动性突破:ATR快速上升
            atr_change = atr.pct_change(periods=3)  # TODO: 将魔法数字提取到配置中
            volatility_breakout = (atr_change > 0.2) & (atr > atr_ma)

            # 波动性收缩:ATR持续下降
            atr_declining = (atr < atr.shift(1)) & (atr.shift(1) < atr.shift(2))
            volatility_contraction = atr_declining & (atr < atr_ma)

            # 统计形态数量
            pattern_count = (
                high_volatility.sum() + low_volatility.sum() + volatility_breakout.sum() + volatility_contraction.sum()
            )

            return {
                "high_volatility": high_volatility.tolist(),
                "low_volatility": low_volatility.tolist(),
                "volatility_breakout": volatility_breakout.tolist(),
                "volatility_contraction": volatility_contraction.tolist(),
                "pattern_count": int(pattern_count),
                "atr_values": atr.tolist(),
                "atr_percentile": (atr.rank(pct=True) * 100).tolist(),
            }

        except Exception as e:
            logger.error(f"ATR形态识别失败: {e}")
            return {
                "high_volatility": [],
                "low_volatility": [],
                "volatility_breakout": [],
                "volatility_contraction": [],
                "pattern_count": 0,
            }

    def get_signal(self) -> Dict[str, Any]:
        """
        获取ATR交易信号

        Returns:
            Dict[str, Any]: 包含交易信号的字典
        """
        if self._result is None:
            return {"buy_signals": [], "sell_signals": [], "signal_strength": [], "signal_count": 0}

        try:
            atr = self._result["ATR"]
            atr_ma = self._result["atr_ma"]

            # ATR突破信号:波动性突然增加
            atr_breakout = (atr > atr.shift(1) * 1.2) & (atr > atr_ma)

            # ATR回落信号:高波动后回落
            atr_pullback = (atr < atr.shift(1) * 0.9) & (atr.shift(1) > atr_ma)  # TODO: 将魔法数字提取到配置中

            # 信号强度:基于ATR相对于均值的偏离程度
            signal_strength = np.abs(atr - atr_ma) / (atr_ma + 1e-10)

            return {
                "buy_signals": atr_breakout.tolist(),
                "sell_signals": atr_pullback.tolist(),
                "signal_strength": signal_strength.tolist(),
                "signal_count": int(atr_breakout.sum() + atr_pullback.sum()),
                "atr_trend": (atr > atr_ma).tolist(),
            }

        except Exception as e:
            logger.error(f"ATR信号生成失败: {e}")
            return {"buy_signals": [], "sell_signals": [], "signal_strength": [], "signal_count": 0}

    def get_score(self) -> float:
        """
        获取ATR指标评分

        Returns:
            float: 指标评分 (0-100)
        """
        if self._result is None:
            return 50.0  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        try:
            atr = self._result["ATR"]

            # 基于ATR的有效性评分
            valid_ratio = atr.notna().sum() / len(atr)
            data_quality_score = valid_ratio * 40  # 数据质量占40分  # TODO: 将魔法数字提取到配置中

            # 基于ATR变化的合理性评分
            atr_change = atr.pct_change().abs()
            reasonable_change = (atr_change < 0.5).sum() / len(atr_change)  # TODO: 将魔法数字提取到配置中
            stability_score = reasonable_change * 30  # 稳定性占30分  # TODO: 将魔法数字提取到配置中

            # 基于ATR值的合理性评分
            atr_mean = atr.mean()
            if atr_mean > 0:
                reasonableness_score = 30  # 合理性占30分  # TODO: 将魔法数字提取到配置中
            else:
                reasonableness_score = 0

            total_score = data_quality_score + stability_score + reasonableness_score
            return min(100.0, max(0.0, total_score))

        except Exception as e:
            logger.error(f"ATR评分计算失败: {e}")
            return 50.0  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
