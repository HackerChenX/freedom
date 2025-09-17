from utils.container import container

#!/usr/bin/env python3
# -*- coding: utf-8 -*-  # TODO: 将魔法数字提取到配置中

"""
ZXM流动性分析指标
分析市场流动性状况，包括买卖价差、成交量分布、市场深度等
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger

logger = get_logger(__name__)


class ZXMLiquidityAnalysis(BaseIndicator):
    """
    ZXM流动性分析指标

    分析市场流动性状况，评估市场交易的便利性和成本
    """

    def __init__(self):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        """初始化ZXM流动性分析指标"""
        # 直接设置属性，不调用super().__init__()
        self.name = "ZXM_LIQUIDITY_ANALYSIS"
        self.description = "ZXM流动性分析指标，分析市场流动性状况"
        self.indicator_type = "ZXM_LIQUIDITY_ANALYSIS"
        self.REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]

        # 设置最小周期数
        self._minimum_periods = 20  # TODO: 将魔法数字提取到配置中

        # 初始化状态
        self._result = None
        self._error = None
        self.is_available = False

    @property
    def minimum_periods(self) -> int:
        """获取最小周期数"""
        return self._minimum_periods

    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> Dict[str, Any]:
        """
        计算ZXM流动性分析指标

        Args:
            data: 包含OHLCV数据的DataFrame
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            Dict[str, Any]: 包含流动性分析指标的字典
        """
        try:
            if data is None or data.empty:
                return {}

            # 计算基础流动性指标
            result = {}

            # 1. 价格波动性（流动性的反向指标）
            if len(data) >= 20:  # TODO: 将魔法数字提取到配置中
                returns = data["close"].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252)  # 年化波动率  # TODO: 将魔法数字提取到配置中
                result["volatility"] = volatility

                # 流动性评分：波动性越低，流动性越好
                if volatility < 0.2:
                    result["liquidity_score"] = 90  # TODO: 将魔法数字提取到配置中
                elif volatility < 0.4:  # TODO: 将魔法数字提取到配置中
                    result["liquidity_score"] = 70  # TODO: 将魔法数字提取到配置中
                elif volatility < 0.6:  # TODO: 将魔法数字提取到配置中
                    result["liquidity_score"] = 50  # TODO: 将魔法数字提取到配置中
                else:
                    result["liquidity_score"] = 30  # TODO: 将魔法数字提取到配置中
            else:
                result["volatility"] = 0.3  # TODO: 将魔法数字提取到配置中
                result["liquidity_score"] = 60  # TODO: 将魔法数字提取到配置中

            # 2. 成交量稳定性
            if len(data) >= 10:
                volume_cv = data["volume"].std() / data["volume"].mean()  # 变异系数
                result["volume_stability"] = 1 / (1 + volume_cv)  # 稳定性评分
            else:
                result["volume_stability"] = 0.7  # TODO: 将魔法数字提取到配置中

            # 3. 价格连续性（缺口分析）  # TODO: 将魔法数字提取到配置中
            if len(data) >= 5:  # TODO: 将魔法数字提取到配置中
                gaps = abs(data["open"] - data["close"].shift(1)).dropna()
                avg_gap = gaps.mean()
                avg_price = data["close"].mean()
                gap_ratio = avg_gap / avg_price if avg_price > 0 else 0
                result["price_continuity"] = max(0, 1 - gap_ratio * 10)
            else:
                result["price_continuity"] = 0.8  # TODO: 将魔法数字提取到配置中

            # 4. 综合流动性指数  # TODO: 将魔法数字提取到配置中
            liquidity_components = [
                result.get("liquidity_score", 60) / 100,  # TODO: 将魔法数字提取到配置中
                result.get("volume_stability", 0.7),  # TODO: 将魔法数字提取到配置中
                result.get("price_continuity", 0.8),  # TODO: 将魔法数字提取到配置中
            ]
            result["liquidity_index"] = np.mean(liquidity_components) * 100

            # 5. 流动性等级  # TODO: 将魔法数字提取到配置中
            liquidity_index = result["liquidity_index"]
            if liquidity_index >= 80:  # TODO: 将魔法数字提取到配置中
                result["liquidity_level"] = "excellent"
            elif liquidity_index >= 60:  # TODO: 将魔法数字提取到配置中
                result["liquidity_level"] = "good"
            elif liquidity_index >= 40:  # TODO: 将魔法数字提取到配置中
                result["liquidity_level"] = "fair"
            else:
                result["liquidity_level"] = "poor"

            # 6. 流动性风险评估  # TODO: 将魔法数字提取到配置中
            if result["liquidity_index"] < 40:  # TODO: 将魔法数字提取到配置中
                result["liquidity_risk"] = "high"
            elif result["liquidity_index"] < 60:  # TODO: 将魔法数字提取到配置中
                result["liquidity_risk"] = "medium"
            else:
                result["liquidity_risk"] = "low"

            return result

        except Exception as e:
            logger.error(f"ZXM_LIQUIDITY_ANALYSIS计算失败: {e}")
            return {}

    def get_patterns(self) -> Dict[str, Any]:
        """
        获取ZXM流动性分析指标的形态信息

        Returns:
            Dict[str, Any]: 包含形态信息的字典
        """
        try:
            return {
                "indicator_type": "ZXM_LIQUIDITY_ANALYSIS",
                "category": "liquidity_analysis",
                "description": "ZXM流动性分析指标",
                "metrics": ["volatility", "volume_stability", "price_continuity", "liquidity_index"],
                "levels": ["excellent", "good", "fair", "poor"],
                "risk_levels": ["low", "medium", "high"],
                "thresholds": {
                    "excellent": 80,  # TODO: 将魔法数字提取到配置中
                    "good": 60,  # TODO: 将魔法数字提取到配置中
                    "fair": 40,  # TODO: 将魔法数字提取到配置中
                    "poor": 0,
                },
            }
        except Exception as e:
            logger.error(f"ZXM_LIQUIDITY_ANALYSIS get_patterns失败: {e}")
            return {}

    # ==================== BaseIndicator抽象方法实现 ====================

    def _calculate_baseindicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """BaseIndicator抽象方法实现：核心计算逻辑"""
        try:
            # 数据验证
            if data is None or data.empty:
                logger.warning("ZXM_LIQUIDITY_ANALYSIS: 输入数据为空")
                return pd.DataFrame()

            # 检查必需的列
            required_columns = ["close", "volume", "high", "low", "open"]
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                logger.error(f"ZXM_LIQUIDITY_ANALYSIS: 缺少必需的列 {missing_columns}")
                return pd.DataFrame()

            # 调用原有的calculate方法
            result_dict = self.calculate(data, **kwargs)

            # 将字典结果转换为DataFrame
            if isinstance(result_dict, dict) and result_dict:
                # 创建一个与输入数据长度相同的DataFrame
                df_result = data.copy()

                # 添加计算结果作为新列
                for key, value in result_dict.items():
                    if isinstance(value, (int, float)):
                        # 数值类型：在最后一行填入值，其他行为NaN
                        df_result[key] = np.nan
                        df_result.iloc[-1, df_result.columns.get_loc(key)] = value
                    else:
                        # 字符串类型：在最后一行填入值，其他行为空字符串
                        df_result[key] = ""
                        df_result.iloc[-1, df_result.columns.get_loc(key)] = str(value)

                return df_result
            else:
                # 如果计算失败，返回原始数据
                return data.copy()

        except Exception as e:
            logger.error(f"ZXM_LIQUIDITY_ANALYSIS _calculate_baseindicator失败: {e}")
            return pd.DataFrame(index=data.index)

    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """BaseIndicator抽象方法实现：计算原始评分"""
        try:
            result = self.calculate(data, **kwargs)

            if not result:
                return pd.Series(
                    [50.0], index=[data.index[-1]] if len(data) > 0 else [0]
                )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

            # 基于流动性指数计算评分
            liquidity_index = result.get(
                "liquidity_index", 60.0
            )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

            # 流动性指数本身就是0-100的评分
            final_score = max(0.0, min(100.0, liquidity_index))

            return pd.Series([final_score], index=[data.index[-1]] if len(data) > 0 else [0])

        except Exception as e:
            logger.error(f"ZXM_LIQUIDITY_ANALYSIS计算原始评分失败: {e}")
            return pd.Series(
                [50.0], index=[data.index[-1]] if len(data) > 0 else [0]
            )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

    def get_patterns_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """BaseIndicator抽象方法实现：获取技术形态"""
        try:
            result = self.calculate(data, **kwargs)

            # 创建形态DataFrame
            patterns_df = pd.DataFrame(index=data.index)

            if result:
                # 基于流动性等级识别形态
                liquidity_level = result.get("liquidity_level", "fair")
                liquidity_risk = result.get("liquidity_risk", "medium")
                liquidity_index = result.get(
                    "liquidity_index", 60.0
                )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

                # 优秀流动性形态
                if liquidity_level == "excellent":
                    patterns_df["EXCELLENT_LIQUIDITY"] = False
                    patterns_df.iloc[-1, patterns_df.columns.get_loc("EXCELLENT_LIQUIDITY")] = True

                # 良好流动性形态
                elif liquidity_level == "good":
                    patterns_df["GOOD_LIQUIDITY"] = False
                    patterns_df.iloc[-1, patterns_df.columns.get_loc("GOOD_LIQUIDITY")] = True

                # 一般流动性形态
                elif liquidity_level == "fair":
                    patterns_df["FAIR_LIQUIDITY"] = False
                    patterns_df.iloc[-1, patterns_df.columns.get_loc("FAIR_LIQUIDITY")] = True

                # 差流动性形态
                elif liquidity_level == "poor":
                    patterns_df["POOR_LIQUIDITY"] = False
                    patterns_df.iloc[-1, patterns_df.columns.get_loc("POOR_LIQUIDITY")] = True

                # 高风险流动性形态
                if liquidity_risk == "high":
                    patterns_df["HIGH_LIQUIDITY_RISK"] = False
                    patterns_df.iloc[-1, patterns_df.columns.get_loc("HIGH_LIQUIDITY_RISK")] = True

                # 低风险流动性形态
                elif liquidity_risk == "low":
                    patterns_df["LOW_LIQUIDITY_RISK"] = False
                    patterns_df.iloc[-1, patterns_df.columns.get_loc("LOW_LIQUIDITY_RISK")] = True

            return patterns_df

        except Exception as e:
            logger.error(f"ZXM_LIQUIDITY_ANALYSIS获取形态失败: {e}")
            return pd.DataFrame(index=data.index)

    def calculate_confidence_Indicator_Base_Indicator(
        self, score: pd.Series, patterns: pd.DataFrame, signals: dict
    ) -> float:
        """BaseIndicator抽象方法实现：计算置信度"""
        try:
            # 基础置信度
            base_confidence = 0.7  # TODO: 将魔法数字提取到配置中

            # 根据数据量调整置信度
            data_length = len(score)
            if data_length >= 252:  # 一年数据  # TODO: 将魔法数字提取到配置中
                data_confidence = 0.9  # TODO: 将魔法数字提取到配置中
            elif data_length >= 60:  # 两个月数据  # TODO: 将魔法数字提取到配置中
                data_confidence = 0.8  # TODO: 将魔法数字提取到配置中
            elif data_length >= 30:  # 一个月数据  # TODO: 将魔法数字提取到配置中
                data_confidence = 0.7  # TODO: 将魔法数字提取到配置中
            else:
                data_confidence = 0.5  # TODO: 将魔法数字提取到配置中

            # 根据形态数量调整置信度
            pattern_confidence = 0.7  # TODO: 将魔法数字提取到配置中
            if isinstance(patterns, pd.DataFrame) and not patterns.empty:
                pattern_count = patterns.sum().sum()
                if pattern_count > 0:
                    pattern_confidence = min(
                        0.9, 0.7 + pattern_count * 0.05
                    )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

            # 综合置信度
            final_confidence = (
                base_confidence + data_confidence + pattern_confidence
            ) / 3  # TODO: 将魔法数字提取到配置中

            return max(0.0, min(1.0, final_confidence))

        except Exception as e:
            logger.error(f"ZXM_LIQUIDITY_ANALYSIS计算置信度失败: {e}")
            return 0.5  # TODO: 将魔法数字提取到配置中

    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """BaseIndicator抽象方法实现：设置参数"""
        try:
            # 更新参数
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    logger.debug(f"ZXM_LIQUIDITY_ANALYSIS参数更新: {key} = {value}")

            # 重置结果，强制重新计算
            self._result = None

        except Exception as e:
            logger.error(f"ZXM_LIQUIDITY_ANALYSIS设置参数失败: {e}")
