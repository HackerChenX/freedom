from utils.container import container
from indicators.base_indicator import BaseIndicator

"""
通用多周期指标计算器
抽离公共的指标计算逻辑，支持周期+指标的维度分析
"""

from typing import Dict, List, Optional, Any, Union
import pandas as pd
from datetime import datetime

from utils.logger import get_logger
from utils.decorators import exception_handler, performance_monitor
from db.services.multi_period_data_service import MultiPeriodDataService, Period
from indicators.complete_indicator_registry import get_indicator, get_indicator_registry
from indicators.signal_method_adapter import get_unified_indicator_signal
from db.sql_manager import SQLManager, QueryType

logger = get_logger(__name__)


class UniversalMultiPeriodCalculator(BaseIndicator):
    """
    通用多周期指标计算器

    核心功能：
    - 支持全周期分析（15分钟、30分钟、60分钟、日线、周线、月线）
    - 抽离公共的指标计算逻辑
    - 周期+指标的二维分析矩阵
    - 跨周期信号一致性验证
    - 多周期信号聚合
    """

    def __init__(self):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        """初始化通用多周期指标计算器"""
        self.data_service = MultiPeriodDataService()
        self.indicator_registry = get_indicator_registry()

        # 全周期支持（默认分析所有周期）
        self.all_periods = [
            Period.MIN_15,  # 15分钟
            Period.MIN_30,  # 30分钟
            Period.MIN_60,  # 60分钟
            Period.DAILY,  # 日线
            Period.WEEKLY,  # 周线
            Period.MONTHLY,  # 月线
        ]

        # 周期权重配置（用于信号聚合）
        self.period_weights = {
            Period.MIN_15: 0.05,  # 15分钟线权重最低  # TODO: 将魔法数字提取到配置中
            Period.MIN_30: 0.10,  # 30分钟线
            Period.MIN_60: 0.15,  # 60分钟线  # TODO: 将魔法数字提取到配置中
            Period.DAILY: 0.40,  # 日线权重最高  # TODO: 将魔法数字提取到配置中
            Period.WEEKLY: 0.25,  # 周线权重较高  # TODO: 将魔法数字提取到配置中
            Period.MONTHLY: 0.05,  # 月线权重较低  # TODO: 将魔法数字提取到配置中
        }

        logger.info("通用多周期指标计算器初始化完成，支持全周期分析")

    @exception_handler(reraise=True)
    @performance_monitor(threshold=15.0)  # TODO: 将魔法数字提取到配置中
    def calculate_multi_period_indicators(
        self,
        stock_code: str,
        target_date: str,
        indicator_names: Optional[List[str]] = None,
        periods: Optional[List[Period]] = None,
    ) -> Dict[str, Any]:
        """
        计算多周期指标（全周期分析）

        Args:
            stock_code: 股票代码
            target_date: 目标分析日期
            indicator_names: 指标名称列表，None表示使用所有指标
            periods: 周期列表，None表示使用全部周期

        Returns:
            Dict[str, Any]: 多周期指标分析结果
        """
        # 默认使用全周期分析
        if periods is None:
            periods = self.all_periods
            logger.info("使用全周期分析：15分钟、30分钟、60分钟、日线、周线、月线")

        # 默认使用所有指标
        if indicator_names is None:
            all_indicators = self.indicator_registry.get_all_indicators()
            indicator_names = list(all_indicators.keys())
            logger.info(f"使用全部{len(indicator_names)}个指标进行多周期分析")

        # 获取多周期数据（不指定periods，获取全部周期）
        logger.info(f"开始获取{stock_code}的全周期数据")
        multi_period_data = self.data_service.get_stock_multi_period_data(
            stock_code=stock_code,
            target_date=target_date,
            periods=None,  # 获取全部周期数据
            lookback_days=None,  # 使用默认的充足数据量
        )

        # 构建周期+指标的二维分析矩阵
        analysis_matrix = self._build_analysis_matrix(multi_period_data, indicator_names, periods)

        # 进行跨周期信号一致性分析
        consistency_analysis = self._analyze_cross_period_consistency(analysis_matrix, indicator_names, periods)

        # 生成多周期聚合信号
        aggregated_signals = self._generate_aggregated_signals(analysis_matrix, indicator_names, periods)

        # 计算综合评分
        overall_score = self._calculate_overall_score(analysis_matrix, consistency_analysis, aggregated_signals)

        return {
            "stock_code": stock_code,
            "target_date": target_date,
            "analysis_matrix": analysis_matrix,
            "consistency_analysis": consistency_analysis,
            "aggregated_signals": aggregated_signals,
            "overall_score": overall_score,
            "periods_analyzed": [p.value for p in periods],
            "indicators_analyzed": indicator_names,
            "data_quality": self._assess_data_quality(multi_period_data, periods),
        }

    def _build_analysis_matrix(
        self, multi_period_data: Dict[Period, pd.DataFrame], indicator_names: List[str], periods: List[Period]
    ) -> Dict[str, Dict[str, Any]]:
        """
        构建周期+指标的二维分析矩阵

        Args:
            multi_period_data: 多周期数据
            indicator_names: 指标名称列表
            periods: 周期列表

        Returns:
            Dict: 分析矩阵 {period_value: {indicator_name: result}}
        """
        analysis_matrix = {}

        for period in periods:
            period_data = multi_period_data.get(period, pd.DataFrame())
            period_key = period.value
            analysis_matrix[period_key] = {}

            if period_data.empty:
                logger.warning(f"{period_key}周期数据为空，跳过分析")
                continue

            logger.info(f"开始分析{period_key}周期，数据点数: {len(period_data)}")

            # 为每个指标计算该周期的结果
            for indicator_name in indicator_names:
                try:
                    indicator_result = self._calculate_single_indicator(period_data, indicator_name, period)
                    analysis_matrix[period_key][indicator_name] = indicator_result

                except Exception as e:
                    logger.warning(f"{period_key}周期{indicator_name}指标计算失败: {e}")
                    analysis_matrix[period_key][indicator_name] = {"signal": "ERROR", "strength": 0.0, "error": str(e)}

        return analysis_matrix

    def _calculate_single_indicator(self, data: pd.DataFrame, indicator_name: str, period: Period) -> Dict[str, Any]:
        """
        计算单个指标在特定周期的结果

        Args:
            data: 股票数据
            indicator_name: 指标名称
            period: 周期

        Returns:
            Dict: 指标计算结果
        """
        try:
            # 获取指标实例
            indicator = get_indicator(indicator_name)
            if not indicator:
                return {"signal": "UNAVAILABLE", "strength": 0.0, "error": f"指标{indicator_name}不可用"}

            # 计算指标
            calc_result = indicator.calculate(data)

            # 获取统一信号
            signal_result = get_unified_indicator_signal(indicator, data, indicator_name)

            # 构建结果
            result = {
                "signal": signal_result.get("signal", "UNKNOWN"),
                "strength": signal_result.get("strength", 0.0),
                "value": signal_result.get("value", None),
                "method_used": signal_result.get("method_used", "unknown"),
                "period": period.value,
                "data_points": len(data),
                "calculation_success": True,
            }

            # 特殊处理：检测金叉死叉等形态
            if indicator_name in ["KDJ", "MACD", "STOCH"]:
                cross_analysis = self._detect_cross_patterns(calc_result, indicator_name)
                result.update(cross_analysis)

            return result

        except Exception as e:
            logger.error(f"计算{indicator_name}指标失败: {e}")
            return {"signal": "ERROR", "strength": 0.0, "error": str(e), "calculation_success": False}

    def _detect_cross_patterns(self, calc_result: Any, indicator_name: str) -> Dict[str, Any]:
        """检测金叉死叉等交叉形态"""
        cross_info = {}

        try:
            if indicator_name == "MACD" and isinstance(calc_result, dict):
                # MACD金叉死叉检测
                if "macd" in calc_result and "signal" in calc_result:
                    macd_line = calc_result["macd"]
                    signal_line = calc_result["signal"]

                    if len(macd_line) >= 2 and len(signal_line) >= 2:
                        # 检测最近的交叉
                        if macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2]:
                            cross_info["cross_pattern"] = "GOLDEN_CROSS"
                            cross_info["cross_strength"] = 0.8  # TODO: 将魔法数字提取到配置中
                        elif macd_line.iloc[-1] < signal_line.iloc[-1] and macd_line.iloc[-2] >= signal_line.iloc[-2]:
                            cross_info["cross_pattern"] = "DEATH_CROSS"
                            cross_info["cross_strength"] = -0.8  # TODO: 将魔法数字提取到配置中

            elif indicator_name == "KDJ" and isinstance(calc_result, dict):
                # KDJ金叉死叉检测
                if "k" in calc_result and "d" in calc_result:
                    k_line = calc_result["k"]
                    d_line = calc_result["d"]

                    if len(k_line) >= 2 and len(d_line) >= 2:
                        if k_line.iloc[-1] > d_line.iloc[-1] and k_line.iloc[-2] <= d_line.iloc[-2]:
                            cross_info["cross_pattern"] = "GOLDEN_CROSS"
                            cross_info["cross_strength"] = 0.7  # TODO: 将魔法数字提取到配置中
                        elif k_line.iloc[-1] < d_line.iloc[-1] and k_line.iloc[-2] >= d_line.iloc[-2]:
                            cross_info["cross_pattern"] = "DEATH_CROSS"
                            cross_info["cross_strength"] = -0.7  # TODO: 将魔法数字提取到配置中

        except Exception as e:
            logger.debug(f"检测{indicator_name}交叉形态失败: {e}")

        return cross_info

    def _analyze_cross_period_consistency(
        self, analysis_matrix: Dict[str, Dict[str, Any]], indicator_names: List[str], periods: List[Period]
    ) -> Dict[str, Any]:
        """分析跨周期信号一致性"""
        consistency_analysis = {
            "consistent_indicators": [],
            "inconsistent_indicators": [],
            "period_agreement": {},
            "overall_consistency_score": 0.0,
        }

        for indicator_name in indicator_names:
            signals = []
            for period in periods:
                period_key = period.value
                if period_key in analysis_matrix and indicator_name in analysis_matrix[period_key]:
                    signal = analysis_matrix[period_key][indicator_name].get("signal", "UNKNOWN")
                    signals.append(signal)

            # 计算一致性
            if signals:
                unique_signals = set(signals)
                if len(unique_signals) == 1 and "UNKNOWN" not in unique_signals:
                    consistency_analysis["consistent_indicators"].append(indicator_name)
                else:
                    consistency_analysis["inconsistent_indicators"].append(indicator_name)

        # 计算整体一致性评分
        total_indicators = len(indicator_names)
        consistent_count = len(consistency_analysis["consistent_indicators"])
        if total_indicators > 0:
            consistency_analysis["overall_consistency_score"] = consistent_count / total_indicators

        return consistency_analysis

    def _generate_aggregated_signals(
        self, analysis_matrix: Dict[str, Dict[str, Any]], indicator_names: List[str], periods: List[Period]
    ) -> Dict[str, Any]:
        """生成多周期聚合信号"""
        aggregated_signals = {}

        for indicator_name in indicator_names:
            weighted_score = 0.0
            total_weight = 0.0

            for period in periods:
                period_key = period.value
                if period_key in analysis_matrix and indicator_name in analysis_matrix[period_key]:

                    indicator_result = analysis_matrix[period_key][indicator_name]
                    signal = indicator_result.get("signal", "UNKNOWN")
                    strength = indicator_result.get("strength", 0.0)

                    # 转换信号为数值
                    signal_value = self._signal_to_value(signal)

                    # 加权计算
                    weight = self.period_weights.get(period, 0.0)
                    weighted_score += signal_value * strength * weight
                    total_weight += weight

            # 计算聚合信号
            if total_weight > 0:
                final_score = weighted_score / total_weight
                aggregated_signal = self._value_to_signal(final_score)

                aggregated_signals[indicator_name] = {
                    "aggregated_signal": aggregated_signal,
                    "aggregated_strength": abs(final_score),
                    "weighted_score": final_score,
                }

        return aggregated_signals

    def _signal_to_value(self, signal: str) -> float:
        """将信号转换为数值"""
        signal_map = {
            "BUY": 1.0,
            "STRONG_BUY": 1.0,
            "SELL": -1.0,
            "STRONG_SELL": -1.0,
            "HOLD": 0.0,
            "NEUTRAL": 0.0,
            "UNKNOWN": 0.0,
            "ERROR": 0.0,
        }
        return signal_map.get(signal, 0.0)

    def _value_to_signal(self, value: float) -> str:
        """将数值转换为信号"""
        if value > 0.5:  # TODO: 将魔法数字提取到配置中
            return "BUY"
        elif value < -0.5:  # TODO: 将魔法数字提取到配置中
            return "SELL"
        else:
            return "HOLD"

    def _calculate_overall_score(
        self,
        analysis_matrix: Dict[str, Dict[str, Any]],
        consistency_analysis: Dict[str, Any],
        aggregated_signals: Dict[str, Any],
    ) -> float:
        """计算综合评分"""
        try:
            # 基于聚合信号计算评分
            buy_signals = sum(
                1 for signal_info in aggregated_signals.values() if signal_info.get("aggregated_signal") == "BUY"
            )
            sell_signals = sum(
                1 for signal_info in aggregated_signals.values() if signal_info.get("aggregated_signal") == "SELL"
            )
            total_signals = len(aggregated_signals)

            if total_signals == 0:
                return 50.0  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

            # 基础评分
            base_score = (
                50.0 + (buy_signals - sell_signals) / total_signals * 30.0
            )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

            # 一致性加权
            consistency_bonus = (
                consistency_analysis.get("overall_consistency_score", 0.0) * 20.0
            )  # TODO: 将魔法数字提取到配置中

            final_score = min(max(base_score + consistency_bonus, 0.0), 100.0)
            return final_score

        except Exception as e:
            logger.error(f"计算综合评分失败: {e}")
            return 50.0  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

    def _assess_data_quality(
        self, multi_period_data: Dict[Period, pd.DataFrame], periods: List[Period]
    ) -> Dict[str, Any]:
        """评估数据质量"""
        quality_assessment = {
            "periods_with_data": 0,
            "total_periods": len(periods),
            "data_completeness": {},
            "overall_quality": "GOOD",
        }

        for period in periods:
            period_data = multi_period_data.get(period, pd.DataFrame())
            period_key = period.value

            if not period_data.empty:
                quality_assessment["periods_with_data"] += 1
                quality_assessment["data_completeness"][period_key] = len(period_data)
            else:
                quality_assessment["data_completeness"][period_key] = 0

        # 评估整体质量
        completeness_ratio = quality_assessment["periods_with_data"] / quality_assessment["total_periods"]
        if completeness_ratio >= 0.8:  # TODO: 将魔法数字提取到配置中
            quality_assessment["overall_quality"] = "EXCELLENT"
        elif completeness_ratio >= 0.6:  # TODO: 将魔法数字提取到配置中
            quality_assessment["overall_quality"] = "GOOD"
        elif completeness_ratio >= 0.4:  # TODO: 将魔法数字提取到配置中
            quality_assessment["overall_quality"] = "FAIR"
        else:
            quality_assessment["overall_quality"] = "POOR"

        return quality_assessment
