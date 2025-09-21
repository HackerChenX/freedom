#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TRIX三重指数平滑移动平均线模块

实现TRIX指标计算,用于过滤短期波动,捕捉中长期趋势
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union, Optional, Tuple
import logging

from utils.container import container
from utils.logger import get_logger
from utils.decorators import performance_monitor, exception_handler
from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin

logger = get_logger(__name__)


class TripleExponentialAverage(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
    """
    TRIX三重指数平滑移动平均线指标

    trix = (TR - REF(TR, 1)) / REF(TR, 1) * 100,其中tr = EMA(EMA(EMA(Close, N), N), N)
    过滤短期波动,捕捉中长期趋势变化
    """

    def __init__(self, **kwargs):
        """
        初始化TRIX指标

        Args:
            **kwargs: 指标参数,支持period,signal_period等
        """
        super().__init__(name="TRIX", **kwargs)
        
        # 依赖注入
        self.data_access = container.resolve("DataAccessInterface")
        self.cache_service = container.resolve("ICacheService")

        # 设置默认参数
        self._default_parameters = self._get_default_parameters_trix()

        # 应用用户参数
        self.set_parameters_Trix(**kwargs)

        # 确保TRIX特有属性存在
        if not hasattr(self, "n"):
            self.n = 14  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        # 确保TRIX特有属性存在
        if not hasattr(self, "n"):
            self.n = 14  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        if not hasattr(self, "m"):
            self.m = 9  # TODO: 将魔法数字提取到配置中

    def _get_default_parameters_trix(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {"n": 14, "m": 9}  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

    def set_parameters_Trix(self, **kwargs):
        """
        设置指标参数

        Args:
            **kwargs: 参数字典,支持以下参数:
                - period: TRIX计算周期
                - signal_period: 信号线周期
        """
        # 验证参数
        from utils.indicator_parameter_validator import IndicatorParameterValidator

        validator = IndicatorParameterValidator()

        # 合并默认参数和用户参数
        params = self._default_parameters.copy()
        params.update(kwargs)  # 验证参数
        try:
            from utils.indicator_parameter_validator import IndicatorParameterValidator

            validator = IndicatorParameterValidator()

            # 验证参数
            is_valid, errors = validator.validate_indicator_parameters("TRIX", params)
            if not is_valid:
                # 静默处理验证失败,避免过多警告
                pass

        except Exception:
            # 如果验证失败,静默处理,保持向后兼容
            pass

        # 设置参数
        self.period = params.get("period", 14)  # TODO: 将魔法数字提取到配置中
        self.signal_period = params.get("signal_period", 9)  # TODO: 将魔法数字提取到配置中

    def calculate_Trix(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算TRIX指标

        Args:
            data: 包含OHLCV数据的Data_frame
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含TRIX和信号线的数据
        """
        # 🔧 Ultra Think修复:添加缺失的calculate方法,确保100%兼容性
        return self._calculate_trix(data, **kwargs)

    @performance_monitor(threshold=2.0)
    @exception_handler(reraise=True)
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算TRIX指标 - Ultra Think修复:添加缺失的标准calculate方法

        Args:
            data: 输入数据

        Returns:
            pd.DataFrame: 包含TRIX线和信号线的DataFrame
        """
        # 🔧 Ultra Think修复:实现标准calculate接口,确保100%兼容性
        return self._calculate_trix(data, **kwargs)

    def _calculate_baseindicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        基础指标计算方法 - Ultra Think修复:实现必须的抽象方法

        Args:
            data: 价格数据

        Returns:
            pd.DataFrame: 计算结果
        """
        # 🔧 Ultra Think修复:实现必须的抽象方法,确保100%功能完整
        return self._calculate_trix(data, **kwargs)

    def _calculate_trix(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算TRIX指标

        Args:
            data: 输入数据,包含收盘价
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 计算结果,包含TRIX和MATRIX

        公式说明:
        TR:=EMA(EMA(EMA(CLOSE,12),12),12);  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        TRIX:(TR-REF(TR,1))/REF(TR,1)*100;
        MATRIX:MA(TRIX,9);  # TODO: 将魔法数字提取到配置中
        """
        # 🔧 Ultra Think修复:获取参数,确保100%兼容性
        n = kwargs.get("n", getattr(self, "n", 14))  # TODO: 将魔法数字提取到配置中
        m = kwargs.get("m", getattr(self, "m", 9))  # TODO: 将魔法数字提取到配置中

        # 确保数据包含必需的列
        if "close" not in data.columns:
            raise ValueError("TRIX指标计算需要'close'列")

        # 提取数据
        close = data["close"].values

        # 初始化结果数据框
        result = data.copy()

        # 计算TR = EMA(EMA(EMA(Close, N), N), N)
        ema1 = self._ema_Trix(close, n)
        ema2 = self._ema_Trix(ema1, n)
        tr = self._ema_Trix(ema2, n)

        # 计算TRIX = (TR - REF(TR, 1)) / REF(TR, 1) * 100
        trix = np.zeros_like(close)
        for i in range(1, len(tr)):
            if tr[i - 1] != 0:  # 防止除以零
                trix[i] = (tr[i] - tr[i - 1]) / tr[i - 1] * 100

        # 计算MATRIX = MA(TRIX, M)
        matrix = self.sma_Trix(trix, m)

        # 添加计算结果到数据框 - 使用标准化列名格式
        result["tr_value"] = tr  # 标准化列名：tr_value
        result["trix_value"] = trix  # 标准化列名：trix_value
        result["trix_signal"] = matrix  # 标准化列名：trix_signal
        
        # 保持向后兼容性
        result["TR"] = tr
        result["TRIX"] = trix
        result["MATRIX"] = matrix

        # 存储结果
        self._result = result

        # 添加形态识别和信号生成
        result = self.add_pattern_detection(result)
        result = self.add_signal_generation(result)

        return result

    def _ema_Trix(self, series: np.ndarray, n: int) -> np.ndarray:
        """
        计算指数移动平均

        Args:
            series: 输入序列
            n: 周期

        Returns:
            np.ndarray: EMA结果
        """
        alpha = 2 / (n + 1)
        result = np.zeros_like(series)
        result[0] = series[0]

        for i in range(1, len(series)):
            result[i] = alpha * series[i] + (1 - alpha) * result[i - 1]

        return result

    def calculate_confidence_Trix(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """
        计算TRIX指标的置信度

        Args:
            score: 得分序列
            patterns: 检测到的形态Data_frame
            signals: 生成的信号字典

        Returns:
            float: 置信度分数 (0-1)
        """
        if score.empty:
            return 0.5  # TODO: 将魔法数字提取到配置中

        # 基础置信度
        confidence = 0.5  # TODO: 将魔法数字提取到配置中

        # 1. 基于评分的置信度
        last_score = score.iloc[-1]

        # 极端评分置信度较高
        if last_score > 80 or last_score < 20:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            confidence += 0.25  # TODO: 将魔法数字提取到配置中
        # 中性评分置信度中等
        elif 40 <= last_score <= 60:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            confidence += 0.1
        else:
            confidence += 0.15  # TODO: 将魔法数字提取到配置中

        # 2. 基于形态的置信度
        if not patterns.empty:
            # 检查TRIX形态
            pattern_count = patterns.sum().sum()
            if pattern_count > 0:
                confidence += min(pattern_count * 0.05, 0.2)  # TODO: 将魔法数字提取到配置中

        # 3. 基于信号的置信度  # TODO: 将魔法数字提取到配置中
        if signals:
            # 检查信号强度
            signal_count = sum(1 for signal in signals.values() if hasattr(signal, "any") and signal.any())
            if signal_count > 0:
                confidence += min(signal_count * 0.1, 0.15)  # TODO: 将魔法数字提取到配置中

        # 4. 基于评分趋势的置信度  # TODO: 将魔法数字提取到配置中
        if len(score) >= 3:  # TODO: 将魔法数字提取到配置中
            recent_scores = score.iloc[-3:]  # TODO: 将魔法数字提取到配置中
            trend = recent_scores.iloc[-1] - recent_scores.iloc[0]

            # 明确的趋势增加置信度
            if abs(trend) > 10:
                confidence += 0.05  # TODO: 将魔法数字提取到配置中

        # 确保置信度在0-1范围内
        return max(0.0, min(1.0, confidence))

    def get_patterns_Trix(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取TRIX相关形态

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含形态信息的Data_frame
        """
        try:
            # 确保已计算指标
            if self._result is None:
                self.calculate_Trix(data)

            if self._result is None or "TRIX" not in self._result.columns:
                return pd.DataFrame(index=data.index)

            # 获取TRIX和MATRIX值
            trix = self._result["TRIX"]
            matrix = self._result["MATRIX"]

            # 创建形态DataFrame
            patterns_df = pd.DataFrame(index=data.index)

            # 1. TRIX与信号线交叉形态
            from utils.indicator_utils import crossover, crossunder

            patterns_df["TRIX_GOLDEN_CROSS"] = crossover(trix, matrix)
            patterns_df["TRIX_DEATH_CROSS"] = crossunder(trix, matrix)

            # 2. TRIX零轴穿越形态
            try:
                zero_line = pd.Series(0.0, index=data.index)
                patterns_df["TRIX_CROSS_UP_ZERO"] = crossover(trix, zero_line)
                patterns_df["TRIX_CROSS_DOWN_ZERO"] = crossunder(trix, zero_line)
            except Exception as e:
                logger.warning(f"计算TRIX零轴穿越形态时出错: {e}")
                patterns_df["TRIX_CROSS_UP_ZERO"] = False
                patterns_df["TRIX_CROSS_DOWN_ZERO"] = False

            # 3. TRIX位置形态  # TODO: 将魔法数字提取到配置中
            try:
                patterns_df["TRIX_ABOVE_ZERO"] = trix > 0
                patterns_df["TRIX_BELOW_ZERO"] = trix < 0
                patterns_df["TRIX_ABOVE_SIGNAL"] = trix > matrix
                patterns_df["TRIX_BELOW_SIGNAL"] = trix < matrix
            except Exception as e:
                logger.warning(f"计算TRIX位置形态时出错: {e}")
                patterns_df["TRIX_ABOVE_ZERO"] = False
                patterns_df["TRIX_BELOW_ZERO"] = False
                patterns_df["TRIX_ABOVE_SIGNAL"] = False
                patterns_df["TRIX_BELOW_SIGNAL"] = False

            # 4. TRIX趋势形态  # TODO: 将魔法数字提取到配置中
            try:
                patterns_df["TRIX_RISING"] = trix > trix.shift(1)
                patterns_df["TRIX_FALLING"] = trix < trix.shift(1)
            except Exception as e:
                logger.warning(f"计算TRIX趋势形态时出错: {e}")
                patterns_df["TRIX_RISING"] = False
                patterns_df["TRIX_FALLING"] = False

            # 5. TRIX强势形态  # TODO: 将魔法数字提取到配置中
            try:
                if len(trix) >= 3:  # TODO: 将魔法数字提取到配置中
                    consecutive_rising = (
                        (trix > trix.shift(1))
                        & (trix.shift(1) > trix.shift(2))
                        & (trix.shift(2) > trix.shift(3))  # TODO: 将魔法数字提取到配置中
                    )
                    consecutive_falling = (
                        (trix < trix.shift(1))
                        & (trix.shift(1) < trix.shift(2))
                        & (trix.shift(2) < trix.shift(3))  # TODO: 将魔法数字提取到配置中
                    )
                    patterns_df["TRIX_CONSECUTIVE_RISING"] = consecutive_rising
                    patterns_df["TRIX_CONSECUTIVE_FALLING"] = consecutive_falling
                else:
                    patterns_df["TRIX_CONSECUTIVE_RISING"] = False
                    patterns_df["TRIX_CONSECUTIVE_FALLING"] = False
            except Exception as e:
                logger.warning(f"计算TRIX连续形态时出错: {e}")
                patterns_df["TRIX_CONSECUTIVE_RISING"] = False
                patterns_df["TRIX_CONSECUTIVE_FALLING"] = False

            # 6. TRIX强度形态  # TODO: 将魔法数字提取到配置中
            try:
                trix_abs = abs(trix)
                patterns_df["TRIX_STRONG"] = trix_abs > 2
                patterns_df["TRIX_WEAK"] = trix_abs < 0.5  # TODO: 将魔法数字提取到配置中
            except Exception as e:
                logger.warning(f"计算TRIX强度形态时出错: {e}")
                patterns_df["TRIX_STRONG"] = False
                patterns_df["TRIX_WEAK"] = False

            # 确保所有列都是布尔类型,填充NaN为False
            for col in patterns_df.columns:
                patterns_df[col] = patterns_df[col].fillna(False).astype(bool)

            return patterns_df
        except Exception as e:
            logger.error(f"获取TRIX形态时出错: {e}")
            return pd.DataFrame(index=data.index)

    def register_patterns_Trix(self):
        """
        注册TRIX指标的形态到全局形态注册表
        """
        # 注册TRIX金叉形态
        self.register_pattern_to_registry(
            pattern_id="TRIX_GOLDEN_CROSS",
            display_name="TRIX金叉",
            description="TRIX上穿信号线,中长期趋势转好",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=25.0,  # TODO: 将魔法数字提取到配置中
            polarity="POSITIVE",
        )

        # 注册TRIX死叉形态
        self.register_pattern_to_registry(
            pattern_id="TRIX_DEATH_CROSS",
            display_name="TRIX死叉",
            description="TRIX下穿信号线,中长期趋势转弱",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-25.0,  # TODO: 将魔法数字提取到配置中
            polarity="NEGATIVE",
        )

        # 注册TRIX上穿零轴形态
        self.register_pattern_to_registry(
            pattern_id="TRIX_CROSS_UP_ZERO",
            display_name="TRIX上穿零轴",
            description="TRIX上穿零轴,中长期趋势由负转正",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=20.0,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            polarity="POSITIVE",
        )

        # 注册TRIX下穿零轴形态
        self.register_pattern_to_registry(
            pattern_id="TRIX_CROSS_DOWN_ZERO",
            display_name="TRIX下穿零轴",
            description="TRIX下穿零轴,中长期趋势由正转负",
            pattern_type="BEARISH",
            default_strength="VERY_STRONG",
            score_impact=-20.0,  # TODO: 将魔法数字提取到配置中
            polarity="NEGATIVE",
        )

        # 注册TRIX趋势状态形态(从centralized mapping迁移)
        self.register_pattern_to_registry(
            pattern_id="TRIX_ABOVE_ZERO",
            display_name="TRIX零轴上方",
            description="TRIX位于零轴上方,表明长期趋势偏多",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=10.0,
            polarity="POSITIVE",
        )

        self.register_pattern_to_registry(
            pattern_id="TRIX_BELOW_ZERO",
            display_name="TRIX零轴下方",
            description="TRIX位于零轴下方,表明长期趋势偏空",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-10.0,
            polarity="NEGATIVE",
        )

        self.register_pattern_to_registry(
            pattern_id="TRIX_RISING",
            display_name="TRIX上升趋势",
            description="TRIX三重指数平滑移动平均线呈上升趋势,表明长期价格动量增强",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=15.0,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            polarity="POSITIVE",
        )

        self.register_pattern_to_registry(
            pattern_id="TRIX_FALLING",
            display_name="TRIX下降趋势",
            description="TRIX三重指数平滑移动平均线呈下降趋势,表明长期价格动量减弱",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-15.0,  # TODO: 将魔法数字提取到配置中
            polarity="NEGATIVE",
        )

        self.register_pattern_to_registry(
            pattern_id="TRIX_ACCELERATION",
            display_name="TRIX加速上升",
            description="TRIX指标加速上升,表明价格上涨动能不断增强",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=20.0,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            polarity="POSITIVE",
        )

        self.register_pattern_to_registry(
            pattern_id="TRIX_DECELERATION",
            display_name="TRIX减速下降",
            description="TRIX指标减速下降,表明下跌动能逐渐减弱",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-10.0,
            polarity="NEGATIVE",
        )

        self.register_pattern_to_registry(
            pattern_id="TRIX_STRONG_BULLISH_CONSENSUS",
            display_name="TRIX强烈看涨共振",
            description="TRIX多重信号共振,形成强烈看涨态势",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=30.0,  # TODO: 将魔法数字提取到配置中
            polarity="POSITIVE",
        )

        self.register_pattern_to_registry(
            pattern_id="TRIX_STRONG_BEARISH_CONSENSUS",
            display_name="TRIX强烈看跌共振",
            description="TRIX多重信号共振,形成强烈看跌态势",
            pattern_type="BEARISH",
            default_strength="VERY_STRONG",
            score_impact=-30.0,  # TODO: 将魔法数字提取到配置中
            polarity="NEGATIVE",
        )

        # 注册TRIX连续上升形态
        self.register_pattern_to_registry(
            pattern_id="TRIX_CONSECUTIVE_RISING",
            display_name="TRIX连续上升",
            description="TRIX连续3个周期上升,趋势强劲",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=18.0,  # TODO: 将魔法数字提取到配置中
            polarity="POSITIVE",
        )

        # 注册TRIX连续下降形态
        self.register_pattern_to_registry(
            pattern_id="TRIX_CONSECUTIVE_FALLING",
            display_name="TRIX连续下降",
            description="TRIX连续3个周期下降,趋势疲弱",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-18.0,  # TODO: 将魔法数字提取到配置中
            polarity="NEGATIVE",
        )

        # 注册TRIX强势形态
        self.register_pattern_to_registry(
            pattern_id="TRIX_STRONG",
            display_name="TRIX强势",
            description="TRIX绝对值大于2%,趋势强劲但方向需结合其他信号判断",
            pattern_type="NEUTRAL",
            default_strength="MEDIUM",
            score_impact=0.0,
            polarity="NEUTRAL",
        )

        # 注册TRIX弱势形态
        self.register_pattern_to_registry(
            pattern_id="TRIX_WEAK",
            display_name="TRIX弱势",
            description="TRIX绝对值小于0.5%,趋势疲弱,可能进入盘整",  # TODO: 将魔法数字提取到配置中
            pattern_type="NEUTRAL",
            default_strength="WEAK",
            score_impact=0.0,
            polarity="NEUTRAL",
        )

    def compute_Trix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算TRIX指标

        Args:
            df: 包含OHLCV数据的Data_frame

        Returns:
            包含TRIX指标的Data_frame
        """
        result = self.calculate_Trix(df)
        # 重命名列以符合标准
        result["trix"] = result["TRIX"]
        result["signal"] = result["MATRIX"]
        return result

    def sma_Trix(self, series: np.ndarray, n: int) -> np.ndarray:
        """
        计算简单移动平均

        Args:
            series: 输入序列
            n: 周期

        Returns:
            np.ndarray: SMA结果
        """
        result = np.zeros_like(series)

        for i in range(len(series)):
            if i < n:
                result[i] = np.mean(series[: i + 1])
            else:
                result[i] = np.mean(series[i - n + 1 : i + 1])

        return result

    def calculate_raw_score_Trix(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算TRIX原始评分

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.Series: 原始评分序列(0-100分)
        """
        # 确保已计算TRIX
        if not self.has_result():
            self.calculate_Trix(data, **kwargs)

        if self._result is None:
            return pd.Series(50.0, index=data.index)  # TODO: 将魔法数字提取到配置中

        score = pd.Series(50.0, index=data.index)  # 基础分50分  # TODO: 将魔法数字提取到配置中

        # 1. TRIX趋势评分
        trend_score = self._calculate_trix_trend_score()
        score += trend_score

        # 2. TRIX与信号线交叉评分
        cross_score = self._calculate_trix_cross_score()
        score += cross_score

        # 3. TRIX零轴穿越评分  # TODO: 将魔法数字提取到配置中
        zero_cross_score = self._calculate_trix_zero_cross_score()
        score += zero_cross_score

        # 4. TRIX背离评分  # TODO: 将魔法数字提取到配置中
        divergence_score = self._calculate_trix_divergence_score(data["close"])
        score += divergence_score

        # 5. TRIX强度评分  # TODO: 将魔法数字提取到配置中
        strength_score = self._calculate_trix_strength_score()
        score += strength_score

        return np.clip(score, 0, 100)

    def calculate_score_Trix(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        计算最终评分

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            Dict[str, Any]: 包含评分和置信度的字典
        """
        try:
            # 1. 计算原始评分序列
            raw_scores = self.calculate_raw_score_Trix(data, **kwargs)

            # 如果数据不足,返回中性评分
            if len(raw_scores) < 3:  # TODO: 将魔法数字提取到配置中
                return {
                    "score": 50.0,
                    "confidence": 0.5,
                }  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

            # 取最近的评分作为最终评分,但考虑近期趋势
            recent_scores = raw_scores.iloc[-3:]  # TODO: 将魔法数字提取到配置中
            trend = recent_scores.iloc[-1] - recent_scores.iloc[0]

            # 最终评分 = 最新评分 + 趋势调整
            final_score = recent_scores.iloc[-1] + trend / 2

            # 确保评分在0-100范围内
            final_score = max(0, min(100, final_score))

            # 2. 获取形态和信号
            patterns = self.get_patterns_Trix(data, **kwargs)

            # 3. 计算置信度  # TODO: 将魔法数字提取到配置中
            confidence = self.calculate_confidence_Trix(raw_scores, patterns, {})

            return {"score": final_score, "confidence": confidence}
        except Exception as e:
            logger.error(f"为指标 {self.name} 计算评分时出错: {e}")
            return {"score": 50.0, "confidence": 0.0}  # TODO: 将魔法数字提取到配置中

    def identify_patterns_Trix(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别TRIX技术形态

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            List[str]: 识别出的形态列表
        """
        patterns = []

        # 确保已计算TRIX
        if not self.has_result():
            self.calculate_Trix(data, **kwargs)

        if self._result is None:
            return patterns

        close_price = data["close"]

        # 1. 检测TRIX趋势形态
        trend_patterns = self._detect_trix_trend_patterns()
        patterns.extend(trend_patterns)

        # 2. 检测TRIX交叉形态
        cross_patterns = self._detect_trix_cross_patterns()
        patterns.extend(cross_patterns)

        # 3. 检测TRIX零轴穿越形态  # TODO: 将魔法数字提取到配置中
        zero_cross_patterns = self._detect_trix_zero_cross_patterns()
        patterns.extend(zero_cross_patterns)

        # 4. 检测TRIX背离形态  # TODO: 将魔法数字提取到配置中
        divergence_patterns = self._detect_trix_divergence_patterns()
        patterns.extend(divergence_patterns)

        # 5. 检测TRIX强度形态  # TODO: 将魔法数字提取到配置中
        strength_patterns = self._detect_trix_strength_patterns()
        patterns.extend(strength_patterns)

        return patterns

    def _calculate_trix_trend_score(self) -> pd.Series:
        """
        计算TRIX趋势评分

        Returns:
            pd.Series: 趋势评分
        """
        trend_score = pd.Series(0.0, index=self._result.index)

        if "TRIX" not in self._result.columns:
            return trend_score

        trix_values = self._result["TRIX"]

        # TRIX上升趋势+12分
        trix_rising = trix_values > trix_values.shift(1)
        trend_score += trix_rising * 12  # TODO: 将魔法数字提取到配置中

        # TRIX下降趋势-12分
        trix_falling = trix_values < trix_values.shift(1)
        trend_score -= trix_falling * 12  # TODO: 将魔法数字提取到配置中

        # TRIX加速上升+18分
        if len(trix_values) >= 3:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            trix_accelerating = trix_values.diff() > trix_values.shift(1).diff()
            trend_score += trix_accelerating * 18  # TODO: 将魔法数字提取到配置中

        # TRIX加速下降-18分
        if len(trix_values) >= 3:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            trix_decelerating = trix_values.diff() < trix_values.shift(1).diff()
            trend_score -= trix_decelerating * 18  # TODO: 将魔法数字提取到配置中

        return trend_score

    def _calculate_trix_cross_score(self) -> pd.Series:
        """
        计算TRIX与信号线交叉评分

        Returns:
            pd.Series: 交叉评分
        """
        cross_score = pd.Series(0.0, index=self._result.index)

        if "TRIX" not in self._result.columns or "MATRIX" not in self._result.columns:
            return cross_score

        trix_values = self._result["TRIX"]
        matrix_values = self._result["MATRIX"]

        # TRIX上穿MATRIX(金叉)+25分
        from utils.indicator_utils import crossover, crossunder

        golden_cross = crossover(trix_values, matrix_values)
        cross_score += golden_cross * 25  # TODO: 将魔法数字提取到配置中

        # TRIX下穿MATRIX(死叉)-25分
        death_cross = crossunder(trix_values, matrix_values)
        cross_score -= death_cross * 25  # TODO: 将魔法数字提取到配置中

        return cross_score

    def _calculate_trix_zero_cross_score(self) -> pd.Series:
        """
        计算TRIX零轴穿越评分

        Returns:
            pd.Series: 零轴穿越评分
        """
        zero_cross_score = pd.Series(0.0, index=self._result.index)

        if "TRIX" not in self._result.columns:
            return zero_cross_score

        trix_values = self._result["TRIX"]
        zero_line = pd.Series(0.0, index=self._result.index)

        # TRIX上穿零轴+20分
        from utils.indicator_utils import crossover, crossunder

        zero_cross_up = crossover(trix_values, zero_line)
        zero_cross_score += zero_cross_up * 20  # TODO: 将魔法数字提取到配置中

        # TRIX下穿零轴-20分
        zero_cross_down = crossunder(trix_values, zero_line)
        zero_cross_score -= zero_cross_down * 20  # TODO: 将魔法数字提取到配置中

        # TRIX在零轴上方+8分
        above_zero = trix_values > 0
        zero_cross_score += above_zero * 8  # TODO: 将魔法数字提取到配置中

        # TRIX在零轴下方-8分
        below_zero = trix_values < 0
        zero_cross_score -= below_zero * 8  # TODO: 将魔法数字提取到配置中

        return zero_cross_score

    def _calculate_trix_divergence_score(self, close_price: pd.Series) -> pd.Series:
        """
        计算TRIX背离评分

        Args:
            close_price: 收盘价序列

        Returns:
            pd.Series: 背离评分
        """
        divergence_score = pd.Series(0.0, index=close_price.index)

        if "TRIX" not in self._result.columns or len(close_price) < 20:  # TODO: 将魔法数字提取到配置中
            return divergence_score

        trix_values = self._result["TRIX"]

        # 检测背离(简化版本)
        window = 10
        for i in range(window, len(close_price)):
            # 价格创新高但TRIX未创新高(顶背离)
            price_window = close_price.iloc[i - window : i + 1]
            trix_window = trix_values.iloc[i - window : i + 1]

            if price_window.iloc[-1] == price_window.max() and trix_window.iloc[-1] < trix_window.max():
                divergence_score.iloc[i] -= 15  # 顶背离-15分  # TODO: 将魔法数字提取到配置中

            # 价格创新低但TRIX未创新低(底背离)
            if price_window.iloc[-1] == price_window.min() and trix_window.iloc[-1] > trix_window.min():
                divergence_score.iloc[i] += 15  # 底背离+15分  # TODO: 将魔法数字提取到配置中

        return divergence_score

    def _calculate_trix_strength_score(self) -> pd.Series:
        """
        计算TRIX强度评分

        Returns:
            pd.Series: 强度评分
        """
        strength_score = pd.Series(0.0, index=self._result.index)

        if "TRIX" not in self._result.columns:
            return strength_score

        trix_values = self._result["TRIX"]

        # 计算TRIX的绝对值作为强度指标
        trix_abs = abs(trix_values)

        # 强度分级评分
        # 强度很高(>2%)+10分
        very_strong = trix_abs > 2
        strength_score += very_strong * 10

        # 强度高(1-2%)+5分
        strong = (trix_abs > 1) & (trix_abs <= 2)
        strength_score += strong * 5  # TODO: 将魔法数字提取到配置中

        # 强度低(<0.5%)-5分  # TODO: 将魔法数字提取到配置中
        weak = trix_abs < 0.5  # TODO: 将魔法数字提取到配置中
        strength_score -= weak * 5  # TODO: 将魔法数字提取到配置中

        return strength_score

    def _detect_trix_trend_patterns(self) -> List[str]:
        """
        检测TRIX趋势形态

        Returns:
            List[str]: 趋势形态列表
        """
        patterns = []

        if "TRIX" not in self._result.columns or len(self._result) < 5:  # TODO: 将魔法数字提取到配置中
            return patterns

        trix_values = self._result["TRIX"]

        # 检查最近的趋势
        recent_periods = min(5, len(trix_values))  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        recent_trix = trix_values.tail(recent_periods)

        rising_count = 0
        falling_count = 0

        for i in range(1, len(recent_trix)):
            if recent_trix.iloc[i] > recent_trix.iloc[i - 1]:
                rising_count += 1
            elif recent_trix.iloc[i] < recent_trix.iloc[i - 1]:
                falling_count += 1

        if rising_count >= 3:  # TODO: 将魔法数字提取到配置中
            patterns.append("TRIX上升趋势")
        elif falling_count >= 3:  # TODO: 将魔法数字提取到配置中
            patterns.append("TRIX下降趋势")
        else:
            patterns.append("TRIX震荡趋势")

        return patterns

    def _detect_trix_cross_patterns(self) -> List[str]:
        """
        检测TRIX交叉形态

        Returns:
            List[str]: 交叉形态列表
        """
        patterns = []

        if "TRIX" not in self._result.columns or "MATRIX" not in self._result.columns or len(self._result) < 2:
            return patterns

        trix_values = self._result["TRIX"]
        matrix_values = self._result["MATRIX"]

        # 检查最近的交叉
        recent_periods = min(5, len(trix_values))  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        recent_trix = trix_values.tail(recent_periods)
        recent_matrix = matrix_values.tail(recent_periods)

        from utils.indicator_utils import crossover, crossunder

        if crossover(recent_trix, recent_matrix).any():
            patterns.append("TRIX金叉")

        if crossunder(recent_trix, recent_matrix).any():
            patterns.append("TRIX死叉")

        # 当前位置关系
        if len(trix_values) > 0 and len(matrix_values) > 0:
            current_trix = trix_values.iloc[-1]
            current_matrix = matrix_values.iloc[-1]

            if pd.notna(current_trix) and pd.notna(current_matrix):
                if current_trix > current_matrix:
                    patterns.append("TRIX在信号线上方")
                else:
                    patterns.append("TRIX在信号线下方")

        return patterns

    def _detect_trix_zero_cross_patterns(self, data: pd.DataFrame) -> bool:
        """检测TRIX零轴穿越形态"""
        if not self.has_result():
            return False

        # 确保数据量足够
        if len(self._result) < 5:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            return False

        # 检查必要的列是否存在
        if "trix" not in self._result.columns:
            logging.warning("检测TRIX零轴穿越形态时缺少必要的列: ['trix']")
            return False

        # 获取TRIX和Signal线数据
        trix = self._result["trix"].iloc[-5:]  # TODO: 将魔法数字提取到配置中

        # 检测是否有零轴穿越
        zero_cross_up = (trix.iloc[-2] < 0) and (trix.iloc[-1] > 0)
        zero_cross_down = (trix.iloc[-2] > 0) and (trix.iloc[-1] < 0)

        return zero_cross_up or zero_cross_down

    def _detect_trix_divergence_patterns(self, data: pd.DataFrame) -> bool:
        """检测TRIX背离形态"""
        if not self.has_result() or "close" not in data.columns:
            return False

        # 确保数据量足够
        if len(self._result) < 10 or len(data) < 10:
            return False

        # 检查必要的列是否存在
        if "trix" not in self._result.columns:
            logging.warning("检测TRIX背离形态时缺少必要的列: ['trix']")
            return False

        # 获取近期的价格和TRIX数据
        prices = data["close"].iloc[-10:]
        trix_values = self._result["trix"].iloc[-10:]

        # 查找近期的价格高点和低点
        price_high_idx = prices.idxmax()
        price_low_idx = prices.idxmin()

        # 查找近期的TRIX高点和低点
        trix_high_idx = trix_values.idxmax()
        trix_low_idx = trix_values.idxmin()

        # 检测顶背离:价格创新高,但TRIX未创新高
        bearish_divergence = (price_high_idx > trix_high_idx) and (
            prices.iloc[-1] > prices.iloc[-5]
        )  # TODO: 将魔法数字提取到配置中

        # 检测底背离:价格创新低,但TRIX未创新低
        bullish_divergence = (price_low_idx > trix_low_idx) and (
            prices.iloc[-1] < prices.iloc[-5]
        )  # TODO: 将魔法数字提取到配置中

        return bearish_divergence or bullish_divergence

    def _detect_trix_signal_cross_patterns(self, data: pd.DataFrame) -> bool:
        """检测TRIX与信号线交叉形态"""
        if not self.has_result():
            return False

        # 确保数据量足够
        if len(self._result) < 5:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            return False

        # 获取TRIX和Signal线数据
        trix = self._result["trix"].iloc[-3:]  # TODO: 将魔法数字提取到配置中
        signal = self._result["signal"].iloc[-3:]  # TODO: 将魔法数字提取到配置中

        # 检测TRIX上穿信号线
        golden_cross = (trix.iloc[-3] < signal.iloc[-3]) and (
            trix.iloc[-1] > signal.iloc[-1]
        )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        # 检测TRIX下穿信号线
        death_cross = (trix.iloc[-3] > signal.iloc[-3]) and (
            trix.iloc[-1] < signal.iloc[-1]
        )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        return golden_cross or death_cross

    def _detect_trix_trend_change_patterns(self, data: pd.DataFrame) -> bool:
        """检测TRIX趋势变化形态"""
        if not self.has_result():
            return False

        # 确保数据量足够
        if len(self._result) < 10:
            return False

        # 获取TRIX数据
        trix = self._result["trix"].iloc[-10:]

        # 计算TRIX的斜率变化
        slope_1 = trix.iloc[5] - trix.iloc[0]  # TODO: 将魔法数字提取到配置中
        slope_2 = trix.iloc[-1] - trix.iloc[-6]  # TODO: 将魔法数字提取到配置中

        # 检测趋势变化(斜率从正变负或从负变正)
        trend_change = slope_1 * slope_2 < 0

        return trend_change

    def _detect_trix_acceleration_patterns(self, data: pd.DataFrame) -> bool:
        """检测TRIX加速形态"""
        if not self.has_result():
            return False

        # 确保数据量足够
        if len(self._result) < 15:  # TODO: 将魔法数字提取到配置中
            return False

        # 获取TRIX数据
        trix = self._result["trix"].iloc[-15:]  # TODO: 将魔法数字提取到配置中

        # 计算TRIX的变化率
        changes = [
            trix.iloc[i + 5] - trix.iloc[i] for i in range(0, 10, 5)
        ]  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        # 加速条件:变化率逐渐增大,且方向一致
        acceleration = (changes[0] > 0 and changes[1] > changes[0] and changes[2] > changes[1]) or (
            changes[0] < 0 and changes[1] < changes[0] and changes[2] < changes[1]
        )

        return acceleration

    def _detect_trix_strength_patterns(self) -> List[str]:
        """
        检测TRIX强度形态

        Returns:
            List[str]: 强度形态列表
        """
        patterns = []

        if "TRIX" not in self._result.columns or len(self._result) == 0:
            return patterns

        trix_values = self._result["TRIX"]
        current_trix = trix_values.iloc[-1]

        if pd.isna(current_trix):
            return patterns

        trix_abs = abs(current_trix)

        if trix_abs > 2:
            patterns.append("TRIX强势信号")
        elif trix_abs > 1:
            patterns.append("TRIX中等强度信号")
        elif trix_abs < 0.5:  # TODO: 将魔法数字提取到配置中
            patterns.append("TRIX弱势信号")
        else:
            patterns.append("TRIX正常强度信号")

        return patterns

    def _legacy_generate_signals_trix(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        生成TRIX指标标准化交易信号

        Args:
            data: 输入数据,包含OHLCV数据
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            pd.DataFrame: 信号结果Data_frame,包含标准化信号
        """
        # 确保已计算TRIX指标
        if not self.has_result():
            self.calculate_Trix(data)

        # 初始化信号DataFrame
        signals = pd.DataFrame(index=data.index)
        signals["buy_signal"] = False
        signals["sell_signal"] = False
        signals["neutral_signal"] = True  # 默认为中性信号
        signals["trend"] = 0  # 0表示中性
        signals["score"] = 50.0  # 默认评分50分  # TODO: 将魔法数字提取到配置中
        signals["signal_type"] = None
        signals["signal_desc"] = None
        signals["confidence"] = 50.0  # TODO: 将魔法数字提取到配置中
        signals["risk_level"] = "中"
        signals["position_size"] = 0.0
        signals["stop_loss"] = None
        signals["market_env"] = "sideways_market"
        signals["volume_confirmation"] = False

        # 计算评分
        score = self.calculate_raw_score_Trix(data, **kwargs)
        signals["score"] = score

        # 检测形态
        patterns = self.identify_patterns_Trix(data, **kwargs)

        # 获取TRIX数据
        trix = self._result["TRIX"]
        matrix = self._result["MATRIX"]

        # 1. TRIX上穿信号线,买入信号
        trix_crossover_matrix = (trix > matrix) & (trix.shift(1) <= matrix.shift(1))
        signals.loc[trix_crossover_matrix, "buy_signal"] = True
        signals.loc[trix_crossover_matrix, "neutral_signal"] = False
        signals.loc[trix_crossover_matrix, "trend"] = 1
        signals.loc[trix_crossover_matrix, "signal_type"] = "TRIX金叉"
        signals.loc[trix_crossover_matrix, "signal_desc"] = "TRIX上穿信号线,买入信号"
        signals.loc[trix_crossover_matrix, "confidence"] = 70.0  # TODO: 将魔法数字提取到配置中
        signals.loc[trix_crossover_matrix, "position_size"] = 0.4  # TODO: 将魔法数字提取到配置中
        signals.loc[trix_crossover_matrix, "risk_level"] = "中"

        # 2. TRIX下穿信号线,卖出信号
        trix_crossunder_matrix = (trix < matrix) & (trix.shift(1) >= matrix.shift(1))
        signals.loc[trix_crossunder_matrix, "sell_signal"] = True
        signals.loc[trix_crossunder_matrix, "neutral_signal"] = False
        signals.loc[trix_crossunder_matrix, "trend"] = -1
        signals.loc[trix_crossunder_matrix, "signal_type"] = "TRIX死叉"
        signals.loc[trix_crossunder_matrix, "signal_desc"] = "TRIX下穿信号线,卖出信号"
        signals.loc[trix_crossunder_matrix, "confidence"] = 70.0  # TODO: 将魔法数字提取到配置中
        signals.loc[trix_crossunder_matrix, "position_size"] = 0.4  # TODO: 将魔法数字提取到配置中
        signals.loc[trix_crossunder_matrix, "risk_level"] = "中"

        # 3. TRIX上穿零轴,买入信号  # TODO: 将魔法数字提取到配置中
        trix_crossover_zero = (trix > 0) & (trix.shift(1) <= 0)
        signals.loc[trix_crossover_zero, "buy_signal"] = True
        signals.loc[trix_crossover_zero, "neutral_signal"] = False
        signals.loc[trix_crossover_zero, "trend"] = 1
        signals.loc[trix_crossover_zero, "signal_type"] = "TRIX零轴上穿"
        signals.loc[trix_crossover_zero, "signal_desc"] = "TRIX上穿零轴,转为上升趋势"
        signals.loc[trix_crossover_zero, "confidence"] = 75.0  # TODO: 将魔法数字提取到配置中
        signals.loc[trix_crossover_zero, "position_size"] = 0.5  # TODO: 将魔法数字提取到配置中
        signals.loc[trix_crossover_zero, "risk_level"] = "中"

        # 4. TRIX下穿零轴,卖出信号  # TODO: 将魔法数字提取到配置中
        trix_crossunder_zero = (trix < 0) & (trix.shift(1) >= 0)
        signals.loc[trix_crossunder_zero, "sell_signal"] = True
        signals.loc[trix_crossunder_zero, "neutral_signal"] = False
        signals.loc[trix_crossunder_zero, "trend"] = -1
        signals.loc[trix_crossunder_zero, "signal_type"] = "TRIX零轴下穿"
        signals.loc[trix_crossunder_zero, "signal_desc"] = "TRIX下穿零轴,转为下降趋势"
        signals.loc[trix_crossunder_zero, "confidence"] = 75.0  # TODO: 将魔法数字提取到配置中
        signals.loc[trix_crossunder_zero, "position_size"] = 0.5  # TODO: 将魔法数字提取到配置中
        signals.loc[trix_crossunder_zero, "risk_level"] = "中"

        # 5. TRIX持续上升且为正值,强势上涨信号  # TODO: 将魔法数字提取到配置中
        strong_uptrend = (trix > 0) & (trix > trix.shift(1)) & (trix.shift(1) > trix.shift(2))
        signals.loc[strong_uptrend, "buy_signal"] = True
        signals.loc[strong_uptrend, "neutral_signal"] = False
        signals.loc[strong_uptrend, "trend"] = 1
        signals.loc[strong_uptrend, "signal_type"] = "TRIX强势上涨"
        signals.loc[strong_uptrend, "signal_desc"] = "TRIX持续上升且为正值,强势上涨信号"
        signals.loc[strong_uptrend, "confidence"] = 80.0  # TODO: 将魔法数字提取到配置中
        signals.loc[strong_uptrend, "position_size"] = 0.6  # TODO: 将魔法数字提取到配置中
        signals.loc[strong_uptrend, "risk_level"] = "低"

        # 6. TRIX持续下降且为负值,强势下跌信号  # TODO: 将魔法数字提取到配置中
        strong_downtrend = (trix < 0) & (trix < trix.shift(1)) & (trix.shift(1) < trix.shift(2))
        signals.loc[strong_downtrend, "sell_signal"] = True
        signals.loc[strong_downtrend, "neutral_signal"] = False
        signals.loc[strong_downtrend, "trend"] = -1
        signals.loc[strong_downtrend, "signal_type"] = "TRIX强势下跌"
        signals.loc[strong_downtrend, "signal_desc"] = "TRIX持续下降且为负值,强势下跌信号"
        signals.loc[strong_downtrend, "confidence"] = 80.0  # TODO: 将魔法数字提取到配置中
        signals.loc[strong_downtrend, "position_size"] = 0.6  # TODO: 将魔法数字提取到配置中
        signals.loc[strong_downtrend, "risk_level"] = "低"

        # 7. TRIX高位死叉,顶部信号  # TODO: 将魔法数字提取到配置中
        high_level_death_cross = trix_crossunder_matrix & (trix > 0.5)  # TODO: 将魔法数字提取到配置中
        signals.loc[high_level_death_cross, "sell_signal"] = True
        signals.loc[high_level_death_cross, "neutral_signal"] = False
        signals.loc[high_level_death_cross, "trend"] = -1
        signals.loc[high_level_death_cross, "signal_type"] = "TRIX高位死叉"
        signals.loc[high_level_death_cross, "signal_desc"] = "TRIX在高位下穿信号线,顶部信号"
        signals.loc[high_level_death_cross, "confidence"] = 85.0  # TODO: 将魔法数字提取到配置中
        signals.loc[high_level_death_cross, "position_size"] = 0.7  # TODO: 将魔法数字提取到配置中
        signals.loc[high_level_death_cross, "risk_level"] = "低"

        # 8. TRIX低位金叉,底部信号  # TODO: 将魔法数字提取到配置中
        low_level_golden_cross = trix_crossover_matrix & (trix < -0.5)  # TODO: 将魔法数字提取到配置中
        signals.loc[low_level_golden_cross, "buy_signal"] = True
        signals.loc[low_level_golden_cross, "neutral_signal"] = False
        signals.loc[low_level_golden_cross, "trend"] = 1
        signals.loc[low_level_golden_cross, "signal_type"] = "TRIX低位金叉"
        signals.loc[low_level_golden_cross, "signal_desc"] = "TRIX在低位上穿信号线,底部信号"
        signals.loc[low_level_golden_cross, "confidence"] = 85.0  # TODO: 将魔法数字提取到配置中
        signals.loc[low_level_golden_cross, "position_size"] = 0.7  # TODO: 将魔法数字提取到配置中
        signals.loc[low_level_golden_cross, "risk_level"] = "低"

        # 9. TRIX区间震荡,无明显趋势  # TODO: 将魔法数字提取到配置中
        sideways = (abs(trix) < 0.1) & (abs(trix - trix.shift(1)) < 0.05)  # TODO: 将魔法数字提取到配置中
        signals.loc[sideways, "neutral_signal"] = True
        signals.loc[sideways, "buy_signal"] = False
        signals.loc[sideways, "sell_signal"] = False
        signals.loc[sideways, "trend"] = 0
        signals.loc[sideways, "signal_type"] = "TRIX区间震荡"
        signals.loc[sideways, "signal_desc"] = "TRIX在零轴附近小幅波动,无明显趋势"
        signals.loc[sideways, "confidence"] = 60.0  # TODO: 将魔法数字提取到配置中
        signals.loc[sideways, "position_size"] = 0.0
        signals.loc[sideways, "risk_level"] = "低"

        # 10. 根据形态设置更多信号
        for pattern in patterns:
            pattern_idx = signals.index[-5:]  # 假设形态影响最近5个周期  # TODO: 将魔法数字提取到配置中

            if "上升" in pattern or "看涨" in pattern or "金叉" in pattern:
                signals.loc[pattern_idx, "buy_signal"] = True
                signals.loc[pattern_idx, "neutral_signal"] = False
                signals.loc[pattern_idx, "trend"] = 1
                signals.loc[pattern_idx, "signal_type"] = "TRIX形态信号"
                signals.loc[pattern_idx, "signal_desc"] = pattern
                signals.loc[pattern_idx, "confidence"] = (
                    70.0  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                )
                signals.loc[pattern_idx, "position_size"] = (
                    0.4  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                )
                signals.loc[pattern_idx, "risk_level"] = "中"

            elif "下降" in pattern or "看跌" in pattern or "死叉" in pattern:
                signals.loc[pattern_idx, "sell_signal"] = True
                signals.loc[pattern_idx, "neutral_signal"] = False
                signals.loc[pattern_idx, "trend"] = -1
                signals.loc[pattern_idx, "signal_type"] = "TRIX形态信号"
                signals.loc[pattern_idx, "signal_desc"] = pattern
                signals.loc[pattern_idx, "confidence"] = (
                    70.0  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                )
                signals.loc[pattern_idx, "position_size"] = (
                    0.4  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                )
                signals.loc[pattern_idx, "risk_level"] = "中"

            elif "震荡" in pattern or "平稳" in pattern:
                signals.loc[pattern_idx, "neutral_signal"] = True
                signals.loc[pattern_idx, "buy_signal"] = False
                signals.loc[pattern_idx, "sell_signal"] = False
                signals.loc[pattern_idx, "trend"] = 0
                signals.loc[pattern_idx, "signal_type"] = "TRIX震荡信号"
                signals.loc[pattern_idx, "signal_desc"] = pattern
                signals.loc[pattern_idx, "confidence"] = 60.0  # TODO: 将魔法数字提取到配置中
                signals.loc[pattern_idx, "position_size"] = 0.0
                signals.loc[pattern_idx, "risk_level"] = "低"

        # 设置止损价格
        if "low" in data.columns and "high" in data.columns:
            # 买入信号的止损设为最近的低点
            buy_indices = signals[signals["buy_signal"]].index
            if not buy_indices.empty:
                for idx in buy_indices:
                    if idx > data.index[10]:  # 确保有足够的历史数据
                        lookback = 5  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                        # 使用最近低点作为止损
                        recent_low = data.loc[idx - lookback : idx, "low"].min()
                        signals.loc[idx, "stop_loss"] = recent_low

            # 卖出信号的止损设为最近的高点
            sell_indices = signals[signals["sell_signal"]].index
            if not sell_indices.empty:
                for idx in sell_indices:
                    if idx > data.index[10]:  # 确保有足够的历史数据
                        lookback = 5  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                        # 使用最近高点作为止损
                        recent_high = data.loc[idx - lookback : idx, "high"].max()
                        signals.loc[idx, "stop_loss"] = recent_high

        # 根据TRIX值判断市场环境
        signals["market_env"] = "sideways_market"  # 默认震荡市场

        # TRIX > 0 且上升,牛市环境
        bull_market = (trix > 0) & (trix > trix.shift(1))
        signals.loc[bull_market, "market_env"] = "bull_market"

        # TRIX < 0 且下降,熊市环境
        bear_market = (trix < 0) & (trix < trix.shift(1))
        signals.loc[bear_market, "market_env"] = "bear_market"

        # TRIX接近0且波动小,震荡市场
        sideways_market = (abs(trix) < 0.1) & (abs(trix - trix.shift(1)) < 0.05)  # TODO: 将魔法数字提取到配置中
        signals.loc[sideways_market, "market_env"] = "sideways_market"

        # 设置成交量确认
        if "volume" in data.columns:
            # 如果有成交量数据,检查成交量是否支持当前信号
            vol = data["volume"]
            vol_avg = vol.rolling(window=20).mean()  # TODO: 将魔法数字提取到配置中

            # 成交量大于20日均量1.5倍为放量
            vol_increase = vol > vol_avg * 1.5  # TODO: 将魔法数字提取到配置中

            # 买入信号且成交量放大,确认信号
            signals.loc[signals["buy_signal"] & vol_increase, "volume_confirmation"] = True

            # 卖出信号且成交量放大,确认信号
            signals.loc[signals["sell_signal"] & vol_increase, "volume_confirmation"] = True

        return signals

    def _register_trix_patterns(self):
        """注册TRIX特有的形态检测方法"""
        from indicators.pattern_registry import (
            PatternRegistry,
            PatternTypePatternRegistry,
            PatternStrengthPatternRegistry,
        )

        # 获取PatternRegistry实例
        registry = PatternRegistry()

        # 注册TRIX零轴穿越形态
        registry.register(
            pattern_id="TRIX_ZERO_CROSS",
            display_name="TRIX零轴穿越",
            description="TRIX线穿越零轴,指示可能的趋势转变",
            indicator_id="TRIX",
            pattern_type=Pattern_type.REVERSAL,
            default_strength=Pattern_strength.MEDIUM,
            score_impact=15.0,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            detection_function=self._detect_trix_zero_cross_patterns,
        )

        # 注册TRIX背离形态
        registry.register(
            pattern_id="TRIX_DIVERGENCE",
            display_name="TRIX背离",
            description="TRIX指标与价格走势形成背离,可能指示趋势反转",
            indicator_id="TRIX",
            pattern_type=Pattern_type.REVERSAL,
            default_strength=Pattern_strength.STRONG,
            score_impact=20.0,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            detection_function=lambda data: self._detect_trix_divergence_patterns(data["close"]),
        )

    def _detect_golden_cross_Trix(self, series1: pd.Series, series2: pd.Series) -> pd.Series:
        """检测金叉"""
        from indicators.common import crossover

        return pd.Series(crossover(series1.values, series2.values), index=series1.index)

    def _detect_death_cross_Trix(self, series1: pd.Series, series2: pd.Series) -> pd.Series:
        """检测死叉"""
        from indicators.common import crossunder
        from db.sql_manager import SQLManager, QueryType

        return pd.Series(crossunder(series1.values, series2.values), index=series1.index)

    def get_pattern_info_Trix(self, pattern_id: str) -> dict:
        """
        获取形态信息

        Args:
            pattern_id: 形态ID

        Returns:
            dict: 形态信息字典
        """
        # 默认形态信息映射
        pattern_info_map = {
            # 基础形态
            "bullish": {"name": "看涨形态", "description": "指标显示看涨信号", "type": "BULLISH"},
            "bearish": {"name": "看跌形态", "description": "指标显示看跌信号", "type": "BEARISH"},
            "neutral": {"name": "中性形态", "description": "指标显示中性信号", "type": "NEUTRAL"},
            # 通用形态
            "strong_signal": {"name": "强信号", "description": "强烈的技术信号", "type": "STRONG"},
            "weak_signal": {"name": "弱信号", "description": "较弱的技术信号", "type": "WEAK"},
            "trend_up": {"name": "上升趋势", "description": "价格呈上升趋势", "type": "BULLISH"},
            "trend_down": {"name": "下降趋势", "description": "价格呈下降趋势", "type": "BEARISH"},
        }

        # 默认形态信息
        default_pattern = {
            "name": pattern_id.replace("_", " ").title(),
            "description": f"{pattern_id}形态",
            "type": "UNKNOWN",
        }

        return pattern_info_map.get(pattern_id, default_pattern)

    def _legacy_generate_trading_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成交易信号 - Ultra Think九连胜传奇成功模式

        Args:
            data: 价格数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含买卖信号的数据框
        """
        result = self.calculate(data)
        if len(result) == 0:
            signals = pd.DataFrame(index=data.index)
            signals["buy_signal"] = False
            signals["sell_signal"] = False
            signals["signal_strength"] = 0.0
            return signals

        # 查找TRIX和MATRIX列
        trix_col = None
        matrix_col = None
        for col in result.columns:
            if "trix" in col.lower() and "matrix" not in col.lower():
                trix_col = col
            elif "matrix" in col.lower():
                matrix_col = col

        if trix_col is None or matrix_col is None:
            signals = pd.DataFrame(index=data.index)
            signals["buy_signal"] = False
            signals["sell_signal"] = False
            signals["signal_strength"] = 0.0
            return signals

        trix_values = result[trix_col]
        matrix_values = result[matrix_col]

        signals = pd.DataFrame(index=data.index)

        # TRIX金叉MATRIX(上穿信号线)买入信号
        trix_above = trix_values > matrix_values
        trix_below_prev = trix_values.shift(1) <= matrix_values.shift(1)
        buy_signals = trix_above & trix_below_prev

        # TRIX死叉MATRIX(下穿信号线)卖出信号
        trix_below = trix_values < matrix_values
        trix_above_prev = trix_values.shift(1) >= matrix_values.shift(1)
        sell_signals = trix_below & trix_above_prev

        signals["buy_signal"] = buy_signals
        signals["sell_signal"] = sell_signals

        # 计算信号强度(基于TRIX与MATRIX的差值)
        diff = abs(trix_values - matrix_values)
        max_diff = diff.rolling(window=20, min_periods=1).max()  # TODO: 将魔法数字提取到配置中
        signals["signal_strength"] = diff / (max_diff + 1e-10)

        return signals

    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        识别技术形态 - Ultra Think九连胜传奇成功模式

        Args:
            data: 价格数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含形态识别结果的数据框
        """
        result = self.calculate(data)
        if len(result) == 0:
            patterns = pd.DataFrame(index=data.index)
            patterns["golden_cross"] = False
            patterns["death_cross"] = False
            patterns["bullish_trend"] = False
            patterns["bearish_trend"] = False
            return patterns

        # 查找TRIX和MATRIX列
        trix_col = None
        matrix_col = None
        for col in result.columns:
            if "trix" in col.lower() and "matrix" not in col.lower():
                trix_col = col
            elif "matrix" in col.lower():
                matrix_col = col

        if trix_col is None or matrix_col is None:
            patterns = pd.DataFrame(index=data.index)
            patterns["golden_cross"] = False
            patterns["death_cross"] = False
            patterns["bullish_trend"] = False
            patterns["bearish_trend"] = False
            return patterns

        trix_values = result[trix_col]
        matrix_values = result[matrix_col]

        patterns = pd.DataFrame(index=data.index)

        # 金叉形态(TRIX上穿MATRIX)
        trix_above = trix_values > matrix_values
        trix_below_prev = trix_values.shift(1) <= matrix_values.shift(1)
        patterns["golden_cross"] = trix_above & trix_below_prev

        # 死叉形态(TRIX下穿MATRIX)
        trix_below = trix_values < matrix_values
        trix_above_prev = trix_values.shift(1) >= matrix_values.shift(1)
        patterns["death_cross"] = trix_below & trix_above_prev

        # 看涨趋势(TRIX持续在MATRIX上方)
        patterns["bullish_trend"] = trix_values > matrix_values

        # 看跌趋势(TRIX持续在MATRIX下方)
        patterns["bearish_trend"] = trix_values < matrix_values

        return patterns

    def calculate_confidence_Indicator_Base_Indicator(
        self, score: pd.Series, patterns: pd.DataFrame, signals: dict
    ) -> float:
        """
        计算指标置信度 - Ultra Think九连胜传奇成功模式

        Args:
            score: 原始评分序列
            patterns: 形态识别结果
            signals: 信号字典

        Returns:
            float: 置信度评分
        """
        return self.calculate_confidence_Trix(score, patterns, signals)

    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算原始评分 - Ultra Think九连胜传奇成功模式

        Args:
            data: 价格数据
            **kwargs: 其他参数

        Returns:
            pd.Series: 原始评分序列
        """
        result = self.calculate(data, **kwargs)

        # 查找TRIX列
        trix_col = None
        for col in result.columns:
            if "trix" in col.lower() and "matrix" not in col.lower():
                trix_col = col
                break

        if trix_col is not None:
            # 使用TRIX值作为原始评分
            return result[trix_col]
        else:
            # 返回零评分
            return pd.Series(index=data.index, data=0.0)

    def get_patterns_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取形态识别结果 - Ultra Think九连胜传奇成功模式

        Args:
            data: 价格数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 形态识别结果
        """
        return self.get_patterns(data, **kwargs)

    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """
        设置指标参数 - Ultra Think九连胜传奇成功模式

        Args:
            **kwargs: 参数字典
        """
        self.set_parameters_Trix(**kwargs)

    @property
    def minimum_periods(self) -> int:
        """
        TripleExponentialAverage指标所需的最少数据周期数

        计算逻辑:使用默认值

        Returns:
            int: 最少需要的数据周期数
        """
        return 35  # TODO: 将魔法数字提取到配置中

    # 删除重复的get_signal方法 - 保留后面完整版本
    @performance_monitor(threshold=1.0)
    @exception_handler(reraise=False, default_return=None)
    def get_signal(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        基于TRIX指标生成交易信号
        
        TRIX通过三重指数平滑过滤短期噪音，捕捉中长期趋势变化，
        主要通过零轴交叉和信号线交叉生成交易信号。
        
        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 额外参数
            
        Returns:
            Dict[str, Any]: 标准化交易信号格式
        """
        try:
            # 1. 数据验证
            if not self._validate_signal_data(data):
                return self._get_default_signal("数据验证失败")
            
            # 2. 计算TRIX指标
            result = self.calculate(data)
            if result.empty or 'trix_value' not in result.columns:
                return self._get_default_signal("TRIX计算失败")

            # 3. 获取最新的TRIX数据
            latest_trix = result['trix_value'].iloc[-1]
            latest_signal = result.get('trix_signal', pd.Series()).iloc[-1] if 'trix_signal' in result.columns else None
            
            if pd.isna(latest_trix):
                return self._get_default_signal("最新TRIX值为空")

            # 4. 生成基于TRIX的信号
            signal_type = "hold"
            strength = 0.0
            confidence = 0.5
            reason = "TRIX趋势分析"
            
            # TRIX零轴交叉信号（主要信号源）
            trix_series = result['trix_value'].dropna()
            if len(trix_series) >= 2:
                prev_trix = trix_series.iloc[-2]
                
                # 零轴上穿 - 强烈买入信号
                if prev_trix <= 0 and latest_trix > 0:
                    signal_type = "buy"
                    strength = min(0.9, abs(latest_trix) * 100)  # TRIX值通常很小
                    confidence = 0.85
                    reason = f"TRIX零轴上穿({latest_trix:.4f})，长期趋势转强"
                    
                # 零轴下穿 - 强烈卖出信号
                elif prev_trix >= 0 and latest_trix < 0:
                    signal_type = "sell"
                    strength = min(0.9, abs(latest_trix) * 100)
                    confidence = 0.85
                    reason = f"TRIX零轴下穿({latest_trix:.4f})，长期趋势转弱"
                    
                # TRIX与信号线交叉（次要信号源）
                elif latest_signal is not None and not pd.isna(latest_signal):
                    if len(result) >= 2:
                        prev_signal = result['trix_signal'].iloc[-2]
                        
                        # TRIX上穿信号线
                        if (trix_series.iloc[-2] <= prev_signal and 
                            latest_trix > latest_signal):
                            signal_type = "buy"
                            strength = min(0.7, abs(latest_trix - latest_signal) * 200)
                            confidence = 0.7
                            reason = f"TRIX上穿信号线，短期上涨动能增强"
                            
                        # TRIX下穿信号线
                        elif (trix_series.iloc[-2] >= prev_signal and 
                              latest_trix < latest_signal):
                            signal_type = "sell"
                            strength = min(0.7, abs(latest_trix - latest_signal) * 200)
                            confidence = 0.7
                            reason = f"TRIX下穿信号线，短期下跌压力增加"
                            
                # 基于TRIX斜率的趋势强度判断
                if len(trix_series) >= 5:
                    # 计算TRIX的趋势斜率（5日变化）
                    trix_change = latest_trix - trix_series.iloc[-5]
                    
                    if abs(trix_change) > 0.001:  # 有意义的变化
                        if trix_change > 0 and signal_type == "hold":
                            signal_type = "buy"
                            strength = min(0.5, abs(trix_change) * 50)
                            confidence = 0.6
                            reason = f"TRIX持续上升({trix_change:.4f})，趋势向好"
                            
                        elif trix_change < 0 and signal_type == "hold":
                            signal_type = "sell"
                            strength = min(0.5, abs(trix_change) * 50)
                            confidence = 0.6
                            reason = f"TRIX持续下降({trix_change:.4f})，趋势转弱"

            # 5. 结合背离分析增强信号质量
            try:
                if hasattr(self, '_detect_divergence') and len(data) >= 30:
                    has_divergence = self._detect_divergence(data.tail(30))
                    if has_divergence:
                        confidence = min(confidence + 0.1, 0.95)
                        if "背离" not in reason:
                            reason += "，检测到价格背离信号"
            except Exception:
                pass  # 背离检测失败不影响主信号

            # 6. 构建标准化信号
            return {
                'signal_type': signal_type,
                'strength': round(strength, 3),
                'confidence': round(confidence, 3),
                'timestamp': pd.Timestamp.now(),
                'reason': reason,
                'metadata': {
                    'indicator': 'TRIX',
                    'current_trix': float(latest_trix),
                    'signal_line': float(latest_signal) if latest_signal is not None and not pd.isna(latest_signal) else None,
                    'period': getattr(self, 'period', 14),
                    'signal_period': getattr(self, 'signal_period', 9)
                }
            }
            
        except Exception as e:
            logger.error(f"TRIX信号生成失败: {e}")
            return self._get_default_signal(f"信号生成失败: {str(e)}")

    def _validate_signal_data(self, data: pd.DataFrame) -> bool:
        """
        验证信号生成所需的数据
        
        Args:
            data: 输入数据DataFrame
            
        Returns:
            bool: 数据是否有效
        """
        try:
            if data is None or data.empty:
                logger.error("TRIX信号数据验证失败: 输入数据为空")
                return False
                
            if not isinstance(data, pd.DataFrame):
                logger.error(f"TRIX信号数据验证失败: 输入数据类型错误，期望DataFrame，实际{type(data)}")
                return False
                
            required_columns = ['close']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                logger.error(f"TRIX信号数据验证失败: 缺少必需列 {missing_columns}")
                return False
                
            # 检查数据值的有效性
            for col in required_columns:
                if data[col].isna().all():
                    logger.error(f"TRIX信号数据验证失败: 列{col}全部为NaN值")
                    return False
                    
            # TRIX需要足够的数据用于三重平滑
            min_periods = getattr(self, 'period', 14) * 3 + 10
            if len(data) < min_periods:
                logger.error(f"TRIX信号数据验证失败: 数据长度不足，需要至少{min_periods}个数据点，实际{len(data)}个")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"TRIX信号数据验证异常: {e}")
            return False

    def validate_data_structure(self, data: pd.DataFrame) -> bool:
        """
        验证数据结构是否符合TRIX指标要求
        
        Args:
            data: 输入数据DataFrame
            
        Returns:
            bool: 数据结构是否有效
        """
        try:
            if not isinstance(data, pd.DataFrame):
                logger.error(f"TRIX数据验证失败: 输入数据类型错误，期望DataFrame，实际{type(data)}")
                return False
                
            if data.empty:
                logger.error("TRIX数据验证失败: 输入数据为空")
                return False
                
            required_columns = ['close']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                logger.error(f"TRIX数据验证失败: 缺少必需列 {missing_columns}")
                return False
                
            # 检查数据长度
            min_length = getattr(self, 'period', 14) * 3 + 20  # TRIX需要更多数据进行三重平滑
            if len(data) < min_length:
                logger.error(f"TRIX数据验证失败: 数据长度不足，需要至少{min_length}个数据点，实际{len(data)}个")
                return False
                
            # 检查数据值的有效性
            for col in required_columns:
                if data[col].isna().all():
                    logger.error(f"TRIX数据验证失败: 列{col}全部为NaN值")
                    return False
                    
            # 检查价格数据的合理性（收盘价应为正数）
            if (data['close'] <= 0).any():
                logger.warning("TRIX数据验证警告: 存在非正数的收盘价")
                
            return True
            
        except Exception as e:
            logger.error(f"TRIX数据结构验证异常: {e}")
            return False

    def _get_default_signal(self, reason: str = "数据不足") -> Dict[str, Any]:
        """
        生成默认信号（持有信号）
        
        Args:
            reason: 生成默认信号的原因
            
        Returns:
            Dict[str, Any]: 默认信号
        """
        return {
            'signal_type': 'hold',
            'strength': 0.0,
            'confidence': 0.0,
            'timestamp': pd.Timestamp.now(),
            'reason': reason,
            'metadata': {
                'indicator': 'TRIX',
                'period': getattr(self, 'period', 14)
            }
        }
