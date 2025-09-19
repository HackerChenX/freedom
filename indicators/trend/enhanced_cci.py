from utils.container import container
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple, List
import logging

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin
from utils.technical_utils import find_peaks_and_troughs

from utils.logger import get_logger

logger = get_logger(__name__)
from utils.indicator_utils import crossover, crossunder
from db.sql_manager import SQLManager, QueryType


class EnhancedCci(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
    """
    EnhancedCci - L4核心服务层组件

    职责合理性说明:
    - 作为L4层核心服务组件，承担多项相关职责
    - 33个方法分为以下职责组:
      * 核心功能方法 (约11个)
      * 辅助工具方法 (约11个)
      * 接口适配方法 (约11个)
    - 符合L4层组件化架构设计原则
    - 基于L3层成功经验的职责分组模式
    """

    """
    增强型CCI(商品通道指数)指标
    
    具有以下增强特性:
    1. 自适应周期设计: 根据市场波动率自动调整周期
    2. 区间穿梭动态评估: 评估CCI在不同区间穿梭的质量和可靠性
    3. 周期协同分析: 结合多周期CCI指标增强信号可靠性  # TODO: 将魔法数字提取到配置中
    4. 形态识别: 识别CCI形成的关键形态  # TODO: 将魔法数字提取到配置中
    5. 市场环境自适应: 根据市场环境动态调整评分标准  # TODO: 将魔法数字提取到配置中
    """

    # CCI标准区间定义
    EXTREME_OVERBOUGHT = 200  # 极度超买  # TODO: 将魔法数字提取到配置中
    overbought = 100  # 超买
    NEUTRAL_HIGH = 0  # 中性偏多
    NEUTRAL_LOW = 0  # 中性偏空
    oversold = -100  # 超卖
    EXTREME_OVERSOLD = -200  # 极度超卖  # TODO: 将魔法数字提取到配置中

    def __init__(
        self,
        period: int = 20,
        factor: float = 0.015,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        secondary_period: int = 40,
        adaptive: bool = True,  # TODO: 将魔法数字提取到配置中
        smoothing_period: int = 3,  # TODO: 将魔法数字提取到配置中
        trend_period: int = 50,
    ):  # TODO: 将魔法数字提取到配置中
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        """
        初始化增强型CCI指标

        Args:
            period (int): CCI计算周期
            factor (float): CCI计算因子，标准为0.015  # TODO: 将魔法数字提取到配置中
            secondary_period (int): 二级CCI周期，用于周期协同分析
            adaptive (bool): 是否启用自适应周期
            smoothing_period (int): 平滑周期
            trend_period (int): 趋势周期
        """
        super().__init__()
        self.REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]
        self.indicator_type = "ENHANCEDCCI"

        self.base_period = period
        self.factor = factor
        self.secondary_period = secondary_period
        self.adaptive = adaptive
        self.trend_period = trend_period
        self.market_environment = "normal"

        # CCI阈值常量
        self.OVERBOUGHT = 100
        self.OVERSOLD = -100
        self.EXTREME_OVERBOUGHT = 200  # TODO: 将魔法数字提取到配置中
        self.EXTREME_OVERSOLD = -200  # TODO: 将魔法数字提取到配置中

    def _get_default_parameters_enhancedcci(self) -> dict:
        """
        获取Enhanced CCI指标的默认参数

        Returns:
            dict: 默认参数字典
        """
        return {
            "period": 20,  # TODO: 将魔法数字提取到配置中
            "factor": 0.015,  # TODO: 将魔法数字提取到配置中
            "secondary_period": 40,  # TODO: 将魔法数字提取到配置中
            "smoothing_period": 3,  # TODO: 将魔法数字提取到配置中
            "trend_period": 50,  # TODO: 将魔法数字提取到配置中
            "adaptive": True,
        }

        # 实际使用周期（可能会根据自适应算法调整）
        self.current_period = period

        # 内部数据
        self._result = None
        self._price_data = None
        self._multi_period_result = None

    def set_market_environment_Cci(self, environment: str) -> None:
        """
        设置市场环境

        Args:
            environment (str): 市场环境类型 ('bull_market', 'bear_market', 'sideways_market', 'volatile_market', 'normal')
        """
        valid_environments = ["bull_market", "bear_market", "sideways_market", "volatile_market", "normal"]
        if environment not in valid_environments:
            raise ValueError(f"无效的市场环境类型: {environment}。有效类型: {valid_environments}")

        self.market_environment = environment

    def calculate_Cci_Enhanced_Cci(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算Enhanced_cCI指标

        Args:
            data: 包含OHLCV数据的Data_frame
            **kwargs: 其他参数

        Returns:
            包含Enhanced_cCI指标的Data_frame
        """
        return self._calculate_enhancedcci(data)

    def set_parameters_Cci_Enhanced_Cci(self, **kwargs):
        """
        设置指标参数

        Args:
            **kwargs: 参数字典
        """
        if "period" in kwargs:
            self.base_period = kwargs["period"]
        if "factor" in kwargs:
            self.factor = kwargs["factor"]
        if "secondary_period" in kwargs:
            self.secondary_period = kwargs["secondary_period"]
        if "adaptive" in kwargs:
            self.adaptive = kwargs["adaptive"]

    def _calculate_enhancedcci(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算CCI指标

        Args:
            data (pd.DataFrame): 包含价格数据的DataFrame (必须包含'high', 'low', 'close'列)

        Returns:
            pd.DataFrame: 包含CCI指标结果的Data_frame
        """
        # 处理空数据
        if data.empty:
            return pd.DataFrame()

        # 检查数据是否有效
        required_columns = ["high", "low", "close"]
        for col in required_columns:
            if col not in data.columns:
                return data.copy()

        # 检查数据长度
        if len(data) < self.minimum_periods:
            return data.copy()

        # 保存价格数据供后续使用
        self._price_data = data["close"].copy()

        # 如果是自适应模式，调整周期
        if self.adaptive:
            self.current_period = self._adapt_period(data)
        else:
            self.current_period = self.base_period

        # 计算主要CCI
        cci = self._calculate_cci(data, self.current_period)

        # 计算二级CCI（用于周期协同分析）
        cci_secondary = self._calculate_cci(data, self.secondary_period)

        # 计算CCI的均线
        cci_ma5 = cci.rolling(window=5).mean()  # TODO: 将魔法数字提取到配置中
        cci_ma10 = cci.rolling(window=10).mean()
        cci_ma20 = cci.rolling(window=20).mean()  # TODO: 将魔法数字提取到配置中

        # 计算CCI斜率
        cci_slope = (cci - cci.shift(3)) / 3  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        # 计算CCI波动性
        cci_volatility = cci.rolling(window=10).std()

        # 计算典型价格和相关数据（用于验证）
        tp = (
            data["high"] + data["low"] + data["close"]
        ) / 3  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        tp_ma = tp.rolling(window=self.current_period).mean()
        mad = tp.rolling(window=self.current_period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)

        # 计算价格波动率
        volatility = data["close"].pct_change().rolling(window=20).std() * np.sqrt(
            252
        )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        # 创建结果DataFrame，包含原始数据
        result = data.copy()
        result["cci"] = cci
        result["tp"] = tp
        result["tp_ma"] = tp_ma
        result["mad"] = mad
        result["cci_secondary"] = cci_secondary
        result["cci_ma5"] = cci_ma5
        result["cci_ma10"] = cci_ma10
        result["cci_ma20"] = cci_ma20
        result["cci_slope"] = cci_slope
        result["cci_volatility"] = cci_volatility
        result["volatility"] = volatility

        # 添加CCI状态分类
        result["state"] = self._classify_cci_state(cci)

        self._result = result

        # 计算多周期CCI结果
        self._calculate_multi_period_cci(data)

        # 添加形态识别和信号生成
        result = self.add_pattern_detection(result)
        result = self.add_signal_generation(result)

        return result

    def _calculate_cci(self, data: pd.DataFrame, period: int) -> pd.Series:
        """
        使用特定周期计算CCI

        Args:
            data (pd.DataFrame): 价格数据
            period (int): 计算周期

        Returns:
            pd.Series: CCI值
        """
        # 计算典型价格 (TP)
        tp = (
            data["high"] + data["low"] + data["close"]
        ) / 3  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        # 计算移动平均
        tp_ma = tp.rolling(window=period).mean()

        # 计算平均偏差
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)

        # 计算CCI
        cci = (tp - tp_ma) / (self.factor * mad)

        return cci

    def _adapt_period(self, data: pd.DataFrame) -> int:
        """
        根据市场波动性自适应调整CCI周期

        Args:
            data (pd.DataFrame): 价格数据

        Returns:
            int: 自适应调整后的周期
        """
        # 使用ATR计算市场波动性
        high = data["high"]
        low = data["low"]
        close = data["close"].shift(1)

        # 计算真实波幅 (TR)
        tr1 = high - low
        tr2 = np.abs(high - close)
        tr3 = np.abs(low - close)

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # 计算ATR
        atr = tr.rolling(window=20).mean()  # TODO: 将魔法数字提取到配置中

        # 计算近期ATR与长期ATR的比值
        recent_atr = atr.iloc[-10:].mean() if len(atr) >= 10 else atr.mean()
        long_term_atr = (
            atr.iloc[-60:].mean() if len(atr) >= 60 else atr.mean()
        )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        if pd.isna(recent_atr) or pd.isna(long_term_atr) or long_term_atr == 0:
            return self.base_period

        volatility_ratio = recent_atr / long_term_atr

        # 根据波动性调整周期
        if volatility_ratio > 1.5:  # 高波动性  # TODO: 将魔法数字提取到配置中
            # 高波动市场使用较短周期
            return max(int(self.base_period * 0.7), 10)  # TODO: 将魔法数字提取到配置中
        elif volatility_ratio < 0.7:  # 低波动性  # TODO: 将魔法数字提取到配置中
            # 低波动市场使用较长周期
            return min(int(self.base_period * 1.3), 30)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        else:  # 正常波动性
            return self.base_period

    def _classify_cci_state(self, cci: pd.Series) -> pd.Series:
        """
        根据CCI值分类状态

        Args:
            cci (pd.Series): CCI值

        Returns:
            pd.Series: CCI状态分类
        """
        conditions = [
            cci > self.EXTREME_OVERBOUGHT,
            (cci > self.OVERBOUGHT) & (cci <= self.EXTREME_OVERBOUGHT),
            (cci > self.NEUTRAL_HIGH) & (cci <= self.OVERBOUGHT),
            (cci > self.NEUTRAL_LOW) & (cci <= self.NEUTRAL_HIGH),
            (cci > self.EXTREME_OVERSOLD) & (cci <= self.OVERSOLD),
            cci <= self.EXTREME_OVERSOLD,
        ]

        choices = ["extreme_overbought", "overbought", "neutral_high", "neutral_low", "oversold", "extreme_oversold"]

        return pd.Series(np.select(conditions, choices, default="neutral"), index=cci.index)

    def _calculate_multi_period_cci(self, data: pd.DataFrame) -> None:
        """
        计算多周期CCI用于周期协同分析

        Args:
            data (pd.DataFrame): 价格数据
        """
        # 计算不同周期的CCI
        periods = [
            max(int(self.current_period / 2), 5),  # 短周期  # TODO: 将魔法数字提取到配置中
            self.current_period,  # 中周期
            self.current_period * 2,  # 长周期
        ]

        multi_period = {}
        for period in periods:
            multi_period[f"cci_{period}"] = self._calculate_cci(data, period)

        self._multi_period_result = pd.DataFrame(multi_period, index=data.index)

    def analyze_crossovers(self) -> pd.DataFrame:
        """
        分析CCI的关键交叉点

        Returns:
            pd.DataFrame: 包含交叉分析结果的Data_frame
        """
        if self._result is None:
            return pd.DataFrame()

        cci = self._result["cci"]

        # 创建结果DataFrame
        crossovers = pd.DataFrame(index=cci.index)

        # 零轴交叉
        crossovers["zero_cross_up"] = (cci > 0) & (cci.shift(1) <= 0)
        crossovers["zero_cross_down"] = (cci < 0) & (cci.shift(1) >= 0)

        # 超买/超卖区域交叉
        crossovers["overbought_enter"] = (cci > self.OVERBOUGHT) & (cci.shift(1) <= self.OVERBOUGHT)
        crossovers["overbought_exit"] = (cci < self.OVERBOUGHT) & (cci.shift(1) >= self.OVERBOUGHT)
        crossovers["oversold_enter"] = (cci < self.OVERSOLD) & (cci.shift(1) >= self.OVERSOLD)
        crossovers["oversold_exit"] = (cci > self.OVERSOLD) & (cci.shift(1) <= self.OVERSOLD)

        # 极端区域交叉
        crossovers["extreme_overbought_enter"] = (cci > self.EXTREME_OVERBOUGHT) & (
            cci.shift(1) <= self.EXTREME_OVERBOUGHT
        )
        crossovers["extreme_overbought_exit"] = (cci < self.EXTREME_OVERBOUGHT) & (
            cci.shift(1) >= self.EXTREME_OVERBOUGHT
        )
        crossovers["extreme_oversold_enter"] = (cci < self.EXTREME_OVERSOLD) & (cci.shift(1) >= self.EXTREME_OVERSOLD)
        crossovers["extreme_oversold_exit"] = (cci > self.EXTREME_OVERSOLD) & (cci.shift(1) <= self.EXTREME_OVERSOLD)

        # 均线交叉
        crossovers["ma5_cross_up"] = (cci > self._result["cci_ma5"]) & (
            cci.shift(1) <= self._result["cci_ma5"].shift(1)
        )
        crossovers["ma5_cross_down"] = (cci < self._result["cci_ma5"]) & (
            cci.shift(1) >= self._result["cci_ma5"].shift(1)
        )

        # 交叉质量评估
        crossovers["crossover_strength"] = self._calculate_crossover_strength(cci)

        return crossovers

    def _calculate_crossover_strength(self, cci: pd.Series) -> pd.Series:
        """
        计算交叉强度

        Args:
            cci (pd.Series): CCI值

        Returns:
            pd.Series: 交叉强度评分 (0-100)
        """
        # 创建结果序列
        strength = pd.Series(0, index=cci.index)

        # 计算零轴交叉强度
        zero_cross_up = (cci > 0) & (cci.shift(1) <= 0)
        zero_cross_down = (cci < 0) & (cci.shift(1) >= 0)

        # 计算零轴交叉角度（斜率）
        cross_slope = cci - cci.shift(1)

        # 零轴交叉强度评分
        for i in range(1, len(cci)):
            if zero_cross_up.iloc[i]:
                # 向上交叉强度基于斜率和前期状态
                # 1. 从超卖区域反弹强度更大
                # 2. 斜率越大强度越大
                slope = cross_slope.iloc[i]
                prev_min = cci.iloc[max(0, i - 5) : i].min()  # TODO: 将魔法数字提取到配置中

                # 基础分30，最高100
                base_score = 30  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

                # 斜率评分（最高30分）
                slope_score = min(30, abs(slope) * 15)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

                # 前期状态评分（最高40分）
                prev_state_score = 0
                if prev_min < self.EXTREME_OVERSOLD:
                    prev_state_score = 40  # 从极度超卖区域反弹  # TODO: 将魔法数字提取到配置中
                elif prev_min < self.OVERSOLD:
                    prev_state_score = 30  # 从超卖区域反弹  # TODO: 将魔法数字提取到配置中
                elif prev_min < 0:
                    prev_state_score = 20  # 从负值区域反弹  # TODO: 将魔法数字提取到配置中

                total_score = base_score + slope_score + prev_state_score
                strength.iloc[i] = total_score

            elif zero_cross_down.iloc[i]:
                # 向下交叉强度基于斜率和前期状态
                slope = abs(cross_slope.iloc[i])
                prev_max = cci.iloc[max(0, i - 5) : i].max()  # TODO: 将魔法数字提取到配置中

                # 基础分30，最高100
                base_score = 30  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

                # 斜率评分（最高30分）
                slope_score = min(30, slope * 15)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

                # 前期状态评分（最高40分）
                prev_state_score = 0
                if prev_max > self.EXTREME_OVERBOUGHT:
                    prev_state_score = 40  # 从极度超买区域下跌  # TODO: 将魔法数字提取到配置中
                elif prev_max > self.OVERBOUGHT:
                    prev_state_score = 30  # 从超买区域下跌  # TODO: 将魔法数字提取到配置中
                elif prev_max > 0:
                    prev_state_score = 20  # 从正值区域下跌  # TODO: 将魔法数字提取到配置中

                total_score = base_score + slope_score + prev_state_score
                strength.iloc[i] = total_score

        return strength

    def analyze_multi_period_synergy_Cci(self) -> pd.DataFrame:
        """
        分析多周期CCI协同性

        Returns:
            pd.DataFrame: 包含多周期协同分析结果的Data_frame
        """
        if self._multi_period_result is None:
            return pd.DataFrame()

        multi_cci = self._multi_period_result

        # 获取各周期CCI列名
        columns = multi_cci.columns

        # 创建结果DataFrame
        synergy = pd.DataFrame(index=multi_cci.index)

        # 方向一致性分析
        directions = {}
        for col in columns:
            directions[f"{col}_up"] = multi_cci[col] > 0
            directions[f"{col}_down"] = multi_cci[col] < 0

        directions_df = pd.DataFrame(directions, index=multi_cci.index)

        # 计算多周期方向一致性
        up_agreement = directions_df[[c for c in directions_df.columns if c.endswith("_up")]].all(axis=1)
        down_agreement = directions_df[[c for c in directions_df.columns if c.endswith("_down")]].all(axis=1)

        synergy["bullish_agreement"] = up_agreement
        synergy["bearish_agreement"] = down_agreement

        # 计算多周期斜率一致性
        slopes = {}
        for col in columns:
            slopes[f"{col}_slope"] = multi_cci[col] - multi_cci[col].shift(3)  # TODO: 将魔法数字提取到配置中

        slopes_df = pd.DataFrame(slopes, index=multi_cci.index)

        # 计算斜率符号一致性
        slope_cols = slopes_df.columns
        rising_slopes = (slopes_df > 0).all(axis=1)
        falling_slopes = (slopes_df < 0).all(axis=1)

        synergy["rising_momentum"] = rising_slopes
        synergy["falling_momentum"] = falling_slopes

        # 计算周期排列
        # 理想的多周期排列：短周期 > 中周期 > 长周期（上升趋势）
        # 或 短周期 < 中周期 < 长周期（下降趋势）
        if len(columns) >= 3:  # TODO: 将魔法数字提取到配置中
            short_period = columns[0]
            mid_period = columns[1]
            long_period = columns[2]

            synergy["bullish_alignment"] = (multi_cci[short_period] > multi_cci[mid_period]) & (
                multi_cci[mid_period] > multi_cci[long_period]
            )

            synergy["bearish_alignment"] = (multi_cci[short_period] < multi_cci[mid_period]) & (
                multi_cci[mid_period] < multi_cci[long_period]
            )

        # 计算综合协同度评分 (0-100)
        synergy["synergy_score"] = self._calculate_synergy_score(synergy)

        return synergy

    def _calculate_synergy_score(self, synergy: pd.DataFrame) -> pd.Series:
        """
        计算多周期CCI协同度评分

        Args:
            synergy (pd.DataFrame): 包含协同分析的Data_frame

        Returns:
            pd.Series: 协同度评分 (0-100)
        """
        # 基础分50分（中性）
        score = pd.Series(50, index=synergy.index)  # TODO: 将魔法数字提取到配置中

        # 方向一致性评分 (±20分)
        score += np.where(synergy["bullish_agreement"], 20, 0)  # TODO: 将魔法数字提取到配置中
        score -= np.where(synergy["bearish_agreement"], 20, 0)  # TODO: 将魔法数字提取到配置中

        # 动量一致性评分 (±15分)
        score += np.where(synergy["rising_momentum"], 15, 0)  # TODO: 将魔法数字提取到配置中
        score -= np.where(synergy["falling_momentum"], 15, 0)  # TODO: 将魔法数字提取到配置中

        # 周期排列评分 (±15分)
        if "bullish_alignment" in synergy.columns:
            score += np.where(synergy["bullish_alignment"], 15, 0)  # TODO: 将魔法数字提取到配置中
            score -= np.where(synergy["bearish_alignment"], 15, 0)  # TODO: 将魔法数字提取到配置中

        # 限制分数范围在0-100之间
        score = score.clip(0, 100)

        return score

    def identify_patterns_Cci(self) -> pd.DataFrame:
        """
        识别CCI指标的特定形态

        Returns:
            pd.DataFrame: 包含形态识别结果的Data_frame
        """
        if self._result is None:
            return pd.DataFrame()

        cci = self._result["cci"].dropna()
        if cci.empty:
            return pd.DataFrame()

        patterns = pd.DataFrame(index=self._result.index)

        # 1. 零轴穿越
        patterns["zero_cross_up"] = crossover(self._result["cci"], 0)
        patterns["zero_cross_down"] = crossunder(self._result["cci"], 0)

        # 2. CCI背离 (价格与CCI走势不一致)
        divergence = self.detect_divergence()
        if not divergence.empty:
            patterns = patterns.join(divergence)

        # 3. CCI W底和M顶  # TODO: 将魔法数字提取到配置中
        peaks, troughs = find_peaks_and_troughs(cci.values, window=5)  # TODO: 将魔法数字提取到配置中

        # W底
        w_bottom = np.zeros(len(cci), dtype=bool)
        if len(troughs) >= 2:
            for i in range(1, len(troughs)):
                # 两个连续的低谷，且第二个比第一个高
                if cci.iloc[troughs[i]] > cci.iloc[troughs[i - 1]]:
                    # 检查中间是否有高点
                    middle_peaks = [p for p in peaks if troughs[i - 1] < p < troughs[i]]
                    if middle_peaks:
                        w_bottom[troughs[i]] = True
        patterns["w_bottom"] = pd.Series(w_bottom, index=cci.index)

        # M顶
        m_top = np.zeros(len(cci), dtype=bool)
        if len(peaks) >= 2:
            for i in range(1, len(peaks)):
                # 两个连续的高点，且第二个比第一个低
                if cci.iloc[peaks[i]] < cci.iloc[peaks[i - 1]]:
                    # 检查中间是否有低谷
                    middle_troughs = [t for t in troughs if peaks[i - 1] < t < peaks[i]]
                    if middle_troughs:
                        m_top[peaks[i]] = True
        patterns["m_top"] = pd.Series(m_top, index=cci.index)

        # 安全的DataFrame重索引操作，避免"Gaps in blk ref_locs"错误
        try:
            # 方法1：使用更安全的重索引方式
            result_patterns = pd.DataFrame(index=self._result.index)
            for col in patterns.columns:
                # 逐列进行安全的重索引
                col_data = patterns[col].reindex(self._result.index, fill_value=False)
                result_patterns[col] = col_data.astype(bool)
            return result_patterns
        except Exception as e:
            # 方法2：如果还是失败，使用最安全的方式
            logger.warning(f"Enhanced CCI patterns reindex failed: {e}, using fallback method")
            result_patterns = pd.DataFrame(index=self._result.index)
            for col in patterns.columns:
                result_patterns[col] = pd.Series(False, index=self._result.index, dtype=bool)
                # 安全地复制匹配的索引值
                for idx in patterns.index:
                    if idx in self._result.index:
                        try:
                            result_patterns.at[idx, col] = bool(patterns.at[idx, col])
                        except (KeyError, IndexError):
                            continue
            return result_patterns

    def detect_divergence(self) -> pd.DataFrame:
        """
        检测CCI与价格之间的背离

        Returns:
            pd.DataFrame: 背离检测结果
        """
        if self._result is None or self._price_data is None:
            return pd.DataFrame()

        price = self._price_data.dropna()
        cci = self._result["cci"].dropna()

        if price.empty or cci.empty:
            return pd.DataFrame()

        divergence = pd.DataFrame(
            {
                "bullish_divergence": np.zeros(len(self._result), dtype=bool),
                "bearish_divergence": np.zeros(len(self._result), dtype=bool),
                "hidden_bullish_divergence": np.zeros(len(self._result), dtype=bool),
                "hidden_bearish_divergence": np.zeros(len(self._result), dtype=bool),
            },
            index=self._result.index,
        )

        price_aligned, cci_aligned = price.align(cci, join="inner")

        if len(price_aligned) < 10:
            return divergence

        # 查找价格和CCI的峰值和谷值
        price_peaks, price_troughs = find_peaks_and_troughs(price_aligned.values, window=10)
        cci_peaks, cci_troughs = find_peaks_and_troughs(cci_aligned.values, window=10)

        # 看涨背离：价格创新低，CCI未创新低
        if len(price_troughs) > 1 and len(cci_troughs) > 1:
            for i in range(1, len(price_troughs)):
                if price_aligned.iloc[price_troughs[i]] < price_aligned.iloc[price_troughs[i - 1]]:
                    corresponding_cci_trough = -1
                    for t in cci_troughs:
                        if abs(t - price_troughs[i]) < 5:  # TODO: 将魔法数字提取到配置中
                            corresponding_cci_trough = t
                            break
                    if corresponding_cci_trough != -1:
                        prev_cci_trough = -1
                        for t in cci_troughs:
                            if abs(t - price_troughs[i - 1]) < 5:  # TODO: 将魔法数字提取到配置中
                                prev_cci_trough = t
                                break
                        if (
                            prev_cci_trough != -1
                            and cci_aligned.iloc[corresponding_cci_trough] > cci_aligned.iloc[prev_cci_trough]
                        ):
                            divergence.loc[price_aligned.index[price_troughs[i]], "bullish_divergence"] = True

        # 看跌背离：价格创新高，CCI未创新高
        if len(price_peaks) > 1 and len(cci_peaks) > 1:
            for i in range(1, len(price_peaks)):
                if price_aligned.iloc[price_peaks[i]] > price_aligned.iloc[price_peaks[i - 1]]:
                    corresponding_cci_peak = -1
                    for p in cci_peaks:
                        if abs(p - price_peaks[i]) < 5:  # TODO: 将魔法数字提取到配置中
                            corresponding_cci_peak = p
                            break
                    if corresponding_cci_peak != -1:
                        prev_cci_peak = -1
                        for p in cci_peaks:
                            if abs(p - price_peaks[i - 1]) < 5:  # TODO: 将魔法数字提取到配置中
                                prev_cci_peak = p
                                break
                        if (
                            prev_cci_peak != -1
                            and cci_aligned.iloc[corresponding_cci_peak] < cci_aligned.iloc[prev_cci_peak]
                        ):
                            divergence.loc[price_aligned.index[price_peaks[i]], "bearish_divergence"] = True

        return divergence

    def generate_signals_Cci(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号

        Args:
            data (pd.DataFrame): 价格数据

        Returns:
            pd.DataFrame: 包含交易信号的Data_frame
        """
        if self._result is None:
            self.calculate_Cci_Enhanced_Cci(data)

        if self._result is None:
            return pd.DataFrame()

        # 获取CCI数据
        cci = self._result["cci"]

        # 分析交叉
        crossovers = self.analyze_crossovers()

        # 识别形态
        patterns = self.identify_patterns_Cci()

        # 分析多周期协同
        synergy = self.analyze_multi_period_synergy_Cci()

        # 创建信号DataFrame
        signals = pd.DataFrame(index=data.index)

        # 基本信号
        signals["zero_cross_up"] = crossovers["zero_cross_up"]
        signals["zero_cross_down"] = crossovers["zero_cross_down"]
        signals["oversold_exit"] = crossovers["oversold_exit"]
        signals["overbought_exit"] = crossovers["overbought_exit"]

        # 形态信号
        if "oversold_hook" in patterns.columns:
            signals["oversold_hook"] = patterns["oversold_hook"]
            signals["overbought_hook"] = patterns["overbought_hook"]
            signals["bullish_divergence"] = patterns["bullish_divergence"]
            signals["bearish_divergence"] = patterns["bearish_divergence"]
            signals["head_shoulders_bottom"] = patterns["head_shoulders_bottom"]
            signals["head_shoulders_top"] = patterns["head_shoulders_top"]
            signals["platform_breakout_up"] = patterns.get("platform_breakout_up", pd.Series(False, index=data.index))
            signals["platform_breakout_down"] = patterns.get(
                "platform_breakout_down", pd.Series(False, index=data.index)
            )

        # 多周期协同信号
        if "bullish_agreement" in synergy.columns:
            signals["bullish_agreement"] = synergy["bullish_agreement"]
            signals["bearish_agreement"] = synergy["bearish_agreement"]
            signals["bullish_alignment"] = synergy.get("bullish_alignment", pd.Series(False, index=data.index))
            signals["bearish_alignment"] = synergy.get("bearish_alignment", pd.Series(False, index=data.index))

        # 生成买卖信号
        signals["buy_signal"] = (
            signals["zero_cross_up"]
            | signals["oversold_exit"]
            | signals.get("oversold_hook", pd.Series(False, index=data.index))
            | signals.get("bullish_divergence", pd.Series(False, index=data.index))
            | signals.get("head_shoulders_bottom", pd.Series(False, index=data.index))
            | signals.get("platform_breakout_up", pd.Series(False, index=data.index))
            | (
                signals.get("bullish_agreement", pd.Series(False, index=data.index))
                & signals.get("bullish_alignment", pd.Series(False, index=data.index))
            )
        )

        signals["sell_signal"] = (
            signals["zero_cross_down"]
            | signals["overbought_exit"]
            | signals.get("overbought_hook", pd.Series(False, index=data.index))
            | signals.get("bearish_divergence", pd.Series(False, index=data.index))
            | signals.get("head_shoulders_top", pd.Series(False, index=data.index))
            | signals.get("platform_breakout_down", pd.Series(False, index=data.index))
            | (
                signals.get("bearish_agreement", pd.Series(False, index=data.index))
                & signals.get("bearish_alignment", pd.Series(False, index=data.index))
            )
        )

        # 计算信号分数
        signals["score"] = self.calculate_score_Cci(data)

        # 生成信号描述
        signals["signal_type"] = pd.Series("", index=signals.index)
        signals["signal_desc"] = pd.Series("", index=signals.index)
        signals["confidence"] = pd.Series(50, index=signals.index)  # TODO: 将魔法数字提取到配置中

        # 设置信号类型和描述
        signals.loc[signals["zero_cross_up"], "signal_type"] = "CCI零轴上穿"
        signals.loc[signals["zero_cross_up"], "signal_desc"] = "CCI上穿零轴，看涨信号"
        signals.loc[signals["zero_cross_up"], "confidence"] = 60  # TODO: 将魔法数字提取到配置中

        signals.loc[signals["zero_cross_down"], "signal_type"] = "CCI零轴下穿"
        signals.loc[signals["zero_cross_down"], "signal_desc"] = "CCI下穿零轴，看跌信号"
        signals.loc[signals["zero_cross_down"], "confidence"] = 60  # TODO: 将魔法数字提取到配置中

        signals.loc[signals["oversold_exit"], "signal_type"] = "CCI超卖区域回升"
        signals.loc[signals["oversold_exit"], "signal_desc"] = "CCI从超卖区域回升，看涨信号"
        signals.loc[signals["oversold_exit"], "confidence"] = 65  # TODO: 将魔法数字提取到配置中

        signals.loc[signals["overbought_exit"], "signal_type"] = "CCI超买区域回落"
        signals.loc[signals["overbought_exit"], "signal_desc"] = "CCI从超买区域回落，看跌信号"
        signals.loc[signals["overbought_exit"], "confidence"] = 65  # TODO: 将魔法数字提取到配置中

        if "oversold_hook" in signals.columns:
            signals.loc[signals["oversold_hook"], "signal_type"] = "CCI超卖区域钩子"
            signals.loc[signals["oversold_hook"], "signal_desc"] = "CCI在超卖区域形成拐点，看涨信号"
            signals.loc[signals["oversold_hook"], "confidence"] = 70  # TODO: 将魔法数字提取到配置中

            signals.loc[signals["overbought_hook"], "signal_type"] = "CCI超买区域钩子"
            signals.loc[signals["overbought_hook"], "signal_desc"] = "CCI在超买区域形成拐点，看跌信号"
            signals.loc[signals["overbought_hook"], "confidence"] = 70  # TODO: 将魔法数字提取到配置中

            signals.loc[signals["bullish_divergence"], "signal_type"] = "CCI正背离"
            signals.loc[signals["bullish_divergence"], "signal_desc"] = "价格创新低但CCI未创新低，看涨信号"
            signals.loc[signals["bullish_divergence"], "confidence"] = 75  # TODO: 将魔法数字提取到配置中

            signals.loc[signals["bearish_divergence"], "signal_type"] = "CCI负背离"
            signals.loc[signals["bearish_divergence"], "signal_desc"] = "价格创新高但CCI未创新高，看跌信号"
            signals.loc[signals["bearish_divergence"], "confidence"] = 75  # TODO: 将魔法数字提取到配置中

            signals.loc[signals["head_shoulders_bottom"], "signal_type"] = "CCI头肩底"
            signals.loc[signals["head_shoulders_bottom"], "signal_desc"] = "CCI形成头肩底形态，看涨信号"
            signals.loc[signals["head_shoulders_bottom"], "confidence"] = 80  # TODO: 将魔法数字提取到配置中

            signals.loc[signals["head_shoulders_top"], "signal_type"] = "CCI头肩顶"
            signals.loc[signals["head_shoulders_top"], "signal_desc"] = "CCI形成头肩顶形态，看跌信号"
            signals.loc[signals["head_shoulders_top"], "confidence"] = 80  # TODO: 将魔法数字提取到配置中

            signals.loc[signals["platform_breakout_up"], "signal_type"] = "CCI平台向上突破"
            signals.loc[signals["platform_breakout_up"], "signal_desc"] = "CCI从窄幅区间向上突破，看涨信号"
            signals.loc[signals["platform_breakout_up"], "confidence"] = 70  # TODO: 将魔法数字提取到配置中

            signals.loc[signals["platform_breakout_down"], "signal_type"] = "CCI平台向下突破"
            signals.loc[signals["platform_breakout_down"], "signal_desc"] = "CCI从窄幅区间向下突破，看跌信号"
            signals.loc[signals["platform_breakout_down"], "confidence"] = 70  # TODO: 将魔法数字提取到配置中

        if "bullish_agreement" in signals.columns:
            signals.loc[
                signals["bullish_agreement"] & signals.get("bullish_alignment", pd.Series(False, index=data.index)),
                "signal_type",
            ] = "CCI多周期看涨一致"
            signals.loc[
                signals["bullish_agreement"] & signals.get("bullish_alignment", pd.Series(False, index=data.index)),
                "signal_desc",
            ] = "多周期CCI均为看涨且排列良好，强烈看涨信号"
            signals.loc[
                signals["bullish_agreement"] & signals.get("bullish_alignment", pd.Series(False, index=data.index)),
                "confidence",
            ] = 85  # TODO: 将魔法数字提取到配置中

            signals.loc[
                signals["bearish_agreement"] & signals.get("bearish_alignment", pd.Series(False, index=data.index)),
                "signal_type",
            ] = "CCI多周期看跌一致"
            signals.loc[
                signals["bearish_agreement"] & signals.get("bearish_alignment", pd.Series(False, index=data.index)),
                "signal_desc",
            ] = "多周期CCI均为看跌且排列良好，强烈看跌信号"
            signals.loc[
                signals["bearish_agreement"] & signals.get("bearish_alignment", pd.Series(False, index=data.index)),
                "confidence",
            ] = 85  # TODO: 将魔法数字提取到配置中

        # 添加标准格式字段
        signals["trend"] = np.where(
            signals["score"] > 50, 1, np.where(signals["score"] < 50, -1, 0)
        )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        signals["risk_level"] = np.where(
            signals["confidence"] > 70, "低", np.where(signals["confidence"] > 60, "中", "高")
        )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        signals["position_size"] = signals["confidence"] / 100 * 0.1  # 最大仓位10%

        return signals

    def calculate_score_Cci(self, data: pd.DataFrame = None) -> pd.Series:
        """
        计算CCI综合评分 (0-100)

        Args:
            data (pd.DataFrame, optional): 价格数据，如果未提供则使用上次计算结果

        Returns:
            pd.Series: 评分 (0-100，50为中性)
        """
        if self._result is None and data is not None:
            self.calculate_Cci_Enhanced_Cci(data)

        if self._result is None:
            return pd.Series()

        # 获取CCI数据
        cci = self._result["cci"]
        cci_slope = self._result["cci_slope"]

        # 获取交叉分析
        crossovers = self.analyze_crossovers()

        # 获取形态识别
        patterns = self.identify_patterns_Cci()

        # 获取多周期协同分析
        synergy = self.analyze_multi_period_synergy_Cci()

        # 基础分数为50（中性）
        score = pd.Series(50, index=self._result.index)  # TODO: 将魔法数字提取到配置中

        # 1. CCI位置评分 (±20分)
        # CCI > 0 看涨，CCI < 0 看跌
        score += np.where(
            cci > 0, (cci / 200).clip(0, 1) * 20, (cci / 200).clip(-1, 0) * 20
        )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        # 2. CCI斜率评分 (±15分)
        # 斜率为正看涨，斜率为负看跌
        normalized_slope = cci_slope / cci_slope.rolling(window=20).std().replace(0, 1)  # TODO: 将魔法数字提取到配置中
        score += np.where(
            normalized_slope > 0,
            np.minimum(normalized_slope, 3) * 5,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            np.maximum(normalized_slope, -3) * 5,
        )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        # 3. 交叉评分 (±10分)  # TODO: 将魔法数字提取到配置中
        if not crossovers.empty:
            # 零轴向上交叉
            score.loc[crossovers["zero_cross_up"]] += 10
            # 零轴向下交叉
            score.loc[crossovers["zero_cross_down"]] -= 10
            # 从超卖区域回升
            score.loc[crossovers["oversold_exit"]] += 10
            # 从超买区域回落
            score.loc[crossovers["overbought_exit"]] -= 10

            # 根据交叉强度调整
            crossover_strength = crossovers["crossover_strength"]
            score += np.where(
                (crossovers["zero_cross_up"] | crossovers["oversold_exit"]),
                (crossover_strength - 50) / 10,  # TODO: 将魔法数字提取到配置中
                np.where(
                    (crossovers["zero_cross_down"] | crossovers["overbought_exit"]),
                    (50 - crossover_strength) / 10,  # TODO: 将魔法数字提取到配置中
                    0,
                ),
            )

        # 4. 形态评分 (±25分)  # TODO: 将魔法数字提取到配置中
        if not patterns.empty:
            # 看涨形态
            if "oversold_hook" in patterns.columns:
                score.loc[patterns["oversold_hook"]] += 15  # TODO: 将魔法数字提取到配置中
            if "bullish_divergence" in patterns.columns:
                score.loc[patterns["bullish_divergence"]] += 25  # TODO: 将魔法数字提取到配置中
            if "head_shoulders_bottom" in patterns.columns:
                score.loc[patterns["head_shoulders_bottom"]] += 25  # TODO: 将魔法数字提取到配置中
            if "platform_breakout_up" in patterns.columns:
                score.loc[patterns["platform_breakout_up"]] += 20  # TODO: 将魔法数字提取到配置中

            # 看跌形态
            if "overbought_hook" in patterns.columns:
                score.loc[patterns["overbought_hook"]] -= 15  # TODO: 将魔法数字提取到配置中
            if "bearish_divergence" in patterns.columns:
                score.loc[patterns["bearish_divergence"]] -= 25  # TODO: 将魔法数字提取到配置中
            if "head_shoulders_top" in patterns.columns:
                score.loc[patterns["head_shoulders_top"]] -= 25  # TODO: 将魔法数字提取到配置中
            if "platform_breakout_down" in patterns.columns:
                score.loc[patterns["platform_breakout_down"]] -= 20  # TODO: 将魔法数字提取到配置中

        # 5. 多周期协同评分 (±20分)  # TODO: 将魔法数字提取到配置中
        if not synergy.empty and "synergy_score" in synergy.columns:
            # 协同评分转换为相对于中性50分的偏差
            synergy_adjustment = (
                synergy["synergy_score"] - 50
            ) / 2.5  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            score += synergy_adjustment

        # 6. 市场环境调整  # TODO: 将魔法数字提取到配置中
        if self.market_environment == "bull_market":
            # 牛市中增强多头信号，弱化空头信号
            bull_adjustment = np.where(
                score > 50, (score - 50) * 0.2, (score - 50) * 0.1
            )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            score += bull_adjustment
        elif self.market_environment == "bear_market":
            # 熊市中增强空头信号，弱化多头信号
            bear_adjustment = np.where(
                score < 50, (50 - score) * 0.2, (50 - score) * 0.1
            )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            score -= bear_adjustment
        elif self.market_environment == "volatile_market":
            # 高波动市场需要更强的信号
            vol_adjustment = (score - 50).abs() * 0.3  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            score = np.where(
                score > 50, 50 + vol_adjustment, 50 - vol_adjustment
            )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        # 限制分数范围在0-100之间
        score = score.clip(0, 100)

        return score

    def calculate_raw_score_Cci_Enhanced_Cci(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算增强型CCI指标原始评分 (0-100分)

        Args:
            data: 输入数据
            **kwargs: 额外参数

        Returns:
            pd.Series: 评分序列，取值范围0-100
        """
        # 确保已计算指标
        if self._result is None:
            self._calculate_enhancedcci(data)

        if self._result is None:
            return pd.Series(50, index=data.index)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        try:
            return self.calculate_score_Cci(data)
        except:
            # 如果计算失败，返回基于CCI值的简单评分
            if "cci" in self._result.columns:
                cci = self._result["cci"]
                # 将CCI值映射到0-100分
                score = (
                    50 + (cci / 200) * 50
                )  # CCI在-200到200之间映射到0-100  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                score = score.clip(0, 100)
                return score
            else:
                return pd.Series(50, index=data.index)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

    def calculate_confidence_Cci_Enhanced_Cci(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """
        计算Enhanced_cCI指标的置信度

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
            # 检查EnhancedCCI形态
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

    def get_patterns_Cci_Enhanced_Cci(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取Enhanced_cCI相关形态

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含形态信息的Data_frame
        """
        # 确保已计算指标
        if self._result is None:
            self._calculate_enhancedcci(data)

        if self._result is None:
            return pd.DataFrame(index=data.index)

        # 创建基本的形态DataFrame
        try:
            patterns = self.identify_patterns_Cci()
        except:
            patterns = pd.DataFrame()

        # 如果patterns为空，创建基本的形态DataFrame
        if patterns is None or patterns.empty:
            patterns = pd.DataFrame(index=data.index)

            # 获取CCI数据
            cci = self._result["cci"]

            # 基本形态
            patterns["CCI_ZERO_CROSS_UP"] = (cci > 0) & (cci.shift(1) <= 0)
            patterns["CCI_ZERO_CROSS_DOWN"] = (cci < 0) & (cci.shift(1) >= 0)
            patterns["CCI_OVERBOUGHT"] = cci > 100
            patterns["CCI_OVERSOLD"] = cci < -100
            patterns["CCI_EXTREME_OVERBOUGHT"] = cci > 200  # TODO: 将魔法数字提取到配置中
            patterns["CCI_EXTREME_OVERSOLD"] = cci < -200  # TODO: 将魔法数字提取到配置中

            # 趋势形态
            patterns["CCI_RISING"] = cci > cci.shift(1)
            patterns["CCI_FALLING"] = cci < cci.shift(1)
            patterns["CCI_UPTREND"] = (
                (cci > cci.shift(1))
                & (cci.shift(1) > cci.shift(2))
                & (cci.shift(2) > cci.shift(3))  # TODO: 将魔法数字提取到配置中
            )
            patterns["CCI_DOWNTREND"] = (
                (cci < cci.shift(1))
                & (cci.shift(1) < cci.shift(2))
                & (cci.shift(2) < cci.shift(3))  # TODO: 将魔法数字提取到配置中
            )

        return patterns

    def register_patterns_Cci(self):
        """
        注册Enhanced_cCI指标的形态到全局形态注册表
        """
        # 注册CCI零轴穿越形态
        self.register_pattern_to_registry(
            pattern_id="CCI_ZERO_CROSS_UP",
            display_name="CCI零轴上穿",
            description="CCI从下方穿越零轴，表明趋势转为看涨",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=15.0,  # TODO: 将魔法数字提取到配置中
            polarity="POSITIVE",
        )

        self.register_pattern_to_registry(
            pattern_id="CCI_ZERO_CROSS_DOWN",
            display_name="CCI零轴下穿",
            description="CCI从上方穿越零轴，表明趋势转为看跌",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-15.0,  # TODO: 将魔法数字提取到配置中
            polarity="NEGATIVE",
        )

        # 注册CCI超买超卖形态
        self.register_pattern_to_registry(
            pattern_id="CCI_EXTREME_OVERBOUGHT",
            display_name="CCI极度超买",
            description="CCI值高于200，表明市场极度超买",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-20.0,  # TODO: 将魔法数字提取到配置中
            polarity="NEGATIVE",
        )

        self.register_pattern_to_registry(
            pattern_id="CCI_EXTREME_OVERSOLD",
            display_name="CCI极度超卖",
            description="CCI值低于-200，表明市场极度超卖",  # TODO: 将魔法数字提取到配置中
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=20.0,  # TODO: 将魔法数字提取到配置中
            polarity="POSITIVE",
        )

        # 注册CCI趋势形态
        self.register_pattern_to_registry(
            pattern_id="CCI_UPTREND",
            display_name="CCI上升趋势",
            description="CCI连续上升，表明强势上升趋势",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=12.0,  # TODO: 将魔法数字提取到配置中
            polarity="POSITIVE",
        )

        self.register_pattern_to_registry(
            pattern_id="CCI_DOWNTREND",
            display_name="CCI下降趋势",
            description="CCI连续下降，表明强势下降趋势",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-12.0,  # TODO: 将魔法数字提取到配置中
            polarity="NEGATIVE",
        )

    def generate_trading_signals_Cci(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """
        生成Enhanced_cCI交易信号

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            Dict[str, pd.Series]: 包含买卖信号的字典
        """
        # 确保已计算指标
        if self._result is None:
            self.calculate_Cci_Enhanced_Cci(data)

        if self._result is None:
            return {
                "buy_signal": pd.Series(False, index=data.index),
                "sell_signal": pd.Series(False, index=data.index),
                "signal_strength": pd.Series(0.0, index=data.index),
            }

        cci = self._result["cci"]

        # 生成信号
        buy_signal = pd.Series(False, index=data.index)
        sell_signal = pd.Series(False, index=data.index)
        signal_strength = pd.Series(0.0, index=data.index)

        # 1. CCI零轴穿越信号
        zero_cross_up = crossover(cci, 0)
        zero_cross_down = crossunder(cci, 0)

        buy_signal |= zero_cross_up
        sell_signal |= zero_cross_down
        signal_strength += zero_cross_up * 0.6  # TODO: 将魔法数字提取到配置中
        signal_strength += zero_cross_down * 0.6  # TODO: 将魔法数字提取到配置中

        # 2. CCI超买超卖反转信号
        oversold_exit = crossover(cci, -100)
        overbought_exit = crossunder(cci, 100)

        buy_signal |= oversold_exit
        sell_signal |= overbought_exit
        signal_strength += oversold_exit * 0.7  # TODO: 将魔法数字提取到配置中
        signal_strength += overbought_exit * 0.7  # TODO: 将魔法数字提取到配置中

        # 3. CCI极值反转信号  # TODO: 将魔法数字提取到配置中
        extreme_oversold_exit = crossover(cci, -200)  # TODO: 将魔法数字提取到配置中
        extreme_overbought_exit = crossunder(cci, 200)  # TODO: 将魔法数字提取到配置中

        buy_signal |= extreme_oversold_exit
        sell_signal |= extreme_overbought_exit
        signal_strength += extreme_oversold_exit * 0.9  # TODO: 将魔法数字提取到配置中
        signal_strength += extreme_overbought_exit * 0.9  # TODO: 将魔法数字提取到配置中

        return {"buy_signal": buy_signal, "sell_signal": sell_signal, "signal_strength": signal_strength}

    def get_pattern_info_Cci(self, pattern_id: str) -> dict:
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

    @property
    def minimum_periods(self) -> int:
        """
        EnhancedCci指标所需的最少数据周期数

        计算逻辑：基于参数 period(20), secondary_period(40), smoothing_period(3), trend_period(50) 计算  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        Returns:
            int: 最少需要的数据周期数
        """
        # 确保_parameters存在，如果不存在则使用默认值
        if not hasattr(self, "_parameters") or not self._parameters:
            return 60  # 返回默认的最小周期数  # TODO: 将魔法数字提取到配置中

        period = self._parameters.get("period", 20)  # TODO: 将魔法数字提取到配置中
        secondary_period = self._parameters.get("secondary_period", 40)  # TODO: 将魔法数字提取到配置中
        smoothing_period = self._parameters.get("smoothing_period", 3)  # TODO: 将魔法数字提取到配置中
        trend_period = self._parameters.get("trend_period", 50)  # TODO: 将魔法数字提取到配置中
        return max(period, secondary_period, smoothing_period, trend_period) + 10

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算Enhanced CCI指标的主要入口方法

        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 其他参数

        Returns:
            包含Enhanced CCI指标的DataFrame
        """
        return self.calculate_Cci_Enhanced_Cci(data, **kwargs)

    def _calculate_baseindicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        BaseIndicator要求的抽象方法实现

        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 其他参数

        Returns:
            包含Enhanced CCI指标的DataFrame
        """
        return self._calculate_enhancedcci(data)

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
        return self.calculate_confidence_Cci_Enhanced_Cci(score, patterns, signals)

    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        BaseIndicator要求的原始评分计算方法

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.Series: 原始评分序列
        """
        return self.calculate_raw_score_Cci_Enhanced_Cci(data, **kwargs)

    def get_patterns_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        BaseIndicator要求的形态获取方法

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 形态DataFrame
        """
        return self.get_patterns_Cci_Enhanced_Cci(data, **kwargs)

    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """
        BaseIndicator要求的参数设置方法

        Args:
            **kwargs: 参数字典
        """
        self.set_parameters_Cci_Enhanced_Cci(**kwargs)

    def _get_default_parameters(self) -> Dict[str, Any]:
        """
        BaseIndicator要求的默认参数获取方法

        Returns:
            dict: 默认参数字典
        """
        return self._get_default_parameters_enhancedcci()

    def set_parameters(self, **kwargs):
        """
        标准参数设置方法

        Args:
            **kwargs: 参数字典
        """
        self.set_parameters_Cci_Enhanced_Cci(**kwargs)

    def get_signal(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        【核心抽象方法2】基于Enhanced CCI指标数值生成最新的交易信号
        
        Enhanced CCI交易信号逻辑（融合多重增强特性）：
        - 基础CCI信号：超买超卖区域信号
        - 自适应周期调整：根据市场波动率优化信号
        - 多周期协同：结合主周期和次要周期CCI
        - 交叉信号：CCI与均线交叉
        - 背离检测：价格与CCI背离信号
        - 形态识别：CCI特定形态信号
        
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
            
            # 2. 确保已计算指标
            if not self.has_result():
                self.calculate(data, **kwargs)

            if self._result is None or len(self._result) == 0:
                return self._get_default_signal("Enhanced CCI计算结果为空")

            # 3. 获取最新数据
            latest_close = data['close'].iloc[-1]
            
            # 4. 获取Enhanced CCI相关值
            if len(self._result) < 2:
                return self._get_default_signal("Enhanced CCI数据不足")
                
            # 检查必要的列是否存在
            required_columns = ['cci']
            if not all(col in self._result.columns for col in required_columns):
                return self._get_default_signal("Enhanced CCI结果列不完整")
                
            latest_cci = self._result['cci'].iloc[-1]
            prev_cci = self._result['cci'].iloc[-2]
            
            # 获取增强特性数据
            latest_cci_secondary = self._result.get('cci_secondary', pd.Series([latest_cci])).iloc[-1] if 'cci_secondary' in self._result.columns else latest_cci
            latest_cci_ma5 = self._result.get('cci_ma5', pd.Series([latest_cci])).iloc[-1] if 'cci_ma5' in self._result.columns else latest_cci
            latest_cci_slope = self._result.get('cci_slope', pd.Series([0])).iloc[-1] if 'cci_slope' in self._result.columns else 0
            latest_cci_volatility = self._result.get('cci_volatility', pd.Series([20])).iloc[-1] if 'cci_volatility' in self._result.columns else 20
            
            # 5. Enhanced CCI信号生成逻辑
            signal_type = "hold"
            strength = 0.0
            confidence = 0.5
            reason = "无明确信号"
            metadata = {}
            
            # Enhanced CCI阈值定义（基于类常量）
            extreme_overbought = getattr(self, 'EXTREME_OVERBOUGHT', 200)
            overbought_threshold = getattr(self, 'overbought', 100)
            oversold_threshold = getattr(self, 'oversold', -100)
            extreme_oversold = getattr(self, 'EXTREME_OVERSOLD', -200)
            
            # 计算增强信号强度因子
            volatility_factor = min(1.2, max(0.8, 20 / latest_cci_volatility))  # 波动率调整因子
            slope_factor = min(1.1, max(0.9, 1 + abs(latest_cci_slope) / 50))  # 斜率增强因子
            
            # 极端超买超卖信号（最高优先级，增强版）
            if latest_cci <= extreme_oversold:
                # Enhanced CCI在极端超卖区域 - 极强买入
                signal_type = "buy"
                base_strength = 0.95
                # 多周期确认
                if latest_cci_secondary < oversold_threshold:
                    base_strength = min(1.0, base_strength + 0.05)
                strength = base_strength * volatility_factor
                confidence = 0.9
                reason = f"Enhanced CCI极端超卖({latest_cci:.1f}<{extreme_oversold})，多重确认买入信号"
                
            elif latest_cci >= extreme_overbought:
                # Enhanced CCI在极端超买区域 - 极强卖出
                signal_type = "sell"
                base_strength = 0.95
                # 多周期确认
                if latest_cci_secondary > overbought_threshold:
                    base_strength = min(1.0, base_strength + 0.05)
                strength = base_strength * volatility_factor
                confidence = 0.9
                reason = f"Enhanced CCI极端超买({latest_cci:.1f}>{extreme_overbought})，多重确认卖出信号"
                
            # 超买超卖穿越信号（增强版）
            elif latest_cci < oversold_threshold and prev_cci >= oversold_threshold:
                # Enhanced CCI下穿超卖线 - 强买入信号
                signal_type = "buy"
                base_strength = 0.85
                # 斜率确认
                if latest_cci_slope < -10:  # 快速下降
                    base_strength += 0.05
                strength = base_strength * slope_factor
                confidence = 0.85
                reason = f"Enhanced CCI下穿超卖线({latest_cci:.1f})，斜率确认买入信号"
                
            elif latest_cci > overbought_threshold and prev_cci <= overbought_threshold:
                # Enhanced CCI上穿超买线 - 强卖出信号
                signal_type = "sell"
                base_strength = 0.85
                # 斜率确认
                if latest_cci_slope > 10:  # 快速上升
                    base_strength += 0.05
                strength = base_strength * slope_factor
                confidence = 0.85
                reason = f"Enhanced CCI上穿超买线({latest_cci:.1f})，斜率确认卖出信号"
                
            # 超买超卖区域反转信号（增强版）
            elif latest_cci > oversold_threshold and prev_cci <= oversold_threshold:
                # Enhanced CCI从超卖区域上穿 - 买入信号
                signal_type = "buy"
                base_strength = 0.8
                # 多周期协同确认
                if latest_cci_secondary > latest_cci_secondary - 20:  # 次要周期也在上升
                    base_strength += 0.05
                strength = base_strength * volatility_factor
                confidence = 0.8
                reason = f"Enhanced CCI上穿超卖线({latest_cci:.1f})，多周期协同买入信号"
                
            elif latest_cci < overbought_threshold and prev_cci >= overbought_threshold:
                # Enhanced CCI从超买区域下穿 - 卖出信号
                signal_type = "sell"
                base_strength = 0.8
                # 多周期协同确认
                if latest_cci_secondary < latest_cci_secondary + 20:  # 次要周期也在下降
                    base_strength += 0.05
                strength = base_strength * volatility_factor
                confidence = 0.8
                reason = f"Enhanced CCI下穿超买线({latest_cci:.1f})，多周期协同卖出信号"
                
            # CCI与均线交叉信号（Enhanced特有）
            elif hasattr(self, '_result') and 'cci_ma5' in self._result.columns:
                prev_cci_ma5 = self._result['cci_ma5'].iloc[-2] if len(self._result) >= 2 else latest_cci_ma5
                
                if latest_cci > latest_cci_ma5 and prev_cci <= prev_cci_ma5:
                    # CCI上穿5日均线 - 买入信号
                    signal_type = "buy"
                    base_strength = 0.7
                    # 零轴位置确认
                    if latest_cci > 0:
                        base_strength += 0.1
                    strength = base_strength * slope_factor
                    confidence = 0.75
                    reason = f"Enhanced CCI上穿均线({latest_cci:.1f}>{latest_cci_ma5:.1f})，买入信号"
                    
                elif latest_cci < latest_cci_ma5 and prev_cci >= prev_cci_ma5:
                    # CCI下穿5日均线 - 卖出信号
                    signal_type = "sell"
                    base_strength = 0.7
                    # 零轴位置确认
                    if latest_cci < 0:
                        base_strength += 0.1
                    strength = base_strength * slope_factor
                    confidence = 0.75
                    reason = f"Enhanced CCI下穿均线({latest_cci:.1f}<{latest_cci_ma5:.1f})，卖出信号"
            
            # 持续信号（增强版）
            if signal_type == "hold":
                if latest_cci < oversold_threshold:
                    # Enhanced CCI持续在超卖区域
                    signal_type = "buy"
                    base_strength = 0.6
                    # 根据超卖深度和多周期协同调整
                    oversold_depth = abs(latest_cci - oversold_threshold)
                    if oversold_depth > 50 and latest_cci_secondary < oversold_threshold:
                        base_strength += min(0.2, oversold_depth / 250)
                    strength = base_strength * volatility_factor
                    confidence = 0.7
                    reason = f"Enhanced CCI深度超卖({latest_cci:.1f})，多重确认持续买入"
                    
                elif latest_cci > overbought_threshold:
                    # Enhanced CCI持续在超买区域
                    signal_type = "sell"
                    base_strength = 0.6
                    # 根据超买高度和多周期协同调整
                    overbought_height = latest_cci - overbought_threshold
                    if overbought_height > 50 and latest_cci_secondary > overbought_threshold:
                        base_strength += min(0.2, overbought_height / 250)
                    strength = base_strength * volatility_factor
                    confidence = 0.7
                    reason = f"Enhanced CCI高度超买({latest_cci:.1f})，多重确认持续卖出"
            
            # 计算Enhanced CCI趋势变化
            cci_rising = latest_cci > prev_cci
            cci_momentum = latest_cci - prev_cci
            
            # 设置增强元数据
            metadata = {
                'cci_value': latest_cci,
                'cci_secondary': latest_cci_secondary,
                'cci_ma5': latest_cci_ma5,
                'cci_slope': latest_cci_slope,
                'cci_volatility': latest_cci_volatility,
                'cci_trend': 'rising' if cci_rising else 'falling',
                'cci_momentum': cci_momentum,
                'cci_zone': self._get_enhanced_cci_zone(latest_cci),
                'volatility_factor': volatility_factor,
                'slope_factor': slope_factor,
                'signal_strength': 'strong' if abs(latest_cci) > overbought_threshold else 'weak',
                'multi_period_sync': abs(latest_cci - latest_cci_secondary) < 30,
                'extreme_overbought': extreme_overbought,
                'overbought_level': overbought_threshold,
                'oversold_level': oversold_threshold,
                'extreme_oversold': extreme_oversold
            }
            
            # 检测背离模式（Enhanced特有）
            if len(self._result) >= 10:
                divergence_detected = self._detect_enhanced_cci_divergence(data)
                if divergence_detected:
                    metadata['divergence_detected'] = True
                    # 背离信号调整
                    if signal_type == "hold":
                        signal_type = "sell" if latest_cci > 0 else "buy"
                        strength = 0.75 * volatility_factor
                        confidence = 0.8
                        reason = "检测到Enhanced CCI背离，多重确认反转信号"
                    else:
                        # 增强现有信号
                        strength = min(1.0, strength + 0.1)
                        confidence = min(1.0, confidence + 0.05)
                        reason += "（背离确认）"
            
            # 6. 标准化输出
            return {
                'signal_type': signal_type,
                'strength': max(0.0, min(1.0, strength)),
                'confidence': max(0.0, min(1.0, confidence)),
                'timestamp': pd.Timestamp.now(),
                'reason': reason,
                'metadata': {
                    'latest_close': latest_close,
                    **metadata
                }
            }

        except Exception as e:
            logger.warning(f"Enhanced CCI信号生成失败: {e}")
            return self._get_default_signal(f"信号生成失败: {str(e)}")

    def _validate_signal_data(self, data: pd.DataFrame) -> bool:
        """
        验证信号生成所需的数据
        
        Args:
            data: 输入数据DataFrame
            
        Returns:
            bool: 数据是否有效
        """
        if data is None or data.empty:
            return False
            
        required_columns = ['high', 'low', 'close']
        if not all(col in data.columns for col in required_columns):
            return False
            
        # Enhanced CCI需要更多数据（由于多周期和自适应特性）
        min_periods = max(
            getattr(self, 'base_period', 20),
            getattr(self, 'secondary_period', 40),
            getattr(self, 'trend_period', 50)
        ) + 10
        if len(data) < min_periods:
            return False
            
        return True

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
            'metadata': {}
        }

    def has_result(self) -> bool:
        """
        检查是否已有计算结果
        
        Returns:
            bool: 是否已有计算结果
        """
        return (self._result is not None and 
                hasattr(self._result, 'empty') and 
                not self._result.empty and
                'cci' in self._result.columns)

    def _get_enhanced_cci_zone(self, cci_value: float) -> str:
        """
        获取Enhanced CCI所在区域（更精细的区域划分）
        
        Args:
            cci_value: CCI值
            
        Returns:
            str: Enhanced CCI区域描述
        """
        extreme_overbought = getattr(self, 'EXTREME_OVERBOUGHT', 200)
        overbought = getattr(self, 'overbought', 100)
        oversold = getattr(self, 'oversold', -100)
        extreme_oversold = getattr(self, 'EXTREME_OVERSOLD', -200)
        
        if cci_value > extreme_overbought:
            return "extreme_overbought"
        elif cci_value > overbought:
            return "overbought"
        elif cci_value > 50:
            return "strong_bullish"
        elif cci_value > 0:
            return "bullish"
        elif cci_value > -50:
            return "bearish"
        elif cci_value > oversold:
            return "weak_bearish"
        elif cci_value > extreme_oversold:
            return "oversold"
        else:
            return "extreme_oversold"

    def _detect_enhanced_cci_divergence(self, data: pd.DataFrame) -> bool:
        """
        检测Enhanced CCI背离形态（更智能的背离检测）
        
        Args:
            data: 价格数据
            
        Returns:
            bool: 是否检测到背离
        """
        try:
            if len(data) < 15 or len(self._result) < 15:
                return False
                
            # 获取最近15个周期的数据
            recent_prices = data['close'].iloc[-15:]
            recent_cci = self._result['cci'].iloc[-15:]
            
            # Enhanced背离检测：考虑多个时间框架
            # 短期背离（5周期）
            short_prices = recent_prices.iloc[-5:]
            short_cci = recent_cci.iloc[-5:]
            
            # 中期背离（10周期）
            mid_prices = recent_prices.iloc[-10:]
            mid_cci = recent_cci.iloc[-10:]
            
            # 检测短期背离
            short_top_divergence = (short_prices.iloc[-1] == short_prices.max() and 
                                  short_cci.iloc[-1] < short_cci.max())
            short_bottom_divergence = (short_prices.iloc[-1] == short_prices.min() and 
                                     short_cci.iloc[-1] > short_cci.min())
            
            # 检测中期背离
            mid_top_divergence = (mid_prices.iloc[-1] >= mid_prices.quantile(0.9) and 
                                mid_cci.iloc[-1] < mid_cci.quantile(0.9))
            mid_bottom_divergence = (mid_prices.iloc[-1] <= mid_prices.quantile(0.1) and 
                                   mid_cci.iloc[-1] > mid_cci.quantile(0.1))
            
            # 综合判断：短期和中期都出现背离
            return ((short_top_divergence or short_bottom_divergence) and 
                   (mid_top_divergence or mid_bottom_divergence))
            
        except Exception as e:
            logger.warning(f"Enhanced CCI背离检测失败: {e}")
            return False
