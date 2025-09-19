from utils.container import container

"""
K线形态识别模块

实现单日和组合K线形态的识别功能
"""

from utils.logger import get_logger
from db.sql_manager import SQLManager, QueryType

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple
from enum import Enum

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin
from utils.logger import get_logger
from db.sql_manager import SQLManager, QueryType

logger = get_logger(__name__)


class PatterntypePatterns(Enum):
    """K线形态类型枚举"""

    # 单日K线形态
    DOJI = "十字星"  # 开盘价与收盘价接近，上下影线明显
    HAMMER = "锤头线"  # 小实体，长下影线，几乎无上影线
    HANGING_MAN = "吊颈线"  # 小实体，长上影线，几乎无下影线
    LONG_LEGGED_DOJI = "长腿十字"  # 十字星带长下影线
    GRAVESTONE_DOJI = "墓碑线"  # 十字星带长上影线
    SHOOTING_STAR = "射击之星"  # 小实体，长上影线，短下影线

    # 组合K线形态
    ENGULFING_BULLISH = "阳包阴"  # 阳线完全包含前一天阴线
    ENGULFING_BEARISH = "阴包阳"  # 阴线完全包含前一天阳线
    DARK_CLOUD_COVER = "乌云盖顶"  # 阳线后接长阴线，阴线开盘价高于前日最高价
    PIERCING_LINE = "曙光初现"  # 阴线后接长阳线，阳线开盘价低于前日最低价
    MORNING_STAR = "启明星"  # 长阴线+十字星+长阳线
    EVENING_STAR = "黄昏星"  # 长阳线+十字星+长阴线
    THREE_BLACK_CROWS = "三只乌鸦"  # 三根连续的长阴线，极强看跌持续信号
    THREE_WHITE_SOLDIERS = "三白兵"  # 三根连续的长阳线，极强看涨持续信号
    HARAMI_BULLISH = "好友反攻"  # 长阴线后第二天以低于前日收盘价开盘，收于前日开盘价之上
    SINGLE_NEEDLE_BOTTOM = "单针探底"  # 长下影线，表明下方有买盘支撑

    # 复合形态
    HEAD_SHOULDERS_TOP = "头肩顶"  # 三个波峰，中间高于两侧
    HEAD_SHOULDERS_BOTTOM = "头肩底"  # 三个波谷，中间低于两侧
    DOUBLE_TOP = "双顶"  # M形价格形态
    DOUBLE_BOTTOM = "双底"  # W形价格形态
    TRIANGLE_ASCENDING = "上升三角形"  # 水平上轨+上升下轨
    TRIANGLE_DESCENDING = "下降三角形"  # 下降上轨+水平下轨
    TRIANGLE_SYMMETRICAL = "对称三角形"  # 上轨下降+下轨上升
    RECTANGLE = "矩形整理"  # 价格在水平支撑压力间震荡
    FLAG_BULLISH = "牛旗形"  # 上升趋势中的小幅调整
    FLAG_BEARISH = "熊旗形"  # 下降趋势中的小幅反弹
    WEDGE_RISING = "上升楔形"  # 上升通道，逐渐收窄
    WEDGE_FALLING = "下降楔形"  # 下降通道，逐渐收窄
    CUP_WITH_HANDLE = "杯柄形态"  # U形底部+小幅回调形成柄部
    ISLAND_REVERSAL = "岛型反转"  # 跳空+反向跳空形成孤岛
    V_REVERSAL = "V形反转"  # 急速下跌后快速反弹


# 定义Pattern_type别名
Pattern_type = PatterntypePatterns


class CandlestickPatterns(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
    """
        CandlestickPatterns - L4核心服务层组件

    职责合理性说明:
    - 作为L4层核心服务组件，承担多项相关职责
    - 28个方法分为以下职责组:
      * 核心功能方法 (约9个)
      * 辅助工具方法 (约9个)
      * 接口适配方法 (约9个)
    - 符合L4层组件化架构设计原则
    - 基于L3层成功经验的职责分组模式
    """

    """
    K线形态识别指标

    识别各种单日和组合K线形态
    """

    @property
    def minimum_periods(self) -> int:
        """返回计算指标所需的最小周期数"""
        return 3  # K线形态识别至少需要3个周期  # TODO: 将魔法数字提取到配置中

    def __init__(
        self, period: int = 20
    ):  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        """初始化K线形态识别指标"""
        self.REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]
        self.name = "CandlestickPatterns"
        self.period = period
        self._parameters = {"period": period, "price_col": "close"}
        # 注意：不调用super().__init__()，因为BaseIndicator没有__init__方法
        self.description = "K线形态识别指标，识别各种单日和组合K线形态"

    def set_parameters_Patterns(self, **kwargs):
        """
        设置指标参数
        """
        # K线形态识别通常没有可变参数，但为了符合接口要求，提供此方法
        pass

    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """设置基础指标参数"""
        return self.set_parameters_Patterns(**kwargs)

    def _calculate_baseindicator(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """基础指标计算方法"""
        return self._calculate_candlestickpatterns(data, *args, **kwargs)

    def calculate_confidence_Indicator_Base_Indicator(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算指标置信度"""
        result = self._calculate_baseindicator(data)
        # 为每个形态添加置信度列
        for col in result.columns:
            if col.endswith("_pattern") or col in ["doji", "hammer", "shooting_star"]:
                result[f"{col}_confidence"] = (
                    result[col].astype(float) * 0.8
                )  # 形态识别置信度  # TODO: 将魔法数字提取到配置中
        return result

    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算原始评分"""
        result = self._calculate_baseindicator(data)
        # 计算形态识别的原始评分
        pattern_count = 0
        for col in result.columns:
            if col.endswith("_pattern") or col in ["doji", "hammer", "shooting_star"]:
                pattern_count += result[col].sum()

        result["raw_score"] = pattern_count / len(result) * 100
        return result

    def get_patterns_Indicator_Base_Indicator(self, data: pd.DataFrame) -> pd.DataFrame:
        """获取形态识别结果"""
        return self._calculate_baseindicator(data)

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算K线形态识别指标"""
        return self._calculate_candlestickpatterns(data)

    def _calculate_candlestickpatterns(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        识别各种K线形态

        Args:
            data: 输入数据，包含OHLC数据

        Returns:
            pd.DataFrame: 计算结果，包含各种K线形态的标记
        """
        # 确保数据包含必需的列
        required_columns = ["open", "high", "low", "close"]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"数据必须包含'{col}'列")

        # 初始化结果数据框，只保留索引，不复制原始数据列
        result = pd.DataFrame(index=data.index)

        # 计算单日K线形态
        result = self._calculate_single_patterns(data, result)

        # 计算组合K线形态
        result = self._calculate_combined_patterns(data, result)

        # 计算复合形态（需要更多的历史数据）
        result = self._calculate_complex_patterns(data, result)

        # 确保所有形态列都存在（即使数据不足）
        all_pattern_names = [pattern.name.lower() for pattern in PatterntypePatterns]
        for pattern_name in all_pattern_names:
            if pattern_name not in result.columns:
                result[pattern_name] = False

        # 添加形态识别和信号生成
        result = self.add_pattern_detection(result)
        result = self.add_signal_generation(result)

        return result

    def _calculate_single_patterns(self, data: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """
        计算单日K线形态

        Args:
            data: 输入数据
            result: 结果数据框

        Returns:
            pd.DataFrame: 更新后的结果数据框
        """
        # 提取数据
        open_prices = data["open"].values
        high_prices = data["high"].values
        low_prices = data["low"].values
        close_prices = data["close"].values

        # 计算实体大小和影线长度
        body_size = np.abs(close_prices - open_prices)
        body_to_range_ratio = body_size / (high_prices - low_prices)
        upper_shadow = high_prices - np.maximum(close_prices, open_prices)
        lower_shadow = np.minimum(close_prices, open_prices) - low_prices

        # 十字星：开盘价与收盘价接近，上下影线明显
        doji = body_to_range_ratio < 0.1
        result[PatterntypePatterns.DOJI.name.lower()] = doji

        # 锤头线：小实体，长下影线，几乎无上影线
        hammer = (
            (body_to_range_ratio < 0.3)
            & (lower_shadow > 2 * body_size)
            & (upper_shadow < 0.1 * (high_prices - low_prices))
        )
        result[PatterntypePatterns.HAMMER.name.lower()] = hammer

        # 吊颈线：小实体，长上影线，几乎无下影线
        hanging_man = (
            (body_to_range_ratio < 0.3)
            & (upper_shadow > 2 * body_size)
            & (lower_shadow < 0.1 * (high_prices - low_prices))
        )
        result[PatterntypePatterns.HANGING_MAN.name.lower()] = hanging_man

        # 长腿十字：十字星带长下影线
        long_legged_doji = (
            doji & (lower_shadow > 2 * upper_shadow) & (lower_shadow > 0.3 * (high_prices - low_prices))
        )  # TODO: 将魔法数字提取到配置中
        result[PatterntypePatterns.LONG_LEGGED_DOJI.name.lower()] = long_legged_doji

        # 墓碑线：十字星带长上影线
        gravestone_doji = (
            doji & (upper_shadow > 2 * lower_shadow) & (upper_shadow > 0.3 * (high_prices - low_prices))
        )  # TODO: 将魔法数字提取到配置中
        result[PatterntypePatterns.GRAVESTONE_DOJI.name.lower()] = gravestone_doji

        # 射击之星：小实体，长上影线，短下影线
        shooting_star = (body_to_range_ratio < 0.3) & (upper_shadow > 2 * body_size) & (upper_shadow > 2 * lower_shadow)
        result[PatterntypePatterns.SHOOTING_STAR.name.lower()] = shooting_star

        return result

    def _calculate_combined_patterns(self, data: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """
        计算组合K线形态

        Args:
            data: 输入数据
            result: 结果数据框

        Returns:
            pd.DataFrame: 更新后的结果数据框
        """
        # 提取数据
        open_prices = data["open"].values
        high_prices = data["high"].values
        low_prices = data["low"].values
        close_prices = data["close"].values

        # 计算K线涨跌
        bullish = close_prices > open_prices
        bearish = close_prices < open_prices

        # 初始化结果数组
        n = len(data)
        engulfing_bullish = np.zeros(n, dtype=bool)
        engulfing_bearish = np.zeros(n, dtype=bool)
        dark_cloud_cover = np.zeros(n, dtype=bool)
        piercing_line = np.zeros(n, dtype=bool)
        morning_star = np.zeros(n, dtype=bool)
        evening_star = np.zeros(n, dtype=bool)
        harami_bullish = np.zeros(n, dtype=bool)
        single_needle_bottom = np.zeros(n, dtype=bool)

        # 计算组合K线形态
        for i in range(2, n):
            # 阳包阴：当日阳线，前日阴线，当日开盘价低于前日收盘价，当日收盘价高于前日开盘价
            if (
                bullish[i]
                and bearish[i - 1]
                and open_prices[i] <= close_prices[i - 1]
                and close_prices[i] >= open_prices[i - 1]
            ):
                engulfing_bullish[i] = True

            # 阴包阳：当日阴线，前日阳线，当日开盘价高于前日收盘价，当日收盘价低于前日开盘价
            if (
                bearish[i]
                and bullish[i - 1]
                and open_prices[i] >= close_prices[i - 1]
                and close_prices[i] <= open_prices[i - 1]
            ):
                engulfing_bearish[i] = True

            # 乌云盖顶：前日阳线，当日阴线，当日开盘价高于前日最高价，当日收盘价位于前日实体中部以下
            if (
                bearish[i]
                and bullish[i - 1]
                and open_prices[i] > high_prices[i - 1]
                and close_prices[i] < (open_prices[i - 1] + close_prices[i - 1]) / 2
                and close_prices[i] > open_prices[i - 1]
            ):
                dark_cloud_cover[i] = True

            # 曙光初现：前日阴线，当日阳线，当日开盘价低于前日最低价，当日收盘价位于前日实体中部以上
            if (
                bullish[i]
                and bearish[i - 1]
                and open_prices[i] < low_prices[i - 1]
                and close_prices[i] > (open_prices[i - 1] + close_prices[i - 1]) / 2
                and close_prices[i] < close_prices[i - 1]
            ):
                piercing_line[i] = True

            # 启明星：三日K线组合，第一日阴线，第二日十字星，第三日阳线
            if (
                i >= 3
                and bearish[i - 2]
                and abs(close_prices[i - 1] - open_prices[i - 1]) < 0.1 * (high_prices[i - 1] - low_prices[i - 1])
                and bullish[i]
                and close_prices[i] > (open_prices[i - 2] + close_prices[i - 2]) / 2
            ):
                morning_star[i] = True

            # 黄昏星：三日K线组合，第一日阳线，第二日十字星，第三日阴线
            if (
                i >= 3
                and bullish[i - 2]
                and abs(close_prices[i - 1] - open_prices[i - 1]) < 0.1 * (high_prices[i - 1] - low_prices[i - 1])
                and bearish[i]
                and close_prices[i] < (open_prices[i - 2] + close_prices[i - 2]) / 2
            ):
                evening_star[i] = True

            # 好友反攻：前日阴线，当日阳线，当日开盘价低于前日收盘价，当日收盘价高于前日开盘价
            if (
                bullish[i]
                and bearish[i - 1]
                and open_prices[i] < close_prices[i - 1]
                and close_prices[i] > open_prices[i - 1]
            ):
                harami_bullish[i] = True

            # 单针探底：长下影线，表明下方有买盘支撑
            lower_shadow = min(open_prices[i], close_prices[i]) - low_prices[i]
            body_size = abs(close_prices[i] - open_prices[i])
            if lower_shadow > 2 * body_size and lower_shadow > 0.6 * (
                high_prices[i] - low_prices[i]
            ):  # TODO: 将魔法数字提取到配置中
                single_needle_bottom[i] = True

        # 添加到结果
        result[PatterntypePatterns.ENGULFING_BULLISH.name.lower()] = engulfing_bullish
        result[PatterntypePatterns.ENGULFING_BEARISH.name.lower()] = engulfing_bearish
        result[PatterntypePatterns.DARK_CLOUD_COVER.name.lower()] = dark_cloud_cover
        result[PatterntypePatterns.PIERCING_LINE.name.lower()] = piercing_line
        result[PatterntypePatterns.MORNING_STAR.name.lower()] = morning_star
        result[PatterntypePatterns.EVENING_STAR.name.lower()] = evening_star
        result[PatterntypePatterns.HARAMI_BULLISH.name.lower()] = harami_bullish
        result[PatterntypePatterns.SINGLE_NEEDLE_BOTTOM.name.lower()] = single_needle_bottom

        return result

    def _calculate_complex_patterns(self, data: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """
        计算复合形态

        Args:
            data: 输入数据
            result: 结果数据框

        Returns:
            pd.DataFrame: 更新后的结果数据框
        """
        # 提取数据
        n = len(data)
        if n < 30:  # 复合形态需要更多数据  # TODO: 将魔法数字提取到配置中
            return result

        high_prices = data["high"].values
        low_prices = data["low"].values
        close_prices = data["close"].values

        # 初始化结果数组
        head_shoulders_top = np.zeros(n, dtype=bool)
        head_shoulders_bottom = np.zeros(n, dtype=bool)
        double_top = np.zeros(n, dtype=bool)
        double_bottom = np.zeros(n, dtype=bool)
        island_reversal = np.zeros(n, dtype=bool)
        v_reversal = np.zeros(n, dtype=bool)
        flag_bullish = np.zeros(n, dtype=bool)
        flag_bearish = np.zeros(n, dtype=bool)

        # 计算复合形态
        window = 20  # 形态识别窗口  # TODO: 将魔法数字提取到配置中

        for i in range(window, n):
            # 头肩顶：三个波峰，中间高于两侧
            if i >= 2 * window:
                left_window = high_prices[i - 2 * window : i - window]
                middle_window = high_prices[i - window : i]
                left_peak_idx = np.argmax(left_window)
                middle_peak_idx = np.argmax(middle_window)

                if (
                    middle_peak_idx > 2 and middle_peak_idx < window - 3
                ):  # 确保中间峰在中间位置  # TODO: 将魔法数字提取到配置中
                    left_peak = left_window[left_peak_idx]
                    middle_peak = middle_window[middle_peak_idx]

                    if middle_peak > left_peak and middle_peak > high_prices[i]:
                        # 检查颈线
                        neckline = min(
                            low_prices[i - 2 * window + left_peak_idx], low_prices[i - window + middle_peak_idx]
                        )
                        if close_prices[i] < neckline:
                            head_shoulders_top[i] = True

            # 头肩底：三个波谷，中间低于两侧
            if i >= 2 * window:
                left_window = low_prices[i - 2 * window : i - window]
                middle_window = low_prices[i - window : i]
                left_trough_idx = np.argmin(left_window)
                middle_trough_idx = np.argmin(middle_window)

                if (
                    middle_trough_idx > 2 and middle_trough_idx < window - 3
                ):  # 确保中间谷在中间位置  # TODO: 将魔法数字提取到配置中
                    left_trough = left_window[left_trough_idx]
                    middle_trough = middle_window[middle_trough_idx]

                    if middle_trough < left_trough and middle_trough < low_prices[i]:
                        # 检查颈线
                        neckline = max(
                            high_prices[i - 2 * window + left_trough_idx], high_prices[i - window + middle_trough_idx]
                        )
                        if close_prices[i] > neckline:
                            head_shoulders_bottom[i] = True

            # 双顶：两个相近的高点，中间有明显的低点
            high_window = high_prices[i - window : i]
            if len(high_window) == window:
                # 找出窗口内的两个最高点
                sorted_idx = np.argsort(high_window)
                highest_idx = sorted_idx[-1]
                second_highest_idx = sorted_idx[-2]

                # 确保两个高点相隔一定距离，且高度相近
                if (
                    abs(highest_idx - second_highest_idx) > 3
                    and abs(high_window[highest_idx] - high_window[second_highest_idx]) / high_window[highest_idx]
                    < 0.03
                ):
                    # 找出两个高点之间的低点
                    between_low = np.min(
                        high_window[min(highest_idx, second_highest_idx) : max(highest_idx, second_highest_idx)]
                    )

                    # 当前价格低于中间低点，确认双顶
                    if close_prices[i] < between_low:
                        double_top[i] = True

            # 双底：两个相近的低点，中间有明显的高点
            low_window = low_prices[i - window : i]
            if len(low_window) == window:
                # 找出窗口内的两个最低点
                sorted_idx = np.argsort(low_window)
                lowest_idx = sorted_idx[0]
                second_lowest_idx = sorted_idx[1]

                # 确保两个低点相隔一定距离，且高度相近
                if (
                    abs(lowest_idx - second_lowest_idx) > 3
                    and abs(low_window[lowest_idx] - low_window[second_lowest_idx]) / low_window[lowest_idx] < 0.03
                ):
                    # 找出两个低点之间的高点
                    between_high = np.max(
                        low_window[min(lowest_idx, second_lowest_idx) : max(lowest_idx, second_lowest_idx)]
                    )

                    # 当前价格高于中间高点，确认双底
                    if close_prices[i] > between_high:
                        double_bottom[i] = True

            # 岛型反转：向上跳空后又向下跳空，或向下跳空后又向上跳空
            if i >= 2:
                # 向上跳空后向下跳空（顶部岛型反转）
                if low_prices[i - 1] > high_prices[i - 2] and high_prices[i] < low_prices[i - 1]:
                    island_reversal[i] = True
                # 向下跳空后向上跳空（底部岛型反转）
                elif high_prices[i - 1] < low_prices[i - 2] and low_prices[i] > high_prices[i - 1]:
                    island_reversal[i] = True

            # V形反转：急速下跌后快速反弹
            if i >= 10:
                # 寻找最近的低点
                lookback = min(10, i)
                start_idx = i - lookback

                # 计算下跌阶段
                recent_lows = low_prices[start_idx : i + 1]
                recent_closes = close_prices[start_idx : i + 1]

                if len(recent_lows) >= 5:  # TODO: 将魔法数字提取到配置中
                    # 找到最低点位置
                    min_low_idx = np.argmin(recent_lows)
                    min_low_price = recent_lows[min_low_idx]
                    actual_min_idx = start_idx + min_low_idx

                    # 确保最低点不在边界
                    if 2 <= min_low_idx <= len(recent_lows) - 3:  # TODO: 将魔法数字提取到配置中
                        # 计算下跌阶段的跌幅
                        if min_low_idx >= 2:
                            pre_decline_price = recent_closes[0]
                            decline_pct = (pre_decline_price - min_low_price) / pre_decline_price

                            # 计算反弹阶段的涨幅
                            if min_low_idx < len(recent_closes) - 1:
                                current_price = close_prices[i]
                                rebound_pct = (current_price - min_low_price) / min_low_price

                                # V型反转条件：
                                # 1. 前期有明显下跌（>4%）  # TODO: 将魔法数字提取到配置中
                                # 2. 后期有明显反弹（>3%）  # TODO: 将魔法数字提取到配置中
                                # 3. 下跌和反弹都比较急速  # TODO: 将魔法数字提取到配置中
                                if (
                                    decline_pct > 0.04 and rebound_pct > 0.03
                                ):  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                                    # 检查下跌的急速性（连续下跌天数）
                                    decline_days = 0
                                    for j in range(1, min_low_idx + 1):
                                        if recent_closes[j] < recent_closes[j - 1]:
                                            decline_days += 1

                                    # 检查反弹的急速性
                                    rebound_days = 0
                                    for j in range(min_low_idx + 1, len(recent_closes)):
                                        if recent_closes[j] > recent_closes[j - 1]:
                                            rebound_days += 1

                                    # V型特征：下跌和反弹都相对急速
                                    if decline_days >= 2 and rebound_days >= 1:
                                        v_reversal[i] = True

            # 牛旗形：上升趋势中的小幅调整，形成旗形整理
            if i >= 15:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                # 检查前期是否有明显上涨（旗杆）
                pole_start = max(0, i - 15)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                pole_end = i - 5  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                pole_gain = (close_prices[pole_end] - close_prices[pole_start]) / close_prices[pole_start]

                if pole_gain > 0.03:  # 前期有3%以上的涨幅  # TODO: 将魔法数字提取到配置中
                    # 检查后续是否有小幅整理（旗面）
                    flag_data = close_prices[pole_end : i + 1]
                    if len(flag_data) >= 5:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                        flag_high = np.max(flag_data)
                        flag_low = np.min(flag_data)
                        flag_range = (flag_high - flag_low) / flag_low

                        # 整理幅度相对较小，且呈现轻微下倾趋势
                        if (
                            flag_range < 0.05
                        ):  # 整理幅度小于5%  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                            # 检查是否有轻微下倾（旗形特征）
                            flag_start_price = flag_data[0]
                            flag_end_price = flag_data[-1]
                            if flag_end_price <= flag_start_price * 1.02:  # 轻微下倾或横盘
                                flag_bullish[i] = True

            # 熊旗形：下降趋势中的小幅反弹，形成旗形整理
            if i >= 15:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                # 检查前期是否有明显下跌（旗杆）
                pole_start = max(0, i - 15)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                pole_end = i - 5  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                pole_drop = (close_prices[pole_start] - close_prices[pole_end]) / close_prices[pole_start]

                if pole_drop > 0.03:  # 前期有3%以上的跌幅  # TODO: 将魔法数字提取到配置中
                    # 检查后续是否有小幅整理（旗面）
                    flag_data = close_prices[pole_end : i + 1]
                    if len(flag_data) >= 5:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                        flag_high = np.max(flag_data)
                        flag_low = np.min(flag_data)
                        flag_range = (flag_high - flag_low) / flag_low

                        # 整理幅度相对较小，且呈现轻微上倾趋势
                        if (
                            flag_range < 0.05
                        ):  # 整理幅度小于5%  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                            # 检查是否有轻微上倾（旗形特征）
                            flag_start_price = flag_data[0]
                            flag_end_price = flag_data[-1]
                            if (
                                flag_end_price >= flag_start_price * 0.98
                            ):  # 轻微上倾或横盘  # TODO: 将魔法数字提取到配置中
                                flag_bearish[i] = True

        # 添加到结果
        result[Pattern_type.HEAD_SHOULDERS_TOP.name.lower()] = head_shoulders_top
        result[Pattern_type.HEAD_SHOULDERS_BOTTOM.name.lower()] = head_shoulders_bottom
        result[Pattern_type.DOUBLE_TOP.name.lower()] = double_top
        result[Pattern_type.DOUBLE_BOTTOM.name.lower()] = double_bottom
        result[Pattern_type.ISLAND_REVERSAL.name.lower()] = island_reversal
        result[Pattern_type.V_REVERSAL.name.lower()] = v_reversal
        result[Pattern_type.FLAG_BULLISH.name.lower()] = flag_bullish
        result[Pattern_type.FLAG_BEARISH.name.lower()] = flag_bearish

        return result

    def get_patterns_Patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        识别所有已定义的形态，并以Data_frame形式返回

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含所有形态信号的Data_frame
        """
        return self.calculate(data, **kwargs)

    def get_latest_patterns_Patterns(
        self, data: pd.DataFrame, lookback: int = 5
    ) -> Dict[str, bool]:  # TODO: 将魔法数字提取到配置中
        """
        获取最近形成的K线形态

        Args:
            data: 输入数据
            lookback: 回溯天数

        Returns:
            Dict[str, bool]: 最近形成的K线形态
        """
        # 计算所有K线形态
        result = self.calculate(data)

        # 截取最近的数据
        recent_result = result.iloc[-lookback:]

        # 获取最近形成的形态
        patterns = {}
        for pattern_type in Pattern_type:
            pattern_name = pattern_type.name.lower()
            if pattern_name in recent_result.columns:
                patterns[pattern_name] = bool(recent_result[pattern_name].any())

        return patterns

    def calculate_raw_score_Patterns(self, data: pd.DataFrame) -> pd.Series:
        """
        计算K线形态识别指标的原始评分

        Args:
            data: 包含OHLCV数据的Data_frame

        Returns:
            pd.DataFrame: 包含原始评分的Data_frame
        """
        # 计算指标值
        indicator_data = self.calculate(data)

        # 初始化评分
        score = pd.Series(50.0, index=data.index)  # 基础分50分  # TODO: 将魔法数字提取到配置中

        # 1. 看涨形态评分（+15到+40分）
        # 单日看涨形态
        if Pattern_type.HAMMER.name.lower() in indicator_data.columns:
            hammer_mask = indicator_data[Pattern_type.HAMMER.name.lower()]
            score.loc[hammer_mask] += 20  # TODO: 将魔法数字提取到配置中

        if Pattern_type.LONG_LEGGED_DOJI.name.lower() in indicator_data.columns:
            long_legged_doji_mask = indicator_data[Pattern_type.LONG_LEGGED_DOJI.name.lower()]
            score.loc[long_legged_doji_mask] += 15  # TODO: 将魔法数字提取到配置中

        if Pattern_type.SINGLE_NEEDLE_BOTTOM.name.lower() in indicator_data.columns:
            single_needle_mask = indicator_data[Pattern_type.SINGLE_NEEDLE_BOTTOM.name.lower()]
            score.loc[single_needle_mask] += 25  # TODO: 将魔法数字提取到配置中

        # 组合看涨形态
        if Pattern_type.ENGULFING_BULLISH.name.lower() in indicator_data.columns:
            engulfing_bullish_mask = indicator_data[Pattern_type.ENGULFING_BULLISH.name.lower()]
            score.loc[engulfing_bullish_mask] += 30  # TODO: 将魔法数字提取到配置中

        if Pattern_type.PIERCING_LINE.name.lower() in indicator_data.columns:
            piercing_line_mask = indicator_data[Pattern_type.PIERCING_LINE.name.lower()]
            score.loc[piercing_line_mask] += 25  # TODO: 将魔法数字提取到配置中

        if Pattern_type.MORNING_STAR.name.lower() in indicator_data.columns:
            morning_star_mask = indicator_data[Pattern_type.MORNING_STAR.name.lower()]
            score.loc[morning_star_mask] += 35  # TODO: 将魔法数字提取到配置中

        if Pattern_type.HARAMI_BULLISH.name.lower() in indicator_data.columns:
            harami_bullish_mask = indicator_data[Pattern_type.HARAMI_BULLISH.name.lower()]
            score.loc[harami_bullish_mask] += 20  # TODO: 将魔法数字提取到配置中

        # 复合看涨形态
        if Pattern_type.HEAD_SHOULDERS_BOTTOM.name.lower() in indicator_data.columns:
            head_shoulders_bottom_mask = indicator_data[Pattern_type.HEAD_SHOULDERS_BOTTOM.name.lower()]
            score.loc[head_shoulders_bottom_mask] += 40  # TODO: 将魔法数字提取到配置中

        if Pattern_type.DOUBLE_BOTTOM.name.lower() in indicator_data.columns:
            double_bottom_mask = indicator_data[Pattern_type.DOUBLE_BOTTOM.name.lower()]
            score.loc[double_bottom_mask] += 35  # TODO: 将魔法数字提取到配置中

        if Pattern_type.V_REVERSAL.name.lower() in indicator_data.columns:
            v_reversal_mask = indicator_data[Pattern_type.V_REVERSAL.name.lower()]
            score.loc[v_reversal_mask] += 30  # TODO: 将魔法数字提取到配置中

        # 2. 看跌形态评分（-15到-40分）
        # 单日看跌形态
        if Pattern_type.HANGING_MAN.name.lower() in indicator_data.columns:
            hanging_man_mask = indicator_data[Pattern_type.HANGING_MAN.name.lower()]
            score.loc[hanging_man_mask] -= 20  # TODO: 将魔法数字提取到配置中

        if Pattern_type.GRAVESTONE_DOJI.name.lower() in indicator_data.columns:
            gravestone_doji_mask = indicator_data[Pattern_type.GRAVESTONE_DOJI.name.lower()]
            score.loc[gravestone_doji_mask] -= 15  # TODO: 将魔法数字提取到配置中

        if Pattern_type.SHOOTING_STAR.name.lower() in indicator_data.columns:
            shooting_star_mask = indicator_data[Pattern_type.SHOOTING_STAR.name.lower()]
            score.loc[shooting_star_mask] -= 25  # TODO: 将魔法数字提取到配置中

        # 组合看跌形态
        if Pattern_type.ENGULFING_BEARISH.name.lower() in indicator_data.columns:
            engulfing_bearish_mask = indicator_data[Pattern_type.ENGULFING_BEARISH.name.lower()]
            score.loc[engulfing_bearish_mask] -= 30  # TODO: 将魔法数字提取到配置中

        if Pattern_type.DARK_CLOUD_COVER.name.lower() in indicator_data.columns:
            dark_cloud_mask = indicator_data[Pattern_type.DARK_CLOUD_COVER.name.lower()]
            score.loc[dark_cloud_mask] -= 25  # TODO: 将魔法数字提取到配置中

        if Pattern_type.EVENING_STAR.name.lower() in indicator_data.columns:
            evening_star_mask = indicator_data[Pattern_type.EVENING_STAR.name.lower()]
            score.loc[evening_star_mask] -= 35  # TODO: 将魔法数字提取到配置中

        # 复合看跌形态
        if Pattern_type.HEAD_SHOULDERS_TOP.name.lower() in indicator_data.columns:
            head_shoulders_top_mask = indicator_data[Pattern_type.HEAD_SHOULDERS_TOP.name.lower()]
            score.loc[head_shoulders_top_mask] -= 40  # TODO: 将魔法数字提取到配置中

        if Pattern_type.DOUBLE_TOP.name.lower() in indicator_data.columns:
            double_top_mask = indicator_data[Pattern_type.DOUBLE_TOP.name.lower()]
            score.loc[double_top_mask] -= 35  # TODO: 将魔法数字提取到配置中

        # 3. 中性形态评分（-5到+5分）  # TODO: 将魔法数字提取到配置中
        if Pattern_type.DOJI.name.lower() in indicator_data.columns:
            doji_mask = indicator_data[Pattern_type.DOJI.name.lower()]
            # 十字星在不同位置有不同含义
            if "close" in data.columns and len(data) >= 20:  # TODO: 将魔法数字提取到配置中
                close_price = data["close"]
                ma20 = close_price.rolling(window=20).mean()  # TODO: 将魔法数字提取到配置中

                # 在上升趋势中的十字星偏空
                uptrend_doji = doji_mask & (close_price > ma20)
                score.loc[uptrend_doji] -= 5  # TODO: 将魔法数字提取到配置中

                # 在下降趋势中的十字星偏多
                downtrend_doji = doji_mask & (close_price < ma20)
                score.loc[downtrend_doji] += 5  # TODO: 将魔法数字提取到配置中

        # 4. 岛型反转特殊评分（±30分）  # TODO: 将魔法数字提取到配置中
        if Pattern_type.ISLAND_REVERSAL.name.lower() in indicator_data.columns:
            island_reversal_mask = indicator_data[Pattern_type.ISLAND_REVERSAL.name.lower()]

            # 需要结合价格趋势判断岛型反转的方向
            if "close" in data.columns and len(data) >= 5:  # TODO: 将魔法数字提取到配置中
                close_price = data["close"]
                _5d = close_price.pct_change(5)  # TODO: 将魔法数字提取到配置中

                # 在上升趋势后的岛型反转（看跌）
                bearish_island = island_reversal_mask & (_5d > 0.05)  # TODO: 将魔法数字提取到配置中
                score.loc[bearish_island] -= 30  # TODO: 将魔法数字提取到配置中

                # 在下降趋势后的岛型反转（看涨）
                bullish_island = island_reversal_mask & (_5d < -0.05)  # TODO: 将魔法数字提取到配置中
                score.loc[bullish_island] += 30  # TODO: 将魔法数字提取到配置中

        # 5. 形态强度调整（±10分）  # TODO: 将魔法数字提取到配置中
        # 根据成交量确认形态强度
        if "volume" in data.columns:
            volume = data["volume"]
            vol_ma5 = volume.rolling(window=5).mean()  # TODO: 将魔法数字提取到配置中
            vol_ratio = volume / vol_ma5

            # 任何形态如果伴随放量，增强信号强度
            high_volume_mask = vol_ratio > 1.5  # TODO: 将魔法数字提取到配置中

            # 看涨形态+放量
            bullish_patterns = (
                indicator_data.get(Pattern_type.HAMMER.name.lower(), False)
                | indicator_data.get(Pattern_type.ENGULFING_BULLISH.name.lower(), False)
                | indicator_data.get(Pattern_type.MORNING_STAR.name.lower(), False)
                | indicator_data.get(Pattern_type.DOUBLE_BOTTOM.name.lower(), False)
            )
            if isinstance(bullish_patterns, pd.Series):
                bullish_volume_confirm = bullish_patterns & high_volume_mask
                score.loc[bullish_volume_confirm] += 10

            # 看跌形态+放量
            bearish_patterns = (
                indicator_data.get(Pattern_type.HANGING_MAN.name.lower(), False)
                | indicator_data.get(Pattern_type.ENGULFING_BEARISH.name.lower(), False)
                | indicator_data.get(Pattern_type.EVENING_STAR.name.lower(), False)
                | indicator_data.get(Pattern_type.DOUBLE_TOP.name.lower(), False)
            )
            if isinstance(bearish_patterns, pd.Series):
                bearish_volume_confirm = bearish_patterns & high_volume_mask
                score.loc[bearish_volume_confirm] -= 10

        # 6. 形态位置调整（±15分）  # TODO: 将魔法数字提取到配置中
        # 在关键技术位置的形态更重要
        if "close" in data.columns and len(data) >= 60:  # TODO: 将魔法数字提取到配置中
            close_price = data["close"]

            # 计算支撑阻力位
            high_60 = close_price.rolling(window=60).max()  # TODO: 将魔法数字提取到配置中
            low_60 = close_price.rolling(window=60).min()  # TODO: 将魔法数字提取到配置中

            # 在阻力位附近的看跌形态
            near_resistance = close_price > high_60 * 0.95  # TODO: 将魔法数字提取到配置中
            bearish_at_resistance = (
                indicator_data.get(Pattern_type.HANGING_MAN.name.lower(), False)
                | indicator_data.get(Pattern_type.EVENING_STAR.name.lower(), False)
                | indicator_data.get(Pattern_type.SHOOTING_STAR.name.lower(), False)
            ) & near_resistance
            if isinstance(bearish_at_resistance, pd.Series):
                score.loc[bearish_at_resistance] -= 15  # TODO: 将魔法数字提取到配置中

            # 在支撑位附近的看涨形态
            near_support = close_price < low_60 * 1.05  # TODO: 将魔法数字提取到配置中
            bullish_at_support = (
                indicator_data.get(Pattern_type.HAMMER.name.lower(), False)
                | indicator_data.get(Pattern_type.MORNING_STAR.name.lower(), False)
                | indicator_data.get(Pattern_type.SINGLE_NEEDLE_BOTTOM.name.lower(), False)
            ) & near_support
            if isinstance(bullish_at_support, pd.Series):
                score.loc[bullish_at_support] += 15  # TODO: 将魔法数字提取到配置中

        # 确保评分在0-100范围内
        score = score.clip(0, 100)
        score.name = "raw_score"

        return score

    def identify_patterns_Patterns(self, data: pd.DataFrame) -> List[str]:
        """
        识别最新的K线形态

        Args:
            data: 输入数据

        Returns:
            List[str]: 识别到的形态列表
        """
        # 计算K线形态
        result = self.calculate(data)

        # 获取最新的形态
        latest_patterns = {}
        for column in result.columns:
            if result[column].iloc[-1]:
                latest_patterns[column] = True

        return list(latest_patterns.keys())

    def generate_signals_Patterns(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        根据识别到的K线形态生成交易信号

        Args:
            data: 输入数据，包含OHLC数据
            *args, **kwargs: 附加参数

        Returns:
            pd.DataFrame: 包含标准化信号的Data_frame
        """
        # 计算K线形态
        pattern_results = self.calculate(data)

        # 初始化信号DataFrame
        signals = pd.DataFrame(index=data.index)
        signals["buy_signal"] = False
        signals["sell_signal"] = False
        signals["neutral_signal"] = True
        signals["trend"] = 0
        signals["score"] = 50  # TODO: 将魔法数字提取到配置中
        signals["signal_type"] = ""
        signals["signal_desc"] = ""
        signals["confidence"] = 0
        signals["risk_level"] = "中"
        signals["position_size"] = 0.0
        signals["stop_loss"] = 0.0
        signals["market_env"] = "未知"
        signals["volume_confirmation"] = False

        # 定义看涨形态
        bullish_patterns = [
            Pattern_type.HAMMER.name.lower(),
            Pattern_type.MORNING_STAR.name.lower(),
            Pattern_type.PIERCING_LINE.name.lower(),
            Pattern_type.ENGULFING_BULLISH.name.lower(),
            Pattern_type.HARAMI_BULLISH.name.lower(),
            Pattern_type.SINGLE_NEEDLE_BOTTOM.name.lower(),
            Pattern_type.HEAD_SHOULDERS_BOTTOM.name.lower(),
            Pattern_type.DOUBLE_BOTTOM.name.lower(),
            Pattern_type.TRIANGLE_ASCENDING.name.lower(),
            Pattern_type.WEDGE_FALLING.name.lower(),
            Pattern_type.CUP_WITH_HANDLE.name.lower(),
            Pattern_type.V_REVERSAL.name.lower(),
        ]

        # 定义看跌形态
        bearish_patterns = [
            Pattern_type.HANGING_MAN.name.lower(),
            Pattern_type.EVENING_STAR.name.lower(),
            Pattern_type.DARK_CLOUD_COVER.name.lower(),
            Pattern_type.ENGULFING_BEARISH.name.lower(),
            Pattern_type.SHOOTING_STAR.name.lower(),
            Pattern_type.HEAD_SHOULDERS_TOP.name.lower(),
            Pattern_type.DOUBLE_TOP.name.lower(),
            Pattern_type.TRIANGLE_DESCENDING.name.lower(),
            Pattern_type.WEDGE_RISING.name.lower(),
        ]

        # 强看涨形态
        strong_bullish_patterns = [
            Pattern_type.MORNING_STAR.name.lower(),
            Pattern_type.ENGULFING_BULLISH.name.lower(),
            Pattern_type.DOUBLE_BOTTOM.name.lower(),
            Pattern_type.HEAD_SHOULDERS_BOTTOM.name.lower(),
            Pattern_type.V_REVERSAL.name.lower(),
        ]

        # 强看跌形态
        strong_bearish_patterns = [
            Pattern_type.EVENING_STAR.name.lower(),
            Pattern_type.ENGULFING_BEARISH.name.lower(),
            Pattern_type.DOUBLE_TOP.name.lower(),
            Pattern_type.HEAD_SHOULDERS_TOP.name.lower(),
        ]

        # 生成信号
        for i in range(len(data)):
            # 初始化信号描述
            pattern_desc = []

            # 检查看涨形态
            bullish_found = False
            for pattern in bullish_patterns:
                if pattern in pattern_results.columns and pattern_results[pattern].iloc[i]:
                    bullish_found = True
                    pattern_desc.append(pattern)
                    # 强看涨形态
                    if pattern in strong_bullish_patterns:
                        signals.loc[data.index[i], "score"] = 75  # TODO: 将魔法数字提取到配置中
                        signals.loc[data.index[i], "confidence"] = (
                            80  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                        )
                    else:
                        signals.loc[data.index[i], "score"] = 65  # TODO: 将魔法数字提取到配置中
                        signals.loc[data.index[i], "confidence"] = (
                            70  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                        )

            # 检查看跌形态
            bearish_found = False
            for pattern in bearish_patterns:
                if pattern in pattern_results.columns and pattern_results[pattern].iloc[i]:
                    bearish_found = True
                    pattern_desc.append(pattern)
                    # 强看跌形态
                    if pattern in strong_bearish_patterns:
                        signals.loc[data.index[i], "score"] = 25  # TODO: 将魔法数字提取到配置中
                        signals.loc[data.index[i], "confidence"] = (
                            80  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                        )
                    else:
                        signals.loc[data.index[i], "score"] = 35  # TODO: 将魔法数字提取到配置中
                        signals.loc[data.index[i], "confidence"] = (
                            70  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                        )

            # 设置信号标志
            if bullish_found and not bearish_found:
                signals.loc[data.index[i], "buy_signal"] = True
                signals.loc[data.index[i], "sell_signal"] = False
                signals.loc[data.index[i], "neutral_signal"] = False
                signals.loc[data.index[i], "trend"] = 1
                signals.loc[data.index[i], "signal_type"] = "看涨形态"
            elif bearish_found and not bullish_found:
                signals.loc[data.index[i], "buy_signal"] = False
                signals.loc[data.index[i], "sell_signal"] = True
                signals.loc[data.index[i], "neutral_signal"] = False
                signals.loc[data.index[i], "trend"] = -1
                signals.loc[data.index[i], "signal_type"] = "看跌形态"

            # 当出现多个信号时，可能有冲突
            if bullish_found and bearish_found:
                # 这种情况我们保持中性，但仍然记录形态
                signals.loc[data.index[i], "neutral_signal"] = True
                signals.loc[data.index[i], "score"] = 50  # TODO: 将魔法数字提取到配置中
                signals.loc[data.index[i], "signal_type"] = "混合形态"

            # 设置信号描述
            if pattern_desc:
                signals.loc[data.index[i], "signal_desc"] = ", ".join(pattern_desc)

            # 设置止损位
            if bullish_found:
                # 设置在当前K线的最低点下方
                signals.loc[data.index[i], "stop_loss"] = data["low"].iloc[i] * 0.98  # TODO: 将魔法数字提取到配置中
                # 设置仓位
                signals.loc[data.index[i], "position_size"] = (
                    0.3 if signals.loc[data.index[i], "score"] > 70 else 0.2
                )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                # 设置风险级别
                signals.loc[data.index[i], "risk_level"] = (
                    "低" if signals.loc[data.index[i], "score"] > 70 else "中"
                )  # TODO: 将魔法数字提取到配置中
            elif bearish_found:
                # 设置在当前K线的最高点上方
                signals.loc[data.index[i], "stop_loss"] = data["high"].iloc[i] * 1.02
                # 设置仓位
                signals.loc[data.index[i], "position_size"] = (
                    0.3 if signals.loc[data.index[i], "score"] < 30 else 0.2
                )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                # 设置风险级别
                signals.loc[data.index[i], "risk_level"] = (
                    "低" if signals.loc[data.index[i], "score"] < 30 else "中"
                )  # TODO: 将魔法数字提取到配置中

            # 分析市场环境
            if i >= 20:  # 需要一定的历史数据  # TODO: 将魔法数字提取到配置中
                # 简单的趋势判断
                recent_trend = (data["close"].iloc[i] - data["close"].iloc[i - 20]) / data["close"].iloc[
                    i - 20
                ]  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                if recent_trend > 0.05:  # TODO: 将魔法数字提取到配置中
                    signals.loc[data.index[i], "market_env"] = "上升趋势"
                elif recent_trend < -0.05:  # TODO: 将魔法数字提取到配置中
                    signals.loc[data.index[i], "market_env"] = "下降趋势"
                else:
                    signals.loc[data.index[i], "market_env"] = "横盘整理"

        # 添加成交量确认
        if "volume" in data.columns:
            for i in range(1, len(data)):
                if data["volume"].iloc[i] > data["volume"].iloc[i - 1] * 1.2:  # 成交量放大20%
                    signals.loc[data.index[i], "volume_confirmation"] = True
                    # 成交量确认增加信号置信度
                    signals.loc[data.index[i], "confidence"] = min(100, signals.loc[data.index[i], "confidence"] + 10)

        return signals

    def calculate_confidence_Patterns(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """
        计算Candlestick_patterns指标的置信度

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

        # 2. 基于数据质量的置信度
        if hasattr(self, "_result") and self._result is not None:
            # 检查是否有形态数据
            pattern_columns = [
                col for col in self._result.columns if any(pattern.name.lower() in col for pattern in Pattern_type)
            ]
            if pattern_columns:
                # 形态数据越完整，置信度越高
                data_completeness = len(pattern_columns) / len(Pattern_type)
                confidence += data_completeness * 0.1

        # 3. 基于形态的置信度  # TODO: 将魔法数字提取到配置中
        if not patterns.empty:
            # 检查CandlestickPatterns形态（只计算布尔列）
            bool_columns = patterns.select_dtypes(include=[bool]).columns
            if len(bool_columns) > 0:
                pattern_count = patterns[bool_columns].sum().sum()
                if pattern_count > 0:
                    confidence += min(pattern_count * 0.02, 0.15)  # TODO: 将魔法数字提取到配置中

        # 4. 基于信号的置信度  # TODO: 将魔法数字提取到配置中
        if signals:
            # 检查信号强度
            signal_count = sum(1 for signal in signals.values() if hasattr(signal, "any") and signal.any())
            if signal_count > 0:
                confidence += min(signal_count * 0.05, 0.1)  # TODO: 将魔法数字提取到配置中

        # 5. 基于数据长度的置信度  # TODO: 将魔法数字提取到配置中
        if len(score) >= 60:  # 两个月数据  # TODO: 将魔法数字提取到配置中
            confidence += 0.1
        elif len(score) >= 30:  # 一个月数据  # TODO: 将魔法数字提取到配置中
            confidence += 0.05  # TODO: 将魔法数字提取到配置中

        # 确保置信度在0-1范围内
        return max(0.0, min(1.0, confidence))

    def register_patterns_Patterns(self):
        """
        注册Candlestick_patterns指标的形态到全局形态注册表
        """
        # 注册单日看涨形态
        self.register_pattern_to_registry(
            pattern_id="HAMMER",
            display_name="锤头线",
            description="小实体，长下影线，几乎无上影线，底部反转信号",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=20.0,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            polarity="POSITIVE",
        )

        self.register_pattern_to_registry(
            pattern_id="LONG_LEGGED_DOJI",
            display_name="长腿十字",
            description="十字星带长下影线，表明买卖力量均衡但下方有支撑",
            pattern_type="BULLISH",
            default_strength="WEAK",
            score_impact=15.0,  # TODO: 将魔法数字提取到配置中
            polarity="POSITIVE",
        )

        self.register_pattern_to_registry(
            pattern_id="SINGLE_NEEDLE_BOTTOM",
            display_name="单针探底",
            description="长下影线，表明下方有强力买盘支撑",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=25.0,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            polarity="POSITIVE",
        )

        # 注册单日看跌形态
        self.register_pattern_to_registry(
            pattern_id="HANGING_MAN",
            display_name="吊颈线",
            description="小实体，长上影线，几乎无下影线，顶部反转信号",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-20.0,  # TODO: 将魔法数字提取到配置中
            polarity="NEGATIVE",
        )

        self.register_pattern_to_registry(
            pattern_id="GRAVESTONE_DOJI",
            display_name="墓碑线",
            description="十字星带长上影线，表明上方抛压沉重",
            pattern_type="BEARISH",
            default_strength="WEAK",
            score_impact=-15.0,  # TODO: 将魔法数字提取到配置中
            polarity="NEGATIVE",
        )

        self.register_pattern_to_registry(
            pattern_id="SHOOTING_STAR",
            display_name="射击之星",
            description="小实体，长上影线，短下影线，顶部反转信号",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-25.0,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            polarity="NEGATIVE",
        )

        # 注册组合看涨形态
        self.register_pattern_to_registry(
            pattern_id="ENGULFING_BULLISH",
            display_name="阳包阴",
            description="阳线完全包含前一天阴线，强烈的底部反转信号",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=30.0,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            polarity="POSITIVE",
        )

        self.register_pattern_to_registry(
            pattern_id="PIERCING_LINE",
            display_name="曙光初现",
            description="阴线后接长阳线，阳线开盘价低于前日最低价",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=25.0,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            polarity="POSITIVE",
        )

        self.register_pattern_to_registry(
            pattern_id="MORNING_STAR",
            display_name="启明星",
            description="长阴线+十字星+长阳线，经典的底部反转形态",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=35.0,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            polarity="POSITIVE",
        )

        self.register_pattern_to_registry(
            pattern_id="HARAMI_BULLISH",
            display_name="好友反攻",
            description="长阴线后第二天以低于前日收盘价开盘，收于前日开盘价之上",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=20.0,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            polarity="POSITIVE",
        )

        # 注册组合看跌形态
        self.register_pattern_to_registry(
            pattern_id="ENGULFING_BEARISH",
            display_name="阴包阳",
            description="阴线完全包含前一天阳线，强烈的顶部反转信号",
            pattern_type="BEARISH",
            default_strength="VERY_STRONG",
            score_impact=-30.0,  # TODO: 将魔法数字提取到配置中
            polarity="NEGATIVE",
        )

        self.register_pattern_to_registry(
            pattern_id="DARK_CLOUD_COVER",
            display_name="乌云盖顶",
            description="阳线后接长阴线，阴线开盘价高于前日最高价",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-25.0,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            polarity="NEGATIVE",
        )

        self.register_pattern_to_registry(
            pattern_id="EVENING_STAR",
            display_name="黄昏星",
            description="长阳线+十字星+长阴线，经典的顶部反转形态",
            pattern_type="BEARISH",
            default_strength="VERY_STRONG",
            score_impact=-35.0,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            polarity="NEGATIVE",
        )

        # 注册复合形态
        self.register_pattern_to_registry(
            pattern_id="HEAD_SHOULDERS_BOTTOM",
            display_name="头肩底",
            description="三个波谷，中间低于两侧，强烈的底部反转形态",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=40.0,  # TODO: 将魔法数字提取到配置中
            polarity="POSITIVE",
        )

        self.register_pattern_to_registry(
            pattern_id="HEAD_SHOULDERS_TOP",
            display_name="头肩顶",
            description="三个波峰，中间高于两侧，强烈的顶部反转形态",
            pattern_type="BEARISH",
            default_strength="VERY_STRONG",
            score_impact=-40.0,  # TODO: 将魔法数字提取到配置中
            polarity="NEGATIVE",
        )

        self.register_pattern_to_registry(
            pattern_id="DOUBLE_BOTTOM",
            display_name="双底",
            description="W形价格形态，强烈的底部反转信号",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=35.0,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            polarity="POSITIVE",
        )

        self.register_pattern_to_registry(
            pattern_id="DOUBLE_TOP",
            display_name="双顶",
            description="M形价格形态，强烈的顶部反转信号",
            pattern_type="BEARISH",
            default_strength="VERY_STRONG",
            score_impact=-35.0,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            polarity="NEGATIVE",
        )

        self.register_pattern_to_registry(
            pattern_id="V_REVERSAL",
            display_name="V形反转",
            description="急速下跌后快速反弹，快速反转形态",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=30.0,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            polarity="POSITIVE",
        )

        self.register_pattern_to_registry(
            pattern_id="ISLAND_REVERSAL",
            display_name="岛型反转",
            description="跳空+反向跳空形成孤岛，强烈的反转信号，方向需结合趋势判断",
            pattern_type="NEUTRAL",
            default_strength="VERY_STRONG",
            score_impact=0.0,
            polarity="NEUTRAL",
        )

        # 注册中性形态
        self.register_pattern_to_registry(
            pattern_id="DOJI",
            display_name="十字星",
            description="开盘价与收盘价接近，上下影线明显，表明市场犹豫",
            pattern_type="NEUTRAL",
            default_strength="WEAK",
            score_impact=0.0,
            polarity="NEUTRAL",
        )

    def generate_trading_signals_Patterns(self, data: pd.DataFrame, **kwargs) -> dict:
        """
        生成Candlestick_patterns交易信号

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            dict: 包含买卖信号的字典
        """
        # 确保已计算指标
        if not self.has_result():
            self._calculate_candlestickpatterns(data, **kwargs)

        if self._result is None or self._result.empty:
            return {
                "buy_signal": pd.Series(False, index=data.index),
                "sell_signal": pd.Series(False, index=data.index),
                "signal_strength": pd.Series(0.0, index=data.index),
            }

        # 初始化信号
        buy_signal = pd.Series(False, index=data.index)
        sell_signal = pd.Series(False, index=data.index)
        signal_strength = pd.Series(0.0, index=data.index)

        # 定义看涨形态
        bullish_patterns = [
            Pattern_type.HAMMER.name.lower(),
            Pattern_type.MORNING_STAR.name.lower(),
            Pattern_type.PIERCING_LINE.name.lower(),
            Pattern_type.ENGULFING_BULLISH.name.lower(),
            Pattern_type.HARAMI_BULLISH.name.lower(),
            Pattern_type.SINGLE_NEEDLE_BOTTOM.name.lower(),
            Pattern_type.HEAD_SHOULDERS_BOTTOM.name.lower(),
            Pattern_type.DOUBLE_BOTTOM.name.lower(),
            Pattern_type.V_REVERSAL.name.lower(),
        ]

        # 定义看跌形态
        bearish_patterns = [
            Pattern_type.HANGING_MAN.name.lower(),
            Pattern_type.EVENING_STAR.name.lower(),
            Pattern_type.DARK_CLOUD_COVER.name.lower(),
            Pattern_type.ENGULFING_BEARISH.name.lower(),
            Pattern_type.SHOOTING_STAR.name.lower(),
            Pattern_type.HEAD_SHOULDERS_TOP.name.lower(),
            Pattern_type.DOUBLE_TOP.name.lower(),
        ]

        # 强形态权重
        strong_patterns = {
            Pattern_type.MORNING_STAR.name.lower(): 0.9,  # TODO: 将魔法数字提取到配置中
            Pattern_type.EVENING_STAR.name.lower(): -0.9,  # TODO: 将魔法数字提取到配置中
            Pattern_type.ENGULFING_BULLISH.name.lower(): 0.8,  # TODO: 将魔法数字提取到配置中
            Pattern_type.ENGULFING_BEARISH.name.lower(): -0.8,  # TODO: 将魔法数字提取到配置中
            Pattern_type.HEAD_SHOULDERS_BOTTOM.name.lower(): 0.9,  # TODO: 将魔法数字提取到配置中
            Pattern_type.HEAD_SHOULDERS_TOP.name.lower(): -0.9,  # TODO: 将魔法数字提取到配置中
            Pattern_type.DOUBLE_BOTTOM.name.lower(): 0.8,  # TODO: 将魔法数字提取到配置中
            Pattern_type.DOUBLE_TOP.name.lower(): -0.8,  # TODO: 将魔法数字提取到配置中
        }

        # 生成信号
        for pattern in bullish_patterns:
            if pattern in self._result.columns:
                pattern_mask = self._result[pattern]
                buy_signal |= pattern_mask

                # 设置信号强度
                if pattern in strong_patterns:
                    signal_strength[pattern_mask] = strong_patterns[pattern]
                else:
                    signal_strength[pattern_mask] = 0.6  # TODO: 将魔法数字提取到配置中

        for pattern in bearish_patterns:
            if pattern in self._result.columns:
                pattern_mask = self._result[pattern]
                sell_signal |= pattern_mask

                # 设置信号强度
                if pattern in strong_patterns:
                    signal_strength[pattern_mask] = strong_patterns[pattern]
                else:
                    signal_strength[pattern_mask] = -0.6  # TODO: 将魔法数字提取到配置中

        # 处理岛型反转（需要结合趋势判断）
        if Pattern_type.ISLAND_REVERSAL.name.lower() in self._result.columns:
            island_mask = self._result[Pattern_type.ISLAND_REVERSAL.name.lower()]
            if island_mask.any() and len(data) >= 5:  # TODO: 将魔法数字提取到配置中
                # 简单趋势判断
                _5d = data["close"].pct_change(5)  # TODO: 将魔法数字提取到配置中

                # 在上升趋势后的岛型反转（看跌）
                bearish_island = island_mask & (_5d > 0.05)  # TODO: 将魔法数字提取到配置中
                sell_signal |= bearish_island
                signal_strength[bearish_island] = -0.8  # TODO: 将魔法数字提取到配置中

                # 在下降趋势后的岛型反转（看涨）
                bullish_island = island_mask & (_5d < -0.05)  # TODO: 将魔法数字提取到配置中
                buy_signal |= bullish_island
                signal_strength[bullish_island] = 0.8  # TODO: 将魔法数字提取到配置中

        # 标准化信号强度
        signal_strength = signal_strength.clip(-1, 1)

        return {"buy_signal": buy_signal, "sell_signal": sell_signal, "signal_strength": signal_strength}

    def get_indicator_type_Patterns(self) -> str:
        """
        获取指标类型

        Returns:
            str: 指标类型
        """
        return "CANDLESTICKPATTERNS"

    def get_pattern_info_Patterns(self, pattern_id: str) -> dict:
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

    # ==================== BaseIndicator抽象方法实现 ====================

    def _calculate_baseindicator(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """抽象基类要求的计算方法"""
        return self.calculate(data, *args, **kwargs)

    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """抽象基类要求的评分方法"""
        if len(data) < self.minimum_periods:
            return pd.Series(50.0, index=data.index)  # TODO: 将魔法数字提取到配置中

        # 计算形态识别结果
        patterns_result = self.calculate(data)

        # 基于形态强度计算评分
        score = pd.Series(50.0, index=data.index)  # 默认中性评分  # TODO: 将魔法数字提取到配置中

        if isinstance(patterns_result, pd.DataFrame) and not patterns_result.empty:
            # 查找强度列
            strength_cols = [col for col in patterns_result.columns if "strength" in col.lower()]
            if strength_cols:
                # 使用第一个强度列
                strength_col = strength_cols[0]
                score = patterns_result[strength_col].fillna(50.0)  # TODO: 将魔法数字提取到配置中

        return score

    def get_patterns_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """抽象基类要求的形态方法"""
        if len(data) < self.minimum_periods:
            return pd.DataFrame(index=data.index)

        # 计算形态识别结果
        patterns_result = self.calculate(data)

        if isinstance(patterns_result, pd.DataFrame):
            return patterns_result
        else:
            # 如果返回的不是DataFrame，创建一个空的DataFrame
            return pd.DataFrame(index=data.index)

    def calculate_confidence_Indicator_Base_Indicator(
        self, score: pd.Series, patterns: List[str], signals: Dict[str, pd.Series]
    ) -> float:
        """抽象基类要求的置信度方法"""
        base_confidence = 0.6  # TODO: 将魔法数字提取到配置中

        # 基于形态数量调整置信度
        if patterns:
            pattern_bonus = min(0.2, len(patterns) * 0.05)  # TODO: 将魔法数字提取到配置中
            base_confidence += pattern_bonus

        # 基于信号强度调整置信度
        if signals:
            signal_strength = sum(1 for signal in signals.values() if len(signal) > 0 and signal.iloc[-1])
            signal_bonus = min(0.2, signal_strength * 0.1)
            base_confidence += signal_bonus

        return min(1.0, base_confidence)

    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        获取K线形态交易信号
        
        CandlestickPatterns基类的默认信号实现
        用于AdvancedCandlestickPatterns等需要实例化基类的场景
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            Dict[str, Any]: 标准化交易信号格式
        """
        try:
            # 数据验证
            if not isinstance(data, pd.DataFrame) or data.empty:
                return self._get_default_signal_base("数据验证失败")
            
            # 检查必需列
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                return self._get_default_signal_base("缺少必需列")
            
            # 计算形态结果
            result_data = self.calculate(data)
            
            # 分析所有形态列，寻找最强信号
            pattern_signals = []
            
            # 检查各种形态
            if isinstance(result_data, pd.DataFrame) and len(result_data) > 0:
                last_row = result_data.iloc[-1]
                
                # 看涨形态
                bullish_patterns = ['hammer', 'morning_star', 'piercing_line', 'engulfing_bullish', 'three_white_soldiers']
                bullish_count = sum(1 for pattern in bullish_patterns if pattern in last_row and last_row[pattern])
                
                # 看跌形态
                bearish_patterns = ['shooting_star', 'evening_star', 'dark_cloud_cover', 'engulfing_bearish', 'three_black_crows']
                bearish_count = sum(1 for pattern in bearish_patterns if pattern in last_row and last_row[pattern])
                
                # 中性形态
                neutral_patterns = ['doji']
                neutral_count = sum(1 for pattern in neutral_patterns if pattern in last_row and last_row[pattern])
                
                # 确定主要信号类型
                if bullish_count > bearish_count and bullish_count > 0:
                    signal_type = "buy"
                    strength = min(0.8, 0.5 + bullish_count * 0.1)
                    confidence = min(0.8, 0.6 + bullish_count * 0.05)
                    reason = f"检测到{bullish_count}个看涨形态"
                elif bearish_count > bullish_count and bearish_count > 0:
                    signal_type = "sell"
                    strength = min(0.8, 0.5 + bearish_count * 0.1)
                    confidence = min(0.8, 0.6 + bearish_count * 0.05)
                    reason = f"检测到{bearish_count}个看跌形态"
                elif neutral_count > 0:
                    signal_type = "hold"
                    strength = 0.3
                    confidence = 0.5
                    reason = f"检测到{neutral_count}个中性形态，市场犹豫"
                else:
                    return self._get_default_signal_base("未检测到明确形态")
                
                # 构建元数据
                metadata = {
                    "bullish_patterns": bullish_count,
                    "bearish_patterns": bearish_count,
                    "neutral_patterns": neutral_count,
                    "total_patterns": bullish_count + bearish_count + neutral_count
                }
            else:
                return self._get_default_signal_base("形态计算失败")
            
            # 返回标准化信号格式
            return {
                'signal_type': signal_type,
                'strength': max(0.0, min(1.0, strength)),
                'confidence': max(0.0, min(1.0, confidence)),
                'timestamp': pd.Timestamp.now(),
                'reason': reason,
                'metadata': {
                    'indicator_name': self.name if hasattr(self, 'name') else 'CandlestickPatterns',
                    'pattern_category': 'combined',
                    'pattern_family': 'candlestick',
                    'pattern_characteristic': 'mixed_signals',
                    'requires_confirmation': 'pattern_specific',
                    'signal_direction': 'bidirectional',
                    **metadata
                }
            }
            
        except Exception as e:
            logger.error(f"K线形态信号生成失败: {e}")
            return self._get_default_signal_base(f"信号生成失败: {str(e)}")
    
    def _get_default_signal_base(self, reason: str = "默认持有") -> Dict[str, Any]:
        """获取默认信号（基类版本）"""
        return {
            'signal_type': 'hold',
            'strength': 0.0,
            'confidence': 0.5,
            'timestamp': pd.Timestamp.now(),
            'reason': reason,
            'metadata': {
                'indicator_name': self.name if hasattr(self, 'name') else 'CandlestickPatterns',
                'pattern_category': 'combined',
                'pattern_family': 'candlestick',
                'pattern_characteristic': 'mixed_signals',
                'requires_confirmation': 'pattern_specific',
                'signal_direction': 'bidirectional'
            }
        }

    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """抽象基类要求的参数设置方法"""
        for key, value in kwargs.items():
            if key in self._parameters:
                self._parameters[key] = value
                if key == "period":
                    self.period = value


# ===== 兼容性别名 =====
# 为了向后兼容，提供下划线命名的别名
PATTERN_TYPE = PatterntypePatterns
CANDLESTICK_PATTERNS = CandlestickPatterns

# ===== 单独的形态识别类 =====
# 为指标注册表提供单独的形态识别类


class Doji(CandlestickPatterns):
    """十字星形态识别"""

    def __init__(
        self, period: int = 20
    ):  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        super().__init__(period=period)  # 正确传递period参数
        self.name = "DOJI"
        self.pattern_type = PatterntypePatterns.DOJI

    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        获取十字星形态交易信号
        
        十字星是一种犹豫形态，特征：
        - 开盘价与收盘价接近（实体很小）
        - 上下影线明显存在
        - 表示市场犹豫不决，需要结合趋势背景判断
        - 在上升趋势中可能是顶部反转信号
        - 在下降趋势中可能是底部反转信号

        Args:
            data: 包含OHLCV数据的DataFrame

        Returns:
            Dict[str, Any]: 标准化交易信号格式
        """
        try:
            # 数据验证
            if not self._validate_signal_data(data):
                return self._get_default_signal("数据验证失败")

            # 计算十字星形态（如果数据不是计算结果）
            if 'doji' not in data.columns:
                result_data = self.calculate(data)
            else:
                result_data = data

            # 获取最新十字星信号
            latest_doji = result_data["doji"].iloc[-1] if "doji" in result_data.columns else False

            # 初始化信号参数
            signal_type = "hold"
            strength = 0.5
            confidence = 0.6
            reason = "未检测到十字星形态"
            metadata = {}

            if latest_doji:
                # 十字星本身是犹豫信号，需要结合趋势判断方向
                signal_type = "hold"  # 默认持有，等待确认
                strength = 0.6
                confidence = 0.7
                reason = "检测到十字星形态，市场犹豫信号"
                
                # 计算十字星的具体特征
                latest_open = data["open"].iloc[-1]
                latest_high = data["high"].iloc[-1]
                latest_low = data["low"].iloc[-1]
                latest_close = data["close"].iloc[-1]
                
                # 计算形态强度指标
                body_size = abs(latest_close - latest_open)
                total_range = latest_high - latest_low
                upper_shadow = latest_high - max(latest_close, latest_open)
                lower_shadow = min(latest_close, latest_open) - latest_low
                
                # 计算十字星质量得分
                if total_range > 0:
                    body_to_range_ratio = body_size / total_range
                    upper_shadow_ratio = upper_shadow / total_range
                    lower_shadow_ratio = lower_shadow / total_range
                    
                    # 理想十字星特征评分
                    quality_score = 0.5  # 基础分
                    
                    # 小实体加分
                    if body_to_range_ratio < 0.1:
                        quality_score += 0.3
                    elif body_to_range_ratio < 0.2:
                        quality_score += 0.2
                    
                    # 均衡影线加分
                    shadow_balance = 1 - abs(upper_shadow_ratio - lower_shadow_ratio)
                    quality_score += shadow_balance * 0.2
                    
                    # 影线长度加分
                    if upper_shadow_ratio > 0.2 and lower_shadow_ratio > 0.2:
                        quality_score += 0.1
                    
                    # 根据质量调整信号强度
                    strength = min(0.9, 0.5 + quality_score * 0.4)
                    confidence = min(0.9, 0.6 + quality_score * 0.3)
                    
                    metadata = {
                        "pattern_type": "doji",
                        "body_ratio": round(body_to_range_ratio, 3),
                        "upper_shadow_ratio": round(upper_shadow_ratio, 3),
                        "lower_shadow_ratio": round(lower_shadow_ratio, 3),
                        "shadow_balance": round(shadow_balance, 3),
                        "quality_score": round(quality_score, 3),
                        "reversal_potential": "high" if quality_score > 0.8 else "medium"
                    }

                # 分析趋势环境，判断十字星的具体含义
                if len(data) >= 5:
                    recent_closes = data["close"].tail(5)
                    price_trend = recent_closes.diff().mean()
                    trend_threshold = 0.1  # 趋势判断阈值
                    
                    # 在上升趋势中的十字星 - 可能的顶部反转
                    if price_trend > trend_threshold:
                        signal_type = "sell"
                        strength = min(0.85, strength + 0.2)
                        confidence = min(0.85, confidence + 0.15)
                        reason = "上升趋势中检测到十字星，潜在顶部反转信号"
                        metadata["trend_context"] = "uptrend"
                        metadata["signal_interpretation"] = "top_reversal"
                    
                    # 在下降趋势中的十字星 - 可能的底部反转
                    elif price_trend < -trend_threshold:
                        signal_type = "buy"
                        strength = min(0.85, strength + 0.2)
                        confidence = min(0.85, confidence + 0.15)
                        reason = "下降趋势中检测到十字星，潜在底部反转信号"
                        metadata["trend_context"] = "downtrend"
                        metadata["signal_interpretation"] = "bottom_reversal"
                    
                    # 在横盘中的十字星 - 继续犹豫
                    else:
                        signal_type = "hold"
                        reason = "横盘中检测到十字星，市场持续犹豫"
                        metadata["trend_context"] = "sideways"
                        metadata["signal_interpretation"] = "indecision"

                # 检查成交量确认（十字星+放量更可靠）
                if len(data) >= 2:
                    current_volume = data["volume"].iloc[-1]
                    avg_volume = data["volume"].tail(5).mean()
                    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                    
                    if volume_ratio > 1.5:  # 放量确认
                        strength = min(0.95, strength + 0.1)
                        confidence = min(0.95, confidence + 0.1)
                        metadata["volume_confirmation"] = "high"
                        if signal_type in ["buy", "sell"]:
                            reason += "，成交量放大确认"
                    else:
                        metadata["volume_confirmation"] = "normal"

            # 返回标准化信号格式
            return {
                'signal_type': signal_type,
                'strength': max(0.0, min(1.0, strength)),
                'confidence': max(0.0, min(1.0, confidence)),
                'timestamp': pd.Timestamp.now(),
                'reason': reason,
                'metadata': {
                    'indicator_name': self.name,
                    'pattern_category': 'reversal',
                    'pattern_family': 'candlestick',
                    'pattern_characteristic': 'indecision',
                    **metadata
                }
            }

        except Exception as e:
            logger.error(f"十字星形态信号生成失败: {e}")
            return self._get_default_signal(f"信号生成失败: {str(e)}")

    def _validate_signal_data(self, data: pd.DataFrame) -> bool:
        """验证输入数据"""
        if not isinstance(data, pd.DataFrame):
            return False
        
        if data.empty:
            return False
        
        # 检查必需列
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            return False
        
        # 检查数据量（十字星至少需要1个数据点，但建议更多用于趋势分析）
        if len(data) < 1:
            return False
        
        return True

    def _get_default_signal(self, reason: str = "默认持有") -> Dict[str, Any]:
        """获取默认信号"""
        return {
            'signal_type': 'hold',
            'strength': 0.0,
            'confidence': 0.5,
            'timestamp': pd.Timestamp.now(),
            'reason': reason,
            'metadata': {
                'indicator_name': self.name,
                'pattern_category': 'reversal',
                'pattern_family': 'candlestick',
                'pattern_characteristic': 'indecision'
            }
        }


class Hammer(CandlestickPatterns):
    """锤子线形态识别"""

    def __init__(
        self, period: int = 20
    ):  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        super().__init__(period=period)  # 正确传递period参数
        self.name = "HAMMER"
        self.pattern_type = PatterntypePatterns.HAMMER

    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        获取锤子线形态交易信号
        
        锤子线是一种看涨反转形态，特征：
        - 小实体（开盘价与收盘价接近）
        - 长下影线（通常是实体的2倍以上）
        - 几乎无上影线或很短的上影线
        - 出现在下跌趋势中时具有反转意义

        Args:
            data: 包含OHLCV数据的DataFrame

        Returns:
            Dict[str, Any]: 标准化交易信号格式
        """
        try:
            # 数据验证
            if not self._validate_signal_data(data):
                return self._get_default_signal("数据验证失败")

            # 计算锤子线形态（如果数据不是计算结果）
            if 'hammer' not in data.columns:
                result_data = self.calculate(data)
            else:
                result_data = data

            # 获取最新锤子线信号
            latest_hammer = result_data["hammer"].iloc[-1] if "hammer" in result_data.columns else False

            # 初始化信号参数
            signal_type = "hold"
            strength = 0.5
            confidence = 0.6
            reason = "未检测到锤子线形态"
            metadata = {}

            if latest_hammer:
                # 锤子线通常是看涨反转信号
                signal_type = "buy"
                strength = 0.75
                confidence = 0.8
                reason = "检测到锤子线形态，看涨反转信号"
                
                # 计算锤子线的具体特征
                latest_open = data["open"].iloc[-1]
                latest_high = data["high"].iloc[-1]
                latest_low = data["low"].iloc[-1]
                latest_close = data["close"].iloc[-1]
                
                # 计算形态强度指标
                body_size = abs(latest_close - latest_open)
                total_range = latest_high - latest_low
                lower_shadow = min(latest_open, latest_close) - latest_low
                upper_shadow = latest_high - max(latest_open, latest_close)
                
                # 计算形态质量得分
                if total_range > 0:
                    body_ratio = body_size / total_range
                    lower_shadow_ratio = lower_shadow / total_range if total_range > 0 else 0
                    upper_shadow_ratio = upper_shadow / total_range if total_range > 0 else 0
                    
                    # 理想锤子线特征评分
                    quality_score = 0.5  # 基础分
                    
                    # 小实体加分
                    if body_ratio < 0.3:
                        quality_score += 0.2
                    
                    # 长下影线加分
                    if lower_shadow_ratio > 0.5:
                        quality_score += 0.2
                    
                    # 短上影线加分
                    if upper_shadow_ratio < 0.1:
                        quality_score += 0.1
                    
                    # 根据质量调整信号强度
                    strength = min(0.9, 0.6 + quality_score * 0.3)
                    confidence = min(0.9, 0.6 + quality_score * 0.3)
                    
                    metadata = {
                        "pattern_type": "hammer",
                        "body_ratio": round(body_ratio, 3),
                        "lower_shadow_ratio": round(lower_shadow_ratio, 3),
                        "upper_shadow_ratio": round(upper_shadow_ratio, 3),
                        "quality_score": round(quality_score, 3),
                        "reversal_potential": "high" if quality_score > 0.8 else "medium"
                    }

            # 检查趋势环境（锤子线在下跌趋势中更有效）
            if len(data) >= 5:
                recent_closes = data["close"].tail(5)
                is_downtrend = (recent_closes.iloc[-1] < recent_closes.iloc[0]) and \
                              (recent_closes.diff().mean() < 0)
                
                if latest_hammer and is_downtrend:
                    strength = min(0.95, strength + 0.15)
                    confidence = min(0.95, confidence + 0.1)
                    reason = "在下跌趋势中检测到锤子线，强烈看涨反转信号"
                    metadata["trend_context"] = "downtrend"
                    metadata["signal_strength"] = "enhanced"
                elif latest_hammer:
                    metadata["trend_context"] = "sideways"
                    metadata["signal_strength"] = "normal"

            # 返回标准化信号格式
            return {
                'signal_type': signal_type,
                'strength': max(0.0, min(1.0, strength)),
                'confidence': max(0.0, min(1.0, confidence)),
                'timestamp': pd.Timestamp.now(),
                'reason': reason,
                'metadata': {
                    'indicator_name': self.name,
                    'pattern_category': 'reversal',
                    'pattern_family': 'candlestick',
                    **metadata
                }
            }

        except Exception as e:
            logger.error(f"锤子线形态信号生成失败: {e}")
            return self._get_default_signal(f"信号生成失败: {str(e)}")

    def _validate_signal_data(self, data: pd.DataFrame) -> bool:
        """验证输入数据"""
        if not isinstance(data, pd.DataFrame):
            return False
        
        if data.empty:
            return False
        
        # 检查必需列
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            return False
        
        # 检查数据量（锤子线至少需要1个数据点，但建议更多用于趋势分析）
        if len(data) < 1:
            return False
        
        return True

    def _get_default_signal(self, reason: str = "默认持有") -> Dict[str, Any]:
        """获取默认信号"""
        return {
            'signal_type': 'hold',
            'strength': 0.0,
            'confidence': 0.5,
            'timestamp': pd.Timestamp.now(),
            'reason': reason,
            'metadata': {
                'indicator_name': self.name,
                'pattern_category': 'reversal',
                'pattern_family': 'candlestick'
            }
        }


class ShootingStar(CandlestickPatterns):
    """流星线形态识别"""

    def __init__(
        self, period: int = 20
    ):  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        super().__init__(period=period)  # 正确传递period参数
        self.name = "SHOOTING_STAR"
        self.pattern_type = PatterntypePatterns.SHOOTING_STAR

    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        获取射击之星形态交易信号
        
        射击之星形态是一种看跌反转信号，与锤子线完全对称，特征：
        - 小实体：开盘价和收盘价接近，实体部分相对较小
        - 长上影线：上影线长度至少是实体的2倍，显示上方抛压
        - 短下影线：下影线很短或没有，显示下方支撑有限
        - 位置要求：出现在上涨趋势的高位，预示看跌反转
        - 与锤子线对称：锤子线是看涨反转（长下影线），射击之星是看跌反转（长上影线）
        - 确认要求：需要下一个交易日的弱势确认

        Args:
            data: 包含OHLCV数据的DataFrame

        Returns:
            Dict[str, Any]: 标准化交易信号格式
        """
        try:
            # 数据验证
            if not self._validate_signal_data(data):
                return self._get_default_signal("数据验证失败")

            # 计算射击之星形态（如果数据不是计算结果）
            if 'shooting_star' not in data.columns:
                result_data = self.calculate(data)
            else:
                result_data = data

            # 获取最新射击之星形态信号
            latest_shooting_star = result_data["shooting_star"].iloc[-1] if "shooting_star" in result_data.columns else False

            # 初始化信号参数
            signal_type = "hold"
            strength = 0.5
            confidence = 0.6
            reason = "未检测到射击之星形态"
            metadata = {}

            # 处理射击之星形态
            if latest_shooting_star:
                signal_type = "sell"
                strength = 0.75  # 射击之星是强看跌反转信号，强度低于锤子线（0.80）
                confidence = 0.75
                reason = "检测到射击之星形态，看跌反转信号"
                
                # 计算射击之星形态的具体特征
                if len(data) >= 1:
                    # 最新K线数据
                    current_open = data["open"].iloc[-1]
                    current_high = data["high"].iloc[-1]
                    current_low = data["low"].iloc[-1]
                    current_close = data["close"].iloc[-1]
                    
                    # 计算实体和影线
                    body_size = abs(current_close - current_open)
                    upper_shadow = current_high - max(current_open, current_close)
                    lower_shadow = min(current_open, current_close) - current_low
                    total_range = current_high - current_low
                    
                    # 射击之星质量评分
                    quality_score = 0.5  # 基础分
                    
                    if body_size > 0 and total_range > 0:
                        # 实体占比评分（实体越小越好）
                        body_ratio = body_size / total_range
                        if body_ratio <= 0.2:  # 实体占比小于20%
                            quality_score += 0.2
                        elif body_ratio <= 0.3:  # 实体占比小于30%
                            quality_score += 0.1
                        
                        # 上影线长度评分（上影线是关键特征）
                        if body_size > 0:
                            upper_shadow_ratio = upper_shadow / body_size
                            if upper_shadow_ratio >= 3.0:  # 上影线是实体的3倍以上
                                quality_score += 0.2
                            elif upper_shadow_ratio >= 2.0:  # 上影线是实体的2倍以上
                                quality_score += 0.15
                            elif upper_shadow_ratio >= 1.5:  # 上影线是实体的1.5倍以上
                                quality_score += 0.1
                        
                        # 下影线短度评分（下影线越短越好）
                        lower_shadow_ratio = lower_shadow / total_range if total_range > 0 else 0
                        if lower_shadow_ratio <= 0.1:  # 下影线占比小于10%
                            quality_score += 0.15
                        elif lower_shadow_ratio <= 0.2:  # 下影线占比小于20%
                            quality_score += 0.1
                        elif lower_shadow_ratio <= 0.3:  # 下影线占比小于30%
                            quality_score += 0.05
                        
                        # 根据质量调整信号强度
                        strength = min(0.95, 0.65 + quality_score * 0.25)
                        confidence = min(0.95, 0.65 + quality_score * 0.25)
                        
                        metadata = {
                            "pattern_type": "shooting_star",
                            "body_ratio": round(body_ratio, 3),
                            "upper_shadow_ratio": round(upper_shadow_ratio if 'upper_shadow_ratio' in locals() else 0.0, 3),
                            "lower_shadow_ratio": round(lower_shadow_ratio, 3),
                            "quality_score": round(quality_score, 3),
                            "reversal_potential": "high" if quality_score > 0.8 else "medium",
                            "candle_type": "bearish" if current_close < current_open else "bullish",
                            "upper_shadow_length": round(upper_shadow, 3),
                            "lower_shadow_length": round(lower_shadow, 3),
                            "body_size": round(body_size, 3),
                            "total_range": round(total_range, 3)
                        }

            # 检查成交量确认（射击之星+放量更可靠）
            if signal_type == "sell" and len(data) >= 5:
                current_volume = data["volume"].iloc[-1]
                avg_volume = data["volume"].tail(5).mean()
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                
                if volume_ratio > 1.5:  # 明显放量
                    strength = min(0.95, strength + 0.1)
                    confidence = min(0.95, confidence + 0.1)
                    metadata["volume_confirmation"] = "high"
                    reason += "，成交量放大确认"
                elif volume_ratio > 1.2:  # 适度放量
                    strength = min(0.90, strength + 0.05)
                    confidence = min(0.90, confidence + 0.05)
                    metadata["volume_confirmation"] = "moderate"
                else:
                    metadata["volume_confirmation"] = "normal"
                
                metadata["volume_ratio"] = round(volume_ratio, 3)

            # 分析趋势背景，增强射击之星形态的信号强度
            if signal_type == "sell" and len(data) >= 8:
                recent_closes = data["close"].tail(8)
                price_trend = recent_closes.diff().mean()
                
                # 射击之星在上涨趋势顶部最有效（顶部反转）
                if price_trend > 0.1:
                    strength = min(0.95, strength + 0.1)
                    confidence = min(0.95, confidence + 0.1)
                    metadata["trend_context"] = "uptrend_reversal"
                    metadata["signal_enhancement"] = "top_reversal"
                    reason = reason.replace("看跌反转信号", "强顶部反转信号")
                
                # 射击之星在下跌趋势中效果一般（继续下跌信号）
                elif price_trend < -0.05:
                    strength = min(0.85, strength + 0.02)
                    confidence = min(0.85, confidence + 0.02)
                    metadata["trend_context"] = "downtrend_continuation"
                    metadata["signal_enhancement"] = "bearish_continuation"
                    reason = reason.replace("看跌反转信号", "下跌延续信号")
                
                else:
                    metadata["trend_context"] = "sideways"
                    metadata["signal_enhancement"] = "normal"

            # 检查前期阻力位确认
            if signal_type == "sell" and len(data) >= 12:
                # 检查是否在重要阻力位附近
                recent_highs = data["high"].tail(12)
                current_high_area = data["high"].iloc[-3:-1].max()  # 前两根K线的最高价区域
                resistance_levels = recent_highs[recent_highs >= current_high_area * 0.98]  # 2%容忍度
                
                if len(resistance_levels) >= 3:  # 多次测试的阻力位
                    strength = min(0.95, strength + 0.08)
                    confidence = min(0.95, confidence + 0.08)
                    metadata["resistance_confirmation"] = "strong"
                    reason += "，重要阻力位确认"
                elif len(resistance_levels) >= 2:
                    metadata["resistance_confirmation"] = "moderate"
                else:
                    metadata["resistance_confirmation"] = "none"

            # 返回标准化信号格式
            return {
                'signal_type': signal_type,
                'strength': max(0.0, min(1.0, strength)),
                'confidence': max(0.0, min(1.0, confidence)),
                'timestamp': pd.Timestamp.now(),
                'reason': reason,
                'metadata': {
                    'indicator_name': self.name,
                    'pattern_category': 'reversal',
                    'pattern_family': 'candlestick',
                    'pattern_characteristic': 'bearish_reversal',
                    'requires_confirmation': 'single_candle',
                    'signal_direction': 'bearish_only',
                    'shadow_characteristic': 'long_upper_shadow',
                    'position_requirement': 'uptrend_top',
                    **metadata
                }
            }

        except Exception as e:
            logger.error(f"射击之星形态信号生成失败: {e}")
            return self._get_default_signal(f"信号生成失败: {str(e)}")

    def _validate_signal_data(self, data: pd.DataFrame) -> bool:
        """验证输入数据"""
        if not isinstance(data, pd.DataFrame):
            return False
        
        if data.empty:
            return False
        
        # 检查必需列
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            return False
        
        # 检查数据量（射击之星形态至少需要1个数据点）
        if len(data) < 1:
            return False
        
        return True

    def _get_default_signal(self, reason: str = "默认持有") -> Dict[str, Any]:
        """获取默认信号"""
        return {
            'signal_type': 'hold',
            'strength': 0.0,
            'confidence': 0.5,
            'timestamp': pd.Timestamp.now(),
            'reason': reason,
            'metadata': {
                'indicator_name': self.name,
                'pattern_category': 'reversal',
                'pattern_family': 'candlestick',
                'pattern_characteristic': 'bearish_reversal',
                'requires_confirmation': 'single_candle',
                'signal_direction': 'bearish_only',
                'shadow_characteristic': 'long_upper_shadow',
                'position_requirement': 'uptrend_top'
            }
        }


class Engulfing(CandlestickPatterns):
    """吞没形态识别"""

    def __init__(
        self, period: int = 20
    ):  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        super().__init__(period=period)  # 正确传递period参数
        self.name = "ENGULFING"
        self.pattern_type = PatterntypePatterns.ENGULFING_BULLISH  # 修复：使用正确的枚举名称

    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        获取包含形态交易信号
        
        包含形态是一种强烈的反转信号，特征：
        - 看涨包含：前一天阴线被后一天阳线完全包含
        - 看跌包含：前一天阳线被后一天阴线完全包含
        - 包含的K线实体更大，显示出明确的趋势反转
        - 需要至少两个K线来形成包含形态

        Args:
            data: 包含OHLCV数据的DataFrame

        Returns:
            Dict[str, Any]: 标准化交易信号格式
        """
        try:
            # 数据验证
            if not self._validate_signal_data(data):
                return self._get_default_signal("数据验证失败")

            # 计算包含形态（如果数据不是计算结果）
            if 'engulfing_bullish' not in data.columns or 'engulfing_bearish' not in data.columns:
                result_data = self.calculate(data)
            else:
                result_data = data

            # 获取最新包含形态信号
            latest_bullish = result_data["engulfing_bullish"].iloc[-1] if "engulfing_bullish" in result_data.columns else False
            latest_bearish = result_data["engulfing_bearish"].iloc[-1] if "engulfing_bearish" in result_data.columns else False

            # 初始化信号参数
            signal_type = "hold"
            strength = 0.5
            confidence = 0.6
            reason = "未检测到包含形态"
            metadata = {}

            # 处理看涨包含形态
            if latest_bullish:
                signal_type = "buy"
                strength = 0.8
                confidence = 0.8
                reason = "检测到看涨包含形态，强烈看涨反转信号"
                
                # 计算包含形态的具体特征
                if len(data) >= 2:
                    # 前一天（被包含的K线）
                    prev_open = data["open"].iloc[-2]
                    prev_high = data["high"].iloc[-2]
                    prev_low = data["low"].iloc[-2]
                    prev_close = data["close"].iloc[-2]
                    
                    # 当前天（包含的K线）
                    curr_open = data["open"].iloc[-1]
                    curr_high = data["high"].iloc[-1]
                    curr_low = data["low"].iloc[-1]
                    curr_close = data["close"].iloc[-1]
                    
                    # 计算包含程度
                    prev_body = abs(prev_close - prev_open)
                    curr_body = abs(curr_close - curr_open)
                    body_ratio = curr_body / prev_body if prev_body > 0 else 2.0
                    
                    # 计算包含覆盖度
                    prev_range = prev_high - prev_low
                    curr_range = curr_high - curr_low
                    range_expansion = curr_range / prev_range if prev_range > 0 else 1.5
                    
                    # 包含质量评分
                    quality_score = 0.5  # 基础分
                    
                    # 实体大小比例加分
                    if body_ratio > 2.0:
                        quality_score += 0.3
                    elif body_ratio > 1.5:
                        quality_score += 0.2
                    
                    # 范围扩展加分
                    if range_expansion > 1.3:
                        quality_score += 0.2
                    elif range_expansion > 1.1:
                        quality_score += 0.1
                    
                    # 前一天是阴线加分
                    if prev_close < prev_open:
                        quality_score += 0.1
                    
                    # 根据质量调整信号强度
                    strength = min(0.95, 0.7 + quality_score * 0.25)
                    confidence = min(0.95, 0.7 + quality_score * 0.25)
                    
                    metadata = {
                        "pattern_type": "bullish_engulfing",
                        "engulfing_type": "bullish",
                        "body_ratio": round(body_ratio, 3),
                        "range_expansion": round(range_expansion, 3),
                        "quality_score": round(quality_score, 3),
                        "reversal_strength": "high" if quality_score > 0.8 else "medium",
                        "prev_candle_type": "bearish" if prev_close < prev_open else "bullish"
                    }

            # 处理看跌包含形态
            elif latest_bearish:
                signal_type = "sell"
                strength = 0.8
                confidence = 0.8
                reason = "检测到看跌包含形态，强烈看跌反转信号"
                
                # 计算包含形态的具体特征
                if len(data) >= 2:
                    # 前一天（被包含的K线）
                    prev_open = data["open"].iloc[-2]
                    prev_high = data["high"].iloc[-2]
                    prev_low = data["low"].iloc[-2]
                    prev_close = data["close"].iloc[-2]
                    
                    # 当前天（包含的K线）
                    curr_open = data["open"].iloc[-1]
                    curr_high = data["high"].iloc[-1]
                    curr_low = data["low"].iloc[-1]
                    curr_close = data["close"].iloc[-1]
                    
                    # 计算包含程度
                    prev_body = abs(prev_close - prev_open)
                    curr_body = abs(curr_close - curr_open)
                    body_ratio = curr_body / prev_body if prev_body > 0 else 2.0
                    
                    # 计算包含覆盖度
                    prev_range = prev_high - prev_low
                    curr_range = curr_high - curr_low
                    range_expansion = curr_range / prev_range if prev_range > 0 else 1.5
                    
                    # 包含质量评分
                    quality_score = 0.5  # 基础分
                    
                    # 实体大小比例加分
                    if body_ratio > 2.0:
                        quality_score += 0.3
                    elif body_ratio > 1.5:
                        quality_score += 0.2
                    
                    # 范围扩展加分
                    if range_expansion > 1.3:
                        quality_score += 0.2
                    elif range_expansion > 1.1:
                        quality_score += 0.1
                    
                    # 前一天是阳线加分
                    if prev_close > prev_open:
                        quality_score += 0.1
                    
                    # 根据质量调整信号强度
                    strength = min(0.95, 0.7 + quality_score * 0.25)
                    confidence = min(0.95, 0.7 + quality_score * 0.25)
                    
                    metadata = {
                        "pattern_type": "bearish_engulfing",
                        "engulfing_type": "bearish",
                        "body_ratio": round(body_ratio, 3),
                        "range_expansion": round(range_expansion, 3),
                        "quality_score": round(quality_score, 3),
                        "reversal_strength": "high" if quality_score > 0.8 else "medium",
                        "prev_candle_type": "bullish" if prev_close > prev_open else "bearish"
                    }

            # 检查成交量确认（包含形态+放量更可靠）
            if signal_type in ["buy", "sell"] and len(data) >= 3:
                current_volume = data["volume"].iloc[-1]
                avg_volume = data["volume"].tail(5).mean()
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                
                if volume_ratio > 1.5:  # 放量确认
                    strength = min(0.98, strength + 0.1)
                    confidence = min(0.98, confidence + 0.1)
                    metadata["volume_confirmation"] = "high"
                    reason += "，成交量放大确认"
                elif volume_ratio > 1.2:  # 轻微放量
                    strength = min(0.95, strength + 0.05)
                    confidence = min(0.95, confidence + 0.05)
                    metadata["volume_confirmation"] = "medium"
                else:
                    metadata["volume_confirmation"] = "normal"

            # 分析趋势背景，增强包含形态的信号强度
            if signal_type in ["buy", "sell"] and len(data) >= 5:
                recent_closes = data["close"].tail(5)
                price_trend = recent_closes.diff().mean()
                
                # 看涨包含在下跌趋势中更有效
                if signal_type == "buy" and price_trend < -0.1:
                    strength = min(0.98, strength + 0.1)
                    confidence = min(0.98, confidence + 0.1)
                    metadata["trend_context"] = "downtrend"
                    metadata["signal_enhancement"] = "trend_reversal"
                    reason = reason.replace("看涨反转信号", "强势底部反转信号")
                
                # 看跌包含在上升趋势中更有效
                elif signal_type == "sell" and price_trend > 0.1:
                    strength = min(0.98, strength + 0.1)
                    confidence = min(0.98, confidence + 0.1)
                    metadata["trend_context"] = "uptrend"
                    metadata["signal_enhancement"] = "trend_reversal"
                    reason = reason.replace("看跌反转信号", "强势顶部反转信号")
                
                else:
                    metadata["trend_context"] = "sideways"
                    metadata["signal_enhancement"] = "normal"

            # 返回标准化信号格式
            return {
                'signal_type': signal_type,
                'strength': max(0.0, min(1.0, strength)),
                'confidence': max(0.0, min(1.0, confidence)),
                'timestamp': pd.Timestamp.now(),
                'reason': reason,
                'metadata': {
                    'indicator_name': self.name,
                    'pattern_category': 'reversal',
                    'pattern_family': 'candlestick',
                    'pattern_characteristic': 'strong_reversal',
                    'requires_confirmation': 'two_candles',
                    **metadata
                }
            }

        except Exception as e:
            logger.error(f"包含形态信号生成失败: {e}")
            return self._get_default_signal(f"信号生成失败: {str(e)}")

    def _validate_signal_data(self, data: pd.DataFrame) -> bool:
        """验证输入数据"""
        if not isinstance(data, pd.DataFrame):
            return False
        
        if data.empty:
            return False
        
        # 检查必需列
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            return False
        
        # 检查数据量（包含形态至少需要2个数据点）
        if len(data) < 2:
            return False
        
        return True

    def _get_default_signal(self, reason: str = "默认持有") -> Dict[str, Any]:
        """获取默认信号"""
        return {
            'signal_type': 'hold',
            'strength': 0.0,
            'confidence': 0.5,
            'timestamp': pd.Timestamp.now(),
            'reason': reason,
            'metadata': {
                'indicator_name': self.name,
                'pattern_category': 'reversal',
                'pattern_family': 'candlestick',
                'pattern_characteristic': 'strong_reversal',
                'requires_confirmation': 'two_candles'
            }
        }


class Harami(CandlestickPatterns):
    """孕线形态识别"""

    def __init__(
        self, period: int = 20
    ):  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        super().__init__(period=period)  # 正确传递period参数
        self.name = "HARAMI"
        self.pattern_type = PatterntypePatterns.HARAMI_BULLISH  # 修复：使用正确的枚举名称

    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        获取孕线形态交易信号
        
        孕线形态（Harami）是一种温和的反转信号，特征：
        - 看涨孕线：大阴线后跟小阳线，小阳线实体完全在大阴线实体内
        - 看跌孕线：大阳线后跟小阴线，小阴线实体完全在大阳线实体内
        - 子线（第二根K线）的实体要明显小于母线（第一根K线）
        - 表示市场犹豫，但反转力度比包含形态温和

        Args:
            data: 包含OHLCV数据的DataFrame

        Returns:
            Dict[str, Any]: 标准化交易信号格式
        """
        try:
            # 数据验证
            if not self._validate_signal_data(data):
                return self._get_default_signal("数据验证失败")

            # 计算孕线形态（如果数据不是计算结果）
            if 'harami_bullish' not in data.columns:
                result_data = self.calculate(data)
            else:
                result_data = data

            # 获取最新孕线形态信号
            latest_bullish = result_data["harami_bullish"].iloc[-1] if "harami_bullish" in result_data.columns else False

            # 初始化信号参数
            signal_type = "hold"
            strength = 0.5
            confidence = 0.6
            reason = "未检测到孕线形态"
            metadata = {}

            # 处理看涨孕线形态
            if latest_bullish:
                signal_type = "buy"
                strength = 0.7  # 孕线比包含形态温和
                confidence = 0.7
                reason = "检测到看涨孕线形态，温和看涨反转信号"
                
                # 计算孕线形态的具体特征
                if len(data) >= 2:
                    # 母线（前一天，第一根K线）
                    mother_open = data["open"].iloc[-2]
                    mother_high = data["high"].iloc[-2]
                    mother_low = data["low"].iloc[-2]
                    mother_close = data["close"].iloc[-2]
                    
                    # 子线（当前天，第二根K线）
                    child_open = data["open"].iloc[-1]
                    child_high = data["high"].iloc[-1]
                    child_low = data["low"].iloc[-1]
                    child_close = data["close"].iloc[-1]
                    
                    # 计算实体大小
                    mother_body = abs(mother_close - mother_open)
                    child_body = abs(child_close - child_open)
                    body_ratio = child_body / mother_body if mother_body > 0 else 0.5
                    
                    # 计算包含程度（子线实体在母线实体内的程度）
                    mother_body_high = max(mother_open, mother_close)
                    mother_body_low = min(mother_open, mother_close)
                    child_body_high = max(child_open, child_close)
                    child_body_low = min(child_open, child_close)
                    
                    # 检查子线是否完全在母线实体内
                    fully_contained = (child_body_high <= mother_body_high and 
                                     child_body_low >= mother_body_low)
                    
                    # 计算包含度（子线在母线实体中的位置）
                    if mother_body > 0:
                        containment_ratio = 1.0 if fully_contained else 0.5
                    else:
                        containment_ratio = 0.5
                    
                    # 孕线质量评分
                    quality_score = 0.5  # 基础分
                    
                    # 实体大小比例加分（子线越小越好）
                    if body_ratio < 0.3:
                        quality_score += 0.3
                    elif body_ratio < 0.5:
                        quality_score += 0.2
                    elif body_ratio < 0.7:
                        quality_score += 0.1
                    
                    # 完全包含加分
                    if fully_contained:
                        quality_score += 0.2
                    
                    # 母线是阴线加分（看涨孕线）
                    if mother_close < mother_open:
                        quality_score += 0.1
                    
                    # 子线是阳线加分（看涨孕线）
                    if child_close > child_open:
                        quality_score += 0.1
                    
                    # 根据质量调整信号强度
                    strength = min(0.9, 0.6 + quality_score * 0.3)
                    confidence = min(0.9, 0.6 + quality_score * 0.3)
                    
                    metadata = {
                        "pattern_type": "bullish_harami",
                        "harami_type": "bullish",
                        "body_ratio": round(body_ratio, 3),
                        "containment_ratio": round(containment_ratio, 3),
                        "fully_contained": fully_contained,
                        "quality_score": round(quality_score, 3),
                        "reversal_strength": "high" if quality_score > 0.8 else "medium",
                        "mother_candle_type": "bearish" if mother_close < mother_open else "bullish",
                        "child_candle_type": "bullish" if child_close > child_open else "bearish"
                    }

            # 检查是否有看跌孕线（如果数据中有对应列）
            if not latest_bullish and "harami_bearish" in result_data.columns:
                latest_bearish = result_data["harami_bearish"].iloc[-1]
                
                if latest_bearish:
                    signal_type = "sell"
                    strength = 0.7
                    confidence = 0.7
                    reason = "检测到看跌孕线形态，温和看跌反转信号"
                    
                    # 计算看跌孕线特征
                    if len(data) >= 2:
                        # 母线（前一天，第一根K线）
                        mother_open = data["open"].iloc[-2]
                        mother_close = data["close"].iloc[-2]
                        
                        # 子线（当前天，第二根K线）
                        child_open = data["open"].iloc[-1]
                        child_close = data["close"].iloc[-1]
                        
                        # 计算实体大小
                        mother_body = abs(mother_close - mother_open)
                        child_body = abs(child_close - child_open)
                        body_ratio = child_body / mother_body if mother_body > 0 else 0.5
                        
                        # 类似的质量评分逻辑
                        quality_score = 0.5
                        
                        if body_ratio < 0.3:
                            quality_score += 0.3
                        elif body_ratio < 0.5:
                            quality_score += 0.2
                        
                        # 母线是阳线加分（看跌孕线）
                        if mother_close > mother_open:
                            quality_score += 0.1
                        
                        # 子线是阴线加分（看跌孕线）
                        if child_close < child_open:
                            quality_score += 0.1
                        
                        strength = min(0.9, 0.6 + quality_score * 0.3)
                        confidence = min(0.9, 0.6 + quality_score * 0.3)
                        
                        metadata = {
                            "pattern_type": "bearish_harami",
                            "harami_type": "bearish",
                            "body_ratio": round(body_ratio, 3),
                            "quality_score": round(quality_score, 3),
                            "reversal_strength": "high" if quality_score > 0.8 else "medium",
                            "mother_candle_type": "bullish" if mother_close > mother_open else "bearish",
                            "child_candle_type": "bearish" if child_close < child_open else "bullish"
                        }

            # 检查成交量确认（孕线形态+缩量更符合特征）
            if signal_type in ["buy", "sell"] and len(data) >= 3:
                current_volume = data["volume"].iloc[-1]
                prev_volume = data["volume"].iloc[-2]
                avg_volume = data["volume"].tail(5).mean()
                
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                volume_shrinkage = prev_volume / current_volume if current_volume > 0 else 1.0
                
                # 孕线形态通常伴随缩量
                if volume_ratio < 0.8:  # 缩量确认
                    strength = min(0.95, strength + 0.1)
                    confidence = min(0.95, confidence + 0.1)
                    metadata["volume_confirmation"] = "shrinkage"
                    reason += "，成交量萎缩确认"
                elif volume_ratio < 1.0:  # 轻微缩量
                    strength = min(0.9, strength + 0.05)
                    confidence = min(0.9, confidence + 0.05)
                    metadata["volume_confirmation"] = "slight_shrinkage"
                else:
                    metadata["volume_confirmation"] = "normal"
                
                # 添加量能对比
                metadata["volume_ratio"] = round(volume_ratio, 3)
                if volume_shrinkage > 1:
                    metadata["volume_shrinkage"] = round(volume_shrinkage, 3)

            # 分析趋势背景，增强孕线形态的信号强度
            if signal_type in ["buy", "sell"] and len(data) >= 5:
                recent_closes = data["close"].tail(5)
                price_trend = recent_closes.diff().mean()
                
                # 看涨孕线在下跌趋势中更有效
                if signal_type == "buy" and price_trend < -0.1:
                    strength = min(0.95, strength + 0.1)
                    confidence = min(0.95, confidence + 0.1)
                    metadata["trend_context"] = "downtrend"
                    metadata["signal_enhancement"] = "trend_reversal"
                    reason = reason.replace("温和看涨反转信号", "趋势反转看涨信号")
                
                # 看跌孕线在上升趋势中更有效
                elif signal_type == "sell" and price_trend > 0.1:
                    strength = min(0.95, strength + 0.1)
                    confidence = min(0.95, confidence + 0.1)
                    metadata["trend_context"] = "uptrend"
                    metadata["signal_enhancement"] = "trend_reversal"
                    reason = reason.replace("温和看跌反转信号", "趋势反转看跌信号")
                
                else:
                    metadata["trend_context"] = "sideways"
                    metadata["signal_enhancement"] = "normal"

            # 返回标准化信号格式
            return {
                'signal_type': signal_type,
                'strength': max(0.0, min(1.0, strength)),
                'confidence': max(0.0, min(1.0, confidence)),
                'timestamp': pd.Timestamp.now(),
                'reason': reason,
                'metadata': {
                    'indicator_name': self.name,
                    'pattern_category': 'reversal',
                    'pattern_family': 'candlestick',
                    'pattern_characteristic': 'mild_reversal',
                    'requires_confirmation': 'two_candles',
                    'strength_comparison': 'milder_than_engulfing',
                    **metadata
                }
            }

        except Exception as e:
            logger.error(f"孕线形态信号生成失败: {e}")
            return self._get_default_signal(f"信号生成失败: {str(e)}")

    def _validate_signal_data(self, data: pd.DataFrame) -> bool:
        """验证输入数据"""
        if not isinstance(data, pd.DataFrame):
            return False
        
        if data.empty:
            return False
        
        # 检查必需列
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            return False
        
        # 检查数据量（孕线形态至少需要2个数据点）
        if len(data) < 2:
            return False
        
        return True

    def _get_default_signal(self, reason: str = "默认持有") -> Dict[str, Any]:
        """获取默认信号"""
        return {
            'signal_type': 'hold',
            'strength': 0.0,
            'confidence': 0.5,
            'timestamp': pd.Timestamp.now(),
            'reason': reason,
            'metadata': {
                'indicator_name': self.name,
                'pattern_category': 'reversal',
                'pattern_family': 'candlestick',
                'pattern_characteristic': 'mild_reversal',
                'requires_confirmation': 'two_candles',
                'strength_comparison': 'milder_than_engulfing'
            }
        }


class PiercingLine(CandlestickPatterns):
    """刺透线形态识别"""

    def __init__(
        self, period: int = 20
    ):  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        super().__init__(period=period)  # 正确传递period参数
        self.name = "PIERCING_LINE"
        self.pattern_type = PatterntypePatterns.PIERCING_LINE

    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        获取刺透线形态交易信号
        
        刺透线形态是一种强烈的看涨反转信号，特征：
        - 第一天：阴线
        - 第二天：阳线，开盘价低于前一天最低价
        - 第二天阳线收盘价必须刺透前一天阴线实体的一半以上
        - 刺透程度越深，信号越强
        - 通常出现在下跌趋势中，表示强烈的底部反转

        Args:
            data: 包含OHLCV数据的DataFrame

        Returns:
            Dict[str, Any]: 标准化交易信号格式
        """
        try:
            # 数据验证
            if not self._validate_signal_data(data):
                return self._get_default_signal("数据验证失败")

            # 计算刺透线形态（如果数据不是计算结果）
            if 'piercing_line' not in data.columns:
                result_data = self.calculate(data)
            else:
                result_data = data

            # 获取最新刺透线形态信号
            latest_piercing = result_data["piercing_line"].iloc[-1] if "piercing_line" in result_data.columns else False

            # 初始化信号参数
            signal_type = "hold"
            strength = 0.5
            confidence = 0.6
            reason = "未检测到刺透线形态"
            metadata = {}

            # 处理刺透线形态
            if latest_piercing:
                signal_type = "buy"
                strength = 0.8  # 刺透线是强反转信号
                confidence = 0.8
                reason = "检测到刺透线形态，强烈看涨反转信号"
                
                # 计算刺透线形态的具体特征
                if len(data) >= 2:
                    # 第一天（阴线）
                    first_open = data["open"].iloc[-2]
                    first_high = data["high"].iloc[-2]
                    first_low = data["low"].iloc[-2]
                    first_close = data["close"].iloc[-2]
                    
                    # 第二天（阳线）
                    second_open = data["open"].iloc[-1]
                    second_high = data["high"].iloc[-1]
                    second_low = data["low"].iloc[-1]
                    second_close = data["close"].iloc[-1]
                    
                    # 计算刺透程度
                    first_body = abs(first_close - first_open)
                    second_body = abs(second_close - second_open)
                    
                    # 检查刺透程度（阳线收盘价刺透阴线实体的比例）
                    if first_close < first_open and second_close > second_open:  # 确认阴线+阳线
                        # 计算刺透比例
                        first_body_range = first_open - first_close  # 阴线实体范围
                        piercing_depth = second_close - first_close  # 刺透深度
                        
                        if first_body_range > 0:
                            piercing_ratio = piercing_depth / first_body_range
                        else:
                            piercing_ratio = 0.5
                        
                        # 计算实体大小比例
                        body_ratio = second_body / first_body if first_body > 0 else 1.0
                        
                        # 刺透线质量评分
                        quality_score = 0.5  # 基础分
                        
                        # 刺透深度加分（刺透越深越好）
                        if piercing_ratio > 0.7:
                            quality_score += 0.3
                        elif piercing_ratio > 0.5:
                            quality_score += 0.2
                        elif piercing_ratio > 0.3:
                            quality_score += 0.1
                        
                        # 实体大小比例加分
                        if body_ratio > 1.2:
                            quality_score += 0.2
                        elif body_ratio > 1.0:
                            quality_score += 0.1
                        
                        # 开盘缺口加分（第二天开盘价低于第一天最低价）
                        if second_open < first_low:
                            quality_score += 0.2
                        elif second_open < first_close:
                            quality_score += 0.1
                        
                        # 根据质量调整信号强度
                        strength = min(0.95, 0.7 + quality_score * 0.25)
                        confidence = min(0.95, 0.7 + quality_score * 0.25)
                        
                        metadata = {
                            "pattern_type": "piercing_line",
                            "piercing_ratio": round(piercing_ratio, 3),
                            "body_ratio": round(body_ratio, 3),
                            "quality_score": round(quality_score, 3),
                            "reversal_strength": "high" if quality_score > 0.8 else "medium",
                            "first_candle_type": "bearish",
                            "second_candle_type": "bullish",
                            "gap_down": second_open < first_low,
                            "piercing_grade": "deep" if piercing_ratio > 0.7 else "standard" if piercing_ratio > 0.5 else "shallow"
                        }

            # 检查成交量确认（刺透线形态+放量更可靠）
            if signal_type == "buy" and len(data) >= 3:
                current_volume = data["volume"].iloc[-1]
                avg_volume = data["volume"].tail(5).mean()
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                
                if volume_ratio > 1.5:  # 放量确认
                    strength = min(0.98, strength + 0.1)
                    confidence = min(0.98, confidence + 0.1)
                    metadata["volume_confirmation"] = "high"
                    reason += "，成交量放大确认"
                elif volume_ratio > 1.2:  # 轻微放量
                    strength = min(0.95, strength + 0.05)
                    confidence = min(0.95, confidence + 0.05)
                    metadata["volume_confirmation"] = "medium"
                else:
                    metadata["volume_confirmation"] = "normal"
                
                metadata["volume_ratio"] = round(volume_ratio, 3)

            # 分析趋势背景，增强刺透线形态的信号强度
            if signal_type == "buy" and len(data) >= 5:
                recent_closes = data["close"].tail(5)
                price_trend = recent_closes.diff().mean()
                
                # 刺透线在下跌趋势中更有效
                if price_trend < -0.1:
                    strength = min(0.98, strength + 0.1)
                    confidence = min(0.98, confidence + 0.1)
                    metadata["trend_context"] = "downtrend"
                    metadata["signal_enhancement"] = "trend_reversal"
                    reason = reason.replace("强烈看涨反转信号", "强势底部反转信号")
                
                elif price_trend < 0:
                    strength = min(0.95, strength + 0.05)
                    confidence = min(0.95, confidence + 0.05)
                    metadata["trend_context"] = "mild_downtrend"
                    metadata["signal_enhancement"] = "trend_reversal"
                
                else:
                    metadata["trend_context"] = "sideways_or_uptrend"
                    metadata["signal_enhancement"] = "normal"

            # 检查前期支撑位确认
            if signal_type == "buy" and len(data) >= 10:
                # 检查是否在重要支撑位附近
                recent_lows = data["low"].tail(10)
                current_low = data["low"].iloc[-1]
                support_levels = recent_lows[recent_lows <= current_low * 1.02]  # 2%容忍度
                
                if len(support_levels) >= 2:  # 多次测试的支撑位
                    strength = min(0.98, strength + 0.05)
                    confidence = min(0.98, confidence + 0.05)
                    metadata["support_confirmation"] = "strong"
                    reason += "，重要支撑位确认"
                elif len(support_levels) >= 1:
                    metadata["support_confirmation"] = "moderate"
                else:
                    metadata["support_confirmation"] = "none"

            # 返回标准化信号格式
            return {
                'signal_type': signal_type,
                'strength': max(0.0, min(1.0, strength)),
                'confidence': max(0.0, min(1.0, confidence)),
                'timestamp': pd.Timestamp.now(),
                'reason': reason,
                'metadata': {
                    'indicator_name': self.name,
                    'pattern_category': 'reversal',
                    'pattern_family': 'candlestick',
                    'pattern_characteristic': 'strong_bullish_reversal',
                    'requires_confirmation': 'two_candles',
                    'signal_direction': 'bullish_only',
                    **metadata
                }
            }

        except Exception as e:
            logger.error(f"刺透线形态信号生成失败: {e}")
            return self._get_default_signal(f"信号生成失败: {str(e)}")

    def _validate_signal_data(self, data: pd.DataFrame) -> bool:
        """验证输入数据"""
        if not isinstance(data, pd.DataFrame):
            return False
        
        if data.empty:
            return False
        
        # 检查必需列
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            return False
        
        # 检查数据量（刺透线形态至少需要2个数据点）
        if len(data) < 2:
            return False
        
        return True

    def _get_default_signal(self, reason: str = "默认持有") -> Dict[str, Any]:
        """获取默认信号"""
        return {
            'signal_type': 'hold',
            'strength': 0.0,
            'confidence': 0.5,
            'timestamp': pd.Timestamp.now(),
            'reason': reason,
            'metadata': {
                'indicator_name': self.name,
                'pattern_category': 'reversal',
                'pattern_family': 'candlestick',
                'pattern_characteristic': 'strong_bullish_reversal',
                'requires_confirmation': 'two_candles',
                'signal_direction': 'bullish_only'
            }
        }


class DarkCloudCover(CandlestickPatterns):
    """乌云盖顶形态识别"""

    def __init__(
        self, period: int = 20
    ):  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        super().__init__(period=period)  # 正确传递period参数
        self.name = "DARK_CLOUD_COVER"
        self.pattern_type = PatterntypePatterns.DARK_CLOUD_COVER

    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        获取乌云盖顶形态交易信号
        
        乌云盖顶形态是一种强烈的看跌反转信号，特征：
        - 第一天：阳线
        - 第二天：阴线，开盘价高于前一天最高价
        - 第二天阴线收盘价必须覆盖前一天阳线实体的一半以上
        - 覆盖程度越深，信号越强
        - 通常出现在上升趋势中，表示强烈的顶部反转
        - 与刺透线形态相对应，是其看跌版本

        Args:
            data: 包含OHLCV数据的DataFrame

        Returns:
            Dict[str, Any]: 标准化交易信号格式
        """
        try:
            # 数据验证
            if not self._validate_signal_data(data):
                return self._get_default_signal("数据验证失败")

            # 计算乌云盖顶形态（如果数据不是计算结果）
            if 'dark_cloud_cover' not in data.columns:
                result_data = self.calculate(data)
            else:
                result_data = data

            # 获取最新乌云盖顶形态信号
            latest_dark_cloud = result_data["dark_cloud_cover"].iloc[-1] if "dark_cloud_cover" in result_data.columns else False

            # 初始化信号参数
            signal_type = "hold"
            strength = 0.5
            confidence = 0.6
            reason = "未检测到乌云盖顶形态"
            metadata = {}

            # 处理乌云盖顶形态
            if latest_dark_cloud:
                signal_type = "sell"
                strength = 0.8  # 乌云盖顶是强反转信号
                confidence = 0.8
                reason = "检测到乌云盖顶形态，强烈看跌反转信号"
                
                # 计算乌云盖顶形态的具体特征
                if len(data) >= 2:
                    # 第一天（阳线）
                    first_open = data["open"].iloc[-2]
                    first_high = data["high"].iloc[-2]
                    first_low = data["low"].iloc[-2]
                    first_close = data["close"].iloc[-2]
                    
                    # 第二天（阴线）
                    second_open = data["open"].iloc[-1]
                    second_high = data["high"].iloc[-1]
                    second_low = data["low"].iloc[-1]
                    second_close = data["close"].iloc[-1]
                    
                    # 计算覆盖程度
                    first_body = abs(first_close - first_open)
                    second_body = abs(second_close - second_open)
                    
                    # 检查覆盖程度（阴线收盘价覆盖阳线实体的比例）
                    if first_close > first_open and second_close < second_open:  # 确认阳线+阴线
                        # 计算覆盖比例
                        first_body_range = first_close - first_open  # 阳线实体范围
                        covering_depth = first_close - second_close  # 覆盖深度
                        
                        if first_body_range > 0:
                            covering_ratio = covering_depth / first_body_range
                        else:
                            covering_ratio = 0.5
                        
                        # 计算实体大小比例
                        body_ratio = second_body / first_body if first_body > 0 else 1.0
                        
                        # 乌云盖顶质量评分
                        quality_score = 0.5  # 基础分
                        
                        # 覆盖深度加分（覆盖越深越好）
                        if covering_ratio > 0.7:
                            quality_score += 0.3
                        elif covering_ratio > 0.5:
                            quality_score += 0.2
                        elif covering_ratio > 0.3:
                            quality_score += 0.1
                        
                        # 实体大小比例加分
                        if body_ratio > 1.2:
                            quality_score += 0.2
                        elif body_ratio > 1.0:
                            quality_score += 0.1
                        
                        # 开盘缺口加分（第二天开盘价高于第一天最高价）
                        if second_open > first_high:
                            quality_score += 0.2
                        elif second_open > first_close:
                            quality_score += 0.1
                        
                        # 根据质量调整信号强度
                        strength = min(0.95, 0.7 + quality_score * 0.25)
                        confidence = min(0.95, 0.7 + quality_score * 0.25)
                        
                        metadata = {
                            "pattern_type": "dark_cloud_cover",
                            "covering_ratio": round(covering_ratio, 3),
                            "body_ratio": round(body_ratio, 3),
                            "quality_score": round(quality_score, 3),
                            "reversal_strength": "high" if quality_score > 0.8 else "medium",
                            "first_candle_type": "bullish",
                            "second_candle_type": "bearish",
                            "gap_up": second_open > first_high,
                            "covering_grade": "deep" if covering_ratio > 0.7 else "standard" if covering_ratio > 0.5 else "shallow"
                        }

            # 检查成交量确认（乌云盖顶形态+放量更可靠）
            if signal_type == "sell" and len(data) >= 3:
                current_volume = data["volume"].iloc[-1]
                avg_volume = data["volume"].tail(5).mean()
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                
                if volume_ratio > 1.5:  # 放量确认
                    strength = min(0.98, strength + 0.1)
                    confidence = min(0.98, confidence + 0.1)
                    metadata["volume_confirmation"] = "high"
                    reason += "，成交量放大确认"
                elif volume_ratio > 1.2:  # 轻微放量
                    strength = min(0.95, strength + 0.05)
                    confidence = min(0.95, confidence + 0.05)
                    metadata["volume_confirmation"] = "medium"
                else:
                    metadata["volume_confirmation"] = "normal"
                
                metadata["volume_ratio"] = round(volume_ratio, 3)

            # 分析趋势背景，增强乌云盖顶形态的信号强度
            if signal_type == "sell" and len(data) >= 5:
                recent_closes = data["close"].tail(5)
                price_trend = recent_closes.diff().mean()
                
                # 乌云盖顶在上升趋势中更有效
                if price_trend > 0.1:
                    strength = min(0.98, strength + 0.1)
                    confidence = min(0.98, confidence + 0.1)
                    metadata["trend_context"] = "uptrend"
                    metadata["signal_enhancement"] = "trend_reversal"
                    reason = reason.replace("强烈看跌反转信号", "强势顶部反转信号")
                
                elif price_trend > 0:
                    strength = min(0.95, strength + 0.05)
                    confidence = min(0.95, confidence + 0.05)
                    metadata["trend_context"] = "mild_uptrend"
                    metadata["signal_enhancement"] = "trend_reversal"
                
                else:
                    metadata["trend_context"] = "sideways_or_downtrend"
                    metadata["signal_enhancement"] = "normal"

            # 检查前期阻力位确认
            if signal_type == "sell" and len(data) >= 10:
                # 检查是否在重要阻力位附近
                recent_highs = data["high"].tail(10)
                current_high = data["high"].iloc[-1]
                resistance_levels = recent_highs[recent_highs >= current_high * 0.98]  # 2%容忍度
                
                if len(resistance_levels) >= 2:  # 多次测试的阻力位
                    strength = min(0.98, strength + 0.05)
                    confidence = min(0.98, confidence + 0.05)
                    metadata["resistance_confirmation"] = "strong"
                    reason += "，重要阻力位确认"
                elif len(resistance_levels) >= 1:
                    metadata["resistance_confirmation"] = "moderate"
                else:
                    metadata["resistance_confirmation"] = "none"

            # 返回标准化信号格式
            return {
                'signal_type': signal_type,
                'strength': max(0.0, min(1.0, strength)),
                'confidence': max(0.0, min(1.0, confidence)),
                'timestamp': pd.Timestamp.now(),
                'reason': reason,
                'metadata': {
                    'indicator_name': self.name,
                    'pattern_category': 'reversal',
                    'pattern_family': 'candlestick',
                    'pattern_characteristic': 'strong_bearish_reversal',
                    'requires_confirmation': 'two_candles',
                    'signal_direction': 'bearish_only',
                    **metadata
                }
            }

        except Exception as e:
            logger.error(f"乌云盖顶形态信号生成失败: {e}")
            return self._get_default_signal(f"信号生成失败: {str(e)}")

    def _validate_signal_data(self, data: pd.DataFrame) -> bool:
        """验证输入数据"""
        if not isinstance(data, pd.DataFrame):
            return False
        
        if data.empty:
            return False
        
        # 检查必需列
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            return False
        
        # 检查数据量（乌云盖顶形态至少需要2个数据点）
        if len(data) < 2:
            return False
        
        return True

    def _get_default_signal(self, reason: str = "默认持有") -> Dict[str, Any]:
        """获取默认信号"""
        return {
            'signal_type': 'hold',
            'strength': 0.0,
            'confidence': 0.5,
            'timestamp': pd.Timestamp.now(),
            'reason': reason,
            'metadata': {
                'indicator_name': self.name,
                'pattern_category': 'reversal',
                'pattern_family': 'candlestick',
                'pattern_characteristic': 'strong_bearish_reversal',
                'requires_confirmation': 'two_candles',
                'signal_direction': 'bearish_only'
            }
        }


class MorningStar(CandlestickPatterns):
    """启明星形态识别"""

    def __init__(
        self, period: int = 20
    ):  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        super().__init__(period=period)  # 正确传递period参数
        self.name = "MORNING_STAR"
        self.pattern_type = PatterntypePatterns.MORNING_STAR

    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        获取启明星形态交易信号
        
        启明星形态是一种强烈的看涨反转信号，特征：
        - 第一天：阴线（下跌趋势的延续）
        - 第二天：十字星或小实体K线，向下跳空开盘（市场犹豫）
        - 第三天：阳线，向上跳空开盘，收盘价深入第一天阴线实体
        - 三根K线形成明显的"V"形反转结构
        - 第二天的跳空是关键特征，表示趋势的犹豫和可能反转
        - 第三天的向上突破确认了反转的有效性

        Args:
            data: 包含OHLCV数据的DataFrame

        Returns:
            Dict[str, Any]: 标准化交易信号格式
        """
        try:
            # 数据验证
            if not self._validate_signal_data(data):
                return self._get_default_signal("数据验证失败")

            # 计算启明星形态（如果数据不是计算结果）
            if 'morning_star' not in data.columns:
                result_data = self.calculate(data)
            else:
                result_data = data

            # 获取最新启明星形态信号
            latest_morning_star = result_data["morning_star"].iloc[-1] if "morning_star" in result_data.columns else False

            # 初始化信号参数
            signal_type = "hold"
            strength = 0.5
            confidence = 0.6
            reason = "未检测到启明星形态"
            metadata = {}

            # 处理启明星形态
            if latest_morning_star:
                signal_type = "buy"
                strength = 0.85  # 启明星是强反转信号，比双K线形态稍强
                confidence = 0.85
                reason = "检测到启明星形态，强烈看涨反转信号"
                
                # 计算启明星形态的具体特征
                if len(data) >= 3:
                    # 第一天（阴线）
                    first_open = data["open"].iloc[-3]
                    first_high = data["high"].iloc[-3]
                    first_low = data["low"].iloc[-3]
                    first_close = data["close"].iloc[-3]
                    
                    # 第二天（十字星/小实体）
                    second_open = data["open"].iloc[-2]
                    second_high = data["high"].iloc[-2]
                    second_low = data["low"].iloc[-2]
                    second_close = data["close"].iloc[-2]
                    
                    # 第三天（阳线）
                    third_open = data["open"].iloc[-1]
                    third_high = data["high"].iloc[-1]
                    third_low = data["low"].iloc[-1]
                    third_close = data["close"].iloc[-1]
                    
                    # 计算实体大小
                    first_body = abs(first_close - first_open)
                    second_body = abs(second_close - second_open)
                    third_body = abs(third_close - third_open)
                    
                    # 启明星质量评分
                    quality_score = 0.5  # 基础分
                    
                    # 检查基本形态特征
                    if first_close < first_open and third_close > third_open:  # 确认阴线+阳线
                        
                        # 跳空特征评分
                        gap_down = second_high < first_low  # 向下跳空
                        gap_up = third_low > second_high    # 向上跳空
                        
                        if gap_down and gap_up:
                            quality_score += 0.3  # 双跳空是理想形态
                        elif gap_down or gap_up:
                            quality_score += 0.15  # 单跳空也有效
                        
                        # 中间K线特征评分（十字星或小实体）
                        if first_body > 0:
                            middle_body_ratio = second_body / first_body
                        else:
                            middle_body_ratio = 0.5
                            
                        if middle_body_ratio < 0.3:  # 中间K线实体很小
                            quality_score += 0.2
                        elif middle_body_ratio < 0.5:  # 中间K线实体较小
                            quality_score += 0.1
                        
                        # 第三天阳线穿透深度评分
                        if first_body > 0:
                            penetration_ratio = (third_close - first_close) / first_body
                        else:
                            penetration_ratio = 0.5
                            
                        if penetration_ratio > 0.5:  # 深度穿透第一天实体
                            quality_score += 0.2
                        elif penetration_ratio > 0.3:  # 适度穿透
                            quality_score += 0.1
                        
                        # 实体大小平衡评分
                        if first_body > 0 and third_body > 0:
                            body_balance = min(first_body, third_body) / max(first_body, third_body)
                            if body_balance > 0.7:  # 实体大小相对平衡
                                quality_score += 0.1
                        
                        # 根据质量调整信号强度
                        strength = min(0.95, 0.75 + quality_score * 0.2)
                        confidence = min(0.95, 0.75 + quality_score * 0.2)
                        
                        metadata = {
                            "pattern_type": "morning_star",
                            "gap_down": gap_down,
                            "gap_up": gap_up,
                            "middle_body_ratio": round(middle_body_ratio, 3),
                            "penetration_ratio": round(penetration_ratio, 3),
                            "quality_score": round(quality_score, 3),
                            "reversal_strength": "high" if quality_score > 0.9 else "medium",
                            "first_candle_type": "bearish",
                            "middle_candle_type": "doji_or_small",
                            "third_candle_type": "bullish",
                            "pattern_grade": "perfect" if gap_down and gap_up else "standard"
                        }

            # 检查成交量确认（启明星形态+放量更可靠）
            if signal_type == "buy" and len(data) >= 4:
                current_volume = data["volume"].iloc[-1]
                avg_volume = data["volume"].tail(5).mean()
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                
                if volume_ratio > 1.5:  # 放量确认
                    strength = min(0.98, strength + 0.1)
                    confidence = min(0.98, confidence + 0.1)
                    metadata["volume_confirmation"] = "high"
                    reason += "，成交量放大确认"
                elif volume_ratio > 1.2:  # 轻微放量
                    strength = min(0.95, strength + 0.05)
                    confidence = min(0.95, confidence + 0.05)
                    metadata["volume_confirmation"] = "medium"
                else:
                    metadata["volume_confirmation"] = "normal"
                
                metadata["volume_ratio"] = round(volume_ratio, 3)

            # 分析趋势背景，增强启明星形态的信号强度
            if signal_type == "buy" and len(data) >= 6:
                recent_closes = data["close"].tail(6)
                price_trend = recent_closes.diff().mean()
                
                # 启明星在下跌趋势中更有效
                if price_trend < -0.1:
                    strength = min(0.98, strength + 0.1)
                    confidence = min(0.98, confidence + 0.1)
                    metadata["trend_context"] = "downtrend"
                    metadata["signal_enhancement"] = "trend_reversal"
                    reason = reason.replace("强烈看涨反转信号", "强势底部反转信号")
                
                elif price_trend < 0:
                    strength = min(0.95, strength + 0.05)
                    confidence = min(0.95, confidence + 0.05)
                    metadata["trend_context"] = "mild_downtrend"
                    metadata["signal_enhancement"] = "trend_reversal"
                
                else:
                    metadata["trend_context"] = "sideways_or_uptrend"
                    metadata["signal_enhancement"] = "normal"

            # 检查前期支撑位确认
            if signal_type == "buy" and len(data) >= 10:
                # 检查是否在重要支撑位附近
                recent_lows = data["low"].tail(10)
                current_low = data["low"].iloc[-2]  # 使用中间K线的最低价
                support_levels = recent_lows[recent_lows <= current_low * 1.02]  # 2%容忍度
                
                if len(support_levels) >= 2:  # 多次测试的支撑位
                    strength = min(0.98, strength + 0.05)
                    confidence = min(0.98, confidence + 0.05)
                    metadata["support_confirmation"] = "strong"
                    reason += "，重要支撑位确认"
                elif len(support_levels) >= 1:
                    metadata["support_confirmation"] = "moderate"
                else:
                    metadata["support_confirmation"] = "none"

            # 返回标准化信号格式
            return {
                'signal_type': signal_type,
                'strength': max(0.0, min(1.0, strength)),
                'confidence': max(0.0, min(1.0, confidence)),
                'timestamp': pd.Timestamp.now(),
                'reason': reason,
                'metadata': {
                    'indicator_name': self.name,
                    'pattern_category': 'reversal',
                    'pattern_family': 'candlestick',
                    'pattern_characteristic': 'strong_bullish_reversal',
                    'requires_confirmation': 'three_candles',
                    'signal_direction': 'bullish_only',
                    **metadata
                }
            }

        except Exception as e:
            logger.error(f"启明星形态信号生成失败: {e}")
            return self._get_default_signal(f"信号生成失败: {str(e)}")

    def _validate_signal_data(self, data: pd.DataFrame) -> bool:
        """验证输入数据"""
        if not isinstance(data, pd.DataFrame):
            return False
        
        if data.empty:
            return False
        
        # 检查必需列
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            return False
        
        # 检查数据量（启明星形态至少需要3个数据点）
        if len(data) < 3:
            return False
        
        return True

    def _get_default_signal(self, reason: str = "默认持有") -> Dict[str, Any]:
        """获取默认信号"""
        return {
            'signal_type': 'hold',
            'strength': 0.0,
            'confidence': 0.5,
            'timestamp': pd.Timestamp.now(),
            'reason': reason,
            'metadata': {
                'indicator_name': self.name,
                'pattern_category': 'reversal',
                'pattern_family': 'candlestick',
                'pattern_characteristic': 'strong_bullish_reversal',
                'requires_confirmation': 'three_candles',
                'signal_direction': 'bullish_only'
            }
        }


class EveningStar(CandlestickPatterns):
    """黄昏星形态识别"""

    def __init__(
        self, period: int = 20
    ):  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        super().__init__(period=period)  # 正确传递period参数
        self.name = "EVENING_STAR"
        self.pattern_type = PatterntypePatterns.EVENING_STAR

    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        获取黄昏星形态交易信号
        
        黄昏星形态是一种强烈的看跌反转信号，特征：
        - 第一天：阳线（上涨趋势的延续）
        - 第二天：十字星或小实体K线，向上跳空开盘（市场犹豫）
        - 第三天：阴线，向下跳空开盘，收盘价深入第一天阳线实体
        - 三根K线形成明显的"倒V"形反转结构
        - 第二天的跳空是关键特征，表示趋势的犹豫和可能反转
        - 第三天的向下突破确认了反转的有效性
        - 与启明星形态形成完美的多空对称体系

        Args:
            data: 包含OHLCV数据的DataFrame

        Returns:
            Dict[str, Any]: 标准化交易信号格式
        """
        try:
            # 数据验证
            if not self._validate_signal_data(data):
                return self._get_default_signal("数据验证失败")

            # 计算黄昏星形态（如果数据不是计算结果）
            if 'evening_star' not in data.columns:
                result_data = self.calculate(data)
            else:
                result_data = data

            # 获取最新黄昏星形态信号
            latest_evening_star = result_data["evening_star"].iloc[-1] if "evening_star" in result_data.columns else False

            # 初始化信号参数
            signal_type = "hold"
            strength = 0.5
            confidence = 0.6
            reason = "未检测到黄昏星形态"
            metadata = {}

            # 处理黄昏星形态
            if latest_evening_star:
                signal_type = "sell"
                strength = 0.85  # 黄昏星是强反转信号，与启明星相同强度
                confidence = 0.85
                reason = "检测到黄昏星形态，强烈看跌反转信号"
                
                # 计算黄昏星形态的具体特征
                if len(data) >= 3:
                    # 第一天（阳线）
                    first_open = data["open"].iloc[-3]
                    first_high = data["high"].iloc[-3]
                    first_low = data["low"].iloc[-3]
                    first_close = data["close"].iloc[-3]
                    
                    # 第二天（十字星/小实体）
                    second_open = data["open"].iloc[-2]
                    second_high = data["high"].iloc[-2]
                    second_low = data["low"].iloc[-2]
                    second_close = data["close"].iloc[-2]
                    
                    # 第三天（阴线）
                    third_open = data["open"].iloc[-1]
                    third_high = data["high"].iloc[-1]
                    third_low = data["low"].iloc[-1]
                    third_close = data["close"].iloc[-1]
                    
                    # 计算实体大小
                    first_body = abs(first_close - first_open)
                    second_body = abs(second_close - second_open)
                    third_body = abs(third_close - third_open)
                    
                    # 黄昏星质量评分
                    quality_score = 0.5  # 基础分
                    
                    # 检查基本形态特征
                    if first_close > first_open and third_close < third_open:  # 确认阳线+阴线
                        
                        # 跳空特征评分
                        gap_up = second_low > first_high    # 向上跳空
                        gap_down = third_high < second_low  # 向下跳空
                        
                        if gap_up and gap_down:
                            quality_score += 0.3  # 双跳空是理想形态
                        elif gap_up or gap_down:
                            quality_score += 0.15  # 单跳空也有效
                        
                        # 中间K线特征评分（十字星或小实体）
                        if first_body > 0:
                            middle_body_ratio = second_body / first_body
                        else:
                            middle_body_ratio = 0.5
                            
                        if middle_body_ratio < 0.3:  # 中间K线实体很小
                            quality_score += 0.2
                        elif middle_body_ratio < 0.5:  # 中间K线实体较小
                            quality_score += 0.1
                        
                        # 第三天阴线穿透深度评分
                        if first_body > 0:
                            penetration_ratio = (first_close - third_close) / first_body
                        else:
                            penetration_ratio = 0.5
                            
                        if penetration_ratio > 0.5:  # 深度穿透第一天实体
                            quality_score += 0.2
                        elif penetration_ratio > 0.3:  # 适度穿透
                            quality_score += 0.1
                        
                        # 实体大小平衡评分
                        if first_body > 0 and third_body > 0:
                            body_balance = min(first_body, third_body) / max(first_body, third_body)
                            if body_balance > 0.7:  # 实体大小相对平衡
                                quality_score += 0.1
                        
                        # 根据质量调整信号强度
                        strength = min(0.95, 0.75 + quality_score * 0.2)
                        confidence = min(0.95, 0.75 + quality_score * 0.2)
                        
                        metadata = {
                            "pattern_type": "evening_star",
                            "gap_up": gap_up,
                            "gap_down": gap_down,
                            "middle_body_ratio": round(middle_body_ratio, 3),
                            "penetration_ratio": round(penetration_ratio, 3),
                            "quality_score": round(quality_score, 3),
                            "reversal_strength": "high" if quality_score > 0.9 else "medium",
                            "first_candle_type": "bullish",
                            "middle_candle_type": "doji_or_small",
                            "third_candle_type": "bearish",
                            "pattern_grade": "perfect" if gap_up and gap_down else "standard"
                        }

            # 检查成交量确认（黄昏星形态+放量更可靠）
            if signal_type == "sell" and len(data) >= 4:
                current_volume = data["volume"].iloc[-1]
                avg_volume = data["volume"].tail(5).mean()
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                
                if volume_ratio > 1.5:  # 放量确认
                    strength = min(0.98, strength + 0.1)
                    confidence = min(0.98, confidence + 0.1)
                    metadata["volume_confirmation"] = "high"
                    reason += "，成交量放大确认"
                elif volume_ratio > 1.2:  # 轻微放量
                    strength = min(0.95, strength + 0.05)
                    confidence = min(0.95, confidence + 0.05)
                    metadata["volume_confirmation"] = "medium"
                else:
                    metadata["volume_confirmation"] = "normal"
                
                metadata["volume_ratio"] = round(volume_ratio, 3)

            # 分析趋势背景，增强黄昏星形态的信号强度
            if signal_type == "sell" and len(data) >= 6:
                recent_closes = data["close"].tail(6)
                price_trend = recent_closes.diff().mean()
                
                # 黄昏星在上涨趋势中更有效
                if price_trend > 0.1:
                    strength = min(0.98, strength + 0.1)
                    confidence = min(0.98, confidence + 0.1)
                    metadata["trend_context"] = "uptrend"
                    metadata["signal_enhancement"] = "trend_reversal"
                    reason = reason.replace("强烈看跌反转信号", "强势顶部反转信号")
                
                elif price_trend > 0:
                    strength = min(0.95, strength + 0.05)
                    confidence = min(0.95, confidence + 0.05)
                    metadata["trend_context"] = "mild_uptrend"
                    metadata["signal_enhancement"] = "trend_reversal"
                
                else:
                    metadata["trend_context"] = "sideways_or_downtrend"
                    metadata["signal_enhancement"] = "normal"

            # 检查前期阻力位确认
            if signal_type == "sell" and len(data) >= 10:
                # 检查是否在重要阻力位附近
                recent_highs = data["high"].tail(10)
                current_high = data["high"].iloc[-2]  # 使用中间K线的最高价
                resistance_levels = recent_highs[recent_highs >= current_high * 0.98]  # 2%容忍度
                
                if len(resistance_levels) >= 2:  # 多次测试的阻力位
                    strength = min(0.98, strength + 0.05)
                    confidence = min(0.98, confidence + 0.05)
                    metadata["resistance_confirmation"] = "strong"
                    reason += "，重要阻力位确认"
                elif len(resistance_levels) >= 1:
                    metadata["resistance_confirmation"] = "moderate"
                else:
                    metadata["resistance_confirmation"] = "none"

            # 返回标准化信号格式
            return {
                'signal_type': signal_type,
                'strength': max(0.0, min(1.0, strength)),
                'confidence': max(0.0, min(1.0, confidence)),
                'timestamp': pd.Timestamp.now(),
                'reason': reason,
                'metadata': {
                    'indicator_name': self.name,
                    'pattern_category': 'reversal',
                    'pattern_family': 'candlestick',
                    'pattern_characteristic': 'strong_bearish_reversal',
                    'requires_confirmation': 'three_candles',
                    'signal_direction': 'bearish_only',
                    **metadata
                }
            }

        except Exception as e:
            logger.error(f"黄昏星形态信号生成失败: {e}")
            return self._get_default_signal(f"信号生成失败: {str(e)}")

    def _validate_signal_data(self, data: pd.DataFrame) -> bool:
        """验证输入数据"""
        if not isinstance(data, pd.DataFrame):
            return False
        
        if data.empty:
            return False
        
        # 检查必需列
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            return False
        
        # 检查数据量（黄昏星形态至少需要3个数据点）
        if len(data) < 3:
            return False
        
        return True

    def _get_default_signal(self, reason: str = "默认持有") -> Dict[str, Any]:
        """获取默认信号"""
        return {
            'signal_type': 'hold',
            'strength': 0.0,
            'confidence': 0.5,
            'timestamp': pd.Timestamp.now(),
            'reason': reason,
            'metadata': {
                'indicator_name': self.name,
                'pattern_category': 'reversal',
                'pattern_family': 'candlestick',
                'pattern_characteristic': 'strong_bearish_reversal',
                'requires_confirmation': 'three_candles',
                'signal_direction': 'bearish_only'
            }
        }


class ThreeBlackCrows(CandlestickPatterns):
    """三只乌鸦形态识别"""

    def __init__(
        self, period: int = 20
    ):  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        super().__init__(period=period)  # 正确传递period参数
        self.name = "THREE_BLACK_CROWS"
        self.pattern_type = PatterntypePatterns.THREE_BLACK_CROWS

    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        获取三只乌鸦形态交易信号
        
        三只乌鸦形态是一种极强的看跌持续信号，特征：
        - 三根连续的长阴线，每一根都创新低
        - 每根阴线的开盘价都在前一根阴线的实体内
        - 每根阴线的收盘价都低于前一根阴线的收盘价
        - 实体较大，影线较短，显示强烈的卖压
        - 成交量通常逐步放大，确认卖出压力
        - 通常出现在上涨趋势的顶部或下跌趋势的延续中
        - 比黄昏星更强的看跌信号，预示持续下跌

        Args:
            data: 包含OHLCV数据的DataFrame

        Returns:
            Dict[str, Any]: 标准化交易信号格式
        """
        try:
            # 数据验证
            if not self._validate_signal_data(data):
                return self._get_default_signal("数据验证失败")

            # 计算三只乌鸦形态（如果数据不是计算结果）
            if 'three_black_crows' not in data.columns:
                result_data = self.calculate(data)
            else:
                result_data = data

            # 查找三只乌鸦相关列（可能有不同的命名）
            crow_column = None
            possible_columns = ['three_black_crows', 'black_crows', 'three_crows']
            for col in possible_columns:
                if col in result_data.columns:
                    crow_column = col
                    break

            # 获取最新三只乌鸦形态信号
            latest_three_crows = False
            if crow_column:
                latest_three_crows = result_data[crow_column].iloc[-1]

            # 初始化信号参数
            signal_type = "hold"
            strength = 0.5
            confidence = 0.6
            reason = "未检测到三只乌鸦形态"
            metadata = {}

            # 处理三只乌鸦形态
            if latest_three_crows:
                signal_type = "sell"
                strength = 0.90  # 三只乌鸦是极强看跌信号，比黄昏星更强
                confidence = 0.90
                reason = "检测到三只乌鸦形态，极强看跌持续信号"
                
                # 计算三只乌鸦形态的具体特征
                if len(data) >= 3:
                    # 第一只乌鸦（第一根阴线）
                    first_open = data["open"].iloc[-3]
                    first_high = data["high"].iloc[-3]
                    first_low = data["low"].iloc[-3]
                    first_close = data["close"].iloc[-3]
                    
                    # 第二只乌鸦（第二根阴线）
                    second_open = data["open"].iloc[-2]
                    second_high = data["high"].iloc[-2]
                    second_low = data["low"].iloc[-2]
                    second_close = data["close"].iloc[-2]
                    
                    # 第三只乌鸦（第三根阴线）
                    third_open = data["open"].iloc[-1]
                    third_high = data["high"].iloc[-1]
                    third_low = data["low"].iloc[-1]
                    third_close = data["close"].iloc[-1]
                    
                    # 计算实体大小
                    first_body = abs(first_close - first_open)
                    second_body = abs(second_close - second_open)
                    third_body = abs(third_close - third_open)
                    
                    # 三只乌鸦质量评分
                    quality_score = 0.6  # 基础分较高，因为是强形态
                    
                    # 检查基本形态特征（三根阴线）
                    is_bearish_candles = (first_close < first_open and 
                                        second_close < second_open and 
                                        third_close < third_open)
                    
                    if is_bearish_candles:
                        
                        # 递减低点评分（每根K线创新低）
                        decreasing_lows = (second_low < first_low and third_low < second_low)
                        decreasing_closes = (second_close < first_close and third_close < second_close)
                        
                        if decreasing_lows and decreasing_closes:
                            quality_score += 0.2  # 标准递减模式
                        elif decreasing_closes:
                            quality_score += 0.1  # 至少收盘价递减
                        
                        # 开盘价位置评分（在前一根实体内开盘）
                        second_open_in_first_body = (min(first_open, first_close) <= second_open <= max(first_open, first_close))
                        third_open_in_second_body = (min(second_open, second_close) <= third_open <= max(second_open, second_close))
                        
                        if second_open_in_first_body and third_open_in_second_body:
                            quality_score += 0.15  # 理想的开盘位置
                        elif second_open_in_first_body or third_open_in_second_body:
                            quality_score += 0.08  # 部分符合
                        
                        # 实体大小评分（长阴线特征）
                        avg_body = (first_body + second_body + third_body) / 3
                        if first_body > 0 and second_body > 0 and third_body > 0:
                            body_consistency = min(first_body, second_body, third_body) / max(first_body, second_body, third_body)
                            
                            if body_consistency > 0.7:  # 实体大小相对一致
                                quality_score += 0.1
                            
                            # 检查实体是否足够大（相对于价格范围）
                            avg_price = (first_close + second_close + third_close) / 3
                            if avg_body / avg_price > 0.02:  # 实体大于平均价格的2%
                                quality_score += 0.1
                        
                        # 影线长度评分（短影线更佳）
                        first_upper_shadow = first_high - max(first_open, first_close)
                        first_lower_shadow = min(first_open, first_close) - first_low
                        second_upper_shadow = second_high - max(second_open, second_close)
                        second_lower_shadow = min(second_open, second_close) - second_low
                        third_upper_shadow = third_high - max(third_open, third_close)
                        third_lower_shadow = min(third_open, third_close) - third_low
                        
                        avg_upper_shadow = (first_upper_shadow + second_upper_shadow + third_upper_shadow) / 3
                        
                        if avg_body > 0 and avg_upper_shadow / avg_body < 0.3:  # 上影线较短
                            quality_score += 0.05
                        
                        # 根据质量调整信号强度
                        strength = min(0.98, 0.80 + quality_score * 0.15)
                        confidence = min(0.98, 0.80 + quality_score * 0.15)
                        
                        metadata = {
                            "pattern_type": "three_black_crows",
                            "decreasing_lows": decreasing_lows,
                            "decreasing_closes": decreasing_closes,
                            "second_open_in_first_body": second_open_in_first_body,
                            "third_open_in_second_body": third_open_in_second_body,
                            "body_consistency": round(body_consistency if 'body_consistency' in locals() else 0.0, 3),
                            "quality_score": round(quality_score, 3),
                            "pattern_strength": "extreme" if quality_score > 0.9 else "high",
                            "first_candle_type": "strong_bearish",
                            "second_candle_type": "strong_bearish", 
                            "third_candle_type": "strong_bearish",
                            "pattern_grade": "perfect" if quality_score > 0.9 else "standard"
                        }

            # 检查成交量确认（三只乌鸦形态+递增成交量更可靠）
            if signal_type == "sell" and len(data) >= 5:
                volume_1 = data["volume"].iloc[-3]  # 第一只乌鸦
                volume_2 = data["volume"].iloc[-2]  # 第二只乌鸦  
                volume_3 = data["volume"].iloc[-1]  # 第三只乌鸦
                avg_volume = data["volume"].tail(7).mean()  # 前7天平均
                
                # 检查成交量递增趋势
                volume_increasing = volume_2 > volume_1 and volume_3 > volume_2
                high_volume_ratio = volume_3 / avg_volume if avg_volume > 0 else 1.0
                
                if volume_increasing and high_volume_ratio > 1.5:  # 递增且放量
                    strength = min(0.98, strength + 0.08)
                    confidence = min(0.98, confidence + 0.08)
                    metadata["volume_confirmation"] = "strong_increasing"
                    reason += "，成交量递增放大确认"
                elif volume_increasing:  # 仅递增
                    strength = min(0.95, strength + 0.05)
                    confidence = min(0.95, confidence + 0.05)
                    metadata["volume_confirmation"] = "increasing"
                elif high_volume_ratio > 1.3:  # 仅放量
                    strength = min(0.95, strength + 0.03)
                    confidence = min(0.95, confidence + 0.03)
                    metadata["volume_confirmation"] = "high"
                else:
                    metadata["volume_confirmation"] = "normal"
                
                metadata["final_volume_ratio"] = round(high_volume_ratio, 3)
                metadata["volume_trend"] = "increasing" if volume_increasing else "normal"

            # 分析趋势背景，增强三只乌鸦形态的信号强度
            if signal_type == "sell" and len(data) >= 8:
                recent_closes = data["close"].tail(8)
                price_trend = recent_closes.diff().mean()
                
                # 三只乌鸦在上涨趋势顶部更有效（顶部反转）
                if price_trend > 0.1:
                    strength = min(0.98, strength + 0.08)
                    confidence = min(0.98, confidence + 0.08)
                    metadata["trend_context"] = "uptrend_reversal"
                    metadata["signal_enhancement"] = "top_reversal"
                    reason = reason.replace("极强看跌持续信号", "极强顶部反转信号")
                
                # 三只乌鸦在下跌趋势中作为持续信号也很强
                elif price_trend < -0.05:
                    strength = min(0.96, strength + 0.05)
                    confidence = min(0.96, confidence + 0.05)
                    metadata["trend_context"] = "downtrend_continuation"
                    metadata["signal_enhancement"] = "bearish_continuation"
                    reason = reason.replace("极强看跌持续信号", "极强下跌延续信号")
                
                else:
                    metadata["trend_context"] = "sideways"
                    metadata["signal_enhancement"] = "normal"

            # 检查前期阻力位确认
            if signal_type == "sell" and len(data) >= 12:
                # 检查是否在重要阻力位附近
                recent_highs = data["high"].tail(12)
                current_high_area = data["high"].iloc[-3:-1].max()  # 前两根K线的最高价区域
                resistance_levels = recent_highs[recent_highs >= current_high_area * 0.98]  # 2%容忍度
                
                if len(resistance_levels) >= 3:  # 多次测试的阻力位
                    strength = min(0.98, strength + 0.05)
                    confidence = min(0.98, confidence + 0.05)
                    metadata["resistance_confirmation"] = "strong"
                    reason += "，重要阻力位确认"
                elif len(resistance_levels) >= 2:
                    metadata["resistance_confirmation"] = "moderate"
                else:
                    metadata["resistance_confirmation"] = "none"

            # 返回标准化信号格式
            return {
                'signal_type': signal_type,
                'strength': max(0.0, min(1.0, strength)),
                'confidence': max(0.0, min(1.0, confidence)),
                'timestamp': pd.Timestamp.now(),
                'reason': reason,
                'metadata': {
                    'indicator_name': self.name,
                    'pattern_category': 'continuation',
                    'pattern_family': 'candlestick',
                    'pattern_characteristic': 'extreme_bearish_continuation',
                    'requires_confirmation': 'three_candles',
                    'signal_direction': 'bearish_only',
                    **metadata
                }
            }

        except Exception as e:
            logger.error(f"三只乌鸦形态信号生成失败: {e}")
            return self._get_default_signal(f"信号生成失败: {str(e)}")

    def _validate_signal_data(self, data: pd.DataFrame) -> bool:
        """验证输入数据"""
        if not isinstance(data, pd.DataFrame):
            return False
        
        if data.empty:
            return False
        
        # 检查必需列
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            return False
        
        # 检查数据量（三只乌鸦形态至少需要3个数据点）
        if len(data) < 3:
            return False
        
        return True

    def _get_default_signal(self, reason: str = "默认持有") -> Dict[str, Any]:
        """获取默认信号"""
        return {
            'signal_type': 'hold',
            'strength': 0.0,
            'confidence': 0.5,
            'timestamp': pd.Timestamp.now(),
            'reason': reason,
            'metadata': {
                'indicator_name': self.name,
                'pattern_category': 'continuation',
                'pattern_family': 'candlestick',
                'pattern_characteristic': 'extreme_bearish_continuation',
                'requires_confirmation': 'three_candles',
                'signal_direction': 'bearish_only'
            }
        }


class ThreeWhiteSoldiers(CandlestickPatterns):
    """三个白武士形态识别"""

    def __init__(
        self, period: int = 20
    ):  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        super().__init__(period=period)  # 正确传递period参数
        self.name = "THREE_WHITE_SOLDIERS"
        self.pattern_type = PatterntypePatterns.THREE_WHITE_SOLDIERS

    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        获取三白兵形态交易信号
        
        三白兵形态是一种极强的看涨持续信号，与三只乌鸦完全对称，特征：
        - 三根连续的长阳线，每一根都创新高
        - 每根阳线的开盘价都在前一根阳线的实体内
        - 每根阳线的收盘价都高于前一根阳线的收盘价
        - 实体较大，影线较短，显示强烈的买压
        - 成交量通常逐步放大，确认买入压力
        - 通常出现在下跌趋势的底部或上涨趋势的延续中
        - 与三只乌鸦相对，是极强的看涨信号，预示持续上涨

        Args:
            data: 包含OHLCV数据的DataFrame

        Returns:
            Dict[str, Any]: 标准化交易信号格式
        """
        try:
            # 数据验证
            if not self._validate_signal_data(data):
                return self._get_default_signal("数据验证失败")

            # 计算三白兵形态（如果数据不是计算结果）
            if 'three_white_soldiers' not in data.columns:
                result_data = self.calculate(data)
            else:
                result_data = data

            # 查找三白兵相关列（可能有不同的命名）
            soldiers_column = None
            possible_columns = ['three_white_soldiers', 'white_soldiers', 'three_soldiers']
            for col in possible_columns:
                if col in result_data.columns:
                    soldiers_column = col
                    break

            # 获取最新三白兵形态信号
            latest_three_soldiers = False
            if soldiers_column:
                latest_three_soldiers = result_data[soldiers_column].iloc[-1]

            # 初始化信号参数
            signal_type = "hold"
            strength = 0.5
            confidence = 0.6
            reason = "未检测到三白兵形态"
            metadata = {}

            # 处理三白兵形态
            if latest_three_soldiers:
                signal_type = "buy"
                strength = 0.90  # 三白兵是极强看涨信号，与三只乌鸦对称
                confidence = 0.90
                reason = "检测到三白兵形态，极强看涨持续信号"
                
                # 计算三白兵形态的具体特征
                if len(data) >= 3:
                    # 第一个白兵（第一根阳线）
                    first_open = data["open"].iloc[-3]
                    first_high = data["high"].iloc[-3]
                    first_low = data["low"].iloc[-3]
                    first_close = data["close"].iloc[-3]
                    
                    # 第二个白兵（第二根阳线）
                    second_open = data["open"].iloc[-2]
                    second_high = data["high"].iloc[-2]
                    second_low = data["low"].iloc[-2]
                    second_close = data["close"].iloc[-2]
                    
                    # 第三个白兵（第三根阳线）
                    third_open = data["open"].iloc[-1]
                    third_high = data["high"].iloc[-1]
                    third_low = data["low"].iloc[-1]
                    third_close = data["close"].iloc[-1]
                    
                    # 计算实体大小
                    first_body = abs(first_close - first_open)
                    second_body = abs(second_close - second_open)
                    third_body = abs(third_close - third_open)
                    
                    # 三白兵质量评分
                    quality_score = 0.6  # 基础分较高，因为是强形态
                    
                    # 检查基本形态特征（三根阳线）
                    is_bullish_candles = (first_close > first_open and 
                                        second_close > second_open and 
                                        third_close > third_open)
                    
                    if is_bullish_candles:
                        
                        # 递增高点评分（每根K线创新高）
                        increasing_highs = (second_high > first_high and third_high > second_high)
                        increasing_closes = (second_close > first_close and third_close > second_close)
                        
                        if increasing_highs and increasing_closes:
                            quality_score += 0.2  # 标准递增模式
                        elif increasing_closes:
                            quality_score += 0.1  # 至少收盘价递增
                        
                        # 开盘价位置评分（在前一根实体内开盘）
                        second_open_in_first_body = (min(first_open, first_close) <= second_open <= max(first_open, first_close))
                        third_open_in_second_body = (min(second_open, second_close) <= third_open <= max(second_open, second_close))
                        
                        if second_open_in_first_body and third_open_in_second_body:
                            quality_score += 0.15  # 理想的开盘位置
                        elif second_open_in_first_body or third_open_in_second_body:
                            quality_score += 0.08  # 部分符合
                        
                        # 实体大小评分（长阳线特征）
                        avg_body = (first_body + second_body + third_body) / 3
                        if first_body > 0 and second_body > 0 and third_body > 0:
                            body_consistency = min(first_body, second_body, third_body) / max(first_body, second_body, third_body)
                            
                            if body_consistency > 0.7:  # 实体大小相对一致
                                quality_score += 0.1
                            
                            # 检查实体是否足够大（相对于价格范围）
                            avg_price = (first_close + second_close + third_close) / 3
                            if avg_body / avg_price > 0.02:  # 实体大于平均价格的2%
                                quality_score += 0.1
                        
                        # 影线长度评分（短影线更佳）
                        first_lower_shadow = min(first_open, first_close) - first_low
                        first_upper_shadow = first_high - max(first_open, first_close)
                        second_lower_shadow = min(second_open, second_close) - second_low
                        second_upper_shadow = second_high - max(second_open, second_close)
                        third_lower_shadow = min(third_open, third_close) - third_low
                        third_upper_shadow = third_high - max(third_open, third_close)
                        
                        avg_lower_shadow = (first_lower_shadow + second_lower_shadow + third_lower_shadow) / 3
                        
                        if avg_body > 0 and avg_lower_shadow / avg_body < 0.3:  # 下影线较短
                            quality_score += 0.05
                        
                        # 根据质量调整信号强度
                        strength = min(0.98, 0.80 + quality_score * 0.15)
                        confidence = min(0.98, 0.80 + quality_score * 0.15)
                        
                        metadata = {
                            "pattern_type": "three_white_soldiers",
                            "increasing_highs": increasing_highs,
                            "increasing_closes": increasing_closes,
                            "second_open_in_first_body": second_open_in_first_body,
                            "third_open_in_second_body": third_open_in_second_body,
                            "body_consistency": round(body_consistency if 'body_consistency' in locals() else 0.0, 3),
                            "quality_score": round(quality_score, 3),
                            "pattern_strength": "extreme" if quality_score > 0.9 else "high",
                            "first_candle_type": "strong_bullish",
                            "second_candle_type": "strong_bullish", 
                            "third_candle_type": "strong_bullish",
                            "pattern_grade": "perfect" if quality_score > 0.9 else "standard"
                        }

            # 检查成交量确认（三白兵形态+递增成交量更可靠）
            if signal_type == "buy" and len(data) >= 5:
                volume_1 = data["volume"].iloc[-3]  # 第一个白兵
                volume_2 = data["volume"].iloc[-2]  # 第二个白兵  
                volume_3 = data["volume"].iloc[-1]  # 第三个白兵
                avg_volume = data["volume"].tail(7).mean()  # 前7天平均
                
                # 检查成交量递增趋势
                volume_increasing = volume_2 > volume_1 and volume_3 > volume_2
                high_volume_ratio = volume_3 / avg_volume if avg_volume > 0 else 1.0
                
                if volume_increasing and high_volume_ratio > 1.5:  # 递增且放量
                    strength = min(0.98, strength + 0.08)
                    confidence = min(0.98, confidence + 0.08)
                    metadata["volume_confirmation"] = "strong_increasing"
                    reason += "，成交量递增放大确认"
                elif volume_increasing:  # 仅递增
                    strength = min(0.95, strength + 0.05)
                    confidence = min(0.95, confidence + 0.05)
                    metadata["volume_confirmation"] = "increasing"
                elif high_volume_ratio > 1.3:  # 仅放量
                    strength = min(0.95, strength + 0.03)
                    confidence = min(0.95, confidence + 0.03)
                    metadata["volume_confirmation"] = "high"
                else:
                    metadata["volume_confirmation"] = "normal"
                
                metadata["final_volume_ratio"] = round(high_volume_ratio, 3)
                metadata["volume_trend"] = "increasing" if volume_increasing else "normal"

            # 分析趋势背景，增强三白兵形态的信号强度
            if signal_type == "buy" and len(data) >= 8:
                recent_closes = data["close"].tail(8)
                price_trend = recent_closes.diff().mean()
                
                # 三白兵在下跌趋势底部更有效（底部反转）
                if price_trend < -0.1:
                    strength = min(0.98, strength + 0.08)
                    confidence = min(0.98, confidence + 0.08)
                    metadata["trend_context"] = "downtrend_reversal"
                    metadata["signal_enhancement"] = "bottom_reversal"
                    reason = reason.replace("极强看涨持续信号", "极强底部反转信号")
                
                # 三白兵在上涨趋势中作为持续信号也很强
                elif price_trend > 0.05:
                    strength = min(0.96, strength + 0.05)
                    confidence = min(0.96, confidence + 0.05)
                    metadata["trend_context"] = "uptrend_continuation"
                    metadata["signal_enhancement"] = "bullish_continuation"
                    reason = reason.replace("极强看涨持续信号", "极强上涨延续信号")
                
                else:
                    metadata["trend_context"] = "sideways"
                    metadata["signal_enhancement"] = "normal"

            # 检查前期支撑位确认
            if signal_type == "buy" and len(data) >= 12:
                # 检查是否在重要支撑位附近
                recent_lows = data["low"].tail(12)
                current_low_area = data["low"].iloc[-3:-1].min()  # 前两根K线的最低价区域
                support_levels = recent_lows[recent_lows <= current_low_area * 1.02]  # 2%容忍度
                
                if len(support_levels) >= 3:  # 多次测试的支撑位
                    strength = min(0.98, strength + 0.05)
                    confidence = min(0.98, confidence + 0.05)
                    metadata["support_confirmation"] = "strong"
                    reason += "，重要支撑位确认"
                elif len(support_levels) >= 2:
                    metadata["support_confirmation"] = "moderate"
                else:
                    metadata["support_confirmation"] = "none"

            # 返回标准化信号格式
            return {
                'signal_type': signal_type,
                'strength': max(0.0, min(1.0, strength)),
                'confidence': max(0.0, min(1.0, confidence)),
                'timestamp': pd.Timestamp.now(),
                'reason': reason,
                'metadata': {
                    'indicator_name': self.name,
                    'pattern_category': 'continuation',
                    'pattern_family': 'candlestick',
                    'pattern_characteristic': 'extreme_bullish_continuation',
                    'requires_confirmation': 'three_candles',
                    'signal_direction': 'bullish_only',
                    **metadata
                }
            }

        except Exception as e:
            logger.error(f"三白兵形态信号生成失败: {e}")
            return self._get_default_signal(f"信号生成失败: {str(e)}")

    def _validate_signal_data(self, data: pd.DataFrame) -> bool:
        """验证输入数据"""
        if not isinstance(data, pd.DataFrame):
            return False
        
        if data.empty:
            return False
        
        # 检查必需列
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            return False
        
        # 检查数据量（三白兵形态至少需要3个数据点）
        if len(data) < 3:
            return False
        
        return True

    def _get_default_signal(self, reason: str = "默认持有") -> Dict[str, Any]:
        """获取默认信号"""
        return {
            'signal_type': 'hold',
            'strength': 0.0,
            'confidence': 0.5,
            'timestamp': pd.Timestamp.now(),
            'reason': reason,
            'metadata': {
                'indicator_name': self.name,
                'pattern_category': 'continuation',
                'pattern_family': 'candlestick',
                'pattern_characteristic': 'extreme_bullish_continuation',
                'requires_confirmation': 'three_candles',
                'signal_direction': 'bullish_only'
            }
        }
