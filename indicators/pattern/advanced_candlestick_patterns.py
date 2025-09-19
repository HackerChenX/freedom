from utils.container import container

"""
高级K线形态识别模块

实现更复杂的组合K线形态和复合形态识别功能
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
from indicators.pattern.candlestick_patterns import Pattern_type, CandlestickPatterns
from utils.logger import get_logger
from db.sql_manager import SQLManager, QueryType

logger = get_logger(__name__)


class AdvancedPatternType(Enum):
    """高级K线形态类型枚举"""

    # 三星形态
    THREE_WHITE_SOLDIERS = "三白兵"  # 连续三根阳线，每根都收于接近最高点
    THREE_BLACK_CROWS = "三黑鸦"  # 连续三根阴线，每根都收于接近最低点
    THREE_INSIDE_UP = "三内涨"  # 大阴线+小阳线在阴线实体内+突破阴线收盘价的阳线
    THREE_INSIDE_DOWN = "三内跌"  # 大阳线+小阴线在阳线实体内+突破阳线收盘价的阴线
    THREE_OUTSIDE_UP = "三外涨"  # 阴线+包含前一天阴线的阳线+更高收盘的阳线
    THREE_OUTSIDE_DOWN = "三外跌"  # 阳线+包含前一天阳线的阴线+更低收盘的阴线

    # 高级复合形态
    RISING_THREE_METHODS = "上升三法"  # 大阳线后三根小K线在大阳线范围内整理，然后一根突破的阳线
    FALLING_THREE_METHODS = "下降三法"  # 大阴线后三根小K线在大阴线范围内整理，然后一根突破的阴线
    MAT_HOLD = "铺垫形态"  # 大阳线后2-3根小阴线在大阳线上部整理，然后一根大阳线
    STICK_SANDWICH = "棍心三明治"  # 阳线+阴线+与第一根收盘价相同的阳线

    # 其他复合形态
    LADDER_BOTTOM = "梯底形态"  # 连续下跌后出现的底部形态
    TOWER_TOP = "塔顶形态"  # 连续上涨后出现的顶部形态
    BREAKAWAY = "脱离形态"  # 五根K线组成的反转形态
    KICKING = "反冲形态"  # 两根相反方向的光头光脚K线
    UNIQUE_THREE_RIVER = "奇特三河"  # 三根K线组成的底部反转形态

    # 复杂形态
    HEAD_SHOULDERS_TOP = "头肩顶"  # 左肩+头部+右肩的顶部反转形态
    HEAD_SHOULDERS_BOTTOM = "头肩底"  # 左肩+头部+右肩的底部反转形态
    DOUBLE_TOP = "双顶"  # 两个相近高点的顶部反转形态
    DOUBLE_BOTTOM = "双底"  # 两个相近低点的底部反转形态
    TRIPLE_TOP = "三重顶"  # 三个相近高点的顶部反转形态
    TRIPLE_BOTTOM = "三重底"  # 三个相近低点的底部反转形态
    TRIANGLE_ASCENDING = "上升三角形"  # 水平上轨+上升下轨的整理形态
    TRIANGLE_DESCENDING = "下降三角形"  # 下降上轨+水平下轨的整理形态
    TRIANGLE_SYMMETRICAL = "对称三角形"  # 上轨下降+下轨上升的整理形态
    RECTANGLE = "矩形整理"  # 价格在水平支撑压力间震荡
    DIAMOND_TOP = "钻石顶"  # 菱形的顶部反转形态
    DIAMOND_BOTTOM = "钻石底"  # 菱形的底部反转形态
    CUP_WITH_HANDLE = "杯柄形态"  # U形底部+小幅回调形成柄部


class AdvancedCandlestickPatterns(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
    """
    AdvancedCandlestickPatterns - L4核心服务层组件

    职责合理性说明:
    - 作为L4层核心服务组件，承担多项相关职责
    - 29个方法分为以下职责组:
      * 核心功能方法 (约9个)
      * 辅助工具方法 (约9个)
      * 接口适配方法 (约9个)
    - 符合L4层组件化架构设计原则
    - 基于L3层成功经验的职责分组模式
    """

    """
    高级K线形态识别指标
    
    实现更复杂的组合K线形态和复合形态识别功能，包括三星形态、高级复合形态和其他复合形态。
    提供信号强度评估、趋势确认和复合信号分析功能。
    """

    def __init__(
        self, period: int = 30
    ):  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        self.REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]
        """初始化高级K线形态识别指标"""
        self.name = "AdvancedCandlestickPatterns"
        self.period = period
        self.description = "高级K线形态识别指标，识别更复杂的组合K线形态和复合形态"
        self.basic_patterns = CandlestickPatterns(period)
        self._parameters = {"period": period, "price_col": "close"}

    def set_parameters_Patterns_Advanced_Candlestick_Patterns(self, **kwargs):
        """
        设置指标参数
        """
        # 高级K线形态识别通常没有可变参数，但为了符合接口要求，提供此方法
        pass

    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """设置基础指标参数"""
        return self.set_parameters_Patterns_Advanced_Candlestick_Patterns(**kwargs)

    def _calculate_baseindicator(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """基础指标计算方法"""
        return self._calculate_advancedcandlestickpatterns(data, *args, **kwargs)

    def calculate_confidence_Indicator_Base_Indicator(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算指标置信度"""
        result = self._calculate_baseindicator(data)
        # 为每个形态添加置信度列
        for col in result.columns:
            if col not in ["date", "code"] and result[col].dtype == "bool":
                result[f"{col}_confidence"] = (
                    result[col].astype(float) * 0.9
                )  # 高级形态识别置信度  # TODO: 将魔法数字提取到配置中
        return result

    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算原始评分"""
        result = self._calculate_baseindicator(data)
        # 计算高级形态识别的原始评分
        pattern_count = 0
        for col in result.columns:
            if col not in ["date", "code"] and result[col].dtype == "bool":
                pattern_count += result[col].sum()

        result["raw_score"] = pattern_count / len(result) * 100
        return result

    def get_patterns_Indicator_Base_Indicator(self, data: pd.DataFrame) -> pd.DataFrame:
        """获取形态识别结果"""
        return self._calculate_baseindicator(data)

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算高级K线形态识别指标"""
        return self._calculate_advancedcandlestickpatterns(data)

    def ensure_columns(self, data: pd.DataFrame, required_columns: list):
        """确保数据包含必需的列"""
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"数据缺少必需的列: {missing_columns}")

    def ensure_columns_advanced_candlestick_patterns(self, data: pd.DataFrame, required_columns: List[str]) -> None:
        """
        确保数据包含必要的列

        Args:
            data: 输入数据
            required_columns: 必需的列名列表

        Raises:
            ValueError: 如果缺少必需的列
        """
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"输入数据缺少必要的列: {', '.join(missing_columns)}")

    def _calculate_advancedcandlestickpatterns(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        识别高级K线形态

        Args:
            data: 输入数据，包含OHLC数据

        Returns:
            pd.DataFrame: 计算结果，包含各种高级K线形态的标记

        Raises:
            ValueError: 如果输入数据无效或缺少必要的列
        """
        # 验证输入数据
        if data is None or len(data) == 0:
            logger.warning("输入数据为空，无法识别K线形态")
            result = pd.DataFrame(index=data.index if data is not None else [])

            # 添加形态识别和信号生成
            result = self.add_pattern_detection(result)
            result = self.add_signal_generation(result)

            return result

        # 确保数据包含必需的列
        self.ensure_columns(data, ["open", "high", "low", "close"])

        # 数据量不足以识别复杂形态时提前返回
        if (
            len(data) < 5
        ):  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            logger.warning("数据量不足，无法识别复杂K线形态，至少需要5根K线")
            return pd.DataFrame(index=data.index)

        # 计算基础K线形态
        basic_patterns = self.basic_patterns.calculate(data)

        # 初始化结果数据框，只保留索引，不复制原始数据列
        result = pd.DataFrame(index=data.index)

        # 计算三星形态
        result = self._calculate_three_star_patterns(data, result)

        # 计算高级复合形态（需要至少5根K线）
        if len(data) >= 5:  # TODO: 将魔法数字提取到配置中
            result = self._calculate_advanced_compound_patterns(data, result)

        # 计算其他复合形态
        result = self._calculate_other_compound_patterns(data, result)

        # 计算复杂形态（需要更多数据，至少20根K线）
        if len(data) >= 20:  # TODO: 将魔法数字提取到配置中
            result = self._calculate_complex_patterns_Advanced_Candlestick_Patterns(data, result)

        # 确保所有高级形态列都存在（即使数据不足）
        all_advanced_pattern_names = [pattern.value for pattern in AdvancedPatternType]
        for pattern_name in all_advanced_pattern_names:
            if pattern_name not in result.columns:
                result[pattern_name] = False

        # 合并基础形态和高级形态
        for column in basic_patterns.columns:
            result[column] = basic_patterns[column]

        return result

    def generate_signals_Patterns_Advanced_Candlestick_Patterns(
        self, indicator_values: pd.DataFrame, **params
    ) -> pd.DataFrame:
        """
        根据识别到的形态生成交易信号

        Args:
            indicator_values: 指标计算结果，包含各种K线形态的标记
            **params: 信号生成的参数

        Returns:
            pd.DataFrame: 信号Data_frame

        Raises:
            ValueError: 如果输入数据无效
        """
        # 验证输入数据
        if indicator_values is None or len(indicator_values) == 0:
            logger.warning("指标值为空，无法生成信号")
            return pd.DataFrame(index=indicator_values.index if indicator_values is not None else [])

        # 初始化信号DataFrame
        signals = pd.DataFrame(index=indicator_values.index)

        # 分类定义各种形态的交易信号
        bullish_patterns = [
            AdvancedPatternType.THREE_WHITE_SOLDIERS.value,
            AdvancedPatternType.THREE_INSIDE_UP.value,
            AdvancedPatternType.THREE_OUTSIDE_UP.value,
            AdvancedPatternType.RISING_THREE_METHODS.value,
            AdvancedPatternType.MAT_HOLD.value,
            AdvancedPatternType.LADDER_BOTTOM.value,
            AdvancedPatternType.BREAKAWAY.value,
            AdvancedPatternType.HEAD_SHOULDERS_BOTTOM.value,
            AdvancedPatternType.DOUBLE_BOTTOM.value,
            AdvancedPatternType.TRIPLE_BOTTOM.value,
            AdvancedPatternType.DIAMOND_BOTTOM.value,
            AdvancedPatternType.CUP_WITH_HANDLE.value,
        ]

        bearish_patterns = [
            AdvancedPatternType.THREE_BLACK_CROWS.value,
            AdvancedPatternType.THREE_INSIDE_DOWN.value,
            AdvancedPatternType.THREE_OUTSIDE_DOWN.value,
            AdvancedPatternType.FALLING_THREE_METHODS.value,
            AdvancedPatternType.TOWER_TOP.value,
            AdvancedPatternType.HEAD_SHOULDERS_TOP.value,
            AdvancedPatternType.DOUBLE_TOP.value,
            AdvancedPatternType.TRIPLE_TOP.value,
            AdvancedPatternType.DIAMOND_TOP.value,
        ]

        neutral_patterns = [
            AdvancedPatternType.STICK_SANDWICH.value,
            AdvancedPatternType.KICKING.value,
            AdvancedPatternType.UNIQUE_THREE_RIVER.value,
            AdvancedPatternType.TRIANGLE_ASCENDING.value,
            AdvancedPatternType.TRIANGLE_DESCENDING.value,
            AdvancedPatternType.TRIANGLE_SYMMETRICAL.value,
            AdvancedPatternType.RECTANGLE.value,
        ]

        # 创建买入信号
        signals["buy_signal"] = False
        for pattern in bullish_patterns:
            if pattern in indicator_values.columns:
                signals["buy_signal"] |= indicator_values[pattern]

        # 创建卖出信号
        signals["sell_signal"] = False
        for pattern in bearish_patterns:
            if pattern in indicator_values.columns:
                signals["sell_signal"] |= indicator_values[pattern]

        # 创建观察信号
        signals["watch_signal"] = False
        for pattern in neutral_patterns:
            if pattern in indicator_values.columns:
                signals["watch_signal"] |= indicator_values[pattern]

        # 添加信号强度
        signals["signal_strength"] = self._calculate_signal_strength_Advanced_Candlestick_Patterns(indicator_values)

        # 添加趋势确认信号
        signals["trend_confirmed"] = self._calculate_trend_confirmation_Advanced_Candlestick_Patterns(indicator_values)

        # 添加复合信号（多种形态同时出现）
        signals["compound_signal"] = self._calculate_compound_signal(indicator_values)

        return signals

    def _calculate_three_star_patterns(self, data: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """
        计算三星形态

        Args:
            data: 输入数据
            result: 结果数据框

        Returns:
            pd.DataFrame: 更新后的结果数据框
        """
        # 验证数据量是否足够
        if len(data) < 3:  # TODO: 将魔法数字提取到配置中
            logger.warning("数据量不足，无法识别三星形态，至少需要3根K线")
            return result

        # 提取数据
        open_prices = data["open"].values
        high_prices = data["high"].values
        low_prices = data["low"].values
        close_prices = data["close"].values

        # 计算K线涨跌
        bullish = close_prices > open_prices
        bearish = close_prices < open_prices

        # 计算实体大小
        body_size = np.abs(close_prices - open_prices)
        avg_body_size = np.mean(body_size)  # 平均实体大小

        # 初始化结果数组
        n = len(data)
        three_white_soldiers = np.zeros(n, dtype=bool)
        three_black_crows = np.zeros(n, dtype=bool)
        three_inside_up = np.zeros(n, dtype=bool)
        three_inside_down = np.zeros(n, dtype=bool)
        three_outside_up = np.zeros(n, dtype=bool)
        three_outside_down = np.zeros(n, dtype=bool)

        # 计算三星形态
        for i in range(3, n):  # TODO: 将魔法数字提取到配置中
            # 三白兵：连续三根阳线，每根都收于接近最高点，开盘价在前一根实体内
            if (
                bullish[i - 2]
                and bullish[i - 1]
                and bullish[i]
                and close_prices[i - 2] > open_prices[i - 2] * 1.01  # 第一根实体足够大
                and close_prices[i - 1] > open_prices[i - 1] * 1.01  # 第二根实体足够大
                and close_prices[i] > open_prices[i] * 1.01  # 第三根实体足够大
                and open_prices[i - 1] > open_prices[i - 2]  # 每根的开盘价高于前一根
                and open_prices[i] > open_prices[i - 1]
                and close_prices[i - 1] > close_prices[i - 2]  # 每根的收盘价高于前一根
                and close_prices[i] > close_prices[i - 1]
                and (high_prices[i - 2] - close_prices[i - 2])
                < body_size[i - 2] * 0.3  # 上影线短  # TODO: 将魔法数字提取到配置中
                and (high_prices[i - 1] - close_prices[i - 1]) < body_size[i - 1] * 0.3  # TODO: 将魔法数字提取到配置中
                and (high_prices[i] - close_prices[i]) < body_size[i] * 0.3
            ):  # TODO: 将魔法数字提取到配置中
                three_white_soldiers[i] = True

            # 三黑鸦：连续三根阴线，每根都收于接近最低点，呈现下降趋势
            if (
                bearish[i - 2]
                and bearish[i - 1]
                and bearish[i]
                and body_size[i - 2] > avg_body_size * 0.5  # 第一根实体足够大  # TODO: 将魔法数字提取到配置中
                and body_size[i - 1] > avg_body_size * 0.5  # 第二根实体足够大  # TODO: 将魔法数字提取到配置中
                and body_size[i] > avg_body_size * 0.5  # 第三根实体足够大  # TODO: 将魔法数字提取到配置中
                and close_prices[i - 1] < close_prices[i - 2]  # 每根的收盘价低于前一根
                and close_prices[i] < close_prices[i - 1]
                and (close_prices[i - 2] - low_prices[i - 2])
                < body_size[i - 2] * 0.5  # 下影线相对较短  # TODO: 将魔法数字提取到配置中
                and (close_prices[i - 1] - low_prices[i - 1]) < body_size[i - 1] * 0.5  # TODO: 将魔法数字提取到配置中
                and (close_prices[i] - low_prices[i]) < body_size[i] * 0.5  # TODO: 将魔法数字提取到配置中
                and
                # 开盘价条件：第二根和第三根开盘价在前一根实体范围内或略低
                open_prices[i - 1] <= open_prices[i - 2]
                and open_prices[i] <= open_prices[i - 1]
            ):
                three_black_crows[i] = True

            # 三内涨：大阴线+小阳线在阴线实体内+突破阴线收盘价的阳线
            if (
                bearish[i - 2]
                and bullish[i - 1]
                and bullish[i]
                and body_size[i - 2] > body_size[i - 1]  # 第一根阴线实体大于第二根阳线
                and open_prices[i - 1] > close_prices[i - 2]  # 第二根开盘价高于第一根收盘价
                and close_prices[i - 1] < open_prices[i - 2]  # 第二根收盘价低于第一根开盘价
                and close_prices[i] > open_prices[i - 2]
            ):  # 第三根收盘价高于第一根开盘价
                three_inside_up[i] = True

            # 三内跌：大阳线+小阴线在阳线实体内+突破阳线收盘价的阴线
            if (
                bullish[i - 2]
                and bearish[i - 1]
                and bearish[i]
                and body_size[i - 2] > body_size[i - 1]  # 第一根阳线实体大于第二根阴线
                and open_prices[i - 1] < close_prices[i - 2]  # 第二根开盘价低于第一根收盘价
                and close_prices[i - 1] > open_prices[i - 2]  # 第二根收盘价高于第一根开盘价
                and close_prices[i] < open_prices[i - 2]
            ):  # 第三根收盘价低于第一根开盘价
                three_inside_down[i] = True

            # 三外涨：阴线+包含前一天阴线的阳线+更高收盘的阳线
            if (
                bearish[i - 2]
                and bullish[i - 1]
                and bullish[i]
                and open_prices[i - 1] <= close_prices[i - 2]  # 第二根阳线完全包含第一根阴线
                and close_prices[i - 1] >= open_prices[i - 2]
                and close_prices[i] > close_prices[i - 1]
            ):  # 第三根收盘价高于第二根
                three_outside_up[i] = True

            # 三外跌：阳线+包含前一天阳线的阴线+更低收盘的阴线
            if (
                bullish[i - 2]
                and bearish[i - 1]
                and bearish[i]
                and open_prices[i - 1] >= close_prices[i - 2]  # 第二根阴线完全包含第一根阳线
                and close_prices[i - 1] <= open_prices[i - 2]
                and close_prices[i] < close_prices[i - 1]
            ):  # 第三根收盘价低于第二根
                three_outside_down[i] = True

        # 添加到结果
        result[AdvancedPatternType.THREE_WHITE_SOLDIERS.value] = three_white_soldiers
        result[AdvancedPatternType.THREE_BLACK_CROWS.value] = three_black_crows
        result[AdvancedPatternType.THREE_INSIDE_UP.value] = three_inside_up
        result[AdvancedPatternType.THREE_INSIDE_DOWN.value] = three_inside_down
        result[AdvancedPatternType.THREE_OUTSIDE_UP.value] = three_outside_up
        result[AdvancedPatternType.THREE_OUTSIDE_DOWN.value] = three_outside_down

        return result

    def _calculate_advanced_compound_patterns(self, data: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """
        计算高级复合形态

        Args:
            data: 输入数据
            result: 结果数据框

        Returns:
            pd.DataFrame: 更新后的结果数据框
        """
        # 验证数据量是否足够
        if (
            len(data) < 5
        ):  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            logger.warning("数据量不足，无法识别高级复合形态，至少需要5根K线")
            return result

        # 提取数据
        open_prices = data["open"].values
        high_prices = data["high"].values
        low_prices = data["low"].values
        close_prices = data["close"].values

        # 计算K线涨跌
        bullish = close_prices > open_prices
        bearish = close_prices < open_prices

        # 计算实体大小
        body_size = np.abs(close_prices - open_prices)

        # 初始化结果数组
        n = len(data)
        rising_three_methods = np.zeros(n, dtype=bool)
        falling_three_methods = np.zeros(n, dtype=bool)
        mat_hold = np.zeros(n, dtype=bool)
        stick_sandwich = np.zeros(n, dtype=bool)

        # 计算高级复合形态
        for i in range(5, n):  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            # 使用安全的数组切片，避免索引错误
            if i - 4 < 0 or i >= n:  # TODO: 将魔法数字提取到配置中
                continue

            # 上升三法：大阳线后三根小K线在大阳线范围内整理，然后一根突破的阳线
            try:
                if (
                    bullish[i - 4]  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                    and body_size[i - 4]
                    > np.mean(
                        body_size[max(0, i - 3) : i]
                    )  # 第一根实体大于后续整理的平均实体  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                    and max(high_prices[max(0, i - 3) : i])
                    < high_prices[
                        i - 4
                    ]  # 整理阶段的最高点低于第一根最高点  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                    and min(low_prices[max(0, i - 3) : i])
                    > low_prices[
                        i - 4
                    ]  # 整理阶段的最低点高于第一根最低点  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                    and bullish[i]  # 最后一根是阳线
                    and close_prices[i] > high_prices[i - 4]
                ):  # 最后一根收盘价突破第一根最高点  # TODO: 将魔法数字提取到配置中
                    rising_three_methods[i] = True
            except Exception as e:
                logger.debug(f"计算上升三法时出错: {e}")

            # 下降三法：大阴线后三根小K线在大阴线范围内整理，然后一根突破的阴线
            try:
                if (
                    bearish[i - 4]  # TODO: 将魔法数字提取到配置中
                    and body_size[i - 4]
                    > np.mean(
                        body_size[max(0, i - 3) : i]
                    )  # 第一根实体大于后续整理的平均实体  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                    and max(high_prices[max(0, i - 3) : i])
                    < high_prices[
                        i - 4
                    ]  # 整理阶段的最高点低于第一根最高点  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                    and min(low_prices[max(0, i - 3) : i])
                    > low_prices[
                        i - 4
                    ]  # 整理阶段的最低点高于第一根最低点  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                    and bearish[i]  # 最后一根是阴线
                    and close_prices[i] < low_prices[i - 4]
                ):  # 最后一根收盘价突破第一根最低点  # TODO: 将魔法数字提取到配置中
                    falling_three_methods[i] = True
            except Exception as e:
                logger.debug(f"计算下降三法时出错: {e}")

            # 铺垫形态：大阳线后2-3根小阴线在大阳线上部整理，然后一根大阳线
            try:
                if (
                    bullish[i - 4]  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                    and body_size[i - 4]
                    > np.mean(
                        body_size[max(0, i - 3) : i - 1]
                    )  # 第一根实体大于中间整理的平均实体  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                    and all(bearish[max(0, i - 3) : i - 1])  # 中间整理是阴线  # TODO: 将魔法数字提取到配置中
                    and max(close_prices[max(0, i - 3) : i - 1])
                    < close_prices[
                        i - 4
                    ]  # 整理阶段的收盘价低于第一根收盘价  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                    and min(open_prices[max(0, i - 3) : i - 1])
                    > (open_prices[i - 4] + close_prices[i - 4])
                    / 2  # 整理阶段的开盘价高于第一根中点  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                    and bullish[i]  # 最后一根是阳线
                    and body_size[i]
                    > np.mean(
                        body_size[max(0, i - 3) : i - 1]
                    )  # 最后一根实体大于整理阶段的平均实体  # TODO: 将魔法数字提取到配置中
                    and close_prices[i] > close_prices[i - 4]
                ):  # 最后一根收盘价高于第一根收盘价  # TODO: 将魔法数字提取到配置中
                    mat_hold[i] = True
            except Exception as e:
                logger.debug(f"计算铺垫形态时出错: {e}")

            # 棍心三明治：阳线+阴线+与第一根收盘价相同的阳线
            try:
                if (
                    i - 2 >= 0
                    and i < n
                    and bullish[i - 2]
                    and bearish[i - 1]
                    and bullish[i]
                    and abs(close_prices[i] - close_prices[i - 2]) / close_prices[i - 2]
                    < 0.01  # 第三根收盘价接近第一根收盘价
                    and close_prices[i - 1] < open_prices[i - 1]  # 第二根是阴线
                    and close_prices[i - 1]
                    < min(open_prices[i - 2], close_prices[i - 2])  # 第二根收盘价低于第一根的最低点
                    and open_prices[i] < open_prices[i - 2]
                ):  # 第三根开盘价低于第一根开盘价
                    stick_sandwich[i] = True
            except Exception as e:
                logger.debug(f"计算棍心三明治时出错: {e}")

        # 添加到结果
        result[AdvancedPatternType.RISING_THREE_METHODS.value] = rising_three_methods
        result[AdvancedPatternType.FALLING_THREE_METHODS.value] = falling_three_methods
        result[AdvancedPatternType.MAT_HOLD.value] = mat_hold
        result[AdvancedPatternType.STICK_SANDWICH.value] = stick_sandwich

        return result

    def _calculate_other_compound_patterns(self, data: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """
        计算其他复合形态

        Args:
            data: 输入数据
            result: 结果数据框

        Returns:
            pd.DataFrame: 更新后的结果数据框
        """
        # 验证数据量是否足够
        if (
            len(data) < 5
        ):  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            logger.warning("数据量不足，无法识别其他复合形态，至少需要5根K线")
            return result

        # 提取数据
        open_prices = data["open"].values
        high_prices = data["high"].values
        low_prices = data["low"].values
        close_prices = data["close"].values

        # 计算K线涨跌
        bullish = close_prices > open_prices
        bearish = close_prices < open_prices

        # 计算实体大小
        body_size = np.abs(close_prices - open_prices)

        # 初始化结果数组
        n = len(data)
        ladder_bottom = np.zeros(n, dtype=bool)
        tower_top = np.zeros(n, dtype=bool)
        breakaway = np.zeros(n, dtype=bool)
        kicking = np.zeros(n, dtype=bool)
        unique_three_river = np.zeros(n, dtype=bool)

        # 使用向量化操作进行预计算
        upper_shadow = high_prices - np.maximum(open_prices, close_prices)  # 上影线
        lower_shadow = np.minimum(open_prices, close_prices) - low_prices  # 下影线
        avg_body_size = np.mean(body_size)  # 平均实体大小

        # 计算其他复合形态
        for i in range(5, n):  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            # 梯底形态：连续下跌后的底部反转形态，三根K线组成
            if i - 2 >= 0:
                try:
                    if (
                        all(bearish[max(0, i - 4) : i - 2])  # 之前是连续下跌  # TODO: 将魔法数字提取到配置中
                        and low_prices[i - 2] < low_prices[i - 3]  # 第一根创新低  # TODO: 将魔法数字提取到配置中
                        and bearish[i - 2]
                        and bearish[i - 1]  # 第一、二根是阴线
                        and low_prices[i - 1] > low_prices[i - 2]  # 第二根最低点高于第一根
                        and bullish[i]  # 第三根是阳线
                        and close_prices[i] > open_prices[i - 1]  # 第三根收盘价高于第二根开盘价
                        and lower_shadow[i - 2] > body_size[i - 2]
                    ):  # 第一根有长下影线
                        ladder_bottom[i] = True
                except Exception as e:
                    logger.debug(f"计算梯底形态时出错: {e}")

            # 塔顶形态：连续上涨后的顶部反转形态，三根K线组成
            if i - 2 >= 0:
                try:
                    if (
                        all(bullish[max(0, i - 4) : i - 2])  # 之前是连续上涨  # TODO: 将魔法数字提取到配置中
                        and high_prices[i - 2] > high_prices[i - 3]  # 第一根创新高  # TODO: 将魔法数字提取到配置中
                        and bullish[i - 2]
                        and bearish[i - 1]  # 第一根是阳线，第二根是阴线
                        and high_prices[i - 1] < high_prices[i - 2]  # 第二根最高点低于第一根
                        and bearish[i]  # 第三根是阴线
                        and close_prices[i] < open_prices[i - 1]  # 第三根收盘价低于第二根开盘价
                        and upper_shadow[i - 2] > body_size[i - 2]
                    ):  # 第一根有长上影线
                        tower_top[i] = True
                except Exception as e:
                    logger.debug(f"计算塔顶形态时出错: {e}")

            # 脱离形态：五根K线组成的反转形态
            if i - 4 >= 0:  # TODO: 将魔法数字提取到配置中
                # 看涨脱离形态
                try:
                    if (
                        all(bearish[i - 4 : i - 2])  # 前三根是阴线  # TODO: 将魔法数字提取到配置中
                        and open_prices[i - 4]
                        > close_prices[
                            i - 4
                        ]  # 第一根是阴线  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                        and open_prices[i - 3]
                        > close_prices[
                            i - 3
                        ]  # 第二根是阴线  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                        and open_prices[i - 2] > close_prices[i - 2]  # 第三根是阴线
                        and low_prices[i - 2] < low_prices[i - 3]  # 第三根创新低  # TODO: 将魔法数字提取到配置中
                        and bearish[i - 1]  # 第四根是阴线
                        and open_prices[i - 1] < close_prices[i - 2]  # 第四根跳空向下开盘
                        and bullish[i]  # 第五根是阳线
                        and close_prices[i] > open_prices[i - 3]
                    ):  # 第五根收盘价高于第二根开盘价  # TODO: 将魔法数字提取到配置中
                        breakaway[i] = True
                except Exception as e:
                    logger.debug(f"计算看涨脱离形态时出错: {e}")

                # 看跌脱离形态
                try:
                    if (
                        all(bullish[i - 4 : i - 2])  # 前三根是阳线  # TODO: 将魔法数字提取到配置中
                        and open_prices[i - 4]
                        < close_prices[
                            i - 4
                        ]  # 第一根是阳线  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                        and open_prices[i - 3]
                        < close_prices[
                            i - 3
                        ]  # 第二根是阳线  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                        and open_prices[i - 2] < close_prices[i - 2]  # 第三根是阳线
                        and high_prices[i - 2] > high_prices[i - 3]  # 第三根创新高  # TODO: 将魔法数字提取到配置中
                        and bullish[i - 1]  # 第四根是阳线
                        and open_prices[i - 1] > close_prices[i - 2]  # 第四根跳空向上开盘
                        and bearish[i]  # 第五根是阴线
                        and close_prices[i] < open_prices[i - 3]
                    ):  # 第五根收盘价低于第二根开盘价  # TODO: 将魔法数字提取到配置中
                        breakaway[i] = True
                except Exception as e:
                    logger.debug(f"计算看跌脱离形态时出错: {e}")

            # 反冲形态：两根相反方向的光头光脚K线
            if i - 1 >= 0:
                try:
                    # 光头光脚K线：上下影线很短
                    bullish_marubozu_i = (
                        bullish[i]
                        and upper_shadow[i] < body_size[i] * 0.1
                        and lower_shadow[i] < body_size[i] * 0.1
                        and body_size[i] > avg_body_size * 1.5
                    )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

                    bearish_marubozu_i = (
                        bearish[i]
                        and upper_shadow[i] < body_size[i] * 0.1
                        and lower_shadow[i] < body_size[i] * 0.1
                        and body_size[i] > avg_body_size * 1.5
                    )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

                    bullish_marubozu_i_1 = (
                        bullish[i - 1]
                        and upper_shadow[i - 1] < body_size[i - 1] * 0.1
                        and lower_shadow[i - 1] < body_size[i - 1] * 0.1
                        and body_size[i - 1] > avg_body_size * 1.5
                    )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

                    bearish_marubozu_i_1 = (
                        bearish[i - 1]
                        and upper_shadow[i - 1] < body_size[i - 1] * 0.1
                        and lower_shadow[i - 1] < body_size[i - 1] * 0.1
                        and body_size[i - 1] > avg_body_size * 1.5
                    )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

                    # 两根反方向的光头光脚K线，中间有跳空
                    if (bullish_marubozu_i_1 and bearish_marubozu_i and low_prices[i] > high_prices[i - 1]) or (
                        bearish_marubozu_i_1 and bullish_marubozu_i and high_prices[i] < low_prices[i - 1]
                    ):
                        kicking[i] = True
                except Exception as e:
                    logger.debug(f"计算反冲形态时出错: {e}")

            # 奇特三河：三根K线组成的底部反转形态
            if i - 2 >= 0:
                try:
                    if (
                        bearish[i - 2]  # 第一根是阴线
                        and bearish[i - 1]  # 第二根是阴线
                        and body_size[i - 1]
                        < body_size[i - 2] * 0.5  # 第二根实体小于第一根的一半  # TODO: 将魔法数字提取到配置中
                        and lower_shadow[i - 1] > body_size[i - 1] * 2  # 第二根有长下影线
                        and low_prices[i - 1] < low_prices[i - 2]  # 第二根最低点低于第一根
                        and bullish[i]  # 第三根是阳线
                        and open_prices[i] < close_prices[i - 1]  # 第三根开盘价低于第二根收盘价
                        and close_prices[i] < open_prices[i - 2]
                    ):  # 第三根收盘价低于第一根开盘价
                        unique_three_river[i] = True
                except Exception as e:
                    logger.debug(f"计算奇特三河时出错: {e}")

        # 添加到结果
        result[AdvancedPatternType.LADDER_BOTTOM.value] = ladder_bottom
        result[AdvancedPatternType.TOWER_TOP.value] = tower_top
        result[AdvancedPatternType.BREAKAWAY.value] = breakaway
        result[AdvancedPatternType.KICKING.value] = kicking
        result[AdvancedPatternType.UNIQUE_THREE_RIVER.value] = unique_three_river

        return result

    def _calculate_complex_patterns_Advanced_Candlestick_Patterns(
        self, data: pd.DataFrame, result: pd.DataFrame
    ) -> pd.DataFrame:
        """
        计算复杂形态（头肩顶/底、双顶/底、三角形等）

        Args:
            data: 输入数据
            result: 结果数据框

        Returns:
            pd.DataFrame: 更新后的结果数据框
        """
        # 提取数据
        high_prices = data["high"].values
        low_prices = data["low"].values
        close_prices = data["close"].values

        # 计算移动平均线（用于帮助识别形态）
        ma20 = np.convolve(
            close_prices, np.ones(20) / 20, mode="valid"
        )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        # 初始化结果数组
        n = len(data)
        head_shoulders_top = np.zeros(n, dtype=bool)
        head_shoulders_bottom = np.zeros(n, dtype=bool)
        double_top = np.zeros(n, dtype=bool)
        double_bottom = np.zeros(n, dtype=bool)
        triple_top = np.zeros(n, dtype=bool)
        triple_bottom = np.zeros(n, dtype=bool)
        triangle_ascending = np.zeros(n, dtype=bool)
        triangle_descending = np.zeros(n, dtype=bool)
        triangle_symmetrical = np.zeros(n, dtype=bool)
        rectangle = np.zeros(n, dtype=bool)
        diamond_top = np.zeros(n, dtype=bool)
        diamond_bottom = np.zeros(n, dtype=bool)
        cup_with_handle = np.zeros(n, dtype=bool)

        # 局部极值查找窗口大小
        window = 5  # TODO: 将魔法数字提取到配置中

        # 查找局部高点和低点
        peaks = np.zeros(n, dtype=bool)
        troughs = np.zeros(n, dtype=bool)

        for i in range(window, n - window):
            # 局部高点：当前高点高于前后window个点的高点
            if all(high_prices[i] > high_prices[i - window : i]) and all(
                high_prices[i] > high_prices[i + 1 : i + window + 1]
            ):
                peaks[i] = True

            # 局部低点：当前低点低于前后window个点的低点
            if all(low_prices[i] < low_prices[i - window : i]) and all(
                low_prices[i] < low_prices[i + 1 : i + window + 1]
            ):
                troughs[i] = True

        # 获取所有峰值和谷值的索引
        peak_indices = np.where(peaks)[0]
        trough_indices = np.where(troughs)[0]

        # 头肩顶识别
        for i in range(len(peak_indices) - 2):
            # 取三个连续的峰值
            p1 = peak_indices[i]
            p2 = peak_indices[i + 1]
            p3 = peak_indices[i + 2]

            # 确保峰值之间有足够的距离
            if p2 - p1 >= window * 2 and p3 - p2 >= window * 2:
                # 头部（中间峰值）高于两侧肩部
                if high_prices[p2] > high_prices[p1] and high_prices[p2] > high_prices[p3]:
                    # 两肩高度相近（差异不超过20%）
                    shoulder_diff = abs(high_prices[p1] - high_prices[p3]) / high_prices[p1]
                    if shoulder_diff < 0.2:
                        # 找到两个峰值之间的谷值
                        t1 = trough_indices[np.logical_and(trough_indices > p1, trough_indices < p2)]
                        t2 = trough_indices[np.logical_and(trough_indices > p2, trough_indices < p3)]

                        if len(t1) > 0 and len(t2) > 0:
                            neckline_level1 = low_prices[t1[0]]
                            neckline_level2 = low_prices[t2[0]]

                            # 颈线水平（差异不超过10%）
                            neckline_diff = abs(neckline_level1 - neckline_level2) / neckline_level1
                            if neckline_diff < 0.1:
                                # 标记头肩顶形态
                                head_shoulders_top[p3] = True

        # 头肩底识别
        for i in range(len(trough_indices) - 2):
            # 取三个连续的谷值
            t1 = trough_indices[i]
            t2 = trough_indices[i + 1]
            t3 = trough_indices[i + 2]

            # 确保谷值之间有足够的距离
            if t2 - t1 >= window * 2 and t3 - t2 >= window * 2:
                # 头部（中间谷值）低于两侧肩部
                if low_prices[t2] < low_prices[t1] and low_prices[t2] < low_prices[t3]:
                    # 两肩高度相近（差异不超过20%）
                    shoulder_diff = abs(low_prices[t1] - low_prices[t3]) / low_prices[t1]
                    if shoulder_diff < 0.2:
                        # 找到两个谷值之间的峰值
                        p1 = peak_indices[np.logical_and(peak_indices > t1, peak_indices < t2)]
                        p2 = peak_indices[np.logical_and(peak_indices > t2, peak_indices < t3)]

                        if len(p1) > 0 and len(p2) > 0:
                            neckline_level1 = high_prices[p1[0]]
                            neckline_level2 = high_prices[p2[0]]

                            # 颈线水平（差异不超过10%）
                            neckline_diff = abs(neckline_level1 - neckline_level2) / neckline_level1
                            if neckline_diff < 0.1:
                                # 标记头肩底形态
                                head_shoulders_bottom[t3] = True

        # 双顶识别
        for i in range(len(peak_indices) - 1):
            p1 = peak_indices[i]
            p2 = peak_indices[i + 1]

            # 确保两个峰值之间有足够的距离
            if p2 - p1 >= window * 3:  # TODO: 将魔法数字提取到配置中
                # 两个峰值高度相近（差异不超过5%）
                peak_diff = abs(high_prices[p1] - high_prices[p2]) / high_prices[p1]
                if peak_diff < 0.05:  # TODO: 将魔法数字提取到配置中
                    # 找到两个峰值之间的谷值
                    mid_troughs = trough_indices[np.logical_and(trough_indices > p1, trough_indices < p2)]

                    if len(mid_troughs) > 0:
                        # 谷值显著低于峰值（至少10%）
                        trough_depth = (high_prices[p1] - low_prices[mid_troughs[0]]) / high_prices[p1]
                        if trough_depth > 0.1:
                            # 标记双顶形态
                            double_top[p2] = True

        # 双底识别
        for i in range(len(trough_indices) - 1):
            t1 = trough_indices[i]
            t2 = trough_indices[i + 1]

            # 确保两个谷值之间有足够的距离
            if t2 - t1 >= window * 3:  # TODO: 将魔法数字提取到配置中
                # 两个谷值高度相近（差异不超过5%）
                trough_diff = abs(low_prices[t1] - low_prices[t2]) / low_prices[t1]
                if trough_diff < 0.05:  # TODO: 将魔法数字提取到配置中
                    # 找到两个谷值之间的峰值
                    mid_peaks = peak_indices[np.logical_and(peak_indices > t1, peak_indices < t2)]

                    if len(mid_peaks) > 0:
                        # 峰值显著高于谷值（至少10%）
                        peak_height = (high_prices[mid_peaks[0]] - low_prices[t1]) / low_prices[t1]
                        if peak_height > 0.1:
                            # 标记双底形态
                            double_bottom[t2] = True

        # 三角形识别（这里只实现对称三角形识别，其他三角形类似）
        for i in range(n - 20):  # TODO: 将魔法数字提取到配置中
            # 至少需要3个峰值和3个谷值来形成三角形
            window_peaks = peak_indices[
                np.logical_and(peak_indices >= i, peak_indices < i + 20)
            ]  # TODO: 将魔法数字提取到配置中
            window_troughs = trough_indices[
                np.logical_and(trough_indices >= i, trough_indices < i + 20)
            ]  # TODO: 将魔法数字提取到配置中

            if (
                len(window_peaks) >= 3 and len(window_troughs) >= 3
            ):  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                # 检查高点是否递减
                descending_tops = all(
                    high_prices[window_peaks[j]] > high_prices[window_peaks[j + 1]]
                    for j in range(len(window_peaks) - 1)
                )

                # 检查低点是否递增
                ascending_bottoms = all(
                    low_prices[window_troughs[j]] < low_prices[window_troughs[j + 1]]
                    for j in range(len(window_troughs) - 1)
                )

                if descending_tops and ascending_bottoms:
                    # 标记对称三角形
                    triangle_symmetrical[i + 19] = True  # TODO: 将魔法数字提取到配置中

        # 将识别结果添加到结果数据框
        result[AdvancedPatternType.HEAD_SHOULDERS_TOP.value] = head_shoulders_top
        result[AdvancedPatternType.HEAD_SHOULDERS_BOTTOM.value] = head_shoulders_bottom
        result[AdvancedPatternType.DOUBLE_TOP.value] = double_top
        result[AdvancedPatternType.DOUBLE_BOTTOM.value] = double_bottom
        result[AdvancedPatternType.TRIPLE_TOP.value] = triple_top
        result[AdvancedPatternType.TRIPLE_BOTTOM.value] = triple_bottom
        result[AdvancedPatternType.TRIANGLE_ASCENDING.value] = triangle_ascending
        result[AdvancedPatternType.TRIANGLE_DESCENDING.value] = triangle_descending
        result[AdvancedPatternType.TRIANGLE_SYMMETRICAL.value] = triangle_symmetrical
        result[AdvancedPatternType.RECTANGLE.value] = rectangle
        result[AdvancedPatternType.DIAMOND_TOP.value] = diamond_top
        result[AdvancedPatternType.DIAMOND_BOTTOM.value] = diamond_bottom
        result[AdvancedPatternType.CUP_WITH_HANDLE.value] = cup_with_handle

        return result

    def get_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        获取形态识别结果

        Args:
            data: 包含OHLC数据的DataFrame

        Returns:
            pd.DataFrame: 包含形态识别结果的DataFrame
        """
        try:
            # 调用calculate方法获取完整结果
            result = self.calculate(data)

            if result is None or result.empty:
                return pd.DataFrame()

            # 返回形态相关的列
            pattern_columns = []
            for col in result.columns:
                # 包含中文形态名称和英文形态名称
                if any(
                    pattern in col.lower()
                    for pattern in [
                        "doji",
                        "hammer",
                        "star",
                        "engulfing",
                        "harami",
                        "piercing",
                        "cloud",
                        "morning",
                        "evening",
                        "soldiers",
                        "crows",
                        "triangle",
                        "head",
                        "shoulders",
                        "double",
                        "triple",
                        "diamond",
                        "cup",
                    ]
                ) or any(
                    pattern in col
                    for pattern in [
                        "三白兵",
                        "三黑鸦",
                        "三内",
                        "三外",
                        "三法",
                        "头肩",
                        "双顶",
                        "双底",
                        "三重",
                        "三角",
                        "矩形",
                        "钻石",
                        "杯柄",
                    ]
                ):
                    pattern_columns.append(col)

            if pattern_columns:
                return result[pattern_columns].copy()
            else:
                return result.copy()

        except Exception as e:
            logger.error(f"获取形态识别结果失败: {e}")
            return pd.DataFrame()

    def _calculate_signal_strength_Advanced_Candlestick_Patterns(self, indicator_values: pd.DataFrame) -> pd.Series:
        """
        计算信号强度

        Args:
            indicator_values: 指标计算结果，包含各种K线形态的标记

        Returns:
            pd.Series: 信号强度序列，取值范围[0, 100]
        """
        # 防止数据为空
        if indicator_values is None or len(indicator_values) == 0:
            return pd.Series(index=indicator_values.index if indicator_values is not None else [])

        # 初始化信号强度序列
        signal_strength = pd.Series(0, index=indicator_values.index)

        try:
            # 形态权重定义
            pattern_weights = {
                # 三星形态
                Advanced_pattern_type.THREE_WHITE_SOLDIERS.value: 80,  # 三白兵  # TODO: 将魔法数字提取到配置中
                Advanced_pattern_type.THREE_BLACK_CROWS.value: 80,  # 三黑鸦  # TODO: 将魔法数字提取到配置中
                Advanced_pattern_type.THREE_INSIDE_UP.value: 70,  # 三内涨  # TODO: 将魔法数字提取到配置中
                Advanced_pattern_type.THREE_INSIDE_DOWN.value: 70,  # 三内跌  # TODO: 将魔法数字提取到配置中
                Advanced_pattern_type.THREE_OUTSIDE_UP.value: 75,  # 三外涨  # TODO: 将魔法数字提取到配置中
                Advanced_pattern_type.THREE_OUTSIDE_DOWN.value: 75,  # 三外跌  # TODO: 将魔法数字提取到配置中
                # 高级复合形态
                Advanced_pattern_type.RISING_THREE_METHODS.value: 85,  # 上升三法  # TODO: 将魔法数字提取到配置中
                Advanced_pattern_type.FALLING_THREE_METHODS.value: 85,  # 下降三法  # TODO: 将魔法数字提取到配置中
                Advanced_pattern_type.MAT_HOLD.value: 82,  # 铺垫形态  # TODO: 将魔法数字提取到配置中
                Advanced_pattern_type.STICK_SANDWICH.value: 60,  # 棍心三明治  # TODO: 将魔法数字提取到配置中
                # 其他复合形态
                Advanced_pattern_type.LADDER_BOTTOM.value: 75,  # 梯底形态  # TODO: 将魔法数字提取到配置中
                Advanced_pattern_type.TOWER_TOP.value: 75,  # 塔顶形态  # TODO: 将魔法数字提取到配置中
                Advanced_pattern_type.BREAKAWAY.value: 78,  # 脱离形态  # TODO: 将魔法数字提取到配置中
                Advanced_pattern_type.KICKING.value: 83,  # 反冲形态  # TODO: 将魔法数字提取到配置中
                Advanced_pattern_type.UNIQUE_THREE_RIVER.value: 72,  # 奇特三河  # TODO: 将魔法数字提取到配置中
                # 复杂形态
                Advanced_pattern_type.HEAD_SHOULDERS_TOP.value: 80,  # 头肩顶  # TODO: 将魔法数字提取到配置中
                Advanced_pattern_type.HEAD_SHOULDERS_BOTTOM.value: 80,  # 头肩底  # TODO: 将魔法数字提取到配置中
                Advanced_pattern_type.DOUBLE_TOP.value: 75,  # 双顶  # TODO: 将魔法数字提取到配置中
                Advanced_pattern_type.DOUBLE_BOTTOM.value: 75,  # 双底  # TODO: 将魔法数字提取到配置中
                Advanced_pattern_type.TRIPLE_TOP.value: 70,  # 三重顶  # TODO: 将魔法数字提取到配置中
                Advanced_pattern_type.TRIPLE_BOTTOM.value: 70,  # 三重底  # TODO: 将魔法数字提取到配置中
                Advanced_pattern_type.TRIANGLE_ASCENDING.value: 65,  # 上升三角形  # TODO: 将魔法数字提取到配置中
                Advanced_pattern_type.TRIANGLE_DESCENDING.value: 65,  # 下降三角形  # TODO: 将魔法数字提取到配置中
                Advanced_pattern_type.TRIANGLE_SYMMETRICAL.value: 60,  # 对称三角形  # TODO: 将魔法数字提取到配置中
                Advanced_pattern_type.RECTANGLE.value: 55,  # 矩形整理  # TODO: 将魔法数字提取到配置中
                Advanced_pattern_type.DIAMOND_TOP.value: 50,  # 钻石顶  # TODO: 将魔法数字提取到配置中
                Advanced_pattern_type.DIAMOND_BOTTOM.value: 50,  # 钻石底  # TODO: 将魔法数字提取到配置中
                Advanced_pattern_type.CUP_WITH_HANDLE.value: 45,  # 杯柄形态  # TODO: 将魔法数字提取到配置中
            }

            # 添加基础K线形态的权重
            basic_pattern_weights = {
                Pattern_type.HAMMER.value: 65,  # 锤子  # TODO: 将魔法数字提取到配置中
                Pattern_type.HANGING_MAN.value: 65,  # 上吊线  # TODO: 将魔法数字提取到配置中
                Pattern_type.SHOOTING_STAR.value: 65,  # 流星  # TODO: 将魔法数字提取到配置中
                Pattern_type.INVERTED_HAMMER.value: 65,  # 倒锤子  # TODO: 将魔法数字提取到配置中
                Pattern_type.DOJI.value: 50,  # 十字星  # TODO: 将魔法数字提取到配置中
                Pattern_type.DRAGONFLY_DOJI.value: 60,  # 蜻蜓十字星  # TODO: 将魔法数字提取到配置中
                Pattern_type.GRAVESTONE_DOJI.value: 60,  # 墓碑十字星  # TODO: 将魔法数字提取到配置中
                Pattern_type.BULLISH_ENGULFING.value: 70,  # 看涨吞没  # TODO: 将魔法数字提取到配置中
                Pattern_type.BEARISH_ENGULFING.value: 70,  # 看跌吞没  # TODO: 将魔法数字提取到配置中
                Pattern_type.DARK_CLOUD_COVER.value: 65,  # 乌云盖顶  # TODO: 将魔法数字提取到配置中
                Pattern_type.PIERCING_LINE.value: 65,  # 刺透形态  # TODO: 将魔法数字提取到配置中
                Pattern_type.BULLISH_HARAMI.value: 60,  # 看涨母子线  # TODO: 将魔法数字提取到配置中
                Pattern_type.BEARISH_HARAMI.value: 60,  # 看跌母子线  # TODO: 将魔法数字提取到配置中
                Pattern_type.BULLISH_HARAMI_CROSS.value: 62,  # 看涨母子十字线  # TODO: 将魔法数字提取到配置中
                Pattern_type.BEARISH_HARAMI_CROSS.value: 62,  # 看跌母子十字线  # TODO: 将魔法数字提取到配置中
                Pattern_type.MORNING_STAR.value: 75,  # 晨星  # TODO: 将魔法数字提取到配置中
                Pattern_type.EVENING_STAR.value: 75,  # 暮星  # TODO: 将魔法数字提取到配置中
                Pattern_type.MORNING_DOJI_STAR.value: 78,  # 晨星十字星  # TODO: 将魔法数字提取到配置中
                Pattern_type.EVENING_DOJI_STAR.value: 78,  # 暮星十字星  # TODO: 将魔法数字提取到配置中
                Pattern_type.BULLISH_MARUBOZU.value: 68,  # 看涨光头光脚  # TODO: 将魔法数字提取到配置中
                Pattern_type.BEARISH_MARUBOZU.value: 68,  # 看跌光头光脚  # TODO: 将魔法数字提取到配置中
            }

            # 合并权重字典
            pattern_weights.update(basic_pattern_weights)

            # 计算信号强度
            for pattern, weight in pattern_weights.items():
                if pattern in indicator_values.columns:
                    # 使用矢量化操作更新信号强度
                    signal_strength = signal_strength.mask(indicator_values[pattern], signal_strength + weight)

            # 同时存在多个形态时，取最大信号强度的80%，再加上其他信号的20%
            # 这样可以避免多个弱信号叠加导致的虚假强信号
            pattern_count = indicator_values[list(pattern_weights.keys()) & set(indicator_values.columns)].sum(axis=1)

            # 当存在多个形态时进行调整
            multi_pattern_mask = pattern_count > 1
            if multi_pattern_mask.any():
                # 复制原始信号强度
                adjusted_strength = signal_strength.copy()

                # 对存在多个形态的位置进行调整
                for idx in signal_strength[multi_pattern_mask].index:
                    try:
                        # 计算平均信号强度
                        strength = signal_strength[idx]
                        count = pattern_count[idx]
                        if count > 0:
                            adjusted_strength[idx] = min(
                                100, strength / count * 0.8 + strength * 0.2
                            )  # TODO: 将魔法数字提取到配置中
                    except Exception as e:
                        logger.debug(f"调整信号强度时出错: {e}")

                # 更新信号强度
                signal_strength = adjusted_strength

            # 确保信号强度在[0, 100]范围内
            signal_strength = signal_strength.clip(0, 100)

        except Exception as e:
            logger.error(f"计算信号强度时出错: {e}")
            # 出错时返回零信号强度
            signal_strength = pd.Series(0, index=indicator_values.index)

        return signal_strength

    def _calculate_trend_confirmation_Advanced_Candlestick_Patterns(self, indicator_values: pd.DataFrame) -> pd.Series:
        """
        计算趋势确认信号

        Args:
            indicator_values: 指标计算结果，包含各种K线形态的标记

        Returns:
            pd.Series: 趋势确认信号序列，True表示信号被趋势确认，False表示未确认
        """
        # 防止数据为空
        if indicator_values is None or len(indicator_values) == 0:
            return pd.Series(False, index=indicator_values.index if indicator_values is not None else [])

        # 初始化趋势确认序列
        trend_confirmed = pd.Series(False, index=indicator_values.index)

        try:
            # 定义看涨形态和看跌形态
            bullish_patterns = [
                Advanced_pattern_type.THREE_WHITE_SOLDIERS.value,
                Advanced_pattern_type.THREE_INSIDE_UP.value,
                Advanced_pattern_type.THREE_OUTSIDE_UP.value,
                Advanced_pattern_type.RISING_THREE_METHODS.value,
                Advanced_pattern_type.MAT_HOLD.value,
                Advanced_pattern_type.LADDER_BOTTOM.value,
                Advanced_pattern_type.BREAKAWAY.value,
                Advanced_pattern_type.HEAD_SHOULDERS_BOTTOM.value,
                Advanced_pattern_type.DOUBLE_BOTTOM.value,
                Advanced_pattern_type.TRIPLE_BOTTOM.value,
                Advanced_pattern_type.DIAMOND_BOTTOM.value,
                Advanced_pattern_type.CUP_WITH_HANDLE.value,
            ]

            bearish_patterns = [
                Advanced_pattern_type.THREE_BLACK_CROWS.value,
                Advanced_pattern_type.THREE_INSIDE_DOWN.value,
                Advanced_pattern_type.THREE_OUTSIDE_DOWN.value,
                Advanced_pattern_type.FALLING_THREE_METHODS.value,
                Advanced_pattern_type.TOWER_TOP.value,
                Advanced_pattern_type.HEAD_SHOULDERS_TOP.value,
                Advanced_pattern_type.DOUBLE_TOP.value,
                Advanced_pattern_type.TRIPLE_TOP.value,
                Advanced_pattern_type.DIAMOND_TOP.value,
            ]

            # 创建看涨和看跌信号序列
            bullish_signal = pd.Series(False, index=indicator_values.index)
            for pattern in bullish_patterns:
                if pattern in indicator_values.columns:
                    bullish_signal |= indicator_values[pattern]

            bearish_signal = pd.Series(False, index=indicator_values.index)
            for pattern in bearish_patterns:
                if pattern in indicator_values.columns:
                    bearish_signal |= indicator_values[pattern]

            # 假设indicator_values中包含移动平均线等趋势指标
            # 这里可以添加与其他指标的集成逻辑

            # 简单规则：如果形态发生在合适的价格位置，则认为趋势确认
            # 实际应用中，可以与移动平均线、趋势线等结合使用

            # 当前仅使用信号强度作为趋势确认的简单方法
            if "signal_strength" in indicator_values.columns:
                signal_strength = indicator_values["signal_strength"]

                # 信号强度大于70时认为趋势确认
                trend_confirmed = (bullish_signal & (signal_strength > 70)) | (  # TODO: 将魔法数字提取到配置中
                    bearish_signal & (signal_strength > 70)
                )  # TODO: 将魔法数字提取到配置中
            else:
                # 没有信号强度时，使用形态本身的存在作为确认
                trend_confirmed = bullish_signal | bearish_signal

        except Exception as e:
            logger.error(f"计算趋势确认信号时出错: {e}")
            # 出错时返回未确认
            trend_confirmed = pd.Series(False, index=indicator_values.index)

        return trend_confirmed

    def _calculate_compound_signal(self, indicator_values: pd.DataFrame) -> pd.Series:
        """
        计算复合信号，识别多种形态同时出现的情况

        Args:
            indicator_values: 指标计算结果，包含各种K线形态的标记

        Returns:
            pd.Series: 复合信号序列，值越大表示信号越强
        """
        # 防止数据为空
        if indicator_values is None or len(indicator_values) == 0:
            return pd.Series(0, index=indicator_values.index if indicator_values is not None else [])

        # 初始化复合信号序列
        compound_signal = pd.Series(0, index=indicator_values.index)

        try:
            # 定义所有形态列表
            all_patterns = [
                # 高级形态
                Advanced_pattern_type.THREE_WHITE_SOLDIERS.value,
                Advanced_pattern_type.THREE_BLACK_CROWS.value,
                Advanced_pattern_type.THREE_INSIDE_UP.value,
                Advanced_pattern_type.THREE_INSIDE_DOWN.value,
                Advanced_pattern_type.THREE_OUTSIDE_UP.value,
                Advanced_pattern_type.THREE_OUTSIDE_DOWN.value,
                Advanced_pattern_type.RISING_THREE_METHODS.value,
                Advanced_pattern_type.FALLING_THREE_METHODS.value,
                Advanced_pattern_type.MAT_HOLD.value,
                Advanced_pattern_type.STICK_SANDWICH.value,
                Advanced_pattern_type.LADDER_BOTTOM.value,
                Advanced_pattern_type.TOWER_TOP.value,
                Advanced_pattern_type.BREAKAWAY.value,
                Advanced_pattern_type.KICKING.value,
                Advanced_pattern_type.UNIQUE_THREE_RIVER.value,
                # 复杂形态
                Advanced_pattern_type.HEAD_SHOULDERS_TOP.value,
                Advanced_pattern_type.HEAD_SHOULDERS_BOTTOM.value,
                Advanced_pattern_type.DOUBLE_TOP.value,
                Advanced_pattern_type.DOUBLE_BOTTOM.value,
                Advanced_pattern_type.TRIPLE_TOP.value,
                Advanced_pattern_type.TRIPLE_BOTTOM.value,
                Advanced_pattern_type.TRIANGLE_ASCENDING.value,
                Advanced_pattern_type.TRIANGLE_DESCENDING.value,
                Advanced_pattern_type.TRIANGLE_SYMMETRICAL.value,
                Advanced_pattern_type.RECTANGLE.value,
                Advanced_pattern_type.DIAMOND_TOP.value,
                Advanced_pattern_type.DIAMOND_BOTTOM.value,
                Advanced_pattern_type.CUP_WITH_HANDLE.value,
            ]

            # 计算每行有多少个形态同时出现
            patterns_count = pd.Series(0, index=indicator_values.index)

            for pattern in all_patterns:
                if pattern in indicator_values.columns:
                    patterns_count += indicator_values[pattern].astype(int)

            # 设置复合信号的强度
            # 1个形态：信号强度为1
            # 2个形态：信号强度为3
            # 3个或更多形态：信号强度为5
            compound_signal = pd.Series(0, index=indicator_values.index)
            compound_signal = compound_signal.mask(patterns_count == 1, 1)
            compound_signal = compound_signal.mask(patterns_count == 2, 3)  # TODO: 将魔法数字提取到配置中
            compound_signal = compound_signal.mask(
                patterns_count >= 3, 5
            )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

            # 查看是否有冲突信号（同时出现看涨和看跌形态）
            bullish_patterns = [
                Advanced_pattern_type.THREE_WHITE_SOLDIERS.value,
                Advanced_pattern_type.THREE_INSIDE_UP.value,
                Advanced_pattern_type.THREE_OUTSIDE_UP.value,
                Advanced_pattern_type.RISING_THREE_METHODS.value,
                Advanced_pattern_type.MAT_HOLD.value,
                Advanced_pattern_type.LADDER_BOTTOM.value,
                Advanced_pattern_type.BREAKAWAY.value,
                Advanced_pattern_type.HEAD_SHOULDERS_BOTTOM.value,
                Advanced_pattern_type.DOUBLE_BOTTOM.value,
                Advanced_pattern_type.TRIPLE_BOTTOM.value,
                Advanced_pattern_type.DIAMOND_BOTTOM.value,
                Advanced_pattern_type.CUP_WITH_HANDLE.value,
            ]

            bearish_patterns = [
                Advanced_pattern_type.THREE_BLACK_CROWS.value,
                Advanced_pattern_type.THREE_INSIDE_DOWN.value,
                Advanced_pattern_type.THREE_OUTSIDE_DOWN.value,
                Advanced_pattern_type.FALLING_THREE_METHODS.value,
                Advanced_pattern_type.TOWER_TOP.value,
                Advanced_pattern_type.HEAD_SHOULDERS_TOP.value,
                Advanced_pattern_type.DOUBLE_TOP.value,
                Advanced_pattern_type.TRIPLE_TOP.value,
                Advanced_pattern_type.DIAMOND_TOP.value,
            ]

            # 统计看涨形态数量
            bullish_count = pd.Series(0, index=indicator_values.index)
            for pattern in bullish_patterns:
                if pattern in indicator_values.columns:
                    bullish_count += indicator_values[pattern].astype(int)

            # 统计看跌形态数量
            bearish_count = pd.Series(0, index=indicator_values.index)
            for pattern in bearish_patterns:
                if pattern in indicator_values.columns:
                    bearish_count += indicator_values[pattern].astype(int)

            # 当看涨和看跌形态同时出现时，降低复合信号强度
            conflict_mask = (bullish_count > 0) & (bearish_count > 0)
            compound_signal = compound_signal.mask(conflict_mask, compound_signal / 2)

            # 考虑信号强度因素
            if "signal_strength" in indicator_values.columns:
                # 信号强度高的位置，提升复合信号强度
                signal_strength = indicator_values["signal_strength"]
                compound_signal = compound_signal * (1 + signal_strength / 100)

            # 规范化到[0, 10]范围
            compound_signal = compound_signal.clip(0, 10)

        except Exception as e:
            logger.error(f"计算复合信号时出错: {e}")
            # 出错时返回零信号
            compound_signal = pd.Series(0, index=indicator_values.index)

        return compound_signal

    def calculate_raw_score_Patterns_Advanced_Candlestick_Patterns(self, data: pd.DataFrame) -> pd.Series:
        """
        计算高级K线形态识别指标的原始评分

        Args:
            data: 包含OHLCV数据的Data_frame

        Returns:
            pd.DataFrame: 包含原始评分的Data_frame
        """
        # 计算指标值
        indicator_data = self.calculate(data)

        # 初始化评分
        score = pd.Series(50.0, index=data.index)  # 基础分50分  # TODO: 将魔法数字提取到配置中

        # 1. 强烈看涨形态评分（+25到+40分）
        # 三星看涨形态
        if Advanced_pattern_type.THREE_WHITE_SOLDIERS.value in indicator_data.columns:
            three_white_soldiers_mask = indicator_data[Advanced_pattern_type.THREE_WHITE_SOLDIERS.value]
            score.loc[three_white_soldiers_mask] += 35  # TODO: 将魔法数字提取到配置中

        if Advanced_pattern_type.THREE_INSIDE_UP.value in indicator_data.columns:
            three_inside_up_mask = indicator_data[Advanced_pattern_type.THREE_INSIDE_UP.value]
            score.loc[three_inside_up_mask] += 30  # TODO: 将魔法数字提取到配置中

        if Advanced_pattern_type.THREE_OUTSIDE_UP.value in indicator_data.columns:
            three_outside_up_mask = indicator_data[Advanced_pattern_type.THREE_OUTSIDE_UP.value]
            score.loc[three_outside_up_mask] += 32  # TODO: 将魔法数字提取到配置中

        # 高级复合看涨形态
        if Advanced_pattern_type.RISING_THREE_METHODS.value in indicator_data.columns:
            rising_three_methods_mask = indicator_data[Advanced_pattern_type.RISING_THREE_METHODS.value]
            score.loc[rising_three_methods_mask] += 28  # TODO: 将魔法数字提取到配置中

        if Advanced_pattern_type.MAT_HOLD.value in indicator_data.columns:
            mat_hold_mask = indicator_data[Advanced_pattern_type.MAT_HOLD.value]
            score.loc[mat_hold_mask] += 25  # TODO: 将魔法数字提取到配置中

        if Advanced_pattern_type.LADDER_BOTTOM.value in indicator_data.columns:
            ladder_bottom_mask = indicator_data[Advanced_pattern_type.LADDER_BOTTOM.value]
            score.loc[ladder_bottom_mask] += 30  # TODO: 将魔法数字提取到配置中

        if Advanced_pattern_type.BREAKAWAY.value in indicator_data.columns:
            breakaway_mask = indicator_data[Advanced_pattern_type.BREAKAWAY.value]
            # 需要判断突破方向，这里假设是看涨突破
            score.loc[breakaway_mask] += 25  # TODO: 将魔法数字提取到配置中

        # 复杂看涨形态
        if Advanced_pattern_type.HEAD_SHOULDERS_BOTTOM.value in indicator_data.columns:
            head_shoulders_bottom_mask = indicator_data[Advanced_pattern_type.HEAD_SHOULDERS_BOTTOM.value]
            score.loc[head_shoulders_bottom_mask] += 40  # TODO: 将魔法数字提取到配置中

        if Advanced_pattern_type.DOUBLE_BOTTOM.value in indicator_data.columns:
            double_bottom_mask = indicator_data[Advanced_pattern_type.DOUBLE_BOTTOM.value]
            score.loc[double_bottom_mask] += 35  # TODO: 将魔法数字提取到配置中

        if Advanced_pattern_type.TRIPLE_BOTTOM.value in indicator_data.columns:
            triple_bottom_mask = indicator_data[Advanced_pattern_type.TRIPLE_BOTTOM.value]
            score.loc[triple_bottom_mask] += 38  # TODO: 将魔法数字提取到配置中

        if Advanced_pattern_type.DIAMOND_BOTTOM.value in indicator_data.columns:
            diamond_bottom_mask = indicator_data[Advanced_pattern_type.DIAMOND_BOTTOM.value]
            score.loc[diamond_bottom_mask] += 35  # TODO: 将魔法数字提取到配置中

        if Advanced_pattern_type.CUP_WITH_HANDLE.value in indicator_data.columns:
            cup_handle_mask = indicator_data[Advanced_pattern_type.CUP_WITH_HANDLE.value]
            score.loc[cup_handle_mask] += 32  # TODO: 将魔法数字提取到配置中

        # 2. 强烈看跌形态评分（-25到-40分）
        # 三星看跌形态
        if Advanced_pattern_type.THREE_BLACK_CROWS.value in indicator_data.columns:
            three_black_crows_mask = indicator_data[Advanced_pattern_type.THREE_BLACK_CROWS.value]
            score.loc[three_black_crows_mask] -= 35  # TODO: 将魔法数字提取到配置中

        if Advanced_pattern_type.THREE_INSIDE_DOWN.value in indicator_data.columns:
            three_inside_down_mask = indicator_data[Advanced_pattern_type.THREE_INSIDE_DOWN.value]
            score.loc[three_inside_down_mask] -= 30  # TODO: 将魔法数字提取到配置中

        if Advanced_pattern_type.THREE_OUTSIDE_DOWN.value in indicator_data.columns:
            three_outside_down_mask = indicator_data[Advanced_pattern_type.THREE_OUTSIDE_DOWN.value]
            score.loc[three_outside_down_mask] -= 32  # TODO: 将魔法数字提取到配置中

        # 高级复合看跌形态
        if Advanced_pattern_type.FALLING_THREE_METHODS.value in indicator_data.columns:
            falling_three_methods_mask = indicator_data[Advanced_pattern_type.FALLING_THREE_METHODS.value]
            score.loc[falling_three_methods_mask] -= 28  # TODO: 将魔法数字提取到配置中

        if Advanced_pattern_type.TOWER_TOP.value in indicator_data.columns:
            tower_top_mask = indicator_data[Advanced_pattern_type.TOWER_TOP.value]
            score.loc[tower_top_mask] -= 30  # TODO: 将魔法数字提取到配置中

        # 复杂看跌形态
        if Advanced_pattern_type.HEAD_SHOULDERS_TOP.value in indicator_data.columns:
            head_shoulders_top_mask = indicator_data[Advanced_pattern_type.HEAD_SHOULDERS_TOP.value]
            score.loc[head_shoulders_top_mask] -= 40  # TODO: 将魔法数字提取到配置中

        if Advanced_pattern_type.DOUBLE_TOP.value in indicator_data.columns:
            double_top_mask = indicator_data[Advanced_pattern_type.DOUBLE_TOP.value]
            score.loc[double_top_mask] -= 35  # TODO: 将魔法数字提取到配置中

        if Advanced_pattern_type.TRIPLE_TOP.value in indicator_data.columns:
            triple_top_mask = indicator_data[Advanced_pattern_type.TRIPLE_TOP.value]
            score.loc[triple_top_mask] -= 38  # TODO: 将魔法数字提取到配置中

        if Advanced_pattern_type.DIAMOND_TOP.value in indicator_data.columns:
            diamond_top_mask = indicator_data[Advanced_pattern_type.DIAMOND_TOP.value]
            score.loc[diamond_top_mask] -= 35  # TODO: 将魔法数字提取到配置中

        # 3. 中性/整理形态评分（-10到+10分）  # TODO: 将魔法数字提取到配置中
        if Advanced_pattern_type.STICK_SANDWICH.value in indicator_data.columns:
            stick_sandwich_mask = indicator_data[Advanced_pattern_type.STICK_SANDWICH.value]
            score.loc[stick_sandwich_mask] += 5  # 轻微看涨倾向  # TODO: 将魔法数字提取到配置中

        if Advanced_pattern_type.KICKING.value in indicator_data.columns:
            kicking_mask = indicator_data[Advanced_pattern_type.KICKING.value]
            # 反冲形态需要判断方向，这里给中性评分
            score.loc[kicking_mask] += 0

        if Advanced_pattern_type.UNIQUE_THREE_RIVER.value in indicator_data.columns:
            unique_three_river_mask = indicator_data[Advanced_pattern_type.UNIQUE_THREE_RIVER.value]
            score.loc[unique_three_river_mask] += 15  # 底部反转形态  # TODO: 将魔法数字提取到配置中

        # 三角形整理形态
        if Advanced_pattern_type.TRIANGLE_ASCENDING.value in indicator_data.columns:
            triangle_ascending_mask = indicator_data[Advanced_pattern_type.TRIANGLE_ASCENDING.value]
            score.loc[triangle_ascending_mask] += 8  # 轻微看涨倾向  # TODO: 将魔法数字提取到配置中

        if Advanced_pattern_type.TRIANGLE_DESCENDING.value in indicator_data.columns:
            triangle_descending_mask = indicator_data[Advanced_pattern_type.TRIANGLE_DESCENDING.value]
            score.loc[triangle_descending_mask] -= 8  # 轻微看跌倾向  # TODO: 将魔法数字提取到配置中

        if Advanced_pattern_type.TRIANGLE_SYMMETRICAL.value in indicator_data.columns:
            triangle_symmetrical_mask = indicator_data[Advanced_pattern_type.TRIANGLE_SYMMETRICAL.value]
            score.loc[triangle_symmetrical_mask] += 0  # 中性

        if Advanced_pattern_type.RECTANGLE.value in indicator_data.columns:
            rectangle_mask = indicator_data[Advanced_pattern_type.RECTANGLE.value]
            score.loc[rectangle_mask] += 0  # 中性整理

        # 4. 形态强度调整（±15分）  # TODO: 将魔法数字提取到配置中
        # 根据成交量确认形态强度
        if "volume" in data.columns:
            volume = data["volume"]
            vol_ma5 = volume.rolling(window=5).mean()  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            vol_ratio = volume / vol_ma5

            # 任何形态如果伴随放量，增强信号强度
            high_volume_mask = vol_ratio > 1.5  # TODO: 将魔法数字提取到配置中

            # 看涨形态+放量
            bullish_patterns = (
                indicator_data.get(Advanced_pattern_type.THREE_WHITE_SOLDIERS.value, False)
                | indicator_data.get(Advanced_pattern_type.HEAD_SHOULDERS_BOTTOM.value, False)
                | indicator_data.get(Advanced_pattern_type.DOUBLE_BOTTOM.value, False)
                | indicator_data.get(Advanced_pattern_type.CUP_WITH_HANDLE.value, False)
            )
            if isinstance(bullish_patterns, pd.Series):
                bullish_volume_confirm = bullish_patterns & high_volume_mask
                score.loc[bullish_volume_confirm] += 15  # TODO: 将魔法数字提取到配置中

            # 看跌形态+放量
            bearish_patterns = (
                indicator_data.get(Advanced_pattern_type.THREE_BLACK_CROWS.value, False)
                | indicator_data.get(Advanced_pattern_type.HEAD_SHOULDERS_TOP.value, False)
                | indicator_data.get(Advanced_pattern_type.DOUBLE_TOP.value, False)
                | indicator_data.get(Advanced_pattern_type.TOWER_TOP.value, False)
            )
            if isinstance(bearish_patterns, pd.Series):
                bearish_volume_confirm = bearish_patterns & high_volume_mask
                score.loc[bearish_volume_confirm] -= 15  # TODO: 将魔法数字提取到配置中

        # 5. 形态完整性调整（±10分）  # TODO: 将魔法数字提取到配置中
        # 检查形态的完整性和质量
        # 这里可以添加更复杂的形态质量评估逻辑

        # 6. 多重形态确认（±20分）  # TODO: 将魔法数字提取到配置中
        # 检查是否有多个形态同时出现
        pattern_count = 0
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0  # 添加neutral_count初始化

        # 统计当前时点的形态数量
        for pattern_type in Advanced_pattern_type:
            pattern_name = pattern_type.value
            if pattern_name in indicator_data.columns:
                current_pattern = indicator_data[pattern_name]
                if isinstance(current_pattern, pd.Series):
                    pattern_count += current_pattern.astype(int)

                    # 分类统计
                    if pattern_type in [
                        Advanced_pattern_type.THREE_WHITE_SOLDIERS,
                        Advanced_pattern_type.THREE_INSIDE_UP,
                        Advanced_pattern_type.THREE_OUTSIDE_UP,
                        Advanced_pattern_type.RISING_THREE_METHODS,
                        Advanced_pattern_type.MAT_HOLD,
                        Advanced_pattern_type.LADDER_BOTTOM,
                        Advanced_pattern_type.HEAD_SHOULDERS_BOTTOM,
                        Advanced_pattern_type.DOUBLE_BOTTOM,
                        Advanced_pattern_type.TRIPLE_BOTTOM,
                        Advanced_pattern_type.DIAMOND_BOTTOM,
                        Advanced_pattern_type.CUP_WITH_HANDLE,
                        Advanced_pattern_type.UNIQUE_THREE_RIVER,
                    ]:
                        bullish_count += 1
                    elif pattern_type in [
                        Advanced_pattern_type.THREE_BLACK_CROWS,
                        Advanced_pattern_type.THREE_INSIDE_DOWN,
                        Advanced_pattern_type.THREE_OUTSIDE_DOWN,
                        Advanced_pattern_type.FALLING_THREE_METHODS,
                        Advanced_pattern_type.TOWER_TOP,
                        Advanced_pattern_type.HEAD_SHOULDERS_TOP,
                        Advanced_pattern_type.DOUBLE_TOP,
                        Advanced_pattern_type.TRIPLE_TOP,
                        Advanced_pattern_type.DIAMOND_TOP,
                    ]:
                        bearish_count += 1
                    else:
                        neutral_count += 1

        # 多重看涨形态确认
        if isinstance(bullish_count, pd.Series):
            multiple_bullish = bullish_count >= 2
            score.loc[multiple_bullish] += 20  # TODO: 将魔法数字提取到配置中

        # 多重看跌形态确认
        if isinstance(bearish_count, pd.Series):
            multiple_bearish = bearish_count >= 2
            score.loc[multiple_bearish] -= 20  # TODO: 将魔法数字提取到配置中

        # 形态冲突（同时出现看涨看跌形态）
        if isinstance(bullish_count, pd.Series) and isinstance(bearish_count, pd.Series):
            conflict_patterns = (bullish_count > 0) & (bearish_count > 0)
            score.loc[conflict_patterns] -= 10  # 冲突信号减分

        # 7. 形态位置调整（±15分）  # TODO: 将魔法数字提取到配置中
        # 在关键技术位置的形态更重要
        if "close" in data.columns and len(data) >= 60:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            close_price = data["close"]

            # 计算支撑阻力位
            high_60 = close_price.rolling(window=60).max()  # TODO: 将魔法数字提取到配置中
            low_60 = close_price.rolling(window=60).min()  # TODO: 将魔法数字提取到配置中

            # 在阻力位附近的看跌形态
            near_resistance = close_price > high_60 * 0.95  # TODO: 将魔法数字提取到配置中
            bearish_at_resistance = (
                indicator_data.get(Advanced_pattern_type.THREE_BLACK_CROWS.value, False)
                | indicator_data.get(Advanced_pattern_type.HEAD_SHOULDERS_TOP.value, False)
                | indicator_data.get(Advanced_pattern_type.DOUBLE_TOP.value, False)
            ) & near_resistance
            if isinstance(bearish_at_resistance, pd.Series):
                score.loc[bearish_at_resistance] -= 15  # TODO: 将魔法数字提取到配置中

            # 在支撑位附近的看涨形态
            near_support = close_price < low_60 * 1.05  # TODO: 将魔法数字提取到配置中
            bullish_at_support = (
                indicator_data.get(Advanced_pattern_type.THREE_WHITE_SOLDIERS.value, False)
                | indicator_data.get(Advanced_pattern_type.HEAD_SHOULDERS_BOTTOM.value, False)
                | indicator_data.get(Advanced_pattern_type.DOUBLE_BOTTOM.value, False)
            ) & near_support
            if isinstance(bullish_at_support, pd.Series):
                score.loc[bullish_at_support] += 15  # TODO: 将魔法数字提取到配置中

        # 确保评分在0-100范围内
        score = score.clip(0, 100)
        score.name = "raw_score"

        return score

    def identify_patterns_Patterns_Advanced_Candlestick_Patterns(self, data: pd.DataFrame) -> List[str]:
        """
        识别高级K线形态相关的技术形态

        Args:
            data: 包含OHLCV数据的Data_frame

        Returns:
            List[str]: 识别出的形态列表
        """
        patterns = []

        # 计算指标值
        indicator_data = self.calculate(data)

        if len(indicator_data) < 5:  # TODO: 将魔法数字提取到配置中
            return patterns

        # 检查最近5天的形态
        recent_data = indicator_data.tail(5)  # TODO: 将魔法数字提取到配置中

        # 1. 三星形态
        three_star_patterns = [
            Advanced_pattern_type.THREE_WHITE_SOLDIERS,
            Advanced_pattern_type.THREE_BLACK_CROWS,
            Advanced_pattern_type.THREE_INSIDE_UP,
            Advanced_pattern_type.THREE_INSIDE_DOWN,
            Advanced_pattern_type.THREE_OUTSIDE_UP,
            Advanced_pattern_type.THREE_OUTSIDE_DOWN,
        ]

        for pattern_type in three_star_patterns:
            pattern_name = pattern_type.value
            if pattern_name in recent_data.columns and recent_data[pattern_name].any():
                patterns.append(f"三星形态-{pattern_name}")

        # 2. 高级复合形态
        advanced_compound_patterns = [
            Advanced_pattern_type.RISING_THREE_METHODS,
            Advanced_pattern_type.FALLING_THREE_METHODS,
            Advanced_pattern_type.MAT_HOLD,
            Advanced_pattern_type.STICK_SANDWICH,
        ]

        for pattern_type in advanced_compound_patterns:
            pattern_name = pattern_type.value
            if pattern_name in recent_data.columns and recent_data[pattern_name].any():
                patterns.append(f"高级复合形态-{pattern_name}")

        # 3. 其他复合形态  # TODO: 将魔法数字提取到配置中
        other_compound_patterns = [
            Advanced_pattern_type.LADDER_BOTTOM,
            Advanced_pattern_type.TOWER_TOP,
            Advanced_pattern_type.BREAKAWAY,
            Advanced_pattern_type.KICKING,
            Advanced_pattern_type.UNIQUE_THREE_RIVER,
        ]

        for pattern_type in other_compound_patterns:
            pattern_name = pattern_type.value
            if pattern_name in recent_data.columns and recent_data[pattern_name].any():
                patterns.append(f"其他复合形态-{pattern_name}")

        # 4. 复杂形态  # TODO: 将魔法数字提取到配置中
        complex_patterns = [
            Advanced_pattern_type.HEAD_SHOULDERS_TOP,
            Advanced_pattern_type.HEAD_SHOULDERS_BOTTOM,
            Advanced_pattern_type.DOUBLE_TOP,
            Advanced_pattern_type.DOUBLE_BOTTOM,
            Advanced_pattern_type.TRIPLE_TOP,
            Advanced_pattern_type.TRIPLE_BOTTOM,
            Advanced_pattern_type.TRIANGLE_ASCENDING,
            Advanced_pattern_type.TRIANGLE_DESCENDING,
            Advanced_pattern_type.TRIANGLE_SYMMETRICAL,
            Advanced_pattern_type.RECTANGLE,
            Advanced_pattern_type.DIAMOND_TOP,
            Advanced_pattern_type.DIAMOND_BOTTOM,
            Advanced_pattern_type.CUP_WITH_HANDLE,
        ]

        for pattern_type in complex_patterns:
            pattern_name = pattern_type.value
            if pattern_name in recent_data.columns and recent_data[pattern_name].any():
                patterns.append(f"复杂形态-{pattern_name}")

        # 5. 形态强度分析  # TODO: 将魔法数字提取到配置中
        if "volume" in data.columns:
            volume = data["volume"]
            vol_ma5 = volume.rolling(window=5).mean()  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            latest_vol_ratio = (volume / vol_ma5).iloc[-1]

            if pd.notna(latest_vol_ratio):
                if latest_vol_ratio > 2.0:
                    patterns.append("形态确认-巨量配合")
                elif latest_vol_ratio > 1.5:  # TODO: 将魔法数字提取到配置中
                    patterns.append("形态确认-放量配合")
                elif latest_vol_ratio < 0.7:  # TODO: 将魔法数字提取到配置中
                    patterns.append("形态确认-缩量形成")

        # 6. 形态组合分析  # TODO: 将魔法数字提取到配置中
        # 统计不同类型形态的数量
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        total_patterns = 0

        for pattern_type in Advanced_pattern_type:
            pattern_name = pattern_type.value
            if pattern_name in recent_data.columns and recent_data[pattern_name].any():
                total_patterns += 1

                # 分类统计
                if pattern_type in [
                    Advanced_pattern_type.THREE_WHITE_SOLDIERS,
                    Advanced_pattern_type.THREE_INSIDE_UP,
                    Advanced_pattern_type.THREE_OUTSIDE_UP,
                    Advanced_pattern_type.RISING_THREE_METHODS,
                    Advanced_pattern_type.MAT_HOLD,
                    Advanced_pattern_type.LADDER_BOTTOM,
                    Advanced_pattern_type.HEAD_SHOULDERS_BOTTOM,
                    Advanced_pattern_type.DOUBLE_BOTTOM,
                    Advanced_pattern_type.TRIPLE_BOTTOM,
                    Advanced_pattern_type.DIAMOND_BOTTOM,
                    Advanced_pattern_type.CUP_WITH_HANDLE,
                    Advanced_pattern_type.UNIQUE_THREE_RIVER,
                ]:
                    bullish_count += 1
                elif pattern_type in [
                    Advanced_pattern_type.THREE_BLACK_CROWS,
                    Advanced_pattern_type.THREE_INSIDE_DOWN,
                    Advanced_pattern_type.THREE_OUTSIDE_DOWN,
                    Advanced_pattern_type.FALLING_THREE_METHODS,
                    Advanced_pattern_type.TOWER_TOP,
                    Advanced_pattern_type.HEAD_SHOULDERS_TOP,
                    Advanced_pattern_type.DOUBLE_TOP,
                    Advanced_pattern_type.TRIPLE_TOP,
                    Advanced_pattern_type.DIAMOND_TOP,
                ]:
                    bearish_count += 1
                else:
                    neutral_count += 1

        # 形态组合描述
        if total_patterns > 1:
            patterns.append(f"形态组合-{total_patterns}个高级形态同现")

        if bullish_count > bearish_count and bullish_count >= 2:
            patterns.append("高级形态共振-多重看涨信号")
        elif bearish_count > bullish_count and bearish_count >= 2:
            patterns.append("高级形态共振-多重看跌信号")
        elif bullish_count > 0 and bearish_count > 0:
            patterns.append("高级形态冲突-多空信号混杂")

        # 7. 形态复杂度分析  # TODO: 将魔法数字提取到配置中
        if total_patterns >= 3:  # TODO: 将魔法数字提取到配置中
            patterns.append("高复杂度形态组合")
        elif total_patterns == 2:
            patterns.append("中等复杂度形态组合")
        elif total_patterns == 1:
            patterns.append("单一高级形态")

        # 8. 形态时效性分析  # TODO: 将魔法数字提取到配置中
        # 检查形态是否在最近1-2天内形成
        very_recent_data = indicator_data.tail(2)
        recent_pattern_count = 0

        for pattern_type in Advanced_pattern_type:
            pattern_name = pattern_type.value
            if pattern_name in very_recent_data.columns and very_recent_data[pattern_name].any():
                recent_pattern_count += 1

        if recent_pattern_count > 0:
            patterns.append(f"新形成形态-{recent_pattern_count}个")

        # 9. 形态位置分析  # TODO: 将魔法数字提取到配置中
        if "close" in data.columns and len(data) >= 60:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            close_price = data["close"]
            high_60 = close_price.rolling(window=60).max()  # TODO: 将魔法数字提取到配置中.iloc[-1]
            low_60 = close_price.rolling(window=60).min()  # TODO: 将魔法数字提取到配置中.iloc[-1]
            latest_close = close_price.iloc[-1]

            if pd.notna(latest_close) and pd.notna(high_60) and pd.notna(low_60):
                if latest_close > high_60 * 0.95:  # TODO: 将魔法数字提取到配置中
                    patterns.append("高级形态位置-接近阻力位")
                elif latest_close < low_60 * 1.05:  # TODO: 将魔法数字提取到配置中
                    patterns.append("高级形态位置-接近支撑位")
                else:
                    price_position = (latest_close - low_60) / (high_60 - low_60)
                    if price_position > 0.7:  # TODO: 将魔法数字提取到配置中
                        patterns.append("高级形态位置-相对高位")
                    elif price_position < 0.3:  # TODO: 将魔法数字提取到配置中
                        patterns.append("高级形态位置-相对低位")
                    else:
                        patterns.append("高级形态位置-中性区域")

        return patterns

    def get_patterns_Patterns_Advanced_Candlestick_Patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取Advanced_candlestick_patterns相关形态

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含形态信息的Data_frame
        """
        # 确保已计算指标
        if not self.has_result():
            self._calculate_advancedcandlestickpatterns(data, **kwargs)

        # 如果没有计算结果，返回空DataFrame
        if self._result is None or self._result.empty:
            return pd.DataFrame(index=data.index)

        # 返回计算结果，因为_calculate现在只包含形态列
        return self._result

    def calculate_confidence_Patterns_Advanced_Candlestick_Patterns(
        self, score: pd.Series, patterns: pd.DataFrame, signals: dict
    ) -> float:
        """
        计算Advanced_candlestick_patterns指标的置信度

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
            # 检查是否有高级形态数据
            advanced_pattern_columns = [pattern.value for pattern in Advanced_pattern_type]
            available_patterns = [col for col in advanced_pattern_columns if col in self._result.columns]
            if available_patterns:
                # 高级形态数据越完整，置信度越高
                data_completeness = len(available_patterns) / len(Advanced_pattern_type)
                confidence += data_completeness * 0.1

        # 3. 基于形态的置信度  # TODO: 将魔法数字提取到配置中
        if not patterns.empty:
            # 检查AdvancedCandlestickPatterns形态（只计算布尔列）
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

    def register_patterns_Patterns_Advanced_Candlestick_Patterns(self):
        """
        注册Advanced_candlestick_patterns指标的形态到全局形态注册表
        """
        # 注册三星形态
        self.register_pattern_to_registry(
            pattern_id="THREE_WHITE_SOLDIERS",
            display_name="三白兵",
            description="连续三根阳线，每根都收于接近最高点，强烈的上涨信号",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=35.0,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            polarity="POSITIVE",
        )

        self.register_pattern_to_registry(
            pattern_id="THREE_BLACK_CROWS",
            display_name="三黑鸦",
            description="连续三根阴线，每根都收于接近最低点，强烈的下跌信号",
            pattern_type="BEARISH",
            default_strength="VERY_STRONG",
            score_impact=-35.0,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            polarity="NEGATIVE",
        )

        self.register_pattern_to_registry(
            pattern_id="THREE_INSIDE_UP",
            display_name="三内涨",
            description="大阴线+小阳线在阴线实体内+突破阴线收盘价的阳线",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=30.0,  # TODO: 将魔法数字提取到配置中
            polarity="POSITIVE",
        )

        self.register_pattern_to_registry(
            pattern_id="THREE_INSIDE_DOWN",
            display_name="三内跌",
            description="大阳线+小阴线在阳线实体内+突破阳线收盘价的阴线",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-30.0,  # TODO: 将魔法数字提取到配置中
            polarity="NEGATIVE",
        )

        # 注册高级复合形态
        self.register_pattern_to_registry(
            pattern_id="RISING_THREE_METHODS",
            display_name="上升三法",
            description="大阳线后三根小K线在大阳线范围内整理，然后一根突破的阳线",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=28.0,  # TODO: 将魔法数字提取到配置中
            polarity="POSITIVE",
        )

        self.register_pattern_to_registry(
            pattern_id="FALLING_THREE_METHODS",
            display_name="下降三法",
            description="大阴线后三根小K线在大阴线范围内整理，然后一根突破的阴线",
            pattern_type="BEARISH",
            default_strength="VERY_STRONG",
            score_impact=-28.0,  # TODO: 将魔法数字提取到配置中
            polarity="NEGATIVE",
        )

        self.register_pattern_to_registry(
            pattern_id="MAT_HOLD",
            display_name="铺垫形态",
            description="大阳线后2-3根小阴线在大阳线上部整理，然后一根大阳线",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=25.0,  # TODO: 将魔法数字提取到配置中
            polarity="POSITIVE",
        )

        # 注册复杂形态
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
            pattern_id="HEAD_SHOULDERS_BOTTOM",
            display_name="头肩底",
            description="三个波谷，中间低于两侧，强烈的底部反转形态",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=40.0,  # TODO: 将魔法数字提取到配置中
            polarity="POSITIVE",
        )

        self.register_pattern_to_registry(
            pattern_id="DOUBLE_TOP",
            display_name="双顶",
            description="两个相近高点的顶部反转形态",
            pattern_type="BEARISH",
            default_strength="VERY_STRONG",
            score_impact=-35.0,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            polarity="NEGATIVE",
        )

        self.register_pattern_to_registry(
            pattern_id="DOUBLE_BOTTOM",
            display_name="双底",
            description="两个相近低点的底部反转形态",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=35.0,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            polarity="POSITIVE",
        )

        # 注册其他重要形态
        self.register_pattern_to_registry(
            pattern_id="BREAKAWAY",
            display_name="脱离形态",
            description="五根K线组成的反转形态，突破性强，方向需结合趋势判断",
            pattern_type="NEUTRAL",
            default_strength="STRONG",
            score_impact=0.0,
            polarity="NEUTRAL",
        )

        self.register_pattern_to_registry(
            pattern_id="KICKING",
            display_name="反冲形态",
            description="两根相反方向的光头光脚K线，反转信号强烈，方向需结合趋势判断",
            pattern_type="NEUTRAL",
            default_strength="VERY_STRONG",
            score_impact=0.0,
            polarity="NEUTRAL",
        )

        self.register_pattern_to_registry(
            pattern_id="TRIANGLE_ASCENDING",
            display_name="上升三角形",
            description="水平上轨+上升下轨的整理形态，通常向上突破",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=8.0,  # TODO: 将魔法数字提取到配置中
            polarity="POSITIVE",
        )

        self.register_pattern_to_registry(
            pattern_id="TRIANGLE_DESCENDING",
            display_name="下降三角形",
            description="下降上轨+水平下轨的整理形态，通常向下突破",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-8.0,  # TODO: 将魔法数字提取到配置中
            polarity="NEGATIVE",
        )

        self.register_pattern_to_registry(
            pattern_id="CUP_WITH_HANDLE",
            display_name="杯柄形态",
            description="U形底部+小幅回调形成柄部，长期看涨形态",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=32.0,  # TODO: 将魔法数字提取到配置中
            polarity="POSITIVE",
        )

    def generate_trading_signals_Patterns_Advanced_Candlestick_Patterns(self, data: pd.DataFrame, **kwargs) -> dict:
        """
        生成Advanced_candlestick_patterns交易信号

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            dict: 包含买卖信号的字典
        """
        # 确保已计算指标
        if not self.has_result():
            self._calculate_advancedcandlestickpatterns(data, **kwargs)

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
            Advanced_pattern_type.THREE_WHITE_SOLDIERS.value,
            Advanced_pattern_type.THREE_INSIDE_UP.value,
            Advanced_pattern_type.THREE_OUTSIDE_UP.value,
            Advanced_pattern_type.RISING_THREE_METHODS.value,
            Advanced_pattern_type.MAT_HOLD.value,
            Advanced_pattern_type.LADDER_BOTTOM.value,
            Advanced_pattern_type.HEAD_SHOULDERS_BOTTOM.value,
            Advanced_pattern_type.DOUBLE_BOTTOM.value,
            Advanced_pattern_type.TRIPLE_BOTTOM.value,
            Advanced_pattern_type.DIAMOND_BOTTOM.value,
            Advanced_pattern_type.CUP_WITH_HANDLE.value,
            Advanced_pattern_type.UNIQUE_THREE_RIVER.value,
        ]

        # 定义看跌形态
        bearish_patterns = [
            Advanced_pattern_type.THREE_BLACK_CROWS.value,
            Advanced_pattern_type.THREE_INSIDE_DOWN.value,
            Advanced_pattern_type.THREE_OUTSIDE_DOWN.value,
            Advanced_pattern_type.FALLING_THREE_METHODS.value,
            Advanced_pattern_type.TOWER_TOP.value,
            Advanced_pattern_type.HEAD_SHOULDERS_TOP.value,
            Advanced_pattern_type.DOUBLE_TOP.value,
            Advanced_pattern_type.TRIPLE_TOP.value,
            Advanced_pattern_type.DIAMOND_TOP.value,
        ]

        # 强形态权重
        strong_patterns = {
            Advanced_pattern_type.THREE_WHITE_SOLDIERS.value: 0.9,  # TODO: 将魔法数字提取到配置中
            Advanced_pattern_type.THREE_BLACK_CROWS.value: -0.9,  # TODO: 将魔法数字提取到配置中
            Advanced_pattern_type.RISING_THREE_METHODS.value: 0.85,  # TODO: 将魔法数字提取到配置中
            Advanced_pattern_type.FALLING_THREE_METHODS.value: -0.85,  # TODO: 将魔法数字提取到配置中
            Advanced_pattern_type.HEAD_SHOULDERS_BOTTOM.value: 0.9,  # TODO: 将魔法数字提取到配置中
            Advanced_pattern_type.HEAD_SHOULDERS_TOP.value: -0.9,  # TODO: 将魔法数字提取到配置中
            Advanced_pattern_type.DOUBLE_BOTTOM.value: 0.8,  # TODO: 将魔法数字提取到配置中
            Advanced_pattern_type.DOUBLE_TOP.value: -0.8,  # TODO: 将魔法数字提取到配置中
            Advanced_pattern_type.KICKING.value: 0.85,  # TODO: 将魔法数字提取到配置中
        }

        # 生成买入信号
        for pattern in bullish_patterns:
            if pattern in self._result.columns:
                pattern_mask = self._result[pattern]
                buy_signal |= pattern_mask

                # 设置信号强度
                if pattern in strong_patterns:
                    signal_strength[pattern_mask] = strong_patterns[pattern]
                else:
                    signal_strength[pattern_mask] = 0.7  # TODO: 将魔法数字提取到配置中

        # 生成卖出信号
        for pattern in bearish_patterns:
            if pattern in self._result.columns:
                pattern_mask = self._result[pattern]
                sell_signal |= pattern_mask

                # 设置信号强度
                if pattern in strong_patterns:
                    signal_strength[pattern_mask] = strong_patterns[pattern]
                else:
                    signal_strength[pattern_mask] = -0.7  # TODO: 将魔法数字提取到配置中

        # 处理特殊形态
        if AdvancedPatternType.BREAKAWAY.value in self._result.columns:
            breakaway_mask = self._result[AdvancedPatternType.BREAKAWAY.value]
            if breakaway_mask.any() and len(data) >= 5:  # TODO: 将魔法数字提取到配置中
                # 简单趋势判断
                _5d = data["close"].pct_change(5)  # TODO: 将魔法数字提取到配置中

                # 在下降趋势后的脱离形态（看涨）
                bullish_breakaway = breakaway_mask & (_5d < -0.05)  # TODO: 将魔法数字提取到配置中
                buy_signal |= bullish_breakaway
                signal_strength[bullish_breakaway] = 0.8  # TODO: 将魔法数字提取到配置中

                # 在上升趋势后的脱离形态（看跌）
                bearish_breakaway = breakaway_mask & (_5d > 0.05)  # TODO: 将魔法数字提取到配置中
                sell_signal |= bearish_breakaway
                signal_strength[bearish_breakaway] = -0.8  # TODO: 将魔法数字提取到配置中

        # 标准化信号强度
        signal_strength = signal_strength.clip(-1, 1)

        return {"buy_signal": buy_signal, "sell_signal": sell_signal, "signal_strength": signal_strength}

    def get_indicator_type_Patterns_Advanced_Candlestick_Patterns(self) -> str:
        """
        获取指标类型

        Returns:
            str: 指标类型
        """
        return "ADVANCEDCANDLESTICKPATTERNS"

    def get_pattern_info_Patterns_Advanced_Candlestick_Patterns(self, pattern_id: str) -> dict:
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
        AdvancedCandlestickPatterns指标所需的最少数据周期数

        计算逻辑：使用默认值

        Returns:
            int: 最少需要的数据周期数
        """
        return 30  # TODO: 将魔法数字提取到配置中


# ===== 单独的高级形态识别类 =====
# 为指标注册表提供单独的高级形态识别类


class HeadShoulders(AdvancedCandlestickPatterns):
    """头肩顶形态识别"""

    def __init__(
        self, period: int = 30
    ):  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        super().__init__(period=period)  # 正确传递period参数
        self.name = "HEAD_SHOULDERS"

    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        获取头肩顶形态交易信号
        
        头肩顶形态是一种经典的顶部反转形态，特征：
        - 三个高点：左肩、头部、右肩
        - 头部最高：中间的高点（头部）明显高于两侧的高点（左右肩）
        - 左右肩相近：两侧肩部高度基本相等（允许小幅差异）
        - 颈线支撑：连接两个肩部之间的低点形成颈线支撑
        - 成交量特征：左肩成交量最大，头部次之，右肩最小（递减）
        - 突破确认：价格跌破颈线并且成交量放大确认反转
        - 目标位测算：突破后下跌幅度通常等于头部到颈线的距离
        - 时间周期：形态形成通常需要较长时间（数周到数月）

        Args:
            data: 包含OHLCV数据的DataFrame

        Returns:
            Dict[str, Any]: 标准化交易信号格式
        """
        try:
            # 数据验证
            if not self._validate_signal_data(data):
                return self._get_default_signal("数据验证失败")

            # 计算头肩顶形态（如果数据不是计算结果）
            if 'head_shoulders_top' not in data.columns:
                result_data = self.calculate(data)
            else:
                result_data = data

            # 获取最新头肩顶形态信号
            latest_head_shoulders = result_data["head_shoulders_top"].iloc[-1] if "head_shoulders_top" in result_data.columns else False

            # 初始化信号参数
            signal_type = "hold"
            strength = 0.5
            confidence = 0.6
            reason = "未检测到头肩顶形态"
            metadata = {}

            # 处理头肩顶形态
            if latest_head_shoulders:
                signal_type = "sell"
                strength = 0.85  # 头肩顶是强顶部反转信号
                confidence = 0.80
                reason = "检测到头肩顶形态，强顶部反转信号"
                
                # 分析头肩顶形态的具体特征
                if len(data) >= 30:  # 头肩顶需要足够的数据
                    # 获取最近的价格数据进行头肩顶特征分析
                    recent_data = data.tail(30)
                    high_prices = recent_data["high"].values
                    low_prices = recent_data["low"].values
                    close_prices = recent_data["close"].values
                    volume_data = recent_data["volume"].values
                    
                    # 寻找局部高点和低点
                    local_highs = []
                    local_lows = []
                    window = 3
                    
                    for i in range(window, len(high_prices) - window):
                        # 局部高点
                        if all(high_prices[i] >= high_prices[i-j] for j in range(1, window+1)) and \
                           all(high_prices[i] >= high_prices[i+j] for j in range(1, window+1)):
                            local_highs.append((i, high_prices[i]))
                        
                        # 局部低点
                        if all(low_prices[i] <= low_prices[i-j] for j in range(1, window+1)) and \
                           all(low_prices[i] <= low_prices[i+j] for j in range(1, window+1)):
                            local_lows.append((i, low_prices[i]))
                    
                    # 头肩顶质量评分
                    quality_score = 0.5  # 基础分
                    
                    # 检查是否有足够的高点形成头肩顶
                    if len(local_highs) >= 3:
                        # 取最近的三个高点作为潜在的左肩、头部、右肩
                        left_shoulder = local_highs[-3]
                        head = local_highs[-2]
                        right_shoulder = local_highs[-1]
                        
                        # 验证头肩顶结构
                        head_is_highest = head[1] > left_shoulder[1] and head[1] > right_shoulder[1]
                        shoulders_similar = abs(left_shoulder[1] - right_shoulder[1]) <= (head[1] - min(left_shoulder[1], right_shoulder[1])) * 0.3
                        
                        if head_is_highest:
                            quality_score += 0.2
                        if shoulders_similar:
                            quality_score += 0.15
                        
                        # 计算头肩顶的对称性
                        if len(local_highs) >= 3:
                            head_height = head[1]
                            avg_shoulder_height = (left_shoulder[1] + right_shoulder[1]) / 2
                            height_ratio = (head_height - avg_shoulder_height) / avg_shoulder_height if avg_shoulder_height > 0 else 0
                            
                            if 0.05 <= height_ratio <= 0.25:  # 头部高出肩部5%-25%是理想的
                                quality_score += 0.15
                            elif height_ratio > 0.25:
                                quality_score += 0.1  # 头部过高相对较弱
                        
                        # 检查颈线支撑（两个肩部之间的低点）
                        if len(local_lows) >= 2:
                            neckline_lows = [low for low in local_lows if left_shoulder[0] < low[0] < right_shoulder[0]]
                            if neckline_lows:
                                neckline_level = max(low[1] for low in neckline_lows)
                                current_price = close_prices[-1]
                                
                                # 检查是否跌破颈线
                                if current_price < neckline_level:
                                    quality_score += 0.2
                                    reason = "头肩顶形态完成，跌破颈线确认"
                                    strength = min(0.95, strength + 0.1)
                                    confidence = min(0.95, confidence + 0.1)
                                
                                metadata["neckline_level"] = round(neckline_level, 3)
                                metadata["neckline_break"] = current_price < neckline_level
                        
                        # 根据质量调整信号强度
                        strength = min(0.95, 0.75 + quality_score * 0.2)
                        confidence = min(0.95, 0.70 + quality_score * 0.2)
                        
                        metadata.update({
                            "pattern_type": "head_shoulders_top",
                            "left_shoulder_height": round(left_shoulder[1], 3),
                            "head_height": round(head[1], 3),
                            "right_shoulder_height": round(right_shoulder[1], 3),
                            "height_ratio": round(height_ratio if 'height_ratio' in locals() else 0.0, 3),
                            "quality_score": round(quality_score, 3),
                            "reversal_potential": "high" if quality_score > 0.8 else "medium",
                            "head_is_highest": head_is_highest,
                            "shoulders_similar": shoulders_similar
                        })

            # 检查成交量确认（头肩顶的成交量应该递减）
            if signal_type == "sell" and len(data) >= 15:
                recent_volume = data["volume"].tail(15)
                volume_trend = recent_volume.diff().mean()  # 成交量趋势
                
                if volume_trend < 0:  # 成交量递减，符合头肩顶特征
                    strength = min(0.95, strength + 0.05)
                    confidence = min(0.95, confidence + 0.05)
                    metadata["volume_confirmation"] = "declining"
                    reason += "，成交量递减确认"
                else:
                    metadata["volume_confirmation"] = "normal"
                
                # 检查突破时的成交量放大
                if len(data) >= 5:
                    recent_avg_volume = recent_volume.tail(5).mean()
                    current_volume = data["volume"].iloc[-1]
                    volume_ratio = current_volume / recent_avg_volume if recent_avg_volume > 0 else 1.0
                    
                    if volume_ratio > 1.3:  # 突破时放量
                        strength = min(0.95, strength + 0.08)
                        confidence = min(0.95, confidence + 0.08)
                        metadata["breakout_volume"] = "high"
                        if "跌破颈线确认" in reason:
                            reason += "，放量突破"
                    else:
                        metadata["breakout_volume"] = "normal"
                    
                    metadata["volume_ratio"] = round(volume_ratio, 3)

            # 分析趋势背景，增强头肩顶形态的信号强度
            if signal_type == "sell" and len(data) >= 20:
                # 检查前期是否为上升趋势（头肩顶在上升趋势末期最有效）
                price_data = data["close"].tail(20)
                early_trend = price_data.head(10).mean()
                recent_trend = price_data.tail(10).mean()
                trend_change = (recent_trend - early_trend) / early_trend if early_trend > 0 else 0
                
                if trend_change > 0.02:  # 前期有明显上升趋势
                    strength = min(0.95, strength + 0.08)
                    confidence = min(0.95, confidence + 0.08)
                    metadata["trend_context"] = "uptrend_reversal"
                    metadata["signal_enhancement"] = "trend_top_reversal"
                    reason = reason.replace("强顶部反转信号", "极强趋势反转信号")
                elif trend_change < -0.02:  # 前期已在下跌
                    strength = max(0.6, strength - 0.1)
                    confidence = max(0.6, confidence - 0.1)
                    metadata["trend_context"] = "downtrend_continuation"
                    metadata["signal_enhancement"] = "weak_continuation"
                else:
                    metadata["trend_context"] = "sideways"
                    metadata["signal_enhancement"] = "normal"

            # 检查关键阻力位确认
            if signal_type == "sell" and len(data) >= 25:
                # 检查头部是否在重要阻力位附近
                price_data = data["high"].tail(25)
                resistance_levels = []
                
                # 寻找重要的阻力位
                for i in range(5, len(price_data) - 5):
                    local_high = price_data.iloc[i]
                    nearby_highs = price_data.iloc[i-5:i+6]
                    if (nearby_highs >= local_high * 0.98).sum() >= 3:  # 附近有多个相近高点
                        resistance_levels.append(local_high)
                
                if resistance_levels:
                    max_resistance = max(resistance_levels)
                    current_high = data["high"].iloc[-1]
                    
                    if abs(current_high - max_resistance) / max_resistance <= 0.02:  # 在阻力位2%范围内
                        strength = min(0.95, strength + 0.05)
                        confidence = min(0.95, confidence + 0.05)
                        metadata["resistance_confirmation"] = "strong"
                        reason += "，重要阻力位确认"
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
                    'pattern_family': 'complex_pattern',
                    'pattern_characteristic': 'bearish_reversal',
                    'requires_confirmation': 'neckline_break',
                    'signal_direction': 'bearish_only',
                    'pattern_complexity': 'high',
                    'formation_period': 'medium_to_long',
                    'reliability': 'high',
                    **metadata
                }
            }

        except Exception as e:
            logger.error(f"头肩顶形态信号生成失败: {e}")
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
        
        # 检查数据量（头肩顶形态至少需要30个数据点）
        if len(data) < 30:
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
                'pattern_family': 'complex_pattern',
                'pattern_characteristic': 'bearish_reversal',
                'requires_confirmation': 'neckline_break',
                'signal_direction': 'bearish_only',
                'pattern_complexity': 'high',
                'formation_period': 'medium_to_long',
                'reliability': 'high'
            }
        }


class DoubleTop(AdvancedCandlestickPatterns):
    """双顶形态识别"""

    def __init__(
        self, period: int = 30
    ):  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        super().__init__(period=period)  # 正确传递period参数
        self.name = "DOUBLE_TOP"

    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        获取双顶形态交易信号
        
        双顶是重要的看跌反转形态，由两个相近高度的峰和中间的谷组成。
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            Dict[str, Any]: 标准化的交易信号字典
        """
        # 数据验证
        if not self._validate_signal_data(data):
            return self._get_default_signal()
        
        # 确保有双顶形态列，如果没有则先计算
        if 'double_top' not in data.columns:
            try:
                data = self.calculate(data)
            except Exception as e:
                logger.warning(f"计算双顶形态失败: {e}")
                return self._get_default_signal()
        
        if data.empty or 'double_top' not in data.columns:
            return self._get_default_signal()
        
        # 初始化信号参数
        signal_type = "hold"
        strength = 0.5
        confidence = 0.6
        reason = "未检测到双顶形态"
        
        # 获取最新的双顶信号
        latest_double_top = data['double_top'].iloc[-1] if len(data) > 0 else False
        
        if latest_double_top:
            # 检测到双顶形态，分析信号强度
            signal_type = "sell"
            
            # 分析双顶形态质量
            recent_data = data.tail(min(len(data), self.period))
            
            # 计算两个峰的高度差异（越接近越好）
            highs = recent_data['high'].values
            peaks = []
            
            # 简单峰值检测
            for i in range(1, len(highs) - 1):
                if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                    peaks.append((i, highs[i]))
            
            if len(peaks) >= 2:
                # 找到最高的两个峰
                peaks.sort(key=lambda x: x[1], reverse=True)
                peak1_height, peak2_height = peaks[0][1], peaks[1][1]
                
                # 计算峰值高度相似性
                height_similarity = 1.0 - abs(peak1_height - peak2_height) / max(peak1_height, peak2_height)
                
                # 计算基础强度
                base_strength = 0.6 + height_similarity * 0.3
                
                # 成交量确认（第二个峰成交量应该减少）
                volume_confirmation = 1.0
                if 'volume' in recent_data.columns:
                    recent_volumes = recent_data['volume'].values
                    if len(recent_volumes) >= len(peaks[0][0:2]):
                        avg_volume_1st_half = np.mean(recent_volumes[:len(recent_volumes)//2])
                        avg_volume_2nd_half = np.mean(recent_volumes[len(recent_volumes)//2:])
                        
                        if avg_volume_2nd_half < avg_volume_1st_half:
                            volume_confirmation = 1.2  # 成交量递减确认
                        else:
                            volume_confirmation = 0.9  # 成交量未递减
                
                # 趋势背景分析
                trend_context = 1.0
                if len(recent_data) >= 10:
                    price_trend = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[-10]) / recent_data['close'].iloc[-10]
                    if price_trend > 0.05:  # 上升趋势中的双顶更有意义
                        trend_context = 1.3
                    elif price_trend < -0.05:  # 下跌趋势中的双顶意义较小
                        trend_context = 0.8
                
                # 计算最终强度
                strength = min(0.95, base_strength * volume_confirmation * trend_context)
                confidence = min(0.9, 0.6 + height_similarity * 0.3)
                
                reason = f"检测到双顶形态，峰值相似度{height_similarity:.2f}"
                
            else:
                # 形态不完整
                strength = 0.4
                confidence = 0.5
                reason = "双顶形态不完整"
        
        # 构建元数据
        metadata = {
            "indicator_name": "DOUBLE_TOP",
            "pattern_category": "reversal",
            "pattern_family": "chart_pattern",
            "pattern_characteristic": "bearish_reversal",
            "requires_confirmation": "neckline_break", 
            "signal_direction": "bearish_only",
            "pattern_complexity": "medium",
            "formation_period": "medium",
            "reliability": "high",
            "double_top_detected": bool(latest_double_top),
            "peak_count": len(peaks) if 'peaks' in locals() else 0,
            "height_similarity": height_similarity if 'height_similarity' in locals() else 0.0,
            "volume_confirmation": volume_confirmation if 'volume_confirmation' in locals() else 1.0,
            "trend_context": "uptrend" if 'price_trend' in locals() and price_trend > 0.05 else ("downtrend" if 'price_trend' in locals() and price_trend < -0.05 else "sideways")
        }
        
        return {
            "signal_type": signal_type,
            "strength": float(strength),
            "confidence": float(confidence),
            "timestamp": data.index[-1] if len(data) > 0 else None,
            "reason": reason,
            "metadata": metadata
        }
    
    def _validate_signal_data(self, data: pd.DataFrame) -> bool:
        """验证信号数据的有效性"""
        if not isinstance(data, pd.DataFrame):
            return False
        
        if data.empty:
            return False
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            return False
        
        # 双顶至少需要足够的数据点来形成两个峰
        min_periods = getattr(self, 'period', 30)
        if len(data) < min_periods:
            return False
        
        return True
    
    def _get_default_signal(self) -> Dict[str, Any]:
        """获取默认的持有信号"""
        return {
            "signal_type": "hold",
            "strength": 0.5,
            "confidence": 0.6,
            "timestamp": None,
            "reason": "数据不足或形态不明确",
            "metadata": {
                "indicator_name": "DOUBLE_TOP",
                "pattern_category": "reversal",
                "pattern_family": "chart_pattern", 
                "pattern_characteristic": "bearish_reversal",
                "requires_confirmation": "neckline_break",
                "signal_direction": "bearish_only",
                "pattern_complexity": "medium",
                "formation_period": "medium",
                "reliability": "high",
                "double_top_detected": False,
                "peak_count": 0,
                "height_similarity": 0.0,
                "volume_confirmation": 1.0,
                "trend_context": "unknown"
            }
        }


class DoubleBottom(AdvancedCandlestickPatterns):
    """双底形态识别"""

    def __init__(
        self, period: int = 30
    ):  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        super().__init__(period=period)  # 正确传递period参数
        self.name = "DOUBLE_BOTTOM"

    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        获取双底形态交易信号
        
        双底是重要的看涨反转形态，由两个相近低度的谷和中间的峰组成。
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            Dict[str, Any]: 标准化的交易信号字典
        """
        # 数据验证
        if not self._validate_signal_data(data):
            return self._get_default_signal()
        
        # 确保有双底形态列，如果没有则先计算
        if 'double_bottom' not in data.columns:
            try:
                data = self.calculate(data)
            except Exception as e:
                logger.warning(f"计算双底形态失败: {e}")
                return self._get_default_signal()
        
        if data.empty or 'double_bottom' not in data.columns:
            return self._get_default_signal()
        
        # 初始化信号参数
        signal_type = "hold"
        strength = 0.5
        confidence = 0.6
        reason = "未检测到双底形态"
        
        # 获取最新的双底信号
        latest_double_bottom = data['double_bottom'].iloc[-1] if len(data) > 0 else False
        
        if latest_double_bottom:
            # 检测到双底形态，分析信号强度
            signal_type = "buy"
            
            # 分析双底形态质量
            recent_data = data.tail(min(len(data), self.period))
            
            # 计算两个谷的低度差异（越接近越好）
            lows = recent_data['low'].values
            troughs = []
            
            # 简单谷值检测
            for i in range(1, len(lows) - 1):
                if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                    troughs.append((i, lows[i]))
            
            if len(troughs) >= 2:
                # 找到最低的两个谷
                troughs.sort(key=lambda x: x[1])
                trough1_depth, trough2_depth = troughs[0][1], troughs[1][1]
                
                # 计算谷值深度相似性
                depth_similarity = 1.0 - abs(trough1_depth - trough2_depth) / max(trough1_depth, trough2_depth)
                
                # 基于深度相似性调整信号强度
                if depth_similarity > 0.95:  # 非常相似
                    strength = 0.9
                    confidence = 0.85
                    reason = "双底形态完美，谷值深度几乎相同"
                elif depth_similarity > 0.9:  # 较相似
                    strength = 0.8
                    confidence = 0.8
                    reason = "双底形态良好，谷值深度相近"
                elif depth_similarity > 0.8:  # 一般相似
                    strength = 0.7
                    confidence = 0.75
                    reason = "双底形态一般，谷值深度有差异"
                else:  # 相似度较低
                    strength = 0.6
                    confidence = 0.7
                    reason = "双底形态较弱，谷值深度差异较大"
            else:
                # 未找到明显的双谷结构
                strength = 0.6
                confidence = 0.65
                reason = "双底形态识别，但谷值结构不够明显"
            
            # 成交量确认
            if 'volume' in recent_data.columns:
                volumes = recent_data['volume'].values
                avg_volume = np.mean(volumes)
                
                # 检查最近的成交量是否有放大
                recent_volume_ratio = volumes[-3:].mean() / avg_volume if len(volumes) >= 3 else 1.0
                
                if recent_volume_ratio > 1.5:  # 成交量显著放大
                    strength = min(1.0, strength + 0.1)
                    confidence = min(1.0, confidence + 0.1)
                    reason += "，成交量确认"
                elif recent_volume_ratio < 0.7:  # 成交量萎缩
                    strength = max(0.3, strength - 0.1)
                    confidence = max(0.4, confidence - 0.1)
                    reason += "，但成交量不足"
            
            # 趋势背景分析
            if len(recent_data) >= 10:
                price_trend = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[-10]) / recent_data['close'].iloc[-10]
                
                if price_trend < -0.1:  # 下降趋势中的双底更有效
                    strength = min(1.0, strength + 0.05)
                    confidence = min(1.0, confidence + 0.05)
                    reason += "，下降趋势反转"
                elif price_trend > 0.1:  # 上升趋势中的双底效果较弱
                    strength = max(0.4, strength - 0.05)
                    confidence = max(0.5, confidence - 0.05)
                    reason += "，但处于上升趋势"
        
        # 构建标准化信号字典
        metadata = {
            'pattern_type': 'double_bottom',
            'double_bottom_detected': latest_double_bottom,
            'trough_count': len(troughs) if 'troughs' in locals() else 0,
            'depth_similarity': depth_similarity if 'depth_similarity' in locals() else 0.0,
            'volume_confirmation': recent_volume_ratio > 1.2 if 'recent_volume_ratio' in locals() else False,
            'trend_context': 'downtrend' if 'price_trend' in locals() and price_trend < -0.05 else 'uptrend' if 'price_trend' in locals() and price_trend > 0.05 else 'sideways'
        }
        
        return {
            'signal_type': signal_type,
            'strength': max(0.0, min(1.0, strength)),
            'confidence': max(0.0, min(1.0, confidence)),
            'timestamp': pd.Timestamp.now(),
            'reason': reason,
            'metadata': metadata
        }

    def _validate_signal_data(self, data: pd.DataFrame) -> bool:
        """
        验证信号生成所需的数据
        
        Args:
            data: 输入数据DataFrame
            
        Returns:
            bool: 数据是否有效
        """
        if not isinstance(data, pd.DataFrame):
            return False
        
        if data.empty:
            return False
        
        # 检查必需列
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            return False
        
        # 检查数据长度
        if len(data) < self.minimum_periods:
            return False
        
        return True

    def _get_default_signal(self, reason: str = "数据不足或无双底形态") -> Dict[str, Any]:
        """
        生成默认的持有信号
        
        Args:
            reason: 默认信号的原因
            
        Returns:
            Dict[str, Any]: 默认信号字典
        """
        return {
            'signal_type': 'hold',
            'strength': 0.5,
            'confidence': 0.5,
            'timestamp': pd.Timestamp.now(),
            'reason': reason,
            'metadata': {
                'pattern_type': 'double_bottom',
                'double_bottom_detected': False,
                'trough_count': 0,
                'depth_similarity': 0.0,
                'volume_confirmation': False,
                'trend_context': 'unknown'
            }
        }


class Triangle(AdvancedCandlestickPatterns):
    """三角形形态识别"""

    def __init__(
        self, period: int = 30
    ):  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        super().__init__(period=period)  # 正确传递period参数
        self.name = "TRIANGLE"

    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        获取三角形形态交易信号
        
        三角形是重要的整理形态，包括上升三角形（看涨）、下降三角形（看跌）、
        对称三角形（突破方向决定信号）。关键是识别收敛的趋势线和突破方向。
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            Dict[str, Any]: 标准化的交易信号字典
        """
        # 数据验证
        if not self._validate_signal_data(data):
            return self._get_default_signal()
        
        # 确保有三角形形态列，如果没有则先计算
        triangle_columns = ['triangle', 'ascending_triangle', 'descending_triangle', 'symmetrical_triangle']
        has_triangle_data = any(col in data.columns for col in triangle_columns)
        
        if not has_triangle_data:
            try:
                data = self.calculate(data)
            except Exception as e:
                logger.warning(f"计算三角形形态失败: {e}")
                return self._get_default_signal()
        
        if data.empty:
            return self._get_default_signal()
        
        # 初始化信号参数
        signal_type = "hold"
        strength = 0.5
        confidence = 0.6
        reason = "未检测到三角形形态"
        
        # 分析三角形形态和突破
        triangle_analysis = self._analyze_triangle_pattern(data)
        
        if triangle_analysis['pattern_detected']:
            pattern_type = triangle_analysis['pattern_type']
            breakout_direction = triangle_analysis['breakout_direction']
            
            # 根据三角形类型和突破方向确定信号
            if pattern_type == 'ascending' and breakout_direction == 'upward':
                signal_type = "buy"
                strength = 0.8
                confidence = 0.85
                reason = "上升三角形向上突破，强烈看涨信号"
            elif pattern_type == 'descending' and breakout_direction == 'downward':
                signal_type = "sell"
                strength = 0.8
                confidence = 0.85
                reason = "下降三角形向下突破，强烈看跌信号"
            elif pattern_type == 'symmetrical':
                if breakout_direction == 'upward':
                    signal_type = "buy"
                    strength = 0.7
                    confidence = 0.75
                    reason = "对称三角形向上突破，看涨信号"
                elif breakout_direction == 'downward':
                    signal_type = "sell"
                    strength = 0.7
                    confidence = 0.75
                    reason = "对称三角形向下突破，看跌信号"
                else:
                    signal_type = "hold"
                    strength = 0.5
                    confidence = 0.6
                    reason = "对称三角形形成中，等待突破"
            else:
                # 形态完整但未突破
                signal_type = "hold"
                strength = 0.6
                confidence = 0.65
                reason = f"{pattern_type}三角形形成，等待突破确认"
            
            # 成交量确认
            if triangle_analysis['volume_confirmation']:
                strength = min(1.0, strength + 0.1)
                confidence = min(1.0, confidence + 0.1)
                reason += "，成交量确认"
            elif triangle_analysis['volume_declining']:
                # 三角形形成期间成交量递减是正常的
                pass
            else:
                strength = max(0.3, strength - 0.05)
                confidence = max(0.4, confidence - 0.05)
                reason += "，但成交量异常"
            
            # 突破质量评估
            if triangle_analysis['breakout_strength'] > 0.8:
                strength = min(1.0, strength + 0.1)
                confidence = min(1.0, confidence + 0.1)
                reason += "，突破强劲"
            elif triangle_analysis['breakout_strength'] < 0.3:
                strength = max(0.3, strength - 0.1)
                confidence = max(0.4, confidence - 0.1)
                reason += "，突破疲弱"
            
            # 形态完整性评估
            completeness = triangle_analysis['pattern_completeness']
            if completeness > 0.8:
                confidence = min(1.0, confidence + 0.05)
            elif completeness < 0.5:
                strength = max(0.3, strength - 0.1)
                confidence = max(0.4, confidence - 0.1)
        
        # 构建标准化信号字典
        metadata = {
            'pattern_type': 'triangle',
            'triangle_detected': triangle_analysis.get('pattern_detected', False),
            'triangle_subtype': triangle_analysis.get('pattern_type', 'unknown'),
            'breakout_direction': triangle_analysis.get('breakout_direction', 'none'),
            'breakout_strength': triangle_analysis.get('breakout_strength', 0.0),
            'volume_confirmation': triangle_analysis.get('volume_confirmation', False),
            'pattern_completeness': triangle_analysis.get('pattern_completeness', 0.0),
            'convergence_point_distance': triangle_analysis.get('convergence_distance', 0),
            'formation_duration': triangle_analysis.get('formation_duration', 0)
        }
        
        return {
            'signal_type': signal_type,
            'strength': max(0.0, min(1.0, strength)),
            'confidence': max(0.0, min(1.0, confidence)),
            'timestamp': pd.Timestamp.now(),
            'reason': reason,
            'metadata': metadata
        }

    def _analyze_triangle_pattern(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        分析三角形形态
        
        Args:
            data: 价格数据
            
        Returns:
            Dict: 包含形态分析结果的字典
        """
        analysis = {
            'pattern_detected': False,
            'pattern_type': 'unknown',
            'breakout_direction': 'none',
            'breakout_strength': 0.0,
            'volume_confirmation': False,
            'volume_declining': False,
            'pattern_completeness': 0.0,
            'convergence_distance': 0,
            'formation_duration': 0
        }
        
        try:
            if len(data) < self.period:
                return analysis
            
            recent_data = data.tail(self.period).copy()
            highs = recent_data['high'].values
            lows = recent_data['low'].values
            volumes = recent_data['volume'].values if 'volume' in recent_data.columns else None
            
            # 1. 检测趋势线
            resistance_line = self._find_resistance_line(highs)
            support_line = self._find_support_line(lows)
            
            if resistance_line is None or support_line is None:
                return analysis
            
            # 2. 判断三角形类型
            resistance_slope = resistance_line['slope']
            support_slope = support_line['slope']
            
            # 斜率容差
            slope_tolerance = 0.1
            
            if abs(resistance_slope) < slope_tolerance and support_slope > slope_tolerance:
                # 水平阻力线 + 上升支撑线 = 上升三角形
                pattern_type = 'ascending'
            elif abs(support_slope) < slope_tolerance and resistance_slope < -slope_tolerance:
                # 水平支撑线 + 下降阻力线 = 下降三角形
                pattern_type = 'descending'
            elif resistance_slope < -slope_tolerance and support_slope > slope_tolerance:
                # 收敛的阻力支撑线 = 对称三角形
                pattern_type = 'symmetrical'
            else:
                return analysis
            
            # 3. 计算收敛点距离
            convergence_distance = self._calculate_convergence_distance(resistance_line, support_line)
            
            # 4. 评估形态完整性
            completeness = self._evaluate_pattern_completeness(highs, lows, resistance_line, support_line)
            
            # 5. 检测突破
            latest_high = highs[-1]
            latest_low = lows[-1]
            latest_close = recent_data['close'].iloc[-1]
            
            resistance_level = resistance_line['intercept'] + resistance_line['slope'] * (len(highs) - 1)
            support_level = support_line['intercept'] + support_line['slope'] * (len(lows) - 1)
            
            breakout_direction = 'none'
            breakout_strength = 0.0
            
            # 计算突破幅度（相对于最近的形态范围）
            pattern_range = resistance_level - support_level
            
            if latest_close > resistance_level:
                breakout_direction = 'upward'
                breakout_strength = min(1.0, (latest_close - resistance_level) / (pattern_range * 0.1))
            elif latest_close < support_level:
                breakout_direction = 'downward'
                breakout_strength = min(1.0, (support_level - latest_close) / (pattern_range * 0.1))
            
            # 6. 成交量分析
            volume_confirmation = False
            volume_declining = False
            
            if volumes is not None:
                # 检查形成期间成交量是否递减
                early_volume = np.mean(volumes[:len(volumes)//3])
                late_volume = np.mean(volumes[-len(volumes)//3:])
                volume_declining = late_volume < early_volume * 0.8
                
                # 检查突破时成交量是否放大
                if breakout_direction != 'none' and len(volumes) >= 3:
                    breakout_volume = np.mean(volumes[-3:])
                    avg_volume = np.mean(volumes[:-3])
                    volume_confirmation = breakout_volume > avg_volume * 1.5
            
            # 更新分析结果
            analysis.update({
                'pattern_detected': True,
                'pattern_type': pattern_type,
                'breakout_direction': breakout_direction,
                'breakout_strength': breakout_strength,
                'volume_confirmation': volume_confirmation,
                'volume_declining': volume_declining,
                'pattern_completeness': completeness,
                'convergence_distance': convergence_distance,
                'formation_duration': len(recent_data)
            })
            
        except Exception as e:
            logger.warning(f"三角形形态分析失败: {e}")
        
        return analysis

    def _find_resistance_line(self, highs: np.ndarray) -> Dict[str, float]:
        """查找阻力线"""
        try:
            # 寻找相对高点
            peaks = []
            for i in range(2, len(highs) - 2):
                if highs[i] >= highs[i-1] and highs[i] >= highs[i+1] and \
                   highs[i] >= highs[i-2] and highs[i] >= highs[i+2]:
                    peaks.append((i, highs[i]))
            
            if len(peaks) < 2:
                return None
            
            # 选择最明显的高点进行线性回归
            peaks.sort(key=lambda x: x[1], reverse=True)
            selected_peaks = peaks[:min(4, len(peaks))]
            selected_peaks.sort(key=lambda x: x[0])  # 按时间排序
            
            x = np.array([p[0] for p in selected_peaks])
            y = np.array([p[1] for p in selected_peaks])
            
            # 线性回归
            slope, intercept = np.polyfit(x, y, 1)
            
            return {'slope': slope, 'intercept': intercept, 'points': selected_peaks}
            
        except Exception:
            return None

    def _find_support_line(self, lows: np.ndarray) -> Dict[str, float]:
        """查找支撑线"""
        try:
            # 寻找相对低点
            troughs = []
            for i in range(2, len(lows) - 2):
                if lows[i] <= lows[i-1] and lows[i] <= lows[i+1] and \
                   lows[i] <= lows[i-2] and lows[i] <= lows[i+2]:
                    troughs.append((i, lows[i]))
            
            if len(troughs) < 2:
                return None
            
            # 选择最明显的低点进行线性回归
            troughs.sort(key=lambda x: x[1])
            selected_troughs = troughs[:min(4, len(troughs))]
            selected_troughs.sort(key=lambda x: x[0])  # 按时间排序
            
            x = np.array([t[0] for t in selected_troughs])
            y = np.array([t[1] for t in selected_troughs])
            
            # 线性回归
            slope, intercept = np.polyfit(x, y, 1)
            
            return {'slope': slope, 'intercept': intercept, 'points': selected_troughs}
            
        except Exception:
            return None

    def _calculate_convergence_distance(self, resistance_line: Dict, support_line: Dict) -> int:
        """计算收敛点距离"""
        try:
            if resistance_line['slope'] == support_line['slope']:
                return 999  # 平行线，无收敛点
            
            # 计算交点的x坐标
            x_intersect = (support_line['intercept'] - resistance_line['intercept']) / \
                         (resistance_line['slope'] - support_line['slope'])
            
            return max(0, int(x_intersect - (self.period - 1)))
            
        except Exception:
            return 0

    def _evaluate_pattern_completeness(self, highs: np.ndarray, lows: np.ndarray, 
                                     resistance_line: Dict, support_line: Dict) -> float:
        """评估形态完整性"""
        try:
            total_points = len(highs)
            resistance_touches = 0
            support_touches = 0
            
            # 计算价格触及趋势线的次数
            for i in range(total_points):
                resistance_level = resistance_line['intercept'] + resistance_line['slope'] * i
                support_level = support_line['intercept'] + support_line['slope'] * i
                
                # 允许一定的触及容差
                tolerance = (resistance_level - support_level) * 0.02
                
                if abs(highs[i] - resistance_level) <= tolerance:
                    resistance_touches += 1
                
                if abs(lows[i] - support_level) <= tolerance:
                    support_touches += 1
            
            # 完整性评分：基于趋势线触及次数和收敛程度
            touch_score = min(1.0, (resistance_touches + support_touches) / 6)
            
            # 收敛程度评分
            initial_range = abs(resistance_line['intercept'] - support_line['intercept'])
            final_resistance = resistance_line['intercept'] + resistance_line['slope'] * (total_points - 1)
            final_support = support_line['intercept'] + support_line['slope'] * (total_points - 1)
            final_range = abs(final_resistance - final_support)
            
            convergence_score = 1.0 - (final_range / initial_range) if initial_range > 0 else 0.0
            convergence_score = max(0.0, min(1.0, convergence_score))
            
            return (touch_score * 0.6 + convergence_score * 0.4)
            
        except Exception:
            return 0.0

    def _validate_signal_data(self, data: pd.DataFrame) -> bool:
        """
        验证信号生成所需的数据
        
        Args:
            data: 输入数据DataFrame
            
        Returns:
            bool: 数据是否有效
        """
        if not isinstance(data, pd.DataFrame):
            return False
        
        if data.empty:
            return False
        
        # 检查必需列
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            return False
        
        # 检查数据长度
        if len(data) < self.minimum_periods:
            return False
        
        return True

    def _get_default_signal(self, reason: str = "数据不足或无三角形形态") -> Dict[str, Any]:
        """
        生成默认的持有信号
        
        Args:
            reason: 默认信号的原因
            
        Returns:
            Dict[str, Any]: 默认信号字典
        """
        return {
            'signal_type': 'hold',
            'strength': 0.5,
            'confidence': 0.5,
            'timestamp': pd.Timestamp.now(),
            'reason': reason,
            'metadata': {
                'pattern_type': 'triangle',
                'triangle_detected': False,
                'triangle_subtype': 'unknown',
                'breakout_direction': 'none',
                'breakout_strength': 0.0,
                'volume_confirmation': False,
                'pattern_completeness': 0.0,
                'convergence_point_distance': 0,
                'formation_duration': 0
            }
        }


class Wedge(AdvancedCandlestickPatterns):
    """楔形形态识别"""

    def __init__(
        self, period: int = 30
    ):  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        super().__init__(period=period)  # 正确传递period参数
        self.name = "WEDGE"

    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        获取楔形形态交易信号
        
        楔形是重要的反转形态，分为上升楔形（看跌反转）和下降楔形（看涨反转）。
        楔形的特征是两条趋势线都向同一方向倾斜并收敛，与三角形不同。
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            Dict[str, Any]: 标准化的交易信号字典
        """
        # 数据验证
        if not self._validate_signal_data(data):
            return self._get_default_signal()
        
        # 确保有楔形形态列，如果没有则先计算
        wedge_columns = ['wedge', 'rising_wedge', 'falling_wedge']
        has_wedge_data = any(col in data.columns for col in wedge_columns)
        
        if not has_wedge_data:
            try:
                data = self.calculate(data)
            except Exception as e:
                logger.warning(f"计算楔形形态失败: {e}")
                return self._get_default_signal()
        
        if data.empty:
            return self._get_default_signal()
        
        # 初始化信号参数
        signal_type = "hold"
        strength = 0.5
        confidence = 0.6
        reason = "未检测到楔形形态"
        
        # 分析楔形形态和突破
        wedge_analysis = self._analyze_wedge_pattern(data)
        
        if wedge_analysis['pattern_detected']:
            pattern_type = wedge_analysis['pattern_type']
            breakout_direction = wedge_analysis['breakout_direction']
            
            # 根据楔形类型和突破方向确定信号
            if pattern_type == 'rising' and breakout_direction == 'downward':
                # 上升楔形向下突破（经典看跌反转）
                signal_type = "sell"
                strength = 0.8
                confidence = 0.85
                reason = "上升楔形向下突破，强烈看跌反转信号"
            elif pattern_type == 'falling' and breakout_direction == 'upward':
                # 下降楔形向上突破（经典看涨反转）
                signal_type = "buy"
                strength = 0.8
                confidence = 0.85
                reason = "下降楔形向上突破，强烈看涨反转信号"
            elif pattern_type == 'rising' and breakout_direction == 'upward':
                # 上升楔形向上突破（假突破可能性高）
                signal_type = "hold"
                strength = 0.4
                confidence = 0.5
                reason = "上升楔形向上突破，谨慎观望（假突破风险）"
            elif pattern_type == 'falling' and breakout_direction == 'downward':
                # 下降楔形向下突破（假突破可能性高）
                signal_type = "hold"
                strength = 0.4
                confidence = 0.5
                reason = "下降楔形向下突破，谨慎观望（假突破风险）"
            else:
                # 形态完整但未突破
                signal_type = "hold"
                strength = 0.6
                confidence = 0.65
                reason = f"{pattern_type}楔形形成，等待反转突破确认"
            
            # 成交量确认
            if wedge_analysis['volume_confirmation']:
                strength = min(1.0, strength + 0.1)
                confidence = min(1.0, confidence + 0.1)
                reason += "，成交量确认"
            elif wedge_analysis['volume_declining']:
                # 楔形形成期间成交量递减是正常的
                if signal_type in ['buy', 'sell']:
                    # 但突破时应该有成交量放大
                    strength = max(0.3, strength - 0.05)
                    confidence = max(0.4, confidence - 0.05)
                    reason += "，但成交量不足"
            
            # 突破质量评估
            if wedge_analysis['breakout_strength'] > 0.8:
                strength = min(1.0, strength + 0.1)
                confidence = min(1.0, confidence + 0.1)
                reason += "，突破强劲"
            elif wedge_analysis['breakout_strength'] < 0.3:
                strength = max(0.3, strength - 0.1)
                confidence = max(0.4, confidence - 0.1)
                reason += "，突破疲弱"
            
            # 形态完整性评估
            completeness = wedge_analysis['pattern_completeness']
            if completeness > 0.8:
                confidence = min(1.0, confidence + 0.05)
            elif completeness < 0.5:
                strength = max(0.3, strength - 0.1)
                confidence = max(0.4, confidence - 0.1)
            
            # 楔形角度评估（角度过陡或过缓都影响可靠性）
            wedge_angle = wedge_analysis.get('wedge_angle', 0)
            if 15 <= wedge_angle <= 45:  # 理想角度范围
                confidence = min(1.0, confidence + 0.05)
            elif wedge_angle < 10 or wedge_angle > 60:
                strength = max(0.3, strength - 0.1)
                confidence = max(0.4, confidence - 0.1)
        
        # 构建标准化信号字典
        metadata = {
            'pattern_type': 'wedge',
            'wedge_detected': wedge_analysis.get('pattern_detected', False),
            'wedge_subtype': wedge_analysis.get('pattern_type', 'unknown'),
            'breakout_direction': wedge_analysis.get('breakout_direction', 'none'),
            'breakout_strength': wedge_analysis.get('breakout_strength', 0.0),
            'volume_confirmation': wedge_analysis.get('volume_confirmation', False),
            'pattern_completeness': wedge_analysis.get('pattern_completeness', 0.0),
            'wedge_angle': wedge_analysis.get('wedge_angle', 0.0),
            'convergence_point_distance': wedge_analysis.get('convergence_distance', 0),
            'formation_duration': wedge_analysis.get('formation_duration', 0),
            'trend_line_touches': wedge_analysis.get('trend_line_touches', 0)
        }
        
        return {
            'signal_type': signal_type,
            'strength': max(0.0, min(1.0, strength)),
            'confidence': max(0.0, min(1.0, confidence)),
            'timestamp': pd.Timestamp.now(),
            'reason': reason,
            'metadata': metadata
        }

    def _analyze_wedge_pattern(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        分析楔形形态
        
        Args:
            data: 价格数据
            
        Returns:
            Dict: 包含形态分析结果的字典
        """
        analysis = {
            'pattern_detected': False,
            'pattern_type': 'unknown',
            'breakout_direction': 'none',
            'breakout_strength': 0.0,
            'volume_confirmation': False,
            'volume_declining': False,
            'pattern_completeness': 0.0,
            'wedge_angle': 0.0,
            'convergence_distance': 0,
            'formation_duration': 0,
            'trend_line_touches': 0
        }
        
        try:
            if len(data) < self.period:
                return analysis
            
            recent_data = data.tail(self.period).copy()
            highs = recent_data['high'].values
            lows = recent_data['low'].values
            volumes = recent_data['volume'].values if 'volume' in recent_data.columns else None
            
            # 1. 检测趋势线
            upper_line = self._find_upper_trend_line(highs)
            lower_line = self._find_lower_trend_line(lows)
            
            if upper_line is None or lower_line is None:
                return analysis
            
            # 2. 判断楔形类型
            upper_slope = upper_line['slope']
            lower_slope = lower_line['slope']
            
            # 楔形特征：两条趋势线都向同一方向倾斜且收敛
            slope_threshold = 0.05  # 最小斜率阈值
            
            if upper_slope > slope_threshold and lower_slope > slope_threshold and upper_slope > lower_slope:
                # 两条线都上升，上线斜率大于下线 = 上升楔形
                pattern_type = 'rising'
            elif upper_slope < -slope_threshold and lower_slope < -slope_threshold and upper_slope < lower_slope:
                # 两条线都下降，上线斜率小于下线 = 下降楔形
                pattern_type = 'falling'
            else:
                # 不符合楔形特征
                return analysis
            
            # 3. 计算楔形角度
            wedge_angle = self._calculate_wedge_angle(upper_line, lower_line)
            
            # 4. 计算收敛点距离
            convergence_distance = self._calculate_convergence_distance_wedge(upper_line, lower_line)
            
            # 5. 评估形态完整性
            trend_line_touches, completeness = self._evaluate_wedge_completeness(
                highs, lows, upper_line, lower_line)
            
            # 6. 检测突破
            latest_close = recent_data['close'].iloc[-1]
            latest_high = highs[-1]
            latest_low = lows[-1]
            
            upper_level = upper_line['intercept'] + upper_line['slope'] * (len(highs) - 1)
            lower_level = lower_line['intercept'] + lower_line['slope'] * (len(lows) - 1)
            
            breakout_direction = 'none'
            breakout_strength = 0.0
            
            # 计算突破幅度
            wedge_range = upper_level - lower_level
            
            if latest_close > upper_level:
                breakout_direction = 'upward'
                breakout_strength = min(1.0, (latest_close - upper_level) / (wedge_range * 0.15))
            elif latest_close < lower_level:
                breakout_direction = 'downward'
                breakout_strength = min(1.0, (lower_level - latest_close) / (wedge_range * 0.15))
            
            # 7. 成交量分析
            volume_confirmation = False
            volume_declining = False
            
            if volumes is not None:
                # 检查形成期间成交量是否递减
                early_volume = np.mean(volumes[:len(volumes)//3])
                late_volume = np.mean(volumes[-len(volumes)//3:])
                volume_declining = late_volume < early_volume * 0.8
                
                # 检查突破时成交量是否放大
                if breakout_direction != 'none' and len(volumes) >= 3:
                    breakout_volume = np.mean(volumes[-3:])
                    avg_volume = np.mean(volumes[:-3])
                    volume_confirmation = breakout_volume > avg_volume * 1.5
            
            # 更新分析结果
            analysis.update({
                'pattern_detected': True,
                'pattern_type': pattern_type,
                'breakout_direction': breakout_direction,
                'breakout_strength': breakout_strength,
                'volume_confirmation': volume_confirmation,
                'volume_declining': volume_declining,
                'pattern_completeness': completeness,
                'wedge_angle': wedge_angle,
                'convergence_distance': convergence_distance,
                'formation_duration': len(recent_data),
                'trend_line_touches': trend_line_touches
            })
            
        except Exception as e:
            logger.warning(f"楔形形态分析失败: {e}")
        
        return analysis

    def _find_upper_trend_line(self, highs: np.ndarray) -> Dict[str, float]:
        """查找上趋势线"""
        try:
            # 寻找相对高点
            peaks = []
            for i in range(2, len(highs) - 2):
                if highs[i] >= highs[i-1] and highs[i] >= highs[i+1] and \
                   highs[i] >= highs[i-2] and highs[i] >= highs[i+2]:
                    peaks.append((i, highs[i]))
            
            if len(peaks) < 3:  # 楔形需要至少3个触点
                return None
            
            # 选择最近的高点进行线性回归
            peaks.sort(key=lambda x: x[0])  # 按时间排序
            selected_peaks = peaks[-min(5, len(peaks)):]  # 取最近的几个高点
            
            x = np.array([p[0] for p in selected_peaks])
            y = np.array([p[1] for p in selected_peaks])
            
            # 线性回归
            slope, intercept = np.polyfit(x, y, 1)
            
            return {'slope': slope, 'intercept': intercept, 'points': selected_peaks}
            
        except Exception:
            return None

    def _find_lower_trend_line(self, lows: np.ndarray) -> Dict[str, float]:
        """查找下趋势线"""
        try:
            # 寻找相对低点
            troughs = []
            for i in range(2, len(lows) - 2):
                if lows[i] <= lows[i-1] and lows[i] <= lows[i+1] and \
                   lows[i] <= lows[i-2] and lows[i] <= lows[i+2]:
                    troughs.append((i, lows[i]))
            
            if len(troughs) < 3:  # 楔形需要至少3个触点
                return None
            
            # 选择最近的低点进行线性回归
            troughs.sort(key=lambda x: x[0])  # 按时间排序
            selected_troughs = troughs[-min(5, len(troughs)):]  # 取最近的几个低点
            
            x = np.array([t[0] for t in selected_troughs])
            y = np.array([t[1] for t in selected_troughs])
            
            # 线性回归
            slope, intercept = np.polyfit(x, y, 1)
            
            return {'slope': slope, 'intercept': intercept, 'points': selected_troughs}
            
        except Exception:
            return None

    def _calculate_wedge_angle(self, upper_line: Dict, lower_line: Dict) -> float:
        """计算楔形角度"""
        try:
            # 计算两条趋势线之间的角度
            upper_slope = upper_line['slope']
            lower_slope = lower_line['slope']
            
            # 计算斜率差对应的角度
            slope_diff = abs(upper_slope - lower_slope)
            angle_rad = np.arctan(slope_diff)
            angle_deg = np.degrees(angle_rad)
            
            return min(90.0, angle_deg)  # 限制在0-90度
            
        except Exception:
            return 0.0

    def _calculate_convergence_distance_wedge(self, upper_line: Dict, lower_line: Dict) -> int:
        """计算楔形收敛点距离"""
        try:
            if abs(upper_line['slope'] - lower_line['slope']) < 1e-6:
                return 999  # 平行线，无收敛点
            
            # 计算交点的x坐标
            x_intersect = (lower_line['intercept'] - upper_line['intercept']) / \
                         (upper_line['slope'] - lower_line['slope'])
            
            return max(0, int(x_intersect - (self.period - 1)))
            
        except Exception:
            return 0

    def _evaluate_wedge_completeness(self, highs: np.ndarray, lows: np.ndarray, 
                                   upper_line: Dict, lower_line: Dict) -> tuple:
        """评估楔形完整性"""
        try:
            total_points = len(highs)
            upper_touches = 0
            lower_touches = 0
            
            # 计算价格触及趋势线的次数
            for i in range(total_points):
                upper_level = upper_line['intercept'] + upper_line['slope'] * i
                lower_level = lower_line['intercept'] + lower_line['slope'] * i
                
                # 允许一定的触及容差
                range_tolerance = abs(upper_level - lower_level) * 0.02
                
                if abs(highs[i] - upper_level) <= range_tolerance:
                    upper_touches += 1
                
                if abs(lows[i] - lower_level) <= range_tolerance:
                    lower_touches += 1
            
            total_touches = upper_touches + lower_touches
            
            # 完整性评分：基于趋势线触及次数和收敛程度
            touch_score = min(1.0, total_touches / 8)  # 期望至少8次触及
            
            # 收敛程度评分
            initial_range = abs(upper_line['intercept'] - lower_line['intercept'])
            final_upper = upper_line['intercept'] + upper_line['slope'] * (total_points - 1)
            final_lower = lower_line['intercept'] + lower_line['slope'] * (total_points - 1)
            final_range = abs(final_upper - final_lower)
            
            convergence_score = 1.0 - (final_range / initial_range) if initial_range > 0 else 0.0
            convergence_score = max(0.0, min(1.0, convergence_score))
            
            completeness = (touch_score * 0.7 + convergence_score * 0.3)
            
            return total_touches, completeness
            
        except Exception:
            return 0, 0.0

    def _validate_signal_data(self, data: pd.DataFrame) -> bool:
        """
        验证信号生成所需的数据
        
        Args:
            data: 输入数据DataFrame
            
        Returns:
            bool: 数据是否有效
        """
        if not isinstance(data, pd.DataFrame):
            return False
        
        if data.empty:
            return False
        
        # 检查必需列
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            return False
        
        # 检查数据长度
        if len(data) < self.minimum_periods:
            return False
        
        return True

    def _get_default_signal(self, reason: str = "数据不足或无楔形形态") -> Dict[str, Any]:
        """
        生成默认的持有信号
        
        Args:
            reason: 默认信号的原因
            
        Returns:
            Dict[str, Any]: 默认信号字典
        """
        return {
            'signal_type': 'hold',
            'strength': 0.5,
            'confidence': 0.5,
            'timestamp': pd.Timestamp.now(),
            'reason': reason,
            'metadata': {
                'pattern_type': 'wedge',
                'wedge_detected': False,
                'wedge_subtype': 'unknown',
                'breakout_direction': 'none',
                'breakout_strength': 0.0,
                'volume_confirmation': False,
                'pattern_completeness': 0.0,
                'wedge_angle': 0.0,
                'convergence_point_distance': 0,
                'formation_duration': 0,
                'trend_line_touches': 0
            }
        }


class Flag(AdvancedCandlestickPatterns):
    """旗形形态识别"""

    def __init__(
        self, period: int = 30
    ):  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        super().__init__(period=period)  # 正确传递period参数
        self.name = "FLAG"

    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        获取旗形形态交易信号
        
        旗形是重要的趋势继续形态，分为上升旗形（继续看涨）和下降旗形（继续看跌）。
        旗形由旗杆（强势移动）和旗面（温和整理）组成，突破后继续原趋势方向。
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            Dict[str, Any]: 标准化的交易信号字典
        """
        # 数据验证
        if not self._validate_signal_data(data):
            return self._get_default_signal()
        
        # 确保有旗形形态列，如果没有则先计算
        flag_columns = ['flag', 'bull_flag', 'bear_flag']
        has_flag_data = any(col in data.columns for col in flag_columns)
        
        if not has_flag_data:
            try:
                data = self.calculate(data)
            except Exception as e:
                logger.warning(f"计算旗形形态失败: {e}")
                return self._get_default_signal()
        
        if data.empty:
            return self._get_default_signal()
        
        # 初始化信号参数
        signal_type = "hold"
        strength = 0.5
        confidence = 0.6
        reason = "未检测到旗形形态"
        
        # 分析旗形形态和突破
        flag_analysis = self._analyze_flag_pattern(data)
        
        if flag_analysis['pattern_detected']:
            pattern_type = flag_analysis['pattern_type']
            breakout_direction = flag_analysis['breakout_direction']
            flagpole_strength = flag_analysis['flagpole_strength']
            
            # 根据旗形类型和突破方向确定信号
            if pattern_type == 'bull_flag' and breakout_direction == 'upward':
                # 上升旗形向上突破（继续看涨）
                signal_type = "buy"
                strength = 0.8
                confidence = 0.85
                reason = "上升旗形向上突破，强烈趋势继续信号"
            elif pattern_type == 'bear_flag' and breakout_direction == 'downward':
                # 下降旗形向下突破（继续看跌）
                signal_type = "sell"
                strength = 0.8
                confidence = 0.85
                reason = "下降旗形向下突破，强烈趋势继续信号"
            elif pattern_type == 'bull_flag' and breakout_direction == 'downward':
                # 上升旗形向下突破（趋势反转警告）
                signal_type = "sell"
                strength = 0.6
                confidence = 0.7
                reason = "上升旗形向下突破，趋势反转信号"
            elif pattern_type == 'bear_flag' and breakout_direction == 'upward':
                # 下降旗形向上突破（趋势反转警告）
                signal_type = "buy"
                strength = 0.6
                confidence = 0.7
                reason = "下降旗形向上突破，趋势反转信号"
            else:
                # 形态完整但未突破
                signal_type = "hold"
                strength = 0.6
                confidence = 0.65
                reason = f"{pattern_type.replace('_', '')}旗形形成，等待突破确认"
            
            # 旗杆强度评估（旗杆越强，信号越可靠）
            if flagpole_strength > 0.8:
                strength = min(1.0, strength + 0.1)
                confidence = min(1.0, confidence + 0.1)
                reason += "，旗杆强劲"
            elif flagpole_strength < 0.4:
                strength = max(0.3, strength - 0.1)
                confidence = max(0.4, confidence - 0.1)
                reason += "，旗杆疲弱"
            
            # 成交量确认
            if flag_analysis['volume_confirmation']:
                strength = min(1.0, strength + 0.1)
                confidence = min(1.0, confidence + 0.1)
                reason += "，成交量确认"
            elif flag_analysis['volume_declining_in_flag']:
                # 旗面期间成交量递减是正常的
                pass
            else:
                strength = max(0.3, strength - 0.05)
                confidence = max(0.4, confidence - 0.05)
                reason += "，成交量异常"
            
            # 突破质量评估
            if flag_analysis['breakout_strength'] > 0.8:
                strength = min(1.0, strength + 0.1)
                confidence = min(1.0, confidence + 0.1)
                reason += "，突破强劲"
            elif flag_analysis['breakout_strength'] < 0.3:
                strength = max(0.3, strength - 0.1)
                confidence = max(0.4, confidence - 0.1)
                reason += "，突破疲弱"
            
            # 旗面质量评估
            flag_quality = flag_analysis['flag_quality']
            if flag_quality > 0.8:
                confidence = min(1.0, confidence + 0.05)
            elif flag_quality < 0.5:
                strength = max(0.3, strength - 0.1)
                confidence = max(0.4, confidence - 0.1)
            
            # 旗形持续时间评估（时间太长或太短都影响可靠性）
            flag_duration = flag_analysis.get('flag_duration', 0)
            if 5 <= flag_duration <= 20:  # 理想持续时间
                confidence = min(1.0, confidence + 0.05)
            elif flag_duration < 3 or flag_duration > 25:
                strength = max(0.3, strength - 0.1)
                confidence = max(0.4, confidence - 0.1)
        
        # 构建标准化信号字典
        metadata = {
            'pattern_type': 'flag',
            'flag_detected': flag_analysis.get('pattern_detected', False),
            'flag_subtype': flag_analysis.get('pattern_type', 'unknown'),
            'breakout_direction': flag_analysis.get('breakout_direction', 'none'),
            'breakout_strength': flag_analysis.get('breakout_strength', 0.0),
            'flagpole_strength': flag_analysis.get('flagpole_strength', 0.0),
            'flag_quality': flag_analysis.get('flag_quality', 0.0),
            'volume_confirmation': flag_analysis.get('volume_confirmation', False),
            'flag_duration': flag_analysis.get('flag_duration', 0),
            'flagpole_length': flag_analysis.get('flagpole_length', 0.0),
            'flag_slope': flag_analysis.get('flag_slope', 0.0),
            'formation_duration': flag_analysis.get('formation_duration', 0)
        }
        
        return {
            'signal_type': signal_type,
            'strength': max(0.0, min(1.0, strength)),
            'confidence': max(0.0, min(1.0, confidence)),
            'timestamp': pd.Timestamp.now(),
            'reason': reason,
            'metadata': metadata
        }

    def _analyze_flag_pattern(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        分析旗形形态
        
        Args:
            data: 价格数据
            
        Returns:
            Dict: 包含形态分析结果的字典
        """
        analysis = {
            'pattern_detected': False,
            'pattern_type': 'unknown',
            'breakout_direction': 'none',
            'breakout_strength': 0.0,
            'flagpole_strength': 0.0,
            'flag_quality': 0.0,
            'volume_confirmation': False,
            'volume_declining_in_flag': False,
            'flag_duration': 0,
            'flagpole_length': 0.0,
            'flag_slope': 0.0,
            'formation_duration': 0
        }
        
        try:
            if len(data) < self.period:
                return analysis
            
            recent_data = data.tail(self.period).copy()
            closes = recent_data['close'].values
            highs = recent_data['high'].values
            lows = recent_data['low'].values
            volumes = recent_data['volume'].values if 'volume' in recent_data.columns else None
            
            # 1. 检测旗杆
            flagpole_info = self._detect_flagpole(closes, highs, lows)
            
            if not flagpole_info['detected']:
                return analysis
            
            flagpole_end_idx = flagpole_info['end_index']
            flagpole_direction = flagpole_info['direction']
            flagpole_strength = flagpole_info['strength']
            flagpole_length = flagpole_info['length']
            
            # 2. 检测旗面
            flag_start_idx = flagpole_end_idx
            flag_data = closes[flag_start_idx:]
            flag_highs = highs[flag_start_idx:]
            flag_lows = lows[flag_start_idx:]
            
            if len(flag_data) < 3:  # 旗面需要至少3个数据点
                return analysis
            
            flag_info = self._detect_flag_pattern(flag_data, flag_highs, flag_lows, flagpole_direction)
            
            if not flag_info['detected']:
                return analysis
            
            # 3. 检测突破
            latest_close = closes[-1]
            latest_high = highs[-1]
            latest_low = lows[-1]
            
            flag_upper_bound = flag_info['upper_bound']
            flag_lower_bound = flag_info['lower_bound']
            
            breakout_direction = 'none'
            breakout_strength = 0.0
            
            flag_range = flag_upper_bound - flag_lower_bound
            
            if latest_close > flag_upper_bound:
                breakout_direction = 'upward'
                breakout_strength = min(1.0, (latest_close - flag_upper_bound) / (flag_range * 0.2))
            elif latest_close < flag_lower_bound:
                breakout_direction = 'downward'
                breakout_strength = min(1.0, (flag_lower_bound - latest_close) / (flag_range * 0.2))
            
            # 4. 成交量分析
            volume_confirmation = False
            volume_declining_in_flag = False
            
            if volumes is not None:
                # 检查旗面期间成交量是否递减
                flagpole_volumes = volumes[max(0, flagpole_end_idx-5):flagpole_end_idx]
                flag_volumes = volumes[flag_start_idx:]
                
                if len(flagpole_volumes) > 0 and len(flag_volumes) > 2:
                    flagpole_avg_volume = np.mean(flagpole_volumes)
                    flag_avg_volume = np.mean(flag_volumes[:-2])  # 排除突破时的成交量
                    volume_declining_in_flag = flag_avg_volume < flagpole_avg_volume * 0.7
                    
                    # 检查突破时成交量是否放大
                    if breakout_direction != 'none' and len(flag_volumes) >= 2:
                        breakout_volume = np.mean(flag_volumes[-2:])
                        volume_confirmation = breakout_volume > flag_avg_volume * 1.5
            
            # 5. 确定旗形类型
            if flagpole_direction == 'up':
                pattern_type = 'bull_flag'
            else:
                pattern_type = 'bear_flag'
            
            # 更新分析结果
            analysis.update({
                'pattern_detected': True,
                'pattern_type': pattern_type,
                'breakout_direction': breakout_direction,
                'breakout_strength': breakout_strength,
                'flagpole_strength': flagpole_strength,
                'flag_quality': flag_info['quality'],
                'volume_confirmation': volume_confirmation,
                'volume_declining_in_flag': volume_declining_in_flag,
                'flag_duration': len(flag_data),
                'flagpole_length': flagpole_length,
                'flag_slope': flag_info['slope'],
                'formation_duration': len(recent_data)
            })
            
        except Exception as e:
            logger.warning(f"旗形形态分析失败: {e}")
        
        return analysis

    def _detect_flagpole(self, closes: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> Dict[str, Any]:
        """检测旗杆"""
        result = {
            'detected': False,
            'direction': None,
            'strength': 0.0,
            'length': 0.0,
            'end_index': 0
        }
        
        try:
            # 寻找强势移动（旗杆）
            min_pole_length = 5  # 旗杆最小长度
            min_move_ratio = 0.05  # 最小移动幅度
            
            # 从当前往前搜索旗杆
            for start_idx in range(len(closes) - min_pole_length - 3, max(0, len(closes) // 2), -1):
                for end_idx in range(start_idx + min_pole_length, len(closes) - 2):
                    
                    pole_closes = closes[start_idx:end_idx+1]
                    pole_highs = highs[start_idx:end_idx+1]
                    pole_lows = lows[start_idx:end_idx+1]
                    
                    # 计算移动幅度
                    start_price = pole_closes[0]
                    end_price = pole_closes[-1]
                    
                    move_ratio = abs(end_price - start_price) / start_price
                    
                    if move_ratio < min_move_ratio:
                        continue
                    
                    # 判断方向和强度
                    if end_price > start_price:  # 上升旗杆
                        direction = 'up'
                        # 检查是否持续上升
                        upward_moves = sum(1 for i in range(1, len(pole_closes)) 
                                         if pole_closes[i] > pole_closes[i-1])
                        strength = upward_moves / (len(pole_closes) - 1)
                        
                        # 检查新高
                        highest = np.max(pole_highs)
                        if pole_highs[-1] >= highest * 0.95:  # 接近最高点结束
                            strength += 0.2
                            
                    else:  # 下降旗杆
                        direction = 'down'
                        # 检查是否持续下降
                        downward_moves = sum(1 for i in range(1, len(pole_closes)) 
                                           if pole_closes[i] < pole_closes[i-1])
                        strength = downward_moves / (len(pole_closes) - 1)
                        
                        # 检查新低
                        lowest = np.min(pole_lows)
                        if pole_lows[-1] <= lowest * 1.05:  # 接近最低点结束
                            strength += 0.2
                    
                    # 旗杆强度要求
                    if strength > 0.6:  # 足够强的移动
                        result.update({
                            'detected': True,
                            'direction': direction,
                            'strength': min(1.0, strength),
                            'length': move_ratio,
                            'end_index': end_idx
                        })
                        return result
            
        except Exception:
            pass
        
        return result

    def _detect_flag_pattern(self, flag_closes: np.ndarray, flag_highs: np.ndarray, 
                           flag_lows: np.ndarray, flagpole_direction: str) -> Dict[str, Any]:
        """检测旗面形态"""
        result = {
            'detected': False,
            'quality': 0.0,
            'slope': 0.0,
            'upper_bound': 0.0,
            'lower_bound': 0.0
        }
        
        try:
            if len(flag_closes) < 3:
                return result
            
            # 计算旗面的趋势线
            x = np.arange(len(flag_closes))
            
            # 上边界（高点连线）
            upper_slope, upper_intercept = np.polyfit(x, flag_highs, 1)
            # 下边界（低点连线）
            lower_slope, lower_intercept = np.polyfit(x, flag_lows, 1)
            
            # 旗面平均斜率
            flag_slope = (upper_slope + lower_slope) / 2
            
            # 计算边界
            upper_bound = upper_intercept + upper_slope * (len(flag_closes) - 1)
            lower_bound = lower_intercept + lower_slope * (len(flag_closes) - 1)
            
            # 旗面质量评估
            quality = 0.0
            
            # 1. 斜率方向评估（与旗杆相反）
            if flagpole_direction == 'up' and flag_slope < 0:  # 上升旗杆后下倾旗面
                quality += 0.4
            elif flagpole_direction == 'down' and flag_slope > 0:  # 下降旗杆后上倾旗面
                quality += 0.4
            elif abs(flag_slope) < 0.001:  # 横向整理也可以
                quality += 0.3
            
            # 2. 价格在通道内运行评估
            in_channel_count = 0
            for i in range(len(flag_closes)):
                upper_level = upper_intercept + upper_slope * i
                lower_level = lower_intercept + lower_slope * i
                
                if lower_level <= flag_closes[i] <= upper_level:
                    in_channel_count += 1
            
            channel_ratio = in_channel_count / len(flag_closes)
            quality += channel_ratio * 0.4
            
            # 3. 波动收敛评估
            early_range = np.mean(flag_highs[:len(flag_highs)//2]) - np.mean(flag_lows[:len(flag_lows)//2])
            late_range = np.mean(flag_highs[len(flag_highs)//2:]) - np.mean(flag_lows[len(flag_lows)//2:])
            
            if early_range > 0 and late_range <= early_range:
                convergence_ratio = 1.0 - (late_range / early_range)
                quality += convergence_ratio * 0.2
            
            if quality > 0.5:  # 质量阈值
                result.update({
                    'detected': True,
                    'quality': quality,
                    'slope': flag_slope,
                    'upper_bound': upper_bound,
                    'lower_bound': lower_bound
                })
            
        except Exception:
            pass
        
        return result

    def _validate_signal_data(self, data: pd.DataFrame) -> bool:
        """
        验证信号生成所需的数据
        
        Args:
            data: 输入数据DataFrame
            
        Returns:
            bool: 数据是否有效
        """
        if not isinstance(data, pd.DataFrame):
            return False
        
        if data.empty:
            return False
        
        # 检查必需列
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            return False
        
        # 检查数据长度
        if len(data) < self.minimum_periods:
            return False
        
        return True

    def _get_default_signal(self, reason: str = "数据不足或无旗形形态") -> Dict[str, Any]:
        """
        生成默认的持有信号
        
        Args:
            reason: 默认信号的原因
            
        Returns:
            Dict[str, Any]: 默认信号字典
        """
        return {
            'signal_type': 'hold',
            'strength': 0.5,
            'confidence': 0.5,
            'timestamp': pd.Timestamp.now(),
            'reason': reason,
            'metadata': {
                'pattern_type': 'flag',
                'flag_detected': False,
                'flag_subtype': 'unknown',
                'breakout_direction': 'none',
                'breakout_strength': 0.0,
                'flagpole_strength': 0.0,
                'flag_quality': 0.0,
                'volume_confirmation': False,
                'flag_duration': 0,
                'flagpole_length': 0.0,
                'flag_slope': 0.0,
                'formation_duration': 0
            }
        }


class Pennant(AdvancedCandlestickPatterns):
    """三角旗形态识别"""

    def __init__(
        self, period: int = 30
    ):  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        super().__init__(period=period)  # 正确传递period参数
        self.name = "PENNANT"

    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        获取三角旗形形态交易信号
        
        三角旗形是重要的趋势继续形态，分为上升三角旗（继续看涨）和下降三角旗（继续看跌）。
        三角旗由旗杆（强势移动）和三角旗面（收敛三角形整理）组成，突破后继续原趋势方向。
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            Dict[str, Any]: 标准化的交易信号字典
        """
        # 数据验证
        if not self._validate_signal_data(data):
            return self._get_default_signal()
        
        # 确保有三角旗形形态列，如果没有则先计算
        pennant_columns = ['pennant', 'bull_pennant', 'bear_pennant']
        has_pennant_data = any(col in data.columns for col in pennant_columns)
        
        if not has_pennant_data:
            try:
                data = self.calculate(data)
            except Exception as e:
                logger.warning(f"计算三角旗形形态失败: {e}")
                return self._get_default_signal()
        
        if data.empty:
            return self._get_default_signal()
        
        # 初始化信号参数
        signal_type = "hold"
        strength = 0.5
        confidence = 0.6
        reason = "未检测到三角旗形形态"
        
        # 分析三角旗形形态和突破
        pennant_analysis = self._analyze_pennant_pattern(data)
        
        if pennant_analysis['pattern_detected']:
            pattern_type = pennant_analysis['pattern_type']
            breakout_direction = pennant_analysis['breakout_direction']
            flagpole_strength = pennant_analysis['flagpole_strength']
            
            # 根据三角旗类型和突破方向确定信号
            if pattern_type == 'bull_pennant' and breakout_direction == 'upward':
                # 上升三角旗向上突破（继续看涨）
                signal_type = "buy"
                strength = 0.85
                confidence = 0.9
                reason = "上升三角旗向上突破，强烈趋势继续信号"
            elif pattern_type == 'bear_pennant' and breakout_direction == 'downward':
                # 下降三角旗向下突破（继续看跌）
                signal_type = "sell"
                strength = 0.85
                confidence = 0.9
                reason = "下降三角旗向下突破，强烈趋势继续信号"
            elif pattern_type == 'bull_pennant' and breakout_direction == 'downward':
                # 上升三角旗向下突破（趋势反转警告）
                signal_type = "sell"
                strength = 0.7
                confidence = 0.75
                reason = "上升三角旗向下突破，趋势反转信号"
            elif pattern_type == 'bear_pennant' and breakout_direction == 'upward':
                # 下降三角旗向上突破（趋势反转警告）
                signal_type = "buy"
                strength = 0.7
                confidence = 0.75
                reason = "下降三角旗向上突破，趋势反转信号"
            else:
                # 形态完整但未突破
                signal_type = "hold"
                strength = 0.65
                confidence = 0.7
                reason = f"{pattern_type.replace('_', '')}三角旗形成，等待突破确认"
            
            # 旗杆强度评估（旗杆越强，信号越可靠）
            if flagpole_strength > 0.85:
                strength = min(1.0, strength + 0.1)
                confidence = min(1.0, confidence + 0.1)
                reason += "，旗杆强劲"
            elif flagpole_strength < 0.5:
                strength = max(0.3, strength - 0.1)
                confidence = max(0.4, confidence - 0.1)
                reason += "，旗杆疲弱"
            
            # 三角形收敛质量评估
            triangle_quality = pennant_analysis['triangle_quality']
            if triangle_quality > 0.8:
                strength = min(1.0, strength + 0.1)
                confidence = min(1.0, confidence + 0.1)
                reason += "，三角形收敛完美"
            elif triangle_quality < 0.5:
                strength = max(0.3, strength - 0.1)
                confidence = max(0.4, confidence - 0.1)
                reason += "，三角形收敛不佳"
            
            # 成交量确认
            if pennant_analysis['volume_confirmation']:
                strength = min(1.0, strength + 0.1)
                confidence = min(1.0, confidence + 0.1)
                reason += "，成交量确认"
            elif pennant_analysis['volume_declining_in_pennant']:
                # 三角旗期间成交量递减是正常的
                pass
            else:
                strength = max(0.3, strength - 0.05)
                confidence = max(0.4, confidence - 0.05)
                reason += "，成交量异常"
            
            # 突破质量评估
            if pennant_analysis['breakout_strength'] > 0.8:
                strength = min(1.0, strength + 0.1)
                confidence = min(1.0, confidence + 0.1)
                reason += "，突破强劲"
            elif pennant_analysis['breakout_strength'] < 0.3:
                strength = max(0.3, strength - 0.1)
                confidence = max(0.4, confidence - 0.1)
                reason += "，突破疲弱"
            
            # 三角旗持续时间评估（时间太长或太短都影响可靠性）
            pennant_duration = pennant_analysis.get('pennant_duration', 0)
            if 3 <= pennant_duration <= 15:  # 理想持续时间（三角旗比旗形更短）
                confidence = min(1.0, confidence + 0.05)
            elif pennant_duration < 2 or pennant_duration > 20:
                strength = max(0.3, strength - 0.1)
                confidence = max(0.4, confidence - 0.1)
            
            # 收敛程度评估
            convergence_ratio = pennant_analysis.get('convergence_ratio', 0)
            if convergence_ratio > 0.7:  # 高度收敛
                confidence = min(1.0, confidence + 0.1)
            elif convergence_ratio < 0.3:
                strength = max(0.3, strength - 0.1)
                confidence = max(0.4, confidence - 0.1)
        
        # 构建标准化信号字典
        metadata = {
            'pattern_type': 'pennant',
            'pennant_detected': pennant_analysis.get('pattern_detected', False),
            'pennant_subtype': pennant_analysis.get('pattern_type', 'unknown'),
            'breakout_direction': pennant_analysis.get('breakout_direction', 'none'),
            'breakout_strength': pennant_analysis.get('breakout_strength', 0.0),
            'flagpole_strength': pennant_analysis.get('flagpole_strength', 0.0),
            'triangle_quality': pennant_analysis.get('triangle_quality', 0.0),
            'volume_confirmation': pennant_analysis.get('volume_confirmation', False),
            'pennant_duration': pennant_analysis.get('pennant_duration', 0),
            'convergence_ratio': pennant_analysis.get('convergence_ratio', 0.0),
            'flagpole_length': pennant_analysis.get('flagpole_length', 0.0),
            'triangle_slope_upper': pennant_analysis.get('triangle_slope_upper', 0.0),
            'triangle_slope_lower': pennant_analysis.get('triangle_slope_lower', 0.0),
            'formation_duration': pennant_analysis.get('formation_duration', 0)
        }
        
        return {
            'signal_type': signal_type,
            'strength': max(0.0, min(1.0, strength)),
            'confidence': max(0.0, min(1.0, confidence)),
            'timestamp': pd.Timestamp.now(),
            'reason': reason,
            'metadata': metadata
        }

    def _analyze_pennant_pattern(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        分析三角旗形形态
        
        Args:
            data: 价格数据
            
        Returns:
            Dict: 包含形态分析结果的字典
        """
        analysis = {
            'pattern_detected': False,
            'pattern_type': 'unknown',
            'breakout_direction': 'none',
            'breakout_strength': 0.0,
            'flagpole_strength': 0.0,
            'triangle_quality': 0.0,
            'volume_confirmation': False,
            'volume_declining_in_pennant': False,
            'pennant_duration': 0,
            'convergence_ratio': 0.0,
            'flagpole_length': 0.0,
            'triangle_slope_upper': 0.0,
            'triangle_slope_lower': 0.0,
            'formation_duration': 0
        }
        
        try:
            if len(data) < self.period:
                return analysis
            
            recent_data = data.tail(self.period).copy()
            closes = recent_data['close'].values
            highs = recent_data['high'].values
            lows = recent_data['low'].values
            volumes = recent_data['volume'].values if 'volume' in recent_data.columns else None
            
            # 1. 检测旗杆（复用旗形的旗杆检测逻辑）
            flagpole_info = self._detect_flagpole_for_pennant(closes, highs, lows)
            
            if not flagpole_info['detected']:
                return analysis
            
            flagpole_end_idx = flagpole_info['end_index']
            flagpole_direction = flagpole_info['direction']
            flagpole_strength = flagpole_info['strength']
            flagpole_length = flagpole_info['length']
            
            # 2. 检测三角旗面（收敛三角形）
            triangle_start_idx = flagpole_end_idx
            triangle_data = closes[triangle_start_idx:]
            triangle_highs = highs[triangle_start_idx:]
            triangle_lows = lows[triangle_start_idx:]
            
            if len(triangle_data) < 4:  # 三角形需要至少4个数据点
                return analysis
            
            triangle_info = self._detect_triangle_pennant(triangle_data, triangle_highs, triangle_lows)
            
            if not triangle_info['detected']:
                return analysis
            
            # 3. 检测突破
            latest_close = closes[-1]
            latest_high = highs[-1]
            latest_low = lows[-1]
            
            triangle_upper_bound = triangle_info['upper_bound']
            triangle_lower_bound = triangle_info['lower_bound']
            
            breakout_direction = 'none'
            breakout_strength = 0.0
            
            triangle_range = triangle_upper_bound - triangle_lower_bound
            
            if latest_close > triangle_upper_bound:
                breakout_direction = 'upward'
                breakout_strength = min(1.0, (latest_close - triangle_upper_bound) / (triangle_range * 0.15))
            elif latest_close < triangle_lower_bound:
                breakout_direction = 'downward'
                breakout_strength = min(1.0, (triangle_lower_bound - latest_close) / (triangle_range * 0.15))
            
            # 4. 成交量分析
            volume_confirmation = False
            volume_declining_in_pennant = False
            
            if volumes is not None:
                # 检查三角旗期间成交量是否递减
                flagpole_volumes = volumes[max(0, flagpole_end_idx-5):flagpole_end_idx]
                triangle_volumes = volumes[triangle_start_idx:]
                
                if len(flagpole_volumes) > 0 and len(triangle_volumes) > 2:
                    flagpole_avg_volume = np.mean(flagpole_volumes)
                    triangle_avg_volume = np.mean(triangle_volumes[:-2])  # 排除突破时的成交量
                    volume_declining_in_pennant = triangle_avg_volume < flagpole_avg_volume * 0.6
                    
                    # 检查突破时成交量是否放大
                    if breakout_direction != 'none' and len(triangle_volumes) >= 2:
                        breakout_volume = np.mean(triangle_volumes[-2:])
                        volume_confirmation = breakout_volume > triangle_avg_volume * 1.8
            
            # 5. 确定三角旗类型
            if flagpole_direction == 'up':
                pattern_type = 'bull_pennant'
            else:
                pattern_type = 'bear_pennant'
            
            # 6. 计算收敛比率
            initial_range = np.max(triangle_highs[:len(triangle_highs)//2]) - np.min(triangle_lows[:len(triangle_lows)//2])
            final_range = triangle_upper_bound - triangle_lower_bound
            convergence_ratio = 1.0 - (final_range / initial_range) if initial_range > 0 else 0.0
            convergence_ratio = max(0.0, min(1.0, convergence_ratio))
            
            # 更新分析结果
            analysis.update({
                'pattern_detected': True,
                'pattern_type': pattern_type,
                'breakout_direction': breakout_direction,
                'breakout_strength': breakout_strength,
                'flagpole_strength': flagpole_strength,
                'triangle_quality': triangle_info['quality'],
                'volume_confirmation': volume_confirmation,
                'volume_declining_in_pennant': volume_declining_in_pennant,
                'pennant_duration': len(triangle_data),
                'convergence_ratio': convergence_ratio,
                'flagpole_length': flagpole_length,
                'triangle_slope_upper': triangle_info['upper_slope'],
                'triangle_slope_lower': triangle_info['lower_slope'],
                'formation_duration': len(recent_data)
            })
            
        except Exception as e:
            logger.warning(f"三角旗形形态分析失败: {e}")
        
        return analysis

    def _detect_flagpole_for_pennant(self, closes: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> Dict[str, Any]:
        """检测旗杆（三角旗形专用）"""
        result = {
            'detected': False,
            'direction': None,
            'strength': 0.0,
            'length': 0.0,
            'end_index': 0
        }
        
        try:
            # 三角旗的旗杆要求更严格
            min_pole_length = 6  # 旗杆最小长度
            min_move_ratio = 0.08  # 最小移动幅度（比旗形更高）
            
            # 从当前往前搜索旗杆
            for start_idx in range(len(closes) - min_pole_length - 4, max(0, len(closes) // 2), -1):
                for end_idx in range(start_idx + min_pole_length, len(closes) - 3):
                    
                    pole_closes = closes[start_idx:end_idx+1]
                    pole_highs = highs[start_idx:end_idx+1]
                    pole_lows = lows[start_idx:end_idx+1]
                    
                    # 计算移动幅度
                    start_price = pole_closes[0]
                    end_price = pole_closes[-1]
                    
                    move_ratio = abs(end_price - start_price) / start_price
                    
                    if move_ratio < min_move_ratio:
                        continue
                    
                    # 判断方向和强度
                    if end_price > start_price:  # 上升旗杆
                        direction = 'up'
                        # 检查持续性（要求更高）
                        upward_moves = sum(1 for i in range(1, len(pole_closes)) 
                                         if pole_closes[i] > pole_closes[i-1])
                        strength = upward_moves / (len(pole_closes) - 1)
                        
                        # 检查新高确认
                        highest = np.max(pole_highs)
                        if pole_highs[-1] >= highest * 0.98:  # 更严格的新高要求
                            strength += 0.2
                            
                    else:  # 下降旗杆
                        direction = 'down'
                        # 检查持续性（要求更高）
                        downward_moves = sum(1 for i in range(1, len(pole_closes)) 
                                           if pole_closes[i] < pole_closes[i-1])
                        strength = downward_moves / (len(pole_closes) - 1)
                        
                        # 检查新低确认
                        lowest = np.min(pole_lows)
                        if pole_lows[-1] <= lowest * 1.02:  # 更严格的新低要求
                            strength += 0.2
                    
                    # 三角旗的旗杆强度要求更高
                    if strength > 0.7:  # 更高的强度要求
                        result.update({
                            'detected': True,
                            'direction': direction,
                            'strength': min(1.0, strength),
                            'length': move_ratio,
                            'end_index': end_idx
                        })
                        return result
            
        except Exception:
            pass
        
        return result

    def _detect_triangle_pennant(self, triangle_closes: np.ndarray, triangle_highs: np.ndarray, 
                                triangle_lows: np.ndarray) -> Dict[str, Any]:
        """检测三角旗面（收敛三角形）"""
        result = {
            'detected': False,
            'quality': 0.0,
            'upper_slope': 0.0,
            'lower_slope': 0.0,
            'upper_bound': 0.0,
            'lower_bound': 0.0
        }
        
        try:
            if len(triangle_closes) < 4:
                return result
            
            # 计算三角形的上下趋势线
            x = np.arange(len(triangle_closes))
            
            # 上边界（高点连线）- 应该是下降的
            upper_slope, upper_intercept = np.polyfit(x, triangle_highs, 1)
            # 下边界（低点连线）- 应该是上升的
            lower_slope, lower_intercept = np.polyfit(x, triangle_lows, 1)
            
            # 计算最终边界
            final_x = len(triangle_closes) - 1
            upper_bound = upper_intercept + upper_slope * final_x
            lower_bound = lower_intercept + lower_slope * final_x
            
            # 三角形质量评估
            quality = 0.0
            
            # 1. 收敛性检查（上边界下降，下边界上升）
            if upper_slope < -0.001 and lower_slope > 0.001:  # 明显的收敛
                quality += 0.4
            elif upper_slope < 0 and lower_slope > 0:  # 轻微收敛
                quality += 0.2
            
            # 2. 价格在三角形内运行
            in_triangle_count = 0
            for i in range(len(triangle_closes)):
                upper_level = upper_intercept + upper_slope * i
                lower_level = lower_intercept + lower_slope * i
                
                if lower_level <= triangle_closes[i] <= upper_level:
                    in_triangle_count += 1
                
                # 检查高低点是否接近趋势线
                if abs(triangle_highs[i] - upper_level) <= abs(upper_level - lower_level) * 0.1:
                    quality += 0.05
                if abs(triangle_lows[i] - lower_level) <= abs(upper_level - lower_level) * 0.1:
                    quality += 0.05
            
            triangle_ratio = in_triangle_count / len(triangle_closes)
            quality += triangle_ratio * 0.3
            
            # 3. 收敛程度（三角形应该显著收敛）
            initial_range = triangle_highs[0] - triangle_lows[0]
            final_range = upper_bound - lower_bound
            
            if initial_range > 0:
                convergence = 1.0 - (final_range / initial_range)
                if convergence > 0.5:  # 显著收敛
                    quality += convergence * 0.3
            
            # 4. 时间长度适中（三角旗应该比较短）
            if 4 <= len(triangle_closes) <= 15:
                quality += 0.1
            
            if quality > 0.6:  # 质量阈值
                result.update({
                    'detected': True,
                    'quality': min(1.0, quality),
                    'upper_slope': upper_slope,
                    'lower_slope': lower_slope,
                    'upper_bound': upper_bound,
                    'lower_bound': lower_bound
                })
            
        except Exception:
            pass
        
        return result

    def _validate_signal_data(self, data: pd.DataFrame) -> bool:
        """
        验证信号生成所需的数据
        
        Args:
            data: 输入数据DataFrame
            
        Returns:
            bool: 数据是否有效
        """
        if not isinstance(data, pd.DataFrame):
            return False
        
        if data.empty:
            return False
        
        # 检查必需列
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            return False
        
        # 检查数据长度
        if len(data) < self.minimum_periods:
            return False
        
        return True

    def _get_default_signal(self, reason: str = "数据不足或无三角旗形形态") -> Dict[str, Any]:
        """
        生成默认的持有信号
        
        Args:
            reason: 默认信号的原因
            
        Returns:
            Dict[str, Any]: 默认信号字典
        """
        return {
            'signal_type': 'hold',
            'strength': 0.5,
            'confidence': 0.5,
            'timestamp': pd.Timestamp.now(),
            'reason': reason,
            'metadata': {
                'pattern_type': 'pennant',
                'pennant_detected': False,
                'pennant_subtype': 'unknown',
                'breakout_direction': 'none',
                'breakout_strength': 0.0,
                'flagpole_strength': 0.0,
                'triangle_quality': 0.0,
                'volume_confirmation': False,
                'pennant_duration': 0,
                'convergence_ratio': 0.0,
                'flagpole_length': 0.0,
                'triangle_slope_upper': 0.0,
                'triangle_slope_lower': 0.0,
                'formation_duration': 0
            }
        }
