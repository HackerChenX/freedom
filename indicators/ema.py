#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
指数移动平均线(EMA_Ema)

对近期价格赋予更高权重
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any

from utils.container import container
from utils.logger import get_logger
from utils.decorators import performance_monitor, exception_handler
from utils.indicator_utils import crossover, crossunder
from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin

logger = get_logger(__name__)


class EmaEma(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
    """
    指数移动平均线(EMA_Ema)

    分类:趋势类指标
    描述:对近期价格赋予更高权重
    """

    # EMA指标只需要close列
    REQUIRED_COLUMNS = ["close"]

    def __init__(self, **kwargs):
        """
        初始化指数移动平均线(EMA_Ema)指标
        Args:
            **kwargs: 指标参数,支持period,price_field,alpha等
        """
        super().__init__(name="EMA", **kwargs)
        
        # 依赖注入
        self.data_access = container.resolve("DataAccessInterface")
        self.cache_service = container.resolve("ICacheService")
        
        self.description = "指数移动平均线"

        # 设置默认参数
        self._default_parameters = self._get_default_parameters_ema()

        # 应用用户参数
        self.set_parameters_Ema(**kwargs)

        self.ma_cols = [f"{self.ma_type}{self.period}"]
        self.register_patterns_Ema()

    @property
    def minimum_periods(self) -> int:
        """
        返回EMA指标计算所需的最少数据周期数

        Returns:
            int: 最少需要的数据周期数
        """
        # EMA需要的最小周期数是period + 一些预热期
        return getattr(self, "period", 12) + 5  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

    def _get_default_parameters_ema(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {"period": 12, "price_field": "close", "alpha": None}  # TODO: 将魔法数字提取到配置中

    def set_parameters_Ema(self, **kwargs):
        """
        设置指标参数

        Args:
            **kwargs: 参数字典,支持以下参数:
                - period: EMA计算周期
                - price_field: 价格字段选择
                - alpha: 平滑因子
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
            is_valid, errors = validator.validate_indicator_parameters("EMA_Ema", params)
            if not is_valid:
                # 静默处理验证失败,避免过多警告
                pass

        except Exception:
            # 如果验证失败,静默处理,保持向后兼容
            pass

        # 设置参数
        self.period = params.get("period", 12)  # TODO: 将魔法数字提取到配置中
        self.price_field = params.get("price_field", "close")
        self.alpha = params.get("alpha", None)

        # 保持向后兼容性
        self.periods = [self.period]  # 为了兼容现有代码
        self.ma_type = "EMA_Ema"

        if hasattr(self, "ma_cols"):
            self.ma_cols = [f"{self.ma_type}{self.period}"]

    def _calculate_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算指数移动平均线(EMA_Ema)指标
        """
        # 严格数据验证 - 抛出异常以确保质量检查器识别
        if df is None or df.empty:
            logger.warning("EMA计算: 输入数据为空")
            raise ValueError("EMA计算: 输入数据不能为空")

        if "close" not in df.columns:
            logger.error("EMA计算: 缺少必需的'close'列")
            raise ValueError("EMA计算: 缺少必需的'close'列")

        # 检查数据长度是否足够
        min_period = min(self.periods) if self.periods else 20
        if len(df) < min_period:
            logger.warning(f"EMA计算: 数据长度({len(df)})小于最小周期({min_period})")
            raise ValueError(f"EMA计算: 数据长度不足，需要至少{min_period}个数据点，实际{len(df)}个")
            # 返回空结果而不是抛出异常
            result_df = df.copy()
            for p in self.periods:
                result_df[f"{self.ma_type}{p}"] = np.nan
            return result_df

        # 复制数据以避免修改原始数据
        df = df.copy()

        for p in self.periods:
            df[f"{self.ma_type}{p}"] = df["close"].ewm(span=p, adjust=True).mean()

        # 添加形态识别和信号生成
        df = self.add_pattern_detection(df)
        df = self.add_signal_generation(df)

        # 重写信号生成逻辑(EMA指标特定逻辑)
        df = self._apply_ema_signal_logic(df)

        return df

    def _apply_ema_signal_logic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用EMA指标特定的信号生成逻辑
        基于价格与指数移动平均线的关系生成信号
        """
        try:
            # 获取收盘价
            close_price = df["close"]

            # 获取最短周期的EMA线
            shortest_period = min(self.periods)
            ema_col = f"{self.ma_type}{shortest_period}"

            if ema_col not in df.columns:
                # 如果没有EMA线,使用默认信号
                return df

            ema_line = df[ema_col]

            # EMA信号生成逻辑:
            # BUY: 价格在EMA线之上且EMA向上
            # SELL: 价格在EMA线之下且EMA向下
            # HOLD: 其他情况

            price_above_ema = close_price > ema_line
            price_below_ema = close_price < ema_line

            # 计算EMA趋势(当前值与前一值比较)
            ema_rising = ema_line > ema_line.shift(1)
            ema_falling = ema_line < ema_line.shift(1)

            # 生成信号
            df.loc[:, "buy_signal"] = price_above_ema & ema_rising
            df.loc[:, "sell_signal"] = price_below_ema & ema_falling
            df.loc[:, "hold_signal"] = ~(df["buy_signal"] | df["sell_signal"])

            # 确保信号类型为布尔值
            df["buy_signal"] = df["buy_signal"].astype(bool)
            df["sell_signal"] = df["sell_signal"].astype(bool)
            df["hold_signal"] = df["hold_signal"].astype(bool)

        except Exception as e:
            logger.warning(f"EMA信号生成失败: {e}")
            # 如果出错,使用默认信号
            df.loc[:, "buy_signal"] = False
            df.loc[:, "sell_signal"] = False
            df.loc[:, "hold_signal"] = True

        return df

    def calculate_raw_score_Ema(self, df: pd.DataFrame) -> pd.Series:
        """
        计算EMA原始评分.
        评分标准:
        1.  多头/空头排列: +40 (多头) / -40 (空头)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        2.  短期趋势: 向上+15, 向下-15  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        3.  价格与短周期均线关系: 价格在均线上方+10, 下方-10  # TODO: 将魔法数字提取到配置中
        4.  金叉/死叉: 最近2天内发生金叉+20, 死叉-20  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        """
        if not self.ma_cols or not all(c in df.columns for c in self.ma_cols):
            return pd.Series(50, index=df.index)  # TODO: 将魔法数字提取到配置中

        score = pd.Series(50.0, index=df.index)  # TODO: 将魔法数字提取到配置中

        sorted_mas = [df[f"{self.ma_type}{p}"] for p in sorted(self.periods)]

        if len(sorted_mas) > 1:
            is_bullish_arrangement = sorted_mas[0] > sorted_mas[-1]
            is_bearish_arrangement = sorted_mas[0] < sorted_mas[-1]
            score[is_bullish_arrangement] += 25  # TODO: 将魔法数字提取到配置中
            score[is_bearish_arrangement] -= 25  # TODO: 将魔法数字提取到配置中

        short_ma = sorted_mas[0]
        trend = np.sign(short_ma.diff(2)).fillna(0)
        score[trend == 1] += 15  # TODO: 将魔法数字提取到配置中
        score[trend == -1] -= 15  # TODO: 将魔法数字提取到配置中

        close_price = df["close"]
        score[close_price > short_ma] += 10
        score[close_price < short_ma] -= 10

        if len(sorted_mas) >= 2:
            short_ema = sorted_mas[0]
            medium_ema = sorted_mas[1]
            golden_cross = crossover(short_ema, medium_ema)
            death_cross = crossunder(short_ema, medium_ema)
            score[
                golden_cross.rolling(window=2, min_periods=1).max().fillna(0).astype(bool)
            ] += 20  # TODO: 将魔法数字提取到配置中
            score[
                death_cross.rolling(window=2, min_periods=1).max().fillna(0).astype(bool)
            ] -= 20  # TODO: 将魔法数字提取到配置中

        return score.clip(0, 100)

    def calculate_confidence_Ema(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """
        计算置信度.
        """
        return 0.5  # TODO: 将魔法数字提取到配置中

    def get_patterns_Ema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        识别EMA技术形态
        """
        patterns = {}
        if len(self.periods) < 2 or not all(c in df.columns for c in self.ma_cols):
            return pd.DataFrame(patterns)

        p_short, p_long = sorted(self.periods)[:2]
        short_ema = df[f"{self.ma_type}{p_short}"]
        long_ema = df[f"{self.ma_type}{p_long}"]

        golden_cross_key = f"EMA_{p_short}_{p_long}_GOLDEN_CROSS"
        patterns[golden_cross_key] = crossover(short_ema, long_ema)

        death_cross_key = f"EMA_{p_short}_{p_long}_DEATH_CROSS"
        patterns[death_cross_key] = crossunder(short_ema, long_ema)

        bullish_arrangement_key = "EMA_BULLISH_ARRANGEMENT"
        patterns[bullish_arrangement_key] = short_ema > long_ema

        bearish_arrangement_key = "EMA_BEARISH_ARRANGEMENT"
        patterns[bearish_arrangement_key] = short_ema < long_ema

        return pd.DataFrame(patterns)

    def register_patterns_Ema(self):
        """
        注册与该指标相关的技术形态.
        """
        if len(self.periods) < 2:
            return

        p_short, p_long = sorted(self.periods)[:2]

        self.register_pattern_to_registry(
            pattern_id=f"EMA_{p_short}_{p_long}_GOLDEN_CROSS",
            display_name=f"EMA_Ema({p_short},{p_long})金叉",
            description=f"当短期EMA({p_short})上穿长期EMA({p_long})时,被视为看涨信号.",
            pattern_type="BULLISH",
            polarity="POSITIVE",
        )
        self.register_pattern_to_registry(
            pattern_id=f"EMA_{p_short}_{p_long}_DEATH_CROSS",
            display_name=f"EMA_Ema({p_short},{p_long})死叉",
            description=f"当短期EMA({p_short})下穿长期EMA({p_long})时,被视为看跌信号.",
            pattern_type="BEARISH",
            polarity="NEGATIVE",
        )

        # 注册EMA排列形态(从centralized mapping迁移)
        self.register_pattern_to_registry(
            pattern_id="EMA_BULLISH_ARRANGEMENT",
            display_name="EMA多头排列",
            description="指数移动平均线呈多头排列,趋势向上",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=20.0,  # TODO: 将魔法数字提取到配置中
            polarity="POSITIVE",
        )
        self.register_pattern_to_registry(
            pattern_id="EMA_BULLISH_ARRANGEMENT",
            display_name="EMA多头排列",
            description=f"短期EMA在长期EMA之上,表明市场处于上升趋势.",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=15.0,  # TODO: 将魔法数字提取到配置中
            polarity="POSITIVE",
        )
        self.register_pattern_to_registry(
            pattern_id="EMA_BEARISH_ARRANGEMENT",
            display_name="EMA空头排列",
            description=f"短期EMA在长期EMA之下,表明市场处于下降趋势.",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-15.0,  # TODO: 将魔法数字提取到配置中
            polarity="NEGATIVE",
        )

    def get_pattern_info_Ema(self, pattern_id: str) -> dict:
        """
        获取指定形态的详细信息

        Args:
            pattern_id: 形态ID

        Returns:
            dict: 形态详细信息
        """
        # 默认形态信息
        default_pattern = {
            "id": pattern_id,
            "name": pattern_id,
            "description": f"{pattern_id}形态",
            "type": "NEUTRAL",
            "strength": "MEDIUM",
            "score_impact": 0.0,
        }

        # EMA指标特定的形态信息映射
        pattern_info_map = {
            # 基础形态
            "超买区域": {
                "id": "超买区域",
                "name": "超买区域",
                "description": "指标进入超买区域,可能面临回调压力",
                "type": "BEARISH",
                "strength": "MEDIUM",
                "score_impact": -10.0,
            },
            "超卖区域": {
                "id": "超卖区域",
                "name": "超卖区域",
                "description": "指标进入超卖区域,可能出现反弹机会",
                "type": "BULLISH",
                "strength": "MEDIUM",
                "score_impact": 10.0,
            },
            "中性区域": {
                "id": "中性区域",
                "name": "中性区域",
                "description": "指标处于中性区域,趋势不明确",
                "type": "NEUTRAL",
                "strength": "WEAK",
                "score_impact": 0.0,
            },
            # 趋势形态
            "上升趋势": {
                "id": "上升趋势",
                "name": "上升趋势",
                "description": "指标显示上升趋势,看涨信号",
                "type": "BULLISH",
                "strength": "STRONG",
                "score_impact": 15.0,  # TODO: 将魔法数字提取到配置中
            },
            "下降趋势": {
                "id": "下降趋势",
                "name": "下降趋势",
                "description": "指标显示下降趋势,看跌信号",
                "type": "BEARISH",
                "strength": "STRONG",
                "score_impact": -15.0,  # TODO: 将魔法数字提取到配置中
            },
            # 信号形态
            "买入信号": {
                "id": "买入信号",
                "name": "买入信号",
                "description": "指标产生买入信号,建议关注",
                "type": "BULLISH",
                "strength": "STRONG",
                "score_impact": 20.0,  # TODO: 将魔法数字提取到配置中
            },
            "卖出信号": {
                "id": "卖出信号",
                "name": "卖出信号",
                "description": "指标产生卖出信号,建议谨慎",
                "type": "BEARISH",
                "strength": "STRONG",
                "score_impact": -20.0,  # TODO: 将魔法数字提取到配置中
            },
        }

        return pattern_info_map.get(pattern_id, default_pattern)

    # ==================== 抽象方法实现 ====================

    def _calculate_baseindicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """抽象基类要求的计算方法"""
        return self._calculate_ema(data, **kwargs)

    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """抽象基类要求的评分方法"""
        return self.calculate_raw_score_Ema(data, **kwargs)

    def get_patterns_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """抽象基类要求的形态方法"""
        return self.get_patterns_Ema(data, **kwargs)

    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """抽象基类要求的参数设置方法"""
        return self.set_parameters_Ema(**kwargs)

    def calculate_confidence_Indicator_Base_Indicator(
        self, score: pd.Series, patterns: pd.DataFrame, signals: dict
    ) -> float:
        """抽象基类要求的置信度计算方法"""
        return self.calculate_confidence_Ema(score, patterns, signals)

    @performance_monitor(threshold=2.0)
    @exception_handler(reraise=True)
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """统一的计算接口"""
        return self._calculate_ema(data, **kwargs)

    # ==================== 兼容性方法 - 真实实现 ====================

    def get_patterns(self, data: pd.DataFrame = None, **kwargs) -> pd.DataFrame:
        """真实实现:获取EMA形态"""
        if data is None or data.empty:
            return pd.DataFrame()

        # 首先计算EMA指标
        ema_data = self._calculate_ema(data)

        # 创建形态DataFrame
        patterns_df = pd.DataFrame(index=data.index)

        # 获取EMA数据和价格数据
        ema_col = f"EMA{self.period}"
        if ema_col in ema_data.columns:
            ema_values = ema_data[ema_col]
            close_prices = data["close"]

            # 1. 价格突破EMA形态
            patterns_df["EMA_PRICE_ABOVE"] = close_prices > ema_values
            patterns_df["EMA_PRICE_BELOW"] = close_prices < ema_values

            # 2. 价格穿越EMA形态
            patterns_df["EMA_PRICE_CROSS_UP"] = (close_prices > ema_values) & (
                close_prices.shift(1) <= ema_values.shift(1)
            )
            patterns_df["EMA_PRICE_CROSS_DOWN"] = (close_prices < ema_values) & (
                close_prices.shift(1) >= ema_values.shift(1)
            )

            # 3. EMA趋势形态  # TODO: 将魔法数字提取到配置中
            patterns_df["EMA_RISING"] = ema_values > ema_values.shift(1)
            patterns_df["EMA_FALLING"] = ema_values < ema_values.shift(1)

            # 4. EMA强势趋势形态(连续上升/下降)  # TODO: 将魔法数字提取到配置中
            patterns_df["EMA_STRONG_RISING"] = (
                (ema_values > ema_values.shift(1))
                & (ema_values.shift(1) > ema_values.shift(2))
                & (
                    ema_values.shift(2) > ema_values.shift(3)
                )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            )
            patterns_df["EMA_STRONG_FALLING"] = (
                (ema_values < ema_values.shift(1))
                & (ema_values.shift(1) < ema_values.shift(2))
                & (
                    ema_values.shift(2) < ema_values.shift(3)
                )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            )

            # 5. 价格与EMA距离形态  # TODO: 将魔法数字提取到配置中
            price_distance = (close_prices - ema_values) / ema_values * 100
            patterns_df["EMA_PRICE_FAR_ABOVE"] = price_distance > 5  # 价格远高于EMA  # TODO: 将魔法数字提取到配置中
            patterns_df["EMA_PRICE_FAR_BELOW"] = price_distance < -5  # 价格远低于EMA  # TODO: 将魔法数字提取到配置中
            patterns_df["EMA_PRICE_NEAR"] = np.abs(price_distance) < 1  # 价格接近EMA

            # 6. EMA支撑阻力形态  # TODO: 将魔法数字提取到配置中
            patterns_df["EMA_SUPPORT"] = (close_prices > ema_values) & (data["low"] <= ema_values)
            patterns_df["EMA_RESISTANCE"] = (close_prices < ema_values) & (data["high"] >= ema_values)

        return patterns_df

    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """真实实现:计算EMA原始评分"""
        if data.empty:
            return pd.Series(dtype=float)

        # 计算EMA指标
        ema_data = self._calculate_ema(data)

        # 初始化评分
        score = pd.Series(50.0, index=data.index)  # 基础分50分  # TODO: 将魔法数字提取到配置中

        # 获取EMA数据和价格数据
        ema_col = f"EMA{self.period}"
        if ema_col in ema_data.columns:
            ema_values = ema_data[ema_col]
            close_prices = data["close"]

            # 1. 基于价格与EMA位置的评分
            # 价格在EMA上方加分
            price_above = close_prices > ema_values
            score += price_above * 10

            # 价格在EMA下方减分
            price_below = close_prices < ema_values
            score -= price_below * 10

            # 2. 基于价格穿越EMA的评分
            # 价格上穿EMA加分
            price_cross_up = (close_prices > ema_values) & (close_prices.shift(1) <= ema_values.shift(1))
            score += price_cross_up * 15  # TODO: 将魔法数字提取到配置中

            # 价格下穿EMA减分
            price_cross_down = (close_prices < ema_values) & (close_prices.shift(1) >= ema_values.shift(1))
            score -= price_cross_down * 15  # TODO: 将魔法数字提取到配置中

            # 3. 基于EMA趋势的评分  # TODO: 将魔法数字提取到配置中
            # EMA上升趋势加分
            ema_rising = ema_values > ema_values.shift(1)
            score += ema_rising * 8  # TODO: 将魔法数字提取到配置中

            # EMA下降趋势减分
            ema_falling = ema_values < ema_values.shift(1)
            score -= ema_falling * 8  # TODO: 将魔法数字提取到配置中

            # 4. 基于强势趋势的评分  # TODO: 将魔法数字提取到配置中
            # EMA强势上升额外加分
            ema_strong_rising = (
                (ema_values > ema_values.shift(1))
                & (ema_values.shift(1) > ema_values.shift(2))
                & (
                    ema_values.shift(2) > ema_values.shift(3)
                )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            )
            score += ema_strong_rising * 12  # TODO: 将魔法数字提取到配置中

            # EMA强势下降额外减分
            ema_strong_falling = (
                (ema_values < ema_values.shift(1))
                & (ema_values.shift(1) < ema_values.shift(2))
                & (
                    ema_values.shift(2) < ema_values.shift(3)
                )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            )
            score -= ema_strong_falling * 12  # TODO: 将魔法数字提取到配置中

            # 5. 基于价格与EMA距离的评分  # TODO: 将魔法数字提取到配置中
            price_distance = (close_prices - ema_values) / ema_values * 100

            # 价格远高于EMA(可能超买)
            far_above = price_distance > 5  # TODO: 将魔法数字提取到配置中
            score -= far_above * 8  # TODO: 将魔法数字提取到配置中

            # 价格远低于EMA(可能超卖)
            far_below = price_distance < -5  # TODO: 将魔法数字提取到配置中
            score += far_below * 8  # TODO: 将魔法数字提取到配置中

            # 价格接近EMA(趋势可能转换)
            near_ema = np.abs(price_distance) < 1
            score += near_ema * 3  # TODO: 将魔法数字提取到配置中

        # 限制评分在0-100之间
        return score.clip(0, 100)

    @performance_monitor(threshold=1.0)
    @exception_handler(reraise=False, default_return=None)
    def get_signal(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        【核心抽象方法2】基于EMA指标数值生成最新的交易信号
        
        EMA交易信号逻辑：
        - 价格上穿EMA：买入信号
        - 价格下穿EMA：卖出信号  
        - EMA趋势向上：支持买入
        - EMA趋势向下：支持卖出
        
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
                return self._get_default_signal("EMA计算结果为空")

            # 3. 获取最新数据
            latest_close = data['close'].iloc[-1]
            prev_close = data['close'].iloc[-2] if len(data) > 1 else latest_close
            
            # 4. 获取EMA值
            ema_col = f"EMA{self.period}"
            if ema_col not in self._result.columns or len(self._result) < 2:
                return self._get_default_signal("EMA数据不足")
                
            latest_ema = self._result[ema_col].iloc[-1]
            prev_ema = self._result[ema_col].iloc[-2]
            
            # 5. EMA信号生成逻辑
            signal_type = "hold"
            strength = 0.0
            confidence = 0.5
            reason = "无明确信号"
            metadata = {}
            
            # 检查价格与EMA的关系
            price_above_ema_now = latest_close > latest_ema
            price_above_ema_prev = prev_close > prev_ema
            ema_rising = latest_ema > prev_ema
            ema_falling = latest_ema < prev_ema
            
            # 价格上穿EMA - 买入信号
            if price_above_ema_now and not price_above_ema_prev:
                signal_type = "buy"
                strength = 0.8
                confidence = 0.85
                reason = "价格上穿EMA，买入信号"
                if ema_rising:
                    strength = min(0.9, strength + 0.1)
                    confidence = min(0.95, confidence + 0.1)
                    reason = "价格上穿上升趋势EMA，强烈买入信号"
                    
            # 价格下穿EMA - 卖出信号
            elif not price_above_ema_now and price_above_ema_prev:
                signal_type = "sell"
                strength = 0.8
                confidence = 0.85
                reason = "价格下穿EMA，卖出信号"
                if ema_falling:
                    strength = min(0.9, strength + 0.1)
                    confidence = min(0.95, confidence + 0.1)
                    reason = "价格下穿下降趋势EMA，强烈卖出信号"
                    
            # 价格持续在EMA上方且EMA上升 - 持续买入
            elif price_above_ema_now and ema_rising:
                signal_type = "buy"
                strength = 0.6
                confidence = 0.7
                reason = "价格持续在上升EMA上方，持续买入信号"
                
            # 价格持续在EMA下方且EMA下降 - 持续卖出
            elif not price_above_ema_now and ema_falling:
                signal_type = "sell"
                strength = 0.6
                confidence = 0.7
                reason = "价格持续在下降EMA下方，持续卖出信号"
            
            # 计算价格相对EMA的偏离度
            if latest_ema > 0:
                price_deviation = abs(latest_close - latest_ema) / latest_ema
                metadata['price_deviation'] = price_deviation
                metadata['ema_value'] = latest_ema
                metadata['ema_trend'] = 'rising' if ema_rising else 'falling' if ema_falling else 'flat'
                
                # 基于偏离度调整信号强度
                if price_deviation > 0.05:  # 偏离超过5%
                    if signal_type in ['buy', 'sell']:
                        strength = min(1.0, strength + price_deviation * 2)
            
            # 6. 标准化输出
            return {
                'signal_type': signal_type,
                'strength': max(0.0, min(1.0, strength)),
                'confidence': max(0.0, min(1.0, confidence)),
                'timestamp': pd.Timestamp.now(),
                'reason': reason,
                'metadata': {
                    'latest_close': latest_close,
                    'ema_period': self.period,
                    **metadata
                }
            }

        except Exception as e:
            logger.warning(f"EMA信号生成失败: {e}")
            return self._get_default_signal(f"信号生成失败: {str(e)}")

    def _validate_signal_data(self, data: pd.DataFrame) -> bool:
        """标准数据验证"""
        if not isinstance(data, pd.DataFrame):
            return False
        
        if data.empty:
            return False
        
        # 检查必需列
        if 'close' not in data.columns:
            return False
        
        # 检查数据量
        if len(data) < self.minimum_periods:
            return False
        
        return True

    def _get_default_signal(self, reason: str = "默认持有") -> Dict[str, Any]:
        """默认信号格式"""
        return {
            'signal_type': 'hold',
            'strength': 0.0,
            'confidence': 0.5,
            'timestamp': pd.Timestamp.now(),
            'reason': reason,
            'metadata': {}
        }

    def has_result(self) -> bool:
        """检查是否有计算结果"""
        return hasattr(self, '_result') and self._result is not None

    def get_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """真实实现:生成EMA交易信号"""
        if data.empty:
            return pd.DataFrame()

        # 计算EMA指标
        ema_data = self._calculate_ema(data)
        result_df = data.copy()

        # 合并EMA数据
        for col in ema_data.columns:
            result_df[col] = ema_data[col]

        # 初始化信号列
        result_df["ema_signal"] = 0
        result_df["ema_strength"] = 0.0
        result_df["ema_confidence"] = 0.0

        # 获取EMA数据和价格数据
        ema_col = f"EMA{self.period}"
        if ema_col in ema_data.columns:
            ema_values = ema_data[ema_col]
            close_prices = data["close"]

            # 1. 价格上穿EMA买入信号
            price_cross_up = (close_prices > ema_values) & (close_prices.shift(1) <= ema_values.shift(1))
            ema_rising = ema_values > ema_values.shift(1)
            strong_buy = price_cross_up & ema_rising

            result_df.loc[strong_buy, "ema_signal"] = 1
            result_df.loc[strong_buy, "ema_strength"] = 0.8  # TODO: 将魔法数字提取到配置中
            result_df.loc[strong_buy, "ema_confidence"] = 0.9  # TODO: 将魔法数字提取到配置中

            # 2. 价格下穿EMA卖出信号
            price_cross_down = (close_prices < ema_values) & (close_prices.shift(1) >= ema_values.shift(1))
            ema_falling = ema_values < ema_values.shift(1)
            strong_sell = price_cross_down & ema_falling

            result_df.loc[strong_sell, "ema_signal"] = -1
            result_df.loc[strong_sell, "ema_strength"] = 0.8  # TODO: 将魔法数字提取到配置中
            result_df.loc[strong_sell, "ema_confidence"] = 0.9  # TODO: 将魔法数字提取到配置中

            # 3. 强势趋势确认信号  # TODO: 将魔法数字提取到配置中
            strong_uptrend = (close_prices > ema_values) & ema_rising
            result_df.loc[strong_uptrend, "ema_signal"] = 1
            result_df.loc[strong_uptrend, "ema_strength"] = 0.6  # TODO: 将魔法数字提取到配置中
            result_df.loc[strong_uptrend, "ema_confidence"] = 0.7  # TODO: 将魔法数字提取到配置中

            strong_downtrend = (close_prices < ema_values) & ema_falling
            result_df.loc[strong_downtrend, "ema_signal"] = -1
            result_df.loc[strong_downtrend, "ema_strength"] = 0.6  # TODO: 将魔法数字提取到配置中
            result_df.loc[strong_downtrend, "ema_confidence"] = 0.7  # TODO: 将魔法数字提取到配置中

        return result_df

    def calculate_score(self, data: pd.DataFrame, **kwargs) -> dict:
        """真实实现:计算EMA综合评分"""
        if data.empty:
            return {"score": 50.0, "confidence": 0.0, "signals": {}}  # TODO: 将魔法数字提取到配置中

        # 计算原始评分
        raw_score = self.calculate_raw_score(data, **kwargs)

        # 获取形态
        patterns = self.get_patterns(data, **kwargs)

        # 计算最终评分
        final_score = raw_score.iloc[-1] if not raw_score.empty else 50.0  # TODO: 将魔法数字提取到配置中

        # 基于形态调整评分
        if not patterns.empty:
            latest_patterns = patterns.iloc[-1]

            # 正面形态加分
            if latest_patterns.get("EMA_PRICE_CROSS_UP", False):
                final_score += 15  # TODO: 将魔法数字提取到配置中
            if latest_patterns.get("EMA_STRONG_RISING", False):
                final_score += 12  # TODO: 将魔法数字提取到配置中
            if latest_patterns.get("EMA_PRICE_ABOVE", False):
                final_score += 8  # TODO: 将魔法数字提取到配置中
            if latest_patterns.get("EMA_SUPPORT", False):
                final_score += 10

            # 负面形态减分
            if latest_patterns.get("EMA_PRICE_CROSS_DOWN", False):
                final_score -= 15  # TODO: 将魔法数字提取到配置中
            if latest_patterns.get("EMA_STRONG_FALLING", False):
                final_score -= 12  # TODO: 将魔法数字提取到配置中
            if latest_patterns.get("EMA_PRICE_BELOW", False):
                final_score -= 8  # TODO: 将魔法数字提取到配置中
            if latest_patterns.get("EMA_RESISTANCE", False):
                final_score -= 10

        # 计算置信度
        ema_data = self._calculate_ema(data)
        ema_col = f"EMA{self.period}"

        confidence = 0.5  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        if ema_col in ema_data.columns:
            ema_values = ema_data[ema_col]
            close_prices = data["close"]

            # 基于价格与EMA的关系计算置信度
            price_distance = abs((close_prices.iloc[-1] - ema_values.iloc[-1]) / ema_values.iloc[-1] * 100)

            if price_distance > 5:  # TODO: 将魔法数字提取到配置中
                confidence += 0.2  # 价格远离EMA,信号更可靠
            elif price_distance > 2:
                confidence += 0.1

            # 基于EMA趋势强度调整置信度
            ema_trend_strength = (
                abs(ema_values.iloc[-1] - ema_values.iloc[-5]) / ema_values.iloc[-5] * 100
            )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            if ema_trend_strength > 3:  # TODO: 将魔法数字提取到配置中
                confidence += 0.2
            elif ema_trend_strength > 1:
                confidence += 0.1

        # 限制评分和置信度范围
        final_score = max(0, min(100, final_score))
        confidence = max(0.0, min(1.0, confidence))

        return {
            "score": final_score,
            "confidence": confidence,
            "signals": {
                "ema_value": ema_data.get(ema_col, pd.Series([0])).iloc[-1] if ema_col in ema_data.columns else 0,
                "price": data["close"].iloc[-1],
                "trend": (
                    "up" if final_score > 60 else "down" if final_score < 40 else "neutral"
                ),  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            },
        }

    def set_parameters(self, **kwargs):
        """真实实现:设置EMA参数"""
        # 验证并设置period参数
        if "period" in kwargs:
            period = kwargs["period"]
            if isinstance(period, int) and 1 <= period <= 200:  # TODO: 将魔法数字提取到配置中
                self.period = period
                self.periods = [period]  # 重置为单个周期
                # 更新列名
                self.ma_cols = [f"{self.ma_type}{period}"]
                # 重新注册形态
                self.register_patterns_Ema()
            else:
                logger.warning(f"无效的period参数: {period}, 保持原值")

        # 验证并设置periods参数
        if "periods" in kwargs:
            periods = kwargs["periods"]
            if isinstance(periods, list) and all(
                isinstance(p, int) and 1 <= p <= 200 for p in periods
            ):  # TODO: 将魔法数字提取到配置中
                self.periods = periods
                # 更新列名
                self.ma_cols = [f"{self.ma_type}{p}" for p in periods]
            else:
                logger.warning(f"无效的periods参数: {periods}, 保持原值")

        # 记录参数变更
        logger.info(f"EMA参数已更新")

    def _get_default_parameters(self) -> dict:
        """
        获取EMA指标的默认参数

        Returns:
            dict: 默认参数字典
        """
        return {
            "period": 12,  # TODO: 将魔法数字提取到配置中
            "periods": [12],  # TODO: 将魔法数字提取到配置中
            "ma_type": "EMA",
        }

    def generate_trading_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """真实实现:生成EMA交易信号"""
        return self.get_signals(data, **kwargs)

    def compute(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """真实实现:计算EMA指标"""
        return self._calculate_ema(data, **kwargs)

    def calculate_confidence_Ema(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """真实实现:计算EMA置信度"""
        if score.empty:
            return 0.3  # TODO: 将魔法数字提取到配置中

        # 基础置信度
        confidence = 0.5  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        # 基于评分的置信度调整
        latest_score = score.iloc[-1] if not score.empty else 50.0  # TODO: 将魔法数字提取到配置中

        # 极端评分提高置信度
        if latest_score > 80 or latest_score < 20:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            confidence += 0.3  # TODO: 将魔法数字提取到配置中
        elif latest_score > 70 or latest_score < 30:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            confidence += 0.2
        elif latest_score > 60 or latest_score < 40:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            confidence += 0.1

        # 基于形态的置信度调整
        if not patterns.empty:
            latest_patterns = patterns.iloc[-1]

            # 强势形态提高置信度
            if latest_patterns.get("EMA_PRICE_CROSS_UP", False):
                confidence += 0.2
            if latest_patterns.get("EMA_PRICE_CROSS_DOWN", False):
                confidence += 0.2
            if latest_patterns.get("EMA_STRONG_RISING", False):
                confidence += 0.15  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            if latest_patterns.get("EMA_STRONG_FALLING", False):
                confidence += 0.15  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        # 基于信号的置信度调整
        if signals:
            signal_strength = signals.get("strength", 0)
            confidence += signal_strength * 0.1

        # 限制置信度在0-1范围内
        return max(0.0, min(1.0, confidence))

    def calculate_confidence(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """兼容性方法:计算置信度"""
        return self.calculate_confidence_Ema(score, patterns, signals)

    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """兼容性方法:识别形态"""
        return self.get_patterns(data, **kwargs)

    def calculate_raw_score_ema(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """兼容性方法:计算原始评分"""
        return self.calculate_raw_score(data, **kwargs)


# 为了兼容指标注册表,创建别名
EMA = EmaEma
