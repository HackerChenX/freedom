from utils.container import container
import pandas as pd
import numpy as np
from typing import List, Dict, Any

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin
from utils.indicator_utils import crossover, crossunder
from indicators.pattern_registry import PatternTypePatternRegistry
from utils.logger import get_logger


# 使用字符串替代Pattern_type枚举
class Pattern_type:
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"


logger = get_logger(__name__)


class MaMa(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
    """
    移动平均线(MA_Ma)
    分类:趋势类指标
    描述:计算价格的简单移动平均.
    """

    # MA指标只需要close列
    REQUIRED_COLUMNS = ["close"]

    def __init__(self, **kwargs):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        """
        初始化移动平均线(MA_Ma)指标
        Args:
            **kwargs: 指标参数,支持period,price_field等
        """
        super().__init__()
        self.name = "MA_Ma"
        self.description = "移动平均线"

        # 设置默认参数
        self._default_parameters = self._get_default_parameters()

        # 应用用户参数
        self.set_parameters_Ma(**kwargs)

        # ma_cols在set_parameters_Ma中设置
        self.register_patterns_Ma()

    @property
    def minimum_periods(self) -> int:
        """返回计算指标所需的最小周期数"""
        # 返回最大周期数作为最小周期要求
        periods = self._default_parameters.get("periods", [20])  # TODO: 将魔法数字提取到配置中
        main_period = self._default_parameters.get("period", 20)  # TODO: 将魔法数字提取到配置中
        return max(max(periods), main_period)

    def _get_default_parameters(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {
            "periods": [
                5,
                10,
                20,
                60,
            ],  # 标准多周期MA  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            "period": 20,  # 主要周期保持兼容性  # TODO: 将魔法数字提取到配置中
            "price_field": "close",
        }

    def set_parameters_Ma(self, **kwargs):
        """
        设置指标参数

        Args:
            **kwargs: 参数字典,支持以下参数:
                - period: 移动平均线周期
                - ma_type: MA类型 ('SMA', 'EMA', 'WMA')
                - price_field: 价格字段选择
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
            is_valid, errors = validator.validate_indicator_parameters("MA_Ma", params)
            if not is_valid:
                # 静默处理验证失败,避免过多警告
                pass
        except Exception:
            # 如果验证器模块有问题,静默处理
            params = self._default_parameters.copy()
            params.update(kwargs)

        # 设置参数
        self.period = params.get("period", 20)  # TODO: 将魔法数字提取到配置中
        self.periods = params.get(
            "periods", [5, 10, 20, 60]
        )  # 多周期支持  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        self.price_field = params.get("price_field", "close")
        self.ma_type_param = params.get("ma_type", "SMA")  # 支持SMA,EMA,WMA

        # 确保主要周期在periods列表中
        if self.period not in self.periods:
            self.periods.append(self.period)

        self.ma_type = "MA"  # 更标准的命名

        # 设置MA列名
        self.ma_cols = [f"{self.ma_type}{p}" for p in self.periods]

    def _calculate_ma(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算移动平均线(支持SMA,EMA,WMA)
        """
        # 边界条件检查
        if data is None or data.empty:
            return pd.DataFrame()

        if len(data) == 0:
            return pd.DataFrame()

        # 从原始数据开始,确保保留所有基础列
        result_df = data.copy()

        # 确保close列存在
        if "close" not in data.columns:
            raise ValueError("数据中缺少'close'列")

        close_series = data["close"]

        # 处理close列包含NaN的情况
        if close_series.isna().all():
            # 如果所有值都是NaN,返回带有NaN的MA列的结果
            result_df["ma"] = np.nan
            for p in self.periods:
                result_df[f"{self.ma_type}{p}"] = np.nan
            return result_df

        # 处理close列不是Series的情况
        if not isinstance(close_series, pd.Series):
            # 获取close列的值
            close_values = close_series.values if hasattr(close_series, "values") else close_series

            # 如果是多维数组,展平它
            if hasattr(close_values, "flatten"):
                close_values = close_values.flatten()

            # 确保数据长度与索引长度匹配
            expected_length = len(data.index)
            if len(close_values) != expected_length:
                # 如果长度不匹配,只取需要的长度
                if len(close_values) > expected_length:
                    close_values = close_values[:expected_length]
                else:
                    # 如果数据不足,用NaN填充
                    close_values = np.pad(
                        close_values, (0, expected_length - len(close_values)), constant_values=np.nan
                    )

            # 创建正确的Series
            close_series = pd.Series(close_values, index=data.index)

        # 计算移动平均线
        for p in self.periods:
            ma_values = self._calculate_single_ma(close_series, p, self.ma_type_param)
            result_df[f"{self.ma_type}{p}"] = ma_values

        # 添加主要周期的ma列(用于测试兼容性)
        main_ma = self._calculate_single_ma(close_series, self.period, self.ma_type_param)
        result_df["ma"] = main_ma

        # 添加常用别名
        if "MA5" in result_df.columns:
            result_df["ma5"] = result_df["MA5"]
        if "MA10" in result_df.columns:
            result_df["ma10"] = result_df["MA10"]
        if "MA20" in result_df.columns:
            result_df["ma20"] = result_df["MA20"]
        if "MA60" in result_df.columns:
            result_df["ma60"] = result_df["MA60"]

        # 添加形态识别和信号生成
        result_df = self.add_pattern_detection(result_df)
        result_df = self.add_signal_generation(result_df)

        # 重写信号生成逻辑(MA指标特定逻辑)
        result_df = self._apply_ma_signal_logic(result_df)

        return result_df

    def _calculate_single_ma(self, series: pd.Series, period: int, ma_type: str) -> pd.Series:
        """
        计算单一周期的移动平均线

        Args:
            series: 价格序列
            period: 周期
            ma_type: MA类型 ('SMA', 'EMA', 'WMA')

        Returns:
            计算后的MA序列
        """
        if ma_type.upper() == "SMA":
            # 简单移动平均
            return series.rolling(window=period, min_periods=period).mean()

        elif ma_type.upper() == "EMA":
            # 指数移动平均
            return series.ewm(span=period, adjust=False).mean()

        elif ma_type.upper() == "WMA":
            # 加权移动平均
            return self._calculate_wma(series, period)

        else:
            # 默认使用SMA
            return series.rolling(window=period, min_periods=period).mean()

    def _calculate_wma(self, series: pd.Series, period: int) -> pd.Series:
        """
        计算加权移动平均线(WMA)

        Args:
            series: 价格序列
            period: 周期

        Returns:
            WMA序列
        """
        weights = np.arange(1, period + 1)
        weight_sum = weights.sum()

        def wma_func(x):
            if len(x) < period:
                return np.nan
            return np.dot(x[-period:], weights) / weight_sum

        return series.rolling(window=period, min_periods=period).apply(wma_func, raw=True)

    def _apply_ma_signal_logic(self, result_df: pd.DataFrame) -> pd.DataFrame:
        """
        应用MA指标特定的信号生成逻辑
        基于价格与移动平均线的关系生成信号
        """
        try:
            # 获取收盘价
            close_price = result_df["close"]

            # 获取主要MA线(使用设定的周期)
            ma_col = f"{self.ma_type}{self.period}"
            if ma_col not in result_df.columns:
                # 如果没有主要MA线,使用默认信号
                return result_df

            ma_line = result_df[ma_col]

            # MA信号生成逻辑:
            # BUY: 价格在MA线之上
            # SELL: 价格在MA线之下
            # HOLD: 价格接近MA线(±1%范围内)

            price_above_ma = close_price > ma_line
            price_below_ma = close_price < ma_line
            price_near_ma = (abs(close_price - ma_line) / ma_line) <= 0.01  # 1%范围内

            # 生成信号
            result_df.loc[:, "buy_signal"] = price_above_ma & ~price_near_ma
            result_df.loc[:, "sell_signal"] = price_below_ma & ~price_near_ma
            result_df.loc[:, "hold_signal"] = price_near_ma

            # 确保信号类型为布尔值
            result_df["buy_signal"] = result_df["buy_signal"].astype(bool)
            result_df["sell_signal"] = result_df["sell_signal"].astype(bool)
            result_df["hold_signal"] = result_df["hold_signal"].astype(bool)

        except Exception as e:
            logger.warning(f"MA信号生成失败: {e}")
            # 如果出错,使用默认信号
            result_df.loc[:, "buy_signal"] = False
            result_df.loc[:, "sell_signal"] = False
            result_df.loc[:, "hold_signal"] = True

        return result_df

    def get_signal(self, data: pd.DataFrame, **kwargs) -> str:
        """
        获取MA信号 - 实现抽象方法

        Returns:
            str: BUY, SELL, 或 HOLD
        """
        try:
            if not self.has_result():
                self.calculate(data, **kwargs)

            if self._result is None or len(self._result) == 0:
                return "HOLD"

            # 获取最新的买卖信号
            if 'buy_signal' in self._result.columns and self._result['buy_signal'].iloc[-1]:
                return "BUY"
            elif 'sell_signal' in self._result.columns and self._result['sell_signal'].iloc[-1]:
                return "SELL"
            else:
                return "HOLD"

        except Exception as e:
            logger.warning(f"MA信号获取失败: {e}")
            return "HOLD"

    def has_result(self) -> bool:
        """检查是否有计算结果"""
        return hasattr(self, '_result') and self._result is not None

    def calculate_raw_score_Ma(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算MA原始评分.
        """
        # 确保已计算MA指标
        if not self.has_result():
            self.calculate(data, **kwargs)

        if self._result is None:
            return pd.Series(
                50.0, index=data.index
            )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        # 检查MA列是否存在
        if not self.ma_cols or not all(c in self._result.columns for c in self.ma_cols):
            return pd.Series(
                50.0, index=data.index
            )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        score = pd.Series(50.0, index=data.index)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        sorted_mas = [self._result[f"{self.ma_type}{p}"] for p in sorted(self.periods)]

        if len(sorted_mas) > 1:
            is_bullish_arrangement = sorted_mas[0] > sorted_mas[-1]
            is_bearish_arrangement = sorted_mas[0] < sorted_mas[-1]
            score[is_bullish_arrangement] += 25  # TODO: 将魔法数字提取到配置中
            score[is_bearish_arrangement] -= 25  # TODO: 将魔法数字提取到配置中

        short_ma = sorted_mas[0]
        trend = np.sign(short_ma.diff(2)).fillna(0)
        score[trend == 1] += 15  # TODO: 将魔法数字提取到配置中
        score[trend == -1] -= 15  # TODO: 将魔法数字提取到配置中

        close_price = data["close"]
        score[close_price > short_ma] += 10
        score[close_price < short_ma] -= 10

        if len(sorted_mas) >= 2:
            short_ma_series = sorted_mas[0]
            medium_ma_series = sorted_mas[1]
            golden_cross = crossover(short_ma_series, medium_ma_series)
            death_cross = crossunder(short_ma_series, medium_ma_series)
            score[
                golden_cross.rolling(window=2, min_periods=1).max().fillna(0).astype(bool)
            ] += 20  # TODO: 将魔法数字提取到配置中
            score[
                death_cross.rolling(window=2, min_periods=1).max().fillna(0).astype(bool)
            ] -= 20  # TODO: 将魔法数字提取到配置中

        return score.clip(0, 100)

    def calculate_confidence_Ma(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """
        计算MA指标的置信度

        Args:
            score: 得分序列
            patterns: 检测到的形态Data_frame
            signals: 生成的信号字典

        Returns:
            float: 置信度分数 (0-1)
        """
        if score.empty:
            return 0.5  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

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
        if isinstance(patterns, pd.DataFrame) and not patterns.empty:
            # 统计最近几个周期的形态数量
            try:
                numeric_cols = patterns.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    recent_data = (
                        patterns[numeric_cols].iloc[-5:] if len(patterns) >= 5 else patterns[numeric_cols]
                    )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                    recent_patterns = recent_data.sum().sum()
                    confidence += min(recent_patterns * 0.05, 0.2)  # TODO: 将魔法数字提取到配置中
            except:
                pass

        # 3. 基于评分稳定性的置信度  # TODO: 将魔法数字提取到配置中
        if len(score) >= 5:  # TODO: 将魔法数字提取到配置中
            recent_scores = score.iloc[-5:]  # TODO: 将魔法数字提取到配置中
            score_stability = 1.0 - (recent_scores.std() / 50.0)  # TODO: 将魔法数字提取到配置中
            confidence += score_stability * 0.1

        # 4. 基于MA排列的置信度  # TODO: 将魔法数字提取到配置中
        if hasattr(self, "_result") and self._result is not None and not self._result.empty:
            try:
                # 检查MA排列的一致性
                ma_cols = [col for col in self._result.columns if col.startswith(self.ma_type)]
                if len(ma_cols) >= 2:
                    # 获取最新的MA值
                    latest_mas = []
                    for col in ma_cols:
                        if not self._result[col].empty:
                            latest_ma = (
                                self._result[col].dropna().iloc[-1] if not self._result[col].dropna().empty else None
                            )
                            if latest_ma is not None:
                                latest_mas.append(latest_ma)

                    if len(latest_mas) >= 2:
                        # 检查MA排列是否有序
                        sorted_mas = sorted(latest_mas)
                        if latest_mas == sorted_mas or latest_mas == sorted_mas[::-1]:
                            confidence += 0.1  # MA排列有序,增加置信度
            except:
                pass

        return min(confidence, 1.0)

    def get_patterns_Ma(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        识别MA技术形态
        - 金叉/死叉:基于最短和次短周期均线.
        - 多头/空头排列:基于最短和最长周期均线.
        """
        patterns = {}
        if len(self.periods) < 2 or not all(c in data.columns for c in self.ma_cols):
            return pd.DataFrame(patterns, index=data.index)

        sorted_periods = sorted(self.periods)
        p_short, p_medium = sorted_periods[0], sorted_periods[1]
        p_long = sorted_periods[-1]

        short_ma = data[f"{self.ma_type}{p_short}"]
        medium_ma = data[f"{self.ma_type}{p_medium}"]
        long_ma = data[f"{self.ma_type}{p_long}"]

        patterns[f"MA_{p_short}_{p_medium}_GOLDEN_CROSS"] = crossover(short_ma, medium_ma)
        patterns[f"MA_{p_short}_{p_medium}_DEATH_CROSS"] = crossunder(short_ma, medium_ma)
        patterns["MA_BULLISH_ARRANGEMENT"] = short_ma > long_ma
        patterns["MA_BEARISH_ARRANGEMENT"] = short_ma < long_ma

        return pd.DataFrame(patterns)

    def register_patterns_Ma(self):
        """
        注册与该指标相关的技术形态.
        """
        if len(self.periods) < 2:
            return

        sorted_periods = sorted(self.periods)
        p_short, p_medium = sorted_periods[0], sorted_periods[1]
        p_long = sorted_periods[-1]

        self.register_pattern_to_registry(
            pattern_id=f"MA_{p_short}_{p_medium}_GOLDEN_CROSS",
            display_name=f"MA({p_short},{p_medium})金叉",
            description=f"当短期MA({p_short})上穿中期MA({p_medium})时,被视为看涨信号.",
            pattern_type="BULLISH",
            polarity="POSITIVE",
        )
        self.register_pattern_to_registry(
            pattern_id=f"MA_{p_short}_{p_medium}_DEATH_CROSS",
            display_name=f"MA({p_short},{p_medium})死叉",
            description=f"当短期MA({p_short})下穿中期MA({p_medium})时,被视为看跌信号.",
            pattern_type="BEARISH",
            polarity="NEGATIVE",
        )

        # 注册MA排列形态
        self.register_pattern_to_registry(
            pattern_id="MA_BULLISH_ARRANGEMENT",
            display_name="均线多头排列",
            description="短期均线在长期均线之上,形成多头排列",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=20.0,
            polarity="POSITIVE",
        )

        self.register_pattern_to_registry(
            pattern_id="MA_BEARISH_ARRANGEMENT",
            display_name="均线空头排列",
            description="短期均线在长期均线之下,形成空头排列",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-20.0,  # TODO: 将魔法数字提取到配置中
            polarity="NEGATIVE",
        )

        # 注册MA支撑阻力形态
        self.register_pattern_to_registry(
            pattern_id="MA_SUPPORT",
            display_name="均线支撑",
            description="价格在均线获得支撑",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=15.0,  # TODO: 将魔法数字提取到配置中
            polarity="POSITIVE",
        )

        self.register_pattern_to_registry(
            pattern_id="MA_RESISTANCE",
            display_name="均线阻力",
            description="价格在均线遇到阻力",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-15.0,  # TODO: 将魔法数字提取到配置中
            polarity="NEGATIVE",
        )
        self.register_pattern_to_registry(
            pattern_id="MA_BULLISH_ARRANGEMENT",
            display_name="MA多头排列",
            description=f"短期MA({p_short})在长期MA({p_long})之上,表明市场处于强劲上升趋势.",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=25.0,  # TODO: 将魔法数字提取到配置中
            polarity="POSITIVE",
        )
        self.register_pattern_to_registry(
            pattern_id="MA_BEARISH_ARRANGEMENT",
            display_name="MA空头排列",
            description=f"短期MA({p_short})在长期MA({p_long})之下,表明市场处于强劲下降趋势.",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-25.0,  # TODO: 将魔法数字提取到配置中
            polarity="NEGATIVE",
        )

    def get_pattern_info_Ma(self, pattern_id: str) -> dict:
        """
        获取指定形态的详细信息

        Args:
            pattern_id: 形态ID

        Returns:
            dict: 形态详细信息
        """
        if len(self.periods) < 2:
            return {
                "id": pattern_id,
                "name": "均线趋势分析",
                "description": f"基于移动平均线的趋势分析: {pattern_id}",
                "type": "NEUTRAL",
                "strength": "WEAK",
                "score_impact": 0.0,
            }

        sorted_periods = sorted(self.periods)
        p_short, p_medium = sorted_periods[0], sorted_periods[1]
        p_long = sorted_periods[-1]

        pattern_info_map = {
            f"MA_{p_short}_{p_medium}_GOLDEN_CROSS": {
                "id": f"MA_{p_short}_{p_medium}_GOLDEN_CROSS",
                "name": f"MA_Ma({p_short},{p_medium})金叉",
                "description": f"短期MA({p_short})上穿中期MA({p_medium}),看涨信号",
                "type": "BULLISH",
                "strength": "STRONG",
                "score_impact": 20.0,  # TODO: 将魔法数字提取到配置中
            },
            f"MA_{p_short}_{p_medium}_DEATH_CROSS": {
                "id": f"MA_{p_short}_{p_medium}_DEATH_CROSS",
                "name": f"MA_Ma({p_short},{p_medium})死叉",
                "description": f"短期MA({p_short})下穿中期MA({p_medium}),看跌信号",
                "type": "BEARISH",
                "strength": "STRONG",
                "score_impact": -20.0,  # TODO: 将魔法数字提取到配置中
            },
            "MA_BULLISH_ARRANGEMENT": {
                "id": "MA_BULLISH_ARRANGEMENT",
                "name": "MA多头排列",
                "description": f"短期MA({p_short})在长期MA({p_long})之上,强劲上升趋势",
                "type": "BULLISH",
                "strength": "MEDIUM",
                "score_impact": 25.0,  # TODO: 将魔法数字提取到配置中
            },
            "MA_BEARISH_ARRANGEMENT": {
                "id": "MA_BEARISH_ARRANGEMENT",
                "name": "MA空头排列",
                "description": f"短期MA({p_short})在长期MA({p_long})之下,强劲下降趋势",
                "type": "BEARISH",
                "strength": "MEDIUM",
                "score_impact": -25.0,  # TODO: 将魔法数字提取到配置中
            },
        }

        return pattern_info_map.get(
            pattern_id,
            {
                "id": pattern_id,
                "name": "均线趋势分析",
                "description": f"基于移动平均线的趋势分析: {pattern_id}",
                "type": "NEUTRAL",
                "strength": "WEAK",
                "score_impact": 0.0,
            },
        )

    # ========================= 抽象方法实现 =========================
    def _calculate_baseindicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return self._calculate_ma(data, **kwargs)

    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        return self.calculate_raw_score(data, **kwargs)

    def get_patterns_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return self.get_patterns(data, **kwargs)

    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        return self.set_parameters(**kwargs)

    def calculate_confidence_Indicator_Base_Indicator(
        self, score: pd.Series, patterns: pd.DataFrame, signals: dict
    ) -> float:
        return self.calculate_confidence(score, patterns, signals)

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算移动平均线指标

        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 额外参数

        Returns:
            包含MA计算结果的DataFrame

        Raises:
            ValueError: 当输入数据无效时
        """
        # 数据验证
        if data is None:
            raise ValueError("输入数据不能为None")

        if data.empty:
            raise ValueError("输入数据不能为空")

        if not isinstance(data, pd.DataFrame):
            raise ValueError("输入数据必须是pandas DataFrame")

        if "close" not in data.columns:
            raise ValueError("数据中缺少必需的'close'列")

        # 检查数据长度
        if len(data) < 1:
            raise ValueError("数据长度不足,至少需要1个数据点")

        # 检查close列数据类型
        try:
            # 尝试转换为数值类型
            close_values = pd.to_numeric(data["close"], errors="coerce")
            if close_values.isna().all():
                raise ValueError("close列包含无效的数值数据")
        except Exception as e:
            raise ValueError(f"close列数据类型无效: {str(e)}")

        try:
            return self._calculate_ma(data, **kwargs)
        except Exception as e:
            raise ValueError(f"MA计算失败: {str(e)}")

    # ========================= 兼容性方法 =========================
    def get_patterns(self, data: pd.DataFrame = None, **kwargs) -> pd.DataFrame:
        """
        获取MA形态识别结果

        Returns:
            pd.DataFrame: 形态识别结果,包含各种MA形态
        """
        if data is None and hasattr(self, "_result") and self._result is not None:
            data_to_use = self._result
        else:
            data_to_use = self.calculate(data, **kwargs) if data is not None else pd.DataFrame()

        if data_to_use.empty:
            return pd.DataFrame()

        patterns = pd.DataFrame(index=data_to_use.index)

        # 检查是否有MA列
        ma_cols = [col for col in data_to_use.columns if col.startswith("SMA") or col.startswith("MA")]
        if ma_cols:
            # 添加基本形态
            patterns["MA_UPTREND"] = False
            patterns["MA_DOWNTREND"] = False
            patterns["MA_BULLISH_ARRANGEMENT"] = False
            patterns["MA_BEARISH_ARRANGEMENT"] = False

            # 如果有多个MA,检测多头/空头排列
            if len(ma_cols) >= 2:
                ma_short = data_to_use[ma_cols[0]]
                ma_long = data_to_use[ma_cols[-1]]

                patterns["MA_BULLISH_ARRANGEMENT"] = ma_short > ma_long
                patterns["MA_BEARISH_ARRANGEMENT"] = ma_short < ma_long
                patterns["MA_UPTREND"] = ma_short > ma_short.shift(3)  # TODO: 将魔法数字提取到配置中
                patterns["MA_DOWNTREND"] = ma_short < ma_short.shift(3)  # TODO: 将魔法数字提取到配置中

        return patterns

    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算MA原始评分

        Returns:
            pd.Series: 评分序列,取值范围0-100
        """
        if data is None:
            return pd.Series(50.0)  # TODO: 将魔法数字提取到配置中

        # 确保计算了MA
        if not hasattr(self, "_result") or self._result is None:
            result = self.calculate(data, **kwargs)
        else:
            result = self._result

        if result.empty:
            return pd.Series(
                50.0, index=data.index
            )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        # 基于MA趋势计算评分
        ma_cols = [col for col in result.columns if col.startswith("SMA") or col.startswith("MA")]
        if not ma_cols:
            return pd.Series(
                50.0, index=data.index
            )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        ma = result[ma_cols[0]]
        price = data["close"] if "close" in data.columns else result["close"]

        # 评分逻辑:价格相对于MA的位置
        score = pd.Series(50.0, index=data.index)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        # 价格在MA之上加分,之下减分
        price_above_ma = price > ma
        price_below_ma = price < ma

        score[price_above_ma] = 60.0  # TODO: 将魔法数字提取到配置中
        score[price_below_ma] = 40.0  # TODO: 将魔法数字提取到配置中

        # 趋势方向调整
        ma_trend = ma.rolling(5).mean() > ma.rolling(5).mean().shift(
            5
        )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        score[ma_trend] += 10
        score[~ma_trend] -= 10

        return score.clip(0, 100)

    def get_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成MA交易信号

        Returns:
            pd.DataFrame: 包含交易信号的DataFrame
        """
        if data is None:
            data = self._result if hasattr(self, "_result") and self._result is not None else pd.DataFrame()

        if data.empty:
            return pd.DataFrame()

        # 确保数据包含MA
        ma_cols = [col for col in data.columns if col.startswith("SMA") or col.startswith("MA")]
        if not ma_cols:
            data = self.calculate(data, **kwargs)
            ma_cols = [col for col in data.columns if col.startswith("SMA") or col.startswith("MA")]

        signals = pd.DataFrame(index=data.index)

        if ma_cols and "close" in data.columns:
            ma = data[ma_cols[0]]
            price = data["close"]

            # 生成信号
            signals["ma_signal"] = 0
            signals["ma_strength"] = 0.0
            signals["ma_confidence"] = 0.0

            # 价格上穿MA买入信号
            price_cross_above = crossover(price, ma)
            signals.loc[price_cross_above, "ma_signal"] = 1
            signals.loc[price_cross_above, "ma_strength"] = 0.7  # TODO: 将魔法数字提取到配置中
            signals.loc[price_cross_above, "ma_confidence"] = 0.6  # TODO: 将魔法数字提取到配置中

            # 价格下穿MA卖出信号
            price_cross_below = crossunder(price, ma)
            signals.loc[price_cross_below, "ma_signal"] = -1
            signals.loc[price_cross_below, "ma_strength"] = 0.7  # TODO: 将魔法数字提取到配置中
            signals.loc[price_cross_below, "ma_confidence"] = 0.6  # TODO: 将魔法数字提取到配置中

        return signals

    def calculate_score(self, data: pd.DataFrame, **kwargs) -> dict:
        """
        计算MA综合评分

        Returns:
            dict: 包含评分信息的字典
        """
        raw_score = self.calculate_raw_score(data, **kwargs)
        patterns = self.get_patterns(data, **kwargs)
        signals = self.get_signals(data, **kwargs)

        # 计算平均分数
        avg_score = raw_score.mean() if not raw_score.empty else 50.0  # TODO: 将魔法数字提取到配置中

        # 计算置信度
        confidence = self.calculate_confidence(raw_score, patterns, signals)

        return {
            "average_score": avg_score,
            "latest_score": raw_score.iloc[-1] if not raw_score.empty else 50.0,  # TODO: 将魔法数字提取到配置中
            "confidence": confidence,
            "signal_strength": signals["ma_strength"].mean() if "ma_strength" in signals.columns else 0.0,
            "pattern_count": patterns.sum().sum() if not patterns.empty else 0,
        }

    def set_parameters(self, **kwargs):
        """
        设置指标参数

        Args:
            **kwargs: 参数字典
        """
        return self.set_parameters_Ma(**kwargs)

    def calculate_confidence(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """
        计算MA置信度

        Returns:
            float: 置信度值,范围0-1
        """
        # 简单的置信度计算
        if score.empty:
            return 0.5  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        # 基于评分的稳定性
        score_std = score.std()
        confidence = max(0.3, 1.0 - score_std / 100.0)  # TODO: 将魔法数字提取到配置中

        return min(0.9, confidence)  # TODO: 将魔法数字提取到配置中

    def generate_trading_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成交易信号(兼容性方法)

        Returns:
            pd.DataFrame: 交易信号DataFrame
        """
        return self.get_signals(data, **kwargs)

    def identify_patterns(self, data: pd.DataFrame = None, **kwargs) -> List[str]:
        """
        识别MA指标形态

        Args:
            data: 数据DataFrame
            **kwargs: 其他参数

        Returns:
            List[str]: 识别出的形态列表
        """
        if data is None:
            data = self._result if hasattr(self, "_result") and self._result is not None else pd.DataFrame()

        if data.empty:
            return []

        # 确保数据包含MA列
        ma_cols = [col for col in data.columns if "MA" in col or "ma" in col]
        if not ma_cols:
            data = self.calculate(data, **kwargs)
            ma_cols = [col for col in data.columns if "MA" in col or "ma" in col]

        patterns = []

        # 检查是否有足够的数据
        if len(data) < 20:  # TODO: 将魔法数字提取到配置中
            return patterns

        # 获取最新数据
        latest_idx = data.index[-1]

        # 检查金叉死叉形态
        if "ma5" in data.columns and "ma20" in data.columns:
            ma5 = data["ma5"]
            ma20 = data["ma20"]

            # 检查金叉:MA5上穿MA20
            if len(ma5) >= 2 and len(ma20) >= 2:
                if ma5.iloc[-1] > ma20.iloc[-1] and ma5.iloc[-2] <= ma20.iloc[-2]:
                    patterns.append("MA金叉")
                elif ma5.iloc[-1] < ma20.iloc[-1] and ma5.iloc[-2] >= ma20.iloc[-2]:
                    patterns.append("MA死叉")

        # 检查多头排列
        if "ma5" in data.columns and "ma10" in data.columns and "ma20" in data.columns:
            ma5_val = data["ma5"].iloc[-1]
            ma10_val = data["ma10"].iloc[-1]
            ma20_val = data["ma20"].iloc[-1]

            if ma5_val > ma10_val > ma20_val:
                patterns.append("MA多头排列")
            elif ma5_val < ma10_val < ma20_val:
                patterns.append("MA空头排列")

        # 检查趋势跟随
        if "close" in data.columns and "ma20" in data.columns:
            close_val = data["close"].iloc[-1]
            ma20_val = data["ma20"].iloc[-1]

            if close_val > ma20_val * 1.02:  # 价格明显高于MA20
                patterns.append("MA强势上涨")
            elif close_val < ma20_val * 0.98:  # 价格明显低于MA20  # TODO: 将魔法数字提取到配置中
                patterns.append("MA弱势下跌")
            else:
                patterns.append("MA横盘整理")

        return patterns if patterns else ["MA正常运行"]


# 类别名
MA = MaMa
