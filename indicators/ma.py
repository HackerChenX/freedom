import pandas as pd
import numpy as np
from typing import List, Dict, Any

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from utils.indicator_utils import crossover, crossunder
from indicators.pattern_registry import PatternTypePatternRegistry
from utils.dependency_injection import get_logger

logger = get_logger(__name__)

class MaMa(BaseIndicator, PatternSignalMixin):
    """
    移动平均线(MA_Ma)
    分类：趋势类指标
    描述：计算价格的简单移动平均。
    """
    # MA指标只需要close列
    REQUIRED_COLUMNS = ['close']

    def __init__(self, **kwargs):
        """
        初始化移动平均线(MA_Ma)指标
        Args:
            **kwargs: 指标参数，支持period、price_field等
        """
        super().__init__(name="MA_Ma", description="移动平均线")

        # 设置默认参数
        self._default_parameters = self._get_default_parameters_ma()

        # 应用用户参数
        self.set_parameters_Ma(**kwargs)

        self.ma_cols = [f'{self.ma_type}{self.period}']
        self.register_patterns_Ma()

    def _get_default_parameters_ma(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {"period": 20, "price_field": "close"}

    def set_parameters_Ma(self, **kwargs):
        """
        设置指标参数

        Args:
            **kwargs: 参数字典，支持以下参数：
                - period: 移动平均线周期
                - price_field: 价格字段选择
        """
        # 验证参数
        from utils.indicator_parameter_validator import IndicatorParameterValidator
        validator = IndicatorParameterValidator()

        # 合并默认参数和用户参数
        params = self._default_parameters.copy()
        params.update(kwargs)

        # 验证参数
        is_valid, errors = validator.validate_indicator_parameters('MA_Ma', params)
        if not is_valid:
            # 静默处理验证失败，避免过多警告
            pass

        # 设置参数
        self.period = params.get('period', 20)
        self.price_field = params.get('price_field', 'close')

        # 保持向后兼容性
        self.periods = [self.period]  # 为了兼容现有代码
        self.ma_type = 'SMA'

        if hasattr(self, 'ma_cols'):
            self.ma_cols = [f'{self.ma_type}{self.period}']

    def _calculate_ma(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算简单移动平均线(SMA)
        """
        # 从原始数据开始，确保保留所有基础列
        result_df = data.copy()

        # 确保close列存在
        if 'close' not in data.columns:
            raise ValueError("数据中缺少'close'列")

        close_series = data['close']

        # 处理close列不是Series的情况
        if not isinstance(close_series, pd.Series):
            # 获取close列的值
            close_values = close_series.values if hasattr(close_series, 'values') else close_series

            # 如果是多维数组，展平它
            if hasattr(close_values, 'flatten'):
                close_values = close_values.flatten()

            # 确保数据长度与索引长度匹配
            expected_length = len(data.index)
            if len(close_values) != expected_length:
                # 如果长度不匹配，只取需要的长度
                if len(close_values) > expected_length:
                    close_values = close_values[:expected_length]
                else:
                    # 如果数据不足，用NaN填充
                    close_values = np.pad(close_values, (0, expected_length - len(close_values)),
                                        constant_values=np.nan)

            # 创建正确的Series
            close_series = pd.Series(close_values, index=data.index)

        # 计算移动平均线
        for p in self.periods:
            ma_values = close_series.rolling(window=p).mean()
            result_df[f'{self.ma_type}{p}'] = ma_values

        
        # 添加形态识别和信号生成
        result_df = self.add_pattern_detection(result_df)
        result_df = self.add_signal_generation(result_df)

        # 重写信号生成逻辑（MA指标特定逻辑）
        result_df = self._apply_ma_signal_logic(result_df)

        return result_df

    def _apply_ma_signal_logic(self, result_df: pd.DataFrame) -> pd.DataFrame:
        """
        应用MA指标特定的信号生成逻辑
        基于价格与移动平均线的关系生成信号
        """
        try:
            # 获取收盘价
            close_price = result_df['close']

            # 获取主要MA线（使用设定的周期）
            ma_col = f'{self.ma_type}{self.period}'
            if ma_col not in result_df.columns:
                # 如果没有主要MA线，使用默认信号
                return result_df

            ma_line = result_df[ma_col]

            # MA信号生成逻辑：
            # BUY: 价格在MA线之上
            # SELL: 价格在MA线之下
            # HOLD: 价格接近MA线（±1%范围内）

            price_above_ma = close_price > ma_line
            price_below_ma = close_price < ma_line
            price_near_ma = (abs(close_price - ma_line) / ma_line) <= 0.01  # 1%范围内

            # 生成信号
            result_df.loc[:, 'buy_signal'] = price_above_ma & ~price_near_ma
            result_df.loc[:, 'sell_signal'] = price_below_ma & ~price_near_ma
            result_df.loc[:, 'hold_signal'] = price_near_ma

            # 确保信号类型为布尔值
            result_df['buy_signal'] = result_df['buy_signal'].astype(bool)
            result_df['sell_signal'] = result_df['sell_signal'].astype(bool)
            result_df['hold_signal'] = result_df['hold_signal'].astype(bool)

        except Exception as e:
            logger.warning(f"MA信号生成失败: {e}")
            # 如果出错，使用默认信号
            result_df.loc[:, 'buy_signal'] = False
            result_df.loc[:, 'sell_signal'] = False
            result_df.loc[:, 'hold_signal'] = True

        return result_df

    def calculate_raw_score_Ma(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算MA原始评分。
        """
        # 确保已计算MA指标
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)
        
        # 检查MA列是否存在
        if not self.ma_cols or not all(c in self._result.columns for c in self.ma_cols):
            return pd.Series(50.0, index=data.index)

        score = pd.Series(50.0, index=data.index)
        
        sorted_mas = [self._result[f'{self.ma_type}{p}'] for p in sorted(self.periods)]
        
        if len(sorted_mas) > 1:
            is_bullish_arrangement = (sorted_mas[0] > sorted_mas[-1])
            is_bearish_arrangement = (sorted_mas[0] < sorted_mas[-1])
            score[is_bullish_arrangement] += 25
            score[is_bearish_arrangement] -= 25

        short_ma = sorted_mas[0]
        trend = np.sign(short_ma.diff(2)).fillna(0)
        score[trend == 1] += 15
        score[trend == -1] -= 15

        close_price = data['close']
        score[close_price > short_ma] += 10
        score[close_price < short_ma] -= 10
        
        if len(sorted_mas) >= 2:
            short_ma_series = sorted_mas[0]
            medium_ma_series = sorted_mas[1]
            golden_cross = crossover(short_ma_series, medium_ma_series)
            death_cross = crossunder(short_ma_series, medium_ma_series)
            score[golden_cross.rolling(window=2, min_periods=1).max().fillna(0).astype(bool)] += 20
            score[death_cross.rolling(window=2, min_periods=1).max().fillna(0).astype(bool)] -= 20

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
            return 0.5

        # 基础置信度
        confidence = 0.5

        # 1. 基于评分的置信度
        last_score = score.iloc[-1]

        # 极端评分置信度较高
        if last_score > 80 or last_score < 20:
            confidence += 0.25
        # 中性评分置信度中等
        elif 40 <= last_score <= 60:
            confidence += 0.1
        else:
            confidence += 0.15

        # 2. 基于形态的置信度
        if isinstance(patterns, pd.DataFrame) and not patterns.empty:
            # 统计最近几个周期的形态数量
            try:
                numeric_cols = patterns.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    recent_data = patterns[numeric_cols].iloc[-5:] if len(patterns) >= 5 else patterns[numeric_cols]
                    recent_patterns = recent_data.sum().sum()
                    confidence += min(recent_patterns * 0.05, 0.2)
            except:
                pass

        # 3. 基于评分稳定性的置信度
        if len(score) >= 5:
            recent_scores = score.iloc[-5:]
            score_stability = 1.0 - (recent_scores.std() / 50.0)
            confidence += score_stability * 0.1

        # 4. 基于MA排列的置信度
        if hasattr(self, '_result') and self._result is not None and not self._result.empty:
            try:
                # 检查MA排列的一致性
                ma_cols = [col for col in self._result.columns if col.startswith(self.ma_type)]
                if len(ma_cols) >= 2:
                    # 获取最新的MA值
                    latest_mas = []
                    for col in ma_cols:
                        if not self._result[col].empty:
                            latest_ma = self._result[col].dropna().iloc[-1] if not self._result[col].dropna().empty else None
                            if latest_ma is not None:
                                latest_mas.append(latest_ma)

                    if len(latest_mas) >= 2:
                        # 检查MA排列是否有序
                        sorted_mas = sorted(latest_mas)
                        if latest_mas == sorted_mas or latest_mas == sorted_mas[::-1]:
                            confidence += 0.1  # MA排列有序，增加置信度
            except:
                pass

        return min(confidence, 1.0)

    def get_patterns_Ma(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        识别MA技术形态
        - 金叉/死叉：基于最短和次短周期均线。
        - 多头/空头排列：基于最短和最长周期均线。
        """
        patterns = {}
        if len(self.periods) < 2 or not all(c in data.columns for c in self.ma_cols):
            return pd.DataFrame(patterns, index=data.index)

        sorted_periods = sorted(self.periods)
        p_short, p_medium = sorted_periods[0], sorted_periods[1]
        p_long = sorted_periods[-1]

        short_ma = data[f'{self.ma_type}{p_short}']
        medium_ma = data[f'{self.ma_type}{p_medium}']
        long_ma = data[f'{self.ma_type}{p_long}']

        patterns[f"MA_{p_short}_{p_medium}_GOLDEN_CROSS"] = crossover(short_ma, medium_ma)
        patterns[f"MA_{p_short}_{p_medium}_DEATH_CROSS"] = crossunder(short_ma, medium_ma)
        patterns["MA_BULLISH_ARRANGEMENT"] = short_ma > long_ma
        patterns["MA_BEARISH_ARRANGEMENT"] = short_ma < long_ma
        
        return pd.DataFrame(patterns)

    def register_patterns_Ma(self):
        """
        注册与该指标相关的技术形态。
        """
        if len(self.periods) < 2:
            return
            
        sorted_periods = sorted(self.periods)
        p_short, p_medium = sorted_periods[0], sorted_periods[1]
        p_long = sorted_periods[-1]
        
        self.register_pattern_to_registry(
            pattern_id=f"MA_{p_short}_{p_medium}_GOLDEN_CROSS",
            display_name=f"MA_Ma({p_short},{p_medium})金叉",
            description=f"当短期MA({p_short})上穿中期MA({p_medium})时，被视为看涨信号。",
            pattern_type=Pattern_type.BULLISH,
            polarity="POSITIVE"
        )
        self.register_pattern_to_registry(
            pattern_id=f"MA_{p_short}_{p_medium}_DEATH_CROSS",
            display_name=f"MA_Ma({p_short},{p_medium})死叉",
            description=f"当短期MA({p_short})下穿中期MA({p_medium})时，被视为看跌信号。",
            pattern_type=Pattern_type.BEARISH,
            polarity="NEGATIVE"
        )

        # 注册MA排列形态（从centralized mapping迁移）
        self.register_pattern_to_registry(
            pattern_id="MA_BULLISH_ARRANGEMENT",
            display_name="均线多头排列",
            description="短期均线在长期均线之上，形成多头排列",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=20.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="MA_BEARISH_ARRANGEMENT",
            display_name="均线空头排列",
            description="短期均线在长期均线之下，形成空头排列",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-20.0,
            polarity="NEGATIVE"
        )

        # 注册MA支撑阻力形态
        self.register_pattern_to_registry(
            pattern_id="MA_SUPPORT",
            display_name="均线支撑",
            description="价格在均线获得支撑",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=15.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="MA_RESISTANCE",
            display_name="均线阻力",
            description="价格在均线遇到阻力",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-15.0,
            polarity="NEGATIVE"
        )
        self.register_pattern_to_registry(
            pattern_id="MA_BULLISH_ARRANGEMENT",
            display_name="MA多头排列",
            description=f"短期MA({p_short})在长期MA({p_long})之上，表明市场处于强劲上升趋势。",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=25.0,
            polarity="POSITIVE"
        )
        self.register_pattern_to_registry(
            pattern_id="MA_BEARISH_ARRANGEMENT",
            display_name="MA空头排列",
            description=f"短期MA({p_short})在长期MA({p_long})之下，表明市场处于强劲下降趋势。",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-25.0,
            polarity="NEGATIVE"
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
                "score_impact": 0.0
            }

        sorted_periods = sorted(self.periods)
        p_short, p_medium = sorted_periods[0], sorted_periods[1]
        p_long = sorted_periods[-1]

        pattern_info_map = {
            f"MA_{p_short}_{p_medium}_GOLDEN_CROSS": {
                "id": f"MA_{p_short}_{p_medium}_GOLDEN_CROSS",
                "name": f"MA_Ma({p_short},{p_medium})金叉",
                "description": f"短期MA({p_short})上穿中期MA({p_medium})，看涨信号",
                "type": "BULLISH",
                "strength": "STRONG",
                "score_impact": 20.0
            },
            f"MA_{p_short}_{p_medium}_DEATH_CROSS": {
                "id": f"MA_{p_short}_{p_medium}_DEATH_CROSS",
                "name": f"MA_Ma({p_short},{p_medium})死叉",
                "description": f"短期MA({p_short})下穿中期MA({p_medium})，看跌信号",
                "type": "BEARISH",
                "strength": "STRONG",
                "score_impact": -20.0
            },
            "MA_BULLISH_ARRANGEMENT": {
                "id": "MA_BULLISH_ARRANGEMENT",
                "name": "MA多头排列",
                "description": f"短期MA({p_short})在长期MA({p_long})之上，强劲上升趋势",
                "type": "BULLISH",
                "strength": "MEDIUM",
                "score_impact": 25.0
            },
            "MA_BEARISH_ARRANGEMENT": {
                "id": "MA_BEARISH_ARRANGEMENT",
                "name": "MA空头排列",
                "description": f"短期MA({p_short})在长期MA({p_long})之下，强劲下降趋势",
                "type": "BEARISH",
                "strength": "MEDIUM",
                "score_impact": -25.0
            }
        }

        return pattern_info_map.get(pattern_id, {
            "id": pattern_id,
            "name": "均线趋势分析",
            "description": f"基于移动平均线的趋势分析: {pattern_id}",
            "type": "NEUTRAL",
            "strength": "WEAK",
            "score_impact": 0.0
        })