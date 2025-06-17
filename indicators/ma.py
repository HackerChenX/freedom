import pandas as pd
import numpy as np
from typing import List

from indicators.base_indicator import BaseIndicator
from utils.indicator_utils import crossover, crossunder
from indicators.pattern_registry import PatternType

class MA(BaseIndicator):
    """
    移动平均线(MA)
    分类：趋势类指标
    描述：计算价格的简单移动平均。
    """
    # MA指标只需要close列
    REQUIRED_COLUMNS = ['close']

    def __init__(self, periods: List[int] = None, ma_type: str = 'SMA'):
        """
        初始化移动平均线(MA)指标
        Args:
            periods: 计算周期列表，默认为[5, 10, 20, 60]
            ma_type: 均线类型，默认为'SMA' (简单移动平均)
        """
        super().__init__(name="MA", description="移动平均线")
        self.periods = periods if periods is not None else [5, 10, 20, 60]
        # For simplicity, this class will only handle SMA. 'ma_type' is for consistency.
        self.ma_type = 'SMA' 
        self.ma_cols = [f'{self.ma_type}{p}' for p in self.periods]
        self.register_patterns()

    def set_parameters(self, periods: List[int] = None, ma_type: str = None):
        """设置指标参数"""
        if periods is not None:
            self.periods = periods
        # ma_type is ignored to keep it simple
        self.ma_cols = [f'{self.ma_type}{p}' for p in self.periods]
        self.register_patterns()

    def _calculate(self, data: pd.DataFrame) -> pd.DataFrame:
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

        return result_df

    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算MA原始评分。
        """
        if not self.ma_cols or not all(c in data.columns for c in self.ma_cols):
            return pd.Series(50, index=data.index)

        score = pd.Series(50.0, index=data.index)
        
        sorted_mas = [data[f'{self.ma_type}{p}'] for p in sorted(self.periods)]
        
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

    def calculate_confidence(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """
        计算MA指标的置信度

        Args:
            score: 得分序列
            patterns: 检测到的形态DataFrame
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

    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
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

    def register_patterns(self):
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
            display_name=f"MA({p_short},{p_medium})金叉",
            description=f"当短期MA({p_short})上穿中期MA({p_medium})时，被视为看涨信号。",
            pattern_type=PatternType.BULLISH,
            polarity="POSITIVE"
        )
        self.register_pattern_to_registry(
            pattern_id=f"MA_{p_short}_{p_medium}_DEATH_CROSS",
            display_name=f"MA({p_short},{p_medium})死叉",
            description=f"当短期MA({p_short})下穿中期MA({p_medium})时，被视为看跌信号。",
            pattern_type=PatternType.BEARISH,
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

    def get_pattern_info(self, pattern_id: str) -> dict:
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
                "name": "未知形态",
                "description": "未定义的形态",
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
                "name": f"MA({p_short},{p_medium})金叉",
                "description": f"短期MA({p_short})上穿中期MA({p_medium})，看涨信号",
                "type": "BULLISH",
                "strength": "STRONG",
                "score_impact": 20.0
            },
            f"MA_{p_short}_{p_medium}_DEATH_CROSS": {
                "id": f"MA_{p_short}_{p_medium}_DEATH_CROSS",
                "name": f"MA({p_short},{p_medium})死叉",
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
            "name": "未知形态",
            "description": "未定义的形态",
            "type": "NEUTRAL",
            "strength": "WEAK",
            "score_impact": 0.0
        })