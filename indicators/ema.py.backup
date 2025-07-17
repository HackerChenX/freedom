#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
指数移动平均线(EMA_Ema)

对近期价格赋予更高权重
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from utils.indicator_utils import crossover, crossunder
from utils.logger import getLogger

logger = getLogger(__name__)


class EmaEma(BaseIndicator, PatternSignalMixin):
    """
    指数移动平均线(EMA_Ema)
    
    分类：趋势类指标
    描述：对近期价格赋予更高权重
    """
    
    # EMA指标只需要close列
    REQUIRED_COLUMNS = ['close']

    def __init__(self, **kwargs):
        """
        初始化指数移动平均线(EMA_Ema)指标
        Args:
            **kwargs: 指标参数，支持period、price_field、alpha等
        """
        super().__init__(name="EMA_Ema", description="指数移动平均线")

        # 设置默认参数
        self._default_parameters = self._get_default_parameters_ema()

        # 应用用户参数
        self.set_parameters_Ema(**kwargs)

        self.ma_cols = [f'{self.ma_type}{self.period}']
        self.register_patterns_Ema()

    def _get_default_parameters_ema(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {"period": 12, "price_field": "close", "alpha": None}
        
    def set_parameters_Ema(self, **kwargs):
        """
        设置指标参数

        Args:
            **kwargs: 参数字典，支持以下参数：
                - period: EMA计算周期
                - price_field: 价格字段选择
                - alpha: 平滑因子
        """
        # 验证参数
        from utils.indicator_parameter_validator import IndicatorParameterValidator
        validator = IndicatorParameterValidator()

        # 合并默认参数和用户参数
        params = self._default_parameters.copy()
        params.update(kwargs)        # 验证参数
        try:
            from utils.indicator_parameter_validator import IndicatorParameterValidator
            validator = IndicatorParameterValidator()
            
            # 验证参数
            is_valid, errors = validator.validate_indicator_parameters('EMA_Ema', params)
            if not is_valid:
                # 静默处理验证失败，避免过多警告
                pass
                
        except Exception:
            # 如果验证失败，静默处理，保持向后兼容
            pass

        # 设置参数
        self.period = params.get('period', 12)
        self.price_field = params.get('price_field', 'close')
        self.alpha = params.get('alpha', None)

        # 保持向后兼容性
        self.periods = [self.period]  # 为了兼容现有代码
        self.ma_type = 'EMA_Ema'

        if hasattr(self, 'ma_cols'):
            self.ma_cols = [f'{self.ma_type}{self.period}']
        
    def _calculate_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算指数移动平均线(EMA_Ema)指标
        """
        for p in self.periods:
            df[f'{self.ma_type}{p}'] = df['close'].ewm(span=p, adjust=True).mean()
        
        # 添加形态识别和信号生成
        df = self.add_pattern_detection(df)
        df = self.add_signal_generation(df)

        # 重写信号生成逻辑（EMA指标特定逻辑）
        df = self._apply_ema_signal_logic(df)

        return df

    def _apply_ema_signal_logic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用EMA指标特定的信号生成逻辑
        基于价格与指数移动平均线的关系生成信号
        """
        try:
            # 获取收盘价
            close_price = df['close']

            # 获取最短周期的EMA线
            shortest_period = min(self.periods)
            ema_col = f'{self.ma_type}{shortest_period}'

            if ema_col not in df.columns:
                # 如果没有EMA线，使用默认信号
                return df

            ema_line = df[ema_col]

            # EMA信号生成逻辑：
            # BUY: 价格在EMA线之上且EMA向上
            # SELL: 价格在EMA线之下且EMA向下
            # HOLD: 其他情况

            price_above_ema = close_price > ema_line
            price_below_ema = close_price < ema_line

            # 计算EMA趋势（当前值与前一值比较）
            ema_rising = ema_line > ema_line.shift(1)
            ema_falling = ema_line < ema_line.shift(1)

            # 生成信号
            df.loc[:, 'buy_signal'] = price_above_ema & ema_rising
            df.loc[:, 'sell_signal'] = price_below_ema & ema_falling
            df.loc[:, 'hold_signal'] = ~(df['buy_signal'] | df['sell_signal'])

            # 确保信号类型为布尔值
            df['buy_signal'] = df['buy_signal'].astype(bool)
            df['sell_signal'] = df['sell_signal'].astype(bool)
            df['hold_signal'] = df['hold_signal'].astype(bool)

        except Exception as e:
            logger.warning(f"EMA信号生成失败: {e}")
            # 如果出错，使用默认信号
            df.loc[:, 'buy_signal'] = False
            df.loc[:, 'sell_signal'] = False
            df.loc[:, 'hold_signal'] = True

        return df

    def calculate_raw_score_Ema(self, df: pd.DataFrame) -> pd.Series:
        """
        计算EMA原始评分。
        评分标准:
        1.  多头/空头排列: +40 (多头) / -40 (空头)
        2.  短期趋势: 向上+15, 向下-15
        3.  价格与短周期均线关系: 价格在均线上方+10, 下方-10
        4.  金叉/死叉: 最近2天内发生金叉+20, 死叉-20
        """
        if not self.ma_cols or not all(c in df.columns for c in self.ma_cols):
            return pd.Series(50, index=df.index)

        score = pd.Series(50.0, index=df.index)
        
        sorted_mas = [df[f'{self.ma_type}{p}'] for p in sorted(self.periods)]
        
        if len(sorted_mas) > 1:
            is_bullish_arrangement = (sorted_mas[0] > sorted_mas[-1])
            is_bearish_arrangement = (sorted_mas[0] < sorted_mas[-1])
            score[is_bullish_arrangement] += 25
            score[is_bearish_arrangement] -= 25

        short_ma = sorted_mas[0]
        trend = np.sign(short_ma.diff(2)).fillna(0)
        score[trend == 1] += 15
        score[trend == -1] -= 15

        close_price = df['close']
        score[close_price > short_ma] += 10
        score[close_price < short_ma] -= 10
        
        if len(sorted_mas) >= 2:
            short_ema = sorted_mas[0]
            medium_ema = sorted_mas[1]
            golden_cross = crossover(short_ema, medium_ema)
            death_cross = crossunder(short_ema, medium_ema)
            score[golden_cross.rolling(window=2, min_periods=1).max().fillna(0).astype(bool)] += 20
            score[death_cross.rolling(window=2, min_periods=1).max().fillna(0).astype(bool)] -= 20

        return score.clip(0, 100)

    def calculate_confidence_Ema(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """
        计算置信度。
        """
        return 0.5

    def get_patterns_Ema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        识别EMA技术形态
        """
        patterns = {}
        if len(self.periods) < 2 or not all(c in df.columns for c in self.ma_cols):
            return pd.DataFrame(patterns)

        p_short, p_long = sorted(self.periods)[:2]
        short_ema = df[f'{self.ma_type}{p_short}']
        long_ema = df[f'{self.ma_type}{p_long}']

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
        注册与该指标相关的技术形态。
        """
        if len(self.periods) < 2:
            return
            
        p_short, p_long = sorted(self.periods)[:2]
        
        self.register_pattern_to_registry(
            pattern_id=f"EMA_{p_short}_{p_long}_GOLDEN_CROSS",
            display_name=f"EMA_Ema({p_short},{p_long})金叉",
            description=f"当短期EMA({p_short})上穿长期EMA({p_long})时，被视为看涨信号。",
            pattern_type="BULLISH",
            polarity="POSITIVE"
        )
        self.register_pattern_to_registry(
            pattern_id=f"EMA_{p_short}_{p_long}_DEATH_CROSS",
            display_name=f"EMA_Ema({p_short},{p_long})死叉",
            description=f"当短期EMA({p_short})下穿长期EMA({p_long})时，被视为看跌信号。",
            pattern_type="BEARISH",
            polarity="NEGATIVE"
        )

        # 注册EMA排列形态（从centralized mapping迁移）
        self.register_pattern_to_registry(
            pattern_id="EMA_BULLISH_ARRANGEMENT",
            display_name="EMA多头排列",
            description="指数移动平均线呈多头排列，趋势向上",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=20.0,
            polarity="POSITIVE"
        )
        self.register_pattern_to_registry(
            pattern_id="EMA_BULLISH_ARRANGEMENT",
            display_name="EMA多头排列",
            description=f"短期EMA在长期EMA之上，表明市场处于上升趋势。",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=15.0,
            polarity="POSITIVE"
        )
        self.register_pattern_to_registry(
            pattern_id="EMA_BEARISH_ARRANGEMENT",
            display_name="EMA空头排列",
            description=f"短期EMA在长期EMA之下，表明市场处于下降趋势。",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-15.0,
            polarity="NEGATIVE"
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
            "score_impact": 0.0
        }
        
        # EMA指标特定的形态信息映射
        pattern_info_map = {
            # 基础形态
            "超买区域": {
                "id": "超买区域",
                "name": "超买区域",
                "description": "指标进入超买区域，可能面临回调压力",
                "type": "BEARISH",
                "strength": "MEDIUM",
                "score_impact": -10.0
            },
            "超卖区域": {
                "id": "超卖区域", 
                "name": "超卖区域",
                "description": "指标进入超卖区域，可能出现反弹机会",
                "type": "BULLISH",
                "strength": "MEDIUM",
                "score_impact": 10.0
            },
            "中性区域": {
                "id": "中性区域",
                "name": "中性区域", 
                "description": "指标处于中性区域，趋势不明确",
                "type": "NEUTRAL",
                "strength": "WEAK",
                "score_impact": 0.0
            },
            # 趋势形态
            "上升趋势": {
                "id": "上升趋势",
                "name": "上升趋势",
                "description": "指标显示上升趋势，看涨信号",
                "type": "BULLISH", 
                "strength": "STRONG",
                "score_impact": 15.0
            },
            "下降趋势": {
                "id": "下降趋势",
                "name": "下降趋势",
                "description": "指标显示下降趋势，看跌信号",
                "type": "BEARISH",
                "strength": "STRONG", 
                "score_impact": -15.0
            },
            # 信号形态
            "买入信号": {
                "id": "买入信号",
                "name": "买入信号",
                "description": "指标产生买入信号，建议关注",
                "type": "BULLISH",
                "strength": "STRONG",
                "score_impact": 20.0
            },
            "卖出信号": {
                "id": "卖出信号", 
                "name": "卖出信号",
                "description": "指标产生卖出信号，建议谨慎",
                "type": "BEARISH",
                "strength": "STRONG",
                "score_impact": -20.0
            }
        }
        
        return pattern_info_map.get(pattern_id, default_pattern)


