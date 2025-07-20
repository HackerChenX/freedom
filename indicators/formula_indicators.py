#!/usr/bin/env python3
from utils.dependency_injection import get_logger
"""
公式指标模块
包含各种技术分析公式指标
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from utils.dependency_injection import get_logger

logger = get_logger(__name__)


class FormulaIndicators(BaseIndicator, PatternSignalMixin):
    """
    公式指标基类
    """
    
    def __init__(self, **kwargs):
        super().__init__()
        self.name = "FORMULA_INDICATORS"
        self._default_parameters = self._get_default_parameters_formulaindicators()
        self.set_parameters_Indicators_formulaindicators(**kwargs)
    
    def _get_default_parameters_formulaindicators(self) -> Dict[str, Any]:
        return {"period": 14}
    
    def set_parameters_Indicators_formulaindicators(self, **kwargs):
        try:
            from utils.indicator_parameter_validator import IndicatorParameterValidator
            validator = IndicatorParameterValidator()
            params = self._default_parameters.copy()
            params.update(kwargs)
            is_valid, errors = validator.validate_indicator_parameters('FORMULA_INDICATORS', params)
            if not is_valid:
                params = self._default_parameters.copy()
            self.period = params.get('period', 14)
        except Exception:
            self.period = 14
    
    def calculate_Indicators_Formula_Indicators(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        result = self._calculate_formulaindicators(data, **kwargs)
        self._result = result
        return result
    
    def _calculate_formulaindicators(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        df[f'FORMULA_VALUE'] = df['close'].rolling(window=self.period).mean()
        
        # 添加形态识别和信号生成
        df = self.add_pattern_detection(df)
        df = self.add_signal_generation(df)

        return df
    
    def calculate_raw_score_Indicators_formulaindicators(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if not self.has_result():
            self.calculate_Indicators_Formula_Indicators(data, **kwargs)
        return pd.Series(50.0, index=data.index)
    
    def calculate_confidence_Indicators_formulaindicators(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        return 0.5
    
    def get_patterns_Indicators_formulaindicators(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return pd.DataFrame(index=data.index)


class CrossOver(FORMULA_INDICATORS):
    """
    交叉指标 - 通用交叉信号检测器
    
    特点:
    1. 检测各种技术指标的交叉信号
    2. 支持金叉、死叉、零轴交叉等多种交叉类型
    3. 可配置交叉确认周期和强度阈值
    4. 提供交叉信号的强度和可靠性评估
    
    计算方法:
    1. 计算快线和慢线（可以是移动平均、指标线等）
    2. 检测交叉点：快线从下方穿越慢线（金叉）或从上方穿越慢线（死叉）
    3. 评估交叉强度：基于交叉前后的价格和指标变化
    4. 确认交叉有效性：检查交叉后的持续性
    
    参数:
    - fast_period: 快线周期，默认为5
    - slow_period: 慢线周期，默认为20
    - confirm_period: 交叉确认周期，默认为3
    - cross_type: 交叉类型（'ma', 'price', 'indicator'），默认为'ma'
    """
    
    def _detect_golden_cross_Formula_Indicators(self, fast_line: pd.Series, slow_line: pd.Series) -> pd.Series:
        """检测金叉信号"""
        golden_cross = pd.Series(0.0, index=fast_line.index)
        
        # 基本金叉：快线从下方穿越慢线
        basic_golden = (fast_line > slow_line) & (fast_line.shift(1) <= slow_line.shift(1))
        
        # 确认金叉：交叉后持续确认周期
        confirmed_golden = basic_golden.copy()
        for i in range(1, self.confirm_period):
            confirmed_golden &= (fast_line.shift(-i) > slow_line.shift(-i))
        
        # 强金叉：在低位发生的金叉
        price_level = (fast_line + slow_line) / 2
        price_percentile = price_level.rolling(window=50).rank(pct=True)
        strong_golden = confirmed_golden & (price_percentile < 0.3)
        
        # 评分
        golden_cross[basic_golden] += 5.0
        golden_cross[confirmed_golden] += 10.0
        golden_cross[strong_golden] += 15.0
        
        return golden_cross
    
    def _detect_death_cross_Formula_Indicators(self, fast_line: pd.Series, slow_line: pd.Series) -> pd.Series:
        """检测死叉信号"""
        death_cross = pd.Series(0.0, index=fast_line.index)
        
        # 基本死叉：快线从上方穿越慢线
        basic_death = (fast_line < slow_line) & (fast_line.shift(1) >= slow_line.shift(1))
        
        # 确认死叉：交叉后持续确认周期
        confirmed_death = basic_death.copy()
        for i in range(1, self.confirm_period):
            confirmed_death &= (fast_line.shift(-i) < slow_line.shift(-i))
        
        # 强死叉：在高位发生的死叉
        price_level = (fast_line + slow_line) / 2
        price_percentile = price_level.rolling(window=50).rank(pct=True)
        strong_death = confirmed_death & (price_percentile > 0.7)
        
        # 评分
        death_cross[basic_death] += 5.0
        death_cross[confirmed_death] += 10.0
        death_cross[strong_death] += 15.0
        
        return death_cross
    
    def _calculate_cross_strength(self, fast_line: pd.Series, slow_line: pd.Series) -> pd.Series:
        """计算交叉强度"""
        # 基于快慢线的分离度计算强度
        separation = abs(fast_line - slow_line) / slow_line.replace(0, np.nan)
        
        # 标准化强度到0-2之间
        strength = separation.rolling(window=20).rank(pct=True) * 2
        
        return strength.fillna(1.0)
    
    def _calculate_cross_reliability(self, fast_line: pd.Series, slow_line: pd.Series, volume: pd.series = None) -> pd.Series:
        """计算交叉可靠性"""
        reliability = pd.Series(1.0, index=fast_line.index)
        
        # 基于趋势一致性的可靠性
        fast_trend = fast_line > fast_line.shift(1)
        slow_trend = slow_line > slow_line.shift(1)
        trend_consistency = (fast_trend == slow_trend).rolling(window=5).mean()
        reliability *= trend_consistency
        
        # 如果有成交量数据，考虑成交量确认
        if volume is not None:
            volume_trend = volume > volume.rolling(window=5).mean()
            volume_confirmation = volume_trend.rolling(window=3).mean()
            reliability *= (0.7 + 0.3 * volume_confirmation)
        
        return reliability.fillna(1.0)
    
    def _calculate_trend_confirmation(self, fast_line: pd.Series, slow_line: pd.Series) -> pd.Series:
        """计算趋势确认"""
        confirmation = pd.Series(0.0, index=fast_line.index)
        
        # 快线在慢线上方且都在上升
        bullish_trend = (fast_line > slow_line) & (fast_line > fast_line.shift(1)) & (slow_line > slow_line.shift(1))
        confirmation[bullish_trend] += 5.0
        
        # 快线在慢线下方且都在下降
        bearish_trend = (fast_line < slow_line) & (fast_line < fast_line.shift(1)) & (slow_line < slow_line.shift(1))
        confirmation[bearish_trend] -= 5.0
        
        return confirmation
    
    def calculate_raw_score_Indicators_formulaindicators_duplicate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算交叉指标的原始评分
        
        基于交叉信号的强度、可靠性和趋势确认进行评分：
        1. 交叉信号强度：金叉/死叉的强度评估
        2. 交叉可靠性：基于趋势一致性和成交量确认
        3. 趋势确认：交叉后的趋势持续性
        4. 交叉位置：交叉发生的价格位置（高位/低位）
        """
        if not self.has_result():
            self.calculate_Indicators_Formula_Indicators(data, **kwargs)
        
        if 'CROSS_OVER_VALUE' not in self._result.columns:
            return pd.Series(50.0, index=data.index)
        
        cross_signal = self._result['CROSS_OVER_VALUE'].fillna(0)
        golden_cross = self._result['golden_cross'].fillna(0)
        death_cross = self._result['death_cross'].fillna(0)
        cross_strength = self._result['cross_strength'].fillna(1)
        cross_reliability = self._result['cross_reliability'].fillna(1)
        
        scores = pd.Series(index=data.index, dtype=float)
        
        for i in range(len(cross_signal)):
            if i < max(self.fast_period, self.slow_period):
                scores.iloc[i] = 50.0
                continue
            
            score = 50.0  # 基础分数
            
            # 获取当前数据
            current_signal = cross_signal.iloc[i]
            current_golden = golden_cross.iloc[i]
            current_death = death_cross.iloc[i]
            current_strength = cross_strength.iloc[i]
            current_reliability = cross_reliability.iloc[i]
            
            # 1. 交叉信号评分 (40分)
            if current_golden > 0:
                # 金叉信号
                if current_golden >= 20:
                    signal_score = 40.0  # 强金叉
                elif current_golden >= 10:
                    signal_score = 30.0  # 确认金叉
                else:
                    signal_score = 20.0  # 基本金叉
            elif current_death > 0:
                # 死叉信号
                if current_death >= 20:
                    signal_score = 0.0   # 强死叉
                elif current_death >= 10:
                    signal_score = 10.0  # 确认死叉
                else:
                    signal_score = 20.0  # 基本死叉
            else:
                # 无交叉信号
                signal_score = 20.0
            
            score += signal_score - 20.0  # 调整基准
            
            # 2. 交叉强度评分 (25分)
            if current_strength > 1.5:
                strength_score = 25.0  # 强交叉
            elif current_strength > 1.2:
                strength_score = 20.0  # 中等强度
            elif current_strength > 0.8:
                strength_score = 15.0  # 一般强度
            else:
                strength_score = 10.0  # 弱交叉
            
            score += strength_score - 15.0  # 调整基准
            
            # 3. 交叉可靠性评分 (25分)
            if current_reliability > 0.8:
                reliability_score = 25.0  # 高可靠性
            elif current_reliability > 0.6:
                reliability_score = 20.0  # 中等可靠性
            elif current_reliability > 0.4:
                reliability_score = 15.0  # 一般可靠性
            else:
                reliability_score = 10.0  # 低可靠性
            
            score += reliability_score - 15.0  # 调整基准
            
            # 4. 综合信号评分 (10分)
            if current_signal > 15:
                综合_score = 10.0  # 强烈买入
            elif current_signal > 5:
                综合_score = 8.0   # 买入
            elif current_signal > -5:
                综合_score = 5.0   # 中性
            elif current_signal > -15:
                综合_score = 2.0   # 卖出
            else:
                综合_score = 0.0   # 强烈卖出
            
            score += 综合_score - 5.0  # 调整基准
            
            # 确保分数在合理范围内
            score = max(0, min(100, score))
            scores.iloc[i] = score
        
        return scores


class KDJCondition(FORMULA_INDICATORS):
    """KDJ条件指标 - 基于KDJ指标的条件判断"""
    
    def _calculate_oversold_condition(self, k: pd.Series, d: pd.Series, j: pd.Series) -> pd.Series:
        """计算超卖条件"""
        oversold = pd.Series(0.0, index=k.index)
        
        # 基本超卖条件：K、D、J都在20以下
        basic_oversold = (k < 20) & (d < 20) & (j < 20)
        oversold[basic_oversold] += 15
        
        # 强超卖条件：K、D、J都在10以下
        strong_oversold = (k < 10) & (d < 10) & (j < 10)
        oversold[strong_oversold] += 10
        
        # 连续超卖：连续3天以上超卖
        consecutive_oversold = basic_oversold.rolling(window=3).sum() >= 3
        oversold[consecutive_oversold] += 8
        
        return oversold
    
    def _calculate_overbought_condition(self, k: pd.Series, d: pd.Series, j: pd.Series) -> pd.Series:
        """计算超买条件"""
        overbought = pd.Series(0.0, index=k.index)
        
        # 基本超买条件：K、D、J都在80以上
        basic_overbought = (k > 80) & (d > 80) & (j > 80)
        overbought[basic_overbought] -= 10  # 超买是卖出信号，给负分
        
        # 强超买条件：K、D、J都在90以上
        strong_overbought = (k > 90) & (d > 90) & (j > 90)
        overbought[strong_overbought] -= 8
        
        # 连续超买：连续3天以上超买
        consecutive_overbought = basic_overbought.rolling(window=3).sum() >= 3
        overbought[consecutive_overbought] -= 5
        
        return overbought
    
    def _calculate_golden_cross(self, k: pd.Series, d: pd.Series) -> pd.Series:
        """计算KDJ金叉"""
        golden_cross = pd.Series(0.0, index=k.index)
        
        # K线上穿D线
        k_cross_d = (k > d) & (k.shift(1) <= d.shift(1))
        
        # 在低位金叉更有意义
        low_golden_cross = k_cross_d & (k < 50) & (d < 50)
        golden_cross[low_golden_cross] += 20
        
        # 一般金叉
        normal_golden_cross = k_cross_d & ~low_golden_cross
        golden_cross[normal_golden_cross] += 10
        
        return golden_cross
    
    def _calculate_death_cross(self, k: pd.Series, d: pd.Series) -> pd.Series:
        """计算KDJ死叉"""
        death_cross = pd.Series(0.0, index=k.index)
        
        # K线下穿D线
        k_cross_d = (k < d) & (k.shift(1) >= d.shift(1))
        
        # 在高位死叉更有意义
        high_death_cross = k_cross_d & (k > 50) & (d > 50)
        death_cross[high_death_cross] -= 15  # 死叉是卖出信号
        
        # 一般死叉
        normal_death_cross = k_cross_d & ~high_death_cross
        death_cross[normal_death_cross] -= 8
        
        return death_cross
    
    def _calculate_divergence(self, df: pd.DataFrame, j: pd.Series) -> pd.Series:
        """计算KDJ背离"""
        divergence = pd.Series(0.0, index=df.index)
        
        close = df['close']
        
        # 计算价格和J值的相关性
        for i in range(10, len(df)):
            price_window = close.iloc[i-9:i+1]
            j_window = j.iloc[i-9:i+1]
            
            # 价格创新高但J值没有创新高（顶背离）
            if (price_window.iloc[-1] == price_window.max() and 
                j_window.iloc[-1] < j_window.max()):
                divergence.iloc[i] -= 12  # 顶背离是卖出信号
            
            # 价格创新低但J值没有创新低（底背离）
            elif (price_window.iloc[-1] == price_window.min() and 
                  j_window.iloc[-1] > j_window.min()):
                divergence.iloc[i] += 15  # 底背离是买入信号
        
        return divergence
    
class MACDCondition(FORMULA_INDICATORS):
    """MACD条件指标 - 基于MACD指标的条件判断"""
    
    def _calculate_macd_golden_cross(self, macd_line: pd.Series, signal_line: pd.Series) -> pd.Series:
        """计算MACD金叉"""
        golden_cross = pd.Series(0.0, index=macd_line.index)
        
        # MACD线上穿信号线
        cross_up = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
        
        # 在零轴下方金叉更有意义
        below_zero_cross = cross_up & (macd_line < 0) & (signal_line < 0)
        golden_cross[below_zero_cross] += 20
        
        # 在零轴上方金叉
        above_zero_cross = cross_up & (macd_line > 0) & (signal_line > 0)
        golden_cross[above_zero_cross] += 15
        
        # 一般金叉
        normal_cross = cross_up & ~below_zero_cross & ~above_zero_cross
        golden_cross[normal_cross] += 10
        
        return golden_cross
    
    def _calculate_macd_death_cross(self, macd_line: pd.Series, signal_line: pd.Series) -> pd.Series:
        """计算MACD死叉"""
        death_cross = pd.Series(0.0, index=macd_line.index)
        
        # MACD线下穿信号线
        cross_down = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
        
        # 在零轴上方死叉更有意义
        above_zero_cross = cross_down & (macd_line > 0) & (signal_line > 0)
        death_cross[above_zero_cross] -= 15  # 死叉是卖出信号
        
        # 在零轴下方死叉
        below_zero_cross = cross_down & (macd_line < 0) & (signal_line < 0)
        death_cross[below_zero_cross] -= 10
        
        # 一般死叉
        normal_cross = cross_down & ~above_zero_cross & ~below_zero_cross
        death_cross[normal_cross] -= 8
        
        return death_cross
    
    def _calculate_zero_cross(self, macd_line: pd.Series) -> pd.Series:
        """计算MACD零轴穿越"""
        zero_cross = pd.Series(0.0, index=macd_line.index)
        
        # MACD上穿零轴
        cross_above_zero = (macd_line > 0) & (macd_line.shift(1) <= 0)
        zero_cross[cross_above_zero] += 18
        
        # MACD下穿零轴
        cross_below_zero = (macd_line < 0) & (macd_line.shift(1) >= 0)
        zero_cross[cross_below_zero] -= 12
        
        return zero_cross
    
    def _calculate_macd_divergence(self, df: pd.DataFrame, macd_line: pd.Series) -> pd.Series:
        """计算MACD背离"""
        divergence = pd.Series(0.0, index=df.index)
        
        close = df['close']
        
        # 计算价格和MACD的相关性
        for i in range(10, len(df)):
            price_window = close.iloc[i-9:i+1]
            macd_window = macd_line.iloc[i-9:i+1]
            
            # 价格创新高但MACD没有创新高（顶背离）
            if (price_window.iloc[-1] == price_window.max() and 
                macd_window.iloc[-1] < macd_window.max()):
                divergence.iloc[i] -= 15  # 顶背离是卖出信号
            
            # 价格创新低但MACD没有创新低（底背离）
            elif (price_window.iloc[-1] == price_window.min() and 
                  macd_window.iloc[-1] > macd_window.min()):
                divergence.iloc[i] += 18  # 底背离是买入信号
        
        return divergence
    
    def _calculate_histogram_signal(self, histogram: pd.Series) -> pd.Series:
        """计算MACD柱状体信号"""
        hist_signal = pd.Series(0.0, index=histogram.index)
        
        # 柱状体由负转正
        hist_turn_positive = (histogram > 0) & (histogram.shift(1) <= 0)
        hist_signal[hist_turn_positive] += 12
        
        # 柱状体由正转负
        hist_turn_negative = (histogram < 0) & (histogram.shift(1) >= 0)
        hist_signal[hist_turn_negative] -= 10
        
        # 柱状体连续放大
        hist_expanding = histogram.abs() > histogram.abs().shift(1)
        hist_signal[hist_expanding & (histogram > 0)] += 5
        hist_signal[hist_expanding & (histogram < 0)] -= 3
        
        # 柱状体收缩
        hist_contracting = histogram.abs() < histogram.abs().shift(1)
        hist_signal[hist_contracting & (histogram > 0)] -= 2
        hist_signal[hist_contracting & (histogram < 0)] += 3
        
        return hist_signal
    
class MACondition(FORMULA_INDICATORS):
    """
    MA条件指标 - 基于移动平均线的条件判断
    
    特点:
    1. 多周期移动平均线排列分析
    2. 价格与移动平均线的位置关系
    3. 移动平均线的趋势分析
    4. 支撑阻力位识别
    
    计算方法:
    1. 计算多个周期的移动平均线
    2. 分析移动平均线的排列（多头/空头排列）
    3. 判断价格与均线的关系
    4. 计算均线的趋势强度
    
    参数:
    - short_period: 短期均线周期，默认为5
    - medium_period: 中期均线周期，默认为10
    - long_period: 长期均线周期，默认为20
    - trend_period: 趋势分析周期，默认为30
    """
    
    def _calculate_bullish_alignment(self, close: pd.Series, ma_short: pd.Series, ma_medium: pd.Series, ma_long: pd.Series) -> pd.Series:
        """计算多头排列"""
        bullish = pd.Series(0.0, index=close.index)
        
        # 完美多头排列：价格 > 短期均线 > 中期均线 > 长期均线
        perfect_bullish = (close > ma_short) & (ma_short > ma_medium) & (ma_medium > ma_long)
        bullish[perfect_bullish] += 25.0
        
        # 部分多头排列：价格 > 短期均线 > 中期均线
        partial_bullish = (close > ma_short) & (ma_short > ma_medium) & ~perfect_bullish
        bullish[partial_bullish] += 15.0
        
        # 基本多头：价格 > 短期均线
        basic_bullish = (close > ma_short) & ~partial_bullish & ~perfect_bullish
        bullish[basic_bullish] += 8.0
        
        # 均线向上趋势加分
        ma_uptrend = (ma_short > ma_short.shift(1)) & (ma_medium > ma_medium.shift(1)) & (ma_long > ma_long.shift(1))
        bullish[ma_uptrend] += 10.0
        
        return bullish
    
    def _calculate_bearish_alignment(self, close: pd.Series, ma_short: pd.Series, ma_medium: pd.Series, ma_long: pd.Series) -> pd.Series:
        """计算空头排列"""
        bearish = pd.Series(0.0, index=close.index)
        
        # 完美空头排列：价格 < 短期均线 < 中期均线 < 长期均线
        perfect_bearish = (close < ma_short) & (ma_short < ma_medium) & (ma_medium < ma_long)
        bearish[perfect_bearish] += 20.0
        
        # 部分空头排列：价格 < 短期均线 < 中期均线
        partial_bearish = (close < ma_short) & (ma_short < ma_medium) & ~perfect_bearish
        bearish[partial_bearish] += 12.0
        
        # 基本空头：价格 < 短期均线
        basic_bearish = (close < ma_short) & ~partial_bearish & ~perfect_bearish
        bearish[basic_bearish] += 6.0
        
        # 均线向下趋势加分
        ma_downtrend = (ma_short < ma_short.shift(1)) & (ma_medium < ma_medium.shift(1)) & (ma_long < ma_long.shift(1))
        bearish[ma_downtrend] += 8.0
        
        return bearish
    
    def _calculate_price_position(self, close: pd.Series, ma_short: pd.Series, ma_medium: pd.Series, ma_long: pd.Series) -> pd.Series:
        """计算价格相对于均线的位置"""
        position = pd.Series(0.0, index=close.index)
        
        # 价格在所有均线之上
        above_all = (close > ma_short) & (close > ma_medium) & (close > ma_long)
        position[above_all] += 15.0
        
        # 价格在部分均线之上
        above_some = ((close > ma_short) | (close > ma_medium) | (close > ma_long)) & ~above_all
        position[above_some] += 5.0
        
        # 价格在所有均线之下
        below_all = (close < ma_short) & (close < ma_medium) & (close < ma_long)
        position[below_all] -= 10.0
        
        # 价格偏离度评分
        avg_ma = (ma_short + ma_medium + ma_long) / 3
        deviation = (close - avg_ma) / avg_ma.replace(0, np.nan)
        
        # 适度偏离给正分，过度偏离给负分
        moderate_deviation = (abs(deviation) > 0.02) & (abs(deviation) < 0.1)
        position[moderate_deviation] += 8.0
        
        excessive_deviation = abs(deviation) > 0.15
        position[excessive_deviation] -= 5.0
        
        return position
    
    def _calculate_trend_strength(self, ma_short: pd.Series, ma_medium: pd.Series, ma_long: pd.Series, ma_trend: pd.Series) -> pd.Series:
        """计算趋势强度"""
        strength = pd.Series(0.0, index=ma_short.index)
        
        # 计算各均线的斜率
        short_slope = (ma_short - ma_short.shift(5)) / ma_short.shift(5).replace(0, np.nan)
        medium_slope = (ma_medium - ma_medium.shift(5)) / ma_medium.shift(5).replace(0, np.nan)
        long_slope = (ma_long - ma_long.shift(5)) / ma_long.shift(5).replace(0, np.nan)
        
        # 上升趋势强度
        strong_uptrend = (short_slope > 0.02) & (medium_slope > 0.01) & (long_slope > 0.005)
        strength[strong_uptrend] += 20.0
        
        moderate_uptrend = (short_slope > 0.01) & (medium_slope > 0.005) & ~strong_uptrend
        strength[moderate_uptrend] += 12.0
        
        # 下降趋势强度
        strong_downtrend = (short_slope < -0.02) & (medium_slope < -0.01) & (long_slope < -0.005)
        strength[strong_downtrend] -= 15.0
        
        moderate_downtrend = (short_slope < -0.01) & (medium_slope < -0.005) & ~strong_downtrend
        strength[moderate_downtrend] -= 8.0
        
        return strength
    
    def _calculate_support_resistance(self, close: pd.Series, ma_short: pd.Series, ma_medium: pd.Series, ma_long: pd.Series) -> pd.Series:
        """计算支撑阻力"""
        support_resistance = pd.Series(0.0, index=close.index)
        
        # 均线作为支撑
        ma_support = ((close > ma_short * 0.99) & (close < ma_short * 1.01)) | \
                    ((close > ma_medium * 0.99) & (close < ma_medium * 1.01)) | \
                    ((close > ma_long * 0.99) & (close < ma_long * 1.01))
        
        # 在上升趋势中的支撑更有效
        uptrend = (ma_short > ma_short.shift(5)) & (ma_medium > ma_medium.shift(5))
        effective_support = ma_support & uptrend
        support_resistance[effective_support] += 12.0
        
        # 均线作为阻力
        ma_resistance = ((close > ma_short * 0.99) & (close < ma_short * 1.01)) | \
                       ((close > ma_medium * 0.99) & (close < ma_medium * 1.01)) | \
                       ((close > ma_long * 0.99) & (close < ma_long * 1.01))
        
        # 在下降趋势中的阻力更有效
        downtrend = (ma_short < ma_short.shift(5)) & (ma_medium < ma_medium.shift(5))
        effective_resistance = ma_resistance & downtrend
        support_resistance[effective_resistance] -= 8.0
        
        return support_resistance
    
    def _calculate_convergence_divergence(self, ma_short: pd.Series, ma_medium: pd.Series, ma_long: pd.Series) -> pd.Series:
        """计算均线收敛发散"""
        convergence_divergence = pd.Series(0.0, index=ma_short.index)
        
        # 计算均线间距
        short_medium_gap = abs(ma_short - ma_medium) / ma_medium.replace(0, np.nan)
        medium_long_gap = abs(ma_medium - ma_long) / ma_long.replace(0, np.nan)
        
        # 均线收敛（间距缩小）
        convergence = (short_medium_gap < short_medium_gap.shift(5)) & (medium_long_gap < medium_long_gap.shift(5))
        convergence_divergence[convergence] += 8.0
        
        # 均线发散（间距扩大）
        divergence = (short_medium_gap > short_medium_gap.shift(5)) & (medium_long_gap > medium_long_gap.shift(5))
        
        # 在趋势方向上的发散是好信号
        uptrend_divergence = divergence & (ma_short > ma_medium) & (ma_medium > ma_long)
        convergence_divergence[uptrend_divergence] += 10.0
        
        downtrend_divergence = divergence & (ma_short < ma_medium) & (ma_medium < ma_long)
        convergence_divergence[downtrend_divergence] -= 6.0
        
        return convergence_divergence
    
class GenericCondition(FORMULA_INDICATORS):
    """
    通用条件指标 - 基于多种技术指标的综合条件判断
    
    特点:
    1. 整合多种技术指标的信号
    2. 综合评估市场状态
    3. 提供多维度的条件判断
    4. 适用于各种市场环境
    
    计算方法:
    1. 计算多个基础技术指标
    2. 对各指标进行标准化处理
    3. 根据权重计算综合评分
    4. 生成买卖信号和风险评估
    
    参数:
    - ma_period: 移动平均周期，默认为20
    - rsi_period: RSI周期，默认为14
    - volume_period: 成交量周期，默认为10
    - trend_period: 趋势分析周期，默认为30
    """
    
    def _calculate_price_condition(self, close: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
        """计算价格条件"""
        price_condition = pd.Series(0.0, index=close.index)
        
        # 移动平均条件
        ma = close.rolling(window=self.ma_period).mean()
        
        # 价格在均线上方
        above_ma = close > ma
        price_condition[above_ma] += 15.0
        
        # 价格突破近期高点
        recent_high = high.rolling(window=10).max()
        breakout_high = close > recent_high.shift(1)
        price_condition[breakout_high] += 20.0
        
        # 价格跌破近期低点
        recent_low = low.rolling(window=10).min()
        breakdown_low = close < recent_low.shift(1)
        price_condition[breakdown_low] -= 15.0
        
        # 价格相对位置
        price_range = high.rolling(window=20).max() - low.rolling(window=20).min()
        price_position = (close - low.rolling(window=20).min()) / price_range.replace(0, np.nan)
        
        # 高位给正分，低位给负分
        high_position = price_position > 0.8
        price_condition[high_position] += 10.0
        
        low_position = price_position < 0.2
        price_condition[low_position] -= 8.0
        
        return price_condition
    
    def _calculate_trend_condition(self, close: pd.Series) -> pd.Series:
        """计算趋势条件"""
        trend_condition = pd.Series(0.0, index=close.index)
        
        # 短期趋势
        ma_short = close.rolling(window=5).mean()
        ma_medium = close.rolling(window=20).mean()
        ma_long = close.rolling(window=self.trend_period).mean()
        
        # 多头排列
        bullish_alignment = (ma_short > ma_medium) & (ma_medium > ma_long)
        trend_condition[bullish_alignment] += 25.0
        
        # 空头排列
        bearish_alignment = (ma_short < ma_medium) & (ma_medium < ma_long)
        trend_condition[bearish_alignment] -= 20.0
        
        # 趋势强度
        trend_strength = (close - close.shift(10)) / close.shift(10).replace(0, np.nan)
        
        strong_uptrend = trend_strength > 0.05
        trend_condition[strong_uptrend] += 15.0
        
        strong_downtrend = trend_strength < -0.05
        trend_condition[strong_downtrend] -= 12.0
        
        return trend_condition
    
    def _calculate_momentum_condition(self, close: pd.Series) -> pd.Series:
        """计算动量条件"""
        momentum_condition = pd.Series(0.0, index=close.index)
        
        # RSI计算
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        
        # RSI超买超卖
        rsi_oversold = rsi < 30
        momentum_condition[rsi_oversold] += 20.0
        
        rsi_overbought = rsi > 70
        momentum_condition[rsi_overbought] -= 15.0
        
        # 动量变化
        momentum = close - close.shift(5)
        momentum_change = momentum - momentum.shift(5)
        
        # 动量加速
        momentum_acceleration = momentum_change > 0
        momentum_condition[momentum_acceleration] += 10.0
        
        # 动量减速
        momentum_deceleration = momentum_change < 0
        momentum_condition[momentum_deceleration] -= 8.0
        
        return momentum_condition
    
    def _calculate_volume_condition(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """计算成交量条件"""
        volume_condition = pd.Series(0.0, index=close.index)
        
        if volume.sum() == 0:  # 如果没有成交量数据
            return volume_condition
        
        # 成交量均线
        volume_ma = volume.rolling(window=self.volume_period).mean()
        
        # 放量上涨
        volume_up = (volume > volume_ma * 1.5) & (close > close.shift(1))
        volume_condition[volume_up] += 15.0
        
        # 放量下跌
        volume_down = (volume > volume_ma * 1.5) & (close < close.shift(1))
        volume_condition[volume_down] -= 10.0
        
        # 缩量上涨
        volume_shrink_up = (volume < volume_ma * 0.7) & (close > close.shift(1))
        volume_condition[volume_shrink_up] += 8.0
        
        # 缩量下跌
        volume_shrink_down = (volume < volume_ma * 0.7) & (close < close.shift(1))
        volume_condition[volume_shrink_down] -= 5.0
        
        return volume_condition
    
    def _calculate_volatility_condition(self, close: pd.Series) -> pd.Series:
        """计算波动率条件"""
        volatility_condition = pd.Series(0.0, index=close.index)
        
        # 计算波动率
        returns = close.pct_change()
        volatility = returns.rolling(window=20).std()
        
        # 波动率分位数
        volatility_percentile = volatility.rolling(window=50).rank(pct=True)
        
        # 低波动率给正分（稳定上涨）
        low_volatility = volatility_percentile < 0.3
        volatility_condition[low_volatility] += 10.0
        
        # 高波动率给负分（风险较高）
        high_volatility = volatility_percentile > 0.8
        volatility_condition[high_volatility] -= 8.0
        
        # 波动率突增
        volatility_spike = volatility > volatility.rolling(window=10).mean() * 2
        volatility_condition[volatility_spike] -= 12.0
        
        return volatility_condition
    
    def _calculate_support_resistance_condition(self, close: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
        """计算支撑阻力条件"""
        support_resistance_condition = pd.Series(0.0, index=close.index)
        
        # 计算支撑阻力位
        resistance = high.rolling(window=20).max()
        support = low.rolling(window=20).min()
        
        # 突破阻力位
        breakout_resistance = close > resistance.shift(1)
        support_resistance_condition[breakout_resistance] += 12.0
        
        # 跌破支撑位
        breakdown_support = close < support.shift(1)
        support_resistance_condition[breakdown_support] -= 10.0
        
        # 接近支撑位反弹
        near_support = (close - support) / support.replace(0, np.nan) < 0.02
        support_bounce = near_support & (close > close.shift(1))
        support_resistance_condition[support_bounce] += 8.0
        
        # 接近阻力位回调
        near_resistance = (resistance - close) / resistance.replace(0, np.nan) < 0.02
        resistance_rejection = near_resistance & (close < close.shift(1))
        support_resistance_condition[resistance_rejection] -= 6.0
        
        return support_resistance_condition
    
# 为了向后兼容，创建别名
formula_indicators = FORMULA_INDICATORS
