#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
成交量震荡指标(VOSC)

通过对成交量的长短期移动平均差值的百分比来衡量成交量的变化和趋势
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple

from indicators.base_indicator import BaseIndicator
from indicators.common import crossover, crossunder
from utils.logger import get_logger
from indicators.pattern_registry import PatternRegistry, PatternType, PatternStrength

logger = get_logger(__name__)


class VOSC(BaseIndicator):
    """
    成交量震荡指标(VOSC) (VOSC)
    
    分类：量能类指标
    描述：通过对成交量的长短期移动平均差值的百分比来衡量成交量的变化和趋势
    """
    
    def __init__(self, short_period: int = 12, long_period: int = 26):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """
        初始化成交量震荡指标(VOSC)指标
        
        Args:
            short_period: 短期移动平均周期，默认为12
            long_period: 长期移动平均周期，默认为26
        """
        super().__init__(name="VOSC", description="成交量震荡指标，通过对成交量的长短期移动平均差值的百分比来衡量成交量的变化和趋势")
        self.short_period = short_period
        self.long_period = long_period
        
    def _validate_dataframe(self, df: pd.DataFrame, required_columns: List[str]) -> None:
        """
        验证DataFrame是否包含所需的列
        
        Args:
            df: 输入数据
            required_columns: 所需的列名列表
            
        Raises:
            ValueError: 如果缺少必要的列
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"输入数据缺少必要的列: {', '.join(missing_columns)}")
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算VOSC指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
                
        Returns:
            包含VOSC指标的DataFrame
        """
        return self.calculate(df)
        
    def _calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算成交量震荡指标(VOSC)指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
                必须包含以下列：
                - volume: 成交量
                
        Returns:
            添加了VOSC指标列的DataFrame
        """
        if df.empty:
            return df
            
        # 确保数据包含必要的列
        required_columns = ['volume']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 计算短期和长期成交量移动平均
        short_ma = df_copy['volume'].rolling(window=self.short_period).mean()
        long_ma = df_copy['volume'].rolling(window=self.long_period).mean()
        
        # 计算VOSC值
        # VOSC = (短期成交量均线 - 长期成交量均线) / 长期成交量均线 * 100
        df_copy['vosc'] = (short_ma - long_ma) / long_ma * 100
        
        # 计算VOSC的移动平均作为信号线
        df_copy['vosc_signal'] = df_copy['vosc'].rolling(window=9).mean()
        
        # 存储结果
        self._result = df_copy[['vosc', 'vosc_signal']]
        
        return df_copy
        
    def get_signals(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成成交量震荡指标(VOSC)指标交易信号
        
        Args:
            df: 包含价格数据和VOSC指标的DataFrame
            **kwargs: 额外参数
                
        Returns:
            添加了信号列的DataFrame:
            - vosc_buy_signal: 1=买入信号, 0=无信号
            - vosc_sell_signal: 1=卖出信号, 0=无信号
        """
        if df.empty:
            return df
            
        # 检查必要的指标列是否存在
        required_columns = ['vosc', 'vosc_signal']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 初始化信号列
        df_copy['vosc_buy_signal'] = 0
        df_copy['vosc_sell_signal'] = 0
        
        # VOSC上穿信号线为买入信号
        for i in range(1, len(df_copy)):
            if df_copy['vosc'].iloc[i-1] < df_copy['vosc_signal'].iloc[i-1] and \
               df_copy['vosc'].iloc[i] > df_copy['vosc_signal'].iloc[i]:
                df_copy.iloc[i, df_copy.columns.get_loc('vosc_buy_signal')] = 1
            
            # VOSC下穿信号线为卖出信号
            elif df_copy['vosc'].iloc[i-1] > df_copy['vosc_signal'].iloc[i-1] and \
                 df_copy['vosc'].iloc[i] < df_copy['vosc_signal'].iloc[i]:
                df_copy.iloc[i, df_copy.columns.get_loc('vosc_sell_signal')] = 1
        
        return df_copy
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算VOSC原始评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 原始评分序列（0-100分）
        """
        # 确保已计算VOSC
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)
        
        score = pd.Series(50.0, index=data.index)  # 基础分50分
        
        # 1. VOSC零轴穿越评分
        zero_cross_score = self._calculate_vosc_zero_cross_score()
        score += zero_cross_score
        
        # 2. VOSC与信号线交叉评分
        signal_cross_score = self._calculate_vosc_signal_cross_score()
        score += signal_cross_score
        
        # 3. VOSC趋势评分
        trend_score = self._calculate_vosc_trend_score()
        score += trend_score
        
        # 4. VOSC极值评分
        extreme_score = self._calculate_vosc_extreme_score()
        score += extreme_score
        
        # 5. VOSC与价格关系评分
        price_relation_score = self._calculate_vosc_price_relation_score(data)
        score += price_relation_score
        
        return np.clip(score, 0, 100)
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别VOSC技术形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别出的形态列表
        """
        patterns = []
        
        # 确保已计算VOSC
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return patterns
        
        # 1. 检测VOSC零轴穿越形态
        zero_cross_patterns = self._detect_vosc_zero_cross_patterns()
        patterns.extend(zero_cross_patterns)
        
        # 2. 检测VOSC与信号线交叉形态
        signal_cross_patterns = self._detect_vosc_signal_cross_patterns()
        patterns.extend(signal_cross_patterns)
        
        # 3. 检测VOSC趋势形态
        trend_patterns = self._detect_vosc_trend_patterns()
        patterns.extend(trend_patterns)
        
        # 4. 检测VOSC极值形态
        extreme_patterns = self._detect_vosc_extreme_patterns()
        patterns.extend(extreme_patterns)
        
        # 5. 检测VOSC与价格关系形态
        price_relation_patterns = self._detect_vosc_price_relation_patterns(data)
        patterns.extend(price_relation_patterns)
        
        return patterns
    
    def _calculate_vosc_zero_cross_score(self) -> pd.Series:
        """
        计算VOSC零轴穿越评分
        
        Returns:
            pd.Series: 零轴穿越评分
        """
        zero_cross_score = pd.Series(0.0, index=self._result.index)
        
        vosc_values = self._result['vosc']
        
        # VOSC上穿零轴+25分
        vosc_cross_up_zero = crossover(vosc_values, 0)
        zero_cross_score += vosc_cross_up_zero * 25
        
        # VOSC下穿零轴-25分
        vosc_cross_down_zero = crossunder(vosc_values, 0)
        zero_cross_score -= vosc_cross_down_zero * 25
        
        # VOSC在零轴上方+8分
        vosc_above_zero = vosc_values > 0
        zero_cross_score += vosc_above_zero * 8
        
        # VOSC在零轴下方-8分
        vosc_below_zero = vosc_values < 0
        zero_cross_score -= vosc_below_zero * 8
        
        return zero_cross_score
    
    def _calculate_vosc_signal_cross_score(self) -> pd.Series:
        """
        计算VOSC与信号线交叉评分
        
        Returns:
            pd.Series: 信号线交叉评分
        """
        signal_cross_score = pd.Series(0.0, index=self._result.index)
        
        vosc_values = self._result['vosc']
        signal_values = self._result['vosc_signal']
        
        # VOSC上穿信号线+20分
        vosc_cross_up_signal = crossover(vosc_values, signal_values)
        signal_cross_score += vosc_cross_up_signal * 20
        
        # VOSC下穿信号线-20分
        vosc_cross_down_signal = crossunder(vosc_values, signal_values)
        signal_cross_score -= vosc_cross_down_signal * 20
        
        # VOSC在信号线上方+5分
        vosc_above_signal = vosc_values > signal_values
        signal_cross_score += vosc_above_signal * 5
        
        # VOSC在信号线下方-5分
        vosc_below_signal = vosc_values < signal_values
        signal_cross_score -= vosc_below_signal * 5
        
        return signal_cross_score
    
    def _calculate_vosc_trend_score(self) -> pd.Series:
        """
        计算VOSC趋势评分
        
        Returns:
            pd.Series: 趋势评分
        """
        trend_score = pd.Series(0.0, index=self._result.index)
        
        vosc_values = self._result['vosc']
        
        # VOSC上升趋势+10分
        vosc_rising = vosc_values > vosc_values.shift(1)
        trend_score += vosc_rising * 10
        
        # VOSC下降趋势-10分
        vosc_falling = vosc_values < vosc_values.shift(1)
        trend_score -= vosc_falling * 10
        
        # VOSC连续上升（3个周期）+15分
        if len(vosc_values) >= 3:
            consecutive_rising = (
                (vosc_values > vosc_values.shift(1)) &
                (vosc_values.shift(1) > vosc_values.shift(2)) &
                (vosc_values.shift(2) > vosc_values.shift(3))
            )
            trend_score += consecutive_rising * 15
        
        # VOSC连续下降（3个周期）-15分
        if len(vosc_values) >= 3:
            consecutive_falling = (
                (vosc_values < vosc_values.shift(1)) &
                (vosc_values.shift(1) < vosc_values.shift(2)) &
                (vosc_values.shift(2) < vosc_values.shift(3))
            )
            trend_score -= consecutive_falling * 15
        
        return trend_score
    
    def _calculate_vosc_extreme_score(self) -> pd.Series:
        """
        计算VOSC极值评分
        
        Returns:
            pd.Series: 极值评分
        """
        extreme_score = pd.Series(0.0, index=self._result.index)
        
        vosc_values = self._result['vosc']
        
        # VOSC极度超买（>50）-20分
        vosc_extreme_overbought = vosc_values > 50
        extreme_score -= vosc_extreme_overbought * 20
        
        # VOSC极度超卖（<-50）+20分
        vosc_extreme_oversold = vosc_values < -50
        extreme_score += vosc_extreme_oversold * 20
        
        # VOSC超买（>20）-10分
        vosc_overbought = (vosc_values > 20) & (vosc_values <= 50)
        extreme_score -= vosc_overbought * 10
        
        # VOSC超卖（<-20）+10分
        vosc_oversold = (vosc_values < -20) & (vosc_values >= -50)
        extreme_score += vosc_oversold * 10
        
        return extreme_score
    
    def _calculate_vosc_price_relation_score(self, data: pd.DataFrame) -> pd.Series:
        """
        计算VOSC与价格关系评分
        
        Args:
            data: 价格数据
            
        Returns:
            pd.Series: 价格关系评分
        """
        price_relation_score = pd.Series(0.0, index=self._result.index)
        
        if 'close' not in data.columns:
            return price_relation_score
        
        close_price = data['close']
        vosc_values = self._result['vosc']
        
        # 计算价格变化率
        price_change = close_price.pct_change()
        
        # 价格上涨且VOSC为正+12分
        price_up_vosc_positive = (price_change > 0) & (vosc_values > 0)
        price_relation_score += price_up_vosc_positive * 12
        
        # 价格下跌且VOSC为负-12分
        price_down_vosc_negative = (price_change < 0) & (vosc_values < 0)
        price_relation_score -= price_down_vosc_negative * 12
        
        # 价格上涨但VOSC为负（量价背离）-15分
        price_up_vosc_negative = (price_change > 0) & (vosc_values < 0)
        price_relation_score -= price_up_vosc_negative * 15
        
        # 价格下跌但VOSC为正（量价背离）+15分
        price_down_vosc_positive = (price_change < 0) & (vosc_values > 0)
        price_relation_score += price_down_vosc_positive * 15
        
        return price_relation_score
    
    def _detect_vosc_zero_cross_patterns(self) -> List[str]:
        """
        检测VOSC零轴穿越形态
        
        Returns:
            List[str]: 零轴穿越形态列表
        """
        patterns = []
        
        vosc_values = self._result['vosc']
        
        # 检查最近的零轴穿越
        recent_periods = min(5, len(vosc_values))
        recent_vosc = vosc_values.tail(recent_periods)
        
        if crossover(recent_vosc, 0).any():
            patterns.append("VOSC上穿零轴")
        
        if crossunder(recent_vosc, 0).any():
            patterns.append("VOSC下穿零轴")
        
        # 检查当前位置
        if len(vosc_values) > 0:
            current_vosc = vosc_values.iloc[-1]
            if not pd.isna(current_vosc):
                if current_vosc > 0:
                    patterns.append("VOSC零轴上方")
                elif current_vosc < 0:
                    patterns.append("VOSC零轴下方")
                else:
                    patterns.append("VOSC零轴位置")
        
        return patterns
    
    def _detect_vosc_signal_cross_patterns(self) -> List[str]:
        """
        检测VOSC与信号线交叉形态
        
        Returns:
            List[str]: 信号线交叉形态列表
        """
        patterns = []
        
        vosc_values = self._result['vosc']
        signal_values = self._result['vosc_signal']
        
        # 检查最近的信号线穿越
        recent_periods = min(5, len(vosc_values))
        recent_vosc = vosc_values.tail(recent_periods)
        recent_signal = signal_values.tail(recent_periods)
        
        if crossover(recent_vosc, recent_signal).any():
            patterns.append("VOSC上穿信号线")
        
        if crossunder(recent_vosc, recent_signal).any():
            patterns.append("VOSC下穿信号线")
        
        # 检查当前位置关系
        if len(vosc_values) > 0 and len(signal_values) > 0:
            current_vosc = vosc_values.iloc[-1]
            current_signal = signal_values.iloc[-1]
            
            if not pd.isna(current_vosc) and not pd.isna(current_signal):
                if current_vosc > current_signal:
                    patterns.append("VOSC信号线上方")
                elif current_vosc < current_signal:
                    patterns.append("VOSC信号线下方")
                else:
                    patterns.append("VOSC信号线重合")
        
        return patterns
    
    def _detect_vosc_trend_patterns(self) -> List[str]:
        """
        检测VOSC趋势形态
        
        Returns:
            List[str]: 趋势形态列表
        """
        patterns = []
        
        vosc_values = self._result['vosc']
        
        # 检查VOSC趋势
        if len(vosc_values) >= 3:
            recent_3 = vosc_values.tail(3)
            if len(recent_3) == 3 and not recent_3.isna().any():
                if (recent_3.iloc[2] > recent_3.iloc[1] > recent_3.iloc[0]):
                    patterns.append("VOSC连续上升")
                elif (recent_3.iloc[2] < recent_3.iloc[1] < recent_3.iloc[0]):
                    patterns.append("VOSC连续下降")
        
        # 检查当前趋势
        if len(vosc_values) >= 2:
            current_vosc = vosc_values.iloc[-1]
            prev_vosc = vosc_values.iloc[-2]
            
            if not pd.isna(current_vosc) and not pd.isna(prev_vosc):
                if current_vosc > prev_vosc:
                    patterns.append("VOSC上升")
                elif current_vosc < prev_vosc:
                    patterns.append("VOSC下降")
                else:
                    patterns.append("VOSC平稳")
        
        return patterns
    
    def _detect_vosc_extreme_patterns(self) -> List[str]:
        """
        检测VOSC极值形态
        
        Returns:
            List[str]: 极值形态列表
        """
        patterns = []
        
        vosc_values = self._result['vosc']
        
        if len(vosc_values) > 0:
            current_vosc = vosc_values.iloc[-1]
            
            if pd.isna(current_vosc):
                return patterns
            
            if current_vosc > 50:
                patterns.append("VOSC极度超买")
            elif current_vosc > 20:
                patterns.append("VOSC超买")
            elif current_vosc < -50:
                patterns.append("VOSC极度超卖")
            elif current_vosc < -20:
                patterns.append("VOSC超卖")
            elif -10 <= current_vosc <= 10:
                patterns.append("VOSC中性区域")
        
        return patterns
    
    def _detect_vosc_price_relation_patterns(self, data: pd.DataFrame) -> List[str]:
        """
        检测VOSC与价格关系形态
        
        Args:
            data: 价格数据
            
        Returns:
            List[str]: 价格关系形态列表
        """
        patterns = []
        
        if 'close' not in data.columns:
            return patterns
        
        close_price = data['close']
        vosc_values = self._result['vosc']
        
        if len(close_price) >= 5:
            # 检查最近5个周期的价格和VOSC关系
            recent_price = close_price.tail(5)
            recent_vosc = vosc_values.tail(5)
            
            # 计算价格和VOSC的趋势
            price_trend = recent_price.iloc[-1] - recent_price.iloc[0]
            vosc_trend = recent_vosc.iloc[-1] - recent_vosc.iloc[0]
            
            # 量价配合
            if price_trend > 0 and vosc_trend > 0:
                patterns.append("量价配合上涨")
            elif price_trend < 0 and vosc_trend < 0:
                patterns.append("量价配合下跌")
            # 量价背离
            elif price_trend > 0 and vosc_trend < 0:
                patterns.append("量价背离上涨")
            elif price_trend < 0 and vosc_trend > 0:
                patterns.append("量价背离下跌")
            else:
                patterns.append("量价关系中性")
        
        return patterns
        
    def generate_trading_signals(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """
        生成交易信号
        
        Args:
            data: 输入数据
            **kwargs: 额外参数
            
        Returns:
            Dict[str, pd.Series]: 包含交易信号的字典
        """
        # 确保已计算指标
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        # 初始化信号
        signals = {}
        signals['buy_signal'] = pd.Series(False, index=data.index)
        signals['sell_signal'] = pd.Series(False, index=data.index)
        signals['signal_strength'] = pd.Series(0, index=data.index)
    
        # 在这里实现指标特定的信号生成逻辑
        # 此处提供默认实现
    
        return signals
        
    def plot(self, df: pd.DataFrame, ax=None, **kwargs):
        """
        绘制成交量震荡指标(VOSC)指标图表
        
        Args:
            df: 包含VOSC指标的DataFrame
            ax: matplotlib轴对象，如果为None则创建新的
            **kwargs: 额外绘图参数
            
        Returns:
            matplotlib轴对象
        """
        import matplotlib.pyplot as plt
        
        # 检查必要的指标列是否存在
        required_columns = ['vosc', 'vosc_signal']
        self._validate_dataframe(df, required_columns)
        
        # 创建新的轴对象（如果未提供）
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
            
        # 绘制VOSC指标线
        ax.plot(df.index, df['vosc'], label='VOSC')
        ax.plot(df.index, df['vosc_signal'], label='信号线', linestyle='--')
        
        # 添加零轴线
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        ax.set_ylabel('成交量震荡指标(VOSC)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        return ax

    def _register_vosc_patterns(self):
        """
        注册VOSC指标相关形态
        """
        # 获取PatternRegistry实例
        registry = PatternRegistry()
        
        # 注册VOSC零轴穿越形态
        registry.register(
            pattern_id="VOSC_CROSS_ABOVE_ZERO",
            display_name="VOSC上穿零轴",
            description="VOSC从下方穿越零轴，表明短期成交量超过长期成交量，看涨信号",
            indicator_id="VOSC",
            pattern_type=PatternType.BULLISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=15.0
        )
        
        registry.register(
            pattern_id="VOSC_CROSS_BELOW_ZERO",
            display_name="VOSC下穿零轴",
            description="VOSC从上方穿越零轴，表明短期成交量低于长期成交量，看跌信号",
            indicator_id="VOSC",
            pattern_type=PatternType.BEARISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=-15.0
        )
        
        # 注册VOSC与信号线交叉形态
        registry.register(
            pattern_id="VOSC_GOLDEN_CROSS",
            display_name="VOSC金叉",
            description="VOSC上穿信号线，表明成交量动量增强",
            indicator_id="VOSC",
            pattern_type=PatternType.BULLISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=12.0
        )
        
        registry.register(
            pattern_id="VOSC_DEATH_CROSS",
            display_name="VOSC死叉",
            description="VOSC下穿信号线，表明成交量动量减弱",
            indicator_id="VOSC",
            pattern_type=PatternType.BEARISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=-12.0
        )
        
        # 注册VOSC趋势形态
        registry.register(
            pattern_id="VOSC_UPTREND",
            display_name="VOSC上升趋势",
            description="VOSC连续上升，表明成交量持续增加",
            indicator_id="VOSC",
            pattern_type=PatternType.BULLISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=10.0
        )
        
        registry.register(
            pattern_id="VOSC_DOWNTREND",
            display_name="VOSC下降趋势",
            description="VOSC连续下降，表明成交量持续萎缩",
            indicator_id="VOSC",
            pattern_type=PatternType.BEARISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=-10.0
        )
        
        # 注册VOSC极值形态
        registry.register(
            pattern_id="VOSC_EXTREME_HIGH",
            display_name="VOSC极高值",
            description="VOSC值异常高，表明短期成交量远超长期成交量，可能出现爆量",
            indicator_id="VOSC",
            pattern_type=PatternType.BULLISH,
            default_strength=PatternStrength.STRONG,
            score_impact=18.0
        )
        
        registry.register(
            pattern_id="VOSC_EXTREME_LOW",
            display_name="VOSC极低值",
            description="VOSC值异常低，表明短期成交量远低于长期成交量，可能出现极度萎缩",
            indicator_id="VOSC",
            pattern_type=PatternType.BEARISH,
            default_strength=PatternStrength.STRONG,
            score_impact=-18.0
        )
        
        # 注册VOSC与价格关系形态
        registry.register(
            pattern_id="VOSC_PRICE_CONFIRMATION",
            display_name="VOSC价格确认",
            description="VOSC与价格同向变动，成交量确认价格趋势",
            indicator_id="VOSC",
            pattern_type=PatternType.NEUTRAL,
            default_strength=PatternStrength.MEDIUM,
            score_impact=10.0
        )
        
        registry.register(
            pattern_id="VOSC_PRICE_NON_CONFIRMATION",
            display_name="VOSC价格不确认",
            description="VOSC与价格反向变动，成交量不支持价格趋势",
            indicator_id="VOSC",
            pattern_type=PatternType.NEUTRAL,
            default_strength=PatternStrength.MEDIUM,
            score_impact=-10.0
        )

