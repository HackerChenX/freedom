#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
动量指标(Momentum)

衡量价格变化速度，预测趋势转折点
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple

from indicators.base_indicator import BaseIndicator
from indicators.common import crossover, crossunder
from utils.logger import get_logger

logger = get_logger(__name__)


class Momentum(BaseIndicator):
    """
    动量指标(Momentum) (MTM)
    
    分类：动量类指标
    描述：衡量价格变化速度，预测趋势转折点
    """
    
    def __init__(self, period: int = 10, signal_period: int = 6, calculation_method: str = "difference"):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """
        初始化动量指标(Momentum)
        
        Args:
            period: 计算周期，默认为10
            signal_period: 信号线平滑周期，默认为6
            calculation_method: 计算方法，可选"difference"(差值法)或"ratio"(比率法)
        """
        super().__init__(name="Momentum", description="动量指标，衡量价格变化速度，预测趋势转折点")
        self.period = period
        self.signal_period = signal_period
        self.calculation_method = calculation_method
    
    def set_parameters(self, period: int = None, signal_period: int = None, calculation_method: str = None):
        """
        设置指标参数
        """
        if period is not None:
            self.period = period
        if signal_period is not None:
            self.signal_period = signal_period
        if calculation_method is not None:
            self.calculation_method = calculation_method
    
    def _validate_dataframe(self, df: pd.DataFrame, required_columns: List[str]) -> None:
        """
        验证DataFrame是否包含所需的列
        
        Args:
            df: 包含价格数据的DataFrame
            required_columns: 所需的列名列表
        
        Raises:
            ValueError: 如果DataFrame不包含所需的列，或者行数少于所需的最小行数
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"DataFrame缺少必要的列: {', '.join(missing_columns)}")
        
        min_rows = self.period + self.signal_period
        if len(df) < min_rows:
            raise ValueError(f"DataFrame至少需要 {min_rows} 行数据，但只有 {len(df)} 行")
    
    def _calculate(self, df: pd.DataFrame, price_column: str = "close") -> pd.DataFrame:
        """
        计算动量指标
        
        Args:
            df: 包含价格数据的DataFrame
            price_column: 用于计算动量的价格列名，默认为'close'
        
        Returns:
            包含动量指标结果的DataFrame
        """
        required_columns = [price_column]
        self._validate_dataframe(df, required_columns)
        
        result = pd.DataFrame(index=df.index)
        
        # 计算动量
        if self.calculation_method == "difference":
            # 差值法: 当前价格 - N期前价格
            result['mtm'] = df[price_column] - df[price_column].shift(self.period)
        else:
            # 比率法: (当前价格 / N期前价格) * 100
            result['mtm'] = (df[price_column] / df[price_column].shift(self.period)) * 100
        
        # 计算信号线(MTM的移动平均)
        result['signal'] = result['mtm'].rolling(window=self.signal_period).mean()
        
        # 存储结果
        self._result = result
        
        return result
    
    def generate_signals(self, df: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        Args:
            df: 原始数据DataFrame
            result: 计算指标后的DataFrame
        
        Returns:
            添加了交易信号的DataFrame
        """
        # 初始化信号列
        result['buy_signal'] = 0
        result['sell_signal'] = 0
        result['mtm_buy_signal'] = 0  # 添加与指标名称相关的信号列
        result['mtm_sell_signal'] = 0  # 添加与指标名称相关的信号列
        
        # MTM上穿信号线买入
        buy_cross = crossover(result['mtm'].values, result['signal'].values)
        result.loc[buy_cross, 'buy_signal'] = 1
        result.loc[buy_cross, 'mtm_buy_signal'] = 1
        
        # MTM下穿信号线卖出
        sell_cross = crossunder(result['mtm'].values, result['signal'].values)
        result.loc[sell_cross, 'sell_signal'] = 1
        result.loc[sell_cross, 'mtm_sell_signal'] = 1
        
        return result
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算Momentum原始评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 原始评分序列（0-100分）
        """
        # 确保已计算Momentum
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)
        
        score = pd.Series(50.0, index=data.index)  # 基础分50分
        
        # 1. Momentum零轴穿越评分
        zero_cross_score = self._calculate_momentum_zero_cross_score()
        score += zero_cross_score
        
        # 2. Momentum与信号线交叉评分
        signal_cross_score = self._calculate_momentum_signal_cross_score()
        score += signal_cross_score
        
        # 3. Momentum趋势评分
        trend_score = self._calculate_momentum_trend_score()
        score += trend_score
        
        # 4. Momentum背离评分
        divergence_score = self._calculate_momentum_divergence_score(data)
        score += divergence_score
        
        # 5. Momentum强度评分
        strength_score = self._calculate_momentum_strength_score()
        score += strength_score
        
        return np.clip(score, 0, 100)
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别Momentum技术形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别出的形态列表
        """
        patterns = []
        
        # 确保已计算Momentum
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return patterns
        
        # 1. 检测Momentum零轴穿越形态
        zero_cross_patterns = self._detect_momentum_zero_cross_patterns()
        patterns.extend(zero_cross_patterns)
        
        # 2. 检测Momentum与信号线交叉形态
        signal_cross_patterns = self._detect_momentum_signal_cross_patterns()
        patterns.extend(signal_cross_patterns)
        
        # 3. 检测Momentum趋势形态
        trend_patterns = self._detect_momentum_trend_patterns()
        patterns.extend(trend_patterns)
        
        # 4. 检测Momentum背离形态
        divergence_patterns = self._detect_momentum_divergence_patterns(data)
        patterns.extend(divergence_patterns)
        
        # 5. 检测Momentum强度形态
        strength_patterns = self._detect_momentum_strength_patterns()
        patterns.extend(strength_patterns)
        
        return patterns
    
    def _calculate_momentum_zero_cross_score(self) -> pd.Series:
        """
        计算Momentum零轴穿越评分

        Returns:
            pd.Series: 零轴穿越评分
        """
        zero_cross_score = pd.Series(0.0, index=self._result.index)

        mtm_values = self._result['mtm']
        valid_mask = mtm_values.notna()

        # Momentum上穿零轴+15分
        mtm_cross_up_zero = crossover(mtm_values, 0) & valid_mask
        zero_cross_score += mtm_cross_up_zero * 15

        # Momentum下穿零轴-15分
        mtm_cross_down_zero = crossunder(mtm_values, 0) & valid_mask
        zero_cross_score -= mtm_cross_down_zero * 15

        # Momentum在零轴上方+5分
        mtm_above_zero = (mtm_values > 0) & valid_mask
        zero_cross_score += mtm_above_zero * 5

        # Momentum在零轴下方-5分
        mtm_below_zero = (mtm_values < 0) & valid_mask
        zero_cross_score -= mtm_below_zero * 5

        return zero_cross_score
    
    def _calculate_momentum_signal_cross_score(self) -> pd.Series:
        """
        计算Momentum与信号线交叉评分

        Returns:
            pd.Series: 信号线交叉评分
        """
        signal_cross_score = pd.Series(0.0, index=self._result.index)

        mtm_values = self._result['mtm']
        signal_values = self._result['signal']
        valid_mask = mtm_values.notna() & signal_values.notna()

        # Momentum上穿信号线+12分
        mtm_cross_up_signal = crossover(mtm_values, signal_values) & valid_mask
        signal_cross_score += mtm_cross_up_signal * 12

        # Momentum下穿信号线-12分
        mtm_cross_down_signal = crossunder(mtm_values, signal_values) & valid_mask
        signal_cross_score -= mtm_cross_down_signal * 12

        # Momentum在信号线上方+3分
        mtm_above_signal = (mtm_values > signal_values) & valid_mask
        signal_cross_score += mtm_above_signal * 3

        # Momentum在信号线下方-3分
        mtm_below_signal = (mtm_values < signal_values) & valid_mask
        signal_cross_score -= mtm_below_signal * 3

        return signal_cross_score
    
    def _calculate_momentum_trend_score(self) -> pd.Series:
        """
        计算Momentum趋势评分

        Returns:
            pd.Series: 趋势评分
        """
        trend_score = pd.Series(0.0, index=self._result.index)

        mtm_values = self._result['mtm']
        valid_mask = mtm_values.notna()

        # Momentum上升趋势+6分
        mtm_rising = (mtm_values > mtm_values.shift(1)) & valid_mask & valid_mask.shift(1)
        trend_score += mtm_rising * 6

        # Momentum下降趋势-6分
        mtm_falling = (mtm_values < mtm_values.shift(1)) & valid_mask & valid_mask.shift(1)
        trend_score -= mtm_falling * 6

        # Momentum连续上升（3个周期）+10分
        if len(mtm_values) >= 3:
            consecutive_rising = (
                (mtm_values > mtm_values.shift(1)) &
                (mtm_values.shift(1) > mtm_values.shift(2)) &
                (mtm_values.shift(2) > mtm_values.shift(3)) &
                valid_mask & valid_mask.shift(1) & valid_mask.shift(2) & valid_mask.shift(3)
            )
            trend_score += consecutive_rising * 10

        # Momentum连续下降（3个周期）-10分
        if len(mtm_values) >= 3:
            consecutive_falling = (
                (mtm_values < mtm_values.shift(1)) &
                (mtm_values.shift(1) < mtm_values.shift(2)) &
                (mtm_values.shift(2) < mtm_values.shift(3)) &
                valid_mask & valid_mask.shift(1) & valid_mask.shift(2) & valid_mask.shift(3)
            )
            trend_score -= consecutive_falling * 10

        return trend_score
    
    def _calculate_momentum_divergence_score(self, data: pd.DataFrame) -> pd.Series:
        """
        计算Momentum背离评分

        Args:
            data: 价格数据

        Returns:
            pd.Series: 背离评分
        """
        divergence_score = pd.Series(0.0, index=self._result.index)

        if 'close' not in data.columns:
            return divergence_score

        close_price = data['close']
        mtm_values = self._result['mtm']

        # 简化的背离检测，减少评分幅度
        if len(close_price) >= 20:
            # 检查最近20个周期的价格和Momentum趋势
            recent_periods = 20

            for i in range(recent_periods, len(close_price)):
                # 检查Momentum值是否有效
                if pd.isna(mtm_values.iloc[i]):
                    continue

                # 寻找最近的价格和Momentum峰值/谷值
                price_window = close_price.iloc[i-recent_periods:i+1]
                mtm_window = mtm_values.iloc[i-recent_periods:i+1].dropna()

                if len(mtm_window) == 0:
                    continue

                # 检查是否为价格新高/新低
                current_price = close_price.iloc[i]
                current_mtm = mtm_values.iloc[i]

                price_is_high = current_price >= price_window.max()
                price_is_low = current_price <= price_window.min()
                mtm_is_high = current_mtm >= mtm_window.max()
                mtm_is_low = current_mtm <= mtm_window.min()

                # 正背离：价格创新低但Momentum未创新低（减少评分）
                if price_is_low and not mtm_is_low:
                    divergence_score.iloc[i] += 15

                # 负背离：价格创新高但Momentum未创新高（减少评分）
                elif price_is_high and not mtm_is_high:
                    divergence_score.iloc[i] -= 15

        return divergence_score
    
    def _calculate_momentum_strength_score(self) -> pd.Series:
        """
        计算Momentum强度评分

        Returns:
            pd.Series: 强度评分
        """
        strength_score = pd.Series(0.0, index=self._result.index)

        mtm_values = self._result['mtm']
        valid_mask = mtm_values.notna()

        # 计算Momentum变化幅度
        mtm_change = mtm_values.diff()
        change_valid_mask = mtm_change.notna() & valid_mask

        # 计算Momentum的标准差作为强度参考
        mtm_std = mtm_values.rolling(20, min_periods=10).std()
        std_valid_mask = mtm_std.notna()

        # Momentum大幅上升+8分
        large_rise = (mtm_change > mtm_std) & change_valid_mask & std_valid_mask
        strength_score += large_rise * 8

        # Momentum大幅下降-8分
        large_fall = (mtm_change < -mtm_std) & change_valid_mask & std_valid_mask
        strength_score -= large_fall * 8

        # Momentum极值强度评分
        mtm_abs = np.abs(mtm_values)
        mtm_percentile_80 = mtm_abs.rolling(50, min_periods=25).quantile(0.8)
        percentile_valid_mask = mtm_percentile_80.notna()

        # Momentum处于极值区域±5分
        extreme_momentum = (mtm_abs > mtm_percentile_80) & valid_mask & percentile_valid_mask
        extreme_direction = np.sign(mtm_values)
        strength_score += extreme_momentum * extreme_direction * 5

        return strength_score
    
    def _detect_momentum_zero_cross_patterns(self) -> List[str]:
        """
        检测Momentum零轴穿越形态
        
        Returns:
            List[str]: 零轴穿越形态列表
        """
        patterns = []
        
        mtm_values = self._result['mtm']
        
        # 检查最近的零轴穿越
        recent_periods = min(5, len(mtm_values))
        recent_mtm = mtm_values.tail(recent_periods)
        
        if crossover(recent_mtm, 0).any():
            patterns.append("Momentum上穿零轴")
        
        if crossunder(recent_mtm, 0).any():
            patterns.append("Momentum下穿零轴")
        
        # 检查当前位置
        if len(mtm_values) > 0:
            current_mtm = mtm_values.iloc[-1]
            if not pd.isna(current_mtm):
                if current_mtm > 0:
                    patterns.append("Momentum零轴上方")
                elif current_mtm < 0:
                    patterns.append("Momentum零轴下方")
                else:
                    patterns.append("Momentum零轴位置")
        
        return patterns
    
    def _detect_momentum_signal_cross_patterns(self) -> List[str]:
        """
        检测Momentum与信号线交叉形态
        
        Returns:
            List[str]: 信号线交叉形态列表
        """
        patterns = []
        
        mtm_values = self._result['mtm']
        signal_values = self._result['signal']
        
        # 检查最近的信号线穿越
        recent_periods = min(5, len(mtm_values))
        recent_mtm = mtm_values.tail(recent_periods)
        recent_signal = signal_values.tail(recent_periods)
        
        if crossover(recent_mtm, recent_signal).any():
            patterns.append("Momentum上穿信号线")
        
        if crossunder(recent_mtm, recent_signal).any():
            patterns.append("Momentum下穿信号线")
        
        # 检查当前位置关系
        if len(mtm_values) > 0 and len(signal_values) > 0:
            current_mtm = mtm_values.iloc[-1]
            current_signal = signal_values.iloc[-1]
            
            if not pd.isna(current_mtm) and not pd.isna(current_signal):
                if current_mtm > current_signal:
                    patterns.append("Momentum信号线上方")
                elif current_mtm < current_signal:
                    patterns.append("Momentum信号线下方")
                else:
                    patterns.append("Momentum信号线重合")
        
        return patterns
    
    def _detect_momentum_trend_patterns(self) -> List[str]:
        """
        检测Momentum趋势形态
        
        Returns:
            List[str]: 趋势形态列表
        """
        patterns = []
        
        mtm_values = self._result['mtm']
        
        # 检查Momentum趋势
        if len(mtm_values) >= 3:
            recent_3 = mtm_values.tail(3)
            if len(recent_3) == 3 and not recent_3.isna().any():
                if (recent_3.iloc[2] > recent_3.iloc[1] > recent_3.iloc[0]):
                    patterns.append("Momentum连续上升")
                elif (recent_3.iloc[2] < recent_3.iloc[1] < recent_3.iloc[0]):
                    patterns.append("Momentum连续下降")
        
        # 检查当前趋势
        if len(mtm_values) >= 2:
            current_mtm = mtm_values.iloc[-1]
            prev_mtm = mtm_values.iloc[-2]
            
            if not pd.isna(current_mtm) and not pd.isna(prev_mtm):
                if current_mtm > prev_mtm:
                    patterns.append("Momentum上升")
                elif current_mtm < prev_mtm:
                    patterns.append("Momentum下降")
                else:
                    patterns.append("Momentum平稳")
        
        return patterns
    
    def _detect_momentum_divergence_patterns(self, data: pd.DataFrame) -> List[str]:
        """
        检测Momentum背离形态
        
        Args:
            data: 价格数据
            
        Returns:
            List[str]: 背离形态列表
        """
        patterns = []
        
        if 'close' not in data.columns:
            return patterns
        
        close_price = data['close']
        mtm_values = self._result['mtm']
        
        if len(close_price) >= 20:
            # 检查最近20个周期的趋势
            recent_price = close_price.tail(20)
            recent_mtm = mtm_values.tail(20)
            
            # 简化的背离检测
            price_trend = recent_price.iloc[-1] - recent_price.iloc[0]
            mtm_trend = recent_mtm.iloc[-1] - recent_mtm.iloc[0]
            
            # 背离检测
            if price_trend < -0.02 and mtm_trend > 0:  # 价格下跌但Momentum上升
                patterns.append("Momentum正背离")
            elif price_trend > 0.02 and mtm_trend < 0:  # 价格上涨但Momentum下降
                patterns.append("Momentum负背离")
            elif abs(price_trend) < 0.01 and abs(mtm_trend) < 0.1:
                patterns.append("Momentum价格同步")
        
        return patterns
    
    def _detect_momentum_strength_patterns(self) -> List[str]:
        """
        检测Momentum强度形态
        
        Returns:
            List[str]: 强度形态列表
        """
        patterns = []
        
        mtm_values = self._result['mtm']
        
        if len(mtm_values) >= 2:
            current_mtm = mtm_values.iloc[-1]
            prev_mtm = mtm_values.iloc[-2]
            
            if not pd.isna(current_mtm) and not pd.isna(prev_mtm):
                mtm_change = current_mtm - prev_mtm
                
                # 计算变化强度
                if len(mtm_values) >= 20:
                    mtm_std = mtm_values.tail(20).std()
                    
                    if mtm_change > mtm_std:
                        patterns.append("Momentum急速上升")
                    elif mtm_change > mtm_std * 0.5:
                        patterns.append("Momentum大幅上升")
                    elif mtm_change < -mtm_std:
                        patterns.append("Momentum急速下降")
                    elif mtm_change < -mtm_std * 0.5:
                        patterns.append("Momentum大幅下降")
                    elif abs(mtm_change) <= mtm_std * 0.1:
                        patterns.append("Momentum变化平缓")
        
        # 检查极值状态
        if len(mtm_values) >= 50:
            current_mtm = mtm_values.iloc[-1]
            mtm_abs = np.abs(mtm_values.tail(50))
            percentile_80 = mtm_abs.quantile(0.8)
            
            if abs(current_mtm) > percentile_80:
                if current_mtm > 0:
                    patterns.append("Momentum极值上方")
                else:
                    patterns.append("Momentum极值下方")
        
        return patterns
    
    def compute(self, df: pd.DataFrame, price_column: str = "close") -> pd.DataFrame:
        """
        计算动量指标并生成交易信号
        
        Args:
            df: 包含价格数据的DataFrame
            price_column: 用于计算动量的价格列名，默认为'close'
        
        Returns:
            包含动量指标和交易信号的DataFrame
        """
        try:
            result = self.calculate(df, price_column)
            result = self.generate_signals(df, result)
            self._result = result
            return result
        except Exception as e:
            self._error = str(e)
            logger.error(f"计算指标 {self.name} 时出错: {str(e)}")
            raise

    def calculate_confidence(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """
        计算Momentum指标的置信度

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

        return min(confidence, 1.0)

    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        识别所有已注册的Momentum相关形态

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含所有形态信号的DataFrame
        """
        # 确保已计算Momentum指标
        if not self.has_result():
            self.calculate(data, **kwargs)

        if self._result is None:
            return pd.DataFrame(index=data.index)

        patterns_df = pd.DataFrame(index=data.index)
        mtm = self._result['mtm']
        signal = self._result['signal']

        # 1. Momentum零轴穿越形态
        patterns_df['MTM_CROSS_ABOVE_ZERO'] = crossover(mtm, 0)
        patterns_df['MTM_CROSS_BELOW_ZERO'] = crossunder(mtm, 0)
        patterns_df['MTM_ABOVE_ZERO'] = mtm > 0
        patterns_df['MTM_BELOW_ZERO'] = mtm < 0

        # 2. Momentum与信号线交叉形态
        patterns_df['MTM_CROSS_ABOVE_SIGNAL'] = crossover(mtm, signal)
        patterns_df['MTM_CROSS_BELOW_SIGNAL'] = crossunder(mtm, signal)
        patterns_df['MTM_ABOVE_SIGNAL'] = mtm > signal
        patterns_df['MTM_BELOW_SIGNAL'] = mtm < signal

        # 3. Momentum趋势形态
        patterns_df['MTM_RISING'] = mtm > mtm.shift(1)
        patterns_df['MTM_FALLING'] = mtm < mtm.shift(1)

        # 连续上升/下降
        if len(mtm) >= 3:
            patterns_df['MTM_CONSECUTIVE_RISING'] = (
                (mtm > mtm.shift(1)) &
                (mtm.shift(1) > mtm.shift(2)) &
                (mtm.shift(2) > mtm.shift(3))
            )
            patterns_df['MTM_CONSECUTIVE_FALLING'] = (
                (mtm < mtm.shift(1)) &
                (mtm.shift(1) < mtm.shift(2)) &
                (mtm.shift(2) < mtm.shift(3))
            )

        # 4. Momentum背离形态（简化版）
        if 'close' in data.columns and len(data) >= 20:
            close_price = data['close']

            # 计算20周期的价格和Momentum趋势
            price_trend = close_price.rolling(20).apply(lambda x: x.iloc[-1] - x.iloc[0])
            mtm_trend = mtm.rolling(20).apply(lambda x: x.iloc[-1] - x.iloc[0])

            # 背离检测
            patterns_df['MTM_BULLISH_DIVERGENCE'] = (price_trend < -0.02) & (mtm_trend > 0)
            patterns_df['MTM_BEARISH_DIVERGENCE'] = (price_trend > 0.02) & (mtm_trend < 0)

        # 5. Momentum强度形态
        mtm_change = mtm.diff()
        if len(mtm) >= 20:
            mtm_std = mtm.rolling(20).std()
            patterns_df['MTM_LARGE_RISE'] = mtm_change > mtm_std
            patterns_df['MTM_LARGE_FALL'] = mtm_change < -mtm_std

            # 极值形态
            mtm_abs = np.abs(mtm)
            mtm_percentile_80 = mtm_abs.rolling(50).quantile(0.8)
            patterns_df['MTM_EXTREME_HIGH'] = (mtm > 0) & (mtm_abs > mtm_percentile_80)
            patterns_df['MTM_EXTREME_LOW'] = (mtm < 0) & (mtm_abs > mtm_percentile_80)

        return patterns_df

    def register_patterns(self):
        """
        注册Momentum指标的技术形态
        """
        # 注册Momentum零轴穿越形态
        self.register_pattern_to_registry(
            pattern_id="MTM_CROSS_ABOVE_ZERO",
            display_name="Momentum上穿零轴",
            description="Momentum从负值区域穿越零轴，表示动量转为正向",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=25.0
        )
    def get_pattern_info(self, pattern_id: str) -> dict:
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
        
        # Momentum指标特定的形态信息映射
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


        self.register_pattern_to_registry(
            pattern_id="MTM_CROSS_BELOW_ZERO",
            display_name="Momentum下穿零轴",
            description="Momentum从正值区域穿越零轴，表示动量转为负向",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-25.0
        )

        # 注册Momentum与信号线交叉形态
        self.register_pattern_to_registry(
            pattern_id="MTM_CROSS_ABOVE_SIGNAL",
            display_name="Momentum上穿信号线",
            description="Momentum上穿其移动平均线，买入信号",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=20.0
        )

        self.register_pattern_to_registry(
            pattern_id="MTM_CROSS_BELOW_SIGNAL",
            display_name="Momentum下穿信号线",
            description="Momentum下穿其移动平均线，卖出信号",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-20.0
        )

        # 注册Momentum背离形态
        self.register_pattern_to_registry(
            pattern_id="MTM_BULLISH_DIVERGENCE",
            display_name="Momentum底背离",
            description="价格创新低但Momentum未创新低，表明下跌动能减弱",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=25.0
        )

        self.register_pattern_to_registry(
            pattern_id="MTM_BEARISH_DIVERGENCE",
            display_name="Momentum顶背离",
            description="价格创新高但Momentum未创新高，表明上涨动能减弱",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-25.0
        )

        # 注册Momentum趋势形态
        self.register_pattern_to_registry(
            pattern_id="MTM_CONSECUTIVE_RISING",
            display_name="Momentum连续上升",
            description="Momentum连续3个周期上升，表明上涨动能持续增强",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=15.0
        )

        self.register_pattern_to_registry(
            pattern_id="MTM_CONSECUTIVE_FALLING",
            display_name="Momentum连续下降",
            description="Momentum连续3个周期下降，表明下跌动能持续增强",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-15.0
        )

        # 注册Momentum强度形态
        self.register_pattern_to_registry(
            pattern_id="MTM_EXTREME_HIGH",
            display_name="Momentum极值高位",
            description="Momentum处于极值高位，表明强烈的上涨动能",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=12.0
        )

        self.register_pattern_to_registry(
            pattern_id="MTM_EXTREME_LOW",
            display_name="Momentum极值低位",
            description="Momentum处于极值低位，表明强烈的下跌动能",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-12.0
        )