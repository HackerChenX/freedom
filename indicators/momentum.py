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
    
    def calculate(self, df: pd.DataFrame, price_column: str = "close") -> pd.DataFrame:
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
        
        # Momentum上穿零轴+25分
        mtm_cross_up_zero = crossover(mtm_values, 0)
        zero_cross_score += mtm_cross_up_zero * 25
        
        # Momentum下穿零轴-25分
        mtm_cross_down_zero = crossunder(mtm_values, 0)
        zero_cross_score -= mtm_cross_down_zero * 25
        
        # Momentum在零轴上方+8分
        mtm_above_zero = mtm_values > 0
        zero_cross_score += mtm_above_zero * 8
        
        # Momentum在零轴下方-8分
        mtm_below_zero = mtm_values < 0
        zero_cross_score -= mtm_below_zero * 8
        
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
        
        # Momentum上穿信号线+20分
        mtm_cross_up_signal = crossover(mtm_values, signal_values)
        signal_cross_score += mtm_cross_up_signal * 20
        
        # Momentum下穿信号线-20分
        mtm_cross_down_signal = crossunder(mtm_values, signal_values)
        signal_cross_score -= mtm_cross_down_signal * 20
        
        # Momentum在信号线上方+5分
        mtm_above_signal = mtm_values > signal_values
        signal_cross_score += mtm_above_signal * 5
        
        # Momentum在信号线下方-5分
        mtm_below_signal = mtm_values < signal_values
        signal_cross_score -= mtm_below_signal * 5
        
        return signal_cross_score
    
    def _calculate_momentum_trend_score(self) -> pd.Series:
        """
        计算Momentum趋势评分
        
        Returns:
            pd.Series: 趋势评分
        """
        trend_score = pd.Series(0.0, index=self._result.index)
        
        mtm_values = self._result['mtm']
        
        # Momentum上升趋势+10分
        mtm_rising = mtm_values > mtm_values.shift(1)
        trend_score += mtm_rising * 10
        
        # Momentum下降趋势-10分
        mtm_falling = mtm_values < mtm_values.shift(1)
        trend_score -= mtm_falling * 10
        
        # Momentum连续上升（3个周期）+15分
        if len(mtm_values) >= 3:
            consecutive_rising = (
                (mtm_values > mtm_values.shift(1)) &
                (mtm_values.shift(1) > mtm_values.shift(2)) &
                (mtm_values.shift(2) > mtm_values.shift(3))
            )
            trend_score += consecutive_rising * 15
        
        # Momentum连续下降（3个周期）-15分
        if len(mtm_values) >= 3:
            consecutive_falling = (
                (mtm_values < mtm_values.shift(1)) &
                (mtm_values.shift(1) < mtm_values.shift(2)) &
                (mtm_values.shift(2) < mtm_values.shift(3))
            )
            trend_score -= consecutive_falling * 15
        
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
        
        # 简化的背离检测
        if len(close_price) >= 20:
            # 检查最近20个周期的价格和Momentum趋势
            recent_periods = 20
            
            for i in range(recent_periods, len(close_price)):
                # 寻找最近的价格和Momentum峰值/谷值
                price_window = close_price.iloc[i-recent_periods:i+1]
                mtm_window = mtm_values.iloc[i-recent_periods:i+1]
                
                # 检查是否为价格新高/新低
                current_price = close_price.iloc[i]
                current_mtm = mtm_values.iloc[i]
                
                price_is_high = current_price >= price_window.max()
                price_is_low = current_price <= price_window.min()
                mtm_is_high = current_mtm >= mtm_window.max()
                mtm_is_low = current_mtm <= mtm_window.min()
                
                # 正背离：价格创新低但Momentum未创新低
                if price_is_low and not mtm_is_low:
                    divergence_score.iloc[i] += 30
                
                # 负背离：价格创新高但Momentum未创新高
                elif price_is_high and not mtm_is_high:
                    divergence_score.iloc[i] -= 30
        
        return divergence_score
    
    def _calculate_momentum_strength_score(self) -> pd.Series:
        """
        计算Momentum强度评分
        
        Returns:
            pd.Series: 强度评分
        """
        strength_score = pd.Series(0.0, index=self._result.index)
        
        mtm_values = self._result['mtm']
        
        # 计算Momentum变化幅度
        mtm_change = mtm_values.diff()
        
        # 计算Momentum的标准差作为强度参考
        mtm_std = mtm_values.rolling(20).std()
        
        # Momentum大幅上升+12分
        large_rise = mtm_change > mtm_std
        strength_score += large_rise * 12
        
        # Momentum大幅下降-12分
        large_fall = mtm_change < -mtm_std
        strength_score -= large_fall * 12
        
        # Momentum极值强度评分
        mtm_abs = np.abs(mtm_values)
        mtm_percentile_80 = mtm_abs.rolling(50).quantile(0.8)
        
        # Momentum处于极值区域±8分
        extreme_momentum = mtm_abs > mtm_percentile_80
        extreme_direction = np.sign(mtm_values)
        strength_score += extreme_momentum * extreme_direction * 8
        
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
    
    def plot(self, df: pd.DataFrame, result: pd.DataFrame, ax=None):
        """
        绘制动量指标图表
        
        Args:
            df: 原始数据DataFrame
            result: 计算指标后的DataFrame
            ax: matplotlib轴对象，如果为None则创建新的
        
        Returns:
            matplotlib轴对象
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 7))
        
        # 绘制MTM和信号线
        ax.plot(result['mtm'], label=f'MTM({self.period})')
        ax.plot(result['signal'], label=f'Signal({self.signal_period})')
        
        # 绘制零线
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # 标记买卖信号
        buy_signals = result[result['buy_signal'] == 1].index
        sell_signals = result[result['sell_signal'] == 1].index
        
        ax.scatter(buy_signals, result.loc[buy_signals, 'mtm'], color='green', marker='^', s=100, label='买入信号')
        ax.scatter(sell_signals, result.loc[sell_signals, 'mtm'], color='red', marker='v', s=100, label='卖出信号')
        
        ax.set_title(f'动量指标(MTM) - 周期:{self.period}')
        ax.set_ylabel('动量值')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
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