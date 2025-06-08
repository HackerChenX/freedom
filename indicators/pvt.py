#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
价格成交量趋势指标(PVT)

通过价格变化与成交量相结合，反映价格趋势的强度和持续性
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple

from indicators.base_indicator import BaseIndicator
from indicators.common import crossover, crossunder
from utils.logger import get_logger

logger = get_logger(__name__)


class PVT(BaseIndicator):
    """
    价格成交量趋势指标(PVT) (PVT)
    
    分类：量能类指标
    描述：通过价格变化与成交量相结合，反映价格趋势的强度和持续性
    """
    
    def __init__(self, ma_period: int = 12):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """
        初始化价格成交量趋势指标(PVT)指标
        
        Args:
            ma_period: 移动平均周期，默认为12
        """
        super().__init__(name="PVT", description="价格成交量趋势指标，通过价格变化与成交量相结合，反映价格趋势的强度和持续性")
        self.ma_period = ma_period
        
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
        计算PVT指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
                
        Returns:
            包含PVT指标的DataFrame
        """
        return self.calculate(df)
        
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算价格成交量趋势指标(PVT)指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
                必须包含以下列：
                - close: 收盘价
                - volume: 成交量
                
        Returns:
            添加了PVT指标列的DataFrame
        """
        if df.empty:
            return df
            
        # 确保数据包含必要的列
        required_columns = ['close', 'volume']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 计算价格变化率
        price_change = df_copy['close'].pct_change()
        
        # 计算PVT
        # PVT = 昨日PVT + 今日成交量 * 价格变化率
        df_copy['pvt'] = df_copy['volume'] * price_change
        df_copy['pvt'] = df_copy['pvt'].cumsum()
        
        # 计算PVT的移动平均作为信号线
        df_copy['pvt_signal'] = df_copy['pvt'].rolling(window=self.ma_period).mean()
        
        # 存储结果
        self._result = df_copy[['pvt', 'pvt_signal']]
        
        return df_copy
        
    def get_signals(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成价格成交量趋势指标(PVT)指标交易信号
        
        Args:
            df: 包含价格数据和PVT指标的DataFrame
            **kwargs: 额外参数
                
        Returns:
            添加了信号列的DataFrame:
            - pvt_buy_signal: 1=买入信号, 0=无信号
            - pvt_sell_signal: 1=卖出信号, 0=无信号
        """
        if df.empty:
            return df
            
        # 检查必要的指标列是否存在
        required_columns = ['pvt', 'pvt_signal']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 初始化信号列
        df_copy['pvt_buy_signal'] = 0
        df_copy['pvt_sell_signal'] = 0
        
        # PVT上穿信号线为买入信号
        for i in range(1, len(df_copy)):
            if df_copy['pvt'].iloc[i-1] < df_copy['pvt_signal'].iloc[i-1] and \
               df_copy['pvt'].iloc[i] > df_copy['pvt_signal'].iloc[i]:
                df_copy.iloc[i, df_copy.columns.get_loc('pvt_buy_signal')] = 1
            
            # PVT下穿信号线为卖出信号
            elif df_copy['pvt'].iloc[i-1] > df_copy['pvt_signal'].iloc[i-1] and \
                 df_copy['pvt'].iloc[i] < df_copy['pvt_signal'].iloc[i]:
                df_copy.iloc[i, df_copy.columns.get_loc('pvt_sell_signal')] = 1
        
        return df_copy
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算PVT原始评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 原始评分序列（0-100分）
        """
        # 确保已计算PVT
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)
        
        score = pd.Series(50.0, index=data.index)  # 基础分50分
        
        # 1. PVT与信号线交叉评分
        signal_cross_score = self._calculate_pvt_signal_cross_score()
        score += signal_cross_score
        
        # 2. PVT趋势评分
        trend_score = self._calculate_pvt_trend_score()
        score += trend_score
        
        # 3. PVT背离评分
        divergence_score = self._calculate_pvt_divergence_score(data)
        score += divergence_score
        
        # 4. PVT强度评分
        strength_score = self._calculate_pvt_strength_score()
        score += strength_score
        
        # 5. PVT与价格关系评分
        price_relation_score = self._calculate_pvt_price_relation_score(data)
        score += price_relation_score
        
        return np.clip(score, 0, 100)
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别PVT技术形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别出的形态列表
        """
        patterns = []
        
        # 确保已计算PVT
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return patterns
        
        # 1. 检测PVT与信号线交叉形态
        signal_cross_patterns = self._detect_pvt_signal_cross_patterns()
        patterns.extend(signal_cross_patterns)
        
        # 2. 检测PVT趋势形态
        trend_patterns = self._detect_pvt_trend_patterns()
        patterns.extend(trend_patterns)
        
        # 3. 检测PVT背离形态
        divergence_patterns = self._detect_pvt_divergence_patterns(data)
        patterns.extend(divergence_patterns)
        
        # 4. 检测PVT强度形态
        strength_patterns = self._detect_pvt_strength_patterns()
        patterns.extend(strength_patterns)
        
        # 5. 检测PVT与价格关系形态
        price_relation_patterns = self._detect_pvt_price_relation_patterns(data)
        patterns.extend(price_relation_patterns)
        
        return patterns
    
    def _calculate_pvt_signal_cross_score(self) -> pd.Series:
        """
        计算PVT与信号线交叉评分
        
        Returns:
            pd.Series: 信号线交叉评分
        """
        signal_cross_score = pd.Series(0.0, index=self._result.index)
        
        pvt_values = self._result['pvt']
        signal_values = self._result['pvt_signal']
        
        # PVT上穿信号线+25分
        pvt_cross_up_signal = crossover(pvt_values, signal_values)
        signal_cross_score += pvt_cross_up_signal * 25
        
        # PVT下穿信号线-25分
        pvt_cross_down_signal = crossunder(pvt_values, signal_values)
        signal_cross_score -= pvt_cross_down_signal * 25
        
        # PVT在信号线上方+8分
        pvt_above_signal = pvt_values > signal_values
        signal_cross_score += pvt_above_signal * 8
        
        # PVT在信号线下方-8分
        pvt_below_signal = pvt_values < signal_values
        signal_cross_score -= pvt_below_signal * 8
        
        return signal_cross_score
    
    def _calculate_pvt_trend_score(self) -> pd.Series:
        """
        计算PVT趋势评分
        
        Returns:
            pd.Series: 趋势评分
        """
        trend_score = pd.Series(0.0, index=self._result.index)
        
        pvt_values = self._result['pvt']
        
        # PVT上升趋势+12分
        pvt_rising = pvt_values > pvt_values.shift(1)
        trend_score += pvt_rising * 12
        
        # PVT下降趋势-12分
        pvt_falling = pvt_values < pvt_values.shift(1)
        trend_score -= pvt_falling * 12
        
        # PVT连续上升（3个周期）+18分
        if len(pvt_values) >= 3:
            consecutive_rising = (
                (pvt_values > pvt_values.shift(1)) &
                (pvt_values.shift(1) > pvt_values.shift(2)) &
                (pvt_values.shift(2) > pvt_values.shift(3))
            )
            trend_score += consecutive_rising * 18
        
        # PVT连续下降（3个周期）-18分
        if len(pvt_values) >= 3:
            consecutive_falling = (
                (pvt_values < pvt_values.shift(1)) &
                (pvt_values.shift(1) < pvt_values.shift(2)) &
                (pvt_values.shift(2) < pvt_values.shift(3))
            )
            trend_score -= consecutive_falling * 18
        
        return trend_score
    
    def _calculate_pvt_divergence_score(self, data: pd.DataFrame) -> pd.Series:
        """
        计算PVT背离评分
        
        Args:
            data: 价格数据
            
        Returns:
            pd.Series: 背离评分
        """
        divergence_score = pd.Series(0.0, index=self._result.index)
        
        if 'close' not in data.columns:
            return divergence_score
        
        close_price = data['close']
        pvt_values = self._result['pvt']
        
        # 简化的背离检测
        if len(close_price) >= 20:
            # 检查最近20个周期的价格和PVT趋势
            recent_periods = 20
            
            for i in range(recent_periods, len(close_price)):
                # 寻找最近的价格和PVT峰值/谷值
                price_window = close_price.iloc[i-recent_periods:i+1]
                pvt_window = pvt_values.iloc[i-recent_periods:i+1]
                
                # 检查是否为价格新高/新低
                current_price = close_price.iloc[i]
                current_pvt = pvt_values.iloc[i]
                
                price_is_high = current_price >= price_window.max()
                price_is_low = current_price <= price_window.min()
                pvt_is_high = current_pvt >= pvt_window.max()
                pvt_is_low = current_pvt <= pvt_window.min()
                
                # 正背离：价格创新低但PVT未创新低
                if price_is_low and not pvt_is_low:
                    divergence_score.iloc[i] += 30
                
                # 负背离：价格创新高但PVT未创新高
                elif price_is_high and not pvt_is_high:
                    divergence_score.iloc[i] -= 30
        
        return divergence_score
    
    def _calculate_pvt_strength_score(self) -> pd.Series:
        """
        计算PVT强度评分
        
        Returns:
            pd.Series: 强度评分
        """
        strength_score = pd.Series(0.0, index=self._result.index)
        
        pvt_values = self._result['pvt']
        
        # 计算PVT变化幅度
        pvt_change = pvt_values.diff()
        
        # 计算PVT的标准差作为强度参考
        pvt_std = pvt_values.rolling(20).std()
        
        # PVT大幅上升+15分
        large_rise = pvt_change > pvt_std
        strength_score += large_rise * 15
        
        # PVT大幅下降-15分
        large_fall = pvt_change < -pvt_std
        strength_score -= large_fall * 15
        
        # PVT快速变化（绝对值>2倍标准差）额外±10分
        rapid_change = np.abs(pvt_change) > 2 * pvt_std
        rapid_change_direction = np.sign(pvt_change)
        strength_score += rapid_change * rapid_change_direction * 10
        
        return strength_score
    
    def _calculate_pvt_price_relation_score(self, data: pd.DataFrame) -> pd.Series:
        """
        计算PVT与价格关系评分
        
        Args:
            data: 价格数据
            
        Returns:
            pd.Series: 价格关系评分
        """
        price_relation_score = pd.Series(0.0, index=self._result.index)
        
        if 'close' not in data.columns:
            return price_relation_score
        
        close_price = data['close']
        pvt_values = self._result['pvt']
        
        # 计算价格变化率
        price_change = close_price.pct_change()
        pvt_change = pvt_values.pct_change()
        
        # 价格和PVT同向上涨+15分
        both_rising = (price_change > 0) & (pvt_change > 0)
        price_relation_score += both_rising * 15
        
        # 价格和PVT同向下跌-15分
        both_falling = (price_change < 0) & (pvt_change < 0)
        price_relation_score -= both_falling * 15
        
        # 价格上涨但PVT下跌（背离）-20分
        price_up_pvt_down = (price_change > 0) & (pvt_change < 0)
        price_relation_score -= price_up_pvt_down * 20
        
        # 价格下跌但PVT上涨（背离）+20分
        price_down_pvt_up = (price_change < 0) & (pvt_change > 0)
        price_relation_score += price_down_pvt_up * 20
        
        return price_relation_score
    
    def _detect_pvt_signal_cross_patterns(self) -> List[str]:
        """
        检测PVT与信号线交叉形态
        
        Returns:
            List[str]: 信号线交叉形态列表
        """
        patterns = []
        
        pvt_values = self._result['pvt']
        signal_values = self._result['pvt_signal']
        
        # 检查最近的信号线穿越
        recent_periods = min(5, len(pvt_values))
        recent_pvt = pvt_values.tail(recent_periods)
        recent_signal = signal_values.tail(recent_periods)
        
        if crossover(recent_pvt, recent_signal).any():
            patterns.append("PVT上穿信号线")
        
        if crossunder(recent_pvt, recent_signal).any():
            patterns.append("PVT下穿信号线")
        
        # 检查当前位置关系
        if len(pvt_values) > 0 and len(signal_values) > 0:
            current_pvt = pvt_values.iloc[-1]
            current_signal = signal_values.iloc[-1]
            
            if not pd.isna(current_pvt) and not pd.isna(current_signal):
                if current_pvt > current_signal:
                    patterns.append("PVT信号线上方")
                elif current_pvt < current_signal:
                    patterns.append("PVT信号线下方")
                else:
                    patterns.append("PVT信号线重合")
        
        return patterns
    
    def _detect_pvt_trend_patterns(self) -> List[str]:
        """
        检测PVT趋势形态
        
        Returns:
            List[str]: 趋势形态列表
        """
        patterns = []
        
        pvt_values = self._result['pvt']
        
        # 检查PVT趋势
        if len(pvt_values) >= 3:
            recent_3 = pvt_values.tail(3)
            if len(recent_3) == 3 and not recent_3.isna().any():
                if (recent_3.iloc[2] > recent_3.iloc[1] > recent_3.iloc[0]):
                    patterns.append("PVT连续上升")
                elif (recent_3.iloc[2] < recent_3.iloc[1] < recent_3.iloc[0]):
                    patterns.append("PVT连续下降")
        
        # 检查当前趋势
        if len(pvt_values) >= 2:
            current_pvt = pvt_values.iloc[-1]
            prev_pvt = pvt_values.iloc[-2]
            
            if not pd.isna(current_pvt) and not pd.isna(prev_pvt):
                if current_pvt > prev_pvt:
                    patterns.append("PVT上升")
                elif current_pvt < prev_pvt:
                    patterns.append("PVT下降")
                else:
                    patterns.append("PVT平稳")
        
        return patterns
    
    def _detect_pvt_divergence_patterns(self, data: pd.DataFrame) -> List[str]:
        """
        检测PVT背离形态
        
        Args:
            data: 价格数据
            
        Returns:
            List[str]: 背离形态列表
        """
        patterns = []
        
        if 'close' not in data.columns:
            return patterns
        
        close_price = data['close']
        pvt_values = self._result['pvt']
        
        if len(close_price) >= 20:
            # 检查最近20个周期的趋势
            recent_price = close_price.tail(20)
            recent_pvt = pvt_values.tail(20)
            
            # 简化的背离检测
            price_trend = recent_price.iloc[-1] - recent_price.iloc[0]
            pvt_trend = recent_pvt.iloc[-1] - recent_pvt.iloc[0]
            
            # 背离检测
            if price_trend < -0.02 and pvt_trend > 0:  # 价格下跌但PVT上升
                patterns.append("PVT正背离")
            elif price_trend > 0.02 and pvt_trend < 0:  # 价格上涨但PVT下降
                patterns.append("PVT负背离")
            elif abs(price_trend) < 0.01 and abs(pvt_trend) < 0.1:
                patterns.append("PVT价格同步")
        
        return patterns
    
    def _detect_pvt_strength_patterns(self) -> List[str]:
        """
        检测PVT强度形态
        
        Returns:
            List[str]: 强度形态列表
        """
        patterns = []
        
        pvt_values = self._result['pvt']
        
        if len(pvt_values) >= 2:
            current_pvt = pvt_values.iloc[-1]
            prev_pvt = pvt_values.iloc[-2]
            
            if not pd.isna(current_pvt) and not pd.isna(prev_pvt):
                pvt_change = current_pvt - prev_pvt
                
                # 计算变化强度
                if len(pvt_values) >= 20:
                    pvt_std = pvt_values.tail(20).std()
                    
                    if pvt_change > pvt_std:
                        patterns.append("PVT急速上升")
                    elif pvt_change > pvt_std * 0.5:
                        patterns.append("PVT大幅上升")
                    elif pvt_change < -pvt_std:
                        patterns.append("PVT急速下降")
                    elif pvt_change < -pvt_std * 0.5:
                        patterns.append("PVT大幅下降")
                    elif abs(pvt_change) <= pvt_std * 0.1:
                        patterns.append("PVT变化平缓")
        
        return patterns
    
    def _detect_pvt_price_relation_patterns(self, data: pd.DataFrame) -> List[str]:
        """
        检测PVT与价格关系形态
        
        Args:
            data: 价格数据
            
        Returns:
            List[str]: 价格关系形态列表
        """
        patterns = []
        
        if 'close' not in data.columns:
            return patterns
        
        close_price = data['close']
        pvt_values = self._result['pvt']
        
        if len(close_price) >= 5:
            # 检查最近5个周期的价格和PVT关系
            recent_price = close_price.tail(5)
            recent_pvt = pvt_values.tail(5)
            
            # 计算价格和PVT的趋势
            price_trend = recent_price.iloc[-1] - recent_price.iloc[0]
            pvt_trend = recent_pvt.iloc[-1] - recent_pvt.iloc[0]
            
            # 量价配合
            if price_trend > 0 and pvt_trend > 0:
                patterns.append("价量配合上涨")
            elif price_trend < 0 and pvt_trend < 0:
                patterns.append("价量配合下跌")
            # 量价背离
            elif price_trend > 0 and pvt_trend < 0:
                patterns.append("价量背离上涨")
            elif price_trend < 0 and pvt_trend > 0:
                patterns.append("价量背离下跌")
            else:
                patterns.append("价量关系中性")
        
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
        绘制价格成交量趋势指标(PVT)指标图表
        
        Args:
            df: 包含PVT指标的DataFrame
            ax: matplotlib轴对象，如果为None则创建新的
            **kwargs: 额外绘图参数
            
        Returns:
            matplotlib轴对象
        """
        import matplotlib.pyplot as plt
        
        # 检查必要的指标列是否存在
        required_columns = ['pvt', 'pvt_signal']
        self._validate_dataframe(df, required_columns)
        
        # 创建新的轴对象（如果未提供）
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
            
        # 绘制PVT指标线
        ax.plot(df.index, df['pvt'], label='PVT')
        ax.plot(df.index, df['pvt_signal'], label='信号线', linestyle='--')
        
        # 添加零轴线
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        ax.set_ylabel('价格成交量趋势指标(PVT)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        return ax

