#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
蔡金指标(Chaikin)

蔡金指标是基于累积/分布线的动量指标，用于衡量资金流入流出的动量
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple

from indicators.base_indicator import BaseIndicator
from utils.signal_utils import crossover, crossunder
from utils.logger import get_logger

logger = get_logger(__name__)


class Chaikin(BaseIndicator):
    """
    蔡金指标(Chaikin) (Chaikin)
    
    分类：量能类指标
    描述：蔡金指标是基于累积/分布线的动量指标，用于衡量资金流入流出的动量
    """
    
    def __init__(self, fast_period: int = 3, slow_period: int = 10):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """
        初始化蔡金指标(Chaikin)
        
        Args:
            fast_period: 快速EMA周期，默认为3
            slow_period: 慢速EMA周期，默认为10
        """
        super().__init__(name="Chaikin", description="蔡金指标，基于累积/分布线的动量指标")
        self.fast_period = fast_period
        self.slow_period = slow_period
        
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
        计算Chaikin指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
                
        Returns:
            包含Chaikin指标的DataFrame
        """
        return self.calculate(df)
        
    def _calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算蔡金指标(Chaikin)
        
        Args:
            df: 包含OHLCV数据的DataFrame
                必须包含以下列：
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - volume: 成交量
                
        Returns:
            添加了Chaikin指标列的DataFrame
        """
        if df.empty:
            return df
            
        # 确保数据包含必要的列
        required_columns = ['high', 'low', 'close', 'volume']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 计算累积/分布线 (A/D Line)
        # CLV = ((Close - Low) - (High - Close)) / (High - Low)
        # A/D = Previous A/D + CLV * Volume
        
        # 计算CLV (Close Location Value)
        high_low_diff = df_copy['high'] - df_copy['low']
        # 避免除以零
        high_low_diff = high_low_diff.replace(0, 0.000001)
        
        clv = ((df_copy['close'] - df_copy['low']) - (df_copy['high'] - df_copy['close'])) / high_low_diff
        
        # 计算A/D线
        ad_line = (clv * df_copy['volume']).cumsum()
        df_copy['ad_line'] = ad_line
        
        # 计算Chaikin震荡器
        # Chaikin = EMA(A/D, fast_period) - EMA(A/D, slow_period)
        ad_ema_fast = ad_line.ewm(span=self.fast_period).mean()
        ad_ema_slow = ad_line.ewm(span=self.slow_period).mean()
        
        df_copy['chaikin_oscillator'] = ad_ema_fast - ad_ema_slow
        
        # 计算Chaikin的移动平均作为信号线
        df_copy['chaikin_signal'] = df_copy['chaikin_oscillator'].ewm(span=5).mean()
        
        # 存储结果
        self._result = df_copy[['ad_line', 'chaikin_oscillator', 'chaikin_signal']]
        
        return df_copy
        
    def get_signals(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成蔡金指标(Chaikin)交易信号
        
        Args:
            df: 包含价格数据和Chaikin指标的DataFrame
            **kwargs: 额外参数
                
        Returns:
            添加了信号列的DataFrame:
            - chaikin_buy_signal: 1=买入信号, 0=无信号
            - chaikin_sell_signal: 1=卖出信号, 0=无信号
        """
        if df.empty:
            return df
            
        # 检查必要的指标列是否存在
        required_columns = ['chaikin_oscillator', 'chaikin_signal']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 初始化信号列
        df_copy['chaikin_buy_signal'] = 0
        df_copy['chaikin_sell_signal'] = 0
        
        # 生成交易信号
        for i in range(1, len(df_copy)):
            # 1. Chaikin震荡器上穿零轴
            if (df_copy['chaikin_oscillator'].iloc[i-1] <= 0 and 
                df_copy['chaikin_oscillator'].iloc[i] > 0):
                df_copy.iloc[i, df_copy.columns.get_loc('chaikin_buy_signal')] = 1
            
            # 2. Chaikin震荡器下穿零轴
            elif (df_copy['chaikin_oscillator'].iloc[i-1] >= 0 and 
                  df_copy['chaikin_oscillator'].iloc[i] < 0):
                df_copy.iloc[i, df_copy.columns.get_loc('chaikin_sell_signal')] = 1
            
            # 3. Chaikin震荡器上穿信号线
            elif (df_copy['chaikin_oscillator'].iloc[i-1] <= df_copy['chaikin_signal'].iloc[i-1] and 
                  df_copy['chaikin_oscillator'].iloc[i] > df_copy['chaikin_signal'].iloc[i]):
                df_copy.iloc[i, df_copy.columns.get_loc('chaikin_buy_signal')] = 1
            
            # 4. Chaikin震荡器下穿信号线
            elif (df_copy['chaikin_oscillator'].iloc[i-1] >= df_copy['chaikin_signal'].iloc[i-1] and 
                  df_copy['chaikin_oscillator'].iloc[i] < df_copy['chaikin_signal'].iloc[i]):
                df_copy.iloc[i, df_copy.columns.get_loc('chaikin_sell_signal')] = 1
        
        return df_copy
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算Chaikin原始评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 原始评分序列（0-100分）
        """
        # 确保已计算Chaikin
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)
        
        score = pd.Series(50.0, index=data.index)  # 基础分50分
        
        # 1. Chaikin零轴穿越评分
        zero_cross_score = self._calculate_chaikin_zero_cross_score()
        score += zero_cross_score
        
        # 2. Chaikin背离评分
        divergence_score = self._calculate_chaikin_divergence_score(data)
        score += divergence_score
        
        # 3. Chaikin趋势评分
        trend_score = self._calculate_chaikin_trend_score()
        score += trend_score
        
        # 4. Chaikin强度评分
        strength_score = self._calculate_chaikin_strength_score()
        score += strength_score
        
        # 5. A/D线评分
        ad_line_score = self._calculate_ad_line_score(data)
        score += ad_line_score
        
        return np.clip(score, 0, 100)
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别Chaikin技术形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别出的形态列表
        """
        patterns = []
        
        # 确保已计算Chaikin
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return patterns
        
        # 1. 检测Chaikin零轴穿越形态
        zero_cross_patterns = self._detect_chaikin_zero_cross_patterns()
        patterns.extend(zero_cross_patterns)
        
        # 2. 检测Chaikin背离形态
        divergence_patterns = self._detect_chaikin_divergence_patterns(data)
        patterns.extend(divergence_patterns)
        
        # 3. 检测Chaikin趋势形态
        trend_patterns = self._detect_chaikin_trend_patterns()
        patterns.extend(trend_patterns)
        
        # 4. 检测Chaikin强度形态
        strength_patterns = self._detect_chaikin_strength_patterns()
        patterns.extend(strength_patterns)
        
        # 5. 检测A/D线形态
        ad_line_patterns = self._detect_ad_line_patterns(data)
        patterns.extend(ad_line_patterns)
        
        return patterns
    
    def _calculate_chaikin_zero_cross_score(self) -> pd.Series:
        """
        计算Chaikin零轴穿越评分
        
        Returns:
            pd.Series: 零轴穿越评分
        """
        zero_cross_score = pd.Series(0.0, index=self._result.index)
        
        chaikin_oscillator = self._result['chaikin_oscillator']
        
        # Chaikin上穿零轴+25分
        chaikin_cross_up_zero = crossover(chaikin_oscillator, 0)
        zero_cross_score += chaikin_cross_up_zero * 25
        
        # Chaikin下穿零轴-25分
        chaikin_cross_down_zero = crossunder(chaikin_oscillator, 0)
        zero_cross_score -= chaikin_cross_down_zero * 25
        
        # Chaikin在零轴上方+10分
        chaikin_above_zero = chaikin_oscillator > 0
        zero_cross_score += chaikin_above_zero * 10
        
        # Chaikin在零轴下方-10分
        chaikin_below_zero = chaikin_oscillator < 0
        zero_cross_score -= chaikin_below_zero * 10
        
        return zero_cross_score
    
    def _calculate_chaikin_divergence_score(self, data: pd.DataFrame) -> pd.Series:
        """
        计算Chaikin背离评分
        
        Args:
            data: 价格数据
            
        Returns:
            pd.Series: 背离评分
        """
        divergence_score = pd.Series(0.0, index=self._result.index)
        
        if 'close' not in data.columns:
            return divergence_score
        
        close_price = data['close']
        chaikin_oscillator = self._result['chaikin_oscillator']
        
        # 简化的背离检测
        if len(close_price) >= 20:
            # 检查最近20个周期的价格和Chaikin趋势
            recent_periods = 20
            
            for i in range(recent_periods, len(close_price)):
                # 寻找最近的价格和Chaikin峰值/谷值
                price_window = close_price.iloc[i-recent_periods:i+1]
                chaikin_window = chaikin_oscillator.iloc[i-recent_periods:i+1]
                
                # 检查是否为价格新高/新低
                current_price = close_price.iloc[i]
                current_chaikin = chaikin_oscillator.iloc[i]
                
                price_is_high = current_price >= price_window.max()
                price_is_low = current_price <= price_window.min()
                chaikin_is_high = current_chaikin >= chaikin_window.max()
                chaikin_is_low = current_chaikin <= chaikin_window.min()
                
                # 正背离：价格创新低但Chaikin未创新低
                if price_is_low and not chaikin_is_low:
                    divergence_score.iloc[i] += 30
                
                # 负背离：价格创新高但Chaikin未创新高
                elif price_is_high and not chaikin_is_high:
                    divergence_score.iloc[i] -= 30
        
        return divergence_score
    
    def _calculate_chaikin_trend_score(self) -> pd.Series:
        """
        计算Chaikin趋势评分
        
        Returns:
            pd.Series: 趋势评分
        """
        trend_score = pd.Series(0.0, index=self._result.index)
        
        chaikin_oscillator = self._result['chaikin_oscillator']
        
        # Chaikin上升趋势+12分
        chaikin_rising = chaikin_oscillator > chaikin_oscillator.shift(1)
        trend_score += chaikin_rising * 12
        
        # Chaikin下降趋势-12分
        chaikin_falling = chaikin_oscillator < chaikin_oscillator.shift(1)
        trend_score -= chaikin_falling * 12
        
        # Chaikin连续上升（3个周期）+18分
        if len(chaikin_oscillator) >= 3:
            consecutive_rising = (
                (chaikin_oscillator > chaikin_oscillator.shift(1)) &
                (chaikin_oscillator.shift(1) > chaikin_oscillator.shift(2)) &
                (chaikin_oscillator.shift(2) > chaikin_oscillator.shift(3))
            )
            trend_score += consecutive_rising * 18
        
        # Chaikin连续下降（3个周期）-18分
        if len(chaikin_oscillator) >= 3:
            consecutive_falling = (
                (chaikin_oscillator < chaikin_oscillator.shift(1)) &
                (chaikin_oscillator.shift(1) < chaikin_oscillator.shift(2)) &
                (chaikin_oscillator.shift(2) < chaikin_oscillator.shift(3))
            )
            trend_score -= consecutive_falling * 18
        
        return trend_score
    
    def _calculate_chaikin_strength_score(self) -> pd.Series:
        """
        计算Chaikin强度评分
        
        Returns:
            pd.Series: 强度评分
        """
        strength_score = pd.Series(0.0, index=self._result.index)
        
        chaikin_oscillator = self._result['chaikin_oscillator']
        
        # 计算Chaikin变化幅度
        chaikin_change = chaikin_oscillator.diff()
        
        # 计算Chaikin的标准差作为强度参考
        chaikin_std = chaikin_oscillator.rolling(20).std()
        
        # Chaikin大幅上升+15分
        large_rise = chaikin_change > chaikin_std
        strength_score += large_rise * 15
        
        # Chaikin大幅下降-15分
        large_fall = chaikin_change < -chaikin_std
        strength_score -= large_fall * 15
        
        # Chaikin快速变化（绝对值>2倍标准差）额外±10分
        rapid_change = np.abs(chaikin_change) > 2 * chaikin_std
        rapid_change_direction = np.sign(chaikin_change)
        strength_score += rapid_change * rapid_change_direction * 10
        
        return strength_score
    
    def _calculate_ad_line_score(self, data: pd.DataFrame) -> pd.Series:
        """
        计算A/D线评分
        
        Args:
            data: 价格数据
            
        Returns:
            pd.Series: A/D线评分
        """
        ad_line_score = pd.Series(0.0, index=self._result.index)
        
        if 'close' not in data.columns:
            return ad_line_score
        
        close_price = data['close']
        ad_line = self._result['ad_line']
        
        # A/D线趋势评分
        ad_rising = ad_line > ad_line.shift(1)
        ad_line_score += ad_rising * 8
        
        ad_falling = ad_line < ad_line.shift(1)
        ad_line_score -= ad_falling * 8
        
        # A/D线与价格关系评分
        price_change = close_price.pct_change()
        ad_change = ad_line.pct_change()
        
        # 价格和A/D线同向变化+10分
        same_direction = (price_change > 0) & (ad_change > 0) | (price_change < 0) & (ad_change < 0)
        ad_line_score += same_direction * 10
        
        # 价格和A/D线反向变化-10分
        opposite_direction = (price_change > 0) & (ad_change < 0) | (price_change < 0) & (ad_change > 0)
        ad_line_score -= opposite_direction * 10
        
        return ad_line_score
    
    def _detect_chaikin_zero_cross_patterns(self) -> List[str]:
        """
        检测Chaikin零轴穿越形态
        
        Returns:
            List[str]: 零轴穿越形态列表
        """
        patterns = []
        
        chaikin_oscillator = self._result['chaikin_oscillator']
        
        # 检查最近的零轴穿越
        recent_periods = min(5, len(chaikin_oscillator))
        recent_chaikin = chaikin_oscillator.tail(recent_periods)
        
        if crossover(recent_chaikin, 0).any():
            patterns.append("Chaikin上穿零轴")
        
        if crossunder(recent_chaikin, 0).any():
            patterns.append("Chaikin下穿零轴")
        
        # 检查当前位置
        if len(chaikin_oscillator) > 0:
            current_chaikin = chaikin_oscillator.iloc[-1]
            if not pd.isna(current_chaikin):
                if current_chaikin > 0:
                    patterns.append("Chaikin零轴上方")
                elif current_chaikin < 0:
                    patterns.append("Chaikin零轴下方")
                else:
                    patterns.append("Chaikin零轴位置")
        
        return patterns
    
    def _detect_chaikin_divergence_patterns(self, data: pd.DataFrame) -> List[str]:
        """
        检测Chaikin背离形态
        
        Args:
            data: 价格数据
            
        Returns:
            List[str]: 背离形态列表
        """
        patterns = []
        
        if 'close' not in data.columns:
            return patterns
        
        close_price = data['close']
        chaikin_oscillator = self._result['chaikin_oscillator']
        
        if len(close_price) >= 20:
            # 检查最近20个周期的趋势
            recent_price = close_price.tail(20)
            recent_chaikin = chaikin_oscillator.tail(20)
            
            # 简化的背离检测
            price_trend = recent_price.iloc[-1] - recent_price.iloc[0]
            chaikin_trend = recent_chaikin.iloc[-1] - recent_chaikin.iloc[0]
            
            # 背离检测
            if price_trend < -0.02 and chaikin_trend > 0:  # 价格下跌但Chaikin上升
                patterns.append("Chaikin正背离")
            elif price_trend > 0.02 and chaikin_trend < 0:  # 价格上涨但Chaikin下降
                patterns.append("Chaikin负背离")
            elif abs(price_trend) < 0.01 and abs(chaikin_trend) < 0.1:
                patterns.append("Chaikin价格同步")
        
        return patterns
    
    def _detect_chaikin_trend_patterns(self) -> List[str]:
        """
        检测Chaikin趋势形态
        
        Returns:
            List[str]: 趋势形态列表
        """
        patterns = []
        
        chaikin_oscillator = self._result['chaikin_oscillator']
        
        # 检查Chaikin趋势
        if len(chaikin_oscillator) >= 3:
            recent_3 = chaikin_oscillator.tail(3)
            if len(recent_3) == 3 and not recent_3.isna().any():
                if (recent_3.iloc[2] > recent_3.iloc[1] > recent_3.iloc[0]):
                    patterns.append("Chaikin连续上升")
                elif (recent_3.iloc[2] < recent_3.iloc[1] < recent_3.iloc[0]):
                    patterns.append("Chaikin连续下降")
        
        # 检查当前趋势
        if len(chaikin_oscillator) >= 2:
            current_chaikin = chaikin_oscillator.iloc[-1]
            prev_chaikin = chaikin_oscillator.iloc[-2]
            
            if not pd.isna(current_chaikin) and not pd.isna(prev_chaikin):
                if current_chaikin > prev_chaikin:
                    patterns.append("Chaikin上升")
                elif current_chaikin < prev_chaikin:
                    patterns.append("Chaikin下降")
                else:
                    patterns.append("Chaikin平稳")
        
        return patterns
    
    def _detect_chaikin_strength_patterns(self) -> List[str]:
        """
        检测Chaikin强度形态
        
        Returns:
            List[str]: 强度形态列表
        """
        patterns = []
        
        chaikin_oscillator = self._result['chaikin_oscillator']
        
        if len(chaikin_oscillator) >= 2:
            current_chaikin = chaikin_oscillator.iloc[-1]
            prev_chaikin = chaikin_oscillator.iloc[-2]
            
            if not pd.isna(current_chaikin) and not pd.isna(prev_chaikin):
                chaikin_change = current_chaikin - prev_chaikin
                
                # 计算变化强度
                if len(chaikin_oscillator) >= 20:
                    chaikin_std = chaikin_oscillator.tail(20).std()
                    
                    if chaikin_change > chaikin_std:
                        patterns.append("Chaikin急速上升")
                    elif chaikin_change > chaikin_std * 0.5:
                        patterns.append("Chaikin大幅上升")
                    elif chaikin_change < -chaikin_std:
                        patterns.append("Chaikin急速下降")
                    elif chaikin_change < -chaikin_std * 0.5:
                        patterns.append("Chaikin大幅下降")
                    elif abs(chaikin_change) <= chaikin_std * 0.1:
                        patterns.append("Chaikin变化平缓")
        
        return patterns
    
    def _detect_ad_line_patterns(self, data: pd.DataFrame) -> List[str]:
        """
        检测A/D线形态
        
        Args:
            data: 价格数据
            
        Returns:
            List[str]: A/D线形态列表
        """
        patterns = []
        
        if 'close' not in data.columns:
            return patterns
        
        close_price = data['close']
        ad_line = self._result['ad_line']
        
        if len(ad_line) >= 2:
            current_ad = ad_line.iloc[-1]
            prev_ad = ad_line.iloc[-2]
            
            if not pd.isna(current_ad) and not pd.isna(prev_ad):
                if current_ad > prev_ad:
                    patterns.append("A/D线上升")
                elif current_ad < prev_ad:
                    patterns.append("A/D线下降")
                else:
                    patterns.append("A/D线平稳")
        
        # 检查A/D线与价格关系
        if len(close_price) >= 5 and len(ad_line) >= 5:
            recent_price = close_price.tail(5)
            recent_ad = ad_line.tail(5)
            
            price_trend = recent_price.iloc[-1] - recent_price.iloc[0]
            ad_trend = recent_ad.iloc[-1] - recent_ad.iloc[0]
            
            if price_trend > 0 and ad_trend > 0:
                patterns.append("价格A/D线同步上升")
            elif price_trend < 0 and ad_trend < 0:
                patterns.append("价格A/D线同步下降")
            elif price_trend > 0 and ad_trend < 0:
                patterns.append("价格上升A/D线下降")
            elif price_trend < 0 and ad_trend > 0:
                patterns.append("价格下降A/D线上升")
        
        return patterns
        
    def plot(self, df: pd.DataFrame, ax=None, **kwargs):
        """
        绘制蔡金指标(Chaikin)图表
        
        Args:
            df: 包含Chaikin指标的DataFrame
            ax: matplotlib轴对象，如果为None则创建新的
            **kwargs: 额外绘图参数
            
        Returns:
            matplotlib轴对象
        """
        import matplotlib.pyplot as plt
        
        # 检查必要的指标列是否存在
        required_columns = ['ad_line', 'chaikin_oscillator', 'chaikin_signal']
        self._validate_dataframe(df, required_columns)
        
        # 创建新的轴对象（如果未提供）
        if ax is None:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        else:
            ax1 = ax
            ax2 = None
        
        # 绘制A/D线
        ax1.plot(df.index, df['ad_line'], label='A/D线', color='blue', linewidth=2)
        ax1.set_title(f"蔡金指标(Chaikin) - A/D线")
        ax1.set_ylabel('A/D线值')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # 如果有第二个子图，绘制Chaikin震荡器
        if ax2 is not None:
            ax2.plot(df.index, df['chaikin_oscillator'], label='Chaikin震荡器', color='green', linewidth=2)
            ax2.plot(df.index, df['chaikin_signal'], label='信号线', color='red', linewidth=1, linestyle='--')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            ax2.set_ylabel('震荡器值')
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)
        
        return ax1 if ax2 is None else (ax1, ax2) 