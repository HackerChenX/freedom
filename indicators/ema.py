#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
指数移动平均线(EMA)

对近期价格赋予更高权重
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple, Any
import re

from indicators.base_indicator import BaseIndicator
from indicators.common import crossover, crossunder
from utils.logger import get_logger

logger = get_logger(__name__)


class EMA(BaseIndicator):
    """
    指数移动平均线(EMA) (EMA)
    
    分类：趋势类指标
    描述：对近期价格赋予更高权重
    """
    
    def __init__(self, name: str = "EMA", description: str = "指数移动平均线", periods: List[int] = None):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """
        初始化指数移动平均线(EMA)指标
        
        Args:
            name: 指标名称
            description: 指标描述
            periods: 计算周期列表，默认为[5, 10, 20, 60]
        """
        super().__init__(name=name, description=description)
        
        if periods is None:
            self.periods = [5, 10, 20, 60]
        else:
            self.periods = periods
        
        # 注册EMA形态
        self._register_ema_patterns()
        
    def _validate_dataframe(self, df: pd.DataFrame, required_columns: List[str]) -> None:
        """
        验证DataFrame是否包含所需的列
        
        Args:
            df: 要验证的DataFrame
            required_columns: 所需的列名列表
            
        Raises:
            ValueError: 如果缺少必要的列
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"DataFrame缺少必要的列: {', '.join(missing_columns)}")
        
    def _calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算指数移动平均线(EMA)指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
                必须包含以下列：
                - close: 收盘价
                - high: 最高价
                - low: 最低价
                
        Returns:
            添加了EMA指标列的DataFrame
        """
        if df.empty:
            return df
            
        # 确保数据包含必要的列
        required_columns = ['close', 'high', 'low']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 计算不同周期的EMA
        for p in self.periods:
            df_copy[f'EMA{p}'] = df_copy['close'].ewm(span=p, adjust=False).mean()
        
        return df_copy
        
    def get_signals(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成指数移动平均线(EMA)指标交易信号
        
        Args:
            df: 包含价格数据和EMA指标的DataFrame
            **kwargs: 额外参数
                overbought: 超买阈值
                oversold: 超卖阈值
                
        Returns:
            添加了信号列的DataFrame:
            - ema_signal: 1=买入信号, -1=卖出信号, 0=无信号
        """
        if df.empty:
            return df
            
        # 检查必要的指标列是否存在
        required_columns = [f'EMA{self.periods[0]}']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 初始化信号列
        df_copy[f'ema_signal'] = 0
        
        # 如果有多个周期，可以检测金叉和死叉
        if len(self.periods) >= 2 and self.periods[0] < self.periods[1]:
            short_period = self.periods[0]
            long_period = self.periods[1]
            
            # 金叉信号（短期EMA上穿长期EMA）
            df_copy.loc[crossover(df_copy[f'EMA{short_period}'], df_copy[f'EMA{long_period}']), f'ema_signal'] = 1
            
            # 死叉信号（短期EMA下穿长期EMA）
            df_copy.loc[crossunder(df_copy[f'EMA{short_period}'], df_copy[f'EMA{long_period}']), f'ema_signal'] = -1
        
        return df_copy
        
    def plot(self, df: pd.DataFrame, ax=None, **kwargs):
        """
        绘制指数移动平均线(EMA)指标图表
        
        Args:
            df: 包含EMA指标的DataFrame
            ax: matplotlib轴对象，如果为None则创建新的
            **kwargs: 额外绘图参数
            
        Returns:
            matplotlib轴对象
        """
        import matplotlib.pyplot as plt
        
        # 检查必要的指标列是否存在
        for p in self.periods:
            required_columns = [f'EMA{p}']
            self._validate_dataframe(df, required_columns)
        
        # 创建新的轴对象（如果未提供）
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
            
        # 绘制各个周期的EMA指标线
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink']
        for i, p in enumerate(self.periods):
            color = colors[i % len(colors)]
            ax.plot(df.index, df[f'EMA{p}'], label=f'EMA({p})', color=color)
        
        ax.set_ylabel(f'指数移动平均线(EMA)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        return ax

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算指标并返回结果
        
        Args:
            df: 输入DataFrame
            
        Returns:
            包含计算结果的DataFrame
        """
        return self.calculate(df)

    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算EMA原始评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 原始评分序列（0-100分）
        """
        # 确保已计算EMA
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)
        
        score = pd.Series(50.0, index=data.index)  # 基础分50分
        
        # 获取价格数据
        close_price = data['close']
        
        # 1. 价格与EMA关系评分
        price_ema_score = self._calculate_price_ema_score(close_price)
        score += price_ema_score
        
        # 2. EMA交叉评分
        cross_score = self._calculate_ema_cross_score()
        score += cross_score
        
        # 3. EMA趋势评分
        trend_score = self._calculate_ema_trend_score()
        score += trend_score
        
        # 4. EMA排列评分
        arrangement_score = self._calculate_ema_arrangement_score()
        score += arrangement_score
        
        # 5. 价格穿越评分
        penetration_score = self._calculate_price_ema_penetration_score(close_price)
        score += penetration_score
        
        return np.clip(score, 0, 100)
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别EMA技术形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别出的形态列表
        """
        patterns = []
        
        # 确保已计算EMA
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return patterns
        
        close_price = data['close']
        
        # 1. 检测EMA交叉形态
        cross_patterns = self._detect_ema_cross_patterns()
        patterns.extend(cross_patterns)
        
        # 2. 检测EMA排列形态
        arrangement_patterns = self._detect_ema_arrangement_patterns()
        patterns.extend(arrangement_patterns)
        
        # 3. 检测价格与EMA关系形态
        price_patterns = self._detect_price_ema_patterns(close_price)
        patterns.extend(price_patterns)
        
        # 4. 检测EMA趋势形态
        trend_patterns = self._detect_ema_trend_patterns()
        patterns.extend(trend_patterns)
        
        # 5. 检测EMA支撑阻力形态
        support_resistance_patterns = self._detect_ema_support_resistance_patterns(close_price)
        patterns.extend(support_resistance_patterns)
        
        return patterns
    
    def _calculate_price_ema_score(self, close_price: pd.Series) -> pd.Series:
        """
        计算价格与EMA关系评分
        
        Args:
            close_price: 收盘价序列
            
        Returns:
            pd.Series: 价格关系评分
        """
        price_score = pd.Series(0.0, index=close_price.index)
        
        for period in self.periods:
            ema_col = f'EMA{period}'
            if ema_col in self._result.columns:
                ema_values = self._result[ema_col]
                
                # 价格在EMA上方+8分（EMA对近期价格更敏感，权重稍高）
                above_ema = close_price > ema_values
                price_score += above_ema * 8
                
                # 价格在EMA下方-8分
                below_ema = close_price < ema_values
                price_score -= below_ema * 8
                
                # 价格距离EMA的相对位置评分
                price_distance = (close_price - ema_values) / ema_values * 100
                
                # 距离适中（1-3%）额外加分
                moderate_distance = (abs(price_distance) >= 1) & (abs(price_distance) <= 3)
                price_score += moderate_distance * 5
        
        return price_score / len(self.periods)  # 平均化
    
    def _calculate_ema_cross_score(self) -> pd.Series:
        """
        计算EMA交叉评分
        
        Returns:
            pd.Series: 交叉评分
        """
        cross_score = pd.Series(0.0, index=self._result.index)
        
        # 需要至少两个周期才能计算交叉
        if len(self.periods) < 2:
            return cross_score
        
        sorted_periods = sorted(self.periods)
        
        for i in range(len(sorted_periods) - 1):
            short_period = sorted_periods[i]
            long_period = sorted_periods[i + 1]
            
            short_ema = f'EMA{short_period}'
            long_ema = f'EMA{long_period}'
            
            if short_ema in self._result.columns and long_ema in self._result.columns:
                # 金叉（短期EMA上穿长期EMA）+25分（EMA反应更快，权重更高）
                golden_cross = crossover(self._result[short_ema], self._result[long_ema])
                cross_score += golden_cross * 25
                
                # 死叉（短期EMA下穿长期EMA）-25分
                death_cross = crossunder(self._result[short_ema], self._result[long_ema])
                cross_score -= death_cross * 25
        
        return cross_score
    
    def _calculate_ema_trend_score(self) -> pd.Series:
        """
        计算EMA趋势评分
        
        Returns:
            pd.Series: 趋势评分
        """
        trend_score = pd.Series(0.0, index=self._result.index)
        
        for period in self.periods:
            ema_col = f'EMA{period}'
            if ema_col in self._result.columns:
                ema_values = self._result[ema_col]
                
                # EMA上升趋势+10分（EMA对趋势变化更敏感）
                ema_rising = ema_values > ema_values.shift(1)
                trend_score += ema_rising * 10
                
                # EMA下降趋势-10分
                ema_falling = ema_values < ema_values.shift(1)
                trend_score -= ema_falling * 10
                
                # EMA加速上升+15分
                if len(ema_values) >= 3:
                    ema_accelerating = (ema_values.diff() > ema_values.shift(1).diff())
                    trend_score += ema_accelerating * 15
                
                # EMA加速下降-15分
                if len(ema_values) >= 3:
                    ema_decelerating = (ema_values.diff() < ema_values.shift(1).diff())
                    trend_score -= ema_decelerating * 15
        
        return trend_score / len(self.periods)  # 平均化
    
    def _calculate_ema_arrangement_score(self) -> pd.Series:
        """
        计算EMA排列评分
        
        Returns:
            pd.Series: 排列评分
        """
        arrangement_score = pd.Series(0.0, index=self._result.index)
        
        if len(self.periods) < 3:
            return arrangement_score
        
        sorted_periods = sorted(self.periods)
        
        # 检查多头排列（短期EMA在上，长期EMA在下）
        bullish_arrangement = pd.Series(True, index=self._result.index)
        bearish_arrangement = pd.Series(True, index=self._result.index)
        
        for i in range(len(sorted_periods) - 1):
            short_ema = f'EMA{sorted_periods[i]}'
            long_ema = f'EMA{sorted_periods[i + 1]}'
            
            if short_ema in self._result.columns and long_ema in self._result.columns:
                # 多头排列：短期EMA > 长期EMA
                bullish_arrangement &= (self._result[short_ema] > self._result[long_ema])
                
                # 空头排列：短期EMA < 长期EMA
                bearish_arrangement &= (self._result[short_ema] < self._result[long_ema])
        
        # 多头排列+30分（EMA排列信号更强）
        arrangement_score += bullish_arrangement * 30
        
        # 空头排列-30分
        arrangement_score -= bearish_arrangement * 30
        
        return arrangement_score
    
    def _calculate_price_ema_penetration_score(self, close_price: pd.Series) -> pd.Series:
        """
        计算价格穿越EMA评分
        
        Args:
            close_price: 收盘价序列
            
        Returns:
            pd.Series: 穿越评分
        """
        penetration_score = pd.Series(0.0, index=close_price.index)
        
        for period in self.periods:
            ema_col = f'EMA{period}'
            if ema_col in self._result.columns:
                ema_values = self._result[ema_col]
                
                # 价格上穿EMA+18分（EMA穿越信号更敏感）
                price_cross_up = crossover(close_price, ema_values)
                penetration_score += price_cross_up * 18
                
                # 价格下穿EMA-18分
                price_cross_down = crossunder(close_price, ema_values)
                penetration_score -= price_cross_down * 18
        
        return penetration_score / len(self.periods)  # 平均化
    
    def _detect_ema_cross_patterns(self) -> List[str]:
        """
        检测EMA交叉形态
        
        Returns:
            List[str]: 交叉形态列表
        """
        patterns = []
        
        if len(self.periods) < 2:
            return patterns
        
        sorted_periods = sorted(self.periods)
        
        for i in range(len(sorted_periods) - 1):
            short_period = sorted_periods[i]
            long_period = sorted_periods[i + 1]
            
            short_ema = f'EMA{short_period}'
            long_ema = f'EMA{long_period}'
            
            if short_ema in self._result.columns and long_ema in self._result.columns:
                # 检查最近的交叉
                recent_periods = min(5, len(self._result))
                recent_short = self._result[short_ema].tail(recent_periods)
                recent_long = self._result[long_ema].tail(recent_periods)
                
                if crossover(recent_short, recent_long).any():
                    patterns.append(f"EMA{short_period}上穿EMA{long_period}")
                
                if crossunder(recent_short, recent_long).any():
                    patterns.append(f"EMA{short_period}下穿EMA{long_period}")
        
        return patterns
    
    def _detect_ema_arrangement_patterns(self) -> List[str]:
        """
        检测EMA排列形态
        
        Returns:
            List[str]: 排列形态列表
        """
        patterns = []
        
        if len(self.periods) < 3:
            return patterns
        
        sorted_periods = sorted(self.periods)
        
        # 检查当前排列状态
        if len(self._result) > 0:
            current_bullish = True
            current_bearish = True
            
            for i in range(len(sorted_periods) - 1):
                short_ema = f'EMA{sorted_periods[i]}'
                long_ema = f'EMA{sorted_periods[i + 1]}'
                
                if short_ema in self._result.columns and long_ema in self._result.columns:
                    current_short = self._result[short_ema].iloc[-1]
                    current_long = self._result[long_ema].iloc[-1]
                    
                    if pd.isna(current_short) or pd.isna(current_long):
                        continue
                    
                    if current_short <= current_long:
                        current_bullish = False
                    if current_short >= current_long:
                        current_bearish = False
            
            if current_bullish:
                patterns.append("EMA多头排列")
            elif current_bearish:
                patterns.append("EMA空头排列")
            else:
                patterns.append("EMA交织状态")
        
        return patterns
    
    def _detect_price_ema_patterns(self, close_price: pd.Series) -> List[str]:
        """
        检测价格与EMA关系形态
        
        Args:
            close_price: 收盘价序列
            
        Returns:
            List[str]: 价格关系形态列表
        """
        patterns = []
        
        if len(close_price) == 0:
            return patterns
        
        current_price = close_price.iloc[-1]
        above_count = 0
        below_count = 0
        
        for period in self.periods:
            ema_col = f'EMA{period}'
            if ema_col in self._result.columns:
                current_ema = self._result[ema_col].iloc[-1]
                
                if pd.isna(current_ema):
                    continue
                
                if current_price > current_ema:
                    above_count += 1
                elif current_price < current_ema:
                    below_count += 1
        
        total_ema = above_count + below_count
        if total_ema > 0:
            above_ratio = above_count / total_ema
            
            if above_ratio >= 0.8:
                patterns.append("价格强势突破EMA")
            elif above_ratio >= 0.6:
                patterns.append("价格温和上行EMA")
            elif above_ratio <= 0.2:
                patterns.append("价格强势跌破EMA")
            elif above_ratio <= 0.4:
                patterns.append("价格温和下行EMA")
            else:
                patterns.append("价格EMA附近震荡")
        
        # 检查价格穿越
        recent_periods = min(5, len(close_price))
        for period in self.periods:
            ema_col = f'EMA{period}'
            if ema_col in self._result.columns:
                recent_price = close_price.tail(recent_periods)
                recent_ema = self._result[ema_col].tail(recent_periods)
                
                if crossover(recent_price, recent_ema).any():
                    patterns.append(f"价格上穿EMA{period}")
                
                if crossunder(recent_price, recent_ema).any():
                    patterns.append(f"价格下穿EMA{period}")
        
        return patterns
    
    def _detect_ema_trend_patterns(self) -> List[str]:
        """
        检测EMA趋势形态
        
        Returns:
            List[str]: 趋势形态列表
        """
        patterns = []
        
        rising_count = 0
        falling_count = 0
        
        for period in self.periods:
            ema_col = f'EMA{period}'
            if ema_col in self._result.columns and len(self._result) >= 2:
                ema_values = self._result[ema_col]
                current_ema = ema_values.iloc[-1]
                prev_ema = ema_values.iloc[-2]
                
                if pd.isna(current_ema) or pd.isna(prev_ema):
                    continue
                
                if current_ema > prev_ema:
                    rising_count += 1
                elif current_ema < prev_ema:
                    falling_count += 1
        
        total_ema = rising_count + falling_count
        if total_ema > 0:
            rising_ratio = rising_count / total_ema
            
            if rising_ratio >= 0.8:
                patterns.append("EMA全面上升")
            elif rising_ratio >= 0.6:
                patterns.append("EMA多数上升")
            elif rising_ratio <= 0.2:
                patterns.append("EMA全面下降")
            elif rising_ratio <= 0.4:
                patterns.append("EMA多数下降")
            else:
                patterns.append("EMA方向分化")
        
        return patterns
    
    def _detect_ema_support_resistance_patterns(self, close_price: pd.Series) -> List[str]:
        """
        检测EMA支撑阻力形态
        
        Args:
            close_price: 收盘价序列
            
        Returns:
            List[str]: 支撑阻力形态列表
        """
        patterns = []
        
        if len(close_price) < 5:
            return patterns
        
        recent_periods = min(10, len(close_price))
        recent_price = close_price.tail(recent_periods)
        
        for period in self.periods:
            ema_col = f'EMA{period}'
            if ema_col in self._result.columns:
                recent_ema = self._result[ema_col].tail(recent_periods)
                
                # 检查支撑：价格多次接近EMA但未跌破
                support_touches = 0
                resistance_touches = 0
                
                for i in range(1, len(recent_price)):
                    if pd.isna(recent_price.iloc[i]) or pd.isna(recent_ema.iloc[i]):
                        continue
                        
                    price_diff = abs(recent_price.iloc[i] - recent_ema.iloc[i]) / recent_ema.iloc[i]
                    
                    if price_diff < 0.015:  # 1.5%以内认为是接触（EMA更敏感）
                        if recent_price.iloc[i] >= recent_ema.iloc[i]:
                            if i > 0 and recent_price.iloc[i-1] < recent_ema.iloc[i-1]:
                                support_touches += 1
                        else:
                            if i > 0 and recent_price.iloc[i-1] > recent_ema.iloc[i-1]:
                                resistance_touches += 1
                
                if support_touches >= 2:
                    patterns.append(f"EMA{period}形成支撑")
                
                if resistance_touches >= 2:
                    patterns.append(f"EMA{period}形成阻力")
        
        return patterns

    def generate_signals(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        生成EMA指标标准化交易信号
        
        Args:
            data: 输入数据，包含OHLCV数据
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            pd.DataFrame: 信号结果DataFrame，包含标准化信号
        """
        # 确保已计算EMA指标
        if not self.has_result():
            self.calculate(data)
        
        # 初始化信号DataFrame
        signals = pd.DataFrame(index=data.index)
        signals['buy_signal'] = False
        signals['sell_signal'] = False
        signals['neutral_signal'] = True  # 默认为中性信号
        signals['trend'] = 0  # 0表示中性
        signals['score'] = 50.0  # 默认评分50分
        signals['signal_type'] = None
        signals['signal_desc'] = None
        signals['confidence'] = 50.0
        signals['risk_level'] = '中'
        signals['position_size'] = 0.0
        signals['stop_loss'] = None
        signals['market_env'] = 'sideways_market'
        signals['volume_confirmation'] = False
        
        # 计算评分
        score = self.calculate_raw_score(data, **kwargs)
        signals['score'] = score
        
        # 检测形态
        patterns = self.identify_patterns(data, **kwargs)
        
        # 获取价格和EMA数据
        close_price = data['close']
        
        # 需要至少两个周期才能检测交叉
        if len(self.periods) >= 2 and self.periods[0] < self.periods[1]:
            short_period = self.periods[0]
            long_period = self.periods[1]
            
            short_ema = self._result[f'EMA{short_period}']
            long_ema = self._result[f'EMA{long_period}']
            
            # 设置买入信号（短期EMA上穿长期EMA）
            buy_signal_idx = crossover(short_ema, long_ema)
            signals.loc[buy_signal_idx, 'buy_signal'] = True
            signals.loc[buy_signal_idx, 'neutral_signal'] = False
            signals.loc[buy_signal_idx, 'trend'] = 1
            signals.loc[buy_signal_idx, 'signal_type'] = 'EMA金叉'
            signals.loc[buy_signal_idx, 'signal_desc'] = f'EMA{short_period}上穿EMA{long_period}'
            signals.loc[buy_signal_idx, 'confidence'] = 75.0
            signals.loc[buy_signal_idx, 'position_size'] = 0.4
            signals.loc[buy_signal_idx, 'risk_level'] = '中'
            
            # 设置卖出信号（短期EMA下穿长期EMA）
            sell_signal_idx = crossunder(short_ema, long_ema)
            signals.loc[sell_signal_idx, 'sell_signal'] = True
            signals.loc[sell_signal_idx, 'neutral_signal'] = False
            signals.loc[sell_signal_idx, 'trend'] = -1
            signals.loc[sell_signal_idx, 'signal_type'] = 'EMA死叉'
            signals.loc[sell_signal_idx, 'signal_desc'] = f'EMA{short_period}下穿EMA{long_period}'
            signals.loc[sell_signal_idx, 'confidence'] = 75.0
            signals.loc[sell_signal_idx, 'position_size'] = 0.4
            signals.loc[sell_signal_idx, 'risk_level'] = '中'
        
        # 使用主要周期的EMA检测价格与EMA的关系
        main_period = self.periods[0]
        main_ema = self._result[f'EMA{main_period}']
        
        # 设置价格上穿EMA的买入信号
        price_cross_ema_up_idx = crossover(close_price, main_ema)
        signals.loc[price_cross_ema_up_idx, 'buy_signal'] = True
        signals.loc[price_cross_ema_up_idx, 'neutral_signal'] = False
        signals.loc[price_cross_ema_up_idx, 'trend'] = 1
        signals.loc[price_cross_ema_up_idx, 'signal_type'] = '价格上穿EMA'
        signals.loc[price_cross_ema_up_idx, 'signal_desc'] = f'价格上穿EMA{main_period}'
        signals.loc[price_cross_ema_up_idx, 'confidence'] = 70.0
        signals.loc[price_cross_ema_up_idx, 'position_size'] = 0.3
        signals.loc[price_cross_ema_up_idx, 'risk_level'] = '中'
        
        # 设置价格下穿EMA的卖出信号
        price_cross_ema_down_idx = crossunder(close_price, main_ema)
        signals.loc[price_cross_ema_down_idx, 'sell_signal'] = True
        signals.loc[price_cross_ema_down_idx, 'neutral_signal'] = False
        signals.loc[price_cross_ema_down_idx, 'trend'] = -1
        signals.loc[price_cross_ema_down_idx, 'signal_type'] = '价格下穿EMA'
        signals.loc[price_cross_ema_down_idx, 'signal_desc'] = f'价格下穿EMA{main_period}'
        signals.loc[price_cross_ema_down_idx, 'confidence'] = 70.0
        signals.loc[price_cross_ema_down_idx, 'position_size'] = 0.3
        signals.loc[price_cross_ema_down_idx, 'risk_level'] = '中'
        
        # 多EMA排列形成多头排列信号
        if len(self.periods) >= 3:
            periods_sorted = sorted(self.periods)
            
            # 检查当前周期是否形成多头排列（短期EMA > 中期EMA > 长期EMA）
            is_bullish_arrangement = True
            for i in range(len(periods_sorted) - 1):
                short_ema = self._result[f'EMA{periods_sorted[i]}']
                long_ema = self._result[f'EMA{periods_sorted[i+1]}']
                if not (short_ema.iloc[-1] > long_ema.iloc[-1]):
                    is_bullish_arrangement = False
                    break
            
            if is_bullish_arrangement:
                # 最近5个周期形成多头排列
                bullish_arr_idx = signals.index[-5:]
                signals.loc[bullish_arr_idx, 'buy_signal'] = True
                signals.loc[bullish_arr_idx, 'neutral_signal'] = False
                signals.loc[bullish_arr_idx, 'trend'] = 1
                signals.loc[bullish_arr_idx, 'signal_type'] = 'EMA多头排列'
                signals.loc[bullish_arr_idx, 'signal_desc'] = 'EMA形成多头排列，短期均线位于长期均线上方'
                signals.loc[bullish_arr_idx, 'confidence'] = 80.0
                signals.loc[bullish_arr_idx, 'position_size'] = 0.5
                signals.loc[bullish_arr_idx, 'risk_level'] = '低'
            
            # 检查当前周期是否形成空头排列（短期EMA < 中期EMA < 长期EMA）
            is_bearish_arrangement = True
            for i in range(len(periods_sorted) - 1):
                short_ema = self._result[f'EMA{periods_sorted[i]}']
                long_ema = self._result[f'EMA{periods_sorted[i+1]}']
                if not (short_ema.iloc[-1] < long_ema.iloc[-1]):
                    is_bearish_arrangement = False
                    break
            
            if is_bearish_arrangement:
                # 最近5个周期形成空头排列
                bearish_arr_idx = signals.index[-5:]
                signals.loc[bearish_arr_idx, 'sell_signal'] = True
                signals.loc[bearish_arr_idx, 'neutral_signal'] = False
                signals.loc[bearish_arr_idx, 'trend'] = -1
                signals.loc[bearish_arr_idx, 'signal_type'] = 'EMA空头排列'
                signals.loc[bearish_arr_idx, 'signal_desc'] = 'EMA形成空头排列，短期均线位于长期均线下方'
                signals.loc[bearish_arr_idx, 'confidence'] = 80.0
                signals.loc[bearish_arr_idx, 'position_size'] = 0.5
                signals.loc[bearish_arr_idx, 'risk_level'] = '低'
        
        # 根据形态设置更多信号
        for pattern in patterns:
            pattern_idx = signals.index[-5:]  # 假设形态影响最近5个周期
            
            if '支撑' in pattern:
                signals.loc[pattern_idx, 'buy_signal'] = True
                signals.loc[pattern_idx, 'neutral_signal'] = False
                signals.loc[pattern_idx, 'trend'] = 1
                signals.loc[pattern_idx, 'signal_type'] = 'EMA支撑'
                signals.loc[pattern_idx, 'signal_desc'] = pattern
                signals.loc[pattern_idx, 'confidence'] = 75.0
                signals.loc[pattern_idx, 'position_size'] = 0.4
                signals.loc[pattern_idx, 'risk_level'] = '低'
            
            elif '阻力' in pattern:
                signals.loc[pattern_idx, 'sell_signal'] = True
                signals.loc[pattern_idx, 'neutral_signal'] = False
                signals.loc[pattern_idx, 'trend'] = -1
                signals.loc[pattern_idx, 'signal_type'] = 'EMA阻力'
                signals.loc[pattern_idx, 'signal_desc'] = pattern
                signals.loc[pattern_idx, 'confidence'] = 75.0
                signals.loc[pattern_idx, 'position_size'] = 0.4
                signals.loc[pattern_idx, 'risk_level'] = '低'
        
        # 设置止损价格
        if 'low' in data.columns and 'high' in data.columns:
            # 买入信号的止损设为最近的低点或者主EMA线下方
            buy_indices = signals[signals['buy_signal']].index
            if not buy_indices.empty:
                for idx in buy_indices:
                    if idx > data.index[10]:  # 确保有足够的历史数据
                        lookback = 5
                        # 使用最近低点和主EMA值的较小值作为止损
                        recent_low = data.loc[idx-lookback:idx, 'low'].min()
                        ema_stop = main_ema.loc[idx] * 0.98
                        signals.loc[idx, 'stop_loss'] = min(recent_low, ema_stop)
            
            # 卖出信号的止损设为最近的高点或者主EMA线上方
            sell_indices = signals[signals['sell_signal']].index
            if not sell_indices.empty:
                for idx in sell_indices:
                    if idx > data.index[10]:  # 确保有足够的历史数据
                        lookback = 5
                        # 使用最近高点和主EMA值的较大值作为止损
                        recent_high = data.loc[idx-lookback:idx, 'high'].max()
                        ema_stop = main_ema.loc[idx] * 1.02
                        signals.loc[idx, 'stop_loss'] = max(recent_high, ema_stop)
        
        return signals

    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        识别所有已定义的EMA形态，并以DataFrame形式返回

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含所有形态信号的DataFrame
        """
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        result = self._result
        if result is None:
            return pd.DataFrame(index=data.index)

        patterns_df = pd.DataFrame(index=result.index)
        
        close_prices = data['close']
        
        for period in self.periods:
            ema_col = f'EMA{period}'
            if ema_col in result.columns:
                ema_line = result[ema_col]
                patterns_df[f'PRICE_CROSS_ABOVE_EMA{period}'] = crossover(close_prices, ema_line)
                patterns_df[f'PRICE_CROSS_BELOW_EMA{period}'] = crossunder(close_prices, ema_line)

        return patterns_df

    def generate_trading_signals(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """
        生成交易信号
        
        Args:
            data: 输入数据，包含OHLCV数据
            **kwargs: 额外参数
            
        Returns:
            Dict[str, pd.Series]: 信号字典，包含不同类型的信号
        """
        # 确保已计算EMA指标
        if not self.has_result():
            self.calculate(data)
            
        signals = {}
        
        # 获取价格和EMA数据
        close_price = data['close']
        
        # 初始化信号序列
        signals['buy'] = pd.Series(False, index=data.index)
        signals['sell'] = pd.Series(False, index=data.index)
        signals['exit_long'] = pd.Series(False, index=data.index)
        signals['exit_short'] = pd.Series(False, index=data.index)
        
        # 需要至少两个周期才能检测交叉
        if len(self.periods) >= 2 and self.periods[0] < self.periods[1]:
            short_period = self.periods[0]
            long_period = self.periods[1]
            
            short_ema = self._result[f'EMA{short_period}']
            long_ema = self._result[f'EMA{long_period}']
            
            # 设置买入信号（短期EMA上穿长期EMA）
            signals['buy'] = self.crossover(short_ema, long_ema)
            
            # 设置卖出信号（短期EMA下穿长期EMA）
            signals['sell'] = self.crossunder(short_ema, long_ema)
            
            # 设置平仓信号
            signals['exit_long'] = signals['sell']
            signals['exit_short'] = signals['buy']
        
        # 使用主要周期的EMA检测价格与EMA的关系
        main_period = self.periods[0]
        main_ema = self._result[f'EMA{main_period}']
        
        # 设置价格上穿EMA的买入信号
        price_cross_ema_up = self.crossover(close_price, main_ema)
        signals['price_above_ema'] = price_cross_ema_up
        
        # 设置价格下穿EMA的卖出信号
        price_cross_ema_down = self.crossunder(close_price, main_ema)
        signals['price_below_ema'] = price_cross_ema_down
        
        # 合并信号
        signals['buy'] = signals['buy'] | price_cross_ema_up
        signals['sell'] = signals['sell'] | price_cross_ema_down
        
        return signals
        
    def _register_ema_patterns(self):
        """注册EMA指标的形态"""
        from indicators.pattern_registry import PatternRegistry, PatternType
        
        # 获取PatternRegistry实例
        registry = PatternRegistry()
        
        # 注册EMA交叉形态
        registry.register(
            pattern_id="EMA_GOLDEN_CROSS",
            display_name="EMA金叉",
            description="短周期EMA上穿长周期EMA，看涨信号",
            indicator_id="EMA",
            pattern_type=PatternType.REVERSAL,
            score_impact=15.0
        )
        
        registry.register(
            pattern_id="EMA_DEATH_CROSS",
            display_name="EMA死叉",
            description="短周期EMA下穿长周期EMA，看跌信号",
            indicator_id="EMA",
            pattern_type=PatternType.REVERSAL,
            score_impact=-15.0
        )
        
        # 注册EMA排列形态
        registry.register(
            pattern_id="EMA_BULLISH_ALIGNMENT",
            display_name="EMA多头排列",
            description="短周期EMA位于长周期EMA上方，呈阶梯状排列，强势上涨信号",
            indicator_id="EMA",
            pattern_type=PatternType.TREND,
            score_impact=20.0
        )
        
        registry.register(
            pattern_id="EMA_BEARISH_ALIGNMENT",
            display_name="EMA空头排列",
            description="短周期EMA位于长周期EMA下方，呈阶梯状排列，强势下跌信号",
            indicator_id="EMA",
            pattern_type=PatternType.TREND,
            score_impact=-20.0
        )
        
        registry.register(
            pattern_id="EMA_INTERWEAVED",
            display_name="EMA交织状态",
            description="各周期EMA交织在一起，表示市场处于震荡整理中",
            indicator_id="EMA",
            pattern_type=PatternType.CONSOLIDATION,
            score_impact=0.0
        )
        
        # 注册价格与EMA关系形态
        registry.register(
            pattern_id="PRICE_STRONG_ABOVE_EMA",
            display_name="价格强势突破EMA",
            description="价格远高于多数EMA，表示强势上涨",
            indicator_id="EMA",
            pattern_type=PatternType.MOMENTUM,
            score_impact=18.0
        )
        
        registry.register(
            pattern_id="PRICE_ABOVE_EMA",
            display_name="价格温和上行EMA",
            description="价格位于多数EMA上方但距离不远",
            indicator_id="EMA",
            pattern_type=PatternType.TREND,
            score_impact=12.0
        )
        
        registry.register(
            pattern_id="PRICE_STRONG_BELOW_EMA",
            display_name="价格强势跌破EMA",
            description="价格远低于多数EMA，表示强势下跌",
            indicator_id="EMA",
            pattern_type=PatternType.MOMENTUM,
            score_impact=-18.0
        )
        
        registry.register(
            pattern_id="PRICE_BELOW_EMA",
            display_name="价格温和下行EMA",
            description="价格位于多数EMA下方但距离不远",
            indicator_id="EMA",
            pattern_type=PatternType.TREND,
            score_impact=-12.0
        )
        
        registry.register(
            pattern_id="PRICE_NEAR_EMA",
            display_name="价格EMA附近震荡",
            description="价格在EMA附近波动，表示市场处于震荡状态",
            indicator_id="EMA",
            pattern_type=PatternType.CONSOLIDATION,
            score_impact=0.0
        )
        
        registry.register(
            pattern_id="PRICE_CROSS_ABOVE_EMA",
            display_name="价格上穿EMA",
            description="价格上穿某一周期的EMA，可能是买入信号",
            indicator_id="EMA",
            pattern_type=PatternType.REVERSAL,
            score_impact=15.0
        )
        
        registry.register(
            pattern_id="PRICE_CROSS_BELOW_EMA",
            display_name="价格下穿EMA",
            description="价格下穿某一周期的EMA，可能是卖出信号",
            indicator_id="EMA",
            pattern_type=PatternType.REVERSAL,
            score_impact=-15.0
        )
        
        # 注册EMA趋势形态
        registry.register(
            pattern_id="EMA_STRONG_UPTREND",
            display_name="EMA强势上升",
            description="所有周期的EMA都快速上升，表示强势上涨趋势",
            indicator_id="EMA",
            pattern_type=PatternType.TREND,
            score_impact=20.0
        )
        
        registry.register(
            pattern_id="EMA_MODERATE_UPTREND",
            display_name="EMA温和上升",
            description="大部分周期的EMA平缓上升",
            indicator_id="EMA",
            pattern_type=PatternType.TREND,
            score_impact=10.0
        )
        
        registry.register(
            pattern_id="EMA_STRONG_DOWNTREND",
            display_name="EMA强势下降",
            description="所有周期的EMA都快速下降，表示强势下跌趋势",
            indicator_id="EMA",
            pattern_type=PatternType.TREND,
            score_impact=-20.0
        )
        
        registry.register(
            pattern_id="EMA_MODERATE_DOWNTREND",
            display_name="EMA温和下降",
            description="大部分周期的EMA平缓下降",
            indicator_id="EMA",
            pattern_type=PatternType.TREND,
            score_impact=-10.0
        )
        
        registry.register(
            pattern_id="EMA_FLAT",
            display_name="EMA盘整",
            description="多数EMA水平移动，表示市场处于盘整状态",
            indicator_id="EMA",
            pattern_type=PatternType.CONSOLIDATION,
            score_impact=0.0
        )
        
        # 注册EMA支撑/阻力形态
        registry.register(
            pattern_id="EMA_SUPPORT",
            display_name="EMA支撑",
            description="EMA作为价格的支撑位，价格触及后反弹",
            indicator_id="EMA",
            pattern_type=PatternType.SUPPORT,
            score_impact=15.0
        )
        
        registry.register(
            pattern_id="EMA_RESISTANCE",
            display_name="EMA阻力",
            description="EMA作为价格的阻力位，价格触及后回落",
            indicator_id="EMA",
            pattern_type=PatternType.RESISTANCE,
            score_impact=-15.0
        )
        
        registry.register(
            pattern_id="EMA_MULTIPLE_SUPPORT",
            display_name="EMA多重支撑",
            description="多条EMA在相近价位形成支撑带",
            indicator_id="EMA",
            pattern_type=PatternType.SUPPORT,
            score_impact=20.0
        )
        
        registry.register(
            pattern_id="EMA_MULTIPLE_RESISTANCE",
            display_name="EMA多重阻力",
            description="多条EMA在相近价位形成阻力带",
            indicator_id="EMA",
            pattern_type=PatternType.RESISTANCE,
            score_impact=-20.0
        )

