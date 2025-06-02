#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
加权移动平均线(WMA)

对不同时期价格赋予不同权重
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple, Any

from indicators.base_indicator import BaseIndicator, PatternResult
from indicators.common import crossover, crossunder
from utils.logger import get_logger

logger = get_logger(__name__)


class WMA(BaseIndicator):
    """
    加权移动平均线(WMA) (WMA)
    
    分类：趋势类指标
    描述：对不同时期价格赋予不同权重
    """
    
    def __init__(self, period: int = 14, periods: List[int] = None):
        """
        初始化加权移动平均线(WMA)指标
        
        Args:
            period: 计算周期，默认为14
            periods: 多个计算周期，如果提供，将计算多个周期的WMA
        """
        super().__init__()
        self.period = period
        self.periods = periods if periods is not None else [period]
        self.name = "WMA"
        
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
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算加权移动平均线(WMA)指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
                必须包含以下列：
                - close: 收盘价
                - high: 最高价
                - low: 最低价
                
        Returns:
            添加了WMA指标列的DataFrame
        """
        if df.empty:
            return df
            
        # 确保数据包含必要的列
        required_columns = ['close', 'high', 'low']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 计算不同周期的WMA
        for p in self.periods:
            # 创建权重数组，权重与周期成正比
            weights = np.arange(1, p + 1)
            # 计算权重和
            weights_sum = weights.sum()
            
            # 应用加权移动平均计算
            df_copy[f'WMA{p}'] = df_copy['close'].rolling(window=p).apply(
                lambda x: np.sum(x * weights[-len(x):]) / np.sum(weights[-len(x):]), 
                raw=True
            )
        
        # 存储结果
        self._result = df_copy[[f'WMA{p}' for p in self.periods]]
        
        return df_copy
        
    def get_signals(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成加权移动平均线(WMA)指标交易信号
        
        Args:
            df: 包含价格数据和WMA指标的DataFrame
            **kwargs: 额外参数
                overbought: 超买阈值
                oversold: 超卖阈值
                
        Returns:
            添加了信号列的DataFrame:
            - wma_signal: 1=买入信号, -1=卖出信号, 0=无信号
        """
        if df.empty:
            return df
        
        df_copy = df.copy()
        
        # 初始化信号列
        df_copy[f'wma_signal'] = 0
        
        # 如果有多个周期，可以检测金叉和死叉
        if len(self.periods) >= 2 and self.periods[0] < self.periods[1]:
            short_period = self.periods[0]
            long_period = self.periods[1]
            
            # 检查必要的指标列是否存在
            required_columns = [f'WMA{short_period}', f'WMA{long_period}']
            self._validate_dataframe(df_copy, required_columns)
            
            # 金叉信号（短期WMA上穿长期WMA）
            df_copy.loc[crossover(df_copy[f'WMA{short_period}'], df_copy[f'WMA{long_period}']), f'wma_signal'] = 1
            
            # 死叉信号（短期WMA下穿长期WMA）
            df_copy.loc[crossunder(df_copy[f'WMA{short_period}'], df_copy[f'WMA{long_period}']), f'wma_signal'] = -1
        
        return df_copy
        
    def plot(self, df: pd.DataFrame, ax=None, **kwargs):
        """
        绘制加权移动平均线(WMA)指标图表
        
        Args:
            df: 包含WMA指标的DataFrame
            ax: matplotlib轴对象，如果为None则创建新的
            **kwargs: 额外绘图参数
            
        Returns:
            matplotlib轴对象
        """
        import matplotlib.pyplot as plt
        
        # 创建新的轴对象（如果未提供）
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        
        # 绘制各个周期的WMA指标线
        colors = ['blue', 'red', 'green', 'purple', 'orange']
        for i, p in enumerate(self.periods):
            # 检查必要的指标列是否存在
            required_columns = [f'WMA{p}']
            self._validate_dataframe(df, required_columns)
            
            color = colors[i % len(colors)]
            ax.plot(df.index, df[f'WMA{p}'], label=f'WMA({p})', color=color)
        
        ax.set_ylabel(f'加权移动平均线(WMA)')
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
        计算WMA原始评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 原始评分序列（0-100分）
        """
        # 确保已计算WMA
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)
        
        score = pd.Series(50.0, index=data.index)  # 基础分50分
        
        # 获取价格数据
        close_price = data['close']
        
        # 1. 价格与WMA关系评分
        price_wma_score = self._calculate_price_wma_score(close_price)
        score += price_wma_score
        
        # 2. WMA交叉评分
        cross_score = self._calculate_wma_cross_score()
        score += cross_score
        
        # 3. WMA趋势评分
        trend_score = self._calculate_wma_trend_score()
        score += trend_score
        
        # 4. WMA排列评分
        arrangement_score = self._calculate_wma_arrangement_score()
        score += arrangement_score
        
        # 5. 价格穿越评分
        penetration_score = self._calculate_price_wma_penetration_score(close_price)
        score += penetration_score
        
        return np.clip(score, 0, 100)
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别WMA技术形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别出的形态列表
        """
        patterns = []
        
        # 确保已计算WMA
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return patterns
        
        close_price = data['close']
        
        # 1. 检测WMA交叉形态
        cross_patterns = self._detect_wma_cross_patterns()
        patterns.extend(cross_patterns)
        
        # 2. 检测WMA排列形态
        arrangement_patterns = self._detect_wma_arrangement_patterns()
        patterns.extend(arrangement_patterns)
        
        # 3. 检测价格与WMA关系形态
        price_patterns = self._detect_price_wma_patterns(close_price)
        patterns.extend(price_patterns)
        
        # 4. 检测WMA趋势形态
        trend_patterns = self._detect_wma_trend_patterns()
        patterns.extend(trend_patterns)
        
        # 5. 检测WMA支撑阻力形态
        support_resistance_patterns = self._detect_wma_support_resistance_patterns(close_price)
        patterns.extend(support_resistance_patterns)
        
        return patterns
    
    def _calculate_price_wma_score(self, close_price: pd.Series) -> pd.Series:
        """
        计算价格与WMA关系评分
        
        Args:
            close_price: 收盘价序列
            
        Returns:
            pd.Series: 价格关系评分
        """
        price_score = pd.Series(0.0, index=close_price.index)
        
        for period in self.periods:
            wma_col = f'WMA{period}'
            if wma_col in self._result.columns:
                wma_values = self._result[wma_col]
                
                # 价格在WMA上方+6分（WMA对近期价格权重更高，反应更敏感）
                above_wma = close_price > wma_values
                price_score += above_wma * 6
                
                # 价格在WMA下方-6分
                below_wma = close_price < wma_values
                price_score -= below_wma * 6
                
                # 价格距离WMA的相对位置评分
                price_distance = (close_price - wma_values) / wma_values * 100
                
                # 距离适中（1-3%）额外加分
                moderate_distance = (abs(price_distance) >= 1) & (abs(price_distance) <= 3)
                price_score += moderate_distance * 4
        
        return price_score / len(self.periods)  # 平均化
    
    def _calculate_wma_cross_score(self) -> pd.Series:
        """
        计算WMA交叉评分
        
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
            
            short_wma = f'WMA{short_period}'
            long_wma = f'WMA{long_period}'
            
            if short_wma in self._result.columns and long_wma in self._result.columns:
                # 金叉（短期WMA上穿长期WMA）+22分（WMA反应更快，权重稍高）
                golden_cross = crossover(self._result[short_wma], self._result[long_wma])
                cross_score += golden_cross * 22
                
                # 死叉（短期WMA下穿长期WMA）-22分
                death_cross = crossunder(self._result[short_wma], self._result[long_wma])
                cross_score -= death_cross * 22
        
        return cross_score
    
    def _calculate_wma_trend_score(self) -> pd.Series:
        """
        计算WMA趋势评分
        
        Returns:
            pd.Series: 趋势评分
        """
        trend_score = pd.Series(0.0, index=self._result.index)
        
        for period in self.periods:
            wma_col = f'WMA{period}'
            if wma_col in self._result.columns:
                wma_values = self._result[wma_col]
                
                # WMA上升趋势+9分（WMA对趋势变化更敏感）
                wma_rising = wma_values > wma_values.shift(1)
                trend_score += wma_rising * 9
                
                # WMA下降趋势-9分
                wma_falling = wma_values < wma_values.shift(1)
                trend_score -= wma_falling * 9
                
                # WMA加速上升+13分
                if len(wma_values) >= 3:
                    wma_accelerating = (wma_values.diff() > wma_values.shift(1).diff())
                    trend_score += wma_accelerating * 13
                
                # WMA加速下降-13分
                if len(wma_values) >= 3:
                    wma_decelerating = (wma_values.diff() < wma_values.shift(1).diff())
                    trend_score -= wma_decelerating * 13
        
        return trend_score / len(self.periods)  # 平均化
    
    def _calculate_wma_arrangement_score(self) -> pd.Series:
        """
        计算WMA排列评分
        
        Returns:
            pd.Series: 排列评分
        """
        arrangement_score = pd.Series(0.0, index=self._result.index)
        
        if len(self.periods) < 3:
            return arrangement_score
        
        sorted_periods = sorted(self.periods)
        
        # 检查多头排列（短期WMA在上，长期WMA在下）
        bullish_arrangement = pd.Series(True, index=self._result.index)
        bearish_arrangement = pd.Series(True, index=self._result.index)
        
        for i in range(len(sorted_periods) - 1):
            short_wma = f'WMA{sorted_periods[i]}'
            long_wma = f'WMA{sorted_periods[i + 1]}'
            
            if short_wma in self._result.columns and long_wma in self._result.columns:
                # 多头排列：短期WMA > 长期WMA
                bullish_arrangement &= (self._result[short_wma] > self._result[long_wma])
                
                # 空头排列：短期WMA < 长期WMA
                bearish_arrangement &= (self._result[short_wma] < self._result[long_wma])
        
        # 多头排列+27分（WMA排列信号更强）
        arrangement_score += bullish_arrangement * 27
        
        # 空头排列-27分
        arrangement_score -= bearish_arrangement * 27
        
        return arrangement_score
    
    def _calculate_price_wma_penetration_score(self, close_price: pd.Series) -> pd.Series:
        """
        计算价格穿越WMA评分
        
        Args:
            close_price: 收盘价序列
            
        Returns:
            pd.Series: 穿越评分
        """
        penetration_score = pd.Series(0.0, index=close_price.index)
        
        for period in self.periods:
            wma_col = f'WMA{period}'
            if wma_col in self._result.columns:
                wma_values = self._result[wma_col]
                
                # 价格上穿WMA+16分（WMA穿越信号较强）
                price_cross_up = crossover(close_price, wma_values)
                penetration_score += price_cross_up * 16
                
                # 价格下穿WMA-16分
                price_cross_down = crossunder(close_price, wma_values)
                penetration_score -= price_cross_down * 16
        
        return penetration_score / len(self.periods)  # 平均化
    
    def _detect_wma_cross_patterns(self) -> List[str]:
        """
        检测WMA交叉形态
        
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
            
            short_wma = f'WMA{short_period}'
            long_wma = f'WMA{long_period}'
            
            if short_wma in self._result.columns and long_wma in self._result.columns:
                # 检查最近的交叉
                recent_periods = min(5, len(self._result))
                recent_short = self._result[short_wma].tail(recent_periods)
                recent_long = self._result[long_wma].tail(recent_periods)
                
                if crossover(recent_short, recent_long).any():
                    patterns.append(f"WMA{short_period}上穿WMA{long_period}")
                
                if crossunder(recent_short, recent_long).any():
                    patterns.append(f"WMA{short_period}下穿WMA{long_period}")
        
        return patterns
    
    def _detect_wma_arrangement_patterns(self) -> List[str]:
        """
        检测WMA排列形态
        
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
                short_wma = f'WMA{sorted_periods[i]}'
                long_wma = f'WMA{sorted_periods[i + 1]}'
                
                if short_wma in self._result.columns and long_wma in self._result.columns:
                    current_short = self._result[short_wma].iloc[-1]
                    current_long = self._result[long_wma].iloc[-1]
                    
                    if pd.isna(current_short) or pd.isna(current_long):
                        continue
                    
                    if current_short <= current_long:
                        current_bullish = False
                    if current_short >= current_long:
                        current_bearish = False
            
            if current_bullish:
                patterns.append("WMA多头排列")
            elif current_bearish:
                patterns.append("WMA空头排列")
            else:
                patterns.append("WMA交织状态")
        
        return patterns
    
    def _detect_price_wma_patterns(self, close_price: pd.Series) -> List[str]:
        """
        检测价格与WMA关系形态
        
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
            wma_col = f'WMA{period}'
            if wma_col in self._result.columns:
                current_wma = self._result[wma_col].iloc[-1]
                
                if pd.isna(current_wma):
                    continue
                
                if current_price > current_wma:
                    above_count += 1
                elif current_price < current_wma:
                    below_count += 1
        
        total_wma = above_count + below_count
        if total_wma > 0:
            above_ratio = above_count / total_wma
            
            if above_ratio >= 0.8:
                patterns.append("价格强势突破WMA")
            elif above_ratio >= 0.6:
                patterns.append("价格温和上行WMA")
            elif above_ratio <= 0.2:
                patterns.append("价格强势跌破WMA")
            elif above_ratio <= 0.4:
                patterns.append("价格温和下行WMA")
            else:
                patterns.append("价格WMA附近震荡")
        
        # 检查价格穿越
        recent_periods = min(5, len(close_price))
        for period in self.periods:
            wma_col = f'WMA{period}'
            if wma_col in self._result.columns:
                recent_price = close_price.tail(recent_periods)
                recent_wma = self._result[wma_col].tail(recent_periods)
                
                if crossover(recent_price, recent_wma).any():
                    patterns.append(f"价格上穿WMA{period}")
                
                if crossunder(recent_price, recent_wma).any():
                    patterns.append(f"价格下穿WMA{period}")
        
        return patterns
    
    def _detect_wma_trend_patterns(self) -> List[str]:
        """
        检测WMA趋势形态
        
        Returns:
            List[str]: 趋势形态列表
        """
        patterns = []
        
        rising_count = 0
        falling_count = 0
        
        for period in self.periods:
            wma_col = f'WMA{period}'
            if wma_col in self._result.columns and len(self._result) >= 2:
                wma_values = self._result[wma_col]
                current_wma = wma_values.iloc[-1]
                prev_wma = wma_values.iloc[-2]
                
                if pd.isna(current_wma) or pd.isna(prev_wma):
                    continue
                
                if current_wma > prev_wma:
                    rising_count += 1
                elif current_wma < prev_wma:
                    falling_count += 1
        
        total_wma = rising_count + falling_count
        if total_wma > 0:
            rising_ratio = rising_count / total_wma
            
            if rising_ratio >= 0.8:
                patterns.append("WMA全面上升")
            elif rising_ratio >= 0.6:
                patterns.append("WMA多数上升")
            elif rising_ratio <= 0.2:
                patterns.append("WMA全面下降")
            elif rising_ratio <= 0.4:
                patterns.append("WMA多数下降")
            else:
                patterns.append("WMA方向分化")
        
        return patterns
    
    def _detect_wma_support_resistance_patterns(self, close_price: pd.Series) -> List[str]:
        """
        检测WMA支撑阻力形态
        
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
            wma_col = f'WMA{period}'
            if wma_col in self._result.columns:
                recent_wma = self._result[wma_col].tail(recent_periods)
                
                # 检查支撑：价格多次接近WMA但未跌破
                support_touches = 0
                resistance_touches = 0
                
                for i in range(1, len(recent_price)):
                    if pd.isna(recent_price.iloc[i]) or pd.isna(recent_wma.iloc[i]):
                        continue
                        
                    price_diff = abs(recent_price.iloc[i] - recent_wma.iloc[i]) / recent_wma.iloc[i]
                    
                    if price_diff < 0.02:  # 2%以内认为是接触
                        if recent_price.iloc[i] >= recent_wma.iloc[i]:
                            if i > 0 and recent_price.iloc[i-1] < recent_wma.iloc[i-1]:
                                support_touches += 1
                        else:
                            if i > 0 and recent_price.iloc[i-1] > recent_wma.iloc[i-1]:
                                resistance_touches += 1
                
                if support_touches >= 2:
                    patterns.append(f"WMA{period}形成支撑")
                
                if resistance_touches >= 2:
                    patterns.append(f"WMA{period}形成阻力")
        
        return patterns

    def generate_signals(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        生成WMA指标标准化交易信号
        
        Args:
            data: 输入数据，包含OHLCV数据
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            pd.DataFrame: 信号结果DataFrame，包含标准化信号
        """
        # 实现生成信号的逻辑
        pass
        
    def generate_trading_signals(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """
        生成交易信号
        
        Args:
            data: 输入数据，包含OHLCV数据
            **kwargs: 额外参数
            
        Returns:
            Dict[str, pd.Series]: 信号字典，包含不同类型的信号
        """
        # 确保已计算WMA指标
        if not self.has_result():
            self.calculate(data)
            
        signals = {}
        
        # 获取价格和WMA数据
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
            
            short_wma = self._result[f'WMA{short_period}']
            long_wma = self._result[f'WMA{long_period}']
            
            # 设置买入信号（短期WMA上穿长期WMA）
            signals['buy'] = crossover(short_wma, long_wma)
            
            # 设置卖出信号（短期WMA下穿长期WMA）
            signals['sell'] = crossunder(short_wma, long_wma)
            
            # 设置平仓信号
            signals['exit_long'] = signals['sell']
            signals['exit_short'] = signals['buy']
        
        # 使用主要周期的WMA检测价格与WMA的关系
        main_period = self.periods[0]
        main_wma = self._result[f'WMA{main_period}']
        
        # 设置价格上穿WMA的买入信号
        price_cross_wma_up = crossover(close_price, main_wma)
        signals['price_above_wma'] = price_cross_wma_up
        
        # 设置价格下穿WMA的卖出信号
        price_cross_wma_down = crossunder(close_price, main_wma)
        signals['price_below_wma'] = price_cross_wma_down
        
        # 合并信号
        signals['buy'] = signals['buy'] | price_cross_wma_up
        signals['sell'] = signals['sell'] | price_cross_wma_down
        
        return signals
    
    def _register_wma_patterns(self):
        """注册WMA特有的形态检测方法"""
        super()._register_ma_patterns()
        self.register_pattern(self._detect_wma_convergence, "WMA收敛")
        self.register_pattern(self._detect_wma_divergence, "WMA发散")
    
    def _detect_wma_convergence(self, data: pd.DataFrame) -> Optional[PatternResult]:
        # ... 实现WMA收敛检测逻辑 ...
        return PatternResult(pattern_name="WMA收敛", strength=strength, duration=duration)
    
    def _detect_wma_divergence(self, data: pd.DataFrame) -> Optional[PatternResult]:
        # ... 实现WMA发散检测逻辑 ...
        return PatternResult(pattern_name="WMA发散", strength=strength, duration=duration)

    def get_patterns(self, data: pd.DataFrame) -> List[PatternResult]:
        """检测并返回所有识别的形态"""
        self.ensure_columns(data, ['close'] + [f'WMA{period}' for period in self.periods])
        patterns = []
        
        for pattern_func in self._pattern_registry.values():
            result = pattern_func(data)
            if result:
                patterns.append(result)
        
        return patterns

