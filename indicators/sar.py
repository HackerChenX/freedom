#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
抛物线转向系统(SAR)

判断价格趋势反转信号，提供买卖点
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple

from indicators.base_indicator import BaseIndicator
from indicators.common import crossover, crossunder
from utils.logger import get_logger

logger = get_logger(__name__)


class SAR(BaseIndicator):
    """
    抛物线转向系统(SAR) (SAR)
    
    分类：趋势跟踪指标
    描述：判断价格趋势反转信号，提供买卖点
    """
    
    def __init__(self, acceleration: float = 0.02, maximum: float = 0.2):
        """
        初始化抛物线转向系统(SAR)指标
        
        Args:
            acceleration: 加速因子，默认为0.02
            maximum: 加速因子最大值，默认为0.2
        """
        super().__init__()
        self.acceleration = acceleration
        self.maximum = maximum
        self.name = "SAR"
        
    def _validate_dataframe(self, df: pd.DataFrame, required_columns: List[str]) -> None:
        """
        验证DataFrame是否包含所需的列
        
        Args:
            df: 数据帧
            required_columns: 所需的列名列表
        
        Raises:
            ValueError: 如果DataFrame不包含所需的列
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"DataFrame缺少所需的列: {missing_columns}")
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算抛物线转向系统(SAR)指标
        
        Args:
            df: 包含OHLC数据的DataFrame
            
        Returns:
            包含SAR值的DataFrame
        """
        self._validate_dataframe(df, ['high', 'low', 'close'])
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        length = len(df)
        sar = np.zeros(length)
        trend = np.zeros(length)  # 1为上升趋势，-1为下降趋势
        ep = np.zeros(length)     # 极点价格
        af = np.zeros(length)     # 加速因子
        
        # 初始化
        trend[0] = 1  # 假设初始为上升趋势
        ep[0] = high[0]  # 极点价格初始为第一个最高价
        sar[0] = low[0]  # SAR初始为第一个最低价
        af[0] = self.acceleration
        
        # 计算SAR
        for i in range(1, length):
            # 上一个周期是上升趋势
            if trend[i-1] == 1:
                # 计算当前SAR值
                sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
                
                # 确保SAR不高于前两个周期的最低价
                if i >= 2:
                    sar[i] = min(sar[i], min(low[i-1], low[i-2]))
                
                # 判断趋势是否反转
                if low[i] < sar[i]:
                    # 趋势反转为下降
                    trend[i] = -1
                    sar[i] = ep[i-1]  # SAR值设为前期极点
                    ep[i] = low[i]     # 极点设为当前最低价
                    af[i] = self.acceleration  # 加速因子重置
                else:
                    # 继续上升趋势
                    trend[i] = 1
                    # 更新极点和加速因子
                    if high[i] > ep[i-1]:
                        ep[i] = high[i]
                        af[i] = min(af[i-1] + self.acceleration, self.maximum)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
            
            # 上一个周期是下降趋势
            else:
                # 计算当前SAR值
                sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
                
                # 确保SAR不低于前两个周期的最高价
                if i >= 2:
                    sar[i] = max(sar[i], max(high[i-1], high[i-2]))
                
                # 判断趋势是否反转
                if high[i] > sar[i]:
                    # 趋势反转为上升
                    trend[i] = 1
                    sar[i] = ep[i-1]  # SAR值设为前期极点
                    ep[i] = high[i]    # 极点设为当前最高价
                    af[i] = self.acceleration  # 加速因子重置
                else:
                    # 继续下降趋势
                    trend[i] = -1
                    # 更新极点和加速因子
                    if low[i] < ep[i-1]:
                        ep[i] = low[i]
                        af[i] = min(af[i-1] + self.acceleration, self.maximum)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
        
        # 创建结果DataFrame
        result = pd.DataFrame(index=df.index)
        result['sar'] = sar
        result['trend'] = trend
        result['ep'] = ep
        result['af'] = af
        
        # 存储结果
        self._result = result
        
        return result
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        根据SAR生成买卖信号
        
        Args:
            df: 包含SAR值的DataFrame
            
        Returns:
            包含买卖信号的DataFrame
        """
        signals = pd.DataFrame(index=df.index)
        trend = df['trend'].values
        
        buy_signals = np.zeros(len(df))
        sell_signals = np.zeros(len(df))
        
        # 寻找趋势反转点作为信号
        for i in range(1, len(df)):
            # 趋势由下降转为上升，买入信号
            if trend[i] == 1 and trend[i-1] == -1:
                buy_signals[i] = 1
            
            # 趋势由上升转为下降，卖出信号
            if trend[i] == -1 and trend[i-1] == 1:
                sell_signals[i] = 1
        
        signals['buy'] = buy_signals
        signals['sell'] = sell_signals
        
        return signals
    
    def plot(self, df: pd.DataFrame, ax=None):
        """
        绘制SAR图表
        
        Args:
            df: 包含SAR值和OHLC数据的DataFrame
            ax: matplotlib轴对象，如果为None则创建新的
            
        Returns:
            matplotlib轴对象
        """
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制K线图
        ax.plot(df.index, df['close'], color='black', label='Close')
        
        # 绘制SAR点
        for i in range(len(df)):
            if df['trend'].iloc[i] == 1:  # 上升趋势
                ax.scatter(df.index[i], df['sar'].iloc[i], color='green', marker='v', s=20)
            else:  # 下降趋势
                ax.scatter(df.index[i], df['sar'].iloc[i], color='red', marker='^', s=20)
        
        # 添加买卖信号
        signals = self.generate_signals(df)
        buy_points = df.index[signals['buy'] == 1]
        sell_points = df.index[signals['sell'] == 1]
        
        if len(buy_points) > 0:
            buy_prices = df.loc[buy_points, 'close']
            ax.scatter(buy_points, buy_prices, color='green', marker='^', s=100, label='Buy')
        
        if len(sell_points) > 0:
            sell_prices = df.loc[sell_points, 'close']
            ax.scatter(sell_points, sell_prices, color='red', marker='v', s=100, label='Sell')
        
        ax.set_title(f'SAR (acceleration={self.acceleration}, maximum={self.maximum})')
        ax.set_ylabel('Price')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return ax
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算SAR原始评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 原始评分序列（0-100分）
        """
        # 确保已计算SAR
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)
        
        score = pd.Series(50.0, index=data.index)  # 基础分50分
        
        # 获取价格数据
        close_price = data['close']
        high_price = data['high']
        low_price = data['low']
        
        # 1. 价格与SAR关系评分
        price_sar_score = self._calculate_price_sar_score(close_price)
        score += price_sar_score
        
        # 2. SAR趋势反转评分
        reversal_score = self._calculate_sar_reversal_score()
        score += reversal_score
        
        # 3. SAR趋势持续评分
        trend_score = self._calculate_sar_trend_score()
        score += trend_score
        
        # 4. SAR距离评分
        distance_score = self._calculate_sar_distance_score(close_price)
        score += distance_score
        
        # 5. SAR加速因子评分
        acceleration_score = self._calculate_sar_acceleration_score()
        score += acceleration_score
        
        return np.clip(score, 0, 100)
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别SAR技术形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别出的形态列表
        """
        patterns = []
        
        # 确保已计算SAR
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return patterns
        
        close_price = data['close']
        
        # 1. 检测SAR趋势反转形态
        reversal_patterns = self._detect_sar_reversal_patterns()
        patterns.extend(reversal_patterns)
        
        # 2. 检测SAR趋势持续形态
        trend_patterns = self._detect_sar_trend_patterns()
        patterns.extend(trend_patterns)
        
        # 3. 检测价格与SAR关系形态
        price_patterns = self._detect_price_sar_patterns(close_price)
        patterns.extend(price_patterns)
        
        # 4. 检测SAR加速形态
        acceleration_patterns = self._detect_sar_acceleration_patterns()
        patterns.extend(acceleration_patterns)
        
        # 5. 检测SAR支撑阻力形态
        support_resistance_patterns = self._detect_sar_support_resistance_patterns(close_price)
        patterns.extend(support_resistance_patterns)
        
        return patterns
    
    def _calculate_price_sar_score(self, close_price: pd.Series) -> pd.Series:
        """
        计算价格与SAR关系评分
        
        Args:
            close_price: 收盘价序列
            
        Returns:
            pd.Series: 价格关系评分
        """
        price_score = pd.Series(0.0, index=close_price.index)
        
        if 'sar' not in self._result.columns:
            return price_score
        
        sar_values = self._result['sar']
        
        # 价格在SAR上方+15分（强烈看涨信号）
        above_sar = close_price > sar_values
        price_score += above_sar * 15
        
        # 价格在SAR下方-15分（强烈看跌信号）
        below_sar = close_price < sar_values
        price_score -= below_sar * 15
        
        # 价格距离SAR的相对位置评分
        price_distance = abs(close_price - sar_values) / close_price * 100
        
        # 距离适中（1-5%）额外加分，表示信号可靠
        moderate_distance = (price_distance >= 1) & (price_distance <= 5)
        price_score += moderate_distance * 8
        
        # 距离过近（<0.5%）可能即将反转，减分
        too_close = price_distance < 0.5
        price_score -= too_close * 5
        
        return price_score
    
    def _calculate_sar_reversal_score(self) -> pd.Series:
        """
        计算SAR趋势反转评分
        
        Returns:
            pd.Series: 反转评分
        """
        reversal_score = pd.Series(0.0, index=self._result.index)
        
        if 'trend' not in self._result.columns:
            return reversal_score
        
        trend_values = self._result['trend']
        
        # 检测趋势反转
        for i in range(1, len(trend_values)):
            # 由下降转为上升（买入信号）+30分
            if trend_values.iloc[i] == 1 and trend_values.iloc[i-1] == -1:
                reversal_score.iloc[i] += 30
            
            # 由上升转为下降（卖出信号）-30分
            elif trend_values.iloc[i] == -1 and trend_values.iloc[i-1] == 1:
                reversal_score.iloc[i] -= 30
        
        return reversal_score
    
    def _calculate_sar_trend_score(self) -> pd.Series:
        """
        计算SAR趋势持续评分
        
        Returns:
            pd.Series: 趋势评分
        """
        trend_score = pd.Series(0.0, index=self._result.index)
        
        if 'trend' not in self._result.columns:
            return trend_score
        
        trend_values = self._result['trend']
        
        # 计算趋势持续时间
        trend_duration = pd.Series(0, index=self._result.index)
        current_duration = 0
        current_trend = None
        
        for i in range(len(trend_values)):
            if pd.isna(trend_values.iloc[i]):
                continue
                
            if trend_values.iloc[i] == current_trend:
                current_duration += 1
            else:
                current_trend = trend_values.iloc[i]
                current_duration = 1
            
            trend_duration.iloc[i] = current_duration
        
        # 上升趋势持续时间越长，评分越高（但有上限）
        uptrend_duration = trend_duration * (trend_values == 1)
        trend_score += np.minimum(uptrend_duration * 2, 20)  # 最多+20分
        
        # 下降趋势持续时间越长，评分越低（但有下限）
        downtrend_duration = trend_duration * (trend_values == -1)
        trend_score -= np.minimum(downtrend_duration * 2, 20)  # 最多-20分
        
        return trend_score
    
    def _calculate_sar_distance_score(self, close_price: pd.Series) -> pd.Series:
        """
        计算SAR距离评分
        
        Args:
            close_price: 收盘价序列
            
        Returns:
            pd.Series: 距离评分
        """
        distance_score = pd.Series(0.0, index=close_price.index)
        
        if 'sar' not in self._result.columns or 'trend' not in self._result.columns:
            return distance_score
        
        sar_values = self._result['sar']
        trend_values = self._result['trend']
        
        # 计算价格与SAR的距离
        distance = abs(close_price - sar_values) / close_price * 100
        
        # 上升趋势中，距离扩大表示趋势加强
        uptrend_mask = trend_values == 1
        distance_expanding_up = uptrend_mask & (distance > distance.shift(1))
        distance_score += distance_expanding_up * 10
        
        # 下降趋势中，距离扩大表示趋势加强
        downtrend_mask = trend_values == -1
        distance_expanding_down = downtrend_mask & (distance > distance.shift(1))
        distance_score -= distance_expanding_down * 10
        
        # 距离收缩可能预示反转
        distance_contracting = distance < distance.shift(1)
        distance_score -= distance_contracting * 5
        
        return distance_score
    
    def _calculate_sar_acceleration_score(self) -> pd.Series:
        """
        计算SAR加速因子评分
        
        Returns:
            pd.Series: 加速因子评分
        """
        acceleration_score = pd.Series(0.0, index=self._result.index)
        
        if 'af' not in self._result.columns or 'trend' not in self._result.columns:
            return acceleration_score
        
        af_values = self._result['af']
        trend_values = self._result['trend']
        
        # 加速因子增加表示趋势加强
        af_increasing = af_values > af_values.shift(1)
        
        # 上升趋势中加速因子增加+8分
        uptrend_acceleration = (trend_values == 1) & af_increasing
        acceleration_score += uptrend_acceleration * 8
        
        # 下降趋势中加速因子增加-8分
        downtrend_acceleration = (trend_values == -1) & af_increasing
        acceleration_score -= downtrend_acceleration * 8
        
        # 加速因子达到最大值表示强趋势
        max_acceleration = af_values >= self.maximum * 0.9  # 接近最大值
        acceleration_score += max_acceleration * (trend_values == 1) * 12
        acceleration_score -= max_acceleration * (trend_values == -1) * 12
        
        return acceleration_score
    
    def _detect_sar_reversal_patterns(self) -> List[str]:
        """
        检测SAR趋势反转形态
        
        Returns:
            List[str]: 反转形态列表
        """
        patterns = []
        
        if 'trend' not in self._result.columns or len(self._result) < 2:
            return patterns
        
        trend_values = self._result['trend']
        
        # 检查最近的反转
        recent_periods = min(5, len(trend_values))
        recent_trend = trend_values.tail(recent_periods)
        
        for i in range(1, len(recent_trend)):
            current_trend = recent_trend.iloc[i]
            prev_trend = recent_trend.iloc[i-1]
            
            if pd.isna(current_trend) or pd.isna(prev_trend):
                continue
            
            # 由下降转为上升
            if current_trend == 1 and prev_trend == -1:
                patterns.append("SAR上升反转")
            
            # 由上升转为下降
            elif current_trend == -1 and prev_trend == 1:
                patterns.append("SAR下降反转")
        
        return patterns
    
    def _detect_sar_trend_patterns(self) -> List[str]:
        """
        检测SAR趋势持续形态
        
        Returns:
            List[str]: 趋势形态列表
        """
        patterns = []
        
        if 'trend' not in self._result.columns or len(self._result) < 5:
            return patterns
        
        trend_values = self._result['trend']
        current_trend = trend_values.iloc[-1]
        
        if pd.isna(current_trend):
            return patterns
        
        # 计算当前趋势持续时间
        duration = 0
        for i in range(len(trend_values) - 1, -1, -1):
            if pd.isna(trend_values.iloc[i]) or trend_values.iloc[i] != current_trend:
                break
            duration += 1
        
        if current_trend == 1:
            if duration >= 10:
                patterns.append("SAR长期上升趋势")
            elif duration >= 5:
                patterns.append("SAR中期上升趋势")
            else:
                patterns.append("SAR短期上升趋势")
        else:
            if duration >= 10:
                patterns.append("SAR长期下降趋势")
            elif duration >= 5:
                patterns.append("SAR中期下降趋势")
            else:
                patterns.append("SAR短期下降趋势")
        
        return patterns
    
    def _detect_price_sar_patterns(self, close_price: pd.Series) -> List[str]:
        """
        检测价格与SAR关系形态
        
        Args:
            close_price: 收盘价序列
            
        Returns:
            List[str]: 价格关系形态列表
        """
        patterns = []
        
        if 'sar' not in self._result.columns or len(close_price) == 0:
            return patterns
        
        sar_values = self._result['sar']
        current_price = close_price.iloc[-1]
        current_sar = sar_values.iloc[-1]
        
        if pd.isna(current_price) or pd.isna(current_sar):
            return patterns
        
        # 价格与SAR的相对位置
        if current_price > current_sar:
            distance_pct = (current_price - current_sar) / current_price * 100
            if distance_pct > 5:
                patterns.append("价格强势脱离SAR")
            elif distance_pct > 2:
                patterns.append("价格温和高于SAR")
            else:
                patterns.append("价格略高于SAR")
        else:
            distance_pct = (current_sar - current_price) / current_price * 100
            if distance_pct > 5:
                patterns.append("价格强势跌破SAR")
            elif distance_pct > 2:
                patterns.append("价格温和低于SAR")
            else:
                patterns.append("价格略低于SAR")
        
        # 检查价格穿越SAR
        recent_periods = min(5, len(close_price))
        recent_price = close_price.tail(recent_periods)
        recent_sar = sar_values.tail(recent_periods)
        
        if crossover(recent_price, recent_sar).any():
            patterns.append("价格上穿SAR")
        
        if crossunder(recent_price, recent_sar).any():
            patterns.append("价格下穿SAR")
        
        return patterns
    
    def _detect_sar_acceleration_patterns(self) -> List[str]:
        """
        检测SAR加速形态
        
        Returns:
            List[str]: 加速形态列表
        """
        patterns = []
        
        if 'af' not in self._result.columns or 'trend' not in self._result.columns:
            return patterns
        
        af_values = self._result['af']
        trend_values = self._result['trend']
        
        if len(af_values) < 3:
            return patterns
        
        current_af = af_values.iloc[-1]
        current_trend = trend_values.iloc[-1]
        
        if pd.isna(current_af) or pd.isna(current_trend):
            return patterns
        
        # 加速因子变化
        recent_af = af_values.tail(3)
        if len(recent_af) >= 2:
            if recent_af.iloc[-1] > recent_af.iloc[-2]:
                if current_trend == 1:
                    patterns.append("SAR上升加速")
                else:
                    patterns.append("SAR下降加速")
        
        # 加速因子达到最大值
        if current_af >= self.maximum * 0.9:
            if current_trend == 1:
                patterns.append("SAR上升趋势强劲")
            else:
                patterns.append("SAR下降趋势强劲")
        
        return patterns
    
    def _detect_sar_support_resistance_patterns(self, close_price: pd.Series) -> List[str]:
        """
        检测SAR支撑阻力形态
        
        Args:
            close_price: 收盘价序列
            
        Returns:
            List[str]: 支撑阻力形态列表
        """
        patterns = []
        
        if 'sar' not in self._result.columns or 'trend' not in self._result.columns:
            return patterns
        
        sar_values = self._result['sar']
        trend_values = self._result['trend']
        
        if len(close_price) < 5:
            return patterns
        
        recent_periods = min(10, len(close_price))
        recent_price = close_price.tail(recent_periods)
        recent_sar = sar_values.tail(recent_periods)
        recent_trend = trend_values.tail(recent_periods)
        
        # 检查SAR作为支撑
        support_count = 0
        resistance_count = 0
        
        for i in range(1, len(recent_price)):
            if pd.isna(recent_price.iloc[i]) or pd.isna(recent_sar.iloc[i]):
                continue
            
            # 价格接近SAR但未跌破（支撑）
            if (recent_trend.iloc[i] == 1 and 
                recent_price.iloc[i] > recent_sar.iloc[i] and
                abs(recent_price.iloc[i] - recent_sar.iloc[i]) / recent_price.iloc[i] < 0.02):
                support_count += 1
            
            # 价格接近SAR但未突破（阻力）
            elif (recent_trend.iloc[i] == -1 and 
                  recent_price.iloc[i] < recent_sar.iloc[i] and
                  abs(recent_price.iloc[i] - recent_sar.iloc[i]) / recent_price.iloc[i] < 0.02):
                resistance_count += 1
        
        if support_count >= 2:
            patterns.append("SAR形成支撑")
        
        if resistance_count >= 2:
            patterns.append("SAR形成阻力")
        
        return patterns

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算SAR指标
        
        Args:
            df: 包含OHLC数据的DataFrame
            
        Returns:
            包含SAR值和信号的DataFrame
        """
        try:
            result = self.calculate(df)
            signals = self.generate_signals(result)
            # 合并结果
            result['buy_signal'] = signals['buy']
            result['sell_signal'] = signals['sell']
            
            return result
        except Exception as e:
            logger.error(f"计算指标 {self.name} 时出错: {str(e)}")
            raise

