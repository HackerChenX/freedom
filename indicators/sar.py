#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
抛物线转向系统(SAR)

判断价格趋势反转信号，提供买卖点
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple, Any

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
        
        # 注册SAR形态
        self._register_sar_patterns()
        
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

    def _register_sar_patterns(self):
        """
        注册SAR形态
        """
        from indicators.pattern_registry import PatternRegistry, PatternType
        
        # 获取PatternRegistry实例
        registry = PatternRegistry()
        
        # 注册趋势反转形态
        registry.register(
            pattern_id="SAR_BULLISH_REVERSAL",
            display_name="SAR做多信号",
            description="SAR由下降趋势转为上升趋势，产生做多信号",
            indicator_id="SAR",
            pattern_type=PatternType.REVERSAL,
            score_impact=15.0
        )
        
        registry.register(
            pattern_id="SAR_BEARISH_REVERSAL",
            display_name="SAR做空信号",
            description="SAR由上升趋势转为下降趋势，产生做空信号",
            indicator_id="SAR",
            pattern_type=PatternType.REVERSAL,
            score_impact=-15.0
        )
        
        # 注册趋势持续形态
        registry.register(
            pattern_id="SAR_STRONG_UPTREND",
            display_name="SAR强势上升趋势",
            description="SAR长期保持在价格下方，表示强势上升趋势",
            indicator_id="SAR",
            pattern_type=PatternType.TREND,
            score_impact=10.0
        )
        
        registry.register(
            pattern_id="SAR_UPTREND",
            display_name="SAR上升趋势",
            description="SAR保持在价格下方，表示上升趋势",
            indicator_id="SAR",
            pattern_type=PatternType.TREND,
            score_impact=7.0
        )
        
        registry.register(
            pattern_id="SAR_SHORT_UPTREND",
            display_name="SAR短期上升趋势",
            description="SAR刚刚转为上升趋势",
            indicator_id="SAR",
            pattern_type=PatternType.TREND,
            score_impact=5.0
        )
        
        registry.register(
            pattern_id="SAR_STRONG_DOWNTREND",
            display_name="SAR强势下降趋势",
            description="SAR长期保持在价格上方，表示强势下降趋势",
            indicator_id="SAR",
            pattern_type=PatternType.TREND,
            score_impact=-10.0
        )
        
        registry.register(
            pattern_id="SAR_DOWNTREND",
            display_name="SAR下降趋势",
            description="SAR保持在价格上方，表示下降趋势",
            indicator_id="SAR",
            pattern_type=PatternType.TREND,
            score_impact=-7.0
        )
        
        registry.register(
            pattern_id="SAR_SHORT_DOWNTREND",
            display_name="SAR短期下降趋势",
            description="SAR刚刚转为下降趋势",
            indicator_id="SAR",
            pattern_type=PatternType.TREND,
            score_impact=-5.0
        )
        
        # 注册SAR距离形态
        registry.register(
            pattern_id="SAR_CLOSE_TO_PRICE",
            display_name="SAR接近价格",
            description="SAR与价格距离较近，可能即将反转",
            indicator_id="SAR",
            pattern_type=PatternType.WARNING,
            score_impact=0.0
        )
        
        registry.register(
            pattern_id="SAR_MODERATE_DISTANCE",
            display_name="SAR与价格中等距离",
            description="SAR与价格保持中等距离",
            indicator_id="SAR",
            pattern_type=PatternType.CONTINUATION,
            score_impact=5.0
        )
        
        registry.register(
            pattern_id="SAR_FAR_FROM_PRICE",
            display_name="SAR远离价格",
            description="SAR与价格距离较远，趋势强劲",
            indicator_id="SAR",
            pattern_type=PatternType.TREND,
            score_impact=8.0
        )
        
        # 注册加速因子形态
        registry.register(
            pattern_id="SAR_HIGH_ACCELERATION",
            display_name="SAR高加速趋势",
            description="SAR加速因子较高，趋势强劲",
            indicator_id="SAR",
            pattern_type=PatternType.MOMENTUM,
            score_impact=10.0
        )
        
        registry.register(
            pattern_id="SAR_MEDIUM_ACCELERATION",
            display_name="SAR中等加速趋势",
            description="SAR加速因子中等，趋势稳定",
            indicator_id="SAR",
            pattern_type=PatternType.MOMENTUM,
            score_impact=5.0
        )
        
        registry.register(
            pattern_id="SAR_LOW_ACCELERATION",
            display_name="SAR低加速趋势",
            description="SAR加速因子较低，趋势刚开始或较弱",
            indicator_id="SAR",
            pattern_type=PatternType.MOMENTUM,
            score_impact=2.0
        )
        
        # 注册趋势稳定性形态
        registry.register(
            pattern_id="SAR_STABLE_TREND",
            display_name="SAR稳定趋势",
            description="SAR趋势稳定，没有频繁转向",
            indicator_id="SAR",
            pattern_type=PatternType.STABILITY,
            score_impact=8.0
        )
        
        registry.register(
            pattern_id="SAR_VOLATILE_TREND",
            display_name="SAR波动趋势",
            description="SAR趋势不稳定，频繁转向",
            indicator_id="SAR",
            pattern_type=PatternType.STABILITY,
            score_impact=-5.0
        )
        
        # 注册支撑/阻力形态
        registry.register(
            pattern_id="SAR_AS_SUPPORT",
            display_name="SAR支撑",
            description="SAR作为价格支撑位",
            indicator_id="SAR",
            pattern_type=PatternType.SUPPORT_RESISTANCE,
            score_impact=12.0
        )
        
        registry.register(
            pattern_id="SAR_AS_RESISTANCE",
            display_name="SAR阻力",
            description="SAR作为价格阻力位",
            indicator_id="SAR",
            pattern_type=PatternType.SUPPORT_RESISTANCE,
            score_impact=-12.0
        )

    def get_patterns(self, data: pd.DataFrame, **kwargs) -> List[Dict[str, Any]]:
        """
        获取SAR形态列表
        
        Args:
            data: 输入K线数据
            **kwargs: 其他参数
            
        Returns:
            List[Dict[str, Any]]: 形态识别结果列表
        """
        from indicators.base_indicator import PatternResult, SignalStrength
        from indicators.pattern_registry import PatternRegistry, PatternType
        
        # 确保已计算SAR
        if not self.has_result():
            self.calculate(data)
        
        patterns = []
        
        # 如果没有结果或数据不足，返回空列表
        if self._result is None or len(self._result) < 3:
            return patterns
        
        # 获取价格和SAR数据
        sar = self._result['sar']
        trend = self._result['trend']
        close = data['close']
        high = data['high']
        low = data['low']
        
        # 最近的趋势变化
        recent_trends = trend.tail(10)
        
        # 1. SAR趋势反转形态
        if len(trend) >= 2:
            # 检测上升趋势开始（做多信号）
            if trend.iloc[-1] == 1 and trend.iloc[-2] == -1:
                reversal_strength = self._calculate_reversal_strength(data, sar, trend, True)
                patterns.append(PatternResult(
                    pattern_id="SAR_BULLISH_REVERSAL",
                    display_name="SAR做多信号",
                    strength=reversal_strength,
                    duration=1,
                    details={"sar_value": sar.iloc[-1], "price": close.iloc[-1]}
                ).to_dict())
            
            # 检测下降趋势开始（做空信号）
            elif trend.iloc[-1] == -1 and trend.iloc[-2] == 1:
                reversal_strength = self._calculate_reversal_strength(data, sar, trend, False)
                patterns.append(PatternResult(
                    pattern_id="SAR_BEARISH_REVERSAL",
                    display_name="SAR做空信号",
                    strength=reversal_strength,
                    duration=1,
                    details={"sar_value": sar.iloc[-1], "price": close.iloc[-1]}
                ).to_dict())
        
        # 2. SAR趋势持续形态
        current_trend = trend.iloc[-1]
        trend_duration = self._calculate_trend_duration(trend)
        
        if current_trend == 1:  # 上升趋势
            if trend_duration >= 10:
                # 长期上升趋势
                patterns.append(PatternResult(
                    pattern_id="SAR_STRONG_UPTREND",
                    display_name="SAR强势上升趋势",
                    strength=85,
                    duration=trend_duration,
                    details={"duration": trend_duration, "sar_value": sar.iloc[-1]}
                ).to_dict())
            elif trend_duration >= 5:
                # 中期上升趋势
                patterns.append(PatternResult(
                    pattern_id="SAR_UPTREND",
                    display_name="SAR上升趋势",
                    strength=75,
                    duration=trend_duration,
                    details={"duration": trend_duration, "sar_value": sar.iloc[-1]}
                ).to_dict())
            else:
                # 短期上升趋势
                patterns.append(PatternResult(
                    pattern_id="SAR_SHORT_UPTREND",
                    display_name="SAR短期上升趋势",
                    strength=65,
                    duration=trend_duration,
                    details={"duration": trend_duration, "sar_value": sar.iloc[-1]}
                ).to_dict())
        else:  # 下降趋势
            if trend_duration >= 10:
                # 长期下降趋势
                patterns.append(PatternResult(
                    pattern_id="SAR_STRONG_DOWNTREND",
                    display_name="SAR强势下降趋势",
                    strength=85,
                    duration=trend_duration,
                    details={"duration": trend_duration, "sar_value": sar.iloc[-1]}
                ).to_dict())
            elif trend_duration >= 5:
                # 中期下降趋势
                patterns.append(PatternResult(
                    pattern_id="SAR_DOWNTREND",
                    display_name="SAR下降趋势",
                    strength=75,
                    duration=trend_duration,
                    details={"duration": trend_duration, "sar_value": sar.iloc[-1]}
                ).to_dict())
            else:
                # 短期下降趋势
                patterns.append(PatternResult(
                    pattern_id="SAR_SHORT_DOWNTREND",
                    display_name="SAR短期下降趋势",
                    strength=65,
                    duration=trend_duration,
                    details={"duration": trend_duration, "sar_value": sar.iloc[-1]}
                ).to_dict())
        
        # 3. SAR与价格的距离关系
        sar_price_distance = self._calculate_sar_price_distance(sar, close, high, low)
        
        if sar_price_distance == "CLOSE":
            # SAR接近价格，可能即将反转
            patterns.append(PatternResult(
                pattern_id="SAR_CLOSE_TO_PRICE",
                display_name="SAR接近价格",
                strength=90,
                duration=1,
                details={"sar_value": sar.iloc[-1], "price": close.iloc[-1]}
            ).to_dict())
        elif sar_price_distance == "MODERATE":
            # SAR与价格中等距离
            patterns.append(PatternResult(
                pattern_id="SAR_MODERATE_DISTANCE",
                display_name="SAR与价格中等距离",
                strength=70,
                duration=1,
                details={"sar_value": sar.iloc[-1], "price": close.iloc[-1]}
            ).to_dict())
        elif sar_price_distance == "FAR":
            # SAR与价格较远，趋势强劲
            patterns.append(PatternResult(
                pattern_id="SAR_FAR_FROM_PRICE",
                display_name="SAR远离价格",
                strength=80,
                duration=1,
                details={"sar_value": sar.iloc[-1], "price": close.iloc[-1]}
            ).to_dict())
        
        # 4. SAR加速因子状态
        af = self._result['af']
        current_af = af.iloc[-1]
        
        if current_af >= 0.15:
            # 加速因子较高，趋势强烈
            patterns.append(PatternResult(
                pattern_id="SAR_HIGH_ACCELERATION",
                display_name="SAR高加速趋势",
                strength=85,
                duration=1,
                details={"acceleration_factor": current_af}
            ).to_dict())
        elif current_af >= 0.1:
            # 加速因子中等，趋势稳定
            patterns.append(PatternResult(
                pattern_id="SAR_MEDIUM_ACCELERATION",
                display_name="SAR中等加速趋势",
                strength=75,
                duration=1,
                details={"acceleration_factor": current_af}
            ).to_dict())
        else:
            # 加速因子较低，趋势刚开始或较弱
            patterns.append(PatternResult(
                pattern_id="SAR_LOW_ACCELERATION",
                display_name="SAR低加速趋势",
                strength=65,
                duration=1,
                details={"acceleration_factor": current_af}
            ).to_dict())
        
        # 5. SAR趋势稳定性
        trend_stability = self._calculate_trend_stability(trend)
        
        if trend_stability == "STABLE":
            # 稳定趋势
            patterns.append(PatternResult(
                pattern_id="SAR_STABLE_TREND",
                display_name="SAR稳定趋势",
                strength=80,
                duration=5,
                details={"stability": "stable"}
            ).to_dict())
        elif trend_stability == "VOLATILE":
            # 不稳定趋势，频繁转向
            patterns.append(PatternResult(
                pattern_id="SAR_VOLATILE_TREND",
                display_name="SAR波动趋势",
                strength=60,
                duration=5,
                details={"stability": "volatile"}
            ).to_dict())
        
        # 6. SAR支撑/阻力形态
        support_resistance = self._detect_sar_support_resistance(sar, close, trend)
        
        if support_resistance == "SUPPORT":
            # SAR作为支撑
            patterns.append(PatternResult(
                pattern_id="SAR_AS_SUPPORT",
                display_name="SAR支撑",
                strength=75,
                duration=3,
                details={"type": "support"}
            ).to_dict())
        elif support_resistance == "RESISTANCE":
            # SAR作为阻力
            patterns.append(PatternResult(
                pattern_id="SAR_AS_RESISTANCE",
                display_name="SAR阻力",
                strength=75,
                duration=3,
                details={"type": "resistance"}
            ).to_dict())
        
        return patterns

    def _calculate_reversal_strength(self, data: pd.DataFrame, sar: pd.Series, trend: pd.Series, is_bullish: bool) -> float:
        """
        计算SAR反转信号的强度
        
        Args:
            data: 价格数据
            sar: SAR值序列
            trend: 趋势序列
            is_bullish: 是否为看涨反转
            
        Returns:
            float: 信号强度
        """
        close = data['close']
        high = data['high']
        low = data['low']
        
        # 基础强度
        base_strength = 75.0
        
        # 计算当前价格与SAR的距离
        current_close = close.iloc[-1]
        current_sar = sar.iloc[-1]
        
        # 根据价格与SAR的相对距离调整强度
        if is_bullish:
            price_range = high.iloc[-1] - low.iloc[-1]
            if price_range > 0:
                distance_ratio = (current_close - current_sar) / price_range
                distance_score = min(15, distance_ratio * 30)
            else:
                distance_score = 0
        else:
            price_range = high.iloc[-1] - low.iloc[-1]
            if price_range > 0:
                distance_ratio = (current_sar - current_close) / price_range
                distance_score = min(15, distance_ratio * 30)
            else:
                distance_score = 0
        
        # 检查之前趋势的持续时间
        prev_trend_duration = 0
        for i in range(2, min(20, len(trend))):
            if trend.iloc[-i] != trend.iloc[-2]:
                break
            prev_trend_duration += 1
        
        # 之前趋势持续时间越长，反转信号越强
        duration_score = min(10, prev_trend_duration / 2)
        
        # 计算总强度
        total_strength = base_strength + distance_score + duration_score
        
        return min(95, total_strength)

    def _calculate_trend_duration(self, trend: pd.Series) -> int:
        """
        计算当前趋势持续的天数
        
        Args:
            trend: 趋势序列
        
        Returns:
            int: 趋势持续天数
        """
        if len(trend) == 0:
            return 0
        
        current_trend = trend.iloc[-1]
        duration = 0
        
        for i in range(1, len(trend) + 1):
            if i > len(trend) or trend.iloc[-i] != current_trend:
                break
            duration += 1
        
        return duration

    def _calculate_sar_price_distance(self, sar: pd.Series, close: pd.Series, high: pd.Series, low: pd.Series) -> str:
        """
        计算SAR与价格的距离关系
        
        Args:
            sar: SAR值序列
            close: 收盘价序列
            high: 最高价序列
            low: 最低价序列
        
        Returns:
            str: 距离关系描述
        """
        if len(sar) == 0 or len(close) == 0:
            return "UNKNOWN"
        
        current_sar = sar.iloc[-1]
        current_close = close.iloc[-1]
        
        # 计算当日价格范围
        current_range = high.iloc[-1] - low.iloc[-1]
        
        if current_range == 0:
            return "MODERATE"
        
        # 计算SAR与价格的距离占价格范围的比例
        distance_ratio = abs(current_sar - current_close) / current_range
        
        if distance_ratio < 0.5:
            return "CLOSE"
        elif distance_ratio < 1.5:
            return "MODERATE"
        else:
            return "FAR"

    def _calculate_trend_stability(self, trend: pd.Series) -> str:
        """
        计算SAR趋势的稳定性
        
        Args:
            trend: 趋势序列
        
        Returns:
            str: 稳定性描述
        """
        if len(trend) < 10:
            return "UNKNOWN"
        
        # 统计最近10天内的趋势变化次数
        recent_trend = trend.tail(10)
        changes = 0
        
        for i in range(1, len(recent_trend)):
            if recent_trend.iloc[i] != recent_trend.iloc[i-1]:
                changes += 1
        
        if changes <= 1:
            return "STABLE"
        else:
            return "VOLATILE"

    def _detect_sar_support_resistance(self, sar: pd.Series, close: pd.Series, trend: pd.Series) -> Optional[str]:
        """
        检测SAR是否形成支撑或阻力
        
        Args:
            sar: SAR值序列
            close: 收盘价序列
            trend: 趋势序列
        
        Returns:
            Optional[str]: 支撑/阻力描述
        """
        if len(sar) < 5 or len(close) < 5:
            return None
        
        current_trend = trend.iloc[-1]
        current_close = close.iloc[-1]
        current_sar = sar.iloc[-1]
        
        # 检查最近5天的价格是否接近SAR
        recent_close = close.tail(5)
        recent_sar = sar.tail(5)
        
        min_distance = float('inf')
        for i in range(len(recent_close)):
            distance = abs(recent_close.iloc[i] - recent_sar.iloc[i])
            min_distance = min(min_distance, distance)
        
        # 如果价格曾经接近SAR但未突破
        if min_distance < (current_close * 0.01):  # 接近度阈值为1%
            if current_trend == 1:  # 上升趋势
                return "SUPPORT"
            else:  # 下降趋势
                return "RESISTANCE"
        
        return None

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
