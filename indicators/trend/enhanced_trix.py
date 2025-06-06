"""
增强型TRIX三重指数平滑移动平均线模块

实现增强型TRIX指标计算，提供自适应参数、多周期协同分析、形态识别等功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple

from indicators.base_indicator import BaseIndicator, MarketEnvironment, SignalStrength
from indicators.trix import TRIX
from utils.logger import get_logger
from utils.technical_utils import find_peaks_and_troughs

logger = get_logger(__name__)


class EnhancedTRIX(TRIX):
    """
    增强型TRIX三重指数平滑移动平均线指标
    
    具有以下增强特性:
    1. 自适应周期调整：根据市场波动率动态调整TRIX参数
    2. 零轴交叉质量评估：评估TRIX与零轴交叉的可靠性
    3. 背离检测系统：检测TRIX与价格之间的背离关系
    4. 多周期TRIX协同分析：结合不同周期的TRIX指标提高信号可靠性
    5. 市场环境自适应：根据市场环境动态调整评分标准
    """
    
    def __init__(self, 
                 n: int = 12, 
                 m: int = 9,
                 secondary_n: int = 24,
                 multi_periods: List[int] = None,
                 adaptive_period: bool = True,
                 volatility_lookback: int = 20):
        """
        初始化增强型TRIX指标
        
        Args:
            n: 主要周期，默认为12
            m: 信号线周期，默认为9
            secondary_n: 次要周期，默认为24
            multi_periods: 多周期分析参数，默认为[6, 12, 24, 48]
            adaptive_period: 是否启用自适应周期，默认为True
            volatility_lookback: 波动率计算回溯期，默认为20
        """
        super().__init__(n=n, m=m)
        self.name = "EnhancedTRIX"
        self.description = "增强型TRIX三重指数平滑移动平均线，优化参数自适应性，增加多周期协同分析和市场环境感知"
        self.secondary_n = secondary_n
        self.multi_periods = multi_periods or [6, 12, 24, 48]
        self.adaptive_period = adaptive_period
        self.volatility_lookback = volatility_lookback
        self.indicator_type = "trend"  # 指标类型：趋势类
        self.market_environment = "normal"
        
        # 内部变量
        self._secondary_trix = None
        self._multi_period_trix = {}
        self._price_data = None
        self._adaptive_n = n  # 自适应后的周期
    
    def get_indicator_type(self) -> str:
        """
        获取指标类型
        
        Returns:
            str: 指标类型
        """
        return self.indicator_type
    
    def set_market_environment(self, environment: str) -> None:
        """
        设置市场环境
        
        Args:
            environment (str): 市场环境类型 ('bull_market', 'bear_market', 'sideways_market', 'volatile_market', 'normal')
        """
        valid_environments = ['bull_market', 'bear_market', 'sideways_market', 'volatile_market', 'normal']
        if environment not in valid_environments:
            raise ValueError(f"无效的市场环境类型: {environment}。有效类型: {valid_environments}")
        
        self.market_environment = environment
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算增强型TRIX指标
        
        Args:
            data: 输入数据，包含OHLC数据
            
        Returns:
            pd.DataFrame: 计算结果，包含TRIX及其相关指标
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close"])
        
        # 保存价格数据用于后续分析
        self._price_data = data['close'].copy()
        
        # 如果启用自适应周期，则调整参数
        if self.adaptive_period:
            self._adjust_period_by_volatility(data)
        
        # 使用调整后的周期计算主要TRIX
        result = super().calculate(data, self._adaptive_n, self.m)
        
        # 计算次要周期TRIX
        secondary_trix = TRIX(n=self.secondary_n, m=self.m)
        secondary_result = secondary_trix.calculate(data)
        result['trix_secondary'] = secondary_result['TRIX']
        result['matrix_secondary'] = secondary_result['MATRIX']
        self._secondary_trix = result['trix_secondary']
        
        # 计算多周期TRIX
        for period in self.multi_periods:
            if period != self._adaptive_n and period != self.secondary_n:
                multi_trix = TRIX(n=period, m=self.m)
                multi_result = multi_trix.calculate(data)
                result[f'trix_{period}'] = multi_result['TRIX']
                result[f'matrix_{period}'] = multi_result['MATRIX']
                self._multi_period_trix[period] = result[f'trix_{period}']
        
        # 计算TRIX动态特性
        result['trix_momentum'] = result['TRIX'] - result['TRIX'].shift(3)
        result['trix_slope'] = self._calculate_slope(result['TRIX'], 5)
        result['trix_accel'] = result['trix_slope'] - result['trix_slope'].shift(1)
        
        # 计算TRIX波动率
        result['trix_volatility'] = result['TRIX'].rolling(window=self.volatility_lookback).std()
        
        # 保存结果
        self._result = result
        
        return result
    
    def _adjust_period_by_volatility(self, data: pd.DataFrame) -> None:
        """
        根据市场波动率动态调整TRIX周期参数
        
        Args:
            data: 包含价格数据的DataFrame
        """
        # 计算价格波动率
        close = data['close']
        
        # 计算价格变化率
        returns = close.pct_change()
        
        # 计算波动率（标准差）
        volatility = returns.rolling(window=self.volatility_lookback).std().iloc[-1]
        
        # 如果波动率数据不足，则使用默认周期
        if pd.isna(volatility):
            self._adaptive_n = self.n
            return
        
        # 计算历史波动率
        historical_volatility = returns.rolling(window=self.volatility_lookback*5).std().iloc[-1]
        
        # 如果历史波动率数据不足，则使用默认周期
        if pd.isna(historical_volatility) or historical_volatility == 0:
            self._adaptive_n = self.n
            return
        
        # 计算相对波动率
        relative_volatility = volatility / historical_volatility if historical_volatility > 0 else 1.0
        
        # 根据相对波动率调整周期
        if relative_volatility > 1.5:  # 高波动市场
            # 增加周期以过滤噪声
            self._adaptive_n = int(self.n * 1.5)
        elif relative_volatility < 0.7:  # 低波动市场
            # 减少周期以提高敏感度
            self._adaptive_n = max(int(self.n * 0.7), 6)  # 确保最小周期为6
        else:  # 正常波动市场
            # 使用默认周期
            self._adaptive_n = self.n
        
        # 根据市场环境进一步调整
        if self.market_environment == 'bull_market':
            # 牛市中略微减少周期，更敏感地捕捉上涨趋势
            self._adaptive_n = max(int(self._adaptive_n * 0.9), 6)
        elif self.market_environment == 'bear_market':
            # 熊市中略微增加周期，过滤更多噪声
            self._adaptive_n = int(self._adaptive_n * 1.1)
        elif self.market_environment == 'volatile_market':
            # 高波动市场中增加周期，过滤更多噪声
            self._adaptive_n = int(self._adaptive_n * 1.2)
        
        logger.debug(f"调整TRIX周期: 原始={self.n}, 调整后={self._adaptive_n}, "
                    f"相对波动率={relative_volatility:.2f}, 市场环境={self.market_environment}")
    
    def _calculate_slope(self, series: pd.Series, period: int = 5) -> pd.Series:
        """
        计算序列的斜率
        
        Args:
            series: 输入序列
            period: 计算斜率的周期
            
        Returns:
            pd.Series: 斜率序列
        """
        slope = pd.Series(np.nan, index=series.index)
        
        for i in range(period, len(series)):
            y = series.iloc[i-period:i].values
            x = np.arange(period)
            
            # 使用线性回归计算斜率
            if len(y) == period and not np.isnan(y).any():
                slope.iloc[i] = np.polyfit(x, y, 1)[0]
        
        return slope 

    def detect_divergence(self) -> pd.DataFrame:
        """
        检测TRIX与价格之间的背离关系
        
        Returns:
            pd.DataFrame: 包含背离分析结果的DataFrame
        """
        if self._result is None or self._price_data is None:
            return pd.DataFrame()
            
        # 获取TRIX和价格数据
        trix = self._result['TRIX']
        price = self._price_data
        
        # 创建结果DataFrame
        divergence = pd.DataFrame(index=price.index)
        divergence['bullish_divergence'] = False
        divergence['bearish_divergence'] = False
        divergence['hidden_bullish_divergence'] = False
        divergence['hidden_bearish_divergence'] = False
        divergence['divergence_strength'] = 0.0
        
        # 查找价格和TRIX的高点和低点
        price_peaks = find_peaks_and_troughs(price, window=10, peak_type='peak')
        price_troughs = find_peaks_and_troughs(price, window=10, peak_type='trough')
        trix_peaks = find_peaks_and_troughs(trix, window=10, peak_type='peak')
        trix_troughs = find_peaks_and_troughs(trix, window=10, peak_type='trough')
        
        # 最小背离长度(防止检测到太短的背离)
        min_divergence_length = 5
        # 最大背离长度(防止检测到太长的背离)
        max_divergence_length = 30
        
        # 常规看涨背离：价格创新低但TRIX未创新低
        for i in range(1, len(price_troughs)):
            if i >= len(price_troughs) or price_troughs[i] >= len(price):
                continue
                
            current_trough_idx = price_troughs[i]
            prev_trough_idx = price_troughs[i-1]
            
            # 检查背离长度是否合适
            if (current_trough_idx - prev_trough_idx < min_divergence_length or 
                current_trough_idx - prev_trough_idx > max_divergence_length):
                continue
                
            # 价格创新低
            if price.iloc[current_trough_idx] < price.iloc[prev_trough_idx]:
                # 查找对应的TRIX值
                current_trix_trough = None
                prev_trix_trough = None
                
                # 在价格低点附近查找TRIX低点
                for tt in trix_troughs:
                    if abs(tt - current_trough_idx) <= 3:
                        current_trix_trough = tt
                    if abs(tt - prev_trough_idx) <= 3:
                        prev_trix_trough = tt
                
                # 如果找到了对应的TRIX低点
                if current_trix_trough is not None and prev_trix_trough is not None:
                    # TRIX未创新低
                    if trix.iloc[current_trix_trough] > trix.iloc[prev_trix_trough]:
                        # 计算背离强度
                        price_change = (price.iloc[current_trough_idx] / price.iloc[prev_trough_idx]) - 1
                        trix_change = (trix.iloc[current_trix_trough] / trix.iloc[prev_trix_trough]) - 1
                        # 防止除以零
                        if max(abs(price_change), abs(trix_change)) > 0:
                            strength = abs(price_change - trix_change) / max(abs(price_change), abs(trix_change))
                        else:
                            strength = 0
                        
                        # 记录背离
                        divergence.iloc[current_trough_idx:current_trough_idx+5, 0] = True  # bullish_divergence
                        divergence.iloc[current_trough_idx:current_trough_idx+5, 4] = strength  # divergence_strength
        
        # 常规看跌背离：价格创新高但TRIX未创新高
        for i in range(1, len(price_peaks)):
            if i >= len(price_peaks) or price_peaks[i] >= len(price):
                continue
                
            current_peak_idx = price_peaks[i]
            prev_peak_idx = price_peaks[i-1]
            
            # 检查背离长度是否合适
            if (current_peak_idx - prev_peak_idx < min_divergence_length or 
                current_peak_idx - prev_peak_idx > max_divergence_length):
                continue
                
            # 价格创新高
            if price.iloc[current_peak_idx] > price.iloc[prev_peak_idx]:
                # 查找对应的TRIX值
                current_trix_peak = None
                prev_trix_peak = None
                
                # 在价格高点附近查找TRIX高点
                for tp in trix_peaks:
                    if abs(tp - current_peak_idx) <= 3:
                        current_trix_peak = tp
                    if abs(tp - prev_peak_idx) <= 3:
                        prev_trix_peak = tp
                
                # 如果找到了对应的TRIX高点
                if current_trix_peak is not None and prev_trix_peak is not None:
                    # TRIX未创新高
                    if trix.iloc[current_trix_peak] < trix.iloc[prev_trix_peak]:
                        # 计算背离强度
                        price_change = (price.iloc[current_peak_idx] / price.iloc[prev_peak_idx]) - 1
                        trix_change = (trix.iloc[current_trix_peak] / trix.iloc[prev_trix_peak]) - 1
                        # 防止除以零
                        if max(abs(price_change), abs(trix_change)) > 0:
                            strength = abs(price_change - trix_change) / max(abs(price_change), abs(trix_change))
                        else:
                            strength = 0
                        
                        # 记录背离
                        divergence.iloc[current_peak_idx:current_peak_idx+5, 1] = True  # bearish_divergence
                        divergence.iloc[current_peak_idx:current_peak_idx+5, 4] = strength  # divergence_strength
        
        # 隐藏看涨背离：价格更高的低点但TRIX更低的低点
        for i in range(1, len(price_troughs)):
            if i >= len(price_troughs) or price_troughs[i] >= len(price):
                continue
                
            current_trough_idx = price_troughs[i]
            prev_trough_idx = price_troughs[i-1]
            
            # 检查背离长度是否合适
            if (current_trough_idx - prev_trough_idx < min_divergence_length or 
                current_trough_idx - prev_trough_idx > max_divergence_length):
                continue
                
            # 价格更高的低点
            if price.iloc[current_trough_idx] > price.iloc[prev_trough_idx]:
                # 查找对应的TRIX值
                current_trix_trough = None
                prev_trix_trough = None
                
                # 在价格低点附近查找TRIX低点
                for tt in trix_troughs:
                    if abs(tt - current_trough_idx) <= 3:
                        current_trix_trough = tt
                    if abs(tt - prev_trough_idx) <= 3:
                        prev_trix_trough = tt
                
                # 如果找到了对应的TRIX低点
                if current_trix_trough is not None and prev_trix_trough is not None:
                    # TRIX更低的低点
                    if trix.iloc[current_trix_trough] < trix.iloc[prev_trix_trough]:
                        # 计算背离强度
                        price_change = (price.iloc[current_trough_idx] / price.iloc[prev_trough_idx]) - 1
                        trix_change = (trix.iloc[current_trix_trough] / trix.iloc[prev_trix_trough]) - 1
                        # 防止除以零
                        if max(abs(price_change), abs(trix_change)) > 0:
                            strength = abs(price_change - trix_change) / max(abs(price_change), abs(trix_change))
                        else:
                            strength = 0
                        
                        # 记录背离
                        divergence.iloc[current_trough_idx:current_trough_idx+5, 2] = True  # hidden_bullish_divergence
                        divergence.iloc[current_trough_idx:current_trough_idx+5, 4] = strength  # divergence_strength
        
        # 隐藏看跌背离：价格更低的高点但TRIX更高的高点
        for i in range(1, len(price_peaks)):
            if i >= len(price_peaks) or price_peaks[i] >= len(price):
                continue
                
            current_peak_idx = price_peaks[i]
            prev_peak_idx = price_peaks[i-1]
            
            # 检查背离长度是否合适
            if (current_peak_idx - prev_peak_idx < min_divergence_length or 
                current_peak_idx - prev_peak_idx > max_divergence_length):
                continue
                
            # 价格更低的高点
            if price.iloc[current_peak_idx] < price.iloc[prev_peak_idx]:
                # 查找对应的TRIX值
                current_trix_peak = None
                prev_trix_peak = None
                
                # 在价格高点附近查找TRIX高点
                for tp in trix_peaks:
                    if abs(tp - current_peak_idx) <= 3:
                        current_trix_peak = tp
                    if abs(tp - prev_peak_idx) <= 3:
                        prev_trix_peak = tp
                
                # 如果找到了对应的TRIX高点
                if current_trix_peak is not None and prev_trix_peak is not None:
                    # TRIX更高的高点
                    if trix.iloc[current_trix_peak] > trix.iloc[prev_trix_peak]:
                        # 计算背离强度
                        price_change = (price.iloc[current_peak_idx] / price.iloc[prev_peak_idx]) - 1
                        trix_change = (trix.iloc[current_trix_peak] / trix.iloc[prev_trix_peak]) - 1
                        # 防止除以零
                        if max(abs(price_change), abs(trix_change)) > 0:
                            strength = abs(price_change - trix_change) / max(abs(price_change), abs(trix_change))
                        else:
                            strength = 0
                        
                        # 记录背离
                        divergence.iloc[current_peak_idx:current_peak_idx+5, 3] = True  # hidden_bearish_divergence
                        divergence.iloc[current_peak_idx:current_peak_idx+5, 4] = strength  # divergence_strength
        
        return divergence
    
    def analyze_multi_period_synergy(self) -> pd.DataFrame:
        """
        多周期TRIX协同分析
        
        Returns:
            pd.DataFrame: 包含多周期协同分析结果的DataFrame
        """
        if self._result is None:
            return pd.DataFrame()
            
        # 创建结果DataFrame
        synergy = pd.DataFrame(index=self._result.index)
        
        # 获取主要TRIX
        primary_trix = self._result['TRIX']
        secondary_trix = self._secondary_trix
        
        # 分析多周期趋势一致性
        synergy['primary_above_zero'] = primary_trix > 0
        synergy['primary_below_zero'] = primary_trix < 0
        synergy['secondary_above_zero'] = secondary_trix > 0
        synergy['secondary_below_zero'] = secondary_trix < 0
        
        # 计算TRIX方向
        synergy['primary_rising'] = primary_trix > primary_trix.shift(1)
        synergy['primary_falling'] = primary_trix < primary_trix.shift(1)
        synergy['secondary_rising'] = secondary_trix > secondary_trix.shift(1)
        synergy['secondary_falling'] = secondary_trix < secondary_trix.shift(1)
        
        # 计算多周期一致性
        bullish_count = synergy['primary_above_zero'].astype(int) + synergy['primary_rising'].astype(int)
        bearish_count = synergy['primary_below_zero'].astype(int) + synergy['primary_falling'].astype(int)
        
        # 添加次要周期的状态
        bullish_count += synergy['secondary_above_zero'].astype(int) + synergy['secondary_rising'].astype(int)
        bearish_count += synergy['secondary_below_zero'].astype(int) + synergy['secondary_falling'].astype(int)
        
        # 添加其他周期的状态
        for period, trix_series in self._multi_period_trix.items():
            period_above_zero = trix_series > 0
            period_below_zero = trix_series < 0
            period_rising = trix_series > trix_series.shift(1)
            period_falling = trix_series < trix_series.shift(1)
            
            synergy[f'trix{period}_above_zero'] = period_above_zero
            synergy[f'trix{period}_below_zero'] = period_below_zero
            synergy[f'trix{period}_rising'] = period_rising
            synergy[f'trix{period}_falling'] = period_falling
            
            bullish_count += period_above_zero.astype(int) + period_rising.astype(int)
            bearish_count += period_below_zero.astype(int) + period_falling.astype(int)
        
        # 计算总检查数（正面+负面特征的总数）
        total_checks = len(self._multi_period_trix) * 2 + 4  # 每个周期有2个检查（零轴位置和方向），加上主要和次要周期的4个检查
        
        # 计算看涨和看跌比例
        synergy['bullish_ratio'] = bullish_count / total_checks
        synergy['bearish_ratio'] = bearish_count / total_checks
        
        # 计算一致性得分（0-100）
        synergy['consensus_score'] = 50 + (synergy['bullish_ratio'] - synergy['bearish_ratio']) * 50
        
        # 多周期交叉信号
        synergy['multi_period_bullish_signal'] = False
        synergy['multi_period_bearish_signal'] = False
        
        # 检测多周期协同交叉信号
        primary_cross_up_zero = self.crossover(primary_trix, 0)
        primary_cross_down_zero = self.crossunder(primary_trix, 0)
        secondary_cross_up_zero = self.crossover(secondary_trix, 0)
        secondary_cross_down_zero = self.crossunder(secondary_trix, 0)
        
        # 主要周期和次要周期同时发出信号
        bullish_signal = primary_cross_up_zero & (secondary_trix > 0)
        bearish_signal = primary_cross_down_zero & (secondary_trix < 0)
        
        # 在较短周期信号之后较长周期也发出信号，强化确认
        for period, trix_series in self._multi_period_trix.items():
            if period > self._adaptive_n:  # 只考虑长周期
                period_cross_up_zero = self.crossover(trix_series, 0)
                period_cross_down_zero = self.crossunder(trix_series, 0)
                
                # 在主周期信号后10个周期内，长周期也发出相同信号
                for i in range(len(bullish_signal)):
                    if i >= 10 and bullish_signal.iloc[i-10]:
                        for j in range(1, 11):
                            if i+j < len(period_cross_up_zero) and period_cross_up_zero.iloc[i+j]:
                                bullish_signal.iloc[i+j] = True
                
                for i in range(len(bearish_signal)):
                    if i >= 10 and bearish_signal.iloc[i-10]:
                        for j in range(1, 11):
                            if i+j < len(period_cross_down_zero) and period_cross_down_zero.iloc[i+j]:
                                bearish_signal.iloc[i+j] = True
        
        synergy['multi_period_bullish_signal'] = bullish_signal
        synergy['multi_period_bearish_signal'] = bearish_signal
        
        return synergy 

    def evaluate_zero_cross_quality(self) -> pd.DataFrame:
        """
        评估TRIX零轴交叉质量
        
        Returns:
            pd.DataFrame: 包含零轴交叉质量评估结果的DataFrame
        """
        if self._result is None:
            return pd.DataFrame()
            
        # 获取TRIX数据
        trix = self._result['TRIX']
        
        # 创建结果DataFrame
        quality = pd.DataFrame(index=self._result.index)
        
        # 检测零轴交叉
        cross_up_zero = self.crossover(trix, 0)
        cross_down_zero = self.crossunder(trix, 0)
        
        quality['cross_up_zero'] = cross_up_zero
        quality['cross_down_zero'] = cross_down_zero
        
        # 评估交叉角度
        quality['cross_angle'] = 0.0
        
        for i in range(5, len(trix)):
            if cross_up_zero.iloc[i] or cross_down_zero.iloc[i]:
                # 计算交叉前后5个周期的斜率
                pre_slope = (trix.iloc[i] - trix.iloc[i-5]) / 5
                quality.iloc[i, 2] = abs(pre_slope)  # cross_angle
        
        # 评估交叉后的加速度
        quality['post_cross_acceleration'] = 0.0
        
        for i in range(5, len(trix)-5):
            if cross_up_zero.iloc[i] or cross_down_zero.iloc[i]:
                if i+5 < len(trix):
                    # 计算交叉后5个周期的加速度
                    slope1 = (trix.iloc[i+1] - trix.iloc[i]) / 1
                    slope5 = (trix.iloc[i+5] - trix.iloc[i]) / 5
                    accel = slope5 - slope1
                    
                    if (cross_up_zero.iloc[i] and accel > 0) or (cross_down_zero.iloc[i] and accel < 0):
                        quality.iloc[i, 3] = abs(accel)  # post_cross_acceleration
        
        # 评估交叉持续性
        quality['cross_persistence'] = 0.0
        
        for i in range(5, len(trix)-10):
            if cross_up_zero.iloc[i]:
                # 检查交叉后10个周期内是否保持方向
                persistence = 0
                for j in range(1, 11):
                    if i+j < len(trix) and trix.iloc[i+j] > 0:
                        persistence += 1
                
                quality.iloc[i, 4] = persistence / 10  # cross_persistence
            
            elif cross_down_zero.iloc[i]:
                # 检查交叉后10个周期内是否保持方向
                persistence = 0
                for j in range(1, 11):
                    if i+j < len(trix) and trix.iloc[i+j] < 0:
                        persistence += 1
                
                quality.iloc[i, 4] = persistence / 10  # cross_persistence
        
        # 综合评分（0-100）
        quality['cross_quality_score'] = 0.0
        
        for i in range(len(trix)):
            if cross_up_zero.iloc[i] or cross_down_zero.iloc[i]:
                # 综合考虑角度、加速度和持续性
                angle_score = min(40, quality.iloc[i, 2] * 80)
                accel_score = min(30, quality.iloc[i, 3] * 60)
                persistence_score = quality.iloc[i, 4] * 30
                
                quality.iloc[i, 5] = angle_score + accel_score + persistence_score  # cross_quality_score
        
        return quality
    
    def identify_patterns(self) -> pd.DataFrame:
        """
        识别TRIX形态
        
        Returns:
            pd.DataFrame: 包含形态识别结果的DataFrame
        """
        if self._result is None:
            return pd.DataFrame()
            
        # 获取TRIX数据
        trix = self._result['TRIX']
        matrix = self._result['MATRIX']
        
        # 创建结果DataFrame
        patterns = pd.DataFrame(index=self._result.index)
        
        # 基础趋势形态
        patterns['above_zero'] = trix > 0
        patterns['below_zero'] = trix < 0
        patterns['rising'] = trix > trix.shift(1)
        patterns['falling'] = trix < trix.shift(1)
        
        # 计算TRIX交叉信号
        patterns['golden_cross'] = self.crossover(trix, matrix)
        patterns['death_cross'] = self.crossunder(trix, matrix)
        patterns['cross_up_zero'] = self.crossover(trix, 0)
        patterns['cross_down_zero'] = self.crossunder(trix, 0)
        
        # 高质量零轴交叉
        zero_cross_quality = self.evaluate_zero_cross_quality()
        if 'cross_quality_score' in zero_cross_quality.columns:
            high_quality_threshold = 70
            patterns['high_quality_cross_up_zero'] = (patterns['cross_up_zero'] & 
                                                    (zero_cross_quality['cross_quality_score'] > high_quality_threshold))
            patterns['high_quality_cross_down_zero'] = (patterns['cross_down_zero'] & 
                                                      (zero_cross_quality['cross_quality_score'] > high_quality_threshold))
        
        # 获取背离分析结果
        divergence = self.detect_divergence()
        if not divergence.empty:
            patterns['bullish_divergence'] = divergence['bullish_divergence']
            patterns['bearish_divergence'] = divergence['bearish_divergence']
            patterns['hidden_bullish_divergence'] = divergence['hidden_bullish_divergence']
            patterns['hidden_bearish_divergence'] = divergence['hidden_bearish_divergence']
        
        # 多周期协同分析
        synergy = self.analyze_multi_period_synergy()
        if not synergy.empty:
            patterns['multi_period_bullish_signal'] = synergy['multi_period_bullish_signal']
            patterns['multi_period_bearish_signal'] = synergy['multi_period_bearish_signal']
            patterns['strong_bullish_consensus'] = synergy['consensus_score'] > 70
            patterns['strong_bearish_consensus'] = synergy['consensus_score'] < 30
        
        # 趋势加速/减速
        if 'trix_accel' in self._result.columns:
            patterns['acceleration'] = self._result['trix_accel'] > 0
            patterns['deceleration'] = self._result['trix_accel'] < 0
        
        # 钝化形态（TRIX在零轴附近徘徊）
        patterns['stagnation_near_zero'] = self._detect_stagnation(trix, threshold=0.1, periods=5)
        
        return patterns
    
    def _detect_stagnation(self, trix: pd.Series, threshold: float, periods: int) -> pd.Series:
        """
        检测TRIX在零轴附近的钝化形态
        
        Args:
            trix: TRIX序列
            threshold: 零轴附近的阈值
            periods: 持续周期数
            
        Returns:
            pd.Series: 钝化形态识别结果
        """
        stagnation = pd.Series(False, index=trix.index)
        
        if len(trix) < periods:
            return stagnation
        
        # 检查连续periods个周期TRIX都在零轴附近
        for i in range(periods, len(trix)):
            window = trix.iloc[i-periods+1:i+1]
            
            # 检查是否所有值都在零轴附近
            near_zero = (abs(window) < threshold).all()
            
            # 检查波动性是否低
            low_volatility = window.std() < threshold/2
            
            if near_zero and low_volatility:
                stagnation.iloc[i] = True
        
        return stagnation
    
    def calculate_score(self, data: pd.DataFrame = None) -> pd.Series:
        """
        计算TRIX综合评分 (0-100)
        
        Args:
            data (pd.DataFrame, optional): 价格数据，如果未提供则使用上次计算结果
            
        Returns:
            pd.Series: 评分 (0-100，50为中性)
        """
        if self._result is None and data is not None:
            self.calculate(data)
            
        if self._result is None:
            return pd.Series()
        
        # 获取TRIX数据
        trix = self._result['TRIX']
        matrix = self._result['MATRIX']
        trix_momentum = self._result['trix_momentum']
        trix_slope = self._result['trix_slope']
        
        # 获取背离分析
        divergence = self.detect_divergence()
        
        # 获取零轴交叉质量评估
        zero_cross = self.evaluate_zero_cross_quality()
        
        # 获取形态识别
        patterns = self.identify_patterns()
        
        # 获取多周期协同分析
        synergy = self.analyze_multi_period_synergy()
        
        # 基础分数为50（中性）
        score = pd.Series(50, index=self._result.index)
        
        # 1. TRIX基础评分 (±20分)
        # TRIX > 0 看涨，TRIX < 0 看跌
        score += np.where(trix > 0, np.minimum(trix * 200, 20), np.maximum(trix * 200, -20))
        
        # 2. TRIX与信号线关系评分 (±15分)
        # TRIX > MATRIX 看涨，TRIX < MATRIX 看跌
        trix_vs_matrix = trix - matrix
        normalized_diff = trix_vs_matrix / trix.rolling(window=20).std().replace(0, 0.001)
        score += np.where(trix_vs_matrix > 0, 
                        np.minimum(normalized_diff * 5, 15), 
                        np.maximum(normalized_diff * 5, -15))
        
        # 3. TRIX动量评分 (±10分)
        # 动量为正看涨，动量为负看跌
        normalized_momentum = trix_momentum / trix_momentum.rolling(window=20).std().replace(0, 0.001)
        score += np.where(trix_momentum > 0, 
                        np.minimum(normalized_momentum * 3, 10), 
                        np.maximum(normalized_momentum * 3, -10))
        
        # 4. TRIX斜率评分 (±10分)
        # 斜率为正看涨，斜率为负看跌
        score += np.where(trix_slope > 0, 
                        np.minimum(trix_slope * 50, 10), 
                        np.maximum(trix_slope * 50, -10))
        
        # 5. 零轴交叉评分 (±15分)
        if not zero_cross.empty:
            # 向上交叉零轴
            upward_cross = zero_cross.get('zero_cross_up', pd.Series(False, index=score.index))
            if isinstance(upward_cross, pd.Series) and not upward_cross.empty:
                cross_quality_up = zero_cross.get('cross_quality', pd.Series(50, index=score.index))
                score.loc[upward_cross] += np.minimum((cross_quality_up.loc[upward_cross] - 50) / 10 * 15, 15)
            
            # 向下交叉零轴
            downward_cross = zero_cross.get('zero_cross_down', pd.Series(False, index=score.index))
            if isinstance(downward_cross, pd.Series) and not downward_cross.empty:
                cross_quality_down = zero_cross.get('cross_quality', pd.Series(50, index=score.index))
                score.loc[downward_cross] -= np.minimum((cross_quality_down.loc[downward_cross] - 50) / 10 * 15, 15)
        
        # 6. 背离评分 (±15分)
        if not divergence.empty:
            # 牛市背离
            bullish_div = divergence.get('bullish_divergence', pd.Series(False, index=score.index))
            if isinstance(bullish_div, pd.Series) and not bullish_div.empty:
                div_strength = divergence.get('divergence_strength', pd.Series(0.5, index=score.index))
                score.loc[bullish_div] += np.minimum(div_strength.loc[bullish_div] * 30, 15)
            
            # 熊市背离
            bearish_div = divergence.get('bearish_divergence', pd.Series(False, index=score.index))
            if isinstance(bearish_div, pd.Series) and not bearish_div.empty:
                div_strength = divergence.get('divergence_strength', pd.Series(0.5, index=score.index))
                score.loc[bearish_div] -= np.minimum(div_strength.loc[bearish_div] * 30, 15)
        
        # 7. 多周期协同评分 (±15分)
        if not synergy.empty:
            bull_synergy = synergy.get('bullish_agreement', pd.Series(False, index=score.index))
            if isinstance(bull_synergy, pd.Series) and not bull_synergy.empty:
                synergy_strength = synergy.get('synergy_strength', pd.Series(0.5, index=score.index))
                score.loc[bull_synergy] += np.minimum(synergy_strength.loc[bull_synergy] * 30, 15)
            
            bear_synergy = synergy.get('bearish_agreement', pd.Series(False, index=score.index))
            if isinstance(bear_synergy, pd.Series) and not bear_synergy.empty:
                synergy_strength = synergy.get('synergy_strength', pd.Series(0.5, index=score.index))
                score.loc[bear_synergy] -= np.minimum(synergy_strength.loc[bear_synergy] * 30, 15)
        
        # 8. 特殊形态评分 (±10分)
        if not patterns.empty:
            # 看涨形态
            for pattern in ['hook_bottom', 'bottom_reversal', 'breakout_up']:
                if pattern in patterns.columns:
                    pattern_signal = patterns[pattern]
                    if isinstance(pattern_signal, pd.Series) and not pattern_signal.empty:
                        score.loc[pattern_signal] += 10
            
            # 看跌形态
            for pattern in ['hook_top', 'top_reversal', 'breakout_down']:
                if pattern in patterns.columns:
                    pattern_signal = patterns[pattern]
                    if isinstance(pattern_signal, pd.Series) and not pattern_signal.empty:
                        score.loc[pattern_signal] -= 10
        
        # 9. 市场环境调整
        if self.market_environment == "bull_market":
            # 牛市中增强多头信号，弱化空头信号
            bull_adjustment = np.where(score > 50, (score - 50) * 0.2, (score - 50) * 0.1)
            score += bull_adjustment
        elif self.market_environment == "bear_market":
            # 熊市中增强空头信号，弱化多头信号
            bear_adjustment = np.where(score < 50, (50 - score) * 0.2, (50 - score) * 0.1)
            score -= bear_adjustment
        elif self.market_environment == "volatile_market":
            # 高波动市场需要更强的信号
            vol_adjustment = (score - 50).abs() * 0.3
            score = np.where(score > 50, 50 + vol_adjustment, 50 - vol_adjustment)
        
        # 限制分数范围在0-100之间
        score = score.clip(0, 100)
        
        return score
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算增强型TRIX指标原始评分 (0-100分)
        
        Args:
            data: 输入数据
            **kwargs: 额外参数
            
        Returns:
            pd.Series: 评分序列，取值范围0-100
        """
        # 直接使用现有的calculate_score方法
        if not self.has_result():
            self.calculate(data)
        
        return self.calculate_score()
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        Args:
            data: 输入数据
            
        Returns:
            pd.DataFrame: 包含交易信号的DataFrame
        """
        # 确保已计算TRIX
        if self._result is None:
            self.calculate(data)
            
        if self._result is None:
            return pd.DataFrame()
            
        # 获取TRIX数据
        trix = self._result['TRIX']
        
        # 计算TRIX综合评分
        score = self.calculate_score()
        
        # 识别形态
        patterns = self.identify_patterns()
        
        # 创建信号DataFrame
        signals = pd.DataFrame(index=self._result.index)
        signals['trix'] = trix
        signals['score'] = score
        
        # 生成基础信号
        signals['buy_signal'] = False
        signals['sell_signal'] = False
        signals['neutral_signal'] = True
        
        # 买入信号条件
        buy_conditions = [
            (score > 70),  # 评分高于70
            patterns.get('cross_up_zero', pd.Series(False, index=self._result.index)),  # 零轴上穿
            patterns.get('golden_cross', pd.Series(False, index=self._result.index)) & (trix > 0),  # 金叉且在零轴上方
            patterns.get('bullish_divergence', pd.Series(False, index=self._result.index)),  # 正背离
            patterns.get('multi_period_bullish_signal', pd.Series(False, index=self._result.index))  # 多周期看涨信号
        ]
        
        # 卖出信号条件
        sell_conditions = [
            (score < 30),  # 评分低于30
            patterns.get('cross_down_zero', pd.Series(False, index=self._result.index)),  # 零轴下穿
            patterns.get('death_cross', pd.Series(False, index=self._result.index)) & (trix < 0),  # 死叉且在零轴下方
            patterns.get('bearish_divergence', pd.Series(False, index=self._result.index)),  # 负背离
            patterns.get('multi_period_bearish_signal', pd.Series(False, index=self._result.index))  # 多周期看跌信号
        ]
        
        # 合并买入信号
        for condition in buy_conditions:
            signals['buy_signal'] = signals['buy_signal'] | condition
        
        # 合并卖出信号
        for condition in sell_conditions:
            signals['sell_signal'] = signals['sell_signal'] | condition
        
        # 处理买卖信号冲突
        conflict = signals['buy_signal'] & signals['sell_signal']
        if conflict.any():
            # 使用评分解决冲突
            signals.loc[conflict & (score >= 50), 'sell_signal'] = False
            signals.loc[conflict & (score < 50), 'buy_signal'] = False
        
        # 更新中性信号
        signals['neutral_signal'] = ~(signals['buy_signal'] | signals['sell_signal'])
        
        # 添加信号描述和类型
        signals['signal_type'] = ''
        signals['signal_desc'] = ''
        signals['trend'] = 0  # 默认中性
        
        # 买入信号类型和描述
        buy_signals = signals['buy_signal']
        signals.loc[buy_signals & patterns.get('cross_up_zero', False), 'signal_type'] = 'TRIX零轴上穿'
        signals.loc[buy_signals & patterns.get('cross_up_zero', False), 'signal_desc'] = 'TRIX上穿零轴，显示转入上升趋势'
        
        signals.loc[buy_signals & patterns.get('golden_cross', False), 'signal_type'] = 'TRIX金叉'
        signals.loc[buy_signals & patterns.get('golden_cross', False), 'signal_desc'] = 'TRIX上穿信号线，显示短期上升动能增强'
        
        signals.loc[buy_signals & patterns.get('bullish_divergence', False), 'signal_type'] = 'TRIX正背离'
        signals.loc[buy_signals & patterns.get('bullish_divergence', False), 'signal_desc'] = '价格创新低但TRIX未创新低，显示下跌动能减弱'
        
        signals.loc[buy_signals & patterns.get('multi_period_bullish_signal', False), 'signal_type'] = 'TRIX多周期看涨'
        signals.loc[buy_signals & patterns.get('multi_period_bullish_signal', False), 'signal_desc'] = '多周期TRIX共振发出看涨信号'
        
        # 卖出信号类型和描述
        sell_signals = signals['sell_signal']
        signals.loc[sell_signals & patterns.get('cross_down_zero', False), 'signal_type'] = 'TRIX零轴下穿'
        signals.loc[sell_signals & patterns.get('cross_down_zero', False), 'signal_desc'] = 'TRIX下穿零轴，显示转入下降趋势'
        
        signals.loc[sell_signals & patterns.get('death_cross', False), 'signal_type'] = 'TRIX死叉'
        signals.loc[sell_signals & patterns.get('death_cross', False), 'signal_desc'] = 'TRIX下穿信号线，显示短期下降动能增强'
        
        signals.loc[sell_signals & patterns.get('bearish_divergence', False), 'signal_type'] = 'TRIX负背离'
        signals.loc[sell_signals & patterns.get('bearish_divergence', False), 'signal_desc'] = '价格创新高但TRIX未创新高，显示上涨动能减弱'
        
        signals.loc[sell_signals & patterns.get('multi_period_bearish_signal', False), 'signal_type'] = 'TRIX多周期看跌'
        signals.loc[sell_signals & patterns.get('multi_period_bearish_signal', False), 'signal_desc'] = '多周期TRIX共振发出看跌信号'
        
        # 设置趋势方向
        signals['trend'] = 0  # 默认中性
        signals.loc[buy_signals, 'trend'] = 1  # 上升趋势
        signals.loc[sell_signals, 'trend'] = -1  # 下降趋势
        
        # 计算信号置信度
        signals['confidence'] = self._calculate_signal_confidence(signals, patterns)
        
        # 计算建议止损价
        if 'close' in data.columns:
            signals['stop_loss'] = self._calculate_stop_loss(data, signals)
        
        # 添加市场环境信息
        signals['market_env'] = self.market_environment
        
        return signals
    
    def _calculate_signal_confidence(self, signals: pd.DataFrame, patterns: pd.DataFrame) -> pd.Series:
        """
        计算信号置信度
        
        Args:
            signals: 信号DataFrame
            patterns: 形态DataFrame
            
        Returns:
            pd.Series: 信号置信度 (0-100)
        """
        confidence = pd.Series(50, index=signals.index)
        
        # 根据评分计算基础置信度
        score = signals['score']
        
        # 高评分对应高置信度
        confidence_from_score = np.where(score > 50, 50 + (score - 50) * 0.8, 50 - (50 - score) * 0.8)
        confidence = confidence_from_score
        
        # 增强型形态提高置信度
        for pattern, boost in [
            ('bullish_divergence', 15),
            ('bearish_divergence', 15),
            ('hidden_bullish_divergence', 10),
            ('hidden_bearish_divergence', 10),
            ('high_quality_cross_up_zero', 20),
            ('high_quality_cross_down_zero', 20),
            ('multi_period_bullish_signal', 15),
            ('multi_period_bearish_signal', 15),
            ('strong_bullish_consensus', 10),
            ('strong_bearish_consensus', 10)
        ]:
            if pattern in patterns.columns:
                confidence = np.where(patterns[pattern], np.minimum(100, confidence + boost), confidence)
        
        # 零轴位置对确信度的影响
        trix_values = self._result['TRIX']
        
        # TRIX远离零轴时信号更可靠
        confidence += np.where(abs(trix_values) > 0.5, 5, 0)
        confidence += np.where(abs(trix_values) > 1.0, 5, 0)
        
        return confidence
    
    def _calculate_stop_loss(self, data: pd.DataFrame, signals: pd.DataFrame) -> pd.Series:
        """
        计算建议止损价
        
        Args:
            data: 价格数据
            signals: 信号DataFrame
            
        Returns:
            pd.Series: 建议止损价
        """
        stop_loss = pd.Series(np.nan, index=signals.index)
        
        # 获取价格数据
        close = data['close']
        
        if 'low' in data.columns:
            low = data['low']
        else:
            low = close
        
        # 计算ATR (如果可能)
        atr = None
        if 'high' in data.columns and 'low' in data.columns:
            high = data['high']
            atr = self.atr(high, low, close, 14)
        
        # 买入信号的止损
        for i in range(len(signals)):
            if signals['buy_signal'].iloc[i]:
                current_close = close.iloc[i]
                
                if atr is not None and i < len(atr) and not np.isnan(atr.iloc[i]):
                    # 使用ATR计算动态止损
                    atr_value = atr.iloc[i]
                    confidence = signals['confidence'].iloc[i]
                    
                    # 根据信号置信度调整ATR倍数
                    if confidence >= 80:
                        atr_multiplier = 2.0  # 高置信度，较宽止损
                    elif confidence >= 60:
                        atr_multiplier = 1.5  # 中等置信度，中等止损
                    else:
                        atr_multiplier = 1.0  # 低置信度，紧止损
                    
                    stop_loss.iloc[i] = current_close - (atr_value * atr_multiplier)
                else:
                    # 使用最近低点作为止损
                    if i >= 5:
                        recent_low = low.iloc[i-5:i+1].min()
                        stop_loss.iloc[i] = recent_low * 0.99  # 微调1%
        
        # 卖出信号的止损 (反向操作的止损位)
        for i in range(len(signals)):
            if signals['sell_signal'].iloc[i]:
                current_close = close.iloc[i]
                
                if atr is not None and i < len(atr) and not np.isnan(atr.iloc[i]):
                    # 使用ATR计算动态止损
                    atr_value = atr.iloc[i]
                    confidence = signals['confidence'].iloc[i]
                    
                    # 根据信号置信度调整ATR倍数
                    if confidence >= 80:
                        atr_multiplier = 2.0
                    elif confidence >= 60:
                        atr_multiplier = 1.5
                    else:
                        atr_multiplier = 1.0
                    
                    stop_loss.iloc[i] = current_close + (atr_value * atr_multiplier)
                else:
                    # 使用最近高点作为止损
                    if 'high' in data.columns and i >= 5:
                        high = data['high']
                        recent_high = high.iloc[i-5:i+1].max()
                        stop_loss.iloc[i] = recent_high * 1.01  # 微调1%
        
        return stop_loss 