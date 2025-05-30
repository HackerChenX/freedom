"""
增强型威廉指标(Williams %R)模块

威廉指标(WR)是一个典型的超买超卖类指标，本模块实现了增强型WR指标，
增加了动态阈值调整、多周期协同分析、形态识别等功能，使WR指标更具适应性和准确性。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple

from indicators.base_indicator import BaseIndicator, MarketEnvironment, SignalStrength
from indicators.wr import WR
from utils.logger import get_logger
from utils.technical_utils import find_peaks_and_troughs

logger = get_logger(__name__)


class EnhancedWR(WR):
    """
    增强型威廉指标(Williams %R)
    
    具有以下增强特性:
    1. 动态阈值调整：根据市场波动率和趋势强度动态调整超买超卖阈值
    2. 背离识别增强：改进的算法用于识别WR与价格之间的背离关系
    3. 多周期WR协同：结合不同周期的WR指标，提高信号可靠性
    4. 震荡带识别：构建WR震荡带分析系统，预测可能的突破方向
    5. 市场环境自适应：在不同市场环境下动态调整评分标准
    """
    
    def __init__(self, 
                 period: int = 14, 
                 secondary_period: int = 28,
                 multi_periods: List[int] = None,
                 adaptive_threshold: bool = True,
                 volatility_lookback: int = 20):
        """
        初始化增强型威廉指标
        
        Args:
            period: 主要周期，默认为14
            secondary_period: 次要周期，默认为28
            multi_periods: 多周期分析参数，默认为[6, 14, 28, 56]
            adaptive_threshold: 是否启用自适应阈值，默认为True
            volatility_lookback: 波动率计算回溯期，默认为20
        """
        super().__init__(period=period)
        self.name = "EnhancedWR"
        self.description = "增强型威廉指标，优化超买超卖判断，增加多周期协同分析和市场环境感知"
        self.secondary_period = secondary_period
        self.multi_periods = multi_periods or [6, 14, 28, 56]
        self.adaptive_threshold = adaptive_threshold
        self.volatility_lookback = volatility_lookback
        self.indicator_type = "oscillator"  # 指标类型：震荡类
        self.market_environment = "normal"
        
        # 动态阈值
        self.overbought_threshold = -20
        self.oversold_threshold = -80
        self.extreme_overbought_threshold = -10
        self.extreme_oversold_threshold = -90
        
        # 内部变量
        self._secondary_wr = None
        self._multi_period_wr = {}
        self._price_data = None
        
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
        计算增强型威廉指标
        
        Args:
            data: 输入数据，包含OHLC数据
            
        Returns:
            pd.DataFrame: 计算结果，包含WR及其相关指标
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close", "high", "low"])
        
        # 保存价格数据用于后续分析
        self._price_data = data['close'].copy()
        
        # 调用父类方法计算基础WR
        result = super().calculate(data)
        
        # 计算次要周期WR
        highest_high_secondary = data['high'].rolling(window=self.secondary_period).max()
        lowest_low_secondary = data['low'].rolling(window=self.secondary_period).min()
        result['wr_secondary'] = -100 * (highest_high_secondary - data['close']) / (highest_high_secondary - lowest_low_secondary)
        self._secondary_wr = result['wr_secondary']
        
        # 计算多周期WR
        for period in self.multi_periods:
            if period != self.period and period != self.secondary_period:
                highest_high_multi = data['high'].rolling(window=period).max()
                lowest_low_multi = data['low'].rolling(window=period).min()
                result[f'wr_{period}'] = -100 * (highest_high_multi - data['close']) / (highest_high_multi - lowest_low_multi)
                self._multi_period_wr[period] = result[f'wr_{period}']
        
        # 计算WR波动率
        result['wr_volatility'] = result['wr'].rolling(window=self.volatility_lookback).std()
        
        # 计算WR变化率
        result['wr_rate_of_change'] = result['wr'].diff(3)
        
        # 计算WR平均值
        result['wr_mean'] = result['wr'].rolling(window=self.volatility_lookback).mean()
        
        # 计算WR动量
        result['wr_momentum'] = result['wr'] - result['wr'].shift(5)
        
        # 动态调整超买超卖阈值
        if self.adaptive_threshold:
            self._adjust_thresholds_by_market(result)
        
        # 保存结果
        self._result = result
        
        return result
        
    def _adjust_thresholds_by_market(self, data: pd.DataFrame) -> None:
        """
        根据市场环境和波动率动态调整超买超卖阈值
        
        Args:
            data: 包含WR和价格数据的DataFrame
        """
        # 获取最新的WR波动率
        if 'wr_volatility' in data.columns and not data['wr_volatility'].isna().all():
            latest_volatility = data['wr_volatility'].iloc[-1]
            historical_volatility = data['wr_volatility'].mean()
            volatility_ratio = latest_volatility / historical_volatility if historical_volatility > 0 else 1
            
            # 根据波动率调整阈值
            if volatility_ratio > 1.5:  # 高波动环境
                # 扩大超买超卖区间
                self.overbought_threshold = -15
                self.oversold_threshold = -85
                self.extreme_overbought_threshold = -5
                self.extreme_oversold_threshold = -95
            elif volatility_ratio < 0.7:  # 低波动环境
                # 收窄超买超卖区间
                self.overbought_threshold = -25
                self.oversold_threshold = -75
                self.extreme_overbought_threshold = -15
                self.extreme_oversold_threshold = -85
            else:  # 正常波动环境
                # 使用默认阈值
                self.overbought_threshold = -20
                self.oversold_threshold = -80
                self.extreme_overbought_threshold = -10
                self.extreme_oversold_threshold = -90
        
        # 根据市场环境进一步调整阈值
        if self.market_environment == 'bull_market':
            # 牛市中提高超买阈值，减少卖出信号
            self.overbought_threshold -= 5
            self.extreme_overbought_threshold -= 5
        elif self.market_environment == 'bear_market':
            # 熊市中降低超卖阈值，减少买入信号
            self.oversold_threshold -= 5
            self.extreme_oversold_threshold -= 5
        elif self.market_environment == 'volatile_market':
            # 高波动市场需要更极端的信号
            self.overbought_threshold -= 5
            self.oversold_threshold -= 5
            self.extreme_overbought_threshold -= 5
            self.extreme_oversold_threshold -= 5 

    def detect_enhanced_divergence(self) -> pd.DataFrame:
        """
        增强型背离识别
        
        Returns:
            pd.DataFrame: 包含背离分析结果的DataFrame
        """
        if self._result is None or self._price_data is None:
            return pd.DataFrame()
            
        # 获取WR和价格数据
        wr = self._result['wr']
        price = self._price_data
        
        # 创建结果DataFrame
        divergence = pd.DataFrame(index=price.index)
        divergence['bullish_divergence'] = False
        divergence['bearish_divergence'] = False
        divergence['hidden_bullish_divergence'] = False
        divergence['hidden_bearish_divergence'] = False
        divergence['divergence_strength'] = 0.0
        
        # 查找价格和WR的高点和低点
        price_peaks = find_peaks_and_troughs(price, window=10, peak_type='peak')
        price_troughs = find_peaks_and_troughs(price, window=10, peak_type='trough')
        # 注意WR是反向指标，低点对应价格高点，高点对应价格低点
        wr_peaks = find_peaks_and_troughs(wr, window=10, peak_type='peak')  # WR高点(对应价格低点)
        wr_troughs = find_peaks_and_troughs(wr, window=10, peak_type='trough')  # WR低点(对应价格高点)
        
        # 最小背离长度(防止检测到太短的背离)
        min_divergence_length = 5
        # 最大背离长度(防止检测到太长的背离)
        max_divergence_length = 30
        
        # 常规看涨背离：价格创新低但WR未创新高
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
                # 查找对应的WR值（WR高点对应价格低点）
                current_wr_peak = None
                prev_wr_peak = None
                
                # 在价格低点附近查找WR高点
                for wp in wr_peaks:
                    if abs(wp - current_trough_idx) <= 3:
                        current_wr_peak = wp
                    if abs(wp - prev_trough_idx) <= 3:
                        prev_wr_peak = wp
                
                # 如果找到了对应的WR高点
                if current_wr_peak is not None and prev_wr_peak is not None:
                    # WR未创新高（注意WR是负值，所以比较方向相反）
                    if wr.iloc[current_wr_peak] < wr.iloc[prev_wr_peak]:
                        # 计算背离强度
                        price_change = (price.iloc[current_trough_idx] / price.iloc[prev_trough_idx]) - 1
                        wr_change = (wr.iloc[current_wr_peak] / wr.iloc[prev_wr_peak]) - 1
                        strength = abs(price_change - wr_change) / max(abs(price_change), abs(wr_change))
                        
                        # 记录背离
                        divergence.iloc[current_trough_idx:current_trough_idx+5, 0] = True  # bullish_divergence
                        divergence.iloc[current_trough_idx:current_trough_idx+5, 4] = strength  # divergence_strength
        
        # 常规看跌背离：价格创新高但WR未创新低
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
                # 查找对应的WR值（WR低点对应价格高点）
                current_wr_trough = None
                prev_wr_trough = None
                
                # 在价格高点附近查找WR低点
                for wt in wr_troughs:
                    if abs(wt - current_peak_idx) <= 3:
                        current_wr_trough = wt
                    if abs(wt - prev_peak_idx) <= 3:
                        prev_wr_trough = wt
                
                # 如果找到了对应的WR低点
                if current_wr_trough is not None and prev_wr_trough is not None:
                    # WR未创新低（注意WR是负值，所以比较方向相反）
                    if wr.iloc[current_wr_trough] > wr.iloc[prev_wr_trough]:
                        # 计算背离强度
                        price_change = (price.iloc[current_peak_idx] / price.iloc[prev_peak_idx]) - 1
                        wr_change = (wr.iloc[current_wr_trough] / wr.iloc[prev_wr_trough]) - 1
                        strength = abs(price_change - wr_change) / max(abs(price_change), abs(wr_change))
                        
                        # 记录背离
                        divergence.iloc[current_peak_idx:current_peak_idx+5, 1] = True  # bearish_divergence
                        divergence.iloc[current_peak_idx:current_peak_idx+5, 4] = strength  # divergence_strength
        
        # 隐藏背离（趋势确认型背离）也可以添加，但逻辑类似，此处略过
        
        return divergence
    
    def analyze_multi_period_synergy(self) -> pd.DataFrame:
        """
        多周期WR协同分析
        
        Returns:
            pd.DataFrame: 包含多周期协同分析结果的DataFrame
        """
        if self._result is None:
            return pd.DataFrame()
            
        # 创建结果DataFrame
        synergy = pd.DataFrame(index=self._result.index)
        
        # 获取所有周期的WR
        primary_wr = self._result['wr']
        secondary_wr = self._secondary_wr
        
        # 分析多周期超买超卖状态
        synergy['all_overbought'] = primary_wr > self.overbought_threshold
        synergy['all_oversold'] = primary_wr < self.oversold_threshold
        synergy['primary_overbought'] = primary_wr > self.overbought_threshold
        synergy['primary_oversold'] = primary_wr < self.oversold_threshold
        synergy['secondary_overbought'] = secondary_wr > self.overbought_threshold
        synergy['secondary_oversold'] = secondary_wr < self.oversold_threshold
        
        # 计算多周期一致性
        overbought_count = synergy['primary_overbought'].astype(int)
        oversold_count = synergy['primary_oversold'].astype(int)
        
        # 添加次要周期的状态
        overbought_count += synergy['secondary_overbought'].astype(int)
        oversold_count += synergy['secondary_oversold'].astype(int)
        
        # 添加其他周期的状态
        for period, wr_series in self._multi_period_wr.items():
            period_overbought = wr_series > self.overbought_threshold
            period_oversold = wr_series < self.oversold_threshold
            
            synergy[f'wr{period}_overbought'] = period_overbought
            synergy[f'wr{period}_oversold'] = period_oversold
            
            overbought_count += period_overbought.astype(int)
            oversold_count += period_oversold.astype(int)
        
        # 计算总周期数（主周期+次要周期+其他周期）
        total_periods = 2 + len(self._multi_period_wr)
        
        # 计算超买超卖比例
        synergy['overbought_ratio'] = overbought_count / total_periods
        synergy['oversold_ratio'] = oversold_count / total_periods
        
        # 计算一致性得分（0-100）
        synergy['consensus_score'] = 50 + (synergy['overbought_ratio'] - synergy['oversold_ratio']) * 100
        
        # 多周期突破信号
        synergy['multi_period_bullish_breakout'] = False
        synergy['multi_period_bearish_breakout'] = False
        
        # 检测多周期协同突破（当多个周期的WR同时穿越超买超卖线）
        primary_cross_up = self.crossover(primary_wr, self.oversold_threshold)
        primary_cross_down = self.crossunder(primary_wr, self.overbought_threshold)
        secondary_cross_up = self.crossover(secondary_wr, self.oversold_threshold)
        secondary_cross_down = self.crossunder(secondary_wr, self.overbought_threshold)
        
        # 主周期和次要周期同时突破
        synergy['multi_period_bullish_breakout'] = primary_cross_up & secondary_cross_up
        synergy['multi_period_bearish_breakout'] = primary_cross_down & secondary_cross_down
        
        return synergy
    
    def identify_oscillation_band(self) -> pd.DataFrame:
        """
        WR震荡带分析
        
        Returns:
            pd.DataFrame: 包含震荡带分析结果的DataFrame
        """
        if self._result is None:
            return pd.DataFrame()
            
        # 获取WR数据
        wr = self._result['wr']
        
        # 创建结果DataFrame
        oscillation = pd.DataFrame(index=self._result.index)
        
        # 计算WR的移动平均和标准差
        wr_ma = wr.rolling(window=20).mean()
        wr_std = wr.rolling(window=20).std()
        
        # 定义震荡带
        oscillation['upper_band'] = wr_ma + wr_std * 2
        oscillation['lower_band'] = wr_ma - wr_std * 2
        oscillation['middle_band'] = wr_ma
        
        # 计算WR与震荡带的关系
        oscillation['above_upper'] = wr > oscillation['upper_band']
        oscillation['below_lower'] = wr < oscillation['lower_band']
        oscillation['in_band'] = ~(oscillation['above_upper'] | oscillation['below_lower'])
        
        # 检测WR从震荡带突破
        oscillation['breakout_up'] = (wr > oscillation['upper_band']) & (wr.shift(1) <= oscillation['upper_band'].shift(1))
        oscillation['breakout_down'] = (wr < oscillation['lower_band']) & (wr.shift(1) >= oscillation['lower_band'].shift(1))
        
        # 检测震荡带宽度变化
        band_width = oscillation['upper_band'] - oscillation['lower_band']
        oscillation['band_width'] = band_width
        oscillation['band_width_expanding'] = band_width > band_width.shift(1)
        oscillation['band_width_contracting'] = band_width < band_width.shift(1)
        
        # 检测极窄震荡带（潜在爆发区域）
        avg_band_width = band_width.rolling(window=60).mean()
        oscillation['extremely_narrow_band'] = band_width < (avg_band_width * 0.5)
        
        return oscillation
    
    def identify_patterns(self) -> pd.DataFrame:
        """
        识别WR形态
        
        Returns:
            pd.DataFrame: 包含形态识别结果的DataFrame
        """
        if self._result is None:
            return pd.DataFrame()
            
        # 获取WR数据
        wr = self._result['wr']
        
        # 创建结果DataFrame
        patterns = pd.DataFrame(index=self._result.index)
        
        # 基础超买超卖形态
        patterns['overbought'] = wr > self.overbought_threshold
        patterns['oversold'] = wr < self.oversold_threshold
        patterns['extreme_overbought'] = wr > self.extreme_overbought_threshold
        patterns['extreme_oversold'] = wr < self.extreme_oversold_threshold
        
        # 计算WR穿越信号
        patterns['cross_above_oversold'] = self.crossover(wr, self.oversold_threshold)
        patterns['cross_below_overbought'] = self.crossunder(wr, self.overbought_threshold)
        patterns['cross_above_midline'] = self.crossover(wr, -50)
        patterns['cross_below_midline'] = self.crossunder(wr, -50)
        
        # W底形态识别
        patterns['w_bottom'] = self._detect_w_bottom(wr)
        
        # M顶形态识别
        patterns['m_top'] = self._detect_m_top(wr)
        
        # 计算多周期协同状态
        if hasattr(self, '_multi_period_wr') and self._multi_period_wr:
            synergy = self.analyze_multi_period_synergy()
            patterns['multi_period_consensus_bullish'] = synergy['consensus_score'] > 70
            patterns['multi_period_consensus_bearish'] = synergy['consensus_score'] < 30
            patterns['multi_period_bullish_breakout'] = synergy['multi_period_bullish_breakout']
            patterns['multi_period_bearish_breakout'] = synergy['multi_period_bearish_breakout']
        
        # 获取背离分析结果
        divergence = self.detect_enhanced_divergence()
        if not divergence.empty:
            patterns['bullish_divergence'] = divergence['bullish_divergence']
            patterns['bearish_divergence'] = divergence['bearish_divergence']
        
        # 震荡带分析
        oscillation = self.identify_oscillation_band()
        if not oscillation.empty:
            patterns['breakout_up'] = oscillation['breakout_up']
            patterns['breakout_down'] = oscillation['breakout_down']
            patterns['extremely_narrow_band'] = oscillation['extremely_narrow_band']
        
        # 钝化形态（WR在超买/超卖区域徘徊）
        patterns['overbought_stagnation'] = self._detect_stagnation(wr, threshold=self.overbought_threshold, periods=5, direction='high')
        patterns['oversold_stagnation'] = self._detect_stagnation(wr, threshold=self.oversold_threshold, periods=5, direction='low')
        
        return patterns
    
    def _detect_w_bottom(self, wr: pd.Series) -> pd.Series:
        """
        识别W底形态
        
        Args:
            wr: WR序列
            
        Returns:
            pd.Series: W底形态识别结果
        """
        w_bottom = pd.Series(False, index=wr.index)
        
        if len(wr) < 20:
            return w_bottom
        
        # W底特征：两个低点，中间有一个高点，第二个低点高于第一个低点
        for i in range(15, len(wr)):
            # 获取最近15个周期的数据
            window = wr.iloc[i-15:i+1]
            
            # 找到局部最低点
            min_indices = window.iloc[1:-1].nsmallest(2).index
            if len(min_indices) < 2:
                continue
                
            # 确保两个低点之间有高点
            low1_idx = min_indices[0]
            low2_idx = min_indices[1]
            
            if low1_idx >= low2_idx:
                continue
                
            # 找到两个低点之间的最高点
            mid_high = window.loc[low1_idx:low2_idx].max()
            mid_high_idx = window.loc[low1_idx:low2_idx].idxmax()
            
            # W底条件：
            # 1. 两个低点都在超卖区域
            # 2. 中间高点显著高于两个低点
            # 3. 第二个低点高于第一个低点（背离）
            if (window.loc[low1_idx] < self.oversold_threshold and 
                window.loc[low2_idx] < self.oversold_threshold and
                mid_high > self.oversold_threshold + 10 and
                window.loc[low2_idx] > window.loc[low1_idx] and
                mid_high_idx > low1_idx and mid_high_idx < low2_idx):
                
                w_bottom.loc[i] = True
        
        return w_bottom
    
    def _detect_m_top(self, wr: pd.Series) -> pd.Series:
        """
        识别M顶形态
        
        Args:
            wr: WR序列
            
        Returns:
            pd.Series: M顶形态识别结果
        """
        m_top = pd.Series(False, index=wr.index)
        
        if len(wr) < 20:
            return m_top
        
        # M顶特征：两个高点，中间有一个低点，第二个高点低于第一个高点
        for i in range(15, len(wr)):
            # 获取最近15个周期的数据
            window = wr.iloc[i-15:i+1]
            
            # 找到局部最高点
            max_indices = window.iloc[1:-1].nlargest(2).index
            if len(max_indices) < 2:
                continue
                
            # 确保两个高点之间有低点
            high1_idx = max_indices[0]
            high2_idx = max_indices[1]
            
            if high1_idx >= high2_idx:
                continue
                
            # 找到两个高点之间的最低点
            mid_low = window.loc[high1_idx:high2_idx].min()
            mid_low_idx = window.loc[high1_idx:high2_idx].idxmin()
            
            # M顶条件：
            # 1. 两个高点都在超买区域
            # 2. 中间低点显著低于两个高点
            # 3. 第二个高点低于第一个高点（背离）
            if (window.loc[high1_idx] > self.overbought_threshold and 
                window.loc[high2_idx] > self.overbought_threshold and
                mid_low < self.overbought_threshold - 10 and
                window.loc[high2_idx] < window.loc[high1_idx] and
                mid_low_idx > high1_idx and mid_low_idx < high2_idx):
                
                m_top.loc[i] = True
        
        return m_top
    
    def _detect_stagnation(self, wr: pd.Series, threshold: float, periods: int, direction: str) -> pd.Series:
        """
        检测WR在超买/超卖区域的钝化形态
        
        Args:
            wr: WR序列
            threshold: 超买/超卖阈值
            periods: 持续周期数
            direction: 方向，'high'表示超买区域，'low'表示超卖区域
            
        Returns:
            pd.Series: 钝化形态识别结果
        """
        stagnation = pd.Series(False, index=wr.index)
        
        if len(wr) < periods:
            return stagnation
        
        # 检查连续periods个周期WR都在超买/超卖区域
        for i in range(periods, len(wr)):
            window = wr.iloc[i-periods+1:i+1]
            
            if direction == 'high':
                # 超买区域钝化
                if (window > threshold).all() and (window.max() - window.min()) < 10:
                    stagnation.iloc[i] = True
            else:
                # 超卖区域钝化
                if (window < threshold).all() and (window.max() - window.min()) < 10:
                    stagnation.iloc[i] = True
        
        return stagnation
    
    def calculate_score(self, data: pd.DataFrame = None) -> pd.Series:
        """
        计算WR综合评分 (0-100)
        
        Args:
            data: 输入数据，如果未提供则使用上次计算结果
            
        Returns:
            pd.Series: 评分 (0-100，50为中性)
        """
        # 确保已计算WR
        if self._result is None and data is not None:
            self.calculate(data)
            
        if self._result is None:
            return pd.Series()
            
        # 获取WR数据
        wr = self._result['wr']
        
        # 基础分数为50（中性）
        score = pd.Series(50, index=self._result.index)
        
        # 1. 超买超卖区评分 (±20分)
        # WR < 超卖阈值（通常为-80），加分
        oversold_score = np.where(wr < self.oversold_threshold, 
                                  20 * np.minimum(1, (self.oversold_threshold - wr) / 20), 
                                  0)
        # WR > 超买阈值（通常为-20），减分
        overbought_score = np.where(wr > self.overbought_threshold, 
                                    20 * np.minimum(1, (wr - self.overbought_threshold) / 20), 
                                    0)
        
        score = score + oversold_score - overbought_score
        
        # 2. 极端超买超卖评分 (±10分额外加分)
        # 极度超卖，额外加分
        extreme_oversold_score = np.where(wr < self.extreme_oversold_threshold, 
                                         10 * np.minimum(1, (self.extreme_oversold_threshold - wr) / 10), 
                                         0)
        # 极度超买，额外减分
        extreme_overbought_score = np.where(wr > self.extreme_overbought_threshold, 
                                           10 * np.minimum(1, (wr - self.extreme_overbought_threshold) / 10), 
                                           0)
        
        score = score + extreme_oversold_score - extreme_overbought_score
        
        # 3. WR动量评分 (±10分)
        if 'wr_momentum' in self._result:
            wr_momentum = self._result['wr_momentum']
            # WR向上动量，加分
            score += np.where(wr_momentum > 0, np.minimum(10, wr_momentum), 0)
            # WR向下动量，减分
            score -= np.where(wr_momentum < 0, np.minimum(10, -wr_momentum), 0)
        
        # 4. 多周期协同评分 (±15分)
        if hasattr(self, '_multi_period_wr') and self._multi_period_wr:
            synergy = self.analyze_multi_period_synergy()
            if 'consensus_score' in synergy:
                # 将多周期共识得分(0-100)映射到±15分
                consensus_score = synergy['consensus_score']
                multi_period_score = (consensus_score - 50) * 0.3  # 0.3 = 15/50
                score += multi_period_score
        
        # 5. 形态识别评分 (±25分)
        patterns = self.identify_patterns()
        
        # 看涨形态
        if 'bullish_divergence' in patterns:
            # 正背离+25分
            score += patterns['bullish_divergence'] * 25
        
        if 'w_bottom' in patterns:
            # W底形态+20分
            score += patterns['w_bottom'] * 20
        
        if 'cross_above_oversold' in patterns:
            # WR上穿超卖线+15分
            score += patterns['cross_above_oversold'] * 15
        
        # 看跌形态
        if 'bearish_divergence' in patterns:
            # 负背离-25分
            score -= patterns['bearish_divergence'] * 25
        
        if 'm_top' in patterns:
            # M顶形态-20分
            score -= patterns['m_top'] * 20
        
        if 'cross_below_overbought' in patterns:
            # WR下穿超买线-15分
            score -= patterns['cross_below_overbought'] * 15
        
        # 6. 震荡带分析评分 (±10分)
        oscillation = self.identify_oscillation_band()
        if not oscillation.empty:
            if 'breakout_up' in oscillation:
                # 向上突破震荡带+10分
                score += oscillation['breakout_up'] * 10
            
            if 'breakout_down' in oscillation:
                # 向下突破震荡带-10分
                score -= oscillation['breakout_down'] * 10
                
            if 'extremely_narrow_band' in oscillation:
                # 极窄震荡带表示潜在爆发，根据WR当前位置调整分数
                narrow_band_score = np.where(wr > -50, 5, -5)  # WR > -50看涨，否则看跌
                score += oscillation['extremely_narrow_band'] * narrow_band_score
        
        # 7. 市场环境适应评分调整
        if self.market_environment == 'bull_market':
            # 牛市中提高看涨信号权重，降低看跌信号权重
            bull_adjustment = np.where(score > 50, (score - 50) * 0.2, (score - 50) * 0.1)
            score += bull_adjustment
        elif self.market_environment == 'bear_market':
            # 熊市中提高看跌信号权重，降低看涨信号权重
            bear_adjustment = np.where(score < 50, (50 - score) * 0.2, (50 - score) * 0.1)
            score -= bear_adjustment
        
        # 限制得分范围在0-100之间
        return np.clip(score, 0, 100)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        Args:
            data: 输入数据
            
        Returns:
            pd.DataFrame: 包含交易信号的DataFrame
        """
        # 确保已计算WR
        if self._result is None:
            self.calculate(data)
            
        if self._result is None:
            return pd.DataFrame()
            
        # 获取WR数据
        wr = self._result['wr']
        
        # 计算WR综合评分
        score = self.calculate_score()
        
        # 识别形态
        patterns = self.identify_patterns()
        
        # 创建信号DataFrame
        signals = pd.DataFrame(index=self._result.index)
        signals['wr'] = wr
        signals['score'] = score
        
        # 生成基础信号
        signals['buy_signal'] = (
            (score > 70) |  # 评分高于70
            patterns.get('cross_above_oversold', False) |  # WR上穿超卖线
            patterns.get('bullish_divergence', False) |  # 正背离
            patterns.get('w_bottom', False)  # W底形态
        )
        
        signals['sell_signal'] = (
            (score < 30) |  # 评分低于30
            patterns.get('cross_below_overbought', False) |  # WR下穿超买线
            patterns.get('bearish_divergence', False) |  # 负背离
            patterns.get('m_top', False)  # M顶形态
        )
        
        # 应用多周期协同确认
        if hasattr(self, '_multi_period_wr') and self._multi_period_wr:
            synergy = self.analyze_multi_period_synergy()
            
            # 多周期共识强烈看涨/看跌时增强信号
            if 'consensus_score' in synergy:
                strong_bullish_consensus = synergy['consensus_score'] > 80
                strong_bearish_consensus = synergy['consensus_score'] < 20
                
                # 增强买入信号
                signals['buy_signal'] = signals['buy_signal'] | (
                    (score > 65) & strong_bullish_consensus
                )
                
                # 增强卖出信号
                signals['sell_signal'] = signals['sell_signal'] | (
                    (score < 35) & strong_bearish_consensus
                )
        
        # 添加信号描述和类型
        signals['signal_type'] = ''
        signals['signal_desc'] = ''
        
        # 买入信号类型和描述
        buy_signals = signals['buy_signal']
        signals.loc[buy_signals & patterns.get('cross_above_oversold', False), 'signal_type'] = 'WR超卖反弹'
        signals.loc[buy_signals & patterns.get('cross_above_oversold', False), 'signal_desc'] = 'WR上穿超卖线，显示超卖反弹信号'
        
        signals.loc[buy_signals & patterns.get('bullish_divergence', False), 'signal_type'] = 'WR正背离'
        signals.loc[buy_signals & patterns.get('bullish_divergence', False), 'signal_desc'] = '价格创新低但WR未创新高，显示潜在反转信号'
        
        signals.loc[buy_signals & patterns.get('w_bottom', False), 'signal_type'] = 'WR-W底形态'
        signals.loc[buy_signals & patterns.get('w_bottom', False), 'signal_desc'] = 'WR形成W底形态，显示强势反转信号'
        
        # 卖出信号类型和描述
        sell_signals = signals['sell_signal']
        signals.loc[sell_signals & patterns.get('cross_below_overbought', False), 'signal_type'] = 'WR超买回落'
        signals.loc[sell_signals & patterns.get('cross_below_overbought', False), 'signal_desc'] = 'WR下穿超买线，显示超买回落信号'
        
        signals.loc[sell_signals & patterns.get('bearish_divergence', False), 'signal_type'] = 'WR负背离'
        signals.loc[sell_signals & patterns.get('bearish_divergence', False), 'signal_desc'] = '价格创新高但WR未创新低，显示潜在顶部信号'
        
        signals.loc[sell_signals & patterns.get('m_top', False), 'signal_type'] = 'WR-M顶形态'
        signals.loc[sell_signals & patterns.get('m_top', False), 'signal_desc'] = 'WR形成M顶形态，显示强势顶部信号'
        
        # 添加其他形态信号
        for pattern_name in patterns.columns:
            if pattern_name not in ['bullish_divergence', 'bearish_divergence', 'w_bottom', 'm_top', 
                                   'cross_above_oversold', 'cross_below_overbought']:
                pattern_signal = patterns[pattern_name]
                if pattern_signal.any():
                    # 只对未有信号类型的记录赋值
                    mask = (pattern_signal) & (signals['signal_type'] == '')
                    signals.loc[mask, 'signal_type'] = f'WR-{pattern_name}'
                    signals.loc[mask, 'signal_desc'] = f'WR形成{pattern_name}形态'
        
        # 计算信号置信度
        signals['confidence'] = self._calculate_signal_confidence(signals, patterns)
        
        # 计算建议止损价
        if 'close' in data.columns:
            signals['stop_loss'] = self._calculate_stop_loss(data, signals)
        
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
            ('w_bottom', 20),
            ('m_top', 20),
            ('multi_period_consensus_bullish', 10),
            ('multi_period_consensus_bearish', 10),
            ('multi_period_bullish_breakout', 15),
            ('multi_period_bearish_breakout', 15)
        ]:
            if pattern in patterns.columns:
                confidence = np.where(patterns[pattern], np.minimum(100, confidence + boost), confidence)
        
        # 极端超买超卖提高置信度
        if 'extreme_oversold' in patterns.columns:
            confidence = np.where(patterns['extreme_oversold'], np.minimum(100, confidence + 10), confidence)
            
        if 'extreme_overbought' in patterns.columns:
            confidence = np.where(patterns['extreme_overbought'], np.minimum(100, confidence + 10), confidence)
        
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