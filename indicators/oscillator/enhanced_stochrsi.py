"""
增强型随机相对强弱指标(STOCHRSI)模块

实现增强型STOCHRSI指标计算，提供动态阈值、多周期协同分析、形态识别等功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple

from indicators.base_indicator import BaseIndicator, MarketEnvironment, SignalStrength
from indicators.stochrsi import STOCHRSI
from utils.logger import get_logger
from utils.technical_utils import find_peaks_and_troughs

logger = get_logger(__name__)


class EnhancedSTOCHRSI(STOCHRSI):
    """
    增强型随机相对强弱指标(STOCHRSI)
    
    具有以下增强特性:
    1. 动态阈值调整：根据市场波动率和趋势强度动态调整超买超卖阈值
    2. 多周期分析框架：结合不同周期的STOCHRSI指标增强信号可靠性
    3. 背离检测算法：检测STOCHRSI与价格之间的背离关系
    4. K和D线交叉质量评估：评估交叉信号的可靠性
    5. 形态识别系统：识别W底、M顶等特定形态
    6. 市场环境自适应：根据市场环境动态调整评分标准
    """
    
    def __init__(self, 
                 n: int = 14, 
                 m: int = 3, 
                 p: int = 3,
                 secondary_n: int = 28,
                 multi_periods: List[int] = None,
                 adaptive_threshold: bool = True,
                 volatility_lookback: int = 20):
        """
        初始化增强型STOCHRSI指标
        
        Args:
            n: RSI周期，默认为14
            m: K值周期，默认为3
            p: D值周期，默认为3
            secondary_n: 次要RSI周期，默认为28
            multi_periods: 多周期分析参数，默认为[7, 14, 28, 56]
            adaptive_threshold: 是否启用自适应阈值，默认为True
            volatility_lookback: 波动率计算回溯期，默认为20
        """
        super().__init__(n=n, m=m, p=p)
        self.name = "EnhancedSTOCHRSI"
        self.description = "增强型随机相对强弱指标，优化参数自适应性，增加多周期协同分析和市场环境感知"
        self.secondary_n = secondary_n
        self.multi_periods = multi_periods or [7, 14, 28, 56]
        self.adaptive_threshold = adaptive_threshold
        self.volatility_lookback = volatility_lookback
        self.indicator_type = "oscillator"  # 指标类型：震荡类
        self.market_environment = "normal"
        
        # 动态阈值
        self.overbought_threshold = 80
        self.oversold_threshold = 20
        self.extreme_overbought_threshold = 90
        self.extreme_oversold_threshold = 10
        
        # 内部变量
        self._secondary_stochrsi = None
        self._multi_period_stochrsi = {}
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
        计算增强型STOCHRSI指标
        
        Args:
            data: 输入数据，包含OHLC数据
            
        Returns:
            pd.DataFrame: 计算结果，包含STOCHRSI及其相关指标
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close"])
        
        # 保存价格数据用于后续分析
        self._price_data = data['close'].copy()
        
        # 如果启用自适应阈值，则调整阈值
        if self.adaptive_threshold:
            self._adjust_thresholds_by_market(data)
        
        # 调用父类方法计算基础STOCHRSI
        result = super().calculate(data, self.n, self.m, self.p)
        
        # 计算次要周期STOCHRSI
        secondary_stochrsi = STOCHRSI(n=self.secondary_n, m=self.m, p=self.p)
        secondary_result = secondary_stochrsi.calculate(data)
        result['STOCHRSI_K_SECONDARY'] = secondary_result['STOCHRSI_K']
        result['STOCHRSI_D_SECONDARY'] = secondary_result['STOCHRSI_D']
        self._secondary_stochrsi = (result['STOCHRSI_K_SECONDARY'], result['STOCHRSI_D_SECONDARY'])
        
        # 计算多周期STOCHRSI
        for period in self.multi_periods:
            if period != self.n and period != self.secondary_n:
                multi_stochrsi = STOCHRSI(n=period, m=self.m, p=self.p)
                multi_result = multi_stochrsi.calculate(data)
                result[f'STOCHRSI_K_{period}'] = multi_result['STOCHRSI_K']
                result[f'STOCHRSI_D_{period}'] = multi_result['STOCHRSI_D']
                self._multi_period_stochrsi[period] = (result[f'STOCHRSI_K_{period}'], result[f'STOCHRSI_D_{period}'])
        
        # 计算STOCHRSI的动态特性
        # 计算斜率（变化速率）
        result['K_SLOPE'] = self._calculate_slope(result['STOCHRSI_K'], 3)
        result['D_SLOPE'] = self._calculate_slope(result['STOCHRSI_D'], 3)
        
        # 计算加速度
        result['K_ACCEL'] = result['K_SLOPE'] - result['K_SLOPE'].shift(1)
        result['D_ACCEL'] = result['D_SLOPE'] - result['D_SLOPE'].shift(1)
        
        # 计算STOCHRSI波动率
        result['STOCHRSI_VOLATILITY'] = (result['STOCHRSI_K'] - result['STOCHRSI_D']).abs().rolling(window=self.volatility_lookback).std()
        
        # 保存结果
        self._result = result
        
        return result
    
    def _adjust_thresholds_by_market(self, data: pd.DataFrame) -> None:
        """
        根据市场环境和波动率动态调整超买超卖阈值
        
        Args:
            data: 包含价格数据的DataFrame
        """
        # 计算价格波动率
        close = data['close']
        
        # 计算价格变化率
        returns = close.pct_change()
        
        # 计算波动率（标准差）
        volatility = returns.rolling(window=self.volatility_lookback).std().iloc[-1]
        
        # 如果波动率数据不足，则使用默认阈值
        if pd.isna(volatility):
            return
        
        # 计算历史波动率
        historical_volatility = returns.rolling(window=self.volatility_lookback*5).std().iloc[-1]
        
        # 如果历史波动率数据不足，则使用默认阈值
        if pd.isna(historical_volatility) or historical_volatility == 0:
            return
        
        # 计算相对波动率
        relative_volatility = volatility / historical_volatility if historical_volatility > 0 else 1.0
        
        # 根据相对波动率调整阈值
        if relative_volatility > 1.5:  # 高波动市场
            # 扩大超买超卖区间
            self.overbought_threshold = 75
            self.oversold_threshold = 25
            self.extreme_overbought_threshold = 85
            self.extreme_oversold_threshold = 15
        elif relative_volatility < 0.7:  # 低波动市场
            # 收窄超买超卖区间
            self.overbought_threshold = 85
            self.oversold_threshold = 15
            self.extreme_overbought_threshold = 95
            self.extreme_oversold_threshold = 5
        else:  # 正常波动市场
            # 使用默认阈值
            self.overbought_threshold = 80
            self.oversold_threshold = 20
            self.extreme_overbought_threshold = 90
            self.extreme_oversold_threshold = 10
        
        # 根据市场环境进一步调整阈值
        if self.market_environment == 'bull_market':
            # 牛市中提高超买阈值，减少卖出信号
            self.overbought_threshold += 5
            self.extreme_overbought_threshold += 5
        elif self.market_environment == 'bear_market':
            # 熊市中降低超卖阈值，减少买入信号
            self.oversold_threshold -= 5
            self.extreme_oversold_threshold -= 5
        elif self.market_environment == 'volatile_market':
            # 高波动市场需要更极端的信号
            self.overbought_threshold += 5
            self.oversold_threshold -= 5
            self.extreme_overbought_threshold += 5
            self.extreme_oversold_threshold -= 5
        
        logger.debug(f"调整STOCHRSI阈值: 超买={self.overbought_threshold}, 超卖={self.oversold_threshold}, "
                    f"极端超买={self.extreme_overbought_threshold}, 极端超卖={self.extreme_oversold_threshold}, "
                    f"相对波动率={relative_volatility:.2f}, 市场环境={self.market_environment}")
    
    def _calculate_slope(self, series: pd.Series, period: int = 3) -> pd.Series:
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
        检测STOCHRSI与价格之间的背离关系
        
        Returns:
            pd.DataFrame: 包含背离分析结果的DataFrame
        """
        if self._result is None or self._price_data is None:
            return pd.DataFrame()
            
        # 获取STOCHRSI和价格数据
        stochrsi_k = self._result['STOCHRSI_K']
        price = self._price_data
        
        # 创建结果DataFrame
        divergence = pd.DataFrame(index=price.index)
        divergence['bullish_divergence'] = False
        divergence['bearish_divergence'] = False
        divergence['hidden_bullish_divergence'] = False
        divergence['hidden_bearish_divergence'] = False
        divergence['divergence_strength'] = 0.0
        
        # 查找价格和STOCHRSI的高点和低点
        price_peaks = find_peaks_and_troughs(price, window=10, peak_type='peak')
        price_troughs = find_peaks_and_troughs(price, window=10, peak_type='trough')
        stochrsi_peaks = find_peaks_and_troughs(stochrsi_k, window=5, peak_type='peak')
        stochrsi_troughs = find_peaks_and_troughs(stochrsi_k, window=5, peak_type='trough')
        
        # 最小背离长度(防止检测到太短的背离)
        min_divergence_length = 5
        # 最大背离长度(防止检测到太长的背离)
        max_divergence_length = 30
        
        # 常规看涨背离：价格创新低但STOCHRSI未创新低
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
                # 查找对应的STOCHRSI值
                current_stochrsi_trough = None
                prev_stochrsi_trough = None
                
                # 在价格低点附近查找STOCHRSI低点
                for st in stochrsi_troughs:
                    if abs(st - current_trough_idx) <= 3:
                        current_stochrsi_trough = st
                    if abs(st - prev_trough_idx) <= 3:
                        prev_stochrsi_trough = st
                
                # 如果找到了对应的STOCHRSI低点
                if current_stochrsi_trough is not None and prev_stochrsi_trough is not None:
                    # STOCHRSI未创新低
                    if stochrsi_k.iloc[current_stochrsi_trough] > stochrsi_k.iloc[prev_stochrsi_trough]:
                        # 计算背离强度
                        price_change = (price.iloc[current_trough_idx] / price.iloc[prev_trough_idx]) - 1
                        stochrsi_change = (stochrsi_k.iloc[current_stochrsi_trough] / stochrsi_k.iloc[prev_stochrsi_trough]) - 1
                        
                        # 防止除以零
                        if max(abs(price_change), abs(stochrsi_change)) > 0:
                            strength = abs(price_change - stochrsi_change) / max(abs(price_change), abs(stochrsi_change))
                        else:
                            strength = 0
                        
                        # 记录背离
                        divergence.iloc[current_trough_idx:current_trough_idx+5, 0] = True  # bullish_divergence
                        divergence.iloc[current_trough_idx:current_trough_idx+5, 4] = strength  # divergence_strength
        
        # 常规看跌背离：价格创新高但STOCHRSI未创新高
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
                # 查找对应的STOCHRSI值
                current_stochrsi_peak = None
                prev_stochrsi_peak = None
                
                # 在价格高点附近查找STOCHRSI高点
                for sp in stochrsi_peaks:
                    if abs(sp - current_peak_idx) <= 3:
                        current_stochrsi_peak = sp
                    if abs(sp - prev_peak_idx) <= 3:
                        prev_stochrsi_peak = sp
                
                # 如果找到了对应的STOCHRSI高点
                if current_stochrsi_peak is not None and prev_stochrsi_peak is not None:
                    # STOCHRSI未创新高
                    if stochrsi_k.iloc[current_stochrsi_peak] < stochrsi_k.iloc[prev_stochrsi_peak]:
                        # 计算背离强度
                        price_change = (price.iloc[current_peak_idx] / price.iloc[prev_peak_idx]) - 1
                        stochrsi_change = (stochrsi_k.iloc[current_stochrsi_peak] / stochrsi_k.iloc[prev_stochrsi_peak]) - 1
                        
                        # 防止除以零
                        if max(abs(price_change), abs(stochrsi_change)) > 0:
                            strength = abs(price_change - stochrsi_change) / max(abs(price_change), abs(stochrsi_change))
                        else:
                            strength = 0
                        
                        # 记录背离
                        divergence.iloc[current_peak_idx:current_peak_idx+5, 1] = True  # bearish_divergence
                        divergence.iloc[current_peak_idx:current_peak_idx+5, 4] = strength  # divergence_strength
        
        # 隐藏看涨背离：价格更高的低点但STOCHRSI更低的低点
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
                # 查找对应的STOCHRSI值
                current_stochrsi_trough = None
                prev_stochrsi_trough = None
                
                # 在价格低点附近查找STOCHRSI低点
                for st in stochrsi_troughs:
                    if abs(st - current_trough_idx) <= 3:
                        current_stochrsi_trough = st
                    if abs(st - prev_trough_idx) <= 3:
                        prev_stochrsi_trough = st
                
                # 如果找到了对应的STOCHRSI低点
                if current_stochrsi_trough is not None and prev_stochrsi_trough is not None:
                    # STOCHRSI更低的低点
                    if stochrsi_k.iloc[current_stochrsi_trough] < stochrsi_k.iloc[prev_stochrsi_trough]:
                        # 计算背离强度
                        price_change = (price.iloc[current_trough_idx] / price.iloc[prev_trough_idx]) - 1
                        stochrsi_change = (stochrsi_k.iloc[current_stochrsi_trough] / stochrsi_k.iloc[prev_stochrsi_trough]) - 1
                        
                        # 防止除以零
                        if max(abs(price_change), abs(stochrsi_change)) > 0:
                            strength = abs(price_change - stochrsi_change) / max(abs(price_change), abs(stochrsi_change))
                        else:
                            strength = 0
                        
                        # 记录背离
                        divergence.iloc[current_trough_idx:current_trough_idx+5, 2] = True  # hidden_bullish_divergence
                        divergence.iloc[current_trough_idx:current_trough_idx+5, 4] = strength  # divergence_strength
        
        # 隐藏看跌背离：价格更低的高点但STOCHRSI更高的高点
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
                # 查找对应的STOCHRSI值
                current_stochrsi_peak = None
                prev_stochrsi_peak = None
                
                # 在价格高点附近查找STOCHRSI高点
                for sp in stochrsi_peaks:
                    if abs(sp - current_peak_idx) <= 3:
                        current_stochrsi_peak = sp
                    if abs(sp - prev_peak_idx) <= 3:
                        prev_stochrsi_peak = sp
                
                # 如果找到了对应的STOCHRSI高点
                if current_stochrsi_peak is not None and prev_stochrsi_peak is not None:
                    # STOCHRSI更高的高点
                    if stochrsi_k.iloc[current_stochrsi_peak] > stochrsi_k.iloc[prev_stochrsi_peak]:
                        # 计算背离强度
                        price_change = (price.iloc[current_peak_idx] / price.iloc[prev_peak_idx]) - 1
                        stochrsi_change = (stochrsi_k.iloc[current_stochrsi_peak] / stochrsi_k.iloc[prev_stochrsi_peak]) - 1
                        
                        # 防止除以零
                        if max(abs(price_change), abs(stochrsi_change)) > 0:
                            strength = abs(price_change - stochrsi_change) / max(abs(price_change), abs(stochrsi_change))
                        else:
                            strength = 0
                        
                        # 记录背离
                        divergence.iloc[current_peak_idx:current_peak_idx+5, 3] = True  # hidden_bearish_divergence
                        divergence.iloc[current_peak_idx:current_peak_idx+5, 4] = strength  # divergence_strength
        
        return divergence
    
    def analyze_multi_period_synergy(self) -> pd.DataFrame:
        """
        多周期STOCHRSI协同分析
        
        Returns:
            pd.DataFrame: 包含多周期协同分析结果的DataFrame
        """
        if self._result is None:
            return pd.DataFrame()
            
        # 创建结果DataFrame
        synergy = pd.DataFrame(index=self._result.index)
        
        # 获取主要STOCHRSI
        primary_k = self._result['STOCHRSI_K']
        primary_d = self._result['STOCHRSI_D']
        
        # 次要周期STOCHRSI
        secondary_k, secondary_d = self._secondary_stochrsi
        
        # 分析多周期超买超卖状态
        synergy['primary_overbought'] = primary_k > self.overbought_threshold
        synergy['primary_oversold'] = primary_k < self.oversold_threshold
        synergy['secondary_overbought'] = secondary_k > self.overbought_threshold
        synergy['secondary_oversold'] = secondary_k < self.oversold_threshold
        
        # 分析多周期交叉信号
        synergy['primary_golden_cross'] = self.crossover(primary_k, primary_d)
        synergy['primary_death_cross'] = self.crossunder(primary_k, primary_d)
        synergy['secondary_golden_cross'] = self.crossover(secondary_k, secondary_d)
        synergy['secondary_death_cross'] = self.crossunder(secondary_k, secondary_d)
        
        # 计算多周期一致性
        bullish_count = (synergy['primary_golden_cross'].astype(int) + 
                         synergy['secondary_golden_cross'].astype(int) + 
                         synergy['primary_oversold'].astype(int) + 
                         synergy['secondary_oversold'].astype(int))
        
        bearish_count = (synergy['primary_death_cross'].astype(int) + 
                         synergy['secondary_death_cross'].astype(int) + 
                         synergy['primary_overbought'].astype(int) + 
                         synergy['secondary_overbought'].astype(int))
        
        # 添加其他周期的状态
        for period, (k, d) in self._multi_period_stochrsi.items():
            period_overbought = k > self.overbought_threshold
            period_oversold = k < self.oversold_threshold
            period_golden_cross = self.crossover(k, d)
            period_death_cross = self.crossunder(k, d)
            
            synergy[f'period{period}_overbought'] = period_overbought
            synergy[f'period{period}_oversold'] = period_oversold
            synergy[f'period{period}_golden_cross'] = period_golden_cross
            synergy[f'period{period}_death_cross'] = period_death_cross
            
            bullish_count += period_golden_cross.astype(int) + period_oversold.astype(int)
            bearish_count += period_death_cross.astype(int) + period_overbought.astype(int)
        
        # 计算总检查数
        total_checks = 4 + len(self._multi_period_stochrsi) * 4  # 主要和次要周期各有4个检查，每个额外周期也有4个检查
        
        # 计算看涨和看跌比例
        synergy['bullish_ratio'] = bullish_count / total_checks
        synergy['bearish_ratio'] = bearish_count / total_checks
        
        # 计算一致性得分（0-100）
        synergy['consensus_score'] = 50 + (synergy['bullish_ratio'] - synergy['bearish_ratio']) * 50
        
        # 识别强烈共振信号
        synergy['strong_bullish_signal'] = (synergy['primary_golden_cross'] & 
                                           synergy['primary_oversold'] & 
                                           synergy['consensus_score'] > 70)
        
        synergy['strong_bearish_signal'] = (synergy['primary_death_cross'] & 
                                           synergy['primary_overbought'] & 
                                           synergy['consensus_score'] < 30)
        
        return synergy
    
    def evaluate_cross_quality(self) -> pd.DataFrame:
        """
        评估K和D线交叉质量
        
        Returns:
            pd.DataFrame: 包含交叉质量评估结果的DataFrame
        """
        if self._result is None:
            return pd.DataFrame()
            
        # 获取STOCHRSI数据
        k = self._result['STOCHRSI_K']
        d = self._result['STOCHRSI_D']
        
        # 创建结果DataFrame
        quality = pd.DataFrame(index=self._result.index)
        
        # 检测交叉
        golden_cross = self.crossover(k, d)
        death_cross = self.crossunder(k, d)
        
        quality['golden_cross'] = golden_cross
        quality['death_cross'] = death_cross
        
        # 评估交叉角度
        quality['cross_angle'] = 0.0
        
        for i in range(3, len(k)):
            if golden_cross.iloc[i] or death_cross.iloc[i]:
                # 计算交叉前后3个周期的K和D线斜率
                k_slope = (k.iloc[i] - k.iloc[i-3]) / 3
                d_slope = (d.iloc[i] - d.iloc[i-3]) / 3
                
                # 交叉角度是斜率差的绝对值
                quality.iloc[i, 2] = abs(k_slope - d_slope)  # cross_angle
        
        # 评估交叉位置
        quality['cross_position_score'] = 0.0
        
        for i in range(len(k)):
            if golden_cross.iloc[i]:
                # 金叉位置得分（超卖区域得分高）
                if k.iloc[i] < 20:
                    quality.iloc[i, 3] = 1.0  # 最高分
                elif k.iloc[i] < 30:
                    quality.iloc[i, 3] = 0.8
                elif k.iloc[i] < 50:
                    quality.iloc[i, 3] = 0.5
                else:
                    quality.iloc[i, 3] = 0.3
            
            elif death_cross.iloc[i]:
                # 死叉位置得分（超买区域得分高）
                if k.iloc[i] > 80:
                    quality.iloc[i, 3] = 1.0  # 最高分
                elif k.iloc[i] > 70:
                    quality.iloc[i, 3] = 0.8
                elif k.iloc[i] > 50:
                    quality.iloc[i, 3] = 0.5
                else:
                    quality.iloc[i, 3] = 0.3
        
        # 评估交叉后的分离速度
        quality['separation_speed'] = 0.0
        
        for i in range(len(k)-3):
            if golden_cross.iloc[i] or death_cross.iloc[i]:
                if i+3 < len(k):
                    # 计算交叉后3个周期的K和D线分离距离
                    separation = abs(k.iloc[i+3] - d.iloc[i+3])
                    quality.iloc[i, 4] = separation
        
        # 综合评分（0-100）
        quality['cross_quality_score'] = 0.0
        
        for i in range(len(k)):
            if golden_cross.iloc[i] or death_cross.iloc[i]:
                # 综合考虑角度、位置和分离速度
                angle_score = min(40, quality.iloc[i, 2] * 80)
                position_score = quality.iloc[i, 3] * 40
                separation_score = min(20, quality.iloc[i, 4] * 2)
                
                quality.iloc[i, 5] = angle_score + position_score + separation_score
        
        return quality
    
    def identify_patterns(self) -> pd.DataFrame:
        """
        识别STOCHRSI形态
        
        Returns:
            pd.DataFrame: 包含形态识别结果的DataFrame
        """
        if self._result is None:
            return pd.DataFrame()
            
        # 获取STOCHRSI数据
        k = self._result['STOCHRSI_K']
        d = self._result['STOCHRSI_D']
        
        # 创建结果DataFrame
        patterns = pd.DataFrame(index=self._result.index)
        
        # 基础形态
        patterns['overbought'] = k > self.overbought_threshold
        patterns['oversold'] = k < self.oversold_threshold
        patterns['extreme_overbought'] = k > self.extreme_overbought_threshold
        patterns['extreme_oversold'] = k < self.extreme_oversold_threshold
        
        # 交叉信号
        patterns['golden_cross'] = self.crossover(k, d)
        patterns['death_cross'] = self.crossunder(k, d)
        
        # 超买超卖区域的交叉信号
        patterns['oversold_golden_cross'] = patterns['oversold'] & patterns['golden_cross']
        patterns['overbought_death_cross'] = patterns['overbought'] & patterns['death_cross']
        
        # 中线交叉（50水平线）
        patterns['k_cross_above_50'] = self.crossover(k, 50)
        patterns['k_cross_below_50'] = self.crossunder(k, 50)
        
        # 高质量交叉
        cross_quality = self.evaluate_cross_quality()
        if 'cross_quality_score' in cross_quality.columns:
            high_quality_threshold = 70
            patterns['high_quality_golden_cross'] = (patterns['golden_cross'] & 
                                                   (cross_quality['cross_quality_score'] > high_quality_threshold))
            patterns['high_quality_death_cross'] = (patterns['death_cross'] & 
                                                  (cross_quality['cross_quality_score'] > high_quality_threshold))
        
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
            patterns['strong_bullish_signal'] = synergy['strong_bullish_signal']
            patterns['strong_bearish_signal'] = synergy['strong_bearish_signal']
            patterns['strong_bullish_consensus'] = synergy['consensus_score'] > 70
            patterns['strong_bearish_consensus'] = synergy['consensus_score'] < 30
        
        # W底和M顶形态
        patterns['w_bottom'] = self._detect_w_bottom(k)
        patterns['m_top'] = self._detect_m_top(k)
        
        # 钝化形态（K线在超买或超卖区域徘徊）
        patterns['overbought_stagnation'] = self._detect_stagnation(k, threshold=self.overbought_threshold, periods=5, direction='high')
        patterns['oversold_stagnation'] = self._detect_stagnation(k, threshold=self.oversold_threshold, periods=5, direction='low')
        
        # 趋势加速/减速
        if 'K_ACCEL' in self._result.columns:
            patterns['k_acceleration'] = self._result['K_ACCEL'] > 0
            patterns['k_deceleration'] = self._result['K_ACCEL'] < 0
        
        return patterns
    
    def _detect_w_bottom(self, k: pd.Series) -> pd.Series:
        """
        识别W底形态
        
        Args:
            k: STOCHRSI_K序列
            
        Returns:
            pd.Series: W底形态识别结果
        """
        w_bottom = pd.Series(False, index=k.index)
        
        if len(k) < 15:
            return w_bottom
        
        # W底特征：两个低点，中间有一个高点，第二个低点高于第一个低点
        for i in range(15, len(k)):
            # 获取最近15个周期的数据
            window = k.iloc[i-15:i+1]
            
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
            # 1. 两个低点都在超卖区域或接近超卖区域
            # 2. 中间高点显著高于两个低点
            # 3. 第二个低点高于第一个低点（背离）
            if (window.loc[low1_idx] < self.oversold_threshold + 10 and 
                window.loc[low2_idx] < self.oversold_threshold + 10 and
                mid_high > window.loc[low1_idx] + 15 and
                window.loc[low2_idx] > window.loc[low1_idx] and
                mid_high_idx > low1_idx and mid_high_idx < low2_idx):
                
                w_bottom.loc[i] = True
        
        return w_bottom
    
    def _detect_m_top(self, k: pd.Series) -> pd.Series:
        """
        识别M顶形态
        
        Args:
            k: STOCHRSI_K序列
            
        Returns:
            pd.Series: M顶形态识别结果
        """
        m_top = pd.Series(False, index=k.index)
        
        if len(k) < 15:
            return m_top
        
        # M顶特征：两个高点，中间有一个低点，第二个高点低于第一个高点
        for i in range(15, len(k)):
            # 获取最近15个周期的数据
            window = k.iloc[i-15:i+1]
            
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
            # 1. 两个高点都在超买区域或接近超买区域
            # 2. 中间低点显著低于两个高点
            # 3. 第二个高点低于第一个高点（背离）
            if (window.loc[high1_idx] > self.overbought_threshold - 10 and 
                window.loc[high2_idx] > self.overbought_threshold - 10 and
                mid_low < window.loc[high1_idx] - 15 and
                window.loc[high2_idx] < window.loc[high1_idx] and
                mid_low_idx > high1_idx and mid_low_idx < high2_idx):
                
                m_top.loc[i] = True
        
        return m_top
    
    def _detect_stagnation(self, k: pd.Series, threshold: float, periods: int, direction: str) -> pd.Series:
        """
        检测STOCHRSI_K在超买/超卖区域的钝化形态
        
        Args:
            k: STOCHRSI_K序列
            threshold: 超买/超卖阈值
            periods: 持续周期数
            direction: 方向，'high'表示超买区域，'low'表示超卖区域
            
        Returns:
            pd.Series: 钝化形态识别结果
        """
        stagnation = pd.Series(False, index=k.index)
        
        if len(k) < periods:
            return stagnation
        
        # 检查连续periods个周期K都在超买/超卖区域
        for i in range(periods, len(k)):
            window = k.iloc[i-periods+1:i+1]
            
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
        计算STOCHRSI综合评分 (0-100)
        
        Args:
            data (pd.DataFrame, optional): 价格数据，如果未提供则使用上次计算结果
            
        Returns:
            pd.Series: 评分 (0-100，50为中性)
        """
        if self._result is None and data is not None:
            self.calculate(data)
            
        if self._result is None:
            return pd.Series()
        
        # 获取STOCHRSI数据
        k = self._result['STOCHRSI_K']
        d = self._result['STOCHRSI_D']
        k_slope = self._result['K_SLOPE']
        d_slope = self._result['D_SLOPE']
        
        # 获取背离分析
        divergence = self.detect_divergence()
        
        # 获取多周期协同分析
        synergy = self.analyze_multi_period_synergy()
        
        # 获取交叉质量评估
        cross_quality = self.evaluate_cross_quality()
        
        # 获取形态识别
        patterns = self.identify_patterns()
        
        # 基础分数为50（中性）
        score = pd.Series(50, index=self._result.index)
        
        # 1. K线位置评分 (±15分)
        # K > 80 看跌，K < 20 看涨
        score += np.where(k < self.oversold_threshold, 
                         np.minimum((self.oversold_threshold - k) / 20 * 15, 15),
                         np.where(k > self.overbought_threshold,
                                 np.maximum((self.overbought_threshold - k) / 20 * 15, -15),
                                 (50 - k) / 30 * 7.5))  # 中间区域小幅调整
        
        # 2. K与D线关系评分 (±10分)
        # K > D 看涨，K < D 看跌
        k_vs_d = k - d
        score += np.where(k_vs_d > 0, 
                         np.minimum(k_vs_d / 10 * 10, 10),
                         np.maximum(k_vs_d / 10 * 10, -10))
        
        # 3. K线斜率评分 (±10分)
        # 斜率为正看涨，斜率为负看跌
        score += np.where(k_slope > 0, 
                         np.minimum(k_slope * 5, 10),
                         np.maximum(k_slope * 5, -10))
        
        # 4. K线交叉D线评分 (±15分)
        if not cross_quality.empty:
            # 金叉信号（K上穿D）
            golden_cross = cross_quality.get('golden_cross', pd.Series(False, index=score.index))
            if isinstance(golden_cross, pd.Series) and not golden_cross.empty:
                cross_strength = cross_quality.get('cross_strength', pd.Series(0.5, index=score.index))
                score.loc[golden_cross] += np.minimum(cross_strength.loc[golden_cross] * 30, 15)
            
            # 死叉信号（K下穿D）
            death_cross = cross_quality.get('death_cross', pd.Series(False, index=score.index))
            if isinstance(death_cross, pd.Series) and not death_cross.empty:
                cross_strength = cross_quality.get('cross_strength', pd.Series(0.5, index=score.index))
                score.loc[death_cross] -= np.minimum(cross_strength.loc[death_cross] * 30, 15)
        
        # 5. 背离评分 (±15分)
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
            
            # 隐藏背离也可以加入评分
            hidden_bullish = divergence.get('hidden_bullish_divergence', pd.Series(False, index=score.index))
            if isinstance(hidden_bullish, pd.Series) and not hidden_bullish.empty:
                score.loc[hidden_bullish] += 10
                
            hidden_bearish = divergence.get('hidden_bearish_divergence', pd.Series(False, index=score.index))
            if isinstance(hidden_bearish, pd.Series) and not hidden_bearish.empty:
                score.loc[hidden_bearish] -= 10
        
        # 6. 多周期协同评分 (±15分)
        if not synergy.empty:
            bull_synergy = synergy.get('bullish_agreement', pd.Series(False, index=score.index))
            if isinstance(bull_synergy, pd.Series) and not bull_synergy.empty:
                synergy_strength = synergy.get('synergy_strength', pd.Series(0.5, index=score.index))
                score.loc[bull_synergy] += np.minimum(synergy_strength.loc[bull_synergy] * 30, 15)
            
            bear_synergy = synergy.get('bearish_agreement', pd.Series(False, index=score.index))
            if isinstance(bear_synergy, pd.Series) and not bear_synergy.empty:
                synergy_strength = synergy.get('synergy_strength', pd.Series(0.5, index=score.index))
                score.loc[bear_synergy] -= np.minimum(synergy_strength.loc[bear_synergy] * 30, 15)
        
        # 7. 特殊形态评分 (±15分)
        if not patterns.empty:
            # 看涨形态: W底、低位徘徊突破等
            for pattern in ['w_bottom', 'oversold_reversal', 'positive_hook']:
                if pattern in patterns.columns:
                    pattern_signal = patterns[pattern]
                    if isinstance(pattern_signal, pd.Series) and not pattern_signal.empty:
                        score.loc[pattern_signal] += 15
            
            # 看跌形态: M顶、高位徘徊突破等
            for pattern in ['m_top', 'overbought_reversal', 'negative_hook']:
                if pattern in patterns.columns:
                    pattern_signal = patterns[pattern]
                    if isinstance(pattern_signal, pd.Series) and not pattern_signal.empty:
                        score.loc[pattern_signal] -= 15
        
        # 8. 极端区域评分 (±5分)
        # 在极端区域的额外加减分
        score += np.where(k < self.extreme_oversold_threshold, 5, 
                         np.where(k > self.extreme_overbought_threshold, -5, 0))
        
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
        计算增强型STOCHRSI指标原始评分 (0-100分)
        
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
        # 确保已计算STOCHRSI
        if self._result is None:
            self.calculate(data)
            
        if self._result is None:
            return pd.DataFrame()
            
        # 获取STOCHRSI数据
        k = self._result['STOCHRSI_K']
        d = self._result['STOCHRSI_D']
        
        # 计算STOCHRSI综合评分
        score = self.calculate_score()
        
        # 识别形态
        patterns = self.identify_patterns()
        
        # 创建信号DataFrame
        signals = pd.DataFrame(index=self._result.index)
        signals['stochrsi_k'] = k
        signals['stochrsi_d'] = d
        signals['score'] = score
        
        # 生成基础信号
        signals['buy_signal'] = False
        signals['sell_signal'] = False
        signals['neutral_signal'] = True
        
        # 买入信号条件
        buy_conditions = [
            (score > 70),  # 评分高于70
            patterns.get('oversold_golden_cross', pd.Series(False, index=self._result.index)),  # 超卖区域金叉
            patterns.get('bullish_divergence', pd.Series(False, index=self._result.index)),  # 正背离
            patterns.get('w_bottom', pd.Series(False, index=self._result.index)),  # W底形态
            patterns.get('strong_bullish_signal', pd.Series(False, index=self._result.index))  # 多周期看涨信号
        ]
        
        # 卖出信号条件
        sell_conditions = [
            (score < 30),  # 评分低于30
            patterns.get('overbought_death_cross', pd.Series(False, index=self._result.index)),  # 超买区域死叉
            patterns.get('bearish_divergence', pd.Series(False, index=self._result.index)),  # 负背离
            patterns.get('m_top', pd.Series(False, index=self._result.index)),  # M顶形态
            patterns.get('strong_bearish_signal', pd.Series(False, index=self._result.index))  # 多周期看跌信号
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
        signals.loc[buy_signals & patterns.get('oversold_golden_cross', False), 'signal_type'] = 'STOCHRSI超卖金叉'
        signals.loc[buy_signals & patterns.get('oversold_golden_cross', False), 'signal_desc'] = 'STOCHRSI在超卖区域形成金叉，显示超卖反弹信号'
        
        signals.loc[buy_signals & patterns.get('bullish_divergence', False), 'signal_type'] = 'STOCHRSI正背离'
        signals.loc[buy_signals & patterns.get('bullish_divergence', False), 'signal_desc'] = '价格创新低但STOCHRSI未创新低，显示下跌动能减弱'
        
        signals.loc[buy_signals & patterns.get('w_bottom', False), 'signal_type'] = 'STOCHRSI-W底'
        signals.loc[buy_signals & patterns.get('w_bottom', False), 'signal_desc'] = 'STOCHRSI形成W底形态，显示强劲反转信号'
        
        signals.loc[buy_signals & patterns.get('strong_bullish_signal', False), 'signal_type'] = 'STOCHRSI多周期看涨'
        signals.loc[buy_signals & patterns.get('strong_bullish_signal', False), 'signal_desc'] = '多周期STOCHRSI共振发出看涨信号'
        
        # 卖出信号类型和描述
        sell_signals = signals['sell_signal']
        signals.loc[sell_signals & patterns.get('overbought_death_cross', False), 'signal_type'] = 'STOCHRSI超买死叉'
        signals.loc[sell_signals & patterns.get('overbought_death_cross', False), 'signal_desc'] = 'STOCHRSI在超买区域形成死叉，显示超买回落信号'
        
        signals.loc[sell_signals & patterns.get('bearish_divergence', False), 'signal_type'] = 'STOCHRSI负背离'
        signals.loc[sell_signals & patterns.get('bearish_divergence', False), 'signal_desc'] = '价格创新高但STOCHRSI未创新高，显示上涨动能减弱'
        
        signals.loc[sell_signals & patterns.get('m_top', False), 'signal_type'] = 'STOCHRSI-M顶'
        signals.loc[sell_signals & patterns.get('m_top', False), 'signal_desc'] = 'STOCHRSI形成M顶形态，显示强劲见顶信号'
        
        signals.loc[sell_signals & patterns.get('strong_bearish_signal', False), 'signal_type'] = 'STOCHRSI多周期看跌'
        signals.loc[sell_signals & patterns.get('strong_bearish_signal', False), 'signal_desc'] = '多周期STOCHRSI共振发出看跌信号'
        
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
            ('w_bottom', 20),
            ('m_top', 20),
            ('high_quality_golden_cross', 15),
            ('high_quality_death_cross', 15),
            ('strong_bullish_signal', 15),
            ('strong_bearish_signal', 15),
            ('strong_bullish_consensus', 10),
            ('strong_bearish_consensus', 10)
        ]:
            if pattern in patterns.columns:
                confidence = np.where(patterns[pattern], np.minimum(100, confidence + boost), confidence)
        
        # 超买超卖区域对确信度的影响
        k_values = self._result['STOCHRSI_K']
        
        # 在极端超买超卖区域的信号更可靠
        confidence += np.where(k_values < self.extreme_oversold_threshold, 10, 0)
        confidence += np.where(k_values > self.extreme_overbought_threshold, 10, 0)
        
        # K和D的分离程度对确信度的影响
        d_values = self._result['STOCHRSI_D']
        separation = abs(k_values - d_values)
        
        # 大分离更可靠
        confidence += np.where(separation > 15, 5, 0)
        
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
    
    def atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        计算平均真实波幅(ATR)
        
        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            period: 计算周期
            
        Returns:
            pd.Series: ATR序列
        """
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.DataFrame([tr1, tr2, tr3]).max()
        atr = tr.rolling(window=period).mean()
        
        return atr 