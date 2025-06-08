#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
时间周期分析指标模块

实现时间周期分析功能，用于识别不同级别的循环规律和重要的时间周期点
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple
from datetime import datetime, timedelta
import scipy.fftpack
from scipy import signal

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger

logger = get_logger(__name__)


class TimeCycleAnalysis(BaseIndicator):
    """
    时间周期分析指标
    
    分析不同级别的循环规律，识别重要的时间周期点
    """
    
    def __init__(self, 
                 min_cycle_days: int = 10, 
                 max_cycle_days: int = 252, 
                 n_cycles: int = 5):
        """
        初始化时间周期分析指标
        
        Args:
            min_cycle_days: 最小周期天数，默认为10日
            max_cycle_days: 最大周期天数，默认为252日（约一年交易日）
            n_cycles: 检测的主要周期数量，默认为5
        """
        super().__init__(name="TimeCycleAnalysis", description="时间周期分析指标，识别不同级别的循环规律")
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        self.min_cycle_days = min_cycle_days
        self.max_cycle_days = max_cycle_days
        self.n_cycles = n_cycles
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算时间周期分析指标
        
        Args:
            data: 输入数据，包含价格数据
            
        Returns:
            pd.DataFrame: 计算结果，包含周期分析信息
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close"])
        
        # 初始化结果数据框
        result = data.copy()
        
        # 使用价格数据进行周期分析
        price_series = data["close"].values
        
        # 去除趋势，使用一阶差分
        detrended = np.diff(price_series)
        
        # 使用傅里叶变换进行周期分析
        if len(detrended) > self.max_cycle_days * 2:  # 确保数据足够长
            cycles = self._detect_cycles(detrended)
            
            # 生成周期指标
            result = self._generate_cycle_indicators(result, cycles, len(price_series))
            
            # 预测未来的转折点
            result = self._predict_turning_points(result, cycles, len(price_series))
        else:
            logger.warning(f"数据长度不足，无法进行可靠的周期分析。当前长度: {len(detrended)}，需要至少: {self.max_cycle_days * 2}")
        
        return result
    
    def _detect_cycles(self, detrended_series: np.ndarray) -> List[Dict[str, Any]]:
        """
        检测价格序列中的周期
        
        Args:
            detrended_series: 去趋势的价格序列
            
        Returns:
            List[Dict[str, Any]]: 检测到的周期信息列表
        """
        # 执行快速傅里叶变换
        n = len(detrended_series)
        fft_values = scipy.fftpack.fft(detrended_series)
        power = np.abs(fft_values[:n//2]) ** 2
        freqs = scipy.fftpack.fftfreq(n, 1)[:n//2]
        
        # 转换频率为周期天数
        periods = 1 / freqs[1:]  # 跳过第一个元素（直流分量）
        power = power[1:]
        
        # 仅保留在指定范围内的周期
        valid_indices = (periods >= self.min_cycle_days) & (periods <= self.max_cycle_days)
        valid_periods = periods[valid_indices]
        valid_power = power[valid_indices]
        
        # 检测功率谱中的峰值
        peaks_indices = signal.find_peaks(valid_power)[0]
        
        # 按功率排序峰值
        sorted_indices = np.argsort(valid_power[peaks_indices])[::-1]
        top_peaks = peaks_indices[sorted_indices[:self.n_cycles]]
        
        # 收集主要周期信息
        cycles = []
        for i, peak_idx in enumerate(top_peaks):
            cycle_length = int(round(valid_periods[peak_idx]))
            power_value = valid_power[peak_idx]
            
            cycles.append({
                "rank": i + 1,  # 周期排名
                "length": cycle_length,  # 周期长度（天）
                "power": power_value,  # 周期强度
                "normalized_power": power_value / np.sum(valid_power)  # 归一化强度
            })
        
        return cycles
    
    def _generate_cycle_indicators(self, result: pd.DataFrame, cycles: List[Dict[str, Any]], 
                                 data_length: int) -> pd.DataFrame:
        """
        生成周期指标
        
        Args:
            result: 结果数据框
            cycles: 检测到的周期列表
            data_length: 原始数据长度
            
        Returns:
            pd.DataFrame: 添加了周期指标的数据框
        """
        # 初始化周期列
        result["cycle_position"] = 0.0
        
        # 添加主要周期信息
        for i, cycle in enumerate(cycles):
            cycle_length = cycle["length"]
            
            # 为每个主要周期创建一个位置指标（0-1之间的值，表示在周期中的位置）
            # 使用向量化操作代替循环
            indices = np.arange(data_length)
            cycle_position = (indices % cycle_length) / cycle_length
            
            # 计算周期角度（0-360度）
            cycle_angle = cycle_position * 360
            
            # 使用向量化计算正弦波和余弦波
            cycle_sine = np.sin(np.radians(cycle_angle))
            cycle_cosine = np.cos(np.radians(cycle_angle))
            
            # 添加到结果
            result[f"cycle_{i+1}_position"] = cycle_position
            result[f"cycle_{i+1}_sine"] = cycle_sine
            result[f"cycle_{i+1}_cosine"] = cycle_cosine
            result[f"cycle_{i+1}_length"] = cycle_length
        
        # 计算组合周期指标（加权平均）
        if cycles:
            total_power = sum(cycle["normalized_power"] for cycle in cycles)
            
            # 防止除以零
            if total_power > 0:
                # 使用向量化操作计算加权位置
                weighted_position = np.zeros(data_length)
                for i, cycle in enumerate(cycles):
                    weighted_position += result[f"cycle_{i+1}_position"].values * cycle["normalized_power"]
                
                weighted_position /= total_power
                
                result["combined_cycle_position"] = weighted_position
                result["combined_cycle_angle"] = weighted_position * 360
                result["combined_cycle_sine"] = np.sin(np.radians(result["combined_cycle_angle"]))
        
        return result
    
    def _predict_turning_points(self, result: pd.DataFrame, cycles: List[Dict[str, Any]],
                               data_length: int) -> pd.DataFrame:
        """
        预测未来的转折点
        
        Args:
            result: 结果数据框
            cycles: 检测到的周期列表
            data_length: 原始数据长度
            
        Returns:
            pd.DataFrame: 添加了转折点预测的数据框
        """
        # 初始化转折点列
        result["potential_turning_point"] = False
        
        # 计算主要周期的转折点（位于0.0, 0.25, 0.5, 0.75的位置）
        for i, cycle in enumerate(cycles):
            # 获取周期位置
            position_col = f"cycle_{i+1}_position"
            
            # 使用向量化操作计算高点、低点和中间转折点
            # 高点（位置接近0.0或1.0）
            high_points = (result[position_col] < 0.05) | (result[position_col] > 0.95)
            
            # 低点（位置接近0.5）
            low_points = (result[position_col] > 0.45) & (result[position_col] < 0.55)
            
            # 中间转折点（位置接近0.25或0.75）
            mid_points = ((result[position_col] > 0.20) & (result[position_col] < 0.30)) | \
                         ((result[position_col] > 0.70) & (result[position_col] < 0.80))
            
            # 标记转折点
            result[f"cycle_{i+1}_high"] = high_points
            result[f"cycle_{i+1}_low"] = low_points
            result[f"cycle_{i+1}_mid"] = mid_points
            
            # 更新潜在转折点
            result.loc[high_points | low_points, "potential_turning_point"] = True
        
        # 预测未来30天的主要转折点
        future_points = []
        last_date = pd.to_datetime(result.index[-1])
        
        # 预测每个周期的未来转折点
        for i, cycle in enumerate(cycles):
            cycle_length = cycle["length"]
            
            # 获取当前周期位置
            current_position = result[f"cycle_{i+1}_position"].iloc[-1]
            
            # 计算下一个高点和低点的天数
            days_to_next_high = ((1.0 - current_position) * cycle_length) % cycle_length
            if days_to_next_high < 1:  # 如果当前就接近高点
                days_to_next_high += cycle_length
                
            days_to_next_low = ((0.5 - current_position) * cycle_length) % cycle_length
            if days_to_next_low < 1:  # 如果当前就接近低点
                days_to_next_low += cycle_length
                
            # 添加未来转折点
            if days_to_next_high <= 30:
                future_date = last_date + timedelta(days=int(days_to_next_high))
                future_points.append({
                    "date": future_date,
                    "type": "高点",
                    "cycle": i+1,
                    "days_ahead": days_to_next_high,
                    "strength": cycle["normalized_power"]
                })
                
            if days_to_next_low <= 30:
                future_date = last_date + timedelta(days=int(days_to_next_low))
                future_points.append({
                    "date": future_date,
                    "type": "低点",
                    "cycle": i+1,
                    "days_ahead": days_to_next_low,
                    "strength": cycle["normalized_power"]
                })
        
        # 将未来转折点保存到结果对象的属性中
        result.future_turning_points = future_points
        
        return result
    
    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成周期分析信号
        
        Args:
            data: 输入数据，包含周期分析指标
            
        Returns:
            pd.DataFrame: 包含周期交易信号的数据框
        """
        if "combined_cycle_sine" not in data.columns:
            data = self.calculate(data)
        
        # 初始化信号列
        data["cycle_signal"] = 0
        
        if "combined_cycle_sine" in data.columns:
            # 使用组合周期的正弦波生成信号
            for i in range(1, len(data)):
                # 从负到正穿越零轴：买入信号
                if data["combined_cycle_sine"].iloc[i-1] < 0 and data["combined_cycle_sine"].iloc[i] >= 0:
                    data.iloc[i, data.columns.get_loc("cycle_signal")] = 1
                
                # 从正到负穿越零轴：卖出信号
                elif data["combined_cycle_sine"].iloc[i-1] > 0 and data["combined_cycle_sine"].iloc[i] <= 0:
                    data.iloc[i, data.columns.get_loc("cycle_signal")] = -1
        
        return data
    
    def get_dominant_cycles(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        获取主导周期信息
        
        Args:
            data: 输入数据，包含周期分析指标
            
        Returns:
            List[Dict[str, Any]]: 主导周期列表
        """
        if "cycle_1_length" not in data.columns:
            data = self.calculate(data)
        
        dominant_cycles = []
        
        # 收集所有周期信息
        for i in range(1, self.n_cycles + 1):
            length_col = f"cycle_{i}_length"
            
            if length_col in data.columns:
                cycle_length = data[length_col].iloc[0]
                
                # 周期类型分类
                if cycle_length <= 20:
                    cycle_type = "短期周期"
                elif cycle_length <= 60:
                    cycle_type = "中期周期"
                else:
                    cycle_type = "长期周期"
                
                dominant_cycles.append({
                    "rank": i,
                    "length": cycle_length,
                    "type": cycle_type
                })
        
        return dominant_cycles
    
    def get_current_cycle_phase(self, data: pd.DataFrame) -> str:
        """
        获取当前周期阶段
        
        Args:
            data: 输入数据，包含周期分析指标
            
        Returns:
            str: 当前周期阶段描述
        """
        if "combined_cycle_position" not in data.columns:
            data = self.calculate(data)
        
        # 获取最新的组合周期位置
        if "combined_cycle_position" in data.columns:
            position = data["combined_cycle_position"].iloc[-1]
            
            # 判断周期阶段
            if position < 0.125:
                return "初始上升阶段"
            elif position < 0.25:
                return "加速上升阶段"
            elif position < 0.375:
                return "上升减速阶段"
            elif position < 0.5:
                return "顶部转折阶段"
            elif position < 0.625:
                return "初始下降阶段"
            elif position < 0.75:
                return "加速下降阶段"
            elif position < 0.875:
                return "下降减速阶段"
            else:
                return "底部转折阶段"
        
        return "无法确定"

    def _extract_dominant_cycles(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        提取主导周期
        
        Args:
            data: 输入数据，包含价格数据
            
        Returns:
            List[Dict[str, Any]]: 主导周期列表
        """
        try:
            # 获取价格序列
            price_series = data["close"].values
            
            # 验证数据有效性
            if len(price_series) < self.min_cycle_days * 2:
                logger.warning(f"数据量不足以进行周期分析，至少需要 {self.min_cycle_days * 2} 个数据点")
                return []
            
            # 对价格序列进行去趋势（使用简单的差分）
            detrended_prices = np.diff(price_series)
            
            # 应用汉宁窗函数减少频谱泄漏
            window = np.hanning(len(detrended_prices))
            windowed_prices = detrended_prices * window
            
            # 计算FFT
            try:
                fft_result = np.fft.fft(windowed_prices)
                
                # 计算频率
                n = len(fft_result)
                freq = np.fft.fftfreq(n, d=1)
                
                # 只考虑正频率部分
                positive_freq_idx = np.where(freq > 0)[0]
                
                # 根据周期范围过滤频率
                min_freq = 1 / self.max_cycle_days
                max_freq = 1 / self.min_cycle_days
                
                filtered_idx = np.where((freq > min_freq) & (freq < max_freq))[0]
                
                if len(filtered_idx) == 0:
                    logger.warning("在指定周期范围内未找到有效频率")
                    return []
                
                # 计算功率谱
                power_spectrum = np.abs(fft_result) ** 2
                
                # 在过滤的频率范围内找到主要周期
                sorted_idx = np.argsort(-power_spectrum[filtered_idx])
                
                dominant_cycles = []
                
                # 提取主要周期
                for i in range(min(self.n_cycles, len(sorted_idx))):
                    idx = filtered_idx[sorted_idx[i]]
                    
                    # 周期长度（以天为单位）
                    cycle_length = 1 / freq[idx]
                    
                    # 计算振幅和相位
                    amplitude = np.abs(fft_result[idx]) / n * 2  # 乘以2是因为我们只考虑了一半的频谱
                    phase = np.angle(fft_result[idx])
                    
                    # 归一化幅度（相对于最大周期）
                    if i == 0:
                        max_amplitude = amplitude
                        norm_amplitude = 1.0
                    else:
                        norm_amplitude = amplitude / max_amplitude
                    
                    # 添加周期信息
                    dominant_cycles.append({
                        "length": round(cycle_length),
                        "frequency": freq[idx],
                        "amplitude": amplitude,
                        "normalized_amplitude": norm_amplitude,
                        "phase": phase,
                        "power": power_spectrum[idx]
                    })
                    
                    logger.debug(f"检测到主要周期: {round(cycle_length)} 天，幅度: {norm_amplitude:.2f}")
                
                return dominant_cycles
                
            except Exception as e:
                logger.error(f"FFT计算失败: {e}")
                return []
            
        except Exception as e:
            logger.error(f"提取主导周期时发生错误: {e}")
            return []

    def generate_signals(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        生成时间周期分析信号
        
        Args:
            data: 输入数据，包含OHLCV数据
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            pd.DataFrame: 信号结果DataFrame，包含标准化信号
        """
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
        
        # 计算时间周期指标
        cycle_result = self.calculate(data)
        
        # 检查是否有有效的周期结果
        if cycle_result.empty or 'combined_cycle_sine' not in cycle_result.columns:
            return signals  # 如果没有有效的周期结果，返回默认信号
        
        # 获取当前周期相位
        current_phase = self.get_current_cycle_phase(data)
        
        # 获取主要周期
        dominant_cycles = self.get_dominant_cycles(data)
        
        # 使用组合周期正弦波判断趋势拐点
        combined_sine = cycle_result['combined_cycle_sine']
        
        # 计算组合周期正弦波的一阶和二阶差分，用于判断拐点
        sine_diff1 = combined_sine.diff()
        sine_diff2 = sine_diff1.diff()
        
        # 初始化买卖信号
        for i in range(5, len(signals)):
            # 底部拐点（正弦波由负变正，且二阶导数为正）
            if (combined_sine.iloc[i-1] < 0 and combined_sine.iloc[i] >= 0 and 
                sine_diff1.iloc[i] > 0 and sine_diff2.iloc[i] > 0):
                signals.loc[signals.index[i], 'buy_signal'] = True
                signals.loc[signals.index[i], 'neutral_signal'] = False
                signals.loc[signals.index[i], 'trend'] = 1
                signals.loc[signals.index[i], 'signal_type'] = '周期底部'
                signals.loc[signals.index[i], 'signal_desc'] = '时间周期底部拐点，潜在买入机会'
                signals.loc[signals.index[i], 'confidence'] = 70
                signals.loc[signals.index[i], 'score'] = 75
                
            # 顶部拐点（正弦波由正变负，且二阶导数为负）
            elif (combined_sine.iloc[i-1] > 0 and combined_sine.iloc[i] <= 0 and 
                  sine_diff1.iloc[i] < 0 and sine_diff2.iloc[i] < 0):
                signals.loc[signals.index[i], 'sell_signal'] = True
                signals.loc[signals.index[i], 'neutral_signal'] = False
                signals.loc[signals.index[i], 'trend'] = -1
                signals.loc[signals.index[i], 'signal_type'] = '周期顶部'
                signals.loc[signals.index[i], 'signal_desc'] = '时间周期顶部拐点，潜在卖出机会'
                signals.loc[signals.index[i], 'confidence'] = 70
                signals.loc[signals.index[i], 'score'] = 25
            
            # 上升趋势确认（正弦波为正且一阶导数为正）
            elif combined_sine.iloc[i] > 0.5 and sine_diff1.iloc[i] > 0:
                signals.loc[signals.index[i], 'trend'] = 1
                signals.loc[signals.index[i], 'signal_type'] = '上升周期'
                signals.loc[signals.index[i], 'signal_desc'] = '时间周期处于上升阶段'
                signals.loc[signals.index[i], 'score'] = 65
                
            # 下降趋势确认（正弦波为负且一阶导数为负）
            elif combined_sine.iloc[i] < -0.5 and sine_diff1.iloc[i] < 0:
                signals.loc[signals.index[i], 'trend'] = -1
                signals.loc[signals.index[i], 'signal_type'] = '下降周期'
                signals.loc[signals.index[i], 'signal_desc'] = '时间周期处于下降阶段'
                signals.loc[signals.index[i], 'score'] = 35
        
        # 提高主要周期转折点的置信度
        if dominant_cycles:
            # 检查多个主要周期是否同时在转折点
            for i in range(5, len(signals)):
                cycle_alignment = 0
                for j, cycle in enumerate(dominant_cycles[:3]):  # 只考虑前3个主要周期
                    cycle_col = f"cycle_{j+1}_sine"
                    if cycle_col in cycle_result.columns:
                        cycle_sine = cycle_result[cycle_col]
                        # 检查是否在底部拐点
                        if (cycle_sine.iloc[i-1] < 0 and cycle_sine.iloc[i] >= 0):
                            cycle_alignment += 1
                        # 检查是否在顶部拐点
                        elif (cycle_sine.iloc[i-1] > 0 and cycle_sine.iloc[i] <= 0):
                            cycle_alignment -= 1
                
                # 多个周期同时在底部拐点，增强买入信号
                if cycle_alignment >= 2:
                    if signals['buy_signal'].iloc[i]:
                        # 增加置信度
                        signals.loc[signals.index[i], 'confidence'] = min(90, signals['confidence'].iloc[i] + 10)
                        signals.loc[signals.index[i], 'signal_desc'] = '多个时间周期底部共振，强烈买入信号'
                    else:
                        # 生成新的买入信号
                        signals.loc[signals.index[i], 'buy_signal'] = True
                        signals.loc[signals.index[i], 'neutral_signal'] = False
                        signals.loc[signals.index[i], 'trend'] = 1
                        signals.loc[signals.index[i], 'signal_type'] = '周期底部共振'
                        signals.loc[signals.index[i], 'signal_desc'] = '多个时间周期底部共振，强烈买入信号'
                        signals.loc[signals.index[i], 'confidence'] = 80
                        signals.loc[signals.index[i], 'score'] = 80
                
                # 多个周期同时在顶部拐点，增强卖出信号
                elif cycle_alignment <= -2:
                    if signals['sell_signal'].iloc[i]:
                        # 增加置信度
                        signals.loc[signals.index[i], 'confidence'] = min(90, signals['confidence'].iloc[i] + 10)
                        signals.loc[signals.index[i], 'signal_desc'] = '多个时间周期顶部共振，强烈卖出信号'
                    else:
                        # 生成新的卖出信号
                        signals.loc[signals.index[i], 'sell_signal'] = True
                        signals.loc[signals.index[i], 'neutral_signal'] = False
                        signals.loc[signals.index[i], 'trend'] = -1
                        signals.loc[signals.index[i], 'signal_type'] = '周期顶部共振'
                        signals.loc[signals.index[i], 'signal_desc'] = '多个时间周期顶部共振，强烈卖出信号'
                        signals.loc[signals.index[i], 'confidence'] = 80
                        signals.loc[signals.index[i], 'score'] = 20
        
        # 根据当前周期相位调整市场环境
        if current_phase:
            if current_phase in ['上升阶段', '加速上升阶段']:
                signals['market_env'] = 'bull_market'
            elif current_phase in ['顶部阶段', '减速上升阶段']:
                signals['market_env'] = 'top_market'
            elif current_phase in ['下降阶段', '加速下降阶段']:
                signals['market_env'] = 'bear_market'
            elif current_phase in ['底部阶段', '减速下降阶段']:
                signals['market_env'] = 'bottom_market'
        
        # 使用价格走势确认信号
        if 'close' in data.columns:
            close_prices = data['close']
            # 计算短期和长期移动平均线
            ma20 = close_prices.rolling(window=20).mean()
            ma60 = close_prices.rolling(window=60).mean()
            
            # 价格确认趋势
            uptrend = close_prices > ma20
            downtrend = close_prices < ma20
            
            # 使用移动平均线交叉确认信号
            ma_crossover = (ma20.shift(1) <= ma60.shift(1)) & (ma20 > ma60)
            ma_crossunder = (ma20.shift(1) >= ma60.shift(1)) & (ma20 < ma60)
            
            # 增强或减弱信号
            for i in range(len(signals)):
                if signals['buy_signal'].iloc[i]:
                    if uptrend.iloc[i] or ma_crossover.iloc[i]:
                        # 价格趋势确认，增加置信度
                        signals.loc[signals.index[i], 'confidence'] = min(90, signals['confidence'].iloc[i] + 10)
                    elif downtrend.iloc[i]:
                        # 价格趋势矛盾，降低置信度
                        signals.loc[signals.index[i], 'confidence'] = max(30, signals['confidence'].iloc[i] - 20)
                
                if signals['sell_signal'].iloc[i]:
                    if downtrend.iloc[i] or ma_crossunder.iloc[i]:
                        # 价格趋势确认，增加置信度
                        signals.loc[signals.index[i], 'confidence'] = min(90, signals['confidence'].iloc[i] + 10)
                    elif uptrend.iloc[i]:
                        # 价格趋势矛盾，降低置信度
                        signals.loc[signals.index[i], 'confidence'] = max(30, signals['confidence'].iloc[i] - 20)
        
        # 使用成交量确认信号
        if 'volume' in data.columns:
            volume = data['volume']
            vol_ma5 = volume.rolling(window=5).mean()
            vol_ratio = volume / vol_ma5
            
            # 成交量放大确认
            high_volume = vol_ratio > 1.5
            signals.loc[high_volume, 'volume_confirmation'] = True
            
            # 成交量确认增强信号可靠性
            for i in range(len(signals)):
                if (signals['buy_signal'].iloc[i] or signals['sell_signal'].iloc[i]) and high_volume.iloc[i]:
                    current_confidence = signals['confidence'].iloc[i]
                    signals.loc[signals.index[i], 'confidence'] = min(90, current_confidence + 10)
        
        # 更新风险等级和仓位建议
        for i in range(len(signals)):
            confidence = signals['confidence'].iloc[i]
            
            # 根据信号强度和置信度设置风险等级
            if confidence >= 80:
                signals.loc[signals.index[i], 'risk_level'] = '低'
            elif confidence >= 60:
                signals.loc[signals.index[i], 'risk_level'] = '中'
            else:
                signals.loc[signals.index[i], 'risk_level'] = '高'
            
            # 设置建议仓位
            if signals['buy_signal'].iloc[i] or signals['sell_signal'].iloc[i]:
                if confidence >= 80:
                    signals.loc[signals.index[i], 'position_size'] = 0.1  # 10%仓位
                elif confidence >= 70:
                    signals.loc[signals.index[i], 'position_size'] = 0.07  # 7%仓位
                elif confidence >= 60:
                    signals.loc[signals.index[i], 'position_size'] = 0.05  # 5%仓位
        
        # 计算动态止损
        for i in range(len(signals)):
            if signals['buy_signal'].iloc[i]:
                # 买入信号的止损
                if i >= 5 and i < len(data):
                    # 使用前5个交易日的最低价作为止损参考
                    stop_level = data['low'].iloc[i-5:i].min() * 0.98
                    signals.loc[signals.index[i], 'stop_loss'] = stop_level
            
            elif signals['sell_signal'].iloc[i]:
                # 卖出信号的止损
                if i >= 5 and i < len(data):
                    # 使用前5个交易日的最高价作为止损参考
                    stop_level = data['high'].iloc[i-5:i].max() * 1.02
                    signals.loc[signals.index[i], 'stop_loss'] = stop_level
        
        return signals

    def calculate_raw_score(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算时间周期分析的原始评分（0-100分）
        
        Args:
            data: 输入数据，包含OHLCV数据
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            pd.DataFrame: 包含原始评分的DataFrame
        """
        # 初始化评分DataFrame
        scores = pd.DataFrame(index=data.index)
        scores['raw_score'] = 50.0  # 默认评分50分（中性）
        
        # 计算时间周期指标
        cycle_result = self.calculate(data)
        
        # 检查是否有有效的周期结果
        if cycle_result.empty or 'combined_cycle_sine' not in cycle_result.columns:
            return scores  # 如果没有有效的周期结果，返回默认评分
        
        # 获取组合周期正弦波
        combined_sine = cycle_result['combined_cycle_sine']
        
        # 计算组合周期正弦波的一阶和二阶差分，用于判断拐点
        sine_diff1 = combined_sine.diff()
        sine_diff2 = sine_diff1.diff()
        
        # 获取主要周期
        dominant_cycles = self.get_dominant_cycles(data)
        
        # 获取当前周期相位
        current_phase = self.get_current_cycle_phase(data)
        
        # 初始化基于周期的评分
        for i in range(5, len(scores)):
            # 基础评分：基于组合正弦波的值和方向
            base_score = 50 + combined_sine.iloc[i] * 25  # 将-1到1的正弦波值映射到25-75分范围
            
            # 考虑趋势方向（一阶导数）
            if sine_diff1.iloc[i] > 0:
                base_score += 10  # 上升趋势加分
            elif sine_diff1.iloc[i] < 0:
                base_score -= 10  # 下降趋势减分
            
            # 考虑趋势加速度（二阶导数）
            if sine_diff2.iloc[i] > 0:
                base_score += 5  # 加速上升或减速下降加分
            elif sine_diff2.iloc[i] < 0:
                base_score -= 5  # 减速上升或加速下降减分
            
            # 底部拐点（正弦波由负变正，且二阶导数为正）- 强烈买入信号
            if (combined_sine.iloc[i-1] < 0 and combined_sine.iloc[i] >= 0 and 
                sine_diff1.iloc[i] > 0 and sine_diff2.iloc[i] > 0):
                base_score = 75  # 明确的买入信号
            
            # 顶部拐点（正弦波由正变负，且二阶导数为负）- 强烈卖出信号
            elif (combined_sine.iloc[i-1] > 0 and combined_sine.iloc[i] <= 0 and 
                  sine_diff1.iloc[i] < 0 and sine_diff2.iloc[i] < 0):
                base_score = 25  # 明确的卖出信号
            
            # 记录基础评分
            scores.loc[scores.index[i], 'raw_score'] = base_score
        
        # 调整评分：多周期共振
        if dominant_cycles:
            for i in range(5, len(scores)):
                cycle_alignment = 0
                # 检查多个主要周期是否同时在转折点
                for j, cycle in enumerate(dominant_cycles[:3]):  # 只考虑前3个主要周期
                    cycle_col = f"cycle_{j+1}_sine"
                    if cycle_col in cycle_result.columns:
                        cycle_sine = cycle_result[cycle_col]
                        # 检查是否在底部拐点
                        if (cycle_sine.iloc[i-1] < 0 and cycle_sine.iloc[i] >= 0):
                            cycle_alignment += 1
                        # 检查是否在顶部拐点
                        elif (cycle_sine.iloc[i-1] > 0 and cycle_sine.iloc[i] <= 0):
                            cycle_alignment -= 1
                
                # 多个周期同时在底部拐点，强烈买入信号
                if cycle_alignment >= 2:
                    scores.loc[scores.index[i], 'raw_score'] = min(100, scores['raw_score'].iloc[i] + 20)
                # 多个周期同时在顶部拐点，强烈卖出信号
                elif cycle_alignment <= -2:
                    scores.loc[scores.index[i], 'raw_score'] = max(0, scores['raw_score'].iloc[i] - 20)
        
        # 调整评分：周期相位影响
        if current_phase:
            phase_adjustment = 0
            
            if current_phase in ['上升阶段', '加速上升阶段']:
                phase_adjustment = 10  # 上升阶段加分
            elif current_phase in ['顶部阶段', '减速上升阶段']:
                phase_adjustment = -5  # 顶部阶段轻微减分
            elif current_phase in ['下降阶段', '加速下降阶段']:
                phase_adjustment = -10  # 下降阶段减分
            elif current_phase in ['底部阶段', '减速下降阶段']:
                phase_adjustment = 5  # 底部阶段轻微加分
            
            # 应用相位调整
            scores['raw_score'] = scores['raw_score'] + phase_adjustment
        
        # 使用价格趋势确认调整评分
        if 'close' in data.columns:
            close_prices = data['close']
            # 计算短期和长期移动平均线
            ma20 = close_prices.rolling(window=20).mean()
            ma60 = close_prices.rolling(window=60).mean()
            
            # 价格确认趋势
            uptrend = close_prices > ma20
            downtrend = close_prices < ma20
            
            # 使用移动平均线交叉确认信号
            ma_crossover = (ma20.shift(1) <= ma60.shift(1)) & (ma20 > ma60)
            ma_crossunder = (ma20.shift(1) >= ma60.shift(1)) & (ma20 < ma60)
            
            # 价格趋势确认，增强或减弱评分
            for i in range(5, len(scores)):
                current_score = scores['raw_score'].iloc[i]
                
                # 价格上升趋势确认
                if uptrend.iloc[i] or ma_crossover.iloc[i]:
                    if current_score > 50:  # 如果评分已经偏向买入
                        scores.loc[scores.index[i], 'raw_score'] = min(100, current_score + 10)
                    elif current_score < 50:  # 如果评分偏向卖出但价格上升
                        scores.loc[scores.index[i], 'raw_score'] = min(50, current_score + 5)
                
                # 价格下降趋势确认
                elif downtrend.iloc[i] or ma_crossunder.iloc[i]:
                    if current_score < 50:  # 如果评分已经偏向卖出
                        scores.loc[scores.index[i], 'raw_score'] = max(0, current_score - 10)
                    elif current_score > 50:  # 如果评分偏向买入但价格下降
                        scores.loc[scores.index[i], 'raw_score'] = max(50, current_score - 5)
        
        # 限制评分范围在0-100之间
        scores['raw_score'] = scores['raw_score'].clip(0, 100)
        
        return scores

    def identify_patterns(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        识别时间周期分析中的关键形态
        
        Args:
            data: 输入数据，包含OHLCV数据
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            pd.DataFrame: 包含识别出的形态信息的DataFrame
        """
        # 初始化形态DataFrame
        patterns = pd.DataFrame(index=data.index)
        patterns['pattern'] = None
        patterns['pattern_strength'] = 0.0
        patterns['pattern_desc'] = None
        
        # 计算时间周期指标
        cycle_result = self.calculate(data)
        
        # 检查是否有有效的周期结果
        if cycle_result.empty or 'combined_cycle_sine' not in cycle_result.columns:
            return patterns  # 如果没有有效的周期结果，返回空形态
        
        # 获取组合周期正弦波
        combined_sine = cycle_result['combined_cycle_sine']
        
        # 计算组合周期正弦波的一阶和二阶差分，用于判断拐点
        sine_diff1 = combined_sine.diff()
        sine_diff2 = sine_diff1.diff()
        
        # 获取主要周期
        dominant_cycles = self.get_dominant_cycles(data)
        
        # 定义要识别的周期形态
        for i in range(5, len(patterns)):
            # 1. 周期底部拐点
            if (combined_sine.iloc[i-1] < 0 and combined_sine.iloc[i] >= 0 and 
                sine_diff1.iloc[i] > 0 and sine_diff2.iloc[i] > 0):
                patterns.loc[patterns.index[i], 'pattern'] = '周期底部拐点'
                patterns.loc[patterns.index[i], 'pattern_strength'] = 0.8
                patterns.loc[patterns.index[i], 'pattern_desc'] = '时间周期底部拐点，潜在买入机会'
            
            # 2. 周期顶部拐点
            elif (combined_sine.iloc[i-1] > 0 and combined_sine.iloc[i] <= 0 and 
                  sine_diff1.iloc[i] < 0 and sine_diff2.iloc[i] < 0):
                patterns.loc[patterns.index[i], 'pattern'] = '周期顶部拐点'
                patterns.loc[patterns.index[i], 'pattern_strength'] = 0.8
                patterns.loc[patterns.index[i], 'pattern_desc'] = '时间周期顶部拐点，潜在卖出机会'
            
            # 3. 上升周期加速
            elif combined_sine.iloc[i] > 0 and sine_diff1.iloc[i] > 0 and sine_diff2.iloc[i] > 0:
                patterns.loc[patterns.index[i], 'pattern'] = '上升周期加速'
                patterns.loc[patterns.index[i], 'pattern_strength'] = 0.6
                patterns.loc[patterns.index[i], 'pattern_desc'] = '时间周期上升阶段加速，趋势增强'
            
            # 4. 下降周期加速
            elif combined_sine.iloc[i] < 0 and sine_diff1.iloc[i] < 0 and sine_diff2.iloc[i] < 0:
                patterns.loc[patterns.index[i], 'pattern'] = '下降周期加速'
                patterns.loc[patterns.index[i], 'pattern_strength'] = 0.6
                patterns.loc[patterns.index[i], 'pattern_desc'] = '时间周期下降阶段加速，趋势增强'
            
            # 5. 上升减速（可能接近顶部）
            elif combined_sine.iloc[i] > 0 and sine_diff1.iloc[i] > 0 and sine_diff2.iloc[i] < 0:
                patterns.loc[patterns.index[i], 'pattern'] = '上升周期减速'
                patterns.loc[patterns.index[i], 'pattern_strength'] = 0.5
                patterns.loc[patterns.index[i], 'pattern_desc'] = '时间周期上升阶段减速，可能接近顶部'
            
            # 6. 下降减速（可能接近底部）
            elif combined_sine.iloc[i] < 0 and sine_diff1.iloc[i] < 0 and sine_diff2.iloc[i] > 0:
                patterns.loc[patterns.index[i], 'pattern'] = '下降周期减速'
                patterns.loc[patterns.index[i], 'pattern_strength'] = 0.5
                patterns.loc[patterns.index[i], 'pattern_desc'] = '时间周期下降阶段减速，可能接近底部'
        
        # 识别多周期共振形态
        if dominant_cycles:
            for i in range(5, len(patterns)):
                cycle_alignment = 0
                aligned_cycles = []
                
                # 检查多个主要周期是否同时在转折点
                for j, cycle in enumerate(dominant_cycles[:3]):  # 只考虑前3个主要周期
                    cycle_col = f"cycle_{j+1}_sine"
                    if cycle_col in cycle_result.columns:
                        cycle_sine = cycle_result[cycle_col]
                        # 检查是否在底部拐点
                        if (cycle_sine.iloc[i-1] < 0 and cycle_sine.iloc[i] >= 0):
                            cycle_alignment += 1
                            aligned_cycles.append(cycle['length'])
                        # 检查是否在顶部拐点
                        elif (cycle_sine.iloc[i-1] > 0 and cycle_sine.iloc[i] <= 0):
                            cycle_alignment -= 1
                            aligned_cycles.append(cycle['length'])
                
                # 多个周期同时在底部拐点，形成底部共振
                if cycle_alignment >= 2:
                    patterns.loc[patterns.index[i], 'pattern'] = '周期底部共振'
                    patterns.loc[patterns.index[i], 'pattern_strength'] = 0.9
                    patterns.loc[patterns.index[i], 'pattern_desc'] = f'多个时间周期底部共振（{", ".join([str(c) for c in aligned_cycles])}日），强烈买入信号'
                
                # 多个周期同时在顶部拐点，形成顶部共振
                elif cycle_alignment <= -2:
                    patterns.loc[patterns.index[i], 'pattern'] = '周期顶部共振'
                    patterns.loc[patterns.index[i], 'pattern_strength'] = 0.9
                    patterns.loc[patterns.index[i], 'pattern_desc'] = f'多个时间周期顶部共振（{", ".join([str(c) for c in aligned_cycles])}日），强烈卖出信号'
        
        # 识别重复周期形态 - 寻找规律性出现的高低点
        if len(dominant_cycles) > 0:
            main_cycle = dominant_cycles[0]['length']
            # 检查是否有明显的重复周期
            if main_cycle > 10 and main_cycle < 100:  # 忽略太短或太长的周期
                # 查找周期性高低点
                cycle_peaks = []
                cycle_troughs = []
                
                for i in range(5, len(patterns)-5):
                    # 查找局部高点
                    if (combined_sine.iloc[i] > combined_sine.iloc[i-1] and
                        combined_sine.iloc[i] > combined_sine.iloc[i+1] and
                        combined_sine.iloc[i] > 0.5):
                        cycle_peaks.append(i)
                    
                    # 查找局部低点
                    if (combined_sine.iloc[i] < combined_sine.iloc[i-1] and
                        combined_sine.iloc[i] < combined_sine.iloc[i+1] and
                        combined_sine.iloc[i] < -0.5):
                        cycle_troughs.append(i)
                
                # 检查高点间距是否符合主周期
                if len(cycle_peaks) >= 2:
                    peak_intervals = [cycle_peaks[j] - cycle_peaks[j-1] for j in range(1, len(cycle_peaks))]
                    avg_peak_interval = sum(peak_intervals) / len(peak_intervals)
                    
                    # 如果平均间距接近主周期，标记为重复高点周期
                    if abs(avg_peak_interval - main_cycle) < main_cycle * 0.2:
                        for peak in cycle_peaks:
                            patterns.loc[patterns.index[peak], 'pattern'] = '重复周期高点'
                            patterns.loc[patterns.index[peak], 'pattern_strength'] = 0.7
                            patterns.loc[patterns.index[peak], 'pattern_desc'] = f'重复周期高点（{main_cycle:.0f}日周期），潜在卖出时机'
                
                # 检查低点间距是否符合主周期
                if len(cycle_troughs) >= 2:
                    trough_intervals = [cycle_troughs[j] - cycle_troughs[j-1] for j in range(1, len(cycle_troughs))]
                    avg_trough_interval = sum(trough_intervals) / len(trough_intervals)
                    
                    # 如果平均间距接近主周期，标记为重复低点周期
                    if abs(avg_trough_interval - main_cycle) < main_cycle * 0.2:
                        for trough in cycle_troughs:
                            patterns.loc[patterns.index[trough], 'pattern'] = '重复周期低点'
                            patterns.loc[patterns.index[trough], 'pattern_strength'] = 0.7
                            patterns.loc[patterns.index[trough], 'pattern_desc'] = f'重复周期低点（{main_cycle:.0f}日周期），潜在买入时机'
        
        # 识别波浪形态（连续的上升-下降-上升或下降-上升-下降）
        for i in range(20, len(patterns)):
            recent_sine = combined_sine.iloc[i-20:i+1]
            
            # 如果有至少3个交替的上升和下降段，可能是波浪形态
            zero_crossings = ((recent_sine.shift(1) * recent_sine) < 0).sum()
            if zero_crossings >= 3:
                # 计算波浪的规律性
                sine_peaks = recent_sine[
                    (recent_sine > recent_sine.shift(1)) & 
                    (recent_sine > recent_sine.shift(-1))
                ]
                sine_troughs = recent_sine[
                    (recent_sine < recent_sine.shift(1)) & 
                    (recent_sine < recent_sine.shift(-1))
                ]
                
                # 如果波峰和波谷分布均匀，更可能是规则的波浪形态
                if len(sine_peaks) >= 2 and len(sine_troughs) >= 2:
                    peak_intervals = np.diff(sine_peaks.index)
                    trough_intervals = np.diff(sine_troughs.index)
                    
                    if peak_intervals.std() / peak_intervals.mean() < 0.3 and trough_intervals.std() / trough_intervals.mean() < 0.3:
                        patterns.loc[patterns.index[i], 'pattern'] = '周期波浪形态'
                        patterns.loc[patterns.index[i], 'pattern_strength'] = 0.6
                        patterns.loc[patterns.index[i], 'pattern_desc'] = '规则的周期波浪形态，可用于节奏交易'
        
        # 使用价格走势确认形态
        if 'close' in data.columns and 'high' in data.columns and 'low' in data.columns:
            close_prices = data['close']
            highs = data['high']
            lows = data['low']
            
            # 价格形态与周期形态确认
            for i in range(20, len(patterns)):
                if patterns['pattern'].iloc[i] in ['周期底部拐点', '周期底部共振', '下降周期减速', '重复周期低点']:
                    # 检查价格是否形成低点并反弹
                    price_low_formed = (
                        lows.iloc[i-1] > lows.iloc[i] and 
                        lows.iloc[i] < lows.iloc[i+1] and
                        close_prices.iloc[i+1] > close_prices.iloc[i]
                    )
                    if price_low_formed:
                        # 增强形态强度
                        patterns.loc[patterns.index[i], 'pattern_strength'] = min(1.0, patterns['pattern_strength'].iloc[i] + 0.2)
                        patterns.loc[patterns.index[i], 'pattern_desc'] = patterns['pattern_desc'].iloc[i] + '，价格形态确认'
                
                elif patterns['pattern'].iloc[i] in ['周期顶部拐点', '周期顶部共振', '上升周期减速', '重复周期高点']:
                    # 检查价格是否形成高点并回落
                    price_high_formed = (
                        highs.iloc[i-1] < highs.iloc[i] and 
                        highs.iloc[i] > highs.iloc[i+1] and
                        close_prices.iloc[i+1] < close_prices.iloc[i]
                    )
                    if price_high_formed:
                        # 增强形态强度
                        patterns.loc[patterns.index[i], 'pattern_strength'] = min(1.0, patterns['pattern_strength'].iloc[i] + 0.2)
                        patterns.loc[patterns.index[i], 'pattern_desc'] = patterns['pattern_desc'].iloc[i] + '，价格形态确认'
        
        return patterns 