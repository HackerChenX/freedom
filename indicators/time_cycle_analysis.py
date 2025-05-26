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
        result = pd.DataFrame(index=data.index)
        
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