"""
艾略特波浪理论分析模块

实现波浪识别和预测功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple
from enum import Enum

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger

logger = get_logger(__name__)


class WaveDirection(Enum):
    """波浪方向枚举"""
    UP = 1    # 上升波浪
    DOWN = -1  # 下降波浪


class WaveType(Enum):
    """波浪类型枚举"""
    IMPULSE = "推动浪"    # 推动浪(1,3,5浪)
    CORRECTIVE = "调整浪"  # 调整浪(2,4浪)
    SUBWAVE = "子浪"      # 子浪


class WavePattern(Enum):
    """波浪形态枚举"""
    FIVE_WAVE = "五浪结构"        # 标准五浪结构
    ZIG_ZAG = "锯齿形调整"         # 锯齿形调整(5-3-5)
    FLAT = "平台形调整"            # 平台形调整(3-3-5)
    TRIANGLE = "三角形调整"        # 三角形调整(3-3-3-3-3)
    DIAGONAL = "对角线三角形"      # 对角线三角形
    COMBINATION = "组合调整"       # 组合调整
    UNKNOWN = "未知形态"           # 未知形态


class ElliottWave(BaseIndicator):
    """
    艾略特波浪理论分析指标
    
    识别和分析价格波浪结构，预测可能的波浪发展
    """
    
    def __init__(self):
        """初始化艾略特波浪理论分析指标"""
        super().__init__(name="ElliottWave", description="艾略特波浪理论分析指标，识别和分析价格波浪结构")
    
    def calculate(self, data: pd.DataFrame, min_wave_height: float = 0.03, 
                max_wave_count: int = 9, *args, **kwargs) -> pd.DataFrame:
        """
        分析艾略特波浪结构
        
        Args:
            data: 输入数据，包含OHLC数据
            min_wave_height: 最小波浪高度（相对于价格的百分比），默认为3%
            max_wave_count: 最大波浪数量，默认为9（完整的5+4波浪）
            
        Returns:
            pd.DataFrame: 计算结果，包含波浪结构信息
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["high", "low", "close"])
        
        # 初始化结果数据框
        result = pd.DataFrame(index=data.index)
        
        # 识别摆动点
        swing_points = self._identify_swing_points(data, min_wave_height)
        
        # 生成波浪
        waves = self._generate_waves(data, swing_points, max_wave_count)
        
        # 识别波浪形态
        wave_pattern, wave_labels = self._identify_wave_pattern(waves)
        
        # 添加到结果
        result = self._add_waves_to_result(result, waves, wave_labels)
        result["wave_pattern"] = wave_pattern.value if wave_pattern else WavePattern.UNKNOWN.value
        
        # 预测下一波浪
        next_wave_prediction = self._predict_next_wave(data, waves, wave_pattern)
        if next_wave_prediction is not None:
            result["next_wave_prediction"] = next_wave_prediction
        
        return result
    
    def _identify_swing_points(self, data: pd.DataFrame, min_wave_height: float) -> List[int]:
        """
        识别价格摆动点
        
        Args:
            data: 输入数据
            min_wave_height: 最小波浪高度
            
        Returns:
            List[int]: 摆动点索引列表
        """
        n = len(data)
        swing_points = []
        
        # 添加起始点
        swing_points.append(0)
        
        # 标记本地极大值和极小值
        for i in range(1, n-1):
            # 局部极大值
            if data["high"].iloc[i] > data["high"].iloc[i-1] and data["high"].iloc[i] > data["high"].iloc[i+1]:
                # 检查波浪高度
                if i > 0 and swing_points:
                    last_point = swing_points[-1]
                    height_pct = abs(data["high"].iloc[i] - data["low"].iloc[last_point]) / data["low"].iloc[last_point]
                    if height_pct >= min_wave_height:
                        swing_points.append(i)
            
            # 局部极小值
            elif data["low"].iloc[i] < data["low"].iloc[i-1] and data["low"].iloc[i] < data["low"].iloc[i+1]:
                # 检查波浪高度
                if i > 0 and swing_points:
                    last_point = swing_points[-1]
                    height_pct = abs(data["low"].iloc[i] - data["high"].iloc[last_point]) / data["high"].iloc[last_point]
                    if height_pct >= min_wave_height:
                        swing_points.append(i)
        
        # 添加结束点
        if n-1 not in swing_points:
            swing_points.append(n-1)
        
        return swing_points
    
    def _generate_waves(self, data: pd.DataFrame, swing_points: List[int], max_wave_count: int) -> List[Dict]:
        """
        生成波浪
        
        Args:
            data: 输入数据
            swing_points: 摆动点索引列表
            max_wave_count: 最大波浪数量
            
        Returns:
            List[Dict]: 波浪列表
        """
        waves = []
        
        # 确保至少有两个点
        if len(swing_points) < 2:
            return waves
        
        # 生成波浪
        for i in range(1, min(len(swing_points), max_wave_count + 1)):
            start_idx = swing_points[i-1]
            end_idx = swing_points[i]
            
            # 判断波浪方向
            if data["close"].iloc[end_idx] > data["close"].iloc[start_idx]:
                direction = WaveDirection.UP
            else:
                direction = WaveDirection.DOWN
            
            # 计算波浪高度
            if direction == WaveDirection.UP:
                height = data["high"].iloc[end_idx] - data["low"].iloc[start_idx]
                height_pct = height / data["low"].iloc[start_idx]
            else:
                height = data["high"].iloc[start_idx] - data["low"].iloc[end_idx]
                height_pct = height / data["high"].iloc[start_idx]
            
            # 计算波浪时间
            time_length = end_idx - start_idx
            
            # 创建波浪
            wave = {
                "start_idx": start_idx,
                "end_idx": end_idx,
                "direction": direction,
                "height": height,
                "height_pct": height_pct,
                "time_length": time_length,
                "wave_number": i
            }
            
            waves.append(wave)
        
        return waves
    
    def _identify_wave_pattern(self, waves: List[Dict]) -> Tuple[Optional[WavePattern], List[str]]:
        """
        识别波浪形态
        
        Args:
            waves: 波浪列表
            
        Returns:
            Tuple[Optional[WavePattern], List[str]]: (波浪形态, 波浪标签列表)
        """
        if not waves:
            return None, []
        
        # 根据波浪数量和方向识别形态
        wave_count = len(waves)
        wave_labels = [""] * wave_count
        
        # 检查是否是五浪结构（3上升，2下降）
        if wave_count >= 5:
            # 检查方向：1上，2下，3上，4下，5上
            if (waves[0]["direction"] == WaveDirection.UP and
                waves[1]["direction"] == WaveDirection.DOWN and
                waves[2]["direction"] == WaveDirection.UP and
                waves[3]["direction"] == WaveDirection.DOWN and
                waves[4]["direction"] == WaveDirection.UP):
                
                # 检查波浪3是否是最长的推动浪
                if waves[2]["height_pct"] > waves[0]["height_pct"] and waves[2]["height_pct"] > waves[4]["height_pct"]:
                    # 标记波浪
                    wave_labels[0] = "1"
                    wave_labels[1] = "2"
                    wave_labels[2] = "3"
                    wave_labels[3] = "4"
                    wave_labels[4] = "5"
                    
                    # 标记后续的调整浪
                    if wave_count > 5:
                        if waves[5]["direction"] == WaveDirection.DOWN:
                            wave_labels[5] = "A"
                            if wave_count > 6 and waves[6]["direction"] == WaveDirection.UP:
                                wave_labels[6] = "B"
                                if wave_count > 7 and waves[7]["direction"] == WaveDirection.DOWN:
                                    wave_labels[7] = "C"
                    
                    return WavePattern.FIVE_WAVE, wave_labels
        
        # 检查是否是锯齿形调整（A下，B上，C下）
        if wave_count >= 3:
            if (waves[0]["direction"] == WaveDirection.DOWN and
                waves[1]["direction"] == WaveDirection.UP and
                waves[2]["direction"] == WaveDirection.DOWN):
                
                # 检查B浪不超过A浪起点
                if wave_count > 1:
                    start_price_a = waves[0]["end_idx"]
                    end_price_b = waves[1]["end_idx"]
                    if end_price_b <= start_price_a:
                        wave_labels[0] = "A"
                        wave_labels[1] = "B"
                        wave_labels[2] = "C"
                        return WavePattern.ZIG_ZAG, wave_labels
        
        # 检查是否是平台形调整（A下，B上接近A起点，C下）
        if wave_count >= 3:
            if (waves[0]["direction"] == WaveDirection.DOWN and
                waves[1]["direction"] == WaveDirection.UP and
                waves[2]["direction"] == WaveDirection.DOWN):
                
                # 检查B浪接近A浪起点
                if wave_count > 1:
                    start_price_a = waves[0]["end_idx"]
                    end_price_b = waves[1]["end_idx"]
                    if abs(end_price_b - start_price_a) / start_price_a < 0.03:
                        wave_labels[0] = "A"
                        wave_labels[1] = "B"
                        wave_labels[2] = "C"
                        return WavePattern.FLAT, wave_labels
        
        # 检查是否是三角形调整（五个递减波浪）
        if wave_count >= 5:
            # 检查波浪高度递减
            is_triangle = True
            for i in range(1, 5):
                if waves[i]["height_pct"] >= waves[i-1]["height_pct"]:
                    is_triangle = False
                    break
            
            if is_triangle:
                wave_labels[0] = "A"
                wave_labels[1] = "B"
                wave_labels[2] = "C"
                wave_labels[3] = "D"
                wave_labels[4] = "E"
                return WavePattern.TRIANGLE, wave_labels
        
        # 默认未知形态
        for i in range(wave_count):
            wave_labels[i] = str(i+1)
        
        return WavePattern.UNKNOWN, wave_labels
    
    def _add_waves_to_result(self, result: pd.DataFrame, waves: List[Dict], wave_labels: List[str]) -> pd.DataFrame:
        """
        将波浪信息添加到结果数据框
        
        Args:
            result: 结果数据框
            waves: 波浪列表
            wave_labels: 波浪标签列表
            
        Returns:
            pd.DataFrame: 更新后的结果数据框
        """
        # 初始化波浪列
        result["wave_number"] = np.nan
        result["wave_label"] = ""
        result["wave_start"] = np.nan
        result["wave_end"] = np.nan
        result["wave_direction"] = np.nan
        
        # 添加波浪信息
        for i, wave in enumerate(waves):
            start_idx = wave["start_idx"]
            end_idx = wave["end_idx"]
            label = wave_labels[i] if i < len(wave_labels) else str(i+1)
            
            # 标记波浪起点
            result.iloc[start_idx, result.columns.get_loc("wave_start")] = 1
            result.iloc[start_idx, result.columns.get_loc("wave_label")] = label
            result.iloc[start_idx, result.columns.get_loc("wave_number")] = i+1
            result.iloc[start_idx, result.columns.get_loc("wave_direction")] = wave["direction"].value
            
            # 标记波浪结束点
            result.iloc[end_idx, result.columns.get_loc("wave_end")] = 1
            
            # 标记波浪内的所有点
            for j in range(start_idx, end_idx + 1):
                if pd.isna(result.iloc[j, result.columns.get_loc("wave_number")]):
                    result.iloc[j, result.columns.get_loc("wave_number")] = i+1
                    result.iloc[j, result.columns.get_loc("wave_label")] = label
                    result.iloc[j, result.columns.get_loc("wave_direction")] = wave["direction"].value
        
        return result
    
    def _predict_next_wave(self, data: pd.DataFrame, waves: List[Dict], wave_pattern: Optional[WavePattern]) -> Optional[float]:
        """
        预测下一个波浪
        
        Args:
            data: 输入数据
            waves: 波浪列表
            wave_pattern: 波浪形态
            
        Returns:
            Optional[float]: 预测的下一波浪价格目标
        """
        if not waves or wave_pattern is None:
            return None
        
        # 获取最后一个波浪
        last_wave = waves[-1]
        last_price = data["close"].iloc[-1]
        
        # 根据波浪形态和当前位置预测下一波浪
        if wave_pattern == WavePattern.FIVE_WAVE:
            # 根据已完成的波浪数量预测
            if len(waves) == 5:  # 完成了5浪推动结构，预测A浪
                # A浪通常回调到4浪区域
                if len(waves) >= 4:
                    wave_4_low = data["low"].iloc[waves[3]["end_idx"]]
                    return wave_4_low
            elif len(waves) == 6:  # 完成了A浪，预测B浪
                # B浪通常回调到A浪起点的50-62%
                wave_5_high = data["high"].iloc[waves[4]["end_idx"]]
                wave_a_low = data["low"].iloc[waves[5]["end_idx"]]
                retracement = wave_5_high - wave_a_low
                return wave_a_low + retracement * 0.618
            elif len(waves) == 7:  # 完成了B浪，预测C浪
                # C浪通常达到A浪的1-1.618倍
                wave_a_length = waves[5]["height"]
                wave_b_high = data["high"].iloc[waves[6]["end_idx"]]
                return wave_b_high - wave_a_length * 1.618
        
        elif wave_pattern == WavePattern.ZIG_ZAG:
            # 锯齿形调整完成后，通常开始新的推动浪
            if len(waves) == 3:  # 完成了A-B-C调整
                # 新的推动浪通常超过之前的高点
                if waves[0]["direction"] == WaveDirection.DOWN:  # 下跌锯齿形
                    start_idx = waves[0]["start_idx"]
                    prev_high = data["high"].iloc[start_idx]
                    return prev_high * 1.1
        
        # 默认预测：使用斐波那契延展
        if last_wave["direction"] == WaveDirection.UP:
            # 预测回调
            first_wave_idx = waves[0]["start_idx"]
            first_price = data["low"].iloc[first_wave_idx]
            last_wave_height = last_wave["height"]
            return last_price - last_wave_height * 0.618
        else:
            # 预测反弹
            first_wave_idx = waves[0]["start_idx"]
            first_price = data["high"].iloc[first_wave_idx]
            last_wave_height = last_wave["height"]
            return last_price + last_wave_height * 0.618
    
    def plot_elliott_waves(self, data: pd.DataFrame, result: pd.DataFrame, ax=None):
        """
        绘制艾略特波浪
        
        Args:
            data: 输入数据
            result: 波浪分析结果
            ax: matplotlib轴对象
            
        Returns:
            matplotlib.axes.Axes: 轴对象
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            
            if ax is None:
                fig, ax = plt.subplots(figsize=(12, 6))
            
            # 绘制K线图
            ax.plot(data.index, data["close"], 'k-', linewidth=1)
            
            # 绘制波浪起点和终点
            wave_starts = result[result["wave_start"] == 1]
            wave_ends = result[result["wave_end"] == 1]
            
            # 绘制波浪起点
            for idx, row in wave_starts.iterrows():
                if row["wave_direction"] == WaveDirection.UP.value:
                    price = data.loc[idx, "low"]
                    marker = '^'
                    color = 'g'
                else:
                    price = data.loc[idx, "high"]
                    marker = 'v'
                    color = 'r'
                
                ax.scatter(idx, price, marker=marker, color=color, s=100)
                ax.annotate(row["wave_label"], (idx, price), 
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=10, fontweight='bold')
            
            # 绘制波浪终点
            for idx, row in wave_ends.iterrows():
                if idx in wave_starts.index:
                    continue  # 跳过同时是起点的终点
                
                if result.loc[idx, "wave_direction"] == WaveDirection.UP.value:
                    price = data.loc[idx, "high"]
                    marker = '^'
                    color = 'g'
                else:
                    price = data.loc[idx, "low"]
                    marker = 'v'
                    color = 'r'
                
                ax.scatter(idx, price, marker=marker, color=color, s=100)
            
            # 连接波浪点
            prev_idx = None
            prev_price = None
            
            for idx, row in pd.concat([wave_starts, wave_ends]).sort_index().iterrows():
                if "wave_start" in row and row["wave_start"] == 1:
                    if row["wave_direction"] == WaveDirection.UP.value:
                        price = data.loc[idx, "low"]
                    else:
                        price = data.loc[idx, "high"]
                elif "wave_end" in row and row["wave_end"] == 1:
                    if row["wave_direction"] == WaveDirection.UP.value:
                        price = data.loc[idx, "high"]
                    else:
                        price = data.loc[idx, "low"]
                else:
                    continue
                
                if prev_idx is not None and prev_price is not None:
                    ax.plot([prev_idx, idx], [prev_price, price], 'b-', linewidth=2)
                
                prev_idx = idx
                prev_price = price
            
            # 添加波浪形态标注
            if "wave_pattern" in result.columns:
                pattern = result["wave_pattern"].iloc[0]
                ax.set_title(f'Elliott Wave Analysis - {pattern}')
            else:
                ax.set_title('Elliott Wave Analysis')
            
            # 添加预测
            if "next_wave_prediction" in result.columns and not pd.isna(result["next_wave_prediction"].iloc[0]):
                prediction = result["next_wave_prediction"].iloc[0]
                ax.axhline(y=prediction, color='purple', linestyle='--', 
                         label=f'Next Wave Target: {prediction:.2f}')
                ax.legend(loc='best')
            
            # 设置日期格式
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            
            return ax
        
        except ImportError:
            logger.error("绘图需要安装matplotlib库")
            return None 