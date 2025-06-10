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
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """初始化艾略特波浪理论分析指标"""
        super().__init__(name="ElliottWave", description="艾略特波浪理论分析指标，识别和分析价格波浪结构")
    
    def _calculate(self, data: pd.DataFrame, min_wave_height: float = 0.03, 
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
        result = data.copy()
        
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
    
    def calculate_raw_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算艾略特波浪理论指标的原始评分
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            pd.DataFrame: 包含原始评分的DataFrame
        """
        # 初始化评分
        score = pd.Series(50.0, index=data.index)  # 基础分50分
        
        if len(data) < 20:
            return pd.DataFrame({'score': score}, index=data.index)
        
        # 计算波浪分析
        try:
            wave_result = self.calculate(data)
        except Exception as e:
            logger.warning(f"计算艾略特波浪失败: {e}")
            return pd.DataFrame({'score': score}, index=data.index)
        
        if wave_result.empty:
            return pd.DataFrame({'score': score}, index=data.index)
        
        close_price = data['close']
        latest_close = close_price.iloc[-1]
        
        # 1. 波浪形态完整性评分（±30分）
        if 'wave_pattern' in wave_result.columns:
            pattern = wave_result['wave_pattern'].iloc[0]
            
            if pattern == WavePattern.FIVE_WAVE.value:
                # 五浪结构是最强的推动形态
                # 检查当前在第几浪
                if 'wave_number' in wave_result.columns:
                    latest_wave = wave_result['wave_number'].iloc[-1]
                    
                    if pd.notna(latest_wave):
                        if latest_wave in [1, 3, 5]:  # 推动浪
                            score.iloc[-10:] += 25
                        elif latest_wave in [2, 4]:  # 调整浪
                            score.iloc[-10:] += 10
                        elif latest_wave > 5:  # 调整阶段
                            if latest_wave in [6, 8]:  # A浪、C浪
                                score.iloc[-10:] -= 20
                            elif latest_wave == 7:  # B浪
                                score.iloc[-10:] -= 10
            
            elif pattern == WavePattern.ZIG_ZAG.value:
                # 锯齿形调整，通常是强烈的调整形态
                score.iloc[-10:] -= 15
            
            elif pattern == WavePattern.FLAT.value:
                # 平台形调整，相对温和
                score.iloc[-10:] -= 8
            
            elif pattern == WavePattern.TRIANGLE.value:
                # 三角形调整，通常预示突破
                score.iloc[-10:] += 5
            
            elif pattern == WavePattern.DIAGONAL.value:
                # 对角线三角形，通常是结束形态
                score.iloc[-10:] -= 12
        
        # 2. 波浪位置评分（±25分）
        if 'wave_number' in wave_result.columns and 'wave_direction' in wave_result.columns:
            latest_wave_num = wave_result['wave_number'].iloc[-1]
            latest_wave_dir = wave_result['wave_direction'].iloc[-1]
            
            if pd.notna(latest_wave_num) and pd.notna(latest_wave_dir):
                # 第3浪通常是最强的推动浪
                if latest_wave_num == 3 and latest_wave_dir == WaveDirection.UP.value:
                    score.iloc[-5:] += 25
                elif latest_wave_num == 3 and latest_wave_dir == WaveDirection.DOWN.value:
                    score.iloc[-5:] -= 25
                
                # 第5浪是最后的推动浪，可能出现背离
                elif latest_wave_num == 5:
                    if latest_wave_dir == WaveDirection.UP.value:
                        score.iloc[-5:] += 15  # 仍然看涨，但力度减弱
                    else:
                        score.iloc[-5:] -= 15
                
                # 第1浪是新趋势的开始
                elif latest_wave_num == 1:
                    if latest_wave_dir == WaveDirection.UP.value:
                        score.iloc[-5:] += 20
                    else:
                        score.iloc[-5:] -= 20
                
                # 调整浪（2、4浪）
                elif latest_wave_num in [2, 4]:
                    if latest_wave_dir == WaveDirection.DOWN.value:
                        score.iloc[-5:] -= 10  # 上升趋势中的调整
                    else:
                        score.iloc[-5:] += 10  # 下降趋势中的反弹
        
        # 3. 波浪预测目标评分（±20分）
        if 'next_wave_prediction' in wave_result.columns:
            prediction = wave_result['next_wave_prediction'].iloc[-1]
            
            if pd.notna(prediction):
                # 计算当前价格与预测目标的距离
                distance_pct = abs(latest_close - prediction) / latest_close
                
                # 如果接近预测目标（5%以内），给予相应评分
                if distance_pct < 0.05:
                    if prediction > latest_close:  # 预测上涨目标
                        score.iloc[-3:] += 20
                    else:  # 预测下跌目标
                        score.iloc[-3:] -= 20
                elif distance_pct < 0.10:  # 10%以内
                    if prediction > latest_close:
                        score.iloc[-3:] += 15
                    else:
                        score.iloc[-3:] -= 15
                elif distance_pct < 0.15:  # 15%以内
                    if prediction > latest_close:
                        score.iloc[-3:] += 10
                    else:
                        score.iloc[-3:] -= 10
        
        # 4. 波浪比例关系评分（±15分）
        # 检查波浪是否符合斐波那契比例关系
        if 'wave_number' in wave_result.columns and len(data) >= 30:
            # 获取波浪起点和终点
            wave_starts = wave_result[wave_result['wave_start'] == 1]
            wave_ends = wave_result[wave_result['wave_end'] == 1]
            
            if len(wave_starts) >= 3:  # 至少有3个波浪
                # 检查第3浪与第1浪的比例关系
                try:
                    wave_1_start = wave_starts.index[0]
                    wave_1_end = wave_starts.index[1] if len(wave_starts) > 1 else wave_ends.index[0]
                    wave_3_start = wave_starts.index[1] if len(wave_starts) > 1 else wave_1_end
                    wave_3_end = wave_starts.index[2] if len(wave_starts) > 2 else wave_ends.index[-1]
                    
                    # 计算波浪长度
                    wave_1_length = abs(data['close'].loc[wave_1_end] - data['close'].loc[wave_1_start])
                    wave_3_length = abs(data['close'].loc[wave_3_end] - data['close'].loc[wave_3_start])
                    
                    if wave_1_length > 0:
                        ratio = wave_3_length / wave_1_length
                        
                        # 检查是否符合斐波那契比例（1.618, 2.618等）
                        fib_ratios = [1.0, 1.618, 2.618, 4.236]
                        for fib_ratio in fib_ratios:
                            if abs(ratio - fib_ratio) / fib_ratio < 0.1:  # 10%误差范围
                                if fib_ratio >= 1.618:  # 强势比例
                                    score.iloc[-5:] += 15
                                else:  # 等长比例
                                    score.iloc[-5:] += 10
                                break
                
                except (IndexError, KeyError):
                    pass  # 忽略计算错误
        
        # 5. 成交量确认评分（±12分）
        if 'volume' in data.columns and 'wave_number' in wave_result.columns:
            volume = data['volume']
            vol_ma10 = volume.rolling(window=10).mean()
            latest_vol_ratio = (volume / vol_ma10).iloc[-1]
            latest_wave_num = wave_result['wave_number'].iloc[-1]
            latest_wave_dir = wave_result['wave_direction'].iloc[-1]
            
            if pd.notna(latest_vol_ratio) and pd.notna(latest_wave_num) and pd.notna(latest_wave_dir):
                # 推动浪应该伴随放量
                if latest_wave_num in [1, 3, 5] and latest_vol_ratio > 1.2:
                    if latest_wave_dir == WaveDirection.UP.value:
                        score.iloc[-3:] += 12
                    else:
                        score.iloc[-3:] -= 12
                
                # 调整浪通常伴随缩量
                elif latest_wave_num in [2, 4] and latest_vol_ratio < 0.8:
                    score.iloc[-3:] += 8
                
                # 第5浪如果出现量价背离，警示顶部
                elif latest_wave_num == 5 and latest_vol_ratio < 0.9:
                    if latest_wave_dir == WaveDirection.UP.value:
                        score.iloc[-3:] -= 10  # 量价背离，警示顶部
        
        # 6. 波浪时间关系评分（±10分）
        # 检查波浪的时间比例关系
        if 'wave_start' in wave_result.columns and 'wave_end' in wave_result.columns:
            wave_starts = wave_result[wave_result['wave_start'] == 1]
            
            if len(wave_starts) >= 2:
                try:
                    # 计算最近两个波浪的时间长度
                    recent_waves = wave_starts.tail(2)
                    wave_indices = recent_waves.index.tolist()
                    
                    if len(wave_indices) >= 2:
                        wave_1_time = wave_indices[1] - wave_indices[0]
                        
                        # 如果有第三个波浪的开始
                        if len(wave_starts) >= 3:
                            wave_2_start = wave_indices[1]
                            current_time = len(data) - 1 - wave_2_start
                            
                            # 检查时间比例关系
                            if wave_1_time > 0:
                                time_ratio = current_time / wave_1_time
                                
                                # 斐波那契时间比例
                                fib_time_ratios = [0.618, 1.0, 1.618]
                                for fib_ratio in fib_time_ratios:
                                    if abs(time_ratio - fib_ratio) / fib_ratio < 0.2:  # 20%误差
                                        score.iloc[-2:] += 10
                                        break
                
                except (IndexError, KeyError):
                    pass
        
        # 7. 波浪完成度评分（±8分）
        # 根据当前波浪的完成程度给分
        if 'wave_number' in wave_result.columns:
            latest_wave_num = wave_result['wave_number'].iloc[-1]
            
            if pd.notna(latest_wave_num):
                # 完整的5浪结构
                if latest_wave_num == 5:
                    score.iloc[-2:] += 8  # 推动浪即将完成
                elif latest_wave_num > 5:
                    # 调整浪阶段
                    if latest_wave_num == 8:  # C浪完成
                        score.iloc[-2:] += 8  # 调整即将结束
                    elif latest_wave_num in [6, 7]:  # A浪、B浪
                        score.iloc[-2:] -= 5  # 仍在调整中
        
        # 确保评分在0-100范围内
        score = score.clip(0, 100)
        
        return pd.DataFrame({'score': score}, index=data.index)
    
    def identify_patterns(self, data: pd.DataFrame) -> List[str]:
        """
        识别艾略特波浪理论相关的技术形态
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            List[str]: 识别出的形态列表
        """
        patterns = []
        
        if len(data) < 20:
            return patterns
        
        # 计算波浪分析
        try:
            wave_result = self.calculate(data)
        except Exception as e:
            logger.warning(f"计算艾略特波浪失败: {e}")
            return patterns
        
        if wave_result.empty:
            return patterns
        
        # 1. 波浪形态识别
        if 'wave_pattern' in wave_result.columns:
            pattern = wave_result['wave_pattern'].iloc[0]
            patterns.append(f"艾略特波浪形态-{pattern}")
            
            # 具体形态分析
            if pattern == WavePattern.FIVE_WAVE.value:
                patterns.append("五浪推动结构")
            elif pattern == WavePattern.ZIG_ZAG.value:
                patterns.append("锯齿形调整")
            elif pattern == WavePattern.FLAT.value:
                patterns.append("平台形调整")
            elif pattern == WavePattern.TRIANGLE.value:
                patterns.append("三角形调整")
            elif pattern == WavePattern.DIAGONAL.value:
                patterns.append("对角线三角形")
            elif pattern == WavePattern.COMBINATION.value:
                patterns.append("组合调整")
        
        # 2. 当前波浪位置识别
        if 'wave_number' in wave_result.columns and 'wave_direction' in wave_result.columns:
            latest_wave_num = wave_result['wave_number'].iloc[-1]
            latest_wave_dir = wave_result['wave_direction'].iloc[-1]
            
            if pd.notna(latest_wave_num) and pd.notna(latest_wave_dir):
                direction_str = "上升" if latest_wave_dir == WaveDirection.UP.value else "下降"
                patterns.append(f"当前波浪-第{int(latest_wave_num)}浪({direction_str})")
                
                # 特殊波浪位置
                if latest_wave_num == 1:
                    patterns.append("新趋势起始浪")
                elif latest_wave_num == 3:
                    patterns.append("主推动浪")
                elif latest_wave_num == 5:
                    patterns.append("推动浪结束浪")
                elif latest_wave_num in [2, 4]:
                    patterns.append("调整浪")
                elif latest_wave_num == 6:
                    patterns.append("A浪调整")
                elif latest_wave_num == 7:
                    patterns.append("B浪反弹")
                elif latest_wave_num == 8:
                    patterns.append("C浪调整")
        
        # 3. 波浪预测分析
        if 'next_wave_prediction' in wave_result.columns:
            prediction = wave_result['next_wave_prediction'].iloc[-1]
            
            if pd.notna(prediction):
                close_price = data['close'].iloc[-1]
                if prediction > close_price:
                    patterns.append(f"波浪预测-上涨目标{prediction:.2f}")
                else:
                    patterns.append(f"波浪预测-下跌目标{prediction:.2f}")
                
                # 距离分析
                distance_pct = abs(close_price - prediction) / close_price * 100
                if distance_pct < 5:
                    patterns.append("接近波浪目标")
                elif distance_pct < 10:
                    patterns.append("临近波浪目标")
                else:
                    patterns.append("远离波浪目标")
        
        # 4. 波浪比例关系分析
        if 'wave_start' in wave_result.columns:
            wave_starts = wave_result[wave_result['wave_start'] == 1]
            
            if len(wave_starts) >= 3:
                try:
                    # 分析最近的波浪比例
                    wave_indices = wave_starts.index.tolist()[-3:]
                    
                    if len(wave_indices) >= 3:
                        # 计算波浪长度
                        wave_1_length = abs(data['close'].loc[wave_indices[1]] - data['close'].loc[wave_indices[0]])
                        wave_3_length = abs(data['close'].loc[wave_indices[2]] - data['close'].loc[wave_indices[1]])
                        
                        if wave_1_length > 0:
                            ratio = wave_3_length / wave_1_length
                            
                            if abs(ratio - 1.618) / 1.618 < 0.1:
                                patterns.append("黄金比例波浪-1.618倍")
                            elif abs(ratio - 2.618) / 2.618 < 0.1:
                                patterns.append("斐波那契比例波浪-2.618倍")
                            elif abs(ratio - 1.0) / 1.0 < 0.1:
                                patterns.append("等长波浪关系")
                            else:
                                patterns.append(f"波浪比例关系-{ratio:.2f}倍")
                
                except (IndexError, KeyError):
                    pass
        
        # 5. 成交量确认分析
        if 'volume' in data.columns and 'wave_number' in wave_result.columns:
            volume = data['volume']
            vol_ma10 = volume.rolling(window=10).mean()
            latest_vol_ratio = (volume / vol_ma10).iloc[-1]
            latest_wave_num = wave_result['wave_number'].iloc[-1]
            latest_wave_dir = wave_result['wave_direction'].iloc[-1]
            
            if pd.notna(latest_vol_ratio) and pd.notna(latest_wave_num) and pd.notna(latest_wave_dir):
                if latest_wave_num in [1, 3, 5]:  # 推动浪
                    if latest_vol_ratio > 1.2:
                        patterns.append("推动浪放量确认")
                    elif latest_vol_ratio < 0.8:
                        patterns.append("推动浪缩量警示")
                
                elif latest_wave_num in [2, 4]:  # 调整浪
                    if latest_vol_ratio < 0.8:
                        patterns.append("调整浪缩量确认")
                    elif latest_vol_ratio > 1.2:
                        patterns.append("调整浪放量异常")
                
                # 第5浪量价背离分析
                if latest_wave_num == 5 and latest_wave_dir == WaveDirection.UP.value:
                    if latest_vol_ratio < 0.9:
                        patterns.append("第5浪量价背离")
        
        # 6. 波浪时间分析
        if 'wave_start' in wave_result.columns:
            wave_starts = wave_result[wave_result['wave_start'] == 1]
            
            if len(wave_starts) >= 2:
                try:
                    recent_waves = wave_starts.tail(2)
                    wave_indices = recent_waves.index.tolist()
                    
                    if len(wave_indices) >= 2:
                        wave_time = wave_indices[1] - wave_indices[0]
                        current_time = len(data) - 1 - wave_indices[1]
                        
                        if wave_time > 0:
                            time_ratio = current_time / wave_time
                            
                            if abs(time_ratio - 0.618) / 0.618 < 0.2:
                                patterns.append("斐波那契时间比例-0.618")
                            elif abs(time_ratio - 1.0) / 1.0 < 0.2:
                                patterns.append("等时间波浪关系")
                            elif abs(time_ratio - 1.618) / 1.618 < 0.2:
                                patterns.append("斐波那契时间比例-1.618")
                
                except (IndexError, KeyError):
                    pass
        
        # 7. 波浪完成度分析
        if 'wave_number' in wave_result.columns:
            latest_wave_num = wave_result['wave_number'].iloc[-1]
            
            if pd.notna(latest_wave_num):
                if latest_wave_num == 5:
                    patterns.append("五浪推动即将完成")
                elif latest_wave_num == 8:
                    patterns.append("三浪调整即将完成")
                elif latest_wave_num < 5:
                    completion = (latest_wave_num / 5) * 100
                    patterns.append(f"推动浪完成度-{completion:.0f}%")
                elif latest_wave_num > 5 and latest_wave_num <= 8:
                    completion = ((latest_wave_num - 5) / 3) * 100
                    patterns.append(f"调整浪完成度-{completion:.0f}%")
        
        # 8. 波浪结构质量分析
        if 'wave_start' in wave_result.columns and 'wave_end' in wave_result.columns:
            wave_starts = wave_result[wave_result['wave_start'] == 1]
            wave_count = len(wave_starts)
            
            if wave_count >= 5:
                patterns.append("完整波浪结构")
            elif wave_count >= 3:
                patterns.append("部分波浪结构")
            else:
                patterns.append("初期波浪结构")
            
            # 波浪清晰度分析
            if wave_count >= 3:
                # 检查波浪是否清晰（价格变化足够大）
                try:
                    wave_indices = wave_starts.index.tolist()
                    total_range = 0
                    wave_ranges = []
                    
                    for i in range(len(wave_indices) - 1):
                        start_price = data['close'].loc[wave_indices[i]]
                        end_price = data['close'].loc[wave_indices[i + 1]]
                        wave_range = abs(end_price - start_price) / start_price
                        wave_ranges.append(wave_range)
                        total_range += wave_range
                    
                    avg_wave_range = total_range / len(wave_ranges) if wave_ranges else 0
                    
                    if avg_wave_range > 0.05:  # 平均波浪幅度大于5%
                        patterns.append("清晰波浪结构")
                    elif avg_wave_range > 0.02:  # 平均波浪幅度大于2%
                        patterns.append("中等波浪结构")
                    else:
                        patterns.append("模糊波浪结构")
                
                except (IndexError, KeyError):
                    pass
        
        return patterns
    
    def generate_signals(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        生成艾略特波浪理论分析信号
        
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
        
        # 计算波浪指标
        elliott_result = self.calculate(data, **kwargs)
        
        # 检查是否有wave_pattern
        if 'wave_pattern' in elliott_result.columns:
            wave_pattern = elliott_result['wave_pattern'].iloc[-1]
        else:
            wave_pattern = WavePattern.UNKNOWN.value
        
        # 获取识别的形态
        patterns = self.identify_patterns(data)
        
        # 计算原始评分
        if kwargs.get('use_raw_score', True):
            score_data = self.calculate_raw_score(data)
            if 'score' in score_data.columns:
                signals['score'] = score_data['score']
        
        # 根据评分生成买卖信号
        for i in range(len(signals)):
            score = signals['score'].iloc[i]
            
            if score >= 70:
                signals.loc[signals.index[i], 'buy_signal'] = True
                signals.loc[signals.index[i], 'neutral_signal'] = False
                signals.loc[signals.index[i], 'trend'] = 1
            elif score <= 30:
                signals.loc[signals.index[i], 'sell_signal'] = True
                signals.loc[signals.index[i], 'neutral_signal'] = False
                signals.loc[signals.index[i], 'trend'] = -1
        
        # 波浪完成情况下的信号
        for wave_num in range(1, 6):
            wave_col = f"wave_{wave_num}"
            if wave_col in elliott_result.columns:
                # 推动浪完成信号(1,3,5浪)
                if wave_num in [1, 3, 5]:
                    # 在第五浪完成位置生成卖出信号
                    if wave_num == 5:
                        complete_indices = elliott_result.index[elliott_result[wave_col].notna() & 
                                                               (elliott_result[wave_col] == '5')]
                        for idx in complete_indices:
                            pos = signals.index.get_loc(idx)
                            signals.loc[signals.index[pos], 'sell_signal'] = True
                            signals.loc[signals.index[pos], 'neutral_signal'] = False
                            signals.loc[signals.index[pos], 'trend'] = -1
                            signals.loc[signals.index[pos], 'signal_type'] = '第五浪完成'
                            signals.loc[signals.index[pos], 'signal_desc'] = '艾略特五浪结构完成，可能开始调整'
                            signals.loc[signals.index[pos], 'confidence'] = 75
                    # 在第三浪完成位置生成部分获利信号
                    elif wave_num == 3:
                        complete_indices = elliott_result.index[elliott_result[wave_col].notna() & 
                                                               (elliott_result[wave_col] == '3')]
                        for idx in complete_indices:
                            pos = signals.index.get_loc(idx)
                            # 不改变之前的买卖信号，但提供信号类型
                            signals.loc[signals.index[pos], 'signal_type'] = '第三浪完成'
                            signals.loc[signals.index[pos], 'signal_desc'] = '艾略特第三浪完成，可能进入第四浪调整'
                            signals.loc[signals.index[pos], 'confidence'] = 65
                
                # 调整浪完成信号(2,4浪)
                elif wave_num in [2, 4]:
                    # 在第四浪完成位置生成买入信号(为第五浪做准备)
                    if wave_num == 4:
                        complete_indices = elliott_result.index[elliott_result[wave_col].notna() & 
                                                               (elliott_result[wave_col] == '4')]
                        for idx in complete_indices:
                            pos = signals.index.get_loc(idx)
                            signals.loc[signals.index[pos], 'buy_signal'] = True
                            signals.loc[signals.index[pos], 'neutral_signal'] = False
                            signals.loc[signals.index[pos], 'trend'] = 1
                            signals.loc[signals.index[pos], 'signal_type'] = '第四浪完成'
                            signals.loc[signals.index[pos], 'signal_desc'] = '艾略特第四浪调整完成，准备进入第五浪'
                            signals.loc[signals.index[pos], 'confidence'] = 70
                    # 在第二浪完成位置生成买入信号(为第三浪做准备)
                    elif wave_num == 2:
                        complete_indices = elliott_result.index[elliott_result[wave_col].notna() & 
                                                               (elliott_result[wave_col] == '2')]
                        for idx in complete_indices:
                            pos = signals.index.get_loc(idx)
                            signals.loc[signals.index[pos], 'buy_signal'] = True
                            signals.loc[signals.index[pos], 'neutral_signal'] = False
                            signals.loc[signals.index[pos], 'trend'] = 1
                            signals.loc[signals.index[pos], 'signal_type'] = '第二浪完成'
                            signals.loc[signals.index[pos], 'signal_desc'] = '艾略特第二浪调整完成，准备进入第三浪'
                            signals.loc[signals.index[pos], 'confidence'] = 80
        
        # 处理特定波浪形态
        if wave_pattern != WavePattern.UNKNOWN.value:
            if wave_pattern == WavePattern.FIVE_WAVE.value:
                # 五浪结构后通常是调整浪开始
                signals.loc[signals.index[-1], 'market_env'] = 'correction_market'
            elif wave_pattern == WavePattern.ZIG_ZAG.value:
                # 锯齿形调整通常在完成后开始反转
                signals.loc[signals.index[-1], 'market_env'] = 'reversal_market'
            elif wave_pattern == WavePattern.FLAT.value or wave_pattern == WavePattern.TRIANGLE.value:
                # 平台形和三角形调整通常是盘整市场
                signals.loc[signals.index[-1], 'market_env'] = 'sideways_market'
        
        # 基于形态的信号增强
        for pattern in patterns:
            # 处理第三浪形态
            if "强势第三浪形成" in pattern:
                for i in range(len(signals)-1, max(0, len(signals)-5), -1):
                    if not signals['signal_type'].iloc[i]:
                        signals.loc[signals.index[i], 'buy_signal'] = True
                        signals.loc[signals.index[i], 'neutral_signal'] = False
                        signals.loc[signals.index[i], 'trend'] = 1
                        signals.loc[signals.index[i], 'signal_type'] = '强势第三浪'
                        signals.loc[signals.index[i], 'signal_desc'] = '波浪理论强势第三浪形成'
                        signals.loc[signals.index[i], 'confidence'] = 85
                        break
            
            # 处理第五浪形态
            elif "第五浪延长" in pattern:
                for i in range(len(signals)-1, max(0, len(signals)-5), -1):
                    if not signals['signal_type'].iloc[i]:
                        signals.loc[signals.index[i], 'sell_signal'] = True
                        signals.loc[signals.index[i], 'neutral_signal'] = False
                        signals.loc[signals.index[i], 'trend'] = -1
                        signals.loc[signals.index[i], 'signal_type'] = '第五浪延长'
                        signals.loc[signals.index[i], 'signal_desc'] = '波浪理论第五浪延长，可能即将完成'
                        signals.loc[signals.index[i], 'confidence'] = 70
                        break
            
            # 处理调整浪形态
            elif "ABC调整完成" in pattern:
                for i in range(len(signals)-1, max(0, len(signals)-5), -1):
                    if not signals['signal_type'].iloc[i]:
                        signals.loc[signals.index[i], 'buy_signal'] = True
                        signals.loc[signals.index[i], 'neutral_signal'] = False
                        signals.loc[signals.index[i], 'trend'] = 1
                        signals.loc[signals.index[i], 'signal_type'] = 'ABC调整完成'
                        signals.loc[signals.index[i], 'signal_desc'] = '波浪理论ABC调整浪完成，可能开始新的上升'
                        signals.loc[signals.index[i], 'confidence'] = 75
                        break
        
        # 成交量确认
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
            score = signals['score'].iloc[i]
            confidence = signals['confidence'].iloc[i]
            
            # 根据信号强度和置信度设置风险等级
            if (score >= 80 or score <= 20) and confidence >= 70:
                signals.loc[signals.index[i], 'risk_level'] = '低'
            elif (score >= 70 or score <= 30) and confidence >= 60:
                signals.loc[signals.index[i], 'risk_level'] = '中'
            else:
                signals.loc[signals.index[i], 'risk_level'] = '高'
            
            # 设置建议仓位
            if signals['buy_signal'].iloc[i]:
                if confidence >= 80:
                    signals.loc[signals.index[i], 'position_size'] = 0.1  # 10%仓位
                elif confidence >= 70:
                    signals.loc[signals.index[i], 'position_size'] = 0.07  # 7%仓位
                elif confidence >= 60:
                    signals.loc[signals.index[i], 'position_size'] = 0.05  # 5%仓位
            elif signals['sell_signal'].iloc[i]:
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