"""
斐波那契工具模块

实现斐波那契回调线、扩展线和时间序列等工具
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple
from enum import Enum

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger

logger = get_logger(__name__)


class FibonacciType(Enum):
    """斐波那契工具类型枚举"""
    RETRACEMENT = "回调线"     # 下跌后回调支撑位
    EXTENSION = "扩展线"       # 上涨后目标位
    TIME_SERIES = "时间序列"   # 时间周期推演


class FibonacciTools(BaseIndicator):
    """
    斐波那契工具指标
    
    计算斐波那契回调线、扩展线和时间序列
    """
    
    # 斐波那契回调线比例
    RETRACEMENT_LEVELS = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    
    # 斐波那契扩展线比例
    EXTENSION_LEVELS = [0, 0.618, 1.0, 1.618, 2.618, 4.236]
    
    # 斐波那契时间序列
    TIME_SERIES = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
    
    def __init__(self):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """初始化斐波那契工具指标"""
        super().__init__(name="FibonacciTools", description="斐波那契工具指标，计算回调线、扩展线和时间序列")

    def set_parameters(self, **kwargs):
        """
        设置指标参数

        Args:
            **kwargs: 参数字典，可包含：
                - swing_window: 摆动点检测窗口，默认10
                - fib_type: 斐波那契类型，默认RETRACEMENT
        """
        self.swing_window = kwargs.get('swing_window', 10)
        self.fib_type = kwargs.get('fib_type', FibonacciType.RETRACEMENT)
    
    def _calculate(self, data: pd.DataFrame, swing_high_idx: int = None, swing_low_idx: int = None, 
                fib_type: Union[FibonacciType, str] = FibonacciType.RETRACEMENT, *args, **kwargs) -> pd.DataFrame:
        """
        计算斐波那契工具
        
        Args:
            data: 输入数据，包含OHLC数据
            swing_high_idx: 摆动高点索引，如果为None则自动检测
            swing_low_idx: 摆动低点索引，如果为None则自动检测
            fib_type: 斐波那契工具类型，默认为回调线
            
        Returns:
            pd.DataFrame: 计算结果，包含斐波那契水平线
        """
        if isinstance(fib_type, str):
            try:
                fib_type = FibonacciType[fib_type.upper()]
            except KeyError:
                logger.warning(f"不支持的斐波那契工具类型: {fib_type}")
                return pd.DataFrame(index=data.index)

        # 确保数据包含必需的列
        required_columns = ["high", "low", "close"]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"数据必须包含'{col}'列")
        
        # 如果未指定摆动点，则自动检测
        if swing_high_idx is None or swing_low_idx is None:
            swing_high_idx, swing_low_idx = self._detect_swing_points(data)
        
        # 根据工具类型计算斐波那契水平线
        if fib_type == FibonacciType.RETRACEMENT:
            result = self.calculate_retracement(data, swing_high_idx, swing_low_idx)
        elif fib_type == FibonacciType.EXTENSION:
            result = self.calculate_extension(data, swing_high_idx, swing_low_idx)
        elif fib_type == FibonacciType.TIME_SERIES:
            result = self.calculate_time_series(data, swing_low_idx)
        else:
            logger.warning(f"不支持的斐波那契工具类型: {fib_type}")
            result = pd.DataFrame(index=data.index)

        # 添加形态识别列（避免循环调用）
        try:
            identified_patterns = self.identify_patterns(data)

            # 初始化所有可能的形态列
            pattern_columns = [
                'FIB_GOLDEN_RATIO_SUPPORT', 'FIB_GOLDEN_RATIO_RESISTANCE',
                'FIB_50_PERCENT_RETRACEMENT', 'FIB_382_RETRACEMENT', 'FIB_618_RETRACEMENT',
                'FIB_EXTENSION_TARGET', 'FIB_GOLDEN_EXTENSION', 'FIB_100_EXTENSION',
                'FIB_CLUSTER_SUPPORT', 'FIB_CLUSTER_RESISTANCE',
                'FIB_BREAKOUT_UP', 'FIB_BREAKOUT_DOWN',
                'FIB_SUPPORT_BOUNCE', 'FIB_RESISTANCE_PULLBACK',
                'FIB_TIME_CYCLE', 'FIB_VOLUME_CONFIRMATION',
                'FIB_TREND_ALIGNMENT', 'FIB_REVERSAL_SIGNAL'
            ]

            for col in pattern_columns:
                result[col] = False

            # 根据识别的形态设置相应的布尔值
            for pattern in identified_patterns:
                if "黄金分割位" in pattern:
                    if "支撑" in pattern:
                        result['FIB_GOLDEN_RATIO_SUPPORT'] = True
                    elif "阻力" in pattern:
                        result['FIB_GOLDEN_RATIO_RESISTANCE'] = True

                if "50%回调位" in pattern:
                    result['FIB_50_PERCENT_RETRACEMENT'] = True
                elif "0.382" in pattern:
                    result['FIB_382_RETRACEMENT'] = True
                elif "0.618" in pattern:
                    result['FIB_618_RETRACEMENT'] = True

                if "扩展位" in pattern:
                    if "黄金扩展位" in pattern:
                        result['FIB_GOLDEN_EXTENSION'] = True
                    elif "100%扩展位" in pattern:
                        result['FIB_100_EXTENSION'] = True
                    else:
                        result['FIB_EXTENSION_TARGET'] = True

                if "聚集区" in pattern:
                    if "支撑" in pattern:
                        result['FIB_CLUSTER_SUPPORT'] = True
                    elif "阻力" in pattern:
                        result['FIB_CLUSTER_RESISTANCE'] = True

                if "向上突破" in pattern:
                    result['FIB_BREAKOUT_UP'] = True
                elif "向下突破" in pattern:
                    result['FIB_BREAKOUT_DOWN'] = True

                if "支撑反弹" in pattern:
                    result['FIB_SUPPORT_BOUNCE'] = True
                elif "阻力回调" in pattern:
                    result['FIB_RESISTANCE_PULLBACK'] = True

                if "时间节点" in pattern:
                    result['FIB_TIME_CYCLE'] = True

                if "放量" in pattern:
                    result['FIB_VOLUME_CONFIRMATION'] = True

                if "趋势" in pattern:
                    result['FIB_TREND_ALIGNMENT'] = True

        except Exception as e:
            logger.warning(f"形态识别失败: {e}")
            # 如果形态识别失败，至少添加空的形态列
            pattern_columns = [
                'FIB_GOLDEN_RATIO_SUPPORT', 'FIB_GOLDEN_RATIO_RESISTANCE',
                'FIB_50_PERCENT_RETRACEMENT', 'FIB_382_RETRACEMENT', 'FIB_618_RETRACEMENT',
                'FIB_EXTENSION_TARGET', 'FIB_GOLDEN_EXTENSION', 'FIB_100_EXTENSION',
                'FIB_CLUSTER_SUPPORT', 'FIB_CLUSTER_RESISTANCE',
                'FIB_BREAKOUT_UP', 'FIB_BREAKOUT_DOWN',
                'FIB_SUPPORT_BOUNCE', 'FIB_RESISTANCE_PULLBACK',
                'FIB_TIME_CYCLE', 'FIB_VOLUME_CONFIRMATION',
                'FIB_TREND_ALIGNMENT', 'FIB_REVERSAL_SIGNAL'
            ]

            for col in pattern_columns:
                result[col] = False

        return result
    
    def calculate_retracement(self, data: pd.DataFrame, swing_high_idx: int, swing_low_idx: int) -> pd.DataFrame:
        """
        计算斐波那契回调线
        
        Args:
            data: 输入数据
            swing_high_idx: 摆动高点索引
            swing_low_idx: 摆动低点索引
            
        Returns:
            pd.DataFrame: 回调线水平
        """
        # 确保高点和低点正确
        if swing_high_idx > swing_low_idx:
            # 下跌后的回调：从高点到低点
            high_price = data["high"].iloc[swing_high_idx]
            low_price = data["low"].iloc[swing_low_idx]
            price_range = high_price - low_price
        else:
            # 上涨后的回调：从低点到高点
            high_price = data["high"].iloc[swing_low_idx]
            low_price = data["low"].iloc[swing_high_idx]
            price_range = high_price - low_price
            # 交换索引，确保高点在前，低点在后
            swing_high_idx, swing_low_idx = swing_low_idx, swing_high_idx
        
        # 初始化结果数据框
        result = data.copy()
        
        # 记录起止点
        result["swing_high"] = np.nan
        result["swing_low"] = np.nan
        result.iloc[swing_high_idx, result.columns.get_loc("swing_high")] = high_price
        result.iloc[swing_low_idx, result.columns.get_loc("swing_low")] = low_price
        
        # 计算回调线
        for level in self.RETRACEMENT_LEVELS:
            level_name = f"fib_{(level * 1000):.0f}"
            level_price = low_price + price_range * level
            result[level_name] = level_price
        
        return result
    
    def calculate_extension(self, data: pd.DataFrame, swing_high_idx: int, swing_low_idx: int) -> pd.DataFrame:
        """
        计算斐波那契扩展线
        
        Args:
            data: 输入数据
            swing_high_idx: 摆动高点索引
            swing_low_idx: 摆动低点索引
            
        Returns:
            pd.DataFrame: 扩展线水平
        """
        # 确保点的顺序正确：先高点，后低点，用于看跌扩展
        if swing_high_idx < swing_low_idx:
            # 先低点，后高点，用于看涨扩展
            high_price = data["high"].iloc[swing_low_idx]
            low_price = data["low"].iloc[swing_high_idx]
            price_range = high_price - low_price
            is_bullish = True
        else:
            # 先高点，后低点，用于看跌扩展
            high_price = data["high"].iloc[swing_high_idx]
            low_price = data["low"].iloc[swing_low_idx]
            price_range = high_price - low_price
            is_bullish = False
        
        # 初始化结果数据框
        result = data.copy()
        
        # 记录起止点
        result["swing_high"] = np.nan
        result["swing_low"] = np.nan
        result.iloc[swing_high_idx, result.columns.get_loc("swing_high" if not is_bullish else "swing_low")] = high_price if not is_bullish else low_price
        result.iloc[swing_low_idx, result.columns.get_loc("swing_low" if not is_bullish else "swing_high")] = low_price if not is_bullish else high_price
        
        # 计算扩展线
        for level in self.EXTENSION_LEVELS:
            level_name = f"fib_ext_{(level * 1000):.0f}".replace('.', '_')
            if is_bullish:
                # 看涨扩展：从低点开始向上扩展
                level_price = high_price + price_range * level
            else:
                # 看跌扩展：从低点开始向下扩展
                level_price = low_price - price_range * level
            result[level_name] = level_price
        
        return result
    
    def calculate_time_series(self, data: pd.DataFrame, start_idx: int) -> pd.DataFrame:
        """
        计算斐波那契时间序列
        
        Args:
            data: 输入数据
            start_idx: 起始点索引
            
        Returns:
            pd.DataFrame: 时间序列点
        """
        # 初始化结果数据框
        result = data.copy()
        
        # 记录起始点
        result["fib_time_start"] = np.nan
        result.iloc[start_idx, result.columns.get_loc("fib_time_start")] = data["close"].iloc[start_idx]
        
        # 计算时间序列点
        for level in self.TIME_SERIES:
            level_name = f"fib_time_{level}"
            target_idx = start_idx + level
            if target_idx < len(data):
                result[level_name] = np.nan
                result.iloc[target_idx, result.columns.get_loc(level_name)] = data["close"].iloc[target_idx]
        
        return result
    
    def _detect_swing_points(self, data: pd.DataFrame, window: int = 10) -> Tuple[int, int]:
        """
        自动检测摆动高点和低点
        
        Args:
            data: 输入数据
            window: 检测窗口
            
        Returns:
            Tuple[int, int]: (高点索引, 低点索引)
        """
        n = len(data)
        if n < 2 * window:
            logger.warning(f"数据长度不足，无法检测摆动点: {n} < {2*window}")
            return 0, n-1
        
        # 在最近的窗口内寻找最高点和最低点
        recent_data = data.iloc[-2*window:]
        high_idx = recent_data["high"].idxmax()
        low_idx = recent_data["low"].idxmin()
        
        # 将索引转换为相对于原始数据的位置
        high_pos = data.index.get_loc(high_idx)
        low_pos = data.index.get_loc(low_idx)
        
        return high_pos, low_pos
    
    def plot_fibonacci_levels(self, data: pd.DataFrame, result: pd.DataFrame, 
                            fib_type: FibonacciType = FibonacciType.RETRACEMENT,
                            ax=None):
        """
        绘制斐波那契水平线
        
        Args:
            data: 输入数据
            result: 斐波那契计算结果
            fib_type: 斐波那契工具类型
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
            
            # 绘制高点和低点
            if "swing_high" in result.columns and "swing_low" in result.columns:
                high_idx = result["swing_high"].first_valid_index()
                low_idx = result["swing_low"].first_valid_index()
                
                if high_idx is not None and low_idx is not None:
                    ax.scatter(high_idx, result.loc[high_idx, "swing_high"], 
                             marker='^', color='g', s=100, label='Swing High')
                    ax.scatter(low_idx, result.loc[low_idx, "swing_low"], 
                             marker='v', color='r', s=100, label='Swing Low')
            
            # 绘制斐波那契水平线
            if fib_type == FibonacciType.RETRACEMENT:
                levels = self.RETRACEMENT_LEVELS
                prefix = "fib_"
                colors = ['red', 'orange', 'gold', 'green', 'blue', 'purple', 'black']
            elif fib_type == FibonacciType.EXTENSION:
                levels = self.EXTENSION_LEVELS
                prefix = "fib_ext_"
                colors = ['red', 'orange', 'gold', 'green', 'blue', 'purple']
            else:
                # 时间序列不绘制水平线
                return ax
            
            for i, level in enumerate(levels):
                level_name = f"{prefix}{level:.3f}".replace(".", "_")
                if level_name in result.columns:
                    level_value = result[level_name].iloc[0]
                    ax.axhline(y=level_value, color=colors[i % len(colors)], 
                             linestyle='--', alpha=0.7,
                             label=f"{level:.3f} - {level_value:.2f}")
            
            # 设置图例
            ax.legend(loc='best')
            
            # 设置标题
            ax.set_title(f'Fibonacci {fib_type.value}')
            
            # 设置日期格式
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            
            return ax
        
        except ImportError:
            logger.error("绘图需要安装matplotlib库")
            return None 
    
    def calculate_raw_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算斐波那契工具指标的原始评分
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            pd.DataFrame: 包含原始评分的DataFrame
        """
        # 初始化评分
        score = pd.Series(50.0, index=data.index)  # 基础分50分
        
        if len(data) < 20:
            return pd.DataFrame({'score': score}, index=data.index)
        
        # 自动检测摆动点
        swing_high_idx, swing_low_idx = self._detect_swing_points(data)
        
        # 计算回调线和扩展线
        try:
            retracement_result = self.calculate_retracement(data, swing_high_idx, swing_low_idx)
            extension_result = self.calculate_extension(data, swing_high_idx, swing_low_idx)
        except Exception as e:
            logger.warning(f"计算斐波那契水平线失败: {e}")
            return pd.DataFrame({'score': score}, index=data.index)
        
        close_price = data['close']
        latest_close = close_price.iloc[-1]
        
        # 1. 关键斐波那契水平支撑/阻力评分（±20分）
        # 检查当前价格是否接近关键斐波那契水平
        key_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        
        for level in key_levels:
            level_name = f"fib_{(level * 1000):.0f}"
            if level_name in retracement_result.columns:
                level_price = retracement_result[level_name].iloc[0]
                
                # 计算价格与斐波那契水平的距离（百分比）
                price_distance = abs(latest_close - level_price) / latest_close
                
                # 如果价格接近斐波那契水平（误差在2%以内）
                if price_distance < 0.02:
                    # 根据斐波那契水平的重要性给分
                    if level == 0.618:  # 黄金分割位最重要
                        if latest_close > level_price:  # 价格在支撑位上方
                            score.iloc[-5:] += 20
                        else:  # 价格在阻力位下方
                            score.iloc[-5:] -= 20
                    elif level in [0.382, 0.5]:  # 次重要水平
                        if latest_close > level_price:
                            score.iloc[-5:] += 15
                        else:
                            score.iloc[-5:] -= 15
                    else:  # 其他水平
                        if latest_close > level_price:
                            score.iloc[-5:] += 10
                        else:
                            score.iloc[-5:] -= 10
        
        # 2. 斐波那契扩展目标评分（±15分）
        # 检查价格是否接近扩展目标
        extension_levels = [0.618, 1.0, 1.618]
        
        for level in extension_levels:
            level_name = f"fib_ext_{(level * 1000):.0f}".replace('.', '_')
            if level_name in extension_result.columns:
                level_price = extension_result[level_name].iloc[0]
                
                # 计算价格与扩展目标的距离
                price_distance = abs(latest_close - level_price) / latest_close
                
                # 如果价格接近扩展目标（误差在3%以内）
                if price_distance < 0.03:
                    if level == 1.618:  # 黄金扩展位
                        score.iloc[-3:] += 15
                    elif level == 1.0:  # 100%扩展
                        score.iloc[-3:] += 12
                    else:  # 其他扩展位
                        score.iloc[-3:] += 8
        
        # 3. 斐波那契聚集区评分（±25分）
        # 检查是否有多个斐波那契水平聚集
        fib_levels = []
        
        # 收集所有斐波那契水平
        for level in key_levels:
            level_name = f"fib_{(level * 1000):.0f}"
            if level_name in retracement_result.columns:
                fib_levels.append(retracement_result[level_name].iloc[0])
        
        for level in extension_levels:
            level_name = f"fib_ext_{(level * 1000):.0f}".replace('.', '_')
            if level_name in extension_result.columns:
                fib_levels.append(extension_result[level_name].iloc[0])
        
        # 检查聚集区
        if len(fib_levels) >= 2:
            fib_levels = sorted(fib_levels)
            
            # 寻找价格聚集区（多个水平在5%范围内）
            for i in range(len(fib_levels)):
                cluster_count = 1
                cluster_center = fib_levels[i]
                
                for j in range(i+1, len(fib_levels)):
                    if abs(fib_levels[j] - cluster_center) / cluster_center < 0.05:
                        cluster_count += 1
                    else:
                        break
                
                # 如果当前价格接近聚集区
                if cluster_count >= 2:
                    distance_to_cluster = abs(latest_close - cluster_center) / latest_close
                    if distance_to_cluster < 0.03:
                        # 聚集区的重要性随聚集水平数量增加
                        cluster_score = min(25, cluster_count * 8)
                        
                        if latest_close > cluster_center:  # 价格在聚集区上方（支撑）
                            score.iloc[-3:] += cluster_score
                        else:  # 价格在聚集区下方（阻力）
                            score.iloc[-3:] -= cluster_score
                        break
        
        # 4. 斐波那契突破评分（±20分）
        # 检查价格是否突破重要斐波那契水平
        if len(data) >= 5:
            recent_closes = close_price.iloc[-5:]
            
            for level in [0.382, 0.5, 0.618]:
                level_name = f"fib_{(level * 1000):.0f}"
                if level_name in retracement_result.columns:
                    level_price = retracement_result[level_name].iloc[0]
                    
                    # 检查是否有突破
                    # 向上突破：之前在水平下方，现在在上方
                    prev_below = (recent_closes.iloc[:-1] < level_price).any()
                    now_above = recent_closes.iloc[-1] > level_price
                    
                    # 向下突破：之前在水平上方，现在在下方
                    prev_above = (recent_closes.iloc[:-1] > level_price).any()
                    now_below = recent_closes.iloc[-1] < level_price
                    
                    if prev_below and now_above:  # 向上突破
                        breakthrough_score = 20 if level == 0.618 else 15
                        score.iloc[-3:] += breakthrough_score
                    elif prev_above and now_below:  # 向下突破
                        breakthrough_score = 20 if level == 0.618 else 15
                        score.iloc[-3:] -= breakthrough_score
        
        # 5. 成交量确认评分（±10分）
        # 斐波那契水平的突破或反弹如果伴随放量，增强信号
        if 'volume' in data.columns and len(data) >= 10:
            volume = data['volume']
            vol_ma5 = volume.rolling(window=5).mean()
            latest_vol_ratio = (volume / vol_ma5).iloc[-1]
            
            # 检查是否在重要斐波那契水平附近且伴随放量
            near_important_level = False
            
            for level in [0.382, 0.5, 0.618]:
                level_name = f"fib_{(level * 1000):.0f}"
                if level_name in retracement_result.columns:
                    level_price = retracement_result[level_name].iloc[0]
                    price_distance = abs(latest_close - level_price) / latest_close
                    
                    if price_distance < 0.03:  # 接近重要水平
                        near_important_level = True
                        break
            
            if near_important_level and pd.notna(latest_vol_ratio):
                if latest_vol_ratio > 1.5:  # 放量
                    score.iloc[-2:] += 10
                elif latest_vol_ratio < 0.7:  # 缩量
                    score.iloc[-2:] -= 5
        
        # 6. 趋势一致性评分（±15分）
        # 检查斐波那契信号是否与趋势一致
        if len(data) >= 20:
            ma20 = close_price.rolling(window=20).mean()
            latest_ma20 = ma20.iloc[-1]
            
            # 判断趋势方向
            if latest_close > latest_ma20 * 1.02:  # 上升趋势
                # 在上升趋势中，斐波那契支撑位更重要
                for level in [0.382, 0.5, 0.618]:
                    level_name = f"fib_{(level * 1000):.0f}"
                    if level_name in retracement_result.columns:
                        level_price = retracement_result[level_name].iloc[0]
                        
                        # 如果价格在斐波那契支撑位上方且接近
                        if latest_close > level_price:
                            price_distance = abs(latest_close - level_price) / latest_close
                            if price_distance < 0.05:
                                score.iloc[-3:] += 15
                                break
            
            elif latest_close < latest_ma20 * 0.98:  # 下降趋势
                # 在下降趋势中，斐波那契阻力位更重要
                for level in [0.382, 0.5, 0.618]:
                    level_name = f"fib_{(level * 1000):.0f}"
                    if level_name in retracement_result.columns:
                        level_price = retracement_result[level_name].iloc[0]
                        
                        # 如果价格在斐波那契阻力位下方且接近
                        if latest_close < level_price:
                            price_distance = abs(latest_close - level_price) / latest_close
                            if price_distance < 0.05:
                                score.iloc[-3:] -= 15
                                break
        
        # 确保评分在0-100范围内
        score = score.clip(0, 100)
        
        return pd.DataFrame({'score': score}, index=data.index)
    
    def identify_patterns(self, data: pd.DataFrame) -> List[str]:
        """
        识别斐波那契工具相关的技术形态
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            List[str]: 识别出的形态列表
        """
        patterns = []
        
        if len(data) < 20:
            return patterns
        
        # 自动检测摆动点
        try:
            swing_high_idx, swing_low_idx = self._detect_swing_points(data)
            retracement_result = self.calculate_retracement(data, swing_high_idx, swing_low_idx)
            extension_result = self.calculate_extension(data, swing_high_idx, swing_low_idx)
        except Exception as e:
            logger.warning(f"计算斐波那契水平线失败: {e}")
            return patterns
        
        close_price = data['close']
        latest_close = close_price.iloc[-1]
        
        # 1. 关键斐波那契水平识别
        key_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        
        for level in key_levels:
            level_name = f"fib_{(level * 1000):.0f}"
            if level_name in retracement_result.columns:
                level_price = retracement_result[level_name].iloc[0]
                price_distance = abs(latest_close - level_price) / latest_close
                
                if price_distance < 0.02:  # 接近斐波那契水平
                    if level == 0.618:
                        patterns.append(f"黄金分割位-{level:.3f}")
                    elif level == 0.5:
                        patterns.append(f"50%回调位-{level:.3f}")
                    else:
                        patterns.append(f"斐波那契回调位-{level:.3f}")
                    
                    # 判断支撑还是阻力
                    if latest_close > level_price:
                        patterns.append(f"斐波那契支撑-{level:.3f}")
                    else:
                        patterns.append(f"斐波那契阻力-{level:.3f}")
        
        # 2. 斐波那契扩展目标识别
        extension_levels = [0.618, 1.0, 1.618, 2.618]
        
        for level in extension_levels:
            level_name = f"fib_ext_{(level * 1000):.0f}".replace('.', '_')
            if level_name in extension_result.columns:
                level_price = extension_result[level_name].iloc[0]
                price_distance = abs(latest_close - level_price) / latest_close
                
                if price_distance < 0.03:  # 接近扩展目标
                    if level == 1.618:
                        patterns.append(f"黄金扩展位-{level:.3f}")
                    elif level == 1.0:
                        patterns.append(f"100%扩展位-{level:.3f}")
                    else:
                        patterns.append(f"斐波那契扩展位-{level:.3f}")
        
        # 3. 斐波那契聚集区识别
        fib_levels = []
        level_names = []
        
        # 收集所有斐波那契水平
        for level in key_levels:
            level_name = f"fib_{(level * 1000):.0f}"
            if level_name in retracement_result.columns:
                fib_levels.append(retracement_result[level_name].iloc[0])
                level_names.append(f"回调{level:.3f}")
        
        for level in extension_levels:
            level_name = f"fib_ext_{(level * 1000):.0f}".replace('.', '_')
            if level_name in extension_result.columns:
                fib_levels.append(extension_result[level_name].iloc[0])
                level_names.append(f"扩展{level:.3f}")
        
        # 检查聚集区
        if len(fib_levels) >= 2:
            fib_data = list(zip(fib_levels, level_names))
            fib_data.sort(key=lambda x: x[0])  # 按价格排序
            
            # 寻找聚集区
            for i in range(len(fib_data)):
                cluster_levels = [fib_data[i]]
                cluster_center = fib_data[i][0]
                
                for j in range(i+1, len(fib_data)):
                    if abs(fib_data[j][0] - cluster_center) / cluster_center < 0.05:
                        cluster_levels.append(fib_data[j])
                    else:
                        break
                
                # 如果形成聚集区且当前价格接近
                if len(cluster_levels) >= 2:
                    distance_to_cluster = abs(latest_close - cluster_center) / latest_close
                    if distance_to_cluster < 0.03:
                        level_desc = "+".join([level[1] for level in cluster_levels])
                        patterns.append(f"斐波那契聚集区-{level_desc}")
                        
                        if latest_close > cluster_center:
                            patterns.append("聚集区支撑")
                        else:
                            patterns.append("聚集区阻力")
                        break
        
        # 4. 斐波那契突破识别
        if len(data) >= 5:
            recent_closes = close_price.iloc[-5:]
            
            for level in [0.382, 0.5, 0.618]:
                level_name = f"fib_{(level * 1000):.0f}"
                if level_name in retracement_result.columns:
                    level_price = retracement_result[level_name].iloc[0]
                    
                    # 检查突破
                    prev_below = (recent_closes.iloc[:-1] < level_price).any()
                    now_above = recent_closes.iloc[-1] > level_price
                    prev_above = (recent_closes.iloc[:-1] > level_price).any()
                    now_below = recent_closes.iloc[-1] < level_price
                    
                    if prev_below and now_above:
                        patterns.append(f"向上突破斐波那契-{level:.3f}")
                        if level == 0.618:
                            patterns.append("突破黄金分割位")
                    elif prev_above and now_below:
                        patterns.append(f"向下突破斐波那契-{level:.3f}")
                        if level == 0.618:
                            patterns.append("跌破黄金分割位")
        
        # 5. 斐波那契反弹/回调识别
        if len(data) >= 10:
            # 检查是否在斐波那契水平附近发生反弹或回调
            for level in [0.382, 0.5, 0.618]:
                level_name = f"fib_{(level * 1000):.0f}"
                if level_name in retracement_result.columns:
                    level_price = retracement_result[level_name].iloc[0]
                    
                    # 检查最近是否触及该水平并反弹
                    recent_lows = data['low'].iloc[-5:]
                    recent_highs = data['high'].iloc[-5:]
                    
                    # 检查是否触及支撑位并反弹
                    touched_support = (recent_lows <= level_price * 1.02).any()
                    bounced_up = latest_close > level_price * 1.01
                    
                    if touched_support and bounced_up:
                        patterns.append(f"斐波那契支撑反弹-{level:.3f}")
                        if level == 0.618:
                            patterns.append("黄金分割位支撑反弹")
                    
                    # 检查是否触及阻力位并回调
                    touched_resistance = (recent_highs >= level_price * 0.98).any()
                    pulled_back = latest_close < level_price * 0.99
                    
                    if touched_resistance and pulled_back:
                        patterns.append(f"斐波那契阻力回调-{level:.3f}")
                        if level == 0.618:
                            patterns.append("黄金分割位阻力回调")
        
        # 6. 成交量确认分析
        if 'volume' in data.columns and len(data) >= 10:
            volume = data['volume']
            vol_ma5 = volume.rolling(window=5).mean()
            latest_vol_ratio = (volume / vol_ma5).iloc[-1]
            
            # 检查重要斐波那契水平附近的成交量
            near_important_level = False
            important_level_name = ""
            
            for level in [0.382, 0.5, 0.618]:
                level_name = f"fib_{(level * 1000):.0f}"
                if level_name in retracement_result.columns:
                    level_price = retracement_result[level_name].iloc[0]
                    price_distance = abs(latest_close - level_price) / latest_close
                    
                    if price_distance < 0.03:
                        near_important_level = True
                        important_level_name = f"{level:.3f}"
                        break
            
            if near_important_level and pd.notna(latest_vol_ratio):
                if latest_vol_ratio > 1.5:
                    patterns.append(f"斐波那契水平放量-{important_level_name}")
                elif latest_vol_ratio < 0.7:
                    patterns.append(f"斐波那契水平缩量-{important_level_name}")
        
        # 7. 趋势一致性分析
        if len(data) >= 20:
            ma20 = close_price.rolling(window=20).mean()
            latest_ma20 = ma20.iloc[-1]
            
            if latest_close > latest_ma20 * 1.02:  # 上升趋势
                patterns.append("上升趋势中的斐波那契分析")
                
                # 检查是否在关键支撑位上方
                for level in [0.382, 0.5, 0.618]:
                    level_name = f"fib_{(level * 1000):.0f}"
                    if level_name in retracement_result.columns:
                        level_price = retracement_result[level_name].iloc[0]
                        
                        if latest_close > level_price:
                            price_distance = abs(latest_close - level_price) / latest_close
                            if price_distance < 0.05:
                                patterns.append(f"上升趋势斐波那契支撑-{level:.3f}")
                                break
            
            elif latest_close < latest_ma20 * 0.98:  # 下降趋势
                patterns.append("下降趋势中的斐波那契分析")
                
                # 检查是否在关键阻力位下方
                for level in [0.382, 0.5, 0.618]:
                    level_name = f"fib_{(level * 1000):.0f}"
                    if level_name in retracement_result.columns:
                        level_price = retracement_result[level_name].iloc[0]
                        
                        if latest_close < level_price:
                            price_distance = abs(latest_close - level_price) / latest_close
                            if price_distance < 0.05:
                                patterns.append(f"下降趋势斐波那契阻力-{level:.3f}")
                                break
            else:
                patterns.append("横盘趋势中的斐波那契分析")
        
        # 8. 斐波那契时间序列分析
        try:
            time_series_result = self.calculate_time_series(data, swing_low_idx)
            
            # 检查是否接近重要时间节点
            current_idx = len(data) - 1
            for time_level in self.TIME_SERIES[:8]:  # 检查前8个时间序列
                target_idx = swing_low_idx + time_level
                
                if abs(current_idx - target_idx) <= 2:  # 接近时间节点
                    patterns.append(f"斐波那契时间节点-{time_level}周期")
                    
                    if time_level in [8, 13, 21]:  # 重要时间节点
                        patterns.append(f"重要时间节点-{time_level}")
        
        except Exception as e:
            logger.warning(f"斐波那契时间序列分析失败: {e}")
        
        return patterns 
        
    def generate_signals(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        生成斐波那契工具指标信号
        
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
        
        # 计算原始评分
        if kwargs.get('use_raw_score', True):
            score_data = self.calculate_raw_score(data)
            if 'score' in score_data.columns:
                signals['score'] = score_data['score']
        
        # 获取识别的形态
        patterns = self.identify_patterns(data)
        
        # 自动检测摆动点
        try:
            swing_high_idx, swing_low_idx = self._detect_swing_points(data)
            retracement_result = self.calculate_retracement(data, swing_high_idx, swing_low_idx)
        except Exception as e:
            logger.warning(f"计算斐波那契水平线失败: {e}")
            return signals
        
        close_price = data['close'].values
        latest_close = data['close'].iloc[-1]
        
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
        
        # 生成斐波那契水平突破信号
        for level in [0.382, 0.5, 0.618]:
            level_name = f"fib_{(level * 1000):.0f}"
            if level_name in retracement_result.columns:
                level_prices = retracement_result[level_name].values
                
                for i in range(5, len(signals)):
                    if i >= len(level_prices):
                        continue
                    
                    # 向上突破
                    if (close_price[i] > level_prices[i]) and (close_price[i-1] <= level_prices[i-1]):
                        signals.loc[signals.index[i], 'buy_signal'] = True
                        signals.loc[signals.index[i], 'neutral_signal'] = False
                        signals.loc[signals.index[i], 'trend'] = 1
                        
                        if level == 0.618:
                            signals.loc[signals.index[i], 'signal_type'] = '突破黄金分割位'
                            signals.loc[signals.index[i], 'signal_desc'] = f'价格突破0.618黄金分割位'
                            signals.loc[signals.index[i], 'confidence'] = 75
                        else:
                            signals.loc[signals.index[i], 'signal_type'] = f'突破{level:.3f}回调位'
                            signals.loc[signals.index[i], 'signal_desc'] = f'价格突破{level:.3f}斐波那契回调位'
                            signals.loc[signals.index[i], 'confidence'] = 65
                    
                    # 向下突破
                    elif (close_price[i] < level_prices[i]) and (close_price[i-1] >= level_prices[i-1]):
                        signals.loc[signals.index[i], 'sell_signal'] = True
                        signals.loc[signals.index[i], 'neutral_signal'] = False
                        signals.loc[signals.index[i], 'trend'] = -1
                        
                        if level == 0.618:
                            signals.loc[signals.index[i], 'signal_type'] = '跌破黄金分割位'
                            signals.loc[signals.index[i], 'signal_desc'] = f'价格跌破0.618黄金分割位'
                            signals.loc[signals.index[i], 'confidence'] = 75
                        else:
                            signals.loc[signals.index[i], 'signal_type'] = f'跌破{level:.3f}回调位'
                            signals.loc[signals.index[i], 'signal_desc'] = f'价格跌破{level:.3f}斐波那契回调位'
                            signals.loc[signals.index[i], 'confidence'] = 65
        
        # 处理基于形态的信号增强
        for pattern in patterns:
            # 黄金分割位相关信号
            if "黄金分割位支撑反弹" in pattern:
                # 找到最近的斐波那契反弹位
                for i in range(len(signals)-1, max(0, len(signals)-10), -1):
                    if not signals['signal_type'].iloc[i]:  # 如果没有设置信号类型
                        signals.loc[signals.index[i], 'buy_signal'] = True
                        signals.loc[signals.index[i], 'neutral_signal'] = False
                        signals.loc[signals.index[i], 'trend'] = 1
                        signals.loc[signals.index[i], 'signal_type'] = '黄金分割位支撑反弹'
                        signals.loc[signals.index[i], 'signal_desc'] = '价格在0.618黄金分割位获得支撑反弹'
                        signals.loc[signals.index[i], 'confidence'] = 70
                        break
            
            elif "黄金分割位阻力回调" in pattern:
                # 找到最近的斐波那契回调位
                for i in range(len(signals)-1, max(0, len(signals)-10), -1):
                    if not signals['signal_type'].iloc[i]:  # 如果没有设置信号类型
                        signals.loc[signals.index[i], 'sell_signal'] = True
                        signals.loc[signals.index[i], 'neutral_signal'] = False
                        signals.loc[signals.index[i], 'trend'] = -1
                        signals.loc[signals.index[i], 'signal_type'] = '黄金分割位阻力回调'
                        signals.loc[signals.index[i], 'signal_desc'] = '价格在0.618黄金分割位遇阻回落'
                        signals.loc[signals.index[i], 'confidence'] = 70
                        break
            
            # 斐波那契聚集区信号
            elif "聚集区支撑" in pattern:
                for i in range(len(signals)-1, max(0, len(signals)-5), -1):
                    if not signals['signal_type'].iloc[i]:
                        signals.loc[signals.index[i], 'buy_signal'] = True
                        signals.loc[signals.index[i], 'neutral_signal'] = False
                        signals.loc[signals.index[i], 'trend'] = 1
                        signals.loc[signals.index[i], 'signal_type'] = '斐波那契聚集区支撑'
                        signals.loc[signals.index[i], 'signal_desc'] = '价格在多个斐波那契水平聚集区获得支撑'
                        signals.loc[signals.index[i], 'confidence'] = 80
                        break
            
            elif "聚集区阻力" in pattern:
                for i in range(len(signals)-1, max(0, len(signals)-5), -1):
                    if not signals['signal_type'].iloc[i]:
                        signals.loc[signals.index[i], 'sell_signal'] = True
                        signals.loc[signals.index[i], 'neutral_signal'] = False
                        signals.loc[signals.index[i], 'trend'] = -1
                        signals.loc[signals.index[i], 'signal_type'] = '斐波那契聚集区阻力'
                        signals.loc[signals.index[i], 'signal_desc'] = '价格在多个斐波那契水平聚集区遇阻回落'
                        signals.loc[signals.index[i], 'confidence'] = 80
                        break
            
            # 时间节点信号
            elif "重要时间节点" in pattern:
                time_level = int(pattern.split('-')[1])
                for i in range(len(signals)-1, max(0, len(signals)-3), -1):
                    if not signals['signal_type'].iloc[i]:
                        # 根据最近的价格趋势判断买卖方向
                        if len(data) >= 5:
                            recent_trend = data['close'].iloc[-5:].diff().mean()
                            if recent_trend > 0:
                                signals.loc[signals.index[i], 'buy_signal'] = True
                                signals.loc[signals.index[i], 'trend'] = 1
                                signals.loc[signals.index[i], 'signal_desc'] = f'斐波那契{time_level}周期时间节点-上升反转'
                            else:
                                signals.loc[signals.index[i], 'sell_signal'] = True
                                signals.loc[signals.index[i], 'trend'] = -1
                                signals.loc[signals.index[i], 'signal_desc'] = f'斐波那契{time_level}周期时间节点-下降反转'
                            
                            signals.loc[signals.index[i], 'neutral_signal'] = False
                            signals.loc[signals.index[i], 'signal_type'] = f'斐波那契时间节点-{time_level}'
                            signals.loc[signals.index[i], 'confidence'] = 65
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
        
        # 设置市场环境
        if len(data) >= 20:
            ma20 = data['close'].rolling(window=20).mean()
            latest_ma20 = ma20.iloc[-1]
            
            if latest_close > latest_ma20 * 1.02:
                signals.loc[signals.index[-1], 'market_env'] = 'bullish_market'  # 上升市场
            elif latest_close < latest_ma20 * 0.98:
                signals.loc[signals.index[-1], 'market_env'] = 'bearish_market'  # 下降市场
            else:
                signals.loc[signals.index[-1], 'market_env'] = 'sideways_market'  # 震荡市场
        
        # 计算止损位
        for i in range(len(signals)):
            if signals['buy_signal'].iloc[i]:
                # 买入信号的止损：使用下方最近的斐波那契水平
                for level in [0.618, 0.5, 0.382]:
                    level_name = f"fib_{(level * 1000):.0f}"
                    if level_name in retracement_result.columns:
                        level_price = retracement_result[level_name].iloc[0]
                        if level_price < close_price[i]:
                            signals.loc[signals.index[i], 'stop_loss'] = level_price * 0.98
                            break
            
            elif signals['sell_signal'].iloc[i]:
                # 卖出信号的止损：使用上方最近的斐波那契水平
                for level in [0.382, 0.5, 0.618]:
                    level_name = f"fib_{(level * 1000):.0f}"
                    if level_name in retracement_result.columns:
                        level_price = retracement_result[level_name].iloc[0]
                        if level_price > close_price[i]:
                            signals.loc[signals.index[i], 'stop_loss'] = level_price * 1.02
                            break
        
        return signals

    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取FibonacciTools相关形态

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含形态信息的DataFrame
        """
        # 确保已计算指标
        if not self.has_result():
            self._calculate(data, **kwargs)

        patterns = pd.DataFrame(index=data.index)

        # 如果没有计算结果，返回空DataFrame
        if self._result is None or self._result.empty:
            return patterns

        # 基于识别的形态创建布尔列
        identified_patterns = self.identify_patterns(data)

        # 初始化所有可能的形态列
        pattern_columns = [
            'FIB_GOLDEN_RATIO_SUPPORT', 'FIB_GOLDEN_RATIO_RESISTANCE',
            'FIB_50_PERCENT_RETRACEMENT', 'FIB_382_RETRACEMENT', 'FIB_618_RETRACEMENT',
            'FIB_EXTENSION_TARGET', 'FIB_GOLDEN_EXTENSION', 'FIB_100_EXTENSION',
            'FIB_CLUSTER_SUPPORT', 'FIB_CLUSTER_RESISTANCE',
            'FIB_BREAKOUT_UP', 'FIB_BREAKOUT_DOWN',
            'FIB_SUPPORT_BOUNCE', 'FIB_RESISTANCE_PULLBACK',
            'FIB_TIME_CYCLE', 'FIB_VOLUME_CONFIRMATION',
            'FIB_TREND_ALIGNMENT', 'FIB_REVERSAL_SIGNAL'
        ]

        for col in pattern_columns:
            patterns[col] = False

        # 根据识别的形态设置相应的布尔值
        for pattern in identified_patterns:
            if "黄金分割位" in pattern:
                if "支撑" in pattern:
                    patterns['FIB_GOLDEN_RATIO_SUPPORT'] = True
                elif "阻力" in pattern:
                    patterns['FIB_GOLDEN_RATIO_RESISTANCE'] = True

            if "50%回调位" in pattern:
                patterns['FIB_50_PERCENT_RETRACEMENT'] = True
            elif "0.382" in pattern:
                patterns['FIB_382_RETRACEMENT'] = True
            elif "0.618" in pattern:
                patterns['FIB_618_RETRACEMENT'] = True

            if "扩展位" in pattern:
                if "黄金扩展位" in pattern:
                    patterns['FIB_GOLDEN_EXTENSION'] = True
                elif "100%扩展位" in pattern:
                    patterns['FIB_100_EXTENSION'] = True
                else:
                    patterns['FIB_EXTENSION_TARGET'] = True

            if "聚集区" in pattern:
                if "支撑" in pattern:
                    patterns['FIB_CLUSTER_SUPPORT'] = True
                elif "阻力" in pattern:
                    patterns['FIB_CLUSTER_RESISTANCE'] = True

            if "向上突破" in pattern:
                patterns['FIB_BREAKOUT_UP'] = True
            elif "向下突破" in pattern:
                patterns['FIB_BREAKOUT_DOWN'] = True

            if "支撑反弹" in pattern:
                patterns['FIB_SUPPORT_BOUNCE'] = True
            elif "阻力回调" in pattern:
                patterns['FIB_RESISTANCE_PULLBACK'] = True

            if "时间节点" in pattern:
                patterns['FIB_TIME_CYCLE'] = True

            if "放量" in pattern:
                patterns['FIB_VOLUME_CONFIRMATION'] = True

            if "趋势" in pattern:
                patterns['FIB_TREND_ALIGNMENT'] = True

        return patterns

    def calculate_confidence(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """
        计算FibonacciTools指标的置信度

        Args:
            score: 得分序列
            patterns: 检测到的形态DataFrame
            signals: 生成的信号字典

        Returns:
            float: 置信度分数 (0-1)
        """
        if score.empty:
            return 0.5

        # 基础置信度
        confidence = 0.5

        # 1. 基于评分的置信度
        last_score = score.iloc[-1]

        # 极端评分置信度较高
        if last_score > 80 or last_score < 20:
            confidence += 0.25
        # 中性评分置信度中等
        elif 40 <= last_score <= 60:
            confidence += 0.1
        else:
            confidence += 0.15

        # 2. 基于数据质量的置信度
        if hasattr(self, '_result') and self._result is not None:
            # 检查是否有斐波那契水平数据
            fib_columns = [col for col in self._result.columns if 'fib_' in col]
            if fib_columns:
                # 斐波那契数据越完整，置信度越高
                data_completeness = len(fib_columns) / 10  # 假设最多10个斐波那契水平
                confidence += min(data_completeness * 0.1, 0.1)

        # 3. 基于形态的置信度
        if not patterns.empty:
            # 检查FibonacciTools形态（只计算布尔列）
            bool_columns = patterns.select_dtypes(include=[bool]).columns
            if len(bool_columns) > 0:
                pattern_count = patterns[bool_columns].sum().sum()
                if pattern_count > 0:
                    confidence += min(pattern_count * 0.02, 0.15)

        # 4. 基于信号的置信度
        if signals:
            # 检查信号强度
            signal_count = sum(1 for signal in signals.values() if hasattr(signal, 'any') and signal.any())
            if signal_count > 0:
                confidence += min(signal_count * 0.05, 0.1)

        # 5. 基于数据长度的置信度
        if len(score) >= 60:  # 两个月数据
            confidence += 0.1
        elif len(score) >= 30:  # 一个月数据
            confidence += 0.05

        # 确保置信度在0-1范围内
        return max(0.0, min(1.0, confidence))

    def register_patterns(self):
        """
        注册FibonacciTools指标的形态到全局形态注册表
        """
        # 注册黄金分割位形态
        self.register_pattern_to_registry(
            pattern_id="FIB_GOLDEN_RATIO_SUPPORT",
            display_name="黄金分割位支撑",
            description="价格在0.618黄金分割位获得支撑，强烈的买入信号",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=30.0
        )

        self.register_pattern_to_registry(
            pattern_id="FIB_GOLDEN_RATIO_RESISTANCE",
            display_name="黄金分割位阻力",
            description="价格在0.618黄金分割位遇阻，强烈的卖出信号",
            pattern_type="BEARISH",
            default_strength="VERY_STRONG",
            score_impact=-30.0
        )

        # 注册回调位形态
        self.register_pattern_to_registry(
            pattern_id="FIB_50_PERCENT_RETRACEMENT",
            display_name="50%回调位",
            description="价格接近50%斐波那契回调位，重要的支撑/阻力位",
            pattern_type="NEUTRAL",
            default_strength="STRONG",
            score_impact=15.0
        )

        self.register_pattern_to_registry(
            pattern_id="FIB_382_RETRACEMENT",
            display_name="38.2%回调位",
            description="价格接近38.2%斐波那契回调位，浅度回调支撑",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=10.0
        )

        self.register_pattern_to_registry(
            pattern_id="FIB_618_RETRACEMENT",
            display_name="61.8%回调位",
            description="价格接近61.8%斐波那契回调位，深度回调支撑",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=20.0
        )

        # 注册扩展位形态
        self.register_pattern_to_registry(
            pattern_id="FIB_GOLDEN_EXTENSION",
            display_name="黄金扩展位",
            description="价格接近1.618黄金扩展位，重要的目标位",
            pattern_type="NEUTRAL",
            default_strength="STRONG",
            score_impact=20.0
        )

        self.register_pattern_to_registry(
            pattern_id="FIB_100_EXTENSION",
            display_name="100%扩展位",
            description="价格接近100%扩展位，等幅扩展目标",
            pattern_type="NEUTRAL",
            default_strength="MEDIUM",
            score_impact=15.0
        )

        # 注册聚集区形态
        self.register_pattern_to_registry(
            pattern_id="FIB_CLUSTER_SUPPORT",
            display_name="斐波那契聚集区支撑",
            description="多个斐波那契水平聚集形成强力支撑",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=35.0
        )

        self.register_pattern_to_registry(
            pattern_id="FIB_CLUSTER_RESISTANCE",
            display_name="斐波那契聚集区阻力",
            description="多个斐波那契水平聚集形成强力阻力",
            pattern_type="BEARISH",
            default_strength="VERY_STRONG",
            score_impact=-35.0
        )

        # 注册突破形态
        self.register_pattern_to_registry(
            pattern_id="FIB_BREAKOUT_UP",
            display_name="斐波那契向上突破",
            description="价格向上突破重要斐波那契水平",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=25.0
        )

        self.register_pattern_to_registry(
            pattern_id="FIB_BREAKOUT_DOWN",
            display_name="斐波那契向下突破",
            description="价格向下突破重要斐波那契水平",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-25.0
        )

        # 注册反弹回调形态
        self.register_pattern_to_registry(
            pattern_id="FIB_SUPPORT_BOUNCE",
            display_name="斐波那契支撑反弹",
            description="价格在斐波那契水平获得支撑并反弹",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=18.0
        )

        self.register_pattern_to_registry(
            pattern_id="FIB_RESISTANCE_PULLBACK",
            display_name="斐波那契阻力回调",
            description="价格在斐波那契水平遇阻并回调",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-18.0
        )

        # 注册时间周期形态
        self.register_pattern_to_registry(
            pattern_id="FIB_TIME_CYCLE",
            display_name="斐波那契时间周期",
            description="接近重要的斐波那契时间节点",
            pattern_type="NEUTRAL",
            default_strength="MEDIUM",
            score_impact=12.0
        )

        # 注册确认形态
        self.register_pattern_to_registry(
            pattern_id="FIB_VOLUME_CONFIRMATION",
            display_name="斐波那契成交量确认",
            description="斐波那契水平附近伴随成交量放大",
            pattern_type="NEUTRAL",
            default_strength="MEDIUM",
            score_impact=10.0
        )

        self.register_pattern_to_registry(
            pattern_id="FIB_TREND_ALIGNMENT",
            display_name="斐波那契趋势一致",
            description="斐波那契信号与主趋势方向一致",
            pattern_type="NEUTRAL",
            default_strength="STRONG",
            score_impact=15.0
        )

    def generate_trading_signals(self, data: pd.DataFrame, **kwargs) -> dict:
        """
        生成FibonacciTools交易信号

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            dict: 包含买卖信号的字典
        """
        # 确保已计算指标
        if not self.has_result():
            self._calculate(data, **kwargs)

        if self._result is None or self._result.empty:
            return {
                'buy_signal': pd.Series(False, index=data.index),
                'sell_signal': pd.Series(False, index=data.index),
                'signal_strength': pd.Series(0.0, index=data.index)
            }

        # 使用generate_signals方法生成详细信号
        detailed_signals = self.generate_signals(data, **kwargs)

        # 转换为简化的信号格式
        buy_signal = detailed_signals['buy_signal']
        sell_signal = detailed_signals['sell_signal']

        # 计算信号强度
        signal_strength = pd.Series(0.0, index=data.index)

        # 基于置信度计算信号强度
        confidence = detailed_signals['confidence']

        # 买入信号强度
        signal_strength[buy_signal] = confidence[buy_signal] / 100.0

        # 卖出信号强度（负值）
        signal_strength[sell_signal] = -confidence[sell_signal] / 100.0

        # 标准化信号强度
        signal_strength = signal_strength.clip(-1, 1)

        return {
            'buy_signal': buy_signal,
            'sell_signal': sell_signal,
            'signal_strength': signal_strength
        }

    def get_indicator_type(self) -> str:
        """
        获取指标类型

        Returns:
            str: 指标类型
        """
        return "FIBONACCITOOLS"