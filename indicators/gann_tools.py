"""
江恩理论工具模块

实现江恩角度线、江恩方格和时间周期工具
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple
from enum import Enum
from datetime import datetime, timedelta

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger

logger = get_logger(__name__)


class GannAngle(Enum):
    """江恩角度枚举"""
    ANGLE_1X8 = "1×8角度线"      # 1×8角度线，约为82.5度
    ANGLE_1X4 = "1×4角度线"      # 1×4角度线，约为75度
    ANGLE_1X3 = "1×3角度线"      # 1×3角度线，约为71.6度
    ANGLE_1X2 = "1×2角度线"      # 1×2角度线，约为63.4度
    ANGLE_1X1 = "1×1角度线"      # 1×1角度线，约为45度
    ANGLE_2X1 = "2×1角度线"      # 2×1角度线，约为26.6度
    ANGLE_3X1 = "3×1角度线"      # 3×1角度线，约为18.4度
    ANGLE_4X1 = "4×1角度线"      # 4×1角度线，约为15度
    ANGLE_8X1 = "8×1角度线"      # 8×1角度线，约为7.5度


class GannTimeCycle(Enum):
    """江恩时间周期枚举"""
    CYCLE_30 = "30日周期"
    CYCLE_45 = "45日周期"
    CYCLE_60 = "60日周期"
    CYCLE_90 = "90日周期"
    CYCLE_120 = "120日周期"
    CYCLE_144 = "144日周期"
    CYCLE_180 = "180日周期"
    CYCLE_270 = "270日周期"
    CYCLE_360 = "360日周期"


class GannTools(BaseIndicator):
    """
    江恩理论工具指标
    
    计算江恩角度线、江恩方格和时间周期
    """
    
    # 江恩角度线比例
    ANGLE_RATIOS = {
        GannAngle.ANGLE_1X8: (1, 8),
        GannAngle.ANGLE_1X4: (1, 4),
        GannAngle.ANGLE_1X3: (1, 3),
        GannAngle.ANGLE_1X2: (1, 2),
        GannAngle.ANGLE_1X1: (1, 1),
        GannAngle.ANGLE_2X1: (2, 1),
        GannAngle.ANGLE_3X1: (3, 1),
        GannAngle.ANGLE_4X1: (4, 1),
        GannAngle.ANGLE_8X1: (8, 1)
    }
    
    # 江恩时间周期
    TIME_CYCLES = {
        GannTimeCycle.CYCLE_30: 30,
        GannTimeCycle.CYCLE_45: 45,
        GannTimeCycle.CYCLE_60: 60,
        GannTimeCycle.CYCLE_90: 90,
        GannTimeCycle.CYCLE_120: 120,
        GannTimeCycle.CYCLE_144: 144,
        GannTimeCycle.CYCLE_180: 180,
        GannTimeCycle.CYCLE_270: 270,
        GannTimeCycle.CYCLE_360: 360
    }
    
    def __init__(self):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """初始化江恩理论工具指标"""
        super().__init__(name="GannTools", description="江恩理论工具指标，计算角度线和时间周期")
    
    def _calculate(self, data: pd.DataFrame, pivot_idx: int = None, price_unit: float = None, 
                time_unit: int = 1, *args, **kwargs) -> pd.DataFrame:
        """
        计算江恩理论工具
        
        Args:
            data: 输入数据，包含OHLC数据
            pivot_idx: 支点索引，如果为None则使用第一个点
            price_unit: 价格单位，如果为None则自动计算
            time_unit: 时间单位，默认为1天
            
        Returns:
            pd.DataFrame: 计算结果，包含江恩角度线和时间周期
        """
        # 确保数据包含必需的列
        required_columns = ["close"]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"数据必须包含'{col}'列")
        
        # 如果未指定支点，则使用第一个点
        if pivot_idx is None:
            pivot_idx = 0
        
        # 获取支点价格
        pivot_price = data["close"].iloc[pivot_idx]
        
        # 如果未指定价格单位，则自动计算
        if price_unit is None:
            # 使用价格的0.1%作为基本单位
            price_unit = pivot_price * 0.001
        
        # 初始化结果数据框
        result = data.copy()
        
        # 计算角度线
        result = self._calculate_angle_lines(data, result, pivot_idx, pivot_price, price_unit, time_unit)
        
        # 计算时间周期
        result = self._calculate_time_cycles(data, result, pivot_idx)

        # 添加形态识别列（避免循环调用）
        try:
            # 直接进行简单的形态识别，避免调用identify_patterns
            identified_patterns = []

            if 'close' in data.columns and len(data) >= 10:
                close_price = data['close']
                latest_price = close_price.iloc[-1]

                # 简单的江恩1x1线分析
                for angle in GannAngle:
                    angle_name = f"gann_{angle.name.lower().replace('angle_', '')}"
                    if angle_name in result.columns:
                        angle_line = result[angle_name].iloc[-1]

                        if pd.notna(angle_line):
                            distance_pct = (latest_price - angle_line) / angle_line * 100

                            if abs(distance_pct) < 2:
                                if distance_pct > 0:
                                    identified_patterns.append(f"接近{angle.value}支撑")
                                else:
                                    identified_patterns.append(f"接近{angle.value}阻力")

            # 初始化所有可能的形态列
            pattern_columns = [
                'GANN_1X1_SUPPORT', 'GANN_1X1_RESISTANCE',
                'GANN_1X2_SUPPORT', 'GANN_1X2_RESISTANCE',
                'GANN_2X1_SUPPORT', 'GANN_2X1_RESISTANCE',
                'GANN_ANGLE_CLUSTER_SUPPORT', 'GANN_ANGLE_CLUSTER_RESISTANCE',
                'GANN_1X1_BREAKOUT_UP', 'GANN_1X1_BREAKOUT_DOWN',
                'GANN_TIME_CYCLE_LOW', 'GANN_TIME_CYCLE_HIGH',
                'GANN_SQUARE_SUPPORT', 'GANN_SQUARE_RESISTANCE',
                'GANN_PRICE_TARGET_UP', 'GANN_PRICE_TARGET_DOWN',
                'GANN_VOLUME_CONFIRMATION', 'GANN_TREND_ALIGNMENT'
            ]

            for col in pattern_columns:
                result[col] = False

            # 根据识别的形态设置相应的布尔值
            for pattern in identified_patterns:
                if "江恩1x1线" in pattern:
                    if "支撑" in pattern:
                        result['GANN_1X1_SUPPORT'] = True
                    elif "阻力" in pattern:
                        result['GANN_1X1_RESISTANCE'] = True
                    elif "突破" in pattern:
                        result['GANN_1X1_BREAKOUT_UP'] = True
                    elif "跌破" in pattern:
                        result['GANN_1X1_BREAKOUT_DOWN'] = True

                if "江恩1x2" in pattern:
                    if "支撑" in pattern:
                        result['GANN_1X2_SUPPORT'] = True
                    elif "阻力" in pattern:
                        result['GANN_1X2_RESISTANCE'] = True

                if "江恩2x1" in pattern:
                    if "支撑" in pattern:
                        result['GANN_2X1_SUPPORT'] = True
                    elif "阻力" in pattern:
                        result['GANN_2X1_RESISTANCE'] = True

                if "江恩多线" in pattern:
                    if "支撑" in pattern:
                        result['GANN_ANGLE_CLUSTER_SUPPORT'] = True
                    elif "阻力" in pattern:
                        result['GANN_ANGLE_CLUSTER_RESISTANCE'] = True

                if "江恩周期" in pattern:
                    if "低点" in pattern:
                        result['GANN_TIME_CYCLE_LOW'] = True
                    elif "高点" in pattern:
                        result['GANN_TIME_CYCLE_HIGH'] = True

                if "江恩方形" in pattern:
                    if "支撑" in pattern:
                        result['GANN_SQUARE_SUPPORT'] = True
                    elif "阻力" in pattern:
                        result['GANN_SQUARE_RESISTANCE'] = True

                if "江恩上涨目标" in pattern:
                    result['GANN_PRICE_TARGET_UP'] = True
                elif "江恩下跌目标" in pattern:
                    result['GANN_PRICE_TARGET_DOWN'] = True

                if "放量" in pattern or "巨量" in pattern:
                    result['GANN_VOLUME_CONFIRMATION'] = True

                if "趋势" in pattern:
                    result['GANN_TREND_ALIGNMENT'] = True

        except Exception as e:
            logger.warning(f"形态识别失败: {e}")
            # 如果形态识别失败，至少添加空的形态列
            pattern_columns = [
                'GANN_1X1_SUPPORT', 'GANN_1X1_RESISTANCE',
                'GANN_1X2_SUPPORT', 'GANN_1X2_RESISTANCE',
                'GANN_2X1_SUPPORT', 'GANN_2X1_RESISTANCE',
                'GANN_ANGLE_CLUSTER_SUPPORT', 'GANN_ANGLE_CLUSTER_RESISTANCE',
                'GANN_1X1_BREAKOUT_UP', 'GANN_1X1_BREAKOUT_DOWN',
                'GANN_TIME_CYCLE_LOW', 'GANN_TIME_CYCLE_HIGH',
                'GANN_SQUARE_SUPPORT', 'GANN_SQUARE_RESISTANCE',
                'GANN_PRICE_TARGET_UP', 'GANN_PRICE_TARGET_DOWN',
                'GANN_VOLUME_CONFIRMATION', 'GANN_TREND_ALIGNMENT'
            ]

            for col in pattern_columns:
                result[col] = False

        return result
    
    def _calculate_angle_lines(self, data: pd.DataFrame, result: pd.DataFrame, 
                             pivot_idx: int, pivot_price: float, 
                             price_unit: float, time_unit: int) -> pd.DataFrame:
        """
        计算江恩角度线
        
        Args:
            data: 输入数据
            result: 结果数据框
            pivot_idx: 支点索引
            pivot_price: 支点价格
            price_unit: 价格单位
            time_unit: 时间单位
            
        Returns:
            pd.DataFrame: 更新后的结果数据框
        """
        n = len(data)
        
        # 初始化角度线列
        for angle in GannAngle:
            column_name = f"gann_{angle.name.lower().replace('angle_', '')}"
            result[column_name] = np.nan
        
        # 计算每个角度线
        for angle, (price_ratio, time_ratio) in self.ANGLE_RATIOS.items():
            column_name = f"gann_{angle.name.lower().replace('angle_', '')}"
            # 计算角度线的斜率
            slope = (price_ratio * price_unit) / (time_ratio * time_unit)
            
            # 计算角度线上的点
            for i in range(n):
                # 计算与支点的时间差
                time_diff = i - pivot_idx
                
                # 计算角度线上的价格
                if time_diff >= 0:
                    # 向上角度线
                    angle_price = pivot_price + slope * time_diff
                else:
                    # 向下角度线
                    angle_price = pivot_price - slope * abs(time_diff)
                
                # 添加到结果
                result.iloc[i, result.columns.get_loc(column_name)] = angle_price
        
        return result
    
    def _calculate_time_cycles(self, data: pd.DataFrame, result: pd.DataFrame, pivot_idx: int) -> pd.DataFrame:
        """
        计算江恩时间周期
        
        Args:
            data: 输入数据
            result: 结果数据框
            pivot_idx: 支点索引
            
        Returns:
            pd.DataFrame: 更新后的结果数据框
        """
        n = len(data)
        
        # 初始化时间周期列
        result["time_cycle"] = np.nan
        
        # 计算每个时间周期
        for cycle, days in self.TIME_CYCLES.items():
            cycle_name = f"cycle_{days}"
            result[cycle_name] = np.nan
            
            # 计算周期点
            for i in range(days, n, days):
                target_idx = pivot_idx + i
                if target_idx < n:
                    result.iloc[target_idx, result.columns.get_loc(cycle_name)] = data["close"].iloc[target_idx]
                    result.iloc[target_idx, result.columns.get_loc("time_cycle")] = days
        
        return result
    
    def calculate_gann_square(self, data: pd.DataFrame, pivot_idx: int = None, 
                            levels: int = 9, *args, **kwargs) -> pd.DataFrame:
        """
        计算江恩方格
        
        Args:
            data: 输入数据
            pivot_idx: 支点索引，如果为None则使用第一个点
            levels: 方格级别数，默认为9
            
        Returns:
            pd.DataFrame: 江恩方格计算结果
        """
        # 确保数据包含必需的列
        required_columns = ["close"]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"数据必须包含'{col}'列")
        
        # 如果未指定支点，则使用第一个点
        if pivot_idx is None:
            pivot_idx = 0
        
        # 获取支点价格
        pivot_price = data["close"].iloc[pivot_idx]
        
        # 计算方格价格单位
        # 使用价格的平方根作为基本单位
        price_unit = np.sqrt(pivot_price)
        
        # 初始化结果数据框
        result = pd.DataFrame(index=pd.RangeIndex(2*levels+1))
        result["level"] = np.arange(-levels, levels+1)
        
        # 计算价格和时间方格
        result["price"] = pivot_price + result["level"] * price_unit
        result["time_factor"] = result["level"].apply(lambda x: abs(x) if x != 0 else 1)
        
        return result
    
    def calculate_raw_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算江恩工具指标的原始评分
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            pd.DataFrame: 包含原始评分的DataFrame
        """
        # 计算指标值
        indicator_data = self.calculate(data)
        
        # 初始化评分
        score = pd.Series(50.0, index=data.index)  # 基础分50分
        
        if 'close' not in data.columns:
            return pd.DataFrame({'score': score}, index=data.index)
        
        close_price = data['close']
        
        # 1. 江恩角度线支撑阻力评分（±20分）
        # 1x1角度线最重要
        if GannAngle.ANGLE_1X1.value in indicator_data.columns:
            angle_1x1 = indicator_data[GannAngle.ANGLE_1X1.value]
            
            # 价格在1x1线上方为支撑，下方为阻力
            support_mask = close_price > angle_1x1
            resistance_mask = close_price < angle_1x1
            
            # 计算距离1x1线的距离
            distance_pct = abs(close_price - angle_1x1) / close_price
            
            # 距离越近，信号越强
            close_to_line = distance_pct < 0.02  # 2%以内认为接近
            
            score.loc[support_mask & close_to_line] += 20
            score.loc[resistance_mask & close_to_line] -= 20
            
            # 一般距离的支撑阻力
            medium_distance = (distance_pct >= 0.02) & (distance_pct < 0.05)
            score.loc[support_mask & medium_distance] += 10
            score.loc[resistance_mask & medium_distance] -= 10
        
        # 2. 其他重要角度线评分（±15分）
        important_angles = [GannAngle.ANGLE_1X2, GannAngle.ANGLE_2X1, 
                          GannAngle.ANGLE_1X4, GannAngle.ANGLE_4X1]
        
        for angle in important_angles:
            angle_name = angle.value
            if angle_name in indicator_data.columns:
                angle_line = indicator_data[angle_name]
                
                # 价格接近角度线时的支撑阻力
                distance_pct = abs(close_price - angle_line) / close_price
                close_to_line = distance_pct < 0.03
                
                support_mask = (close_price > angle_line) & close_to_line
                resistance_mask = (close_price < angle_line) & close_to_line
                
                score.loc[support_mask] += 15
                score.loc[resistance_mask] -= 15
        
        # 3. 江恩时间周期评分（±20分）
        if 'time_cycle' in indicator_data.columns:
            time_cycle_mask = ~indicator_data['time_cycle'].isna()
            
            # 在时间周期点附近的评分
            for i in range(len(data)):
                if time_cycle_mask.iloc[i]:
                    cycle_days = indicator_data['time_cycle'].iloc[i]
                    
                    # 根据周期重要性调整评分
                    if cycle_days in [144, 360]:  # 重要周期
                        # 检查是否在周期转折点
                        if i >= 5 and i < len(data) - 5:
                            # 计算前后5天的价格变化
                            before_price = close_price.iloc[i-5:i].mean()
                            after_price = close_price.iloc[i+1:i+6].mean()
                            current_price = close_price.iloc[i]
                            
                            # 如果是低点转折（看涨）
                            if current_price < before_price and after_price > current_price:
                                score.iloc[i] += 20
                            # 如果是高点转折（看跌）
                            elif current_price > before_price and after_price < current_price:
                                score.iloc[i] -= 20
                    
                    elif cycle_days in [90, 180]:  # 中等重要周期
                        if i >= 3 and i < len(data) - 3:
                            before_price = close_price.iloc[i-3:i].mean()
                            after_price = close_price.iloc[i+1:i+4].mean()
                            current_price = close_price.iloc[i]
                            
                            if current_price < before_price and after_price > current_price:
                                score.iloc[i] += 15
                            elif current_price > before_price and after_price < current_price:
                                score.iloc[i] -= 15
        
        # 4. 江恩角度线突破评分（±25分）
        # 检查价格突破重要角度线
        if GannAngle.ANGLE_1X1.value in indicator_data.columns:
            angle_1x1 = indicator_data[GannAngle.ANGLE_1X1.value]
            
            # 向上突破1x1线
            upward_break = (close_price > angle_1x1) & (close_price.shift(1) <= angle_1x1.shift(1))
            score.loc[upward_break] += 25
            
            # 向下突破1x1线
            downward_break = (close_price < angle_1x1) & (close_price.shift(1) >= angle_1x1.shift(1))
            score.loc[downward_break] -= 25
        
        # 5. 成交量确认评分（±10分）
        if 'volume' in data.columns:
            volume = data['volume']
            vol_ma5 = volume.rolling(window=5).mean()
            vol_ratio = volume / vol_ma5
            
            # 放量突破角度线
            high_volume = vol_ratio > 1.5
            
            # 结合角度线突破和放量
            if GannAngle.ANGLE_1X1.value in indicator_data.columns:
                angle_1x1 = indicator_data[GannAngle.ANGLE_1X1.value]
                
                upward_break = (close_price > angle_1x1) & (close_price.shift(1) <= angle_1x1.shift(1))
                downward_break = (close_price < angle_1x1) & (close_price.shift(1) >= angle_1x1.shift(1))
                
                volume_confirmed_up = upward_break & high_volume
                volume_confirmed_down = downward_break & high_volume
                
                score.loc[volume_confirmed_up] += 10
                score.loc[volume_confirmed_down] -= 10
        
        # 6. 江恩扇形线聚集评分（±15分）
        # 当多条角度线聚集时，支撑阻力更强
        angle_columns = [angle.value for angle in GannAngle if angle.value in indicator_data.columns]
        
        if len(angle_columns) >= 3:
            for i in range(len(data)):
                current_price = close_price.iloc[i]
                
                # 计算当前价格与各角度线的距离
                distances = []
                for angle_col in angle_columns:
                    if pd.notna(indicator_data[angle_col].iloc[i]):
                        distance = abs(current_price - indicator_data[angle_col].iloc[i]) / current_price
                        distances.append(distance)
                
                if distances:
                    # 如果有多条线在3%范围内聚集
                    close_lines = sum(1 for d in distances if d < 0.03)
                    
                    if close_lines >= 3:
                        # 判断是支撑还是阻力
                        above_lines = sum(1 for angle_col in angle_columns 
                                        if pd.notna(indicator_data[angle_col].iloc[i]) and 
                                        current_price > indicator_data[angle_col].iloc[i])
                        
                        if above_lines > len(angle_columns) / 2:
                            score.iloc[i] += 15  # 多线支撑
                        else:
                            score.iloc[i] -= 15  # 多线阻力
        
        # 7. 江恩价格目标评分（±25分）
        # 基于江恩理论的价格目标计算
        if len(data) >= 30:
            # 寻找重要的高低点
            high_30 = close_price.rolling(window=30).max()
            low_30 = close_price.rolling(window=30).min()
            
            # 计算江恩价格目标（使用平方根关系）
            for i in range(30, len(data)):
                current_price = close_price.iloc[i]
                recent_high = high_30.iloc[i]
                recent_low = low_30.iloc[i]
                
                # 江恩价格目标计算
                price_range = recent_high - recent_low
                sqrt_low = np.sqrt(recent_low)
                sqrt_high = np.sqrt(recent_high)
                
                # 上涨目标
                target_up = (sqrt_low + (sqrt_high - sqrt_low) * 1.618) ** 2
                # 下跌目标
                target_down = (sqrt_high - (sqrt_high - sqrt_low) * 1.618) ** 2
                
                # 如果接近目标价位（5%以内）
                if abs(current_price - target_up) / current_price < 0.05:
                    score.iloc[i] += 25  # 接近上涨目标
                elif abs(current_price - target_down) / current_price < 0.05:
                    score.iloc[i] -= 25  # 接近下跌目标
        
        # 确保评分在0-100范围内
        score = score.clip(0, 100)
        
        return pd.DataFrame({'score': score}, index=data.index)
    
    def identify_patterns(self, data: pd.DataFrame) -> List[str]:
        """
        识别江恩工具相关的技术形态
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            List[str]: 识别出的形态列表
        """
        patterns = []
        
        # 计算指标值
        indicator_data = self.calculate(data)
        
        if len(indicator_data) < 10 or 'close' not in data.columns:
            return patterns
        
        close_price = data['close']
        latest_price = close_price.iloc[-1]
        
        # 1. 江恩角度线分析
        angle_analysis = []
        
        for angle in GannAngle:
            angle_name = angle.value
            if angle_name in indicator_data.columns:
                angle_line = indicator_data[angle_name].iloc[-1]
                
                if pd.notna(angle_line):
                    distance_pct = (latest_price - angle_line) / angle_line * 100
                    
                    if abs(distance_pct) < 2:
                        if distance_pct > 0:
                            angle_analysis.append(f"接近{angle_name}支撑")
                        else:
                            angle_analysis.append(f"接近{angle_name}阻力")
                    elif abs(distance_pct) < 5:
                        if distance_pct > 0:
                            angle_analysis.append(f"临近{angle_name}支撑")
                        else:
                            angle_analysis.append(f"临近{angle_name}阻力")
        
        patterns.extend(angle_analysis)
        
        # 2. 江恩1x1线特殊分析
        if GannAngle.ANGLE_1X1.value in indicator_data.columns:
            angle_1x1 = indicator_data[GannAngle.ANGLE_1X1.value]
            latest_1x1 = angle_1x1.iloc[-1]
            
            if pd.notna(latest_1x1):
                # 检查最近5天的1x1线关系
                recent_prices = close_price.tail(5)
                recent_1x1 = angle_1x1.tail(5)
                
                above_count = sum(recent_prices > recent_1x1)
                below_count = sum(recent_prices < recent_1x1)
                
                if above_count >= 4:
                    patterns.append("江恩1x1线强支撑")
                elif below_count >= 4:
                    patterns.append("江恩1x1线强阻力")
                elif above_count == below_count:
                    patterns.append("江恩1x1线争夺激烈")
                
                # 检查突破
                if len(close_price) >= 2:
                    prev_price = close_price.iloc[-2]
                    prev_1x1 = angle_1x1.iloc[-2]
                    
                    if pd.notna(prev_1x1):
                        if latest_price > latest_1x1 and prev_price <= prev_1x1:
                            patterns.append("突破江恩1x1线")
                        elif latest_price < latest_1x1 and prev_price >= prev_1x1:
                            patterns.append("跌破江恩1x1线")
        
        # 3. 江恩时间周期分析
        if 'time_cycle' in indicator_data.columns:
            time_cycles = indicator_data['time_cycle'].dropna()
            
            if not time_cycles.empty:
                # 检查最近的时间周期
                recent_cycles = time_cycles.tail(3)
                
                for i, (idx, cycle_days) in enumerate(recent_cycles.items()):
                    cycle_days = int(cycle_days)

                    # 计算距离当前的天数（使用数值索引）
                    numeric_idx = data.index.get_loc(idx) if idx in data.index else len(data) - 1 - i
                    days_from_cycle = len(data) - 1 - numeric_idx
                    
                    if days_from_cycle <= 5:
                        if cycle_days >= 144:
                            patterns.append(f"重要江恩周期{cycle_days}天")
                        else:
                            patterns.append(f"江恩周期{cycle_days}天")
                    
                    # 预测下一个周期
                    if days_from_cycle == 0:  # 当前就是周期点
                        patterns.append(f"下一周期预计{cycle_days}天后")
        
        # 4. 江恩角度线聚集分析
        angle_columns = [angle.value for angle in GannAngle if angle.value in indicator_data.columns]
        
        if len(angle_columns) >= 3:
            latest_angles = []
            for angle_col in angle_columns:
                angle_value = indicator_data[angle_col].iloc[-1]
                if pd.notna(angle_value):
                    distance = abs(latest_price - angle_value) / latest_price
                    latest_angles.append((angle_col, angle_value, distance))
            
            # 按距离排序
            latest_angles.sort(key=lambda x: x[2])
            
            # 检查聚集情况
            close_angles = [angle for angle in latest_angles if angle[2] < 0.03]
            
            if len(close_angles) >= 3:
                patterns.append(f"江恩多线聚集-{len(close_angles)}条线")
                
                # 判断聚集性质
                above_count = sum(1 for angle in close_angles if latest_price > angle[1])
                if above_count > len(close_angles) / 2:
                    patterns.append("江恩多线支撑聚集")
                else:
                    patterns.append("江恩多线阻力聚集")
        
        # 5. 江恩价格目标分析
        if len(data) >= 30:
            high_30 = close_price.rolling(window=30).max().iloc[-1]
            low_30 = close_price.rolling(window=30).min().iloc[-1]
            
            # 计算江恩价格目标
            sqrt_low = np.sqrt(low_30)
            sqrt_high = np.sqrt(high_30)
            
            # 上涨目标
            target_up = (sqrt_low + (sqrt_high - sqrt_low) * 1.618) ** 2
            # 下跌目标  
            target_down = (sqrt_high - (sqrt_high - sqrt_low) * 1.618) ** 2
            
            # 检查是否接近目标
            up_distance = abs(latest_price - target_up) / latest_price
            down_distance = abs(latest_price - target_down) / latest_price
            
            if up_distance < 0.05:
                patterns.append(f"接近江恩上涨目标{target_up:.2f}")
            elif down_distance < 0.05:
                patterns.append(f"接近江恩下跌目标{target_down:.2f}")
            
            # 当前价格位置分析
            price_position = (latest_price - low_30) / (high_30 - low_30)
            
            if price_position > 0.8:
                patterns.append("江恩价格区间-高位")
            elif price_position < 0.2:
                patterns.append("江恩价格区间-低位")
            else:
                patterns.append("江恩价格区间-中位")
        
        # 6. 江恩方形分析
        try:
            gann_square = self.calculate_gann_square(data)
            
            if not gann_square.empty:
                # 找到最接近当前价格的方形价位
                price_distances = abs(gann_square['price'] - latest_price)
                closest_idx = price_distances.idxmin()
                closest_price = gann_square.loc[closest_idx, 'price']
                closest_level = gann_square.loc[closest_idx, 'level']
                
                distance_pct = abs(latest_price - closest_price) / latest_price
                
                if distance_pct < 0.02:
                    if closest_level > 0:
                        patterns.append(f"江恩方形阻力位{closest_price:.2f}")
                    elif closest_level < 0:
                        patterns.append(f"江恩方形支撑位{closest_price:.2f}")
                    else:
                        patterns.append(f"江恩方形中心位{closest_price:.2f}")
        
        except Exception as e:
            logger.warning(f"江恩方形分析出错: {e}")
        
        # 7. 江恩角度线趋势分析
        if GannAngle.ANGLE_1X1.value in indicator_data.columns and len(data) >= 10:
            angle_1x1 = indicator_data[GannAngle.ANGLE_1X1.value]
            recent_1x1 = angle_1x1.tail(10)
            
            # 计算1x1线的趋势
            if len(recent_1x1.dropna()) >= 5:
                x = np.arange(len(recent_1x1.dropna()))
                y = recent_1x1.dropna().values
                
                # 线性回归计算趋势
                slope = np.polyfit(x, y, 1)[0]
                
                if slope > 0:
                    patterns.append("江恩1x1线上升趋势")
                elif slope < 0:
                    patterns.append("江恩1x1线下降趋势")
                else:
                    patterns.append("江恩1x1线水平趋势")
        
        # 8. 成交量与江恩线配合分析
        if 'volume' in data.columns:
            volume = data['volume']
            vol_ma5 = volume.rolling(window=5).mean()
            latest_vol_ratio = (volume / vol_ma5).iloc[-1]
            
            if pd.notna(latest_vol_ratio):
                if latest_vol_ratio > 2.0:
                    patterns.append("江恩分析-巨量配合")
                elif latest_vol_ratio > 1.5:
                    patterns.append("江恩分析-放量配合")
                elif latest_vol_ratio < 0.7:
                    patterns.append("江恩分析-缩量运行")
        
        return patterns

    def generate_signals(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        生成江恩工具指标信号
        
        Args:
            data: 输入数据，包含OHLCV数据
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            pd.DataFrame: 信号结果DataFrame，包含标准化信号
        """
        # 计算指标值
        indicator_data = self.calculate(data)
        
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
        
        # 计算评分
        if kwargs.get('use_raw_score', True):
            score_data = self.calculate_raw_score(data)
            if 'score' in score_data.columns:
                signals['score'] = score_data['score']
        
        # 获取识别的形态
        patterns = self.identify_patterns(data)
        
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
        
        # 基于江恩角度线的信号生成
        if 'close' in data.columns and GannAngle.ANGLE_1X1.value in indicator_data.columns:
            close_price = data['close']
            angle_1x1 = indicator_data[GannAngle.ANGLE_1X1.value]
            
            # 突破江恩1x1线生成买入信号
            upward_break = (close_price > angle_1x1) & (close_price.shift(1) <= angle_1x1.shift(1))
            signals.loc[upward_break, 'buy_signal'] = True
            signals.loc[upward_break, 'neutral_signal'] = False
            signals.loc[upward_break, 'trend'] = 1
            signals.loc[upward_break, 'signal_type'] = '江恩1x1线向上突破'
            signals.loc[upward_break, 'signal_desc'] = '价格突破江恩1x1角度线'
            
            # 跌破江恩1x1线生成卖出信号
            downward_break = (close_price < angle_1x1) & (close_price.shift(1) >= angle_1x1.shift(1))
            signals.loc[downward_break, 'sell_signal'] = True
            signals.loc[downward_break, 'neutral_signal'] = False
            signals.loc[downward_break, 'trend'] = -1
            signals.loc[downward_break, 'signal_type'] = '江恩1x1线向下突破'
            signals.loc[downward_break, 'signal_desc'] = '价格跌破江恩1x1角度线'
        
        # 基于江恩时间周期的信号生成
        if 'time_cycle' in indicator_data.columns and 'close' in data.columns:
            for i in range(len(signals)):
                if pd.notna(indicator_data['time_cycle'].iloc[i]):
                    cycle_days = indicator_data['time_cycle'].iloc[i]
                    
                    # 检查是否为重要周期点
                    if cycle_days >= 144 and i >= 5 and i < len(data) - 5:
                        close_price = data['close']
                        before_price = close_price.iloc[i-5:i].mean()
                        after_price = close_price.iloc[i+1:i+6].mean() if i+6 <= len(close_price) else None
                        current_price = close_price.iloc[i]
                        
                        if after_price is not None:
                            # 低点转折，生成买入信号
                            if current_price < before_price and after_price > current_price:
                                signals.loc[signals.index[i], 'buy_signal'] = True
                                signals.loc[signals.index[i], 'neutral_signal'] = False
                                signals.loc[signals.index[i], 'trend'] = 1
                                signals.loc[signals.index[i], 'signal_type'] = f'江恩{int(cycle_days)}周期低点'
                                signals.loc[signals.index[i], 'signal_desc'] = f'江恩{int(cycle_days)}日周期出现低点转折'
                                signals.loc[signals.index[i], 'confidence'] = 70
                            
                            # 高点转折，生成卖出信号
                            elif current_price > before_price and after_price < current_price:
                                signals.loc[signals.index[i], 'sell_signal'] = True
                                signals.loc[signals.index[i], 'neutral_signal'] = False
                                signals.loc[signals.index[i], 'trend'] = -1
                                signals.loc[signals.index[i], 'signal_type'] = f'江恩{int(cycle_days)}周期高点'
                                signals.loc[signals.index[i], 'signal_desc'] = f'江恩{int(cycle_days)}日周期出现高点转折'
                                signals.loc[signals.index[i], 'confidence'] = 70
        
        # 基于形态的信号增强
        if patterns:
            for pattern in patterns:
                if "突破江恩1x1线" in pattern:
                    # 找到最近的1x1线突破日
                    for i in range(len(signals)-1, max(0, len(signals)-10), -1):
                        if signals['signal_type'].iloc[i] == '江恩1x1线向上突破':
                            signals.loc[signals.index[i], 'confidence'] = min(90, signals['confidence'].iloc[i] + 10)
                            break
                
                elif "跌破江恩1x1线" in pattern:
                    # 找到最近的1x1线跌破日
                    for i in range(len(signals)-1, max(0, len(signals)-10), -1):
                        if signals['signal_type'].iloc[i] == '江恩1x1线向下突破':
                            signals.loc[signals.index[i], 'confidence'] = min(90, signals['confidence'].iloc[i] + 10)
                            break
                
                elif "江恩多线支撑聚集" in pattern:
                    # 增强最近的信号置信度
                    signals.loc[signals.index[-1], 'confidence'] = min(90, signals['confidence'].iloc[-1] + 15)
                    if not signals['signal_type'].iloc[-1]:
                        signals.loc[signals.index[-1], 'signal_type'] = '江恩多线支撑'
                        signals.loc[signals.index[-1], 'signal_desc'] = '多条江恩角度线形成支撑'
                
                elif "江恩多线阻力聚集" in pattern:
                    # 增强最近的信号置信度
                    signals.loc[signals.index[-1], 'confidence'] = min(90, signals['confidence'].iloc[-1] + 15)
                    if not signals['signal_type'].iloc[-1]:
                        signals.loc[signals.index[-1], 'signal_type'] = '江恩多线阻力'
                        signals.loc[signals.index[-1], 'signal_desc'] = '多条江恩角度线形成阻力'
        
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
                if score >= 80 and confidence >= 70:
                    signals.loc[signals.index[i], 'position_size'] = 0.1  # 10%仓位
                elif score >= 70 and confidence >= 60:
                    signals.loc[signals.index[i], 'position_size'] = 0.07  # 7%仓位
                elif score >= 60:
                    signals.loc[signals.index[i], 'position_size'] = 0.05  # 5%仓位
            elif signals['sell_signal'].iloc[i]:
                if score <= 20 and confidence >= 70:
                    signals.loc[signals.index[i], 'position_size'] = 0.1  # 10%仓位
                elif score <= 30 and confidence >= 60:
                    signals.loc[signals.index[i], 'position_size'] = 0.07  # 7%仓位
                elif score <= 40:
                    signals.loc[signals.index[i], 'position_size'] = 0.05  # 5%仓位
        
        # 计算动态止损
        if 'close' in data.columns:
            # 动态止损计算
            for i in range(len(signals)):
                if signals['buy_signal'].iloc[i] and i < len(data):
                    # 买入信号的止损
                    if GannAngle.ANGLE_1X1.value in indicator_data.columns:
                        # 使用江恩1x1线作为参考止损位
                        angle_1x1 = indicator_data[GannAngle.ANGLE_1X1.value].iloc[i]
                        if pd.notna(angle_1x1):
                            signals.loc[signals.index[i], 'stop_loss'] = angle_1x1 * 0.97  # 江恩线下方3%
                    elif GannAngle.ANGLE_1X2.value in indicator_data.columns:
                        # 使用江恩1x2线作为备选止损位
                        angle_1x2 = indicator_data[GannAngle.ANGLE_1X2.value].iloc[i]
                        if pd.notna(angle_1x2):
                            signals.loc[signals.index[i], 'stop_loss'] = angle_1x2 * 0.97
                
                elif signals['sell_signal'].iloc[i] and i < len(data):
                    # 卖出信号的止损
                    if GannAngle.ANGLE_1X1.value in indicator_data.columns:
                        # 使用江恩1x1线作为参考止损位
                        angle_1x1 = indicator_data[GannAngle.ANGLE_1X1.value].iloc[i]
                        if pd.notna(angle_1x1):
                            signals.loc[signals.index[i], 'stop_loss'] = angle_1x1 * 1.03  # 江恩线上方3%
                    elif GannAngle.ANGLE_1X2.value in indicator_data.columns:
                        # 使用江恩1x2线作为备选止损位
                        angle_1x2 = indicator_data[GannAngle.ANGLE_1X2.value].iloc[i]
                        if pd.notna(angle_1x2):
                            signals.loc[signals.index[i], 'stop_loss'] = angle_1x2 * 1.03
        
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
                if (signals['buy_signal'].iloc[i] or signals['sell_signal'].iloc[i]) and signals['volume_confirmation'].iloc[i]:
                    current_confidence = signals['confidence'].iloc[i]
                    signals.loc[signals.index[i], 'confidence'] = min(90, current_confidence + 10)
        
        return signals

    def set_parameters(self, **kwargs):
        """
        设置指标参数

        Args:
            **kwargs: 参数字典，可包含：
                - pivot_idx: 支点索引，默认0
                - price_unit: 价格单位，默认None（自动计算）
                - time_unit: 时间单位，默认1
                - levels: 江恩方格级别数，默认9
        """
        self.pivot_idx = kwargs.get('pivot_idx', 0)
        self.price_unit = kwargs.get('price_unit', None)
        self.time_unit = kwargs.get('time_unit', 1)
        self.levels = kwargs.get('levels', 9)

    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取GannTools相关形态

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
            'GANN_1X1_SUPPORT', 'GANN_1X1_RESISTANCE',
            'GANN_1X2_SUPPORT', 'GANN_1X2_RESISTANCE',
            'GANN_2X1_SUPPORT', 'GANN_2X1_RESISTANCE',
            'GANN_ANGLE_CLUSTER_SUPPORT', 'GANN_ANGLE_CLUSTER_RESISTANCE',
            'GANN_1X1_BREAKOUT_UP', 'GANN_1X1_BREAKOUT_DOWN',
            'GANN_TIME_CYCLE_LOW', 'GANN_TIME_CYCLE_HIGH',
            'GANN_SQUARE_SUPPORT', 'GANN_SQUARE_RESISTANCE',
            'GANN_PRICE_TARGET_UP', 'GANN_PRICE_TARGET_DOWN',
            'GANN_VOLUME_CONFIRMATION', 'GANN_TREND_ALIGNMENT'
        ]

        for col in pattern_columns:
            patterns[col] = False

        # 根据识别的形态设置相应的布尔值
        for pattern in identified_patterns:
            if "江恩1x1线" in pattern:
                if "支撑" in pattern:
                    patterns['GANN_1X1_SUPPORT'] = True
                elif "阻力" in pattern:
                    patterns['GANN_1X1_RESISTANCE'] = True
                elif "突破" in pattern:
                    patterns['GANN_1X1_BREAKOUT_UP'] = True
                elif "跌破" in pattern:
                    patterns['GANN_1X1_BREAKOUT_DOWN'] = True

            if "江恩1x2" in pattern:
                if "支撑" in pattern:
                    patterns['GANN_1X2_SUPPORT'] = True
                elif "阻力" in pattern:
                    patterns['GANN_1X2_RESISTANCE'] = True

            if "江恩2x1" in pattern:
                if "支撑" in pattern:
                    patterns['GANN_2X1_SUPPORT'] = True
                elif "阻力" in pattern:
                    patterns['GANN_2X1_RESISTANCE'] = True

            if "江恩多线" in pattern:
                if "支撑" in pattern:
                    patterns['GANN_ANGLE_CLUSTER_SUPPORT'] = True
                elif "阻力" in pattern:
                    patterns['GANN_ANGLE_CLUSTER_RESISTANCE'] = True

            if "江恩周期" in pattern:
                if "低点" in pattern:
                    patterns['GANN_TIME_CYCLE_LOW'] = True
                elif "高点" in pattern:
                    patterns['GANN_TIME_CYCLE_HIGH'] = True

            if "江恩方形" in pattern:
                if "支撑" in pattern:
                    patterns['GANN_SQUARE_SUPPORT'] = True
                elif "阻力" in pattern:
                    patterns['GANN_SQUARE_RESISTANCE'] = True

            if "江恩上涨目标" in pattern:
                patterns['GANN_PRICE_TARGET_UP'] = True
            elif "江恩下跌目标" in pattern:
                patterns['GANN_PRICE_TARGET_DOWN'] = True

            if "放量" in pattern or "巨量" in pattern:
                patterns['GANN_VOLUME_CONFIRMATION'] = True

            if "趋势" in pattern:
                patterns['GANN_TREND_ALIGNMENT'] = True

        return patterns

    def calculate_confidence(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """
        计算GannTools指标的置信度

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
            # 检查是否有江恩角度线数据
            gann_angle_columns = [col for col in self._result.columns if 'gann_' in col]
            if gann_angle_columns:
                # 江恩角度线数据越完整，置信度越高
                data_completeness = len(gann_angle_columns) / 9  # 假设最多9条角度线
                confidence += min(data_completeness * 0.1, 0.1)

        # 3. 基于形态的置信度
        if not patterns.empty:
            # 检查GannTools形态（只计算布尔列）
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
        if len(score) >= 144:  # 江恩重要周期
            confidence += 0.1
        elif len(score) >= 90:  # 江恩中等周期
            confidence += 0.05

        # 确保置信度在0-1范围内
        return max(0.0, min(1.0, confidence))

    def register_patterns(self):
        """
        注册GannTools指标的形态到全局形态注册表
        """
        # 注册江恩1x1线形态
        self.register_pattern_to_registry(
            pattern_id="GANN_1X1_SUPPORT",
            display_name="江恩1x1线支撑",
            description="价格在江恩1x1角度线获得支撑，重要的买入信号",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=30.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="GANN_1X1_RESISTANCE",
            display_name="江恩1x1线阻力",
            description="价格在江恩1x1角度线遇阻，重要的卖出信号",
            pattern_type="BEARISH",
            default_strength="VERY_STRONG",
            score_impact=-30.0,
            polarity="NEGATIVE"
        )

        # 注册江恩1x2线形态
        self.register_pattern_to_registry(
            pattern_id="GANN_1X2_SUPPORT",
            display_name="江恩1x2线支撑",
            description="价格在江恩1x2角度线获得支撑",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=20.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="GANN_1X2_RESISTANCE",
            display_name="江恩1x2线阻力",
            description="价格在江恩1x2角度线遇阻",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-20.0,
            polarity="NEGATIVE"
        )

        # 注册江恩2x1线形态
        self.register_pattern_to_registry(
            pattern_id="GANN_2X1_SUPPORT",
            display_name="江恩2x1线支撑",
            description="价格在江恩2x1角度线获得支撑",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=20.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="GANN_2X1_RESISTANCE",
            display_name="江恩2x1线阻力",
            description="价格在江恩2x1角度线遇阻",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-20.0,
            polarity="NEGATIVE"
        )

        # 注册江恩角度线聚集形态
        self.register_pattern_to_registry(
            pattern_id="GANN_ANGLE_CLUSTER_SUPPORT",
            display_name="江恩多线聚集支撑",
            description="多条江恩角度线聚集形成强力支撑",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=35.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="GANN_ANGLE_CLUSTER_RESISTANCE",
            display_name="江恩多线聚集阻力",
            description="多条江恩角度线聚集形成强力阻力",
            pattern_type="BEARISH",
            default_strength="VERY_STRONG",
            score_impact=-35.0,
            polarity="NEGATIVE"
        )

        # 注册江恩突破形态
        self.register_pattern_to_registry(
            pattern_id="GANN_1X1_BREAKOUT_UP",
            display_name="江恩1x1线向上突破",
            description="价格向上突破江恩1x1角度线",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=40.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="GANN_1X1_BREAKOUT_DOWN",
            display_name="江恩1x1线向下突破",
            description="价格向下突破江恩1x1角度线",
            pattern_type="BEARISH",
            default_strength="VERY_STRONG",
            score_impact=-40.0,
            polarity="NEGATIVE"
        )

        # 注册江恩时间周期形态
        self.register_pattern_to_registry(
            pattern_id="GANN_TIME_CYCLE_LOW",
            display_name="江恩时间周期低点",
            description="在江恩重要时间周期出现低点转折",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=25.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="GANN_TIME_CYCLE_HIGH",
            display_name="江恩时间周期高点",
            description="在江恩重要时间周期出现高点转折",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-25.0,
            polarity="NEGATIVE"
        )

        # 注册江恩方格形态
        self.register_pattern_to_registry(
            pattern_id="GANN_SQUARE_SUPPORT",
            display_name="江恩方格支撑",
            description="价格在江恩方格支撑位获得支撑",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=15.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="GANN_SQUARE_RESISTANCE",
            display_name="江恩方格阻力",
            description="价格在江恩方格阻力位遇阻",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-15.0,
            polarity="NEGATIVE"
        )

        # 注册江恩价格目标形态
        self.register_pattern_to_registry(
            pattern_id="GANN_PRICE_TARGET_UP",
            display_name="江恩上涨价格目标",
            description="价格接近江恩理论计算的上涨目标位，需要观察反应",
            pattern_type="NEUTRAL",
            default_strength="STRONG",
            score_impact=0.0,
            polarity="NEUTRAL"
        )

        self.register_pattern_to_registry(
            pattern_id="GANN_PRICE_TARGET_DOWN",
            display_name="江恩下跌价格目标",
            description="价格接近江恩理论计算的下跌目标位，需要观察反应",
            pattern_type="NEUTRAL",
            default_strength="STRONG",
            score_impact=0.0,
            polarity="NEUTRAL"
        )

        # 注册确认形态
        self.register_pattern_to_registry(
            pattern_id="GANN_VOLUME_CONFIRMATION",
            display_name="江恩成交量确认",
            description="江恩信号伴随成交量放大确认，增强信号可靠性",
            pattern_type="NEUTRAL",
            default_strength="MEDIUM",
            score_impact=0.0,
            polarity="NEUTRAL"
        )

        self.register_pattern_to_registry(
            pattern_id="GANN_TREND_ALIGNMENT",
            display_name="江恩趋势一致",
            description="江恩信号与主趋势方向一致，增强信号可靠性",
            pattern_type="NEUTRAL",
            default_strength="STRONG",
            score_impact=0.0,
            polarity="NEUTRAL"
        )

    def generate_trading_signals(self, data: pd.DataFrame, **kwargs) -> dict:
        """
        生成GannTools交易信号

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

        # 基于评分计算信号强度
        scores = detailed_signals['score']

        # 买入信号强度
        signal_strength[buy_signal] = (scores[buy_signal] - 50) / 50.0

        # 卖出信号强度（负值）
        signal_strength[sell_signal] = -(50 - scores[sell_signal]) / 50.0

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
        return "GANNTOOLS"

    def get_pattern_info(self, pattern_id: str) -> dict:
        """
        获取形态信息
        
        Args:
            pattern_id: 形态ID
            
        Returns:
            dict: 形态信息字典
        """
        # 默认形态信息映射
        pattern_info_map = {
            # 基础形态
            'bullish': {'name': '看涨形态', 'description': '指标显示看涨信号', 'type': 'BULLISH'},
            'bearish': {'name': '看跌形态', 'description': '指标显示看跌信号', 'type': 'BEARISH'},
            'neutral': {'name': '中性形态', 'description': '指标显示中性信号', 'type': 'NEUTRAL'},
            
            # 通用形态
            'strong_signal': {'name': '强信号', 'description': '强烈的技术信号', 'type': 'STRONG'},
            'weak_signal': {'name': '弱信号', 'description': '较弱的技术信号', 'type': 'WEAK'},
            'trend_up': {'name': '上升趋势', 'description': '价格呈上升趋势', 'type': 'BULLISH'},
            'trend_down': {'name': '下降趋势', 'description': '价格呈下降趋势', 'type': 'BEARISH'},
        }
        
        # 默认形态信息
        default_pattern = {
            'name': pattern_id.replace('_', ' ').title(),
            'description': f'{pattern_id}形态',
            'type': 'UNKNOWN'
        }
        
        return pattern_info_map.get(pattern_id, default_pattern)
