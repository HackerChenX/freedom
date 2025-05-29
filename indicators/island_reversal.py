#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
岛型反转指标模块

实现岛型反转形态识别功能，用于识别跳空+反向跳空形成孤岛的形态
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger

logger = get_logger(__name__)


class IslandReversal(BaseIndicator):
    """
    岛型反转指标
    
    识别跳空+反向跳空形成孤岛的价格形态，用于短期急剧反转信号
    """
    
    def __init__(self, gap_threshold: float = 0.01, island_max_days: int = 5):
        """
        初始化岛型反转指标
        
        Args:
            gap_threshold: 跳空阈值，默认为1%
            island_max_days: 岛型最大天数，默认为5日
        """
        super().__init__(name="IslandReversal", description="岛型反转指标，识别跳空+反向跳空形成孤岛的形态")
        self.gap_threshold = gap_threshold
        self.island_max_days = island_max_days
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算岛型反转指标
        
        Args:
            data: 输入数据，包含OHLC数据
            
        Returns:
            pd.DataFrame: 计算结果，包含岛型反转信号
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["open", "high", "low", "close"])
        
        # 初始化结果数据框
        result = pd.DataFrame(index=data.index)
        
        # 初始化跳空和岛型反转标记
        result["up_gap"] = False  # 向上跳空
        result["down_gap"] = False  # 向下跳空
        result["top_island_reversal"] = False  # 顶部岛型反转
        result["bottom_island_reversal"] = False  # 底部岛型反转
        
        # 计算跳空
        for i in range(1, len(data)):
            # 向上跳空：当日最低价高于前日最高价
            if data["low"].iloc[i] > data["high"].iloc[i-1] * (1 + self.gap_threshold):
                result.iloc[i, result.columns.get_loc("up_gap")] = True
            
            # 向下跳空：当日最高价低于前日最低价
            if data["high"].iloc[i] < data["low"].iloc[i-1] * (1 - self.gap_threshold):
                result.iloc[i, result.columns.get_loc("down_gap")] = True
        
        # 识别岛型反转
        for i in range(self.island_max_days + 1, len(data)):
            # 查找前island_max_days天内的跳空
            for j in range(1, self.island_max_days + 1):
                # 顶部岛型反转：先向上跳空进入，然后向下跳空离开
                if (result["up_gap"].iloc[i-j] and 
                    result["down_gap"].iloc[i]):
                    
                    # 检查中间区域是否孤立（无与前后区间重叠）
                    island_min = data["low"].iloc[i-j:i].min()
                    island_max = data["high"].iloc[i-j:i].max()
                    
                    before_max = data["high"].iloc[i-j-1]
                    after_min = data["low"].iloc[i]
                    
                    # 增强条件：确保岛型区域明显孤立
                    if (island_min > before_max * (1 + self.gap_threshold*0.5) and 
                        island_max > after_min * (1 + self.gap_threshold*0.5) and
                        data["close"].iloc[i-j:i].mean() > data["close"].iloc[i-j-5:i-j].mean()):
                        result.iloc[i, result.columns.get_loc("top_island_reversal")] = True
                        break
                
                # 底部岛型反转：先向下跳空进入，然后向上跳空离开
                if (result["down_gap"].iloc[i-j] and 
                    result["up_gap"].iloc[i]):
                    
                    # 检查中间区域是否孤立（无与前后区间重叠）
                    island_min = data["low"].iloc[i-j:i].min()
                    island_max = data["high"].iloc[i-j:i].max()
                    
                    before_min = data["low"].iloc[i-j-1]
                    after_max = data["high"].iloc[i]
                    
                    # 增强条件：确保岛型区域明显孤立
                    if (island_max < before_min * (1 - self.gap_threshold*0.5) and 
                        island_min < after_max * (1 - self.gap_threshold*0.5) and
                        data["close"].iloc[i-j:i].mean() < data["close"].iloc[i-j-5:i-j].mean()):
                        result.iloc[i, result.columns.get_loc("bottom_island_reversal")] = True
                        break
        
        return result
    
    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成岛型反转信号
        
        Args:
            data: 输入数据，包含岛型反转指标
            
        Returns:
            pd.DataFrame: 包含岛型反转交易信号的数据框
        """
        if "top_island_reversal" not in data.columns:
            data = self.calculate(data)
        
        # 初始化信号列
        data["island_signal"] = 0
        
        # 生成交易信号
        for i in range(len(data)):
            # 顶部岛型反转：卖出信号
            if data["top_island_reversal"].iloc[i]:
                data.iloc[i, data.columns.get_loc("island_signal")] = -1
            
            # 底部岛型反转：买入信号
            elif data["bottom_island_reversal"].iloc[i]:
                data.iloc[i, data.columns.get_loc("island_signal")] = 1
        
        return data
    
    def get_island_details(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        获取岛型反转的详细信息
        
        Args:
            data: 输入数据，包含OHLC数据
            
        Returns:
            List[Dict[str, Any]]: 岛型反转详细信息列表
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["open", "high", "low", "close"])
        
        # 计算岛型反转指标
        indicator_result = self.calculate(data)
        
        island_details = []
        
        # 查找所有岛型反转
        for i in range(self.island_max_days + 1, len(data)):
            if indicator_result["top_island_reversal"].iloc[i] or indicator_result["bottom_island_reversal"].iloc[i]:
                # 确定岛型类型
                island_type = "顶部岛型反转" if indicator_result["top_island_reversal"].iloc[i] else "底部岛型反转"
                
                # 查找岛型起始位置
                start_idx = i
                for j in range(1, self.island_max_days + 1):
                    if (island_type == "顶部岛型反转" and indicator_result["up_gap"].iloc[i-j]) or \
                       (island_type == "底部岛型反转" and indicator_result["down_gap"].iloc[i-j]):
                        start_idx = i - j
                        break
                
                # 计算岛型区域价格特征
                island_data = data.iloc[start_idx:i+1]
                island_high = island_data["high"].max()
                island_low = island_data["low"].min()
                island_days = len(island_data)
                
                # 计算前后跳空幅度
                if start_idx > 0 and i < len(data):
                    if island_type == "顶部岛型反转":
                        entry_gap = (island_data["low"].iloc[0] - data["high"].iloc[start_idx-1]) / data["high"].iloc[start_idx-1]
                        exit_gap = (island_data["high"].iloc[-1] - data["low"].iloc[i]) / data["low"].iloc[i]
                    else:
                        entry_gap = (data["low"].iloc[start_idx-1] - island_data["high"].iloc[0]) / island_data["high"].iloc[0]
                        exit_gap = (data["high"].iloc[i] - island_data["low"].iloc[-1]) / island_data["low"].iloc[-1]
                    
                    # 保存岛型信息
                    island_info = {
                        "type": island_type,
                        "start_date": data.index[start_idx],
                        "end_date": data.index[i],
                        "days": island_days,
                        "high": island_high,
                        "low": island_low,
                        "entry_gap": abs(entry_gap),
                        "exit_gap": abs(exit_gap)
                    }
                    
                    island_details.append(island_info)
        
        return island_details
    
    def get_gap_statistics(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        获取跳空统计信息
        
        Args:
            data: 输入数据，包含OHLC数据
            
        Returns:
            Dict[str, float]: 跳空统计信息
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["open", "high", "low", "close"])
        
        # 计算岛型反转指标
        indicator_result = self.calculate(data)
        
        # 计算跳空统计信息
        up_gaps = indicator_result["up_gap"].sum()
        down_gaps = indicator_result["down_gap"].sum()
        top_islands = indicator_result["top_island_reversal"].sum()
        bottom_islands = indicator_result["bottom_island_reversal"].sum()
        
        total_bars = len(data)
        
        statistics = {
            "up_gap_ratio": up_gaps / total_bars if total_bars > 0 else 0,
            "down_gap_ratio": down_gaps / total_bars if total_bars > 0 else 0,
            "top_island_ratio": top_islands / total_bars if total_bars > 0 else 0,
            "bottom_island_ratio": bottom_islands / total_bars if total_bars > 0 else 0,
            "island_to_gap_ratio": (top_islands + bottom_islands) / (up_gaps + down_gaps) if (up_gaps + down_gaps) > 0 else 0
        }
        
        return statistics
    
    def calculate_raw_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算岛型反转指标的原始评分
        
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
        
        # 1. 岛型反转确认评分（±45分）
        if 'bottom_island_reversal' in indicator_data.columns:
            bottom_island_mask = indicator_data['bottom_island_reversal']
            score.loc[bottom_island_mask] += 45  # 底部岛型反转是强烈买入信号
        
        if 'top_island_reversal' in indicator_data.columns:
            top_island_mask = indicator_data['top_island_reversal']
            score.loc[top_island_mask] -= 45  # 顶部岛型反转是强烈卖出信号
        
        # 2. 岛型持续天数评分（±15分）
        # 获取岛型反转的详细信息
        try:
            # 传递原始数据而不是指标数据
            island_details = self.get_island_details(data)
            
            for island_info in island_details:
                end_date = island_info['end_date']
                island_days = island_info['days']
                island_type = island_info['type']
                
                if end_date in data.index:
                    idx_pos = data.index.get_loc(end_date)
                    
                    # 持续天数评分：天数越多，信号越强
                    duration_score = min(15, island_days * 3)  # 最多15分
                    
                    if island_type == "底部岛型反转":
                        score.iloc[idx_pos] += duration_score
                    elif island_type == "顶部岛型反转":
                        score.iloc[idx_pos] -= duration_score
        
        except Exception as e:
            logger.warning(f"计算岛型持续天数评分时出错: {e}")
        
        # 3. 跳空幅度评分（±10分）
        try:
            # 传递原始数据而不是指标数据
            island_details = self.get_island_details(data)
            
            for island_info in island_details:
                end_date = island_info['end_date']
                entry_gap = island_info['entry_gap']
                exit_gap = island_info['exit_gap']
                island_type = island_info['type']
                
                if end_date in data.index:
                    idx_pos = data.index.get_loc(end_date)
                    
                    # 跳空幅度评分：幅度越大，信号越可靠
                    avg_gap = (entry_gap + exit_gap) / 2
                    gap_score = min(10, avg_gap * 100)  # 最多10分
                    
                    if island_type == "底部岛型反转":
                        score.iloc[idx_pos] += gap_score
                    elif island_type == "顶部岛型反转":
                        score.iloc[idx_pos] -= gap_score
        
        except Exception as e:
            logger.warning(f"计算跳空幅度评分时出错: {e}")
        
        # 4. 成交量确认评分（±10分）
        if 'volume' in data.columns:
            volume = data['volume']
            vol_ma5 = volume.rolling(window=5).mean()
            vol_ratio = volume / vol_ma5
            
            # 放量确认岛型反转
            high_volume = vol_ratio > 1.5
            
            if 'bottom_island_reversal' in indicator_data.columns:
                bottom_island_mask = indicator_data['bottom_island_reversal']
                volume_confirmed_bottom = bottom_island_mask & high_volume
                score.loc[volume_confirmed_bottom] += 10
            
            if 'top_island_reversal' in indicator_data.columns:
                top_island_mask = indicator_data['top_island_reversal']
                volume_confirmed_top = top_island_mask & high_volume
                score.loc[volume_confirmed_top] -= 10
        
        # 5. 岛型反转位置评分（±20分）
        # 在重要技术位置的岛型反转更有意义
        if len(data) >= 20:
            ma20 = close_price.rolling(window=20).mean()
            
            # 检查岛型反转是否发生在重要位置
            if 'bottom_island_reversal' in indicator_data.columns:
                bottom_island_mask = indicator_data['bottom_island_reversal']
                
                for i in range(len(data)):
                    if bottom_island_mask.iloc[i] and pd.notna(ma20.iloc[i]):
                        current_price = close_price.iloc[i]
                        ma20_value = ma20.iloc[i]
                        
                        # 在重要支撑位附近的底部岛型反转
                        if abs(current_price - ma20_value) / ma20_value < 0.02:  # 2%范围内
                            score.iloc[i] += 20
                        elif current_price < ma20_value * 0.95:  # 明显低于均线
                            score.iloc[i] += 15
            
            if 'top_island_reversal' in indicator_data.columns:
                top_island_mask = indicator_data['top_island_reversal']
                
                for i in range(len(data)):
                    if top_island_mask.iloc[i] and pd.notna(ma20.iloc[i]):
                        current_price = close_price.iloc[i]
                        ma20_value = ma20.iloc[i]
                        
                        # 在重要阻力位附近的顶部岛型反转
                        if abs(current_price - ma20_value) / ma20_value < 0.02:  # 2%范围内
                            score.iloc[i] -= 20
                        elif current_price > ma20_value * 1.05:  # 明显高于均线
                            score.iloc[i] -= 15
        
        # 6. 岛型反转后的确认评分（±25分）
        # 检查岛型反转后的价格走势是否确认反转
        if ('bottom_island_reversal' in indicator_data.columns or 
            'top_island_reversal' in indicator_data.columns):
            
            # 获取所有岛型反转的位置
            island_indices = []
            
            if 'bottom_island_reversal' in indicator_data.columns:
                bottom_indices = indicator_data.index[indicator_data['bottom_island_reversal']].tolist()
                island_indices.extend([(idx, 'bottom') for idx in bottom_indices])
            
            if 'top_island_reversal' in indicator_data.columns:
                top_indices = indicator_data.index[indicator_data['top_island_reversal']].tolist()
                island_indices.extend([(idx, 'top') for idx in top_indices])
            
            # 检查每个岛型反转后的走势
            for island_idx, island_type in island_indices:
                idx_pos = data.index.get_loc(island_idx)
                
                # 检查反转后5天的走势
                if idx_pos + 5 < len(data):
                    island_price = close_price.iloc[idx_pos]
                    future_prices = close_price.iloc[idx_pos+1:idx_pos+6]
                    
                    if len(future_prices) >= 3:
                        # 计算反转后的价格趋势
                        price_trend = (future_prices.iloc[-1] - island_price) / island_price
                        
                        if island_type == 'bottom':
                            # 底部岛型反转后应该上涨
                            if price_trend > 0.03:  # 上涨超过3%
                                score.iloc[idx_pos] += 25
                            elif price_trend < -0.03:  # 反转失败
                                score.iloc[idx_pos] -= 15
                        
                        elif island_type == 'top':
                            # 顶部岛型反转后应该下跌
                            if price_trend < -0.03:  # 下跌超过3%
                                score.iloc[idx_pos] -= 25
                            elif price_trend > 0.03:  # 反转失败
                                score.iloc[idx_pos] += 15
        
        # 7. 跳空质量评分（±15分）
        # 评估跳空的质量和可靠性
        if ('up_gap' in indicator_data.columns and 'down_gap' in indicator_data.columns):
            up_gap_mask = indicator_data['up_gap']
            down_gap_mask = indicator_data['down_gap']
            
            # 计算跳空的实际幅度
            for i in range(1, len(data)):
                if up_gap_mask.iloc[i]:
                    # 向上跳空的幅度
                    gap_size = (data['low'].iloc[i] - data['high'].iloc[i-1]) / data['high'].iloc[i-1]
                    
                    if gap_size > self.gap_threshold * 2:  # 大幅跳空
                        score.iloc[i] += 15
                    elif gap_size > self.gap_threshold * 1.5:  # 中等跳空
                        score.iloc[i] += 10
                    else:  # 小幅跳空
                        score.iloc[i] += 5
                
                elif down_gap_mask.iloc[i]:
                    # 向下跳空的幅度
                    gap_size = (data['low'].iloc[i-1] - data['high'].iloc[i]) / data['high'].iloc[i]
                    
                    if gap_size > self.gap_threshold * 2:  # 大幅跳空
                        score.iloc[i] -= 15
                    elif gap_size > self.gap_threshold * 1.5:  # 中等跳空
                        score.iloc[i] -= 10
                    else:  # 小幅跳空
                        score.iloc[i] -= 5
        
        # 8. 岛型反转频率评分（±10分）
        # 如果短期内出现多个岛型反转，可能表示市场不稳定
        if ('bottom_island_reversal' in indicator_data.columns and 
            'top_island_reversal' in indicator_data.columns and len(data) >= 30):
            
            bottom_island_mask = indicator_data['bottom_island_reversal']
            top_island_mask = indicator_data['top_island_reversal']
            
            # 检查最近30天内的岛型反转次数
            for i in range(30, len(data)):
                recent_bottom_islands = bottom_island_mask.iloc[i-30:i+1].sum()
                recent_top_islands = top_island_mask.iloc[i-30:i+1].sum()
                total_islands = recent_bottom_islands + recent_top_islands
                
                if total_islands >= 3:  # 频繁岛型反转，市场不稳定
                    score.iloc[i] -= 10
                elif total_islands == 2:  # 适度岛型反转
                    score.iloc[i] += 5
        
        # 9. 市场趋势环境评分（±12分）
        # 在不同市场环境下，岛型反转的意义不同
        if len(data) >= 30:
            ma30 = close_price.rolling(window=30).mean()
            
            if ('bottom_island_reversal' in indicator_data.columns and 
                'top_island_reversal' in indicator_data.columns):
                
                bottom_island_mask = indicator_data['bottom_island_reversal']
                top_island_mask = indicator_data['top_island_reversal']
                
                for i in range(30, len(data)):
                    if pd.notna(ma30.iloc[i]):
                        current_price = close_price.iloc[i]
                        ma30_value = ma30.iloc[i]
                        
                        # 判断市场趋势
                        if current_price > ma30_value * 1.05:  # 上升趋势
                            if bottom_island_mask.iloc[i]:
                                score.iloc[i] += 12  # 上升趋势中的底部岛型反转更有意义
                            elif top_island_mask.iloc[i]:
                                score.iloc[i] -= 8   # 上升趋势中的顶部岛型反转意义较小
                        
                        elif current_price < ma30_value * 0.95:  # 下降趋势
                            if top_island_mask.iloc[i]:
                                score.iloc[i] -= 12  # 下降趋势中的顶部岛型反转更有意义
                            elif bottom_island_mask.iloc[i]:
                                score.iloc[i] += 8   # 下降趋势中的底部岛型反转意义较小
        
        # 确保评分在0-100范围内
        score = score.clip(0, 100)
        
        return pd.DataFrame({'score': score}, index=data.index)
    
    def identify_patterns(self, data: pd.DataFrame) -> List[str]:
        """
        识别岛型反转相关的技术形态
        
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
        
        # 1. 基本岛型反转形态识别
        if 'bottom_island_reversal' in indicator_data.columns:
            bottom_island_mask = indicator_data['bottom_island_reversal']
            recent_bottom_islands = bottom_island_mask.tail(10)
            
            if recent_bottom_islands.any():
                bottom_count = recent_bottom_islands.sum()
                patterns.append(f"近期底部岛型反转-{bottom_count}次")
                
                # 检查最新的底部岛型反转
                if bottom_island_mask.iloc[-1]:
                    patterns.append("当前底部岛型反转")
                elif recent_bottom_islands.tail(5).any():
                    patterns.append("近期底部岛型反转")
        
        if 'top_island_reversal' in indicator_data.columns:
            top_island_mask = indicator_data['top_island_reversal']
            recent_top_islands = top_island_mask.tail(10)
            
            if recent_top_islands.any():
                top_count = recent_top_islands.sum()
                patterns.append(f"近期顶部岛型反转-{top_count}次")
                
                # 检查最新的顶部岛型反转
                if top_island_mask.iloc[-1]:
                    patterns.append("当前顶部岛型反转")
                elif recent_top_islands.tail(5).any():
                    patterns.append("近期顶部岛型反转")
        
        # 2. 跳空形态分析
        if ('up_gap' in indicator_data.columns and 'down_gap' in indicator_data.columns):
            up_gap_mask = indicator_data['up_gap']
            down_gap_mask = indicator_data['down_gap']
            
            recent_up_gaps = up_gap_mask.tail(10).sum()
            recent_down_gaps = down_gap_mask.tail(10).sum()
            
            if recent_up_gaps > 0:
                patterns.append(f"近期向上跳空-{recent_up_gaps}次")
            if recent_down_gaps > 0:
                patterns.append(f"近期向下跳空-{recent_down_gaps}次")
            
            # 分析跳空频率
            total_gaps = recent_up_gaps + recent_down_gaps
            if total_gaps >= 3:
                patterns.append("高频跳空")
            elif total_gaps >= 2:
                patterns.append("中频跳空")
            elif total_gaps >= 1:
                patterns.append("低频跳空")
        
        # 3. 岛型反转详细分析
        try:
            # 传递原始数据而不是指标数据
            island_details = self.get_island_details(data)
            
            if island_details:
                # 分析最近的岛型反转
                recent_islands = [island for island in island_details 
                                if (data.index[-1] - island['end_date']).days <= 30]
                
                if recent_islands:
                    patterns.append(f"30天内岛型反转-{len(recent_islands)}次")
                    
                    # 分析岛型特征
                    for island in recent_islands[-3:]:  # 最近3个岛型
                        island_type = island['type']
                        days = island['days']
                        entry_gap = island['entry_gap']
                        exit_gap = island['exit_gap']
                        
                        # 岛型持续时间分析
                        if days == 1:
                            patterns.append(f"{island_type}-单日岛型")
                        elif days <= 3:
                            patterns.append(f"{island_type}-短期岛型")
                        else:
                            patterns.append(f"{island_type}-长期岛型")
                        
                        # 跳空幅度分析
                        avg_gap = (entry_gap + exit_gap) / 2
                        if avg_gap > 0.03:
                            patterns.append(f"{island_type}-大幅跳空")
                        elif avg_gap > 0.015:
                            patterns.append(f"{island_type}-中等跳空")
                        else:
                            patterns.append(f"{island_type}-小幅跳空")
        
        except Exception as e:
            logger.warning(f"分析岛型反转详情时出错: {e}")
        
        # 4. 跳空统计分析
        try:
            # 传递原始数据而不是指标数据
            gap_stats = self.get_gap_statistics(data)
            
            up_gap_ratio = gap_stats.get('up_gap_ratio', 0)
            down_gap_ratio = gap_stats.get('down_gap_ratio', 0)
            island_to_gap_ratio = gap_stats.get('island_to_gap_ratio', 0)
            
            # 跳空频率分析
            if up_gap_ratio > 0.05:  # 5%以上
                patterns.append("高频向上跳空")
            elif up_gap_ratio > 0.02:  # 2-5%
                patterns.append("中频向上跳空")
            elif up_gap_ratio > 0:
                patterns.append("低频向上跳空")
            
            if down_gap_ratio > 0.05:  # 5%以上
                patterns.append("高频向下跳空")
            elif down_gap_ratio > 0.02:  # 2-5%
                patterns.append("中频向下跳空")
            elif down_gap_ratio > 0:
                patterns.append("低频向下跳空")
            
            # 岛型转换率分析
            if island_to_gap_ratio > 0.3:  # 30%以上的跳空形成岛型
                patterns.append("高岛型转换率")
            elif island_to_gap_ratio > 0.1:  # 10-30%
                patterns.append("中岛型转换率")
            elif island_to_gap_ratio > 0:
                patterns.append("低岛型转换率")
        
        except Exception as e:
            logger.warning(f"分析跳空统计时出错: {e}")
        
        # 5. 成交量配合分析
        if 'volume' in data.columns:
            volume = data['volume']
            vol_ma5 = volume.rolling(window=5).mean()
            latest_vol_ratio = (volume / vol_ma5).iloc[-1]
            
            if pd.notna(latest_vol_ratio):
                if latest_vol_ratio > 2.0:
                    patterns.append("岛型反转-巨量配合")
                elif latest_vol_ratio > 1.5:
                    patterns.append("岛型反转-放量配合")
                elif latest_vol_ratio < 0.7:
                    patterns.append("岛型反转-缩量形成")
        
        # 6. 岛型反转后的走势分析
        if ('bottom_island_reversal' in indicator_data.columns or 
            'top_island_reversal' in indicator_data.columns):
            
            # 查找最近的岛型反转
            recent_islands = []
            
            if 'bottom_island_reversal' in indicator_data.columns:
                bottom_indices = indicator_data.index[indicator_data['bottom_island_reversal']].tolist()
                recent_islands.extend([(idx, 'bottom') for idx in bottom_indices[-3:]])
            
            if 'top_island_reversal' in indicator_data.columns:
                top_indices = indicator_data.index[indicator_data['top_island_reversal']].tolist()
                recent_islands.extend([(idx, 'top') for idx in top_indices[-3:]])
            
            # 分析最近岛型反转后的走势
            for island_idx, island_type in recent_islands:
                idx_pos = data.index.get_loc(island_idx)
                
                if idx_pos + 5 < len(data):
                    island_price = close_price.iloc[idx_pos]
                    current_price = close_price.iloc[-1]
                    price_change = (current_price - island_price) / island_price
                    
                    if island_type == 'bottom':
                        if price_change > 0.05:
                            patterns.append("底部岛型反转后持续上涨")
                        elif price_change < -0.03:
                            patterns.append("底部岛型反转后失败")
                        else:
                            patterns.append("底部岛型反转后震荡")
                    
                    elif island_type == 'top':
                        if price_change < -0.05:
                            patterns.append("顶部岛型反转后持续下跌")
                        elif price_change > 0.03:
                            patterns.append("顶部岛型反转后失败")
                        else:
                            patterns.append("顶部岛型反转后震荡")
        
        # 7. 市场环境分析
        if len(data) >= 30:
            ma20 = close_price.rolling(window=20).mean()
            latest_price = close_price.iloc[-1]
            latest_ma20 = ma20.iloc[-1]
            
            if pd.notna(latest_ma20):
                if latest_price > latest_ma20 * 1.05:
                    patterns.append("岛型反转-上升趋势环境")
                elif latest_price < latest_ma20 * 0.95:
                    patterns.append("岛型反转-下降趋势环境")
                else:
                    patterns.append("岛型反转-横盘环境")
        
        # 8. 岛型反转强度分析
        if ('bottom_island_reversal' in indicator_data.columns and 
            'top_island_reversal' in indicator_data.columns):
            
            # 分析岛型反转的强度
            try:
                # 传递原始数据而不是指标数据
                island_details = self.get_island_details(data)
                
                if island_details:
                    recent_island = island_details[-1]  # 最近的岛型反转
                    avg_gap = (recent_island['entry_gap'] + recent_island['exit_gap']) / 2
                    days = recent_island['days']
                    
                    # 根据跳空幅度和持续时间判断强度
                    if avg_gap > 0.03 and days >= 2:
                        patterns.append("强烈岛型反转")
                    elif avg_gap > 0.02 or days >= 3:
                        patterns.append("明显岛型反转")
                    else:
                        patterns.append("弱岛型反转")
            
            except Exception as e:
                logger.warning(f"分析岛型反转强度时出错: {e}")
        
        # 9. 岛型反转时机分析
        # 分析岛型反转出现的时机是否合适
        if len(data) >= 50:
            # 计算价格的波动性
            returns = close_price.pct_change().dropna()
            volatility = returns.rolling(window=20).std().iloc[-1]
            
            if pd.notna(volatility):
                if volatility > 0.03:  # 高波动环境
                    patterns.append("岛型反转-高波动环境")
                elif volatility > 0.015:  # 中等波动环境
                    patterns.append("岛型反转-中等波动环境")
                else:  # 低波动环境
                    patterns.append("岛型反转-低波动环境")
        
        return patterns
    
    def generate_signals(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        生成岛型反转指标信号
        
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
        
        # 基于岛型反转信号生成买卖信号
        if 'bottom_island_reversal' in indicator_data.columns:
            bottom_island_mask = indicator_data['bottom_island_reversal']
            signals.loc[bottom_island_mask, 'buy_signal'] = True
            signals.loc[bottom_island_mask, 'neutral_signal'] = False
            signals.loc[bottom_island_mask, 'trend'] = 1
            signals.loc[bottom_island_mask, 'signal_type'] = '底部岛型反转'
            signals.loc[bottom_island_mask, 'signal_desc'] = '价格经历下跳空后出现上跳空形成底部岛型'
        
        if 'top_island_reversal' in indicator_data.columns:
            top_island_mask = indicator_data['top_island_reversal']
            signals.loc[top_island_mask, 'sell_signal'] = True
            signals.loc[top_island_mask, 'neutral_signal'] = False
            signals.loc[top_island_mask, 'trend'] = -1
            signals.loc[top_island_mask, 'signal_type'] = '顶部岛型反转'
            signals.loc[top_island_mask, 'signal_desc'] = '价格经历上跳空后出现下跳空形成顶部岛型'
        
        # 计算信号强度并更新置信度
        for i in range(len(signals)):
            score = signals['score'].iloc[i]
            
            # 基于评分确定置信度
            if score >= 80 or score <= 20:
                signals.loc[signals.index[i], 'confidence'] = 85  # 岛型反转是非常明确的信号
            elif score >= 70 or score <= 30:
                signals.loc[signals.index[i], 'confidence'] = 75
            elif score >= 60 or score <= 40:
                signals.loc[signals.index[i], 'confidence'] = 65
            
            # 基于评分确定风险等级
            if score >= 80 or score <= 20:
                signals.loc[signals.index[i], 'risk_level'] = '低'  # 极强信号风险较低
            elif score >= 70 or score <= 30:
                signals.loc[signals.index[i], 'risk_level'] = '中'  # 明显信号风险一般
            else:
                signals.loc[signals.index[i], 'risk_level'] = '高'  # 弱信号风险较高
            
            # 基于评分确定建议仓位
            if signals['buy_signal'].iloc[i]:
                if score >= 80:
                    signals.loc[signals.index[i], 'position_size'] = 0.12  # 12%仓位，岛型反转是更强信号
                elif score >= 70:
                    signals.loc[signals.index[i], 'position_size'] = 0.08  # 8%仓位
                elif score >= 60:
                    signals.loc[signals.index[i], 'position_size'] = 0.05  # 5%仓位
            elif signals['sell_signal'].iloc[i]:
                if score <= 20:
                    signals.loc[signals.index[i], 'position_size'] = 0.12  # 12%仓位
                elif score <= 30:
                    signals.loc[signals.index[i], 'position_size'] = 0.08  # 8%仓位
                elif score <= 40:
                    signals.loc[signals.index[i], 'position_size'] = 0.05  # 5%仓位
        
        # 计算动态止损
        if 'high' in data.columns and 'low' in data.columns and 'close' in data.columns:
            # 计算ATR
            atr = self.atr(data['high'], data['low'], data['close'], 14)
            
            # 获取岛型详情以找到实际的岛型区域
            try:
                island_details = self.get_island_details(data)
                
                for island_info in island_details:
                    end_date = island_info['end_date']
                    island_type = island_info['type']
                    island_low = island_info['low']
                    island_high = island_info['high']
                    
                    if end_date in signals.index:
                        idx_pos = signals.index.get_loc(end_date)
                        
                        if idx_pos < len(atr):
                            current_atr = atr.iloc[idx_pos]
                            
                            if island_type == "底部岛型反转":
                                # 底部岛型反转的止损设置在岛型最低点或当前价格-2ATR，取较低者
                                signals.loc[end_date, 'stop_loss'] = min(
                                    island_low - 0.5 * current_atr,
                                    data['close'].iloc[idx_pos] - 2 * current_atr
                                )
                            
                            elif island_type == "顶部岛型反转":
                                # 顶部岛型反转的止损设置在岛型最高点或当前价格+2ATR，取较高者
                                signals.loc[end_date, 'stop_loss'] = max(
                                    island_high + 0.5 * current_atr,
                                    data['close'].iloc[idx_pos] + 2 * current_atr
                                )
            
            except Exception as e:
                logger.warning(f"计算岛型反转止损时出错: {e}")
                
                # 使用默认ATR止损方法作为备选
                for i in range(len(signals)):
                    if signals['buy_signal'].iloc[i] and i < len(data):
                        signals.loc[signals.index[i], 'stop_loss'] = data['close'].iloc[i] - 2 * atr.iloc[i]
                    
                    elif signals['sell_signal'].iloc[i] and i < len(data):
                        signals.loc[signals.index[i], 'stop_loss'] = data['close'].iloc[i] + 2 * atr.iloc[i]
        
        # 成交量确认
        if 'volume' in data.columns:
            volume = data['volume']
            vol_ma5 = volume.rolling(window=5).mean()
            vol_ratio = volume / vol_ma5
            
            # 成交量放大确认
            high_volume = vol_ratio > 1.5
            signals.loc[high_volume, 'volume_confirmation'] = True
        
        # 检测市场环境
        try:
            market_env = self.detect_market_environment(data)
            signals['market_env'] = market_env.value
        except Exception as e:
            logger.warning(f"检测市场环境时出错: {e}")
        
        return signals 