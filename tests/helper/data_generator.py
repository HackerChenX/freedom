"""
测试数据生成器模块

提供用于生成测试数据的工具和方法
"""

import pandas as pd
import numpy as np
import datetime
from typing import List, Dict, Any, Union, Optional


class TestDataGenerator:
    """测试数据生成器类"""
    
    @staticmethod
    def generate_price_sequence(sequence_specs: List[Dict[str, Any]], 
                                base_date: str = '2023-01-01',
                                base_volume: int = 10000,
                                apply_noise: bool = True,
                                noise_level: float = 0.01) -> pd.DataFrame:
        """
        生成价格序列数据
        
        Args:
            sequence_specs: 序列规格列表，每个规格是一个字典，包含序列类型和参数
            base_date: 起始日期
            base_volume: 基础成交量
            apply_noise: 是否应用噪声
            noise_level: 噪声级别
            
        Returns:
            包含价格序列的DataFrame
        """
        # 初始化数据列表
        data_list = []
        current_date = pd.to_datetime(base_date)
        
        # 生成每个序列片段
        for spec in sequence_specs:
            # 获取序列类型和周期数
            seq_type = spec.get('type', 'sideways')
            periods = spec.get('periods', 50)
            
            # 根据序列类型生成数据
            if seq_type == 'trend':
                segment = TestDataGenerator._generate_trend(
                    periods=periods,
                    start_price=spec.get('start_price', 100),
                    end_price=spec.get('end_price', 110),
                    base_volume=base_volume,
                    volume_trend=spec.get('volume_trend', 'follow_price')
                )
            elif seq_type == 'v_shape':
                segment = TestDataGenerator._generate_v_shape(
                    periods=periods,
                    start_price=spec.get('start_price', 100),
                    bottom_price=spec.get('bottom_price', 90),
                    base_volume=base_volume
                )
            elif seq_type == 'head_shoulders':
                segment = TestDataGenerator._generate_head_shoulders(
                    periods=periods,
                    start_price=spec.get('start_price', 100),
                    peak_price=spec.get('peak_price', 110),
                    base_volume=base_volume,
                    inverse=spec.get('inverse', False)
                )
            elif seq_type == 'double_top':
                segment = TestDataGenerator._generate_double_top(
                    periods=periods,
                    start_price=spec.get('start_price', 100),
                    peak_price=spec.get('peak_price', 110),
                    base_volume=base_volume,
                    inverse=spec.get('inverse', False)
                )
            elif seq_type == 'triangle':
                segment = TestDataGenerator._generate_triangle(
                    periods=periods,
                    start_price=spec.get('start_price', 100),
                    end_price=spec.get('end_price', 110),
                    height=spec.get('height', 10),
                    base_volume=base_volume,
                    pattern=spec.get('pattern', 'ascending')
                )
            else:  # 默认为横盘
                segment = TestDataGenerator._generate_sideways(
                    periods=periods,
                    start_price=spec.get('start_price', 100),
                    volatility=spec.get('volatility', 0.02),
                    base_volume=base_volume
                )
            
            # 应用噪声
            if apply_noise:
                segment = TestDataGenerator._apply_noise(segment, noise_level)
            
            # 添加日期
            dates = [current_date + datetime.timedelta(days=i) for i in range(periods)]
            segment['date'] = dates
            
            # 更新当前日期
            current_date = dates[-1] + datetime.timedelta(days=1)
            
            # 添加到数据列表
            data_list.append(segment)
        
        # 合并所有数据
        data = pd.concat(data_list)
        
        # 设置索引
        data = data.set_index('date')
        
        # 添加股票代码和其他字段
        data['code'] = '000001'
        data['name'] = '测试股票'
        data['level'] = 'D'
        data['industry'] = '软件服务'
        data['datetime_value'] = data.index
        data['seq'] = range(len(data))
        data['turnover_rate'] = data['volume'] / base_volume * 5
        data['price_change'] = data['close'].diff().fillna(0)
        data['price_range'] = (data['high'] - data['low']) / data['close'] * 100
        
        return data
    
    @staticmethod
    def _generate_trend(periods: int, 
                        start_price: float, 
                        end_price: float, 
                        base_volume: int,
                        volume_trend: str = 'follow_price') -> pd.DataFrame:
        """
        生成趋势序列
        
        Args:
            periods: 周期数
            start_price: 起始价格
            end_price: 结束价格
            base_volume: 基础成交量
            volume_trend: 成交量趋势类型
            
        Returns:
            包含趋势序列的DataFrame
        """
        # 计算价格变化
        price_change = (end_price - start_price) / (periods - 1)
        
        # 生成价格序列
        close_prices = np.array([start_price + i * price_change for i in range(periods)])
        
        # 生成OHLC数据
        data = TestDataGenerator._generate_ohlc_from_close(close_prices)
        
        # 生成成交量
        if volume_trend == 'follow_price':
            # 成交量跟随价格变化
            if end_price > start_price:
                # 上涨趋势，成交量增加
                volume_factor = np.linspace(0.8, 1.5, periods)
            else:
                # 下跌趋势，成交量减少
                volume_factor = np.linspace(1.2, 0.5, periods)
        elif volume_trend == 'inverse':
            # 成交量与价格变化相反
            if end_price > start_price:
                # 上涨趋势，成交量减少
                volume_factor = np.linspace(1.2, 0.5, periods)
            else:
                # 下跌趋势，成交量增加
                volume_factor = np.linspace(0.8, 1.5, periods)
        else:
            # 随机成交量
            volume_factor = 0.8 + 0.4 * np.random.random(periods)
            
        data['volume'] = (base_volume * volume_factor).astype(int)
        
        return data
    
    @staticmethod
    def _generate_v_shape(periods: int, 
                          start_price: float, 
                          bottom_price: float, 
                          base_volume: int) -> pd.DataFrame:
        """
        生成V形反转序列
        
        Args:
            periods: 周期数
            start_price: 起始价格
            bottom_price: 底部价格
            base_volume: 基础成交量
            
        Returns:
            包含V形反转序列的DataFrame
        """
        # 计算中点
        mid_point = periods // 2
        
        # 生成下降和上升部分
        down_periods = mid_point
        up_periods = periods - mid_point
        
        # 生成下降部分
        down_trend = TestDataGenerator._generate_trend(
            periods=down_periods, 
            start_price=start_price, 
            end_price=bottom_price,
            base_volume=base_volume,
            volume_trend='inverse'
        )
        
        # 生成上升部分
        up_trend = TestDataGenerator._generate_trend(
            periods=up_periods, 
            start_price=bottom_price, 
            end_price=start_price,
            base_volume=base_volume,
            volume_trend='follow_price'
        )
        
        # 合并两部分
        data = pd.concat([down_trend, up_trend])
        
        return data
    
    @staticmethod
    def _generate_head_shoulders(periods: int, 
                                start_price: float, 
                                peak_price: float, 
                                base_volume: int,
                                inverse: bool = False) -> pd.DataFrame:
        """
        生成头肩顶/底序列
        
        Args:
            periods: 周期数
            start_price: 起始价格
            peak_price: 头部/底部价格
            base_volume: 基础成交量
            inverse: 是否为头肩底
            
        Returns:
            包含头肩顶/底序列的DataFrame
        """
        # 计算各个点的位置
        segment_size = periods // 6
        left_shoulder_idx = segment_size
        head_idx = 3 * segment_size
        right_shoulder_idx = 5 * segment_size
        
        # 计算各个点的价格
        if not inverse:
            # 头肩顶
            shoulder_price = start_price + 0.7 * (peak_price - start_price)
            price_points = [
                start_price,  # 起点
                shoulder_price,  # 左肩
                start_price + 0.4 * (peak_price - start_price),  # 左肩后回调
                peak_price,  # 头部
                start_price + 0.4 * (peak_price - start_price),  # 头部后回调
                shoulder_price,  # 右肩
                start_price * 0.9  # 终点（突破颈线）
            ]
        else:
            # 头肩底
            shoulder_price = start_price - 0.7 * (start_price - peak_price)
            price_points = [
                start_price,  # 起点
                shoulder_price,  # 左肩
                start_price - 0.4 * (start_price - peak_price),  # 左肩后反弹
                peak_price,  # 头部
                start_price - 0.4 * (start_price - peak_price),  # 头部后反弹
                shoulder_price,  # 右肩
                start_price * 1.1  # 终点（突破颈线）
            ]
        
        # 生成完整的价格序列
        close_prices = np.zeros(periods)
        position_points = [0, left_shoulder_idx, 2 * segment_size, 
                          head_idx, 4 * segment_size, right_shoulder_idx, periods - 1]
        
        # 在关键点之间进行线性插值
        for i in range(len(position_points) - 1):
            start_idx = position_points[i]
            end_idx = position_points[i + 1]
            start_val = price_points[i]
            end_val = price_points[i + 1]
            
            for j in range(start_idx, end_idx + 1):
                progress = (j - start_idx) / (end_idx - start_idx)
                close_prices[j] = start_val + progress * (end_val - start_val)
        
        # 生成OHLC数据
        data = TestDataGenerator._generate_ohlc_from_close(close_prices)
        
        # 生成成交量
        volume = np.ones(periods) * base_volume
        # 在肩部和头部增加成交量
        volume[left_shoulder_idx - 5:left_shoulder_idx + 5] *= 1.5
        volume[head_idx - 5:head_idx + 5] *= 2.0
        volume[right_shoulder_idx - 5:right_shoulder_idx + 5] *= 1.3
        # 在突破颈线时大幅增加成交量
        volume[-5:] *= 2.5
        
        data['volume'] = volume.astype(int)
        
        return data
    
    @staticmethod
    def _generate_double_top(periods: int, 
                            start_price: float, 
                            peak_price: float, 
                            base_volume: int,
                            inverse: bool = False) -> pd.DataFrame:
        """
        生成双顶/底序列
        
        Args:
            periods: 周期数
            start_price: 起始价格
            peak_price: 顶部/底部价格
            base_volume: 基础成交量
            inverse: 是否为双底
            
        Returns:
            包含双顶/底序列的DataFrame
        """
        # 计算各个点的位置
        segment_size = periods // 5
        first_peak_idx = segment_size
        middle_idx = 2.5 * segment_size
        second_peak_idx = 4 * segment_size
        
        # 计算各个点的价格
        if not inverse:
            # 双顶
            middle_price = start_price + 0.4 * (peak_price - start_price)
            price_points = [
                start_price,  # 起点
                peak_price,  # 第一个顶
                middle_price,  # 中间点
                peak_price,  # 第二个顶
                start_price * 0.9  # 终点（突破颈线）
            ]
        else:
            # 双底
            middle_price = start_price - 0.4 * (start_price - peak_price)
            price_points = [
                start_price,  # 起点
                peak_price,  # 第一个底
                middle_price,  # 中间点
                peak_price,  # 第二个底
                start_price * 1.1  # 终点（突破颈线）
            ]
        
        # 生成完整的价格序列
        close_prices = np.zeros(periods)
        position_points = [0, first_peak_idx, int(middle_idx), second_peak_idx, periods - 1]
        
        # 在关键点之间进行线性插值
        for i in range(len(position_points) - 1):
            start_idx = position_points[i]
            end_idx = position_points[i + 1]
            start_val = price_points[i]
            end_val = price_points[i + 1]
            
            for j in range(start_idx, end_idx + 1):
                progress = (j - start_idx) / (end_idx - start_idx)
                close_prices[j] = start_val + progress * (end_val - start_val)
        
        # 生成OHLC数据
        data = TestDataGenerator._generate_ohlc_from_close(close_prices)
        
        # 生成成交量
        volume = np.ones(periods) * base_volume
        # 在顶/底部增加成交量
        volume[first_peak_idx - 3:first_peak_idx + 3] *= 1.7
        volume[second_peak_idx - 3:second_peak_idx + 3] *= 1.5
        # 在突破颈线时大幅增加成交量
        volume[-5:] *= 2.2
        
        data['volume'] = volume.astype(int)
        
        return data
    
    @staticmethod
    def _generate_triangle(periods: int, 
                          start_price: float, 
                          end_price: float, 
                          height: float,
                          base_volume: int,
                          pattern: str = 'ascending') -> pd.DataFrame:
        """
        生成三角形整理序列
        
        Args:
            periods: 周期数
            start_price: 起始价格
            end_price: 结束价格
            height: 三角形高度
            base_volume: 基础成交量
            pattern: 三角形类型，可选'ascending'(上升),'descending'(下降),'symmetric'(对称)
            
        Returns:
            包含三角形整理序列的DataFrame
        """
        # 生成收盘价
        close_prices = np.zeros(periods)
        
        if pattern == 'ascending':
            # 上升三角形
            support_line = start_price + np.linspace(0, 0.8 * height, periods)
            resistance_line = np.ones(periods) * (start_price + height)
        elif pattern == 'descending':
            # 下降三角形
            support_line = np.ones(periods) * start_price
            resistance_line = start_price + height - np.linspace(0, 0.8 * height, periods)
        else:
            # 对称三角形
            middle_line = start_price + np.linspace(0, height / 2, periods)
            support_line = start_price - np.linspace(height / 2, 0.1 * height, periods)
            resistance_line = start_price + height - np.linspace(0, 0.8 * height, periods)
        
        # 在支撑线和阻力线之间震荡，后期突破
        for i in range(periods):
            if i < periods - 10:
                # 震荡期
                phase = (i % 10) / 10
                if phase < 0.5:
                    # 向上
                    progress = phase * 2
                    close_prices[i] = support_line[i] + progress * (resistance_line[i] - support_line[i])
                else:
                    # 向下
                    progress = (phase - 0.5) * 2
                    close_prices[i] = resistance_line[i] - progress * (resistance_line[i] - support_line[i])
            else:
                # 突破期
                if end_price > start_price:
                    # 向上突破
                    close_prices[i] = resistance_line[i] + (i - (periods - 10)) * (end_price - resistance_line[i]) / 10
                else:
                    # 向下突破
                    close_prices[i] = support_line[i] - (i - (periods - 10)) * (support_line[i] - end_price) / 10
        
        # 生成OHLC数据
        data = TestDataGenerator._generate_ohlc_from_close(close_prices)
        
        # 生成成交量
        volume = np.ones(periods) * base_volume
        # 在三角形整理期间，成交量逐渐萎缩
        volume_factor = np.linspace(1, 0.6, periods - 10)
        volume[:periods - 10] = volume[:periods - 10] * volume_factor
        # 在突破时放量
        volume[-10:] = volume[-10:] * np.linspace(0.7, 2.0, 10)
        
        data['volume'] = volume.astype(int)
        
        return data
    
    @staticmethod
    def _generate_sideways(periods: int, 
                          start_price: float, 
                          volatility: float,
                          base_volume: int) -> pd.DataFrame:
        """
        生成横盘整理序列
        
        Args:
            periods: 周期数
            start_price: 起始价格
            volatility: 波动率
            base_volume: 基础成交量
            
        Returns:
            包含横盘整理序列的DataFrame
        """
        # 生成随机波动
        changes = np.random.normal(0, volatility * start_price, periods)
        
        # 计算价格
        close_prices = np.zeros(periods)
        close_prices[0] = start_price
        
        for i in range(1, periods):
            close_prices[i] = close_prices[i-1] + changes[i]
        
        # 生成OHLC数据
        data = TestDataGenerator._generate_ohlc_from_close(close_prices)
        
        # 生成随机成交量
        volume_factor = 0.8 + 0.4 * np.random.random(periods)
        data['volume'] = (base_volume * volume_factor).astype(int)
        
        return data
    
    @staticmethod
    def _generate_ohlc_from_close(close_prices: np.ndarray) -> pd.DataFrame:
        """
        从收盘价生成OHLC数据
        
        Args:
            close_prices: 收盘价数组
            
        Returns:
            包含OHLC数据的DataFrame
        """
        periods = len(close_prices)
        
        # 生成随机波动幅度
        volatility = np.mean(np.abs(np.diff(close_prices))) * 2
        
        # 初始化数据
        open_prices = np.zeros(periods)
        high_prices = np.zeros(periods)
        low_prices = np.zeros(periods)
        
        # 生成第一天的开盘价
        open_prices[0] = close_prices[0] * (1 - 0.01 + 0.02 * np.random.random())
        
        # 生成后续日期的数据
        for i in range(1, periods):
            # 开盘价基于前一天的收盘价
            open_prices[i] = close_prices[i-1] * (1 - 0.01 + 0.02 * np.random.random())
            
            # 确定最高价和最低价
            price_range = volatility * (0.5 + np.random.random())
            
            if close_prices[i] > open_prices[i]:
                # 上涨日
                high_prices[i] = close_prices[i] + price_range * 0.3 * np.random.random()
                low_prices[i] = open_prices[i] - price_range * 0.7 * np.random.random()
            else:
                # 下跌日
                high_prices[i] = open_prices[i] + price_range * 0.7 * np.random.random()
                low_prices[i] = close_prices[i] - price_range * 0.3 * np.random.random()
        
        # 生成第一天的高低价
        price_range = volatility * (0.5 + np.random.random())
        if close_prices[0] > open_prices[0]:
            high_prices[0] = close_prices[0] + price_range * 0.3 * np.random.random()
            low_prices[0] = open_prices[0] - price_range * 0.7 * np.random.random()
        else:
            high_prices[0] = open_prices[0] + price_range * 0.7 * np.random.random()
            low_prices[0] = close_prices[0] - price_range * 0.3 * np.random.random()
        
        # 创建DataFrame
        data = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices
        })
        
        return data
    
    @staticmethod
    def _apply_noise(data: pd.DataFrame, noise_level: float) -> pd.DataFrame:
        """
        对数据应用随机噪声
        
        Args:
            data: 原始数据
            noise_level: 噪声级别
            
        Returns:
            添加噪声后的数据
        """
        result = data.copy()
        
        # 计算价格平均值作为噪声基准
        price_avg = result['close'].mean()
        
        # 为OHLC价格添加噪声
        for col in ['open', 'high', 'low', 'close']:
            noise = np.random.normal(0, noise_level * price_avg, len(result))
            result[col] = result[col] + noise
        
        # 确保高低价关系正确
        result['high'] = np.maximum(
            result['high'],
            np.maximum(result['open'], result['close'])
        )
        result['low'] = np.minimum(
            result['low'],
            np.minimum(result['open'], result['close'])
        )
        
        # 为成交量添加噪声
        volume_avg = result['volume'].mean()
        volume_noise = np.random.normal(0, noise_level * volume_avg, len(result))
        result['volume'] = np.maximum(1, result['volume'] + volume_noise.astype(int))
        
        return result 