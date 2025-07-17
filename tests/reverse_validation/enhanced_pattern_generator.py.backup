#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
增强版技术指标形态数据生成器

通过反向工程技术指标计算，生成真正具备目标技术形态特征的价格数据
确保生成的数据能够被技术指标正确识别
"""

import pandas as pd
import numpy as np
import datetime
import sys
import os
from typing import Dict, List, Any, Optional, Tuple

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from technical_indicators import Technical_indicators


class Enhanced_pattern_generator:
    """增强版形态数据生成器"""

    def __init__(self):
        """初始化生成器"""
        self.indicators = Technical_indicators()

    def generate_rsi_overbought_data(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """
        生成RSI超买形态数据

        通过构造强势上涨行情，确保RSI值超过70
        """
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')

        # 构造强势上涨的价格序列
        # 前30天缓慢上涨，后20天加速上涨，确保RSI超买
        prices = []
        current_price = base_price

        # 前30天：缓慢上涨，建立上升趋势
        for i in range(30):
            daily_change = np.random.uniform(0.005, 0.02)  # 0.5%-2%的日涨幅
            current_price *= (1 + daily_change)
            prices.append(current_price)

        # 后20天：连续上涨，确保RSI超买
        for i in range(20):
            daily_change = np.random.uniform(0.01, 0.03)  # 1%-3%的日涨幅
            current_price *= (1 + daily_change)
            prices.append(current_price)

        # 生成OHLC数据
        data = self._generate_ohlc_from_close_Enhanced_Pattern_Generator(dates, prices)

        # 验证RSI是否超买
        rsi = self.indicators.calculate_rsi(data)
        if rsi.iloc[-1] < 70:
            # 如果RSI不够高，调整最后几天的价格
            adjustment_factor = 1.1
            for i in range(-5, 0):
                data.loc[data.index[i], 'close'] *= adjustment_factor
                data.loc[data.index[i], 'high'] = max(data.loc[data.index[i], 'high'],
                                                     data.loc[data.index[i], 'close'])

        return self._standardize_data_format_Enhanced_Pattern_Generator(data, 'RSI_OVERBOUGHT')

    def _generate_ohlc_from_close_Enhanced_Pattern_Generator(self, dates: pd.Datetime_index, close_prices: List[float]) -> pd.DataFrame:
        """
        从收盘价生成OHLC数据

        Args:
            dates: 日期索引
            close_prices: 收盘价列表

        Returns:
            包含OHLC的Data_frame
        """
        data = pd.DataFrame({
            'date': dates,
            'close': close_prices
        })

        # 生成开盘价（前一日收盘价加小幅波动）
        open_prices = [close_prices[0]]  # 第一天开盘价等于收盘价
        for i in range(1, len(close_prices)):
            gap = np.random.uniform(-0.02, 0.02)  # -2%到2%的跳空
            open_price = close_prices[i-1] * (1 + gap)
            open_prices.append(open_price)

        data['open'] = open_prices

        # 生成高价和低价
        high_prices = []
        low_prices = []

        for i in range(len(close_prices)):
            open_price = open_prices[i]
            close_price = close_prices[i]

            # 计算日内波动范围
            daily_range = abs(close_price - open_price) * np.random.uniform(1.2, 2.0)

            # 生成高价
            high_price = max(open_price, close_price) + daily_range * np.random.uniform(0.1, 0.5)
            high_prices.append(high_price)

            # 生成低价
            low_price = min(open_price, close_price) - daily_range * np.random.uniform(0.1, 0.5)
            low_prices.append(low_price)

        data['high'] = high_prices
        data['low'] = low_prices

        # 生成成交量
        base_volume = 10000
        volumes = []
        for i in range(len(close_prices)):
            # 价格波动大的时候成交量也大
            price_change = abs(close_prices[i] - open_prices[i]) / open_prices[i]
            volume_multiplier = 1 + price_change * 5
            volume = int(base_volume * volume_multiplier * np.random.uniform(0.5, 1.5))
            volumes.append(volume)

        data['volume'] = volumes

        return data

    def _standardize_data_format_Enhanced_Pattern_Generator(self, data: pd.DataFrame, pattern_name: str) -> pd.DataFrame:
        """
        标准化数据格式，确保与stock_info格式完全一致

        Args:
            data: 原始数据
            pattern_name: 形态名称

        Returns:
            标准化后的数据
        """
        # 计算价格变化和换手率
        price_changes = data['close'].diff().fillna(0).round(2)
        price_ranges = ((data['high'] - data['low']) / data['close'] * 100).round(2)
        turnover_rates = np.random.uniform(0.5, 5.0, len(data)).round(2)

        # 添加必需字段
        data['code'] = f'TEST_{pattern_name}'
        data['name'] = f'测试股票_{pattern_name}'
        data['level'] = 'D'  # 日线
        data['industry'] = '软件服务'
        data['seq'] = range(len(data))
        data['turnover_rate'] = turnover_rates
        data['price_change'] = price_changes
        data['price_range'] = price_ranges
        data['datetime'] = pd.to_datetime(data['date'])

        # 重新排列列顺序
        column_order = [
            'code', 'name', 'date', 'level', 'open', 'high', 'low', 'close',
            'volume', 'turnover_rate', 'price_change', 'price_range', 'industry', 'datetime', 'seq'
        ]

        return data[column_order]

    def generate_all_rsi_patterns(self) -> Dict[str, pd.DataFrame]:
        """生成所有RSI形态数据"""
        patterns = {}

        patterns['RSI_OVERBOUGHT'] = self.generate_rsi_overbought_data()
        patterns['RSI_OVERSOLD'] = self.generate_rsi_oversold_data()
        patterns['RSI_GOLDEN_CROSS'] = self.generate_rsi_golden_cross_data()
        patterns['RSI_DEATH_CROSS'] = self.generate_rsi_death_cross_data()
        patterns['RSI_DIVERGENCE'] = self.generate_rsi_divergence_data()

        return patterns

    def generate_rsi_oversold_data(self, base_price: float = 100, periods: int = 40) -> pd.DataFrame:
        """生成RSI超卖形态数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')

        prices = []
        current_price = base_price

        # 前20天：缓慢下跌
        for i in range(20):
            daily_change = np.random.uniform(-0.02, -0.005)
            current_price *= (1 + daily_change)
            prices.append(current_price)

        # 后20天：加速下跌，确保RSI超卖
        for i in range(20):
            daily_change = np.random.uniform(-0.04, -0.015)
            current_price *= (1 + daily_change)
            prices.append(current_price)

        data = self._generate_ohlc_from_close_Enhanced_Pattern_Generator(dates, prices)

        # 验证并调整RSI
        rsi = self.indicators.calculate_rsi(data)
        if rsi.iloc[-1] > 30:
            adjustment_factor = 0.9
            for i in range(-5, 0):
                data.loc[data.index[i], 'close'] *= adjustment_factor
                data.loc[data.index[i], 'low'] = min(data.loc[data.index[i], 'low'],
                                                    data.loc[data.index[i], 'close'])

        return self._standardize_data_format_Enhanced_Pattern_Generator(data, 'RSI_OVERSOLD')

    def generate_rsi_golden_cross_data(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """生成RSI金叉形态数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')

        prices = []
        current_price = base_price

        # 前25天：温和下跌，让RSI降到30-40区间
        for i in range(25):
            daily_change = np.random.uniform(-0.015, -0.002)  # 更温和的下跌
            current_price *= (1 + daily_change)
            prices.append(current_price)

        # 中间15天：横盘整理，RSI在40-50区间波动
        for i in range(15):
            daily_change = np.random.uniform(-0.008, 0.008)  # 小幅波动
            current_price *= (1 + daily_change)
            prices.append(current_price)

        # 最后10天：温和上涨，RSI突破50
        for i in range(10):
            daily_change = np.random.uniform(0.005, 0.015)  # 温和上涨
            current_price *= (1 + daily_change)
            prices.append(current_price)

        data = self._generate_ohlc_from_close_Enhanced_Pattern_Generator(dates, prices)

        # 验证RSI是否在合理范围内
        rsi = self.indicators.calculate_rsi(data)
        final_rsi = rsi.iloc[-1]

        # 如果RSI过高，调整最后几天的价格
        if final_rsi > 80:
            adjustment_factor = 0.95
            for i in range(-3, 0):
                data.loc[data.index[i], 'close'] *= adjustment_factor

        return self._standardize_data_format_Enhanced_Pattern_Generator(data, 'RSI_GOLDEN_CROSS')

    def generate_rsi_death_cross_data(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """生成RSI死叉形态数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')

        prices = []
        current_price = base_price

        # 前25天：温和上涨，让RSI升到60-70区间
        for i in range(25):
            daily_change = np.random.uniform(0.002, 0.015)  # 更温和的上涨
            current_price *= (1 + daily_change)
            prices.append(current_price)

        # 中间15天：横盘整理，RSI在50-60区间波动
        for i in range(15):
            daily_change = np.random.uniform(-0.008, 0.008)  # 小幅波动
            current_price *= (1 + daily_change)
            prices.append(current_price)

        # 最后10天：温和下跌，RSI跌破50
        for i in range(10):
            daily_change = np.random.uniform(-0.015, -0.005)  # 温和下跌
            current_price *= (1 + daily_change)
            prices.append(current_price)

        data = self._generate_ohlc_from_close_Enhanced_Pattern_Generator(dates, prices)

        # 验证RSI是否在合理范围内
        rsi = self.indicators.calculate_rsi(data)
        final_rsi = rsi.iloc[-1]

        # 如果RSI过低，调整最后几天的价格
        if final_rsi < 20:
            adjustment_factor = 1.05
            for i in range(-3, 0):
                data.loc[data.index[i], 'close'] *= adjustment_factor

        return self._standardize_data_format_Enhanced_Pattern_Generator(data, 'RSI_DEATH_CROSS')

    def generate_rsi_divergence_data(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """生成RSI背离形态数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')

        prices = []
        current_price = base_price

        # 前20天：第一波上涨
        for i in range(20):
            daily_change = np.random.uniform(0.01, 0.025)
            current_price *= (1 + daily_change)
            prices.append(current_price)

        first_peak = current_price

        # 中间15天：回调整理
        for i in range(15):
            daily_change = np.random.uniform(-0.015, 0.005)
            current_price *= (1 + daily_change)
            prices.append(current_price)

        # 最后15天：第二波上涨，价格创新高但涨幅较小
        for i in range(15):
            daily_change = np.random.uniform(0.002, 0.008)
            current_price *= (1 + daily_change)
            prices.append(current_price)

        # 确保价格创新高
        if current_price <= first_peak:
            prices[-5:] = [p * 1.02 for p in prices[-5:]]

        data = self._generate_ohlc_from_close_Enhanced_Pattern_Generator(dates, prices)
        return self._standardize_data_format_Enhanced_Pattern_Generator(data, 'RSI_DIVERGENCE')