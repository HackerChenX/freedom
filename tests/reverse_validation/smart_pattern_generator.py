#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
智能形态数据生成器

通过迭代优化和反馈调整，生成真正符合技术指标要求的数据
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

from technical_indicators import TechnicalIndicators


class SmartPatternGenerator:
    """智能形态数据生成器"""

    def __init__(self):
        """初始化生成器"""
        self.indicators = TechnicalIndicators()

    def generate_rsi_golden_cross_data_v2(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """
        智能生成RSI金叉形态数据

        通过迭代调整确保RSI真正从50以下突破到50以上
        """
        max_attempts = 10

        for attempt in range(max_attempts):
            dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')

            # 策略：先下跌让RSI降低，然后上涨让RSI突破50
            prices = []
            current_price = base_price

            # 第一阶段：下跌20天，让RSI降到30-40区间
            for i in range(20):
                daily_change = np.random.uniform(-0.02, -0.005)
                current_price *= (1 + daily_change)
                prices.append(current_price)

            # 第二阶段：横盘15天，让RSI稳定在40-50区间
            for i in range(15):
                daily_change = np.random.uniform(-0.005, 0.005)
                current_price *= (1 + daily_change)
                prices.append(current_price)

            # 第三阶段：上涨15天，让RSI突破50
            for i in range(15):
                daily_change = np.random.uniform(0.005, 0.02)
                current_price *= (1 + daily_change)
                prices.append(current_price)

            # 生成OHLC数据
            data = self._generate_ohlc_from_close(dates, prices)

            # 验证RSI金叉
            rsi = self.indicators.calculate_rsi(data)

            # 检查是否有金叉
            golden_cross_found = False
            if len(rsi) >= 10:
                recent_rsi = rsi.iloc[-10:]
                for i in range(1, len(recent_rsi)):
                    if recent_rsi.iloc[i-1] <= 50 and recent_rsi.iloc[i] > 50:
                        golden_cross_found = True
                        break

            if golden_cross_found:
                return self._standardize_data_format(data, 'RSI_GOLDEN_CROSS')

        # 如果多次尝试都失败，使用强制调整
        return self._force_rsi_golden_cross(base_price, periods)

    def _force_rsi_golden_cross(self, base_price: float, periods: int) -> pd.DataFrame:
        """强制生成RSI金叉数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')

        # 使用更精确的价格控制
        prices = [base_price]

        # 前30天：缓慢下跌，确保RSI降到50以下
        for i in range(1, 31):
            # 计算目标价格，确保下跌趋势
            target_decline = 0.15  # 总共下跌15%
            daily_decline = target_decline / 30
            new_price = prices[-1] * (1 - daily_decline + np.random.uniform(-0.005, 0.005))
            prices.append(new_price)

        # 中间10天：小幅波动
        for i in range(10):
            change = np.random.uniform(-0.003, 0.003)
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)

        # 最后9天：明确上涨，确保RSI突破50
        for i in range(9):
            # 逐步加大涨幅
            daily_gain = 0.008 + i * 0.002  # 从0.8%逐步增加到2.6%
            new_price = prices[-1] * (1 + daily_gain)
            prices.append(new_price)

        # 确保价格数组长度与日期数组匹配
        while len(prices) < periods:
            prices.append(prices[-1] * (1 + np.random.uniform(0.005, 0.015)))

        prices = prices[:periods]  # 确保不超过期望长度

        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'RSI_GOLDEN_CROSS')

    def generate_rsi_death_cross_data_v2(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """智能生成RSI死叉形态数据"""
        return self._force_rsi_death_cross(base_price, periods)

    def _force_rsi_death_cross(self, base_price: float, periods: int) -> pd.DataFrame:
        """强制生成RSI死叉数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')

        prices = [base_price]

        # 前30天：缓慢上涨，确保RSI升到50以上
        for i in range(1, 31):
            target_gain = 0.15  # 总共上涨15%
            daily_gain = target_gain / 30
            new_price = prices[-1] * (1 + daily_gain + np.random.uniform(-0.005, 0.005))
            prices.append(new_price)

        # 中间10天：小幅波动
        for i in range(10):
            change = np.random.uniform(-0.003, 0.003)
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)

        # 最后9天：明确下跌，确保RSI跌破50
        for i in range(9):
            daily_loss = -0.008 - i * 0.002  # 从-0.8%逐步增加到-2.6%
            new_price = prices[-1] * (1 + daily_loss)
            prices.append(new_price)

        # 确保价格数组长度与日期数组匹配
        while len(prices) < periods:
            prices.append(prices[-1] * (1 + np.random.uniform(-0.015, -0.005)))

        prices = prices[:periods]  # 确保不超过期望长度

        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'RSI_DEATH_CROSS')

    def generate_rsi_divergence_data_v2(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """智能生成RSI背离形态数据 - 精确控制版本"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')

        prices = [base_price]

        # 第一波：12天温和上涨，RSI达到70-75区间
        for i in range(1, 13):
            daily_gain = 0.015 + np.random.uniform(-0.003, 0.003)  # 1.5%日涨幅
            new_price = prices[-1] * (1 + daily_gain)
            prices.append(new_price)

        first_peak_price = prices[-1]

        # 深度回调：18天，让价格和RSI都大幅回落
        for i in range(18):
            daily_change = np.random.uniform(-0.018, -0.008)  # 深度回调
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 第二波：20天非常温和的上涨，价格创新高但RSI不创新高
        # 关键：使用极小的涨幅，确保RSI不会过高
        current_price = prices[-1]
        target_price = first_peak_price * 1.02  # 只比第一波高2%

        # 计算每日所需涨幅
        total_gain_needed = (target_price / current_price) - 1
        daily_gain_base = total_gain_needed / 20

        for i in range(20):
            # 使用非常小的涨幅，加上随机波动
            daily_gain = daily_gain_base + np.random.uniform(-0.002, 0.002)
            daily_gain = max(0.001, min(daily_gain, 0.008))  # 限制在0.1%-0.8%之间
            new_price = prices[-1] * (1 + daily_gain)
            prices.append(new_price)

        # 确保价格数组长度与日期数组匹配
        while len(prices) < periods:
            # 最后几天继续小幅上涨
            daily_gain = np.random.uniform(0.001, 0.005)
            new_price = prices[-1] * (1 + daily_gain)
            prices.append(new_price)

        prices = prices[:periods]  # 确保不超过期望长度

        # 最终验证：确保价格创新高
        final_price = prices[-1]
        if final_price <= first_peak_price:
            # 微调最后几个价格点
            adjustment = (first_peak_price * 1.01 - final_price) / 5
            for i in range(-5, 0):
                prices[i] += adjustment * (5 + i)

        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'RSI_DIVERGENCE')

    def _generate_ohlc_from_close(self, dates, close_prices):
        """从收盘价生成OHLC数据"""
        data = pd.DataFrame({
            'date': dates,
            'close': close_prices
        })

        # 生成开盘价
        open_prices = [close_prices[0]]
        for i in range(1, len(close_prices)):
            gap = np.random.uniform(-0.01, 0.01)
            open_price = close_prices[i-1] * (1 + gap)
            open_prices.append(open_price)

        data['open'] = open_prices

        # 生成高价和低价
        high_prices = []
        low_prices = []

        for i in range(len(close_prices)):
            open_price = open_prices[i]
            close_price = close_prices[i]

            daily_range = abs(close_price - open_price) * np.random.uniform(1.2, 2.0)

            high_price = max(open_price, close_price) + daily_range * np.random.uniform(0.1, 0.5)
            high_prices.append(high_price)

            low_price = min(open_price, close_price) - daily_range * np.random.uniform(0.1, 0.5)
            low_prices.append(low_price)

        data['high'] = high_prices
        data['low'] = low_prices

        # 生成成交量
        base_volume = 10000
        volumes = []
        for i in range(len(close_prices)):
            price_change = abs(close_prices[i] - open_prices[i]) / open_prices[i]
            volume_multiplier = 1 + price_change * 5
            volume = int(base_volume * volume_multiplier * np.random.uniform(0.5, 1.5))
            volumes.append(volume)

        data['volume'] = volumes

        return data

    def _standardize_data_format(self, data, pattern_name):
        """标准化数据格式"""
        price_changes = data['close'].diff().fillna(0).round(2)
        price_ranges = ((data['high'] - data['low']) / data['close'] * 100).round(2)
        turnover_rates = np.random.uniform(0.5, 5.0, len(data)).round(2)

        data['code'] = f'TEST_{pattern_name}'
        data['name'] = f'测试股票_{pattern_name}'
        data['level'] = 'D'
        data['industry'] = '软件服务'
        data['seq'] = range(len(data))
        data['turnover_rate'] = turnover_rates
        data['price_change'] = price_changes
        data['price_range'] = price_ranges
        data['datetime'] = pd.to_datetime(data['date'])

        column_order = [
            'code', 'name', 'date', 'level', 'open', 'high', 'low', 'close',
            'volume', 'turnover_rate', 'price_change', 'price_range', 'industry', 'datetime', 'seq'
        ]

        return data[column_order]

    # P1重要指标形态生成方法

    def generate_sar_uptrend_data(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """生成SAR上升趋势形态数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')

        prices = [base_price]

        # 持续上涨，形成SAR上升趋势
        for i in range(1, periods):
            daily_change = np.random.uniform(0.005, 0.015)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'SAR_UPTREND')

    def generate_sar_downtrend_data(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """生成SAR下降趋势形态数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')

        prices = [base_price]

        # 持续下跌，形成SAR下降趋势
        for i in range(1, periods):
            daily_change = np.random.uniform(-0.015, -0.005)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'SAR_DOWNTREND')

    def generate_sar_reversal_data(self, base_price: float = 100, periods: int = 60) -> pd.DataFrame:
        """生成SAR转向信号形态数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')

        prices = [base_price]

        # 前30天上涨
        for i in range(1, 31):
            daily_change = np.random.uniform(0.008, 0.015)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 后30天下跌，形成转向
        for i in range(30):
            daily_change = np.random.uniform(-0.015, -0.008)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'SAR_REVERSAL')

    def generate_adx_strong_trend_data(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """生成ADX强趋势形态数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')

        prices = [base_price]

        # 强势上涨，形成强趋势
        for i in range(1, periods):
            daily_change = np.random.uniform(0.015, 0.025)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'ADX_STRONG_TREND')

    def generate_adx_weak_trend_data(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """生成ADX弱趋势形态数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')

        prices = [base_price]

        # 横盘整理，形成弱趋势
        for i in range(1, periods):
            daily_change = np.random.uniform(-0.005, 0.005)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'ADX_WEAK_TREND')

    def generate_dmi_bullish_data(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """生成DMI多头形态数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')

        prices = [base_price]

        # 持续上涨，+DI > -DI
        for i in range(1, periods):
            daily_change = np.random.uniform(0.008, 0.018)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'DMI_BULLISH')

    def generate_dmi_bearish_data(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """生成DMI空头形态数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')

        prices = [base_price]

        # 持续下跌，-DI > +DI
        for i in range(1, periods):
            daily_change = np.random.uniform(-0.018, -0.008)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'DMI_BEARISH')

    def generate_trix_golden_cross_data(self, base_price: float = 100, periods: int = 60) -> pd.DataFrame:
        """生成TRIX金叉形态数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')

        prices = [base_price]

        # 前30天下跌
        for i in range(1, 31):
            daily_change = np.random.uniform(-0.012, -0.005)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 后30天上涨，形成金叉
        for i in range(30):
            daily_change = np.random.uniform(0.008, 0.015)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'TRIX_GOLDEN_CROSS')

    def generate_roc_overbought_data(self, base_price: float = 100, periods: int = 40) -> pd.DataFrame:
        """生成ROC超买形态数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')

        prices = [base_price]

        # 快速上涨，形成ROC超买
        for i in range(1, periods):
            daily_change = np.random.uniform(0.02, 0.03)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'ROC_OVERBOUGHT')

    def generate_cmo_oversold_data(self, base_price: float = 100, periods: int = 40) -> pd.DataFrame:
        """生成CMO超卖形态数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')

        prices = [base_price]

        # 快速下跌，形成CMO超卖
        for i in range(1, periods):
            daily_change = np.random.uniform(-0.03, -0.02)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'CMO_OVERSOLD')

    def generate_sar_support_data(self, base_price: float = 100, periods: int = 60) -> pd.DataFrame:
        """生成SAR支撑形态数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')

        prices = [base_price]

        # 前30天上涨建立SAR支撑
        for i in range(1, 31):
            daily_change = np.random.uniform(0.008, 0.015)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 中间15天回调测试SAR支撑
        for i in range(15):
            daily_change = np.random.uniform(-0.008, 0.003)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 后15天在SAR支撑位反弹
        for i in range(15):
            daily_change = np.random.uniform(0.005, 0.012)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'SAR_SUPPORT')

    def generate_sar_resistance_data(self, base_price: float = 100, periods: int = 60) -> pd.DataFrame:
        """生成SAR阻力形态数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')

        prices = [base_price]

        # 前30天下跌建立SAR阻力
        for i in range(1, 31):
            daily_change = np.random.uniform(-0.015, -0.008)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 中间15天反弹测试SAR阻力
        for i in range(15):
            daily_change = np.random.uniform(-0.003, 0.008)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 后15天在SAR阻力位回落
        for i in range(15):
            daily_change = np.random.uniform(-0.012, -0.005)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'SAR_RESISTANCE')

    def generate_adx_rising_data(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """生成ADX上升形态数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')

        prices = [base_price]

        # 前20天横盘，ADX较低
        for i in range(1, 21):
            daily_change = np.random.uniform(-0.005, 0.005)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 后30天逐渐形成趋势，ADX上升
        for i in range(30):
            # 逐渐增大波动，形成趋势
            volatility = 0.005 + 0.015 * (i / 30)
            daily_change = np.random.uniform(0, volatility)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'ADX_RISING')

    def generate_adx_falling_data(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """生成ADX下降形态数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')

        prices = [base_price]

        # 前20天强趋势，ADX较高
        for i in range(1, 21):
            daily_change = np.random.uniform(0.01, 0.02)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 后30天趋势减弱，ADX下降
        for i in range(30):
            # 逐渐减小波动，趋势减弱
            volatility = 0.015 * (1 - i / 30)
            daily_change = np.random.uniform(-volatility, volatility)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'ADX_FALLING')

    def generate_adx_divergence_data(self, base_price: float = 100, periods: int = 60) -> pd.DataFrame:
        """生成ADX背离形态数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')

        prices = [base_price]

        # 第一波上涨：20天强势上涨
        for i in range(1, 21):
            daily_change = np.random.uniform(0.02, 0.03)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 回调：20天
        for i in range(20):
            daily_change = np.random.uniform(-0.015, -0.005)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 第二波上涨：20天温和上涨，价格创新高但ADX不创新高
        for i in range(20):
            daily_change = np.random.uniform(0.005, 0.012)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 确保价格创新高
        first_peak = max(prices[:21])
        if prices[-1] <= first_peak:
            adjustment = (first_peak * 1.02 - prices[-1]) / 5
            for i in range(-5, 0):
                prices[i] += adjustment * (5 + i)

        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'ADX_DIVERGENCE')

    # EMA形态生成方法
    def generate_ema_golden_cross_data(self, base_price: float = 100, periods: int = 60) -> pd.DataFrame:
        """生成EMA金叉形态数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')

        prices = [base_price]

        # 第一阶段：30天下跌，让短期EMA低于长期EMA
        for i in range(1, 31):
            daily_change = np.random.uniform(-0.015, -0.005)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 第二阶段：15天横盘，让EMA接近
        for i in range(15):
            daily_change = np.random.uniform(-0.005, 0.005)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 第三阶段：15天上涨，让短期EMA上穿长期EMA
        for i in range(15):
            daily_change = np.random.uniform(0.01, 0.02)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 确保长度匹配
        while len(prices) < periods:
            prices.append(prices[-1] * (1 + np.random.uniform(0.008, 0.015)))
        prices = prices[:periods]

        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'EMA_GOLDEN_CROSS')

    def generate_ema_death_cross_data(self, base_price: float = 100, periods: int = 60) -> pd.DataFrame:
        """生成EMA死叉形态数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')

        prices = [base_price]

        # 第一阶段：30天上涨，让短期EMA高于长期EMA
        for i in range(1, 31):
            daily_change = np.random.uniform(0.005, 0.015)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 第二阶段：15天横盘，让EMA接近
        for i in range(15):
            daily_change = np.random.uniform(-0.005, 0.005)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 第三阶段：15天下跌，让短期EMA下穿长期EMA
        for i in range(15):
            daily_change = np.random.uniform(-0.02, -0.01)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 确保长度匹配
        while len(prices) < periods:
            prices.append(prices[-1] * (1 + np.random.uniform(-0.015, -0.008)))
        prices = prices[:periods]

        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'EMA_DEATH_CROSS')

    def generate_ema_trend_confirmation_data(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """生成EMA趋势确认形态数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')

        prices = [base_price]

        # 持续上涨，形成明确的上升趋势（短期EMA持续在长期EMA上方）
        for i in range(1, periods):
            daily_change = np.random.uniform(0.008, 0.018)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'EMA_TREND_CONFIRMATION')

    def generate_ema_divergence_data(self, base_price: float = 100, periods: int = 60) -> pd.DataFrame:
        """生成EMA背离形态数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')

        prices = [base_price]

        # 第一波上涨：20天强势上涨
        for i in range(1, 21):
            daily_change = np.random.uniform(0.02, 0.03)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 回调：20天
        for i in range(20):
            daily_change = np.random.uniform(-0.015, -0.005)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 第二波上涨：20天温和上涨，价格创新高但EMA增长有限
        for i in range(20):
            daily_change = np.random.uniform(0.005, 0.012)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 确保价格创新高
        first_peak = max(prices[:21])
        if prices[-1] <= first_peak:
            adjustment = (first_peak * 1.02 - prices[-1]) / 5
            for i in range(-5, 0):
                prices[i] += adjustment * (5 + i)

        # 确保长度匹配
        while len(prices) < periods:
            prices.append(prices[-1] * (1 + np.random.uniform(0.003, 0.008)))
        prices = prices[:periods]

        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'EMA_DIVERGENCE')

    def generate_ema_support_resistance_data(self, base_price: float = 100, periods: int = 60) -> pd.DataFrame:
        """生成EMA支撑阻力形态数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')

        prices = [base_price]

        # 第一阶段：25天上涨，让价格在EMA上方
        for i in range(1, 26):
            daily_change = np.random.uniform(0.008, 0.015)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 第二阶段：20天回调到EMA附近
        for i in range(20):
            daily_change = np.random.uniform(-0.012, -0.003)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 第三阶段：15天在EMA获得支撑反弹
        for i in range(15):
            daily_change = np.random.uniform(0.008, 0.018)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 确保长度匹配
        while len(prices) < periods:
            prices.append(prices[-1] * (1 + np.random.uniform(0.005, 0.015)))
        prices = prices[:periods]

        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'EMA_SUPPORT_RESISTANCE')

    # MA形态生成方法
    def generate_ma_golden_cross_data(self, base_price: float = 100, periods: int = 60) -> pd.DataFrame:
        """生成MA金叉形态数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')

        prices = [base_price]

        # 第一阶段：30天下跌，让短期MA低于长期MA
        for i in range(1, 31):
            daily_change = np.random.uniform(-0.015, -0.005)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 第二阶段：15天横盘，让MA接近
        for i in range(15):
            daily_change = np.random.uniform(-0.005, 0.005)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 第三阶段：15天上涨，让短期MA上穿长期MA
        for i in range(15):
            daily_change = np.random.uniform(0.01, 0.02)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 确保长度匹配
        while len(prices) < periods:
            prices.append(prices[-1] * (1 + np.random.uniform(0.008, 0.015)))
        prices = prices[:periods]

        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'MA_GOLDEN_CROSS')

    def generate_ma_death_cross_data(self, base_price: float = 100, periods: int = 60) -> pd.DataFrame:
        """生成MA死叉形态数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')

        prices = [base_price]

        # 第一阶段：30天上涨，让短期MA高于长期MA
        for i in range(1, 31):
            daily_change = np.random.uniform(0.005, 0.015)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 第二阶段：15天横盘，让MA接近
        for i in range(15):
            daily_change = np.random.uniform(-0.005, 0.005)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 第三阶段：15天下跌，让短期MA下穿长期MA
        for i in range(15):
            daily_change = np.random.uniform(-0.02, -0.01)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 确保长度匹配
        while len(prices) < periods:
            prices.append(prices[-1] * (1 + np.random.uniform(-0.015, -0.008)))
        prices = prices[:periods]

        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'MA_DEATH_CROSS')

    def generate_ma_bullish_alignment_data(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """生成MA多头排列形态数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')

        prices = [base_price]

        # 持续上涨，形成多头排列（短期MA > 长期MA）
        for i in range(1, periods):
            daily_change = np.random.uniform(0.008, 0.018)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'MA_BULLISH_ALIGNMENT')

    def generate_ma_bearish_alignment_data(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """生成MA空头排列形态数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')

        prices = [base_price]

        # 持续下跌，形成空头排列（短期MA < 长期MA）
        for i in range(1, periods):
            daily_change = np.random.uniform(-0.018, -0.008)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'MA_BEARISH_ALIGNMENT')

    def generate_ma_support_data(self, base_price: float = 100, periods: int = 60) -> pd.DataFrame:
        """生成MA支撑形态数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')

        prices = [base_price]

        # 第一阶段：25天上涨，让价格在MA上方
        for i in range(1, 26):
            daily_change = np.random.uniform(0.008, 0.015)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 第二阶段：20天回调到MA附近
        for i in range(20):
            daily_change = np.random.uniform(-0.012, -0.003)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 第三阶段：15天在MA获得支撑反弹
        for i in range(15):
            daily_change = np.random.uniform(0.008, 0.018)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 确保长度匹配
        while len(prices) < periods:
            prices.append(prices[-1] * (1 + np.random.uniform(0.005, 0.015)))
        prices = prices[:periods]

        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'MA_SUPPORT')

    # BOLL形态生成方法
    def generate_boll_upper_breakout_data(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """生成BOLL上轨突破形态数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')

        prices = [base_price]

        # 前35天：横盘整理，让布林带收窄，价格在中轨附近
        for i in range(1, 36):
            daily_change = np.random.uniform(-0.005, 0.005)  # 极小波动
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 后15天：突然强势上涨，确保突破上轨
        for i in range(15):
            # 逐步加大涨幅，确保突破
            daily_change = 0.02 + i * 0.005  # 从2%逐步增加到9.5%
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 确保长度匹配
        while len(prices) < periods:
            prices.append(prices[-1] * (1 + np.random.uniform(0.015, 0.025)))
        prices = prices[:periods]

        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'BOLL_UPPER_BREAKOUT')

    def generate_boll_lower_breakout_data(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """生成BOLL下轨突破形态数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')

        prices = [base_price]

        # 前35天：横盘整理，让布林带收窄，价格在中轨附近
        for i in range(1, 36):
            daily_change = np.random.uniform(-0.005, 0.005)  # 极小波动
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 后15天：突然强势下跌，确保突破下轨
        for i in range(15):
            # 逐步加大跌幅，确保突破
            daily_change = -0.02 - i * 0.005  # 从-2%逐步增加到-9.5%
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 确保长度匹配
        while len(prices) < periods:
            prices.append(prices[-1] * (1 + np.random.uniform(-0.025, -0.015)))
        prices = prices[:periods]

        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'BOLL_LOWER_BREAKOUT')

    def generate_boll_squeeze_data(self, base_price: float = 100, periods: int = 60) -> pd.DataFrame:
        """生成BOLL收口形态数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')

        prices = [base_price]

        # 前25天：较大波动，让布林带张开
        for i in range(1, 26):
            daily_change = np.random.uniform(-0.03, 0.03)  # 大幅波动
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 后35天：逐渐减小波动，让布林带收口
        for i in range(35):
            # 波动幅度逐渐减小
            volatility = 0.025 * (1 - i / 35)  # 从2.5%逐渐减小到0
            daily_change = np.random.uniform(-volatility, volatility)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 确保长度匹配
        while len(prices) < periods:
            daily_change = np.random.uniform(-0.001, 0.001)  # 极小波动
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        prices = prices[:periods]

        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'BOLL_SQUEEZE')

    def generate_boll_expansion_data(self, base_price: float = 100, periods: int = 60) -> pd.DataFrame:
        """生成BOLL开口形态数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')

        prices = [base_price]

        # 前25天：极小波动，让布林带收窄
        for i in range(1, 26):
            daily_change = np.random.uniform(-0.002, 0.002)  # 极小波动
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 后35天：逐渐增大波动，让布林带开口
        for i in range(35):
            # 波动幅度逐渐增大
            volatility = 0.002 + 0.03 * (i / 35)  # 从0.2%逐渐增大到3.2%
            daily_change = np.random.uniform(-volatility, volatility)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 确保长度匹配
        while len(prices) < periods:
            daily_change = np.random.uniform(-0.03, 0.03)  # 大幅波动
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        prices = prices[:periods]

        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'BOLL_EXPANSION')

    def generate_boll_middle_support_data(self, base_price: float = 100, periods: int = 60) -> pd.DataFrame:
        """生成BOLL中轨支撑形态数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')

        prices = [base_price]

        # 前25天：温和上涨，让价格在中轨上方
        for i in range(1, 26):
            daily_change = np.random.uniform(0.008, 0.015)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 中间20天：回调到中轨附近，但不跌破
        for i in range(20):
            # 逐渐减小跌幅，模拟接近中轨时的支撑
            daily_change = np.random.uniform(-0.015 + i * 0.0005, -0.002)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 后15天：在中轨获得支撑反弹
        for i in range(15):
            daily_change = np.random.uniform(0.008, 0.018)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 确保长度匹配
        while len(prices) < periods:
            prices.append(prices[-1] * (1 + np.random.uniform(0.005, 0.015)))
        prices = prices[:periods]

        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'BOLL_MIDDLE_SUPPORT')

    # KDJ形态生成方法
    def generate_kdj_golden_cross_data(self, base_price: float = 100, periods: int = 60) -> pd.DataFrame:
        """生成KDJ金叉形态数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')

        prices = [base_price]

        # 第一阶段：30天下跌，让KDJ降到低位且K<D
        for i in range(1, 31):
            daily_change = np.random.uniform(-0.02, -0.008)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 第二阶段：15天横盘，让KDJ在低位震荡
        for i in range(15):
            daily_change = np.random.uniform(-0.005, 0.005)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 第三阶段：15天上涨，让K线上穿D线形成金叉
        for i in range(15):
            daily_change = np.random.uniform(0.01, 0.025)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 确保长度匹配
        while len(prices) < periods:
            prices.append(prices[-1] * (1 + np.random.uniform(0.008, 0.018)))
        prices = prices[:periods]

        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'KDJ_GOLDEN_CROSS')

    def generate_kdj_death_cross_data(self, base_price: float = 100, periods: int = 60) -> pd.DataFrame:
        """生成KDJ死叉形态数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')

        prices = [base_price]

        # 第一阶段：30天上涨，让KDJ升到高位且K>D
        for i in range(1, 31):
            daily_change = np.random.uniform(0.008, 0.02)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 第二阶段：15天横盘，让KDJ在高位震荡
        for i in range(15):
            daily_change = np.random.uniform(-0.005, 0.005)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 第三阶段：15天下跌，让K线下穿D线形成死叉
        for i in range(15):
            daily_change = np.random.uniform(-0.025, -0.01)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 确保长度匹配
        while len(prices) < periods:
            prices.append(prices[-1] * (1 + np.random.uniform(-0.018, -0.008)))
        prices = prices[:periods]

        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'KDJ_DEATH_CROSS')

    def generate_kdj_overbought_data(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """生成KDJ超买形态数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')

        prices = [base_price]

        # 持续上涨，让KDJ指标升到80以上
        for i in range(1, periods):
            daily_change = np.random.uniform(0.015, 0.03)  # 强势上涨
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'KDJ_OVERBOUGHT')

    def generate_kdj_oversold_data(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """生成KDJ超卖形态数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')

        prices = [base_price]

        # 持续下跌，让KDJ指标降到20以下
        for i in range(1, periods):
            daily_change = np.random.uniform(-0.03, -0.015)  # 强势下跌
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'KDJ_OVERSOLD')

    def generate_kdj_divergence_data(self, base_price: float = 100, periods: int = 60) -> pd.DataFrame:
        """生成KDJ背离形态数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')

        prices = [base_price]

        # 第一波上涨：20天强势上涨
        for i in range(1, 21):
            daily_change = np.random.uniform(0.02, 0.03)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 回调：20天
        for i in range(20):
            daily_change = np.random.uniform(-0.015, -0.005)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 第二波上涨：20天温和上涨，价格创新高但KDJ不创新高
        for i in range(20):
            daily_change = np.random.uniform(0.005, 0.012)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 确保价格创新高
        first_peak = max(prices[:21])
        if prices[-1] <= first_peak:
            adjustment = (first_peak * 1.02 - prices[-1]) / 5
            for i in range(-5, 0):
                prices[i] += adjustment * (5 + i)

        # 确保长度匹配
        while len(prices) < periods:
            prices.append(prices[-1] * (1 + np.random.uniform(0.003, 0.008)))
        prices = prices[:periods]

        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'KDJ_DIVERGENCE')

    # MACD形态生成方法
    def generate_macd_golden_cross_data(self, base_price: float = 100, periods: int = 70) -> pd.DataFrame:
        """生成MACD金叉形态数据 - 确保真正的交叉"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')

        prices = [base_price]

        # 第一阶段：40天持续下跌，让DIF和DEA都为负值，且DEA > DIF
        for i in range(1, 41):
            # 持续下跌，确保MACD指标为负
            daily_change = np.random.uniform(-0.025, -0.01)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 第二阶段：15天缓慢反弹，让DIF开始上升但仍低于DEA
        for i in range(15):
            # 缓慢反弹，DIF上升但DEA由于平滑效应仍然较高
            daily_change = np.random.uniform(0.005, 0.012)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 第三阶段：15天加速上涨，确保DIF上穿DEA
        for i in range(15):
            # 加速上涨，确保DIF快速上升穿越DEA
            daily_change = np.random.uniform(0.015, 0.025)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 确保长度匹配
        while len(prices) < periods:
            prices.append(prices[-1] * (1 + np.random.uniform(0.01, 0.02)))
        prices = prices[:periods]

        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'MACD_GOLDEN_CROSS')

    def generate_macd_death_cross_data(self, base_price: float = 100, periods: int = 70) -> pd.DataFrame:
        """生成MACD死叉形态数据 - 确保真正的交叉"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')

        prices = [base_price]

        # 第一阶段：40天持续上涨，让DIF和DEA都为正值，且DIF > DEA
        for i in range(1, 41):
            # 持续上涨，确保MACD指标为正
            daily_change = np.random.uniform(0.01, 0.025)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 第二阶段：15天缓慢下跌，让DIF开始下降但仍高于DEA
        for i in range(15):
            # 缓慢下跌，DIF下降但DEA由于平滑效应仍然较低
            daily_change = np.random.uniform(-0.012, -0.005)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 第三阶段：15天加速下跌，确保DIF下穿DEA
        for i in range(15):
            # 加速下跌，确保DIF快速下降穿越DEA
            daily_change = np.random.uniform(-0.025, -0.015)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 确保长度匹配
        while len(prices) < periods:
            prices.append(prices[-1] * (1 + np.random.uniform(-0.02, -0.01)))
        prices = prices[:periods]

        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'MACD_DEATH_CROSS')

    def generate_macd_above_zero_golden_data(self, base_price: float = 100, periods: int = 60) -> pd.DataFrame:
        """生成MACD零轴上金叉形态数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')

        prices = [base_price]

        # 第一阶段：25天上涨，让MACD在零轴上方且DIF>DEA
        for i in range(1, 26):
            daily_change = np.random.uniform(0.01, 0.018)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 第二阶段：20天小幅回调，让DIF下降到DEA以下但保持在零轴上方
        for i in range(20):
            daily_change = np.random.uniform(-0.008, 0.003)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 第三阶段：15天再次上涨，形成零轴上金叉
        for i in range(15):
            daily_change = np.random.uniform(0.012, 0.022)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 确保长度匹配
        while len(prices) < periods:
            prices.append(prices[-1] * (1 + np.random.uniform(0.01, 0.018)))
        prices = prices[:periods]

        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'MACD_ABOVE_ZERO_GOLDEN')

    def generate_macd_below_zero_death_data(self, base_price: float = 100, periods: int = 60) -> pd.DataFrame:
        """生成MACD零轴下死叉形态数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')

        prices = [base_price]

        # 第一阶段：25天下跌，让MACD在零轴下方且DIF<DEA
        for i in range(1, 26):
            daily_change = np.random.uniform(-0.018, -0.01)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 第二阶段：20天小幅反弹，让DIF上升到DEA以上但保持在零轴下方
        for i in range(20):
            daily_change = np.random.uniform(-0.003, 0.008)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 第三阶段：15天再次下跌，形成零轴下死叉
        for i in range(15):
            daily_change = np.random.uniform(-0.022, -0.012)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 确保长度匹配
        while len(prices) < periods:
            prices.append(prices[-1] * (1 + np.random.uniform(-0.018, -0.01)))
        prices = prices[:periods]

        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'MACD_BELOW_ZERO_DEATH')

    def generate_macd_histogram_divergence_data(self, base_price: float = 100, periods: int = 60) -> pd.DataFrame:
        """生成MACD柱状图背离形态数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')

        prices = [base_price]

        # 第一波上涨：20天强势上涨，MACD柱状图达到高位
        for i in range(1, 21):
            daily_change = np.random.uniform(0.02, 0.03)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 回调：20天明显回调
        for i in range(20):
            daily_change = np.random.uniform(-0.018, -0.005)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 第二波上涨：20天温和上涨，价格创新高但MACD柱状图不创新高
        for i in range(20):
            daily_change = np.random.uniform(0.005, 0.012)  # 较小的涨幅
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)

        # 确保价格创新高
        first_peak = max(prices[:21])
        if prices[-1] <= first_peak:
            adjustment = (first_peak * 1.02 - prices[-1]) / 5
            for i in range(-5, 0):
                prices[i] += adjustment * (5 + i)

        # 确保长度匹配
        while len(prices) < periods:
            prices.append(prices[-1] * (1 + np.random.uniform(0.003, 0.008)))
        prices = prices[:periods]

        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'MACD_HISTOGRAM_DIVERGENCE')