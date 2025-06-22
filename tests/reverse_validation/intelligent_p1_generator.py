#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
智能P1指标数据生成器

基于技术指标的数学原理，智能生成确保形态成功的价格序列
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class IntelligentP1Generator:
    """智能P1指标数据生成器"""
    
    def __init__(self):
        pass
    
    def _generate_ohlc_from_close(self, dates: pd.DatetimeIndex, close_prices: List[float]) -> pd.DataFrame:
        """从收盘价生成OHLC数据"""
        data = []
        for i, (date, close) in enumerate(zip(dates, close_prices)):
            # 生成合理的OHLC数据
            volatility = 0.02  # 2%的日内波动
            high = close * (1 + np.random.uniform(0, volatility))
            low = close * (1 - np.random.uniform(0, volatility))
            
            if i == 0:
                open_price = close
            else:
                # 开盘价接近前一日收盘价
                open_price = close_prices[i-1] * (1 + np.random.uniform(-0.01, 0.01))
            
            data.append({
                'date': date,
                'open': open_price,
                'high': max(open_price, high, close),
                'low': min(open_price, low, close),
                'close': close,
                'volume': np.random.randint(1000000, 10000000)
            })
        
        return pd.DataFrame(data)
    
    def _standardize_data_format(self, data: pd.DataFrame, pattern_name: str) -> pd.DataFrame:
        """标准化数据格式"""
        # 确保列顺序正确
        column_order = ['date', 'open', 'high', 'low', 'close', 'volume']
        return data[column_order]
    
    def generate_sar_uptrend_intelligent(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """智能生成SAR上升趋势数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        # 确保生成明确的上升趋势
        prices = [base_price]
        for i in range(1, periods):
            # 持续上涨，偶尔小幅回调
            if i % 7 == 0:  # 每7天一次小回调
                daily_change = np.random.uniform(-0.005, 0.005)
            else:
                daily_change = np.random.uniform(0.008, 0.02)
            
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'SAR_UPTREND')
    
    def generate_sar_downtrend_intelligent(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """智能生成SAR下降趋势数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        # 确保生成明确的下降趋势
        prices = [base_price]
        for i in range(1, periods):
            # 持续下跌，偶尔小幅反弹
            if i % 7 == 0:  # 每7天一次小反弹
                daily_change = np.random.uniform(-0.005, 0.005)
            else:
                daily_change = np.random.uniform(-0.02, -0.008)
            
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'SAR_DOWNTREND')
    
    def generate_sar_reversal_intelligent(self, base_price: float = 100, periods: int = 60) -> pd.DataFrame:
        """智能生成SAR转向信号数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        prices = [base_price]
        mid_point = periods // 2
        
        # 前半段：明确上涨
        for i in range(1, mid_point):
            daily_change = np.random.uniform(0.01, 0.02)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        # 后半段：明确下跌
        for i in range(mid_point, periods):
            daily_change = np.random.uniform(-0.02, -0.01)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'SAR_REVERSAL')
    
    def generate_sar_support_intelligent(self, base_price: float = 100, periods: int = 60) -> pd.DataFrame:
        """智能生成SAR支撑数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        prices = [base_price]
        
        # 前20天：上涨建立支撑
        for i in range(1, 21):
            daily_change = np.random.uniform(0.01, 0.02)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        # 中间20天：回调但不跌破支撑
        peak_price = prices[-1]
        for i in range(20):
            # 回调但保持在一定水平之上
            daily_change = np.random.uniform(-0.01, 0.005)
            new_price = max(prices[-1] * (1 + daily_change), peak_price * 0.9)
            prices.append(new_price)
        
        # 后20天：重新上涨
        for i in range(20):
            daily_change = np.random.uniform(0.008, 0.018)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'SAR_SUPPORT')
    
    def generate_sar_resistance_intelligent(self, base_price: float = 100, periods: int = 60) -> pd.DataFrame:
        """智能生成SAR阻力数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        prices = [base_price]
        
        # 前20天：下跌建立阻力
        for i in range(1, 21):
            daily_change = np.random.uniform(-0.02, -0.01)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        # 中间20天：反弹但不突破阻力
        low_price = prices[-1]
        for i in range(20):
            # 反弹但保持在一定水平之下
            daily_change = np.random.uniform(-0.005, 0.01)
            new_price = min(prices[-1] * (1 + daily_change), low_price * 1.1)
            prices.append(new_price)
        
        # 后20天：重新下跌
        for i in range(20):
            daily_change = np.random.uniform(-0.018, -0.008)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'SAR_RESISTANCE')
    
    def generate_adx_strong_trend_intelligent(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """智能生成ADX强趋势数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        prices = [base_price]
        
        # 生成强烈的单向趋势
        for i in range(1, periods):
            # 强势上涨，很少回调
            daily_change = np.random.uniform(0.015, 0.03)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'ADX_STRONG_TREND')
    
    def generate_adx_weak_trend_intelligent(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """智能生成ADX弱趋势数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        prices = [base_price]
        
        # 生成横盘整理，无明确趋势
        for i in range(1, periods):
            # 小幅随机波动
            daily_change = np.random.uniform(-0.008, 0.008)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'ADX_WEAK_TREND')
    
    def generate_adx_rising_intelligent(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """智能生成ADX上升数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        prices = [base_price]
        
        # 前半段：横盘，ADX较低
        mid_point = periods // 2
        for i in range(1, mid_point):
            daily_change = np.random.uniform(-0.005, 0.005)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        # 后半段：逐渐形成趋势，ADX上升
        for i in range(mid_point, periods):
            # 逐渐增强的趋势
            trend_strength = (i - mid_point) / (periods - mid_point)
            daily_change = np.random.uniform(0.005 + trend_strength * 0.02, 0.01 + trend_strength * 0.025)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'ADX_RISING')
    
    def generate_adx_falling_intelligent(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """智能生成ADX下降数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        prices = [base_price]
        
        # 前半段：强趋势，ADX较高
        mid_point = periods // 2
        for i in range(1, mid_point):
            daily_change = np.random.uniform(0.015, 0.025)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        # 后半段：趋势减弱，ADX下降
        for i in range(mid_point, periods):
            # 逐渐减弱的趋势
            trend_strength = 1 - (i - mid_point) / (periods - mid_point)
            daily_change = np.random.uniform(-0.005 + trend_strength * 0.02, 0.005 + trend_strength * 0.015)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'ADX_FALLING')
    
    def generate_adx_divergence_intelligent(self, base_price: float = 100, periods: int = 60) -> pd.DataFrame:
        """智能生成ADX背离数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        prices = [base_price]
        
        # 第一波：20天强势上涨
        for i in range(1, 21):
            daily_change = np.random.uniform(0.025, 0.035)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        # 回调：20天
        for i in range(20):
            daily_change = np.random.uniform(-0.015, -0.005)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        # 第二波：20天温和上涨，价格创新高但趋势强度减弱
        first_peak = max(prices[:21])
        for i in range(20):
            daily_change = np.random.uniform(0.005, 0.012)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        # 确保价格创新高
        if prices[-1] <= first_peak:
            prices[-1] = first_peak * 1.05
        
        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'ADX_DIVERGENCE')


def test_intelligent_generator():
    """测试智能生成器"""
    generator = IntelligentP1Generator()
    
    print("测试智能P1数据生成器...")
    
    # 测试SAR数据生成
    sar_uptrend = generator.generate_sar_uptrend_intelligent()
    print(f"SAR上升趋势数据: {len(sar_uptrend)}行")
    print(f"价格变化: {sar_uptrend['close'].iloc[0]:.2f} -> {sar_uptrend['close'].iloc[-1]:.2f}")
    
    # 测试ADX数据生成
    adx_strong = generator.generate_adx_strong_trend_intelligent()
    print(f"ADX强趋势数据: {len(adx_strong)}行")
    print(f"价格变化: {adx_strong['close'].iloc[0]:.2f} -> {adx_strong['close'].iloc[-1]:.2f}")
    
    print("智能生成器测试完成！")


if __name__ == '__main__':
    test_intelligent_generator()
