#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
智能P2常用指标数据生成器

基于P1阶段的成功经验，为P2常用指标智能生成确保形态成功的价格序列
支持15个P2指标：STOCHRSI、PSY、WR、BIAS、VOL、OBV、MFI、EMV、CCI、MOMENTUM、VOSC、VR、PVT、CHAIKIN、AD
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class Intelligent_p2_generator:
    """智能P2指标数据生成器"""
    
    def __init__(self):
        pass
    
    def _generate_ohlc_from_close_Intelligent_P2_Generator(self, dates: pd.Datetime_index, close_prices: List[float]) -> pd.DataFrame:
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
            
            # 生成成交量（重要：P2指标多数需要成交量）
            base_volume = 5000000
            volume_factor = 1 + abs(close - close_prices[0]) / close_prices[0]  # 价格变化越大，成交量越大
            volume = int(base_volume * volume_factor * np.random.uniform(0.5, 2.0))
            
            data.append({
                'date': date,
                'open': open_price,
                'high': max(open_price, high, close),
                'low': min(open_price, low, close),
                'close': close,
                'volume': volume
            })
        
        return pd.DataFrame(data)
    
    def _standardize_data_format_Intelligent_P2_Generator(self, data: pd.DataFrame, pattern_name: str) -> pd.DataFrame:
        """标准化数据格式"""
        # 确保列顺序正确
        column_order = ['date', 'open', 'high', 'low', 'close', 'volume']
        return data[column_order]
    
    def generate_stochrsi_overbought_data(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """生成StochRSI超买数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        prices = [base_price]
        
        # 前30天快速上涨，形成RSI超买
        for i in range(1, 31):
            daily_change = np.random.uniform(0.02, 0.04)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        # 后20天继续上涨但速度放缓
        for i in range(20):
            daily_change = np.random.uniform(0.005, 0.015)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        data = self._generate_ohlc_from_close_Intelligent_P2_Generator(dates, prices)
        return self._standardize_data_format_Intelligent_P2_Generator(data, 'STOCHRSI_OVERBOUGHT')
    
    def generate_stochrsi_oversold_data(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """生成StochRSI超卖数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        prices = [base_price]
        
        # 前30天快速下跌，形成RSI超卖
        for i in range(1, 31):
            daily_change = np.random.uniform(-0.04, -0.02)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        # 后20天继续下跌但速度放缓
        for i in range(20):
            daily_change = np.random.uniform(-0.015, -0.005)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        data = self._generate_ohlc_from_close_Intelligent_P2_Generator(dates, prices)
        return self._standardize_data_format_Intelligent_P2_Generator(data, 'STOCHRSI_OVERSOLD')
    
    def generate_psy_bullish_data(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """生成PSY多头数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        prices = [base_price]
        
        # 大部分时间上涨，形成高PSY值
        for i in range(1, periods):
            if i % 5 == 0:  # 每5天一次小回调
                daily_change = np.random.uniform(-0.01, 0.005)
            else:
                daily_change = np.random.uniform(0.005, 0.02)
            
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        data = self._generate_ohlc_from_close_Intelligent_P2_Generator(dates, prices)
        return self._standardize_data_format_Intelligent_P2_Generator(data, 'PSY_BULLISH')
    
    def generate_psy_bearish_data(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """生成PSY空头数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        prices = [base_price]
        
        # 大部分时间下跌，形成低PSY值
        for i in range(1, periods):
            if i % 5 == 0:  # 每5天一次小反弹
                daily_change = np.random.uniform(-0.005, 0.01)
            else:
                daily_change = np.random.uniform(-0.02, -0.005)
            
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        data = self._generate_ohlc_from_close_Intelligent_P2_Generator(dates, prices)
        return self._standardize_data_format_Intelligent_P2_Generator(data, 'PSY_BEARISH')
    
    def generate_wr_overbought_data(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """生成WR超买数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        prices = [base_price]
        
        # 持续上涨，收盘价接近高点
        for i in range(1, periods):
            daily_change = np.random.uniform(0.01, 0.025)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        data = self._generate_ohlc_from_close_Intelligent_P2_Generator(dates, prices)
        return self._standardize_data_format_Intelligent_P2_Generator(data, 'WR_OVERBOUGHT')
    
    def generate_wr_oversold_data(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """生成WR超卖数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        prices = [base_price]
        
        # 持续下跌，收盘价接近低点
        for i in range(1, periods):
            daily_change = np.random.uniform(-0.025, -0.01)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        data = self._generate_ohlc_from_close_Intelligent_P2_Generator(dates, prices)
        return self._standardize_data_format_Intelligent_P2_Generator(data, 'WR_OVERSOLD')
    
    def generate_bias_positive_data(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """生成BIAS正乖离数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        prices = [base_price]
        
        # 前20天缓慢上涨建立均线
        for i in range(1, 21):
            daily_change = np.random.uniform(0.002, 0.008)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        # 后30天快速上涨，远离均线
        for i in range(30):
            daily_change = np.random.uniform(0.015, 0.03)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        data = self._generate_ohlc_from_close_Intelligent_P2_Generator(dates, prices)
        return self._standardize_data_format_Intelligent_P2_Generator(data, 'BIAS_POSITIVE')
    
    def generate_bias_negative_data(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """生成BIAS负乖离数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        prices = [base_price]
        
        # 前20天缓慢下跌建立均线
        for i in range(1, 21):
            daily_change = np.random.uniform(-0.008, -0.002)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        # 后30天快速下跌，远离均线
        for i in range(30):
            daily_change = np.random.uniform(-0.03, -0.015)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        data = self._generate_ohlc_from_close_Intelligent_P2_Generator(dates, prices)
        return self._standardize_data_format_Intelligent_P2_Generator(data, 'BIAS_NEGATIVE')
    
    def generate_vol_surge_data(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """生成成交量放大数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        prices = [base_price]
        
        # 前30天正常波动
        for i in range(1, 31):
            daily_change = np.random.uniform(-0.01, 0.01)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        # 后20天价格突破，成交量放大
        for i in range(20):
            daily_change = np.random.uniform(0.02, 0.04)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        data = self._generate_ohlc_from_close_Intelligent_P2_Generator(dates, prices)
        
        # 特别处理成交量：后20天成交量明显放大
        for i in range(len(data)):
            if i >= 30:  # 后20天
                data.loc[i, 'volume'] *= np.random.uniform(2.0, 4.0)
        
        return self._standardize_data_format_Intelligent_P2_Generator(data, 'VOL_SURGE')
    
    def generate_obv_uptrend_data(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """生成OBV上升趋势数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        prices = [base_price]
        
        # 持续上涨，确保OBV上升
        for i in range(1, periods):
            # 大部分时间上涨
            if i % 8 == 0:  # 偶尔小回调
                daily_change = np.random.uniform(-0.005, 0.002)
            else:
                daily_change = np.random.uniform(0.008, 0.02)
            
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        data = self._generate_ohlc_from_close_Intelligent_P2_Generator(dates, prices)
        return self._standardize_data_format_Intelligent_P2_Generator(data, 'OBV_UPTREND')
    
    def generate_mfi_overbought_data(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """生成MFI超买数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        prices = [base_price]
        
        # 快速上涨配合大成交量
        for i in range(1, periods):
            daily_change = np.random.uniform(0.015, 0.03)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        data = self._generate_ohlc_from_close_Intelligent_P2_Generator(dates, prices)
        
        # 上涨时成交量放大
        for i in range(len(data)):
            if i > 0 and data.loc[i, 'close'] > data.loc[i-1, 'close']:
                data.loc[i, 'volume'] *= np.random.uniform(1.5, 3.0)
        
        return self._standardize_data_format_Intelligent_P2_Generator(data, 'MFI_OVERBOUGHT')
    
    def generate_generic_pattern_data_Generator(self, pattern_name: str, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """生成通用形态数据（用于其他指标）"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        prices = [base_price]
        
        # 根据形态名称生成相应的价格序列
        if 'OVERBOUGHT' in pattern_name or 'HIGH' in pattern_name or 'BULLISH' in pattern_name:
            # 上涨形态
            for i in range(1, periods):
                daily_change = np.random.uniform(0.01, 0.025)
                new_price = prices[-1] * (1 + daily_change)
                prices.append(new_price)
        elif 'OVERSOLD' in pattern_name or 'LOW' in pattern_name or 'BEARISH' in pattern_name:
            # 下跌形态
            for i in range(1, periods):
                daily_change = np.random.uniform(-0.025, -0.01)
                new_price = prices[-1] * (1 + daily_change)
                prices.append(new_price)
        elif 'DIVERGENCE' in pattern_name:
            # 背离形态：价格上涨但指标不配合
            for i in range(1, periods//2):
                daily_change = np.random.uniform(0.02, 0.03)
                new_price = prices[-1] * (1 + daily_change)
                prices.append(new_price)
            for i in range(periods//2):
                daily_change = np.random.uniform(0.005, 0.015)
                new_price = prices[-1] * (1 + daily_change)
                prices.append(new_price)
        else:
            # 默认：温和上涨
            for i in range(1, periods):
                daily_change = np.random.uniform(0.005, 0.015)
                new_price = prices[-1] * (1 + daily_change)
                prices.append(new_price)
        
        data = self._generate_ohlc_from_close_Intelligent_P2_Generator(dates, prices)
        return self._standardize_data_format_Intelligent_P2_Generator(data, pattern_name)


def test_intelligent_p2_generator():
    """测试智能P2生成器"""
    generator = Intelligent_p2_generator()
    
    print("测试智能P2数据生成器...")
    
    # 测试几个关键指标的数据生成
    stochrsi_data = generator.generate_stochrsi_overbought_data()
    print(f"StochRSI超买数据: {len(stochrsi_data)}行")
    print(f"价格变化: {stochrsi_data['close'].iloc[0]:.2f} -> {stochrsi_data['close'].iloc[-1]:.2f}")
    
    vol_data = generator.generate_vol_surge_data()
    print(f"成交量放大数据: {len(vol_data)}行")
    print(f"成交量变化: {vol_data['volume'].iloc[0]:,} -> {vol_data['volume'].iloc[-1]:,}")
    
    print("智能P2生成器测试完成！")


if __name__ == '__main__':
    test_intelligent_p2_generator()
