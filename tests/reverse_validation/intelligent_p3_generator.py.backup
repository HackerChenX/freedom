#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
智能P3专业指标数据生成器

基于P1/P2阶段的成功经验，为P3专业指标智能生成确保形态成功的价格序列
支持10个P3指标：ATR、KC、VORTEX、AROON、ICHIMOKU、WMA、VIX、VOLUME_RATIO、ENHANCED_CCI、ENHANCED_DMI
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class Intelligent_p3_generator:
    """智能P3专业指标数据生成器"""
    
    def __init__(self):
        pass
    
    def _generate_ohlc_from_close_Intelligent_P3_Generator(self, dates: pd.Datetime_index, close_prices: List[float], 
                                 volatility_factor: float = 1.0) -> pd.DataFrame:
        """从收盘价生成OHLC数据，支持可变波动率"""
        data = []
        for i, (date, close) in enumerate(zip(dates, close_prices)):
            # 根据波动率因子调整日内波动
            base_volatility = 0.02
            volatility = base_volatility * volatility_factor
            
            high = close * (1 + np.random.uniform(0, volatility))
            low = close * (1 - np.random.uniform(0, volatility))
            
            if i == 0:
                open_price = close
            else:
                # 开盘价接近前一日收盘价
                gap_factor = np.random.uniform(-0.01, 0.01)
                open_price = close_prices[i-1] * (1 + gap_factor)
            
            # 生成成交量
            base_volume = 5000000
            volume_factor = 1 + abs(close - close_prices[0]) / close_prices[0]
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
    
    def _standardize_data_format_Intelligent_P3_Generator(self, data: pd.DataFrame, pattern_name: str) -> pd.DataFrame:
        """标准化数据格式"""
        column_order = ['date', 'open', 'high', 'low', 'close', 'volume']
        return data[column_order]
    
    def generate_atr_high_volatility_data(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """生成ATR高波动率数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        prices = [base_price]
        
        # 生成高波动率的价格序列
        for i in range(1, periods):
            # 随机大幅波动
            daily_change = np.random.uniform(-0.05, 0.05)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        # 使用高波动率因子
        data = self._generate_ohlc_from_close_Intelligent_P3_Generator(dates, prices, volatility_factor=2.0)
        return self._standardize_data_format_Intelligent_P3_Generator(data, 'ATR_HIGH_VOLATILITY')
    
    def generate_atr_low_volatility_data(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """生成ATR低波动率数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        prices = [base_price]
        
        # 生成低波动率的价格序列
        for i in range(1, periods):
            # 小幅波动
            daily_change = np.random.uniform(-0.01, 0.01)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        # 使用低波动率因子
        data = self._generate_ohlc_from_close_Intelligent_P3_Generator(dates, prices, volatility_factor=0.3)
        return self._standardize_data_format_Intelligent_P3_Generator(data, 'ATR_LOW_VOLATILITY')
    
    def generate_kc_breakout_data(self, base_price: float = 100, periods: int = 60) -> pd.DataFrame:
        """生成KC突破数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        prices = [base_price]
        
        # 前40天横盘整理
        for i in range(1, 41):
            daily_change = np.random.uniform(-0.005, 0.005)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        # 后20天突破上涨
        for i in range(20):
            daily_change = np.random.uniform(0.02, 0.04)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        data = self._generate_ohlc_from_close_Intelligent_P3_Generator(dates, prices)
        return self._standardize_data_format_Intelligent_P3_Generator(data, 'KC_BREAKOUT')
    
    def generate_vortex_bullish_data(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """生成Vortex多头数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        prices = [base_price]
        
        # 生成VI+>VI-的上涨趋势
        for i in range(1, periods):
            daily_change = np.random.uniform(0.01, 0.025)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        # 增加波动率以突出Vortex特征
        data = self._generate_ohlc_from_close_Intelligent_P3_Generator(dates, prices, volatility_factor=1.5)
        return self._standardize_data_format_Intelligent_P3_Generator(data, 'VORTEX_BULLISH')
    
    def generate_aroon_uptrend_data(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """生成Aroon上升趋势数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        prices = [base_price]
        
        # 生成持续创新高的上升趋势
        for i in range(1, periods):
            # 确保经常创新高
            if i % 5 == 0:  # 每5天创一次新高
                daily_change = np.random.uniform(0.02, 0.03)
            else:
                daily_change = np.random.uniform(0.005, 0.015)
            
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        data = self._generate_ohlc_from_close_Intelligent_P3_Generator(dates, prices)
        return self._standardize_data_format_Intelligent_P3_Generator(data, 'AROON_UPTREND')
    
    def generate_ichimoku_bullish_data(self, base_price: float = 100, periods: int = 80) -> pd.DataFrame:
        """生成Ichimoku多头数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        prices = [base_price]
        
        # 前30天建立基础
        for i in range(1, 31):
            daily_change = np.random.uniform(0.002, 0.008)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        # 中间30天强势上涨
        for i in range(30):
            daily_change = np.random.uniform(0.015, 0.025)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        # 后20天继续上涨
        for i in range(20):
            daily_change = np.random.uniform(0.008, 0.018)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        data = self._generate_ohlc_from_close_Intelligent_P3_Generator(dates, prices)
        return self._standardize_data_format_Intelligent_P3_Generator(data, 'ICHIMOKU_BULLISH')
    
    def generate_wma_golden_cross_data(self, base_price: float = 100, periods: int = 60) -> pd.DataFrame:
        """生成WMA金叉数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        prices = [base_price]
        
        # 前30天下跌
        for i in range(1, 31):
            daily_change = np.random.uniform(-0.015, -0.005)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        # 后30天上涨，形成金叉
        for i in range(30):
            daily_change = np.random.uniform(0.01, 0.02)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        data = self._generate_ohlc_from_close_Intelligent_P3_Generator(dates, prices)
        return self._standardize_data_format_Intelligent_P3_Generator(data, 'WMA_GOLDEN_CROSS')
    
    def generate_vix_high_data(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """生成VIX高波动数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        prices = [base_price]
        
        # 生成高波动率的随机游走
        for i in range(1, periods):
            # 大幅随机波动
            daily_change = np.random.uniform(-0.06, 0.06)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        # 使用极高波动率因子
        data = self._generate_ohlc_from_close_Intelligent_P3_Generator(dates, prices, volatility_factor=3.0)
        return self._standardize_data_format_Intelligent_P3_Generator(data, 'VIX_HIGH')
    
    def generate_volume_ratio_surge_data(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """生成成交量比率放大数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        prices = [base_price]
        
        # 前30天正常波动
        for i in range(1, 31):
            daily_change = np.random.uniform(-0.01, 0.01)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        # 后20天价格上涨，成交量放大
        for i in range(20):
            daily_change = np.random.uniform(0.015, 0.03)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        data = self._generate_ohlc_from_close_Intelligent_P3_Generator(dates, prices)
        
        # 特别处理成交量：后20天成交量显著放大
        for i in range(len(data)):
            if i >= 30:  # 后20天
                data.loc[i, 'volume'] *= np.random.uniform(3.0, 5.0)
        
        return self._standardize_data_format_Intelligent_P3_Generator(data, 'VOLUME_RATIO_SURGE')
    
    def generate_enhanced_cci_overbought_data(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """生成增强版CCI超买数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        prices = [base_price]
        
        # 快速上涨形成CCI超买
        for i in range(1, periods):
            daily_change = np.random.uniform(0.02, 0.035)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        data = self._generate_ohlc_from_close_Intelligent_P3_Generator(dates, prices)
        return self._standardize_data_format_Intelligent_P3_Generator(data, 'ENHANCED_CCI_OVERBOUGHT')
    
    def generate_enhanced_dmi_bullish_data(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """生成增强版DMI多头数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        prices = [base_price]
        
        # 持续上涨，+DI > -DI
        for i in range(1, periods):
            daily_change = np.random.uniform(0.012, 0.022)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        # 增加波动率以突出DMI特征
        data = self._generate_ohlc_from_close_Intelligent_P3_Generator(dates, prices, volatility_factor=1.3)
        return self._standardize_data_format_Intelligent_P3_Generator(data, 'ENHANCED_DMI_BULLISH')
    
    def generate_generic_pattern_data(self, pattern_name: str, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """生成通用形态数据（用于其他指标）"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        prices = [base_price]
        
        # 根据形态名称生成相应的价格序列
        if any(keyword in pattern_name for keyword in ['BULLISH', 'UPTREND', 'HIGH', 'OVERBOUGHT', 'BREAKOUT']):
            # 上涨形态
            for i in range(1, periods):
                daily_change = np.random.uniform(0.01, 0.025)
                new_price = prices[-1] * (1 + daily_change)
                prices.append(new_price)
        elif any(keyword in pattern_name for keyword in ['BEARISH', 'DOWNTREND', 'LOW', 'OVERSOLD', 'BREAKDOWN']):
            # 下跌形态
            for i in range(1, periods):
                daily_change = np.random.uniform(-0.025, -0.01)
                new_price = prices[-1] * (1 + daily_change)
                prices.append(new_price)
        elif 'DIVERGENCE' in pattern_name:
            # 背离形态
            for i in range(1, periods//2):
                daily_change = np.random.uniform(0.02, 0.03)
                new_price = prices[-1] * (1 + daily_change)
                prices.append(new_price)
            for i in range(periods//2):
                daily_change = np.random.uniform(0.005, 0.015)
                new_price = prices[-1] * (1 + daily_change)
                prices.append(new_price)
        elif 'SQUEEZE' in pattern_name:
            # 收缩形态
            for i in range(1, periods):
                daily_change = np.random.uniform(-0.005, 0.005)
                new_price = prices[-1] * (1 + daily_change)
                prices.append(new_price)
            # 使用低波动率
            data = self._generate_ohlc_from_close_Intelligent_P3_Generator(dates, prices, volatility_factor=0.5)
            return self._standardize_data_format_Intelligent_P3_Generator(data, pattern_name)
        else:
            # 默认：温和上涨
            for i in range(1, periods):
                daily_change = np.random.uniform(0.005, 0.015)
                new_price = prices[-1] * (1 + daily_change)
                prices.append(new_price)
        
        data = self._generate_ohlc_from_close_Intelligent_P3_Generator(dates, prices)
        return self._standardize_data_format_Intelligent_P3_Generator(data, pattern_name)


def test_intelligent_p3_generator():
    """测试智能P3生成器"""
    generator = Intelligent_p3_generator()
    
    print("测试智能P3数据生成器...")
    
    # 测试几个关键指标的数据生成
    atr_data = generator.generate_atr_high_volatility_data()
    print(f"ATR高波动数据: {len(atr_data)}行")
    print(f"价格变化: {atr_data['close'].iloc[0]:.2f} -> {atr_data['close'].iloc[-1]:.2f}")
    
    ichimoku_data = generator.generate_ichimoku_bullish_data()
    print(f"Ichimoku多头数据: {len(ichimoku_data)}行")
    print(f"价格变化: {ichimoku_data['close'].iloc[0]:.2f} -> {ichimoku_data['close'].iloc[-1]:.2f}")
    
    print("智能P3生成器测试完成！")


if __name__ == '__main__':
    test_intelligent_p3_generator()
