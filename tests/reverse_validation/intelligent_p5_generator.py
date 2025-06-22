#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
智能P5系统分析指标数据生成器

基于P1/P2/P3/P4阶段的成功经验，为P5系统分析指标智能生成确保形态成功的价格序列
支持12个P5指标，每个指标4个形态（系统分析指标通常使用4个形态）
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class IntelligentP5Generator:
    """智能P5系统分析指标数据生成器"""
    
    def __init__(self):
        pass
    
    def _generate_ohlc_from_close(self, dates: pd.DatetimeIndex, close_prices: List[float], 
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
            
            # 生成成交量（系统分析指标对成交量敏感）
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
    
    def _standardize_data_format(self, data: pd.DataFrame, pattern_name: str) -> pd.DataFrame:
        """标准化数据格式"""
        column_order = ['date', 'open', 'high', 'low', 'close', 'volume']
        return data[column_order]
    
    def generate_high_performance_data(self, base_price: float = 100, periods: int = 60) -> pd.DataFrame:
        """生成高性能数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        prices = [base_price]
        
        # 前20天建立基础
        for i in range(1, 21):
            daily_change = np.random.uniform(0.005, 0.015)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        # 中间20天强势表现
        for i in range(20):
            daily_change = np.random.uniform(0.02, 0.04)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        # 后20天持续优秀表现
        for i in range(20):
            daily_change = np.random.uniform(0.015, 0.025)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        data = self._generate_ohlc_from_close(dates, prices)
        
        # 特别处理成交量：高性能期间成交量放大
        for i in range(len(data)):
            if i >= 20:  # 后40天
                data.loc[i, 'volume'] *= np.random.uniform(1.5, 2.5)
        
        return self._standardize_data_format(data, 'HIGH_PERFORMANCE')
    
    def generate_positive_sentiment_data(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """生成积极情绪数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        prices = [base_price]
        
        # 持续积极的市场情绪：大部分时间上涨
        for i in range(1, periods):
            if i % 7 == 0:  # 每7天一次小回调
                daily_change = np.random.uniform(-0.005, 0.005)
            else:
                daily_change = np.random.uniform(0.01, 0.02)
            
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        data = self._generate_ohlc_from_close(dates, prices)
        
        # 积极情绪期间成交量活跃
        for i in range(len(data)):
            if i % 7 != 0:  # 非回调日成交量放大
                data.loc[i, 'volume'] *= np.random.uniform(1.2, 2.0)
        
        return self._standardize_data_format(data, 'POSITIVE_SENTIMENT')
    
    def generate_low_risk_data(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """生成低风险数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        prices = [base_price]
        
        # 低风险：小幅稳定波动
        for i in range(1, periods):
            daily_change = np.random.uniform(-0.008, 0.008)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        # 使用低波动率因子
        data = self._generate_ohlc_from_close(dates, prices, volatility_factor=0.5)
        return self._standardize_data_format(data, 'LOW_RISK')
    
    def generate_strong_trend_data(self, base_price: float = 100, periods: int = 60) -> pd.DataFrame:
        """生成强趋势数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        prices = [base_price]
        
        # 强势上升趋势
        for i in range(1, periods):
            # 大部分时间上涨，偶尔小回调
            if i % 10 == 0:  # 每10天一次小回调
                daily_change = np.random.uniform(-0.01, 0.005)
            else:
                daily_change = np.random.uniform(0.012, 0.022)
            
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'STRONG_TREND')
    
    def generate_high_momentum_data(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """生成高动量数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        prices = [base_price]
        
        # 高动量：加速上涨
        acceleration = 1.0
        for i in range(1, periods):
            # 动量逐渐增强
            acceleration += 0.001
            base_change = 0.01 * acceleration
            daily_change = np.random.uniform(base_change, base_change + 0.01)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        data = self._generate_ohlc_from_close(dates, prices)
        
        # 高动量期间成交量递增
        for i in range(len(data)):
            volume_multiplier = 1 + i * 0.02  # 成交量逐渐放大
            data.loc[i, 'volume'] *= volume_multiplier
        
        return self._standardize_data_format(data, 'HIGH_MOMENTUM')
    
    def generate_low_volatility_data(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """生成低波动率数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        prices = [base_price]
        
        # 低波动率：非常稳定的价格变化
        for i in range(1, periods):
            daily_change = np.random.uniform(-0.005, 0.005)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        # 使用极低波动率因子
        data = self._generate_ohlc_from_close(dates, prices, volatility_factor=0.3)
        return self._standardize_data_format(data, 'LOW_VOLATILITY')
    
    def generate_high_liquidity_data(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """生成高流动性数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        prices = [base_price]
        
        # 高流动性：价格变化平稳，成交量大
        for i in range(1, periods):
            daily_change = np.random.uniform(-0.01, 0.01)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        data = self._generate_ohlc_from_close(dates, prices)
        
        # 高流动性：成交量持续放大，振幅相对较小
        for i in range(len(data)):
            data.loc[i, 'volume'] *= np.random.uniform(2.0, 4.0)
        
        # 调整振幅使其更小
        data['high'] = data['close'] * (1 + np.random.uniform(0, 0.01, len(data)))
        data['low'] = data['close'] * (1 - np.random.uniform(0, 0.01, len(data)))
        
        return self._standardize_data_format(data, 'HIGH_LIQUIDITY')
    
    def generate_high_efficiency_data(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """生成高效率数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        prices = [base_price]
        
        # 高效率：价格变化方向性强，少回调
        trend_direction = 1  # 上升趋势
        for i in range(1, periods):
            # 高效率市场：价格变化方向一致
            daily_change = np.random.uniform(0.008, 0.018) * trend_direction
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'HIGH_EFFICIENCY')
    
    def generate_stable_system_data(self, base_price: float = 100, periods: int = 60) -> pd.DataFrame:
        """生成稳定系统数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        prices = [base_price]
        
        # 系统稳定：价格和成交量都很稳定
        for i in range(1, periods):
            daily_change = np.random.uniform(-0.006, 0.006)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        data = self._generate_ohlc_from_close(dates, prices, volatility_factor=0.4)
        
        # 成交量也保持稳定
        base_volume = data['volume'].mean()
        for i in range(len(data)):
            data.loc[i, 'volume'] = base_volume * np.random.uniform(0.8, 1.2)
        
        return self._standardize_data_format(data, 'STABLE_SYSTEM')
    
    def generate_generic_system_pattern_data(self, pattern_name: str, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """生成通用系统形态数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        prices = [base_price]
        
        # 根据形态名称生成相应的价格序列
        if any(keyword in pattern_name for keyword in ['HIGH', 'POSITIVE', 'STRONG', 'GOOD']):
            # 积极形态
            for i in range(1, periods):
                daily_change = np.random.uniform(0.008, 0.02)
                new_price = prices[-1] * (1 + daily_change)
                prices.append(new_price)
        elif any(keyword in pattern_name for keyword in ['LOW', 'NEGATIVE', 'WEAK', 'POOR']):
            # 消极形态
            for i in range(1, periods):
                daily_change = np.random.uniform(-0.02, -0.008)
                new_price = prices[-1] * (1 + daily_change)
                prices.append(new_price)
        elif any(keyword in pattern_name for keyword in ['STABLE', 'STEADY', 'BALANCED']):
            # 稳定形态
            for i in range(1, periods):
                daily_change = np.random.uniform(-0.005, 0.005)
                new_price = prices[-1] * (1 + daily_change)
                prices.append(new_price)
        else:
            # 默认：温和上涨
            for i in range(1, periods):
                daily_change = np.random.uniform(0.002, 0.012)
                new_price = prices[-1] * (1 + daily_change)
                prices.append(new_price)
        
        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, pattern_name)


def test_intelligent_p5_generator():
    """测试智能P5生成器"""
    generator = IntelligentP5Generator()
    
    print("测试智能P5数据生成器...")
    
    # 测试几个关键指标的数据生成
    performance_data = generator.generate_high_performance_data()
    print(f"高性能数据: {len(performance_data)}行")
    print(f"价格变化: {performance_data['close'].iloc[0]:.2f} -> {performance_data['close'].iloc[-1]:.2f}")
    
    sentiment_data = generator.generate_positive_sentiment_data()
    print(f"积极情绪数据: {len(sentiment_data)}行")
    print(f"价格变化: {sentiment_data['close'].iloc[0]:.2f} -> {sentiment_data['close'].iloc[-1]:.2f}")
    
    stability_data = generator.generate_stable_system_data()
    print(f"稳定系统数据: {len(stability_data)}行")
    print(f"价格变化: {stability_data['close'].iloc[0]:.2f} -> {stability_data['close'].iloc[-1]:.2f}")
    
    print("智能P5生成器测试完成！")


if __name__ == '__main__':
    test_intelligent_p5_generator()
