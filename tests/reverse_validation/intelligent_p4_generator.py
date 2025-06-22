#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
智能P4 ZXM系列指标数据生成器

基于P1/P2/P3阶段的成功经验，为P4 ZXM系列指标智能生成确保形态成功的价格序列
支持15个P4指标，每个指标4个形态（ZXM系列通常使用4个形态而非5个）
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class IntelligentP4Generator:
    """智能P4 ZXM系列指标数据生成器"""
    
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
            
            # 生成成交量（ZXM指标对成交量敏感）
            base_volume = 5000000
            volume_factor = 1 + abs(close - close_prices[0]) / close_prices[0]
            volume = int(base_volume * volume_factor * np.random.uniform(0.5, 2.0))
            
            # 添加换手率数据（ZXM_TURNOVER需要）
            turnover_rate = volume / 100000000 * 100  # 简化计算
            
            data.append({
                'date': date,
                'open': open_price,
                'high': max(open_price, high, close),
                'low': min(open_price, low, close),
                'close': close,
                'volume': volume,
                'turnover_rate': turnover_rate
            })
        
        return pd.DataFrame(data)
    
    def _standardize_data_format(self, data: pd.DataFrame, pattern_name: str) -> pd.DataFrame:
        """标准化数据格式"""
        column_order = ['date', 'open', 'high', 'low', 'close', 'volume', 'turnover_rate']
        return data[column_order]
    
    def generate_zxm_macd_buy_signal_data(self, base_price: float = 100, periods: int = 60) -> pd.DataFrame:
        """生成ZXM MACD买点信号数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        prices = [base_price]
        
        # 前30天下跌，形成MACD负值
        for i in range(1, 31):
            daily_change = np.random.uniform(-0.02, -0.005)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        # 后30天缓慢上涨，MACD接近0轴
        for i in range(30):
            daily_change = np.random.uniform(0.002, 0.008)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'ZXM_MACD_BUY_SIGNAL')
    
    def generate_zxm_turnover_active_data(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """生成ZXM换手率活跃数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        prices = [base_price]
        
        # 前30天正常波动
        for i in range(1, 31):
            daily_change = np.random.uniform(-0.01, 0.01)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        # 后20天活跃交易，换手率提高
        for i in range(20):
            daily_change = np.random.uniform(0.01, 0.02)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        data = self._generate_ohlc_from_close(dates, prices)
        
        # 特别处理换手率：后20天换手率明显提高
        for i in range(len(data)):
            if i >= 30:  # 后20天
                data.loc[i, 'turnover_rate'] *= np.random.uniform(2.0, 4.0)
        
        return self._standardize_data_format(data, 'ZXM_TURNOVER_ACTIVE')
    
    def generate_zxm_volume_shrink_data(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """生成ZXM缩量数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        prices = [base_price]
        
        # 前30天正常成交量
        for i in range(1, 31):
            daily_change = np.random.uniform(-0.005, 0.005)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        # 后20天缩量整理
        for i in range(20):
            daily_change = np.random.uniform(-0.003, 0.003)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        data = self._generate_ohlc_from_close(dates, prices)
        
        # 特别处理成交量：后20天成交量明显缩小
        for i in range(len(data)):
            if i >= 30:  # 后20天
                data.loc[i, 'volume'] *= np.random.uniform(0.3, 0.6)
                data.loc[i, 'turnover_rate'] *= np.random.uniform(0.3, 0.6)
        
        return self._standardize_data_format(data, 'ZXM_VOLUME_SHRINK')
    
    def generate_zxm_ma_callback_data(self, base_price: float = 100, periods: int = 60) -> pd.DataFrame:
        """生成ZXM均线回调数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        prices = [base_price]
        
        # 前30天上涨建立均线
        for i in range(1, 31):
            daily_change = np.random.uniform(0.008, 0.015)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        # 后30天回调到均线附近
        peak_price = prices[-1]
        for i in range(30):
            # 逐渐回调到接近20日均线
            target_price = peak_price * (0.95 + 0.03 * (30 - i) / 30)
            daily_change = (target_price - prices[-1]) / prices[-1]
            daily_change += np.random.uniform(-0.005, 0.005)  # 添加随机性
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'ZXM_MA_CALLBACK')
    
    def generate_zxm_absorb_data(self, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """生成ZXM主力吸筹数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        prices = [base_price]
        
        # 前20天正常波动
        for i in range(1, 21):
            daily_change = np.random.uniform(-0.01, 0.01)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        # 后30天主力吸筹：小幅波动，成交量放大
        for i in range(30):
            daily_change = np.random.uniform(-0.008, 0.008)  # 小幅波动
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        data = self._generate_ohlc_from_close(dates, prices)
        
        # 特别处理成交量：后30天成交量放大但价格波动小
        for i in range(len(data)):
            if i >= 20:  # 后30天
                data.loc[i, 'volume'] *= np.random.uniform(1.5, 2.5)
                data.loc[i, 'turnover_rate'] *= np.random.uniform(1.5, 2.5)
        
        return self._standardize_data_format(data, 'ZXM_ABSORB')
    
    def generate_zxm_elasticity_data(self, base_price: float = 100, periods: int = 80) -> pd.DataFrame:
        """生成ZXM弹性数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        prices = [base_price]
        
        # 前40天下跌到低点
        for i in range(1, 41):
            daily_change = np.random.uniform(-0.015, -0.005)
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        # 后40天弹性反弹
        low_price = prices[-1]
        for i in range(40):
            # 弹性反弹，振幅较大
            if i % 5 == 0:  # 每5天一次大幅反弹
                daily_change = np.random.uniform(0.05, 0.1)
            else:
                daily_change = np.random.uniform(0.01, 0.03)
            
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        # 使用高波动率因子
        data = self._generate_ohlc_from_close(dates, prices, volatility_factor=2.0)
        return self._standardize_data_format(data, 'ZXM_ELASTICITY')
    
    def generate_zxm_trend_up_data(self, base_price: float = 100, periods: int = 60) -> pd.DataFrame:
        """生成ZXM上升趋势数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        prices = [base_price]
        
        # 持续上升趋势
        for i in range(1, periods):
            # 大部分时间上涨
            if i % 8 == 0:  # 偶尔小回调
                daily_change = np.random.uniform(-0.005, 0.002)
            else:
                daily_change = np.random.uniform(0.008, 0.018)
            
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        data = self._generate_ohlc_from_close(dates, prices)
        return self._standardize_data_format(data, 'ZXM_TREND_UP')
    
    def generate_generic_zxm_pattern_data(self, pattern_name: str, base_price: float = 100, periods: int = 50) -> pd.DataFrame:
        """生成通用ZXM形态数据"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        prices = [base_price]
        
        # 根据形态名称生成相应的价格序列
        if any(keyword in pattern_name for keyword in ['BUY', 'SIGNAL', 'ACTIVE', 'UP', 'BULLISH']):
            # 买点/上涨形态
            for i in range(1, periods):
                daily_change = np.random.uniform(0.005, 0.02)
                new_price = prices[-1] * (1 + daily_change)
                prices.append(new_price)
        elif any(keyword in pattern_name for keyword in ['SELL', 'DOWN', 'BEARISH', 'DECLINE']):
            # 卖点/下跌形态
            for i in range(1, periods):
                daily_change = np.random.uniform(-0.02, -0.005)
                new_price = prices[-1] * (1 + daily_change)
                prices.append(new_price)
        elif any(keyword in pattern_name for keyword in ['SHRINK', 'CALLBACK', 'CONSOLIDATION']):
            # 缩量/回调形态
            for i in range(1, periods):
                daily_change = np.random.uniform(-0.008, 0.008)
                new_price = prices[-1] * (1 + daily_change)
                prices.append(new_price)
        elif 'ELASTICITY' in pattern_name:
            # 弹性形态：先跌后涨
            for i in range(1, periods//2):
                daily_change = np.random.uniform(-0.02, -0.01)
                new_price = prices[-1] * (1 + daily_change)
                prices.append(new_price)
            for i in range(periods//2):
                daily_change = np.random.uniform(0.01, 0.03)
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


def test_intelligent_p4_generator():
    """测试智能P4生成器"""
    generator = IntelligentP4Generator()
    
    print("测试智能P4数据生成器...")
    
    # 测试几个关键指标的数据生成
    macd_data = generator.generate_zxm_macd_buy_signal_data()
    print(f"ZXM MACD买点数据: {len(macd_data)}行")
    print(f"价格变化: {macd_data['close'].iloc[0]:.2f} -> {macd_data['close'].iloc[-1]:.2f}")
    
    turnover_data = generator.generate_zxm_turnover_active_data()
    print(f"ZXM换手率活跃数据: {len(turnover_data)}行")
    print(f"换手率变化: {turnover_data['turnover_rate'].iloc[0]:.2f} -> {turnover_data['turnover_rate'].iloc[-1]:.2f}")
    
    elasticity_data = generator.generate_zxm_elasticity_data()
    print(f"ZXM弹性数据: {len(elasticity_data)}行")
    print(f"价格变化: {elasticity_data['close'].iloc[0]:.2f} -> {elasticity_data['close'].iloc[-1]:.2f}")
    
    print("智能P4生成器测试完成！")


if __name__ == '__main__':
    test_intelligent_p4_generator()
