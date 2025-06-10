import pandas as pd
import numpy as np
from typing import List, Dict, Any

class TestDataGenerator:
    """
    一个用于生成标准测试数据DataFrame的工具类。
    所有生成器都返回一个包含 'open', 'high', 'low', 'close', 'volume' 列的DataFrame。
    为了简化，OHLC价格被设置为相同。
    """

    @staticmethod
    def _create_df(prices: np.ndarray, dates=None) -> pd.DataFrame:
        """根据价格数组创建标准的DataFrame。"""
        if dates is None:
            dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=len(prices), freq='D'))
        
        data = {
            'open': prices,
            'high': prices,
            'low': prices,
            'close': prices,
            'volume': np.full(len(prices), 1000)
        }
        return pd.DataFrame(data, index=dates)

    @staticmethod
    def generate_v_shape(start_price: float = 100.0, bottom_price: float = 90.0, periods: int = 50) -> pd.DataFrame:
        """
        生成V形反转的价格序列。
        序列前半部分下跌，后半部分反弹。
        """
        mid_point = periods // 2
        decline = np.linspace(start_price, bottom_price, mid_point)
        rebound = np.linspace(bottom_price, start_price, periods - mid_point)
        prices = np.concatenate([decline, rebound])
        return TestDataGenerator._create_df(prices)

    @staticmethod
    def generate_m_shape(start_price: float = 100.0, top_price: float = 110.0, periods: int = 50) -> pd.DataFrame:
        """
        生成M形顶的价格序列。
        序列前半部分上涨，后半部分下跌。
        """
        mid_point = periods // 2
        rise = np.linspace(start_price, top_price, mid_point)
        fall = np.linspace(top_price, start_price, periods - mid_point)
        prices = np.concatenate([rise, fall])
        return TestDataGenerator._create_df(prices)

    @staticmethod
    def generate_steady_trend(start_price: float = 100.0, end_price: float = 110.0, periods: int = 100) -> pd.DataFrame:
        """生成平稳的单边趋势（上涨或下跌）。"""
        prices = np.linspace(start_price, end_price, periods)
        return TestDataGenerator._create_df(prices)

    @staticmethod
    def generate_sideways_channel(price_level: float = 100.0, volatility: float = 2.0, periods: int = 100) -> pd.DataFrame:
        """生成在固定价格中枢附近小幅波动的震荡行情。"""
        t = np.linspace(0, 4 * np.pi, periods)
        prices = price_level + volatility * np.sin(t)
        return TestDataGenerator._create_df(prices)

    @staticmethod
    def generate_price_sequence(config: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        根据配置列表拼接生成复杂行情序列。
        例如: [{'type': 'trend', 'end_price': 110, 'periods': 30}, {'type': 'channel', 'periods': 40}]
        """
        all_prices = np.array([])
        
        for i, segment in enumerate(config):
            periods = segment['periods']
            
            if i == 0:
                start_price = segment.get('start_price', 100.0)
            else:
                start_price = all_prices[-1]

            seg_type = segment.get('type')

            if seg_type == 'trend':
                prices = np.linspace(start_price, segment['end_price'], periods)
            elif seg_type == 'channel':
                volatility = segment.get('volatility', 2.0)
                t = np.linspace(0, 2 * np.pi * (periods / 20), periods)
                prices = start_price + volatility * np.sin(t)
            elif seg_type == 'v_shape':
                mid_point = periods // 2
                decline = np.linspace(start_price, segment['bottom_price'], mid_point)
                rebound = np.linspace(segment['bottom_price'], start_price, periods - mid_point)
                prices = np.concatenate([decline, rebound])
            elif seg_type == 'm_shape':
                mid_point = periods // 2
                rise = np.linspace(start_price, segment['top_price'], mid_point)
                fall = np.linspace(segment['top_price'], start_price, periods - mid_point)
                prices = np.concatenate([rise, fall])
            else:
                prices = np.full(periods, start_price)

            all_prices = np.concatenate([all_prices, prices])
            
        return TestDataGenerator._create_df(all_prices) 