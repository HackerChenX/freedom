import unittest
import pandas as pd

from indicators.boll import BOLL
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator

class TestBOLL(unittest.TestCase, IndicatorTestMixin):
    """BOLL指标单元测试类"""

    def setUp(self):
        """准备数据和指标实例"""
        self.indicator = BOLL(period=20, std_dev=2)
        self.expected_columns = ['boll_upper', 'boll_middle', 'boll_lower']
        self.data = TestDataGenerator.generate_steady_trend(periods=100)

    def test_boll_squeeze(self):
        """测试布林带缩口（squeeze）"""
        # 在窄幅震荡行情中，布林带宽度应该会变小
        data = TestDataGenerator.generate_sideways_channel(price_level=100, volatility=1, periods=100)
        result = self.indicator.calculate(data)
        result['width'] = result['boll_upper'] - result['boll_lower']
        
        # 期望在震荡后期，带宽小于一个阈值
        self.assertLess(result['width'].iloc[-1], 5, "布林带在震荡市中未能有效缩口") # 5是基于经验的阈值

    def test_boll_breakout(self):
        """测试布林带开口（breakout）"""
        # V形反转的剧烈波动应该导致布林带开口
        data = TestDataGenerator.generate_v_shape(start_price=100, bottom_price=80, periods=50)
        result = self.indicator.calculate(data)
        result['width'] = result['boll_upper'] - result['boll_lower']

        # 期望在趋势开始后，带宽显著增大
        # 比较趋势开始前和结束后的带宽
        initial_width = result['width'].iloc[25] # 波动发生前的带宽
        final_width = result['width'].iloc[-1]   # 波动发生后的带宽
        self.assertGreater(final_width, initial_width * 1.5, "布林带在趋势行情中未能有效开口")
        
    def test_price_within_bands(self):
        """测试价格大部分时间在布林带轨道内"""
        data = TestDataGenerator.generate_price_sequence([
             {'type': 'channel', 'volatility': 3, 'periods': 200}
        ])
        result = self.indicator.calculate(data)
        
        # 计算价格在轨道内的比例
        within_bands = (data['close'] <= result['boll_upper']) & (data['close'] >= result['boll_lower'])
        # 剔除计算初期的NaN值
        within_bands_ratio = within_bands.iloc[self.indicator.period:].mean()
        
        # 根据统计学，约95%的数据点应落在2个标准差内
        self.assertGreater(within_bands_ratio, 0.9, "大部分价格点应落在布林带轨道内")


if __name__ == '__main__':
    unittest.main() 