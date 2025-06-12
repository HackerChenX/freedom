import unittest
import pandas as pd

from indicators.kdj import KDJ
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator
from tests.helper.log_capture import LogCaptureMixin

class TestKDJ(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    """KDJ指标单元测试类"""

    def setUp(self):
        """准备数据和指标实例"""
        LogCaptureMixin.setUp(self)  # 显式调用Mixin的setUp
        self.indicator = KDJ(n=9, m1=3, m2=3)
        self.expected_columns = ['K', 'D', 'J']
        # 使用一个包含多种走势的数据进行通用测试
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 30},
            {'type': 'trend', 'start_price': 120, 'end_price': 100, 'periods': 30},
        ])

    def tearDown(self):
        """清理日志捕获器"""
        LogCaptureMixin.tearDown(self) # 显式调用Mixin的tearDown

    def test_golden_cross(self):
        """测试KDJ金叉"""
        data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 100, 'periods': 20},
            {'type': 'trend', 'start_price': 100, 'end_price': 90, 'periods': 10}, # 下跌
            {'type': 'trend', 'start_price': 90, 'end_price': 105, 'periods': 15}, # 反弹
        ])
        calculated_data = self.indicator.calculate(data)
        patterns = self.indicator.get_patterns(calculated_data)
        self.assertIn('KDJ_GOLDEN_CROSS', patterns.columns, "模式结果中缺少 KDJ_GOLDEN_CROSS 列")
        self.assertGreater(patterns['KDJ_GOLDEN_CROSS'].sum(), 0, "未检测到KDJ金叉")

    def test_death_cross(self):
        """测试KDJ死叉"""
        data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 100, 'periods': 20},
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 10}, # 上涨
            {'type': 'trend', 'start_price': 110, 'end_price': 95, 'periods': 15}, # 回调
        ])
        calculated_data = self.indicator.calculate(data)
        patterns = self.indicator.get_patterns(calculated_data)
        self.assertIn('KDJ_DEATH_CROSS', patterns.columns, "模式结果中缺少 KDJ_DEATH_CROSS 列")
        self.assertGreater(patterns['KDJ_DEATH_CROSS'].sum(), 0, "未检测到KDJ死叉")

    def test_j_value_overbought(self):
        """测试J值超买"""
        # 生成更剧烈的上涨趋势数据
        data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 100, 'periods': 20},  # 横盘
            {'type': 'trend', 'start_price': 100, 'end_price': 150, 'periods': 10},  # 快速上涨
        ])
        result = self.indicator.calculate(data)
        # J值在持续上涨后应超过100
        self.assertTrue((result['J'].iloc[-5:] > 100).any(), "J值未进入超买区")

    def test_j_value_oversold(self):
        """测试J值超卖"""
        # 生成更剧烈的下跌趋势数据
        data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 100, 'periods': 20},  # 横盘
            {'type': 'trend', 'start_price': 100, 'end_price': 70, 'periods': 10},   # 快速下跌
        ])
        result = self.indicator.calculate(data)
        # J值在持续下跌后应低于0
        self.assertTrue((result['J'].iloc[-5:] < 0).any(), "J值未进入超卖区")

    def test_score_calculation(self):
        """测试KDJ评分计算功能"""
        # 生成测试数据
        data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 100, 'periods': 20},  # 横盘
            {'type': 'trend', 'start_price': 100, 'end_price': 70, 'periods': 10},   # 快速下跌
            {'type': 'trend', 'start_price': 70, 'end_price': 120, 'periods': 10},   # 快速上涨
        ])
        
        # 计算KDJ
        result = self.indicator.calculate(data)
        
        # 将KDJ结果合并到原始数据中
        data_with_kdj = data.copy()
        data_with_kdj['K'] = result['K']
        data_with_kdj['D'] = result['D']
        data_with_kdj['J'] = result['J']
        
        # 计算评分
        score = self.indicator.calculate_score(data_with_kdj)
        
        # 验证评分类型和范围
        self.assertIsInstance(score, float, "评分应为浮点数")
        self.assertTrue(0 <= score <= 100, "评分应在0-100范围内")
        
        # 验证原始评分计算
        raw_score = self.indicator.calculate_raw_score(data_with_kdj)
        self.assertIsInstance(raw_score, pd.Series, "原始评分应为Series")
        self.assertTrue(all(0 <= s <= 100 for s in raw_score if not pd.isna(s)), "原始评分应在0-100范围内")
        
        # 验证评分与KDJ值的关系
        # 在超卖区域应该有较高的评分
        oversold_data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 100, 'periods': 20},  # 横盘
            {'type': 'trend', 'start_price': 100, 'end_price': 70, 'periods': 10},   # 快速下跌
        ])
        oversold_result = self.indicator.calculate(oversold_data)
        oversold_data_with_kdj = oversold_data.copy()
        oversold_data_with_kdj['K'] = oversold_result['K']
        oversold_data_with_kdj['D'] = oversold_result['D']
        oversold_data_with_kdj['J'] = oversold_result['J']
        oversold_score = self.indicator.calculate_score(oversold_data_with_kdj)
        self.assertGreater(oversold_score, 50, "超卖区域的评分应该较高")
        
        # 在超买区域应该有较低的评分
        overbought_data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 70, 'end_price': 70, 'periods': 20},    # 横盘
            {'type': 'trend', 'start_price': 70, 'end_price': 120, 'periods': 10},   # 快速上涨
        ])
        overbought_result = self.indicator.calculate(overbought_data)
        overbought_data_with_kdj = overbought_data.copy()
        overbought_data_with_kdj['K'] = overbought_result['K']
        overbought_data_with_kdj['D'] = overbought_result['D']
        overbought_data_with_kdj['J'] = overbought_result['J']
        overbought_score = self.indicator.calculate_score(overbought_data_with_kdj)
        self.assertLess(overbought_score, 50, "超买区域的评分应该较低")

if __name__ == '__main__':
    unittest.main() 