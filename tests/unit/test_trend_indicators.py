"""
趋势指标单元测试
"""
import unittest
import pandas as pd
import numpy as np
from indicators.ma import MA
from indicators.ema import EMA
from indicators.wma import WMA
from indicators.dmi import DMI
from indicators.atr import ATR
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator
from tests.helper.log_capture import LogCaptureMixin

class TestMA(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    def setUp(self):
        LogCaptureMixin.setUp(self)
        self.indicator = MA(periods=[5, 20])
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 50}
        ])
        self.expected_columns = ['MA5', 'MA20']

    def tearDown(self):
        LogCaptureMixin.tearDown(self)

class TestEMA(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    def setUp(self):
        LogCaptureMixin.setUp(self)
        self.indicator = EMA(periods=[12, 26])
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 50}
        ])
        self.expected_columns = ['EMA12', 'EMA26']

    def tearDown(self):
        LogCaptureMixin.tearDown(self)

class TestWMA(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    def setUp(self):
        LogCaptureMixin.setUp(self)
        self.indicator = WMA(periods=[5, 10])
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 50}
        ])
        self.expected_columns = ['WMA5', 'WMA10']

    def tearDown(self):
        LogCaptureMixin.tearDown(self)

class TestDMI(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    def setUp(self):
        LogCaptureMixin.setUp(self)
        self.indicator = DMI(period=14, adx_period=6)
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 50}
        ])
        self.expected_columns = ['pdi', 'mdi', 'adx', 'adxr']

    def tearDown(self):
        LogCaptureMixin.tearDown(self)

class TestATR(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    """ATR指标单元测试类"""

    def setUp(self):
        """准备数据和指标实例"""
        LogCaptureMixin.setUp(self)  # 显式调用Mixin的setUp
        # ATR argument is period not periods
        self.indicator = ATR(params={'period': 14})
        self.expected_columns = ['TR', 'ATR14']
        # 使用一个包含多种走势的数据进行通用测试
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 30},
            {'type': 'trend', 'start_price': 120, 'end_price': 100, 'periods': 30},
        ])

    def tearDown(self):
        """清理日志捕获器"""
        LogCaptureMixin.tearDown(self)  # 显式调用Mixin的tearDown

    def test_basic_calculation(self):
        """测试ATR基础计算功能"""
        result = self.indicator.calculate(self.data)

        # 验证返回类型
        self.assertIsInstance(result, pd.DataFrame, "ATR计算结果应为DataFrame")

        # 验证包含ATR列
        self.assertIn('TR', result.columns, "结果应包含TR列")
        self.assertIn('ATR14', result.columns, "结果应包含ATR14列")

        # 验证ATR值为正数
        atr_values = result['ATR14'].dropna()
        self.assertTrue(all(val >= 0 for val in atr_values), "ATR值应为非负数")

        # 验证TR值为正数
        tr_values = result['TR'].dropna()
        self.assertTrue(all(val >= 0 for val in tr_values), "TR值应为非负数")

    def test_volatility_detection(self):
        """测试ATR波动性检测"""
        # 生成高波动性数据
        high_vol_data = TestDataGenerator.generate_price_sequence([
            {'type': 'v_shape', 'start_price': 100, 'bottom_price': 80, 'periods': 30}
        ])

        result = self.indicator.calculate(high_vol_data)

        # 在高波动性数据中，ATR应该相对较高
        atr_values = result['ATR14'].dropna()
        if len(atr_values) > 10:
            # 检查ATR是否反映了波动性
            avg_atr = atr_values.mean()
            self.assertGreater(avg_atr, 0, "高波动性数据中ATR应大于0")

    def test_low_volatility_detection(self):
        """测试ATR低波动性检测"""
        # 生成低波动性数据（横盘）
        low_vol_data = TestDataGenerator.generate_price_sequence([
            {'type': 'sideways', 'start_price': 100, 'volatility': 0.005, 'periods': 30}
        ])

        result = self.indicator.calculate(low_vol_data)

        # 在低波动性数据中，ATR应该相对较低
        atr_values = result['ATR14'].dropna()
        if len(atr_values) > 10:
            # 检查ATR是否反映了低波动性
            avg_atr = atr_values.mean()
            self.assertGreater(avg_atr, 0, "即使在低波动性数据中ATR也应大于0")

    def test_signal_generation(self):
        """测试ATR信号生成"""
        # 生成包含多种走势的数据
        data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 100, 'periods': 20},  # 横盘
            {'type': 'trend', 'start_price': 100, 'end_price': 90, 'periods': 10},   # 下跌
            {'type': 'trend', 'start_price': 90, 'end_price': 110, 'periods': 15},   # 反弹
        ])

        signals = self.indicator.generate_signals(data)

        # 验证信号DataFrame结构
        self.assertIsInstance(signals, pd.DataFrame, "信号结果应为DataFrame")
        self.assertIn('volatility_high', signals.columns, "信号结果应包含volatility_high列")
        self.assertIn('volatility_low', signals.columns, "信号结果应包含volatility_low列")
        self.assertIn('atr_rising', signals.columns, "信号结果应包含atr_rising列")
        self.assertIn('atr_falling', signals.columns, "信号结果应包含atr_falling列")

    def test_pattern_detection(self):
        """测试ATR形态检测"""
        data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 100, 'periods': 20},  # 横盘
            {'type': 'v_shape', 'start_price': 100, 'bottom_price': 85, 'periods': 20},  # V形反转
        ])

        patterns = self.indicator.get_patterns(data)

        # 验证形态检测结果
        self.assertIsInstance(patterns, pd.DataFrame, "形态检测结果应为DataFrame")

        # 检查是否包含ATR形态列
        expected_patterns = ['ATR_UPWARD_BREAKOUT', 'ATR_DOWNWARD_BREAKOUT',
                           'VOLATILITY_COMPRESSION', 'VOLATILITY_EXPANSION']
        for pattern in expected_patterns:
            self.assertIn(pattern, patterns.columns, f"应包含{pattern}形态")

    def test_score_calculation(self):
        """测试ATR评分计算功能"""
        # 生成测试数据
        data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 100, 'periods': 20},  # 横盘
            {'type': 'trend', 'start_price': 100, 'end_price': 90, 'periods': 10},   # 下跌
            {'type': 'trend', 'start_price': 90, 'end_price': 110, 'periods': 15},   # 反弹
        ])

        # 验证原始评分计算
        raw_score = self.indicator.calculate_raw_score(data)
        self.assertIsInstance(raw_score, pd.Series, "原始评分应为Series")
        self.assertTrue(all(0 <= s <= 100 for s in raw_score if not pd.isna(s)), "原始评分应在0-100范围内")

    def test_parameter_setting(self):
        """测试ATR参数设置"""
        # 测试参数设置方法
        new_period = 21
        new_threshold = 2.5

        self.indicator.set_parameters(period=new_period, high_volatility_threshold=new_threshold)

        # 验证参数是否正确设置
        self.assertEqual(self.indicator.params['period'], new_period, "周期参数设置失败")
        self.assertEqual(self.indicator.params['high_volatility_threshold'], new_threshold, "阈值参数设置失败")

        # 验证新参数下的计算
        result = self.indicator.calculate(self.data)
        self.assertIn(f'ATR{new_period}', result.columns, f"结果应包含ATR{new_period}列")

    def test_edge_cases(self):
        """测试边界条件"""
        # 测试数据长度不足的情况
        short_data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 105, 'periods': 5}
        ])

        result = self.indicator.calculate(short_data)
        self.assertIsInstance(result, pd.DataFrame, "短数据计算结果应为DataFrame")

        # 测试价格无变化的情况
        flat_data = TestDataGenerator.generate_price_sequence([
            {'type': 'flat', 'start_price': 100, 'periods': 20}
        ])

        # 确保价格完全相等
        for i in range(len(flat_data)):
            flat_data.iloc[i, flat_data.columns.get_loc('high')] = 100
            flat_data.iloc[i, flat_data.columns.get_loc('low')] = 100
            flat_data.iloc[i, flat_data.columns.get_loc('close')] = 100
            flat_data.iloc[i, flat_data.columns.get_loc('open')] = 100

        result = self.indicator.calculate(flat_data)
        self.assertIsInstance(result, pd.DataFrame, "平价数据计算结果应为DataFrame")

        # 在价格无变化的情况下，ATR应该为0或接近0
        atr_values = result['ATR14'].dropna()
        if len(atr_values) > 0:
            self.assertTrue(all(val <= 0.01 for val in atr_values), "价格无变化时ATR应接近0")

    def test_robustness(self):
        """测试ATR指标的鲁棒性"""
        # 测试包含异常值的数据
        data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 20}
        ])

        # 添加一些异常值
        data.loc[data.index[10], 'high'] = data.loc[data.index[10], 'high'] * 2    # 最高价异常
        data.loc[data.index[15], 'low'] = data.loc[data.index[15], 'low'] * 0.5     # 最低价异常

        result = self.indicator.calculate(data)

        # 验证计算没有出错
        self.assertIsInstance(result, pd.DataFrame, "异常数据计算结果应为DataFrame")
        self.assertIn('ATR14', result.columns, "结果应包含ATR14列")

        # 验证ATR值是有限的数值
        atr_values = result['ATR14'].dropna()
        self.assertTrue(all(np.isfinite(val) for val in atr_values), "ATR值应为有限数值")
        self.assertTrue(all(val >= 0 for val in atr_values), "ATR值应为非负数")

if __name__ == '__main__':
    unittest.main() 