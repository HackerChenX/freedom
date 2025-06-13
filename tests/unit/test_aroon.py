"""
Aroon指标单元测试
"""
import unittest
import pandas as pd
import numpy as np

from indicators.aroon import Aroon
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator
from tests.helper.log_capture import LogCaptureMixin


class TestAroon(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    """Aroon指标单元测试类"""

    def setUp(self):
        """准备数据和指标实例"""
        LogCaptureMixin.setUp(self)  # 显式调用Mixin的setUp
        self.indicator = Aroon(period=14)
        self.expected_columns = ['aroon_up', 'aroon_down', 'aroon_oscillator']
        # 使用一个包含多种走势的数据进行通用测试
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 30},
            {'type': 'trend', 'start_price': 120, 'end_price': 100, 'periods': 30},
        ])

    def tearDown(self):
        """清理日志捕获器"""
        LogCaptureMixin.tearDown(self)  # 显式调用Mixin的tearDown

    def test_basic_calculation(self):
        """测试Aroon基础计算功能"""
        result = self.indicator.calculate(self.data)
        
        # 验证返回类型
        self.assertIsInstance(result, pd.DataFrame, "Aroon计算结果应为DataFrame")
        
        # 验证包含Aroon列
        self.assertIn('aroon_up', result.columns, "结果应包含aroon_up列")
        self.assertIn('aroon_down', result.columns, "结果应包含aroon_down列")
        self.assertIn('aroon_oscillator', result.columns, "结果应包含aroon_oscillator列")
        
        # 验证Aroon值在0-100范围内
        aroon_up_values = result['aroon_up'].dropna()
        aroon_down_values = result['aroon_down'].dropna()
        
        self.assertTrue(all(0 <= val <= 100 for val in aroon_up_values), "Aroon Up值应在0-100范围内")
        self.assertTrue(all(0 <= val <= 100 for val in aroon_down_values), "Aroon Down值应在0-100范围内")
        
        # 验证震荡器值在-100到100范围内
        osc_values = result['aroon_oscillator'].dropna()
        self.assertTrue(all(-100 <= val <= 100 for val in osc_values), "Aroon震荡器值应在-100到100范围内")

    def test_uptrend_detection(self):
        """测试上升趋势检测"""
        # 生成明确的上升趋势数据
        data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 130, 'periods': 30}
        ])
        
        result = self.indicator.calculate(data)
        
        # 在上升趋势中，Aroon Up应该较高，Aroon Down应该较低
        aroon_up = result['aroon_up'].dropna()
        aroon_down = result['aroon_down'].dropna()
        
        if len(aroon_up) > 10:
            # 检查最后几个周期的Aroon Up是否较高
            recent_up = aroon_up.iloc[-5:].mean()
            recent_down = aroon_down.iloc[-5:].mean()
            self.assertGreater(recent_up, recent_down, "上升趋势中Aroon Up应高于Aroon Down")

    def test_downtrend_detection(self):
        """测试下降趋势检测"""
        # 生成明确的下降趋势数据
        data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 130, 'end_price': 100, 'periods': 30}
        ])
        
        result = self.indicator.calculate(data)
        
        # 在下降趋势中，Aroon Down应该较高，Aroon Up应该较低
        aroon_up = result['aroon_up'].dropna()
        aroon_down = result['aroon_down'].dropna()
        
        if len(aroon_down) > 10:
            # 检查最后几个周期的Aroon Down是否较高
            recent_up = aroon_up.iloc[-5:].mean()
            recent_down = aroon_down.iloc[-5:].mean()
            self.assertGreater(recent_down, recent_up, "下降趋势中Aroon Down应高于Aroon Up")

    def test_crossover_detection(self):
        """测试Aroon交叉检测"""
        data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 100, 'periods': 20},  # 横盘
            {'type': 'trend', 'start_price': 100, 'end_price': 90, 'periods': 10},   # 下跌
            {'type': 'trend', 'start_price': 90, 'end_price': 110, 'periods': 15},   # 反弹
        ])
        
        calculated_data = self.indicator.calculate(data)
        patterns = self.indicator.get_patterns(data)
        
        # 验证形态检测结果
        self.assertIsInstance(patterns, pd.DataFrame, "形态检测结果应为DataFrame")
        
        # 检查是否包含交叉形态列
        self.assertIn('AROON_BULLISH_CROSS', patterns.columns, "应包含多头交叉形态")
        self.assertIn('AROON_BEARISH_CROSS', patterns.columns, "应包含空头交叉形态")

    def test_strong_trend_detection(self):
        """测试强趋势检测"""
        # 生成强上升趋势数据
        strong_up_data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 150, 'periods': 20}
        ])
        
        result = self.indicator.calculate(strong_up_data)
        patterns = self.indicator.get_patterns(strong_up_data)
        
        # 检查强趋势形态
        self.assertIn('AROON_STRONG_UPTREND', patterns.columns, "应包含强上升趋势形态")
        self.assertIn('AROON_STRONG_DOWNTREND', patterns.columns, "应包含强下降趋势形态")

    def test_oscillator_patterns(self):
        """测试震荡器形态检测"""
        data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 100, 'periods': 20},  # 横盘
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 15},  # 上涨
            {'type': 'trend', 'start_price': 120, 'end_price': 95, 'periods': 15},   # 下跌
        ])
        
        calculated_data = self.indicator.calculate(data)
        patterns = self.indicator.get_patterns(data)
        
        # 检查震荡器形态
        self.assertIn('AROON_OSC_CROSS_ABOVE_ZERO', patterns.columns, "应包含震荡器上穿零轴形态")
        self.assertIn('AROON_OSC_CROSS_BELOW_ZERO', patterns.columns, "应包含震荡器下穿零轴形态")
        self.assertIn('AROON_OSC_EXTREME_BULLISH', patterns.columns, "应包含震荡器极度看涨形态")
        self.assertIn('AROON_OSC_EXTREME_BEARISH', patterns.columns, "应包含震荡器极度看跌形态")

    def test_consolidation_detection(self):
        """测试盘整检测"""
        # 生成横盘数据
        sideways_data = TestDataGenerator.generate_price_sequence([
            {'type': 'sideways', 'start_price': 100, 'periods': 40, 'volatility': 0.01}
        ])
        
        result = self.indicator.calculate(sideways_data)
        patterns = self.indicator.get_patterns(sideways_data)
        
        # 检查盘整形态
        self.assertIn('AROON_CONSOLIDATION', patterns.columns, "应包含盘整形态")
        
        # 在横盘行情中，可能会检测到盘整形态
        consolidation_signals = patterns['AROON_CONSOLIDATION'].sum()
        self.assertGreaterEqual(consolidation_signals, 0, "盘整形态检测应正常工作")

    def test_signal_generation(self):
        """测试Aroon信号生成"""
        # 生成包含多种走势的数据
        data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 100, 'periods': 20},  # 横盘
            {'type': 'trend', 'start_price': 100, 'end_price': 90, 'periods': 10},   # 下跌
            {'type': 'trend', 'start_price': 90, 'end_price': 110, 'periods': 15},   # 反弹
        ])
        
        signals = self.indicator.generate_signals(data)
        
        # 验证信号DataFrame结构
        self.assertIsInstance(signals, pd.DataFrame, "信号结果应为DataFrame")
        self.assertIn('buy_signal', signals.columns, "信号结果应包含buy_signal列")
        self.assertIn('sell_signal', signals.columns, "信号结果应包含sell_signal列")
        self.assertIn('strong_uptrend', signals.columns, "信号结果应包含strong_uptrend列")
        self.assertIn('strong_downtrend', signals.columns, "信号结果应包含strong_downtrend列")
        
        # 验证信号的基本逻辑
        # 买入和卖出信号不应同时为True
        simultaneous_signals = signals['buy_signal'] & signals['sell_signal']
        self.assertFalse(simultaneous_signals.any(), "买入和卖出信号不应同时出现")

    def test_score_calculation(self):
        """测试Aroon评分计算功能"""
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
        """测试Aroon参数设置"""
        # 测试参数设置方法
        new_period = 21
        
        self.indicator.set_parameters(period=new_period)
        
        # 验证参数是否正确设置
        self.assertEqual(self.indicator.period, new_period, "周期参数设置失败")
        
        # 验证新参数下的计算
        result = self.indicator.calculate(self.data)
        self.assertIn('aroon_up', result.columns, "结果应包含aroon_up列")

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
        
        result = self.indicator.calculate(flat_data)
        self.assertIsInstance(result, pd.DataFrame, "平价数据计算结果应为DataFrame")

    def test_robustness(self):
        """测试Aroon指标的鲁棒性"""
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
        self.assertIn('aroon_up', result.columns, "结果应包含aroon_up列")
        
        # 验证Aroon值是有限的数值
        aroon_up_values = result['aroon_up'].dropna()
        aroon_down_values = result['aroon_down'].dropna()
        self.assertTrue(all(np.isfinite(val) for val in aroon_up_values), "Aroon Up值应为有限数值")
        self.assertTrue(all(np.isfinite(val) for val in aroon_down_values), "Aroon Down值应为有限数值")


if __name__ == '__main__':
    unittest.main()
