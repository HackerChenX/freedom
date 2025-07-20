"""
高级理论与分析指标单元测试
"""
import unittest
import pytest
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator
from indicators.complete_indicator_registry import complete_registry

class TestChipDistribution_Indicators_Test_Advanced_Analysis_Indicators(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        super().setUp()
        self.indicator = complete_registry.create_indicator('ZXM_CHIP_DISTRIBUTION')
        self.expected_columns = ['chip_concentration', 'profit_ratio', 'chip_width_90pct', 'avg_cost']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'sideways', 'start_price': 100, 'volatility': 0.02, 'periods': 200}
        ])

class TestElliottWave_Indicators_Test_Advanced_Analysis_Indicators(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        super().setUp()
        self.indicator = complete_registry.create_indicator('ELLIOTT_WAVE')
        self.expected_columns = ['wave_number', 'wave_label', 'wave_direction', 'wave_pattern']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 50},
            {'type': 'trend', 'start_price': 120, 'end_price': 110, 'periods': 30},
            {'type': 'trend', 'start_price': 110, 'end_price': 130, 'periods': 50}
        ])

class TestFibonacci(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        super().setUp()
        self.indicator = complete_registry.create_indicator('FIBONACCI')
        self.expected_columns = ['fib_ret_0.236', 'fib_ret_0.382', 'fib_ret_0.5', 'fib_ret_0.618', 'fib_ret_0.786']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 150, 'periods': 100}
        ])

class TestFibonacciTools_Indicators_Test_Advanced_Analysis_Indicators(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        super().setUp()
        self.indicator = complete_registry.create_indicator('FIBONACCI')
        self.expected_columns = ['swing_high', 'swing_low', 'fib_236', 'fib_382', 'fib_500', 'fib_618', 'fib_786']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'v_shape', 'start_price': 120, 'bottom_price': 100, 'periods': 100}
        ])

class TestGannTools_Indicators_Test_Advanced_Analysis_Indicators(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        super().setUp()
        self.indicator = complete_registry.create_indicator('GANN')
        self.expected_columns = ['gann_1x1', 'gann_2x1', 'time_cycle']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 100}
        ])

class TestInstitutionalBehavior_Indicators_Test_Advanced_Analysis_Indicators(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        super().setUp()
        self.indicator = complete_registry.create_indicator('ZXM_INSTITUTION_BEHAVIOR')
        self.expected_columns = ['inst_cost', 'inst_profit_ratio', 'inst_concentration']
        # This may require tick-level data, which TestDataGenerator should simulate
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'sideways', 'start_price': 100, 'volatility': 0.02, 'periods': 100}
        ])

class TestMultiPeriodResonance(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        super().setUp()
        self.indicator = complete_registry.create_indicator('COMPOSITE')
        self.expected_columns = ['resonance_signal', 'resonance_level']
        # This indicator needs multi-period data, which is a complex setup
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 200}
        ])

class TestSentimentAnalysis_Indicators(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        super().setUp()
        self.indicator = complete_registry.create_indicator('ZXM_MARKET_SENTIMENT')
        self.expected_columns = ['sentiment_index', 'sentiment_category']
        # This requires external data (e.g., news), so we test if it runs with price data
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'sideways', 'start_price': 100, 'volatility': 0.02, 'periods': 100}
        ])

class TestTimeCycleAnalysis(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        super().setUp()
        self.indicator = complete_registry.create_indicator('ZXM_CYCLE_POSITION', max_cycle_days=50)
        self.expected_columns = ['cycle_position', 'potential_turning_point']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'v_shape', 'start_price': 100, 'bottom_price': 90, 'periods': 100},
            {'type': 'v_shape', 'start_price': 90, 'bottom_price': 80, 'periods': 100}
        ])

class TestTrendClassification_Indicators(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        super().setUp()
        self.indicator = complete_registry.create_indicator('COMPOSITE')
        self.expected_columns = ['trend_type', 'trend_strength']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 50},
            {'type': 'sideways', 'start_price': 120, 'volatility': 0.02, 'periods': 50}
        ])

class TestTrendStrength_Indicators(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        super().setUp()
        self.indicator = complete_registry.create_indicator('COMPOSITE')
        self.expected_columns = ['trend_strength', 'trend_direction', 'trend_category']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 105, 'periods': 50},
            {'type': 'trend', 'start_price': 105, 'end_price': 125, 'periods': 50}
        ])