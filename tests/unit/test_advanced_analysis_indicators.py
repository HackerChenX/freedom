"""
高级理论与分析指标单元测试
"""
import unittest
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator
from indicators.factory import IndicatorFactory

class TestChipDistribution(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        self.indicator = IndicatorFactory.create_indicator('CHIPDISTRIBUTION')
        self.expected_columns = ['cost_price', 'concentration_ratio']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'sideways', 'price': 100, 'periods': 200}
        ])

class TestElliottWave(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        self.indicator = IndicatorFactory.create_indicator('ELLIOTTWAVE')
        self.expected_columns = ['wave_pattern', 'wave_count']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 50},
            {'type': 'trend', 'start_price': 120, 'end_price': 110, 'periods': 30},
            {'type': 'trend', 'start_price': 110, 'end_price': 130, 'periods': 50}
        ])

class TestFibonacci(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        self.indicator = IndicatorFactory.create_indicator('FIBONACCI')
        self.expected_columns = ['fib_level_0.236', 'fib_level_0.382', 'fib_level_0.618']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 150, 'periods': 100}
        ])

class TestFibonacciTools(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        self.indicator = IndicatorFactory.create_indicator('FIBONACCITOOLS')
        self.expected_columns = ['fib_fan', 'fib_arc']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'v_shape', 'start_price': 120, 'bottom_price': 100, 'periods': 100}
        ])

class TestGannTools(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        self.indicator = IndicatorFactory.create_indicator('GANNTOOLS')
        self.expected_columns = ['gann_fan', 'gann_angle']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 100}
        ])

class TestInstitutionalBehavior(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        self.indicator = IndicatorFactory.create_indicator('INSTITUTIONALBEHAVIOR')
        self.expected_columns = ['large_order_inflow', 'net_inflow_ratio']
        # This may require tick-level data, which TestDataGenerator should simulate
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'sideways', 'price': 100, 'periods': 100}
        ])

class TestMultiPeriodResonance(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        self.indicator = IndicatorFactory.create_indicator('MULTIPERIODRESONANCE')
        self.expected_columns = ['resonance_signal', 'resonance_strength']
        # This indicator needs multi-period data, which is a complex setup
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 200}
        ])

class TestSentimentAnalysis(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        self.indicator = IndicatorFactory.create_indicator('SENTIMENTANALYSIS')
        self.expected_columns = ['sentiment_score', 'news_impact']
        # This requires external data (e.g., news), so we test if it runs with price data
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'sideways', 'price': 100, 'periods': 100}
        ])

class TestTimeCycleAnalysis(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        self.indicator = IndicatorFactory.create_indicator('TIMECYCLEANALYSIS')
        self.expected_columns = ['dominant_cycle', 'cycle_phase']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'v_shape', 'start_price': 100, 'bottom_price': 90, 'periods': 100},
            {'type': 'v_shape', 'start_price': 90, 'bottom_price': 80, 'periods': 100}
        ])

class TestTrendClassification(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        self.indicator = IndicatorFactory.create_indicator('TRENDCLASSIFICATION')
        self.expected_columns = ['trend_type', 'trend_start_date']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 50},
            {'type': 'sideways', 'price': 120, 'periods': 50}
        ])

class TestTrendStrength(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        self.indicator = IndicatorFactory.create_indicator('TRENDSTRENGTH')
        self.expected_columns = ['trend_strength_score', 'is_trending']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 105, 'periods': 50},
            {'type': 'trend', 'start_price': 105, 'end_price': 125, 'periods': 50}
        ])

if __name__ == '__main__':
    unittest.main() 