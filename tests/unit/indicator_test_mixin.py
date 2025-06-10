import pandas as pd

class IndicatorTestMixin:
    """
    一个包含通用指标测试逻辑的Mixin类。
    这个类不继承自 unittest.TestCase，因此不会被测试加载器发现。
    具体的测试类将继承这个Mixin和unittest.TestCase。
    """
    def test_calculation_runs_without_error(self):
        """测试指标的 calculate 方法是否能无错误地运行。"""
        self.assertIsNotNone(getattr(self, 'indicator', None), "self.indicator must be set in setUp")
        self.assertIsNotNone(getattr(self, 'data', None), "self.data must be set in setUp")
        try:
            _ = self.indicator.calculate(self.data)
        except Exception as e:
            self.fail(f"Indicator calculate() raised an exception unexpectedly: {e}")

    def test_returns_dataframe_with_expected_columns(self):
        """测试 calculate 方法是否返回包含预期列的 DataFrame。"""
        self.assertIsNotNone(getattr(self, 'indicator', None), "self.indicator must be set in setUp")
        self.assertIsNotNone(getattr(self, 'data', None), "self.data must be set in setUp")
        result = self.indicator.calculate(self.data)
        self.assertIsInstance(result, pd.DataFrame, "calculate() should return a pandas DataFrame.")
        if hasattr(self, 'expected_columns'):
            self.assertTrue(all(col in result.columns for col in self.expected_columns), 
                            "Resulting DataFrame is missing one or more expected columns.")

    def test_handles_insufficient_data(self):
        """测试指标在数据不足时是否能优雅地处理。"""
        self.assertIsNotNone(getattr(self, 'indicator', None), "self.indicator must be set in setUp")
        self.assertIsNotNone(getattr(self, 'data', None), "self.data must be set in setUp")
        insufficient_data = self.data.head(5)
        try:
            result = self.indicator.calculate(insufficient_data)
            self.assertIsInstance(result, pd.DataFrame)
        except Exception as e:
            self.fail(f"Indicator failed to handle insufficient data gracefully: {e}")

    def test_pattern_detection_returns_dataframe(self):
        """测试 get_patterns 方法返回一个DataFrame（如果存在）。"""
        self.assertIsNotNone(getattr(self, 'indicator', None), "self.indicator must be set in setUp")
        self.assertIsNotNone(getattr(self, 'data', None), "self.data must be set in setUp")
        if not hasattr(self.indicator, 'get_patterns'):
            self.skipTest("Indicator does not have a 'get_patterns' method.")
        
        patterns = self.indicator.get_patterns(self.data)
        self.assertIsInstance(patterns, pd.DataFrame, "get_patterns() should return a pandas DataFrame.")
        if not patterns.empty:
            self.assertTrue(all(dtype == 'bool' for dtype in patterns.dtypes),
                            "All columns in the patterns DataFrame should be boolean.") 