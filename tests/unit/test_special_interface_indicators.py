"""
特殊接口指标测试模块

测试那些需要特殊接口或参数传递的指标
"""

import unittest
import pandas as pd
import numpy as np
from indicators.factory import IndicatorFactory
from tests.helper.data_generator import TestDataGenerator
from tests.unit.indicator_test_mixin import IndicatorTestMixin


class TestSpecialInterfaceIndicators(unittest.TestCase, IndicatorTestMixin):
    """特殊接口指标测试类"""

    def setUp(self):
        """准备测试数据和环境"""
        # 生成测试数据
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 50},  # 上涨趋势
            {'type': 'trend', 'start_price': 120, 'end_price': 90, 'periods': 50},   # 下跌趋势
            {'type': 'v_shape', 'start_price': 90, 'bottom_price': 80, 'periods': 50},  # V形反转
            {'type': 'sideways', 'start_price': 100, 'volatility': 0.02, 'periods': 50}  # 横盘整理
        ])
        # 确保数据包含所有必要的列
        self._ensure_stock_info_fields(self.data)
        
        # 自动注册所有指标
        IndicatorFactory.auto_register_all_indicators()
        self.supported_indicators = IndicatorFactory.get_supported_indicators()

    def test_zxm_pattern_indicator(self):
        """测试ZXM形态指标"""
        # 创建指标
        zxm_pattern = IndicatorFactory.create_indicator('ZXMPATTERNINDICATOR')
        self.assertIsNotNone(zxm_pattern, "ZXMPATTERNINDICATOR创建失败")
        
        # 从DataFrame中提取所需数据
        high_prices = self.data['high'].values
        low_prices = self.data['low'].values
        close_prices = self.data['close'].values
        open_prices = self.data['open'].values
        volumes = self.data['volume'].values
        
        # 使用正确的参数调用calculate方法
        result = zxm_pattern.calculate(open_prices, high_prices, low_prices, close_prices, volumes)
        
        # 验证结果
        self.assertIsNotNone(result, "ZXM形态指标计算结果不应为None")
        self.assertIsInstance(result, dict, "ZXM形态指标计算结果应为字典")
        
        # 检查结果是否包含预期的键
        expected_keys = ['class_one_buy', 'class_two_buy', 'class_three_buy']
        for key in expected_keys:
            self.assertIn(key, result, f"结果中应包含{key}键")
            self.assertIsInstance(result[key], np.ndarray, f"{key}的值应为numpy数组")
        
        # 测试原始评分计算
        score = zxm_pattern.calculate_raw_score(self.data)
        self.assertIsInstance(score, pd.Series, "原始评分结果应为Series")
        self.assertTrue(all(0 <= s <= 100 for s in score if not pd.isna(s)), "评分应在0-100范围内")

    def test_kdj_condition(self):
        """测试KDJ条件指标"""
        # 创建KDJ指标并计算
        kdj = IndicatorFactory.create_indicator('KDJ')
        kdj_data = kdj.calculate(self.data.copy())
        
        # 确保KDJ数据包含所需的列
        self.assertIn('k', kdj_data.columns, "KDJ计算结果应包含k列")
        self.assertIn('d', kdj_data.columns, "KDJ计算结果应包含d列")
        
        # 创建KDJ条件指标
        kdj_condition = IndicatorFactory.create_indicator('KDJ_CONDITION')
        self.assertIsNotNone(kdj_condition, "KDJ_CONDITION创建失败")
        
        # 设置条件参数并计算
        kdj_condition.line = 'k'
        kdj_condition.operator = '>'
        kdj_condition.value = 20
        
        # 计算条件指标
        result = kdj_condition.calculate(kdj_data)
        
        # 验证结果
        self.assertIsNotNone(result, "KDJ条件指标计算结果不应为None")
        self.assertIn('condition_met', result.columns, "结果应包含condition_met列")
        
        # 测试条件逻辑
        # k > 20应为True，k <= 20应为False
        expected = (kdj_data['k'] > 20).astype(int)
        pd.testing.assert_series_equal(result['condition_met'], expected, "条件结果应匹配预期")
        
        # 测试原始评分计算
        score = kdj_condition.calculate_raw_score(result)
        self.assertIsInstance(score, pd.Series, "原始评分结果应为Series")
        self.assertTrue(all(0 <= s <= 100 for s in score if not pd.isna(s)), "评分应在0-100范围内")

    def test_macd_condition(self):
        """测试MACD条件指标"""
        # 创建MACD指标并计算
        macd = IndicatorFactory.create_indicator('MACD')
        macd_data = macd.calculate(self.data.copy())
        
        # 确保MACD数据包含所需的列
        required_cols = ['dif', 'dea', 'macd']
        for col in required_cols:
            self.assertIn(col, macd_data.columns, f"MACD计算结果应包含{col}列")
        
        # 创建MACD条件指标
        macd_condition = IndicatorFactory.create_indicator('MACD_CONDITION')
        self.assertIsNotNone(macd_condition, "MACD_CONDITION创建失败")
        
        # 设置条件参数并计算
        macd_condition.line = 'dif'
        macd_condition.operator = '>'
        macd_condition.value = 0
        
        # 计算条件指标
        result = macd_condition.calculate(macd_data)
        
        # 验证结果
        self.assertIsNotNone(result, "MACD条件指标计算结果不应为None")
        self.assertIn('condition_met', result.columns, "结果应包含condition_met列")
        
        # 测试条件逻辑
        # dif > 0应为True，dif <= 0应为False
        expected = (macd_data['dif'] > 0).astype(int)
        pd.testing.assert_series_equal(result['condition_met'], expected, "条件结果应匹配预期")
        
        # 测试原始评分计算
        score = macd_condition.calculate_raw_score(result)
        self.assertIsInstance(score, pd.Series, "原始评分结果应为Series")
        self.assertTrue(all(0 <= s <= 100 for s in score if not pd.isna(s)), "评分应在0-100范围内")

    def test_ma_condition(self):
        """测试MA条件指标"""
        # 创建MA条件指标
        ma_condition = IndicatorFactory.create_indicator('MA_CONDITION')
        self.assertIsNotNone(ma_condition, "MA_CONDITION创建失败")
        
        # 设置条件参数并计算
        ma_condition.ma_type = 'MA'
        ma_condition.ma_period = 5
        ma_condition.operator = '>'
        ma_condition.compare_value = 'CLOSE'
        
        # 计算条件指标
        result = ma_condition.calculate(self.data.copy())
        
        # 验证结果
        self.assertIsNotNone(result, "MA条件指标计算结果不应为None")
        self.assertIn('condition_met', result.columns, "结果应包含condition_met列")
        self.assertIn('MA5', result.columns, "结果应包含MA5列")
        
        # 测试条件逻辑
        # MA5 > close应为True，MA5 <= close应为False
        expected = (result['MA5'] > self.data['close']).astype(int)
        pd.testing.assert_series_equal(result['condition_met'], expected, "条件结果应匹配预期")
        
        # 测试原始评分计算
        score = ma_condition.calculate_raw_score(result)
        self.assertIsInstance(score, pd.Series, "原始评分结果应为Series")
        self.assertTrue(all(0 <= s <= 100 for s in score if not pd.isna(s)), "评分应在0-100范围内")

    def test_generic_condition(self):
        """测试通用条件指标"""
        # 创建通用条件指标
        generic_condition = IndicatorFactory.create_indicator('GENERIC_CONDITION')
        self.assertIsNotNone(generic_condition, "GENERIC_CONDITION创建失败")
        
        # 设置条件参数并计算
        generic_condition.condition = "MA5>MA10"
        
        # 计算条件指标
        result = generic_condition.calculate(self.data.copy())
        
        # 验证结果
        self.assertIsNotNone(result, "通用条件指标计算结果不应为None")
        self.assertIn('condition_met', result.columns, "结果应包含condition_met列")
        self.assertIn('MA5', result.columns, "结果应包含MA5列")
        self.assertIn('MA10', result.columns, "结果应包含MA10列")
        
        # 测试条件逻辑
        # MA5 > MA10应为True，MA5 <= MA10应为False
        expected = (result['MA5'] > result['MA10']).astype(int)
        pd.testing.assert_series_equal(result['condition_met'], expected, "条件结果应匹配预期")
        
        # 测试原始评分计算
        score = generic_condition.calculate_raw_score(result)
        self.assertIsInstance(score, pd.Series, "原始评分结果应为Series")
        self.assertTrue(all(0 <= s <= 100 for s in score if not pd.isna(s)), "评分应在0-100范围内")


if __name__ == '__main__':
    unittest.main() 