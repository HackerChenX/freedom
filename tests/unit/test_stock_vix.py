"""
StockVIX指标单元测试
"""
import unittest
import pandas as pd
import numpy as np
from indicators.stock_vix import StockVIX
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator
from tests.helper.log_capture import LogCaptureMixin


class TestStockVIX(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    """StockVIX指标测试类"""
    
    def setUp(self):
        """设置测试环境"""
        # 显式调用LogCaptureMixin的setUp
        LogCaptureMixin.setUp(self)
        
        self.indicator = StockVIX()
        self.expected_columns = [
            'returns_volatility', 'parkinson_volatility', 'garman_klass_volatility',
            'ewma_volatility', 'garch_volatility', 'atr_volatility', 'stock_vix',
            'volatility_zone', 'volatility_trend', 'predicted_volatility', 'volatility_anomaly'
        ]
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 80}
        ])
    
    def tearDown(self):
        """清理日志捕获器"""
        LogCaptureMixin.tearDown(self)
    
    def test_stock_vix_initialization(self):
        """测试StockVIX初始化"""
        # 测试默认初始化
        default_indicator = StockVIX()
        self.assertEqual(default_indicator._parameters['window'], 22)
        self.assertEqual(default_indicator._parameters['alpha'], 0.94)
        
        # 测试自定义初始化
        custom_params = {'window': 30, 'alpha': 0.9}
        custom_indicator = StockVIX(params=custom_params)
        self.assertEqual(custom_indicator._parameters['window'], 30)
        self.assertEqual(custom_indicator._parameters['alpha'], 0.9)
    
    def test_stock_vix_calculation_accuracy(self):
        """测试StockVIX计算准确性"""
        result = self.indicator.calculate(self.data)
        
        # 验证StockVIX列存在
        for col in self.expected_columns:
            self.assertIn(col, result.columns, f"缺少列: {col}")
        
        # 验证VIX值的合理性
        if 'stock_vix' in result.columns:
            vix_values = result['stock_vix'].dropna()
            
            if len(vix_values) > 0:
                # VIX值应该是正数
                self.assertTrue(all(v > 0 for v in vix_values), "VIX值应该是正数")
                # VIX值应该是有限数值
                self.assertTrue(all(np.isfinite(v) for v in vix_values), "VIX值应该是有限数值")
    
    def test_stock_vix_score_range(self):
        """测试StockVIX评分范围"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        
        # 验证评分在0-100范围内
        valid_scores = raw_score.dropna()
        self.assertTrue(all(0 <= s <= 100 for s in valid_scores), "评分应在0-100范围内")
    
    def test_stock_vix_confidence_calculation(self):
        """测试StockVIX置信度计算"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, patterns, {})
        
        # 验证置信度在0-1范围内
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_stock_vix_parameter_update(self):
        """测试StockVIX参数更新"""
        new_params = {
            'window': 30,
            'alpha': 0.9,
            'normal_periods': 200
        }
        self.indicator.set_parameters(**new_params)
        
        # 验证参数更新
        self.assertEqual(self.indicator._parameters['window'], 30)
        self.assertEqual(self.indicator._parameters['alpha'], 0.9)
        self.assertEqual(self.indicator._parameters['normal_periods'], 200)
    
    def test_stock_vix_required_columns(self):
        """测试StockVIX必需列"""
        self.assertTrue(hasattr(self.indicator, 'REQUIRED_COLUMNS'))
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in expected_columns:
            self.assertIn(col, self.indicator.REQUIRED_COLUMNS)
    
    def test_stock_vix_patterns(self):
        """测试StockVIX形态识别"""
        # 先计算指标
        result = self.indicator.calculate(self.data)
        self.assertIsInstance(result, pd.DataFrame)
        
        # 然后获取形态
        patterns = self.indicator.get_patterns(self.data)
        
        # 验证返回DataFrame
        self.assertIsInstance(patterns, pd.DataFrame)
        
        # 验证基本的形态列存在
        if not patterns.empty and len(patterns.columns) > 0:
            expected_patterns = [
                'VIX_VERY_LOW_VOLATILITY', 'VIX_LOW_VOLATILITY', 'VIX_NORMAL_VOLATILITY',
                'VIX_HIGH_VOLATILITY', 'VIX_VERY_HIGH_VOLATILITY'
            ]
            
            for pattern in expected_patterns:
                self.assertIn(pattern, patterns.columns, f"缺少形态列: {pattern}")
    
    def test_stock_vix_signals(self):
        """测试StockVIX信号生成"""
        signals = self.indicator.generate_trading_signals(self.data)
        
        # 验证信号DataFrame结构
        self.assertIsInstance(signals, dict)
        expected_signal_keys = ['buy_signal', 'sell_signal', 'signal_strength']
        for key in expected_signal_keys:
            self.assertIn(key, signals, f"缺少信号键: {key}")
            self.assertIsInstance(signals[key], pd.Series)
    
    def test_stock_vix_volatility_calculations(self):
        """测试StockVIX各种波动率计算"""
        result = self.indicator.calculate(self.data)
        
        # 验证各种波动率指标
        volatility_indicators = [
            'returns_volatility', 'parkinson_volatility', 'garman_klass_volatility',
            'ewma_volatility', 'garch_volatility'
        ]
        
        for vol_indicator in volatility_indicators:
            if vol_indicator in result.columns:
                vol_values = result[vol_indicator].dropna()
                if len(vol_values) > 0:
                    # 波动率应该是正数
                    self.assertTrue(all(v >= 0 for v in vol_values), 
                                   f"{vol_indicator}应该是非负数")
                    # 波动率应该是有限数值
                    self.assertTrue(all(np.isfinite(v) for v in vol_values), 
                                   f"{vol_indicator}应该是有限数值")
    
    def test_stock_vix_zone_classification(self):
        """测试StockVIX波动区域分类"""
        result = self.indicator.calculate(self.data)
        
        # 验证波动区域列存在
        self.assertIn('volatility_zone', result.columns)
        
        # 验证波动区域值的合理性
        zone_values = result['volatility_zone'].dropna()
        if len(zone_values) > 0:
            valid_zones = ['极低波动', '低波动', '正常波动', '高波动', '极高波动']
            for zone in zone_values.unique():
                self.assertIn(zone, valid_zones, f"无效的波动区域: {zone}")
    
    def test_stock_vix_trend_calculation(self):
        """测试StockVIX趋势计算"""
        result = self.indicator.calculate(self.data)
        
        # 验证趋势列存在
        self.assertIn('volatility_trend', result.columns)
        
        # 验证趋势值的合理性
        trend_values = result['volatility_trend'].dropna()
        if len(trend_values) > 0:
            valid_trends = [-1, 0, 1]
            for trend in trend_values.unique():
                self.assertIn(trend, valid_trends, f"无效的趋势值: {trend}")
    
    def test_stock_vix_anomaly_detection(self):
        """测试StockVIX异常检测"""
        result = self.indicator.calculate(self.data)
        
        # 验证异常检测列存在
        self.assertIn('volatility_anomaly', result.columns)
        
        # 验证异常值的合理性
        anomaly_values = result['volatility_anomaly'].dropna()
        if len(anomaly_values) > 0:
            valid_anomalies = [-1, 0, 1]
            for anomaly in anomaly_values.unique():
                self.assertIn(anomaly, valid_anomalies, f"无效的异常值: {anomaly}")
    
    def test_stock_vix_prediction(self):
        """测试StockVIX波动率预测"""
        result = self.indicator.calculate(self.data)
        
        # 验证预测列存在
        self.assertIn('predicted_volatility', result.columns)
        
        # 验证预测值的合理性
        prediction_values = result['predicted_volatility'].dropna()
        if len(prediction_values) > 0:
            # 预测值应该是有限数值
            self.assertTrue(all(np.isfinite(v) for v in prediction_values), 
                           "预测波动率应该是有限数值")
    
    def test_stock_vix_percentile_calculation(self):
        """测试StockVIX百分位计算"""
        # 需要足够的数据进行百分位计算
        long_data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 300}
        ])
        
        result = self.indicator.calculate(long_data)
        
        # 验证百分位列存在
        if 'vix_percentile' in result.columns:
            percentile_values = result['vix_percentile'].dropna()
            if len(percentile_values) > 0:
                # 百分位应该在0-100范围内
                self.assertTrue(all(0 <= v <= 100 for v in percentile_values), 
                               "VIX百分位应该在0-100范围内")
    
    def test_stock_vix_strength_calculation(self):
        """测试StockVIX强度计算"""
        # 需要足够的数据进行强度计算
        long_data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 300}
        ])
        
        result = self.indicator.calculate(long_data)
        
        # 验证强度列存在
        if 'volatility_strength' in result.columns:
            strength_values = result['volatility_strength'].dropna()
            if len(strength_values) > 0:
                # 强度应该在0-100范围内
                self.assertTrue(all(0 <= v <= 100 for v in strength_values), 
                               "波动率强度应该在0-100范围内")
    
    def test_stock_vix_atr_volatility(self):
        """测试StockVIX ATR波动率"""
        result = self.indicator.calculate(self.data)
        
        # 验证ATR波动率列存在
        if 'atr_volatility' in result.columns:
            atr_vol_values = result['atr_volatility'].dropna()
            if len(atr_vol_values) > 0:
                # ATR波动率应该是正数
                self.assertTrue(all(v >= 0 for v in atr_vol_values), 
                               "ATR波动率应该是非负数")
    
    def test_stock_vix_comprehensive_patterns(self):
        """测试StockVIX综合形态"""
        # 先计算指标
        result = self.indicator.calculate(self.data)
        
        # 获取形态
        patterns = self.indicator.get_patterns(self.data)
        
        # 验证综合形态存在
        if not patterns.empty:
            comprehensive_patterns = [
                'VIX_UPTREND', 'VIX_DOWNTREND', 'VIX_ANOMALY_SPIKE',
                'VIX_ANOMALY_DROP', 'VIX_RISING', 'VIX_FALLING'
            ]
            
            for pattern in comprehensive_patterns:
                if pattern in patterns.columns:
                    # 形态值应该是布尔值
                    pattern_values = patterns[pattern].dropna()
                    if len(pattern_values) > 0:
                        unique_values = pattern_values.unique()
                        for val in unique_values:
                            self.assertIsInstance(val, (bool, np.bool_), f"{pattern}应该是布尔值")
    
    def test_no_errors_during_calculation(self):
        """测试计算过程中无ERROR日志"""
        self.clear_logs()
        
        # 执行计算
        result = self.indicator.calculate(self.data)
        
        # 验证无ERROR日志
        self.assert_no_logs('ERROR')
        
        # 验证结果
        self.assertIsInstance(result, pd.DataFrame)
        for col in self.expected_columns:
            self.assertIn(col, result.columns)
    
    def test_no_errors_during_pattern_detection(self):
        """测试形态检测过程中无ERROR日志"""
        self.clear_logs()
        
        # 执行形态检测
        patterns = self.indicator.get_patterns(self.data)
        
        # 验证无ERROR日志
        self.assert_no_logs('ERROR')
        
        # 验证结果
        self.assertIsInstance(patterns, pd.DataFrame)
    
    def test_stock_vix_register_patterns(self):
        """测试StockVIX形态注册"""
        # 调用形态注册
        self.indicator.register_patterns()
        
        # 验证形态已注册（通过检查是否有异常抛出）
        self.assertTrue(True, "形态注册应该成功完成")
    
    def test_stock_vix_edge_cases(self):
        """测试StockVIX边界情况"""
        # 测试数据不足的情况
        small_data = self.data.head(10)
        result = self.indicator.calculate(small_data)
        
        # StockVIX应该能处理数据不足的情况
        self.assertIsInstance(result, pd.DataFrame)
        for col in self.expected_columns:
            self.assertIn(col, result.columns)
    
    def test_stock_vix_validation(self):
        """测试StockVIX数据验证"""
        # 测试缺少必需列的情况
        invalid_data = self.data.drop(['high'], axis=1)

        # BaseIndicator会处理缺失列并返回空DataFrame
        result = self.indicator.calculate(invalid_data)
        self.assertIsInstance(result, pd.DataFrame)
    
    def test_stock_vix_indicator_type(self):
        """测试StockVIX指标类型"""
        indicator_type = self.indicator.get_indicator_type()
        self.assertEqual(indicator_type, "STOCKVIX", "指标类型应该是STOCKVIX")


if __name__ == '__main__':
    unittest.main()
