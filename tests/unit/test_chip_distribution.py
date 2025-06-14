"""
ChipDistribution指标单元测试
"""
import unittest
import pandas as pd
import numpy as np
from indicators.chip_distribution import ChipDistribution
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator
from tests.helper.log_capture import LogCaptureMixin


class TestChipDistribution(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    """ChipDistribution指标测试类"""
    
    def setUp(self):
        """设置测试环境"""
        # 显式调用LogCaptureMixin的setUp
        LogCaptureMixin.setUp(self)
        
        self.indicator = ChipDistribution(periods=[5, 10, 20, 60, 120])
        self.expected_columns = [
            'avg_cost', 'chip_concentration', 'profit_ratio', 'chip_width_90pct',
            'untrapped_difficulty', 'chip_looseness', 'profit_ratio_change', 'cost_deviation'
        ]
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 80}
        ])
    
    def tearDown(self):
        """清理日志捕获器"""
        LogCaptureMixin.tearDown(self)
    
    def test_chip_distribution_initialization(self):
        """测试ChipDistribution初始化"""
        # 测试默认初始化
        default_indicator = ChipDistribution()
        self.assertEqual(default_indicator._parameters['half_life'], 60)
        self.assertEqual(default_indicator._parameters['price_precision'], 0.01)
        self.assertEqual(default_indicator.periods, [5, 10, 20, 60, 120])

        # 测试自定义初始化
        custom_indicator = ChipDistribution(periods=[10, 20, 30])
        self.assertEqual(custom_indicator.periods, [10, 20, 30])
    
    def test_chip_distribution_calculation_accuracy(self):
        """测试ChipDistribution计算准确性"""
        result = self.indicator.calculate(self.data)
        
        # 验证ChipDistribution列存在
        for col in self.expected_columns:
            self.assertIn(col, result.columns, f"缺少列: {col}")
        
        # 验证筹码分布值的合理性
        for col in ['chip_concentration', 'profit_ratio']:
            if col in result.columns:
                values = result[col].dropna()
                
                if len(values) > 0:
                    # 这些值应该在0-1范围内
                    self.assertTrue(all(0 <= v <= 1 for v in values), f"{col}值应该在0-1范围内")
    
    def test_chip_distribution_score_range(self):
        """测试ChipDistribution评分范围"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        
        # 验证评分在0-100范围内
        valid_scores = raw_score.dropna()
        self.assertTrue(all(0 <= s <= 100 for s in valid_scores), "评分应在0-100范围内")
    
    def test_chip_distribution_confidence_calculation(self):
        """测试ChipDistribution置信度计算"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, patterns, {})
        
        # 验证置信度在0-1范围内
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_chip_distribution_parameter_update(self):
        """测试ChipDistribution参数更新"""
        new_periods = [15, 30, 60]
        self.indicator.set_parameters(periods=new_periods)

        # 验证参数更新
        self.assertEqual(self.indicator.periods, new_periods)
    
    def test_chip_distribution_required_columns(self):
        """测试ChipDistribution必需列"""
        self.assertTrue(hasattr(self.indicator, 'REQUIRED_COLUMNS'))
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in expected_columns:
            self.assertIn(col, self.indicator.REQUIRED_COLUMNS)
    
    def test_chip_distribution_patterns(self):
        """测试ChipDistribution形态识别"""
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
                'CHIP_HIGH_CONCENTRATION', 'CHIP_DISPERSED',
                'CHIP_HIGH_PROFIT', 'CHIP_LOW_PROFIT',
                'PRICE_FAR_ABOVE_COST', 'PRICE_FAR_BELOW_COST', 'PRICE_NEAR_COST'
            ]
            
            for pattern in expected_patterns:
                self.assertIn(pattern, patterns.columns, f"缺少形态列: {pattern}")
    
    def test_chip_distribution_signals(self):
        """测试ChipDistribution信号生成"""
        signals = self.indicator.generate_trading_signals(self.data)
        
        # 验证信号DataFrame结构
        self.assertIsInstance(signals, dict)
        expected_signal_keys = ['buy_signal', 'sell_signal', 'signal_strength']
        for key in expected_signal_keys:
            self.assertIn(key, signals, f"缺少信号键: {key}")
            self.assertIsInstance(signals[key], pd.Series)
    
    def test_chip_distribution_concentration_calculation(self):
        """测试ChipDistribution集中度计算"""
        result = self.indicator.calculate(self.data)
        
        # 验证集中度列存在
        self.assertIn('chip_concentration', result.columns)
        
        # 验证集中度值的合理性
        concentration_values = result['chip_concentration'].dropna()
        if len(concentration_values) > 0:
            # 集中度应该在0-1范围内
            self.assertTrue(all(0 <= v <= 1 for v in concentration_values), 
                           "筹码集中度应该在0-1范围内")
    
    def test_chip_distribution_profit_ratio_calculation(self):
        """测试ChipDistribution获利盘比例计算"""
        result = self.indicator.calculate(self.data)
        
        # 验证获利盘比例列存在
        self.assertIn('profit_ratio', result.columns)
        
        # 验证获利盘比例值的合理性
        profit_values = result['profit_ratio'].dropna()
        if len(profit_values) > 0:
            # 获利盘比例应该在0-1范围内
            self.assertTrue(all(0 <= v <= 1 for v in profit_values), 
                           "获利盘比例应该在0-1范围内")
    
    def test_chip_distribution_avg_cost_calculation(self):
        """测试ChipDistribution平均成本计算"""
        result = self.indicator.calculate(self.data)
        
        # 验证平均成本列存在
        self.assertIn('avg_cost', result.columns)
        
        # 验证平均成本值的合理性
        cost_values = result['avg_cost'].dropna()
        if len(cost_values) > 0:
            # 平均成本应该是正数
            self.assertTrue(all(v > 0 for v in cost_values), "平均成本应该是正数")
    
    def test_chip_distribution_untrapped_difficulty(self):
        """测试ChipDistribution解套难度计算"""
        result = self.indicator.calculate(self.data)
        
        # 验证解套难度列存在
        self.assertIn('untrapped_difficulty', result.columns)
        
        # 验证解套难度值的合理性
        difficulty_values = result['untrapped_difficulty'].dropna()
        if len(difficulty_values) > 0:
            # 解套难度应该是正数
            self.assertTrue(all(v > 0 for v in difficulty_values), "解套难度应该是正数")
    
    def test_chip_distribution_chip_width(self):
        """测试ChipDistribution筹码宽度计算"""
        result = self.indicator.calculate(self.data)
        
        # 验证筹码宽度列存在
        self.assertIn('chip_width_90pct', result.columns)
        
        # 验证筹码宽度值的合理性
        width_values = result['chip_width_90pct'].dropna()
        if len(width_values) > 0:
            # 筹码宽度应该是正数
            self.assertTrue(all(v >= 0 for v in width_values), "筹码宽度应该是非负数")
    
    def test_chip_distribution_comprehensive_patterns(self):
        """测试ChipDistribution综合形态"""
        # 先计算指标
        result = self.indicator.calculate(self.data)
        
        # 获取形态
        patterns = self.indicator.get_patterns(self.data)
        
        # 验证综合形态存在
        if not patterns.empty:
            comprehensive_patterns = [
                'CHIP_BOTTOM_ACCUMULATION', 'CHIP_TOP_DISTRIBUTION', 'CHIP_MAIN_WAVE'
            ]
            
            for pattern in comprehensive_patterns:
                if pattern in patterns.columns:
                    # 形态值应该是布尔值
                    pattern_values = patterns[pattern].dropna()
                    if len(pattern_values) > 0:
                        unique_values = pattern_values.unique()
                        for val in unique_values:
                            self.assertIsInstance(val, (bool, np.bool_), f"{pattern}应该是布尔值")
    
    def test_chip_distribution_with_turnover_rate(self):
        """测试ChipDistribution带换手率数据的计算"""
        # 添加换手率数据
        data_with_turnover = self.data.copy()
        data_with_turnover['turnover_rate'] = np.random.uniform(0.5, 5.0, len(self.data))
        
        result = self.indicator.calculate(data_with_turnover)
        
        # 验证计算结果
        self.assertIsInstance(result, pd.DataFrame)
        for col in self.expected_columns:
            self.assertIn(col, result.columns)
    
    def test_chip_distribution_market_environment(self):
        """测试ChipDistribution市场环境设置"""
        # 测试设置不同的市场环境
        environments = ['bull_market', 'bear_market', 'sideways_market', 'volatile_market', 'normal']
        
        for env in environments:
            self.indicator.set_market_environment(env)
            self.assertEqual(self.indicator.market_environment, env)
        
        # 测试无效环境
        with self.assertRaises(ValueError):
            self.indicator.set_market_environment('invalid_environment')
    
    def test_chip_distribution_cost_deviation(self):
        """测试ChipDistribution成本偏离度计算"""
        result = self.indicator.calculate(self.data)
        
        # 验证成本偏离度列存在
        self.assertIn('cost_deviation', result.columns)
        
        # 验证成本偏离度值的合理性
        deviation_values = result['cost_deviation'].dropna()
        if len(deviation_values) > 0:
            # 成本偏离度应该是有限数值
            self.assertTrue(all(np.isfinite(v) for v in deviation_values), 
                           "成本偏离度应该是有限数值")
    
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
    
    def test_chip_distribution_register_patterns(self):
        """测试ChipDistribution形态注册"""
        # 调用形态注册
        self.indicator.register_patterns()
        
        # 验证形态已注册（通过检查是否有异常抛出）
        self.assertTrue(True, "形态注册应该成功完成")
    
    def test_chip_distribution_edge_cases(self):
        """测试ChipDistribution边界情况"""
        # 测试数据不足的情况
        small_data = self.data.head(10)
        result = self.indicator.calculate(small_data)
        
        # ChipDistribution应该能处理数据不足的情况
        self.assertIsInstance(result, pd.DataFrame)
        for col in self.expected_columns:
            self.assertIn(col, result.columns)
    
    def test_chip_distribution_validation(self):
        """测试ChipDistribution数据验证"""
        # 测试缺少必需列的情况
        invalid_data = self.data.drop(['volume'], axis=1)
        
        # BaseIndicator会处理缺失列并返回空DataFrame
        result = self.indicator.calculate(invalid_data)
        self.assertIsInstance(result, pd.DataFrame)
    
    def test_chip_distribution_indicator_type(self):
        """测试ChipDistribution指标类型"""
        indicator_type = self.indicator.get_indicator_type()
        self.assertEqual(indicator_type, "CHIPDISTRIBUTION", "指标类型应该是CHIPDISTRIBUTION")


if __name__ == '__main__':
    unittest.main()
