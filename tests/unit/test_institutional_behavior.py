"""
InstitutionalBehavior指标单元测试
"""
import unittest
import pandas as pd
import numpy as np
from indicators.institutional_behavior import InstitutionalBehavior
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator
from tests.helper.log_capture import LogCaptureMixin


class TestInstitutionalBehavior(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    """InstitutionalBehavior指标测试类"""
    
    def setUp(self):
        """设置测试环境"""
        # 显式调用LogCaptureMixin的setUp
        LogCaptureMixin.setUp(self)
        
        self.indicator = InstitutionalBehavior()
        self.expected_columns = [
            'inst_concentration', 'inst_profit_ratio', 'inst_cost', 'inst_activity_score',
            'inst_phase', 'behavior_pattern', 'phase_change', 'behavior_intensity', 'behavior_description'
        ]
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 80}
        ])
    
    def tearDown(self):
        """清理日志捕获器"""
        LogCaptureMixin.tearDown(self)
    
    def test_institutional_behavior_initialization(self):
        """测试InstitutionalBehavior初始化"""
        # 测试默认初始化
        default_indicator = InstitutionalBehavior()
        self.assertEqual(default_indicator.volume_quantile, 0.85)
        
        # 测试参数设置
        default_indicator.set_parameters(volume_quantile=0.9)
        self.assertEqual(default_indicator.volume_quantile, 0.9)
    
    def test_institutional_behavior_calculation_accuracy(self):
        """测试InstitutionalBehavior计算准确性"""
        result = self.indicator.calculate(self.data)
        
        # 验证InstitutionalBehavior列存在
        for col in self.expected_columns:
            self.assertIn(col, result.columns, f"缺少列: {col}")
        
        # 验证机构指标值的合理性
        if 'inst_concentration' in result.columns:
            concentration_values = result['inst_concentration'].dropna()
            if len(concentration_values) > 0:
                # 集中度应该在0-1范围内
                self.assertTrue(all(0 <= v <= 1 for v in concentration_values), 
                               "机构集中度应该在0-1范围内")
        
        if 'inst_profit_ratio' in result.columns:
            profit_values = result['inst_profit_ratio'].dropna()
            if len(profit_values) > 0:
                # 获利盘比例应该在0-1范围内
                self.assertTrue(all(0 <= v <= 1 for v in profit_values), 
                               "机构获利盘比例应该在0-1范围内")
    
    def test_institutional_behavior_score_range(self):
        """测试InstitutionalBehavior评分范围"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        
        # 验证评分在0-100范围内
        valid_scores = raw_score.dropna()
        self.assertTrue(all(0 <= s <= 100 for s in valid_scores), "评分应在0-100范围内")
    
    def test_institutional_behavior_confidence_calculation(self):
        """测试InstitutionalBehavior置信度计算"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, patterns, {})
        
        # 验证置信度在0-1范围内
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_institutional_behavior_parameter_update(self):
        """测试InstitutionalBehavior参数更新"""
        new_volume_quantile = 0.8
        self.indicator.set_parameters(volume_quantile=new_volume_quantile)
        
        # 验证参数更新
        self.assertEqual(self.indicator.volume_quantile, new_volume_quantile)
    
    def test_institutional_behavior_required_columns(self):
        """测试InstitutionalBehavior必需列"""
        self.assertTrue(hasattr(self.indicator, 'REQUIRED_COLUMNS'))
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in expected_columns:
            self.assertIn(col, self.indicator.REQUIRED_COLUMNS)
    
    def test_institutional_behavior_patterns(self):
        """测试InstitutionalBehavior形态识别"""
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
                'INST_ABSORPTION_PHASE', 'INST_CONTROL_PHASE', 'INST_RALLY_PHASE',
                'INST_DISTRIBUTION_PHASE', 'INST_WAITING_PHASE'
            ]
            
            for pattern in expected_patterns:
                self.assertIn(pattern, patterns.columns, f"缺少形态列: {pattern}")
    
    def test_institutional_behavior_signals(self):
        """测试InstitutionalBehavior信号生成"""
        signals = self.indicator.generate_trading_signals(self.data)
        
        # 验证信号DataFrame结构
        self.assertIsInstance(signals, dict)
        expected_signal_keys = ['buy_signal', 'sell_signal', 'signal_strength']
        for key in expected_signal_keys:
            self.assertIn(key, signals, f"缺少信号键: {key}")
            self.assertIsInstance(signals[key], pd.Series)
    
    def test_institutional_behavior_phase_determination(self):
        """测试InstitutionalBehavior阶段判断"""
        result = self.indicator.calculate(self.data)
        
        # 验证阶段列存在
        self.assertIn('inst_phase', result.columns)
        
        # 验证阶段值的合理性
        phase_values = result['inst_phase'].dropna()
        if len(phase_values) > 0:
            valid_phases = ["吸筹期", "控盘期", "拉升期", "出货期", "观望期"]
            for phase in phase_values.unique():
                self.assertIn(phase, valid_phases, f"无效的机构阶段: {phase}")
    
    def test_institutional_behavior_pattern_calculation(self):
        """测试InstitutionalBehavior行为模式计算"""
        result = self.indicator.calculate(self.data)
        
        # 验证行为模式列存在
        self.assertIn('behavior_pattern', result.columns)
        
        # 验证行为模式值的合理性
        pattern_values = result['behavior_pattern'].dropna()
        if len(pattern_values) > 0:
            valid_patterns = ["温和吸筹", "强势吸筹", "洗盘", "拉升", "加速拉升", "出货", "集中出货", "未知"]
            for pattern in pattern_values.unique():
                self.assertIn(pattern, valid_patterns, f"无效的行为模式: {pattern}")
    
    def test_institutional_behavior_intensity_calculation(self):
        """测试InstitutionalBehavior行为强度计算"""
        result = self.indicator.calculate(self.data)
        
        # 验证行为强度列存在
        self.assertIn('behavior_intensity', result.columns)
        
        # 验证行为强度值的合理性
        intensity_values = result['behavior_intensity'].dropna()
        if len(intensity_values) > 0:
            # 行为强度应该是非负数
            self.assertTrue(all(v >= 0 for v in intensity_values), "行为强度应该是非负数")
    
    def test_institutional_behavior_phase_transitions(self):
        """测试InstitutionalBehavior阶段转换"""
        result = self.indicator.calculate(self.data)
        
        # 验证阶段转换列存在
        self.assertIn('phase_change', result.columns)
        
        # 验证阶段转换值的合理性
        change_values = result['phase_change'].dropna()
        if len(change_values) > 0:
            valid_changes = ["无变化", "吸筹完成", "开始拉升", "开始出货", "新一轮开始"]
            for change in change_values.unique():
                # 允许动态生成的转换描述
                if "→" not in change:
                    self.assertIn(change, valid_changes, f"无效的阶段转换: {change}")
    
    def test_institutional_behavior_activity_score(self):
        """测试InstitutionalBehavior活跃度评分"""
        result = self.indicator.calculate(self.data)
        
        # 验证活跃度评分列存在
        self.assertIn('inst_activity_score', result.columns)
        
        # 验证活跃度评分值的合理性
        activity_values = result['inst_activity_score'].dropna()
        if len(activity_values) > 0:
            # 活跃度评分应该是有限数值
            self.assertTrue(all(np.isfinite(v) for v in activity_values), 
                           "活跃度评分应该是有限数值")
    
    def test_institutional_behavior_cost_calculation(self):
        """测试InstitutionalBehavior成本计算"""
        result = self.indicator.calculate(self.data)
        
        # 验证成本列存在
        self.assertIn('inst_cost', result.columns)
        
        # 验证成本值的合理性
        cost_values = result['inst_cost'].dropna()
        if len(cost_values) > 0:
            # 成本应该是正数
            self.assertTrue(all(v > 0 for v in cost_values), "机构成本应该是正数")
    
    def test_institutional_behavior_classify_behavior(self):
        """测试InstitutionalBehavior行为分类"""
        # 需要足够的数据进行分类
        long_data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 100}
        ])
        
        classifications = self.indicator.classify_institutional_behavior(long_data)
        
        # 验证分类结果
        self.assertIsInstance(classifications, list)
        
        # 验证分类结果结构
        for classification in classifications:
            self.assertIsInstance(classification, dict)
            self.assertIn('type', classification)
            self.assertIn('description', classification)
    
    def test_institutional_behavior_predict_absorption(self):
        """测试InstitutionalBehavior吸筹预测"""
        # 需要足够的数据进行预测
        long_data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 105, 'periods': 80}
        ])
        
        prediction = self.indicator.predict_absorption_completion(long_data)
        
        # 验证预测结果
        self.assertIsInstance(prediction, dict)
        expected_keys = [
            'is_in_absorption', 'completion_days_min', 'completion_days_max',
            'confidence', 'description'
        ]
        for key in expected_keys:
            self.assertIn(key, prediction)
    
    def test_institutional_behavior_comprehensive_patterns(self):
        """测试InstitutionalBehavior综合形态"""
        # 先计算指标
        result = self.indicator.calculate(self.data)
        
        # 获取形态
        patterns = self.indicator.get_patterns(self.data)
        
        # 验证综合形态存在
        if not patterns.empty:
            comprehensive_patterns = [
                'INST_HIGH_CONCENTRATION', 'INST_EXTREME_ACTIVITY',
                'INST_ABSORPTION_COMPLETE', 'INST_RALLY_START'
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
    
    def test_institutional_behavior_register_patterns(self):
        """测试InstitutionalBehavior形态注册"""
        # 调用形态注册
        self.indicator.register_patterns()
        
        # 验证形态已注册（通过检查是否有异常抛出）
        self.assertTrue(True, "形态注册应该成功完成")
    
    def test_institutional_behavior_edge_cases(self):
        """测试InstitutionalBehavior边界情况"""
        # 测试数据不足的情况
        small_data = self.data.head(10)
        result = self.indicator.calculate(small_data)
        
        # InstitutionalBehavior应该能处理数据不足的情况
        self.assertIsInstance(result, pd.DataFrame)
        for col in self.expected_columns:
            self.assertIn(col, result.columns)
    
    def test_institutional_behavior_validation(self):
        """测试InstitutionalBehavior数据验证"""
        # 测试缺少必需列的情况
        invalid_data = self.data.drop(['volume'], axis=1)

        # BaseIndicator会处理缺失列并返回空DataFrame
        result = self.indicator.calculate(invalid_data)
        self.assertIsInstance(result, pd.DataFrame)
    
    def test_institutional_behavior_indicator_type(self):
        """测试InstitutionalBehavior指标类型"""
        indicator_type = self.indicator.get_indicator_type()
        self.assertEqual(indicator_type, "INSTITUTIONALBEHAVIOR", "指标类型应该是INSTITUTIONALBEHAVIOR")


if __name__ == '__main__':
    unittest.main()
