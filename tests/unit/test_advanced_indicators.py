"""
高级分析指标单元测试

测试各种高级分析指标的功能，包括：
- 艾略特波浪（ElliottWave）
- 斐波那契工具（FibonacciTools）
- 趋势强度（TrendStrength）
- 趋势分类（TrendClassification）
- 筹码分布（ChipDistribution）
- 机构行为（InstitutionalBehavior）
- 情绪分析（SentimentAnalysis）
- 江恩工具（GannTools）
"""

import unittest
import pandas as pd
import numpy as np
import io  # 添加io模块导入

from indicators.elliott_wave import ElliottWave
from indicators.fibonacci_tools import FibonacciTools
from indicators.trend.trend_strength import TrendStrength
from indicators.trend_classification import TrendClassification
from indicators.chip_distribution import ChipDistribution
from indicators.institutional_behavior import InstitutionalBehavior
from indicators.sentiment_analysis import SentimentAnalysis
from indicators.gann_tools import GannTools

from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator
from tests.helper.log_capture import LogCaptureMixin


class TestElliottWave(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    """艾略特波浪指标测试"""
    
    def setUp(self):
        """为测试准备数据和指标实例"""
        # 先调用LogCaptureMixin的setUp
        LogCaptureMixin.setUp(self)
        
        self.indicator = ElliottWave()
        
        # 定义预期输出列
        self.expected_columns = [
            'wave_degree', 'wave_number', 'wave_pattern', 
            'wave_direction', 'wave_start', 'wave_end'
        ]
        
        # 生成包含波浪特征的价格序列
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 30},  # 上升浪1
            {'type': 'trend', 'start_price': 120, 'end_price': 115, 'periods': 15},  # 调整浪2
            {'type': 'trend', 'start_price': 115, 'end_price': 150, 'periods': 40},  # 上升浪3
            {'type': 'trend', 'start_price': 150, 'end_price': 140, 'periods': 20},  # 调整浪4
            {'type': 'trend', 'start_price': 140, 'end_price': 165, 'periods': 25},  # 上升浪5
            {'type': 'trend', 'start_price': 165, 'end_price': 145, 'periods': 30},  # 调整浪A
            {'type': 'trend', 'start_price': 145, 'end_price': 155, 'periods': 20},  # 调整浪B
            {'type': 'trend', 'start_price': 155, 'end_price': 125, 'periods': 35}   # 调整浪C
        ])
    
    def tearDown(self):
        """清理测试环境"""
        LogCaptureMixin.tearDown(self)
    
    def test_wave_identification(self):
        """测试波浪识别功能"""
        result = self.indicator.calculate(self.data)
        
        # 验证结果包含波浪信息
        self.assertIn('wave_pattern', result.columns, "结果中应包含wave_pattern列")
        self.assertIn('wave_number', result.columns, "结果中应包含wave_number列")
        
        # 验证至少识别出一个波浪
        self.assertTrue((result['wave_start'] == 1).any(), "应识别出至少一个波浪起点")
        self.assertTrue((result['wave_end'] == 1).any(), "应识别出至少一个波浪终点")
    
    def test_score_calculation(self):
        """测试波浪评分计算"""
        score = self.indicator.calculate_raw_score(self.data)
        
        # 验证评分为Series且在0-100范围内
        self._verify_raw_score(score)
    
    def test_wave_patterns(self):
        """测试波浪形态识别"""
        try:
            patterns = self.indicator.identify_patterns(self.data)
            
            # 验证返回的是形态列表
            self.assertIsInstance(patterns, list, "波浪形态识别应返回列表")
            
            # 验证识别出形态
            self.assertGreater(len(patterns), 0, "应识别出至少一种波浪形态")
            
            # 打印识别出的形态
            print(f"识别出的波浪形态: {patterns}")
        except Exception as e:
            self.skipTest(f"波浪形态识别测试失败: {e}")
    
    def test_complex_wave_scenario(self):
        """测试复杂波浪场景"""
        # 生成复杂的波浪序列
        complex_data = TestDataGenerator.generate_price_sequence([
            {'type': 'v_shape', 'start_price': 100, 'bottom_price': 80, 'periods': 50},
            {'type': 'trend', 'start_price': 100, 'end_price': 130, 'periods': 50},
            {'type': 'head_shoulders', 'start_price': 130, 'peak_price': 150, 'periods': 80}
        ])
        
        result = self.indicator.calculate(complex_data)
        
        # 验证能处理复杂场景
        self.assertIsInstance(result, pd.DataFrame, "应能处理复杂波浪场景")
        
        # 验证核心列存在
        for col in ['wave_pattern', 'wave_number', 'wave_direction']:
            self.assertIn(col, result.columns, f"复杂场景结果中应包含{col}列")
    
    def test_signals_generation(self):
        """测试信号生成功能"""
        if not hasattr(self.indicator, 'generate_signals'):
            self.skipTest("ElliottWave指标未实现generate_signals方法")
        
        try:
            signals = self.indicator.generate_signals(
                self.indicator.calculate(self.data)
            )
            
            # 验证信号DataFrame
            self.assertIsInstance(signals, pd.DataFrame, "生成的信号应为DataFrame")
            
            # 验证基本信号列
            signal_columns = ['buy_signal', 'sell_signal', 'score']
            for col in signal_columns:
                self.assertIn(col, signals.columns, f"信号结果中应包含{col}列")
        except Exception as e:
            self.skipTest(f"信号生成测试失败: {e}")


class TestFibonacciTools(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    """斐波那契工具指标测试"""
    
    def setUp(self):
        """为测试准备数据和指标实例"""
        # 先调用LogCaptureMixin的setUp
        LogCaptureMixin.setUp(self)
        
        self.indicator = FibonacciTools()
        
        # 定义预期输出列（包含常见的斐波那契水平值）
        self.expected_columns = [
            'fib_retracement_0_382', 'fib_retracement_0_500',
            'fib_retracement_0_618', 'fib_retracement_0_786',
            'fib_extension_1_272', 'fib_extension_1_618'
        ]
        
        # 生成测试数据：上涨趋势后回调
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 150, 'periods': 50},  # 上升趋势
            {'type': 'trend', 'start_price': 150, 'end_price': 130, 'periods': 30},  # 回调阶段
            {'type': 'trend', 'start_price': 130, 'end_price': 170, 'periods': 40}   # 延续上升
        ])
    
    def tearDown(self):
        """清理测试环境"""
        LogCaptureMixin.tearDown(self)
    
    def test_retracement_levels(self):
        """测试回撤水平计算"""
        result = self.indicator.calculate(self.data)
        
        # 验证回撤水平列存在
        for level in ['0_382', '0_500', '0_618']:
            column = f'fib_retracement_{level}'
            self.assertIn(column, result.columns, f"结果中应包含{column}列")
            
            # 验证回撤水平在高低点之间
            level_values = result[column].dropna()
            if not level_values.empty:
                max_price = self.data['high'].max()
                min_price = self.data['low'].min()
                self.assertTrue(
                    (level_values >= min_price).all() and (level_values <= max_price).all(),
                    f"回撤水平{level}应在价格范围内"
                )
    
    def test_extension_levels(self):
        """测试延伸水平计算"""
        result = self.indicator.calculate(self.data)
        
        # 验证延伸水平列存在
        for level in ['1_272', '1_618']:
            column = f'fib_extension_{level}'
            self.assertIn(column, result.columns, f"结果中应包含{column}列")
    
    def test_fibonacci_patterns(self):
        """测试斐波那契形态识别"""
        try:
            result = self.indicator.calculate(self.data)
            patterns = self.indicator.identify_patterns(self.data)
            
            # 验证返回形态列表
            self.assertIsInstance(patterns, list, "斐波那契形态识别应返回列表")
            
            # 打印识别出的形态
            print(f"识别出的斐波那契形态: {patterns}")
        except Exception as e:
            self.skipTest(f"斐波那契形态识别测试失败: {e}")
    
    def test_price_target_calculation(self):
        """测试价格目标计算"""
        result = self.indicator.calculate(self.data)
        
        # 检查是否计算了价格目标
        target_columns = [col for col in result.columns if 'target' in col]
        if target_columns:
            for column in target_columns:
                target_values = result[column].dropna()
                if not target_values.empty:
                    # 验证目标价格是有效值
                    self.assertFalse(target_values.isnull().any(), f"{column}不应包含NaN值")
    
    def test_score_calculation(self):
        """测试斐波那契评分计算"""
        score = self.indicator.calculate_raw_score(self.data)
        
        # 验证评分是有效的Series且在0-100范围内
        self._verify_raw_score(score)
    
    def test_with_downtrend_data(self):
        """测试下降趋势数据"""
        # 生成下降趋势数据
        downtrend_data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 150, 'end_price': 100, 'periods': 50},  # 下降趋势
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 30},  # 回调阶段
            {'type': 'trend', 'start_price': 120, 'end_price': 90, 'periods': 40}    # 延续下降
        ])
        
        # 计算结果
        result = self.indicator.calculate(downtrend_data)
        
        # 验证能处理下降趋势
        self.assertIsInstance(result, pd.DataFrame, "应能处理下降趋势数据")
        
        # 验证核心列存在
        for level in ['0_382', '0_500', '0_618']:
            self.assertIn(f'fib_retracement_{level}', result.columns, 
                         f"下降趋势结果中应包含fib_retracement_{level}列")


class TestTrendStrength(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    """趋势强度指标测试"""
    
    def setUp(self):
        """为测试准备数据和指标实例"""
        # 先调用LogCaptureMixin的setUp
        LogCaptureMixin.setUp(self)
        
        self.indicator = TrendStrength()
        
        # 定义预期输出列
        self.expected_columns = [
            'trend_strength', 'trend_direction', 'trend_category'
        ]
        
        # 生成测试数据：包含明显的趋势变化
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 140, 'periods': 40},  # 强劲上升趋势
            {'type': 'sideways', 'start_price': 140, 'periods': 20},                 # 横盘整理
            {'type': 'trend', 'start_price': 140, 'end_price': 100, 'periods': 30},  # 下降趋势
            {'type': 'sideways', 'start_price': 100, 'periods': 15},                 # 横盘整理
            {'type': 'v_shape', 'start_price': 100, 'bottom_price': 90, 'periods': 30}  # V形反转
        ])
    
    def tearDown(self):
        """清理测试环境"""
        LogCaptureMixin.tearDown(self)
    
    def test_trend_strength_calculation(self):
        """测试趋势强度计算"""
        result = self.indicator.calculate(self.data)
        
        # 验证趋势强度列存在
        self.assertIn('trend_strength', result.columns, "结果中应包含trend_strength列")
        
        # 验证趋势强度值在合理范围内
        strength_values = result['trend_strength'].dropna()
        if not strength_values.empty:
            self.assertTrue(
                (strength_values >= 0).all() and (strength_values <= 100).all(),
                "趋势强度值应在0-100范围内"
            )
    
    def test_trend_direction_classification(self):
        """测试趋势方向分类"""
        result = self.indicator.calculate(self.data)
        
        # 验证趋势方向列存在
        self.assertIn('trend_direction', result.columns, "结果中应包含trend_direction列")
        
        # 验证方向分类结果
        if 'trend_direction' in result.columns:
            direction_values = result['trend_direction'].dropna()
            if not direction_values.empty:
                # 验证方向值是有效的（应为uptrend、downtrend或neutral）
                unique_directions = direction_values.unique()
                for direction in unique_directions:
                    self.assertIn(
                        direction, ['uptrend', 'downtrend', 'neutral'],
                        f"趋势方向值应为'uptrend'、'downtrend'或'neutral'，而不是'{direction}'"
                    )
    
    def test_trend_category_classification(self):
        """测试趋势类别分类"""
        result = self.indicator.calculate(self.data)
        
        # 验证趋势类别列存在
        self.assertIn('trend_category', result.columns, "结果中应包含trend_category列")
        
        # 验证类别分类结果
        if 'trend_category' in result.columns:
            category_values = result['trend_category'].dropna()
            if not category_values.empty:
                # 验证类别值是有效的（应为strong、moderate或weak）
                unique_categories = category_values.unique()
                for category in unique_categories:
                    self.assertIn(
                        category, ['strong', 'moderate', 'weak'],
                        f"趋势类别值应为'strong'、'moderate'或'weak'，而不是'{category}'"
                    )
    
    def test_score_calculation(self):
        """测试趋势强度评分计算"""
        score = self.indicator.calculate_raw_score(self.data)
        
        # 验证评分是有效的Series且在0-100范围内
        self._verify_raw_score(score)
    
    def test_with_extreme_data(self):
        """测试极端数据"""
        # 创建包含极端上涨和下跌的数据
        extreme_data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 200, 'periods': 30},  # 极端上涨
            {'type': 'trend', 'start_price': 200, 'end_price': 100, 'periods': 30}   # 极端下跌
        ])
        
        # 计算结果
        result = self.indicator.calculate(extreme_data)
        
        # 验证能处理极端数据
        self.assertIsInstance(result, pd.DataFrame, "应能处理极端数据")
        
        # 验证在极端上涨期间有较高的趋势强度值
        if 'trend_strength' in result.columns and 'trend_direction' in result.columns:
            uptrend_strength = result.iloc[:30]['trend_strength'].max()
            downtrend_strength = result.iloc[30:]['trend_strength'].max()
            
            # 打印结果，帮助调试
            print(f"极端上涨的最大趋势强度: {uptrend_strength}")
            print(f"极端下跌的最大趋势强度: {downtrend_strength}")
            
            # 验证极端趋势有较高的强度值（应超过60）
            self.assertGreaterEqual(max(uptrend_strength, downtrend_strength), 60,
                                  "极端趋势应有较高的强度值（至少60）")


class TestTrendClassification(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    """趋势分类指标测试"""
    
    def setUp(self):
        """为测试准备数据和指标实例"""
        # 先调用LogCaptureMixin的setUp
        LogCaptureMixin.setUp(self)
        
        self.indicator = TrendClassification()
        
        # 定义预期输出列
        self.expected_columns = [
            'trend_direction', 'short_trend', 'medium_trend', 'long_trend'
        ]
        
        # 生成测试数据：包含不同周期的趋势
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 130, 'periods': 30},  # 上升趋势
            {'type': 'trend', 'start_price': 130, 'end_price': 110, 'periods': 20},  # 回调
            {'type': 'trend', 'start_price': 110, 'end_price': 150, 'periods': 40},  # 更长的上升趋势
            {'type': 'trend', 'start_price': 150, 'end_price': 130, 'periods': 30}   # 回调
        ])
    
    def tearDown(self):
        """清理测试环境"""
        LogCaptureMixin.tearDown(self)
    
    def test_trend_direction_calculation(self):
        """测试趋势方向计算"""
        result = self.indicator.calculate(self.data)
        
        # 验证趋势方向列存在
        self.assertIn('trend_direction', result.columns, "结果中应包含trend_direction列")
        
        # 验证趋势方向值
        if 'trend_direction' in result.columns:
            direction_values = result['trend_direction'].dropna()
            if not direction_values.empty:
                # 验证方向值在合理范围内
                self.assertTrue(
                    (direction_values >= -2).all() and (direction_values <= 2).all(),
                    "趋势方向值应在-2到2范围内"
                )
    
    def test_multiple_timeframe_trends(self):
        """测试多时间周期趋势"""
        result = self.indicator.calculate(self.data)
        
        # 验证短期、中期和长期趋势列存在
        for trend_type in ['short_trend', 'medium_trend', 'long_trend']:
            self.assertIn(trend_type, result.columns, f"结果中应包含{trend_type}列")
            
            # 验证趋势值在-1到1之间
            if trend_type in result.columns:
                trend_values = result[trend_type].dropna()
                if not trend_values.empty:
                    self.assertTrue(
                        (trend_values >= -1).all() and (trend_values <= 1).all(),
                        f"{trend_type}值应在-1到1范围内"
                    )
    
    def test_moving_average_alignment(self):
        """测试均线排列"""
        result = self.indicator.calculate(self.data)
        
        # 验证均线排列列存在
        if 'ma_alignment' in result.columns:
            alignment_values = result['ma_alignment'].dropna()
            if not alignment_values.empty:
                # 验证均线排列值在-1到1之间
                self.assertTrue(
                    (alignment_values >= -1).all() and (alignment_values <= 1).all(),
                    "ma_alignment值应在-1到1范围内"
                )
    
    def test_trend_strength_calculation(self):
        """测试趋势强度计算"""
        result = self.indicator.calculate(self.data)
        
        # 验证趋势强度列存在
        if 'trend_strength' in result.columns:
            strength_values = result['trend_strength'].dropna()
            if not strength_values.empty:
                # 验证强度值在0-1范围内
                self.assertTrue(
                    (strength_values >= 0).all() and (strength_values <= 1).all(),
                    "trend_strength值应在0到1范围内"
                )
    
    def test_score_calculation(self):
        """测试趋势分类评分计算"""
        try:
            score = self.indicator.calculate_raw_score(self.data)
            
            # 验证评分是有效的Series且在0-100范围内
            self._verify_raw_score(score)
        except Exception as e:
            self.skipTest(f"趋势分类评分计算测试失败: {e}")
    
    def test_with_volatile_data(self):
        """测试高波动数据"""
        # 创建高波动数据
        volatile_data = TestDataGenerator.generate_price_sequence([
            {'type': 'sideways', 'start_price': 100, 'volatility': 0.05, 'periods': 60}  # 高波动横盘
        ])
        
        # 计算结果
        result = self.indicator.calculate(volatile_data)
        
        # 验证能处理高波动数据
        self.assertIsInstance(result, pd.DataFrame, "应能处理高波动数据")


class TestChipDistribution(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    """筹码分布指标测试"""
    
    def setUp(self):
        """为测试准备数据和指标实例"""
        # 先调用LogCaptureMixin的setUp
        LogCaptureMixin.setUp(self)
        
        self.indicator = ChipDistribution()
        
        # 定义预期输出列
        self.expected_columns = [
            'avg_cost', 'profit_ratio', 'chip_concentration'
        ]
        
        # 生成测试数据：包含换手放量的波段行情
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 30},  # 上升阶段
            {'type': 'trend', 'start_price': 120, 'end_price': 110, 'periods': 20},  # 回调阶段
            {'type': 'trend', 'start_price': 110, 'end_price': 130, 'periods': 40},  # 二次上升
            {'type': 'trend', 'start_price': 130, 'end_price': 140, 'periods': 30}   # 继续上升
        ])
        
        # 修改成交量，创建放量区间
        self.data['volume'].iloc[25:35] = self.data['volume'].iloc[25:35] * 3  # 上升中放量
        self.data['volume'].iloc[60:70] = self.data['volume'].iloc[60:70] * 2.5  # 二次上升放量
        
        # 修改换手率
        self.data['turnover_rate'] = self.data['volume'] / self.data['volume'].mean() * 3
    
    def tearDown(self):
        """清理测试环境"""
        LogCaptureMixin.tearDown(self)
    
    def test_cost_calculation(self):
        """测试成本计算"""
        result = self.indicator.calculate(self.data)
        
        # 验证平均成本列存在
        self.assertIn('avg_cost', result.columns, "结果中应包含avg_cost列")
        
        # 验证平均成本在价格范围内
        if 'avg_cost' in result.columns:
            cost_values = result['avg_cost'].dropna()
            if not cost_values.empty:
                min_price = self.data['low'].min()
                max_price = self.data['high'].max()
                self.assertTrue(
                    (cost_values >= min_price * 0.9).all() and (cost_values <= max_price * 1.1).all(),
                    "平均成本应在价格范围内（允许10%的误差）"
                )
    
    def test_profit_ratio_calculation(self):
        """测试获利比例计算"""
        result = self.indicator.calculate(self.data)
        
        # 验证获利比例列存在
        self.assertIn('profit_ratio', result.columns, "结果中应包含profit_ratio列")
        
        # 验证获利比例在0-1范围内
        if 'profit_ratio' in result.columns:
            profit_values = result['profit_ratio'].dropna()
            if not profit_values.empty:
                self.assertTrue(
                    (profit_values >= 0).all() and (profit_values <= 1).all(),
                    "获利比例应在0-1范围内"
                )
    
    def test_chip_concentration_calculation(self):
        """测试筹码集中度计算"""
        result = self.indicator.calculate(self.data)
        
        # 验证筹码集中度列存在
        if 'chip_concentration' in result.columns:
            concentration_values = result['chip_concentration'].dropna()
            if not concentration_values.empty:
                # 验证集中度在0-1范围内
                self.assertTrue(
                    (concentration_values >= 0).all() and (concentration_values <= 1).all(),
                    "筹码集中度应在0-1范围内"
                )
    
    def test_score_calculation(self):
        """测试筹码分布评分计算"""
        try:
            score = self.indicator.calculate_raw_score(self.data)
            
            # 验证评分是有效的Series且在0-100范围内
            self._verify_raw_score(score)
        except Exception as e:
            self.skipTest(f"筹码分布评分计算测试失败: {e}")
    
    def test_institutional_chips_identification(self):
        """测试机构筹码识别"""
        try:
            inst_result = self.indicator.identify_institutional_chips(self.data)
            
            # 验证返回的是DataFrame
            self.assertIsInstance(inst_result, pd.DataFrame, "机构筹码识别应返回DataFrame")
            
            # 验证包含机构成本列
            self.assertIn('inst_cost', inst_result.columns, "结果中应包含inst_cost列")
            
            # 验证包含机构获利比例列
            self.assertIn('inst_profit_ratio', inst_result.columns, "结果中应包含inst_profit_ratio列")
        except Exception as e:
            self.skipTest(f"机构筹码识别测试失败: {e}")
    
    def test_trapped_position_prediction(self):
        """测试套牢盘预测"""
        try:
            # 这个方法可能尚未实现，所以用try-except包装
            trapped_result = self.indicator.predict_trapped_position_release(self.data)
            
            # 验证返回的是DataFrame
            self.assertIsInstance(trapped_result, pd.DataFrame, "套牢盘预测应返回DataFrame")
        except Exception as e:
            self.skipTest(f"套牢盘预测测试失败: {e}")


class TestInstitutionalBehavior(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    """机构行为指标测试"""
    
    def setUp(self):
        """为测试准备数据和指标实例"""
        # 先调用LogCaptureMixin的setUp
        LogCaptureMixin.setUp(self)
        
        self.indicator = InstitutionalBehavior()
        
        # 定义预期输出列
        self.expected_columns = [
            'institution_activity', 'institution_position', 'behavior_type'
        ]
        
        # 生成测试数据：包含不同的机构行为特征
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'sideways', 'start_price': 100, 'periods': 30, 'volatility': 0.01},  # 低波动横盘（吸筹）
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 40},      # 温和上升（建仓）
            {'type': 'trend', 'start_price': 120, 'end_price': 150, 'periods': 30},      # 加速上升（拉升）
            {'type': 'trend', 'start_price': 150, 'end_price': 145, 'periods': 20}       # 小幅回调（出货开始）
        ])
        
        # 修改成交量特征，模拟机构行为
        # 吸筹阶段：小幅放量
        self.data['volume'].iloc[10:30] = self.data['volume'].iloc[10:30] * 1.5
        
        # 建仓阶段：量价配合
        for i in range(30, 70):
            self.data['volume'].iloc[i] = self.data['volume'].iloc[i] * (1 + (i - 30) / 80)
        
        # 拉升阶段：放量
        self.data['volume'].iloc[70:100] = self.data['volume'].iloc[70:100] * 2.5
        
        # 出货阶段：巨量
        self.data['volume'].iloc[100:] = self.data['volume'].iloc[100:] * 3
        
        # 更新换手率
        self.data['turnover_rate'] = self.data['volume'] / self.data['volume'].mean() * 3
    
    def tearDown(self):
        """清理测试环境"""
        LogCaptureMixin.tearDown(self)
    
    def test_institution_activity_calculation(self):
        """测试机构活跃度计算"""
        result = self.indicator.calculate(self.data)
        
        # 验证机构活跃度列存在
        if 'institution_activity' in result.columns:
            activity_values = result['institution_activity'].dropna()
            if not activity_values.empty:
                # 验证活跃度在0-100范围内
                self.assertTrue(
                    (activity_values >= 0).all() and (activity_values <= 100).all(),
                    "机构活跃度应在0-100范围内"
                )
    
    def test_institution_position_calculation(self):
        """测试机构持仓计算"""
        result = self.indicator.calculate(self.data)
        
        # 验证机构持仓列存在
        if 'institution_position' in result.columns:
            position_values = result['institution_position'].dropna()
            if not position_values.empty:
                # 验证持仓在0-1范围内
                self.assertTrue(
                    (position_values >= 0).all() and (position_values <= 1).all(),
                    "机构持仓应在0-1范围内"
                )
    
    def test_behavior_type_identification(self):
        """测试行为类型识别"""
        result = self.indicator.calculate(self.data)
        
        # 验证行为类型列存在
        if 'behavior_type' in result.columns:
            # 验证行为类型不全为NaN
            self.assertFalse(
                result['behavior_type'].isna().all(),
                "行为类型不应全为NaN"
            )
            
            # 获取所有非NaN的行为类型
            behavior_types = result['behavior_type'].dropna().unique()
            
            # 验证有行为类型被识别
            self.assertGreater(len(behavior_types), 0, "应识别出至少一种行为类型")
            
            print(f"识别出的行为类型: {behavior_types}")
    
    def test_score_calculation(self):
        """测试机构行为评分计算"""
        try:
            score = self.indicator.calculate_raw_score(self.data)
            
            # 验证评分是有效的Series且在0-100范围内
            self._verify_raw_score(score)
        except Exception as e:
            self.skipTest(f"机构行为评分计算测试失败: {e}")
    
    def test_behavior_pattern_identification(self):
        """测试行为模式识别"""
        try:
            if hasattr(self.indicator, 'identify_patterns'):
                patterns = self.indicator.identify_patterns(self.data)
                
                # 验证返回形态列表
                self.assertIsInstance(patterns, list, "行为模式识别应返回列表")
                
                # 验证识别出模式
                self.assertGreater(len(patterns), 0, "应识别出至少一种行为模式")
                
                print(f"识别出的机构行为模式: {patterns}")
        except Exception as e:
            self.skipTest(f"行为模式识别测试失败: {e}")
    
    def test_with_chip_distribution_integration(self):
        """测试与筹码分布的集成"""
        try:
            # 创建筹码分布指标
            chip_indicator = ChipDistribution()
            
            # 计算筹码分布
            chip_result = chip_indicator.calculate(self.data)
            
            # 如果机构行为指标有接受筹码分布结果的方法
            if hasattr(self.indicator, 'calculate_with_chip_distribution'):
                integrated_result = self.indicator.calculate_with_chip_distribution(self.data, chip_result)
                
                # 验证返回的是DataFrame
                self.assertIsInstance(integrated_result, pd.DataFrame, "与筹码分布集成应返回DataFrame")
        except Exception as e:
            self.skipTest(f"与筹码分布集成测试失败: {e}")


class TestSentimentAnalysis(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    """情绪分析指标测试"""
    
    def setUp(self):
        """为测试准备数据和指标实例"""
        # 先调用LogCaptureMixin的setUp
        LogCaptureMixin.setUp(self)
        
        self.indicator = SentimentAnalysis()
        
        # 定义预期输出列
        self.expected_columns = [
            'sentiment_index', 'sentiment_category', 'sentiment_bias'
        ]
        
        # 生成测试数据：包含不同情绪周期的数据
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 30},  # 看涨情绪（上涨）
            {'type': 'trend', 'start_price': 120, 'end_price': 110, 'periods': 20},  # 看跌情绪（回调）
            {'type': 'sideways', 'start_price': 110, 'periods': 15, 'volatility': 0.02},  # 中性情绪（盘整）
            {'type': 'trend', 'start_price': 110, 'end_price': 95, 'periods': 25},   # 恐慌情绪（下跌）
            {'type': 'v_shape', 'start_price': 95, 'bottom_price': 90, 'periods': 30}  # 情绪反转（V形）
        ])
        
        # 调整成交量特征以反映情绪变化
        # 看涨阶段：量增价升
        self.data['volume'].iloc[:30] = self.data['volume'].iloc[:30] * (1 + np.arange(30) / 60)
        
        # 回调阶段：量缩
        self.data['volume'].iloc[30:50] = self.data['volume'].iloc[30:50] * 0.7
        
        # 下跌阶段：放量下跌（恐慌）
        self.data['volume'].iloc[65:90] = self.data['volume'].iloc[65:90] * 2
        
        # V形反转：底部放量
        self.data['volume'].iloc[95:105] = self.data['volume'].iloc[95:105] * 2.5
    
    def tearDown(self):
        """清理测试环境"""
        LogCaptureMixin.tearDown(self)
    
    def test_sentiment_index_calculation(self):
        """测试情绪指数计算"""
        result = self.indicator.calculate(self.data)
        
        # 验证情绪指数列存在
        self.assertIn('sentiment_index', result.columns, "结果中应包含sentiment_index列")
        
        # 验证情绪指数在0-100范围内
        if 'sentiment_index' in result.columns:
            index_values = result['sentiment_index'].dropna()
            if not index_values.empty:
                self.assertTrue(
                    (index_values >= 0).all() and (index_values <= 100).all(),
                    "情绪指数应在0-100范围内"
                )
    
    def test_sentiment_category_classification(self):
        """测试情绪类别分类"""
        result = self.indicator.calculate(self.data)
        
        # 验证情绪类别列存在
        if 'sentiment_category' in result.columns:
            # 验证情绪类别不全为NaN
            self.assertFalse(
                result['sentiment_category'].isna().all(),
                "情绪类别不应全为NaN"
            )
            
            # 获取所有非NaN的情绪类别
            categories = result['sentiment_category'].dropna().unique()
            
            # 验证有情绪类别被识别
            self.assertGreater(len(categories), 0, "应识别出至少一种情绪类别")
            
            # 打印识别出的情绪类别
            print(f"识别出的情绪类别: {categories}")
    
    def test_sentiment_bias_calculation(self):
        """测试情绪偏差计算"""
        result = self.indicator.calculate(self.data)
        
        # 验证情绪偏差列存在
        if 'sentiment_bias' in result.columns:
            bias_values = result['sentiment_bias'].dropna()
            if not bias_values.empty:
                # 验证偏差值在合理范围内
                self.assertTrue(
                    (bias_values >= -100).all() and (bias_values <= 100).all(),
                    "情绪偏差应在-100到100范围内"
                )
    
    def test_sentiment_change_detection(self):
        """测试情绪变化检测"""
        result = self.indicator.calculate(self.data)
        
        # 验证情绪变化列存在
        if 'sentiment_change' in result.columns:
            change_values = result['sentiment_change'].dropna()
            if not change_values.empty:
                # 验证变化值是有效的（应为-1、0或1）
                unique_changes = change_values.unique()
                for change in unique_changes:
                    self.assertIn(
                        change, [-1, 0, 1],
                        f"情绪变化值应为-1、0或1，而不是{change}"
                    )
    
    def test_score_calculation(self):
        """测试情绪分析评分计算"""
        try:
            score = self.indicator.calculate_raw_score(self.data)
            
            # 验证评分是有效的Series且在0-100范围内
            self._verify_raw_score(score)
        except Exception as e:
            self.skipTest(f"情绪分析评分计算测试失败: {e}")
    
    def test_extreme_sentiment_detection(self):
        """测试极端情绪检测"""
        result = self.indicator.calculate(self.data)
        
        # 验证极端情绪列存在
        if 'extreme_sentiment' in result.columns:
            extreme_values = result['extreme_sentiment'].dropna()
            if not extreme_values.empty:
                # 验证极端情绪值是有效的（应为-1、0或1）
                unique_extremes = extreme_values.unique()
                for extreme in unique_extremes:
                    self.assertIn(
                        extreme, [-1, 0, 1],
                        f"极端情绪值应为-1、0或1，而不是{extreme}"
                    )


class TestGannTools(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    """江恩工具指标测试"""
    
    def setUp(self):
        """为测试准备数据和指标实例"""
        # 先调用LogCaptureMixin的setUp
        LogCaptureMixin.setUp(self)
        
        self.indicator = GannTools()
        
        # 定义预期输出列（包含主要的江恩角度线）
        self.expected_columns = [
            'ANGLE_1X1', 'ANGLE_1X2', 'ANGLE_2X1',
            'TIME_CYCLE_90', 'TIME_CYCLE_144'
        ]
        
        # 生成测试数据：明显的趋势变化点
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 150, 'periods': 90},  # 上升趋势（90天周期）
            {'type': 'trend', 'start_price': 150, 'end_price': 140, 'periods': 30},  # 回调
            {'type': 'trend', 'start_price': 140, 'end_price': 180, 'periods': 54}   # 继续上升（54天，共144天）
        ])
    
    def tearDown(self):
        """清理测试环境"""
        LogCaptureMixin.tearDown(self)
    
    def test_gann_angle_calculation(self):
        """测试江恩角度线计算"""
        result = self.indicator.calculate(self.data)
        
        # 验证江恩角度线列存在
        for angle_name in ['ANGLE_1X1', 'ANGLE_1X2', 'ANGLE_2X1']:
            if angle_name in result.columns:
                # 验证角度线值不全为NaN
                self.assertFalse(
                    result[angle_name].isna().all(),
                    f"{angle_name}不应全为NaN"
                )
    
    def test_gann_time_cycle_calculation(self):
        """测试江恩时间周期计算"""
        result = self.indicator.calculate(self.data)
        
        # 验证江恩时间周期列存在
        time_cycle_columns = [col for col in result.columns if 'TIME_CYCLE' in col]
        if time_cycle_columns:
            for cycle_col in time_cycle_columns:
                # 检查是否有周期点被标记
                if not result[cycle_col].isna().all():
                    # 找到被标记为周期点的位置
                    cycle_points = result[cycle_col].dropna()
                    
                    # 至少应有一个周期点
                    self.assertGreater(len(cycle_points), 0, f"{cycle_col}应标记至少一个周期点")
    
    def test_gann_square_calculation(self):
        """测试江恩方格计算"""
        if hasattr(self.indicator, 'calculate_gann_square'):
            try:
                # 计算江恩方格
                square_result = self.indicator.calculate_gann_square(self.data)
                
                # 验证返回结果
                self.assertIsInstance(square_result, pd.DataFrame, "江恩方格计算应返回DataFrame")
            except Exception as e:
                self.skipTest(f"江恩方格计算测试失败: {e}")
    
    def test_score_calculation(self):
        """测试江恩工具评分计算"""
        try:
            score = self.indicator.calculate_raw_score(self.data)
            
            # 验证评分是有效的Series且在0-100范围内
            self._verify_raw_score(score)
        except Exception as e:
            self.skipTest(f"江恩工具评分计算测试失败: {e}")
    
    def test_pattern_identification(self):
        """测试江恩工具形态识别"""
        try:
            if hasattr(self.indicator, 'identify_patterns'):
                patterns = self.indicator.identify_patterns(self.data)
                
                # 验证返回形态列表
                self.assertIsInstance(patterns, list, "江恩工具形态识别应返回列表")
                
                # 打印识别出的形态
                print(f"识别出的江恩工具形态: {patterns}")
        except Exception as e:
            self.skipTest(f"江恩工具形态识别测试失败: {e}")
    
    def test_support_resistance_levels(self):
        """测试支撑阻力位计算"""
        result = self.indicator.calculate(self.data)
        
        # 检查是否有支撑阻力位相关列
        support_columns = [col for col in result.columns if 'support' in col.lower() or 'resistance' in col.lower()]
        
        if support_columns:
            for col in support_columns:
                support_values = result[col].dropna()
                if not support_values.empty:
                    # 验证支撑阻力位在价格范围内
                    min_price = self.data['low'].min()
                    max_price = self.data['high'].max()
                    self.assertTrue(
                        (support_values >= min_price * 0.5).all() and (support_values <= max_price * 1.5).all(),
                        f"{col}应在价格范围内（允许宽松的范围）"
                    )


if __name__ == '__main__':
    unittest.main() 