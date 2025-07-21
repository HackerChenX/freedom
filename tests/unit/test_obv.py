#!/usr/bin/env python3

import unittest
import pandas as pd
import numpy as np
import warnings
import io
import logging
from typing import Dict, Any

from indicators.obv import OnBalanceVolume


class TestOBVIndicator(unittest.TestCase):
    """OBV指标测试类"""
    
    def setUp(self):
        """测试前准备"""
        warnings.filterwarnings('ignore')
        
        # 设置日志处理器
        self.log_stream = io.StringIO()
        self.log_handler = logging.StreamHandler(self.log_stream)
        self.log_handler.setLevel(logging.ERROR)
        
        # 获取相关的logger并添加处理器
        loggers = [
            logging.getLogger('indicators.obv'),
            logging.getLogger('indicators.complete_indicator_registry'),
            logging.getLogger('analysis.engines'),
            logging.getLogger('utils')
        ]
        
        for logger in loggers:
            logger.addHandler(self.log_handler)
            logger.setLevel(logging.ERROR)
        
        # 创建测试数据
        self.data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109] * 5,
            'high': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114] * 5,
            'low': [95, 96, 97, 98, 99, 100, 101, 102, 103, 104] * 5,
            'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109] * 5,
            'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900] * 5
        })
        
        # 创建OBV指标实例
        self.indicator = OnBalanceVolume(signal_period=10)
    
    def tearDown(self):
        """测试后清理"""
        # 移除日志处理器
        loggers = [
            logging.getLogger('indicators.obv'),
            logging.getLogger('indicators.complete_indicator_registry'),
            logging.getLogger('analysis.engines'),
            logging.getLogger('utils')
        ]
        
        for logger in loggers:
            logger.removeHandler(self.log_handler)
        
        self.log_handler.close()
    
    def assertNoLogs(self, level):
        """检查没有指定级别的日志"""
        log_output = self.log_stream.getvalue()
        self.assertEqual(log_output.strip(), '', f"发现{level}级别日志: {log_output}")
    
    def test_obv_initialization(self):
        """测试OBV指标初始化"""
        indicator = OnBalanceVolume()
        self.assertEqual(indicator.name, "OBV")
        self.assertIsInstance(indicator.signal_period, int)
        self.assertEqual(indicator.signal_period, 10)  # 默认值
        
        # 测试自定义参数
        custom_indicator = OnBalanceVolume(signal_period=20)
        self.assertEqual(custom_indicator.signal_period, 20)
    
    def test_obv_required_columns(self):
        """测试OBV所需列"""
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in expected_columns:
            self.assertIn(col, self.indicator.REQUIRED_COLUMNS)
    
    def test_obv_calculation(self):
        """测试OBV计算"""
        result = self.indicator.calculate(self.data)
        
        # 验证返回DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        
        # 验证包含必要的列
        expected_columns = ['OBV', 'obv', 'obv_ma', 'obv_signal']
        for col in expected_columns:
            self.assertIn(col, result.columns, f"缺少列: {col}")
        
        # 验证数据类型
        self.assertTrue(pd.api.types.is_numeric_dtype(result['OBV']))
        self.assertTrue(pd.api.types.is_numeric_dtype(result['obv']))
        
        # 验证OBV计算逻辑
        self.assertFalse(result['OBV'].isna().all(), "OBV值不应该全为NaN")
        
        # OBV首个值应该为0
        self.assertEqual(result['OBV'].iloc[0], 0, "OBV首个值应该为0")
    
    def test_obv_patterns(self):
        """测试OBV形态检测"""
        # 先计算指标
        result = self.indicator.calculate(self.data)
        
        # 获取形态
        patterns = self.indicator.get_patterns(self.data)
        
        # 验证返回DataFrame
        self.assertIsInstance(patterns, pd.DataFrame)
        
        # 验证形态列存在
        expected_patterns = ['OBV_RISING', 'OBV_FALLING', 'OBV_STABLE']
        for pattern in expected_patterns:
            self.assertIn(pattern, patterns.columns, f"缺少形态: {pattern}")
        
        # 验证形态数据类型
        for pattern in expected_patterns:
            if pattern in patterns.columns:
                self.assertTrue(pd.api.types.is_bool_dtype(patterns[pattern]) or 
                              patterns[pattern].dtype == 'object', 
                              f"形态 {pattern} 应该是布尔类型")
    
    def test_obv_signals(self):
        """测试OBV信号生成"""
        signals = self.indicator.get_signals(self.data)
        
        # 验证信号格式
        self.assertIsInstance(signals, dict)
        expected_signal_keys = ['buy_signal', 'sell_signal', 'hold_signal', 'signal_strength']
        for key in expected_signal_keys:
            self.assertIn(key, signals, f"缺少信号键: {key}")
            self.assertIsInstance(signals[key], pd.Series, f"信号 {key} 应该是Series类型")
    
    def test_obv_raw_score(self):
        """测试OBV原始评分"""
        score = self.indicator.calculate_raw_score(self.data)
        
        # 验证评分类型
        self.assertIsInstance(score, pd.Series)
        
        # 验证评分范围（通常在0-100之间）
        valid_scores = score.dropna()
        if len(valid_scores) > 0:
            self.assertTrue(all(0 <= s <= 100 for s in valid_scores), 
                          f"评分应在0-100范围内，实际范围: {valid_scores.min()}-{valid_scores.max()}")
    
    def test_obv_comprehensive_score(self):
        """测试OBV综合评分"""
        score_result = self.indicator.calculate_score(self.data)
        
        # 验证评分结果格式
        self.assertIsInstance(score_result, dict)
        self.assertIn('latest_score', score_result)
        self.assertIn('confidence', score_result)
        
        # 验证评分范围
        self.assertGreaterEqual(score_result['latest_score'], 0.0)
        self.assertLessEqual(score_result['latest_score'], 100.0)
        
        # 验证置信度范围
        self.assertGreaterEqual(score_result['confidence'], 0.0)
        self.assertLessEqual(score_result['confidence'], 1.0)
    
    def test_obv_confidence_calculation(self):
        """测试OBV置信度计算"""
        # 先计算指标
        result = self.indicator.calculate(self.data)
        score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        signals = self.indicator.get_signals(self.data)
        
        confidence = self.indicator.calculate_confidence(score, patterns, signals)
        
        # 验证置信度类型和范围
        self.assertIsInstance(confidence, (float, np.floating))
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_obv_set_parameters(self):
        """测试OBV参数设置"""
        new_signal_period = 15
        
        self.indicator.set_parameters(signal_period=new_signal_period)
        self.assertEqual(self.indicator.signal_period, new_signal_period)
    
    def test_obv_volume_logic(self):
        """测试OBV成交量逻辑"""
        # 创建特定的测试数据来验证OBV逻辑
        test_data = pd.DataFrame({
            'open': [100, 100, 100, 100, 100],
            'high': [105, 105, 105, 105, 105], 
            'low': [95, 95, 95, 95, 95],
            'close': [100, 101, 99, 102, 98],  # 价格变化：不变, 上涨, 下跌, 上涨, 下跌
            'volume': [1000, 1000, 1000, 1000, 1000]
        })
        
        result = self.indicator.calculate(test_data)
        obv_values = result['OBV'].values
        
        # 验证OBV逻辑
        self.assertEqual(obv_values[0], 0)  # 初始值为0
        self.assertEqual(obv_values[1], 1000)  # 价格上涨，OBV += volume
        self.assertEqual(obv_values[2], 0)  # 价格下跌，OBV -= volume
        self.assertEqual(obv_values[3], 1000)  # 价格上涨，OBV += volume
        self.assertEqual(obv_values[4], 0)  # 价格下跌，OBV -= volume
    
    def test_obv_edge_cases(self):
        """测试OBV边界情况"""
        # 测试数据不足的情况
        short_data = pd.DataFrame({
            'open': [100],
            'high': [105],
            'low': [95],
            'close': [100],
            'volume': [1000]
        })
        
        result = self.indicator.calculate(short_data)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)
    
    def test_obv_no_error_logs(self):
        """测试OBV计算过程中无ERROR日志"""
        # 执行计算
        result = self.indicator.calculate(self.data)
        
        # 验证无ERROR日志
        self.assertNoLogs('ERROR')
        
        # 验证结果
        self.assertIsInstance(result, pd.DataFrame)
    
    def test_obv_patterns_no_error_logs(self):
        """测试OBV形态检测过程中无ERROR日志"""
        # 执行形态检测
        patterns = self.indicator.get_patterns(self.data)
        
        # 验证无ERROR日志
        self.assertNoLogs('ERROR')
        
        # 验证结果
        self.assertIsInstance(patterns, pd.DataFrame)
    
    def test_obv_compatibility_methods(self):
        """测试OBV兼容性方法"""
        # 测试compute方法（calculate的别名）
        result1 = self.indicator.calculate(self.data)
        result2 = self.indicator.compute(self.data)
        
        # 结果应该相同
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_obv_has_result(self):
        """测试OBV结果检查方法"""
        # 初始状态应该没有结果
        self.assertFalse(self.indicator.has_result())
        
        # 计算后应该有结果
        self.indicator.calculate(self.data)
        self.assertTrue(self.indicator.has_result())


if __name__ == '__main__':
    unittest.main() 