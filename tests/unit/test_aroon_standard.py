"""
Aroon指标单元测试
"""
import unittest
import pandas as pd
import numpy as np
import io
import logging

from indicators.aroon import Aroon
from utils.logger import get_logger
from tests.helper.log_capture import LogCaptureMixin

logger = get_logger(__name__)


class TestAroon(unittest.TestCase, LogCaptureMixin):
    """Aroon指标测试类"""

    def setUp(self):
        """设置测试环境"""
        # 设置日志捕获
        self.log_stream = io.StringIO()
        self.log_handler = logging.StreamHandler(self.log_stream)
        self.logger = logging.getLogger()
        self.logger.addHandler(self.log_handler)
        self.logger.setLevel(logging.DEBUG)

        self.aroon = Aroon(period=14)
        
        # 创建测试数据
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        
        # 生成上升趋势数据用于测试
        price_trend = np.linspace(100, 120, 50)
        noise = np.random.normal(0, 0.5, 50)
        
        self.test_data = pd.DataFrame({
            'open': price_trend + noise,
            'high': price_trend + np.abs(noise) + 1,
            'low': price_trend - np.abs(noise) - 1,
            'close': price_trend + noise * 0.5,
            'volume': np.random.randint(1000000, 5000000, 50)
        }, index=dates)
        
        # 确保高低价顺序正确
        self.test_data['high'] = np.maximum.reduce([
            self.test_data['open'], self.test_data['high'], 
            self.test_data['low'], self.test_data['close']
        ])
        self.test_data['low'] = np.minimum.reduce([
            self.test_data['open'], self.test_data['high'], 
            self.test_data['low'], self.test_data['close']
        ])

    def tearDown(self):
        """清理测试环境"""
        if hasattr(self, 'log_handler') and self.log_handler:
            self.logger.removeHandler(self.log_handler)
            self.log_handler.close()

    def test_aroon_initialization(self):
        """测试Aroon初始化"""
        aroon = Aroon()
        self.assertEqual(aroon.name, "AROON")
        self.assertEqual(aroon.period, 14)
        self.assertIsNone(aroon._result)
        
        # 测试自定义参数
        aroon_custom = Aroon(period=21)
        self.assertEqual(aroon_custom.period, 21)

    def test_aroon_calculation(self):
        """测试Aroon计算功能"""
        result = self.aroon.calculate(self.test_data)
        
        # 验证返回类型
        self.assertIsInstance(result, pd.DataFrame)
        
        # 验证包含必要列
        required_columns = ['aroon_up', 'aroon_down', 'aroon_oscillator']
        for col in required_columns:
            self.assertIn(col, result.columns)
        
        # 验证Aroon值范围（0-100）
        aroon_up_values = result['aroon_up'].dropna()
        aroon_down_values = result['aroon_down'].dropna()
        
        if len(aroon_up_values) > 0:
            self.assertTrue(all(0 <= v <= 100 for v in aroon_up_values))
        if len(aroon_down_values) > 0:
            self.assertTrue(all(0 <= v <= 100 for v in aroon_down_values))

    def test_aroon_patterns(self):
        """测试Aroon形态识别"""
        result = self.aroon.calculate(self.test_data)
        patterns = self.aroon.get_patterns(self.test_data)
        
        # 验证返回DataFrame
        self.assertIsInstance(patterns, pd.DataFrame)
        
        # 验证基本形态列存在
        expected_patterns = [
            'AROON_UPTREND', 'AROON_DOWNTREND', 'AROON_CONSOLIDATION',
            'AROON_BULLISH_CROSS', 'AROON_BEARISH_CROSS'
        ]
        
        for pattern in expected_patterns:
            self.assertIn(pattern, patterns.columns)

    def test_aroon_signals(self):
        """测试Aroon信号生成"""
        signals = self.aroon.get_signals(self.test_data)
        
        # 验证信号字典结构
        self.assertIsInstance(signals, dict)
        expected_signal_keys = ['aroon_buy_signal', 'aroon_sell_signal']
        for key in expected_signal_keys:
            self.assertIn(key, signals)
            self.assertIsInstance(signals[key], pd.Series)

    def test_aroon_score_calculation(self):
        """测试Aroon评分计算"""
        score = self.aroon.calculate_raw_score(self.test_data)
        
        # 验证评分类型和范围
        self.assertIsInstance(score, pd.Series)
        valid_scores = score.dropna()
        if len(valid_scores) > 0:
            self.assertTrue(all(0 <= s <= 100 for s in valid_scores))

    def test_aroon_confidence_calculation(self):
        """测试Aroon置信度计算"""
        confidence = self.aroon.calculate_confidence(self.test_data)
        
        # 验证置信度类型和范围
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_aroon_parameter_setting(self):
        """测试Aroon参数设置"""
        new_period = 21
        self.aroon.set_parameters(period=new_period)
        
        # 验证参数更新
        self.assertEqual(self.aroon.period, new_period)

    def test_aroon_uptrend_detection(self):
        """测试上升趋势检测"""
        # 生成明确的上升趋势数据
        uptrend_dates = pd.date_range('2023-01-01', periods=30, freq='D')
        uptrend_prices = np.linspace(100, 130, 30)
        
        uptrend_data = pd.DataFrame({
            'open': uptrend_prices,
            'high': uptrend_prices + 2,
            'low': uptrend_prices - 2,
            'close': uptrend_prices + 1,
            'volume': np.random.randint(1000000, 5000000, 30)
        }, index=uptrend_dates)
        
        result = self.aroon.calculate(uptrend_data)
        
        # 在上升趋势中，Aroon Up应该较高
        aroon_up = result['aroon_up'].dropna()
        if len(aroon_up) > 5:  # 确保有足够的数据
            # 最后几个值应该相对较高
            recent_aroon_up = aroon_up.iloc[-5:].mean()
            self.assertGreater(recent_aroon_up, 30)  # 上升趋势中AROON UP应该 > 30

    def test_aroon_downtrend_detection(self):
        """测试下降趋势检测"""
        # 生成明确的下降趋势数据
        downtrend_dates = pd.date_range('2023-01-01', periods=30, freq='D')
        downtrend_prices = np.linspace(130, 100, 30)
        
        downtrend_data = pd.DataFrame({
            'open': downtrend_prices,
            'high': downtrend_prices + 2,
            'low': downtrend_prices - 2,
            'close': downtrend_prices - 1,
            'volume': np.random.randint(1000000, 5000000, 30)
        }, index=downtrend_dates)
        
        result = self.aroon.calculate(downtrend_data)
        
        # 在下降趋势中，Aroon Down应该较高
        aroon_down = result['aroon_down'].dropna()
        if len(aroon_down) > 5:  # 确保有足够的数据
            # 最后几个值应该相对较高
            recent_aroon_down = aroon_down.iloc[-5:].mean()
            self.assertGreater(recent_aroon_down, 30)  # 下降趋势中AROON DOWN应该 > 30

    def test_aroon_oscillator_values(self):
        """测试Aroon震荡器值"""
        result = self.aroon.calculate(self.test_data)
        
        # 验证震荡器计算正确性
        if 'aroon_up' in result.columns and 'aroon_down' in result.columns:
            calculated_osc = result['aroon_up'] - result['aroon_down']
            aroon_osc = result['aroon_oscillator']
            
            # 比较计算结果（允许小的浮点误差）
            diff = abs(calculated_osc - aroon_osc).dropna()
            if len(diff) > 0:
                self.assertTrue(all(d < 1e-10 for d in diff))

    def test_aroon_edge_cases(self):
        """测试Aroon边界情况"""
        # 测试数据不足的情况
        small_data = self.test_data.head(5)
        result = self.aroon.calculate(small_data)
        
        # 应该能处理小数据集
        self.assertIsInstance(result, pd.DataFrame)

    def test_aroon_with_missing_data(self):
        """测试包含缺失数据的情况"""
        data_with_nan = self.test_data.copy()
        data_with_nan.loc[data_with_nan.index[10:15], 'high'] = np.nan
        
        result = self.aroon.calculate(data_with_nan)
        
        # 应该能处理NaN值
        self.assertIsInstance(result, pd.DataFrame)

    def test_aroon_all_interfaces(self):
        """测试所有公共接口"""
        # 测试主要计算接口
        result1 = self.aroon.calculate(self.test_data)
        result2 = self.aroon.compute(self.test_data)
        
        # 两种计算方式应该返回相同结果
        self.assertIsInstance(result1, pd.DataFrame)
        self.assertIsInstance(result2, pd.DataFrame)
        
        # 测试信号生成接口
        signals1 = self.aroon.get_signals(self.test_data)
        signals2 = self.aroon.generate_trading_signals(self.test_data)
        
        self.assertIsInstance(signals1, dict)
        self.assertIsInstance(signals2, dict)
        
        # 测试评分接口
        score1 = self.aroon.calculate_raw_score(self.test_data)
        score2 = self.aroon.calculate_score(self.test_data)
        
        self.assertIsInstance(score1, pd.Series)
        self.assertIsInstance(score2, pd.Series)

    def test_aroon_no_errors_during_calculation(self):
        """测试计算过程中无ERROR日志"""
        # 清除之前的日志
        self.log_stream.seek(0)
        self.log_stream.truncate(0)
        
        # 执行计算
        result = self.aroon.calculate(self.test_data)
        
        # 检查日志中是否有ERROR
        log_contents = self.log_stream.getvalue()
        self.assertNotIn('ERROR', log_contents)
        
        # 验证结果
        self.assertIsInstance(result, pd.DataFrame)

    def test_aroon_no_errors_during_pattern_detection(self):
        """测试形态检测过程中无ERROR日志"""
        # 清除之前的日志
        self.log_stream.seek(0)
        self.log_stream.truncate(0)
        
        # 执行形态检测
        patterns = self.aroon.get_patterns(self.test_data)
        
        # 检查日志中是否有ERROR
        log_contents = self.log_stream.getvalue()
        self.assertNotIn('ERROR', log_contents)
        
        # 验证结果
        self.assertIsInstance(patterns, pd.DataFrame)

    def test_aroon_register_patterns(self):
        """测试Aroon形态注册"""
        # 调用形态注册
        self.aroon.register_patterns()
        
        # 验证形态注册成功完成（通过检查是否有异常抛出）
        self.assertTrue(True)

    def test_aroon_has_result(self):
        """测试结果状态检查"""
        # 初始状态应该没有结果
        self.assertFalse(self.aroon.has_result())
        
        # 计算后应该有结果
        self.aroon.calculate(self.test_data)
        self.assertTrue(self.aroon.has_result())


if __name__ == '__main__':
    unittest.main() 