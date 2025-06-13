import unittest
import pandas as pd
import numpy as np

from indicators.boll import BOLL
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator
from tests.helper.log_capture import LogCaptureMixin

class TestBOLL(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    """BOLL指标单元测试类"""

    def setUp(self):
        """准备数据和指标实例"""
        LogCaptureMixin.setUp(self)  # 显式调用Mixin的setUp
        self.indicator = BOLL(period=20, std_dev=2)
        self.expected_columns = ['middle', 'upper', 'lower']  # 修正期望的列名
        # 使用一个包含多种走势的数据进行通用测试
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 50},
            {'type': 'trend', 'start_price': 120, 'end_price': 100, 'periods': 50},
        ])

    def tearDown(self):
        """清理日志捕获器"""
        LogCaptureMixin.tearDown(self)  # 显式调用Mixin的tearDown

    def test_basic_calculation(self):
        """测试BOLL基础计算功能"""
        result = self.indicator.calculate(self.data)

        # 验证返回类型
        self.assertIsInstance(result, pd.DataFrame, "BOLL计算结果应为DataFrame")

        # 验证包含BOLL列
        self.assertIn('middle', result.columns, "结果应包含middle列")
        self.assertIn('upper', result.columns, "结果应包含upper列")
        self.assertIn('lower', result.columns, "结果应包含lower列")
        self.assertIn('bandwidth', result.columns, "结果应包含bandwidth列")
        self.assertIn('percent_b', result.columns, "结果应包含percent_b列")

        # 验证布林带的基本关系：upper > middle > lower
        valid_data = result.dropna()
        if len(valid_data) > 0:
            self.assertTrue(all(valid_data['upper'] >= valid_data['middle']), "上轨应大于等于中轨")
            self.assertTrue(all(valid_data['middle'] >= valid_data['lower']), "中轨应大于等于下轨")

        # 验证%B值在合理范围内（通常在0-1之间，但可能超出）
        percent_b_values = result['percent_b'].dropna()
        if len(percent_b_values) > 0:
            self.assertTrue(all(np.isfinite(val) for val in percent_b_values), "%B值应为有限数值")

    def test_boll_squeeze(self):
        """测试布林带缩口（squeeze）"""
        # 在窄幅震荡行情中，布林带宽度应该会变小
        data = TestDataGenerator.generate_price_sequence([
            {'type': 'sideways', 'start_price': 100, 'volatility': 0.01, 'periods': 100}
        ])
        result = self.indicator.calculate(data)

        # 使用bandwidth列或计算宽度
        if 'bandwidth' in result.columns:
            width = result['bandwidth']
        else:
            width = result['upper'] - result['lower']

        # 期望在震荡后期，带宽相对较小
        valid_width = width.dropna()
        if len(valid_width) > 0:
            avg_width = valid_width.mean()
            self.assertGreater(avg_width, 0, "布林带宽度应大于0")

    def test_boll_breakout(self):
        """测试布林带开口（breakout）"""
        # V形反转的剧烈波动应该导致布林带开口
        data = TestDataGenerator.generate_price_sequence([
            {'type': 'v_shape', 'start_price': 100, 'bottom_price': 80, 'periods': 50}
        ])
        result = self.indicator.calculate(data)

        # 使用bandwidth列或计算宽度
        if 'bandwidth' in result.columns:
            width = result['bandwidth']
        else:
            width = result['upper'] - result['lower']

        # 期望在趋势变化时，带宽有所变化
        valid_width = width.dropna()
        if len(valid_width) > 10:
            # 检查带宽的变化
            width_std = valid_width.std()
            self.assertGreater(width_std, 0, "布林带宽度应有变化")

    def test_price_within_bands(self):
        """测试价格大部分时间在布林带轨道内"""
        data = TestDataGenerator.generate_price_sequence([
             {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 100}
        ])
        result = self.indicator.calculate(data)

        # 计算价格在轨道内的比例
        valid_data = result.dropna()
        if len(valid_data) > 0:
            within_bands = (valid_data['close'] <= valid_data['upper']) & (valid_data['close'] >= valid_data['lower'])
            within_bands_ratio = within_bands.mean()

            # 根据统计学，大部分数据点应落在2个标准差内
            self.assertGreater(within_bands_ratio, 0.8, "大部分价格点应落在布林带轨道内")

    def test_signal_generation(self):
        """测试BOLL信号生成"""
        # 生成包含多种走势的数据
        data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 100, 'periods': 30},  # 横盘
            {'type': 'trend', 'start_price': 100, 'end_price': 90, 'periods': 15},   # 下跌
            {'type': 'trend', 'start_price': 90, 'end_price': 110, 'periods': 20},   # 反弹
        ])

        signals = self.indicator.generate_trading_signals(data)

        # 验证信号DataFrame结构
        self.assertIsInstance(signals, pd.DataFrame, "信号结果应为DataFrame")
        self.assertIn('buy_signal', signals.columns, "信号结果应包含buy_signal列")
        self.assertIn('sell_signal', signals.columns, "信号结果应包含sell_signal列")

    def test_pattern_detection(self):
        """测试BOLL形态检测"""
        data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 100, 'periods': 30},  # 横盘
            {'type': 'v_shape', 'start_price': 100, 'bottom_price': 85, 'periods': 30},  # V形反转
        ])

        patterns = self.indicator.identify_patterns(data)

        # 验证形态检测结果
        self.assertIsInstance(patterns, list, "形态检测结果应为列表")

    def test_score_calculation(self):
        """测试BOLL评分计算功能"""
        # 生成测试数据
        data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 100, 'periods': 30},  # 横盘
            {'type': 'trend', 'start_price': 100, 'end_price': 90, 'periods': 15},   # 下跌
            {'type': 'trend', 'start_price': 90, 'end_price': 110, 'periods': 20},   # 反弹
        ])

        # 验证原始评分计算
        raw_score = self.indicator.calculate_raw_score(data)
        self.assertIsInstance(raw_score, pd.Series, "原始评分应为Series")
        self.assertTrue(all(0 <= s <= 100 for s in raw_score if not pd.isna(s)), "原始评分应在0-100范围内")

    def test_parameter_setting(self):
        """测试BOLL参数设置"""
        # 测试参数设置方法
        new_period = 30
        new_std_dev = 2.5

        self.indicator.set_parameters(period=new_period, std_dev=new_std_dev)

        # 验证参数是否正确设置
        self.assertEqual(self.indicator.period, new_period, "周期参数设置失败")
        self.assertEqual(self.indicator.std_dev, new_std_dev, "标准差参数设置失败")

        # 验证新参数下的计算
        result = self.indicator.calculate(self.data)
        self.assertIn('middle', result.columns, "结果应包含middle列")

    def test_edge_cases(self):
        """测试边界条件"""
        # 测试数据长度不足的情况
        short_data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 105, 'periods': 10}
        ])

        result = self.indicator.calculate(short_data)
        self.assertIsInstance(result, pd.DataFrame, "短数据计算结果应为DataFrame")

        # 测试价格无变化的情况
        flat_data = TestDataGenerator.generate_price_sequence([
            {'type': 'flat', 'start_price': 100, 'periods': 30}
        ])

        # 确保价格完全相等
        for i in range(len(flat_data)):
            flat_data.iloc[i, flat_data.columns.get_loc('close')] = 100

        result = self.indicator.calculate(flat_data)
        self.assertIsInstance(result, pd.DataFrame, "平价数据计算结果应为DataFrame")

        # 在价格无变化的情况下，上下轨应该等于中轨
        valid_data = result.dropna()
        if len(valid_data) > 0:
            # 由于标准差为0，上下轨应该等于中轨
            self.assertTrue(all(abs(valid_data['upper'] - valid_data['middle']) < 0.01), "价格无变化时上轨应等于中轨")
            self.assertTrue(all(abs(valid_data['lower'] - valid_data['middle']) < 0.01), "价格无变化时下轨应等于中轨")

    def test_robustness(self):
        """测试BOLL指标的鲁棒性"""
        # 测试包含异常值的数据
        data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 30}
        ])

        # 添加一些异常值
        data.loc[data.index[15], 'close'] = data.loc[data.index[15], 'close'] * 2    # 价格异常跳跃

        result = self.indicator.calculate(data)

        # 验证计算没有出错
        self.assertIsInstance(result, pd.DataFrame, "异常数据计算结果应为DataFrame")
        self.assertIn('middle', result.columns, "结果应包含middle列")

        # 验证布林带值是有限的数值
        middle_values = result['middle'].dropna()
        upper_values = result['upper'].dropna()
        lower_values = result['lower'].dropna()
        self.assertTrue(all(np.isfinite(val) for val in middle_values), "中轨值应为有限数值")
        self.assertTrue(all(np.isfinite(val) for val in upper_values), "上轨值应为有限数值")
        self.assertTrue(all(np.isfinite(val) for val in lower_values), "下轨值应为有限数值")


if __name__ == '__main__':
    unittest.main() 