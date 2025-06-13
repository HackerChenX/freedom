import unittest
import pandas as pd
import numpy as np

from indicators.rsi import RSI
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator
from tests.helper.log_capture import LogCaptureMixin


class TestRSI(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    """RSI指标单元测试类"""

    def setUp(self):
        """准备数据和指标实例"""
        LogCaptureMixin.setUp(self)  # 显式调用Mixin的setUp
        self.indicator = RSI(period=14, ma_periods=[5, 10], overbought=70.0, oversold=30.0)
        self.expected_columns = [f'rsi_{self.indicator.period}']
        # 使用一个包含多种走势的数据进行通用测试
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 30},
            {'type': 'trend', 'start_price': 120, 'end_price': 100, 'periods': 30},
        ])

    def tearDown(self):
        """清理日志捕获器"""
        LogCaptureMixin.tearDown(self)  # 显式调用Mixin的tearDown

    def clear_logs_before_test(self):
        """在测试前清除日志"""
        self.clear_logs()

    def test_basic_calculation(self):
        """测试RSI基础计算功能"""
        result = self.indicator.calculate(self.data)

        # 验证返回类型
        self.assertIsInstance(result, pd.DataFrame, "RSI计算结果应为DataFrame")

        # 验证包含RSI列
        rsi_col = f'rsi_{self.indicator.period}'
        self.assertIn(rsi_col, result.columns, f"结果应包含{rsi_col}列")

        # 验证RSI值在0-100范围内
        rsi_values = result[rsi_col].dropna()
        self.assertTrue(all(0 <= val <= 100 for val in rsi_values), "RSI值应在0-100范围内")

        # 验证RSI均线列
        if self.indicator.ma_periods:
            ma_short_col = f'rsi_ma_{self.indicator.ma_periods[0]}'
            ma_long_col = f'rsi_ma_{self.indicator.ma_periods[1]}'
            self.assertIn(ma_short_col, result.columns, f"结果应包含{ma_short_col}列")
            self.assertIn(ma_long_col, result.columns, f"结果应包含{ma_long_col}列")

    def test_rsi_overbought_detection(self):
        """测试RSI超买检测"""
        # 生成强劲上涨趋势数据
        data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 100, 'periods': 20},  # 横盘
            {'type': 'trend', 'start_price': 100, 'end_price': 150, 'periods': 15},  # 快速上涨
        ])
        result = self.indicator.calculate(data)

        # RSI在持续上涨后应超过70
        rsi_col = f'rsi_{self.indicator.period}'
        self.assertTrue((result[rsi_col].iloc[-5:] > 70).any(), "RSI未进入超买区")

        # 验证超买标记
        self.assertTrue(result['rsi_overbought'].iloc[-5:].any(), "未检测到超买状态")

    def test_rsi_oversold_detection(self):
        """测试RSI超卖检测"""
        # 生成强劲下跌趋势数据
        data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 100, 'periods': 20},  # 横盘
            {'type': 'trend', 'start_price': 100, 'end_price': 70, 'periods': 15},   # 快速下跌
        ])
        result = self.indicator.calculate(data)

        # RSI在持续下跌后应低于30
        rsi_col = f'rsi_{self.indicator.period}'
        self.assertTrue((result[rsi_col].iloc[-5:] < 30).any(), "RSI未进入超卖区")

        # 验证超卖标记
        self.assertTrue(result['rsi_oversold'].iloc[-5:].any(), "未检测到超卖状态")

    def test_rsi_neutral_zone(self):
        """测试RSI在中性区域"""
        # 生成横盘数据
        data = TestDataGenerator.generate_price_sequence([
            {'type': 'sideways', 'start_price': 100, 'periods': 50, 'volatility': 0.01}
        ])
        result = self.indicator.calculate(data)

        # 在横盘行情中，RSI应该在30和70之间波动
        rsi_col = f'rsi_{self.indicator.period}'
        rsi_values = result[rsi_col].dropna()
        neutral_values = rsi_values[(rsi_values >= 30) & (rsi_values <= 70)]
        self.assertGreater(len(neutral_values), len(rsi_values) * 0.7, "大部分RSI值应在中性区域")

    def test_golden_cross(self):
        """测试RSI均线金叉"""
        data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 100, 'periods': 20},  # 横盘
            {'type': 'trend', 'start_price': 100, 'end_price': 90, 'periods': 10},   # 下跌
            {'type': 'trend', 'start_price': 90, 'end_price': 105, 'periods': 15},   # 反弹
        ])
        calculated_data = self.indicator.calculate(data)
        patterns = self.indicator.get_patterns(data)

        self.assertIn('RSI_GOLDEN_CROSS', patterns.columns, "模式结果中缺少 RSI_GOLDEN_CROSS 列")
        # 在反弹过程中应该检测到金叉
        self.assertGreater(patterns['RSI_GOLDEN_CROSS'].sum(), 0, "未检测到RSI金叉")

    def test_death_cross(self):
        """测试RSI均线死叉"""
        data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 100, 'periods': 20},  # 横盘
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 10},  # 上涨
            {'type': 'trend', 'start_price': 110, 'end_price': 95, 'periods': 15},   # 回调
        ])
        calculated_data = self.indicator.calculate(data)
        patterns = self.indicator.get_patterns(data)

        self.assertIn('RSI_DEATH_CROSS', patterns.columns, "模式结果中缺少 RSI_DEATH_CROSS 列")
        # 在回调过程中应该检测到死叉
        self.assertGreater(patterns['RSI_DEATH_CROSS'].sum(), 0, "未检测到RSI死叉")

    def test_overbought_oversold_patterns(self):
        """测试超买超卖形态检测"""
        # 生成超买数据
        overbought_data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 100, 'periods': 20},  # 横盘
            {'type': 'trend', 'start_price': 100, 'end_price': 150, 'periods': 15},  # 快速上涨
        ])
        overbought_result = self.indicator.calculate(overbought_data)
        overbought_patterns = self.indicator.get_patterns(overbought_data)

        # 检查超买形态
        self.assertIn('RSI_OVERBOUGHT', overbought_patterns.columns, "模式结果中缺少 RSI_OVERBOUGHT 列")
        self.assertTrue(overbought_patterns['RSI_OVERBOUGHT'].iloc[-5:].any(), "未检测到超买形态")

        # 生成超卖数据
        oversold_data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 100, 'periods': 20},  # 横盘
            {'type': 'trend', 'start_price': 100, 'end_price': 70, 'periods': 15},   # 快速下跌
        ])
        oversold_result = self.indicator.calculate(oversold_data)
        oversold_patterns = self.indicator.get_patterns(oversold_data)

        # 检查超卖形态
        self.assertIn('RSI_OVERSOLD', oversold_patterns.columns, "模式结果中缺少 RSI_OVERSOLD 列")
        self.assertTrue(oversold_patterns['RSI_OVERSOLD'].iloc[-5:].any(), "未检测到超卖形态")

    def test_signal_generation(self):
        """测试RSI信号生成"""
        # 生成包含多种走势的数据
        data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 100, 'periods': 20},  # 横盘
            {'type': 'trend', 'start_price': 100, 'end_price': 70, 'periods': 10},   # 下跌到超卖
            {'type': 'trend', 'start_price': 70, 'end_price': 120, 'periods': 15},   # 反弹到超买
        ])

        signals = self.indicator.generate_signals(data)

        # 验证信号DataFrame结构
        self.assertIsInstance(signals, pd.DataFrame, "信号结果应为DataFrame")
        self.assertIn('buy_signal', signals.columns, "信号结果应包含buy_signal列")
        self.assertIn('sell_signal', signals.columns, "信号结果应包含sell_signal列")

        # 验证信号逻辑
        # 在超卖区域应该有买入信号
        self.assertTrue(signals['buy_signal'].any(), "应该检测到买入信号")
        # 在超买区域应该有卖出信号
        self.assertTrue(signals['sell_signal'].any(), "应该检测到卖出信号")

    def test_score_calculation(self):
        """测试RSI评分计算功能"""
        # 生成测试数据
        data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 100, 'periods': 20},  # 横盘
            {'type': 'trend', 'start_price': 100, 'end_price': 70, 'periods': 10},   # 快速下跌
            {'type': 'trend', 'start_price': 70, 'end_price': 120, 'periods': 15},   # 快速上涨
        ])

        # 计算RSI
        result = self.indicator.calculate(data)

        # 将RSI结果合并到原始数据中
        data_with_rsi = data.copy()
        for col in result.columns:
            data_with_rsi[col] = result[col]

        # 计算评分
        score_result = self.indicator.calculate_score(data_with_rsi)

        # 验证评分类型和范围
        self.assertIsInstance(score_result, dict, "评分结果应为字典")
        self.assertIn('score', score_result, "评分结果应包含score键")
        self.assertIn('confidence', score_result, "评分结果应包含confidence键")

        score = score_result['score']
        confidence = score_result['confidence']
        self.assertIsInstance(score, (int, float), "评分应为数值")
        self.assertIsInstance(confidence, (int, float), "置信度应为数值")
        self.assertTrue(0 <= score <= 100, "评分应在0-100范围内")
        self.assertTrue(0 <= confidence <= 1, "置信度应在0-1范围内")

        # 验证原始评分计算
        raw_score = self.indicator.calculate_raw_score(data_with_rsi)
        self.assertIsInstance(raw_score, pd.Series, "原始评分应为Series")
        self.assertTrue(all(0 <= s <= 100 for s in raw_score if not pd.isna(s)), "原始评分应在0-100范围内")

    def test_parameter_setting(self):
        """测试RSI参数设置"""
        # 测试参数设置方法
        new_period = 21
        new_overbought = 75.0
        new_oversold = 25.0
        new_ma_periods = [3, 7]

        self.indicator.set_parameters(
            period=new_period,
            overbought=new_overbought,
            oversold=new_oversold,
            ma_periods=new_ma_periods
        )

        # 验证参数是否正确设置
        self.assertEqual(self.indicator.period, new_period, "周期参数设置失败")
        self.assertEqual(self.indicator.overbought, new_overbought, "超买阈值设置失败")
        self.assertEqual(self.indicator.oversold, new_oversold, "超卖阈值设置失败")
        self.assertEqual(self.indicator.ma_periods, new_ma_periods, "均线周期设置失败")

        # 验证新参数下的计算
        result = self.indicator.calculate(self.data)
        rsi_col = f'rsi_{new_period}'
        self.assertIn(rsi_col, result.columns, f"结果应包含{rsi_col}列")

    def test_edge_cases(self):
        """测试边界条件"""
        # 测试数据长度不足的情况
        short_data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 105, 'periods': 5}
        ])

        result = self.indicator.calculate(short_data)
        self.assertIsInstance(result, pd.DataFrame, "短数据计算结果应为DataFrame")

        # 测试价格无变化的情况
        flat_data = TestDataGenerator.generate_price_sequence([
            {'type': 'flat', 'start_price': 100, 'periods': 30}
        ])

        result = self.indicator.calculate(flat_data)
        self.assertIsInstance(result, pd.DataFrame, "平价数据计算结果应为DataFrame")

        # 测试空数据 - 应该抛出异常或返回空DataFrame
        empty_data = pd.DataFrame()
        try:
            result = self.indicator.calculate(empty_data)
            self.assertIsInstance(result, pd.DataFrame, "空数据计算结果应为DataFrame")
        except ValueError:
            # 空数据抛出异常是可以接受的
            pass

    def test_robustness(self):
        """测试RSI指标的鲁棒性"""
        # 测试包含异常值的数据
        data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 20}
        ])

        # 添加一些异常值
        data.loc[data.index[10], 'close'] = data.loc[data.index[10], 'close'] * 2  # 价格翻倍
        data.loc[data.index[15], 'close'] = data.loc[data.index[15], 'close'] * 0.5  # 价格减半

        # 重新计算OHLV以保持一致性
        for i in [10, 15]:
            idx = data.index[i]
            close_price = data.loc[idx, 'close']
            data.loc[idx, 'open'] = close_price * 0.99
            data.loc[idx, 'high'] = close_price * 1.01
            data.loc[idx, 'low'] = close_price * 0.98

        result = self.indicator.calculate(data)

        # 验证计算没有出错
        self.assertIsInstance(result, pd.DataFrame, "异常数据计算结果应为DataFrame")
        rsi_col = f'rsi_{self.indicator.period}'
        self.assertIn(rsi_col, result.columns, f"结果应包含{rsi_col}列")

        # 验证RSI值仍在合理范围内
        rsi_values = result[rsi_col].dropna()
        self.assertTrue(all(0 <= val <= 100 for val in rsi_values), "即使有异常值，RSI值也应在0-100范围内")


if __name__ == '__main__':
    unittest.main()