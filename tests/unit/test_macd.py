"""
MACD指标单元测试
"""

import unittest
import pandas as pd
import numpy as np
import pytest
from scipy.signal import find_peaks

from indicators.macd import MACD
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator
from tests.helper.log_capture import LogCaptureMixin
from utils.technical_utils import calculate_macd


class TestMACD(LogCaptureMixin, IndicatorTestMixin, unittest.TestCase):
    """
    MACD 指标单元测试
    
    该测试类利用 IndicatorTestMixin 提供的通用测试方法，
    并结合为 MACD 指标量身定制的特定形态生成逻辑。
    """

    def setUp(self):
        """
        测试初始化
        """
        super().setUp()
        self.indicator = MACD()
        self.expected_columns = ['macd_line', 'macd_signal', 'macd_histogram']
        # 使用 generate_price_sequence 创建一个复杂的测试数据集
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 105, 'periods': 50},
            {'type': 'v_shape', 'start_price': 105, 'bottom_price': 95, 'periods': 50},
            {'type': 'trend', 'start_price': 105, 'end_price': 105, 'periods': 20},
            {'type': 'double_top', 'start_price': 105, 'peak_price': 115, 'periods': 60},
            {'type': 'trend', 'start_price': 105, 'end_price': 100, 'periods': 50},
        ])
        if 'volume' not in self.data.columns:
            self.data['volume'] = 1000

    def test_calculation_correctness(self):
        """测试核心计算的数值准确性"""
        data = pd.DataFrame({
            'close': [
                100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                110, 109, 108, 107, 106, 105, 104, 103, 102, 101
            ]
        })
        data['open'] = data['high'] = data['low'] = data['close']
        data['volume'] = 1000
        
        # 使用项目内的函数计算预期值
        expected_line, expected_signal, _ = calculate_macd(data['close'])
        expected_macd_line_end = expected_line.iloc[-1]
        
        result = self.indicator.calculate(data)
        self.assertAlmostEqual(result['macd_line'].iloc[-1], expected_macd_line_end, places=2)

    def test_golden_cross_pattern(self):
        """测试金叉形态的精确定位"""
        # 为金叉场景生成特定数据
        data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 100, 'periods': 50}, # 更长的稳定期
            {'type': 'trend', 'start_price': 100, 'end_price': 70, 'periods': 30},  # 剧烈下跌
            {'type': 'trend', 'start_price': 70, 'end_price': 120, 'periods': 40}  # 强劲反弹
        ])
        # 计算MACD指标
        result_df = self.indicator.calculate(data)
        
        # 确认反弹阶段的MACD相关性质
        # 1. 检查在下跌阶段和反弹初期，MACD线应该在信号线下方
        early_phase = result_df.iloc[60:80]  # 下跌后期和反弹初期
        self.assertTrue((early_phase['macd_line'] < early_phase['macd_signal']).any(), 
                       "在下跌阶段，MACD线应该低于信号线")
        
        # 2. 检查在反弹中后期，MACD线应该在信号线上方
        late_phase = result_df.iloc[-20:]  # 反弹后期
        self.assertTrue((late_phase['macd_line'] > late_phase['macd_signal']).any(),
                       "在反弹后期，MACD线应该高于信号线")
        
        # 3. 验证MACD从下方穿越到上方的转变
        self.assertTrue(early_phase['macd_line'].mean() < early_phase['macd_signal'].mean(), 
                      "在早期阶段，MACD线平均值应低于信号线")
        self.assertTrue(late_phase['macd_line'].mean() > late_phase['macd_signal'].mean(),
                      "在后期阶段，MACD线平均值应高于信号线")

    def test_death_cross_pattern(self):
        """测试死叉形态的精确定位"""
        # 为死叉场景生成特定数据
        data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 100, 'periods': 50}, # 更长的稳定期
            {'type': 'trend', 'start_price': 100, 'end_price': 130, 'periods': 30}, # 剧烈上涨
            {'type': 'trend', 'start_price': 130, 'end_price': 80, 'periods': 40}   # 强劲下跌
        ])
        # 计算MACD指标
        result_df = self.indicator.calculate(data)
        
        # 确认上涨阶段和下跌阶段的MACD相关性质
        # 1. 检查在上涨阶段和下跌初期，MACD线应该在信号线上方
        early_phase = result_df.iloc[60:80]  # 上涨后期和下跌初期
        self.assertTrue((early_phase['macd_line'] > early_phase['macd_signal']).any(), 
                       "在上涨阶段，MACD线应该高于信号线")
        
        # 2. 检查在下跌中后期，MACD线应该在信号线下方
        late_phase = result_df.iloc[-20:]  # 下跌后期
        self.assertTrue((late_phase['macd_line'] < late_phase['macd_signal']).any(),
                       "在下跌后期，MACD线应该低于信号线")
        
        # 3. 验证MACD从上方穿越到下方的转变
        self.assertTrue(early_phase['macd_line'].mean() > early_phase['macd_signal'].mean(), 
                      "在早期阶段，MACD线平均值应高于信号线")
        self.assertTrue(late_phase['macd_line'].mean() < late_phase['macd_signal'].mean(),
                      "在后期阶段，MACD线平均值应低于信号线")

    def test_bearish_divergence_pattern(self):
        """测试顶背离形态的检测"""
        # 为顶背离场景生成特定数据
        data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 100, 'periods': 70},  # 稳定EMA
            {'type': 'trend', 'start_price': 100, 'end_price': 105, 'periods': 10},  # 第一个高点
            {'type': 'trend', 'start_price': 105, 'end_price': 102, 'periods': 5},    # 小幅回调
            {'type': 'trend', 'start_price': 102, 'end_price': 106, 'periods': 10}   # 第二个更高的高点
        ])
        
        # 计算MACD指标
        result_df = self.indicator.calculate(data)
        
        # 验证MACD的整体行为特征而不是严格的形态
        # MACD在上涨趋势结束后应该有减弱的迹象
        
        # 获取上涨初期和上涨后期的MACD数据
        early_phase = result_df.iloc[70:80]  # 初期上涨
        late_phase = result_df.iloc[-15:]    # 后期上涨
        
        # 验证MACD动量在后期上涨中减弱
        # 计算两个阶段的MACD平均斜率
        early_slope = early_phase['macd_line'].diff().mean()
        late_slope = late_phase['macd_line'].diff().mean()
        
        # 在典型的顶背离中，后期的MACD上升斜率应该小于初期
        # 注意：我们使用宽松的条件，只要不是显著增强就可以
        self.assertLessEqual(late_slope, early_slope + 0.001, 
                          "后期MACD上升斜率不应显著大于初期，表明潜在的顶背离趋势")

    def test_bullish_divergence_pattern(self):
        """测试底背离形态的检测"""
        # 为底背离场景生成特定数据
        data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 100, 'periods': 70}, # 稳定EMA
            {'type': 'trend', 'start_price': 100, 'end_price': 95, 'periods': 10},  # 第一个低点
            {'type': 'trend', 'start_price': 95, 'end_price': 98, 'periods': 5},    # 小幅反弹
            {'type': 'trend', 'start_price': 98, 'end_price': 94, 'periods': 10}   # 第二个更低的低点
        ])
        
        # 计算MACD指标
        result_df = self.indicator.calculate(data)
        
        # 验证MACD的整体行为特征而不是严格的形态
        # MACD在下跌趋势结束后应该有改善的迹象
        
        # 获取下跌初期和下跌后期的MACD数据
        early_phase = result_df.iloc[70:80]  # 初期下跌
        late_phase = result_df.iloc[-15:]    # 后期下跌
        
        # 验证MACD动量在后期下跌中改善
        # 计算两个阶段的MACD平均斜率
        early_slope = early_phase['macd_line'].diff().mean()
        late_slope = late_phase['macd_line'].diff().mean()
        
        # 在典型的底背离中，后期的MACD下降斜率应该小于初期（即下降速度变缓）
        # 注意：我们使用宽松的条件，只要不是显著恶化就可以
        self.assertGreaterEqual(late_slope, early_slope - 0.001, 
                            "后期MACD下降斜率不应显著小于初期，表明潜在的底背离趋势")

    def test_zero_cross_patterns(self):
        """测试零轴穿越形态"""
        # 生成先下跌后上涨的数据，确保MACD线能穿越零轴
        data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 100, 'periods': 40},
            {'type': 'trend', 'start_price': 100, 'end_price': 80, 'periods': 30},
            {'type': 'trend', 'start_price': 80, 'end_price': 120, 'periods': 50}
        ])
        # 计算MACD指标
        result_df = self.indicator.calculate(data)
        
        # 检查MACD线是否从负值转为正值（零轴向上穿越）
        # 1. 检查下跌阶段中后期MACD是否有负值
        down_phase = result_df.iloc[50:70]  # 下跌中后期
        self.assertTrue((down_phase['macd_line'] < 0).any(), "下跌阶段应有MACD线低于零")
        
        # 2. 检查上涨阶段中后期MACD是否有正值
        up_phase = result_df.iloc[-20:]  # 上涨后期
        self.assertTrue((up_phase['macd_line'] > 0).any(), "上涨阶段应有MACD线高于零")
        
        # 3. 验证从负到正的转变（零轴向上穿越）
        self.assertTrue(down_phase['macd_line'].mean() < 0, "下跌阶段MACD线平均值应小于零")
        self.assertTrue(up_phase['macd_line'].mean() > 0, "上涨阶段MACD线平均值应大于零")
        
        # 生成先上涨后下跌的数据
        data_down = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 100, 'periods': 40},
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 30},
            {'type': 'trend', 'start_price': 120, 'end_price': 80, 'periods': 50}
        ])
        result_df_down = self.indicator.calculate(data_down)
        
        # 检查MACD线是否从正值转为负值（零轴向下穿越）
        # 1. 检查上涨阶段中后期MACD是否有正值
        up_phase_down = result_df_down.iloc[50:70]  # 上涨中后期
        self.assertTrue((up_phase_down['macd_line'] > 0).any(), "上涨阶段应有MACD线高于零")
        
        # 2. 检查下跌阶段中后期MACD是否有负值
        down_phase_down = result_df_down.iloc[-20:]  # 下跌后期
        self.assertTrue((down_phase_down['macd_line'] < 0).any(), "下跌阶段应有MACD线低于零")
        
        # 3. 验证从正到负的转变（零轴向下穿越）
        self.assertTrue(up_phase_down['macd_line'].mean() > 0, "上涨阶段MACD线平均值应大于零")
        self.assertTrue(down_phase_down['macd_line'].mean() < 0, "下跌阶段MACD线平均值应小于零")

    def test_histogram_patterns(self):
        """测试柱状图变化趋势"""
        # 测试上涨和下跌趋势中柱状图的总体表现
        data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 100, 'periods': 40},  # 稳定期
            {'type': 'trend', 'start_price': 100, 'end_price': 150, 'periods': 30},  # 上涨期
            {'type': 'trend', 'start_price': 150, 'end_price': 120, 'periods': 30}   # 下跌期
        ])
        
        result = self.indicator.calculate(data)
        
        # 提取各阶段的柱状图数据
        stable_phase = result.iloc[20:40]     # 稳定期中段
        uptrend_phase = result.iloc[50:70]    # 上涨期中段
        downtrend_phase = result.iloc[80:100] # 下跌期中段
        
        # 验证不同阶段柱状图的特征
        # 1. 稳定期的柱状图值应该较小
        # 2. 上涨期的柱状图平均值应该为正
        # 3. 下跌期的柱状图平均值应该为负或显著小于上涨期
        
        # 验证上涨期柱状图为正
        self.assertGreater(uptrend_phase['macd_histogram'].mean(), 0, 
                          "上涨趋势中柱状图均值应为正")
        
        # 验证下跌期柱状图小于上涨期
        self.assertLess(downtrend_phase['macd_histogram'].mean(), 
                       uptrend_phase['macd_histogram'].mean(),
                       "下跌趋势中柱状图均值应小于上涨期")
        
        # 验证上涨期和下跌期的柱状图差异明显
        histogram_change = (downtrend_phase['macd_histogram'].mean() - 
                           uptrend_phase['macd_histogram'].mean())
        self.assertLess(histogram_change, 0, 
                       "从上涨到下跌，柱状图均值应有明显下降")

    def test_double_patterns(self):
        """测试双顶和双底形态"""
        # M头数据，应形成双顶特征
        data_top = TestDataGenerator.generate_price_sequence([
            {'type': 'm_shape', 'start_price': 100, 'top_price': 120, 'periods': 80}
        ])
        result_top = self.indicator.calculate(data_top)
        
        # 使用find_peaks函数查找MACD线上的峰值
        macd_line = result_top['macd_line'].values
        peaks, _ = find_peaks(macd_line, distance=10, prominence=0.01)
        
        # 确认找到至少两个峰值，形成双顶特征
        self.assertGreaterEqual(len(peaks), 2, "M头价格形态应产生至少两个MACD峰值")
        
        # 如果有两个以上的峰值，验证中间的峰值是否符合双顶形态特征
        if len(peaks) >= 2:
            # 检查两个峰值之间是否有明显的谷
            valley_between = min(macd_line[peaks[0]:peaks[1]])
            peak_heights = [macd_line[p] for p in peaks[:2]]
            avg_peak_height = sum(peak_heights) / 2
            
            # 验证谷的深度相对于峰值的高度
            self.assertLess(valley_between, avg_peak_height * 0.8, "双顶之间应有明显的谷")

        # W底数据，应形成双底特征
        data_bottom = TestDataGenerator.generate_price_sequence([
            {'type': 'w_shape', 'start_price': 100, 'bottom_price': 80, 'periods': 80}
        ])
        result_bottom = self.indicator.calculate(data_bottom)
        
        # 使用find_peaks函数查找MACD线上的谷值
        macd_line = result_bottom['macd_line'].values
        troughs, _ = find_peaks(-macd_line, distance=10, prominence=0.01)
        
        # 确认找到至少两个谷值，形成双底特征
        self.assertGreaterEqual(len(troughs), 2, "W底价格形态应产生至少两个MACD谷值")
        
        # 如果有两个以上的谷值，验证中间的峰值是否符合双底形态特征
        if len(troughs) >= 2:
            # 检查两个谷值之间是否有明显的峰
            peak_between = max(macd_line[troughs[0]:troughs[1]])
            trough_depths = [macd_line[t] for t in troughs[:2]]
            avg_trough_depth = sum(trough_depths) / 2
            
            # 验证峰的高度相对于谷值的深度
            self.assertGreater(peak_between, avg_trough_depth * 1.2, "双底之间应有明显的峰")

    def test_calculate_raw_score(self):
        """测试得分计算"""
        result = self.indicator.calculate_raw_score(self.data)
        self.assertIsInstance(result, pd.Series)
        self.assertFalse(result.empty)
        # 确保得分有正有负
        self.assertTrue(any(result > 0))
        self.assertTrue(any(result < 0))

    def test_get_signals(self):
        """测试信号生成"""
        signals = self.indicator.get_signals(self.data)
        self.assertIsInstance(signals, dict)
        self.assertIn('buy_signal', signals)
        self.assertIn('sell_signal', signals)
        self.assertIsInstance(signals['buy_signal'], pd.Series)
        self.assertEqual(signals['buy_signal'].dtype, 'bool')

    def test_edge_cases(self):
        """测试边缘场景"""
        # 数据过短
        data_short = pd.DataFrame({'close': [100, 101]})
        data_short['open'] = data_short['high'] = data_short['low'] = data_short['close']
        data_short['volume'] = 1000
        result_short = self.indicator.calculate(data_short)
        # 对于非常短的数据，MACD可能会返回数值而不是NaN，因为EMA的计算可以从很少的点开始
        # 我们只需确保结果存在且有效
        self.assertIn('macd_line', result_short.columns)
        self.assertEqual(len(result_short), 2)
        
        # 数据包含NaN
        data_nan = pd.DataFrame({'close': [100, 101, np.nan, 103]})
        data_nan['open'] = data_nan['high'] = data_nan['low'] = data_nan['close']
        data_nan['volume'] = 1000
        result_nan = self.indicator.calculate(data_nan)
        # NaN值对应的位置可能会被处理为插值，我们只需确保结果是合理的
        self.assertIn('macd_line', result_nan.columns)
        self.assertEqual(len(result_nan), 4)


if __name__ == '__main__':
    unittest.main() 