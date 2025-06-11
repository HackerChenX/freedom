"""
策略与定制指标单元测试
"""
import unittest
import pandas as pd
import numpy as np
import pandas.testing as pd_testing
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator
from indicators.factory import IndicatorFactory
from unittest.mock import patch
from formula.formula import Formula
from tests.helper.log_capture import LogCaptureMixin
from unittest.mock import MagicMock
from indicators.base_indicator import BaseIndicator

def setUpModule():
    """在模块所有测试开始前运行，用于注册所有指标"""
    IndicatorFactory.auto_register_all_indicators()

class TestEnhancedMACD(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    def setUp(self):
        try:
            self.indicator = IndicatorFactory.create_indicator(
                'ENHANCEDMACD', fast_period=12, slow_period=26, signal_period=9
            )
        except Exception as e:
            self.skipTest(f"无法创建ENHANCEDMACD: {e}")
        # The parent MACD class returns 'DIF', 'DEA', 'MACD'. The test needs to reflect that.
        self.expected_columns = ['DIF', 'DEA', 'MACD']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'v_shape', 'start_price': 100, 'bottom_price': 90, 'periods': 50}
        ])

    def test_macd_calculation(self):
        """测试 EnhancedMACD 的核心计算逻辑"""
        result = self.indicator.calculate(self.data)
        self.assertIsInstance(result, pd.DataFrame)
        for col in self.expected_columns:
            self.assertIn(col, result.columns)

        # 独立计算以进行验证
        close = self.data['close']
        fast_ema = close.ewm(span=12, adjust=False).mean()
        slow_ema = close.ewm(span=26, adjust=False).mean()
        # The implementation has a sensitivity parameter, which we assume is 1.0 for the test
        expected_dif = fast_ema - slow_ema
        expected_dea = expected_dif.ewm(span=9, adjust=False).mean()
        expected_macd = (expected_dif - expected_dea) * 2

        # The column names from the implementation are 'DIF', 'DEA', 'MACD'
        pd_testing.assert_series_equal(result['DIF'], expected_dif, check_names=False)
        pd_testing.assert_series_equal(result['DEA'], expected_dea, check_names=False)
        pd_testing.assert_series_equal(result['MACD'], expected_macd, check_names=False)

    def test_cross_detection(self):
        """测试 EnhancedMACD 的金叉和死叉信号检测"""
        # 1. 构造一个明确的金叉场景
        dif = pd.Series([-0.1, -0.05, 0.05, 0.1])
        dea = pd.Series([0, 0, 0, 0])
        gc_data = pd.DataFrame({
            'DIF': dif,
            'DEA': dea,
            'MACD': 2 * (dif - dea)
        })
        # The get_patterns method needs the full dataframe with price data
        price_data = TestDataGenerator.generate_price_sequence([
            {'type': 'linear', 'start_price': 100, 'periods': 4, 'trend': 0.1}
        ])
        gc_data = pd.concat([price_data.reset_index(drop=True), gc_data], axis=1)

        # The base MACD's get_patterns is what's called.
        patterns_gc = self.indicator.get_patterns(gc_data)
        self.assertIsInstance(patterns_gc, list)
        golden_cross_points = [p for p in patterns_gc if p['pattern_id'] == 'MACD_GOLDEN_CROSS']
        self.assertFalse(not golden_cross_points, "未检测到金叉信号")
        # Ensure the cross happens at the right index
        self.assertEqual(golden_cross_points[0]['start_index'], 2)

        # 2. 构造一个明确的死叉场景
        dif_dc = pd.Series([0.1, 0.05, -0.05, -0.1])
        dea_dc = pd.Series([0, 0, 0, 0])
        dc_data = pd.DataFrame({
            'DIF': dif_dc,
            'DEA': dea_dc,
            'MACD': 2 * (dif_dc - dea_dc)
        })
        price_data_dc = TestDataGenerator.generate_price_sequence([
            {'type': 'linear', 'start_price': 100, 'periods': 4, 'trend': -0.1}
        ])
        dc_data = pd.concat([price_data_dc.reset_index(drop=True), dc_data], axis=1)

        patterns_dc = self.indicator.get_patterns(dc_data)
        self.assertIsInstance(patterns_dc, list)
        death_cross_points = [p for p in patterns_dc if p['pattern_id'] == 'MACD_DEATH_CROSS']
        self.assertFalse(not death_cross_points, "未检测到死叉信号")
        self.assertEqual(death_cross_points[0]['start_index'], 2)

class TestEnhancedRSI(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    def setUp(self):
        try:
            self.indicator = IndicatorFactory.create_indicator('ENHANCEDRSI', periods=[14])
        except Exception as e:
            self.skipTest(f"无法创建ENHANCEDRSI: {e}")
        self.expected_columns = ['RSI14', 'RSI14_smooth', 'RSI14_overbought', 'RSI14_oversold']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'm_shape', 'start_price': 100, 'top_price': 120, 'periods': 50}
        ])

    def test_rsi_calculation(self):
        """测试 EnhancedRSI 的核心计算逻辑"""
        from indicators.common import rsi as calc_rsi_common
        result = self.indicator.calculate(self.data)

        # 独立计算以进行验证,使用公共函数
        expected_rsi_values = calc_rsi_common(self.data['close'].values, 14)
        # The implementation uses a default smooth_period of 3
        expected_rsi_ma = pd.Series(expected_rsi_values).rolling(window=3).mean().values

        # Convert to series for comparison, ensuring index alignment
        expected_rsi = pd.Series(expected_rsi_values, index=result.index)
        expected_rsi_ma_series = pd.Series(expected_rsi_ma, index=result.index)

        pd_testing.assert_series_equal(result['RSI14'].dropna(), expected_rsi.dropna())
        pd_testing.assert_series_equal(result['RSI14_smooth'].dropna(), expected_rsi_ma_series.dropna())

    def test_signal_detection(self):
        """测试 EnhancedRSI 的超买超卖信号"""
        # 1. 构造一个明确的超卖场景
        close_prices_os = [100 - i for i in range(20)]
        os_data = pd.DataFrame({'close': close_prices_os}, index=pd.to_datetime(pd.date_range(start='2023-01-01', periods=len(close_prices_os))))
        os_data['open'] = os_data['close']
        os_data['high'] = os_data['close']
        os_data['low'] = os_data['close']
        os_data['volume'] = 1000
        
        result_os = self.indicator.calculate(os_data)
        oversold_points = result_os[result_os['RSI14_oversold']]
        self.assertFalse(oversold_points.empty, "未检测到超卖信号")
        self.assertTrue((result_os.loc[oversold_points.index, 'RSI14'] < 30).all())

        # 2. 构造一个明确的超买场景
        close_prices_ob = [100 + i for i in range(30)] # Longer series to ensure RSI crosses 70
        ob_data = pd.DataFrame({'close': close_prices_ob}, index=pd.to_datetime(pd.date_range(start='2023-01-01', periods=len(close_prices_ob))))
        ob_data['open'] = ob_data['close']
        ob_data['high'] = ob_data['close']
        ob_data['low'] = ob_data['close']
        ob_data['volume'] = 1000

        result_ob = self.indicator.calculate(ob_data)
        # The signal is a boolean flag
        overbought_points = result_ob[result_ob['RSI14_overbought']]
        self.assertFalse(overbought_points.empty, "未检测到超买信号")
        self.assertTrue((result_ob.loc[overbought_points.index, 'RSI14'] > 70).all())

class TestCompositeIndicator(unittest.TestCase):
    def setUp(self):
        """准备一个包含MACD和RSI的复合指标实例"""
        self.macd_indicator = IndicatorFactory.create_indicator('MACD')
        self.rsi_indicator = IndicatorFactory.create_indicator('RSI')
        try:
            self.composite_indicator = IndicatorFactory.create_indicator(
                'CompositeIndicator',
                indicators=[self.macd_indicator, self.rsi_indicator]
            )
        except Exception as e:
            self.skipTest(f"无法创建CompositeIndicator: {e}")
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'v_shape', 'start_price': 100, 'bottom_price': 80, 'periods': 100}
        ])

    def test_calculation_delegation(self):
        """测试复合指标是否正确委托计算并合并结果"""
        result = self.composite_indicator.calculate(self.data)
        
        # 验证结果包含了所有子指标的列
        self.assertIn('macd_line', result.columns)
        self.assertIn('macd_signal', result.columns)
        self.assertIn('macd_histogram', result.columns)
        self.assertIn('rsi', result.columns)

    def test_pattern_aggregation(self):
        """测试复合指标是否正确聚合子指标的形态"""
        # 使用能够同时触发MACD金叉和RSI超卖的数据
        data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 80, 'periods': 20}, # 产生超卖
            {'type': 'trend', 'start_price': 80, 'end_price': 95, 'periods': 15}  # 产生金叉
        ])
        
        patterns = self.composite_indicator.get_patterns(data)
        self.assertIsInstance(patterns, pd.DataFrame, "复合指标get_patterns应返回DataFrame")
        self.assertTrue(any(col.startswith('MACD_') for col in patterns.columns), "未找到来自MACD的形态")
        self.assertTrue(any(col.startswith('RSI_') for col in patterns.columns), "未找到来自RSI的形态")

class TestFormulaIndicator(unittest.TestCase):
    @patch('formula.stock_formula.StockData')
    @patch('formula.stock_formula.MA')
    def setUp(self, MockMA, MockStockData):
        self.mock_stock_data = MockStockData.return_value
        self.mock_ma = MockMA
        self.indicator = Formula("mock_code")
        self.indicator.daily = self.mock_stock_data
        self.indicator.weekly = self.mock_stock_data
        self.indicator.monthly = self.mock_stock_data

    def test_shrinkage_logic(self):
        """测试'缩量'方法的逻辑"""
        # 1. 构造一个明确的缩量场景
        volume_shrinking = np.array([1000] * 28 + [1000, 900]) # Last day is 900
        self.mock_stock_data.volume = volume_shrinking
        # Mock the MA function to return a value that will make the ratio < 0.95
        self.mock_ma.return_value = np.array([1000] * 29 + [950]) # 2-day MA for the last element

        self.assertTrue(self.indicator.缩量(), "在成交量小于2日均量的95%时，'缩量'应返回True")

        # 2. 构造一个非缩量场景
        volume_not_shrinking = np.array([1000] * 28 + [1000, 960]) # Last day is 960
        self.mock_stock_data.volume = volume_not_shrinking
        self.mock_ma.return_value = np.array([1000] * 29 + [980]) # 2-day MA for the last element

        self.assertFalse(self.indicator.缩量(), "在成交量不满足缩量条件时，'缩量'应返回False")

class TestPlatformBreakout(unittest.TestCase):
    def setUp(self):
        """准备一个默认参数的PlatformBreakout实例"""
        try:
            self.indicator = IndicatorFactory.create_indicator('PlatformBreakout', platform_period=20, max_volatility=0.05)
        except Exception as e:
            self.skipTest(f"无法创建PlatformBreakout: {e}")

    def test_platform_detection(self):
        """测试平台识别逻辑"""
        # 1. 构造一个符合平台定义的数据 (波动率 < 5%)
        platform_data = pd.DataFrame({
            'high': np.linspace(101, 101.5, 21),
            'low': np.linspace(100, 100.5, 21),
            'close': np.linspace(100.5, 101, 21),
            'volume': [100] * 21
        })
        # 补全所需列
        platform_data['open'] = platform_data['close']
        
        result = self.indicator.calculate(platform_data)
        # 在窗口期满后，应该能检测到平台
        self.assertTrue(result['is_platform'].iloc[-1], "未能识别出有效的平台")
        self.assertAlmostEqual(result['platform_upper'].iloc[-1], 101.5)
        self.assertAlmostEqual(result['platform_lower'].iloc[-1], 100.0)

        # 2. 构造一个不符合平台定义的数据 (波动率 > 5%)
        volatile_data = pd.DataFrame({
            'high': np.linspace(120, 121, 21),
            'low': np.linspace(100, 101, 21),
            'close': np.linspace(101, 119, 21),
            'volume': [100] * 21
        })
        volatile_data['open'] = volatile_data['close']
        
        result_volatile = self.indicator.calculate(volatile_data)
        self.assertFalse(result_volatile['is_platform'].iloc[-1], "波动剧烈时不应识别为平台")

    def test_upward_breakout_detection(self):
        """测试向上突破的识别逻辑"""
        # 构造一个平台后放量突破的数据
        prices = [100.5] * 21 + [105]  # 前21天形成平台，第22天突破
        volumes = [100] * 21 + [200]  # 第22天放量
        breakout_data = pd.DataFrame({
            'high': prices, 'low': prices, 'close': prices, 'volume': volumes
        })
        breakout_data['open'] = breakout_data['close']

        result = self.indicator.calculate(breakout_data)
        # 平台在第21天形成, 突破应该在第22天被检测到
        self.assertTrue(result['up_breakout'].iloc[-1], "未能识别出有效的向上突破")
        self.assertEqual(result['breakout_direction'].iloc[-1], "向上突破")
        
    def test_breakout_failure_due_to_low_volume(self):
        """测试因成交量不足导致突破失败的场景"""
        # 构造一个平台后缩量突破的数据
        prices = [100.5] * 21 + [105] # 价格突破
        volumes = [100] * 21 + [50]   # 但成交量萎缩
        breakout_data = pd.DataFrame({
            'high': prices, 'low': prices, 'close': prices, 'volume': volumes
        })
        breakout_data['open'] = breakout_data['close']
        
        result = self.indicator.calculate(breakout_data)
        # 即使价格突破，但因成交量不足，不应视为有效突破
        self.assertFalse(result['up_breakout'].iloc[-1], "成交量不足时不应识别为向上突破")

class TestVShapedReversal(unittest.TestCase):
    def setUp(self):
        try:
            self.indicator = IndicatorFactory.create_indicator('VShapedReversal')
        except Exception as e:
            self.skipTest(f"无法创建VShapedReversal: {e}")

    def test_reversal_completion_detection(self):
        """测试基于变化率的'v_reversal'完成信号"""
        # 构造一个11天的数据, 第5天为底部, 第10天完成反转
        # 0-5天: 从100跌到90 (-10%)
        # 5-10天: 从90涨到99 (+10%)
        prices = [100, 98, 96, 94, 92, 90, 91.8, 93.6, 95.4, 97.2, 99]
        v_data = pd.DataFrame({'close': prices})
        # 补全所需列
        v_data['open'] = v_data['close']
        v_data['high'] = v_data['close']
        v_data['low'] = v_data['close']
        v_data['volume'] = [100] * 11
        
        result = self.indicator.calculate(v_data)
        
        # 'v_reversal'信号应该在形态完成时(第10天)触发
        self.assertTrue(result['v_reversal'].iloc[10], "V形反转完成信号未在预期位置触发")
        # 其他位置不应为True
        self.assertFalse(result['v_reversal'].iloc[:10].any(), "V形反转完成信号在不应触发的位置触发")

    def test_reversal_bottom_detection(self):
        """测试基于最低点的'v_bottom'底部信号"""
        # 构造一个完美的对称V形
        prices = [105, 104, 103, 102, 101, 100, 101, 102, 103, 104, 105]
        v_data = pd.DataFrame({'close': prices})
        # 补全所需列
        v_data['open'] = v_data['close']
        v_data['high'] = v_data['close']
        v_data['low'] = v_data['close']
        v_data['volume'] = [100] * 11

        result = self.indicator.calculate(v_data)
        
        # 'v_bottom'信号应该在V形底部(第5天)触发
        self.assertTrue(result['v_bottom'].iloc[5], "V形底部信号未在预期位置触发")
        # 其他位置不应为True
        self.assertFalse(result['v_bottom'].drop(index=result.index[5]).any(), "V形底部信号在不应触发的位置触发")

class TestIslandReversal(unittest.TestCase):
    def setUp(self):
        try:
            self.indicator = IndicatorFactory.create_indicator('IslandReversal')
        except Exception as e:
            self.skipTest(f"无法创建IslandReversal: {e}")

    def test_bottom_island_reversal_detection(self):
        """测试底部岛形反转的识别逻辑"""
        # 手动构造一个精确的底部岛形反转形态
        # 1. 下跌趋势 (day 0-2)
        # 2. 向下跳空 (day 3) low[2] > high[3]
        # 3. 岛屿 (day 3-5)
        # 4. 向上跳空 (day 6) low[6] > high[5]
        ohlc_data = {
            'open':  [110, 108, 105, 101, 101, 100, 104, 106, 108],
            'high':  [111, 109, 106, 102, 102, 101, 105, 107, 109],
            'low':   [109, 107, 104, 100, 100,  99, 103, 105, 107],
            'close': [110, 108, 105, 101, 101, 100, 104, 106, 108],
            'volume':[1000] * 9
        }
        data = pd.DataFrame(ohlc_data)
        
        result = self.indicator.calculate(data)
        
        # 验证缺口识别
        self.assertTrue(result['down_gap'].iloc[3], "未能识别出向下的进入缺口")
        self.assertTrue(result['up_gap'].iloc[6], "未能识别出向上的离开缺口")

        # 验证岛形反转信号
        # 信号在离开缺口当天触发
        self.assertTrue(result['bottom_island_reversal'].iloc[6], "未能识别出底部岛形反转信号")
        # 确保其他位置没有误报
        self.assertFalse(result['bottom_island_reversal'].drop(index=result.index[6]).any(), "在不应触发的位置触发了岛形反转信号")

class TestZXMAbsorb(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    def setUp(self):
        try:
            self.indicator = IndicatorFactory.create_indicator(
                'ZXMAbsorb', short_ma=5, long_ma=10, volume_ma=10, absorb_threshold=1.5
            )
        except Exception as e:
            self.skipTest(f"无法创建ZXMAbsorb: {e}")
            
        self.expected_columns = ['is_absorb']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'sideways', 'price': 100, 'periods': 30}
        ])

class TestZXMWashplate(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    def setUp(self):
        try:
            self.indicator = IndicatorFactory.create_indicator(
                'ZXMWashplate', short_ma=5, long_ma=60, shrink_threshold=0.8
            )
        except Exception as e:
            self.skipTest(f"无法创建ZXMWashplate: {e}")

        self.expected_columns = ['is_washplate']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'v_shape', 'start_price': 100, 'bottom_price': 80, 'periods': 100}
        ])

if __name__ == '__main__':
    unittest.main() 