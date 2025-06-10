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

class TestEnhancedMACD(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        self.indicator = IndicatorFactory.create_indicator('ENHANCEDMACD', short_period=12, long_period=26, signal_period=9)
        self.expected_columns = ['macd', 'signal', 'hist', 'golden_cross', 'death_cross']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'v_shape', 'start_price': 100, 'bottom_price': 90, 'periods': 50}
        ])

    def test_macd_calculation(self):
        """测试 EnhancedMACD 的核心计算逻辑"""
        result = self.indicator.calculate(self.data)
        
        # 独立计算以进行验证
        ema_short = self.data['close'].ewm(span=12, adjust=False).mean()
        ema_long = self.data['close'].ewm(span=26, adjust=False).mean()
        expected_macd = ema_short - ema_long
        expected_signal = expected_macd.ewm(span=9, adjust=False).mean()
        expected_hist = expected_macd - expected_signal
        
        pd_testing.assert_series_equal(result['macd'], expected_macd, name='macd')
        pd_testing.assert_series_equal(result['signal'], expected_signal, name='signal')
        pd_testing.assert_series_equal(result['hist'], expected_hist, name='hist')

    def test_cross_detection(self):
        """测试 EnhancedMACD 的金叉和死叉信号检测"""
        # 1. 构造一个明确的金叉场景
        # 确保数据足够长以计算MACD
        close_prices_gc = [100 + i for i in range(10)] + [110 - i for i in range(5)] + [105 + i for i in range(15)]
        gc_data = pd.DataFrame({'close': close_prices_gc}, index=pd.to_datetime(pd.date_range(start='2023-01-01', periods=len(close_prices_gc))))
        # 为TestDataGenerator添加所有必要的列
        gc_data['open'] = gc_data['close']
        gc_data['high'] = gc_data['close']
        gc_data['low'] = gc_data['close']
        gc_data['volume'] = 1000

        result_gc = self.indicator.calculate(gc_data)
        # 金叉通常发生在hist由负变正时
        golden_cross_points = result_gc[result_gc['golden_cross']]
        self.assertFalse(golden_cross_points.empty, "未检测到金叉信号")
        # 我们可以检查金叉点前后的hist值
        gc_index = golden_cross_points.index[0]
        gc_loc = result_gc.index.get_loc(gc_index)
        self.assertTrue(result_gc['hist'].iloc[gc_loc] > 0)
        self.assertTrue(result_gc['hist'].iloc[gc_loc - 1] < 0)

        # 2. 构造一个明确的死叉场景
        close_prices_dc = [100 - i for i in range(10)] + [90 + i for i in range(5)] + [95 - i for i in range(15)]
        dc_data = pd.DataFrame({'close': close_prices_dc}, index=pd.to_datetime(pd.date_range(start='2023-01-01', periods=len(close_prices_dc))))
        dc_data['open'] = dc_data['close']
        dc_data['high'] = dc_data['close']
        dc_data['low'] = dc_data['close']
        dc_data['volume'] = 1000
        
        result_dc = self.indicator.calculate(dc_data)
        death_cross_points = result_dc[result_dc['death_cross']]
        self.assertFalse(death_cross_points.empty, "未检测到死叉信号")
        # 死叉通常发生在hist由正变负时
        dc_index = death_cross_points.index[0]
        dc_loc = result_dc.index.get_loc(dc_index)
        self.assertTrue(result_dc['hist'].iloc[dc_loc] < 0)
        self.assertTrue(result_dc['hist'].iloc[dc_loc - 1] > 0)

class TestEnhancedRSI(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        self.indicator = IndicatorFactory.create_indicator('ENHANCEDRSI', period=14, ma_period=6, oversold_threshold=30, overbought_threshold=70)
        self.expected_columns = ['rsi', 'rsi_ma', 'oversold_signal', 'overbought_signal']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'm_shape', 'start_price': 100, 'top_price': 120, 'periods': 50}
        ])

    def test_rsi_calculation(self):
        """测试 EnhancedRSI 的核心计算逻辑"""
        result = self.indicator.calculate(self.data)

        # 独立计算以进行验证
        delta = self.data['close'].diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
        loss = -delta.where(delta < 0, 0).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / loss
        expected_rsi = 100 - (100 / (1 + rs))
        expected_rsi_ma = expected_rsi.rolling(window=6).mean()

        pd_testing.assert_series_equal(result['rsi'], expected_rsi, name='rsi')
        pd_testing.assert_series_equal(result['rsi_ma'].dropna(), expected_rsi_ma.dropna(), name='rsi_ma')

    def test_signal_detection(self):
        """测试 EnhancedRSI 的超买超卖信号"""
        # 1. 构造一个明确的超卖场景
        # 连续下跌以拉低RSI
        close_prices_os = [100 - i for i in range(20)]
        os_data = pd.DataFrame({'close': close_prices_os}, index=pd.to_datetime(pd.date_range(start='2023-01-01', periods=len(close_prices_os))))
        os_data['open'] = os_data['close']
        os_data['high'] = os_data['close']
        os_data['low'] = os_data['close']
        os_data['volume'] = 1000
        
        result_os = self.indicator.calculate(os_data)
        oversold_points = result_os[result_os['oversold_signal']]
        self.assertFalse(oversold_points.empty, "未检测到超卖信号")
        # 验证信号点的RSI值确实低于阈值
        self.assertTrue((result_os.loc[oversold_points.index, 'rsi'] < 30).all())

        # 2. 构造一个明确的超买场景
        # 连续上涨以拉高RSI
        close_prices_ob = [100 + i for i in range(20)]
        ob_data = pd.DataFrame({'close': close_prices_ob}, index=pd.to_datetime(pd.date_range(start='2023-01-01', periods=len(close_prices_ob))))
        ob_data['open'] = ob_data['close']
        ob_data['high'] = ob_data['close']
        ob_data['low'] = ob_data['close']
        ob_data['volume'] = 1000

        result_ob = self.indicator.calculate(ob_data)
        overbought_points = result_ob[result_ob['overbought_signal']]
        self.assertFalse(overbought_points.empty, "未检测到超买信号")
        # 验证信号点的RSI值确实高于阈值
        self.assertTrue((result_ob.loc[overbought_points.index, 'rsi'] > 70).all())

class TestCompositeIndicator(unittest.TestCase):
    def setUp(self):
        """准备一个包含MACD和RSI的复合指标实例"""
        self.macd_indicator = IndicatorFactory.create_indicator('MACD')
        self.rsi_indicator = IndicatorFactory.create_indicator('RSI')
        self.composite_indicator = IndicatorFactory.create_indicator(
            'COMPOSITEINDICATOR',
            indicators=[self.macd_indicator, self.rsi_indicator]
        )
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'v_shape', 'start_price': 100, 'bottom_price': 80, 'periods': 100}
        ])

    def test_calculation_delegation(self):
        """测试复合指标是否正确委托计算并合并结果"""
        result = self.composite_indicator.calculate(self.data)
        
        # 验证结果包含了所有子指标的列
        self.assertIn('macd', result.columns)
        self.assertIn('signal', result.columns)
        self.assertIn('hist', result.columns)
        self.assertIn('rsi', result.columns)

    def test_pattern_aggregation(self):
        """测试复合指标是否正确聚合子指标的形态"""
        # 使用能够同时触发MACD金叉和RSI超卖的数据
        data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 80, 'periods': 20}, # 产生超卖
            {'type': 'trend', 'start_price': 80, 'end_price': 95, 'periods': 15}  # 产生金叉
        ])
        
        patterns = self.composite_indicator.get_patterns(data)
        self.assertTrue(len(patterns) > 0, "复合指标未能识别出任何形态")
        
        # 验证是否同时包含来自两个指标的形态
        pattern_names = [p.get('pattern_id', '') for p in patterns]
        has_macd_pattern = any('macd' in name for name in pattern_names)
        has_rsi_pattern = any('rsi' in name for name in pattern_names)
        
        self.assertTrue(has_macd_pattern, "未找到来自MACD的形态")
        self.assertTrue(has_rsi_pattern, "未找到来自RSI的形态")

class TestFormulaIndicator(unittest.TestCase):
    @patch('formula.stock_formula.StockData')
    def setUp(self, MockStockData):
        """通过Mocking StockData来准备一个隔离的Formula实例"""
        # 创建一个mock实例来代表StockData的实例
        self.mock_stock_data_instance = MockStockData.return_value
        
        # 实例化被测对象，此时其内部对StockData()的调用已被mock
        self.indicator = Formula(code="mock_code")
        
        # 将实例中的数据获取对象替换为我们的mock实例
        # 这一步确保我们在测试方法中可以控制返回的数据
        self.indicator.daily = self.mock_stock_data_instance

    def test_shrinkage_logic(self):
        """测试'缩量'方法的逻辑"""
        # 1. 构造缩量场景 (成交量连续下降)
        shrinking_volume = pd.DataFrame({
            'close': [10] * 5,
            'volume': [100, 90, 80, 70, 60]
        })
        # 配置mock实例返回此数据
        self.mock_stock_data_instance.close = shrinking_volume['close'].values
        self.mock_stock_data_instance.volume = shrinking_volume['volume'].values
        
        # 断言'缩量'方法返回True
        self.assertTrue(self.indicator.缩量(), "在成交量递减时，'缩量'应返回True")

        # 2. 构造不缩量场景 (成交量上升)
        increasing_volume = pd.DataFrame({
            'close': [10] * 5,
            'volume': [60, 70, 80, 90, 100]
        })
        self.mock_stock_data_instance.close = increasing_volume['close'].values
        self.mock_stock_data_instance.volume = increasing_volume['volume'].values

        # 断言'缩量'方法返回False
        self.assertFalse(self.indicator.缩量(), "在成交量递增时，'缩量'应返回False")

class TestPlatformBreakout(unittest.TestCase):
    def setUp(self):
        """准备一个默认参数的PlatformBreakout实例"""
        self.indicator = IndicatorFactory.create_indicator('PLATFORMBREAKOUT', platform_period=20, max_volatility=0.05)

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
        """准备一个默认参数的VShapedReversal实例"""
        self.indicator = IndicatorFactory.create_indicator(
            'VSHAPEDREVERSAL', 
            decline_period=5, 
            rebound_period=5,
            decline_threshold=0.05,
            rebound_threshold=0.05
        )

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
        """准备一个默认参数的IslandReversal实例"""
        self.indicator = IndicatorFactory.create_indicator(
            'ISLANDREVERSAL',
            gap_threshold=0.01,
            island_max_days=5
        )

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

class TestZXMAbsorb(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        self.indicator = IndicatorFactory.create_indicator('ZXMABSORB')
        self.expected_columns = ['absorb_signal']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'sideways', 'price': 100, 'periods': 100}
        ])

class TestZXMWashplate(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        self.indicator = IndicatorFactory.create_indicator('ZXMWASHPLATE')
        self.expected_columns = ['washplate_signal']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'm_shape', 'start_price': 100, 'top_price': 105, 'periods': 100}
        ])

if __name__ == '__main__':
    unittest.main() 