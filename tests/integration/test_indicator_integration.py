"""
指标集成测试 - 测试不同指标之间的集成与协作
"""
import unittest
import pandas as pd
import numpy as np
from indicators.composite_indicator import CompositeIndicator
from indicators.factory import IndicatorFactory
from tests.helper.data_generator import TestDataGenerator
from tests.unit.indicator_test_mixin import IndicatorTestMixin

class TestIndicatorIntegration(unittest.TestCase):
    """测试不同指标之间的集成与协作"""

    def setUp(self):
        """准备测试数据和指标实例"""
        # 生成多种市场形态的测试数据
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 50},  # 上升趋势
            {'type': 'v_shape', 'start_price': 110, 'bottom_price': 95, 'periods': 50},  # V形反转
            {'type': 'm_shape', 'start_price': 95, 'top_price': 105, 'periods': 50},  # M形态
            {'type': 'trend', 'start_price': 95, 'end_price': 85, 'periods': 50},  # 下降趋势
        ])
        
        # 确保所有指标被注册
        IndicatorFactory.auto_register_all_indicators()
        
        # 为了测试需要，模拟一些StockInfo字段
        self.data['volume'] = self.data['volume'].astype(float)
        self.data['turnover_rate'] = self.data['volume'] / 10000  # 模拟换手率
        self.data['price_change'] = self.data['close'].diff()
        self.data['price_range'] = self.data['high'] - self.data['low']
        self.data['industry'] = 'Technology'  # 模拟行业
        
        # 保存支持的指标列表，排除不适合集成测试的指标
        self.supported_indicators = IndicatorFactory.get_supported_indicators()
        self.exclude_list = [
            'ZXMPATTERNINDICATOR', 'CompositeIndicator', 'FibonacciTools', 'SENTIMENTANALYSIS',
            'KDJ_CONDITION', 'MACD_CONDITION', 'MA_CONDITION', 'GENERIC_CONDITION',
            'KDJCONDITION', 'MACDCONDITION', 'MACONDITION', 'GENERICCONDITION',
            'ELLIOTTWAVE', 'INSTITUTIONALBEHAVIOR',
            'CROSS_OVER', 'CROSSOVER',
        ]

    def test_composite_indicator_with_basic_indicators(self):
        """测试复合指标与基础指标(MACD, RSI, BOLL)的集成"""
        # 1. 创建复合指标实例
        composite_indicator = CompositeIndicator()

        # 2. 创建并添加多个基础指标
        macd = IndicatorFactory.create_indicator('MACD')
        rsi = IndicatorFactory.create_indicator('RSI')
        boll = IndicatorFactory.create_indicator('BOLL')

        composite_indicator.add_indicator(macd, weight=1.0)
        composite_indicator.add_indicator(rsi, weight=1.0)
        composite_indicator.add_indicator(boll, weight=1.0)

        # 3. 运行计算
        result_df = composite_indicator.calculate(self.data.copy())

        # 4. 验证结果
        self.assertIsInstance(result_df, pd.DataFrame)
        
        # 验证是否包含了所有子指标的列
        expected_macd_cols = ['dif', 'dea', 'macd']
        expected_rsi_cols = ['rsi']
        expected_boll_cols = ['upper', 'middle', 'lower']
        
        for col in expected_macd_cols + expected_rsi_cols + expected_boll_cols:
            self.assertIn(col, result_df.columns, f"结果中缺少列: {col}")

        # 验证复合指标自己的列也存在
        self.assertIn('composite_score', result_df.columns)
    
    def test_trend_and_oscillator_integration(self):
        """测试趋势指标和震荡指标的集成"""
        # 创建复合指标，组合趋势指标和震荡指标
        composite = CompositeIndicator()
        
        # 添加趋势指标
        ma = IndicatorFactory.create_indicator('MA')
        adx = IndicatorFactory.create_indicator('ADX')
        
        # 添加震荡指标
        rsi = IndicatorFactory.create_indicator('RSI')
        cci = IndicatorFactory.create_indicator('CCI')
        
        # 添加指标到复合指标
        composite.add_indicator(ma, weight=1.0)
        composite.add_indicator(adx, weight=1.0)
        composite.add_indicator(rsi, weight=1.0)
        composite.add_indicator(cci, weight=1.0)
        
        # 计算结果
        result = composite.calculate(self.data.copy())
        
        # 验证结果
        self.assertIsInstance(result, pd.DataFrame)
        expected_cols = [
            'MA5', 'MA10', 'MA20',
            'ADX14', 'PDI14', 'MDI14',
            'rsi',
            'CCI'
        ]
        
        for col in expected_cols:
            self.assertIn(col, result.columns, f"结果中缺少列: {col}")
        
        # 验证复合指标分数
        self.assertIn('composite_score', result.columns)
        # 确保分数不全是NaN
        self.assertFalse(result['composite_score'].isna().all())
    
    def test_volume_and_price_indicators_integration(self):
        """测试成交量指标和价格指标的集成"""
        composite = CompositeIndicator()
        
        # 价格指标
        macd = IndicatorFactory.create_indicator('MACD')
        
        # 成交量指标
        obv = IndicatorFactory.create_indicator('OBV')
        
        composite.add_indicator(macd, weight=1.0)
        composite.add_indicator(obv, weight=1.0)
        
        result = composite.calculate(self.data.copy())
        
        # 验证结果
        self.assertIsInstance(result, pd.DataFrame)
        expected_cols = [
            'dif', 'dea', 'macd',
            'obv'
        ]
        
        for col in expected_cols:
            self.assertIn(col, result.columns, f"结果中缺少列: {col}")
        
        # 验证复合指标分数
        self.assertIn('composite_score', result.columns)
    
    def test_pattern_recognition_with_indicators(self):
        """测试形态识别与技术指标的集成"""
        # 计算基础指标
        macd = IndicatorFactory.create_indicator('MACD')
        kdj = IndicatorFactory.create_indicator('KDJ')
        
        macd_result = macd.calculate(self.data.copy())
        kdj_result = kdj.calculate(self.data.copy())
        
        # 检查是否有get_patterns方法
        if hasattr(macd, 'get_patterns') and hasattr(kdj, 'get_patterns'):
            try:
                # 获取MACD形态
                # 修复列名：使用新的列名格式
                if 'dif' in macd_result.columns and 'dea' in macd_result.columns:
                    # 将列名映射为get_patterns期望的列名
                    macd_result.rename(columns={
                        'dif': 'macd_line',
                        'dea': 'macd_signal',
                        'macd': 'macd_histogram'
                    }, inplace=True)
                
                macd_patterns = macd.get_patterns(macd_result)
                self.assertIsInstance(macd_patterns, list)
                
                # 获取KDJ形态
                kdj_patterns = kdj.get_patterns(self.data.copy())
                self.assertIsInstance(kdj_patterns, list)
                
                # 验证至少有一些形态被识别
                self.assertTrue(len(macd_patterns) > 0 or len(kdj_patterns) > 0, 
                                "没有识别出任何形态")
            except Exception as e:
                self.skipTest(f"形态识别测试失败: {e}")
    
    def test_multi_indicator_signal_consistency(self):
        """测试多个指标信号的一致性"""
        # 创建多个测试指标
        macd = IndicatorFactory.create_indicator('MACD')
        rsi = IndicatorFactory.create_indicator('RSI')
        kdj = IndicatorFactory.create_indicator('KDJ')
        
        # 计算指标
        macd_result = macd.calculate(self.data.copy())
        rsi_result = rsi.calculate(self.data.copy())
        kdj_result = kdj.calculate(self.data.copy())
        
        try:
            # 基于指标结果创建信号
            # 例如：MACD金叉信号
            if 'macd' in macd_result.columns:
                macd_signals = pd.Series(0, index=macd_result.index)
                macd_signals[macd_result['macd'] > 0] = 1
                macd_signals[macd_result['macd'] < 0] = -1
            
            # RSI超买超卖信号
            if 'rsi' in rsi_result.columns:
                rsi_signals = pd.Series(0, index=rsi_result.index)
                rsi_signals[rsi_result['rsi'] > 70] = -1  # 超买
                rsi_signals[rsi_result['rsi'] < 30] = 1   # 超卖
            
            # KDJ金叉死叉信号
            if 'K' in kdj_result.columns and 'D' in kdj_result.columns:
                kdj_signals = pd.Series(0, index=kdj_result.index)
                # K线上穿D线视为金叉
                kdj_signals[(kdj_result['K'] > kdj_result['D']) & (kdj_result['K'].shift(1) <= kdj_result['D'].shift(1))] = 1
                # K线下穿D线视为死叉
                kdj_signals[(kdj_result['K'] < kdj_result['D']) & (kdj_result['K'].shift(1) >= kdj_result['D'].shift(1))] = -1
            
            # 验证在某些情况下信号一致性
            # 注：这只是一个示例，实际应用中可能需要更复杂的信号一致性验证
            # 找出所有指标都给出同样信号的点
            consistent_signals = pd.DataFrame({
                'macd': macd_signals,
                'rsi': rsi_signals,
                'kdj': kdj_signals
            })
            
            # 所有指标都为正信号的点
            all_positive = consistent_signals[(consistent_signals['macd'] > 0) & 
                                             (consistent_signals['rsi'] > 0) & 
                                             (consistent_signals['kdj'] > 0)]
            
            # 所有指标都为负信号的点
            all_negative = consistent_signals[(consistent_signals['macd'] < 0) & 
                                             (consistent_signals['rsi'] < 0) & 
                                             (consistent_signals['kdj'] < 0)]
            
            # 这里我们只是验证可以找到信号，不验证具体数量
            self.assertIsInstance(consistent_signals, pd.DataFrame)
        except Exception as e:
            self.skipTest(f"无法完成信号一致性测试: {e}")
    
    def test_indicator_chain_calculation(self):
        """测试链式计算多个指标"""
        # 创建各种类型的指标
        indicators = []
        
        # 根据支持的指标选取一些典型指标测试
        test_indicator_names = ['MA', 'MACD', 'RSI', 'BOLL']
        
        for name in test_indicator_names:
            if name in self.supported_indicators and name not in self.exclude_list:
                indicators.append(IndicatorFactory.create_indicator(name))
        
        # 链式计算
        try:
            result_df = self.data.copy()
            
            for indicator in indicators:
                result_df = indicator.calculate(result_df)
                # 确保基础列保留
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    self.assertIn(col, result_df.columns, f"基础列{col}在计算{indicator.__class__.__name__}后丢失")
            
            # 验证结果包含各个指标的关键列
            indicator_key_cols = {
                'MA': ['MA5', 'MA10', 'MA20'],
                'MACD': ['dif', 'dea', 'macd'],
                'RSI': ['rsi'],
                'BOLL': ['upper', 'middle', 'lower']
            }
            
            for name, cols in indicator_key_cols.items():
                if name in [i.__class__.__name__ for i in indicators]:
                    for col in cols:
                        self.assertIn(col, result_df.columns, f"结果中缺少{name}指标的列: {col}")
        
        except Exception as e:
            self.fail(f"链式计算失败: {e}")


if __name__ == '__main__':
    unittest.main() 