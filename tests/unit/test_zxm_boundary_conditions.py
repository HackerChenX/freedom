"""
ZXM指标边界条件处理测试模块

专门测试ZXM指标在边界条件下的稳定性和错误处理能力
包括：Na_n值、数据不足、极值情况等
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 导入需要测试的ZXM指标
from indicators.zxm.buy_point_indicators import ZXMturnover_rate, ZXMVolume_shrink, ZXMBSAbsorb
from indicators.zxm.trend_indicators import ZXMDaily_trend_up, ZXMWeekly_trend_up
from indicators.zxm.elasticity_indicators import Amplitude_elasticity, ZXMRise_elasticity
from indicators.zxm.score_indicators import ZXMElasticity_score, ZXMBuy_point_score, Stock_score_calculator
from db.sql_manager import SQLManager, QueryType


class Test_zXMBoundary_conditions(unittest.TestCase):
    """ZXM指标边界条件测试类"""
    
    def setUp(self):
        """设置测试数据"""
        self.boundary_scenarios = self._generate_boundary_test_scenarios()
    
    def _generate_boundary_test_scenarios(self):
        """生成边界条件测试场景"""
        scenarios = {}
        
        # 场景1：数据不足
        scenarios['insufficient_data'] = self._create_insufficient_data_scenario()
        
        # 场景2：包含NaN值
        scenarios['nan_values'] = self._create_nan_values_scenario()
        
        # 场景3：极值数据
        scenarios['extreme_values'] = self._create_extreme_values_scenario()
        
        # 场景4：零值数据
        scenarios['zero_values'] = self._create_zero_values_scenario()
        
        # 场景5：单一值数据
        scenarios['constant_values'] = self._create_constant_values_scenario()
        
        # 场景6：空数据
        scenarios['empty_data'] = self._create_empty_data_scenario()
        
        return scenarios
    
    def _create_insufficient_data_scenario(self):
        """创建数据不足场景：只有5个数据点"""
        dates = pd.date_range(start='2023-01-01', periods=5, freq='D')
        data = pd.DataFrame({
            'datetime': dates,
            'open': [100, 101, 102, 103, 104],
            'high': [105, 106, 107, 108, 109],
            'low': [95, 96, 97, 98, 99],
            'close': [100, 101, 102, 103, 104],
            'volume': [1000000, 1100000, 1200000, 1300000, 1400000],
            'turnover_rate': [0.5, 0.6, 0.7, 0.8, 0.9],
        })
        return data
    
    def _create_nan_values_scenario(self):
        """创建包含NaN值的场景"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'datetime': dates,
            'open': [100] * 100,
            'high': [105] * 100,
            'low': [95] * 100,
            'close': [100] * 100,
            'volume': [1000000] * 100,
            'turnover_rate': [0.5] * 100,
        })
        
        # 在关键位置插入NaN值
        data.loc[10:15, 'close'] = np.nan
        data.loc[20:25, 'volume'] = np.nan
        data.loc[30:35, 'turnover_rate'] = np.nan
        
        return data
    
    def _create_extreme_values_scenario(self):
        """创建极值数据场景"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'datetime': dates,
            'open': [100] * 100,
            'high': [105] * 100,
            'low': [95] * 100,
            'close': [100] * 100,
            'volume': [1000000] * 100,
            'turnover_rate': [0.5] * 100,
        })
        
        # 插入极值
        data.loc[10, 'close'] = 1e10  # 极大值
        data.loc[20, 'close'] = 1e-10  # 极小值
        data.loc[30, 'volume'] = 1e15  # 极大成交量
        data.loc[40, 'turnover_rate'] = 1000  # 极大换手率
        
        return data
    
    def _create_zero_values_scenario(self):
        """创建零值数据场景"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'datetime': dates,
            'open': [100] * 100,
            'high': [105] * 100,
            'low': [95] * 100,
            'close': [100] * 100,
            'volume': [1000000] * 100,
            'turnover_rate': [0.5] * 100,
        })
        
        # 插入零值
        data.loc[10:15, 'volume'] = 0
        data.loc[20:25, 'turnover_rate'] = 0
        
        return data
    
    def _create_constant_values_scenario(self):
        """创建单一值数据场景：所有价格都相同"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'datetime': dates,
            'open': [100] * 100,
            'high': [100] * 100,  # 所有价格相同
            'low': [100] * 100,
            'close': [100] * 100,
            'volume': [1000000] * 100,
            'turnover_rate': [0.5] * 100,
        })
        return data
    
    def _create_empty_data_scenario(self):
        """创建空数据场景"""
        return pd.DataFrame()
    
    def test_insufficient_data_handling(self):
        """测试数据不足情况的处理"""
        print("=== 测试数据不足情况处理 ===")
        
        insufficient_data = self.boundary_scenarios['insufficient_data']
        indicators_to_test = [
            ('ZXMturnover_rate', ZXMturnover_rate()),
            ('ZXMVolumeShrink', ZXMVolumeShrink()),
            ('ZXMDailyTrendUp', ZXMDailyTrendUp()),
            ('AmplitudeElasticity', AmplitudeElasticity()),
        ]
        
        for indicator_name, indicator in indicators_to_test:
            with self.sub_test(indicator=indicator_name):
                try:
                    result = indicator.calculate(insufficient_data)
                    
                    # 验证结果不为空且包含必要的信号列
                    self.assertIsInstance(result, pd.DataFrame, f"{indicator_name}应该返回DataFrame")
                    self.assertIn('buy_signal', result.columns, f"{indicator_name}应该包含buy_signal列")
                    self.assertIn('sell_signal', result.columns, f"{indicator_name}应该包含sell_signal列")
                    self.assertIn('hold_signal', result.columns, f"{indicator_name}应该包含hold_signal列")
                    
                    # 验证信号列的数据类型
                    self.assertEqual(result['buy_signal'].dtype, bool, f"{indicator_name}的buy_signal应该是布尔类型")
                    self.assertEqual(result['sell_signal'].dtype, bool, f"{indicator_name}的sell_signal应该是布尔类型")
                    self.assertEqual(result['hold_signal'].dtype, bool, f"{indicator_name}的hold_signal应该是布尔类型")
                    
                    print(f"✅ {indicator_name} 数据不足处理正常")
                    
                except Exception as e:
                    self.fail(f"{indicator_name} 在数据不足时应该优雅处理，但抛出异常: {e}")
    
    def test_nan_values_handling(self):
        """测试NaN值处理"""
        print("=== 测试NaN值处理 ===")
        
        nan_data = self.boundary_scenarios['nan_values']
        indicators_to_test = [
            ('ZXMturnover_rate', ZXMturnover_rate()),
            ('ZXMVolumeShrink', ZXMVolumeShrink()),
            ('AmplitudeElasticity', AmplitudeElasticity()),
        ]
        
        for indicator_name, indicator in indicators_to_test:
            with self.sub_test(indicator=indicator_name):
                try:
                    result = indicator.calculate(nan_data)
                    
                    # 验证结果不为空
                    self.assertIsInstance(result, pd.DataFrame, f"{indicator_name}应该返回DataFrame")
                    self.assertGreater(len(result), 0, f"{indicator_name}结果不应该为空")
                    
                    # 验证信号列存在且类型正确
                    self.assertIn('buy_signal', result.columns, f"{indicator_name}应该包含buy_signal列")
                    self.assertEqual(result['buy_signal'].dtype, bool, f"{indicator_name}的buy_signal应该是布尔类型")
                    
                    # 验证没有无限值
                    numeric_columns = result.select_dtypes(include=[np.number]).columns
                    for col in numeric_columns:
                        self.assertFalse(np.isinf(result[col]).any(), f"{indicator_name}的{col}列不应该包含无限值")
                    
                    print(f"✅ {indicator_name} NaN值处理正常")
                    
                except Exception as e:
                    self.fail(f"{indicator_name} 在包含NaN值时应该优雅处理，但抛出异常: {e}")
    
    def test_extreme_values_handling(self):
        """测试极值处理"""
        print("=== 测试极值处理 ===")
        
        extreme_data = self.boundary_scenarios['extreme_values']
        indicators_to_test = [
            ('ZXMturnover_rate', ZXMturnover_rate()),
            ('AmplitudeElasticity', AmplitudeElasticity()),
            ('StockScoreCalculator', StockScoreCalculator()),
        ]
        
        for indicator_name, indicator in indicators_to_test:
            with self.sub_test(indicator=indicator_name):
                try:
                    result = indicator.calculate(extreme_data)
                    
                    # 验证结果稳定性
                    self.assertIsInstance(result, pd.DataFrame, f"{indicator_name}应该返回DataFrame")
                    
                    # 验证没有无限值或NaN值在关键列中
                    if 'buy_signal' in result.columns:
                        self.assertFalse(result['buy_signal'].isna().any(), f"{indicator_name}的buy_signal不应该包含NaN")
                    
                    print(f"✅ {indicator_name} 极值处理正常")
                    
                except Exception as e:
                    self.fail(f"{indicator_name} 在极值数据时应该优雅处理，但抛出异常: {e}")
    
    def test_zero_values_handling(self):
        """测试零值处理"""
        print("=== 测试零值处理 ===")
        
        zero_data = self.boundary_scenarios['zero_values']
        indicators_to_test = [
            ('ZXMturnover_rate', ZXMturnover_rate()),
            ('ZXMVolumeShrink', ZXMVolumeShrink()),
        ]
        
        for indicator_name, indicator in indicators_to_test:
            with self.sub_test(indicator=indicator_name):
                try:
                    result = indicator.calculate(zero_data)
                    
                    # 验证零值不会导致除零错误
                    self.assertIsInstance(result, pd.DataFrame, f"{indicator_name}应该返回DataFrame")
                    
                    # 验证信号列正常
                    if 'buy_signal' in result.columns:
                        self.assertEqual(result['buy_signal'].dtype, bool, f"{indicator_name}的buy_signal应该是布尔类型")
                    
                    print(f"✅ {indicator_name} 零值处理正常")
                    
                except Exception as e:
                    self.fail(f"{indicator_name} 在零值数据时应该优雅处理，但抛出异常: {e}")
    
    def test_constant_values_handling(self):
        """测试单一值处理"""
        print("=== 测试单一值处理 ===")
        
        constant_data = self.boundary_scenarios['constant_values']
        indicators_to_test = [
            ('AmplitudeElasticity', AmplitudeElasticity()),
            ('ZXMRiseElasticity', ZXMRiseElasticity()),
        ]
        
        for indicator_name, indicator in indicators_to_test:
            with self.sub_test(indicator=indicator_name):
                try:
                    result = indicator.calculate(constant_data)
                    
                    # 验证单一值不会导致计算错误
                    self.assertIsInstance(result, pd.DataFrame, f"{indicator_name}应该返回DataFrame")
                    
                    # 对于弹性指标，单一值应该产生False信号
                    if 'XG' in result.columns:
                        # 单一价格应该没有弹性
                        self.assertFalse(result['XG'].any(), f"{indicator_name}在单一价格时XG应该为False")
                    
                    print(f"✅ {indicator_name} 单一值处理正常")
                    
                except Exception as e:
                    self.fail(f"{indicator_name} 在单一值数据时应该优雅处理，但抛出异常: {e}")
    
    def test_empty_data_handling(self):
        """测试空数据处理"""
        print("=== 测试空数据处理 ===")
        
        empty_data = self.boundary_scenarios['empty_data']
        indicators_to_test = [
            ('ZXMturnover_rate', ZXMturnover_rate()),
            ('AmplitudeElasticity', AmplitudeElasticity()),
        ]
        
        for indicator_name, indicator in indicators_to_test:
            with self.sub_test(indicator=indicator_name):
                try:
                    result = indicator.calculate(empty_data)
                    
                    # 空数据应该返回空DataFrame或包含默认值的DataFrame
                    self.assertIsInstance(result, pd.DataFrame, f"{indicator_name}应该返回DataFrame")
                    
                    print(f"✅ {indicator_name} 空数据处理正常")
                    
                except Exception as e:
                    # 空数据可能抛出异常，这是可以接受的
                    print(f"ℹ️ {indicator_name} 在空数据时抛出异常（可接受）: {e}")


if __name__ == '__main__':
    unittest.main()
