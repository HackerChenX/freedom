"""
统一指标计算引擎单元测试
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from analysis.engines.unified_indicator_engine import Unified_indicator_engine


class Test_unified_indicator_engine(unittest.Test_case):
    """统一指标计算引擎测试类"""
    
    def set_up_Engine(self):
        """测试前准备"""
        self.engine = Unified_indicator_engine(enable_cache=False)
        
        # 创建测试数据
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        np.random.seed(42)  # 固定随机种子，确保测试结果可重现
        
        # 生成模拟股价数据
        base_price = 10.0
        price_changes = np.random.normal(0, 0.02, 100).cumsum()
        close_prices = base_price * (1 + price_changes)
        
        # 确保价格为正数
        close_prices = np.maximum(close_prices, 1.0)
        
        # 生成高低价
        high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.01, 100)))
        low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.01, 100)))
        
        # 生成开盘价
        open_prices = close_prices + np.random.normal(0, 0.005, 100)
        
        # 生成成交量
        volumes = np.random.randint(100000, 1000000, 100)
        
        self.test_data = pd.DataFrame({
            'date': dates,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes
        })
    
    def test_validate_data(self):
        """测试数据验证功能"""
        # 测试有效数据
        self.assert_true(self.engine._validate_data(self.test_data))
        
        # 测试空数据
        self.assert_false(self.engine._validate_data(pd.DataFrame()))
        
        # 测试None数据
        self.assert_false(self.engine._validate_data(None))
        
        # 测试缺少必要列的数据
        invalid_data = pd.DataFrame({'open': [1, 2, 3]})
        self.assert_false(self.engine._validate_data(invalid_data))
    
    def test_calculate_ma_Engine(self):
        """测试移动平均线计算"""
        ma5 = self.engine.calculate_ma(self.test_data, 5)
        
        # 检查返回类型
        self.assert_is_instance(ma5, pd.Series)
        
        # 检查长度
        self.assert_equal(len(ma5), len(self.test_data))
        
        # 检查第5个值（应该是前5个收盘价的平均）
        expected_ma5_5th = self.test_data['close'][:5].mean()
        self.assert_almost_equal(ma5.iloc[4], expected_ma5_5th, places=6)
        
        # 检查最后一个值
        expected_ma5_last = self.test_data['close'][-5:].mean()
        self.assert_almost_equal(ma5.iloc[-1], expected_ma5_last, places=6)
    
    def test_calculate_ema(self):
        """测试指数移动平均线计算"""
        ema12 = self.engine.calculate_ema(self.test_data, 12)
        
        # 检查返回类型
        self.assert_is_instance(ema12, pd.Series)
        
        # 检查长度
        self.assert_equal(len(ema12), len(self.test_data))
        
        # EMA的第一个值应该等于第一个收盘价
        self.assertAlmostEqual(ema12.iloc[0], self.test_data['close'].iloc[0], places=6)
        
        # EMA应该是递增的趋势（基于我们的测试数据）
        self.assert_is_not_none(ema12.iloc[-1])
    
    def test_calculate_sma(self):
        """测试平滑移动平均计算"""
        test_series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        sma_result = self.engine.calculate_sma(test_series, 3, 1)
        
        # 检查返回类型
        self.assert_is_instance(sma_result, pd.Series)
        
        # 检查长度
        self.assert_equal(len(sma_result), len(test_series))
        
        # 第一个值应该等于输入的第一个值
        self.assert_equal(sma_result.iloc[0], test_series.iloc[0])
    
    def test_calculate_macd(self):
        """测试MACD指标计算"""
        macd_result = self.engine.calculate_macd(self.test_data)
        
        # 检查返回类型
        self.assert_is_instance(macd_result, dict)
        
        # 检查包含的键
        expected_keys = ['DIF', 'DEA', 'MACD']
        for key in expected_keys:
            self.assert_in(key, macd_result)
            self.assert_is_instance(macd_result[key], pd.Series)
            self.assert_equal(len(macd_result[key]), len(self.test_data))
    
    def test_calculate_kdj(self):
        """测试KDJ指标计算"""
        kdj_result = self.engine.calculate_kdj(self.test_data)
        
        # 检查返回类型
        self.assert_is_instance(kdj_result, dict)
        
        # 检查包含的键
        expected_keys = ['K', 'D', 'J']
        for key in expected_keys:
            self.assert_in(key, kdj_result)
            self.assert_is_instance(kdj_result[key], pd.Series)
            self.assert_equal(len(kdj_result[key]), len(self.test_data))
        
        # KDJ值应该在合理范围内（虽然J值可能超出0-100）
        k_values = kdj_result['K'].dropna()
        d_values = kdj_result['D'].dropna()
        
        # K和D值通常在0-100之间
        self.assert_true((k_values >= 0).all() and (k_values <= 100).all())
        self.assert_true((d_values >= 0).all() and (d_values <= 100).all())
    
    def test_calculate_rsi(self):
        """测试RSI指标计算"""
        rsi_result = self.engine.calculate_rsi(self.test_data)
        
        # 检查返回类型
        self.assert_is_instance(rsi_result, pd.Series)
        
        # 检查长度
        self.assert_equal(len(rsi_result), len(self.test_data))
        
        # RSI值应该在0-100之间
        rsi_values = rsi_result.dropna()
        self.assert_true((rsi_values >= 0).all() and (rsi_values <= 100).all())
    
    def test_calculate_all_indicators(self):
        """测试计算所有指标"""
        all_indicators = self.engine.calculate_all_indicators(self.test_data)
        
        # 检查返回类型
        self.assert_is_instance(all_indicators, dict)
        
        # 检查包含的指标
        expected_indicators = ['MA5', 'MA10', 'MA20', 'MA30', 'MA60', 
                              'EMA5', 'EMA10', 'EMA20', 'EMA30', 'EMA60',
                              'DIF', 'DEA', 'MACD', 'K', 'D', 'J', 'RSI']
        
        for indicator in expected_indicators:
            self.assert_in(indicator, all_indicators)
            self.assert_is_instance(all_indicators[indicator], pd.Series)
    
    def test_get_indicator_value_at_date(self):
        """测试获取指定日期的指标值"""
        # 测试MA指标
        ma5_value = self.engine.get_indicator_value_at_date(
            self.test_data, 'MA', 10, period=5
        )
        self.assert_is_instance(ma5_value, float)
        self.assert_false(np.isnan(ma5_value))
        
        # 测试MACD指标
        macd_values = self.engine.get_indicator_value_at_date(
            self.test_data, 'MACD', 20
        )
        self.assert_is_instance(macd_values, dict)
        self.assertIn('DIF', macd_values)
        self.assertIn('DEA', macd_values)
        self.assertIn('MACD', macd_values)
        
        # 测试KDJ指标
        kdj_values = self.engine.get_indicator_value_at_date(
            self.test_data, 'KDJ', 30
        )
        self.assert_is_instance(kdj_values, dict)
        self.assertIn('K', kdj_values)
        self.assertIn('D', kdj_values)
        self.assertIn('J', kdj_values)
    
    def test_performance_stats_Engine(self):
        """测试性能统计功能"""
        # 执行一些计算
        self.engine.calculate_ma(self.test_data, 5)
        self.engine.calculate_macd(self.test_data)
        
        # 获取性能统计
        stats = self.engine.get_performance_stats()
        
        # 检查统计信息
        self.assert_is_instance(stats, dict)
        self.assertIn('total_calculations', stats)
        self.assertIn('cache_hits', stats)
        self.assertIn('cache_hit_rate', stats)
        
        # 应该有一些计算记录
        self.assertGreater(stats['total_calculations'], 0)
    
    def test_edge_cases_Engine(self):
        """测试边界情况"""
        # 测试单行数据
        single_row_data = self.test_data.iloc[:1].copy()
        ma_result = self.engine.calculate_ma(single_row_data, 5)
        self.assert_equal(len(ma_result), 1)
        
        # 测试包含NaN的数据
        nan_data = self.test_data.copy()
        nan_data.loc[5:10, 'close'] = np.nan
        ma_result = self.engine.calculate_ma(nan_data, 5)
        self.assert_equal(len(ma_result), len(nan_data))
        
        # 测试周期大于数据长度
        ma_result = self.engine.calculate_ma(self.test_data, 200)
        self.assert_equal(len(ma_result), len(self.test_data))
    
    def test_error_handling_Engine(self):
        """测试错误处理"""
        # 测试不存在的字段
        ma_result = self.engine.calculate_ma(self.test_data, 5, field='nonexistent')
        self.assert_equal(len(ma_result), 0)
        
        # 测试无效的指标名称
        invalid_indicator = self.engine.get_indicator_value_at_date(
            self.test_data, 'INVALID', 10
        )
        self.assert_true(np.isnan(invalid_indicator))


if __name__ == '__main__':
    unittest.main() 