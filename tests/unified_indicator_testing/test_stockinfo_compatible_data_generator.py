#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
StockInfoCompatibleDataGenerator的单元测试

验证StockInfo兼容数据生成器的功能
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from tests.unified_indicator_testing.components.stockinfo_compatible_data_generator import StockInfoCompatibleDataGenerator


class TestStockInfoCompatibleDataGenerator(unittest.TestCase):
    """StockInfo兼容数据生成器的单元测试"""
    
    def setUp(self):
        """测试前准备"""
        self.generator = StockInfoCompatibleDataGenerator()
    
    def tearDown(self):
        """测试后清理"""
        self.generator.cleanup()
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.generator)
        self.assertIsNotNone(self.generator.stockinfo_fields)
        self.assertIsNotNone(self.generator.indicator_history_requirements)
        self.assertIsNotNone(self.generator.industries)
        
        # 验证支持的指标
        supported_indicators = self.generator.get_supported_indicators()
        self.assertIn('MACD', supported_indicators)
        self.assertIn('RSI', supported_indicators)
        self.assertIn('KDJ', supported_indicators)
    
    def test_generate_stockinfo_compatible_data(self):
        """测试生成StockInfo兼容数据"""
        data = self.generator.generate_stockinfo_compatible_data(
            indicator_name='MACD',
            pattern_type='GOLDEN_CROSS',
            stock_code='TEST001',
            history_days=60
        )
        
        # 验证数据结构
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(len(data), 0)
        
        # 验证必需字段
        required_fields = ['date', 'code', 'name', 'open', 'high', 'low', 'close', 'volume']
        for field in required_fields:
            self.assertIn(field, data.columns, f"缺少必需字段: {field}")
        
        # 验证数据类型
        self.assertTrue(data['open'].dtype in ['float64', 'float32'])
        self.assertTrue(data['high'].dtype in ['float64', 'float32'])
        self.assertTrue(data['low'].dtype in ['float64', 'float32'])
        self.assertTrue(data['close'].dtype in ['float64', 'float32'])
        self.assertTrue(data['volume'].dtype in ['float64', 'float32'])
        
        # 验证股票代码
        self.assertTrue((data['code'] == 'TEST001').all())
        
        # 验证价格逻辑
        for i in range(len(data)):
            high = data.iloc[i]['high']
            low = data.iloc[i]['low']
            open_price = data.iloc[i]['open']
            close = data.iloc[i]['close']
            
            self.assertLessEqual(low, open_price, f"第{i}行: low > open")
            self.assertLessEqual(open_price, high, f"第{i}行: open > high")
            self.assertLessEqual(low, close, f"第{i}行: low > close")
            self.assertLessEqual(close, high, f"第{i}行: close > high")
    
    def test_generate_random_stockinfo_data(self):
        """测试生成随机StockInfo数据"""
        data = self.generator.generate_random_stockinfo_data('RANDOM001', 30)
        
        # 验证数据结构
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), 30)
        
        # 验证必需字段
        required_fields = ['date', 'code', 'name', 'open', 'high', 'low', 'close', 'volume']
        for field in required_fields:
            self.assertIn(field, data.columns)
        
        # 验证股票代码
        self.assertTrue((data['code'] == 'RANDOM001').all())
    
    def test_indicator_history_requirements(self):
        """测试指标历史数据需求"""
        # 测试已知指标
        self.assertEqual(self.generator.get_indicator_history_requirement('MACD'), 60)
        self.assertEqual(self.generator.get_indicator_history_requirement('RSI'), 30)
        self.assertEqual(self.generator.get_indicator_history_requirement('KDJ'), 20)
        
        # 测试未知指标（应返回默认值）
        self.assertEqual(self.generator.get_indicator_history_requirement('UNKNOWN'), 60)
    
    def test_stockinfo_structure_validation(self):
        """测试stockInfo结构验证"""
        # 生成测试数据
        data = self.generator.generate_stockinfo_compatible_data(
            'RSI', 'OVERSOLD', 'VALID001', 30
        )
        
        # 验证结构
        is_valid = self.generator._validate_stockinfo_structure(data)
        self.assertTrue(is_valid)
        
        # 测试无效数据
        invalid_data = pd.DataFrame({
            'code': ['INVALID001'],
            'name': ['无效股票']
            # 缺少必需字段
        })
        
        is_valid = self.generator._validate_stockinfo_structure(invalid_data)
        self.assertFalse(is_valid)
    
    def test_data_compatibility_validation(self):
        """测试数据兼容性验证"""
        # 生成测试数据
        data = self.generator.generate_stockinfo_compatible_data(
            'BOLL', 'UPPER_BREAKOUT', 'COMPAT001', 40
        )
        
        # 验证兼容性
        report = self.generator.validate_data_compatibility(data)
        
        self.assertIsInstance(report, dict)
        self.assertIn('is_compatible', report)
        self.assertIn('missing_fields', report)
        self.assertIn('invalid_data_types', report)
        self.assertIn('data_quality_issues', report)
        self.assertIn('recommendations', report)
        
        # 应该是兼容的
        self.assertTrue(report['is_compatible'])
        self.assertEqual(len(report['missing_fields']), 0)
    
    def test_large_scale_data_generation(self):
        """测试大规模数据生成"""
        stock_codes = [f'LARGE{i:03d}' for i in range(10)]  # 测试10只股票
        
        results = self.generator.generate_large_scale_data(
            stock_codes=stock_codes,
            indicator_name='VOL',
            pattern_type='VOLUME_SPIKE',
            history_days=20
        )
        
        # 验证结果
        self.assertIsInstance(results, dict)
        self.assertGreaterEqual(len(results), 8)  # 至少80%成功率
        
        # 验证每个结果
        for stock_code, data in results.items():
            self.assertIn(stock_code, stock_codes)
            self.assertIsInstance(data, pd.DataFrame)
            self.assertGreater(len(data), 0)
            self.assertTrue((data['code'] == stock_code).all())
    
    def test_price_series_generation(self):
        """测试价格序列生成"""
        base_price = 20.0
        length = 50
        
        prices = self.generator._generate_realistic_price_series(base_price, length)
        
        # 验证价格序列
        self.assertEqual(len(prices), length)
        self.assertEqual(prices[0], base_price)
        
        # 验证价格在合理范围内
        for price in prices:
            self.assertGreater(price, 0)
            self.assertLess(price, 1000)
    
    def test_volume_series_generation(self):
        """测试成交量序列生成"""
        length = 30
        
        volumes = self.generator._generate_realistic_volume_series(length)
        
        # 验证成交量序列
        self.assertEqual(len(volumes), length)
        
        # 验证成交量在合理范围内
        for volume in volumes:
            self.assertGreater(volume, 0)
            self.assertLess(volume, 100000000)
    
    def test_data_type_conversion(self):
        """测试数据类型转换"""
        # 创建测试数据
        test_data = pd.DataFrame({
            'date': ['20250721', '20250722'],
            'code': ['TYPE001', 'TYPE001'],
            'name': ['类型测试', '类型测试'],
            'open': ['10.5', '11.0'],  # 字符串格式
            'high': [11.0, 12.0],
            'low': [10.0, 10.5],
            'close': [10.8, 11.5],
            'volume': [1000000, 1200000],
            'seq': [1, 2]
        })
        
        # 转换数据类型
        converted_data = self.generator._ensure_correct_dtypes(test_data)
        
        # 验证转换结果
        self.assertTrue(converted_data['open'].dtype in ['float64', 'float32'])
        self.assertTrue(converted_data['seq'].dtype in ['int32', 'int64'])
        self.assertEqual(converted_data['date'].dtype, 'object')
    
    def test_error_handling(self):
        """测试错误处理"""
        # 测试无效参数
        try:
            data = self.generator.generate_stockinfo_compatible_data(
                indicator_name='',
                pattern_type='',
                stock_code='',
                history_days=0
            )
            # 应该返回有效数据，即使参数无效
            self.assertIsInstance(data, pd.DataFrame)
        except Exception as e:
            self.fail(f"错误处理失败: {e}")
    
    def test_industry_assignment(self):
        """测试行业分配"""
        data = self.generator.generate_stockinfo_compatible_data(
            'CCI', 'OVERBOUGHT', 'INDUSTRY001', 25
        )
        
        # 验证行业字段
        self.assertIn('industry', data.columns)
        
        # 验证行业值在预定义列表中
        for industry in data['industry'].unique():
            self.assertIn(industry, self.generator.industries)
    
    def test_date_format(self):
        """测试日期格式"""
        data = self.generator.generate_stockinfo_compatible_data(
            'WR', 'REVERSAL', 'DATE001', 15
        )
        
        # 验证日期格式
        for date_str in data['date']:
            # 应该是YYYYMMDD格式
            self.assertEqual(len(date_str), 8)
            self.assertTrue(date_str.isdigit())
            
            # 验证可以解析为日期
            try:
                datetime.strptime(date_str, '%Y%m%d')
            except ValueError:
                self.fail(f"无效的日期格式: {date_str}")


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)
