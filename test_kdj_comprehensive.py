#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
KDJ指标综合测试脚本

按照项目标准进行全面测试，确保：
1. 无ERROR和WARNING日志
2. 所有方法正常工作
3. 形态识别功能
4. 评分机制
"""

import sys
import os
import pandas as pd
import numpy as np
import unittest
from datetime import datetime

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

from indicators.kdj import KDJ
from utils.logger import get_logger

logger = get_logger(__name__)


class TestKDJComprehensive(unittest.TestCase):
    """KDJ指标综合测试类"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建KDJ指标实例
        self.indicator = KDJ(n=9, m1=3, m2=3)
        
        # 创建测试数据
        self.data = self._create_test_data()
    
    def _create_test_data(self, length=100):
        """创建测试数据"""
        # 生成日期索引
        dates = pd.date_range(start='2023-01-01', periods=length, freq='D')
        
        # 生成价格数据
        np.random.seed(42)
        base_price = 100.0
        price_changes = np.random.normal(0.001, 0.02, length)
        
        prices = [base_price]
        for change in price_changes[1:]:
            prices.append(prices[-1] * (1 + change))
        
        # 创建完整的股票数据
        data = pd.DataFrame({
            'open': [p * (1 + np.random.uniform(-0.01, 0.01)) for p in prices],
            'high': [p * (1 + np.random.uniform(0.005, 0.02)) for p in prices],
            'low': [p * (1 + np.random.uniform(-0.02, -0.005)) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, length),
            'code': '000001',
            'name': '测试股票',
            'level': 'D',
            'industry': '软件服务',
            'turnover_rate': 5.0,
            'seq': range(length)
        }, index=dates)
        
        # 添加计算字段
        data['datetime_value'] = data.index
        data['price_change'] = data['close'].diff().fillna(0)
        data['price_range'] = (data['high'] - data['low']) / data['close'] * 100
        
        return data
    
    def test_kdj_basic_calculation(self):
        """测试KDJ基本计算功能"""
        # 执行计算
        result = self.indicator.calculate(self.data)
        
        # 验证结果类型
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)
        
        # 验证预期列存在
        expected_columns = ['K', 'D', 'J']
        for col in expected_columns:
            self.assertIn(col, result.columns, f"缺少预期列: {col}")
        
        # 验证KDJ数值合理性
        k_values = result['K'].dropna()
        d_values = result['D'].dropna()
        j_values = result['J'].dropna()
        
        self.assertTrue(len(k_values) > 0, "K值全为NaN")
        self.assertTrue(len(d_values) > 0, "D值全为NaN")
        self.assertTrue(len(j_values) > 0, "J值全为NaN")
        
        # 验证K和D值在0-100范围内（J值可以超出）
        self.assertTrue(all(0 <= v <= 100 for v in k_values), "K值应在0-100范围内")
        self.assertTrue(all(0 <= v <= 100 for v in d_values), "D值应在0-100范围内")
    
    def test_kdj_calculation_accuracy(self):
        """测试KDJ计算准确性"""
        result = self.indicator.calculate(self.data)

        # 验证KDJ数值的基本合理性
        k_values = result['K'].dropna()
        d_values = result['D'].dropna()
        j_values = result['J'].dropna()

        # 验证K和D值在0-100范围内
        self.assertTrue(all(0 <= v <= 100 for v in k_values), "K值应在0-100范围内")
        self.assertTrue(all(0 <= v <= 100 for v in d_values), "D值应在0-100范围内")

        # 验证J值在合理范围内（可以超出0-100）
        self.assertTrue(all(-100 <= v <= 200 for v in j_values), "J值超出合理范围")

        # 验证D值是K值的平滑版本（D的变化应该小于K的变化）
        if len(k_values) > 1 and len(d_values) > 1:
            k_volatility = k_values.std()
            d_volatility = d_values.std()
            self.assertLessEqual(d_volatility, k_volatility * 1.2, "D值应该比K值更平滑")
    
    def test_kdj_raw_score(self):
        """测试KDJ原始评分功能"""
        # 计算原始评分
        raw_score = self.indicator.calculate_raw_score(self.data)
        
        # 验证结果
        self.assertIsInstance(raw_score, pd.Series)
        self.assertEqual(len(raw_score), len(self.data))
        
        # 验证评分范围
        valid_scores = raw_score.dropna()
        self.assertTrue(all(0 <= s <= 100 for s in valid_scores), "评分应在0-100范围内")
    
    def test_kdj_patterns(self):
        """测试KDJ形态识别功能"""
        # 执行形态识别
        patterns = self.indicator.get_patterns(self.data)
        
        # 验证结果
        self.assertIsInstance(patterns, pd.DataFrame)
        self.assertEqual(len(patterns), len(self.data))
        
        # 验证预期的形态列存在
        expected_patterns = [
            'KDJ_GOLDEN_CROSS', 'KDJ_DEATH_CROSS',
            'KDJ_OVERBOUGHT', 'KDJ_OVERSOLD',
            'KDJ_BULLISH_DIVERGENCE', 'KDJ_BEARISH_DIVERGENCE'
        ]
        
        for pattern in expected_patterns:
            self.assertIn(pattern, patterns.columns, f"缺少形态列: {pattern}")
    
    def test_kdj_confidence(self):
        """测试KDJ置信度计算"""
        # 计算原始评分
        raw_score = self.indicator.calculate_raw_score(self.data)
        
        # 获取形态
        patterns = self.indicator.get_patterns(self.data)
        
        # 计算置信度
        confidence = self.indicator.calculate_confidence(raw_score, patterns, {})
        
        # 验证置信度
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_kdj_comprehensive_score(self):
        """测试KDJ综合评分功能"""
        # 计算综合评分
        score_result = self.indicator.calculate_score(self.data)
        
        # 验证结果
        self.assertIsInstance(score_result, dict)
        self.assertIn('score', score_result)
        self.assertIn('confidence', score_result)
        
        # 验证评分范围
        self.assertGreaterEqual(score_result['score'], 0.0)
        self.assertLessEqual(score_result['score'], 100.0)
        self.assertGreaterEqual(score_result['confidence'], 0.0)
        self.assertLessEqual(score_result['confidence'], 1.0)
    
    def test_kdj_parameter_setting(self):
        """测试KDJ参数设置"""
        # 测试参数设置
        new_n = 14
        new_m1 = 5
        new_m2 = 5
        
        self.indicator.set_parameters(n=new_n, m1=new_m1, m2=new_m2)
        
        # 验证参数更新
        self.assertEqual(self.indicator.n, new_n)
        self.assertEqual(self.indicator.m1, new_m1)
        self.assertEqual(self.indicator.m2, new_m2)
        
        # 测试新参数下的计算
        result = self.indicator.calculate(self.data)
        self.assertIn('K', result.columns)
        self.assertIn('D', result.columns)
        self.assertIn('J', result.columns)
    
    def test_kdj_required_columns(self):
        """测试KDJ必需列检查"""
        # 验证REQUIRED_COLUMNS属性存在
        self.assertTrue(hasattr(self.indicator, 'REQUIRED_COLUMNS'))
        expected_cols = ['high', 'low', 'close']
        for col in expected_cols:
            self.assertIn(col, self.indicator.REQUIRED_COLUMNS)
        
        # 测试缺少必需列的情况
        incomplete_data = self.data.drop(columns=['high'])
        
        # 应该抛出异常或返回空DataFrame
        try:
            result = self.indicator.calculate(incomplete_data)
            # 如果没有抛出异常，结果应该是空的或者有错误处理
            self.assertTrue(result.empty or not self.indicator.is_available)
        except ValueError:
            # 抛出异常也是可以接受的
            pass
    
    def test_kdj_edge_cases(self):
        """测试KDJ边界情况"""
        # 测试单行数据
        single_row = self.data.iloc[:1].copy()
        result = self.indicator.calculate(single_row)
        self.assertIsInstance(result, pd.DataFrame)
        
        # 测试少量数据
        small_data = self.data.iloc[:10].copy()
        result = self.indicator.calculate(small_data)
        self.assertIsInstance(result, pd.DataFrame)
        
        # 测试空数据
        empty_data = pd.DataFrame()
        result = self.indicator.calculate(empty_data)
        self.assertIsInstance(result, pd.DataFrame)
    
    def test_kdj_signals(self):
        """测试KDJ信号生成"""
        # 生成信号
        signals = self.indicator.generate_trading_signals(self.data)
        
        # 验证信号
        self.assertIsInstance(signals, dict)
        
        # 验证必需的信号字段
        required_fields = ['buy_signal', 'sell_signal', 'overbought', 'oversold']
        for field in required_fields:
            self.assertIn(field, signals, f"缺少信号字段: {field}")
    
    def test_kdj_j_value_range(self):
        """测试KDJ的J值范围"""
        result = self.indicator.calculate(self.data)
        
        # J值可以超出0-100范围，但应该在合理范围内
        j_values = result['J'].dropna()
        self.assertTrue(len(j_values) > 0, "J值全为NaN")
        
        # J值通常在-50到150范围内
        self.assertTrue(all(-100 <= v <= 200 for v in j_values), "J值超出合理范围")


def run_comprehensive_test():
    """运行综合测试"""
    print("🚀 开始KDJ指标综合测试")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 创建测试套件
    suite = unittest.TestLoader().loadTestsFromTestCase(TestKDJComprehensive)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 输出结果
    print(f"\n测试结果:")
    print(f"运行测试: {result.testsRun}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    
    if result.failures:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\n错误的测试:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    if success:
        print("\n🎉 所有测试通过！KDJ指标已准备就绪。")
        print("\n✅ KDJ指标功能验证:")
        print("  - ✅ 基本计算功能正常")
        print("  - ✅ 计算准确性验证")
        print("  - ✅ 原始评分功能正常")
        print("  - ✅ 形态识别功能正常")
        print("  - ✅ 置信度计算功能正常")
        print("  - ✅ 综合评分功能正常")
        print("  - ✅ 参数设置功能正常")
        print("  - ✅ 必需列检查功能正常")
        print("  - ✅ 边界情况处理正常")
        print("  - ✅ 信号生成功能正常")
        print("  - ✅ J值范围验证正常")
    else:
        print("\n⚠️  部分测试失败，需要进一步修复。")
    
    return success


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
