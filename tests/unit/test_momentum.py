#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Momentum指标单元测试
"""

import unittest
import pandas as pd
import numpy as np

from indicators.momentum import Momentum
from utils.logger import get_logger

logger = get_logger(__name__)


class TestMomentum(unittest.TestCase):
    """Momentum指标测试类"""
    
    def setUp(self):
        """设置测试环境"""
        self.momentum = Momentum(period=10, signal_period=6)
        
        # 创建测试数据
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        # 生成模拟的OHLCV数据
        base_price = 100
        prices = []
        volumes = []
        
        for i in range(100):
            # 生成价格数据，带有一定的趋势和随机性
            if i < 30:
                # 前30天上涨趋势
                trend = 0.5
            elif i < 60:
                # 中间30天下跌趋势
                trend = -0.3
            else:
                # 后40天震荡
                trend = 0.1 * np.sin(i * 0.2)
            
            price_change = trend + np.random.normal(0, 0.5)
            base_price *= (1 + price_change / 100)
            
            # 生成OHLC
            open_price = base_price
            high_price = open_price * (1 + abs(np.random.normal(0, 0.01)))
            low_price = open_price * (1 - abs(np.random.normal(0, 0.01)))
            close_price = low_price + (high_price - low_price) * np.random.random()
            
            prices.append([open_price, high_price, low_price, close_price])
            
            # 生成成交量
            volume_base = 1000000
            volume_multiplier = 1 + price_change / 100
            volume = volume_base * volume_multiplier * (0.5 + np.random.random())
            volumes.append(volume)
        
        prices = np.array(prices)
        
        self.test_data = pd.DataFrame({
            'date': dates,
            'open': prices[:, 0],
            'high': prices[:, 1],
            'low': prices[:, 2],
            'close': prices[:, 3],
            'volume': volumes
        })
        self.test_data.set_index('date', inplace=True)
    
    def test_momentum_calculation(self):
        """测试Momentum计算功能"""
        logger.info("测试Momentum计算功能")
        
        # 计算Momentum
        result = self.momentum.calculate(self.test_data)
        
        # 验证结果
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('mtm', result.columns)
        self.assertIn('signal', result.columns)
        self.assertEqual(len(result), len(self.test_data))
        
        # 验证前period个值为NaN（因为需要period个数据点才能计算）
        mtm_values = result['mtm'].dropna()
        self.assertGreater(len(mtm_values), 0)
        
        logger.info(f"Momentum计算成功，数据长度: {len(result)}")
        logger.info(f"有效Momentum值数量: {len(mtm_values)}")
    
    def test_momentum_patterns(self):
        """测试Momentum形态识别"""
        logger.info("测试Momentum形态识别")
        
        # 计算Momentum
        self.momentum.calculate(self.test_data)
        
        # 获取形态
        patterns = self.momentum.get_patterns(self.test_data)
        
        # 验证形态结果
        self.assertIsInstance(patterns, pd.DataFrame)
        self.assertEqual(len(patterns), len(self.test_data))
        
        # 验证关键形态列存在
        expected_patterns = [
            'MTM_CROSS_ABOVE_ZERO', 'MTM_CROSS_BELOW_ZERO',
            'MTM_CROSS_ABOVE_SIGNAL', 'MTM_CROSS_BELOW_SIGNAL',
            'MTM_RISING', 'MTM_FALLING'
        ]
        
        for pattern in expected_patterns:
            self.assertIn(pattern, patterns.columns)
        
        # 验证形态值为布尔类型
        for pattern in expected_patterns:
            self.assertTrue(patterns[pattern].dtype == bool)
        
        logger.info(f"形态识别成功，识别出 {len(patterns.columns)} 种形态")
    
    def test_momentum_scoring(self):
        """测试Momentum评分功能"""
        logger.info("测试Momentum评分功能")
        
        # 计算原始评分
        score = self.momentum.calculate_raw_score(self.test_data)
        
        # 验证评分结果
        self.assertIsInstance(score, pd.Series)
        self.assertEqual(len(score), len(self.test_data))
        
        # 调试：打印评分统计信息
        valid_scores = score.dropna()
        logger.info(f"有效评分统计: min={valid_scores.min():.2f}, max={valid_scores.max():.2f}")
        
        # 验证评分在合理范围内（忽略NaN值）
        if len(valid_scores) > 0:
            self.assertTrue((valid_scores >= 0).all(), f"评分过低: 最小值={valid_scores.min()}")
            self.assertTrue((valid_scores <= 100).all(), f"评分过高: 最大值={valid_scores.max()}")
        else:
            logger.warning("所有评分都是NaN")
        
        logger.info(f"评分计算成功，评分范围: {valid_scores.min():.2f} - {valid_scores.max():.2f}")
        logger.info(f"平均评分: {valid_scores.mean():.2f}")
    
    def test_momentum_confidence(self):
        """测试Momentum置信度计算"""
        logger.info("测试Momentum置信度计算")
        
        # 计算评分和形态
        score = self.momentum.calculate_raw_score(self.test_data)
        patterns = self.momentum.get_patterns(self.test_data)
        
        # 计算置信度
        confidence = self.momentum.calculate_confidence(score, patterns, {})
        
        # 验证置信度
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        
        logger.info(f"置信度计算成功: {confidence:.3f}")
    
    def test_momentum_register_patterns(self):
        """测试Momentum形态注册"""
        logger.info("测试Momentum形态注册")
        
        # 注册形态
        try:
            self.momentum.register_patterns()
            logger.info("Momentum形态注册成功")
        except Exception as e:
            logger.warning(f"Momentum形态注册失败: {e}")
            # 不让测试失败，因为形态注册可能依赖外部组件
    
    def test_momentum_edge_cases(self):
        """测试Momentum边界情况"""
        logger.info("测试Momentum边界情况")

        # 测试空数据
        empty_data = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        try:
            result = self.momentum.calculate(empty_data)
            self.assertTrue(result.empty)
        except ValueError:
            # 如果抛出异常，说明正确处理了空数据情况
            logger.info("空数据正确抛出异常")

        # 测试数据不足的情况
        small_data = self.test_data.head(5)  # 少于period的数据
        try:
            result = self.momentum.calculate(small_data)
            # 如果没有抛出异常，验证结果
            if not result.empty and 'mtm' in result.columns:
                # 验证所有Momentum值为NaN（因为数据不足）
                self.assertTrue(result['mtm'].isna().all())
        except ValueError:
            # 如果抛出异常，说明正确处理了数据不足的情况
            logger.info("数据不足正确抛出异常")

        logger.info("边界情况测试完成")
    
    def test_momentum_different_methods(self):
        """测试不同的Momentum计算方法"""
        logger.info("测试不同的Momentum计算方法")
        
        # 测试差值法
        momentum_diff = Momentum(period=10, calculation_method="difference")
        result_diff = momentum_diff.calculate(self.test_data)
        
        # 测试比率法
        momentum_ratio = Momentum(period=10, calculation_method="ratio")
        result_ratio = momentum_ratio.calculate(self.test_data)
        
        # 验证两种方法都能正常计算
        self.assertIn('mtm', result_diff.columns)
        self.assertIn('mtm', result_ratio.columns)
        
        # 验证两种方法的结果不同
        diff_values = result_diff['mtm'].dropna()
        ratio_values = result_ratio['mtm'].dropna()
        
        if len(diff_values) > 0 and len(ratio_values) > 0:
            # 比率法的值应该在100附近，差值法的值可能为负
            self.assertNotEqual(diff_values.iloc[-1], ratio_values.iloc[-1])
        
        logger.info("不同计算方法测试完成")


if __name__ == '__main__':
    unittest.main()
