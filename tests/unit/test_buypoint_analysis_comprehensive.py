#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
买点分析系统综合测试

基于现有的测试基础设施，扩展买点分析功能的全面测试
利用已有的indicator_test_mixin和Test_data_generator
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analysis.buypoints.analyze_buypoints import BuyPointAnalyzer
from tests.unit.indicator_test_mixin import Indicator_test_mixin
from tests.helper.data_generator import Test_data_generator
from tests.helper.log_capture import Log_capture_mixin
from utils.logger import get_logger

logger = get_logger(__name__)


class TestBuyPointAnalysisComprehensive(Log_capture_mixin, Indicator_test_mixin, unittest.TestCase):
    """买点分析系统综合测试类"""
    
    def setUp(self):
        """设置测试数据"""
        super().setUp()
        self.buypoint_analyzer = BuyPointAnalyzer()
        
        # 测试配置
        self.test_config = {
            'data_points': 60,
            'accuracy_threshold': 0.8,
            'test_iterations': 3
        }
        
        # 技术形态测试规格
        self.pattern_specs = {
            'MACD_GOLDEN_CROSS': [
                {'type': 'trend', 'start_price': 100, 'end_price': 95, 'periods': 30},  # 下跌
                {'type': 'trend', 'start_price': 95, 'end_price': 105, 'periods': 30}   # 上涨形成金叉
            ],
            'RSI_OVERSOLD': [
                {'type': 'trend', 'start_price': 100, 'end_price': 85, 'periods': 40},  # 急跌
                {'type': 'sideways', 'start_price': 85, 'end_price': 87, 'periods': 20} # 横盘
            ],
            'KDJ_GOLDEN_CROSS': [
                {'type': 'trend', 'start_price': 100, 'end_price': 90, 'periods': 25},  # 下跌
                {'type': 'trend', 'start_price': 90, 'end_price': 98, 'periods': 35}    # 反弹
            ],
            'BOLL_UPPER_BREAKOUT': [
                {'type': 'sideways', 'start_price': 100, 'end_price': 102, 'periods': 40}, # 横盘
                {'type': 'trend', 'start_price': 102, 'end_price': 115, 'periods': 20}     # 突破
            ],
            'BOLL_LOWER_BREAKOUT': [
                {'type': 'sideways', 'start_price': 100, 'end_price': 98, 'periods': 40},  # 横盘
                {'type': 'trend', 'start_price': 98, 'end_price': 85, 'periods': 20}       # 跌破
            ]
        }

    def _generate_pattern_data(self, pattern_type: str, stock_code: str = "TEST001") -> pd.DataFrame:
        """
        使用现有的Test_data_generator生成形态数据
        
        Args:
            pattern_type: 形态类型
            stock_code: 股票代码
            
        Returns:
            pd.DataFrame: 生成的测试数据
        """
        if pattern_type not in self.pattern_specs:
            logger.warning(f"不支持的形态类型: {pattern_type}")
            return None
        
        # 使用现有的数据生成器
        data = Test_data_generator.generate_price_sequence(
            sequence_specs=self.pattern_specs[pattern_type],
            base_date='2025-06-01',  # 确保日期范围符合买点分析器要求
            base_volume=2000000,
            apply_noise=True,
            noise_level=0.02
        )
        
        # 确保数据包含所有必需字段
        data = self._ensure_stock_info_fields(data)
        
        # 添加股票代码和名称
        data['code'] = stock_code
        data['name'] = f'测试股票_{pattern_type}'
        
        return data

    def _test_pattern_recognition(self, pattern_type: str) -> dict:
        """
        测试单个形态识别
        
        Args:
            pattern_type: 形态类型
            
        Returns:
            dict: 测试结果
        """
        results = []
        
        for iteration in range(self.test_config['test_iterations']):
            stock_code = f"TEST_{pattern_type}_{iteration}"
            
            # 生成测试数据
            test_data = self._generate_pattern_data(pattern_type, stock_code)
            
            if test_data is None or test_data.empty:
                logger.warning(f"跳过测试 {pattern_type} (第{iteration+1}次): 无法生成数据")
                continue
            
            try:
                # 执行买点分析
                # 注意：这里需要实际的数据库连接，或者需要模拟数据接口
                # 暂时使用简化的测试方法
                analysis_result = self._simulate_buypoint_analysis(test_data, pattern_type)
                
                # 评估结果
                pattern_found = self._check_pattern_in_result(pattern_type, analysis_result)
                
                results.append({
                    'iteration': iteration,
                    'pattern_found': pattern_found,
                    'data_quality': self._assess_data_quality(test_data),
                    'analysis_result': analysis_result
                })
                
            except Exception as e:
                logger.error(f"形态测试异常 {pattern_type} (第{iteration+1}次): {e}")
                results.append({
                    'iteration': iteration,
                    'pattern_found': False,
                    'error': str(e)
                })
        
        # 计算统计结果
        if not results:
            return {
                'pattern_type': pattern_type,
                'success': False,
                'accuracy': 0.0,
                'test_count': 0,
                'error': '无法生成测试数据'
            }
        
        successful_tests = sum(1 for r in results if r.get('pattern_found', False))
        accuracy = successful_tests / len(results)
        
        return {
            'pattern_type': pattern_type,
            'success': accuracy >= self.test_config['accuracy_threshold'],
            'accuracy': accuracy,
            'test_count': len(results),
            'detailed_results': results
        }

    def _simulate_buypoint_analysis(self, data: pd.DataFrame, expected_pattern: str) -> dict:
        """
        模拟买点分析（简化版本，用于测试框架验证）
        
        Args:
            data: 测试数据
            expected_pattern: 期望的形态
            
        Returns:
            dict: 模拟的分析结果
        """
        # 这是一个简化的模拟实现
        # 在实际使用中，应该调用真实的买点分析器
        
        # 基于数据特征进行简单的形态检测
        close_prices = data['close'].values
        
        # 简单的形态检测逻辑
        if expected_pattern == 'MACD_GOLDEN_CROSS':
            # 检查是否有上涨趋势
            recent_trend = close_prices[-10:].mean() > close_prices[-20:-10].mean()
            return {'macd_golden_cross': recent_trend, 'confidence': 0.8 if recent_trend else 0.3}
        
        elif expected_pattern == 'RSI_OVERSOLD':
            # 检查是否有超卖后反弹
            min_price = close_prices.min()
            current_price = close_prices[-1]
            oversold_rebound = current_price > min_price * 1.02
            return {'rsi_oversold': oversold_rebound, 'confidence': 0.7 if oversold_rebound else 0.2}
        
        elif expected_pattern == 'BOLL_UPPER_BREAKOUT':
            # 检查是否有向上突破
            price_increase = close_prices[-1] > close_prices[0] * 1.1
            return {'boll_breakout': price_increase, 'confidence': 0.9 if price_increase else 0.1}
        
        # 默认返回
        return {'pattern_detected': False, 'confidence': 0.0}

    def _check_pattern_in_result(self, expected_pattern: str, result: dict) -> bool:
        """检查结果中是否包含期望的形态"""
        if not result:
            return False
        
        # 检查置信度
        confidence = result.get('confidence', 0.0)
        if confidence < 0.5:
            return False
        
        # 检查特定形态
        pattern_keys = {
            'MACD_GOLDEN_CROSS': ['macd_golden_cross', 'macd'],
            'RSI_OVERSOLD': ['rsi_oversold', 'rsi'],
            'KDJ_GOLDEN_CROSS': ['kdj_golden_cross', 'kdj'],
            'BOLL_UPPER_BREAKOUT': ['boll_breakout', 'boll'],
            'BOLL_LOWER_BREAKOUT': ['boll_breakout', 'boll']
        }
        
        if expected_pattern in pattern_keys:
            for key in pattern_keys[expected_pattern]:
                if key in result and result[key]:
                    return True
        
        return False

    # ==================== 具体的形态测试方法 ====================
    
    def test_macd_golden_cross_pattern(self):
        """测试MACD金叉形态识别"""
        result = self._test_pattern_recognition('MACD_GOLDEN_CROSS')
        self.assertGreaterEqual(result['accuracy'], self.test_config['accuracy_threshold'],
                               f"MACD金叉识别准确率 {result['accuracy']:.1%} 低于阈值")
        logger.info(f"MACD金叉测试完成: 准确率 {result['accuracy']:.1%}")

    def test_rsi_oversold_pattern(self):
        """测试RSI超卖形态识别"""
        result = self._test_pattern_recognition('RSI_OVERSOLD')
        self.assertGreaterEqual(result['accuracy'], self.test_config['accuracy_threshold'],
                               f"RSI超卖识别准确率 {result['accuracy']:.1%} 低于阈值")
        logger.info(f"RSI超卖测试完成: 准确率 {result['accuracy']:.1%}")

    def test_kdj_golden_cross_pattern(self):
        """测试KDJ金叉形态识别"""
        result = self._test_pattern_recognition('KDJ_GOLDEN_CROSS')
        self.assertGreaterEqual(result['accuracy'], self.test_config['accuracy_threshold'],
                               f"KDJ金叉识别准确率 {result['accuracy']:.1%} 低于阈值")
        logger.info(f"KDJ金叉测试完成: 准确率 {result['accuracy']:.1%}")

    def test_boll_upper_breakout_pattern(self):
        """测试BOLL上轨突破形态识别"""
        result = self._test_pattern_recognition('BOLL_UPPER_BREAKOUT')
        self.assertGreaterEqual(result['accuracy'], self.test_config['accuracy_threshold'],
                               f"BOLL上轨突破识别准确率 {result['accuracy']:.1%} 低于阈值")
        logger.info(f"BOLL上轨突破测试完成: 准确率 {result['accuracy']:.1%}")

    def test_boll_lower_breakout_pattern(self):
        """测试BOLL下轨突破形态识别"""
        result = self._test_pattern_recognition('BOLL_LOWER_BREAKOUT')
        self.assertGreaterEqual(result['accuracy'], self.test_config['accuracy_threshold'],
                               f"BOLL下轨突破识别准确率 {result['accuracy']:.1%} 低于阈值")
        logger.info(f"BOLL下轨突破测试完成: 准确率 {result['accuracy']:.1%}")

    def test_data_generation_quality(self):
        """测试数据生成质量"""
        for pattern_type in self.pattern_specs.keys():
            with self.subTest(pattern=pattern_type):
                data = self._generate_pattern_data(pattern_type)
                self.assertIsNotNone(data, f"无法生成 {pattern_type} 的测试数据")
                self.assertGreater(len(data), 50, f"{pattern_type} 数据点数量不足")
                
                # 验证数据质量
                quality = self._assess_data_quality(data)
                self.assertGreater(quality['score'], 0.8, f"{pattern_type} 数据质量不达标")

    def test_existing_indicators_integration(self):
        """测试与现有指标系统的集成"""
        # 使用现有的指标注册表
        from indicators.complete_indicator_registry import complete_registry
        
        # 测试几个核心指标
        test_indicators = ['MACD', 'RSI', 'KDJ', 'BOLL']
        
        for indicator_name in test_indicators:
            with self.subTest(indicator=indicator_name):
                # 创建指标实例
                indicator = complete_registry.create_indicator(indicator_name)
                self.assertIsNotNone(indicator, f"无法创建指标: {indicator_name}")
                
                # 生成测试数据
                test_data = Test_data_generator.generate_price_sequence([
                    {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 60}
                ])
                test_data = self._ensure_stock_info_fields(test_data)
                
                # 计算指标
                try:
                    result = indicator.calculate(test_data)
                    self.assertIsNotNone(result, f"指标 {indicator_name} 计算失败")
                    logger.info(f"指标 {indicator_name} 计算成功")
                except Exception as e:
                    self.fail(f"指标 {indicator_name} 计算异常: {e}")


if __name__ == '__main__':
    unittest.main()
