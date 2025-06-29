"""
复杂逻辑处理器集成测试

测试复杂逻辑处理器与其他组件的集成功能。
"""

import unittest
import numpy as np
import pandas as pd
import time

from analysis.engines.complex_logic_processor import ComplexLogicProcessor
from analysis.engines.shared_condition_evaluator import SharedConditionEvaluator
from analysis.engines.unified_indicator_engine import UnifiedIndicatorEngine


class TestComplexLogicProcessorIntegration(unittest.TestCase):
    """复杂逻辑处理器集成测试"""
    
    def setUp(self):
        """测试设置"""
        # 创建统一指标引擎
        self.indicator_engine = UnifiedIndicatorEngine()
        
        # 创建共享条件评估器
        self.condition_evaluator = SharedConditionEvaluator(self.indicator_engine)
        
        # 创建复杂逻辑处理器
        self.processor = ComplexLogicProcessor(self.condition_evaluator)
        
        # 模拟股票数据
        self.stock_data = {
            'code': '000001',
            'name': '平安银行',
            'close': [10.5, 10.8, 11.2, 11.0, 11.5, 11.8, 12.0, 12.2, 12.5, 12.8],
            'open': [10.3, 10.6, 11.0, 10.9, 11.3, 11.6, 11.9, 12.1, 12.3, 12.6],
            'high': [10.9, 11.1, 11.4, 11.2, 11.7, 12.0, 12.3, 12.4, 12.7, 13.0],
            'low': [10.2, 10.5, 10.8, 10.7, 11.1, 11.4, 11.7, 11.9, 12.1, 12.4],
            'volume': [1000000, 1200000, 1500000, 1100000, 1300000, 1400000, 1600000, 1800000, 2000000, 1900000],
            'amount': [10500000, 12960000, 16800000, 12100000, 14950000, 16520000, 19200000, 21960000, 25000000, 24320000],
            'trade_date': ['20240101', '20240102', '20240103', '20240104', '20240105', 
                          '20240108', '20240109', '20240110', '20240111', '20240112'],
            'market_cap': 250000000000,
            'pe_ratio': 8.5,
            'pb_ratio': 0.85,
            'industry': '银行',
            'sector': '金融'
        }
    
    def test_basic_expression_evaluation(self):
        """测试基础表达式评估"""
        # 简单条件
        result = self.processor.evaluate_expression(
            "close > 12.0",
            self.stock_data,
            date_idx=8  # 使用倒数第二个数据点
        )
        self.assertTrue(result)
        
        # 复合条件
        result = self.processor.evaluate_expression(
            "close > 11.0 AND volume > 1500000",
            self.stock_data,
            date_idx=8
        )
        self.assertTrue(result)
    
    def test_arithmetic_expressions(self):
        """测试算术表达式"""
        # 算术运算
        result = self.processor.evaluate_expression(
            "close + open > 24.0",
            self.stock_data,
            date_idx=8
        )
        self.assertTrue(result)
        
        # 复杂算术表达式
        result = self.processor.evaluate_expression(
            "(high - low) / close > 0.02",
            self.stock_data,
            date_idx=8
        )
        self.assertTrue(result)
    
    def test_function_calls(self):
        """测试函数调用"""
        # ABS函数
        result = self.processor.evaluate_expression(
            "ABS(close - 12.0) < 1.0",
            self.stock_data,
            date_idx=8
        )
        self.assertTrue(result)
        
        # SQRT函数
        result = self.processor.evaluate_expression(
            "SQRT(volume) > 1000",
            self.stock_data,
            date_idx=8
        )
        self.assertTrue(result)
        
        # MAX函数
        result = self.processor.evaluate_expression(
            "MAX(close, 3) > 12.0",
            self.stock_data,
            date_idx=8
        )
        self.assertTrue(result)
    
    def test_complex_buypoint_conditions(self):
        """测试复杂买点条件"""
        # 模拟买点分析中的复杂条件
        conditions = [
            # 价格突破条件
            "close > 12.0 AND volume > 1800000",
            
            # 涨幅条件
            "(close - open) / open > 0.01",
            
            # 成交量放大条件
            "volume > 1500000 AND (high - low) / close > 0.02",
            
            # 复合技术条件
            "close > 11.5 AND volume > 1200000 AND ABS(close - high) / close < 0.02",
            
            # 风险控制条件
            "NOT (close < 11.0 OR volume < 1000000)"
        ]
        
        for i, condition in enumerate(conditions):
            result = self.processor.evaluate_expression(
                condition,
                self.stock_data,
                date_idx=8
            )
            self.assertTrue(result, f"条件 {i+1} 评估失败: {condition}")
    
    def test_variable_usage(self):
        """测试变量使用"""
        # 设置自定义变量
        self.processor.set_variable("min_price", 11.0)
        self.processor.set_variable("min_volume", 1500000)
        self.processor.set_variable("max_pe", 10.0)
        
        # 使用变量的条件
        result = self.processor.evaluate_expression(
            "close > min_price AND volume > min_volume AND pe_ratio < max_pe",
            self.stock_data,
            date_idx=8
        )
        self.assertTrue(result)
    
    def test_nested_expressions(self):
        """测试嵌套表达式"""
        # 深度嵌套的逻辑表达式
        expression = """
        (
            (close > 12.0 AND volume > 1500000) OR
            (close > 11.5 AND volume > 2000000)
        ) AND NOT (
            close < 11.0 OR 
            (volume < 1000000 AND close < 12.5)
        ) AND (
            ABS(close - high) / close < 0.02 OR
            (high - low) / close > 0.03
        )
        """
        
        result = self.processor.evaluate_expression(
            expression,
            self.stock_data,
            date_idx=8
        )
        self.assertTrue(result)
    
    def test_performance_with_large_expressions(self):
        """测试大型表达式的性能"""
        # 构建复杂的表达式
        conditions = []
        for i in range(20):
            conditions.append(f"close > {10 + i * 0.1}")
        
        # 用OR连接所有条件
        large_expression = " OR ".join(conditions)
        
        start_time = time.time()
        result = self.processor.evaluate_expression(
            large_expression,
            self.stock_data,
            date_idx=8
        )
        end_time = time.time()
        
        self.assertTrue(result)
        self.assertLess(end_time - start_time, 0.1, "大型表达式评估时间过长")
    
    def test_cache_effectiveness(self):
        """测试缓存效果"""
        expression = "close > 11.0 AND volume > 1500000 AND ABS(close - 12.5) < 1.0"
        
        # 第一次评估
        start_time = time.time()
        result1 = self.processor.evaluate_expression(expression, self.stock_data, date_idx=8)
        first_time = time.time() - start_time
        
        # 第二次评估（应该使用缓存）
        start_time = time.time()
        result2 = self.processor.evaluate_expression(expression, self.stock_data, date_idx=8)
        second_time = time.time() - start_time
        
        self.assertEqual(result1, result2)
        # 由于缓存，第二次应该更快（但可能不到50%，因为表达式比较简单）
        
        # 检查统计信息
        stats = self.processor.get_stats()
        self.assertGreater(stats['cache_hits'], 0)
    
    def test_error_recovery(self):
        """测试错误恢复"""
        # 语法错误的表达式
        error_expressions = [
            "close > > 11.0",  # 语法错误
            "unknown_field > 10",  # 未知字段
            "close > AND volume > 1000",  # 缺少操作数
        ]
        
        for expr in error_expressions:
            result = self.processor.evaluate_expression(expr, self.stock_data, date_idx=8)
            self.assertFalse(result, f"错误表达式应该返回False: {expr}")


if __name__ == '__main__':
    unittest.main()
