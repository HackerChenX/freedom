"""
复杂逻辑处理器单元测试

测试复杂逻辑表达式的解析和评估功能。
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from analysis.engines.complex_logic_processor import (
    ComplexLogicProcessor,
    LogicExpressionLexer,
    LogicExpressionParser,
    TokenType,
    Token
)
from analysis.engines.shared_condition_evaluator import SharedConditionEvaluator


class TestLogicExpressionLexer(unittest.TestCase):
    """逻辑表达式词法分析器测试"""
    
    def test_tokenize_simple_expression(self):
        """测试简单表达式词法分析"""
        lexer = LogicExpressionLexer("price > 10")
        tokens = lexer.tokenize()
        
        self.assertEqual(len(tokens), 4)  # price, >, 10, EOF
        self.assertEqual(tokens[0].type, TokenType.IDENTIFIER)
        self.assertEqual(tokens[0].value, "price")
        self.assertEqual(tokens[1].type, TokenType.OPERATOR)
        self.assertEqual(tokens[1].value, ">")
        self.assertEqual(tokens[2].type, TokenType.NUMBER)
        self.assertEqual(tokens[2].value, "10")
        self.assertEqual(tokens[3].type, TokenType.EOF)
    
    def test_tokenize_complex_expression(self):
        """测试复杂表达式词法分析"""
        lexer = LogicExpressionLexer("CROSS(MA(close, 5), MA(close, 20)) AND volume > 1000000")
        tokens = lexer.tokenize()
        
        # 验证关键标记
        token_values = [token.value for token in tokens if token.type != TokenType.EOF]
        self.assertIn("CROSS", token_values)
        self.assertIn("MA", token_values)
        self.assertIn("AND", token_values)
        self.assertIn("volume", token_values)
    
    def test_tokenize_logical_operators(self):
        """测试逻辑运算符词法分析"""
        lexer = LogicExpressionLexer("a AND b OR c NOT d")
        tokens = lexer.tokenize()
        
        and_tokens = [t for t in tokens if t.type == TokenType.AND]
        or_tokens = [t for t in tokens if t.type == TokenType.OR]
        not_tokens = [t for t in tokens if t.type == TokenType.NOT]
        
        self.assertEqual(len(and_tokens), 1)
        self.assertEqual(len(or_tokens), 1)
        self.assertEqual(len(not_tokens), 1)
    
    def test_tokenize_parentheses(self):
        """测试括号词法分析"""
        lexer = LogicExpressionLexer("(a > b) AND (c < d)")
        tokens = lexer.tokenize()
        
        left_paren_tokens = [t for t in tokens if t.type == TokenType.LEFT_PAREN]
        right_paren_tokens = [t for t in tokens if t.type == TokenType.RIGHT_PAREN]
        
        self.assertEqual(len(left_paren_tokens), 2)
        self.assertEqual(len(right_paren_tokens), 2)
    
    def test_tokenize_strings(self):
        """测试字符串词法分析"""
        lexer = LogicExpressionLexer('name = "test" AND type = \'stock\'')
        tokens = lexer.tokenize()
        
        string_tokens = [t for t in tokens if t.type == TokenType.STRING]
        self.assertEqual(len(string_tokens), 2)
        self.assertEqual(string_tokens[0].value, "test")
        self.assertEqual(string_tokens[1].value, "stock")


class TestLogicExpressionParser(unittest.TestCase):
    """逻辑表达式语法分析器测试"""
    
    def test_parse_simple_comparison(self):
        """测试简单比较表达式解析"""
        lexer = LogicExpressionLexer("price > 10")
        tokens = lexer.tokenize()
        parser = LogicExpressionParser(tokens)
        
        ast = parser.parse()
        
        self.assertEqual(ast['type'], 'comparison')
        self.assertEqual(ast['operator'], '>')
        self.assertEqual(ast['left']['type'], 'identifier')
        self.assertEqual(ast['left']['value'], 'price')
        self.assertEqual(ast['right']['type'], 'number')
        self.assertEqual(ast['right']['value'], 10.0)
    
    def test_parse_logical_and(self):
        """测试AND逻辑表达式解析"""
        lexer = LogicExpressionLexer("price > 10 AND volume > 1000")
        tokens = lexer.tokenize()
        parser = LogicExpressionParser(tokens)
        
        ast = parser.parse()
        
        self.assertEqual(ast['type'], 'logical')
        self.assertEqual(ast['operator'], 'AND')
        self.assertEqual(ast['left']['type'], 'comparison')
        self.assertEqual(ast['right']['type'], 'comparison')
    
    def test_parse_logical_or(self):
        """测试OR逻辑表达式解析"""
        lexer = LogicExpressionLexer("price > 100 OR volume > 1000000")
        tokens = lexer.tokenize()
        parser = LogicExpressionParser(tokens)
        
        ast = parser.parse()
        
        self.assertEqual(ast['type'], 'logical')
        self.assertEqual(ast['operator'], 'OR')
    
    def test_parse_logical_not(self):
        """测试NOT逻辑表达式解析"""
        lexer = LogicExpressionLexer("NOT price > 10")
        tokens = lexer.tokenize()
        parser = LogicExpressionParser(tokens)
        
        ast = parser.parse()
        
        self.assertEqual(ast['type'], 'logical')
        self.assertEqual(ast['operator'], 'NOT')
        self.assertIn('operand', ast)
    
    def test_parse_parentheses(self):
        """测试括号表达式解析"""
        lexer = LogicExpressionLexer("(price > 10 AND volume > 1000) OR close < 5")
        tokens = lexer.tokenize()
        parser = LogicExpressionParser(tokens)
        
        ast = parser.parse()
        
        self.assertEqual(ast['type'], 'logical')
        self.assertEqual(ast['operator'], 'OR')
    
    def test_parse_function_call(self):
        """测试函数调用解析"""
        lexer = LogicExpressionLexer("MA(close, 20) > 10")
        tokens = lexer.tokenize()
        parser = LogicExpressionParser(tokens)
        
        ast = parser.parse()
        
        self.assertEqual(ast['type'], 'comparison')
        self.assertEqual(ast['left']['type'], 'function_call')
        self.assertEqual(ast['left']['function'], 'MA')
        self.assertEqual(len(ast['left']['args']), 2)


class TestComplexLogicProcessor(unittest.TestCase):
    """复杂逻辑处理器测试"""
    
    def setUp(self):
        """测试设置"""
        self.processor = ComplexLogicProcessor()
        
        # 模拟数据
        self.test_data = {
            'close': [10, 11, 12, 13, 14],
            'open': [9, 10, 11, 12, 13],
            'high': [11, 12, 13, 14, 15],
            'low': [9, 10, 11, 12, 13],
            'volume': [1000, 1100, 1200, 1300, 1400],
            'price': 12.5,
            'amount': 15000
        }
    
    def test_evaluate_simple_comparison(self):
        """测试简单比较表达式评估"""
        result = self.processor.evaluate_expression(
            "price > 10",
            self.test_data
        )
        
        self.assertTrue(result)
    
    def test_evaluate_logical_and(self):
        """测试AND逻辑表达式评估"""
        result = self.processor.evaluate_expression(
            "price > 10 AND amount > 10000",
            self.test_data
        )
        
        self.assertTrue(result)
        
        result = self.processor.evaluate_expression(
            "price > 20 AND amount > 10000",
            self.test_data
        )
        
        self.assertFalse(result)
    
    def test_evaluate_logical_or(self):
        """测试OR逻辑表达式评估"""
        result = self.processor.evaluate_expression(
            "price > 20 OR amount > 10000",
            self.test_data
        )
        
        self.assertTrue(result)
        
        result = self.processor.evaluate_expression(
            "price > 20 OR amount > 20000",
            self.test_data
        )
        
        self.assertFalse(result)
    
    def test_evaluate_logical_not(self):
        """测试NOT逻辑表达式评估"""
        result = self.processor.evaluate_expression(
            "NOT price > 20",
            self.test_data
        )
        
        self.assertTrue(result)
    
    def test_evaluate_parentheses(self):
        """测试括号表达式评估"""
        result = self.processor.evaluate_expression(
            "(price > 10 AND amount > 5000) OR (price < 5 AND amount > 20000)",
            self.test_data
        )
        
        self.assertTrue(result)
    
    def test_custom_functions(self):
        """测试自定义函数"""
        # 测试ABS函数
        result = self.processor.evaluate_expression(
            "ABS(-5) = 5",
            self.test_data
        )
        
        self.assertTrue(result)
        
        # 测试SQRT函数
        result = self.processor.evaluate_expression(
            "SQRT(16) = 4",
            self.test_data
        )
        
        self.assertTrue(result)
    
    def test_custom_variables(self):
        """测试自定义变量"""
        self.processor.set_variable("threshold", 15)
        
        result = self.processor.evaluate_expression(
            "amount > threshold * 1000",
            self.test_data
        )
        
        self.assertFalse(result)
        
        # 清空变量
        self.processor.clear_variables()
        self.assertIsNone(self.processor.get_variable("threshold"))
    
    def test_register_custom_function(self):
        """测试注册自定义函数"""
        def custom_double(args, data, date_idx):
            return args[0] * 2
        
        self.processor.register_function("DOUBLE", custom_double)
        
        result = self.processor.evaluate_expression(
            "DOUBLE(5) = 10",
            self.test_data
        )
        
        self.assertTrue(result)
    
    def test_cache_functionality(self):
        """测试缓存功能"""
        expression = "price > 10 AND amount > 5000"
        
        # 第一次评估
        result1 = self.processor.evaluate_expression(expression, self.test_data)
        
        # 第二次评估（应该使用缓存）
        result2 = self.processor.evaluate_expression(expression, self.test_data)
        
        self.assertEqual(result1, result2)
        
        stats = self.processor.get_stats()
        self.assertGreater(stats['cache_hits'], 0)
    
    def test_error_handling(self):
        """测试错误处理"""
        # 语法错误
        result = self.processor.evaluate_expression(
            "price > > 10",
            self.test_data
        )
        
        self.assertFalse(result)
        
        # 未知标识符
        result = self.processor.evaluate_expression(
            "unknown_field > 10",
            self.test_data
        )
        
        self.assertFalse(result)
    
    def test_complex_expression(self):
        """测试复杂表达式"""
        # 模拟复杂的买点分析条件
        expression = """
        (price > 10 AND amount > 5000) AND 
        NOT (price > 20 OR amount < 1000) AND
        ABS(price - 12) < 2
        """
        
        result = self.processor.evaluate_expression(expression, self.test_data)
        
        self.assertTrue(result)
    
    def test_statistics(self):
        """测试统计信息"""
        # 重置统计信息
        self.processor.reset_stats()
        
        # 评估几个表达式
        expressions = [
            "price > 10",
            "amount > 5000",
            "price > 10 AND amount > 5000"
        ]
        
        for expr in expressions:
            self.processor.evaluate_expression(expr, self.test_data)
        
        stats = self.processor.get_stats()
        
        self.assertEqual(stats['expressions_evaluated'], 3)
        self.assertGreaterEqual(stats['expressions_parsed'], 2)  # 由于缓存，可能有重复
        self.assertGreaterEqual(stats['avg_eval_time'], 0)
    
    def test_clear_cache_and_reset_stats(self):
        """测试清空缓存和重置统计"""
        # 评估表达式
        self.processor.evaluate_expression("price > 10", self.test_data)
        
        # 清空缓存
        self.processor.clear_cache()
        
        # 重置统计
        self.processor.reset_stats()
        
        stats = self.processor.get_stats()
        self.assertEqual(stats['expressions_evaluated'], 0)
        self.assertEqual(stats['expressions_parsed'], 0)


if __name__ == '__main__':
    unittest.main() 