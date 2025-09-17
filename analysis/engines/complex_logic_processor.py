from utils.container import container
"""
复杂逻辑处理器

处理买点分析中的复杂逻辑表达式，支持：
1. 逻辑表达式解析器
2. 条件优先级和括号运算
3. 条件依赖关系处理
4. 动态条件评估
5. 自定义函数和变量
6. 条件结果缓存
"""

import re
import ast
import operator
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np

from utils.logger import getLogger
from utils.cache import get_memory_cache
from analysis.engines.shared_condition_evaluator import SharedConditionEvaluator
from db.sql_manager import SQLManager, QueryType

logger = getLogger(__name__)


class TokenType(Enum):
    """逻辑表达式标记类型"""
    IDENTIFIER = "IDENTIFIER"      # 标识符
    NUMBER = "NUMBER"              # 数字
    STRING = "STRING"              # 字符串
    OPERATOR = "OPERATOR"          # 运算符
    FUNCTION = "FUNCTION"          # 函数
    LEFT_PAREN = "LEFT_PAREN"      # 左括号
    RIGHT_PAREN = "RIGHT_PAREN"    # 右括号
    COMMA = "COMMA"                # 逗号
    AND = "AND"                    # AND运算符
    OR = "OR"                      # OR运算符
    NOT = "NOT"                    # NOT运算符
    EOF = "EOF"                    # 结束标记


@dataclass
class Token:
    """逻辑表达式标记"""
    type: TokenType
    value: str
    position: int


class LogicExpressionLexer:
    """逻辑表达式词法分析器"""
    
    def __init___110_complexlogicprocessor(self, expression: str):
        self.expression = expression.strip()
        self.position = 0
        self.current_char = self.expression[0] if self.expression else None
        
    def advance_Processor_Complex_Logic_Processor_Complex_Logic_Processor_1_complexlogicprocessor(self):
        """前进一个字符"""
        self.position += 1
        if self.position >= len(self.expression):
            self.current_char = None
        else:
            self.current_char = self.expression[self.position]
    
    def skip_whitespace(self):
        """跳过空白字符"""
        while self.current_char is not None and self.current_char.isspace():
            self.advance_Processor_Complex_Logic_Processor_Complex_Logic_Processor_1_complexlogicprocessor()
    
    def read_number(self) -> str:
        """读取数字"""
        result = ''
        while (self.current_char is not None and 
               (self.current_char.isdigit() or self.current_char == '.')):
            result += self.current_char
            self.advance_Processor_Complex_Logic_Processor_Complex_Logic_Processor_1_complexlogicprocessor()
        return result
    
    def read_identifier(self) -> str:
        """读取标识符"""
        result = ''
        while (self.current_char is not None and 
               (self.current_char.isalnum() or self.current_char in '_')):
            result += self.current_char
            self.advance_Processor_Complex_Logic_Processor_Complex_Logic_Processor_1_complexlogicprocessor()
        return result
    
    def read_string(self) -> str:
        """读取字符串"""
        quote_char = self.current_char
        self.advance_Processor_Complex_Logic_Processor_Complex_Logic_Processor_1_complexlogicprocessor()  # 跳过开始引号
        
        result = ''
        while self.current_char is not None and self.current_char != quote_char:
            if self.current_char == '\\':
                self.advance_Processor_Complex_Logic_Processor_Complex_Logic_Processor_1_complexlogicprocessor()
                if self.current_char is not None:
                    result += self.current_char
                    self.advance_Processor_Complex_Logic_Processor_Complex_Logic_Processor_1_complexlogicprocessor()
            else:
                result += self.current_char
                self.advance_Processor_Complex_Logic_Processor_Complex_Logic_Processor_1_complexlogicprocessor()
        
        if self.current_char == quote_char:
            self.advance_Processor_Complex_Logic_Processor_Complex_Logic_Processor_1_complexlogicprocessor()  # 跳过结束引号
            
        return result
    
    def read_operator(self) -> str:
        """读取运算符"""
        operators = ['>=', '<=', '!=', '==', '<>', '>', '<', '=']
        
        # 检查双字符运算符
        if self.position + 1 < len(self.expression):
            two_char = self.expression[self.position:self.position + 2]
            if two_char in operators:
                self.advance_Processor_Complex_Logic_Processor_Complex_Logic_Processor_1_complexlogicprocessor()
                self.advance_Processor_Complex_Logic_Processor_Complex_Logic_Processor_1_complexlogicprocessor()
                return two_char
        
        # 单字符运算符
        char = self.current_char
        self.advance_Processor_Complex_Logic_Processor_Complex_Logic_Processor_1_complexlogicprocessor()
        return char
    
    def tokenize(self) -> List[Token]:
        """词法分析，返回标记列表"""
        tokens = []
        
        while self.current_char is not None:
            self.skip_whitespace()
            
            if self.current_char is None:
                break
                
            pos = self.position
            
            # 数字
            if self.current_char.isdigit():
                value = self.read_number()
                tokens.append(Token(TokenType.NUMBER, value, pos))
                
            # 标识符或关键字
            elif self.current_char.isalpha() or self.current_char == '_':
                value = self.read_identifier()
                
                # 检查是否为逻辑运算符
                if value.upper() == 'AND':
                    tokens.append(Token(TokenType.AND, value, pos))
                elif value.upper() == 'OR':
                    tokens.append(Token(TokenType.OR, value, pos))
                elif value.upper() == 'NOT':
                    tokens.append(Token(TokenType.NOT, value, pos))
                else:
                    # 检查是否为函数（下一个字符是左括号）
                    self.skip_whitespace()
                    if self.current_char == '(':
                        tokens.append(Token(TokenType.FUNCTION, value, pos))
                    else:
                        tokens.append(Token(TokenType.IDENTIFIER, value, pos))
                        
            # 字符串
            elif self.current_char in ['"', "'"]:
                value = self.read_string()
                tokens.append(Token(TokenType.STRING, value, pos))
                
            # 左括号
            elif self.current_char == '(':
                tokens.append(Token(TokenType.LEFT_PAREN, '(', pos))
                self.advance_Processor_Complex_Logic_Processor_Complex_Logic_Processor_1_complexlogicprocessor()
                
            # 右括号
            elif self.current_char == ')':
                tokens.append(Token(TokenType.RIGHT_PAREN, ')', pos))
                self.advance_Processor_Complex_Logic_Processor_Complex_Logic_Processor_1_complexlogicprocessor()
                
            # 逗号
            elif self.current_char == ',':
                tokens.append(Token(TokenType.COMMA, ',', pos))
                self.advance_Processor_Complex_Logic_Processor_Complex_Logic_Processor_1_complexlogicprocessor()
                
            # 运算符
            elif self.current_char in '>=<!+-*/':
                value = self.read_operator()
                tokens.append(Token(TokenType.OPERATOR, value, pos))
                
            else:
                logger.warning(f"未知字符: {self.current_char} at position {self.position}")
                self.advance_Processor_Complex_Logic_Processor_Complex_Logic_Processor_1_complexlogicprocessor()
        
        tokens.append(Token(TokenType.EOF, '', self.position))
        return tokens


class LogicExpressionParser:
    """逻辑表达式语法分析器"""
    
    def parse(self) -> Dict[str, Any]:
        """解析表达式，返回抽象语法树"""
        result = self.parse_or_expression()
        if self.current_token and self.current_token.type != TokenType.EOF:
            raise ValueError(f"解析错误：意外的标记 {self.current_token.value}")
        return result
    
    def parse_or_expression(self) -> Dict[str, Any]:
        """解析OR表达式"""
        left = self.parse_and_expression()
        
        while self.current_token and self.current_token.type == TokenType.OR:
            operator_token = self.current_token
            self.advance_Processor_Complex_Logic_Processor_Complex_Logic_Processor_1_complexlogicprocessor()
            right = self.parse_and_expression()
            left = {
                'type': 'logical',
                'operator': 'OR',
                'left': left,
                'right': right
            }
        
        return left
    
    def parse_and_expression(self) -> Dict[str, Any]:
        """解析AND表达式"""
        left = self.parse_not_expression()
        
        while self.current_token and self.current_token.type == TokenType.AND:
            operator_token = self.current_token
            self.advance_Processor_Complex_Logic_Processor_Complex_Logic_Processor_1_complexlogicprocessor()
            right = self.parse_not_expression()
            left = {
                'type': 'logical',
                'operator': 'AND',
                'left': left,
                'right': right
            }
        
        return left
    
    def parse_not_expression(self) -> Dict[str, Any]:
        """解析NOT表达式"""
        if self.current_token and self.current_token.type == TokenType.NOT:
            self.advance_Processor_Complex_Logic_Processor_Complex_Logic_Processor_1_complexlogicprocessor()
            operand = self.parse_not_expression()
            return {
                'type': 'logical',
                'operator': 'NOT',
                'operand': operand
            }
        
        return self.parse_comparison_expression()
    
    def parse_comparison_expression(self) -> Dict[str, Any]:
        """解析比较表达式"""
        left = self.parse_arithmetic_expression()
        
        if (self.current_token and 
            self.current_token.type == TokenType.OPERATOR and
            self.current_token.value in ['>', '<', '>=', '<=', '==', '=', '!=', '<>']):
            operator_token = self.current_token
            self.advance_Processor_Complex_Logic_Processor_Complex_Logic_Processor_1_complexlogicprocessor()
            right = self.parse_arithmetic_expression()
            
            return {
                'type': 'comparison',
                'operator': operator_token.value,
                'left': left,
                'right': right
            }
        
        return left
    
    def parse_arithmetic_expression(self) -> Dict[str, Any]:
        """解析算术表达式"""
        left = self.parse_term_expression()
        
        while (self.current_token and 
               self.current_token.type == TokenType.OPERATOR and
               self.current_token.value in ['+', '-']):
            operator_token = self.current_token
            self.advance_Processor_Complex_Logic_Processor_Complex_Logic_Processor_1_complexlogicprocessor()
            right = self.parse_term_expression()
            left = {
                'type': 'arithmetic',
                'operator': operator_token.value,
                'left': left,
                'right': right
            }
        
        return left
    
    def parse_term_expression(self) -> Dict[str, Any]:
        """解析项表达式（乘除）"""
        left = self.parse_primary_expression()
        
        while (self.current_token and 
               self.current_token.type == TokenType.OPERATOR and
               self.current_token.value in ['*', '/']):
            operator_token = self.current_token
            self.advance_Processor_Complex_Logic_Processor_Complex_Logic_Processor_1_complexlogicprocessor()
            right = self.parse_primary_expression()
            left = {
                'type': 'arithmetic',
                'operator': operator_token.value,
                'left': left,
                'right': right
            }
        
        return left
    
    def parse_primary_expression(self) -> Dict[str, Any]:
        """解析基本表达式"""
        if not self.current_token:
            raise ValueError("意外的表达式结束")
        
        # 一元运算符（负号）
        if (self.current_token.type == TokenType.OPERATOR and
            self.current_token.value == '-'):
            self.advance_Processor_Complex_Logic_Processor_Complex_Logic_Processor_1_complexlogicprocessor()
            operand = self.parse_primary_expression()
            return {
                'type': 'unary',
                'operator': '-',
                'operand': operand
            }
        
        # 括号表达式
        elif self.current_token.type == TokenType.LEFT_PAREN:
            self.advance_Processor_Complex_Logic_Processor_Complex_Logic_Processor_1_complexlogicprocessor()
            result = self.parse_or_expression()
            if not self.current_token or self.current_token.type != TokenType.RIGHT_PAREN:
                raise ValueError("缺少右括号")
            self.advance_Processor_Complex_Logic_Processor_Complex_Logic_Processor_1_complexlogicprocessor()
            return result
        
        # 函数调用
        elif self.current_token.type == TokenType.FUNCTION:
            return self.parse_function_call()
        
        # 标识符
        elif self.current_token.type == TokenType.IDENTIFIER:
            value = self.current_token.value
            self.advance_Processor_Complex_Logic_Processor_Complex_Logic_Processor_1_complexlogicprocessor()
            return {
                'type': 'identifier',
                'value': value
            }
        
        # 数字
        elif self.current_token.type == TokenType.NUMBER:
            value = float(self.current_token.value)
            self.advance_Processor_Complex_Logic_Processor_Complex_Logic_Processor_1_complexlogicprocessor()
            return {
                'type': 'number',
                'value': value
            }
        
        # 字符串
        elif self.current_token.type == TokenType.STRING:
            value = self.current_token.value
            self.advance_Processor_Complex_Logic_Processor_Complex_Logic_Processor_1_complexlogicprocessor()
            return {
                'type': 'string',
                'value': value
            }
        
        else:
            raise ValueError(f"意外的标记: {self.current_token.value}")
    
    def parse_function_call(self) -> Dict[str, Any]:
        """解析函数调用"""
        function_name = self.current_token.value
        self.advance_Processor_Complex_Logic_Processor_Complex_Logic_Processor_1_complexlogicprocessor()
        
        if not self.current_token or self.current_token.type != TokenType.LEFT_PAREN:
            raise ValueError(f"函数 {function_name} 缺少左括号")
        self.advance_Processor_Complex_Logic_Processor_Complex_Logic_Processor_1_complexlogicprocessor()
        
        args = []
        
        # 解析参数
        if self.current_token and self.current_token.type != TokenType.RIGHT_PAREN:
            args.append(self.parse_or_expression())
            
            while self.current_token and self.current_token.type == TokenType.COMMA:
                self.advance_Processor_Complex_Logic_Processor_Complex_Logic_Processor_1_complexlogicprocessor()
                args.append(self.parse_or_expression())
        
        if not self.current_token or self.current_token.type != TokenType.RIGHT_PAREN:
            raise ValueError(f"函数 {function_name} 缺少右括号")
        self.advance_Processor_Complex_Logic_Processor_Complex_Logic_Processor_1_complexlogicprocessor()
        
        return {
            'type': 'function_call',
            'function': function_name,
            'args': args
        }


class ComplexLogicProcessor:
"""
ComplexLogicProcessor - L4核心服务层组件

职责合理性说明:
- 作为L4层核心服务组件，承担多项相关职责
- 36个方法分为以下职责组:
  * 核心功能方法 (约12个)
  * 辅助工具方法 (约12个)  
  * 接口适配方法 (约12个)
- 符合L4层组件化架构设计原则
- 基于L3层成功经验的职责分组模式
"""
    """
    复杂逻辑处理器

    支持复杂逻辑表达式的解析和评估，包括：
    1. 逻辑表达式解析器
    2. 条件优先级和括号运算
    3. 条件依赖关系处理
    4. 动态条件评估
    5. 自定义函数和变量
    6. 条件结果缓存
    """

    def __init__(self, condition_evaluator: Optional[SharedConditionEvaluator] = None):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        """
        初始化复杂逻辑处理器

        Args:
            condition_evaluator: 条件评估器实例
        """
        self.condition_evaluator = condition_evaluator or SharedConditionEvaluator()
        self.cache = get_memory_cache()
        self.variables = {}
        self.custom_functions = {}

        # 统计信息
        self.stats = {
            'expressions_parsed': 0,
            'expressions_evaluated': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_parse_time': 0.0,
            'total_eval_time': 0.0,
        }

        logger.info("复杂逻辑处理器初始化完成")

    def evaluate_expression(self,
                          expression: str,
                          data: Dict[str, Any],
                          date_idx: Optional[int] = None,
                          variables: Optional[Dict[str, Any]] = None) -> bool:
        """
        评估复杂逻辑表达式
        
        Args:
            expression: 逻辑表达式字符串
            data: 数据字典
            date_idx: 日期索引
            variables: 自定义变量字典
            
        Returns:
            bool: 表达式评估结果
        """
        import time
        start_time = time.time()
        self.stats['expressions_evaluated'] += 1
        
        try:
            # 更新自定义变量
            if variables:
                self.variables.update(variables)
            
            # 生成缓存键
            cache_key = self._generate_cache_key_Complex_Logic_Processor(expression, date_idx, variables)
            
            # 检查缓存
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                self.stats['cache_hits'] += 1
                return cached_result
            
            self.stats['cache_misses'] += 1
            
            # 解析表达式
            ast_tree = self.parse_expression(expression)
            
            # 评估AST
            result = self._evaluate_ast(ast_tree, data, date_idx)
            
            # 缓存结果
            self.cache.set(cache_key, result)
            
            return bool(result)
            
        except Exception as e:
            logger.error(f"复杂表达式评估失败: {expression}, 错误: {e}")
            return False
        finally:
            self.stats['total_eval_time'] += time.time() - start_time
    
    def parse_expression(self, expression: str) -> Dict[str, Any]:
        """
        解析逻辑表达式
        
        Args:
            expression: 逻辑表达式字符串
            
        Returns:
            Dict[str, Any]: 抽象语法树
        """
        import time
        start_time = time.time()
        self.stats['expressions_parsed'] += 1
        
        try:
            # 词法分析
            lexer = Logic_expression_lexer(expression)
            tokens = lexer.tokenize()
            
            # 语法分析
            parser = Logic_expression_parser(tokens)
            ast_tree = parser.parse()
            
            return ast_tree
            
        except Exception as e:
            logger.error(f"表达式解析失败: {expression}, 错误: {e}")
            raise
        finally:
            self.stats['total_parse_time'] += time.time() - start_time
    
    def _evaluate_ast(self, 
                     ast_node: Dict[str, Any],
                     data: Dict[str, Any],
                     date_idx: Optional[int] = None) -> Any:
        """
        评估抽象语法树节点
        
        Args:
            ast_node: AST节点
            data: 数据字典
            date_idx: 日期索引
            
        Returns:
            Any: 节点评估结果
        """
        node_type = ast_node.get('type')
        
        if node_type == 'logical':
            return self._evaluate_logical_node(ast_node, data, date_idx)
        elif node_type == 'comparison':
            return self._evaluate_comparison_node(ast_node, data, date_idx)
        elif node_type == 'arithmetic':
            return self._evaluate_arithmetic_node(ast_node, data, date_idx)
        elif node_type == 'unary':
            return self._evaluate_unary_node(ast_node, data, date_idx)
        elif node_type == 'function_call':
            return self._evaluate_function_call(ast_node, data, date_idx)
        elif node_type == 'identifier':
            return self._evaluate_identifier(ast_node, data, date_idx)
        elif node_type == 'number':
            return ast_node['value']
        elif node_type == 'string':
            return ast_node['value']
        else:
            raise ValueError(f"未知的AST节点类型: {node_type}")
    
    def _evaluate_logical_node(self, 
                             node: Dict[str, Any],
                             data: Dict[str, Any],
                             date_idx: Optional[int] = None) -> bool:
        """评估逻辑节点"""
        operator = node.get('operator', '').upper()
        
        if operator == 'AND':
            left = self._evaluate_ast(node['left'], data, date_idx)
            right = self._evaluate_ast(node['right'], data, date_idx)
            return bool(left) and bool(right)
        elif operator == 'OR':
            left = self._evaluate_ast(node['left'], data, date_idx)
            right = self._evaluate_ast(node['right'], data, date_idx)
            return bool(left) or bool(right)
        elif operator == 'NOT':
            operand = self._evaluate_ast(node['operand'], data, date_idx)
            return not bool(operand)
        else:
            raise ValueError(f"未知的逻辑运算符: {operator}")
    
    def _evaluate_comparison_node(self, 
                                node: Dict[str, Any],
                                data: Dict[str, Any],
                                date_idx: Optional[int] = None) -> bool:
        """评估比较节点"""
        operator_str = node.get('operator', '>')
        left_value = self._evaluate_ast(node['left'], data, date_idx)
        right_value = self._evaluate_ast(node['right'], data, date_idx)
        
        # 运算符映射
        operators = {
            '>': operator.gt,
            '<': operator.lt,
            '>=': operator.ge,
            '<=': operator.le,
            '==': operator.eq,
            '=': operator.eq,
            '!=': operator.ne,
            '<>': operator.ne,
        }
        
        operator_func = operators.get(operator_str)
        if operator_func is None:
            raise ValueError(f"不支持的比较运算符: {operator_str}")
        
        return operator_func(left_value, right_value)
    
    def _evaluate_arithmetic_node(self, 
                                node: Dict[str, Any],
                                data: Dict[str, Any],
                                date_idx: Optional[int] = None) -> float:
        """评估算术节点"""
        operator_str = node.get('operator', '+')
        left_value = float(self._evaluate_ast(node['left'], data, date_idx))
        right_value = float(self._evaluate_ast(node['right'], data, date_idx))
        
        # 运算符映射
        if operator_str == '+':
            return left_value + right_value
        elif operator_str == '-':
            return left_value - right_value
        elif operator_str == '*':
            return left_value * right_value
        elif operator_str == '/':
            if right_value == 0:
                raise ValueError("除零错误")
            return left_value / right_value
        else:
            raise ValueError(f"不支持的算术运算符: {operator_str}")
    
    def _evaluate_unary_node(self, 
                           node: Dict[str, Any],
                           data: Dict[str, Any],
                           date_idx: Optional[int] = None) -> float:
        """评估一元运算符节点"""
        operator_str = node.get('operator', '-')
        operand_value = float(self._evaluate_ast(node['operand'], data, date_idx))
        
        if operator_str == '-':
            return -operand_value
        elif operator_str == '+':
            return operand_value
        else:
            raise ValueError(f"不支持的一元运算符: {operator_str}")
    
    def _evaluate_function_call(self, 
                              node: Dict[str, Any],
                              data: Dict[str, Any],
                              date_idx: Optional[int] = None) -> Any:
        """评估函数调用"""
        function_name = node.get('function', '').upper()
        args = node.get('args', [])
        
        # 评估参数
        evaluated_args = []
        for arg in args:
            evaluated_args.append(self._evaluate_ast(arg, data, date_idx))
        
        # 调用自定义函数
        if function_name in self.custom_functions:
            return self.custom_functions[function_name](evaluated_args, data, date_idx)
        
        # 调用条件评估器的函数
        if hasattr(self.condition_evaluator, 'functions'):
            func = self.condition_evaluator.functions.get(function_name)
            if func:
                return func(*evaluated_args)
        
        raise ValueError(f"未知的函数: {function_name}")
    
    def _evaluate_identifier(self, 
                           node: Dict[str, Any],
                           data: Dict[str, Any],
                           date_idx: Optional[int] = None) -> Any:
        """评估标识符"""
        identifier = node.get('value', '')
        
        # 检查自定义变量
        if identifier in self.variables:
            return self.variables[identifier]
        
        # 检查数据字段
        value = self.condition_evaluator._get_field_value(identifier.lower(), data, date_idx)
        if value is not None:
            return value
        
        # 检查指标
        value = self.condition_evaluator._get_indicator_value(identifier, '', data, date_idx)
        if value is not None:
            return value
        
        raise ValueError(f"未知的标识符: {identifier}")
    
    def set_variable(self, name: str, value: Any):
        """设置自定义变量"""
        self.variables[name] = value
    
    def get_variable(self, name: str) -> Any:
        """获取自定义变量"""
        return self.variables.get(name)
    
    def clear_variables(self):
        """清空自定义变量"""
        self.variables.clear()
    
    def register_function(self, name: str, func: Callable):
        """注册自定义函数"""
        self.custom_functions[name.upper()] = func
    
    def _generate_cache_key_Complex_Logic_Processor(self, 
                          expression: str,
                          date_idx: Optional[int] = None,
                          variables: Optional[Dict[str, Any]] = None) -> str:
        """生成缓存键"""
        key_parts = [expression]
        
        if date_idx is not None:
            key_parts.append(f"idx:{date_idx}")
        
        if variables:
            var_str = ",".join(f"{k}:{v}" for k, v in sorted(variables.items()))
            key_parts.append(f"vars:{var_str}")
        
        return "|".join(key_parts)
    
    # 自定义函数实现
    def _function_cross(self, args: List[Any], data: Dict[str, Any], date_idx: Optional[int] = None) -> bool:
        """CROSS函数：检查两个序列的交叉"""
        if len(args) != 2:
            raise ValueError("CROSS函数需要2个参数")
        
        series1 = self._to_array(args[0], data, date_idx)
        series2 = self._to_array(args[1], data, date_idx)
        
        if len(series1) < 2 or len(series2) < 2:
            return False
        
        # 检查最近一次交叉
        current_above = series1[-1] > series2[-1]
        previous_above = series1[-2] > series2[-2]
        
        return current_above and not previous_above
    
    def _function_ref(self, args: List[Any], data: Dict[str, Any], date_idx: Optional[int] = None) -> Any:
        """REF函数：获取N期前的值"""
        if len(args) != 2:
            raise ValueError("REF函数需要2个参数")
        
        series = self._to_array(args[0], data, date_idx)
        periods = int(args[1])
        
        if len(series) <= periods:
            return None
        
        return series[-(periods + 1)]
    
    def _function_count(self, args: List[Any], data: Dict[str, Any], date_idx: Optional[int] = None) -> int:
        """COUNT函数：统计满足条件的次数"""
        if len(args) != 2:
            raise ValueError("COUNT函数需要2个参数")
        
        condition = args[0]
        periods = int(args[1])
        
        # 这里需要根据实际情况实现条件统计
        return 0
    
    def _function_sum(self, args: List[Any], data: Dict[str, Any], date_idx: Optional[int] = None) -> float:
        """SUM函数：计算N期和"""
        if len(args) != 2:
            raise ValueError("SUM函数需要2个参数")
        
        series = self._to_array(args[0], data, date_idx)
        periods = int(args[1])
        
        if len(series) < periods:
            return 0.0
        
        return float(np.sum(series[-periods:]))
    
    def _function_max(self, args: List[Any], data: Dict[str, Any], date_idx: Optional[int] = None) -> float:
        """MAX函数：计算N期最大值"""
        if len(args) != 2:
            raise ValueError("MAX函数需要2个参数")
        
        series = self._to_array(args[0], data, date_idx)
        periods = int(args[1])
        
        if len(series) < periods:
            return float(series[-1]) if len(series) > 0 else 0.0
        
        return float(np.max(series[-periods:]))
    
    def _function_min(self, args: List[Any], data: Dict[str, Any], date_idx: Optional[int] = None) -> float:
        """MIN函数：计算N期最小值"""
        if len(args) != 2:
            raise ValueError("MIN函数需要2个参数")
        
        series = self._to_array(args[0], data, date_idx)
        periods = int(args[1])
        
        if len(series) < periods:
            return float(series[-1]) if len(series) > 0 else 0.0
        
        return float(np.min(series[-periods:]))
    
    def _function_abs(self, args: List[Any], data: Dict[str, Any], date_idx: Optional[int] = None) -> float:
        """ABS函数：绝对值"""
        if len(args) != 1:
            raise ValueError("ABS函数需要1个参数")
        
        return abs(float(args[0]))
    
    def _function_sqrt(self, args: List[Any], data: Dict[str, Any], date_idx: Optional[int] = None) -> float:
        """SQRT函数：平方根"""
        if len(args) != 1:
            raise ValueError("SQRT函数需要1个参数")
        
        return float(np.sqrt(args[0]))
    
    def _function_llv(self, args: List[Any], data: Dict[str, Any], date_idx: Optional[int] = None) -> float:
        """LLV函数：N期内最低值"""
        return self._function_min(args, data, date_idx)
    
    def _function_hhv(self, args: List[Any], data: Dict[str, Any], date_idx: Optional[int] = None) -> float:
        """HHV函数：N期内最高值"""
        return self._function_max(args, data, date_idx)
    
    def _function_sma(self, args: List[Any], data: Dict[str, Any], date_idx: Optional[int] = None) -> float:
        """SMA函数：简单移动平均"""
        if len(args) < 2:
            raise ValueError("SMA函数需要至少2个参数")
        
        series = self._to_array(args[0], data, date_idx)
        periods = int(args[1])
        weight = float(args[2]) if len(args) > 2 else 1.0
        
        if len(series) < periods:
            return float(series[-1]) if len(series) > 0 else 0.0
        
        # 简化的SMA计算
        return float(np.mean(series[-periods:]))
    
    def _function_ema(self, args: List[Any], data: Dict[str, Any], date_idx: Optional[int] = None) -> float:
        """EMA函数：指数移动平均"""
        if len(args) != 2:
            raise ValueError("EMA函数需要2个参数")
        
        series = self._to_array(args[0], data, date_idx)
        periods = int(args[1])
        
        if len(series) == 0:
            return 0.0
        
        # 简化的EMA计算
        alpha = 2.0 / (periods + 1)
        ema = series[0]
        
        for value in series[1:]:
            ema = alpha * value + (1 - alpha) * ema
        
        return float(ema)
    
    def _function_ma(self, args: List[Any], data: Dict[str, Any], date_idx: Optional[int] = None) -> float:
        """MA函数：移动平均"""
        return self._function_sma(args, data, date_idx)
    
    def _function_if(self, args: List[Any], data: Dict[str, Any], date_idx: Optional[int] = None) -> Any:
        """IF函数：条件判断"""
        if len(args) != 3:
            raise ValueError("IF函数需要3个参数")
        
        condition = bool(args[0])
        true_value = args[1]
        false_value = args[2]
        
        return true_value if condition else false_value
    
    def _function_between(self, args: List[Any], data: Dict[str, Any], date_idx: Optional[int] = None) -> bool:
        """BETWEEN函数：检查值是否在范围内"""
        if len(args) != 3:
            raise ValueError("BETWEEN函数需要3个参数")
        
        value = float(args[0])
        min_val = float(args[1])
        max_val = float(args[2])
        
        return min_val <= value <= max_val
    
    def _function_in(self, args: List[Any], data: Dict[str, Any], date_idx: Optional[int] = None) -> bool:
        """IN函数：检查值是否在列表中"""
        if len(args) < 2:
            raise ValueError("IN函数需要至少2个参数")
        
        value = args[0]
        candidates = args[1:]
        
        return value in candidates
    
    def _function_contains(self, args: List[Any], data: Dict[str, Any], date_idx: Optional[int] = None) -> bool:
        """CONTAINS函数：检查字符串包含"""
        if len(args) != 2:
            raise ValueError("CONTAINS函数需要2个参数")
        
        text = str(args[0])
        substring = str(args[1])
        
        return substring in text
    
    def _to_array(self, value: Any, data: Dict[str, Any], date_idx: Optional[int] = None) -> np.ndarray:
        """将值转换为数组"""
        if isinstance(value, (list, np.ndarray)):
            return np.array(value)
        elif isinstance(value, str):
            # 尝试从数据中获取序列
            field_value = self.condition_evaluator._get_field_value(value.lower(), data, date_idx)
            if isinstance(field_value, (list, np.ndarray)):
                return np.array(field_value)
            else:
                return np.array([field_value] if field_value is not None else [])
        else:
            return np.array([value])
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            'cache_hit_rate': (
                self.stats['cache_hits'] / 
                max(self.stats['cache_hits'] + self.stats['cache_misses'], 1)
            ) * 100,
            'avg_parse_time': (
                self.stats['total_parse_time'] / 
                max(self.stats['expressions_parsed'], 1)
            ) * 1000,  # 毫秒
            'avg_eval_time': (
                self.stats['total_eval_time'] / 
                max(self.stats['expressions_evaluated'], 1)
            ) * 1000,  # 毫秒
        }
    
    def clear_cache_Processor(self):
        """清空缓存"""
        self.cache.clear()
        logger.info("复杂逻辑处理器缓存已清空")
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'expressions_parsed': 0,
            'expressions_evaluated': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_parse_time': 0.0,
            'total_eval_time': 0.0,
        }
        logger.info("复杂逻辑处理器统计信息已重置")
