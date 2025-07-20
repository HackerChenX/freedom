#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
策略条件解析器

支持复杂条件组合逻辑的解析和评估
实现 AND、OR、NOT 操作符和嵌套表达式
遵循六层架构规范
"""

import re
import ast
import operator
from typing import Dict, List, Any, Union, Optional, Callable
from dataclasses import dataclass

from utils.logger import getLogger
from utils.decorators import performance_monitor

logger = getLogger(__name__)


@dataclass
class ConditionResult:
    """条件评估结果"""
    condition_id: str
    field: str
    operator: str
    value: Any
    actual_value: Any
    result: bool
    weight: float = 1.0
    error: Optional[str] = None


class ConditionParser:
    """
    策略条件解析器
    
    支持以下功能：
    1. 简单条件评估（向后兼容）
    2. 复杂条件组合（AND、OR、NOT）
    3. 嵌套表达式解析
    4. 条件权重计算
    """
    
    def __init__(self):
        """初始化条件解析器"""
        self.operators = {
            '>': operator.gt,
            '<': operator.lt,
            '>=': operator.ge,
            '<=': operator.le,
            '=': operator.eq,
            '!=': operator.ne
        }
        
        self.logical_operators = {
            'AND': operator.and_,
            'OR': operator.or_,
            'NOT': operator.not_
        }
        
        logger.debug("条件解析器初始化完成")
    
    @performance_monitor(threshold=1.0)
    def evaluate_conditions(
        self,
        conditions: Union[List[Dict[str, Any]], Dict[str, Any]],
        indicator_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        评估条件
        
        Args:
            conditions: 条件配置（简单数组或复杂对象）
            indicator_data: 指标数据
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        try:
            if isinstance(conditions, list):
                # 简单条件数组（向后兼容）
                return self._evaluate_simple_conditions(conditions, indicator_data)
            elif isinstance(conditions, dict):
                # 复杂条件组合
                return self._evaluate_complex_conditions(conditions, indicator_data)
            else:
                raise ValueError(f"不支持的条件格式: {type(conditions)}")
                
        except Exception as e:
            logger.error(f"条件评估失败: {e}")
            return {
                'success': False,
                'result': False,
                'score': 0.0,
                'error': str(e),
                'details': []
            }
    
    def _evaluate_simple_conditions(
        self,
        conditions: List[Dict[str, Any]],
        indicator_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """评估简单条件数组"""
        results = []
        total_weight = 0
        weighted_score = 0
        
        for i, condition in enumerate(conditions):
            condition_id = condition.get('id', f'condition_{i}')
            weight = condition.get('weight', 1.0)
            total_weight += weight
            
            result = self._evaluate_single_condition(
                condition, indicator_data, condition_id
            )
            results.append(result)
            
            if result.result:
                weighted_score += weight
        
        # 计算最终评分
        final_score = weighted_score / total_weight if total_weight > 0 else 0
        overall_result = final_score > 0.5  # 超过50%权重的条件满足
        
        return {
            'success': True,
            'result': overall_result,
            'score': final_score,
            'total_conditions': len(conditions),
            'met_conditions': sum(1 for r in results if r.result),
            'weighted_score': weighted_score,
            'total_weight': total_weight,
            'details': results,
            'logic_type': 'simple_array'
        }
    
    def _evaluate_complex_conditions(
        self,
        conditions: Dict[str, Any],
        indicator_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """评估复杂条件组合"""
        if 'expression' in conditions:
            # 表达式模式
            return self._evaluate_expression(conditions, indicator_data)
        elif 'logic' in conditions and 'conditions' in conditions:
            # 逻辑操作符模式
            return self._evaluate_logic_conditions(conditions, indicator_data)
        else:
            raise ValueError("复杂条件必须包含 'expression' 或 'logic'+'conditions'")
    
    def _evaluate_expression(
        self,
        conditions: Dict[str, Any],
        indicator_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """评估表达式模式的条件"""
        expression = conditions.get('expression', '')
        sub_conditions = conditions.get('conditions', [])
        
        # 构建条件映射
        condition_map = {}
        condition_results = []
        
        for i, condition in enumerate(sub_conditions):
            condition_id = condition.get('id', f'C{i}')
            result = self._evaluate_single_condition(
                condition, indicator_data, condition_id
            )
            condition_results.append(result)
            condition_map[condition_id] = result.result
        
        # 解析和评估表达式
        try:
            expression_result = self._parse_expression(expression, condition_map)
            
            # 计算加权评分
            total_weight = sum(c.get('weight', 1.0) for c in sub_conditions)
            weighted_score = sum(
                r.weight for r in condition_results if r.result
            )
            final_score = weighted_score / total_weight if total_weight > 0 else 0
            
            return {
                'success': True,
                'result': expression_result,
                'score': final_score,
                'expression': expression,
                'condition_map': condition_map,
                'weighted_score': weighted_score,
                'total_weight': total_weight,
                'details': condition_results,
                'logic_type': 'expression'
            }
            
        except Exception as e:
            logger.error(f"表达式评估失败: {expression}, 错误: {e}")
            return {
                'success': False,
                'result': False,
                'score': 0.0,
                'error': f"表达式评估失败: {e}",
                'expression': expression,
                'details': condition_results,
                'logic_type': 'expression'
            }
    
    def _evaluate_logic_conditions(
        self,
        conditions: Dict[str, Any],
        indicator_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """评估逻辑操作符模式的条件"""
        logic = conditions.get('logic', 'AND')
        sub_conditions = conditions.get('conditions', [])
        weight = conditions.get('weight', 1.0)
        
        results = []
        for i, condition in enumerate(sub_conditions):
            if isinstance(condition, dict):
                if 'logic' in condition:
                    # 嵌套复杂条件
                    result = self._evaluate_complex_conditions(condition, indicator_data)
                    # 转换为 ConditionResult 格式
                    condition_result = ConditionResult(
                        condition_id=f'nested_{i}',
                        field='nested',
                        operator=condition.get('logic', 'UNKNOWN'),
                        value=condition,
                        actual_value=result.get('score', 0),
                        result=result.get('result', False),
                        weight=condition.get('weight', 1.0)
                    )
                    results.append(condition_result)
                else:
                    # 简单条件
                    condition_id = condition.get('id', f'condition_{i}')
                    result = self._evaluate_single_condition(
                        condition, indicator_data, condition_id
                    )
                    results.append(result)
        
        # 应用逻辑操作符
        if logic == 'AND':
            final_result = all(r.result for r in results)
        elif logic == 'OR':
            final_result = any(r.result for r in results)
        elif logic == 'NOT':
            # NOT 操作符只对第一个条件取反
            final_result = not results[0].result if results else False
        else:
            raise ValueError(f"不支持的逻辑操作符: {logic}")
        
        # 计算加权评分
        total_weight = sum(r.weight for r in results)
        weighted_score = sum(r.weight for r in results if r.result)
        final_score = weighted_score / total_weight if total_weight > 0 else 0
        
        return {
            'success': True,
            'result': final_result,
            'score': final_score,
            'logic': logic,
            'total_conditions': len(results),
            'met_conditions': sum(1 for r in results if r.result),
            'weighted_score': weighted_score,
            'total_weight': total_weight,
            'details': results,
            'logic_type': 'logic_operator'
        }
    
    def _evaluate_single_condition(
        self,
        condition: Dict[str, Any],
        indicator_data: Dict[str, Any],
        condition_id: str
    ) -> ConditionResult:
        """评估单个条件"""
        try:
            field = condition.get('field')
            op = condition.get('operator')
            value = condition.get('value')
            weight = condition.get('weight', 1.0)
            
            # 获取实际值
            actual_value = self._get_field_value(indicator_data, field)
            
            if actual_value is None:
                return ConditionResult(
                    condition_id=condition_id,
                    field=field,
                    operator=op,
                    value=value,
                    actual_value=None,
                    result=False,
                    weight=weight,
                    error=f"字段 {field} 不存在"
                )
            
            # 执行比较
            result = self._compare_values(actual_value, op, value)
            
            return ConditionResult(
                condition_id=condition_id,
                field=field,
                operator=op,
                value=value,
                actual_value=actual_value,
                result=result,
                weight=weight
            )
            
        except Exception as e:
            logger.warning(f"单个条件评估失败: {condition}, 错误: {e}")
            return ConditionResult(
                condition_id=condition_id,
                field=condition.get('field', 'unknown'),
                operator=condition.get('operator', 'unknown'),
                value=condition.get('value'),
                actual_value=None,
                result=False,
                weight=condition.get('weight', 1.0),
                error=str(e)
            )
    
    def _get_field_value(self, indicator_data: Dict[str, Any], field: str) -> Any:
        """从指标数据中获取字段值"""
        if isinstance(indicator_data, dict):
            return indicator_data.get(field)
        elif hasattr(indicator_data, field):
            return getattr(indicator_data, field)
        else:
            return None
    
    def _compare_values(self, actual_value: Any, operator: str, expected_value: Any) -> bool:
        """比较值"""
        try:
            if operator in self.operators:
                return self.operators[operator](actual_value, expected_value)
            elif operator == 'between':
                if isinstance(expected_value, list) and len(expected_value) == 2:
                    return expected_value[0] <= actual_value <= expected_value[1]
                return False
            elif operator == 'cross_up':
                # 简化的金叉判断
                return actual_value > expected_value
            elif operator == 'cross_down':
                # 简化的死叉判断
                return actual_value < expected_value
            else:
                logger.warning(f"不支持的操作符: {operator}")
                return False
                
        except Exception as e:
            logger.warning(f"值比较失败: {actual_value} {operator} {expected_value}, 错误: {e}")
            return False
    
    def _parse_expression(self, expression: str, condition_map: Dict[str, bool]) -> bool:
        """解析条件表达式"""
        try:
            # 清理表达式
            clean_expr = expression.strip()
            
            # 替换条件标识符为布尔值
            for condition_id, result in condition_map.items():
                clean_expr = clean_expr.replace(condition_id, str(result))
            
            # 替换逻辑操作符
            clean_expr = clean_expr.replace('AND', ' and ')
            clean_expr = clean_expr.replace('OR', ' or ')
            clean_expr = clean_expr.replace('NOT', ' not ')
            
            # 安全评估表达式
            try:
                # 使用 ast.literal_eval 的安全版本
                result = eval(clean_expr, {"__builtins__": {}}, {})
                return bool(result)
            except:
                # 回退到简单的字符串替换方法
                return self._simple_expression_eval(expression, condition_map)
                
        except Exception as e:
            logger.error(f"表达式解析失败: {expression}, 错误: {e}")
            return False
    
    def _simple_expression_eval(self, expression: str, condition_map: Dict[str, bool]) -> bool:
        """简单的表达式评估（回退方法）"""
        try:
            # 简化的表达式评估
            expr = expression.upper()
            
            # 处理简单的 AND/OR 组合
            if 'AND' in expr:
                parts = expr.split('AND')
                return all(
                    condition_map.get(part.strip(), False) for part in parts
                    if part.strip() in condition_map
                )
            elif 'OR' in expr:
                parts = expr.split('OR')
                return any(
                    condition_map.get(part.strip(), False) for part in parts
                    if part.strip() in condition_map
                )
            else:
                # 单个条件
                return condition_map.get(expr.strip(), False)
                
        except Exception as e:
            logger.error(f"简单表达式评估失败: {expression}, 错误: {e}")
            return False
