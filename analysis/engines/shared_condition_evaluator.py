"""
共享条件评估器

统一买点分析和策略选股的条件评估逻辑，支持复杂的逻辑运算和函数调用。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable
import operator
import re
import ast
from datetime import datetime

from utils.logger import getLogger
from utils.cache import Memory_cache
from analysis.engines.unified_indicator_engine import Unified_indicator_engine

logger = getLogger(__name__)


class SharedConditionEvaluator:
    """
    共享条件评估器
    
    提供统一的条件评估能力，支持：
    1. 基础比较运算符 (>, <, >=, <=, ==, !=)
    2. 逻辑运算符 (AND, OR, NOT)
    3. 函数调用 (CROSS, REF, COUNT, SUM等)
    4. 嵌套条件评估
    5. 复合逻辑表达式
    """
    
    def __init___116(self, indicator_engine: Optional[Unified_indicator_engine] = None):
        """
        初始化共享条件评估器
        
        Args:
            indicator_engine: 统一指标计算引擎实例
        """
        self.indicator_engine = indicator_engine or Unified_indicator_engine()
        self.cache = Memory_cache.get_instance()
        
        # 性能统计
        self.stats = {
            'evaluations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_time': 0.0
        }
        
        # 比较运算符映射
        self.comparison_operators = {
            '>': operator.gt,
            '<': operator.lt,
            '>=': operator.ge,
            '<=': operator.le,
            '==': operator.eq,
            '!=': operator.ne,
            '=': operator.eq,  # 支持单等号
        }
        
        # 逻辑运算符映射
        self.logical_operators = {
            'AND': self._logical_and,
            'OR': self._logical_or,
            'NOT': self._logical_not,
            '&&': self._logical_and,
            '||': self._logical_or,
            '!': self._logical_not,
        }
        
        # 函数映射
        self.functions = {
            'CROSS': self._cross,
            'REF': self._ref,
            'COUNT': self._count,
            'SUM': self._sum,
            'MAX': self._max,
            'MIN': self._min,
            'ABS': self._abs,
            'SQRT': self._sqrt,
            'LLV': self._llv,
            'HHV': self._hhv,
            'SMA': self._sma,
            'EMA': self._ema,
            'MA': self._ma,
        }
        
        logger.info("共享条件评估器已初始化")
    
    def evaluate_condition(self, 
                          condition: Union[str, Dict[str, Any]], 
                          data: Dict[str, Any],
                          date_idx: Optional[int] = None) -> bool:
        """
        评估单个条件
        
        Args:
            condition: 条件表达式或条件配置字典
            data: 数据字典，包含指标数据和股票数据
            date_idx: 日期索引，如果为None则使用最新数据
            
        Returns:
            bool: 条件是否满足
        """
        import time
        start_time = time.time()
        self.stats['evaluations'] += 1
        
        try:
            # 生成缓存键
            cache_key = self._generate_cache_key_Shared_Condition_Evaluator(condition, date_idx)
            
            # 检查缓存
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                self.stats['cache_hits'] += 1
                return cached_result
            
            self.stats['cache_misses'] += 1
            
            # 评估条件
            if isinstance(condition, str):
                result = self._evaluate_expression(condition, data, date_idx)
            elif isinstance(condition, dict):
                result = self._evaluate_condition_dict(condition, data, date_idx)
            else:
                raise ValueError(f"不支持的条件类型: {type(condition)}")
            
            # 缓存结果
            self.cache.set(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"条件评估失败: {e}")
            return False
        finally:
            self.stats['total_time'] += time.time() - start_time
    
    def evaluate_conditions(self, 
                           conditions: List[Union[str, Dict[str, Any]]],
                           data: Dict[str, Any],
                           logic: str = "AND",
                           date_idx: Optional[int] = None) -> bool:
        """
        评估多个条件
        
        Args:
            conditions: 条件列表
            data: 数据字典
            logic: 逻辑运算符 ("AND", "OR")
            date_idx: 日期索引
            
        Returns:
            bool: 条件组合是否满足
        """
        if not conditions:
            return True
        
        results = []
        for condition in conditions:
            result = self.evaluate_condition(condition, data, date_idx)
            results.append(result)
        
        if logic.upper() == "AND":
            return all(results)
        elif logic.upper() == "OR":
            return any(results)
        else:
            raise ValueError(f"不支持的逻辑运算符: {logic}")
    
    def _evaluate_expression(self, 
                           expression: str, 
                           data: Dict[str, Any],
                           date_idx: Optional[int] = None) -> bool:
        """
        评估表达式字符串
        
        Args:
            expression: 表达式字符串，如 "MA5 > MA10 AND RSI < 30"
            data: 数据字典
            date_idx: 日期索引
            
        Returns:
            bool: 表达式结果
        """
        # 替换表达式中的变量为实际值
        processed_expr = self._replace_variables(expression, data, date_idx)
        
        # 解析并评估表达式
        return self._parse_and_evaluate(processed_expr)
    
    def _evaluate_condition_dict(self, 
                               condition: Dict[str, Any],
                               data: Dict[str, Any],
                               date_idx: Optional[int] = None) -> bool:
        """
        评估条件字典
        
        Args:
            condition: 条件配置字典
            data: 数据字典
            date_idx: 日期索引
            
        Returns:
            bool: 条件是否满足
        """
        condition_type = condition.get('type', 'basic')
        
        if condition_type == 'basic':
            return self._evaluate_basic_condition(condition, data, date_idx)
        elif condition_type == 'indicator':
            return self._evaluate_indicator_condition(condition, data, date_idx)
        elif condition_type == 'pattern':
            return self._evaluate_pattern_condition(condition, data, date_idx)
        elif condition_type == 'logical':
            return self._evaluate_logical_condition(condition, data, date_idx)
        elif condition_type == 'expression':
            expression = condition.get('expression', '')
            return self._evaluate_expression(expression, data, date_idx)
        else:
            raise ValueError(f"不支持的条件类型: {condition_type}")
    
    def _evaluate_basic_condition(self, 
                                condition: Dict[str, Any],
                                data: Dict[str, Any],
                                date_idx: Optional[int] = None) -> bool:
        """
        评估基础条件（价格、成交量等）
        """
        field = condition.get('field', 'close')
        operator_str = condition.get('operator', '>')
        value = condition.get('value', 0)
        reference_field = condition.get('reference_field')
        
        # 获取字段值
        field_value = self._get_field_value(field, data, date_idx)
        if field_value is None:
            return False
        
        # 获取比较值
        if reference_field:
            compare_value = self._get_field_value(reference_field, data, date_idx)
            if compare_value is None:
                return False
        else:
            compare_value = value
        
        # 执行比较
        operator_func = self.comparison_operators.get(operator_str)
        if operator_func is None:
            raise ValueError(f"不支持的比较运算符: {operator_str}")
        
        return operator_func(field_value, compare_value)
    
    def _evaluate_indicator_condition(self, 
                                    condition: Dict[str, Any],
                                    data: Dict[str, Any],
                                    date_idx: Optional[int] = None) -> bool:
        """
        评估指标条件
        """
        indicator_name = condition.get('indicator', '')
        field = condition.get('field', '')
        operator_str = condition.get('operator', '>')
        value = condition.get('value', 0)
        reference_indicator = condition.get('reference_indicator')
        reference_field = condition.get('reference_field')
        
        # 获取指标值
        indicator_value = self._get_indicator_value(
            indicator_name, field, data, date_idx)
        if indicator_value is None:
            return False
        
        # 获取比较值
        if reference_indicator and reference_field:
            compare_value = self._get_indicator_value(
                reference_indicator, reference_field, data, date_idx)
            if compare_value is None:
                return False
        elif reference_field:
            compare_value = self._get_field_value(reference_field, data, date_idx)
            if compare_value is None:
                return False
        else:
            compare_value = value
        
        # 执行比较
        operator_func = self.comparison_operators.get(operator_str)
        if operator_func is None:
            raise ValueError(f"不支持的比较运算符: {operator_str}")
        
        return operator_func(indicator_value, compare_value)
    
    def _evaluate_pattern_condition(self, 
                                  condition: Dict[str, Any],
                                  data: Dict[str, Any],
                                  date_idx: Optional[int] = None) -> bool:
        """
        评估形态条件
        """
        pattern_name = condition.get('pattern', '')
        
        # 从买点分析数据中获取形态信息
        if pattern_name in data:
            return bool(data[pattern_name])
        
        # 如果没有直接的形态数据，尝试计算
        return self._calculate_pattern(pattern_name, data, date_idx)
    
    def _evaluate_logical_condition(self, 
                                  condition: Dict[str, Any],
                                  data: Dict[str, Any],
                                  date_idx: Optional[int] = None) -> bool:
        """
        评估逻辑条件
        """
        operator_str = condition.get('operator', 'AND').upper()
        sub_conditions = condition.get('conditions', [])
        
        if not sub_conditions:
            return True
        
        if operator_str == 'AND':
            return all(self.evaluate_condition(cond, data, date_idx) 
                      for cond in sub_conditions)
        elif operator_str == 'OR':
            return any(self.evaluate_condition(cond, data, date_idx) 
                      for cond in sub_conditions)
        elif operator_str == 'NOT':
            if len(sub_conditions) != 1:
                raise ValueError("NOT运算符只能有一个子条件")
            return not self.evaluate_condition(sub_conditions[0], data, date_idx)
        else:
            raise ValueError(f"不支持的逻辑运算符: {operator_str}")
    
    def _get_field_value(self, 
                        field: str, 
                        data: Dict[str, Any],
                        date_idx: Optional[int] = None) -> Optional[float]:
        """
        获取字段值
        """
        if field in data:
            value = data[field]
            if isinstance(value, (list, np.ndarray)) and date_idx is not None:
                if 0 <= date_idx < len(value):
                    return float(value[date_idx])
                else:
                    return float(value[-1])  # 使用最新值
            elif isinstance(value, (int, float)):
                return float(value)
        
        return None
    
    def _get_indicator_value(self, 
                           indicator_name: str,
                           field: str,
                           data: Dict[str, Any],
                           date_idx: Optional[int] = None) -> Optional[float]:
        """
        获取指标值
        """
        # 首先尝试从数据中直接获取
        full_field_name = f"{indicator_name.lower()}_{field.lower()}" if field else indicator_name.lower()
        value = self._get_field_value(full_field_name, data, date_idx)
        if value is not None:
            return value
        
        # 尝试其他可能的字段名
        alternative_names = [
            f"{indicator_name.lower()}{field.lower()}",
            f"{indicator_name.upper()}_{field.upper()}",
            f"{indicator_name.upper()}{field.upper()}",
            indicator_name.lower(),
            indicator_name.upper()
        ]
        
        for name in alternative_names:
            value = self._get_field_value(name, data, date_idx)
            if value is not None:
                return value
        
        return None
    
    def _calculate_pattern(self, 
                         pattern_name: str,
                         data: Dict[str, Any],
                         date_idx: Optional[int] = None) -> bool:
        """
        计算形态
        """
        # 这里可以实现各种形态识别逻辑
        # 暂时返回False，后续可以扩展
        logger.warning(f"形态 {pattern_name} 的计算逻辑尚未实现")
        return False
    
    def _replace_variables(self, 
                         expression: str,
                         data: Dict[str, Any],
                         date_idx: Optional[int] = None) -> str:
        """
        替换表达式中的变量为实际值
        """
        # 匹配变量模式，如 MA5, RSI, CLOSE等
        pattern = r'\b([A-Z][A-Z0-9_]*)\b'
        
        def replace_var(match):
            var_name = match.group(1)
            value = self._get_field_value(var_name.lower(), data, date_idx)
            if value is not None:
                return str(value)
            
            # 尝试作为指标获取
            value = self._get_indicator_value(var_name, '', data, date_idx)
            if value is not None:
                return str(value)
            
            # 如果找不到，保持原样
            return var_name
        
        return re.sub(pattern, replace_var, expression)
    
    def _parse_and_evaluate(self, expression: str) -> bool:
        """
        解析并评估表达式
        """
        try:
            # 替换逻辑运算符
            expression = expression.replace(' AND ', ' and ')
            expression = expression.replace(' OR ', ' or ')
            expression = expression.replace(' NOT ', ' not ')
            
            # 安全评估表达式
            result = eval(expression)
            return bool(result)
        except Exception as e:
            logger.error(f"表达式评估失败: {expression}, 错误: {e}")
            return False
    
    def _generate_cache_key_Shared_Condition_Evaluator(self, 
                          condition: Union[str, Dict[str, Any]],
                          date_idx: Optional[int] = None) -> str:
        """
        生成缓存键
        """
        if isinstance(condition, str):
            key_data = condition
        else:
            key_data = str(sorted(condition.items()))
        
        return f"condition_{hash(key_data)}_{date_idx}"
    
    # 逻辑运算符函数
    def _logical_and(self, *args) -> bool:
        return all(args)
    
    def _logical_or(self, *args) -> bool:
        return any(args)
    
    def _logical_not(self, arg) -> bool:
        return not arg
    
    # 函数实现
    def _cross(self, series1: np.ndarray, series2: Union[np.ndarray, float]) -> np.ndarray:
        """向上穿越"""
        if isinstance(series2, (int, float)):
            series2 = np.full_like(series1, series2)
        
        # 当前值大于series2且前一天值小于等于series2
        cross_above = (series1 > series2) & (np.roll(series1, 1) <= np.roll(series2, 1))
        cross_above[0] = False  # 第一个值无法比较
        return cross_above
    
    def _ref(self, series: np.ndarray, periods: int) -> np.ndarray:
        """引用前N期数据"""
        return np.roll(series, periods)
    
    def _count(self, condition: np.ndarray, periods: int) -> np.ndarray:
        """计算满足条件的周期数"""
        result = np.zeros_like(condition, dtype=float)
        for i in range(len(condition)):
            start_idx = max(0, i - periods + 1)
            result[i] = np.sum(condition[start_idx:i+1])
        return result
    
    def _sum(self, series: np.ndarray, periods: int) -> np.ndarray:
        """求和"""
        result = np.zeros_like(series)
        for i in range(len(series)):
            start_idx = max(0, i - periods + 1)
            result[i] = np.sum(series[start_idx:i+1])
        return result
    
    def _max(self, series: np.ndarray, periods: int) -> np.ndarray:
        """最大值"""
        result = np.zeros_like(series)
        for i in range(len(series)):
            start_idx = max(0, i - periods + 1)
            result[i] = np.max(series[start_idx:i+1])
        return result
    
    def _min(self, series: np.ndarray, periods: int) -> np.ndarray:
        """最小值"""
        result = np.zeros_like(series)
        for i in range(len(series)):
            start_idx = max(0, i - periods + 1)
            result[i] = np.min(series[start_idx:i+1])
        return result
    
    def _abs(self, series: np.ndarray) -> np.ndarray:
        """绝对值"""
        return np.abs(series)
    
    def _sqrt(self, series: np.ndarray) -> np.ndarray:
        """平方根"""
        return np.sqrt(np.maximum(series, 0))
    
    def _llv(self, series: np.ndarray, periods: int) -> np.ndarray:
        """最低值"""
        return self._min(series, periods)
    
    def _hhv(self, series: np.ndarray, periods: int) -> np.ndarray:
        """最高值"""
        return self._max(series, periods)
    
    def _sma(self, series: np.ndarray, periods: int, weight: float = 1.0) -> np.ndarray:
        """简单移动平均"""
        if hasattr(self.indicator_engine, 'calculate_sma'):
            return self.indicator_engine.calculate_sma(series, periods, weight)
        else:
            # 简单实现
            result = np.zeros_like(series)
            for i in range(len(series)):
                if i == 0:
                    result[i] = series[i]
                else:
                    result[i] = (series[i] * weight + result[i-1] * (periods - weight)) / periods
            return result
    
    def _ema(self, series: np.ndarray, periods: int) -> np.ndarray:
        """指数移动平均"""
        return self.indicator_engine.calculate_ema(series, periods)
    
    def _ma(self, series: np.ndarray, periods: int) -> np.ndarray:
        """移动平均"""
        return self.indicator_engine.calculate_ma(series, periods)
    
    def get_stats_Evaluator(self) -> Dict[str, Any]:
        """获取性能统计"""
        if self.stats['evaluations'] > 0:
            avg_time = self.stats['total_time'] / self.stats['evaluations']
            cache_hit_rate = self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses'])
        else:
            avg_time = 0
            cache_hit_rate = 0
        
        return {
            'evaluations': self.stats['evaluations'],
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'cache_hit_rate': cache_hit_rate,
            'total_time': self.stats['total_time'],
            'avg_time_per_evaluation': avg_time
        }
    
    def clear_cache_Evaluator(self):
        """清空缓存"""
        self.cache.clear()
        logger.info("条件评估器缓存已清空")
    
    def reset_stats_Evaluator(self):
        """重置统计"""
        self.stats = {
            'evaluations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_time': 0.0
        }
        logger.info("条件评估器统计已重置") 