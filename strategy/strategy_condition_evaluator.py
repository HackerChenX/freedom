"""
策略条件评估器

提供高性能的策略条件评估能力
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable
import operator
import re
import ast

from utils.logger import get_logger
from db.data_manager import DataManager
from indicators.indicator_manager import IndicatorManager
from utils.decorators import performance_monitor, cache_result

logger = get_logger(__name__)

class StrategyConditionEvaluator:
    """策略条件评估器，用于高效评估选股策略条件"""
    
    def __init__(self):
        """初始化条件评估器"""
        self.data_manager = DataManager()
        self.indicator_manager = IndicatorManager()
        self.condition_cache = {}
        
        # 操作符映射
        self.operators = {
            '>': operator.gt,
            '<': operator.lt,
            '>=': operator.ge,
            '<=': operator.le,
            '==': operator.eq,
            '!=': operator.ne,
            'and': operator.and_,
            'or': operator.or_,
            'not': operator.not_,
            'cross_above': self._cross_above,
            'cross_below': self._cross_below,
            'between': self._between,
            'increasing': self._is_increasing,
            'decreasing': self._is_decreasing,
            'top_percent': self._top_percent,
            'bottom_percent': self._bottom_percent
        }
    
    def _cross_above(self, series1: pd.Series, series2: Union[pd.Series, float]) -> pd.Series:
        """
        检查series1是否向上穿越series2
        
        Args:
            series1: 第一个数据序列
            series2: 第二个数据序列或固定值
            
        Returns:
            pd.Series: 布尔序列，True表示向上穿越
        """
        if isinstance(series2, (int, float)):
            series2 = pd.Series(series2, index=series1.index)
            
        # 获取前一天的值
        prev_series1 = series1.shift(1)
        prev_series2 = series2.shift(1)
        
        # 当前值大于等于series2且前一天值小于series2
        cross_above = (series1 >= series2) & (prev_series1 < prev_series2)
        
        return cross_above
    
    def _cross_below(self, series1: pd.Series, series2: Union[pd.Series, float]) -> pd.Series:
        """
        检查series1是否向下穿越series2
        
        Args:
            series1: 第一个数据序列
            series2: 第二个数据序列或固定值
            
        Returns:
            pd.Series: 布尔序列，True表示向下穿越
        """
        if isinstance(series2, (int, float)):
            series2 = pd.Series(series2, index=series1.index)
            
        # 获取前一天的值
        prev_series1 = series1.shift(1)
        prev_series2 = series2.shift(1)
        
        # 当前值小于等于series2且前一天值大于series2
        cross_below = (series1 <= series2) & (prev_series1 > prev_series2)
        
        return cross_below
    
    def _between(self, series: pd.Series, lower: float, upper: float) -> pd.Series:
        """
        检查series是否在lower和upper之间
        
        Args:
            series: 数据序列
            lower: 下限
            upper: 上限
            
        Returns:
            pd.Series: 布尔序列，True表示在范围内
        """
        return (series >= lower) & (series <= upper)
    
    def _is_increasing(self, series: pd.Series, periods: int = 3) -> pd.Series:
        """
        检查series是否连续上升
        
        Args:
            series: 数据序列
            periods: 连续周期数
            
        Returns:
            pd.Series: 布尔序列，True表示连续上升
        """
        result = pd.Series(True, index=series.index)
        
        for i in range(1, periods + 1):
            result = result & (series > series.shift(i))
            
        return result
    
    def _is_decreasing(self, series: pd.Series, periods: int = 3) -> pd.Series:
        """
        检查series是否连续下降
        
        Args:
            series: 数据序列
            periods: 连续周期数
            
        Returns:
            pd.Series: 布尔序列，True表示连续下降
        """
        result = pd.Series(True, index=series.index)
        
        for i in range(1, periods + 1):
            result = result & (series < series.shift(i))
            
        return result
    
    def _top_percent(self, series: pd.Series, percent: float) -> pd.Series:
        """
        检查series是否在前N%
        
        Args:
            series: 数据序列
            percent: 百分比（0-100）
            
        Returns:
            pd.Series: 布尔序列，True表示在前N%
        """
        # 计算分位数阈值
        threshold = series.quantile(1 - percent/100)
        
        # 大于阈值的值在前N%
        return series > threshold
    
    def _bottom_percent(self, series: pd.Series, percent: float) -> pd.Series:
        """
        检查series是否在后N%
        
        Args:
            series: 数据序列
            percent: 百分比（0-100）
            
        Returns:
            pd.Series: 布尔序列，True表示在后N%
        """
        # 计算分位数阈值
        threshold = series.quantile(percent/100)
        
        # 小于阈值的值在后N%
        return series < threshold
    
    @performance_monitor()
    @cache_result(cache_size=100)
    def evaluate_condition(self, condition: Dict[str, Any], 
                         stock_data: pd.DataFrame,
                         date: str) -> bool:
        """
        评估单个条件
        
        Args:
            condition: 条件配置
            stock_data: 股票数据
            date: 评估日期
            
        Returns:
            bool: 条件评估结果
        """
        try:
            condition_type = condition.get("type", "")
            
            if condition_type == "price":
                return self._evaluate_price_condition(condition, stock_data, date)
            elif condition_type == "volume":
                return self._evaluate_volume_condition(condition, stock_data, date)
            elif condition_type == "indicator":
                return self._evaluate_indicator_condition(condition, stock_data, date)
            elif condition_type == "fundamental":
                return self._evaluate_fundamental_condition(condition, stock_data, date)
            elif condition_type == "pattern":
                return self._evaluate_pattern_condition(condition, stock_data, date)
            elif condition_type == "logic":
                return self._evaluate_logic_condition(condition, stock_data, date)
            elif condition_type == "expression":
                return self._evaluate_expression_condition(condition, stock_data, date)
            else:
                logger.warning(f"未知的条件类型: {condition_type}")
                return False
                
        except Exception as e:
            logger.error(f"评估条件时出错: {e}")
            return False
    
    @performance_monitor()
    def evaluate_conditions(self, conditions: List[Dict[str, Any]], 
                         stock_data: pd.DataFrame,
                         date: str,
                         logic: str = "and") -> bool:
        """
        评估多个条件
        
        Args:
            conditions: 条件列表
            stock_data: 股票数据
            date: 评估日期
            logic: 条件间逻辑关系，"and"或"or"或"not"
            
        Returns:
            bool: 条件评估结果
        """
        if not conditions:
            return True
        
        # 处理NOT逻辑操作符 - NOT只对第一个条件取反
        if logic.lower() == "not" and conditions:
            first_result = self.evaluate_condition(conditions[0], stock_data, date)
            # 对第一个条件结果取反
            return not first_result
            
        # 评估所有条件
        results = []
        current_logic = logic.lower()
        
        for i, condition in enumerate(conditions):
            # 检查是否存在逻辑运算符条件，它会改变后续条件的逻辑关系
            if condition.get("type") == "logic":
                # 获取新的逻辑运算符
                current_logic = condition.get("value", "").lower()
                # 跳过逻辑运算符条件的评估
                continue
                
            # 评估当前条件
            condition_result = self.evaluate_condition(condition, stock_data, date)
            results.append(condition_result)
        
        # 根据逻辑关系组合结果
        if current_logic == "and":
            return all(results)
        elif current_logic == "or":
            return any(results)
        else:
            logger.warning(f"未知的逻辑操作符: {current_logic}，默认使用AND")
            return all(results)
    
    def _evaluate_price_condition(self, condition: Dict[str, Any], 
                               stock_data: pd.DataFrame,
                               date: str) -> bool:
        """
        评估价格条件
        
        Args:
            condition: 条件配置
            stock_data: 股票数据
            date: 评估日期
            
        Returns:
            bool: 条件评估结果
        """
        # 获取价格列和比较值
        price_field = condition.get("field", "close")
        operator_str = condition.get("operator", ">")
        value = condition.get("value", 0)
        
        # 获取当前日期的价格
        current_price = self._get_value_on_date(stock_data, price_field, date)
        
        if current_price is None:
            return False
            
        # 获取操作符函数
        op_func = self.operators.get(operator_str)
        
        if op_func is None:
            logger.warning(f"未知的操作符: {operator_str}")
            return False
            
        # 执行比较
        return op_func(current_price, value)
    
    def _evaluate_volume_condition(self, condition: Dict[str, Any], 
                                stock_data: pd.DataFrame,
                                date: str) -> bool:
        """
        评估成交量条件
        
        Args:
            condition: 条件配置
            stock_data: 股票数据
            date: 评估日期
            
        Returns:
            bool: 条件评估结果
        """
        # 获取成交量和比较值
        operator_str = condition.get("operator", ">")
        value = condition.get("value", 0)
        periods = condition.get("periods", 1)
        
        # 获取当前日期的成交量
        current_volume = self._get_value_on_date(stock_data, "volume", date)
        
        if current_volume is None:
            return False
            
        # 如果需要与历史成交量比较
        if periods > 1:
            # 获取前N天的平均成交量
            avg_volume = self._get_average_value(stock_data, "volume", date, periods)
            value = avg_volume * value
            
        # 获取操作符函数
        op_func = self.operators.get(operator_str)
        
        if op_func is None:
            logger.warning(f"未知的操作符: {operator_str}")
            return False
            
        # 执行比较
        return op_func(current_volume, value)
    
    def _evaluate_indicator_condition(self, condition: Dict[str, Any], 
                                   stock_data: pd.DataFrame,
                                   date: str) -> bool:
        """
        评估技术指标条件
        
        Args:
            condition: 条件配置
            stock_data: 股票数据
            date: 评估日期
            
        Returns:
            bool: 条件评估结果
        """
        # 获取指标信息
        indicator_name = condition.get("indicator", "")
        parameter = condition.get("parameter", "")
        operator_str = condition.get("operator", ">")
        value = condition.get("value", 0)
        
        # 如果条件中指定了比较的指标
        compare_indicator = condition.get("compare_indicator", "")
        compare_parameter = condition.get("compare_parameter", "")
        
        # 获取指标值
        indicator_value = self._get_indicator_value(stock_data, indicator_name, parameter, date)
        
        if indicator_value is None:
            return False
            
        # 如果需要与其他指标比较
        if compare_indicator:
            compare_value = self._get_indicator_value(stock_data, compare_indicator, compare_parameter, date)
            
            if compare_value is None:
                return False
                
            value = compare_value
        
        # 获取操作符函数
        op_func = self.operators.get(operator_str)
        
        if op_func is None:
            logger.warning(f"未知的操作符: {operator_str}")
            return False
            
        # 执行比较
        return op_func(indicator_value, value)
    
    def _evaluate_fundamental_condition(self, condition: Dict[str, Any], 
                                     stock_data: pd.DataFrame,
                                     date: str) -> bool:
        """
        评估基本面条件
        
        Args:
            condition: 条件配置
            stock_data: 股票数据
            date: 评估日期
            
        Returns:
            bool: 条件评估结果
        """
        # 获取基本面指标和比较值
        field = condition.get("field", "")
        operator_str = condition.get("operator", ">")
        value = condition.get("value", 0)
        
        # 获取基本面数据
        fundamental_value = self._get_fundamental_value(stock_data, field, date)
        
        if fundamental_value is None:
            return False
            
        # 获取操作符函数
        op_func = self.operators.get(operator_str)
        
        if op_func is None:
            logger.warning(f"未知的操作符: {operator_str}")
            return False
            
        # 执行比较
        return op_func(fundamental_value, value)
    
    def _evaluate_pattern_condition(self, condition: Dict[str, Any], 
                                 stock_data: pd.DataFrame,
                                 date: str) -> bool:
        """
        评估形态条件
        
        Args:
            condition: 条件配置
            stock_data: 股票数据
            date: 评估日期
            
        Returns:
            bool: 条件评估结果
        """
        # 获取形态名称
        pattern_name = condition.get("pattern", "")
        
        # 调用形态识别
        return self._check_pattern(stock_data, pattern_name, date)
    
    def _evaluate_logic_condition(self, condition: Dict[str, Any], 
                               stock_data: pd.DataFrame,
                               date: str) -> bool:
        """
        评估逻辑条件
        
        Args:
            condition: 条件配置
            stock_data: 股票数据
            date: 评估日期
            
        Returns:
            bool: 条件评估结果
        """
        # 获取逻辑操作和子条件
        logic_operator = condition.get("logic", "and").lower()
        sub_conditions = condition.get("conditions", [])
        
        # 检查NOT逻辑
        if logic_operator == "not":
            # NOT只能用于单个条件
            if len(sub_conditions) != 1:
                logger.warning(f"NOT逻辑应该只用于单个条件，但发现了{len(sub_conditions)}个条件")
                # 如果有多个条件，我们只处理第一个
                if len(sub_conditions) == 0:
                    return True  # 没有条件就返回True
                sub_conditions = [sub_conditions[0]]
            
            # 对单个条件的结果取反
            result = self.evaluate_condition(sub_conditions[0], stock_data, date)
            return not result
        
        # 处理组合条件 (AND/OR)
        return self.evaluate_conditions(sub_conditions, stock_data, date, logic_operator)
    
    def _evaluate_expression_condition(self, condition: Dict[str, Any], 
                                    stock_data: pd.DataFrame,
                                    date: str) -> bool:
        """
        评估表达式条件
        
        Args:
            condition: 条件配置
            stock_data: 股票数据
            date: 评估日期
            
        Returns:
            bool: 条件评估结果
        """
        # 获取表达式
        expression = condition.get("expression", "")
        
        if not expression:
            return False
            
        # 解析并执行表达式
        return self._evaluate_expression(expression, stock_data, date)
    
    def _get_value_on_date(self, data: pd.DataFrame, field: str, date: str) -> Optional[float]:
        """
        获取指定日期的字段值
        
        Args:
            data: 数据框
            field: 字段名
            date: 日期
            
        Returns:
            Optional[float]: 字段值，如果不存在则返回None
        """
        if field not in data.columns:
            logger.warning(f"字段 {field} 不存在于数据中")
            return None
            
        # 找到日期索引
        if date in data.index:
            return data.loc[date, field]
        else:
            # 尝试找到最接近的日期
            try:
                nearest_date = data.index[data.index <= date][-1]
                return data.loc[nearest_date, field]
            except (IndexError, KeyError):
                logger.warning(f"未找到日期 {date} 或之前的数据")
                return None
    
    def _get_average_value(self, data: pd.DataFrame, field: str, 
                        date: str, periods: int) -> Optional[float]:
        """
        获取指定日期前N个周期的平均值
        
        Args:
            data: 数据框
            field: 字段名
            date: 日期
            periods: 周期数
            
        Returns:
            Optional[float]: 平均值，如果不存在则返回None
        """
        if field not in data.columns:
            logger.warning(f"字段 {field} 不存在于数据中")
            return None
            
        # 找到日期位置
        try:
            if date in data.index:
                date_loc = data.index.get_loc(date)
            else:
                # 找到最接近的日期
                nearest_date = data.index[data.index <= date][-1]
                date_loc = data.index.get_loc(nearest_date)
                
            # 获取前N个周期的数据
            start_loc = max(0, date_loc - periods + 1)
            values = data.iloc[start_loc:date_loc+1][field]
            
            if len(values) > 0:
                return values.mean()
            else:
                return None
                
        except (IndexError, KeyError):
            logger.warning(f"未找到日期 {date} 或之前的数据")
            return None
    
    def _get_indicator_value(self, stock_data: pd.DataFrame, 
                          indicator_name: str, parameter: str, 
                          date: str) -> Optional[float]:
        """
        获取技术指标值
        
        Args:
            stock_data: 股票数据
            indicator_name: 指标名称
            parameter: 参数名
            date: 日期
            
        Returns:
            Optional[float]: 指标值，如果不存在则返回None
        """
        # 检查是否已有计算结果
        indicator_col = f"{indicator_name}_{parameter}"
        
        if indicator_col in stock_data.columns:
            return self._get_value_on_date(stock_data, indicator_col, date)
            
        # 调用指标管理器计算指标
        try:
            indicator_data = self.indicator_manager.calculate_indicator(
                stock_data, indicator_name)
                
            if indicator_data is not None and parameter in indicator_data.columns:
                # 合并指标数据到股票数据
                stock_data[indicator_col] = indicator_data[parameter]
                
                # 返回指定日期的值
                return self._get_value_on_date(stock_data, indicator_col, date)
            else:
                logger.warning(f"指标 {indicator_name} 的参数 {parameter} 不存在")
                return None
                
        except Exception as e:
            logger.error(f"计算指标 {indicator_name} 时出错: {e}")
            return None
    
    def _get_fundamental_value(self, stock_data: pd.DataFrame, 
                            field: str, date: str) -> Optional[float]:
        """
        获取基本面数据
        
        Args:
            stock_data: 股票数据
            field: 字段名
            date: 日期
            
        Returns:
            Optional[float]: 基本面数据，如果不存在则返回None
        """
        # 检查是否已有数据
        if field in stock_data.columns:
            return self._get_value_on_date(stock_data, field, date)
            
        # 获取股票代码
        stock_code = stock_data.get("code", "").iloc[0] if "code" in stock_data.columns else ""
        
        if not stock_code:
            logger.warning("无法获取股票代码")
            return None
            
        # 调用数据管理器获取基本面数据
        try:
            fundamental_data = self.data_manager.get_fundamental_data(
                stock_code, [field], date)
                
            if fundamental_data is not None and field in fundamental_data.columns:
                # 返回指定字段的值
                return fundamental_data[field].iloc[0]
            else:
                logger.warning(f"基本面数据 {field} 不存在")
                return None
                
        except Exception as e:
            logger.error(f"获取基本面数据 {field} 时出错: {e}")
            return None
    
    def _check_pattern(self, stock_data: pd.DataFrame, 
                    pattern_name: str, date: str) -> bool:
        """
        检查是否存在指定形态
        
        Args:
            stock_data: 股票数据
            pattern_name: 形态名称
            date: 日期
            
        Returns:
            bool: 是否存在该形态
        """
        # 检查是否已有计算结果
        pattern_col = f"pattern_{pattern_name}"
        
        if pattern_col in stock_data.columns:
            value = self._get_value_on_date(stock_data, pattern_col, date)
            return bool(value) if value is not None else False
            
        # 调用形态识别器识别形态
        try:
            # 这里需要调用形态识别功能，暂时返回False
            logger.warning(f"形态识别功能未实现: {pattern_name}")
            return False
                
        except Exception as e:
            logger.error(f"识别形态 {pattern_name} 时出错: {e}")
            return False
    
    def _evaluate_expression(self, expression: str, 
                          stock_data: pd.DataFrame,
                          date: str) -> bool:
        """
        评估表达式
        
        Args:
            expression: 表达式字符串
            stock_data: 股票数据
            date: 日期
            
        Returns:
            bool: 表达式结果
        """
        try:
            # 替换变量
            expr = self._replace_variables(expression, stock_data, date)
            
            # 安全执行表达式
            return self._safe_eval(expr)
            
        except Exception as e:
            logger.error(f"评估表达式 '{expression}' 时出错: {e}")
            return False
    
    def _replace_variables(self, expression: str, 
                        stock_data: pd.DataFrame,
                        date: str) -> str:
        """
        替换表达式中的变量
        
        Args:
            expression: 表达式字符串
            stock_data: 股票数据
            date: 日期
            
        Returns:
            str: 替换变量后的表达式
        """
        # 匹配变量 ${var_name}
        pattern = r'\${([^}]*)}'
        
        def replace_var(match):
            var_name = match.group(1)
            
            # 处理指标变量，格式：indicator.name.parameter
            if var_name.startswith('indicator.'):
                parts = var_name.split('.')
                if len(parts) >= 3:
                    indicator_name = parts[1]
                    parameter = parts[2]
                    value = self._get_indicator_value(stock_data, indicator_name, parameter, date)
                    return str(value) if value is not None else 'None'
            
            # 处理价格变量，格式：price.field
            elif var_name.startswith('price.'):
                parts = var_name.split('.')
                if len(parts) >= 2:
                    field = parts[1]
                    value = self._get_value_on_date(stock_data, field, date)
                    return str(value) if value is not None else 'None'
            
            # 处理基本面变量，格式：fundamental.field
            elif var_name.startswith('fundamental.'):
                parts = var_name.split('.')
                if len(parts) >= 2:
                    field = parts[1]
                    value = self._get_fundamental_value(stock_data, field, date)
                    return str(value) if value is not None else 'None'
            
            # 未识别的变量
            logger.warning(f"未识别的变量: {var_name}")
            return 'None'
        
        # 替换所有变量
        return re.sub(pattern, replace_var, expression)
    
    def _safe_eval(self, expression: str) -> bool:
        """
        安全执行表达式
        
        Args:
            expression: 表达式字符串
            
        Returns:
            bool: 表达式结果
        """
        # 定义允许的函数和运算符
        safe_locals = {
            "True": True,
            "False": False,
            "None": None,
            "abs": abs,
            "max": max,
            "min": min,
            "sum": sum,
            "len": len,
            "round": round,
            "and": operator.and_,
            "or": operator.or_,
            "not": operator.not_
        }
        
        try:
            # 将表达式中的 'and', 'or', 'not' 关键字转换为函数调用
            # 例如 'a and b' -> 'and(a, b)'
            expression = expression.replace(" and ", " and(").replace(" or ", " or(")
            if " not " in expression:
                expression = expression.replace(" not ", " not(") + ")"
            # 给每个逻辑运算增加对应的右括号
            close_brackets = expression.count("and(") + expression.count("or(")
            expression += ")" * close_brackets
            
            # 使用ast.literal_eval进行安全评估
            return bool(eval(expression, {"__builtins__": {}}, safe_locals))
        except Exception as e:
            logger.error(f"安全执行表达式 '{expression}' 时出错: {e}")
            return False
    
    def clear_cache(self):
        """清除缓存"""
        self.condition_cache.clear()
        logger.info("已清除条件评估缓存") 