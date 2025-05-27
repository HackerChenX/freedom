"""
通达信公式专用指标模块

提供通达信公式转换为选股策略时使用的特定指标实现
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
import re

from indicators.base_indicator import BaseIndicator
from indicators.technical_indicators import KDJ, MACD, MA
from utils.logger import get_logger
from indicators.factory import IndicatorFactory

logger = get_logger(__name__)


class CrossOver(BaseIndicator):
    """
    交叉指标，检测快线是否上穿慢线
    """
    
    def __init__(self, fast_line: str = "", slow_line: str = "", **kwargs):
        """
        初始化交叉指标
        
        Args:
            fast_line: 快线表达式
            slow_line: 慢线表达式
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        self.fast_line = fast_line
        self.slow_line = slow_line
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算指标值
        
        Args:
            data: 输入数据
            
        Returns:
            包含指标值的DataFrame
        """
        result = data.copy()
        
        # 解析快线和慢线
        fast_values = self._parse_line_expression(self.fast_line, data)
        slow_values = self._parse_line_expression(self.slow_line, data)
        
        if fast_values is not None and slow_values is not None:
            # 计算交叉信号
            # 交叉条件: 前一个时刻快线低于慢线，当前时刻快线高于慢线
            cross_signal = np.zeros(len(data))
            
            for i in range(1, len(data)):
                if fast_values[i-1] < slow_values[i-1] and fast_values[i] >= slow_values[i]:
                    cross_signal[i] = 1
            
            result['cross_signal'] = cross_signal
            
        return result
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        Args:
            data: 包含指标值的DataFrame
            
        Returns:
            带有信号的DataFrame
        """
        result = data.copy()
        
        if 'cross_signal' in result.columns:
            result['signal'] = 0
            result.loc[result['cross_signal'] > 0, 'signal'] = 1
            
        return result
    
    def _parse_line_expression(self, expression: str, data: pd.DataFrame) -> Optional[np.ndarray]:
        """
        解析线表达式
        
        Args:
            expression: 线表达式
            data: 输入数据
            
        Returns:
            解析后的数值数组
        """
        try:
            # 简单的价格引用替换
            expression = expression.replace('C', 'close').replace('O', 'open') \
                                   .replace('H', 'high').replace('L', 'low') \
                                   .replace('V', 'volume')
            
            # 检查表达式中的函数调用
            ma_match = re.search(r'MA\s*\(\s*(\w+)\s*,\s*(\d+)\s*\)', expression)
            if ma_match:
                series_name = ma_match.group(1).lower()
                period = int(ma_match.group(2))
                
                if series_name in data.columns:
                    ma_indicator = MA(periods=[period])
                    result = ma_indicator.calculate(data)
                    return result[f'MA{period}'].values
            
            # 简单的列引用
            if expression.lower() in data.columns:
                return data[expression.lower()].values
                
            # 其他复杂表达式解析
            # 这里可以扩展更多的解析逻辑
            
            logger.warning(f"无法解析表达式: {expression}")
            return None
        except Exception as e:
            logger.error(f"解析表达式 {expression} 出错: {e}")
            return None


class KDJCondition(BaseIndicator):
    """
    KDJ条件指标，检测KDJ指标线的条件
    """
    
    def __init__(self, line: str = "K", operator: str = ">", value: float = 0, **kwargs):
        """
        初始化KDJ条件指标
        
        Args:
            line: KDJ指标线，可选K/D/J
            operator: 比较操作符，如> < >= <= =
            value: 比较值
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        self.line = line.upper()
        self.operator = operator
        self.value = value
        
        # KDJ默认参数
        self.n = kwargs.get('n', 9)
        self.m1 = kwargs.get('m1', 3)
        self.m2 = kwargs.get('m2', 3)
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算指标值
        
        Args:
            data: 输入数据
            
        Returns:
            包含指标值的DataFrame
        """
        result = data.copy()
        
        # 计算KDJ指标
        kdj = KDJ(n=self.n, m1=self.m1, m2=self.m2)
        kdj_result = kdj.calculate(data)
        
        # 合并结果
        result = pd.concat([result, kdj_result], axis=1)
        
        # 检查条件
        if self.line in ['K', 'D', 'J']:
            line_value = result[self.line.lower()].values
            condition = self._evaluate_condition(line_value, self.operator, self.value)
            result['condition_met'] = condition
            
        return result
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        Args:
            data: 包含指标值的DataFrame
            
        Returns:
            带有信号的DataFrame
        """
        result = data.copy()
        
        if 'condition_met' in result.columns:
            result['signal'] = 0
            result.loc[result['condition_met'] > 0, 'signal'] = 1
            
        return result
    
    def _evaluate_condition(self, values: np.ndarray, operator: str, threshold: float) -> np.ndarray:
        """
        评估条件
        
        Args:
            values: 数值数组
            operator: 比较操作符
            threshold: 阈值
            
        Returns:
            条件结果数组，1表示满足条件，0表示不满足
        """
        result = np.zeros_like(values)
        
        if operator == '>':
            result[values > threshold] = 1
        elif operator == '>=':
            result[values >= threshold] = 1
        elif operator == '<':
            result[values < threshold] = 1
        elif operator == '<=':
            result[values <= threshold] = 1
        elif operator in ['=', '==']:
            result[np.isclose(values, threshold)] = 1
        elif operator in ['!=', '<>']:
            result[~np.isclose(values, threshold)] = 1
            
        return result


class MACDCondition(BaseIndicator):
    """
    MACD条件指标，检测MACD指标线的条件
    """
    
    def __init__(self, line: str = "MACD", operator: str = ">", value: float = 0, **kwargs):
        """
        初始化MACD条件指标
        
        Args:
            line: MACD指标线，可选MACD/DIF/DEA
            operator: 比较操作符，如> < >= <= =
            value: 比较值
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        self.line = line.upper()
        self.operator = operator
        self.value = value
        
        # MACD默认参数
        self.short = kwargs.get('short', 12)
        self.long = kwargs.get('long', 26)
        self.mid = kwargs.get('mid', 9)
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算指标值
        
        Args:
            data: 输入数据
            
        Returns:
            包含指标值的DataFrame
        """
        result = data.copy()
        
        # 计算MACD指标
        macd = MACD(short=self.short, long=self.long, mid=self.mid)
        macd_result = macd.calculate(data)
        
        # 合并结果
        result = pd.concat([result, macd_result], axis=1)
        
        # 检查条件
        line_map = {
            'DIF': 'dif',
            'DEA': 'dea',
            'MACD': 'macd'
        }
        
        if self.line in line_map:
            line_value = result[line_map[self.line]].values
            condition = self._evaluate_condition(line_value, self.operator, self.value)
            result['condition_met'] = condition
            
        return result
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        Args:
            data: 包含指标值的DataFrame
            
        Returns:
            带有信号的DataFrame
        """
        result = data.copy()
        
        if 'condition_met' in result.columns:
            result['signal'] = 0
            result.loc[result['condition_met'] > 0, 'signal'] = 1
            
        return result
    
    def _evaluate_condition(self, values: np.ndarray, operator: str, threshold: float) -> np.ndarray:
        """
        评估条件
        
        Args:
            values: 数值数组
            operator: 比较操作符
            threshold: 阈值
            
        Returns:
            条件结果数组，1表示满足条件，0表示不满足
        """
        result = np.zeros_like(values)
        
        if operator == '>':
            result[values > threshold] = 1
        elif operator == '>=':
            result[values >= threshold] = 1
        elif operator == '<':
            result[values < threshold] = 1
        elif operator == '<=':
            result[values <= threshold] = 1
        elif operator in ['=', '==']:
            result[np.isclose(values, threshold)] = 1
        elif operator in ['!=', '<>']:
            result[~np.isclose(values, threshold)] = 1
            
        return result


class MACondition(BaseIndicator):
    """
    均线条件指标，检测均线与价格或其他均线的关系
    """
    
    def __init__(self, ma_type: str = "MA", ma_period: int = 5, 
               operator: str = ">", compare_value: str = "CLOSE", **kwargs):
        """
        初始化均线条件指标
        
        Args:
            ma_type: 均线类型，MA/EMA/WMA
            ma_period: 均线周期
            operator: 比较操作符，如> < >= <= =
            compare_value: 比较值，可以是CLOSE/OPEN/HIGH/LOW或另一个均线的表达式
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        self.ma_type = ma_type.upper()
        self.ma_period = ma_period
        self.operator = operator
        self.compare_value = compare_value
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算指标值
        
        Args:
            data: 输入数据
            
        Returns:
            包含指标值的DataFrame
        """
        result = data.copy()
        
        # 计算均线
        ma = MA(periods=[self.ma_period])
        ma_result = ma.calculate(data)
        
        # 合并结果
        for col in ma_result.columns:
            result[col] = ma_result[col]
        
        # 解析比较值
        compare_values = self._parse_compare_value(self.compare_value, data)
        
        # 检查条件
        if compare_values is not None:
            ma_values = result[f'MA{self.ma_period}'].values
            condition = self._evaluate_condition(ma_values, self.operator, compare_values)
            result['condition_met'] = condition
            
        return result
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        Args:
            data: 包含指标值的DataFrame
            
        Returns:
            带有信号的DataFrame
        """
        result = data.copy()
        
        if 'condition_met' in result.columns:
            result['signal'] = 0
            result.loc[result['condition_met'] > 0, 'signal'] = 1
            
        return result
    
    def _parse_compare_value(self, compare_value: str, data: pd.DataFrame) -> Optional[np.ndarray]:
        """
        解析比较值
        
        Args:
            compare_value: 比较值表达式
            data: 输入数据
            
        Returns:
            解析后的数值数组
        """
        try:
            # 尝试将比较值转换为数值
            try:
                return np.full(len(data), float(compare_value))
            except ValueError:
                pass
                
            # 简单的价格引用替换
            compare_value = compare_value.replace('C', 'close').replace('O', 'open') \
                                       .replace('H', 'high').replace('L', 'low')
            
            # 检查表达式中的函数调用
            ma_match = re.search(r'MA\s*\(\s*(\w+)\s*,\s*(\d+)\s*\)', compare_value)
            if ma_match:
                series_name = ma_match.group(1).lower()
                period = int(ma_match.group(2))
                
                if series_name in data.columns:
                    ma_indicator = MA(periods=[period])
                    result = ma_indicator.calculate(data)
                    return result[f'MA{period}'].values
            
            # 简单的列引用
            if compare_value.lower() in data.columns:
                return data[compare_value.lower()].values
                
            # 其他复杂表达式解析
            # 这里可以扩展更多的解析逻辑
            
            logger.warning(f"无法解析比较值: {compare_value}")
            return None
        except Exception as e:
            logger.error(f"解析比较值 {compare_value} 出错: {e}")
            return None
    
    def _evaluate_condition(self, values: np.ndarray, operator: str, compare_values: np.ndarray) -> np.ndarray:
        """
        评估条件
        
        Args:
            values: 均线值数组
            operator: 比较操作符
            compare_values: 比较值数组
            
        Returns:
            条件结果数组，1表示满足条件，0表示不满足
        """
        result = np.zeros_like(values)
        
        if operator == '>':
            result[values > compare_values] = 1
        elif operator == '>=':
            result[values >= compare_values] = 1
        elif operator == '<':
            result[values < compare_values] = 1
        elif operator == '<=':
            result[values <= compare_values] = 1
        elif operator in ['=', '==']:
            result[np.isclose(values, compare_values)] = 1
        elif operator in ['!=', '<>']:
            result[~np.isclose(values, compare_values)] = 1
            
        return result


class GenericCondition(BaseIndicator):
    """
    通用条件指标，支持通达信公式中的通用条件表达式
    """
    
    def __init__(self, condition: str = "", **kwargs):
        """
        初始化通用条件指标
        
        Args:
            condition: 条件表达式
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        self.condition = condition
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算指标值
        
        Args:
            data: 输入数据
            
        Returns:
            包含指标值的DataFrame
        """
        result = data.copy()
        
        # 简单实现：所有数据都满足条件
        # 实际情况需要根据条件表达式进行解析和计算
        result['condition_met'] = 1
        
        return result
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        Args:
            data: 包含指标值的DataFrame
            
        Returns:
            带有信号的DataFrame
        """
        result = data.copy()
        
        if 'condition_met' in result.columns:
            result['signal'] = 0
            result.loc[result['condition_met'] > 0, 'signal'] = 1
            
        return result


# 注册指标到工厂
IndicatorFactory.register_indicator("CROSS_OVER", CrossOver)
IndicatorFactory.register_indicator("KDJ_CONDITION", KDJCondition)
IndicatorFactory.register_indicator("MACD_CONDITION", MACDCondition)
IndicatorFactory.register_indicator("MA_CONDITION", MACondition)
IndicatorFactory.register_indicator("GENERIC_CONDITION", GenericCondition) 