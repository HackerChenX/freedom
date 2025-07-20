"""
策略条件评估器模块

负责评估策略条件，计算股票是否满足策略要求
"""

import pandas as pd
import numpy as np
import operator
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import logging

from utils.dependency_injection import get_config
from utils.dependency_injection import get_logger
from utils.decorators import performance_monitor, cache_result, exception_handler
from indicators.complete_indicator_registry import complete_registry
from utils.dependency_injection import get_service
from db.interfaces.data_access_interface import DataAccessInterface
from utils.parameter_standardizer import ParameterStandardizer
from utils.indicator_parameter_validator import IndicatorParameterValidator
from utils.exceptions import (
    StrategyEvaluationError,
    IndicatorExecutionError,
    DataValidationError
)

logger = get_logger(__name__)

class StrategyConditionEvaluator:
    """策略条件评估器，用于高效评估选股策略条件"""
    
    def __init__(self):
        """初始化条件评估器"""
        try:
            self.data_manager = get_service(DataAccessInterface)
        except Exception as e:
            logger.warning(f"依赖注入获取DataAccessInterface失败: {e}, 使用临时实现")
            # 临时实现，直接导入
            from db.managers.data_access_manager import DataAccessManager
            self.data_manager = DataAccessManager()
            
        self.indicator_registry = complete_registry
        self.condition_cache = {}

        # 初始化参数标准化器和验证器
        try:
            self.parameter_standardizer = ParameterStandardizer()
            self.parameter_validator = IndicatorParameterValidator()
        except Exception as e:
            logger.warning(f"参数标准化器/验证器初始化失败: {e}")
            self.parameter_standardizer = None
            self.parameter_validator = None

        logger.info("策略条件评估器已初始化，支持参数标准化和验证")
        
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
        }

    @performance_monitor(threshold=2.0)
    @exception_handler(reraise=True)
    def evaluate_conditions(self, stock_code: str, conditions: List[Dict], start_date: str, end_date: str) -> bool:
        """
        评估股票是否满足策略条件
        
        Args:
            stock_code: 股票代码
            conditions: 条件列表
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            bool: 是否满足所有条件
        """
        try:
            # 获取股票数据
            stock_data = self._get_stock_data(stock_code, start_date, end_date)
            if stock_data.empty:
                logger.warning(f"股票 {stock_code} 数据为空")
                return False
            
            # 计算所有需要的指标
            indicator_values = self._calculate_indicators(stock_data, conditions)
            
            # 评估所有条件
            for condition in conditions:
                if not self._evaluate_single_condition(condition, indicator_values, stock_data):
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"评估股票 {stock_code} 条件失败: {e}")
            return False
            
    def _get_stock_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取股票数据"""
        try:
            return self.data_manager.get_stock_data(stock_code, start_date, end_date)
        except Exception as e:
            logger.error(f"获取股票 {stock_code} 数据失败: {e}")
            return pd.DataFrame()
            
    def _calculate_indicators(self, stock_data: pd.DataFrame, conditions: List[Dict]) -> Dict[str, Any]:
        """计算指标值"""
        indicator_values = {}
        
        for condition in conditions:
            indicator_name = condition.get('indicator')
            if not indicator_name or indicator_name in indicator_values:
                continue
                
            try:
                # 获取指标计算器
                indicator_calculator = self.indicator_registry.get_indicator(indicator_name)
                if not indicator_calculator:
                    logger.warning(f"未找到指标: {indicator_name}")
                    continue
                
                # 计算指标值
                params = condition.get('parameters', {})
                result = indicator_calculator.calculate(stock_data, **params)
                indicator_values[indicator_name] = result
                
            except Exception as e:
                logger.error(f"计算指标 {indicator_name} 失败: {e}")
                
        return indicator_values
        
    def _evaluate_single_condition(self, condition: Dict, indicator_values: Dict, stock_data: pd.DataFrame) -> bool:
        """评估单个条件"""
        try:
            indicator_name = condition.get('indicator')
            operator_str = condition.get('operator', '>')
            threshold = condition.get('threshold', 0)
            
            if indicator_name not in indicator_values:
                return False
                
            indicator_value = indicator_values[indicator_name]
            
            # 获取最新值
            if isinstance(indicator_value, pd.Series):
                latest_value = indicator_value.iloc[-1]
            elif isinstance(indicator_value, (list, np.ndarray)):
                latest_value = indicator_value[-1]
            else:
                latest_value = indicator_value
                
            # 应用操作符
            operator_func = self.operators.get(operator_str)
            if not operator_func:
                logger.warning(f"不支持的操作符: {operator_str}")
                return False
                
            return operator_func(latest_value, threshold)
            
        except Exception as e:
            logger.error(f"评估条件失败: {e}")
            return False 