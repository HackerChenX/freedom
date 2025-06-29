"""
增强版策略执行器

使用买点分析的计算引擎，提供更准确的策略选股能力
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime

from analysis.buypoints.analyze_buypoints import BuyPointAnalyzer
from strategy.strategy_executor import StrategyExecutor
from utils.logger import get_logger

logger = get_logger(__name__)


class EnhancedStrategyExecutor(StrategyExecutor):
    """
    增强版策略执行器
    
    集成买点分析的计算引擎，提供更准确的指标计算和条件评估
    """
    
    def __init__(self):
        """初始化增强版策略执行器"""
        super().__init__()
        self.buypoint_analyzer = BuyPointAnalyzer()
        logger.info("增强版策略执行器已初始化，集成买点分析引擎")
    
    def _evaluate_stock_with_buypoint_engine(self, 
                                           stock_code: str,
                                           conditions: List[Dict[str, Any]],
                                           date: str) -> Dict[str, Any]:
        """
        使用买点分析引擎评估股票
        
        Args:
            stock_code: 股票代码
            conditions: 策略条件列表
            date: 评估日期
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        try:
            # 转换日期格式
            eval_date = date.replace('-', '')
            
            # 使用买点分析器分析股票
            buypoint_result = self.buypoint_analyzer.analyze_stock(
                stock_code=stock_code,
                buy_date=eval_date,
                stock_name=""
            )
            
            if not buypoint_result:
                return {
                    'stock_code': stock_code,
                    'meets_conditions': False,
                    'score': 0,
                    'details': {'error': '无法获取买点分析数据'}
                }
            
            # 评估策略条件
            condition_results = []
            total_score = 0
            
            for condition in conditions:
                result = self._evaluate_condition_with_buypoint_data(
                    condition, buypoint_result, date)
                condition_results.append(result)
                
                if result['met']:
                    total_score += result.get('score', 10)
            
            # 判断是否满足所有条件
            meets_conditions = all(r['met'] for r in condition_results)
            
            return {
                'stock_code': stock_code,
                'meets_conditions': meets_conditions,
                'score': total_score,
                'details': {
                    'condition_results': condition_results,
                    'buypoint_data': buypoint_result
                }
            }
            
        except Exception as e:
            logger.error(f"使用买点引擎评估股票 {stock_code} 失败: {e}")
            return {
                'stock_code': stock_code,
                'meets_conditions': False,
                'score': 0,
                'details': {'error': str(e)}
            }
    
    def _evaluate_condition_with_buypoint_data(self, 
                                             condition: Dict[str, Any],
                                             buypoint_data: Dict[str, Any],
                                             date: str) -> Dict[str, Any]:
        """
        使用买点数据评估条件
        
        Args:
            condition: 条件配置
            buypoint_data: 买点分析数据
            date: 评估日期
            
        Returns:
            Dict[str, Any]: 条件评估结果
        """
        try:
            condition_type = condition.get('type', 'basic')
            
            if condition_type == 'indicator':
                return self._evaluate_indicator_condition_enhanced(
                    condition, buypoint_data, date)
            elif condition_type == 'pattern':
                return self._evaluate_pattern_condition_enhanced(
                    condition, buypoint_data, date)
            elif condition_type == 'basic':
                return self._evaluate_basic_condition_enhanced(
                    condition, buypoint_data, date)
            else:
                return {
                    'met': False,
                    'score': 0,
                    'details': f'未知条件类型: {condition_type}'
                }
                
        except Exception as e:
            logger.error(f"评估条件失败: {e}")
            return {
                'met': False,
                'score': 0,
                'details': f'条件评估错误: {str(e)}'
            }
    
    def _evaluate_indicator_condition_enhanced(self, 
                                             condition: Dict[str, Any],
                                             buypoint_data: Dict[str, Any],
                                             date: str) -> Dict[str, Any]:
        """
        增强版指标条件评估
        
        Args:
            condition: 指标条件配置
            buypoint_data: 买点分析数据
            date: 评估日期
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        indicator_id = condition.get('indicator_id', '')
        field = condition.get('field', '')
        operator = condition.get('operator', '>')
        value = condition.get('value', 0)
        reference_field = condition.get('reference_field', '')
        
        # 映射指标字段到买点数据字段
        field_mapping = {
            'MA5': 'ma5',
            'MA10': 'ma10', 
            'MA20': 'ma20',
            'MA30': 'ma30',
            'MA60': 'ma60',
            'RSI': 'rsi',  # 需要在买点分析中添加RSI计算
            'MACD': 'macd',
            'DIF': 'dif',
            'DEA': 'dea',
            'K': 'kdj_k',
            'D': 'kdj_d',
            'J': 'kdj_j',
            'close': 'close',
            'volume': 'vol'
        }
        
        # 获取实际字段名
        actual_field = field_mapping.get(field, field.lower())
        
        # 获取指标值
        if actual_field in buypoint_data:
            actual_value = buypoint_data[actual_field]
        else:
            return {
                'met': False,
                'score': 0,
                'details': f'字段 {field} 不存在于买点数据中'
            }
        
        # 获取比较值
        if reference_field:
            # 如果是与其他字段比较
            ref_field = field_mapping.get(reference_field, reference_field.lower())
            if ref_field in buypoint_data:
                compare_value = buypoint_data[ref_field]
            else:
                return {
                    'met': False,
                    'score': 0,
                    'details': f'参考字段 {reference_field} 不存在'
                }
        else:
            # 如果是与固定值比较
            compare_value = value
        
        # 执行比较
        if operator == '>':
            met = actual_value > compare_value
        elif operator == '<':
            met = actual_value < compare_value
        elif operator == '>=':
            met = actual_value >= compare_value
        elif operator == '<=':
            met = actual_value <= compare_value
        elif operator == '==':
            met = abs(actual_value - compare_value) < 1e-6
        else:
            return {
                'met': False,
                'score': 0,
                'details': f'不支持的操作符: {operator}'
            }
        
        return {
            'met': met,
            'score': 15 if met else 0,
            'details': {
                'field': field,
                'actual_value': actual_value,
                'operator': operator,
                'compare_value': compare_value,
                'condition_desc': f'{field}({actual_value}) {operator} {compare_value}'
            }
        }
    
    def _evaluate_pattern_condition_enhanced(self, 
                                           condition: Dict[str, Any],
                                           buypoint_data: Dict[str, Any],
                                           date: str) -> Dict[str, Any]:
        """
        增强版形态条件评估
        
        Args:
            condition: 形态条件配置
            buypoint_data: 买点分析数据
            date: 评估日期
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        pattern_name = condition.get('pattern', '')
        
        # 映射形态名称到买点数据字段
        pattern_mapping = {
            'touch_ma': 'touch_ma',
            'price_stable': 'price_stable',
            'ma_up': 'ma_up',
            'money_in': 'money_in',
            'kpattern': 'kpattern',
            'vol_shrink': 'vol_shrink',
            'macd_gold': 'macd_gold',
            'absorb_signal': 'xc'  # 吸筹信号
        }
        
        # 获取形态字段
        pattern_field = pattern_mapping.get(pattern_name, pattern_name)
        
        if pattern_field in buypoint_data:
            met = bool(buypoint_data[pattern_field])
            return {
                'met': met,
                'score': 20 if met else 0,
                'details': {
                    'pattern': pattern_name,
                    'detected': met
                }
            }
        else:
            return {
                'met': False,
                'score': 0,
                'details': f'形态 {pattern_name} 不存在于买点数据中'
            }
    
    def _evaluate_basic_condition_enhanced(self, 
                                         condition: Dict[str, Any],
                                         buypoint_data: Dict[str, Any],
                                         date: str) -> Dict[str, Any]:
        """
        增强版基础条件评估
        
        Args:
            condition: 基础条件配置
            buypoint_data: 买点分析数据
            date: 评估日期
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        field = condition.get('field', 'close')
        operator = condition.get('operator', '>')
        value = condition.get('value', 0)
        
        # 基础字段映射
        field_mapping = {
            'close': 'close',
            'volume': 'vol',
            'price': 'close'
        }
        
        actual_field = field_mapping.get(field, field)
        
        if actual_field in buypoint_data:
            actual_value = buypoint_data[actual_field]
            
            # 执行比较
            if operator == '>':
                met = actual_value > value
            elif operator == '<':
                met = actual_value < value
            elif operator == '>=':
                met = actual_value >= value
            elif operator == '<=':
                met = actual_value <= value
            else:
                met = False
            
            return {
                'met': met,
                'score': 10 if met else 0,
                'details': {
                    'field': field,
                    'actual_value': actual_value,
                    'operator': operator,
                    'expected_value': value
                }
            }
        else:
            return {
                'met': False,
                'score': 0,
                'details': f'字段 {field} 不存在'
            } 