#!/usr/bin/env python3
"""
参数标准化工具

提供策略条件和指标参数的标准化功能，支持旧格式自动转换
"""

from typing import Dict, List, Any, Optional
from utils.indicator_parameter_validator import IndicatorParameterValidator
from utils.logger import get_logger

logger = get_logger(__name__)


class ParameterStandardizer:
    """参数标准化器"""
    
    def __init__(self):
        """初始化标准化器"""
        self.validator = IndicatorParameterValidator(silent_mode=True)

    def standardize_condition(self, condition: Dict[str, Any]) -> Dict[str, Any]:
        """
        标准化条件（兼容性方法）

        Args:
            condition: 原始条件字典

        Returns:
            标准化后的条件字典
        """
        return self.standardize_indicator_condition(condition)
    
    def standardize_indicator_condition(self, condition: Dict[str, Any]) -> Dict[str, Any]:
        """
        标准化指标条件
        
        Args:
            condition: 原始条件字典
            
        Returns:
            标准化后的条件字典
        """
        try:
            # 提取基本信息
            indicator_id = self._extract_indicator_id(condition)
            signal_type = self._extract_signal_type(condition)
            parameter = self._extract_parameter(condition, indicator_id)
            
            # 构建标准化条件
            standardized = {
                'type': condition.get('type', 'indicator'),
                'indicator_id': indicator_id,
                'signal_type': signal_type,
                'parameter': parameter,
                'operator': condition.get('operator', '='),
                'value': condition.get('value', 1)
            }
            
            # 处理参数
            parameters = self._standardize_parameters(condition, indicator_id)
            if parameters:
                standardized['parameters'] = parameters
            
            # 保留其他字段
            for key in ['period', 'weight', 'description']:
                if key in condition:
                    standardized[key] = condition[key]
            
            return standardized
            
        except Exception as e:
            logger.error(f"标准化指标条件失败: {e}")
            return condition  # 返回原条件
    
    def _extract_indicator_id(self, condition: Dict[str, Any]) -> str:
        """提取指标ID"""
        # 支持多种字段名
        for field in ['indicator_id', 'indicator', 'indicator_name']:
            if field in condition:
                return str(condition[field]).upper()
        
        return 'UNKNOWN'
    
    def _extract_signal_type(self, condition: Dict[str, Any]) -> str:
        """提取信号类型"""
        signal_type = condition.get('signal_type', 'BUY')
        return str(signal_type).upper()
    
    def _extract_parameter(self, condition: Dict[str, Any], indicator_id: str) -> str:
        """提取或生成参数名"""
        # 如果已有parameter字段且不为空，直接使用
        if condition.get('parameter'):
            return condition['parameter']
        
        # 根据指标和信号类型自动生成参数名
        signal_type = self._extract_signal_type(condition)
        return self._generate_parameter_name(indicator_id, signal_type)
    
    def _generate_parameter_name(self, indicator_id: str, signal_type: str) -> str:
        """生成参数名"""
        # 获取指标的可用信号
        signals = self.validator.get_available_signals(indicator_id)
        
        if signals:
            # 根据信号类型选择合适的信号
            if signal_type == 'BUY':
                # 优先选择买入相关的信号
                for signal in signals:
                    signal_name = signal.get('name', '')
                    if any(keyword in signal_name.lower() for keyword in ['buy', 'bullish', 'golden', 'oversold']):
                        return signal_name
            elif signal_type == 'SELL':
                # 优先选择卖出相关的信号
                for signal in signals:
                    signal_name = signal.get('name', '')
                    if any(keyword in signal_name.lower() for keyword in ['sell', 'bearish', 'death', 'overbought']):
                        return signal_name
            
            # 如果没有找到特定信号，返回第一个
            if signals:
                return signals[0].get('name', f'{indicator_id.lower()}_signal')
        
        # 默认参数名
        return f'{indicator_id.lower()}_{signal_type.lower()}_signal'
    
    def _standardize_parameters(self, condition: Dict[str, Any], indicator_id: str) -> Dict[str, Any]:
        """标准化参数"""
        # 获取原始参数
        original_params = condition.get('parameters', {})
        
        # 获取默认参数
        default_params = self.validator.get_default_parameters(indicator_id)
        
        # 合并参数
        standardized_params = default_params.copy()
        standardized_params.update(original_params)
        
        # 验证参数
        is_valid, errors = self.validator.validate_indicator_parameters(indicator_id, standardized_params)
        if not is_valid:
            logger.debug(f"参数验证失败 {indicator_id}: {'; '.join(errors)}")
            # 使用默认参数
            return default_params
        
        return standardized_params
    
    def validate_and_standardize(self, strategy_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证并标准化整个策略配置
        
        Args:
            strategy_config: 策略配置字典
            
        Returns:
            标准化后的策略配置
        """
        try:
            if 'strategy' not in strategy_config:
                raise ValueError("策略配置缺少strategy字段")
            
            strategy = strategy_config['strategy'].copy()
            
            # 标准化条件列表
            if 'conditions' in strategy and strategy['conditions']:
                standardized_conditions = []
                
                for condition in strategy['conditions']:
                    if isinstance(condition, dict) and self._is_indicator_condition(condition):
                        # 标准化指标条件
                        standardized_condition = self.standardize_indicator_condition(condition)
                        standardized_conditions.append(standardized_condition)
                    else:
                        # 保留逻辑操作符等其他条件
                        standardized_conditions.append(condition)
                
                strategy['conditions'] = standardized_conditions
            
            return {'strategy': strategy}
            
        except Exception as e:
            logger.error(f"标准化策略配置失败: {e}")
            return strategy_config
    
    def _is_indicator_condition(self, condition: Dict[str, Any]) -> bool:
        """判断是否为指标条件"""
        # 检查是否包含指标相关字段
        indicator_fields = ['indicator_id', 'indicator', 'indicator_name']
        return any(field in condition for field in indicator_fields)
    
    def convert_legacy_format(self, legacy_condition: Dict[str, Any]) -> Dict[str, Any]:
        """
        转换旧格式条件
        
        Args:
            legacy_condition: 旧格式条件
            
        Returns:
            新格式条件
        """
        # 处理各种旧格式
        converted = {}
        
        # 基本字段映射
        field_mapping = {
            'indicator': 'indicator_id',
            'signal': 'signal_type',
            'param': 'parameter',
            'op': 'operator',
            'val': 'value'
        }
        
        for old_field, new_field in field_mapping.items():
            if old_field in legacy_condition:
                converted[new_field] = legacy_condition[old_field]
        
        # 复制其他字段
        for key, value in legacy_condition.items():
            if key not in field_mapping:
                converted[key] = value
        
        # 确保必需字段存在
        if 'type' not in converted:
            converted['type'] = 'indicator'
        
        return self.standardize_indicator_condition(converted)
    
    def batch_standardize_conditions(self, conditions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量标准化条件列表
        
        Args:
            conditions: 条件列表
            
        Returns:
            标准化后的条件列表
        """
        standardized_conditions = []
        
        for condition in conditions:
            if isinstance(condition, dict) and self._is_indicator_condition(condition):
                standardized = self.standardize_indicator_condition(condition)
                standardized_conditions.append(standardized)
            else:
                standardized_conditions.append(condition)
        
        return standardized_conditions
    
    def get_standardization_summary(self, original_config: Dict[str, Any], 
                                  standardized_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        获取标准化摘要
        
        Args:
            original_config: 原始配置
            standardized_config: 标准化后配置
            
        Returns:
            标准化摘要
        """
        summary = {
            'original_conditions': 0,
            'standardized_conditions': 0,
            'indicator_conditions': 0,
            'logic_conditions': 0,
            'changes_made': []
        }
        
        try:
            # 统计原始条件
            if 'strategy' in original_config and 'conditions' in original_config['strategy']:
                summary['original_conditions'] = len(original_config['strategy']['conditions'])
            
            # 统计标准化后条件
            if 'strategy' in standardized_config and 'conditions' in standardized_config['strategy']:
                conditions = standardized_config['strategy']['conditions']
                summary['standardized_conditions'] = len(conditions)
                
                for condition in conditions:
                    if isinstance(condition, dict):
                        if self._is_indicator_condition(condition):
                            summary['indicator_conditions'] += 1
                        else:
                            summary['logic_conditions'] += 1
            
            # 检测变化
            if summary['original_conditions'] != summary['standardized_conditions']:
                summary['changes_made'].append('条件数量发生变化')
            
        except Exception as e:
            logger.error(f"生成标准化摘要失败: {e}")
        
        return summary
