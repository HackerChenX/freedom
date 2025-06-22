#!/usr/bin/env python3
"""
指标参数验证器

提供指标参数的验证功能，基于Schema定义进行参数类型、范围、必需性验证
"""

import yaml
import os
from typing import Dict, List, Tuple, Any, Optional
from utils.logger import get_logger

logger = get_logger(__name__)


class IndicatorParameterValidator:
    """指标参数验证器"""
    
    def __init__(self, schema_file: str = "config/indicator_parameter_schemas.yaml", silent_mode: bool = True):
        """
        初始化验证器

        Args:
            schema_file: Schema定义文件路径
            silent_mode: 静默模式，默认为True
        """
        self.schema_file = schema_file
        self.silent_mode = silent_mode
        self.schemas = self._load_schemas()
    
    def _load_schemas(self) -> Dict[str, Any]:
        """加载Schema定义"""
        try:
            if os.path.exists(self.schema_file):
                with open(self.schema_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                    return data.get('indicators', {})
            else:
                logger.debug(f"Schema文件不存在: {self.schema_file}")
                return {}
        except Exception as e:
            logger.error(f"加载Schema文件失败: {e}")
            return {}
    
    def get_indicator_schema(self, indicator_id: str) -> Optional[Dict[str, Any]]:
        """获取指标Schema定义"""
        return self.schemas.get(indicator_id)
    
    def get_default_parameters(self, indicator_id: str) -> Dict[str, Any]:
        """获取指标默认参数"""
        schema = self.get_indicator_schema(indicator_id)
        if not schema or 'parameters' not in schema:
            return {}
        
        defaults = {}
        for param_name, param_def in schema['parameters'].items():
            if 'default' in param_def:
                defaults[param_name] = param_def['default']
        
        return defaults
    
    def get_available_signals(self, indicator_id: str) -> List[Dict[str, Any]]:
        """获取指标可用信号"""
        schema = self.get_indicator_schema(indicator_id)
        if not schema or 'signals' not in schema:
            return []
        
        return schema['signals']
    
    def get_available_patterns(self, indicator_id: str) -> List[Dict[str, Any]]:
        """获取指标可用形态"""
        schema = self.get_indicator_schema(indicator_id)
        if not schema or 'patterns' not in schema:
            return []
        
        return schema['patterns']
    
    def validate_indicator_parameters(self, indicator_id: str, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        验证指标参数
        
        Args:
            indicator_id: 指标ID
            parameters: 参数字典
            
        Returns:
            (is_valid, error_messages)
        """
        schema = self.get_indicator_schema(indicator_id)
        if not schema:
            if not self.silent_mode:
                logger.debug(f"未找到指标 {indicator_id} 的Schema定义，跳过验证")
            return True, []  # 没有Schema定义时认为验证通过
        
        if 'parameters' not in schema:
            return True, []  # 没有参数定义，认为有效
        
        errors = []
        param_schemas = schema['parameters']
        
        # 验证每个参数
        for param_name, param_value in parameters.items():
            if param_name not in param_schemas:
                errors.append(f"未知参数: {param_name}")
                continue
            
            param_schema = param_schemas[param_name]
            param_errors = self._validate_parameter(param_name, param_value, param_schema)
            errors.extend(param_errors)
        
        # 检查必需参数
        for param_name, param_schema in param_schemas.items():
            if param_schema.get('required', False) and param_name not in parameters:
                errors.append(f"缺少必需参数: {param_name}")
        
        return len(errors) == 0, errors
    
    def _validate_parameter(self, param_name: str, param_value: Any, param_schema: Dict[str, Any]) -> List[str]:
        """验证单个参数"""
        errors = []
        
        # 类型验证
        expected_type = param_schema.get('type', 'string')
        if not self._validate_type(param_value, expected_type):
            errors.append(f"参数 {param_name} 类型错误，期望 {expected_type}，实际 {type(param_value).__name__}")
            return errors  # 类型错误时不继续验证
        
        # 范围验证
        if expected_type in ['integer', 'float']:
            if 'min' in param_schema and param_value < param_schema['min']:
                errors.append(f"参数 {param_name} 值 {param_value} 小于最小值 {param_schema['min']}")
            
            if 'max' in param_schema and param_value > param_schema['max']:
                errors.append(f"参数 {param_name} 值 {param_value} 大于最大值 {param_schema['max']}")
        
        # 枚举值验证
        if 'allowed_values' in param_schema:
            if param_value not in param_schema['allowed_values']:
                errors.append(f"参数 {param_name} 值 {param_value} 不在允许的值列表中: {param_schema['allowed_values']}")
        
        return errors
    
    def _validate_type(self, value: Any, expected_type: str) -> bool:
        """验证参数类型"""
        if expected_type == 'integer':
            return isinstance(value, int)
        elif expected_type == 'float':
            return isinstance(value, (int, float))
        elif expected_type == 'boolean':
            return isinstance(value, bool)
        elif expected_type == 'string':
            return isinstance(value, str)
        elif expected_type == 'array':
            return isinstance(value, list)
        else:
            return True  # 未知类型，默认通过
    
    def validate_signal_parameter(self, indicator_id: str, signal_name: str) -> bool:
        """验证信号参数是否有效"""
        signals = self.get_available_signals(indicator_id)
        signal_names = [s.get('name') for s in signals]
        return signal_name in signal_names
    
    def validate_pattern_id(self, indicator_id: str, pattern_id: str) -> bool:
        """验证形态ID是否有效"""
        patterns = self.get_available_patterns(indicator_id)
        pattern_ids = [p.get('id') for p in patterns]
        return pattern_id in pattern_ids
    
    def get_parameter_info(self, indicator_id: str, param_name: str) -> Optional[Dict[str, Any]]:
        """获取参数详细信息"""
        schema = self.get_indicator_schema(indicator_id)
        if not schema or 'parameters' not in schema:
            return None
        
        return schema['parameters'].get(param_name)
    
    def list_all_indicators(self) -> List[str]:
        """列出所有支持的指标"""
        return list(self.schemas.keys())
    
    def get_indicator_info(self, indicator_id: str) -> Optional[Dict[str, Any]]:
        """获取指标基本信息"""
        schema = self.get_indicator_schema(indicator_id)
        if not schema:
            return None
        
        return {
            'name': schema.get('name', ''),
            'description': schema.get('description', ''),
            'parameter_count': len(schema.get('parameters', {})),
            'signal_count': len(schema.get('signals', [])),
            'pattern_count': len(schema.get('patterns', []))
        }
