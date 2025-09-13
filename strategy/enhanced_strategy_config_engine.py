"""
增强策略配置引擎

支持通达信风格公式解析，提供灵活的策略配置和验证机制
遵循六层架构规范，实现高性能的策略配置管理
"""

import re
import json
import yaml
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

from utils.dependency_injection import get_logger
from utils.decorators import performance_monitor, exception_handler
from utils.unified_container import get_container
from enums.pattern_types import Candle_pattern_type
from enums.signal_strength import Signal_strength

logger = get_logger(__name__)


class OperatorType(Enum):
    """操作符类型枚举"""
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    GT = ">"
    LT = "<"
    GTE = ">="
    LTE = "<="
    EQ = "=="
    NEQ = "!="
    BETWEEN = "BETWEEN"
    IN = "IN"


class FunctionType(Enum):
    """函数类型枚举"""
    SUCCESS_RATE = "SUCCESS_RATE"
    PATTERN_HIT = "PATTERN_HIT"
    BACKTEST_SCORE = "BACKTEST_SCORE"
    TREND_STRENGTH = "TREND_STRENGTH"
    VOLUME_RATIO = "VOLUME_RATIO"


@dataclass
class StrategyCondition:
    """策略条件数据类"""
    indicator: str
    pattern: str
    period: str
    operator: OperatorType
    value: Union[float, int, str, List]
    weight: float = 1.0
    required: bool = True


@dataclass
class StrategyRule:
    """策略规则数据类"""
    rule_id: str
    name: str
    description: str
    conditions: List[StrategyCondition]
    logic_operator: OperatorType
    min_score: float = 60.0
    max_results: int = 50


@dataclass
class StrategyConfig:
    """策略配置数据类"""
    strategy_id: str
    name: str
    description: str
    version: str
    rules: List[StrategyRule]
    global_settings: Dict[str, Any]
    created_time: str
    updated_time: str


class EnhancedStrategyConfigEngine:
    """
    增强策略配置引擎
    
    支持通达信风格公式解析，提供灵活的策略配置管理
    """
    
    def __init__(self):
        """初始化策略配置引擎"""
        self.logger = logger
        container = get_container()
        try:
            self.pattern_registry = container.resolve("PatternRegistry")
        except:
            self.pattern_registry = None
        try:
            self.indicator_registry = container.resolve("IndicatorRegistry")
        except:
            self.indicator_registry = None
        
        # 通达信风格语法模式
        self.syntax_patterns = {
            'pattern_syntax': r'([A-Z_]+)\.([A-Z_]+)\.([A-Z_]+)',  # PATTERN.INDICATOR.PERIOD
            'comparison': r'([A-Z_]+(?:\.[A-Z_]+){2})\s*([><=!]+)\s*([0-9.]+)',
            'function_call': r'([A-Z_]+)\(([^)]*)\)',
            'logical_operator': r'\b(AND|OR|NOT)\b',
            'parentheses': r'\(([^)]+)\)'
        }
        
        # 支持的周期映射
        self.period_mapping = {
            'DAILY': '日线',
            'WEEKLY': '周线', 
            'MONTHLY': '月线',
            '30MIN': '30分钟',
            '15MIN': '15分钟',
            '5MIN': '5分钟',
            '1MIN': '1分钟'
        }
        
        # 支持的指标映射
        self.indicator_mapping = {
            'MACD': 'MACD',
            'RSI': 'RSI',
            'KDJ': 'KDJ',
            'BOLL': 'BOLL',
            'MA': 'MA',
            'VOL': 'VOL',
            'CCI': 'CCI',
            'WR': 'WR'
        }
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=2.0)
    def parse_formula(self, formula: str) -> List[StrategyCondition]:
        """
        解析通达信风格公式
        
        Args:
            formula: 通达信风格公式字符串
            
        Returns:
            List[StrategyCondition]: 解析后的策略条件列表
            
        Example:
            formula = "GOLDEN_CROSS.MACD.DAILY AND OVERSOLD.RSI.DAILY AND VOLUME_SURGE.VOL.DAILY"
        """
        self.logger.info(f"开始解析公式: {formula}")
        
        conditions = []
        
        # 预处理：移除多余空格，标准化格式
        formula = self._preprocess_formula(formula)
        
        # 分解复合条件
        condition_parts = self._split_conditions(formula)
        
        for part in condition_parts:
            condition = self._parse_single_condition(part.strip())
            if condition:
                conditions.append(condition)
        
        self.logger.info(f"成功解析 {len(conditions)} 个条件")
        return conditions
    
    def _preprocess_formula(self, formula: str) -> str:
        """预处理公式字符串"""
        # 移除多余空格
        formula = re.sub(r'\s+', ' ', formula.strip())
        
        # 标准化操作符
        formula = formula.replace(' AND ', ' AND ')
        formula = formula.replace(' OR ', ' OR ')
        formula = formula.replace(' NOT ', ' NOT ')
        
        return formula
    
    def _split_conditions(self, formula: str) -> List[str]:
        """分解复合条件"""
        # 简化处理：按 AND/OR 分割
        # 实际实现中需要考虑括号优先级
        parts = []
        current_part = ""
        
        tokens = formula.split()
        for token in tokens:
            if token in ['AND', 'OR']:
                if current_part.strip():
                    parts.append(current_part.strip())
                    current_part = ""
            else:
                current_part += " " + token
        
        if current_part.strip():
            parts.append(current_part.strip())
        
        return parts
    
    def _parse_single_condition(self, condition_str: str) -> Optional[StrategyCondition]:
        """解析单个条件"""
        try:
            # 匹配模式语法：PATTERN.INDICATOR.PERIOD
            pattern_match = re.match(self.syntax_patterns['pattern_syntax'], condition_str)
            
            if pattern_match:
                pattern_name = pattern_match.group(1)
                indicator_name = pattern_match.group(2)
                period_name = pattern_match.group(3)
                
                # 验证指标和周期
                if not self._validate_indicator_period(indicator_name, period_name):
                    self.logger.warning(f"无效的指标或周期: {indicator_name}.{period_name}")
                    return None
                
                return StrategyCondition(
                    indicator=self.indicator_mapping.get(indicator_name, indicator_name),
                    pattern=pattern_name.lower(),
                    period=self.period_mapping.get(period_name, period_name),
                    operator=OperatorType.EQ,
                    value=True,
                    weight=1.0,
                    required=True
                )
            
            self.logger.warning(f"无法解析条件: {condition_str}")
            return None
            
        except Exception as e:
            self.logger.error(f"解析条件时出错: {e}")
            return None
    
    def _validate_indicator_period(self, indicator: str, period: str) -> bool:
        """验证指标和周期的有效性"""
        return (indicator in self.indicator_mapping and 
                period in self.period_mapping)
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=1.0)
    def create_strategy_config(self, config_data: Dict[str, Any]) -> StrategyConfig:
        """
        创建策略配置
        
        Args:
            config_data: 策略配置数据
            
        Returns:
            StrategyConfig: 策略配置对象
        """
        self.logger.info(f"创建策略配置: {config_data.get('name', 'Unknown')}")
        
        # 验证必需字段
        required_fields = ['name', 'description', 'rules']
        for field in required_fields:
            if field not in config_data:
                raise ValueError(f"缺少必需字段: {field}")
        
        # 生成策略ID和时间戳
        strategy_id = config_data.get('strategy_id', self._generate_strategy_id())
        current_time = datetime.now().isoformat()
        
        # 解析规则
        rules = []
        for rule_data in config_data['rules']:
            rule = self._create_strategy_rule(rule_data)
            if rule:
                rules.append(rule)
        
        return StrategyConfig(
            strategy_id=strategy_id,
            name=config_data['name'],
            description=config_data['description'],
            version=config_data.get('version', '1.0.0'),
            rules=rules,
            global_settings=config_data.get('global_settings', {}),
            created_time=current_time,
            updated_time=current_time
        )
    
    def _create_strategy_rule(self, rule_data: Dict[str, Any]) -> Optional[StrategyRule]:
        """创建策略规则"""
        try:
            rule_id = rule_data.get('rule_id', self._generate_rule_id())
            
            # 解析条件
            conditions = []
            if 'formula' in rule_data:
                conditions = self.parse_formula(rule_data['formula'])
            elif 'conditions' in rule_data:
                for cond_data in rule_data['conditions']:
                    condition = self._create_condition_from_data(cond_data)
                    if condition:
                        conditions.append(condition)
            
            return StrategyRule(
                rule_id=rule_id,
                name=rule_data.get('name', f'Rule_{rule_id}'),
                description=rule_data.get('description', ''),
                conditions=conditions,
                logic_operator=OperatorType(rule_data.get('logic_operator', 'AND')),
                min_score=rule_data.get('min_score', 60.0),
                max_results=rule_data.get('max_results', 50)
            )
            
        except Exception as e:
            self.logger.error(f"创建策略规则时出错: {e}")
            return None
    
    def _create_condition_from_data(self, cond_data: Dict[str, Any]) -> Optional[StrategyCondition]:
        """从数据创建条件"""
        try:
            return StrategyCondition(
                indicator=cond_data['indicator'],
                pattern=cond_data['pattern'],
                period=cond_data['period'],
                operator=OperatorType(cond_data.get('operator', '==')),
                value=cond_data['value'],
                weight=cond_data.get('weight', 1.0),
                required=cond_data.get('required', True)
            )
        except Exception as e:
            self.logger.error(f"创建条件时出错: {e}")
            return None
    
    def _generate_strategy_id(self) -> str:
        """生成策略ID"""
        return f"strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def _generate_rule_id(self) -> str:
        """生成规则ID"""
        return f"rule_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    @exception_handler(reraise=True)
    @performance_monitor(threshold=1.0)
    def validate_strategy_config(self, config: StrategyConfig) -> Dict[str, Any]:
        """
        验证策略配置的有效性

        Args:
            config: 策略配置对象

        Returns:
            Dict[str, Any]: 验证结果
        """
        self.logger.info(f"验证策略配置: {config.name}")

        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'score': 100.0
        }

        # 验证基本信息
        if not config.name or len(config.name.strip()) == 0:
            validation_result['errors'].append("策略名称不能为空")
            validation_result['is_valid'] = False

        if not config.rules or len(config.rules) == 0:
            validation_result['errors'].append("策略必须包含至少一个规则")
            validation_result['is_valid'] = False

        # 验证规则
        for rule in config.rules:
            rule_validation = self._validate_strategy_rule(rule)
            if not rule_validation['is_valid']:
                validation_result['errors'].extend(rule_validation['errors'])
                validation_result['is_valid'] = False
            validation_result['warnings'].extend(rule_validation['warnings'])

        # 计算验证分数
        if validation_result['is_valid']:
            penalty = len(validation_result['warnings']) * 5
            validation_result['score'] = max(60.0, 100.0 - penalty)
        else:
            validation_result['score'] = 0.0

        self.logger.info(f"验证完成，得分: {validation_result['score']}")
        return validation_result

    def _validate_strategy_rule(self, rule: StrategyRule) -> Dict[str, Any]:
        """验证策略规则"""
        result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }

        # 验证条件
        if not rule.conditions or len(rule.conditions) == 0:
            result['errors'].append(f"规则 {rule.name} 必须包含至少一个条件")
            result['is_valid'] = False

        # 验证每个条件
        for condition in rule.conditions:
            condition_validation = self._validate_condition(condition)
            if not condition_validation['is_valid']:
                result['errors'].extend(condition_validation['errors'])
                result['is_valid'] = False
            result['warnings'].extend(condition_validation['warnings'])

        return result

    def _validate_condition(self, condition: StrategyCondition) -> Dict[str, Any]:
        """验证策略条件"""
        result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }

        # 验证指标是否存在
        if condition.indicator not in self.indicator_mapping.values():
            result['warnings'].append(f"指标 {condition.indicator} 可能不被支持")

        # 验证周期是否存在
        if condition.period not in self.period_mapping.values():
            result['warnings'].append(f"周期 {condition.period} 可能不被支持")

        # 验证权重
        if condition.weight <= 0:
            result['errors'].append("条件权重必须大于0")
            result['is_valid'] = False

        return result

    @exception_handler(reraise=True)
    @performance_monitor(threshold=1.0)
    def export_strategy_config(self, config: StrategyConfig, format_type: str = 'json') -> str:
        """
        导出策略配置

        Args:
            config: 策略配置对象
            format_type: 导出格式 ('json', 'yaml')

        Returns:
            str: 导出的配置字符串
        """
        self.logger.info(f"导出策略配置: {config.name}, 格式: {format_type}")

        # 转换为字典
        config_dict = asdict(config)

        # 处理枚举类型
        config_dict = self._serialize_enums(config_dict)

        if format_type.lower() == 'json':
            return json.dumps(config_dict, indent=2, ensure_ascii=False)
        elif format_type.lower() == 'yaml':
            return yaml.dump(config_dict, default_flow_style=False, allow_unicode=True)
        else:
            raise ValueError(f"不支持的导出格式: {format_type}")

    def _serialize_enums(self, obj: Any) -> Any:
        """序列化枚举类型"""
        if isinstance(obj, dict):
            return {k: self._serialize_enums(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_enums(item) for item in obj]
        elif isinstance(obj, Enum):
            return obj.value
        else:
            return obj

    @exception_handler(reraise=True)
    @performance_monitor(threshold=1.0)
    def import_strategy_config(self, config_str: str, format_type: str = 'json') -> StrategyConfig:
        """
        导入策略配置

        Args:
            config_str: 配置字符串
            format_type: 配置格式 ('json', 'yaml')

        Returns:
            StrategyConfig: 策略配置对象
        """
        self.logger.info(f"导入策略配置，格式: {format_type}")

        if format_type.lower() == 'json':
            config_data = json.loads(config_str)
        elif format_type.lower() == 'yaml':
            config_data = yaml.safe_load(config_str)
        else:
            raise ValueError(f"不支持的导入格式: {format_type}")

        return self.create_strategy_config(config_data)

    @exception_handler(reraise=False, default_return=[])
    @performance_monitor(threshold=2.0)
    def get_available_patterns(self, indicator: str) -> List[Dict[str, Any]]:
        """
        获取指定指标的可用形态

        Args:
            indicator: 指标名称

        Returns:
            List[Dict[str, Any]]: 可用形态列表
        """
        try:
            if self.pattern_registry:
                patterns = self.pattern_registry.get_patterns_by_indicator(indicator)
                return [
                    {
                        'name': pattern['name'],
                        'description': pattern.get('description', ''),
                        'type': pattern.get('type', ''),
                        'polarity': pattern.get('polarity', 'NEUTRAL')
                    }
                    for pattern in patterns
                ]
            return []
        except Exception as e:
            self.logger.error(f"获取形态列表时出错: {e}")
            return []

    @exception_handler(reraise=False, default_return=[])
    @performance_monitor(threshold=1.0)
    def get_supported_indicators(self) -> List[Dict[str, Any]]:
        """
        获取支持的指标列表

        Returns:
            List[Dict[str, Any]]: 支持的指标列表
        """
        try:
            indicators = []
            for key, value in self.indicator_mapping.items():
                indicators.append({
                    'key': key,
                    'name': value,
                    'patterns': self.get_available_patterns(value)
                })
            return indicators
        except Exception as e:
            self.logger.error(f"获取指标列表时出错: {e}")
            return []

    @exception_handler(reraise=False, default_return=[])
    @performance_monitor(threshold=1.0)
    def get_supported_periods(self) -> List[Dict[str, Any]]:
        """
        获取支持的周期列表

        Returns:
            List[Dict[str, Any]]: 支持的周期列表
        """
        try:
            periods = []
            for key, value in self.period_mapping.items():
                periods.append({
                    'key': key,
                    'name': value,
                    'description': f"{value}数据"
                })
            return periods
        except Exception as e:
            self.logger.error(f"获取周期列表时出错: {e}")
            return []
