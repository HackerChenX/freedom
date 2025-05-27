"""
策略解析器模块

负责解析策略配置并转换为系统可执行的格式
"""

from typing import Dict, List, Any, Optional, Union
import json
import os
import yaml
from datetime import datetime

from indicators.factory import IndicatorFactory
from utils.logger import get_logger
from enums.period import Period
from utils.exceptions import (
    StrategyParseError, 
    StrategyValidationError, 
    ConfigFileError,
    IndicatorNotFoundError,
    IndicatorParameterError
)

logger = get_logger(__name__)


class StrategyParser:
    """
    策略解析器，负责解析策略配置并构建策略执行计划
    """
    
    def __init__(self):
        """初始化策略解析器"""
        self.indicator_factory = IndicatorFactory
        
    def parse_from_file(self, file_path: str) -> Dict[str, Any]:
        """
        从文件中解析策略配置
        
        Args:
            file_path: 配置文件路径
            
        Returns:
            Dict[str, Any]: 解析后的策略执行计划
        
        Raises:
            ConfigFileError: 文件不存在或格式不支持
            StrategyParseError: 解析失败
        """
        if not os.path.exists(file_path):
            raise ConfigFileError(f"策略配置文件不存在: {file_path}")
        
        # 根据文件扩展名确定解析方式
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            elif file_ext in ['.yml', '.yaml']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
            else:
                raise ConfigFileError(f"不支持的配置文件格式: {file_ext}", 
                                     {"supported_formats": [".json", ".yml", ".yaml"]})
                
            return self.parse_strategy(config)
        except json.JSONDecodeError as e:
            raise StrategyParseError(f"JSON解析错误: {str(e)}", {"file": file_path, "position": f"行 {e.lineno}, 列 {e.colno}"})
        except yaml.YAMLError as e:
            raise StrategyParseError(f"YAML解析错误: {str(e)}", {"file": file_path})
        except Exception as e:
            logger.error(f"解析策略配置文件失败: {e}")
            raise StrategyParseError(f"解析策略配置文件失败: {str(e)}", {"file": file_path})
            
    def parse_from_string(self, config_str: str, format_type: str = 'json') -> Dict[str, Any]:
        """
        从字符串中解析策略配置
        
        Args:
            config_str: 配置字符串
            format_type: 格式类型，支持'json'和'yaml'
            
        Returns:
            Dict[str, Any]: 解析后的策略执行计划
            
        Raises:
            StrategyParseError: 解析失败
        """
        try:
            if format_type.lower() == 'json':
                config = json.loads(config_str)
            elif format_type.lower() in ['yml', 'yaml']:
                config = yaml.safe_load(config_str)
            else:
                raise StrategyParseError(f"不支持的配置格式类型: {format_type}", 
                                        {"supported_formats": ["json", "yaml", "yml"]})
                
            return self.parse_strategy(config)
        except json.JSONDecodeError as e:
            raise StrategyParseError(f"JSON解析错误: {str(e)}", {"position": f"行 {e.lineno}, 列 {e.colno}"})
        except yaml.YAMLError as e:
            raise StrategyParseError(f"YAML解析错误: {str(e)}")
        except Exception as e:
            logger.error(f"解析策略配置字符串失败: {e}")
            raise StrategyParseError(f"解析策略配置字符串失败: {str(e)}")
    
    def parse_strategy(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        解析策略配置并返回执行计划
        
        Args:
            config: 策略配置字典
            
        Returns:
            Dict[str, Any]: 解析后的策略执行计划
            
        Raises:
            StrategyValidationError: 配置验证失败
            StrategyParseError: 解析过程中发生错误
        """
        # 检查配置格式
        if "strategy" not in config:
            raise StrategyValidationError("无效的策略配置: 缺少'strategy'节点")
            
        strategy = config["strategy"]
        logger.debug(f"策略配置: {strategy}")
        
        # 检查必要字段
        required_fields = ["id", "name", "conditions"]
        missing_fields = [field for field in required_fields if field not in strategy]
        if missing_fields:
            raise StrategyValidationError(
                f"策略配置缺少必要字段: {', '.join(missing_fields)}", 
                {"required_fields": required_fields, "strategy_id": strategy.get("id", "unknown")}
            )
        
        try:
            # 解析条件
            logger.debug(f"条件配置: {strategy.get('conditions')}")
            conditions = self._parse_conditions(strategy["conditions"])
            logger.debug(f"解析后的条件: {conditions}")
            
            # 解析过滤器
            filters = self._parse_filters(strategy.get("filters", {}))
            
            # 解析排序
            sort = self._parse_sort(strategy.get("sort", []))
            
            # 创建执行计划
            execution_plan = {
                "strategy_id": strategy["id"],
                "name": strategy["name"],
                "description": strategy.get("description", ""),
                "version": strategy.get("version", "1.0"),
                "author": strategy.get("author", "system"),
                "create_time": strategy.get("create_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                "update_time": strategy.get("update_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                "conditions": conditions,
                "filters": filters,
                "sort": sort
            }
            
            return execution_plan
        except Exception as e:
            if isinstance(e, (StrategyValidationError, StrategyParseError)):
                raise
            logger.error(f"解析策略时发生错误: {e}")
            raise StrategyParseError(f"解析策略时发生错误: {str(e)}", {"strategy_id": strategy.get("id", "unknown")})
        
    def _parse_conditions(self, conditions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        解析策略条件
        
        Args:
            conditions: 条件配置列表
            
        Returns:
            List[Dict[str, Any]]: 解析后的条件列表
            
        Raises:
            StrategyValidationError: 条件配置验证失败
        """
        parsed_conditions = []
        
        # 打印条件列表，帮助调试
        logger.debug(f"条件列表: {conditions}")
        
        # 预处理：检查OR逻辑的情况，进行条件分组
        # 如果第一个条件后面紧跟着'OR'逻辑，需要特殊处理
        has_or_logic = False
        for i, condition in enumerate(conditions):
            if i > 0 and "logic" in condition and condition["logic"].upper() == "OR":
                has_or_logic = True
                break
        
        # 如果存在OR逻辑，需要调整条件顺序和逻辑顺序
        if has_or_logic:
            logger.debug("检测到OR逻辑，调整条件结构")
            # 第一阶段：将所有指标条件放入parsed_conditions
            indicator_conditions = []
            for condition in conditions:
                if "logic" not in condition:
                    # 这是一个指标条件
                    if "indicator_id" not in condition:
                        raise StrategyValidationError(
                            "条件缺少 indicator_id 字段", 
                            {"condition": condition}
                        )
                    indicator_conditions.append(condition)
            
            # 如果只有指标条件，不需要特殊处理
            if len(indicator_conditions) <= 1:
                logger.debug("只有一个指标条件，无需处理OR逻辑")
            else:
                # 第二阶段：生成OR关系的条件组
                # 先添加第一个指标条件
                parsed_conditions.append(indicator_conditions[0])
                
                # 添加后续指标条件，每个前面加OR
                for i in range(1, len(indicator_conditions)):
                    # 添加OR逻辑
                    parsed_conditions.append({"type": "logic", "value": "OR"})
                    # 添加指标条件
                    parsed_conditions.append(indicator_conditions[i])
                
                logger.debug(f"调整后的条件结构: {parsed_conditions}")
                return parsed_conditions
        
        # 逻辑标记，用于支持多层嵌套条件
        logic_stack = ["AND"]  # 默认最外层为AND
        
        for condition_index, condition in enumerate(conditions):
            if "logic" in condition:
                # 逻辑运算符
                logic_op = condition["logic"].upper()
                if logic_op not in ["AND", "OR", "NOT"]:
                    raise StrategyValidationError(
                        f"不支持的逻辑运算符: {logic_op}", 
                        {"supported_operators": ["AND", "OR", "NOT"], "condition_index": condition_index}
                    )
                
                parsed_conditions.append({
                    "type": "logic",
                    "value": logic_op
                })
                
                # 更新当前逻辑状态
                if logic_op != "NOT":  # NOT是一元运算符，不改变逻辑堆栈
                    logic_stack[-1] = logic_op
            elif "group" in condition:
                # 条件分组开始
                parsed_conditions.append({
                    "type": "group_start",
                    "logic": condition.get("logic", "AND")
                })
                
                # 压入新的逻辑状态
                logic_stack.append(condition.get("logic", "AND"))
            elif "end_group" in condition:
                # 条件分组结束
                if len(logic_stack) <= 1:
                    raise StrategyValidationError(
                        "条件分组不匹配: 多余的结束分组", 
                        {"condition_index": condition_index}
                    )
                
                parsed_conditions.append({
                    "type": "group_end"
                })
                
                # 弹出当前逻辑状态
                logic_stack.pop()
            else:
                # 指标条件
                if "indicator_id" not in condition:
                    raise StrategyValidationError(
                        "条件缺少 indicator_id 字段", 
                        {"condition_index": condition_index}
                    )
                if "period" not in condition:
                    raise StrategyValidationError(
                        "条件缺少 period 字段", 
                        {"condition_index": condition_index, "indicator_id": condition.get("indicator_id", "unknown")}
                    )
                
                indicator_id = condition["indicator_id"]
                period_str = condition["period"]
                
                # 验证周期类型
                try:
                    period = Period.from_string(period_str)
                except ValueError:
                    raise StrategyValidationError(
                        f"不支持的周期类型: {period_str}", 
                        {"supported_periods": [p.value for p in Period], "condition_index": condition_index}
                    )
                
                # 获取参数
                parameters = condition.get("parameters", {})
                
                # 验证指标是否存在
                try:
                    indicator = self.indicator_factory.create(indicator_id, **parameters)
                    if indicator is None:
                        raise StrategyValidationError(
                            f"指标不存在或创建失败: {indicator_id}", 
                            {"condition_index": condition_index}
                        )
                except Exception as e:
                    if isinstance(e, StrategyValidationError):
                        raise
                    raise StrategyValidationError(
                        f"创建指标实例失败: {indicator_id}, 错误: {str(e)}", 
                        {"condition_index": condition_index, "parameters": parameters}
                    )
                
                # 获取信号类型
                signal_type = condition.get("signal_type", "BUY")
                if signal_type not in ["BUY", "SELL", "OBSERVE"]:
                    raise StrategyValidationError(
                        f"不支持的信号类型: {signal_type}", 
                        {"supported_signal_types": ["BUY", "SELL", "OBSERVE"], "condition_index": condition_index}
                    )
                
                # 添加解析后的条件
                parsed_conditions.append({
                    "type": "indicator",
                    "indicator_id": indicator_id,
                    "period": period.value,
                    "parameters": parameters,
                    "signal_type": signal_type,
                    "required": condition.get("required", True),
                    "weight": condition.get("weight", 1.0)
                })
        
        return parsed_conditions
        
    def _parse_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        解析过滤条件
        
        Args:
            filters: 过滤条件配置
            
        Returns:
            Dict[str, Any]: 解析后的过滤条件
        """
        parsed_filters = {}
        
        # 支持的过滤器类型
        supported_filters = [
            "market", "industry", "market_cap", "price",
            "volume", "turnover_rate", "pe_ratio", "pb_ratio"
        ]
        
        for key, value in filters.items():
            if key not in supported_filters:
                logger.warning(f"未知的过滤条件类型: {key}，将被忽略")
                continue
                
            parsed_filters[key] = value
            
        return parsed_filters
        
    def _parse_sort(self, sort: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        解析排序条件
        
        Args:
            sort: 排序条件配置
            
        Returns:
            List[Dict[str, Any]]: 解析后的排序条件
        """
        parsed_sort = []
        
        for sort_item in sort:
            if "field" not in sort_item:
                logger.warning(f"排序项缺少field字段: {sort_item}，将被忽略")
                continue
                
            direction = sort_item.get("direction", "DESC").upper()
            if direction not in ["ASC", "DESC"]:
                logger.warning(f"排序方向无效: {direction}，默认使用DESC")
                direction = "DESC"
                
            parsed_sort.append({
                "field": sort_item["field"],
                "direction": direction
            })
            
        return parsed_sort
    
    def validate_strategy(self, config: Dict[str, Any]) -> bool:
        """
        验证策略配置
        
        Args:
            config: 策略配置
            
        Returns:
            bool: 是否验证通过
            
        Raises:
            StrategyValidationError: 验证失败
        """
        try:
            self.parse_strategy(config)
            return True
        except Exception as e:
            raise StrategyValidationError(
                f"策略配置验证失败: {str(e)}", 
                {"config": config.get("strategy", {}).get("id", "unknown")}
            ) 