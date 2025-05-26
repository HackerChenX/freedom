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
from enums.period_types import PeriodType

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
            FileNotFoundError: 文件不存在
            ValueError: 文件格式不支持或解析失败
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"策略配置文件不存在: {file_path}")
        
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
                raise ValueError(f"不支持的配置文件格式: {file_ext}")
                
            return self.parse_strategy(config)
        except Exception as e:
            logger.error(f"解析策略配置文件失败: {e}")
            raise ValueError(f"解析策略配置文件失败: {e}")
            
    def parse_from_string(self, config_str: str, format_type: str = 'json') -> Dict[str, Any]:
        """
        从字符串中解析策略配置
        
        Args:
            config_str: 配置字符串
            format_type: 格式类型，支持'json'和'yaml'
            
        Returns:
            Dict[str, Any]: 解析后的策略执行计划
        """
        try:
            if format_type.lower() == 'json':
                config = json.loads(config_str)
            elif format_type.lower() in ['yml', 'yaml']:
                config = yaml.safe_load(config_str)
            else:
                raise ValueError(f"不支持的配置格式类型: {format_type}")
                
            return self.parse_strategy(config)
        except Exception as e:
            logger.error(f"解析策略配置字符串失败: {e}")
            raise ValueError(f"解析策略配置字符串失败: {e}")
    
    def parse_strategy(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        解析策略配置并返回执行计划
        
        Args:
            config: 策略配置字典
            
        Returns:
            Dict[str, Any]: 解析后的策略执行计划
        """
        # 检查配置格式
        if "strategy" not in config:
            raise ValueError("无效的策略配置: 缺少'strategy'节点")
            
        strategy = config["strategy"]
        
        # 检查必要字段
        required_fields = ["id", "name", "conditions"]
        for field in required_fields:
            if field not in strategy:
                raise ValueError(f"策略配置缺少必要字段: {field}")
        
        # 解析条件
        conditions = self._parse_conditions(strategy["conditions"])
        
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
        
    def _parse_conditions(self, conditions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        解析策略条件
        
        Args:
            conditions: 条件配置列表
            
        Returns:
            List[Dict[str, Any]]: 解析后的条件列表
        """
        parsed_conditions = []
        
        # 逻辑标记，用于支持多层嵌套条件
        logic_stack = ["AND"]  # 默认最外层为AND
        
        for condition in conditions:
            if "logic" in condition:
                # 逻辑运算符
                logic_op = condition["logic"].upper()
                if logic_op not in ["AND", "OR", "NOT"]:
                    raise ValueError(f"不支持的逻辑运算符: {logic_op}")
                
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
                    raise ValueError("条件分组不匹配: 多余的结束分组")
                
                parsed_conditions.append({
                    "type": "group_end"
                })
                
                # 弹出当前逻辑状态
                logic_stack.pop()
            else:
                # 指标条件
                if "indicator_id" not in condition:
                    raise ValueError("条件缺少 indicator_id 字段")
                if "period" not in condition:
                    raise ValueError("条件缺少 period 字段")
                
                indicator_id = condition["indicator_id"]
                period = condition["period"]
                
                # 验证周期类型
                try:
                    period_type = getattr(PeriodType, period)
                except AttributeError:
                    raise ValueError(f"不支持的周期类型: {period}")
                
                # 解析参数
                parameters = condition.get("parameters", {})
                
                # 创建指标实例
                indicator = self.indicator_factory.create(indicator_id, **parameters)
                if indicator is None:
                    raise ValueError(f"创建指标失败: {indicator_id}")
                
                parsed_conditions.append({
                    "type": "indicator",
                    "indicator_id": indicator_id,
                    "indicator": indicator,
                    "period": period,
                    "parameters": parameters,
                    "signal_type": condition.get("signal_type", "DEFAULT")
                })
                
        # 检查分组是否匹配
        if len(logic_stack) != 1:
            raise ValueError("条件分组不匹配: 缺少结束分组")
            
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
        
        # 支持的排序字段
        supported_fields = [
            "signal_strength", "market_cap", "price", "volume",
            "turnover_rate", "pe_ratio", "pb_ratio", "change_percent"
        ]
        
        for sort_item in sort:
            if "field" not in sort_item:
                raise ValueError("排序条件缺少 field 字段")
            if "direction" not in sort_item:
                raise ValueError("排序条件缺少 direction 字段")
                
            field = sort_item["field"]
            direction = sort_item["direction"].upper()
            
            if field not in supported_fields:
                logger.warning(f"未知的排序字段: {field}，将被忽略")
                continue
                
            if direction not in ["ASC", "DESC"]:
                raise ValueError(f"不支持的排序方向: {direction}")
                
            parsed_sort.append({
                "field": field,
                "direction": direction
            })
            
        return parsed_sort

    def validate_strategy(self, config: Dict[str, Any]) -> bool:
        """
        验证策略配置的有效性
        
        Args:
            config: 策略配置字典
            
        Returns:
            bool: 验证是否通过
            
        Raises:
            ValueError: 验证失败时抛出异常
        """
        try:
            self.parse_strategy(config)
            return True
        except Exception as e:
            logger.error(f"策略配置验证失败: {e}")
            raise ValueError(f"策略配置验证失败: {e}") 