from strategy.unified_base_strategy import UnifiedBaseStrategy
#!/usr/bin/env python3
"""
策略格式兼容性转换器

统一新旧策略配置格式，解决技术债务问题
"""

import copy
from typing import Dict, List, Any, Optional
from utils.logger import get_logger
from utils.dependency_injection import get_container

logger = get_logger(__name__)


class StrategyFormatConverter:
    """策略格式转换器，统一新旧策略配置格式"""
    
    def __init__(self):
        """初始化格式转换器"""
        self.supported_formats = ["v1.0", "v2.0", "legacy"]
        logger.info("策略格式转换器初始化完成")
    
    def detect_format_version(self, strategy_config: Dict[str, Any]) -> str:
        """
        检测策略配置格式版本
        
        Args:
            strategy_config: 策略配置字典
            
        Returns:
            str: 格式版本 ("v1.0", "v2.0", "legacy")
        """
        try:
            # 检查是否有strategy包装
            if "strategy" in strategy_config:
                strategy = strategy_config["strategy"]
                
                # 检查ID字段
                if "strategy_id" in strategy:
                    return "v2.0"  # 新格式
                elif "id" in strategy:
                    return "v1.0"  # 标准格式
                else:
                    return "legacy"  # 遗留格式
            else:
                # 直接的策略配置
                if "strategy_id" in strategy_config:
                    return "v2.0"
                elif "id" in strategy_config:
                    return "v1.0"
                else:
                    return "legacy"
                    
        except Exception as e:
            logger.warning(f"检测策略格式版本失败: {e}")
            return "legacy"
    
    def normalize_to_v2(self, strategy_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        将策略配置标准化为v2.0格式
        
        Args:
            strategy_config: 原始策略配置
            
        Returns:
            Dict[str, Any]: 标准化后的v2.0格式配置
        """
        try:
            format_version = self.detect_format_version(strategy_config)
            logger.debug(f"检测到策略格式版本: {format_version}")
            
            if format_version == "v2.0":
                return self._ensure_v2_completeness(strategy_config)
            elif format_version == "v1.0":
                return self._convert_v1_to_v2(strategy_config)
            else:
                return self._convert_legacy_to_v2(strategy_config)
                
        except Exception as e:
            logger.error(f"策略格式标准化失败: {e}")
            raise ValueError(f"策略格式标准化失败: {e}")
    
    def _ensure_v2_completeness(self, strategy_config: Dict[str, Any]) -> Dict[str, Any]:
        """确保v2.0格式的完整性"""
        normalized = copy.deepcopy(strategy_config)
        
        # 确保有strategy包装
        if "strategy" not in normalized:
            normalized = {"strategy": normalized}
        
        strategy = normalized["strategy"]
        
        # 确保必要字段存在
        if "strategy_id" not in strategy and "id" in strategy:
            strategy["strategy_id"] = strategy["id"]
        
        # 清理条件中的逻辑连接符
        if "conditions" in strategy:
            strategy["conditions"] = self._clean_conditions(strategy["conditions"])
        
        # 确保result_filters存在
        if "result_filters" not in strategy:
            strategy["result_filters"] = {"max_results": 50}
        
        return normalized
    
    def _convert_v1_to_v2(self, strategy_config: Dict[str, Any]) -> Dict[str, Any]:
        """将v1.0格式转换为v2.0格式"""
        normalized = copy.deepcopy(strategy_config)
        
        # 确保有strategy包装
        if "strategy" not in normalized:
            normalized = {"strategy": normalized}
        
        strategy = normalized["strategy"]
        
        # 转换ID字段
        if "id" in strategy:
            strategy["strategy_id"] = strategy["id"]
        
        # 转换条件格式
        if "conditions" in strategy:
            strategy["conditions"] = self._convert_v1_conditions(strategy["conditions"])
        
        # 添加默认的result_filters
        if "result_filters" not in strategy:
            strategy["result_filters"] = {"max_results": 50}
        
        return normalized
    
    def _convert_legacy_to_v2(self, strategy_config: Dict[str, Any]) -> Dict[str, Any]:
        """将遗留格式转换为v2.0格式"""
        normalized = {
            "strategy": {
                "strategy_id": strategy_config.get("name", "LEGACY_STRATEGY"),
                "name": strategy_config.get("name", "遗留策略"),
                "description": strategy_config.get("description", "从遗留格式转换的策略"),
                "version": "1.0",
                "author": "system",
                "conditions": [],
                "filters": strategy_config.get("filters", {}),
                "result_filters": {"max_results": 50}
            }
        }
        
        # 转换条件
        if "conditions" in strategy_config:
            normalized["strategy"]["conditions"] = self._convert_legacy_conditions(
                strategy_config["conditions"]
            )
        
        return normalized
    
    def _convert_v1_conditions(self, conditions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """转换v1.0条件格式"""
        converted = []
        
        for condition in conditions:
            # 跳过纯逻辑连接符
            if "logic" in condition and len(condition) == 1:
                continue
            
            # 转换复杂指标条件为简单格式
            if "indicator_id" in condition:
                converted_condition = self._convert_indicator_condition(condition)
                converted.append(converted_condition)
            else:
                # 保持简单条件不变
                converted.append(condition)
        
        return converted
    
    def _convert_legacy_conditions(self, conditions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """转换遗留条件格式"""
        converted = []
        
        for condition in conditions:
            # 尝试转换为标准格式
            if "pattern_id" in condition:
                # 形态条件转换为指标条件
                converted_condition = {
                    "type": "indicator",
                    "indicator_id": condition.get("indicator_id", "UNKNOWN"),
                    "operator": ">",
                    "value": condition.get("score_threshold", 60),
                    "parameters": {
                        "pattern": condition.get("pattern_id"),
                        "min_strength": condition.get("min_strength", 0.6)
                    }
                }
                converted.append(converted_condition)
            else:
                # 保持原有条件
                converted.append(condition)
        
        return converted
    
    def _convert_indicator_condition(self, condition: Dict[str, Any]) -> Dict[str, Any]:
        """转换指标条件为简单格式"""
        return {
            "type": "indicator",
            "indicator_id": condition.get("indicator_id"),
            "operator": condition.get("operator", ">"),
            "value": condition.get("value", 0),
            "parameters": condition.get("parameters", {})
        }
    
    def _clean_conditions(self, conditions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """清理条件中的逻辑连接符"""
        cleaned = []
        for condition in conditions:
            # 跳过纯逻辑连接符
            if not ("logic" in condition and len(condition) == 1):
                cleaned.append(condition)
        return cleaned
    
    def convert_to_execution_format(self, strategy_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        转换为执行格式
        
        Args:
            strategy_config: 策略配置
            
        Returns:
            Dict[str, Any]: 执行格式配置
        """
        # 先标准化为v2.0格式
        normalized = self.normalize_to_v2(strategy_config)
        
        # 提取执行所需的关键信息
        strategy = normalized["strategy"]
        execution_config = {
            "strategy_id": strategy.get("strategy_id"),
            "name": strategy.get("name"),
            "conditions": strategy.get("conditions", []),
            "filters": strategy.get("filters", {}),
            "result_filters": strategy.get("result_filters", {"max_results": 50}),
            "parameters": strategy.get("parameters", {})
        }
        
        return execution_config
    
    def validate_format(self, strategy_config: Dict[str, Any]) -> bool:
        """
        验证策略配置格式
        
        Args:
            strategy_config: 策略配置
            
        Returns:
            bool: 是否有效
        """
        try:
            normalized = self.normalize_to_v2(strategy_config)
            strategy = normalized.get("strategy", {})
            
            # 检查必要字段
            required_fields = ["strategy_id", "name"]
            for field in required_fields:
                if field not in strategy:
                    logger.error(f"策略配置缺少必要字段: {field}")
                    return False
            
            # 检查条件格式
            conditions = strategy.get("conditions", [])
            for condition in conditions:
                if not isinstance(condition, dict):
                    logger.error(f"无效的条件格式: {condition}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"验证策略配置格式失败: {e}")
            return False


# ===== 依赖注入和兼容性接口 =====

def create_format_converter() -> StrategyFormatConverter:
    """创建格式转换器实例（兼容性方法）"""
    return StrategyFormatConverter()


def get_format_converter() -> StrategyFormatConverter:
    """
    获取格式转换器实例（依赖注入方式）
    
    Returns:
        StrategyFormatConverter: 格式转换器实例
    """
    try:
        container = get_container()
        return container.resolve(StrategyFormatConverter)
    except Exception as e:
        logger.warning(f"从依赖注入容器获取StrategyFormatConverter失败，创建新实例: {e}")
        return StrategyFormatConverter()


# 注册到依赖注入容器
try:
    container = get_container()
    if not container.is_registered(StrategyFormatConverter):
        container.register_singleton(StrategyFormatConverter, StrategyFormatConverter)
        logger.info("StrategyFormatConverter已注册到依赖注入容器")
except Exception as e:
    logger.warning(f"注册StrategyFormatConverter到依赖注入容器失败: {e}")
