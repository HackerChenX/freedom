#!/usr/bin/env python3
"""
策略格式兼容性转换器

统一新旧策略配置格式，解决技术债务问题
"""

import copy
from typing import Dict, List, Any, Optional
from utils.logger import get_logger

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
        try:
            # 提取参数
            params = condition.get("parameters", {})
            indicator_id = condition.get("indicator_id", "UNKNOWN")
            
            # 构建简化条件
            converted = {
                "type": "indicator",
                "indicator_id": indicator_id,
                "period": condition.get("period", "DAILY"),
                "operator": params.get("operator", ">"),
                "value": params.get("value", 0),
                "parameters": params
            }
            
            return converted
            
        except Exception as e:
            logger.warning(f"转换指标条件失败: {e}")
            return condition
    
    def _clean_conditions(self, conditions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """清理条件列表，移除独立的逻辑连接符"""
        cleaned = []
        
        for condition in conditions:
            # 跳过独立的逻辑连接符
            if "logic" in condition and len(condition) == 1:
                continue
            
            # 保留有实际内容的条件
            if "type" in condition or "indicator_id" in condition:
                cleaned.append(condition)
        
        return cleaned
    
    def convert_to_execution_format(self, strategy_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        转换为执行器可用的格式
        
        Args:
            strategy_config: 标准化的策略配置
            
        Returns:
            Dict[str, Any]: 执行器格式的策略配置
        """
        try:
            # 先标准化为v2.0格式
            normalized = self.normalize_to_v2(strategy_config)
            strategy = normalized["strategy"]
            
            # 转换为执行器期望的格式
            execution_format = {
                "strategy_id": strategy["strategy_id"],
                "name": strategy["name"],
                "description": strategy.get("description", ""),
                "conditions": strategy.get("conditions", []),
                "filters": strategy.get("filters", {}),
                "result_filters": strategy.get("result_filters", {"max_results": 50})
            }
            
            return execution_format
            
        except Exception as e:
            logger.error(f"转换为执行格式失败: {e}")
            raise ValueError(f"转换为执行格式失败: {e}")
    
    def validate_format(self, strategy_config: Dict[str, Any]) -> bool:
        """
        验证策略格式是否有效
        
        Args:
            strategy_config: 策略配置
            
        Returns:
            bool: 是否有效
        """
        try:
            # 尝试标准化
            normalized = self.normalize_to_v2(strategy_config)
            
            # 检查必要字段
            strategy = normalized.get("strategy", {})
            required_fields = ["strategy_id", "name", "conditions"]
            
            for field in required_fields:
                if field not in strategy:
                    logger.warning(f"策略配置缺少必要字段: {field}")
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"策略格式验证失败: {e}")
            return False


def create_format_converter() -> StrategyFormatConverter:
    """创建策略格式转换器实例"""
    return StrategyFormatConverter()


# 全局转换器实例
_converter_instance = None


def get_format_converter() -> StrategyFormatConverter:
    """获取全局策略格式转换器实例"""
    global _converter_instance
    if _converter_instance is None:
        _converter_instance = create_format_converter()
    return _converter_instance
