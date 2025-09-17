#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
预警配置管理器

提供动态配置预警规则的功能，支持：
1. 预警规则的动态创建、修改、删除
2. 配置文件的加载和保存
3. 配置模板管理
4. 配置验证和校验
"""

import os
import json
import yaml
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path

from utils.logger import get_logger
from utils.exception_handler import exception_handler
from utils.performance_monitor import performance_monitor
from enums.signal_types import SignalType
from db.sql_manager import SQLManager, QueryType

logger = get_logger(__name__)


@dataclass
class AlertRuleConfig:
    """预警规则配置"""
    id: str
    name: str
    description: str
    indicators: List[str]
    conditions: Dict[str, Any]
    signal_type: str
    priority: int
    enabled: bool = True
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    
    def __post_init__(self):
        """初始化后处理"""
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()


class AlertConfigManager:
    """预警配置管理器"""
    
    def __init__(self, config_dir: str = "config/alerts"):
        """
        初始化配置管理器
        
        Args:
            config_dir: 配置文件目录
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.config_file = self.config_dir / "alert_rules.yaml"
        self.template_file = self.config_dir / "rule_templates.yaml"
        
        self.rules_config: Dict[str, AlertRuleConfig] = {}
        self.templates: Dict[str, Dict[str, Any]] = {}
        
        # 加载配置
        self._load_configurations()
        
        logger.info(f"预警配置管理器初始化完成，配置目录: {self.config_dir}")
    
    @exception_handler(reraise=True)
    def _load_configurations(self):
        """加载配置文件"""
        # 加载预警规则配置
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f) or {}
                
                for rule_id, rule_data in config_data.get('rules', {}).items():
                    self.rules_config[rule_id] = AlertRuleConfig(**rule_data)
                
                logger.info(f"加载 {len(self.rules_config)} 个预警规则配置")
            except Exception as e:
                logger.error(f"加载预警规则配置失败: {e}")
        else:
            # 创建默认配置
            self._create_default_config()
        
        # 加载模板配置
        if self.template_file.exists():
            try:
                with open(self.template_file, 'r', encoding='utf-8') as f:
                    self.templates = yaml.safe_load(f) or {}
                
                logger.info(f"加载 {len(self.templates)} 个预警规则模板")
            except Exception as e:
                logger.error(f"加载预警规则模板失败: {e}")
        else:
            # 创建默认模板
            self._create_default_templates()
    
    def _create_default_config(self):
        """创建默认配置"""
        default_rules = {
            "rsi_overbought_oversold": AlertRuleConfig(
                id="rsi_overbought_oversold",
                name="RSI超买超卖",
                description="RSI指标超买(>70)或超卖(<30)预警",
                indicators=["RSI"],
                conditions={
                    "rsi_overbought": 70,
                    "rsi_oversold": 30
                },
                signal_type="RISK_WARNING",
                priority=3
            ),
            "macd_golden_death_cross": AlertRuleConfig(
                id="macd_golden_death_cross",
                name="MACD金叉死叉",
                description="MACD指标金叉买入、死叉卖出信号",
                indicators=["MACD"],
                conditions={
                    "golden_cross_threshold": 0.01,
                    "death_cross_threshold": -0.01
                },
                signal_type="BUY",
                priority=4
            ),
            "kdj_overbought_oversold": AlertRuleConfig(
                id="kdj_overbought_oversold",
                name="KDJ超买超卖",
                description="KDJ指标超买(>80)或超卖(<20)预警",
                indicators=["KDJ"],
                conditions={
                    "k_overbought": 80,
                    "k_oversold": 20,
                    "d_overbought": 80,
                    "d_oversold": 20
                },
                signal_type="OPPORTUNITY",
                priority=3
            )
        }
        
        self.rules_config = default_rules
        self._save_config()
        logger.info("创建默认预警规则配置")
    
    def _create_default_templates(self):
        """创建默认模板"""
        default_templates = {
            "rsi_template": {
                "name": "RSI模板",
                "description": "RSI指标预警模板",
                "indicators": ["RSI"],
                "conditions": {
                    "rsi_overbought": {"type": "float", "min": 60, "max": 90, "default": 70},
                    "rsi_oversold": {"type": "float", "min": 10, "max": 40, "default": 30}
                },
                "signal_type": "RISK_WARNING",
                "priority": {"type": "int", "min": 1, "max": 5, "default": 3}
            },
            "macd_template": {
                "name": "MACD模板",
                "description": "MACD指标预警模板",
                "indicators": ["MACD"],
                "conditions": {
                    "golden_cross_threshold": {"type": "float", "min": 0.001, "max": 0.1, "default": 0.01},
                    "death_cross_threshold": {"type": "float", "min": -0.1, "max": -0.001, "default": -0.01}
                },
                "signal_type": "BUY",
                "priority": {"type": "int", "min": 1, "max": 5, "default": 4}
            },
            "volume_template": {
                "name": "成交量模板",
                "description": "成交量异常预警模板",
                "indicators": ["VOL"],
                "conditions": {
                    "volume_surge_ratio": {"type": "float", "min": 1.5, "max": 10.0, "default": 3.0},
                    "volume_shrink_ratio": {"type": "float", "min": 0.1, "max": 0.8, "default": 0.3}
                },
                "signal_type": "RISK_WARNING",
                "priority": {"type": "int", "min": 1, "max": 5, "default": 3}
            }
        }
        
        self.templates = default_templates
        self._save_templates()
        logger.info("创建默认预警规则模板")
    
    @exception_handler(reraise=True)
    def _save_config(self):
        """保存配置到文件"""
        config_data = {
            "metadata": {
                "version": "1.0",
                "updated_at": datetime.now().isoformat(),
                "total_rules": len(self.rules_config)
            },
            "rules": {
                rule_id: asdict(rule_config) 
                for rule_id, rule_config in self.rules_config.items()
            }
        }
        
        with open(self.config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"保存 {len(self.rules_config)} 个预警规则配置到 {self.config_file}")
    
    @exception_handler(reraise=True)
    def _save_templates(self):
        """保存模板到文件"""
        template_data = {
            "metadata": {
                "version": "1.0",
                "updated_at": datetime.now().isoformat(),
                "total_templates": len(self.templates)
            },
            "templates": self.templates
        }
        
        with open(self.template_file, 'w', encoding='utf-8') as f:
            yaml.dump(template_data, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"保存 {len(self.templates)} 个预警规则模板到 {self.template_file}")
    
    @exception_handler(reraise=True)
    def create_rule(self, rule_config: AlertRuleConfig) -> bool:
        """
        创建新的预警规则
        
        Args:
            rule_config: 预警规则配置
            
        Returns:
            bool: 创建是否成功
        """
        if rule_config.id in self.rules_config:
            logger.warning(f"预警规则 {rule_config.id} 已存在")
            return False
        
        # 验证配置
        if not self._validate_rule_config(rule_config):
            return False
        
        self.rules_config[rule_config.id] = rule_config
        self._save_config()
        
        logger.info(f"创建预警规则: {rule_config.id} - {rule_config.name}")
        return True
    
    @exception_handler(reraise=True)
    def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """
        更新预警规则
        
        Args:
            rule_id: 规则ID
            updates: 更新的字段
            
        Returns:
            bool: 更新是否成功
        """
        if rule_id not in self.rules_config:
            logger.error(f"预警规则 {rule_id} 不存在")
            return False
        
        rule_config = self.rules_config[rule_id]
        
        # 更新字段
        for field, value in updates.items():
            if hasattr(rule_config, field):
                setattr(rule_config, field, value)
        
        rule_config.updated_at = datetime.now().isoformat()
        
        # 验证更新后的配置
        if not self._validate_rule_config(rule_config):
            return False
        
        self._save_config()
        
        logger.info(f"更新预警规则: {rule_id} - {updates}")
        return True
    
    @exception_handler(reraise=True)
    def delete_rule(self, rule_id: str) -> bool:
        """
        删除预警规则
        
        Args:
            rule_id: 规则ID
            
        Returns:
            bool: 删除是否成功
        """
        if rule_id not in self.rules_config:
            logger.error(f"预警规则 {rule_id} 不存在")
            return False
        
        del self.rules_config[rule_id]
        self._save_config()
        
        logger.info(f"删除预警规则: {rule_id}")
        return True
    
    def _validate_rule_config(self, rule_config: AlertRuleConfig) -> bool:
        """验证规则配置"""
        try:
            # 验证必填字段
            if not rule_config.id or not rule_config.name:
                logger.error("规则ID和名称不能为空")
                return False
            
            # 验证信号类型
            valid_signal_types = ["BUY", "SELL", "RISK_WARNING", "OPPORTUNITY"]
            if rule_config.signal_type not in valid_signal_types:
                logger.error(f"无效的信号类型: {rule_config.signal_type}")
                return False
            
            # 验证优先级
            if not 1 <= rule_config.priority <= 5:
                logger.error(f"优先级必须在1-5之间: {rule_config.priority}")
                return False
            
            # 验证条件
            if not isinstance(rule_config.conditions, dict):
                logger.error("条件必须是字典类型")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"验证规则配置失败: {e}")
            return False
    
    def get_rule(self, rule_id: str) -> Optional[AlertRuleConfig]:
        """获取预警规则"""
        return self.rules_config.get(rule_id)
    
    def get_all_rules(self) -> Dict[str, AlertRuleConfig]:
        """获取所有预警规则"""
        return self.rules_config.copy()
    
    def get_enabled_rules(self) -> Dict[str, AlertRuleConfig]:
        """获取启用的预警规则"""
        return {
            rule_id: rule_config 
            for rule_id, rule_config in self.rules_config.items() 
            if rule_config.enabled
        }
    
    def get_templates(self) -> Dict[str, Dict[str, Any]]:
        """获取所有模板"""
        return self.templates.copy()
    
    @exception_handler(reraise=True)
    def create_rule_from_template(self, template_id: str, rule_id: str, 
                                 rule_name: str, custom_conditions: Dict[str, Any] = None) -> bool:
        """
        从模板创建预警规则
        
        Args:
            template_id: 模板ID
            rule_id: 新规则ID
            rule_name: 新规则名称
            custom_conditions: 自定义条件
            
        Returns:
            bool: 创建是否成功
        """
        if template_id not in self.templates:
            logger.error(f"模板 {template_id} 不存在")
            return False
        
        template = self.templates[template_id]
        
        # 构建条件
        conditions = {}
        for condition_name, condition_config in template.get('conditions', {}).items():
            if custom_conditions and condition_name in custom_conditions:
                conditions[condition_name] = custom_conditions[condition_name]
            else:
                conditions[condition_name] = condition_config.get('default')
        
        # 创建规则配置
        rule_config = AlertRuleConfig(
            id=rule_id,
            name=rule_name,
            description=template.get('description', ''),
            indicators=template.get('indicators', []),
            conditions=conditions,
            signal_type=template.get('signal_type', 'OPPORTUNITY'),
            priority=template.get('priority', {}).get('default', 3)
        )
        
        return self.create_rule(rule_config)


# 全局配置管理器实例
_alert_config_manager = None


def get_alert_config_manager() -> AlertConfigManager:
    """获取预警配置管理器实例"""
    global _alert_config_manager
    if _alert_config_manager is None:
        _alert_config_manager = AlertConfigManager()
    return _alert_config_manager
