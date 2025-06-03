#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
策略生成器

根据指标形态生成选股策略
"""

import os
import sys
import json
from typing import Dict, List, Any, Optional

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from utils.logger import get_logger

logger = get_logger(__name__)

class StrategyGenerator:
    """策略生成器"""
    
    def __init__(self):
        """初始化策略生成器"""
        pass
    
    def generate_strategy(self, 
                       strategy_name: str, 
                       conditions: List[Dict[str, Any]],
                       condition_logic: str = "OR") -> Dict[str, Any]:
        """
        生成选股策略
        
        Args:
            strategy_name: 策略名称
            conditions: 策略条件列表
            condition_logic: 条件逻辑，可选值：AND, OR
            
        Returns:
            Dict[str, Any]: 生成的策略
        """
        try:
            # 验证条件逻辑
            if condition_logic not in ["AND", "OR"]:
                logger.warning(f"无效的条件逻辑: {condition_logic}，使用默认值 OR")
                condition_logic = "OR"
            
            # 构建策略
            strategy = {
                "name": strategy_name,
                "description": f"自动生成的策略：基于买点共性指标",
                "version": "1.0",
                "author": "系统自动生成",
                "condition_logic": condition_logic,
                "conditions": conditions,
                "parameters": {}
            }
            
            # 从条件中提取参数
            parameters = {}
            for idx, condition in enumerate(conditions):
                # 对于每个条件，提取其中的可配置参数
                if 'score_threshold' in condition:
                    param_name = f"score_threshold_{idx}"
                    parameters[param_name] = {
                        "name": f"形态{idx+1}得分阈值",
                        "description": f"指标 {condition.get('indicator', condition.get('pattern', ''))} 形态的得分阈值",
                        "type": "float",
                        "default": condition['score_threshold'],
                        "min": 0,
                        "max": 100
                    }
                    
                    # 将条件中的固定值替换为参数引用
                    condition['score_threshold'] = f"${{{param_name}}}"
            
            # 更新策略参数
            strategy["parameters"] = parameters
            
            logger.info(f"已生成策略 {strategy_name}，包含 {len(conditions)} 个条件")
            return strategy
            
        except Exception as e:
            logger.error(f"生成策略时出错: {e}")
            return {
                "name": strategy_name,
                "description": "生成策略时出错",
                "conditions": [],
                "condition_logic": "OR",
                "parameters": {}
            }
    
    def save_strategy(self, 
                   strategy: Dict[str, Any], 
                   file_path: str) -> bool:
        """
        保存策略到文件
        
        Args:
            strategy: 策略对象
            file_path: 文件路径
            
        Returns:
            bool: 是否成功保存
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # 写入文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(strategy, f, ensure_ascii=False, indent=2)
                
            logger.info(f"策略已保存到 {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存策略时出错: {e}")
            return False
    
    def load_strategy(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        从文件加载策略
        
        Args:
            file_path: 文件路径
            
        Returns:
            Optional[Dict[str, Any]]: 加载的策略，如果失败则返回None
        """
        try:
            # 读取文件
            with open(file_path, 'r', encoding='utf-8') as f:
                strategy = json.load(f)
                
            logger.info(f"已从 {file_path} 加载策略")
            return strategy
            
        except Exception as e:
            logger.error(f"加载策略时出错: {e}")
            return None 