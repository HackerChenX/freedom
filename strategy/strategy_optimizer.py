#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
策略优化器模块

用于优化选股策略的参数和组合，提高策略的有效性
"""

import os
import sys
import json
import copy
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Set, Callable
from itertools import combinations
import logging
from datetime import datetime, timedelta
import itertools

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from utils.logger import getLogger
from utils.path_utils import get_result_dir
from utils.decorators import performance_monitor, exception_handler
from strategy.strategy_executor import Strategy_executor
from utils.dependency_injection import get_service
from db.interfaces.data_access_interface import Data_access_interface
from utils.dependency_injection import get_service
from db.interfaces.data_access_interface import DataAccessInterface
from strategy.strategy_manager import StrategyManager

# 获取日志记录器
logger = getLogger(__name__)


class StrategyOptimizer:
    """
    策略优化器类
    
    用于优化选股策略的参数和组合，提高策略的有效性
    """
    
    def __init__(self, data_access: Optional[Data_access_interface] = None):
        """
        初始化策略优化器
        
        Args:
            data_access: 数据访问接口实例，如果为None则从依赖注入容器获取
        """
        logger.info("初始化策略优化器")
        
        # 使用依赖注入获取数据访问接口
        self.data_access = data_access or get_service(Data_access_interface)
        
        # 创建策略执行器
        self.strategy_executor = Strategy_executor()
        
        # 优化结果存储
        self.optimization_results = {}
        
        # 结果输出目录
        self.result_dir = get_result_dir()
        os.makedirs(self.result_dir, exist_ok=True)
        
        self.data_manager = get_service(DataAccessInterface)
        self.strategy_manager = StrategyManager()
        self.optimization_cache = {}
        
        logger.info("策略优化器初始化完成")
    
    @performance_monitor(threshold_seconds=10.0)
    @exception_handler(reraise=False, default_return={})
    def optimize_strategy_parameters(self, strategy_config: Dict[str, Any], 
                                   parameter_ranges: Dict[str, List[Any]],
                                   optimization_target: str = "total_return",
                                   max_iterations: int = 100) -> Dict[str, Any]:
        """
        优化策略参数
        
        Args:
            strategy_config: 基础策略配置
            parameter_ranges: 参数范围字典，格式为 {参数名: [可选值列表]}
            optimization_target: 优化目标，如 "total_return", "sharpe_ratio", "win_rate"
            max_iterations: 最大迭代次数
            
        Returns:
            Dict[str, Any]: 优化结果，包含最佳参数和性能指标
        """
        logger.info(f"开始策略参数优化，目标: {optimization_target}")
        
        # 生成参数组合
        param_combinations = self._generate_parameter_combinations(parameter_ranges, max_iterations)
        
        best_params = None
        best_score = float('-inf')
        optimization_history = []
        
        for i, params in enumerate(param_combinations):
            try:
                logger.debug(f"测试参数组合 {i+1}/{len(param_combinations)}: {params}")
                
                # 创建测试策略配置
                test_config = copy.deepcopy(strategy_config)
                self._apply_parameters_to_config(test_config, params)
                
                # 执行策略
                performance = self._evaluate_strategy_performance(test_config)
                
                if performance and optimization_target in performance:
                    score = performance[optimization_target]
                    
                    # 记录优化历史
                    optimization_history.append({
                        'iteration': i + 1,
                        'parameters': params.copy(),
                        'performance': performance.copy(),
                        'score': score
                    })
                    
                    # 更新最佳参数
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                        logger.info(f"发现更优参数组合，分数: {score:.4f}")
                
            except Exception as e:
                logger.warning(f"参数组合 {i+1} 测试失败: {e}")
                continue
        
        # 构建优化结果
        optimization_result = {
            'best_parameters': best_params,
            'best_score': best_score,
            'optimization_target': optimization_target,
            'total_iterations': len(param_combinations),
            'successful_iterations': len(optimization_history),
            'optimization_history': optimization_history,
            'timestamp': datetime.now().isoformat()
        }
        
        # 保存优化结果
        self._save_optimization_result(optimization_result)
        
        logger.info(f"策略参数优化完成，最佳分数: {best_score:.4f}")
        return optimization_result
    
    @performance_monitor(threshold_seconds=15.0)
    @exception_handler(reraise=False, default_return={})
    def optimize_strategy_combination(self, strategy_configs: List[Dict[str, Any]],
                                    combination_weights: Optional[List[float]] = None,
                                    max_strategies: int = 5) -> Dict[str, Any]:
        """
        优化策略组合
        
        Args:
            strategy_configs: 策略配置列表
            combination_weights: 策略权重列表，如果为None则自动优化
            max_strategies: 组合中最大策略数量
            
        Returns:
            Dict[str, Any]: 优化结果
        """
        logger.info(f"开始策略组合优化，候选策略数: {len(strategy_configs)}")
        
        best_combination = None
        best_score = float('-inf')
        combination_history = []
        
        # 生成策略组合
        for combo_size in range(1, min(max_strategies + 1, len(strategy_configs) + 1)):
            for strategy_combo in combinations(range(len(strategy_configs)), combo_size):
                try:
                    # 获取当前组合的策略配置
                    combo_configs = [strategy_configs[i] for i in strategy_combo]
                    
                    # 如果没有提供权重，使用等权重
                    if combination_weights is None:
                        weights = [1.0 / len(combo_configs)] * len(combo_configs)
                    else:
                        weights = [combination_weights[i] for i in strategy_combo]
                        # 归一化权重
                        total_weight = sum(weights)
                        weights = [w / total_weight for w in weights]
                    
                    # 评估组合性能
                    performance = self._evaluate_combination_performance(combo_configs, weights)
                    
                    if performance:
                        score = performance.get('total_return', 0)
                        
                        combination_history.append({
                            'strategy_indices': list(strategy_combo),
                            'weights': weights,
                            'performance': performance,
                            'score': score
                        })
                        
                        if score > best_score:
                            best_score = score
                            best_combination = {
                                'strategy_indices': list(strategy_combo),
                                'weights': weights,
                                'performance': performance
                            }
                            logger.info(f"发现更优组合，分数: {score:.4f}")
                
                except Exception as e:
                    logger.warning(f"组合评估失败: {e}")
                    continue
        
        # 构建组合优化结果
        combination_result = {
            'best_combination': best_combination,
            'best_score': best_score,
            'total_combinations_tested': len(combination_history),
            'combination_history': combination_history,
            'timestamp': datetime.now().isoformat()
        }
        
        # 保存组合优化结果
        self._save_combination_result(combination_result)
        
        logger.info(f"策略组合优化完成，最佳分数: {best_score:.4f}")
        return combination_result
    
    def _generate_parameter_combinations(self, parameter_ranges: Dict[str, List[Any]], 
                                       max_combinations: int) -> List[Dict[str, Any]]:
        """
        生成参数组合
        
        Args:
            parameter_ranges: 参数范围
            max_combinations: 最大组合数
            
        Returns:
            List[Dict[str, Any]]: 参数组合列表
        """
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        
        # 生成笛卡尔积
        combinations_iter = itertools.product(*param_values)
        
        # 限制组合数量
        combinations_list = []
        for i, combo in enumerate(combinations_iter):
            if i >= max_combinations:
                break
            combinations_list.append(dict(zip(param_names, combo)))
        
        logger.info(f"生成参数组合数: {len(combinations_list)}")
        return combinations_list
    
    def _apply_parameters_to_config(self, config: Dict[str, Any], params: Dict[str, Any]):
        """
        将参数应用到策略配置
        
        Args:
            config: 策略配置
            params: 参数字典
        """
        if 'strategy' not in config:
            config['strategy'] = {}
        
        if 'parameters' not in config['strategy']:
            config['strategy']['parameters'] = {}
        
        config['strategy']['parameters'].update(params)
    
    def _evaluate_strategy_performance(self, strategy_config: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """
        评估策略性能
        
        Args:
            strategy_config: 策略配置
            
        Returns:
            Optional[Dict[str, float]]: 性能指标字典
        """
        try:
            # 使用策略执行器执行策略
            execution_result = self.strategy_executor.execute_strategy(strategy_config)
            
            if not execution_result or 'selected_stocks' not in execution_result:
                return None
            
            # 计算性能指标
            selected_stocks = execution_result['selected_stocks']
            
            # 模拟性能计算（实际应用中需要更复杂的回测）
            performance = {
                'total_return': np.random.uniform(0.05, 0.25),  # 模拟总收益率
                'sharpe_ratio': np.random.uniform(0.5, 2.0),   # 模拟夏普比率
                'max_drawdown': np.random.uniform(0.05, 0.20), # 模拟最大回撤
                'win_rate': np.random.uniform(0.4, 0.8),       # 模拟胜率
                'selected_count': len(selected_stocks)
            }
            
            return performance
            
        except Exception as e:
            logger.error(f"策略性能评估失败: {e}")
            return None
    
    def _evaluate_combination_performance(self, strategy_configs: List[Dict[str, Any]], 
                                        weights: List[float]) -> Optional[Dict[str, float]]:
        """
        评估策略组合性能
        
        Args:
            strategy_configs: 策略配置列表
            weights: 权重列表
            
        Returns:
            Optional[Dict[str, float]]: 组合性能指标
        """
        try:
            individual_performances = []
            
            # 评估每个策略的性能
            for config in strategy_configs:
                performance = self._evaluate_strategy_performance(config)
                if performance:
                    individual_performances.append(performance)
                else:
                    return None
            
            # 计算加权组合性能
            combined_performance = {}
            for metric in ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']:
                weighted_sum = sum(perf[metric] * weight 
                                 for perf, weight in zip(individual_performances, weights))
                combined_performance[metric] = weighted_sum
            
            # 计算组合的总选股数（去重）
            total_selected = sum(perf.get('selected_count', 0) for perf in individual_performances)
            combined_performance['selected_count'] = total_selected
            
            return combined_performance
            
        except Exception as e:
            logger.error(f"策略组合性能评估失败: {e}")
            return None
    
    def _save_optimization_result(self, result: Dict[str, Any]):
        """
        保存优化结果
        
        Args:
            result: 优化结果
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"strategy_optimization_{timestamp}.json"
            filepath = os.path.join(self.result_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"优化结果已保存: {filepath}")
            
        except Exception as e:
            logger.error(f"保存优化结果失败: {e}")
    
    def _save_combination_result(self, result: Dict[str, Any]):
        """
        保存组合优化结果
        
        Args:
            result: 组合优化结果
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"strategy_combination_{timestamp}.json"
            filepath = os.path.join(self.result_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"组合优化结果已保存: {filepath}")
            
        except Exception as e:
            logger.error(f"保存组合优化结果失败: {e}")
    
    @exception_handler(reraise=False, default_return=[])
    def get_optimization_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取优化历史记录
        
        Args:
            limit: 返回记录数限制
            
        Returns:
            List[Dict[str, Any]]: 优化历史记录列表
        """
        try:
            history_files = []
            
            # 扫描结果目录
            for filename in os.listdir(self.result_dir):
                if filename.startswith('strategy_optimization_') and filename.endswith('.json'):
                    filepath = os.path.join(self.result_dir, filename)
                    history_files.append((filepath, os.path.getmtime(filepath)))
            
            # 按修改时间排序
            history_files.sort(key=lambda x: x[1], reverse=True)
            
            # 读取历史记录
            history_records = []
            for filepath, _ in history_files[:limit]:
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        record = json.load(f)
                        record['file_path'] = filepath
                        history_records.append(record)
                except Exception as e:
                    logger.warning(f"读取历史记录失败 {filepath}: {e}")
                    continue
            
            return history_records
            
        except Exception as e:
            logger.error(f"获取优化历史失败: {e}")
            return []


# 便捷函数
def create_strategy_optimizer() -> Strategy_optimizer_Strategy_Optimizer:
    """创建策略优化器实例"""
    return Strategy_optimizer_Strategy_Optimizer()


def optimize_parameters(strategy_config: Dict[str, Any], 
                       parameter_ranges: Dict[str, List[Any]],
                       optimization_target: str = "total_return") -> Dict[str, Any]:
    """
    便捷的参数优化函数
    
    Args:
        strategy_config: 策略配置
        parameter_ranges: 参数范围
        optimization_target: 优化目标
        
    Returns:
        Dict[str, Any]: 优化结果
    """
    optimizer = create_strategy_optimizer()
    return optimizer.optimize_strategy_parameters(
        strategy_config, parameter_ranges, optimization_target
    )


def optimize_combination(strategy_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    便捷的策略组合优化函数
    
    Args:
        strategy_configs: 策略配置列表
        
    Returns:
        Dict[str, Any]: 优化结果
    """
    optimizer = create_strategy_optimizer()
    return optimizer.optimize_strategy_combination(strategy_configs) 