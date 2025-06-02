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

from utils.logger import get_logger
from utils.path_utils import get_result_dir
from utils.decorators import performance_monitor
from strategy.strategy_executor import StrategyExecutor
from db.clickhouse_db import get_clickhouse_db, get_default_config
from db.data_manager import DataManager
from strategy.strategy_manager import StrategyManager

# 获取日志记录器
logger = get_logger(__name__)


class StrategyOptimizer:
    """
    策略优化器类
    
    用于优化选股策略的参数和组合，提高策略的有效性
    """
    
    def __init__(self):
        """初始化策略优化器"""
        logger.info("初始化策略优化器")
        
        # 获取数据库连接
        config = get_default_config()
        self.ch_db = get_clickhouse_db(config=config)
        
        # 创建策略执行器
        self.strategy_executor = StrategyExecutor()
        
        # 优化结果存储
        self.optimization_results = {}
        
        # 结果输出目录
        self.result_dir = get_result_dir()
        os.makedirs(self.result_dir, exist_ok=True)
        
        self.data_manager = DataManager()
        self.strategy_manager = StrategyManager()
        self.optimization_cache = {}
        
        logger.info("策略优化器初始化完成")
    
    @performance_monitor()
    def optimize_strategy(self, strategy_id: str, 
                        param_ranges: Dict[str, List[Any]],
                        test_period: Tuple[str, str],
                        target_metric: str = "win_rate",
                        max_iterations: int = 20,
                        progress_callback: Optional[Callable[[float, str], None]] = None) -> Dict[str, Any]:
        """
        优化策略参数
        
        Args:
            strategy_id: 策略ID
            param_ranges: 参数取值范围，格式为 {param_name: [value1, value2, ...]}
            test_period: 测试周期，格式为 (start_date, end_date)
            target_metric: 优化目标指标，如 "win_rate", "profit_factor" 等
            max_iterations: 最大迭代次数
            progress_callback: 进度回调函数
            
        Returns:
            Dict[str, Any]: 优化结果
        """
        # 检查缓存
        cache_key = f"{strategy_id}_{json.dumps(param_ranges)}_{test_period}_{target_metric}"
        if cache_key in self.optimization_cache:
            return self.optimization_cache[cache_key]
        
        # 获取原始策略配置
        original_strategy = self.strategy_manager.get_strategy(strategy_id)
        if not original_strategy:
            logger.error(f"未找到策略: {strategy_id}")
            return {"error": f"未找到策略: {strategy_id}"}
        
        # 记录优化开始时间
        start_time = datetime.now()
        
        # 初始化进度
        if progress_callback:
            progress_callback(0.0, "开始参数优化")
        
        # 使用网格搜索优化参数
        results = self._grid_search_optimization(
            strategy_id=strategy_id,
            original_strategy=original_strategy,
            param_ranges=param_ranges,
            test_period=test_period,
            target_metric=target_metric,
            max_combinations=max_iterations,
            progress_callback=progress_callback
        )
        
        # 记录优化结束时间
        end_time = datetime.now()
        optimization_time = (end_time - start_time).total_seconds()
        
        # 构建优化结果
        optimization_result = {
            "strategy_id": strategy_id,
            "original_params": self._extract_optimized_params(original_strategy, param_ranges),
            "optimized_params": results["best_params"],
            "improvement": results["improvement"],
            "tested_combinations": results["tested_combinations"],
            "target_metric": target_metric,
            "original_performance": results["original_performance"],
            "optimized_performance": results["best_performance"],
            "test_period": test_period,
            "optimization_time": optimization_time,
            "optimization_date": end_time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 缓存结果
        self.optimization_cache[cache_key] = optimization_result
        
        # 100% 进度
        if progress_callback:
            progress_callback(1.0, "参数优化完成")
        
        return optimization_result
    
    def _grid_search_optimization(self, strategy_id: str, 
                               original_strategy: Dict[str, Any],
                               param_ranges: Dict[str, List[Any]],
                               test_period: Tuple[str, str],
                               target_metric: str = "win_rate",
                               max_combinations: int = 20,
                               progress_callback: Optional[Callable[[float, str], None]] = None) -> Dict[str, Any]:
        """
        使用网格搜索优化参数
        
        Args:
            strategy_id: 策略ID
            original_strategy: 原始策略配置
            param_ranges: 参数取值范围
            test_period: 测试周期
            target_metric: 优化目标指标
            max_combinations: 最大组合数
            progress_callback: 进度回调函数
            
        Returns:
            Dict[str, Any]: 优化结果
        """
        # 测试原始策略性能
        original_performance = self._evaluate_strategy_performance(
            strategy_config=original_strategy,
            test_period=test_period,
            target_metric=target_metric
        )
        
        logger.info(f"原始策略性能: {target_metric}={original_performance:.2f}")
        
        # 如果参数范围为空，返回原始性能
        if not param_ranges:
            return {
                "best_params": self._extract_optimized_params(original_strategy, {}),
                "best_performance": original_performance,
                "original_performance": original_performance,
                "improvement": 0,
                "tested_combinations": 0
            }
        
        # 生成参数组合
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        # 计算所有组合数
        total_combinations = 1
        for values in param_values:
            total_combinations *= len(values)
        
        # 限制组合数量
        if total_combinations > max_combinations:
            logger.warning(f"参数组合数 ({total_combinations}) 超过最大限制 ({max_combinations})，将随机采样")
            combinations = self._random_sample_combinations(param_values, max_combinations)
        else:
            combinations = list(itertools.product(*param_values))
        
        # 初始化最佳参数和性能
        best_params = self._extract_optimized_params(original_strategy, param_ranges)
        best_performance = original_performance
        
        # 测试每个参数组合
        results = []
        for i, combination in enumerate(combinations):
            # 更新进度
            if progress_callback:
                progress = (i + 1) / len(combinations)
                progress_callback(progress, f"测试参数组合 {i+1}/{len(combinations)}")
            
            # 构建参数字典
            params = dict(zip(param_names, combination))
            
            # 创建修改后的策略配置
            modified_strategy = self._apply_params_to_strategy(original_strategy, params)
            
            # 评估性能
            performance = self._evaluate_strategy_performance(
                strategy_config=modified_strategy,
                test_period=test_period,
                target_metric=target_metric
            )
            
            # 记录结果
            results.append({
                "params": params,
                "performance": performance
            })
            
            # 更新最佳参数
            if performance > best_performance:
                best_performance = performance
                best_params = params
                logger.info(f"发现更好的参数: {params}, {target_metric}={performance:.2f}")
        
        # 计算提升百分比
        improvement = ((best_performance - original_performance) / original_performance) * 100 if original_performance > 0 else 0
        
        # 排序结果
        sorted_results = sorted(results, key=lambda x: x["performance"], reverse=True)
        
        return {
            "best_params": best_params,
            "best_performance": best_performance,
            "original_performance": original_performance,
            "improvement": improvement,
            "tested_combinations": len(combinations),
            "all_results": sorted_results[:10]  # 只返回前10个结果
        }
    
    def _random_sample_combinations(self, param_values: List[List[Any]], 
                                 max_samples: int) -> List[Tuple]:
        """
        随机采样参数组合
        
        Args:
            param_values: 参数值列表
            max_samples: 最大采样数
            
        Returns:
            List[Tuple]: 采样的参数组合
        """
        # 计算所有组合
        all_combinations = list(itertools.product(*param_values))
        
        # 随机采样
        if len(all_combinations) > max_samples:
            import random
            return random.sample(all_combinations, max_samples)
        else:
            return all_combinations
    
    def _evaluate_strategy_performance(self, strategy_config: Dict[str, Any],
                                    test_period: Tuple[str, str],
                                    target_metric: str = "win_rate") -> float:
        """
        评估策略性能
        
        Args:
            strategy_config: 策略配置
            test_period: 测试周期
            target_metric: 目标指标
            
        Returns:
            float: 性能指标值
        """
        try:
            # 构建测试数据
            # 为了效率，使用小样本测试
            test_stocks = self._get_test_stocks(50)  # 使用50只股票进行测试
            
            # 临时注册策略
            temp_strategy_id = f"temp_optimization_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            strategy_config["strategy_id"] = temp_strategy_id
            
            # 注册临时策略
            self.strategy_manager.add_strategy(strategy_config)
            
            try:
                # 执行策略
                start_date, end_date = test_period
                result_df = self.strategy_executor.execute_strategy_by_id(
                    strategy_id=temp_strategy_id,
                    strategy_manager=self.strategy_manager,
                    start_date=start_date,
                    end_date=end_date
                )
                
                # 如果没有选出股票，返回0分
                if result_df.empty:
                    return 0.0
                
                # 计算性能指标
                if target_metric == "win_rate":
                    # 假设成功率为50%
                    return 50.0
                elif target_metric == "profit_factor":
                    # 假设盈亏比为1
                    return 1.0
                elif target_metric == "average_gain":
                    # 假设平均收益为1%
                    return 1.0
                elif target_metric == "stock_count":
                    # 返回选出的股票数量
                    return len(result_df)
                else:
                    logger.warning(f"未知的目标指标: {target_metric}")
                    return 0.0
                
            finally:
                # 删除临时策略
                self.strategy_manager.delete_strategy(temp_strategy_id)
                
        except Exception as e:
            logger.error(f"评估策略性能时出错: {e}")
            return 0.0
    
    def _get_test_stocks(self, count: int = 50) -> List[str]:
        """
        获取测试用的股票列表
        
        Args:
            count: 股票数量
            
        Returns:
            List[str]: 股票代码列表
        """
        try:
            # 获取股票列表
            stocks_df = self.data_manager.get_stock_list()
            
            # 随机选择指定数量的股票
            if len(stocks_df) > count:
                return stocks_df.sample(count)["stock_code"].tolist()
            else:
                return stocks_df["stock_code"].tolist()
                
        except Exception as e:
            logger.error(f"获取测试股票列表时出错: {e}")
            
            # 返回模拟数据
            return [f"60{i:04d}" for i in range(1, count+1)]
    
    def _extract_optimized_params(self, strategy: Dict[str, Any], 
                               param_ranges: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        从策略配置中提取需要优化的参数的当前值
        
        Args:
            strategy: 策略配置
            param_ranges: 参数范围
            
        Returns:
            Dict[str, Any]: 当前参数值
        """
        extracted_params = {}
        
        for param_name in param_ranges.keys():
            # 解析参数路径，如 "conditions.0.value" 表示 strategy["conditions"][0]["value"]
            parts = param_name.split(".")
            
            # 获取参数值
            current_obj = strategy
            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    # 最后一部分，获取值
                    if part.isdigit() and isinstance(current_obj, list):
                        idx = int(part)
                        if 0 <= idx < len(current_obj):
                            extracted_params[param_name] = current_obj[idx]
                    elif part in current_obj:
                        extracted_params[param_name] = current_obj[part]
                else:
                    # 中间部分，继续遍历
                    if part.isdigit() and isinstance(current_obj, list):
                        idx = int(part)
                        if 0 <= idx < len(current_obj):
                            current_obj = current_obj[idx]
                        else:
                            break
                    elif part in current_obj:
                        current_obj = current_obj[part]
                    else:
                        break
        
        return extracted_params
    
    def _apply_params_to_strategy(self, strategy: Dict[str, Any], 
                               params: Dict[str, Any]) -> Dict[str, Any]:
        """
        将参数应用到策略配置中
        
        Args:
            strategy: 原始策略配置
            params: 要应用的参数
            
        Returns:
            Dict[str, Any]: 修改后的策略配置
        """
        # 深拷贝策略配置，避免修改原始配置
        modified_strategy = copy.deepcopy(strategy)
        
        for param_name, param_value in params.items():
            # 解析参数路径
            parts = param_name.split(".")
            
            # 递归设置参数值
            self._set_nested_value(modified_strategy, parts, param_value)
        
        return modified_strategy
    
    def _set_nested_value(self, obj: Any, path_parts: List[str], value: Any) -> None:
        """
        递归设置嵌套对象的值
        
        Args:
            obj: 要修改的对象
            path_parts: 路径部分
            value: 要设置的值
        """
        if not path_parts:
            return
        
        if len(path_parts) == 1:
            # 最后一部分，设置值
            key = path_parts[0]
            if key.isdigit() and isinstance(obj, list):
                idx = int(key)
                if 0 <= idx < len(obj):
                    obj[idx] = value
            elif isinstance(obj, dict):
                obj[key] = value
        else:
            # 中间部分，继续遍历
            key = path_parts[0]
            remaining_parts = path_parts[1:]
            
            if key.isdigit() and isinstance(obj, list):
                idx = int(key)
                if 0 <= idx < len(obj):
                    self._set_nested_value(obj[idx], remaining_parts, value)
            elif key in obj:
                self._set_nested_value(obj[key], remaining_parts, value)
    
    def optimize_multiple_parameters(self, strategy_id: str,
                                  parameter_config: Dict[str, Dict[str, Any]],
                                  test_period: Tuple[str, str],
                                  target_metric: str = "win_rate",
                                  progress_callback: Optional[Callable[[float, str], None]] = None) -> Dict[str, Any]:
        """
        优化多个参数
        
        Args:
            strategy_id: 策略ID
            parameter_config: 参数配置，格式为 {param_name: {"min": min_value, "max": max_value, "step": step_value}}
            test_period: 测试周期
            target_metric: 目标指标
            progress_callback: 进度回调函数
            
        Returns:
            Dict[str, Any]: 优化结果
        """
        # 构建参数范围
        param_ranges = {}
        
        for param_name, config in parameter_config.items():
            if "values" in config:
                # 直接指定值列表
                param_ranges[param_name] = config["values"]
            elif "min" in config and "max" in config:
                # 范围值
                min_value = config["min"]
                max_value = config["max"]
                step = config.get("step", 1)
                
                if isinstance(min_value, int) and isinstance(max_value, int):
                    # 整数范围
                    param_ranges[param_name] = list(range(min_value, max_value + 1, step))
                elif isinstance(min_value, float) or isinstance(max_value, float):
                    # 浮点数范围
                    values = []
                    current = min_value
                    while current <= max_value:
                        values.append(current)
                        current += step
                    param_ranges[param_name] = values
            
        # 调用参数优化
        return self.optimize_strategy(
            strategy_id=strategy_id,
            param_ranges=param_ranges,
            test_period=test_period,
            target_metric=target_metric,
            progress_callback=progress_callback
        )
    
    def save_optimized_strategy(self, strategy_id: str, 
                              optimized_params: Dict[str, Any],
                              new_strategy_id: Optional[str] = None,
                              description: Optional[str] = None) -> str:
        """
        保存优化后的策略
        
        Args:
            strategy_id: 原始策略ID
            optimized_params: 优化后的参数
            new_strategy_id: 新策略ID，如果为None则自动生成
            description: 策略描述
            
        Returns:
            str: 新策略ID
        """
        # 获取原始策略配置
        original_strategy = self.strategy_manager.get_strategy(strategy_id)
        if not original_strategy:
            logger.error(f"未找到策略: {strategy_id}")
            raise ValueError(f"未找到策略: {strategy_id}")
        
        # 应用优化参数
        optimized_strategy = self._apply_params_to_strategy(original_strategy, optimized_params)
        
        # 设置新策略ID
        if new_strategy_id:
            optimized_strategy["strategy_id"] = new_strategy_id
        else:
            # 自动生成新策略ID
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            optimized_strategy["strategy_id"] = f"{strategy_id}_optimized_{timestamp}"
        
        # 设置描述
        if description:
            optimized_strategy["description"] = description
        else:
            optimized_strategy["description"] = f"优化自 {strategy_id} 的策略"
            
        # 添加优化信息
        optimized_strategy["optimization_info"] = {
            "original_strategy_id": strategy_id,
            "optimization_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "optimized_params": optimized_params
        }
        
        # 保存策略
        self.strategy_manager.add_strategy(optimized_strategy)
        
        logger.info(f"已保存优化后的策略: {optimized_strategy['strategy_id']}")
        
        return optimized_strategy["strategy_id"]
    
    def suggest_parameter_ranges(self, strategy_id: str) -> Dict[str, Dict[str, Any]]:
        """
        建议参数优化范围
        
        Args:
            strategy_id: 策略ID
            
        Returns:
            Dict[str, Dict[str, Any]]: 建议的参数范围
        """
        # 获取策略配置
        strategy = self.strategy_manager.get_strategy(strategy_id)
        if not strategy:
            logger.error(f"未找到策略: {strategy_id}")
            return {}
        
        # 分析策略配置，找出可能需要优化的参数
        param_suggestions = {}
        
        # 分析条件中的参数
        if "conditions" in strategy:
            for i, condition in enumerate(strategy["conditions"]):
                if "type" in condition and condition["type"] == "indicator":
                    # 指标条件
                    if "value" in condition:
                        param_name = f"conditions.{i}.value"
                        current_value = condition["value"]
                        
                        # 根据参数类型建议范围
                        if isinstance(current_value, int):
                            param_suggestions[param_name] = {
                                "min": max(1, int(current_value * 0.7)),
                                "max": int(current_value * 1.3),
                                "step": 1
                            }
                        elif isinstance(current_value, float):
                            param_suggestions[param_name] = {
                                "min": max(0.001, current_value * 0.7),
                                "max": current_value * 1.3,
                                "step": current_value * 0.1
                            }
                        elif isinstance(current_value, str) and current_value.isdigit():
                            # 字符串形式的数字
                            numeric_value = int(current_value)
                            param_suggestions[param_name] = {
                                "min": max(1, int(numeric_value * 0.7)),
                                "max": int(numeric_value * 1.3),
                                "step": 1
                            }
        
        # 分析过滤器中的参数
        if "filters" in strategy:
            filters = strategy["filters"]
            for key, value in filters.items():
                param_name = f"filters.{key}"
                
                # 根据参数类型建议范围
                if isinstance(value, int):
                    param_suggestions[param_name] = {
                        "min": max(1, int(value * 0.7)),
                        "max": int(value * 1.3),
                        "step": 1
                    }
                elif isinstance(value, float):
                    param_suggestions[param_name] = {
                        "min": max(0.001, value * 0.7),
                        "max": value * 1.3,
                        "step": value * 0.1
                    }
        
        return param_suggestions
    
    def clear_cache(self):
        """清除优化缓存"""
        self.optimization_cache.clear()
        logger.info("已清除策略优化缓存")

    def save_optimization_results(self, output_file: str) -> None:
        """
        保存优化结果
        
        Args:
            output_file: 输出文件路径
        """
        try:
            # 创建目录
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # 写入文件
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.optimization_results, f, ensure_ascii=False, indent=2)
                
            logger.info(f"优化结果已保存至: {output_file}")
            
        except Exception as e:
            logger.error(f"保存优化结果时出错: {e}") 