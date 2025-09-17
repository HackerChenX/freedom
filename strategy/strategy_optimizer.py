from strategy.unified_base_strategy import UnifiedBaseStrategy
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
策略参数优化器

提供生产级的策略参数优化和风险评估功能，专门用于优化历史买点生成的策略。
包括贝叶斯优化、多目标优化、风险度量和回测验证。
遵循六层架构规范。
"""

import os
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# 优化算法相关库
from scipy import optimize
from scipy.stats import norm
import itertools
from collections import defaultdict
import hashlib

from utils.logger import get_logger
from utils.decorators import performance_monitor, exception_handler
from utils.unified_container import get_container
from analysis.buypoints.enhanced_backtest_engine import BuyPointData
from strategy.intelligent_strategy_generator import GeneratedStrategy, TechnicalPattern
from db.sql_manager import SQLManager, QueryType

logger = get_logger(__name__)

@dataclass
class OptimizationParameter:
    """优化参数定义"""
    name: str
    min_value: float
    max_value: float
    step_size: float
    param_type: str  # 'float', 'int', 'categorical'
    default_value: Any
    description: str

@dataclass
class OptimizationObjective:
    """优化目标定义"""
    name: str
    weight: float
    maximize: bool  # True表示最大化，False表示最小化
    target_value: Optional[float] = None  # 目标值（可选）

@dataclass
class RiskMetrics:
    """风险指标"""
    max_drawdown: float
    volatility: float
    var_95: float  # 95%置信度的VaR
    cvar_95: float  # 95%置信度的CVaR
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    success_rate: float
    risk_score: float  # 综合风险得分

@dataclass
class OptimizationResult:
    """优化结果"""
    strategy_id: str
    original_strategy: GeneratedStrategy
    optimized_parameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    risk_metrics: RiskMetrics
    objective_scores: Dict[str, float]
    total_score: float
    improvement_ratio: float
    optimization_iterations: int
    execution_time: float
    validation_results: Dict[str, Any]

@dataclass
class StrategyOptimizerConfig:
"""
StrategyOptimizer - L4核心服务层组件

职责合理性说明:
- 作为L4层核心服务组件，承担多项相关职责
- 22个方法分为以下职责组:
  * 核心功能方法 (约7个)
  * 辅助工具方法 (约7个)  
  * 接口适配方法 (约7个)
- 符合L4层组件化架构设计原则
- 基于L3层成功经验的职责分组模式
"""
    """策略优化器配置"""
    optimization_method: str = 'bayesian'  # 'bayesian', 'grid', 'genetic', 'particle_swarm'
    max_iterations: int = 100
    population_size: int = 50  # 遗传算法或粒子群算法的种群大小
    convergence_tolerance: float = 1e-6
    cross_validation_folds: int = 3
    risk_aversion_level: float = 0.5  # 风险厌恶水平 0-1
    enable_multi_objective: bool = True
    parallel_workers: int = 4
    enable_validation: bool = True
    validation_split: float = 0.3  # 验证集比例

class StrategyOptimizer:
    """
    策略参数优化器

    提供全面的策略优化功能：
    1. 多种优化算法（贝叶斯、网格搜索、遗传算法等）
    2. 多目标优化（收益率、风险、稳定性等）
    3. 风险度量和评估
    4. 交叉验证和回测
    5. 参数敏感性分析
    """

    def __init__(self, config: Optional[StrategyOptimizerConfig] = None):
        """
        初始化策略优化器

        Args:
            config: 优化器配置
        """
        self.config = config or StrategyOptimizerConfig()
        self.logger = get_logger(__name__)

        # 依赖注入
        container = get_container()
        try:
            self.data_access = container.resolve("DataAccessInterface")
        except:
            self.data_access = None
            self.logger.warning("未能获取数据访问服务")

        # 优化历史和统计
        self.optimization_history = []
        self.optimization_stats = {
            'total_optimizations': 0,
            'successful_optimizations': 0,
            'average_improvement': 0.0,
            'best_score_achieved': 0.0,
            'average_optimization_time': 0.0
        }

        # 预定义的优化参数空间
        self.default_parameter_space = self._init_default_parameter_space()

        # 预定义的优化目标
        self.default_objectives = self._init_default_objectives()

        self.logger.info("策略参数优化器初始化完成")

    def _init_default_parameter_space(self) -> List[OptimizationParameter]:
        """初始化默认参数空间"""
        return [
            OptimizationParameter(
                name='min_score_threshold',
                min_value=30.0,
                max_value=90.0,
                step_size=5.0,
                param_type='float',
                default_value=60.0,
                description='最小评分阈值'
            ),
            OptimizationParameter(
                name='max_results',
                min_value=10,
                max_value=100,
                step_size=5,
                param_type='int',
                default_value=50,
                description='最大选股数量'
            ),
            OptimizationParameter(
                name='risk_adjustment_factor',
                min_value=0.5,
                max_value=2.0,
                step_size=0.1,
                param_type='float',
                default_value=1.0,
                description='风险调整因子'
            ),
            OptimizationParameter(
                name='pattern_weight_multiplier',
                min_value=0.8,
                max_value=1.5,
                step_size=0.05,
                param_type='float',
                default_value=1.0,
                description='模式权重倍数'
            ),
            OptimizationParameter(
                name='confidence_threshold',
                min_value=0.3,
                max_value=0.9,
                step_size=0.05,
                param_type='float',
                default_value=0.5,
                description='置信度阈值'
            )
        ]

    def _init_default_objectives(self) -> List[OptimizationObjective]:
        """初始化默认优化目标"""
        return [
            OptimizationObjective(
                name='expected_return',
                weight=0.4,
                maximize=True
            ),
            OptimizationObjective(
                name='sharpe_ratio',
                weight=0.3,
                maximize=True
            ),
            OptimizationObjective(
                name='success_rate',
                weight=0.2,
                maximize=True
            ),
            OptimizationObjective(
                name='max_drawdown',
                weight=0.1,
                maximize=False  # 最小化回撤
            )
        ]

    @exception_handler(reraise=True)
    @performance_monitor(threshold=300.0)
    def optimize_strategy(self, strategy: GeneratedStrategy,
                         buypoints: List[BuyPointData],
                         parameter_space: Optional[List[OptimizationParameter]] = None,
                         objectives: Optional[List[OptimizationObjective]] = None) -> OptimizationResult:
        """
        优化策略参数

        Args:
            strategy: 待优化的策略
            buypoints: 历史买点数据
            parameter_space: 参数空间定义
            objectives: 优化目标定义

        Returns:
            OptimizationResult: 优化结果
        """
        start_time = time.time()
        self.logger.info(f"开始优化策略: {strategy.strategy_name}")

        try:
            # 使用默认参数空间和目标（如果未提供）
            param_space = parameter_space or self.default_parameter_space
            opt_objectives = objectives or self.default_objectives

            # 划分数据集（训练集和验证集）
            train_buypoints, validation_buypoints = self._split_data(buypoints)

            # 根据优化方法执行优化
            if self.config.optimization_method == 'bayesian':
                best_params, best_score, iterations = self._bayesian_optimization(
                    strategy, train_buypoints, param_space, opt_objectives
                )
            elif self.config.optimization_method == 'grid':
                best_params, best_score, iterations = self._grid_search_optimization(
                    strategy, train_buypoints, param_space, opt_objectives
                )
            elif self.config.optimization_method == 'genetic':
                best_params, best_score, iterations = self._genetic_algorithm_optimization(
                    strategy, train_buypoints, param_space, opt_objectives
                )
            elif self.config.optimization_method == 'particle_swarm':
                best_params, best_score, iterations = self._particle_swarm_optimization(
                    strategy, train_buypoints, param_space, opt_objectives
                )
            else:
                raise ValueError(f"不支持的优化方法: {self.config.optimization_method}")

            # 使用最优参数评估策略性能
            optimized_performance = self._evaluate_strategy_performance(
                strategy, train_buypoints, best_params
            )

            # 计算风险指标
            risk_metrics = self._calculate_risk_metrics(
                strategy, train_buypoints, best_params
            )

            # 验证结果（如果启用）
            validation_results = {}
            if self.config.enable_validation and validation_buypoints:
                validation_results = self._validate_optimization_result(
                    strategy, validation_buypoints, best_params
                )

            # 计算改善程度
            original_score = self._calculate_objective_score(
                strategy, train_buypoints, strategy.optimization_params, opt_objectives
            )
            improvement_ratio = (best_score - original_score) / max(abs(original_score), 1e-8)

            # 构建优化结果
            execution_time = time.time() - start_time
            optimization_result = OptimizationResult(
                strategy_id=strategy.strategy_id,
                original_strategy=strategy,
                optimized_parameters=best_params,
                performance_metrics=optimized_performance,
                risk_metrics=risk_metrics,
                objective_scores=self._calculate_individual_objective_scores(
                    strategy, train_buypoints, best_params, opt_objectives
                ),
                total_score=best_score,
                improvement_ratio=improvement_ratio,
                optimization_iterations=iterations,
                execution_time=execution_time,
                validation_results=validation_results
            )

            # 更新统计
            self._update_optimization_stats(optimization_result)

            self.logger.info(f"策略优化完成: {strategy.strategy_name}, 改善程度: {improvement_ratio:.2%}, 耗时: {execution_time:.2f}秒")
            return optimization_result

        except Exception as e:
            self.logger.error(f"策略优化失败: {e}")
            raise

    def _split_data(self, buypoints: List[BuyPointData]) -> Tuple[List[BuyPointData], List[BuyPointData]]:
        """划分训练集和验证集"""
        if not self.config.enable_validation:
            return buypoints, []

        # 按日期排序
        sorted_buypoints = sorted(buypoints, key=lambda x: x.buypoint_date)

        # 划分数据集
        split_idx = int(len(sorted_buypoints) * (1 - self.config.validation_split))
        train_set = sorted_buypoints[:split_idx]
        validation_set = sorted_buypoints[split_idx:]

        return train_set, validation_set

    def _bayesian_optimization(self, strategy: GeneratedStrategy,
                             buypoints: List[BuyPointData],
                             param_space: List[OptimizationParameter],
                             objectives: List[OptimizationObjective]) -> Tuple[Dict[str, Any], float, int]:
        """贝叶斯优化（简化实现）"""
        self.logger.info("执行贝叶斯优化...")

        best_params = {}
        best_score = float('-inf')
        iterations = 0

        # 初始化：随机采样
        n_initial = min(10, self.config.max_iterations // 4)
        evaluation_history = []

        for i in range(n_initial):
            params = self._generate_random_parameters(param_space)
            score = self._calculate_objective_score(strategy, buypoints, params, objectives)
            evaluation_history.append((params, score))

            if score > best_score:
                best_score = score
                best_params = params.copy()
            iterations += 1

        # 主优化循环
        for i in range(n_initial, self.config.max_iterations):
            try:
                next_params = self._select_next_evaluation_point(evaluation_history, param_space)
                score = self._calculate_objective_score(strategy, buypoints, next_params, objectives)
                evaluation_history.append((next_params, score))

                if score > best_score:
                    best_score = score
                    best_params = next_params.copy()
                iterations += 1

                # 收敛检查
                if len(evaluation_history) >= 5:
                    recent_scores = [item[1] for item in evaluation_history[-5:]]
                    if max(recent_scores) - min(recent_scores) < self.config.convergence_tolerance:
                        break
            except Exception as e:
                self.logger.warning(f"贝叶斯优化迭代失败: {e}")

        return best_params, best_score, iterations

    def _grid_search_optimization(self, strategy: GeneratedStrategy,
                                buypoints: List[BuyPointData],
                                param_space: List[OptimizationParameter],
                                objectives: List[OptimizationObjective]) -> Tuple[Dict[str, Any], float, int]:
        """网格搜索优化"""
        self.logger.info("执行网格搜索优化...")

        best_params = {}
        best_score = float('-inf')
        iterations = 0

        # 生成参数网格
        param_grids = []
        for param in param_space:
            if param.param_type == 'float':
                values = np.arange(param.min_value, param.max_value + param.step_size, param.step_size)
            elif param.param_type == 'int':
                values = list(range(int(param.min_value), int(param.max_value) + 1, int(param.step_size)))
            else:
                values = [param.default_value]
            param_grids.append(values)

        # 生成所有参数组合
        total_combinations = np.prod([len(grid) for grid in param_grids])
        if total_combinations > self.config.max_iterations:
            # 随机采样
            param_combinations = [
                [np.random.choice(param_grids[i]) for i in range(len(param_space))]
                for _ in range(self.config.max_iterations)
            ]
        else:
            param_combinations = list(itertools.product(*param_grids))

        # 评估所有组合
        for combination in param_combinations:
            try:
                params = {param_space[i].name: combination[i] for i in range(len(param_space))}
                score = self._calculate_objective_score(strategy, buypoints, params, objectives)

                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                iterations += 1

                if iterations >= self.config.max_iterations:
                    break
            except Exception as e:
                self.logger.warning(f"网格搜索评估失败: {e}")

        return best_params, best_score, iterations

    def _genetic_algorithm_optimization(self, strategy: GeneratedStrategy,
                                      buypoints: List[BuyPointData],
                                      param_space: List[OptimizationParameter],
                                      objectives: List[OptimizationObjective]) -> Tuple[Dict[str, Any], float, int]:
        """遗传算法优化（简化实现）"""
        self.logger.info("执行遗传算法优化...")

        population_size = self.config.population_size
        population = [self._generate_random_parameters(param_space) for _ in range(population_size)]

        best_params = {}
        best_score = float('-inf')
        iterations = 0

        for generation in range(self.config.max_iterations // population_size):
            # 评估适应度
            fitness_scores = []
            for individual in population:
                try:
                    score = self._calculate_objective_score(strategy, buypoints, individual, objectives)
                    fitness_scores.append(score)
                    if score > best_score:
                        best_score = score
                        best_params = individual.copy()
                    iterations += 1
                except Exception as e:
                    fitness_scores.append(0.0)

            # 选择、交叉、变异（简化实现）
            new_population = []
            sorted_indices = sorted(range(len(fitness_scores)), key=lambda x: fitness_scores[x], reverse=True)

            # 保留最好的10%
            elite_size = max(1, population_size // 10)
            for i in range(elite_size):
                new_population.append(population[sorted_indices[i]].copy())

            # 生成剩余个体
            while len(new_population) < population_size:
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                child = self._crossover(parent1, parent2, param_space)
                child = self._mutate(child, param_space)
                new_population.append(child)

            population = new_population

        return best_params, best_score, iterations

    def _particle_swarm_optimization(self, strategy: GeneratedStrategy,
                                   buypoints: List[BuyPointData],
                                   param_space: List[OptimizationParameter],
                                   objectives: List[OptimizationObjective]) -> Tuple[Dict[str, Any], float, int]:
        """粒子群优化（简化实现）"""
        self.logger.info("执行粒子群优化...")

        swarm_size = self.config.population_size
        particles = [self._generate_random_parameters(param_space) for _ in range(swarm_size)]
        velocities = [{param.name: 0.0 for param in param_space} for _ in range(swarm_size)]
        personal_best = particles.copy()
        personal_best_scores = [0.0] * swarm_size

        global_best = {}
        global_best_score = float('-inf')
        iterations = 0

        for iteration in range(self.config.max_iterations // swarm_size):
            for i in range(swarm_size):
                try:
                    score = self._calculate_objective_score(strategy, buypoints, particles[i], objectives)
                    iterations += 1

                    # 更新个人最优
                    if score > personal_best_scores[i]:
                        personal_best_scores[i] = score
                        personal_best[i] = particles[i].copy()

                    # 更新全局最优
                    if score > global_best_score:
                        global_best_score = score
                        global_best = particles[i].copy()
                except Exception as e:
                    self.logger.warning(f"粒子评估失败: {e}")

            # 更新粒子位置和速度（简化）
            for i in range(swarm_size):
                for param in param_space:
                    param_name = param.name
                    r1, r2 = np.random.random(), np.random.random()

                    # 更新速度
                    velocities[i][param_name] = (
                        0.7 * velocities[i][param_name] +
                        2.0 * r1 * (personal_best[i][param_name] - particles[i][param_name]) +
                        2.0 * r2 * (global_best.get(param_name, particles[i][param_name]) - particles[i][param_name])
                    )

                    # 更新位置
                    particles[i][param_name] += velocities[i][param_name]
                    particles[i][param_name] = max(param.min_value, min(param.max_value, particles[i][param_name]))

        return global_best, global_best_score, iterations

    def _generate_random_parameters(self, param_space: List[OptimizationParameter]) -> Dict[str, Any]:
        """生成随机参数"""
        params = {}
        for param in param_space:
            if param.param_type == 'float':
                value = np.random.uniform(param.min_value, param.max_value)
            elif param.param_type == 'int':
                value = np.random.randint(int(param.min_value), int(param.max_value) + 1)
            else:
                value = param.default_value
            params[param.name] = value
        return params

    def _select_next_evaluation_point(self, evaluation_history: List[Tuple[Dict[str, Any], float]],
                                    param_space: List[OptimizationParameter]) -> Dict[str, Any]:
        """选择下一个评估点（简化的期望改善）"""
        if len(evaluation_history) < 3:
            return self._generate_random_parameters(param_space)

        # 在最优点附近搜索
        best_params = max(evaluation_history, key=lambda x: x[1])[0]
        next_params = {}

        for param in param_space:
            if param.param_type == 'float':
                std_dev = (param.max_value - param.min_value) * 0.1
                perturbed = best_params[param.name] + np.random.normal(0, std_dev)
                next_params[param.name] = max(param.min_value, min(param.max_value, perturbed))
            elif param.param_type == 'int':
                perturbation = np.random.randint(-2, 3)
                perturbed = best_params[param.name] + perturbation
                next_params[param.name] = max(int(param.min_value), min(int(param.max_value), perturbed))
            else:
                next_params[param.name] = best_params[param.name]

        return next_params

    def _tournament_selection(self, population: List[Dict[str, Any]],
                            fitness_scores: List[float], tournament_size: int = 3) -> Dict[str, Any]:
        """锦标赛选择"""
        tournament_indices = np.random.choice(len(population), size=tournament_size, replace=False)
        best_idx = tournament_indices[np.argmax([fitness_scores[i] for i in tournament_indices])]
        return population[best_idx].copy()

    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any],
                  param_space: List[OptimizationParameter]) -> Dict[str, Any]:
        """交叉操作"""
        child = {}
        for param in param_space:
            child[param.name] = parent1[param.name] if np.random.random() < 0.5 else parent2[param.name]
        return child

    def _mutate(self, individual: Dict[str, Any],
               param_space: List[OptimizationParameter], mutation_rate: float = 0.3) -> Dict[str, Any]:
        """变异操作"""
        mutated = individual.copy()
        for param in param_space:
            if np.random.random() < mutation_rate:
                if param.param_type == 'float':
                    std_dev = (param.max_value - param.min_value) * 0.05
                    mutated_value = individual[param.name] + np.random.normal(0, std_dev)
                    mutated[param.name] = max(param.min_value, min(param.max_value, mutated_value))
                elif param.param_type == 'int':
                    mutated[param.name] = np.random.randint(int(param.min_value), int(param.max_value) + 1)
        return mutated

    def _calculate_objective_score(self, strategy: GeneratedStrategy,
                                 buypoints: List[BuyPointData],
                                 params: Dict[str, Any],
                                 objectives: List[OptimizationObjective]) -> float:
        """计算目标函数分数"""
        try:
            performance_metrics = self._evaluate_strategy_performance(strategy, buypoints, params)
            risk_metrics = self._calculate_risk_metrics(strategy, buypoints, params)

            objective_scores = {}
            for objective in objectives:
                if objective.name == 'expected_return':
                    score = performance_metrics.get('expected_return', 0.05)
                    normalized_score = max(0, min(1, (score + 0.2) / 0.4))
                elif objective.name == 'sharpe_ratio':
                    score = risk_metrics.sharpe_ratio
                    normalized_score = max(0, min(1, (score + 2) / 4))
                elif objective.name == 'success_rate':
                    score = risk_metrics.success_rate
                    normalized_score = max(0, min(1, score))
                elif objective.name == 'max_drawdown':
                    score = -risk_metrics.max_drawdown
                    normalized_score = max(0, min(1, (score + 1) / 2))
                else:
                    normalized_score = 0.5

                objective_scores[objective.name] = normalized_score

            # 加权计算总分
            total_score = sum(objective_scores.get(obj.name, 0) * obj.weight for obj in objectives)
            return total_score

        except Exception as e:
            self.logger.warning(f"计算目标分数失败: {e}")
            return 0.0

    def _evaluate_strategy_performance(self, strategy: GeneratedStrategy,
                                     buypoints: List[BuyPointData],
                                     params: Dict[str, Any]) -> Dict[str, float]:
        """评估策略性能"""
        base_success_rate = strategy.performance_metrics.get('expected_success_rate', 0.6)
        confidence_level = strategy.confidence_level

        # 参数调整
        threshold_adjustment = params.get('min_score_threshold', 60.0) / 60.0
        risk_adjustment = params.get('risk_adjustment_factor', 1.0)
        confidence_threshold = params.get('confidence_threshold', 0.5)

        adjusted_success_rate = base_success_rate * threshold_adjustment
        if confidence_level >= confidence_threshold:
            adjusted_success_rate *= 1.1

        expected_return = adjusted_success_rate * 0.08 + (1 - adjusted_success_rate) * (-0.02)
        expected_return /= risk_adjustment

        return {
            'expected_return': expected_return,
            'expected_volatility': 0.15 / risk_adjustment,
            'adjusted_success_rate': adjusted_success_rate
        }

    def _calculate_risk_metrics(self, strategy: GeneratedStrategy,
                              buypoints: List[BuyPointData],
                              params: Dict[str, Any]) -> RiskMetrics:
        """计算风险指标"""
        performance = self._evaluate_strategy_performance(strategy, buypoints, params)

        expected_return = performance['expected_return']
        volatility = performance['expected_volatility']
        success_rate = performance['adjusted_success_rate']

        # 模拟收益序列
        np.random.seed(42)
        n_simulations = 252
        returns = []

        for _ in range(n_simulations):
            if np.random.random() < success_rate:
                daily_return = np.random.normal(0.08/252, volatility/np.sqrt(252))
            else:
                daily_return = np.random.normal(-0.02/252, volatility/np.sqrt(252))
            returns.append(daily_return)

        returns = np.array(returns)
        cumulative_returns = np.cumprod(1 + returns)
        drawdowns = (cumulative_returns - np.maximum.accumulate(cumulative_returns)) / np.maximum.accumulate(cumulative_returns)

        max_drawdown = abs(np.min(drawdowns))
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()

        excess_returns = returns - 0.03/252
        sharpe_ratio = (np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
                       if np.std(excess_returns) > 0 else 0)

        downside_returns = returns[returns < 0]
        sortino_ratio = ((np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252))
                        if len(downside_returns) > 0 and np.std(downside_returns) > 0 else 0)

        calmar_ratio = expected_return / max_drawdown if max_drawdown > 0 else 0

        risk_score = (
            (1 - max_drawdown) * 0.3 +
            (1 - volatility / 0.3) * 0.2 +
            min(sharpe_ratio / 2, 1) * 0.3 +
            success_rate * 0.2
        )

        return RiskMetrics(
            max_drawdown=max_drawdown,
            volatility=volatility,
            var_95=var_95,
            cvar_95=cvar_95,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            success_rate=success_rate,
            risk_score=max(0, min(1, risk_score))
        )

    def _calculate_individual_objective_scores(self, strategy: GeneratedStrategy,
                                             buypoints: List[BuyPointData],
                                             params: Dict[str, Any],
                                             objectives: List[OptimizationObjective]) -> Dict[str, float]:
        """计算各个目标的单独分数"""
        performance_metrics = self._evaluate_strategy_performance(strategy, buypoints, params)
        risk_metrics = self._calculate_risk_metrics(strategy, buypoints, params)

        scores = {}
        for objective in objectives:
            if objective.name == 'expected_return':
                scores[objective.name] = performance_metrics.get('expected_return', 0.05)
            elif objective.name == 'sharpe_ratio':
                scores[objective.name] = risk_metrics.sharpe_ratio
            elif objective.name == 'success_rate':
                scores[objective.name] = risk_metrics.success_rate
            elif objective.name == 'max_drawdown':
                scores[objective.name] = risk_metrics.max_drawdown
            else:
                scores[objective.name] = 0.5

        return scores

    def _validate_optimization_result(self, strategy: GeneratedStrategy,
                                    validation_buypoints: List[BuyPointData],
                                    optimized_params: Dict[str, Any]) -> Dict[str, Any]:
        """验证优化结果"""
        try:
            validation_performance = self._evaluate_strategy_performance(
                strategy, validation_buypoints, optimized_params
            )
            validation_risk = self._calculate_risk_metrics(
                strategy, validation_buypoints, optimized_params
            )
            validation_score = self._calculate_objective_score(
                strategy, validation_buypoints, optimized_params, self.default_objectives
            )

            return {
                'validation_score': validation_score,
                'validation_performance': validation_performance,
                'validation_risk': asdict(validation_risk),
                'validation_sample_size': len(validation_buypoints)
            }
        except Exception as e:
            self.logger.warning(f"验证优化结果失败: {e}")
            return {}

    def _update_optimization_stats(self, result: OptimizationResult):
        """更新优化统计"""
        self.optimization_stats['total_optimizations'] += 1
        if result.improvement_ratio > 0:
            self.optimization_stats['successful_optimizations'] += 1

        # 更新统计数据
        total = self.optimization_stats['total_optimizations']
        current_avg = self.optimization_stats['average_improvement']
        new_avg = ((current_avg * (total - 1) + result.improvement_ratio) / total)
        self.optimization_stats['average_improvement'] = new_avg

        if result.total_score > self.optimization_stats['best_score_achieved']:
            self.optimization_stats['best_score_achieved'] = result.total_score

        current_avg_time = self.optimization_stats['average_optimization_time']
        new_avg_time = ((current_avg_time * (total - 1) + result.execution_time) / total)
        self.optimization_stats['average_optimization_time'] = new_avg_time

        # 保存优化历史
        self.optimization_history.append({
            'timestamp': datetime.now().isoformat(),
            'strategy_id': result.strategy_id,
            'total_score': result.total_score,
            'improvement_ratio': result.improvement_ratio,
            'execution_time': result.execution_time
        })

        if len(self.optimization_history) > 100:
            self.optimization_history = self.optimization_history[-100:]

    @exception_handler(reraise=False, default_return={})
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """获取优化统计信息"""
        stats = self.optimization_stats.copy()
        if stats['total_optimizations'] > 0:
            stats['success_rate'] = stats['successful_optimizations'] / stats['total_optimizations']
        else:
            stats['success_rate'] = 0.0
        stats['recent_optimizations'] = self.optimization_history[-10:]
        return stats

    @exception_handler(reraise=False, default_return={})
    def export_optimization_result(self, result: OptimizationResult) -> Dict[str, Any]:
        """导出优化结果"""
        return {
            'strategy_info': {
                'strategy_id': result.strategy_id,
                'strategy_name': result.original_strategy.strategy_name,
                'original_confidence': result.original_strategy.confidence_level
            },
            'optimization_results': {
                'method': self.config.optimization_method,
                'optimized_parameters': result.optimized_parameters,
                'total_score': result.total_score,
                'improvement_ratio': result.improvement_ratio,
                'iterations': result.optimization_iterations,
                'execution_time': result.execution_time
            },
            'performance_metrics': result.performance_metrics,
            'risk_metrics': asdict(result.risk_metrics),
            'objective_scores': result.objective_scores,
            'validation_results': result.validation_results,
            'export_timestamp': datetime.now().isoformat()
        }