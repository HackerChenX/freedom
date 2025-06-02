#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
策略验证模块

用于验证选股策略的有效性和稳定性
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Union
import os
import datetime
import json
from collections import defaultdict

from utils.logger import get_logger
from utils.decorators import performance_monitor, time_it
from utils.path_utils import get_backtest_result_dir

logger = get_logger(__name__)

class StrategyValidator:
    """
    策略验证器类
    
    验证选股策略的有效性和稳定性
    """
    
    def __init__(self):
        """初始化策略验证器"""
        self.result_dir = get_backtest_result_dir()
        os.makedirs(self.result_dir, exist_ok=True)
    
    @time_it
    @performance_monitor()
    def validate_backtest_results(self, results: List[Dict[str, Any]], 
                                output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        验证多个回测结果的一致性和稳定性
        
        Args:
            results: 回测结果列表
            output_file: 输出文件路径
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        if not results:
            logger.warning("回测结果列表为空，无法进行验证")
            return {}
        
        # 提取性能指标
        performance_metrics = []
        for result in results:
            if 'performance' in result:
                metrics = {
                    'strategy_name': result.get('strategy_name', 'unnamed'),
                    'win_rate': result['performance'].get('win_rate', 0),
                    'profit_loss_ratio': result['performance'].get('profit_loss_ratio', 0),
                    'max_drawdown': result['performance'].get('max_drawdown', 0),
                    'sharpe_ratio': result['performance'].get('sharpe_ratio', 0),
                    'annual_return': result['performance'].get('annual_return', 0),
                    'total_return': result['performance'].get('total_return', 0),
                    'trade_count': result['performance'].get('trade_count', 0)
                }
                performance_metrics.append(metrics)
        
        if not performance_metrics:
            logger.warning("没有找到有效的性能指标，无法进行验证")
            return {}
        
        # 转换为DataFrame以便分析
        metrics_df = pd.DataFrame(performance_metrics)
        
        # 计算各指标的统计信息
        stats = {}
        for column in metrics_df.columns:
            if column != 'strategy_name':
                stats[column] = {
                    'mean': metrics_df[column].mean(),
                    'median': metrics_df[column].median(),
                    'std': metrics_df[column].std(),
                    'min': metrics_df[column].min(),
                    'max': metrics_df[column].max(),
                    'coefficient_of_variation': metrics_df[column].std() / metrics_df[column].mean() if metrics_df[column].mean() != 0 else float('inf')
                }
        
        # 计算整体稳定性得分
        stability_scores = {}
        for metric, metric_stats in stats.items():
            if metric in ['win_rate', 'profit_loss_ratio', 'sharpe_ratio', 'annual_return', 'total_return']:
                # 这些指标越高越好，且我们希望波动小
                cv = metric_stats['coefficient_of_variation']
                mean_value = metric_stats['mean']
                
                # 稳定性得分：均值越高、变异系数越低越好
                if cv < float('inf') and cv > 0:
                    stability_scores[metric] = mean_value / cv
                else:
                    stability_scores[metric] = 0
            elif metric in ['max_drawdown']:
                # 这些指标越低越好，且我们希望波动小
                cv = metric_stats['coefficient_of_variation']
                mean_value = metric_stats['mean']
                
                # 稳定性得分：均值越低、变异系数越低越好
                if cv < float('inf') and cv > 0 and mean_value > 0:
                    stability_scores[metric] = 1 / (mean_value * cv)
                else:
                    stability_scores[metric] = 0
        
        # 计算综合稳定性得分
        if stability_scores:
            overall_stability = sum(stability_scores.values()) / len(stability_scores)
        else:
            overall_stability = 0
        
        # 标记最佳和最差策略
        best_strategy = None
        worst_strategy = None
        
        if not metrics_df.empty:
            # 按总收益率排序
            metrics_df_sorted = metrics_df.sort_values('total_return', ascending=False)
            best_strategy = metrics_df_sorted.iloc[0].to_dict()
            worst_strategy = metrics_df_sorted.iloc[-1].to_dict()
        
        # 构建验证结果
        validation_result = {
            'stats': stats,
            'stability_scores': stability_scores,
            'overall_stability': overall_stability,
            'best_strategy': best_strategy,
            'worst_strategy': worst_strategy,
            'strategy_count': len(results),
            'validation_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 输出验证结果
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(validation_result, f, ensure_ascii=False, indent=2)
            logger.info(f"验证结果已保存到: {output_file}")
        
        return validation_result
    
    @time_it
    @performance_monitor()
    def cross_validate_strategy(self, strategy: Dict[str, Any], 
                              stock_pools: List[List[str]], 
                              backtest_func: callable,
                              periods: List[Tuple[str, str]],
                              output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        交叉验证策略在不同股票池和时间段的表现
        
        Args:
            strategy: 策略配置
            stock_pools: 多个股票池
            backtest_func: 回测函数
            periods: 时间段列表，每个元素为(start_date, end_date)
            output_file: 输出文件路径
            
        Returns:
            Dict[str, Any]: 交叉验证结果
        """
        results = []
        
        # 在每个股票池和时间段上运行回测
        for i, stock_pool in enumerate(stock_pools):
            for j, (start_date, end_date) in enumerate(periods):
                logger.info(f"运行交叉验证: 股票池 {i+1}/{len(stock_pools)}, 时间段 {j+1}/{len(periods)}")
                
                # 运行回测
                result = backtest_func(strategy, stock_pool, start_date, end_date)
                
                # 添加标识信息
                result['pool_id'] = i
                result['period_id'] = j
                result['pool_size'] = len(stock_pool)
                result['period_start'] = start_date
                result['period_end'] = end_date
                
                results.append(result)
        
        # 分析结果
        cross_validation_result = self._analyze_cross_validation(results)
        
        # 输出结果
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(cross_validation_result, f, ensure_ascii=False, indent=2)
            logger.info(f"交叉验证结果已保存到: {output_file}")
        
        return cross_validation_result
    
    def _analyze_cross_validation(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析交叉验证结果
        
        Args:
            results: 交叉验证结果列表
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        # 提取性能指标
        metrics_by_pool = defaultdict(list)
        metrics_by_period = defaultdict(list)
        
        for result in results:
            if 'performance' not in result:
                continue
                
            pool_id = result.get('pool_id', 0)
            period_id = result.get('period_id', 0)
            
            metrics = {
                'win_rate': result['performance'].get('win_rate', 0),
                'profit_loss_ratio': result['performance'].get('profit_loss_ratio', 0),
                'max_drawdown': result['performance'].get('max_drawdown', 0),
                'sharpe_ratio': result['performance'].get('sharpe_ratio', 0),
                'annual_return': result['performance'].get('annual_return', 0),
                'total_return': result['performance'].get('total_return', 0),
                'trade_count': result['performance'].get('trade_count', 0)
            }
            
            metrics_by_pool[pool_id].append(metrics)
            metrics_by_period[period_id].append(metrics)
        
        # 计算每个股票池的平均表现
        pool_performance = {}
        for pool_id, metrics_list in metrics_by_pool.items():
            if not metrics_list:
                continue
                
            pool_metrics = {}
            for metric in metrics_list[0].keys():
                values = [m[metric] for m in metrics_list]
                pool_metrics[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
            
            pool_performance[str(pool_id)] = pool_metrics
        
        # 计算每个时间段的平均表现
        period_performance = {}
        for period_id, metrics_list in metrics_by_period.items():
            if not metrics_list:
                continue
                
            period_metrics = {}
            for metric in metrics_list[0].keys():
                values = [m[metric] for m in metrics_list]
                period_metrics[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
            
            period_performance[str(period_id)] = period_metrics
        
        # 计算策略稳定性
        stability = {}
        for metric in ['win_rate', 'profit_loss_ratio', 'max_drawdown', 'sharpe_ratio', 'annual_return', 'total_return']:
            all_values = []
            
            for result in results:
                if 'performance' in result and metric in result['performance']:
                    all_values.append(result['performance'][metric])
            
            if all_values:
                stability[metric] = {
                    'mean': np.mean(all_values),
                    'std': np.std(all_values),
                    'coefficient_of_variation': np.std(all_values) / np.mean(all_values) if np.mean(all_values) != 0 else float('inf')
                }
        
        # 构建分析结果
        return {
            'pool_performance': pool_performance,
            'period_performance': period_performance,
            'stability': stability,
            'result_count': len(results),
            'analysis_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    @time_it
    @performance_monitor()
    def evaluate_strategy_robustness(self, strategy: Dict[str, Any], 
                                   stock_pool: List[str],
                                   backtest_func: callable,
                                   period: Tuple[str, str],
                                   parameter_variations: Dict[str, List[Any]],
                                   output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        评估策略在参数变化下的鲁棒性
        
        Args:
            strategy: 基准策略配置
            stock_pool: 股票池
            backtest_func: 回测函数
            period: 时间段(start_date, end_date)
            parameter_variations: 参数变化列表，格式为 {'param_path': [value1, value2, ...]}
            output_file: 输出文件路径
            
        Returns:
            Dict[str, Any]: 鲁棒性评估结果
        """
        results = []
        base_result = None
        
        # 首先运行基准策略
        logger.info("运行基准策略")
        base_result = backtest_func(strategy, stock_pool, period[0], period[1])
        base_result['variation'] = 'base'
        results.append(base_result)
        
        # 对每个参数进行变化测试
        for param_path, variations in parameter_variations.items():
            path_parts = param_path.split('.')
            
            for i, value in enumerate(variations):
                logger.info(f"测试参数 {param_path} 变化 {i+1}/{len(variations)}: {value}")
                
                # 创建策略副本
                strategy_copy = json.loads(json.dumps(strategy))
                
                # 修改参数
                current = strategy_copy
                for part in path_parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                
                current[path_parts[-1]] = value
                
                # 运行回测
                result = backtest_func(strategy_copy, stock_pool, period[0], period[1])
                
                # 添加变化信息
                result['variation'] = f"{param_path}={value}"
                result['param_path'] = param_path
                result['param_value'] = value
                
                results.append(result)
        
        # 分析结果
        robustness_result = self._analyze_robustness(results, base_result)
        
        # 输出结果
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(robustness_result, f, ensure_ascii=False, indent=2)
            logger.info(f"鲁棒性评估结果已保存到: {output_file}")
        
        return robustness_result
    
    def _analyze_robustness(self, results: List[Dict[str, Any]], 
                          base_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析鲁棒性评估结果
        
        Args:
            results: 评估结果列表
            base_result: 基准结果
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        # 提取基准性能指标
        if 'performance' not in base_result:
            logger.warning("基准结果中没有性能指标")
            return {}
        
        base_metrics = {
            'win_rate': base_result['performance'].get('win_rate', 0),
            'profit_loss_ratio': base_result['performance'].get('profit_loss_ratio', 0),
            'max_drawdown': base_result['performance'].get('max_drawdown', 0),
            'sharpe_ratio': base_result['performance'].get('sharpe_ratio', 0),
            'annual_return': base_result['performance'].get('annual_return', 0),
            'total_return': base_result['performance'].get('total_return', 0),
            'trade_count': base_result['performance'].get('trade_count', 0)
        }
        
        # 计算每个变化的性能变化率
        variations = []
        
        for result in results:
            if result == base_result or 'performance' not in result:
                continue
            
            metrics = {
                'win_rate': result['performance'].get('win_rate', 0),
                'profit_loss_ratio': result['performance'].get('profit_loss_ratio', 0),
                'max_drawdown': result['performance'].get('max_drawdown', 0),
                'sharpe_ratio': result['performance'].get('sharpe_ratio', 0),
                'annual_return': result['performance'].get('annual_return', 0),
                'total_return': result['performance'].get('total_return', 0),
                'trade_count': result['performance'].get('trade_count', 0)
            }
            
            # 计算变化率
            changes = {}
            for metric in metrics:
                base_value = base_metrics[metric]
                current_value = metrics[metric]
                
                if base_value != 0:
                    change_ratio = (current_value - base_value) / abs(base_value)
                else:
                    change_ratio = float('inf') if current_value != 0 else 0
                
                changes[metric] = change_ratio
            
            variation_info = {
                'variation': result.get('variation', ''),
                'param_path': result.get('param_path', ''),
                'param_value': result.get('param_value', ''),
                'metrics': metrics,
                'changes': changes,
                'mean_absolute_change': np.mean([abs(c) for c in changes.values() if c != float('inf')])
            }
            
            variations.append(variation_info)
        
        # 按参数分组计算敏感度
        param_sensitivity = defaultdict(list)
        
        for variation in variations:
            param_path = variation.get('param_path', '')
            if param_path:
                param_sensitivity[param_path].append(variation['mean_absolute_change'])
        
        sensitivity = {}
        for param_path, changes in param_sensitivity.items():
            sensitivity[param_path] = np.mean(changes) if changes else 0
        
        # 找出最敏感和最不敏感的参数
        sorted_sensitivity = sorted(sensitivity.items(), key=lambda x: x[1], reverse=True)
        most_sensitive = sorted_sensitivity[:3] if len(sorted_sensitivity) >= 3 else sorted_sensitivity
        least_sensitive = sorted_sensitivity[-3:] if len(sorted_sensitivity) >= 3 else []
        
        # 构建分析结果
        return {
            'base_metrics': base_metrics,
            'variations': variations,
            'sensitivity': sensitivity,
            'most_sensitive_params': most_sensitive,
            'least_sensitive_params': least_sensitive,
            'analysis_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    @staticmethod
    def get_instance() -> 'StrategyValidator':
        """获取单例实例"""
        if not hasattr(StrategyValidator, '_instance'):
            StrategyValidator._instance = StrategyValidator()
        return StrategyValidator._instance 