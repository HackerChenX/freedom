#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from strategy.strategy_factory import StrategyFactory
from utils.strategy_validator import StrategyValidator
from db.clickhouse_db import get_clickhouse_db
from utils.logger import get_logger
from utils.path_utils import get_backtest_result_dir, get_strategies_dir

# 获取日志记录器
logger = get_logger(__name__)

def create_visualization(result, output_prefix):
    """
    创建策略验证结果的可视化图表
    
    Args:
        result: 验证结果
        output_prefix: 输出文件前缀
    """
    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_prefix)
        os.makedirs(output_dir, exist_ok=True)
        
        # 多周期验证结果可视化
        if 'periods' in result:
            # 提取数据
            dates = []
            selection_ratios = []
            returns = []
            
            for period in result['periods']:
                # 使用结束日期作为标签
                dates.append(period['end_date'])
                selection_ratios.append(period['selection_ratio'])
                
                # 如果有收益率数据
                if 'avg_return' in period:
                    returns.append(period['avg_return'])
            
            # 绘制选股比例图
            plt.figure(figsize=(10, 6))
            plt.plot(dates, selection_ratios, marker='o', linestyle='-', linewidth=2)
            plt.title('策略选股比例随时间变化')
            plt.xlabel('日期')
            plt.ylabel('选股比例')
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{output_prefix}_selection_ratio.png")
            plt.close()
            
            # 如果有收益率数据，绘制收益率图
            if returns:
                plt.figure(figsize=(10, 6))
                plt.plot(dates, returns, marker='o', linestyle='-', linewidth=2, color='green')
                plt.axhline(y=0, color='r', linestyle='--')
                plt.title('策略平均收益率随时间变化')
                plt.xlabel('日期')
                plt.ylabel('平均收益率')
                plt.grid(True)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(f"{output_prefix}_avg_return.png")
                plt.close()
        
        # 参数敏感性分析结果可视化
        if 'sensitivity' in result and 'parameter_sensitivity' in result['sensitivity']:
            for param, data in result['sensitivity']['parameter_sensitivity'].items():
                if 'value_stats' in data:
                    # 提取数据
                    values = []
                    selection_ratios = []
                    returns = []
                    
                    for value, stats in data['value_stats'].items():
                        values.append(value)
                        selection_ratios.append(stats['avg_selection_ratio'])
                        
                        if 'avg_return' in stats:
                            returns.append(stats['avg_return'])
                    
                    # 绘制参数敏感性图
                    plt.figure(figsize=(10, 6))
                    plt.plot(values, selection_ratios, marker='o', linestyle='-', linewidth=2)
                    plt.title(f'参数 {param} 对选股比例的影响')
                    plt.xlabel(f'参数 {param} 值')
                    plt.ylabel('平均选股比例')
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(f"{output_prefix}_param_{param}_selection.png")
                    plt.close()
                    
                    # 如果有收益率数据，绘制收益率敏感性图
                    if returns:
                        plt.figure(figsize=(10, 6))
                        plt.plot(values, returns, marker='o', linestyle='-', linewidth=2, color='green')
                        plt.axhline(y=0, color='r', linestyle='--')
                        plt.title(f'参数 {param} 对平均收益率的影响')
                        plt.xlabel(f'参数 {param} 值')
                        plt.ylabel('平均收益率')
                        plt.grid(True)
                        plt.tight_layout()
                        plt.savefig(f"{output_prefix}_param_{param}_return.png")
                        plt.close()
        
        # 策略对比结果可视化
        if 'strategy_results' in result:
            # 提取数据
            names = []
            selection_ratios = []
            returns = []
            
            for strategy in result['strategy_results']:
                names.append(strategy['strategy_name'])
                selection_ratios.append(strategy['selection_ratio'])
                
                if 'avg_return' in strategy:
                    returns.append(strategy['avg_return'])
            
            # 绘制选股比例对比图
            plt.figure(figsize=(12, 6))
            bars = plt.bar(names, selection_ratios)
            plt.title('不同策略选股比例对比')
            plt.xlabel('策略名称')
            plt.ylabel('选股比例')
            plt.grid(True, axis='y')
            plt.xticks(rotation=45, ha='right')
            
            # 为每个柱子添加数值标签
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2%}',
                        ha='center', va='bottom', rotation=0)
            
            plt.tight_layout()
            plt.savefig(f"{output_prefix}_strategy_comparison_selection.png")
            plt.close()
            
            # 如果有收益率数据，绘制收益率对比图
            if returns:
                plt.figure(figsize=(12, 6))
                bars = plt.bar(names, returns, color='green')
                plt.axhline(y=0, color='r', linestyle='--')
                plt.title('不同策略平均收益率对比')
                plt.xlabel('策略名称')
                plt.ylabel('平均收益率')
                plt.grid(True, axis='y')
                plt.xticks(rotation=45, ha='right')
                
                # 为每个柱子添加数值标签
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.2%}',
                            ha='center', va='bottom', rotation=0)
                
                plt.tight_layout()
                plt.savefig(f"{output_prefix}_strategy_comparison_return.png")
                plt.close()
        
        # 策略有效性监测结果可视化
        if 'effectiveness_trend' in result:
            # TODO: 实现策略有效性趋势可视化
            pass
        
        logger.info(f"验证结果可视化已保存到: {output_prefix}*.png")
        
    except Exception as e:
        logger.error(f"创建可视化图表时出错: {e}")

def main():
    """命令行入口函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="策略验证工具")
    
    # 添加子命令
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # 多周期验证子命令
    multi_period_parser = subparsers.add_parser('multi_period', help='多周期验证')
    multi_period_parser.add_argument('-s', '--strategy', required=True, help='策略ID')
    multi_period_parser.add_argument('-o', '--output', required=True, help='输出文件路径')
    multi_period_parser.add_argument('-p', '--pool', help='股票池文件路径')
    multi_period_parser.add_argument('-v', '--visualize', action='store_true', help='是否生成可视化结果')
    multi_period_parser.add_argument('--start_date', help='起始日期，格式YYYYMMDD')
    multi_period_parser.add_argument('--end_date', help='结束日期，格式YYYYMMDD')
    multi_period_parser.add_argument('--periods', type=int, default=3, help='验证周期数量')
    multi_period_parser.add_argument('--period_length', type=int, default=30, help='每个周期的天数')
    
    # 参数敏感性分析子命令
    sensitivity_parser = subparsers.add_parser('sensitivity', help='参数敏感性分析')
    sensitivity_parser.add_argument('-s', '--strategy', required=True, help='策略ID')
    sensitivity_parser.add_argument('-o', '--output', required=True, help='输出文件路径')
    sensitivity_parser.add_argument('-p', '--parameters', required=True, 
                                  help='参数配置文件路径，JSON格式')
    sensitivity_parser.add_argument('-v', '--visualize', action='store_true', help='是否生成可视化结果')
    sensitivity_parser.add_argument('--start_date', help='分析开始日期，格式YYYYMMDD')
    sensitivity_parser.add_argument('--end_date', help='分析结束日期，格式YYYYMMDD')
    sensitivity_parser.add_argument('--pool', help='股票池文件路径')
    
    # 策略对比子命令
    compare_parser = subparsers.add_parser('compare', help='策略对比')
    compare_parser.add_argument('-s', '--strategies', required=True, 
                              help='策略ID列表，用逗号分隔')
    compare_parser.add_argument('-o', '--output', required=True, help='输出文件路径')
    compare_parser.add_argument('-v', '--visualize', action='store_true', help='是否生成可视化结果')
    compare_parser.add_argument('--start_date', help='对比开始日期，格式YYYYMMDD')
    compare_parser.add_argument('--end_date', help='对比结束日期，格式YYYYMMDD')
    compare_parser.add_argument('--pool', help='股票池文件路径')
    
    # 有效性监测子命令
    monitor_parser = subparsers.add_parser('monitor', help='策略有效性监测')
    monitor_parser.add_argument('-s', '--strategy', required=True, help='策略ID')
    monitor_parser.add_argument('-o', '--output', required=True, help='输出文件路径')
    monitor_parser.add_argument('-v', '--visualize', action='store_true', help='是否生成可视化结果')
    monitor_parser.add_argument('-p', '--periods', type=int, default=6, help='监测周期数量')
    monitor_parser.add_argument('--period_length', type=int, default=30, help='每个周期的天数')
    monitor_parser.add_argument('--pool', help='股票池文件路径')
    
    # 策略优化子命令
    optimize_parser = subparsers.add_parser('optimize', help='策略参数优化')
    optimize_parser.add_argument('-s', '--strategy', required=True, help='策略ID')
    optimize_parser.add_argument('-o', '--output', required=True, help='输出文件路径')
    optimize_parser.add_argument('-m', '--method', choices=['grid', 'bayesian', 'genetic'], 
                               default='grid', help='优化方法')
    optimize_parser.add_argument('-p', '--parameters', required=True, 
                               help='参数范围配置文件路径，JSON格式')
    optimize_parser.add_argument('--start_date', help='优化开始日期，格式YYYYMMDD')
    optimize_parser.add_argument('--end_date', help='优化结束日期，格式YYYYMMDD')
    optimize_parser.add_argument('--pool', help='股票池文件路径')
    optimize_parser.add_argument('--iterations', type=int, default=20, help='优化迭代次数')
    
    # 多策略组合子命令
    combine_parser = subparsers.add_parser('combine', help='多策略组合优化')
    combine_parser.add_argument('-s', '--strategies', required=True, 
                              help='策略ID列表，用逗号分隔')
    combine_parser.add_argument('-o', '--output', required=True, help='输出文件路径')
    combine_parser.add_argument('-v', '--visualize', action='store_true', help='是否生成可视化结果')
    combine_parser.add_argument('--start_date', help='组合优化开始日期，格式YYYYMMDD')
    combine_parser.add_argument('--end_date', help='组合优化结束日期，格式YYYYMMDD')
    combine_parser.add_argument('--pool', help='股票池文件路径')
    combine_parser.add_argument('--method', choices=['equal', 'performance', 'correlation', 'sharpe'], 
                              default='performance', help='组合权重计算方法')
    
    args = parser.parse_args()
    
    # 创建验证器
    validator = StrategyValidator(strategy=args.strategy, start_date=args.start_date, end_date=args.end_date)
    
    # 运行验证
    is_valid = validator.validate()
    
    if is_valid:
        logger.info(f"策略 '{args.strategy}' 验证通过。")
    else:
        logger.error(f"策略 '{args.strategy}' 验证失败。")

if __name__ == "__main__":
    main() 