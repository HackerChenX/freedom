#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
运行优化版回测系统

提供命令行接口用于运行优化回测分析
"""

import os
import sys

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

import argparse
import json
import datetime
from typing import Dict, List, Any, Optional

from scripts.backtest.optimized_backtest import OptimizedBacktest
from utils.logger import get_logger
from utils.path_utils import get_backtest_result_dir, get_strategies_dir

# 获取日志记录器
logger = get_logger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='运行优化版回测系统')
    
    # 基本参数
    parser.add_argument('--mode', type=str, required=True, choices=['analyze', 'generate', 'backtest', 'optimize'],
                       help='运行模式：分析(analyze)、生成策略(generate)、回测(backtest)或优化(optimize)')
    
    # 分析模式参数
    parser.add_argument('--input', type=str, help='输入CSV文件路径（包含股票代码和买点日期）')
    parser.add_argument('--output', type=str, help='输出文件路径')
    parser.add_argument('--config', type=str, help='配置文件路径')
    
    # 生成策略参数
    parser.add_argument('--analysis-result', type=str, help='分析结果文件路径（用于生成策略）')
    parser.add_argument('--threshold', type=int, default=2, help='形态出现次数阈值（用于生成策略）')
    
    # 回测参数
    parser.add_argument('--strategy', type=str, help='策略文件路径（用于回测）')
    parser.add_argument('--stock-pool', type=str, help='股票池文件路径（用于回测）')
    parser.add_argument('--start-date', type=str, help='回测开始日期（YYYYMMDD）')
    parser.add_argument('--end-date', type=str, help='回测结束日期（YYYYMMDD）')
    
    # 并行计算参数
    parser.add_argument('--cpu-cores', type=int, help='并行计算使用的CPU核心数')
    
    return parser.parse_args()

def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        Dict[str, Any]: 配置字典
    """
    if not config_path:
        return {}
        
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"加载配置文件时出错: {e}")
        return {}

def load_analysis_result(result_path: str) -> List[Any]:
    """
    加载分析结果
    
    Args:
        result_path: 结果文件路径
        
    Returns:
        List[Any]: 分析结果列表
    """
    try:
        with open(result_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('stocks', [])
    except Exception as e:
        logger.error(f"加载分析结果时出错: {e}")
        return []

def load_strategy(strategy_path: str) -> Dict[str, Any]:
    """
    加载策略
    
    Args:
        strategy_path: 策略文件路径
        
    Returns:
        Dict[str, Any]: 策略配置
    """
    try:
        with open(strategy_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"加载策略文件时出错: {e}")
        return {}

def load_stock_pool(pool_path: str) -> List[str]:
    """
    加载股票池
    
    Args:
        pool_path: 股票池文件路径
        
    Returns:
        List[str]: 股票代码列表
    """
    try:
        stock_pool = []
        with open(pool_path, 'r', encoding='utf-8') as f:
            for line in f:
                code = line.strip()
                if code:
                    stock_pool.append(code)
        return stock_pool
    except Exception as e:
        logger.error(f"加载股票池文件时出错: {e}")
        return []

def run_analyze_mode(args):
    """运行分析模式"""
    if not args.input:
        logger.error("分析模式需要指定输入文件")
        return
        
    if not args.output:
        # 默认输出文件
        output_dir = get_backtest_result_dir()
        os.makedirs(output_dir, exist_ok=True)
        args.output = os.path.join(output_dir, f"analysis_result_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    # 加载配置
    config = load_config(args.config)
    
    # 创建回测实例
    backtest = OptimizedBacktest(config=config, cpu_cores=args.cpu_cores)
    
    # 运行批量分析
    logger.info(f"开始分析，输入文件: {args.input}")
    results = backtest.batch_analyze(args.input, args.output, custom_config=config)
    logger.info(f"分析完成，结果已保存到: {args.output}")
    
    return results

def run_generate_mode(args):
    """运行生成策略模式"""
    if not args.analysis_result:
        logger.error("生成策略模式需要指定分析结果文件")
        return
        
    if not args.output:
        # 默认输出文件
        output_dir = get_strategies_dir()
        os.makedirs(output_dir, exist_ok=True)
        args.output = os.path.join(output_dir, f"strategy_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    # 加载分析结果
    analysis_results = load_analysis_result(args.analysis_result)
    
    if not analysis_results:
        logger.error("分析结果为空，无法生成策略")
        return
    
    # 创建回测实例
    backtest = OptimizedBacktest()
    
    # 生成策略
    logger.info(f"开始生成策略，基于分析结果: {args.analysis_result}")
    strategy = backtest.generate_strategy(analysis_results, args.output, threshold=args.threshold)
    logger.info(f"策略生成完成，已保存到: {args.output}")
    
    return strategy

def run_backtest_mode(args):
    """运行回测模式"""
    if not args.strategy:
        logger.error("回测模式需要指定策略文件")
        return
        
    if not args.stock_pool:
        logger.error("回测模式需要指定股票池文件")
        return
        
    if not args.start_date or not args.end_date:
        logger.error("回测模式需要指定开始日期和结束日期")
        return
        
    if not args.output:
        # 默认输出文件
        output_dir = get_backtest_result_dir()
        os.makedirs(output_dir, exist_ok=True)
        args.output = os.path.join(output_dir, f"backtest_result_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    # 加载策略和股票池
    strategy = load_strategy(args.strategy)
    stock_pool = load_stock_pool(args.stock_pool)
    
    if not strategy:
        logger.error("策略为空，无法进行回测")
        return
        
    if not stock_pool:
        logger.error("股票池为空，无法进行回测")
        return
    
    # 加载配置
    config = load_config(args.config)
    
    # 创建回测实例
    backtest = OptimizedBacktest(config=config)
    
    # 运行回测
    logger.info(f"开始回测，策略: {args.strategy}, 股票池: {args.stock_pool}")
    result = backtest.backtest_strategy(strategy, stock_pool, args.start_date, args.end_date, args.output)
    logger.info(f"回测完成，结果已保存到: {args.output}")
    
    return result

def run_optimize_mode(args):
    """运行优化模式（结合分析、生成和回测）"""
    logger.info("开始优化模式")
    
    # 检查必要参数
    if not args.input:
        logger.error("优化模式需要指定输入文件")
        return
        
    if not args.stock_pool:
        logger.error("优化模式需要指定股票池文件")
        return
        
    if not args.start_date or not args.end_date:
        logger.error("优化模式需要指定开始日期和结束日期")
        return
    
    # 设置默认输出文件
    output_dir = get_backtest_result_dir()
    strategies_dir = get_strategies_dir()
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(strategies_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    analysis_output = args.output or os.path.join(output_dir, f"analysis_result_{timestamp}.json")
    strategy_output = os.path.join(strategies_dir, f"strategy_{timestamp}.json")
    backtest_output = os.path.join(output_dir, f"backtest_result_{timestamp}.json")
    
    # 加载配置
    config = load_config(args.config)
    
    # 创建回测实例
    backtest = OptimizedBacktest(config=config, cpu_cores=args.cpu_cores)
    
    # 第一步：分析
    logger.info("第1步: 运行分析")
    analysis_results = backtest.batch_analyze(args.input, analysis_output, custom_config=config)
    
    # 第二步：生成策略
    logger.info("第2步: 生成策略")
    strategy = backtest.generate_strategy(analysis_results, strategy_output, threshold=args.threshold or 2)
    
    # 第三步：回测策略
    logger.info("第3步: 回测策略")
    stock_pool = load_stock_pool(args.stock_pool)
    if not stock_pool:
        logger.error("股票池为空，无法进行回测")
        return
        
    result = backtest.backtest_strategy(strategy, stock_pool, args.start_date, args.end_date, backtest_output)
    
    logger.info("优化模式完成！")
    logger.info(f"分析结果: {analysis_output}")
    logger.info(f"生成策略: {strategy_output}")
    logger.info(f"回测结果: {backtest_output}")
    
    return {
        'analysis_results': analysis_results,
        'strategy': strategy,
        'backtest_result': result
    }

def main():
    """主函数"""
    args = parse_args()
    
    # 运行对应模式
    if args.mode == 'analyze':
        run_analyze_mode(args)
    elif args.mode == 'generate':
        run_generate_mode(args)
    elif args.mode == 'backtest':
        run_backtest_mode(args)
    elif args.mode == 'optimize':
        run_optimize_mode(args)
    else:
        logger.error(f"未知的运行模式: {args.mode}")

if __name__ == '__main__':
    main() 