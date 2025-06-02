#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
综合回测系统

用于技术形态回测和统计分析
"""

import os
import sys
import logging
import datetime
import argparse
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set, Union

# 获取项目根目录
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from utils.logger import get_logger
from utils.decorators import performance_monitor, time_it
from scripts.backtest.backtest_runner import BacktestRunner
from scripts.backtest.data_manager import BacktestDataManager
from scripts.backtest.pattern_analyzer import PatternAnalyzer
from scripts.backtest.strategy_manager import StrategyManager

# 获取日志记录器
logger = get_logger(__name__)


@time_it
def run_backtest(stock_code: str, start_date: str, end_date: str, 
                strategy: str = "pattern", **kwargs) -> Dict[str, Any]:
    """
    运行回测
    
    Args:
        stock_code: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        strategy: 策略类型
        **kwargs: 其他参数
        
    Returns:
        Dict[str, Any]: 回测结果
    """
    # 初始化回测运行器
    runner = BacktestRunner()
    
    # 运行回测
    if strategy == "pattern":
        result = runner.run_pattern_recognition(
            stock_code=stock_code,
            start_date=start_date,
            end_date=end_date,
            pattern_ids=kwargs.get("pattern_ids"),
            min_strength=kwargs.get("min_strength", 0.6)
        )
    elif strategy == "multi_period":
        result = runner.run_multi_period_analysis(
            stock_code=stock_code,
            start_date=start_date,
            end_date=end_date,
            periods=kwargs.get("periods"),
            pattern_combinations=kwargs.get("pattern_combinations"),
            min_strength=kwargs.get("min_strength", 60.0)
        )
    elif strategy == "zxm":
        result = runner.run_zxm_analysis(
            stock_code=stock_code,
            start_date=start_date,
            end_date=end_date,
            periods=kwargs.get("periods"),
            buy_score_threshold=kwargs.get("buy_score_threshold", 60.0)
        )
    else:
        logger.error(f"未知的策略类型: {strategy}")
        return {"success": False, "error": f"未知的策略类型: {strategy}"}
    
    return result


@time_it
def batch_backtest(stock_codes: List[str], start_date: str, end_date: str,
                  strategy: str = "pattern", output_file: str = None, **kwargs) -> List[Dict[str, Any]]:
    """
    批量回测
    
    Args:
        stock_codes: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        strategy: 策略类型
        output_file: 输出文件
        **kwargs: 其他参数
        
    Returns:
        List[Dict[str, Any]]: 回测结果列表
    """
    # 初始化回测运行器
    runner = BacktestRunner()
    
    # 批量回测
    if strategy == "pattern":
        results = runner.batch_run_pattern_recognition(
            stock_codes=stock_codes,
            start_date=start_date,
            end_date=end_date,
            pattern_ids=kwargs.get("pattern_ids"),
            min_strength=kwargs.get("min_strength", 0.6)
        )
    elif strategy == "zxm":
        results = runner.batch_run_zxm_analysis(
            stock_codes=stock_codes,
            start_date=start_date,
            end_date=end_date,
            periods=kwargs.get("periods"),
            buy_score_threshold=kwargs.get("buy_score_threshold", 60.0)
        )
    else:
        logger.error(f"未知的策略类型: {strategy}")
        return [{"success": False, "error": f"未知的策略类型: {strategy}"}]
    
    # 导出结果
    if output_file:
        runner.export_results_to_csv(output_file, results)
        logger.info(f"回测结果已导出到: {output_file}")
    
    return results


@time_it
def analyze_patterns(results: List[Dict[str, Any]], 
                    output_file: str = None) -> Dict[str, Any]:
    """
    分析形态统计
    
    Args:
        results: 回测结果列表
        output_file: 输出文件
        
    Returns:
        Dict[str, Any]: 统计信息
    """
    # 初始化回测运行器
    runner = BacktestRunner()
    
    # 生成统计
    stats = runner.generate_pattern_statistics(results)
    
    # 导出统计
    if output_file:
        import json
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info(f"统计信息已导出到: {output_file}")
    
    return stats


@time_it
def analyze_zxm(results: List[Dict[str, Any]], 
               output_file: str = None) -> Dict[str, Any]:
    """
    分析ZXM统计
    
    Args:
        results: 回测结果列表
        output_file: 输出文件
        
    Returns:
        Dict[str, Any]: 统计信息
    """
    # 初始化回测运行器
    runner = BacktestRunner()
    
    # 生成统计
    stats = runner.generate_zxm_statistics(results)
    
    # 导出统计
    if output_file:
        import json
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info(f"统计信息已导出到: {output_file}")
    
    return stats


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="综合回测系统")
    
    # 添加子命令
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # 单股回测命令
    single_parser = subparsers.add_parser("single", help="单股回测")
    single_parser.add_argument("--stock", "-s", type=str, required=True, help="股票代码")
    single_parser.add_argument("--start-date", "-sd", type=str, required=True, help="开始日期")
    single_parser.add_argument("--end-date", "-ed", type=str, required=True, help="结束日期")
    single_parser.add_argument("--strategy", "-st", type=str, default="pattern", 
                             choices=["pattern", "multi_period", "zxm"], help="策略类型")
    single_parser.add_argument("--output", "-o", type=str, help="输出文件")
    
    # 批量回测命令
    batch_parser = subparsers.add_parser("batch", help="批量回测")
    batch_parser.add_argument("--stock-file", "-sf", type=str, required=True, help="股票列表文件")
    batch_parser.add_argument("--start-date", "-sd", type=str, required=True, help="开始日期")
    batch_parser.add_argument("--end-date", "-ed", type=str, required=True, help="结束日期")
    batch_parser.add_argument("--strategy", "-st", type=str, default="pattern", 
                            choices=["pattern", "zxm"], help="策略类型")
    batch_parser.add_argument("--output", "-o", type=str, default="backtest_results.csv", 
                            help="输出文件")
    
    # 统计命令
    stats_parser = subparsers.add_parser("stats", help="生成统计信息")
    stats_parser.add_argument("--input", "-i", type=str, required=True, help="输入文件")
    stats_parser.add_argument("--type", "-t", type=str, required=True, 
                            choices=["pattern", "zxm"], help="统计类型")
    stats_parser.add_argument("--output", "-o", type=str, default="stats.json", 
                            help="输出文件")
    
    # 直接调用BacktestRunner
    runner_parser = subparsers.add_parser("runner", help="直接调用BacktestRunner")
    runner_parser.add_argument("--args", "-a", type=str, required=True, 
                             help="传递给BacktestRunner的参数，格式为命令行参数")
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    if args.command == "single":
        # 单股回测
        result = run_backtest(
            stock_code=args.stock,
            start_date=args.start_date,
            end_date=args.end_date,
            strategy=args.strategy
        )
        
        # 输出结果
        if args.strategy == "pattern":
            print(f"\n股票 {args.stock} 的形态识别结果:")
            patterns = result.get("patterns", [])
            if patterns:
                print(f"发现 {len(patterns)} 个形态:")
                for pattern in patterns:
                    print(f"- {pattern['pattern_name']}: 强度 {pattern['strength']:.2f}")
                    print(f"  描述: {pattern['description']}")
            else:
                print("未发现符合条件的形态")
                
        elif args.strategy == "zxm":
            print(result.get("zxm_text", "ZXM分析失败"))
            
        # 导出结果
        if args.output:
            runner = BacktestRunner()
            runner.export_results_to_csv(args.output, [result])
            print(f"结果已导出到: {args.output}")
    
    elif args.command == "batch":
        # 读取股票列表
        try:
            with open(args.stock_file, "r") as f:
                stock_codes = [line.strip() for line in f if line.strip()]
        except Exception as e:
            logger.error(f"读取股票列表文件失败: {e}")
            print(f"错误: 读取股票列表文件失败 - {e}")
            return
        
        print(f"开始批量回测 {len(stock_codes)} 只股票...")
        
        # 批量回测
        results = batch_backtest(
            stock_codes=stock_codes,
            start_date=args.start_date,
            end_date=args.end_date,
            strategy=args.strategy,
            output_file=args.output
        )
        
        print(f"回测完成，结果已导出到: {args.output}")
    
    elif args.command == "stats":
        # 读取输入文件
        try:
            df = pd.read_csv(args.input)
            results = df.to_dict("records")
        except Exception as e:
            logger.error(f"读取输入文件失败: {e}")
            print(f"错误: 读取输入文件失败 - {e}")
            return
        
        print(f"开始生成 {args.type} 类型的统计信息...")
        
        # 生成统计
        if args.type == "pattern":
            stats = analyze_patterns(results, args.output)
        elif args.type == "zxm":
            stats = analyze_zxm(results, args.output)
        
        print(f"统计信息已生成: {args.output}")
    
    elif args.command == "runner":
        # 直接调用BacktestRunner
        import shlex
        import subprocess
        
        runner_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_runner.py")
        cmd = [sys.executable, runner_script] + shlex.split(args.args)
        
        print(f"执行命令: {' '.join(cmd)}")
        subprocess.call(cmd)
    
    else:
        print("未指定有效的命令，使用 -h 查看帮助")


if __name__ == "__main__":
    main()
