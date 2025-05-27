#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
股票观察命令行工具

提供策略组合执行和观察信号识别功能
"""

import os
import sys
import argparse
import json
from datetime import datetime, timedelta
import pandas as pd

# 将项目根目录添加到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from strategy.strategy_combiner import StrategyCombiner
from strategy.signal_watcher import SignalWatcher
from strategy.strategy_manager import StrategyManager
from db.data_manager import DataManager
from utils.logger import get_logger

logger = get_logger("stock_watch")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="股票观察工具 - 提供策略组合执行和观察信号识别功能")
    
    # 创建子命令解析器
    subparsers = parser.add_subparsers(dest="command", help="命令")
    
    # 策略组合命令
    combine_parser = subparsers.add_parser("combine", help="执行策略组合")
    combine_parser.add_argument("-s", "--strategies", required=True, help="策略ID列表，以逗号分隔")
    combine_parser.add_argument("-w", "--weights", help="策略权重列表，以逗号分隔，与策略ID顺序对应")
    combine_parser.add_argument("-p", "--pool", help="股票池，以逗号分隔。不指定则使用策略过滤器选择")
    combine_parser.add_argument("-b", "--begin_date", help="开始日期，格式：YYYY-MM-DD")
    combine_parser.add_argument("-e", "--end_date", help="结束日期，格式：YYYY-MM-DD")
    combine_parser.add_argument("-o", "--output", help="输出文件路径")
    combine_parser.add_argument("--parallel", action="store_true", default=True, help="是否并行执行策略")
    combine_parser.add_argument("--max_workers", type=int, default=5, help="最大并行工作线程数")
    combine_parser.add_argument("--save", action="store_true", help="是否保存结果到数据库")
    combine_parser.add_argument("--analyze", action="store_true", help="是否分析策略组合")
    
    # 观察信号命令
    watch_parser = subparsers.add_parser("watch", help="寻找观察信号")
    watch_parser.add_argument("-s", "--strategy", required=True, help="策略ID")
    watch_parser.add_argument("-p", "--pool", help="股票池，以逗号分隔。不指定则使用策略过滤器选择")
    watch_parser.add_argument("-b", "--begin_date", help="开始日期，格式：YYYY-MM-DD")
    watch_parser.add_argument("-e", "--end_date", help="结束日期，格式：YYYY-MM-DD")
    watch_parser.add_argument("-t", "--threshold", type=float, default=0.7, help="接近度阈值（0-1之间）")
    watch_parser.add_argument("-d", "--trend_days", type=int, default=3, help="趋势分析天数")
    watch_parser.add_argument("-o", "--output", help="输出文件路径")
    
    # 观察信号详情命令
    detail_parser = subparsers.add_parser("detail", help="获取观察信号详情")
    detail_parser.add_argument("-s", "--strategy", required=True, help="策略ID")
    detail_parser.add_argument("-c", "--code", required=True, help="股票代码")
    detail_parser.add_argument("-b", "--begin_date", help="开始日期，格式：YYYY-MM-DD")
    detail_parser.add_argument("-e", "--end_date", help="结束日期，格式：YYYY-MM-DD")
    detail_parser.add_argument("-d", "--days", type=int, default=30, help="回溯天数")
    detail_parser.add_argument("-o", "--output", help="输出文件路径")
    
    # 列表命令
    list_parser = subparsers.add_parser("list", help="列出策略")
    list_parser.add_argument("-a", "--all", action="store_true", help="显示所有策略信息")
    list_parser.add_argument("-f", "--filter", help="按名称或描述过滤策略")
    
    return parser.parse_args()


def execute_combine(args):
    """执行策略组合命令"""
    # 解析策略ID列表
    strategy_ids = [s.strip() for s in args.strategies.split(",")]
    
    # 解析策略权重
    weights = None
    if args.weights:
        weight_values = [float(w.strip()) for w in args.weights.split(",")]
        weights = dict(zip(strategy_ids, weight_values))
    
    # 解析股票池
    stock_pool = None
    if args.pool:
        stock_pool = [s.strip() for s in args.pool.split(",")]
    
    # 创建策略组合器
    combiner = StrategyCombiner()
    
    # 执行策略组合
    result = combiner.execute_strategies(
        strategy_ids=strategy_ids,
        weights=weights,
        stock_pool=stock_pool,
        start_date=args.begin_date,
        end_date=args.end_date,
        parallel=args.parallel,
        max_workers=args.max_workers
    )
    
    if result.empty:
        logger.info("未找到满足条件的股票")
        return
    
    # 如果需要保存结果
    if args.save:
        data_manager = DataManager()
        success = data_manager.save_selection_result(result)
        if success:
            logger.info(f"已成功保存 {len(result)} 条选股结果到数据库")
        else:
            logger.error("保存结果到数据库失败")
    
    # 如果需要分析策略组合
    if args.analyze:
        analysis = combiner.get_strategy_analysis(strategy_ids)
        print("\n策略组合分析：")
        print(f"策略数量: {analysis['strategy_count']}")
        
        if analysis["common_filters"]:
            print("\n共同过滤条件:")
            for key, value in analysis["common_filters"].items():
                print(f"  {key}: {value}")
        
        print("\n指标使用情况:")
        for indicator, count in list(analysis["indicator_usage"].items())[:10]:  # 只显示前10个
            print(f"  {indicator}: {count}次")
        
        print("\n周期使用情况:")
        for period, count in analysis["period_usage"].items():
            print(f"  {period}: {count}次")
    
    # 显示结果
    print("\n选股结果：")
    print(f"共找到 {len(result)} 只满足条件的股票")
    
    # 显示前10只股票
    display_columns = ["rank", "stock_code", "stock_name", "combined_score", "strategy_count"]
    print(result[display_columns].head(10).to_string(index=False))
    
    # 如果指定了输出文件
    if args.output:
        result.to_csv(args.output, index=False, encoding="utf-8-sig")
        logger.info(f"已将结果保存到 {args.output}")


def execute_watch(args):
    """执行观察信号命令"""
    # 获取策略配置
    strategy_manager = StrategyManager()
    strategy_plan = strategy_manager.get_strategy(args.strategy)
    
    if not strategy_plan:
        logger.error(f"策略 {args.strategy} 不存在")
        return
    
    # 解析股票池
    stock_pool = None
    if args.pool:
        stock_pool = [s.strip() for s in args.pool.split(",")]
    else:
        # 使用策略过滤器获取股票池
        data_manager = DataManager()
        filters = strategy_plan.get("filters", {})
        stock_list_df = data_manager.get_stock_list(filters)
        
        if stock_list_df.empty:
            logger.warning("未找到符合条件的股票")
            return
            
        stock_pool = stock_list_df['stock_code'].tolist()
    
    # 创建观察信号处理器
    watcher = SignalWatcher()
    
    # 寻找观察信号
    result = watcher.find_watch_signals(
        strategy_plan=strategy_plan,
        stock_pool=stock_pool,
        start_date=args.begin_date,
        end_date=args.end_date,
        threshold=args.threshold,
        trend_days=args.trend_days
    )
    
    if result.empty:
        logger.info("未找到观察信号")
        return
    
    # 显示结果
    print("\n观察信号结果：")
    print(f"共找到 {len(result)} 只具有观察信号的股票")
    
    # 显示前10只股票
    display_columns = ["rank", "stock_code", "stock_name", "proximity", "trend_score", "completion_ratio"]
    print(result[display_columns].head(10).to_string(index=False))
    
    # 如果指定了输出文件
    if args.output:
        result.to_csv(args.output, index=False, encoding="utf-8-sig")
        logger.info(f"已将结果保存到 {args.output}")


def execute_detail(args):
    """执行观察信号详情命令"""
    # 获取策略配置
    strategy_manager = StrategyManager()
    strategy_plan = strategy_manager.get_strategy(args.strategy)
    
    if not strategy_plan:
        logger.error(f"策略 {args.strategy} 不存在")
        return
    
    # 创建观察信号处理器
    watcher = SignalWatcher()
    
    # 获取观察信号详情
    result = watcher.get_watch_signal_details(
        stock_code=args.code,
        strategy_plan=strategy_plan,
        start_date=args.begin_date,
        end_date=args.end_date,
        lookback_days=args.days
    )
    
    if "error" in result:
        logger.error(f"获取观察信号详情失败: {result['error']}")
        return
    
    # 显示基本信息
    print(f"\n股票: {result['stock_code']} {result['stock_name']}")
    print(f"策略: {result['strategy_id']}")
    print(f"时间范围: {result['start_date']} 至 {result['end_date']}")
    
    # 显示条件详情
    print(f"\n共有 {len(result['condition_details'])} 个条件:")
    
    for i, condition in enumerate(result['condition_details']):
        print(f"\n条件 {i+1}:")
        print(f"  指标: {condition['indicator_id']}")
        print(f"  周期: {condition['period']}")
        print(f"  信号类型: {condition['signal_type']}")
        
        # 显示参数
        if condition['parameters']:
            print("  参数:")
            for key, value in condition['parameters'].items():
                print(f"    {key}: {value}")
        
        # 显示信号历史摘要
        signal_history = condition['signal_history']
        if signal_history:
            signal_count = sum(1 for s in signal_history if s['signal'])
            print(f"  信号历史: 总共 {len(signal_history)} 天，其中 {signal_count} 天有信号")
            
            # 显示最近5天的信号
            if len(signal_history) > 0:
                print("  最近信号:")
                for day in signal_history[-5:]:
                    signal_str = "有信号" if day['signal'] else "无信号"
                    strength_str = f"强度: {day['strength']:.2f}" if day['strength'] > 0 else ""
                    print(f"    {day['date']}: {signal_str} {strength_str}")
    
    # 如果指定了输出文件
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"已将详细结果保存到 {args.output}")


def execute_list(args):
    """执行列出策略命令"""
    strategy_manager = StrategyManager()
    strategies = strategy_manager.list_strategies()
    
    if not strategies:
        logger.info("未找到任何策略")
        return
    
    # 如果指定了过滤条件
    if args.filter:
        filter_text = args.filter.lower()
        strategies = [s for s in strategies if filter_text in s['name'].lower() 
                     or filter_text in s.get('description', '').lower()
                     or filter_text in s['strategy_id'].lower()]
    
    if not strategies:
        logger.info("未找到匹配的策略")
        return
    
    print(f"\n共找到 {len(strategies)} 个策略:")
    
    for i, strategy in enumerate(strategies):
        print(f"\n{i+1}. {strategy['strategy_id']} - {strategy['name']}")
        
        if args.all:
            if 'description' in strategy and strategy['description']:
                print(f"   描述: {strategy['description']}")
            
            condition_count = len(strategy.get('conditions', []))
            print(f"   条件数: {condition_count}")
            
            if 'filters' in strategy and strategy['filters']:
                print("   过滤器:")
                for key, value in strategy['filters'].items():
                    print(f"     {key}: {value}")


def main():
    """主函数"""
    args = parse_args()
    
    if args.command == "combine":
        execute_combine(args)
    elif args.command == "watch":
        execute_watch(args)
    elif args.command == "detail":
        execute_detail(args)
    elif args.command == "list":
        execute_list(args)
    else:
        print("请指定要执行的命令。使用 --help 查看帮助信息。")


if __name__ == "__main__":
    main() 