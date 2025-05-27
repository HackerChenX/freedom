#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
选股命令行工具

提供命令行界面执行选股策略，支持各种参数配置
"""

import os
import sys
import argparse
import json
import yaml
import pandas as pd
from datetime import datetime
import time
import threading
from typing import Dict, List, Optional, Any

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from strategy.strategy_parser import StrategyParser
from strategy.strategy_executor import StrategyExecutor
from strategy.strategy_manager import StrategyManager
from strategy.signal_watcher import SignalWatcher
from strategy.result_filter import ResultFilter
from db.data_manager import DataManager
from utils.logger import get_logger, init_logging
from utils.path_utils import get_result_dir
from utils.visualization import (
    plot_selection_result_distribution,
    plot_technical_indicators,
    plot_condition_heatmap,
    create_html_report
)
from utils.exceptions import (
    StrategyExecutionError, 
    StrategyValidationError, 
    DataAccessError
)

logger = get_logger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="选股工具")
    
    # 基础参数
    parser.add_argument("--strategy", "-s", help="策略ID或策略文件路径", required=True)
    parser.add_argument("--output", "-o", help="输出文件路径", default=None)
    parser.add_argument("--format", "-f", help="输出格式 (csv, excel, json, html)", default="csv")
    parser.add_argument("--date", "-d", help="选股日期 (YYYY-MM-DD)", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--limit", "-l", type=int, help="限制结果数量", default=0)
    parser.add_argument("--watch", "-w", help="是否包含观察信号", action="store_true")
    parser.add_argument("--visualize", "-v", help="生成可视化报告", action="store_true")
    parser.add_argument("--threads", "-t", type=int, help="最大线程数", default=None)
    parser.add_argument("--log-level", help="日志级别 (DEBUG, INFO, WARNING, ERROR)", default="INFO")
    parser.add_argument("--config", "-c", help="配置文件路径", default=None)
    
    # 排序参数
    parser.add_argument("--sort-by", help="排序字段", default="signal_strength")
    parser.add_argument("--sort-order", help="排序顺序 (ASC, DESC)", default="DESC")
    
    # 过滤参数
    filter_group = parser.add_argument_group("过滤参数")
    filter_group.add_argument("--min-cap", type=float, help="最小市值 (亿元)", default=None)
    filter_group.add_argument("--max-cap", type=float, help="最大市值 (亿元)", default=None)
    filter_group.add_argument("--min-price", type=float, help="最小价格", default=None)
    filter_group.add_argument("--max-price", type=float, help="最大价格", default=None)
    filter_group.add_argument("--industry", help="行业名称 (逗号分隔)", default=None)
    filter_group.add_argument("--market", help="市场名称 (逗号分隔)", default=None)
    
    args = parser.parse_args()
    return args


def print_progress(progress: float, message: str):
    """打印进度条"""
    bar_length = 50
    filled_length = int(bar_length * progress)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    percent = progress * 100
    
    sys.stdout.write(f'\r进度: |{bar}| {percent:.1f}% {message}')
    sys.stdout.flush()
    
    if progress >= 1:
        sys.stdout.write('\n')


def progress_callback(progress: float, message: str):
    """进度回调函数"""
    print_progress(progress, message)


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    if not config_path or not os.path.exists(config_path):
        return {}
        
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.endswith('.json'):
                return json.load(f)
            elif config_path.endswith(('.yaml', '.yml')):
                return yaml.safe_load(f)
            else:
                logger.warning(f"不支持的配置文件格式: {config_path}")
                return {}
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return {}


def is_strategy_file(strategy_path: str) -> bool:
    """判断是否为策略文件路径"""
    if not strategy_path:
        return False
        
    # 检查是否为文件路径
    if os.path.exists(strategy_path) and os.path.isfile(strategy_path):
        # 检查文件扩展名
        ext = os.path.splitext(strategy_path)[1].lower()
        return ext in ['.json', '.yaml', '.yml']
        
    return False


def execute_strategy(args):
    """执行选股策略"""
    try:
        # 初始化组件
        strategy_manager = StrategyManager()
        strategy_executor = StrategyExecutor(max_workers=args.threads)
        signal_watcher = SignalWatcher()
        result_filter = ResultFilter()
        
        # 确定策略来源
        if is_strategy_file(args.strategy):
            # 从文件加载策略
            parser = StrategyParser()
            logger.info(f"从文件加载策略: {args.strategy}")
            strategy_plan = parser.parse_from_file(args.strategy)
            
            # 输出策略计划用于调试
            print("\n调试信息 - 策略计划:")
            print(f"策略ID: {strategy_plan.get('strategy_id')}")
            print(f"策略名称: {strategy_plan.get('name')}")
            print(f"条件数量: {len(strategy_plan.get('conditions', []))}")
            print("条件列表:")
            for i, cond in enumerate(strategy_plan.get('conditions', [])):
                print(f"  条件 {i+1}: {cond}")
            print("\n")
            
            # 执行策略
            result_df = strategy_executor.execute_strategy(
                strategy_plan=strategy_plan,
                end_date=args.date,
                progress_callback=progress_callback
            )
        else:
            # 从策略管理器获取策略
            logger.info(f"使用策略ID: {args.strategy}")
            
            # 执行策略
            result_df = strategy_executor.execute_strategy_by_id(
                strategy_id=args.strategy,
                strategy_manager=strategy_manager,
                end_date=args.date,
                progress_callback=progress_callback
            )
        
        if result_df.empty:
            logger.warning("选股结果为空")
            print("\n未找到符合条件的股票")
            return
        
        # 添加观察信号
        if args.watch:
            logger.info("添加观察信号")
            result_df = signal_watcher.add_watch_signals(result_df)
        
        # 应用过滤器
        filter_config = {
            'market_cap': {},
            'price': {},
        }
        
        if args.min_cap is not None:
            filter_config['market_cap']['min'] = args.min_cap
        
        if args.max_cap is not None:
            filter_config['market_cap']['max'] = args.max_cap
        
        if args.min_price is not None:
            filter_config['price']['min'] = args.min_price
        
        if args.max_price is not None:
            filter_config['price']['max'] = args.max_price
        
        if args.industry:
            filter_config['industry'] = args.industry.split(',')
        
        if args.market:
            filter_config['market'] = args.market.split(',')
        
        # 应用过滤器
        result_df = result_filter.apply_filters(result_df, filter_config)
        
        # 排序
        result_df = result_filter.sort_results(
            result_df, 
            sort_field=args.sort_by, 
            ascending=args.sort_order.upper() != 'DESC'
        )
        
        # 限制结果数量
        if args.limit > 0 and len(result_df) > args.limit:
            result_df = result_df.iloc[:args.limit]
        
        # 输出结果
        output_result(result_df, args)
        
        # 生成可视化报告
        if args.visualize:
            generate_visualization(result_df, args)
        
    except Exception as e:
        logger.error(f"执行策略出错: {e}")
        print(f"\n执行策略出错: {e}")
        sys.exit(1)


def output_result(result_df: pd.DataFrame, args):
    """输出结果"""
    # 确定输出文件路径
    if args.output:
        output_path = args.output
    else:
        # 生成默认输出路径
        result_dir = get_result_dir()
        os.makedirs(result_dir, exist_ok=True)
        
        date_str = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"stock_selection_{date_str}.{args.format}"
        output_path = os.path.join(result_dir, filename)
    
    # 输出到文件
    try:
        # 确保目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        # 根据格式输出
        if args.format.lower() == 'csv':
            result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        elif args.format.lower() == 'excel':
            result_df.to_excel(output_path, index=False, engine='openpyxl')
        elif args.format.lower() == 'json':
            # 处理非原生JSON类型
            result_json = result_df.copy()
            for col in result_json.columns:
                if result_json[col].dtype == 'object':
                    result_json[col] = result_json[col].apply(
                        lambda x: json.dumps(x) if not isinstance(x, str) else x
                    )
            
            result_json.to_json(output_path, orient='records', force_ascii=False, indent=4)
        elif args.format.lower() == 'html':
            result_df.to_html(output_path, index=False, encoding='utf-8')
        else:
            logger.warning(f"不支持的输出格式: {args.format}，使用CSV格式")
            result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        logger.info(f"结果已保存至: {output_path}")
        print(f"\n结果已保存至: {output_path}")
        
        # 输出摘要
        print(f"\n共找到 {len(result_df)} 只符合条件的股票:")
        
        # 显示前10只股票
        display_count = min(10, len(result_df))
        for i in range(display_count):
            row = result_df.iloc[i]
            print(f"{i+1}. {row['stock_code']} {row['stock_name']} "
                  f"(信号强度: {row['signal_strength']:.2f})")
        
        if len(result_df) > 10:
            print(f"... 及其他 {len(result_df) - 10} 只")
            
    except Exception as e:
        logger.error(f"输出结果失败: {e}")
        print(f"\n输出结果失败: {e}")


def generate_visualization(result_df: pd.DataFrame, args):
    """生成可视化报告"""
    try:
        print("\n生成可视化报告...")
        
        # 创建输出目录
        if args.output:
            output_dir = os.path.dirname(args.output)
        else:
            output_dir = get_result_dir()
            
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成报告日期
        date_str = datetime.now().strftime("%Y%m%d%H%M%S")
        report_name = f"选股报告_{args.strategy}_{date_str}"
        
        # 生成结果分布图
        print("生成结果分布图...")
        distribution_img = plot_selection_result_distribution(
            result_df=result_df,
            title=f"选股结果分布 ({args.strategy})"
        )
        
        # 生成条件热力图
        print("生成条件热力图...")
        heatmap_img = plot_condition_heatmap(
            result_df=result_df,
            title=f"条件满足热力图 ({args.strategy})"
        )
        
        # 创建HTML报告
        print("创建HTML报告...")
        charts = [distribution_img, heatmap_img]
        descriptions = [
            f"选股日期: {args.date}, 共找到 {len(result_df)} 只符合条件的股票",
            "条件满足热力图展示了每只股票满足的具体条件"
        ]
        
        report_path = os.path.join(output_dir, f"{report_name}.html")
        create_html_report(
            title=f"选股策略 {args.strategy} 执行报告",
            charts=charts,
            descriptions=descriptions,
            output_file=report_path
        )
        
        print(f"可视化报告已生成: {report_path}")
        
    except Exception as e:
        logger.error(f"生成可视化报告失败: {e}")
        print(f"生成可视化报告失败: {e}")


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志级别
    init_logging(level=args.log_level)
    
    # 加载配置文件
    if args.config:
        config = load_config(args.config)
        # 使用配置文件中的值覆盖命令行参数
        for key, value in config.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)
    
    # 执行策略
    execute_strategy(args)


if __name__ == "__main__":
    main() 