#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
形态组合回测和策略生成

用于执行形态组合分析和回测，并基于回测结果生成选股策略
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
import pandas as pd

# 将项目根目录添加到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from scripts.backtest.advanced_backtest import AdvancedBacktester
from analysis.pattern_recognition_analyzer import PatternRecognitionAnalyzer
from db.clickhouse_db import get_clickhouse_db
from utils.logger import get_logger
from utils.date_utils import get_previous_trade_date, get_next_trade_date
from utils.file_utils import ensure_dir_exists

logger = get_logger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="形态组合回测和策略生成")
    
    parser.add_argument("--mode", choices=["backtest", "validate"], default="backtest",
                       help="执行模式：backtest-进行回测；validate-验证已有策略")
    
    parser.add_argument("--stock_list", type=str, default="",
                       help="股票代码列表文件路径，每行一个股票代码")
    
    parser.add_argument("--industry", type=str, default="",
                       help="按行业选择股票，如'银行'、'计算机'等")
    
    parser.add_argument("--index", type=str, default="",
                       help="按指数成分股选择股票，如'000300.SH'(沪深300)")
    
    parser.add_argument("--start_date", type=str, default="",
                       help="回测开始日期，格式：YYYY-MM-DD")
    
    parser.add_argument("--end_date", type=str, default="",
                       help="回测结束日期，格式：YYYY-MM-DD")
    
    parser.add_argument("--forward_days", type=int, default=5,
                       help="向前看多少天，默认5天")
    
    parser.add_argument("--min_pattern_strength", type=float, default=60.0,
                       help="最小形态强度，默认60.0")
    
    parser.add_argument("--require_multiple_indicators", action="store_true",
                       help="是否要求多指标确认")
    
    parser.add_argument("--min_success_rate", type=float, default=60.0,
                       help="生成策略时的最小成功率，默认60.0")
    
    parser.add_argument("--min_profit_factor", type=float, default=2.0,
                       help="生成策略时的最小盈亏比，默认2.0")
    
    parser.add_argument("--strategy_file", type=str, default="",
                       help="策略文件路径，用于验证模式")
    
    parser.add_argument("--validation_date", type=str, default="",
                       help="策略验证日期，格式：YYYY-MM-DD，默认为最近交易日")
    
    parser.add_argument("--output_dir", type=str, default="data/result/pattern_combo",
                       help="输出目录，默认为data/result/pattern_combo")
    
    args = parser.parse_args()
    
    # 设置默认值
    if not args.start_date:
        args.start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    
    if not args.end_date:
        args.end_date = datetime.now().strftime("%Y-%m-%d")
    
    if not args.validation_date and args.mode == "validate":
        db = get_clickhouse_db()
        args.validation_date = db.get_last_trade_date()
    
    return args


def get_stock_list(args):
    """根据参数获取股票列表"""
    db = get_clickhouse_db()
    
    if args.stock_list:
        # 从文件读取股票列表
        try:
            with open(args.stock_list, 'r') as f:
                stock_codes = [line.strip() for line in f if line.strip()]
            return stock_codes
        except Exception as e:
            logger.error(f"读取股票列表文件失败: {e}")
            return []
    
    elif args.industry:
        # 按行业获取股票
        try:
            stocks = db.get_stocks_by_industry(args.industry)
            return [row['stock_code'] for row in stocks]
        except Exception as e:
            logger.error(f"获取行业 {args.industry} 的股票列表失败: {e}")
            return []
    
    elif args.index:
        # 按指数成分股获取股票
        try:
            stocks = db.get_index_stocks(args.index)
            return [row['stock_code'] for row in stocks]
        except Exception as e:
            logger.error(f"获取指数 {args.index} 的成分股列表失败: {e}")
            return []
    
    else:
        # 默认使用沪深300成分股
        try:
            stocks = db.get_index_stocks("000300.SH")
            return [row['stock_code'] for row in stocks]
        except Exception as e:
            logger.error(f"获取沪深300成分股列表失败: {e}")
            return []


def run_backtest(args, stock_codes):
    """执行回测"""
    logger.info(f"开始执行形态组合回测，共 {len(stock_codes)} 只股票")
    
    # 创建回测器
    backtester = AdvancedBacktester()
    
    # 执行形态组合回测
    results = backtester.backtest_with_pattern_combination(
        stock_codes=stock_codes,
        start_date=args.start_date,
        end_date=args.end_date,
        forward_days=args.forward_days,
        min_pattern_strength=args.min_pattern_strength,
        require_multiple_indicators=args.require_multiple_indicators
    )
    
    # 保存回测结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backtest_file = os.path.join(args.output_dir, f"pattern_combo_backtest_{timestamp}.json")
    
    ensure_dir_exists(args.output_dir)
    
    try:
        with open(backtest_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"回测结果已保存到 {backtest_file}")
    except Exception as e:
        logger.error(f"保存回测结果失败: {e}")
    
    # 生成策略
    strategy = backtester.generate_strategy_from_backtest(
        backtest_results=results,
        min_success_rate=args.min_success_rate,
        min_profit_factor=args.min_profit_factor
    )
    
    # 保存策略
    strategy_file = os.path.join(args.output_dir, f"pattern_combo_strategy_{timestamp}.json")
    
    try:
        with open(strategy_file, 'w', encoding='utf-8') as f:
            json.dump(strategy, f, ensure_ascii=False, indent=2)
        logger.info(f"策略已保存到 {strategy_file}")
    except Exception as e:
        logger.error(f"保存策略失败: {e}")
    
    # 打印回测结果摘要
    print("\n回测结果摘要:")
    print(f"回测周期: {args.start_date} 至 {args.end_date}")
    print(f"前瞻天数: {args.forward_days}天")
    print(f"成功率: {results['performance']['success_rate']:.2f}%")
    print(f"平均收益: {results['performance']['avg_gain']:.2f}%")
    print(f"最大收益: {results['performance']['max_gain']:.2f}%")
    print(f"平均亏损: {results['performance']['avg_loss']:.2f}%")
    print(f"盈亏比: {results['performance']['profit_factor']:.2f}")
    print(f"形态组合数量: {len(results['pattern_combinations'])}")
    
    # 返回生成的策略和回测结果文件路径
    return strategy, strategy_file, results, backtest_file


def validate_strategy(args, stock_codes):
    """验证策略"""
    logger.info(f"开始验证策略，共 {len(stock_codes)} 只股票")
    
    if not args.strategy_file:
        logger.error("未指定策略文件")
        return None
    
    # 读取策略文件
    try:
        with open(args.strategy_file, 'r', encoding='utf-8') as f:
            strategy = json.load(f)
    except Exception as e:
        logger.error(f"读取策略文件失败: {e}")
        return None
    
    # 创建回测器
    backtester = AdvancedBacktester()
    
    # 验证策略
    validation_results = backtester.validate_strategy(
        strategy=strategy,
        stock_codes=stock_codes,
        date=args.validation_date
    )
    
    # 保存验证结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    validation_file = os.path.join(args.output_dir, f"strategy_validation_{timestamp}.json")
    
    ensure_dir_exists(args.output_dir)
    
    try:
        with open(validation_file, 'w', encoding='utf-8') as f:
            json.dump(validation_results, f, ensure_ascii=False, indent=2)
        logger.info(f"验证结果已保存到 {validation_file}")
    except Exception as e:
        logger.error(f"保存验证结果失败: {e}")
    
    # 打印验证结果摘要
    print("\n策略验证结果摘要:")
    print(f"验证日期: {args.validation_date}")
    print(f"总股票数: {validation_results['total_stocks']}")
    print(f"匹配股票数: {validation_results['match_count']}")
    print(f"匹配率: {validation_results['match_rate']:.2f}%")
    
    if validation_results['matched_stocks']:
        print("\n匹配的股票:")
        for stock in validation_results['matched_stocks'][:10]:  # 只显示前10个
            print(f"{stock['stock_code']} - {stock['stock_name']}")
        
        if len(validation_results['matched_stocks']) > 10:
            print(f"... 还有 {len(validation_results['matched_stocks']) - 10} 只股票")
    
    return validation_results, validation_file


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 获取股票列表
    stock_codes = get_stock_list(args)
    
    if not stock_codes:
        logger.error("股票列表为空，无法执行回测或验证")
        return
    
    logger.info(f"获取到 {len(stock_codes)} 只股票")
    
    # 根据模式执行相应操作
    if args.mode == "backtest":
        run_backtest(args, stock_codes)
    elif args.mode == "validate":
        validate_strategy(args, stock_codes)
    else:
        logger.error(f"不支持的模式: {args.mode}")


if __name__ == "__main__":
    main() 