#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
运行形态识别分析与回测

集成形态识别分析和回测功能，提供统一的命令行接口
"""

import os
import sys
import json
from datetime import datetime, timedelta
import pandas as pd
import argparse
import logging

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from analysis.pattern_recognition_analyzer import PatternRecognitionAnalyzer
from scripts.backtest.pattern_backtest import PatternBacktester
from db.clickhouse_db import get_clickhouse_db
from utils.logger import get_logger
from utils.path_utils import get_result_path, ensure_dir

logger = get_logger("pattern_analysis_backtest")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="运行形态识别分析和回测")
    
    # 股票参数
    parser.add_argument("--stock", type=str, help="股票代码，多个股票用逗号分隔")
    parser.add_argument("--stock_list", type=str, help="股票代码列表文件路径")
    parser.add_argument("--index", type=str, help="指数代码，用于分析和回测指数成分股")
    
    # 日期参数
    parser.add_argument("--start_date", type=str, help="开始日期，格式：YYYY-MM-DD")
    parser.add_argument("--end_date", type=str, help="结束日期，格式：YYYY-MM-DD")
    parser.add_argument("--days", type=int, default=90, help="回溯天数，默认90天")
    
    # 指标参数
    parser.add_argument("--indicators", type=str, default="MACD,KDJ,RSI", 
                       help="要分析的指标，多个指标用逗号分隔")
    
    # 周期参数
    parser.add_argument("--periods", type=str, default="DAILY", 
                       help="要分析的周期，多个周期用逗号分隔")
    
    # 回测参数
    parser.add_argument("--forward_days", type=int, default=5, 
                       help="回测向前看的天数，默认5天")
    parser.add_argument("--threshold", type=float, default=0.02, 
                       help="回测成功阈值，默认2%")
    
    # 输出参数
    parser.add_argument("--output", type=str, help="输出结果文件路径")
    parser.add_argument("--verbose", action="store_true", help="是否输出详细信息")
    
    # 模式参数
    parser.add_argument("--mode", type=str, choices=["analyze", "backtest", "both"], 
                       default="both", help="运行模式：仅分析，仅回测，或两者都运行")
    
    # 解析参数
    args = parser.parse_args()
    
    # 参数验证
    if not args.stock and not args.stock_list and not args.index:
        parser.error("必须指定至少一个参数：--stock, --stock_list 或 --index")
    
    # 设置默认日期
    if not args.end_date:
        args.end_date = datetime.now().strftime("%Y-%m-%d")
    
    if not args.start_date:
        start_date = datetime.strptime(args.end_date, "%Y-%m-%d") - timedelta(days=args.days)
        args.start_date = start_date.strftime("%Y-%m-%d")
    
    return args


def get_stock_list(args):
    """获取股票列表"""
    stocks = []
    db = get_clickhouse_db()
    
    # 从单个股票参数获取
    if args.stock:
        stocks.extend(args.stock.split(','))
    
    # 从股票列表文件获取
    if args.stock_list:
        try:
            with open(args.stock_list, 'r', encoding='utf-8') as f:
                for line in f:
                    stock = line.strip()
                    if stock and not stock.startswith('#'):
                        stocks.append(stock)
        except Exception as e:
            logger.error(f"读取股票列表文件 {args.stock_list} 出错: {e}")
    
    # 从指数成分股获取
    if args.index:
        try:
            index_stocks = db.get_index_stocks(args.index)
            stocks.extend(index_stocks)
        except Exception as e:
            logger.error(f"获取指数 {args.index} 成分股出错: {e}")
    
    # 去重
    stocks = list(set(stocks))
    
    # 过滤非法股票代码
    valid_stocks = []
    for stock in stocks:
        if db.is_valid_stock(stock):
            valid_stocks.append(stock)
        else:
            logger.warning(f"股票代码 {stock} 无效，已忽略")
    
    return valid_stocks


def run_pattern_analysis(stock_code, args):
    """运行形态识别分析"""
    # 初始化分析器
    analyzer = PatternRecognitionAnalyzer(
        indicators=args.indicators.split(','),
        periods=args.periods.split(',')
    )
    
    # 获取数据
    db = get_clickhouse_db()
    periods_data = {}
    
    # 为每个周期获取数据
    for period in args.periods.split(','):
        try:
            period_data = db.get_kline_data(
                stock_code=stock_code,
                start_date=args.start_date,
                end_date=args.end_date,
                period=period
            )
            
            if period_data is not None and not period_data.empty:
                periods_data[period] = period_data
                logger.info(f"获取到 {stock_code} {period} 周期的数据 {len(period_data)} 条")
            else:
                logger.warning(f"未获取到 {stock_code} {period} 周期的数据")
        except Exception as e:
            logger.error(f"获取 {stock_code} {period} 周期数据时出错: {e}")
    
    if not periods_data:
        logger.error(f"未获取到 {stock_code} 的任何数据，跳过分析")
        return None
    
    # 获取股票名称
    stock_name = db.get_stock_name(stock_code)
    
    # 运行分析
    result = analyzer.analyze(
        data=periods_data,
        stock_code=stock_code,
        stock_name=stock_name
    )
    
    return result


def run_backtest(stock_codes, args):
    """运行形态回测"""
    # 初始化回测器
    backtester = PatternBacktester(
        indicators=args.indicators.split(','),
        periods=args.periods.split(',')
    )
    
    # 运行多股票回测
    results = backtester.backtest_multiple_stocks(
        stock_codes=stock_codes,
        start_date=args.start_date,
        end_date=args.end_date,
        forward_days=args.forward_days,
        threshold=args.threshold
    )
    
    return results


def save_results(results, filepath, prefix=None):
    """保存结果到文件"""
    # 如果未指定文件名，则生成默认文件名
    if not filepath:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if prefix:
            filepath = f"data/result/{prefix}_{timestamp}.json"
        else:
            filepath = f"data/result/results_{timestamp}.json"
    
    # 确保目录存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # 保存结果
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"已保存结果到 {filepath}")
    
    return filepath


def print_analysis_summary(result, stock_code):
    """打印分析摘要"""
    if not result:
        logger.warning(f"股票 {stock_code} 的分析结果为空")
        return
    
    # 获取评分摘要
    score_summary = result.get("scores", {})
    total_score = score_summary.get("total_score", 0)
    recommendation = score_summary.get("recommendation", "未知")
    
    # 获取形态数量
    pattern_count = 0
    for period, period_data in result.get("periods", {}).items():
        pattern_count += len(period_data.get("patterns", []))
    
    # 打印摘要
    print(f"\n===== 股票 {stock_code} 形态分析摘要 =====")
    print(f"总评分: {total_score:.2f}")
    print(f"推荐意见: {recommendation}")
    print(f"检测到 {pattern_count} 个形态")
    
    # 打印主要看涨和看跌形态
    bullish_patterns = score_summary.get("bullish_patterns", [])
    bearish_patterns = score_summary.get("bearish_patterns", [])
    
    if bullish_patterns:
        print("\n主要看涨形态:")
        for pattern in bullish_patterns[:3]:  # 显示前3个看涨形态
            print(f"  - {pattern.get('display_name', '未知')} (强度: {pattern.get('strength', 0):.1f})")
    
    if bearish_patterns:
        print("\n主要看跌形态:")
        for pattern in bearish_patterns[:3]:  # 显示前3个看跌形态
            print(f"  - {pattern.get('display_name', '未知')} (强度: {pattern.get('strength', 0):.1f})")


def print_backtest_summary(results):
    """打印回测摘要"""
    if not results:
        logger.warning("回测结果为空")
        return
    
    summary = results.get("summary", {})
    
    print("\n===== 形态回测摘要 =====")
    print(f"股票数量: {summary.get('total_stocks', 0)}")
    print(f"买点数量: {summary.get('total_buy_points', 0)}")
    print(f"成功率: {summary.get('overall_success_rate', 0) * 100:.2f}%")
    print(f"平均收益: {summary.get('overall_avg_gain', 0) * 100:.2f}%")
    
    # 打印表现最好的形态
    top_patterns = summary.get("top_patterns_by_success_rate", [])
    
    if top_patterns:
        print("\n表现最好的形态 (成功率):")
        for i, pattern in enumerate(top_patterns[:5]):  # 显示前5个形态
            success_rate = pattern.get("success_rate", 0) * 100
            count = pattern.get("count", 0)
            display_name = pattern.get("display_name", "未知")
            period = pattern.get("period", "未知")
            print(f"  {i+1}. {display_name} ({period}): {success_rate:.2f}% ({count}次)")
    
    # 打印按平均收益排序的形态
    top_gain_patterns = summary.get("top_patterns_by_avg_gain", [])
    
    if top_gain_patterns:
        print("\n表现最好的形态 (平均收益):")
        for i, pattern in enumerate(top_gain_patterns[:5]):  # 显示前5个形态
            avg_gain = pattern.get("avg_gain", 0) * 100
            count = pattern.get("count", 0)
            display_name = pattern.get("display_name", "未知")
            period = pattern.get("period", "未知")
            print(f"  {i+1}. {display_name} ({period}): {avg_gain:.2f}% ({count}次)")
    
    # 打印各指标的表现
    indicator_stats = summary.get("indicator_stats", {})
    
    if indicator_stats:
        print("\n各指标表现:")
        for indicator, stats in indicator_stats.items():
            success_rate = stats.get("success_rate", 0) * 100
            avg_gain = stats.get("avg_gain", 0) * 100
            count = stats.get("count", 0)
            print(f"  {indicator}: 成功率 {success_rate:.2f}%, 平均收益 {avg_gain:.2f}% ({count}次)")


def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志级别
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # 获取股票列表
    stock_codes = get_stock_list(args)
    
    if not stock_codes:
        logger.error("没有有效的股票代码，程序退出")
        return
    
    logger.info(f"将处理 {len(stock_codes)} 只股票")
    
    # 根据模式运行
    if args.mode in ["analyze", "both"]:
        # 运行形态识别分析
        analysis_results = {}
        
        for stock_code in stock_codes:
            try:
                logger.info(f"分析股票 {stock_code}")
                result = run_pattern_analysis(stock_code, args)
                
                if result:
                    analysis_results[stock_code] = result
                    
                    # 如果是详细模式，打印分析摘要
                    if args.verbose:
                        print_analysis_summary(result, stock_code)
            except Exception as e:
                logger.error(f"分析股票 {stock_code} 时出错: {e}")
        
        # 保存分析结果
        if analysis_results:
            save_results(analysis_results, args.output, "analysis")
    
    if args.mode in ["backtest", "both"]:
        # 运行形态回测
        try:
            logger.info("开始回测")
            backtest_results = run_backtest(stock_codes, args)
            
            # 保存回测结果
            if backtest_results:
                save_results(backtest_results, args.output, "backtest")
                
                # 打印回测摘要
                print_backtest_summary(backtest_results)
        except Exception as e:
            logger.error(f"回测时出错: {e}")
    
    logger.info("处理完成")


if __name__ == "__main__":
    main() 