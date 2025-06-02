#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
运行形态识别分析

对指定股票进行多指标形态识别分析，生成分析报告
"""

import os
import sys
import json
import pandas as pd
import argparse
from datetime import datetime, timedelta

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from analysis.pattern_recognition_analyzer import PatternRecognitionAnalyzer
from indicators.factory import IndicatorFactory
from db.clickhouse_db import get_clickhouse_db
from utils.logger import get_logger
from utils.path_utils import get_result_path, ensure_dir
from enums.kline_period import KlinePeriod

logger = get_logger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="形态识别分析工具")
    
    parser.add_argument("--stock", "-s", required=True, help="股票代码，例如：000001.SZ")
    parser.add_argument("--start", default=None, help="开始日期，格式：YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="结束日期，格式：YYYY-MM-DD")
    parser.add_argument("--periods", default="DAILY,WEEKLY", help="周期列表，多个周期用逗号分隔")
    parser.add_argument("--indicators", default="MACD,KDJ,RSI", help="指标列表，多个指标用逗号分隔")
    parser.add_argument("--days", type=int, default=120, help="获取最近多少天的数据")
    parser.add_argument("--output", "-o", default=None, help="输出文件路径")
    parser.add_argument("--detail", action="store_true", help="是否输出详细信息")
    
    return parser.parse_args()


def get_stock_data(stock_code, periods, start_date=None, end_date=None, days=120):
    """获取股票数据"""
    db = get_clickhouse_db()
    
    # 如果没有指定开始日期，则使用当前日期减去指定天数
    if not start_date:
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    
    # 如果没有指定结束日期，则使用当前日期
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    # 获取股票名称
    stock_name = db.get_stock_name(stock_code)
    
    # 按周期获取数据
    data = {}
    
    for period in periods:
        try:
            period_data = db.get_kline_data(
                stock_code=stock_code,
                start_date=start_date,
                end_date=end_date,
                period=period
            )
            
            if period_data is not None and not period_data.empty:
                # 确保数据按日期排序
                period_data = period_data.sort_index()
                data[period] = period_data
                logger.info(f"获取到 {stock_code} {period} 周期的数据 {len(period_data)} 条")
            else:
                logger.warning(f"未获取到 {stock_code} {period} 周期的数据")
        except Exception as e:
            logger.error(f"获取 {stock_code} {period} 周期的数据时出错: {e}")
    
    return data, stock_name


def main():
    """主函数"""
    args = parse_args()
    
    # 解析参数
    stock_code = args.stock
    start_date = args.start
    end_date = args.end
    days = args.days
    
    # 解析周期列表
    periods = [p.strip() for p in args.periods.split(",")]
    
    # 解析指标列表
    indicators = [i.strip() for i in args.indicators.split(",")]
    
    logger.info(f"开始分析股票 {stock_code}")
    logger.info(f"分析周期: {periods}")
    logger.info(f"分析指标: {indicators}")
    
    # 获取股票数据
    stock_data, stock_name = get_stock_data(
        stock_code=stock_code,
        periods=periods,
        start_date=start_date,
        end_date=end_date,
        days=days
    )
    
    if not stock_data:
        logger.error(f"未获取到 {stock_code} 的数据，退出分析")
        sys.exit(1)
    
    # 创建形态识别分析器
    analyzer = PatternRecognitionAnalyzer(indicators=indicators, periods=periods)
    
    # 运行分析
    results = analyzer.analyze(
        data=stock_data,
        stock_code=stock_code,
        stock_name=stock_name
    )
    
    # 获取评分摘要
    score_summary = analyzer.get_score_summary()
    
    # 输出结果摘要
    print("\n" + "="*60)
    print(f"股票: {stock_code} ({stock_name})")
    print(f"分析时间: {results['analysis_time']}")
    print("="*60)
    
    # 输出总评分和推荐
    print(f"\n综合评分: {score_summary['total_score']:.2f}/100")
    print(f"推荐意见: {score_summary['recommendation']}")
    print(f"评分置信度: {score_summary['confidence']:.2f}")
    
    # 输出各指标评分
    print("\n各指标评分:")
    for indicator, score in score_summary['by_indicator'].items():
        print(f"  - {indicator}: {score:.2f}/100")
    
    # 输出各周期评分
    print("\n各周期评分:")
    for period, score in score_summary['by_period'].items():
        print(f"  - {period}: {score:.2f}/100")
    
    # 输出最新形态
    latest_patterns = analyzer.get_latest_patterns(top_n=5)
    
    print("\n最新形态(Top 5):")
    for pattern in latest_patterns:
        signal_type = "看涨" if pattern['signal_type'] == "bullish" else "看跌" if pattern['signal_type'] == "bearish" else "中性"
        print(f"  - {pattern['display_name']} ({pattern['indicator_id']}/{pattern['period']}): {signal_type}")
    
    # 输出跨周期共同形态
    common_patterns = results['common_patterns']
    
    if common_patterns:
        print("\n跨周期共同形态:")
        for pattern in common_patterns[:3]:  # 只显示前3个
            periods_str = ", ".join(pattern['periods'])
            print(f"  - {pattern['display_name']} ({pattern['indicator_id']}): 出现在 {periods_str} 周期")
    
    # 如果指定了详细模式，输出更多信息
    if args.detail:
        print("\n\n详细形态信息:")
        
        for period, period_data in results['periods'].items():
            print(f"\n{period} 周期形态:")
            for pattern in period_data['patterns'][:10]:  # 限制显示数量
                print(f"  - {pattern['display_name']}")
    
    # 保存结果
    if args.output:
        output_path = args.output
    else:
        # 默认保存到results目录
        output_dir = get_result_path("pattern_analysis")
        ensure_dir(output_dir)
        output_path = os.path.join(
            output_dir, 
            f"{stock_code}_pattern_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
    
    analyzer.save_results(output_path)
    print(f"\n分析结果已保存到: {output_path}")


if __name__ == "__main__":
    main() 