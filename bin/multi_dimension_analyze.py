#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import sys
import json
import argparse
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from analysis.multi_dimension_analyzer import MultiDimensionAnalyzer
from utils.logger import get_logger
from utils.path_utils import get_result_dir

# 获取日志记录器
logger = get_logger(__name__)

def main():
    """命令行入口函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="多维度分析工具")
    
    # 添加子命令
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # 单股分析子命令
    single_parser = subparsers.add_parser('single_stock', help='单个股票分析')
    single_parser.add_argument('-s', '--stock', required=True, help='股票代码')
    single_parser.add_argument('-d', '--date', help='分析日期，格式YYYYMMDD，默认为当天')
    single_parser.add_argument('-p', '--periods', default='DAILY,WEEKLY', 
                            help='分析周期，用逗号分隔，如DAILY,WEEKLY')
    single_parser.add_argument('-o', '--output', help='输出文件路径')
    single_parser.add_argument('-f', '--format', choices=['json', 'markdown', 'excel'], 
                             default='json', help='输出格式')
    
    # 股票组分析子命令
    group_parser = subparsers.add_parser('stock_group', help='股票组分析')
    group_parser.add_argument('-f', '--file', required=True, help='股票代码文件路径，每行一个股票代码')
    group_parser.add_argument('-d', '--date', help='分析日期，格式YYYYMMDD，默认为当天')
    group_parser.add_argument('-p', '--periods', default='DAILY', 
                           help='分析周期，用逗号分隔，如DAILY,WEEKLY')
    group_parser.add_argument('-o', '--output', help='输出文件路径')
    group_parser.add_argument('--format', choices=['json', 'markdown', 'excel'], 
                            default='json', help='输出格式')
    
    # 市场相关性分析子命令
    correlation_parser = subparsers.add_parser('correlation', help='市场相关性分析')
    correlation_parser.add_argument('-f', '--file', required=True, help='股票代码文件路径，每行一个股票代码')
    correlation_parser.add_argument('-d', '--date', help='分析结束日期，格式YYYYMMDD，默认为当天')
    correlation_parser.add_argument('--days', type=int, default=30, help='分析的天数')
    correlation_parser.add_argument('-o', '--output', help='输出文件路径')
    correlation_parser.add_argument('--format', choices=['json', 'markdown', 'excel'], 
                                  default='json', help='输出格式')
    
    args = parser.parse_args()
    
    # 创建多维度分析器
    analyzer = MultiDimensionAnalyzer()
    
    # 处理日期参数，默认为当天
    if not hasattr(args, 'date') or not args.date:
        args.date = datetime.now().strftime("%Y%m%d")
    
    # 处理输出文件参数，默认为结果目录下的文件
    if not hasattr(args, 'output') or not args.output:
        result_dir = get_result_dir()
        os.makedirs(result_dir, exist_ok=True)
        args.output = os.path.join(result_dir, f"multi_dimension_{args.command}_{args.date}.{args.format}")
    
    if args.command == 'single_stock':
        # 解析周期参数
        periods = [p.strip() for p in args.periods.split(',') if p.strip()]
        
        # 执行单股分析
        result = analyzer.analyze_single_stock(args.stock, args.date, periods)
        
        # 输出结果
        if args.format == 'json' or args.format == 'markdown':
            analyzer.save_results(args.output, args.format)
        elif args.format == 'excel':
            analyzer.export_to_excel(args.output)
        
        # 打印关键信息
        assessment = result.get("assessment", {})
        if assessment:
            print(f"分析结果摘要: {assessment.get('summary', '')}")
            print(f"推荐: {assessment.get('recommendation', '')}")
            print(f"综合评分: {assessment.get('overall_score', 0)}")
        
        print(f"详细结果已保存到: {args.output}")
        
    elif args.command == 'stock_group':
        # 读取股票代码文件
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                stock_codes = [line.strip() for line in f if line.strip()]
        except Exception as e:
            logger.error(f"读取股票代码文件时出错: {e}")
            return
        
        if not stock_codes:
            logger.error("股票代码列表为空")
            return
        
        # 解析周期参数
        periods = [p.strip() for p in args.periods.split(',') if p.strip()]
        
        # 执行股票组分析
        result = analyzer.analyze_stock_group(stock_codes, args.date, periods)
        
        # 输出结果
        if args.format == 'json' or args.format == 'markdown':
            analyzer.save_results(args.output, args.format)
        elif args.format == 'excel':
            analyzer.export_to_excel(args.output)
        
        # 打印关键信息
        assessment = result.get("assessment", {})
        if assessment:
            print(f"分析结果摘要: {assessment.get('summary', '')}")
            print(f"推荐: {assessment.get('recommendation', '')}")
        
        print(f"详细结果已保存到: {args.output}")
        
    elif args.command == 'correlation':
        # 读取股票代码文件
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                stock_codes = [line.strip() for line in f if line.strip()]
        except Exception as e:
            logger.error(f"读取股票代码文件时出错: {e}")
            return
        
        if not stock_codes:
            logger.error("股票代码列表为空")
            return
        
        # 执行市场相关性分析
        result = analyzer.analyze_market_correlation(stock_codes, args.date, args.days)
        
        # 输出结果
        if args.format == 'json' or args.format == 'markdown':
            analyzer.save_results(args.output, args.format)
        elif args.format == 'excel':
            analyzer.export_to_excel(args.output)
        
        # 打印关键信息
        avg_correlations = result.get("avg_correlations", {})
        if avg_correlations:
            print("平均相关系数:")
            for index_code, corr in sorted(avg_correlations.items(), key=lambda x: abs(x[1]), reverse=True):
                print(f"  {index_code}: {corr:.4f}")
        
        print(f"详细结果已保存到: {args.output}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 