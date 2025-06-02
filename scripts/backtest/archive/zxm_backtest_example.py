#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import os

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

import argparse
import datetime
import pandas as pd
import numpy as np
import json
from typing import List, Dict, Any, Optional, Tuple, Union

from enums.kline_period import KlinePeriod
from utils.logger import get_logger
from utils.path_utils import get_backtest_result_dir
from db.db_manager import DBManager
from unified_backtest import UnifiedBacktest

# 获取日志记录器
logger = get_logger(__name__)

def zxm_backtest_single_stock(code: str, buy_date: str, output_file: str = None, pattern_type: str = "ZXM选股"):
    """
    使用ZXM体系指标对单只股票进行回测分析
    
    Args:
        code: 股票代码
        buy_date: 买点日期，格式为YYYYMMDD
        output_file: 输出文件名，默认为None，表示使用默认文件名
        pattern_type: 买点类型描述
    """
    try:
        logger.info(f"开始对股票 {code} 进行ZXM体系回测分析，买点日期：{buy_date}")
        
        # 创建统一回测系统实例
        backtest = UnifiedBacktest()
        
        # 分析股票
        result = backtest.analyze_stock(code, buy_date, pattern_type)
        
        # 如果未指定输出文件，则使用默认文件名
        if output_file is None:
            result_dir = get_backtest_result_dir()
            output_file = os.path.join(result_dir, f"zxm_backtest_{code}_{buy_date}.json")
        
        # 保存结果
        backtest.save_results(output_file)
        
        # 显示ZXM相关指标分析结果
        if result and 'periods' in result and 'daily' in result['periods']:
            daily_result = result['periods']['daily']
            
            if 'indicators' in daily_result:
                indicators = daily_result['indicators']
                
                print_zxm_indicators_results(result)
            else:
                print(f"未找到ZXM体系指标分析结果")
        else:
            print(f"分析失败或未找到日线周期分析结果")
        
        return result
    
    except Exception as e:
        logger.error(f"对股票 {code} 进行ZXM体系回测分析时出错: {e}")
        return None


def zxm_backtest_from_file(input_file: str, output_file: str = None, pattern_type: str = "ZXM选股"):
    """
    从文件中批量读取股票信息进行ZXM体系回测分析
    
    Args:
        input_file: 输入文件路径，包含股票代码和买点日期
        output_file: 输出文件路径，默认为None，表示使用默认文件名
        pattern_type: 买点类型描述
    """
    try:
        logger.info(f"开始从文件 {input_file} 批量读取股票进行ZXM体系回测分析")
        
        # 创建统一回测系统实例
        backtest = UnifiedBacktest()
        
        # 如果未指定输出文件，则使用默认文件名
        if output_file is None:
            result_dir = get_backtest_result_dir()
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(result_dir, f"zxm_batch_backtest_{timestamp}.json")
        
        # 从文件读取股票信息
        stocks = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split(',')
                if len(parts) >= 2:
                    code = parts[0].strip()
                    buy_date = parts[1].strip()
                    stocks.append((code, buy_date))
        
        # 批量分析股票
        for code, buy_date in stocks:
            try:
                logger.info(f"分析股票 {code}，买点日期：{buy_date}")
                backtest.analyze_stock(code, buy_date, pattern_type)
            except Exception as e:
                logger.error(f"分析股票 {code} 时出错: {e}")
        
        # 保存结果
        backtest.save_results(output_file)
        
        # 打印统计信息
        backtest._print_pattern_stats()
        
        # 按类别统计ZXM相关形态的出现次数
        zxm_trend_patterns = {pattern: count for pattern, count in backtest.pattern_stats.items() 
                             if pattern.startswith('ZXM') and ('趋势' in pattern or 'trend' in pattern.lower())}
        
        zxm_elasticity_patterns = {pattern: count for pattern, count in backtest.pattern_stats.items() 
                                  if pattern.startswith('ZXM') and ('弹性' in pattern or 'elastic' in pattern.lower())}
        
        zxm_buypoint_patterns = {pattern: count for pattern, count in backtest.pattern_stats.items() 
                                if pattern.startswith('ZXM') and ('买点' in pattern or 'buypoint' in pattern.lower())}
        
        # 打印ZXM形态统计
        print("\n=== ZXM体系形态统计 ===")
        
        if zxm_trend_patterns:
            print("\nZXM趋势形态:")
            for pattern, count in sorted(zxm_trend_patterns.items(), key=lambda x: x[1], reverse=True):
                percentage = count / len(stocks) * 100 if stocks else 0
                print(f"{pattern}: {count} 次 ({percentage:.2f}%)")
        
        if zxm_elasticity_patterns:
            print("\nZXM弹性形态:")
            for pattern, count in sorted(zxm_elasticity_patterns.items(), key=lambda x: x[1], reverse=True):
                percentage = count / len(stocks) * 100 if stocks else 0
                print(f"{pattern}: {count} 次 ({percentage:.2f}%)")
        
        if zxm_buypoint_patterns:
            print("\nZXM买点形态:")
            for pattern, count in sorted(zxm_buypoint_patterns.items(), key=lambda x: x[1], reverse=True):
                percentage = count / len(stocks) * 100 if stocks else 0
                print(f"{pattern}: {count} 次 ({percentage:.2f}%)")
        
        # 打印ZXM整体形态统计
        other_zxm_patterns = {pattern: count for pattern, count in backtest.pattern_stats.items() 
                            if pattern.startswith('ZXM') and pattern not in zxm_trend_patterns 
                            and pattern not in zxm_elasticity_patterns and pattern not in zxm_buypoint_patterns}
        
        if other_zxm_patterns:
            print("\nZXM其他形态:")
            for pattern, count in sorted(other_zxm_patterns.items(), key=lambda x: x[1], reverse=True):
                percentage = count / len(stocks) * 100 if stocks else 0
                print(f"{pattern}: {count} 次 ({percentage:.2f}%)")
        
        print(f"\n详细结果已保存至: {output_file}")
        
        return backtest.analysis_results
    
    except Exception as e:
        logger.error(f"批量ZXM体系回测分析时出错: {e}")
        return None


def print_zxm_indicators_results(result, title="ZXM体系指标分析结果"):
    """
    打印ZXM体系指标分析结果
    
    Args:
        result: 分析结果字典
        title: 标题
    """
    print("=" * 50)
    print(title)
    print("=" * 50)
    
    # 打印基本信息
    print(f"股票代码: {result.get('code', 'N/A')}")
    print(f"股票名称: {result.get('name', 'N/A')}")
    print(f"行业: {result.get('industry', 'N/A')}")
    print(f"买点日期: {result.get('buy_date', 'N/A')}")
    print(f"买点类型: {result.get('pattern_type', 'N/A')}")
    print(f"买点价格: {result.get('buy_price', 'N/A')}")
    print("-" * 50)
    
    # 打印ZXM指标结果
    if 'periods' in result:
        for period_name, period_data in result['periods'].items():
            if not period_data or 'indicators' not in period_data:
                continue
                
            period_label = {
                'daily': '日线',
                'min15': '15分钟',
                'min30': '30分钟',
                'min60': '60分钟',
                'weekly': '周线',
                'monthly': '月线'
            }.get(period_name, period_name)
            
            print(f"\n{period_label}周期指标:")
            print("-" * 40)
            
            indicators = period_data['indicators']
            
            # ZXM趋势指标
            if 'zxm_trend' in indicators:
                print(f"\n{period_label} ZXM趋势指标:")
                for trend_name, trend_data in indicators['zxm_trend'].items():
                    print(f"  - {trend_name}:")
                    for k, v in trend_data.items():
                        print(f"    - {k}: {v}")
            
            # ZXM弹性指标
            if 'zxm_elasticity' in indicators:
                print(f"\n{period_label} ZXM弹性指标:")
                for elasticity_name, elasticity_data in indicators['zxm_elasticity'].items():
                    print(f"  - {elasticity_name}:")
                    for k, v in elasticity_data.items():
                        print(f"    - {k}: {v}")
            
            # ZXM买点指标
            if 'zxm_buypoint' in indicators:
                print(f"\n{period_label} ZXM买点指标:")
                for buypoint_name, buypoint_data in indicators['zxm_buypoint'].items():
                    print(f"  - {buypoint_name}:")
                    for k, v in buypoint_data.items():
                        print(f"    - {k}: {v}")
            
            # ZXM得分
            if 'zxm_trend_score' in indicators:
                print(f"\n{period_label} ZXM得分:")
                print(f"  - 趋势得分: {indicators['zxm_trend_score']}")
                print(f"  - 弹性得分: {indicators['zxm_elasticity_score']}")
                print(f"  - 买点得分: {indicators['zxm_buypoint_score']}")
            
            # 识别的形态
            if 'patterns' in period_data and period_data['patterns']:
                print(f"\n{period_label}识别的形态:")
                for pattern in period_data['patterns']:
                    print(f"  - {pattern}")
    
    # 打印跨周期共性特征
    if 'cross_period_patterns' in result and result['cross_period_patterns']:
        print("\n跨周期共性特征:")
        for pattern in result['cross_period_patterns']:
            print(f"  - {pattern}")
    
    print("=" * 50)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='ZXM体系指标回测分析工具')
    parser.add_argument('--mode', type=str, choices=['single', 'batch'], default='single', help='分析模式：单只股票或批量分析')
    parser.add_argument('--code', type=str, help='股票代码（单只股票模式）')
    parser.add_argument('--date', type=str, help='买点日期，格式为YYYYMMDD（单只股票模式）')
    parser.add_argument('--input', type=str, help='输入文件路径，包含股票代码和买点日期（批量分析模式）')
    parser.add_argument('--output', type=str, help='输出文件路径')
    parser.add_argument('--pattern', type=str, default='ZXM选股', help='买点类型描述')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        if not args.code or not args.date:
            parser.error("单只股票模式下，--code和--date参数是必需的")
        
        zxm_backtest_single_stock(args.code, args.date, args.output, args.pattern)
    
    elif args.mode == 'batch':
        if not args.input:
            parser.error("批量分析模式下，--input参数是必需的")
        
        zxm_backtest_from_file(args.input, args.output, args.pattern)


if __name__ == "__main__":
    main() 