#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
高级K线形态识别测试工具

用于测试AdvancedCandlestickPatterns指标的识别功能
"""

import os
import sys
import argparse
import pandas as pd
from datetime import datetime, timedelta

# 将项目根目录添加到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from indicators.pattern.advanced_candlestick_patterns import AdvancedCandlestickPatterns, AdvancedPatternType
from db.data_manager import DataManager
from enums.period import Period  # 使用新的统一周期枚举
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='测试高级K线形态识别功能')
    
    parser.add_argument('-s', '--stock', required=True, help='股票代码')
    parser.add_argument('-p', '--period', default='DAILY', choices=[p.name for p in Period], 
                      help='K线周期，默认为日线')
    parser.add_argument('-d', '--days', type=int, default=120, help='历史数据天数，默认120天')
    parser.add_argument('-o', '--output', help='输出CSV文件路径，不指定则输出到控制台')
    parser.add_argument('-v', '--verbose', action='store_true', help='显示详细信息')
    
    return parser.parse_args()


def get_historical_data(stock_code, period, days):
    """
    获取历史K线数据
    
    Args:
        stock_code: 股票代码
        period: 周期
        days: 天数
        
    Returns:
        K线数据DataFrame
    """
    data_manager = DataManager()
    
    # 计算日期范围
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    
    # 获取K线数据
    data = data_manager.get_kline_data(
        stock_code=stock_code,
        period=period,
        start_date=start_date,
        end_date=end_date
    )
    
    return data


def test_pattern_recognition(data, verbose=False):
    """
    测试形态识别功能
    
    Args:
        data: K线数据
        verbose: 是否显示详细信息
        
    Returns:
        识别结果DataFrame
    """
    # 创建高级K线形态识别指标
    indicator = AdvancedCandlestickPatterns()
    
    # 计算指标
    result = indicator.calculate(data)
    
    # 生成信号
    signals = indicator.generate_signals(result)
    
    # 合并数据
    merged = pd.DataFrame(index=data.index)
    merged['date'] = data['date']
    merged['open'] = data['open']
    merged['high'] = data['high']
    merged['low'] = data['low']
    merged['close'] = data['close']
    
    # 添加信号列
    merged['buy_signal'] = signals['buy_signal']
    merged['sell_signal'] = signals['sell_signal']
    merged['watch_signal'] = signals['watch_signal']
    merged['signal_strength'] = signals['signal_strength']
    
    # 统计识别到的形态
    pattern_stats = {}
    for column in result.columns:
        if result[column].any():
            pattern_stats[column] = result[column].sum()
    
    # 添加形态列
    if verbose:
        for column in result.columns:
            merged[column] = result[column]
    else:
        # 只保留有信号的形态列
        for column, count in pattern_stats.items():
            merged[column] = result[column]
    
    # 添加形态描述列
    merged['patterns'] = ''
    for i in merged.index:
        patterns = []
        for column in result.columns:
            if result.loc[i, column]:
                patterns.append(column)
        merged.loc[i, 'patterns'] = ', '.join(patterns)
    
    return merged, pattern_stats


def main():
    """主函数"""
    args = parse_args()
    
    logger.info(f"开始测试股票 {args.stock} 的高级K线形态识别功能")
    
    # 获取历史数据
    period = Period[args.period]  # 使用统一的周期枚举
    data = get_historical_data(args.stock, period, args.days)
    
    if data.empty:
        logger.error(f"未找到股票 {args.stock} 的历史数据")
        sys.exit(1)
    
    logger.info(f"获取到 {len(data)} 条K线数据")
    
    # 测试形态识别
    result, pattern_stats = test_pattern_recognition(data, args.verbose)
    
    # 输出识别统计
    print("\n识别到的形态统计:")
    if pattern_stats:
        for pattern, count in sorted(pattern_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"{pattern}: {count}次")
    else:
        print("未识别到任何形态")
    
    # 筛选有信号的记录
    signal_records = result[result['buy_signal'] | result['sell_signal'] | result['watch_signal']]
    
    # 输出信号记录
    if not signal_records.empty:
        print(f"\n共找到 {len(signal_records)} 条信号记录:")
        
        for _, row in signal_records.iterrows():
            signal_type = "买入" if row['buy_signal'] else "卖出" if row['sell_signal'] else "观察"
            print(f"{row['date']} - {signal_type}信号，强度: {row['signal_strength']:.2f}, 形态: {row['patterns']}")
    else:
        print("\n未找到任何信号记录")
    
    # 保存结果
    if args.output:
        try:
            # 确保输出目录存在
            output_dir = os.path.dirname(args.output)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
            # 保存结果
            result.to_csv(args.output, index=False, encoding='utf-8-sig')
            logger.info(f"已保存结果到文件: {args.output}")
        except Exception as e:
            logger.error(f"保存结果到文件失败: {e}")
    
    logger.info("测试完成")


if __name__ == '__main__':
    main() 