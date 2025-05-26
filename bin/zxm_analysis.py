#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZXM体系分析命令行工具

基于ZXM体系分析股票买点和吸筹形态
"""

import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from db.clickhouse_db import get_clickhouse_db
from indicators.factory import IndicatorFactory
from indicators.pattern.zxm_patterns import ZXMPatternIndicator
from enums.indicator_types import IndicatorType, TimeFrame
from utils.logger import setup_logging
from utils import path_utils
from utils import date_utils


def setup_argparser() -> argparse.ArgumentParser:
    """设置命令行参数解析器"""
    parser = argparse.ArgumentParser(description="ZXM体系分析工具")
    
    parser.add_argument("--stock", "-s", type=str, required=True,
                        help="股票代码，例如：000001.SZ")
    
    parser.add_argument("--start-date", "-sd", type=str, default=None,
                        help="开始日期，格式：YYYY-MM-DD，默认为180天前")
    
    parser.add_argument("--end-date", "-ed", type=str, default=None,
                        help="结束日期，格式：YYYY-MM-DD，默认为今天")
    
    parser.add_argument("--timeframe", "-tf", type=str, default="daily",
                        choices=["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"],
                        help="时间周期，默认为日线")
    
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="输出文件路径，默认为results/zxm_analysis_{stock_code}_{timeframe}_{date}.csv")
    
    parser.add_argument("--detailed", "-d", action="store_true",
                        help="是否输出详细分析结果")
    
    parser.add_argument("--multi-timeframe", "-mt", action="store_true",
                        help="是否进行多周期联合分析")
    
    return parser


def get_stock_data(stock_code: str, start_date: str, end_date: str, 
                   timeframe: str) -> pd.DataFrame:
    """
    获取股票数据
    
    Args:
        stock_code: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        timeframe: 时间周期
        
    Returns:
        包含OHLCV数据的DataFrame
    """
    db = get_clickhouse_db()
    
    # 构建查询语句
    if timeframe == "daily":
        table = "stock_daily_data"
        time_column = "trade_date"
    elif timeframe == "weekly":
        table = "stock_weekly_data"
        time_column = "trade_date"
    elif timeframe == "monthly":
        table = "stock_monthly_data"
        time_column = "trade_date"
    else:
        # 分钟级数据
        table = "stock_min_data"
        time_column = "trade_time"
    
    query = f"""
    SELECT 
        {time_column} as date,
        open,
        high,
        low,
        close,
        volume
    FROM {table}
    WHERE ts_code = %(stock_code)s
    """
    
    # 添加时间过滤条件
    if timeframe in ["daily", "weekly", "monthly"]:
        query += """
        AND trade_date >= %(start_date)s
        AND trade_date <= %(end_date)s
        """
    else:
        query += """
        AND trade_time >= %(start_date)s
        AND trade_time <= %(end_date)s
        """
        # 分钟级数据需要按指定的周期聚合
        interval_map = {
            "1min": 1,
            "5min": 5,
            "15min": 15,
            "30min": 30,
            "60min": 60
        }
        
        if timeframe in interval_map:
            query += f"""
            AND minute_freq = {interval_map[timeframe]}
            """
    
    query += f"""
    ORDER BY {time_column} ASC
    """
    
    # 执行查询
    params = {
        "stock_code": stock_code,
        "start_date": start_date.replace('-', ''),
        "end_date": end_date.replace('-', '')
    }
    
    df = db.query_to_dataframe(query, params)
    
    # 检查是否有数据
    if df.empty:
        logging.error(f"未找到股票 {stock_code} 在指定时间范围和周期下的数据")
        sys.exit(1)
    
    return df


def analyze_zxm_patterns(data: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    分析ZXM体系买点和吸筹形态
    
    Args:
        data: 包含OHLCV数据的DataFrame
        
    Returns:
        包含各种形态识别结果的字典
    """
    # 创建ZXM模式识别器
    zxm_indicator = ZXMPatternIndicator()
    
    # 计算结果
    result = zxm_indicator.calculate(
        data['open'].values,
        data['high'].values,
        data['low'].values,
        data['close'].values,
        data['volume'].values
    )
    
    return result


def format_results(data: pd.DataFrame, results: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    格式化分析结果
    
    Args:
        data: 原始数据DataFrame
        results: ZXM分析结果
        
    Returns:
        包含分析结果的DataFrame
    """
    # 创建结果DataFrame
    result_df = data.copy()
    
    # 添加识别结果
    for key, value in results.items():
        result_df[key] = value
    
    # 计算所有买点信号
    buy_columns = [col for col in result_df.columns if 'buy' in col]
    if buy_columns:
        result_df['any_buy_signal'] = result_df[buy_columns].any(axis=1)
    
    # 计算所有吸筹信号
    absorption_columns = [col for col in result_df.columns if col not in buy_columns and col not in data.columns]
    if absorption_columns:
        result_df['any_absorption_signal'] = result_df[absorption_columns].any(axis=1)
    
    return result_df


def analyze_multi_timeframe(stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    进行多周期联合分析
    
    Args:
        stock_code: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        
    Returns:
        包含多周期分析结果的DataFrame
    """
    # 定义要分析的周期
    timeframes = ["daily", "60min", "15min"]
    
    # 存储各周期分析结果
    results = {}
    
    # 分析每个周期
    for tf in timeframes:
        # 获取数据
        df = get_stock_data(stock_code, start_date, end_date, tf)
        
        # 分析ZXM形态
        patterns = analyze_zxm_patterns(df)
        
        # 格式化结果
        result_df = format_results(df, patterns)
        
        # 存储结果
        results[tf] = result_df
    
    # 找到日线结果中有买点的日期
    daily_buy_dates = results["daily"][results["daily"]["any_buy_signal"]]["date"].tolist()
    
    # 创建多周期共振分析结果
    resonance_df = pd.DataFrame()
    
    if daily_buy_dates:
        # 对每个日线买点，检查60分钟和15分钟级别是否也有买点
        for date in daily_buy_dates:
            # 日期格式转换
            if isinstance(date, str):
                date_str = date
                date_obj = datetime.strptime(date, "%Y%m%d")
            else:
                date_str = date.strftime("%Y%m%d")
                date_obj = date
            
            # 找到这一天的60分钟和15分钟数据
            next_day = (date_obj + timedelta(days=1)).strftime("%Y%m%d")
            
            # 60分钟买点
            min60_data = results["60min"]
            min60_date_mask = (min60_data["date"] >= date_str) & (min60_data["date"] < next_day)
            has_60min_buy = min60_data[min60_date_mask]["any_buy_signal"].any()
            
            # 15分钟买点
            min15_data = results["15min"]
            min15_date_mask = (min15_data["date"] >= date_str) & (min15_data["date"] < next_day)
            has_15min_buy = min15_data[min15_date_mask]["any_buy_signal"].any()
            
            # 记录结果
            resonance_df = resonance_df.append({
                "date": date_str,
                "daily_buy": True,
                "60min_buy": has_60min_buy,
                "15min_buy": has_15min_buy,
                "resonance_level": 1 + int(has_60min_buy) + int(has_15min_buy)  # 1-3的共振等级
            }, ignore_index=True)
    
    return resonance_df


def main():
    """主函数"""
    # 解析命令行参数
    parser = setup_argparser()
    args = parser.parse_args()
    
    # 设置日志
    setup_logging()
    
    # 处理日期参数
    if args.start_date is None:
        start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
    else:
        start_date = args.start_date
    
    if args.end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    else:
        end_date = args.end_date
    
    # 股票代码格式化
    stock_code = args.stock.upper()
    
    # 输出文件路径
    if args.output is None:
        today = datetime.now().strftime("%Y%m%d")
        output_dir = path_utils.get_result_dir()
        output_file = os.path.join(output_dir, f"zxm_analysis_{stock_code}_{args.timeframe}_{today}.csv")
    else:
        output_file = args.output
    
    logging.info(f"分析股票: {stock_code}")
    logging.info(f"时间范围: {start_date} 至 {end_date}")
    logging.info(f"时间周期: {args.timeframe}")
    
    # 多周期分析
    if args.multi_timeframe:
        logging.info("进行多周期联合分析...")
        result_df = analyze_multi_timeframe(stock_code, start_date, end_date)
        
        # 输出共振分析结果
        if not result_df.empty:
            # 保存到文件
            result_df.to_csv(output_file, index=False)
            logging.info(f"多周期共振分析结果已保存到: {output_file}")
            
            # 显示结果概要
            resonance_count = result_df["resonance_level"].value_counts().to_dict()
            logging.info("多周期共振分析结果概要:")
            for level, count in sorted(resonance_count.items()):
                logging.info(f"共振等级 {level}: {count} 个买点")
        else:
            logging.info("未发现多周期共振买点")
    
    # 单周期分析
    else:
        # 获取股票数据
        data = get_stock_data(stock_code, start_date, end_date, args.timeframe)
        logging.info(f"获取到 {len(data)} 条记录")
        
        # 分析ZXM形态
        logging.info("分析ZXM体系形态...")
        results = analyze_zxm_patterns(data)
        
        # 格式化结果
        result_df = format_results(data, results)
        
        # 保存结果
        result_df.to_csv(output_file, index=False)
        logging.info(f"分析结果已保存到: {output_file}")
        
        # 显示结果概要
        buy_signals = result_df["any_buy_signal"].sum() if "any_buy_signal" in result_df.columns else 0
        absorption_signals = result_df["any_absorption_signal"].sum() if "any_absorption_signal" in result_df.columns else 0
        
        logging.info("分析结果概要:")
        logging.info(f"共识别出 {buy_signals} 个买点信号")
        logging.info(f"共识别出 {absorption_signals} 个吸筹形态信号")
        
        # 显示详细结果
        if args.detailed and not result_df.empty:
            logging.info("详细买点信号:")
            buy_columns = [col for col in result_df.columns if 'buy' in col and col != 'any_buy_signal']
            for col in buy_columns:
                signal_count = result_df[col].sum()
                if signal_count > 0:
                    logging.info(f"- {col}: {signal_count} 个信号")
            
            logging.info("详细吸筹形态信号:")
            absorption_columns = [col for col in result_df.columns if col not in buy_columns and 
                                 col not in data.columns and col != 'any_absorption_signal']
            for col in absorption_columns:
                signal_count = result_df[col].sum()
                if signal_count > 0:
                    logging.info(f"- {col}: {signal_count} 个信号")


if __name__ == "__main__":
    main() 