#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import os

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional

from formula import formula
from enums.kline_period import KlinePeriod
from utils.logger import get_logger
from indicators.factory import IndicatorFactory

logger = get_logger(__name__)

def analyze_15min_data(code="600585", date="20250408"):
    """
    分析15分钟数据
    
    Args:
        code: 股票代码
        date: 日期，格式为YYYYMMDD
    """
    try:
        # 日期前后范围
        start_date = "20250401"
        end_date = "20250430"
        
        logger.info(f"分析股票 {code} 的15分钟数据，日期范围: {start_date} - {end_date}")
        
        # 获取15分钟数据
        logger.info("获取15分钟数据...")
        stock_data = formula.StockData(code, KlinePeriod.MIN_15, start=start_date, end=end_date)
        
        if stock_data.history is None or len(stock_data.history) == 0:
            logger.error("未获取到15分钟数据")
            return
            
        # 将数据转换为DataFrame
        data = pd.DataFrame({
            'date': stock_data.history['date'],
            'datetime': stock_data.history['datetime'] if 'datetime' in stock_data.history.columns else None,
            'open': stock_data.open,
            'high': stock_data.high,
            'low': stock_data.low,
            'close': stock_data.close,
            'volume': stock_data.volume
        })
        
        # 计算技术指标
        logger.info("计算技术指标...")
        
        # 计算MACD
        macd_indicator = IndicatorFactory.create_indicator("MACD")
        macd_result = macd_indicator.compute(data)
        
        # 计算KDJ
        kdj_indicator = IndicatorFactory.create_indicator("KDJ")
        kdj_result = kdj_indicator.compute(data)
        
        # 计算RSI
        rsi_indicator = IndicatorFactory.create_indicator("RSI", periods=[6, 12, 24])
        rsi_result = rsi_indicator.compute(data)
        
        # 计算BOLL
        boll_indicator = IndicatorFactory.create_indicator("BOLL")
        boll_result = boll_indicator.compute(data)
        
        # 合并指标结果
        result = data.copy()
        for df in [macd_result, kdj_result, rsi_result, boll_result]:
            for col in df.columns:
                if col not in result.columns:
                    result[col] = df[col]
        
        # 分析指定日期的数据
        date_obj = datetime.datetime.strptime(date, "%Y%m%d").date()
        day_data = result[result['date'] == date_obj]
        
        if len(day_data) == 0:
            logger.warning(f"未找到 {date} 的数据")
            logger.info(f"可用日期: {result['date'].unique()}")
            return
            
        # 打印该日的15分钟数据
        logger.info(f"==== {date} 15分钟数据分析 ====")
        logger.info(f"数据条数: {len(day_data)}")
        
        # 打印开盘价、收盘价、最高价、最低价
        logger.info(f"开盘价: {day_data['open'].iloc[0]}")
        logger.info(f"收盘价: {day_data['close'].iloc[-1]}")
        logger.info(f"最高价: {day_data['high'].max()}")
        logger.info(f"最低价: {day_data['low'].min()}")
        
        # 打印MACD指标
        logger.info("==== MACD指标 ====")
        logger.info(f"DIF: {day_data['DIF'].iloc[-1]:.4f}")
        logger.info(f"DEA: {day_data['DEA'].iloc[-1]:.4f}")
        logger.info(f"MACD: {day_data['MACD'].iloc[-1]:.4f}")
        
        # 判断MACD是否金叉
        macd_cross = False
        for i in range(1, len(day_data)):
            if (day_data['DIF'].iloc[i-1] < day_data['DEA'].iloc[i-1] and 
                day_data['DIF'].iloc[i] > day_data['DEA'].iloc[i]):
                macd_cross = True
                logger.info(f"MACD金叉出现在: {day_data.iloc[i]['datetime']}")
                
        if not macd_cross:
            logger.info("当日未出现MACD金叉")
        
        # 打印KDJ指标
        logger.info("==== KDJ指标 ====")
        logger.info(f"K值: {day_data['K'].iloc[-1]:.4f}")
        logger.info(f"D值: {day_data['D'].iloc[-1]:.4f}")
        logger.info(f"J值: {day_data['J'].iloc[-1]:.4f}")
        
        # 判断KDJ是否金叉
        kdj_cross = False
        for i in range(1, len(day_data)):
            if (day_data['K'].iloc[i-1] < day_data['D'].iloc[i-1] and 
                day_data['K'].iloc[i] > day_data['D'].iloc[i]):
                kdj_cross = True
                logger.info(f"KDJ金叉出现在: {day_data.iloc[i]['datetime']}")
                
        if not kdj_cross:
            logger.info("当日未出现KDJ金叉")
        
        # 打印RSI指标
        logger.info("==== RSI指标 ====")
        if 'RSI6' in day_data.columns:
            logger.info(f"RSI6: {day_data['RSI6'].iloc[-1]:.4f}")
        if 'RSI12' in day_data.columns:
            logger.info(f"RSI12: {day_data['RSI12'].iloc[-1]:.4f}")
        if 'RSI24' in day_data.columns:
            logger.info(f"RSI24: {day_data['RSI24'].iloc[-1]:.4f}")
        
        # 打印BOLL指标
        logger.info("==== BOLL指标 ====")
        logger.info(f"BOLL上轨: {day_data['upper'].iloc[-1]:.4f}")
        logger.info(f"BOLL中轨: {day_data['middle'].iloc[-1]:.4f}")
        logger.info(f"BOLL下轨: {day_data['lower'].iloc[-1]:.4f}")
        
        # 判断价格是否突破BOLL轨道
        if day_data['close'].iloc[-1] > day_data['upper'].iloc[-1]:
            logger.info("价格突破BOLL上轨")
        elif day_data['close'].iloc[-1] < day_data['lower'].iloc[-1]:
            logger.info("价格跌破BOLL下轨")
        else:
            logger.info("价格在BOLL轨道内")
        
        # 返回结果
        return {
            'code': code,
            'date': date,
            'data': day_data,
            'macd_cross': macd_cross,
            'kdj_cross': kdj_cross
        }
            
    except Exception as e:
        logger.error(f"分析过程中出错: {e}")
        raise
        
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="分析15分钟数据")
    parser.add_argument("--code", type=str, default="600585", help="股票代码")
    parser.add_argument("--date", type=str, default="20250408", help="日期，格式为YYYYMMDD")
    
    args = parser.parse_args()
    analyze_15min_data(args.code, args.date) 