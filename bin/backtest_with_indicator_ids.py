#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
指标ID回测工具

回测并输出指标唯一ID，便于直接生成选股策略
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yaml
import argparse
import traceback

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from strategy.strategy_parser import StrategyParser
from strategy.strategy_condition_evaluator import StrategyConditionEvaluator
from utils.logger import get_logger, init_logging
from utils.path_utils import get_result_dir
from db.data_manager import DataManager
from enums.period import Period
from indicators.indicator_registry import indicator_registry, IndicatorEnum

logger = get_logger(__name__)

def get_stock_data(data_manager, stock_code, start_date, end_date, period=Period.DAILY):
    """获取股票数据"""
    try:
        logger.info(f"获取股票 {stock_code} 的数据，周期: {period}，开始日期: {start_date}，结束日期: {end_date}")
        
        # 获取K线数据
        k_data = data_manager.get_kline_data(
            stock_code=stock_code,
            period=period,
            start_date=start_date,
            end_date=end_date,
            fields=None  # 获取所有字段
        )
        
        if k_data is None or len(k_data) == 0:
            logger.error(f"获取股票 {stock_code} 的K线数据失败")
            return None
            
        # 获取列名映射
        column_mapping = None
        if hasattr(k_data, 'dtypes'):
            # 检查是否有以col_开头的列，如果有，需要进行映射
            cols = k_data.columns.tolist()
            if any(col.startswith('col_') for col in cols):
                # 映射规则
                column_mapping = {
                    'col_0': 'date',
                    'col_1': 'open',
                    'col_2': 'high',
                    'col_3': 'low',
                    'col_4': 'close',
                    'col_5': 'volume'
                }
        
        # 应用列名映射
        if column_mapping:
            logger.info(f"应用列名映射: {column_mapping}")
            k_data = k_data.rename(columns=column_mapping)
        
        # 确保所有必需的列都存在
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in k_data.columns]
        
        if missing_columns:
            logger.error(f"股票 {stock_code} 的K线数据缺少必需的列: {missing_columns}")
            return None
            
        # 确保数据类型正确
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in k_data.columns:
                k_data[col] = pd.to_numeric(k_data[col], errors='coerce')
        
        # 确保数据有索引并且索引是日期类型
        if not isinstance(k_data.index, pd.DatetimeIndex) and 'date' in k_data.columns:
            k_data = k_data.set_index('date')
            k_data.index = pd.to_datetime(k_data.index)
            
        return k_data
        
    except Exception as e:
        logger.error(f"获取股票 {stock_code} 数据时出错: {e}")
        logger.error(traceback.format_exc())
        return None

def evaluate_indicators(evaluator, stock_data, stock_code, indicator_configs):
    """评估所有指标"""
    results = {}
    passing_indicators = []
    
    for indicator_config in indicator_configs:
        indicator_id = indicator_config["indicator_id"]
        parameters = indicator_config.get("parameters", {})
        period = indicator_config.get("period", "DAILY")
        
        # 创建单个指标条件
        condition = {
            "indicator_id": indicator_id,
            "period": period,
            "parameters": parameters,
            "signal_type": "BUY"
        }
        
        try:
            # 评估单个指标
            indicator_result = evaluator._evaluate_indicator_condition(condition, stock_data, stock_code)
            
            # 记录结果
            results[indicator_id] = {
                "result": indicator_result,
                "parameters": parameters,
                "period": period
            }
            
            if indicator_result:
                passing_indicators.append(indicator_id)
                
        except Exception as e:
            logger.error(f"评估指标 {indicator_id} 时出错: {e}")
            logger.error(traceback.format_exc())
            results[indicator_id] = {
                "result": False,
                "error": str(e),
                "parameters": parameters,
                "period": period
            }
    
    return {
        "details": results,
        "passing_indicators": passing_indicators
    }

def run_backtest(stock_list_file, start_date, end_date, output_file=None):
    """运行回测"""
    try:
        # 初始化
        init_logging(level="INFO")
        data_manager = DataManager()
        evaluator = StrategyConditionEvaluator()
        
        # 加载股票列表
        stocks = pd.read_csv(stock_list_file)
        
        # 准备要测试的指标配置
        indicator_configs = [
            # ZXM系列指标
            {"indicator_id": IndicatorEnum.ZXM_TURNOVER, "parameters": {"threshold": 1.0}},
            {"indicator_id": IndicatorEnum.ZXM_DAILY_MACD, "parameters": {"threshold": 0.0}},
            {"indicator_id": IndicatorEnum.ZXM_MA_CALLBACK, "parameters": {"periods": [20, 30, 60, 120]}},
            {"indicator_id": IndicatorEnum.ZXM_RISE_ELASTICITY, "parameters": {"rise_threshold": 1.02}},
            {"indicator_id": IndicatorEnum.ZXM_AMPLITUDE_ELASTICITY, "parameters": {"amplitude_threshold": 10.0}},
            {"indicator_id": IndicatorEnum.ZXM_ELASTICITY_SCORE, "parameters": {"threshold": 75}},
            {"indicator_id": IndicatorEnum.ZXM_BUYPOINT_SCORE, "parameters": {"threshold": 75}},
            {"indicator_id": IndicatorEnum.ZXM_DAILY_TREND_UP, "parameters": {"periods": [60, 120]}},
            {"indicator_id": IndicatorEnum.ZXM_ABSORB, "parameters": {"absorb_threshold": 3}},
            
            # 经典技术指标
            {"indicator_id": IndicatorEnum.DIVERGENCE, "parameters": {"type": "MACD", "divergence_type": "BULLISH"}},
            {"indicator_id": IndicatorEnum.KDJ, "parameters": {"n": 9, "m1": 3, "m2": 3, "cross_type": "GOLDEN_CROSS"}},
            {"indicator_id": IndicatorEnum.MA, "parameters": {"periods": [5, 10, 20], "is_bullish": True}},
            {"indicator_id": IndicatorEnum.BOLL, "parameters": {"period": 20, "std_dev": 2, "breakout_type": "MIDDLE"}},
            {"indicator_id": IndicatorEnum.VR, "parameters": {"period": 26, "ma_period": 6, "cross_type": "GOLDEN_CROSS"}},
            {"indicator_id": IndicatorEnum.OBV, "parameters": {"ma_period": 30, "cross_type": "GOLDEN_CROSS"}}
        ]
        
        # 指标成功次数统计
        indicator_stats = {config["indicator_id"]: {"success": 0, "total": 0} for config in indicator_configs}
        
        # 指标组合统计
        pattern_stats = {}
        
        # 买点记录
        buy_points = []
        
        # 处理每只股票
        for _, row in stocks.iterrows():
            stock_code = row['stock_code'] if 'stock_code' in row else row['code']
            stock_name = row['stock_name'] if 'stock_name' in row else ""
            
            # 确保股票代码是字符串
            if isinstance(stock_code, (int, float)):
                stock_code = str(int(stock_code)).zfill(6)
                
            logger.info(f"处理股票 {stock_code} - {stock_name}")
            
            # 获取股票数据
            stock_data = get_stock_data(data_manager, stock_code, start_date, end_date)
            
            if stock_data is None or len(stock_data) == 0:
                continue
                
            # 评估所有指标
            evaluation = evaluate_indicators(evaluator, stock_data, stock_code, indicator_configs)
            
            # 更新指标统计
            for indicator_id, result in evaluation["details"].items():
                indicator_stats[indicator_id]["total"] += 1
                if result["result"]:
                    indicator_stats[indicator_id]["success"] += 1
            
            # 如果有通过的指标，记录买点
            passing_indicators = evaluation["passing_indicators"]
            if passing_indicators:
                # 创建指标组合的标识
                pattern_key = ",".join(sorted(passing_indicators))
                if pattern_key not in pattern_stats:
                    pattern_stats[pattern_key] = {"count": 0, "indicators": passing_indicators}
                pattern_stats[pattern_key]["count"] += 1
                
                # 记录买点
                buy_date = stock_data.index[-1].strftime("%Y-%m-%d")
                buy_points.append({
                    "stock_code": stock_code,
                    "stock_name": stock_name,
                    "buy_date": buy_date,
                    "indicators": passing_indicators,
                    "close": stock_data['close'].iloc[-1],
                    "indicator_count": len(passing_indicators)
                })
                
                logger.info(f"股票 {stock_code} 在 {buy_date} 满足条件，命中指标: {passing_indicators}")
            else:
                logger.info(f"股票 {stock_code} 不满足任何条件")
        
        # 计算成功率
        for indicator_id in indicator_stats:
            total = indicator_stats[indicator_id]["total"]
            success = indicator_stats[indicator_id]["success"]
            success_rate = success / total * 100 if total > 0 else 0
            indicator_stats[indicator_id]["success_rate"] = success_rate
        
        # 按成功率排序指标
        sorted_indicators = sorted(
            indicator_stats.items(), 
            key=lambda x: x[1]["success_rate"], 
            reverse=True
        )
        
        # 按出现次数排序模式
        sorted_patterns = sorted(
            pattern_stats.items(),
            key=lambda x: x[1]["count"],
            reverse=True
        )
        
        # 生成回测报告
        report = "# 回测报告\n\n"
        report += f"回测日期: {start_date} 至 {end_date}\n"
        report += f"股票数量: {len(stocks)}\n"
        report += f"买点数量: {len(buy_points)}\n\n"
        
        # 输出指标统计
        report += "## 指标统计\n\n"
        report += "| 指标ID | 成功率 | 成功次数 | 总次数 |\n"
        report += "|--------|--------|----------|--------|\n"
        
        for indicator_id, stats in sorted_indicators:
            report += f"| {indicator_id} | {stats['success_rate']:.2f}% | {stats['success']} | {stats['total']} |\n"
        
        report += "\n## 最常见的技术形态\n\n"
        
        # 输出前10个最常见的形态
        for i, (pattern_key, stats) in enumerate(sorted_patterns[:10]):
            indicators = stats["indicators"]
            report += f"{i+1}. {', '.join(indicators)} (频率: {stats['count']})\n"
        
        # 输出指标ID列表
        report += "\n## 命中指标ID列表\n\n"
        top_indicators = [indicator_id for indicator_id, _ in sorted_indicators if indicator_stats[indicator_id]["success_rate"] > 50]
        
        for indicator_id in top_indicators:
            display_name = indicator_registry.get_display_name(indicator_id)
            report += f"- {indicator_id} ({display_name})\n"
        
        # 保存买点数据
        if buy_points:
            buy_points_df = pd.DataFrame(buy_points)
            buy_points_file = os.path.join(get_result_dir(), "backtest_buypoints.csv")
            buy_points_df.to_csv(buy_points_file, index=False)
            logger.info(f"买点数据已保存到: {buy_points_file}")
        
        # 保存回测报告
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"回测报告已保存到: {output_file}")
        else:
            print(report)
        
        return {
            "indicator_stats": indicator_stats,
            "pattern_stats": pattern_stats,
            "buy_points": buy_points,
            "report": report
        }
        
    except Exception as e:
        logger.error(f"回测过程中出错: {e}")
        logger.error(traceback.format_exc())
        return None

def main():
    parser = argparse.ArgumentParser(description="使用指标ID进行回测")
    parser.add_argument("-s", "--stocks", required=True, help="股票列表文件路径")
    parser.add_argument("-b", "--begin", required=True, help="开始日期，格式: YYYY-MM-DD")
    parser.add_argument("-e", "--end", required=True, help="结束日期，格式: YYYY-MM-DD")
    parser.add_argument("-o", "--output", help="输出回测报告文件路径")
    
    args = parser.parse_args()
    
    run_backtest(
        stock_list_file=args.stocks,
        start_date=args.begin,
        end_date=args.end,
        output_file=args.output
    )

if __name__ == "__main__":
    main() 