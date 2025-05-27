#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
综合回测系统

整合各种回测功能，支持指标唯一ID方式进行回测，并可直接输出为选股策略
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yaml
import json
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

def run_backtest(stock_list_file, start_date, end_date, indicator_list=None, output_file=None):
    """运行回测"""
    try:
        # 初始化
        init_logging(level="INFO")
        data_manager = DataManager()
        evaluator = StrategyConditionEvaluator()
        
        # 加载股票列表
        stocks = pd.read_csv(stock_list_file)
        
        # 准备要测试的指标配置
        if indicator_list:
            # 如果提供了指标列表，就使用这些指标
            logger.info(f"使用指定的指标列表: {indicator_list}")
            indicator_configs = []
            for indicator_id in indicator_list:
                # 获取默认参数
                default_params = {
                    IndicatorEnum.ZXM_TURNOVER: {"threshold": 1.0},
                    IndicatorEnum.ZXM_DAILY_MACD: {"threshold": 0.0},
                    IndicatorEnum.ZXM_MA_CALLBACK: {"periods": [20, 30, 60, 120]},
                    IndicatorEnum.ZXM_RISE_ELASTICITY: {"rise_threshold": 1.02},
                    IndicatorEnum.ZXM_AMPLITUDE_ELASTICITY: {"amplitude_threshold": 10.0},
                    IndicatorEnum.ZXM_ELASTICITY_SCORE: {"threshold": 75},
                    IndicatorEnum.ZXM_BUYPOINT_SCORE: {"threshold": 75},
                    IndicatorEnum.ZXM_DAILY_TREND_UP: {"periods": [60, 120]},
                    IndicatorEnum.ZXM_ABSORB: {"absorb_threshold": 3},
                    IndicatorEnum.DIVERGENCE: {"type": "MACD", "divergence_type": "BULLISH"},
                    IndicatorEnum.KDJ: {"n": 9, "m1": 3, "m2": 3, "cross_type": "GOLDEN_CROSS"},
                    IndicatorEnum.MA: {"periods": [5, 10, 20], "is_bullish": True},
                    IndicatorEnum.BOLL: {"period": 20, "std_dev": 2, "breakout_type": "MIDDLE"},
                    IndicatorEnum.VR: {"period": 26, "ma_period": 6, "cross_type": "GOLDEN_CROSS"},
                    IndicatorEnum.OBV: {"ma_period": 30, "cross_type": "GOLDEN_CROSS"}
                }.get(indicator_id, {})
                
                indicator_configs.append({
                    "indicator_id": indicator_id,
                    "parameters": default_params
                })
        else:
            # 否则使用默认的指标列表
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

def generate_strategy_from_backtest(backtest_result, output_file=None):
    """从回测结果生成策略配置"""
    try:
        # 获取指标统计
        indicator_stats = backtest_result.get("indicator_stats", {})
        pattern_stats = backtest_result.get("pattern_stats", {})
        
        # 按成功率排序指标
        sorted_indicators = sorted(
            indicator_stats.items(), 
            key=lambda x: x[1]["success_rate"], 
            reverse=True
        )
        
        # 选择成功率超过50%的指标
        selected_indicators = [
            indicator_id for indicator_id, stats in sorted_indicators
            if stats["success_rate"] > 50
        ]
        
        # 如果没有合适的指标，尝试使用最常见的模式
        if not selected_indicators and pattern_stats:
            # 按出现次数排序
            sorted_patterns = sorted(
                pattern_stats.items(),
                key=lambda x: x[1]["count"],
                reverse=True
            )
            
            # 取最常见的模式
            if sorted_patterns:
                pattern_key, pattern_info = sorted_patterns[0]
                selected_indicators = pattern_info["indicators"]
        
        # 如果仍然没有指标，使用默认的ZXM指标
        if not selected_indicators:
            selected_indicators = [
                IndicatorEnum.ZXM_TURNOVER,
                IndicatorEnum.ZXM_DAILY_MACD,
                IndicatorEnum.ZXM_BUYPOINT_SCORE
            ]
            
        logger.info(f"选择的指标: {selected_indicators}")
        
        # 创建策略条件
        conditions = []
        for idx, indicator_id in enumerate(selected_indicators):
            # 获取指标默认参数
            params = {
                IndicatorEnum.ZXM_TURNOVER: {"threshold": 1.0},
                IndicatorEnum.ZXM_DAILY_MACD: {"threshold": 0.0},
                IndicatorEnum.ZXM_MA_CALLBACK: {"periods": [20, 30, 60, 120]},
                IndicatorEnum.ZXM_RISE_ELASTICITY: {"rise_threshold": 1.02},
                IndicatorEnum.ZXM_AMPLITUDE_ELASTICITY: {"amplitude_threshold": 10.0},
                IndicatorEnum.ZXM_ELASTICITY_SCORE: {"threshold": 75},
                IndicatorEnum.ZXM_BUYPOINT_SCORE: {"threshold": 75},
                IndicatorEnum.ZXM_DAILY_TREND_UP: {"periods": [60, 120]},
                IndicatorEnum.ZXM_ABSORB: {"absorb_threshold": 3},
                IndicatorEnum.DIVERGENCE: {"type": "MACD", "divergence_type": "BULLISH"},
                IndicatorEnum.KDJ: {"n": 9, "m1": 3, "m2": 3, "cross_type": "GOLDEN_CROSS"},
                IndicatorEnum.MA: {"periods": [5, 10, 20], "is_bullish": True},
                IndicatorEnum.BOLL: {"period": 20, "std_dev": 2, "breakout_type": "MIDDLE"},
                IndicatorEnum.VR: {"period": 26, "ma_period": 6, "cross_type": "GOLDEN_CROSS"},
                IndicatorEnum.OBV: {"ma_period": 30, "cross_type": "GOLDEN_CROSS"}
            }.get(indicator_id, {})
            
            # 添加指标条件
            conditions.append({
                "indicator_id": indicator_id,
                "period": "DAILY",  # 默认使用日线
                "parameters": params,
                "signal_type": "BUY"
            })
            
            # 除了最后一个，每个指标后面都添加OR
            if idx < len(selected_indicators) - 1:
                conditions.append({
                    "logic": "OR"
                })
        
        # 创建策略配置
        strategy_id = f"BACKTEST_STRATEGY_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        strategy_config = {
            "strategy": {
                "id": strategy_id,
                "name": "回测生成策略",
                "description": f"基于回测结果自动生成的策略 - 生成时间 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "version": "1.0",
                "author": "system",
                "create_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "update_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "conditions": conditions,
                "filters": {
                    "market": [],  # 不限制市场
                    "industry": [],  # 不限制行业
                    "market_cap": {
                        "min": 0,
                        "max": 10000
                    },
                    "price": {
                        "min": 0,
                        "max": 500
                    }
                },
                "sort": [
                    {
                        "field": "signal_strength",
                        "direction": "DESC"
                    },
                    {
                        "field": "market_cap",
                        "direction": "ASC"
                    }
                ]
            }
        }
        
        # 保存策略配置
        if output_file:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            # 保存为YAML格式
            with open(output_file, 'w', encoding='utf-8') as f:
                yaml.dump(strategy_config, f, default_flow_style=False, allow_unicode=True)
                
            logger.info(f"策略配置已保存到: {output_file}")
        
        return strategy_config
        
    except Exception as e:
        logger.error(f"生成策略配置时出错: {e}")
        logger.error(traceback.format_exc())
        return None 

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="综合回测系统")
    
    # 添加子命令
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # 回测子命令
    backtest_parser = subparsers.add_parser('backtest', help='执行回测')
    backtest_parser.add_argument('-s', '--stocks', required=True, help='股票列表文件路径')
    backtest_parser.add_argument('-b', '--begin', required=True, help='开始日期，格式: YYYY-MM-DD')
    backtest_parser.add_argument('-e', '--end', required=True, help='结束日期，格式: YYYY-MM-DD')
    backtest_parser.add_argument('-i', '--indicators', help='指标ID列表，用逗号分隔')
    backtest_parser.add_argument('-o', '--output', help='输出回测报告文件路径')
    
    # 生成策略子命令
    generate_parser = subparsers.add_parser('generate', help='从回测结果生成策略')
    generate_parser.add_argument('-r', '--report', required=True, help='回测报告文件路径')
    generate_parser.add_argument('-o', '--output', help='输出策略配置文件路径')
    
    # 完整流程子命令
    workflow_parser = subparsers.add_parser('workflow', help='执行完整回测到策略生成流程')
    workflow_parser.add_argument('-s', '--stocks', required=True, help='股票列表文件路径')
    workflow_parser.add_argument('-b', '--begin', required=True, help='开始日期，格式: YYYY-MM-DD')
    workflow_parser.add_argument('-e', '--end', required=True, help='结束日期，格式: YYYY-MM-DD')
    workflow_parser.add_argument('-i', '--indicators', help='指标ID列表，用逗号分隔')
    workflow_parser.add_argument('-o', '--output_dir', help='输出目录')
    
    args = parser.parse_args()
    
    if args.command == 'backtest':
        # 处理指标列表
        indicator_list = None
        if args.indicators:
            indicator_list = [indicator.strip() for indicator in args.indicators.split(',')]
        
        # 运行回测
        result = run_backtest(
            stock_list_file=args.stocks,
            start_date=args.begin,
            end_date=args.end,
            indicator_list=indicator_list,
            output_file=args.output
        )
        
        if not result:
            logger.error("回测失败")
            return 1
            
        logger.info("回测完成")
        
    elif args.command == 'generate':
        # 加载回测报告
        try:
            with open(args.report, 'r', encoding='utf-8') as f:
                report_text = f.read()
                
            # 解析回测报告
            import re
            
            # 解析指标统计
            indicator_stats = {}
            stats_match = re.search(r'## 指标统计\s+\|[^\n]+\|[^\n]+\|(.*?)(?=\n\n)', report_text, re.DOTALL)
            if stats_match:
                stats_text = stats_match.group(1)
                for line in stats_text.strip().split('\n'):
                    if '|' in line:
                        parts = [p.strip() for p in line.split('|')]
                        if len(parts) >= 5:
                            indicator_id = parts[1]
                            success_rate = float(parts[2].replace('%', ''))
                            success = int(parts[3])
                            total = int(parts[4])
                            
                            indicator_stats[indicator_id] = {
                                "success_rate": success_rate,
                                "success": success,
                                "total": total
                            }
            
            # 解析模式统计
            pattern_stats = {}
            patterns_match = re.search(r'## 最常见的技术形态\s+(.*?)(?=\n\n)', report_text, re.DOTALL)
            if patterns_match:
                patterns_text = patterns_match.group(1)
                for line in patterns_text.strip().split('\n'):
                    match = re.match(r'\d+\.\s+(.*?)\s+\(频率:\s+(\d+)\)', line)
                    if match:
                        indicators = [i.strip() for i in match.group(1).split(',')]
                        count = int(match.group(2))
                        
                        pattern_key = ','.join(sorted(indicators))
                        pattern_stats[pattern_key] = {
                            "count": count,
                            "indicators": indicators
                        }
            
            # 创建回测结果数据结构
            backtest_result = {
                "indicator_stats": indicator_stats,
                "pattern_stats": pattern_stats
            }
            
            # 生成策略
            strategy_config = generate_strategy_from_backtest(backtest_result, args.output)
            
            if not strategy_config:
                logger.error("策略生成失败")
                return 1
                
            logger.info("策略生成完成")
            
        except Exception as e:
            logger.error(f"解析回测报告时出错: {e}")
            logger.error(traceback.format_exc())
            return 1
            
    elif args.command == 'workflow':
        # 处理指标列表
        indicator_list = None
        if args.indicators:
            indicator_list = [indicator.strip() for indicator in args.indicators.split(',')]
        
        # 创建输出目录
        output_dir = args.output_dir or os.path.join(get_result_dir(), f"backtest_workflow_{datetime.now().strftime('%Y%m%d%H%M%S')}")
        os.makedirs(output_dir, exist_ok=True)
        
        # 输出文件路径
        report_file = os.path.join(output_dir, "backtest_report.md")
        strategy_file = os.path.join(output_dir, "strategy.yaml")
        
        # 运行回测
        logger.info("开始执行回测...")
        result = run_backtest(
            stock_list_file=args.stocks,
            start_date=args.begin,
            end_date=args.end,
            indicator_list=indicator_list,
            output_file=report_file
        )
        
        if not result:
            logger.error("回测失败")
            return 1
            
        logger.info("回测完成")
        
        # 生成策略
        logger.info("开始生成策略...")
        strategy_config = generate_strategy_from_backtest(result, strategy_file)
        
        if not strategy_config:
            logger.error("策略生成失败")
            return 1
            
        logger.info("策略生成完成")
        
        # 生成摘要
        summary_file = os.path.join(output_dir, "workflow_summary.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("# 回测流程摘要\n\n")
            f.write(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"股票列表: {args.stocks}\n")
            f.write(f"回测时间范围: {args.begin} 至 {args.end}\n\n")
            
            f.write("## 回测结果\n\n")
            f.write(f"买点数量: {len(result.get('buy_points', []))}\n")
            f.write(f"回测报告: {report_file}\n\n")
            
            f.write("## 生成的策略\n\n")
            f.write(f"策略ID: {strategy_config['strategy']['id']}\n")
            f.write(f"策略名称: {strategy_config['strategy']['name']}\n")
            f.write(f"使用的指标: {', '.join([c['indicator_id'] for c in strategy_config['strategy']['conditions'] if isinstance(c, dict) and 'indicator_id' in c])}\n")
            f.write(f"策略文件: {strategy_file}\n")
        
        logger.info(f"工作流摘要已保存到: {summary_file}")
        logger.info(f"所有输出文件已保存到目录: {output_dir}")
        
    else:
        parser.print_help()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())