#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
from datetime import datetime, timedelta
import argparse
import logging
from typing import List, Dict, Any

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from scripts.backtest.advanced_backtest import AdvancedBacktester
from db.clickhouse_db import get_clickhouse_db
from utils.logger import get_logger
from utils.path_utils import get_result_path, ensure_dir

logger = get_logger("run_advanced_backtest")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="运行高级形态回测")
    
    # 股票参数
    parser.add_argument("--stock", type=str, help="股票代码，多个股票用逗号分隔")
    parser.add_argument("--stock_list", type=str, help="股票代码列表文件路径")
    parser.add_argument("--index", type=str, help="指数代码，用于分析和回测指数成分股")
    
    # 日期参数
    parser.add_argument("--start_date", type=str, help="开始日期，格式：YYYY-MM-DD")
    parser.add_argument("--end_date", type=str, help="结束日期，格式：YYYY-MM-DD")
    parser.add_argument("--days", type=int, default=180, help="回溯天数，默认180天")
    
    # 指标参数
    parser.add_argument("--indicators", type=str, default="MACD,KDJ,RSI", 
                       help="要分析的指标，多个指标用逗号分隔")
    
    # 周期参数
    parser.add_argument("--periods", type=str, default="DAILY,WEEKLY", 
                       help="要分析的周期，多个周期用逗号分隔")
    
    # 回测参数
    parser.add_argument("--forward_days", type=int, default=10, 
                       help="回测向前看的天数，默认10天")
    parser.add_argument("--threshold", type=float, default=0.03, 
                       help="回测成功阈值，默认3%")
    
    # 高级回测参数
    parser.add_argument("--min_strength", type=float, default=70, 
                       help="最小形态强度，默认70")
    parser.add_argument("--combination_mode", type=str, choices=["any", "all"], default="any",
                       help="形态组合模式：any-任一符合, all-全部符合")
    parser.add_argument("--profit_taking", type=float, default=0.08,
                       help="止盈比例，默认8%")
    parser.add_argument("--stop_loss", type=float, default=-0.05,
                       help="止损比例，默认-5%")
    parser.add_argument("--pattern_combo", type=str, default=None,
                       help="形态组合文件路径，JSON格式")
    
    # 输出参数
    parser.add_argument("--output", type=str, help="输出结果文件路径")
    parser.add_argument("--verbose", action="store_true", help="是否输出详细信息")
    
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

def load_pattern_combinations(file_path: str) -> List[Dict[str, Any]]:
    """
    加载形态组合配置
    
    Args:
        file_path: 文件路径
        
    Returns:
        List[Dict[str, Any]]: 形态组合列表
    """
    if not file_path:
        # 默认形态组合
        return [
            {
                "indicator_id": "MACD",
                "pattern_id": "golden_cross",
                "period": "DAILY"
            },
            {
                "indicator_id": "RSI",
                "pattern_id": "oversold",
                "period": "DAILY"
            }
        ]
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            combinations = json.load(f)
        
        logger.info(f"已加载 {len(combinations)} 个形态组合")
        return combinations
    except Exception as e:
        logger.error(f"加载形态组合文件 {file_path} 出错: {e}")
        return []

def save_results(results: Dict[str, Any], filepath: str) -> str:
    """
    保存结果到文件
    
    Args:
        results: 回测结果
        filepath: 文件路径
        
    Returns:
        str: 保存的文件路径
    """
    # 如果未指定文件名，则生成默认文件名
    if not filepath:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"data/result/advanced_backtest_{timestamp}.json"
    
    # 确保目录存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # 保存结果
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"已保存结果到 {filepath}")
    
    return filepath

def print_backtest_summary(results: Dict[str, Any]) -> None:
    """
    打印回测摘要
    
    Args:
        results: 回测结果
    """
    if not results:
        logger.warning("回测结果为空")
        return
    
    # 性能指标
    performance = results.get("performance", {})
    
    print("\n===== 高级形态回测摘要 =====")
    print(f"总交易次数: {performance.get('total_trades', 0)}")
    print(f"成功率: {performance.get('success_rate', 0) * 100:.2f}%")
    print(f"平均收益: {performance.get('avg_gain', 0) * 100:.2f}%")
    print(f"平均最大收益: {performance.get('avg_max_profit', 0) * 100:.2f}%")
    print(f"平均最大回撤: {performance.get('avg_max_drawdown', 0) * 100:.2f}%")
    print(f"平均持有天数: {performance.get('avg_hold_days', 0):.2f}")
    print(f"收益风险比: {performance.get('risk_reward_ratio', 0):.2f}")
    print(f"年化收益率: {performance.get('yearly_return', 0) * 100:.2f}%")
    print(f"夏普比率: {performance.get('sharpe_ratio', 0):.2f}")
    
    # 打印组合结果
    combinations = results.get("combinations", {})
    
    if combinations:
        # 按成功率排序
        sorted_combinations = sorted(
            combinations.items(),
            key=lambda x: (x[1].get("success_rate", 0), x[1].get("count", 0)),
            reverse=True
        )
        
        print("\n表现最好的形态组合 (按成功率):")
        for i, (key, combo) in enumerate(sorted_combinations[:3]):
            success_rate = combo.get("success_rate", 0) * 100
            avg_gain = combo.get("avg_gain", 0) * 100
            count = combo.get("count", 0)
            
            print(f"\n{i+1}. 组合: {key}")
            print(f"   - 成功率: {success_rate:.2f}%")
            print(f"   - 平均收益: {avg_gain:.2f}%")
            print(f"   - 交易次数: {count}")
            
            # 打印组合中的形态
            patterns = combo.get("patterns", [])
            if patterns:
                print("   - 包含形态:")
                for pattern in patterns:
                    indicator = pattern.get("indicator_id", "")
                    pattern_id = pattern.get("pattern_id", "")
                    period = pattern.get("period", "")
                    print(f"     * {indicator} {pattern_id} ({period})")

def main():
    """主函数"""
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
    
    # 加载形态组合
    pattern_combinations = load_pattern_combinations(args.pattern_combo)
    
    if not pattern_combinations:
        logger.error("没有有效的形态组合，程序退出")
        return
    
    # 初始化高级回测器
    backtester = AdvancedBacktester(
        indicators=args.indicators.split(','),
        periods=args.periods.split(',')
    )
    
    # 设置回测配置
    backtester.set_config({
        "min_pattern_strength": args.min_strength,
        "combination_mode": args.combination_mode,
        "profit_taking": args.profit_taking,
        "stop_loss": args.stop_loss
    })
    
    # 运行回测
    try:
        logger.info(f"开始高级形态回测")
        results = backtester.backtest_with_combination(
            stock_codes=stock_codes,
            start_date=args.start_date,
            end_date=args.end_date,
            pattern_combinations=pattern_combinations,
            forward_days=args.forward_days,
            threshold=args.threshold
        )
        
        # 保存结果
        save_results(results, args.output)
        
        # 打印摘要
        print_backtest_summary(results)
        
    except Exception as e:
        logger.error(f"高级回测时出错: {e}")
    
    logger.info("处理完成")

if __name__ == "__main__":
    main() 