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

from utils.logger import get_logger
from utils.path_utils import get_result_dir
from analysis.strategy_comparison import StrategyComparison

# 获取日志记录器
logger = get_logger(__name__)

def validate_date(date_str):
    """验证日期格式是否为YYYYMMDD"""
    if not date_str:
        return None
    try:
        datetime.strptime(date_str, "%Y%m%d")
        return date_str
    except ValueError:
        raise argparse.ArgumentTypeError(f"日期格式错误: {date_str}，应为YYYYMMDD格式")

def validate_strategy_list(strategy_str):
    """验证策略列表格式"""
    if not strategy_str:
        raise argparse.ArgumentTypeError("策略ID列表不能为空")
    
    strategies = [s.strip() for s in strategy_str.split(',') if s.strip()]
    if not strategies:
        raise argparse.ArgumentTypeError("策略ID列表不能为空")
    
    return strategies

def validate_weight_list(weight_str):
    """验证权重列表格式"""
    if not weight_str:
        return None
        
    try:
        weights = [float(w.strip()) for w in weight_str.split(',') if w.strip()]
        if not weights:
            return None
            
        # 检查权重是否非负
        if any(w < 0 for w in weights):
            raise argparse.ArgumentTypeError("策略权重不能为负")
            
        return weights
    except ValueError:
        raise argparse.ArgumentTypeError("权重格式错误，应为逗号分隔的数字")

def validate_dimension_list(dimension_str):
    """验证维度列表格式"""
    if not dimension_str:
        return None
        
    valid_dimensions = {
        "selection_ratio", "return_performance", "stability", 
        "risk", "style", "industry", "overlap"
    }
    
    dimensions = [d.strip() for d in dimension_str.split(',') if d.strip()]
    
    invalid_dimensions = [d for d in dimensions if d not in valid_dimensions]
    if invalid_dimensions:
        raise argparse.ArgumentTypeError(
            f"无效的维度: {', '.join(invalid_dimensions)}，" +
            f"可用维度: {', '.join(valid_dimensions)}"
        )
    
    return dimensions

def main():
    """命令行入口函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="策略比较工具")
    
    # 添加子命令
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # 多维度比较子命令
    multi_dim_parser = subparsers.add_parser('multi_dimension', help='多维度策略比较')
    multi_dim_parser.add_argument('-s', '--strategies', required=True, type=validate_strategy_list, 
                                 help='策略ID列表，用逗号分隔')
    multi_dim_parser.add_argument('-d', '--dimensions', type=validate_dimension_list,
                                 help='比较维度列表，用逗号分隔，可选: selection_ratio,return_performance,stability,risk,style,industry,overlap')
    multi_dim_parser.add_argument('-p', '--periods', help='时间周期列表，格式: name1:start1:end1,name2:start2:end2')
    multi_dim_parser.add_argument('-f', '--file', help='股票代码文件路径，每行一个股票代码')
    multi_dim_parser.add_argument('-o', '--output', help='输出文件路径')
    multi_dim_parser.add_argument('--format', choices=['json', 'markdown', 'excel'], default='json', help='输出格式')
    multi_dim_parser.add_argument('-v', '--visualize', action='store_true', help='是否生成可视化图表')
    
    # 策略组合分析子命令
    combination_parser = subparsers.add_parser('combination', help='策略组合分析')
    combination_parser.add_argument('-s', '--strategies', required=True, type=validate_strategy_list, 
                                   help='策略ID列表，用逗号分隔')
    combination_parser.add_argument('-w', '--weights', type=validate_weight_list,
                                   help='策略权重列表，用逗号分隔，默认等权重')
    combination_parser.add_argument('--start', type=validate_date, help='开始日期，格式YYYYMMDD')
    combination_parser.add_argument('--end', type=validate_date, help='结束日期，格式YYYYMMDD')
    combination_parser.add_argument('-f', '--file', help='股票代码文件路径，每行一个股票代码')
    combination_parser.add_argument('-o', '--output', help='输出文件路径')
    combination_parser.add_argument('--format', choices=['json', 'markdown', 'excel'], default='json', help='输出格式')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # 创建策略比较器
    strategy_comparison = StrategyComparison()
    
    # 处理输出文件参数，默认为结果目录下的文件
    if not hasattr(args, 'output') or not args.output:
        result_dir = get_result_dir()
        os.makedirs(result_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join(result_dir, f"strategy_comparison_{args.command}_{timestamp}.{args.format}")
    
    if args.command == 'multi_dimension':
        # 解析策略ID列表 - 已经由argparse验证
        strategy_ids = args.strategies
        
        # 解析比较维度 - 已经由argparse验证
        dimensions = args.dimensions
        
        # 解析时间周期
        time_periods = None
        if args.periods:
            time_periods = []
            try:
                for period_str in args.periods.split(','):
                    parts = period_str.split(':')
                    if len(parts) >= 3:
                        # 验证日期格式
                        try:
                            datetime.strptime(parts[1], "%Y%m%d")
                            datetime.strptime(parts[2], "%Y%m%d")
                        except ValueError:
                            logger.warning(f"时间周期 {period_str} 的日期格式错误，将被跳过")
                            continue
                            
                        period = {
                            "name": parts[0],
                            "start_date": parts[1],
                            "end_date": parts[2]
                        }
                        time_periods.append(period)
                    else:
                        logger.warning(f"时间周期格式错误: {period_str}，应为 name:start_date:end_date")
            except Exception as e:
                logger.error(f"解析时间周期时出错: {e}")
        
        # 读取股票池
        stock_pool = None
        if args.file:
            try:
                if not os.path.exists(args.file):
                    logger.error(f"股票代码文件不存在: {args.file}")
                else:
                    with open(args.file, 'r', encoding='utf-8') as f:
                        stock_pool = [line.strip() for line in f if line.strip()]
                    logger.info(f"从文件加载了 {len(stock_pool)} 只股票")
            except Exception as e:
                logger.error(f"读取股票代码文件时出错: {e}")
        
        # 执行多维度比较
        result = strategy_comparison.compare_strategies_multi_dimension(
            strategy_ids, time_periods, stock_pool, dimensions)
        
        if isinstance(result, dict) and "error" in result:
            logger.error(f"多维度比较出错: {result['error']}")
            return
        
        # 保存结果
        output_file = strategy_comparison.save_comparison_result(result, args.output, args.format)
        
        # 生成可视化图表
        if args.visualize:
            try:
                visual_file = os.path.splitext(args.output)[0] + ".png"
                visual_result = strategy_comparison.visualize_strategy_comparison(result, visual_file)
                if visual_result:
                    print(f"可视化图表已保存到: {visual_result}")
                else:
                    logger.warning("生成可视化图表失败，可能缺少matplotlib库或数据不足")
            except Exception as e:
                logger.error(f"生成可视化图表时出错: {e}")
        
        # 打印关键信息
        print(f"多维度策略比较完成，比较了 {len(strategy_ids)} 个策略")
        
        overall_scores = result.get("overall_scores", [])
        if overall_scores:
            print("\n综合评分排名:")
            # 只显示前5名或全部（如果少于5个）
            for rank_info in overall_scores[:min(5, len(overall_scores))]:
                print(f"  {rank_info.get('rank', '')}: 策略 {rank_info.get('strategy_id', '')} - 评分 {rank_info.get('overall_score', 0):.4f}")
        
        if output_file:
            print(f"\n详细结果已保存到: {output_file}")
        else:
            logger.error("保存结果失败")
        
    elif args.command == 'combination':
        # 解析策略ID列表 - 已经由argparse验证
        strategy_ids = args.strategies
        
        # 解析策略权重 - 已经由argparse验证
        weights = args.weights
        
        # 日期已经通过argparse验证
        start_date = args.start
        end_date = args.end
        
        # 如果只提供了一个日期，设置默认的另一个日期
        if start_date and not end_date:
            end_date = datetime.now().strftime("%Y%m%d")
        elif end_date and not start_date:
            start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=30)).strftime("%Y%m%d")
        
        # 读取股票池
        stock_pool = None
        if args.file:
            try:
                if not os.path.exists(args.file):
                    logger.error(f"股票代码文件不存在: {args.file}")
                else:
                    with open(args.file, 'r', encoding='utf-8') as f:
                        stock_pool = [line.strip() for line in f if line.strip()]
                    logger.info(f"从文件加载了 {len(stock_pool)} 只股票")
            except Exception as e:
                logger.error(f"读取股票代码文件时出错: {e}")
        
        # 执行策略组合分析
        result = strategy_comparison.analyze_strategy_combination(
            strategy_ids, weights, start_date, end_date, stock_pool)
        
        if isinstance(result, dict) and "error" in result:
            logger.error(f"策略组合分析出错: {result['error']}")
            return
        
        # 保存结果
        output_file = strategy_comparison.save_comparison_result(result, args.output, args.format)
        
        # 打印关键信息
        print(f"策略组合分析完成，组合了 {len(strategy_ids)} 个策略")
        
        combined_stocks_count = result.get("combined_stock_count", 0)
        selection_ratio = result.get("selection_ratio", 0)
        print(f"组合选出 {combined_stocks_count} 只股票，选股比例 {selection_ratio:.2%}")
        
        comparison = result.get("comparison", {})
        if comparison:
            avg_return_improvement = comparison.get("average_return_improvement", 0)
            avg_positive_improvement = comparison.get("average_positive_improvement", 0)
            print(f"组合相比单一策略平均收益提升: {avg_return_improvement:.2%}")
            print(f"组合相比单一策略平均胜率提升: {avg_positive_improvement:.2%}")
        
        performance = result.get("performance", {})
        if performance:
            weighted_return = performance.get("weighted_return", 0)
            weighted_positive = performance.get("weighted_positive_ratio", 0)
            print(f"组合加权收益: {weighted_return:.2%}")
            print(f"组合胜率: {weighted_positive:.2%}")
        
        if output_file:
            print(f"\n详细结果已保存到: {output_file}")
        else:
            logger.error("保存结果失败")
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 