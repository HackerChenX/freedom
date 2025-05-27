#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import sys
import argparse
from datetime import datetime

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from scripts.backtest.backtest_strategy_integrator import BacktestStrategyIntegrator
from utils.logger import get_logger

# 获取日志记录器
logger = get_logger(__name__)

def main():
    """命令行入口函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="回测选股集成工具")
    
    # 添加子命令
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # 生成策略子命令
    generate_parser = subparsers.add_parser('generate', help='从回测结果生成策略')
    generate_parser.add_argument('-i', '--input', required=True, help='回测结果文件路径')
    generate_parser.add_argument('-o', '--output', help='输出策略文件路径，默认自动生成')
    
    # 验证策略子命令
    validate_parser = subparsers.add_parser('validate', help='验证生成的策略')
    validate_parser.add_argument('-i', '--input', required=True, help='策略文件路径')
    validate_parser.add_argument('-s', '--start_date', help='验证开始日期，格式YYYYMMDD')
    validate_parser.add_argument('-e', '--end_date', help='验证结束日期，格式YYYYMMDD')
    validate_parser.add_argument('-p', '--pool', help='验证股票池文件，每行一个股票代码')
    validate_parser.add_argument('-o', '--output', required=True, help='输出验证结果文件路径')
    
    # 优化策略子命令
    optimize_parser = subparsers.add_parser('optimize', help='优化策略参数')
    optimize_parser.add_argument('-i', '--input', required=True, help='策略文件路径')
    optimize_parser.add_argument('-v', '--validation', required=True, help='验证结果文件路径')
    optimize_parser.add_argument('-o', '--output', required=True, help='输出优化后的策略文件路径')
    
    # 完整流程子命令
    workflow_parser = subparsers.add_parser('workflow', help='执行完整的集成流程')
    workflow_parser.add_argument('-i', '--input', required=True, help='回测结果文件路径')
    workflow_parser.add_argument('-o', '--output_dir', required=True, help='输出目录')
    workflow_parser.add_argument('-p', '--pool', help='验证股票池文件，每行一个股票代码')
    
    args = parser.parse_args()
    
    # 创建集成器
    integrator = BacktestStrategyIntegrator()
    
    if args.command == 'generate':
        # 生成策略
        logger.info(f"从回测结果生成策略: {args.input}")
        strategy_config = integrator.generate_strategy(args.input, args.output)
        
        if strategy_config:
            logger.info(f"策略生成成功: {strategy_config['strategy']['name']}")
        else:
            logger.error("策略生成失败")
        
    elif args.command == 'validate':
        # 读取策略配置
        try:
            import json
            with open(args.input, 'r', encoding='utf-8') as f:
                strategy_config = json.load(f)
        except Exception as e:
            logger.error(f"读取策略文件时出错: {e}")
            return
        
        # 读取股票池
        stock_pool = None
        if args.pool:
            try:
                with open(args.pool, 'r', encoding='utf-8') as f:
                    stock_pool = [line.strip() for line in f if line.strip()]
            except Exception as e:
                logger.error(f"读取股票池文件时出错: {e}")
        
        # 验证策略
        logger.info(f"开始验证策略: {strategy_config['strategy']['name']}")
        validation_result = integrator.validate_strategy(
            strategy_config, args.start_date, args.end_date, stock_pool)
        
        # 保存验证结果
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(validation_result, f, indent=2, ensure_ascii=False)
            
        logger.info(f"验证结果已保存到: {args.output}")
        
    elif args.command == 'optimize':
        # 读取策略配置
        try:
            import json
            with open(args.input, 'r', encoding='utf-8') as f:
                strategy_config = json.load(f)
        except Exception as e:
            logger.error(f"读取策略文件时出错: {e}")
            return
        
        # 读取验证结果
        try:
            with open(args.validation, 'r', encoding='utf-8') as f:
                validation_result = json.load(f)
        except Exception as e:
            logger.error(f"读取验证结果文件时出错: {e}")
            return
        
        # 优化策略
        logger.info(f"开始优化策略: {strategy_config['strategy']['name']}")
        optimized_config = integrator.optimize_strategy(strategy_config, validation_result)
        
        # 保存优化后的策略
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(optimized_config, f, indent=2, ensure_ascii=False)
            
        logger.info(f"优化后的策略已保存到: {args.output}")
        
    elif args.command == 'workflow':
        # 执行完整流程
        logger.info(f"开始执行完整集成流程: {args.input}")
        
        # 确保输出目录存在
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 生成策略
        strategy_file = os.path.join(args.output_dir, "generated_strategy.json")
        strategy_config = integrator.generate_strategy(args.input, strategy_file)
        
        if not strategy_config:
            logger.error("策略生成失败")
            return
            
        logger.info(f"策略生成成功: {strategy_config['strategy']['name']}")
        
        # 读取股票池
        stock_pool = None
        if args.pool:
            try:
                with open(args.pool, 'r', encoding='utf-8') as f:
                    stock_pool = [line.strip() for line in f if line.strip()]
            except Exception as e:
                logger.error(f"读取股票池文件时出错: {e}")
        
        # 验证策略
        validation_file = os.path.join(args.output_dir, "validation_result.json")
        validation_result = integrator.validate_strategy(strategy_config, stock_pool=stock_pool)
        
        # 保存验证结果
        import json
        with open(validation_file, 'w', encoding='utf-8') as f:
            json.dump(validation_result, f, indent=2, ensure_ascii=False)
            
        logger.info(f"验证结果已保存到: {validation_file}")
        
        # 优化策略
        optimized_file = os.path.join(args.output_dir, "optimized_strategy.json")
        optimized_config = integrator.optimize_strategy(strategy_config, validation_result)
        
        # 保存优化后的策略
        with open(optimized_file, 'w', encoding='utf-8') as f:
            json.dump(optimized_config, f, indent=2, ensure_ascii=False)
            
        logger.info(f"优化后的策略已保存到: {optimized_file}")
        
        # 验证优化后的策略
        optimized_validation_file = os.path.join(args.output_dir, "optimized_validation_result.json")
        optimized_validation = integrator.validate_strategy(optimized_config, stock_pool=stock_pool)
        
        # 保存优化后的验证结果
        with open(optimized_validation_file, 'w', encoding='utf-8') as f:
            json.dump(optimized_validation, f, indent=2, ensure_ascii=False)
            
        logger.info(f"优化后的验证结果已保存到: {optimized_validation_file}")
        
        # 生成结果摘要
        summary_file = os.path.join(args.output_dir, "integration_summary.txt")
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("# 回测选股集成流程摘要\n\n")
            f.write(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"回测结果文件: {args.input}\n\n")
            
            f.write("## 生成的策略\n\n")
            f.write(f"策略名称: {strategy_config['strategy']['name']}\n")
            f.write(f"策略ID: {strategy_config['strategy']['id']}\n")
            f.write(f"策略文件: {strategy_file}\n\n")
            
            f.write("## 原始验证结果\n\n")
            f.write(f"选出股票数: {validation_result.get('selected_stocks', 0)}\n")
            f.write(f"选股比例: {validation_result.get('selection_ratio', 0):.2%}\n")
            f.write(f"验证结果文件: {validation_file}\n\n")
            
            f.write("## 优化后的策略\n\n")
            f.write(f"策略名称: {optimized_config['strategy']['name']}\n")
            f.write(f"策略ID: {optimized_config['strategy']['id']}\n")
            f.write(f"策略文件: {optimized_file}\n\n")
            
            f.write("## 优化后的验证结果\n\n")
            f.write(f"选出股票数: {optimized_validation.get('selected_stocks', 0)}\n")
            f.write(f"选股比例: {optimized_validation.get('selection_ratio', 0):.2%}\n")
            f.write(f"验证结果文件: {optimized_validation_file}\n\n")
            
            f.write("## 对比\n\n")
            orig_ratio = validation_result.get('selection_ratio', 0)
            opt_ratio = optimized_validation.get('selection_ratio', 0)
            ratio_change = (opt_ratio - orig_ratio) / orig_ratio if orig_ratio > 0 else 0
            
            f.write(f"选股比例变化: {ratio_change:.2%}\n")
            
            if opt_ratio > 0.01 and opt_ratio < 0.1:
                f.write("结论: 优化后的策略选股比例在合理范围内，建议采用\n")
            elif opt_ratio <= 0.01:
                f.write("结论: 优化后的策略选股比例偏低，建议进一步调整参数\n")
            else:
                f.write("结论: 优化后的策略选股比例偏高，建议收紧条件\n")
        
        logger.info(f"集成流程摘要已保存到: {summary_file}")
        logger.info("完整集成流程执行完毕")
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 