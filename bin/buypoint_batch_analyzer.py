#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
买点批量分析命令行工具

分析多个股票买点的共性指标特征，提取共性指标并生成选股策略
"""

import os
import sys
import argparse
from datetime import datetime

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from utils.logger import get_logger, setup_logger
from utils.path_utils import ensure_dir_exists
from analysis.buypoints.buypoint_batch_analyzer import BuyPointBatchAnalyzer

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='买点批量分析工具')
    
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='输入的CSV文件路径，包含股票代码和买点日期')
    
    parser.add_argument('--output', '-o', type=str, default='data/result/buypoint_analysis',
                        help='输出结果的目录路径')
    
    parser.add_argument('--min-hit-ratio', '-r', type=float, default=0.6,
                        help='共性指标的最小命中比例，默认为0.6（60%%）')
    
    parser.add_argument('--strategy-name', '-s', type=str, default='BuyPointCommonStrategy',
                        help='生成策略的名称，默认为"BuyPointCommonStrategy"')
    
    parser.add_argument('--log-level', '-l', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='日志级别，默认为INFO')
    
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 设置日志
    log_dir = os.path.join(root_dir, 'logs')
    ensure_dir_exists(log_dir)
    log_file = os.path.join(log_dir, f'buypoint_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    setup_logger(log_file=log_file, log_level=args.log_level)
    
    logger = get_logger(__name__)
    logger.info("买点批量分析工具启动")
    
    try:
        # 确保输出目录存在
        ensure_dir_exists(args.output)
        
        # 创建分析器实例
        analyzer = BuyPointBatchAnalyzer()
        
        # 运行分析
        analyzer.run_analysis(
            input_csv=args.input,
            output_dir=args.output,
            min_hit_ratio=args.min_hit_ratio,
            strategy_name=args.strategy_name
        )
        
        logger.info(f"分析完成，结果已保存到: {args.output}")
        
        # 打印结果路径
        report_path = os.path.join(args.output, 'common_indicators_report.md')
        strategy_path = os.path.join(args.output, 'generated_strategy.json')
        
        print("\n==================== 分析完成 ====================")
        print(f"共性指标报告: {report_path}")
        print(f"生成的策略: {strategy_path}")
        print("==================================================\n")
        
    except Exception as e:
        logger.error(f"运行过程中出错: {e}")
        print(f"错误: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main()) 