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
    
    # 确保输出目录存在
    ensure_dir_exists(args.output)
    
    # 设置日志
    log_dir = os.path.join(root_dir, 'logs')
    ensure_dir_exists(log_dir)
    log_file = os.path.join(log_dir, f'buypoint_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    setup_logger(log_file=log_file, log_level=args.log_level)
    
    logger = get_logger(__name__)
    logger.info("买点批量分析工具启动")
    
    try:
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
        validation_path = os.path.join(args.output, 'validation_report.md')

        # P2级任务：改进用户体验 - 更美观的输出格式
        print("\n" + "="*60)
        print("🎉 买点分析完成")
        print("="*60)

        print(f"📊 共性指标报告: {report_path}")
        print(f"🎯 生成的策略: {strategy_path}")

        # P0级任务：显示验证结果
        if os.path.exists(validation_path):
            print(f"📋 策略验证报告: {validation_path}")

            # 尝试读取验证结果并显示关键信息
            try:
                validation_json_path = os.path.join(args.output, 'validation_report.json')
                if os.path.exists(validation_json_path):
                    import json
                    with open(validation_json_path, 'r', encoding='utf-8') as f:
                        validation_data = json.load(f)

                    match_rate = validation_data.get('match_analysis', {}).get('match_rate', 0)
                    quality_grade = validation_data.get('quality_grade', '未知')

                    print(f"📈 策略匹配率: {match_rate:.2%}")
                    print(f"⭐ 策略质量: {quality_grade}")

                    if match_rate >= 0.6:
                        print("✅ 策略验证通过 (匹配率 ≥ 60%)")
                    else:
                        print("⚠️  策略匹配率偏低，建议查看优化建议")

                    # 显示优化信息
                    if 'optimization_result' in validation_data:
                        print("🔧 已执行智能优化")

            except Exception as e:
                logger.warning(f"读取验证结果时出错: {e}")

        # 显示系统健康报告
        health_report_path = os.path.join(args.output, 'system_health_report.md')
        if os.path.exists(health_report_path):
            print(f"💊 系统健康报告: {health_report_path}")

        print("="*60)
        print("✨ 分析完成，感谢使用选股系统！")
        print("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"运行过程中出错: {e}")
        print(f"错误: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main()) 