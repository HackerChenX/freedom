#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import os
import argparse
from datetime import datetime

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from utils.logger import get_logger
from utils.path_utils import get_backtest_result_dir, get_strategies_dir, get_data_dir
from scripts.backtest.unified_backtest import UnifiedBacktest, analyze_buypoint_indicators

# 获取日志记录器
logger = get_logger(__name__)

def generate_output_filename(input_file: str, pattern_type: str = ""):
    """
    根据输入文件和模式类型生成输出文件名
    
    Args:
        input_file: 输入文件路径
        pattern_type: 模式类型
        
    Returns:
        str: 输出文件路径
    """
    # 获取输入文件名（不含扩展名）
    base_name = os.path.basename(input_file).split('.')[0]
    
    # 获取当前日期时间
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 生成输出文件名
    if pattern_type:
        output_name = f"{base_name}_{pattern_type}_{now}.json"
    else:
        output_name = f"{base_name}_{now}.json"
    
    # 完整路径
    output_file = os.path.join(get_backtest_result_dir(), output_name)
    return output_file

def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(description="统一回测系统运行工具")
    parser.add_argument("--input", "-i", help="输入文件路径，包含股票代码和买点日期", required=True)
    parser.add_argument("--output", "-o", help="输出文件路径，不指定则自动生成", default="")
    parser.add_argument("--type", "-t", help="买点类型描述", default="")
    parser.add_argument("--check", "-c", action="store_true", help="运行指标检查（不执行回测）")
    
    args = parser.parse_args()
    
    # 如果选择了检查模式，则运行指标检查
    if args.check:
        logger.info("执行指标检查模式...")
        from scripts.backtest.indicator_check import IndicatorChecker
        checker = IndicatorChecker()
        checker.run_all_checks()
        return
    
    # 验证输入文件
    if not os.path.exists(args.input):
        logger.error(f"输入文件不存在: {args.input}")
        return
    
    # 生成输出文件路径（如果未指定）
    output_file = args.output if args.output else generate_output_filename(args.input, args.type)
    
    logger.info(f"开始回测分析，输入文件: {args.input}，输出文件: {output_file}")
    logger.info("注意: 系统将获取每个股票的完整历史数据，买点日期仅作为数据截止日期")
    
    # 创建回测实例并执行分析
    backtest = UnifiedBacktest()
    backtest.batch_analyze(args.input, output_file, args.type)
    
    # 添加额外步骤：使用系统命令处理JSON文件，移除末尾的百分号
    fixed_output_file = output_file.replace(".json", "_fixed.json")
    os.system(f"cat {output_file} | tr -d '%' > {fixed_output_file}")
    os.system(f"mv {fixed_output_file} {output_file}")
    
    # 生成策略文件
    strategy_file = os.path.join(get_strategies_dir(), f"strategy_{datetime.now().strftime('%Y%m%d%H%M%S')}.py")
    backtest.generate_strategy(strategy_file)
    
    # Markdown报告路径
    markdown_file = output_file.replace('.json', '.md')
    
    logger.info(f"回测分析完成，结果已保存到: {output_file}")
    logger.info(f"Markdown格式报告已保存到: {markdown_file}")
    logger.info(f"策略文件已生成: {strategy_file}")

if __name__ == "__main__":
    main() 