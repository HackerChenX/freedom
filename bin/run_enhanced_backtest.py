#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
增强版回测系统命令行脚本

使用增强版回测系统执行买点分析，解决周期与指标混淆问题
"""

import sys
import os

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

import argparse
import json
from typing import Dict, Any, List

from enums.period import Period
from scripts.backtest.enhanced_backtest import EnhancedBacktest
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='增强版回测系统')
    
    # 输入输出参数
    parser.add_argument('-i', '--input', required=True, help='输入CSV文件路径，包含股票代码和买点日期')
    parser.add_argument('-o', '--output', required=True, help='输出JSON文件路径，保存分析结果')
    parser.add_argument('-s', '--strategy', help='输出策略文件路径，不指定则不生成策略')
    
    # 周期配置
    parser.add_argument('-p', '--periods', help='要分析的周期，逗号分隔，如"DAILY,WEEKLY,MIN60"')
    
    # 回测参数
    parser.add_argument('--days-before', type=int, default=20, help='买点前分析天数')
    parser.add_argument('--days-after', type=int, default=10, help='买点后分析天数')
    
    # 指标配置
    parser.add_argument('--indicators', help='要分析的指标，逗号分隔，如"MACD,KDJ,RSI"')
    parser.add_argument('--disable-indicators', help='禁用的指标，逗号分隔')
    
    # 策略生成参数
    parser.add_argument('--threshold', type=int, default=2, help='形态出现次数阈值，用于策略生成')
    
    # 高级配置
    parser.add_argument('--config', help='配置文件路径，JSON格式')
    
    return parser.parse_args()


def build_config(args) -> Dict[str, Any]:
    """构建配置"""
    config = {
        'days_before': args.days_before,
        'days_after': args.days_after,
        'indicators': {}
    }
    
    # 解析周期
    if args.periods:
        periods = []
        for period_str in args.periods.split(','):
            try:
                period = Period.from_string(period_str.strip())
                periods.append(period)
            except Exception as e:
                logger.warning(f"无效的周期: {period_str}, 错误: {e}")
        
        if periods:
            config['periods'] = periods
    
    # 解析指标
    if args.indicators:
        enabled_indicators = [ind.strip() for ind in args.indicators.split(',')]
        for indicator in enabled_indicators:
            config['indicators'][indicator] = {'enabled': True}
    
    # 禁用指标
    if args.disable_indicators:
        disabled_indicators = [ind.strip() for ind in args.disable_indicators.split(',')]
        for indicator in disabled_indicators:
            config['indicators'][indicator] = {'enabled': False}
    
    # 从文件加载高级配置
    if args.config:
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
            
            # 合并配置
            for key, value in file_config.items():
                if key == 'indicators' and 'indicators' in config:
                    # 合并指标配置
                    for ind, ind_conf in value.items():
                        if ind not in config['indicators']:
                            config['indicators'][ind] = ind_conf
                else:
                    config[key] = value
                    
        except Exception as e:
            logger.error(f"加载配置文件出错: {e}")
    
    return config


def main():
    """主函数"""
    args = parse_args()
    
    try:
        # 构建配置
        config = build_config(args)
        
        # 初始化回测系统
        backtest = EnhancedBacktest.get_instance()
        
        # 执行批量分析
        logger.info(f"开始批量分析，输入文件: {args.input}")
        results = backtest.batch_analyze(args.input, args.output, config)
        
        # 生成策略
        if args.strategy and results:
            logger.info(f"开始生成策略，输出文件: {args.strategy}")
            backtest.generate_strategy(results, args.strategy, args.threshold)
        
        logger.info("分析完成")
        
    except Exception as e:
        logger.error(f"执行出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 