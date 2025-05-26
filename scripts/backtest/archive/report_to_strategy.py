#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import os

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

import argparse
import json
import re
from typing import Dict, List, Any, Optional
import datetime

from utils.logger import get_logger
from utils.path_utils import get_backtest_result_dir

logger = get_logger(__name__)

def parse_markdown_report(report_file: str) -> Dict[str, Any]:
    """
    解析Markdown格式的分析报告
    
    Args:
        report_file: 报告文件路径
        
    Returns:
        Dict: 解析结果
    """
    try:
        with open(report_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 提取基本信息
        code_match = re.search(r'\((\d+)\)', content)
        code = code_match.group(1) if code_match else None
        
        name_match = re.search(r'# ([^\(]+)\(', content)
        name = name_match.group(1).strip() if name_match else None
        
        date_match = re.search(r'买点日期\*\*: (\d+)', content)
        date = date_match.group(1) if date_match else None
        
        # 提取技术特征
        patterns = []
        
        # KDJ金叉
        if 'KDJ金叉' in content:
            patterns.append('KDJ金叉')
            
        # MACD金叉
        if 'MACD金叉' in content:
            patterns.append('MACD金叉')
            
        # 均线多头排列
        if '均线多头排列' in content:
            patterns.append('均线多头排列')
            
        # 均线支撑
        if re.search(r'(收盘价.*接近.*均线|均线支撑)', content):
            patterns.append('均线支撑')
            
        # 创建结果
        result = {
            'code': code,
            'name': name,
            'date': date,
            'patterns': patterns,
            'strategy_type': 'pattern_matching'
        }
        
        return result
        
    except Exception as e:
        logger.error(f"解析报告 {report_file} 时出错: {e}")
        return {}


def generate_strategy_json(parse_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    生成策略JSON
    
    Args:
        parse_result: 解析结果
        
    Returns:
        Dict: 策略JSON
    """
    code = parse_result.get('code')
    name = parse_result.get('name')
    patterns = parse_result.get('patterns', [])
    
    if not code or not patterns:
        return {}
        
    # 创建策略条件
    conditions = []
    
    if 'KDJ金叉' in patterns:
        conditions.append({
            'type': 'indicator',
            'indicator': 'KDJ',
            'condition': 'golden_cross',
            'lookback': 3,
            'parameters': {'period': 9}
        })
        
    if 'MACD金叉' in patterns:
        conditions.append({
            'type': 'indicator',
            'indicator': 'MACD',
            'condition': 'golden_cross',
            'lookback': 3,
            'parameters': {}
        })
        
    if '均线多头排列' in patterns:
        conditions.append({
            'type': 'ma_alignment',
            'alignment': 'bull',
            'ma_periods': [5, 10, 20],
            'lookback': 1
        })
        
    if '均线支撑' in patterns:
        conditions.append({
            'type': 'price_ma_relation',
            'relation': 'near',
            'ma_period': 20,
            'threshold': 0.02,
            'lookback': 1
        })
        
    # 创建策略JSON
    strategy_name = ''
    if 'KDJ金叉' in patterns:
        strategy_name = 'kdj_golden_cross'
    elif 'MACD金叉' in patterns:
        strategy_name = 'macd_golden_cross'
    elif '均线多头排列' in patterns:
        strategy_name = 'ma_bull_alignment'
    else:
        strategy_name = 'custom_pattern'
        
    strategy_name = f"{strategy_name}_{code}"
    
    strategy = {
        'name': strategy_name,
        'description': f"{name}({code})的{','.join(patterns)}策略",
        'conditions': conditions,
        'logic': 'AND',
        'target_stocks': [code],
        'created_at': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return strategy


def convert_report_to_strategy(report_file: str, output_dir: Optional[str] = None) -> str:
    """
    将分析报告转换为选股策略
    
    Args:
        report_file: 报告文件路径
        output_dir: 输出目录路径，如果为None则使用默认目录
        
    Returns:
        str: 策略文件路径
    """
    try:
        if not output_dir:
            output_dir = f"{get_backtest_result_dir()}/strategies"
            
        os.makedirs(output_dir, exist_ok=True)
        
        # 解析报告
        parse_result = parse_markdown_report(report_file)
        
        if not parse_result:
            logger.error(f"解析报告 {report_file} 失败")
            return None
            
        # 生成策略JSON
        strategy = generate_strategy_json(parse_result)
        
        if not strategy:
            logger.error(f"生成策略失败")
            return None
            
        # 保存策略文件
        strategy_name = strategy['name']
        strategy_file = f"{output_dir}/{strategy_name}.json"
        
        with open(strategy_file, 'w', encoding='utf-8') as f:
            json.dump(strategy, f, ensure_ascii=False, indent=2)
            
        logger.info(f"策略文件已保存到 {strategy_file}")
        return strategy_file
        
    except Exception as e:
        logger.error(f"转换报告 {report_file} 为策略时出错: {e}")
        return None


def main():
    """报告转换为策略主函数"""
    parser = argparse.ArgumentParser(description="将分析报告转换为选股策略")
    
    parser.add_argument("--report", required=True, help="报告文件路径")
    parser.add_argument("--output-dir", help="输出目录路径")
    
    args = parser.parse_args()
    
    strategy_file = convert_report_to_strategy(args.report, args.output_dir)
    
    if strategy_file:
        logger.info(f"转换成功，策略文件: {strategy_file}")
    else:
        logger.error("转换失败")
        sys.exit(1)

if __name__ == "__main__":
    main() 