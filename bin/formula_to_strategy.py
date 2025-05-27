#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
通达信公式转换工具

将通达信风格的选股公式转换为系统选股策略
"""

import os
import sys
import argparse
import json
import yaml
from datetime import datetime

# 将项目根目录添加到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from formula.formula_converter import FormulaConverter, FormulaEditor
from strategy.strategy_manager import StrategyManager
from utils.logger import get_logger
from utils.path_utils import get_strategy_dir, get_result_dir

logger = get_logger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='通达信公式转换为选股策略')
    
    # 公式源参数
    source_group = parser.add_argument_group('公式源')
    source_group.add_argument('-f', '--file', help='通达信公式文件路径')
    source_group.add_argument('-t', '--text', help='通达信公式文本')
    
    # 策略信息
    strategy_group = parser.add_argument_group('策略信息')
    strategy_group.add_argument('-n', '--name', help='策略名称，默认从公式注释中提取')
    strategy_group.add_argument('-d', '--desc', help='策略描述，默认从公式注释中提取')
    strategy_group.add_argument('-a', '--author', default='system', help='策略作者，默认为system')
    
    # 操作选项
    operation_group = parser.add_argument_group('操作选项')
    operation_group.add_argument('--validate', action='store_true', help='仅验证公式语法')
    operation_group.add_argument('--test', help='测试公式，需要提供股票代码')
    operation_group.add_argument('--save', action='store_true', help='保存生成的策略')
    
    # 输出选项
    output_group = parser.add_argument_group('输出选项')
    output_group.add_argument('-o', '--output', help='输出文件路径，支持.json和.yaml格式')
    output_group.add_argument('--pretty', action='store_true', help='美化输出')
    
    return parser.parse_args()


def load_formula(file_path=None, formula_text=None):
    """
    加载通达信公式
    
    Args:
        file_path: 公式文件路径
        formula_text: 公式文本
        
    Returns:
        公式文本
    """
    if formula_text:
        return formula_text
    elif file_path:
        if not os.path.exists(file_path):
            logger.error(f"公式文件 {file_path} 不存在")
            sys.exit(1)
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"读取公式文件失败: {e}")
            sys.exit(1)
    else:
        logger.error("必须提供公式文件或公式文本")
        sys.exit(1)


def validate_formula(formula_text):
    """
    验证公式语法
    
    Args:
        formula_text: 公式文本
    """
    editor = FormulaEditor()
    is_valid, message = editor.validate_formula(formula_text)
    
    if is_valid:
        print("公式语法验证通过")
        print(f"提示: {message}")
    else:
        print("公式语法验证失败")
        print(f"错误: {message}")
        sys.exit(1)


def test_formula(formula_text, stock_code, start_date=None, end_date=None):
    """
    测试公式
    
    Args:
        formula_text: 公式文本
        stock_code: 股票代码
        start_date: 开始日期
        end_date: 结束日期
    """
    editor = FormulaEditor()
    result = editor.test_formula(formula_text, stock_code, start_date, end_date)
    
    if result["success"]:
        print("公式测试成功")
        print(f"买入条件: {result['buy_condition']}")
        
        # 显示策略配置摘要
        strategy_config = result["strategy_config"]["strategy"]
        print("\n策略配置摘要:")
        print(f"ID: {strategy_config['id']}")
        print(f"名称: {strategy_config['name']}")
        print(f"描述: {strategy_config['description']}")
        print(f"条件数量: {len(strategy_config['conditions'])}")
    else:
        print("公式测试失败")
        print(f"错误: {result['message']}")
        sys.exit(1)


def convert_formula(formula_text, strategy_name=None, strategy_desc=None, author="system"):
    """
    转换公式为策略
    
    Args:
        formula_text: 公式文本
        strategy_name: 策略名称
        strategy_desc: 策略描述
        author: 策略作者
        
    Returns:
        策略配置字典
    """
    converter = FormulaConverter()
    
    try:
        strategy_config = converter.convert(
            formula_text,
            strategy_name=strategy_name or "",
            strategy_desc=strategy_desc or "",
            author=author
        )
        
        return strategy_config
    except Exception as e:
        logger.error(f"转换公式失败: {e}")
        print(f"转换公式失败: {e}")
        sys.exit(1)


def save_strategy(strategy_config):
    """
    保存策略
    
    Args:
        strategy_config: 策略配置
        
    Returns:
        策略ID
    """
    strategy_manager = StrategyManager()
    
    try:
        strategy_id = strategy_manager.create_strategy(strategy_config)
        print(f"策略已保存，ID: {strategy_id}")
        return strategy_id
    except Exception as e:
        logger.error(f"保存策略失败: {e}")
        print(f"保存策略失败: {e}")
        sys.exit(1)


def save_to_file(strategy_config, output_file, pretty=False):
    """
    保存策略配置到文件
    
    Args:
        strategy_config: 策略配置
        output_file: 输出文件路径
        pretty: 是否美化输出
    """
    if not output_file:
        return
        
    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        # 根据文件扩展名确定格式
        file_ext = os.path.splitext(output_file)[1].lower()
        
        if file_ext == '.json':
            with open(output_file, 'w', encoding='utf-8') as f:
                if pretty:
                    json.dump(strategy_config, f, ensure_ascii=False, indent=2)
                else:
                    json.dump(strategy_config, f, ensure_ascii=False)
        elif file_ext in ['.yml', '.yaml']:
            with open(output_file, 'w', encoding='utf-8') as f:
                yaml.dump(strategy_config, f, allow_unicode=True, default_flow_style=False)
        else:
            # 默认为JSON
            output_file = output_file + '.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                if pretty:
                    json.dump(strategy_config, f, ensure_ascii=False, indent=2)
                else:
                    json.dump(strategy_config, f, ensure_ascii=False)
                    
        print(f"策略配置已保存到文件: {output_file}")
    except Exception as e:
        logger.error(f"保存策略配置到文件失败: {e}")
        print(f"保存策略配置到文件失败: {e}")


def main():
    """主函数"""
    args = parse_args()
    
    # 加载公式
    formula_text = load_formula(args.file, args.text)
    
    # 验证公式
    if args.validate:
        validate_formula(formula_text)
        return
        
    # 测试公式
    if args.test:
        test_formula(formula_text, args.test)
        return
        
    # 转换公式
    strategy_config = convert_formula(
        formula_text,
        strategy_name=args.name,
        strategy_desc=args.desc,
        author=args.author
    )
    
    # 显示策略信息
    strategy_info = strategy_config["strategy"]
    print("\n生成的策略信息:")
    print(f"ID: {strategy_info['id']}")
    print(f"名称: {strategy_info['name']}")
    print(f"描述: {strategy_info['description']}")
    print(f"条件数量: {len(strategy_info['conditions'])}")
    
    # 保存策略
    if args.save:
        strategy_id = save_strategy(strategy_config)
    
    # 保存到文件
    if args.output:
        save_to_file(strategy_config, args.output, args.pretty)


if __name__ == '__main__':
    main() 