#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
通达信公式导出工具

从回测分析报告中提取通达信公式并导出为单独的公式文件
方便用户直接导入到通达信软件中使用
"""

import os
import re
import sys
import argparse
from datetime import datetime

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from utils.path_utils import get_doc_dir, ensure_dir_exists
from utils.logger import get_logger

logger = get_logger(__name__)


def extract_formulas(markdown_file):
    """从Markdown文件中提取通达信公式"""
    logger.info(f"从文件提取公式: {markdown_file}")
    
    formulas = {}
    current_name = None
    formula_content = []
    capture_mode = False
    
    with open(markdown_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            
            # 匹配公式名称
            name_match = re.search(r'###\s+\d+\.\s+(.*?)选股公式', line)
            if name_match:
                current_name = name_match.group(1).strip()
                logger.debug(f"找到公式名称: {current_name}")
                continue
                
            # 开始捕获公式内容
            if line.strip() == '```' and current_name and not capture_mode:
                capture_mode = True
                formula_content = []
                continue
                
            # 结束捕获公式内容
            if line.strip() == '```' and capture_mode:
                capture_mode = False
                formulas[current_name] = '\n'.join(formula_content)
                logger.debug(f"提取到公式: {current_name}, 长度: {len(formula_content)}行")
                continue
                
            # 捕获公式内容
            if capture_mode:
                formula_content.append(line)
    
    logger.info(f"共提取到 {len(formulas)} 个公式")
    return formulas


def export_formulas(formulas, output_dir):
    """将提取的公式导出为单独的文件"""
    logger.info(f"导出公式到目录: {output_dir}")
    ensure_dir_exists(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d")
    
    for name, content in formulas.items():
        # 构建文件名
        safe_name = name.replace(' ', '_')
        filename = f"{safe_name}选股公式_{timestamp}.txt"
        filepath = os.path.join(output_dir, filename)
        
        # 写入文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"已导出公式: {filepath}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='通达信公式导出工具')
    parser.add_argument('-f', '--file', help='回测分析报告文件路径')
    parser.add_argument('-o', '--output', help='公式导出目录')
    
    args = parser.parse_args()
    
    # 设置默认值
    if not args.file:
        doc_dir = get_doc_dir()
        # 尝试查找最新的回测分析报告
        for filename in os.listdir(doc_dir):
            if '回测分析' in filename and filename.endswith('.md'):
                args.file = os.path.join(doc_dir, filename)
                logger.info(f"自动选择回测报告文件: {args.file}")
                break
        
        if not args.file:
            logger.error("未找到回测分析报告文件，请使用-f参数指定文件路径")
            return
    
    if not args.output:
        args.output = os.path.join(root_dir, 'data', 'result', '通达信公式')
        logger.info(f"使用默认导出目录: {args.output}")
    
    # 提取并导出公式
    formulas = extract_formulas(args.file)
    if formulas:
        export_formulas(formulas, args.output)
        logger.info("公式导出完成!")
    else:
        logger.error("未从文件中提取到任何公式")


if __name__ == "__main__":
    main() 