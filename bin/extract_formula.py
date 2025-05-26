#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
通达信公式提取工具

从回测分析报告中提取通达信公式并导出为单独文件，方便导入到通达信软件中
"""

import os
import sys

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.logger import get_logger
from scripts.utils.formula_exporter import extract_formulas, export_formulas
from utils.path_utils import get_doc_dir, ensure_dir_exists

logger = get_logger(__name__)


def main():
    """主函数：从命令行运行公式提取工具"""
    # 初始化日志
    logger.info("启动通达信公式提取工具")

    import argparse
    
    parser = argparse.ArgumentParser(description='通达信公式提取工具')
    parser.add_argument('-f', '--file', help='指定回测分析报告文件路径')
    parser.add_argument('-o', '--output', help='指定公式导出目录')
    parser.add_argument('-a', '--all', action='store_true', help='处理所有回测分析报告文件')
    
    args = parser.parse_args()
    
    # 设置默认输出目录
    if not args.output:
        args.output = os.path.join(root_dir, 'data', 'result', '通达信公式')
        logger.info(f"使用默认输出目录: {args.output}")
    
    ensure_dir_exists(args.output)
    
    # 处理所有回测报告
    if args.all:
        doc_dir = get_doc_dir()
        report_files = []
        
        for filename in os.listdir(doc_dir):
            if '回测分析' in filename and filename.endswith('.md'):
                report_files.append(os.path.join(doc_dir, filename))
        
        if not report_files:
            logger.error("未找到任何回测分析报告文件")
            return
        
        logger.info(f"找到 {len(report_files)} 个回测分析报告文件")
        
        for report_file in report_files:
            logger.info(f"处理文件: {os.path.basename(report_file)}")
            formulas = extract_formulas(report_file)
            if formulas:
                export_formulas(formulas, args.output)
            else:
                logger.warning(f"从文件 {os.path.basename(report_file)} 中未提取到任何公式")
    
    # 处理单个指定文件
    elif args.file:
        if not os.path.exists(args.file):
            logger.error(f"文件不存在: {args.file}")
            return
            
        formulas = extract_formulas(args.file)
        if formulas:
            export_formulas(formulas, args.output)
            logger.info("公式导出完成!")
        else:
            logger.error("未从文件中提取到任何公式")
    
    # 自动查找最新的回测报告
    else:
        doc_dir = get_doc_dir()
        latest_file = None
        latest_mtime = 0
        
        for filename in os.listdir(doc_dir):
            if '回测分析' in filename and filename.endswith('.md'):
                file_path = os.path.join(doc_dir, filename)
                mtime = os.path.getmtime(file_path)
                
                if mtime > latest_mtime:
                    latest_mtime = mtime
                    latest_file = file_path
        
        if latest_file:
            logger.info(f"自动选择最新的回测报告: {os.path.basename(latest_file)}")
            formulas = extract_formulas(latest_file)
            if formulas:
                export_formulas(formulas, args.output)
                logger.info("公式导出完成!")
            else:
                logger.error("未从文件中提取到任何公式")
        else:
            logger.error("未找到任何回测分析报告文件")


if __name__ == "__main__":
    main() 