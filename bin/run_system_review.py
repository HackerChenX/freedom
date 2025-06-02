#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
系统回顾测试脚本

用于启动系统回顾测试并生成分析报告
"""

import sys
import os
import subprocess
import time
from datetime import datetime

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from utils.logger import get_logger

# 获取日志记录器
logger = get_logger("system_review")


def main():
    """主函数"""
    start_time = time.time()
    
    # 确保目录存在
    review_dir = os.path.join(root_dir, "tests", "review")
    os.makedirs(review_dir, exist_ok=True)
    
    report_dir = os.path.join(root_dir, "doc", "测试报告")
    os.makedirs(report_dir, exist_ok=True)
    
    # 记录开始信息
    logger.info("="*80)
    logger.info(f"开始系统回顾测试: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    
    # 运行测试启动器
    review_script = os.path.join(review_dir, "run_all_tests.py")
    
    # 检查脚本是否存在
    if not os.path.exists(review_script):
        logger.error(f"回顾测试脚本不存在: {review_script}")
        return 1
    
    # 执行测试
    try:
        logger.info(f"执行回顾测试脚本: {review_script}")
        result = subprocess.run([sys.executable, review_script], check=True)
        
        if result.returncode != 0:
            logger.error("回顾测试执行失败")
            return result.returncode
            
    except subprocess.CalledProcessError as e:
        logger.error(f"回顾测试执行出错: {e}")
        return e.returncode
    except Exception as e:
        logger.error(f"执行回顾测试时发生未知错误: {e}")
        return 1
    
    # 计算总耗时
    end_time = time.time()
    duration = end_time - start_time
    
    # 记录完成信息
    logger.info("="*80)
    logger.info(f"系统回顾测试完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"总耗时: {duration:.2f} 秒")
    logger.info("测试报告已生成到: " + os.path.join(report_dir))
    logger.info("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 