#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
修复ZXM_MARKET_SENTIMENT指标
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.dependency_injection import get_logger

logger = get_logger(__name__)


def fix_market_sentiment_class():
    """修复MarketSentiment类"""
    logger.info("🔧 开始修复ZXM_MARKET_SENTIMENT指标...")
    
    # 读取当前文件
    file_path = "indicators/sentiment_analysis.py"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        logger.info(f"✅ 读取文件成功: {file_path}")
        
        # 检查是否需要添加get_patterns方法
        if 'def get_patterns(' not in content:
            logger.info("🔧 添加get_patterns方法...")
            
            # 找到类的结尾位置，在最后添加get_patterns方法
            # 这里我们需要手动编辑文件
            logger.info("⚠️ 需要手动添加get_patterns方法")
            return False
        
        # 检查是否需要改进空数据处理
        if 'if data is None or data.empty:' not in content:
            logger.info("🔧 需要改进空数据处理...")
            return False
        
        logger.info("✅ 文件检查完成")
        return True
        
    except Exception as e:
        logger.error(f"❌ 修复过程中发生异常: {e}")
        return False


def main():
    """主函数"""
    success = fix_market_sentiment_class()
    
    if success:
        logger.info("✅ ZXM_MARKET_SENTIMENT修复完成")
    else:
        logger.error("❌ ZXM_MARKET_SENTIMENT修复失败，需要手动修复")
    
    return success


if __name__ == "__main__":
    main()
