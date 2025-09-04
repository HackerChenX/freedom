#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
快速指标质量检查脚本
用于快速验证少量指标的质量状态
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from scripts.comprehensive_indicator_quality_monitor import ComprehensiveIndicatorQualityMonitor
from utils.dependency_injection import get_logger

logger = get_logger(__name__)


def quick_test():
    """快速测试前10个指标"""
    logger.info("🚀 开始快速指标质量检查...")
    
    try:
        monitor = ComprehensiveIndicatorQualityMonitor()
        
        # 只测试前10个指标
        results = monitor.run_comprehensive_test(
            test_codes=['000001'],
            max_indicators=10
        )
        
        # 显示简要结果
        summary = results['summary']
        logger.info("=" * 50)
        logger.info("📊 快速检查结果:")
        logger.info(f"  - 测试指标数: {summary['total_indicators']}")
        logger.info(f"  - 通过指标数: {summary['passed_indicators']}")
        logger.info(f"  - 警告指标数: {summary['warning_indicators']}")
        logger.info(f"  - 失败指标数: {summary['failed_indicators']}")
        logger.info(f"  - 通过率: {(summary['passed_indicators']/summary['total_indicators']*100):.1f}%")
        logger.info(f"  - 执行时间: {summary['execution_time']:.2f} 秒")
        
        if summary['failed_indicators'] == 0:
            logger.info("✅ 快速检查通过！")
            return True
        else:
            logger.warning("⚠️ 发现质量问题，建议运行完整测试")
            return False
            
    except Exception as e:
        logger.error(f"❌ 快速检查失败: {e}")
        return False


if __name__ == "__main__":
    success = quick_test()
    exit(0 if success else 1)
