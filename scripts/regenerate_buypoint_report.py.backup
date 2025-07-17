#!/usr/bin/env python3
"""
重新生成买点分析报告脚本

该脚本用于重新生成优化后的买点共性指标分析报告，修复命中率计算错误并提升报告质量。
"""

import os
import sys
import json
from datetime import datetime

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from utils.logger import get_logger
from analysis.buypoints.buypoint_batch_analyzer import Buy_point_batch_analyzer

logger = get_logger(__name__)

def load_analysis_results(results_file: str):
    """加载分析结果"""
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"加载分析结果失败: {e}")
        return []

def regenerate_report():
    """重新生成买点分析报告"""
    try:
        # 设置路径
        results_dir = "data/result/zxm_methods_fix_test"
        results_file = os.path.join(results_dir, "analysis_results.json")
        report_file = os.path.join(results_dir, "common_indicators_report.md")
        
        # 检查结果文件是否存在
        if not os.path.exists(results_file):
            logger.error(f"分析结果文件不存在: {results_file}")
            return
        
        # 加载分析结果
        logger.info("加载分析结果...")
        analysis_results = load_analysis_results(results_file)
        
        if not analysis_results:
            logger.error("分析结果为空")
            return
        
        logger.info(f"加载了 {len(analysis_results)} 个买点分析结果")
        
        # 创建批量分析器
        analyzer = Buy_point_batch_analyzer()
        
        # 提取共性指标
        logger.info("提取共性指标...")
        common_indicators = analyzer.extract_common_indicators(
            buypoint_results=analysis_results,
            min_hit_ratio=0.3  # 降低阈值以获取更多指标
        )
        
        if not common_indicators:
            logger.warning("未能提取到共性指标")
            return
        
        # 统计信息
        total_indicators = sum(len(indicators) for indicators in common_indicators.values())
        logger.info(f"提取到 {len(common_indicators)} 个周期的共性指标，共 {total_indicators} 个指标形态")
        
        # 生成优化后的报告
        logger.info("生成优化后的报告...")
        analyzer._generate_indicators_report(common_indicators, report_file)
        
        logger.info(f"报告已生成: {report_file}")
        
        # 显示统计信息
        print("\n" + "="*60)
        print("📊 报告生成完成")
        print("="*60)
        print(f"📁 报告文件: {report_file}")
        print(f"📈 分析周期: {len(common_indicators)} 个")
        print(f"📊 共性指标: {total_indicators} 个")
        print(f"🕒 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 显示各周期的指标数量
        print(f"\n📋 各周期指标统计:")
        for period, indicators in common_indicators.items():
            print(f"  • {period}: {len(indicators)} 个指标")
        
        print("\n✅ 优化内容:")
        print("  • 修复了命中率计算错误（限制在0-100%）")
        print("  • 重新计算平均得分（处理0分问题）")
        print("  • 添加了详细的分析说明和应用建议")
        print("  • 优化了表格排序（按命中率和得分）")
        print("  • 增加了系统性能和技术支持信息")
        
    except Exception as e:
        logger.error(f"重新生成报告时出错: {e}")

def main_regeneratebuypointreport():
    """主函数"""
    print("🚀 开始重新生成买点分析报告...")
    print("📋 优化目标:")
    print("  1. 修复命中率计算错误")
    print("  2. 重新计算平均得分")
    print("  3. 添加详细分析说明")
    print("  4. 提升报告可读性")
    print()
    
    regenerate_report()

if __name__ == "__main__":
    main_regeneratebuypointreport()
