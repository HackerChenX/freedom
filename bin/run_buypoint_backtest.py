#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
买点回测主程序
实现您设想的完整买点回测工作流程
"""

import sys
import os
import argparse
from datetime import datetime

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.logger import get_logger
from analysis.buypoints.buypoint_backtest_engine import BuyPointBacktestEngine

logger = get_logger(__name__)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='买点回测分析工具')
    parser.add_argument('--buypoints', '-b', 
                       default='data/buypoints.csv',
                       help='买点文件路径 (默认: data/buypoints.csv)')
    parser.add_argument('--verbose', '-v', 
                       action='store_true',
                       help='详细输出模式')
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        logger.info("=" * 80)
        logger.info("🚀 买点回测分析系统启动")
        logger.info("=" * 80)
        logger.info(f"📋 买点文件: {args.buypoints}")
        logger.info(f"🕐 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 检查买点文件是否存在
        if not os.path.exists(args.buypoints):
            logger.error(f"❌ 买点文件不存在: {args.buypoints}")
            logger.info("💡 请确保buypoints.csv文件存在，格式如下:")
            logger.info("   stock_code,buypoint_date")
            logger.info("   603359,20250512")
            logger.info("   000001,20250520")
            return 1
        
        # 创建回测引擎
        logger.info("🔧 初始化买点回测引擎...")
        engine = BuyPointBacktestEngine()
        
        # 运行回测
        logger.info("🎯 开始执行买点回测...")
        results = engine.run_buypoint_backtest(args.buypoints)
        
        # 显示结果摘要
        display_results_summary(results)
        
        logger.info("=" * 80)
        logger.info("🎉 买点回测分析完成！")
        logger.info("=" * 80)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("⚠️ 用户中断执行")
        return 1
    except Exception as e:
        logger.error(f"❌ 执行失败: {e}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        return 1


def display_results_summary(results):
    """显示结果摘要"""
    try:
        logger.info("\n" + "=" * 60)
        logger.info("📊 回测结果摘要")
        logger.info("=" * 60)
        
        # 基本统计
        summary = results.get('summary', {})
        logger.info(f"📋 处理买点数: {results.get('buypoints_processed', 0)}")
        logger.info(f"✅ 成功分析: {summary.get('successful_analysis', 0)}")
        logger.info(f"❌ 失败分析: {summary.get('failed_analysis', 0)}")
        
        # 热门"指标+周期"组合
        top_combinations = summary.get('top_indicator_period_combinations', {})
        if top_combinations:
            logger.info("\n🏆 热门指标+周期组合排行 (前5名):")
            for i, (combo_key, combo_info) in enumerate(list(top_combinations.items())[:5], 1):
                display_name = combo_info.get('display_name', combo_key)
                count = combo_info.get('count', 0)
                logger.info(f"  {i}. {display_name}: {count}次命中")

        # 仅指标名排行（参考）
        top_indicators_only = summary.get('top_indicators_only', {})
        if top_indicators_only:
            logger.info("\n📊 仅指标名统计 (参考，前3名):")
            for i, (indicator, count) in enumerate(list(top_indicators_only.items())[:3], 1):
                logger.info(f"  {i}. {indicator}: {count}次命中 (跨所有周期)")
        
        # 周期效果
        period_effectiveness = summary.get('period_effectiveness', {})
        if period_effectiveness:
            logger.info("\n📈 周期效果分析:")
            for period, stats in period_effectiveness.items():
                success_rate = stats.get('success_rate', 0) * 100
                avg_hits = stats.get('avg_hits', 0)
                logger.info(f"  {period:8s}: 平均{avg_hits:.1f}个指标命中, 成功率{success_rate:.1f}%")
        
        # 平均评分
        avg_scores = summary.get('average_scores', {})
        if avg_scores:
            logger.info(f"\n📊 平均评分: {avg_scores.get('mean', 0):.1f}分")
            logger.info(f"   中位数: {avg_scores.get('median', 0):.1f}分")
            logger.info(f"   标准差: {avg_scores.get('std', 0):.1f}")
        
        # 生成的策略
        strategies = results.get('generated_strategies', [])
        if strategies:
            logger.info(f"\n🎯 生成选股策略: {len(strategies)}个")
            for i, strategy in enumerate(strategies[:3], 1):  # 显示前3个
                logger.info(f"  策略{i}: {strategy['name']}")
                logger.info(f"    成功次数: {strategy['success_count']}")
                logger.info(f"    平均评分: {strategy['avg_score']:.1f}分")
                logger.info(f"    形态数量: {len(strategy['patterns'])}")
        
        # 验证结果
        verification = results.get('verification_results', {})
        if verification.get('verification_success'):
            selected = verification.get('selected_stocks', {})
            logger.info(f"\n✅ 双向验证成功!")
            logger.info(f"   验证策略: {selected.get('strategy_name', 'N/A')}")
            logger.info(f"   选出股票: {selected.get('count', 0)}只")
            if 'stocks' in selected and selected['stocks']:
                logger.info(f"   股票列表: {', '.join(selected['stocks'])}")
        
        # 文件保存信息
        logger.info(f"\n📄 详细结果已保存到 results/buypoint_backtest/ 目录")
        
    except Exception as e:
        logger.error(f"显示结果摘要失败: {e}")


def create_sample_buypoints():
    """创建示例买点文件"""
    sample_content = """stock_code,buypoint_date
603359,20250512
000001,20250520
600036,20250515
000858,20250518
600000,20250522"""
    
    try:
        os.makedirs('data', exist_ok=True)
        with open('data/buypoints.csv', 'w', encoding='utf-8') as f:
            f.write(sample_content)
        logger.info("✅ 已创建示例买点文件: data/buypoints.csv")
        return True
    except Exception as e:
        logger.error(f"创建示例文件失败: {e}")
        return False


def check_system_requirements():
    """检查系统要求"""
    try:
        # 检查必要的模块
        required_modules = [
            'pandas', 'numpy', 'clickhouse_driver'
        ]
        
        missing_modules = []
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing_modules.append(module)
        
        if missing_modules:
            logger.error(f"❌ 缺少必要模块: {', '.join(missing_modules)}")
            logger.info("💡 请安装缺少的模块:")
            logger.info(f"   pip install {' '.join(missing_modules)}")
            return False
        
        # 检查数据目录
        if not os.path.exists('data'):
            logger.info("📁 创建数据目录...")
            os.makedirs('data', exist_ok=True)
        
        # 检查结果目录
        if not os.path.exists('results'):
            logger.info("📁 创建结果目录...")
            os.makedirs('results', exist_ok=True)
        
        return True
        
    except Exception as e:
        logger.error(f"系统检查失败: {e}")
        return False


if __name__ == "__main__":
    # 检查系统要求
    if not check_system_requirements():
        sys.exit(1)
    
    # 如果没有买点文件，提供创建选项
    if not os.path.exists('data/buypoints.csv'):
        logger.warning("⚠️ 未找到买点文件 data/buypoints.csv")
        response = input("是否创建示例买点文件? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            if not create_sample_buypoints():
                sys.exit(1)
        else:
            logger.info("💡 请手动创建 data/buypoints.csv 文件")
            sys.exit(1)
    
    # 运行主程序
    exit_code = main()
    sys.exit(exit_code)
