#!/usr/bin/env python3
"""
重新运行买点分析脚本
使用修复后的指标代码重新生成分析数据
"""

import os
import sys
import json
from datetime import datetime

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from utils.logger import get_logger
from analysis.buypoints.auto_indicator_analyzer import AutoIndicatorAnalyzer
from analysis.buypoints.period_data_processor import PeriodDataProcessor

logger = get_logger(__name__)

def main():
    """重新运行买点分析"""
    print("🚀 开始重新运行买点分析...")
    print("📋 目标:")
    print("  1. 使用修复后的指标代码")
    print("  2. 重新生成分析数据")
    print("  3. 消除'未知形态'问题")
    print("  4. 生成标准化的技术形态描述")
    print()

    try:
        # 初始化数据处理器和分析器
        data_processor = PeriodDataProcessor()
        analyzer = AutoIndicatorAnalyzer()

        # 设置分析参数
        stock_code = "000001"  # 平安银行
        buypoint_date = "2024-12-16"
        periods = ['15min', 'daily', 'weekly', 'monthly', '30min', '60min']

        print(f"📊 分析股票: {stock_code}")
        print(f"📅 买点日期: {buypoint_date}")
        print(f"📈 分析周期: {periods}")
        print()

        # 运行分析
        logger.info(f"开始分析股票 {stock_code} 在 {buypoint_date} 的买点...")

        # 获取各周期数据
        logger.info("获取多周期数据...")
        period_data = data_processor.get_multi_period_data(stock_code, buypoint_date, periods)

        if not period_data:
            logger.error("未能获取任何周期的数据")
            return False

        # 记录获取到的数据
        for period, data in period_data.items():
            if data is not None and not data.empty:
                logger.info(f"成功获取 {period} 周期数据，共 {len(data)} 条记录")
            else:
                logger.warning(f"未能获取 {period} 周期数据")

        # 分析各周期指标
        result = {
            'stock_code': stock_code,
            'buypoint_date': buypoint_date,
            'period_analysis': {}
        }

        # 构建目标行索引字典（使用最后一行作为分析目标）
        target_rows = {}
        for period, data in period_data.items():
            if data is not None and not data.empty:
                target_rows[period] = len(data) - 1

        logger.info("开始分析所有周期指标...")
        all_results = analyzer.analyze_all_indicators(period_data, target_rows)

        # 转换结果格式
        for period, indicators in all_results.items():
            if indicators:
                result['period_analysis'][period] = {
                    'indicators': indicators
                }
                logger.info(f"成功分析 {period} 周期，找到 {len(indicators)} 个指标")
            else:
                logger.warning(f"分析 {period} 周期失败")

        if result['period_analysis']:
            # 保存结果
            output_dir = "data/result/zxm_methods_fix_test"
            os.makedirs(output_dir, exist_ok=True)

            output_file = os.path.join(output_dir, "analysis_results_fixed.json")

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump([result], f, ensure_ascii=False, indent=2)

            logger.info(f"分析结果已保存到: {output_file}")

            # 统计指标数量
            total_indicators = 0
            unknown_patterns = 0

            for period_data in result.get('period_analysis', {}).values():
                for indicator in period_data.get('indicators', []):
                    total_indicators += 1
                    if indicator.get('pattern_name') == '未知形态':
                        unknown_patterns += 1

            print("============================================================")
            print("📊 分析完成")
            print("============================================================")
            print(f"📁 结果文件: {output_file}")
            print(f"📈 总指标数: {total_indicators}")
            print(f"❌ 未知形态: {unknown_patterns}")
            if total_indicators > 0:
                print(f"✅ 修复率: {((total_indicators - unknown_patterns) / total_indicators * 100):.1f}%")
            print()

            if unknown_patterns == 0:
                print("🎉 所有'未知形态'已成功修复！")
            else:
                print(f"⚠️  仍有 {unknown_patterns} 个'未知形态'需要进一步修复")

            return True

        else:
            logger.error("买点分析失败 - 未能分析任何周期")
            return False
            
    except Exception as e:
        logger.error(f"重新运行买点分析时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
