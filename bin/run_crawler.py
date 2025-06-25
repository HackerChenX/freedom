#!/usr/bin/env python3
"""
股市信息爬虫系统启动脚本

用于启动和管理爬虫系统
"""

import sys
import os
import argparse
import time
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crawler.config import CrawlerConfig
from crawler.processors.concept_extractor import ConceptStockExtractor
from utils.logger import get_logger

logger = get_logger(__name__)


def test_concept_extraction():
    """测试概念股提取功能"""
    logger.info("开始测试概念股提取功能...")

    # 测试文本（基于您提供的示例）
    test_text = """
    【中金油气化工】原材料成本下行叠加关税影响消化，轮胎迎来向上拐点
    天胶价格从高点下降幅度超3,000元/吨，2H25轮胎企业或明显受益。
    4-5月份国内外天然橡胶主产区先后进入开割季，橡胶供应释放，价格从2月21日的高点17,220元/吨
    逐步跌至目前的13,760元/吨，跌幅达3,460元/吨，合成橡胶价格同步下行。
    考虑原材料的库存周期，我们预计2H25轮胎企业或明显受益。
    轮胎企业逐步消化关税冲击，盈利能力有望逐步修复。
    相关企业包括中策橡胶（未覆盖）、赛轮轮胎(601058)、森麒麟(002984)、玲珑轮胎(601966)等。
    """

    # 初始化概念股提取器
    extractor = ConceptStockExtractor()

    # 提取信息
    result = extractor.extract_stocks(test_text)

    print("\n=== 概念股提取结果 ===")
    print(f"股票代码: {result['stock_codes']}")
    print(f"公司名称: {result['company_names']}")
    print(f"概念关键词: {result['concepts']}")
    print(f"置信度: {result['confidence']:.2f}")

    return result


def show_config():
    """显示配置信息"""
    print("\n=== 爬虫系统配置 ===")
    print(f"Redis: {CrawlerConfig.REDIS_HOST}:{CrawlerConfig.REDIS_PORT}")
    print(f"ClickHouse: {CrawlerConfig.CLICKHOUSE_HOST}:{CrawlerConfig.CLICKHOUSE_PORT}")
    print(f"最大工作线程: {CrawlerConfig.MAX_WORKERS}")
    print(f"使用代理: {CrawlerConfig.USE_PROXY}")

    print("\n=== 数据源配置 ===")
    for name, config in CrawlerConfig.DATA_SOURCES.items():
        status = "启用" if config['enabled'] else "禁用"
        print(f"{config['name']} ({name}): {status} - 优先级: {config['priority']}")


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='股市信息爬虫系统')
    parser.add_argument('--test', action='store_true', help='运行测试模式')
    parser.add_argument('--config', action='store_true', help='显示配置信息')
    parser.add_argument('--demo', action='store_true', help='运行演示模式')

    args = parser.parse_args()

    logger.info("=== 股市信息爬虫系统启动 ===")
    logger.info(f"启动时间: {datetime.now()}")

    if args.config:
        show_config()
    elif args.test or args.demo:
        test_concept_extraction()
    else:
        print("请使用 --help 查看可用选项")
        print("示例:")
        print("  python bin/run_crawler.py --demo    # 运行演示")
        print("  python bin/run_crawler.py --config  # 显示配置")
        print("  python bin/run_crawler.py --test    # 运行测试")


if __name__ == "__main__":
    main()