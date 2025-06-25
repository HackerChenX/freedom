"""
股市信息爬虫系统演示

展示如何使用爬虫系统获取股市信息
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crawler.processors.concept_extractor import ConceptStockExtractor
from utils.logger import get_logger

logger = get_logger(__name__)


def demo_concept_extraction():
    """演示概念股提取"""
    logger.info("开始演示概念股提取...")

    # 测试文本
    test_text = """
    【中金油气化工】原材料成本下行叠加关税影响消化，轮胎迎来向上拐点
    天胶价格从高点下降幅度超3,000元/吨，2H25轮胎企业或明显受益。
    相关企业包括中策橡胶（未覆盖）、赛轮轮胎(601058)、森麒麟(002984)、玲珑轮胎(601966)等。
    新能源汽车、锂电池、光伏等概念值得关注。
    """

    # 初始化概念股提取器
    extractor = ConceptStockExtractor()

    # 提取信息
    result = extractor.extract_stocks(test_text)

    logger.info("提取结果:")
    logger.info(f"股票代码: {result['stock_codes']}")
    logger.info(f"公司名称: {result['company_names']}")
    logger.info(f"概念关键词: {result['concepts']}")
    logger.info(f"置信度: {result['confidence']:.2f}")

    return result


def main():
    """主函数"""
    logger.info("=== 股市信息爬虫系统演示 ===")

    # 演示概念股提取
    print("\n1. 概念股提取演示:")
    concept_result = demo_concept_extraction()

    print("\n2. 爬虫功能演示:")
    print("（为避免对目标网站造成压力，此演示已注释）")

    logger.info("演示完成")


if __name__ == "__main__":
    main()