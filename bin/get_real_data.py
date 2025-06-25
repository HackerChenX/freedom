#!/usr/bin/env python3
"""
真实股市数据获取脚本

通过公开API获取真实的股市数据
"""

import sys
import os
import json
import argparse
from datetime import datetime
from typing import Dict, List, Any

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crawler.spiders.api_data_spider import APIDataSpider
from crawler.processors.concept_extractor import ConceptStockExtractor
from utils.logger import get_logger

logger = get_logger(__name__)


class RealDataCollector:
    """真实数据收集器"""

    def __init__(self):
        self.api_spider = APIDataSpider()
        self.concept_extractor = ConceptStockExtractor()

        # 热门股票代码列表
        self.hot_stocks = [
            '000001',  # 平安银行
            '000002',  # 万科A
            '000858',  # 五粮液
            '002594',  # 比亚迪
            '300750',  # 宁德时代
            '601058',  # 赛轮轮胎
            '002984',  # 森麒麟
            '601966',  # 玲珑轮胎
            '600519',  # 贵州茅台
            '600036',  # 招商银行
        ]

    def collect_real_stock_data(self) -> List[Dict[str, Any]]:
        """收集真实股票数据"""
        logger.info("开始收集真实股票数据...")
        print("正在获取股票实时数据...")

        stock_data = self.api_spider.get_stock_realtime_data(self.hot_stocks)

        logger.info(f"成功收集 {len(stock_data)} 只股票的实时数据")
        return stock_data

    def analyze_stock_data(self, stock_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析股票数据"""
        if not stock_data:
            return {}

        analysis = {
            'total_stocks': len(stock_data),
            'rising_stocks': 0,
            'falling_stocks': 0,
            'flat_stocks': 0,
            'top_gainers': [],
            'top_losers': [],
            'analysis_time': datetime.now()
        }

        gains = []

        for stock in stock_data:
            current = stock.get('current_price', 0)
            yesterday = stock.get('yesterday_close', 0)

            if current > 0 and yesterday > 0:
                change_pct = (current - yesterday) / yesterday * 100
                stock['change_pct'] = change_pct
                gains.append((stock['stock_code'], stock['name'], change_pct))

                if change_pct > 0:
                    analysis['rising_stocks'] += 1
                elif change_pct < 0:
                    analysis['falling_stocks'] += 1
                else:
                    analysis['flat_stocks'] += 1

        # 排序获取涨跌幅榜
        gains.sort(key=lambda x: x[2], reverse=True)
        analysis['top_gainers'] = gains[:5]
        analysis['top_losers'] = gains[-5:]

        return analysis

    def extract_concepts_from_stocks(self, stock_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """从股票数据中提取概念信息"""
        concept_analysis = {
            'detected_stocks': [],
            'concept_keywords': [],
            'total_confidence': 0.0
        }

        # 构建包含股票信息的文本
        stock_text = "当前关注的股票包括："
        for stock in stock_data:
            stock_name = stock.get('name', '')
            stock_code = stock.get('stock_code', '')
            if stock_name and stock_code:
                stock_text += f"{stock_name}({stock_code})、"

        # 使用概念提取器分析
        extraction_result = self.concept_extractor.extract_stocks(stock_text)

        concept_analysis['detected_stocks'] = extraction_result.get('stock_codes', [])
        concept_analysis['concept_keywords'] = extraction_result.get('concepts', [])
        concept_analysis['total_confidence'] = extraction_result.get('confidence', 0.0)

        return concept_analysis


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='真实股市数据获取工具')
    parser.add_argument('--output', default='real_stock_data.json',
                       help='输出文件名')
    parser.add_argument('--stocks', nargs='+',
                       help='指定股票代码列表')

    args = parser.parse_args()

    print("=== 真实股市数据获取工具 ===")
    print(f"启动时间: {datetime.now()}")
    print("=" * 50)

    # 创建数据收集器
    collector = RealDataCollector()

    # 如果指定了股票代码，使用指定的代码
    if args.stocks:
        collector.hot_stocks = args.stocks
        print(f"使用指定股票代码: {args.stocks}")

    try:
        # 收集股票数据
        stock_data = collector.collect_real_stock_data()

        if stock_data:
            print(f"\n=== 获取到 {len(stock_data)} 只股票的实时数据 ===")

            # 显示股票数据
            for i, stock in enumerate(stock_data, 1):
                name = stock.get('name', 'Unknown')
                code = stock.get('stock_code', 'Unknown')
                current = stock.get('current_price', 0)
                yesterday = stock.get('yesterday_close', 0)

                if current > 0 and yesterday > 0:
                    change = current - yesterday
                    change_pct = (change / yesterday) * 100
                    change_str = f"{change:+.2f} ({change_pct:+.2f}%)"
                else:
                    change_str = "N/A"

                print(f"{i:2d}. {name}({code}): {current:.2f} {change_str}")

            # 分析数据
            analysis = collector.analyze_stock_data(stock_data)

            print(f"\n=== 市场分析 ===")
            print(f"上涨股票: {analysis['rising_stocks']}")
            print(f"下跌股票: {analysis['falling_stocks']}")
            print(f"平盘股票: {analysis['flat_stocks']}")

            if analysis['top_gainers']:
                print(f"\n涨幅榜前5:")
                for code, name, pct in analysis['top_gainers']:
                    print(f"  {name}({code}): +{pct:.2f}%")

            if analysis['top_losers']:
                print(f"\n跌幅榜前5:")
                for code, name, pct in analysis['top_losers']:
                    print(f"  {name}({code}): {pct:.2f}%")

            # 概念分析
            concept_analysis = collector.extract_concepts_from_stocks(stock_data)

            print(f"\n=== 概念分析 ===")
            print(f"识别到的股票代码: {concept_analysis['detected_stocks']}")
            print(f"相关概念: {concept_analysis['concept_keywords']}")
            print(f"分析置信度: {concept_analysis['total_confidence']:.2f}")

            # 保存数据
            output_data = {
                'collection_time': datetime.now().isoformat(),
                'stock_data': stock_data,
                'market_analysis': analysis,
                'concept_analysis': concept_analysis
            }

            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2, default=str)

            print(f"\n数据已保存到: {args.output}")
            print("🎉 真实数据获取完成！")

        else:
            print("❌ 未获取到任何股票数据")
            print("可能的原因:")
            print("1. 网络连接问题")
            print("2. API服务不可用")
            print("3. 股票代码错误")

    except Exception as e:
        logger.error(f"数据获取过程中发生错误: {e}")
        print(f"❌ 数据获取失败: {e}")


if __name__ == "__main__":
    main()