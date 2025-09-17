#!/usr/bin/env python3
"""
增强版真实数据获取脚本

使用多种方式获取真实的股市数据
"""

import sys
import os
import json
import requests
import argparse
from datetime import datetime
from typing import Dict, List, Any

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crawler.processors.concept_extractor import Concept_stock_extractor
from utils.logger import get_logger
from db.sql_manager import SQLManager, QueryType

logger = get_logger(__name__)


class EnhancedDataCollector:
    """增强版数据收集器"""

    def __init___7(self):
        self.concept_extractor = Concept_stock_extractor()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

        # 热门股票代码列表
        self.hot_stocks = [
            '000001',  # 平安银行
            '000002',  # 万科A
            '002594',  # 比亚迪
            '300750',  # 宁德时代
            '601058',  # 赛轮轮胎
            '002984',  # 森麒麟
            '601966',  # 玲珑轮胎
            '600519',  # 贵州茅台
            '600036',  # 招商银行
            '000858',  # 五粮液
        ]

    def get_stock_data_from_sina(self, stock_codes: List[str]) -> List[Dict[str, Any]]:
        """从新浪财经获取股票数据"""
        results = []

        for code in stock_codes:
            try:
                # 构建API URL
                if code.startswith('6'):
                    symbol = f"sh{code}"
                else:
                    symbol = f"sz{code}"

                url = f"https://hq.sinajs.cn/list={symbol}"

                response = self.session.get(url, timeout=10)
                if response.status_code == 200:
                    content = response.text
                    if 'var hq_str_' in content and '""' not in content:
                        data_str = content.split('"')[1]
                        data_parts = data_str.split(',')

                        if len(data_parts) >= 32 and data_parts[0]:
                            stock_data = {
                                'stock_code': code,
                                'name': data_parts[0],
                                'open_price': float(data_parts[1]) if data_parts[1] else 0,
                                'yesterday_close': float(data_parts[2]) if data_parts[2] else 0,
                                'current_price': float(data_parts[3]) if data_parts[3] else 0,
                                'high_price': float(data_parts[4]) if data_parts[4] else 0,
                                'low_price': float(data_parts[5]) if data_parts[5] else 0,
                                'volume': int(data_parts[8]) if data_parts[8] else 0,
                                'amount': float(data_parts[9]) if data_parts[9] else 0,
                                'date': data_parts[30],
                                'time': data_parts[31],
                                'source': 'sina_finance',
                                'crawl_time': datetime.now()
                            }

                            # 计算涨跌幅
                            if stock_data['current_price'] > 0 and stock_data['yesterday_close'] > 0:
                                change = stock_data['current_price'] - stock_data['yesterday_close']
                                change_pct = (change / stock_data['yesterday_close']) * 100
                                stock_data['change'] = change
                                stock_data['change_pct'] = change_pct

                            results.append(stock_data)
                            print(f"✅ 获取 {stock_data['name']}({code}) 数据成功")
                        else:
                            print(f"❌ {code} 数据格式异常")
                    else:
                        print(f"❌ {code} 无数据返回")
                else:
                    print(f"❌ {code} 请求失败: {response.status_code}")

            except Exception as e:
                print(f"❌ 获取 {code} 数据失败: {e}")

        return results

    def collect_real_stock_data(self) -> List[Dict[str, Any]]:
        """收集真实股票数据"""
        print("正在尝试获取股票数据...")

        # 尝试新浪财经
        print("\n📊 尝试新浪财经API...")
        sina_data = self.get_stock_data_from_sina(self.hot_stocks)

        if sina_data:
            print(f"✅ 新浪财经获取到 {len(sina_data)} 只股票数据")
            return sina_data

        print("❌ 无法获取股票数据")
        return []

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
            change_pct = stock.get('change_pct', 0)

            if change_pct > 0:
                analysis['rising_stocks'] += 1
            elif change_pct < 0:
                analysis['falling_stocks'] += 1
            else:
                analysis['flat_stocks'] += 1

            gains.append((stock['stock_code'], stock['name'], change_pct))

        # 排序获取涨跌幅榜
        gains.sort(key=lambda x: x[2], reverse=True)
        analysis['top_gainers'] = gains[:5]
        analysis['top_losers'] = gains[-5:]

        return analysis


def main_12():
    """主函数"""
    parser = argparse.ArgumentParser(description='增强版真实股市数据获取工具')
    parser.add_argument('--output', default='real_stock_data_enhanced.json',
                       help='输出文件名')
    parser.add_argument('--stocks', nargs='+',
                       help='指定股票代码列表')

    args = parser.parse_args()

    print("=== 增强版真实股市数据获取工具 ===")
    print(f"启动时间: {datetime.now()}")
    print("=" * 50)

    # 创建数据收集器
    collector = Enhanced_data_collector()

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
                change = stock.get('change', 0)
                change_pct = stock.get('change_pct', 0)

                if change_pct > 0:
                    change_str = f"+{change:.2f} (+{change_pct:.2f}%)"
                    status = "📈"
                elif change_pct < 0:
                    change_str = f"{change:.2f} ({change_pct:.2f}%)"
                    status = "📉"
                else:
                    change_str = "0.00 (0.00%)"
                    status = "➡️"

                print(f"{i:2d}. {status} {name}({code}): {current:.2f} {change_str}")

            # 分析数据
            analysis = collector.analyze_stock_data(stock_data)

            print(f"\n=== 市场分析 ===")
            print(f"📈 上涨股票: {analysis['rising_stocks']}")
            print(f"📉 下跌股票: {analysis['falling_stocks']}")
            print(f"➡️ 平盘股票: {analysis['flat_stocks']}")

            if analysis['top_gainers']:
                print(f"\n🏆 涨幅榜前5:")
                for code, name, pct in analysis['top_gainers']:
                    print(f"  📈 {name}({code}): +{pct:.2f}%")

            if analysis['top_losers']:
                print(f"\n📉 跌幅榜前5:")
                for code, name, pct in analysis['top_losers']:
                    print(f"  📉 {name}({code}): {pct:.2f}%")

            # 概念分析
            stock_text = "当前关注的股票包括："
            for stock in stock_data:
                stock_name = stock.get('name', '')
                stock_code = stock.get('stock_code', '')
                if stock_name and stock_code:
                    stock_text += f"{stock_name}({stock_code})、"

            extraction_result = collector.concept_extractor.extract_stocks(stock_text)

            print(f"\n=== 概念分析 ===")
            print(f"🎯 识别到的股票代码: {extraction_result.get('stock_codes', [])}")
            print(f"💡 相关概念: {extraction_result.get('concepts', [])}")
            print(f"🎲 分析置信度: {extraction_result.get('confidence', 0):.2f}")

            # 保存数据
            output_data = {
                'collection_time': datetime.now().isoformat(),
                'stock_data': stock_data,
                'market_analysis': analysis,
                'concept_analysis': extraction_result
            }

            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2, default=str)

            print(f"\n💾 数据已保存到: {args.output}")
            print("🎉 真实数据获取完成！")

        else:
            print("❌ 未获取到任何股票数据")
            print("可能的原因:")
            print("1. 网络连接问题")
            print("2. API服务不可用")
            print("3. 股票代码错误")
            print("4. 交易时间外（股市休市）")

    except Exception as e:
        logger.error(f"数据获取过程中发生错误: {e}")
        print(f"❌ 数据获取失败: {e}")


if __name__ == "__main__":
    main_12()