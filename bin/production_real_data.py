#!/usr/bin/env python3
"""
生产级真实数据获取脚本

使用多个免费API获取真实股市数据，适用于生产环境
"""

import sys
import os
import json
import requests
import time
from datetime import datetime
from typing import Dict, List, Any

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crawler.processors.concept_extractor import Concept_stock_extractor
from utils.logger import get_logger

logger = get_logger(__name__)


class ProductionDataCollector:
    """生产级数据收集器"""

    def __init___5(self):
        self.concept_extractor = Concept_stock_extractor()

        # 设置请求头
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        # 股票代码列表
        self.stock_codes = [
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

    def get_eastmoney_data(self, stock_codes: List[str]) -> List[Dict[str, Any]]:
        """从东方财富获取数据"""
        results = []

        print("📊 尝试东方财富API...")

        try:
            # 构建股票代码字符串
            secids = []
            for code in stock_codes:
                if code.startswith('6'):
                    secids.append(f"1.{code}")  # 上海
                else:
                    secids.append(f"0.{code}")  # 深圳

            secid_str = ','.join(secids)

            url = "https://push2.eastmoney.com/api/qt/ulist.np/get"
            params = {
                'fltt': '2',
                'invt': '2',
                'fields': 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f22,f11,f62,f128,f136,f115,f152',
                'secids': secid_str
            }

            response = requests.get(url, params=params, headers=self.headers, timeout=15)

            if response.status_code == 200:
                data = response.json()

                if 'data' in data and 'diff' in data['data']:
                    for item in data['data']['diff']:
                        try:
                            stock_code = item.get('f12', '')
                            stock_name = item.get('f14', '')

                            if stock_code and stock_name:
                                stock_data = {
                                    'stock_code': stock_code,
                                    'name': stock_name,
                                    'current_price': item.get('f2', 0) / 100 if item.get('f2') else 0,
                                    'change_pct': item.get('f3', 0) / 100 if item.get('f3') else 0,
                                    'change': item.get('f4', 0) / 100 if item.get('f4') else 0,
                                    'volume': item.get('f5', 0),
                                    'amount': item.get('f6', 0),
                                    'open_price': item.get('f17', 0) / 100 if item.get('f17') else 0,
                                    'high_price': item.get('f15', 0) / 100 if item.get('f15') else 0,
                                    'low_price': item.get('f16', 0) / 100 if item.get('f16') else 0,
                                    'yesterday_close': item.get('f18', 0) / 100 if item.get('f18') else 0,
                                    'market_cap': item.get('f20', 0),
                                    'pe_ratio': item.get('f9', 0) / 100 if item.get('f9') else 0,
                                    'source': 'eastmoney',
                                    'crawl_time': datetime.now()
                                }

                                results.append(stock_data)

                                # 显示状态
                                if stock_data['change_pct'] > 0:
                                    status = "📈"
                                elif stock_data['change_pct'] < 0:
                                    status = "📉"
                                else:
                                    status = "➡️"

                                print(f"✅ {status} {stock_name}({stock_code}): {stock_data['current_price']:.2f} ({stock_data['change_pct']:+.2f}%)")

                        except Exception as e:
                            print(f"❌ 解析股票数据失败: {e}")

                print(f"✅ 东方财富API获取到 {len(results)} 只股票数据")
            else:
                print(f"❌ 东方财富API请求失败: {response.status_code}")

        except Exception as e:
            print(f"❌ 东方财富API异常: {e}")

        return results

    def get_netease_data(self, stock_codes: List[str]) -> List[Dict[str, Any]]:
        """从网易财经获取数据"""
        results = []

        print("📊 尝试网易财经API...")

        for code in stock_codes:
            try:
                # 构建API URL
                if code.startswith('6'):
                    symbol = f"0{code}"
                else:
                    symbol = f"1{code}"

                url = f"https://api.money.126.net/data/feed/{symbol}"

                response = requests.get(url, headers=self.headers, timeout=10)

                if response.status_code == 200:
                    content = response.text
                    # 网易返回的是JSONP格式，需要处理
                    if content.startswith('_ntes_quote_callback('):
                        json_str = content[21:-2]  # 去掉前后的函数调用
                        data = json.loads(json_str)

                        if symbol in data:
                            item = data[symbol]

                            stock_data = {
                                'stock_code': code,
                                'name': item.get('name', ''),
                                'current_price': float(item.get('price', 0)),
                                'yesterday_close': float(item.get('yestclose', 0)),
                                'open_price': float(item.get('open', 0)),
                                'high_price': float(item.get('high', 0)),
                                'low_price': float(item.get('low', 0)),
                                'volume': int(item.get('volume', 0)),
                                'amount': float(item.get('turnover', 0)),
                                'source': 'netease',
                                'crawl_time': datetime.now()
                            }

                            # 计算涨跌幅
                            if stock_data['current_price'] and stock_data['yesterday_close']:
                                change = stock_data['current_price'] - stock_data['yesterday_close']
                                change_pct = (change / stock_data['yesterday_close']) * 100
                                stock_data['change'] = change
                                stock_data['change_pct'] = change_pct

                            results.append(stock_data)
                            print(f"✅ 网易财经: {stock_data['name']}({code}) 数据获取成功")
                        else:
                            print(f"❌ 网易财经: {code} 数据不存在")
                    else:
                        print(f"❌ 网易财经: {code} 数据格式异常")
                else:
                    print(f"❌ 网易财经: {code} 请求失败 ({response.status_code})")

                time.sleep(0.3)  # 添加延迟

            except Exception as e:
                print(f"❌ 网易财经: {code} 获取失败 - {e}")

        return results

    def collect_production_data(self) -> List[Dict[str, Any]]:
        """收集生产级数据"""
        print("🚀 开始收集生产级股市数据...")
        print(f"目标股票: {len(self.stock_codes)} 只")
        print("=" * 50)

        # 首先尝试东方财富（通常最稳定）
        eastmoney_data = self.get_eastmoney_data(self.stock_codes)

        if eastmoney_data:
            print(f"\n✅ 东方财富API成功获取 {len(eastmoney_data)} 只股票数据")
            return eastmoney_data

        # 如果东方财富失败，尝试网易财经
        print("\n📊 东方财富API失败，尝试网易财经...")
        netease_data = self.get_netease_data(self.stock_codes)

        if netease_data:
            print(f"\n✅ 网易财经API成功获取 {len(netease_data)} 只股票数据")
            return netease_data

        print("\n❌ 所有数据源都无法获取数据")
        return []

    def analyze_market_data(self, stock_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析市场数据"""
        if not stock_data:
            return {}

        analysis = {
            'total_stocks': len(stock_data),
            'rising_stocks': 0,
            'falling_stocks': 0,
            'flat_stocks': 0,
            'strong_rising': 0,  # 涨幅>3%
            'strong_falling': 0,  # 跌幅>3%
            'top_gainers': [],
            'top_losers': [],
            'market_sentiment': '',
            'total_market_cap': 0,
            'total_volume': 0,
            'analysis_time': datetime.now()
        }

        gains = []
        total_change = 0

        for stock in stock_data:
            change_pct = stock.get('change_pct', 0)
            market_cap = stock.get('market_cap', 0)
            volume = stock.get('volume', 0)

            # 统计涨跌
            if change_pct > 0:
                analysis['rising_stocks'] += 1
                if change_pct > 3:
                    analysis['strong_rising'] += 1
            elif change_pct < 0:
                analysis['falling_stocks'] += 1
                if change_pct < -3:
                    analysis['strong_falling'] += 1
            else:
                analysis['flat_stocks'] += 1

            # 收集数据
            gains.append((stock['stock_code'], stock['name'], change_pct))
            total_change += change_pct
            analysis['total_market_cap'] += market_cap
            analysis['total_volume'] += volume

        # 计算市场情绪
        avg_change = total_change / len(stock_data)
        if avg_change > 1:
            analysis['market_sentiment'] = '乐观'
        elif avg_change < -1:
            analysis['market_sentiment'] = '悲观'
        else:
            analysis['market_sentiment'] = '中性'

        # 排序获取涨跌幅榜
        gains.sort(key=lambda x: x[2], reverse=True)
        analysis['top_gainers'] = gains[:5]
        analysis['top_losers'] = gains[-5:]
        analysis['average_change'] = avg_change

        return analysis

    def extract_market_concepts(self, stock_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """提取市场概念"""
        # 构建包含所有股票的文本
        stock_text = "今日关注股票："
        for stock in stock_data:
            name = stock.get('name', '')
            code = stock.get('stock_code', '')
            change_pct = stock.get('change_pct', 0)

            if name and code:
                if change_pct > 0:
                    stock_text += f"{name}({code})上涨{change_pct:.2f}%、"
                elif change_pct < 0:
                    stock_text += f"{name}({code})下跌{abs(change_pct):.2f}%、"
                else:
                    stock_text += f"{name}({code})平盘、"

        # 使用概念提取器分析
        extraction_result = self.concept_extractor.extract_stocks(stock_text)

        return {
            'detected_stocks': extraction_result.get('stock_codes', []),
            'concept_keywords': extraction_result.get('concepts', []),
            'confidence': extraction_result.get('confidence', 0.0),
            'analysis_text': stock_text
        }


def main_7():
    """主函数"""
    print("=== 生产级股市数据获取系统 ===")
    print(f"启动时间: {datetime.now()}")
    print("适用于生产环境的真实数据获取")
    print("=" * 50)

    # 创建数据收集器
    collector = Production_data_collector()

    try:
        # 收集数据
        stock_data = collector.collect_production_data()

        if stock_data:
            print(f"\n🎉 成功获取 {len(stock_data)} 只股票的实时数据")
            print("=" * 50)

            # 显示详细数据
            print("\n📊 股票详细信息:")
            for i, stock in enumerate(stock_data, 1):
                name = stock.get('name', 'Unknown')
                code = stock.get('stock_code', 'Unknown')
                current = stock.get('current_price', 0)
                change = stock.get('change', 0)
                change_pct = stock.get('change_pct', 0)
                volume = stock.get('volume', 0)

                if change_pct > 0:
                    status = "📈"
                    change_str = f"+{change:.2f} (+{change_pct:.2f}%)"
                elif change_pct < 0:
                    status = "📉"
                    change_str = f"{change:.2f} ({change_pct:.2f}%)"
                else:
                    status = "➡️"
                    change_str = "0.00 (0.00%)"

                print(f"{i:2d}. {status} {name}({code})")
                print(f"    价格: {current:.2f} 变动: {change_str}")
                print(f"    成交量: {volume:,}")

            # 市场分析
            analysis = collector.analyze_market_data(stock_data)

            print(f"\n📈 市场分析:")
            print(f"总股票数: {analysis['total_stocks']}")
            print(f"上涨股票: {analysis['rising_stocks']} (强势上涨: {analysis['strong_rising']})")
            print(f"下跌股票: {analysis['falling_stocks']} (强势下跌: {analysis['strong_falling']})")
            print(f"平盘股票: {analysis['flat_stocks']}")
            print(f"平均涨跌幅: {analysis['average_change']:.2f}%")
            print(f"市场情绪: {analysis['market_sentiment']}")

            if analysis['top_gainers']:
                print(f"\n🏆 涨幅榜:")
                for code, name, pct in analysis['top_gainers']:
                    print(f"  📈 {name}({code}): +{pct:.2f}%")

            if analysis['top_losers']:
                print(f"\n📉 跌幅榜:")
                for code, name, pct in analysis['top_losers']:
                    print(f"  📉 {name}({code}): {pct:.2f}%")

            # 概念分析
            concept_analysis = collector.extract_market_concepts(stock_data)

            print(f"\n💡 概念分析:")
            print(f"识别股票代码: {concept_analysis['detected_stocks']}")
            print(f"相关概念: {concept_analysis['concept_keywords']}")
            print(f"分析置信度: {concept_analysis['confidence']:.2f}")

            # 保存数据
            output_data = {
                'collection_time': datetime.now().isoformat(),
                'data_source': stock_data[0]['source'] if stock_data else 'unknown',
                'stock_count': len(stock_data),
                'stock_data': stock_data,
                'market_analysis': analysis,
                'concept_analysis': concept_analysis
            }

            output_file = f"production_stock_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2, default=str)

            print(f"\n💾 数据已保存到: {output_file}")
            print("🎉 生产级数据获取完成！")

        else:
            print("❌ 未获取到任何股票数据")
            print("\n可能的原因:")
            print("1. 网络连接问题")
            print("2. API服务暂时不可用")
            print("3. 交易时间外（股市休市）")
            print("4. 需要等待几分钟后重试")

    except Exception as e:
        logger.error(f"数据获取过程中发生错误: {e}")
        print(f"❌ 系统错误: {e}")


if __name__ == "__main__":
    main_7()