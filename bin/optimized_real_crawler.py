#!/usr/bin/env python3
"""
优化版真实爬虫系统

基于成功案例优化，提高真实内容提取效果
专门针对东方财富股吧等可访问的真实数据源
"""

import sys
import os
import json
import time
import random
import re
from datetime import datetime
from typing import Dict, List, Any

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crawler.processors.concept_extractor import Concept_stock_extractor
from utils.logger import get_logger
from db.sql_manager import SQLManager, QueryType

logger = get_logger(__name__)

import requests


class OptimizedRealCrawler:
    """优化版真实爬虫"""

    def __init__(self):
        self.concept_extractor = Concept_stock_extractor()
        self.session = requests.Session()

        # 高级User-Agent池
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ]

        # 优化的数据源配置（基于成功案例）
        self.proven_sources = {
            'eastmoney_guba': {
                'name': '东方财富股吧',
                'base_url': 'https://guba.eastmoney.com',
                'stock_urls': [
                    'https://guba.eastmoney.com/list,000001.html',  # 平安银行 - 已验证可访问
                    'https://guba.eastmoney.com/list,002594.html',  # 比亚迪
                    'https://guba.eastmoney.com/list,300750.html',  # 宁德时代
                    'https://guba.eastmoney.com/list,601058.html',  # 赛轮轮胎
                    'https://guba.eastmoney.com/list,600519.html',  # 贵州茅台
                    'https://guba.eastmoney.com/list,600036.html',  # 招商银行
                ],
                'hot_urls': [
                    'https://guba.eastmoney.com/remenhuati.html'
                ]
            }
        }

    def get_optimized_headers(self) -> Dict[str, str]:
        """获取优化的请求头"""
        return {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'no-cache',
            'DNT': '1'
        }

    def safe_request(self, url: str) -> Dict[str, Any]:
        """安全请求"""
        try:
            headers = self.get_optimized_headers()
            self.session.headers.update(headers)

            print(f"📡 请求: {url}")

            response = self.session.get(url, timeout=15)

            if response.status_code == 200:
                content = response.text
                print(f"✅ 成功获取内容，长度: {len(content)}")
                return {
                    'success': True,
                    'content': content,
                    'url': response.url
                }
            else:
                print(f"❌ HTTP错误: {response.status_code}")

        except Exception as e:
            print(f"❌ 请求失败: {e}")

        return {'success': False}

    def extract_enhanced_discussions(self, html_content: str, source_name: str, base_url: str) -> List[Dict[str, Any]]:
        """增强版讨论内容提取"""
        discussions = []

        try:
            print(f"🔍 从 {source_name} 提取真实讨论内容...")

            # 移除脚本和样式
            content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.IGNORECASE | re.DOTALL)
            content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.IGNORECASE | re.DOTALL)

            # 增强的提取模式
            enhanced_patterns = [
                # Meta描述中的股票信息
                r'content="[^"]*([^"]*股吧[^"]*)"',
                r'content="[^"]*([^"]*\([0-9]{6}\)[^"]*)"',
                # 页面标题中的股票信息
                r'<title[^>]*>([^<]*股[^<]*)</title>',
                # 股票代码和名称组合
                r'([^<>]{10,100}[\(（][0-9]{6}[\)）][^<>]{10,100})',
                # 股市相关讨论
                r'([^<>]{20,200}[股票|投资|分析|涨停|跌停|买入|卖出|看好|看空][^<>]{20,200})',
                # 技术分析内容
                r'([^<>]{15,150}[技术面|基本面|K线|均线|成交量|市盈率][^<>]{15,150})',
                # 市场观点
                r'([^<>]{15,150}[建议|推荐|关注|持有|减仓|加仓][^<>]{15,150})',
                # 行业板块讨论
                r'([^<>]{15,150}[板块|概念|龙头|领涨|领跌][^<>]{15,150})'
            ]

            found_texts = set()

            for pattern in enhanced_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0] if match else ''

                    # 清理文本
                    clean_text = re.sub(r'\s+', ' ', match).strip()
                    clean_text = re.sub(r'[<>]', '', clean_text)

                    # 过滤条件
                    if (len(clean_text) > 15 and len(clean_text) < 300 and
                        any(keyword in clean_text for keyword in ['股', '市', '投资', '分析', '涨', '跌', '银行', '汽车', '科技']) and
                        not any(spam in clean_text for spam in ['广告', '推广', '客服', '联系', 'script', 'style'])):
                        found_texts.add(clean_text)

            # 处理找到的文本
            for i, discussion_text in enumerate(list(found_texts)[:8]):  # 增加数量限制
                try:
                    # 提取概念股信息
                    extraction_result = self.concept_extractor.extract_stocks(discussion_text)

                    # 降低过滤门槛，保留更多内容
                    if (extraction_result.get('stock_codes') or
                        extraction_result.get('concepts') or
                        extraction_result.get('confidence', 0) > 0.1 or
                        any(keyword in discussion_text for keyword in ['股票', '投资', '分析', '银行', '汽车'])):

                        discussion = {
                            'id': f"optimized_{source_name.lower().replace(' ', '_')}_{int(time.time())}_{i}",
                            'title': discussion_text[:60] + "..." if len(discussion_text) > 60 else discussion_text,
                            'content': discussion_text,
                            'url': base_url,
                            'source': source_name,
                            'content_type': 'user_discussion',
                            'stock_codes': extraction_result.get('stock_codes', []),
                            'concepts': extraction_result.get('concepts', []),
                            'confidence': extraction_result.get('confidence', 0),
                            'crawl_time': datetime.now(),
                            'content_length': len(discussion_text),
                            'disclaimer': "仅供参考，不作为投资建议",
                            'data_source_type': 'community_discussion',
                            'extraction_method': 'optimized_html_parsing'
                        }

                        discussions.append(discussion)
                        print(f"✅ 提取讨论 {i+1}: {discussion_text[:50]}...")
                        print(f"   股票代码: {discussion['stock_codes']}")
                        print(f"   概念: {discussion['concepts']}")
                        print(f"   置信度: {discussion['confidence']:.2f}")

                except Exception as e:
                    print(f"❌ 处理讨论文本失败: {e}")

            print(f"📊 从 {source_name} 提取到 {len(discussions)} 条讨论")

        except Exception as e:
            print(f"❌ 内容提取失败: {e}")

        return discussions

    def crawl_optimized_sources(self) -> List[Dict[str, Any]]:
        """爬取优化的数据源"""
        all_discussions = []

        print("🚀 开始优化版真实爬取...")
        print("=" * 60)

        for source_key, source_config in self.proven_sources.items():
            print(f"\n🔍 爬取 {source_config['name']}...")

            # 爬取股票页面
            for url in source_config['stock_urls']:
                try:
                    result = self.safe_request(url)

                    if result.get('success'):
                        discussions = self.extract_enhanced_discussions(
                            result['content'],
                            source_config['name'],
                            url
                        )
                        all_discussions.extend(discussions)

                        print(f"✅ {url}: 提取到 {len(discussions)} 条讨论")
                    else:
                        print(f"❌ {url}: 访问失败")

                    # 添加延迟
                    time.sleep(random.uniform(2, 4))

                except Exception as e:
                    print(f"❌ 爬取 {url} 异常: {e}")

            # 爬取热门话题页面
            for url in source_config['hot_urls']:
                try:
                    result = self.safe_request(url)

                    if result.get('success'):
                        discussions = self.extract_enhanced_discussions(
                            result['content'],
                            source_config['name'],
                            url
                        )
                        all_discussions.extend(discussions)

                        print(f"✅ {url}: 提取到 {len(discussions)} 条讨论")
                    else:
                        print(f"❌ {url}: 访问失败")

                    time.sleep(random.uniform(3, 6))

                except Exception as e:
                    print(f"❌ 爬取 {url} 异常: {e}")

        print(f"\n🎉 优化版爬取完成，总计: {len(all_discussions)} 条真实讨论")
        return all_discussions


def main_optimizedrealcrawler():
    """主函数"""
    print("=== 优化版真实股市社区爬虫系统 ===")
    print(f"启动时间: {datetime.now()}")
    print("基于成功案例优化，专注可访问的真实数据源")
    print("=" * 60)

    # 创建优化版爬虫实例
    crawler = Optimized_real_crawler()

    try:
        # 爬取优化的数据源
        discussions = crawler.crawl_optimized_sources()

        if discussions:
            print(f"\n🎉 成功爬取 {len(discussions)} 条真实股市讨论")
            print("=" * 60)

            # 显示爬取结果
            for i, discussion in enumerate(discussions, 1):
                print(f"\n💬 真实讨论 {i}: {discussion['title']}")
                print(f"   来源: {discussion['source']}")
                print(f"   股票代码: {discussion['stock_codes']}")
                print(f"   相关概念: {discussion['concepts']}")
                print(f"   置信度: {discussion['confidence']:.2f}")
                print(f"   免责声明: {discussion['disclaimer']}")
                print(f"   完整内容: {discussion['content']}")

            # 数据分析
            sources = {}
            all_stock_codes = []
            all_concepts = []

            for discussion in discussions:
                source = discussion.get('source', 'unknown')
                sources[source] = sources.get(source, 0) + 1
                all_stock_codes.extend(discussion.get('stock_codes', []))
                all_concepts.extend(discussion.get('concepts', []))

            unique_stock_codes = list(set(all_stock_codes))
            unique_concepts = list(set(all_concepts))

            print(f"\n📊 优化版数据分析:")
            print(f"总讨论数: {len(discussions)}")
            print(f"数据源分布: {sources}")
            print(f"发现股票代码: {len(unique_stock_codes)} 个")
            print(f"相关概念: {len(unique_concepts)} 个")

            if unique_stock_codes:
                print(f"\n🎯 发现的股票代码:")
                for code in unique_stock_codes:
                    print(f"  • {code}")

            if unique_concepts:
                print(f"\n💡 相关概念:")
                for concept in unique_concepts:
                    print(f"  • {concept}")

            # 保存数据
            output_data = {
                'crawl_time': datetime.now().isoformat(),
                'total_discussions': len(discussions),
                'discussions': discussions,
                'analysis': {
                    'sources': sources,
                    'unique_stock_codes': unique_stock_codes,
                    'unique_concepts': unique_concepts
                },
                'disclaimer_notice': "所有讨论内容均包含免责声明：仅供参考，不作为投资建议",
                'technical_note': "优化版真实爬虫，基于成功案例改进"
            }

            output_file = f"optimized_real_discussions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2, default=str)

            print(f"\n💾 优化版真实数据已保存到: {output_file}")
            print("🎉 优化版真实爬虫任务完成！")

        else:
            print("❌ 未爬取到任何真实讨论内容")

    except Exception as e:
        logger.error(f"优化版爬虫运行过程中发生错误: {e}")
        print(f"❌ 优化版爬虫运行失败: {e}")


if __name__ == "__main__":
    main_optimizedrealcrawler()