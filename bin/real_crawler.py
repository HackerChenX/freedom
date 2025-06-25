#!/usr/bin/env python3
"""
真实爬虫系统

直接爬取网页获取股市新闻、分析报告、讨论等真实内容
"""

import sys
import os
import json
import requests
import time
import re
from datetime import datetime
from typing import Dict, List, Any
from urllib.parse import urljoin, urlparse

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crawler.processors.concept_extractor import ConceptStockExtractor
from utils.logger import get_logger

logger = get_logger(__name__)

# 尝试导入BeautifulSoup，如果没有则使用简单的HTML解析
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    print("⚠️  未安装BeautifulSoup4，将使用简单HTML解析")


class RealWebCrawler:
    """真实网页爬虫"""

    def __init__(self):
        self.concept_extractor = ConceptStockExtractor()

        # 设置请求会话
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })

        # 目标网站配置
        self.target_sites = {
            'eastmoney_news': {
                'name': '东方财富新闻',
                'base_url': 'https://finance.eastmoney.com',
                'news_url': 'https://finance.eastmoney.com/news/cgsxw.html',
                'enabled': True
            },
            'sina_finance': {
                'name': '新浪财经',
                'base_url': 'https://finance.sina.com.cn',
                'news_url': 'https://finance.sina.com.cn/stock/',
                'enabled': True
            },
            'hexun': {
                'name': '和讯网',
                'base_url': 'https://www.hexun.com',
                'news_url': 'https://stock.hexun.com/',
                'enabled': True
            }
        }

    def get_page_content(self, url: str) -> Dict[str, Any]:
        """获取页面内容"""
        try:
            print(f"📄 正在获取: {url}")

            response = self.session.get(url, timeout=15)
            response.raise_for_status()

            # 尝试检测编码
            if response.encoding == 'ISO-8859-1':
                response.encoding = response.apparent_encoding or 'utf-8'

            html_content = response.text

            # 简单HTML解析
            parsed_data = self.parse_html_simple(html_content)

            parsed_data['url'] = url
            parsed_data['status_code'] = response.status_code
            parsed_data['crawl_time'] = datetime.now()

            print(f"✅ 成功获取页面: {parsed_data['title'][:50]}...")
            return parsed_data

        except Exception as e:
            print(f"❌ 获取页面失败: {url} - {e}")
            return {
                'url': url,
                'title': '',
                'content': '',
                'links': [],
                'error': str(e),
                'crawl_time': datetime.now()
            }

    def parse_html_simple(self, html_content: str) -> Dict[str, Any]:
        """简单HTML解析（不依赖外部库）"""
        result = {
            'title': '',
            'content': '',
            'links': []
        }

        # 提取标题
        title_match = re.search(r'<title[^>]*>(.*?)</title>', html_content, re.IGNORECASE | re.DOTALL)
        if title_match:
            result['title'] = re.sub(r'<[^>]+>', '', title_match.group(1)).strip()

        # 提取链接
        link_pattern = r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>'
        links = re.findall(link_pattern, html_content, re.IGNORECASE | re.DOTALL)
        for href, text in links:
            clean_text = re.sub(r'<[^>]+>', '', text).strip()
            if clean_text and len(clean_text) > 5 and '股' in clean_text:
                result['links'].append({
                    'url': href,
                    'text': clean_text
                })

        # 提取正文内容
        # 移除脚本和样式
        content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.IGNORECASE | re.DOTALL)
        content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.IGNORECASE | re.DOTALL)

        # 移除HTML标签
        content = re.sub(r'<[^>]+>', ' ', content)

        # 清理空白字符
        content = re.sub(r'\s+', ' ', content).strip()

        result['content'] = content

        return result

    def crawl_finance_news(self, max_pages: int = 2) -> List[Dict[str, Any]]:
        """爬取财经新闻"""
        all_articles = []

        print("🕷️ 开始爬取财经新闻...")

        for site_key, site_config in self.target_sites.items():
            if not site_config['enabled']:
                continue

            print(f"\n📰 爬取 {site_config['name']}...")

            try:
                # 获取新闻列表页
                news_page = self.get_page_content(site_config['news_url'])

                if news_page.get('error'):
                    print(f"❌ {site_config['name']} 访问失败")
                    continue

                # 从链接中筛选新闻文章
                article_links = []
                for link in news_page.get('links', []):
                    link_text = link['text']
                    link_url = link['url']

                    # 筛选包含股市相关关键词的链接
                    if any(keyword in link_text for keyword in ['股', '市', '涨', '跌', '投资', '基金', '证券', '上市', '公司']):
                        # 处理相对URL
                        if link_url.startswith('/'):
                            link_url = urljoin(site_config['base_url'], link_url)
                        elif not link_url.startswith('http'):
                            continue

                        article_links.append({
                            'url': link_url,
                            'title': link_text,
                            'source': site_config['name']
                        })

                print(f"📄 找到 {len(article_links)} 个相关链接")

                # 爬取文章详情（限制数量）
                for i, article_link in enumerate(article_links[:5]):  # 每个网站最多爬5篇
                    try:
                        print(f"📖 正在爬取文章 {i+1}/5: {article_link['title'][:30]}...")

                        article_content = self.get_page_content(article_link['url'])

                        if not article_content.get('error'):
                            # 提取概念股信息
                            extraction_result = self.concept_extractor.extract_stocks(
                                article_content.get('content', '')
                            )

                            article = {
                                'id': f"{site_key}_{int(time.time())}_{i}",
                                'title': article_link['title'],
                                'url': article_link['url'],
                                'source': article_link['source'],
                                'content': article_content.get('content', '')[:2000],  # 限制长度
                                'stock_codes': extraction_result.get('stock_codes', []),
                                'concepts': extraction_result.get('concepts', []),
                                'confidence': extraction_result.get('confidence', 0),
                                'crawl_time': datetime.now(),
                                'content_length': len(article_content.get('content', ''))
                            }

                            all_articles.append(article)

                            print(f"✅ 成功提取: 股票代码{len(article['stock_codes'])}个, 概念{len(article['concepts'])}个")

                        # 添加延迟避免被封
                        time.sleep(2)

                    except Exception as e:
                        print(f"❌ 爬取文章失败: {e}")

                # 网站间延迟
                time.sleep(3)

            except Exception as e:
                print(f"❌ 爬取 {site_config['name']} 失败: {e}")

        return all_articles

    def analyze_crawled_data(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析爬取的数据"""
        if not articles:
            return {}

        analysis = {
            'total_articles': len(articles),
            'sources': {},
            'all_stock_codes': [],
            'all_concepts': [],
            'high_confidence_articles': 0,
            'analysis_time': datetime.now()
        }

        # 统计数据源
        for article in articles:
            source = article.get('source', 'unknown')
            if source not in analysis['sources']:
                analysis['sources'][source] = 0
            analysis['sources'][source] += 1

            # 收集股票代码和概念
            analysis['all_stock_codes'].extend(article.get('stock_codes', []))
            analysis['all_concepts'].extend(article.get('concepts', []))

            # 统计高置信度文章
            if article.get('confidence', 0) > 0.5:
                analysis['high_confidence_articles'] += 1

        # 去重并统计
        analysis['unique_stock_codes'] = list(set(analysis['all_stock_codes']))
        analysis['unique_concepts'] = list(set(analysis['all_concepts']))
        analysis['stock_code_count'] = len(analysis['unique_stock_codes'])
        analysis['concept_count'] = len(analysis['unique_concepts'])

        return analysis


def main():
    """主函数"""
    print("=== 真实股市信息爬虫系统 ===")
    print(f"启动时间: {datetime.now()}")
    print("直接爬取网页获取股市新闻和分析内容")
    print("=" * 50)

    # 创建爬虫实例
    crawler = RealWebCrawler()

    try:
        # 开始爬取
        articles = crawler.crawl_finance_news()

        if articles:
            print(f"\n🎉 成功爬取 {len(articles)} 篇文章")
            print("=" * 50)

            # 显示爬取结果
            for i, article in enumerate(articles, 1):
                print(f"\n📰 文章 {i}: {article['title'][:50]}...")
                print(f"   来源: {article['source']}")
                print(f"   URL: {article['url']}")
                print(f"   股票代码: {article['stock_codes']}")
                print(f"   相关概念: {article['concepts']}")
                print(f"   置信度: {article['confidence']:.2f}")
                print(f"   内容长度: {article['content_length']} 字符")

                # 显示内容预览
                content_preview = article['content'][:200].replace('\n', ' ').strip()
                print(f"   内容预览: {content_preview}...")

            # 数据分析
            analysis = crawler.analyze_crawled_data(articles)

            print(f"\n📊 爬取数据分析:")
            print(f"总文章数: {analysis['total_articles']}")
            print(f"数据源分布: {analysis['sources']}")
            print(f"发现股票代码: {analysis['stock_code_count']} 个")
            print(f"相关概念: {analysis['concept_count']} 个")
            print(f"高置信度文章: {analysis['high_confidence_articles']} 篇")

            if analysis['unique_stock_codes']:
                print(f"\n🎯 发现的股票代码:")
                for code in analysis['unique_stock_codes'][:10]:  # 显示前10个
                    print(f"  • {code}")

            if analysis['unique_concepts']:
                print(f"\n💡 相关概念:")
                for concept in analysis['unique_concepts'][:10]:  # 显示前10个
                    print(f"  • {concept}")

            # 保存数据
            output_data = {
                'crawl_time': datetime.now().isoformat(),
                'article_count': len(articles),
                'articles': articles,
                'analysis': analysis
            }

            output_file = f"crawled_finance_news_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2, default=str)

            print(f"\n💾 数据已保存到: {output_file}")
            print("🎉 真实爬虫任务完成！")

        else:
            print("❌ 未爬取到任何文章")
            print("\n可能的原因:")
            print("1. 网络连接问题")
            print("2. 目标网站反爬虫机制")
            print("3. 网站结构发生变化")
            print("4. 需要调整爬虫策略")

    except Exception as e:
        logger.error(f"爬虫运行过程中发生错误: {e}")
        print(f"❌ 爬虫运行失败: {e}")


if __name__ == "__main__":
    main()