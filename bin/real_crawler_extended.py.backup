#!/usr/bin/env python3
"""
扩展版真实爬虫系统

增加对股市讨论社区的爬取功能：雪球网、淘股吧、韭研公社
获取第一手用户讨论内容，包含免责声明
"""

import sys
import os
import json
import requests
import time
import re
import random
from datetime import datetime
from typing import Dict, List, Any
from urllib.parse import urljoin, urlparse

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crawler.processors.concept_extractor import Concept_stock_extractor
from utils.logger import get_logger

logger = get_logger(__name__)


class ExtendedStockCrawler:
    """扩展版股市爬虫 - 包含讨论社区"""

    def __init___3(self):
        self.concept_extractor = Concept_stock_extractor()

        # 设置请求会话
        self.session = requests.Session()

        # User-Agent池
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ]

        # 目标社区配置
        self.community_sites = {
            'xueqiu': {
                'name': '雪球网',
                'base_url': 'https://xueqiu.com',
                'hot_url': 'https://xueqiu.com/today',
                'enabled': True,
                'delay': (2, 4)  # 随机延迟2-4秒
            },
            'taoguba': {
                'name': '淘股吧',
                'base_url': 'https://www.taoguba.com.cn',
                'hot_url': 'https://www.taoguba.com.cn/hotTopics',
                'enabled': True,
                'delay': (3, 5)  # 随机延迟3-5秒
            },
            'jiyanshe': {
                'name': '韭研公社',
                'base_url': 'https://www.jiyanshe.com',
                'hot_url': 'https://www.jiyanshe.com/hot',
                'enabled': True,
                'delay': (2, 3)  # 随机延迟2-3秒
            }
        }

        # 财经新闻网站配置
        self.news_sites = {
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
            }
        }

    def get_random_headers(self) -> Dict[str, str]:
        """获取随机请求头"""
        return {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }

    def safe_request_Extended(self, url: str, site_config: Dict[str, Any]) -> Dict[str, Any]:
        """安全请求页面内容"""
        try:
            # 更新请求头
            self.session.headers.update(self.get_random_headers())

            print(f"📄 正在访问: {url}")

            response = self.session.get(url, timeout=15)
            response.raise_for_status()

            # 尝试检测编码
            if response.encoding == 'ISO-8859-1':
                response.encoding = response.apparent_encoding or 'utf-8'

            html_content = response.text
            parsed_data = self.parse_html_simple(html_content)

            parsed_data['url'] = url
            parsed_data['status_code'] = response.status_code
            parsed_data['crawl_time'] = datetime.now()

            print(f"✅ 成功获取: {parsed_data['title'][:50]}...")

            # 添加随机延迟
            delay_range = site_config.get('delay', (2, 4))
            delay_time = random.uniform(delay_range[0], delay_range[1])
            time.sleep(delay_time)

            return parsed_data

        except Exception as e:
            print(f"❌ 访问失败: {url} - {e}")
            return {
                'url': url,
                'title': '',
                'content': '',
                'links': [],
                'error': str(e),
                'crawl_time': datetime.now()
            }

    def parse_html_simple(self, html_content: str) -> Dict[str, Any]:
        """简单HTML解析"""
        result = {
            'title': '',
            'content': '',
            'links': []
        }

        # 提取标题
        title_match = re.search(r'<title[^>]*>(.*?)</title>', html_content, re.IGNORECASE | re.DOTALL)
        if title_match:
            result['title'] = re.sub(r'<[^>]+>', '', title_match.group(1)).strip()

        # 提取链接 - 重点关注讨论帖子链接
        link_patterns = [
            r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
            r'href=["\']([^"\']+)["\'][^>]*[^>]*>(.*?)<'
        ]

        for pattern in link_patterns:
            links = re.findall(pattern, html_content, re.IGNORECASE | re.DOTALL)
            for href, text in links:
                clean_text = re.sub(r'<[^>]+>', '', text).strip()
                # 筛选包含股市相关内容的链接
                if (clean_text and len(clean_text) > 5 and
                    any(keyword in clean_text for keyword in ['股', '市', '涨', '跌', '买', '卖', '投资', '分析', '机会', '风险'])):
                    result['links'].append({
                        'url': href,
                        'text': clean_text
                    })

        # 提取正文内容
        content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.IGNORECASE | re.DOTALL)
        content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.IGNORECASE | re.DOTALL)
        content = re.sub(r'<[^>]+>', ' ', content)
        content = re.sub(r'\s+', ' ', content).strip()

        result['content'] = content
        return result

    def crawl_community_discussions(self) -> List[Dict[str, Any]]:
        """爬取社区讨论内容"""
        all_discussions = []

        print("🗣️ 开始爬取股市讨论社区...")

        for site_key, site_config in self.community_sites.items():
            if not site_config['enabled']:
                continue

            print(f"\n💬 爬取 {site_config['name']}...")

            try:
                # 获取热门话题页面
                hot_page = self.safe_request_Extended(site_config['hot_url'], site_config)

                if hot_page.get('error'):
                    print(f"❌ {site_config['name']} 访问失败")
                    continue

                # 从链接中筛选讨论帖子
                discussion_links = []
                for link in hot_page.get('links', []):
                    link_text = link['text']
                    link_url = link['url']

                    # 处理相对URL
                    if link_url.startswith('/'):
                        link_url = urljoin(site_config['base_url'], link_url)
                    elif not link_url.startswith('http'):
                        continue

                    # 筛选讨论帖子（排除广告、导航等）
                    if (len(link_text) > 10 and len(link_text) < 100 and
                        any(keyword in link_text for keyword in ['股', '市', '涨', '跌', '买入', '卖出', '分析', '看好', '看空'])):
                        discussion_links.append({
                            'url': link_url,
                            'title': link_text,
                            'source': site_config['name']
                        })

                print(f"📄 找到 {len(discussion_links)} 个讨论帖子")

                # 爬取讨论详情（每个社区限制3个帖子）
                for i, discussion_link in enumerate(discussion_links[:3]):
                    try:
                        print(f"💬 正在爬取讨论 {i+1}/3: {discussion_link['title'][:30]}...")

                        discussion_content = self.safe_request_Extended(discussion_link['url'], site_config)

                        if not discussion_content.get('error'):
                            # 提取概念股信息
                            extraction_result = self.concept_extractor.extract_stocks(
                                discussion_content.get('content', '')
                            )

                            # 构建讨论数据结构（包含免责声明）
                            discussion = {
                                'id': f"{site_key}_discussion_{int(time.time())}_{i}",
                                'title': discussion_link['title'],
                                'url': discussion_link['url'],
                                'source': discussion_link['source'],
                                'content_type': 'user_discussion',  # 标识为用户讨论内容
                                'content': discussion_content.get('content', '')[:1500],  # 限制长度
                                'stock_codes': extraction_result.get('stock_codes', []),
                                'concepts': extraction_result.get('concepts', []),
                                'confidence': extraction_result.get('confidence', 0),
                                'crawl_time': datetime.now(),
                                'content_length': len(discussion_content.get('content', '')),
                                'disclaimer': "仅供参考，不作为投资建议",  # 必需的免责声明
                                'data_source_type': 'community_discussion'  # 数据源类型标识
                            }

                            all_discussions.append(discussion)

                            print(f"✅ 成功提取讨论: 股票代码{len(discussion['stock_codes'])}个, 概念{len(discussion['concepts'])}个")

                    except Exception as e:
                        print(f"❌ 爬取讨论失败: {e}")

            except Exception as e:
                print(f"❌ 爬取 {site_config['name']} 失败: {e}")

        return all_discussions

    def crawl_finance_news(self) -> List[Dict[str, Any]]:
        """爬取财经新闻"""
        all_articles = []

        print("📰 开始爬取财经新闻...")

        for site_key, site_config in self.news_sites.items():
            if not site_config['enabled']:
                continue

            print(f"\n📰 爬取 {site_config['name']}...")

            try:
                # 获取新闻列表页
                news_page = self.safe_request_Extended(site_config['news_url'], {'delay': (1, 2)})

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

                print(f"📄 找到 {len(article_links)} 个相关新闻")

                # 爬取新闻详情（每个网站限制3篇）
                for i, article_link in enumerate(article_links[:3]):
                    try:
                        print(f"📖 正在爬取新闻 {i+1}/3: {article_link['title'][:30]}...")

                        article_content = self.safe_request_Extended(article_link['url'], {'delay': (1, 2)})

                        if not article_content.get('error'):
                            # 提取概念股信息
                            extraction_result = self.concept_extractor.extract_stocks(
                                article_content.get('content', '')
                            )

                            # 构建新闻数据结构（包含免责声明）
                            article = {
                                'id': f"{site_key}_news_{int(time.time())}_{i}",
                                'title': article_link['title'],
                                'url': article_link['url'],
                                'source': article_link['source'],
                                'content_type': 'financial_news',  # 标识为财经新闻
                                'content': article_content.get('content', '')[:2000],  # 限制长度
                                'stock_codes': extraction_result.get('stock_codes', []),
                                'concepts': extraction_result.get('concepts', []),
                                'confidence': extraction_result.get('confidence', 0),
                                'crawl_time': datetime.now(),
                                'content_length': len(article_content.get('content', '')),
                                'disclaimer': "仅供参考，不作为投资建议",  # 必需的免责声明
                                'data_source_type': 'financial_news'  # 数据源类型标识
                            }

                            all_articles.append(article)

                            print(f"✅ 成功提取新闻: 股票代码{len(article['stock_codes'])}个, 概念{len(article['concepts'])}个")

                    except Exception as e:
                        print(f"❌ 爬取新闻失败: {e}")

            except Exception as e:
                print(f"❌ 爬取 {site_config['name']} 失败: {e}")

        return all_articles

    def analyze_all_data(self, discussions: List[Dict[str, Any]], news: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析所有爬取的数据"""
        all_content = discussions + news

        if not all_content:
            return {}

        analysis = {
            'total_content': len(all_content),
            'discussions_count': len(discussions),
            'news_count': len(news),
            'sources': {},
            'content_types': {},
            'all_stock_codes': [],
            'all_concepts': [],
            'high_confidence_content': 0,
            'analysis_time': datetime.now()
        }

        # 统计数据源和内容类型
        for content in all_content:
            source = content.get('source', 'unknown')
            content_type = content.get('content_type', 'unknown')

            if source not in analysis['sources']:
                analysis['sources'][source] = 0
            analysis['sources'][source] += 1

            if content_type not in analysis['content_types']:
                analysis['content_types'][content_type] = 0
            analysis['content_types'][content_type] += 1

            # 收集股票代码和概念
            analysis['all_stock_codes'].extend(content.get('stock_codes', []))
            analysis['all_concepts'].extend(content.get('concepts', []))

            # 统计高置信度内容
            if content.get('confidence', 0) > 0.5:
                analysis['high_confidence_content'] += 1

        # 去重并统计
        analysis['unique_stock_codes'] = list(set(analysis['all_stock_codes']))
        analysis['unique_concepts'] = list(set(analysis['all_concepts']))
        analysis['stock_code_count'] = len(analysis['unique_stock_codes'])
        analysis['concept_count'] = len(analysis['unique_concepts'])

        return analysis


def main_4():
    """主函数"""
    print("=== 扩展版股市信息爬虫系统 ===")
    print(f"启动时间: {datetime.now()}")
    print("包含讨论社区：雪球网、淘股吧、韭研公社")
    print("包含财经新闻：东方财富、新浪财经")
    print("所有内容均包含免责声明")
    print("=" * 60)

    # 创建扩展爬虫实例
    crawler = Extended_stock_crawler()

    try:
        # 爬取社区讨论
        print("\n🗣️ 第一阶段：爬取股市讨论社区...")
        discussions = crawler.crawl_community_discussions()

        # 爬取财经新闻
        print("\n📰 第二阶段：爬取财经新闻...")
        news = crawler.crawl_finance_news()

        # 合并所有内容
        all_content = discussions + news

        if all_content:
            print(f"\n🎉 总共成功爬取 {len(all_content)} 条内容")
            print(f"   - 社区讨论: {len(discussions)} 条")
            print(f"   - 财经新闻: {len(news)} 条")
            print("=" * 60)

            # 显示爬取结果
            for i, content in enumerate(all_content, 1):
                content_type_icon = "💬" if content['content_type'] == 'user_discussion' else "📰"

                print(f"\n{content_type_icon} 内容 {i}: {content['title'][:50]}...")
                print(f"   来源: {content['source']}")
                print(f"   类型: {content['content_type']}")
                print(f"   URL: {content['url']}")
                print(f"   股票代码: {content['stock_codes']}")
                print(f"   相关概念: {content['concepts']}")
                print(f"   置信度: {content['confidence']:.2f}")
                print(f"   免责声明: {content['disclaimer']}")

                # 显示内容预览
                content_preview = content['content'][:150].replace('\n', ' ').strip()
                print(f"   内容预览: {content_preview}...")

            # 数据分析
            analysis = crawler.analyze_all_data(discussions, news)

            print(f"\n📊 综合数据分析:")
            print(f"总内容数: {analysis['total_content']}")
            print(f"社区讨论: {analysis['discussions_count']} 条")
            print(f"财经新闻: {analysis['news_count']} 条")
            print(f"数据源分布: {analysis['sources']}")
            print(f"内容类型分布: {analysis['content_types']}")
            print(f"发现股票代码: {analysis['stock_code_count']} 个")
            print(f"相关概念: {analysis['concept_count']} 个")
            print(f"高置信度内容: {analysis['high_confidence_content']} 条")

            if analysis['unique_stock_codes']:
                print(f"\n🎯 发现的股票代码:")
                for code in analysis['unique_stock_codes'][:15]:  # 显示前15个
                    print(f"  • {code}")

            if analysis['unique_concepts']:
                print(f"\n💡 相关概念:")
                for concept in analysis['unique_concepts'][:15]:  # 显示前15个
                    print(f"  • {concept}")

            # 保存数据
            output_data = {
                'crawl_time': datetime.now().isoformat(),
                'total_content_count': len(all_content),
                'discussions_count': len(discussions),
                'news_count': len(news),
                'discussions': discussions,
                'news': news,
                'analysis': analysis,
                'disclaimer_notice': "所有内容均包含免责声明：仅供参考，不作为投资建议"
            }

            output_file = f"extended_stock_content_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2, default=str)

            print(f"\n💾 数据已保存到: {output_file}")
            print("🎉 扩展版爬虫任务完成！")

            # 免责声明提醒
            print("\n⚠️  重要提醒:")
            print("所有爬取的内容均包含免责声明字段")
            print("内容仅供参考，不作为投资建议")
            print("投资有风险，决策需谨慎")

        else:
            print("❌ 未爬取到任何内容")
            print("\n可能的原因:")
            print("1. 网络连接问题")
            print("2. 目标网站反爬虫机制")
            print("3. 网站结构发生变化")
            print("4. 需要调整爬虫策略")

    except Exception as e:
        logger.error(f"扩展爬虫运行过程中发生错误: {e}")
        print(f"❌ 扩展爬虫运行失败: {e}")


if __name__ == "__main__":
    main_4()