#!/usr/bin/env python3
"""
高级社区爬虫系统

解决雪球网、淘股吧等社区的反爬虫机制
使用requests + 智能解析，无需Selenium依赖
"""

import sys
import os
import json
import time
import random
import re
from datetime import datetime
from typing import Dict, List, Any
from urllib.parse import urljoin, urlparse

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crawler.processors.concept_extractor import ConceptStockExtractor
from utils.logger import get_logger

logger = get_logger(__name__)

import requests


class AdvancedCommunityCrawler:
    """高级社区爬虫 - 无Selenium版本"""

    def __init__(self):
        self.concept_extractor = ConceptStockExtractor()

        # 创建会话
        self.session = requests.Session()

        # 高级User-Agent池
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ]

        # 社区配置
        self.communities = {
            'xueqiu': {
                'name': '雪球网',
                'base_url': 'https://xueqiu.com',
                'api_urls': [
                    'https://xueqiu.com/statuses/public_timeline_by_category.json?since_id=-1&max_id=-1&count=15&category=-1',
                    'https://xueqiu.com/v4/statuses/public_timeline_by_category.json'
                ],
                'web_urls': [
                    'https://xueqiu.com/today',
                    'https://xueqiu.com'
                ]
            },
            'taoguba': {
                'name': '淘股吧',
                'base_url': 'https://www.taoguba.com.cn',
                'web_urls': [
                    'https://www.taoguba.com.cn/hotTopics',
                    'https://www.taoguba.com.cn/new',
                    'https://www.taoguba.com.cn'
                ]
            },
            'eastmoney_guba': {
                'name': '东方财富股吧',
                'base_url': 'https://guba.eastmoney.com',
                'web_urls': [
                    'https://guba.eastmoney.com/remenhuati',
                    'https://guba.eastmoney.com'
                ]
            }
        }

    def get_advanced_headers(self, referer: str = None) -> Dict[str, str]:
        """获取高级请求头"""
        headers = {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1'
        }

        if referer:
            headers['Referer'] = referer

        return headers

    def safe_request(self, url: str, retries: int = 3) -> Dict[str, Any]:
        """安全请求页面"""
        for attempt in range(retries):
            try:
                # 更新请求头
                headers = self.get_advanced_headers()
                self.session.headers.update(headers)

                print(f"📡 请求 (尝试 {attempt + 1}/{retries}): {url}")

                response = self.session.get(url, timeout=15)

                # 检查响应状态
                if response.status_code == 200:
                    content = response.text

                    # 基本内容检查
                    if len(content) > 500:
                        print(f"✅ 成功获取内容，长度: {len(content)}")
                        return {
                            'success': True,
                            'content': content,
                            'url': url,
                            'status_code': response.status_code,
                            'headers': dict(response.headers)
                        }
                    else:
                        print(f"⚠️  内容过短，可能被阻止")

                else:
                    print(f"❌ HTTP状态码: {response.status_code}")

                # 添加重试延迟
                if attempt < retries - 1:
                    delay = random.uniform(2, 5) * (attempt + 1)
                    print(f"⏳ 等待 {delay:.1f} 秒后重试...")
                    time.sleep(delay)

            except Exception as e:
                print(f"❌ 请求失败 (尝试 {attempt + 1}): {e}")
                if attempt < retries - 1:
                    time.sleep(random.uniform(3, 6))

        return {'success': False, 'error': '所有重试都失败'}

    def extract_discussions_from_html(self, html_content: str, source: str, base_url: str) -> List[Dict[str, Any]]:
        """从HTML中提取讨论内容"""
        discussions = []

        try:
            # 移除脚本和样式
            content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.IGNORECASE | re.DOTALL)
            content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.IGNORECASE | re.DOTALL)

            # 查找包含股市关键词的文本段落
            stock_patterns = [
                r'[^<>]{10,200}[股票投资分析涨跌买卖][^<>]{10,200}',
                r'[^<>]{5,100}[\(（][0-9]{6}[\)）][^<>]{5,100}',
                r'[^<>]{10,150}[市场行情趋势][^<>]{10,150}'
            ]

            found_discussions = []

            for pattern in stock_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    # 清理文本
                    clean_text = re.sub(r'\s+', ' ', match).strip()
                    if len(clean_text) > 20 and len(clean_text) < 500:
                        found_discussions.append(clean_text)

            # 去重并处理
            unique_discussions = list(set(found_discussions))

            for i, discussion_text in enumerate(unique_discussions[:5]):  # 限制数量
                # 提取概念股信息
                extraction_result = self.concept_extractor.extract_stocks(discussion_text)

                # 只保留有股票代码或概念的讨论
                if (extraction_result.get('stock_codes') or
                    extraction_result.get('concepts') or
                    extraction_result.get('confidence', 0) > 0.3):

                    discussion = {
                        'id': f"{source}_discussion_{int(time.time())}_{i}",
                        'title': discussion_text[:50] + "..." if len(discussion_text) > 50 else discussion_text,
                        'content': discussion_text,
                        'url': base_url,
                        'source': self.communities[source]['name'],
                        'content_type': 'user_discussion',
                        'stock_codes': extraction_result.get('stock_codes', []),
                        'concepts': extraction_result.get('concepts', []),
                        'confidence': extraction_result.get('confidence', 0),
                        'crawl_time': datetime.now(),
                        'content_length': len(discussion_text),
                        'disclaimer': "仅供参考，不作为投资建议",
                        'data_source_type': 'community_discussion'
                    }

                    discussions.append(discussion)

        except Exception as e:
            print(f"❌ 内容提取失败: {e}")

        return discussions

    def crawl_xueqiu_community(self) -> List[Dict[str, Any]]:
        """爬取雪球网社区讨论"""
        discussions = []

        print("🔍 开始爬取雪球网社区...")

        # 尝试多个URL
        for url in self.communities['xueqiu']['web_urls']:
            try:
                result = self.safe_request(url)

                if result.get('success'):
                    found_discussions = self.extract_discussions_from_html(
                        result['content'], 'xueqiu', url
                    )
                    discussions.extend(found_discussions)

                    if found_discussions:
                        print(f"✅ 雪球网 {url}: 找到 {len(found_discussions)} 条讨论")
                        break  # 找到内容就停止
                    else:
                        print(f"⚠️  雪球网 {url}: 未找到有效讨论")
                else:
                    print(f"❌ 雪球网 {url}: 访问失败")

                # 添加延迟
                time.sleep(random.uniform(3, 6))

            except Exception as e:
                print(f"❌ 雪球网爬取异常: {e}")

        return discussions

    def crawl_taoguba_community(self) -> List[Dict[str, Any]]:
        """爬取淘股吧社区讨论"""
        discussions = []

        print("🔍 开始爬取淘股吧社区...")

        # 尝试多个URL
        for url in self.communities['taoguba']['web_urls']:
            try:
                result = self.safe_request(url)

                if result.get('success'):
                    found_discussions = self.extract_discussions_from_html(
                        result['content'], 'taoguba', url
                    )
                    discussions.extend(found_discussions)

                    if found_discussions:
                        print(f"✅ 淘股吧 {url}: 找到 {len(found_discussions)} 条讨论")
                        break  # 找到内容就停止
                    else:
                        print(f"⚠️  淘股吧 {url}: 未找到有效讨论")
                else:
                    print(f"❌ 淘股吧 {url}: 访问失败")

                # 添加延迟
                time.sleep(random.uniform(3, 6))

            except Exception as e:
                print(f"❌ 淘股吧爬取异常: {e}")

        return discussions

    def crawl_eastmoney_guba(self) -> List[Dict[str, Any]]:
        """爬取东方财富股吧（替代韭研公社）"""
        discussions = []

        print("🔍 开始爬取东方财富股吧...")

        # 尝试多个URL
        for url in self.communities['eastmoney_guba']['web_urls']:
            try:
                result = self.safe_request(url)

                if result.get('success'):
                    found_discussions = self.extract_discussions_from_html(
                        result['content'], 'eastmoney_guba', url
                    )
                    discussions.extend(found_discussions)

                    if found_discussions:
                        print(f"✅ 东方财富股吧 {url}: 找到 {len(found_discussions)} 条讨论")
                        break  # 找到内容就停止
                    else:
                        print(f"⚠️  东方财富股吧 {url}: 未找到有效讨论")
                else:
                    print(f"❌ 东方财富股吧 {url}: 访问失败")

                # 添加延迟
                time.sleep(random.uniform(3, 6))

            except Exception as e:
                print(f"❌ 东方财富股吧爬取异常: {e}")

        return discussions

    def crawl_all_communities(self) -> List[Dict[str, Any]]:
        """爬取所有社区讨论"""
        all_discussions = []

        print("🚀 开始爬取所有股市讨论社区...")
        print("=" * 60)

        # 爬取雪球网
        xueqiu_discussions = self.crawl_xueqiu_community()
        all_discussions.extend(xueqiu_discussions)

        # 爬取淘股吧
        taoguba_discussions = self.crawl_taoguba_community()
        all_discussions.extend(taoguba_discussions)

        # 爬取东方财富股吧（替代韭研公社）
        eastmoney_discussions = self.crawl_eastmoney_guba()
        all_discussions.extend(eastmoney_discussions)

        print(f"\n📊 爬取总结:")
        print(f"雪球网: {len(xueqiu_discussions)} 条")
        print(f"淘股吧: {len(taoguba_discussions)} 条")
        print(f"东方财富股吧: {len(eastmoney_discussions)} 条")
        print(f"总计: {len(all_discussions)} 条讨论")

        return all_discussions

    def analyze_discussions(self, discussions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析讨论数据"""
        if not discussions:
            return {}

        analysis = {
            'total_discussions': len(discussions),
            'sources': {},
            'all_stock_codes': [],
            'all_concepts': [],
            'high_confidence_discussions': 0,
            'analysis_time': datetime.now()
        }

        # 统计数据源
        for discussion in discussions:
            source = discussion.get('source', 'unknown')
            if source not in analysis['sources']:
                analysis['sources'][source] = 0
            analysis['sources'][source] += 1

            # 收集股票代码和概念
            analysis['all_stock_codes'].extend(discussion.get('stock_codes', []))
            analysis['all_concepts'].extend(discussion.get('concepts', []))

            # 统计高置信度讨论
            if discussion.get('confidence', 0) > 0.5:
                analysis['high_confidence_discussions'] += 1

        # 去重并统计
        analysis['unique_stock_codes'] = list(set(analysis['all_stock_codes']))
        analysis['unique_concepts'] = list(set(analysis['all_concepts']))
        analysis['stock_code_count'] = len(analysis['unique_stock_codes'])
        analysis['concept_count'] = len(analysis['unique_concepts'])

        return analysis


def main():
    """主函数"""
    print("=== 高级股市社区爬虫系统 ===")
    print(f"启动时间: {datetime.now()}")
    print("目标社区: 雪球网、淘股吧、东方财富股吧")
    print("技术方案: 智能请求头 + 反爬虫对抗 + 内容智能提取")
    print("=" * 60)

    # 创建高级爬虫实例
    crawler = AdvancedCommunityCrawler()

    try:
        # 爬取所有社区讨论
        discussions = crawler.crawl_all_communities()

        if discussions:
            print(f"\n🎉 成功爬取 {len(discussions)} 条社区讨论")
            print("=" * 60)

            # 显示爬取结果
            for i, discussion in enumerate(discussions, 1):
                print(f"\n💬 讨论 {i}: {discussion['title']}")
                print(f"   来源: {discussion['source']}")
                print(f"   股票代码: {discussion['stock_codes']}")
                print(f"   相关概念: {discussion['concepts']}")
                print(f"   置信度: {discussion['confidence']:.2f}")
                print(f"   免责声明: {discussion['disclaimer']}")

                # 显示内容预览
                content_preview = discussion['content'][:100].replace('\n', ' ').strip()
                print(f"   内容预览: {content_preview}...")

            # 数据分析
            analysis = crawler.analyze_discussions(discussions)

            print(f"\n📊 讨论数据分析:")
            print(f"总讨论数: {analysis['total_discussions']}")
            print(f"数据源分布: {analysis['sources']}")
            print(f"发现股票代码: {analysis['stock_code_count']} 个")
            print(f"相关概念: {analysis['concept_count']} 个")
            print(f"高置信度讨论: {analysis['high_confidence_discussions']} 条")

            if analysis['unique_stock_codes']:
                print(f"\n🎯 发现的股票代码:")
                for code in analysis['unique_stock_codes'][:10]:
                    print(f"  • {code}")

            if analysis['unique_concepts']:
                print(f"\n💡 相关概念:")
                for concept in analysis['unique_concepts'][:10]:
                    print(f"  • {concept}")

            # 保存数据
            output_data = {
                'crawl_time': datetime.now().isoformat(),
                'total_discussions': len(discussions),
                'discussions': discussions,
                'analysis': analysis,
                'disclaimer_notice': "所有讨论内容均包含免责声明：仅供参考，不作为投资建议"
            }

            output_file = f"advanced_community_discussions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2, default=str)

            print(f"\n💾 数据已保存到: {output_file}")
            print("🎉 高级社区爬虫任务完成！")

            # 免责声明提醒
            print("\n⚠️  重要提醒:")
            print("所有爬取的讨论内容均包含免责声明字段")
            print("内容仅供参考，不作为投资建议")
            print("投资有风险，决策需谨慎")

        else:
            print("❌ 未爬取到任何社区讨论")
            print("\n可能的原因:")
            print("1. 网络连接问题")
            print("2. 目标网站反爬虫机制升级")
            print("3. 需要更高级的反爬虫技术")
            print("4. 建议稍后重试或调整策略")

    except Exception as e:
        logger.error(f"高级爬虫运行过程中发生错误: {e}")
        print(f"❌ 高级爬虫运行失败: {e}")


if __name__ == "__main__":
    main()