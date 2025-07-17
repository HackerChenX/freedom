#!/usr/bin/env python3
"""
高级真实爬虫系统

使用高级技术绕过反爬虫机制，获取真实的股市讨论内容
不依赖Selenium，使用requests + 智能解析
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

from crawler.processors.concept_extractor import Concept_stock_extractor
from utils.logger import get_logger

logger = get_logger(__name__)

import requests


class AdvancedRealCrawler:
    """高级真实爬虫 - 无Selenium依赖"""

    def __init___10(self):
        self.concept_extractor = Concept_stock_extractor()

        # 创建高级会话
        self.session = requests.Session()

        # 高级User-Agent池（真实浏览器）
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15'
        ]

        # 真实数据源配置
        self.real_sources = {
            'eastmoney_guba': {
                'name': '东方财富股吧',
                'base_url': 'https://guba.eastmoney.com',
                'web_urls': [
                    'https://guba.eastmoney.com/list,000001.html',
                    'https://guba.eastmoney.com/list,002594.html',
                    'https://guba.eastmoney.com/remenhuati.html'
                ]
            },
            'sina_guba': {
                'name': '新浪股吧',
                'base_url': 'https://guba.sina.com.cn',
                'web_urls': [
                    'https://guba.sina.com.cn/000001',
                    'https://guba.sina.com.cn/002594'
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
            'DNT': '1'
        }

        if referer:
            headers['Referer'] = referer

        return headers

    def safe_request_with_retry(self, url: str, retries: int = 3) -> Dict[str, Any]:
        """带重试的安全请求"""
        for attempt in range(retries):
            try:
                # 更新请求头
                headers = self.get_advanced_headers()
                self.session.headers.update(headers)

                print(f"📡 请求 (尝试 {attempt + 1}/{retries}): {url}")

                response = self.session.get(url, timeout=15, allow_redirects=True)

                print(f"   状态码: {response.status_code}")
                print(f"   内容长度: {len(response.text)}")

                if response.status_code == 200:
                    content = response.text

                    # 检查是否有有效内容
                    if len(content) > 1000:
                        print(f"✅ 成功获取内容")
                        return {
                            'success': True,
                            'content': content,
                            'url': response.url,
                            'status_code': response.status_code
                        }
                    else:
                        print(f"⚠️  内容过短，可能被阻止")

                else:
                    print(f"❌ HTTP错误: {response.status_code}")

                # 添加重试延迟
                if attempt < retries - 1:
                    delay = random.uniform(3, 8) * (attempt + 1)
                    print(f"⏳ 等待 {delay:.1f} 秒后重试...")
                    time.sleep(delay)

            except Exception as e:
                print(f"❌ 请求异常 (尝试 {attempt + 1}): {e}")
                if attempt < retries - 1:
                    time.sleep(random.uniform(2, 5))

        return {'success': False, 'error': '所有重试都失败'}

    def extract_real_discussions_from_html(self, html_content: str, source_name: str, base_url: str) -> List[Dict[str, Any]]:
        """从HTML中提取真实讨论内容"""
        discussions = []

        try:
            print(f"🔍 从 {source_name} 提取真实讨论内容...")

            # 移除脚本和样式
            content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.IGNORECASE | re.DOTALL)
            content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.IGNORECASE | re.DOTALL)

            # 查找真实的讨论内容模式
            discussion_patterns = [
                # 股票代码模式
                r'[^<>]{20,300}[\(（][0-9]{6}[\)）][^<>]{20,300}',
                # 股市关键词模式
                r'[^<>]{30,400}[股票|投资|分析|涨停|跌停|买入|卖出][^<>]{30,400}',
                # 技术分析模式
                r'[^<>]{20,300}[技术面|基本面|K线|均线|成交量][^<>]{20,300}',
                # 市场观点模式
                r'[^<>]{20,300}[看好|看空|建议|推荐|关注][^<>]{20,300}'
            ]

            found_texts = set()  # 使用set去重

            for pattern in discussion_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    # 清理文本
                    clean_text = re.sub(r'\s+', ' ', match).strip()

                    # 过滤条件
                    if (len(clean_text) > 30 and len(clean_text) < 200 and
                        any(keyword in clean_text for keyword in ['股', '市', '投资', '分析', '涨', '跌']) and
                        not any(spam in clean_text for spam in ['广告', '推广', '客服', '联系'])):
                        found_texts.add(clean_text)

            # 处理找到的文本
            for i, discussion_text in enumerate(list(found_texts)[:5]):  # 限制数量
                try:
                    # 提取概念股信息
                    extraction_result = self.concept_extractor.extract_stocks(discussion_text)

                    # 只保留有价值的讨论
                    if (extraction_result.get('stock_codes') or
                        extraction_result.get('concepts') or
                        extraction_result.get('confidence', 0) > 0.2):

                        discussion = {
                            'id': f"real_{source_name.lower()}_{int(time.time())}_{i}",
                            'title': discussion_text[:50] + "..." if len(discussion_text) > 50 else discussion_text,
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
                            'extraction_method': 'real_html_parsing'
                        }

                        discussions.append(discussion)
                        print(f"✅ 提取真实讨论: {discussion_text[:40]}...")
                        print(f"   股票代码: {discussion['stock_codes']}")
                        print(f"   概念: {discussion['concepts']}")

                except Exception as e:
                    print(f"❌ 处理讨论文本失败: {e}")

            print(f"📊 从 {source_name} 提取到 {len(discussions)} 条真实讨论")

        except Exception as e:
            print(f"❌ 内容提取失败: {e}")

        return discussions

    def crawl_all_real_sources(self) -> List[Dict[str, Any]]:
        """爬取所有真实数据源"""
        all_discussions = []

        print("🚀 开始爬取真实股市讨论社区...")
        print("=" * 60)

        for source_key, source_config in self.real_sources.items():
            print(f"\n🔍 爬取 {source_config['name']}...")

            source_discussions = []

            # 尝试每个URL
            for url in source_config['web_urls']:
                try:
                    result = self.safe_request_with_retry(url)

                    if result.get('success'):
                        discussions = self.extract_real_discussions_from_html(
                            result['content'],
                            source_config['name'],
                            url
                        )
                        source_discussions.extend(discussions)

                        if discussions:
                            print(f"✅ {source_config['name']} {url}: 找到 {len(discussions)} 条真实讨论")
                            break  # 找到内容就停止尝试其他URL
                        else:
                            print(f"⚠️  {source_config['name']} {url}: 未找到有效讨论")
                    else:
                        print(f"❌ {source_config['name']} {url}: 访问失败")

                    # 添加延迟
                    time.sleep(random.uniform(5, 10))

                except Exception as e:
                    print(f"❌ {source_config['name']} 爬取异常: {e}")

            all_discussions.extend(source_discussions)
            print(f"📊 {source_config['name']} 总计: {len(source_discussions)} 条讨论")

        print(f"\n🎉 所有真实数据源爬取完成，总计: {len(all_discussions)} 条讨论")
        return all_discussions


def main_17():
    """主函数"""
    print("=== 高级真实股市社区爬虫系统 ===")
    print(f"启动时间: {datetime.now()}")
    print("技术方案: 高级requests + 智能HTML解析 + 反爬虫对抗")
    print("目标: 获取真实的股市讨论内容")
    print("=" * 60)

    # 创建高级真实爬虫实例
    crawler = Advanced_real_crawler()

    try:
        # 爬取所有真实数据源
        discussions = crawler.crawl_all_real_sources()

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
                print(f"   提取方法: {discussion['extraction_method']}")

                # 显示完整内容
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

            print(f"\n📊 真实数据分析:")
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
                'technical_note': "使用高级HTML解析技术从真实网站提取内容"
            }

            output_file = f"real_community_discussions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2, default=str)

            print(f"\n💾 真实数据已保存到: {output_file}")
            print("🎉 高级真实爬虫任务完成！")

        else:
            print("❌ 未爬取到任何真实讨论内容")
            print("\n可能的原因:")
            print("1. 目标网站反爬虫机制过于严格")
            print("2. 网站结构发生变化")
            print("3. 需要更高级的反爬虫技术")
            print("4. 建议使用Selenium或其他浏览器自动化工具")

    except Exception as e:
        logger.error(f"高级真实爬虫运行过程中发生错误: {e}")
        print(f"❌ 高级真实爬虫运行失败: {e}")


if __name__ == "__main__":
    main_17()