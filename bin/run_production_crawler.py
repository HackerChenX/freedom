#!/usr/bin/env python3
"""
正式爬虫系统启动脚本

用于启动生产级别的股市信息爬虫
"""

import sys
import os
import argparse
import time
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crawler.config import CrawlerConfig
from crawler.anti_crawler import AntiCrawlerModule
from crawler.spiders.taoguba_spider import TaogubaSpider
from crawler.processors.concept_extractor import ConceptStockExtractor
from crawler.monitoring.performance_monitor import PerformanceMonitor
from crawler.monitoring.alert_manager import AlertManager
from crawler.integration.system_integrator import SystemIntegrator
from utils.logger import get_logger

logger = get_logger(__name__)


class ProductionCrawler:
    """生产级爬虫系统"""

    def __init__(self):
        self.anti_crawler = AntiCrawlerModule()
        self.concept_extractor = ConceptStockExtractor()
        self.performance_monitor = PerformanceMonitor()
        self.alert_manager = AlertManager()
        self.system_integrator = SystemIntegrator()

        # 爬虫实例
        self.spiders = {
            'taoguba': TaogubaSpider(self.anti_crawler)
        }

        # 统计信息
        self.stats = {
            'total_articles': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'start_time': datetime.now()
        }

    def initialize(self):
        """初始化系统"""
        logger.info("初始化生产爬虫系统...")

        # 启动性能监控
        self.performance_monitor.start()

        # 初始化系统集成
        if not self.system_integrator.initialize():
            logger.warning("系统集成初始化失败，将使用本地模式")

        logger.info("生产爬虫系统初始化完成")

    def crawl_real_data(self, source='taoguba', max_pages=3, max_articles_per_page=5):
        """爬取真实数据"""
        logger.info(f"开始爬取 {source} 真实数据，最大页数: {max_pages}")

        if source not in self.spiders:
            logger.error(f"不支持的数据源: {source}")
            return []

        spider = self.spiders[source]
        articles = []

        for page in range(1, max_pages + 1):
            try:
                logger.info(f"正在爬取第 {page} 页...")

                # 获取文章URL列表
                start_time = time.time()
                article_urls = spider.get_article_urls(page)
                response_time = time.time() - start_time

                # 记录性能指标
                success = len(article_urls) > 0
                self.performance_monitor.record_request(success, response_time)

                if not article_urls:
                    logger.warning(f"第 {page} 页未获取到文章链接")
                    continue

                logger.info(f"第 {page} 页获取到 {len(article_urls)} 个链接")

                # 爬取文章详情
                for i, url in enumerate(article_urls[:max_articles_per_page]):
                    try:
                        logger.info(f"爬取文章 {i+1}/{min(len(article_urls), max_articles_per_page)}: {url}")

                        start_time = time.time()
                        response = spider.get_page(url)

                        if response:
                            article = spider.parse_article_detail(response)
                            response_time = time.time() - start_time

                            if article and article.get('title') and article.get('content'):
                                # 提取概念股信息
                                extraction_result = self.concept_extractor.extract_stocks(
                                    article.get('content', '')
                                )

                                # 合并提取结果
                                article.update(extraction_result)
                                article['crawl_time'] = datetime.now()

                                articles.append(article)
                                self.stats['total_articles'] += 1
                                self.stats['successful_extractions'] += 1

                                # 同步到系统
                                self.system_integrator.sync_article_data(article)

                                # 输出结果
                                print(f"\n=== 文章 {len(articles)} ===")
                                print(f"标题: {article.get('title', 'Unknown')}")
                                print(f"来源: {article.get('source', 'Unknown')}")
                                print(f"URL: {article.get('url', 'Unknown')}")
                                print(f"股票代码: {extraction_result.get('stock_codes', [])}")
                                print(f"概念关键词: {extraction_result.get('concepts', [])}")
                                print(f"置信度: {extraction_result.get('confidence', 0):.2f}")
                                print(f"内容预览: {article.get('content', '')[:200]}...")

                                # 记录成功的性能指标
                                self.performance_monitor.record_request(True, response_time)
                            else:
                                logger.warning("文章解析失败或内容为空")
                                self.performance_monitor.record_request(False, response_time, 'parse_error')
                                self.stats['failed_extractions'] += 1
                        else:
                            logger.warning(f"无法获取文章内容: {url}")
                            self.performance_monitor.record_request(False, 0, 'request_failed')
                            self.stats['failed_extractions'] += 1

                        # 添加延迟避免被封
                        time.sleep(3)

                    except Exception as e:
                        logger.error(f"处理文章失败: {url} - {e}")
                        self.stats['failed_extractions'] += 1

                # 页面间延迟
                time.sleep(5)

            except Exception as e:
                logger.error(f"爬取第 {page} 页失败: {e}")

        return articles

    def get_stats(self):
        """获取统计信息"""
        runtime = datetime.now() - self.stats['start_time']

        return {
            'runtime_minutes': runtime.total_seconds() / 60,
            'total_articles': self.stats['total_articles'],
            'successful_extractions': self.stats['successful_extractions'],
            'failed_extractions': self.stats['failed_extractions'],
            'success_rate': (self.stats['successful_extractions'] / max(self.stats['total_articles'], 1)) * 100,
            'performance_metrics': self.performance_monitor.get_metrics(),
            'system_health': self.system_integrator.health_check()
        }

    def shutdown(self):
        """关闭系统"""
        logger.info("关闭生产爬虫系统...")
        self.performance_monitor.stop()
        logger.info("系统已关闭")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='生产级股市信息爬虫系统')
    parser.add_argument('--source', default='taoguba', choices=['taoguba'],
                       help='数据源选择')
    parser.add_argument('--pages', type=int, default=2,
                       help='爬取页数 (默认: 2)')
    parser.add_argument('--articles-per-page', type=int, default=3,
                       help='每页文章数 (默认: 3)')
    parser.add_argument('--output', default='crawled_data.json',
                       help='输出文件名')
    parser.add_argument('--use-mock-data', action='store_true',
                       help='使用模拟数据（用于演示）')

    args = parser.parse_args()

    print("=== 生产级股市信息爬虫系统 ===")
    print(f"启动时间: {datetime.now()}")
    print(f"数据源: {args.source}")
    print(f"爬取页数: {args.pages}")
    print(f"每页文章数: {args.articles_per_page}")
    print("=" * 50)

    # 创建爬虫实例
    crawler = ProductionCrawler()

    try:
        # 初始化系统
        crawler.initialize()

        # 开始爬取
        print(f"\n开始爬取 {args.source} 数据...")
        articles = crawler.crawl_real_data(
            source=args.source,
            max_pages=args.pages,
            max_articles_per_page=args.articles_per_page
        )

        # 保存数据
        if articles:
            import json
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(articles, f, ensure_ascii=False, indent=2, default=str)
            print(f"\n数据已保存到: {args.output}")

        # 显示统计信息
        stats = crawler.get_stats()
        print(f"\n=== 爬取统计 ===")
        print(f"运行时间: {stats['runtime_minutes']:.1f} 分钟")
        print(f"总文章数: {stats['total_articles']}")
        print(f"成功提取: {stats['successful_extractions']}")
        print(f"失败提取: {stats['failed_extractions']}")
        print(f"成功率: {stats['success_rate']:.1f}%")

        # 显示性能指标
        perf = stats['performance_metrics']
        print(f"\n=== 性能指标 ===")
        print(f"总请求数: {perf['total_requests']}")
        print(f"成功率: {perf['success_rate']:.1f}%")
        print(f"平均响应时间: {perf['avg_response_time']:.2f}秒")

        # 显示系统健康状态
        health = stats['system_health']
        print(f"\n=== 系统状态 ===")
        print(f"整体状态: {health['overall_status']}")
        print(f"组件状态: {health['components']}")

        if articles:
            print(f"\n🎉 成功爬取 {len(articles)} 篇文章！")

            # 显示概念股统计
            all_stock_codes = []
            all_concepts = []
            for article in articles:
                all_stock_codes.extend(article.get('stock_codes', []))
                all_concepts.extend(article.get('concepts', []))

            unique_stocks = list(set(all_stock_codes))
            unique_concepts = list(set(all_concepts))

            print(f"发现股票代码: {unique_stocks}")
            print(f"发现概念关键词: {unique_concepts}")
        else:
            print("⚠️  未获取到任何文章数据")

    except KeyboardInterrupt:
        print("\n用户中断爬取...")
    except Exception as e:
        logger.error(f"爬取过程中发生错误: {e}")
        print(f"❌ 爬取失败: {e}")
    finally:
        # 关闭系统
        crawler.shutdown()
        print("\n系统已安全关闭")


if __name__ == "__main__":
    main()