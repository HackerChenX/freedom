"""
股市信息爬虫系统

主要功能:
- 多源数据采集 (淘股吧、雪球、韭研公社、慧博投研、萝卜投研)
- 智能反爬虫机制
- 实时数据处理与存储
- 信息提取与分析
"""

__version__ = "1.0.0"
__author__ = "Stock Analysis Team"

try:
    from crawler.scheduler import CrawlerScheduler
except ImportError:
    CrawlerScheduler = None

try:
    from crawler.anti_crawler import AntiCrawlerModule
except ImportError:
    AntiCrawlerModule = None

__all__ = [
    'CrawlerScheduler',
    'AntiCrawlerModule'
]