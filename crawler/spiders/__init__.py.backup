"""
爬虫模块

包含各个平台的专用爬虫实现
"""

try:
    from crawler.spiders.taoguba_spider import TaogubaSpider
except ImportError:
    TaogubaSpider = None

try:
    from crawler.spiders.xueqiu_spider import XueqiuSpider
except ImportError:
    XueqiuSpider = None

try:
    from crawler.spiders.jiuyan_spider import JiuyanSpider
except ImportError:
    JiuyanSpider = None

try:
    from crawler.spiders.hibor_spider import HiborSpider
except ImportError:
    HiborSpider = None

try:
    from crawler.spiders.robo_spider import RoboSpider
except ImportError:
    RoboSpider = None

__all__ = [
    'TaogubaSpider',
    'XueqiuSpider',
    'JiuyanSpider',
    'HiborSpider',
    'RoboSpider'
]