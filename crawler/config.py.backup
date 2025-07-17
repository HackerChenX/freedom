"""
爬虫系统配置

包含各种配置参数和设置
"""

import os
from typing import Dict, List


class Crawler_config:
    """爬虫配置类"""

    # Redis配置
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
    REDIS_DB = int(os.getenv('REDIS_DB', 0))

    # ClickHouse配置
    CLICKHOUSE_HOST = os.getenv('CLICKHOUSE_HOST', 'localhost')
    CLICKHOUSE_PORT = int(os.getenv('CLICKHOUSE_PORT', 8123))
    CLICKHOUSE_USER = os.getenv('CLICKHOUSE_USER', 'default')
    CLICKHOUSE_PASSWORD = os.getenv('CLICKHOUSE_PASSWORD', '')
    CLICKHOUSE_DATABASE = os.getenv('CLICKHOUSE_DATABASE', 'stock_crawler')

    # 爬虫设置
    MAX_WORKERS = int(os.getenv('MAX_WORKERS', 5))
    REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', 30))
    RETRY_TIMES = int(os.getenv('RETRY_TIMES', 3))

    # 延迟设置
    MIN_DELAY = float(os.getenv('MIN_DELAY', 1.0))
    MAX_DELAY = float(os.getenv('MAX_DELAY', 3.0))

    # 代理设置
    USE_PROXY = os.getenv('USE_PROXY', 'true').lower() == 'true'
    PROXY_POOL_SIZE = int(os.getenv('PROXY_POOL_SIZE', 100))

    # 数据源配置
    DATA_SOURCES = {
        'taoguba': {
            'name': '淘股吧',
            'base_url': 'https://www.taoguba.com.cn',
            'enabled': True,
            'priority': 1,
            'rate_limit': 10,  # 每分钟请求数
        },
        'xueqiu': {
            'name': '雪球',
            'base_url': 'https://xueqiu.com',
            'enabled': True,
            'priority': 2,
            'rate_limit': 15,
        },
        'jiuyan': {
            'name': '韭研公社',
            'base_url': 'https://www.jiuyan.info',
            'enabled': True,
            'priority': 3,
            'rate_limit': 8,
        },
        'hibor': {
            'name': '慧博投研',
            'base_url': 'https://www.hibor.com.cn',
            'enabled': True,
            'priority': 4,
            'rate_limit': 5,
        },
        'robo': {
            'name': '萝卜投研',
            'base_url': 'https://robo.datayes.com',
            'enabled': False,  # 需要API认证
            'priority': 5,
            'rate_limit': 20,
        }
    }