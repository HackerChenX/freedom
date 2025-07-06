"""
基础爬虫类

定义爬虫的通用接口和基础功能
"""

import time
import requests
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime
from bs4 import Beautiful_soup
from utils.logger import get_logger

logger = get_logger(__name__)


class Base_spider(ABC):
    """基础爬虫类"""

    def __init__(self, name: str, base_url: str, anti_crawler_module=None):
        self.name = name
        self.base_url = base_url
        self.anti_crawler = anti_crawler_module
        self.session = None
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }

    def init_session(self):
        """初始化会话"""
        if self.anti_crawler:
            self.session = self.anti_crawler.get_session()
        else:
            self.session = requests.Session()
            self.session.headers.update(self.headers)

    def get_page(self, url: str, params: dict = None, retries: int = 3) -> Optional[requests.Response]:
        """获取页面内容"""
        if not self.session:
            self.init_session()

        for attempt in range(retries):
            try:
                # 检查访问频率
                if self.anti_crawler:
                    domain = url.split('/')[2]
                    if not self.anti_crawler.check_rate_limit(domain):
                        time.sleep(60)  # 等待1分钟
                        continue

                response = self.session.get(url, params=params, timeout=30)

                # 检查是否被反爬虫拦截
                if self.anti_crawler and self.anti_crawler.handle_block(response, self.session):
                    continue

                if response.status_code == 200:
                    logger.info(f"成功获取页面: {url}")
                    return response
                else:
                    logger.warning(f"页面返回状态码 {response.status_code}: {url}")

            except Exception as e:
                logger.error(f"获取页面失败 (尝试 {attempt + 1}/{retries}): {url} - {e}")

            # 添加延迟
            if self.anti_crawler:
                self.anti_crawler.add_delay()
            else:
                time.sleep(2)

        return None

    def parse_html(self, html: str) -> Beautiful_soup:
        """解析HTML"""
        return BeautifulSoup(html, 'html.parser')

    @abstractmethod
    def parse_article_list_Spider_Base_Spider(self, response: requests.Response) -> List[Dict[str, Any]]:
        """解析文章列表"""
        pass

    @abstractmethod
    def parse_article_detail_Spider_Base_Spider(self, response: requests.Response) -> Dict[str, Any]:
        """解析文章详情"""
        pass

    @abstractmethod
    def get_article_urls_Spider_Base_Spider(self, page: int = 1) -> List[str]:
        """获取文章URL列表"""
        pass