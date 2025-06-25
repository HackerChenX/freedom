"""
反爬虫模块

提供代理池管理、User-Agent轮换、验证码识别等反爬虫功能
"""

import random
import time
import requests
from typing import List, Dict, Optional
from utils.logger import get_logger

# 可选导入redis
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False

logger = get_logger(__name__)


class ProxyPool:
    """代理池管理器"""

    def __init__(self, redis_host='localhost', redis_port=6379):
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
            except Exception:
                self.redis_client = None
        else:
            self.redis_client = None

        self.proxy_sources = [
            'http://proxy-api-1.com/get',
            'http://proxy-api-2.com/get',
            # 添加更多代理源
        ]
        self.failed_proxies = set()

        # 本地代理缓存（当redis不可用时）
        self.local_proxy_cache = set()

    def get_proxy(self) -> Optional[str]:
        """获取可用代理"""
        # 从Redis缓存获取
        if self.redis_client:
            try:
                proxy = self.redis_client.spop('available_proxies')
                if proxy:
                    logger.info(f"从缓存获取代理: {proxy}")
                    return proxy
            except Exception as e:
                logger.warning(f"Redis操作失败: {e}")

        # 从本地缓存获取
        if self.local_proxy_cache:
            proxy = self.local_proxy_cache.pop()
            logger.info(f"从本地缓存获取代理: {proxy}")
            return proxy

        # 从代理提供商获取新代理
        return self._fetch_new_proxy()

    def _fetch_new_proxy(self) -> Optional[str]:
        """从代理提供商获取新代理"""
        for source in self.proxy_sources:
            try:
                response = requests.get(source, timeout=10)
                if response.status_code == 200:
                    proxy_data = response.json()
                    proxy = f"http://{proxy_data['ip']}:{proxy_data['port']}"

                    # 验证代理可用性
                    if self._validate_proxy(proxy):
                        self.redis_client.sadd('available_proxies', proxy)
                        logger.info(f"获取新代理: {proxy}")
                        return proxy
            except Exception as e:
                logger.error(f"从 {source} 获取代理失败: {e}")

        return None

    def _validate_proxy(self, proxy: str) -> bool:
        """验证代理可用性"""
        try:
            response = requests.get(
                'http://httpbin.org/ip',
                proxies={'http': proxy, 'https': proxy},
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"代理验证失败 {proxy}: {e}")
            return False

    def mark_failed(self, proxy: str):
        """标记失败代理"""
        self.failed_proxies.add(proxy)
        self.redis_client.srem('available_proxies', proxy)
        self.redis_client.sadd('failed_proxies', proxy)
        logger.warning(f"标记代理失败: {proxy}")

    def get_proxy_count(self) -> int:
        """获取可用代理数量"""
        return self.redis_client.scard('available_proxies')


class UserAgentRotator:
    """User-Agent轮换器"""

    def __init__(self):
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (X11; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/91.0.864.59',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
        ]
        self.current_index = 0

    def get_random_ua(self) -> str:
        """获取随机User-Agent"""
        return random.choice(self.user_agents)

    def get_next_ua(self) -> str:
        """获取下一个User-Agent"""
        ua = self.user_agents[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.user_agents)
        return ua


class CaptchaSolver:
    """验证码识别器"""

    def __init__(self):
        self.ocr_api_key = None  # 配置OCR API密钥

    def solve_captcha(self, image_data: bytes) -> Optional[str]:
        """识别验证码"""
        try:
            # 这里可以集成第三方验证码识别服务
            # 如：打码平台、OCR服务等

            # 示例：使用本地OCR
            return self._local_ocr(image_data)
        except Exception as e:
            logger.error(f"验证码识别失败: {e}")
            return None

    def _local_ocr(self, image_data: bytes) -> Optional[str]:
        """本地OCR识别"""
        # 这里可以使用tesseract、paddleocr等
        # 暂时返回None，需要根据实际情况实现
        return None


class AntiCrawlerModule:
    """反爬虫处理模块"""

    def __init__(self, redis_host='localhost', redis_port=6379):
        self.proxy_pool = ProxyPool(redis_host, redis_port)
        self.ua_rotator = UserAgentRotator()
        self.captcha_solver = CaptchaSolver()
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
            except Exception:
                self.redis_client = None
        else:
            self.redis_client = None

    def get_session(self, use_proxy=True) -> requests.Session:
        """获取配置好的会话"""
        session = requests.Session()

        # 设置User-Agent
        session.headers.update({
            'User-Agent': self.ua_rotator.get_random_ua(),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })

        # 设置代理
        if use_proxy:
            proxy = self.proxy_pool.get_proxy()
            if proxy:
                session.proxies = {'http': proxy, 'https': proxy}

        return session

    def handle_block(self, response: requests.Response, session: requests.Session) -> bool:
        """处理被封禁的情况"""
        if response.status_code == 403 or '验证码' in response.text:
            logger.warning("检测到反爬虫机制，尝试处理...")

            # 标记当前代理失败
            if session.proxies:
                proxy = session.proxies.get('http')
                if proxy:
                    self.proxy_pool.mark_failed(proxy)

            # 更换代理和User-Agent
            new_proxy = self.proxy_pool.get_proxy()
            if new_proxy:
                session.proxies = {'http': new_proxy, 'https': new_proxy}

            session.headers['User-Agent'] = self.ua_rotator.get_random_ua()

            # 随机延迟
            import time
            time.sleep(random.uniform(5, 15))

            return True

        return False

    def check_rate_limit(self, domain: str) -> bool:
        """检查访问频率限制"""
        if not self.redis_client:
            return True

        key = f"rate_limit:{domain}"
        current_count = self.redis_client.get(key)

        if current_count is None:
            # 第一次访问
            self.redis_client.setex(key, 60, 1)  # 1分钟内计数
            return True

        if int(current_count) >= 10:  # 每分钟最多10次请求
            logger.warning(f"域名 {domain} 访问频率过高，需要等待")
            return False

        self.redis_client.incr(key)
        return True

    def add_delay(self, min_delay=1, max_delay=3):
        """添加随机延迟"""
        import time
        delay = random.uniform(min_delay, max_delay)
        logger.debug(f"随机延迟 {delay:.2f} 秒")
        time.sleep(delay)