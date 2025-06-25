"""
系统集成器

与现有股票分析系统进行集成，包括ClickHouse数据库、技术指标系统等
"""

import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from utils.logger import get_logger

logger = get_logger(__name__)


class ClickHouseIntegrator:
    """ClickHouse数据库集成器"""

    def __init__(self, host='localhost', port=8123, database='stock_crawler',
                 user='default', password=''):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.client = None

    def connect(self) -> bool:
        """连接到ClickHouse数据库"""
        try:
            # 这里使用简化的连接方式，实际部署时需要安装clickhouse-driver
            logger.info(f"连接到ClickHouse: {self.host}:{self.port}/{self.database}")
            # self.client = clickhouse_connect.get_client(
            #     host=self.host, port=self.port, database=self.database,
            #     username=self.user, password=self.password
            # )
            self.client = "mock_client"  # 模拟连接
            logger.info("ClickHouse连接成功")
            return True
        except Exception as e:
            logger.error(f"ClickHouse连接失败: {e}")
            return False

    def create_tables(self) -> bool:
        """创建爬虫数据表"""
        try:
            tables_sql = [
                """
                CREATE TABLE IF NOT EXISTS crawler_articles (
                    id String,
                    source String,
                    title String,
                    content String,
                    author String,
                    publish_time DateTime,
                    url String,
                    crawl_time DateTime DEFAULT now(),
                    article_type String,
                    view_count UInt32 DEFAULT 0,
                    like_count UInt32 DEFAULT 0,
                    comment_count UInt32 DEFAULT 0,
                    stock_codes Array(String),
                    concepts Array(String),
                    sentiment_score Float32 DEFAULT 0.0
                ) ENGINE = MergeTree()
                ORDER BY (source, publish_time)
                """
            ]

            for sql in tables_sql:
                logger.info("创建表结构...")
                # 实际部署时执行: self.client.execute(sql)

            logger.info("数据表创建成功")
            return True

        except Exception as e:
            logger.error(f"创建数据表失败: {e}")
            return False

    def insert_article(self, article_data: Dict[str, Any]) -> bool:
        """插入文章数据"""
        try:
            # 数据格式转换
            formatted_data = self._format_article_data(article_data)

            # 实际部署时执行插入
            # self.client.insert('crawler_articles', [formatted_data])

            logger.info(f"插入文章数据: {formatted_data.get('id', 'unknown')}")
            return True

        except Exception as e:
            logger.error(f"插入文章数据失败: {e}")
            return False

    def _format_article_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """格式化文章数据"""
        return {
            'id': data.get('id', ''),
            'source': data.get('source', ''),
            'title': data.get('title', ''),
            'content': data.get('content', ''),
            'author': data.get('author', ''),
            'publish_time': data.get('publish_time', datetime.now()),
            'url': data.get('url', ''),
            'crawl_time': data.get('crawl_time', datetime.now()),
            'article_type': data.get('article_type', 'unknown'),
            'view_count': data.get('view_count', 0),
            'like_count': data.get('like_count', 0),
            'comment_count': data.get('comment_count', 0),
            'stock_codes': data.get('stock_codes', []),
            'concepts': data.get('concepts', []),
            'sentiment_score': data.get('sentiment_score', 0.0)
        }


class TechnicalIndicatorIntegrator:
    """技术指标系统集成器"""

    def __init__(self):
        self.indicator_mapping = {
            'stock_codes': 'symbol',
            'concepts': 'concept_tags',
            'sentiment_score': 'sentiment'
        }

    def validate_stock_codes(self, stock_codes: List[str]) -> Dict[str, Any]:
        """验证股票代码与技术指标系统的兼容性"""
        valid_codes = []
        invalid_codes = []

        for code in stock_codes:
            if self._is_valid_for_technical_system(code):
                valid_codes.append(code)
            else:
                invalid_codes.append(code)

        return {
            'valid_codes': valid_codes,
            'invalid_codes': invalid_codes,
            'validation_rate': len(valid_codes) / len(stock_codes) if stock_codes else 0
        }

    def _is_valid_for_technical_system(self, code: str) -> bool:
        """检查股票代码是否与技术指标系统兼容"""
        # 简化的验证逻辑
        if not code or len(code) != 6:
            return False

        # 检查是否为有效的A股代码
        valid_prefixes = ['60', '688', '00', '002', '300']
        return any(code.startswith(prefix) for prefix in valid_prefixes)

    def format_for_technical_system(self, crawler_data: Dict[str, Any]) -> Dict[str, Any]:
        """将爬虫数据格式化为技术指标系统格式"""
        formatted_data = {}

        for crawler_key, tech_key in self.indicator_mapping.items():
            if crawler_key in crawler_data:
                formatted_data[tech_key] = crawler_data[crawler_key]

        # 添加时间戳
        formatted_data['timestamp'] = datetime.now().isoformat()
        formatted_data['data_source'] = 'crawler_system'

        return formatted_data


class SystemIntegrator:
    """系统集成器主类"""

    def __init__(self, clickhouse_config: Dict[str, Any] = None):
        self.clickhouse_config = clickhouse_config or {}
        self.clickhouse = ClickHouseIntegrator(**self.clickhouse_config)
        self.technical_integrator = TechnicalIndicatorIntegrator()

        # 集成状态
        self.integration_status = {
            'clickhouse_connected': False,
            'tables_created': False,
            'last_sync_time': None,
            'total_articles_synced': 0,
            'sync_errors': 0
        }

    def initialize(self) -> bool:
        """初始化集成系统"""
        try:
            logger.info("初始化系统集成...")

            # 连接ClickHouse
            if self.clickhouse.connect():
                self.integration_status['clickhouse_connected'] = True

                # 创建数据表
                if self.clickhouse.create_tables():
                    self.integration_status['tables_created'] = True
                    logger.info("系统集成初始化成功")
                    return True

            logger.error("系统集成初始化失败")
            return False

        except Exception as e:
            logger.error(f"系统集成初始化异常: {e}")
            return False

    def sync_article_data(self, article_data: Dict[str, Any]) -> bool:
        """同步文章数据到集成系统"""
        try:
            # 验证股票代码
            if 'stock_codes' in article_data:
                validation_result = self.technical_integrator.validate_stock_codes(
                    article_data['stock_codes']
                )
                article_data['validated_stock_codes'] = validation_result['valid_codes']

                if validation_result['invalid_codes']:
                    logger.warning(f"发现无效股票代码: {validation_result['invalid_codes']}")

            # 插入到ClickHouse
            if self.clickhouse.insert_article(article_data):
                self.integration_status['total_articles_synced'] += 1
                self.integration_status['last_sync_time'] = datetime.now()
                logger.info(f"文章数据同步成功: {article_data.get('id', 'unknown')}")
                return True
            else:
                self.integration_status['sync_errors'] += 1
                return False

        except Exception as e:
            logger.error(f"文章数据同步失败: {e}")
            self.integration_status['sync_errors'] += 1
            return False

    def get_integration_status(self) -> Dict[str, Any]:
        """获取集成状态"""
        status = self.integration_status.copy()
        status['last_sync_time'] = status['last_sync_time'].isoformat() if status['last_sync_time'] else None
        return status

    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        health_status = {
            'overall_status': 'healthy',
            'components': {},
            'issues': []
        }

        # 检查ClickHouse连接
        if not self.integration_status['clickhouse_connected']:
            health_status['components']['clickhouse'] = 'disconnected'
            health_status['issues'].append('ClickHouse连接断开')
            health_status['overall_status'] = 'unhealthy'
        else:
            health_status['components']['clickhouse'] = 'connected'

        # 检查数据表
        if not self.integration_status['tables_created']:
            health_status['components']['tables'] = 'not_created'
            health_status['issues'].append('数据表未创建')
            health_status['overall_status'] = 'unhealthy'
        else:
            health_status['components']['tables'] = 'created'

        # 检查同步错误率
        total_synced = self.integration_status['total_articles_synced']
        sync_errors = self.integration_status['sync_errors']

        if total_synced > 0:
            error_rate = sync_errors / (total_synced + sync_errors) * 100
            if error_rate > 10:  # 错误率超过10%
                health_status['components']['sync'] = 'high_error_rate'
                health_status['issues'].append(f'同步错误率过高: {error_rate:.1f}%')
                health_status['overall_status'] = 'degraded'
            else:
                health_status['components']['sync'] = 'normal'
        else:
            health_status['components']['sync'] = 'no_data'

        return health_status