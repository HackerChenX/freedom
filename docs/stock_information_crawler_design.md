# 股市信息爬虫系统技术方案

## 1. 系统概述

### 1.1 项目目标
构建一个高效、稳定的股市信息爬虫系统，用于获取各大投研平台的"小作文"、研报、概念股信息等，为投资决策提供信息支持。

### 1.2 目标数据源
- **淘股吧**: 散户讨论、热门话题、概念股挖掘
- **雪球**: 专业投资者观点、公司分析、财报解读
- **韭研公社**: 投研报告、行业分析、政策解读
- **慧博投研**: 券商研报、深度分析、投资策略
- **萝卜投研**: 量化分析、技术指标、市场情绪

### 1.3 核心功能
- 多源数据采集与整合
- 智能反爬虫机制
- 实时数据处理与存储
- 信息提取与分类
- 概念股自动识别
- 情绪分析与热度计算

## 2. 系统架构设计

### 2.1 整体架构
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   数据采集层     │    │   数据处理层     │    │   数据存储层     │
│                │    │                │    │                │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │ 爬虫调度器   │ │    │ │ 数据清洗器   │ │    │ │ ClickHouse  │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │ 反爬虫模块   │ │    │ │ NLP处理器   │ │    │ │ Redis缓存   │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │ 多源爬虫     │ │    │ │ 信息提取器   │ │    │ │ 文件存储     │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 2.2 技术栈选择
- **爬虫框架**: Scrapy + Selenium + Requests
- **反爬虫**: 代理池 + User-Agent轮换 + 验证码识别
- **数据处理**: Pandas + jieba + transformers
- **任务调度**: Celery + Redis
- **数据存储**: ClickHouse + Redis + MinIO
- **监控告警**: Prometheus + Grafana + 钉钉/企微

## 3. 核心模块设计

### 3.1 爬虫调度器 (CrawlerScheduler)
```python
class CrawlerScheduler:
    """爬虫任务调度器"""

    def __init__(self):
        self.task_queue = TaskQueue()
        self.crawler_pool = CrawlerPool()
        self.monitor = CrawlerMonitor()

    def schedule_tasks(self):
        """调度爬虫任务"""
        pass

    def manage_crawlers(self):
        """管理爬虫实例"""
        pass
```

### 3.2 反爬虫模块 (AntiCrawlerModule)
```python
class AntiCrawlerModule:
    """反爬虫处理模块"""

    def __init__(self):
        self.proxy_pool = ProxyPool()
        self.ua_rotator = UserAgentRotator()
        self.captcha_solver = CaptchaSolver()

    def get_session(self):
        """获取配置好的会话"""
        pass

    def handle_block(self, response):
        """处理被封禁的情况"""
        pass
```

### 3.3 数据处理器 (DataProcessor)
```python
class DataProcessor:
    """数据处理器"""

    def __init__(self):
        self.cleaner = DataCleaner()
        self.nlp_processor = NLPProcessor()
        self.extractor = InformationExtractor()

    def process_article(self, raw_data):
        """处理文章数据"""
        pass

    def extract_concepts(self, content):
        """提取概念股信息"""
        pass
```

## 4. 数据源分析与策略

### 4.1 淘股吧 (taoguba.com.cn)
- **特点**: 散户聚集地，信息量大但质量参差不齐
- **反爬策略**: IP限制 + 验证码 + 动态加载
- **爬取策略**:
  - 使用代理池轮换IP
  - Selenium处理动态内容
  - 重点关注热门话题和概念股讨论

### 4.2 雪球 (xueqiu.com)
- **特点**: 专业投资者较多，内容质量较高
- **反爬策略**: 登录验证 + API限流 + 动态token
- **爬取策略**:
  - 模拟登录获取token
  - 使用官方API接口
  - 重点关注大V观点和公司分析

### 4.3 慧博投研 (hibor.com.cn)
- **特点**: 券商研报聚合平台，内容专业
- **反爬策略**: 付费内容 + 下载限制 + IP监控
- **爬取策略**:
  - 重点爬取免费摘要部分
  - 分析研报标题和关键词
  - 提取投资评级和目标价

### 4.4 韭研公社 (jiuyan.info)
- **特点**: 投研报告聚合，政策解读专业
- **反爬策略**: 频率限制 + 内容加密
- **爬取策略**:
  - 控制访问频率
  - 重点关注政策解读和行业分析

### 4.5 萝卜投研 (robo.datayes.com)
- **特点**: 量化分析平台，数据质量高
- **反爬策略**: API认证 + 数据加密
- **爬取策略**:
  - 重点获取市场情绪指标
  - 关注量化分析报告

## 5. 数据存储设计

### 5.1 ClickHouse表结构
```sql
-- 原始文章表
CREATE TABLE raw_articles (
    id String,
    source String,
    title String,
    content String,
    author String,
    publish_time DateTime,
    url String,
    crawl_time DateTime DEFAULT now(),
    article_type String,
    view_count UInt32,
    like_count UInt32,
    comment_count UInt32
) ENGINE = MergeTree()
ORDER BY (source, publish_time);

-- 概念股信息表
CREATE TABLE concept_stocks (
    concept_name String,
    stock_codes Array(String),
    article_id String,
    confidence Float32,
    extract_time DateTime DEFAULT now(),
    mention_count UInt32,
    sentiment_score Float32
) ENGINE = MergeTree()
ORDER BY (concept_name, extract_time);

-- 情绪分析表
CREATE TABLE sentiment_analysis (
    article_id String,
    sentiment_score Float32,
    emotion_type String,
    keywords Array(String),
    analysis_time DateTime DEFAULT now(),
    confidence Float32
) ENGINE = MergeTree()
ORDER BY (analysis_time);

-- 热点话题表
CREATE TABLE hot_topics (
    topic_name String,
    related_stocks Array(String),
    heat_score Float32,
    article_count UInt32,
    first_mention DateTime,
    last_update DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY (heat_score DESC, last_update);
```

### 5.2 Redis缓存策略
- **代理池缓存**: 存储可用代理IP，TTL=30分钟
- **任务队列**: 存储待处理的爬虫任务
- **去重缓存**: 存储已爬取文章的URL哈希，TTL=7天
- **限流缓存**: 记录各站点的访问频率
- **热点缓存**: 缓存热门话题和概念股，TTL=1小时

## 6. 信息提取与分析

### 6.1 NLP处理流程
1. **文本预处理**: 去除HTML标签、特殊字符清理
2. **分词处理**: 使用jieba进行中文分词
3. **关键词提取**: TF-IDF + TextRank算法
4. **实体识别**: 识别股票代码、公司名称、概念名称
5. **情绪分析**: 使用预训练模型分析文章情绪倾向

### 6.2 概念股识别算法
```python
class ConceptStockExtractor:
    """概念股提取器"""

    def __init__(self):
        self.stock_dict = self.load_stock_dict()
        self.concept_dict = self.load_concept_dict()
        self.nlp_model = self.load_nlp_model()

    def extract_stocks(self, text):
        """从文本中提取股票信息"""
        # 1. 正则匹配股票代码
        stock_codes = self._extract_stock_codes(text)

        # 2. 实体识别公司名称
        company_names = self._extract_company_names(text)

        # 3. 概念关键词匹配
        concepts = self._extract_concepts(text)

        # 4. 置信度计算
        confidence = self._calculate_confidence(stock_codes, company_names, concepts)

        return {
            'stock_codes': stock_codes,
            'company_names': company_names,
            'concepts': concepts,
            'confidence': confidence
        }

    def _extract_stock_codes(self, text):
        """提取股票代码"""
        import re
        pattern = r'[0-9]{6}'
        return re.findall(pattern, text)

    def _extract_company_names(self, text):
        """提取公司名称"""
        # 使用NER模型识别公司实体
        pass

    def _extract_concepts(self, text):
        """提取概念关键词"""
        # 基于概念词典匹配
        pass

    def _calculate_confidence(self, codes, names, concepts):
        """计算提取置信度"""
        # 基于多种信号计算综合置信度
        pass
```

### 6.3 情绪分析模型
```python
class SentimentAnalyzer:
    """情绪分析器"""

    def __init__(self):
        self.model = self.load_sentiment_model()
        self.tokenizer = self.load_tokenizer()

    def analyze_sentiment(self, text):
        """分析文本情绪"""
        # 预处理文本
        processed_text = self.preprocess_text(text)

        # 模型预测
        sentiment_score = self.model.predict(processed_text)

        # 情绪分类
        emotion_type = self.classify_emotion(sentiment_score)

        return {
            'sentiment_score': sentiment_score,
            'emotion_type': emotion_type,
            'confidence': self.calculate_confidence(sentiment_score)
        }

    def classify_emotion(self, score):
        """情绪分类"""
        if score > 0.6:
            return 'positive'
        elif score < -0.6:
            return 'negative'
        else:
            return 'neutral'
```

## 7. 反爬虫策略详解

### 7.1 代理池管理
```python
class ProxyPool:
    """代理池管理器"""

    def __init__(self):
        self.proxy_list = []
        self.failed_proxies = set()
        self.redis_client = redis.Redis()

    def get_proxy(self):
        """获取可用代理"""
        # 从Redis缓存获取
        proxy = self.redis_client.spop('available_proxies')
        if proxy:
            return proxy.decode()

        # 从代理提供商获取新代理
        return self.fetch_new_proxy()

    def mark_failed(self, proxy):
        """标记失败代理"""
        self.failed_proxies.add(proxy)
        self.redis_client.sadd('failed_proxies', proxy)

    def validate_proxy(self, proxy):
        """验证代理可用性"""
        try:
            response = requests.get(
                'http://httpbin.org/ip',
                proxies={'http': proxy, 'https': proxy},
                timeout=10
            )
            return response.status_code == 200
        except:
            return False
```

### 7.2 User-Agent轮换
```python
class UserAgentRotator:
    """User-Agent轮换器"""

    def __init__(self):
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
            # 更多User-Agent...
        ]
        self.current_index = 0

    def get_random_ua(self):
        """获取随机User-Agent"""
        import random
        return random.choice(self.user_agents)

    def get_next_ua(self):
        """获取下一个User-Agent"""
        ua = self.user_agents[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.user_agents)
        return ua
```