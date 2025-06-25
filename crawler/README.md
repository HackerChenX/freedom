# 股市信息爬虫系统

## 概述

这是一个专门用于爬取股市信息和投研"小作文"的爬虫系统，支持多个主流投资平台的数据采集，包括淘股吧、雪球、韭研公社、慧博投研、萝卜投研等。

## 主要功能

### 🕷️ 多源数据采集
- **淘股吧**: 散户讨论、热门话题、概念股挖掘
- **雪球**: 专业投资者观点、公司分析、财报解读
- **韭研公社**: 投研报告、行业分析、政策解读
- **慧博投研**: 券商研报、深度分析、投资策略
- **萝卜投研**: 量化分析、技术指标、市场情绪

### 🛡️ 智能反爬虫
- 代理池管理和轮换
- User-Agent随机化
- 访问频率控制
- 验证码识别支持
- 自动重试机制

### 🧠 智能信息提取
- 股票代码自动识别
- 公司名称实体提取
- 概念股关键词匹配
- 情绪分析和热度计算
- 置信度评估

### 📊 数据存储与处理
- ClickHouse高性能存储
- Redis缓存和任务队列
- 数据清洗和结构化
- 实时处理流水线

## 快速开始

### 1. 安装依赖

```bash
pip install requests beautifulsoup4 redis clickhouse-driver pandas
```

### 2. 运行演示

```bash
# 测试概念股提取功能
python bin/run_crawler.py --demo

# 查看系统配置
python bin/run_crawler.py --config

# 运行测试
python bin/run_crawler.py --test
```

### 3. 演示结果

运行演示后，您将看到类似以下的输出：

```
=== 概念股提取结果 ===
股票代码: ['601058', '002984', '601966']
公司名称: []
概念关键词: []
置信度: 0.50
```

## 系统架构

```
crawler/
├── __init__.py              # 模块初始化
├── config.py               # 配置管理
├── anti_crawler.py         # 反爬虫模块
├── scheduler.py            # 任务调度器
├── spiders/                # 爬虫实现
│   ├── __init__.py
│   ├── base_spider.py      # 基础爬虫类
│   └── taoguba_spider.py   # 淘股吧爬虫
├── processors/             # 数据处理
│   ├── __init__.py
│   └── concept_extractor.py # 概念股提取器
└── demo.py                 # 演示脚本
```

## 核心组件

### 反爬虫模块 (AntiCrawlerModule)
- **代理池管理**: 自动获取和验证代理IP
- **User-Agent轮换**: 模拟不同浏览器访问
- **频率控制**: 避免触发反爬虫机制
- **异常处理**: 自动处理封禁和验证码

### 概念股提取器 (ConceptStockExtractor)
- **股票代码识别**: 正则匹配6位股票代码
- **公司名称提取**: 基于NLP的实体识别
- **概念关键词**: 预定义概念词典匹配
- **置信度计算**: 多维度信号综合评估

### 爬虫调度器 (CrawlerScheduler)
- **任务队列**: 优先级队列管理
- **线程池**: 并发爬取控制
- **监控统计**: 实时性能监控
- **错误重试**: 自动重试失败任务

## 配置说明

系统支持通过环境变量进行配置：

```bash
# Redis配置
export REDIS_HOST=localhost
export REDIS_PORT=6379

# ClickHouse配置
export CLICKHOUSE_HOST=localhost
export CLICKHOUSE_PORT=8123

# 爬虫设置
export MAX_WORKERS=5
export USE_PROXY=true
```

## 使用示例

### 基本用法

```python
from crawler.processors.concept_extractor import ConceptStockExtractor

# 初始化提取器
extractor = ConceptStockExtractor()

# 提取股票信息
text = "赛轮轮胎(601058)、森麒麟(002984)等新能源概念股值得关注"
result = extractor.extract_stocks(text)

print(f"股票代码: {result['stock_codes']}")
print(f"概念关键词: {result['concepts']}")
print(f"置信度: {result['confidence']}")
```

### 高级用法

```python
from crawler.anti_crawler import AntiCrawlerModule
from crawler.spiders.taoguba_spider import TaogubaSpider

# 初始化反爬虫模块
anti_crawler = AntiCrawlerModule()

# 初始化爬虫
spider = TaogubaSpider(anti_crawler_module=anti_crawler)

# 获取文章列表
urls = spider.get_article_urls(page=1)

# 爬取文章详情
for url in urls[:3]:  # 只爬取前3篇作为演示
    response = spider.get_page(url)
    if response:
        article = spider.parse_article_detail(response)
        print(f"标题: {article.get('title')}")
```

## 注意事项

### 法律合规
- 遵守robots.txt协议
- 控制访问频率，避免对目标网站造成压力
- 仅爬取公开信息，不涉及用户隐私
- 用于学习和研究目的

### 技术建议
- 建议在生产环境中配置Redis和ClickHouse
- 使用代理池以提高稳定性
- 定期更新User-Agent列表
- 监控爬虫运行状态

## 扩展开发

### 添加新的数据源

1. 继承BaseSpider类
2. 实现必要的抽象方法
3. 在config.py中添加配置
4. 注册到调度器中

### 自定义信息提取

1. 扩展ConceptStockExtractor类
2. 添加新的提取规则
3. 更新概念词典
4. 调整置信度计算

## 技术支持

如有问题或建议，请联系开发团队或提交Issue。

---

**免责声明**: 本系统仅用于技术学习和研究目的，使用者需要遵守相关法律法规和网站服务条款。