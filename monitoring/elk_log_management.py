#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
ELK/EFK 日志管理系统

提供企业级日志处理解决方案：
1. Elasticsearch 日志存储和搜索
2. Logstash/Fluentd 日志收集和处理
3. Kibana 日志可视化和分析
4. 结构化日志记录
5. 日志聚合和分析
6. 日志告警和监控
"""

import os
import json
import time
import yaml
import logging
import threading
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import queue
import gzip
import socket
import traceback

from utils.logger import get_logger
from utils.exception_handler import exception_handler
from utils.performance_monitor import performance_monitor

logger = get_logger(__name__)


class LogLevel(Enum):
    """日志级别"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(Enum):
    """日志格式"""
    JSON = "json"
    LOGSTASH = "logstash"
    PLAIN = "plain"
    CEF = "cef"  # Common Event Format


@dataclass
class LogEntry:
    """日志条目"""
    timestamp: datetime
    level: LogLevel
    message: str
    logger_name: str
    module: str
    function: str
    line_number: int
    thread_id: int
    process_id: int
    hostname: str
    application: str = "stock_analysis_system"
    environment: str = "production"
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    extra_fields: Optional[Dict[str, Any]] = None


@dataclass
class ElasticsearchConfig:
    """Elasticsearch配置"""
    hosts: List[str] = None
    username: str = ""
    password: str = ""
    index_pattern: str = "stock-logs-%Y.%m.%d"
    index_template_name: str = "stock-logs-template"
    max_retries: int = 3
    timeout: int = 30

    def __post_init__(self):
        if self.hosts is None:
            self.hosts = ["localhost:9200"]


@dataclass
class KibanaConfig:
    """Kibana配置"""
    url: str = "http://localhost:5601"
    username: str = ""
    password: str = ""
    space_id: str = "default"


class StructuredLogger:
    """
    结构化日志记录器

    提供统一的结构化日志记录功能
    """

    def __init__(self, name: str, format_type: LogFormat = LogFormat.JSON,
                 output_file: Optional[str] = None, console_output: bool = True):
        """
        初始化结构化日志记录器

        Args:
            name: 日志记录器名称
            format_type: 日志格式类型
            output_file: 输出文件路径
            console_output: 是否输出到控制台
        """
        self.name = name
        self.format_type = format_type
        self.output_file = output_file
        self.console_output = console_output

        # 创建日志目录
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        # 日志队列（用于异步写入）
        self.log_queue = queue.Queue()
        self.log_handlers: List[Callable[[LogEntry], None]] = []

        # 启动日志处理线程
        self.logging_active = True
        self.log_thread = threading.Thread(target=self._process_logs, daemon=True)
        self.log_thread.start()

        # 主机名
        self.hostname = socket.gethostname()

        logger.info(f"结构化日志记录器初始化完成: {name}")

    def add_handler(self, handler: Callable[[LogEntry], None]):
        """添加日志处理器"""
        self.log_handlers.append(handler)

    def _create_log_entry(self, level: LogLevel, message: str,
                         extra_fields: Optional[Dict[str, Any]] = None) -> LogEntry:
        """创建日志条目"""
        frame = None
        try:
            # 获取调用栈信息
            import inspect
            frame = inspect.currentframe().f_back.f_back
            module = frame.f_globals.get('__name__', 'unknown')
            function = frame.f_code.co_name
            line_number = frame.f_lineno
        except Exception:
            module = 'unknown'
            function = 'unknown'
            line_number = 0

        return LogEntry(
            timestamp=datetime.now(),
            level=level,
            message=message,
            logger_name=self.name,
            module=module,
            function=function,
            line_number=line_number,
            thread_id=threading.get_ident(),
            process_id=os.getpid(),
            hostname=self.hostname,
            extra_fields=extra_fields or {}
        )

    def debug(self, message: str, **kwargs):
        """记录DEBUG日志"""
        entry = self._create_log_entry(LogLevel.DEBUG, message, kwargs)
        self.log_queue.put(entry)

    def info(self, message: str, **kwargs):
        """记录INFO日志"""
        entry = self._create_log_entry(LogLevel.INFO, message, kwargs)
        self.log_queue.put(entry)

    def warning(self, message: str, **kwargs):
        """记录WARNING日志"""
        entry = self._create_log_entry(LogLevel.WARNING, message, kwargs)
        self.log_queue.put(entry)

    def error(self, message: str, **kwargs):
        """记录ERROR日志"""
        entry = self._create_log_entry(LogLevel.ERROR, message, kwargs)
        self.log_queue.put(entry)

    def critical(self, message: str, **kwargs):
        """记录CRITICAL日志"""
        entry = self._create_log_entry(LogLevel.CRITICAL, message, kwargs)
        self.log_queue.put(entry)

    def exception(self, message: str, **kwargs):
        """记录异常日志"""
        # 添加异常信息
        kwargs['exception'] = traceback.format_exc()
        entry = self._create_log_entry(LogLevel.ERROR, message, kwargs)
        self.log_queue.put(entry)

    def _process_logs(self):
        """处理日志队列"""
        while self.logging_active:
            try:
                log_entry = self.log_queue.get(timeout=1)

                # 格式化日志
                formatted_log = self._format_log(log_entry)

                # 输出到控制台
                if self.console_output:
                    print(formatted_log)

                # 写入文件
                if self.output_file:
                    self._write_to_file(formatted_log)

                # 调用自定义处理器
                for handler in self.log_handlers:
                    try:
                        handler(log_entry)
                    except Exception as e:
                        print(f"日志处理器执行失败: {e}")

            except queue.Empty:
                continue
            except Exception as e:
                print(f"处理日志失败: {e}")

    def _format_log(self, log_entry: LogEntry) -> str:
        """格式化日志"""
        if self.format_type == LogFormat.JSON:
            return json.dumps(asdict(log_entry), default=str, ensure_ascii=False)

        elif self.format_type == LogFormat.LOGSTASH:
            logstash_format = {
                "@timestamp": log_entry.timestamp.isoformat(),
                "@version": "1",
                "level": log_entry.level.value,
                "message": log_entry.message,
                "logger_name": log_entry.logger_name,
                "source": {
                    "module": log_entry.module,
                    "function": log_entry.function,
                    "line": log_entry.line_number
                },
                "process": {
                    "thread_id": log_entry.thread_id,
                    "process_id": log_entry.process_id
                },
                "host": log_entry.hostname,
                "application": log_entry.application,
                "environment": log_entry.environment
            }

            if log_entry.extra_fields:
                logstash_format.update(log_entry.extra_fields)

            return json.dumps(logstash_format, default=str, ensure_ascii=False)

        elif self.format_type == LogFormat.PLAIN:
            return (f"{log_entry.timestamp.isoformat()} "
                   f"[{log_entry.level.value}] "
                   f"{log_entry.logger_name} - {log_entry.message}")

        else:
            return json.dumps(asdict(log_entry), default=str, ensure_ascii=False)

    def _write_to_file(self, formatted_log: str):
        """写入日志文件"""
        try:
            with open(self.output_file, 'a', encoding='utf-8') as f:
                f.write(formatted_log + '\n')
        except Exception as e:
            print(f"写入日志文件失败: {e}")

    def close(self):
        """关闭日志记录器"""
        self.logging_active = False
        if self.log_thread.is_alive():
            self.log_thread.join(timeout=5)


class ElasticsearchClient:
    """
    Elasticsearch客户端

    提供日志存储和搜索功能
    """

    def __init__(self, config: ElasticsearchConfig):
        """
        初始化Elasticsearch客户端

        Args:
            config: Elasticsearch配置
        """
        self.config = config
        self.client = None

        # 尝试导入并初始化elasticsearch客户端
        try:
            from elasticsearch import Elasticsearch
from db.sql_manager import SQLManager, QueryType

            self.client = Elasticsearch(
                hosts=config.hosts,
                http_auth=(config.username, config.password) if config.username else None,
                timeout=config.timeout,
                max_retries=config.max_retries
            )

            # 测试连接
            if self.client.ping():
                logger.info("Elasticsearch连接成功")
            else:
                logger.warning("Elasticsearch连接失败")

        except ImportError:
            logger.warning("elasticsearch包未安装，使用模拟模式")
        except Exception as e:
            logger.error(f"Elasticsearch客户端初始化失败: {e}")

    @exception_handler(reraise=True)
    def create_index_template(self) -> bool:
        """创建索引模板"""
        if not self.client:
            return False

        try:
            template = {
                "index_patterns": [self.config.index_pattern.replace('%Y.%m.%d', '*')],
                "template": {
                    "settings": {
                        "number_of_shards": 1,
                        "number_of_replicas": 1,
                        "index.refresh_interval": "5s"
                    },
                    "mappings": {
                        "properties": {
                            "@timestamp": {"type": "date"},
                            "timestamp": {"type": "date"},
                            "level": {"type": "keyword"},
                            "message": {
                                "type": "text",
                                "fields": {
                                    "keyword": {"type": "keyword", "ignore_above": 256}
                                }
                            },
                            "logger_name": {"type": "keyword"},
                            "module": {"type": "keyword"},
                            "function": {"type": "keyword"},
                            "line_number": {"type": "integer"},
                            "thread_id": {"type": "long"},
                            "process_id": {"type": "long"},
                            "hostname": {"type": "keyword"},
                            "application": {"type": "keyword"},
                            "environment": {"type": "keyword"},
                            "trace_id": {"type": "keyword"},
                            "span_id": {"type": "keyword"},
                            "user_id": {"type": "keyword"},
                            "request_id": {"type": "keyword"},
                            "extra_fields": {"type": "object"}
                        }
                    }
                }
            }

            self.client.indices.put_index_template(
                name=self.config.index_template_name,
                body=template
            )

            logger.info(f"索引模板创建成功: {self.config.index_template_name}")
            return True

        except Exception as e:
            logger.error(f"创建索引模板失败: {e}")
            return False

    @exception_handler(reraise=True)
    def index_log(self, log_entry: LogEntry) -> bool:
        """索引日志条目"""
        if not self.client:
            return False

        try:
            index_name = log_entry.timestamp.strftime(self.config.index_pattern)

            doc = {
                "@timestamp": log_entry.timestamp.isoformat(),
                "timestamp": log_entry.timestamp.isoformat(),
                "level": log_entry.level.value,
                "message": log_entry.message,
                "logger_name": log_entry.logger_name,
                "module": log_entry.module,
                "function": log_entry.function,
                "line_number": log_entry.line_number,
                "thread_id": log_entry.thread_id,
                "process_id": log_entry.process_id,
                "hostname": log_entry.hostname,
                "application": log_entry.application,
                "environment": log_entry.environment,
                "trace_id": log_entry.trace_id,
                "span_id": log_entry.span_id,
                "user_id": log_entry.user_id,
                "request_id": log_entry.request_id
            }

            if log_entry.extra_fields:
                doc["extra_fields"] = log_entry.extra_fields

            response = self.client.index(
                index=index_name,
                document=doc
            )

            return response.get('result') == 'created'

        except Exception as e:
            logger.error(f"索引日志失败: {e}")
            return False

    @exception_handler(reraise=True)
    def search_logs(self, query: Dict[str, Any], size: int = 100) -> List[Dict[str, Any]]:
        """搜索日志"""
        if not self.client:
            return []

        try:
            response = self.client.search(
                index=self.config.index_pattern.replace('%Y.%m.%d', '*'),
                body={
                    "query": query,
                    "size": size,
                    "sort": [{"@timestamp": {"order": "desc"}}]
                }
            )

            hits = response.get('hits', {}).get('hits', [])
            return [hit['_source'] for hit in hits]

        except Exception as e:
            logger.error(f"搜索日志失败: {e}")
            return []

    @exception_handler(reraise=True)
    def get_log_statistics(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """获取日志统计信息"""
        if not self.client:
            return {}

        try:
            query = {
                "bool": {
                    "filter": [
                        {
                            "range": {
                                "@timestamp": {
                                    "gte": start_time.isoformat(),
                                    "lte": end_time.isoformat()
                                }
                            }
                        }
                    ]
                }
            }

            # 按级别统计
            level_agg = self.client.search(
                index=self.config.index_pattern.replace('%Y.%m.%d', '*'),
                body={
                    "query": query,
                    "size": 0,
                    "aggs": {
                        "levels": {
                            "terms": {"field": "level"}
                        }
                    }
                }
            )

            # 按应用统计
            app_agg = self.client.search(
                index=self.config.index_pattern.replace('%Y.%m.%d', '*'),
                body={
                    "query": query,
                    "size": 0,
                    "aggs": {
                        "applications": {
                            "terms": {"field": "application"}
                        }
                    }
                }
            )

            # 按时间统计
            time_agg = self.client.search(
                index=self.config.index_pattern.replace('%Y.%m.%d', '*'),
                body={
                    "query": query,
                    "size": 0,
                    "aggs": {
                        "logs_over_time": {
                            "date_histogram": {
                                "field": "@timestamp",
                                "interval": "1h"
                            }
                        }
                    }
                }
            )

            return {
                "total_logs": level_agg['hits']['total']['value'],
                "level_distribution": {
                    bucket['key']: bucket['doc_count']
                    for bucket in level_agg['aggregations']['levels']['buckets']
                },
                "application_distribution": {
                    bucket['key']: bucket['doc_count']
                    for bucket in app_agg['aggregations']['applications']['buckets']
                },
                "time_distribution": [
                    {
                        "timestamp": bucket['key_as_string'],
                        "count": bucket['doc_count']
                    }
                    for bucket in time_agg['aggregations']['logs_over_time']['buckets']
                ]
            }

        except Exception as e:
            logger.error(f"获取日志统计失败: {e}")
            return {}


class LogAggregator:
    """
    日志聚合器

    收集、处理和转发日志到Elasticsearch
    """

    def __init__(self, elasticsearch_config: ElasticsearchConfig,
                 buffer_size: int = 1000, flush_interval: int = 10):
        """
        初始化日志聚合器

        Args:
            elasticsearch_config: Elasticsearch配置
            buffer_size: 缓冲区大小
            flush_interval: 刷新间隔（秒）
        """
        self.elasticsearch_client = ElasticsearchClient(elasticsearch_config)
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval

        self.log_buffer: List[LogEntry] = []
        self.buffer_lock = threading.RLock()

        # 启动定时刷新线程
        self.aggregating_active = True
        self.flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self.flush_thread.start()

        logger.info("日志聚合器初始化完成")

    def add_log(self, log_entry: LogEntry):
        """添加日志条目"""
        with self.buffer_lock:
            self.log_buffer.append(log_entry)

            # 如果缓冲区满了，立即刷新
            if len(self.log_buffer) >= self.buffer_size:
                self._flush_buffer()

    def _flush_loop(self):
        """定时刷新循环"""
        while self.aggregating_active:
            try:
                time.sleep(self.flush_interval)
                self._flush_buffer()
            except Exception as e:
                logger.error(f"刷新日志缓冲区失败: {e}")

    def _flush_buffer(self):
        """刷新缓冲区"""
        with self.buffer_lock:
            if not self.log_buffer:
                return

            # 批量索引日志
            batch_logs = self.log_buffer.copy()
            self.log_buffer.clear()

            for log_entry in batch_logs:
                try:
                    self.elasticsearch_client.index_log(log_entry)
                except Exception as e:
                    logger.error(f"索引日志条目失败: {e}")

            if batch_logs:
                logger.debug(f"刷新了 {len(batch_logs)} 条日志到Elasticsearch")

    def close(self):
        """关闭聚合器"""
        self.aggregating_active = False
        self._flush_buffer()  # 最后刷新一次

        if self.flush_thread.is_alive():
            self.flush_thread.join(timeout=5)


class KibanaManager:
    """
    Kibana管理器

    自动创建Kibana仪表盘和可视化
    """

    def __init__(self, config: KibanaConfig):
        """
        初始化Kibana管理器

        Args:
            config: Kibana配置
        """
        self.config = config
        self.session = None

        # 设置HTTP会话
        import requests
        self.session = requests.Session()

        if config.username and config.password:
            self.session.auth = (config.username, config.password)

        logger.info("Kibana管理器初始化完成")

    @exception_handler(reraise=True)
    def create_index_pattern(self, pattern: str = "stock-logs-*") -> Dict[str, Any]:
        """创建索引模式"""
        try:
            url = f"{self.config.url}/api/saved_objects/index-pattern"

            payload = {
                "attributes": {
                    "title": pattern,
                    "timeFieldName": "@timestamp"
                }
            }

            response = self.session.post(url, json=payload)
            response.raise_for_status()

            result = response.json()
            logger.info(f"Kibana索引模式创建成功: {pattern}")

            return result

        except Exception as e:
            logger.error(f"创建Kibana索引模式失败: {e}")
            return {}

    @exception_handler(reraise=True)
    def create_dashboard(self) -> Dict[str, Any]:
        """创建日志分析仪表盘"""
        try:
            dashboard_config = {
                "attributes": {
                    "title": "股票分析系统 - 日志分析",
                    "type": "dashboard",
                    "description": "股票分析系统日志监控和分析仪表盘",
                    "panelsJSON": json.dumps([
                        {
                            "version": "8.0.0",
                            "id": "log-level-pie",
                            "type": "visualization",
                            "gridData": {"x": 0, "y": 0, "w": 24, "h": 15}
                        },
                        {
                            "version": "8.0.0",
                            "id": "log-timeline",
                            "type": "visualization",
                            "gridData": {"x": 24, "y": 0, "w": 24, "h": 15}
                        }
                    ])
                }
            }

            url = f"{self.config.url}/api/saved_objects/dashboard"
            response = self.session.post(url, json=dashboard_config)
            response.raise_for_status()

            result = response.json()
            logger.info("Kibana仪表盘创建成功")

            return result

        except Exception as e:
            logger.error(f"创建Kibana仪表盘失败: {e}")
            return {}


class ELKLogManager:
    """
    ELK日志管理系统

    整合Elasticsearch、Logstash、Kibana功能的统一日志管理
    """

    def __init__(self, elasticsearch_config: ElasticsearchConfig,
                 kibana_config: Optional[KibanaConfig] = None):
        """
        初始化ELK日志管理系统

        Args:
            elasticsearch_config: Elasticsearch配置
            kibana_config: Kibana配置
        """
        self.elasticsearch_config = elasticsearch_config
        self.kibana_config = kibana_config

        # 初始化组件
        self.elasticsearch_client = ElasticsearchClient(elasticsearch_config)
        self.log_aggregator = LogAggregator(elasticsearch_config)
        self.kibana_manager = KibanaManager(kibana_config) if kibana_config else None

        # 结构化日志记录器集合
        self.structured_loggers: Dict[str, StructuredLogger] = {}

        logger.info("ELK日志管理系统初始化完成")

    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=5.0)
    def setup_logging_infrastructure(self) -> Dict[str, Any]:
        """设置日志基础设施"""
        results = {}

        try:
            # 创建Elasticsearch索引模板
            template_created = self.elasticsearch_client.create_index_template()
            results['elasticsearch_template'] = template_created

            # 创建Kibana索引模式和仪表盘
            if self.kibana_manager:
                index_pattern = self.kibana_manager.create_index_pattern()
                results['kibana_index_pattern'] = index_pattern

                dashboard = self.kibana_manager.create_dashboard()
                results['kibana_dashboard'] = dashboard

            logger.info("日志基础设施设置完成")

        except Exception as e:
            logger.error(f"设置日志基础设施失败: {e}")
            results['error'] = str(e)

        return results

    def get_structured_logger(self, name: str, format_type: LogFormat = LogFormat.JSON,
                             enable_elasticsearch: bool = True) -> StructuredLogger:
        """获取结构化日志记录器"""
        if name not in self.structured_loggers:
            # 创建日志记录器
            structured_logger = StructuredLogger(
                name=name,
                format_type=format_type,
                output_file=f"logs/{name}.log"
            )

            # 如果启用Elasticsearch，添加处理器
            if enable_elasticsearch:
                structured_logger.add_handler(self.log_aggregator.add_log)

            self.structured_loggers[name] = structured_logger

        return self.structured_loggers[name]

    @exception_handler(reraise=True)
    def search_logs(self, level: Optional[str] = None, message: Optional[str] = None,
                   start_time: Optional[datetime] = None, end_time: Optional[datetime] = None,
                   limit: int = 100) -> List[Dict[str, Any]]:
        """搜索日志"""
        query = {"bool": {"must": []}}

        # 按级别过滤
        if level:
            query["bool"]["must"].append({"term": {"level": level.upper()}})

        # 按消息内容过滤
        if message:
            query["bool"]["must"].append({
                "match": {"message": {"query": message, "fuzziness": "AUTO"}}
            })

        # 按时间范围过滤
        if start_time or end_time:
            time_range = {}
            if start_time:
                time_range["gte"] = start_time.isoformat()
            if end_time:
                time_range["lte"] = end_time.isoformat()

            query["bool"]["must"].append({"range": {"@timestamp": time_range}})

        return self.elasticsearch_client.search_logs(query, limit)

    @exception_handler(reraise=True)
    def get_log_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """获取日志统计信息"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)

        return self.elasticsearch_client.get_log_statistics(start_time, end_time)

    @exception_handler(reraise=True)
    def create_log_alert(self, alert_name: str, conditions: Dict[str, Any]) -> Dict[str, Any]:
        """创建日志告警（模拟实现）"""
        # 这里应该集成实际的告警系统，如ElastAlert
        alert_config = {
            "name": alert_name,
            "conditions": conditions,
            "created_at": datetime.now().isoformat(),
            "status": "active"
        }

        logger.info(f"创建日志告警: {alert_name}")
        return alert_config

    def export_logstash_config(self, config_dir: str = "config/logstash") -> str:
        """导出Logstash配置"""
        config_path = Path(config_dir)
        config_path.mkdir(parents=True, exist_ok=True)

        logstash_config = """
input {
  file {
    path => "/app/logs/*.log"
    start_position => "beginning"
    codec => "json"
    type => "application_logs"
  }
  beats {
    port => 5044
  }
}

filter {
  if [type] == "application_logs" {
    # 解析时间戳
    date {
      match => [ "timestamp", "ISO8601" ]
    }

    # 添加字段
    mutate {
      add_field => { "log_source" => "stock_analysis_system" }
    }

    # 解析异常信息
    if [extra_fields][exception] {
      mutate {
        add_field => { "has_exception" => true }
      }
    }
  }
}

output {
  elasticsearch {
    hosts => ["localhost:9200"]
    index => "stock-logs-%{+YYYY.MM.dd}"
  }
  stdout {
    codec => rubydebug
  }
}
"""

        config_file = config_path / "logstash.conf"
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(logstash_config.strip())

        logger.info(f"Logstash配置已导出到: {config_file}")
        return str(config_file)

    def export_docker_compose(self, config_dir: str = "config/elk") -> str:
        """导出ELK Docker Compose配置"""
        config_path = Path(config_dir)
        config_path.mkdir(parents=True, exist_ok=True)

        docker_compose = {
            "version": "3.8",
            "services": {
                "elasticsearch": {
                    "image": "elasticsearch:8.5.0",
                    "container_name": "elasticsearch",
                    "environment": [
                        "discovery.type=single-node",
                        "ES_JAVA_OPTS=-Xms512m -Xmx512m",
                        "xpack.security.enabled=false"
                    ],
                    "ports": ["9200:9200", "9300:9300"],
                    "volumes": ["elasticsearch-data:/usr/share/elasticsearch/data"],
                    "networks": ["elk"]
                },
                "logstash": {
                    "image": "logstash:8.5.0",
                    "container_name": "logstash",
                    "volumes": [
                        "./logstash/logstash.conf:/usr/share/logstash/pipeline/logstash.conf",
                        "../logs:/app/logs"
                    ],
                    "ports": ["5044:5044", "9600:9600"],
                    "depends_on": ["elasticsearch"],
                    "networks": ["elk"]
                },
                "kibana": {
                    "image": "kibana:8.5.0",
                    "container_name": "kibana",
                    "environment": [
                        "ELASTICSEARCH_HOSTS=http://elasticsearch:9200"
                    ],
                    "ports": ["5601:5601"],
                    "depends_on": ["elasticsearch"],
                    "networks": ["elk"]
                }
            },
            "volumes": {
                "elasticsearch-data": {}
            },
            "networks": {
                "elk": {"driver": "bridge"}
            }
        }

        compose_file = config_path / "docker-compose.yml"
        with open(compose_file, 'w', encoding='utf-8') as f:
            yaml.dump(docker_compose, f, default_flow_style=False)

        logger.info(f"ELK Docker Compose配置已导出到: {compose_file}")
        return str(compose_file)

    def close(self):
        """关闭日志管理系统"""
        # 关闭所有结构化日志记录器
        for structured_logger in self.structured_loggers.values():
            structured_logger.close()

        # 关闭日志聚合器
        if self.log_aggregator:
            self.log_aggregator.close()

        logger.info("ELK日志管理系统已关闭")


# 全局日志管理器实例
_elk_log_manager = None


def get_elk_log_manager(elasticsearch_hosts: List[str] = None,
                       kibana_url: str = "http://localhost:5601") -> ELKLogManager:
    """
    获取ELK日志管理器实例（单例模式）

    Args:
        elasticsearch_hosts: Elasticsearch主机列表
        kibana_url: Kibana URL

    Returns:
        ELKLogManager: ELK日志管理器实例
    """
    global _elk_log_manager

    if _elk_log_manager is None:
        elasticsearch_config = ElasticsearchConfig(hosts=elasticsearch_hosts)
        kibana_config = KibanaConfig(url=kibana_url)
        _elk_log_manager = ELKLogManager(elasticsearch_config, kibana_config)

    return _elk_log_manager


def create_elk_log_manager(elasticsearch_hosts: List[str] = None,
                          kibana_url: str = "http://localhost:5601") -> ELKLogManager:
    """
    创建新的ELK日志管理器实例

    Args:
        elasticsearch_hosts: Elasticsearch主机列表
        kibana_url: Kibana URL

    Returns:
        ELKLogManager: 新的ELK日志管理器实例
    """
    elasticsearch_config = ElasticsearchConfig(hosts=elasticsearch_hosts)
    kibana_config = KibanaConfig(url=kibana_url)
    return ELKLogManager(elasticsearch_config, kibana_config)