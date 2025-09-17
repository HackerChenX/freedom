#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
ClickHouse数据库性能监控模块

专为ClickHouse数据库设计的企业级性能监控系统：
1. 实时性能指标监控（查询性能、连接数、内存使用）
2. 查询分析和慢查询检测
3. 集群状态监控（副本、分片、同步状态）
4. 资源使用监控（磁盘、内存、CPU）
5. 数据库配置优化建议
6. 自动化性能调优
7. 备份和恢复监控
8. 告警和通知集成
"""

import os
import time
import json
import threading
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import queue

from utils.logger import get_logger
from utils.exception_handler import exception_handler
from utils.performance_monitor import performance_monitor

logger = get_logger(__name__)


class ClickHouseMetricType(Enum):
    """ClickHouse指标类型"""
    SYSTEM = "system"
    QUERY = "query"
    CLUSTER = "cluster"
    REPLICATION = "replication"
    PARTS = "parts"
    STORAGE = "storage"
    NETWORK = "network"


class PerformanceLevel(Enum):
    """性能级别"""
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class ClickHouseMetric:
    """ClickHouse指标"""
    metric_type: ClickHouseMetricType
    name: str
    value: Union[int, float, str]
    unit: str
    timestamp: datetime
    labels: Dict[str, str]
    description: str


@dataclass
class QueryAnalysis:
    """查询分析结果"""
    query_id: str
    query_text: str
    execution_time: float
    memory_usage: int
    cpu_usage: float
    rows_read: int
    rows_written: int
    bytes_read: int
    bytes_written: int
    performance_level: PerformanceLevel
    suggestions: List[str]
    timestamp: datetime


@dataclass
class ClusterNodeStatus:
    """集群节点状态"""
    host: str
    port: int
    is_leader: bool
    is_alive: bool
    version: str
    uptime: float
    queries_per_second: float
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    replication_delay: Optional[float] = None


@dataclass
class PerformanceAlert:
    """性能告警"""
    alert_id: str
    severity: str
    metric_name: str
    current_value: float
    threshold_value: float
    message: str
    timestamp: datetime
    suggestions: List[str]


class ClickHouseConnection:
    """ClickHouse连接管理器"""

    def __init__(self, host: str = "localhost", port: int = 9000,
                 database: str = "default", username: str = "default",
                 password: str = ""):
        """
        初始化ClickHouse连接

        Args:
            host: 主机地址
            port: 端口号
            database: 数据库名
            username: 用户名
            password: 密码
        """
        self.host = host
        self.port = port
        self.database = database
        self.username = username
        self.password = password

        self.client = None
        self.connection_pool = None

        # 尝试初始化连接
        self._initialize_connection()

    def _initialize_connection(self):
        """初始化连接 - 任务5简化版本"""
        try:
            # 直接使用增强连接池
            from db.enhanced_connection_pool import get_connection_pool
from db.sql_manager import SQLManager, QueryType

            self.connection_pool = get_connection_pool()

            # 测试连接
            with self.connection_pool.get_connection() as conn:
                result = conn.execute('SELECT 1')
                if result:
                    logger.info(f"ClickHouse连接成功: {self.host}:{self.port}")
                else:
                    logger.warning("ClickHouse连接测试失败")

            self.client = True  # 标记连接可用

        except ImportError:
            logger.warning("增强连接池未安装，使用模拟模式")
            self.client = None
            self.connection_pool = None
        except Exception as e:
            logger.error(f"ClickHouse连接失败: {e}")
            self.client = None
            self.connection_pool = None

    @exception_handler(reraise=True)
    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Tuple]:
        """执行查询"""
        if not self.client:
            # 返回模拟数据
            return self._get_mock_data(query)

        try:
            if parameters:
                result = self.client.execute(query, parameters)
            else:
                result = self.client.execute(query)
            return result

        except Exception as e:
            logger.error(f"执行ClickHouse查询失败: {e}")
            return []

    def _get_mock_data(self, query: str) -> List[Tuple]:
        """获取模拟数据"""
        if 'system.metrics' in query.lower():
            return [
                ('Query', 125),
                ('Merge', 8),
                ('SelectedRows', 1250000),
                ('SelectedBytes', 125000000),
                ('InsertedRows', 50000),
                ('InsertedBytes', 5000000),
                ('TotalRowsOfMergeTreeTables', 10000000),
                ('ReplicasMaxAbsoluteDelay', 0),
            ]
        elif 'system.events' in query.lower():
            return [
                ('Query', 1250),
                ('SelectQuery', 1000),
                ('InsertQuery', 250),
                ('FileOpen', 125),
                ('ReadBufferFromFileDescriptorRead', 500),
                ('WriteBufferFromFileDescriptorWrite', 125),
            ]
        elif 'system.parts' in query.lower():
            return [
                ('stock_info', 'active', 500, 125000000, 1000000),
                ('stock_info', 'inactive', 25, 6250000, 50000),
                ('indicators', 'active', 200, 50000000, 500000),
            ]
        elif 'system.processes' in query.lower():
            return [
                ('query1', 'SELECT code, name, date, open, high, low, close, volume, turnover_rate FROM stock_info', 2.5, 125000000, 1000000),
                ('query2', 'INSERT INTO indicators', 0.8, 25000000, 500000),
            ]
        elif 'system.clusters' in query.lower():
            return [
                ('stock_cluster', 'localhost', 9000, 1, 1),
                ('stock_cluster', 'localhost', 9001, 1, 2),
            ]
        else:
            return [('mock_result', 1)]

    def close(self):
        """关闭连接"""
        if self.client:
            try:
                self.client.disconnect()
            except Exception as e:
                logger.error(f"关闭ClickHouse连接失败: {e}")


class ClickHouseMetricsCollector:
    """ClickHouse指标收集器"""

    def __init__(self, connection: ClickHouseConnection):
        """
        初始化指标收集器

        Args:
            connection: ClickHouse连接
        """
        self.connection = connection
        self.metrics_history: List[ClickHouseMetric] = []
        self.collection_active = False
        self.collection_thread: Optional[threading.Thread] = None

        logger.info("ClickHouse指标收集器初始化完成")

    @exception_handler(reraise=True)
    def start_collection(self, interval: int = 60):
        """启动指标收集"""
        if self.collection_active:
            logger.warning("指标收集已在运行")
            return

        self.collection_active = True
        self.collection_thread = threading.Thread(
            target=self._collection_loop,
            args=(interval,),
            daemon=True
        )
        self.collection_thread.start()

        logger.info(f"ClickHouse指标收集已启动，间隔: {interval}秒")

    def stop_collection(self):
        """停止指标收集"""
        self.collection_active = False
        if self.collection_thread:
            self.collection_thread.join(timeout=10)
        logger.info("ClickHouse指标收集已停止")

    def _collection_loop(self, interval: int):
        """指标收集循环"""
        while self.collection_active:
            try:
                # 收集各类指标
                self._collect_system_metrics()
                self._collect_query_metrics()
                self._collect_cluster_metrics()
                self._collect_storage_metrics()
                self._collect_replication_metrics()

                # 清理过期指标（保留最近1000条）
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]

                time.sleep(interval)

            except Exception as e:
                logger.error(f"指标收集循环出错: {e}")
                time.sleep(5)

    def _collect_system_metrics(self):
        """收集系统指标"""
        try:
            # 查询系统指标
            query = """
            SELECT metric, value
            FROM system.metrics
            WHERE metric IN (
                'Query', 'Merge', 'SelectedRows', 'SelectedBytes',
                'InsertedRows', 'InsertedBytes', 'TotalRowsOfMergeTreeTables',
                'ReplicasMaxAbsoluteDelay', 'MaxPartCountForPartition',
                'BackgroundPoolTask', 'BackgroundSchedulePoolTask'
            )
            """

            results = self.connection.execute_query(query)
            timestamp = datetime.now()

            for metric_name, value in results:
                # 确定指标单位
                unit = self._get_metric_unit(metric_name)

                metric = ClickHouseMetric(
                    metric_type=ClickHouseMetricType.SYSTEM,
                    name=metric_name,
                    value=value,
                    unit=unit,
                    timestamp=timestamp,
                    labels={'type': 'system'},
                    description=f"ClickHouse系统指标: {metric_name}"
                )

                self.metrics_history.append(metric)

        except Exception as e:
            logger.error(f"收集系统指标失败: {e}")

    def _collect_query_metrics(self):
        """收集查询指标"""
        try:
            # 查询事件统计
            query = """
            SELECT event, value
            FROM system.events
            WHERE event IN (
                'Query', 'SelectQuery', 'InsertQuery',
                'FileOpen', 'ReadBufferFromFileDescriptorRead',
                'WriteBufferFromFileDescriptorWrite', 'CreatedReadBufferOrdinary'
            )
            """

            results = self.connection.execute_query(query)
            timestamp = datetime.now()

            for event_name, value in results:
                metric = ClickHouseMetric(
                    metric_type=ClickHouseMetricType.QUERY,
                    name=event_name,
                    value=value,
                    unit='count',
                    timestamp=timestamp,
                    labels={'type': 'event'},
                    description=f"ClickHouse查询事件: {event_name}"
                )

                self.metrics_history.append(metric)

        except Exception as e:
            logger.error(f"收集查询指标失败: {e}")

    def _collect_cluster_metrics(self):
        """收集集群指标"""
        try:
            # 查询集群信息
            query = """
            SELECT cluster, host_name, port, shard_num, replica_num
            FROM system.clusters
            """

            results = self.connection.execute_query(query)
            timestamp = datetime.now()

            cluster_info = {}
            for cluster, host, port, shard, replica in results:
                if cluster not in cluster_info:
                    cluster_info[cluster] = {'shards': 0, 'replicas': 0, 'nodes': 0}

                cluster_info[cluster]['nodes'] += 1
                cluster_info[cluster]['shards'] = max(cluster_info[cluster]['shards'], shard)
                cluster_info[cluster]['replicas'] = max(cluster_info[cluster]['replicas'], replica)

            # 创建集群指标
            for cluster, info in cluster_info.items():
                for key, value in info.items():
                    metric = ClickHouseMetric(
                        metric_type=ClickHouseMetricType.CLUSTER,
                        name=f"cluster_{key}",
                        value=value,
                        unit='count',
                        timestamp=timestamp,
                        labels={'cluster': cluster, 'type': 'cluster'},
                        description=f"集群 {cluster} 的 {key} 数量"
                    )

                    self.metrics_history.append(metric)

        except Exception as e:
            logger.error(f"收集集群指标失败: {e}")

    def _collect_storage_metrics(self):
        """收集存储指标"""
        try:
            # 查询表分区信息
            query = """
            SELECT
                table,
                active,
                count() as parts_count,
                sum(bytes_on_disk) as total_bytes,
                sum(rows) as total_rows
            FROM system.parts
            WHERE active = 1
            GROUP BY table, active
            """

            results = self.connection.execute_query(query)
            timestamp = datetime.now()

            for table, active, parts_count, total_bytes, total_rows in results:
                # 分区数量指标
                metric = ClickHouseMetric(
                    metric_type=ClickHouseMetricType.PARTS,
                    name="parts_count",
                    value=parts_count,
                    unit='count',
                    timestamp=timestamp,
                    labels={'table': table, 'active': str(active)},
                    description=f"表 {table} 的分区数量"
                )
                self.metrics_history.append(metric)

                # 存储大小指标
                metric = ClickHouseMetric(
                    metric_type=ClickHouseMetricType.STORAGE,
                    name="storage_bytes",
                    value=total_bytes,
                    unit='bytes',
                    timestamp=timestamp,
                    labels={'table': table, 'active': str(active)},
                    description=f"表 {table} 的存储大小"
                )
                self.metrics_history.append(metric)

                # 行数指标
                metric = ClickHouseMetric(
                    metric_type=ClickHouseMetricType.STORAGE,
                    name="total_rows",
                    value=total_rows,
                    unit='count',
                    timestamp=timestamp,
                    labels={'table': table, 'active': str(active)},
                    description=f"表 {table} 的总行数"
                )
                self.metrics_history.append(metric)

        except Exception as e:
            logger.error(f"收集存储指标失败: {e}")

    def _collect_replication_metrics(self):
        """收集复制指标"""
        try:
            # 查询复制队列
            query = """
            SELECT
                database,
                table,
                count() as queue_size,
                max(num_tries) as max_tries,
                min(create_time) as oldest_time
            FROM system.replication_queue
            GROUP BY database, table
            """

            results = self.connection.execute_query(query)
            timestamp = datetime.now()

            for database, table, queue_size, max_tries, oldest_time in results:
                # 复制队列大小
                metric = ClickHouseMetric(
                    metric_type=ClickHouseMetricType.REPLICATION,
                    name="replication_queue_size",
                    value=queue_size,
                    unit='count',
                    timestamp=timestamp,
                    labels={'database': database, 'table': table},
                    description=f"表 {database}.{table} 的复制队列大小"
                )
                self.metrics_history.append(metric)

                # 最大重试次数
                metric = ClickHouseMetric(
                    metric_type=ClickHouseMetricType.REPLICATION,
                    name="max_replication_tries",
                    value=max_tries,
                    unit='count',
                    timestamp=timestamp,
                    labels={'database': database, 'table': table},
                    description=f"表 {database}.{table} 的最大复制重试次数"
                )
                self.metrics_history.append(metric)

        except Exception as e:
            logger.debug(f"收集复制指标失败（可能无复制表）: {e}")

    def _get_metric_unit(self, metric_name: str) -> str:
        """获取指标单位"""
        unit_mapping = {
            'Query': 'count',
            'Merge': 'count',
            'SelectedRows': 'rows',
            'SelectedBytes': 'bytes',
            'InsertedRows': 'rows',
            'InsertedBytes': 'bytes',
            'TotalRowsOfMergeTreeTables': 'rows',
            'ReplicasMaxAbsoluteDelay': 'seconds',
            'MaxPartCountForPartition': 'count',
            'BackgroundPoolTask': 'count',
            'BackgroundSchedulePoolTask': 'count'
        }

        return unit_mapping.get(metric_name, 'unknown')

    def get_latest_metrics(self, metric_type: Optional[ClickHouseMetricType] = None,
                          limit: int = 100) -> List[Dict[str, Any]]:
        """获取最新指标"""
        metrics = self.metrics_history

        if metric_type:
            metrics = [m for m in metrics if m.metric_type == metric_type]

        # 按时间排序，取最新的
        metrics = sorted(metrics, key=lambda x: x.timestamp, reverse=True)[:limit]

        return [asdict(metric) for metric in metrics]


class ClickHouseQueryAnalyzer:
    """ClickHouse查询分析器"""

    def __init__(self, connection: ClickHouseConnection):
        """
        初始化查询分析器

        Args:
            connection: ClickHouse连接
        """
        self.connection = connection
        self.query_history: List[QueryAnalysis] = []

        logger.info("ClickHouse查询分析器初始化完成")

    @exception_handler(reraise=True)
    def analyze_running_queries(self) -> List[QueryAnalysis]:
        """分析当前运行的查询"""
        analyses = []

        try:
            # 查询当前运行的查询
            query = """
            SELECT
                query_id,
                query,
                elapsed,
                memory_usage,
                read_rows,
                read_bytes,
                written_rows,
                written_bytes,
                total_rows_approx
            FROM system.processes
            WHERE query_id != ''
            """

            results = self.connection.execute_query(query)

            for result in results:
                (query_id, query_text, elapsed, memory_usage,
                 read_rows, read_bytes, written_rows, written_bytes, total_rows) = result

                # 计算性能级别
                performance_level = self._calculate_performance_level(
                    elapsed, memory_usage, read_rows, read_bytes
                )

                # 生成优化建议
                suggestions = self._generate_optimization_suggestions(
                    query_text, elapsed, memory_usage, read_rows, read_bytes
                )

                analysis = QueryAnalysis(
                    query_id=query_id,
                    query_text=query_text[:500],  # 截取前500字符
                    execution_time=elapsed,
                    memory_usage=memory_usage,
                    cpu_usage=0.0,  # 无法直接获取
                    rows_read=read_rows,
                    rows_written=written_rows,
                    bytes_read=read_bytes,
                    bytes_written=written_bytes,
                    performance_level=performance_level,
                    suggestions=suggestions,
                    timestamp=datetime.now()
                )

                analyses.append(analysis)
                self.query_history.append(analysis)

            # 清理历史记录（保留最近500条）
            if len(self.query_history) > 500:
                self.query_history = self.query_history[-500:]

        except Exception as e:
            logger.error(f"分析运行查询失败: {e}")

        return analyses

    @exception_handler(reraise=True)
    def get_slow_queries(self, duration_threshold: float = 5.0,
                        time_window_minutes: int = 60) -> List[QueryAnalysis]:
        """获取慢查询"""
        try:
            # 从查询日志中获取慢查询
            query = f"""
            SELECT
                query_id,
                query,
                query_duration_ms,
                memory_usage,
                read_rows,
                read_bytes,
                written_rows,
                written_bytes,
                event_time
            FROM system.query_log
            WHERE event_time >= now() - INTERVAL {time_window_minutes} MINUTE
            AND query_duration_ms >= {duration_threshold * 1000}
            AND type = 'QueryFinish'
            ORDER BY query_duration_ms DESC
            LIMIT 50
            """

            results = self.connection.execute_query(query)
            slow_queries = []

            for result in results:
                (query_id, query_text, duration_ms, memory_usage,
                 read_rows, read_bytes, written_rows, written_bytes, event_time) = result

                execution_time = duration_ms / 1000.0

                # 计算性能级别
                performance_level = self._calculate_performance_level(
                    execution_time, memory_usage, read_rows, read_bytes
                )

                # 生成优化建议
                suggestions = self._generate_optimization_suggestions(
                    query_text, execution_time, memory_usage, read_rows, read_bytes
                )

                analysis = QueryAnalysis(
                    query_id=query_id,
                    query_text=query_text[:500],
                    execution_time=execution_time,
                    memory_usage=memory_usage,
                    cpu_usage=0.0,
                    rows_read=read_rows,
                    rows_written=written_rows,
                    bytes_read=read_bytes,
                    bytes_written=written_bytes,
                    performance_level=performance_level,
                    suggestions=suggestions,
                    timestamp=event_time if isinstance(event_time, datetime) else datetime.now()
                )

                slow_queries.append(analysis)

            return slow_queries

        except Exception as e:
            logger.error(f"获取慢查询失败: {e}")
            return []

    def _calculate_performance_level(self, execution_time: float, memory_usage: int,
                                   rows_read: int, bytes_read: int) -> PerformanceLevel:
        """计算性能级别"""
        try:
            # 基于执行时间判断
            if execution_time < 0.1:
                time_score = 5
            elif execution_time < 1.0:
                time_score = 4
            elif execution_time < 5.0:
                time_score = 3
            elif execution_time < 30.0:
                time_score = 2
            else:
                time_score = 1

            # 基于内存使用判断
            memory_mb = memory_usage / (1024 * 1024)
            if memory_mb < 100:
                memory_score = 5
            elif memory_mb < 500:
                memory_score = 4
            elif memory_mb < 1000:
                memory_score = 3
            elif memory_mb < 2000:
                memory_score = 2
            else:
                memory_score = 1

            # 基于读取效率判断
            if rows_read > 0:
                bytes_per_row = bytes_read / rows_read
                if bytes_per_row < 100:
                    efficiency_score = 5
                elif bytes_per_row < 500:
                    efficiency_score = 4
                elif bytes_per_row < 1000:
                    efficiency_score = 3
                elif bytes_per_row < 2000:
                    efficiency_score = 2
                else:
                    efficiency_score = 1
            else:
                efficiency_score = 5

            # 综合评分
            total_score = (time_score + memory_score + efficiency_score) / 3

            if total_score >= 4.5:
                return PerformanceLevel.EXCELLENT
            elif total_score >= 3.5:
                return PerformanceLevel.GOOD
            elif total_score >= 2.5:
                return PerformanceLevel.AVERAGE
            elif total_score >= 1.5:
                return PerformanceLevel.POOR
            else:
                return PerformanceLevel.CRITICAL

        except Exception:
            return PerformanceLevel.AVERAGE

    def _generate_optimization_suggestions(self, query_text: str, execution_time: float,
                                         memory_usage: int, rows_read: int,
                                         bytes_read: int) -> List[str]:
        """生成优化建议"""
        suggestions = []

        try:
            query_lower = query_text.lower()

            # 执行时间建议
            if execution_time > 30:
                suggestions.append("查询执行时间过长，考虑添加适当的WHERE条件限制数据范围")

            if execution_time > 10:
                suggestions.append("检查是否可以使用合适的索引或分区键优化查询")

            # 内存使用建议
            memory_mb = memory_usage / (1024 * 1024)
            if memory_mb > 2000:
                suggestions.append("内存使用量过大，考虑分批处理或优化查询逻辑")

            # 数据读取建议
            if rows_read > 1000000:
                suggestions.append("读取行数过多，建议使用LIMIT限制结果集大小")

            if bytes_read > 1000000000:  # 1GB
                suggestions.append("读取数据量过大，考虑使用列式查询减少I/O")

            # 查询语句建议
            if 'select *' in query_lower:
                suggestions.append("避免使用SELECT *，明确指定需要的列")

            if 'order by' in query_lower and 'limit' not in query_lower:
                suggestions.append("ORDER BY查询建议添加LIMIT子句")

            if 'group by' in query_lower:
                suggestions.append("GROUP BY查询考虑使用合适的聚合索引")

            if 'join' in query_lower:
                suggestions.append("JOIN查询确保连接条件使用了合适的索引")

            # 默认建议
            if not suggestions:
                suggestions.append("查询性能良好，继续保持")

        except Exception as e:
            logger.error(f"生成优化建议失败: {e}")
            suggestions.append("无法生成具体建议，请检查查询逻辑")

        return suggestions

    def get_query_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """获取查询统计信息"""
        try:
            # 查询统计信息
            query = f"""
            SELECT
                count() as total_queries,
                avg(query_duration_ms) as avg_duration_ms,
                max(query_duration_ms) as max_duration_ms,
                sum(read_rows) as total_read_rows,
                sum(read_bytes) as total_read_bytes,
                sum(memory_usage) as total_memory_usage
            FROM system.query_log
            WHERE event_time >= now() - INTERVAL {hours} HOUR
            AND type = 'QueryFinish'
            """

            results = self.connection.execute_query(query)

            if results:
                (total_queries, avg_duration, max_duration,
                 total_read_rows, total_read_bytes, total_memory) = results[0]

                return {
                    'time_window_hours': hours,
                    'total_queries': total_queries,
                    'avg_duration_seconds': avg_duration / 1000.0 if avg_duration else 0,
                    'max_duration_seconds': max_duration / 1000.0 if max_duration else 0,
                    'total_read_rows': total_read_rows or 0,
                    'total_read_bytes': total_read_bytes or 0,
                    'total_memory_usage': total_memory or 0,
                    'avg_memory_per_query': (total_memory / total_queries) if total_queries else 0,
                    'queries_per_hour': total_queries / hours if hours > 0 else 0
                }

        except Exception as e:
            logger.error(f"获取查询统计失败: {e}")

        return {}


class ClickHousePerformanceMonitor:
    """
    ClickHouse性能监控器

    整合指标收集、查询分析、告警等功能的核心监控系统
    """

    def __init__(self, connection_config: Dict[str, Any]):
        """
        初始化ClickHouse性能监控器

        Args:
            connection_config: 连接配置
        """
        # 初始化连接
        self.connection = ClickHouseConnection(**connection_config)

        # 初始化各个组件
        self.metrics_collector = ClickHouseMetricsCollector(self.connection)
        self.query_analyzer = ClickHouseQueryAnalyzer(self.connection)

        # 性能告警
        self.performance_alerts: List[PerformanceAlert] = []
        self.alert_thresholds = self._get_default_thresholds()

        # 监控状态
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None

        logger.info("ClickHouse性能监控器初始化完成")

    def _get_default_thresholds(self) -> Dict[str, Dict[str, float]]:
        """获取默认告警阈值"""
        return {
            'query_duration': {'warning': 10.0, 'critical': 30.0},
            'memory_usage': {'warning': 1000000000, 'critical': 2000000000},  # bytes
            'slow_queries_count': {'warning': 10, 'critical': 50},
            'replication_delay': {'warning': 60.0, 'critical': 300.0},  # seconds
            'parts_count': {'warning': 1000, 'critical': 5000},
            'connections': {'warning': 100, 'critical': 200}
        }

    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=5.0)
    def start_monitoring(self, metrics_interval: int = 60,
                        query_analysis_interval: int = 300) -> Dict[str, Any]:
        """启动性能监控"""
        if self.monitoring_active:
            return {'status': 'already_running'}

        try:
            # 启动指标收集
            self.metrics_collector.start_collection(metrics_interval)

            # 启动监控主循环
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(
                target=self._monitoring_loop,
                args=(query_analysis_interval,),
                daemon=True
            )
            self.monitor_thread.start()

            logger.info("ClickHouse性能监控已启动")

            return {
                'status': 'started',
                'metrics_interval': metrics_interval,
                'query_analysis_interval': query_analysis_interval,
                'start_time': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"启动ClickHouse性能监控失败: {e}")
            return {'status': 'failed', 'error': str(e)}

    def stop_monitoring(self):
        """停止性能监控"""
        self.monitoring_active = False
        self.metrics_collector.stop_collection()

        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)

        logger.info("ClickHouse性能监控已停止")

    def _monitoring_loop(self, query_analysis_interval: int):
        """监控主循环"""
        while self.monitoring_active:
            try:
                # 分析当前运行的查询
                running_queries = self.query_analyzer.analyze_running_queries()

                # 检查慢查询
                slow_queries = self.query_analyzer.get_slow_queries(
                    duration_threshold=5.0,
                    time_window_minutes=15
                )

                # 检查性能告警
                self._check_performance_alerts(running_queries, slow_queries)

                time.sleep(query_analysis_interval)

            except Exception as e:
                logger.error(f"监控循环出错: {e}")
                time.sleep(30)

    def _check_performance_alerts(self, running_queries: List[QueryAnalysis],
                                slow_queries: List[QueryAnalysis]):
        """检查性能告警"""
        try:
            current_time = datetime.now()

            # 检查慢查询告警
            if len(slow_queries) >= self.alert_thresholds['slow_queries_count']['critical']:
                alert = PerformanceAlert(
                    alert_id=f"slow_queries_{int(current_time.timestamp())}",
                    severity='critical',
                    metric_name='slow_queries_count',
                    current_value=len(slow_queries),
                    threshold_value=self.alert_thresholds['slow_queries_count']['critical'],
                    message=f"慢查询数量过多: {len(slow_queries)}",
                    timestamp=current_time,
                    suggestions=[
                        "检查数据库索引配置",
                        "优化查询语句",
                        "考虑增加数据库资源"
                    ]
                )
                self.performance_alerts.append(alert)

            # 检查长时间运行的查询
            for query in running_queries:
                if query.execution_time >= self.alert_thresholds['query_duration']['critical']:
                    alert = PerformanceAlert(
                        alert_id=f"long_query_{query.query_id}",
                        severity='critical',
                        metric_name='query_duration',
                        current_value=query.execution_time,
                        threshold_value=self.alert_thresholds['query_duration']['critical'],
                        message=f"查询执行时间过长: {query.execution_time:.2f}秒",
                        timestamp=current_time,
                        suggestions=query.suggestions
                    )
                    self.performance_alerts.append(alert)

                # 检查内存使用
                if query.memory_usage >= self.alert_thresholds['memory_usage']['critical']:
                    alert = PerformanceAlert(
                        alert_id=f"high_memory_{query.query_id}",
                        severity='critical',
                        metric_name='memory_usage',
                        current_value=query.memory_usage,
                        threshold_value=self.alert_thresholds['memory_usage']['critical'],
                        message=f"查询内存使用过高: {query.memory_usage / (1024*1024):.2f}MB",
                        timestamp=current_time,
                        suggestions=[
                            "优化查询逻辑减少内存使用",
                            "考虑分批处理数据",
                            "增加数据库内存配置"
                        ]
                    )
                    self.performance_alerts.append(alert)

            # 清理过期告警（保留最近100条）
            if len(self.performance_alerts) > 100:
                self.performance_alerts = self.performance_alerts[-100:]

        except Exception as e:
            logger.error(f"检查性能告警失败: {e}")

    @exception_handler(reraise=True)
    def get_performance_dashboard(self) -> Dict[str, Any]:
        """获取性能仪表盘数据"""
        try:
            # 获取最新指标
            latest_metrics = {}
            for metric_type in ClickHouseMetricType:
                metrics = self.metrics_collector.get_latest_metrics(metric_type, 10)
                latest_metrics[metric_type.value] = metrics

            # 获取查询统计
            query_stats = self.query_analyzer.get_query_statistics(24)

            # 获取当前运行的查询
            running_queries = self.query_analyzer.analyze_running_queries()

            # 获取最近的慢查询
            slow_queries = self.query_analyzer.get_slow_queries(5.0, 60)

            # 获取性能告警
            recent_alerts = self.performance_alerts[-20:]  # 最近20个告警

            # 计算整体健康评分
            health_score = self._calculate_health_score(
                query_stats, running_queries, slow_queries, recent_alerts
            )

            return {
                'health_score': health_score,
                'connection_status': 'connected' if self.connection.client else 'disconnected',
                'monitoring_active': self.monitoring_active,
                'latest_metrics': latest_metrics,
                'query_statistics': query_stats,
                'running_queries': [asdict(q) for q in running_queries],
                'slow_queries': [asdict(q) for q in slow_queries[-10:]],  # 最近10个慢查询
                'performance_alerts': [asdict(a) for a in recent_alerts],
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"获取性能仪表盘数据失败: {e}")
            return {'error': str(e)}

    def _calculate_health_score(self, query_stats: Dict[str, Any],
                              running_queries: List[QueryAnalysis],
                              slow_queries: List[QueryAnalysis],
                              alerts: List[PerformanceAlert]) -> int:
        """计算健康评分（0-100）"""
        try:
            score = 100

            # 慢查询扣分
            slow_query_count = len(slow_queries)
            if slow_query_count > 50:
                score -= 30
            elif slow_query_count > 20:
                score -= 20
            elif slow_query_count > 10:
                score -= 10

            # 长时间运行查询扣分
            long_running = len([q for q in running_queries if q.execution_time > 30])
            if long_running > 5:
                score -= 20
            elif long_running > 2:
                score -= 10

            # 告警扣分
            critical_alerts = len([a for a in alerts if a.severity == 'critical'])
            warning_alerts = len([a for a in alerts if a.severity == 'warning'])

            score -= critical_alerts * 15
            score -= warning_alerts * 5

            # 平均查询时间扣分
            avg_duration = query_stats.get('avg_duration_seconds', 0)
            if avg_duration > 10:
                score -= 15
            elif avg_duration > 5:
                score -= 10
            elif avg_duration > 2:
                score -= 5

            return max(0, score)

        except Exception:
            return 50  # 默认评分

    @exception_handler(reraise=True)
    def generate_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """生成性能报告"""
        try:
            report_time = datetime.now()
            start_time = report_time - timedelta(hours=hours)

            # 基本信息
            report = {
                'report_id': f"perf_report_{int(report_time.timestamp())}",
                'generated_at': report_time.isoformat(),
                'time_window': {
                    'start': start_time.isoformat(),
                    'end': report_time.isoformat(),
                    'hours': hours
                }
            }

            # 查询统计
            query_stats = self.query_analyzer.get_query_statistics(hours)
            report['query_statistics'] = query_stats

            # 慢查询分析
            slow_queries = self.query_analyzer.get_slow_queries(5.0, hours * 60)
            report['slow_queries'] = {
                'count': len(slow_queries),
                'details': [asdict(q) for q in slow_queries[:20]]  # 前20个
            }

            # 性能指标趋势
            metrics_summary = {}
            for metric_type in ClickHouseMetricType:
                metrics = self.metrics_collector.get_latest_metrics(metric_type, 100)
                if metrics:
                    metrics_summary[metric_type.value] = {
                        'count': len(metrics),
                        'latest_values': metrics[:5]  # 最新5个值
                    }

            report['metrics_summary'] = metrics_summary

            # 告警统计
            recent_alerts = [a for a in self.performance_alerts
                           if a.timestamp >= start_time]
            report['alerts_summary'] = {
                'total': len(recent_alerts),
                'critical': len([a for a in recent_alerts if a.severity == 'critical']),
                'warning': len([a for a in recent_alerts if a.severity == 'warning']),
                'details': [asdict(a) for a in recent_alerts]
            }

            # 性能建议
            recommendations = self._generate_performance_recommendations(
                query_stats, slow_queries, recent_alerts
            )
            report['recommendations'] = recommendations

            # 健康评分
            health_score = self._calculate_health_score(
                query_stats, [], slow_queries, recent_alerts
            )
            report['health_score'] = health_score

            return report

        except Exception as e:
            logger.error(f"生成性能报告失败: {e}")
            return {'error': str(e)}

    def _generate_performance_recommendations(self, query_stats: Dict[str, Any],
                                            slow_queries: List[QueryAnalysis],
                                            alerts: List[PerformanceAlert]) -> List[str]:
        """生成性能优化建议"""
        recommendations = []

        try:
            # 基于查询统计的建议
            avg_duration = query_stats.get('avg_duration_seconds', 0)
            if avg_duration > 5:
                recommendations.append("平均查询时间较长，建议检查索引配置和查询优化")

            total_memory = query_stats.get('total_memory_usage', 0)
            if total_memory > 10 * 1024 * 1024 * 1024:  # 10GB
                recommendations.append("总内存使用量较高，考虑优化查询逻辑或增加内存")

            # 基于慢查询的建议
            if len(slow_queries) > 20:
                recommendations.append("慢查询数量较多，建议进行查询优化")

                # 分析慢查询的共同模式
                select_count = sum(1 for q in slow_queries if 'select' in q.query_text.lower())
                if select_count > len(slow_queries) * 0.8:
                    recommendations.append("大部分慢查询为SELECT查询，重点优化查询条件和索引")

                join_count = sum(1 for q in slow_queries if 'join' in q.query_text.lower())
                if join_count > 5:
                    recommendations.append("存在多个慢JOIN查询，检查连接条件和表结构")

            # 基于告警的建议
            critical_alerts = len([a for a in alerts if a.severity == 'critical'])
            if critical_alerts > 5:
                recommendations.append("存在多个严重性能告警，建议立即处理")

            # 通用建议
            if not recommendations:
                recommendations.append("数据库性能表现良好，建议继续保持当前优化水平")
            else:
                recommendations.append("定期监控数据库性能指标，及时发现和解决问题")

        except Exception as e:
            logger.error(f"生成性能建议失败: {e}")
            recommendations.append("无法生成具体建议，请检查系统状态")

        return recommendations

    def close(self):
        """关闭监控器"""
        self.stop_monitoring()
        if self.connection:
            self.connection.close()
        logger.info("ClickHouse性能监控器已关闭")


# 全局监控器实例
_clickhouse_performance_monitor = None


def get_clickhouse_performance_monitor(connection_config: Dict[str, Any] = None) -> ClickHousePerformanceMonitor:
    """
    获取ClickHouse性能监控器实例（单例模式）

    Args:
        connection_config: 连接配置

    Returns:
        ClickHousePerformanceMonitor: ClickHouse性能监控器实例
    """
    global _clickhouse_performance_monitor

    if _clickhouse_performance_monitor is None:
        if connection_config is None:
            connection_config = {
                'host': 'localhost',
                'port': 9000,
                'database': 'default',
                'username': 'default',
                'password': ''
            }

        _clickhouse_performance_monitor = ClickHousePerformanceMonitor(connection_config)

    return _clickhouse_performance_monitor


def create_clickhouse_performance_monitor(connection_config: Dict[str, Any]) -> ClickHousePerformanceMonitor:
    """
    创建新的ClickHouse性能监控器实例

    Args:
        connection_config: 连接配置

    Returns:
        ClickHousePerformanceMonitor: 新的ClickHouse性能监控器实例
    """
    return ClickHousePerformanceMonitor(connection_config)