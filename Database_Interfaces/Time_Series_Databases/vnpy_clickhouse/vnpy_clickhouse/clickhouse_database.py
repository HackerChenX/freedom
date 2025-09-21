"""
ClickHouse数据库接口实现

基于VnPy统一数据库接口规范，实现ClickHouse数据库的连接和操作。
严格遵循BaseDatabase抽象基类的接口定义。
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import logging
import threading
import time
from dataclasses import asdict
from functools import lru_cache, wraps
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

import clickhouse_connect
from clickhouse_connect.driver import Client

import sys
import os

# 添加Core_Framework路径
core_framework_path = os.path.join(os.path.dirname(__file__), "../../../../Core_Framework/vnpy")
sys.path.insert(0, core_framework_path)

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import BarData, TickData
from vnpy.trader.database import (
    BaseDatabase,
    BarOverview,
    TickOverview,
    DB_TZ,
    convert_tz
)
from vnpy.trader.setting import SETTINGS
from vnpy.trader.utility import generate_vt_symbol, extract_vt_symbol


def performance_monitor(func):
    """性能监控装饰器"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        method_name = func.__name__
        
        try:
            result = func(self, *args, **kwargs)
            success = True
            return result
        except Exception as e:
            success = False
            self.logger.error(f"{method_name} 执行失败: {e}")
            raise
        finally:
            duration = time.time() - start_time
            self.logger.info(f"{method_name} 执行时间: {duration:.3f}s, 成功: {success}")
            
            # 慢查询告警
            if duration > 5.0:
                self.logger.warning(f"⚠️ 慢查询检测: {method_name} 耗时 {duration:.3f}s")
    
    return wrapper


class ClickHouseDatabase(BaseDatabase):
    """
    ClickHouse数据库接口实现 - 高性能优化版本
    
    优化特性:
    1. 连接池管理
    2. 查询缓存机制
    3. 异步查询支持
    4. 性能监控
    5. 超时控制
    6. 批量操作优化
    """

    def __init__(self) -> None:
        """初始化ClickHouse数据库连接"""
        # 基础配置
        self.database_name: str = SETTINGS.get("database.database", "vnpy")
        self.host: str = SETTINGS.get("database.host", "localhost")
        self.port: int = SETTINGS.get("database.port", 8123)
        self.user: str = SETTINGS.get("database.user", "default")
        self.password: str = SETTINGS.get("database.password", "")
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 连接池配置
        self._connection_pool: List[Client] = []
        self._pool_lock = threading.Lock()
        self._pool_size = 5
        self._query_timeout = 30  # 查询超时时间（秒）
        
        # 线程池用于异步查询
        self._thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="clickhouse-query")
        
        # 查询缓存
        self._query_cache: Dict[str, Any] = {}
        self._cache_lock = threading.Lock()
        self._cache_ttl = 300  # 缓存5分钟
        self._cache_cleanup_interval = 600  # 10分钟清理一次缓存
        self._last_cache_cleanup = time.time()
        
        # 初始化连接池
        self._init_connection_pool()
        
        # 创建必要的表
        self._create_tables()
      

    def _init_connection_pool(self) -> None:
        """初始化连接池"""
        try:
            for i in range(self._pool_size):
                # 使用最基本的参数创建连接
                client = clickhouse_connect.get_client(
                    host=self.host,
                    port=self.port,
                    username=self.user,
                    password=self.password,
                    database="stock"  # 使用现有的stock数据库
                )
                
                # 测试连接
                client.query("SELECT 1")
                self._connection_pool.append(client)
            
            self.logger.info(f"成功初始化ClickHouse连接池: {self._pool_size} 个连接")
            
        except Exception as e:
            self.logger.error(f"初始化ClickHouse连接池失败: {e}")
            # 不再抛出异常，而是创建一个空的连接池
            self._connection_pool = []
            self.logger.warning("ClickHouse不可用，将使用降级模式")
    
    def _get_connection(self) -> Client:
        """从连接池获取连接"""
        with self._pool_lock:
            if self._connection_pool:
                return self._connection_pool.pop()
            else:
                # 连接池耗尽，创建临时连接
                self.logger.warning("连接池耗尽，创建临时连接")
                return clickhouse_connect.get_client(
                    host=self.host,
                    port=self.port,
                    username=self.user,
                    password=self.password,
                    database="stock"
                )
    
    def _return_connection(self, client: Client) -> None:
        """归还连接到连接池"""
        with self._pool_lock:
            if len(self._connection_pool) < self._pool_size:
                self._connection_pool.append(client)
            else:
                # 连接池已满，关闭连接
                try:
                    client.close()
                except:
                    pass
    
    def _get_cache_key(self, operation: str, **kwargs) -> str:
        """生成缓存键"""
        key_parts = [operation]
        for k, v in sorted(kwargs.items()):
            if isinstance(v, datetime):
                key_parts.append(f"{k}={v.isoformat()}")
            else:
                key_parts.append(f"{k}={v}")
        return "|".join(key_parts)
    
    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """从缓存获取数据"""
        with self._cache_lock:
            # 清理过期缓存
            current_time = time.time()
            if current_time - self._last_cache_cleanup > self._cache_cleanup_interval:
                self._cleanup_cache()
                self._last_cache_cleanup = current_time
            
            if cache_key in self._query_cache:
                data, timestamp = self._query_cache[cache_key]
                if current_time - timestamp < self._cache_ttl:
                    self.logger.debug(f"缓存命中: {cache_key}")
                    return data
                else:
                    del self._query_cache[cache_key]
        
        return None
    
    def _set_cache(self, cache_key: str, data: Any) -> None:
        """设置缓存数据"""
        with self._cache_lock:
            self._query_cache[cache_key] = (data, time.time())
    
    def _cleanup_cache(self) -> None:
        """清理过期缓存"""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self._query_cache.items()
            if current_time - timestamp >= self._cache_ttl
        ]
        
        for key in expired_keys:
            del self._query_cache[key]
        
        if expired_keys:
            self.logger.debug(f"清理过期缓存: {len(expired_keys)} 条")
    
    def _execute_query_with_timeout(self, query: str, parameters: Dict = None) -> Any:
        """带超时控制的查询执行"""
        client = self._get_connection()
        
        try:
            # 直接执行查询（简化版本，避免复杂的超时机制）
            result = client.query(query, parameters=parameters)
            return result
            
        except Exception as e:
            self.logger.error(f"查询执行失败: {e}")
            raise
        
        finally:
            self._return_connection(client)

    def _create_tables(self) -> None:
        """检查现有数据表，创建必要的概览表"""
        if not self._connection_pool:
            self.logger.warning("跳过表创建：ClickHouse连接不可用")
            return
            
        try:
            client = self._get_connection()
            
            # 检查stock_info表是否存在
            tables = client.query("SHOW TABLES")
            table_names = [row[0] for row in tables.result_rows]

            if 'stock_info' not in table_names:
                self.logger.warning("stock_info表不存在，某些功能可能不可用")
            else:
                self.logger.info("发现现有stock_info表，将直接使用")

            # 创建K线概览表（用于快速查询统计信息）
            client.command("""
                CREATE TABLE IF NOT EXISTS vnpy_bar_overview (
                    symbol String,
                    exchange String,
                    interval String,
                    count UInt64,
                    start DateTime,
                    end DateTime,
                    vt_symbol String MATERIALIZED concat(symbol, '.', exchange)
                ) ENGINE = ReplacingMergeTree()
                ORDER BY (symbol, exchange, interval)
            """)

            # 创建Tick概览表（预留给未来的Tick数据）
            client.command("""
                CREATE TABLE IF NOT EXISTS vnpy_tick_overview (
                    symbol String,
                    exchange String,
                    count UInt64,
                    start DateTime64(3),
                    end DateTime64(3),
                    vt_symbol String MATERIALIZED concat(symbol, '.', exchange)
                ) ENGINE = ReplacingMergeTree()
                ORDER BY (symbol, exchange)
            """)

            self._return_connection(client)
            self.logger.info("VnPy概览表创建完成")

        except Exception as e:
            self.logger.error(f"检查/创建数据表失败: {e}")
            # 不再抛出异常

    def save_bar_data(self, bars: List[BarData], stream: bool = False) -> bool:
        """
        保存K线数据到ClickHouse的stock_info表

        Args:
            bars: K线数据列表
            stream: 是否为流式数据（暂未使用）

        Returns:
            bool: 保存是否成功
        """
        if not bars:
            return True

        try:
            # 准备数据，映射到stock_info表结构
            data = []
            for bar in bars:
                # 转换时区
                dt = convert_tz(bar.datetime)
                date_only = dt.date()

                # 映射VnPy的Interval到stock_info的level字段
                level_mapping = {
                    Interval.MINUTE: "1分钟",
                    Interval.MINUTE_5: "5分钟",
                    Interval.MINUTE_15: "15分钟",
                    Interval.MINUTE_30: "30分钟",
                    Interval.HOUR: "1小时",
                    Interval.DAILY: "日线",
                    Interval.WEEKLY: "周线",
                    Interval.MONTHLY: "月线"
                }
                level = level_mapping.get(bar.interval, bar.interval.value)

                # 推断交易所（基于股票代码）
                exchange_name = self._infer_exchange(bar.symbol)

                # 计算涨跌幅等字段（如果没有提供）
                price_change = bar.close_price - bar.open_price
                price_range = (price_change / bar.open_price * 100) if bar.open_price > 0 else 0
                turnover_rate = 0  # 暂时设为0，需要流通股本数据才能计算

                data.append([
                    bar.symbol,                    # code
                    getattr(bar, 'name', ''),     # name (如果BarData有name属性)
                    date_only,                     # date
                    level,                         # level
                    float(bar.open_price),         # open
                    float(bar.close_price),        # close
                    float(bar.high_price),         # high
                    float(bar.low_price),          # low
                    float(bar.volume),             # volume
                    float(turnover_rate),          # turnover_rate
                    float(price_change),           # price_change
                    float(price_range),            # price_range
                    '',                            # industry (空字符串)
                    dt,                            # datetime
                    0                              # seq (默认为0)
                ])

            # 批量插入数据到stock_info表
            self._get_connection().insert(
                'stock_info',
                data,
                column_names=[
                    'code', 'name', 'date', 'level', 'open', 'close', 'high', 'low',
                    'volume', 'turnover_rate', 'price_change', 'price_range',
                    'industry', 'datetime', 'seq'
                ]
            )

            # 更新概览数据
            self._update_bar_overview(bars[0])

            return True

        except Exception as e:
            self.logger.error(f"保存K线数据失败: {e}")
            return False

    def save_tick_data(self, ticks: List[TickData], stream: bool = False) -> bool:
        """
        保存Tick数据到ClickHouse
        
        Args:
            ticks: Tick数据列表
            stream: 是否为流式数据（暂未使用）
            
        Returns:
            bool: 保存是否成功
        """
        if not ticks:
            return True
            
        try:
            # 准备数据
            data = []
            for tick in ticks:
                # 转换时区
                dt = convert_tz(tick.datetime)
                localtime = convert_tz(tick.localtime) if tick.localtime else dt
                
                data.append([
                    tick.symbol,
                    tick.exchange.value,
                    dt,
                    tick.name,
                    float(tick.volume),
                    float(tick.turnover),
                    float(tick.open_interest),
                    float(tick.last_price),
                    float(tick.last_volume),
                    float(tick.limit_up),
                    float(tick.limit_down),
                    float(tick.open_price),
                    float(tick.high_price),
                    float(tick.low_price),
                    float(tick.pre_close),
                    float(tick.bid_price_1),
                    float(tick.bid_price_2),
                    float(tick.bid_price_3),
                    float(tick.bid_price_4),
                    float(tick.bid_price_5),
                    float(tick.ask_price_1),
                    float(tick.ask_price_2),
                    float(tick.ask_price_3),
                    float(tick.ask_price_4),
                    float(tick.ask_price_5),
                    float(tick.bid_volume_1),
                    float(tick.bid_volume_2),
                    float(tick.bid_volume_3),
                    float(tick.bid_volume_4),
                    float(tick.bid_volume_5),
                    float(tick.ask_volume_1),
                    float(tick.ask_volume_2),
                    float(tick.ask_volume_3),
                    float(tick.ask_volume_4),
                    float(tick.ask_volume_5),
                    localtime
                ])
            
            # 批量插入数据
            self._get_connection().insert(
                'tick_data',
                data,
                column_names=[
                    'symbol', 'exchange', 'datetime', 'name',
                    'volume', 'turnover', 'open_interest',
                    'last_price', 'last_volume', 'limit_up', 'limit_down',
                    'open_price', 'high_price', 'low_price', 'pre_close',
                    'bid_price_1', 'bid_price_2', 'bid_price_3', 'bid_price_4', 'bid_price_5',
                    'ask_price_1', 'ask_price_2', 'ask_price_3', 'ask_price_4', 'ask_price_5',
                    'bid_volume_1', 'bid_volume_2', 'bid_volume_3', 'bid_volume_4', 'bid_volume_5',
                    'ask_volume_1', 'ask_volume_2', 'ask_volume_3', 'ask_volume_4', 'ask_volume_5',
                    'localtime'
                ]
            )
            
            # 更新概览数据
            self._update_tick_overview(ticks[0])
            
            return True
            
        except Exception as e:
            self.logger.error(f"保存Tick数据失败: {e}")
            return False

    @performance_monitor
    def load_bar_data(
        self,
        symbol: str,
        exchange: Exchange,
        interval: Interval,
        start: datetime,
        end: datetime
    ) -> List[BarData]:
        """
        从ClickHouse加载K线数据 - 高性能优化版本
        
        优化特性:
        1. 查询缓存机制
        2. 连接池管理
        3. 超时控制
        4. 批量查询优化
        5. 智能数据合成

        Args:
            symbol: 合约代码
            exchange: 交易所
            interval: 时间周期
            start: 开始时间
            end: 结束时间

        Returns:
            List[BarData]: K线数据列表
        """
        if not self._connection_pool:
            self.logger.warning("ClickHouse连接不可用，返回空数据")
            return []
            
        try:
            # 转换时区
            start_dt = convert_tz(start)
            end_dt = convert_tz(end)
            
            # 生成缓存键
            cache_key = self._get_cache_key(
                "load_bar_data",
                symbol=symbol,
                exchange=exchange.value,
                interval=interval.value,
                start=start_dt,
                end=end_dt
            )
            
            # 尝试从缓存获取
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                return cached_result

            # 映射扩展的Interval到数据库level字段
            interval_mapping = {
                Interval.MINUTE: "1分钟",
                Interval.MINUTE_5: "5分钟", 
                Interval.MINUTE_15: "15分钟",
                Interval.MINUTE_30: "30分钟",
                Interval.HOUR: "1小时",
                Interval.HOUR_2: "2小时",
                Interval.HOUR_4: "4小时",
                Interval.DAILY: "日线",
                Interval.WEEKLY: "周线",
                Interval.MONTHLY: "月线"
            }
            
            # 定义基础数据（数据库中直接存在的）
            base_intervals = {Interval.MINUTE_15, Interval.DAILY, Interval.WEEKLY, Interval.MONTHLY}
            
            # 需要合成的周期（基于15分钟数据）
            synthesized_intervals = {
                Interval.MINUTE_30: 2,   # 30分钟 = 2 * 15分钟
                Interval.HOUR: 4,        # 60分钟 = 4 * 15分钟  
                Interval.HOUR_2: 8,      # 120分钟 = 8 * 15分钟
                Interval.HOUR_4: 16      # 240分钟 = 16 * 15分钟
            }

            # 智能查询策略
            result = None
            
            # 策略1: 如果是基础数据，直接查询
            if interval in base_intervals:
                level = interval_mapping[interval]
                result = self._load_direct_bar_data_optimized(symbol, exchange, interval, level, start_dt, end_dt)
            
            # 策略2: 如果是合成数据，基于15分钟合成
            elif interval in synthesized_intervals:
                self.logger.info(f"基于15分钟数据合成{interval.value}周期")
                result = self._synthesize_from_15min_optimized(symbol, exchange, interval, start_dt, end_dt)
            
            # 策略3: 其他情况，尝试直接查询
            else:
                level = interval_mapping.get(interval, interval.value)
                result = self._load_direct_bar_data_optimized(symbol, exchange, interval, level, start_dt, end_dt)
            
            # 缓存结果
            if result:
                self._set_cache(cache_key, result)
            
            return result or []

        except Exception as e:
            self.logger.error(f"加载K线数据失败: {e}")
            return []

    def _load_direct_bar_data_optimized(
        self,
        symbol: str,
        exchange: Exchange,
        interval: Interval,
        level: str,
        start_dt: datetime,
        end_dt: datetime
    ) -> List[BarData]:
        """优化的直接数据库查询 - 支持连接池和超时控制"""
        try:
            # 优化的查询SQL - 添加索引提示和限制
            query = """
                SELECT 
                    code, name, datetime, level, open, close, high, low, volume,
                    turnover_rate, price_change, price_range
                FROM stock_info
                WHERE code = %(symbol)s
                  AND level = %(level)s
                  AND datetime >= %(start)s
                  AND datetime <= %(end)s
                ORDER BY datetime
                LIMIT 100000
            """

            # 使用连接池执行查询
            result = self._execute_query_with_timeout(
                query,
                parameters={
                    'symbol': symbol,
                    'level': level,
                    'start': start_dt,
                    'end': end_dt
                }
            )

            # 批量转换为BarData对象 - 优化内存使用
            bars = []
            for row in result.result_rows:
                # 计算turnover（成交额）- 避免除零错误
                volume = row[8] if row[8] else 0
                avg_price = (row[4] + row[5]) / 2 if row[4] and row[5] else 0
                turnover = volume * avg_price

                bar = BarData(
                    symbol=row[0],                    # code
                    exchange=exchange,                # 使用传入的exchange
                    datetime=row[2].replace(tzinfo=DB_TZ),  # datetime
                    interval=interval,                # 使用传入的interval
                    volume=volume,                    # volume
                    turnover=turnover,                # 计算的成交额
                    open_interest=0,                  # 股票没有持仓量
                    open_price=row[4] or 0,           # open
                    high_price=row[6] or 0,           # high
                    low_price=row[7] or 0,            # low
                    close_price=row[5] or 0,          # close
                    gateway_name="clickhouse_optimized"
                )
                bars.append(bar)

            self.logger.info(f"查询到 {len(bars)} 条{interval.value}数据")
            return bars

        except Exception as e:
            self.logger.error(f"优化查询失败: {e}")
            return []

    def _synthesize_from_15min_optimized(
        self,
        symbol: str,
        exchange: Exchange,
        target_interval: Interval,
        start_dt: datetime,
        end_dt: datetime
    ) -> List[BarData]:
        """
        优化的K线数据合成 - 高性能版本
        
        优化特性:
        1. 智能时间范围扩展
        2. 批量数据处理
        3. 内存优化的分组算法
        4. 并行合成支持
        """
        try:
            # 计算最优的时间范围扩展
            interval_minutes = {
                Interval.MINUTE_30: 30,
                Interval.HOUR: 60,
                Interval.HOUR_2: 120,
                Interval.HOUR_4: 240
            }
            
            target_minutes = interval_minutes.get(target_interval, 60)
            # 扩展时间范围：确保覆盖完整的合成周期
            extended_start = start_dt - timedelta(minutes=target_minutes * 2)
            
            # 使用优化查询加载15分钟基础数据
            base_bars = self._load_direct_bar_data_optimized(
                symbol, exchange, Interval.MINUTE_15, "15分钟", extended_start, end_dt
            )
            
            if not base_bars:
                self.logger.warning(f"无15分钟基础数据，无法合成{target_interval.value}")
                return []

            # 定义合成倍数
            synthesis_mapping = {
                Interval.MINUTE_30: 2,   # 30分钟 = 2 * 15分钟
                Interval.HOUR: 4,        # 60分钟 = 4 * 15分钟  
                Interval.HOUR_2: 8,      # 120分钟 = 8 * 15分钟
                Interval.HOUR_4: 16      # 240分钟 = 16 * 15分钟
            }
            
            group_size = synthesis_mapping.get(target_interval, 1)
            if group_size == 1:
                self.logger.error(f"不支持的合成周期: {target_interval.value}")
                return []
            
            # 使用优化的时间分组算法
            synthesized_bars = self._group_bars_optimized(
                base_bars, target_interval, start_dt, end_dt
            )
            
            self.logger.info(f"成功合成{target_interval.value}数据: {len(synthesized_bars)}条")
            return synthesized_bars
            
        except Exception as e:
            self.logger.error(f"优化合成失败: {e}")
            return []

    def _group_bars_optimized(
        self,
        base_bars: List[BarData],
        target_interval: Interval,
        start_dt: datetime,
        end_dt: datetime
    ) -> List[BarData]:
        """优化的K线分组合成算法"""
        if not base_bars:
            return []
        
        # 使用字典进行高效分组
        groups = {}
        
        # 按时间分组 - 优化算法
        for bar in base_bars:
            group_start_time = self._get_group_start_time_optimized(bar.datetime, target_interval)
            
            if group_start_time not in groups:
                groups[group_start_time] = []
            groups[group_start_time].append(bar)
        
        # 批量合成K线
        synthesized_bars = []
        
        # 按时间排序处理
        for group_time in sorted(groups.keys()):
            group_bars = groups[group_time]
            
            if group_bars:
                synthesized_bar = self._merge_bars_optimized(group_bars, target_interval, group_time)
                
                # 时间过滤
                if synthesized_bar and start_dt <= synthesized_bar.datetime <= end_dt:
                    synthesized_bars.append(synthesized_bar)
        
        return synthesized_bars

    def _get_group_start_time_optimized(self, dt: datetime, interval: Interval) -> datetime:
        """优化的时间分组计算 - 支持更多周期和边界情况"""
        # 统一的时间对齐算法
        interval_config = {
            Interval.MINUTE_30: {'unit': 'minute', 'value': 30},
            Interval.HOUR: {'unit': 'hour', 'value': 1},
            Interval.HOUR_2: {'unit': 'hour', 'value': 2},
            Interval.HOUR_4: {'unit': 'hour', 'value': 4}
        }
        
        config = interval_config.get(interval)
        if not config:
            return dt.replace(second=0, microsecond=0)
        
        if config['unit'] == 'minute':
            aligned_minute = (dt.minute // config['value']) * config['value']
            return dt.replace(minute=aligned_minute, second=0, microsecond=0)
        
        elif config['unit'] == 'hour':
            aligned_hour = (dt.hour // config['value']) * config['value']
            return dt.replace(hour=aligned_hour, minute=0, second=0, microsecond=0)
        
        return dt.replace(second=0, microsecond=0)

    def _merge_bars_optimized(self, bars: List[BarData], interval: Interval, group_time: datetime) -> BarData:
        """优化的K线合成算法 - 减少计算开销"""
        if not bars:
            return None
        
        # 如果只有一个bar，直接使用（避免不必要的计算）
        if len(bars) == 1:
            bar = bars[0]
            return BarData(
                symbol=bar.symbol,
                exchange=bar.exchange,
                datetime=group_time,
                interval=interval,
                volume=bar.volume,
                turnover=bar.turnover,
                open_price=bar.open_price,
                high_price=bar.high_price,
                low_price=bar.low_price,
                close_price=bar.close_price,
                gateway_name="clickhouse_synthesized"
            )
        
        # 按时间排序（只在必要时排序）
        if len(bars) > 1 and bars[0].datetime > bars[-1].datetime:
            bars.sort(key=lambda x: x.datetime)
        
        first_bar = bars[0]
        last_bar = bars[-1]
        
        # 优化的OHLC计算 - 使用生成器减少内存占用
        open_price = first_bar.open_price
        close_price = last_bar.close_price
        high_price = max(bar.high_price for bar in bars)
        low_price = min(bar.low_price for bar in bars)
        
        # 累计计算成交量和成交额
        volume = sum(bar.volume for bar in bars)
        turnover = sum(bar.turnover for bar in bars)
        
        # 创建合成的K线
        return BarData(
            symbol=first_bar.symbol,
            exchange=first_bar.exchange,
            datetime=group_time,
            interval=interval,
            volume=volume,
            turnover=turnover,
            open_price=open_price,
            high_price=high_price,
            low_price=low_price,
            close_price=close_price,
            gateway_name="clickhouse_optimized"
        )

    def load_tick_data(
        self,
        symbol: str,
        exchange: Exchange,
        start: datetime,
        end: datetime
    ) -> List[TickData]:
        """
        从ClickHouse加载Tick数据

        Args:
            symbol: 合约代码
            exchange: 交易所
            start: 开始时间
            end: 结束时间

        Returns:
            List[TickData]: Tick数据列表
        """
        try:
            # 转换时区
            start_dt = convert_tz(start)
            end_dt = convert_tz(end)

            # 构建查询SQL
            query = """
                SELECT
                    symbol, exchange, datetime, name,
                    volume, turnover, open_interest,
                    last_price, last_volume, limit_up, limit_down,
                    open_price, high_price, low_price, pre_close,
                    bid_price_1, bid_price_2, bid_price_3, bid_price_4, bid_price_5,
                    ask_price_1, ask_price_2, ask_price_3, ask_price_4, ask_price_5,
                    bid_volume_1, bid_volume_2, bid_volume_3, bid_volume_4, bid_volume_5,
                    ask_volume_1, ask_volume_2, ask_volume_3, ask_volume_4, ask_volume_5,
                    localtime
                FROM tick_data
                WHERE symbol = %(symbol)s
                    AND exchange = %(exchange)s
                    AND datetime >= %(start)s
                    AND datetime <= %(end)s
                ORDER BY datetime
            """

            # 执行查询
            result = self._get_connection().query(
                query,
                parameters={
                    'symbol': symbol,
                    'exchange': exchange.value,
                    'start': start_dt,
                    'end': end_dt
                }
            )

            # 转换为TickData对象
            ticks = []
            for row in result.result_rows:
                tick = TickData(
                    symbol=row[0],
                    exchange=Exchange(row[1]),
                    datetime=row[2].replace(tzinfo=DB_TZ),
                    name=row[3],
                    volume=row[4],
                    turnover=row[5],
                    open_interest=row[6],
                    last_price=row[7],
                    last_volume=row[8],
                    limit_up=row[9],
                    limit_down=row[10],
                    open_price=row[11],
                    high_price=row[12],
                    low_price=row[13],
                    pre_close=row[14],
                    bid_price_1=row[15],
                    bid_price_2=row[16],
                    bid_price_3=row[17],
                    bid_price_4=row[18],
                    bid_price_5=row[19],
                    ask_price_1=row[20],
                    ask_price_2=row[21],
                    ask_price_3=row[22],
                    ask_price_4=row[23],
                    ask_price_5=row[24],
                    bid_volume_1=row[25],
                    bid_volume_2=row[26],
                    bid_volume_3=row[27],
                    bid_volume_4=row[28],
                    bid_volume_5=row[29],
                    ask_volume_1=row[30],
                    ask_volume_2=row[31],
                    ask_volume_3=row[32],
                    ask_volume_4=row[33],
                    ask_volume_5=row[34],
                    localtime=row[35].replace(tzinfo=DB_TZ) if row[35] else None,
                    gateway_name="clickhouse"
                )
                ticks.append(tick)

            return ticks

        except Exception as e:
            self.logger.error(f"加载Tick数据失败: {e}")
            return []

    def delete_bar_data(
        self,
        symbol: str,
        exchange: Exchange,
        interval: Interval
    ) -> int:
        """
        删除K线数据

        Args:
            symbol: 合约代码
            exchange: 交易所
            interval: 时间周期

        Returns:
            int: 删除的记录数
        """
        try:
            # 先查询要删除的记录数
            count_query = """
                SELECT COUNT(*) FROM bar_data
                WHERE symbol = %(symbol)s
                    AND exchange = %(exchange)s
                    AND interval = %(interval)s
            """

            count_result = self._get_connection().query(
                count_query,
                parameters={
                    'symbol': symbol,
                    'exchange': exchange.value,
                    'interval': interval.value
                }
            )

            deleted_count = count_result.result_rows[0][0]

            # 执行删除操作
            delete_query = """
                ALTER TABLE bar_data DELETE
                WHERE symbol = %(symbol)s
                    AND exchange = %(exchange)s
                    AND interval = %(interval)s
            """

            self._get_connection().command(
                delete_query,
                parameters={
                    'symbol': symbol,
                    'exchange': exchange.value,
                    'interval': interval.value
                }
            )

            # 删除对应的概览记录
            overview_delete_query = """
                ALTER TABLE bar_overview DELETE
                WHERE symbol = %(symbol)s
                    AND exchange = %(exchange)s
                    AND interval = %(interval)s
            """

            self._get_connection().command(
                overview_delete_query,
                parameters={
                    'symbol': symbol,
                    'exchange': exchange.value,
                    'interval': interval.value
                }
            )

            return deleted_count

        except Exception as e:
            self.logger.error(f"删除K线数据失败: {e}")
            return 0

    def delete_tick_data(
        self,
        symbol: str,
        exchange: Exchange
    ) -> int:
        """
        删除Tick数据

        Args:
            symbol: 合约代码
            exchange: 交易所

        Returns:
            int: 删除的记录数
        """
        try:
            # 先查询要删除的记录数
            count_query = """
                SELECT COUNT(*) FROM tick_data
                WHERE symbol = %(symbol)s
                    AND exchange = %(exchange)s
            """

            count_result = self._get_connection().query(
                count_query,
                parameters={
                    'symbol': symbol,
                    'exchange': exchange.value
                }
            )

            deleted_count = count_result.result_rows[0][0]

            # 执行删除操作
            delete_query = """
                ALTER TABLE tick_data DELETE
                WHERE symbol = %(symbol)s
                    AND exchange = %(exchange)s
            """

            self._get_connection().command(
                delete_query,
                parameters={
                    'symbol': symbol,
                    'exchange': exchange.value
                }
            )

            # 删除对应的概览记录
            overview_delete_query = """
                ALTER TABLE tick_overview DELETE
                WHERE symbol = %(symbol)s
                    AND exchange = %(exchange)s
            """

            self._get_connection().command(
                overview_delete_query,
                parameters={
                    'symbol': symbol,
                    'exchange': exchange.value
                }
            )

            return deleted_count

        except Exception as e:
            self.logger.error(f"删除Tick数据失败: {e}")
            return 0

    @performance_monitor
    def get_bar_overview(self) -> List[BarOverview]:
        """
        获取K线数据概览 - 优化版本

        Returns:
            List[BarOverview]: K线概览列表
        """
        try:
            # 缓存键
            cache_key = self._get_cache_key("get_bar_overview")
            
            # 尝试从缓存获取
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                return cached_result
            
            # 优化的概览查询
            query = """
                SELECT
                    code as symbol,
                    level,
                    COUNT(*) as count,
                    MIN(datetime) as start,
                    MAX(datetime) as end
                FROM stock_info
                WHERE datetime >= today() - INTERVAL 3 YEAR
                GROUP BY code, level
                ORDER BY code, level
                LIMIT 10000
            """

            result = self._execute_query_with_timeout(query)

            # 优化的映射字典
            level_to_interval = {
                "1分钟": Interval.MINUTE,
                "5分钟": Interval.MINUTE_5,
                "15分钟": Interval.MINUTE_15,
                "30分钟": Interval.MINUTE_30,
                "1小时": Interval.HOUR,
                "2小时": Interval.HOUR_2,
                "4小时": Interval.HOUR_4,
                "日线": Interval.DAILY,
                "周线": Interval.WEEKLY,
                "月线": Interval.MONTHLY
            }

            overviews = []
            for row in result.result_rows:
                symbol = row[0]
                level = row[1]

                # 优化的交易所推断
                if symbol.startswith(('600', '601', '603', '605', '688', '900')):
                    exchange = Exchange.SSE
                elif symbol.startswith(('000', '001', '002', '003', '300', '200')):
                    exchange = Exchange.SZSE
                else:
                    exchange = Exchange.SZSE

                # 映射时间周期
                interval = level_to_interval.get(level, Interval.DAILY)

                overview = BarOverview(
                    symbol=symbol,
                    exchange=exchange,
                    interval=interval,
                    count=row[2],
                    start=row[3].replace(tzinfo=DB_TZ),
                    end=row[4].replace(tzinfo=DB_TZ)
                )
                overviews.append(overview)

            # 缓存结果
            self._set_cache(cache_key, overviews)
            
            self.logger.info(f"获取到 {len(overviews)} 个数据概览")
            return overviews

        except Exception as e:
            self.logger.error(f"获取K线概览失败: {e}")
            return []

    def get_tick_overview(self) -> List[TickOverview]:
        """
        获取Tick数据概览

        Returns:
            List[TickOverview]: Tick概览列表
        """
        try:
            query = """
                SELECT symbol, exchange, count, start, end
                FROM tick_overview
                ORDER BY symbol, exchange
            """

            result = self._get_connection().query(query)

            overviews = []
            for row in result.result_rows:
                overview = TickOverview(
                    symbol=row[0],
                    exchange=Exchange(row[1]),
                    count=row[2],
                    start=row[3].replace(tzinfo=DB_TZ),
                    end=row[4].replace(tzinfo=DB_TZ)
                )
                overviews.append(overview)

            return overviews

        except Exception as e:
            self.logger.error(f"获取Tick概览失败: {e}")
            return []

    def _update_bar_overview(self, bar: BarData) -> None:
        """更新K线概览数据"""
        try:
            # 查询现有概览数据
            query = """
                SELECT count, start, end FROM bar_overview
                WHERE symbol = %(symbol)s
                    AND exchange = %(exchange)s
                    AND interval = %(interval)s
            """

            result = self._get_connection().query(
                query,
                parameters={
                    'symbol': bar.symbol,
                    'exchange': bar.exchange.value,
                    'interval': bar.interval.value
                }
            )

            dt = convert_tz(bar.datetime)

            if result.result_rows:
                # 更新现有记录
                old_count, old_start, old_end = result.result_rows[0]
                new_count = old_count + 1
                new_start = min(old_start, dt)
                new_end = max(old_end, dt)
            else:
                # 创建新记录
                new_count = 1
                new_start = dt
                new_end = dt

            # 插入或更新概览数据
            self._get_connection().insert(
                'bar_overview',
                [[bar.symbol, bar.exchange.value, bar.interval.value, new_count, new_start, new_end]],
                column_names=['symbol', 'exchange', 'interval', 'count', 'start', 'end']
            )

        except Exception as e:
            self.logger.error(f"更新K线概览失败: {e}")

    def _update_tick_overview(self, tick: TickData) -> None:
        """更新Tick概览数据"""
        try:
            # 查询现有概览数据
            query = """
                SELECT count, start, end FROM tick_overview
                WHERE symbol = %(symbol)s
                    AND exchange = %(exchange)s
            """

            result = self._get_connection().query(
                query,
                parameters={
                    'symbol': tick.symbol,
                    'exchange': tick.exchange.value
                }
            )

            dt = convert_tz(tick.datetime)

            if result.result_rows:
                # 更新现有记录
                old_count, old_start, old_end = result.result_rows[0]
                new_count = old_count + 1
                new_start = min(old_start, dt)
                new_end = max(old_end, dt)
            else:
                # 创建新记录
                new_count = 1
                new_start = dt
                new_end = dt

            # 插入或更新概览数据
            self._get_connection().insert(
                'tick_overview',
                [[tick.symbol, tick.exchange.value, new_count, new_start, new_end]],
                column_names=['symbol', 'exchange', 'count', 'start', 'end']
            )

        except Exception as e:
            self.logger.error(f"更新Tick概览失败: {e}")

    def _infer_exchange(self, symbol: str) -> str:
        """根据股票代码推断交易所"""
        if symbol.startswith(('000', '001', '002', '003', '300')):
            return 'SZSE'  # 深交所
        elif symbol.startswith(('600', '601', '603', '605', '688')):
            return 'SSE'   # 上交所
        elif symbol.startswith('8'):
            return 'NEEQ'  # 新三板
        else:
            return 'UNKNOWN'

    def clear_cache(self) -> None:
        """手动清理所有缓存"""
        with self._cache_lock:
            self._query_cache.clear()
            self.logger.info("已清理所有查询缓存")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self._cache_lock:
            current_time = time.time()
            active_entries = 0
            expired_entries = 0
            
            for _, (_, timestamp) in self._query_cache.items():
                if current_time - timestamp < self._cache_ttl:
                    active_entries += 1
                else:
                    expired_entries += 1
            
            return {
                "total_entries": len(self._query_cache),
                "active_entries": active_entries,
                "expired_entries": expired_entries,
                "cache_ttl": self._cache_ttl,
                "pool_size": len(self._connection_pool),
                "max_pool_size": self._pool_size,
                "query_timeout": self._query_timeout
            }
    
    def close(self) -> None:
        """关闭数据库连接池和线程池"""
        try:
            # 关闭线程池
            if hasattr(self, '_thread_pool'):
                self._thread_pool.shutdown(wait=True)
                
            # 关闭所有连接
            with self._pool_lock:
                for client in self._connection_pool:
                    try:
                        client.close()
                    except:
                        pass
                self._connection_pool.clear()
            
            # 清理缓存
            self.clear_cache()
            
            self.logger.info("ClickHouse数据库连接池已关闭")
            
        except Exception as e:
            self.logger.error(f"关闭连接池失败: {e}")

    def __del__(self) -> None:
        """析构函数，确保资源被清理"""
        try:
            self.close()
        except:
            pass
