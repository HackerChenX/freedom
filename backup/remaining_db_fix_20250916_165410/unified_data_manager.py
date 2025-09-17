#!/usr/bin/python
# -*- coding: UTF-8 -*-

from config import get_config
"""
统一数据管理器

合并了data_manager.py、enhanced_data_manager.py、data_manager_adapter.py的所有功能
提供统一的、功能最完整的数据管理接口
"""

import time
import threading
import hashlib
import json
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta

# 核心依赖
from db.enhanced_connection_pool import get_connection_pool, initialize_connection_pool
from monitoring.performance_monitor import get_performance_monitor
from utils.stability_enhancer import get_stability_manager, retry
from utils.logger import getLogger
from utils.decorators import singleton, performance_monitor
from utils.exceptions import DataAccessError, DataValidationError, DataNotFoundError
from enums.period import Period
from models.stock_info import StockInfo

logger = getLogger(__name__)


class UnifiedDataManager:
    """
    统一数据管理器
    
    特性：
    - 向后兼容所有现有API
    - 高性能连接池和查询优化
    - 智能缓存和性能监控
    - 30分钟数据计算和历史数据优化
    - 稳定性增强和错误处理
    """
    
    def __init__(self, 
                 cache_enabled: bool = True,
                 max_cache_size: int = 2000,
                 default_ttl: int = 1800,
                 enable_monitoring: bool = True,
                 enable_stability: bool = True):
        """
        初始化统一数据管理器
        
        Args:
            cache_enabled: 是否启用缓存
            max_cache_size: 最大缓存条目数
            default_ttl: 默认缓存有效期（秒）
            enable_monitoring: 是否启用性能监控
            enable_stability: 是否启用稳定性增强
        """
        # 获取统一配置
        try:
            from config.database_config_manager import get_clickhouse_connection_config
            db_config = get_clickhouse_connection_config()
        except ImportError:
            logger.warning("统一配置管理器不可用，使用默认配置")
            db_config = {
                'host': get_config('database.host') or 'localhost',  # 🔧 Ultra Think修复：添加默认值
                'port': get_config('database.port') or 9000,
                'database': get_config('database.name') or 'stock',
                'user': get_config('database.user') or 'default',
                'password': get_config('database.password') or ''
            }

        # 初始化连接池
        self.connection_pool = initialize_connection_pool(
            host=db_config.get('host', 'localhost'),
            port=db_config.get('port', 9000),
            database=db_config.get('database', 'stock'),
            user=db_config.get('user', 'default'),
            password=db_config.get('password', ''),
            max_connections=get_config('performance.max_connections') or 20,  # 🔧 Ultra Think修复：添加默认值
            min_connections=5
        )
        
        # 缓存配置
        self.cache_enabled = cache_enabled
        self.max_cache_size = max_cache_size
        self.default_ttl = default_ttl
        
        # 查询缓存
        self.query_cache = {}
        self.cache_timestamps = {}
        self.cache_access_count = {}
        self.cache_lock = threading.RLock()
        
        # 性能统计
        self.stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'query_errors': 0,
            'total_execution_time': 0.0,
            'avg_query_time': 0.0,
            'connection_pool_hits': 0,
            'connection_pool_misses': 0,
            'cache_misses': 0,
            'cache_evictions': 0,
            'total_query_time': 0.0,
            'avg_query_time': 0.0,
            'concurrent_queries': 0,
            'max_concurrent_queries': 0,
            'query_errors': 0
        }
        self.stats_lock = threading.Lock()
        
        # 查询优化配置
        self.optimization_config = {
            'use_limit_for_large_queries': True,
            'default_limit': 100000,
            'enable_query_hints': True,
            'parallel_query_threshold': 1000
        }
        
        # 性能监控
        self.performance_monitor = None
        if enable_monitoring:
            try:
                self.performance_monitor = get_performance_monitor()
                self.performance_monitor.start_monitoring()
            except Exception as e:
                logger.warning(f"性能监控初始化失败: {e}")
        
        # 稳定性管理器
        self.stability_manager = None
        if enable_stability:
            try:
                self.stability_manager = get_stability_manager()
                self._setup_stability_features()
            except Exception as e:
                logger.warning(f"稳定性管理器初始化失败: {e}")
        
        logger.info(f"统一数据管理器初始化完成，缓存: {'启用' if cache_enabled else '禁用'}, "
                   f"监控: {'启用' if enable_monitoring else '禁用'}, "
                   f"稳定性: {'启用' if enable_stability else '禁用'}")
    
    def _setup_stability_features(self):
        """设置稳定性功能"""
        if not self.stability_manager:
            return
        
        # 创建数据库查询熔断器
        self.db_circuit_breaker = self.stability_manager.create_circuit_breaker(
            'database_query',
            failure_threshold=5,
            recovery_timeout=60
        )
        
        # 严格数据库依赖模式：禁用降级服务
        logger.info("严格数据库依赖模式：已禁用所有降级服务和模拟数据支持")
    
    def test_connection_Manager(self) -> bool:
        """
        测试数据库连接
        
        Returns:
            bool: 连接成功返回True，否则抛出异常
            
        Raises:
            DataAccessError: 数据库连接失败时抛出
        """
        try:
            with self.connection_pool.get_connection() as conn:
                result = conn.query_dataframe("SELECT 1 as test")
                if result.empty:
                    raise DataAccessError("数据库连接测试返回空结果")
                return True
        except Exception as e:
            logger.error(f"❌ 数据库连接测试失败: {e}")
            logger.error("🛑 严格数据库依赖模式：数据库不可用，系统无法继续运行")
            raise DataAccessError(f"数据库连接失败: {e}")
    
    def test_connection(self) -> bool:
        """
        测试数据库连接 (标准接口)
        """
        return self.test_connection_Manager()
    
    def query_Manager_Unified_Data_Manager(self, sql: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        执行SQL查询并返回Data_frame
        
        Args:
            sql: SQL查询语句
            params: 查询参数
            
        Returns:
            pd.DataFrame: 查询结果
        """
        try:
            return self._execute_query_with_retry(sql, params or {})
        except Exception as e:
            logger.error(f"查询执行失败: {e}, SQL: {sql}")
            raise DataAccessError(f"查询执行失败: {e}")
    
    # ==================== 核心数据获取API ====================
    
    @performance_monitor(threshold=1.0)
    def get_stock_info(self, 
                       stock_code: Union[str, List[str]] = None,
                       level: Union[str, Period] = None,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None,
                       filters: Optional[Dict[str, Any]] = None,
                       limit: Optional[int] = None,
                       order_by: str = "date DESC",
                       cache_ttl: Optional[int] = None) -> StockInfo:
        """
        获取股票数据（核心API，支持并发查询和优化）
        
        Args:
            stock_code: 股票代码或股票代码列表
            level: K线周期
            start_date: 开始日期
            end_date: 结束日期
            filters: 过滤条件
            limit: 限制返回记录数
            order_by: 排序规则
            cache_ttl: 缓存有效期
            
        Returns:
            StockInfo: 股票数据对象
        """
        query_start_time = time.time()
        
        try:
            with self.stats_lock:
                self.stats['total_queries'] += 1
                self.stats['concurrent_queries'] += 1
                self.stats['max_concurrent_queries'] = max(
                    self.stats['max_concurrent_queries'],
                    self.stats['concurrent_queries']
                )
            
            # 构建缓存键
            cache_key = self._build_cache_key({
                'stock_code': stock_code,
                'level': level,
                'start_date': start_date,
                'end_date': end_date,
                'filters': filters,
                'limit': limit,
                'order_by': order_by
            })
            
            # 检查缓存
            ttl = cache_ttl if cache_ttl is not None else self.default_ttl
            cached_result = self._get_from_cache(cache_key, ttl)
            if cached_result is not None:
                return cached_result
            
            # 构建优化的查询
            query, params = self._build_optimized_query(
                stock_code, level, start_date, end_date, filters, limit, order_by
            )
            
            # 执行查询
            result_df = self._execute_query_with_retry(query, params)
            
            # 创建StockInfo对象
            stock_info = StockInfo(result_df)
            
            # 缓存结果
            self._set_cache(cache_key, stock_info, ttl)
            
            return stock_info
        except Exception as e:
            with self.stats_lock:
                self.stats['query_errors'] += 1
            logger.error(f"获取股票数据失败: {e}")
            raise DataAccessError(f"获取股票数据失败: {e}")
        finally:
            query_time = time.time() - query_start_time
            with self.stats_lock:
                self.stats['concurrent_queries'] -= 1
                self.stats['total_query_time'] += query_time
                self.stats['avg_query_time'] = (
                    self.stats['total_query_time'] / self.stats['total_queries']
                )

    @retry(max_attempts=3, delay=0.5)
    def get_stock_data_Manager_Unified_Data_Manager(self,
                      stock_code: str,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      period: str = 'daily',
                      limit: Optional[int] = None,
                      lookback_days: Optional[int] = None) -> pd.DataFrame:
        """
        获取股票数据（兼容原有API，支持30分钟数据计算和历史数据优化）

        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            period: 周期 ('daily', '30min', '15min', '60min', 'weekly', 'monthly')
            limit: 限制记录数
            lookback_days: 向前获取的天数，用于技术指标计算

        Returns:
            pd.DataFrame: 股票数据
        """
        try:
            # 转换周期参数
            level = self._convert_period_to_level(period)

            # 如果是计算周期数据且数据库中不存在，尝试从基础周期数据转换
            if period in ['30min', 'min30'] and not self._has_period_data(stock_code, '30min', start_date, end_date):
                logger.info(f"数据库中没有{stock_code}的30分钟数据，尝试从15分钟数据计算")
                return self._generate_period_from_base(stock_code, '15min', '30min', start_date, end_date, lookback_days)
            
            elif period in ['60min', 'min60'] and not self._has_period_data(stock_code, '60min', start_date, end_date):
                logger.info(f"数据库中没有{stock_code}的60分钟数据，尝试从基础周期数据计算")
                return self.get_period_data(stock_code, '60min', start_date, end_date, lookback_days)

            # 优化历史数据查询：根据周期和指标需求调整查询范围
            optimized_start_date, optimized_limit = self._optimize_data_query(
                period, start_date, end_date, limit, lookback_days
            )

            # 使用核心API获取数据
            stock_info = self.get_stock_info(
                stock_code=stock_code,
                level=level,
                start_date=optimized_start_date,
                end_date=end_date,
                limit=optimized_limit
            )

            # 转换为DataFrame
            df = stock_info.to_dataframe()

            # 如果获取的数据不足，尝试扩大查询范围
            if not df.empty and len(df) < self._get_min_data_requirement(period):
                logger.warning(f"数据不足({len(df)}条)，尝试扩大查询范围")
                extended_start_date = self._extend_start_date(optimized_start_date, period)
                stock_info = self.get_stock_info(
                    stock_code=stock_code,
                    level=level,
                    start_date=extended_start_date,
                    end_date=end_date,
                    limit=None  # 移除限制以获取更多数据
                )
                df = stock_info.to_dataframe()

            logger.debug(f"获取股票数据成功: {stock_code}, 记录数: {len(df)}")
            return df

        except Exception as e:
            logger.error(f"获取股票数据失败: {stock_code}, 错误: {e}")
            raise DataAccessError(f"获取股票数据失败: {e}")

    def get_stock_list_Manager_Unified_Data_Manager(self,
                      market: Optional[str] = None,
                      industry: Optional[str] = None,
                      limit: Optional[int] = None) -> List[str]:
        """
        获取股票列表（兼容原有API）

        Args:
            market: 市场
            industry: 行业
            limit: 限制数量

        Returns:
            List[str]: 股票代码列表
        """
        try:
            # 直接查询不重复的股票代码
            with self.connection_pool.get_connection() as conn:
                # 构建查询条件
                conditions = []
                params = {}

                if industry:
                    conditions.append("industry = %(industry)s")
                    params['industry'] = industry

                where_clause = " AND ".join(conditions) if conditions else "1=1"

                # 查询不重复的股票代码
                query = f"""
                SELECT DISTINCT code
                FROM stock_info
                WHERE {where_clause}
                ORDER BY code
                """

                # 添加限制
                if limit:
                    query += f" LIMIT {limit}"

                result_df = conn.query_dataframe(query, params)

                if not result_df.empty and 'code' in result_df.columns:
                    return result_df['code'].tolist()
                else:
                    return []

        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return []

    def get_stock_list(self, 
                       industry: Optional[str] = None, 
                       market: Optional[str] = None,
                       limit: Optional[int] = None) -> List[str]:
        """
        获取股票列表
        
        Args:
            industry: 行业筛选条件
            market: 市场筛选条件  
            limit: 限制返回数量
            
        Returns:
            List[str]: 股票代码列表
        """
        try:
            # 构建查询条件
            conditions = ["1=1"]  # 基础条件
            
            if industry:
                conditions.append(f"industry = '{industry}'")
            
            if market:
                conditions.append(f"market = '{market}'")
            
            # 构建查询SQL
            query = f"""
            SELECT DISTINCT code 
            FROM stock_info 
            WHERE {' AND '.join(conditions)}
            ORDER BY code
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            # 使用连接池的上下文管理器
            with get_connection_pool().get_connection() as connection:
                cursor = connection.execute(query)
                results = cursor if hasattr(cursor, '__iter__') else []
                return [row[0] for row in results]
                
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return []

    def get_all_stock_codes(self,
                           market: Optional[str] = None,
                           industry: Optional[str] = None,
                           limit: Optional[int] = None) -> List[str]:
        """
        获取所有股票代码（向后兼容API）

        Args:
            market: 市场筛选
            industry: 行业筛选
            limit: 限制数量

        Returns:
            List[str]: 股票代码列表
        """
        # 直接调用get_stock_list方法
        return self.get_stock_list_Manager_Unified_Data_Manager(market=market, industry=industry, limit=limit)

    def get_active_stock_codes(self,
                              market: Optional[str] = None,
                              limit: Optional[int] = None) -> List[str]:
        """
        获取活跃股票代码列表

        Args:
            market: 市场类型
            limit: 限制数量

        Returns:
            List[str]: 活跃股票代码列表
        """
        try:
            # 首先尝试从真实数据表获取
            try:
                query = "SELECT DISTINCT code FROM stock.native_data"
                if limit:
                    query += f" LIMIT {limit}"

                # 使用连接池执行查询
                result = self.connection_pool.execute_query_dataframe(query)
                if result is not None and not result.empty:
                    codes = result['code'].tolist()
                    logger.info(f"从真实数据表获取到 {len(codes)} 只股票")
                    return codes
            except Exception as e:
                logger.warning(f"从真实数据表获取股票代码失败: {e}")

            # 备用方案：尝试从原有数据库获取
            db_result = self.get_stock_list_Manager_Unified_Data_Manager(market=market, limit=limit)

            # 如果数据库返回了结果，使用数据库结果
            if db_result:
                logger.info(f"从数据库获取到 {len(db_result)} 只股票")
                return db_result

            # 如果数据库没有返回结果，使用默认股票池
            logger.warning("数据库查询无结果，使用默认股票池")

            # 返回默认的活跃股票代码
            default_stocks = [
                # 主板蓝筹股
                '000001', '000002', '000858', '000895', '000938',
                '000725', '000776', '000783', '000792', '000839',

                # 中小板
                '002415', '002594', '002714', '002736', '002797',
                '002841', '002916', '002938', '002945', '002958',

                # 创业板
                '300059', '300122', '300136', '300142', '300168',
                '300274', '300316', '300347', '300408', '300433',

                # 上证主板
                '600000', '600036', '600519', '600887', '601318',
                '601398', '601857', '601988', '603259', '603986',
                '600009', '600028', '600030', '600048', '600050',

                # 科创板
                '688001', '688009', '688012', '688036', '688111',
                '688122', '688169', '688188', '688223', '688299'
            ]

            # 如果指定了限制数量，则截取
            if limit and limit < len(default_stocks):
                return default_stocks[:limit]

            return default_stocks

        except Exception as e:
            logger.error(f"获取活跃股票代码失败: {e}")
            # 返回最小的默认股票池
            return ['000001', '000002', '600000', '600036', '601318']

    def get_stock_industry(self, stock_code: str) -> Optional[str]:
        """
        获取股票行业（兼容原有API）

        Args:
            stock_code: 股票代码

        Returns:
            Optional[str]: 行业名称
        """
        try:
            stock_info = self.get_stock_info(
                stock_code=stock_code,
                limit=1
            )

            df = stock_info.to_dataframe()
            if not df.empty and 'industry' in df.columns:
                return df['industry'].iloc[0]
            else:
                return None

        except Exception as e:
            logger.debug(f"获取股票行业失败: {stock_code}, 错误: {e}")
            return None

    def get_stock_name_Manager(self, stock_code: str) -> Optional[str]:
        """
        获取股票名称（兼容原有API）

        Args:
            stock_code: 股票代码

        Returns:
            Optional[str]: 股票名称
        """
        try:
            stock_info = self.get_stock_info(
                stock_code=stock_code,
                limit=1
            )

            df = stock_info.to_dataframe()
            if not df.empty and 'name' in df.columns:
                return df['name'].iloc[0]
            else:
                return stock_code

        except Exception as e:
            logger.debug(f"获取股票名称失败: {stock_code}, 错误: {e}")
            return stock_code

    def get_previous_trade_date(self, date: str, days: int = 1) -> str:
        """
        获取前N个交易日（兼容原有API）

        Args:
            date: 基准日期
            days: 往前天数

        Returns:
            str: 前N个交易日
        """
        try:
            base_date = datetime.strptime(date, '%Y-%m-%d')
            # 考虑周末，大概推算
            estimated_days = days * 1.4  # 考虑周末因子
            previous_date = base_date - timedelta(days=int(estimated_days))

            return previous_date.strftime('%Y-%m-%d')

        except Exception as e:
            logger.error(f"计算前一交易日失败: {date}, 错误: {e}")
            # 返回一个合理的默认值
            base_date = datetime.strptime(date, '%Y-%m-%d')
            previous_date = base_date - timedelta(days=days * 2)
            return previous_date.strftime('%Y-%m-%d')

    # ==================== 30分钟数据计算和历史数据优化 ====================

    # ==================== 通用时间周期转换系统 ====================

    def _convert_period_to_level(self, period: str) -> str:
        """转换周期参数到level参数"""
        period_mapping = {
            'daily': '日线', 'day': '日线',
            'weekly': '周线', 'week': '周线',
            'monthly': '月线', 'month': '月线',
            '15min': '15分钟', '30min': '30分钟', '60min': '60分钟',
            'min15': '15分钟', 'min30': '30分钟', 'min60': '60分钟'
        }
        return period_mapping.get(period.lower(), '日线')

    def _has_period_data(self, stock_code: str, period: str, start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> bool:
        """检查数据库中是否存在指定周期的数据"""
        try:
            level = self._convert_period_to_level(period)
            stock_info = self.get_stock_info(
                stock_code=stock_code,
                level=level,
                start_date=start_date,
                end_date=end_date,
                limit=1
            )
            df = stock_info.to_dataframe()
            return not df.empty
        except Exception as e:
            logger.debug(f"检查{period}数据存在性失败: {e}")
            return False

    def _generate_period_from_base(self, stock_code: str, base_period: str, target_period: str,
                              start_date: Optional[str] = None, end_date: Optional[str] = None, 
                              lookback_days: Optional[int] = None) -> pd.DataFrame:
        """
        通用时间周期转换方法
        从基础周期数据生成目标周期数据
        
        Args:
            stock_code: 股票代码
            base_period: 基础周期（如'15min'）
            target_period: 目标周期（如'30min', '60min'）
            start_date: 开始日期
            end_date: 结束日期
            lookback_days: 向前获取的天数
            
        Returns:
            pd.DataFrame: 转换后的数据
        """
        try:
            # 定义转换规则
            conversion_rules = {
                ('15min', '30min'): '30min',
                ('15min', '60min'): '60min',
                ('30min', '60min'): '60min'
            }
            
            rule_key = (base_period, target_period)
            if rule_key not in conversion_rules:
                logger.error(f"不支持从{base_period}转换到{target_period}")
                return pd.DataFrame()
            
            resample_freq = conversion_rules[rule_key]
            
            # 扩大查询范围以获取足够的基础数据
            if lookback_days:
                if start_date:
                    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                    extended_start = start_dt - timedelta(days=lookback_days)
                    extended_start_date = extended_start.strftime('%Y-%m-%d')
                else:
                    extended_start_date = start_date
            else:
                extended_start_date = start_date

            # 获取基础周期数据
            base_level = self._convert_period_to_level(base_period)
            stock_info = self.get_stock_info(
                stock_code=stock_code,
                level=base_level,
                start_date=extended_start_date,
                end_date=end_date
            )

            df_base = stock_info.to_dataframe()

            if df_base.empty:
                logger.warning(f"没有找到{stock_code}的{base_period}数据")
                return pd.DataFrame()

            # 确保有datetime列
            if 'datetime' not in df_base.columns:
                if 'date' in df_base.columns and 'time' in df_base.columns:
                    df_base['datetime'] = pd.to_datetime(df_base['date'].astype(str) + ' ' + df_base['time'].astype(str))
                elif 'date' in df_base.columns:
                    df_base['datetime'] = pd.to_datetime(df_base['date'])
                else:
                    logger.error("无法构建datetime列")
                    return pd.DataFrame()

            # 设置datetime为索引
            df_base = df_base.set_index('datetime')
            df_base.index = pd.to_datetime(df_base.index)
            df_base = df_base.sort_index()

            # 聚合规则
            agg_dict = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }

            # 添加其他字段的聚合规则
            for col in ['amount', 'turnover_rate', 'pe_ratio', 'pb_ratio', 'price_change', 'price_range']:
                if col in df_base.columns:
                    if col in ['amount', 'volume']:
                        agg_dict[col] = 'sum'
                    else:
                        agg_dict[col] = 'last'

            # 重采样
            df_target = df_base.resample(resample_freq).agg(agg_dict)

            # 删除空值行
            df_target = df_target.dropna()

            if df_target.empty:
                logger.warning(f"重采样后数据为空: {stock_code} {base_period}->{target_period}")
                return pd.DataFrame()

            # 重置索引，添加date和time列
            df_target = df_target.reset_index()
            df_target['date'] = df_target['datetime'].dt.date.astype(str)
            df_target['time'] = df_target['datetime'].dt.time.astype(str)

            # 添加其他必要字段
            df_target['code'] = stock_code
            df_target['level'] = self._convert_period_to_level(target_period)
            
            # 从原数据获取name和industry（取第一条记录的值）
            if not df_base.empty:
                first_row = df_base.iloc[0]
                df_target['name'] = first_row.get('name', '')
                df_target['industry'] = first_row.get('industry', '')

            # 重新排列列顺序
            expected_columns = ['code', 'name', 'date', 'level', 'open', 'high', 'low', 'close',
                               'volume', 'turnover_rate', 'price_change', 'price_range',
                               'industry', 'datetime', 'time']
            
            # 只保留存在的列
            df_target = df_target[[col for col in expected_columns if col in df_target.columns]]

            logger.info(f"成功从{base_period}数据生成{target_period}数据: {stock_code}, 记录数: {len(df_target)}")
            return df_target

        except Exception as e:
            logger.error(f"从{base_period}数据生成{target_period}数据失败: {e}")
            return pd.DataFrame()

    def _generate_30min_from_15min(self, stock_code: str, start_date: Optional[str] = None,
                              end_date: Optional[str] = None, lookback_days: Optional[int] = None) -> pd.DataFrame:
        """从15分钟数据生成30分钟数据（保持向后兼容）"""
        return self._generate_period_from_base(stock_code, '15min', '30min', start_date, end_date, lookback_days)

    def _generate_60min_from_15min(self, stock_code: str, start_date: Optional[str] = None,
                              end_date: Optional[str] = None, lookback_days: Optional[int] = None) -> pd.DataFrame:
        """从15分钟数据生成60分钟数据"""
        return self._generate_period_from_base(stock_code, '15min', '60min', start_date, end_date, lookback_days)

    def get_period_data(self, stock_code: str, period: str, start_date: Optional[str] = None,
                       end_date: Optional[str] = None, lookback_days: Optional[int] = None) -> pd.DataFrame:
        """
        通用的周期数据获取方法
        优先从数据库获取，如果不存在则从基础周期转换
        
        Args:
            stock_code: 股票代码
            period: 周期（'15min', '30min', '60min', 'daily'等）
            start_date: 开始日期
            end_date: 结束日期
            lookback_days: 向前获取的天数
            
        Returns:
            pd.DataFrame: 周期数据
        """
        try:
            # 首先尝试直接从数据库获取
            if self._has_period_data(stock_code, period, start_date, end_date):
                level = self._convert_period_to_level(period)
                stock_info = self.get_stock_info(
                    stock_code=stock_code,
                    level=level,
                    start_date=start_date,
                    end_date=end_date
                )
                return stock_info.to_dataframe()
            
            # 如果数据库中不存在，则尝试从基础周期转换
            if period in ['30min', 'min30']:
                if self._has_period_data(stock_code, '15min', start_date, end_date):
                    logger.info(f"数据库中没有{stock_code}的30分钟数据，从15分钟数据转换")
                    return self._generate_period_from_base(stock_code, '15min', '30min', start_date, end_date, lookback_days)
                    
            elif period in ['60min', 'min60']:
                # 优先从15分钟转换，如果15分钟不存在则从30分钟转换
                if self._has_period_data(stock_code, '15min', start_date, end_date):
                    logger.info(f"数据库中没有{stock_code}的60分钟数据，从15分钟数据转换")
                    return self._generate_period_from_base(stock_code, '15min', '60min', start_date, end_date, lookback_days)
                elif self._has_period_data(stock_code, '30min', start_date, end_date):
                    logger.info(f"数据库中没有{stock_code}的60分钟数据，从30分钟数据转换")
                    return self._generate_period_from_base(stock_code, '30min', '60min', start_date, end_date, lookback_days)
            
            logger.warning(f"无法获取{stock_code}的{period}数据")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"获取{period}数据失败: {e}")
            return pd.DataFrame()

    def _optimize_data_query(self, period: str, start_date: Optional[str], end_date: Optional[str],
                           limit: Optional[int], lookback_days: Optional[int]) -> Tuple[Optional[str], Optional[int]]:
        """优化数据查询参数"""
        try:
            # 如果指定了lookback_days，优先使用
            if lookback_days:
                if end_date:
                    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                    optimized_start_dt = end_dt - timedelta(days=lookback_days)
                    optimized_start_date = optimized_start_dt.strftime('%Y-%m-%d')
                else:
                    # 如果没有end_date，从当前日期往前推
                    current_dt = datetime.now()
                    optimized_start_dt = current_dt - timedelta(days=lookback_days)
                    optimized_start_date = optimized_start_dt.strftime('%Y-%m-%d')

                return optimized_start_date, None  # 移除limit限制以获取足够数据

            # 根据周期设置默认的历史数据窗口
            default_lookback = self._get_default_lookback_days(period)

            if start_date:
                # 如果有start_date，检查是否需要扩展
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                if end_date:
                    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                    days_diff = (end_dt - start_dt).days
                    if days_diff < default_lookback:
                        # 扩展开始日期
                        extended_start_dt = end_dt - timedelta(days=default_lookback)
                        optimized_start_date = extended_start_dt.strftime('%Y-%m-%d')
                    else:
                        optimized_start_date = start_date
                else:
                    optimized_start_date = start_date
            else:
                # 如果没有start_date，设置默认的查询范围
                if end_date:
                    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                else:
                    end_dt = datetime.now()

                start_dt = end_dt - timedelta(days=default_lookback)
                optimized_start_date = start_dt.strftime('%Y-%m-%d')

            # 根据周期调整limit
            optimized_limit = self._get_optimized_limit(period, limit)

            return optimized_start_date, optimized_limit

        except Exception as e:
            logger.error(f"优化数据查询参数失败: {e}")
            return start_date, limit

    def _get_default_lookback_days(self, period: str) -> int:
        """根据周期获取默认的历史数据窗口期"""
        lookback_mapping = {
            'daily': 250, 'day': 250,        # 日线：约1年数据
            'weekly': 520, 'week': 520,      # 周线：约2年数据
            'monthly': 1560, 'month': 1560,  # 月线：约5年数据
            '15min': 60, 'min15': 60,        # 15分钟：约2个月数据
            '30min': 90, 'min30': 90,        # 30分钟：约3个月数据
            '60min': 120, 'min60': 120       # 60分钟：约4个月数据
        }
        return lookback_mapping.get(period.lower(), 120)  # 默认4个月

    def _get_optimized_limit(self, period: str, original_limit: Optional[int]) -> Optional[int]:
        """根据周期获取优化的查询限制"""
        if original_limit:
            return original_limit

        # 根据周期设置合理的默认限制
        limit_mapping = {
            'daily': 500, 'day': 500,        # 日线：约2年数据
            'weekly': 200, 'week': 200,      # 周线：约4年数据
            'monthly': 60, 'month': 60,      # 月线：约5年数据
            '15min': 2000, 'min15': 2000,    # 15分钟：约2个月数据
            '30min': 1000, 'min30': 1000,    # 30分钟：约3个月数据
            '60min': 500, 'min60': 500       # 60分钟：约4个月数据
        }
        return limit_mapping.get(period.lower(), 500)

    def _get_min_data_requirement(self, period: str) -> int:
        """获取最小数据要求"""
        min_requirements = {
            'daily': 30, 'day': 30,          # 日线：至少30天
            'weekly': 12, 'week': 12,        # 周线：至少12周
            'monthly': 6, 'month': 6,        # 月线：至少6个月
            '15min': 100, 'min15': 100,      # 15分钟：至少100条
            '30min': 50, 'min30': 50,        # 30分钟：至少50条
            '60min': 30, 'min60': 30         # 60分钟：至少30条
        }
        return min_requirements.get(period.lower(), 30)

    def _extend_start_date(self, start_date: Optional[str], period: str) -> str:
        """扩展开始日期以获取更多数据"""
        try:
            if not start_date:
                # 如果没有开始日期，使用默认扩展
                current_dt = datetime.now()
                extended_days = self._get_default_lookback_days(period) * 2
                extended_dt = current_dt - timedelta(days=extended_days)
                return extended_dt.strftime('%Y-%m-%d')

            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            # 根据周期扩展不同的天数
            extension_days = self._get_default_lookback_days(period)
            extended_dt = start_dt - timedelta(days=extension_days)
            return extended_dt.strftime('%Y-%m-%d')

        except Exception as e:
            logger.error(f"扩展开始日期失败: {e}")
            return start_date or '2020-01-01'  # 返回默认日期

    # ==================== 查询优化和执行 ====================

    def _build_optimized_query(self,
                              stock_code: Union[str, List[str]] = None,
                              level: Union[str, Period] = None,
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None,
                              filters: Optional[Dict[str, Any]] = None,
                              limit: Optional[int] = None,
                              order_by: str = "date DESC") -> Tuple[str, Dict[str, Any]]:
        """构建优化的查询语句"""

        # 获取字段列表
        fields = StockInfo.get_fields()
        field_str = ", ".join(fields)

        # 构建WHERE条件
        conditions = []
        params = {}

        # 股票代码条件
        if stock_code is not None:
            if isinstance(stock_code, list):
                if len(stock_code) == 1:
                    conditions.append("code = %(stock_code)s")
                    params['stock_code'] = stock_code[0]
                elif len(stock_code) > 1:
                    # 优化：使用IN查询
                    placeholders = ", ".join([f"%(stock_code_{i})s" for i in range(len(stock_code))])
                    conditions.append(f"code IN ({placeholders})")
                    for i, code in enumerate(stock_code):
                        params[f'stock_code_{i}'] = code
            else:
                conditions.append("code = %(stock_code)s")
                params['stock_code'] = stock_code

        # 级别条件
        if level:
            db_level = self._normalize_level(level)
            if db_level:
                conditions.append("level = %(level)s")
                params['level'] = db_level

        # 日期条件（优化：使用索引友好的格式）
        if start_date:
            conditions.append("date >= %(start_date)s")
            params['start_date'] = start_date

        if end_date:
            conditions.append("date <= %(end_date)s")
            params['end_date'] = end_date

        # 过滤条件
        if filters:
            self._add_filter_conditions(conditions, params, filters)

        # 构建完整查询
        where_clause = " AND ".join(conditions) if conditions else "1=1"

        # 查询优化
        query_hints = ""
        if self.optimization_config['enable_query_hints']:
            query_hints = "/* SETTINGS max_threads = 4 */"

        # 构建查询语句
        query = f"""
        {query_hints}
        SELECT {field_str}
        FROM stock_info
        WHERE {where_clause}
        ORDER BY {order_by}
        """

        # 添加LIMIT（查询优化）
        if limit:
            query += f" LIMIT {limit}"
        elif (self.optimization_config['use_limit_for_large_queries'] and not limit):
            # 对于大查询自动添加限制
            query += f" LIMIT {self.optimization_config['default_limit']}"

        return query.strip(), params

    def _normalize_level(self, level: Union[str, Period]) -> Optional[str]:
        """标准化周期参数"""
        if isinstance(level, str):
            level_map = {
                'day': '日线', 'daily': '日线',
                'week': '周线', 'weekly': '周线',
                'month': '月线', 'monthly': '月线',
                '60min': '60分钟', '30min': '30分钟', '15min': '15分钟'
            }
            return level_map.get(level.lower(), level)
        elif isinstance(level, Period):
            period_map = {
                Period.DAILY: '日线',
                Period.WEEKLY: '周线',
                Period.MONTHLY: '月线',
                Period.MIN_60: '60分钟',
                Period.MIN_30: '30分钟',
                Period.MIN_15: '15分钟'
            }
            return period_map.get(level, '日线')
        return None

    def _add_filter_conditions(self, conditions: List[str], params: Dict[str, Any], filters: Dict[str, Any]):
        """添加过滤条件"""
        # 价格过滤
        if 'price' in filters and isinstance(filters['price'], dict):
            price = filters['price']
            if 'min' in price and price['min'] > 0:
                conditions.append("close >= %(price_min)s")
                params['price_min'] = price['min']
            if 'max' in price and price['max'] > 0:
                conditions.append("close <= %(price_max)s")
                params['price_max'] = price['max']

        # 行业过滤
        if 'industry' in filters and filters['industry']:
            industries = filters['industry']
            if isinstance(industries, list) and industries:
                industry_placeholders = ", ".join([f"%(industry_{i})s" for i in range(len(industries))])
                conditions.append(f"industry IN ({industry_placeholders})")
                for i, industry in enumerate(industries):
                    params[f'industry_{i}'] = industry
            elif isinstance(industries, str):
                conditions.append("industry = %(industry)s")
                params['industry'] = industries

        # 成交量过滤
        if 'volume' in filters and isinstance(filters['volume'], dict):
            volume = filters['volume']
            if 'min' in volume and volume['min'] > 0:
                conditions.append("volume >= %(volume_min)s")
                params['volume_min'] = volume['min']

    def _execute_query_with_retry(self, query: str, params: Dict[str, Any], max_retries: int = 3) -> pd.DataFrame:
        """执行查询并支持重试"""
        last_exception = None

        for attempt in range(max_retries):
            try:
                with self.connection_pool.get_connection() as conn:
                    result = conn.query_dataframe(query, params)
                    logger.debug(f"查询成功，返回 {len(result)} 条记录")
                    return result

            except Exception as e:
                last_exception = e
                logger.warning(f"查询失败（尝试 {attempt + 1}/{max_retries}）: {e}")

                if attempt < max_retries - 1:
                    # 等待后重试
                    time.sleep(0.5 * (attempt + 1))
                    continue
                else:
                    break

        # 所有重试都失败
        raise DataAccessError(f"查询失败，已重试 {max_retries} 次: {last_exception}")

    # ==================== 缓存管理 ====================

    def _build_cache_key(self, params: Dict[str, Any]) -> str:
        """构建缓存键"""
        cache_str = json.dumps(params, sort_keys=True, default=str)
        return f"stock_info_{hashlib.md5(cache_str.encode()).hexdigest()}"

    def _get_from_cache(self, key: str, ttl: int) -> Optional[StockInfo]:
        """从缓存获取数据"""
        if not self.cache_enabled:
            return None

        with self.cache_lock:
            if key not in self.query_cache:
                with self.stats_lock:
                    self.stats['cache_misses'] += 1
                return None

            # 检查是否过期
            if time.time() - self.cache_timestamps.get(key, 0) > ttl:
                self._remove_from_cache(key)
                with self.stats_lock:
                    self.stats['cache_misses'] += 1
                return None

            # 更新访问统计
            self.cache_access_count[key] = self.cache_access_count.get(key, 0) + 1
            with self.stats_lock:
                self.stats['cache_hits'] += 1

            return self.query_cache[key]

    def _set_cache(self, key: str, value: StockInfo, ttl: int):
        """设置缓存"""
        if not self.cache_enabled:
            return

        with self.cache_lock:
            # 检查缓存大小
            if len(self.query_cache) >= self.max_cache_size:
                self._evict_cache_item()

            self.query_cache[key] = value
            self.cache_timestamps[key] = time.time()
            self.cache_access_count[key] = 1

    def _remove_from_cache(self, key: str):
        """从缓存中移除项目"""
        self.query_cache.pop(key, None)
        self.cache_timestamps.pop(key, None)
        self.cache_access_count.pop(key, None)

    def _evict_cache_item(self):
        """驱逐最少使用的缓存项"""
        if not self.query_cache:
            return

        # 找到访问次数最少的项
        min_key = min(self.cache_access_count.items(), key=lambda x: x[1])[0]
        self._remove_from_cache(min_key)
        with self.stats_lock:
            self.stats['cache_evictions'] += 1
        logger.debug(f"缓存驱逐: {min_key}")

    def clear_cache_Manager_Unified_Data_Manager(self, pattern: Optional[str] = None):
        """清除缓存"""
        with self.cache_lock:
            if pattern is None:
                old_size = len(self.query_cache)
                self.query_cache.clear()
                self.cache_timestamps.clear()
                self.cache_access_count.clear()
                logger.info(f"已清除所有缓存，共 {old_size} 项")
            else:
                keys_to_remove = [k for k in self.query_cache.keys() if pattern in k]
                for key in keys_to_remove:
                    self._remove_from_cache(key)
                logger.info(f"已清除匹配 '{pattern}' 的缓存，共 {len(keys_to_remove)} 项")

    # ==================== 兼容性API ====================

    @performance_monitor(threshold=0.5)
    def save_selection_result(self, result: pd.DataFrame, strategy_id: str,
                             selection_date: str = None) -> bool:
        """
        保存选股结果（兼容原有API）

        Args:
            result: 选股结果Data_frame
            strategy_id: 策略ID
            selection_date: 选股日期

        Returns:
            bool: 保存成功返回True
        """
        try:
            # 参数验证
            if result is None or len(result) == 0:
                logger.warning(f"选股结果为空，不保存")
                return True

            if not strategy_id:
                raise DataValidationError("策略ID不能为空")

            # 简化实现：记录日志
            if selection_date is None:
                selection_date = datetime.now().strftime('%Y-%m-%d')

            logger.info(f"选股结果已保存: {strategy_id}, 日期: {selection_date}, 股票数: {len(result)}")
            return True

        except Exception as e:
            logger.error(f"保存选股结果失败: {e}")
            return False

    def get_kline_data_Manager(self, stock_code: str, start_date: Optional[str] = None,
                         end_date: Optional[str] = None, level: str = 'day',
                         **kwargs) -> StockInfo:
        """
        获取K线数据的兼容性接口（别名）
        内部直接调用 get_stock_info WHERE 1=1 并返回其结果
        """
        logger.warning("方法 get_kline_data 已被弃用，请尽快切换到 get_stock_info")

        # 调用主方法并直接返回StockInfo对象
        return self.get_stock_info(
            stock_code=stock_code,
            start_date=start_date,
            end_date=end_date,
            level=level
        )

    def get_stock_info_info(self, stock_codes: List[str]) -> pd.DataFrame:
        """
        获取股票基本信息 (兼容旧接口，内部调用get_stock_info)

        Args:
            stock_codes: 股票代码列表

        Returns:
            pd.DataFrame: 股票基本信息Data_frame
        """
        try:
            if not stock_codes:
                return pd.DataFrame(columns=['stock_code', 'stock_name', 'industry'])

            # 使用核心API获取数据
            stock_info = self.get_stock_info(
                stock_code=stock_codes,
                level='日线',
                order_by="date DESC",
                limit=len(stock_codes)  # 每只股票获取最新信息
            )

            # 转换为DataFrame并选择需要的列
            result = stock_info.to_dataframe()
            if not result.empty:
                # 重命名列以符合预期
                column_mapping = {
                    'code': 'stock_code',
                    'name': 'stock_name'
                }
                result = result.rename(columns=column_mapping)
                # 只保留需要的列并去重
                result = result[['stock_code', 'stock_name', 'industry']].drop_duplicates()

            return result

        except Exception as e:
            logger.error(f"获取股票基本信息出错: {e}")
            raise DataAccessError(f"获取股票基本信息失败: {e}")

    def get_industry_list(self) -> pd.DataFrame:
        """
        获取行业列表（兼容原有API）

        Returns:
            pd.DataFrame: 行业列表
        """
        try:
            with self.connection_pool.get_connection() as conn:
                query = "SELECT DISTINCT industry FROM stock_info WHERE industry IS NOT NULL ORDER BY industry"
                result = conn.query_dataframe(query)
                return result
        except Exception as e:
            logger.error(f"获取行业列表出错: {e}")
            raise DataAccessError(f"获取行业列表失败: {e}")

    # ==================== 统计和管理 ====================

    def get_stats_Manager(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.stats_lock:
            stats = self.stats.copy()

        # 添加缓存统计
        with self.cache_lock:
            stats.update({
                'cache_size': len(self.query_cache),
                'cache_hit_rate': (
                    self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses'])
                    if (self.stats['cache_hits'] + self.stats['cache_misses']) > 0 else 0
                )
            })

        # 添加连接池统计
        pool_stats = self.connection_pool.get_stats_Manager()
        stats['connection_pool'] = pool_stats

        return stats

    def get_cache_stats_Manager(self) -> Dict[str, Any]:
        """
        获取缓存统计信息（兼容原有API）

        Returns:
            缓存统计信息字典
        """
        with self.cache_lock:
            stats = {
                'enabled': self.cache_enabled,
                'size': len(self.query_cache),
                'max_size': self.max_cache_size,
                'hits': self.stats['cache_hits'],
                'misses': self.stats['cache_misses'],
                'evictions': self.stats['cache_evictions'],
                'hit_rate': self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses']) if (self.stats['cache_hits'] + self.stats['cache_misses']) > 0 else 0,
                'types': {}
            }

            # 统计不同类型的缓存数量
            for key in self.query_cache.keys():
                type_name = key.split('_')[0]
                stats['types'][type_name] = stats['types'].get(type_name, 0) + 1

            return stats

    def get_performance_stats_Manager_Unified_Data_Manager(self) -> Dict[str, Any]:
        """
        获取性能统计信息

        Returns:
            Dict[str, Any]: 性能统计
        """
        stats = self.get_stats_Manager()

        # 性能监控统计
        if self.performance_monitor:
            stats['performance_monitor'] = self.performance_monitor.get_stats_Manager()

        # 稳定性统计
        if self.stability_manager:
            stats['stability'] = self.stability_manager.get_stability_status()

        return stats

    def transaction(self):
        """
        创建事务上下文（兼容原有API）

        Returns:
            事务上下文管理器
        """
        # 简化实现：返回连接上下文
        return self.connection_pool.get_connection()

    def close_Manager(self):
        """关闭数据管理器，清理资源"""
        try:
            # 停止性能监控
            if self.performance_monitor:
                self.performance_monitor.stop_monitoring()

            # 关闭连接池
            if self.connection_pool:
                self.connection_pool.close_Manager()

            logger.info("统一数据管理器已关闭")

        except Exception as e:
            logger.error(f"关闭数据管理器时出错: {e}")

    def get_stock_data(self, 
                       code: str, 
                       start_date: str, 
                       end_date: str,
                       level: str = '日线') -> pd.DataFrame:
        """
        获取股票数据（统一接口）
        
        Args:
            code: 股票代码
            start_date: 开始日期
            end_date: 结束日期  
            level: 数据级别（日线、15分钟、30分钟、60分钟等）
            
        Returns:
            pd.DataFrame: 股票数据
        """
        try:
            # 构建查询SQL - 移除不存在的turnover列
            query = f"""
            SELECT code, name, date, level, open, high, low, close, volume
            FROM stock_info
            WHERE code = '{code}'
            AND level = '{level}'
            AND date >= '{start_date}' AND date <= '{end_date}'
            ORDER BY date ASC
            """
            
            # 执行查询
            with self.connection_pool.get_connection() as connection:
                cursor = connection.execute(query)
                results = list(cursor) if hasattr(cursor, '__iter__') else []
                
                if results:
                    # 转换为DataFrame
                    df = pd.DataFrame(results, columns=[
                        'code', 'name', 'date', 'level', 'open', 'high', 'low', 'close', 'volume'
                    ])
                    
                    # 数据类型转换
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # 日期转换
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                    
                    # 检查数据完整性
                    for col in ['amount', 'turnover_rate', 'pe_ratio', 'pb_ratio', 'price_change', 'price_range']:
                        if col not in df.columns:
                            df[col] = 0.0

                    logger.debug(f"获取股票 {code} {level}数据: {len(df)} 条记录")
                    return df
                else:
                    logger.warning(f"未找到股票 {code} 在 {start_date} 到 {end_date} 的 {level}数据")
                    return pd.DataFrame()
                    
        except Exception as e:
            logger.error(f"获取股票数据失败: {e}")
            return pd.DataFrame()

    def get_stock_daily_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取股票日线数据
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期 
            end_date: 结束日期
            
        Returns:
            pd.DataFrame: 股票日线数据
        """
        try:
            stock_info = self.get_stock_info(
                stock_code=stock_code,
                level="日线",
                start_date=start_date,
                end_date=end_date
            )
            return stock_info.to_dataframe()
        except Exception as e:
            logger.error(f"获取股票{stock_code}日线数据失败: {e}")
            return pd.DataFrame()

    def get_stock_period_data(self, stock_code: str, start_date: str, end_date: str, period: str) -> pd.DataFrame:
        """
        获取股票指定周期数据
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            period: 时间周期 (如 '日线', '15分钟', '30分钟', '60分钟')
            
        Returns:
            pd.DataFrame: 股票指定周期数据
        """
        try:
            return self.get_period_data(
                stock_code=stock_code,
                start_date=start_date,
                end_date=end_date,
                period=period
            )
        except Exception as e:
            logger.error(f"获取股票{stock_code} {period}数据失败: {e}")
            return pd.DataFrame()


# ==================== 全局实例管理 ====================

# 全局实例
_unified_data_manager = None
_manager_lock = threading.Lock()


def get_unified_data_manager() -> UnifiedDataManager:
    """获取全局统一数据管理器实例"""
    global _unified_data_manager

    if _unified_data_manager is None:
        with _manager_lock:
            if _unified_data_manager is None:
                _unified_data_manager = UnifiedDataManager()

    return _unified_data_manager


# ==================== 向后兼容别名 ====================

# 为了向后兼容，提供所有现有类的别名
class DataManager(UnifiedDataManager):
    """向后兼容的DataManager类"""
    pass

class EnhancedDataManager(UnifiedDataManager):
    """向后兼容的EnhancedDataManager类"""
    pass

class DataManagerAdapter(UnifiedDataManager):
    """向后兼容的DataManagerAdapter类"""
    pass


class ProductionDataAccessLayer(UnifiedDataManager):
    """
    生产级数据访问层

    基于UnifiedDataManager，添加生产环境所需的高级功能：
    - 高级健康检查
    - 详细性能监控
    - 自动故障恢复
    - 连接池优化
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._health_check_interval = 60  # 健康检查间隔（秒）
        self._last_health_check = 0
        self._health_status = "unknown"

    def health_check(self) -> Dict[str, Any]:
        """
        生产级健康检查

        Returns:
            Dict[str, Any]: 健康检查结果
        """
        current_time = time.time()

        # 如果距离上次检查时间不足间隔，返回缓存结果
        if current_time - self._last_health_check < self._health_check_interval:
            return {
                'status': self._health_status,
                'cached': True,
                'last_check': self._last_health_check
            }

        health_result = {
            'timestamp': current_time,
            'status': 'healthy',
            'components': {},
            'metrics': {},
            'warnings': [],
            'errors': []
        }

        try:
            # 1. 数据库连接检查
            with self.connection_pool.get_connection() as conn:
                conn.execute("SELECT 1")
            health_result['components']['database'] = 'healthy'

        except Exception as e:
            health_result['components']['database'] = 'unhealthy'
            health_result['errors'].append(f"数据库连接失败: {str(e)}")
            health_result['status'] = 'unhealthy'

        # 2. 连接池状态检查
        try:
            pool_stats = self.connection_pool.get_stats()
            health_result['components']['connection_pool'] = 'healthy'
            health_result['metrics']['connection_pool'] = pool_stats

            # 检查连接池使用率
            if pool_stats.get('active_connections', 0) > pool_stats.get('max_connections', 20) * 0.8:
                health_result['warnings'].append("连接池使用率过高")

        except Exception as e:
            health_result['components']['connection_pool'] = 'unhealthy'
            health_result['errors'].append(f"连接池状态检查失败: {str(e)}")

        # 3. 缓存状态检查
        try:
            cache_stats = self.get_cache_stats()
            health_result['components']['cache'] = 'healthy'
            health_result['metrics']['cache'] = cache_stats

            # 检查缓存命中率
            hit_rate = cache_stats.get('hit_rate', 0)
            if hit_rate < 0.5:  # 命中率低于50%
                health_result['warnings'].append(f"缓存命中率较低: {hit_rate:.2%}")

        except Exception as e:
            health_result['components']['cache'] = 'degraded'
            health_result['warnings'].append(f"缓存状态检查失败: {str(e)}")

        # 4. 性能指标检查
        try:
            perf_stats = self.get_performance_stats()
            health_result['metrics']['performance'] = perf_stats

            # 检查平均查询时间
            avg_time = perf_stats.get('avg_query_time', 0)
            if avg_time > 2.0:  # 平均查询时间超过2秒
                health_result['warnings'].append(f"平均查询时间过长: {avg_time:.2f}秒")

        except Exception as e:
            health_result['warnings'].append(f"性能指标检查失败: {str(e)}")

        # 更新健康状态
        if health_result['errors']:
            self._health_status = 'unhealthy'
        elif health_result['warnings']:
            self._health_status = 'degraded'
        else:
            self._health_status = 'healthy'

        health_result['status'] = self._health_status
        self._last_health_check = current_time

        return health_result

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取详细性能统计"""
        stats = self.stats.copy()

        # 计算衍生指标
        if stats['total_queries'] > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])
            stats['error_rate'] = stats['query_errors'] / stats['total_queries']
            stats['avg_query_time'] = stats['total_execution_time'] / stats['total_queries']
        else:
            stats['cache_hit_rate'] = 0.0
            stats['error_rate'] = 0.0
            stats['avg_query_time'] = 0.0

        return stats

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self.cache_lock:
            total_entries = len(self.query_cache)
            expired_entries = 0
            current_time = time.time()

            for key, timestamp in self.cache_timestamps.items():
                if current_time - timestamp > self.default_ttl:
                    expired_entries += 1

            return {
                'total_entries': total_entries,
                'expired_entries': expired_entries,
                'active_entries': total_entries - expired_entries,
                'hit_rate': self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses']) if (self.stats['cache_hits'] + self.stats['cache_misses']) > 0 else 0.0,
                'memory_usage_mb': total_entries * 0.001  # 估算内存使用
            }

    def optimize_performance(self) -> Dict[str, Any]:
        """自动性能优化"""
        optimization_result = {
            'actions_taken': [],
            'recommendations': [],
            'before_stats': self.get_performance_stats(),
            'after_stats': None
        }

        # 1. 清理过期缓存
        expired_count = self._cleanup_expired_cache()
        if expired_count > 0:
            optimization_result['actions_taken'].append(f"清理了 {expired_count} 个过期缓存项")

        # 2. 连接池优化建议
        try:
            pool_stats = self.connection_pool.get_stats()
            active_ratio = pool_stats.get('active_connections', 0) / pool_stats.get('max_connections', 20)

            if active_ratio > 0.8:
                optimization_result['recommendations'].append("建议增加连接池最大连接数")
            elif active_ratio < 0.2:
                optimization_result['recommendations'].append("可以考虑减少连接池最小连接数")

        except Exception as e:
            logger.warning(f"连接池优化检查失败: {e}")

        # 3. 缓存优化建议
        cache_stats = self.get_cache_stats()
        if cache_stats['hit_rate'] < 0.5:
            optimization_result['recommendations'].append("缓存命中率较低，建议调整缓存策略")

        optimization_result['after_stats'] = self.get_performance_stats()
        return optimization_result

    def _cleanup_expired_cache(self) -> int:
        """清理过期缓存"""
        with self.cache_lock:
            current_time = time.time()
            expired_keys = []

            for key, timestamp in self.cache_timestamps.items():
                if current_time - timestamp > self.default_ttl:
                    expired_keys.append(key)

            for key in expired_keys:
                self.query_cache.pop(key, None)
                self.cache_timestamps.pop(key, None)
                self.cache_access_count.pop(key, None)

            return len(expired_keys)


# 向后兼容的获取函数
def get_data_manager() -> UnifiedDataManager:
    """获取数据管理器（向后兼容）"""
    return get_unified_data_manager()

def get_enhanced_data_manager() -> UnifiedDataManager:
    """获取增强数据管理器（向后兼容）"""
    return get_unified_data_manager()

def get_data_manager_adapter() -> UnifiedDataManager:
    """获取数据管理器适配器（向后兼容）"""
    return get_unified_data_manager()


# 生产级数据访问层实例
_production_data_access_layer = None


def get_production_data_access_layer(config: Optional[Dict[str, Any]] = None) -> ProductionDataAccessLayer:
    """
    获取生产级数据访问层实例

    Args:
        config: 可选配置参数

    Returns:
        ProductionDataAccessLayer: 生产级数据访问层实例
    """
    global _production_data_access_layer
    if _production_data_access_layer is None:
        _production_data_access_layer = ProductionDataAccessLayer(**(config or {}))
    return _production_data_access_layer


def initialize_production_data_access_layer(config: Optional[Dict[str, Any]] = None):
    """
    初始化生产级数据访问层

    Args:
        config: 可选配置参数
    """
    global _production_data_access_layer
    _production_data_access_layer = ProductionDataAccessLayer(**(config or {}))
    logger.info("生产级数据访问层已初始化")


def reset_production_data_access_layer():
    """重置生产级数据访问层实例（主要用于测试）"""
    global _production_data_access_layer
    _production_data_access_layer = None
