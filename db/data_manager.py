"""
数据管理器模块

负责从数据库获取数据，并提供高效的缓存机制
"""

import time
import threading
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
import pandas as pd
import json
import hashlib

from db.clickhouse_db import get_clickhouse_db
from utils.logger import get_logger
from utils.decorators import singleton, performance_monitor, safe_run, cache_result
from utils.exceptions import DataAccessError, DataNotFoundError, DataValidationError
from enums.period import Period

logger = get_logger(__name__)


@singleton
class DataManager:
    """
    数据管理器，负责数据的获取和缓存
    
    使用单例模式确保系统中只有一个数据管理器实例
    """
    
    def __init__(self, cache_enabled=True, max_cache_size=1000, default_ttl=3600):
        """
        初始化数据管理器
        
        Args:
            cache_enabled: 是否启用缓存
            max_cache_size: 最大缓存条目数
            default_ttl: 默认缓存有效期（秒）
        """
        self.db = get_clickhouse_db()
        self.cache_enabled = cache_enabled
        self.default_ttl = default_ttl
        self.max_cache_size = max_cache_size
        
        # 缓存数据
        self.cache = {}
        self.cache_timestamps = {}
        self.cache_access_count = {}
        
        # 缓存锁，确保线程安全
        self.cache_lock = threading.RLock()
        
        # 缓存统计
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions = 0
        
        logger.info(f"数据管理器初始化完成，缓存{'启用' if cache_enabled else '禁用'}，"
                   f"最大缓存条目数: {max_cache_size}，默认缓存有效期: {default_ttl}秒")
    
    @performance_monitor(threshold=0.5)
    def get_kline_data(self, stock_code: str, period: Union[str, Period], 
                       start_date: str = None, end_date: str = None, 
                       fields: Optional[List[str]] = None,
                       cache_ttl: Optional[int] = None) -> pd.DataFrame:
        """
        获取K线数据
        
        Args:
            stock_code: 股票代码
            period: 周期(DAILY, WEEKLY, MONTHLY, MIN_60, MIN_30, MIN_15, MIN_5)
            start_date: 开始日期，默认为end_date前30个交易日
            end_date: 结束日期，默认为当前日期
            fields: 字段列表，默认为所有字段
            cache_ttl: 缓存有效期（秒），None表示使用默认值
            
        Returns:
            K线数据DataFrame
            
        Raises:
            DataAccessError: 数据访问错误
            DataNotFoundError: 数据不存在
            DataValidationError: 参数验证错误
        """
        try:
            # 参数验证
            if not stock_code:
                raise DataValidationError("股票代码不能为空")
            
            # 标准化周期格式
            if isinstance(period, str):
                try:
                    period = Period.from_string(period)
                except ValueError:
                    raise DataValidationError(f"不支持的周期类型: {period}", 
                                             {"supported_periods": Period.get_all_period_values()})
            
            # 处理日期参数
            if end_date is None:
                end_date = datetime.now().strftime("%Y-%m-%d")
                
            if start_date is None:
                # 默认获取30个周期的数据
                if period in [Period.DAILY, Period.MIN_60, Period.MIN_30, Period.MIN_15, Period.MIN_5]:
                    days_delta = 30
                elif period == Period.WEEKLY:
                    days_delta = 180  # 约26周
                elif period == Period.MONTHLY:
                    days_delta = 365  # 约12个月
                else:
                    days_delta = 30
                    
                start_date = (datetime.strptime(end_date, "%Y-%m-%d") - 
                             timedelta(days=days_delta)).strftime("%Y-%m-%d")
            
            # 获取周期的中文描述
            period_desc = {
                Period.DAILY: "日线",
                Period.WEEKLY: "周线",
                Period.MONTHLY: "月线",
                Period.MIN_60: "60分钟线",
                Period.MIN_30: "30分钟线",
                Period.MIN_15: "15分钟线",
                Period.MIN_5: "5分钟线"
            }.get(period, "日线")
            
            # 构建查询SQL - 使用stock_info表
            if fields is None:
                fields = ["date", "open", "high", "low", "close", "volume", "turnover_rate", "price_change"]
                
            # 确保必要的字段存在
            if "date" not in fields:
                fields = ["date"] + fields
            
            if "open" not in fields:
                fields.append("open")
            
            if "high" not in fields:
                fields.append("high")
                
            if "low" not in fields:
                fields.append("low")
                
            if "close" not in fields:
                fields.append("close")
                
            if "volume" not in fields and "vol" not in fields:
                fields.append("volume")
                
            # 移除amount字段，因为数据库中不存在
            fields = [f for f in fields if f != 'amount']
                
            field_str = ', '.join(fields)
            
            # 修复：确保WHERE子句只出现一次
            sql = f"""
            SELECT {field_str}
            FROM stock_info
            WHERE code = '{stock_code}'
              AND level = '{period_desc}'
              AND date >= '{start_date}'
              AND date <= '{end_date}'
            ORDER BY date
            """
            
            # 获取缓存有效期
            ttl = cache_ttl if cache_ttl is not None else self.default_ttl
            
            # 检查缓存
            cached_data = self._get_from_cache(sql, ttl)
            if cached_data is not None:
                return cached_data
            
            # 执行查询
            logger.debug(f"执行SQL查询: {sql}")
            try:
                result = self.db.query(sql)
                
                # 检查结果
                if result is None or len(result) == 0:
                    logger.warning(f"未找到股票 {stock_code} 在 {period_desc} 周期从 {start_date} 到 {end_date} 的K线数据")
                    return pd.DataFrame(columns=fields)
                
                # 检查和处理列名
                if len(result.columns) > 0:
                    logger.debug(f"查询结果列名: {result.columns.tolist()}")
                    
                    # 标准化列名 - 处理可能的前缀和格式
                    column_mapping = {}
                    
                    # 处理表名前缀
                    for col in result.columns:
                        for prefix in ["stock_info.", "kline_daily.", "kline."]:
                            if col.startswith(prefix):
                                clean_name = col[len(prefix):]
                                column_mapping[col] = clean_name
                    
                    # 处理col_0, col_1等格式的列名
                    if 'col_0' in result.columns:
                        expected_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'turnover_rate', 'price_change']
                        for i, expected_col in enumerate(expected_cols):
                            col_name = f'col_{i}'
                            if col_name in result.columns and i < len(expected_cols):
                                column_mapping[col_name] = expected_cols[i]
                    
                    # 应用列名映射
                    if column_mapping:
                        logger.info(f"应用列名映射: {column_mapping}")
                        result = result.rename(columns=column_mapping)
                        
                    # 确保所有必需的列都存在，进行列名小写匹配
                    required_cols = ['open', 'high', 'low', 'close']
                    lower_cols = {col.lower(): col for col in result.columns}
                    
                    for req_col in required_cols:
                        if req_col not in result.columns and req_col.lower() in lower_cols:
                            result[req_col] = result[lower_cols[req_col.lower()]]
                
                # 更新缓存
                self._add_to_cache(sql, result)
                
                return result
            except Exception as e:
                logger.error(f"获取K线数据出错: {str(e)}")
                return pd.DataFrame(columns=fields)
        except Exception as e:
            if isinstance(e, (DataValidationError, DataNotFoundError)):
                raise
            logger.error(f"获取K线数据出错: {e}")
            raise DataAccessError(f"获取K线数据失败: {str(e)}", 
                                 {"stock_code": stock_code, "period": str(period), 
                                  "start_date": start_date, "end_date": end_date})
    
    @performance_monitor(threshold=0.5)
    def get_stock_list(self, filters: Optional[Dict[str, Any]] = None, 
                       cache_ttl: Optional[int] = None) -> pd.DataFrame:
        """
        获取股票列表
        
        Args:
            filters: 过滤条件
            cache_ttl: 缓存有效期（秒），None表示使用默认值
            
        Returns:
            股票列表DataFrame
            
        Raises:
            DataAccessError: 数据访问错误
        """
        try:
            # 构建缓存键
            filters_str = json.dumps(filters, sort_keys=True) if filters else "none"
            cache_key = f"stock_list_{filters_str}"
            
            # 获取缓存有效期
            ttl = cache_ttl if cache_ttl is not None else self.default_ttl
            
            # 检查缓存
            cached_data = self._get_from_cache(cache_key, ttl)
            if cached_data is not None:
                return cached_data
            
            # 获取价格过滤条件
            price = None
            if filters and 'price' in filters and isinstance(filters['price'], dict):
                price = filters['price']
            
            # 使用stock_info表代替stock_info表
            # 通过最新日期的日线数据获取股票列表 - 优化查询以减少内存使用
            # 使用更直接的方式，不使用子查询来减少内存占用
            sql = f"""
            SELECT 
                code as stock_code, 
                name as stock_name,
                '' as market,
                industry,
                0 as market_cap,
                close as last_price,
                0 as pe_ratio,
                0 as pb_ratio,
                turnover_rate
            FROM stock_info
            WHERE level = '日线'
            AND date >= toDate(now()) - INTERVAL 7 DAY  -- 限制只查询最近7天的数据
            {" AND close <= " + str(price['max']) if price and 'max' in price and price['max'] > 0 else ""}
            {" AND close >= " + str(price['min']) if price and 'min' in price and price['min'] > 0 else ""}
            ORDER BY date DESC, code ASC
            LIMIT 100  -- 限制结果集大小
            """
            
            # 执行查询
            logger.debug(f"执行SQL: {sql}")
            result = self.db.query(sql)
            
            # 检查结果列名并修复（如果需要）
            if not result.empty:
                # 检查是否需要重命名列
                if 'stock_code' not in result.columns and 'col_0' in result.columns:
                    # 创建列映射
                    expected_columns = ['stock_code', 'stock_name', 'market', 'industry', 
                                      'market_cap', 'last_price', 'pe_ratio', 'pb_ratio', 'turnover_rate']
                    actual_columns = result.columns.tolist()
                    
                    # 构建映射字典
                    column_mapping = {}
                    for i, expected_col in enumerate(expected_columns):
                        if i < len(actual_columns):
                            column_mapping[actual_columns[i]] = expected_col
                    
                    logger.debug(f"应用列名映射: {column_mapping}")
                    result = result.rename(columns=column_mapping)
                    
                logger.info(f"获取到 {len(result)} 支股票")
            else:
                logger.warning("获取股票列表结果为空")
            
            # 缓存结果
            self._set_cache(cache_key, result, ttl)
            
            return result
        except Exception as e:
            logger.error(f"获取股票列表出错: {e}")
            raise DataAccessError(f"获取股票列表失败: {e}")
    
    @performance_monitor(threshold=0.5)
    def save_selection_result(self, result: pd.DataFrame, strategy_id: str, 
                             selection_date: str = None) -> bool:
        """
        保存选股结果
        
        Args:
            result: 选股结果DataFrame
            strategy_id: 策略ID
            selection_date: 选股日期，默认为当前日期
            
        Returns:
            保存成功返回True，否则返回False
        
        Raises:
            DataAccessError: 数据访问错误
            DataValidationError: 参数验证错误
        """
        try:
            # 参数验证
            if result is None or len(result) == 0:
                logger.warning(f"选股结果为空，不保存")
                return True
                
            if not strategy_id:
                raise DataValidationError("策略ID不能为空")
                
            # 处理日期参数
            if selection_date is None:
                selection_date = datetime.now().strftime("%Y-%m-%d")
                
            # 处理结果数据
            result_copy = result.copy()
            
            # 确保必要字段存在
            required_fields = ['stock_code', 'stock_name', 'signal_strength', 'satisfied_conditions']
            for field in required_fields:
                if field not in result_copy.columns:
                    if field in ['stock_code', 'stock_name']:
                        raise DataValidationError(f"选股结果缺少必要字段: {field}")
                    else:
                        result_copy[field] = None
            
            # 将复杂类型转为JSON字符串
            for col in result_copy.columns:
                if result_copy[col].dtype == 'object' and col not in ['stock_code', 'stock_name']:
                    result_copy[col] = result_copy[col].apply(lambda x: json.dumps(x) if x is not None else None)
            
            # 添加选股信息
            result_copy['strategy_id'] = strategy_id
            result_copy['selection_date'] = selection_date
            
            # 构建插入SQL
            fields = ', '.join(result_copy.columns)
            placeholders = ', '.join(['%s'] * len(result_copy.columns))
            
            sql = f"""
            INSERT INTO stock_selection_result ({fields})
            VALUES ({placeholders})
            """
            
            # 构建数据
            data = [tuple(row) for row in result_copy.values]
            
            # 执行插入
            logger.debug(f"执行插入: {sql}")
            affected_rows = self.db.execute(sql, data)
            
            logger.info(f"已保存 {affected_rows} 条选股结果，策略ID: {strategy_id}，日期: {selection_date}")
            return True
        except Exception as e:
            logger.error(f"保存选股结果出错: {e}")
            raise DataAccessError(f"保存选股结果失败: {str(e)}")
    
    @performance_monitor(threshold=0.5)
    def get_selection_history(self, strategy_id: Optional[str] = None, 
                             start_date: Optional[str] = None, 
                             end_date: Optional[str] = None,
                             limit: int = 100) -> pd.DataFrame:
        """
        获取选股历史记录
        
        Args:
            strategy_id: 策略ID，None表示所有策略
            start_date: 开始日期，None表示不限制
            end_date: 结束日期，None表示当前日期
            limit: 返回记录数限制
            
        Returns:
            选股历史记录DataFrame
        
        Raises:
            DataAccessError: 数据访问错误
        """
        try:
            # 构建查询条件
            where_clause = "WHERE 1=1"
            
            if strategy_id:
                where_clause += f" AND strategy_id = '{strategy_id}'"
                
            if start_date:
                where_clause += f" AND selection_date >= '{start_date}'"
                
            if end_date:
                where_clause += f" AND selection_date <= '{end_date}'"
            elif not end_date:
                end_date = datetime.now().strftime("%Y-%m-%d")
                where_clause += f" AND selection_date <= '{end_date}'"
            
            # 构建查询SQL
            sql = f"""
            SELECT strategy_id, selection_date, COUNT(*) as stock_count
            FROM stock_selection_result
            {where_clause}
            GROUP BY strategy_id, selection_date
            ORDER BY selection_date DESC
            LIMIT {limit}
            """
            
            # 执行查询
            logger.debug(f"执行SQL查询: {sql}")
            result = self.db.query(sql)
            
            # 检查结果
            if result is None:
                logger.warning("未找到选股历史记录")
                return pd.DataFrame(columns=['strategy_id', 'selection_date', 'stock_count'])
            
            return result
        except Exception as e:
            logger.error(f"获取选股历史记录出错: {e}")
            raise DataAccessError(f"获取选股历史记录失败: {str(e)}")
    
    @performance_monitor(threshold=0.5)
    def get_selection_result(self, strategy_id: str, selection_date: str) -> pd.DataFrame:
        """
        获取指定日期的选股结果
        
        Args:
            strategy_id: 策略ID
            selection_date: 选股日期
            
        Returns:
            选股结果DataFrame
        
        Raises:
            DataAccessError: 数据访问错误
            DataValidationError: 参数验证错误
        """
        try:
            # 参数验证
            if not strategy_id:
                raise DataValidationError("策略ID不能为空")
                
            if not selection_date:
                raise DataValidationError("选股日期不能为空")
                
            # 构建查询SQL
            sql = f"""
            SELECT *
            FROM stock_selection_result
            WHERE strategy_id = '{strategy_id}'
              AND selection_date = '{selection_date}'
            ORDER BY signal_strength DESC
            """
            
            # 执行查询
            logger.debug(f"执行SQL查询: {sql}")
            result = self.db.query(sql)
            
            # 检查结果
            if result is None or len(result) == 0:
                logger.warning(f"未找到策略 {strategy_id} 在 {selection_date} 的选股结果")
                return pd.DataFrame()
            
            # 处理JSON字段
            for col in result.columns:
                if result[col].dtype == 'object' and col not in ['stock_code', 'stock_name', 'strategy_id', 'selection_date']:
                    try:
                        result[col] = result[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
                    except:
                        pass
            
            return result
        except Exception as e:
            if isinstance(e, DataValidationError):
                raise
            logger.error(f"获取选股结果出错: {e}")
            raise DataAccessError(f"获取选股结果失败: {str(e)}")
    
    def clear_cache(self, pattern: str = None):
        """
        清除缓存
        
        Args:
            pattern: 缓存键匹配模式，None表示清除所有缓存
        """
        with self.cache_lock:
            if pattern is None:
                old_size = len(self.cache)
                self.cache.clear()
                self.cache_timestamps.clear()
                self.cache_access_count.clear()
                logger.info(f"已清除所有缓存，共 {old_size} 项")
            else:
                keys_to_remove = [k for k in self.cache.keys() if pattern in k]
                for key in keys_to_remove:
                    self.cache.pop(key, None)
                    self.cache_timestamps.pop(key, None)
                    self.cache_access_count.pop(key, None)
                logger.info(f"已清除匹配 '{pattern}' 的缓存，共 {len(keys_to_remove)} 项")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            缓存统计信息字典
        """
        with self.cache_lock:
            stats = {
                'enabled': self.cache_enabled,
                'size': len(self.cache),
                'max_size': self.max_cache_size,
                'hits': self.cache_hits,
                'misses': self.cache_misses,
                'evictions': self.cache_evictions,
                'hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
                'types': {}
            }
            
            # 统计不同类型的缓存数量
            for key in self.cache.keys():
                type_name = key.split('_')[0]
                stats['types'][type_name] = stats['types'].get(type_name, 0) + 1
                
            return stats
    
    def transaction(self):
        """
        创建事务上下文
        
        Returns:
            事务上下文管理器
        """
        return self.db.transaction()
    
    def _get_from_cache(self, key: str, ttl: int) -> Optional[Any]:
        """
        从缓存中获取数据
        
        Args:
            key: 缓存键
            ttl: 缓存有效期（秒）
            
        Returns:
            缓存的数据，不存在或已过期返回None
        """
        if not self.cache_enabled:
            return None
            
        with self.cache_lock:
            # 检查缓存是否存在
            if key not in self.cache:
                self.cache_misses += 1
                return None
                
            # 检查缓存是否过期
            current_time = time.time()
            if current_time - self.cache_timestamps.get(key, 0) > ttl:
                # 缓存已过期，删除
                self.cache.pop(key, None)
                self.cache_timestamps.pop(key, None)
                self.cache_access_count.pop(key, None)
                self.cache_misses += 1
                return None
                
            # 更新访问计数
            self.cache_access_count[key] = self.cache_access_count.get(key, 0) + 1
            self.cache_hits += 1
            
            return self.cache[key]
    
    def _add_to_cache(self, key: str, value: Any):
        """
        添加数据到缓存
        
        Args:
            key: 缓存键
            value: 要缓存的数据
        """
        if not self.cache_enabled:
            return
            
        with self.cache_lock:
            # 检查缓存大小是否超限
            if len(self.cache) >= self.max_cache_size and key not in self.cache:
                # 缓存已满，删除最不常用的项
                self._evict_cache_item()
            
            # 更新缓存
            self.cache[key] = value
            self.cache_timestamps[key] = time.time()
            self.cache_access_count[key] = 1
    
    def _evict_cache_item(self):
        """驱逐最不常用的缓存项"""
        if not self.cache:
            return
            
        # 查找访问次数最少的项
        min_key = min(self.cache_access_count.items(), key=lambda x: x[1])[0]
        
        # 删除该项
        self.cache.pop(min_key, None)
        self.cache_timestamps.pop(min_key, None)
        self.cache_access_count.pop(min_key, None)
        
        self.cache_evictions += 1
        logger.debug(f"缓存驱逐: {min_key}")
    
    @performance_monitor(threshold=0.5)
    def get_stock_info(self, stock_code: str, level: str = 'day', 
                       start_date: Optional[str] = None, 
                       end_date: Optional[str] = None,
                       limit: Optional[int] = None,
                       cache_ttl: Optional[int] = None) -> pd.DataFrame:
        """
        获取股票K线数据
        
        Args:
            stock_code: 股票代码
            level: K线周期，day/week/month
            start_date: 开始日期，格式YYYY-MM-DD
            end_date: 结束日期，格式YYYY-MM-DD
            limit: 限制返回条数
            cache_ttl: 缓存有效期（秒），None表示使用默认值
            
        Returns:
            K线数据DataFrame
            
        Raises:
            DataAccessError: 数据访问错误
        """
        try:
            # 将level转换为数据库中的实际级别
            level_map = {
                'day': '日线',
                'week': '周线',
                'month': '月线'
            }
            db_level = level_map.get(level, '日线')
            
            # 构建缓存键
            cache_params = {
                "stock_code": stock_code,
                "level": level,
                "start_date": start_date,
                "end_date": end_date,
                "limit": limit
            }
            cache_key = f"stock_info_{hashlib.md5(json.dumps(cache_params, sort_keys=True).encode()).hexdigest()}"
            
            # 获取缓存有效期
            ttl = cache_ttl if cache_ttl is not None else self.default_ttl
            
            # 检查缓存
            cached_data = self._get_from_cache(cache_key, ttl)
            if cached_data is not None:
                return cached_data
            
            # 构建查询条件
            conditions = []
            conditions.append(f"code = '{stock_code}'")
            conditions.append(f"level = '{db_level}'")
            
            if start_date:
                conditions.append(f"date >= '{start_date}'")
            if end_date:
                conditions.append(f"date <= '{end_date}'")
            
            where_clause = " AND ".join(conditions)
            
            # 构建SQL语句
            sql = f"""
            SELECT
                code,
                date,
                open,
                high,
                low,
                close,
                volume,
                turnover_rate,
                price_change,
                price_range
            FROM stock_info
            WHERE {where_clause}
            ORDER BY date DESC
            """
            
            if limit:
                sql += f" LIMIT {limit}"
            
            # 执行查询
            logger.debug(f"执行SQL查询: {sql}")
            try:
                result = self.db.query(sql)
                
                # 检查结果
                if result is None:
                    logger.warning(f"查询返回None, 股票: {stock_code}, 周期: {db_level}, 时间: {start_date} 到 {end_date}")
                    return pd.DataFrame(columns=['code', 'date', 'open', 'high', 'low', 'close', 'volume', 'turnover_rate', 'change', 'pct_chg'])
                
                if isinstance(result, pd.DataFrame):
                    if len(result) == 0:
                        logger.warning(f"未找到股票 {stock_code} 在 {db_level} 周期从 {start_date} 到 {end_date} 的K线数据")
                        return pd.DataFrame(columns=['code', 'date', 'open', 'high', 'low', 'close', 'volume', 'turnover_rate', 'change', 'pct_chg'])
                    
                    # 确保结果有正确的列名
                    if 'col_0' in result.columns:
                        # 创建映射
                        column_mapping = {}
                        for i, field in enumerate(fields):
                            if f'col_{i}' in result.columns:
                                # 从field中提取列名（处理类似"code as stock_code"的情况）
                                if ' as ' in field:
                                    col_name = field.split(' as ')[1].strip()
                                else:
                                    col_name = field.strip()
                                column_mapping[f'col_{i}'] = col_name
                        
                        if column_mapping:
                            result = result.rename(columns=column_mapping)
                else:
                    logger.warning(f"查询结果不是DataFrame, 是: {type(result)}, 股票: {stock_code}")
                    return pd.DataFrame(columns=['code', 'date', 'open', 'high', 'low', 'close', 'volume', 'turnover_rate', 'change', 'pct_chg'])
                
                # 缓存结果
                self._set_cache(cache_key, result, ttl)
                
                return result
            except Exception as e:
                logger.error(f"获取股票K线数据出错: {e}")
                return pd.DataFrame(columns=['code', 'date', 'open', 'high', 'low', 'close', 'volume', 'turnover_rate', 'change', 'pct_chg'])
        except Exception as e:
            logger.error(f"获取股票K线数据出错: {e}")
            raise DataAccessError(f"获取股票K线数据失败: {e}")
    
    def _set_cache(self, key: str, value: Any, ttl: int) -> None:
        """
        设置缓存
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 缓存有效期（秒）
        """
        if self.cache_enabled:
            # 如果缓存已满，先清理
            if len(self.cache) >= self.max_cache_size:
                self._evict_cache_item()
                
            # 设置缓存
            self.cache[key] = value
            self.cache_timestamps[key] = time.time()
            self.cache_access_count[key] = 1
            logger.debug(f"缓存已设置: {key}")
    
    def _build_filter_clause(self, filters: Optional[Dict[str, Any]]) -> str:
        """
        构建过滤条件
        
        Args:
            filters: 过滤条件
            
        Returns:
            WHERE子句
        """
        if not filters:
            return " WHERE 1=1"
            
        conditions = []
        
        # 市场过滤
        if 'market' in filters and filters['market']:
            markets = filters['market']
            if isinstance(markets, list) and markets:
                market_conditions = []
                for market in markets:
                    market_conditions.append(f"market = '{market}'")
                if market_conditions:
                    conditions.append(f"({' OR '.join(market_conditions)})")
        
        # 行业过滤
        if 'industry' in filters and filters['industry']:
            industries = filters['industry']
            if isinstance(industries, list) and industries:
                industry_conditions = []
                for industry in industries:
                    industry_conditions.append(f"industry = '{industry}'")
                if industry_conditions:
                    conditions.append(f"({' OR '.join(industry_conditions)})")
        
        # 市值过滤
        if 'market_cap' in filters:
            market_cap = filters['market_cap']
            if isinstance(market_cap, dict):
                if 'min' in market_cap and market_cap['min'] > 0:
                    conditions.append(f"market_cap >= {market_cap['min']}")
                if 'max' in market_cap and market_cap['max'] > 0:
                    conditions.append(f"market_cap <= {market_cap['max']}")
        
        # 价格过滤
        if 'price' in filters:
            price = filters['price']
            if isinstance(price, dict):
                if 'min' in price and price['min'] > 0:
                    conditions.append(f"close >= {price['min']}")
                if 'max' in price and price['max'] > 0:
                    conditions.append(f"close <= {price['max']}")
        
        # 拼接条件
        if conditions:
            return " WHERE " + " AND ".join(conditions)
        else:
            return " WHERE 1=1" 