from db.query_executor import get_query_executor
from db.sql_manager import QueryType
#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
数据管理器适配器

提供向后兼容的API接口，将现有的Data_manager调用适配到Enhanced_data_manager
"""

import time
import threading
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from db.unified_data_manager import get_unified_data_manager
from db.enhanced_connection_pool import initialize_connection_pool
from monitoring.performance_monitor import get_performance_monitor
from utils.stability_enhancer import get_stability_manager, retry
from utils.logger import getLogger
from utils.exceptions import DataAccessError, DataValidationError
from enums.period import Period
from models.stock_info import StockInfo

logger = getLogger(__name__)


class DatamanageradapterAdapter:
    """
    数据管理器适配器
    
    提供与原有Data_manager兼容的API接口，内部使用增强的数据管理器
    """
    
    def __init__(self, enable_monitoring: bool = True, enable_stability: bool = True):
        """
        初始化适配器
        
        Args:
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
                'host': 'localhost',
                'port': 9000,
                'database': 'stock'
            }

        # 初始化连接池
        self.connection_pool = initialize_connection_pool(
            host=db_config.get('host', 'localhost'),
            port=db_config.get('port', 9000),
            database=db_config.get('database', 'stock'),
            user=db_config.get('user', 'default'),
            password=db_config.get('password', ''),
            max_connections=20,
            min_connections=5
        )
        
        # 获取增强数据管理器
        self.enhanced_manager = get_unified_data_manager()
        
        # 性能监控
        self.performance_monitor = None
        if enable_monitoring:
            self.performance_monitor = get_performance_monitor()
            self.performance_monitor.start_monitoring()
        
        # 稳定性管理器
        self.stability_manager = None
        if enable_stability:
            self.stability_manager = get_stability_manager()
            self._setup_stability_features_Data_Manager_Adapter()
        
        logger.info("数据管理器适配器初始化完成，已启用优化功能")
    
    def _setup_stability_features_Data_Manager_Adapter(self):
        """设置稳定性功能"""
        if not self.stability_manager:
            return
        
        # 创建数据库查询熔断器
        self.db_circuit_breaker = self.stability_manager.create_circuit_breaker(
            'database_query',
            failure_threshold=5,
            recovery_timeout=60
        )
        
        # 注册降级服务
        def fallback_get_stock_data(stock_code: str, **kwargs):
            """数据获取降级服务"""
            logger.warning(f"使用降级服务获取股票数据: {stock_code}")
            # 返回空的Stock_info对象
            empty_df = pd.DataFrame()
            return Stock_info(empty_df)
        
        self.stability_manager.register_degradation(
            'get_stock_data',
            fallback_get_stock_data,
            lambda: False,  # 暂时不启用降级
            "数据获取服务降级"
        )
    
    @retry(max_attempts=3, delay=0.5)
    def get_stock_data_Adapter(self,
                      stock_code: str,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      period: str = 'daily',
                      limit: Optional[int] = None,
                      lookback_days: Optional[int] = None) -> pd.DataFrame:
        """
        获取股票数据（兼容原有API，支持历史数据查询优化）

        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            period: 周期
            limit: 限制记录数
            lookback_days: 向前获取的天数，用于技术指标计算

        Returns:
            pd.DataFrame: 股票数据
        """
        try:
            # 转换周期参数
            level = self._convert_period_to_level_Data_Manager_Adapter(period)

            # 如果是30分钟数据且数据库中不存在，尝试从15分钟数据计算
            if period in ['30min', 'min30'] and not self._has_30min_data_Data_Manager_Adapter(stock_code, start_date, end_date):
                logger.info(f"数据库中没有{stock_code}的30分钟数据，尝试从15分钟数据计算")
                return self._generate_30min_from_15min_Data_Manager_Adapter(stock_code, start_date, end_date, lookback_days)

            # 优化历史数据查询：根据周期和指标需求调整查询范围
            optimized_start_date, optimized_limit = self._optimize_data_query(
                period, start_date, end_date, limit, lookback_days
            )

            # 使用增强数据管理器获取数据
            stock_info = self.enhanced_manager.get_stock_info_Adapter(
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
                stock_info = self.enhanced_manager.get_stock_info_Adapter(
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
    
    def get_stock_info_Adapter(self, 
                      stock_code: Union[str, List[str]] = None,
                      level: Union[str, Period] = None,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      filters: Optional[Dict[str, Any]] = None,
                      limit: Optional[int] = None,
                                             order_by: str = "date DESC") -> StockInfo:
        """
        获取股票信息（增强API）
        
        Args:
            stock_code: 股票代码或股票代码列表
            level: K线周期
            start_date: 开始日期
            end_date: 结束日期
            filters: 过滤条件
            limit: 限制返回记录数
            order_by: 排序规则
            
        Returns:
            Stock_info: 股票数据对象
        """
        return self.enhanced_manager.get_stock_info_Adapter(
            stock_code=stock_code,
            level=level,
            start_date=start_date,
            end_date=end_date,
            filters=filters,
            limit=limit,
            order_by=order_by
        )
    
    def get_stock_list_Adapter(self,
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
            # 直接查询不重复的股票代码，而不是依赖于有限制的股票信息查询
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
                query_executor.get_stock_list()
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
    
    def get_stock_industry_Adapter(self, stock_code: str) -> Optional[str]:
        """
        获取股票行业（兼容原有API）
        
        Args:
            stock_code: 股票代码
            
        Returns:
            Optional[str]: 行业名称
        """
        try:
            stock_info = self.enhanced_manager.get_stock_info_Adapter(
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

    def get_stock_name_Adapter(self, stock_code: str) -> Optional[str]:
        """
        获取股票名称（兼容原有API）

        Args:
            stock_code: 股票代码

        Returns:
            Optional[str]: 股票名称
        """
        try:
            stock_info = self.enhanced_manager.get_stock_info_Adapter(
                stock_code=stock_code,
                limit=1
            )

            df = stock_info.to_dataframe()
            if not df.empty and 'name' in df.columns:
                return df['name'].iloc[0]
            else:
                # 如果没有名称字段，返回股票代码
                return stock_code

        except Exception as e:
            logger.debug(f"获取股票名称失败: {stock_code}, 错误: {e}")
            # 返回股票代码作为默认值
            return stock_code

    def get_previous_trade_date_Adapter(self, date: str, days: int = 1) -> str:
        """
        获取前N个交易日（兼容原有API）
        
        Args:
            date: 基准日期
            days: 往前天数
            
        Returns:
            str: 前N个交易日
        """
        try:
            # 简化实现：往前推算天数（实际应该查询交易日历）
            from datetime import datetime, timedelta
            
            base_date = datetime.strptime(date, '%Y-%m-%d')
            # 考虑周末，大概推算
            estimated_days = days * 1.4  # 考虑周末因子
            previous_date = base_date - timedelta(days=int(estimated_days))
            
            return previous_date.strftime('%Y-%m-%d')
            
        except Exception as e:
            logger.error(f"计算前一交易日失败: {date}, 错误: {e}")
            # 返回一个合理的默认值
            from datetime import datetime, timedelta
            base_date = datetime.strptime(date, '%Y-%m-%d')
            previous_date = base_date - timedelta(days=days * 2)
            return previous_date.strftime('%Y-%m-%d')
    
    def save_selection_result_Adapter(self, 
                            result: pd.DataFrame, 
                            strategy_id: str, 
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
            # 这里可以扩展保存逻辑
            logger.info(f"保存选股结果: 策略ID={strategy_id}, 记录数={len(result)}")
            
            # 简化实现：记录日志
            if selection_date is None:
                selection_date = datetime.now().strftime('%Y-%m-%d')
            
            logger.info(f"选股结果已保存: {strategy_id}, 日期: {selection_date}, 股票数: {len(result)}")
            return True
            
        except Exception as e:
            logger.error(f"保存选股结果失败: {e}")
            return False
    
    def _convert_period_to_level_Data_Manager_Adapter(self, period: str) -> str:
        """
        转换周期参数到level参数
        
        Args:
            period: 原有的周期参数
            
        Returns:
            str: 转换后的level参数
        """
        period_mapping = {
            'daily': 'DAILY',
            'day': 'DAILY',
            'weekly': 'WEEKLY',
            'week': 'WEEKLY',
            'monthly': 'MONTHLY',
            'month': 'MONTHLY',
            '15min': 'MIN_15',
            '30min': 'MIN_30',
            '60min': 'MIN_60',
            'min15': 'MIN_15',
            'min30': 'MIN_30',
            'min60': 'MIN_60'
        }
        
        return period_mapping.get(period.lower(), 'DAILY')

    def _has_30min_data_Data_Manager_Adapter(self, stock_code: str, start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> bool:
        """
        检查数据库中是否存在30分钟数据

        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            bool: 存在返回True，否则返回False
        """
        try:
            stock_info = self.enhanced_manager.get_stock_info_Adapter(
                stock_code=stock_code,
                level='30分钟',
                start_date=start_date,
                end_date=end_date,
                limit=1
            )
            df = stock_info.to_dataframe()
            return not df.empty
        except Exception as e:
            logger.debug(f"检查30分钟数据存在性失败: {e}")
            return False

    def _generate_30min_from_15min_Data_Manager_Adapter(self, stock_code: str, start_date: Optional[str] = None,
                                  end_date: Optional[str] = None, lookback_days: Optional[int] = None) -> pd.DataFrame:
        """
        从15分钟数据生成30分钟数据

        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            lookback_days: 向前获取的天数

        Returns:
            pd.DataFrame: 30分钟K线数据
        """
        try:
            # 扩大查询范围以获取足够的15分钟数据
            if lookback_days:
                from datetime import datetime, timedelta
                if start_date:
                    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                    extended_start = start_dt - timedelta(days=lookback_days)
                    extended_start_date = extended_start.strftime('%Y-%m-%d')
                else:
                    extended_start_date = start_date
            else:
                extended_start_date = start_date

            # 获取15分钟数据
            stock_info = self.enhanced_manager.get_stock_info_Adapter(
                stock_code=stock_code,
                level='15分钟',
                start_date=extended_start_date,
                end_date=end_date
            )

            df_15min = stock_info.to_dataframe()

            if df_15min.empty:
                logger.warning(f"没有找到{stock_code}的15分钟数据")
                return pd.DataFrame()

            # 确保有datetime列
            if 'datetime' not in df_15min.columns:
                if 'date' in df_15min.columns and 'time' in df_15min.columns:
                    df_15min['datetime'] = pd.to_datetime(df_15min['date'].astype(str) + ' ' + df_15min['time'].astype(str))
                elif 'date' in df_15min.columns:
                    df_15min['datetime'] = pd.to_datetime(df_15min['date'])
                else:
                    logger.error("无法构建datetime列")
                    return pd.DataFrame()

            # 设置datetime为索引
            df_15min = df_15min.set_index('datetime')
            df_15min.index = pd.to_datetime(df_15min.index)

            # 按30分钟重采样
            agg_dict = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }

            # 添加其他可能的字段
            for col in ['amount', 'turnover_rate', 'pe_ratio', 'pb_ratio']:
                if col in df_15min.columns:
                    if col == 'amount':
                        agg_dict[col] = 'sum'
                    else:
                        agg_dict[col] = 'last'

            # 重采样为30分钟
            df_30min = df_15min.resample('30T').agg(agg_dict)

            # 删除空值行
            df_30min = df_30min.dropna()

            # 重置索引，添加date和datetime列
            df_30min = df_30min.reset_index()
            df_30min['date'] = df_30min['datetime'].dt.date.astype(str)
            df_30min['time'] = df_30min['datetime'].dt.time.astype(str)

            # 添加股票代码
            df_30min['code'] = stock_code

            # 重新排列列顺序
            columns_order = ['code', 'date', 'datetime', 'time', 'open', 'high', 'low', 'close', 'volume']
            for col in df_30min.columns:
                if col not in columns_order:
                    columns_order.append(col)

            df_30min = df_30min[columns_order]

            logger.info(f"成功从15分钟数据生成30分钟数据: {stock_code}, 记录数: {len(df_30min)}")
            return df_30min

        except Exception as e:
            logger.error(f"从15分钟数据生成30分钟数据失败: {e}")
            return pd.DataFrame()

    def get_performance_stats_Adapter(self) -> Dict[str, Any]:
        """
        获取性能统计信息
        
        Returns:
            Dict[str, Any]: 性能统计
        """
        stats = {}
        
        # 数据管理器统计
        if self.enhanced_manager:
            stats['data_manager'] = self.enhanced_manager.get_stats()
        
        # 连接池统计
        if self.connection_pool:
            stats['connection_pool'] = self.connection_pool.get_stats()
        
        # 性能监控统计
        if self.performance_monitor:
            stats['performance_monitor'] = self.performance_monitor.get_stats()
        
        # 稳定性统计
        if self.stability_manager:
            stats['stability'] = self.stability_manager.get_stability_status()
        
        return stats
    
    def clear_cache_Adapter(self, pattern: Optional[str] = None):
        """
        清除缓存
        
        Args:
            pattern: 缓存模式，None表示清除所有
        """
        if self.enhanced_manager:
            self.enhanced_manager.clear_cache_Adapter(pattern)
            logger.info(f"缓存已清除: {pattern or '全部'}")
    
    def close_Adapter(self):
        """关闭适配器，清理资源"""
        try:
            # 停止性能监控
            if self.performance_monitor:
                self.performance_monitor.stop_monitoring()
            
            # 关闭连接池
            if self.connection_pool:
                self.connection_pool.close_Adapter()
            
            logger.info("数据管理器适配器已关闭")
            
        except Exception as e:
            logger.error(f"关闭适配器时出错: {e}")


# 全局适配器实例
_data_manager_adapter = None
_adapter_lock = threading.Lock()


def get_data_manager_adapter_Adapter() -> Data_manager_adapter:
    """获取全局数据管理器适配器实例"""
    global _data_manager_adapter
    
    if _data_manager_adapter is None:
        with _adapter_lock:
            if _data_manager_adapter is None:
                _data_manager_adapter = Data_manager_adapter_Adapter()
    
    return _data_manager_adapter


# 为了向后兼容，提供DataManager类的别名
class DatamanagerAdapter(Data_manager_adapter):
    """向后兼容的DataManager类"""
    pass
