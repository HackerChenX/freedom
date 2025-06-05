#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
周期管理器模块

管理不同周期的数据需求、获取、转换和缓存
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Any, Tuple
from enum import Enum
import datetime

from db.db_manager import DBManager
from utils.cache import LRUCache
from utils.logger import get_logger
from enums.period import Period

logger = get_logger(__name__)


class PeriodManager:
    """
    周期管理器类
    
    负责管理不同周期的数据需求、获取、转换和缓存
    确保各个周期的数据隔离，并优化数据获取性能
    """

    # 单例实例
    _instance = None

    def __new__(cls, *args, **kwargs):
        """确保单例模式"""
        if cls._instance is None:
            cls._instance = super(PeriodManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, cache_size: int = 100):
        """
        初始化周期管理器
        
        Args:
            cache_size: 每个周期的缓存大小，默认为100
        """
        # 单例模式下只初始化一次
        if self._initialized:
            return

        self.db_manager = DBManager.get_instance()
        self._initialized = True

        # 针对每个周期的数据缓存
        self.data_cache = {
            Period.MIN_5: LRUCache(cache_size),
            Period.MIN_15: LRUCache(cache_size),
            Period.MIN_30: LRUCache(cache_size),
            Period.MIN_60: LRUCache(cache_size),
            Period.DAILY: LRUCache(cache_size),
            Period.WEEKLY: LRUCache(cache_size),
            Period.MONTHLY: LRUCache(cache_size)
        }

        # 各周期所需的最小数据量
        self.min_data_requirements = {
            Period.MIN_5: 200,
            Period.MIN_15: 200,
            Period.MIN_30: 200,
            Period.MIN_60: 200,
            Period.DAILY: 250,
            Period.WEEKLY: 150,
            Period.MONTHLY: 60
        }

        # 周期名称与枚举的映射
        self.period_map = {
            "5分钟": Period.MIN_5,
            "15分钟": Period.MIN_15,
            "30分钟": Period.MIN_30,
            "60分钟": Period.MIN_60,
            "日线": Period.DAILY,
            "周线": Period.WEEKLY,
            "月线": Period.MONTHLY
        }

        logger.info("周期管理器初始化完成")

    def get_data(self, stock_code: str, period: Union[str, Period],
                 end_date: Optional[str] = None, lookback_days: Optional[int] = None) -> pd.DataFrame:
        """
        获取指定股票和周期的数据
        
        Args:
            stock_code: 股票代码
            period: 周期类型，可以是Period枚举或字符串
            end_date: 结束日期，格式为YYYYMMDD，默认为当前日期
            lookback_days: 向前获取的天数，如果不指定则使用最小数据量
            
        Returns:
            pd.DataFrame: 包含OHLCV数据的DataFrame
        """
        # 转换周期类型
        if isinstance(period, str):
            period = Period.from_string(period)

        # 生成缓存键
        cache_key = f"{stock_code}_{period.value}"
        if end_date:
            cache_key += f"_{end_date}"

        # 检查缓存
        if cache_key in self.data_cache[period]:
            logger.debug(f"从缓存获取数据: {cache_key}")
            return self.data_cache[period][cache_key]

        # 确定需要获取的数据量
        if lookback_days is None:
            lookback_days = self.min_data_requirements[period]

        # 确定结束日期
        if end_date is None:
            end_date = datetime.datetime.now().strftime("%Y%m%d")

        # 从数据库获取数据
        try:
            table_name = Period.get_table_name(period)
            data = self._query_data_from_db(stock_code, table_name, end_date, lookback_days)

            # 缓存数据
            if data is not None and not data.empty:
                self.data_cache[period][cache_key] = data

            return data

        except Exception as e:
            logger.error(f"获取股票 {stock_code} 周期 {period.value} 的数据时出错: {e}")
            return pd.DataFrame()

    def _query_data_from_db(self, stock_code: str, table_name: str,
                            end_date: str, lookback_days: int) -> pd.DataFrame:
        """
        从数据库查询股票数据
        
        Args:
            stock_code: 股票代码
            table_name: 表名
            end_date: 结束日期
            lookback_days: 向前获取的天数
            
        Returns:
            pd.DataFrame: 查询结果
        """
        query = f"""
        SELECT 
            trade_date as date,
            open,
            high,
            low,
            close,
            volume
        FROM {table_name}
        WHERE ts_code = '{stock_code}'
          AND trade_date <= '{end_date}'
        ORDER BY trade_date DESC
        LIMIT {lookback_days}
        """

        try:
            data = self.db_manager.execute_query(query)
            if data is not None and not data.empty:
                # 按日期升序排序
                data = data.sort_values(by='date').reset_index(drop=True)
            return data
        except Exception as e:
            logger.error(f"执行查询时出错: {e}")
            return pd.DataFrame()

    def convert_period(self, data: pd.DataFrame, from_period: Period, to_period: Period) -> pd.DataFrame:
        """
        修正版周期转换，确保正确处理所有数据条目

        Args:
            data: 原始数据（需按seq排序）
            from_period: 原始周期
            to_period: 目标周期

        Returns:
            pd.DataFrame: 转换后的数据
        """
        # 相同周期直接返回副本
        if from_period == to_period:
            return data.copy()

        # 确保数据有序
        data = data.sort_values('seq')

        # 周期层级验证
        period_hierarchy = {
            Period.MIN_5: 1, Period.MIN_15: 2, Period.MIN_30: 3,
            Period.MIN_60: 4, Period.DAILY: 5, Period.WEEKLY: 6, Period.MONTHLY: 7
        }

        if period_hierarchy[from_period] > period_hierarchy[to_period]:
            logger.error(f"不支持从高周期 {from_period.value} 转换到低周期 {to_period.value}")
            return pd.DataFrame()

        # 关键修正：基于数据行数而不是seq值创建分组
        def create_minute_groups(df, from_per, to_per):
            """创建分钟级分组ID"""
            # 确定合并比例
            ratio_dict = {
                (Period.MIN_5, Period.MIN_30): 6,  # 5→30分钟: 6根
                (Period.MIN_5, Period.MIN_60): 12,  # 5→60分钟: 12根
                (Period.MIN_15, Period.MIN_30): 2,  # 15→30分钟: 2根
                (Period.MIN_15, Period.MIN_60): 4,  # 15→60分钟: 4根
                (Period.MIN_30, Period.MIN_60): 2,  # 30→60分钟: 2根
                (Period.MIN_5, Period.DAILY): 288,  # 5分钟→日线
                (Period.MIN_15, Period.DAILY): 96,  # 15分钟→日线
                (Period.MIN_30, Period.DAILY): 48,  # 30分钟→日线
                (Period.MIN_60, Period.DAILY): 24,  # 60分钟→日线
            }

            # 获取合并比例
            ratio = ratio_dict.get((from_per, to_per))
            if ratio is None:
                raise ValueError(f"未定义的周期转换: {from_per.value}→{to_per.value}")

            # 关键修正：基于行号创建分组
            group_ids = np.arange(len(df)) // ratio
            return group_ids

        # 核心聚合函数（保持不变）
        def _agg_func(g):
            if g.empty:
                return pd.Series()

            last_row = g.iloc[-1]
            first_row = g.iloc[0]

            result = {
                'code': last_row['code'],
                'name': last_row['name'],
                'date': last_row['date'],
                'level': to_period.value,
                'open': first_row['open'],
                'high': g['high'].max(),
                'low': g['low'].min(),
                'close': last_row['close'],
                'volume': g['volume'].sum(),
                'seq': last_row['seq'],
                'datetime': last_row['datetime']
            }

            # 其他字段处理（保持不变）
            if 'turnover_rate' in g.columns:
                total_volume = g['volume'].sum()
                if total_volume > 0:
                    weighted_sum = (g['turnover_rate'] * g['volume']).sum()
                    result['turnover_rate'] = weighted_sum / total_volume
                else:
                    result['turnover_rate'] = g['turnover_rate'].mean()

            if 'price_change' in g.columns:
                result['price_change'] = last_row['close'] - first_row['open']

            if 'price_range' in g.columns:
                max_high = g['high'].max()
                min_low = g['low'].min()
                if first_row['open'] != 0:
                    result['price_range'] = (max_high - min_low) / first_row['open'] * 100
                else:
                    result['price_range'] = 0.0

            if 'industry' in g.columns:
                result['industry'] = last_row['industry']

            return pd.Series(result)

        # 尝试转换
        try:
            # 分钟级转换
            if to_period in [Period.MIN_5, Period.MIN_15, Period.MIN_30, Period.MIN_60, Period.DAILY]:
                # 关键修正：使用行索引创建分组
                data['group_id'] = create_minute_groups(data, from_period, to_period)
                grouped = data.groupby('group_id')

                # 执行转换
                result = grouped.apply(_agg_func).reset_index(drop=True)

            # 周线转换
            elif to_period == Period.WEEKLY:
                # 按ISO年周分组
                year = data['datetime'].dt.isocalendar().year
                week = data['datetime'].dt.isocalendar().week
                grouped = data.groupby([year, week])

                # 执行转换（使用相同的聚合函数）
                result = grouped.apply(_agg_func).reset_index(drop=True)

            # 月线转换
            elif to_period == Period.MONTHLY:
                # 按年月分组
                year = data['datetime'].dt.year
                month = data['datetime'].dt.month
                grouped = data.groupby([year, month])

                # 执行转换
                result = grouped.apply(_agg_func).reset_index(drop=True)

            else:
                raise ValueError(f"不支持的目标周期: {to_period}")

            # 确保字段顺序一致
            expected_cols = [
                "code", "name", "date", "level", "open", "high", "low", "close",
                "volume", "turnover_rate", "price_change", "price_range",
                "industry", "datetime", "seq"
            ]

            # 只返回存在的列
            return result[[col for col in expected_cols if col in result.columns]]

        except Exception as e:
            logger.error(f"周期转换失败: {str(e)}", exc_info=True)
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def clear_cache(self, period: Optional[Period] = None):
        """
        清除缓存
        
        Args:
            period: 指定要清除缓存的周期，如不指定则清除所有周期的缓存
        """
        if period is None:
            for p in self.data_cache:
                self.data_cache[p].clear()
            logger.info("已清除所有周期的缓存")
        else:
            self.data_cache[period].clear()
            logger.info(f"已清除周期 {period.value} 的缓存")

    def get_period_id(self, period: Union[str, Period]) -> str:
        """
        获取周期的唯一标识符
        
        Args:
            period: 周期类型，可以是Period枚举或字符串
            
        Returns:
            str: 周期的唯一标识符
        """
        if isinstance(period, str):
            period = Period.from_string(period)
        return period.value

    def get_indicators_period_id(self, indicator_name: str, period: Union[str, Period]) -> str:
        """
        获取指标和周期的组合标识符
        
        Args:
            indicator_name: 指标名称
            period: 周期类型，可以是Period枚举或字符串
            
        Returns:
            str: 指标和周期的组合标识符
        """
        period_id = self.get_period_id(period)
        return f"{indicator_name}_{period_id}"

    def get_min_data_required(self, period: Union[str, Period]) -> int:
        """
        获取指定周期需要的最小数据量
        
        Args:
            period: 周期类型，可以是Period枚举或字符串
            
        Returns:
            int: 最小数据量
        """
        if isinstance(period, str):
            period = Period.from_string(period)
        return self.min_data_requirements[period]

    def set_min_data_required(self, period: Union[str, Period], min_data: int):
        """
        设置指定周期需要的最小数据量
        
        Args:
            period: 周期类型，可以是Period枚举或字符串
            min_data: 最小数据量
        """
        if isinstance(period, str):
            period = Period.from_string(period)
        self.min_data_requirements[period] = min_data
        logger.debug(f"已设置周期 {period.value} 的最小数据量为 {min_data}")

    @staticmethod
    def get_instance() -> 'PeriodManager':
        """
        获取PeriodManager实例
        
        Returns:
            PeriodManager: 周期管理器实例
        """
        if PeriodManager._instance is None:
            PeriodManager()
        return PeriodManager._instance
