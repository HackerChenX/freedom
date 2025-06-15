#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
多周期数据处理器

提供多周期K线数据获取和转换功能，支持15分钟、30分钟、60分钟、日线、周线、月线
"""

import pandas as pd
from typing import Dict, List, Any, Optional
import os
import sys
from datetime import datetime

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from db.clickhouse_db import get_clickhouse_db
from utils.logger import get_logger
from utils.period_manager import PeriodManager
from enums.kline_period import KlinePeriod

logger = get_logger(__name__)

class PeriodDataProcessor:
    """多周期数据处理器"""
    
    def __init__(self):
        """初始化数据处理器"""
        self.db = get_clickhouse_db()
        self.period_manager = PeriodManager()
        self.data_cache = {}
    
    def get_multi_period_data(self, 
                            stock_code: str, 
                            end_date: str,
                            periods: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        获取多周期K线数据
        
        Args:
            stock_code: 股票代码
            end_date: 结束日期（买点日期）
            periods: 需要的周期列表，如['15min', '30min', '60min', 'daily', 'weekly', 'monthly']
            
        Returns:
            Dict[str, pd.DataFrame]: 各周期的K线数据字典
        """
        if periods is None:
            periods = ['15min', '30min', '60min', 'daily', 'weekly', 'monthly']
        
        # 缓存键
        cache_key = f"{stock_code}_{end_date}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        # 初始化结果字典
        result = {}
        
        try:
            # 首先获取15分钟数据
            if '15min' in periods:
                min15_data = self._get_kline_data(
                    stock_code=stock_code,
                    end_date=end_date,
                    period=KlinePeriod.MIN_15
                )
                result['15min'] = min15_data
            
            # 获取其他原生周期数据
            native_periods = {'daily': KlinePeriod.DAILY, 
                            'weekly': KlinePeriod.WEEKLY, 
                            'monthly': KlinePeriod.MONTHLY}
            
            for period_name, period_enum in native_periods.items():
                if period_name in periods:
                    period_data = self._get_kline_data(
                        stock_code=stock_code,
                        end_date=end_date,
                        period=period_enum
                    )
                    result[period_name] = period_data
            
            # 生成30分钟和60分钟数据
            if '30min' in periods and '15min' in result and not result['15min'].empty:
                min30_data = self._convert_period_data(
                    result['15min'], 
                    KlinePeriod.MIN_15, 
                    KlinePeriod.MIN_30
                )
                result['30min'] = min30_data
            
            if '60min' in periods and '15min' in result and not result['15min'].empty:
                min60_data = self._convert_period_data(
                    result['15min'], 
                    KlinePeriod.MIN_15, 
                    KlinePeriod.MIN_60
                )
                result['60min'] = min60_data
            
            # 缓存结果
            self.data_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"获取多周期数据时出错: {e}")
            return {}
    
    def _get_kline_data(self, 
                      stock_code: str, 
                      end_date: str, 
                      period: KlinePeriod) -> pd.DataFrame:
        """
        获取K线数据
        
        Args:
            stock_code: 股票代码
            end_date: 结束日期
            period: K线周期
            
        Returns:
            pd.DataFrame: K线数据
        """
        try:
            # 使用 get_stock_info 替代 get_kline_data
            stock_info = self.db.get_stock_info(
                stock_code=stock_code,
                level=period,
                end_date=end_date,
                order_by="date"
            )
            
            # 转换为DataFrame
            df = stock_info.to_dataframe()

            # 确保列名正确映射
            if not df.empty:
                logger.debug(f"原始DataFrame列名: {list(df.columns)}")

                # 标准化列名，确保使用英文列名
                df = self._standardize_column_names(df)
                logger.debug(f"标准化后DataFrame列名: {list(df.columns)}")

                # 添加衍生数据列，确保MA5、k等列存在
                df = self._add_derived_columns(df)
                logger.debug(f"添加衍生列后DataFrame列名: {list(df.columns)}")

            return df
            
        except Exception as e:
            logger.error(f"获取K线数据失败: {e}")
            return pd.DataFrame()

    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        标准化列名，确保使用英文列名

        Args:
            df: 原始DataFrame

        Returns:
            pd.DataFrame: 标准化列名后的DataFrame
        """
        if df.empty:
            return df

        # 列名映射：中文 -> 英文
        column_mapping = {
            '代码': 'code',
            '名称': 'name',
            '日期': 'date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '换手率': 'turnover_rate',
            '涨跌幅': 'price_change',
            '振幅': 'price_range',
            '行业': 'industry',
            '时间': 'datetime',
            '序号': 'seq'
        }

        # 应用列名映射
        df_copy = df.copy()
        for chinese_name, english_name in column_mapping.items():
            if chinese_name in df_copy.columns:
                df_copy = df_copy.rename(columns={chinese_name: english_name})

        # 确保基本的OHLCV列存在
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df_copy.columns:
                # 如果缺少某列，尝试用其他列填充或设为默认值
                if col in ['open', 'high', 'low'] and 'close' in df_copy.columns:
                    df_copy[col] = df_copy['close']
                elif col == 'volume':
                    df_copy[col] = 0
                else:
                    df_copy[col] = 0

        return df_copy

    def _add_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加衍生数据列，确保指标计算所需的列存在

        Args:
            df: 标准化后的DataFrame

        Returns:
            pd.DataFrame: 添加衍生列后的DataFrame
        """
        if df.empty:
            return df

        df_copy = df.copy()

        # 确保基础OHLCV列存在
        if 'close' in df_copy.columns:
            # 添加MA5列（5日移动平均线）
            if 'MA5' not in df_copy.columns:
                df_copy['MA5'] = df_copy['close'].rolling(window=5, min_periods=1).mean()

            # 添加MA10列（10日移动平均线）
            if 'MA10' not in df_copy.columns:
                df_copy['MA10'] = df_copy['close'].rolling(window=10, min_periods=1).mean()

            # 添加MA20列（20日移动平均线）
            if 'MA20' not in df_copy.columns:
                df_copy['MA20'] = df_copy['close'].rolling(window=20, min_periods=1).mean()

            # 添加更多常用的MA列
            for period in [30, 60]:
                ma_col = f'MA{period}'
                if ma_col not in df_copy.columns:
                    df_copy[ma_col] = df_copy['close'].rolling(window=period, min_periods=1).mean()

        # 添加KDJ相关的k、d、j列（简化版本）
        if all(col in df_copy.columns for col in ['high', 'low', 'close']):
            if 'k' not in df_copy.columns:
                # 计算RSV（未成熟随机值）
                low_min = df_copy['low'].rolling(window=9, min_periods=1).min()
                high_max = df_copy['high'].rolling(window=9, min_periods=1).max()
                rsv = (df_copy['close'] - low_min) / (high_max - low_min) * 100
                rsv = rsv.fillna(50)  # 填充NaN值

                # 计算K值（使用简化的移动平均）
                df_copy['k'] = rsv.rolling(window=3, min_periods=1).mean()

            if 'd' not in df_copy.columns and 'k' in df_copy.columns:
                # 计算D值
                df_copy['d'] = df_copy['k'].rolling(window=3, min_periods=1).mean()

            if 'j' not in df_copy.columns and all(col in df_copy.columns for col in ['k', 'd']):
                # 计算J值
                df_copy['j'] = 3 * df_copy['k'] - 2 * df_copy['d']

        # 添加成交量相关列
        if 'volume' in df_copy.columns:
            # 添加成交量移动平均
            if 'volume_ma5' not in df_copy.columns:
                df_copy['volume_ma5'] = df_copy['volume'].rolling(window=5, min_periods=1).mean()

        return df_copy

    def _get_stock_min_date(self, stock_code: str, period: KlinePeriod) -> str:
        """
        获取股票在特定周期下的最早日期
        
        Args:
            stock_code: 股票代码
            period: K线周期
            
        Returns:
            str: 最早日期，格式YYYY-MM-DD
        """
        try:
            # 使用数据库方法获取最早日期，而不是直接构建SQL
            min_date = self.db.get_stock_min_date(stock_code, period)
            if min_date:
                return min_date
            
            return None
            
        except Exception as e:
            logger.error(f"获取股票最早日期时出错: {e}")
            return None
    
    def _convert_period_data(self, 
                          data: pd.DataFrame, 
                          from_period: KlinePeriod, 
                          to_period: KlinePeriod) -> pd.DataFrame:
        """
        转换K线周期
        
        Args:
            data: 原始K线数据
            from_period: 原始周期
            to_period: 目标周期
            
        Returns:
            pd.DataFrame: 转换后的K线数据
        """
        try:
            # 如果数据为空，直接返回空DataFrame
            if data.empty:
                return pd.DataFrame()
            
            # 使用周期管理器转换周期
            converted_data = self.period_manager.convert_period(data, from_period, to_period)
            return converted_data
            
        except Exception as e:
            logger.error(f"转换周期数据时出错: {e}")
            return pd.DataFrame()

    def _format_date(self, date_str):
        """格式化日期字符串为标准格式"""
        try:
            # 如果日期为空或无效，返回合理的默认值
            if not date_str or date_str == '19700101' or date_str == '1970-01-01':
                return datetime.now().strftime('%Y-%m-%d')
                
            # 尝试解析YYYYMMDD格式
            if len(date_str) == 8 and date_str.isdigit():
                return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
            # 尝试解析YYYY-MM-DD格式
            elif len(date_str) == 10 and date_str[4] == '-' and date_str[7] == '-':
                return date_str
            else:
                logger.warning(f"不支持的日期格式: {date_str}，使用当前日期")
                return datetime.now().strftime('%Y-%m-%d')
        except Exception as e:
            logger.error(f"日期格式转换错误: {e}")
            return datetime.now().strftime('%Y-%m-%d')

    def get_stock_data(self, stock_code, start_date, end_date, period='daily'):
        """获取股票数据"""
        try:
            # 格式化日期
            start_date = self._format_date(start_date)
            end_date = self._format_date(end_date)
            
            # 转换周期格式为枚举
            if isinstance(period, str):
                period_enum = KlinePeriod.from_string(period)
            else:
                period_enum = period
            
            # 使用数据库的get_stock_info方法获取数据，不再直接构建SQL
            data = self.db.get_stock_info(
                stock_code=stock_code,
                level=period_enum,
                start_date=start_date,
                end_date=end_date,
                order_by="date"
            )
            
            # 如果返回的是StockInfo对象，转换为DataFrame
            if hasattr(data, 'to_dataframe'):
                data = data.to_dataframe()
            
            # 标准化列名
            if not data.empty:
                data = self._standardize_column_names(data)
                # 添加衍生数据列
                data = self._add_derived_columns(data)
                # 确保日期列是日期类型
                if 'date' in data.columns:
                    data['date'] = pd.to_datetime(data['date'])
            
            return data
            
        except Exception as e:
            logger.error(f"获取股票数据时出错: {e}")
            return pd.DataFrame()
    
    def get_stock_earliest_date(self, stock_code):
        """获取股票最早的数据日期"""
        try:
            # 使用数据库的get_stock_min_date方法获取最早日期，不再直接构建SQL
            min_date = self.db.get_stock_min_date(stock_code)
            
            if min_date:
                return self._format_date(min_date)
                    
            # 默认返回一个较早的日期
            return "2015-01-01"
            
        except Exception as e:
            logger.error(f"获取股票最早日期时出错: {e}")
            return "2015-01-01" 