#!/usr/bin/python
# -*- coding: UTF-8 -*-

import pandas as pd
from clickhouse_driver import Client
import datetime
import logging
from typing import Dict
import threading

# 导入KlinePeriod枚举，但使用try-except避免循环导入问题
try:
    from enums.kline_period import KlinePeriod
    HAS_KLINE_PERIOD = True
except ImportError:
    HAS_KLINE_PERIOD = False

# 配置日志
logger = logging.getLogger('clickhouse_db')

# 默认配置
DEFAULT_CONFIG = {
    'host': 'localhost',
    'port': 9000,
    'user': 'default',
    'password': '',
    'database': 'stock'
}

def get_default_config():
    """
    获取默认ClickHouse配置
    """
    return DEFAULT_CONFIG.copy()

class ClickHouseDBManager:
    """
    ClickHouse数据库连接管理器
    使用单例模式确保连接复用
    """
    _instance = None
    _connections: Dict[str, Client] = {}  # 连接池
    _lock = threading.RLock()  # 添加可重入锁，保护连接池
    
    @classmethod
    def get_instance(cls):
        """
        获取单例实例
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def get_new_connection(self, host='localhost', port=9000, user='default', password='', database='default') -> Client:
        """
        获取全新的数据库连接，不从连接池获取
        每次调用都创建新连接，适用于多线程环境
        :return: 新创建的Client对象
        """
        try:
            logger.info(f"创建新连接: {host}:{port}:{database}")
            client = Client(
                host=host,
                port=port,
                user=user,
                password=password,
                database=database,
                settings={
                    'max_execution_time': 60,  # 设置查询超时时间为60秒
                    'connect_timeout': 10,     # 连接超时10秒
                    'receive_timeout': 10,     # 接收超时10秒
                    'send_timeout': 10,        # 发送超时10秒
                }
            )
            return client
        except Exception as e:
            logger.error(f"创建新连接失败: {e}")
            raise
    
    def get_connection(self, host='localhost', port=9000, user='default', password='', database='default') -> Client:
        """
        获取数据库连接
        如果已存在相同配置的连接，则直接返回，否则创建新连接
        注意：此方法在多线程环境下可能不安全，建议使用get_new_connection
        """
        # 生成连接的唯一标识
        conn_key = f"{host}:{port}:{user}:{database}"
        
        with self._lock:
            # 如果连接已存在，则直接返回
            if conn_key in self._connections and self._connections[conn_key] is not None:
                logger.debug(f"复用已有连接: {conn_key}")
                return self._connections[conn_key]
            
            # 创建新连接
            logger.info(f"创建新连接并加入连接池: {conn_key}")
            client = Client(
                host=host,
                port=port,
                user=user,
                password=password,
                database=database,
                settings={
                    'max_execution_time': 60,  # 设置查询超时时间为60秒
                    'connect_timeout': 10,     # 连接超时10秒
                    'receive_timeout': 10,     # 接收超时10秒
                    'send_timeout': 10,        # 发送超时10秒
                }
            )
            
            # 存储连接
            self._connections[conn_key] = client
            return client
    
    def close_connection(self, client=None):
        """
        关闭指定数据库连接
        :param client: Client对象，如果为None则什么都不做
        """
        if client is not None:
            try:
                client.disconnect()
                logger.debug("连接已关闭")
            except Exception as e:
                logger.error(f"关闭连接失败: {e}")
    
    def close_all_connections(self):
        """
        关闭所有数据库连接
        """
        with self._lock:
            logger.info(f"关闭所有连接，共 {len(self._connections)} 个连接")
            for conn_key, client in self._connections.items():
                try:
                    if client is not None:
                        client.disconnect()
                except Exception as e:
                    logger.error(f"关闭连接 {conn_key} 失败: {e}")
            self._connections.clear()

class ClickHouseDB:
    def __init__(self, host='localhost', port=9000, user='default', password='', database='default'):
        """
        初始化 ClickHouse 数据库连接
        创建新的连接，避免多线程环境下的连接复用问题
        """
        self.config = {
            'host': host,
            'port': port,
            'user': user,
            'password': password,
            'database': database
        }
        # 使用连接管理器创建新连接，不复用已有连接
        self.db_manager = ClickHouseDBManager.get_instance()
        self.client = self.db_manager.get_new_connection(**self.config)
        
    def reconnect(self, database=None):
        """
        重新连接数据库
        :param database: 可选，指定新的数据库名
        """
        if database:
            self.config['database'] = database
        
        # 关闭旧连接
        if hasattr(self, 'client') and self.client:
            try:
                self.client.disconnect()
            except Exception as e:
                logger.error(f"关闭旧连接失败: {e}")
        
        # 创建新连接
        self.client = self.db_manager.get_new_connection(**self.config)
        
    def create_database(self, database_name):
        """
        创建数据库
        """
        try:
            logger.info(f"创建数据库: {database_name}")
            self.client.execute(f"CREATE DATABASE IF NOT EXISTS {database_name}")
            logger.info(f"数据库创建成功: {database_name}")
        except Exception as e:
            logger.error(f"创建数据库失败: {e}")
            raise
        
    def init_database(self, database_name):
        """
        初始化数据库，创建数据库并切换连接到新数据库
        """
        try:
            # 创建数据库
            self.create_database(database_name)
            
            # 重新连接到新创建的数据库
            self.reconnect(database=database_name)
            
            # 创建必要的表
            self.create_stock_info_table()
            self.create_industry_info_table()
            
            logger.info(f"数据库初始化完成: {database_name}")
            return True
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            return False
        
    def create_stock_info_table(self):
        """
        创建股票信息表
        """
        try:
            logger.info("创建股票信息表")
            self.client.execute('''
            CREATE TABLE IF NOT EXISTS stock_info (
                code String,
                name String,
                date Date,
                level String,
                open Float64,
                close Float64,
                high Float64,
                low Float64,
                volume Float64,
                turnover_rate Float64,
                price_change Float64,
                price_range Float64,
                industry String DEFAULT '',
                datetime DateTime DEFAULT now(),
                seq UInt32 DEFAULT 0
            ) ENGINE = ReplacingMergeTree()
            PRIMARY KEY (code, level, date, datetime, seq)
            ORDER BY (code, level, date, datetime, seq)
            ''')
            logger.info("股票信息表创建成功")
        except Exception as e:
            logger.error(f"创建股票信息表失败: {e}")
            raise
        
    def create_industry_info_table(self):
        """
        创建行业信息表
        """
        try:
            logger.info("创建行业信息表")
            self.client.execute('''
            CREATE TABLE IF NOT EXISTS industry_info (
                symbol String,
                date Date,
                open Float64,
                close Float64,
                high Float64,
                low Float64,
                volume Float64
            ) ENGINE = MergeTree()
            ORDER BY (symbol, date)
            ''')
            logger.info("行业信息表创建成功")
        except Exception as e:
            logger.error(f"创建行业信息表失败: {e}")
            raise
        
    def save_stock_info(self, stock_info: pd.DataFrame, level):
        """
        保存股票信息
        :param stock_info: 需要保存的股票数据DataFrame
        :param level: K线级别，可以是字符串或KlinePeriod枚举
        """
        # 如果level是KlinePeriod枚举实例，转换为字符串
        if HAS_KLINE_PERIOD and hasattr(level, 'value'):
            level = level.value
        
        # 检查是否包含必要的列
        required_columns = ['收盘', '开盘', '最高', '最低', '成交量']
        for col in required_columns:
            if col not in stock_info.columns:
                logger.error(f"缺少必要的列: {col}")
                raise ValueError(f"DataFrame缺少必要的列: {col}")
        
        try:
            # 获取股票代码和名称
            code = None
            # 尝试多种可能的列名获取股票代码
            if stock_info.index.name and str(stock_info.index.name).strip():
                code = stock_info.index.name  # 使用index name作为股票代码
            if code is None and 'code' in stock_info.columns and not pd.isna(stock_info['code'].iloc[0]):
                code = stock_info['code'].iloc[0]
            if code is None and '代码' in stock_info.columns and not pd.isna(stock_info['代码'].iloc[0]):
                code = stock_info['代码'].iloc[0]
            if code is None and 'symbol' in stock_info.columns and not pd.isna(stock_info['symbol'].iloc[0]):
                code = stock_info['symbol'].iloc[0]
            
            # 确保code不为None且为字符串类型
            if code is None:
                raise ValueError("无法获取股票代码，请检查数据格式")
            code = str(code).strip()
            if not code:
                raise ValueError("股票代码不能为空")
            
            name = ''
            if 'name' in stock_info.columns and not pd.isna(stock_info['name'].iloc[0]):
                name = stock_info['name'].iloc[0]
            if not name and '名称' in stock_info.columns and not pd.isna(stock_info['名称'].iloc[0]):
                name = stock_info['名称'].iloc[0]
            
            # 准备批量插入的数据
            data = []
            for _, row in stock_info.iterrows():
                try:
                    # 获取日期，并确保是datetime.date类型
                    date = row['日期']
                    if isinstance(date, str):
                        try:
                            date = datetime.datetime.strptime(date, '%Y-%m-%d').date()
                        except ValueError:
                            try:
                                date = datetime.datetime.strptime(date, '%Y/%m/%d').date()
                            except ValueError:
                                logger.warning(f"无法解析日期字符串: {date}，跳过此记录")
                                continue
                    
                    # 获取时间戳，对于15/30/60分钟级别的数据，使用datetime列
                    datetime_val = None
                    if 'datetime' in row and not pd.isna(row['datetime']):
                        datetime_val = row['datetime']
                    else:
                        # 如果没有datetime列，则使用日期+00:00:00作为时间戳
                        datetime_val = datetime.datetime.combine(date, datetime.time())
                    
                    # 获取序号，用于同一天内数据的排序
                    seq = 0
                    if 'seq' in row and not pd.isna(row['seq']):
                        seq = int(row['seq'])
                    
                    # 获取行业信息
                    industry = ''
                    if 'industry' in row and not pd.isna(row['industry']):
                        industry = row['industry']
                    
                    # 准备数据行
                    data_row = (
                        code,                   # 代码
                        name,                   # 名称
                        date,                   # 日期
                        level,                  # 级别
                        float(row['开盘']),     # 开盘价
                        float(row['收盘']),     # 收盘价
                        float(row['最高']),     # 最高价
                        float(row['最低']),     # 最低价
                        float(row['成交量']),   # 成交量
                        float(row.get('换手率', 0.0)),  # 换手率，如果不存在则为0
                        float(row.get('涨跌幅', 0.0)),  # 涨跌幅，如果不存在则为0
                        float(row.get('振幅', 0.0)),    # 振幅，如果不存在则为0
                        industry,               # 行业
                        datetime_val,           # 时间戳
                        seq                     # 序号
                    )
                    data.append(data_row)
                except Exception as e:
                    logger.error(f"处理数据行失败: {e}，跳过此行")
                    continue
            
            # 如果没有有效数据则直接返回
            if not data:
                logger.warning("没有有效数据可保存")
                return
            
            # 批量插入数据
            logger.info(f"开始批量插入 {len(data)} 条 {level} 级别数据到ClickHouse")
            
            # 创建事务并执行插入
            self.client.execute(
                '''
                INSERT INTO stock_info (
                    code, name, date, level, open, close, high, low, volume, 
                    turnover_rate, price_change, price_range, industry, datetime, seq
                ) VALUES
                ''',
                data
            )
            
            logger.info(f"成功保存 {len(data)} 条 {level} 级别数据到ClickHouse")
        except Exception as e:
            logger.error(f"保存股票数据失败: {e}")
            raise
    
    def save_industry_info(self, industry_info: pd.DataFrame):
        """
        保存行业信息到 ClickHouse
        """
        if industry_info.empty:
            logger.warning("要保存的行业数据为空，跳过保存")
            return
            
        try:
            logger.info(f"开始保存行业数据，共 {len(industry_info)} 条记录")
            data = []
            for index, row in industry_info.iterrows():
                # 确保日期是正确的格式
                date_str = row["日期"]
                if isinstance(date_str, str):
                    # 尝试不同的日期格式
                    try:
                        parsed_date = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
                    except ValueError:
                        try:
                            parsed_date = datetime.datetime.strptime(date_str, '%Y%m%d').date()
                        except ValueError:
                            raise ValueError(f"无法解析日期格式: {date_str}")
                elif isinstance(date_str, (datetime.date, datetime.datetime)):
                    parsed_date = date_str if isinstance(date_str, datetime.date) else date_str.date()
                else:
                    raise ValueError(f"不支持的日期类型: {type(date_str)}")
                    
                data.append((
                    row["板块"],
                    parsed_date,
                    float(row["开盘价"]),
                    float(row["收盘价"]),
                    float(row["最高价"]),
                    float(row["最低价"]),
                    float(row["成交量"])
                ))
            
            self.client.execute(
                '''
                INSERT INTO industry_info 
                (symbol, date, open, close, high, low, volume)
                VALUES
                ''',
                data
            )
            logger.info(f"成功保存行业数据，共 {len(data)} 条记录")
        except Exception as e:
            logger.error(f"保存行业数据失败: {e}")
            raise
        
    def get_stock_info(self, code, level, start="20170101", end="20240101"):
        """
        获取股票K线数据
        :param code: 股票代码
        :param level: K线级别，可以是字符串或KlinePeriod枚举
        :param start: 开始日期
        :param end: 结束日期
        :return: 股票K线数据
        """
        # 如果level是KlinePeriod枚举实例，转换为字符串
        if HAS_KLINE_PERIOD and hasattr(level, 'value'):
            level = level.value
            
        try:
            # 转换日期格式
            if isinstance(start, str):
                if '-' in start:
                    start_date = datetime.datetime.strptime(start, '%Y-%m-%d').date()
                else:
                    start_date = datetime.datetime.strptime(start, '%Y%m%d').date()
            else:
                start_date = start
                
            if isinstance(end, str):
                if '-' in end:
                    end_date = datetime.datetime.strptime(end, '%Y-%m-%d').date()
                else:
                    end_date = datetime.datetime.strptime(end, '%Y%m%d').date()
            else:
                end_date = end
            
            # 检查是否需要计算30分钟或60分钟数据
            if level in ['30分钟', '60分钟']:
                # 从15分钟数据计算
                logger.info(f"从15分钟数据计算{level}K线: {code}, 时间范围: {start_date} 至 {end_date}")
                
                # 查询15分钟数据，不使用toHour和toMinute函数
                data_15min = self.client.execute(f"""
                SELECT
                    code,
                    name,
                    date,
                    open,
                    close,
                    high,
                    low,
                    volume,
                    turnover_rate,
                    price_change,
                    price_range,
                    industry
                FROM stock_info
                WHERE code = '{code}' AND level = '15分钟' AND date BETWEEN '{start_date}' AND '{end_date}'
                ORDER BY date, datetime, seq
                """)
                
                # 如果没有15分钟数据，返回空结果
                if not data_15min:
                    logger.warning(f"未找到15分钟数据，无法计算{level}K线: {code}")
                    return []
                
                # 根据要计算的周期确定合并因子
                merge_factor = 2 if level == '30分钟' else 4  # 30分钟合并2个15分钟，60分钟合并4个15分钟
                
                # 将数据按日期分组，每merge_factor个15分钟数据合并为一个新的K线
                grouped_data = []
                current_group = []
                current_date = None
                group_count = 0
                
                for i, row in enumerate(data_15min):
                    date = row[2]  # 日期
                    
                    # 如果是新的一天或者第一条记录
                    if current_date is None or current_date != date:
                        # 如果当前组有数据，先处理完当前组
                        if current_group:
                            # 合并K线数据
                            first = current_group[0]
                            last = current_group[-1]
                            
                            merged_row = (
                                first[0],  # code
                                first[1],  # name
                                first[2],  # date (使用组内第一个数据的日期)
                                level,     # level
                                first[3],  # open (使用组内第一个数据的开盘价)
                                last[4],   # close (使用组内最后一个数据的收盘价)
                                max(r[5] for r in current_group),  # high (使用组内最高价的最大值)
                                min(r[6] for r in current_group),  # low (使用组内最低价的最小值)
                                sum(r[7] for r in current_group),  # volume (累加组内成交量)
                                sum(r[8] for r in current_group) / len(current_group),  # turnover_rate (平均换手率)
                                sum(r[9] for r in current_group),  # price_change (累加价格变化)
                                sum(r[10] for r in current_group) / len(current_group),  # price_range (平均价格范围)
                                first[11]  # industry
                            )
                            
                            grouped_data.append(merged_row)
                        
                        # 重置当前组和计数器
                        current_group = []
                        current_date = date
                        group_count = 0
                    
                    # 添加当前15分钟数据到组
                    current_group.append(row)
                    group_count += 1
                    
                    # 判断是否需要合并当前组
                    # 达到了合并因子的数量
                    if group_count == merge_factor:
                        # 合并K线数据
                        first = current_group[0]
                        last = current_group[-1]
                        
                        merged_row = (
                            first[0],  # code
                            first[1],  # name
                            first[2],  # date (使用组内第一个数据的日期)
                            level,     # level
                            first[3],  # open (使用组内第一个数据的开盘价)
                            last[4],   # close (使用组内最后一个数据的收盘价)
                            max(r[5] for r in current_group),  # high (使用组内最高价的最大值)
                            min(r[6] for r in current_group),  # low (使用组内最低价的最小值)
                            sum(r[7] for r in current_group),  # volume (累加组内成交量)
                            sum(r[8] for r in current_group) / len(current_group),  # turnover_rate (平均换手率)
                            sum(r[9] for r in current_group),  # price_change (累加价格变化)
                            sum(r[10] for r in current_group) / len(current_group),  # price_range (平均价格范围)
                            first[11]  # industry
                        )
                        
                        grouped_data.append(merged_row)
                        
                        # 重置当前组和计数器
                        current_group = []
                        group_count = 0
                
                # 处理最后一组数据（如果有）
                if current_group:
                    # 合并K线数据
                    first = current_group[0]
                    last = current_group[-1]
                    
                    merged_row = (
                        first[0],  # code
                        first[1],  # name
                        first[2],  # date (使用组内第一个数据的日期)
                        level,     # level
                        first[3],  # open (使用组内第一个数据的开盘价)
                        last[4],   # close (使用组内最后一个数据的收盘价)
                        max(r[5] for r in current_group),  # high (使用组内最高价的最大值)
                        min(r[6] for r in current_group),  # low (使用组内最低价的最小值)
                        sum(r[7] for r in current_group),  # volume (累加组内成交量)
                        sum(r[8] for r in current_group) / len(current_group),  # turnover_rate (平均换手率)
                        sum(r[9] for r in current_group),  # price_change (累加价格变化)
                        sum(r[10] for r in current_group) / len(current_group),  # price_range (平均价格范围)
                        first[11]  # industry
                    )
                    
                    grouped_data.append(merged_row)
                
                logger.info(f"计算{level}K线成功: {code}, 获取到 {len(grouped_data)} 条记录")
                return grouped_data
            else:
                # 查询数据库中存储的K线数据
                logger.info(f"查询股票信息: {code}, 级别: {level}, 时间范围: {start_date} 至 {end_date}")
                
                # 选择查询标准列，确保返回的列数一致
                base_query = f"""
                SELECT
                    code,
                    name,
                    date,
                    level,
                    open,
                    close,
                    high,
                    low,
                    volume,
                    turnover_rate,
                    price_change,
                    price_range,
                    industry
                FROM stock_info
                WHERE code = '{code}' AND level = '{level}' AND date BETWEEN '{start_date}' AND '{end_date}'
                """
                
                # 根据级别选择排序方式
                if level in ['15分钟', '30分钟', '60分钟']:
                    query = base_query + " ORDER BY date, datetime, seq"
                else:
                    query = base_query + " ORDER BY date"
                
                # 执行查询
                data = self.client.execute(query)
                
                # 如果是分钟级别数据，检查是否有足够的每日记录
                if level in ['15分钟', '30分钟', '60分钟']:
                    date_counts = {}
                    for row in data:
                        date_str = str(row[2])
                        date_counts[date_str] = date_counts.get(date_str, 0) + 1
                    
                    # 记录每天的记录数，用于调试
                    for date_str, count in date_counts.items():
                        if count == 1:
                            logger.warning(f"日期 {date_str} 只有1条 {level} 记录")
                        else:
                            logger.info(f"日期 {date_str} 有 {count} 条 {level} 记录")
                
                logger.info(f"查询成功: {code}, 获取到 {len(data)} 条记录")
                return data
        except Exception as e:
            logger.error(f"查询股票信息失败: {e}")
            return []
    
    def get_industry_info(self, symbol, start, end):
        """
        获取行业信息
        :param symbol: 行业代码
        :param start: 开始日期，格式：YYYYMMDD 或 YYYY-MM-DD
        :param end: 结束日期，格式：YYYYMMDD 或 YYYY-MM-DD
        """
        try:
            # 格式化日期
            try:
                # 尝试 YYYYMMDD 格式
                start_date = datetime.datetime.strptime(start, '%Y%m%d').date()
            except ValueError:
                # 尝试 YYYY-MM-DD 格式
                start_date = datetime.datetime.strptime(start, '%Y-%m-%d').date()
                
            try:
                # 尝试 YYYYMMDD 格式
                end_date = datetime.datetime.strptime(end, '%Y%m%d').date()
            except ValueError:
                # 尝试 YYYY-MM-DD 格式
                end_date = datetime.datetime.strptime(end, '%Y-%m-%d').date()
            
            logger.info(f"获取行业数据: {symbol}, {start_date} - {end_date}")
            result = self.client.execute(
                f'''
                SELECT * FROM industry_info 
                WHERE symbol = '{symbol}' AND date BETWEEN '{start_date}' AND '{end_date}'
                ORDER BY date
                '''
            )
            
            # 将结果转换为 DataFrame
            columns = ['symbol', 'date', 'open', 'close', 'high', 'low', 'volume']
            df = pd.DataFrame(result, columns=columns)
            logger.info(f"获取到行业数据: {len(df)}条记录")
            return df
        except Exception as e:
            logger.error(f"获取行业数据失败: {e}")
            return pd.DataFrame()
    
    def update_industry(self, codes, industry):
        """
        更新行业信息
        """
        try:
            if not codes:
                logger.warning("没有股票代码需要更新行业信息")
                return
                
            codes_list = ",".join([f"'{code}'" for code in codes if code != "暂无成份股数据"])
            if not codes_list:
                logger.warning("过滤后没有有效的股票代码需要更新行业信息")
                return
                
            logger.info(f"更新行业信息: {industry}, 共 {len(codes)} 只股票")
            self.client.execute(
                f'''
                ALTER TABLE stock_info
                UPDATE industry = '{industry}'
                WHERE code IN ({codes_list})
                '''
            )
            logger.info(f"行业信息更新成功: {industry}")
        except Exception as e:
            logger.error(f"更新行业信息失败: {e}")
            raise
    
    def get_industry_stock(self, industry):
        """
        获取行业股票
        """
        try:
            logger.info(f"获取行业股票: {industry}")
            result = self.client.execute(
                f'''
                SELECT DISTINCT code FROM stock_info
                WHERE industry = '{industry}'
                '''
            )
            logger.info(f"获取到行业股票: {len(result)}只")
            return result
        except Exception as e:
            logger.error(f"获取行业股票失败: {e}")
            return []
    
    def get_stock_max_date(self, code):
        """
        获取股票数据最大日期
        """
        try:
            logger.info("获取股票数据最大日期")
            result = self.client.execute(
                f'''
                SELECT max(date) FROM stock_info WHERE code = '{code}'
                '''
            )
            max_date = result[0][0] if result and result[0][0] else datetime.date(1970, 1, 1)
            logger.info(f"股票数据最大日期: {max_date}")
            return max_date
        except Exception as e:
            logger.error(f"获取股票数据最大日期失败: {e}")
            return datetime.date(1970, 1, 1)
    
    def get_industry_max_date(self):
        """
        获取行业数据最大日期
        """
        try:
            logger.info("获取行业数据最大日期")
            result = self.client.execute(
                '''
                SELECT max(date) FROM industry_info
                '''
            )
            max_date = result[0][0] if result and result[0][0] else datetime.date(1970, 1, 1)
            logger.info(f"行业数据最大日期: {max_date}")
            return max_date
        except Exception as e:
            logger.error(f"获取行业数据最大日期失败: {e}")
            return datetime.date(1970, 1, 1)
        
    def get_avg_price(self, code, start):
        """
        获取平均价格
        """
        try:
            query = f"""
            SELECT 
                avg(close) as avg_price 
            FROM stock_daily_prices 
            WHERE stock_code = '{code}' 
                AND trade_date >= '{start}'
            """
            result = self.client.execute(query)
            
            if result and result[0][0]:
                return float(result[0][0])
            else:
                return 0
        except Exception as e:
            logger.error(f"获取平均价格失败: {e}")
            return 0
            
    def query_dataframe(self, query):
        """
        执行SQL查询并返回pandas DataFrame格式的结果
        
        Args:
            query: SQL查询字符串
            
        Returns:
            pd.DataFrame: 查询结果的DataFrame
        """
        try:
            # 执行查询
            result = self.client.execute(query, with_column_types=True)
            
            # 解析结果
            if result:
                data, columns = result
                column_names = [col[0] for col in columns]
                
                # 创建DataFrame
                df = pd.DataFrame(data, columns=column_names)
                return df
            else:
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"执行查询失败: {e}")
            return pd.DataFrame()
    
    def __del__(self):
        """
        析构函数，确保资源正确释放
        """
        try:
            if hasattr(self, 'client') and self.client:
                self.client.disconnect()
                logger.debug("连接已在析构函数中关闭")
        except Exception as e:
            logger.error(f"析构函数中关闭连接失败: {e}")

# 创建获取数据库连接的全局便捷函数
def get_clickhouse_db(host=None, port=None, user=None, password=None, database=None, config=None) -> ClickHouseDB:
    """
    获取ClickHouseDB实例的工厂函数
    :param host: 主机地址
    :param port: 端口
    :param user: 用户名
    :param password: 密码
    :param database: 数据库名
    :param config: 完整配置字典，如果提供则优先使用
    :return: ClickHouseDB实例
    """
    # 使用默认配置作为基础
    final_config = DEFAULT_CONFIG.copy()
    
    # 如果提供了完整配置，则使用完整配置
    if config:
        final_config.update(config)
    else:
        # 否则使用单独提供的参数更新配置
        if host:
            final_config['host'] = host
        if port:
            final_config['port'] = port
        if user:
            final_config['user'] = user
        if password:
            final_config['password'] = password
        if database:
            final_config['database'] = database
    
    return ClickHouseDB(**final_config) 