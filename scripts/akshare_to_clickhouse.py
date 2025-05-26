#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
股票数据同步工具
---------------
本模块用于从多个数据源获取股票数据并同步到ClickHouse数据库。
支持多数据源、自动容错和故障转移、增量同步等功能。

主要特点:
1. 多数据源支持: efinance, akshare, baostock
2. 自动数据源切换: 当一个数据源失败时自动切换到下一个
3. 多线程并发: 支持多线程并发同步多只股票
4. 增量同步: 只同步尚未同步到最新日期的数据
5. 多周期支持: 支持15分钟、日线、周线、月线数据

使用方法:
python bin/test_multi_sync.py [--csv 股票代码文件] [--threads 线程数] 
                             [--batch 批次大小] [--force] [--code 股票代码] 
                             [--source 数据源]

数据源优先级: efinance > akshare > baostock
"""

import sys
import os

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

import pandas as pd
import efinance as ef
import akshare as ak  # 引入akshare
import time
import logging
import datetime
import concurrent.futures
import threading
import queue
from logging.handlers import RotatingFileHandler
from db.clickhouse_db import get_clickhouse_db, get_default_config
import random

from enums.kline_period import KlinePeriod

# 确保日志目录存在
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 创建日志文件处理器，使用RotatingFileHandler支持日志轮转
log_file = os.path.join(log_dir, 'akshare_sync.log')
file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# 配置日志记录器
logger = logging.getLogger('akshare_sync')
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# 创建一个线程锁，用于保护日志输出
log_lock = threading.Lock()

# 创建一个线程安全的日志函数
def thread_safe_log(level, message):
    with log_lock:
        if level == 'info':
            logger.info(message)
        elif level == 'error':
            logger.error(message)
        elif level == 'warning':
            logger.warning(message)
        elif level == 'debug':
            logger.debug(message)

# 添加数据源适配器类
class DataSourceAdapter:
    """
    数据源适配器，支持多种股票数据API
    """
    
    # 数据源类型枚举
    SOURCE_EFINANCE = 'efinance'
    SOURCE_AKSHARE = 'akshare'
    SOURCE_BAOSTOCK = 'baostock'
    
    def __init__(self, data_source=SOURCE_EFINANCE):
        """
        初始化数据源适配器
        :param data_source: 数据源类型，默认为efinance
        """
        self.data_source = data_source
        logger.info(f"使用数据源: {data_source}")
        
        # 初始化baostock（如果使用）
        if data_source == self.SOURCE_BAOSTOCK:
            try:
                import baostock as bs
                self.bs = bs
                self.bs.login()
                logger.info("Baostock登录成功")
            except Exception as e:
                logger.error(f"Baostock初始化失败: {e}")
    
    def __del__(self):
        """
        析构函数，确保资源释放
        """
        if hasattr(self, 'bs') and self.data_source == self.SOURCE_BAOSTOCK:
            self.bs.logout()
    
    def get_stock_daily_data(self, stock_code, start_date=None):
        """
        获取股票日线数据
        :param stock_code: 股票代码
        :param start_date: 开始日期 (YYYYMMDD)
        :return: 股票数据DataFrame
        """
        # 确保股票代码是6位，不足前面补0
        stock_code = str(stock_code).zfill(6)
        
        # 根据数据源类型选择不同的实现
        if self.data_source == self.SOURCE_EFINANCE:
            return self._get_daily_efinance(stock_code, start_date)
        elif self.data_source == self.SOURCE_AKSHARE:
            return self._get_daily_akshare(stock_code, start_date)
        elif self.data_source == self.SOURCE_BAOSTOCK:
            return self._get_daily_baostock(stock_code, start_date)
        else:
            logger.error(f"不支持的数据源类型: {self.data_source}")
            return pd.DataFrame()
    
    def _get_daily_efinance(self, stock_code, start_date):
        """使用efinance获取日线数据"""
        try:
            logger.info(f"正在使用efinance获取股票 {stock_code} 的日线数据，开始日期: {start_date}")
            
            # 模拟故障：对特定股票总是返回错误
            if stock_code == '002578':
                logger.error(f"模拟efinance故障：无法获取股票 {stock_code} 的日线数据")
                raise Exception("模拟网络错误：连接东方财富服务器失败")
            
            # klt=101表示日线级别数据
            stock_data = ef.stock.get_quote_history(stock_code, beg=start_date, klt=101)
            
            if stock_data is not None and not stock_data.empty:
                logger.info(f"成功获取股票 {stock_code} 的日线数据，共 {len(stock_data)} 条记录")
                return stock_data
            else:
                logger.warning(f"获取股票 {stock_code} 的日线数据为空")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"efinance获取股票 {stock_code} 的日线数据失败: {e}")
            return pd.DataFrame()
    
    def _get_daily_akshare(self, stock_code, start_date):
        """使用akshare获取日线数据"""
        try:
            logger.info(f"正在使用akshare获取股票 {stock_code} 的日线数据，开始日期: {start_date}")
            
            # 转换日期格式为YYYY-MM-DD
            if start_date:
                start_date = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
            else:
                start_date = "2000-01-01"  # akshare默认起始日期
            
            end_date = datetime.datetime.now().strftime("%Y-%m-%d")
            
            # 判断股票代码所属市场
            if stock_code.startswith(('0', '3')):
                market = 'sz'
            else:
                market = 'sh'
                
            # 使用akshare获取A股历史数据
            stock_zh_a_daily_df = ak.stock_zh_a_hist(symbol=f"{market}{stock_code}", 
                                                     start_date=start_date,
                                                     end_date=end_date,
                                                     adjust="qfq")  # 前复权
            
            if stock_zh_a_daily_df is not None and not stock_zh_a_daily_df.empty:
                # 转换列名以匹配efinance的格式
                stock_zh_a_daily_df.rename(columns={
                    "日期": "日期",
                    "开盘": "开盘",
                    "收盘": "收盘",
                    "最高": "最高",
                    "最低": "最低",
                    "成交量": "成交量",
                    "成交额": "成交额",
                    "振幅": "振幅",
                    "涨跌幅": "涨跌幅",
                    "涨跌额": "涨跌额",
                    "换手率": "换手率"
                }, inplace=True)
                
                logger.info(f"成功获取股票 {stock_code} 的日线数据，共 {len(stock_zh_a_daily_df)} 条记录")
                return stock_zh_a_daily_df
            else:
                logger.warning(f"获取股票 {stock_code} 的日线数据为空")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"akshare获取股票 {stock_code} 的日线数据失败: {e}")
            return pd.DataFrame()
    
    def _get_daily_baostock(self, stock_code, start_date):
        """使用baostock获取日线数据"""
        try:
            logger.info(f"正在使用baostock获取股票 {stock_code} 的日线数据，开始日期: {start_date}")
            
            # 转换日期格式为YYYY-MM-DD
            if start_date:
                start_date = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
            else:
                start_date = "2000-01-01"  # baostock默认起始日期
            
            end_date = datetime.datetime.now().strftime("%Y-%m-%d")
            
            # 判断股票代码所属市场
            if stock_code.startswith(('0', '3')):
                full_code = f"sz.{stock_code}"
            else:
                full_code = f"sh.{stock_code}"
            
            # 使用baostock获取A股历史数据
            rs = self.bs.query_history_k_data_plus(
                full_code,
                "date,open,high,low,close,volume,amount,turn",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="2"  # 前复权
            )
            
            # 获取结果并转换为DataFrame
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            
            if data_list:
                result = pd.DataFrame(data_list, columns=rs.fields)
                # 转换列名以匹配efinance的格式
                result.rename(columns={
                    "date": "日期",
                    "open": "开盘",
                    "high": "最高",
                    "low": "最低",
                    "close": "收盘",
                    "volume": "成交量",
                    "amount": "成交额",
                    "turn": "换手率"
                }, inplace=True)
                
                # 转换数据类型
                for col in ["开盘", "最高", "最低", "收盘", "成交量", "成交额", "换手率"]:
                    if col in result.columns:
                        result[col] = pd.to_numeric(result[col], errors='coerce')
                
                logger.info(f"成功获取股票 {stock_code} 的日线数据，共 {len(result)} 条记录")
                return result
            else:
                logger.warning(f"获取股票 {stock_code} 的日线数据为空")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"baostock获取股票 {stock_code} 的日线数据失败: {e}")
            return pd.DataFrame()
    
    def get_stock_15min_data(self, stock_code, start_date=None):
        """
        获取15分钟K线数据
        :param stock_code: 股票代码
        :param start_date: 开始日期 (YYYYMMDD)
        :return: 股票数据DataFrame
        """
        # 确保股票代码是6位，不足前面补0
        stock_code = str(stock_code).zfill(6)
        
        # 根据数据源类型选择不同的实现
        if self.data_source == self.SOURCE_EFINANCE:
            try:
                logger.info(f"正在使用efinance获取股票 {stock_code} 的15分钟数据，开始日期: {start_date}")
                stock_data = ef.stock.get_quote_history(stock_code, beg=start_date, klt=15)
                
                if stock_data is not None and not stock_data.empty:
                    logger.info(f"成功获取股票 {stock_code} 的15分钟数据，共 {len(stock_data)} 条记录")
                    return stock_data
                else:
                    logger.warning(f"获取股票 {stock_code} 的15分钟数据为空")
                    return pd.DataFrame()
            except Exception as e:
                logger.error(f"efinance获取股票 {stock_code} 的15分钟数据失败: {e}")
                return pd.DataFrame()
        elif self.data_source == self.SOURCE_AKSHARE:
            # 目前akshare不支持15分钟数据
            logger.warning(f"数据源 {self.data_source} 暂不支持15分钟数据获取")
            return pd.DataFrame()
        elif self.data_source == self.SOURCE_BAOSTOCK:
            # 目前baostock不支持15分钟数据
            logger.warning(f"数据源 {self.data_source} 暂不支持15分钟数据获取")
            return pd.DataFrame()
        else:
            logger.warning(f"数据源 {self.data_source} 暂不支持15分钟数据获取")
            return pd.DataFrame()
    
    def get_stock_weekly_data(self, stock_code, start_date=None):
        """获取周线数据"""
        # 根据数据源类型选择不同的实现
        if self.data_source == self.SOURCE_EFINANCE:
            try:
                logger.info(f"正在使用efinance获取股票 {stock_code} 的周线数据，开始日期: {start_date}")
                stock_data = ef.stock.get_quote_history(stock_code, beg=start_date, klt=102)
                
                if stock_data is not None and not stock_data.empty:
                    logger.info(f"成功获取股票 {stock_code} 的周线数据，共 {len(stock_data)} 条记录")
                    return stock_data
                else:
                    logger.warning(f"获取股票 {stock_code} 的周线数据为空")
                    return pd.DataFrame()
            except Exception as e:
                logger.error(f"获取股票 {stock_code} 的周线数据失败: {e}")
                return pd.DataFrame()
        else:
            # 其他数据源的实现可以按需添加
            logger.warning(f"数据源 {self.data_source} 暂不支持周线数据获取")
            return pd.DataFrame()
    
    def get_stock_monthly_data(self, stock_code, start_date=None):
        """获取月线数据"""
        # 根据数据源类型选择不同的实现
        if self.data_source == self.SOURCE_EFINANCE:
            try:
                logger.info(f"正在使用efinance获取股票 {stock_code} 的月线数据，开始日期: {start_date}")
                stock_data = ef.stock.get_quote_history(stock_code, beg=start_date, klt=103)
                
                if stock_data is not None and not stock_data.empty:
                    logger.info(f"成功获取股票 {stock_code} 的月线数据，共 {len(stock_data)} 条记录")
                    return stock_data
                else:
                    logger.warning(f"获取股票 {stock_code} 的月线数据为空")
                    return pd.DataFrame()
            except Exception as e:
                logger.error(f"获取股票 {stock_code} 的月线数据失败: {e}")
                return pd.DataFrame()
        else:
            # 其他数据源的实现可以按需添加
            logger.warning(f"数据源 {self.data_source} 暂不支持月线数据获取")
            return pd.DataFrame()

# 数据源自动切换类
class DataSourceManager:
    """
    数据源管理器，支持自动切换数据源
    """
    
    # 数据源列表，按优先级排序
    SOURCES = ['efinance', 'akshare', 'baostock']
    
    def __init__(self, initial_source='efinance'):
        """
        初始化数据源管理器
        :param initial_source: 初始数据源
        """
        if initial_source in self.SOURCES:
            self.current_source = initial_source
        else:
            logger.warning(f"未知数据源: {initial_source}，使用默认数据源: efinance")
            self.current_source = 'efinance'
        
        # 初始化数据源适配器
        self.adapter = DataSourceAdapter(self.current_source)
        logger.info(f"初始数据源: {self.current_source}")
        
        # 记录数据源失败次数
        self.failure_counts = {source: 0 for source in self.SOURCES}
        
        # 最大失败次数，超过这个次数将尝试切换数据源
        self.max_failures = 1  # 改为1次失败就切换
        
        # 数据源冷却时间（秒），切换后多久可以再次尝试使用
        self.cooldown_time = 300  # 5分钟
        
        # 记录每个数据源上次失败的时间
        self.last_failure_time = {source: 0 for source in self.SOURCES}
        
        # 记录数据源最新数据状态
        self.is_up_to_date = {source: False for source in self.SOURCES}
    
    def get_adapter(self):
        """
        获取当前数据源适配器
        :return: 数据源适配器
        """
        return self.adapter
    
    def record_up_to_date(self, source=None):
        """
        记录数据源已经是最新的
        :param source: 数据源名称，默认为当前数据源
        """
        if source is None:
            source = self.current_source
        
        if source in self.SOURCES:
            self.is_up_to_date[source] = True
            logger.info(f"数据源 {source} 已标记为数据最新")
    
    def record_failure(self, source=None, is_empty_result=False):
        """
        记录数据源失败
        :param source: 数据源名称，默认为当前数据源
        :param is_empty_result: 是否为空结果（而非错误）
        :return: 是否切换了数据源
        """
        if source is None:
            source = self.current_source
        
        if source not in self.SOURCES:
            return False
        
        # 如果是空结果且已经标记为最新，不计算为失败
        if is_empty_result and self.is_up_to_date[source]:
            logger.info(f"数据源 {source} 返回空结果，但已标记为数据最新，不计为失败")
            return False
        
        # 记录失败次数和时间
        self.failure_counts[source] += 1
        self.last_failure_time[source] = time.time()
        
        logger.warning(f"数据源 {source} 失败次数: {self.failure_counts[source]}/{self.max_failures}")
        
        # 检查是否需要切换数据源
        if self.failure_counts[source] >= self.max_failures:
            logger.warning(f"数据源 {source} 失败次数达到阈值，尝试切换数据源")
            return self.switch_source()
        
        return False
    
    def switch_source(self):
        """
        切换到下一个可用的数据源
        :return: 是否成功切换
        """
        current_index = self.SOURCES.index(self.current_source)
        
        # 尝试所有其他数据源
        for i in range(1, len(self.SOURCES)):
            next_index = (current_index + i) % len(self.SOURCES)
            next_source = self.SOURCES[next_index]
            
            # 检查下一个数据源是否在冷却期
            if time.time() - self.last_failure_time[next_source] < self.cooldown_time:
                logger.warning(f"数据源 {next_source} 在冷却期内，跳过")
                continue
            
            # 切换数据源
            logger.warning(f"切换数据源: {self.current_source} -> {next_source}")
            self.current_source = next_source
            
            # 创建新的适配器
            self.adapter = DataSourceAdapter(self.current_source)
            
            # 重置当前数据源的失败计数
            self.failure_counts[self.current_source] = 0
            
            return True
        
        logger.error("所有数据源都已失败或在冷却期，无法切换")
        return False
    
    def reset_failure(self, source=None):
        """
        重置数据源失败计数
        :param source: 数据源名称，默认为当前数据源
        """
        if source is None:
            source = self.current_source
        
        if source in self.SOURCES:
            self.failure_counts[source] = 0
            logger.info(f"重置数据源 {source} 的失败计数")

class AKShareToClickHouse:
    """
    使用AKShare API将股票数据同步到ClickHouse
    """
    
    def __init__(self, clickhouse_config=None, max_workers=10, batch_size=20, force_sync=False, data_source='efinance'):
        """
        初始化同步器
        :param clickhouse_config: ClickHouse配置
        :param max_workers: 最大工作线程数
        :param batch_size: 每批处理的股票数量
        :param force_sync: 是否强制同步，忽略最新日期检查
        :param data_source: 数据源类型
        """
        # 使用clickhouse_db模块提供的默认配置
        self.clickhouse_config = clickhouse_config or get_default_config()
        
        # 确保ClickHouse中有必要的数据库和表
        self.init_clickhouse()
        
        # 多线程相关配置
        self.max_workers = max_workers  # 最大工作线程数
        self.batch_size = batch_size    # 每批处理的股票数量
        self.force_sync = force_sync    # 是否强制同步
        
        # 增加最大连接数限制，避免连接过多
        self.connection_semaphore = threading.Semaphore(self.max_workers * 2)
        
        # 初始化数据源管理器
        self.data_source_manager = DataSourceManager(data_source)
        
        # 获取最新交易日期
        self.latest_trade_date = self.get_latest_trade_date()

    def save_stock_data_to_clickhouse(self, stock_data, level='15分钟'):
        """
        保存股票数据到ClickHouse，使用独立连接直接插入
        :param stock_data: 处理后的股票数据
        :param level: 数据级别
        """
        if stock_data.empty:
            return
        
        # 最大重试次数
        max_retries = 3
        retry_count = 0
        
        # 使用信号量限制并发连接数
        with self.connection_semaphore:
            while retry_count < max_retries:
                # 每次尝试使用新的数据库连接
                db = None
                try:
                    # 获取股票代码
                    stock_code = stock_data['代码'].iloc[0]
                    
                    # 直接尝试插入所有数据，依赖数据库主键约束避免重复
                    thread_safe_log('info', f"正在保存 {len(stock_data)} 条 {level} 级别数据到ClickHouse")
                    
                    # 创建新的连接
                    db = get_clickhouse_db(config=self.clickhouse_config)
                    db.save_stock_info(stock_data, level)
                    thread_safe_log('info', f"{level}数据保存成功")
                    return
                except Exception as e:
                    retry_count += 1
                    thread_safe_log('error', f"保存{level}数据到ClickHouse失败 (尝试 {retry_count}/{max_retries}): {e}")
                    if retry_count < max_retries:
                        wait_time = retry_count * 2  # 递增等待时间
                        thread_safe_log('info', f"等待 {wait_time} 秒后重试...")
                        time.sleep(wait_time)
                    else:
                        thread_safe_log('error', f"达到最大重试次数，放弃保存数据")
                        return
                finally:
                    # 确保在finally块中关闭连接
                    if db and hasattr(db, 'client') and db.client:
                        try:
                            # 显式关闭连接
                            db.client.disconnect()
                        except Exception as e:
                            thread_safe_log('error', f"关闭数据库连接失败: {e}")

    def sync_single_stock(self, stock_code, stock_name, idx=0, total=0):
        """
        同步单只股票的多周期数据
        :param stock_code: 股票代码
        :param stock_name: 股票名称
        :param idx: 当前处理的股票索引
        :param total: 总股票数量
        :return: 是否成功
        """
        # 最大重试次数
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # 确保股票代码是6位，不足前面补0
                stock_code = str(stock_code).zfill(6)
                
                # 对于已知的问题股票，增加特殊处理
                if stock_code in ['000006', '000010']:  # 深振业A等问题股票
                    thread_safe_log('warning', f"检测到问题股票: {stock_code} - {stock_name}，使用额外的延时和重试机制")
                    # 增加延时，避免连接冲突
                    time.sleep(2)
                
                thread_safe_log('info', f"线程 {threading.current_thread().name} 开始处理第 {idx + 1}/{total} 只股票: {stock_code} - {stock_name}")
                
                # 首先检查日线数据是否已经是最新的，如果是则跳过该股票
                # 只有在非强制同步模式下才进行检查
                daily_latest_date = None
                if not self.force_sync:
                    daily_latest_date = self.get_stock_latest_date(stock_code, KlinePeriod.DAILY.value)
                    if daily_latest_date and daily_latest_date >= self.latest_trade_date:
                        thread_safe_log('info', f"股票 {stock_code} - {stock_name} 已同步到最新交易日 {self.latest_trade_date}，跳过同步")
                        # 标记当前数据源数据已是最新
                        current_source = self.data_source_manager.current_source
                        self.data_source_manager.record_up_to_date(current_source)
                        return True
                
                # 获取每个周期的最新日期，作为同步的起始日期
                # 日线数据
                if daily_latest_date:
                    # 将日期向后调整一天，作为同步的起始日期
                    daily_start_date = (daily_latest_date + datetime.timedelta(days=1)).strftime('%Y%m%d')
                else:
                    daily_start_date = self.get_adjusted_start_date(70)
                
                # 15分钟数据
                min15_latest_date = self.get_stock_latest_date(stock_code, KlinePeriod.MIN_15.value)
                if min15_latest_date:
                    # 将日期向后调整一天，作为同步的起始日期
                    min15_start_date = (min15_latest_date + datetime.timedelta(days=1)).strftime('%Y%m%d')
                else:
                    min15_start_date = self.get_adjusted_start_date(70)
                
                # 周线数据
                weekly_latest_date = self.get_stock_latest_date(stock_code, KlinePeriod.WEEKLY.value)
                if weekly_latest_date:
                    # 将日期向后调整一天，作为同步的起始日期
                    weekly_start_date = (weekly_latest_date + datetime.timedelta(days=1)).strftime('%Y%m%d')
                else:
                    weekly_start_date = '20050101'
                
                # 月线数据
                monthly_latest_date = self.get_stock_latest_date(stock_code, KlinePeriod.MONTHLY.value)
                if monthly_latest_date:
                    # 将日期向后调整一天，作为同步的起始日期
                    monthly_start_date = (monthly_latest_date + datetime.timedelta(days=1)).strftime('%Y%m%d')
                else:
                    monthly_start_date = '20000101'
            
                # 获取并处理15分钟级别数据
                thread_safe_log('info', f"从 {min15_start_date} 开始同步股票 {stock_code} 的15分钟数据")
                # 直接获取15分钟数据，某些数据源可能不支持15分钟数据
                try:
                    adapter = self.data_source_manager.get_adapter()
                    stock_data_15min = adapter.get_stock_15min_data(stock_code, min15_start_date)
                    if not stock_data_15min.empty:
                        processed_data = self.prepare_stock_data_for_clickhouse(stock_data_15min, stock_code, stock_name)
                        if not processed_data.empty:
                            self.save_stock_data_to_clickhouse(processed_data, KlinePeriod.MIN_15.value)
                except Exception as e:
                    thread_safe_log('warning', f"获取15分钟数据失败: {e}，继续处理其他级别数据")
            
                # 获取并处理日线数据
                thread_safe_log('info', f"从 {daily_start_date} 开始同步股票 {stock_code} 的日线数据")
                stock_data_daily = self.get_stock_daily_data(stock_code, daily_start_date)
                if not stock_data_daily.empty:
                    processed_data = self.prepare_stock_data_for_clickhouse(stock_data_daily, stock_code, stock_name)
                    if not processed_data.empty:
                        self.save_stock_data_to_clickhouse(processed_data, KlinePeriod.DAILY.value)
            
                # 获取并处理周线数据
                thread_safe_log('info', f"从 {weekly_start_date} 开始同步股票 {stock_code} 的周线数据")
                stock_data_weekly = self.get_stock_weekly_data(stock_code, weekly_start_date)
                if not stock_data_weekly.empty:
                    processed_data = self.prepare_stock_data_for_clickhouse(stock_data_weekly, stock_code, stock_name)
                    if not processed_data.empty:
                        self.save_stock_data_to_clickhouse(processed_data, KlinePeriod.WEEKLY.value)
            
                # 获取并处理月线数据
                thread_safe_log('info', f"从 {monthly_start_date} 开始同步股票 {stock_code} 的月线数据")
                stock_data_monthly = self.get_stock_monthly_data(stock_code, monthly_start_date)
                if not stock_data_monthly.empty:
                    processed_data = self.prepare_stock_data_for_clickhouse(stock_data_monthly, stock_code, stock_name)
                    if not processed_data.empty:
                        self.save_stock_data_to_clickhouse(processed_data, KlinePeriod.MONTHLY.value)
            
                # 添加短暂延时防止API限流，不同线程使用不同的延时，减少同时请求
                delay = 1.0 + (hash(threading.current_thread().name) % 10) / 10.0
                time.sleep(delay)
            
                thread_safe_log('info', f"股票 {stock_code} - {stock_name} 数据同步成功")
                return True
                
            except Exception as e:
                retry_count += 1
                thread_safe_log('error', f"处理股票 {stock_code} 时发生错误 (尝试 {retry_count}/{max_retries}): {e}")
                if retry_count < max_retries:
                    wait_time = retry_count * 5  # 逐渐增加等待时间：5秒、10秒...
                    thread_safe_log('info', f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    thread_safe_log('error', f"达到最大重试次数，放弃处理股票 {stock_code}")
                    return False
        
        return False

    def sync_all_stocks(self, csv_file='stock_code_name.csv', start_date=None):
        """
        同步所有股票数据，使用多线程方式加速
        :param csv_file: 股票代码CSV文件
        :param start_date: 开始日期 (YYYYMMDD)，若未指定则自动计算
        :return: 成功处理的股票数量
        """
        # 读取股票代码列表
        stock_df = self.read_stock_codes(csv_file)
        if stock_df.empty:
            logger.error("无法获取股票代码列表，同步终止")
            return 0
        
        total_stocks = len(stock_df)
        
        # 获取当前日期作为最大结束日期
        today = datetime.datetime.now().strftime('%Y%m%d')
        
        logger.info(f"开始多线程同步 {total_stocks} 只股票的多周期数据，到 {today}")
        logger.info(f"最大线程数: {self.max_workers}, 每批处理股票数: {self.batch_size}")
        
        # 创建结果统计变量
        success_count = 0
        error_count = 0
        skipped_count = 0
        
        # 将股票列表分批处理，避免一次创建过多线程
        for batch_start in range(0, total_stocks, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_stocks)
            logger.info(f"处理第 {batch_start+1}-{batch_end} 只股票，共 {total_stocks} 只")
            
            # 记录当前批次处理的股票数和跳过的股票数
            batch_processed = 0
            batch_skipped = 0
            
            # 创建线程池
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 创建任务列表
                future_to_stock = {}
                
                # 提交任务到线程池
                for idx in range(batch_start, batch_end):
                    row = stock_df.iloc[idx]
                    stock_code = row['code']
                    stock_name = row['name']
                    
                    # 检查是否需要跳过当前股票的同步（非强制同步模式下）
                    if not self.force_sync:
                        # 创建独立连接查询最新日期
                        daily_latest_date = self.get_stock_latest_date(stock_code, KlinePeriod.DAILY.value)
                        if daily_latest_date and daily_latest_date >= self.latest_trade_date:
                            logger.info(f"股票 {stock_code} - {stock_name} 已同步到最新交易日 {self.latest_trade_date}，跳过同步")
                            # 标记当前数据源为数据最新
                            self.data_source_manager.record_up_to_date()
                            skipped_count += 1
                            batch_skipped += 1
                            continue
                    
                    # 添加随机延迟，避免所有线程同时启动
                    time.sleep(0.2 + random.random() * 0.3)  # 200-500ms的随机延迟
                    
                    # 提交任务并保存Future对象
                    future = executor.submit(self.sync_single_stock, stock_code, stock_name, idx, total_stocks)
                    future_to_stock[future] = (stock_code, stock_name)
                    batch_processed += 1
                
                # 处理完成的任务结果
                for future in concurrent.futures.as_completed(future_to_stock):
                    stock_code, stock_name = future_to_stock[future]
                    try:
                        success = future.result()
                        if success:
                            success_count += 1
                        else:
                            error_count += 1
                    except Exception as exc:
                        logger.error(f"股票 {stock_code} - {stock_name} 处理异常: {exc}")
                        error_count += 1
            
            # 批次间休息时间，给数据库和网络一些恢复时间
            # 只有当批次中有股票实际被处理时才需要休息
            if batch_processed > 0:
                sleep_time = 1  # 从10秒改为1秒
                logger.info(f"批次处理完成，有 {batch_processed} 只股票被处理，休息{sleep_time}秒")
                time.sleep(sleep_time)
            else:
                logger.info(f"批次中所有股票 ({batch_skipped} 只) 都已是最新，无需休息")
        
        logger.info(f"多线程同步完成，成功: {success_count}/{total_stocks}, 跳过: {skipped_count}/{total_stocks}, 失败: {error_count}/{total_stocks}")
        return success_count

    def init_clickhouse(self):
        """
        初始化ClickHouse数据库和表
        """
        # 创建新的连接执行初始化
        db = None
        try:
            # 使用ClickHouseDB的init_database方法初始化数据库和表
            database_name = self.clickhouse_config['database']
            db = get_clickhouse_db(config=self.clickhouse_config)
            if not db.init_database(database_name):
                logger.error(f"初始化数据库失败: {database_name}")
                raise Exception(f"初始化数据库失败: {database_name}")
            else:
                logger.info(f"数据库初始化成功: {database_name}")
        except Exception as e:
            logger.error(f"初始化数据库出错: {e}")
            raise
        finally:
            # 确保关闭连接
            if db and hasattr(db, 'client') and db.client:
                try:
                    db.client.disconnect()
                except Exception as e:
                    logger.error(f"关闭数据库连接失败: {e}")

    def read_stock_codes(self, csv_file='stock_code_name.csv'):
        """
        从CSV文件读取股票代码列表
        :param csv_file: CSV文件路径
        :return: 包含股票代码和名称的DataFrame
        """
        try:
            logger.info(f"正在读取股票代码文件: {csv_file}")
            df = pd.read_csv(csv_file)
            
            # 清理股票代码数据
            df['code'] = df['code'].astype(str)
            
            # 过滤掉代码长度不是6位的股票
            original_count = len(df)
            df = df[df['code'].str.len() <= 6]
            if len(df) < original_count:
                logger.warning(f"过滤掉 {original_count - len(df)} 条不符合规范的股票代码记录")
            
            logger.info(f"成功读取股票代码，共 {len(df)} 条记录")
            return df
        except Exception as e:
            logger.error(f"读取股票代码文件失败: {e}")
            return pd.DataFrame()
    
    def get_adjusted_start_date(self, days_back=70):
        """
        计算合适的开始日期，确保至少获取指定交易日数量的数据
        :param days_back: 需要的交易日/周/月数量
        :return: 调整后的开始日期 (YYYYMMDD)
        """
        # 考虑到非交易日和可能的数据缺失，将实际天数扩大1.5倍
        calendar_days = days_back * 1.5
        # 从当前日期往前推指定的自然日
        start_date = (datetime.datetime.now() - datetime.timedelta(days=calendar_days)).strftime('%Y%m%d')
        logger.info(f"为确保至少获取{days_back}个交易周期的数据，设置起始日期为: {start_date}")
        return start_date
    
    def execute_query_safely(self, query, params=None):
        """
        安全执行数据库查询，确保连接正确关闭
        :param query: SQL查询
        :param params: 查询参数
        :return: 查询结果或None（如果查询失败）
        """
        db = None
        # 使用信号量限制并发连接数
        with self.connection_semaphore:
            try:
                # 每次查询都创建新的连接
                db = get_clickhouse_db(config=self.clickhouse_config)
                result = db.client.execute(query, params or {})
                return result
            except Exception as e:
                thread_safe_log('error', f"执行查询失败: {e}")
                return None
            finally:
                # 确保在finally块中关闭连接
                if db and hasattr(db, 'client') and db.client:
                    try:
                        # 显式关闭连接
                        db.client.disconnect()
                    except Exception as e:
                        thread_safe_log('error', f"关闭数据库连接失败: {e}")

    def get_stock_latest_date(self, stock_code, level):
        """
        获取股票在ClickHouse中的最新日期
        :param stock_code: 股票代码
        :param level: K线级别
        :return: 最新日期 (datetime.date类型) 或 None (如果没有数据)
        """
        try:
            # 确保股票代码是6位，不足前面补0
            stock_code = str(stock_code).zfill(6)
            
            # 查询数据库中该股票指定级别的最新记录
            thread_safe_log('info', f"查询股票 {stock_code} {level} 级别的最新记录")
            
            # 创建独立连接进行查询
            with self.connection_semaphore:
                db = None
                try:
                    db = get_clickhouse_db(config=self.clickhouse_config)
                    
                    # 查询最大日期
                    query = f"""
                    SELECT max(date) FROM stock_info 
                    WHERE code = '{stock_code}' AND level = '{level}'
                    """
                    
                    result = db.client.execute(query)
                
                    if result and result[0][0]:
                        latest_date = result[0][0]
                        thread_safe_log('info', f"股票 {stock_code} {level} 级别的最新日期为: {latest_date}")
                        return latest_date  # 直接返回datetime.date类型
                    else:
                        thread_safe_log('warning', f"股票 {stock_code} {level} 级别在数据库中没有记录")
                        return None
                finally:
                    # 确保关闭连接
                    if db and hasattr(db, 'client') and db.client:
                        try:
                            db.client.disconnect()
                        except Exception as e:
                            thread_safe_log('error', f"关闭数据库连接失败: {e}")
        except Exception as e:
            thread_safe_log('error', f"查询股票 {stock_code} {level} 级别最新日期失败: {e}")
            return None

    def prepare_stock_data_for_clickhouse(self, stock_data, stock_code, stock_name):
        """
        准备股票数据以便保存到ClickHouse
        :param stock_data: 原始股票数据
        :param stock_code: 股票代码
        :param stock_name: 股票名称
        :return: 处理后的股票数据
        """
        if stock_data.empty:
            return pd.DataFrame()
        
        try:
            # 重命名列以适应ClickHouse表结构
            stock_data_processed = stock_data.copy()
            
            # 转换日期和时间为标准格式
            # efinance的日期列是'日期'，转换为datetime类型
            if not '日期' in stock_data_processed.columns:
                logger.error(f"股票数据缺少日期列，列名: {stock_data_processed.columns.tolist()}")
                return pd.DataFrame()
            
            # 检查原始数据是否有时间信息（包含在日期字段中）
            has_time_info = False
            sample_date = stock_data_processed['日期'].iloc[0]
            if isinstance(sample_date, str) and len(sample_date) > 10:  # 如果日期字符串长度大于10，可能包含时间信息
                has_time_info = True
            
            # 存储原始数据的完整日期时间字符串（如果有时间信息）
            if has_time_info:
                stock_data_processed['datetime'] = pd.to_datetime(stock_data_processed['日期'])
            else:
                # 如果没有时间信息，但是是分钟级别数据，需要构造时间列
                # 首先检查是否有时间列
                if '时间' in stock_data_processed.columns:
                    # 如果有时间列，将日期和时间合并
                    stock_data_processed['datetime'] = pd.to_datetime(
                        stock_data_processed['日期'].astype(str) + ' ' + 
                        stock_data_processed['时间'].astype(str)
                    )
                else:
                    # 如果没有时间列，且是分钟级别数据，生成自增的序号作为区分
                    stock_data_processed['datetime'] = pd.to_datetime(stock_data_processed['日期'])
                    stock_data_processed = stock_data_processed.sort_values('datetime')  # 确保按日期排序
                    # 添加序号列，用于同一天内数据的排序
                    stock_data_processed['seq'] = range(len(stock_data_processed))
            
            # 将日期列转换为date类型
            stock_data_processed['日期'] = pd.to_datetime(stock_data_processed['日期']).dt.date
            
            # 添加必要的列 - 使用clickhouse_db.py中save_stock_info方法能识别的列名
            stock_data_processed['代码'] = stock_code
            stock_data_processed['名称'] = stock_name
            
            # 转换列名为标准格式
            column_mapping = {
                '开盘': '开盘',
                '收盘': '收盘',
                '最高': '最高',
                '最低': '最低',
                '成交量': '成交量',
                '换手率': '换手率',
                '涨跌幅': '涨跌幅',
            }
            
            # 确保所有必要的列都存在
            for standard_name in ['开盘', '收盘', '最高', '最低', '成交量']:
                if not standard_name in stock_data_processed.columns:
                    logger.error(f"股票数据缺少列: {standard_name}")
                    return pd.DataFrame()
            
            # 计算或设置其他必要的列
            if '换手率' not in stock_data_processed.columns:
                stock_data_processed['换手率'] = 0.0
            
            if '涨跌幅' not in stock_data_processed.columns:
                stock_data_processed['涨跌幅'] = 0.0
            
            # 计算振幅 = (最高价 - 最低价) / 开盘价 * 100
            stock_data_processed['振幅'] = (stock_data_processed['最高'] - stock_data_processed['最低']) / stock_data_processed['开盘'] * 100
            
            return stock_data_processed
        except Exception as e:
            logger.error(f"处理股票 {stock_code} 数据失败: {e}")
            return pd.DataFrame()

    def get_stock_daily_data(self, stock_code, start_date=None):
        """
        获取股票日线级别数据，使用数据源适配器
        :param stock_code: 股票代码
        :param start_date: 开始日期 (YYYYMMDD)
        :return: 股票数据DataFrame
        """
        if start_date is None:
            start_date = self.get_adjusted_start_date(70)
        
        # 最大重试次数
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # 使用适配器获取数据
                adapter = self.data_source_manager.get_adapter()
                current_source = self.data_source_manager.current_source
                logger.info(f"使用数据源 {current_source} 获取股票 {stock_code} 的日线数据")
                stock_data = adapter.get_stock_daily_data(stock_code, start_date)
                
                if not stock_data.empty:
                    # 成功获取数据，重置失败计数
                    self.data_source_manager.reset_failure()
                    return stock_data
                else:
                    # 对于空结果，需要判断是否是因为数据已是最新
                    logger.warning(f"数据源 {current_source} 返回的股票 {stock_code} 日线数据为空")
                    
                    # 检查是否因为已经是最新
                    current_date = datetime.datetime.now().date()
                    start_date_obj = datetime.datetime.strptime(start_date, '%Y%m%d').date() if len(start_date) == 8 else None
                    
                    is_empty_normal = False
                    if start_date_obj and start_date_obj > current_date:
                        logger.info(f"查询日期 {start_date} 晚于当前日期 {current_date}，空结果是正常的")
                        is_empty_normal = True
                        
                    # 检查数据源是否已标记为数据最新
                    if self.data_source_manager.is_up_to_date[current_source]:
                        logger.info(f"数据源 {current_source} 已标记为数据最新，空结果是正常的")
                        is_empty_normal = True
                    
                    # 如果是正常的空结果，不计为失败
                    if is_empty_normal:
                        logger.info(f"空结果是正常的，不计为失败")
                        return pd.DataFrame()
                    
                    # 否则记录失败并尝试切换数据源
                    switched = self.data_source_manager.record_failure(is_empty_result=True)
                    
                    if switched:
                        logger.info(f"由于空结果，已切换数据源到: {self.data_source_manager.current_source}")
                        # 切换数据源后继续尝试
                        continue
                    else:
                        # 没有更多数据源可用，返回空DataFrame
                        return pd.DataFrame()
                    
            except Exception as e:
                retry_count += 1
                logger.error(f"获取股票 {stock_code} 的日线数据失败 (尝试 {retry_count}/{max_retries}): {e}")
                
                # 记录数据源失败
                switched = self.data_source_manager.record_failure(is_empty_result=False)
                
                if switched:
                    logger.info(f"由于错误，已切换数据源到: {self.data_source_manager.current_source}")
                    # 如果切换了数据源，重置重试计数
                    retry_count = 0
                elif retry_count < max_retries:
                    wait_time = retry_count * 5  # 递增等待时间
                    logger.info(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"达到最大重试次数，放弃获取股票 {stock_code} 的日线数据")
                    return pd.DataFrame()
    
    def get_stock_weekly_data(self, stock_code, start_date=None):
        """
        获取股票周线级别数据，使用数据源适配器
        :param stock_code: 股票代码
        :param start_date: 开始日期 (YYYYMMDD)
        :return: 股票数据DataFrame
        """
        if start_date is None:
            # 周线数据设置更久远的起始日期，确保获取足够数据
            start_date = '20050101'
        
        # 最大重试次数
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # 使用适配器获取数据
                adapter = self.data_source_manager.get_adapter()
                stock_data = adapter.get_stock_weekly_data(stock_code, start_date)
                
                if not stock_data.empty:
                    # 成功获取数据，重置失败计数
                    self.data_source_manager.reset_failure()
                    return stock_data
                else:
                    # 返回空DataFrame而不是重试，因为可能确实没有数据
                    return pd.DataFrame()
                    
            except Exception as e:
                retry_count += 1
                logger.error(f"获取股票 {stock_code} 的周线数据失败 (尝试 {retry_count}/{max_retries}): {e}")
                
                # 记录数据源失败
                switched = self.data_source_manager.record_failure()
                
                if switched:
                    logger.info(f"已切换数据源到: {self.data_source_manager.current_source}")
                    # 如果切换了数据源，重置重试计数
                    retry_count = 0
                elif retry_count < max_retries:
                    wait_time = retry_count * 5  # 递增等待时间
                    logger.info(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"达到最大重试次数，放弃获取股票 {stock_code} 的周线数据")
                    return pd.DataFrame()
    
    def get_stock_monthly_data(self, stock_code, start_date=None):
        """
        获取股票月线级别数据，使用数据源适配器
        :param stock_code: 股票代码
        :param start_date: 开始日期 (YYYYMMDD)
        :return: 股票数据DataFrame
        """
        if start_date is None:
            # 月线数据需要更长的历史，从创业板成立前获取数据（约2000年）
            start_date = '20000101'
        
        # 最大重试次数
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # 使用适配器获取数据
                adapter = self.data_source_manager.get_adapter()
                stock_data = adapter.get_stock_monthly_data(stock_code, start_date)
                
                if not stock_data.empty:
                    # 成功获取数据，重置失败计数
                    self.data_source_manager.reset_failure()
                    return stock_data
                else:
                    # 返回空DataFrame而不是重试，因为可能确实没有数据
                    return pd.DataFrame()
                    
            except Exception as e:
                retry_count += 1
                logger.error(f"获取股票 {stock_code} 的月线数据失败 (尝试 {retry_count}/{max_retries}): {e}")
                
                # 记录数据源失败
                switched = self.data_source_manager.record_failure()
                
                if switched:
                    logger.info(f"已切换数据源到: {self.data_source_manager.current_source}")
                    # 如果切换了数据源，重置重试计数
                    retry_count = 0
                elif retry_count < max_retries:
                    wait_time = retry_count * 5  # 递增等待时间
                    logger.info(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"达到最大重试次数，放弃获取股票 {stock_code} 的月线数据")
                    return pd.DataFrame()

    def get_latest_trade_date(self):
        """
        获取最新交易日期
        :return: 最新交易日期 (datetime.date类型)
        """
        try:
            # 由于获取实时交易日期有困难，我们采用更实用的方法
            logger.info("计算最新交易日期...")
            
            # 获取当前日期
            today = datetime.datetime.now().date()
            
            # 周末不是交易日
            if today.weekday() >= 5:  # 5是星期六，6是星期日
                # 如果是周末，返回最近的周五
                days_to_subtract = today.weekday() - 4  # 5减去1得到4，6减去2得到4
                latest_trade_date = today - datetime.timedelta(days=days_to_subtract)
            else:
                # 如果是工作日，返回当天日期
                # 注意：这里没有考虑法定节假日，如果需要更精确，需要引入节假日数据
                latest_trade_date = today
            
            logger.info(f"计算得到最新交易日期: {latest_trade_date}")
            return latest_trade_date
        except Exception as e:
            # 如果发生异常，使用昨天的日期作为备选
            yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).date()
            logger.error(f"计算最新交易日期失败: {e}，使用昨天日期作为备选: {yesterday}")
            return yesterday

if __name__ == "__main__":
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='使用多线程方式同步股票数据到ClickHouse')
    parser.add_argument('--csv', type=str, default='data/reference/stock_code_name.csv', 
                        help='股票代码CSV文件路径')
    parser.add_argument('--threads', type=int, default=10, 
                        help='最大工作线程数')
    parser.add_argument('--batch', type=int, default=20, 
                        help='每批处理的股票数量')
    parser.add_argument('--force', action='store_true',
                        help='强制同步所有股票，忽略最新日期检查')
    parser.add_argument('--code', type=str, default=None,
                        help='只同步指定股票代码，多个代码用逗号分隔，如：000001,600000')
    parser.add_argument('--source', type=str, default='efinance', choices=['efinance', 'akshare', 'baostock'],
                        help='数据源类型，可选值：efinance, akshare, baostock')
    parser.add_argument('--debug', action='store_true',
                        help='开启调试模式，显示更详细的日志')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 设置日志级别
    if args.debug:
        logger.setLevel(logging.DEBUG)
        console_handler.setLevel(logging.DEBUG)
        file_handler.setLevel(logging.DEBUG)
        print("调试模式已开启，将显示更详细的日志信息")
    
    print(f"最大线程数: {args.threads}")
    print(f"每批处理股票数: {args.batch}")
    print(f"股票代码文件: {args.csv}")
    print(f"强制同步: {args.force}")
    print(f"数据源类型: {args.source}")
    
    if args.code:
        print(f"仅同步指定股票: {args.code}")
        # 将指定的股票代码转换为列表
        stock_codes = args.code.split(',')
        # 创建一个临时CSV文件
        import pandas as pd
        temp_df = pd.DataFrame({'code': stock_codes, 'name': [f'股票{code}' for code in stock_codes]})
        temp_csv = 'temp_stock_codes.csv'
        temp_df.to_csv(temp_csv, index=False)
        csv_path = temp_csv
    else:
        csv_path = args.csv
    
    # 初始化同步器，传入自定义参数
    synchronizer = AKShareToClickHouse(max_workers=args.threads, batch_size=args.batch, 
                                      force_sync=args.force, data_source=args.source)
    
    # 获取并显示最新交易日期
    print(f"当前最新交易日期: {synchronizer.latest_trade_date}")
    print(f"只同步尚未同步到最新日期的股票" if not args.force else "强制同步所有股票")
    print(f"初始数据源: {args.source}，自动切换顺序: {' > '.join(DataSourceManager.SOURCES)}")
    
    # 开始同步，自动从每个股票的最新日期开始同步
    print(f"开始多线程同步股票数据到最新交易日，详细日志请查看 logs/akshare_sync.log")
    
    # 记录开始时间
    start_time = time.time()
    
    success_count = synchronizer.sync_all_stocks(csv_path)
    
    # 记录结束时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"同步完成，总耗时: {elapsed_time:.2f}秒")
    print(f"成功处理的股票数量: {success_count}")
    
    # 如果创建了临时文件，清理它
    if args.code:
        import os
        if os.path.exists(temp_csv):
            os.remove(temp_csv) 